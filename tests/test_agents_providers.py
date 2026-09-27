"""Desk data providers: static snapshots and point-in-time historical replay (+ anonymisation)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aurum.agents.providers import DeskDataProvider, HistoricalDeskDataProvider, StaticDeskDataProvider
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro

KINDS = ("market_snapshot", "quant_signals", "risk_status", "macro_snapshot", "calendar", "backtest_stats",
         "positions")


# ----------------------------------------------------------------------------- static
def test_static_provider_basics():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    snap = {"x": np.float64(1.5), "bad": float("nan"), "when": pd.Timestamp("2026-01-05", tz="UTC")}
    p = StaticDeskDataProvider({"market": snap, "risk": lambda t: {"asked_at": t}})
    assert isinstance(p, DeskDataProvider)
    m = p.market_snapshot(now)
    assert m == {"x": 1.5, "bad": None, "when": "2026-01-05T00:00:00+00:00"}
    m["x"] = 99
    assert p.market_snapshot(now)["x"] == 1.5  # defensive copy
    assert p.risk_status(now) == {"asked_at": now.isoformat()}
    assert p.positions(now)["available"] is False
    assert p.as_of(now) == now.isoformat()
    json.dumps(p.market_snapshot(now), allow_nan=False)
    with pytest.raises(KeyError):
        StaticDeskDataProvider({"news": {}})
    with pytest.raises(ValueError):
        p.as_of(pd.Timestamp("2026-01-05 10:00"))


# ----------------------------------------------------------------------------- historical
@pytest.fixture(scope="module")
def market() -> tuple[MarketData, pd.DataFrame, pd.Series]:
    bars = make_synthetic_bars(600, "H1", seed=3, model="trend")
    macro = make_synthetic_macro(bars, seed=3)
    events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=10))
    events["actual"] = np.arange(len(events), dtype=float)
    close = bars["close"]
    signals = pd.DataFrame({
        "mom": np.tanh(close.pct_change(24).fillna(0.0) * 50),
        "rev": -np.tanh(close.pct_change(4).fillna(0.0) * 100),
    }, index=bars.index)
    combined = signals.mean(axis=1)
    return MarketData(bars=bars, macro=macro, events=events), signals, combined


def _provider(md, signals, combined, **kw) -> HistoricalDeskDataProvider:
    return HistoricalDeskDataProvider(
        md, signals=signals, combined=combined, backtest_stats={"oos_sharpe": 0.5},
        risk_status_fn=lambda now: {"drawdown": -0.01}, positions_fn=lambda now: {"lots": 0.2}, **kw,
    )


def _all(p: HistoricalDeskDataProvider, now: pd.Timestamp) -> dict:
    return {k: getattr(p, k)(now) for k in KINDS}


def test_bar_index_at_uses_available_at(market):
    md, signals, combined = market
    p = _provider(md, signals, combined)
    t = 400
    now = md.bars["available_at"].iloc[t]
    assert p.bar_index_at(now) == t
    assert p.bar_index_at(now - pd.Timedelta(seconds=1)) == t - 1  # bar t not complete yet
    with pytest.raises(LookupError):
        p.bar_index_at(md.bars.index[0])


def test_historical_snapshots_are_point_in_time(market):
    """Perturbing everything after `now` must not change any snapshot at `now`."""
    md, signals, combined = market
    t = 420
    now = md.bars["available_at"].iloc[t]
    base = _all(_provider(md, signals, combined), now)

    rng = np.random.default_rng(0)
    bars = md.bars.copy()
    fut = bars.index > bars.index[t]
    shock = np.exp(rng.normal(0, 0.05, fut.sum()))
    for c in ("open", "high", "low", "close"):
        bars.loc[fut, c] = bars.loc[fut, c].to_numpy() * shock
    bars.loc[fut, "spread"] = 5.0
    macro = {}
    for name, f in md.macro.items():
        g = f.copy()
        late = pd.DatetimeIndex(g["available_at"]) > now
        g.loc[late, "value"] = g.loc[late, "value"] * 3 + 7
        macro[name] = g
    ev = md.events.copy()
    ev.loc[ev["time"] > now, "actual"] = -999.0  # future outcomes must stay hidden
    sig2 = signals.copy()
    sig2.loc[fut] = 0.99
    comb2 = combined.copy()
    comb2.loc[fut] = -0.99
    perturbed = _all(_provider(MarketData(bars=bars, macro=macro, events=ev), sig2, comb2), now)
    assert json.dumps(base, sort_keys=True) == json.dumps(perturbed, sort_keys=True)


def test_market_snapshot_content(market):
    md, signals, combined = market
    t = 300
    now = md.bars["available_at"].iloc[t]
    m = _provider(md, signals, combined).market_snapshot(now)
    assert m["last_bar"]["close"] == pytest.approx(md.bars["close"].iloc[t], rel=1e-5)
    assert m["bar_open_time"] == md.bars.index[t].isoformat()
    assert m["bars_in_window"] == 250
    assert len(m["recent_closes"]) == 12
    r1 = 100 * np.log(md.bars["close"].iloc[t] / md.bars["close"].iloc[t - 1])
    assert m["returns_pct"]["1_bars"] == pytest.approx(r1, rel=1e-4)
    assert 0.01 < m["realised_vol_annualised"] < 1.0
    assert m["distance_to_sma_in_atr"]["200"] is not None
    json.dumps(m, allow_nan=False)


def test_quant_signals_snapshot(market):
    md, signals, combined = market
    t = 350
    now = md.bars["available_at"].iloc[t]
    q = _provider(md, signals, combined).quant_signals(now)
    assert q["combined_forecast"] == pytest.approx(combined.iloc[t], rel=1e-5, abs=1e-9)
    assert set(q["strategies"]) == {"mom", "rev"}
    assert len(q["combined_history_last_10_bars"]) == 10
    none = HistoricalDeskDataProvider(md).quant_signals(now)
    assert none["available"] is False


def test_macro_and_calendar_point_in_time(market):
    md, signals, combined = market
    p = _provider(md, signals, combined)
    t = 500
    now = md.bars["available_at"].iloc[t]
    macro = p.macro_snapshot(now)["series"]
    for name, frame in md.macro.items():
        visible = frame[pd.DatetimeIndex(frame["available_at"]) <= now]
        assert macro[name]["level"] == pytest.approx(visible["value"].iloc[-1], rel=1e-5)
        assert macro[name]["observation_date"] == visible.index[-1].isoformat()
    assert macro["us10y"]["change_units"] == "bp" and macro["dxy"]["change_units"] == "log % change"

    cal = p.calendar(now)
    assert all(e["hours_until"] > 0 for e in cal["upcoming"])
    assert all("actual" not in e for e in cal["upcoming"])
    assert all(e["hours_until"] <= 72 for e in cal["upcoming"])
    for e in cal["recently_released"]:
        assert e["hours_since"] >= 0 and "actual" in e


def test_calendar_shows_released_outcome():
    bars = make_synthetic_bars(300, "H1", seed=1)
    t = 200
    now = bars["available_at"].iloc[t]
    ev = pd.DataFrame({
        "time": [now - pd.Timedelta(hours=2), now + pd.Timedelta(hours=5)],
        "name": ["CPI", "NFP"], "currency": ["USD", "USD"], "importance": [3, 3],
        "actual": [3.1, 250.0],
    })
    cal = HistoricalDeskDataProvider(MarketData(bars=bars, events=ev)).calendar(now)
    assert cal["recently_released"][0]["name"] == "CPI" and cal["recently_released"][0]["actual"] == 3.1
    assert cal["upcoming"][0]["name"] == "NFP" and "actual" not in cal["upcoming"][0]
    assert cal["next_high_importance_hours"] == pytest.approx(5.0)


def test_anonymised_snapshots_hide_dates_and_levels(market):
    md, signals, combined = market
    t = 450
    now = md.bars["available_at"].iloc[t]
    p = _provider(md, signals, combined, anonymise=True)
    snaps = _all(p, now)
    blob = json.dumps(snaps)
    real_year = str(md.bars.index[t].year)
    assert real_year not in blob
    assert p.as_of(now).startswith("2101-") or p.as_of(now).startswith("2100-")
    shifted = pd.Timestamp(p.as_of(now))
    assert shifted.weekday() == now.weekday() and shifted.hour == now.hour
    m = snaps["market_snapshot"]
    assert m["anonymised"] is True
    assert m["last_bar"]["close"] == pytest.approx(100.0)
    assert all(90 < c < 110 for c in m["recent_closes"])
    real_close = md.bars["close"].iloc[t]
    assert f"{real_close:.1f}" not in blob
    for s in snaps["macro_snapshot"]["series"].values():
        assert "level" not in s and "observation_date" not in s
    # scale-free quantities are unchanged by anonymisation
    plain = _provider(md, signals, combined).market_snapshot(now)
    assert m["returns_pct"] == plain["returns_pct"]
    assert m["spread"]["current_bps"] == pytest.approx(plain["spread"]["current_bps"])


def test_misaligned_inputs_rejected(market):
    md, signals, combined = market
    with pytest.raises(ValueError):
        HistoricalDeskDataProvider(md, signals=signals.iloc[:-1])


# ----------------------------------------------------------------------------- reviewer: adversarial
def test_anonymise_requires_whole_week_date_shift(market):
    """A shift that is not a whole number of weeks moves the presented weekday: agents would
    see a 'Wednesday' decision time for a real Friday bar and misjudge weekend-gap risk."""
    md, _, _ = market
    with pytest.raises(ValueError):
        HistoricalDeskDataProvider(md, anonymise=True, date_shift_days=10)
    HistoricalDeskDataProvider(md, anonymise=True, date_shift_days=7 * 4000)
    HistoricalDeskDataProvider(md, anonymise=False, date_shift_days=10)  # unused when not anonymising


def _macro_frame(dates: pd.DatetimeIndex, values) -> pd.DataFrame:
    return pd.DataFrame({"value": values, "available_at": dates + pd.Timedelta(hours=21, minutes=30)},
                        index=dates)


def test_macro_snapshot_uses_latest_valid_observation_in_availability_order():
    bars = make_synthetic_bars(400, "H1", seed=2)
    now = bars["available_at"].iloc[380]
    dates = pd.date_range(end=now.normalize() - pd.Timedelta(days=1), periods=40, freq="D", tz="UTC")
    vals = np.linspace(100.0, 139.0, 40)
    vals[-1] = np.nan  # latest release missing (holiday / not yet parsed)
    frame = _macro_frame(dates, vals)
    shuffled = frame.sample(frac=1.0, random_state=0)  # storage order must not matter
    for f in (frame, shuffled):
        p = HistoricalDeskDataProvider(MarketData(bars=bars, macro={"dxy": f}))
        s = p.macro_snapshot(now)["series"]["dxy"]
        assert s["level"] == pytest.approx(138.0)
        assert s["observation_date"] == dates[-2].isoformat()
        expected_age = (now - (dates[-2] + pd.Timedelta(hours=21, minutes=30))).total_seconds() / 3600.0
        assert s["hours_since_available"] == pytest.approx(expected_age)
        assert s["change_1obs"] == pytest.approx(100 * np.log(138.0 / 137.0), rel=1e-5)


def test_quant_signals_with_nan_combined_has_no_fake_agreement(market):
    md, signals, combined = market
    t = 300
    comb = combined.copy()
    comb.iloc[t] = np.nan
    q = HistoricalDeskDataProvider(md, signals=signals, combined=comb).quant_signals(md.bars["available_at"].iloc[t])
    assert q["combined_forecast"] is None
    assert q["share_agreeing_with_combined"] is None


def test_market_snapshot_reports_staleness_over_weekend():
    bars = make_synthetic_bars(200, "H1", seed=5)
    fri = bars.index[bars.index.dayofweek == 4][-1]
    t = bars.index.get_loc(fri)
    sat = bars["available_at"].iloc[t] + pd.Timedelta(hours=13)
    for anon in (False, True):
        m = HistoricalDeskDataProvider(MarketData(bars=bars), anonymise=anon).market_snapshot(sat)
        assert m["hours_since_bar_close"] == pytest.approx(13.0)
    at_close = HistoricalDeskDataProvider(MarketData(bars=bars)).market_snapshot(bars["available_at"].iloc[t])
    assert at_close["hours_since_bar_close"] == 0.0
    assert at_close["bar_close_time"] == bars["available_at"].iloc[t].isoformat()


def test_single_bar_and_empty_inputs_are_json_safe():
    bars = make_synthetic_bars(30, "H1", seed=5)
    for n in (1, 2, 21):
        b = bars.iloc[:n]
        sig = pd.DataFrame({"a": np.linspace(-1, 1, n)}, index=b.index)
        p = HistoricalDeskDataProvider(MarketData(bars=b, macro={}, events=pd.DataFrame(columns=["time", "name"])),
                                       signals=sig, combined=sig["a"], anonymise=True)
        snaps = _all(p, b["available_at"].iloc[-1])
        json.dumps(snaps, allow_nan=False)
        assert snaps["market_snapshot"]["atr_14"] is None if n < 14 else True
        assert snaps["macro_snapshot"]["available"] is False and snaps["calendar"]["available"] is False


# ----------------------------------------------------------------------------- reviewer 2: adversarial
def test_anonymised_calendar_withholds_outcome_levels():
    """'CPI actual 9.1' identifies June 2022 as surely as a date: with anonymise=True a model
    could recall what gold did next. Only the surprise may be shown."""
    bars = make_synthetic_bars(300, "H1", seed=1)
    now = bars["available_at"].iloc[200]
    ev = pd.DataFrame({
        "time": [now - pd.Timedelta(hours=2), now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=5)],
        "name": ["CPI", "NFP", "FOMC"], "currency": ["USD"] * 3, "importance": [3, 3, 3],
        "actual": pd.array([9.1, None, 5.5], dtype="Float64"),
        "forecast": pd.array([8.8, 200.0, 5.25], dtype="Float64"),
        "previous": [8.6, 390.0, 5.0],
    })
    md = MarketData(bars=bars, events=ev)
    anon = HistoricalDeskDataProvider(md, anonymise=True).calendar(now)
    blob = json.dumps(anon)
    for level in ("9.1", "8.8", "8.6", "390"):
        assert level not in blob
    cpi, nfp = anon["recently_released"]
    assert cpi["surprise"] == pytest.approx(0.3) and cpi["outcome_levels_withheld"] is True
    assert nfp["surprise"] is None  # actual missing (pd.NA): no fake number, no "<NA>" string
    assert "<NA>" not in blob and "actual" not in cpi and "forecast" not in cpi
    # an explicit surprise column wins; the plain provider still shows the full print
    ev2 = ev.assign(surprise=[0.25, None, None])
    assert HistoricalDeskDataProvider(MarketData(bars=bars, events=ev2), anonymise=True).calendar(now)[
        "recently_released"][0]["surprise"] == pytest.approx(0.25)
    plain = HistoricalDeskDataProvider(md).calendar(now)["recently_released"]
    assert plain[0]["actual"] == pytest.approx(9.1) and plain[1]["actual"] is None


def test_nullable_missing_values_do_not_blank_whole_snapshots(market):
    """One pd.NA at the decision bar (nullable Float64 inputs) made float() raise, so the
    agents lost the ENTIRE market / quant-signal snapshot instead of seeing one null."""
    md, signals, combined = market
    t = 300
    now = md.bars["available_at"].iloc[t]
    comb = combined.astype("Float64")
    comb.iloc[t] = pd.NA
    sig = signals.astype("Float64")
    sig.iloc[t, 0] = pd.NA
    vol = pd.Series(0.15, index=md.bars.index, dtype="Float64")
    vol.iloc[t] = pd.NA
    p = HistoricalDeskDataProvider(md, signals=sig, combined=comb, vol=vol)
    q = p.quant_signals(now)
    assert q["combined_forecast"] is None and q["strategies"]["mom"]["forecast"] is None
    assert q["strategies"]["rev"]["forecast"] == pytest.approx(float(signals["rev"].iloc[t]), rel=1e-5, abs=1e-9)
    m = p.market_snapshot(now)
    assert m["realised_vol_annualised"] is None and m["last_bar"]["close"] is not None
    json.dumps([q, m], allow_nan=False)


def test_naive_event_times_are_rejected_not_assumed_utc():
    """A naive New-York-local 08:30 read as UTC would reveal the print 4-5 h early."""
    bars = make_synthetic_bars(300, "H1", seed=1)
    ev = pd.DataFrame({"time": [pd.Timestamp("2020-01-16 08:30")], "name": ["CPI"], "importance": [3],
                       "actual": [2.3]})
    with pytest.raises(ValueError, match="tz-aware"):
        HistoricalDeskDataProvider(MarketData(bars=bars, events=ev)).calendar(bars["available_at"].iloc[200])


@pytest.mark.parametrize("anonymise", [False, True])
def test_snapshots_equal_those_built_from_truncated_history(market, anonymise):
    """Point-in-time by construction: the snapshot at `now` must be identical to one built from
    data that physically ends at the last bar completed by `now` (also mid-bar and over
    weekend gaps)."""
    md, signals, combined = market
    bars = md.bars
    fri = int(np.flatnonzero(bars.index.dayofweek == 4)[-3])  # a Friday bar well inside the sample
    cases = [(250, pd.Timedelta(0)), (251, pd.Timedelta(minutes=30)), (fri, pd.Timedelta(hours=30))]
    for t, extra in cases:
        now = bars["available_at"].iloc[t] + extra
        k = int(np.searchsorted(pd.DatetimeIndex(bars["available_at"]), now, side="right"))
        full = _all(_provider(md, signals, combined, anonymise=anonymise), now)
        cut = MarketData(bars=bars.iloc[:k], macro=md.macro, events=md.events)
        trunc = _all(_provider(cut, signals.iloc[:k], combined.iloc[:k], anonymise=anonymise), now)
        assert json.dumps(full, sort_keys=True) == json.dumps(trunc, sort_keys=True), (t, extra)


def test_anonymised_hook_outputs_do_not_leak_real_dates():
    """A position summary's `entry_time` (the natural output of a positions hook) revealed the
    real date, and a stats field named `as_of` overrode the anonymised decision time."""
    bars = make_synthetic_bars(300, "H1", seed=1)
    t = 200
    now = bars["available_at"].iloc[t]
    entry = bars.index[190]
    p = HistoricalDeskDataProvider(
        MarketData(bars=bars), anonymise=True,
        positions_fn=lambda _now: {"lots": 0.5, "entry_time": entry, "fills": [{"at": entry.to_pydatetime()}],
                                   "entry_np": entry.to_datetime64(), "entry_date": entry.date()},
        risk_status_fn=lambda _now: {"halted_since": None, "day_start": entry.normalize()},
        backtest_stats={"as_of": "2020-01-10", "oos_sharpe": 0.5},
    )
    snaps = {k: getattr(p, k)(now) for k in ("positions", "risk_status", "backtest_stats")}
    blob = json.dumps(snaps)
    assert str(entry.year) not in blob
    pos = snaps["positions"]
    assert pd.Timestamp(pos["entry_time"]) == entry + p.date_shift
    assert pd.Timestamp(pos["fills"][0]["at"]).tz_localize(None) == (entry + p.date_shift).tz_localize(None)
    assert pos["entry_date"] == (entry + p.date_shift).date().isoformat()
    assert all(s["as_of"] == p.as_of(now) for s in snaps.values())
    # without anonymisation hook outputs are untouched
    plain = HistoricalDeskDataProvider(MarketData(bars=bars), positions_fn=lambda _n: {"entry_time": entry})
    assert plain.positions(now)["entry_time"] == entry.isoformat()
