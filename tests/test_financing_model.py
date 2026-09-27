"""Rate-based overnight financing: hand-computed amounts, point-in-time rates, fallback,
the simulator/engine plumbing and an independent brute-force reference."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.engine import buy_and_hold_benchmark, run_target_lots
from aurum.core.instrument import XAUUSD
from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import (
    CostModel,
    FinancingModel,
    RateCurve,
    rollover_nights,
    rollover_nights_ns,
)
from aurum.execution.simulator import ExecutionSimulator

RATE = FinancingModel(mode="rate", markup_long=0.025, markup_short=0.025, fallback_rate=0.0)
NO_FRICTION = dict(spread_multiplier=0.0, min_spread=0.0, slippage_fixed=0.0, slippage_range_frac=0.0,
                   commission_per_lot=0.0)
NIGHT_5PCT = 100 * 2000 * (0.05 + 0.025) / 360        # 41.666... USD per lot-night long at 5%


def hand_bars(rows: list[tuple[float, float, float, float]], start: str, spread: float = 0.30) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="1h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["spread"] = spread
    return make_bars(df, "H1")


def macro_frame(obs: list[tuple[str, float, str]]) -> pd.DataFrame:
    """(observation date, value in percent, available_at) rows in the aurum.data.macro format."""
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d, _, _ in obs], name="date")
    return pd.DataFrame({"value": [v for _, v, _ in obs],
                         "available_at": pd.DatetimeIndex([pd.Timestamp(a, tz="UTC") for _, _, a in obs])},
                        index=idx)


FLAT = [(2000.0, 2001.0, 1999.0, 2000.0)] * 4


# ---- FinancingModel arithmetic ----------------------------------------------------------------------
def test_hand_computed_long_short_and_triple() -> None:
    fin = RATE
    # long 1 lot at $2000, fed funds 5%, markup 2.5%: 100*2000*0.075/360 per night
    assert fin.amount(1.0, 1, price=2000.0, rate_nights=0.05) == pytest.approx(-NIGHT_5PCT)
    assert NIGHT_5PCT == pytest.approx(41.6667, abs=1e-4)
    # Wednesday = 3 nights (rate-nights are weight * rate)
    assert fin.amount(1.0, 3, price=2000.0, rate_nights=3 * 0.05) == pytest.approx(-3 * NIGHT_5PCT)
    # short receives rate - markup ...
    assert fin.amount(-2.0, 1, price=2000.0, rate_nights=0.05) == pytest.approx(2 * 100 * 2000 * 0.025 / 360)
    # ... and pays when the rate is below the markup (near-zero-rate years)
    assert fin.amount(-1.0, 1, price=2000.0, rate_nights=0.01) == pytest.approx(-100 * 2000 * 0.015 / 360)
    # lease rate: earned by longs, paid by shorts
    lease = dataclasses.replace(fin, lease_rate=0.01)
    assert lease.amount(1.0, 1, price=2000.0, rate_nights=0.05) == pytest.approx(-100 * 2000 * 0.065 / 360)
    assert lease.amount(-1.0, 1, price=2000.0, rate_nights=0.05) == pytest.approx(100 * 2000 * 0.015 / 360)
    # zero position / zero nights
    assert fin.amount(0.0, 3, price=2000.0) == 0.0 and fin.amount(1.0, 0, price=2000.0) == 0.0
    # nightly() / annual_rate() helpers agree
    assert fin.nightly(1.0, 2000.0, 0.05) == pytest.approx(-NIGHT_5PCT)
    assert fin.annual_rate(1.0, 0.05) == pytest.approx(0.075)
    assert fin.annual_rate(-1.0, 0.05) == pytest.approx(0.025)
    with pytest.raises(ValueError, match="price"):
        fin.amount(1.0, 1)


def test_modes_and_cost_model_plumbing() -> None:
    assert CostModel().financing.mode == "rate"                       # new default
    assert CostModel.zero().financing.mode == "none"                  # fully frictionless
    assert CostModel.zero().swap(1.0, 3, price=2000.0) == 0.0
    fixed = CostModel(financing="fixed")
    assert fixed.swap(1.0, 3) == pytest.approx(3 * XAUUSD.swap_long_per_lot)
    assert fixed.swap(-1.0, 1) == pytest.approx(XAUUSD.swap_short_per_lot)
    # default rate_nights = nights * fallback_rate
    cm = CostModel(financing={"mode": "rate", "fallback_rate": 0.05})
    assert cm.swap(1.0, 1, price=2000.0) == pytest.approx(-NIGHT_5PCT)
    # to_dict round trip (YAML / live runner pass the nested dict)
    again = CostModel(**cm.to_dict())
    assert again == cm and isinstance(again.financing, FinancingModel)
    for bad in ({"mode": "libor"}, {"markup_long": -0.01}, {"day_count": 0}, {"rate_unit": "pct"},
                {"fallback_rate": float("nan")}):
        with pytest.raises(ValueError):
            FinancingModel(**bad)
    with pytest.raises(TypeError):
        CostModel(financing=3.0)


def test_swap_between_uses_rate_as_of_each_rollover() -> None:
    rates = macro_frame([("2024-01-01", 5.0, "2024-01-02 21:30"), ("2024-01-10", 10.0, "2024-01-11 21:30")])
    cm = CostModel(financing=RATE)
    # Tue 9th (1 night) + Wed 10th (3 nights) at 5%; Thu 11th 21:00 is before the 10% print (21:30)
    got = cm.swap_between(1.0, pd.Timestamp("2024-01-09 12:00Z"), pd.Timestamp("2024-01-11 22:00Z"),
                          price=2000.0, rates={"fedfunds": rates})
    assert got == pytest.approx(-5 * NIGHT_5PCT)
    # Friday 12th 21:00: the 10% print is public -> 100*2000*(0.10+0.025)/360
    fri = cm.swap_between(1.0, pd.Timestamp("2024-01-12 12:00Z"), pd.Timestamp("2024-01-12 22:00Z"),
                          price=2000.0, rates={"fedfunds": rates})
    assert fri == pytest.approx(-100 * 2000 * 0.125 / 360)


# ---- point-in-time rate curve --------------------------------------------------------------------------
def test_rate_curve_asof_semantics() -> None:
    frame = macro_frame([("2024-01-05", 5.0, "2024-01-08 21:30"),   # Fri
                         ("2024-01-06", 5.1, "2024-01-08 21:30"),   # Sat } same publication slot:
                         ("2024-01-07", 5.2, "2024-01-08 21:30"),   # Sun } the last row wins
                         ("2024-01-08", 5.3, "2024-01-09 21:30")])
    curve = RateCurve.from_source({"fedfunds": frame})
    assert len(curve) == 2 and curve.name == "fedfunds"
    t = pd.DatetimeIndex(["2024-01-08 21:29:59", "2024-01-08 21:30", "2024-01-09 21:00", "2024-01-09 21:30"],
                         tz="UTC").as_unit("ns").asi8
    np.testing.assert_allclose(curve.asof_ns(t), [np.nan, 0.052, 0.052, 0.053])
    # Series form: indexed by AVAILABILITY time; fraction units
    s = pd.Series([0.04], index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
    assert RateCurve.from_source(s, unit="fraction").asof_ns(t)[-1] == pytest.approx(0.04)
    # missing series / None -> no curve; frames without available_at are rejected
    assert RateCurve.from_source({"dxy": frame}) is None and RateCurve.from_source(None) is None
    with pytest.raises(ValueError, match="available_at"):
        RateCurve.from_source(frame.drop(columns="available_at"))
    with pytest.raises(ValueError, match="tz-aware"):
        RateCurve.from_source(pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01"])))


def test_rate_nights_match_brute_force_and_are_additive() -> None:
    rng = np.random.default_rng(3)
    days = pd.date_range("2023-12-01", "2024-03-31", freq="D", tz="UTC")
    vals = 5.0 + np.cumsum(rng.normal(0, 0.05, len(days)))
    frame = pd.DataFrame({"value": vals, "available_at": days + pd.Timedelta(hours=45, minutes=30)}, index=days)
    curve = RateCurve.from_source(frame)
    fin = FinancingModel(fallback_rate=0.011)
    base = pd.Timestamp("2023-11-25", tz="UTC")
    t0s, t1s, want = [], [], []
    for _ in range(400):
        a = base + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90)))
        b = a + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 9)))
        acc, d = 0.0, a.normalize() - pd.Timedelta(days=1)
        while d <= b:
            r = d + pd.Timedelta(hours=21)
            if a < r <= b and r.weekday() < 5:
                hit = frame.loc[frame["available_at"] <= r, "value"]
                rate = hit.iloc[-1] / 100 if len(hit) else 0.011
                acc += (3 if r.weekday() == 2 else 1) * rate
            d += pd.Timedelta(days=1)
        t0s.append(a.as_unit("ns").value)
        t1s.append(b.as_unit("ns").value)
        want.append(acc)
    got = fin.rate_nights_ns(np.array(t0s), np.array(t1s), XAUUSD, curve)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
    # additivity: splitting an interval never changes the total
    t0, t1 = np.array(t0s), np.array(t1s)
    mid = t0 + (t1 - t0) // 3
    np.testing.assert_allclose(fin.rate_nights_ns(t0, mid, XAUUSD, curve) + fin.rate_nights_ns(mid, t1, XAUUSD, curve),
                               got, rtol=1e-12, atol=1e-15)
    # without rates every night is charged at the fallback
    nights = rollover_nights_ns(t0, t1, XAUUSD)
    np.testing.assert_allclose(fin.rate_nights_ns(t0, t1, XAUUSD, None), 0.011 * nights, rtol=1e-12)


# ---- simulator ----------------------------------------------------------------------------------------
def _sim(bars: pd.DataFrame, rates, fin: FinancingModel = RATE) -> ExecutionSimulator:
    return ExecutionSimulator(bars, costs=CostModel(**NO_FRICTION, financing=fin), rates=rates)


def test_simulator_hand_computed_wednesday_triple_and_tuesday() -> None:
    rates = {"fedfunds": macro_frame([("2024-01-01", 5.0, "2024-01-02 21:30")])}
    wed = hand_bars(FLAT, "2024-01-10 19:00")          # Wed 19:00..22:00; bar 20:00 ends at the rollover
    r1 = _sim(wed, rates).step(1.0)
    assert r1.costs["swap"] == pytest.approx(-3 * NIGHT_5PCT)            # -125.00
    assert r1.costs["swap"] == pytest.approx(-125.0)
    tue = hand_bars(FLAT, "2024-01-09 19:00")
    assert _sim(tue, rates).step(1.0).costs["swap"] == pytest.approx(-NIGHT_5PCT)
    # short 2 lots on the Wednesday: receives 3 * 2 * 100*2000*(0.05-0.025)/360
    assert _sim(wed, rates).step(-2.0).costs["swap"] == pytest.approx(3 * 2 * 100 * 2000 * 0.025 / 360)
    # the notional is valued at the mid close of the rollover bar
    up = hand_bars([(2000, 2001, 1999, 2000), (2000, 2201, 1999, 2200), (2200, 2201, 2199, 2200)],
                   "2024-01-09 19:00")
    assert _sim(up, rates).step(1.0).costs["swap"] == pytest.approx(-100 * 2200 * 0.075 / 360)
    # opened AT the rollover (21:00 open) -> not charged
    sim = _sim(wed, rates)
    sim.step(0.0)
    assert sim.step(1.0).costs["swap"] == 0.0


def test_simulator_rate_published_after_rollover_is_not_used() -> None:
    wed = hand_bars(FLAT, "2024-01-10 19:00")
    rollover = pd.Timestamp("2024-01-10 21:00", tz="UTC")
    base = [("2024-01-01", 5.0, "2024-01-02 21:30")]
    late = macro_frame(base + [("2024-01-10", 20.0, str(rollover + pd.Timedelta(1, "ns")))])
    on_time = macro_frame(base + [("2024-01-10", 20.0, str(rollover))])
    assert _sim(wed, {"fedfunds": late}).step(1.0).costs["swap"] == pytest.approx(-3 * NIGHT_5PCT)
    # available exactly at the rollover instant -> usable (available_at <= R)
    assert _sim(wed, {"fedfunds": on_time}).step(1.0).costs["swap"] == pytest.approx(-3 * 100 * 2000 * 0.225 / 360)


def test_fallback_rate_without_or_before_the_series(caplog) -> None:
    wed = hand_bars(FLAT, "2024-01-10 19:00")
    fin = dataclasses.replace(RATE, fallback_rate=0.05)
    with caplog.at_level(logging.WARNING, logger="aurum.execution.costs"):
        sim = _sim(wed, None, fin)
    assert sim.step(1.0).costs["swap"] == pytest.approx(-3 * NIGHT_5PCT)
    assert sim.result(compute_metrics=False).meta["financing"]["rate_first_available"] is None
    # a series that only starts after the rollover -> fallback, not a back-filled value
    future = {"fedfunds": macro_frame([("2024-01-11", 1.0, "2024-01-12 21:30")])}
    assert _sim(wed, future, fin).step(1.0).costs["swap"] == pytest.approx(-3 * NIGHT_5PCT)
    # a rate source without the configured series name also falls back
    assert _sim(wed, {"dxy": macro_frame([("2024-01-01", 99.0, "2024-01-02")])}, fin).step(1.0).costs[
        "swap"] == pytest.approx(-3 * NIGHT_5PCT)


def test_gap_rollover_valued_at_previous_close() -> None:
    """A rollover inside a gap between bars is charged on the old position at close[t]."""
    rows = [(2000, 2001, 1999, 2000), (2000, 2011, 1999, 2010), (2030, 2031, 2029, 2030), (2030, 2031, 2029, 2030)]
    idx = pd.DatetimeIndex(["2024-01-09 18:00", "2024-01-09 19:00", "2024-01-09 22:00", "2024-01-09 23:00"],
                           tz="UTC")                   # Tuesday; 20:00-22:00 missing (rollover 21:00)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["spread"] = 0.3
    bars = make_bars(df, "H1")
    rates = {"fedfunds": macro_frame([("2024-01-01", 5.0, "2024-01-02 21:30")])}
    sim = _sim(bars, rates)
    sim.step(1.0)
    r = sim.step(1.0)
    assert r.costs["swap"] == pytest.approx(-100 * 2010 * 0.075 / 360)


def _brute_force_swap(bars: pd.DataFrame, lots: float, frame: pd.DataFrame, fin: FinancingModel) -> float:
    """Constant position from the open of bar 1 to the last close: one rollover at a time."""
    opens, avail = list(bars.index), list(pd.DatetimeIndex(bars["available_at"]))
    close = bars["close"].to_numpy()
    total, d = 0.0, opens[1].normalize() - pd.Timedelta(days=1)
    while d <= avail[-1]:
        r = d + pd.Timedelta(hours=21)
        d += pd.Timedelta(days=1)
        if not (opens[1] < r <= avail[-1]) or r.weekday() >= 5:
            continue
        k = next(i for i in range(1, len(bars)) if r <= avail[i])      # first bar ending at/after R
        px = close[k] if opens[k] < r else close[k - 1]                 # inside the bar / in the gap
        hit = frame.loc[frame["available_at"] <= r, "value"]
        rate = hit.iloc[-1] / 100 if len(hit) else fin.fallback_rate
        w = 3 if r.weekday() == 2 else 1
        m = fin.markup_long if lots > 0 else -fin.markup_short
        total += -lots * 100 * px * (rate - fin.lease_rate + m) * w / 360
    return total


@pytest.mark.parametrize("lots,gaps", [(1.3, True), (-2.0, False), (0.7, False)])
def test_simulator_matches_brute_force_rollover_by_rollover(lots: float, gaps: bool) -> None:
    bars = make_synthetic_bars(24 * 30, seed=5, start="2024-01-03", weekend_gaps=gaps)
    bars = bars.loc[bars.index.hour != 20] if gaps else bars          # daily gap across 21:00 too
    days = pd.date_range("2024-01-08", "2024-03-01", freq="B", tz="UTC")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"value": 4.0 + np.cumsum(rng.normal(0, 0.2, len(days))),
                          "available_at": days + pd.Timedelta(hours=45, minutes=30)}, index=days)
    fin = FinancingModel(markup_long=0.02, markup_short=0.03, lease_rate=0.004, fallback_rate=0.07)
    sim = ExecutionSimulator(bars, costs=CostModel(**NO_FRICTION, financing=fin), rates={"fedfunds": frame})
    while not sim.done:
        sim.step(lots)
    res = sim.result(compute_metrics=False)
    want = _brute_force_swap(bars, lots, frame, fin)
    assert res.costs["swap"].sum() == pytest.approx(want, rel=1e-12)
    assert abs(res.reconcile()["residual"]) < 1e-6
    assert res.trades["swap"].sum() == pytest.approx(want, rel=1e-12)
    assert res.meta["financing"]["mode"] == "rate" and res.meta["costs"]["financing"]["mode"] == "rate"


# ---- engine -----------------------------------------------------------------------------------------
def test_engine_reads_md_macro_point_in_time_and_accepts_overrides() -> None:
    bars = make_synthetic_bars(24 * 40, seed=9, start="2024-01-03")
    days = pd.date_range("2023-12-01", "2024-03-01", freq="B", tz="UTC")
    ff = pd.DataFrame({"value": np.linspace(5.5, 3.0, len(days)),
                       "available_at": days + pd.Timedelta(hours=45, minutes=30)}, index=days)
    md = MarketData(bars=bars, macro={"fedfunds": ff})
    path = pd.Series(1.0, index=bars.index)
    res = run_target_lots(md, path, costs=CostModel(**NO_FRICTION), compute_metrics=False)
    sim = ExecutionSimulator(bars, costs=CostModel(**NO_FRICTION), rates={"fedfunds": ff})
    while not sim.done:
        sim.step(1.0)
    np.testing.assert_array_equal(res.equity.to_numpy(), sim.result(compute_metrics=False).equity.to_numpy())
    # rows published after the last bar cannot change anything (point-in-time)
    later = pd.DataFrame({"value": [50.0], "available_at": [bars["available_at"].iloc[-1] + pd.Timedelta(1, "ns")]},
                         index=pd.DatetimeIndex([bars.index[-1].normalize()]))
    md2 = MarketData(bars=bars, macro={"fedfunds": pd.concat([ff, later])})
    res2 = run_target_lots(md2, path, costs=CostModel(**NO_FRICTION), compute_metrics=False)
    np.testing.assert_array_equal(res.equity.to_numpy(), res2.equity.to_numpy())
    # truncated history (a window ending earlier) reproduces the same path
    cut = bars.index[500]
    res3 = run_target_lots(md, path, costs=CostModel(**NO_FRICTION), end=cut, compute_metrics=False)
    np.testing.assert_array_equal(res3.equity.to_numpy(), res.equity.loc[:cut].to_numpy())
    # financing= override and rates= override
    fixed = run_target_lots(md, path, costs=CostModel(**NO_FRICTION), financing="fixed", compute_metrics=False)
    nights = rollover_nights(bars.index[1], bars["available_at"].iloc[-1])
    assert fixed.costs["swap"].sum() == pytest.approx(XAUUSD.swap_long_per_lot * nights)
    none = run_target_lots(md, path, costs=CostModel(**NO_FRICTION), financing="none", compute_metrics=False)
    assert none.costs["swap"].sum() == 0.0
    flat5 = pd.Series([5.0], index=pd.DatetimeIndex(["2020-01-01"], tz="UTC"))
    r5 = run_target_lots(md, path, costs=CostModel(**NO_FRICTION), rates=flat5, compute_metrics=False)
    assert r5.costs["swap"].sum() != pytest.approx(res.costs["swap"].sum())
    assert r5.costs["swap"].sum() < 0 and res.costs["swap"].sum() < 0


def test_buy_and_hold_frictionless_has_no_financing_and_default_pays_rate() -> None:
    bars = make_synthetic_bars(24 * 20, seed=4, start="2024-01-03")
    bh = buy_and_hold_benchmark(bars, frictionless=True)
    assert bh.costs.to_numpy().sum() == 0.0
    paid = buy_and_hold_benchmark(bars)
    lots = paid.meta["lots"]
    nights = rollover_nights(bars.index[1], bars["available_at"].iloc[-1])
    px = bars["close"].mean()
    # ~ -(fallback 3% + markup 2.5%) / 360 * notional per night
    assert paid.costs["swap"].sum() == pytest.approx(-lots * 100 * px * 0.055 / 360 * nights, rel=0.02)
