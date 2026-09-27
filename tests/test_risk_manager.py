"""Tests for aurum.risk.manager."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aurum.core.instrument import XAUUSD
from aurum.core.interfaces import RiskContext, RiskManager
from aurum.data.synthetic import make_synthetic_events
from aurum.risk.manager import RiskLimits, StandardRiskManager, no_new_risk, toward_zero

T0 = pd.Timestamp("2024-03-04 10:00", tz="UTC")  # a Monday
EQ = 100_000.0
PX = 2_000.0


def ctx(target, current=0.0, *, t=T0, equity=EQ, price=PX, spread=0.3, **kw) -> RiskContext:
    return RiskContext(time=t, equity=equity, current_lots=current, target_lots=target, price=price,
                       spread=spread, vol_ann=0.15, **kw)


def loose(**kw) -> RiskLimits:
    base = dict(max_leverage=None, max_margin_utilisation=None, max_daily_loss=None, max_drawdown=None)
    base.update(kw)
    return RiskLimits(**base)


def test_implements_protocol():
    assert isinstance(StandardRiskManager(), RiskManager)


def test_pass_through_when_nothing_fires():
    rm = StandardRiskManager(loose())
    d = rm.evaluate(ctx(0.5, 0.2))
    assert d.approved_lots == 0.5 and not d.halted and not d.modified
    assert rm.events_frame().empty


def test_can_only_reduce_risk_property():
    rng = np.random.default_rng(0)
    limits = RiskLimits(max_lots=1.0, max_leverage=2.0, max_spread=0.5, max_trades_per_day=3,
                        max_daily_loss=None, max_drawdown=None)
    rm = StandardRiskManager(limits)
    for i in range(400):
        target = float(np.round(rng.normal(0, 1.5), 2))
        current = float(np.round(rng.normal(0, 1.0), 2))
        spread = float(rng.choice([0.2, 0.8]))
        t = T0 + pd.Timedelta(hours=i)
        d = rm.evaluate(ctx(target, current, t=t, spread=spread))
        a = d.approved_lots
        # same sign as target (or 0) and never larger in magnitude
        assert a == 0.0 or np.sign(a) == np.sign(target)
        assert abs(a) <= abs(target) + 1e-12
        if spread > 0.5:  # no-new-risk rule: never beyond current in the risk-increasing direction
            assert a == 0.0 or (np.sign(a) == np.sign(current) and abs(a) <= abs(current) + 1e-12)
        if d.modified:
            assert all(isinstance(r, str) and r for r in d.reasons)
    assert no_new_risk(0.5, -0.3) == 0.0 and no_new_risk(0.5, 0.3) == 0.3 and no_new_risk(-0.2, -0.3) == -0.2
    assert toward_zero(0.7, 0.5) == 0.5 and toward_zero(-0.1, 0.5) == 0.0


def test_exposure_caps_with_reasons():
    rm = StandardRiskManager(RiskLimits(max_lots=0.8, max_leverage=None, max_margin_utilisation=None))
    d = rm.evaluate(ctx(1.5))
    assert d.approved_lots == 0.8 and "max_lots" in d.reasons[0]
    rm = StandardRiskManager(RiskLimits(max_leverage=2.0, max_margin_utilisation=None))
    d = rm.evaluate(ctx(-3.0))  # 3 lots = 600k notional = 6x
    assert d.approved_lots == pytest.approx(-1.0) and "max_leverage" in d.reasons[0]


def test_margin_utilisation_cap():
    # margin_rate 1%: 50% utilisation of 100k equity allows 5M notional = 25 lots
    rm = StandardRiskManager(RiskLimits(max_leverage=None, max_margin_utilisation=0.5))
    d = rm.evaluate(ctx(40.0))
    assert d.approved_lots == pytest.approx(25.0)
    assert "margin" in d.reasons[0]
    assert XAUUSD.margin_required(d.approved_lots, PX) <= 0.5 * EQ + 1e-6
    rm2 = StandardRiskManager(RiskLimits(max_leverage=None, max_margin_utilisation=0.1))
    assert rm2.evaluate(ctx(10.0)).approved_lots == pytest.approx(5.0)


def test_daily_loss_halt():
    rm = StandardRiskManager(RiskLimits(max_daily_loss=0.03, max_drawdown=0.5))
    day = pd.Timestamp("2024-03-05 00:00", tz="UTC")
    rm.on_bar(day, 100_000)
    rm.on_bar(day + pd.Timedelta(hours=1), 98_000)
    d = rm.evaluate(ctx(0.5, 0.5, t=day + pd.Timedelta(hours=1), equity=98_000))
    assert not d.halted and d.approved_lots == 0.5
    rm.on_bar(day + pd.Timedelta(hours=2), 96_900)  # -3.1% vs day start
    assert rm.halted
    d = rm.evaluate(ctx(0.5, 0.5, t=day + pd.Timedelta(hours=2), equity=96_900))
    assert d.halted and d.approved_lots == 0.0
    assert "daily loss" in " ".join(d.reasons).lower()
    # persists into the next day (SPEC: hard kill until reset)
    rm.on_bar(day + pd.Timedelta(days=1), 96_900)
    assert rm.evaluate(ctx(0.2, 0.0, t=day + pd.Timedelta(days=1), equity=96_900)).halted


def test_daily_loss_non_persistent_mode_and_day_start():
    lim = RiskLimits(max_daily_loss=0.03, max_drawdown=0.5, daily_loss_persistent=False)
    rm = StandardRiskManager(lim)
    rm.on_bar(pd.Timestamp("2024-03-05 22:00", tz="UTC"), 100_000)
    # first decision of the new day is at 01:00 (not on the boundary): day start = last equity
    rm.on_bar(pd.Timestamp("2024-03-06 01:00", tz="UTC"), 97_500)
    assert rm.state.day_start_equity == 100_000
    rm.on_bar(pd.Timestamp("2024-03-06 02:00", tz="UTC"), 96_900)
    assert rm.halted
    rm.on_bar(pd.Timestamp("2024-03-07 00:00", tz="UTC"), 96_900)
    assert not rm.halted
    assert rm.state.day_start_equity == 96_900
    # broker-rollover day boundary
    rm2 = StandardRiskManager(RiskLimits(daily_reset="rollover"))
    rm2.on_bar(pd.Timestamp("2024-03-06 20:00", tz="UTC"), 100_000)
    rm2.on_bar(pd.Timestamp("2024-03-06 21:00", tz="UTC"), 101_000)
    assert rm2.state.day_start_equity == 101_000


def test_max_drawdown_kill_persists_across_instances(tmp_path):
    path = tmp_path / "risk_state.json"
    lim = RiskLimits(max_drawdown=0.20, max_daily_loss=None)
    rm = StandardRiskManager(lim, state_path=path)
    t = T0
    for eq in (100_000, 110_000, 95_000, 87_900):  # 20.1% below the 110k peak
        rm.on_bar(t, eq)
        t += pd.Timedelta(hours=1)
    assert rm.halted and "drawdown" in rm.halt_reason
    assert path.exists() and json.loads(path.read_text())["halted"] is True

    rm2 = StandardRiskManager(lim, state_path=path)
    assert rm2.halted
    d = rm2.evaluate(ctx(1.0, 0.0, t=t, equity=90_000))
    assert d.halted and d.approved_lots == 0.0
    assert d.reasons and "HALTED" in d.reasons[0]

    with pytest.raises(ValueError):
        rm2.reset_halt()
    with pytest.raises(ValueError):
        rm2.reset_halt(confirm="reset")
    assert rm2.halted
    rm2.reset_halt(confirm="RESET", equity=90_000)
    assert not rm2.halted
    assert rm2.evaluate(ctx(0.3, 0.0, t=t, equity=90_000)).approved_lots == 0.3
    rm3 = StandardRiskManager(lim, state_path=path)
    assert not rm3.halted and rm3.state.peak_equity == 90_000


def test_corrupt_state_file_fails_safe(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    rm = StandardRiskManager(state_path=path)
    assert rm.halted and rm.evaluate(ctx(0.5)).approved_lots == 0.0


def test_manual_halt_and_zero_equity():
    rm = StandardRiskManager()
    rm.halt("operator stop")
    assert rm.evaluate(ctx(0.5, 0.5)).approved_lots == 0.0
    rm.reset_halt(confirm="RESET")
    rm2 = StandardRiskManager()
    d = rm2.evaluate(ctx(0.5, 0.5, equity=-10.0))
    assert d.halted and d.approved_lots == 0.0


def test_spread_guard_blocks_new_risk_allows_reducing():
    rm = StandardRiskManager(loose(max_spread=0.5))
    wide = dict(spread=1.2)
    d = rm.evaluate(ctx(0.5, 0.0, **wide))
    assert d.approved_lots == 0.0 and "spread" in d.reasons[0]
    d = rm.evaluate(ctx(0.8, 0.3, **wide))  # increase -> held at current
    assert d.approved_lots == 0.3
    d = rm.evaluate(ctx(0.1, 0.3, **wide))  # reduction -> allowed untouched
    assert d.approved_lots == 0.1 and not d.modified
    d = rm.evaluate(ctx(-0.5, 0.3, **wide))  # reversal -> close only
    assert d.approved_lots == 0.0
    d = rm.evaluate(ctx(0.8, 0.3, spread=0.3))  # normal spread -> pass
    assert d.approved_lots == 0.8


def _events() -> pd.DataFrame:
    return pd.DataFrame({
        "time": [pd.Timestamp("2024-03-08 13:30", tz="UTC"), pd.Timestamp("2024-03-12 12:30", tz="UTC")],
        "name": ["NFP", "CPI"],
        "importance": [3, 2],
    })


def test_event_blackout_no_new_risk_mode():
    ev = _events()
    rm = StandardRiskManager(loose(), events=ev)
    t_in = pd.Timestamp("2024-03-08 13:10", tz="UTC")
    d = rm.evaluate(ctx(0.5, 0.2, t=t_in))
    assert d.approved_lots == 0.2 and "NFP" in d.reasons[0]
    assert rm.evaluate(ctx(0.1, 0.2, t=t_in)).approved_lots == 0.1
    # after-window
    assert rm.evaluate(ctx(0.5, 0.2, t=pd.Timestamp("2024-03-08 13:55", tz="UTC"))).approved_lots == 0.2
    # outside the window
    assert rm.evaluate(ctx(0.5, 0.2, t=pd.Timestamp("2024-03-08 12:00", tz="UTC"))).approved_lots == 0.5
    # importance-2 CPI in this frame is ignored at min importance 3
    assert rm.evaluate(ctx(0.5, 0.2, t=pd.Timestamp("2024-03-12 12:30", tz="UTC"))).approved_lots == 0.5
    # the same frame passed through the RiskContext
    rm2 = StandardRiskManager(loose())
    up = ev[ev["time"] >= t_in]
    d = rm2.evaluate(ctx(0.5, 0.0, t=t_in, upcoming_events=up))
    assert d.approved_lots == 0.0
    rec = ev[ev["time"] < pd.Timestamp("2024-03-08 13:45", tz="UTC")]
    d = rm2.evaluate(ctx(0.5, 0.0, t=pd.Timestamp("2024-03-08 13:45", tz="UTC"), recent_events=rec))
    assert d.approved_lots == 0.0


def test_event_blackout_flatten_mode_and_lookahead():
    rm = StandardRiskManager(loose(blackout_mode="flatten"), events=_events())
    d = rm.evaluate(ctx(0.5, 0.5, t=pd.Timestamp("2024-03-08 13:00", tz="UTC")))
    assert d.approved_lots == 0.0 and "flatten" in d.reasons[0]
    d = rm.evaluate(ctx(-0.4, -0.4, t=pd.Timestamp("2024-03-08 14:00", tz="UTC")))
    assert d.approved_lots == 0.0
    # H4 bars: an event inside the next holding bar is caught with event_lookahead_min
    t = pd.Timestamp("2024-03-08 12:00", tz="UTC")
    assert StandardRiskManager(loose(), events=_events()).evaluate(ctx(0.5, 0.0, t=t)).approved_lots == 0.5
    rm_h4 = StandardRiskManager(loose(event_lookahead_min=240), events=_events())
    assert rm_h4.evaluate(ctx(0.5, 0.0, t=t)).approved_lots == 0.0


def test_blackout_with_synthetic_calendar():
    ev = make_synthetic_events("2024-01-01", "2024-12-31")
    rm = StandardRiskManager(loose(), events=ev)
    nfp = ev.loc[ev["name"] == "NFP", "time"].iloc[2]
    assert rm.evaluate(ctx(1.0, 0.0, t=nfp - pd.Timedelta(minutes=15))).approved_lots == 0.0
    assert rm.evaluate(ctx(1.0, 0.0, t=nfp + pd.Timedelta(hours=3))).approved_lots == 1.0


def test_max_trades_per_day():
    rm = StandardRiskManager(loose(max_trades_per_day=2))
    t = pd.Timestamp("2024-03-05 01:00", tz="UTC")
    assert rm.evaluate(ctx(0.5, 0.0, t=t)).approved_lots == 0.5          # trade 1
    assert rm.evaluate(ctx(0.8, 0.5, t=t + pd.Timedelta(hours=1))).approved_lots == 0.8  # trade 2
    d = rm.evaluate(ctx(1.0, 0.8, t=t + pd.Timedelta(hours=2)))           # limit hit
    assert d.approved_lots == 0.8 and "max_trades_per_day" in d.reasons[0]
    assert rm.evaluate(ctx(0.2, 0.8, t=t + pd.Timedelta(hours=3))).approved_lots == 0.2  # reduce ok
    # unchanged position does not count as a trade; new day resets the counter
    assert rm.evaluate(ctx(1.0, 0.0, t=pd.Timestamp("2024-03-06 01:00", tz="UTC"))).approved_lots == 1.0
    # preview does not consume the budget
    rm2 = StandardRiskManager(loose(max_trades_per_day=1))
    for _ in range(3):
        assert rm2.evaluate(ctx(0.5, 0.0, t=t), commit=False).approved_lots == 0.5
    assert rm2.state.trades_today == 0


def test_stale_data_guard():
    rm = StandardRiskManager(loose(stale_data_seconds=120))
    d = rm.evaluate(ctx(0.5, 0.1, data_age_seconds=600))
    assert d.approved_lots == 0.1 and "stale" in d.reasons[0]
    assert rm.evaluate(ctx(0.0, 0.1, data_age_seconds=600)).approved_lots == 0.0
    assert rm.evaluate(ctx(0.5, 0.1, data_age_seconds=30)).approved_lots == 0.5
    assert rm.evaluate(ctx(0.5, 0.1)).approved_lots == 0.5  # backtests: age unknown


def test_invalid_inputs_and_rounding():
    rm = StandardRiskManager(loose())
    d = rm.evaluate(ctx(float("nan"), 0.3))
    assert d.approved_lots == 0.0 and "non-finite" in d.reasons[0]
    d = rm.evaluate(ctx(0.5, 0.3, price=float("nan")))
    assert d.approved_lots == 0.3
    d = rm.evaluate(ctx(0.1234, 0.0))
    assert d.approved_lots == 0.12 and "rounded" in d.reasons[0]


def test_events_frame_and_snapshot():
    rm = StandardRiskManager(RiskLimits(max_lots=0.5))
    rm.on_bar(T0, EQ)
    rm.evaluate(ctx(1.0))
    ev = rm.events_frame()
    assert list(ev.columns) == ["time", "requested", "current", "approved", "halted", "reasons"]
    assert len(ev) == 1 and ev["approved"].iloc[0] == 0.5
    snap = rm.snapshot()
    json.dumps(snap)
    assert snap["halted"] is False and snap["drawdown"] == 0.0
    with pytest.raises(ValueError):
        RiskLimits(blackout_mode="panic")
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown=1.5)


# ---------------------------------------------------------------------------------------
# Review regressions (adversarial)
# ---------------------------------------------------------------------------------------
def test_daily_loss_counts_the_last_bar_of_the_day_h1():
    """The 23:00 H1 bar closes at 00:00 = the next day's boundary. Its loss belongs to the
    day that is ending; before the fix the day rolled over first and the loss escaped."""
    rm = StandardRiskManager(RiskLimits(max_daily_loss=0.03, max_drawdown=0.5))
    day = pd.Timestamp("2024-03-05 00:00", tz="UTC")
    rm.on_bar(day, 100_000)
    rm.on_bar(day + pd.Timedelta(hours=23), 97_500)   # -2.5%: fine
    assert not rm.halted
    rm.on_bar(day + pd.Timedelta(hours=24), 96_600)   # -3.4% for 2024-03-05, marked at 00:00
    assert rm.halted and rm.state.halt_kind == "daily_loss"
    assert "2024-03-05" in rm.halt_reason
    d = rm.evaluate(ctx(0.5, 0.5, t=day + pd.Timedelta(hours=24), equity=96_600))
    assert d.halted and d.approved_lots == 0.0
    # a loss that stays inside the limit still rolls over cleanly
    ok = StandardRiskManager(RiskLimits(max_daily_loss=0.03, max_drawdown=0.5))
    ok.on_bar(day, 100_000)
    ok.on_bar(day + pd.Timedelta(hours=24), 97_100)   # -2.9%
    assert not ok.halted and ok.state.day_start_equity == 97_100


def test_daily_loss_on_d1_bars():
    """Every D1 decision sits on a day boundary: without settling the ending day the limit
    could never fire on daily bars (a -5.5% day passed silently before the fix)."""
    rm = StandardRiskManager(RiskLimits(max_daily_loss=0.03, max_drawdown=0.5))
    t = pd.Timestamp("2024-03-05 00:00", tz="UTC")
    for eq in (100_000, 100_500, 95_000):
        rm.on_bar(t, eq)
        t += pd.Timedelta(days=1)
    assert rm.halted and rm.state.halt_kind == "daily_loss"
    ev = rm.events_frame()
    assert ev["halted"].any() and ev["reasons"].str.contains("KILL SWITCH").any()


def test_boundary_settlement_in_non_persistent_mode_clears_next_day():
    lim = RiskLimits(max_daily_loss=0.03, max_drawdown=0.5, daily_loss_persistent=False)
    rm = StandardRiskManager(lim)
    t = pd.Timestamp("2024-03-05 00:00", tz="UTC")
    rm.on_bar(t, 100_000)
    rm.on_bar(t + pd.Timedelta(days=1), 96_000)  # yesterday breached, but yesterday is over
    assert not rm.halted
    assert rm.events_frame()["reasons"].str.contains("daily_loss").any()  # still recorded
    assert rm.evaluate(ctx(0.5, 0.0, t=t + pd.Timedelta(days=1), equity=96_000)).approved_lots == 0.5


def test_boundary_settlement_with_rollover_reset():
    rm = StandardRiskManager(RiskLimits(daily_reset="rollover", max_daily_loss=0.03, max_drawdown=0.5))
    rm.on_bar(pd.Timestamp("2024-03-05 21:00", tz="UTC"), 100_000)   # session start
    rm.on_bar(pd.Timestamp("2024-03-06 20:00", tz="UTC"), 97_500)
    assert not rm.halted
    rm.on_bar(pd.Timestamp("2024-03-06 21:00", tz="UTC"), 96_500)   # close of the session's last bar
    assert rm.halted


@pytest.mark.parametrize(
    "content",
    ["null", "[]", '"halted"', "{}", '{"halted": "false"}', '{"halted": false, "peak_equity": "abc"}',
     '{"halted": false, "peak_equity": NaN}', '{"halted": false, "trades_today": -1}'],
)
def test_malformed_state_file_fails_safe(tmp_path, content):
    """Well-formed JSON with the wrong shape used to crash (AttributeError) or half-load (a
    truthy "false" string, a non-numeric equity that crashed on the next bar)."""
    path = tmp_path / "state.json"
    path.write_text(content)
    rm = StandardRiskManager(state_path=path)
    assert rm.halted and rm.state.halt_kind == "state_file"
    rm.on_bar(T0, EQ)  # must not crash
    assert rm.evaluate(ctx(0.5, 0.0)).approved_lots == 0.0
    rm.reset_halt(confirm="RESET", equity=EQ)
    assert rm.evaluate(ctx(0.5, 0.0)).approved_lots == 0.5


def test_valid_state_file_round_trip(tmp_path):
    path = tmp_path / "state.json"
    rm = StandardRiskManager(RiskLimits(max_daily_loss=None), state_path=path)
    rm.on_bar(T0, 100_000)
    rm.on_bar(T0 + pd.Timedelta(hours=1), 104_000)
    rm2 = StandardRiskManager(RiskLimits(max_daily_loss=None), state_path=path)
    assert not rm2.halted and rm2.state.peak_equity == 104_000 and rm2.state.last_equity == 104_000


def test_event_importance_as_strings():
    """Calendar CSVs often carry importance as text; it used to raise TypeError."""
    t_in = pd.Timestamp("2024-03-08 13:10", tz="UTC")
    for imp, blocked in (("3", True), ("2", False), ("high", True), (None, True)):
        ev = pd.DataFrame({"time": [pd.Timestamp("2024-03-08 13:30", tz="UTC")], "name": ["NFP"], "importance": [imp]})
        d = StandardRiskManager(loose(), events=ev).evaluate(ctx(0.5, 0.0, t=t_in))
        assert (d.approved_lots == 0.0) is blocked, imp
        d2 = StandardRiskManager(loose()).evaluate(ctx(0.5, 0.0, t=t_in, upcoming_events=ev))
        assert (d2.approved_lots == 0.0) is blocked, imp


def test_unknown_data_age_is_treated_as_stale():
    rm = StandardRiskManager(loose(stale_data_seconds=120))
    d = rm.evaluate(ctx(0.5, 0.1, data_age_seconds=float("nan")))
    assert d.approved_lots == 0.1 and "unknown" in d.reasons[0]


def test_limits_accept_numpy_scalars():
    lim = RiskLimits(max_lots=np.int64(2), max_leverage=np.float64(2.5), max_spread=np.float32(0.5))
    assert StandardRiskManager(lim).evaluate(ctx(3.0)).approved_lots == pytest.approx(1.25)
    with pytest.raises(ValueError):
        RiskLimits(max_lots=True)
    with pytest.raises(ValueError):
        RiskLimits(max_lots=np.int64(0))


def test_out_of_order_time_cannot_rebase_the_day():
    """A stale ctx.time (e.g. an agent preview committed with an old timestamp) used to roll
    the day backwards and forwards, re-basing the day start and hiding a -3.5% day."""
    rm = StandardRiskManager(RiskLimits(max_daily_loss=0.03, max_drawdown=0.5, max_trades_per_day=1))
    tue = pd.Timestamp("2024-03-05 00:00", tz="UTC")
    rm.on_bar(tue, 100_000)
    rm.on_bar(tue + pd.Timedelta(hours=10), 98_000)
    assert rm.evaluate(ctx(0.1, 0.0, t=tue + pd.Timedelta(hours=10), equity=98_000)).approved_lots == 0.1
    stale = rm.evaluate(ctx(0.3, 0.1, t=tue - pd.Timedelta(hours=14), equity=98_000))
    assert stale.approved_lots == 0.1                   # trade budget was NOT reset
    assert rm.state.day_start_equity == 100_000
    rm.on_bar(tue + pd.Timedelta(hours=11), 96_500)     # -3.5% vs the real day start
    assert rm.halted and rm.state.halt_kind == "daily_loss"


def test_unparseable_last_time_in_state_fails_safe(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"halted": False, "last_time": "yesterday-ish"}))
    assert StandardRiskManager(state_path=path).halted
