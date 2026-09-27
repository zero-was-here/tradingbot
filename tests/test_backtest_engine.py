"""Backtest engine tests with local stub sizer / risk manager (protocol-conformant)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.engine import (
    average_true_range,
    buy_and_hold_benchmark,
    run_backtest,
    run_target_lots,
)
from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.interfaces import PositionSizer, RiskContext, RiskDecision, RiskManager
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events
from aurum.execution.costs import CostModel
from aurum.models.volatility import ewma_volatility

NO_SWAP = dataclasses.replace(XAUUSD, swap_long_per_lot=0.0, swap_short_per_lot=0.0)


class LinearSizer:
    """lots = forecast * lots_per_unit (rounded) — deterministic and easy to reason about."""

    def __init__(self, lots_per_unit: float = 1.0) -> None:
        self.k = lots_per_unit
        self.calls: list[tuple] = []

    def target_lots(self, forecast: float, vol_ann: float, equity: float, price: float,
                    instrument: Instrument, *, current_lots: float = 0.0, drawdown: float = 0.0) -> float:
        self.calls.append((forecast, vol_ann, equity, price, current_lots, drawdown))
        return instrument.round_lots(forecast * self.k)


class VolTargetStub:
    def target_lots(self, forecast: float, vol_ann: float, equity: float, price: float,
                    instrument: Instrument, *, current_lots: float = 0.0, drawdown: float = 0.0) -> float:
        notional = forecast * 0.10 / max(vol_ann, 1e-6) * equity
        return instrument.round_lots(notional / (instrument.contract_size * price))


class CapRisk:
    """Caps |lots|; optionally halts from a given decision time; records everything."""

    def __init__(self, max_lots: float = 10.0, halt_at: pd.Timestamp | None = None) -> None:
        self.max_lots = max_lots
        self.halt_at = halt_at
        self.on_bar_calls: list[tuple[pd.Timestamp, float]] = []
        self.contexts: list[RiskContext] = []

    def on_bar(self, time: pd.Timestamp, equity: float) -> None:
        self.on_bar_calls.append((time, equity))

    def evaluate(self, ctx: RiskContext) -> RiskDecision:
        self.contexts.append(ctx)
        if self.halt_at is not None and ctx.time >= self.halt_at:
            return RiskDecision(0.0, halted=True, reasons=["max_drawdown"])
        a = max(-self.max_lots, min(self.max_lots, ctx.target_lots))
        return RiskDecision(a, reasons=["max_lots"] if a != ctx.target_lots else [])


class GreedyRisk:
    """Misbehaving risk manager that tries to ADD risk — the engine must clamp it."""

    def on_bar(self, time: pd.Timestamp, equity: float) -> None:
        pass

    def evaluate(self, ctx: RiskContext) -> RiskDecision:
        return RiskDecision(2.0 * ctx.target_lots + 0.5)


def _bars(n: int = 600, seed: int = 0, **kw) -> pd.DataFrame:
    return make_synthetic_bars(n, seed=seed, **kw)


def _momentum_forecast(bars: pd.DataFrame, n: int = 24) -> pd.Series:
    """Causal toy forecast: sign of the n-bar return (uses closes <= t)."""
    return np.sign(bars["close"].pct_change(n)).fillna(0.0)


def test_stubs_satisfy_protocols() -> None:
    assert isinstance(LinearSizer(), PositionSizer)
    assert isinstance(CapRisk(), RiskManager)


def test_engine_calls_sizer_and_risk_point_in_time() -> None:
    bars = _bars(300)
    fc = _momentum_forecast(bars)
    sizer, risk = LinearSizer(2.0), CapRisk(max_lots=1.0)
    res = run_backtest(bars, fc, sizer=sizer, risk=risk)
    n = len(bars)
    assert len(sizer.calls) == n - 1
    assert len(risk.contexts) == n - 1
    assert len(risk.on_bar_calls) == n          # every close, including the last
    vol = ewma_volatility(bars["close"])
    for t in (0, 50, 200, n - 2):
        ctx = risk.contexts[t]
        assert ctx.time == bars["available_at"].iloc[t]          # decision at the close of t
        assert ctx.price == bars["close"].iloc[t]
        assert ctx.spread == bars["spread"].iloc[t]
        assert ctx.vol_ann == pytest.approx(vol.iloc[t])
        assert ctx.current_lots == res.position_close.iloc[t]
        assert ctx.equity == pytest.approx(res.equity.iloc[t])
        assert sizer.calls[t][0] == fc.iloc[t]
        assert risk.on_bar_calls[t][0] == bars["available_at"].iloc[t]
    # executed position at t+1 == approved lots at t (no stops here)
    approved = np.clip(res.target.to_numpy()[:-1], -1.0, 1.0)
    np.testing.assert_allclose(res.positions.to_numpy()[1:], approved)
    # interventions recorded exactly where the cap bit
    capped = np.abs(res.target.to_numpy()[:-1]) > 1.0
    assert len(res.risk_events) == int(capped.sum())
    assert (res.risk_events["reasons"] == "max_lots").all()
    assert (res.risk_events["approved"].abs() <= 1.0).all()
    assert abs(res.reconcile()["residual"]) < 1e-6
    assert res.metrics["n_trades"] == len(res.trades)
    assert res.forecast is not None and res.forecast.iloc[10] == fc.iloc[10]


def test_halt_flattens_and_tags_risk_exit() -> None:
    bars = _bars(200)
    fc = pd.Series(1.0, index=bars.index)
    halt_at = bars["available_at"].iloc[100]
    res = run_backtest(bars, fc, sizer=LinearSizer(1.0), risk=CapRisk(halt_at=halt_at))
    assert (res.positions.iloc[1:101] == 1.0).all()
    assert (res.positions.iloc[101:] == 0.0).all()
    assert res.trades.iloc[0]["exit_reason"] == "risk"
    ev = res.risk_events
    assert ev["halted"].all() and len(ev) == 199 - 100
    assert ev["time"].iloc[0] == halt_at


def test_reduce_only_is_enforced() -> None:
    bars = _bars(150)
    fc = _momentum_forecast(bars, 12) * 0.5
    res = run_backtest(bars, fc, sizer=LinearSizer(1.0), risk=GreedyRisk())
    req = res.target.to_numpy()[:-1]
    pos = res.positions.to_numpy()[1:]
    assert (np.abs(pos) <= np.abs(req) + 1e-9).all()
    assert (pos * req >= -1e-12).all()
    assert res.meta["n_risk_clamped"] > 0
    assert res.risk_events["reasons"].str.contains("clamped").any()


def test_stop_atr_uses_atr_known_at_decision() -> None:
    bars = _bars(800, seed=5)
    fc = _momentum_forecast(bars, 12)
    k = 1.0
    res = run_backtest(bars, fc, sizer=LinearSizer(1.0), stop_atr_mult=k, costs=CostModel())
    atr = average_true_range(bars, 14).to_numpy()
    close, opn = bars["close"].to_numpy(), bars["open"].to_numpy()
    stops = res.trades[res.trades["exit_reason"] == "stop"]
    assert len(stops) >= 5
    fills = res.fills.set_index(["bar", "kind"])
    for _, tr in stops.iterrows():
        e = int(tr["entry_bar"])                     # fill bar of the entry
        d = e - 1                                    # decision bar of the entry
        s = int(tr["side"])
        # distance from the ATR known at the decision, anchored at the entry (mid open of e)
        level = opn[e] - s * k * atr[d]
        mid = fills.loc[(int(tr["exit_bar"]), "stop"), "mid"]
        mid = float(mid.iloc[0]) if isinstance(mid, pd.Series) else float(mid)
        gap = opn[int(tr["exit_bar"])]
        assert mid == pytest.approx(level, abs=1e-9) or mid == pytest.approx(gap, abs=1e-9)
        assert int(tr["exit_bar"]) > e or mid == pytest.approx(level, abs=1e-9)  # never at entry open
    assert close.size == opn.size
    assert abs(res.reconcile()["residual"]) < 1e-6
    assert res.trades["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - res.equity.iloc[0], abs=1e-6)


def test_take_profit_and_cooldown() -> None:
    bars = _bars(800, seed=6)
    fc = pd.Series(1.0, index=bars.index)            # always long -> re-enters after exits
    res = run_backtest(bars, fc, sizer=LinearSizer(1.0), stop_atr_mult=1.0,
                       take_profit_atr_mult=1.5, stop_cooldown_bars=5)
    reasons = set(res.trades["exit_reason"])
    assert {"stop", "take_profit"} <= reasons
    cd = res.risk_events[res.risk_events["reasons"].str.contains("stop_cooldown")]
    assert len(cd) > 0
    # after each stop exit, no re-entry for 5 decisions
    for _, tr in res.trades[res.trades["exit_reason"] == "stop"].iterrows():
        xb = int(tr["exit_bar"])
        assert (res.positions.iloc[xb + 1: xb + 6] == 0.0).all()


def test_point_in_time_future_perturbation() -> None:
    """Changing prices after bar k must not change anything decided up to k."""
    bars = _bars(500, seed=11)
    k = 300
    rng = np.random.default_rng(0)
    pert = bars.copy()
    factor = np.exp(np.cumsum(rng.normal(0, 0.01, len(bars) - k - 1)))
    for col in ("open", "high", "low", "close"):
        vals = pert[col].to_numpy(copy=True)
        vals[k + 1:] *= factor
        pert[col] = vals
    pert["spread"] = pert["spread"].where(pert.index <= pert.index[k], 3.0)
    kw = dict(sizer=VolTargetStub(), risk=CapRisk(max_lots=5.0), stop_atr_mult=2.0)
    a = run_backtest(bars, _momentum_forecast(bars), **kw)
    b = run_backtest(pert, _momentum_forecast(pert), **kw)
    np.testing.assert_array_equal(a.equity.iloc[: k + 1].to_numpy(), b.equity.iloc[: k + 1].to_numpy())
    # the position decided at the close of k (held in k+1) is identical as well
    np.testing.assert_array_equal(a.target.iloc[: k + 1].to_numpy(), b.target.iloc[: k + 1].to_numpy())
    np.testing.assert_array_equal(a.positions.iloc[: k + 2].to_numpy(), b.positions.iloc[: k + 2].to_numpy())
    assert not np.array_equal(a.equity.to_numpy(), b.equity.to_numpy())  # the future did change


def test_window_start_end() -> None:
    bars = _bars(400)
    fc = _momentum_forecast(bars)
    res = run_backtest(bars, fc, sizer=LinearSizer(), start=100, end=bars.index[299])
    assert res.equity.index[0] == bars.index[100]
    assert res.equity.index[-1] == bars.index[299]
    assert res.equity.iloc[0] == 100_000.0
    assert res.positions.iloc[0] == 0.0
    full_vol = ewma_volatility(bars["close"])
    risk = CapRisk()
    run_backtest(bars, fc, sizer=LinearSizer(), risk=risk, start=100, end=299)
    assert risk.contexts[0].vol_ann == pytest.approx(full_vol.iloc[100])   # warmed-up, not reset
    with pytest.raises(ValueError):
        run_backtest(bars, fc, sizer=LinearSizer(), start=299, end=299)


def test_run_target_lots_nan_holds_and_matches_engine() -> None:
    bars = _bars(300, seed=2)
    fc = _momentum_forecast(bars)
    via_engine = run_backtest(bars, fc, sizer=LinearSizer(1.0))
    via_path = run_target_lots(bars, fc * 1.0)
    np.testing.assert_allclose(via_engine.equity.to_numpy(), via_path.equity.to_numpy())
    path = pd.Series(np.nan, index=bars.index)
    path.iloc[10] = 1.0
    path.iloc[50] = -0.5
    res = run_target_lots(bars, path)
    pos = res.positions.to_numpy()
    assert (pos[11:51] == 1.0).all() and (pos[51:] == -0.5).all() and (pos[:11] == 0).all()
    assert len(res.fills) == 2


def test_buy_and_hold_benchmark() -> None:
    bars = _bars(400, seed=4)
    bh = buy_and_hold_benchmark(bars, frictionless=True)
    lots = XAUUSD.round_lots(100_000 / (100 * bars["close"].iloc[0]))
    assert bh.meta["lots"] == lots
    assert (bh.positions.iloc[1:] == lots).all()
    expected = lots * 100 * (bars["close"].iloc[-1] - bars["open"].iloc[1])
    assert bh.equity.iloc[-1] - bh.equity.iloc[0] == pytest.approx(expected)
    costly = buy_and_hold_benchmark(bars, lots=2.0)
    assert (costly.positions.iloc[1:] == 2.0).all()
    assert costly.costs["swap"].sum() < 0 and costly.metrics["total_costs"] > 0
    assert len(costly.trades) == 1 and costly.trades.iloc[0]["exit_reason"] == "end"
    with pytest.raises(ValueError):
        buy_and_hold_benchmark(bars, lots=1.0, notional=1e5)


def test_events_windows_are_point_in_time() -> None:
    bars = _bars(24 * 45, seed=8, start="2024-01-01")
    ev = make_synthetic_events(bars.index[0], bars.index[-1])
    ev["actual"] = np.arange(len(ev), dtype=float)          # an outcome column
    md = MarketData(bars=bars, events=ev)
    risk = CapRisk()
    run_backtest(md, pd.Series(0.0, index=bars.index), sizer=LinearSizer(), risk=risk,
                 event_horizon_hours=6.0)
    nfp = ev["time"].iloc[0]
    seen_upcoming = seen_recent = False
    for ctx in risk.contexts:
        up, rc = ctx.upcoming_events, ctx.recent_events
        assert up is not None and rc is not None
        assert "actual" not in up.columns                   # outcomes hidden before release
        assert (up["time"] >= ctx.time).all() and (up["time"] <= ctx.time + pd.Timedelta(hours=6)).all()
        assert (rc["time"] < ctx.time).all()
        if nfp in set(up["time"]):
            seen_upcoming = True
        if nfp in set(rc["time"]):
            seen_recent = True
            assert "actual" in rc.columns
    assert seen_upcoming and seen_recent


def test_accepts_market_data_and_costs_change_results() -> None:
    bars = _bars(300, seed=9)
    fc = _momentum_forecast(bars)
    cheap = run_backtest(MarketData(bars=bars), fc, sizer=LinearSizer(), costs=CostModel.zero(),
                         instrument=NO_SWAP)
    dear = run_backtest(bars, fc, sizer=LinearSizer(), costs=CostModel(spread_multiplier=3.0))
    assert cheap.metrics["total_costs"] == 0.0
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]
    assert dear.metrics["total_costs"] > 0


def test_performance_100k_bars_under_5s() -> None:
    bars = make_synthetic_bars(100_000, seed=1)
    fc = pd.Series(np.sign(np.sin(np.arange(len(bars)) / 50.0)), index=bars.index)
    # CPU time of this process, not wall time: the requirement is about the engine's own cost,
    # and wall time on a shared/oversubscribed CI box measures the other tenants instead
    # (observed: 22 s wall vs 2.4 s CPU at load average ~200).
    t0 = time.process_time()
    res = run_backtest(bars, fc, sizer=VolTargetStub(), risk=CapRisk(max_lots=5.0), stop_atr_mult=3.0)
    elapsed = time.process_time() - t0
    assert len(res.equity) == 100_000
    assert abs(res.reconcile()["residual"]) < 1e-5
    assert elapsed < 5.0, f"100k-bar backtest took {elapsed:.2f}s CPU"


def test_result_save_includes_pnl_breakdown(tmp_path) -> None:
    bars = _bars(200, seed=12)
    res = run_backtest(bars, _momentum_forecast(bars), sizer=LinearSizer(), risk=CapRisk(max_lots=0.5),
                       stop_atr_mult=2.0)
    out = res.save(tmp_path / "run")
    ts = pd.read_parquet(out / "timeseries.parquet")
    assert {"equity", "positions", "position_close", "cost_spread", "pnl_price", "pnl_net"} <= set(ts.columns)
    np.testing.assert_allclose(ts["equity"].to_numpy(), res.equity.to_numpy())
    assert (out / "trades.csv").exists() and (out / "risk_events.csv").exists()
    assert "sharpe" in (out / "metrics.json").read_text()


# ---- reviewer: adversarial regressions ---------------------------------------------------------------
def _hand_bars(rows: list[tuple[float, float, float, float]], start: str = "2024-01-08 00:00") -> pd.DataFrame:
    from aurum.data.schema import make_bars

    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="1h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["spread"] = 0.30
    return make_bars(df, "H1")


def test_protective_stop_is_placed_at_entry_not_beyond_a_gap() -> None:
    """Regression: stops used to be anchored at close[t]; a gap through that level at the next
    open entered the position and stopped it out at the same open (spread paid twice)."""
    rows = [(2000.0, 2001.0, 1999.0, 2000.0)] * 20 + [(1990.0, 1991.0, 1989.0, 1990.0)] * 6
    bars = _hand_bars(rows)
    fc = pd.Series(0.0, index=bars.index)
    fc.iloc[19:22] = 1.0                        # long decided at close 19 -> filled at the 1990 gap
    k = 1.0
    res = run_backtest(bars, fc, sizer=LinearSizer(1.0), stop_atr_mult=k, instrument=NO_SWAP)
    atr = average_true_range(bars).to_numpy()
    assert bars["close"].iloc[19] - k * atr[19] > bars["open"].iloc[20]   # old level beyond the gap
    assert (res.trades["exit_reason"] != "stop").all()
    assert res.positions.iloc[20:23].tolist() == [1.0, 1.0, 1.0]
    assert res.position_close.iloc[20] == 1.0
    assert len(res.fills) == 2                  # one entry, one signal exit
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "signal" and tr["costs"] == pytest.approx(2 * (15 + 2 + 0.02 * 100 * 2))


def test_stop_levels_fixed_for_life_of_position_and_short_cooldown() -> None:
    # short entered at bar 2 open (100.0); ATR(=2) stop 2.0 above entry -> 102; scaled in at
    # bar 3 without moving the level; bar 5 trades through 102 -> stopped; cooldown blocks
    # re-shorting for 2 decisions but a long is allowed immediately.
    rows = [(100.0, 101.0, 99.0, 100.0)] * 5 + [(100.5, 102.5, 100.0, 102.0)] + \
        [(102.0, 103.0, 101.0, 102.0)] * 6
    bars = _hand_bars(rows)
    path = pd.Series(np.nan, index=bars.index)
    path.iloc[1] = -1.0
    path.iloc[2] = -2.0
    path.iloc[5:8] = -1.0                       # wants to re-short right after the stop
    path.iloc[8] = 1.0
    res = run_target_lots(bars, path, stop_atr_mult=1.0, stop_cooldown_bars=2, instrument=NO_SWAP,
                          costs=CostModel.zero())
    atr = average_true_range(bars).to_numpy()
    stops = res.fills[res.fills["kind"] == "stop"]
    assert len(stops) == 1 and int(stops["bar"].iloc[0]) == 5
    assert float(stops["mid"].iloc[0]) == pytest.approx(100.0 + atr[1])   # entry open + ATR[decision]
    assert float(stops["lots"].iloc[0]) == pytest.approx(2.0)             # whole scaled position
    pos = res.positions.to_numpy()
    assert pos[6] == 0.0 and pos[7] == 0.0      # cooldown: 2 blocked re-short decisions (5, 6)
    assert pos[8] == -1.0                       # cooldown over
    assert pos[9] == 1.0
    cd = res.risk_events[res.risk_events["reasons"].str.contains("stop_cooldown")]
    assert list(cd["bar"]) == [5, 6]


@pytest.mark.parametrize("seed", range(6))
def test_engine_random_paths_keep_pnl_identity_and_reduce_only(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    bars = _bars(700, seed=seed, model="jump" if seed % 2 else "gbm")
    fc = pd.Series(np.round(rng.uniform(-1.2, 1.2, len(bars)), 1), index=bars.index)
    fc = fc.where(rng.random(len(bars)) < 0.15).ffill().fillna(0.0)
    cm = CostModel(spread_multiplier=1.3, slippage_range_frac=0.05, impact_coef=0.02,
                   commission_per_lot=3.0)
    res = run_backtest(bars, fc, sizer=LinearSizer(3.0), risk=CapRisk(max_lots=2.0), costs=cm,
                       stop_atr_mult=float(rng.uniform(0.5, 3.0)),
                       take_profit_atr_mult=float(rng.uniform(0.5, 4.0)),
                       stop_cooldown_bars=int(rng.integers(0, 4)), start=int(rng.integers(0, 50)))
    change = res.equity.iloc[-1] - res.equity.iloc[0]
    assert abs(res.reconcile()["residual"]) < 1e-6
    assert res.trades["pnl"].sum() == pytest.approx(change, abs=1e-6)
    np.testing.assert_allclose(res.equity.diff().iloc[1:], res.pnl["net"].iloc[1:], atol=1e-6)
    req = res.target.to_numpy()[:-1]
    pos = res.positions.to_numpy()[1:]
    assert (np.abs(pos) <= np.minimum(np.abs(req), 2.0) + 1e-9).all()     # risk cap & reduce-only
    assert (pos * req >= -1e-12).all()
    # a position entered (from flat or by reversal) at an open is never stopped/taken out AT
    # that same open: its levels are anchored at the entry, strictly away from the market
    f = res.fills.reset_index(drop=True)
    prev = f["position_after"].shift(1, fill_value=0.0)
    entered = (f["kind"] == "open") & (np.sign(f["position_after"]) != np.sign(prev)) \
        & (f["position_after"] != 0.0)
    entry_open = dict(zip(f.loc[entered, "bar"], f.loc[entered, "mid"], strict=True))
    prot = f[f["kind"].isin(["stop", "take_profit"])]
    n_same_bar = 0
    for _, row in prot.iterrows():
        b = int(row["bar"])
        if b in entry_open:
            n_same_bar += 1
            assert abs(float(row["mid"]) - entry_open[b]) > 1e-9
    assert len(prot) > 0
    assert set(res.trades["exit_reason"]) <= {"signal", "stop", "take_profit", "risk", "end"}


def test_upcoming_events_hide_all_outcome_columns() -> None:
    """The data layer declares actual/forecast/previous as outcome columns (CSV imports); none of
    them (nor surprise-like columns) may be visible before the scheduled release time."""
    bars = _bars(24 * 40, seed=8, start="2024-01-01")
    ev = make_synthetic_events(bars.index[0], bars.index[-1])
    for col in ("actual", "forecast", "previous", "Surprise_Z", "actual_value"):
        ev[col] = np.arange(len(ev), dtype=float)
    risk = CapRisk()
    run_backtest(MarketData(bars=bars, events=ev), pd.Series(0.0, index=bars.index),
                 sizer=LinearSizer(), risk=risk, event_horizon_hours=6.0)
    seen_up = seen_rc = False
    for ctx in risk.contexts:
        up, rc = ctx.upcoming_events, ctx.recent_events
        assert not ({"actual", "forecast", "previous", "Surprise_Z", "actual_value"} & set(up.columns))
        assert {"name", "time", "importance"} <= set(up.columns)
        seen_up |= len(up) > 0
        if len(rc):
            seen_rc = True
            assert {"actual", "forecast", "previous"} <= set(rc.columns)   # released: visible
    assert seen_up and seen_rc


def test_cooldown_is_applied_before_risk_sees_the_order() -> None:
    """The risk manager must evaluate the order actually sent: during a post-stop cooldown a
    blocked re-entry reaches risk as target 0 (so trade counters are not consumed)."""
    rows = [(100.0, 101.0, 99.0, 100.0)] * 3 + [(100.0, 100.5, 96.0, 97.0)] + \
        [(97.0, 98.0, 96.0, 97.0)] * 6
    bars = _hand_bars(rows)
    fc = pd.Series(1.0, index=bars.index)
    risk = CapRisk()
    res = run_backtest(bars, fc, sizer=LinearSizer(1.0), risk=risk, stop_atr_mult=1.0,
                       stop_cooldown_bars=3, instrument=NO_SWAP)
    stop_bar = int(res.fills.loc[res.fills["kind"] == "stop", "bar"].iloc[0])
    assert stop_bar == 3
    blocked = [c for c in risk.contexts if c.bar_index in (3, 4, 5)]
    assert [c.target_lots for c in blocked] == [0.0, 0.0, 0.0]   # sizer wanted +1 each time
    assert res.target.iloc[3:6].tolist() == [1.0, 1.0, 1.0]       # the sizer's request is kept
    assert res.positions.iloc[4:7].tolist() == [0.0, 0.0, 0.0]
    assert res.positions.iloc[7] == 1.0
    ev = res.risk_events.set_index("bar")
    assert all("stop_cooldown" in ev.loc[b, "reasons"] for b in (3, 4, 5))
