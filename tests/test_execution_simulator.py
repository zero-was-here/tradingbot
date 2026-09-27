"""ExecutionSimulator: hand-computed scenarios + property-style PnL invariants."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from aurum.core.instrument import XAUUSD
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import CostModel, FinancingModel
from aurum.execution.simulator import ExecutionSimulator

# These scenarios pin the per-lot ("fixed") swap model they were written for (rate-based
# financing is the CostModel default; it is covered in tests/test_financing_*.py).
FIXED = FinancingModel.fixed()
NO_SLIP = CostModel(min_spread=0.0, slippage_fixed=0.0, slippage_range_frac=0.0, commission_per_lot=0.0,
                    financing=FIXED)
FIXED_SLIP = CostModel(min_spread=0.0, slippage_fixed=0.02, slippage_range_frac=0.0, commission_per_lot=0.0,
                       financing=FIXED)
NO_SWAP = dataclasses.replace(XAUUSD, swap_long_per_lot=0.0, swap_short_per_lot=0.0)


def hand_bars(rows: list[tuple[float, float, float, float]], start: str = "2024-01-08 10:00",
              spread: float = 0.30) -> pd.DataFrame:
    """H1 bars from (open, high, low, close) rows. 2024-01-08 is a Monday."""
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="1h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["spread"] = spread
    return make_bars(df, "H1")


# ---- hand-computed scenarios ---------------------------------------------------------------------
def test_entry_spread_cost_and_mark_to_market() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2000, 2002, 1998, 2001), (2001.5, 2003, 2000, 2002)])
    sim = ExecutionSimulator(bars, costs=NO_SLIP, initial_equity=100_000.0)
    r1 = sim.step(1.0)
    assert r1.costs["spread"] == pytest.approx(15.0)          # 0.15 * 100 oz, exactly
    assert r1.costs["slippage"] == 0.0 and r1.costs["commission"] == 0.0
    assert r1.fills[0].price == pytest.approx(2000.15)
    assert r1.price_pnl == pytest.approx(100.0)                # 1 lot * 100 * (2001 - 2000)
    assert r1.equity == pytest.approx(100_000 - 15 + 100)
    r2 = sim.step(1.0)                                         # hold: no trade
    assert r2.fills == []
    # gap 2001 -> 2001.5 (+50) and intrabar 2001.5 -> 2002 (+50)
    assert r2.price_pnl == pytest.approx(100.0)
    assert r2.equity == pytest.approx(100_185.0)
    assert r2.done
    res = sim.result()
    tr = res.trades
    assert len(tr) == 1
    row = tr.iloc[0]
    assert row["exit_reason"] == "end"
    assert row["entry_price"] == pytest.approx(2000.15)
    assert row["exit_price"] == pytest.approx(2002.0)
    assert row["pnl"] == pytest.approx(185.0)
    assert row["costs"] == pytest.approx(15.0)
    assert row["bars_held"] == 2
    assert res.positions.tolist() == [0.0, 1.0, 1.0]
    assert res.returns.iloc[0] == 0.0
    assert res.equity.iloc[0] == 100_000.0
    with pytest.raises(RuntimeError):
        sim.step(0.0)


def test_reversal_trades_two_lots_and_splits_costs() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2000, 2002, 1998, 2001), (2001, 2003, 1999, 2000),
                      (2000, 2001, 1998, 1999)])
    cm = dataclasses.replace(NO_SLIP, commission_per_lot=3.5)
    sim = ExecutionSimulator(bars, costs=cm)
    r1 = sim.step(1.0)
    assert r1.costs["commission"] == pytest.approx(3.5)
    r2 = sim.step(-1.0)
    assert len(r2.fills) == 1
    f = r2.fills[0]
    assert f.lots == pytest.approx(2.0) and int(f.side) == -1
    assert f.price == pytest.approx(2001 - 0.15)
    assert r2.costs["spread"] == pytest.approx(30.0)
    assert r2.costs["commission"] == pytest.approx(7.0)
    assert r2.position == -1.0
    sim.step(0.0)
    res = sim.result()
    tr = res.trades
    assert list(tr["exit_reason"]) == ["signal", "signal"]
    assert list(tr["side"]) == [1, -1]
    # trade 1: entry 15 + 3.5, half of the reversal (15 + 3.5)
    assert tr.iloc[0]["costs"] == pytest.approx(37.0)
    assert tr.iloc[0]["pnl"] == pytest.approx(100 * (2001 - 2000) - 37.0)
    # trade 2: other half of reversal + exit (15 + 3.5) ; short from 2001 to 2000
    assert tr.iloc[1]["costs"] == pytest.approx(37.0)
    assert tr.iloc[1]["pnl"] == pytest.approx(100 * (2001 - 2000) - 37.0)
    assert tr["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - res.equity.iloc[0])
    assert res.fills["lots"].tolist() == [1.0, 2.0, 1.0]


def test_partial_scale_in_out_is_one_trade() -> None:
    bars = hand_bars([(100, 101, 99, 100)] + [(100 + k, 102 + k, 99 + k, 101 + k) for k in range(5)])
    sim = ExecutionSimulator(bars, costs=NO_SLIP, instrument=NO_SWAP)
    for tgt in (1.0, 2.0, 1.0, 0.0, 0.0):
        sim.step(tgt)
    res = sim.result()
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["lots"] == pytest.approx(2.0)
    # entries at opens 100 & 101 (+0.15), exits at opens 102 & 103 (-0.15)
    assert t["entry_price"] == pytest.approx(100.65)
    assert t["exit_price"] == pytest.approx(102.35)
    assert t["pnl"] == pytest.approx(res.equity.iloc[-1] - res.equity.iloc[0])


def test_swap_triple_on_wednesday() -> None:
    rows = [(2000, 2001, 1999, 2000)] * 4
    wed = hand_bars(rows, start="2024-01-10 19:00")   # Wednesday 19:00..22:00 UTC
    sim = ExecutionSimulator(wed, costs=NO_SLIP)
    r1 = sim.step(1.0)            # filled at 20:00 open, held across Wed 21:00 rollover
    assert r1.costs["swap"] == pytest.approx(-45.0 * 3)
    r2 = sim.step(1.0)
    assert r2.costs["swap"] == 0.0
    tue = hand_bars(rows, start="2024-01-09 19:00")
    sim = ExecutionSimulator(tue, costs=NO_SLIP)
    assert sim.step(1.0).costs["swap"] == pytest.approx(-45.0)
    sim = ExecutionSimulator(wed, costs=NO_SLIP)
    assert sim.step(-2.0).costs["swap"] == pytest.approx(2 * 15.0 * 3)  # short receives
    # opened AT the rollover (21:00 open) -> not charged; closed at 21:00 open -> charged
    sim = ExecutionSimulator(wed, costs=NO_SLIP)
    sim.step(0.0)
    assert sim.step(1.0).costs["swap"] == 0.0
    res = sim.result()
    assert res.costs["swap"].sum() == 0.0
    assert res.trades.iloc[0]["swap"] == 0.0


def test_swap_over_weekend_matches_rollover_calendar() -> None:
    bars = make_synthetic_bars(24 * 12, seed=3, start="2024-01-10")  # spans a weekend
    sim = ExecutionSimulator(bars, costs=NO_SLIP)
    while not sim.done:
        sim.step(1.0)
    res = sim.result()
    # held from the open of bar 1 to the close (available_at) of the last bar
    t0 = bars.index[1]
    t1 = bars["available_at"].iloc[-1]
    nights = 0
    d = t0.normalize()
    while d <= t1:
        r = d + pd.Timedelta(hours=21)
        if t0 < r <= t1 and r.weekday() < 5:
            nights += 3 if r.weekday() == 2 else 1
        d += pd.Timedelta(days=1)
    assert nights >= 7
    assert res.costs["swap"].sum() == pytest.approx(-45.0 * nights)


def test_stop_loss_intrabar_fill() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2000, 2001, 1994, 1996), (1996, 1997, 1995, 1996)])
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    r = sim.step(1.0, stop_price=1995.0)
    assert r.exit_reason == "stop"
    assert r.position == 0.0 and r.position_open == 1.0
    exit_fill = r.fills[1]
    assert exit_fill.price == pytest.approx(1995.0 - 0.15 - 0.02)
    assert r.price_pnl == pytest.approx(100 * (1995 - 2000))
    assert r.pnl == pytest.approx(100 * (1995 - 2000) - 2 * (15 + 2))
    sim.step(0.0)
    res = sim.result()
    assert res.trades.iloc[0]["exit_reason"] == "stop"
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(1994.83)
    assert res.trades.iloc[0]["bars_held"] == 1
    assert res.positions.tolist() == [0.0, 1.0, 0.0]
    assert res.position_close.tolist() == [0.0, 0.0, 0.0]


def test_gap_through_stop_fills_at_open() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2000, 2001, 1998, 2000.5), (1990, 1992, 1988, 1991)])
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    sim.step(1.0)
    r = sim.step(1.0, stop_price=1995.0)
    assert r.exit_reason == "stop"
    assert r.fills[0].price == pytest.approx(1990 - 0.17)   # the open, not the stop level
    assert r.price_pnl == pytest.approx(100 * (1990 - 2000.5))


def test_short_stop_mirrored() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2000, 2006, 1999, 2004)])
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    r = sim.step(-1.0, stop_price=2005.0)
    assert r.exit_reason == "stop"
    assert r.fills[1].price == pytest.approx(2005 + 0.17)
    assert r.price_pnl == pytest.approx(-100 * (2005 - 2000))


def test_stop_first_when_both_touched_and_take_profit() -> None:
    both = hand_bars([(2000, 2001, 1999, 2000), (2000, 2006, 1994, 2003)])
    sim = ExecutionSimulator(both, costs=FIXED_SLIP)
    r = sim.step(1.0, stop_price=1995.0, take_profit=2005.0)
    assert r.exit_reason == "stop"
    assert r.fills[1].price == pytest.approx(1995 - 0.17)

    tp = hand_bars([(2000, 2001, 1999, 2000), (2000, 2006, 1999, 2003)])
    sim = ExecutionSimulator(tp, costs=FIXED_SLIP)
    r = sim.step(1.0, stop_price=1995.0, take_profit=2005.0)
    assert r.exit_reason == "take_profit"
    assert r.fills[1].price == pytest.approx(2005 - 0.15)   # limit: half spread, no slippage
    assert sim.result().trades.iloc[0]["exit_reason"] == "take_profit"

    gap_tp = hand_bars([(2000, 2001, 1999, 2000), (2000, 2001, 1999, 2000), (2007, 2008, 2006, 2007)])
    sim = ExecutionSimulator(gap_tp, costs=FIXED_SLIP)
    sim.step(1.0)
    r = sim.step(1.0, take_profit=2005.0)
    assert r.exit_reason == "take_profit"
    assert r.fills[0].price == pytest.approx(2007 - 0.15)


def test_bankruptcy_forces_flat() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2000, 2001, 1000, 1001), (1001, 1002, 1000, 1001),
                      (1001, 1002, 1000, 1001)])
    sim = ExecutionSimulator(bars, costs=NO_SLIP, initial_equity=10_000.0)
    sim.step(1.0)
    assert sim.bankrupt
    r = sim.step(1.0)
    assert r.position == 0.0
    assert sim.result().trades.iloc[0]["exit_reason"] == "risk"


def test_input_validation_and_reset() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000)] * 3)
    sim = ExecutionSimulator(bars, costs=NO_SLIP)
    with pytest.raises(ValueError):
        sim.step(float("nan"))
    with pytest.raises(ValueError):
        sim.step(1.0, stop_price=float("nan"))
    sim.step(0.37)                                   # rounded onto the 0.01 lot grid
    assert sim.position == pytest.approx(0.37)
    sim.reset(start=1, equity=50_000.0)
    assert sim.index == 1 and sim.equity == 50_000.0 and sim.position == 0.0
    sim.step(1.0)
    res = sim.result()
    assert len(res.equity) == 2 and res.equity.iloc[0] == 50_000.0
    assert set(sim.snapshot()) >= {"equity", "position", "price", "decision_time"}


# ---- property-style invariants ---------------------------------------------------------------------
def _random_run(seed: int, n: int = 300, with_stops: bool = True):
    rng = np.random.default_rng(seed)
    bars = make_synthetic_bars(n, seed=seed, spread=0.35, model="jump" if seed % 3 == 0 else "gbm")
    cm = CostModel(spread_multiplier=1.2, min_spread=0.1, slippage_fixed=0.02,
                   slippage_range_frac=0.03, impact_coef=0.01, commission_per_lot=2.5)
    sim = ExecutionSimulator(bars, costs=cm, initial_equity=250_000.0)
    levels = np.array([-2.0, -1.0, -0.5, -0.13, 0.0, 0.25, 1.0, 1.5, 2.0])
    tgt = 0.0
    targets = []
    while not sim.done:
        if rng.random() < 0.3:                 # persistence -> many "target == current" steps
            tgt = float(rng.choice(levels))
        stop = tp = None
        side = np.sign(tgt) if tgt != 0 else np.sign(sim.position)
        if with_stops and side != 0 and rng.random() < 0.4:
            px = sim.price
            stop = float(px - side * rng.uniform(0.5, 6.0))
            if rng.random() < 0.5:
                tp = float(px + side * rng.uniform(0.5, 6.0))
        targets.append((sim.index, tgt, sim.position))
        sim.step(tgt, stop_price=stop, take_profit=tp)
    return bars, sim, sim.result(), targets


@pytest.mark.parametrize("seed", range(12))
def test_pnl_identity_and_invariants(seed: int) -> None:
    bars, sim, res, targets = _random_run(seed)
    change = res.equity.iloc[-1] - res.equity.iloc[0]
    costs = res.costs[["spread", "slippage", "commission"]].to_numpy().sum()
    # 1) equity change == price pnl - costs + swap
    assert change == pytest.approx(res.pnl["price"].sum() - costs + res.costs["swap"].sum(), abs=1e-6)
    assert abs(res.reconcile()["residual"]) < 1e-6
    # 2) every dollar is attributed to exactly one trade (incl. the 'end' trade)
    assert res.trades["pnl"].sum() == pytest.approx(change, abs=1e-6)
    assert res.trades["costs"].sum() == pytest.approx(costs, abs=1e-6)
    assert res.trades["swap"].sum() == pytest.approx(res.costs["swap"].sum(), abs=1e-6)
    # 3) per-bar: equity diff == net, returns consistent with equity
    np.testing.assert_allclose(res.equity.diff().iloc[1:], res.pnl["net"].iloc[1:], atol=1e-6)
    np.testing.assert_allclose(res.returns.iloc[1:], res.equity.pct_change().iloc[1:], rtol=0, atol=1e-12)
    assert res.returns.iloc[0] == 0.0
    # 4) no trade at the open when the target equals the current position
    open_fills = res.fills[res.fills["kind"] == "open"]
    fill_bars = set(open_fills["bar"])
    for t, tgt, cur in targets:
        if XAUUSD.round_lots(tgt) == pytest.approx(cur):
            assert (t + 1) not in fill_bars
        else:
            assert (t + 1) in fill_bars
    # 5) positions equal executed lots
    signed = res.fills["side"] * res.fills["lots"]
    by_bar = signed.groupby(res.fills["bar"]).sum().reindex(range(len(bars)), fill_value=0.0)
    np.testing.assert_allclose(res.position_close.to_numpy(), by_bar.cumsum().to_numpy(), atol=1e-9)
    open_by_bar = (open_fills["side"] * open_fills["lots"]).groupby(open_fills["bar"]).sum()
    open_by_bar = open_by_bar.reindex(range(len(bars)), fill_value=0.0).to_numpy()
    expected_open = np.r_[0.0, res.position_close.to_numpy()[:-1]] + open_by_bar
    np.testing.assert_allclose(res.positions.to_numpy(), expected_open, atol=1e-9)
    # 6) costs are non-negative; fill prices on the correct side of mid
    assert (res.costs[["spread", "slippage", "commission"]] >= 0).all().all()
    buys = res.fills["side"] == 1
    assert (res.fills.loc[buys, "price"] > res.fills.loc[buys, "mid"]).all()
    assert (res.fills.loc[~buys, "price"] < res.fills.loc[~buys, "mid"]).all()


@pytest.mark.parametrize("seed", range(4))
def test_mark_to_market_independent_formula(seed: int) -> None:
    """Without stops/costs/swap, equity change = sum over bars of mid-price moves * lots."""
    rng = np.random.default_rng(100 + seed)
    bars = make_synthetic_bars(250, seed=seed)
    sim = ExecutionSimulator(bars, costs=CostModel.zero(), instrument=NO_SWAP)
    while not sim.done:
        sim.step(float(rng.choice([-1.0, 0.0, 0.5, 2.0])) if rng.random() < 0.2 else sim.position)
    res = sim.result()
    pos = res.positions.to_numpy()
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    pnl = 100 * (pos[:-1] * (o[1:] - c[:-1]) + pos[1:] * (c[1:] - o[1:]))
    np.testing.assert_allclose(res.equity.diff().iloc[1:].to_numpy(), pnl, atol=1e-8)
    assert res.costs.to_numpy().sum() == 0.0


def test_result_frames_and_metrics_present() -> None:
    _, _, res, _ = _random_run(99, n=200)
    assert {"spread", "slippage", "commission", "swap"} <= set(res.costs.columns)
    assert {"entry_time", "exit_time", "side", "lots", "entry_price", "exit_price", "pnl", "costs",
            "swap", "exit_reason", "bars_held"} <= set(res.trades.columns)
    assert res.metrics["n_trades"] == len(res.trades)
    assert res.metrics["total_costs"] == pytest.approx(res.reconcile()["costs"])
    assert set(res.trades["exit_reason"]) <= {"signal", "stop", "take_profit", "risk", "end"}


# ---- reviewer: adversarial regressions ---------------------------------------------------------------
def test_stop_distance_is_anchored_at_the_entry_open() -> None:
    """A stop given as a distance is placed at ENTRY (mid open of the fill bar) -/+ d, so a gap
    between the decision close and the fill cannot put it on the wrong side of the market."""
    rows = [(2000, 2001, 1999, 2000), (1990, 1992, 1987, 1991), (1991, 1993, 1984, 1985)]
    bars = hand_bars(rows)
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    r = sim.step(1.0, stop_distance=5.0)
    assert r.exit_reason is None and r.position == 1.0
    assert r.stop_price == pytest.approx(1985.0)           # 1990 (entry open) - 5
    assert r.take_profit is None
    assert len(r.fills) == 1                               # entered once, not entered+stopped
    r2 = sim.step(1.0, stop_price=r.stop_price)            # level kept fixed afterwards
    assert r2.exit_reason == "stop" and r2.stop_price == pytest.approx(1985.0)
    assert r2.fills[0].price == pytest.approx(1985.0 - 0.15 - 0.02)
    # contrast: the same distance anchored at the decision close (2000 - 5 = 1995) sits above
    # the 1990 open -> the position is entered and immediately stopped at the open
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    bad = sim.step(1.0, stop_price=1995.0)
    assert bad.exit_reason == "stop" and len(bad.fills) == 2
    assert bad.costs["spread"] == pytest.approx(30.0)      # spread paid twice for nothing


def test_distance_levels_mirror_for_shorts_and_validate() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000), (2003, 2004, 1998, 1999), (1999, 2000, 1998, 1999)])
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    r = sim.step(-1.0, stop_distance=3.0, take_profit_distance=4.0)
    assert r.stop_price == pytest.approx(2006.0) and r.take_profit == pytest.approx(1999.0)
    assert r.exit_reason == "take_profit"
    assert r.fills[1].price == pytest.approx(1999.0 + 0.15)   # limit buy-back: half spread only
    sim = ExecutionSimulator(bars, costs=FIXED_SLIP)
    with pytest.raises(ValueError):
        sim.step(1.0, stop_price=1990.0, stop_distance=5.0)
    with pytest.raises(ValueError):
        sim.step(1.0, take_profit=2010.0, take_profit_distance=5.0)
    with pytest.raises(ValueError):
        sim.step(1.0, stop_distance=0.0)
    # flat after the fill: distances are ignored and no level is reported
    r0 = sim.step(0.0, stop_distance=5.0)
    assert r0.stop_price is None and r0.fills == []


def test_reset_gives_identical_replay() -> None:
    """RL-style reuse: reset() must not leak trade/peak/bankruptcy state between episodes."""
    bars = make_synthetic_bars(200, seed=21)
    sim = ExecutionSimulator(bars, initial_equity=50_000.0)
    rng = np.random.default_rng(5)
    path = rng.choice([-1.0, 0.0, 1.0, 2.0], size=len(bars) - 1)

    def run() -> pd.DataFrame:
        sim.reset()
        for k, tgt in enumerate(path):
            sim.step(float(tgt), stop_distance=3.0 if k % 7 == 0 and tgt != 0 else None)
        res = sim.result(compute_metrics=False)
        return pd.concat([res.equity, res.positions, res.pnl], axis=1)

    a, b = run(), run()
    pd.testing.assert_frame_equal(a, b)
    assert len(sim.result(compute_metrics=False).trades) > 0


def test_single_bar_and_last_bar_edge_cases() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000)])
    sim = ExecutionSimulator(bars)
    assert sim.done
    res = sim.result()
    assert len(res.equity) == 1 and res.trades.empty and res.metrics["n_bars"] == 1
    with pytest.raises(RuntimeError):
        sim.step(1.0)
    bars = hand_bars([(2000, 2001, 1999, 2000)] * 3)
    sim = ExecutionSimulator(bars)
    sim.reset(start=2)
    assert sim.done and len(sim.result(compute_metrics=False).equity) == 1
    with pytest.raises(IndexError):
        sim.reset(start=3)


def test_result_meta_carries_timeframe_for_daily_bucketing() -> None:
    bars = hand_bars([(2000, 2001, 1999, 2000)] * 3)
    sim = ExecutionSimulator(bars)
    sim.step(1.0)
    assert sim.result(compute_metrics=False).meta["timeframe"] == "H1"


def _reference_equity(bars: pd.DataFrame, steps: list[tuple[float, float | None, float | None]],
                      cm: CostModel, inst, equity0: float) -> list[float]:
    """Independent, deliberately naive re-implementation of the SPEC §1/§8 bar model with a
    brute-force rollover calendar (no vectorised night counting, no trade ledger)."""
    def nights(a: pd.Timestamp, b: pd.Timestamp) -> int:
        n, d = 0, a.normalize() - pd.Timedelta(days=1)
        while d <= b:
            r = d + pd.Timedelta(hours=inst.rollover_hour_utc)
            if a < r <= b and r.weekday() < 5:
                n += 3 if r.weekday() == inst.triple_swap_weekday else 1
            d += pd.Timedelta(days=1)
        return n

    def swap(p: float, k: int) -> float:
        return abs(p) * (inst.swap_long_per_lot if p > 0 else inst.swap_short_per_lot) * k if p else 0.0

    def fill_cost(q: float, spread: float, rng_: float, limit: bool) -> float:
        half = 0.5 * max(spread * cm.spread_multiplier, cm.min_spread)
        slip = 0.0 if limit else cm.slippage_fixed + cm.slippage_range_frac * rng_ + cm.impact_coef * np.sqrt(q)
        return (half + slip) * q * inst.contract_size + q * cm.commission_per_lot

    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = bars["spread"].to_numpy()
    t_open, t_av = list(bars.index), list(bars["available_at"])
    cs = inst.contract_size
    eq, pos, out = equity0, 0.0, [equity0]
    for t, (tgt, sl, tp) in enumerate(steps):
        i = t + 1
        eq += pos * cs * (o[i] - c[t]) + swap(pos, nights(t_av[t], t_open[i]))
        tgt = inst.round_lots(tgt)
        if abs(tgt - pos) > 1e-9:
            eq -= fill_cost(abs(tgt - pos), spr[i], h[i] - lo[i], False)
            pos = tgt
        px = o[i]
        if pos != 0 and (sl is not None or tp is not None):
            s = 1 if pos > 0 else -1
            hit = None
            if sl is not None and s * (o[i] - sl) <= 0:
                hit = (o[i], False)
            elif tp is not None and s * (o[i] - tp) >= 0:
                hit = (o[i], True)
            elif sl is not None and ((s > 0 and lo[i] <= sl) or (s < 0 and h[i] >= sl)):
                hit = (sl, False)
            elif tp is not None and ((s > 0 and h[i] >= tp) or (s < 0 and lo[i] <= tp)):
                hit = (tp, True)
            if hit is not None:
                eq += pos * cs * (hit[0] - o[i]) - fill_cost(abs(pos), spr[i], h[i] - lo[i], hit[1])
                pos, px = 0.0, hit[0]
        eq += pos * cs * (c[i] - px) + swap(pos, nights(t_open[i], t_av[i]))
        out.append(eq)
    return out


@pytest.mark.parametrize("seed", range(3))
def test_equity_matches_independent_reference_model(seed: int) -> None:
    rng = np.random.default_rng(500 + seed)
    bars = make_synthetic_bars(24 * 9, seed=seed, spread=0.4, start="2024-01-09 15:00")  # 2 weekends
    inst = dataclasses.replace(XAUUSD, swap_long_per_lot=-40.0, swap_short_per_lot=12.0)
    cm = CostModel(spread_multiplier=1.1, min_spread=0.2, slippage_fixed=0.03,
                   slippage_range_frac=0.04, impact_coef=0.05, commission_per_lot=3.5, financing=FIXED)
    steps: list[tuple[float, float | None, float | None]] = []
    sim = ExecutionSimulator(bars, instrument=inst, costs=cm, initial_equity=150_000.0)
    tgt = 0.0
    while not sim.done:
        if rng.random() < 0.25:
            tgt = float(rng.choice([-1.5, -0.7, 0.0, 0.4, 1.0, 2.3]))
        side = np.sign(tgt) or np.sign(sim.position)
        sl = tp = None
        if side and rng.random() < 0.5:
            sl = float(sim.price - side * rng.uniform(0.3, 4.0))
            tp = float(sim.price + side * rng.uniform(0.3, 4.0)) if rng.random() < 0.5 else None
        steps.append((tgt, sl, tp))
        sim.step(tgt, stop_price=sl, take_profit=tp)
    ref = _reference_equity(bars, steps, cm, inst, 150_000.0)
    res = sim.result(compute_metrics=False)
    np.testing.assert_allclose(res.equity.to_numpy(), np.asarray(ref), rtol=0, atol=1e-6)
    assert res.costs["swap"].abs().sum() > 0 and (res.fills["kind"] != "open").any()
