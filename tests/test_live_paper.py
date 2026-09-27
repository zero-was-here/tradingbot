"""PaperBroker / ReplayFeed: fills must match the ExecutionSimulator exactly (SPEC §0.2)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.core.instrument import XAUUSD
from aurum.core.types import Side
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import CostModel, FinancingModel
from aurum.execution.simulator import ExecutionSimulator
from aurum.live.broker import Broker, OrderRequest, OrderStatusCode, SimulatedClock, net_lots
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.state import StateCorruptError

DELAY = pd.Timedelta(seconds=5)
MAGIC = 77


def _bars(n: int = 300, *, seed: int = 3, gaps: bool = False) -> pd.DataFrame:
    return make_synthetic_bars(n, "H1", seed=seed, weekend_gaps=gaps)


def _broker(bars: pd.DataFrame, *, costs: CostModel | None = None, hedging: bool = False,
            start: int = 0, state_path: Path | None = None) -> tuple[PaperBroker, SimulatedClock, ReplayFeed]:
    feed = ReplayFeed(bars)
    clock = SimulatedClock(bars["available_at"].iloc[start] + DELAY)
    return PaperBroker(feed, costs=costs, clock=clock, hedging=hedging, state_path=state_path), clock, feed


def _order(cid: str, side: int, lots: float, clock: SimulatedClock, **kw) -> OrderRequest:
    return OrderRequest(client_id=cid, symbol="XAUUSD", side=Side(side), lots=lots, time=clock.now(), magic=MAGIC,
                        **kw)


def _reconcile_netting(pb: PaperBroker, clock: SimulatedClock, target: float, cur: float, tag: str,
                       sl_distance: float | None) -> None:
    """Minimal OMS: close then open on sign change; adds/reductions otherwise."""
    d = round(target - cur, 8)
    if abs(d) < 1e-9:
        return
    if cur != 0 and (target == 0 or np.sign(target) != np.sign(cur)):
        pos = pb.positions("XAUUSD", MAGIC)[0]
        r = pb.place_order(_order(f"c{tag}", -int(np.sign(cur)), abs(cur), clock, position_ticket=pos.ticket))
        assert r.ok, r
        if target != 0:
            r = pb.place_order(_order(f"o{tag}", int(np.sign(target)), abs(target), clock, sl_distance=sl_distance))
            assert r.ok, r
    else:
        r = pb.place_order(_order(f"o{tag}", int(np.sign(d)), abs(d), clock,
                                  sl_distance=sl_distance if cur == 0 else None))
        assert r.ok, r


@pytest.mark.parametrize("stop", [None, 2.5])
def test_paper_equals_simulator(stop: float | None) -> None:
    """Same targets, same bars: identical equity at every decision, identical fill prices."""
    bars = _bars(400)
    rng = np.random.default_rng(11)
    targets = rng.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], size=len(bars))
    costs = CostModel(commission_per_lot=3.5, spread_multiplier=1.2, slippage_range_frac=0.05)
    sim = ExecutionSimulator(bars, costs=costs)
    pb, clock, _ = _broker(bars, costs=costs)
    sim_stop = None
    max_diff = 0.0
    for t in range(len(bars) - 1):
        clock.advance_to(bars["available_at"].iloc[t] + DELAY)
        max_diff = max(max_diff, abs(pb.account().equity - sim.equity))
        cur = net_lots(pb.positions("XAUUSD", MAGIC))
        assert cur == pytest.approx(sim.position, abs=1e-9)
        tgt = float(targets[t])
        entry = tgt != 0 and np.sign(tgt) != np.sign(cur)
        sd = stop if (stop is not None and entry) else None
        _reconcile_netting(pb, clock, tgt, cur, str(t), sd)
        keep = sim_stop if (stop is not None and not entry and tgt != 0) else None
        res = sim.step(tgt, stop_distance=sd, stop_price=keep)
        sim_stop = res.stop_price
    clock.advance_to(bars["available_at"].iloc[-1] + DELAY)
    assert abs(pb.account().equity - sim.equity) < 1e-6
    assert max_diff < 1e-6
    # every simulator fill price appears among the paper deals (same prices, same costs)
    sim_prices = np.sort(sim.fills_frame()["price"].to_numpy())
    paper_prices = np.sort(pb.deals_frame()["price"].to_numpy())
    assert np.isin(np.round(sim_prices, 9), np.round(paper_prices, 9)).all()
    if stop is not None:
        assert (sim.result(compute_metrics=False).trades["exit_reason"] == "stop").sum() > 5
        assert pb.deals_frame()["comment"].str.startswith("[sl]").sum() > 5


def test_paper_swap_matches_simulator_over_rollovers() -> None:
    """Hold a position over many rollovers (incl. the triple-swap Wednesday)."""
    bars = _bars(24 * 9)
    # no trading frictions, per-lot swaps (CostModel.zero() also switches financing off)
    costs = dataclasses.replace(CostModel.zero(), financing=FinancingModel.fixed())
    sim = ExecutionSimulator(bars, costs=costs)
    pb, clock, _ = _broker(bars, costs=costs)
    for t in range(len(bars) - 1):
        clock.advance_to(bars["available_at"].iloc[t] + DELAY)
        if t == 0:
            pb.place_order(_order("open", 1, 2.0, clock))
        sim.step(2.0)
    clock.advance_to(bars["available_at"].iloc[-1] + DELAY)
    swap_sim = sim.result(compute_metrics=False).costs["swap"].sum()
    pos = pb.positions("XAUUSD", MAGIC)[0]
    assert swap_sim < 0 and pos.swap == pytest.approx(swap_sim, rel=1e-12)
    assert pb.account().equity == pytest.approx(sim.equity, abs=1e-6)


def test_replay_feed_serves_closed_bars_only() -> None:
    bars = _bars(50, gaps=True)
    feed = ReplayFeed(bars)
    t = bars["available_at"].iloc[9]
    closed = feed.closed_bars(t, 5)
    assert len(closed) == 5 and closed.index[-1] == bars.index[9]
    assert (closed["available_at"] <= t).all()
    # one nanosecond earlier the 10th bar is still forming
    assert feed.closed_bars(t - pd.Timedelta(1, "ns")).index[-1] == bars.index[8]
    q = feed.quote(t + DELAY)
    assert q is not None and q.mid == pytest.approx(bars["open"].iloc[10])
    assert q.spread == pytest.approx(bars["spread"].iloc[10])
    assert feed.mark(t + DELAY) == pytest.approx(bars["close"].iloc[9])
    assert feed.end == bars["available_at"].iloc[-1]


def test_market_closed_between_sessions() -> None:
    bars = _bars(200, gaps=True)
    avail = bars["available_at"]
    gap = int(np.flatnonzero(bars.index[1:] > pd.DatetimeIndex(avail)[:-1])[0])
    pb, clock, feed = _broker(bars, start=gap)
    assert pb.quote("XAUUSD") is None
    r = pb.place_order(_order("x", 1, 1.0, clock))
    assert not r.ok and r.status == OrderStatusCode.MARKET_CLOSED and r.retcode == 10018
    assert pb.positions() == []
    clock.advance_to(bars.index[gap + 1] + pd.Timedelta(minutes=3))
    r = pb.place_order(_order("y", 1, 1.0, clock))
    assert r.ok and r.fill.price > bars["open"].iloc[gap + 1]  # buy pays spread + slippage
    assert feed.next_open_after(avail.iloc[gap]) == bars.index[gap + 1]


def test_order_validation_and_ownership() -> None:
    bars = _bars(60)
    pb, clock, _ = _broker(bars, start=5)
    assert isinstance(pb, Broker)
    r = pb.place_order(_order("bad", 1, 0.015, clock))
    assert r.status == OrderStatusCode.INVALID_VOLUME
    assert pb.place_order(_order("big", 1, 45.0, clock)).ok  # margin 1%: ~81k of 100k equity
    r = pb.place_order(_order("bigger", 1, 20.0, clock))
    assert r.status == OrderStatusCode.NO_MONEY and r.retcode == 10019
    assert pb.close_all("XAUUSD", MAGIC)[0].ok
    r = pb.place_order(_order("ok", 1, 1.0, clock))
    ticket = r.position_ticket
    foreign = OrderRequest(client_id="f", symbol="XAUUSD", side=Side.SELL, lots=1.0, time=clock.now(), magic=999,
                           position_ticket=ticket)
    r2 = pb.place_order(foreign)
    assert not r2.ok and "magic" in r2.message
    assert net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(1.0)
    r3 = pb.close_position(ticket, magic=999)
    assert not r3.ok
    assert pb.find_deals("ok", symbol="XAUUSD", magic=MAGIC)[0].lots == 1.0


def test_hedging_semantics() -> None:
    bars = _bars(60)
    pb, clock, _ = _broker(bars, hedging=True, start=5)
    a = pb.place_order(_order("a", 1, 1.0, clock)).position_ticket
    b = pb.place_order(_order("b", 1, 0.5, clock)).position_ticket
    c = pb.place_order(_order("c", -1, 0.3, clock)).position_ticket  # hedge: a separate short ticket
    assert len({a, b, c}) == 3
    pos = pb.positions("XAUUSD", MAGIC)
    assert sorted(p.lots for p in pos) == [-0.3, 0.5, 1.0]
    assert net_lots(pos) == pytest.approx(1.2)
    r = pb.place_order(_order("d", -1, 0.4, clock, position_ticket=a))  # partial close of a
    assert r.ok and r.position_ticket == a
    assert {p.ticket: p.lots for p in pb.positions()}[a] == pytest.approx(0.6)
    results = pb.close_all("XAUUSD", MAGIC)
    assert all(x.ok for x in results) and pb.positions() == []


def test_netting_reversal_and_vwap() -> None:
    bars = _bars(60)
    costs = CostModel.zero()
    pb, clock, _ = _broker(bars, costs=costs, start=5)
    pb.place_order(_order("a", 1, 1.0, clock))
    clock.advance_to(bars["available_at"].iloc[6] + DELAY)
    pb.place_order(_order("b", 1, 1.0, clock))
    pos = pb.positions()[0]
    assert pos.lots == pytest.approx(2.0)
    assert pos.price_open == pytest.approx((bars["open"].iloc[6] + bars["open"].iloc[7]) / 2)
    pb.place_order(_order("c", -1, 3.0, clock))  # reverse to -1
    assert net_lots(pb.positions()) == pytest.approx(-1.0)
    assert len(pb.positions()) == 1


def test_paper_state_persists(tmp_path: Path) -> None:
    bars = _bars(80)
    path = tmp_path / "paper.json"
    pb, clock, _ = _broker(bars, start=3, state_path=path, costs=CostModel(commission_per_lot=2.0))
    pb.place_order(_order("a", 1, 1.0, clock, sl_distance=5.0))
    clock.advance_to(bars["available_at"].iloc[20] + DELAY)
    eq = pb.account().equity
    pb2 = PaperBroker(pb.feed, costs=pb.costs, clock=clock, state_path=path)
    assert pb2.account().equity == pytest.approx(eq, abs=1e-9)
    assert [p.ticket for p in pb2.positions()] == [p.ticket for p in pb.positions()]
    assert pb2.find_deals("a", symbol="XAUUSD", magic=MAGIC)
    path.write_text("{not json")
    with pytest.raises(StateCorruptError):
        PaperBroker(pb.feed, clock=clock, state_path=path)


def test_latest_bars_timeframe_guard() -> None:
    bars = _bars(30)
    pb, clock, _ = _broker(bars, start=10)
    got = pb.latest_bars("XAUUSD", "H1", 5)
    assert len(got) == 5 and got["available_at"].iloc[-1] <= clock.now()
    with pytest.raises(Exception, match="H4"):
        pb.latest_bars("XAUUSD", "H4", 5)
    assert pb.is_demo() and not pb.is_hedging()
    assert XAUUSD.round_lots(pb.account().equity) > 0


def test_find_deals_matches_the_stored_31_char_comment() -> None:
    """Comments are stored truncated to 31 chars (MT5 semantics); the lookup must compare the
    same way or an unknown-outcome check would miss a deal that exists."""
    pb, clock, _ = _broker(_bars(50))
    cid = "x" * 40
    assert pb.place_order(_order(cid, 1, 0.1, clock)).ok
    assert len(pb.find_deals(cid, symbol="XAUUSD", magic=MAGIC)) == 1
