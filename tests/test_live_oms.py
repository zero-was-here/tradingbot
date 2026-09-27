"""OrderManager: planning (netting/hedging), idempotency across restarts, retries, rejects."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import Side
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import CostModel
from aurum.live.broker import (
    BrokerPosition,
    OrderRequest,
    OrderResult,
    OrderStatusCode,
    SimulatedClock,
    net_lots,
)
from aurum.live.oms import OrderManager, make_client_id
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.state import StateCorruptError

MAGIC = 4242
DELAY = pd.Timedelta(seconds=5)


def _paper(n: int = 120, *, hedging: bool = False, gaps: bool = False, start: int = 10,
           equity: float = 100_000.0) -> tuple[PaperBroker, SimulatedClock, pd.DataFrame]:
    bars = make_synthetic_bars(n, "H1", seed=5, weekend_gaps=gaps)
    clock = SimulatedClock(bars["available_at"].iloc[start] + DELAY)
    pb = PaperBroker(ReplayFeed(bars), costs=CostModel(), initial_equity=equity, clock=clock, hedging=hedging)
    return pb, clock, bars


class Scripted:
    """Broker wrapper: pops scripted behaviours for place_order, delegates the rest.

    Behaviours: "pass" (delegate), "requote", "no_money", "unknown" (not executed),
    "unknown_exec" (executed, outcome lost), "crash" (KeyboardInterrupt before sending),
    "crash_after" (executed, then KeyboardInterrupt).
    """

    def __init__(self, inner: PaperBroker, script: list[str] | None = None) -> None:
        self.inner = inner
        self.script: deque[str] = deque(script or [])
        self.sent: list[OrderRequest] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def place_order(self, order: OrderRequest) -> OrderResult:
        mode = self.script.popleft() if self.script else "pass"
        self.sent.append(order)
        if mode == "pass":
            return self.inner.place_order(order)
        if mode == "requote":
            return OrderResult(ok=False, retcode=10004, message="requote", status=OrderStatusCode.REQUOTE,
                               client_id=order.client_id)
        if mode == "no_money":
            return OrderResult(ok=False, retcode=10019, message="no money", status=OrderStatusCode.NO_MONEY,
                               client_id=order.client_id)
        if mode == "unknown":
            return OrderResult(ok=False, retcode=10012, message="timeout", status=OrderStatusCode.UNKNOWN,
                               client_id=order.client_id)
        if mode == "unknown_exec":
            self.inner.place_order(order)
            return OrderResult(ok=False, retcode=10012, message="timeout", status=OrderStatusCode.UNKNOWN,
                               client_id=order.client_id)
        if mode == "crash":
            raise KeyboardInterrupt("process killed before send")
        if mode == "crash_after":
            self.inner.place_order(order)
            raise KeyboardInterrupt("process killed after send")
        raise AssertionError(mode)


def _pos(ticket: int, lots: float, minutes: int, magic: int = MAGIC) -> BrokerPosition:
    return BrokerPosition(ticket=ticket, symbol="XAUUSD", lots=lots, price_open=1800.0, sl=None, tp=None,
                          magic=magic, comment="", time=pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(minutes=minutes))


def _deals(pb: PaperBroker) -> int:
    return len(pb.deals(magic=MAGIC))


# ------------------------------------------------------------------------------------------------
# planning
# ------------------------------------------------------------------------------------------------
def test_plan_netting() -> None:
    pb, _, _ = _paper()
    oms = OrderManager(pb, magic=MAGIC, hedging=False, max_order_lots=3.0)
    legs = oms.plan(-2.0, [_pos(1, 1.0, 0)], sl_distance=4.0)
    assert [(lg.kind, int(lg.side), lg.lots, lg.ticket) for lg in legs] == [("close", -1, 1.0, 1), ("open", -1, 2.0, None)]
    assert legs[1].sl_distance == 4.0 and legs[0].sl_distance is None
    assert [(lg.kind, lg.lots) for lg in oms.plan(3.0, [_pos(1, 1.0, 0)])] == [("open", 2.0)]
    assert [(lg.kind, lg.lots, lg.ticket) for lg in oms.plan(0.4, [_pos(1, 1.0, 0)])] == [("close", 0.6, 1)]
    assert oms.plan(1.004, [_pos(1, 1.0, 0)]) == []  # below the lot grid
    assert [lg.lots for lg in oms.plan(7.0, [])] == [3.0, 3.0, 1.0]  # max_order_lots split
    assert oms.plan(0.0, []) == []


def test_plan_hedging_closes_opposite_first_then_fifo() -> None:
    pb, _, _ = _paper()
    oms = OrderManager(pb, magic=MAGIC, hedging=True)
    book = [_pos(11, 1.0, 0), _pos(12, 0.5, 10), _pos(13, -0.3, 20)]
    legs = oms.plan(1.0, book)
    assert [(lg.kind, int(lg.side), lg.lots, lg.ticket) for lg in legs] == [
        ("close", 1, 0.3, 13), ("close", -1, 0.5, 11)]
    legs = oms.plan(-1.0, book)
    assert [(lg.kind, int(lg.side), lg.lots, lg.ticket) for lg in legs] == [
        ("close", -1, 1.0, 11), ("close", -1, 0.5, 12), ("open", -1, 0.7, None)]
    legs = oms.plan(0.0, book)
    assert sorted(lg.ticket for lg in legs) == [11, 12, 13] and all(lg.kind == "close" for lg in legs)


def test_client_id_format() -> None:
    t = pd.Timestamp("2026-09-25 14:00", tz="UTC")
    cid = make_client_id(20260926, t, 3)
    assert cid == "20260926-202609251400-3" and len(cid) <= 31


# ------------------------------------------------------------------------------------------------
# execution & idempotency
# ------------------------------------------------------------------------------------------------
def test_reconcile_and_idempotency_across_restart(tmp_path: Path) -> None:
    pb, clock, bars = _paper()
    state = tmp_path / "oms.json"
    t = bars["available_at"].iloc[10]
    rep = OrderManager(pb, magic=MAGIC, state_path=state).reconcile(1.5, t, sl_distance=6.0)
    assert rep.status == "filled" and rep.actual_after == pytest.approx(1.5) and len(rep.fills) == 1
    assert pb.positions("XAUUSD", MAGIC)[0].sl is not None
    n = _deals(pb)
    # restart: a new OMS with the same state file never re-sends the same bar...
    oms2 = OrderManager(pb, magic=MAGIC, state_path=state)
    assert oms2.reconcile(1.5, t).status == "duplicate"
    # ... even if the position changed meanwhile (e.g. a stop fired): no re-entry on that bar
    pb.close_all("XAUUSD", MAGIC)
    n = _deals(pb)
    rep = oms2.reconcile(1.5, t)
    assert rep.status == "duplicate" and _deals(pb) == n
    # the next bar trades normally
    clock.advance_to(bars["available_at"].iloc[11] + DELAY)
    rep = oms2.reconcile(-1.0, bars["available_at"].iloc[11])
    assert rep.status == "filled" and net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(-1.0)
    assert all(cid.startswith(f"{MAGIC}-") for cid in oms2.sent_client_ids())


@pytest.mark.parametrize("mode", ["crash", "crash_after"])
def test_crash_mid_reconcile_never_double_sends(tmp_path: Path, mode: str) -> None:
    pb, clock, bars = _paper()
    state = tmp_path / "oms.json"
    t = bars["available_at"].iloc[10]
    pb.place_order(OrderRequest("seed", "XAUUSD", Side.BUY, 1.0, clock.now(), MAGIC))
    wrapped = Scripted(pb, ["pass", mode])  # leg 0 (close long) ok, leg 1 (open short) crashes
    with pytest.raises(KeyboardInterrupt):
        OrderManager(wrapped, magic=MAGIC, state_path=state).reconcile(-2.0, t)
    before = _deals(pb)
    oms = OrderManager(pb, magic=MAGIC, state_path=state)  # restart
    rep = oms.reconcile(-2.0, t)
    assert net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(-2.0)
    if mode == "crash_after":
        assert rep.status == "noop" and _deals(pb) == before  # the lost send was found in the deals
    else:
        assert rep.status == "filled" and _deals(pb) == before + 1
    ids = oms.sent_client_ids()
    assert len(ids) == len(set(ids))
    assert oms.reconcile(-2.0, t).status == "duplicate"


def test_unknown_outcome_is_verified_not_resent() -> None:
    pb, _, bars = _paper()
    wrapped = Scripted(pb, ["unknown_exec"])
    oms = OrderManager(wrapped, magic=MAGIC, sleep=lambda s: None)
    rep = oms.reconcile(1.0, bars["available_at"].iloc[10])
    assert rep.status == "filled" and rep.legs[0].attempts == 1
    assert _deals(pb) == 1 and len(wrapped.sent) == 1


def test_unknown_not_executed_is_retried_once() -> None:
    pb, _, bars = _paper()
    sleeps: list[float] = []
    wrapped = Scripted(pb, ["unknown", "requote", "pass"])
    oms = OrderManager(wrapped, magic=MAGIC, sleep=sleeps.append, backoff_seconds=0.5)
    rep = oms.reconcile(1.0, bars["available_at"].iloc[10])
    assert rep.status == "filled" and rep.legs[0].attempts == 3
    assert sleeps == [0.5, 1.0] and _deals(pb) == 1
    assert len({o.client_id for o in wrapped.sent}) == 1  # retries reuse the client id


def test_retries_exhausted_and_rejects(tmp_path: Path) -> None:
    pb, _, bars = _paper()
    t = bars["available_at"].iloc[10]
    wrapped = Scripted(pb, ["requote"] * 10)
    oms = OrderManager(wrapped, magic=MAGIC, max_retries=2, sleep=lambda s: None, state_path=tmp_path / "s.json")
    rep = oms.reconcile(1.0, t)
    assert rep.status == "rejected" and rep.legs[0].attempts == 3 and _deals(pb) == 0
    assert oms.decision_status(t) == "in_progress"  # transient: may be resumed for this bar
    wrapped.script = deque(["no_money"])
    rep = oms.reconcile(1.0, t)
    assert rep.status == "rejected" and rep.legs[0].attempts == 1
    assert oms.decision_status(t) == "rejected"
    assert oms.reconcile(1.0, t).status == "duplicate"


def test_market_closed_defers_then_resumes(tmp_path: Path) -> None:
    pb, clock, bars = _paper(200, gaps=True)
    avail = pd.DatetimeIndex(bars["available_at"])
    gap = int(np.flatnonzero(bars.index[1:] > avail[:-1])[0])
    clock.advance_to(avail[gap] + DELAY)
    oms = OrderManager(pb, magic=MAGIC, state_path=tmp_path / "s.json")
    rep = oms.reconcile(1.0, avail[gap])
    assert rep.status == "deferred" and _deals(pb) == 0
    clock.advance_to(bars.index[gap + 1] + pd.Timedelta(minutes=1))
    rep = oms.reconcile(1.0, avail[gap])
    assert rep.status == "filled" and oms.decision_status(avail[gap]) == "complete"
    assert rep.fills[0].price > bars["open"].iloc[gap + 1]


def test_stale_and_superseded_decisions() -> None:
    pb, clock, bars = _paper(200, gaps=True)
    avail = pd.DatetimeIndex(bars["available_at"])
    gap = int(np.flatnonzero(bars.index[1:] > avail[:-1])[0])
    clock.advance_to(avail[gap] + DELAY)
    oms = OrderManager(pb, magic=MAGIC)
    assert oms.reconcile(1.0, avail[gap]).status == "deferred"
    clock.advance_to(avail[gap + 1] + DELAY)
    assert oms.reconcile(0.5, avail[gap + 1]).status == "filled"
    assert oms.decision_status(avail[gap]) == "superseded"
    assert oms.reconcile(1.0, avail[gap]).status == "duplicate"
    assert oms.reconcile(1.0, avail[gap - 3]).status == "stale"


def test_netting_foreign_position_conflict_and_hedging_isolation() -> None:
    pb, clock, bars = _paper()
    pb.place_order(OrderRequest("other-ea", "XAUUSD", Side.SELL, 0.7, clock.now(), 999))
    oms = OrderManager(pb, magic=MAGIC)
    rep = oms.reconcile(1.0, bars["available_at"].iloc[10])
    assert rep.status == "conflict" and "foreign" in rep.errors[0]
    assert pb.positions()[0].magic == 999 and len(pb.positions()) == 1
    # hedging account: the other EA's ticket is simply not ours, never touched
    hb, hclock, hbars = _paper(hedging=True)
    hb.place_order(OrderRequest("other-ea", "XAUUSD", Side.SELL, 0.7, hclock.now(), 999))
    oms = OrderManager(hb, magic=MAGIC)
    rep = oms.reconcile(1.0, hbars["available_at"].iloc[10])
    assert rep.status == "filled"
    rep = oms.reconcile(0.0, hbars["available_at"].iloc[11])
    assert rep.status == "filled"
    left = hb.positions()
    assert len(left) == 1 and left[0].magic == 999 and left[0].lots == pytest.approx(-0.7)


def test_large_orders_split_by_max_lot() -> None:
    pb, _, bars = _paper(equity=50_000_000.0)
    oms = OrderManager(pb, magic=MAGIC)
    rep = oms.reconcile(120.0, bars["available_at"].iloc[10])
    assert [lg.lots for lg in rep.planned] == [50.0, 50.0, 20.0]
    assert rep.status == "filled" and net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(120.0)


def test_dry_run_sends_nothing(tmp_path: Path) -> None:
    pb, _, bars = _paper()
    oms = OrderManager(pb, magic=MAGIC, dry_run=True, state_path=tmp_path / "s.json")
    rep = oms.reconcile(2.0, bars["available_at"].iloc[10], sl_distance=3.0)
    assert rep.status == "dry_run" and rep.dry_run and [lg.lots for lg in rep.planned] == [2.0]
    assert _deals(pb) == 0 and not (tmp_path / "s.json").exists()
    assert rep.to_dict()["planned"][0]["sl_distance"] == 3.0


def test_corrupt_or_foreign_state_refused(tmp_path: Path) -> None:
    pb, _, bars = _paper()
    path = tmp_path / "s.json"
    path.write_text("{oops")
    with pytest.raises(StateCorruptError):
        OrderManager(pb, magic=MAGIC, state_path=path)
    path.unlink()
    OrderManager(pb, magic=MAGIC, state_path=path).reconcile(1.0, bars["available_at"].iloc[10])
    with pytest.raises(StateCorruptError, match="magic"):
        OrderManager(pb, magic=MAGIC + 1, state_path=path)


# ------------------------------------------------------------------------------------------------
# adversarial review: unknown outcomes, NaN targets, flatten, client-id length
# ------------------------------------------------------------------------------------------------
class LinkDrop:
    """The order reaches the venue, then the link drops: ``place_order`` raises and the next
    deal/position lookups fail too. A resend in that state would double the position."""

    def __init__(self, inner: PaperBroker, *, lookups_down: int = 1) -> None:
        self.inner = inner
        self.sent = 0
        self.lookups_down = 0
        self._down_after_send = lookups_down

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def place_order(self, order: OrderRequest) -> OrderResult:
        from aurum.live.broker import BrokerError

        self.sent += 1
        if self.sent == 1:
            self.inner.place_order(order)
            self.lookups_down = self._down_after_send
            raise BrokerError("terminal disconnected after send")
        return self.inner.place_order(order)

    def find_deals(self, *a: Any, **k: Any) -> list:
        from aurum.live.broker import BrokerError

        if self.lookups_down > 0:
            raise BrokerError("history unavailable")
        return self.inner.find_deals(*a, **k)

    def positions(self, *a: Any, **k: Any) -> list[BrokerPosition]:
        from aurum.live.broker import BrokerError

        if self.lookups_down > 0:
            self.lookups_down -= 1
            raise BrokerError("positions unavailable")
        return self.inner.positions(*a, **k)


def test_unverifiable_unknown_outcome_is_never_resent(tmp_path: Path) -> None:
    pb, clock, bars = _paper()
    link = LinkDrop(pb)
    t = bars["available_at"].iloc[10]
    oms = OrderManager(link, magic=MAGIC, sleep=lambda s: None, state_path=tmp_path / "s.json")
    rep = oms.reconcile(1.0, t)
    assert link.sent == 1, "an order whose outcome could not be verified must not be resent"
    assert net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(1.0)  # not 2.0
    assert rep.status == "unknown" and rep.legs[0].status == "unknown"
    assert oms.decision_status(t) == "in_progress"
    # the link is back: resuming the same decision re-plans from the venue position -> nothing to do
    rep = oms.reconcile(1.0, t)
    assert rep.status == "noop" and link.sent == 1 and _deals(pb) == 1


def test_partially_executed_unknown_outcome_is_not_resent_in_full() -> None:
    pb, _, bars = _paper()

    class HalfFill:
        def __init__(self, inner: PaperBroker) -> None:
            self.inner = inner
            self.sent = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def place_order(self, order: OrderRequest) -> OrderResult:
            self.sent += 1
            if self.sent == 1:  # half the volume executes under another deal id, then a timeout
                self.inner.place_order(OrderRequest("venue-split", order.symbol, order.side, order.lots / 2,
                                                    order.time, order.magic))
                return OrderResult(ok=False, retcode=10012, message="timeout", status=OrderStatusCode.UNKNOWN,
                                   client_id=order.client_id)
            return self.inner.place_order(order)

    hf = HalfFill(pb)
    rep = OrderManager(hf, magic=MAGIC, sleep=lambda s: None).reconcile(1.0, bars["available_at"].iloc[10])
    assert hf.sent == 1 and net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(0.5)  # never 1.5
    assert rep.status == "partial" and rep.legs[0].status == "partial"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_target_is_refused_not_flattened(bad: float) -> None:
    pb, clock, bars = _paper()
    oms = OrderManager(pb, magic=MAGIC)
    oms.reconcile(1.0, bars["available_at"].iloc[10])
    clock.advance_to(bars["available_at"].iloc[11] + DELAY)
    with pytest.raises(ValueError, match="finite"):
        oms.reconcile(bad, bars["available_at"].iloc[11])
    assert net_lots(pb.positions("XAUUSD", MAGIC)) == pytest.approx(1.0)


def test_flatten_is_not_blocked_by_decision_idempotency(tmp_path: Path) -> None:
    pb, clock, bars = _paper()
    state = tmp_path / "s.json"
    t = bars["available_at"].iloc[10]
    oms = OrderManager(pb, magic=MAGIC, state_path=state)
    assert oms.reconcile(1.0, t).status == "filled"
    # shutdown within the same minute as the decision: the key collides with a COMPLETE record
    rep = OrderManager(pb, magic=MAGIC, state_path=state).flatten(t + pd.Timedelta(seconds=30))
    assert rep.status == "filled" and net_lots(pb.positions("XAUUSD", MAGIC)) == 0.0
    # a later bar, then a flatten stamped EARLIER than it (clock skew): still flattens
    clock.advance_to(bars["available_at"].iloc[11] + DELAY)
    oms = OrderManager(pb, magic=MAGIC, state_path=state)
    assert oms.reconcile(-2.0, bars["available_at"].iloc[11]).status == "filled"
    rep = oms.flatten(t)
    assert rep.status == "filled" and net_lots(pb.positions("XAUUSD", MAGIC)) == 0.0
    assert oms.flatten(t).status == "noop"  # idempotent: nothing left to close
    ids = oms.sent_client_ids()
    assert len(ids) == len(set(ids)) == _deals(pb)  # every leg has its own client id
    # a normal bar decision for an already-completed bar is still a duplicate
    assert oms.reconcile(1.0, bars["available_at"].iloc[11]).status == "duplicate"


def test_magic_that_would_truncate_client_ids_is_refused() -> None:
    pb, _, _ = _paper()
    with pytest.raises(ValueError, match="too long"):
        OrderManager(pb, magic=123_456_789_012_345)
    with pytest.raises(ValueError, match="positive"):
        OrderManager(pb, magic=0)
    OrderManager(pb, magic=2**31 - 1)  # the largest magic LiveConfig accepts fits
    assert len(make_client_id(2**31 - 1, pd.Timestamp("2099-12-31 23:59", tz="UTC"), 9999)) <= 31
