"""Order management: reconcile a target position with the venue, idempotently (SPEC §11).

The runner decides a *target* (signed lots) at each bar close; the OMS turns the difference
between that target and the ACTUAL venue position of this strategy (its magic number on its
symbol) into orders. Positions at the venue are the source of truth — never an internal
position counter — so a restart, a manual intervention or a broker-side stop can never make
the OMS double up.

Idempotency
-----------
Every order carries ``client_id = f"{magic}-{decision_time:%Y%m%d%H%M}-{seq}"`` (sent as
the MT5 comment). Decisions and their legs are persisted in a JSON state file with a
write-ahead record (``status="sending"``) before each send:

* a decision that completed (target reached, or rejected for a non-transient reason) is
  never executed again: calling :meth:`OrderManager.reconcile` again for the same bar —
  e.g. after a crash and restart — returns ``status="duplicate"`` without sending anything,
  even if the position has changed since (say a stop fired: re-entering on the same bar
  would be a double send);
* a decision interrupted mid-way (crash between legs, market closed, unverified outcome) is
  resumed by recomputing the delta against the venue position, with fresh ``seq`` numbers,
  so already-filled legs are not resent. Legs left in ``"sending"`` by a crash are resolved
  through the venue's deal history (``Broker.find_deals`` on the client id);
* a decision older than the newest one seen is ``"stale"`` and ignored, and a newer decision
  supersedes any unfinished older one;
* if such an in-flight leg cannot be looked up (the deal history fails), nothing new is sent
  for that decision (``status="unknown"``) until it can: the venue might not show the order
  in its positions yet, and re-planning from them could send it twice.

Ownership
---------
Only positions carrying OUR magic on OUR symbol are ever planned against. The adapter must
filter, and the OMS verifies it: a foreign position in a filtered answer (a buggy adapter)
makes the reconcile refuse (``status="conflict"``) rather than close another system's
position or double ours. The account mode comes from the venue (strictly: only a genuine
``True`` is hedging); ``hedging=True`` forced on a venue that reports netting is refused.

Netting vs hedging accounts
---------------------------
On a **netting** account the venue keeps one position per symbol, shared by every EA that
trades it; a foreign-magic position on our symbol means magic isolation is impossible, so
the OMS refuses to trade (``status="conflict"``) rather than net against another system's
book. Reversals are split into a close leg and an open leg (so protective levels attach to
the new position and a failed open leaves us flat, not reversed by half). On a **hedging**
account every entry is its own ticket; reductions close OUR tickets by opposite deals
(``position_ticket``), opposite-side tickets first, then FIFO (oldest first, compatible with
FIFO-regulated brokers). Order sizes above ``max_order_lots`` (default the instrument's
``max_lot``) are split.

Retries
-------
Requotes/throttling are retried with exponential backoff (``backoff_seconds * 2**k``,
capped). An ``unknown`` outcome (timeout, lost connection) is *verified* first — by the
client id in the deal history, then by the change of the net position — and only resent if
it PROVABLY did not execute: both lookups succeeded, no deal carries the client id and the
net position did not move. If a lookup fails (the connection is still down) or the position
moved by something other than the full leg (a partial fill, a concurrent stop-out), the
outcome is *inconclusive*: the leg is NOT resent, the report says ``status="unknown"`` and
the next reconcile re-plans from the venue position (the source of truth). Resending on an
inconclusive check is how a flaky connection turns one order into two.
Market-closed → ``status="deferred"`` (the runner retries when the market reopens);
no-money / invalid volume / invalid stops → ``"rejected"`` (logged, alerted, not retried
for that bar).

Flatten
-------
:meth:`OrderManager.flatten` is an operator/shutdown action, not a bar decision: it always
reconciles to zero, even when its minute key collides with an already-completed decision
(e.g. a shutdown in the same minute as the last bar close) or is older than it (clock skew).
Its legs continue the existing record's ``seq`` numbers, so client ids stay unique. Reducing
to zero from the venue's actual positions cannot double up exposure.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.types import (
    DEFAULT_MAGIC,
    Fill,
    Side,
)
from aurum.live.broker import (
    RETRYABLE_STATUSES,
    Broker,
    BrokerError,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    OrderStatusCode,
    net_lots,
)
from aurum.live.state import StateCorruptError, atomic_write_json, read_json, utc

logger = logging.getLogger(__name__)

__all__ = ["ExecutionReport", "ForeignPositionError", "Leg", "LegReport", "OrderManager", "make_client_id"]

_STATE_FORMAT = "aurum.live.oms"
_STATE_VERSION = 1
_EPS = 1e-9
#: decision statuses after which the same bar is never executed again
TERMINAL = frozenset({"complete", "rejected", "superseded", "conflict"})
#: MT5 order comments (our idempotency key) are truncated by the venue beyond this length.
MAX_CLIENT_ID_LEN = 31


class ForeignPositionError(BrokerError):
    """The venue adapter returned a position of another magic/symbol for OUR filtered query.

    The OMS then refuses to trade: acting on it would either close another system's position
    or (if it is ours but mislabelled) double the exposure. Subclasses :class:`BrokerError`, so
    the runner treats it as venue trouble (alert, back off, retry)."""


def _strict_true(x: Any) -> bool:
    return x is True or (isinstance(x, np.bool_) and bool(x))


def make_client_id(magic: int, decision_time: pd.Timestamp, seq: int) -> str:
    """``f"{magic}-{decision_time:%Y%m%d%H%M}-{seq}"`` (<= 31 chars for a magic of up to 10
    digits and seq < 10**7; :class:`OrderManager` refuses magics that would not fit)."""
    return f"{int(magic)}-{utc(decision_time):%Y%m%d%H%M}-{int(seq)}"


def _decision_key(decision_time: pd.Timestamp) -> str:
    return f"{utc(decision_time):%Y%m%d%H%M}"


def _sign(x: float) -> int:
    return 1 if x > _EPS else (-1 if x < -_EPS else 0)


@dataclass
class Leg:
    """One planned order. ``kind="close"`` targets ``ticket`` with an opposite deal."""

    kind: str
    side: Side
    lots: float
    ticket: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    sl_distance: float | None = None
    tp_distance: float | None = None

    @property
    def signed_lots(self) -> float:
        return float(int(self.side)) * self.lots

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "side": int(self.side), "lots": self.lots, "ticket": self.ticket,
                "stop_loss": self.stop_loss, "take_profit": self.take_profit,
                "sl_distance": self.sl_distance, "tp_distance": self.tp_distance}


@dataclass
class LegReport:
    client_id: str
    leg: Leg
    ok: bool
    status: str
    retcode: int | str | None
    message: str
    attempts: int
    fill: Fill | None = None
    ticket: int | None = None
    requested_price: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        f = self.fill
        return {
            "client_id": self.client_id, **self.leg.to_dict(), "ok": self.ok, "status": self.status,
            "retcode": self.retcode, "message": self.message, "attempts": self.attempts,
            "position_ticket": self.ticket, "requested_price": self.requested_price,
            "fill": None if f is None else {"price": f.price, "lots": f.lots, "side": int(f.side),
                                            "time": f.time, "commission": f.commission,
                                            "spread_cost": f.spread_cost, "slippage_cost": f.slippage_cost,
                                            "broker_ref": f.broker_ref},
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class ExecutionReport:
    """Result of one :meth:`OrderManager.reconcile` call.

    status: ``noop`` (already at target) | ``filled`` | ``partial`` | ``deferred`` (market
    closed) | ``rejected`` | ``unknown`` (an order's outcome could not be verified; it was
    NOT resent — inspect the venue) | ``duplicate`` (bar already executed) | ``stale`` (older
    than the newest decision) | ``conflict`` (foreign position on a netting account) |
    ``dry_run`` | ``error``.
    """

    decision_time: pd.Timestamp
    symbol: str
    magic: int
    target_lots: float
    actual_before: float
    actual_after: float
    status: str
    planned: list[Leg] = field(default_factory=list)
    legs: list[LegReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    hedging: bool = False
    dry_run: bool = False

    @property
    def fills(self) -> list[Fill]:
        return [lr.fill for lr in self.legs if lr.fill is not None]

    @property
    def ok(self) -> bool:
        return self.status in ("noop", "filled", "duplicate", "dry_run")

    @property
    def at_target(self) -> bool:
        return abs(self.actual_after - self.target_lots) < _EPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_time": self.decision_time, "symbol": self.symbol, "magic": self.magic,
            "target_lots": self.target_lots, "actual_before": self.actual_before,
            "actual_after": self.actual_after, "status": self.status, "dry_run": self.dry_run,
            "hedging": self.hedging, "planned": [leg.to_dict() for leg in self.planned],
            "legs": [lr.to_dict() for lr in self.legs], "errors": list(self.errors),
        }


class OrderManager:
    """Idempotent target-position reconciliation against a :class:`~aurum.live.broker.Broker`.

    Parameters
    ----------
    broker         : venue adapter.
    instrument     : lot grid / limits (``round_lots``, ``max_lot``).
    magic          : this strategy's magic number (position ownership).
    symbol         : venue symbol (default ``instrument.symbol``).
    state_path     : JSON file for sent client ids / decision status. ``None`` keeps state in
                     memory only (tests, dry runs) — then idempotency does not survive restarts.
    max_retries    : resend attempts for retryable / provably-unexecuted outcomes.
    backoff_seconds, backoff_max : exponential backoff between attempts.
    sleep          : injectable sleep (tests pass a no-op).
    max_order_lots : per-order size cap (default ``instrument.max_lot``); larger legs are split.
    hedging        : account mode; ``None`` asks the broker (``is_hedging()``).
    dry_run        : plan only — nothing is sent and no decision is recorded.
    keep_decisions : decisions retained in the state file (idempotency only needs the recent
                     ones: a decision older than the newest is refused as stale anyway).
    """

    def __init__(
        self,
        broker: Broker,
        instrument: Instrument = XAUUSD,
        magic: int = DEFAULT_MAGIC,
        *,
        symbol: str | None = None,
        state_path: str | Path | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        backoff_max: float = 8.0,
        sleep: Callable[[float], None] | None = None,
        max_order_lots: float | None = None,
        hedging: bool | None = None,
        dry_run: bool = False,
        keep_decisions: int = 200,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if int(keep_decisions) < 1:
            # the newest decision record IS the idempotency key: pruning it would let a
            # restart re-execute the bar and would drop the write-ahead record mid-send
            raise ValueError("keep_decisions must be >= 1")
        if hedging is not None and not isinstance(hedging, bool):
            raise ValueError(f"hedging must be True, False or None (ask the venue), got {hedging!r}")
        if int(magic) <= 0:
            raise ValueError("magic must be a positive integer")
        longest = make_client_id(int(magic), pd.Timestamp("2099-12-31 23:59", tz="UTC"), 9_999)
        if len(longest) > MAX_CLIENT_ID_LEN:
            # A truncated comment would make legs of one decision indistinguishable in the
            # venue's deal history, breaking outcome verification and crash recovery.
            raise ValueError(f"magic {magic} too long: client ids like {longest!r} exceed the venue's "
                             f"{MAX_CLIENT_ID_LEN}-character order comment")
        self.broker = broker
        self.instrument = instrument
        self.magic = int(magic)
        self.symbol = symbol or instrument.symbol
        self.state_path = Path(state_path) if state_path is not None else None
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self.backoff_max = float(backoff_max)
        self._sleep = sleep if sleep is not None else time.sleep
        cap = instrument.max_lot if max_order_lots is None else min(float(max_order_lots), instrument.max_lot)
        if not cap >= instrument.min_lot:
            raise ValueError("max_order_lots must be >= instrument.min_lot")
        self.max_order_lots = cap
        self._hedging = hedging
        self._hedging_arg = hedging
        if not isinstance(dry_run, bool):
            raise ValueError(f"dry_run must be a bool, got {dry_run!r}")
        self.dry_run = dry_run
        self.keep_decisions = int(keep_decisions)
        self._state: dict[str, Any] = {"format": _STATE_FORMAT, "version": _STATE_VERSION,
                                       "symbol": self.symbol, "magic": self.magic, "decisions": {}}
        if self.state_path is not None and self.state_path.exists():
            self._load()

    # ------------------------------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------------------------------
    def _load(self) -> None:
        assert self.state_path is not None
        data = read_json(self.state_path)  # raises StateCorruptError on garbage
        if not isinstance(data, dict) or data.get("format") != _STATE_FORMAT:
            raise StateCorruptError(f"{self.state_path} is not an OMS state file")
        if int(data.get("version", -1)) != _STATE_VERSION:
            raise StateCorruptError(f"unsupported OMS state version {data.get('version')!r}")
        if data.get("magic") != self.magic or data.get("symbol") != self.symbol:
            raise StateCorruptError(
                f"OMS state {self.state_path} belongs to magic {data.get('magic')} / {data.get('symbol')}, "
                f"not {self.magic} / {self.symbol}: refusing to share it")
        if not isinstance(data.get("decisions"), dict):
            raise StateCorruptError("OMS state lacks a 'decisions' mapping")
        self._state = data
        logger.info("OMS state restored from %s (%d decisions)", self.state_path, len(data["decisions"]))

    def _persist(self) -> None:
        dec = self._state["decisions"]
        if len(dec) > self.keep_decisions:
            for k in sorted(dec)[: len(dec) - self.keep_decisions]:
                del dec[k]
        if self.state_path is not None:
            atomic_write_json(self.state_path, self._state, indent=None, convert=False)

    @property
    def hedging(self) -> bool:
        if self._hedging is None:
            # strict: anything but a genuine True (e.g. "False", 1, None) is a NETTING account,
            # where the foreign-position check below protects other EAs
            self._hedging = _strict_true(self.broker.is_hedging())
        return self._hedging

    def _check_account_mode(self) -> str | None:
        """An explicit ``hedging=True`` on a venue that is NOT hedging would skip the netting
        conflict check, so our deals would net against other systems' positions."""
        if self._hedging_arg is not True:
            return None
        try:
            venue = _strict_true(self.broker.is_hedging())
        except Exception as exc:  # noqa: BLE001 - venue specific
            return f"cannot confirm the venue's account mode ({type(exc).__name__}: {exc}); refusing hedging=True"
        if not venue:
            return ("OrderManager(hedging=True) but the venue account is NETTING: our orders would net against "
                    "other magics' positions; refusing to trade")
        return None

    def _own_positions(self) -> list[BrokerPosition]:
        """This strategy's positions (symbol AND magic), verified: the adapter must filter, and
        a foreign position in the answer is refused rather than traded against."""
        pos = self.broker.positions(self.symbol, self.magic)
        bad = [p for p in pos if p.magic != self.magic or p.symbol != self.symbol]
        if bad:
            raise ForeignPositionError(
                f"venue returned {len(bad)} position(s) not owned by {self.symbol}/magic {self.magic} "
                f"(tickets {[p.ticket for p in bad]}, magics {sorted({p.magic for p in bad})}) for a filtered "
                "query: refusing to trade on an unreliable position list")
        return pos

    def decision_status(self, decision_time: pd.Timestamp) -> str | None:
        rec = self._state["decisions"].get(_decision_key(decision_time))
        return None if rec is None else rec.get("status")

    def sent_client_ids(self) -> list[str]:
        """All client ids with a recorded send attempt (for audits and tests)."""
        return [cid for rec in self._state["decisions"].values() for cid in rec.get("legs", {})]

    def latest_decision(self) -> str | None:
        dec = self._state["decisions"]
        return max(dec) if dec else None

    # ------------------------------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------------------------------
    def _grid(self, lots: float) -> float:
        """Signed lots rounded TOWARD zero on the lot grid, WITHOUT the ``max_lot`` clip of
        ``Instrument.round_lots``: position caps belong to the risk manager; per-order venue
        limits are handled by splitting (``max_order_lots``)."""
        x = float(lots)
        if not math.isfinite(x) or x == 0.0:
            return 0.0
        inst = self.instrument
        mag = round(math.floor(abs(x) / inst.lot_step + 1e-9) * inst.lot_step, 8)
        return 0.0 if mag < inst.min_lot - 1e-12 else math.copysign(mag, x)

    def _chunks(self, qty: float) -> list[float]:
        """Split ``qty`` lots into grid-aligned chunks <= ``max_order_lots``."""
        inst = self.instrument
        out: list[float] = []
        rest = round(qty, 8)
        while rest > _EPS:
            c = inst.round_lots(min(rest, self.max_order_lots))
            if c <= 0:
                if rest > _EPS:
                    logger.info("residual %.4f lots below min_lot %.2f not traded", rest, inst.min_lot)
                break
            out.append(c)
            rest = round(rest - c, 8)
        return out

    def plan(self, target_lots: float, positions: list[BrokerPosition], *,
             stop_loss: float | None = None, take_profit: float | None = None,
             sl_distance: float | None = None, tp_distance: float | None = None) -> list[Leg]:
        """Legs that move ``positions`` (ours) to ``target_lots``. Close legs come first."""
        inst = self.instrument
        target = self._grid(target_lots)
        actual = net_lots(positions)
        if abs(target - actual) < inst.lot_step / 2:
            return []
        tsign = _sign(target)
        fifo = sorted(positions, key=lambda p: (p.time, p.ticket))
        legs: list[Leg] = []
        prot = {"stop_loss": stop_loss, "take_profit": take_profit, "sl_distance": sl_distance,
                "tp_distance": tp_distance}

        def close(p: BrokerPosition, qty: float) -> None:
            side = Side.SELL if p.lots > 0 else Side.BUY
            legs.extend(Leg("close", side, c, ticket=p.ticket) for c in self._chunks(qty))

        def open_(qty: float) -> None:
            side = Side.BUY if tsign > 0 else Side.SELL
            legs.extend(Leg("open", side, c, **prot) for c in self._chunks(qty))

        if self.hedging:
            opposite = [p for p in fifo if tsign == 0 or _sign(p.lots) != tsign]
            same = [p for p in fifo if not (tsign == 0 or _sign(p.lots) != tsign)]
            for p in opposite:
                close(p, abs(p.lots))
            held = round(sum(abs(p.lots) for p in same), 8)
            want = abs(target)
            if want < held - _EPS:
                excess = round(held - want, 8)
                for p in same:
                    if excess <= _EPS:
                        break
                    q = inst.round_lots(min(abs(p.lots), excess))
                    if q > 0:
                        close(p, q)
                        excess = round(excess - q, 8)
            elif want > held + _EPS:
                open_(round(want - held, 8))
        else:
            if actual != 0.0 and (tsign == 0 or _sign(actual) != tsign):
                for p in fifo:
                    close(p, abs(p.lots))
                if tsign != 0:
                    open_(abs(target))
            elif abs(target) < abs(actual):
                excess = round(abs(actual) - abs(target), 8)
                for p in fifo:
                    if excess <= _EPS:
                        break
                    q = inst.round_lots(min(abs(p.lots), excess))
                    if q > 0:
                        close(p, q)
                        excess = round(excess - q, 8)
            else:
                open_(round(abs(target) - abs(actual), 8))
        return legs

    # ------------------------------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------------------------------
    def reconcile(
        self,
        target_lots: float,
        decision_time: pd.Timestamp,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        sl_distance: float | None = None,
        tp_distance: float | None = None,
        reason: str = "signal",
    ) -> ExecutionReport:
        """Move this strategy's venue position to ``target_lots`` for the bar decided at
        ``decision_time`` (see module docstring for idempotency and ordering rules).

        Protective parameters apply to OPEN legs only (``*_distance`` anchors at the fill mid).
        A non-finite ``target_lots`` raises ``ValueError``: silently reading NaN as "flat"
        would liquidate the book on a data/sizing bug (the backtest engine HOLDS instead).
        """
        return self._reconcile(target_lots, decision_time, stop_loss=stop_loss, take_profit=take_profit,
                               sl_distance=sl_distance, tp_distance=tp_distance, reason=reason, force=False)

    def _reconcile(
        self,
        target_lots: float,
        decision_time: pd.Timestamp,
        *,
        stop_loss: float | None,
        take_profit: float | None,
        sl_distance: float | None,
        tp_distance: float | None,
        reason: str,
        force: bool,
    ) -> ExecutionReport:
        if not math.isfinite(float(target_lots)):
            raise ValueError(f"target_lots must be finite, got {target_lots!r} (refusing to guess)")
        dt = utc(decision_time)
        key = _decision_key(dt)
        target = self._grid(target_lots)
        decisions = self._state["decisions"]
        if force and not self.dry_run:
            latest = self.latest_decision()
            if latest is not None and key < latest:
                # keep keys monotone: continue the newest record rather than a stale one
                key = latest
                dt = utc(decisions[latest]["decision_time"])
        rec = decisions.get(key)

        def report(status: str, before: float, after: float, **kw: Any) -> ExecutionReport:
            return ExecutionReport(decision_time=dt, symbol=self.symbol, magic=self.magic, target_lots=target,
                                   actual_before=before, actual_after=after, status=status,
                                   hedging=self.hedging, dry_run=self.dry_run, **kw)

        if not self.dry_run and not force:
            if rec is not None and rec.get("status") in TERMINAL:
                logger.info("decision %s already %s: not executing again", key, rec["status"])
                cur = net_lots(self.broker.positions(self.symbol, self.magic))
                return report("duplicate", cur, cur)
            latest = self.latest_decision()
            if latest is not None and key < latest:
                logger.warning("decision %s is older than the latest %s: ignored (stale)", key, latest)
                cur = net_lots(self.broker.positions(self.symbol, self.magic))
                return report("stale", cur, cur, errors=[f"older than latest decision {latest}"])
        if not self.dry_run:
            for k, r in decisions.items():
                if k < key and r.get("status") not in TERMINAL:
                    r["status"] = "superseded"
                    logger.info("decision %s superseded by %s", k, key)

        def refuse(msg: str, cur: float) -> ExecutionReport:
            logger.error(msg)
            if not self.dry_run:
                decisions[key] = {**(rec or {}), "decision_time": dt.isoformat(), "status": "conflict",
                                  "target": target, "legs": (rec or {}).get("legs", {}),
                                  "next_seq": (rec or {}).get("next_seq", 0), "error": msg}
                self._persist()
            return report("conflict", cur, cur, errors=[msg])

        mode_problem = self._check_account_mode()
        try:
            positions = self._own_positions()
        except ForeignPositionError as exc:
            return refuse(str(exc), math.nan)
        actual = net_lots(positions)
        if mode_problem is not None:
            return refuse(mode_problem, actual)

        if not self.hedging:
            foreign = [p for p in self.broker.positions(self.symbol, None) if p.magic != self.magic]
            if foreign:
                return refuse(f"netting account holds {len(foreign)} foreign position(s) on {self.symbol} "
                              f"(magic {sorted({p.magic for p in foreign})}): magic isolation impossible, "
                              "refusing to trade", actual)

        if not self.dry_run:
            if rec is None:
                rec = {"decision_time": dt.isoformat(), "status": "in_progress", "target": target,
                       "reason": reason, "legs": {}, "next_seq": 0,
                       "created": pd.Timestamp.now(tz="UTC").isoformat()}
                decisions[key] = rec
            else:
                rec["status"] = "in_progress"
                rec["target"] = target
                unverified = self._resolve_sending(rec)
                if unverified and not force:
                    # An order from a previous run may or may not be at the venue and the deal
                    # history cannot tell: re-planning now could send it a second time. Wait
                    # (this bar's next attempt / the next bar re-reads the venue).
                    msg = (f"in-flight order(s) {unverified} from a previous run could not be verified in the "
                           "deal history: not sending on top of them")
                    logger.critical(msg)
                    self._persist()
                    return report("unknown", actual, actual, errors=[msg])
                positions = self._own_positions()
                actual = net_lots(positions)

        legs = self.plan(target, positions, stop_loss=stop_loss, take_profit=take_profit,
                         sl_distance=sl_distance, tp_distance=tp_distance)
        if self.dry_run:
            for leg in legs:
                logger.info("[DRY RUN] would %s %s %.2f lots%s", leg.kind, "BUY" if leg.side > 0 else "SELL",
                            leg.lots, f" on ticket {leg.ticket}" if leg.ticket is not None else "")
            return report("dry_run", actual, actual, planned=legs)
        if not legs:
            rec["status"] = "complete"
            self._persist()
            return report("noop", actual, actual)

        self._persist()
        leg_reports: list[LegReport] = []
        errors: list[str] = []
        status = "filled"
        for leg in legs:
            seq = int(rec["next_seq"])
            rec["next_seq"] = seq + 1
            cid = make_client_id(self.magic, dt, seq)
            rec["legs"][cid] = {**leg.to_dict(), "status": "sending", "attempts": 0,
                                "updated": pd.Timestamp.now(tz="UTC").isoformat()}
            self._persist()  # write-ahead: a crash now leaves a verifiable "sending" record
            lr = self._send(leg, cid, dt, reason)
            leg_reports.append(lr)
            rec["legs"][cid].update(status=lr.status, retcode=lr.retcode, message=lr.message[:200],
                                    attempts=lr.attempts, ticket=lr.ticket,
                                    price=lr.fill.price if lr.fill is not None else None,
                                    updated=pd.Timestamp.now(tz="UTC").isoformat())
            self._persist()
            if not lr.ok:
                errors.append(f"{cid} {leg.kind} {leg.lots:.2f}: {lr.status} ({lr.retcode}) {lr.message}")
                if lr.status == OrderStatusCode.MARKET_CLOSED.value:
                    status = "deferred"
                    rec["status"] = "deferred"
                elif lr.status == OrderStatusCode.UNKNOWN.value:
                    # outcome unverifiable and NOT resent: an operator/next reconcile must look
                    # at the venue; later legs are not sent on top of an unknown state
                    status = "unknown"
                    rec["status"] = "in_progress"
                elif lr.status in (OrderStatusCode.REQUOTE.value,
                                   OrderStatusCode.TOO_MANY_REQUESTS.value, OrderStatusCode.ERROR.value):
                    status = "partial" if any(x.ok for x in leg_reports) else "rejected"
                    rec["status"] = "in_progress"  # transient: may be resumed for this bar
                else:
                    status = "partial" if any(x.ok for x in leg_reports) else "rejected"
                    rec["status"] = "rejected"
                log = logger.warning if status == "deferred" else logger.error
                log("order leg not executed, stopping this reconcile: %s", errors[-1])
                break
        else:
            rec["status"] = "complete"
        after = net_lots(self._own_positions())
        if status == "filled" and abs(after - target) > _EPS:
            # e.g. a partial fill or a residual below min_lot
            errors.append(f"position {after:+.2f} differs from target {target:+.2f} after all legs")
            status = "partial"
        rec["actual_after"] = after
        self._persist()
        return report(status, actual, after, planned=legs, legs=leg_reports, errors=errors)

    def flatten(self, decision_time: pd.Timestamp, *, reason: str = "flatten") -> ExecutionReport:
        """Close every position of this magic on the symbol (a reconcile to zero).

        Unlike a bar decision, a flatten is never refused as ``duplicate``/``stale``: a
        shutdown in the same minute as the last bar decision must still close the book
        (see the module docstring). The netting-account conflict check still applies."""
        return self._reconcile(0.0, decision_time, stop_loss=None, take_profit=None, sl_distance=None,
                               tp_distance=None, reason=reason, force=True)

    def _resolve_sending(self, rec: dict[str, Any]) -> list[str]:
        """Legs left in ``sending`` by a crash: look them up in the venue's deal history.

        The re-plan that follows is computed from the venue's positions, which already reflect
        an order that executed. Returns the client ids that could NOT be verified (lookup
        failed); they are retried on the next call and block new sends until then."""
        unverified: list[str] = []
        for cid, leg in rec.get("legs", {}).items():
            if leg.get("status") not in ("sending", "unverified"):
                continue
            try:
                deals = self.broker.find_deals(cid, symbol=self.symbol, magic=self.magic)
                leg["status"] = "filled" if deals else "not_executed"
                leg["verified"] = True
            except Exception as exc:  # noqa: BLE001 - venue specific
                logger.error("cannot verify in-flight order %s: %s", cid, exc)
                leg["status"] = "unverified"
                leg["verified"] = False
                unverified.append(cid)
            logger.warning("in-flight order %s from a previous run resolved as %s", cid, leg["status"])
        return unverified

    def _request(self, leg: Leg, cid: str, dt: pd.Timestamp, reason: str) -> OrderRequest:
        return OrderRequest(client_id=cid, symbol=self.symbol, side=leg.side, lots=leg.lots, time=dt,
                            magic=self.magic, stop_loss=leg.stop_loss, take_profit=leg.take_profit,
                            sl_distance=leg.sl_distance, tp_distance=leg.tp_distance,
                            position_ticket=leg.ticket, reason=reason)

    def _send(self, leg: Leg, cid: str, dt: pd.Timestamp, reason: str) -> LegReport:
        before = net_lots(self._own_positions())
        attempts = 0
        res: OrderResult | None = None
        while True:
            attempts += 1
            req = self._request(leg, cid, dt, reason)
            try:
                res = self.broker.place_order(req)
            except Exception as exc:  # adapter bug or transport error: outcome unknown
                logger.exception("place_order raised for %s", cid)
                res = OrderResult(ok=False, retcode="exception", message=f"{type(exc).__name__}: {exc}",
                                  status=OrderStatusCode.UNKNOWN, client_id=cid)
            if res.ok:
                break
            if res.status == OrderStatusCode.UNKNOWN:
                verdict, verified = self._verify(leg, cid, before)
                if verdict == "executed":
                    assert verified is not None
                    res = verified
                    break
                if verdict == "inconclusive":
                    # Never resend what may already be at the venue (a flaky link would turn
                    # one order into two); positions are re-read at the next reconcile.
                    logger.critical("order %s: outcome UNKNOWN and not verifiable; NOT resending", cid)
                    res = OrderResult(ok=False, retcode=res.retcode,
                                      message=f"{res.message}; outcome could not be verified, not resent",
                                      status=OrderStatusCode.UNKNOWN, client_id=cid,
                                      requested_price=res.requested_price)
                    break
                # "not_executed": provably nothing happened, a resend is safe
            retry = res.status in RETRYABLE_STATUSES or res.status == OrderStatusCode.UNKNOWN or res.retryable
            if not retry or attempts > self.max_retries:
                break
            delay = min(self.backoff_seconds * (2 ** (attempts - 1)), self.backoff_max)
            logger.warning("order %s %s (%s): retry %d/%d in %.2fs", cid, res.status.value, res.retcode,
                           attempts, self.max_retries, delay)
            self._sleep(delay)
        assert res is not None
        if res.ok:
            logger.info("order %s %s %.2f lots: %s", cid, leg.kind, leg.lots, res.status.value)
        elif res.status == OrderStatusCode.MARKET_CLOSED:
            logger.warning("order %s %s %.2f lots not sent: market closed (deferred)", cid, leg.kind, leg.lots)
        elif res.status == OrderStatusCode.UNKNOWN:
            logger.error("order %s %s %.2f lots: outcome UNKNOWN (%s) %s", cid, leg.kind, leg.lots, res.retcode,
                         res.message)
        else:
            logger.error("order %s %s %.2f lots REJECTED: %s (%s) %s", cid, leg.kind, leg.lots,
                         res.status.value, res.retcode, res.message)
        return LegReport(client_id=cid, leg=leg, ok=res.ok, status=res.status.value, retcode=res.retcode,
                         message=res.message, attempts=attempts, fill=res.fill,
                         ticket=res.position_ticket if res.position_ticket is not None else leg.ticket,
                         requested_price=res.requested_price, extra=dict(res.extra))

    def _verify(self, leg: Leg, cid: str, before: float) -> tuple[str, OrderResult | None]:
        """Did an order with an unconfirmed outcome execute?

        Returns ``("executed", result)`` (found in the deal history, or the net position moved
        in the leg's direction — a synthetic OK result, ``partial`` if it moved by less than the
        leg), ``("not_executed", None)`` ONLY when both lookups succeeded, no deal carries the
        client id and the net position did not move, and ``("inconclusive", None)`` otherwise
        (a lookup failed, or the position moved against/beyond the leg — e.g. a concurrent
        stop-out). Only ``not_executed`` permits a resend.
        """
        deals_ok = True
        try:
            deals = self.broker.find_deals(cid, symbol=self.symbol, magic=self.magic)
        except Exception as exc:
            logger.error("deal lookup failed for %s: %s", cid, exc)
            deals, deals_ok = [], False
        if deals:
            d = deals[-1]
            fill = Fill(client_id=cid, symbol=self.symbol, side=d.side, lots=float(sum(x.lots for x in deals)),
                        price=d.price, time=d.time, commission=abs(sum(x.commission for x in deals)),
                        broker_ref=str(d.ticket))
            full = fill.lots >= leg.lots - _EPS
            logger.warning("order %s with unknown outcome found in deal history: executed %.2f of %.2f lots",
                           cid, fill.lots, leg.lots)
            return "executed", OrderResult(ok=True, retcode="verified", message="executed (verified in deal history)",
                                           fill=fill, broker_ref=str(d.ticket),
                                           status=OrderStatusCode.FILLED if full else OrderStatusCode.PARTIAL,
                                           client_id=cid, position_ticket=d.position_ticket)
        try:
            after = net_lots(self._own_positions())
        except Exception as exc:
            logger.error("position lookup failed for %s: %s", cid, exc)
            return "inconclusive", None
        moved = round(after - before, 8)
        want = leg.signed_lots
        if abs(moved - want) < _EPS:
            logger.warning("order %s with unknown outcome executed (net position moved %+.2f)", cid, moved)
            return "executed", OrderResult(ok=True, retcode="verified", message="executed (verified by position change)",
                                           status=OrderStatusCode.FILLED, client_id=cid, position_ticket=leg.ticket)
        if abs(moved) < _EPS:
            if deals_ok:
                return "not_executed", None
            return "inconclusive", None
        if _sign(moved) == _sign(want) and abs(moved) < abs(want):
            logger.warning("order %s with unknown outcome partially executed (net position moved %+.2f of %+.2f)",
                           cid, moved, want)
            return "executed", OrderResult(ok=True, retcode="verified",
                                           message=f"partially executed (position moved {moved:+.2f} of {want:+.2f})",
                                           status=OrderStatusCode.PARTIAL, client_id=cid, position_ticket=leg.ticket)
        logger.error("order %s: net position moved %+.2f while a %+.2f leg was in flight: inconclusive",
                     cid, moved, want)
        return "inconclusive", None
