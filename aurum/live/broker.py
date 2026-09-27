"""Broker abstraction for the production execution path (SPEC §11).

Every venue adapter (:class:`~aurum.live.paper.PaperBroker`,
:class:`~aurum.live.mt5.MT5Broker`) implements the :class:`Broker` protocol, so the order
manager (:mod:`aurum.live.oms`) and the runner (:mod:`aurum.live.runner`) are venue-agnostic.

Design rules
------------
* **Closed bars only.** ``latest_bars`` never returns the bar that is still forming: a
  decision may only use information available at the bar close (SPEC §0-1). Bars come back
  in the canonical schema of :mod:`aurum.data.schema` (UTC open-time index, MID prices,
  ``spread`` in price units, ``available_at``).
* **Magic-number isolation.** A strategy instance owns exactly the positions carrying its
  ``magic`` on its symbol. ``positions(symbol, magic)`` filters on both and adapters must
  refuse to modify a ticket that does not belong to the caller (the legacy v1 bot closed
  other EAs' positions because it filtered by symbol only).
* **Outcomes are data, not exceptions.** ``place_order`` returns an :class:`OrderResult`
  with a normalised ``status`` (see :class:`OrderStatusCode`) and a ``retryable`` flag; the
  OMS decides what to do. ``status == "unknown"`` means the venue could not confirm the
  outcome (timeout, lost connection): the order may or may not have executed and must be
  verified against positions/deals before any resend.
* **Time is UTC.** ``server_time()`` is the venue's notion of "now" converted to UTC; the
  runner schedules bar closes from it. Tests use :class:`SimulatedClock` so a run over
  hundreds of bars completes instantly.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from aurum.core.types import Fill, Order, Side

logger = logging.getLogger(__name__)

__all__ = [
    "AccountInfo",
    "Broker",
    "BrokerDeal",
    "BrokerError",
    "BrokerPosition",
    "Clock",
    "OrderRequest",
    "OrderResult",
    "OrderStatusCode",
    "Quote",
    "SimulatedClock",
    "SystemClock",
    "net_lots",
]


class BrokerError(RuntimeError):
    """Venue-level failure that is not an order outcome (connection, login, symbol)."""


class OrderStatusCode(str, Enum):
    """Venue-independent order outcome."""

    FILLED = "filled"                 # executed (fully)
    PARTIAL = "partial"               # partially executed (remaining volume cancelled)
    QUEUED = "queued"                 # accepted, will execute later (e.g. at the next open)
    REJECTED = "rejected"             # generic rejection; not retryable
    REQUOTE = "requote"               # price moved: retry with a fresh quote
    MARKET_CLOSED = "market_closed"   # session closed / no quotes: defer, do not hammer
    NO_MONEY = "no_money"             # insufficient margin
    INVALID_VOLUME = "invalid_volume"
    INVALID_STOPS = "invalid_stops"
    TRADE_DISABLED = "trade_disabled"  # symbol/account/terminal does not allow trading
    TOO_MANY_REQUESTS = "too_many_requests"
    UNKNOWN = "unknown"               # outcome not confirmed (timeout/connection): VERIFY
    ERROR = "error"                   # adapter/programming error


#: statuses where an immediate resend (after backoff) is safe without verification.
RETRYABLE_STATUSES = frozenset({OrderStatusCode.REQUOTE, OrderStatusCode.TOO_MANY_REQUESTS})


@dataclass(frozen=True)
class AccountInfo:
    """Account snapshot (money in the account currency, normally USD)."""

    equity: float
    balance: float
    margin: float
    free_margin: float
    currency: str = "USD"
    is_demo: bool = True
    leverage: float = 100.0
    hedging: bool = False
    server: str | None = None
    name: str | None = None

    @property
    def margin_level(self) -> float:
        """Equity / margin (MT5 "margin level" as a fraction; inf when no margin is used)."""
        return math.inf if self.margin <= 0 else self.equity / self.margin

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity, "balance": self.balance, "margin": self.margin,
            "free_margin": self.free_margin, "currency": self.currency, "is_demo": self.is_demo,
            "leverage": self.leverage, "hedging": self.hedging, "server": self.server,
        }


@dataclass(frozen=True)
class BrokerPosition:
    """An open position (ticket) at the venue. ``lots`` is SIGNED (+ long, - short).

    ``sl``/``tp`` are MID price levels (the :class:`OrderRequest` convention), so they can be
    passed back as ``stop_loss``/``take_profit`` unchanged; adapters whose venue stores
    bid/ask trigger levels convert them.
    """

    ticket: int
    symbol: str
    lots: float
    price_open: float
    sl: float | None
    tp: float | None
    magic: int
    comment: str
    time: pd.Timestamp
    swap: float = 0.0
    profit: float = 0.0

    @property
    def side(self) -> Side:
        return Side.BUY if self.lots > 0 else Side.SELL

    @property
    def volume(self) -> float:
        return abs(self.lots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket, "symbol": self.symbol, "lots": self.lots,
            "price_open": self.price_open, "sl": self.sl, "tp": self.tp, "magic": self.magic,
            "comment": self.comment, "time": self.time, "swap": self.swap, "profit": self.profit,
        }


@dataclass(frozen=True)
class BrokerDeal:
    """An executed deal from the venue's history (used to verify uncertain outcomes)."""

    ticket: int
    order: int | None
    position_ticket: int | None
    symbol: str
    side: Side
    lots: float
    price: float
    magic: int
    comment: str
    time: pd.Timestamp
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0


@dataclass(frozen=True)
class Quote:
    """Best bid/ask at ``time`` (UTC). ``bar_range`` is an optional cost-model input
    (high - low of the execution bar, used by the paper broker's slippage term)."""

    time: pd.Timestamp
    bid: float
    ask: float
    bar_range: float = 0.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass
class OrderRequest:
    """A market order sent to a :class:`Broker`.

    ``lots`` is always positive; direction is ``side``. ``position_ticket`` targets an
    existing position: the deal is then the OPPOSITE of that position and closes (part of)
    it — the only safe way to reduce exposure on hedging accounts and on accounts shared
    with other EAs. ``stop_loss``/``take_profit`` are absolute MID price levels attached
    broker-side; ``sl_distance``/``tp_distance`` (price units) are the alternative for a
    position being ENTERED: the adapter anchors them at the execution mid, exactly like
    :meth:`aurum.execution.simulator.ExecutionSimulator.step` (``stop_distance``).
    ``client_id`` is the idempotency key (sent as the order comment).
    """

    client_id: str
    symbol: str
    side: Side
    lots: float
    time: pd.Timestamp
    magic: int
    stop_loss: float | None = None
    take_profit: float | None = None
    sl_distance: float | None = None
    tp_distance: float | None = None
    position_ticket: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.side = Side(int(self.side))
        if not (math.isfinite(self.lots) and self.lots > 0):
            raise ValueError(f"OrderRequest.lots must be positive and finite, got {self.lots!r}")
        for name in ("stop_loss", "take_profit", "sl_distance", "tp_distance"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(v) and v > 0):
                raise ValueError(f"OrderRequest.{name} must be positive or None, got {v!r}")
        if self.stop_loss is not None and self.sl_distance is not None:
            raise ValueError("pass either stop_loss or sl_distance, not both")
        if self.take_profit is not None and self.tp_distance is not None:
            raise ValueError("pass either take_profit or tp_distance, not both")

    @property
    def is_close(self) -> bool:
        return self.position_ticket is not None

    @property
    def signed_lots(self) -> float:
        return float(self.side) * self.lots

    @classmethod
    def from_order(cls, order: Order, *, magic: int, position_ticket: int | None = None) -> OrderRequest:
        """Adapter from the research-side :class:`aurum.core.types.Order`."""
        return cls(client_id=order.client_id, symbol=order.symbol, side=order.side, lots=order.lots,
                   time=order.time, magic=magic, stop_loss=order.stop_loss,
                   take_profit=order.take_profit, position_ticket=position_ticket, reason=order.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id, "symbol": self.symbol, "side": int(self.side),
            "lots": self.lots, "time": self.time, "magic": self.magic, "stop_loss": self.stop_loss,
            "take_profit": self.take_profit, "sl_distance": self.sl_distance,
            "tp_distance": self.tp_distance, "position_ticket": self.position_ticket,
            "reason": self.reason,
        }


@dataclass
class OrderResult:
    """Outcome of :meth:`Broker.place_order`.

    ``ok`` is True when the order executed (``filled``/``partial``) or was accepted for later
    execution (``queued``). ``retcode`` is the venue's raw code (int for MT5), ``status`` the
    normalised :class:`OrderStatusCode`.
    """

    ok: bool
    retcode: int | str
    message: str
    fill: Fill | None = None
    broker_ref: str | None = None
    status: OrderStatusCode = OrderStatusCode.FILLED
    retryable: bool = False
    client_id: str | None = None
    position_ticket: int | None = None
    requested_price: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def executed_lots(self) -> float:
        return 0.0 if self.fill is None else float(self.fill.lots)

    def to_dict(self) -> dict[str, Any]:
        f = self.fill
        return {
            "ok": self.ok, "retcode": self.retcode, "status": self.status.value,
            "message": self.message, "broker_ref": self.broker_ref, "client_id": self.client_id,
            "position_ticket": self.position_ticket, "retryable": self.retryable,
            "requested_price": self.requested_price,
            "fill": None if f is None else {
                "side": int(f.side), "lots": f.lots, "price": f.price, "time": f.time,
                "commission": f.commission, "spread_cost": f.spread_cost,
                "slippage_cost": f.slippage_cost, "broker_ref": f.broker_ref,
            },
        }


def net_lots(positions: list[BrokerPosition]) -> float:
    """Signed sum of position lots, rounded to 1e-8 (float noise from partial closes)."""
    return round(sum(p.lots for p in positions), 8) + 0.0


@runtime_checkable
class Broker(Protocol):
    """Venue interface used by the OMS and the live runner (SPEC §11)."""

    def account(self) -> AccountInfo:
        """Current account snapshot."""
        ...

    def positions(self, symbol: str | None = None, magic: int | None = None) -> list[BrokerPosition]:
        """Open positions, filtered by symbol AND magic when given (None = no filter)."""
        ...

    def place_order(self, order: OrderRequest) -> OrderResult:
        """Send a market order (open, add, or close ``order.position_ticket``)."""
        ...

    def close_position(self, ticket: int, *, magic: int, lots: float | None = None,
                       client_id: str = "") -> OrderResult:
        """Close (part of) one ticket by an opposite deal. Refuses foreign-magic tickets."""
        ...

    def close_all(self, symbol: str, magic: int) -> list[OrderResult]:
        """Close every position of ``symbol`` carrying ``magic`` (and nothing else)."""
        ...

    def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """The last ``n`` CLOSED bars in the canonical schema (never the forming bar)."""
        ...

    def is_demo(self) -> bool:
        """True only for demo/paper accounts (real-money guard input)."""
        ...

    def is_hedging(self) -> bool:
        """True if the account keeps several positions per symbol (MT5 hedging mode)."""
        ...

    def server_time(self) -> pd.Timestamp:
        """Venue "now" in UTC."""
        ...

    def quote(self, symbol: str) -> Quote | None:
        """Current executable quote; None when the market is closed / no fresh quote."""
        ...

    def find_deals(self, client_id: str, *, symbol: str, magic: int,
                   since: pd.Timestamp | None = None) -> list[BrokerDeal]:
        """Executed deals whose comment equals ``client_id`` (outcome verification)."""
        ...


# ---------------------------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------------------------
@runtime_checkable
class Clock(Protocol):
    """Source of "now" (UTC) and of waiting. Swap in :class:`SimulatedClock` for replays."""

    simulated: bool

    def now(self) -> pd.Timestamp: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Wall clock (UTC). ``sleep`` is interruptible through ``stop_event`` (graceful shutdown)."""

    simulated = False

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self.stop_event = stop_event

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def sleep(self, seconds: float) -> None:
        s = max(0.0, float(seconds))
        if self.stop_event is not None:
            self.stop_event.wait(s)
        else:
            time.sleep(s)


class SimulatedClock:
    """Deterministic clock: ``sleep`` advances time instantly (tests, paper replays)."""

    simulated = True

    def __init__(self, start: pd.Timestamp | str) -> None:
        t = pd.Timestamp(start)
        if t.tz is None:
            raise ValueError("SimulatedClock start must be tz-aware (UTC)")
        self._now = t.tz_convert("UTC")

    def now(self) -> pd.Timestamp:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now = self._now + pd.Timedelta(seconds=float(seconds))

    def advance_to(self, t: pd.Timestamp | str) -> None:
        """Move forward to ``t`` (no-op if ``t`` is in the past: time never runs backwards)."""
        ts = pd.Timestamp(t)
        if ts.tz is None:
            raise ValueError("advance_to needs a tz-aware timestamp")
        ts = ts.tz_convert("UTC")
        if ts > self._now:
            self._now = ts
