"""Shared value types for orders, fills and the market-data bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


@dataclass
class MarketData:
    """Everything a feature or strategy may look at, all point-in-time aligned.

    bars   : canonical bars frame (see ``aurum.data.schema``) for the trading timeframe.
    macro  : optional dict name -> DataFrame. Each macro frame has a tz-aware UTC
             DatetimeIndex and an ``available_at`` column; values may only be used once
             ``available_at <= decision time``. Use ``aurum.data.pit.asof_join`` to align.
    events : optional economic-calendar frame (see ``aurum.data.calendar``) with columns
             ``time`` (UTC, scheduled release), ``name``, ``importance`` (1..3), ``currency``.
    """

    bars: pd.DataFrame
    macro: dict[str, pd.DataFrame] = field(default_factory=dict)
    events: pd.DataFrame | None = None

    def slice(self, end: pd.Timestamp | None = None, start: pd.Timestamp | None = None) -> MarketData:
        """Return a view restricted to bars with open time in [start, end] (inclusive).

        Macro/event frames are left untouched: consumers must still align them by
        ``available_at`` against each bar's ``available_at``.
        """
        b = self.bars
        if start is not None:
            b = b.loc[b.index >= start]
        if end is not None:
            b = b.loc[b.index <= end]
        return MarketData(bars=b, macro=self.macro, events=self.events)


class Side(int, Enum):
    SELL = -1
    BUY = 1


class OrderStatus(str, Enum):
    NEW = "new"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    client_id: str                 # idempotency key, e.g. f"{strategy}-{bar_time:%Y%m%d%H%M}"
    symbol: str
    side: Side
    lots: float                    # always positive; direction in ``side``
    time: pd.Timestamp             # decision time (UTC)
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""


@dataclass
class Fill:
    client_id: str
    symbol: str
    side: Side
    lots: float
    price: float                   # executed price including spread & slippage
    time: pd.Timestamp
    commission: float = 0.0        # USD, positive = cost
    slippage_cost: float = 0.0     # USD, positive = cost (vs mid)
    spread_cost: float = 0.0       # USD, positive = cost (vs mid)
    status: OrderStatus = OrderStatus.FILLED
    broker_ref: str | None = None


@dataclass
class Trade:
    """A round trip (flat -> position -> flat, or a reversal leg)."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
    lots: float
    entry_price: float
    exit_price: float
    pnl: float                     # USD, net of costs and swap attributed to the trade
    costs: float                   # USD, commission + spread + slippage
    swap: float                    # USD, signed (+ = received)
    exit_reason: str = "signal"    # "signal" | "stop" | "take_profit" | "risk" | "end"
