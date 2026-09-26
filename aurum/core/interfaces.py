"""Structural interfaces (Protocols) that decouple the layers.

The backtest engine, the RL environment and the live runner all drive the SAME sizer and
risk manager implementations through these protocols — that is what guarantees that what
you backtest is what you trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from aurum.core.instrument import Instrument


@runtime_checkable
class PositionSizer(Protocol):
    def target_lots(
        self,
        forecast: float,
        vol_ann: float,
        equity: float,
        price: float,
        instrument: Instrument,
        *,
        current_lots: float = 0.0,
        drawdown: float = 0.0,
    ) -> float:
        """Signed lots to hold after the next fill (already rounded via instrument.round_lots)."""
        ...


@dataclass
class RiskContext:
    """Snapshot handed to the risk manager at every decision point (bar close)."""

    time: pd.Timestamp                 # decision time (UTC) = bar available_at
    equity: float                      # mark-to-market equity (USD)
    current_lots: float                # signed position now
    target_lots: float                 # signed position requested by sizing/agents
    price: float                       # mid price at decision
    spread: float                      # current spread (price units)
    vol_ann: float                     # annualised volatility forecast (fraction, 0.15 = 15%)
    bar_index: int | None = None
    data_age_seconds: float | None = None       # live only: age of the latest closed bar
    upcoming_events: pd.DataFrame | None = None  # calendar rows with time >= now (may be None)
    recent_events: pd.DataFrame | None = None    # calendar rows with time < now (may be None)
    extra: dict = field(default_factory=dict)


@dataclass
class RiskDecision:
    approved_lots: float               # signed lots allowed after risk checks
    halted: bool = False               # kill-switch engaged: flatten and stop trading
    reasons: list[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(self.reasons)


@runtime_checkable
class RiskManager(Protocol):
    def evaluate(self, ctx: RiskContext) -> RiskDecision: ...

    def on_bar(self, time: pd.Timestamp, equity: float) -> None:
        """Update equity-path state (day start equity, peak) — called once per bar close."""
        ...
