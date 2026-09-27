"""Contract specification for the traded instrument.

Defaults describe a typical retail/prime-broker XAUUSD CFD:
1 lot = 100 troy ounces, price quoted in USD per ounce with 2 decimals.
All money amounts are in the account currency (assumed USD).

Overnight financing is modelled by :class:`aurum.execution.costs.FinancingModel`
(``CostModel.financing``). Its default ``"rate"`` mode charges the USD benchmark rate +/- a
broker markup on the notional and only uses ``triple_swap_weekday`` / ``rollover_hour_utc``
from here; the per-lot ``swap_long_per_lot`` / ``swap_short_per_lot`` quotes below are used
by the ``"fixed"`` mode only (a broker's quoted swap table). The -45/+15 USD defaults are
roughly 9%/yr of a $1,800 notional for a long, far above the 2012-2021 benchmark rates, so
they are kept for backward compatibility, not as a realistic default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    symbol: str = "XAUUSD"
    contract_size: float = 100.0            # ounces per 1.00 lot
    tick_size: float = 0.01                 # minimum price increment (USD/oz)
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 50.0
    margin_rate: float = 0.01               # 1% => 1:100 leverage; margin = notional * margin_rate
    commission_per_lot: float = 0.0         # USD per lot per side (ECN accounts often ~3.5)
    swap_long_per_lot: float = -45.0        # "fixed" financing: USD/lot/night held long (negative = you pay)
    swap_short_per_lot: float = 15.0        # "fixed" financing: USD/lot/night held short
    triple_swap_weekday: int = 2            # 0=Mon ... 2=Wed charges 3 nights (weekend financing)
    rollover_hour_utc: int = 21             # broker "server midnight" in UTC (approx 17:00 New York)
    tags: dict = field(default_factory=dict, compare=False, hash=False)

    # --- conversions -------------------------------------------------------------------
    def notional(self, lots: float, price: float) -> float:
        """USD notional of a (signed) position."""
        return lots * self.contract_size * price

    def pnl(self, lots: float, entry_price: float, exit_price: float) -> float:
        """USD PnL for a signed position moved from entry to exit (no costs)."""
        return lots * self.contract_size * (exit_price - entry_price)

    def value_per_point(self, lots: float = 1.0) -> float:
        """USD value of a $1.00 move in the gold price for ``lots`` lots."""
        return lots * self.contract_size

    def margin_required(self, lots: float, price: float) -> float:
        return abs(self.notional(lots, price)) * self.margin_rate

    def round_lots(self, lots: float) -> float:
        """Round a signed lot size toward zero onto the lot grid and clip to limits.

        Sizes smaller than ``min_lot`` become 0 (we never round *up* into risk).
        """
        if not math.isfinite(lots) or lots == 0:
            return 0.0
        sign = 1.0 if lots > 0 else -1.0
        mag = min(abs(lots), self.max_lot)
        steps = math.floor(mag / self.lot_step + 1e-9)
        mag = round(steps * self.lot_step, 8)
        if mag < self.min_lot - 1e-12:
            return 0.0
        return sign * mag

    def round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)


XAUUSD = Instrument()
