"""Transaction-cost and financing model for XAUUSD CFDs (SPEC §8).

Every USD that leaves the account in a simulation is computed here, so that the research
backtest, the RL environment, the paper broker and the LLM-desk replay all agree
(SPEC §0.2 "one simulator").

Cost components
---------------
Prices in bar frames are **mid** prices; ``spread`` is the full bid/ask spread in USD/oz.

* **Spread** — crossing the book costs half the (effective) spread per ounce:
  a buy fills at ``mid + s/2``, a sell at ``mid - s/2``. The effective spread is
  ``max(spread * spread_multiplier, min_spread)``: data vendors often report a time-averaged
  or best-quote spread, and retail execution is rarely better than a broker floor.
* **Slippage** — adverse price movement between order and execution, in USD/oz::

      slip = slippage_fixed + slippage_range_frac * (high - low) + impact_coef * sqrt(|lots|)

  The range term scales slippage with realised intrabar volatility (fills are worse in
  fast markets); the last term is the square-root market-impact law (Almgren et al. 2005,
  "Direct estimation of equity market impact"; Tóth et al. 2011, "Anomalous price impact
  and the critical nature of liquidity"). ``impact_coef`` is in USD/oz per sqrt(lot) and
  defaults to 0 because retail XAUUSD size is tiny relative to the market.
  Limit orders (take-profits) do not slip adversely, so ``limit=True`` sets ``slip = 0``.
* **Commission** — USD per lot per side (``commission_per_lot``; ``None`` defers to the
  instrument, e.g. ~3.5 on ECN accounts).
* **Swap / financing** — CFD positions held over the daily rollover
  (``instrument.rollover_hour_utc`` on a weekday) are charged or credited
  ``swap_{long,short}_per_lot`` per lot per night. The rollover on
  ``instrument.triple_swap_weekday`` (Wednesday by convention) counts three nights because
  spot settlement (T+2) of that day's roll spans the weekend. Weekend (Sat/Sun) "rollovers"
  do not exist. See :func:`rollover_nights` for the exact interval semantics.

All cost values are positive USD amounts (a cost), swap is signed (+ = received).
Fill prices are *not* rounded to the tick grid: the rounding effect (<= $0.005/oz) is noise
compared with the modelled spread and keeps the PnL identity exact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from aurum.core.instrument import XAUUSD, Instrument

logger = logging.getLogger(__name__)

NS_PER_DAY: int = 86_400 * 10**9
NS_PER_HOUR: int = 3_600 * 10**9
# 1970-01-01 (day 0 of the epoch) was a Thursday.
_EPOCH_WEEKDAY: int = 3


class FillPrice(NamedTuple):
    """Result of :meth:`CostModel.fill_price`: executed price and the USD cost split."""

    price: float            # executed price (USD/oz), includes half-spread and slippage
    spread_cost: float      # USD, >= 0: half effective spread * |lots| * contract_size
    slippage_cost: float    # USD, >= 0: slippage per oz * |lots| * contract_size


def _side_sign(side: object) -> int:
    """Map ``Side`` / +-1 / signed float to +1 (buy) or -1 (sell)."""
    try:
        s = float(side)  # Side is an int Enum -> float works
    except (TypeError, ValueError) as exc:
        raise ValueError(f"side must be +1/-1 or aurum.core.types.Side, got {side!r}") from exc
    if s > 0:
        return 1
    if s < 0:
        return -1
    raise ValueError("side must be non-zero")


@dataclass(frozen=True)
class CostModel:
    """Spread / slippage / commission / swap model (see module docstring).

    Parameters
    ----------
    spread_multiplier : scale applied to the bar's quoted spread (stress-test with 1.5-2.0).
    min_spread        : floor on the effective spread in USD/oz.
    slippage_fixed    : constant adverse slippage per fill in USD/oz.
    slippage_range_frac : fraction of the execution bar's high-low range added as slippage.
    impact_coef       : square-root impact coefficient, USD/oz per sqrt(lot).
    commission_per_lot : USD per lot per side; ``None`` uses ``instrument.commission_per_lot``.
    """

    spread_multiplier: float = 1.0
    min_spread: float = 0.10
    slippage_fixed: float = 0.02
    slippage_range_frac: float = 0.02
    impact_coef: float = 0.0
    commission_per_lot: float | None = None

    def __post_init__(self) -> None:
        for name in ("spread_multiplier", "min_spread", "slippage_fixed", "slippage_range_frac",
                     "impact_coef"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"CostModel.{name} must be finite and >= 0, got {v!r}")
        if self.commission_per_lot is not None and not (
            math.isfinite(self.commission_per_lot) and self.commission_per_lot >= 0
        ):
            raise ValueError("CostModel.commission_per_lot must be None or finite >= 0")

    # ---- constructors ----------------------------------------------------------------------
    @classmethod
    def zero(cls) -> CostModel:
        """A frictionless model (no spread, slippage or commission). Swap still comes from the
        instrument — pass an instrument with zero swaps for a fully frictionless run."""
        return cls(spread_multiplier=0.0, min_spread=0.0, slippage_fixed=0.0,
                   slippage_range_frac=0.0, impact_coef=0.0, commission_per_lot=0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- per-ounce components ----------------------------------------------------------------
    def effective_spread(self, spread: float) -> float:
        """Full spread (USD/oz) actually paid: ``max(spread * multiplier, min_spread)``."""
        s = float(spread)
        if not math.isfinite(s) or s < 0:
            logger.debug("invalid spread %r; using min_spread", spread)
            s = 0.0
        return max(s * self.spread_multiplier, self.min_spread)

    def slippage(self, bar_range: float, lots: float) -> float:
        """Adverse slippage per ounce (USD/oz) for a market/stop order of ``|lots|``."""
        rng = float(bar_range)
        if not math.isfinite(rng) or rng < 0:
            rng = 0.0
        slip = self.slippage_fixed + self.slippage_range_frac * rng
        if self.impact_coef > 0.0:
            slip += self.impact_coef * math.sqrt(abs(float(lots)))
        return slip

    # ---- fills -----------------------------------------------------------------------------
    def fill_price(
        self,
        side: object,
        mid: float,
        spread: float,
        bar_range: float,
        lots: float,
        *,
        instrument: Instrument = XAUUSD,
        limit: bool = False,
    ) -> FillPrice:
        """Executed price and USD cost split for a fill of ``|lots|`` at mid price ``mid``.

        ``side`` is +1/``Side.BUY`` or -1/``Side.SELL``. A buy pays
        ``mid + spread_eff/2 + slip``, a sell receives ``mid - spread_eff/2 - slip``.
        ``limit=True`` models a resting limit order (take-profit): half-spread but no slippage.
        """
        s = _side_sign(side)
        q = abs(float(lots))
        half = 0.5 * self.effective_spread(spread)
        slip = 0.0 if limit else self.slippage(bar_range, q)
        price = float(mid) + s * (half + slip)
        cs = instrument.contract_size
        return FillPrice(price, half * q * cs, slip * q * cs)

    def commission(self, lots: float, *, instrument: Instrument = XAUUSD) -> float:
        """Commission in USD (>= 0) for one side of ``|lots|``."""
        rate = self.commission_per_lot if self.commission_per_lot is not None else instrument.commission_per_lot
        return abs(float(lots)) * float(rate)

    def swap(self, lots: float, nights: float, *, instrument: Instrument = XAUUSD) -> float:
        """Signed financing in USD (+ = received) for holding ``lots`` for ``nights`` nights.

        ``nights`` is the number of *financing nights* already including the triple-swap
        weighting, i.e. the output of :func:`rollover_nights` (a Wednesday rollover = 3).
        """
        lots = float(lots)
        if lots == 0.0 or nights == 0:
            return 0.0
        rate = instrument.swap_long_per_lot if lots > 0 else instrument.swap_short_per_lot
        return abs(lots) * float(rate) * float(nights)

    def swap_between(self, lots: float, start: pd.Timestamp, end: pd.Timestamp, *,
                     instrument: Instrument = XAUUSD) -> float:
        """Signed swap (USD) for holding ``lots`` over the time interval ``(start, end]``."""
        return self.swap(lots, rollover_nights(start, end, instrument), instrument=instrument)

    # ---- convenience ------------------------------------------------------------------------
    def round_trip_cost(self, lots: float, spread: float, bar_range: float = 0.0, *,
                        instrument: Instrument = XAUUSD) -> float:
        """Estimated USD cost of opening and closing ``|lots|`` (market orders both ways),
        excluding swap. Useful for sizers, turnover filters and the LLM execution trader."""
        q = abs(float(lots))
        per_side = (0.5 * self.effective_spread(spread) + self.slippage(bar_range, q)) * q * instrument.contract_size
        return 2.0 * (per_side + self.commission(q, instrument=instrument))


# ---- rollover calendar ------------------------------------------------------------------------
def _weekday_weights(instrument: Instrument) -> np.ndarray:
    """Financing nights charged by the rollover falling on each UTC weekday (Mon=0..Sun=6)."""
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    tsw = int(instrument.triple_swap_weekday)
    if not 0 <= tsw <= 4:
        raise ValueError(f"triple_swap_weekday must be a weekday 0..4, got {tsw}")
    w[tsw] = 3.0
    return w


def _cum_nights(m: np.ndarray, per_day: np.ndarray) -> np.ndarray:
    """S(m) = sum_{k=0}^{m-1} w(k) for epoch-day numbers k (periodic extension for m < 0)."""
    prefix = np.concatenate([[0.0], np.cumsum(per_day)])     # prefix[r] = sum of first r days
    q, r = np.divmod(m, 7)
    return q * prefix[7] + prefix[r]


def rollover_nights_ns(t0_ns: np.ndarray | int, t1_ns: np.ndarray | int,
                       instrument: Instrument = XAUUSD) -> np.ndarray:
    """Vectorised financing nights for holding over ``(t0, t1]`` (int64 ns since epoch, UTC).

    A rollover instant ``R_k = day_k + rollover_hour_utc`` is charged to whoever holds the
    position *immediately before* it, i.e. if ``t0 < R_k <= t1``. With bars that end exactly
    at the rollover this means a position closed at the rollover bar's open is charged and a
    position opened there is not (broker convention: positions open at 23:59:59 server time
    pay swap). Weekend rollovers are free; the triple-swap weekday counts 3 nights.

    The count is computed in O(1) per interval with a periodic cumulative sum, so arbitrary
    gaps (weekends, holidays) are handled exactly. Negative intervals yield 0.
    """
    t0 = np.asarray(t0_ns, dtype=np.int64)
    t1 = np.asarray(t1_ns, dtype=np.int64)
    shift = int(instrument.rollover_hour_utc) * NS_PER_HOUR
    k0 = np.floor_divide(t0 - shift, NS_PER_DAY)
    k1 = np.floor_divide(t1 - shift, NS_PER_DAY)
    w = _weekday_weights(instrument)
    # weight of epoch day k is w[(k + EPOCH_WEEKDAY) % 7]; reorder so index 0 = epoch day 0
    per_day = np.roll(w, -_EPOCH_WEEKDAY)
    nights = _cum_nights(k1 + 1, per_day) - _cum_nights(k0 + 1, per_day)
    return np.maximum(nights, 0.0)


def _to_ns(t: pd.Timestamp | str) -> int:
    ts = pd.Timestamp(t)
    if ts.tz is None:
        raise ValueError("timestamps must be tz-aware (UTC)")
    return int(ts.tz_convert("UTC").as_unit("ns").value)


def rollover_nights(start: pd.Timestamp | str, end: pd.Timestamp | str,
                    instrument: Instrument = XAUUSD) -> float:
    """Financing nights (triple-weighted) for holding a position over ``(start, end]``."""
    return float(rollover_nights_ns(_to_ns(start), _to_ns(end), instrument))
