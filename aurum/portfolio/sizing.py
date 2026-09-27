"""Position sizing: forecast -> signed lots (SPEC §7).

Both sizers implement :class:`aurum.core.interfaces.PositionSizer` and are shared verbatim
by the backtest engine, the RL environment, paper trading and the live runner.

Volatility targeting
    Scaling exposure inversely to forecast volatility keeps ex-ante risk constant through
    calm and turbulent regimes. For gold (and most assets) volatility is far more
    predictable than returns, so vol-managed exposure raises risk-adjusted returns and cuts
    tail risk (Moreira & Muir, 2017, "Volatility-Managed Portfolios", J. Finance 72(4);
    Harvey et al., 2018, "The Impact of Volatility Targeting", J. Portfolio Management).
    Carver (2015, *Systematic Trading*, ch. 9-10) gives the forecast-scaled form used here::

        notional = forecast * target_vol / vol_ann * equity

Turnover control
    A no-trade band around the current position (Carver's "buffering"; see also Garleanu &
    Pedersen, 2013, "Dynamic Trading with Predictable Returns and Transaction Costs") avoids
    paying the spread for tiny rebalances: if
    ``|target - current| < rebalance_band * max(|target|, |current|)`` we keep ``current``.

Drawdown de-risking
    Stepwise exposure cuts after drawdowns (a simple form of CPPI / drawdown control,
    Grossman & Zhou, 1993) limit the risk of ruin when a strategy's edge decays.

All sizes are rounded TOWARD zero onto the instrument's lot grid — rounding never adds risk.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from aurum.core.instrument import Instrument

logger = logging.getLogger(__name__)

DEFAULT_DRAWDOWN_DERISK: tuple[tuple[float, float], ...] = ((0.10, 0.5), (0.15, 0.25))


def _validate_steps(steps: Sequence[Sequence[float]] | None) -> tuple[tuple[float, float], ...]:
    if not steps:
        return ()
    out = []
    for step in steps:
        if len(step) != 2:
            raise ValueError("drawdown_derisk entries must be (drawdown_threshold, multiplier)")
        thr, mult = float(step[0]), float(step[1])
        if not (0.0 < thr < 1.0):
            raise ValueError(f"drawdown threshold must be in (0, 1), got {thr}")
        if not (0.0 <= mult <= 1.0):
            raise ValueError(f"drawdown multiplier must be in [0, 1], got {mult}")
        out.append((thr, mult))
    return tuple(sorted(out))


def drawdown_multiplier(drawdown: float, steps: Sequence[tuple[float, float]]) -> float:
    """Exposure multiplier for the current drawdown: the smallest multiplier among steps
    whose threshold has been reached (steps are cumulative, never re-levering)."""
    dd = abs(float(drawdown)) if math.isfinite(drawdown) else 0.0
    mult = 1.0
    for thr, m in steps:
        if dd >= thr - 1e-12:
            mult = min(mult, m)
    return mult


def _apply_band(target: float, current: float, band: float, hard_cap_lots: float) -> tuple[float, bool]:
    """Keep ``current`` if the trade is inside the no-trade band (and current respects caps)."""
    if band <= 0 or current == 0.0 or target == 0.0:
        return target, False
    if (target > 0) != (current > 0):  # a reversal is never "small"
        return target, False
    if abs(current) > hard_cap_lots + 1e-12:
        return target, False
    if abs(target - current) < band * max(abs(target), abs(current)):
        return current, True
    return target, False


@dataclass
class SizingBreakdown:
    """Step-by-step audit of a sizing decision (for reports, logs and LLM agents)."""

    forecast: float
    vol_ann: float
    equity: float
    price: float
    raw_notional: float = 0.0
    capped_notional: float = 0.0
    drawdown: float = 0.0
    drawdown_mult: float = 1.0
    raw_lots: float = 0.0
    rounded_lots: float = 0.0
    final_lots: float = 0.0
    kept_current: bool = False
    caps_hit: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class VolTargetSizer:
    """Volatility-targeting sizer (SPEC §7).

    Parameters
    ----------
    target_vol       : annualised volatility target for a forecast of +/-1 (0.10 = 10%).
    max_leverage     : cap on ``|notional| / equity``.
    max_lots         : optional absolute cap on lots (the instrument's ``max_lot`` also applies).
    rebalance_band   : relative no-trade band (0.10 = ignore changes smaller than 10%).
    kelly_cap        : optional cap on the position's ex-ante annualised volatility as a
                       fraction of equity. For Gaussian returns the growth-optimal (Kelly)
                       leverage is ``mu / sigma^2``, whose portfolio volatility equals the
                       strategy Sharpe ratio; a half-Kelly policy for an expected Sharpe of
                       0.6 therefore means ``kelly_cap = 0.30``. It bounds
                       ``|notional| * vol_ann / equity`` and guards against a mis-set
                       ``target_vol`` (Thorp, 2006; MacLean, Thorp & Ziemba, 2011).
    drawdown_derisk  : ``((dd_threshold, multiplier), ...)``: exposure multiplier once the
                       drawdown from peak reaches each threshold (lowest multiplier wins).
    min_vol          : floor on the volatility forecast used for sizing (protects against
                       a collapsing vol estimate producing huge positions).
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        max_leverage: float = 2.0,
        max_lots: float | None = None,
        rebalance_band: float = 0.10,
        kelly_cap: float | None = None,
        drawdown_derisk: Sequence[tuple[float, float]] | None = DEFAULT_DRAWDOWN_DERISK,
        *,
        min_vol: float = 0.02,
    ) -> None:
        if not target_vol > 0:
            raise ValueError("target_vol must be positive")
        if not max_leverage > 0:
            raise ValueError("max_leverage must be positive")
        if max_lots is not None and not max_lots > 0:
            raise ValueError("max_lots must be positive or None")
        if not 0.0 <= rebalance_band < 1.0:
            raise ValueError("rebalance_band must be in [0, 1)")
        if kelly_cap is not None and not kelly_cap > 0:
            raise ValueError("kelly_cap must be positive or None")
        if not min_vol > 0:
            raise ValueError("min_vol must be positive")
        self.target_vol = float(target_vol)
        self.max_leverage = float(max_leverage)
        self.max_lots = None if max_lots is None else float(max_lots)
        self.rebalance_band = float(rebalance_band)
        self.kelly_cap = None if kelly_cap is None else float(kelly_cap)
        self.drawdown_derisk = _validate_steps(drawdown_derisk)
        self.min_vol = float(min_vol)

    # ------------------------------------------------------------------------------------
    def breakdown(
        self,
        forecast: float,
        vol_ann: float,
        equity: float,
        price: float,
        instrument: Instrument,
        *,
        current_lots: float = 0.0,
        drawdown: float = 0.0,
    ) -> SizingBreakdown:
        """Full audit trail of :meth:`target_lots` (same arithmetic)."""
        bd = SizingBreakdown(forecast=float(forecast), vol_ann=float(vol_ann), equity=float(equity),
                             price=float(price), drawdown=float(drawdown))
        f = float(forecast)
        if not math.isfinite(f) or f == 0.0:
            bd.note = "zero or non-finite forecast -> flat"
            return bd
        f = max(-1.0, min(1.0, f))
        if not (math.isfinite(equity) and equity > 0):
            bd.note = "non-positive equity -> flat"
            logger.warning("VolTargetSizer: non-positive equity %s -> 0 lots", equity)
            return bd
        if not (math.isfinite(price) and price > 0):
            bd.note = "invalid price -> flat"
            logger.warning("VolTargetSizer: invalid price %s -> 0 lots", price)
            return bd
        if not math.isfinite(vol_ann) or vol_ann <= 0:
            bd.note = "invalid volatility forecast -> flat"
            logger.warning("VolTargetSizer: invalid vol_ann %s -> 0 lots", vol_ann)
            return bd
        vol = max(float(vol_ann), self.min_vol)
        if vol > vol_ann:
            bd.caps_hit.append(f"min_vol {self.min_vol:.3f}")

        notional = f * self.target_vol / vol * equity
        bd.raw_notional = notional
        cap_notional = self.max_leverage * equity
        cap_name = f"max_leverage {self.max_leverage:.2f}"
        if self.kelly_cap is not None:
            kelly_notional = self.kelly_cap * equity / vol
            if kelly_notional < cap_notional:
                cap_notional = kelly_notional
                cap_name = f"kelly_cap {self.kelly_cap:.3f}"
        if abs(notional) > cap_notional:
            bd.caps_hit.append(cap_name)
            notional = math.copysign(cap_notional, notional)
        bd.capped_notional = notional

        mult = drawdown_multiplier(drawdown, self.drawdown_derisk)
        bd.drawdown_mult = mult
        notional *= mult

        lots = notional / (instrument.contract_size * price)
        hard_cap = instrument.max_lot if self.max_lots is None else min(self.max_lots, instrument.max_lot)
        # Leverage cap in lots (for the band check on the current position).
        lev_cap_lots = cap_notional * mult / (instrument.contract_size * price)
        hard_cap_lots = min(hard_cap, lev_cap_lots)
        if abs(lots) > hard_cap:
            bd.caps_hit.append(f"max_lots {hard_cap:g}")
            lots = math.copysign(hard_cap, lots)
        bd.raw_lots = lots
        rounded = instrument.round_lots(lots)
        bd.rounded_lots = rounded

        final, kept = _apply_band(rounded, float(current_lots), self.rebalance_band, hard_cap_lots)
        if kept:
            final = instrument.round_lots(final)
        bd.kept_current = kept
        bd.final_lots = final
        return bd

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
        """Signed lots to hold after the next fill (rounded toward zero on the lot grid)."""
        return self.breakdown(
            forecast, vol_ann, equity, price, instrument, current_lots=current_lots, drawdown=drawdown
        ).final_lots

    def __repr__(self) -> str:
        return (
            f"VolTargetSizer(target_vol={self.target_vol}, max_leverage={self.max_leverage}, "
            f"max_lots={self.max_lots}, rebalance_band={self.rebalance_band}, "
            f"kelly_cap={self.kelly_cap}, drawdown_derisk={self.drawdown_derisk})"
        )


class FixedFractionalSizer:
    """Stop-based fixed-fractional sizing: risk a fixed fraction of equity per trade.

    ``lots = |forecast| * risk_per_trade * equity / (stop_distance * contract_size)`` with
    ``stop_distance = stop_atr * ATR``. If the caller does not pass ``atr=`` the ATR is
    proxied from the volatility forecast as ``price * vol_ann / sqrt(atr_periods_per_year)``
    — i.e. a one-period price standard deviation on the ATR's own timeframe (default daily,
    252 periods/year). Use :meth:`stop_distance` to place the matching protective stop.

    Fixed-fractional sizing (Vince, 1990; Tharp) bounds the loss per trade at roughly
    ``risk_per_trade`` of equity if the stop is honoured; gap risk (weekend, news) can exceed
    it, which is why the risk manager and ``aurum.risk.var.gap_shock`` exist.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.005,
        stop_atr: float = 2.0,
        *,
        max_leverage: float = 2.0,
        max_lots: float | None = None,
        rebalance_band: float = 0.10,
        drawdown_derisk: Sequence[tuple[float, float]] | None = DEFAULT_DRAWDOWN_DERISK,
        atr_periods_per_year: float = 252.0,
    ) -> None:
        if not 0 < risk_per_trade < 0.2:
            raise ValueError("risk_per_trade must be in (0, 0.2)")
        if not stop_atr > 0:
            raise ValueError("stop_atr must be positive")
        if not max_leverage > 0:
            raise ValueError("max_leverage must be positive")
        if not 0.0 <= rebalance_band < 1.0:
            raise ValueError("rebalance_band must be in [0, 1)")
        self.risk_per_trade = float(risk_per_trade)
        self.stop_atr = float(stop_atr)
        self.max_leverage = float(max_leverage)
        self.max_lots = None if max_lots is None else float(max_lots)
        self.rebalance_band = float(rebalance_band)
        self.drawdown_derisk = _validate_steps(drawdown_derisk)
        self.atr_periods_per_year = float(atr_periods_per_year)

    def stop_distance(self, vol_ann: float, price: float, *, atr: float | None = None) -> float:
        """Protective-stop distance in price units (``stop_atr * ATR``)."""
        if atr is not None and math.isfinite(atr) and atr > 0:
            base = float(atr)
        else:
            if not (math.isfinite(vol_ann) and vol_ann > 0 and math.isfinite(price) and price > 0):
                return math.nan
            base = price * vol_ann / math.sqrt(self.atr_periods_per_year)
        return self.stop_atr * base

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
        atr: float | None = None,
    ) -> float:
        """Signed lots such that hitting the stop loses ``|forecast| * risk_per_trade`` of equity."""
        f = float(forecast)
        if not math.isfinite(f) or f == 0.0:
            return 0.0
        f = max(-1.0, min(1.0, f))
        if not (math.isfinite(equity) and equity > 0 and math.isfinite(price) and price > 0):
            logger.warning("FixedFractionalSizer: invalid equity/price (%s, %s) -> 0 lots", equity, price)
            return 0.0
        dist = self.stop_distance(vol_ann, price, atr=atr)
        if not (math.isfinite(dist) and dist > 0):
            logger.warning("FixedFractionalSizer: cannot compute stop distance -> 0 lots")
            return 0.0
        risk_usd = abs(f) * self.risk_per_trade * equity * drawdown_multiplier(drawdown, self.drawdown_derisk)
        lots = math.copysign(risk_usd / (dist * instrument.contract_size), f)
        lev_cap_lots = self.max_leverage * equity / (instrument.contract_size * price)
        hard_cap = instrument.max_lot if self.max_lots is None else min(self.max_lots, instrument.max_lot)
        cap = min(lev_cap_lots, hard_cap)
        if abs(lots) > cap:
            lots = math.copysign(cap, lots)
        rounded = instrument.round_lots(lots)
        final, kept = _apply_band(rounded, float(current_lots), self.rebalance_band, cap)
        return instrument.round_lots(final) if kept else final

    def __repr__(self) -> str:
        return (
            f"FixedFractionalSizer(risk_per_trade={self.risk_per_trade}, stop_atr={self.stop_atr}, "
            f"max_leverage={self.max_leverage}, max_lots={self.max_lots})"
        )
