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
  (``instrument.rollover_hour_utc`` on a weekday) are charged or credited overnight
  financing by :class:`FinancingModel` (``CostModel.financing``). The rollover on
  ``instrument.triple_swap_weekday`` (Wednesday by convention) counts three nights because
  spot settlement (T+2) of that day's roll spans the weekend. Weekend (Sat/Sun) "rollovers"
  do not exist. See :func:`rollover_nights` for the exact interval semantics.

Financing modes
---------------
``"rate"`` (default)
    Interest on the position's notional, the way a spot-FX-style XAU/USD position is
    actually financed (and MT5's ``SYMBOL_SWAP_MODE_INTEREST_CURRENT``: annual rate applied
    to the current price at the rollover). A long XAUUSD position is long gold / short USD:
    it pays the USD benchmark rate, earns the gold lease rate, and the broker adds its
    markup. Per financing night (``day_count = 360``, the USD money-market convention)::

        long  (lots > 0): -lots * contract_size * P * (r - lease + markup_long)  / 360
        short (lots < 0): -lots * contract_size * P * (r - lease - markup_short) / 360

    i.e. a short *receives* ``r - lease - markup_short`` (negative -> it pays too, which is
    what near-zero-rate years look like at a retail broker). ``P`` is the MID price at the
    rollover (the last bar close at or before it; see the simulator) and ``r`` the benchmark
    rate *as of the rollover instant*: the latest observation of ``rate_series`` (default
    ``md.macro["fedfunds"]``, FRED DFF in percent) whose ``available_at <= R``. A rate
    published after the rollover is never used (SPEC §0.1); before the first observation, or
    without a series, ``fallback_rate`` applies (logged). Defaults: ``markup_long =
    markup_short = 0.025`` (2.5%/yr: typical retail CFD brokers charge the benchmark +/- 2-3%
    on metals/FX CFDs; institutional prime brokers ~0.5-1%), ``lease_rate = 0`` (gold lease
    rates have been ~0-0.5%/yr outside short squeezes; set it if you want the gold carry),
    ``fallback_rate = 0.03`` (roughly the 1990-2025 average effective fed funds rate).
``"fixed"``
    Broker-quoted per-lot swaps: ``instrument.swap_{long,short}_per_lot`` USD per lot per
    night (the pre-2026 behaviour, bit-identical; use it to replicate a specific broker's
    swap table).
``"none"``
    No financing at all (frictionless benchmarks, :meth:`CostModel.zero`).

All cost values are positive USD amounts (a cost), swap is signed (+ = received).
Fill prices are *not* rounded to the tick grid: the rounding effect (<= $0.005/oz) is noise
compared with the modelled spread and keeps the PnL identity exact.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from aurum.core.instrument import XAUUSD, Instrument

logger = logging.getLogger(__name__)

__all__ = [
    "FINANCING_MODES",
    "CostModel",
    "FillPrice",
    "FinancingModel",
    "RateCurve",
    "RateSource",
    "rollover_nights",
    "rollover_nights_ns",
]

NS_PER_DAY: int = 86_400 * 10**9
NS_PER_HOUR: int = 3_600 * 10**9
# 1970-01-01 (day 0 of the epoch) was a Thursday.
_EPOCH_WEEKDAY: int = 3

FINANCING_MODES: tuple[str, ...] = ("rate", "fixed", "none")
_RATE_UNITS: dict[str, float] = {"percent": 0.01, "fraction": 1.0, "bps": 1e-4}
_WARNED: set[tuple] = set()


def _warn_once(key: tuple, msg: str, *args: Any) -> None:
    """Log a WARNING once per process for ``key`` (simulators are rebuilt per fold/episode)."""
    if key in _WARNED:
        logger.debug(msg, *args)
        return
    _WARNED.add(key)
    logger.warning(msg, *args)


class FillPrice(NamedTuple):
    """Result of :meth:`CostModel.fill_price`: executed price and the USD cost split."""

    price: float            # executed price (USD/oz), includes half-spread and slippage
    spread_cost: float      # USD, >= 0: half effective spread * |lots| * contract_size
    slippage_cost: float    # USD, >= 0: slippage per oz * |lots| * contract_size


def _is_real(v: object) -> bool:
    """A finite real number (not a bool)."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(float(v))


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


# ---- point-in-time benchmark rate --------------------------------------------------------------
class RateCurve:
    """A point-in-time step curve of an annual benchmark rate (as a FRACTION, 0.05 = 5%).

    ``available_ns`` (sorted, unique int64 ns UTC) are the instants from which each value may
    be used; :meth:`asof_ns` returns, for each query instant ``t``, the value with the latest
    ``available_at <= t`` (NaN before the first one) — the rule of
    :func:`aurum.data.pit.asof_join`, including "the last row wins" when several rows become
    available at the same instant (e.g. FRED's Friday/Saturday/Sunday DFF rows).
    """

    __slots__ = ("available_ns", "name", "values")

    def __init__(self, available_ns: np.ndarray, values: np.ndarray, name: str = "") -> None:
        a = np.asarray(available_ns, dtype=np.int64)
        v = np.asarray(values, dtype=float)
        if a.shape != v.shape or a.ndim != 1:
            raise ValueError("available_ns and values must be 1-d arrays of equal length")
        if len(a) > 1 and not (np.diff(a) > 0).all():
            raise ValueError("available_ns must be strictly increasing")
        if not np.isfinite(v).all():
            raise ValueError("rate values must be finite")
        self.available_ns = a
        self.values = v
        self.name = str(name)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        if not len(self):
            return f"RateCurve({self.name!r}, empty)"
        return (f"RateCurve({self.name!r}, n={len(self)}, {self.first_available} .. "
                f"{self.last_available})")

    @property
    def first_available(self) -> pd.Timestamp | None:
        return pd.Timestamp(int(self.available_ns[0]), tz="UTC") if len(self) else None

    @property
    def last_available(self) -> pd.Timestamp | None:
        return pd.Timestamp(int(self.available_ns[-1]), tz="UTC") if len(self) else None

    def asof_ns(self, t_ns: np.ndarray | int) -> np.ndarray:
        """Latest value with ``available_at <= t`` for each ``t`` (NaN where none yet)."""
        t = np.asarray(t_ns, dtype=np.int64)
        if not len(self):
            return np.full(t.shape, np.nan)
        pos = np.searchsorted(self.available_ns, t, side="right") - 1
        out = self.values[np.clip(pos, 0, None)]
        return np.where(pos >= 0, out, np.nan)

    @classmethod
    def from_source(cls, source: RateSource, *, name: str = "fedfunds",
                    unit: str = "percent") -> RateCurve | None:
        """Build a curve from a rate source (``None`` if it holds no usable series).

        ``source`` may be a ``RateCurve`` (returned as is), a mapping of macro frames
        (``md.macro``; ``source[name]`` is used), a frame with ``available_at`` and a
        ``value`` column (the ``aurum.data.macro`` format), or a Series indexed by the
        tz-aware instant each value became AVAILABLE. ``unit`` converts values to fractions
        (``"percent"``: /100, ``"fraction"``, ``"bps"``: /10000). A frame without
        ``available_at`` is rejected: its values could not be used point-in-time.
        """
        if source is None:
            return None
        if isinstance(source, RateCurve):
            return source
        if unit not in _RATE_UNITS:
            raise ValueError(f"rate unit must be one of {sorted(_RATE_UNITS)}, got {unit!r}")
        scale = _RATE_UNITS[unit]
        label = name
        if isinstance(source, Mapping) and not isinstance(source, (pd.DataFrame, pd.Series)):
            frame = source.get(name)
            if frame is None:
                return None
            source = frame
        if isinstance(source, pd.DataFrame):
            if "available_at" not in source.columns:
                raise ValueError(f"rate series {label!r} has no 'available_at' column: it cannot be "
                                 "used point-in-time")
            if "value" in source.columns:
                col = "value"
            else:
                num = [c for c in source.columns if c != "available_at"
                       and pd.api.types.is_numeric_dtype(source[c])]
                if len(num) != 1:
                    raise ValueError(f"rate series {label!r}: expected a 'value' column")
                col = num[0]
            frame = source
            if isinstance(frame.index, pd.DatetimeIndex) and not frame.index.is_monotonic_increasing:
                frame = frame.sort_index(kind="stable")   # ties on available_at -> latest observation
            avail = pd.DatetimeIndex(frame["available_at"])
            vals = frame[col].to_numpy(dtype=float)
        elif isinstance(source, pd.Series):
            avail = pd.DatetimeIndex(source.index)
            vals = source.to_numpy(dtype=float)
        else:
            raise TypeError(f"unsupported rate source {type(source).__name__} (use md.macro, a macro frame "
                            "with available_at, a Series indexed by availability time, or a RateCurve)")
        if avail.tz is None and len(avail.dropna()):
            raise ValueError(f"rate series {label!r}: availability times must be tz-aware (UTC)")
        if avail.tz is None:
            avail = avail.tz_localize("UTC")
        ok = np.asarray(~avail.isna()) & np.isfinite(vals)
        ns = avail.tz_convert("UTC").as_unit("ns").asi8[ok]
        vals = vals[ok] * scale
        order = np.argsort(ns, kind="stable")
        ns, vals = ns[order], vals[order]
        if len(ns) > 1:  # several rows available at the same instant: keep the last (latest info)
            last = np.r_[ns[1:] != ns[:-1], True]
            ns, vals = ns[last], vals[last]
        return cls(ns, vals, name=label)


RateSource = RateCurve | pd.DataFrame | pd.Series | Mapping[str, pd.DataFrame] | None


# ---- financing -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FinancingModel:
    """Overnight financing of CFD positions (see the module docstring, "Financing modes").

    Parameters
    ----------
    mode          : ``"rate"`` (benchmark rate +/- markup on notional), ``"fixed"``
                    (``instrument.swap_{long,short}_per_lot`` USD/lot/night) or ``"none"``.
    markup_long   : annual broker markup added to the rate a LONG pays (fraction, 0.025 = 2.5%).
    markup_short  : annual markup deducted from the rate a SHORT receives.
    lease_rate    : annual gold lease rate (fraction) earned by longs / paid by shorts.
    rate_series   : name of the benchmark series in ``md.macro`` (percent by default).
    rate_unit     : ``"percent"`` | ``"fraction"`` | ``"bps"`` — unit of that series.
    fallback_rate : annual benchmark (fraction) used before the series starts / without it.
    day_count     : days per year of the rate convention (360 for USD money markets).
    """

    mode: str = "rate"
    markup_long: float = 0.025
    markup_short: float = 0.025
    lease_rate: float = 0.0
    rate_series: str = "fedfunds"
    rate_unit: str = "percent"
    fallback_rate: float = 0.03
    day_count: float = 360.0

    def __post_init__(self) -> None:
        if self.mode not in FINANCING_MODES:
            raise ValueError(f"FinancingModel.mode must be one of {FINANCING_MODES}, got {self.mode!r}")
        for name in ("markup_long", "markup_short"):
            v = getattr(self, name)
            if not (_is_real(v) and 0.0 <= v < 1.0):
                raise ValueError(f"FinancingModel.{name} must be an annual fraction in [0, 1), got {v!r}")
        for name in ("lease_rate", "fallback_rate"):
            v = getattr(self, name)
            if not (_is_real(v) and abs(v) < 1.0):
                raise ValueError(f"FinancingModel.{name} must be an annual fraction in (-1, 1), got {v!r}")
        if not (_is_real(self.day_count) and self.day_count > 0):
            raise ValueError(f"FinancingModel.day_count must be positive, got {self.day_count!r}")
        if self.rate_unit not in _RATE_UNITS:
            raise ValueError(f"FinancingModel.rate_unit must be one of {sorted(_RATE_UNITS)}, "
                             f"got {self.rate_unit!r}")
        if not isinstance(self.rate_series, str) or not self.rate_series:
            raise ValueError("FinancingModel.rate_series must be a non-empty series name")

    # ---- constructors ----------------------------------------------------------------------
    @classmethod
    def fixed(cls) -> FinancingModel:
        """Per-lot swaps from the instrument (the legacy model)."""
        return cls(mode="fixed")

    @classmethod
    def none(cls) -> FinancingModel:
        """No financing at all."""
        return cls(mode="none")

    @classmethod
    def coerce(cls, value: FinancingModel | Mapping[str, Any] | str | None) -> FinancingModel:
        """``FinancingModel`` from a model, a kwargs mapping (YAML), a mode name or ``None``
        (the default model)."""
        if value is None:
            return cls()
        if isinstance(value, FinancingModel):
            return value
        if isinstance(value, str):
            return cls(mode=value)
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"financing must be a FinancingModel, a mapping or a mode name, got "
                        f"{type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- rates -------------------------------------------------------------------------------
    @property
    def uses_rates(self) -> bool:
        return self.mode == "rate"

    def curve(self, rates: RateSource) -> RateCurve | None:
        """Resolve ``rates`` (``md.macro``, a macro frame, a Series or a RateCurve) into the
        point-in-time benchmark curve this model reads (``None`` outside ``"rate"`` mode or
        when the source holds no ``rate_series``)."""
        if self.mode != "rate":
            return None
        return RateCurve.from_source(rates, name=self.rate_series, unit=self.rate_unit)

    def benchmark_ns(self, t_ns: np.ndarray | int, curve: RateCurve | None) -> np.ndarray:
        """Annual benchmark rate (fraction) as of each instant (``fallback_rate`` where the
        curve has no observation available yet, or without a curve)."""
        t = np.asarray(t_ns, dtype=np.int64)
        if curve is None:
            return np.full(t.shape, float(self.fallback_rate))
        r = curve.asof_ns(t)
        return np.where(np.isnan(r), float(self.fallback_rate), r)

    def rate_nights_ns(self, t0_ns: np.ndarray | int, t1_ns: np.ndarray | int,
                       instrument: Instrument = XAUUSD, curve: RateCurve | None = None) -> np.ndarray:
        """Rate-weighted financing nights over ``(t0, t1]``: ``sum_R w(R) * r(R)`` over the
        rollover instants ``R`` in the interval (``w`` = 1, 3 on the triple-swap weekday, 0 on
        weekends — exactly the nights of :func:`rollover_nights_ns`; ``r(R)`` = benchmark as
        of ``R``). Intervals holding one rollover (every bar up to D1) are computed directly,
        longer ones by an explicit sum in time order, so the result for a rollover does not
        depend on how the timeline is cut into intervals (simulator vs paper broker parity).
        """
        t0 = np.asarray(t0_ns, dtype=np.int64)
        t1 = np.asarray(t1_ns, dtype=np.int64)
        shape = np.broadcast(t0, t1).shape
        t0 = np.broadcast_to(t0, shape).ravel()
        t1 = np.broadcast_to(t1, shape).ravel()
        shift = int(instrument.rollover_hour_utc) * NS_PER_HOUR
        k0 = np.floor_divide(t0 - shift, NS_PER_DAY)
        k1 = np.floor_divide(t1 - shift, NS_PER_DAY)
        per_day = np.roll(_weekday_weights(instrument), -_EPOCH_WEEKDAY)
        out = np.zeros(t0.shape)
        span = k1 - k0
        one = np.flatnonzero(span == 1)
        if len(one):
            k = k1[one]
            w = per_day[np.mod(k, 7)]
            live = w > 0
            if live.any():
                r = self.benchmark_ns(k[live] * NS_PER_DAY + shift, curve)
                vals = np.zeros(len(k))
                vals[live] = w[live] * r
                out[one] = vals
        for j in np.flatnonzero(span > 1):
            ks = np.arange(k0[j] + 1, k1[j] + 1, dtype=np.int64)
            w = per_day[np.mod(ks, 7)]
            live = w > 0
            if not live.any():
                continue
            r = self.benchmark_ns(ks[live] * NS_PER_DAY + shift, curve)
            acc = 0.0
            for wi, ri in zip(w[live].tolist(), r.tolist(), strict=True):
                acc += wi * ri
            out[j] = acc
        return out.reshape(shape)

    # ---- USD amounts -----------------------------------------------------------------------------
    def annual_rate(self, lots: float, benchmark: float) -> float:
        """Annual rate applied to the SIGNED notional (financing = -notional * rate / day_count
        per night): ``r - lease + markup_long`` for a long, ``r - lease - markup_short`` for a
        short (so a short receives it when positive and pays when negative)."""
        base = float(benchmark) - self.lease_rate
        return base + self.markup_long if lots > 0 else base - self.markup_short

    def amount(self, lots: float, nights: float, *, price: float | None = None,
               rate_nights: float | None = None, instrument: Instrument = XAUUSD) -> float:
        """Signed financing in USD (+ = received) for holding ``lots`` over ``nights``
        financing nights (already triple-weighted, :func:`rollover_nights`).

        ``"rate"`` mode needs ``price`` (mid at the rollover) and uses ``rate_nights``
        (:meth:`rate_nights_ns`; default ``nights * fallback_rate``)::

            -lots * contract_size * price * (rate_nights + (markup_side - lease) * nights) / day_count

        with ``markup_side = +markup_long`` for longs and ``-markup_short`` for shorts.
        """
        lots = float(lots)
        if lots == 0.0 or nights == 0:
            return 0.0
        mode = self.mode
        if mode == "fixed":
            rate = instrument.swap_long_per_lot if lots > 0 else instrument.swap_short_per_lot
            return abs(lots) * float(rate) * float(nights)
        if mode == "none":
            return 0.0
        if price is None or not math.isfinite(float(price)):
            raise ValueError("rate-based financing needs the (finite) mid price at the rollover")
        n = float(nights)
        rn = n * self.fallback_rate if rate_nights is None else float(rate_nights)
        m = self.markup_long if lots > 0 else -self.markup_short
        return -lots * instrument.contract_size * float(price) * (rn + (m - self.lease_rate) * n) / self.day_count

    def nightly(self, lots: float, price: float, benchmark: float, *,
                instrument: Instrument = XAUUSD) -> float:
        """Signed USD financing of ONE night at ``benchmark`` (convenience for reports/agents)."""
        return self.amount(lots, 1.0, price=price, rate_nights=float(benchmark), instrument=instrument)


@dataclass(frozen=True)
class CostModel:
    """Spread / slippage / commission / financing model (see module docstring).

    Parameters
    ----------
    spread_multiplier : scale applied to the bar's quoted spread (stress-test with 1.5-2.0).
    min_spread        : floor on the effective spread in USD/oz.
    slippage_fixed    : constant adverse slippage per fill in USD/oz.
    slippage_range_frac : fraction of the execution bar's high-low range added as slippage.
    impact_coef       : square-root impact coefficient, USD/oz per sqrt(lot).
    commission_per_lot : USD per lot per side; ``None`` uses ``instrument.commission_per_lot``.
    financing         : :class:`FinancingModel` (a kwargs mapping or a mode name is accepted,
                        e.g. from YAML); default: rate-based financing.
    """

    spread_multiplier: float = 1.0
    min_spread: float = 0.10
    slippage_fixed: float = 0.02
    slippage_range_frac: float = 0.02
    impact_coef: float = 0.0
    commission_per_lot: float | None = None
    financing: FinancingModel = field(default_factory=FinancingModel)

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
        if not isinstance(self.financing, FinancingModel):
            object.__setattr__(self, "financing", FinancingModel.coerce(self.financing))

    # ---- constructors ----------------------------------------------------------------------
    @classmethod
    def zero(cls) -> CostModel:
        """A frictionless model: no spread, slippage, commission or financing."""
        return cls(spread_multiplier=0.0, min_spread=0.0, slippage_fixed=0.0,
                   slippage_range_frac=0.0, impact_coef=0.0, commission_per_lot=0.0,
                   financing=FinancingModel.none())

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

    def swap(self, lots: float, nights: float, *, instrument: Instrument = XAUUSD,
             price: float | None = None, rate_nights: float | None = None) -> float:
        """Signed financing in USD (+ = received) for holding ``lots`` for ``nights`` nights.

        ``nights`` is the number of *financing nights* already including the triple-swap
        weighting, i.e. the output of :func:`rollover_nights` (a Wednesday rollover = 3).
        Delegates to :meth:`FinancingModel.amount`: ``"fixed"`` uses the instrument's per-lot
        swaps, ``"rate"`` needs ``price`` (mid at the rollover) and ``rate_nights``
        (:meth:`FinancingModel.rate_nights_ns`, default ``nights * fallback_rate``).
        """
        return self.financing.amount(lots, nights, price=price, rate_nights=rate_nights,
                                     instrument=instrument)

    def swap_between(self, lots: float, start: pd.Timestamp, end: pd.Timestamp, *,
                     instrument: Instrument = XAUUSD, price: float | None = None,
                     rates: RateSource = None) -> float:
        """Signed swap (USD) for holding ``lots`` over the time interval ``(start, end]``
        (``"rate"`` mode: at a constant mid ``price``, benchmark as of each rollover)."""
        t0, t1 = _to_ns(start), _to_ns(end)
        nights = float(rollover_nights_ns(t0, t1, instrument))
        rn = None
        if self.financing.uses_rates and nights:
            curve = self.financing.curve(rates)
            rn = float(self.financing.rate_nights_ns(t0, t1, instrument, curve))
        return self.swap(lots, nights, instrument=instrument, price=price, rate_nights=rn)

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
