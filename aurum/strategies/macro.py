"""Macro strategies: ``macro_factor`` (dollar & real-yield momentum) and ``risk_off`` (VIX spikes).

Economic rationale
------------------
Gold is a non-yielding, dollar-denominated real asset. Its best documented macro drivers
are (i) US real interest rates — the opportunity cost of holding gold (Erb & Harvey 2013;
Barsky & Summers 1988 "Gibson's paradox") — and (ii) the US dollar, through the numeraire
effect and global purchasing power (Capie, Mills & Wood 2005). Both drivers trend
(monetary-policy cycles are persistent), so *momentum* in the dollar and in real yields
translates into slow-moving pressure on gold. Separately, gold is a short-lived safe haven
in equity stress (Baur & Lucey 2010; Baur & McDermott 2010): acute jumps in implied
volatility are followed by gold strength over the next days-to-weeks — except in "dash for
cash" liquidity squeezes (Oct 2008, Mar 2020) when the dollar spikes and gold is sold to
meet margin calls.

Point-in-time discipline (SPEC §0-1)
-----------------------------------
Every transformation is computed on each macro series' own observation sequence (a
derived row uses observations up to and including that row only). A derived row is
available at the running maximum of the ``available_at`` of its inputs (``cummax`` —
robust to vendors with non-monotone publication times), and is mapped onto trading bars
with :func:`aurum.data.pit.asof_join` against ``bars["available_at"]`` with a staleness
tolerance, so a dead feed decays to "no signal" rather than being carried forever.
Yahoo closes are available the same evening (21:30/22:30 UTC), FRED series the next day.

Warm-up is governed by the macro history (not by the number of bars), so
``warmup_bars == 0``; rows without enough macro history are simply 0.

References
----------
* Erb, C. & Harvey, C. (2013). "The Golden Dilemma". Financial Analysts Journal 69(4).
* Barsky, R. & Summers, L. (1988). "Gibson's Paradox and the Gold Standard". JPE 96(3).
* Capie, F., Mills, T. & Wood, G. (2005). "Gold as a hedge against the dollar". J. Int.
  Financial Markets, Institutions & Money 15(4).
* Baur, D. & Lucey, B. (2010). "Is Gold a Hedge or a Safe Haven? An Analysis of Stocks,
  Bonds and Gold". Financial Review 45(2).
* Baur, D. & McDermott, T. (2010). "Is gold a safe haven? International evidence".
  J. Banking & Finance 34(8).
* Whaley, R. (2000). "The Investor Fear Gauge". J. Portfolio Management 26(3).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.data.pit import asof_join
from aurum.features.macro import DEFAULT_YIELD_SERIES
from aurum.features.volatility import log_pos, safe_div
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import (
    check_params,
    diversification_multiplier,
    ewma_mean,
    latch,
    overlap_correlation,
    z_to_forecast,
)

logger = logging.getLogger(__name__)

__all__ = ["MacroFactor", "RiskOff", "clean_macro", "is_yield_series", "macro_momentum_z"]


def _utc(values: Iterable) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def is_yield_series(name: str, frame: pd.DataFrame) -> bool:
    """Yields/rates (quoted in percent) use level differences; prices use log changes.

    Decided from the name / ``attrs["kind"]`` only, never from the values (a data-dependent
    choice would be a full-sample decision).
    """
    kind = getattr(frame, "attrs", {}).get("kind")
    if kind in ("yield", "rate"):
        return True
    if kind == "price":
        return False
    return name in DEFAULT_YIELD_SERIES


def clean_macro(frame: pd.DataFrame, *, log_values: bool) -> pd.DataFrame:
    """Normalise a macro frame to ``x`` (log value or level) + monotone ``available_at``.

    Sorted by observation date, duplicate dates keep the last print, unusable rows dropped
    (non-positive prints for log series), and ``available_at`` replaced by its running
    maximum so a derived value is never available before any of its inputs.
    """
    if "value" not in frame.columns or "available_at" not in frame.columns:
        raise KeyError("macro frame needs 'value' and 'available_at' columns")
    v = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    f = pd.DataFrame({"x": log_pos(v) if log_values else v}, index=_utc(frame.index))
    f["available_at"] = _utc(frame["available_at"]).as_unit("ns")
    f = f.sort_index(kind="stable")
    f = f[~f.index.duplicated(keep="last")]
    f = f.dropna(subset=["x", "available_at"])
    f = f[np.isfinite(f["x"].to_numpy())]
    if len(f):
        ns = pd.DatetimeIndex(f["available_at"]).asi8
        f["available_at"] = pd.DatetimeIndex(
            np.maximum.accumulate(ns).view("datetime64[ns]")).tz_localize("UTC")
    return f


def macro_momentum_z(x: pd.Series, horizons: tuple[int, ...], vol_halflife: float,
                     vol_min_obs: int) -> pd.Series:
    """Multi-horizon vol-normalised momentum of a daily macro series (~N(0,1) under the null).

    ``z_k = (x_t - x_{t-k}) / (sigma_t sqrt(k))`` with ``sigma`` the zero-mean EWMA std of
    one-observation changes (normalised EWMA, not seeded with the first change — see
    :func:`aurum.strategies.trend.ewma_mean`); horizons are equally weighted and rescaled
    by the analytic diversification multiplier of overlapping horizons.
    """
    d = x.diff()
    sigma = np.sqrt(ewma_mean(d * d, vol_halflife, vol_min_obs))
    sigma = sigma.where(sigma > 0)
    w = np.full(len(horizons), 1.0 / len(horizons))
    fdm = diversification_multiplier(overlap_correlation(horizons), w)
    total = pd.Series(0.0, index=x.index)
    for wi, k in zip(w, horizons, strict=True):
        z = safe_div(x - x.shift(k), sigma * math.sqrt(k))
        total = total + wi * z
    return total * fdm


def _warn_once(strategy: Strategy, key: str, msg: str, *args: Any) -> None:
    """Log a WARNING once per strategy instance and ``key`` (live calls generate every bar)."""
    seen: set[str] = strategy.__dict__.setdefault("_warned", set())
    if key not in seen:
        seen.add(key)
        logger.warning(msg, *args)


def _check_latest(strategy: Strategy, name: str, values: np.ndarray, stale_days: float) -> None:
    """Silent-failure guard: a configured series whose value at the LATEST bar (the live
    decision) is unknown — feed older than ``stale_days`` or too little history — silently
    drops out of the signal; say so."""
    if len(values) and not np.isfinite(values[-1]):
        _warn_once(strategy, f"stale:{name}",
                   "%s: series %r has no usable value at the latest bar (stale > %s days or "
                   "too little history); it contributes nothing", strategy.name, name, stale_days)


def _join(bars: pd.DataFrame, derived: pd.DataFrame, column: str, stale_days: float) -> np.ndarray:
    """``derived[column]`` as of each bar's decision time (``available_at``), NaN if stale."""
    if len(derived) == 0:
        return np.full(len(bars), np.nan)
    tol = pd.Timedelta(days=float(stale_days)) if stale_days else None
    out = asof_join(bars["available_at"], derived[[column, "available_at"]], columns=[column],
                    tolerance=tol)
    return out[column].to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------
# macro_factor
# ---------------------------------------------------------------------------------------
@register_strategy
class MacroFactor(Strategy):
    """Gold vs dollar and real-yield momentum: long gold when the USD and real yields fall.

    Rationale: real yields are gold's opportunity cost and the dollar its numeraire
    (Erb & Harvey 2013; Capie, Mills & Wood 2005). Both follow persistent policy-driven
    trends, so recent declines in DXY and in 10-year TIPS yields forecast continued support
    for gold (and rises forecast pressure). Using *momentum* of the drivers rather than
    contemporaneous changes keeps the rule strictly point-in-time.

    Signal: for each series in ``series`` (name → sign of gold's exposure, default
    ``{"dxy": -1, "real10y": -1}``), the multi-horizon vol-normalised momentum on its own
    daily observations (log changes for prices, level changes for yields — see
    :func:`macro_momentum_z`), aligned to bars as of ``available_at`` with a
    ``stale_days`` tolerance. The signed z's are combined as
    ``sum(w_i z_i) / sqrt(sum w_i^2)`` over the series available at each bar (an ~N(0,1)
    composite if the drivers were independent; DXY and real-yield momentum are positively
    correlated, so the composite is somewhat more dispersed), then Carver-scaled.
    Missing series are skipped (logged); with none available the forecast is 0.
    Defaults: horizons 5/20/60 observations (1 week, 1 month, 1 quarter), EWMA vol
    half-life 60 obs, 10-day staleness tolerance.

    References: Erb & Harvey (2013) FAJ 69(4); Capie, Mills & Wood (2005) JIFMIM 15(4);
    Barsky & Summers (1988) JPE 96(3).
    """

    name = "macro_factor"
    description = ("Macro momentum: long gold when the dollar (DXY) and 10y real yields have been "
                   "falling over 1w/1m/3m (point-in-time daily data), short when rising.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "series": {"dxy": -1.0, "real10y": -1.0},
            "horizons": (5, 20, 60),
            "vol_halflife": 60.0,
            "vol_min_obs": 20,
            "stale_days": 10.0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        series: Mapping[str, float] = p["series"]
        if not series:
            raise ValueError("series must name at least one macro series")
        p["series"] = {str(k): float(v) for k, v in series.items()}
        p["horizons"] = tuple(int(h) for h in p["horizons"])
        if not p["horizons"] or min(p["horizons"]) < 1:
            raise ValueError("horizons must be positive integers")

    @property
    def warmup_bars(self) -> int:
        return 0

    def series_z(self, md: MarketData) -> pd.DataFrame:
        """Per-series signed momentum z aligned to the bars (NaN where unknown/stale)."""
        p = self.params
        bars = md.bars
        cols: dict[str, np.ndarray] = {}
        for name, weight in p["series"].items():
            frame = (md.macro or {}).get(name)
            if frame is None:
                # a configured driver is absent: the composite silently changes meaning
                _warn_once(self, f"missing:{name}",
                           "macro_factor: series %r not in md.macro; skipped", name)
                continue
            f = clean_macro(frame, log_values=not is_yield_series(name, frame))
            f["z"] = macro_momentum_z(f["x"], p["horizons"], p["vol_halflife"], p["vol_min_obs"])
            cols[name] = np.sign(weight) * _join(bars, f, "z", p["stale_days"])
            _check_latest(self, name, cols[name], p["stale_days"])
        return pd.DataFrame(cols, index=bars.index, dtype=float)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        bars = md.bars
        z = self.series_z(md)
        if z.shape[1] == 0:
            logger.warning("macro_factor: none of %s available; forecast is 0",
                           sorted(self.params["series"]))
            return self._finalize(pd.Series(0.0, index=bars.index), bars.index)
        w = np.array([abs(self.params["series"][c]) for c in z.columns])
        vals = z.to_numpy()
        ok = np.isfinite(vals)
        num = np.where(ok, vals * w, 0.0).sum(axis=1)
        den = np.sqrt(np.where(ok, w * w, 0.0).sum(axis=1))
        combined = safe_div(num, den)
        return self._finalize(pd.Series(z_to_forecast(combined), index=bars.index), bars.index)


# ---------------------------------------------------------------------------------------
# risk_off
# ---------------------------------------------------------------------------------------
@register_strategy
class RiskOff(Strategy):
    """VIX-spike safe-haven regime: long gold after acute equity stress (long-only).

    Rationale: gold behaves as a *short-lived* safe haven — it tends to gain in the days
    and weeks after extreme equity losses (Baur & Lucey 2010; Baur & McDermott 2010) as
    investors rotate into perceived safety and real-rate expectations fall. The VIX is the
    market's forward-looking fear gauge (Whaley 2000); a jump of log VIX far above its own
    recent baseline marks the onset of such stress. The exception is a "dash for cash"
    (2008, March 2020): when the dollar itself spikes, leveraged holders sell gold for
    liquidity, so the rule stands aside when DXY momentum is extreme.

    Signal (daily, on the VIX's own observations): ``z = (ln VIX_t - m_{t-1}) / s_{t-1}``
    where ``m``/``s`` are the EWMA mean/std of log VIX up to the PREVIOUS print
    (half-life ``halflife`` obs). The regime switches on when ``z > entry_z`` and off when
    ``z < exit_z`` (hysteresis). While on, forecast
    ``= min(1, 0.5 (z - exit_z) / (entry_z - exit_z))`` — 0.5 at the entry threshold,
    larger for bigger spikes, decaying as fear subsides. Forced to 0 while the DXY 5-day
    vol-normalised momentum exceeds ``dxy_z_max`` (disabled with ``None`` or if DXY is
    missing). Long-only by design: calm markets carry no symmetric short signal.
    Defaults: half-life 60 obs, entry 1.5, exit 0.5, DXY filter 2.0, staleness 10 days.

    References: Baur & Lucey (2010) Financial Review 45(2); Baur & McDermott (2010) JBF 34(8);
    Whaley (2000) JPM 26(3).
    """

    name = "risk_off"
    description = ("Safe-haven regime: long gold (only) after a VIX spike vs its 60-day EWMA "
                   "baseline, standing aside during dollar 'dash for cash' squeezes.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "vix_series": "vix",
            "dxy_series": "dxy",
            "halflife": 60.0,
            "min_obs": 40,
            "entry_z": 1.5,
            "exit_z": 0.5,
            "dxy_z_max": 2.0,
            "dxy_horizon": 5,
            "stale_days": 10.0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        if not float(p["entry_z"]) > float(p["exit_z"]):
            raise ValueError("need entry_z > exit_z")
        p["min_obs"], p["dxy_horizon"] = int(p["min_obs"]), int(p["dxy_horizon"])
        if p["min_obs"] < 2 or p["dxy_horizon"] < 1:
            raise ValueError("min_obs must be >= 2 and dxy_horizon >= 1")

    @property
    def warmup_bars(self) -> int:
        return 0

    def daily_signal(self, vix: pd.DataFrame) -> pd.DataFrame:
        """Daily frame (on VIX observations): ``z``, ``forecast`` and ``available_at``."""
        p = self.params
        f = clean_macro(vix, log_values=True)
        x = f["x"]
        # normalised EWMA (adjust=True): not seeded with the first print (see ewma_mean)
        ew = x.ewm(halflife=float(p["halflife"]), adjust=True, min_periods=p["min_obs"])
        base_mean, base_std = ew.mean().shift(1), ew.std().shift(1)
        z = pd.Series(safe_div(x - base_mean, base_std), index=f.index)
        zv = z.to_numpy()
        k_in, k_out = float(p["entry_z"]), float(p["exit_z"])
        never = np.zeros(len(zv), dtype=bool)
        state = latch(np.asarray(zv > k_in, dtype=bool), never,
                      np.asarray(~(zv >= k_out), dtype=bool), never)
        mag = np.clip(0.5 * (zv - k_out) / (k_in - k_out), 0.0, 1.0)
        f["z"] = z
        f["forecast"] = np.where(state > 0, mag, 0.0)
        return f

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        macro = md.macro or {}
        vix = macro.get(p["vix_series"])
        if vix is None:
            logger.warning("risk_off: series %r not in md.macro; forecast is 0", p["vix_series"])
            return self._finalize(pd.Series(0.0, index=bars.index), bars.index)
        fc = _join(bars, self.daily_signal(vix), "forecast", p["stale_days"])
        _check_latest(self, p["vix_series"], fc, p["stale_days"])
        dxy = macro.get(p["dxy_series"])
        if p["dxy_z_max"] is not None and dxy is not None:
            g = clean_macro(dxy, log_values=not is_yield_series(p["dxy_series"], dxy))
            g["z"] = macro_momentum_z(g["x"], (p["dxy_horizon"],), p["halflife"], p["min_obs"])
            dz = _join(bars, g, "z", p["stale_days"])
            fc = np.where(np.asarray(dz > float(p["dxy_z_max"]), dtype=bool), 0.0, fc)
        return self._finalize(pd.Series(fc, index=bars.index), bars.index)
