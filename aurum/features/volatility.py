"""Volatility features and standalone range-based volatility estimators.

The plain helpers in this module (``true_range``, ``atr``, ``close_to_close_vol``,
``parkinson_vol``, ``garman_klass_vol``, ``rogers_satchell_vol``, ``yang_zhang_vol``,
``ewma_vol``) are imported by strategies, sizing and risk code, so they are deliberately
free of any feature-registry side effects beyond the single ``volatility`` group and have
no dependency on the other feature modules.

Every estimator is *causal*: the value at row ``t`` uses bars ``[t-n+1, t]`` only (rolling
windows with ``min_periods=n``, never centred), so it can be used at the decision time
``available_at[t]`` (SPEC §0-1).

Range-based estimators and their efficiency relative to close-to-close (for a driftless
Brownian motion sampled continuously):

* Parkinson (1980), "The Extreme Value Method for Estimating the Variance of the Rate of
  Return", J. Business 53(1): ``sigma^2 = E[ln(H/L)^2] / (4 ln 2)``; ~5x more efficient but
  biased by drift and by discrete sampling (it under-estimates when the true high/low is
  not observed).
* Garman & Klass (1980), "On the Estimation of Security Price Volatilities from Historical
  Data", J. Business 53(1): ``0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2``; ~7.4x efficient.
* Rogers & Satchell (1991), "Estimating Variance from High, Low and Closing Prices",
  Ann. Appl. Prob. 1(4): ``ln(H/C) ln(H/O) + ln(L/C) ln(L/O)``; unbiased under drift.
* Yang & Zhang (2000), "Drift-Independent Volatility Estimation Based on High, Low, Open,
  and Close Prices", J. Business 73(3): minimum-variance combination of the overnight
  (open vs previous close) variance, the open-to-close variance and Rogers–Satchell,
  handling opening jumps (weekend gaps for gold) and drift.

Annualisation: ``bars_per_year`` defaults to a *timeframe-based constant*
(``default_bars_per_year``) instead of ``aurum.core.timeframes.infer_bars_per_year`` because
the latter measures the density over the WHOLE index, i.e. it would use future timestamps
(the value at ``t`` would change as more data arrives). Pass an explicit value to match
other modules exactly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.timeframes import get_timeframe
from aurum.core.types import MarketData
from aurum.features.base import register_feature

logger = logging.getLogger(__name__)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "TRADING_MINUTES_PER_DAY",
    "atr",
    "bar_minutes",
    "close_to_close_vol",
    "default_bars_per_year",
    "ewma_vol",
    "garman_klass_vol",
    "log_pos",
    "parkinson_vol",
    "rogers_satchell_vol",
    "safe_div",
    "true_range",
    "volatility_features",
    "volatility_lookback",
    "wilder_smooth",
    "yang_zhang_vol",
]

#: Trading days per year used for annualisation of daily-or-slower series.
TRADING_DAYS_PER_YEAR: int = 252
#: Spot gold trades ~23h per weekday (one hour daily maintenance break around 17:00 NY).
TRADING_MINUTES_PER_DAY: int = 23 * 60

_LN2 = math.log(2.0)
_GK_C = 2.0 * _LN2 - 1.0


# ---------------------------------------------------------------------------------------
# small numeric utilities (shared by the other feature modules)
# ---------------------------------------------------------------------------------------
def safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray,
             *, fill: float = np.nan) -> np.ndarray:
    """Element-wise ``num / den`` returning ``fill`` where ``den`` is 0/NaN/inf or the
    result is not finite. Never emits ``inf`` (inf would poison robust scalers)."""
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(n, d).shape, fill, dtype=float)
    ok = np.isfinite(d) & (d != 0.0) & np.isfinite(n)
    np.divide(n, d, out=out, where=ok)
    out[~np.isfinite(out)] = fill
    return out


def log_pos(x: pd.Series | np.ndarray) -> np.ndarray:
    """``log(x)`` where ``x > 0`` and finite, NaN elsewhere (no warnings, no -inf)."""
    a = np.asarray(x, dtype=float)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & (a > 0)
    np.log(a, out=out, where=ok)
    return out


def bar_minutes(bars: pd.DataFrame) -> float:
    """Duration of one bar in minutes, determined point-in-time.

    Uses ``attrs["timeframe"]`` when present, otherwise the median of
    ``available_at - open`` over the first (up to) 100 rows. Only the PREFIX is inspected,
    so the answer at any cutoff never depends on later data. An EMPTY frame without a
    timeframe attribute returns NaN (there is nothing to measure, and nothing to compute).
    """
    tf = bars.attrs.get("timeframe") if hasattr(bars, "attrs") else None
    if tf:
        try:
            return float(get_timeframe(tf).minutes)
        except ValueError:
            logger.debug("unrecognised attrs timeframe %r; falling back to available_at", tf)
    if "available_at" in bars.columns and len(bars) > 0:
        head = bars.iloc[:100]
        delta = pd.DatetimeIndex(head["available_at"]) - head.index
        return float(np.median(delta.total_seconds()) / 60.0)
    if len(bars) == 0:
        return float("nan")
    raise ValueError("cannot determine bar duration: no attrs['timeframe'] and no available_at")


def default_bars_per_year(bars: pd.DataFrame) -> float:
    """Timeframe-based annualisation constant (causal alternative to ``infer_bars_per_year``).

    Intraday: ``252 * (23*60) / minutes`` (5,796 for H1); daily or slower: ``252 * 1440 /
    minutes``. The constant only rescales outputs, so small differences vs the realised bar
    density are harmless for features (they are re-scaled by the pipeline anyway).
    """
    minutes = bar_minutes(bars)
    if minutes <= 0:
        raise ValueError(f"non-positive bar duration {minutes}")
    if minutes >= 1440:
        return TRADING_DAYS_PER_YEAR * 1440.0 / minutes
    return TRADING_DAYS_PER_YEAR * min(TRADING_MINUTES_PER_DAY, 1440.0) / minutes


def _bpy(bars: pd.DataFrame, bars_per_year: float | None) -> float:
    return float(bars_per_year) if bars_per_year is not None else default_bars_per_year(bars)


def _col(bars: pd.DataFrame, name: str) -> np.ndarray:
    return bars[name].to_numpy(dtype=float)


def wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    """Wilder's running moving average (RMA): EWM with ``alpha = 1/n``.

    Wilder (1978) seeds with a simple average of the first ``n`` values; this recursive form
    seeds with the first value instead. The difference decays as ``(1-1/n)^t`` and both are
    causal. ``min_periods=n`` keeps the warm-up NaN.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------------------------------
# standalone helpers
# ---------------------------------------------------------------------------------------
def true_range(bars: pd.DataFrame) -> pd.Series:
    """Wilder's true range ``max(H-L, |H-C[t-1]|, |L-C[t-1]|)`` in price units.

    Row 0 (no previous close) falls back to ``H-L``. Captures gaps (weekend opens), which
    the plain high-low range misses.
    """
    h, lo, c = _col(bars, "high"), _col(bars, "low"), _col(bars, "close")
    prev_c = np.empty_like(c)
    prev_c[:1] = np.nan
    prev_c[1:] = c[:-1]
    tr = np.fmax(h - lo, np.fmax(np.abs(h - prev_c), np.abs(lo - prev_c)))
    return pd.Series(tr, index=bars.index, name="true_range")


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing, price units). NaN for the first ``n-1`` rows."""
    return wilder_smooth(true_range(bars), n).rename(f"atr_{n}")


def close_to_close_vol(bars: pd.DataFrame | pd.Series, n: int = 20,
                       bars_per_year: float | None = None) -> pd.Series:
    """Rolling sample standard deviation of log close-to-close returns, annualised.

    Accepts a bars frame or a close Series (then ``bars_per_year`` is required unless the
    Series index allows no inference — pass it explicitly).
    """
    if isinstance(bars, pd.Series):
        if bars_per_year is None:
            raise ValueError("bars_per_year is required when passing a close Series")
        close, bpy = bars.astype(float), float(bars_per_year)
    else:
        close, bpy = bars["close"].astype(float), _bpy(bars, bars_per_year)
    r = np.log(close).diff()
    return (r.rolling(n, min_periods=n).std(ddof=1) * math.sqrt(bpy)).rename(f"cc_vol_{n}")


def _rolling_mean_to_vol(x: np.ndarray, index: pd.Index, n: int, bpy: float, name: str) -> pd.Series:
    var = pd.Series(x, index=index).rolling(n, min_periods=n).mean().clip(lower=0.0)
    return np.sqrt(var * bpy).rename(name)


def parkinson_vol(bars: pd.DataFrame, n: int = 20, bars_per_year: float | None = None) -> pd.Series:
    """Parkinson (1980) high-low estimator, annualised: ``sqrt(mean(ln(H/L)^2)/(4 ln2) * bpy)``."""
    hl = np.log(_col(bars, "high") / _col(bars, "low"))
    return _rolling_mean_to_vol(hl * hl / (4.0 * _LN2), bars.index, n, _bpy(bars, bars_per_year),
                                f"parkinson_vol_{n}")


def garman_klass_vol(bars: pd.DataFrame, n: int = 20, bars_per_year: float | None = None) -> pd.Series:
    """Garman–Klass (1980) OHLC estimator, annualised.

    Per bar: ``0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2`` which is >= 0 because
    ``|ln(C/O)| <= ln(H/L)``. Ignores opening gaps (see Yang–Zhang for that).
    """
    hl = np.log(_col(bars, "high") / _col(bars, "low"))
    co = np.log(_col(bars, "close") / _col(bars, "open"))
    per_bar = 0.5 * hl * hl - _GK_C * co * co
    return _rolling_mean_to_vol(per_bar, bars.index, n, _bpy(bars, bars_per_year), f"gk_vol_{n}")


def _rs_per_bar(bars: pd.DataFrame) -> np.ndarray:
    o, h, lo, c = (_col(bars, k) for k in ("open", "high", "low", "close"))
    lh, ll = np.log(h), np.log(lo)
    lo_, lc = np.log(o), np.log(c)
    # Each product is >= 0 for valid OHLC (H >= O,C >= L); clip tiny negative rounding.
    return np.maximum((lh - lc) * (lh - lo_) + (ll - lc) * (ll - lo_), 0.0)


def rogers_satchell_vol(bars: pd.DataFrame, n: int = 20, bars_per_year: float | None = None) -> pd.Series:
    """Rogers–Satchell (1991) drift-independent OHLC estimator, annualised."""
    return _rolling_mean_to_vol(_rs_per_bar(bars), bars.index, n, _bpy(bars, bars_per_year),
                                f"rs_vol_{n}")


def yang_zhang_vol(bars: pd.DataFrame, n: int = 20, bars_per_year: float | None = None) -> pd.Series:
    """Yang–Zhang (2000) drift- and gap-robust estimator, annualised.

    ``sigma^2 = var(o) + k var(c) + (1-k) mean(RS)`` with overnight/gap returns
    ``o_t = ln(O_t / C_{t-1})``, open-to-close ``c_t = ln(C_t / O_t)`` and
    ``k = 0.34 / (1.34 + (n+1)/(n-1))`` (the variance-minimising weight). For intraday gold
    bars the "overnight" term is the bar-to-bar gap, dominated by weekend re-opens.
    """
    if n < 2:
        raise ValueError("yang_zhang_vol needs n >= 2")
    o, c = _col(bars, "open"), _col(bars, "close")
    prev_c = np.empty_like(c)
    prev_c[:1] = np.nan
    prev_c[1:] = c[:-1]
    idx = bars.index
    gap = pd.Series(np.log(o / prev_c), index=idx)
    oc = pd.Series(np.log(c / o), index=idx)
    rs = pd.Series(_rs_per_bar(bars), index=idx)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    var = (gap.rolling(n, min_periods=n).var(ddof=1)
           + k * oc.rolling(n, min_periods=n).var(ddof=1)
           + (1.0 - k) * rs.rolling(n, min_periods=n).mean())
    return np.sqrt(var.clip(lower=0.0) * _bpy(bars, bars_per_year)).rename(f"yz_vol_{n}")


def ewma_vol(close: pd.Series, halflife: float = 48.0, *, bars_per_year: float | None = None,
             min_periods: int = 24) -> pd.Series:
    """Zero-mean (RiskMetrics-style) EWMA volatility of log returns.

    Returns per-bar volatility if ``bars_per_year`` is None, otherwise annualised. Unlike
    ``aurum.models.volatility.ewma_volatility`` it does NOT fill the warm-up with a constant
    (features must leave warm-up rows NaN).
    """
    r = np.log(close.astype(float)).diff()
    var = (r * r).ewm(halflife=halflife, adjust=False, min_periods=min_periods).mean()
    vol = np.sqrt(var)
    if bars_per_year is not None:
        vol = vol * math.sqrt(bars_per_year)
    return vol.rename("ewma_vol")


# ---------------------------------------------------------------------------------------
# feature group
# ---------------------------------------------------------------------------------------
_DEFAULT_WINDOWS = (24, 120)
_DEFAULT_LONG = 480
_DEFAULT_VOLOFVOL = 120
VOLATILITY_LOOKBACK = _DEFAULT_LONG + 1


def volatility_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    """Warm-up (bars) of the ``volatility`` group for the effective parameters ``p``:
    close-close/Yang-Zhang windows need ``n`` returns, vol-of-vol needs ``n_min +
    volofvol_window - 1`` rows, the ``chg`` column ``2 n_min``. ``bars_per_year`` and other
    non-window parameters do not move it."""
    wins = sorted({int(w) for w in p.get("windows", _DEFAULT_WINDOWS)})
    n_min = wins[0] if wins else 0
    long_window = int(p.get("long_window", _DEFAULT_LONG))
    halflife = float(p.get("ewma_halflife", 48.0))
    last = max(max([*wins, long_window]), n_min + int(p.get("volofvol_window", _DEFAULT_VOLOFVOL)) - 1,
               2 * n_min, max(2, int(halflife // 2)), int(p.get("atr_n", 14)) - 1)
    return last + 1


@register_feature("volatility", family="volatility", lookback=VOLATILITY_LOOKBACK)
def volatility_features(
    md: MarketData,
    *,
    windows: Sequence[int] = _DEFAULT_WINDOWS,
    long_window: int = _DEFAULT_LONG,
    ewma_halflife: float = 48.0,
    volofvol_window: int = _DEFAULT_VOLOFVOL,
    atr_n: int = 14,
    bars_per_year: float | None = None,
) -> pd.DataFrame:
    """Realised-volatility family (all annualised fractions unless noted).

    Columns (``n`` in ``windows``; ``L`` = ``long_window``):

    * ``volatility_{cc,pk,gk,rs,yz}_{n}`` — close-close, Parkinson, Garman–Klass,
      Rogers–Satchell and Yang–Zhang vols. Their disagreement is informative: e.g.
      range-based >> close-close signals intrabar noise / mean reversion.
    * ``volatility_cc_{L}``, ``volatility_yz_{L}`` — slow baselines.
    * ``volatility_ewma`` — RiskMetrics EWMA vol (halflife ``ewma_halflife`` bars).
    * ``volatility_logratio_yz_{a}_{b}`` — log(short/long) vol: >0 = vol expansion.
    * ``volatility_range_cc_{n_max}`` — Parkinson / close-close (≈1 for a random walk).
    * ``volatility_volofvol`` — rolling std of log(yz_{n_min}) over ``volofvol_window``.
    * ``volatility_chg_{n_min}`` — log change of yz_{n_min} over ``n_min`` bars.
    * ``volatility_atr_pct`` — ATR(``atr_n``) / close (per-bar, not annualised).

    Vol clustering (Mandelbrot 1963; Engle 1982) makes these strong predictors of future
    *risk* even though they carry little directional information.
    """
    bars = md.bars
    bpy = _bpy(bars, bars_per_year)
    wins = sorted({int(w) for w in windows})
    if not wins:
        raise ValueError("windows must not be empty")
    n_min, n_max = wins[0], wins[-1]
    long_window = int(long_window)
    cols: dict[str, pd.Series | np.ndarray] = {}
    yz: dict[int, pd.Series] = {}
    cc: dict[int, pd.Series] = {}
    for n in sorted({*wins, long_window}):
        cc[n] = close_to_close_vol(bars, n, bpy)
        yz[n] = yang_zhang_vol(bars, n, bpy)
    for n in wins:
        cols[f"volatility_cc_{n}"] = cc[n]
        cols[f"volatility_pk_{n}"] = parkinson_vol(bars, n, bpy)
        cols[f"volatility_gk_{n}"] = garman_klass_vol(bars, n, bpy)
        cols[f"volatility_rs_{n}"] = rogers_satchell_vol(bars, n, bpy)
        cols[f"volatility_yz_{n}"] = yz[n]
    if long_window not in wins:
        cols[f"volatility_cc_{long_window}"] = cc[long_window]
        cols[f"volatility_yz_{long_window}"] = yz[long_window]
    cols["volatility_ewma"] = ewma_vol(bars["close"], ewma_halflife, bars_per_year=bpy,
                                       min_periods=max(2, int(ewma_halflife // 2)))
    chain = sorted({*wins, long_window})
    for a, b in zip(chain[:-1], chain[1:], strict=True):
        cols[f"volatility_logratio_yz_{a}_{b}"] = log_pos(safe_div(yz[a], yz[b]))
    cols[f"volatility_range_cc_{n_max}"] = safe_div(cols[f"volatility_pk_{n_max}"], cc[n_max])
    log_yz = pd.Series(log_pos(yz[n_min]), index=bars.index)
    cols["volatility_volofvol"] = log_yz.rolling(volofvol_window, min_periods=volofvol_window).std()
    cols[f"volatility_chg_{n_min}"] = log_yz - log_yz.shift(n_min)
    cols["volatility_atr_pct"] = safe_div(atr(bars, atr_n), bars["close"])
    out = pd.DataFrame(cols, index=bars.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


volatility_features.lookback_fn = volatility_lookback  # type: ignore[attr-defined]
