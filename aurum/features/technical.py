"""Price-based technical features: ``returns``, ``trend``, ``momentum``, ``meanrev``, ``range``.

All outputs are scale-free (log returns, ATR- or volatility-normalised distances, bounded
oscillators) so that a model trained on $1,300 gold transfers to $2,500 gold.

Causality: every column at row ``t`` is a function of bars ``[0, t]`` only — rolling,
expanding and EWM operators, never ``shift(-k)``/centred windows/full-sample statistics
(enforced by ``tests/test_features_leakage.py``).

References
----------
* Moskowitz, Ooi & Pedersen (2012), "Time series momentum", JFE 104(2) — vol-scaled
  multi-horizon momentum.
* Baz, Granger, Harvey, Le Roux & Rattray (2015), "Dissecting Investment Strategies in the
  Cross Section and Time Series" — the ``z exp(-z^2/4)/0.89`` momentum response function.
* Wilder (1978), "New Concepts in Technical Trading Systems" — RSI, ATR, ADX/DMI.
* Connors & Alvarez (2008), "Short Term Trading Strategies That Work" — RSI(2), streaks.
* Crabel (1990), "Day Trading with Short Term Price Patterns" — NR7 / range contraction.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.base import register_feature
from aurum.features.volatility import atr, log_pos, safe_div, true_range, wilder_smooth

logger = logging.getLogger(__name__)

__all__ = [
    "adx",
    "ema",
    "ewm_bar_vol",
    "log_close",
    "log_pos",
    "meanrev_features",
    "meanrev_lookback",
    "momentum_features",
    "momentum_lookback",
    "range_features",
    "range_lookback",
    "returns_features",
    "returns_lookback",
    "rolling_linreg",
    "rolling_zscore",
    "rsi",
    "stochastic",
    "streak",
    "trend_features",
    "trend_lookback",
    "tsmom_response",
]


# ---------------------------------------------------------------------------------------
# reusable indicator helpers (imported by mtf / regime / microstructure)
# ---------------------------------------------------------------------------------------
def log_close(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(np.log(bars["close"].to_numpy(dtype=float)), index=bars.index)


def ema(x: pd.Series, span: int, *, min_periods: int | None = None) -> pd.Series:
    """Recursive EMA with ``alpha = 2/(span+1)`` (the platform/MT5 convention)."""
    mp = span if min_periods is None else min_periods
    return x.ewm(span=span, adjust=False, min_periods=mp).mean()


def ewm_bar_vol(logret: pd.Series, halflife: float, min_periods: int) -> pd.Series:
    """Per-bar zero-mean EWMA volatility ``sqrt(EWM(r^2))`` including the current bar."""
    return np.sqrt((logret * logret).ewm(halflife=halflife, adjust=False,
                                         min_periods=min_periods).mean())


def rolling_zscore(x: pd.Series, n: int, *, min_periods: int | None = None) -> pd.Series:
    """``(x - rolling_mean) / rolling_std`` over the trailing ``n`` values (causal)."""
    mp = n if min_periods is None else min_periods
    r = x.rolling(n, min_periods=mp)
    return pd.Series(safe_div(x - r.mean(), r.std(ddof=1)), index=x.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100] computed as ``100 * RMA(up) / (RMA(up) + RMA(down))``.

    The algebraically equivalent form avoids the ``RS = up/down`` division by zero; a
    perfectly flat window returns 50.
    """
    d = close.astype(float).diff()
    up = wilder_smooth(d.clip(lower=0.0), n)
    dn = wilder_smooth((-d).clip(lower=0.0), n)
    tot = (up + dn).to_numpy()
    val = np.where(tot > 0, 100.0 * safe_div(up, tot, fill=0.0), 50.0)
    val = np.where(up.isna().to_numpy(), np.nan, val)
    return pd.Series(val, index=close.index, name=f"rsi_{n}")


def stochastic(bars: pd.DataFrame, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Lane's stochastic oscillator %K(n) and %D = SMA_d(%K), in [0, 100]."""
    hh = bars["high"].rolling(n, min_periods=n).max()
    ll = bars["low"].rolling(n, min_periods=n).min()
    rng = (hh - ll).to_numpy()
    k = np.where(rng > 0, 100.0 * safe_div(bars["close"] - ll, rng, fill=0.5), 50.0)
    k = pd.Series(np.where(hh.isna().to_numpy(), np.nan, k), index=bars.index)
    return k, k.rolling(d, min_periods=d).mean()


def adx(bars: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX, +DI, -DI (each in [0, 100])."""
    h, lo = bars["high"].astype(float), bars["low"].astype(float)
    up = h.diff()
    dn = -lo.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm[:1] = np.nan
    minus_dm[:1] = np.nan
    atr_w = wilder_smooth(true_range(bars), n)
    pdi = pd.Series(100.0 * safe_div(wilder_smooth(pd.Series(plus_dm, index=bars.index), n), atr_w),
                    index=bars.index)
    mdi = pd.Series(100.0 * safe_div(wilder_smooth(pd.Series(minus_dm, index=bars.index), n), atr_w),
                    index=bars.index)
    tot = (pdi + mdi).to_numpy()
    dx = np.where(tot > 0, 100.0 * safe_div((pdi - mdi).abs(), tot, fill=0.0), 0.0)
    dx = pd.Series(np.where(np.isnan(tot), np.nan, dx), index=bars.index)
    return wilder_smooth(dx, n), pdi, mdi


def rolling_linreg(y: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of ``y`` on time over the trailing ``n`` points: (slope per bar, R^2).

    The slope is the fixed linear filter ``sum_j w_j y_{t-n+1+j}`` with
    ``w_j = (j - (n-1)/2) / Sxx``; it is evaluated by ``n`` shifted multiply-adds (exact
    per window, no running-sum drift, bit-for-bit deterministic). ``R^2 = b^2 Sxx / TSS``.
    """
    if n < 3:
        raise ValueError("rolling_linreg needs n >= 3")
    arr = y.to_numpy(dtype=float)
    size = arr.size
    slope = np.full(size, np.nan)
    if size >= n:
        j = np.arange(n, dtype=float) - (n - 1) / 2.0
        sxx = float(np.dot(j, j))
        m = size - n + 1
        acc = np.zeros(m)
        for k in range(n):
            acc += j[k] * arr[k:k + m]
        slope[n - 1:] = acc / sxx
    else:
        sxx = float(n * (n * n - 1) / 12.0)
    tss = (y.rolling(n, min_periods=n).var(ddof=0) * n).to_numpy()
    r2 = np.clip(safe_div(slope * slope * sxx, tss), 0.0, 1.0)
    return pd.Series(slope, index=y.index), pd.Series(r2, index=y.index)


def tsmom_response(z: np.ndarray | pd.Series) -> np.ndarray:
    """Baz et al. (2015) momentum response ``phi(z) = z exp(-z^2/4) / 0.89``.

    Increases with signal strength up to |z| = sqrt(2) then decays: very stretched trends
    are more prone to reversal. Max |phi| ≈ 1.
    """
    a = np.asarray(z, dtype=float)
    return a * np.exp(-(a * a) / 4.0) / 0.89


def streak(logret: pd.Series) -> pd.Series:
    """Signed count of consecutive up (+) / down (-) bars ending at t (0 on a flat bar)."""
    s = np.sign(logret.to_numpy(dtype=float))
    valid = ~np.isnan(s)
    s0 = np.where(valid, s, 0.0)
    ss = pd.Series(s0, index=logret.index)
    grp = (ss != ss.shift(1)).cumsum()
    cnt = ss.groupby(grp.to_numpy()).cumcount().to_numpy() + 1.0
    out = np.where(valid, s0 * cnt, np.nan)
    return pd.Series(out, index=logret.index)


def _frame(cols: dict, index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(cols, index=index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


def _mx(values: Iterable[Any]) -> int:
    """Largest integer in ``values`` (0 if empty) — for warm-up computations."""
    return max((int(v) for v in values), default=0)


# ---------------------------------------------------------------------------------------
# warm-up (lookback) functions: ``(params, bar_minutes) -> bars``
# ---------------------------------------------------------------------------------------
# Each returns an ``L`` (tight to within a bar or two) such that every column of the group
# is non-NaN from row index ``L`` on (given enough data and no degenerate windows), for the
# EFFECTIVE parameters (signature defaults merged with overrides). ``FeaturePipeline``
# uses them so that overriding a window moves the warm-up, while non-window parameters do
# not. ``tests/test_features_pipeline.py`` checks them against the observed first values.
def returns_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    return max(_mx(p.get("horizons", _RET_HORIZONS)), int(p.get("vol_min_periods", 24))) + 1


def trend_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    spans = _mx(p.get("ema_spans", _EMA_SPANS))
    pairs = _mx(x for pair in p.get("ema_pairs", _EMA_PAIRS) for x in pair)
    fast, slow, sig_n = (int(x) for x in p.get("macd", (12, 26, 9)))
    adx_n, atr_n = int(p.get("adx_n", 14)), int(p.get("atr_n", 14))
    last = max(spans - 1 + int(p.get("slope_bars", 5)), pairs - 1, max(fast, slow) + sig_n - 2,
               2 * adx_n - 1, _mx(p.get("linreg_windows", _LINREG)) - 1, atr_n - 1)
    return last + 1


def momentum_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    last = max(_mx(p.get("rsi_periods", (2, 14))),
               int(p.get("stoch_n", 14)) + int(p.get("stoch_d", 3)) - 2,
               _mx(p.get("tsmom_horizons", _TSMOM_H)), int(p.get("vol_min_periods", 24)),
               int(p.get("up_frac_window", 24)))
    return last + 1


def meanrev_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    return max(_mx(p.get("z_windows", _Z_WINDOWS)), int(p.get("bb_n", 20)), 1) + 1


def range_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    return max(_mx(p.get("donchian", _DONCHIAN)), int(p.get("atr_n", 14)),
               int(p.get("nr_window", 7)), 1) + 1


# ---------------------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------------------
_RET_HORIZONS = (1, 2, 3, 4, 6, 8, 12, 24, 48)
RETURNS_LOOKBACK = max(_RET_HORIZONS) + 1


@register_feature("returns", family="returns", lookback=RETURNS_LOOKBACK)
def returns_features(
    md: MarketData,
    *,
    horizons: Sequence[int] = _RET_HORIZONS,
    vol_halflife: float = 48.0,
    vol_min_periods: int = 24,
) -> pd.DataFrame:
    """Vol-normalised trailing log returns.

    ``returns_z_{h} = ln(C_t / C_{t-h}) / (sigma_t sqrt(h))`` with ``sigma_t`` the per-bar
    EWMA vol (halflife ``vol_halflife``) including bar t: a t-stat-like, regime-comparable
    measure of the recent move. ``returns_log_1`` is the raw 1-bar log return.
    """
    bars = md.bars
    lc = log_close(bars)
    r = lc.diff()
    sig = ewm_bar_vol(r, vol_halflife, vol_min_periods)
    cols: dict[str, pd.Series | np.ndarray] = {"returns_log_1": r}
    for h in sorted({int(x) for x in horizons}):
        cols[f"returns_z_{h}"] = safe_div(lc - lc.shift(h), sig * math.sqrt(h))
    return _frame(cols, bars.index)


# ---------------------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------------------
_EMA_SPANS = (10, 20, 50, 100, 200)
_EMA_PAIRS = ((10, 50), (20, 100), (50, 200))
_LINREG = (20, 50, 100)
TREND_LOOKBACK = max(_EMA_SPANS) + 5


@register_feature("trend", family="trend", lookback=TREND_LOOKBACK)
def trend_features(
    md: MarketData,
    *,
    ema_spans: Sequence[int] = _EMA_SPANS,
    ema_pairs: Sequence[Sequence[int]] = _EMA_PAIRS,
    slope_bars: int = 5,
    atr_n: int = 14,
    adx_n: int = 14,
    macd: Sequence[int] = (12, 26, 9),
    linreg_windows: Sequence[int] = _LINREG,
) -> pd.DataFrame:
    """Trend-following state, expressed in ATR units.

    * ``trend_ema_dist_{s}`` = (C - EMA_s)/ATR; ``trend_ema_slope_{s}`` = per-bar EMA change
      over ``slope_bars`` in ATR; ``trend_ema_cross_{f}_{s}`` = (EMA_f - EMA_s)/ATR.
    * ``trend_macd``/``_signal``/``_hist`` divided by ATR (price-level free MACD).
    * ``trend_adx_{n}`` (ADX/100) and ``trend_di_diff_{n}`` ((+DI - -DI)/100).
    * ``trend_linreg_t_{n}`` signed t-stat of the OLS slope of log price on time and
      ``trend_linreg_r2_{n}`` (note: t-stats are inflated by serial correlation of the
      residuals — use as a strength score, not for inference).
    """
    bars = md.bars
    close = bars["close"].astype(float)
    a = atr(bars, atr_n)
    cols: dict[str, pd.Series | np.ndarray] = {}
    emas: dict[int, pd.Series] = {}
    spans = sorted({int(s) for s in ema_spans} | {int(x) for p in ema_pairs for x in p}
                   | {int(macd[0]), int(macd[1])})
    for s in spans:
        emas[s] = ema(close, s)
    for s in sorted({int(s) for s in ema_spans}):
        cols[f"trend_ema_dist_{s}"] = safe_div(close - emas[s], a)
        cols[f"trend_ema_slope_{s}"] = safe_div(emas[s] - emas[s].shift(slope_bars), a * slope_bars)
    for f, s in ema_pairs:
        cols[f"trend_ema_cross_{int(f)}_{int(s)}"] = safe_div(emas[int(f)] - emas[int(s)], a)
    fast, slow, sig_n = (int(x) for x in macd)
    macd_line = emas[fast] - emas[slow]
    macd_sig = ema(macd_line, sig_n)
    cols["trend_macd"] = safe_div(macd_line, a)
    cols["trend_macd_signal"] = safe_div(macd_sig, a)
    cols["trend_macd_hist"] = safe_div(macd_line - macd_sig, a)
    adx_v, pdi, mdi = adx(bars, adx_n)
    cols[f"trend_adx_{adx_n}"] = adx_v / 100.0
    cols[f"trend_di_diff_{adx_n}"] = (pdi - mdi) / 100.0
    lc = log_close(bars)
    for n in sorted({int(x) for x in linreg_windows}):
        slope, r2 = rolling_linreg(lc, n)
        r2c = np.clip(r2.to_numpy(), 0.0, 1.0 - 1e-12)
        t = np.sign(slope.to_numpy()) * np.sqrt(r2c * (n - 2) / (1.0 - r2c))
        cols[f"trend_linreg_t_{n}"] = t
        cols[f"trend_linreg_r2_{n}"] = r2
    return _frame(cols, bars.index)


# ---------------------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------------------
_TSMOM_H = (24, 72, 120, 240, 480)
MOMENTUM_LOOKBACK = max(_TSMOM_H) + 1


@register_feature("momentum", family="momentum", lookback=MOMENTUM_LOOKBACK)
def momentum_features(
    md: MarketData,
    *,
    rsi_periods: Sequence[int] = (2, 14),
    stoch_n: int = 14,
    stoch_d: int = 3,
    tsmom_horizons: Sequence[int] = _TSMOM_H,
    vol_halflife: float = 48.0,
    vol_min_periods: int = 24,
    up_frac_window: int = 24,
) -> pd.DataFrame:
    """Oscillators and time-series-momentum strength.

    * ``momentum_rsi_{n}`` = (RSI-50)/50 in [-1, 1]; ``momentum_stoch_{k,d}_{n}`` likewise.
    * ``momentum_tsmom_{h}`` = h-bar log return / (sigma_t sqrt(h)) (MOP 2012 signal
      strength); ``momentum_tsmom_resp_{h}`` = Baz et al. response phi(z).
    * ``momentum_tsmom_agg`` = mean of sign(z_h) over horizons (multi-horizon vote).
    * ``momentum_up_frac_{n}`` = share of up bars in the last n bars minus 0.5.
    """
    bars = md.bars
    close = bars["close"].astype(float)
    cols: dict[str, pd.Series | np.ndarray] = {}
    for n in sorted({int(x) for x in rsi_periods}):
        cols[f"momentum_rsi_{n}"] = (rsi(close, n) - 50.0) / 50.0
    k, d = stochastic(bars, stoch_n, stoch_d)
    cols[f"momentum_stoch_k_{stoch_n}"] = (k - 50.0) / 50.0
    cols[f"momentum_stoch_d_{stoch_n}"] = (d - 50.0) / 50.0
    lc = log_close(bars)
    r = lc.diff()
    sig = ewm_bar_vol(r, vol_halflife, vol_min_periods)
    signs = []
    for h in sorted({int(x) for x in tsmom_horizons}):
        z = safe_div(lc - lc.shift(h), sig * math.sqrt(h))
        cols[f"momentum_tsmom_{h}"] = z
        cols[f"momentum_tsmom_resp_{h}"] = tsmom_response(z)
        signs.append(np.sign(z))
    if signs:
        stack = np.vstack(signs)
        # NaN until every horizon is warm (np.mean propagates NaN): keeps warm-up explicit.
        cols["momentum_tsmom_agg"] = stack.mean(axis=0)
    up = r.gt(0).astype(float).where(r.notna())
    cols[f"momentum_up_frac_{up_frac_window}"] = (
        up.rolling(up_frac_window, min_periods=up_frac_window).mean() - 0.5)
    return _frame(cols, bars.index)


# ---------------------------------------------------------------------------------------
# mean reversion
# ---------------------------------------------------------------------------------------
_Z_WINDOWS = (20, 50, 100)
MEANREV_LOOKBACK = max(_Z_WINDOWS) + 1


@register_feature("meanrev", family="meanrev", lookback=MEANREV_LOOKBACK)
def meanrev_features(
    md: MarketData,
    *,
    z_windows: Sequence[int] = _Z_WINDOWS,
    bb_n: int = 20,
    bb_k: float = 2.0,
) -> pd.DataFrame:
    """Stretch-from-equilibrium measures used by fade/mean-reversion models.

    * ``meanrev_z_{n}`` = (C - SMA_n)/std_n.
    * ``meanrev_bb_pctb_{n}`` = Bollinger %B - 0.5 (0 at the middle band, ±0.5 at the bands);
      ``meanrev_bb_width_{n}`` = band width / SMA (squeeze detector).
    * ``meanrev_streak`` = signed consecutive up/down bar count (Connors).
    """
    bars = md.bars
    close = bars["close"].astype(float)
    cols: dict[str, pd.Series | np.ndarray] = {}
    for n in sorted({int(x) for x in z_windows}):
        cols[f"meanrev_z_{n}"] = rolling_zscore(close, n)
    roll = close.rolling(bb_n, min_periods=bb_n)
    mid, sd = roll.mean(), roll.std(ddof=0)
    cols[f"meanrev_bb_pctb_{bb_n}"] = safe_div(close - mid, 2.0 * bb_k * sd, fill=np.nan)
    cols[f"meanrev_bb_width_{bb_n}"] = safe_div(2.0 * bb_k * sd, mid)
    cols["meanrev_streak"] = streak(log_close(bars).diff())
    out = _frame(cols, bars.index)
    # A perfectly flat window has sd == 0: price sits on the middle band -> %B centred.
    pctb = f"meanrev_bb_pctb_{bb_n}"
    flat = (sd == 0).to_numpy()
    out.loc[flat, pctb] = 0.0
    return out


# ---------------------------------------------------------------------------------------
# range / candle anatomy
# ---------------------------------------------------------------------------------------
_DONCHIAN = (20, 55)
RANGE_LOOKBACK = max(_DONCHIAN) + 1


@register_feature("range", family="range", lookback=RANGE_LOOKBACK)
def range_features(
    md: MarketData,
    *,
    donchian: Sequence[int] = _DONCHIAN,
    atr_n: int = 14,
    nr_window: int = 7,
) -> pd.DataFrame:
    """Position within recent ranges and single-bar anatomy.

    * ``range_donchian_pos_{n}`` = close position in the n-bar high/low channel minus 0.5;
      ``range_donchian_width_{n}`` = channel width in ATR.
    * ``range_body`` (C-O)/(H-L), ``range_upper_wick``, ``range_lower_wick``,
      ``range_clv`` close-location value ((C-L)-(H-C))/(H-L) in [-1, 1].
    * ``range_bar_atr`` (H-L)/ATR, ``range_body_atr`` (C-O)/ATR.
    * ``range_inside_bar`` / ``range_outside_bar`` / ``range_nr{k}`` (narrowest range of
      the last k bars) flags — volatility-contraction patterns preceding breakouts.
    """
    bars = md.bars
    o, h, lo, c = (bars[k].astype(float) for k in ("open", "high", "low", "close"))
    a = atr(bars, atr_n)
    cols: dict[str, pd.Series | np.ndarray] = {}
    for n in sorted({int(x) for x in donchian}):
        hh = h.rolling(n, min_periods=n).max()
        ll = lo.rolling(n, min_periods=n).min()
        width = (hh - ll).to_numpy()
        pos = np.where(width > 0, safe_div(c - ll, width, fill=0.5) - 0.5, 0.0)
        cols[f"range_donchian_pos_{n}"] = np.where(np.isnan(width), np.nan, pos)
        cols[f"range_donchian_width_{n}"] = safe_div(width, a)
    rng = (h - lo).to_numpy()
    has = rng > 0
    body = (c - o).to_numpy()
    cols["range_body"] = np.where(has, safe_div(body, rng, fill=0.0), 0.0)
    cols["range_upper_wick"] = np.where(has, safe_div(h - np.maximum(o, c), rng, fill=0.0), 0.0)
    cols["range_lower_wick"] = np.where(has, safe_div(np.minimum(o, c) - lo, rng, fill=0.0), 0.0)
    cols["range_clv"] = np.where(has, safe_div((c - lo) - (h - c), rng, fill=0.0), 0.0)
    cols["range_bar_atr"] = safe_div(rng, a)
    cols["range_body_atr"] = safe_div(body, a)
    ph, pl = h.shift(1), lo.shift(1)
    first = ph.isna().to_numpy()
    cols["range_inside_bar"] = np.where(first, np.nan, ((h <= ph) & (lo >= pl)).astype(float))
    cols["range_outside_bar"] = np.where(first, np.nan, ((h > ph) & (lo < pl)).astype(float))
    rng_s = pd.Series(rng, index=bars.index)
    rmin = rng_s.rolling(nr_window, min_periods=nr_window).min()
    cols[f"range_nr{nr_window}"] = np.where(rmin.isna(), np.nan, (rng_s <= rmin).astype(float))
    return _frame(cols, bars.index)


returns_features.lookback_fn = returns_lookback  # type: ignore[attr-defined]
trend_features.lookback_fn = trend_lookback  # type: ignore[attr-defined]
momentum_features.lookback_fn = momentum_lookback  # type: ignore[attr-defined]
meanrev_features.lookback_fn = meanrev_lookback  # type: ignore[attr-defined]
range_features.lookback_fn = range_lookback  # type: ignore[attr-defined]
