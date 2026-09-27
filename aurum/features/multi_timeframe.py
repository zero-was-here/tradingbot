"""Higher-timeframe context (``mtf``).

Trend and volatility on H4/D1 carry information that is noisy to extract from H1 bars
alone (and many discretionary gold traders anchor on the prior day's high/low/close).

Point-in-time mechanics: HTF bars are built with ``aurum.data.resample.resample_bars``
(via ``resample_anchored`` below, which makes the daily anchor work on pandas 3;
labelled by OPEN, ``available_at`` = bucket end, trailing incomplete bucket dropped) and
HTF features — computed causally ON THE HTF SERIES — are mapped onto base bars with
``align_htf``, i.e. an as-of join on ``available_at``. A base row at decision time T
therefore only sees HTF bars that were complete at T: no 55-minute look-ahead of the
legacy H1->M5 forward-fill.

Default ``daily_anchor_hour_utc = 22`` aligns D1/H4 buckets to the broker day (New York
17:00 close ≈ 21:00–22:00 UTC), so the Sunday-evening re-open belongs to Monday instead of
forming a two-hour "Sunday" daily bar.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.timeframes import Timeframe, get_timeframe
from aurum.core.types import MarketData
from aurum.data.resample import align_htf, resample_bars
from aurum.features.base import register_feature
from aurum.features.technical import ema, ewm_bar_vol, rsi
from aurum.features.volatility import (
    atr,
    bar_minutes,
    default_bars_per_year,
    parkinson_vol,
    safe_div,
)

logger = logging.getLogger(__name__)

__all__ = ["htf_compact_features", "mtf_features", "mtf_lookback", "resample_anchored"]

#: ``htf_compact_features`` settings that ``mtf_features`` does not expose.
_HTF_SLOPE_BARS = 3
_HTF_VOL_HALFLIFE = 20.0


def _htf_bars_needed(p: Mapping[str, Any]) -> int:
    """HTF bars until every ``htf_compact_features`` column is valid (index of the first
    fully-warm HTF row + 1)."""
    ret_h = max((int(k) for k in p.get("ret_horizons", (1, 5, 20))), default=0)
    ema_span = int(p.get("ema_span", 20))
    last_index = max(
        ret_h,                                              # lc - lc.shift(k)
        max(2, int(_HTF_VOL_HALFLIFE // 2)),                # EWM vol min_periods (on diffs)
        ema_span - 1 + _HTF_SLOPE_BARS,                     # EMA slope
        int(p.get("rsi_n", 14)),                            # RSI (on diffs)
        int(p.get("atr_n", 14)) - 1,
        int(p.get("vol_window", 20)) - 1,
        int(p.get("donchian_n", 20)) - 1,
    )
    return last_index + 1


def mtf_lookback(params: Mapping[str, Any], bar_minutes: float) -> int:
    """Warm-up in BASE bars of the ``mtf`` group for ``bar_minutes``-minute bars.

    An HTF bucket holds at most ``htf_minutes / bar_minutes`` base bars (fewer across
    weekends and holidays), so ``(needed + 1) * htf_minutes / bar_minutes`` base bars are
    always enough (``+1`` for a partial first bucket). The previous-day levels need one
    complete broker day after a possibly partial first one (2 days).
    """
    base = float(bar_minutes)
    lb = 0  # no slower HTF and no previous-day levels -> the group emits no columns
    need = _htf_bars_needed(params)
    for name in params.get("htfs", ("H4", "D1")):
        tf = get_timeframe(name)
        if tf.minutes <= base:
            continue
        lb = max(lb, math.ceil((need + 1) * tf.minutes / base))
    if params.get("prev_day", True) and base < 1440:
        # one complete broker day after a partial first one, and the base-bar ATR
        lb = max(lb, math.ceil(2 * 1440 / base), int(params.get("atr_n", 14)))
    return lb


#: H1 value of :func:`mtf_lookback` with default parameters (24 D1 bars -> 576 H1 bars).
MTF_LOOKBACK = mtf_lookback({}, 60.0)


def _resample_quiet(bars: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """``resample_bars`` with a zero anchor, silencing pandas-3's (here irrelevant)
    "offset does not take effect for non-Tick freq" RuntimeWarning for D1."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'offset' keyword does not take effect.*",
                                category=RuntimeWarning)
        return resample_bars(bars, tf)


def resample_anchored(bars: pd.DataFrame, to: str | Timeframe, anchor_hour_utc: int = 0) -> pd.DataFrame:
    """``resample_bars`` with H4/D1 buckets anchored at ``anchor_hour_utc`` on any pandas.

    Older ``resample_bars`` versions resampled with ``"1D"``, which pandas >= 3 treats as a
    calendar-day (non-Tick) offset that silently ignores ``offset=``, so D1 buckets stayed
    at UTC midnight. This wrapper does not depend on how the anchor is implemented: it
    shifts all timestamps by ``24 - anchor`` hours, resamples with a midnight anchor and
    shifts back. The completeness and ``available_at`` rules of ``resample_bars`` compare
    timestamps only with each other, so they are invariant under a common shift and the
    result is exactly the anchored aggregation (tests check it equals both a manual
    group-by and the current ``resample_bars(..., daily_anchor_hour_utc=anchor)``).
    """
    tf = get_timeframe(to)
    shift_h = (24 - int(anchor_hour_utc)) % 24
    if shift_h == 0 or tf.minutes < 240:
        return _resample_quiet(bars, tf)
    delta = pd.Timedelta(hours=shift_h)
    # set_axis first, then shift the column on the SAME (new) index: assigning
    # ``bars["available_at"] + delta`` directly would align on the old labels.
    shifted = bars.set_axis(bars.index + delta, axis=0)
    shifted["available_at"] = shifted["available_at"] + delta
    out = _resample_quiet(shifted, tf)
    out.index = out.index - delta
    out.index.name = "time"
    out["available_at"] = out["available_at"] - delta
    return out


def htf_compact_features(
    htf: pd.DataFrame,
    *,
    ret_horizons: Sequence[int] = (1, 5, 20),
    ema_span: int = 20,
    slope_bars: int = _HTF_SLOPE_BARS,
    rsi_n: int = 14,
    atr_n: int = 14,
    vol_window: int = 20,
    donchian_n: int = 20,
    vol_halflife: float = _HTF_VOL_HALFLIFE,
) -> pd.DataFrame:
    """Compact trend/momentum/vol features computed on an HTF bars frame (causal in HTF
    time). Returns a frame indexed like ``htf`` with an ``available_at`` column."""
    close = htf["close"].astype(float)
    lc = pd.Series(np.log(close.to_numpy()), index=htf.index)
    r = lc.diff()
    sig = ewm_bar_vol(r, vol_halflife, max(2, int(vol_halflife // 2)))
    a = atr(htf, atr_n)
    cols: dict[str, pd.Series | np.ndarray] = {}
    for k in sorted({int(x) for x in ret_horizons}):
        cols[f"ret_z_{k}"] = safe_div(lc - lc.shift(k), sig * math.sqrt(k))
    e = ema(close, ema_span)
    cols[f"ema_dist_{ema_span}"] = safe_div(close - e, a)
    cols[f"ema_slope_{ema_span}"] = safe_div(e - e.shift(slope_bars), a * slope_bars)
    cols[f"rsi_{rsi_n}"] = (rsi(close, rsi_n) - 50.0) / 50.0
    cols[f"pk_vol_{vol_window}"] = parkinson_vol(htf, vol_window, default_bars_per_year(htf))
    hh = htf["high"].rolling(donchian_n, min_periods=donchian_n).max()
    ll = htf["low"].rolling(donchian_n, min_periods=donchian_n).min()
    width = (hh - ll).to_numpy()
    pos = np.where(width > 0, safe_div(close - ll, width, fill=0.5) - 0.5, 0.0)
    cols[f"donchian_pos_{donchian_n}"] = np.where(np.isnan(width), np.nan, pos)
    out = pd.DataFrame(cols, index=htf.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    out["available_at"] = htf["available_at"]
    return out


@register_feature("mtf", family="mtf", lookback=MTF_LOOKBACK)
def mtf_features(
    md: MarketData,
    *,
    htfs: Sequence[str] = ("H4", "D1"),
    daily_anchor_hour_utc: int = 22,
    prev_day: bool = True,
    atr_n: int = 14,
    ret_horizons: Sequence[int] = (1, 5, 20),
    ema_span: int = 20,
    rsi_n: int = 14,
    vol_window: int = 20,
    donchian_n: int = 20,
) -> pd.DataFrame:
    """Higher-timeframe trend/momentum/vol + previous-day levels, point-in-time aligned.

    Columns ``mtf_{htf}_{name}`` (``htf`` lower-cased, e.g. ``mtf_h4_ret_z_5``) for each
    HTF strictly slower than the base bars (others are skipped with an INFO log), plus:

    * ``mtf_pdh_dist`` / ``mtf_pdl_dist`` / ``mtf_pdc_dist``: (C - previous completed
      day's high/low/close) in base-bar ATR units;
    * ``mtf_pd_pos``: close position inside the previous day's range minus 0.5.
    """
    bars = md.bars
    base_minutes = bar_minutes(bars)
    cols: dict[str, pd.Series | np.ndarray] = {}
    d1_cache: pd.DataFrame | None = None
    for name in htfs:
        tf = get_timeframe(name)
        if tf.minutes <= base_minutes:
            logger.info("mtf: skipping %s (not slower than base %.0f-minute bars)", tf.name, base_minutes)
            continue
        htf = resample_anchored(bars, tf, daily_anchor_hour_utc)
        if tf.name == "D1":
            d1_cache = htf
        feats = htf_compact_features(htf, ret_horizons=ret_horizons, ema_span=ema_span, rsi_n=rsi_n,
                                     atr_n=atr_n, vol_window=vol_window, donchian_n=donchian_n)
        names = [c for c in feats.columns if c != "available_at"]
        aligned = align_htf(bars, feats, columns=names)
        prefix = f"mtf_{tf.name.lower()}_"
        for c in names:
            cols[prefix + c] = aligned[c].to_numpy(dtype=float)
    if prev_day and base_minutes < 1440:
        d1 = d1_cache if d1_cache is not None else resample_anchored(
            bars, "D1", daily_anchor_hour_utc)
        pd_levels = align_htf(bars, d1, columns=["high", "low", "close"])
        a = atr(bars, atr_n)
        close = bars["close"].astype(float)
        pdh = pd_levels["high"].to_numpy(dtype=float)
        pdl = pd_levels["low"].to_numpy(dtype=float)
        pdc = pd_levels["close"].to_numpy(dtype=float)
        cols["mtf_pdh_dist"] = safe_div(close - pdh, a)
        cols["mtf_pdl_dist"] = safe_div(close - pdl, a)
        cols["mtf_pdc_dist"] = safe_div(close - pdc, a)
        width = pdh - pdl
        pos = np.where(width > 0, safe_div(close - pdl, width, fill=0.5) - 0.5, 0.0)
        cols["mtf_pd_pos"] = np.where(np.isnan(width), np.nan, pos)
    out = pd.DataFrame(cols, index=bars.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


mtf_features.lookback_fn = mtf_lookback  # type: ignore[attr-defined]
