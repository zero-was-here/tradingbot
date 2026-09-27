"""Model-free market-regime descriptors (``regime``).

Strategy performance is strongly regime-dependent: trend followers need persistent,
efficient price paths; faders need noisy, mean-reverting ones; everything suffers in
volatility spikes. This module provides *causal, parameter-free* regime statistics. Fitted
latent-state models (Gaussian HMM) live in ``aurum.models.regime`` and must be trained on
TRAIN data only — nothing here is fitted.

* Volatility percentile — percentile rank of the current realised vol among the vols of a
  BOUNDED trailing window of ``rank_years`` years (default 1 year = the 52-week convention
  of "IV rank / IV percentile" desks use), converted to bars from the bar size. Never the
  full sample (a full-sample quantile is the classic regime-feature leak). The rank is NaN
  until the window is FULL, so every defined value depends on exactly the last
  ``vol_window + W - 1`` bars: a live runner that computes on any sliding window of at least
  the group's warm-up gets bit-identical values to a research run on the full history.
  ``regime_high_vol`` / ``regime_low_vol`` are derived from this bounded rank.
* ``expanding=True`` adds the legacy expanding-window rank ``regime_vol_pctrank_exp``
  (against ALL past values). It is causal but NOT sliding-window-stable — its value depends
  on where the history starts, so a live runner fed ``N`` bars disagrees with the backtest
  forever — which is why it is off by default.
* Kaufman (1995) efficiency ratio ``|C_t - C_{t-n}| / sum |dC|`` in [0, 1]: 1 = straight
  line, ~0 = pure noise ("Smarter Trading", McGraw-Hill).
* Variance ratio ``VR(q) = Var(r^(q)) / (q Var(r))`` (Lo & MacKinlay 1988, RFS 1(1)):
  > 1 trending/persistent, < 1 mean-reverting. Hurst proxy ``H = 0.5 (1 + ln VR / ln q)``
  from ``VR(q) ~ q^(2H-1)`` for fractional Gaussian noise.
* Trend strength: ``|ln(C_t/C_{t-n})| / (rms_n sqrt(n))`` where ``rms_n`` is the root mean
  square of the SAME ``n`` one-bar log returns (a bounded window, so it is sliding-window
  stable; ``vol_halflife=<float>`` restores the legacy EWMA normaliser, whose dependence on
  the first return of the history only decays geometrically), and a discrete trend state in
  {-1, 0, 1} that is non-zero only when both strength and efficiency are high.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.base import register_feature
from aurum.features.technical import ewm_bar_vol, log_close
from aurum.features.volatility import (
    TRADING_DAYS_PER_YEAR,
    TRADING_MINUTES_PER_DAY,
    bar_minutes,
    log_pos,
    safe_div,
)

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_RANK_YEARS", "efficiency_ratio", "rank_window_bars", "regime_features",
           "regime_lookback", "variance_ratio"]

#: Default horizon of the bounded volatility percentile (the 52-week convention).
DEFAULT_RANK_YEARS = 1.0
#: Bar size assumed when it cannot be measured (empty frame without ``attrs["timeframe"]``).
_DEFAULT_BAR_MINUTES = 60.0


def rank_window_bars(rank_years: float, minutes: float) -> int:
    """Bars in ``rank_years`` years of ``minutes``-minute bars (``aurum.features.volatility``
    annualisation: 252 days of 23 trading hours intraday, 252 days for daily bars)."""
    m = float(minutes)
    if not (math.isfinite(m) and m > 0):
        m = _DEFAULT_BAR_MINUTES
    years = float(rank_years)
    if not (math.isfinite(years) and years > 0):
        raise ValueError(f"rank_years must be positive, got {rank_years!r}")
    per_day = TRADING_MINUTES_PER_DAY / m if m < 1440.0 else 1440.0 / m
    return max(2, int(round(years * TRADING_DAYS_PER_YEAR * per_day)))


def _rank_window(p: Mapping[str, Any], minutes: float) -> int:
    """Bounded rank window in bars: explicit ``rank_window`` or ``rank_years`` of bars."""
    w = p.get("rank_window")
    if w is not None:
        w = int(w)
        if w < 2:
            raise ValueError(f"rank_window must be >= 2, got {w}")
        return w
    return rank_window_bars(float(p.get("rank_years", DEFAULT_RANK_YEARS)), minutes)


def regime_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    """Warm-up (bars) = bars until every column is defined, which for this group is also the
    history after which values no longer depend on where the history starts: the bounded
    vol rank needs ``vol_window`` returns then a FULL window of ``W`` vols; ``VR(q)`` needs
    ``q + vr_window - 1`` rows; ER/trend need their window (+1 for the first return)."""
    vol_window = int(p.get("vol_window", 24))
    trend_window = int(p.get("trend_window", 120))
    ers = [int(x) for x in p.get("er_windows", (20, 100))]
    qs = [int(x) for x in p.get("vr_qs", (4, 16))]
    vr_window = int(p.get("vr_window", 240))
    last = max(vol_window + _rank_window(p, bar_minutes) - 1, max([*ers, trend_window]),
               max(qs, default=0) + vr_window - 1, vr_window)
    if p.get("expanding", False):
        last = max(last, vol_window + int(p.get("rank_min_periods", 240)) - 1)
    hl = p.get("vol_halflife")
    if hl is not None:
        last = max(last, max(2, int(float(hl) // 2)))
    return last + 1


#: H1 value of :func:`regime_lookback` with default parameters (registry ``lookback``).
REGIME_LOOKBACK = regime_lookback({}, 60.0)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio over ``n`` bars (NaN warm-up; 0 for a flat window)."""
    c = close.astype(float)
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    er = np.where(path.to_numpy() > 0, safe_div(net, path, fill=0.0), 0.0)
    return pd.Series(np.where(path.isna().to_numpy() | net.isna().to_numpy(), np.nan, er),
                     index=close.index)


def variance_ratio(logp: pd.Series, q: int, window: int) -> pd.Series:
    """Rolling Lo–MacKinlay variance ratio with overlapping q-bar returns.

    Both variances are measured over the same trailing ``window`` bars. For a random walk
    ``E[VR] ≈ 1`` (small-sample bias ~ -q/window is ignored — the feature is used relatively).
    """
    r1 = logp.diff()
    rq = logp - logp.shift(q)
    v1 = r1.rolling(window, min_periods=window).var(ddof=1)
    vq = rq.rolling(window, min_periods=window).var(ddof=1)
    return pd.Series(safe_div(vq, q * v1), index=logp.index)


def _bar_minutes_or_default(bars: pd.DataFrame) -> float:
    try:
        m = float(bar_minutes(bars))
    except ValueError:
        return _DEFAULT_BAR_MINUTES
    return m if math.isfinite(m) and m > 0 else _DEFAULT_BAR_MINUTES


@register_feature("regime", family="regime", lookback=REGIME_LOOKBACK)
def regime_features(
    md: MarketData,
    *,
    vol_window: int = 24,
    rank_years: float = DEFAULT_RANK_YEARS,
    rank_window: int | None = None,
    expanding: bool = False,
    rank_min_periods: int = 240,
    er_windows: Sequence[int] = (20, 100),
    vr_qs: Sequence[int] = (4, 16),
    vr_window: int = 240,
    trend_window: int = 120,
    vol_halflife: float | None = None,
    strength_threshold: float = 1.0,
    er_threshold: float = 0.3,
    high_vol_pct: float = 0.8,
    low_vol_pct: float = 0.2,
) -> pd.DataFrame:
    """Causal, sliding-window-stable regime descriptors.

    * ``regime_vol_pctrank`` — percentile rank (0..1, average ties) of the ``vol_window``-bar
      realised vol among the last ``W`` values, ``W = rank_window`` bars if given, else
      ``rank_years`` years of bars at the bars' own size (1 year = 5,796 H1 / 1,449 H4 /
      252 D1 bars). NaN until the window is full.
      ``regime_high_vol`` / ``regime_low_vol`` — rank above ``high_vol_pct`` / below
      ``low_vol_pct`` (NaN while the rank is undefined).
    * ``regime_vol_pctrank_exp`` (only with ``expanding=True``) — expanding rank among ALL
      past vols (``rank_min_periods`` minimum); depends on the history start.
    * ``regime_er_{n}`` — Kaufman efficiency ratio.
    * ``regime_vr_{q}`` — rolling variance ratio; ``regime_hurst_{q}`` — Hurst proxy.
    * ``regime_trend_strength`` — |ln(C_t/C_{t-n})| / (rms_n sqrt(n)), n=``trend_window``,
      ``rms_n`` = RMS of the same n one-bar log returns (``vol_halflife`` given: EWMA vol
      with that half-life instead, legacy); ``regime_trend_state`` — sign of that return
      when strength > ``strength_threshold`` AND ER(n) > ``er_threshold``, else 0.
    """
    bars = md.bars
    lc = log_close(bars)
    r = lc.diff()
    cols: dict[str, pd.Series | np.ndarray] = {}
    rv = r.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    w = _rank_window({"rank_window": rank_window, "rank_years": rank_years},
                     _bar_minutes_or_default(bars))
    rank = rv.rolling(w, min_periods=w).rank(pct=True)
    cols["regime_vol_pctrank"] = rank
    if expanding:
        cols["regime_vol_pctrank_exp"] = rv.expanding(min_periods=rank_min_periods).rank(pct=True)
    rk = rank.to_numpy()
    cols["regime_high_vol"] = np.where(np.isnan(rk), np.nan, (rk > high_vol_pct).astype(float))
    cols["regime_low_vol"] = np.where(np.isnan(rk), np.nan, (rk < low_vol_pct).astype(float))
    close = bars["close"].astype(float)
    ers: dict[int, pd.Series] = {}
    for n in sorted({int(x) for x in (*er_windows, trend_window)}):
        ers[n] = efficiency_ratio(close, n)
        if n in {int(x) for x in er_windows}:
            cols[f"regime_er_{n}"] = ers[n]
    for q in sorted({int(x) for x in vr_qs}):
        vr = variance_ratio(lc, q, vr_window)
        cols[f"regime_vr_{q}"] = vr
        cols[f"regime_hurst_{q}"] = 0.5 * (1.0 + log_pos(vr) / math.log(q))
    tw = int(trend_window)
    if vol_halflife is None:
        sig = np.sqrt((r * r).rolling(tw, min_periods=tw).mean())
    else:
        sig = ewm_bar_vol(r, vol_halflife, max(2, int(vol_halflife // 2)))
    ret_n = lc - lc.shift(tw)
    strength = np.abs(safe_div(ret_n, sig * math.sqrt(tw)))
    cols["regime_trend_strength"] = strength
    er_n = ers[tw].to_numpy()
    active = (strength > strength_threshold) & (er_n > er_threshold)
    state = np.where(active, np.sign(ret_n.to_numpy()), 0.0)
    cols["regime_trend_state"] = np.where(np.isnan(strength) | np.isnan(er_n), np.nan, state)
    out = pd.DataFrame(cols, index=bars.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


regime_features.lookback_fn = regime_lookback  # type: ignore[attr-defined]
