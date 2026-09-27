"""Model-free market-regime descriptors (``regime``).

Strategy performance is strongly regime-dependent: trend followers need persistent,
efficient price paths; faders need noisy, mean-reverting ones; everything suffers in
volatility spikes. This module provides *causal, parameter-free* regime statistics. Fitted
latent-state models (Gaussian HMM) live in ``aurum.models.regime`` and must be trained on
TRAIN data only — nothing here is fitted.

* Volatility percentile — expanding-window rank of current realised vol against ALL past
  values (never the full sample; a full-sample quantile is the classic regime-feature
  leak), plus a rolling-window rank for a locally-relative view.
* Kaufman (1995) efficiency ratio ``|C_t - C_{t-n}| / sum |dC|`` in [0, 1]: 1 = straight
  line, ~0 = pure noise ("Smarter Trading", McGraw-Hill).
* Variance ratio ``VR(q) = Var(r^(q)) / (q Var(r))`` (Lo & MacKinlay 1988, RFS 1(1)):
  > 1 trending/persistent, < 1 mean-reverting. Hurst proxy ``H = 0.5 (1 + ln VR / ln q)``
  from ``VR(q) ~ q^(2H-1)`` for fractional Gaussian noise.
* Trend strength: |vol-normalised n-bar return| and a discrete trend state in {-1, 0, 1}
  that is non-zero only when both strength and efficiency are high.
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
from aurum.features.volatility import log_pos, safe_div

logger = logging.getLogger(__name__)

__all__ = ["efficiency_ratio", "regime_features", "regime_lookback", "variance_ratio"]

REGIME_LOOKBACK = 24 + 240


def regime_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    """Warm-up (bars): the vol ranks need ``vol_window`` returns then ``rank_min_periods``
    vol values; ``VR(q)`` needs ``q + vr_window - 1`` rows; ER/trend need their window."""
    vol_window = int(p.get("vol_window", 24))
    trend_window = int(p.get("trend_window", 120))
    ers = [int(x) for x in p.get("er_windows", (20, 100))]
    qs = [int(x) for x in p.get("vr_qs", (4, 16))]
    vr_window = int(p.get("vr_window", 240))
    last = max(vol_window + int(p.get("rank_min_periods", 240)) - 1, max([*ers, trend_window]),
               max(qs, default=0) + vr_window - 1, vr_window,
               max(2, int(float(p.get("vol_halflife", 48.0)) // 2)))
    return last + 1


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


@register_feature("regime", family="regime", lookback=REGIME_LOOKBACK)
def regime_features(
    md: MarketData,
    *,
    vol_window: int = 24,
    rank_window: int = 2000,
    rank_min_periods: int = 240,
    er_windows: Sequence[int] = (20, 100),
    vr_qs: Sequence[int] = (4, 16),
    vr_window: int = 240,
    trend_window: int = 120,
    vol_halflife: float = 48.0,
    strength_threshold: float = 1.0,
    er_threshold: float = 0.3,
    high_vol_pct: float = 0.8,
    low_vol_pct: float = 0.2,
) -> pd.DataFrame:
    """Causal regime descriptors.

    * ``regime_vol_pctrank_exp`` — expanding percentile rank (0..1) of the ``vol_window``-bar
      realised vol among all past values (``rank_min_periods`` minimum);
      ``regime_vol_pctrank_{rank_window}`` — the same over a rolling window;
      ``regime_high_vol`` / ``regime_low_vol`` flags from the expanding rank.
    * ``regime_er_{n}`` — Kaufman efficiency ratio.
    * ``regime_vr_{q}`` — rolling variance ratio; ``regime_hurst_{q}`` — Hurst proxy.
    * ``regime_trend_strength`` — |ln(C_t/C_{t-n})| / (sigma_t sqrt(n)), n=``trend_window``;
      ``regime_trend_state`` — sign of that return when strength > ``strength_threshold``
      AND ER(n) > ``er_threshold``, else 0.
    """
    bars = md.bars
    lc = log_close(bars)
    r = lc.diff()
    cols: dict[str, pd.Series | np.ndarray] = {}
    rv = r.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    rank_exp = rv.expanding(min_periods=rank_min_periods).rank(pct=True)
    cols["regime_vol_pctrank_exp"] = rank_exp
    cols[f"regime_vol_pctrank_{rank_window}"] = rv.rolling(
        rank_window, min_periods=rank_min_periods).rank(pct=True)
    re = rank_exp.to_numpy()
    cols["regime_high_vol"] = np.where(np.isnan(re), np.nan, (re > high_vol_pct).astype(float))
    cols["regime_low_vol"] = np.where(np.isnan(re), np.nan, (re < low_vol_pct).astype(float))
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
    sig = ewm_bar_vol(r, vol_halflife, max(2, int(vol_halflife // 2)))
    ret_n = lc - lc.shift(trend_window)
    strength = np.abs(safe_div(ret_n, sig * math.sqrt(trend_window)))
    cols["regime_trend_strength"] = strength
    er_n = ers[int(trend_window)].to_numpy()
    active = (strength > strength_threshold) & (er_n > er_threshold)
    state = np.where(active, np.sign(ret_n.to_numpy()), 0.0)
    cols["regime_trend_state"] = np.where(np.isnan(strength) | np.isnan(er_n), np.nan, state)
    out = pd.DataFrame(cols, index=bars.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


regime_features.lookback_fn = regime_lookback  # type: ignore[attr-defined]
