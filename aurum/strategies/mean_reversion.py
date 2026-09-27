"""Mean-reversion strategies: ``zscore_fade``, ``rsi2``, ``bollinger_revert``.

Economic rationale
------------------
At short horizons, price moves driven by order-flow imbalances (rather than information)
tend to partially reverse: liquidity providers absorb the imbalance and demand a
premium, so prices overshoot and then revert as the inventory is laid off (Grossman &
Miller 1988; Nagel 2012). Behavioural overreaction to salient moves adds to the effect
(De Bondt & Thaler 1985; Lo & MacKinlay 1990). Spot gold is a deep OTC market dominated by
dealers, with clustered liquidity (London fixes, COMEX open) and frequent stop-runs in thin
hours, which creates intraday overshoots. The effect is regime dependent: in trending,
information-driven markets fading moves is ruinous, so each rule here carries a regime or
trend filter.

All forecasts are long/short and stateful through the vectorised :func:`latch` state
machine (entries on stretched prices, exits on reversion or when the regime turns). The
magnitude is continuous: proportional to how stretched price still is, so conviction
decays as the trade works (expected reversion ∝ displacement).

References
----------
* Grossman, S. & Miller, M. (1988). "Liquidity and Market Structure". J. Finance 43(3).
* Nagel, S. (2012). "Evaporating Liquidity". Review of Financial Studies 25(7), 2005-2039.
* De Bondt, W. & Thaler, R. (1985). "Does the Stock Market Overreact?". J. Finance 40(3).
* Lo, A. & MacKinlay, A. C. (1990). "When Are Contrarian Profits Due to Stock Market
  Overreaction?". Review of Financial Studies 3(2), 175-205.
* Kaufman, P. (1995). *Smarter Trading*. McGraw-Hill — efficiency ratio.
* Connors, L. & Alvarez, C. (2008). *Short Term Trading Strategies That Work*. TradingMarkets.
* Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.regime import efficiency_ratio
from aurum.features.technical import rolling_zscore, rsi
from aurum.features.volatility import safe_div
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import check_params, latch, log_close

logger = logging.getLogger(__name__)

__all__ = ["BollingerRevert", "RSI2", "ZScoreFade"]


def _bool(x: pd.Series | np.ndarray) -> np.ndarray:
    """Boolean numpy array with NaN comparisons already False (pandas semantics)."""
    return np.asarray(x, dtype=bool)


def _prev(x: np.ndarray) -> np.ndarray:
    """``x`` shifted by one bar (first element False)."""
    out = np.zeros_like(x)
    out[1:] = x[:-1]
    return out


# ---------------------------------------------------------------------------------------
# zscore_fade
# ---------------------------------------------------------------------------------------
@register_strategy
class ZScoreFade(Strategy):
    """Fade large deviations of log price from its rolling mean, only in non-trending regimes.

    Rationale: when the path of prices is inefficient (Kaufman's efficiency ratio
    ``|C_t - C_{t-n}| / sum|dC|`` is low, i.e. lots of back-and-forth for little net
    progress), large deviations from the local mean are more likely liquidity-driven
    overshoots than information, and tend to revert (Lo & MacKinlay 1990; Nagel 2012).
    When the ER is high the market is trending and fading is switched off (and open fades
    are closed) — the classic failure mode of mean reversion. The regime is measured on
    the window ending at the PREVIOUS bar, so the shock being faded does not itself make
    the path look inefficient and open the gate.

    Rules: ``z = (ln C - mean_n) / std_n`` over ``n`` bars. Enter long when
    ``z < -entry_z`` and ``ER_er_n[t-1] < er_max``; short symmetric. Exit a long when
    ``z >= -exit_z`` (reverted) or the regime gate closes. Forecast while in a trade:
    ``sign * min(1, |z| / (2 entry_z))`` — 0.5 at entry, fading to ~0 at the exit band.
    Defaults (H1): n = 48 (~2 days), entry 2.0, exit 0.5, ER(48) < 0.3.

    References: Kaufman (1995) *Smarter Trading*; Lo & MacKinlay (1990) RFS 3(2);
    Nagel (2012) RFS 25(7).
    """

    name = "zscore_fade"
    description = ("Fade |z| > 2 of log price vs its 48-bar mean, only when Kaufman's efficiency "
                   "ratio says the market is not trending; exit on reversion.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n": 48, "entry_z": 2.0, "exit_z": 0.5, "er_n": 48, "er_max": 0.3}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["n"], p["er_n"] = int(p["n"]), int(p["er_n"])
        if p["n"] < 3 or p["er_n"] < 2:
            raise ValueError("n must be >= 3 and er_n >= 2")
        if not 0.0 <= float(p["exit_z"]) < float(p["entry_z"]):
            raise ValueError("need 0 <= exit_z < entry_z")
        if p["er_max"] is not None and not 0.0 < float(p["er_max"]) <= 1.0:
            raise ValueError("er_max must be in (0, 1] or None")

    @property
    def warmup_bars(self) -> int:
        p = self.params
        return max(p["n"], p["er_n"] + 1 if p["er_max"] is not None else 0) + 1

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        z = rolling_zscore(log_close(bars), p["n"]).to_numpy()
        if p["er_max"] is not None:
            er = efficiency_ratio(bars["close"], p["er_n"]).shift(1).to_numpy()
            gate = _bool(er < float(p["er_max"]))
        else:
            gate = np.isfinite(z)
        k_in, k_out = float(p["entry_z"]), float(p["exit_z"])
        state = latch(
            entry_long=_bool(z < -k_in) & gate,
            entry_short=_bool(z > k_in) & gate,
            exit_long=_bool(z >= -k_out) | ~gate,
            exit_short=_bool(z <= k_out) | ~gate,
        )
        mag = np.minimum(1.0, np.abs(z) / (2.0 * k_in))
        return self._finalize(pd.Series(state * mag, index=bars.index), bars.index)


# ---------------------------------------------------------------------------------------
# rsi2
# ---------------------------------------------------------------------------------------
@register_strategy
class RSI2(Strategy):
    """Connors RSI(2) pullback strategy with a long-term trend filter.

    Rationale: buy short, sharp pullbacks *within* an uptrend (and sell rallies within a
    downtrend). The trend filter keeps the trade aligned with the slower momentum premium
    while the entry exploits the short-horizon liquidity-driven reversal (Connors & Alvarez
    2008 document it on equity indices; the mechanism — overshoot and inventory
    rebalancing — is generic).

    Rules: long when ``RSI(rsi_n) < lower`` and ``close > SMA(trend_n)``; exit when
    ``close > SMA(exit_n)``. Short when ``RSI > upper`` and ``close < SMA(trend_n)``; exit
    when ``close < SMA(exit_n)``. ``trend_n=None`` disables the trend filter. The forecast
    magnitude is set at (each) entry signal from the RSI depth,
    ``0.5 + 0.5 * (lower - RSI) / lower`` (0.5 at the threshold, 1.0 at RSI = 0), and held.
    Defaults (bar counts as in Connors, applied to H1): RSI(2) < 10 / > 90, SMA(200)
    filter, SMA(5) exit.

    References: Connors & Alvarez (2008) *Short Term Trading Strategies That Work*;
    Wilder (1978) *New Concepts in Technical Trading Systems* (RSI).
    """

    name = "rsi2"
    description = ("Connors RSI(2) pullback: buy RSI2<10 above the 200-bar SMA (sell RSI2>90 "
                   "below it), exit on a close through the 5-bar SMA.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"rsi_n": 2, "lower": 10.0, "upper": 90.0, "trend_n": 200, "exit_n": 5}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["rsi_n"], p["exit_n"] = int(p["rsi_n"]), int(p["exit_n"])
        if p["trend_n"] is not None:
            p["trend_n"] = int(p["trend_n"])
            if p["trend_n"] < 2:
                raise ValueError("trend_n must be >= 2 or None")
        if p["rsi_n"] < 1 or p["exit_n"] < 1:
            raise ValueError("rsi_n and exit_n must be >= 1")
        if not 0.0 < float(p["lower"]) < float(p["upper"]) < 100.0:
            raise ValueError("need 0 < lower < upper < 100")

    @property
    def warmup_bars(self) -> int:
        p = self.params
        # Wilder smoothing is seeded with the first change: allow ~5 RSI periods to settle.
        return max(p["trend_n"] or 0, p["exit_n"], 5 * p["rsi_n"]) + 1

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        close = bars["close"].astype(float)
        c = close.to_numpy()
        r = rsi(close, p["rsi_n"]).to_numpy()
        lo, hi = float(p["lower"]), float(p["upper"])
        if p["trend_n"] is not None:
            sma_t = close.rolling(p["trend_n"], min_periods=p["trend_n"]).mean().to_numpy()
            up, down = _bool(c > sma_t), _bool(c < sma_t)
        else:
            up = down = np.isfinite(r)
        sma_x = close.rolling(p["exit_n"], min_periods=p["exit_n"]).mean().to_numpy()
        entry_long = _bool(r < lo) & up
        entry_short = _bool(r > hi) & down
        state = latch(entry_long, entry_short, _bool(c > sma_x), _bool(c < sma_x))
        depth = np.full(len(c), np.nan)
        depth[entry_long] = 0.5 + 0.5 * (lo - r[entry_long]) / lo
        depth[entry_short] = 0.5 + 0.5 * (r[entry_short] - hi) / (100.0 - hi)
        mag = pd.Series(depth).ffill().to_numpy()
        return self._finalize(pd.Series(state * mag, index=bars.index), bars.index)


# ---------------------------------------------------------------------------------------
# bollinger_revert
# ---------------------------------------------------------------------------------------
@register_strategy
class BollingerRevert(Strategy):
    """Bollinger-band reversion: fade band excursions back to the middle band.

    Rationale: Bollinger bands measure displacement in units of recent dispersion; closes
    outside ``mid ± k·sd`` are statistically stretched. With ``confirm=True`` (default) the
    entry waits for the close to come back INSIDE the band — Bollinger's own
    recommendation (2001) to avoid fading a "walk up the band", i.e. a genuine breakout —
    and trades the reversion to the middle band. A protective exit at ``mid ± stop_k·sd``
    caps the loss when the excursion turns into a trend.

    Rules: long when the previous close was below the lower band and the current close is
    back inside (below the middle); exit at ``close >= mid`` or ``close < mid - stop_k sd``.
    Short symmetric. ``confirm=False`` enters directly on a close outside the band.
    Forecast while in a trade: ``sign * min(1, |close - mid| / (k sd))`` (≈1 at the band,
    → 0 at the middle). Defaults (H1): SMA/sd over 48 bars (~2 days), k = 2, stop 3.5 sd.

    References: Bollinger (2001) *Bollinger on Bollinger Bands*; Lo & MacKinlay (1990)
    RFS 3(2).
    """

    name = "bollinger_revert"
    description = ("Bollinger(48, 2) reversion: enter when price closes back inside a band, "
                   "target the middle band, stop at 3.5 sd.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n": 48, "k": 2.0, "stop_k": 3.5, "confirm": True}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["n"] = int(p["n"])
        if p["n"] < 3:
            raise ValueError("n must be >= 3")
        if not 0.0 < float(p["k"]) < float(p["stop_k"]):
            raise ValueError("need 0 < k < stop_k")
        p["confirm"] = bool(p["confirm"])

    @property
    def warmup_bars(self) -> int:
        return int(self.params["n"]) + 1

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        close = bars["close"].astype(float)
        c = close.to_numpy()
        roll = close.rolling(p["n"], min_periods=p["n"])
        mid = roll.mean().to_numpy()
        sd = roll.std(ddof=0).to_numpy()
        k, ks = float(p["k"]), float(p["stop_k"])
        below, above = _bool(c < mid - k * sd), _bool(c > mid + k * sd)
        stop_long, stop_short = _bool(c < mid - ks * sd), _bool(c > mid + ks * sd)
        if p["confirm"]:
            entry_long = _prev(below) & ~below & _bool(c < mid)
            entry_short = _prev(above) & ~above & _bool(c > mid)
        else:
            entry_long = below & ~stop_long
            entry_short = above & ~stop_short
        state = latch(entry_long, entry_short,
                      _bool(c >= mid) | stop_long, _bool(c <= mid) | stop_short)
        mag = np.minimum(1.0, safe_div(np.abs(c - mid), k * sd))
        return self._finalize(pd.Series(state * mag, index=bars.index), bars.index)
