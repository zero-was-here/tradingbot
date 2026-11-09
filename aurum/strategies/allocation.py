"""Long-biased allocation strategies: ``managed_gold``.

Economic rationale
------------------
The pre-registered walk-forward (docs/RESULTS.md) found no demonstrated *alpha* in the
long/short rule set, while gold's own drift dominated the sample (buy-and-hold ~6.8%/yr
over 2016-2024, at ~16% volatility and a -31% drawdown). The institutional answer to "I
want gold exposure but not its drawdowns" is not an alpha model but a **managed exposure**
product: a strategic long allocation that is tilted by trend and cut when the market is in
a falling, turbulent regime. It earns the asset's risk premium most of the time and gives
up some of it in exchange for shallower drawdowns (the "trend overlay" / "trend-following
risk management" of Clare et al. 2016 and Hurst, Ooi & Pedersen 2017; the moving-average
timing rule of Faber 2007; the volatility-managed portfolios of Moreira & Muir 2017).

Why the three ingredients:

* **Strategic long weight.** Gold has had a positive long-run real return in US dollars
  and acts as a crisis hedge (Baur & Lucey 2010; Erb & Harvey 2013). Holding a constant
  fraction of the risk budget long is the zero-skill baseline the overlay must beat.
* **Trend tilt.** Time-series momentum is the best documented anomaly in futures including
  gold (Moskowitz, Ooi & Pedersen 2012). Here it only *tilts* the allocation around the
  base weight (no short by default), so a wrong trend call costs a fraction of the base
  instead of a full reversal.
* **Downtrend + high-volatility de-risk.** Drawdowns of long-only assets are concentrated in
  periods when price is below its long moving average AND volatility is elevated: returns
  there have been poor and volatility high, so the Sharpe of staying invested is lowest
  (Faber 2007; Moreira & Muir 2017; Harvey et al. 2018 "The impact of volatility
  targeting"). The rule cuts the forecast to ``floor`` in that regime, with hysteresis so
  it does not flicker.

Forecast
--------
::

    trend_t  = FDM * mean_h( z_h * 0.6267 )             Carver-scaled, E|trend| = 0.5 under H0
    z_h      = ln(C_t / C_{t-h}) / (sigma_t * sqrt(h))   vol-normalised h-bar log return
    raw_t    = base_weight + tilt_weight * clip(trend_t, -1, 1)
    derisk_t = latch(enter: ln C_t < EMA_t  and  vol_ratio_t >= vol_enter,
                     exit : ln C_t >= EMA_t  or  vol_ratio_t <  vol_exit)
    raw_t    = min(raw_t, floor)      when derisk_t = 1
    f_t      = clip(raw_t, 0 (or -1 with allow_short), 1)

``vol_ratio`` is the fast/slow EWMA volatility ratio (scale-free, ~1 in a stationary
market), ``EMA`` a normalised exponential moving average of log price (``adjust=True``, so
research on the full history and live on a shorter window agree once a few half-lives
have elapsed). Everything is causal: row ``t`` uses closes up to ``t`` only.

Defaults are for H1 bars (~23 bars/day, 5,796/year): horizons 1/3/6/12 months, an EMA
half-life of 1,600 bars (~70 trading days, roughly a "200-day" EMA), fast/slow vol
half-lives of 1 week / 6 months. For daily bars use 21/63/126/252, 69, 5 and 126 (see
``configs/managed_gold.yaml``).

Interpretation for sizing: with the default ``VolTargetSizer`` (10% target) a forecast of
0.5 holds ~5% annualised volatility of gold, i.e. about a third of a fully invested
position at gold's ~16% volatility. This is an allocation product, not an alpha claim, and
every configuration of it evaluated on real data counts as a trial for the DSR.

References
----------
* Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). "Time series momentum". JFE 104(2).
* Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). "A Century of Evidence on Trend-Following
  Investing". JPM 44(1).
* Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation". Journal of
  Wealth Management 9(4).
* Clare, A., Seaton, J., Smith, P. & Thomas, S. (2016). "The trend is our friend: Risk
  parity, momentum and trend following in global asset allocation". J. Behavioral and
  Experimental Finance 9.
* Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios". Journal of Finance 72(4).
* Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & Van Hemert, O.
  (2018). "The Impact of Volatility Targeting". JPM 45(1).
* Baur, D. & Lucey, B. (2010). "Is Gold a Hedge or a Safe Haven?". Financial Review 45(2).
* Erb, C. & Harvey, C. (2013). "The Golden Dilemma". FAJ 69(4).
* Carver, R. (2015). *Systematic Trading*. Harriman House (forecast scaling, FDM).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import (
    Z_FORECAST_SCALAR,
    bar_volatility,
    check_params,
    diversification_multiplier,
    ewma_mean,
    latch,
    log_close,
    momentum_z,
    overlap_correlation,
)

logger = logging.getLogger(__name__)

__all__ = ["ManagedGold"]


def _positive_int_tuple(values: Any, what: str) -> tuple[int, ...]:
    if isinstance(values, (int, np.integer)):
        values = (values,)
    out = tuple(int(v) for v in values)
    if not out or any(v < 1 for v in out):
        raise ValueError(f"{what} must be a non-empty sequence of positive integers, got {values!r}")
    return out


@register_strategy
class ManagedGold(Strategy):
    """Managed gold exposure: base long weight + trend tilt − downtrend/high-vol de-risk.

    See the module docstring for the rationale, the exact formula and references.

    Parameters (defaults for H1 bars)
    ---------------------------------
    base_weight : strategic forecast when the trend is flat (default 0.5).
    tilt_weight : multiplier of the Carver-scaled trend forecast in [-1, 1] (default 0.5,
                  so the undamped forecast spans [base - 0.5, base + 0.5] = [0, 1]).
    horizons    : momentum lookbacks in bars (default 480/1440/2880/5796 ≈ 1/3/6/12 months).
    trend_halflife : half-life (bars) of the EMA of log price defining the downtrend
                  (default 1600 ≈ 70 trading days).
    vol_fast_halflife / vol_slow_halflife : EWMA half-lives (bars) of the volatility ratio
                  (defaults 120 ≈ 1 week, 2880 ≈ 6 months).
    vol_enter / vol_exit : hysteresis thresholds of the fast/slow vol ratio (1.25 / 1.0).
    floor       : cap on the forecast while de-risked (default 0.0 = go to cash).
    allow_short : lower bound -1 instead of 0 (default False: long-only).
    vol_halflife / vol_min_periods : EWMA per-bar volatility used to normalise momentum
                  (as in ``tsmom``: 240 / 120).
    """

    name = "managed_gold"
    description = ("Managed gold exposure: long-biased base weight 0.5 + multi-horizon trend tilt, "
                   "cut to a floor when price is below its long EMA in a high-vol regime; long-only.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "base_weight": 0.5,
            "tilt_weight": 0.5,
            "horizons": (480, 1440, 2880, 5796),
            "trend_halflife": 1600.0,
            "vol_fast_halflife": 120.0,
            "vol_slow_halflife": 2880.0,
            "vol_enter": 1.25,
            "vol_exit": 1.0,
            "floor": 0.0,
            "allow_short": False,
            "vol_halflife": 240.0,
            "vol_min_periods": 120,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        p["horizons"] = _positive_int_tuple(p["horizons"], "horizons")
        for k in ("base_weight", "tilt_weight", "floor", "vol_enter", "vol_exit"):
            p[k] = float(p[k])
            if not math.isfinite(p[k]):
                raise ValueError(f"{k} must be finite")
        p["allow_short"] = bool(p["allow_short"])
        lo = -1.0 if p["allow_short"] else 0.0
        if not lo <= p["base_weight"] <= 1.0:
            raise ValueError(f"base_weight must be in [{lo:g}, 1], got {p['base_weight']}")
        if not 0.0 <= p["tilt_weight"] <= 2.0:
            raise ValueError(f"tilt_weight must be in [0, 2], got {p['tilt_weight']}")
        if not lo <= p["floor"] <= 1.0:
            raise ValueError(f"floor must be in [{lo:g}, 1], got {p['floor']}")
        if not 0.0 < p["vol_exit"] <= p["vol_enter"]:
            raise ValueError("need 0 < vol_exit <= vol_enter (hysteresis)")
        for k in ("trend_halflife", "vol_fast_halflife", "vol_slow_halflife", "vol_halflife"):
            p[k] = float(p[k])
            if not p[k] > 0:
                raise ValueError(f"{k} must be > 0")
        if not p["vol_fast_halflife"] < p["vol_slow_halflife"]:
            raise ValueError("vol_fast_halflife must be < vol_slow_halflife")
        p["vol_min_periods"] = int(p["vol_min_periods"])
        if p["vol_min_periods"] < 2:
            raise ValueError("vol_min_periods must be >= 2")

    # ------------------------------------------------------------------------------------
    @property
    def warmup_bars(self) -> int:
        """Longest momentum horizon, one EMA half-life and one slow-vol half-life.

        The EMA and the vol EWMAs are normalised (``adjust=True``), so they are honest
        estimates from however much history exists; their ``min_periods`` (one half-life)
        is what makes the first non-zero forecast reproducible on a shorter live window.
        """
        p = self.params
        return max(max(p["horizons"]), self._min_periods(p["trend_halflife"]),
                   self._min_periods(p["vol_slow_halflife"]), p["vol_min_periods"]) + 1

    @staticmethod
    def _min_periods(halflife: float) -> int:
        return max(2, int(math.ceil(halflife)))

    def combination(self) -> tuple[np.ndarray, float]:
        """(equal weights over horizons, forecast diversification multiplier under H0)."""
        h = self.params["horizons"]
        w = np.full(len(h), 1.0 / len(h))
        return w, diversification_multiplier(overlap_correlation(h), w)

    def trend_forecast(self, lc: pd.Series, sigma: pd.Series) -> pd.Series:
        """Carver-scaled multi-horizon momentum in [-1, 1] (NaN where undefined)."""
        w, fdm = self.combination()
        acc = np.zeros(len(lc))
        for wi, h in zip(w, self.params["horizons"], strict=True):
            acc = acc + wi * momentum_z(lc, sigma, h).to_numpy() * Z_FORECAST_SCALAR
        return pd.Series(np.clip(acc * fdm, -1.0, 1.0), index=lc.index, name="trend")

    def components(self, md: MarketData) -> pd.DataFrame:
        """Every intermediate series of the rule (for reports, docs and the dashboard).

        Columns: ``trend`` (Carver-scaled momentum), ``tilt`` (= tilt_weight * trend),
        ``ema`` (EMA of log price), ``below_ema`` (bool), ``vol_ratio`` (fast/slow vol),
        ``derisk`` (latched 0/1 state), ``raw`` (before bounds) and ``forecast`` (final).
        """
        p = self.params
        bars = md.bars
        lc = log_close(bars)
        sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
        trend = self.trend_forecast(lc, sigma)
        ema = ewma_mean(lc, p["trend_halflife"], self._min_periods(p["trend_halflife"]))
        vol_fast = bar_volatility(lc, p["vol_fast_halflife"], p["vol_min_periods"])
        vol_slow = bar_volatility(lc, p["vol_slow_halflife"], self._min_periods(p["vol_slow_halflife"]))
        ratio = (vol_fast / vol_slow).to_numpy(dtype=float)
        lcv, emav = lc.to_numpy(dtype=float), ema.to_numpy(dtype=float)
        valid = np.isfinite(ratio) & np.isfinite(emav)
        below = valid & (lcv < emav)
        enter = below & (ratio >= p["vol_enter"])
        exit_ = valid & (~below | (ratio < p["vol_exit"]))
        none = np.zeros(len(bars), dtype=bool)
        derisk = latch(enter, none, exit_, none)
        tilt = p["tilt_weight"] * np.nan_to_num(trend.to_numpy(dtype=float), nan=0.0)
        raw = p["base_weight"] + tilt
        raw = np.where(derisk > 0, np.minimum(raw, p["floor"]), raw)
        lo = -1.0 if p["allow_short"] else 0.0
        forecast = self._finalize(pd.Series(np.clip(raw, lo, 1.0), index=bars.index), bars.index)
        return pd.DataFrame({
            "trend": trend.to_numpy(), "tilt": tilt, "ema": emav, "below_ema": below,
            "vol_ratio": ratio, "derisk": derisk, "raw": raw, "forecast": forecast.to_numpy(),
        }, index=bars.index)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        return self.components(md)["forecast"].rename(self.name)
