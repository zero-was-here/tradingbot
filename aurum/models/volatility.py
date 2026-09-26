"""Volatility forecasting.

``ewma_volatility`` is the default causal forecaster used by the backtest engine and live
runner. Additional models (GARCH, HAR-RV, range-based blends) live in this module too and
must keep the same causal contract: value at row t uses returns up to and including t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurum.core.timeframes import infer_bars_per_year


def ewma_volatility(
    close: pd.Series,
    *,
    halflife_bars: float = 48.0,
    bars_per_year: float | None = None,
    min_periods: int = 20,
    floor: float = 0.03,
    cap: float = 2.0,
) -> pd.Series:
    """Annualised EWMA volatility of log returns, causal (uses returns up to t).

    Returns a fraction (0.15 == 15% annualised), clipped to [floor, cap]. Warm-up rows get a
    conservative constant 0.20 (never back-filled from future estimates).
    """
    bpy = bars_per_year or infer_bars_per_year(close.index)
    r = np.log(close).diff()
    var = (r**2).ewm(halflife=halflife_bars, min_periods=min_periods, adjust=False).mean()
    vol = np.sqrt(var * bpy)
    vol = vol.clip(lower=floor, upper=cap)
    return vol.fillna(0.20).rename("vol_ann")
