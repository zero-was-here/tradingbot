"""Timeframe definitions shared by every layer.

Conventions (see SPEC.md §1):
  * Bars are indexed by their OPEN time in UTC (tz-aware).
  * A bar becomes *available* (tradeable information) at ``open + duration``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Timeframe:
    name: str          # "M1", "M5", "M15", "M30", "H1", "H4", "D1"
    freq: str          # pandas offset alias usable with resample(), e.g. "5min", "1h", "1D"
    minutes: int

    @property
    def delta(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.minutes)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


_TIMEFRAMES = {
    "M1": Timeframe("M1", "1min", 1),
    "M5": Timeframe("M5", "5min", 5),
    "M15": Timeframe("M15", "15min", 15),
    "M30": Timeframe("M30", "30min", 30),
    "H1": Timeframe("H1", "1h", 60),
    "H4": Timeframe("H4", "4h", 240),
    "D1": Timeframe("D1", "1D", 1440),
}


def get_timeframe(tf: str | Timeframe) -> Timeframe:
    """Parse "H1" / "h1" / Timeframe into a Timeframe."""
    if isinstance(tf, Timeframe):
        return tf
    key = str(tf).upper()
    if key not in _TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {tf!r}; expected one of {sorted(_TIMEFRAMES)}")
    return _TIMEFRAMES[key]


def all_timeframes() -> list[Timeframe]:
    return list(_TIMEFRAMES.values())


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Empirical number of bars per year for annualisation.

    Gold trades ~23h x 5d with holiday gaps, so ``365*24`` is wrong for H1.
    We measure the realised density over the sample instead.
    """
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer bars per year")
    span_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    if span_years <= 0:
        raise ValueError("Index must be increasing")
    return (len(index) - 1) / span_years
