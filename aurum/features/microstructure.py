"""Microstructure (``microstructure``) and trading-session (``session``) features.

Microstructure: transaction-cost and liquidity state. For a retail XAUUSD CFD the quoted
spread and tick volume are the only liquidity observables; they matter for *net* alpha
(a signal worth 0.2 ATR is worthless when the spread is 0.3 ATR) and flag stressed markets
(spread blow-outs around news and the daily rollover).

Session: gold is a 23h market whose liquidity, volatility and drift are strongly
time-of-day dependent (Asian physical demand, the London LBMA auctions at 10:30/15:00
London, the COMEX/US data window from 08:20/08:30 New York). Session boundaries are
defined in LOCAL time and converted with ``zoneinfo`` so they follow DST in London and
New York independently (the two are out of sync for ~3 weeks each year).

Timing convention for ``session``: every column describes the DECISION instant
``available_at[t]`` (the close of bar t; the next fill happens at the open of bar t+1, which
equals that instant except across market breaks such as weekends). Clock times are known in
advance, so this is not look-ahead; it answers "which session will my next fill and
holding period be in", which is what a forecast made at the close of t needs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.base import register_feature
from aurum.features.technical import rolling_zscore
from aurum.features.volatility import atr, bar_minutes, safe_div

logger = logging.getLogger(__name__)

__all__ = [
    "LONDON",
    "NEW_YORK",
    "TOKYO",
    "local_clock",
    "microstructure_features",
    "microstructure_lookback",
    "session_features",
]

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")

MICROSTRUCTURE_LOOKBACK = 121


def microstructure_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    """Warm-up (bars): rolling z-scores need ``z_window`` rows, the gap uses ATR of the
    previous bar (``atr_n``), and the lag-1 autocorrelation pairs ``r_t`` (from row 1) with
    ``r_{t-1}`` (from row 2) over ``autocorr_window`` rows."""
    return max(int(p.get("z_window", 120)), int(p.get("atr_n", 14)) + 1,
               int(p.get("autocorr_window", 120)) + 1)


@register_feature("microstructure", family="microstructure", lookback=MICROSTRUCTURE_LOOKBACK)
def microstructure_features(
    md: MarketData,
    *,
    z_window: int = 120,
    atr_n: int = 14,
    autocorr_window: int = 120,
) -> pd.DataFrame:
    """Spread, volume, gap and short-horizon autocorrelation state.

    * ``microstructure_spread_bps`` spread/close in basis points; ``_spread_z`` rolling z of
      log spread; ``_spread_atr`` spread/ATR (round-trip cost vs typical bar move);
      ``_range_spread`` (H-L)/spread (opportunity per unit of cost).
    * ``microstructure_volume_z`` rolling z of log(1+tick volume) (NaN if volume is
      absent/constant).
    * ``microstructure_gap_atr`` (O_t - C_{t-1}) / ATR_{t-1}; ``_after_gap`` flag = the bar
      opened after a session break (> 1.5 bar durations since the previous bar: weekends,
      holidays, daily maintenance).
    * ``microstructure_autocorr_{n}`` rolling lag-1 autocorrelation of log returns —
      negative values indicate bid/ask bounce or mean-reverting liquidity-provision flow
      (Roll 1984), positive values persistent order flow.
    """
    bars = md.bars
    close = bars["close"].astype(float)
    spread = bars["spread"].astype(float)
    a = atr(bars, atr_n)
    cols: dict[str, pd.Series | np.ndarray] = {}
    cols["microstructure_spread_bps"] = safe_div(spread * 1e4, close)
    log_spread = pd.Series(np.log(spread.where(spread > 0)), index=bars.index)
    cols["microstructure_spread_z"] = rolling_zscore(log_spread, z_window)
    cols["microstructure_spread_atr"] = safe_div(spread, a)
    cols["microstructure_range_spread"] = safe_div(bars["high"] - bars["low"], spread)
    vol = bars["volume"].astype(float)
    cols["microstructure_volume_z"] = rolling_zscore(np.log1p(vol.clip(lower=0.0)), z_window)
    cols["microstructure_gap_atr"] = safe_div(bars["open"] - close.shift(1), a.shift(1))
    step = bars.index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    minutes = bar_minutes(bars)
    cols["microstructure_after_gap"] = np.where(np.isnan(step), np.nan,
                                                (step > 1.5 * minutes).astype(float))
    r = pd.Series(np.log(close.to_numpy()), index=bars.index).diff()
    cols[f"microstructure_autocorr_{autocorr_window}"] = r.rolling(
        autocorr_window, min_periods=autocorr_window).corr(r.shift(1))
    out = pd.DataFrame(cols, index=bars.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


def local_clock(times: pd.DatetimeIndex, tz: ZoneInfo) -> tuple[np.ndarray, np.ndarray]:
    """(weekday Mon=0..Sun=6, fractional hour) of UTC ``times`` in local zone ``tz``.

    ``tz_convert`` with a ``ZoneInfo`` applies the historical DST rules of that zone, so a
    London 08:00 open maps to 07:00 UTC in summer (BST) and 08:00 UTC in winter (GMT).
    """
    loc = times.tz_convert(tz)
    hour = loc.hour.to_numpy() + loc.minute.to_numpy() / 60.0 + loc.second.to_numpy() / 3600.0
    return loc.weekday.to_numpy(), hour


def _in_window(wd: np.ndarray, hour: np.ndarray, start: float, end: float,
               weekdays_only: bool = True) -> np.ndarray:
    m = (hour >= start) & (hour < end)
    if weekdays_only:
        m &= wd < 5
    return m.astype(float)


@register_feature("session", family="session", lookback=0)
def session_features(
    md: MarketData,
    *,
    asia_hours: tuple[float, float] = (8.0, 17.0),
    london_hours: tuple[float, float] = (8.0, 17.0),
    ny_hours: tuple[float, float] = (8.0, 17.0),
    rollover_hours: tuple[float, float] = (16.0, 18.0),
) -> pd.DataFrame:
    """Clock and session features of the decision instant ``available_at[t]``.

    * ``session_hour_{sin,cos}``: UTC time of day (cyclical encoding, no 23->0 jump).
    * ``session_how_{sin,cos}``: UTC hour of week (Monday 00:00 = 0).
    * ``session_ny_hour_{sin,cos}``: New York local time of day (DST-aware) — the clock on
      which US data (08:30 ET), COMEX and the 17:00 ET broker rollover are scheduled.
    * ``session_asia`` (Tokyo local ``asia_hours``, no DST), ``session_london``
      (Europe/London local ``london_hours``), ``session_ny`` (America/New_York local
      ``ny_hours``), ``session_overlap`` (London ∩ New York), all Mon–Fri local.
    * ``session_rollover``: New York local time in ``rollover_hours`` (thin liquidity, wide
      spreads, swap charged).
    * ``session_week_progress``: hours since the Sunday 17:00 ET open / 120, in [0, 1].
    * ``session_friday_late``: Friday after 12:00 ET (pre-weekend de-risking flows).
    """
    bars = md.bars
    t = pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC")
    utc_hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    utc_wd = t.weekday.to_numpy()
    how = utc_wd * 24.0 + utc_hour
    cols: dict[str, np.ndarray] = {
        "session_hour_sin": np.sin(2.0 * math.pi * utc_hour / 24.0),
        "session_hour_cos": np.cos(2.0 * math.pi * utc_hour / 24.0),
        "session_how_sin": np.sin(2.0 * math.pi * how / 168.0),
        "session_how_cos": np.cos(2.0 * math.pi * how / 168.0),
    }
    ny_wd, ny_h = local_clock(t, NEW_YORK)
    ldn_wd, ldn_h = local_clock(t, LONDON)
    tk_wd, tk_h = local_clock(t, TOKYO)
    cols["session_ny_hour_sin"] = np.sin(2.0 * math.pi * ny_h / 24.0)
    cols["session_ny_hour_cos"] = np.cos(2.0 * math.pi * ny_h / 24.0)
    cols["session_asia"] = _in_window(tk_wd, tk_h, *asia_hours)
    cols["session_london"] = _in_window(ldn_wd, ldn_h, *london_hours)
    cols["session_ny"] = _in_window(ny_wd, ny_h, *ny_hours)
    cols["session_overlap"] = cols["session_london"] * cols["session_ny"]
    cols["session_rollover"] = _in_window(ny_wd, ny_h, *rollover_hours, weekdays_only=False)
    since_open = np.mod(ny_wd * 24.0 + ny_h + 7.0, 168.0)  # Sunday 17:00 ET -> 0
    cols["session_week_progress"] = np.clip(since_open / 120.0, 0.0, 1.0)
    cols["session_friday_late"] = ((ny_wd == 4) & (ny_h >= 12.0)).astype(float)
    return pd.DataFrame(cols, index=bars.index, dtype=float)


def _no_lookback(p: Mapping[str, Any], bar_minutes: float = 60.0) -> int:
    """Clock features need no history."""
    return 0


microstructure_features.lookback_fn = microstructure_lookback  # type: ignore[attr-defined]
session_features.lookback_fn = _no_lookback  # type: ignore[attr-defined]
