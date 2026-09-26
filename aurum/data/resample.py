"""Leak-free resampling between timeframes.

Higher-timeframe (HTF) bars are labelled by their OPEN time (like MT5) and carry an
``available_at`` equal to the HTF bar's END. The legacy code labelled H1 bars at 00:00 and
forward-filled them onto M5 bars from 00:00, leaking 55 minutes of future. Here, an HTF bar
is only visible to base bars whose own ``available_at`` >= the HTF bar's ``available_at``.

An HTF bar is emitted only if it is *complete* with respect to the base data: the last
base bar that falls inside it must end exactly at the HTF bar's end, OR a later base bar
exists (market closed early, e.g. Friday). The trailing in-progress HTF bar is dropped.
"""

from __future__ import annotations

import pandas as pd

from aurum.core.timeframes import Timeframe, get_timeframe
from aurum.data.pit import asof_join
from aurum.data.schema import BAR_COLS, validate_bars

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "spread": "mean",
}


def resample_bars(
    bars: pd.DataFrame,
    to: str | Timeframe,
    *,
    daily_anchor_hour_utc: int = 0,
) -> pd.DataFrame:
    """Aggregate canonical bars to a higher timeframe.

    ``daily_anchor_hour_utc`` shifts D1/H4 bucket boundaries (e.g. 21 or 22 to align days to
    the New York 17:00 close used by most brokers). Default 0 = UTC midnight.
    """
    validate_bars(bars)
    tf = get_timeframe(to)
    offset = pd.Timedelta(hours=daily_anchor_hour_utc) if tf.minutes >= 240 else pd.Timedelta(0)
    grouped = bars.resample(tf.freq, label="left", closed="left", offset=offset)
    out = grouped.agg(_AGG)
    last_base_end = grouped["available_at"].max()
    out = out.dropna(subset=["open", "high", "low", "close"])
    last_base_end = last_base_end.reindex(out.index)

    htf_end = out.index + tf.delta
    # Complete if the last base bar in the bucket reaches the bucket end, or if there is any
    # base data after the bucket (early close / missing tail bars).
    data_end = bars["available_at"].iloc[-1]
    complete = (last_base_end >= htf_end) | (htf_end <= data_end)
    out = out.loc[complete.to_numpy()]
    # Available when the bucket is over — but never earlier than the last base bar in it.
    avail = pd.Series(out.index + tf.delta, index=out.index)
    out["available_at"] = avail.where(avail >= last_base_end.loc[out.index], last_base_end.loc[out.index])
    out.index.name = "time"
    out = out[BAR_COLS]
    out.attrs["timeframe"] = tf.name
    validate_bars(out)
    return out


def align_htf(base: pd.DataFrame, htf_frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Map an HTF-indexed frame (with ``available_at``) onto base bars, point-in-time."""
    return asof_join(base["available_at"], htf_frame, columns=columns)
