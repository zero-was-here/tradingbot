"""Leak-free resampling between timeframes.

Higher-timeframe (HTF) bars are labelled by their OPEN time (like MT5) and carry an
``available_at`` equal to the HTF bar's END. The legacy code labelled H1 bars at 00:00 and
forward-filled them onto M5 bars from 00:00, leaking 55 minutes of future. Here, an HTF bar
is only visible to base bars whose own ``available_at`` >= the HTF bar's ``available_at``.

An HTF bar is emitted only if it is *complete* with respect to the base data: the last
base bar that falls inside it must end exactly at the HTF bar's end, OR a later base bar
exists (market closed early, e.g. Friday). The trailing in-progress HTF bar is dropped
(unless the caller asserts the data is final via ``complete_until``).

Bucketing uses fixed-length rules (``Timedelta``) anchored at UTC midnight (+ the optional
daily anchor for H4/D1): pandas 3 treats ``"1D"`` as a calendar-day offset and silently
ignores ``offset=`` for it, which would lose the D1 anchor.
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
    complete_until: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Aggregate canonical bars to a higher timeframe.

    ``daily_anchor_hour_utc`` shifts D1/H4 bucket boundaries (e.g. 21 or 22 to align days to
    the New York 17:00 close used by most brokers). Default 0 = UTC midnight.

    ``complete_until`` (optional, UTC): the base data is known to be final up to this
    instant (e.g. a historical download of whole days). Buckets ending at or before it are
    complete even if their last base bar ends early (market closed), so a dataset ending on
    a Friday keeps its Friday D1/H4 bars. It never makes a bar *available* earlier: an HTF
    bar's ``available_at`` is still its bucket end. Leave ``None`` for live data.
    """
    validate_bars(bars)
    tf = get_timeframe(to)
    if not 0 <= int(daily_anchor_hour_utc) < 24:
        raise ValueError("daily_anchor_hour_utc must be in [0, 24)")
    if len(bars) == 0:
        out = bars[BAR_COLS].copy()
        out.attrs["timeframe"] = tf.name
        return out
    # Refuse to "resample" into a FINER timeframe: every base bar would land alone in a
    # bucket shorter than itself and come back labelled e.g. "M15" while still spanning an
    # hour (available_at = open + 1h), silently corrupting any timeframe-based logic.
    base_span = (pd.DatetimeIndex(bars["available_at"]) - bars.index).min()
    if tf.delta < base_span:
        raise ValueError(
            f"cannot resample bars spanning {base_span} to the finer timeframe {tf.name} ({tf.delta})"
        )
    offset = pd.Timedelta(hours=daily_anchor_hour_utc) if tf.minutes >= 240 else pd.Timedelta(0)
    # A fixed-length (Tick) rule: in pandas 3 "1D" is a calendar-day offset for which
    # resample() silently IGNORES ``offset`` — the D1 anchor would be lost. In UTC a day is
    # always 24h, so the fixed rule is exact. Buckets are aligned to UTC midnight + offset.
    grouped = bars.resample(tf.delta, label="left", closed="left", offset=offset, origin="start_day")
    out = grouped.agg(_AGG)
    last_base_end = grouped["available_at"].max()
    out = out.dropna(subset=["open", "high", "low", "close"])
    last_base_end = last_base_end.reindex(out.index)

    htf_end = out.index + tf.delta
    # Complete if the last base bar in the bucket reaches the bucket end, or if there is any
    # base data after the bucket (early close / missing tail bars).
    data_end = bars["available_at"].max()
    if complete_until is not None:
        cu = pd.Timestamp(complete_until)
        cu = cu.tz_localize("UTC") if cu.tz is None else cu.tz_convert("UTC")
        data_end = max(data_end, cu)
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
