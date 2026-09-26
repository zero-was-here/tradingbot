"""Canonical bar-frame schema. Every module consumes and produces this shape.

A *bars frame* is a ``pd.DataFrame`` with:

  index    : ``DatetimeIndex`` named ``"time"``, tz-aware **UTC**, strictly increasing,
             holding each bar's OPEN time.
  open, high, low, close : float, **mid** prices (USD/oz).
  volume   : float >= 0 (tick volume if real volume is unknown; may be all zeros).
  spread   : float >= 0, typical bid/ask spread during the bar in **price units** (USD/oz),
             e.g. 0.25 means 25 cents. Fill prices are mid +/- spread/2.
  available_at : tz-aware UTC timestamp at which the bar is complete (open + timeframe).
             Anything derived from this bar may only be acted on at/after ``available_at``.

``df.attrs["timeframe"]`` holds the timeframe name (e.g. "H1") when known, but code must
not rely on attrs surviving pandas operations — use ``available_at`` for timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurum.core.timeframes import Timeframe, get_timeframe

PRICE_COLS = ["open", "high", "low", "close"]
BAR_COLS = PRICE_COLS + ["volume", "spread", "available_at"]


class SchemaError(ValueError):
    pass


def make_bars(
    df: pd.DataFrame,
    timeframe: str | Timeframe,
    *,
    default_spread: float | None = None,
) -> pd.DataFrame:
    """Coerce a frame with a UTC DatetimeIndex of bar-open times into the canonical schema.

    Adds ``available_at`` (open + timeframe), fills missing ``volume`` with 0 and missing
    ``spread`` with ``default_spread`` (required if the column is absent).
    """
    tf = get_timeframe(timeframe)
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise SchemaError("bars must have a DatetimeIndex of bar-open times")
    if out.index.tz is None:
        raise SchemaError("bars index must be tz-aware (UTC); localize/convert before make_bars")
    out.index = out.index.tz_convert("UTC")
    out.index.name = "time"
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "spread" not in out.columns:
        if default_spread is None:
            raise SchemaError("no 'spread' column: pass default_spread (price units)")
        out["spread"] = float(default_spread)
    out["available_at"] = out.index + tf.delta
    out = out[BAR_COLS + [c for c in out.columns if c not in BAR_COLS]]
    out.attrs["timeframe"] = tf.name
    validate_bars(out)
    return out


def validate_bars(df: pd.DataFrame, *, check_ohlc: bool = True) -> None:
    """Raise SchemaError if ``df`` violates the canonical schema."""
    missing = [c for c in BAR_COLS if c not in df.columns]
    if missing:
        raise SchemaError(f"bars missing columns: {missing}")
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise SchemaError("index must be a DatetimeIndex")
    if idx.tz is None or str(idx.tz) not in ("UTC", "utc"):
        raise SchemaError(f"index must be tz-aware UTC, got tz={idx.tz}")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise SchemaError("index must be strictly increasing without duplicates")
    if len(df) == 0:
        return
    px = df[PRICE_COLS].to_numpy(dtype=float)
    if not np.isfinite(px).all():
        raise SchemaError("non-finite OHLC values")
    if (px <= 0).any():
        raise SchemaError("non-positive prices")
    if check_ohlc:
        o, h, lo, c = px.T
        tol = 1e-9 * np.maximum(1.0, np.abs(h))
        if (h + tol < np.maximum.reduce([o, c, lo])).any():
            raise SchemaError("high below open/close/low")
        if (lo - tol > np.minimum.reduce([o, c, h])).any():
            raise SchemaError("low above open/close/high")
    if (df["spread"].to_numpy(dtype=float) < 0).any():
        raise SchemaError("negative spread")
    if (df["volume"].to_numpy(dtype=float) < 0).any():
        raise SchemaError("negative volume")
    avail = pd.DatetimeIndex(df["available_at"])
    if avail.tz is None:
        raise SchemaError("available_at must be tz-aware")
    if (avail <= idx).any():
        raise SchemaError("available_at must be strictly after bar open time")


def bars_timeframe(df: pd.DataFrame) -> Timeframe | None:
    tf = df.attrs.get("timeframe")
    return get_timeframe(tf) if tf else None
