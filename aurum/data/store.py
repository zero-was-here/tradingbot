"""Parquet persistence for bars / macro frames and stable content hashing (SPEC §3.6).

Why not just ``df.to_parquet``?  Three properties matter for reproducible research:

1. **Timezones survive.** The index is written as a tz-aware UTC timestamp column and
   re-labelled ``time`` on load; ``available_at`` stays tz-aware UTC.
2. **Metadata survives.** ``df.attrs`` (timeframe, source, symbol, price scale...) is written
   to the parquet *schema metadata* under the key ``aurum`` as JSON, independent of whether
   the installed pandas version round-trips ``attrs`` itself.
3. **Provenance.** :func:`frame_hash` gives a content hash that depends only on column
   names, values and index — not on pandas version, datetime resolution (ns vs us) or
   memory layout — so a run can record exactly which data it saw (SPEC §0.4).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.data.schema import BAR_COLS, validate_bars

logger = logging.getLogger(__name__)

_META_KEY = b"aurum"
STORE_VERSION = 1


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion of attrs values to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_frame(df: pd.DataFrame, path: str | Path, *, metadata: dict | None = None) -> Path:
    """Write any DataFrame to parquet (zstd) with ``attrs`` + ``metadata`` in the schema.

    The write is atomic (temp file + rename) so a crashed job never leaves a truncated
    file that a later run would silently read.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # attrs travel in our own JSON payload below; hide them from pyarrow, which would
    # otherwise try (and, for Timestamps/arrays, fail with a UserWarning) to store them too.
    plain = df.copy(deep=False)
    plain.attrs = {}
    table = pa.Table.from_pandas(plain, preserve_index=True)
    meta = dict(table.schema.metadata or {})
    payload = {
        "store_version": STORE_VERSION,
        "attrs": _json_safe(dict(df.attrs)),
        "index_name": df.index.name,
        "extra": _json_safe(metadata or {}),
    }
    meta[_META_KEY] = json.dumps(payload, sort_keys=True).encode()
    table = table.replace_schema_metadata(meta)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_", suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def load_frame(path: str | Path) -> pd.DataFrame:
    """Read a parquet file written by :func:`save_frame`, restoring ``attrs``.

    tz-aware datetime columns/index come back tz-aware; anything stored under the ``aurum``
    metadata key is restored into ``df.attrs`` (``extra`` metadata under ``attrs["_meta"]``).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path))
    df = table.to_pandas()
    raw = (table.schema.metadata or {}).get(_META_KEY)
    attrs: dict = {}
    if raw:
        payload = json.loads(raw.decode())
        attrs = dict(payload.get("attrs") or {})
        if payload.get("extra"):
            attrs["_meta"] = payload["extra"]
        if payload.get("index_name") is not None:
            df.index.name = payload["index_name"]
    df.attrs = attrs
    return df


_UNIT_ORDER = ("s", "ms", "us", "ns")


def _harmonise_datetime_units(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast the (tz-aware) datetime index and ``columns`` to their finest common unit.

    Frames assembled by different code paths can mix resolutions (e.g. a ``datetime64[s]``
    index from a decoder with a ``datetime64[us]`` ``available_at``). Values are unchanged,
    but ``pd.merge_asof`` refuses keys of different units and ``.asi8`` integers become
    incomparable, so the store writes one unit per frame. Parquet cannot hold seconds, so
    the floor is milliseconds. Returns ``df`` itself when nothing needs casting.
    """
    targets: list[tuple[str | None, pd.DatetimeIndex]] = []
    if isinstance(df.index, pd.DatetimeIndex):
        targets.append((None, df.index))
    for c in columns:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c].dtype):
            targets.append((c, pd.DatetimeIndex(df[c])))
    if not targets:
        return df
    units = {t.unit for _, t in targets}
    unit = max(units | {"ms"}, key=_UNIT_ORDER.index)
    if units == {unit}:
        return df
    out = df.copy()
    for col, values in targets:
        if col is None:
            out.index = values.as_unit(unit)
        else:
            out[col] = values.as_unit(unit)
    out.attrs = dict(df.attrs)
    return out


def _canonical_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Column order ``BAR_COLS + extras``, index named ``time``, one datetime unit.

    :func:`load_bars` returns exactly this layout, so hashing it at save time is what makes
    the stored ``frame_hash`` reproducible on load. (Hashing the caller's frame as-is made a
    frame with, say, ``spread`` before ``volume`` or an unnamed index fail its own
    integrity check on reload.)
    """
    extras = [c for c in bars.columns if c not in BAR_COLS]
    out = bars[BAR_COLS + extras]
    out.attrs = dict(bars.attrs)
    if out.index.name != "time":
        out = out.rename_axis("time")
        out.attrs = dict(bars.attrs)
    return _harmonise_datetime_units(out, ["available_at"])


def save_bars(bars: pd.DataFrame, path: str | Path, *, metadata: dict | None = None) -> Path:
    """Validate and persist a canonical bars frame (see ``aurum.data.schema``).

    The frame is first put into the canonical layout that :func:`load_bars` returns
    (``BAR_COLS`` then extras, index named ``time``, index and ``available_at`` in one
    datetime unit); its ``frame_hash`` is stored in the metadata so :func:`load_bars` can
    detect corruption or silent edits.
    """
    validate_bars(bars)
    bars = _canonical_bars(bars)
    meta = dict(metadata or {})
    meta["frame_hash"] = frame_hash(bars)
    meta["n_rows"] = len(bars)
    if len(bars):
        meta["start"] = str(bars.index[0])
        meta["end"] = str(bars.index[-1])
    out = save_frame(bars, path, metadata=meta)
    logger.info("saved %d bars (%s) to %s", len(bars), bars.attrs.get("timeframe"), out)
    return out


def load_bars(path: str | Path, *, verify_hash: bool = True) -> pd.DataFrame:
    """Load bars saved by :func:`save_bars`; restores UTC index named ``time`` and attrs.

    Raises ``ValueError`` if ``verify_hash`` and the stored content hash does not match.
    """
    df = load_frame(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            raise ValueError(f"{path}: no DatetimeIndex / 'time' column")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "time"
    if "available_at" in df.columns:
        av = pd.DatetimeIndex(df["available_at"])
        df["available_at"] = av.tz_localize("UTC") if av.tz is None else av.tz_convert("UTC")
    extras = [c for c in df.columns if c not in BAR_COLS]
    attrs = dict(df.attrs)
    df = df[BAR_COLS + extras]
    df.attrs = attrs
    validate_bars(df)
    stored = (attrs.get("_meta") or {}).get("frame_hash")
    if verify_hash and stored is not None:
        actual = frame_hash(df)
        if actual != stored:
            raise ValueError(f"{path}: content hash mismatch (stored {stored[:12]}, actual {actual[:12]})")
    return df


# ------------------------------------------------------------------------------------
# hashing
# ------------------------------------------------------------------------------------
def _canonical_bytes(values: Any) -> tuple[str, bytes]:
    """Return (kind, bytes) for an array-like in a version/resolution-independent way."""
    if isinstance(values, (pd.DatetimeIndex, pd.Series)) and isinstance(
        getattr(values, "dtype", None), pd.DatetimeTZDtype
    ):
        idx = pd.DatetimeIndex(values)
        ns = idx.tz_convert("UTC").as_unit("ns").asi8.astype("<i8")
        return "datetime_utc_ns", ns.tobytes()
    arr = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        ns = pd.DatetimeIndex(arr).as_unit("ns").asi8.astype("<i8")
        return "datetime_naive_ns", ns.tobytes()
    if np.issubdtype(arr.dtype, np.timedelta64):
        return "timedelta_ns", pd.TimedeltaIndex(arr).as_unit("ns").asi8.astype("<i8").tobytes()
    if arr.dtype == bool:
        return "bool", arr.astype("u1").tobytes()
    if np.issubdtype(arr.dtype, np.number):
        f = arr.astype("<f8", copy=True)
        f[f == 0.0] = 0.0          # fold -0.0 into +0.0
        f[np.isnan(f)] = np.nan    # canonical NaN payload
        return "float64", f.tobytes()
    # object / string / categorical: stable text representation, NULL-separated
    parts = ["\x00<NA>" if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA) else str(v)
             for v in arr.tolist()]
    return "text", "\x1f".join(parts).encode("utf-8")


def frame_hash(df: pd.DataFrame) -> str:
    """Stable SHA-256 of a frame's column names, values and index.

    Independent of pandas version and datetime resolution (datetimes are hashed as UTC
    nanoseconds, numbers as little-endian float64 with -0.0/NaN canonicalised), but
    sensitive to column order, names, any value change and the index.
    """
    h = hashlib.sha256()
    h.update(f"shape={df.shape}".encode())
    kind, data = _canonical_bytes(df.index)
    h.update(f"index:{df.index.name}:{kind}:".encode())
    h.update(data)
    for col in df.columns:
        kind, data = _canonical_bytes(df[col])
        h.update(f"|col:{col}:{kind}:{len(data)}:".encode())
        h.update(data)
    return h.hexdigest()


__all__ = ["frame_hash", "load_bars", "load_frame", "save_bars", "save_frame"]
