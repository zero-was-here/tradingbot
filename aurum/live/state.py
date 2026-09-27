"""Durable local state for the live stack: atomic JSON files, JSONL logs, heartbeats.

Everything the live runner must remember across a crash or restart (OMS client ids, the
runner's last processed bar, the paper broker's book) is written with
:func:`atomic_write_json`: the payload goes to a temporary file in the same directory, is
``fsync``-ed, and then ``os.replace``-d over the target. ``os.replace`` is atomic on POSIX
and Windows, so a reader (or a restarted process) sees either the old or the new file,
never a torn one. Append-only logs (decisions, alerts) use JSON Lines, one record per line,
so a crash can at worst truncate the final line; :func:`read_jsonl` skips such a line.

A state file that exists but cannot be parsed raises :class:`StateCorruptError`. Callers
guarding money (the OMS) must treat that as fatal and refuse to trade until an operator
inspects the file — silently starting from an empty state could re-send orders.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import logging
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "StateCorruptError",
    "append_jsonl",
    "atomic_write_json",
    "read_json",
    "read_jsonl",
    "to_jsonable",
    "utc",
    "write_heartbeat",
]


class StateCorruptError(RuntimeError):
    """A persisted state file exists but cannot be decoded (fail-safe: do not trade)."""


def utc(ts: Any) -> pd.Timestamp:
    """Parse ``ts`` into a tz-aware UTC Timestamp (naive input is rejected: ambiguous)."""
    t = pd.Timestamp(ts)
    if t is pd.NaT:
        raise ValueError("timestamp is NaT")
    if t.tz is None:
        raise ValueError(f"timestamp {ts!r} must be tz-aware (UTC)")
    return t.tz_convert("UTC")


def to_jsonable(obj: Any, *, float_digits: int | None = 10) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable Python objects.

    Timestamps become ISO strings, numpy scalars Python scalars, non-finite floats ``None``
    (strict JSON has no NaN), dataclasses/enums/Series/DataFrames plain containers.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, float_digits=float_digits)
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return round(v, float_digits) if float_digits is not None else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return None if obj is pd.NaT else obj.isoformat()
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, _dt.timedelta)):
        return pd.Timedelta(obj).total_seconds()
    if obj is pd.NaT:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), float_digits=float_digits)
                for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, float_digits=float_digits) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v, float_digits=float_digits) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r, float_digits=float_digits) for r in obj.to_dict(orient="records")]
    if isinstance(obj, (list, tuple, set, frozenset, np.ndarray)):
        return [to_jsonable(v, float_digits=float_digits) for v in list(obj)]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def atomic_write_json(path: str | os.PathLike, data: Any, *, indent: int | None = 2,
                      convert: bool = True, fsync: bool = True) -> Path:
    """Write ``data`` as JSON to ``path`` atomically (tmp file + fsync + ``os.replace``).

    ``convert=False`` skips :func:`to_jsonable` for payloads that are already JSON-native
    (hot paths such as the OMS state).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    payload = json.dumps(to_jsonable(data) if convert else data, indent=indent, sort_keys=True,
                         allow_nan=False)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        if fsync:
            try:
                os.fsync(fh.fileno())
            except OSError:  # pragma: no cover - fsync unsupported on some filesystems
                pass
    os.replace(tmp, p)
    return p


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    """Read a JSON file; ``default`` if it does not exist; :class:`StateCorruptError` if unreadable."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StateCorruptError(f"state file {p} is unreadable: {exc}") from exc


def append_jsonl(path: str | os.PathLike, record: Mapping[str, Any], *, fsync: bool = False) -> None:
    """Append one JSON record (one line) to ``path``; parent directories are created."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_jsonable(record), sort_keys=False, allow_nan=False)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        if fsync:
            try:
                os.fsync(fh.fileno())
            except OSError:  # pragma: no cover
                pass


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """All records of a JSONL file (a torn final line from a crash is skipped with a warning)."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                logger.warning("skipping unreadable line %d of %s", n, p)


def write_heartbeat(path: str | os.PathLike, **fields: Any) -> Path:
    """Atomically write a heartbeat JSON (``time`` = now UTC unless given, ``pid``)."""
    payload = {"time": pd.Timestamp.now(tz="UTC"), "pid": os.getpid(), **fields}
    return atomic_write_json(path, payload)
