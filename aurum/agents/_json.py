"""Deterministic JSON helpers shared by the desk (tool results, journal, snapshots).

Two properties matter here:

* **Determinism.** Tool results become part of the conversation prefix; serialising them
  with sorted keys and fixed separators keeps the bytes identical for identical data, which
  is what prompt caching keys on (a prefix match on the rendered request).
* **Safety.** Market snapshots contain numpy scalars, pandas timestamps and NaNs that the
  stdlib encoder rejects or renders as invalid JSON (``NaN``). Everything is converted to
  plain JSON types first; non-finite floats become ``null``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["to_jsonable", "dumps", "truncate"]


def _round_sig(x: float, sig: int) -> float:
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.{sig}g}")


def to_jsonable(obj: Any, *, float_sig: int | None = None) -> Any:
    """Recursively convert ``obj`` into JSON-native Python types.

    Missing values of every flavour (``None``, ``NaN``, ``pd.NA``, ``pd.NaT``, numpy ``NaT``)
    become ``null`` — never the strings ``"<NA>"`` / ``"NaT"``, which an agent could read as
    data. Durations become seconds. The function never raises: anything unknown is rendered
    with ``str`` (it is used on journal payloads and operator context, where a crash would
    abort a trading cycle).

    Parameters
    ----------
    obj       : any nesting of mappings, sequences, dataclasses, numpy/pandas scalars.
    float_sig : if given, round floats to this many significant digits (keeps LLM context
                small — six significant digits is far below any decision-relevant precision
                for a forecast or a vol estimate).
    """
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # numpy datetime64/timedelta64 must be handled before np.integer: np.timedelta64 IS an
    # np.signedinteger subclass, and int() on it raises.
    if isinstance(obj, np.datetime64):
        return None if np.isnat(obj) else _safe(lambda: pd.Timestamp(obj).isoformat(), obj)
    if isinstance(obj, np.timedelta64):
        return None if np.isnat(obj) else _safe(lambda: pd.Timedelta(obj).total_seconds(), obj)
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return _round_sig(x, float_sig) if float_sig else x
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, _dt.timedelta):  # includes pd.Timedelta
        return None if pd.isna(obj) else obj.total_seconds()
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, float_sig=float_sig)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name), float_sig=float_sig)
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, float_sig=float_sig) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v, float_sig=float_sig) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r, float_sig=float_sig) for r in obj.to_dict(orient="records")]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "mM":  # tolist() would yield datetime/int objects; keep NaT-aware path
            return [to_jsonable(v, float_sig=float_sig) for v in obj]
        return [to_jsonable(v, float_sig=float_sig) for v in obj.tolist()]
    if isinstance(obj, (pd.Index, pd.api.extensions.ExtensionArray)):
        return [to_jsonable(v, float_sig=float_sig) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v, float_sig=float_sig) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, float_sig=float_sig) for v in obj]
    if hasattr(obj, "model_dump"):  # pydantic (e.g. SDK objects)
        try:
            return to_jsonable(obj.model_dump(mode="json", by_alias=True, exclude_none=True), float_sig=float_sig)
        except Exception:  # pragma: no cover - defensive
            pass
    return _safe(lambda: str(obj), "<unrepresentable>")


def _safe(fn: Any, fallback: Any) -> Any:
    """``fn()``, or a string rendering of ``fallback`` if it raises (never propagate)."""
    try:
        return fn()
    except Exception:
        try:
            return str(fallback)
        except Exception:  # pragma: no cover - pathological __str__
            return "<unrepresentable>"


def dumps(obj: Any, *, float_sig: int | None = None) -> str:
    """Compact, key-sorted JSON (stable bytes for identical content)."""
    return json.dumps(
        to_jsonable(obj, float_sig=float_sig),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def truncate(text: str, max_chars: int | None) -> str:
    """Cut ``text`` to ``max_chars`` with an explicit, visible marker."""
    if max_chars is None or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}...[truncated {omitted} chars]"
