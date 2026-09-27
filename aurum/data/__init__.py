"""Data layer (SPEC §3): canonical bars, point-in-time joins, loaders, Dukascopy/macro
downloads, calendar, synthetic data and the parquet store.

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.data`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "MarketData": "aurum.core.types",
    "BAR_COLS": "aurum.data.schema",
    "SchemaError": "aurum.data.schema",
    "make_bars": "aurum.data.schema",
    "validate_bars": "aurum.data.schema",
    "bars_timeframe": "aurum.data.schema",
    "asof_join": "aurum.data.pit",
    "resample_bars": "aurum.data.resample",
    "load_bars": "aurum.data.store",
    "save_bars": "aurum.data.store",
    "load_frame": "aurum.data.store",
    "save_frame": "aurum.data.store",
    "frame_hash": "aurum.data.store",
    "load_csv": "aurum.data.loaders",
    "load_mt5_csv": "aurum.data.loaders",
    "bars_from_ohlc": "aurum.data.loaders",
    "quality_report": "aurum.data.loaders",
    "download_dukascopy": "aurum.data.dukascopy",
    "build_dataset": "aurum.data.dukascopy",
    "prefetch_cache": "aurum.data.dukascopy",
    "DEFAULT_YAHOO": "aurum.data.macro",
    "DEFAULT_FRED": "aurum.data.macro",
    "fetch_yahoo_daily": "aurum.data.macro",
    "fetch_fred": "aurum.data.macro",
    "load_macro_dir": "aurum.data.macro",
    "save_macro_dir": "aurum.data.macro",
    "generate_rule_based_calendar": "aurum.data.calendar",
    "load_calendar_csv": "aurum.data.calendar",
    "merge_calendars": "aurum.data.calendar",
    "fomc_statements": "aurum.data.calendar",
    "make_synthetic_bars": "aurum.data.synthetic",
    "make_synthetic_macro": "aurum.data.synthetic",
    "make_synthetic_events": "aurum.data.synthetic",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
