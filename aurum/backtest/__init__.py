"""Backtest engine, result container and metrics (SPEC §8).

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.backtest`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "run_backtest": "aurum.backtest.engine",
    "run_target_lots": "aurum.backtest.engine",
    "buy_and_hold_benchmark": "aurum.backtest.engine",
    "average_true_range": "aurum.backtest.engine",
    "ForecastHook": "aurum.backtest.engine",
    "BacktestResult": "aurum.backtest.result",
    "compute_metrics": "aurum.backtest.metrics",
    "daily_returns": "aurum.backtest.metrics",
    "drawdown_series": "aurum.backtest.metrics",
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
