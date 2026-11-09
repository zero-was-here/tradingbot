"""Research tooling (SPEC §9): leak-free splits, overfitting statistics, tearsheets and
the walk-forward protocol.

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.research`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "walk_forward_splits": "aurum.research.splits",
    "purged_kfold": "aurum.research.splits",
    "cpcv_splits": "aurum.research.splits",
    "sharpe": "aurum.research.stats",
    "probabilistic_sharpe": "aurum.research.stats",
    "deflated_sharpe": "aurum.research.stats",
    "min_track_record_length": "aurum.research.stats",
    "pbo_cscv": "aurum.research.stats",
    "stationary_bootstrap": "aurum.research.stats",
    "sharpe_ci": "aurum.research.stats",
    "sharpe_summary": "aurum.research.stats",
    "write_tearsheet": "aurum.research.report",
    "run_walk_forward": "aurum.research.walkforward",
    "run_single_backtest": "aurum.research.walkforward",
    "WalkForwardReport": "aurum.research.walkforward",
    "HoldoutReport": "aurum.research.walkforward",
    "provenance": "aurum.research.walkforward",
    "plan_folds": "aurum.research.walkforward",
    "fit_quant_book": "aurum.research.walkforward",
    "QuantBook": "aurum.research.walkforward",
    "book_statistics": "aurum.research.walkforward",
    "load_summary": "aurum.research.walkforward",
    "record_holdout_look": "aurum.research.walkforward",
    "prior_holdout_looks": "aurum.research.walkforward",
    "forecast_contributions": "aurum.research.walkforward",
    # v2.1 analytics: parameter sweeps, Monte Carlo / stress, explainability
    "run_sweep": "aurum.research.sweep",
    "plateau_score": "aurum.research.sweep",
    "SweepResult": "aurum.research.sweep",
    "monte_carlo": "aurum.research.montecarlo",
    "stress_run": "aurum.research.montecarlo",
    "MonteCarloResult": "aurum.research.montecarlo",
    "StressReport": "aurum.research.montecarlo",
    "explain_decision": "aurum.research.attribution",
    "explain_live_decision": "aurum.research.attribution",
    "live_decisions": "aurum.research.attribution",
    "pnl_attribution": "aurum.research.attribution",
    "desk_journal_summary": "aurum.research.attribution",
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
