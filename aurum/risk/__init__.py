"""Risk management (SPEC §7): the reduce-only pre-trade risk manager and VaR/stress tools.

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.risk`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "RiskLimits": "aurum.risk.manager",
    "StandardRiskManager": "aurum.risk.manager",
    "var_es": "aurum.risk.var",
    "risk_report": "aurum.risk.var",
    "stress_test": "aurum.risk.var",
    "gap_shock": "aurum.risk.var",
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
