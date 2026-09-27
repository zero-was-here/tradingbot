"""Portfolio construction (SPEC §7): forecast combination and position sizing.

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.portfolio`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "ForecastCombiner": "aurum.portfolio.combiner",
    "VolTargetSizer": "aurum.portfolio.sizing",
    "FixedFractionalSizer": "aurum.portfolio.sizing",
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
