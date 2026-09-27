"""Alpha models (SPEC §6). Concrete strategies register themselves on import; use
``list_strategies()`` / ``get_strategy(name, **params)`` (``rl_ppo`` never imports torch at
registration).

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.strategies`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "Strategy": "aurum.strategies.base",
    "register_strategy": "aurum.strategies.base",
    "get_strategy": "aurum.strategies.base",
    "list_strategies": "aurum.strategies.base",
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
