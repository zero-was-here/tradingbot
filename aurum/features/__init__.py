"""Causal feature library (SPEC §4) and the train-only-fitted FeaturePipeline.

Convenience re-exports, resolved lazily on first attribute access (PEP 562) so that
``import aurum.features`` stays cheap, never imports optional heavy dependencies (torch,
anthropic) and cannot create import cycles with the submodules.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "FeatureSpec": "aurum.features.base",
    "register_feature": "aurum.features.base",
    "get_feature": "aurum.features.base",
    "list_features": "aurum.features.base",
    "FeaturePipeline": "aurum.features.pipeline",
    "FeatureSchemaError": "aurum.features.pipeline",
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
