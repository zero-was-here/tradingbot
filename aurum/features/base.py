"""Feature registry.

A *feature function* maps ``MarketData`` to a DataFrame indexed exactly like
``md.bars.index``. Row ``t`` may depend ONLY on information available at
``md.bars["available_at"].iloc[t]`` (i.e. bars[:t+1], plus macro/event rows whose
``available_at`` <= that time). This is enforced by ``tests/test_features_leakage.py``
which perturbs the future and checks that the past does not change.

Rules for implementers:
  * no ``shift(-k)``, no centred windows, no ``bfill``, no full-sample statistics
    (mean/std/quantiles over the whole column). Rolling/expanding/EWM only.
  * Warm-up rows are NaN (the pipeline decides how to handle them).
  * Output columns are prefixed with the feature family, e.g. ``"trend_ema_slope_20"``.
  * Prefer scale-free outputs (returns, ratios, z-scores, ATR-normalised distances) so
    models transfer across price levels ($300 gold in 2001 vs $2,500 in 2024).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from aurum.core.types import MarketData

FeatureFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class FeatureSpec:
    name: str                       # unique registry key, e.g. "trend"
    family: str                     # "trend" | "momentum" | "volatility" | "microstructure" | ...
    fn: FeatureFn
    lookback: int                   # bars of history needed before outputs are valid
    requires_macro: tuple[str, ...] = ()
    requires_events: bool = False
    params: dict = field(default_factory=dict, compare=False, hash=False)
    doc: str = ""

    def compute(self, md: MarketData, **overrides) -> pd.DataFrame:
        kwargs = {**self.params, **overrides}
        out = self.fn(md, **kwargs)
        if not out.index.equals(md.bars.index):
            raise ValueError(f"feature {self.name!r} returned a misaligned index")
        return out


_REGISTRY: dict[str, FeatureSpec] = {}


def register_feature(
    name: str,
    *,
    family: str,
    lookback: int,
    requires_macro: tuple[str, ...] = (),
    requires_events: bool = False,
    **params,
) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator: ``@register_feature("trend", family="trend", lookback=200)``."""

    def deco(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise KeyError(f"feature {name!r} already registered")
        _REGISTRY[name] = FeatureSpec(
            name=name,
            family=family,
            fn=fn,
            lookback=lookback,
            requires_macro=tuple(requires_macro),
            requires_events=requires_events,
            params=params,
            doc=(fn.__doc__ or "").strip(),
        )
        return fn

    return deco


def unregister_feature(name: str) -> None:
    """Remove a feature from the registry (used by tests for temporary features)."""
    _REGISTRY.pop(name, None)


def get_feature(name: str) -> FeatureSpec:
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown feature {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_features() -> list[FeatureSpec]:
    _ensure_loaded()
    return list(_REGISTRY.values())


def _ensure_loaded() -> None:
    """Import feature modules so their decorators run (idempotent)."""
    import importlib

    for mod in (
        "aurum.features.technical",
        "aurum.features.volatility",
        "aurum.features.microstructure",
        "aurum.features.multi_timeframe",
        "aurum.features.macro",
        "aurum.features.calendar",
        "aurum.features.regime",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:  # module not implemented yet
            if exc.name != mod:
                raise
