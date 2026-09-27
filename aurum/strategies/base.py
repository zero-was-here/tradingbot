"""Alpha-model interface.

A Strategy turns point-in-time market data into a *forecast* series:

    forecast[t] in [-1, +1]  — desired directional exposure decided at the CLOSE of bar t
                               (i.e. at ``bars.available_at[t]``). +1 = max long conviction,
                               -1 = max short, 0 = flat. It is executed by the simulator at
                               the OPEN of bar t+1. Magnitude is conviction, NOT lots —
                               volatility targeting / sizing happens in ``aurum.portfolio``.

Contract:
  * ``generate`` must be causal: forecast[t] depends only on data available at close of t.
    ``tests/test_strategies_leakage.py`` perturbs the future to check this.
  * Rows before ``warmup_bars`` are 0.0 (not NaN). Output has no NaN/inf.
  * ``fit`` receives ONLY training data (the walk-forward engine guarantees it); anything
    learned (thresholds, model weights, scalers) must be stored on ``self``.
  * Strategies are cheap to deep-copy; the walk-forward engine clones one per fold.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from aurum.core.types import MarketData


class Strategy(ABC):
    #: registry key, e.g. "tsmom"
    name: ClassVar[str] = "base"
    #: short human description shown in reports and to the LLM agents
    description: ClassVar[str] = ""
    #: whether ``fit`` learns anything from data (affects walk-forward refits)
    trainable: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = {**self.default_params(), **params}
        self.is_fitted: bool = not self.trainable

    # ---- to override -------------------------------------------------------------------
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @property
    def warmup_bars(self) -> int:
        """Bars of history required before the forecast is meaningful."""
        return 0

    @property
    def fit_history_bars(self) -> int:
        """Bars of history BEFORE the training window that ``fit`` may receive purely for
        feature warm-up (the walk-forward engine prepends them to the training slice).

        Those bars are strictly in the past relative to every test row, so using them is not
        leakage; strategies whose own feature pipeline needs a long warm-up (e.g. the ML
        models' one-year regime rank) override this so a rolling 3-year training window is
        not partly consumed by warm-up. Default 0.
        """
        return 0

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> Strategy:
        """Learn parameters from TRAINING data only. Default: nothing to learn."""
        self.is_fitted = True
        return self

    @abstractmethod
    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        """Return forecast in [-1, 1] indexed like ``md.bars.index`` (see module doc)."""

    # ---- helpers -----------------------------------------------------------------------
    def clone(self) -> Strategy:
        return copy.deepcopy(self)

    def _finalize(self, forecast: pd.Series, index: pd.Index) -> pd.Series:
        """Align, clip to [-1, 1], zero the warm-up and any NaN/inf. Call at end of generate()."""
        f = pd.Series(forecast, index=index, dtype=float).reindex(index)
        f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        if self.warmup_bars > 0:
            f.iloc[: self.warmup_bars] = 0.0
        f.name = self.name
        return f

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


_STRATEGIES: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in _STRATEGIES and _STRATEGIES[cls.name] is not cls:
        raise KeyError(f"strategy {cls.name!r} already registered")
    _STRATEGIES[cls.name] = cls
    return cls


def get_strategy(name: str, **params: Any) -> Strategy:
    _ensure_loaded()
    if name not in _STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name](**params)


def list_strategies() -> dict[str, type[Strategy]]:
    _ensure_loaded()
    return dict(_STRATEGIES)


def _ensure_loaded() -> None:
    import importlib

    for mod in (
        "aurum.strategies.trend",
        "aurum.strategies.mean_reversion",
        "aurum.strategies.breakout",
        "aurum.strategies.macro",
        "aurum.strategies.seasonal",
        "aurum.strategies.ml",
        "aurum.strategies.rl",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            if exc.name != mod:
                raise
