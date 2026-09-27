"""Reinforcement learning on the shared execution stack (SPEC §2 ``rl/``).

* :mod:`aurum.rl.env` — :class:`GoldTradingEnv` (gymnasium) driving the SAME sizer, risk
  manager and :class:`~aurum.execution.simulator.ExecutionSimulator` as the backtest engine;
  :func:`rollout` for deterministic, position-aware policy evaluation.
* :mod:`aurum.rl.train` — :func:`train_ppo` (stable-baselines3 PPO, validation-selected
  checkpoints, artifact directories), :func:`load_artifact`, :func:`rollout_artifact`.

Attributes are resolved lazily, so ``import aurum.rl`` imports neither gymnasium nor torch;
the optional ``rl`` extra (``pip install aurum[rl]``) is only needed when they are used.
The episode-anchor helpers below are plain numpy/pandas (used by the ``rl_ppo`` strategy
without importing gymnasium).
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pandas as pd

_ENV_NAMES = ("DEFAULT_ACTION_LEVELS", "STATE_NAMES", "EnvConfig", "GoldTradingEnv",
              "RolloutResult", "rollout")
_TRAIN_NAMES = ("DEFAULT_FEATURE_GROUPS", "POLICY_FILES", "RLArtifact", "RLTrainConfig", "RLTrainResult",
                "backtest_forecast", "check_bar_size", "evaluate_policy", "load_artifact",
                "load_artifact_bytes", "prepare_data", "read_artifact_bytes", "rollout_artifact",
                "train_ppo")

#: Supported calendar anchors for simulated episodes (``rl_ppo``) and a CONSERVATIVE bound,
#: in calendar days, on the gap between two consecutive anchor bars (weekends/holidays
#: included): weekly (ISO week, Monday 00:00 UTC), monthly, quarterly.
EPISODE_ANCHOR_SPAN_DAYS: dict[str, int] = {"W": 10, "M": 34, "Q": 95}

__all__ = [*_ENV_NAMES, *_TRAIN_NAMES, "EPISODE_ANCHOR_SPAN_DAYS", "episode_anchor_mask",
           "episode_anchor_span_bars"]


def _check_anchor(anchor: str) -> None:
    if anchor not in EPISODE_ANCHOR_SPAN_DAYS:
        raise ValueError(f"episode_anchor must be one of {sorted(EPISODE_ANCHOR_SPAN_DAYS)} or "
                         f"None, got {anchor!r}")


def episode_anchor_mask(index: pd.DatetimeIndex, anchor: str | None) -> np.ndarray:
    """``True`` at the first bar (by UTC open time) of every calendar period after the first.

    Anchors depend on timestamps only (known in advance), never on prices, and the mask of
    a prefix of ``index`` is the prefix of the mask — point-in-time safe.
    """
    idx = pd.DatetimeIndex(index)
    out = np.zeros(len(idx), dtype=bool)
    if anchor is None or len(idx) < 2:
        return out
    _check_anchor(anchor)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    year = idx.year.to_numpy(dtype=np.int64)
    if anchor == "M":
        key = year * 12 + idx.month.to_numpy(dtype=np.int64)
    elif anchor == "Q":
        key = year * 4 + idx.quarter.to_numpy(dtype=np.int64)
    else:  # "W": ISO week starting Monday 00:00 UTC
        day = idx.normalize().tz_localize(None) if idx.tz is not None else idx.normalize()
        monday = day - pd.to_timedelta(idx.weekday.to_numpy(), unit="D")
        key = monday.asi8
    out[1:] = key[1:] != key[:-1]
    return out


def episode_anchor_span_bars(anchor: str | None, bar_minutes: float) -> int:
    """Conservative number of bars between two consecutive anchors (0 when ``None``)."""
    if anchor is None:
        return 0
    _check_anchor(anchor)
    if not (math.isfinite(bar_minutes) and bar_minutes > 0):
        raise ValueError(f"invalid bar size {bar_minutes!r} minutes")
    return int(math.ceil(EPISODE_ANCHOR_SPAN_DAYS[anchor] * 1440.0 / bar_minutes))


def __getattr__(name: str) -> Any:
    if name in _ENV_NAMES:
        return getattr(importlib.import_module("aurum.rl.env"), name)
    if name in _TRAIN_NAMES:
        return getattr(importlib.import_module("aurum.rl.train"), name)
    raise AttributeError(f"module 'aurum.rl' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
