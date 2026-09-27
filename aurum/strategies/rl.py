"""``rl_ppo``: forecasts from a PPO policy trained in :mod:`aurum.rl` (SPEC §6).

Economic rationale
------------------
Rule-based strategies hard-code one hypothesis each (trend persistence, mean reversion,
macro sensitivity). A reinforcement-learning policy instead learns a direct mapping from
the causal feature state (multi-horizon returns, trend, volatility and regime descriptors,
session clock, higher-timeframe context) plus its own position to a target exposure,
maximising log wealth net of the SAME transaction costs, swaps, vol-targeted sizing and
risk limits used everywhere else (Moody & Saffell, 2001, "Learning to trade via direct
reinforcement"; Deng et al., 2017, "Deep direct reinforcement learning for financial signal
representation and trading"). Because it observes its own position and unrealised P&L it
can learn cost-aware holding behaviour that a stateless forecast cannot express.

The prior for a large edge is weak: financial rewards have a very low signal-to-noise
ratio, so PPO readily overfits the training period. Validation-based checkpoint selection
and the out-of-sample protocol of ``aurum.research`` are therefore essential; the artifact's
``metrics.json`` records how many validation evaluations were used for selection.

Forecast semantics
------------------
``generate`` rolls the deterministic policy forward bar by bar with its OWN simulated
account (ExecutionSimulator + the training sizer/costs/risk limits), because the policy's
observation includes its current position, unrealised P&L, time in trade and drawdown.
The forecast at ``t`` is the policy's action on the observation at the close of ``t``
(discrete levels, default ``{-1, -0.5, 0, 0.5, 1}``). The simulated account runs the
TRAINING sizer, costs and risk limits and mirrors how training episodes start and end:

* it starts flat at the first bar with a full observation window and, with the default
  ``episode_anchor="M"``, starts a FRESH episode (flat, initial equity) at the first bar of
  every month. A forecast therefore depends only on the bars since the last anchor (plus
  the feature warm-up), not on where the history window starts: a live runner fed a
  sliding window reproduces the backtest (a continuous account made 2-25% of 2020-21 H1
  forecasts depend on the window start). Forecasts before ``warmup_bars`` (feature warm-up
  + one anchor period) are 0.
* when a kill switch (or ruin) would end a training episode, it fires and flattens as in
  training, then the account starts a new episode in place (drawdown peak and risk state
  re-based, :meth:`aurum.rl.env.GoldTradingEnv.rebase_episode`). Forecasts continue after
  a simulated drawdown, yet the policy never observes a drawdown deeper than the ones that
  ended its training episodes (disabling the kills instead fed it >20% drawdowns — never
  seen in training — on 42% of 2012-2021 H1 bars).

The downstream risk manager is the one that may halt real trading. The simulated position
is the policy's own bookkeeping; a downstream backtest reproduces it only while it uses the
training sizer/risk settings from an anchor on and until the first simulated kill switch.

``features`` passed to ``fit``/``generate`` are ignored: the policy must see exactly the
feature columns and scaling it was trained with, so the artifact's own
:class:`~aurum.features.pipeline.FeaturePipeline` recomputes them from ``md`` (causal).
``md`` must have the bar size the policy was trained on (``ValueError`` otherwise).

Artifact integrity: every ``fit`` writes a NEW directory (a unique subdirectory of
``out_dir``), so clones fitted on different walk-forward folds can never overwrite each
other's policy. The policy network is reloaded lazily (clones and pickles stay light), and
the reload refuses to run if the artifact's files changed since it was attached (SHA-256
fingerprint) — a strategy never silently trades a different policy, e.g. one trained on a
later fold that includes its own test period.

Importing this module does NOT import torch, stable-baselines3 or gymnasium; they are
loaded inside ``fit``/``generate`` (optional ``rl`` extra).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from aurum.core.types import MarketData
from aurum.rl import episode_anchor_span_bars
from aurum.strategies.base import Strategy, register_strategy

logger = logging.getLogger(__name__)

__all__ = ["RLPolicyStrategy"]

#: Files whose content defines a policy's behaviour (network, feature scaling, env config).
_FINGERPRINT_FILES = ("policy.zip", "pipeline.json", "config.json")


def _artifact_fingerprint(path: Path) -> str:
    """SHA-256 over the files that define the policy (detects in-place overwrites)."""
    h = hashlib.sha256()
    for name in _FINGERPRINT_FILES:
        h.update(name.encode("utf-8") + b"\0")
        with open(path / name, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


@register_strategy
class RLPolicyStrategy(Strategy):
    """Adapter from a trained :mod:`aurum.rl` PPO artifact to the Strategy interface.

    Parameters (``params``)
    -----------------------
    artifact_dir : load a trained artifact (``policy.zip``, ``pipeline.json``, ``config.json``)
                   instead of training.
    config       : :class:`aurum.rl.train.RLTrainConfig` or its dict form, used by ``fit``.
    val_fraction : trailing fraction of the ``fit`` data held out (time-ordered) for
                   validation-based checkpoint selection / early stopping.
    min_val_bars : minimum validation bars; ``fit`` raises if the split leaves fewer.
    out_dir      : parent directory under which every ``fit`` creates its OWN new artifact
                   subdirectory (``rl_ppo_<last fit bar>_<random>``); default: a new
                   temporary directory. Reusing one ``out_dir`` across folds is safe.
    device       : torch device for inference (``"cpu"`` default for determinism).
    episode_anchor : ``"auto"`` (default: the artifact's ``config.env.episode_anchor``, i.e.
                   the convention its checkpoint was validated with — monthly by default),
                   ``"W"``, ``"M"``, ``"Q"`` or ``None``. The simulated account starts a fresh
                   training-style episode (flat, initial equity) at the first bar of every
                   calendar period, so a forecast depends only on the bars since the last
                   anchor (plus the feature warm-up) — NOT on where the history window
                   starts: a live runner fed a sliding window reproduces the backtest.
                   ``None`` = one continuous simulated account (path-dependent). Overriding
                   the validated convention changes the policy's behaviour (see module doc).
    """

    name = "rl_ppo"
    description = ("PPO policy (stable-baselines3) trained on the shared simulator with "
                   "vol-targeted sizing and full costs; forecast = deterministic action, "
                   "rolled forward with its own simulated position.")
    trainable = True

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"artifact_dir": None, "config": None, "val_fraction": 0.25,
                "min_val_bars": 500, "out_dir": None, "device": "cpu", "episode_anchor": "auto"}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._artifact: Any = None          # aurum.rl.train.RLArtifact (lazy, holds torch)
        self._pipeline: Any = None          # FeaturePipeline (JSON, cheap)
        self._config: Any = None            # RLTrainConfig
        self.artifact_dir_: Path | None = None
        self.artifact_fingerprint_: str | None = None
        self.train_summary_: dict[str, Any] = {}
        if self.params.get("artifact_dir"):
            self.load(self.params["artifact_dir"])

    # ---- persistence ---------------------------------------------------------------------
    def load(self, artifact_dir: str | Path) -> RLPolicyStrategy:
        """Attach a trained artifact. Reads the JSON parts now; the policy network is loaded
        lazily on the first :meth:`generate` (keeps construction torch-free)."""
        from aurum.features.pipeline import FeaturePipeline
        from aurum.rl.train import RLTrainConfig

        p = Path(artifact_dir)
        missing = [f for f in _FINGERPRINT_FILES if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(f"RL artifact {p} is missing {missing}")
        fingerprint = _artifact_fingerprint(p)
        self._config = RLTrainConfig.from_dict(json.loads((p / "config.json").read_text("utf-8")))
        self._pipeline = FeaturePipeline.load(p / "pipeline.json")
        self._artifact = None
        self.artifact_dir_ = p
        self.artifact_fingerprint_ = fingerprint
        self.params["artifact_dir"] = str(p)
        mpath = p / "metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text("utf-8"))
            self.train_summary_ = {k: m.get(k) for k in (
                "val", "selected_timesteps", "n_evals", "timesteps_trained", "early_stopped")}
        self.is_fitted = True
        return self

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, **params: Any) -> RLPolicyStrategy:
        return cls(artifact_dir=str(artifact_dir), **params)

    def _get_artifact(self) -> Any:
        if self._artifact is None:
            if self.artifact_dir_ is None:
                raise RuntimeError("rl_ppo is not fitted: call fit() or load(artifact_dir)")
            expected = getattr(self, "artifact_fingerprint_", None)
            if expected is not None and _artifact_fingerprint(self.artifact_dir_) != expected:
                raise RuntimeError(
                    f"RL artifact {self.artifact_dir_} changed on disk since it was attached "
                    f"(fingerprint {expected[:12]}...); refusing to run a different policy. "
                    "Call load(artifact_dir) explicitly to attach the new files.")
            from aurum.rl.train import load_artifact

            self._artifact = load_artifact(self.artifact_dir_, device=self.params["device"])
        return self._artifact

    def __getstate__(self) -> dict[str, Any]:
        # The torch policy is reloaded from ``artifact_dir_`` on demand, so clones/pickles
        # stay light and never share mutable network state.
        state = self.__dict__.copy()
        state["_artifact"] = None
        return state

    # ---- Strategy interface ----------------------------------------------------------------
    @property
    def _policy_warmup(self) -> int:
        """First bar with a full observation window (where the simulated account starts)."""
        if self._pipeline is None or self._config is None:
            return 0
        return int(self._pipeline.max_lookback + self._config.env.window - 1)

    @property
    def episode_anchor(self) -> str | None:
        """The resolved episode anchor (``"auto"`` -> the artifact's validated convention)."""
        anchor = self.params.get("episode_anchor", "auto")
        if anchor == "auto":
            return None if self._config is None else self._config.env.episode_anchor
        return anchor

    @property
    def warmup_bars(self) -> int:
        """Feature warm-up plus, with an ``episode_anchor``, one (conservative) anchor period:
        from here on every forecast lies after an anchor that every history window of at
        least this length contains, so the forecast does not depend on the window start."""
        warm = self._policy_warmup
        if warm == 0 or self._pipeline is None:
            return warm
        minutes = self._pipeline.bar_minutes
        span = episode_anchor_span_bars(self.episode_anchor, float(minutes) if minutes else 60.0)
        return warm + span

    def _train_config(self) -> Any:
        from aurum.rl.train import RLTrainConfig

        cfg = self.params.get("config")
        if cfg is None:
            return RLTrainConfig()
        if isinstance(cfg, RLTrainConfig):
            return cfg
        return RLTrainConfig.from_dict(cfg)

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> RLPolicyStrategy:
        """Train PPO on ``md`` with a trailing time-ordered validation split."""
        from aurum.rl.train import train_ppo

        cfg = self._train_config()
        n = len(md.bars)
        frac = float(self.params["val_fraction"])
        if not 0.0 < frac < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        n_val = int(round(n * frac))
        if n_val < int(self.params["min_val_bars"]) or n - n_val < 2:
            raise ValueError(f"rl_ppo.fit: {n} bars leave {n_val} validation bars "
                             f"(< min_val_bars={self.params['min_val_bars']})")
        split = n - n_val
        md_tr = MarketData(bars=md.bars.iloc[:split], macro=md.macro, events=md.events)
        md_va = MarketData(bars=md.bars.iloc[split:], macro=md.macro, events=md.events)
        # A NEW directory per fit: clones fitted on different folds must never share (and
        # overwrite) one artifact, or a lazily reloaded policy could be a later fold's.
        prefix = f"rl_ppo_{pd.Timestamp(md.bars.index[-1]):%Y%m%dT%H%M}_"
        parent = self.params.get("out_dir")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
            out = tempfile.mkdtemp(prefix=prefix, dir=str(parent))
        else:
            out = tempfile.mkdtemp(prefix=prefix)
        logger.info("rl_ppo.fit: %d train / %d validation bars -> %s", split, n_val, out)
        res = train_ppo(md_tr, md_va, config=cfg, out_dir=out)
        self.load(res.artifact_dir)
        return self

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        """Deterministic policy forecasts in the policy's discrete levels (or [-1, 1])."""
        if not self.is_fitted or self.artifact_dir_ is None:
            raise RuntimeError("rl_ppo is not fitted: call fit() or load(artifact_dir)")
        from aurum.rl.train import check_bar_size, rollout_artifact

        check_bar_size(self._pipeline, md.bars)
        index = md.bars.index
        warm = self.warmup_bars
        if len(index) < warm + 2:
            logger.info("rl_ppo.generate: %d bars do not cover the %d-bar warm-up; flat",
                        len(index), warm)
            return self._finalize(pd.Series(0.0, index=index), index)
        ro = rollout_artifact(self._get_artifact(), md, start=self._policy_warmup,
                              kill_switches=False,
                              episode_anchor=self.episode_anchor)
        return self._finalize(ro.forecast, index)  # zero before warmup_bars
