"""PPO training on :class:`~aurum.rl.env.GoldTradingEnv` with honest validation (SPEC §0, §9).

Pipeline
--------
1. ``md_train`` and ``md_val`` (``md_val`` strictly after ``md_train``) are concatenated
   only to give the validation slice its feature/vol warm-up. Feature groups are computed
   on the combined history (every group is causal row by row), the
   :class:`~aurum.features.pipeline.FeaturePipeline` scaler is fitted on TRAINING rows only
   and applied to both.
2. ``n_envs`` training environments (``DummyVecEnv``) sample random fixed-length episodes
   over the WHOLE training range. Observations are already scaled by the pipeline, so
   ``VecNormalize`` (optional) normalises rewards only.
3. PPO (Schulman et al., 2017, "Proximal Policy Optimization Algorithms") with an
   ``MlpPolicy``. Every ``eval_freq`` timesteps a callback runs the DETERMINISTIC policy
   bar by bar with its own simulated position (:func:`aurum.rl.env.rollout`) under the
   DEPLOYMENT convention of the ``rl_ppo`` strategy (fresh episodes at
   ``EnvConfig.episode_anchor``, kill switches restart the account), so the forecasts it
   scores on the validation slice are exactly those the deployed strategy emits there
   (:func:`evaluate_policy`). The forecast series is backtested with the real engine
   (:func:`aurum.backtest.engine.run_backtest`: same sizer, risk manager, costs and vol
   from the first validation bar) and scored with
   :func:`aurum.backtest.metrics.compute_metrics`; the engine is also checked against the
   simulated account (they must agree to the cent).
4. The checkpoint with the best validation daily Sharpe is kept; training stops early after
   ``patience`` evaluations without improvement (or at ``max_wall_time_s``).
5. An artifact directory is written: ``policy.zip`` (SB3), ``pipeline.json``,
   ``config.json``, ``metrics.json`` (validation metrics, history, provenance),
   ``data_hash.json`` (SHA-256 of the train/val bars, SPEC §0.4) and ``history.csv``.

Caveat — selection bias. Choosing the best of ``K`` validation evaluations inflates the
validation Sharpe (Bailey & Lopez de Prado, 2014, "The Deflated Sharpe Ratio"). The
reported ``val`` metrics are therefore optimistic; ``metrics.json`` records ``n_evals`` so
callers can deflate them (``aurum.research.stats.sharpe_summary(..., n_trials=n_evals)``),
and a clean out-of-sample test needs data after the validation slice.

Determinism: CPU by default, seeded PPO / environments; ``device="auto"`` picks CUDA, then
Apple MPS, then CPU (GPU kernels may not be bit-reproducible).

``torch`` and ``stable_baselines3`` are imported lazily inside the functions that need them,
so importing this module (e.g. for :class:`RLTrainConfig`) stays cheap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.engine import run_backtest
from aurum.backtest.result import BacktestResult
from aurum.core.types import MarketData
from aurum.data.schema import validate_bars
from aurum.data.store import frame_hash
from aurum.execution.costs import CostModel
from aurum.features.pipeline import FeaturePipeline
from aurum.models.volatility import ewma_volatility
from aurum.portfolio.sizing import VolTargetSizer
from aurum.risk.manager import RiskLimits, StandardRiskManager
from aurum.rl import episode_anchor_mask, episode_anchor_span_bars
from aurum.rl.env import (
    EnvConfig,
    GoldTradingEnv,
    RolloutResult,
    rollout,
    simulated_risk_logging,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_FILES",
    "DEFAULT_FEATURE_GROUPS",
    "POLICY_FILES",
    "EvalOutcome",
    "PreparedData",
    "RLArtifact",
    "RLTrainConfig",
    "RLTrainResult",
    "backtest_forecast",
    "check_bar_size",
    "evaluate_policy",
    "load_artifact",
    "load_artifact_bytes",
    "make_predictor",
    "prepare_data",
    "read_artifact_bytes",
    "resolve_device",
    "rollout_artifact",
    "train_ppo",
]

#: Compact default feature set: price/vol state at several horizons plus session clock and
#: higher-timeframe context (no macro/calendar, so it runs on bars alone).
DEFAULT_FEATURE_GROUPS: tuple[str, ...] = (
    "returns", "trend", "momentum", "volatility", "regime", "session", "mtf",
)
ARTIFACT_FILES = ("policy.zip", "pipeline.json", "config.json", "metrics.json", "data_hash.json")
_METRIC_KEYS = (
    "sharpe", "sortino", "calmar", "total_return", "cagr", "ann_vol", "max_drawdown",
    "n_trades", "trades_per_year", "win_rate", "profit_factor", "exposure",
    "turnover_lots_per_year", "total_costs", "cost_drag_ann", "swap_total", "n_days",
)


# -----------------------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------------------
@dataclass
class RLTrainConfig:
    """Everything that defines a PPO training run (JSON round-trippable).

    Features: ``feature_groups`` / ``feature_overrides`` / ``scaler`` / ``feature_clip`` are
    passed to :class:`FeaturePipeline`. ``env`` configures the environment (window, actions,
    reward, sizer, risk limits, costs). PPO hyper-parameters follow SB3 names.
    ``eval_freq`` is in environment timesteps (summed over ``n_envs``); ``patience`` is the
    number of validation evaluations without a Sharpe improvement of more than
    ``min_delta`` before stopping (``None`` disables early stopping).
    """

    feature_groups: tuple[str, ...] | None = DEFAULT_FEATURE_GROUPS
    feature_overrides: dict[str, dict] = field(default_factory=dict)
    scaler: str = "robust"
    feature_clip: float = 5.0
    env: EnvConfig = field(default_factory=EnvConfig)
    total_timesteps: int = 200_000
    n_envs: int = 4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    normalize_reward: bool = True
    eval_freq: int = 20_000
    patience: int | None = 5
    min_delta: float = 0.0
    select_metric: str = "sharpe"
    max_wall_time_s: float | None = None
    seed: int = 0
    device: str = "cpu"
    torch_threads: int | None = None
    verbose: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.env, Mapping):
            self.env = EnvConfig.from_dict(self.env)
        if self.feature_groups is not None:
            self.feature_groups = tuple(self.feature_groups)
        self.net_arch = tuple(int(x) for x in self.net_arch)
        if self.total_timesteps < 1 or self.n_envs < 1 or self.n_steps < 2:
            raise ValueError("total_timesteps, n_envs must be >= 1 and n_steps >= 2")
        if self.batch_size < 2 or self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("batch_size must be in [2, n_steps * n_envs]")
        if self.activation not in ("tanh", "relu"):
            raise ValueError("activation must be 'tanh' or 'relu'")
        if self.eval_freq < 1:
            raise ValueError("eval_freq must be >= 1")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be >= 1 or None")
        if self.device not in ("cpu", "auto", "cuda", "mps"):
            raise ValueError("device must be 'cpu', 'auto', 'cuda' or 'mps'")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, EnvConfig):
                v = v.to_dict()
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return json.loads(json.dumps(out))  # deep copy + guarantees JSON-ability

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RLTrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown RLTrainConfig keys: {unknown}")
        return cls(**dict(payload))

    def replace(self, **changes: Any) -> RLTrainConfig:
        return dataclasses.replace(self, **changes)


def resolve_device(device: str) -> str:
    """``"auto"`` -> ``"cuda"`` if available, else ``"mps"`` (Apple), else ``"cpu"``."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------------------
# data preparation
# -----------------------------------------------------------------------------------------
@dataclass
class PreparedData:
    """Combined train+val history with train-fitted features.

    ``md`` holds bars ``[train | val]``; rows ``[0, n_train)`` are training bars. ``features``
    is the pipeline-transformed frame on ``md.bars.index``; ``vol`` the causal EWMA vol.
    """

    md: MarketData
    features: pd.DataFrame
    pipeline: FeaturePipeline
    vol: pd.Series
    n_train: int

    @property
    def n_total(self) -> int:
        return len(self.md.bars)

    @property
    def val_range(self) -> tuple[int, int]:
        return self.n_train, self.n_total - 1


def _concat_market(md_train: MarketData, md_val: MarketData | None) -> tuple[MarketData, int]:
    bt = md_train.bars
    if len(bt) < 2:
        raise ValueError("md_train needs at least two bars")
    if md_val is None or len(md_val.bars) == 0:
        return md_train, len(bt)
    bv = md_val.bars
    if bv.index[0] <= bt.index[-1]:
        raise ValueError("md_val must start strictly after md_train ends (time-ordered split); "
                         f"train ends {bt.index[-1]}, val starts {bv.index[0]}")
    bars = pd.concat([bt, bv])
    bars.attrs = dict(bt.attrs)
    validate_bars(bars)
    macro = md_train.macro or md_val.macro or {}
    events = md_train.events if md_train.events is not None else md_val.events
    return MarketData(bars=bars, macro=macro, events=events), len(bt)


def _fit_rows(n_train: int, lookback: int) -> slice:
    """Training rows used to fit the scaler: skip the warm-up when enough rows remain."""
    skip = lookback if n_train - lookback >= max(200, n_train // 4) else 0
    return slice(skip, n_train)


def prepare_data(md_train: MarketData, md_val: MarketData | None, config: RLTrainConfig,
                 pipeline: FeaturePipeline | None = None) -> PreparedData:
    """Compute causal features on train+val, fit the scaler on TRAIN rows, transform all.

    If a FITTED ``pipeline`` is given it is reused as-is (no refit).
    """
    md, n_train = _concat_market(md_train, md_val)
    if pipeline is not None and pipeline.is_fitted:
        check_bar_size(pipeline, md.bars)
    if pipeline is None:
        pipeline = FeaturePipeline(
            groups=list(config.feature_groups) if config.feature_groups is not None else None,
            overrides=config.feature_overrides or None, scaler=config.scaler,
            clip=config.feature_clip)
    raw = pipeline.compute(md)
    if not pipeline.is_fitted:
        rows = _fit_rows(n_train, pipeline.max_lookback)
        pipeline.fit(raw.iloc[rows])
    features = pipeline.transform(raw)
    vol = ewma_volatility(md.bars["close"].astype(float), halflife_bars=config.env.vol_halflife)
    return PreparedData(md=md, features=features, pipeline=pipeline, vol=vol, n_train=n_train)


def _make_env(data: PreparedData, env_cfg: EnvConfig, *, train: bool,
              start: int | None = None, end: int | None = None) -> GoldTradingEnv:
    if train:
        lo, hi = 0, data.n_train - 1
    else:
        lo, hi = data.val_range
    # rates=md.macro: the same point-in-time benchmark run_backtest reads (backtest_forecast
    # passes data.md), so training rewards, rollouts and the engine agree under rate financing
    return GoldTradingEnv(
        data.md.bars, data.features, env_cfg, events=data.md.events, vol=data.vol,
        start=lo if start is None else start, end=hi if end is None else end,
        random_start=train, rates=data.md.macro)


# -----------------------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------------------
def make_predictor(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Fast deterministic ``obs -> action`` for a single observation (SB3 policy, no grad)."""
    import torch

    policy = model.policy
    policy.set_training_mode(False)
    device = policy.device
    space = model.action_space
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)

    def predict(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            action = policy.get_distribution(x).get_actions(deterministic=True)
        a = action.cpu().numpy()[0]
        if low is not None:
            a = np.clip(a, low, high)
        return a

    return predict


@dataclass
class EvalOutcome:
    """Validation evaluation: engine backtest + the rollout that produced the forecasts."""

    metrics: dict[str, float]
    backtest: BacktestResult
    rollout: RolloutResult


def _clean_metrics(m: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in _METRIC_KEYS:
        if k in m:
            v = float(m[k])
            out[k] = v if math.isfinite(v) else float("nan")
    return out


def backtest_forecast(md: MarketData, forecast: pd.Series, env_cfg: EnvConfig, *,
                      vol: pd.Series | None = None, start: int | None = None,
                      end: int | None = None) -> BacktestResult:
    """Backtest any forecast series with the env's sizer, risk limits, costs and vol — the
    like-for-like comparison for an RL policy (baselines, held-out test periods)."""
    if vol is None:
        vol = ewma_volatility(md.bars["close"].astype(float), halflife_bars=env_cfg.vol_halflife)
    risk = (StandardRiskManager(RiskLimits(**env_cfg.risk), events=md.events)
            if env_cfg.risk is not None else None)
    with simulated_risk_logging():  # a research replay, not the real account
        return run_backtest(md, forecast, sizer=VolTargetSizer(**env_cfg.sizer), risk=risk,
                            costs=CostModel(**env_cfg.costs),
                            initial_equity=env_cfg.initial_equity, vol=vol, start=start, end=end)


def _bar_minutes_of(data: PreparedData) -> float:
    from aurum.features.volatility import bar_minutes

    m = data.pipeline.bar_minutes
    return float(m) if m else float(bar_minutes(data.md.bars))


def evaluate_policy(predict: Callable[[np.ndarray], Any], data: PreparedData,
                    env_cfg: EnvConfig, *, start: int | None = None,
                    end: int | None = None) -> EvalOutcome:
    """Out-of-sample evaluation over ``[start, end]`` (default: the validation slice) with
    the DEPLOYMENT convention, then the official metrics from the real engine.

    1. The deterministic policy is rolled forward exactly as the ``rl_ppo`` strategy does
       (:func:`aurum.rl.env.rollout` with ``restart_on_termination=True`` and fresh
       episodes at ``env_cfg.episode_anchor``), starting one anchor period BEFORE ``start``
       (causal: that only adds earlier bars). The forecasts on ``[start, end]`` are then
       exactly the ones a deployed strategy produces there, so checkpoint selection scores
       the behaviour that will actually trade (the policy is sensitive to its account-state
       convention: on real H1 data a continuous account and a monthly-anchored one agreed on
       only ~70% of 2020-21 forecasts).
    2. That forecast series goes through :func:`run_backtest` from ``start`` (fresh account,
       the env's sizer, risk limits, costs and vol) — the same measurement a walk-forward
       backtest of ``rl_ppo`` makes.

    Diagnostic ``engine_rollout_max_abs_diff``: the engine replayed over the rollout's LAST
    fresh episode, up to its first in-place restart, must reproduce the simulated account
    to the cent (env and engine share sizer, risk manager and simulator).
    """
    lo = data.n_train if start is None else int(start)
    hi = data.n_total - 1 if end is None else int(end)
    anchor = env_cfg.episode_anchor
    span = episode_anchor_span_bars(anchor, _bar_minutes_of(data))
    env = _make_env(data, env_cfg, train=False, start=max(0, lo - span), end=hi)
    starts = episode_anchor_mask(data.md.bars.index, anchor) if anchor else None
    ro = rollout(env, predict, compute_metrics=False, restart_on_termination=True,
                 episode_starts=starts)
    a0 = max(lo, env.range_start)
    bt = backtest_forecast(data.md, ro.forecast, env_cfg, vol=data.vol, start=a0,
                           end=env.range_end)
    diff = _engine_parity(data, ro, env_cfg, bt, a0)
    if math.isfinite(diff) and diff > 1e-6 * env_cfg.initial_equity:
        logger.warning("engine vs rollout equity mismatch: max |diff| = %.6f USD", diff)
    metrics = _clean_metrics(bt.metrics)
    fc = ro.forecast.iloc[a0: ro.end + 1]
    metrics.update({
        "frac_long": float((fc > 0).mean()),
        "frac_short": float((fc < 0).mean()),
        "frac_flat": float((fc == 0).mean()),
        "mean_abs_forecast": float(fc.abs().mean()),
        "reward_sum": float(np.nansum(ro.rewards.to_numpy()[a0:])),
        "n_risk_events": float(len(bt.risk_events)) if bt.risk_events is not None else 0.0,
        "n_sim_restarts": float(sum(1 for r in ro.info.get("restart_bars", []) if r > a0)),
        "terminated_early": float(ro.terminated),
        "engine_rollout_max_abs_diff": diff,
    })
    return EvalOutcome(metrics=metrics, backtest=bt, rollout=ro)


def _engine_parity(data: PreparedData, ro: RolloutResult, env_cfg: EnvConfig,
                   bt: BacktestResult, a0: int) -> float:
    """Max |engine - simulated| equity over the last fresh episode up to its first restart."""
    sim_eq = ro.result.equity
    if len(sim_eq) < 2:
        return float("nan")
    index = data.md.bars.index
    seg_lo = int(index.get_loc(sim_eq.index[0]))
    later = [r for r in ro.info.get("restart_bars", []) if r > seg_lo]
    seg_hi = min(later) if later else ro.end
    if seg_hi - seg_lo < 1:
        return float("nan")
    ref = bt if seg_lo == a0 else backtest_forecast(
        data.md, ro.forecast, env_cfg, vol=data.vol, start=seg_lo, end=seg_hi)
    seg = sim_eq.loc[index[seg_lo]: index[seg_hi]]
    return float((ref.equity.reindex(seg.index) - seg).abs().max())


def _score(metrics: Mapping[str, float], key: str) -> float:
    v = float(metrics.get(key, float("nan")))
    return v if math.isfinite(v) else -math.inf


# -----------------------------------------------------------------------------------------
# training
# -----------------------------------------------------------------------------------------
@dataclass
class RLTrainResult:
    """Outcome of :func:`train_ppo` (``model`` is the SELECTED checkpoint)."""

    artifact_dir: Path
    model: Any
    pipeline: FeaturePipeline
    config: RLTrainConfig
    val_metrics: dict[str, float]
    history: pd.DataFrame
    val_eval: EvalOutcome | None
    timesteps: int
    early_stopped: bool
    selected_timesteps: int | None
    data: PreparedData


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent, check=False)
        sha = out.stdout.strip()
        return sha or None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str]:
    import gymnasium
    import stable_baselines3
    import torch

    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__}


def _range_str(bars: pd.DataFrame) -> list[str]:
    return [str(bars.index[0]), str(bars.index[-1])] if len(bars) else []


def train_ppo(
    md_train: MarketData,
    md_val: MarketData,
    *,
    config: RLTrainConfig | None = None,
    out_dir: str | Path | None = None,
    pipeline: FeaturePipeline | None = None,
) -> RLTrainResult:
    """Train PPO on ``md_train``, select/early-stop on ``md_val``; write an artifact directory.

    Parameters
    ----------
    md_train, md_val : time-ordered, non-overlapping market data (``md_val`` after
        ``md_train``). Pass the same ``macro``/``events`` objects to both.
    config  : :class:`RLTrainConfig` (defaults if None).
    out_dir : artifact directory (created; default: a fresh temporary directory).
    pipeline: optional pre-FITTED feature pipeline to reuse instead of fitting one.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from torch import nn

    cfg = config if config is not None else RLTrainConfig()
    t_start = time.monotonic()
    out = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="aurum_rl_"))
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    if cfg.torch_threads:
        import torch

        torch.set_num_threads(int(cfg.torch_threads))

    data = prepare_data(md_train, md_val, cfg, pipeline=pipeline)
    if data.n_total - data.n_train < 2:
        raise ValueError("md_val needs at least two bars")
    logger.info("RL data: %d train bars, %d val bars, %d features (window %d)",
                data.n_train, data.n_total - data.n_train, data.features.shape[1], cfg.env.window)

    def env_fn(rank: int) -> Callable[[], Any]:
        def _init() -> Any:
            return Monitor(_make_env(data, cfg.env, train=True))
        return _init

    venv: Any = DummyVecEnv([env_fn(i) for i in range(cfg.n_envs)])
    venv.seed(cfg.seed)
    if cfg.normalize_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.gamma,
                            clip_reward=10.0)
    activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[cfg.activation]
    arch = list(cfg.net_arch)
    model = PPO(
        "MlpPolicy", venv, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range, ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}, "activation_fn": activation},
        seed=cfg.seed, device=device, verbose=cfg.verbose)

    ckpt_dir = Path(tempfile.mkdtemp(prefix="aurum_rl_ckpt_"))
    best_path = ckpt_dir / "best_policy.zip"
    history: list[dict[str, Any]] = []
    state: dict[str, Any] = {"best": -math.inf, "best_ts": None, "best_metrics": None,
                             "bad": 0, "stopped": False, "last_eval": 0}

    def run_eval(num_timesteps: int) -> bool:
        """Evaluate the current policy; returns False to stop training."""
        was_training = bool(model.policy.training)
        outcome = evaluate_policy(make_predictor(model), data, cfg.env)
        model.policy.set_training_mode(was_training)
        m = outcome.metrics
        score = _score(m, cfg.select_metric)
        ep_rew = [e["r"] for e in model.ep_info_buffer] if model.ep_info_buffer else []
        row = {"timesteps": int(num_timesteps), "wall_s": round(time.monotonic() - t_start, 2),
               "train_ep_reward_mean": float(np.mean(ep_rew)) if ep_rew else float("nan"), **m}
        improved = score > state["best"] + cfg.min_delta
        row["selected"] = bool(improved)
        history.append(row)
        logger.info("RL eval @%d: val %s=%.3f total_return=%.4f trades=%s (best %.3f)",
                    num_timesteps, cfg.select_metric, score, m.get("total_return", float("nan")),
                    m.get("n_trades"), max(score, state["best"]))
        if improved:
            state.update(best=score, best_ts=int(num_timesteps), best_metrics=dict(m), bad=0)
            model.save(best_path)
        else:
            state["bad"] += 1
        if cfg.patience is not None and state["bad"] >= cfg.patience:
            logger.info("RL early stop: %d evaluations without improvement", state["bad"])
            state["stopped"] = True
            return False
        if cfg.max_wall_time_s is not None and time.monotonic() - t_start > cfg.max_wall_time_s:
            logger.info("RL stop: wall-time budget %.0fs reached", cfg.max_wall_time_s)
            state["stopped"] = True
            return False
        return True

    class _ValidationCallback(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps - state["last_eval"] >= cfg.eval_freq:
                state["last_eval"] = self.num_timesteps
                return run_eval(self.num_timesteps)
            return True

    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=_ValidationCallback(),
                    progress_bar=False)
        trained = int(model.num_timesteps)
        if not state["stopped"] and state["last_eval"] != trained:
            run_eval(trained)
        if best_path.exists():
            model = PPO.load(best_path, device=device)
        else:  # no finite validation score at all: keep the final policy
            logger.warning("no validation evaluation produced a finite %s; keeping the final "
                           "policy", cfg.select_metric)
        model.save(out / "policy.zip")
    finally:
        venv.close()
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    val_eval = evaluate_policy(make_predictor(model), data, cfg.env)
    hist_df = pd.DataFrame(history)
    data.pipeline.save(out / "pipeline.json")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    bars = data.md.bars
    data_hash = {
        "train_bars_sha256": frame_hash(bars.iloc[: data.n_train]),
        "val_bars_sha256": frame_hash(bars.iloc[data.n_train:]),
        "train_range": _range_str(bars.iloc[: data.n_train]),
        "val_range": _range_str(bars.iloc[data.n_train:]),
        "n_train_bars": data.n_train,
        "n_val_bars": data.n_total - data.n_train,
        "timeframe": bars.attrs.get("timeframe"),
        "macro_series": sorted((data.md.macro or {}).keys()),
        "has_events": data.md.events is not None,
    }
    (out / "data_hash.json").write_text(json.dumps(data_hash, indent=2), encoding="utf-8")
    metrics = {
        "val": val_eval.metrics,
        "selected_timesteps": state["best_ts"],
        "select_metric": cfg.select_metric,
        "n_evals": len(history),
        "timesteps_trained": trained,
        "early_stopped": bool(state["stopped"]),
        "wall_time_s": round(time.monotonic() - t_start, 2),
        "device": device,
        "n_features": int(data.features.shape[1]),
        "obs_dim": int(cfg.env.window * data.features.shape[1] + 4),
        "history": json.loads(hist_df.to_json(orient="records")) if len(hist_df) else [],
        "data": data_hash,
        "git_sha": _git_sha(),
        "versions": _versions(),
        "note": ("val metrics are for the checkpoint SELECTED on this validation slice (best of "
                 "n_evals): optimistic; deflate with n_trials=n_evals or test on later data."),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float),
                                      encoding="utf-8")
    if len(hist_df):
        hist_df.to_csv(out / "history.csv", index=False)
    logger.info("RL artifact written to %s (val %s %.3f at %s timesteps)", out,
                cfg.select_metric, val_eval.metrics.get(cfg.select_metric, float("nan")),
                state["best_ts"])
    return RLTrainResult(
        artifact_dir=out, model=model, pipeline=data.pipeline, config=cfg,
        val_metrics=val_eval.metrics, history=hist_df, val_eval=val_eval, timesteps=trained,
        early_stopped=bool(state["stopped"]), selected_timesteps=state["best_ts"], data=data)


# -----------------------------------------------------------------------------------------
# artifacts
# -----------------------------------------------------------------------------------------
@dataclass
class RLArtifact:
    """A trained policy with everything needed to reproduce its observations.

    ``path`` is where it was loaded from — informational only (``None`` or a directory that
    no longer exists when it was rebuilt from embedded bytes, see
    :func:`load_artifact_bytes`)."""

    path: Path | None
    model: Any
    pipeline: FeaturePipeline
    config: RLTrainConfig
    metrics: dict[str, Any]
    _predict: Callable[[np.ndarray], Any] | None = field(default=None, repr=False)

    def predict(self, obs: np.ndarray) -> Any:
        """Deterministic action for one observation."""
        if self._predict is None:
            self._predict = make_predictor(self.model)
        return self._predict(obs)


#: Files that define a policy's behaviour (network, feature scaling, env config); together
#: with the optional ``metrics.json`` they are what :func:`read_artifact_bytes` returns.
POLICY_FILES = ("policy.zip", "pipeline.json", "config.json")


def read_artifact_bytes(path: str | Path) -> dict[str, bytes]:
    """The artifact's defining files (``POLICY_FILES`` + ``metrics.json`` when present) as
    bytes, e.g. to embed a policy in a pickled strategy so it no longer depends on the
    directory (see ``aurum.strategies.rl``)."""
    p = Path(path)
    missing = [f for f in POLICY_FILES if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"RL artifact {p} is missing {missing}")
    names = [*POLICY_FILES, *(["metrics.json"] if (p / "metrics.json").exists() else [])]
    return {name: (p / name).read_bytes() for name in names}


def load_artifact_bytes(files: Mapping[str, bytes], *, device: str = "cpu",
                        path: str | Path | None = None) -> RLArtifact:
    """Rebuild an :class:`RLArtifact` from the bytes of its files (no filesystem access):
    ``policy.zip`` (SB3 accepts a file object), ``pipeline.json``, ``config.json`` and the
    optional ``metrics.json``. ``path`` is recorded for provenance only."""
    import io

    from stable_baselines3 import PPO

    missing = [f for f in POLICY_FILES if f not in files]
    if missing:
        raise ValueError(f"RL artifact bytes are missing {missing}")
    cfg = RLTrainConfig.from_dict(json.loads(files["config.json"].decode("utf-8")))
    pipe = FeaturePipeline.from_dict(json.loads(files["pipeline.json"].decode("utf-8")))
    model = PPO.load(io.BytesIO(files["policy.zip"]), device=resolve_device(device))
    raw_metrics = files.get("metrics.json")
    metrics = json.loads(raw_metrics.decode("utf-8")) if raw_metrics else {}
    return RLArtifact(path=None if path is None else Path(path), model=model, pipeline=pipe,
                      config=cfg, metrics=metrics)


def load_artifact(path: str | Path, *, device: str = "cpu") -> RLArtifact:
    """Load ``policy.zip`` + ``pipeline.json`` + ``config.json`` (+ ``metrics.json``)."""
    return load_artifact_bytes(read_artifact_bytes(path), device=device, path=path)


def rollout_artifact(
    artifact: RLArtifact,
    md: MarketData,
    *,
    start: int | str | pd.Timestamp | None = None,
    end: int | str | pd.Timestamp | None = None,
    compute_metrics: bool = False,
    kill_switches: bool = True,
    episode_anchor: str | None = None,
) -> RolloutResult:
    """Run an artifact's deterministic policy over ``md`` with its own simulated position.

    Features are computed on ALL of ``md`` (causal) and scaled with the artifact's fitted
    pipeline; the vol, sizer, risk limits and costs are the artifact's training settings.
    ``md`` must have the bar size the policy was trained on (``ValueError`` otherwise).

    The simulated account always runs the TRAINING risk limits, so its kill switches fire
    (and flatten) exactly as they did in training. ``kill_switches=True`` (evaluation):
    a kill switch or ruin ends the rollout, like the engine's persistent kill.
    ``kill_switches=False`` (strategy adapter): the rollout does not stop there — the
    simulated account starts a new episode in place (drawdown peak and risk state re-based,
    see :meth:`GoldTradingEnv.rebase_episode`), as the next training episode would. A
    long rollout thus keeps producing forecasts after a simulated drawdown while every
    observation stays inside the state distribution the policy was trained on (disabling
    the kills instead would let the drawdown state drift far beyond anything seen in
    training). The real risk manager downstream still gates actual trading.

    ``episode_anchor`` (``"W"``/``"M"``/``"Q"``, see :func:`aurum.rl.episode_anchor_mask`):
    a fresh simulated episode (flat, initial equity) begins at the first bar of every
    calendar period, so the policy's state — and hence its forecasts — after an anchor do
    not depend on where ``md`` starts. ``None`` = one continuous simulated account.
    """
    check_bar_size(artifact.pipeline, md.bars)
    raw = artifact.pipeline.compute(md)
    x = artifact.pipeline.transform(raw)
    env = GoldTradingEnv(md.bars, x, artifact.config.env, events=md.events, start=start,
                         end=end, random_start=False, rates=md.macro)
    starts = episode_anchor_mask(md.bars.index, episode_anchor) if episode_anchor else None
    return rollout(env, artifact.predict, compute_metrics=compute_metrics,
                   restart_on_termination=not kill_switches, episode_starts=starts)


def check_bar_size(pipeline: FeaturePipeline, bars: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``bars`` are not the bar size ``pipeline`` was fitted on.

    A policy's scaler statistics, feature warm-ups, vol annualisation and learned dynamics
    are all specific to one bar size; :class:`FeaturePipeline` only logs a warning (and
    silently re-derives its warm-up), so an H1 policy would otherwise trade H4 bars.
    """
    from aurum.features.volatility import bar_minutes

    fitted = pipeline.bar_minutes
    if fitted is None or len(bars) == 0:
        return
    got = float(bar_minutes(bars))
    if not (math.isfinite(got) and abs(got - float(fitted)) < 1e-6):
        raise ValueError(f"RL policy was trained on {float(fitted):g}-minute bars but got "
                         f"{got:g}-minute bars")


# -----------------------------------------------------------------------------------------
# CLI: real-data experiment
# -----------------------------------------------------------------------------------------
def _baseline_forecasts(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """Simple causal benchmarks run through the same sizer/risk/costs."""
    close = bars["close"].astype(float)
    minutes = (bars.index[1] - bars.index[0]).total_seconds() / 60.0 if len(bars) > 1 else 60.0
    per_day = max(1, int(round(23 * 60 / max(minutes, 1.0))))
    mom = np.sign(np.log(close / close.shift(20 * per_day))).fillna(0.0)
    return {
        "long_voltarget": pd.Series(1.0, index=bars.index),
        "tsmom_20d_sign": mom,
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised manually
    """Train on one period, validate on the next, optionally test on a later untouched one."""
    from aurum.data.calendar import generate_rule_based_calendar
    from aurum.data.macro import load_macro_dir
    from aurum.data.store import load_bars
    from aurum.research.stats import sharpe_summary

    ap = argparse.ArgumentParser(description="Train a PPO policy on XAUUSD bars (aurum.rl).")
    ap.add_argument("--bars", default="data_store/xauusd_H1.parquet")
    ap.add_argument("--macro-dir", default=None)
    ap.add_argument("--events", action="store_true", help="use the rule-based NFP/FOMC calendar")
    ap.add_argument("--train-start", default="2012-01-01")
    ap.add_argument("--train-end", default="2017-12-31")
    ap.add_argument("--val-end", default="2019-12-31")
    ap.add_argument("--test-end", default=None, help="optional untouched test period end")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--episode-length", type=int, default=1024)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--dd-penalty", type=float, default=0.0)
    ap.add_argument("--turnover-penalty", type=float, default=0.0)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="discrete forecast levels (default -1 -0.5 0 0.5 1)")
    ap.add_argument("--rebalance-band", type=float, default=0.10)
    ap.add_argument("--max-wall", type=float, default=18 * 60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/rl/ppo")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("aurum.risk.manager").setLevel(logging.WARNING)  # per-bar interventions

    bars = load_bars(args.bars)
    end = args.test_end or args.val_end
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    day = pd.Timedelta(days=1)
    bars = bars.loc[(bars.index >= ts(args.train_start)) & (bars.index < ts(end) + day)]
    macro = load_macro_dir(args.macro_dir) if args.macro_dir else {}
    events = (generate_rule_based_calendar(bars.index[0] - 30 * day, bars.index[-1] + 30 * day)
              if args.events else None)
    b_tr = bars.loc[bars.index < ts(args.train_end) + day]
    b_va = bars.loc[(bars.index >= ts(args.train_end) + day) & (bars.index < ts(args.val_end) + day)]
    md_tr = MarketData(bars=b_tr, macro=macro, events=events)
    md_va = MarketData(bars=b_va, macro=macro, events=events)
    env_kw: dict[str, Any] = {}
    if args.levels:
        env_kw["action_levels"] = tuple(args.levels)
    env_cfg = EnvConfig(window=args.window, episode_length=args.episode_length,
                        dd_penalty=args.dd_penalty, turnover_penalty=args.turnover_penalty,
                        sizer={"target_vol": 0.10, "max_leverage": 2.0,
                               "rebalance_band": args.rebalance_band}, **env_kw)
    cfg = RLTrainConfig(
        feature_groups=tuple(args.groups) if args.groups else DEFAULT_FEATURE_GROUPS,
        env=env_cfg, total_timesteps=args.timesteps, n_envs=args.n_envs,
        eval_freq=args.eval_freq, patience=args.patience, ent_coef=args.ent_coef,
        gamma=args.gamma, learning_rate=args.lr,
        max_wall_time_s=args.max_wall, seed=args.seed, device=args.device)
    res = train_ppo(md_tr, md_va, config=cfg, out_dir=args.out)
    data = res.data
    lo, hi = data.val_range
    cols = ["sharpe", "total_return", "ann_vol", "max_drawdown", "n_trades", "exposure",
            "total_costs"]

    def compare(md: MarketData, rl_forecast: pd.Series, vol: pd.Series, a: int, b: int
                ) -> dict[str, dict[str, float]]:
        rows = {"rl_ppo (selected)": rl_forecast}
        rows.update(_baseline_forecasts(md.bars))
        rows["flat"] = pd.Series(0.0, index=md.bars.index)
        return {k: _clean_metrics(backtest_forecast(md, f, cfg.env, vol=vol, start=a, end=b).metrics)
                for k, f in rows.items()}

    assert res.val_eval is not None
    val_rows = compare(data.md, res.val_eval.rollout.forecast, data.vol, lo, hi)
    val_range = _range_str(data.md.bars.iloc[lo:hi + 1])
    print("\nVALIDATION", val_range)
    print(pd.DataFrame(val_rows).T.reindex(columns=cols).round(4).to_string())
    from aurum.backtest.metrics import daily_returns

    vr = daily_returns(res.val_eval.backtest.equity)
    n_trials = max(1, len(res.history))
    ss = sharpe_summary(vr.to_numpy(), 252, n_trials=n_trials, bootstrap=False)
    print(f"val daily Sharpe {ss.get('sharpe', float('nan')):.3f}  PSR {ss.get('psr', float('nan')):.3f}"
          f"  DSR(n_trials={n_trials}) {ss.get('dsr', float('nan')):.3f}")
    summary: dict[str, Any] = {"validation": val_rows, "val_range": val_range, "val_stats": ss,
                               "history": res.history.to_dict(orient="records")}
    if args.test_end:
        full = MarketData(bars=bars, macro=macro, events=events)
        art = load_artifact(res.artifact_dir)
        t0 = int(bars.index.searchsorted(ts(args.val_end) + day))
        # Same (deployment) convention as validation, with the artifact's own pipeline.
        data_full = prepare_data(full, None, art.config, pipeline=art.pipeline)
        ev_test = evaluate_policy(art.predict, data_full, art.config.env, start=t0)
        test_rows = compare(full, ev_test.rollout.forecast, data_full.vol, t0, len(bars) - 1)
        print("\nTEST (untouched)", _range_str(bars.iloc[t0:]))
        print(pd.DataFrame(test_rows).T.reindex(columns=cols).round(4).to_string())
        summary["test"] = test_rows
        summary["test_range"] = _range_str(bars.iloc[t0:])
    Path(args.out, "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nartifact: {res.artifact_dir}  timesteps={res.timesteps} "
          f"selected@{res.selected_timesteps} early_stopped={res.early_stopped}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
