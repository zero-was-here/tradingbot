"""Gymnasium environment for XAUUSD built on the shared execution stack (SPEC §0.2, §1, §7-8).

Why a rebuild
-------------
The legacy environment (``legacy/env/xauusd_env.py``) had three defects that make any
result from it meaningless: the observation at ``t`` could see bar ``t+1``; the shaped
reward (``pnl - penalties + bonuses``) was compounded into the reported equity, so
"equity" was not money; and training only ever sampled the first 20k bars. This module
fixes all three by construction:

* **One simulator.** Every step goes forecast -> :class:`~aurum.portfolio.sizing.VolTargetSizer`
  -> optional :class:`~aurum.risk.manager.StandardRiskManager` -> :class:`ExecutionSimulator`,
  exactly the chain of :func:`aurum.backtest.engine.run_backtest`. The equity path of an
  evaluation rollout is therefore identical to the engine's for the same forecast series
  (verified in ``tests/test_rl_env.py``), costs (spread, slippage, commission, swap) included.
* **Timing (SPEC §1).** The observation returned at bar ``t`` uses only feature rows
  ``t-window+1 .. t`` (features are causal by the ``aurum.features`` contract), the causal
  EWMA volatility at ``t`` and the account state marked at ``close[t]``. The action taken on
  that observation fills at the OPEN of ``t+1``; its reward is realised over bar ``t+1``.
* **Reward is not money.** ``reward = scale * log(E_{t+1}/E_t) - lambda_dd * scale *
  max(0, DD_{t+1} - DD_t) - turnover_penalty * |dPosition| / full_position``. Shaping
  terms never touch the simulator; ``info`` carries the true equity, PnL and costs.
* **Whole training range.** Episodes start uniformly at random over the full (warmed-up)
  range with a fixed length, drawn from the env's seeded ``np_random``.

Reward design
-------------
The log-growth term is the per-step increment of ``log(wealth)``, whose sum over an episode
is the episode's log return: maximising it is the Kelly / growth-optimal criterion (Kelly,
1956; Thorp, 2006) and, unlike raw PnL, it is scale-free across equity levels. Vol-targeted
sizing keeps the reward variance roughly stationary across calm and turbulent regimes
(Moreira & Muir, 2017). The optional drawdown term penalises only *increases* of the
drawdown from the running peak — a differentiable proxy for the Calmar/"pain" objectives of
Moody & Saffell (2001, "Learning to trade via direct reinforcement", IEEE TNN 12(4)) — so
losses while underwater weigh ``1 + lambda_dd`` times as much as gains. The turnover term is
an explicit trading-cost prior on top of the simulated costs (Garleanu & Pedersen, 2013).

Agent state in the observation
------------------------------
Appended after the flattened feature window (all computed at the close of ``t``):

* ``state_position``: current lots / lots a +1 forecast would get now (sizer at the current
  vol, equity and price, no drawdown scaling) — i.e. the forecast currently "held";
* ``state_upnl_vol``: unrealised log return of the open position vs its lot-weighted average
  entry mid, in units of DAILY volatility (``vol_ann / sqrt(252)``);
* ``state_time_in_trade``: ``log1p(bars held) / log1p(trade_time_norm)``;
* ``state_drawdown``: drawdown from the episode's equity peak x 10 (10% -> 1.0).

:func:`rollout` drives a policy bar by bar with its own simulated position, which is how
validation (``aurum.rl.train``) and the ``rl_ppo`` strategy adapter produce forecasts whose
observation state matches training.

This module imports ``gymnasium`` but never ``torch`` / ``stable_baselines3``.
"""

from __future__ import annotations

import contextvars
import logging
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from aurum.backtest.result import BacktestResult
from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.interfaces import PositionSizer, RiskContext, RiskManager
from aurum.execution.costs import CostModel, RateSource
from aurum.execution.simulator import ExecutionSimulator, StepResult
from aurum.models.volatility import ewma_volatility
from aurum.portfolio.sizing import VolTargetSizer
from aurum.risk.manager import RiskLimits, StandardRiskManager
from aurum.rl import EPISODE_ANCHOR_SPAN_DAYS

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ACTION_LEVELS",
    "STATE_NAMES",
    "EnvConfig",
    "GoldTradingEnv",
    "RolloutResult",
    "rollout",
    "simulated_risk_logging",
]

# ---- log hygiene for SIMULATED accounts ------------------------------------------------------
# The env's StandardRiskManager is a simulation (training episodes, validation, and the rl_ppo
# strategy, which re-rolls its whole history window on every live bar). Its records ("risk
# intervention", CRITICAL "RISK KILL SWITCH") are indistinguishable from the real account's,
# so they would re-appear on every live bar and mislead operators/alerting. While the context
# flag is set, records of the risk manager's logger are dropped; the env's ``info`` dicts carry
# the reasons instead. The real risk manager (flag unset) logs exactly as before.
_RISK_LOGGER_NAME = "aurum.risk.manager"
_SIMULATED_RISK = contextvars.ContextVar("aurum_rl_simulated_risk", default=False)


class _SimulatedRiskLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _SIMULATED_RISK.get()


def _install_risk_log_filter() -> None:
    lg = logging.getLogger(_RISK_LOGGER_NAME)
    if not any(type(f).__name__ == "_SimulatedRiskLogFilter" for f in lg.filters):
        lg.addFilter(_SimulatedRiskLogFilter())


_install_risk_log_filter()


@contextmanager
def simulated_risk_logging() -> Iterator[None]:
    """Silence ``aurum.risk.manager`` records emitted inside the block (simulated accounts
    only; thread/async-safe via a context variable)."""
    token = _SIMULATED_RISK.set(True)
    try:
        yield
    finally:
        _SIMULATED_RISK.reset(token)

#: Default discrete forecast levels (short, half short, flat, half long, long).
DEFAULT_ACTION_LEVELS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
#: Names of the agent-state entries appended to every observation (in order).
STATE_NAMES: tuple[str, ...] = (
    "state_position", "state_upnl_vol", "state_time_in_trade", "state_drawdown",
)
_N_STATE = len(STATE_NAMES)
_EPS = 1e-9
_TRADING_DAYS = 252.0


def _default_sizer() -> dict[str, Any]:
    return {"target_vol": 0.10, "max_leverage": 2.0, "rebalance_band": 0.10}


def _default_risk() -> dict[str, Any]:
    # Research limits (see docs/INTERFACES.md): a -3% day halts only for the rest of that
    # day; the 20% max-drawdown kill stays a hard stop (and ends the episode).
    return {"daily_loss_persistent": False}


@dataclass
class EnvConfig:
    """Configuration of :class:`GoldTradingEnv` (JSON-serialisable via :meth:`to_dict`).

    window            : feature rows per observation (``t-window+1 .. t``).
    action_mode       : ``"discrete"`` (index into ``action_levels``) or ``"continuous"``
                        (``Box[-1, 1]`` forecast).
    action_levels     : forecast levels of the discrete action space.
    episode_length    : decisions per training episode (random-start mode).
    reward_scale      : multiplier of the per-step log equity growth (100 -> 1% = 1.0).
    dd_penalty        : ``lambda_dd``: weight of drawdown *increases* (same units as log growth).
    turnover_penalty  : reward units per full position (lots of a +1 forecast) traded.
    initial_equity    : account equity at every episode start (USD).
    sizer             : ``VolTargetSizer`` keyword arguments.
    risk              : ``RiskLimits`` keyword arguments (a fresh ``StandardRiskManager`` per
                        episode), or ``None`` for no risk manager.
    costs             : ``CostModel`` keyword arguments (``{}`` = defaults, incl. rate-based
                        financing; ``{"financing": {"mode": "fixed"}}`` etc. to change it).
    vol_halflife      : half-life (bars) of the causal EWMA vol used by the sizer (48 = engine
                        default, which keeps env and ``run_backtest`` identical).
    min_equity_frac   : terminate when equity falls below this fraction of the episode's
                        starting equity ("ruin"), besides simulator bankruptcy.
    terminate_on_halt : end the episode when the risk manager's kill switch is engaged
                        (a transient daily-loss halt with ``daily_loss_persistent=False``
                        does not end it).
    obs_clip          : observations are clipped to ``[-obs_clip, obs_clip]``.
    trade_time_norm   : bars at which ``state_time_in_trade`` reaches 1.
    episode_anchor    : convention for DETERMINISTIC policy rollouts that produce forecasts
                        (validation, test, the ``rl_ppo`` strategy; not training): the
                        simulated account starts a fresh episode at the first bar of every
                        ``"W"``/``"M"``/``"Q"`` calendar period (``None`` = one continuous
                        account). Anchored forecasts do not depend on where the history
                        starts, so validation, backtests and live sliding windows agree.
    """

    window: int = 4
    action_mode: str = "discrete"
    action_levels: tuple[float, ...] = DEFAULT_ACTION_LEVELS
    episode_length: int = 1024
    reward_scale: float = 100.0
    dd_penalty: float = 0.0
    turnover_penalty: float = 0.0
    initial_equity: float = 100_000.0
    sizer: dict[str, Any] = field(default_factory=_default_sizer)
    risk: dict[str, Any] | None = field(default_factory=_default_risk)
    costs: dict[str, Any] = field(default_factory=dict)
    vol_halflife: float = 48.0
    min_equity_frac: float = 0.5
    terminate_on_halt: bool = True
    obs_clip: float = 10.0
    trade_time_norm: int = 240
    episode_anchor: str | None = "M"

    def __post_init__(self) -> None:
        self.action_levels = tuple(float(x) for x in self.action_levels)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        if len(self.action_levels) < 2 or any(abs(x) > 1.0 for x in self.action_levels):
            raise ValueError("action_levels needs >= 2 forecast levels within [-1, 1]")
        if self.episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        if not (self.reward_scale > 0 and math.isfinite(self.reward_scale)):
            raise ValueError("reward_scale must be positive")
        if self.dd_penalty < 0 or self.turnover_penalty < 0:
            raise ValueError("penalties must be >= 0")
        if not self.initial_equity > 0:
            raise ValueError("initial_equity must be positive")
        if not 0.0 <= self.min_equity_frac < 1.0:
            raise ValueError("min_equity_frac must be in [0, 1)")
        if not self.obs_clip > 0:
            raise ValueError("obs_clip must be positive")
        if self.trade_time_norm < 1:
            raise ValueError("trade_time_norm must be >= 1")
        if self.episode_anchor is not None and self.episode_anchor not in EPISODE_ANCHOR_SPAN_DAYS:
            raise ValueError(f"episode_anchor must be one of {sorted(EPISODE_ANCHOR_SPAN_DAYS)} "
                             "or None")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_levels"] = list(self.action_levels)
        return _jsonable(d)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EnvConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown EnvConfig keys: {unknown}")
        return cls(**dict(payload))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _resolve_bound(index: pd.DatetimeIndex, value: int | str | pd.Timestamp | None, *,
                   is_end: bool) -> int:
    """Bar position of an inclusive ``start``/``end`` bound (int position or timestamp)."""
    n = len(index)
    if value is None:
        return n - 1 if is_end else 0
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        pos = int(value)
        if pos < 0:
            pos += n
        if not 0 <= pos < n:
            raise IndexError(f"{'end' if is_end else 'start'}={value} outside [0, {n})")
        return pos
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    if is_end:
        return int(index.searchsorted(ts, side="right")) - 1
    return int(index.searchsorted(ts, side="left"))


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


class GoldTradingEnv(gym.Env):
    """Single-instrument trading environment on :class:`ExecutionSimulator`.

    Parameters
    ----------
    bars      : canonical bars frame (``aurum.data.schema``); history before ``start`` is
                used only for warm-up.
    features  : TRANSFORMED feature frame indexed exactly like ``bars`` (a
                :class:`~aurum.features.pipeline.FeaturePipeline` fitted on the TRAINING
                slice only, then ``transform``-ed over the whole history). Rows before the
                first fully finite row are warm-up and never observed; later NaN -> 0 (the
                scaled training median, as the pipeline does).
    config    : :class:`EnvConfig`.
    events    : optional calendar frame for the risk manager's event blackouts.
    vol       : optional annualised vol forecast per bar (default: causal
                ``ewma_volatility(close, halflife_bars=config.vol_halflife)`` on ``bars``).
    start, end: inclusive range (positions or timestamps) of decision bars: episodes start at
                or after ``start`` (and after the feature warm-up) and never settle a bar
                beyond ``end``.
    random_start : ``True`` (training): uniform random start over the range, fixed
                ``episode_length``. ``False`` (evaluation): start at the first eligible bar
                and run to ``end`` (``reset(options={"start": i})`` overrides the start).
    sizer, costs, instrument : optional objects overriding ``config.sizer`` / ``config.costs``.
    rates     : benchmark-rate source for ``"rate"`` financing (``md.macro`` — pass the same
                one the engine gets so env and :func:`~aurum.backtest.engine.run_backtest`
                equity paths stay identical; ``None`` -> ``financing.fallback_rate``). Rates are
                read as of each rollover by the simulator (point-in-time).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        config: EnvConfig | None = None,
        *,
        events: pd.DataFrame | None = None,
        vol: pd.Series | np.ndarray | None = None,
        start: int | str | pd.Timestamp | None = None,
        end: int | str | pd.Timestamp | None = None,
        random_start: bool = True,
        sizer: PositionSizer | None = None,
        costs: CostModel | None = None,
        instrument: Instrument = XAUUSD,
        rates: RateSource = None,
    ) -> None:
        super().__init__()
        self.config = cfg = config if config is not None else EnvConfig()
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame indexed like bars")
        if not features.index.equals(bars.index):
            raise ValueError("features must be indexed exactly like bars (compute/transform the "
                             "pipeline on the same bars)")
        if features.shape[1] == 0:
            raise ValueError("features has no columns")
        n = len(bars)
        if n < cfg.window + 2:
            raise ValueError(f"need at least window+2={cfg.window + 2} bars, got {n}")
        index = pd.DatetimeIndex(bars.index)
        self.bars = bars
        self.feature_names: list[str] = [str(c) for c in features.columns]
        self.instrument = instrument
        self.events = events
        self.random_start = bool(random_start)
        self.sizer: PositionSizer = sizer if sizer is not None else VolTargetSizer(**cfg.sizer)
        self.costs = costs if costs is not None else CostModel(**cfg.costs)
        self.risk_limits: RiskLimits | None = (
            RiskLimits(**cfg.risk) if cfg.risk is not None else None
        )

        # Features: warm-up rows keep NaN (never observed), later NaN -> 0 (neutral).
        feat = features.to_numpy(dtype=np.float64, copy=True)
        feat[~np.isfinite(feat)] = np.nan
        row_ok = ~np.isnan(feat).any(axis=1)
        if not row_ok.any():
            raise ValueError("features have no fully finite row")
        first_valid = int(np.argmax(row_ok))
        tail = feat[first_valid:]
        tail[np.isnan(tail)] = 0.0
        self._feat = feat.astype(np.float32)
        self.first_valid_row = first_valid
        self.first_obs_index = first_valid + cfg.window - 1

        # Causal vol forecast for the sizer (same default as the backtest engine).
        if vol is None:
            vol_s = ewma_volatility(bars["close"].astype(float), halflife_bars=cfg.vol_halflife)
            vol_arr = vol_s.to_numpy(dtype=float, copy=True)
        elif isinstance(vol, pd.Series):
            vol_arr = vol.reindex(index).ffill().to_numpy(dtype=float, copy=True)
        else:
            vol_arr = np.asarray(vol, dtype=float).copy()
            if vol_arr.shape != (n,):
                raise ValueError(f"vol must have one value per bar ({n}), got {vol_arr.shape}")
        # Same gap rule as run_backtest (non-finite -> 0.20); the sizer flattens on vol <= 0.
        vol_arr[~np.isfinite(vol_arr)] = 0.20
        self._vol: list[float] = vol_arr.tolist()

        lo = _resolve_bound(index, start, is_end=False)
        hi = _resolve_bound(index, end, is_end=True)
        self.range_start = max(lo, self.first_obs_index)
        self.range_end = hi
        if self.range_end - self.range_start < 1:
            raise ValueError(
                f"episode range [{self.range_start}, {self.range_end}] has fewer than two bars "
                f"(feature warm-up ends at bar {self.first_obs_index})")

        # One simulator per env, over the bars it may ever touch.
        sim_bars = bars.iloc[: hi + 1]
        self.sim = ExecutionSimulator(sim_bars, instrument, self.costs, cfg.initial_equity, rates=rates)
        self._close: list[float] = sim_bars["close"].to_numpy(dtype=float).tolist()
        self._open: list[float] = sim_bars["open"].to_numpy(dtype=float).tolist()
        self._spread: list[float] = sim_bars["spread"].to_numpy(dtype=float).tolist()
        self._times: list[pd.Timestamp] = list(sim_bars.index)
        self._dtimes: list[pd.Timestamp] = list(pd.DatetimeIndex(sim_bars["available_at"]))

        # Spaces.
        n_feat = self._feat.shape[1]
        self.obs_dim = cfg.window * n_feat + _N_STATE
        clip = float(cfg.obs_clip)
        self.observation_space = spaces.Box(low=-clip, high=clip, shape=(self.obs_dim,),
                                            dtype=np.float32)
        if cfg.action_mode == "discrete":
            self.action_space = spaces.Discrete(len(cfg.action_levels))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._levels = np.asarray(cfg.action_levels, dtype=float)

        # Episode state (set by reset).
        self.risk: RiskManager | None = None
        self._episode_start = self.range_start
        self._episode_equity = float(cfg.initial_equity)  # reference of the ruin threshold
        self._steps = 0
        self._entry_ref = math.nan
        self._bars_in_trade = 0
        self._last_forecast = 0.0
        self._needs_reset = True

    # ---- helpers ---------------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return int(self._feat.shape[1])

    @property
    def episode_start(self) -> int:
        return self._episode_start

    @property
    def max_start(self) -> int:
        """Largest random episode start (a full episode fits before ``range_end``)."""
        return max(self.range_start, self.range_end - self.config.episode_length)

    @property
    def index(self) -> int:
        """Current decision bar (the bar whose close is "now")."""
        return self.sim.index

    def action_to_forecast(self, action: Any) -> float:
        """Map an action of ``action_space`` to a forecast in [-1, 1]."""
        if self.config.action_mode == "discrete":
            a = int(np.asarray(action).reshape(-1)[0])
            if not 0 <= a < len(self._levels):
                raise ValueError(f"invalid discrete action {action!r}")
            return float(self._levels[a])
        x = float(np.asarray(action, dtype=float).reshape(-1)[0])
        if not math.isfinite(x):
            return 0.0
        return max(-1.0, min(1.0, x))

    def forecast_to_action(self, forecast: float) -> Any:
        """Inverse of :meth:`action_to_forecast` (nearest discrete level)."""
        if self.config.action_mode == "discrete":
            return int(np.argmin(np.abs(self._levels - float(forecast))))
        return np.array([max(-1.0, min(1.0, float(forecast)))], dtype=np.float32)

    def _full_lots(self, t: int, equity: float) -> float:
        """|lots| of a +1 forecast now (no drawdown scaling, from flat): the position unit."""
        lots = self.sizer.target_lots(1.0, self._vol[t], equity, self._close[t], self.instrument,
                                      current_lots=0.0, drawdown=0.0)
        return abs(float(lots))

    def _new_risk_manager(self) -> RiskManager | None:
        if self.risk_limits is None:
            return None
        return StandardRiskManager(self.risk_limits, self.instrument, events=self.events)

    def _state(self, t: int) -> np.ndarray:
        cfg = self.config
        sim = self.sim
        pos = sim.position
        eq = sim.equity
        full = self._full_lots(t, eq) if eq > 0 else 0.0
        pos_frac = pos / full if full > 0 else 0.0
        upnl = 0.0
        if pos != 0.0 and self._entry_ref > 0:
            daily_vol = self._vol[t] / math.sqrt(_TRADING_DAYS)
            upnl = _sign(pos) * math.log(self._close[t] / self._entry_ref) / max(daily_vol, 1e-6)
        tit = math.log1p(self._bars_in_trade) / math.log1p(cfg.trade_time_norm)
        return np.array([
            max(-1.5, min(1.5, pos_frac)),
            max(-5.0, min(5.0, upnl)),
            min(2.0, tit),
            min(5.0, 10.0 * sim.drawdown),
        ], dtype=np.float32)

    def observation_at(self, t: int) -> np.ndarray:
        """Observation for decision bar ``t`` with the CURRENT account state (``t`` must be
        the simulator's index for the state part to be meaningful)."""
        w = self.config.window
        if t < self.first_obs_index:
            raise IndexError(f"bar {t} is inside the feature warm-up (first observable "
                             f"{self.first_obs_index})")
        window = self._feat[t - w + 1: t + 1].reshape(-1)
        obs = np.concatenate([window, self._state(t)])
        clip = self.config.obs_clip
        np.clip(obs, -clip, clip, out=obs)
        return obs.astype(np.float32, copy=False)

    def fresh_observation(self, t: int) -> np.ndarray:
        """Observation at bar ``t`` of a NEW episode (flat, no drawdown) — what
        ``reset(options={"start": t})`` returns — without touching the simulator (also valid
        at ``range_end``, where no episode can be started)."""
        w = self.config.window
        if t < self.first_obs_index:
            raise IndexError(f"bar {t} is inside the feature warm-up (first observable "
                             f"{self.first_obs_index})")
        obs = np.concatenate([self._feat[t - w + 1: t + 1].reshape(-1),
                              np.zeros(_N_STATE, dtype=np.float32)])
        clip = self.config.obs_clip
        np.clip(obs, -clip, clip, out=obs)
        return obs.astype(np.float32, copy=False)

    def _update_trade_state(self, prev_pos: float, res: StepResult) -> None:
        """Track the average entry mid and bars held (for ``state_upnl_vol``/``time_in_trade``)."""
        p = res.position
        fill_mid = self._open[res.index]
        if p == 0.0:
            self._entry_ref = math.nan
            self._bars_in_trade = 0
        elif prev_pos == 0.0 or _sign(p) != _sign(prev_pos) or res.position_open != p:
            # New position (or re-entry after an intrabar exit): entry at the fill's mid.
            self._entry_ref = fill_mid
            self._bars_in_trade = 1
        else:
            if abs(p) > abs(prev_pos) + _EPS and math.isfinite(self._entry_ref):
                add = abs(p) - abs(prev_pos)
                self._entry_ref = (abs(prev_pos) * self._entry_ref + add * fill_mid) / abs(p)
            self._bars_in_trade += 1

    def _info(self, **extra: Any) -> dict[str, Any]:
        sim = self.sim
        t = sim.index
        info: dict[str, Any] = {
            "bar": t,
            "time": self._times[t],
            "decision_time": self._dtimes[t],
            "equity": sim.equity,
            "position": sim.position,
            "drawdown": sim.drawdown,
            "episode_start": self._episode_start,
            "steps": self._steps,
        }
        info.update(extra)
        return info

    # ---- gymnasium API ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode, flat, with ``config.initial_equity``.

        ``options={"start": i}`` forces the start bar (clamped into the eligible range).
        """
        super().reset(seed=seed)
        opts = options or {}
        if "start" in opts and opts["start"] is not None:
            s = int(opts["start"])
            s = min(max(s, self.range_start), self.range_end - 1)
        elif self.random_start:
            s = int(self.np_random.integers(self.range_start, self.max_start + 1))
        else:
            s = self.range_start
        self._episode_start = s
        self.sim.reset(start=s, equity=self.config.initial_equity)
        self.risk = self._new_risk_manager()
        self._episode_equity = float(self.config.initial_equity)
        self._steps = 0
        self._entry_ref = math.nan
        self._bars_in_trade = 0
        self._last_forecast = 0.0
        self._needs_reset = False
        return self.observation_at(s), self._info()

    def rebase_episode(self) -> np.ndarray:
        """Start a new episode IN PLACE at the current bar and return its observation.

        The simulator is NOT reset: equity, the open position (flat after a kill switch,
        which flattens) and the recorded money path continue. Everything the agent state
        and the risk limits measure *relative to the episode* is re-based as at a fresh
        training episode: the drawdown peak := current equity, a fresh risk manager (kill
        switch cleared, day-start equity re-based — what an operator's
        ``reset_halt(confirm="RESET")`` does), the ruin threshold := ``min_equity_frac`` x
        current equity, and the step counter.

        Used by :func:`rollout` (``restart_on_termination=True``) for the ``rl_ppo`` strategy
        rollout: training episodes END at a kill switch, so a policy never sees the deeper
        drawdowns a never-restarting multi-year account would reach. Re-basing keeps its
        observations inside the training state distribution (no train/serve skew) while
        forecasts keep flowing after a simulated kill.
        """
        sim = self.sim
        if sim.bankrupt or not sim.equity > 0:
            raise RuntimeError("cannot re-base a bankrupt account")
        sim.peak_equity = sim.equity
        self.risk = self._new_risk_manager()
        self._episode_equity = float(sim.equity)
        self._episode_start = sim.index
        self._steps = 0
        self._needs_reset = False
        return self.observation_at(sim.index)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Decide at the close of bar ``t``; fill at the open of ``t+1``; reward over ``t+1``."""
        if self._needs_reset:
            raise RuntimeError("call reset() before step() (episode finished)")
        cfg = self.config
        sim = self.sim
        inst = self.instrument
        t = sim.index
        forecast = self.action_to_forecast(action)
        eq = sim.equity
        cur = sim.position
        dd_before = sim.drawdown
        full_before = self._full_lots(t, eq)

        # Sizer -> risk manager -> simulator: the run_backtest chain, bar for bar.
        requested = inst.round_lots(float(self.sizer.target_lots(
            forecast, self._vol[t], eq, self._close[t], inst, current_lots=cur,
            drawdown=dd_before)))
        approved = requested
        halted = False
        reasons: list[str] = []
        risk = self.risk
        if risk is not None:
            ctx = RiskContext(time=self._dtimes[t], equity=eq, current_lots=cur,
                              target_lots=requested, price=self._close[t], spread=self._spread[t],
                              vol_ann=self._vol[t], bar_index=t,
                              extra={"drawdown": dd_before, "bar_time": self._times[t],
                                     "forecast": forecast})
            with simulated_risk_logging():
                risk.on_bar(self._dtimes[t], eq)
                decision = risk.evaluate(ctx)
            halted = bool(decision.halted)
            reasons = [str(r) for r in decision.reasons]
            approved = 0.0 if halted else float(decision.approved_lots)
            lo_b, hi_b = min(0.0, requested), max(0.0, requested)
            if not math.isfinite(approved):
                approved = 0.0
            approved = inst.round_lots(min(max(approved, lo_b), hi_b))  # reduce-only
        changed = abs(approved - requested) > _EPS
        risk_driven = halted or (changed and abs(approved) < abs(cur) - _EPS
                                 and _sign(requested) == _sign(cur))
        res = sim.step(approved, reason="risk" if risk_driven else "signal")
        self._update_trade_state(cur, res)
        self._steps += 1
        self._last_forecast = forecast

        # Reward (shaping never touches the simulator's equity).
        new_eq = res.equity
        floor = cfg.initial_equity * 1e-9
        log_ret = math.log(max(new_eq, floor) / eq)
        dd_after = sim.drawdown
        dd_inc = max(0.0, dd_after - dd_before)
        turnover = abs(approved - cur) / full_before if full_before > 0 else 0.0
        r_growth = cfg.reward_scale * log_ret
        r_dd = cfg.dd_penalty * cfg.reward_scale * dd_inc
        r_turn = cfg.turnover_penalty * turnover
        reward = r_growth - r_dd - r_turn

        # Episode end.
        ruined = sim.bankrupt or new_eq <= cfg.min_equity_frac * self._episode_equity
        hard_halt = False
        if risk is not None and cfg.terminate_on_halt and bool(getattr(risk, "halted", False)):
            kind = getattr(getattr(risk, "state", None), "halt_kind", None)
            transient = (kind == "daily_loss" and self.risk_limits is not None
                         and not self.risk_limits.daily_loss_persistent)
            hard_halt = not transient
        terminated = bool(ruined or hard_halt)
        at_end = sim.index >= self.range_end
        truncated = bool(not terminated and (
            at_end or (self.random_start and self._steps >= cfg.episode_length)))
        if terminated or truncated:
            self._needs_reset = True

        obs = self.observation_at(sim.index)
        info = self._info(
            decision_bar=t,
            forecast=forecast,
            requested_lots=requested,
            approved_lots=approved,
            position_open=res.position_open,
            pnl=res.pnl,
            price_pnl=res.price_pnl,
            costs=dict(res.costs),
            log_return=log_ret,
            reward_components={"growth": r_growth, "drawdown": -r_dd, "turnover": -r_turn},
            halted=halted,
            risk_reasons=reasons,
            ruined=bool(ruined),
            hard_halt=hard_halt,
        )
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:  # pragma: no cover - no rendering
        return None


# ---------------------------------------------------------------------------------------------
# deterministic rollouts (validation, strategy adapter)
# ---------------------------------------------------------------------------------------------
@dataclass
class RolloutResult:
    """Outcome of :func:`rollout` over an evaluation range.

    forecast : forecast decided at each bar's close, indexed like the env's bars (0 outside
               the rollout and after an early termination). The value at the rollout's last
               bar is the policy's decision there (not executed: no next bar in range).
    requested_lots / approved_lots : per decision bar (NaN elsewhere).
    rewards  : shaped reward per decision bar (NaN elsewhere).
    result   : the simulator's :class:`BacktestResult` (true equity/costs); continuous across
               in-place restarts, but it covers only the LAST episode when ``episode_starts``
               began fresh episodes (``info["episode_starts"]``).
    terminated : whether the rollout ended early (ruin or kill switch, without restart).
    n_halts  : decision bars at which the risk manager reported ``halted``.
    n_restarts : in-place episode restarts (``restart_on_termination=True``); the bars at
               which they happened are in ``info["restart_bars"]``.
    """

    forecast: pd.Series
    requested_lots: pd.Series
    approved_lots: pd.Series
    rewards: pd.Series
    result: BacktestResult
    start: int
    end: int
    terminated: bool
    n_halts: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    n_restarts: int = 0


def rollout(
    env: GoldTradingEnv,
    policy: Callable[[np.ndarray], Any],
    *,
    start: int | None = None,
    compute_metrics: bool = True,
    restart_on_termination: bool = False,
    episode_starts: np.ndarray | None = None,
) -> RolloutResult:
    """Drive ``policy(obs) -> action`` bar by bar from ``start`` (default: the env's first
    eligible bar) to ``env.range_end`` with the env's OWN simulated position, so the
    agent-state part of every observation is exactly what it was during training.

    The env should be built with ``random_start=False`` (episode length is then the whole
    range). Returns forecasts aligned to ``env.bars.index``.

    ``restart_on_termination=False`` (evaluation): a kill switch or ruin ends the rollout,
    exactly like ``run_backtest`` with a persistent kill (forecasts are 0 afterwards).
    ``True`` (strategy adapter): the kill still fires (and flattens) as in training, then
    :meth:`GoldTradingEnv.rebase_episode` starts a new episode in place and the rollout
    continues — the policy keeps producing forecasts from states it was trained on. Only a
    bankrupt account still ends the rollout.

    ``episode_starts`` (boolean per bar of ``env.bars``, e.g. calendar anchors from
    :func:`aurum.rl.episode_anchor_mask`): at every flagged bar after the rollout start a
    FRESH episode begins (``env.reset(options={"start": t})``: flat, initial equity, new
    peak and risk manager), exactly like a training episode. The agent state after an
    anchor then depends only on bars since that anchor, so forecasts no longer depend on
    where the history window starts (live sliding windows reproduce backtests).
    """
    if env.random_start:
        raise ValueError("rollout needs an evaluation env (random_start=False)")
    n = len(env.bars)
    starts = None
    if episode_starts is not None:
        starts = np.asarray(episode_starts, dtype=bool)
        if starts.shape != (n,):
            raise ValueError(f"episode_starts must have one flag per bar ({n}), got {starts.shape}")
    fc = np.zeros(n)
    req = np.full(n, np.nan)
    app = np.full(n, np.nan)
    rew = np.full(n, np.nan)
    obs, _ = env.reset(options={"start": start} if start is not None else None)
    s = env.episode_start
    terminated = False
    n_halts = 0
    restart_bars: list[int] = []
    fresh_bars: list[int] = []
    while True:
        t = env.index
        if starts is not None and t > s and starts[t]:
            obs = (env.reset(options={"start": t})[0] if t < env.range_end
                   else env.fresh_observation(t))
            fresh_bars.append(t)
        action = policy(obs)
        fc[t] = env.action_to_forecast(action)
        if t >= env.range_end:
            break
        obs, r, terminated, truncated, info = env.step(action)
        req[t] = info["requested_lots"]
        app[t] = info["approved_lots"]
        rew[t] = r
        n_halts += int(bool(info["halted"]))
        if terminated and restart_on_termination and not env.sim.bankrupt:
            logger.debug("rollout: episode ended at bar %d (ruined=%s hard_halt=%s); "
                         "restarting in place", env.index, info["ruined"], info["hard_halt"])
            restart_bars.append(env.index)
            obs = env.rebase_episode()
            terminated = False
            continue
        if terminated:
            logger.info("rollout terminated early at bar %d (%s): ruined=%s hard_halt=%s",
                        env.index, env.sim.time, info["ruined"], info["hard_halt"])
            break
        if truncated:
            t = env.index  # final bar of the range: record the (unexecuted) decision
            fc[t] = env.action_to_forecast(policy(obs))
            break
    idx = env.bars.index
    res = env.sim.result(compute_metrics=compute_metrics)
    return RolloutResult(
        forecast=pd.Series(fc, index=idx, name="forecast"),
        requested_lots=pd.Series(req, index=idx, name="requested_lots"),
        approved_lots=pd.Series(app, index=idx, name="approved_lots"),
        rewards=pd.Series(rew, index=idx, name="reward"),
        result=res,
        start=s,
        end=env.index,
        terminated=bool(terminated),
        n_halts=n_halts,
        info={"restart_bars": restart_bars, "episode_starts": fresh_bars},
        n_restarts=len(restart_bars),
    )
