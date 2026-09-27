"""Typed run configuration: YAML -> a tree of dataclasses, validated, hashable.

One :class:`AurumConfig` drives every entry point (research backtests, walk-forward,
LLM-desk replay, RL training and the live runner), so the *same* cost model, sizer and
risk limits are used everywhere (SPEC §0.2) and every run can record exactly what it did
(SPEC §0.4: config hash + data hash + git SHA next to the results).

Loading
-------
``load_config(path)`` reads YAML (optionally ``extends: other.yaml`` for inheritance, deep
merged, lists replaced wholesale), applies ``--set a.b=value`` style overrides, then builds
the dataclass tree with strict checking:

* unknown keys raise :class:`ConfigError` with the valid keys and a "did you mean" hint;
* scalar types are checked (YAML dates are accepted where a string is expected);
* semantic validation builds the REAL downstream objects (``Instrument``, ``CostModel``,
  ``VolTargetSizer``, ``RiskLimits``, ``ForecastCombiner``, ``DecisionPolicy``,
  ``DeskConfig``) so that a config accepted here cannot fail later on a parameter range,
  and every problem is reported at once.

Secrets
-------
Secrets never live in YAML. They are read ONLY from environment variables
(:data:`SECRET_ENV`), wrapped in :class:`Secret` (masked ``repr``), excluded from
``to_dict``/hash/saved configs. A YAML file that sets a secret is rejected. No other
field can be overridden from the environment, so a run is fully described by its YAML +
``--set`` overrides.

Durations
---------
Walk-forward windows may be integers (bars) or durations: ``"3Y"``, ``"6M"`` (months),
``"2W"``, ``"10D"``, ``"12h"``, ``"90min"``, ``"500bars"``. :func:`duration_to_bars` turns
them into bar counts from the sample's calendar density (bars per calendar day), which
is a property of the trading calendar, not of prices, so it cannot leak returns.

Hashing
-------
:meth:`AurumConfig.config_hash` is the SHA-256 of the canonical JSON of every field that
can change results (secrets, the output section, the run name and pure execution knobs
such as ``walkforward.n_jobs`` are excluded).
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import difflib
import hashlib
import json
import logging
import math
import os
import re
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from aurum.core.instrument import XAUUSD
from aurum.core.timeframes import get_timeframe
from aurum.core.types import DEFAULT_MAGIC

logger = logging.getLogger(__name__)

__all__ = [
    "AgentsConfig",
    "AurumConfig",
    "BacktestConfig",
    "CombinerConfig",
    "ConfigError",
    "CostsConfig",
    "DataConfig",
    "FeaturesConfig",
    "FinancingConfig",
    "InstrumentConfig",
    "LiveConfig",
    "OutputConfig",
    "PolicyConfig",
    "RESEARCH_RISK_DEFAULTS",
    "RLConfig",
    "RiskConfig",
    "SECRET_ENV",
    "Secret",
    "SizingConfig",
    "StrategyConfig",
    "SyntheticDataConfig",
    "WalkForwardConfig",
    "apply_overrides",
    "config_from_dict",
    "duration_to_bars",
    "load_config",
    "parse_duration",
]


class ConfigError(ValueError):
    """Invalid configuration. ``problems`` lists every issue found (path: message)."""

    def __init__(self, problems: str | Sequence[str]) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        msg = self.problems[0] if len(self.problems) == 1 else (
            f"{len(self.problems)} configuration problems:\n  - " + "\n  - ".join(self.problems))
        super().__init__(msg)


class Secret:
    """A credential read from the environment; ``repr``/``str`` never reveal it."""

    __slots__ = ("_value", "source")

    def __init__(self, value: str, source: str = "") -> None:
        self._value = str(value)
        self.source = source

    def get(self) -> str:
        """The secret value (only call this where the credential is actually used)."""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return f"Secret(<{'set' if self._value else 'empty'}{' from $' + self.source if self.source else ''}>)"

    __str__ = __repr__


#: dotted config path -> environment variable. The ONLY environment overrides that exist.
SECRET_ENV: dict[str, str] = {
    "agents.api_key": "ANTHROPIC_API_KEY",
    "live.mt5_password": "MT5_PASSWORD",            # same names as aurum.live.mt5 / monitor / runner
    "live.alert_webhook": "AURUM_ALERT_WEBHOOK_URL",
    "live.artifact_key": "AURUM_ARTIFACT_KEY",
}

_NOHASH = {"hash": False}          # field metadata: excluded from config_hash
_SECRET = {"hash": False, "secret": True}

#: Research-run risk limits layered over :class:`aurum.risk.manager.RiskLimits` defaults.
#: A research backtest measures the *strategy*: a single -3% day must not flatten the book for
#: the rest of a 14-year sample (``daily_loss_persistent=False``: the daily-loss halt clears
#: the next day), and the permanent max-drawdown kill switch — an operational control for
#: live money — is off; the sizer's drawdown de-risking still applies and every halt is
#: reported. Live limits keep the conservative RiskLimits defaults.
RESEARCH_RISK_DEFAULTS: dict[str, Any] = {"daily_loss_persistent": False, "max_drawdown": None}


# ----------------------------------------------------------------------------------------
# durations
# ----------------------------------------------------------------------------------------
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*([A-Za-z]*)\s*$")
_UNIT_DAYS: dict[str, float] = {
    "y": 365.25, "yr": 365.25, "year": 365.25, "years": 365.25,
    "M": 365.25 / 12.0, "mo": 365.25 / 12.0, "month": 365.25 / 12.0, "months": 365.25 / 12.0,
    "w": 7.0, "week": 7.0, "weeks": 7.0,
    "d": 1.0, "day": 1.0, "days": 1.0,
    "h": 1.0 / 24.0, "hour": 1.0 / 24.0, "hours": 1.0 / 24.0,
    "min": 1.0 / 1440.0, "mins": 1.0 / 1440.0, "minute": 1.0 / 1440.0, "minutes": 1.0 / 1440.0,
}
_BAR_UNITS = {"", "b", "bar", "bars"}


def parse_duration(value: int | float | str) -> tuple[float, str]:
    """Parse ``500`` / ``"500bars"`` -> ``(500, "bars")`` or ``"3Y"`` -> ``(1095.75, "days")``.

    ``M`` (upper case) is months; minutes are ``min``. Other units are case-insensitive.
    """
    if isinstance(value, bool):
        raise ValueError(f"invalid duration {value!r}")
    if isinstance(value, (int, float)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"duration must be positive, got {value!r}")
        return float(value), "bars"
    m = _DURATION_RE.match(str(value))
    if not m:
        raise ValueError(f"invalid duration {value!r} (examples: 500, '500bars', '3Y', '6M', '2W', '10D', '12h')")
    num, unit = float(m.group(1)), m.group(2)
    if num <= 0:
        raise ValueError(f"duration must be positive, got {value!r}")
    if unit.lower() in _BAR_UNITS:
        return num, "bars"
    key = unit if unit == "M" else unit.lower()
    if key not in _UNIT_DAYS:
        raise ValueError(f"unknown duration unit {unit!r} in {value!r} (Y, M, W, D, h, min or bars)")
    return num * _UNIT_DAYS[key], "days"


def duration_to_bars(value: int | float | str, bars_per_day: float) -> int:
    """Convert a duration to a bar count using the sample's bars per CALENDAR day.

    ``bars_per_day`` is calendar density (e.g. ``(n - 1) / span_days`` of the bar index), so
    ``"1Y"`` of H1 gold bars is ~5,900 bars (weekends and the daily break included in the
    calendar span but not in the count).
    """
    amount, kind = parse_duration(value)
    if kind == "bars":
        n = int(round(amount))
    else:
        if not (math.isfinite(bars_per_day) and bars_per_day > 0):
            raise ValueError(f"bars_per_day must be positive, got {bars_per_day}")
        n = int(round(amount * bars_per_day))
    return max(n, 1)


# ----------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------
@dataclass
class SyntheticDataConfig:
    """Generate bars with :func:`aurum.data.synthetic.make_synthetic_bars` instead of loading
    files (smoke tests, demos). ``model``: gbm | trend | mean_revert | regime | jump."""

    n: int = 5000
    model: str = "gbm"
    seed: int = 0
    start: str = "2020-01-06"
    annual_vol: float = 0.16
    drift: float = 0.0
    spread: float = 0.30
    start_price: float = 1800.0
    regime_params: dict[str, Any] | None = None
    macro: bool = True
    events: bool = True


@dataclass
class DataConfig:
    """Where the bars / macro / calendar come from. Relative paths resolve against the
    current working directory (run the CLI from the repository root)."""

    dir: str = "data_store"
    symbol: str = "XAUUSD"
    timeframe: str = "H1"
    bars_path: str | None = None          # default: {dir}/{symbol.lower()}_{timeframe}.parquet
    macro: bool = True
    macro_dir: str | None = None          # default: {dir}/macro
    events: str = "rule_based"            # "rule_based" | "none" | path to a calendar CSV
    start: str | None = None              # restrict LOADED data (inclusive, UTC)
    end: str | None = None
    verify_hash: bool = True
    synthetic: SyntheticDataConfig | None = None

    def resolved_bars_path(self) -> Path:
        if self.bars_path:
            return Path(self.bars_path)
        return Path(self.dir) / f"{self.symbol.lower()}_{self.timeframe.upper()}.parquet"

    def resolved_macro_dir(self) -> Path:
        return Path(self.macro_dir) if self.macro_dir else Path(self.dir) / "macro"

    def load(self) -> Any:
        """Load :class:`aurum.core.types.MarketData` (bars + macro + events) point-in-time.

        Macro frames and the calendar are kept whole: every consumer aligns them on each
        bar's ``available_at`` (``asof_join``), so rows published after a decision are never
        visible to it. Calendar rows are scheduled times, public in advance (SPEC §3.5).
        """
        import pandas as pd

        from aurum.core.types import MarketData

        if self.synthetic is not None:
            from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro

            s = self.synthetic
            bars = make_synthetic_bars(s.n, self.timeframe, seed=s.seed, model=s.model, start=s.start,
                                       annual_vol=s.annual_vol, drift=s.drift, spread=s.spread,
                                       start_price=s.start_price, regime_params=s.regime_params)
            bars = _restrict(bars, self.start, self.end)
            macro = make_synthetic_macro(bars, seed=s.seed) if (s.macro and self.macro) else {}
            events = None
            if s.events and self.events != "none":
                events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7))
            return MarketData(bars=bars, macro=macro, events=events)

        from aurum.data.store import load_bars

        path = self.resolved_bars_path()
        if not path.exists():
            raise FileNotFoundError(
                f"bars file {path} not found (run `aurum data download` or set data.bars_path)")
        bars = _restrict(load_bars(path, verify_hash=self.verify_hash), self.start, self.end)
        if len(bars) < 2:
            raise ValueError(f"{path}: fewer than two bars in [{self.start}, {self.end}]")
        macro: dict = {}
        if self.macro:
            mdir = self.resolved_macro_dir()
            if mdir.exists():
                from aurum.data.macro import load_macro_dir

                macro = load_macro_dir(mdir)
            else:
                logger.warning("macro directory %s not found: running without macro data", mdir)
        events = None
        if self.events == "rule_based":
            from aurum.data.calendar import generate_rule_based_calendar

            events = generate_rule_based_calendar(bars.index[0].normalize(),
                                                  bars.index[-1] + pd.Timedelta(days=14))
        elif self.events != "none":
            from aurum.data.calendar import load_calendar_csv

            events = load_calendar_csv(self.events)
        return MarketData(bars=bars, macro=macro, events=events)


def _restrict(bars: Any, start: str | None, end: str | None) -> Any:
    import pandas as pd

    def _ts(x: str) -> pd.Timestamp:
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

    if start:
        bars = bars.loc[bars.index >= _ts(start)]
    if end:
        e = _ts(end)
        # a date-only end is inclusive of that whole day
        if isinstance(end, str) and len(end.strip()) <= 10:
            e = e + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        bars = bars.loc[bars.index <= e]
    return bars


@dataclass
class InstrumentConfig:
    """Contract specification (see :class:`aurum.core.instrument.Instrument`)."""

    symbol: str = XAUUSD.symbol
    contract_size: float = XAUUSD.contract_size
    tick_size: float = XAUUSD.tick_size
    lot_step: float = XAUUSD.lot_step
    min_lot: float = XAUUSD.min_lot
    max_lot: float = XAUUSD.max_lot
    margin_rate: float = XAUUSD.margin_rate
    commission_per_lot: float = XAUUSD.commission_per_lot
    swap_long_per_lot: float = XAUUSD.swap_long_per_lot     # used by costs.financing.mode: fixed
    swap_short_per_lot: float = XAUUSD.swap_short_per_lot
    triple_swap_weekday: int = XAUUSD.triple_swap_weekday
    rollover_hour_utc: int = XAUUSD.rollover_hour_utc

    def build(self) -> Any:
        from aurum.core.instrument import Instrument

        return Instrument(**dataclasses.asdict(self))


@dataclass
class FinancingConfig:
    """Overnight financing (see :class:`aurum.execution.costs.FinancingModel`).

    ``mode: rate`` (default) charges the benchmark ``rate_series`` from the macro data
    (``fedfunds`` = FRED DFF, percent), read point-in-time at each rollover, plus
    ``markup_long`` / minus ``markup_short`` (annual fractions; ~2-3% at retail CFD brokers)
    on the position's notional, ``/day_count`` per night (triple on the instrument's
    ``triple_swap_weekday``); ``lease_rate`` is the gold lease rate earned by longs.
    ``fallback_rate`` applies before the series starts or without macro data.
    ``mode: fixed`` uses the broker-quoted ``instrument.swap_{long,short}_per_lot``;
    ``mode: none`` disables financing.
    """

    mode: str = "rate"
    markup_long: float = 0.025
    markup_short: float = 0.025
    lease_rate: float = 0.0
    rate_series: str = "fedfunds"
    rate_unit: str = "percent"
    fallback_rate: float = 0.03
    day_count: float = 360.0


@dataclass
class CostsConfig:
    """Execution cost model (see :class:`aurum.execution.costs.CostModel`)."""

    spread_multiplier: float = 1.0
    min_spread: float = 0.10
    slippage_fixed: float = 0.02
    slippage_range_frac: float = 0.02
    impact_coef: float = 0.0
    commission_per_lot: float | None = None
    financing: FinancingConfig = field(default_factory=FinancingConfig)

    def build(self) -> Any:
        from aurum.execution.costs import CostModel

        return CostModel(**dataclasses.asdict(self))


@dataclass
class FeaturesConfig:
    """:class:`aurum.features.pipeline.FeaturePipeline` settings.

    ``enabled``: ``"auto"`` computes features only when a strategy needs them (a trainable
    strategy, or one declaring ``uses_features = True``); ``true``/``false`` force it.
    """

    enabled: bool | str = "auto"
    groups: list[str] | None = None
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    scaler: str = "robust"
    clip: float | None = 5.0
    warmup: int | None = None

    def build(self) -> Any:
        from aurum.features.pipeline import FeaturePipeline

        return FeaturePipeline(groups=self.groups, overrides=copy.deepcopy(self.overrides),
                               scaler=self.scaler, clip=self.clip, warmup=self.warmup)


@dataclass
class StrategyConfig:
    """One strategy configuration. ``id`` (default ``name``) must be unique: it names the
    forecast column, so the same strategy can appear twice with different params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float | None = None           # only for combiner.method == "fixed"
    id: str | None = None
    enabled: bool = True

    @property
    def key(self) -> str:
        return self.id or self.name


@dataclass
class CombinerConfig:
    """:class:`aurum.portfolio.combiner.ForecastCombiner` settings. ``method="fixed"`` uses
    the strategies' ``weight`` values (normalised) with the same FDM logic.

    ``allow_unallocated`` (default true): strategies with a non-positive NET Sharpe get no
    weight and the ``max_weight`` cap never forces weight onto them, so the weights may sum
    to < 1 (less risk); false restores the sum-to-1 behaviour. ``cost_multiplier`` scales the
    combiner's estimated turnover cost (1 = the configured cost model, 0 = score gross).
    """

    method: str = "sharpe_shrink"
    shrinkage: float = 0.5
    max_weight: float = 0.4
    fdm_cap: float = 2.5
    vol_halflife: float = 48.0
    min_periods: int = 20
    corr_floor: float = 0.0
    allow_unallocated: bool = True
    cost_multiplier: float = 1.0

    def build(self) -> Any:
        from aurum.portfolio.combiner import ForecastCombiner

        method = "equal" if self.method == "fixed" else self.method
        return ForecastCombiner(method=method, shrinkage=self.shrinkage, max_weight=self.max_weight,
                                fdm_cap=self.fdm_cap, vol_halflife=self.vol_halflife,
                                min_periods=self.min_periods, corr_floor=self.corr_floor,
                                allow_unallocated=self.allow_unallocated,
                                cost_multiplier=self.cost_multiplier)


@dataclass
class SizingConfig:
    """Position sizer: ``vol_target`` (:class:`VolTargetSizer`, default) or
    ``fixed_fractional`` (:class:`FixedFractionalSizer`)."""

    method: str = "vol_target"
    target_vol: float = 0.10
    max_leverage: float = 2.0
    max_lots: float | None = None
    rebalance_band: float = 0.10
    kelly_cap: float | None = None
    drawdown_derisk: list[list[float]] | None = field(default_factory=lambda: [[0.10, 0.5], [0.15, 0.25]])
    min_vol: float = 0.02
    risk_per_trade: float = 0.005          # fixed_fractional only
    stop_atr: float = 2.0                  # fixed_fractional only

    def build(self) -> Any:
        from aurum.portfolio.sizing import FixedFractionalSizer, VolTargetSizer

        dd = None if self.drawdown_derisk is None else tuple(tuple(float(x) for x in s) for s in self.drawdown_derisk)
        if self.method == "vol_target":
            return VolTargetSizer(target_vol=self.target_vol, max_leverage=self.max_leverage,
                                  max_lots=self.max_lots, rebalance_band=self.rebalance_band,
                                  kelly_cap=self.kelly_cap, drawdown_derisk=dd, min_vol=self.min_vol)
        if self.method == "fixed_fractional":
            return FixedFractionalSizer(risk_per_trade=self.risk_per_trade, stop_atr=self.stop_atr,
                                        max_leverage=self.max_leverage, max_lots=self.max_lots,
                                        rebalance_band=self.rebalance_band, drawdown_derisk=dd)
        raise ConfigError(f"sizing.method: unknown {self.method!r} (vol_target | fixed_fractional)")


@dataclass
class RiskConfig:
    """Risk limits for research runs and for live trading (keyword arguments of
    :class:`aurum.risk.manager.RiskLimits`). ``research`` is layered over
    :data:`RESEARCH_RISK_DEFAULTS`; ``live`` over the RiskLimits defaults (conservative)."""

    enabled: bool = True
    research: dict[str, Any] = field(default_factory=dict)
    live: dict[str, Any] = field(default_factory=dict)

    def limits(self, mode: str = "research") -> Any:
        from aurum.risk.manager import RiskLimits

        if mode == "research":
            kwargs = {**RESEARCH_RISK_DEFAULTS, **self.research}
        elif mode == "live":
            kwargs = dict(self.live)
        else:
            raise ValueError(f"risk mode must be 'research' or 'live', got {mode!r}")
        kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
        return RiskLimits(**kwargs)

    def build(self, mode: str = "research", *, instrument: Any = None, events: Any = None,
              state_path: str | os.PathLike | None = None) -> Any:
        """A fresh :class:`StandardRiskManager` (``None`` when ``enabled`` is false)."""
        if not self.enabled:
            return None
        from aurum.core.instrument import XAUUSD as _X
        from aurum.risk.manager import StandardRiskManager

        return StandardRiskManager(self.limits(mode), instrument or _X, state_path, events=events)


@dataclass
class BacktestConfig:
    """Backtest engine settings (:func:`aurum.backtest.engine.run_backtest`)."""

    initial_equity: float = 100_000.0
    stop_atr_mult: float | None = None
    take_profit_atr_mult: float | None = None
    atr_period: int = 14
    stop_cooldown_bars: int = 0
    event_horizon_hours: float = 24.0
    start: str | None = None               # trading window (inputs are warmed up before it)
    end: str | None = None
    benchmark: bool = True


@dataclass
class WalkForwardConfig:
    """Walk-forward protocol (see :mod:`aurum.research.walkforward`).

    ``train``/``test``/``step``/``embargo`` are bars (int) or durations ("3Y", "6M", ...).
    ``purge``: bars or ``"auto"`` = the largest label horizon declared by a trainable
    strategy (``label_horizon`` attribute or a ``horizon``-like parameter); the effective
    purge is never below that horizon. ``holdout_start``: everything from this timestamp on
    is excluded from every fold and evaluated ONCE at the end.
    ``combiner_fit``: ``"oos"`` (default) fits fold k's combiner on the stitched
    out-of-sample forecasts of folds < k (last ``train`` bars, or all when ``anchored``),
    with equal weights until ``combiner_min_obs`` OOS bars exist; ``"train"`` fits it on the
    fold's training-window forecasts, which are IN-SAMPLE for trainable strategies.
    """

    train: int | str = "3Y"
    test: int | str = "6M"
    step: int | str | None = None
    anchored: bool = False
    purge: int | str = "auto"
    embargo: int | str = 0
    holdout_start: str | None = None
    min_train_bars: int = 500
    min_test_bars: int = 20
    history_bars: int | None = None        # cap on history before train start given to trainable strategies
    regenerate_per_fold: bool = False      # also re-generate NON-trainable strategies per fold
    combiner_fit: str = "oos"              # "oos": weights from EARLIER folds' OOS forecasts | "train"
    combiner_min_obs: int = 500            # OOS bars needed before "oos" weights replace equal weights
    n_trials: int | None = None            # DSR trials; default = number of strategy configs
    pbo_splits: int = 16
    n_boot: int = 1000
    strategy_backtests: bool = True
    on_strategy_error: str = "raise"       # "raise" | "drop"
    n_jobs: int = field(default=0, metadata=_NOHASH)        # 0 = auto (cpu count)
    executor: str = field(default="auto", metadata=_NOHASH)  # auto | process | thread | serial


@dataclass
class PolicyConfig:
    """:class:`aurum.agents.policy.DecisionPolicy` settings."""

    mode: str = "overlay"
    max_abs_forecast: float = 1.0
    on_failure: str = "follow_quant"
    min_confidence: float = 0.0

    def build(self) -> Any:
        from aurum.agents.policy import DecisionPolicy

        return DecisionPolicy(mode=self.mode, max_abs_forecast=self.max_abs_forecast,
                              on_failure=self.on_failure, min_confidence=self.min_confidence)


@dataclass
class AgentsConfig:
    """LLM trading desk (SPEC §10). ``desk`` holds :class:`aurum.agents.config.DeskConfig`
    keyword arguments (``chief``/``specialist``/``role_models`` values are
    :class:`AgentModelConfig` kwargs; ``prices`` maps model -> {input_per_mtok,
    output_per_mtok[, cache_read_multiplier]}). ``api_key`` comes from $ANTHROPIC_API_KEY."""

    enabled: bool = False
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    desk: dict[str, Any] = field(default_factory=dict)
    journal_dir: str = field(default="runs/desk_journal", metadata=_NOHASH)
    anonymise: bool = True
    lookback_bars: int = 250
    replay_every: int = 24
    replay_max_cost_usd: float = 25.0
    expected_cost_per_cycle_usd: float = 0.60   # planning estimate shown before a replay
    api_key: Secret | None = field(default=None, metadata=_SECRET)

    def desk_config(self) -> Any:
        from aurum.agents.config import AgentModelConfig, DeskConfig, ModelPrice

        kw = copy.deepcopy(self.desk)
        for role in ("chief", "specialist"):
            if isinstance(kw.get(role), Mapping):
                kw[role] = AgentModelConfig(**kw[role])
        if isinstance(kw.get("role_models"), Mapping):
            kw["role_models"] = {k: AgentModelConfig(**v) if isinstance(v, Mapping) else v
                                 for k, v in kw["role_models"].items()}
        if isinstance(kw.get("prices"), Mapping):
            from aurum.agents.config import DEFAULT_PRICES

            prices = dict(DEFAULT_PRICES)
            prices.update({k: ModelPrice(**v) if isinstance(v, Mapping) else v for k, v in kw["prices"].items()})
            kw["prices"] = prices
        if isinstance(kw.get("adhoc_tool_whitelist"), list):
            kw["adhoc_tool_whitelist"] = tuple(kw["adhoc_tool_whitelist"])
        return DeskConfig(**kw)


@dataclass
class LiveConfig:
    """Live / paper trading, run by :mod:`aurum.live.runner` (``aurum live run``).

    Real-money trading needs ``allow_live_real: true`` here AND ``--i-understand-real-money``
    on the CLI; ``dry_run`` (default) only plans and logs orders. ``artifact_dir`` is the
    trading artifact written by ``aurum live artifact`` (fitted strategies + combiner).
    ``options`` holds any other :class:`aurum.live.runner.LiveConfig` keys and sections
    (``paper``, ``mt5``, ``monitor``, ``oms``, ``bar_close_delay_seconds``, ...), validated
    by the runner. MT5 credentials are read by the broker from ``MT5_LOGIN``,
    ``MT5_PASSWORD``, ``MT5_SERVER`` and ``MT5_PATH``; the alert webhook from
    ``AURUM_ALERT_WEBHOOK_URL``; the artifact HMAC key from ``AURUM_ARTIFACT_KEY``.
    """

    broker: str = "paper"                  # "paper" | "mt5"
    dry_run: bool = True
    allow_live_real: bool = False
    magic: int = DEFAULT_MAGIC
    poll_seconds: float = 30.0             # retry poll while the market is closed
    symbol: str | None = None              # broker symbol (default: instrument.symbol)
    history_bars: int | None = None        # None: runner uses 3x the artifact's lookback
    state_dir: str = "runs/live"
    artifact_dir: str | None = "artifacts/live"
    use_desk: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    mt5_password: Secret | None = field(default=None, metadata=_SECRET)
    alert_webhook: Secret | None = field(default=None, metadata=_SECRET)
    artifact_key: Secret | None = field(default=None, metadata=_SECRET)


@dataclass
class RLConfig:
    """``aurum rl train``: PPO on :mod:`aurum.rl` (train / validation split by date).

    ``params`` are :class:`aurum.rl.train.RLTrainConfig` keyword arguments, validated when
    the command runs (so loading a config never imports gymnasium/torch).
    """

    train_start: str | None = None
    train_end: str = "2019-12-31"
    val_end: str = "2021-12-31"
    out_dir: str = "runs/rl/ppo"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Where run artefacts go: ``{dir}/{run_name or <command>_<utc>_<hash8>}/``."""

    dir: str = "runs"
    run_name: str | None = None
    save_results: bool = True
    tearsheet: bool = True
    dark_charts: bool = True


@dataclass
class AurumConfig:
    """Root of the configuration tree."""

    name: str = field(default="aurum", metadata=_NOHASH)
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    strategies: list[StrategyConfig] = field(default_factory=list)
    combiner: CombinerConfig = field(default_factory=CombinerConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    output: OutputConfig = field(default_factory=OutputConfig, metadata=_NOHASH)
    source: str | None = field(default=None, metadata=_NOHASH)   # file it was loaded from

    # ---- convenience ----------------------------------------------------------------
    def enabled_strategies(self, only: Sequence[str] | None = None) -> list[StrategyConfig]:
        """Enabled strategy configs, optionally restricted to ids/names in ``only``."""
        out = [s for s in self.strategies if s.enabled]
        if only:
            wanted = set(only)
            unknown = wanted - {s.key for s in self.strategies} - {s.name for s in self.strategies}
            if unknown:
                raise ConfigError(f"--strategy {sorted(unknown)} not in config "
                                  f"(configured: {[s.key for s in self.strategies]})")
            out = [s for s in self.strategies if s.key in wanted or s.name in wanted]
        return out

    def build_strategies(self, only: Sequence[str] | None = None) -> dict[str, Any]:
        """Instantiate enabled strategies from the registry: ``{id: Strategy}``.

        A strategy whose ``default_params`` has a ``seed`` gets ``self.seed`` unless set.
        """
        from aurum.strategies.base import get_strategy, list_strategies

        registry = list_strategies()
        out: dict[str, Any] = {}
        problems = []
        for i, s in enumerate(self.enabled_strategies(only)):
            cls = registry.get(s.name)
            if cls is None:
                close = difflib.get_close_matches(s.name, list(registry), n=3)
                hint = f"; did you mean {close}?" if close else ""
                problems.append(f"strategies[{i}].name: unknown strategy {s.name!r}{hint} "
                                f"(registered: {sorted(registry)})")
                continue
            params = dict(s.params)
            if "seed" in cls.default_params() and "seed" not in params:
                params["seed"] = self.seed
            try:
                out[s.key] = get_strategy(s.name, **params)
            except (TypeError, ValueError) as exc:
                problems.append(f"strategies[{i}] ({s.key}): {exc}")
        if problems:
            raise ConfigError(problems)
        return out

    def live_sizer_kwargs(self) -> dict[str, Any]:
        """:class:`VolTargetSizer` kwargs for the live runner / trading artifact.

        The live runner only implements volatility targeting, so a config whose research runs
        use another sizer cannot be traded as-is (the live book would be sized differently from
        the backtested one: train/serve skew). Raises :class:`ConfigError` in that case.
        """
        if self.sizing.method != "vol_target":
            raise ConfigError(f"sizing.method={self.sizing.method!r}: the live runner only supports 'vol_target' "
                              "sizing, so live positions would not match the research backtests")
        return self._vol_target_kwargs()

    def _vol_target_kwargs(self) -> dict[str, Any]:
        s = self.sizing
        return {"target_vol": s.target_vol, "max_leverage": s.max_leverage, "max_lots": s.max_lots,
                "rebalance_band": s.rebalance_band, "kelly_cap": s.kelly_cap, "min_vol": s.min_vol,
                "drawdown_derisk": s.drawdown_derisk}

    def live_runner_mapping(self) -> dict[str, Any]:
        """The mapping for :meth:`aurum.live.runner.LiveConfig.from_mapping` (no secrets).

        Built from the ``live`` section plus the SAME instrument/costs/sizing/live-risk/desk
        settings the research runs use (SPEC §0.2). ``live.options`` is merged last but may
        only ADD runner settings: overriding a value set here (``allow_live_real``,
        ``dry_run``, ``broker``, ``magic``, the ``risk``/``sizer``/``costs`` sections, ...) is
        rejected by :meth:`validate`, so the typed, validated fields are what the runner gets.
        """
        self.live_sizer_kwargs()   # raises unless the research sizer is the one live implements
        base = self._live_base_mapping()
        problems = _live_option_problems(base, self.live.options)
        if problems:
            raise ConfigError(problems)
        dropped = sorted(k for k, v in self.agents.desk.items() if isinstance(v, Mapping))
        if dropped and self.live.use_desk:
            logger.warning("live desk: nested desk settings %s are not passed to the live runner "
                           "(it builds DeskConfig from flat keys); desk defaults apply", dropped)
        return _deep_merge(base, copy.deepcopy(self.live.options))

    def _live_base_mapping(self) -> dict[str, Any]:
        lv = self.live
        sizer = self._vol_target_kwargs()
        desk_cfg = {k: v for k, v in self.agents.desk.items() if not isinstance(v, Mapping)}
        out: dict[str, Any] = {
            "artifact_dir": lv.artifact_dir, "symbol": lv.symbol or self.instrument.symbol,
            "timeframe": self.data.timeframe, "magic": lv.magic, "broker": lv.broker, "dry_run": lv.dry_run,
            "allow_live_real": lv.allow_live_real, "state_dir": lv.state_dir, "history_bars": lv.history_bars,
            "retry_poll_seconds": lv.poll_seconds,
            "stop_atr_mult": self.backtest.stop_atr_mult, "take_profit_atr_mult": self.backtest.take_profit_atr_mult,
            "atr_period": self.backtest.atr_period,
            "stop_cooldown_bars": self.backtest.stop_cooldown_bars,
            "macro_dir": str(self.data.resolved_macro_dir()) if self.data.macro else None,
            "calendar": "rule_based" if self.data.events == "rule_based" else None,
            "calendar_csv": self.data.events if str(self.data.events).lower().endswith(".csv") else None,
            "risk": copy.deepcopy(self.risk.live),
            "sizer": sizer,
            "costs": dataclasses.asdict(self.costs),
            "desk": {"enabled": lv.use_desk, "mode": self.agents.policy.mode,
                     "on_failure": self.agents.policy.on_failure,
                     "max_abs_forecast": self.agents.policy.max_abs_forecast,
                     "min_confidence": self.agents.policy.min_confidence, "config": desk_cfg},
        }
        return out

    def to_dict(self, *, redact: bool = True, for_hash: bool = False) -> dict[str, Any]:
        """Plain dict (YAML/JSON-able). Secrets are dropped unless ``redact=False`` (then
        they appear masked). ``for_hash`` also drops fields that cannot change results."""
        return _to_plain(self, redact=redact, for_hash=for_hash)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def save(self, path: str | Path) -> Path:
        """Write the resolved (secret-free) config as YAML."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p

    def config_hash(self) -> str:
        """SHA-256 of every result-relevant field (canonical JSON)."""
        payload = json.dumps(self.to_dict(for_hash=True), sort_keys=True, separators=(",", ":"),
                             default=str, allow_nan=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> AurumConfig:
        """Semantic validation (raises :class:`ConfigError` listing every problem)."""
        problems = _semantic_problems(self)
        if problems:
            raise ConfigError(problems)
        return self


# ----------------------------------------------------------------------------------------
# dict <-> dataclass
# ----------------------------------------------------------------------------------------
def _to_plain(obj: Any, *, redact: bool, for_hash: bool) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            if f.metadata.get("secret"):
                if redact or for_hash:
                    continue
            if for_hash and f.metadata.get("hash") is False:
                continue
            out[f.name] = _to_plain(getattr(obj, f.name), redact=redact, for_hash=for_hash)
        return out
    if isinstance(obj, Secret):
        return repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v, redact=redact, for_hash=for_hash) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v, redact=redact, for_hash=for_hash) for v in obj]
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    return obj


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _union_args(tp: Any) -> tuple[Any, ...] | None:
    origin = typing.get_origin(tp)
    if origin is Union or origin is types.UnionType:
        return typing.get_args(tp)
    return None


def _coerce(value: Any, tp: Any, path: str, problems: list[str]) -> Any:
    """Check/convert one YAML value against a type annotation; record problems."""
    if tp is Any:
        return value
    args = _union_args(tp)
    if args is not None:
        if value is None:
            if type(None) in args:
                return None
            problems.append(f"{path}: must not be null (expected {' | '.join(_type_name(a) for a in args)})")
            return None
        # Prefer an exact type match (e.g. int in `int | str`) before coercion attempts.
        non_none = [a for a in args if a is not type(None)]
        for a in non_none:
            if a in (bool, int, float, str) and type(value) is a:
                return value
        for a in non_none:
            sub: list[str] = []
            out = _coerce(value, a, path, sub)
            if not sub:
                return out
        problems.append(f"{path}: expected {' | '.join(_type_name(a) for a in non_none)}, "
                        f"got {type(value).__name__} {value!r}")
        return None
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, Mapping):
            problems.append(f"{path}: expected a mapping, got {type(value).__name__}")
            return tp() if _has_defaults(tp) else None
        return _build(tp, value, path, problems)
    origin = typing.get_origin(tp)
    if origin in (list, Sequence) or tp is list:
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            problems.append(f"{path}: expected a list, got {type(value).__name__} {value!r}")
            return []
        (item_tp,) = typing.get_args(tp) or (Any,)
        return [_coerce(v, item_tp, f"{path}[{i}]", problems) for i, v in enumerate(value)]
    if origin in (dict, Mapping) or tp is dict:
        if not isinstance(value, Mapping):
            problems.append(f"{path}: expected a mapping, got {type(value).__name__} {value!r}")
            return {}
        kt_vt = typing.get_args(tp)
        vt = kt_vt[1] if len(kt_vt) == 2 else Any
        return {str(k): _coerce(v, vt, f"{path}.{k}", problems) for k, v in value.items()}
    if tp is bool:
        if isinstance(value, bool):
            return value
        problems.append(f"{path}: expected true/false, got {value!r}")
        return False
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            problems.append(f"{path}: expected an integer, got {value!r}")
            return 0
        return value
    if tp is float:
        if isinstance(value, bool):
            problems.append(f"{path}: expected a number, got {value!r}")
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):  # PyYAML reads "1e-3" as a string
            try:
                return float(value)
            except ValueError:
                pass
        problems.append(f"{path}: expected a number, got {value!r}")
        return 0.0
    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (dt.datetime, dt.date)):  # YAML turns 2024-01-01 into a date
            return value.isoformat()
        problems.append(f"{path}: expected a string, got {type(value).__name__} {value!r}")
        return ""
    if tp is Secret:
        if isinstance(value, Secret):
            return value
        problems.append(f"{path}: secrets cannot be set in config files")
        return None
    return value


def _has_defaults(cls: type) -> bool:
    return all(f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
               for f in dataclasses.fields(cls))


def _build(cls: type, data: Mapping[str, Any], path: str, problems: list[str]) -> Any:
    hints = typing.get_type_hints(cls)
    fields_ = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in data if k not in fields_]
    for k in unknown:
        close = difflib.get_close_matches(str(k), list(fields_), n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        problems.append(f"{path + '.' if path else ''}{k}: unknown key{hint}; "
                        f"valid keys: {sorted(fields_)}")
    kwargs: dict[str, Any] = {}
    for name, f in fields_.items():
        fpath = f"{path}.{name}" if path else name
        if name in data:
            if f.metadata.get("secret") and data[name] is not None and not isinstance(data[name], Secret):
                env = SECRET_ENV.get(fpath, "an environment variable")
                problems.append(f"{fpath}: secrets must not be stored in config files; set ${env} instead")
                continue
            kwargs[name] = _coerce(data[name], hints[name], fpath, problems)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            problems.append(f"{fpath}: required key missing")
            kwargs[name] = None
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive (missing required handled above)
        problems.append(f"{path or '<root>'}: {exc}")
        return None


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


#: live-runner sections that must come from the typed config (research/live parity), never
#: from ``live.options``: where to set them instead.
_LIVE_TYPED_SECTIONS: dict[str, str] = {"risk": "risk.live", "sizer": "sizing", "costs": "costs"}
#: keys that look like credentials (secrets come from the environment only, SPEC §0.5)
_SECRET_KEY_RE = re.compile(r"(^|_)(pass(word|wd)?|secret|token|api[_-]?key|credentials?|login)$|^env$", re.I)


def _live_option_problems(base: Mapping[str, Any], options: Mapping[str, Any], path: str = "live.options"
                          ) -> list[str]:
    """``live.options`` may ADD runner settings but never override one the typed config sets
    (e.g. ``allow_live_real``/``dry_run``/``broker``/``magic``: the real-money guard, the CLI
    banner and the semantic checks all read the typed fields), nor carry credentials."""
    problems: list[str] = []

    def secrets(node: Any, where: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                kp = f"{where}.{k}"
                if _SECRET_KEY_RE.search(str(k)) or (str(k).lower() == "webhook" and isinstance(v, str)):
                    problems.append(f"{kp}: credentials must not be stored in config files (use the "
                                    f"environment: {sorted(SECRET_ENV.values())}, MT5_LOGIN/MT5_SERVER)")
                else:
                    secrets(v, kp)

    def collide(b: Mapping[str, Any], o: Mapping[str, Any], where: str) -> None:
        for k, v in o.items():
            kp = f"{where}.{k}"
            if where == path and k in _LIVE_TYPED_SECTIONS:
                problems.append(f"{kp}: set these values in '{_LIVE_TYPED_SECTIONS[k]}' (the SAME settings the "
                                "research runs use), not in live.options")
            elif k in b:
                if isinstance(b[k], Mapping) and isinstance(v, Mapping):
                    collide(b[k], v, kp)
                else:
                    problems.append(f"{kp}: overrides a value the typed config already sets "
                                    f"({b[k]!r}); set it in its own config field instead")

    if not isinstance(options, Mapping):
        return [f"{path}: expected a mapping"]
    secrets(options, path)
    collide(base, options, path)
    return problems


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    import yaml

    path = path.resolve()
    if path in seen:
        raise ConfigError(f"circular 'extends' chain: {' -> '.join(str(p) for p in (*seen, path))}")
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    data = dict(data)
    parent = data.pop("extends", None)
    if parent:
        parents = [parent] if isinstance(parent, str) else list(parent)
        merged: dict[str, Any] = {}
        for p in parents:
            ppath = Path(p)
            if not ppath.is_absolute():
                ppath = path.parent / ppath
            merged = _deep_merge(merged, _read_yaml(ppath, (*seen, path)))
        data = _deep_merge(merged, data)
    return data


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``["a.b=1", "strategies=[]", ...]`` (values parsed as YAML) to a raw dict."""
    import yaml

    out = copy.deepcopy(dict(data))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"override {item!r} has an empty key")
        if key in SECRET_ENV:
            raise ConfigError(f"{key}: secrets cannot be overridden; set ${SECRET_ENV[key]}")
        try:
            value = yaml.safe_load(raw) if raw.strip() else None
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {item!r}: invalid value: {exc}") from exc
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = {}
                node[p] = nxt
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {item!r}: {p!r} is not a mapping")
            node = nxt
        node[parts[-1]] = value
    return out


def config_from_dict(data: Mapping[str, Any] | None = None, *, env: Mapping[str, str] | None = None,
                     validate: bool = True, source: str | None = None) -> AurumConfig:
    """Build (and by default validate) an :class:`AurumConfig` from a raw mapping."""
    data = dict(data or {})
    data.pop("extends", None)
    problems: list[str] = []
    cfg = _build(AurumConfig, data, "", problems)
    if problems:
        raise ConfigError(problems)
    cfg.source = source
    _apply_secret_env(cfg, os.environ if env is None else env)
    if validate:
        cfg.validate()
    return cfg


def load_config(source: str | Path | Mapping[str, Any] | None = None, *, overrides: Sequence[str] = (),
                env: Mapping[str, str] | None = None, validate: bool = True) -> AurumConfig:
    """Load a YAML file (or mapping; ``None`` = all defaults), apply overrides and secrets
    from the environment, and validate. Raises :class:`ConfigError` on any problem."""
    if source is None:
        raw: dict[str, Any] = {}
        src = None
    elif isinstance(source, Mapping):
        raw = dict(source)
        src = None
    else:
        path = Path(source)
        raw = _read_yaml(path)
        src = str(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw, env=env, validate=validate, source=src)


def _apply_secret_env(cfg: AurumConfig, env: Mapping[str, str]) -> None:
    for dotted, var in SECRET_ENV.items():
        value = env.get(var)
        if not value:
            continue
        section, attr = dotted.split(".")
        setattr(getattr(cfg, section), attr, Secret(value, var))


# ----------------------------------------------------------------------------------------
# semantic validation
# ----------------------------------------------------------------------------------------
def _try(problems: list[str], path: str, fn: Any) -> Any:
    try:
        return fn()
    except (ValueError, TypeError, KeyError) as exc:
        msg = exc.problems if isinstance(exc, ConfigError) else [str(exc)]
        problems.extend(f"{path}: {m}" for m in msg)
        return None


def _check_ts(problems: list[str], path: str, value: str | None) -> Any:
    if value is None:
        return None
    import pandas as pd

    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        problems.append(f"{path}: not a timestamp ({value!r}): {exc}")
        return None
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _semantic_problems(cfg: AurumConfig) -> list[str]:
    p: list[str] = []
    d = cfg.data
    _try(p, "data.timeframe", lambda: get_timeframe(d.timeframe))
    if d.events not in ("rule_based", "none") and not str(d.events).lower().endswith(".csv"):
        p.append(f"data.events: expected 'rule_based', 'none' or a .csv path, got {d.events!r}")
    s, e = _check_ts(p, "data.start", d.start), _check_ts(p, "data.end", d.end)
    if s is not None and e is not None and s >= e:
        p.append(f"data.start ({d.start}) must be before data.end ({d.end})")
    if d.synthetic is not None:
        if d.synthetic.model not in ("gbm", "trend", "mean_revert", "regime", "jump"):
            p.append(f"data.synthetic.model: unknown {d.synthetic.model!r}")
        if d.synthetic.n < 50:
            p.append("data.synthetic.n must be >= 50")

    _try(p, "instrument", cfg.instrument.build)
    _try(p, "costs", cfg.costs.build)
    if cfg.instrument.lot_step <= 0 or cfg.instrument.contract_size <= 0:
        p.append("instrument: lot_step and contract_size must be positive")

    f = cfg.features
    if not (isinstance(f.enabled, bool) or f.enabled == "auto"):
        p.append(f"features.enabled: expected true, false or 'auto', got {f.enabled!r}")
    if f.scaler not in ("robust", "standard", "none"):
        p.append(f"features.scaler: expected robust | standard | none, got {f.scaler!r}")
    if f.clip is not None and not (f.clip > 0):
        p.append("features.clip must be positive or null")

    keys = [s_.key for s_ in cfg.strategies]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        p.append(f"strategies: duplicate ids {dup} (give repeated strategies distinct 'id's)")
    for i, st in enumerate(cfg.strategies):
        if not st.name:
            p.append(f"strategies[{i}].name must be a non-empty string")
        if st.weight is not None:
            if not (math.isfinite(st.weight) and st.weight >= 0):
                p.append(f"strategies[{i}].weight must be >= 0")
            if cfg.combiner.method != "fixed":
                p.append(f"strategies[{i}].weight is only used with combiner.method='fixed' "
                         f"(method is {cfg.combiner.method!r})")
    if cfg.combiner.method == "fixed":
        ws = [s_.weight if s_.weight is not None else 1.0 for s_ in cfg.strategies if s_.enabled]
        if ws and sum(ws) <= 0:
            p.append("combiner.method='fixed' needs at least one positive strategy weight")
    elif cfg.combiner.method not in ("sharpe_shrink", "equal", "inverse_vol", "hrp"):
        p.append(f"combiner.method: unknown {cfg.combiner.method!r} "
                 "(sharpe_shrink | equal | inverse_vol | hrp | fixed)")
    else:
        _try(p, "combiner", cfg.combiner.build)

    if cfg.sizing.method not in ("vol_target", "fixed_fractional"):
        p.append(f"sizing.method: unknown {cfg.sizing.method!r} (vol_target | fixed_fractional)")
    else:
        _try(p, "sizing", cfg.sizing.build)

    for mode in ("research", "live"):
        _try(p, f"risk.{mode}", lambda m=mode: _limits_with_help(cfg.risk, m))

    b = cfg.backtest
    if not (b.initial_equity > 0):
        p.append("backtest.initial_equity must be positive")
    for name in ("stop_atr_mult", "take_profit_atr_mult"):
        v = getattr(b, name)
        if v is not None and not (v > 0):
            p.append(f"backtest.{name} must be positive or null")
    if b.atr_period < 1:
        p.append("backtest.atr_period must be >= 1")
    if b.stop_cooldown_bars < 0:
        p.append("backtest.stop_cooldown_bars must be >= 0")
    bs, be = _check_ts(p, "backtest.start", b.start), _check_ts(p, "backtest.end", b.end)
    if bs is not None and be is not None and bs >= be:
        p.append("backtest.start must be before backtest.end")

    w = cfg.walkforward
    for name in ("train", "test", "step", "embargo"):
        v = getattr(w, name)
        if v is None or (name == "embargo" and v == 0):
            continue
        _try(p, f"walkforward.{name}", lambda v=v: parse_duration(v))
    if isinstance(w.purge, str) and w.purge != "auto":
        _try(p, "walkforward.purge", lambda: parse_duration(w.purge))
    elif isinstance(w.purge, int) and w.purge < 0:
        p.append("walkforward.purge must be >= 0 or 'auto'")
    _check_ts(p, "walkforward.holdout_start", w.holdout_start)
    if w.min_train_bars < 30:
        p.append("walkforward.min_train_bars must be >= 30 (the combiner needs 30 observations)")
    if w.min_test_bars < 1:
        p.append("walkforward.min_test_bars must be >= 1")
    if w.pbo_splits < 2 or w.pbo_splits % 2:
        p.append("walkforward.pbo_splits must be an even integer >= 2")
    if w.n_boot < 100:
        p.append("walkforward.n_boot must be >= 100")
    if w.n_jobs < 0:
        p.append("walkforward.n_jobs must be >= 0 (0 = auto)")
    if w.executor not in ("auto", "process", "thread", "serial"):
        p.append(f"walkforward.executor: expected auto | process | thread | serial, got {w.executor!r}")
    if w.combiner_fit not in ("oos", "train"):
        p.append(f"walkforward.combiner_fit: expected oos | train, got {w.combiner_fit!r}")
    if w.combiner_min_obs < 30:
        p.append("walkforward.combiner_min_obs must be >= 30")
    if w.on_strategy_error not in ("raise", "drop"):
        p.append(f"walkforward.on_strategy_error: expected raise | drop, got {w.on_strategy_error!r}")
    if w.n_trials is not None and w.n_trials < 1:
        p.append("walkforward.n_trials must be >= 1 or null")
    if w.history_bars is not None and w.history_bars < 0:
        p.append("walkforward.history_bars must be >= 0 or null")

    a = cfg.agents
    _try(p, "agents.policy", a.policy.build)
    _try(p, "agents.desk", a.desk_config)
    if a.replay_every < 1:
        p.append("agents.replay_every must be >= 1")
    if not (a.replay_max_cost_usd > 0):
        p.append("agents.replay_max_cost_usd must be positive")
    if a.lookback_bars < 2:
        p.append("agents.lookback_bars must be >= 2")

    lv = cfg.live
    if lv.broker not in ("paper", "mt5"):
        p.append(f"live.broker: expected paper | mt5, got {lv.broker!r}")
    if not 0 < lv.magic < 2**31:
        p.append("live.magic must be a positive 32-bit integer (MT5 magic number)")
    p.extend(_live_option_problems(cfg._live_base_mapping(), lv.options))
    if not (lv.poll_seconds > 0):
        p.append("live.poll_seconds must be positive")
    if lv.history_bars is not None and lv.history_bars < 100:
        p.append("live.history_bars must be >= 100 or null")
    if lv.allow_live_real and lv.broker == "paper":
        p.append("live.allow_live_real is meaningless with the paper broker")

    r = cfg.rl
    rs, re_, rv = (_check_ts(p, "rl.train_start", r.train_start), _check_ts(p, "rl.train_end", r.train_end),
                   _check_ts(p, "rl.val_end", r.val_end))
    if rs is not None and re_ is not None and rs >= re_:
        p.append("rl.train_start must be before rl.train_end")
    if re_ is not None and rv is not None and re_ >= rv:
        p.append("rl.train_end must be before rl.val_end")
    return p


def _limits_with_help(risk: RiskConfig, mode: str) -> Any:
    try:
        return risk.limits(mode)
    except TypeError as exc:
        from aurum.risk.manager import RiskLimits

        valid = sorted(f.name for f in dataclasses.fields(RiskLimits))
        raise ValueError(f"{exc}; valid keys: {valid}") from exc
