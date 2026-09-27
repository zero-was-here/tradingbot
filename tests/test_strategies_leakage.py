"""Generic point-in-time leakage + output-contract test for EVERY registered strategy.

SPEC §0-1 / ``aurum/strategies/base.py``: ``forecast[t]`` may depend only on data available
at the close of bar ``t`` (``bars.available_at[t]``): bars ``[0, t]`` and macro rows whose
``available_at <= available_at[t]``. For each cutoff ``t`` two alternative histories agree
with the original up to that instant and differ afterwards:

* **perturbed** — every bar after ``t`` comes from an unrelated random path (other model,
  seed, price level, volatility, spread and volume) and every macro row not yet available at
  ``available_at[t]`` gets unrelated values in all numeric columns (including non-positive
  prints);
* **truncated** — bars after ``t`` and macro rows not yet available are removed.

Scheduled event TIMES are public in advance (SPEC §3.5), so the calendar is left intact.
A causal strategy returns bit-identical ``forecast[:t+1]`` in all three histories.

Every strategy registered by an importable ``aurum.strategies.*`` module is tested — the
rule-based ones owned by this module and any ML/RL ones present. Trainable strategies are
fitted ONCE on the first 60% of the original history (bars and macro as known then); the
same fitted object then generates on each history, which is exactly the walk-forward
usage. Negative controls register deliberately leaky strategies under temporary names and
assert the harness flags them (and that the registry is cleaned up afterwards).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.pit import asof_join
from aurum.data.schema import validate_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.strategies.base import _STRATEGIES, Strategy, register_strategy

N_BARS = 3500
CUTOFFS = (400, 1200, 2200, 3100)
TRAIN_FRAC = 0.6

OWNED_MODULES = (
    "aurum.strategies.trend",
    "aurum.strategies.mean_reversion",
    "aurum.strategies.breakout",
    "aurum.strategies.macro",
    "aurum.strategies.seasonal",
)
OTHER_MODULES = ("aurum.strategies.ml", "aurum.strategies.rl")
OWNED = frozenset({
    "tsmom", "ema_cross", "donchian", "kalman_trend", "zscore_fade", "rsi2", "bollinger_revert",
    "vol_squeeze", "orb", "macro_factor", "risk_off", "intraday_seasonality",
})
#: forecasts that are a function of the clock (and fitted state) only: perturbing prices or
#: macro cannot change them, so the "perturbation must change the future" check is skipped.
TIME_ONLY = frozenset({"intraday_seasonality"})

#: Test-only parameter overrides, applied only for keys the strategy actually declares:
#: keep the expensive learners small, and switch off skill gates / pre-tests that would make
#: the forecast identically 0 on synthetic data (a zero forecast makes the check vacuous).
TEST_PARAMS: dict[str, dict[str, Any]] = {
    "intraday_seasonality": {"significance": None, "shrinkage": 500.0, "min_obs": 5},
    # the synthetic VIX path has no spike above +1.5 sd: lower the thresholds so the regime
    # machinery (EWMA baseline, hysteresis, DXY filter, as-of join) is actually exercised.
    "risk_off": {"entry_z": 0.0, "exit_z": -1.0},
    "ml_gbm": {"skill_gate_z": None, "importance": "none",
               "feature_groups": ("returns", "momentum", "volatility")},
    "meta_label": {"skill_gate_z": None, "importance": "none",
                   "feature_groups": ("returns", "momentum", "volatility")},
    "rl_ppo": {"min_val_bars": 200, "config": {
        "total_timesteps": 256, "n_envs": 1, "n_steps": 64, "batch_size": 32, "n_epochs": 1,
        "eval_freq": 128, "patience": None, "net_arch": [16],
        "feature_groups": ["returns", "volatility"], "env": {"episode_length": 128}}},
}


def _import_strategy_modules() -> dict[str, str]:
    """Import every strategy module; return ``{module: error}`` for the ones that fail."""
    errors: dict[str, str] = {}
    for mod in OWNED_MODULES + OTHER_MODULES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            if exc.name != mod:
                errors[mod] = repr(exc)
        except Exception as exc:  # noqa: BLE001 - a broken sibling module must not break collection
            errors[mod] = repr(exc)
    return errors


IMPORT_ERRORS = _import_strategy_modules()
ALL_NAMES = sorted(n for n in _STRATEGIES if not n.startswith("__"))


# ---------------------------------------------------------------------------------------
# market construction
# ---------------------------------------------------------------------------------------
def _base_market() -> MarketData:
    bars = make_synthetic_bars(N_BARS, "H1", seed=5, model="regime")
    macro = make_synthetic_macro(bars, seed=5)
    events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=14))
    return MarketData(bars=bars, macro=macro, events=events)


def perturbed_market(md: MarketData, t: int, seed: int) -> MarketData:
    """Same history up to bar ``t`` (inclusive) and up to ``available_at[t]`` for macro;
    unrelated data afterwards."""
    bars = md.bars
    alt = make_synthetic_bars(len(bars), bars.attrs.get("timeframe", "H1"), seed=1000 + seed,
                              model="jump", start_price=float(bars["close"].iloc[t]) * 1.37,
                              annual_vol=0.45, spread=0.9)
    assert alt.index.equals(bars.index), "synthetic timelines must coincide"
    new = pd.concat([bars.iloc[: t + 1], alt.iloc[t + 1:]])
    new.attrs["timeframe"] = bars.attrs.get("timeframe", "H1")
    validate_bars(new)
    cutoff = bars["available_at"].iloc[t]
    rng = np.random.default_rng(seed)
    macro = {}
    for name, frame in md.macro.items():
        f = frame.copy()
        future = np.asarray(pd.DatetimeIndex(f["available_at"]) > cutoff)
        k = int(future.sum())
        for col in f.columns:
            if col == "available_at" or not pd.api.types.is_numeric_dtype(f[col]):
                continue
            vals = f[col].to_numpy(dtype=float).copy()
            if k:
                scale = float(np.nanmean(np.abs(vals))) or 1.0
                vals[future] = rng.uniform(-1.5, 2.5, k) * scale
            f[col] = vals
        macro[name] = f
    return MarketData(bars=new, macro=macro, events=md.events)


def truncated_market(md: MarketData, t: int) -> MarketData:
    """History cut at bar ``t``; only macro rows available by ``available_at[t]``."""
    cutoff = md.bars["available_at"].iloc[t]
    bars = md.bars.iloc[: t + 1].copy()
    bars.attrs["timeframe"] = md.bars.attrs.get("timeframe", "H1")
    macro = {k: v.loc[np.asarray(pd.DatetimeIndex(v["available_at"]) <= cutoff)].copy()
             for k, v in md.macro.items()}
    return MarketData(bars=bars, macro=macro, events=md.events)


# ---------------------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------------------
def make_strategy(name: str, md: MarketData) -> Strategy:
    """Instantiate ``name`` (with test overrides it declares) and fit it on the first
    ``TRAIN_FRAC`` of ``md`` when trainable."""
    cls = _STRATEGIES[name]
    defaults = cls.default_params()
    params = {k: v for k, v in TEST_PARAMS.get(name, {}).items() if k in defaults}
    strat = cls(**params)
    if strat.trainable:
        strat.fit(truncated_market(md, int(TRAIN_FRAC * len(md.bars)) - 1))
    return strat


def compare_prefix(a: pd.Series, b: pd.Series, t: int, label: str) -> list[str]:
    """``a[:t+1]`` and ``b[:t+1]`` must be identical (bit for bit; NaN == NaN)."""
    x = a.iloc[: t + 1].to_numpy(dtype=float)
    y = b.iloc[: t + 1].to_numpy(dtype=float)
    if x.shape != y.shape:
        return [f"{label}: prefix lengths differ ({x.shape} vs {y.shape})"]
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    if same.all():
        return []
    first = int(np.flatnonzero(~same)[0])
    return [f"{label}: forecast differs first at row {first} ({x[first]!r} vs {y[first]!r}); "
            f"{int((~same).sum())} rows differ"]


def leak_report(strat: Strategy, md: MarketData,
                alternatives: dict[int, list[tuple[str, MarketData]]]) -> list[str]:
    full = strat.generate(md)
    problems: list[str] = []
    for t, alts in alternatives.items():
        for label, alt in alts:
            problems += compare_prefix(full, strat.generate(alt), t, f"{strat.name}@t={t}/{label}")
    return problems


def check_contract(strat: Strategy, md: MarketData, f: pd.Series) -> list[str]:
    """Output contract of ``aurum.strategies.base``: index, finite, [-1, 1], zero warm-up."""
    problems = []
    if not isinstance(f, pd.Series):
        return [f"returned {type(f).__name__}, not a Series"]
    if not f.index.equals(md.bars.index):
        problems.append("index differs from md.bars.index")
    v = f.to_numpy(dtype=float)
    if not np.isfinite(v).all():
        problems.append(f"{int((~np.isfinite(v)).sum())} NaN/inf values")
    if (np.abs(v[np.isfinite(v)]) > 1.0 + 1e-12).any():
        problems.append(f"values outside [-1, 1] (max |f| = {np.nanmax(np.abs(v)):.4f})")
    w = int(strat.warmup_bars)
    if w < 0:
        problems.append(f"negative warmup_bars {w}")
    elif (v[: min(w, len(v))] != 0.0).any():
        problems.append(f"non-zero forecast inside the {w}-bar warm-up")
    return problems


@pytest.fixture(scope="module")
def market() -> MarketData:
    return _base_market()


@pytest.fixture(scope="module")
def alternatives(market: MarketData) -> dict[int, list[tuple[str, MarketData]]]:
    return {t: [("perturbed", perturbed_market(market, t, seed=i)),
                ("truncated", truncated_market(market, t))]
            for i, t in enumerate(CUTOFFS)}


@pytest.fixture(scope="module")
def fitted(market: MarketData) -> Callable[[str], Strategy]:
    """Cached factory: each strategy is built (and fitted) once per module."""
    cache: dict[str, Strategy | BaseException] = {}

    def _get(name: str) -> Strategy:
        if name not in cache:
            try:
                cache[name] = make_strategy(name, market)
            except Exception as exc:  # noqa: BLE001 - reported below
                cache[name] = exc
        got = cache[name]
        if isinstance(got, BaseException):
            if name in OWNED:
                raise got
            pytest.skip(f"{name} (not owned by strategies_rules) could not be built/fitted on "
                        f"synthetic data with the test overrides: {got!r}")
        return got

    return _get


@pytest.fixture
def temp_strategy() -> Iterator[Callable[[type[Strategy]], type[Strategy]]]:
    """Register throw-away strategy classes; always removed from the registry."""
    names: list[str] = []

    def _register(cls: type[Strategy]) -> type[Strategy]:
        register_strategy(cls)
        names.append(cls.name)
        return cls

    try:
        yield _register
    finally:
        for n in names:
            _STRATEGIES.pop(n, None)


# ---------------------------------------------------------------------------------------
# registry / import
# ---------------------------------------------------------------------------------------
def test_owned_modules_import_cleanly() -> None:
    bad = {m: e for m, e in IMPORT_ERRORS.items() if m in OWNED_MODULES}
    assert not bad, f"rule-based strategy modules failed to import: {bad}"


def test_all_rule_strategies_registered() -> None:
    missing = OWNED - set(ALL_NAMES)
    assert not missing, f"missing SPEC §6 rule-based strategies: {sorted(missing)}"


@pytest.mark.parametrize("module", OTHER_MODULES)
def test_other_strategy_modules_importable(module: str) -> None:
    if module in IMPORT_ERRORS:
        pytest.skip(f"{module} is not importable right now (owned by another module): "
                    f"{IMPORT_ERRORS[module]}")


# ---------------------------------------------------------------------------------------
# contract + point-in-time, every registered strategy
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_output_contract(name: str, market: MarketData, fitted: Callable[[str], Strategy]) -> None:
    strat = fitted(name)
    f = strat.generate(market)
    problems = check_contract(strat, market, f)
    assert not problems, f"{name}: " + "; ".join(problems)
    again = strat.generate(market)
    assert np.array_equal(f.to_numpy(), again.to_numpy()), f"{name}: generate is not deterministic"
    clone = strat.clone().generate(market)
    assert np.array_equal(f.to_numpy(), clone.to_numpy()), f"{name}: clone() changes the forecast"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_strategy_is_point_in_time(name: str, market: MarketData, fitted: Callable[[str], Strategy],
                                   alternatives: dict[int, list[tuple[str, MarketData]]]) -> None:
    strat = fitted(name)
    problems = leak_report(strat, market, alternatives)
    assert not problems, "look-ahead detected:\n" + "\n".join(problems[:20])
    for t, alts in alternatives.items():
        for label, alt in alts:
            f = strat.generate(alt)
            bad = check_contract(strat, alt, f)
            assert not bad, f"{name}@t={t}/{label}: " + "; ".join(bad)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_leakage_check_is_not_vacuous(name: str, market: MarketData,
                                      fitted: Callable[[str], Strategy],
                                      alternatives: dict[int, list[tuple[str, MarketData]]]) -> None:
    """The prefix comparison is only meaningful if the forecast is actually non-zero before
    the last cutoff and (for data-driven strategies) reacts to the perturbed future."""
    strat = fitted(name)
    full = strat.generate(market).to_numpy()
    t_last = max(CUTOFFS)
    if not (full[: t_last + 1] != 0).any():
        if name in OWNED:
            pytest.fail(f"{name}: forecast is identically 0 up to t={t_last}; check is vacuous")
        pytest.skip(f"{name}: forecast identically 0 on this synthetic market (not owned here)")
    if name in TIME_ONLY:
        return
    changed = False
    for t in sorted(CUTOFFS):
        other = strat.generate(alternatives[t][0][1]).to_numpy()
        if (full[t + 1:] != other[t + 1:]).any():
            changed = True
            break
    assert changed, f"{name}: perturbing the future never changed the forecast (test is blind)"


@pytest.fixture(scope="module")
def m15_case() -> tuple[MarketData, dict[int, list[tuple[str, MarketData]]]]:
    bars = make_synthetic_bars(6000, "M15", seed=8, model="regime")
    md = MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=8))
    alts = {t: [("perturbed", perturbed_market(md, t, seed=i)), ("truncated", truncated_market(md, t))]
            for i, t in enumerate((700, 2500, 4400, 5900))}
    return md, alts


@pytest.mark.parametrize("name", sorted(OWNED))
def test_rule_strategies_are_point_in_time_on_m15(
        name: str, m15_case: tuple[MarketData, dict[int, list[tuple[str, MarketData]]]]) -> None:
    """Timeframe variant (M15): multi-bar opening ranges, bar-count windows on a finer grid."""
    md, alts = m15_case
    strat = make_strategy(name, md)
    problems = leak_report(strat, md, alts)
    assert not problems, "look-ahead detected on M15:\n" + "\n".join(problems[:20])
    assert not check_contract(strat, md, strat.generate(md))


TRAINABLE_NAMES = sorted(n for n in ALL_NAMES if _STRATEGIES[n].trainable)


def _fit_on(name: str, md: MarketData) -> Strategy:
    cls = _STRATEGIES[name]
    defaults = cls.default_params()
    strat = cls(**{k: v for k, v in TEST_PARAMS.get(name, {}).items() if k in defaults})
    return strat.fit(md) or strat


def fit_macro_leak(name: str, market: MarketData, ref: Strategy) -> list[str] | None:
    """Problems if ``fit`` on (training bars + the FULL macro dict) differs from ``ref``
    (fitted on bars and macro truncated at the end of training); ``None`` if ``fit`` is not
    bit-reproducible (inconclusive)."""
    n_train = int(TRAIN_FRAC * len(market.bars))
    future = perturbed_market(market, n_train - 1, seed=99)
    md_fit = MarketData(bars=market.bars.iloc[:n_train], macro=future.macro, events=market.events)
    a = ref.generate(market).to_numpy()
    b = _fit_on(name, md_fit).generate(market).to_numpy()
    if np.array_equal(a, b):
        return []
    control = _fit_on(name, truncated_market(market, n_train - 1)).generate(market).to_numpy()
    if not np.array_equal(a, control):
        return None
    first = int(np.flatnonzero(a != b)[0])
    return [f"{name}: fit() used macro rows not yet available at the end of training "
            f"(forecast differs first at row {first}, {int((a != b).sum())} rows)"]


@pytest.mark.parametrize("name", TRAINABLE_NAMES)
def test_fit_ignores_macro_not_yet_available_at_train_end(
        name: str, market: MarketData, fitted: Callable[[str], Strategy]) -> None:
    """``aurum.research.walkforward`` fits on ``MarketData(bars=bars[train], macro=<the FULL
    macro dict>)`` (``MarketData.slice`` does the same), so ``fit`` itself must ignore macro
    rows published after the last training bar's close. Fitting with those future rows
    replaced by unrelated values must give the same model as fitting with them removed."""
    problems = fit_macro_leak(name, market, fitted(name))
    if problems is None:
        pytest.skip(f"{name}: fit() is not bit-reproducible, so the comparison is inconclusive")
    assert not problems, problems[0]


class _LeakyFitMacro(Strategy):
    """Trainable negative control: ``fit`` reads the LAST dxy print it is given."""

    name = "__leaky_fit_macro"
    trainable = True

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> Strategy:
        self.level_ = float(md.macro["dxy"]["value"].iloc[-1])
        self.is_fitted = True
        return self

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        return self._finalize(pd.Series(np.tanh(self.level_ / 100.0 - 1.0), index=md.bars.index),
                              md.bars.index)


def test_fit_macro_check_flags_a_leaky_fit(temp_strategy, market: MarketData) -> None:
    temp_strategy(_LeakyFitMacro)
    ref = make_strategy(_LeakyFitMacro.name, market)
    problems = fit_macro_leak(_LeakyFitMacro.name, market, ref)
    assert problems, "fit-time macro leak not detected"


# ---------------------------------------------------------------------------------------
# negative controls: the harness must catch deliberately leaky strategies
# ---------------------------------------------------------------------------------------
class _LeakyNextReturn(Strategy):
    """Trades the sign of the NEXT bar's return (``shift(-1)``)."""

    name = "__leaky_next_return"

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        r = np.log(md.bars["close"]).diff().shift(-1)
        return self._finalize(np.sign(r), md.bars.index)


class _LeakyFullSampleZ(Strategy):
    """Normalises price with full-sample mean/std (uses the whole future)."""

    name = "__leaky_fullsample_z"

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        c = md.bars["close"]
        return self._finalize((c - c.mean()) / (3.0 * c.std()), md.bars.index)


class _LeakyCentred(Strategy):
    """Centred moving average (includes the next bars)."""

    name = "__leaky_centred"

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        c = np.log(md.bars["close"])
        return self._finalize(100.0 * (c.rolling(9, center=True).mean() - c), md.bars.index)


class _LeakyMacroByDate(Strategy):
    """Joins DXY by OBSERVATION date instead of ``available_at`` (classic macro leak)."""

    name = "__leaky_macro_by_date"

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        f = md.macro["dxy"].copy()
        f["available_at"] = f.index  # pretends the close is known at 00:00 of its date
        x = np.log(f["value"].where(f["value"] > 0))
        f["mom"] = -(x - x.shift(5)) * 20.0
        out = asof_join(md.bars["available_at"], f, columns=["mom"])["mom"]
        return self._finalize(out.to_numpy(), md.bars.index)


class _CausalControl(Strategy):
    """Obviously causal: sign of the last 5-bar return."""

    name = "__causal_control"

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        c = np.log(md.bars["close"])
        return self._finalize(np.sign(c - c.shift(5)), md.bars.index)


@pytest.mark.parametrize("cls", [_LeakyNextReturn, _LeakyFullSampleZ, _LeakyCentred, _LeakyMacroByDate],
                         ids=["shift-1", "fullsample-z", "centred", "macro-by-date"])
def test_harness_flags_leaky_strategies(cls: type[Strategy], temp_strategy, market: MarketData,
                                        alternatives: dict[int, list[tuple[str, MarketData]]]) -> None:
    temp_strategy(cls)
    strat = make_strategy(cls.name, market)   # through the registry, like the real ones
    problems = leak_report(strat, market, alternatives)
    assert problems, f"leakage harness failed to flag {cls.__name__}"


def test_harness_accepts_causal_control(temp_strategy, market: MarketData,
                                        alternatives: dict[int, list[tuple[str, MarketData]]]) -> None:
    temp_strategy(_CausalControl)
    strat = make_strategy(_CausalControl.name, market)
    assert leak_report(strat, market, alternatives) == []
    assert not check_contract(strat, market, strat.generate(market))


def test_contract_checker_flags_bad_output(market: MarketData) -> None:
    """The contract checker itself must reject NaN, out-of-range and warm-up violations."""
    strat = _CausalControl()
    idx = market.bars.index
    bad = pd.Series(np.r_[np.nan, np.full(len(idx) - 1, 1.5)], index=idx)
    msgs = check_contract(strat, market, bad)
    assert any("NaN" in m for m in msgs) and any("outside" in m for m in msgs)

    class _Warm(_CausalControl):
        name = "__warm"

        @property
        def warmup_bars(self) -> int:
            return 10

    assert any("warm-up" in m for m in check_contract(_Warm(), market, pd.Series(0.5, index=idx)))


def test_negative_controls_are_cleaned_up() -> None:
    assert not [n for n in _STRATEGIES if n.startswith("__")]
