"""Generic point-in-time leakage test for EVERY registered feature group (SPEC §0-1, §4).

For each cutoff bar ``t`` we build two alternative histories that agree with the original
up to the decision time ``available_at[t]`` and disagree afterwards:

* **perturbed**: every bar after ``t`` is replaced by an unrelated random path (different
  model, seed and price level; volume and spread change too), and every macro row with
  ``available_at > available_at[t]`` gets a different random value (including
  non-positive prints);
* **truncated**: bars after ``t`` and macro rows not yet available are removed.

Scheduled event TIMES are public in advance (SPEC §3.5), so the calendar is left intact.

A causal feature must produce bit-identical rows ``<= t`` (NaN == NaN) in all three
histories. Negative controls register deliberately leaky features under temporary names and
assert that the checker flags them (and clean the registry up afterwards).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.pit import asof_join
from aurum.data.schema import validate_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.features.base import _REGISTRY, get_feature, list_features, register_feature

N_BARS = 2400
CUTOFFS = (150, 700, 1500, 2150)
SPEC_GROUPS = {
    "returns", "trend", "momentum", "meanrev", "range", "volatility", "microstructure",
    "session", "mtf", "macro", "calendar", "regime",
}
TIME_ONLY_GROUPS = {"session", "calendar"}  # depend on timestamps/schedule only
#: Test-only parameter overrides for groups whose DEFAULT warm-up exceeds this 2,400-bar
#: history: ``regime``'s bounded vol rank is NaN until its 1-year window (5,796 H1 bars) is
#: full, so with defaults the prefix checks would compare NaN with NaN. A 500-bar window runs
#: the same code; the default window is checked on a long history in
#: tests/test_stability_regime.py (point-in-time and sliding-window parity).
GROUP_TEST_PARAMS: dict[str, dict] = {"regime": {"rank_window": 500}}
#: Every group registered at collection time (SPEC groups + anything added later).
ALL_GROUPS = sorted({s.name for s in list_features()} | SPEC_GROUPS)


# ---------------------------------------------------------------------------------------
# market construction
# ---------------------------------------------------------------------------------------
def _base_market() -> MarketData:
    bars = make_synthetic_bars(N_BARS, "H1", seed=3, model="regime")
    macro = make_synthetic_macro(bars, seed=3)
    events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=14))
    return MarketData(bars=bars, macro=macro, events=events)


def perturbed_market(md: MarketData, t: int, seed: int) -> MarketData:
    """Same history up to bar ``t`` (inclusive), unrelated data afterwards."""
    bars = md.bars
    alt = make_synthetic_bars(len(bars), "H1", seed=1000 + seed, model="jump",
                              start_price=float(bars["close"].iloc[t]) * 1.37, annual_vol=0.45,
                              spread=0.9)
    assert alt.index.equals(bars.index), "synthetic timelines must coincide"
    new = pd.concat([bars.iloc[: t + 1], alt.iloc[t + 1:]])
    new.attrs["timeframe"] = bars.attrs.get("timeframe", "H1")
    validate_bars(new)
    cutoff_time = bars["available_at"].iloc[t]
    rng = np.random.default_rng(seed)
    macro = {}
    for name, frame in md.macro.items():
        f = frame.copy()
        future = (pd.DatetimeIndex(f["available_at"]) > cutoff_time)
        k = int(future.sum())
        vals = f["value"].to_numpy(dtype=float).copy()
        noise = rng.uniform(-1.5, 2.5, k) * np.abs(vals[future]).mean() if k else np.empty(0)
        vals[future] = noise  # unrelated values, some non-positive
        f["value"] = vals
        macro[name] = f
    return MarketData(bars=new, macro=macro, events=md.events)


def truncated_market(md: MarketData, t: int) -> MarketData:
    """History cut at bar ``t``; only macro rows available by ``available_at[t]``."""
    cutoff_time = md.bars["available_at"].iloc[t]
    bars = md.bars.iloc[: t + 1].copy()
    bars.attrs["timeframe"] = md.bars.attrs.get("timeframe", "H1")
    macro = {k: v.loc[pd.DatetimeIndex(v["available_at"]) <= cutoff_time].copy()
             for k, v in md.macro.items()}
    return MarketData(bars=bars, macro=macro, events=md.events)


# ---------------------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------------------
def compare_prefix(a: pd.DataFrame, b: pd.DataFrame, t: int, label: str) -> list[str]:
    """Rows ``<= t`` must be identical (NaN == NaN); returns human-readable problems."""
    if list(a.columns) != list(b.columns):
        extra = sorted(set(a.columns) ^ set(b.columns))
        return [f"{label}: column set differs ({extra[:5]})"]
    x = a.iloc[: t + 1].to_numpy(dtype=float)
    y = b.iloc[: t + 1].to_numpy(dtype=float)
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    problems = []
    for j in np.flatnonzero(~same.all(axis=0)):
        first = int(np.flatnonzero(~same[:, j])[0])
        problems.append(f"{label}: column {a.columns[j]!r} differs first at row {first} "
                        f"({x[first, j]!r} vs {y[first, j]!r})")
    return problems


def leak_report(name: str, md: MarketData, alternatives: dict[int, list[MarketData]]) -> list[str]:
    """Run the perturbation + truncation checks for one registered group."""
    spec = get_feature(name)
    params = GROUP_TEST_PARAMS.get(name, {})
    full = spec.compute(md, **params)
    problems: list[str] = []
    for t, markets in alternatives.items():
        for i, alt in enumerate(markets):
            other = spec.compute(alt, **params)
            label = f"{name}@t={t}/{'perturbed' if i == 0 else 'truncated'}"
            problems += compare_prefix(full, other, t, label)
    return problems


@pytest.fixture(scope="module")
def market() -> MarketData:
    return _base_market()


@pytest.fixture(scope="module")
def alternatives(market: MarketData) -> dict[int, list[MarketData]]:
    return {t: [perturbed_market(market, t, seed=i), truncated_market(market, t)]
            for i, t in enumerate(CUTOFFS)}


@pytest.fixture
def temp_feature() -> Iterator[Callable[..., str]]:
    """Register throw-away feature functions; always removed from the registry."""
    names: list[str] = []

    def _register(name: str, fn: Callable[..., pd.DataFrame]) -> str:
        register_feature(name, family="negative_control", lookback=0)(fn)
        names.append(name)
        return name

    try:
        yield _register
    finally:
        for n in names:
            _REGISTRY.pop(n, None)


# ---------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------
def test_all_spec_groups_registered() -> None:
    names = {s.name for s in list_features()}
    assert SPEC_GROUPS <= names, f"missing groups: {sorted(SPEC_GROUPS - names)}"


@pytest.mark.parametrize("name", ALL_GROUPS)
def test_group_is_point_in_time(name: str, market: MarketData,
                                alternatives: dict[int, list[MarketData]]) -> None:
    problems = leak_report(name, market, alternatives)
    assert not problems, "look-ahead detected:\n" + "\n".join(problems[:20])


@pytest.mark.parametrize("name", ALL_GROUPS)
def test_leakage_check_is_not_vacuous(name: str, market: MarketData,
                                      alternatives: dict[int, list[MarketData]]) -> None:
    """Every column must be populated before the last cutoff (otherwise the prefix check
    compares NaN with NaN), and price-driven groups must react to the perturbation."""
    spec = get_feature(name)
    params = GROUP_TEST_PARAMS.get(name, {})
    full = spec.compute(market, **params)
    assert full.shape[1] > 0, f"{name} produced no columns"
    prefixes = (f"{name}_", f"{spec.family}_")
    assert all(c.startswith(prefixes) for c in full.columns), "columns must carry the group prefix"
    t_last = max(CUTOFFS)
    populated = full.iloc[: t_last + 1].notna().any()
    assert populated.all(), f"never populated before t={t_last}: {list(populated[~populated].index)}"
    if name not in TIME_ONLY_GROUPS:
        t = min(CUTOFFS)
        other = spec.compute(alternatives[t][0], **params)
        x = full.iloc[t + 1:].to_numpy(dtype=float)
        y = other.iloc[t + 1:].to_numpy(dtype=float)
        changed = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        assert changed.any(), f"{name}: perturbing the future changed nothing (test is blind)"


# ---- negative controls -------------------------------------------------------------------
def _leaky_shift(md: MarketData) -> pd.DataFrame:
    c = md.bars["close"]
    return pd.DataFrame({"leak_next_ret": np.log(c).diff().shift(-1)}, index=md.bars.index)


def _leaky_fullsample_z(md: MarketData) -> pd.DataFrame:
    c = md.bars["close"]
    return pd.DataFrame({"leak_z": (c - c.mean()) / c.std()}, index=md.bars.index)


def _leaky_centred(md: MarketData) -> pd.DataFrame:
    c = md.bars["close"]
    return pd.DataFrame({"leak_centred": c.rolling(5, center=True).mean() / c},
                        index=md.bars.index)


def _leaky_bfill(md: MarketData) -> pd.DataFrame:
    c = md.bars["close"]
    weekly = c.where(md.bars.index.dayofweek == 4).bfill()  # Friday close back-filled
    return pd.DataFrame({"leak_bfill": weekly / c}, index=md.bars.index)


def _leaky_macro_by_date(md: MarketData) -> pd.DataFrame:
    """Joins macro on OBSERVATION date instead of available_at (a classic macro leak)."""
    f = md.macro["dxy"].copy()
    f["available_at"] = f.index  # pretends the close is known at 00:00 of its date
    out = asof_join(md.bars["available_at"], f, columns=["value"])
    return pd.DataFrame({"leak_macro": out["value"].to_numpy()}, index=md.bars.index)


@pytest.mark.parametrize("fn", [_leaky_shift, _leaky_fullsample_z, _leaky_centred,
                                _leaky_bfill, _leaky_macro_by_date],
                         ids=["shift-1", "fullsample-z", "centred", "bfill", "macro-by-date"])
def test_checker_flags_leaky_features(fn, temp_feature, market: MarketData,
                                      alternatives: dict[int, list[MarketData]]) -> None:
    name = temp_feature(f"__leaky_{fn.__name__}", fn)
    problems = leak_report(name, market, alternatives)
    assert problems, f"leakage checker failed to flag {fn.__name__}"


def test_negative_controls_are_cleaned_up() -> None:
    assert not [n for n in _REGISTRY if n.startswith("__leaky_")]


def test_causal_control_passes(temp_feature, market: MarketData,
                               alternatives: dict[int, list[MarketData]]) -> None:
    """Positive control: an obviously causal feature is not flagged (checker precision)."""
    def _ok(md: MarketData) -> pd.DataFrame:
        c = md.bars["close"]
        return pd.DataFrame({"ok_ma": c.rolling(10).mean() / c}, index=md.bars.index)

    name = temp_feature("__causal_control", _ok)
    assert leak_report(name, market, alternatives) == []


# ---- reviewer: boundary cutoffs & other timeframes -------------------------------------------
#: Groups whose logic depends on the bar timeframe or on day/week boundaries.
CALENDAR_SENSITIVE = ("mtf", "macro", "microstructure", "volatility", "session", "calendar")


def boundary_cutoffs(bars: pd.DataFrame, *, min_t: int = 8) -> list[int]:
    """Cutoffs where PIT bugs hide: the very first bars (schema must not depend on history
    length), the last bar before a weekend, the first bar after it, and the bars closing at
    the 22:00 UTC daily anchor and one hour before it (D1 bucket completeness)."""
    idx = bars.index
    avail = pd.DatetimeIndex(bars["available_at"])
    step = np.diff(idx.asi8)
    gap_after = np.flatnonzero(step > 2 * np.median(step))
    picks = [min_t]
    if gap_after.size:
        picks += [int(gap_after[0]), int(gap_after[0]) + 1, int(gap_after[-1]), int(gap_after[-1]) + 1]
    for hour in (21, 22):
        at = np.flatnonzero((avail.hour == hour) & (avail.minute == 0))
        if at.size:
            picks.append(int(at[len(at) // 2]))
    return sorted({t for t in picks if min_t <= t < len(bars) - 2})


@pytest.mark.parametrize("name", CALENDAR_SENSITIVE)
def test_boundary_cutoffs_h1(name: str, market: MarketData) -> None:
    cuts = boundary_cutoffs(market.bars)
    alts = {t: [perturbed_market(market, t, seed=50 + i), truncated_market(market, t)]
            for i, t in enumerate(cuts)}
    problems = leak_report(name, market, alts)
    assert not problems, "\n".join(problems[:20])


@pytest.fixture(scope="module")
def market_m30() -> MarketData:
    bars = make_synthetic_bars(3600, "M30", seed=8, model="regime")
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=8),
                      events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=14)))


def _alt_tf(md: MarketData, t: int, seed: int) -> MarketData:
    """``perturbed_market`` for any timeframe: same timeline, unrelated bars after ``t`` and
    sign-flipped (partly non-positive) macro values not yet available at ``available_at[t]``."""
    tf = md.bars.attrs["timeframe"]
    alt = make_synthetic_bars(len(md.bars), tf, seed=2000 + seed, model="jump",
                              start_price=float(md.bars["close"].iloc[t]) * 0.61, annual_vol=0.5)
    assert alt.index.equals(md.bars.index)
    new = pd.concat([md.bars.iloc[: t + 1], alt.iloc[t + 1:]])
    new.attrs["timeframe"] = tf
    cutoff_time = md.bars["available_at"].iloc[t]
    macro = {}
    for k, v in md.macro.items():
        future = (pd.DatetimeIndex(v["available_at"]) > cutoff_time)
        vals = v["value"].to_numpy(dtype=float).copy()
        vals[future] = -vals[future]
        macro[k] = v.assign(value=vals)
    return MarketData(bars=new, macro=macro, events=md.events)


@pytest.mark.parametrize("name", CALENDAR_SENSITIVE)
def test_boundary_cutoffs_m30(name: str, market_m30: MarketData) -> None:
    md = market_m30
    cuts = boundary_cutoffs(md.bars) + [2500]  # 2500: every daily-based column is warm
    alts = {t: [_alt_tf(md, t, seed=i), truncated_market(md, t)] for i, t in enumerate(cuts)}
    problems = leak_report(name, md, alts)
    assert not problems, "\n".join(problems[:20])


# ---- independent review: tiny histories ---------------------------------------------------------
@pytest.mark.parametrize("name", ALL_GROUPS)
def test_schema_and_values_on_tiny_histories(name: str, market: MarketData) -> None:
    """A history of 0-3 bars (live start-up, first bars of a fold) must give the SAME
    columns as the full history and the full history's first rows (NaN == NaN) — a group
    that drops columns until data arrives breaks ``FeaturePipeline.transform`` in live."""
    spec = get_feature(name)
    params = GROUP_TEST_PARAMS.get(name, {})
    full = spec.compute(market, **params)
    for k in (0, 1, 2, 3):
        if k:
            short = truncated_market(market, k - 1)
        else:
            first_open = market.bars.index[0]
            short = MarketData(bars=market.bars.iloc[:0],
                               macro={n: v.loc[pd.DatetimeIndex(v["available_at"]) <= first_open]
                                      for n, v in market.macro.items()},
                               events=market.events)
        out = spec.compute(short, **params)
        assert list(out.columns) == list(full.columns), (name, k)
        assert out.index.equals(market.bars.index[:k])
        if k:
            assert not compare_prefix(full, out, k - 1, f"{name}@k={k}")
