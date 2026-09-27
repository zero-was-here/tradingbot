"""FeaturePipeline (SPEC §4.1): compute, train-only fit, transform semantics, persistence,
schema enforcement, parity report, and the 100k-bar performance budget."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.features.pipeline import IQR_TO_SIGMA, FeaturePipeline, FeatureSchemaError


def _market(n: int, seed: int = 0) -> MarketData:
    bars = make_synthetic_bars(n, "H1", seed=seed, model="regime")
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=seed),
                      events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7)))


@pytest.fixture(scope="module")
def md() -> MarketData:
    return _market(4000, seed=1)


@pytest.fixture(scope="module")
def raw(md: MarketData) -> pd.DataFrame:
    return FeaturePipeline().compute(md)


def test_default_groups_and_compute(md: MarketData, raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline()
    assert {"returns", "trend", "momentum", "meanrev", "range", "volatility", "microstructure",
            "session", "mtf", "macro", "calendar", "regime"} <= set(pipe.groups)
    assert raw.index.equals(md.bars.index)
    assert raw.columns.is_unique and all(raw.dtypes == np.float64)
    assert not np.isinf(raw.to_numpy()).any()
    assert pipe.max_lookback >= 480


def test_fit_uses_train_only_and_robust_stats(raw: pd.DataFrame) -> None:
    train = raw.iloc[:2500]
    pipe = FeaturePipeline().fit(train)
    st = pipe.stats
    col = "returns_z_12"
    v = train[col].dropna()
    assert st.loc[col, "kind"] == "robust"
    assert st.loc[col, "loc"] == pytest.approx(v.median())
    assert st.loc[col, "scale"] == pytest.approx((v.quantile(0.75) - v.quantile(0.25)) * IQR_TO_SIGMA)
    # changing the test period must not change the fitted statistics
    raw2 = raw.copy()
    raw2.iloc[2500:] = raw2.iloc[2500:] * 10 + 3
    pipe2 = FeaturePipeline().fit(raw2.iloc[:2500])
    pd.testing.assert_frame_equal(pipe.stats, pipe2.stats)


def test_transform_scales_clips_and_fills(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline(clip=3.0)
    x = pipe.fit_transform(raw.iloc[:3000])
    assert list(x.columns) == pipe.columns
    vals = x.to_numpy()
    finite = vals[~np.isnan(vals)]
    assert finite.min() >= -3.0 and finite.max() <= 3.0
    lb = pipe.max_lookback
    assert not x.iloc[lb:].isna().any().any(), "post-warm-up NaN must be filled with 0"
    assert x.iloc[:50].isna().any().any(), "warm-up NaN must be preserved"
    robust = [c for c, k in zip(pipe.columns, pipe.stats["kind"], strict=True) if k == "robust"]
    # Over the rows where the raw value exists, scaled training data has median exactly 0
    # (clipping is monotone, so it preserves the median).
    xs = x[robust].to_numpy(copy=True)
    xs[raw.iloc[:3000][robust].isna().to_numpy()] = np.nan
    np.testing.assert_allclose(np.nanmedian(xs, axis=0), 0.0, atol=1e-9)


def test_discrete_columns_pass_through(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline().fit(raw.iloc[:3000])
    assert pipe.stats.loc["session_london", "kind"] == "discrete"
    x = pipe.transform(raw)
    np.testing.assert_array_equal(x["session_london"].to_numpy(), raw["session_london"].to_numpy())


def test_constant_and_empty_columns_dropped(raw: pd.DataFrame) -> None:
    r = raw.iloc[:3000].copy()
    r["returns_const"] = 1.2345
    r["returns_empty"] = np.nan
    pipe = FeaturePipeline().fit(r)
    assert {"returns_const", "returns_empty"} <= set(pipe.dropped_columns)
    assert "returns_const" not in pipe.columns
    out = pipe.transform(r)  # dropped columns present at transform time: silently ignored
    assert "returns_const" not in out.columns


def test_transform_raises_on_missing_and_unseen_columns(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline().fit(raw.iloc[:3000])
    with pytest.raises(FeatureSchemaError, match="missing"):
        pipe.transform(raw.drop(columns=["trend_adx_14"]))
    extra = raw.assign(brand_new_feature=1.0)
    with pytest.raises(FeatureSchemaError, match="not seen"):
        pipe.transform(extra)
    out = pipe.transform(extra, strict=False)
    assert "brand_new_feature" not in out.columns
    with pytest.raises(RuntimeError):
        FeaturePipeline().transform(raw)


def test_transform_is_causal(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline().fit(raw.iloc[:2500])
    a = pipe.transform(raw)
    r2 = raw.copy()
    r2.iloc[3000:] = np.nan  # the future disappears / changes
    b = pipe.transform(r2)
    pd.testing.assert_frame_equal(a.iloc[:3000], b.iloc[:3000])


def test_transform_on_later_slice_has_no_nan(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline().fit(raw.iloc[:2500])
    test_slice = raw.iloc[3000:3200]
    out = pipe.transform(test_slice)
    assert not out.isna().any().any()


def test_json_round_trip(tmp_path, md: MarketData) -> None:
    pipe = FeaturePipeline(groups=["returns", "trend", "session", "calendar", "macro"],
                           overrides={"returns": {"horizons": (1, 3, 9)}}, clip=4.0)
    raw = pipe.compute(md)
    assert "returns_z_9" in raw.columns
    pipe.fit(raw.iloc[:3000])
    path = pipe.save(tmp_path / "sub" / "pipe.json")
    payload = json.loads(path.read_text())
    assert payload["version"] == 1 and payload["columns"] == pipe.columns
    assert payload["overrides"] == {"returns": {"horizons": [1, 3, 9]}}
    assert set(payload["stats"]) == set(pipe.columns)
    loaded = FeaturePipeline.load(path)
    assert loaded.groups == pipe.groups and loaded.overrides == {"returns": {"horizons": (1, 3, 9)}}
    assert loaded.clip == 4.0 and loaded.max_lookback == pipe.max_lookback and loaded.is_fitted
    raw2 = loaded.compute(md)
    pd.testing.assert_frame_equal(raw, raw2, check_exact=True)
    pd.testing.assert_frame_equal(pipe.transform(raw), loaded.transform(raw2), check_exact=True)
    pd.testing.assert_frame_equal(pipe.stats, loaded.stats)
    # unfitted pipelines round-trip too
    FeaturePipeline(groups=["session"]).save(tmp_path / "u.json")
    assert not FeaturePipeline.load(tmp_path / "u.json").is_fitted


def test_overrides_flow_into_compute_and_lookback(md: MarketData) -> None:
    pipe = FeaturePipeline(groups=["returns", "session"], overrides={"returns": {"horizons": (1, 300)}})
    raw = pipe.compute(md)
    assert "returns_z_300" in raw.columns and "returns_z_48" not in raw.columns
    assert pipe.max_lookback == 301
    assert FeaturePipeline(groups=["session"], warmup=7).max_lookback == 7


def test_constructor_validation() -> None:
    with pytest.raises(KeyError):
        FeaturePipeline(groups=["no_such_group"])
    with pytest.raises(ValueError):
        FeaturePipeline(groups=["returns"], overrides={"trend": {}})
    with pytest.raises(ValueError):
        FeaturePipeline(scaler="minmax")
    with pytest.raises(ValueError):
        FeaturePipeline(clip=0)


def test_requirements_skip_groups(md: MarketData) -> None:
    pipe = FeaturePipeline(groups=["calendar", "macro", "session"])
    raw = pipe.compute(MarketData(bars=md.bars))
    assert not any(c.startswith(("calendar_", "macro_")) for c in raw.columns)
    assert any(c.startswith("session_") for c in raw.columns)


def test_parity_report(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline()
    rep = pipe.parity_report(raw, raw.copy())
    assert rep["ok"].all() and (rep["n_mismatch"] == 0).all()
    b = raw.copy()
    b.iloc[100, b.columns.get_loc("trend_adx_14")] += 1e-3
    b = b.drop(columns=["session_asia"]).assign(extra_col=0.0)
    rep = pipe.parity_report(raw, b, atol=1e-8)
    assert not rep.loc["trend_adx_14", "ok"] and rep.loc["trend_adx_14", "n_mismatch"] == 1
    assert rep.loc["trend_adx_14", "max_abs_diff"] == pytest.approx(1e-3)
    assert not rep.loc["session_asia", "in_b"] and not rep.loc["extra_col", "in_a"]
    assert rep.drop(index=["trend_adx_14", "session_asia", "extra_col"])["ok"].all()


def test_snapshot_is_json_safe(raw: pd.DataFrame) -> None:
    pipe = FeaturePipeline()
    snap = pipe.snapshot(raw.iloc[:10])  # warm-up row: contains NaN -> None
    assert None in snap.values()
    json.dumps(snap)


def test_performance_100k_bars() -> None:
    """Budget: all groups on 100k H1 bars in < 20 s. Measured as process CPU time so the
    check is meaningful on a shared, heavily loaded CI box (wall time is reported)."""
    md100 = _market(100_000, seed=5)
    pipe = FeaturePipeline()
    w0, c0 = time.perf_counter(), time.process_time()
    raw = pipe.compute(md100)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    assert raw.shape[0] == 100_000 and raw.shape[1] > 150
    assert cpu < 20.0, f"all groups on 100k H1 bars took {cpu:.1f}s CPU ({wall:.1f}s wall)"


# ---- reviewer: adversarial cases -------------------------------------------------------------
def _market_tf(n: int, tf: str, seed: int = 3) -> MarketData:
    bars = make_synthetic_bars(n, tf, seed=seed)
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=seed))


@pytest.mark.parametrize(("tf", "n", "groups"), [
    ("H1", 2500, ("mtf", "macro", "returns")),
    ("H1", 1200, ("mtf", "returns")),
    ("M30", 4500, ("mtf", "macro", "returns")),
    ("H4", 900, ("mtf", "macro", "returns")),
])
def test_warmup_nan_is_never_zero_filled(tf: str, n: int, groups: tuple[str, ...]) -> None:
    """'NaN -> 0 only after warm-up': on a full history, every row before a column's first
    valid raw value is warm-up and must stay NaN, on ANY bar timeframe. Daily-based groups
    (macro, mtf) need ~65 / ~26 trading days whatever the bar size, so a fixed H1 bar count
    would zero-fill genuine warm-up rows on M30/M15/M5 (and mtf's D1 EMA slope on H1)."""
    md = _market_tf(n, tf)
    pipe = FeaturePipeline(groups=list(groups))
    raw = pipe.compute(md)
    x = pipe.fit(raw.iloc[n // 2:]).transform(raw)
    for c in x.columns:
        first = int(raw[c].notna().to_numpy().argmax())
        assert first <= pipe.max_lookback, (tf, c, first, pipe.max_lookback)
        assert x[c].iloc[:first].isna().all(), (tf, c)
        assert x[c].iloc[pipe.max_lookback:].notna().all(), (tf, c)


def test_non_window_overrides_do_not_inflate_lookback() -> None:
    base = FeaturePipeline(groups=["volatility", "calendar", "session", "regime"]).max_lookback
    pipe = FeaturePipeline(
        groups=["volatility", "calendar", "session", "regime"],
        overrides={"volatility": {"bars_per_year": 5796.0},
                   "calendar": {"cap_hours": 500.0, "count_horizon_hours": 168.0},
                   "session": {"ny_hours": (8.0, 17.0)},
                   "regime": {"high_vol_pct": 0.9}})
    assert pipe.max_lookback == base
    # genuine window overrides still extend the warm-up
    assert FeaturePipeline(groups=["volatility"],
                           overrides={"volatility": {"long_window": 2000}}).max_lookback == 2001


def test_lookback_follows_bar_timeframe_and_round_trips(tmp_path) -> None:
    pipe = FeaturePipeline(groups=["macro", "returns"])
    h1_lb = pipe.max_lookback
    md = _market_tf(400, "M15")
    pipe.compute(md)
    assert pipe.bar_minutes == 15.0
    assert pipe.max_lookback == 4 * h1_lb
    pipe.fit(pipe.compute(md))
    loaded = FeaturePipeline.load(pipe.save(tmp_path / "p.json"))
    assert loaded.bar_minutes == 15.0 and loaded.max_lookback == pipe.max_lookback
    assert FeaturePipeline(groups=["macro"], bar_minutes=1440).max_lookback == h1_lb // 24


def test_default_group_order_is_import_order_independent() -> None:
    """The default group (hence raw column) order must not depend on which feature module
    happened to be imported first in the process (research vs live runner)."""
    from aurum.features import base

    reference = FeaturePipeline().groups
    saved = dict(base._REGISTRY)
    try:
        items = list(saved.items())
        base._REGISTRY.clear()
        base._REGISTRY.update(dict(reversed(items)))
        assert FeaturePipeline().groups == reference
    finally:
        base._REGISTRY.clear()
        base._REGISTRY.update(saved)
    assert reference[:5] == ["returns", "trend", "momentum", "meanrev", "range"]


def test_compute_on_empty_bars(md: MarketData) -> None:
    pipe = FeaturePipeline()
    empty = MarketData(bars=md.bars.iloc[:0], macro=md.macro, events=md.events)
    out = pipe.compute(empty)
    assert out.shape[0] == 0 and out.index.equals(md.bars.index[:0])


# ---- independent review: warm-up exactness, bar size, slicing ---------------------------------
#: Window overrides that move each group's warm-up well away from its default.
_WINDOW_OVERRIDES: dict[str, dict] = {
    "returns": {"horizons": (1, 100)},
    "trend": {"ema_spans": (10, 300), "ema_pairs": ((10, 300),), "linreg_windows": (150,)},
    "momentum": {"tsmom_horizons": (24, 600), "stoch_n": 50},
    "meanrev": {"z_windows": (200,), "bb_n": 30},
    "range": {"donchian": (100,), "nr_window": 12},
    "volatility": {"windows": (10, 50), "long_window": 200, "volofvol_window": 300},
    "microstructure": {"autocorr_window": 300, "z_window": 50},
    "regime": {"rank_min_periods": 500, "vr_window": 100, "vr_qs": (4, 32)},
    "mtf": {"ema_span": 40, "donchian_n": 30},
    "macro": {"z_min_periods": 100, "change_days": (1, 60)},
}
_BAR_BASED = ("returns", "trend", "momentum", "meanrev", "range", "volatility", "microstructure",
              "regime")


def _first_valid(raw: pd.DataFrame) -> pd.Series:
    nn = raw.notna()
    return pd.Series(np.where(nn.any().to_numpy(), nn.to_numpy().argmax(axis=0), -1),
                     index=raw.columns)


@pytest.mark.parametrize(("tf", "n", "gaps"), [
    ("H1", 6000, True), ("M15", 12000, True), ("H4", 1500, True), ("D1", 700, True),
    ("D1", 700, False),  # daily bars on every calendar day (e.g. broker Sunday D1 bars)
])
@pytest.mark.parametrize("variant", ["default", "override"])
def test_group_lookback_covers_first_valid_value(tf: str, n: int, gaps: bool, variant: str) -> None:
    """For every group, timeframe and window override, every column is defined from row
    ``group_lookback`` on — otherwise ``transform`` zero-fills genuine warm-up rows."""
    bars = make_synthetic_bars(n, tf, seed=4, weekend_gaps=gaps)
    md = MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=4),
                    events=make_synthetic_events(bars.index[0], bars.index[-1]))
    for g in FeaturePipeline().groups:
        ov = _WINDOW_OVERRIDES.get(g) if variant == "override" else None
        pipe = FeaturePipeline(groups=[g], overrides={g: ov} if ov else None)
        raw = pipe.compute(md)
        lb = pipe.group_lookback(g)
        assert lb == pipe.max_lookback
        first = _first_valid(raw)
        if lb < n:
            assert (first >= 0).all(), (tf, g, variant, list(first[first < 0].index))
        assert (first <= lb).all(), (tf, g, variant, first[first > lb].to_dict(), lb)
        if g in _BAR_BASED and tf == "H1":
            # bar-based warm-ups are exact (not just an upper bound)
            assert lb - int(first.max()) <= 2, (g, variant, lb, int(first.max()))


def test_registered_lookback_matches_lookback_fn_at_h1() -> None:
    """The static ``FeatureSpec.lookback`` must agree with the dynamic warm-up at H1."""
    from aurum.features.base import list_features
    from aurum.features.pipeline import _effective_params

    for spec in list_features():
        fn = getattr(spec.fn, "lookback_fn", None)
        assert callable(fn), f"{spec.name} has no lookback_fn"
        assert fn(_effective_params(spec, {}), 60.0) == spec.lookback, spec.name


def test_bar_minutes_validation_and_mismatch_warning(md: MarketData, caplog) -> None:
    with pytest.raises(ValueError):
        FeaturePipeline(groups=["returns"], bar_minutes=0)
    pipe = FeaturePipeline(groups=["returns", "session"])
    pipe.fit(pipe.compute(md))
    assert pipe.bar_minutes == 60.0
    m30 = make_synthetic_bars(300, "M30", seed=1)
    with caplog.at_level("WARNING", logger="aurum.features.pipeline"):
        pipe.compute(MarketData(bars=m30))
    assert any("30-minute" in r.getMessage() for r in caplog.records)


def test_transform_of_slice_equals_slice_of_transform(raw: pd.DataFrame) -> None:
    """Walk-forward usage: transforming a post-warm-up test fold on its own gives exactly
    the rows of the full transform (row-wise scaling, no hidden state from earlier rows)."""
    pipe = FeaturePipeline().fit(raw.iloc[:2500])
    a = 2700
    assert raw.iloc[a].notna().all()
    pd.testing.assert_frame_equal(pipe.transform(raw.iloc[a:]), pipe.transform(raw).iloc[a:])


def test_fallback_lookback_for_groups_without_lookback_fn() -> None:
    """Third-party groups without a ``lookback_fn``: registered lookback, extended only by
    INTEGER overrides (floats such as bars_per_year or thresholds are not windows)."""
    from aurum.features.base import _REGISTRY, register_feature

    def _custom(md: MarketData, *, window: int = 10, scale: float = 2.0) -> pd.DataFrame:
        c = md.bars["close"]
        return pd.DataFrame({"custom_ma": c.rolling(window).mean() / c * scale}, index=c.index)

    register_feature("__custom_lb", family="custom", lookback=11)(_custom)
    try:
        assert FeaturePipeline(groups=["__custom_lb"]).max_lookback == 11
        assert FeaturePipeline(groups=["__custom_lb"],
                               overrides={"__custom_lb": {"scale": 9000.0}}).max_lookback == 11
        assert FeaturePipeline(groups=["__custom_lb"],
                               overrides={"__custom_lb": {"window": 40}}).max_lookback == 41
    finally:
        _REGISTRY.pop("__custom_lb", None)
