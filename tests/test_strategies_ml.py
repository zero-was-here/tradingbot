"""Tests for aurum.strategies.ml (ml_gbm, meta_label).

Covers: forecast-mapping helpers, registry, point-in-time discipline (fit sees only the
training slice, no label beyond train end, perturbing/truncating the future after fit leaves
past forecasts unchanged), a random-walk negative control (OOS AUC ~0.5, gross Sharpe
t-stat < 2), positive controls with a planted predictable signal (OOS AUC > 0.55), meta-
labelling semantics (side from the primary, AFML bet sizing, discretisation, threshold) and
reporting. Models are tiny (few boosting iterations, 2-3 feature groups) to stay fast.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from aurum.core.types import MarketData
from aurum.data.schema import make_bars, validate_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.labels import ewm_vol, fixed_horizon_labels, meta_labels, triple_barrier_labels
from aurum.strategies.base import _STRATEGIES, Strategy
from aurum.strategies.ml import (
    CalibratedGBM,
    MetaLabelStrategy,
    MLGBMStrategy,
    apply_dead_zone,
    bet_size_from_probability,
    carver_scalar,
    discretize_bet,
)

GROUPS = ["returns", "trend", "meanrev"]   # warm-up ~205 H1 bars
FAST = {"max_iter": 40, "min_samples_leaf": 60, "learning_rate": 0.1}
COMMON = dict(feature_groups=GROUPS, model=FAST, importance_repeats=1, importance_max_rows=400,
              n_threads=2, seed=0)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
class MomentumStub(Strategy):
    """Causal primary for the tests (the real tsmom lives in aurum.strategies.trend)."""

    name = "_mom_stub"

    def __init__(self, lookback: int = 48, **params) -> None:
        super().__init__(lookback=lookback, **params)

    @property
    def warmup_bars(self) -> int:
        return int(self.params["lookback"])

    def generate(self, md, features=None):
        lc = np.log(md.bars["close"])
        return self._finalize(np.sign(lc - lc.shift(int(self.params["lookback"]))), md.bars.index)


class AlwaysLong(Strategy):
    name = "_always_long"

    def generate(self, md, features=None):
        return self._finalize(pd.Series(1.0, index=md.bars.index), md.bars.index)


def planted_market(n: int, seed: int, *, beta: float = 0.35, rho: float = 0.95
                   ) -> tuple[MarketData, pd.DataFrame]:
    """Bars whose next-bar drift is ``beta * sigma * z_t`` for a persistent AR(1) ``z`` that
    is known at the close of ``t`` and handed over as an (external) feature."""
    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = rng.standard_normal()
    eta = rng.standard_normal(n)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + math.sqrt(1 - rho**2) * eta[i]
    sigma = 0.002
    r = np.zeros(n)
    r[1:] = beta * sigma * z[:-1] + sigma * rng.standard_normal(n - 1)
    close = 1800.0 * np.exp(np.cumsum(r))
    open_ = np.r_[1800.0, close[:-1]]
    ext = np.abs(rng.normal(0, 0.4 * sigma, (2, n)))
    high = np.maximum(open_, close) * (1 + ext[0])
    low = np.minimum(open_, close) * (1 - ext[1])
    idx = pd.date_range("2021-01-04", periods=n, freq="h", tz="UTC")
    bars = make_bars(pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx),
                     "H1", default_spread=0.3)
    feats = pd.DataFrame({"planted": z, "noise_a": rng.standard_normal(n),
                          "noise_b": rng.standard_normal(n)}, index=idx)
    return MarketData(bars=bars), feats


def perturb_future(md: MarketData, t: int, seed: int) -> MarketData:
    """Same bars up to ``t`` (inclusive), an unrelated path afterwards (same timestamps)."""
    bars = md.bars
    alt = make_synthetic_bars(len(bars), "H1", seed=1000 + seed, model="jump",
                              start_price=float(bars["close"].iloc[t]) * 1.37, annual_vol=0.45, spread=0.9)
    alt.index = bars.index
    alt["available_at"] = bars["available_at"]
    new = pd.concat([bars.iloc[: t + 1], alt.iloc[t + 1:]])
    new.attrs["timeframe"] = "H1"
    validate_bars(new)
    return MarketData(bars=new)


def sharpe_tstat(forecast: pd.Series, close: pd.Series) -> float:
    """t-stat of the gross bar P&L ``f[t] * r[t+1]`` (decided at t, earned over t+1)."""
    r_next = np.log(close).diff().shift(-1)
    pnl = (forecast * r_next).dropna()
    pnl = pnl[forecast.reindex(pnl.index) != 0]
    if len(pnl) < 30 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * math.sqrt(len(pnl)))


# ---------------------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------------------
def test_dead_zone_and_carver_scalar():
    f = np.array([-1.0, -0.05, -0.01, 0.0, 0.015, 0.1, 1.0])
    out = apply_dead_zone(f, 0.02)
    assert out[2] == 0 and out[3] == 0 and out[4] == 0
    assert out[0] == -1 and out[-1] == 1
    assert out[5] == pytest.approx(0.08 / 0.98)
    assert np.all(np.diff(apply_dead_zone(np.linspace(-1, 1, 101), 0.1)) >= 0)  # monotone
    with pytest.raises(ValueError):
        apply_dead_zone(f, 1.0)
    x = np.array([0.1, -0.1, 0.2, -0.2])
    assert np.mean(np.abs(carver_scalar(x, 0.5) * x)) == pytest.approx(0.5)
    assert carver_scalar(x, 0.5, max_scalar=2.0) == 2.0
    assert carver_scalar(x, None) == 1.0
    assert carver_scalar(np.zeros(5), 0.5) == 0.0


def test_bet_size_and_discretisation():
    assert bet_size_from_probability(0.5) == pytest.approx(0.0, abs=1e-12)
    z = 0.1 / math.sqrt(0.24)
    from scipy.stats import norm

    assert bet_size_from_probability(0.6) == pytest.approx(2 * norm.cdf(z) - 1)
    p = np.linspace(0.01, 0.99, 99)
    m = bet_size_from_probability(p)
    assert np.all(np.diff(m) > 0) and np.all(np.abs(m) < 1)
    np.testing.assert_allclose(m, -bet_size_from_probability(1 - p), atol=1e-12)  # antisymmetric
    np.testing.assert_allclose(discretize_bet([0.04, 0.06, 0.26, 1.3, -0.74], 0.1), [0, 0.1, 0.3, 1.0, -0.7])
    np.testing.assert_allclose(discretize_bet([0.123], 0.0), [0.123])


def test_registered_and_untrained_generate_raises():
    assert _STRATEGIES["ml_gbm"] is MLGBMStrategy
    assert _STRATEGIES["meta_label"] is MetaLabelStrategy
    for cls in (MLGBMStrategy, MetaLabelStrategy):
        s = cls()
        assert s.trainable and not s.is_fitted
        with pytest.raises(RuntimeError):
            s.generate(MarketData(bars=make_synthetic_bars(50, "H1")))
    with pytest.raises(KeyError, match="not registered"):
        MetaLabelStrategy(primary="__no_such_strategy__").fit(MarketData(bars=make_synthetic_bars(600, "H1")))


# ---------------------------------------------------------------------------------------
# ml_gbm: point-in-time discipline
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trend_md() -> MarketData:
    return MarketData(bars=make_synthetic_bars(3200, "H1", seed=21, model="trend",
                                               regime_params={"phi": 0.3}))


@pytest.fixture(scope="module")
def fitted_trend_gbm(trend_md) -> MLGBMStrategy:
    s = MLGBMStrategy(**COMMON, target="fixed_horizon", max_holding_bars=1)
    return s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))


def test_ml_gbm_positive_control_trend(trend_md, fitted_trend_gbm):
    s = fitted_trend_gbm
    rep = s.fit_report_
    assert rep["skill_gate_passed"] and s.forecast_scalar_ > 0
    p = s.predict_proba(trend_md)
    lab = fixed_horizon_labels(trend_md.bars, 1)
    oos = lab[(lab["t_idx"] >= 2000) & (lab["label"] != 0)]
    auc = roc_auc_score(oos["label"] > 0, p.loc[oos.index])
    assert auc > 0.55, auc
    f = s.generate(trend_md)
    assert f.index.equals(trend_md.bars.index)
    assert f.between(-1, 1).all() and np.isfinite(f).all()
    assert (f.iloc[: s.warmup_bars] == 0).all()
    assert sharpe_tstat(f.iloc[2000:], trend_md.bars["close"].iloc[2000:]) > 2
    # the planted structure (1-bar autocorrelation) is found by the importance ranking
    assert s.feature_importances_.index[0].startswith("returns_")


def test_ml_gbm_leakage_future_perturbation(trend_md, fitted_trend_gbm):
    s = fitted_trend_gbm
    full = s.generate(trend_md)
    assert (full.iloc[2000:] != 0).mean() > 0.5
    for i, t in enumerate((2100, 2600, 3100)):
        alt = s.generate(perturb_future(trend_md, t, seed=i))
        pd.testing.assert_series_equal(alt.iloc[: t + 1], full.iloc[: t + 1])
        assert not alt.iloc[t + 1:].equals(full.iloc[t + 1:])
        trunc = s.generate(MarketData(bars=trend_md.bars.iloc[: t + 1]))
        pd.testing.assert_series_equal(trunc, full.iloc[: t + 1])


def test_ml_gbm_fit_uses_only_train_and_labels_end_inside_train(trend_md, fitted_trend_gbm):
    s = fitted_trend_gbm
    train_end = trend_md.bars.index[1999]
    lab = s.train_labels_
    assert lab["t1"].max() <= train_end
    assert lab["t_idx"].max() <= 2000 - 1 - 1
    assert s.fit_report_["train_end"] == train_end.isoformat()
    # triple-barrier variant: the last max_holding_bars events are purged as well
    tb = MLGBMStrategy(**COMMON, max_holding_bars=12, importance="none", skill_gate_z=None)
    tb.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert tb.train_labels_["t_idx"].max() <= 2000 - 1 - 12
    assert tb.train_labels_["t1_idx"].max() <= 1999
    # deterministic: an identical refit reproduces the forecasts exactly
    again = MLGBMStrategy(**COMMON, target="fixed_horizon", max_holding_bars=1)
    again.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    pd.testing.assert_series_equal(again.generate(trend_md), s.generate(trend_md))
    clone = s.clone()
    pd.testing.assert_series_equal(clone.generate(trend_md), s.generate(trend_md))


def test_external_features_rows_beyond_train_are_ignored():
    md, feats = planted_market(2400, seed=7)
    train = MarketData(bars=md.bars.iloc[:1600])
    kw = dict(COMMON, importance="none", max_holding_bars=12, skill_gate_z=None)
    a = MLGBMStrategy(**kw).fit(train, features=feats)             # full-history frame
    b = MLGBMStrategy(**kw).fit(train, features=feats.iloc[:1600])  # train rows only
    pd.testing.assert_series_equal(a.generate(md, features=feats), b.generate(md, features=feats))
    with pytest.raises(ValueError, match="externally supplied"):
        a.generate(md)


# ---------------------------------------------------------------------------------------
# ml_gbm: negative and positive controls
# ---------------------------------------------------------------------------------------
def test_ml_gbm_random_walk_has_no_edge():
    bars = make_synthetic_bars(5200, "H1", seed=33, model="gbm")
    md = MarketData(bars=bars)
    n_tr = 3200
    s = MLGBMStrategy(**COMMON, max_holding_bars=4, barrier_vol="horizon", skill_gate_z=None,
                      importance="none")
    s.fit(MarketData(bars=bars.iloc[:n_tr]))
    p = s.predict_proba(md)
    vol = ewm_vol(bars["close"]) * 2.0
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=4, vol=vol,
                                vertical_label="sign")
    oos = lab[(lab["t_idx"] >= n_tr) & (lab["label"] != 0)]
    oos = oos[p.loc[oos.index].notna()]
    auc = roc_auc_score(oos["label"] > 0, p.loc[oos.index])
    assert 0.45 < auc < 0.55, auc
    f = s.generate(md)
    assert (f.iloc[n_tr:] != 0).mean() > 0.3   # it does trade (gate disabled) ...
    assert sharpe_tstat(f.iloc[n_tr:], bars["close"].iloc[n_tr:]) < 2.0   # ... without an edge
    # with the default skill gate a no-skill fit stays flat instead of trading noise
    gated = MLGBMStrategy(**COMMON, max_holding_bars=4, importance="none")
    gated.fit(MarketData(bars=bars.iloc[:n_tr]))
    assert not gated.fit_report_["skill_gate_passed"]   # unconditional: a pass here is a bug
    assert gated.forecast_scalar_ == 0 and (gated.generate(md) == 0).all()


def test_ml_gbm_planted_feature_triple_barrier():
    md, feats = planted_market(3000, seed=1)
    n_tr = 2000
    s = MLGBMStrategy(**COMMON, max_holding_bars=12, calibration="isotonic",
                      sample_weight="uniqueness_x_return", time_decay=0.5)
    s.fit(MarketData(bars=md.bars.iloc[:n_tr]), features=feats)
    assert s.fit_report_["skill_gate_passed"]
    assert s.feature_importances_.index[0] == "planted"
    assert s.importance_by_group().index[0] == "planted"
    p = s.predict_proba(md, features=feats)
    vol = ewm_vol(md.bars["close"]) * math.sqrt(12)
    lab = triple_barrier_labels(md.bars, pt_mult=1, sl_mult=1, max_holding_bars=12, vol=vol,
                                vertical_label="sign")
    oos = lab[(lab["t_idx"] >= n_tr) & (lab["label"] != 0)]
    auc = roc_auc_score(oos["label"] > 0, p.loc[oos.index])
    assert auc > 0.55, auc
    f = s.generate(md, features=feats)
    assert np.corrcoef(f.iloc[n_tr:], feats["planted"].iloc[n_tr:])[0, 1] > 0.5
    info = s.explain()
    json.dumps(info)
    assert info["fit_report"]["n_labels"] == len(s.train_labels_)
    assert "planted" in info["top_features"]


def test_ml_gbm_primary_features_and_event_filters(trend_md):
    s = MLGBMStrategy(**COMMON, primary_features=[MomentumStub(lookback=24)], event_filter="cusum",
                      cusum_mult=1.0, importance="none", skill_gate_z=None, max_holding_bars=6,
                      calibration="none")
    s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert "primary__mom_stub" in s._feature_columns
    assert len(s.train_labels_) < 2000 - s.warmup_bars
    f = s.generate(trend_md)
    assert f.between(-1, 1).all()
    strided = MLGBMStrategy(**COMMON, event_stride=3, importance="none", max_holding_bars=6)
    strided.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert np.all(np.diff(strided.train_labels_["t_idx"].to_numpy()) % 3 == 0)


def test_not_enough_events_raises():
    bars = make_synthetic_bars(400, "H1", seed=2)
    with pytest.raises(ValueError, match="not enough training events"):
        MLGBMStrategy(**COMMON).fit(MarketData(bars=bars))


@pytest.mark.parametrize("class_weight, expect", [("balanced", 0.5), (None, 0.7)])
def test_calibration_does_not_relearn_the_validation_drift(class_weight, expect):
    """Noise features, 30% positives in the fit part and 70% in the validation tail: with
    balancing the calibrated probability stays at 0.5 (no drift bet); without it, the
    calibrator learns the tail's base rate."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("abc"))
    y = np.r_[(rng.random(2400) < 0.3), (rng.random(600) < 0.7)].astype(float)
    t = np.arange(n)
    clf = CalibratedGBM(model=FAST, class_weight=class_weight, val_frac=0.2, n_threads=2)
    clf.fit(X, y, np.ones(n), t, t + 1)
    p_val, _, _ = clf.validation_proba()
    assert p_val.mean() == pytest.approx(expect, abs=0.05)
    assert 0.4 < clf.report_["val_auc"] < 0.6


def test_drop_features_patterns(trend_md):
    s = MLGBMStrategy(**COMMON, drop_features=(r"^returns_", r"_200$"), importance="none",
                      skill_gate_z=None, max_holding_bars=4)
    s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    cols = s._feature_columns
    assert cols and not any(c.startswith("returns_") or c.endswith("_200") for c in cols)
    assert s.generate(trend_md).between(-1, 1).all()


def test_calibrated_gbm_requires_chronological_events():
    X = pd.DataFrame({"a": np.arange(10.0)})
    with pytest.raises(ValueError, match="chronological"):
        CalibratedGBM(min_events=2).fit(X, np.r_[np.zeros(5), np.ones(5)], np.ones(10),
                                        np.arange(10)[::-1], np.arange(10)[::-1] + 1)


# ---------------------------------------------------------------------------------------
# meta_label
# ---------------------------------------------------------------------------------------
def test_meta_label_planted_positive_control():
    md, feats = planted_market(3000, seed=4)
    n_tr = 2000
    s = MetaLabelStrategy(**COMMON, primary=AlwaysLong(), max_holding_bars=12)
    s.fit(MarketData(bars=md.bars.iloc[:n_tr]), features=feats)
    rep = s.fit_report_
    assert rep["skill_gate_passed"] and rep["primary"]["strategy"] == "_always_long"
    assert s.train_labels_["t1_idx"].max() <= n_tr - 1
    p = s.predict_proba(md, features=feats)
    vol = ewm_vol(md.bars["close"]) * math.sqrt(12)
    lab = triple_barrier_labels(md.bars, pt_mult=1, sl_mult=1, max_holding_bars=12, vol=vol, side=1.0)
    y = meta_labels(lab, vertical="return_sign")
    oos = lab.index[(lab["t_idx"] >= n_tr).to_numpy() & y.notna().to_numpy()]
    auc = roc_auc_score(y.loc[oos], p.loc[oos])
    assert auc > 0.55, auc
    f = s.generate(md, features=feats)
    assert (f >= 0).all()                       # never reverses an always-long primary
    active = f[f != 0]
    assert len(active) > 100
    np.testing.assert_allclose(active * 10, np.round(active * 10), atol=1e-9)  # 0.1 grid
    # bets are taken when the planted drift is favourable
    assert feats["planted"].loc[active.index].mean() > 0.3


def test_meta_label_side_threshold_and_scaling(trend_md):
    base = dict(COMMON, primary=MomentumStub(lookback=48), max_holding_bars=8, importance="none",
                skill_gate_z=None)
    s = MetaLabelStrategy(**base).fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    f = s.generate(trend_md)
    side = MomentumStub(lookback=48).generate(trend_md)
    nz = f != 0
    assert nz.any()
    assert (np.sign(f[nz]) == side[nz]).all()   # side always comes from the primary
    assert f.between(-1, 1).all()
    # training events were the primary's bets only
    assert (s.train_labels_["side"] != 0).all()
    np.testing.assert_array_equal(s.train_labels_["side"],
                                  side.iloc[s.train_labels_["t_idx"].to_numpy()].to_numpy())
    # an unreachable probability threshold switches every bet off
    off = MetaLabelStrategy(**{**base, "p_threshold": 0.999}).fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert (off.generate(trend_md) == 0).all()
    # pure AFML sizing (no Carver scaling), continuous sizes, scaled by |primary|
    raw = MetaLabelStrategy(**{**base, "target_abs_forecast": None, "step_size": 0.0,
                               "scale_by_primary": True, "event_filter": "flip",
                               "min_train_events": 40, "val_frac": 0.3})
    raw.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert raw.forecast_scalar_ == 1.0
    g = raw.generate(trend_md)
    assert g.abs().max() < 1.0
    flips = raw.train_labels_["t_idx"].to_numpy()
    assert (side.iloc[flips].to_numpy() != side.iloc[flips - 1].to_numpy()).all()


def test_meta_label_leakage_future_perturbation(trend_md):
    s = MetaLabelStrategy(**COMMON, primary=MomentumStub(lookback=48), max_holding_bars=8,
                          importance="none", skill_gate_z=None)
    s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    full = s.generate(trend_md)
    assert (full.iloc[2000:] != 0).any()
    for i, t in enumerate((2300, 3000)):
        alt = s.generate(perturb_future(trend_md, t, seed=10 + i))
        pd.testing.assert_series_equal(alt.iloc[: t + 1], full.iloc[: t + 1])
        trunc = s.generate(MarketData(bars=trend_md.bars.iloc[: t + 1]))
        pd.testing.assert_series_equal(trunc, full.iloc[: t + 1])


def test_meta_label_random_walk_has_no_edge():
    bars = make_synthetic_bars(5200, "H1", seed=44, model="gbm")
    md = MarketData(bars=bars)
    n_tr = 3200
    s = MetaLabelStrategy(**COMMON, primary=MomentumStub(lookback=24), max_holding_bars=4,
                          importance="none", skill_gate_z=None)
    s.fit(MarketData(bars=bars.iloc[:n_tr]))
    p = s.predict_proba(md)
    side = MomentumStub(lookback=24).generate(md)
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=4,
                                vol=ewm_vol(bars["close"]) * 2.0, side=side)
    y = meta_labels(lab, vertical="return_sign")
    oos = lab.index[(lab["t_idx"] >= n_tr).to_numpy() & y.notna().to_numpy()]
    oos = oos[p.loc[oos].notna().to_numpy()]
    auc = roc_auc_score(y.loc[oos], p.loc[oos])
    assert 0.45 < auc < 0.55, auc
    f = s.generate(md)
    assert sharpe_tstat(f.iloc[n_tr:], bars["close"].iloc[n_tr:]) < 2.0


def test_meta_label_default_primary_is_registry_tsmom(trend_md):
    """Integration with the real primary (skipped while aurum.strategies.trend is absent)."""
    pytest.importorskip("aurum.strategies.trend")
    s = MetaLabelStrategy(**COMMON, primary_params={"horizons": (24, 96, 240)}, max_holding_bars=8,
                          importance="none", skill_gate_z=None)
    assert s.params["primary"] == "tsmom"
    s.fit(MarketData(bars=trend_md.bars.iloc[:2200]))
    assert s.fit_report_["primary"]["strategy"] == "tsmom"
    assert s.label_horizon == 8 and s.warmup_bars >= 241
    f = s.generate(trend_md)
    from aurum.strategies.base import get_strategy

    side = np.sign(get_strategy("tsmom", horizons=(24, 96, 240)).generate(trend_md))
    nz = f != 0
    assert nz.any() and (np.sign(f[nz]) == side[nz]).all()


# ---------------------------------------------------------------------------------------
# adversarial review: warm-ups, transactional refits, trainable primaries, macro leakage
# ---------------------------------------------------------------------------------------
class MemorizingPrimary(Strategy):
    """An over-fitted TRAINABLE primary: on its training bars it replays the realised
    6-bar-ahead direction (perfect in sample), elsewhere a 1-bar momentum sign (no edge on a
    random walk). ``generate`` never reads the future of the ``md`` it is given."""

    name = "_memorizer"
    trainable = True

    def fit(self, md, features=None):
        lc = np.log(md.bars["close"])
        self.memo_ = np.sign(lc.shift(-6) - lc).dropna()
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        lc = np.log(md.bars["close"])
        rule = np.sign(lc - lc.shift(1))
        memo = self.memo_.reindex(md.bars.index)
        return self._finalize(memo.where(memo.notna() & (memo != 0), rule), md.bars.index)


def test_external_features_warmup_rows_are_neither_events_nor_forecasts(trend_md):
    """Walk-forward ``features.enabled=true`` hands over pipeline features whose first rows
    are still NaN (warm-up). They must not be zero-filled into training events or forecasts."""
    from aurum.features.pipeline import FeaturePipeline

    pipe = FeaturePipeline(groups=GROUPS)
    raw = pipe.compute(trend_md)
    first_valid = int(np.argmax(raw.notna().all(axis=1).to_numpy()))
    assert first_valid > 100
    pipe.fit(raw.iloc[:2000])
    for feats in (pipe.transform(raw), raw):          # scaled fold features and raw features
        s = MLGBMStrategy(**COMMON, importance="none", skill_gate_z=None, max_holding_bars=6)
        s.fit(MarketData(bars=trend_md.bars.iloc[:2000]), features=feats.iloc[:2000])
        assert s.train_labels_["t_idx"].min() >= first_valid
        f = s.generate(trend_md, features=feats)
        assert (f.iloc[:first_valid] == 0).all()
        assert (f.iloc[first_valid:] != 0).mean() > 0.3
    m = MetaLabelStrategy(**COMMON, primary=MomentumStub(lookback=24), importance="none",
                          skill_gate_z=None, max_holding_bars=6)
    m.fit(MarketData(bars=trend_md.bars.iloc[:2000]), features=raw.iloc[:2000])
    assert m.train_labels_["t_idx"].min() >= first_valid
    assert (m.generate(trend_md, features=raw).iloc[:first_valid] == 0).all()


def test_primary_feature_warmup_is_excluded_from_training_events(trend_md):
    s = MLGBMStrategy(**COMMON, primary_features=[MomentumStub(lookback=400)], importance="none",
                      skill_gate_z=None, max_holding_bars=6)
    s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert s.warmup_bars == 400
    assert s.train_labels_["t_idx"].min() >= 400   # not the pipeline's ~205-bar warm-up


def test_invalid_options_are_rejected_before_any_work(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:2000])
    with pytest.raises(ValueError, match="event_filter"):
        MLGBMStrategy(**COMMON, event_filter="flip").fit(md)   # only meta_label has "flip"
    with pytest.raises(ValueError, match="target"):
        MLGBMStrategy(**COMMON, target="nope").fit(md)
    with pytest.raises(ValueError, match="sample_weight"):
        MetaLabelStrategy(**COMMON, sample_weight="nope").fit(md)
    with pytest.raises(ValueError, match="primary_oos_frac"):
        MetaLabelStrategy(**COMMON, primary_oos_frac=1.0).fit(md)


def test_failed_refit_leaves_the_previous_fit_intact(trend_md):
    """A refit that raises must not leave a new scaler paired with the old model (that
    silently flipped forecasts) - the previous fit stays complete and self-consistent."""
    kw = dict(COMMON, skill_gate_z=None, max_holding_bars=6)
    s = MLGBMStrategy(**kw).fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    ref = s.generate(trend_md)
    report = dict(s.fit_report_)
    with pytest.raises(ValueError, match="not enough training events"):
        s.fit(MarketData(bars=trend_md.bars.iloc[1500:1900]))
    assert s.is_fitted and s.fit_report_ == report
    pd.testing.assert_series_equal(s.generate(trend_md), ref)
    # a first fit that fails leaves the strategy unfitted
    fresh = MetaLabelStrategy(**kw, primary=MomentumStub(lookback=24))
    with pytest.raises(ValueError):
        fresh.fit(MarketData(bars=trend_md.bars.iloc[:400]))
    assert not fresh.is_fitted
    with pytest.raises(RuntimeError):
        fresh.generate(trend_md)
    # nothing stale survives a successful refit (importances of the previous fit)
    s.params["importance"] = "none"
    s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    assert s.feature_importances_ is None and "top_features" not in s.explain()


def test_trainable_primary_is_fitted_out_of_sample():
    """An over-fitted trainable primary looks perfect on its own training bars. Meta-labels
    built on those in-sample sides teach the meta-model a success rate that never
    materialises; by default the primary is fitted on the first half and the meta-model
    learns from its out-of-sample bets only."""
    bars = make_synthetic_bars(2600, "H1", seed=13, model="gbm")
    train = MarketData(bars=bars.iloc[:2400])
    kw = dict(COMMON, primary=MemorizingPrimary(), max_holding_bars=6, importance="none",
              skill_gate_z=None)
    oos = MetaLabelStrategy(**kw).fit(train)
    assert oos.fit_report_["primary_trainable"]
    assert oos.train_labels_["t_idx"].min() >= 1200 == oos.fit_report_["first_event_bar"]
    assert 0.4 < oos.fit_report_["success_rate"] < 0.6
    leaky = MetaLabelStrategy(**kw, primary_oos_frac=0.0).fit(train)
    assert leaky.fit_report_["success_rate"] > 0.75   # the optimism the default avoids
    g = MLGBMStrategy(**COMMON, primary_features=[MemorizingPrimary()], max_holding_bars=6,
                      importance="none", skill_gate_z=None).fit(train)
    assert g.train_labels_["t_idx"].min() >= 1200


def _perturb_market(md: MarketData, t: int, seed: int) -> MarketData:
    """Bars after ``t`` and macro rows not yet available at ``available_at[t]`` replaced."""
    new = perturb_future(md, t, seed).bars
    cutoff = md.bars["available_at"].iloc[t]
    rng = np.random.default_rng(seed)
    macro = {}
    for name, frame in md.macro.items():
        f = frame.copy()
        fut = np.asarray(pd.DatetimeIndex(f["available_at"]) > cutoff)
        vals = f["value"].to_numpy(dtype=float).copy()
        vals[fut] = vals[fut] * rng.uniform(0.5, 1.5, int(fut.sum()))
        f["value"] = vals
        macro[name] = f
    return MarketData(bars=new, macro=macro, events=md.events)


def test_ml_strategies_with_macro_and_calendar_features_are_causal():
    """The shared leakage test only runs price groups; here the macro (as-of joined daily
    series) and calendar groups feed the models, and the future of bars AND macro moves."""
    from aurum.data.synthetic import make_synthetic_events, make_synthetic_macro

    bars = make_synthetic_bars(2600, "H1", seed=17, model="regime")
    md = MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=3),
                    events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7)))
    ov = {"macro": {"z_window": 40, "z_min_periods": 10, "corr_window": 20, "corr_min_periods": 10}}
    kw = dict(COMMON, feature_groups=["returns", "macro", "calendar"], feature_overrides=ov,
              importance="none", skill_gate_z=None, max_holding_bars=6)
    train = MarketData(bars=bars.iloc[:1800], macro=md.macro, events=md.events)
    for s in (MLGBMStrategy(**kw), MetaLabelStrategy(**kw, primary=MomentumStub(lookback=24))):
        s.fit(train)
        assert any(c.startswith("macro_") for c in s._feature_columns)
        assert any(c.startswith("calendar_") for c in s._feature_columns)
        full = s.generate(md)
        assert (full.iloc[1800:] != 0).any()
        for i, t in enumerate((1900, 2300)):
            alt = s.generate(_perturb_market(md, t, seed=i))
            pd.testing.assert_series_equal(alt.iloc[: t + 1], full.iloc[: t + 1])
        # macro missing at generate time fails loudly instead of trading on zeros
        with pytest.raises(ValueError, match="missing"):
            s.generate(MarketData(bars=bars, events=md.events))


def test_calibrated_gbm_guards_reserved_params_and_zero_weights():
    with pytest.raises(ValueError, match="managed by CalibratedGBM"):
        CalibratedGBM(model={"class_weight": "balanced"})   # would double-balance silently
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    y = (rng.random(n) < 0.5).astype(float)
    w = np.r_[np.zeros(800), np.ones(200)]                # e.g. an extreme time decay
    t = np.arange(n)
    with pytest.raises(ValueError, match="positive sum"):
        CalibratedGBM(model=FAST, n_threads=2).fit(X, y, w, t, t + 1)
