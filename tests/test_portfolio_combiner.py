"""Tests for aurum.portfolio.combiner."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from aurum.data.synthetic import make_synthetic_bars
from aurum.portfolio.combiner import METHODS, ForecastCombiner, cap_weights


@pytest.fixture(scope="module")
def setup():
    bars = make_synthetic_bars(4000, "H1", seed=21, model="gbm")
    close = bars["close"]
    rng = np.random.default_rng(5)
    n = len(close)
    # KNOWN-GOOD signal built from FUTURE returns — only allowed inside this test to create
    # a strategy with genuine skill; library code never sees future returns.
    fut = np.sign(np.log(close).diff().shift(-1).fillna(0.0).to_numpy())
    informative = np.clip(0.25 * fut + rng.normal(0, 0.4, n), -1, 1)
    noise = {f"noise_{i}": np.clip(rng.normal(0, 0.4, n), -1, 1) for i in range(3)}
    slow = np.clip(np.cumsum(rng.normal(0, 0.05, n)), -1, 1)
    fc = pd.DataFrame({"informative": informative, **noise, "slow": slow}, index=close.index)
    return fc, close


@pytest.mark.parametrize("method", METHODS)
def test_weights_sum_to_one_and_respect_cap(setup, method):
    fc, close = setup
    comb = ForecastCombiner(method=method, max_weight=0.4).fit(fc.iloc[:3000], close.iloc[:3000])
    w = comb.weights_
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all()
    assert (w <= 0.4 + 1e-9).all()
    assert list(w.index) == list(fc.columns)
    out = comb.combine(fc.iloc[3000:])
    assert out.index.equals(fc.index[3000:])
    assert out.between(-1, 1).all()
    assert out.name == "combined"
    assert 1.0 <= comb.fdm_ <= comb.fdm_cap


def test_sharpe_shrink_prefers_informative(setup):
    fc, close = setup
    comb = ForecastCombiner(method="sharpe_shrink", shrinkage=0.5, max_weight=0.6).fit(
        fc.iloc[:3000], close.iloc[:3000]
    )
    w = comb.weights_
    assert w.idxmax() == "informative"
    assert w["informative"] > w.drop("informative").max()
    ex = comb.explain()
    assert ex["train_sharpe"]["informative"] == max(ex["train_sharpe"].values())
    assert ex["train_sharpe"]["informative"] > 3.0
    assert set(ex["weights"]) == set(fc.columns)
    json.dumps(ex)  # JSON-serialisable for agents / reports
    # the informative strategy is still capped
    capped = ForecastCombiner(method="sharpe_shrink", max_weight=0.3).fit(fc.iloc[:3000], close.iloc[:3000])
    assert capped.weights_["informative"] == pytest.approx(0.3)
    assert capped.weights_.idxmax() == "informative"


def test_shrinkage_one_equals_inverse_vol_on_positive_sharpes(setup):
    fc, close = setup
    a = ForecastCombiner(method="sharpe_shrink", shrinkage=1.0, max_weight=1.0).fit(fc, close)
    b = ForecastCombiner(method="inverse_vol", max_weight=1.0).fit(fc, close)
    if a.train_sharpe_.mean() > 0:
        pd.testing.assert_series_equal(a.weights_, b.weights_, check_exact=False, atol=1e-12)


def test_fit_uses_only_given_data(setup):
    fc, close = setup
    train_fc = fc.iloc[:2500]
    a = ForecastCombiner().fit(train_fc, close.iloc[:2500])
    # passing the FULL close series (with a manipulated future) must not change anything
    close_mod = close.copy()
    close_mod.iloc[2500:] = close_mod.iloc[2500:] * np.linspace(1, 3, len(close) - 2500)
    b = ForecastCombiner().fit(train_fc, close_mod)
    pd.testing.assert_series_equal(a.weights_, b.weights_)
    assert a.fdm_ == b.fdm_
    pd.testing.assert_series_equal(a.train_sharpe_, b.train_sharpe_)
    # the last training row has no forward return inside the window -> dropped
    streams = a.unit_vol_streams(train_fc, close_mod)
    assert streams.index[-1] < train_fc.index[-1]


def test_fdm_from_forecast_correlation():
    idx = pd.date_range("2023-01-02", periods=2000, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    close = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx)))), index=idx)
    base = np.clip(rng.normal(0, 0.4, len(idx)), -1, 1)
    # identical forecasts -> no diversification -> FDM == 1
    same = pd.DataFrame({"a": base, "b": base, "c": base}, index=idx)
    c1 = ForecastCombiner(method="equal").fit(same, close)
    assert c1.fdm_ == pytest.approx(1.0)
    # independent forecasts -> FDM ~ sqrt(N)
    ind = pd.DataFrame({k: np.clip(rng.normal(0, 0.4, len(idx)), -1, 1) for k in "abcd"}, index=idx)
    c2 = ForecastCombiner(method="equal", fdm_cap=10.0).fit(ind, close)
    assert c2.fdm_ == pytest.approx(2.0, rel=0.05)
    # cap applies
    c3 = ForecastCombiner(method="equal", fdm_cap=1.5).fit(ind, close)
    assert c3.fdm_ == 1.5 and c3.fdm_raw_ > 1.5
    # negatively correlated forecasts: correlation floored at 0 (Carver) -> FDM = sqrt(2)
    neg = pd.DataFrame({"a": base, "b": -base}, index=idx)
    c4 = ForecastCombiner(method="equal", max_weight=1.0, fdm_cap=10.0).fit(neg, close)
    assert c4.fdm_ == pytest.approx(math.sqrt(2.0))
    # combine applies FDM and clips
    out = c2.combine(ind)
    expected = np.clip(ind.to_numpy().mean(axis=1) * c2.fdm_, -1, 1)
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_cap_forcing_weight_onto_losers_is_flagged(setup):
    fc, close = setup
    rng = np.random.default_rng(9)
    good = fc["informative"]
    f = pd.DataFrame(
        {
            "good": good,
            "bad1": np.clip(-0.5 * good + rng.normal(0, 0.3, len(good)), -1, 1),
            "bad2": np.clip(-0.5 * good + rng.normal(0, 0.3, len(good)), -1, 1),
        },
        index=fc.index,
    )
    comb = ForecastCombiner(method="sharpe_shrink", max_weight=0.4).fit(f.iloc[:3000], close.iloc[:3000])
    assert comb.weights_.sum() == pytest.approx(1.0)
    assert comb.weights_.max() <= 0.4 + 1e-12
    assert comb.weights_["good"] == pytest.approx(0.4)
    assert any("forced" in n for n in comb.explain()["notes"])
    # without the cap the losers get nothing
    free = ForecastCombiner(method="sharpe_shrink", max_weight=1.0).fit(f.iloc[:3000], close.iloc[:3000])
    assert free.weights_["good"] == pytest.approx(1.0)


def test_inactive_strategy_gets_zero_weight(setup):
    fc, close = setup
    f = fc.iloc[:2000].copy()
    f["dead"] = 0.0
    comb = ForecastCombiner(method="equal", max_weight=0.5).fit(f, close.iloc[:2000])
    assert comb.weights_["dead"] == 0.0
    assert comb.weights_.sum() == pytest.approx(1.0)
    assert any("inactive" in n for n in comb.explain()["notes"])


def test_hrp_prefers_diversifying_cluster():
    idx = pd.date_range("2023-01-02", periods=3000, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    close = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx)))), index=idx)
    common = rng.normal(0, 0.4, len(idx))
    fc = pd.DataFrame(
        {
            "t1": np.clip(common + rng.normal(0, 0.05, len(idx)), -1, 1),
            "t2": np.clip(common + rng.normal(0, 0.05, len(idx)), -1, 1),
            "t3": np.clip(common + rng.normal(0, 0.05, len(idx)), -1, 1),
            "solo": np.clip(rng.normal(0, 0.4, len(idx)), -1, 1),
        },
        index=idx,
    )
    comb = ForecastCombiner(method="hrp", max_weight=1.0).fit(fc, close)
    w = comb.weights_
    assert w.sum() == pytest.approx(1.0)
    # the uncorrelated strategy gets more weight than any single member of the clone cluster
    assert w["solo"] > w[["t1", "t2", "t3"]].max()


def test_cap_weights_properties():
    w = cap_weights(np.array([10.0, 1.0, 1.0, 0.0]), 0.4)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.4 + 1e-12
    assert w[0] == pytest.approx(0.4)
    # a single positive raw weight: the remainder is spread evenly over the rest
    w2 = cap_weights(np.array([1.0, 0.0, 0.0, 0.0]), 0.4)
    np.testing.assert_allclose(w2, [0.4, 0.2, 0.2, 0.2])
    # infeasible cap relaxes to 1/N
    w3 = cap_weights(np.array([5.0, 1.0]), 0.3)
    np.testing.assert_allclose(w3, [0.5, 0.5])


def test_errors_and_unfitted(setup):
    fc, close = setup
    with pytest.raises(ValueError):
        ForecastCombiner(method="magic")
    with pytest.raises(RuntimeError):
        ForecastCombiner().combine(fc)
    comb = ForecastCombiner().fit(fc.iloc[:1000], close.iloc[:1000])
    with pytest.raises(KeyError):
        comb.combine(fc.drop(columns=["informative"]))
    assert ForecastCombiner().explain()["fitted"] is False
    with pytest.raises(ValueError):
        ForecastCombiner().fit(fc.iloc[:10], close.iloc[:10])


# ---------------------------------------------------------------------------------------
# Review regressions (adversarial)
# ---------------------------------------------------------------------------------------
def test_unsorted_or_duplicated_index_is_rejected(setup):
    """On a reversed index ``shift(-1)`` pairs f[t] with the PREVIOUS return — silent garbage."""
    fc, close = setup
    with pytest.raises(ValueError):
        ForecastCombiner().fit(fc.iloc[:1000].iloc[::-1], close)
    dup = pd.concat([fc.iloc[:500], fc.iloc[499:1000]])
    with pytest.raises(ValueError):
        ForecastCombiner().fit(dup, close)
    with pytest.raises(ValueError):
        ForecastCombiner().fit(fc.iloc[:1000], pd.concat([close.iloc[:1000], close.iloc[999:1000]]))


def test_combine_is_row_wise_causal(setup):
    fc, close = setup
    comb = ForecastCombiner().fit(fc.iloc[:2000], close.iloc[:2000])
    full = comb.combine(fc)
    pert = fc.copy()
    pert.iloc[3000:] = -pert.iloc[3000:]
    pd.testing.assert_series_equal(full.iloc[:3000], comb.combine(pert).iloc[:3000])
    pd.testing.assert_series_equal(full.iloc[2500:2600], comb.combine(fc.iloc[2500:2600]))
