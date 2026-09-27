"""Tests for aurum.models.regime.GaussianHMM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.data.synthetic import make_synthetic_bars
from aurum.models.regime import GaussianHMM


def _two_state(n: int, seed: int, p_stay: float = 0.98, sig=(0.5, 2.0), mu=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    s = np.empty(n, dtype=int)
    s[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        s[t] = s[t - 1] if u[t] < p_stay else 1 - s[t - 1]
    x = np.asarray(mu)[s] + rng.standard_normal(n) * np.asarray(sig)[s]
    idx = pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")
    return pd.Series(x, index=idx, name="ret"), s


@pytest.fixture(scope="module")
def data():
    return _two_state(3000, seed=11)


@pytest.fixture(scope="module")
def model(data):
    x, _ = data
    return GaussianHMM(n_states=2, seed=0).fit(x.iloc[:2000])


def test_recovers_regimes_out_of_sample(model, data):
    x, s = data
    probs = model.filter(x)
    assert list(probs.columns) == ["p_state_0", "p_state_1"]
    pred = probs.to_numpy().argmax(axis=1)
    acc_oos = (pred[2000:] == s[2000:]).mean()
    acc_all = (pred == s).mean()
    assert acc_oos > 0.8
    assert acc_all > 0.8
    # state 0 = calm (lower variance), parameters close to the truth
    assert model.vars_[0, 0] < model.vars_[1, 0]
    assert np.sqrt(model.vars_[0, 0]) == pytest.approx(0.5, rel=0.2)
    assert np.sqrt(model.vars_[1, 0]) == pytest.approx(2.0, rel=0.2)
    assert np.all(np.diag(model.transmat_) > 0.9)


def test_filter_is_causal(model, data):
    x, _ = data
    base = model.filter(x)
    for t in (500, 1500, 2600):
        x2 = x.copy()
        rng = np.random.default_rng(t)
        x2.iloc[t + 1 :] = rng.standard_normal(len(x) - t - 1) * 10.0
        pert = model.filter(x2)
        np.testing.assert_array_equal(base.iloc[: t + 1].to_numpy(), pert.iloc[: t + 1].to_numpy())
    # truncation gives the same answer as the full-sample filter up to t
    trunc = model.filter(x.iloc[:1234])
    np.testing.assert_allclose(trunc.to_numpy(), base.iloc[:1234].to_numpy(), rtol=0, atol=0)


def test_predict_next_is_filter_times_transition(model, data):
    x, _ = data
    filt = model.filter(x)
    nxt = model.predict_next(x)
    np.testing.assert_allclose(nxt.to_numpy(), filt.to_numpy() @ model.transmat_)
    np.testing.assert_allclose(nxt.sum(axis=1).to_numpy(), 1.0)
    np.testing.assert_allclose(filt.sum(axis=1).to_numpy(), 1.0)
    assert list(nxt.columns) == ["p_next_state_0", "p_next_state_1"]
    state = model.filtered_state(x)
    assert set(np.unique(state)) <= {0, 1}


def test_deterministic_and_summary(data):
    x, _ = data
    a = GaussianHMM(2, seed=5).fit(x.iloc[:1500])
    b = GaussianHMM(2, seed=5).fit(x.iloc[:1500])
    np.testing.assert_array_equal(a.transmat_, b.transmat_)
    np.testing.assert_array_equal(a.means_, b.means_)
    s = a.summary()
    assert s["n_states"] == 2 and len(s["expected_durations"]) == 2
    assert np.isclose(sum(s["stationary"]), 1.0)
    assert np.isfinite(a.score(x.iloc[1500:]))


def test_multivariate_with_missing_values():
    x, s = _two_state(1500, seed=3)
    rng = np.random.default_rng(0)
    # a noisy log-vol proxy: Gaussian within each regime (as the model assumes)
    logvol = np.log(np.array([0.5, 2.0]))[s] + rng.normal(0, 0.8, len(x))
    feat = pd.DataFrame({"ret": x, "logvol": logvol}, index=x.index)
    feat.iloc[::50, 1] = np.nan          # partially missing rows
    feat.iloc[7] = np.nan                # fully missing row
    m = GaussianHMM(2, seed=1, n_init=2).fit(feat)
    probs = m.filter(feat)
    assert np.isfinite(probs.to_numpy()).all()
    acc = (probs.to_numpy().argmax(axis=1) == s).mean()
    assert acc > 0.8
    assert m.means_.shape == (2, 2)


def test_three_states_ordered_by_variance():
    rng = np.random.default_rng(4)
    n = 1800
    s = np.repeat(np.array([0, 1, 2, 1, 0, 2]), n // 6)
    x = rng.standard_normal(n) * np.array([0.3, 1.0, 3.0])[s]
    m = GaussianHMM(3, seed=2, n_iter=100, n_init=2).fit(x)
    v = m.vars_[:, 0]
    assert np.all(np.diff(v) > 0)
    assert m.filter(x).shape == (n, 3)


def test_on_synthetic_regime_bars():
    bars = make_synthetic_bars(2500, "H1", seed=2, model="regime")
    r = np.log(bars["close"]).diff().dropna()
    m = GaussianHMM(2, seed=0).fit(r.iloc[:1500])
    p = m.filter(r)
    assert p.index.equals(r.index)
    assert m.vars_[0, 0] < m.vars_[1, 0]


def test_errors():
    with pytest.raises(RuntimeError):
        GaussianHMM().filter(np.zeros(10))
    with pytest.raises(ValueError):
        GaussianHMM(n_states=0)
    with pytest.raises(ValueError):
        GaussianHMM(2).fit(np.array([1.0, 2.0, np.nan]))


# ---------------------------------------------------------------------------------------
# Review regressions (adversarial)
# ---------------------------------------------------------------------------------------
def test_filter_aligns_dataframe_columns_by_name():
    """Emission parameters are positional: a re-ordered live feature frame must not be
    scored against the wrong means/variances (before the fix: probabilities flipped)."""
    x, s = _two_state(1500, seed=6)
    rng = np.random.default_rng(1)
    feat = pd.DataFrame({"ret": x, "level": 100.0 + rng.normal(0, 1.0, len(x))}, index=x.index)
    m = GaussianHMM(2, seed=0, n_init=1).fit(feat)
    base = m.filter(feat)
    pd.testing.assert_frame_equal(m.filter(feat[["level", "ret"]]), base)
    extra = feat.assign(unused=0.0)[["unused", "level", "ret"]]
    pd.testing.assert_frame_equal(m.filter(extra), base)
    assert m.score(feat[["level", "ret"]]) == pytest.approx(m.score(feat))
    with pytest.raises(KeyError):
        m.filter(feat[["ret"]])  # a missing feature used to broadcast silently
    with pytest.raises(ValueError):
        m.filter(feat[["ret"]].to_numpy())
    # models fitted on arrays / Series stay positional
    m1 = GaussianHMM(2, seed=0, n_init=1).fit(x.to_numpy())
    np.testing.assert_allclose(m1.filter(x.rename("anything")).to_numpy(), m1.filter(x.to_numpy()).to_numpy())
