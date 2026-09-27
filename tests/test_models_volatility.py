"""Tests for aurum.models.volatility (EWMA, GARCH(1,1), realised variance, HAR-RV, blend)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aurum.data.synthetic import make_synthetic_bars
from aurum.models.volatility import (
    Garch11,
    HarRV,
    blend_vol,
    daily_realised_variance,
    ewma_volatility,
    har_rv_forecast,
    simulate_garch11,
)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_synthetic_bars(3000, "H1", seed=3, model="regime")


# ---------------------------------------------------------------------------------------
# EWMA (behaviour must be preserved)
# ---------------------------------------------------------------------------------------
def test_ewma_volatility_contract(bars):
    vol = ewma_volatility(bars["close"])
    assert vol.name == "vol_ann"
    assert vol.index.equals(bars.index)
    assert (vol.iloc[:19] == 0.20).all()  # warm-up constant, never back-filled
    assert vol.between(0.03, 2.0).all()
    # causal: perturbing the future does not change the past
    close2 = bars["close"].copy()
    close2.iloc[2000:] *= 1.3
    v2 = ewma_volatility(close2, bars_per_year=infer_bpy(bars))
    v1 = ewma_volatility(bars["close"], bars_per_year=infer_bpy(bars))
    pd.testing.assert_series_equal(v1.iloc[:2000], v2.iloc[:2000])


def infer_bpy(b: pd.DataFrame) -> float:
    from aurum.core.timeframes import infer_bars_per_year

    return infer_bars_per_year(b.index)


# ---------------------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1])
def test_garch_parameter_recovery(seed):
    omega, alpha, beta = 2e-7, 0.08, 0.90
    r, _ = simulate_garch11(8000, omega=omega, alpha=alpha, beta=beta, seed=seed)
    g = Garch11(bars_per_year=252).fit(r)
    p = g.params_
    assert p is not None
    assert abs(p.alpha - alpha) < 0.03
    assert abs(p.beta - beta) < 0.05
    assert abs(p.persistence - (alpha + beta)) < 0.03
    # long-run variance is the well-identified combination of omega/alpha/beta
    assert p.unconditional_variance == pytest.approx(omega / (1 - alpha - beta), rel=0.35)
    assert g.converged_


def test_garch_constraints_on_market_like_data(bars):
    r = np.log(bars["close"]).diff()
    g = Garch11().fit(r)
    p = g.params_
    assert p.omega > 0 and p.alpha >= 0 and p.beta >= 0 and p.alpha + p.beta < 1
    s = g.summary()
    assert set(["omega", "alpha", "beta", "loglik", "half_life_bars"]).issubset(s)
    assert s["unconditional_vol_ann"] > 0


def test_garch_forecast_tracks_true_conditional_vol():
    r, h = simulate_garch11(5000, omega=2e-7, alpha=0.08, beta=0.90, seed=7)
    g = Garch11(bars_per_year=252).fit(r[:3000])
    fc = g.forecast(r, annualise=False).to_numpy()
    # forecast at t is sigma_{t+1}; compare with the true h[t+1] out of sample
    true_next = np.sqrt(h[1:])
    corr = np.corrcoef(fc[3000:-1], true_next[3000:])[0, 1]
    assert corr > 0.9


def test_garch_forecast_is_causal_and_aligned(bars):
    r = np.log(bars["close"]).diff()  # leading NaN
    g = Garch11().fit(r.iloc[:2000])
    fc = g.forecast(r)
    assert fc.index.equals(r.index)
    assert np.isfinite(fc).all()
    assert fc.name == "garch_vol_ann"
    r2 = r.copy()
    r2.iloc[2500:] = r2.iloc[2500:] * 5.0
    fc2 = g.forecast(r2)
    pd.testing.assert_series_equal(fc.iloc[:2500], fc2.iloc[:2500])
    assert not np.allclose(fc.iloc[2500:], fc2.iloc[2500:])


def test_garch_interior_nan_and_horizon():
    r, _ = simulate_garch11(3000, omega=2e-7, alpha=0.08, beta=0.90, seed=2)
    g = Garch11(bars_per_year=252).fit(r)
    r_nan = r.copy()
    r_nan[100] = np.nan
    fc = g.forecast(r_nan)
    assert np.isfinite(fc).all()
    # long horizons converge to the unconditional vol
    long = g.forecast(r, horizon=5000)
    unc = math.sqrt(g.params_.unconditional_variance * 252)
    assert np.allclose(long.to_numpy(), unc, rtol=0.05)
    short = g.forecast(r, horizon=1)
    assert short.std() > long.std()


def test_garch_requires_fit_and_enough_data():
    with pytest.raises(RuntimeError):
        Garch11(bars_per_year=252).forecast(np.zeros(10))
    with pytest.raises(ValueError):
        Garch11().fit(np.random.default_rng(0).normal(size=50))
    with pytest.raises(ValueError):
        Garch11(mean="ar1")


# ---------------------------------------------------------------------------------------
# Realised variance & HAR
# ---------------------------------------------------------------------------------------
def test_daily_realised_variance_point_in_time(bars):
    rv = daily_realised_variance(bars)
    assert {"rv", "n_obs", "available_at"} <= set(rv.columns)
    assert str(rv.index.tz) == "UTC"
    # available_at is the availability of the day's last bar
    day = rv.index[3]
    in_day = bars.loc[(bars.index >= day) & (bars.index < day + pd.Timedelta(days=1))]
    assert rv.loc[day, "available_at"] == in_day["available_at"].max()
    r = np.log(bars["close"]).diff()
    expected = (r.loc[in_day.index] ** 2).sum()
    assert rv.loc[day, "rv"] == pytest.approx(expected)
    # a partial final day (data ends at 12:00 UTC) is dropped unless complete_only=False
    noon = np.flatnonzero((bars.index.hour == 11) & (bars.index.weekday < 4))[-1]
    cut = bars.iloc[: noon + 1]
    partial_day = cut.index[-1].normalize()
    rv_cut = daily_realised_variance(cut)
    assert rv_cut.index[-1] < partial_day
    rv_all = daily_realised_variance(cut, complete_only=False)
    assert rv_all.index[-1] == partial_day
    assert rv_all["available_at"].iloc[-1] == cut["available_at"].iloc[-1]


def _simulate_har(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    b0, bd, bw, bm = 0.2e-5, 0.35, 0.35, 0.2
    rv = np.full(n, 1e-5)
    for t in range(22, n - 1):
        mean_w = rv[t - 4 : t + 1].mean()
        mean_m = rv[t - 21 : t + 1].mean()
        rv[t + 1] = max(b0 + bd * rv[t] + bw * mean_w + bm * mean_m + rng.normal(0, 2e-6), 1e-7)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(rv, index=idx)


def test_har_recovers_coefficients():
    rv = _simulate_har(4000, seed=1)
    m = HarRV().fit(rv.iloc[100:])
    c = m.coef_
    assert c["rv_1"] == pytest.approx(0.35, abs=0.1)
    assert c["rv_5"] == pytest.approx(0.35, abs=0.15)
    assert c["rv_22"] == pytest.approx(0.2, abs=0.15)
    pred = m.predict(rv)
    assert (pred.dropna() > 0).all()
    assert HarRV(log=True).fit(rv.iloc[100:]).predict(rv).dropna().gt(0).all()


def test_har_rv_forecast_is_walk_forward_causal():
    rv = _simulate_har(900, seed=2)
    fc = har_rv_forecast(rv, min_train=250, refit_every=21)
    assert fc.iloc[:250].isna().all()
    assert fc.iloc[260:].notna().all()
    rv2 = rv.copy()
    rv2.iloc[600:] *= 10.0
    fc2 = har_rv_forecast(rv2, min_train=250, refit_every=21)
    pd.testing.assert_series_equal(fc.iloc[:600], fc2.iloc[:600])
    var = har_rv_forecast(rv, output="variance")
    assert np.allclose(np.sqrt(var.dropna() * 252), fc.dropna())
    logfc = har_rv_forecast(rv, log=True, window=300)
    assert logfc.dropna().gt(0).all()


def test_har_on_synthetic_bars(bars):
    rv = daily_realised_variance(bars)["rv"]
    fc = har_rv_forecast(rv, min_train=60, refit_every=10)
    assert fc.dropna().between(0.01, 3.0).all()


# ---------------------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------------------
def test_blend_vol_variance_space_and_missing():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    a = pd.Series([0.1, 0.1, np.nan, 0.2], index=idx)
    b = pd.Series([0.3, 0.3, 0.3, np.nan], index=idx)
    out = blend_vol(a, b)
    assert out.iloc[0] == pytest.approx(math.sqrt((0.01 + 0.09) / 2))
    assert out.iloc[2] == pytest.approx(0.3)
    assert out.iloc[3] == pytest.approx(0.2)
    w = blend_vol(a, b, weights=[3, 1])
    assert w.iloc[0] == pytest.approx(math.sqrt(0.75 * 0.01 + 0.25 * 0.09))
    with pytest.raises(ValueError):
        blend_vol(a, b, weights=[1.0])
    with pytest.raises(ValueError):
        blend_vol()


# ---------------------------------------------------------------------------------------
# Review regressions (adversarial)
# ---------------------------------------------------------------------------------------
def _rv_with_zero_days(n: int = 900, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    rv = pd.Series(np.exp(rng.normal(np.log(1e-4), 0.5, n)), index=idx)
    rv.iloc[100:103] = 0.0  # zero-RV holiday sessions: the log floor binds
    return rv


@pytest.mark.parametrize("log", [True, False])
def test_har_floor_is_point_in_time(log):
    """The positivity floor must not be estimated from data after the forecast origin.

    Before the fix the floor was ``1e-6 x median`` of the WHOLE series, so scaling future RV
    changed every earlier log-HAR forecast whose regressors touched a zero-RV day.
    """
    rv = _rv_with_zero_days()
    fc = har_rv_forecast(rv, log=log, min_train=250, refit_every=21, output="variance")
    rv2 = rv.copy()
    rv2.iloc[600:] *= 50.0
    fc2 = har_rv_forecast(rv2, log=log, min_train=250, refit_every=21, output="variance")
    pd.testing.assert_series_equal(fc.iloc[:600], fc2.iloc[:600])
    assert fc.iloc[260:].notna().all() and (fc.dropna() > 0).all()


def test_har_model_floor_fixed_at_fit():
    rv = _rv_with_zero_days()
    m = HarRV(log=True).fit(rv.iloc[:500])
    assert m.floor_ is not None and m.floor_ > 0
    p1 = m.predict(rv.iloc[:520])
    p2 = m.predict(pd.concat([rv.iloc[:520], rv.iloc[520:] * 50.0]))
    pd.testing.assert_series_equal(p1, p2.iloc[:520])
    lv = HarRV().fit(rv.iloc[:500])
    q1 = lv.predict(rv.iloc[:520])
    q2 = lv.predict(pd.concat([rv.iloc[:520], rv.iloc[520:] * 50.0]))
    pd.testing.assert_series_equal(q1, q2.iloc[:520])
    with pytest.raises(RuntimeError):
        HarRV().predict(rv)


def test_daily_rv_folds_weekend_stub_sessions(bars):
    """Gold reopens Sunday ~22:00 UTC: those bars must not form a 2-bar 'day'."""
    assert (bars.index.weekday == 6).any()  # the synthetic calendar has Sunday-evening bars
    rv = daily_realised_variance(bars)
    assert not rv.index.weekday.isin([5, 6]).any()
    assert rv["n_obs"].min() >= 20  # every session is a full (or Friday-short) trading day
    # the Monday session contains the Sunday bars (and therefore the weekend gap)
    monday = rv.index[rv.index.weekday == 0][1]
    in_sess = bars.loc[(bars.index >= monday - pd.Timedelta(hours=2)) & (bars.index < monday + pd.Timedelta(days=1))]
    r = np.log(bars["close"]).diff()
    assert rv.loc[monday, "n_obs"] == len(in_sess)
    assert rv.loc[monday, "rv"] == pytest.approx(float((r.loc[in_sess.index] ** 2).sum()))
    assert rv.loc[monday, "available_at"] == in_sess["available_at"].max()
    # the unfolded variant reproduces the stubs
    raw = daily_realised_variance(bars, fold_weekends=False)
    assert (raw.index.weekday == 6).any() and raw.loc[raw.index.weekday == 6, "n_obs"].max() <= 2
    # NY-close anchor: the session opening Sunday 21:00 UTC is Monday's (labelled by its end)
    ny = daily_realised_variance(bars, anchor_hour_utc=21)
    assert not ny.index.weekday.isin([5, 6]).any()
    first_mon = ny.index[ny.index.weekday == 0][1]  # [0] is the data start (no prior return)
    sess = bars.loc[(bars.index >= first_mon - pd.Timedelta(hours=3)) & (bars.index < first_mon + pd.Timedelta(hours=21))]
    assert ny.loc[first_mon, "n_obs"] == len(sess)
    # point-in-time: nothing is available before the session's last bar has closed
    assert (ny["available_at"] <= ny.index + pd.Timedelta(hours=21)).all()
    with pytest.raises(ValueError):
        daily_realised_variance(bars, anchor_hour_utc=24)


def test_garch_floor_cap_are_annualised_units():
    r, _ = simulate_garch11(2000, omega=2e-7, alpha=0.08, beta=0.90, seed=3)
    g = Garch11(bars_per_year=252, floor=0.05, cap=0.5).fit(r)
    per_bar = g.forecast(r, annualise=False)
    ann = g.forecast(r)
    np.testing.assert_allclose(per_bar.to_numpy(), Garch11(bars_per_year=252).fit(r).forecast(r, annualise=False))
    assert per_bar.max() < 0.05  # a per-bar vol is never clipped to an annual floor
    assert ann.min() >= 0.05 and ann.max() <= 0.5
