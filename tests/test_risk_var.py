"""Tests for aurum.risk.var."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aurum.backtest.result import BacktestResult
from aurum.core.instrument import XAUUSD
from aurum.data.synthetic import make_synthetic_bars
from aurum.risk.var import (
    DEFAULT_STRESS_SCENARIOS,
    cornish_fisher_quantile,
    cornish_fisher_var_es,
    daily_returns_from_equity,
    gap_shock,
    gaussian_var_es,
    historical_var_es,
    kupiec_pof,
    position_var,
    risk_report,
    rolling_var_backtest,
    stress_test,
    var_es,
)


@pytest.fixture(scope="module")
def normal_sample() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(200_000) * 0.01


def test_gaussian_closed_form():
    var, es = gaussian_var_es(level=0.95, mu=0.0, sigma=1.0)
    assert var == pytest.approx(1.6448536, rel=1e-6)
    assert es == pytest.approx(stats.norm.pdf(1.6448536) / 0.05, rel=1e-6)  # 2.0627
    var99, es99 = gaussian_var_es(level=0.99, mu=0.001, sigma=0.02)
    assert var99 == pytest.approx(2.3263479 * 0.02 - 0.001, rel=1e-6)
    assert es99 > var99


def test_historical_matches_gaussian_on_normal_data(normal_sample):
    hv, hes = historical_var_es(normal_sample, 0.99)
    gv, ges = gaussian_var_es(normal_sample, 0.99)
    assert hv == pytest.approx(gv, rel=0.03)
    assert hes == pytest.approx(ges, rel=0.03)
    assert hes >= hv


def test_cornish_fisher_reduces_to_gaussian_and_reacts_to_tails(normal_sample):
    assert cornish_fisher_quantile(0.05, 0.0, 0.0) == pytest.approx(stats.norm.ppf(0.05))
    v0, e0 = cornish_fisher_var_es(None, 0.95, mu=0.0, sigma=1.0, skew=0.0, excess_kurt=0.0)
    g0, ge0 = gaussian_var_es(level=0.95, mu=0.0, sigma=1.0)
    assert v0 == pytest.approx(g0) and e0 == pytest.approx(ge0)
    # negative skew and fat tails increase modified VaR and ES
    v1, e1 = cornish_fisher_var_es(None, 0.99, mu=0.0, sigma=1.0, skew=-0.8, excess_kurt=3.0)
    g1, ge1 = gaussian_var_es(level=0.99, mu=0.0, sigma=1.0)
    assert v1 > g1 and e1 > ge1 and e1 >= v1
    # with MODERATE fat tails / negative skew (the regime where the expansion is valid) CF
    # tracks the empirical 99% quantile better than the Gaussian. (For very heavy tails,
    # e.g. Student-t with 5 dof, CF over-corrects — documented in the module.)
    rng = np.random.default_rng(2)
    t_sample = stats.t.rvs(df=10, size=200_000, random_state=np.random.default_rng(1)) * 0.01
    mix = np.where(rng.random(200_000) < 0.9, rng.normal(0.001, 0.01, 200_000), rng.normal(-0.01, 0.015, 200_000))
    for sample in (t_sample, mix):
        hv, _ = historical_var_es(sample, 0.99)
        cv, _ = cornish_fisher_var_es(sample, 0.99)
        gv, _ = gaussian_var_es(sample, 0.99)
        assert abs(cv - hv) < abs(gv - hv)


def test_modified_es_matches_numerical_integration():
    s, k, a = -0.5, 2.0, 0.05
    u = (np.arange(200_000) + 0.5) / 200_000 * a
    z = stats.norm.ppf(u)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    numeric_es = -zcf.mean()
    _, es = cornish_fisher_var_es(None, 0.95, mu=0.0, sigma=1.0, skew=s, excess_kurt=k)
    assert es == pytest.approx(numeric_es, rel=2e-3)


def test_var_es_dispatch_and_horizon(normal_sample):
    for m in ("historical", "gaussian", "cornish_fisher"):
        out = var_es(normal_sample, 0.95, m)
        assert out["var"] == pytest.approx(0.01645, rel=0.05)
        assert out["es"] >= out["var"]
        h = var_es(normal_sample, 0.95, m, horizon=4)
        assert h["var"] == pytest.approx(2 * out["var"], rel=0.05)
    with pytest.raises(ValueError):
        var_es(normal_sample, 0.95, "monte_carlo")
    with pytest.raises(ValueError):
        var_es(normal_sample, 1.5)


def test_kupiec_and_rolling_backtest():
    ok = kupiec_pof(1000, 50, 0.95)
    assert ok["lr"] == pytest.approx(0.0, abs=1e-9) and ok["p_value"] == pytest.approx(1.0)
    bad = kupiec_pof(1000, 120, 0.95)
    assert bad["p_value"] < 0.001
    zero = kupiec_pof(500, 0, 0.99)
    assert np.isfinite(zero["lr"]) and zero["p_value"] < 0.05
    rng = np.random.default_rng(3)
    idx = pd.date_range("2015-01-01", periods=3000, freq="B", tz="UTC")
    r = pd.Series(rng.standard_normal(3000) * 0.01, index=idx)
    bt = rolling_var_backtest(r, 0.95, window=250)
    assert bt["n"] == 3000 - 250
    assert bt["p_value"] > 0.01
    # causal: the VaR for day t ignores day t itself — a huge loss on the last day is a breach
    r2 = r.copy()
    r2.iloc[-1] = -0.5
    assert rolling_var_backtest(r2, 0.95, window=250)["breaches"] >= bt["breaches"]
    g = rolling_var_backtest(r, 0.95, window=250, method="gaussian")
    assert g["n"] == bt["n"]


def test_gap_shock_and_stress():
    # 1 lot long at 2000, -5% -> -100 * 1 * 100 = -10,000 USD
    assert gap_shock(1.0, 2000.0, -0.05) == pytest.approx(-10_000.0)
    assert gap_shock(-2.0, 2000.0, -0.05) == pytest.approx(20_000.0)
    # a widened 2.00 spread costs |lots| * 100 * 1.00 on exit
    assert gap_shock(1.0, 2000.0, -0.05, spread=2.0) == pytest.approx(-10_100.0)
    assert gap_shock(0.0, 2000.0, 0.1) == 0.0
    st = stress_test(1.0, 2000.0, 100_000.0)
    assert len(st) == len(DEFAULT_STRESS_SCENARIOS)
    assert st["pnl_usd"].is_monotonic_increasing  # worst first
    assert st.iloc[0]["pnl_usd"] == pytest.approx(-20_000.0)
    assert st.iloc[0]["pnl_pct_equity"] == pytest.approx(-0.2)
    custom = stress_test(-1.0, 2000.0, 50_000.0, scenarios={"spike": 0.08})
    assert custom.iloc[0]["pnl_usd"] == pytest.approx(-16_000.0)


def test_position_var():
    out = position_var(1.0, 2000.0, 0.16, level=0.99)
    sigma = 0.16 / math.sqrt(252)
    assert out["notional"] == pytest.approx(200_000.0)
    assert out["var_usd"] == pytest.approx(2.3263479 * sigma * 200_000, rel=1e-6)
    assert out["es_usd"] > out["var_usd"]


def _result() -> BacktestResult:
    bars = make_synthetic_bars(2500, "H1", seed=4)
    ret = np.log(bars["close"]).diff().fillna(0.0)
    pos = pd.Series(0.5, index=bars.index)
    equity = 100_000 + (pos.shift(1).fillna(0) * XAUUSD.contract_size * bars["close"].diff().fillna(0)).cumsum()
    empty = pd.DataFrame()
    return BacktestResult(
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        positions=pos,
        costs=pd.DataFrame(index=bars.index),
        trades=empty,
        fills=pd.DataFrame({"price": [float(bars["close"].iloc[0])]}),
        meta={"last_price": float(bars["close"].iloc[-1])},
        forecast=ret,
    )


def test_risk_report_on_backtest_result():
    res = _result()
    rep = risk_report(res)
    json.dumps(rep, default=float)
    assert rep["level"] == 0.95
    assert set(rep["daily"]) == {"historical", "gaussian", "cornish_fisher"}
    for v in rep["daily"].values():
        assert v["var"] > 0 and v["es"] >= v["var"]
    assert rep["n_days"] > 20
    assert 0 <= rep["max_drawdown"] < 1
    assert "var_backtest" in rep and rep["var_backtest"]["n"] > 0
    assert rep["position"]["lots"] == 0.5
    assert rep["position"]["price"] == res.meta["last_price"]
    assert len(rep["position"]["stress"]) == len(DEFAULT_STRESS_SCENARIOS)
    assert rep["daily_usd"]["historical"]["var"] == pytest.approx(
        rep["daily"]["historical"]["var"] * rep["last_equity"]
    )
    # a bare equity series works too (no position section)
    rep2 = risk_report(res.equity)
    assert "position" not in rep2 and "daily" in rep2
    # too-short input degrades gracefully
    assert risk_report(res.equity.iloc[:2])["notes"]


def test_daily_returns_from_equity():
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    eq = pd.Series(np.linspace(100, 103, 72), index=idx)
    d = daily_returns_from_equity(eq)
    assert len(d) == 2
    assert d.iloc[0] == pytest.approx(eq.iloc[47] / eq.iloc[23] - 1)


# ---------------------------------------------------------------------------------------
# Review regressions (adversarial)
# ---------------------------------------------------------------------------------------
def test_daily_returns_fold_sunday_stubs_like_metrics():
    """Sunday-evening bars must not form their own "day" (they did: 302 days instead of
    252 on this sample, with ~1/4 of the normal daily std, biasing daily VaR low)."""
    from aurum.backtest.metrics import daily_returns

    res = _result()
    assert (res.equity.index.weekday == 6).any()
    d = daily_returns_from_equity(res.equity)
    assert not d.index.weekday.isin([5, 6]).any()
    ref = daily_returns(res.equity).iloc[1:]  # metrics also books day 1 vs the first mark
    pd.testing.assert_series_equal(d, ref, check_names=False, check_freq=False)
    rep = risk_report(res)
    assert rep["n_days"] == len(ref)
    raw = daily_returns_from_equity(res.equity, fold_weekends=False)
    assert len(raw) > len(d)


def test_daily_returns_accepts_naive_index():
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    eq = pd.Series(np.linspace(100, 103, 72), index=idx)
    assert len(daily_returns_from_equity(eq)) == 2


def test_risk_report_stresses_the_position_held_at_the_close():
    res = _result()
    pc = res.positions.copy()
    pc.iloc[-1] = 0.0  # a protective stop fired inside the last bar
    res.position_close = pc
    rep = risk_report(res)
    assert rep["position"]["lots"] == 0.0
    assert all(row["pnl_usd"] == 0.0 for row in rep["position"]["stress"])


def test_stress_test_and_backtest_edge_cases():
    empty = stress_test(1.0, 2000.0, 1e5, scenarios={})
    assert empty.empty and "pnl_usd" in empty.columns
    short = rolling_var_backtest(pd.Series([0.01, -0.02] * 5), 0.95, window=5)
    full = rolling_var_backtest(pd.Series(np.random.default_rng(0).normal(0, 0.01, 300)), 0.95, window=50)
    assert set(short) == set(full)
