"""Cost-aware ForecastCombiner: net-of-cost scoring, the c_t derivation against the real
simulator, unallocated risk (default) vs the legacy sum-to-1 behaviour, point-in-time use of
the cost inputs, and backwards compatibility of pickled combiners."""

from __future__ import annotations

import dataclasses
import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from aurum.core.instrument import XAUUSD
from aurum.core.timeframes import index_bar_minutes, nominal_bars_per_year
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import CostModel
from aurum.portfolio.combiner import METHODS, ForecastCombiner, cap_weights


@pytest.fixture(scope="module")
def world():
    bars = make_synthetic_bars(4000, "H1", seed=21, model="gbm")
    close = bars["close"]
    rng = np.random.default_rng(5)
    n = len(close)
    # KNOWN-GOOD signals built from FUTURE returns - only inside this test, to create
    # strategies with genuine skill; library code never sees future returns.
    fut = np.sign(np.log(close).diff().shift(-1).fillna(0.0).to_numpy())
    slow_leg = 0.6 * np.sign(np.sin(np.arange(n) / 150.0) + 1e-9)
    fc = pd.DataFrame({
        # persistent position + modest skill: little turnover, cheap to trade
        "slow_skill": np.clip(0.06 * fut + slow_leg, -1, 1),
        # more skill but flips every other bar: higher GROSS Sharpe, eaten by costs
        "churn": np.clip(0.2 * fut + rng.choice([-0.7, 0.7], n), -1, 1),
        "noise": np.clip(rng.normal(0, 0.4, n), -1, 1),
    }, index=close.index)
    return bars, fc


COST = 0.25   # explicit cost per unit of |forecast change| (unit-vol units) for ranking tests


def _unit_cost(cm: CostModel, bars: pd.DataFrame, comb: ForecastCombiner) -> pd.Series:
    """Independent re-implementation of c_t = (spread_eff/2 + slip + comm/oz) / (close * vol)."""
    close = bars["close"]
    r = np.log(close).diff()
    vol = np.sqrt((r**2).ewm(halflife=comb.vol_halflife, min_periods=comb.min_periods, adjust=False).mean())
    eff = np.maximum(bars["spread"] * cm.spread_multiplier, cm.min_spread)
    slip = cm.slippage_fixed + cm.slippage_range_frac * (bars["high"] - bars["low"])
    comm = cm.commission(1.0, instrument=XAUUSD) / XAUUSD.contract_size
    return (eff / 2 + slip + comm) / (close * vol)


def test_turnover_cost_matches_the_formula(world):
    bars, fc = world
    cm = CostModel(spread_multiplier=1.3, min_spread=0.2, slippage_fixed=0.03, slippage_range_frac=0.05,
                   commission_per_lot=3.5)
    comb = ForecastCombiner()
    c = comb.turnover_cost(fc, bars["close"], bars=bars, costs=cm)
    ref = _unit_cost(cm, bars, comb)
    ok = ref.notna()
    np.testing.assert_allclose(c[ok].to_numpy(), ref[ok].to_numpy(), rtol=1e-12)
    assert c.index.equals(fc.index)
    # a spread series alone (no high/low) = the range term is zero
    c2 = comb.turnover_cost(fc, bars["close"], spread=bars["spread"], costs=cm)
    assert (c2[ok] <= c[ok] + 1e-15).all() and (c2[ok] < c[ok]).any()
    # the multiplier scales linearly; costs never negative
    c3 = ForecastCombiner(cost_multiplier=2.0).turnover_cost(fc, bars["close"], bars=bars, costs=cm)
    np.testing.assert_allclose(c3[ok].to_numpy(), 2.0 * c[ok].to_numpy(), rtol=1e-12)
    assert (c[ok] > 0).all()


def test_net_streams_charge_turnover(world):
    bars, fc = world
    comb = ForecastCombiner()
    gross = comb.unit_vol_streams(fc, bars["close"])
    net = comb.unit_vol_streams(fc, bars["close"], cost_per_turnover=0.05)
    f = fc.clip(-1, 1)
    turnover = f.diff().abs().fillna(0.0).loc[gross.index]
    pd.testing.assert_frame_equal(net, gross - 0.05 * turnover)
    # the churner pays far more than the slow strategy
    drag = (gross - net).mean()
    assert drag["churn"] > 5 * drag["slow_skill"]


def test_cost_derivation_matches_the_simulator():
    """sigma*_b * sum_t c_t |df_t| (the combiner's cost in equity units) ~= the spread +
    slippage + commission the ExecutionSimulator actually charges a vol-targeted book."""
    from aurum.backtest.engine import run_backtest
    from aurum.portfolio.sizing import VolTargetSizer

    bars = make_synthetic_bars(4000, "H1", seed=3, model="gbm", weekend_gaps=False)
    idx = bars.index
    fc = pd.DataFrame({"x": np.where((np.arange(len(idx)) // 6) % 2 == 0, 0.6, -0.6)}, index=idx)
    # frictionless financing (swap is a holding cost, not a turnover cost) + standard fills
    cm = dataclasses.replace(CostModel.zero(), spread_multiplier=1.0, min_spread=0.10, slippage_fixed=0.02,
                             slippage_range_frac=0.02)
    tv = 0.10
    res = run_backtest(MarketData(bars), fc["x"], costs=cm, compute_metrics=False, start=idx[200],
                       sizer=VolTargetSizer(target_vol=tv, rebalance_band=0.0, max_leverage=50.0,
                                            drawdown_derisk=None))
    actual = float(res.costs[["spread", "slippage", "commission"]].sum().sum()) / float(res.equity.mean())
    # the sizer annualises vol with the NOMINAL bars/year: sigma*_b = target / sqrt(that)
    sig_b = tv / math.sqrt(nominal_bars_per_year(index_bar_minutes(idx)))
    c = ForecastCombiner().turnover_cost(fc, bars["close"], bars=bars, costs=cm)
    pred = sig_b * float((c * fc["x"].diff().abs()).loc[idx[200]:].sum())
    assert pred == pytest.approx(actual, rel=0.12)


def test_costs_change_the_ranking_and_zero_out_net_losers(world):
    bars, fc = world
    tr = slice(0, 3000)
    close = bars["close"].iloc[tr]
    gross = ForecastCombiner(max_weight=1.0).fit(fc.iloc[tr], close)
    net = ForecastCombiner(max_weight=1.0).fit(fc.iloc[tr], close, cost_per_turnover=COST)
    # gross: the churner looks best; net of costs it loses money and gets nothing
    assert gross.train_sharpe_["churn"] > gross.train_sharpe_["slow_skill"] > 0
    assert gross.weights_["churn"] > gross.weights_["slow_skill"] > 0
    assert net.train_sharpe_["churn"] < 0 < net.train_sharpe_["slow_skill"]
    assert net.weights_["churn"] == 0.0
    assert net.weights_["slow_skill"] > 0
    ex = net.explain()
    assert ex["cost_basis"] == "net of explicit cost_per_turnover"
    assert ex["train_sharpe_gross"]["churn"] == pytest.approx(gross.train_sharpe_["churn"])
    assert ex["train_turnover"]["churn"] > 5 * ex["train_turnover"]["slow_skill"]
    assert ex["avg_cost_per_turnover"] == pytest.approx(COST)
    assert ex["train_cost_per_bar"]["churn"] == pytest.approx(COST * ex["train_turnover"]["churn"])
    json.dumps(ex)
    assert any("GROSS" in n for n in gross.explain()["notes"])
    # a zero explicit cost reproduces the gross fit
    zero = ForecastCombiner(max_weight=1.0).fit(fc.iloc[tr], close, bars=bars, cost_per_turnover=0.0)
    pd.testing.assert_series_equal(zero.weights_, gross.weights_)
    with pytest.raises(ValueError):
        ForecastCombiner().fit(fc.iloc[tr], close, cost_per_turnover=-1.0)


def test_bars_path_equals_the_explicit_estimate(world):
    """fit(bars=...) == fit(cost_per_turnover=turnover_cost(...)): one cost definition."""
    bars, fc = world
    tr = slice(0, 3000)
    cm = CostModel(spread_multiplier=3.0, slippage_fixed=0.1)
    a = ForecastCombiner(max_weight=1.0).fit(fc.iloc[tr], bars["close"].iloc[tr], bars=bars, costs=cm)
    c = ForecastCombiner().turnover_cost(fc.iloc[tr], bars["close"].iloc[tr], bars=bars, costs=cm)
    b = ForecastCombiner(max_weight=1.0).fit(fc.iloc[tr], bars["close"].iloc[tr], cost_per_turnover=c)
    pd.testing.assert_series_equal(a.weights_, b.weights_)
    pd.testing.assert_series_equal(a.train_sharpe_, b.train_sharpe_)
    assert a.explain()["cost_basis"].startswith("net of estimated costs")
    # realistic costs already hurt the churner much more than the slow strategy
    assert (a.train_sharpe_gross_ - a.train_sharpe_)["churn"] > 5 * (a.train_sharpe_gross_ - a.train_sharpe_)["slow_skill"]


@pytest.mark.parametrize("method", METHODS)
def test_every_method_scores_net_streams(world, method):
    bars, fc = world
    tr = slice(0, 3000)
    comb = ForecastCombiner(method=method, max_weight=1.0).fit(fc.iloc[tr], bars["close"].iloc[tr],
                                                               cost_per_turnover=COST)
    assert comb.cost_basis_.startswith("net")
    w = comb.weights_
    assert (w >= 0).all() and w.sum() <= 1.0 + 1e-12
    if method != "sharpe_shrink":
        assert w.sum() == pytest.approx(1.0)       # risk-based methods allocate everything
    else:
        assert w["churn"] == 0.0
    streams = comb.unit_vol_streams(fc.iloc[tr], bars["close"].iloc[tr], cost_per_turnover=COST)
    np.testing.assert_allclose(comb.train_vol_.to_numpy(), streams.std(ddof=1).to_numpy(), rtol=1e-10)


def test_unallocated_risk_is_held_as_less_exposure(world):
    bars, fc = world
    tr = slice(0, 3000)
    comb = ForecastCombiner(max_weight=0.4).fit(fc.iloc[tr], bars["close"].iloc[tr], cost_per_turnover=COST)
    w = comb.weights_
    assert w["churn"] == 0.0
    assert w.sum() < 1.0 and w.max() <= 0.4 + 1e-12
    ex = comb.explain()
    assert ex["unallocated"] == pytest.approx(1.0 - w.sum()) and ex["allow_unallocated"] is True
    # FDM is a property of the allocated sub-portfolio: the combined forecast shrinks by sum(w)
    out = comb.combine(fc.iloc[3000:])
    expected = np.clip(comb.fdm_ * fc.iloc[3000:].to_numpy() @ w.to_numpy(), -1, 1)
    np.testing.assert_allclose(out.to_numpy(), expected)
    one = fc[["slow_skill"]].iloc[tr]
    single = ForecastCombiner(max_weight=0.4).fit(pd.concat([one, fc[["churn"]].iloc[tr]], axis=1),
                                                   bars["close"].iloc[tr], cost_per_turnover=COST)
    # two configured strategies: the cap relaxes to 1/N=0.5 (config-infeasible), the loser gets 0
    assert single.weights_["slow_skill"] == pytest.approx(0.5) and single.weights_["churn"] == 0.0
    assert single.fdm_ == pytest.approx(1.0)
    np.testing.assert_allclose(single.combine(fc.iloc[3000:]).to_numpy(),
                               np.clip(0.5 * fc["slow_skill"].iloc[3000:].to_numpy(), -1, 1))


def test_all_net_losers_leave_the_book_flat_by_default(world):
    bars, fc = world
    tr = slice(0, 3000)
    losers = pd.DataFrame({"a": fc["churn"], "b": -fc["slow_skill"]}).iloc[tr]
    comb = ForecastCombiner().fit(losers, bars["close"].iloc[tr], cost_per_turnover=COST)
    assert (comb.train_sharpe_ < 0).all()
    assert (comb.weights_ == 0.0).all()
    assert (comb.combine(losers) == 0.0).all()
    assert any("flat" in n for n in comb.notes_)
    legacy = ForecastCombiner(allow_unallocated=False).fit(losers, bars["close"].iloc[tr], cost_per_turnover=COST)
    assert legacy.weights_.sum() == pytest.approx(1.0)        # old: inverse-vol fallback
    assert any("fell back to inverse_vol" in n for n in legacy.notes_)


def test_negative_net_sharpe_is_not_rescued_by_shrinkage(world):
    """Shrinking toward a strongly positive cross-sectional mean can make a loser's SHRUNK
    Sharpe positive; its (net) Sharpe still rules it out (legacy mode keeps it)."""
    bars, fc = world
    tr = slice(0, 3000)
    rng = np.random.default_rng(11)
    fut = np.sign(np.log(bars["close"]).diff().shift(-1).fillna(0.0).to_numpy())
    f = pd.DataFrame({"good": np.clip(0.15 * fut + 0.6 * np.sign(fc["slow_skill"].to_numpy()), -1, 1),
                      "meh": np.clip(-0.04 * fut + rng.normal(0, 0.5, len(fut)), -1, 1)},
                     index=fc.index).iloc[tr]
    new = ForecastCombiner(shrinkage=0.9, max_weight=1.0).fit(f, bars["close"].iloc[tr])
    old = ForecastCombiner(shrinkage=0.9, max_weight=1.0, allow_unallocated=False).fit(f, bars["close"].iloc[tr])
    assert new.train_sharpe_["meh"] < 0 < new.train_sharpe_["good"]
    assert new.weights_["meh"] == 0.0 and new.weights_["good"] == pytest.approx(1.0)
    assert old.weights_["meh"] > 0.0                          # the pre-change behaviour


def test_cost_inputs_are_point_in_time(world):
    """Bars (spread/range) after the training window cannot change the fit."""
    bars, fc = world
    a = ForecastCombiner().fit(fc.iloc[:2500], bars["close"], bars=bars)
    mod = bars.copy()
    mod.iloc[2500:, mod.columns.get_loc("spread")] = 50.0
    mod.iloc[2500:, mod.columns.get_loc("high")] = mod["high"].iloc[2500:] * 1.5
    b = ForecastCombiner().fit(fc.iloc[:2500], bars["close"], bars=mod)
    pd.testing.assert_series_equal(a.weights_, b.weights_)
    pd.testing.assert_series_equal(a.train_sharpe_, b.train_sharpe_)
    assert a.fdm_ == b.fdm_


def test_cap_weights_unallocated_mode():
    w = cap_weights(np.array([10.0, 1.0, 0.0, 0.0]), 0.4, allow_unallocated=True)
    np.testing.assert_allclose(w, [0.4, 0.4, 0.0, 0.0])        # zero-scored entries never get weight
    w = cap_weights(np.array([3.0, 1.0, 1.0, 0.0]), 0.4, allow_unallocated=True)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.4 + 1e-12 and w[3] == 0.0
    np.testing.assert_allclose(cap_weights(np.zeros(3), 0.4, allow_unallocated=True), 0.0)
    np.testing.assert_allclose(cap_weights(np.array([1.0, 0.0]), 0.3, allow_unallocated=True), [0.3, 0.0])
    # the default contract is unchanged
    np.testing.assert_allclose(cap_weights(np.array([1.0, 0.0, 0.0, 0.0]), 0.4), [0.4, 0.2, 0.2, 0.2])


def test_old_pickled_combiner_still_combines_and_explains(world):
    """Artifacts written before the cost-aware change lack the new attributes."""
    bars, fc = world
    comb = ForecastCombiner().fit(fc.iloc[:2000], bars["close"].iloc[:2000])
    for attr in ("allow_unallocated", "cost_multiplier", "train_sharpe_gross_", "train_turnover_", "train_cost_",
                 "cost_basis_", "avg_cost_per_turnover_"):
        delattr(comb, attr)
    old = pickle.loads(pickle.dumps(comb))
    out = old.combine(fc.iloc[2000:])
    assert out.between(-1, 1).all()
    ex = old.explain()
    json.dumps(ex)
    assert ex["weights_sum"] == pytest.approx(float(old.weights_.sum()))
    assert "allow_unallocated=False" in repr(old)
