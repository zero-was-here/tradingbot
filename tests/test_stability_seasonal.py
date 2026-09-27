"""``intraday_seasonality``: cost-aware positions (no-trade bands over the weekly cycle).

Unit tests of the periodic dynamic programme, the TRAIN cost estimate, and end-to-end
checks on synthetic bars with a planted hour-of-week pattern: it must still be recovered,
while turnover is bounded and noise never trades.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.engine import run_backtest
from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.execution.costs import CostModel
from aurum.portfolio.sizing import VolTargetSizer
from aurum.strategies.seasonal import IntradaySeasonality, bucket_keys, periodic_cost_aware_positions

GRID_STEP = 0.025


def _md(bars: pd.DataFrame) -> MarketData:
    return MarketData(bars=bars)


def _planted_bars(n: int, seed: int, effects: dict[int, float], *, spread: float = 0.30,
                  wick: float = 0.2) -> pd.DataFrame:
    """GBM H1 bars whose bars OPENING at New York hour ``h`` drift by ``effects[h]`` sigma
    (open = previous close, so the open-to-close of a bar is its whole return)."""
    base = make_synthetic_bars(n, "H1", seed=seed, model="gbm")
    ny_hour = base.index.tz_convert("America/New_York").hour
    sigma = 0.16 / math.sqrt(252 * 23)
    r = np.random.default_rng(seed).standard_normal(n) * sigma
    for h, e in effects.items():
        r += e * sigma * (ny_hour == h).astype(float)
    c = 1800 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick,
                       "close": c, "spread": spread}, index=base.index)
    return make_bars(df, "H1")


def _decision_ny_hour(bars: pd.DataFrame) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(bars["available_at"]).tz_convert("America/New_York").hour)


def _orbit_value(f: np.ndarray, alpha: np.ndarray, kappa: np.ndarray, gamma: float) -> tuple[float, float]:
    df = np.abs(np.diff(np.r_[f[-1], f]))
    return float((f * alpha - 0.5 * gamma * f * f).sum()), float((kappa * df).sum())


# ---------------------------------------------------------------------------------------
# the periodic dynamic programme
# ---------------------------------------------------------------------------------------
def test_dp_without_costs_is_the_frictionless_table() -> None:
    rng = np.random.default_rng(0)
    alpha = rng.normal(0, 0.05, 48)
    gamma = 0.08
    f, info = periodic_cost_aware_positions(alpha, np.zeros(48), gamma)
    assert info["converged"]
    np.testing.assert_allclose(f, np.clip(alpha / gamma, -1, 1), atol=GRID_STEP / 2 + 1e-12)


def _turnover(f: np.ndarray) -> float:
    return float(np.abs(np.diff(np.r_[f[-1], f])).sum())


@pytest.mark.parametrize("edge", [0.5, 0.9, 1.5, 3.0])
def test_dp_lone_bucket_needs_to_beat_the_round_trip(edge: float) -> None:
    """A lone bucket worth ``edge`` round trips (enter + exit): below 1 the book never trades
    it (zero turnover — at most a constant residual it never pays to flatten, the no-trade
    band at zero edge); above, the bucket gets a bigger position than its neighbours, sized
    DOWN from the frictionless ``alpha / gamma`` (the dead zone is scaled to costs)."""
    L, c, gamma = 24, 0.05, 0.2
    alpha = np.zeros(L)
    alpha[7] = edge * 2 * c
    f, _ = periodic_cost_aware_positions(alpha, np.full(L, c), gamma)
    others = np.delete(f, 7)
    assert len(set(others)) == 1
    if edge <= 1.0:
        assert _turnover(f) == 0.0
    else:
        assert f[7] > others[0] and _turnover(f) > 0
        assert f[7] <= min(1.0, alpha[7] / gamma) + 1e-12
        # the step up at the bucket clears its cost: (f7 - f_other) alpha > 2c (f7 - f_other)
        assert alpha[7] > 2 * c
    frictionless = periodic_cost_aware_positions(alpha, np.zeros(L), gamma)[0]
    assert _turnover(f) < _turnover(frictionless)


def test_dp_holds_a_run_of_small_same_sign_buckets_as_one_position() -> None:
    """Five adjacent buckets each worth 0.6 x round trip — none tradeable alone — plus a small
    opposite-sign bucket inside the run are held as ONE position: one step up and one step
    down per cycle."""
    L, c, gamma = 48, 0.05, 0.05
    alpha = np.zeros(L)
    alpha[10:16] = 0.6 * 2 * c
    alpha[13] = -0.2 * 2 * c
    f, _ = periodic_cost_aware_positions(alpha, np.full(L, c), gamma)
    run = f[10:16]
    rest = np.delete(f, range(10, 16))
    assert len(set(run)) == 1 and len(set(rest)) == 1 and run[0] > rest[0]
    assert int((np.diff(np.r_[f[-1], f]) != 0).sum()) == 2
    alone = np.zeros(L)
    alone[10] = alpha[10]
    assert _turnover(periodic_cost_aware_positions(alone, np.full(L, c), gamma)[0]) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_dp_orbit_pays_for_its_costs_and_trades_less(seed: int) -> None:
    """Along the fitted orbit the (risk-adjusted) edge covers the costs it incurs — flat is
    always feasible, so the optimum is never worse — and turnover is below frictionless."""
    rng = np.random.default_rng(seed)
    L = 120
    alpha = rng.normal(0, 0.06, L)
    kappa = rng.uniform(0.03, 0.08, L)
    gamma = float(np.mean(np.abs(alpha))) / 0.5
    f, info = periodic_cost_aware_positions(alpha, kappa, gamma)
    assert info["converged"]
    gross, cost = _orbit_value(f, alpha, kappa, gamma)
    assert gross - cost >= -1e-12
    fric = np.clip(alpha / gamma, -1, 1)
    assert np.abs(np.diff(np.r_[f[-1], f])).sum() < 0.5 * np.abs(np.diff(np.r_[fric[-1], fric])).sum()


# ---------------------------------------------------------------------------------------
# cost estimate from TRAINING bars
# ---------------------------------------------------------------------------------------
def test_training_cost_matches_cost_model() -> None:
    bars = _planted_bars(3000, 1, {}, spread=0.05)            # below min_spread -> floored
    costs = {"min_spread": 0.10, "slippage_fixed": 0.03, "slippage_range_frac": 0.05,
             "commission_per_lot": 3.5}
    s = IntradaySeasonality(costs=costs)
    tt = s.training_targets(_md(bars))
    cm = CostModel(**costs)
    from aurum.strategies.trend import bar_volatility, log_close
    sigma = bar_volatility(log_close(bars), 240.0, 120)
    t = 2000
    nxt = bars.iloc[t + 1]
    per_side = (0.5 * cm.effective_spread(nxt["spread"]) + cm.slippage(nxt["high"] - nxt["low"], 0.0)
                + 3.5 / 100.0)
    assert tt.loc[bars.index[t], "cost"] == pytest.approx(per_side / bars["close"].iloc[t] / sigma.iloc[t])
    assert (tt["cost"] > 0).all()


# ---------------------------------------------------------------------------------------
# end to end on synthetic bars
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planted() -> pd.DataFrame:
    # a one-hour +0.6 sigma effect at the 10:00 NY bar, and a 4-hour block of -0.2 sigma bars
    # (02:00-05:00 NY) that is only worth trading as one held position
    return _planted_bars(26_000, 13, {10: 0.6, 2: -0.2, 3: -0.2, 4: -0.2, 5: -0.2})


def test_planted_pattern_is_recovered_net_of_costs(planted: pd.DataFrame) -> None:
    train = planted.iloc[:16_000]
    s = IntradaySeasonality().fit(_md(train))
    fric = IntradaySeasonality(cost_aware=False).fit(_md(train))
    fs = s.fit_summary_
    assert fs["cost_aware"] and fs["heterogeneous"] and fs["dp"]["converged"]
    f = s.generate(_md(planted)).to_numpy()
    h = _decision_ny_hour(planted)
    # decision at hour h holds the bar opening at h
    assert f[h == 10].mean() > 0.3
    assert f[np.isin(h, [2, 3, 4, 5])].mean() < -0.15
    block = pd.Series(s.table_)
    keys = bucket_keys(planted["available_at"], "America/New_York", 60)
    # the 4-hour block is held as one position on (almost) every weekday
    held = [len({block.get(int(d * 1440 + hh * 60)) for hh in (2, 3, 4, 5)}) == 1 for d in range(5)]
    assert sum(held) >= 4
    active = np.isin(h, [2, 3, 4, 5, 10])
    quiet = h >= 12                                    # well away from the planted hours
    assert np.abs(f[quiet]).mean() < 0.3 * np.abs(f[active]).mean()
    # bounded turnover: far below the frictionless table's
    ff = fric.generate(_md(planted)).to_numpy()
    assert np.abs(np.diff(f)).sum() < 0.35 * np.abs(np.diff(ff)).sum()
    assert fs["orbit_turnover_per_week"] < 0.35 * fs["frictionless_turnover_per_week"]
    assert fs["orbit_expected_gross_per_week"] >= 2.0 * fs["orbit_expected_cost_per_week"] - 1e-12
    assert len(np.unique(keys)) >= 115


def test_cost_aware_beats_frictionless_out_of_sample_net(planted: pd.DataFrame) -> None:
    """Out of sample (bars after the training slice) through the real engine and costs."""
    cut = 16_000
    md = _md(planted)
    out = {}
    for label, params in (("cost_aware", {}), ("frictionless", {"cost_aware": False})):
        s = IntradaySeasonality(**params).fit(_md(planted.iloc[:cut]))
        res = run_backtest(md, s.generate(md), sizer=VolTargetSizer(), costs=CostModel(),
                           start=planted.index[cut])
        out[label] = res.metrics
    assert out["cost_aware"]["trades_per_year"] < 0.4 * out["frictionless"]["trades_per_year"]
    assert out["cost_aware"]["total_costs"] < 0.4 * out["frictionless"]["total_costs"]
    assert out["cost_aware"]["sharpe"] > out["frictionless"]["sharpe"]
    assert out["cost_aware"]["sharpe"] > 1.0


def test_unseen_bucket_holds_the_preceding_position(planted: pd.DataFrame) -> None:
    s = IntradaySeasonality().fit(_md(planted.iloc[:16_000]))
    cyc = np.array(sorted(s.table_))
    # a Saturday decision (never seen in training) holds Friday's last position
    sat = pd.DatetimeIndex(["2024-06-08 16:00", "2024-06-08 17:00"], tz="UTC")
    fake = pd.DataFrame({"open": 1800.0, "high": 1801.0, "low": 1799.0, "close": 1800.0,
                         "spread": 0.3}, index=sat)
    got = s.generate(_md(make_bars(fake, "H1"))).to_numpy()
    fri_last = cyc[cyc < 5 * 1440].max()
    assert (got == s.table_[int(fri_last)]).all()
    # the same function of the clock whatever the prices
    f1 = s.generate(_md(planted)).to_numpy()
    flipped = planted.copy()
    flipped[["open", "high", "low", "close"]] = flipped[["open", "high", "low", "close"]].to_numpy()[::-1]
    flipped["high"] = flipped[["open", "high", "low", "close"]].max(axis=1)
    flipped["low"] = flipped[["open", "high", "low", "close"]].min(axis=1)
    assert np.array_equal(f1, s.generate(_md(flipped)).to_numpy())
    assert s.warmup_bars == 0


def test_cost_multiplier_zero_or_free_costs_is_the_frictionless_table(planted: pd.DataFrame) -> None:
    train = _md(planted.iloc[:8000])
    ref = IntradaySeasonality(cost_aware=False).fit(train).table_
    for params in ({"cost_multiplier": 0.0},
                   {"costs": {"min_spread": 0.0, "spread_multiplier": 0.0, "slippage_fixed": 0.0,
                              "slippage_range_frac": 0.0}}):
        s = IntradaySeasonality(**params).fit(train)
        assert not s.fit_summary_["cost_aware"]
        assert s.table_ == ref


def test_higher_costs_trade_less(planted: pd.DataFrame) -> None:
    train = _md(planted.iloc[:16_000])
    turn = [IntradaySeasonality(cost_multiplier=m).fit(train).fit_summary_["orbit_turnover_per_week"]
            for m in (0.5, 2.0, 8.0)]
    assert turn[0] >= turn[1] >= turn[2]
    assert turn[0] > turn[2]
