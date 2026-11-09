"""Monte Carlo + historical stress (aurum.research.montecarlo): seeded reproducibility, index
parity with aurum.research.stats, ordered fan bands, hand-checked path statistics, speed, gap
shocks through aurum.risk.var, run-directory loading and graceful errors.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import load_config
from aurum.research.montecarlo import (
    drawdown_exceedance,
    fan_bands,
    gap_shock_table,
    load_book,
    monte_carlo,
    parse_horizon,
    path_metrics,
    plot_distributions,
    plot_fan,
    risk_of_ruin,
    simulate_paths,
    stationary_bootstrap_paths,
    stress_run,
    worst_windows,
)
from aurum.research.stats import stationary_bootstrap_indices
from aurum.research.walkforward import run_walk_forward


def _returns(n=1500, seed=0, mu=0.0003, sd=0.007):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range("2016-01-04", periods=n, tz="UTC"))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mc_run")
    base = {"seed": 3, "data": {"synthetic": {"n": 6000, "model": "trend", "seed": 5, "regime_params": {"phi": 0.1}}},
            "strategies": [{"name": "ema_cross", "params": {"fast": 8, "slow": 32}, "id": "ema_fast"},
                           {"name": "zscore_fade"}],
            "combiner": {"method": "fixed"},
            "walkforward": {"train": "4M", "test": "6W", "embargo": 6, "n_boot": 100, "pbo_splits": 8,
                            "min_train_bars": 200, "executor": "serial"},
            "output": {"dir": str(tmp), "tearsheet": False}}
    cfg = load_config(base, env={})
    run_walk_forward(cfg.data.load(), cfg, out_dir=tmp / "run")
    return tmp / "run"


# ---- horizon / simulation --------------------------------------------------------------------------
def test_parse_horizon():
    assert parse_horizon("2Y") == 504 and parse_horizon("6m") == 126 and parse_horizon("4W") == 20
    assert parse_horizon("30D") == 30 and parse_horizon("30") == 30 and parse_horizon(10) == 10
    assert parse_horizon("1.5Y") == 378
    for bad in ("0D", "2X", "", True, -3):
        with pytest.raises(ValueError):
            parse_horizon(bad)


def test_index_scheme_is_stats_stationary_bootstrap():
    a = stationary_bootstrap_indices(250, 40, 6.0, np.random.default_rng(9))
    b = stationary_bootstrap_paths(250, 40, 250, 6.0, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    c = stationary_bootstrap_paths(250, 40, 600, 6.0, np.random.default_rng(9))
    assert c.shape == (40, 600) and c.min() >= 0 and c.max() < 250
    # blocks: consecutive indices advance by one (mod n) far more often than for an iid draw
    steps = (np.diff(c, axis=1) % 250) == 1
    assert 0.75 < steps.mean() < 0.9          # P(continue block) = 1 - 1/6
    with pytest.raises(ValueError):
        stationary_bootstrap_paths(10, 2, 5, 0.5, np.random.default_rng(0))


def test_bootstrap_reproducible_with_seed():
    r = _returns()
    a, ba = simulate_paths(r, 3000, "1Y", seed=42)
    b, bb = simulate_paths(r, 3000, "1Y", seed=42)
    c, _ = simulate_paths(r, 3000, "1Y", seed=43)
    np.testing.assert_array_equal(a, b)
    assert ba == bb and not np.array_equal(a, c)
    assert a.shape == (3000, 253) and np.all(a[:, 0] == 1.0)
    m1, m2 = monte_carlo(r, n_paths=2000, seed=7), monte_carlo(r, n_paths=2000, seed=7)
    pd.testing.assert_frame_equal(m1.metrics, m2.metrics)
    pd.testing.assert_frame_equal(m1.fan, m2.fan)


def test_simulated_moments_match_history():
    r = _returns(2000, seed=1, mu=0.0004, sd=0.006)
    eq, _ = simulate_paths(r, 4000, 252, seed=0)
    sim = np.diff(np.log(eq), axis=1)
    assert np.log1p(r).mean() == pytest.approx(sim.mean(), rel=0.05)
    assert np.log1p(r).std() == pytest.approx(sim.std(), rel=0.03)


def test_fan_bands_ordered_and_start_at_initial():
    mc = monte_carlo(_returns(), n_paths=5000, horizon="2Y", seed=3, initial_equity=100_000.0)
    f = mc.fan
    assert list(f.columns) == ["q05", "q25", "q50", "q75", "q95"] and len(f) == 505
    assert (np.diff(f.to_numpy(), axis=1) >= -1e-9).all()
    assert np.allclose(f.iloc[0], 100_000.0)
    assert (f["q95"].iloc[-1] - f["q05"].iloc[-1]) > (f["q95"].iloc[10] - f["q05"].iloc[10])  # fans out
    assert list(fan_bands(mc.paths, (0.9, 0.1)).columns) == ["q10", "q90"]


def test_speed_10k_paths_2y_under_two_seconds():
    r = _returns(2500)
    monte_carlo(r, n_paths=200, seed=0)          # warm-up (imports)
    t0 = time.perf_counter()
    mc = monte_carlo(r, n_paths=10_000, horizon="2Y", seed=0)
    assert time.perf_counter() - t0 < 2.0
    assert mc.paths.shape == (10_000, 505)


# ---- path statistics (hand-checked) --------------------------------------------------------------
def test_path_metrics_hand_checked():
    paths = np.array([
        [1.0, 1.1, 0.88, 0.99, 1.12, 1.2],     # peak 1.1, trough 0.88 (-20%) at t=2, recovers at t=4
        [1.0, 0.9, 0.8, 0.85, 0.9, 0.95],      # never recovers; trough at t=2
        [1.0, 1.01, 1.02, 1.03, 1.04, 1.05],   # no drawdown
    ])
    m = path_metrics(paths, periods=5)
    assert m["max_drawdown"].iloc[0] == pytest.approx(0.88 / 1.1 - 1)
    assert m["trough_day"].iloc[0] == 2 and m["time_to_recover_days"].iloc[0] == 2
    assert bool(m["recovered"].iloc[0]) and m["longest_underwater_days"].iloc[0] == 2
    assert m["max_drawdown"].iloc[1] == pytest.approx(-0.2) and math.isnan(m["time_to_recover_days"].iloc[1])
    assert not bool(m["recovered"].iloc[1]) and m["longest_underwater_days"].iloc[1] == 5
    assert m["max_drawdown"].iloc[2] == 0.0 and bool(m["recovered"].iloc[2])
    assert m["longest_underwater_days"].iloc[2] == 0
    assert m["cagr"].iloc[0] == pytest.approx(0.2)         # 5 steps = one "year" at periods=5
    assert m["min_equity"].iloc[1] == pytest.approx(0.8)
    ex = drawdown_exceedance(m["max_drawdown"], (0.05, 0.15, 0.25))
    assert list(ex["probability"]) == pytest.approx([2 / 3, 2 / 3, 0.0])
    ru = risk_of_ruin(paths, (0.1, 0.2, 0.3))
    # -10%: paths 0 (low 0.88) and 1 (low 0.80); -20%: only path 1 (touches 0.80 exactly); -30%: none
    assert list(ru["p_touch"]) == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert list(ru["p_end_below"]) == pytest.approx([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        path_metrics(np.ones(5))


def test_monotone_tables_and_recovery_summary():
    mc = monte_carlo(_returns(mu=-0.0002, sd=0.01), n_paths=3000, horizon=252, seed=1)
    assert (np.diff(mc.dd_table["probability"]) <= 0).all()
    assert (np.diff(mc.ruin["p_touch"]) <= 0).all() and (mc.ruin["p_end_below"] <= mc.ruin["p_touch"]).all()
    rec = mc.recovery()
    assert 0 <= rec["p_not_recovered"] <= 1 and rec["ttr_median_days"] <= rec["ttr_p95_days"]
    d = mc.distribution()
    assert list(d.index) == ["q05", "q25", "q50", "q75", "q95", "mean"]
    s = mc.summary()
    assert s["n_paths"] == 3000 and s["horizon_days"] == 252 and json.dumps(s)


def test_bad_returns_raise():
    with pytest.raises(ValueError):
        monte_carlo([0.01])
    with pytest.raises(ValueError):
        monte_carlo([0.01, -1.0, 0.02])
    with pytest.raises(ValueError):
        simulate_paths(_returns(), 0)


# ---- deterministic stress ---------------------------------------------------------------------------
def test_worst_windows_hand_checked():
    r = pd.Series([0.01, -0.02, -0.03, 0.05, -0.01], index=pd.bdate_range("2020-01-06", periods=5, tz="UTC"))
    w = worst_windows(r, (1, 2, 3, 9), equity=1000.0)
    assert w.loc[0, "worst_return"] == pytest.approx(-0.03) and w.loc[0, "start"] == "2020-01-08"
    assert w.loc[1, "worst_return"] == pytest.approx(0.98 * 0.97 - 1)
    assert (w.loc[1, "start"], w.loc[1, "end"]) == ("2020-01-07", "2020-01-08")
    assert w.loc[2, "worst_return"] == pytest.approx(min(1.01 * 0.98 * 0.97, 0.98 * 0.97 * 1.05) - 1)
    assert math.isnan(w.loc[3, "worst_return"])            # window longer than the history
    assert w.loc[0, "pnl_usd_at_current_equity"] == pytest.approx(-30.0)


def test_gap_shock_table_uses_risk_var():
    from aurum.risk.var import gap_shock

    t = gap_shock_table(1.0, 2000.0, 100_000.0, shocks=[-0.05, 0.05], spread=0.5)
    row = t.set_index("shock_pct").loc[-0.05]
    assert row["pnl_usd"] == pytest.approx(gap_shock(1.0, 2000.0, -0.05, spread=0.5))
    assert row["pnl_usd"] == pytest.approx(-10_000.0 - 25.0)
    assert t["pnl_usd"].iloc[0] <= t["pnl_usd"].iloc[-1]   # sorted worst first
    default = gap_shock_table(-0.5, 2000.0, 100_000.0)
    assert len(default) >= 6 and default["pnl_usd"].iloc[0] < 0


# ---- run directories ------------------------------------------------------------------------------
def test_stress_run_on_run_dir(run_dir, tmp_path):
    rep = stress_run(run_dir, n_paths=2000, horizon="1Y", seed=5)
    bd = load_book(run_dir)
    assert rep.exposure["equity"] == pytest.approx(float(bd.equity.iloc[-1]))
    assert rep.exposure["price_source"] == "bars (last close)"      # synthetic config reloads its bars
    assert rep.mc.n_history == len(rep.daily_returns) > 100
    assert rep.gold_moves is not None and len(rep.gold_moves) >= 3
    assert set(rep.gap_shocks["exposure"]) >= {"current"}
    out = rep.save(tmp_path / "stress")
    for f in ("stress.json", "fan.png", "distributions.png", "distribution.csv", "fan_bands.csv",
              "drawdown_exceedance.csv", "risk_of_ruin.csv", "worst_windows.csv", "gap_shocks.csv"):
        assert (out / f).exists(), f
    js = json.loads((out / "stress.json").read_text())
    assert js["monte_carlo"]["n_paths"] == 2000
    # without data: price from the last fill (approximate), noted
    rep2 = stress_run(run_dir, n_paths=500, horizon=60, load_data=False)
    if rep2.exposure.get("price_source") == "last fill mid (approximate)":
        assert any("last fill" in n for n in rep2.notes)
    pd.testing.assert_frame_equal(rep2.worst_windows, rep.worst_windows)


def test_stress_run_on_result_and_series(run_dir):
    bd = load_book(run_dir, "ema_fast")
    rep = stress_run(bd.equity, n_paths=500, horizon=100, price=2000.0)
    assert rep.source == "equity series" and rep.exposure["lots"] == 0.0
    assert any("flat" in n for n in rep.notes)
    with pytest.raises(TypeError):
        stress_run(42)


def test_load_book_errors(run_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_book(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="books in this run"):
        load_book(run_dir, "no_such_book")
    with pytest.raises(FileNotFoundError):
        load_book(run_dir, phase="holdout")


def test_plots_return_figures():
    from matplotlib.figure import Figure

    mc = monte_carlo(_returns(), n_paths=500, horizon=100, seed=0)
    assert isinstance(plot_fan(mc), Figure) and isinstance(plot_distributions(mc), Figure)


def test_stress_run_on_backtest_result():
    from aurum.backtest.engine import run_backtest
    from aurum.data.synthetic import make_synthetic_bars
    from aurum.portfolio.sizing import VolTargetSizer

    bars = make_synthetic_bars(3000, "H1", seed=4, model="trend")
    fc = pd.Series(0.6, index=bars.index)
    res = run_backtest(bars, fc, sizer=VolTargetSizer())
    rep = stress_run(res, n_paths=500, horizon=100, bars=bars, shocks={"crash": -0.08})
    assert rep.source == "BacktestResult"
    lots = float(res.position_close.iloc[-1])
    assert rep.exposure["lots"] == pytest.approx(lots) and lots > 0
    assert rep.exposure["price"] == pytest.approx(float(bars["close"].iloc[-1]))
    crash = rep.gap_shocks.loc[(rep.gap_shocks["exposure"] == "current") & (rep.gap_shocks["scenario"] == "crash")]
    assert float(crash["pnl_usd"].iloc[0]) < 0                # a long book loses on a gap down
    assert "peak" in rep.exposure and rep.exposure["peak"]["leverage"] > 0
