"""Parameter sweeps (aurum.research.sweep): grid parsing, plateau / knife-edge scoring, DSR with
n_trials = grid size (+ sweep ledger), standalone-book equivalence, holdout exclusion, outputs.

Synthetic data only (aurum.data.synthetic); every output goes to tmp_path.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import ConfigError, load_config
from aurum.research.stats import sharpe_summary
from aurum.research.sweep import (
    SWEEP_LEDGER,
    grid_points,
    parse_grid,
    plateau_score,
    plot_sweep,
    point_label,
    read_sweep_ledger,
    research_window,
    run_sweep,
)
from aurum.research.walkforward import run_walk_forward


def _cfg(tmp_path, **wf):
    base = {"train": "4M", "test": "6W", "embargo": 6, "n_boot": 100, "pbo_splits": 8, "min_train_bars": 200,
            "executor": "serial"}
    base.update(wf)
    return load_config({
        "seed": 3,
        "data": {"synthetic": {"n": 6000, "model": "trend", "seed": 11, "regime_params": {"phi": 0.12}}},
        "strategies": [{"name": "ema_cross", "params": {"vol_halflife": 120.0}}],
        "walkforward": base,
        "output": {"dir": str(tmp_path / "out"), "tearsheet": False},
    }, env={})


@pytest.fixture(scope="module")
def tmp_mod(tmp_path_factory):
    return tmp_path_factory.mktemp("sweep")


@pytest.fixture(scope="module")
def cfg_md(tmp_mod):
    cfg = _cfg(tmp_mod)
    return cfg, cfg.data.load()


@pytest.fixture(scope="module")
def sweep(cfg_md, tmp_mod):
    cfg, md = cfg_md
    return run_sweep(md, cfg, "ema_cross", ["fast=4,8", "slow=8,32"], executor="serial",
                     out_dir=tmp_mod / "out" / "sweep_a")


# ---- grid parsing --------------------------------------------------------------------------------------
def test_parse_grid_cli_specs_ranges_and_sorting():
    g = parse_grid(["fast=64,16,32,16", "slow=100:300:100", "mode=b,a", "flag=true,false"])
    assert g == {"fast": [16, 32, 64], "slow": [100, 200, 300], "mode": ["b", "a"], "flag": [True, False]}
    assert parse_grid({"x": [0.5, 0.1]}) == {"x": [0.1, 0.5]}
    assert parse_grid({"x": 3}) == {"x": [3]}
    assert len(grid_points(g)) == 3 * 3 * 2 * 2
    assert point_label({"fast": 16, "slow": 64}) == "fast=16,slow=64"
    for bad in (["fast"], ["=1,2"], ["fast=1,2", "fast=3"], []):
        with pytest.raises(ConfigError):
            parse_grid(bad)
    with pytest.raises(ConfigError):
        parse_grid(["x=1:5:0"])


# ---- plateau / knife edge ----------------------------------------------------------------------------
def _surface(fn, n=5):
    grid = {"a": list(range(n)), "b": list(range(10, 10 + n))}
    rows = [{"a": a, "b": b, "sharpe": fn(i, j), "status": "ok"}
            for i, a in enumerate(grid["a"]) for j, b in enumerate(grid["b"])]
    return pd.DataFrame(rows), grid


def test_plateau_score_flags_knife_edge_spike():
    t, g = _surface(lambda i, j: 1.0 if (i, j) == (2, 2) else 0.1)
    p = plateau_score(t, g, "sharpe")
    assert p["best"] == "a=2,b=12" and p["n_neighbours"] == 8
    assert p["plateau_ratio"] == pytest.approx(0.1)
    assert p["knife_edge"] is True and p["verdict"] == "knife-edge"
    assert p["drop_sd"] > 2


def test_plateau_score_smooth_hill_is_robust():
    t, g = _surface(lambda i, j: 1.0 - 0.05 * ((i - 2) ** 2 + (j - 2) ** 2))
    p = plateau_score(t, g, "sharpe")
    assert p["best"] == "a=2,b=12"
    assert 0.75 <= p["plateau_ratio"] < 1.0
    assert p["knife_edge"] is False and p["verdict"] == "robust plateau"
    assert p["robust_best"] == "a=2,b=12"


def test_plateau_centre_prefers_plateau_over_isolated_peak():
    # an isolated spike at a corner beats a broad plateau by a hair: the plateau centre wins
    def f(i, j):
        if (i, j) == (0, 0):
            return 1.2
        return 1.0 if (2 <= i <= 4 and 2 <= j <= 4) else 0.0
    t, g = _surface(f)
    p = plateau_score(t, g, "sharpe")
    assert p["best"] == "a=0,b=10" and p["knife_edge"] is True
    assert p["robust_best"] == "a=3,b=13" and p["robust_best_value"] == 1.0


def test_plateau_score_invalid_points_lower_is_better_and_undefined():
    t, g = _surface(lambda i, j: 0.5 if (i, j) == (1, 1) else 0.45)
    t.loc[(t["a"] == 1) & (t["b"] == 12), "status"] = "invalid"
    t.loc[(t["a"] == 0) & (t["b"] == 10), "sharpe"] = np.nan
    p = plateau_score(t, g, "sharpe")
    assert p["n_valid"] == 23 and p["n_neighbours"] == 6     # 8 - invalid - NaN
    # lower is better: best = smallest cost, ratio = best / neighbour mean
    c = t.assign(total_costs=lambda d: 100.0 + 10 * d["a"] + d["b"])
    pc = plateau_score(c.assign(status="ok"), g, "total_costs")
    assert pc["maximize"] is False and pc["best"] == "a=0,b=10"
    assert 0 < pc["plateau_ratio"] < 1 and pc["knife_edge"] is False
    neg, g2 = _surface(lambda i, j: -0.1 - 0.01 * (i + j))
    pn = plateau_score(neg, g2, "sharpe")
    assert pn["knife_edge"] is None and pn["verdict"] == "undefined"
    # max drawdown is negative by construction: best = closest to 0, ratio = best / neighbour mean
    dd = neg.rename(columns={"sharpe": "max_drawdown"})
    pd_ = plateau_score(dd, g2, "max_drawdown")
    assert pd_["maximize"] is True and pd_["best"] == "a=0,b=10"
    assert pd_["plateau_ratio"] == pytest.approx(-0.1 / np.mean([-0.11, -0.11, -0.12]))
    empty = plateau_score(t.assign(status="error"), g, "sharpe")
    assert empty["best"] is None and empty["verdict"] == "undefined"


def test_plateau_flags_optimum_on_grid_edge():
    t, g = _surface(lambda i, j: 1.0 - 0.05 * (i + j))          # best at the (0, 0) corner
    p = plateau_score(t, g, "sharpe")
    assert p["best"] == "a=0,b=10" and p["edge_params"] == ["a", "b"] and p["n_neighbours"] == 3
    hill, g2 = _surface(lambda i, j: 1.0 - 0.05 * ((i - 2) ** 2 + (j - 2) ** 2))
    assert plateau_score(hill, g2, "sharpe")["edge_params"] == []


def test_plateau_one_dimensional_grid():
    t = pd.DataFrame({"n": [5, 10, 20, 40], "sharpe": [0.2, 0.9, 0.85, 0.3], "status": "ok"})
    p = plateau_score(t, {"n": [5, 10, 20, 40]}, "sharpe")
    assert p["best"] == "n=10" and p["n_neighbours"] == 2
    assert p["plateau_ratio"] == pytest.approx((0.2 + 0.85) / 2 / 0.9)


# ---- run_sweep ---------------------------------------------------------------------------------------
def test_sweep_table_invalid_points_and_dsr_uses_grid_size(sweep):
    t = sweep.table
    assert len(t) == 4
    assert set(t["status"]) == {"ok", "invalid"}
    inv = t.loc[t["status"] == "invalid"]
    assert list(inv["label"]) == ["fast=8,slow=8"] and "fast < slow" in inv["error"].iloc[0]
    for c in ("sharpe", "cagr", "max_drawdown", "n_trades", "turnover_lots_per_year", "total_costs"):
        assert c in t.columns
    n_valid = int((t["status"] == "ok").sum())
    assert sweep.n_trials == n_valid == 3 and sweep.best["n_trials"] == 3
    best = sweep.best["label"]
    assert best == t.loc[t["sharpe"].idxmax(), "label"]
    s = sharpe_summary(sweep.returns[best].dropna().to_numpy(), 252.0, n_trials=3, n_boot=100, seed=3)
    assert sweep.best["dsr"] == pytest.approx(s["dsr"]) and sweep.best["psr"] == pytest.approx(s["psr"])
    s1 = sharpe_summary(sweep.returns[best].dropna().to_numpy(), 252.0, n_trials=1, bootstrap=False)
    assert sweep.best["dsr"] < s1["dsr"]         # deflating for the search lowers the DSR
    assert math.isfinite(sweep.best["dsr_xs"])
    assert sweep.label == "exploratory"
    assert list(sweep.returns.columns) == list(t.loc[t["status"] == "ok", "label"])


def test_sweep_point_equals_strategy_book_of_a_walk_forward(sweep, cfg_md, tmp_mod):
    cfg, md = cfg_md
    c2 = load_config({**cfg.to_dict(), "strategies": [
        {"name": "ema_cross", "params": {"vol_halflife": 120.0, "fast": 4, "slow": 32}, "id": "pt"},
        {"name": "ema_cross", "params": {"vol_halflife": 120.0, "fast": 4, "slow": 8}, "id": "other"}],
        "source": None, "output": {"save_results": False, "dir": str(tmp_mod / "out")}}, env={})
    rep = run_walk_forward(md, c2, write=False)
    row = sweep.table.set_index("label").loc["fast=4,slow=32"]
    assert row["sharpe"] == pytest.approx(rep.stats.loc["pt", "sharpe"], abs=1e-12)
    assert row["n_trades"] == rep.stats.loc["pt", "n_trades"]
    assert row["total_costs"] == pytest.approx(rep.stats.loc["pt", "total_costs"])


def test_sweep_outputs_and_ledger_adds_prior_trials(sweep, cfg_md, tmp_mod):
    d = sweep.out_dir
    for f in ("sweep.csv", "sweep.json", "sweep.png", "returns.csv"):
        assert (d / f).exists(), f
    js = json.loads((d / "sweep.json").read_text())
    assert js["label"] == "exploratory" and js["n_trials"] == 3 and js["plateau"]["best"] == sweep.best["label"]
    ledger = d.parent / SWEEP_LEDGER
    entries = read_sweep_ledger(ledger)
    assert len(entries) == 1 and entries[0]["n_valid"] == 3 and len(entries[0]["config_hashes"]) == 3
    # a second sweep over the same window: 1 shared point (not a new trial) + 1 new point
    cfg, md = cfg_md
    res2 = run_sweep(md, cfg, "ema_cross", {"fast": [4], "slow": [32, 64]}, executor="serial",
                     out_dir=tmp_mod / "out" / "sweep_b")
    assert res2.prior["n_prior_sweeps"] == 1
    assert res2.prior["n_prior_configs"] == 2          # fast=4/slow=8 and fast=8/slow=32 of the first sweep
    assert res2.n_trials == 2 + 2
    assert any("earlier sweep" in n for n in res2.notes)
    assert len(read_sweep_ledger(ledger)) == 2
    # ledger off: no prior trials, nothing appended
    res3 = run_sweep(md, cfg, "ema_cross", {"fast": [4], "slow": [32]}, executor="serial", write=False,
                     prior_trials=5)
    assert res3.n_trials == 1 + 5 and res3.out_dir is None
    assert len(read_sweep_ledger(ledger)) == 2


def test_sweep_threads_match_serial(sweep, cfg_md):
    cfg, md = cfg_md
    res = run_sweep(md, cfg, "ema_cross", ["fast=4,8", "slow=8,32"], executor="thread", n_jobs=3, write=False)
    a = sweep.table.set_index("label")["sharpe"]
    b = res.table.set_index("label")["sharpe"]
    pd.testing.assert_series_equal(a, b)


def test_sweep_process_pool_matches_serial(sweep, cfg_md):
    cfg, md = cfg_md
    res = run_sweep(md, cfg, "ema_cross", {"fast": [4], "slow": [8, 32]}, executor="process", n_jobs=2,
                    write=False)
    assert res.settings["executor"] == "process"
    a = sweep.table.set_index("label")["sharpe"]
    for lab, v in res.table.set_index("label")["sharpe"].items():
        assert v == pytest.approx(a[lab], abs=1e-12)


def test_sweep_never_touches_holdout(tmp_path):
    cfg0 = _cfg(tmp_path)
    md = cfg0.data.load()
    hs = md.bars.index[-1500]
    cfg = _cfg(tmp_path, holdout_start=str(hs))
    md_r, notes = research_window(md, cfg)
    assert md_r.bars.index[-1] < hs and any("holdout_start" in n for n in notes)
    # end= truncates further (date-only end covers the whole day)
    md_e, notes_e = research_window(md, cfg0, end=str(md.bars.index[3000].date()))
    assert md_e.bars.index[-1].date() == md.bars.index[3000].date() and any("end=" in n for n in notes_e)
    # a holdout already evaluated (ledger) is excluded even when the config has no holdout
    out = tmp_path / "out"
    (out / "protocol").mkdir(parents=True)
    spent = md.bars.index[-2000]
    (out / "protocol" / "holdout_ledger.jsonl").write_text(json.dumps(
        {"symbol": "XAUUSD", "holdout_start": str(spent), "holdout_end": str(md.bars.index[-1]),
         "config_hash": "x"}) + "\n")
    md_l, notes_l = research_window(md, cfg0)
    assert md_l.bars.index[-1] < spent and any("already evaluated" in n for n in notes_l)
    md_off, _ = research_window(md, cfg0, respect_holdout_ledgers=False)
    assert len(md_off.bars) == len(md.bars)
    other = research_window(md, cfg0, holdout_ledgers=[])[0]
    assert len(other.bars) == len(md.bars)
    # the sweep itself: every point config has holdout_start null; window ends before the holdout
    res = run_sweep(md, cfg, "ema_cross", {"fast": [8], "slow": [32]}, executor="serial", write=False)
    assert pd.Timestamp(res.window["end"]) < spent < hs
    assert pd.Timestamp(res.window["oos_end"]) < spent
    with pytest.raises(ConfigError):
        research_window(md, _cfg(tmp_path, holdout_start=str(md.bars.index[0])))


def test_sweep_errors_and_single_split(cfg_md):
    cfg, md = cfg_md
    with pytest.raises(ConfigError):
        run_sweep(md, cfg, "ema_cross", {"fast": [64], "slow": [8]}, executor="serial", write=False)
    with pytest.raises(ConfigError):
        run_sweep(md, cfg, "no_such_strategy", {"fast": [4]}, executor="serial", write=False)
    res = run_sweep(md, cfg, "ema_cross", {"fast": [8]}, walkforward=False, start=str(md.bars.index[3000]),
                    executor="serial", write=False)
    assert res.walkforward is False and any("ONE train/test split" in n for n in res.notes)
    assert res.table["status"].iloc[0] == "ok"
    assert pd.Timestamp(res.window["oos_start"]) >= md.bars.index[3000]


def test_plot_sweep_figures(sweep):
    from matplotlib.figure import Figure

    assert isinstance(sweep.plot(), Figure)
    assert isinstance(sweep.plot(metric="max_drawdown"), Figure)
    t = pd.DataFrame({"n": [5, 10, 20], "sharpe": [0.1, 0.5, np.nan], "status": ["ok", "ok", "error"]})
    assert isinstance(plot_sweep(t, {"n": [5, 10, 20]}, "sharpe", best={"n": 10}), Figure)
    g = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    t3 = pd.DataFrame([{**p, "sharpe": float(sum(p.values())), "status": "ok"} for p in grid_points(g)])
    assert isinstance(plot_sweep(t3, g, "sharpe", x="c", y="a"), Figure)
    with pytest.raises(ValueError):
        plot_sweep(t3, g, "sharpe", x="a", y="a")
