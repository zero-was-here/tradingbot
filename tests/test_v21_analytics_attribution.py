"""Explainability (aurum.research.attribution): contributions replay the combiner exactly (sum to
the combined forecast within clipping), the sizing replay reproduces the saved requested lots,
PnL attribution sums to the book's total PnL (per component too), holdout-phase reading, report
input, graceful errors and the desk journal summary.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import load_config
from aurum.research.attribution import (
    RunData,
    contributions,
    desk_journal_summary,
    explain_decision,
    load_run,
    plot_attribution,
    plot_waterfall,
    pnl_attribution,
)
from aurum.research.walkforward import run_walk_forward


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("attr_run")
    base = {"seed": 3, "data": {"synthetic": {"n": 7000, "model": "trend", "seed": 5, "regime_params": {"phi": 0.1}}},
            "strategies": [{"name": "ema_cross", "params": {"fast": 8, "slow": 32}, "id": "ema_fast", "weight": 2.0},
                           {"name": "ema_cross", "params": {"fast": 24, "slow": 96}, "id": "ema_slow"},
                           {"name": "zscore_fade"}],
            "combiner": {"method": "fixed"},
            "walkforward": {"train": "4M", "test": "6W", "embargo": 6, "n_boot": 100, "pbo_splits": 8,
                            "min_train_bars": 200, "executor": "serial"},
            "output": {"dir": str(tmp), "tearsheet": False}}
    cfg = load_config(base, env={})
    md = cfg.data.load()
    base["walkforward"]["holdout_start"] = str(md.bars.index[-800])
    cfg = load_config(base, env={})
    rep = run_walk_forward(md, cfg, out_dir=tmp / "run", holdout_ledger=False)
    return {"dir": tmp / "run", "report": rep, "md": md}


# ---- contributions --------------------------------------------------------------------------------
def test_contributions_sum_to_combined_forecast(run):
    r = load_run(run["dir"])
    assert r.strategies == ["ema_fast", "ema_slow", "zscore_fade"]
    c = contributions(r)
    np.testing.assert_allclose(c["raw"], c[r.strategies].sum(axis=1), atol=1e-15)
    np.testing.assert_allclose(c["raw"].clip(-1, 1), c["saved"], atol=1e-12)    # replay == saved, all bars
    w = pd.read_csv(run["dir"] / "weights.csv")
    assert w.loc[0, "ema_fast"] == pytest.approx(0.5)      # fixed weights 2:1:1 normalised


def test_contributions_respect_clipping():
    idx = pd.date_range("2021-01-04", periods=3, freq="h", tz="UTC")
    r = RunData(forecasts=pd.DataFrame({"a": [1.0, 0.2, -1.0], "b": [1.0, 0.1, -1.0]}, index=idx),
                combined=pd.Series([1.0, 0.3, -1.0], index=idx), folds=pd.Series("0", index=idx),
                weights=pd.DataFrame({"a": [0.5], "b": [0.5]}, index=["0"]), fdm=pd.Series({"0": 2.0}),
                book=None, risk_events=None, config=None)
    c = contributions(r)
    assert list(c["raw"]) == pytest.approx([2.0, 0.3, -2.0])
    assert list(c["combined"]) == pytest.approx([1.0, 0.3, -1.0]) == list(c["saved"])
    ex = explain_decision(r, idx[0], load_data=False)
    assert ex.raw_combined == pytest.approx(2.0) and ex.combined == 1.0
    assert ex.clip_adjustment == pytest.approx(-1.0)
    assert ex.table["contribution"].sum() == pytest.approx(ex.raw_combined)
    assert ex.table["share"].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no combiner weights"):
        contributions(RunData(forecasts=r.forecasts, combined=r.combined, folds=pd.Series("9", index=idx),
                              weights=r.weights, fdm=r.fdm, book=None, risk_events=None, config=None))


# ---- decision explanation -------------------------------------------------------------------------
def test_explain_decision_replays_combiner_and_sizing(run):
    r = load_run(run["dir"])
    c = contributions(r)
    t = c.index[len(c) // 2]
    ex = explain_decision(run["dir"], t)
    assert ex.time == t and ex.phase == "research" and not ex.notes
    w = pd.read_csv(run["dir"] / "weights.csv").set_index("fold")
    row = w.loc[int(ex.fold)]
    tab = ex.table.set_index("strategy")
    for s in r.strategies:
        assert tab.loc[s, "weight"] == pytest.approx(row[s])
        assert tab.loc[s, "fdm"] == pytest.approx(row["fdm"])
        assert tab.loc[s, "contribution"] == pytest.approx(row[s] * row["fdm"] * tab.loc[s, "forecast"])
    assert tab["contribution"].sum() == pytest.approx(ex.raw_combined)
    assert ex.combined == pytest.approx(ex.saved_combined)
    # the run's synthetic config reloads the same bars: the sizer replay reproduces the saved request
    s = ex.sizing
    assert s["replay_matches_saved"] is True
    assert s["replayed_lots"] == pytest.approx(s["saved_requested_lots"])
    assert 0.03 <= s["vol_ann"] <= 2.0 and s["price"] > 0 and s["equity"] > 0
    assert ex.decision_time == t + pd.Timedelta(hours=1)
    js = json.dumps(ex.to_dict())
    assert "contribution" in js


def test_explain_sizing_replay_on_many_bars_with_md(run):
    r = load_run(run["dir"])
    idx = r.forecasts.index
    for t in idx[[50, 400, 1200, 2500, len(idx) - 2]]:
        ex = explain_decision(r, t, md=run["md"])
        assert ex.sizing.get("replay_matches_saved") is True, (t, ex.sizing, ex.notes)


def test_explain_between_bars_last_bar_and_errors(run):
    r = load_run(run["dir"])
    idx = r.forecasts.index
    ex = explain_decision(r, idx[100] + pd.Timedelta(minutes=30), load_data=False)
    assert ex.time == idx[100] and any("last bar at or before" in n for n in ex.notes)
    assert "vol_ann" not in ex.sizing and "equity" in ex.sizing          # no data: saved values only
    assert explain_decision(r, None, load_data=False).time == idx[-1]
    with pytest.raises(ValueError, match="before the first OOS bar"):
        explain_decision(r, idx[0] - pd.Timedelta(days=1), load_data=False)
    with pytest.raises(ValueError, match="after the last OOS bar"):
        explain_decision(r, idx[-1] + pd.Timedelta(days=30), load_data=False)


def test_explain_holdout_phase_and_report_input(run):
    h = load_run(run["dir"], phase="holdout")
    assert set(h.folds) == {"holdout"} and h.book is not None
    c = contributions(h)
    np.testing.assert_allclose(c["raw"].clip(-1, 1), c["saved"], atol=1e-12)
    ex = explain_decision(run["dir"], h.forecasts.index[200], phase="holdout")
    assert ex.phase == "holdout" and ex.fold == "holdout"
    assert ex.sizing.get("replay_matches_saved") is True
    # the in-memory report gives the same explanation as the directory
    t = h.forecasts.index[0] - pd.Timedelta(days=10)
    a = explain_decision(run["report"], t, load_data=False)
    b = explain_decision(run["dir"], t, load_data=False)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.combined == pytest.approx(b.combined)


def test_load_run_graceful_errors(run, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_run(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="weights.csv"):
        load_run(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="no holdout"):
        load_run(tmp_path / "empty", phase="holdout")
    with pytest.raises(ValueError):
        load_run(run["dir"], phase="bogus")
    # a run without books: forecasts explainable, sizing/attribution report what is missing
    d = tmp_path / "nobooks"
    d.mkdir()
    for f in ("oos_forecasts.parquet", "weights.csv"):
        (d / f).write_bytes((run["dir"] / f).read_bytes())
    r = load_run(d)
    ex = explain_decision(r, r.forecasts.index[10], load_data=False)
    assert any("timeseries not available" in n for n in ex.notes)
    with pytest.raises(FileNotFoundError, match="save_books"):
        pnl_attribution(r)


# ---- PnL attribution --------------------------------------------------------------------------------
def test_pnl_attribution_sums_to_total_pnl(run):
    att = pnl_attribution(run["dir"])
    ts = pd.read_parquet(run["dir"] / "books" / "combined" / "timeseries.parquet")
    total = float(ts["equity"].iloc[-1] - ts["equity"].iloc[0])
    assert att.total_pnl == pytest.approx(total, abs=1e-6)
    assert att.totals.sum() == pytest.approx(total, abs=1e-6)
    np.testing.assert_allclose(att.per_bar.sum(axis=1), ts["pnl_net"].fillna(0.0), atol=1e-9)
    bc = att.by_component
    np.testing.assert_allclose(bc["price"] + bc["costs"] + bc["swap"], bc["net"], atol=1e-6)
    assert bc["net"].sum() == pytest.approx(total, abs=1e-6)
    assert att.by_year.to_numpy().sum() == pytest.approx(total, abs=1e-6)
    assert list(att.per_bar.columns) == ["ema_fast", "ema_slow", "zscore_fade", "unattributed"]
    # the first bar has no earlier decision: unattributed (and zero PnL)
    assert att.per_bar.iloc[0].abs().sum() == pytest.approx(0.0)
    assert json.dumps(att.to_dict())


def test_pnl_attribution_shares_follow_contributions(run):
    """A bar's PnL is split by the signed contributions decided one bar earlier."""
    r = load_run(run["dir"])
    att = pnl_attribution(r)
    c = contributions(r)
    b = r.book
    k = int(np.flatnonzero((c["raw"].abs() > 0.05).to_numpy())[200])
    t_dec, t_pnl = c.index[k], b.index[b.index.get_loc(c.index[k]) + 1]
    share = c.loc[t_dec, r.strategies] / c.loc[t_dec, "raw"]
    expect = share * float(b.loc[t_pnl, "pnl_net"])
    np.testing.assert_allclose(att.per_bar.loc[t_pnl, r.strategies].to_numpy(float), expect.to_numpy(float),
                               atol=1e-9)
    assert att.per_bar.loc[t_pnl, "unattributed"] == 0.0
    with pytest.raises(ValueError):
        pnl_attribution(r, component="bogus")
    price = pnl_attribution(r, component="price")
    assert price.total_pnl == pytest.approx(float(b["pnl_price"].sum()))
    assert pnl_attribution(r, phase="holdout").phase == "research"     # a RunData keeps its own phase


def test_pnl_attribution_holdout_and_plots(run):
    from matplotlib.figure import Figure

    h = pnl_attribution(run["dir"], phase="holdout")
    ts = pd.read_parquet(run["dir"] / "holdout" / "books" / "combined" / "timeseries.parquet")
    assert h.totals.sum() == pytest.approx(float(ts["equity"].iloc[-1] - ts["equity"].iloc[0]), abs=1e-6)
    assert isinstance(plot_attribution(h), Figure)
    ex = explain_decision(run["dir"], None, load_data=False)
    assert isinstance(plot_waterfall(ex), Figure)


# ---- desk journal -----------------------------------------------------------------------------------
def _cycle(cid, quant, final, *, consults=(), create=None, status="decided", action="scale", constrained=False):
    ev = [{"event": "cycle_start", "cycle_id": cid, "now": "2020-02-10T05:00:00+00:00", "quant_forecast": quant}]
    for role in consults:
        ev.append({"event": "tool_call", "cycle_id": cid, "agent": "chief", "tool": "consult_specialist",
                   "input": {"role": role, "question": "?"}})
    if create:
        ev.append({"event": "tool_call", "cycle_id": cid, "agent": "chief", "tool": "create_specialist",
                   "input": {"name": create}})
        ev.append({"event": "agent_created", "cycle_id": cid, "agent": f"adhoc:{create}", "name": create})
        ev.append({"event": "agent_end", "cycle_id": cid, "agent": f"adhoc:{create}", "status": "completed"})
    ev.append({"event": "tool_call", "cycle_id": cid, "agent": "chief", "tool": "submit_decision", "input": {}})
    ev.append({"event": "agent_end", "cycle_id": cid, "agent": "chief", "status": "completed"})
    ev.append({"event": "decision", "cycle_id": cid, "decision": {"action": action}, "failure_reason": None})
    ev.append({"event": "policy", "cycle_id": cid, "outcome": {"action": action, "constrained": constrained,
                                                               "used_fallback": False}})
    ev.append({"event": "cycle_end", "cycle_id": cid, "status": status, "final_forecast": final,
               "quant_forecast": quant, "duration_s": 0.5, "usage": {"total": {"cost_usd": 0.25}}})
    return "\n".join(json.dumps(e) for e in ev) + "\n"


def test_desk_journal_summary_counts(tmp_path):
    d = tmp_path / "journal"
    d.mkdir()
    (d / "a.jsonl").write_text(_cycle("c1", 0.6, 0.3, consults=("macro_strategist", "risk_officer"),
                                      create="event_risk_analyst"))
    (d / "b.jsonl").write_text(_cycle("c2", -0.4, 0.0, consults=("risk_officer",), action="veto",
                                      constrained=True) + "not json\n")
    s = desk_journal_summary(d)
    t = s.totals
    assert t["n_cycles"] == 2 and t["n_files"] == 2 and t["unreadable_lines"] == 1
    assert t["n_consults"] == 3 and t["consults_by_role"] == {"macro_strategist": 1, "risk_officer": 2}
    assert t["n_created_agents"] == 1 and t["created_agents"] == {"event_risk_analyst": 1}
    assert t["actions"] == {"scale": 1, "veto": 1} and t["status"] == {"decided": 2}
    assert t["tool_calls"]["submit_decision"] == 2 and t["n_constrained"] == 1
    assert t["total_cost_usd"] == pytest.approx(0.5) and t["n_direction_flips"] == 0
    assert t["median_final_over_quant"] == pytest.approx(0.25)     # median of 0.5 and 0.0
    assert list(s.cycles["cycle_id"]) == ["c1", "c2"]
    assert desk_journal_summary(d / "a.jsonl").totals["n_cycles"] == 1
    assert json.dumps(s.to_dict())
    with pytest.raises(FileNotFoundError):
        desk_journal_summary(tmp_path / "none")
    (tmp_path / "emptydir").mkdir()
    with pytest.raises(FileNotFoundError):
        desk_journal_summary(tmp_path / "emptydir")


def test_desk_journal_summary_partial_cycle(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"event": "cycle_start", "cycle_id": "z", "quant_forecast": 0.2}) + "\n")
    t = desk_journal_summary(p).totals
    assert t["status"] == {"unfinished": 1} and math.isnan(t["total_cost_usd"])
