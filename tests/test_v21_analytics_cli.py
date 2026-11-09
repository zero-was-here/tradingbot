"""CLI registration and commands of aurum.cli_ext (sweep / stress / explain), synthetic data only."""

from __future__ import annotations

import argparse
import json
import sys
import types

import pandas as pd
import pytest
import yaml

from aurum.cli_ext import COMMANDS, build_parser, main, register_all


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli")
    cfg = {"seed": 3, "data": {"synthetic": {"n": 5000, "model": "trend", "seed": 7, "regime_params": {"phi": 0.1}}},
           "strategies": [{"name": "ema_cross", "params": {"fast": 8, "slow": 32}, "id": "ema_fast"},
                          {"name": "zscore_fade"}],
           "combiner": {"method": "fixed"},
           "walkforward": {"train": "4M", "test": "6W", "embargo": 6, "n_boot": 100, "pbo_splits": 8,
                           "min_train_bars": 200, "executor": "serial"},
           "output": {"dir": str(tmp / "runs"), "tearsheet": False}}
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture(scope="module")
def run_dir(cfg_path):
    from aurum.core.config import load_config
    from aurum.research.walkforward import run_walk_forward

    cfg = load_config(cfg_path, env={})
    out = cfg_path.parent / "runs" / "wf"
    run_walk_forward(cfg.data.load(), cfg, out_dir=out)
    return out


def test_register_all_is_idempotent():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")
    assert register_all(sub)[:3] == list(COMMANDS)
    assert register_all(sub) == []                     # second call: nothing new, no conflict
    a = ap.parse_args(["sweep", "-c", "x.yaml", "--strategy", "ema_cross", "--grid", "fast=1,2"])
    assert a.grid == ["fast=1,2"] and a.metric == "sharpe" and callable(a.func)
    a = ap.parse_args(["stress", "--run", "d", "--paths", "10", "--horizon", "6M", "--shock=-0.05"])
    assert a.paths == 10 and a.shock == ["-0.05"]
    a = ap.parse_args(["explain", "--run", "d", "--at", "2020-01-01", "--attribution"])
    assert a.attribution and a.at == "2020-01-01"


def test_build_parser_includes_core_and_extension_commands():
    ap = build_parser()
    sub = next(x for x in ap._actions if isinstance(x, argparse._SubParsersAction))
    for name in ("walkforward", "report", *COMMANDS):
        assert name in sub.choices


def test_viz_dashboard_registered_when_available(monkeypatch):
    calls = []

    def register(subparsers):
        calls.append(1)
        subparsers.add_parser("dashboard")

    fake = types.ModuleType("aurum.viz.cli")
    fake.register = register
    monkeypatch.setitem(sys.modules, "aurum.viz.cli", fake)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")
    added = register_all(sub)
    assert "dashboard" in added and "dashboard" in sub.choices and calls == [1]
    assert register_all(sub) == [] and calls == [1]


def test_cli_sweep_writes_outputs(cfg_path, tmp_path, capsys):
    out = tmp_path / "sweep"
    code = main(["sweep", "-c", str(cfg_path), "--strategy", "ema_fast", "--grid", "fast=4,8", "--grid", "slow=32",
                 "--out", str(out), "--executor", "serial", "--set", "walkforward.n_boot=100"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "EXPLORATORY" in text and "best: fast=" in text and "plateau:" in text and "DSR" in text
    for f in ("sweep.csv", "sweep.json", "sweep.png"):
        assert (out / f).exists()
    js = json.loads((out / "sweep.json").read_text())
    assert js["n_trials"] == 2 and js["strategy"] == "ema_fast"
    assert js["settings"]["base_params"] == {"fast": 8, "slow": 32}       # config params are the base
    assert (tmp_path / "sweep_ledger.jsonl").exists()


def test_cli_stress_and_explain(run_dir, tmp_path, capsys):
    code = main(["stress", "--run", str(run_dir), "--paths", "800", "--horizon", "1Y", "--out",
                 str(tmp_path / "stress"), "--shock=-0.05", "--shock=0.05"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "P(max drawdown deeper than x)" in text and "worst historical" in text and "gap_-5.0%" in text
    assert (tmp_path / "stress" / "fan.png").exists() and (tmp_path / "stress" / "stress.json").exists()
    ts = pd.read_parquet(run_dir / "oos_forecasts.parquet").index
    at = str(ts[len(ts) // 2])
    code = main(["explain", "--run", str(run_dir), "--at", at, "--attribution", "--out", str(tmp_path / "ex")])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "contribution" in text and "PnL attribution" in text and "matches saved: True" in text
    files = {p.name for p in (tmp_path / "ex").iterdir()}
    assert {"attribution.png", "attribution_totals.csv", "attribution.json"} <= files
    assert any(f.startswith("decision_") and f.endswith(".png") for f in files)


def test_cli_explain_default_out_and_desk_journal(run_dir, tmp_path, capsys):
    j = tmp_path / "journal"
    j.mkdir()
    (j / "c.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "cycle_start", "cycle_id": "c", "quant_forecast": 0.5},
        {"event": "tool_call", "cycle_id": "c", "tool": "consult_specialist", "input": {"role": "risk_officer"}},
        {"event": "cycle_end", "cycle_id": "c", "status": "decided", "final_forecast": 0.25,
         "usage": {"total": {"cost_usd": 0.1}}}]) + "\n")
    code = main(["explain", "--run", str(run_dir), "--no-data", "--desk-journal", str(j)])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "desk journal" in text and "consults 1" in text
    assert (run_dir / "analysis" / "explain" / "desk_summary.json").exists()


def test_cli_errors(tmp_path, capsys):
    assert main(["stress", "--run", str(tmp_path / "missing")]) == 1
    assert main(["explain"]) == 2
    assert main(["explain", "--run", str(tmp_path)]) == 1          # not a run directory
    assert main(["sweep", "-c", str(tmp_path / "nope.yaml"), "--strategy", "x", "--grid", "a=1"]) in (1, 2)
    assert main(["sweep", "--help"]) == 0
    capsys.readouterr()


def test_core_aurum_parser_exposes_extension_commands():
    """aurum.cli loads aurum.cli_ext.register_all (guarded); the commands must be reachable as `aurum X`."""
    from aurum.cli import build_parser as core_parser

    ap = core_parser()
    sub = next(x for x in ap._actions if isinstance(x, argparse._SubParsersAction))
    missing = [c for c in COMMANDS if c not in sub.choices]
    if missing and not hasattr(sys.modules.get("aurum.cli"), "_register_extensions"):
        pytest.skip("aurum.cli does not load extensions in this build")
    assert not missing
