"""v2.1 live operator commands: `aurum live status` (read-only) and `aurum live reset-halt`
(the only CLI path that clears the kill switch: refuses while a runner holds the state
directory, needs the literal --confirm RESET, logs who/when/why)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aurum.cli import EXIT_CONFIRM, EXIT_OK, EXIT_USAGE, main
from aurum.risk.manager import RiskLimits, StandardRiskManager

REPO = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2024-03-04 10:00", tz="UTC")


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture()
def live(tmp_path):
    state = tmp_path / "state"
    cfg = tmp_path / "live.yaml"
    cfg.write_text(yaml.safe_dump({"extends": str(REPO / "configs" / "live_paper.yaml"),
                                   "live": {"state_dir": str(state), "artifact_dir": str(tmp_path / "art")}}))
    return {"state": state, "config": str(cfg)}


def _halted_state(state: Path, *, equity: float = 99_000.0) -> Path:
    """A risk state file as the runner leaves it after a kill (here: an operator halt recorded
    as kind "drawdown"; the 1% drop stays below every automatic limit)."""
    state.mkdir(parents=True, exist_ok=True)
    path = state / "risk_state.json"
    rm = StandardRiskManager(RiskLimits(), state_path=path)
    rm.on_bar(T0, 100_000.0)
    rm.on_bar(T0 + pd.Timedelta(hours=1), equity)
    rm.halt("max drawdown breached (test)", time=T0 + pd.Timedelta(hours=1), kind="drawdown")
    assert StandardRiskManager(RiskLimits(), state_path=path).halted
    return path


def _snapshot(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def _records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ---- reset-halt ---------------------------------------------------------------------------------------
def test_reset_halt_shows_state_and_needs_the_literal_confirmation(capsys, live):
    path = _halted_state(live["state"])
    before = path.read_bytes()
    code, out, _ = run(capsys, "live", "reset-halt", "-c", live["config"])
    assert code == EXIT_CONFIRM
    assert "halted:      True [drawdown]" in out and "max drawdown breached (test)" in out
    assert "--confirm RESET" in out and path.read_bytes() == before
    for wrong in ("reset", "yes", "RESET "):
        assert run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", wrong)[0] == EXIT_CONFIRM
    assert path.read_bytes() == before
    assert not (live["state"] / "decisions.jsonl").exists()


def test_reset_halt_refuses_while_a_runner_holds_the_lock(capsys, live):
    runner = pytest.importorskip("aurum.live.runner")
    path = _halted_state(live["state"])
    before = path.read_bytes()
    fh = runner._lock_state_dir(live["state"] / "runner.lock")   # noqa: SLF001 - what a running runner holds
    try:
        code, out, err = run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", "RESET",
                             "--reason", "should not happen")
        assert code == EXIT_USAGE and "refusing to reset" in err and "stop it first" in err
        assert "halted:      True" in out                       # the state is still shown first
        code, out, _ = run(capsys, "live", "status", "-c", live["config"])
        assert code == EXIT_OK and "RUNNING (lock held)" in out
    finally:
        runner._unlock_state_dir(fh)   # noqa: SLF001
    assert path.read_bytes() == before
    assert not (live["state"] / "decisions.jsonl").exists()


def test_reset_halt_clears_the_kill_switch_and_logs_who_when_why(capsys, live):
    pytest.importorskip("aurum.live.runner")
    path = _halted_state(live["state"], equity=99_000.0)
    code, out, err = run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", "RESET",
                         "--reason", "post-mortem done, limits reviewed")
    assert code == EXIT_OK, err
    assert "kill switch RESET by" in out and "99,000" in out
    rm = StandardRiskManager(RiskLimits(), state_path=path)
    assert not rm.halted and rm.state.peak_equity == 99_000.0 and rm.state.day_start_equity == 99_000.0
    dec = [r for r in _records(live["state"] / "decisions.jsonl") if r.get("type") == "halt_reset"]
    assert len(dec) == 1
    rec = dec[0]
    assert rec["reason"] == "post-mortem done, limits reviewed" and rec["user"] and rec["host"]
    assert rec["previous"]["halt_reason"] == "max drawdown breached (test)" and rec["previous"]["halt_kind"] == "drawdown"
    assert pd.Timestamp(rec["time"]).tzinfo is not None and rec["config"] == live["config"]
    alerts = _records(live["state"] / "alerts.jsonl")
    assert alerts[-1]["level"] == "critical" and alerts[-1]["kind"] == "halt_reset"
    assert "post-mortem done" in alerts[-1]["message"]
    # the lock was released: a runner (or a second reset) can take it again
    code, out, _ = run(capsys, "live", "status", "-c", live["config"])
    assert "not running" in out and "risk:        active" in out and "last reset:" in out
    # a second reset is a no-op that logs nothing
    code, out, _ = run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", "RESET")
    assert code == EXIT_OK and "nothing to reset" in out
    assert len([r for r in _records(live["state"] / "decisions.jsonl") if r.get("type") == "halt_reset"]) == 1


def test_reset_halt_refusals_without_state(capsys, live):
    code, _, err = run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", "RESET")
    assert code == EXIT_USAGE and "does not exist" in err
    live["state"].mkdir()
    (live["state"] / "runner_state.json").write_text("{}")
    code, _, err = run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", "RESET")
    assert code == EXIT_USAGE and "no risk_state.json" in err and "start it once" in err
    assert not (live["state"] / "risk_state.json").exists()     # deleting the file is never a reset


def test_reset_halt_on_an_unreadable_state_file(capsys, live):
    pytest.importorskip("aurum.live.runner")
    live["state"].mkdir()
    path = live["state"] / "risk_state.json"
    path.write_text('{"halted": "false"')                        # torn / hand-edited: the runner starts HALTED
    code, out, _ = run(capsys, "live", "reset-halt", "-c", live["config"])
    assert code == EXIT_CONFIRM and "unreadable" in out and "halted:      True [state_file]" in out
    code, out, err = run(capsys, "live", "reset-halt", "-c", live["config"], "--confirm", "RESET",
                         "--reason", "file repaired by hand")
    assert code == EXIT_OK, err
    assert not StandardRiskManager(RiskLimits(), state_path=path).halted
    assert json.loads(path.read_text())["halted"] is False


# ---- status -------------------------------------------------------------------------------------------
def test_status_without_state_dir(capsys, live):
    code, out, _ = run(capsys, "live", "status", "-c", live["config"])
    assert code == EXIT_OK and "does not exist" in out and "broker=paper dry_run=True" in out
    assert not live["state"].exists()


def test_status_is_read_only_and_summarises_everything(capsys, live):
    state = live["state"]
    _halted_state(state)
    (state / "heartbeat.json").write_text(json.dumps({"time": T0.isoformat(), "status": "sleeping",
                                                      "mode": "DRY-RUN PAPER", "last_bar": "2024-03-04T09:00:00+00:00",
                                                      "n_decisions": 12, "pending": None, "halted": True}))
    (state / "runner_state.json").write_text(json.dumps({"last_bar": "2024-03-04T09:00:00+00:00", "n_decisions": 12,
                                                         "prev_final_forecast": 0.25}))
    (state / "oms_state.json").write_text(json.dumps({
        "format": "aurum.live.oms", "version": 1, "symbol": "XAUUSD", "magic": 20260926,
        "decisions": {"2024-03-04T09:00:00+00:00": {"status": "complete", "target": 0.5},
                      "2024-03-04T10:00:00+00:00": {"status": "deferred", "target": 0.0}}}))
    (state / "paper_broker.json").write_text(json.dumps({"balance": 99_000.0, "n_orders": 4,
                                                         "positions": [{"lots": 0.3}, {"lots": 0.2}]}))
    (state / "decisions.jsonl").write_text(
        json.dumps({"type": "start"}) + "\n"
        + json.dumps({"type": "decision", "time": T0.isoformat(), "status": "dry_run", "combined": 0.4,
                      "final_forecast": 0.4, "approved_lots": 0.0, "current_lots": 0.5, "equity": 99_000.0}) + "\n"
        + '{"torn line\n')
    (state / "alerts.jsonl").write_text(json.dumps({"time": T0.isoformat(), "level": "critical",
                                                    "kind": "risk_halt", "message": "kill switch"}) + "\n")
    before = _snapshot(state)
    code, out, err = run(capsys, "live", "status", "-c", live["config"])
    assert code == EXIT_OK, err
    for needle in ("not running", "HALTED [drawdown] max drawdown breached (test)", "reset-halt",
                   "2 decisions, last 2024-03-04T10:00:00+00:00 -> deferred", "not terminal",
                   "net 0.5 lots in 2 position(s)", "status=dry_run", "kill switch", "sleeping", "12 decisions"):
        assert needle in out, needle
    code, js, _ = run(capsys, "live", "status", "-c", live["config"], "--json")
    s = json.loads(js)
    assert s["risk"]["halted"] is True and s["risk"]["drawdown"] == pytest.approx(0.01)
    assert s["oms"]["n_decisions"] == 2 and s["paper_broker"]["net_lots"] == pytest.approx(0.5)
    assert s["runner_lock"]["exists"] is False and s["last_decision"]["equity"] == 99_000.0
    assert _snapshot(state) == before                         # nothing written, no lock file created


def test_status_reports_unreadable_state_files(capsys, live):
    live["state"].mkdir()
    (live["state"] / "risk_state.json").write_text("{nope")
    (live["state"] / "oms_state.json").write_text("[1,")
    code, out, _ = run(capsys, "live", "status", "-c", live["config"])
    assert code == EXIT_OK
    assert "risk:        unreadable" in out and "starts HALTED" in out
    assert "oms:         unreadable" in out and "refuses to trade" in out


def test_status_works_even_when_the_config_has_semantic_problems(capsys, live, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"extends": live["config"], "strategies": [{"name": "not_registered"}]}))
    assert run(capsys, "config", "validate", "-c", str(bad))[0] == EXIT_USAGE
    code, out, _ = run(capsys, "live", "status", "-c", str(bad))
    assert code == EXIT_OK and "live status:" in out
    assert sys.modules.get("aurum.cli") is not None
