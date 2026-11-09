"""v2.1 integration seams: live decision attribution -> analytics (``explain --live``), the
desk replay's holdout look, the live runner's desk check from ``aurum live run``, lazy
package exports and the macro requirement in ``aurum features list`` (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from test_live_support import build_artifact, synthetic_bars

from aurum.cli import EXIT_OK, EXIT_USAGE, main
from aurum.execution.costs import CostModel
from aurum.live.broker import SimulatedClock
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.runner import LiveConfig, LiveDeskDataProvider, LiveRunner, load_artifact
from aurum.research.attribution import DecisionExplanation, explain_live_decision, live_decisions

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AURUM_ALERT_WEBHOOK_URL", "ANTHROPIC_API_KEY", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
                "AURUM_ARTIFACT_KEY"):
        monkeypatch.delenv(var, raising=False)


def run(capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def _fake_desk(scale: float, n: int) -> Any:
    from aurum.agents import DecisionPolicy, TradingDesk
    from aurum.agents.testing import FakeAnthropicClient, decision_call, message

    client = FakeAnthropicClient({"chief": [message(decision_call(action="scale", scale=scale, confidence=0.9))
                                            for _ in range(n)]})
    return TradingDesk(LiveDeskDataProvider(), client=client, policy=DecisionPolicy(mode="overlay"), journal_dir=None)


@pytest.fixture(scope="module")
def live_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A paper runner's state dir: 10 decisions, the LLM desk (scripted) halving the forecast."""
    root = tmp_path_factory.mktemp("v21_integ_live")
    bars = synthetic_bars(700, weekend_gaps=False)
    art = load_artifact(build_artifact(root / "art", bars, n_train=350))
    clock = SimulatedClock(bars["available_at"].iloc[600])
    broker = PaperBroker(ReplayFeed(bars), costs=CostModel(commission_per_lot=3.5), clock=clock)
    cfg = LiveConfig(dry_run=False, state_dir=str(root / "state"), calendar=None,
                     risk={"daily_loss_persistent": False, "max_daily_loss": 0.05})
    runner = LiveRunner(cfg, broker=broker, artifact=art, clock=clock, desk=_fake_desk(0.5, 10))
    runner.run(max_cycles=10, install_signal_handlers=False)
    return root / "state"


# ---- live decision attribution -> analytics ---------------------------------------------------
def test_live_decisions_table_reproduces_the_runner_arithmetic(live_log: Path) -> None:
    df = live_decisions(live_log)
    assert len(df) == 10
    assert df["exact"].all() and df["method"].notna().all() and df["fdm"].notna().all()
    contrib = df[[c for c in df.columns if c.startswith("c:")]]
    assert contrib.shape[1] >= 2
    # sum_i w_i f_i FDM = raw; clip(raw) = combined (the JSONL keeps 10 significant digits)
    assert np.allclose(contrib.sum(axis=1), df["raw_combined"], atol=1e-8)
    assert np.allclose(df["raw_combined"].clip(-1, 1), df["combined"], atol=1e-8)
    assert np.allclose(df["final_forecast"] - df["combined"], df["desk_delta"], atol=1e-8)
    moved = df["desk_delta"].abs() > 1e-12
    assert moved.any() and np.allclose(df.loc[moved, "desk_delta"], -0.5 * df.loc[moved, "combined"], atol=1e-8)
    # the same reader accepts the JSONL path and parsed records
    assert len(live_decisions(live_log / "decisions.jsonl")) == 10
    recs = [json.loads(x) for x in (live_log / "decisions.jsonl").read_text().splitlines() if x.strip()]
    assert len(live_decisions(recs)) == 10


def test_explain_live_decision(live_log: Path) -> None:
    ex = explain_live_decision(live_log)
    assert isinstance(ex, DecisionExplanation) and ex.phase == "live"
    assert ex.table["contribution"].sum() == pytest.approx(ex.raw_combined, abs=1e-8)
    assert ex.combined == pytest.approx(float(np.clip(ex.raw_combined, -1, 1)), abs=1e-8)
    assert ex.final_forecast is not None
    assert ex.desk_delta == pytest.approx(ex.final_forecast - ex.combined)
    assert ex.sizing["replayed_lots"] == pytest.approx(ex.sizing["saved_requested_lots"])
    assert "approved_lots" in ex.sizing and np.isfinite(ex.sizing["target_vol"])
    d = ex.to_dict()
    json.dumps(d)
    assert d["phase"] == "live" and "desk_delta" in d
    fig = ex.plot()
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    if abs(ex.desk_delta) > 1e-12:
        assert any(lbl.startswith("LLM desk") for lbl in labels) and "final\nforecast" in labels
    # --at: the last decision at or before it; before the first one -> error
    df = live_decisions(live_log)
    t3 = df["bar_time"].iloc[3]
    ex3 = explain_live_decision(live_log, t3 + pd.Timedelta(minutes=30))
    assert ex3.time == t3 and ex3.notes
    with pytest.raises(ValueError, match="before the first"):
        explain_live_decision(live_log, df["bar_time"].iloc[0] - pd.Timedelta(days=30))


def test_explain_live_without_attribution_is_a_clear_error(tmp_path: Path) -> None:
    recs = [{"type": "start"}, {"type": "decision", "time": "2024-01-02T10:00:00Z",
                                "bar_time": "2024-01-02T09:00:00Z", "combined": 0.2, "attribution": None}]
    with pytest.raises(ValueError, match="no live decision with an attribution"):
        explain_live_decision(recs)
    with pytest.raises(FileNotFoundError):
        live_decisions(tmp_path / "nope")
    with pytest.raises(ValueError, match="no decision records"):
        live_decisions([{"type": "start"}])


def test_explain_live_opaque_combiner_record() -> None:
    rec = {"type": "decision", "time": "2024-01-02T10:00:00Z", "bar_time": "2024-01-02T09:00:00Z",
           "combined": 0.3, "final_forecast": 0.3, "status": "ok",
           "attribution": {"method": "Opaque", "fdm": None, "raw_combined": None, "combined": 0.3,
                           "clipped": False, "unallocated": None, "exact": False, "final_forecast": 0.3,
                           "desk_delta": 0.0,
                           "strategies": {"a": {"forecast": 0.5, "weight": None, "contribution": None}}}}
    ex = explain_live_decision([rec])
    assert ex.raw_combined == pytest.approx(0.3) and any("not exact" in n for n in ex.notes)
    ex.plot()


def test_cli_explain_live(capsys: pytest.CaptureFixture, live_log: Path, tmp_path: Path) -> None:
    out = tmp_path / "ex"
    code, stdout, err = run(capsys, "explain", "--live", str(live_log), "--out", str(out))
    assert code == EXIT_OK, err
    assert "explain (live log)" in stdout and "live log summary: 10 decisions" in stdout
    assert "LLM desk changed the forecast" in stdout
    assert (out / "live_decisions.csv").exists()
    assert len(list(out.glob("decision_live_*.png"))) == 1 and len(list(out.glob("decision_live_*.json"))) == 1
    # read-only: nothing new in the runner's state directory
    assert not (live_log / "analysis").exists()
    code, _, err = run(capsys, "explain", "--live", str(live_log), "--run", str(tmp_path))
    assert code == EXIT_USAGE and "not both" in err
    code, _, err = run(capsys, "explain")
    assert code == EXIT_USAGE and "--live" in err


# ---- desk replay -> holdout ledger ------------------------------------------------------------
@pytest.fixture(scope="module")
def replay_cfg(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("v21_integ_replay")
    cfg = {
        "name": "integ_replay", "seed": 3,
        "data": {"synthetic": {"n": 2400, "model": "trend", "seed": 5, "start": "2020-01-06"}, "timeframe": "H1",
                 "events": "none", "macro": False},
        "strategies": [{"name": "ema_cross", "params": {"fast": 8, "slow": 32}},
                       {"name": "zscore_fade"}],
        "walkforward": {"train": "2M", "test": "1M", "holdout_start": "2020-04-15", "n_boot": 200,
                        "executor": "serial", "min_train_bars": 200},
        "risk": {"research": {"max_spread": None}},
        "output": {"dir": str(root / "runs"), "dark_charts": False, "tearsheet": False},
        "agents": {"replay_every": 24, "replay_max_cost_usd": 5.0},
    }
    path = root / "replay.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return {"root": root, "config": str(path)}


def _replay(capsys: pytest.CaptureFixture, cfg: str, start: str, end: str, out: Path) -> tuple[int, str, str]:
    return run(capsys, "desk", "replay", "--config", cfg, "--start", start, "--end", end, "--fake", "--yes",
               "--every", "24", "--out", str(out), "--journal-dir", str(out.parent / "journal"))


def test_desk_replay_into_the_holdout_records_a_look(capsys: pytest.CaptureFixture,
                                                     replay_cfg: dict[str, Any]) -> None:
    from aurum.core.config import load_config
    from aurum.research.walkforward import HOLDOUT_LEDGER, prior_holdout_looks, read_holdout_ledger

    out = replay_cfg["root"] / "into" / "replay"
    code, stdout, err = _replay(capsys, replay_cfg["config"], "2020-04-06", "2020-04-24", out)
    assert code == EXIT_OK, err
    assert "spends the holdout" in stdout and "holdout look recorded in" in stdout
    ledger = out.parent / HOLDOUT_LEDGER
    entries = read_holdout_ledger(ledger)
    assert len(entries) == 1
    e = entries[0]
    cfg = load_config(replay_cfg["config"])
    assert e["kind"] == "desk_replay" and e["fake_client"] is True
    assert e["config_hash"] == f"{cfg.config_hash()}:desk-fake"
    assert pd.Timestamp(e["holdout_start"]) == pd.Timestamp("2020-04-15", tz="UTC")
    assert pd.Timestamp(e["evaluated_start"]) < pd.Timestamp(e["holdout_start"]) <= pd.Timestamp(e["holdout_end"])
    assert e["data_source"] == "synthetic" and e["data_hash"]
    assert set(e["metrics"]) == {"desk_sharpe", "quant_only_sharpe"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["holdout_look"] == str(ledger)
    # a later walk-forward of the quant config counts the replay as ANOTHER configuration
    looks = prior_holdout_looks(entries, start="2020-04-15", end="2020-06-30", config_hash=cfg.config_hash(),
                                symbol=cfg.data.symbol, data_source="synthetic")
    assert looks["n_prior_looks"] == 1 and looks["n_prior_configs"] == 1


def test_desk_replay_before_the_holdout_records_nothing(capsys: pytest.CaptureFixture,
                                                        replay_cfg: dict[str, Any]) -> None:
    out = replay_cfg["root"] / "before" / "replay"
    code, stdout, err = _replay(capsys, replay_cfg["config"], "2020-03-20", "2020-04-10", out)
    assert code == EXIT_OK, err
    assert "holdout look recorded" not in stdout
    assert not (out.parent / "holdout_ledger.jsonl").exists()
    assert json.loads((out / "summary.json").read_text())["holdout_look"] is None


# ---- aurum live run: the runner's own desk check before anything is written ------------------
def test_live_run_runs_the_runner_desk_check_first(capsys: pytest.CaptureFixture, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    fake = types.ModuleType("aurum.live.runner")

    class FakeLiveConfig:
        @classmethod
        def from_mapping(cls, mapping: dict) -> FakeLiveConfig:
            return cls()

    def check(cfg: Any) -> None:
        calls["checked"] = True
        raise ValueError("the LLM desk is enabled but ANTHROPIC_API_KEY is not set")

    fake.LiveConfig, fake.check_desk_prerequisites = FakeLiveConfig, check
    fake.main = lambda argv: calls.setdefault("ran", True)
    monkeypatch.setitem(sys.modules, "aurum.live.runner", fake)
    state = tmp_path / "state"
    code, _, err = run(capsys, "live", "run", "--config", str(REPO / "configs" / "live_paper.yaml"),
                       "--set", f"live.state_dir={state}")
    assert code == EXIT_USAGE and "ANTHROPIC_API_KEY" in err
    assert calls == {"checked": True} and not (state / "aurum_live_config.yaml").exists()


# ---- lazy exports / registry listing -----------------------------------------------------------
def test_new_apis_are_exported_lazily() -> None:
    import aurum.research as research
    import aurum.rl as rl

    for name in ("run_sweep", "plateau_score", "monte_carlo", "stress_run", "explain_decision",
                 "explain_live_decision", "live_decisions", "pnl_attribution", "desk_journal_summary",
                 "record_holdout_look", "forecast_contributions"):
        assert name in research.__all__ and callable(getattr(research, name))
    has_rl = all(importlib.util.find_spec(m) is not None for m in ("gymnasium", "stable_baselines3", "torch"))
    for name in ("rl_config_from_aurum", "artifact_data_ranges"):
        assert name in rl.__all__                       # exported lazily: resolving it needs aurum[rl]
        if has_rl:
            assert callable(getattr(rl, name))


def test_features_list_shows_the_macro_requirement(capsys: pytest.CaptureFixture) -> None:
    code, out, _ = run(capsys, "features", "list", "--json")
    assert code == EXIT_OK
    rows = json.loads(out)
    assert rows["macro"]["requires"] == "macro(any)" and rows["calendar"]["requires"] == "events"


# ---- review fixes ------------------------------------------------------------------------------
def test_sweep_heatmap_labels_follow_the_rendered_cell_colour() -> None:
    """On a sequential scale (all values negative) the darkest cell is the BEST one, not the
    one with the largest |value|: label colours come from the rendered colour."""
    from aurum.research.sweep import plot_sweep

    grid = {"fast": [8, 16], "slow": [32, 64]}
    table = pd.DataFrame({"fast": [8, 8, 16, 16], "slow": [32, 64, 32, 64], "status": "ok",
                          "sharpe": [-0.16, -0.55, -0.70, -0.84]})
    fig = plot_sweep(table, grid, "sharpe", best={"fast": 8, "slow": 32})
    colours = {t.get_text(): t.get_color() for t in fig.axes[0].texts}
    assert colours["-0.16"] == "#ffffff"          # darkest blue: white label
    assert colours["-0.84"] != "#ffffff"          # lightest: ink label
    # diverging scale (signed values): the extremes are dark, the middle light
    table["sharpe"] = [0.9, 0.05, -0.9, -0.1]
    fig = plot_sweep(table, grid, "sharpe")
    colours = {t.get_text(): t.get_color() for t in fig.axes[0].texts}
    assert colours["0.90"] == "#ffffff" and colours["-0.90"] == "#ffffff" and colours["0.05"] != "#ffffff"
