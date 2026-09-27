"""CLI smoke tests: ``aurum.cli.main(argv)`` on synthetic data written to a temporary data
store (no network, no API key). Stub strategies are registered for the duration of the
module and removed afterwards so other tests see an untouched registry."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import aurum.strategies.base as sbase
from aurum.cli import EXIT_CONFIRM, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE, build_parser, main
from aurum.data.macro import save_macro_dir
from aurum.data.store import save_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro
from aurum.strategies.base import Strategy

REPO = Path(__file__).resolve().parents[1]


class CliMom(Strategy):
    name = "cli_stub_mom"
    description = "test stub: short EMA momentum"

    @property
    def warmup_bars(self) -> int:
        return 50

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        s = r.ewm(span=6, adjust=False).mean() / r.ewm(span=48, adjust=False, min_periods=24).std()
        return self._finalize((1.5 * s).clip(-1, 1), md.bars.index)


class CliSlow(Strategy):
    name = "cli_stub_slow"
    description = "test stub: trainable sign learner"
    trainable = True

    def fit(self, md, features=None):
        r = np.log(md.bars["close"]).diff().dropna()
        self.sign_ = float(np.sign(r.autocorr(1)) or 1.0)
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        return self._finalize(self.sign_ * np.tanh(r.rolling(6).sum() / r.rolling(96).std()), md.bars.index)


@pytest.fixture(scope="module", autouse=True)
def _stub_registry():
    added = []
    for cls in (CliMom, CliSlow):
        if cls.name not in sbase._STRATEGIES:
            sbase.register_strategy(cls)
            added.append(cls.name)
    yield
    for name in added:
        sbase._STRATEGIES.pop(name, None)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    data = root / "data"
    bars = make_synthetic_bars(5000, "H1", seed=9, model="trend", regime_params={"phi": 0.12})
    save_bars(bars, data / "xauusd_H1.parquet")
    save_macro_dir(make_synthetic_macro(bars, seed=9), data / "macro")
    cfg = {
        "name": "cli_test", "seed": 1,
        "data": {"dir": str(data), "timeframe": "H1", "events": "rule_based"},
        "strategies": [{"name": "cli_stub_mom"}, {"name": "cli_stub_slow"}],
        "walkforward": {"train": "3M", "test": "1M", "embargo": 4, "executor": "serial", "n_boot": 200,
                        "pbo_splits": 4, "min_train_bars": 200},
        "risk": {"research": {"max_spread": None}},
        "output": {"dir": str(root / "runs"), "dark_charts": False},
        "agents": {"replay_every": 12, "replay_max_cost_usd": 5.0, "desk": {"max_cost_usd_per_cycle": 1.0}},
    }
    path = root / "cli.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return {"root": root, "data": data, "config": str(path), "bars": bars}


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# ---- parser / help --------------------------------------------------------------------------------
def test_help_and_usage(capsys):
    assert run(capsys, "--help")[0] == EXIT_OK
    assert run(capsys)[0] == EXIT_USAGE                          # a command is required
    assert run(capsys, "desk", "replay", "--config", "x")[0] == EXIT_USAGE  # --start/--end required
    assert run(capsys, "data", "download", "--help")[0] == EXIT_OK
    cmds = build_parser()._subparsers._group_actions[0].choices  # noqa: SLF001
    assert {"data", "config", "backtest", "walkforward", "report", "strategies", "features", "desk", "rl",
            "live"} <= set(cmds)


def test_python_dash_m_entry_point():
    out = subprocess.run([sys.executable, "-m", "aurum", "--help"], capture_output=True, text=True, timeout=120,
                         cwd=REPO)
    assert out.returncode == 0 and "walkforward" in out.stdout


# ---- config / registries / data -------------------------------------------------------------------
def test_config_validate_show_and_errors(capsys, env, tmp_path):
    code, out, _ = run(capsys, "config", "validate", "--config", str(REPO / "configs" / "default.yaml"))
    assert code == EXIT_OK and "config OK" in out
    code, out, _ = run(capsys, "config", "show", "--config", env["config"], "--set", "seed=3")
    assert code == EXIT_OK and "seed: 3" in out and "cli_stub_mom" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("walkforwrd: {}\n")
    code, _, err = run(capsys, "config", "validate", "--config", str(bad))
    assert code == EXIT_USAGE and "did you mean 'walkforward'" in err


def test_strategies_and_features_list(capsys):
    code, out, _ = run(capsys, "strategies", "list")
    assert code == EXIT_OK and "cli_stub_mom" in out
    code, out, _ = run(capsys, "strategies", "list", "--json")
    rows = json.loads(out)
    assert rows["cli_stub_slow"]["trainable"] is True
    code, out, _ = run(capsys, "features", "list")
    assert code == EXIT_OK and "returns" in out and "volatility" in out
    assert "returns" in json.loads(run(capsys, "features", "list", "--json")[1])


def test_data_info(capsys, env):
    code, out, _ = run(capsys, "data", "info", "--dir", str(env["data"]))
    assert code == EXIT_OK
    assert "xauusd_H1.parquet" in out and "5000" in out and "ok" in out
    assert "macro" in out and "dxy" in out
    assert run(capsys, "data", "info", "--dir", str(env["root"] / "missing"))[0] == EXIT_USAGE


# ---- research commands -----------------------------------------------------------------------------
def test_backtest_command(capsys, env):
    out_dir = env["root"] / "bt"
    start = str(env["bars"].index[3000].date())
    code, out, err = run(capsys, "backtest", "--config", env["config"], "--start", start,
                         "--strategy", "cli_stub_mom", "--out", str(out_dir), "--no-tearsheet")
    assert code == EXIT_OK, err
    assert "combined" in out and "cli_stub_mom" in out and "cli_stub_slow" not in out
    s = json.loads((out_dir / "summary.json").read_text())
    assert s["kind"] == "backtest" and s["oos_start"].startswith(start)
    assert not (out_dir / "tearsheet.html").exists()
    code, out, _ = run(capsys, "backtest", "--config", env["config"], "--no-write")
    assert code == EXIT_OK and "IN-SAMPLE" in out


def test_walkforward_then_report(capsys, env):
    out_dir = env["root"] / "wf"
    code, out, err = run(capsys, "walkforward", "--config", env["config"], "--out", str(out_dir),
                         "--set", "walkforward.holdout_start=" + str(env["bars"].index[-600].date()))
    assert code == EXIT_OK, err
    assert "PBO" in out or "cli_stub_slow" in out
    assert "FINAL HOLDOUT" in out and "report dir" in out
    assert (out_dir / "tearsheet.html").exists() and (out_dir / "holdout" / "tearsheet.html").exists()
    code, out2, _ = run(capsys, "report", "--run", str(out_dir))
    assert code == EXIT_OK and "combined" in out2 and "FINAL HOLDOUT" in out2
    code, js, _ = run(capsys, "report", "--run", str(out_dir), "--json")
    s = json.loads(js)
    assert s["kind"] == "walkforward" and s["n_folds"] >= 3 and s["holdout"] is not None
    assert run(capsys, "report", "--run", str(env["root"] / "nope"))[0] == EXIT_USAGE


def test_config_error_exit_code(capsys, env):
    code, _, err = run(capsys, "walkforward", "--config", env["config"], "--set", "walkforward.train=3 parsecs")
    assert code == EXIT_USAGE and "walkforward.train" in err
    code, _, err = run(capsys, "backtest", "--config", env["config"], "--strategy", "not_configured")
    assert code == EXIT_USAGE and "not in config" in err


# ---- desk -------------------------------------------------------------------------------------------
def test_desk_demo_offline(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, out, err = run(capsys, "desk", "demo", "--journal-dir", str(tmp_path / "j"))
    assert code == EXIT_OK, err
    assert "status:          decided" in out and "final forecast" in out and "cost:" in out
    assert list((tmp_path / "j").glob("*.jsonl"))


def test_desk_run_requires_credentials(capsys, env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, _, err = run(capsys, "desk", "run", "--config", env["config"])
    assert code == EXIT_USAGE and "ANTHROPIC_API_KEY" in err


def test_desk_replay_needs_confirmation(capsys, env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    idx = env["bars"].index
    code, out, _ = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                       "--end", str(idx[4400]), "--every", "12")
    assert code == EXIT_CONFIRM
    assert "THIS CALLS THE PAID CLAUDE API" in out and "estimated cost" in out and "worst case" in out
    assert "refusing to start without --yes" in out
    # with --yes but no key (and not --fake) it must still refuse
    code, _, err = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                       "--end", str(idx[4400]), "--every", "12", "--yes")
    assert code == EXIT_USAGE and "ANTHROPIC_API_KEY" in err


def test_desk_replay_fake_client_through_same_risk(capsys, env, tmp_path):
    idx = env["bars"].index
    out_dir = tmp_path / "replay"
    code, out, err = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                         "--end", str(idx[4239]), "--every", "24", "--fake", "--yes", "--out", str(out_dir),
                         "--journal-dir", str(tmp_path / "journal"))
    assert code == EXIT_OK, err
    assert "OFFLINE FAKE CLIENT" in out and "quant_only" in out
    dec = pd.read_csv(out_dir / "desk_decisions.csv")
    assert len(dec) == 10                                   # 240 bars, one cycle every 24
    decided = dec[dec["status"] == "decided"]
    assert len(decided) + (dec["status"] == "skipped").sum() == len(dec)
    # overlay 'scale 0.5' can only shrink the quant forecast, never flip it
    assert (dec["final"].abs() <= dec["quant"].abs() + 1e-12).all()
    assert (np.sign(dec["final"]) * np.sign(dec["quant"]) >= 0).all()
    s = json.loads((out_dir / "summary.json").read_text())
    assert s["fake_client"] and s["n_cycles"] == len(dec) - (dec["status"] == "budget_exhausted").sum()
    assert (out_dir / "books" / "desk" / "timeseries.parquet").exists() and (out_dir / "tearsheet.html").exists()
    ts = pd.read_parquet(out_dir / "books" / "desk" / "timeseries.parquet")
    q = pd.read_parquet(out_dir / "books" / "quant_only" / "timeseries.parquet")
    assert (ts["positions"].abs() <= q["positions"].abs().max() + 1e-9).all()


def test_desk_replay_budget_cap(capsys, env, tmp_path):
    idx = env["bars"].index
    code, out, err = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                         "--end", str(idx[4239]), "--every", "24", "--fake", "--yes", "--max-cost", "0.000001",
                         "--out", str(tmp_path / "r"), "--journal-dir", str(tmp_path / "j"))
    assert code == EXIT_OK, err
    dec = pd.read_csv(tmp_path / "r" / "desk_decisions.csv")
    paid = dec[dec["cost_usd"] > 0]
    assert len(paid) == 1                                    # the first paid cycle exhausts the budget
    after = dec.loc[paid.index[0] + 1:]
    assert len(after) and (after["status"] == "budget_exhausted").all()
    assert np.allclose(after["final"], after["quant"])     # quant forecast from then on
    assert "exhausted" in out
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["budget_usd"] == pytest.approx(0.000001)


# ---- rl / live delegation ------------------------------------------------------------------------------
def test_live_run_unavailable_and_delegation(capsys, env, monkeypatch):
    cfg = str(REPO / "configs" / "live_paper.yaml")
    monkeypatch.setitem(sys.modules, "aurum.live.runner", None)   # simulate an absent module
    code, out, err = run(capsys, "live", "run", "--config", cfg)
    assert code == EXIT_UNAVAILABLE and "aurum.live.runner" in err and "dry_run=True" in out
    calls = {}
    fake = types.ModuleType("aurum.live.runner")

    def run_from_config(c, *, i_understand_real_money):
        calls["cfg"], calls["flag"] = c, i_understand_real_money
        return 0

    fake.run_from_config = run_from_config
    monkeypatch.setitem(sys.modules, "aurum.live.runner", fake)
    code, out, _ = run(capsys, "live", "run", "--config", cfg, "--i-understand-real-money")
    assert code == EXIT_OK and calls["flag"] is True and calls["cfg"].live.dry_run is True
    assert "has no effect" in out  # allow_live_real is false in live_paper.yaml


def test_rl_train_delegation_and_unavailable(capsys, env, monkeypatch):
    monkeypatch.setitem(sys.modules, "aurum.rl.train", None)
    code, _, err = run(capsys, "rl", "train", "--config", env["config"])
    assert code == EXIT_UNAVAILABLE and "aurum[rl]" in err
    fake = types.ModuleType("aurum.rl.train")
    fake.train_from_config = lambda c: {"trained": c.name}
    monkeypatch.setitem(sys.modules, "aurum.rl.train", fake)
    code, out, _ = run(capsys, "rl", "train", "--config", env["config"])
    assert code == EXIT_OK and "cli_test" in out


def test_live_artifact_then_paper_run_with_real_runner(capsys, env, tmp_path):
    runner = pytest.importorskip("aurum.live.runner")
    if not (hasattr(runner, "main") and hasattr(getattr(runner, "LiveConfig", None), "from_mapping")):
        pytest.skip("aurum.live.runner has no main/LiveConfig.from_mapping")
    bars = env["bars"]
    art, state = tmp_path / "art", tmp_path / "state"
    sets = ["--set", f"live.artifact_dir={art}", "--set", f"live.state_dir={state}",
            "--set", "live.options={paper: {bars_path: '%s', warmup_bars: 4500}}" % (env["data"] / "xauusd_H1.parquet")]
    wf_dir = tmp_path / "wf"
    code, _, err = run(capsys, "walkforward", "--config", env["config"], "--out", str(wf_dir), "--no-tearsheet")
    assert code == EXIT_OK, err
    code, out, err = run(capsys, "live", "artifact", "--config", env["config"], "--at", str(bars.index[4400]),
                         "--from-run", str(wf_dir), *sets)
    assert code == EXIT_OK, err
    assert (art / "manifest.json").exists() and (art / "strategies.pkl").exists()
    assert "weights:" in out and json.loads((art / "backtest.json").read_text())["source_run"] == str(wf_dir)
    assert run(capsys, "live", "artifact", "--config", env["config"], "--from-run", str(tmp_path / "nope"),
               *sets)[0] == EXIT_USAGE
    manifest = json.loads((art / "manifest.json").read_text())
    assert manifest["strategies"] == ["cli_stub_mom", "cli_stub_slow"]
    assert pd.Timestamp(manifest["training"]["train_end"]) <= bars.index[4400]
    code, out, err = run(capsys, "live", "run", "--config", env["config"], "--max-cycles", "3", *sets)
    assert code == EXIT_OK, err
    assert "dry_run=True" in out
    resolved = yaml.safe_load((state / "aurum_live_config.yaml").read_text())
    assert resolved["dry_run"] is True and resolved["allow_live_real"] is False
    assert resolved["sizer"]["target_vol"] == 0.10 and resolved["timeframe"] == "H1"  # same sizing as research
    lines = [json.loads(x) for x in (state / "decisions.jsonl").read_text().splitlines()]
    decisions = [x for x in lines if x.get("type") == "decision"]
    assert 1 <= len(decisions) <= 3
    assert set(decisions[0]["forecasts"]) == {"cli_stub_mom", "cli_stub_slow"}
    # the live bar is after the artifact's training window (out of sample)
    assert pd.Timestamp(decisions[0]["bar_time"]) > pd.Timestamp(manifest["training"]["train_end"])


def test_data_download_offline_from_local_cache(capsys, tmp_path):
    cache = REPO / "cache" / "dukascopy"
    if not (cache / "XAUUSD" / "2024" / "00" / "15").exists():
        pytest.skip("local Dukascopy cache not present (gitignored)")
    out = tmp_path / "store"
    code, text, err = run(capsys, "data", "download", "--offline", "--no-macro", "--timeframes", "H1",
                          "--start", "2024-01-15", "--end", "2024-01-16", "--cache", str(cache), "--out", str(out))
    assert code == EXIT_OK, err
    assert (out / "xauusd_H1.parquet").exists() and "H1:" in text
    code, text, _ = run(capsys, "data", "info", "--dir", str(out))
    assert code == EXIT_OK and "xauusd_H1.parquet" in text and "ok" in text


# ---- adversarial review -------------------------------------------------------------------------------
def test_desk_replay_policy_holds_at_every_bar_between_cycles(capsys, env, tmp_path):
    """Between desk cycles the last decision is re-applied by the policy to the CURRENT quant
    forecast. Holding the last *forecast* instead let an 'overlay' desk keep positions the
    quant book had reversed (the desk book was on the opposite side of the quant book on more
    than half of the bars): overlay may only scale toward zero or veto, at EVERY bar."""
    idx = env["bars"].index
    out_dir = tmp_path / "replay"
    code, _, err = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                       "--end", str(idx[4239]), "--every", "24", "--fake", "--yes", "--out", str(out_dir),
                       "--journal-dir", str(tmp_path / "journal"))
    assert code == EXIT_OK, err
    d = pd.read_parquet(out_dir / "books" / "desk" / "timeseries.parquet")
    q = pd.read_parquet(out_dir / "books" / "quant_only" / "timeseries.parquet")
    fd, fq = d["forecast"].to_numpy()[:-1], q["forecast"].to_numpy()[:-1]   # last bar: no decision
    assert (np.abs(fd) <= np.abs(fq) + 1e-12).all()
    assert (np.sign(fd) * np.sign(fq) >= 0).all()
    # the scripted Chief always says "scale 0.5": after a decided cycle every bar is 0.5 * quant
    dec = pd.read_csv(out_dir / "desk_decisions.csv")
    status = pd.Series(dec["status"].to_numpy(), index=dec["bar"].to_numpy() - 4000)
    last = status.reindex(range(len(fd))).ffill().to_numpy()
    decided = last == "decided"
    assert decided.sum() > 100
    np.testing.assert_allclose(fd[decided], 0.5 * fq[decided], atol=1e-12)
    np.testing.assert_allclose(fd[last == "skipped"], fq[last == "skipped"], atol=1e-12)
    # the desk book never takes the opposite side of the quant book
    assert (np.sign(d["positions"]) * np.sign(q["positions"]) >= 0).all()
    meta = json.loads((out_dir / "books" / "desk" / "meta.json").read_text())
    assert meta["hook"]["every"] == 1 and meta["hook"]["calls"] == len(fd)


def test_desk_replay_rejects_bad_cadence_and_budget(capsys, env):
    idx = env["bars"].index
    base = ["desk", "replay", "--config", env["config"], "--start", str(idx[4000]), "--end", str(idx[4100]),
            "--fake", "--yes"]
    code, _, err = run(capsys, *base, "--every", "0")          # used to fall back silently to the default
    assert code == EXIT_USAGE and "--every" in err
    code, _, err = run(capsys, *base, "--max-cost", "0")
    assert code == EXIT_USAGE and "--max-cost" in err
    code, out, _ = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                       "--end", str(idx[4240]), "--every", "24")
    assert code == EXIT_CONFIRM and "= 10 cycles" in out      # decisions at bars 4000..4239


def test_live_options_cannot_override_safety_fields(capsys, env, monkeypatch):
    """live.options is passed through to the runner, but it must not silently override the typed,
    validated fields (allow_live_real / dry_run / broker / magic / live risk limits): the CLI
    banner and the semantic checks read the typed fields."""
    cfg = str(REPO / "configs" / "live_paper.yaml")
    for opt in ("{allow_live_real: true}", "{dry_run: false}", "{broker: mt5}", "{magic: 1}",
                "{risk: {max_daily_loss: 0.5}}", "{desk: {mode: discretionary}}"):
        code, _, err = run(capsys, "config", "validate", "--config", cfg, "--set", f"live.options={opt}")
        assert code == EXIT_USAGE and "live.options" in err, opt
    code, _, err = run(capsys, "config", "validate", "--config", cfg,
                       "--set", "live.options={mt5: {env: {MT5_PASSWORD: hunter2}}}")
    assert code == EXIT_USAGE and "credentials" in err and "hunter2" not in err
    # additive runner settings are still allowed
    code, _, _ = run(capsys, "config", "validate", "--config", cfg,
                     "--set", "live.options={on_error: flatten, desk: {on_error: hold}}")
    assert code == EXIT_OK
    # and `live run` refuses before touching the runner
    called = {}
    fake = types.ModuleType("aurum.live.runner")
    fake.run_from_config = lambda c, *, i_understand_real_money: called.setdefault("ran", True)
    monkeypatch.setitem(sys.modules, "aurum.live.runner", fake)
    code, _, err = run(capsys, "live", "run", "--config", cfg, "--set", "live.options={allow_live_real: true}",
                       "--i-understand-real-money")
    assert code == EXIT_USAGE and not called


def test_live_refuses_a_sizer_it_cannot_reproduce(capsys, env, monkeypatch):
    """Research sized with FixedFractionalSizer but live built a VolTargetSizer from the same
    numbers: the live book would not be the backtested book (train/serve skew)."""
    cfg = str(REPO / "configs" / "live_paper.yaml")
    called = {}
    fake = types.ModuleType("aurum.live.runner")
    fake.run_from_config = lambda c, *, i_understand_real_money: called.setdefault("ran", True)
    fake.save_artifact = lambda *a, **k: called.setdefault("saved", True)
    monkeypatch.setitem(sys.modules, "aurum.live.runner", fake)
    code, _, err = run(capsys, "live", "run", "--config", cfg, "--set", "sizing.method=fixed_fractional")
    assert code == EXIT_USAGE and "vol_target" in err and not called
    code, _, err = run(capsys, "live", "artifact", "--config", env["config"], "--set", "sizing.method=fixed_fractional")
    assert code == EXIT_USAGE and "vol_target" in err and not called


def test_desk_replay_caps_each_cycle_at_the_remaining_budget(capsys, env, tmp_path):
    """The replay budget is hard: each cycle's desk budget is min(per-cycle cap, what is left),
    so only an API call already in flight can overshoot it (not a whole cycle's cap)."""
    idx = env["bars"].index
    jdir = tmp_path / "j"
    code, _, err = run(capsys, "desk", "replay", "--config", env["config"], "--start", str(idx[4000]),
                       "--end", str(idx[4239]), "--every", "24", "--fake", "--yes", "--max-cost", "0.015",
                       "--out", str(tmp_path / "r"), "--journal-dir", str(jdir))
    assert code == EXIT_OK, err
    starts = []
    for p in sorted(jdir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("event") == "cycle_start":
                starts.append(ev)
    caps = [ev["limits"]["max_cost_usd"] for ev in starts]
    dec = pd.read_csv(tmp_path / "r" / "desk_decisions.csv")
    paid = dec.loc[dec["status"] != "budget_exhausted", "cost_usd"].to_numpy()
    assert len(caps) == len(paid) >= 2
    assert caps[0] == pytest.approx(0.015)                 # min(desk cap 1.0, the full budget)
    for i in range(1, len(caps)):                          # then: what is left of the budget
        assert caps[i] == pytest.approx(0.015 - paid[:i].sum())
    assert (dec["status"] == "budget_exhausted").any()
