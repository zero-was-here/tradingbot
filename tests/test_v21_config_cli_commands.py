"""v2.1 CLI: --version, strict `config validate`, help text, desk opt-in (agents.enabled),
nested desk settings for the live runner, IN-SAMPLE backtest header, live artifact checks,
rl train env from the top-level config, runner INFO logs and the extension hook."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

import aurum
from aurum.cli import EXIT_CONFIRM, EXIT_OK, EXIT_USAGE, build_parser, main
from aurum.core.config import ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]
SYNTH = {"name": "v21_cli", "seed": 3,
         "data": {"synthetic": {"n": 3000, "model": "trend", "seed": 3}, "events": "none"},
         "strategies": [{"name": "ema_cross", "params": {"fast": 8, "slow": 32}},
                        {"name": "zscore_fade", "params": {"n": 24}}],
         "walkforward": {"min_train_bars": 200, "holdout_start": None, "executor": "serial", "n_boot": 200,
                         "pbo_splits": 4},
         "output": {"save_results": False, "tearsheet": False}}


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def _cfg(tmp_path: Path, data: dict, name: str = "c.yaml") -> str:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return str(p)


# ---- version / help --------------------------------------------------------------------------------
def test_version(capsys):
    text = (REPO / "pyproject.toml").read_text()
    expected = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    assert aurum.__version__ == expected
    code, out, _ = run(capsys, "--version")
    assert code == EXIT_OK and out.strip() == f"aurum {expected}"


def _walk(parser: argparse.ArgumentParser, path: str = "aurum"):
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for name, sub in action.choices.items():
                yield from _walk(sub, f"{path} {name}")
        else:
            yield path, action


def test_every_option_has_help(monkeypatch):
    monkeypatch.setitem(sys.modules, "aurum.cli_ext", None)      # core commands only
    missing = [f"{path}: {action.option_strings or action.dest}" for path, action in _walk(build_parser())
               if not action.help or action.help == argparse.SUPPRESS and action.dest != "help"]
    assert missing == []
    cmds = build_parser()._subparsers._group_actions[0].choices  # noqa: SLF001
    live = cmds["live"]._subparsers._group_actions[0].choices  # noqa: SLF001
    assert {"artifact", "run", "status", "reset-halt"} <= set(live)


def test_module_docstring_lists_every_command(monkeypatch):
    import aurum.cli as cli

    monkeypatch.setitem(sys.modules, "aurum.cli_ext", None)
    doc = cli.__doc__
    for path, _ in _walk(build_parser()):
        if path.count(" ") >= 1:
            assert path in doc, path


# ---- config validate --------------------------------------------------------------------------------
def test_config_validate_strict_reports_checked_warnings_and_skips(capsys, tmp_path, monkeypatch):
    code, out, _ = run(capsys, "config", "validate", "-c", str(REPO / "configs" / "fast.yaml"))
    assert code == EXIT_OK and "# checked:" in out and "strategies (7 of 7" in out and "# config OK" in out
    cfg = _cfg(tmp_path, {"strategies": [{"name": "rl_ppo", "params": {"artifact_dir": str(tmp_path / "x")}}]})
    monkeypatch.setitem(sys.modules, "aurum.live.runner", None)
    code, out, _ = run(capsys, "config", "validate", "-c", cfg)
    assert code == EXIT_OK and "# warning:" in out and "# skipped: live runner settings" in out
    bad = _cfg(tmp_path, {"strategies": [{"name": "tsmomentum"}, {"name": "ema_cross", "params": {"fst": 2}}],
                          "instrument": {"triple_swap_weekday": 7}}, "bad.yaml")
    code, out, err = run(capsys, "config", "show", "-c", bad)
    assert code == EXIT_USAGE and "3 configuration problems" in err
    assert "unknown strategy 'tsmomentum'" in err and "'fst' -> 'fast'" in err and "triple_swap_weekday" in err
    assert "strategies:" not in out                     # `show` prints nothing for an invalid config


def test_commands_validate_strictly_before_loading_data(capsys, tmp_path):
    cfg = _cfg(tmp_path, {"data": {"dir": str(tmp_path / "no_data")}, "strategies": [{"name": "not_registered"}]})
    code, _, err = run(capsys, "backtest", "-c", cfg, "--no-write")
    assert code == EXIT_USAGE and "unknown strategy 'not_registered'" in err   # not a missing-bars error


# ---- desk opt-in -------------------------------------------------------------------------------------
def test_desk_run_requires_agents_enabled(capsys, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"data": {"dir": str(tmp_path / "no_data")}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    code, _, err = run(capsys, "desk", "run", "-c", cfg)
    assert code == EXIT_USAGE and "agents.enabled" in err and "configs/desk_overlay.yaml" in err
    assert "sk-test" not in err
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    code, _, err = run(capsys, "desk", "run", "-c", cfg)                     # both problems at once
    assert code == EXIT_USAGE and "agents.enabled" in err and "ANTHROPIC_API_KEY" in err
    code, _, err = run(capsys, "desk", "run", "-c", cfg, "--set", "agents.enabled=true")
    assert code == EXIT_USAGE and "ANTHROPIC_API_KEY" in err and "agents.enabled" not in err


def test_desk_replay_requires_agents_enabled_unless_fake(capsys, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, SYNTH)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    base = ["desk", "replay", "-c", cfg, "--start", "2020-05-01", "--end", "2020-05-05", "--every", "24"]
    code, out, _ = run(capsys, *base)
    assert code == EXIT_CONFIRM and "agents.enabled is false" in out          # said before anything is spent
    code, _, err = run(capsys, *base, "--yes")
    assert code == EXIT_USAGE and "agents.enabled" in err and "desk_overlay.yaml" in err
    code, out, err = run(capsys, *base, "--yes", "--fake", "--out", str(tmp_path / "r"),
                         "--journal-dir", str(tmp_path / "j"))
    assert code == EXIT_OK, err
    assert "allowed for this --fake replay only" in out and "EQUAL weights" in out


# ---- nested desk settings -> runner ----------------------------------------------------------------
def test_live_runner_mapping_passes_the_full_desk_config():
    cfg = load_config(REPO / "configs" / "desk_overlay.yaml", env={})
    m = cfg.live_runner_mapping()
    assert m["desk_config"]["chief"] == {"effort": "high", "max_turns": 8}
    assert m["desk_config"]["specialist"] == {"effort": "medium", "max_turns": 5}
    assert m["desk_config"]["max_cost_usd_per_cycle"] == 2.0 and m["desk_config"]["cache_ttl"] == "1h"
    assert "chief" not in m["desk"]["config"] and m["desk"]["config"]["max_cost_usd_per_cycle"] == 2.0
    assert m["desk"]["enabled"] is True and m["desk"]["mode"] == "overlay"
    assert yaml.safe_load(yaml.safe_dump(m)) == m                         # what `live run` writes to disk
    runner = pytest.importorskip("aurum.live.runner")
    live_cls = runner.LiveConfig
    if "desk_config" not in {f.name for f in dataclasses.fields(live_cls)}:
        pytest.skip("this aurum.live.runner does not accept desk_config")
    desk = live_cls.from_mapping(m).build_desk_config()
    assert desk.chief.effort == "high" and desk.chief.max_turns == 8 and desk.max_cost_usd_per_cycle == 2.0
    assert desk.cache_ttl == "1h"


def test_desk_config_cannot_come_from_live_options():
    """DeskConfig values (budgets, models) come only from agents.desk, never from live.options."""
    with pytest.raises(ConfigError, match=r"live\.options\.desk_config.*agents\.desk"):
        load_config({"live": {"options": {"desk_config": {"max_cost_usd_per_cycle": 99.0}}}}, env={})
    with pytest.raises(ConfigError, match=r"live\.options\.desk\.config.*agents\.desk"):
        load_config({"live": {"options": {"desk": {"config": {"max_cost_usd_per_cycle": 99.0}}}}}, env={})
    ok = load_config({"live": {"options": {"desk": {"on_error": "hold"}}}}, env={})   # runner-only knobs are fine
    assert ok.live_runner_mapping()["desk"]["on_error"] == "hold"


# ---- backtest / live artifact ----------------------------------------------------------------------
def test_backtest_without_start_says_in_sample(capsys, tmp_path):
    code, out, err = run(capsys, "backtest", "-c", _cfg(tmp_path, SYNTH), "--no-write")
    assert code == EXIT_OK, err
    first = out.splitlines()[0]
    assert first.startswith("== IN-SAMPLE backtest") and "--start" in first
    assert "== backtest (IN-SAMPLE" in out
    code, out, err = run(capsys, "backtest", "-c", _cfg(tmp_path, SYNTH), "--no-write", "--start", "2020-04-01")
    assert code == EXIT_OK, err
    assert "IN-SAMPLE" not in out


def test_live_artifact_refuses_existing_out_before_fitting(capsys, tmp_path, monkeypatch):
    import aurum.research.walkforward as wf

    out_dir = tmp_path / "art"
    out_dir.mkdir()
    monkeypatch.setattr(wf, "fit_quant_book", lambda *a, **k: pytest.fail("fitted before the --out check"))
    cfg = _cfg(tmp_path, {**SYNTH, "data": {"dir": str(tmp_path / "no_data")}})   # loading data would fail
    code, _, err = run(capsys, "live", "artifact", "-c", cfg, "--out", str(out_dir))
    assert code == EXIT_USAGE and "already exists" in err and "--overwrite" in err


def test_live_artifact_warns_about_equal_weights(capsys, tmp_path):
    pytest.importorskip("aurum.live.runner")
    code, out, err = run(capsys, "live", "artifact", "-c", _cfg(tmp_path, SYNTH), "--out", str(tmp_path / "a"))
    assert code == EXIT_OK, err
    assert "WARNING: live artifact: the combiner fell back to EQUAL weights" in out and "--from-run" in out
    assert (tmp_path / "a" / "manifest.json").exists()


# ---- rl train --------------------------------------------------------------------------------------
def test_rl_train_builds_its_config_with_rl_config_from_aurum(capsys, tmp_path, monkeypatch):
    seen: dict = {}
    fake = types.ModuleType("aurum.rl.train")
    sentinel = types.SimpleNamespace(env=types.SimpleNamespace(sizer={"target_vol": 0.07, "max_leverage": 1.5},
                                                               risk={"max_daily_loss": 0.03}))

    def rl_config_from_aurum(cfg):
        seen.setdefault("cfgs", []).append(cfg)
        return sentinel

    def train_ppo(md_train, md_val, *, config, out_dir):
        seen["config"], seen["n"] = config, (len(md_train.bars), len(md_val.bars))
        return types.SimpleNamespace(artifact_dir=out_dir, val_eval=None)

    fake.rl_config_from_aurum, fake.train_ppo = rl_config_from_aurum, train_ppo
    monkeypatch.setitem(sys.modules, "aurum.rl.train", fake)
    data = {**SYNTH, "sizing": {"target_vol": 0.07}, "rl": {"train_start": None, "train_end": "2020-04-30",
                                                           "val_end": "2020-06-30", "out_dir": str(tmp_path / "rl"),
                                                           "params": {"env": {"window": 8}}}}
    code, out, err = run(capsys, "rl", "train", "-c", _cfg(tmp_path, data))
    assert code == EXIT_OK, err
    assert seen["config"] is sentinel and seen["cfgs"][-1].sizing.target_vol == 0.07
    assert seen["n"][0] >= 1000 and seen["n"][1] >= 100
    assert "env: sizer from config (target_vol 0.07" in out and "rl.params.env overrides ['window']" in out


def test_real_rl_config_uses_top_level_sections():
    rl_train = pytest.importorskip("aurum.rl.train")
    if not hasattr(rl_train, "rl_config_from_aurum"):
        pytest.skip("aurum.rl.train.rl_config_from_aurum not available")
    from aurum.cli import _rl_train_config

    cfg = load_config({"sizing": {"target_vol": 0.07}, "costs": {"slippage_fixed": 0.05},
                       "rl": {"params": {"n_envs": 2, "env": {"window": 8}}}}, env={})
    rc = _rl_train_config(rl_train, cfg)
    assert rc.env.sizer["target_vol"] == 0.07 and rc.env.costs["slippage_fixed"] == 0.05
    assert rc.env.window == 8 and rc.n_envs == 2 and rc.seed == cfg.seed


# ---- live run: per-decision INFO logs at default verbosity -------------------------------------------
def test_live_run_shows_runner_info_logs_by_default(capsys, tmp_path, monkeypatch):
    seen = {}
    fake = types.ModuleType("aurum.live.runner")

    class LiveConfig:
        @classmethod
        def from_mapping(cls, m):
            return cls()

    def fake_main(argv):
        log = logging.getLogger("aurum.live.runner")
        seen["info"] = log.isEnabledFor(logging.INFO)
        seen["root"] = logging.getLogger().getEffectiveLevel()
        seen["argv"] = argv
        return 0

    fake.LiveConfig, fake.main = LiveConfig, fake_main
    monkeypatch.setitem(sys.modules, "aurum.live.runner", fake)
    before = logging.getLogger("aurum.live.runner").level
    code, out, err = run(capsys, "live", "run", "-c", str(REPO / "configs" / "live_paper.yaml"),
                         "--set", f"live.state_dir={tmp_path / 'state'}", "--max-cycles", "2")
    assert code == EXIT_OK, err
    assert seen["info"] is True and seen["root"] >= logging.WARNING         # only the runner's INFO lines
    assert seen["argv"][-2:] == ["--max-cycles", "2"] and "desk=off" in out
    assert logging.getLogger("aurum.live.runner").level == before            # restored afterwards


# ---- extension hook --------------------------------------------------------------------------------
def _ext(register) -> types.ModuleType:
    mod = types.ModuleType("aurum.cli_ext")
    mod.register_all = register
    return mod


def test_cli_extension_commands_are_added(capsys, monkeypatch):
    def register_all(sub):
        p = sub.add_parser("hello", help="say hello")
        p.add_argument("--name", default="gold", help="who")
        p.set_defaults(func=lambda a: print(f"hello {a.name}") or 0)

    monkeypatch.setitem(sys.modules, "aurum.cli_ext", _ext(register_all))
    code, out, _ = run(capsys, "hello", "--name", "aurum")
    assert code == EXIT_OK and out.strip() == "hello aurum"


def test_broken_extension_never_breaks_core_commands(capsys, monkeypatch, caplog):
    def broken(sub):
        sub.add_parser("half", help="registered before the crash")
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "aurum.cli_ext", _ext(broken))
    with caplog.at_level(logging.WARNING, logger="aurum.cli"):
        parser = build_parser()
    assert "boom" in caplog.text and "skipped" in caplog.text
    choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert "half" not in choices                                     # rolled back
    assert run(capsys, "strategies", "list", "--json")[0] == EXIT_OK

    def hijack(sub):
        sub.add_parser("backtest", help="replace a core command")

    monkeypatch.setitem(sys.modules, "aurum.cli_ext", _ext(hijack))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="aurum.cli"):
        parser = build_parser()
    assert "clashes" in caplog.text
    bt = parser._subparsers._group_actions[0].choices["backtest"]  # noqa: SLF001
    assert any("--include-holdout" in a.option_strings for a in bt._actions)  # noqa: SLF001 - still the core one

    def no_func(sub):
        sub.add_parser("idle", help="no func")

    monkeypatch.setitem(sys.modules, "aurum.cli_ext", _ext(no_func))
    code, _, err = run(capsys, "idle")
    assert code == EXIT_USAGE and "nothing to run" in err

    monkeypatch.setitem(sys.modules, "aurum.cli_ext", types.ModuleType("aurum.cli_ext"))   # no register_all
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="aurum.cli"):
        build_parser()
    assert "register_all" in caplog.text
