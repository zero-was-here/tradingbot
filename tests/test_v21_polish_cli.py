"""v2.1 polish of the CLI and the config validation, synthetic data only:

* ``aurum backtest --include-holdout --no-write`` records its holdout look in the ledger, like
  ``walkforward --no-write`` (same location, same record shape as a written backtest);
* ``aurum data info --help`` describes what the command reads without ``--config``;
* ``aurum sweep`` resolves ``--strategy`` and the ``--grid`` parameter names before it loads
  any data (one did-you-mean error, exit 2);
* ``live.options.mt5`` keys are checked against ``MT5Broker``'s signature and unknown keys in
  ``live.options.desk`` are errors.

Every ledger written here lives under ``tmp_path``; nothing reads or writes the repository's
``runs/`` or ``data_store/``.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest
import yaml

from aurum.cli import EXIT_OK, EXIT_USAGE, main
from aurum.core.config import ConfigError, DataConfig, load_config

HOLDOUT = "2020-09-01"


def _synthetic_config(tmp_path: Path, **extra) -> Path:
    """6000 synthetic H1 bars from 2020-01-06 (to 2020-12-23), holdout from HOLDOUT, its own
    output root under tmp_path."""
    cfg = {
        "name": "polish_syn", "seed": 1,
        "data": {"synthetic": {"n": 6000, "model": "gbm", "seed": 1}},
        "strategies": [{"name": "tsmom"}, {"name": "ema_cross", "id": "ema_fast", "params": {"fast": 8, "slow": 32}}],
        "walkforward": {"train": "6M", "test": "2M", "holdout_start": HOLDOUT, "n_boot": 100,
                        "executor": "serial", "min_train_bars": 200},
        "output": {"dir": str(tmp_path / "runs"), "tearsheet": False, "dark_charts": False},
    }
    cfg.update(extra)
    p = tmp_path / "polish.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _ledger(tmp_path: Path) -> list[dict]:
    p = tmp_path / "runs" / "holdout_ledger.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


# ---- 1. backtest --include-holdout --no-write records the look ----------------------------------
def test_backtest_include_holdout_no_write_records_the_look(tmp_path, capsys):
    cfg = str(_synthetic_config(tmp_path))
    # without --include-holdout nothing is evaluated in the holdout, so nothing is recorded
    assert main(["backtest", "-c", cfg, "--start", "2020-07-01", "--strategy", "tsmom", "--no-write"]) == EXIT_OK
    assert _ledger(tmp_path) == []

    code = main(["backtest", "-c", cfg, "--start", "2020-07-01", "--strategy", "tsmom", "--include-holdout",
                 "--no-write"])
    out = capsys.readouterr().out
    assert code == EXIT_OK, out
    assert "NOT recorded" not in out
    (look,) = _ledger(tmp_path)
    assert look["kind"] == "backtest" and look["run_dir"] is None and look["data_source"] == "synthetic"
    assert look["holdout_start"].startswith(HOLDOUT) and look["evaluated_start"].startswith("2020-07-01")
    assert look["holdout_end"] == look["evaluated_end"] and look["n_holdout_bars"] > 0
    assert look["strategies"] == ["tsmom"] and look["metrics_window"] == "evaluated" and look["in_sample"] is False
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["holdout_ledger.jsonl"]   # no run dir

    # a WRITTEN backtest appends to the same ledger (<out>/..), with the same record shape, and
    # sees the --no-write look as a prior look of the same config
    out_dir = tmp_path / "runs" / "bt_written"
    assert main(["backtest", "-c", cfg, "--start", "2020-07-01", "--strategy", "tsmom", "--include-holdout",
                 "--out", str(out_dir)]) == EXIT_OK
    first, second = _ledger(tmp_path)
    assert set(first) == set(second)
    assert second["run_dir"] == str(out_dir) and (out_dir / "summary.json").exists()
    assert second["prior_looks"]["n_prior_same_config"] == 1 and second["prior_looks"]["n_prior_configs"] == 0
    assert first["config_hash"] == second["config_hash"] and first["metrics"] == second["metrics"]


def test_cli_holdout_ledger_location(tmp_path):
    from aurum.cli import _cli_holdout_ledger

    cfg = load_config(_synthetic_config(tmp_path), env={})
    assert _cli_holdout_ledger(cfg) is None                       # written: next to the run dir
    cfg.output.save_results = False
    assert _cli_holdout_ledger(cfg) == tmp_path / "runs" / "holdout_ledger.jsonl"


# ---- 2. data info --help --------------------------------------------------------------------------
def test_data_info_help_and_behaviour_without_config(tmp_path, capsys):
    from aurum.data.store import save_bars
    from aurum.data.synthetic import make_synthetic_bars

    assert main(["data", "info", "--help"]) == EXIT_OK
    text = " ".join(capsys.readouterr().out.split())
    assert "the built-in defaults" not in text
    assert "default: none, the --dir directory is inspected" in text
    assert "only applied with --config, ignored without it" in text

    store = tmp_path / "store"
    save_bars(make_synthetic_bars(300, "H1", seed=2, start="2019-01-07"), store / "xauusd_H1.parquet")
    # without --config: --dir is read and --set is ignored (as the help says)
    assert main(["data", "info", "--dir", str(store), "--set", "data.dir=/nonexistent/aurum"]) == EXIT_OK
    assert "xauusd_H1.parquet" in capsys.readouterr().out
    # with --config: the config's data.dir replaces --dir
    cfg = tmp_path / "info.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"dir": str(tmp_path / "missing")}}))
    assert main(["data", "info", "--dir", str(store), "-c", str(cfg)]) == EXIT_USAGE
    assert "missing does not exist" in capsys.readouterr().err


# ---- 3. sweep resolves --strategy / --grid before loading data -------------------------------------
@pytest.fixture
def no_data_load(monkeypatch):
    """Loading data fails loudly: the checks under test must run before it."""
    calls = []

    def load(self, *a, **k):
        calls.append(1)
        raise RuntimeError("data loaded")

    monkeypatch.setattr(DataConfig, "load", load)
    return calls


def test_sweep_unknown_strategy_fails_before_loading_data(tmp_path, capsys, caplog, no_data_load):
    cfg = str(_synthetic_config(tmp_path))
    with caplog.at_level(logging.WARNING):
        code = main(["sweep", "-c", cfg, "--strategy", "ema_crosss", "--grid", "fast=8,16", "--grid", "slow=64"])
    out, err = capsys.readouterr()
    assert code == EXIT_USAGE and no_data_load == []
    assert "loaded" not in out
    assert err.count("ema_crosss") == 1 and "did you mean ['ema_cross']" in err
    assert "['tsmom', 'ema_fast', 'ema_cross']" in err            # the config's ids and names
    assert not [r for r in caplog.records if "ema_crosss" in r.getMessage()]   # not logged (once, on stderr)


def test_sweep_unknown_grid_parameter_fails_before_loading_data(tmp_path, capsys, no_data_load):
    cfg = str(_synthetic_config(tmp_path))
    code = main(["sweep", "-c", cfg, "--strategy", "ema_fast", "--grid", "fsat=4,8", "--grid", "slow=64"])
    err = capsys.readouterr().err
    assert code == EXIT_USAGE and no_data_load == []
    assert "unknown parameter(s) ['fsat'] for strategy 'ema_cross'" in err and "'fsat' -> 'fast'" in err
    assert "valid: ['fast', 'slow'," in err


@pytest.mark.parametrize("name", ["ema_fast", "ema_cross", "tsmom", "donchian"])
def test_sweep_valid_names_pass_the_preflight(tmp_path, capsys, no_data_load, name):
    """A config id, a config name and a registry name that is not in the config all resolve;
    the command then goes on to load the data (here: the sentinel error, exit 1)."""
    grid = {"ema_fast": "fast=4,8", "ema_cross": "slow=32,64", "tsmom": "vol_halflife=12,24",
            "donchian": "entry_n=60,120"}[name]
    code = main(["sweep", "-c", str(_synthetic_config(tmp_path)), "--strategy", name, "--grid", grid])
    assert code == 1 and no_data_load == [1], capsys.readouterr().err


def test_sweep_preflight_resolution(tmp_path):
    from aurum.cli import CLIError
    from aurum.cli_ext import _sweep_preflight

    cfg = load_config(_synthetic_config(tmp_path), env={}, strict=True)
    assert _sweep_preflight(cfg, "ema_fast", {"fast": [4, 8]}) == "ema_cross"     # config id -> registry name
    assert _sweep_preflight(cfg, "rsi2", {}) == "rsi2"                              # registry only
    with pytest.raises(CLIError) as ei:
        _sweep_preflight(cfg, "nope_strategy", {})
    assert ei.value.code == EXIT_USAGE and "did you mean" not in str(ei.value)


# ---- 4a. live.options.mt5 keys vs MT5Broker's signature ---------------------------------------------
def _problems(data: dict, *, strict: bool = True) -> list[str]:
    with pytest.raises(ConfigError) as ei:
        load_config(data, env={}, strict=strict)
    return ei.value.problems


@pytest.mark.parametrize("strict", [True, False])
def test_mt5_option_keys_are_checked_against_the_broker_signature(strict):
    problems = _problems({"live": {"options": {"mt5": {"server_tzz": "auto", "symbol": "XAUUSD.a",
                                                       "connect": False, "password": "x"}}}}, strict=strict)
    text = "\n".join(problems)
    assert "live.options.mt5.server_tzz: unknown MT5Broker setting (did you mean 'server_tz'?)" in text
    assert "live.options.mt5.symbol: not a config setting (the runner passes live.symbol" in text
    assert "live.options.mt5.connect: not a config setting" in text
    assert sum("live.options.mt5.password" in p for p in problems) == 1      # the credential check only
    valid = text.split("valid keys: ")[1].splitlines()[0]
    for k in ("server_tz", "server_tz_fallback", "tz_state_path", "tz_reprobe_seconds", "deviation_points",
              "price_basis", "requote_retries", "requote_delay_s", "tz_probe_seconds"):
        assert repr(k) in valid, k
    for k in ("symbol", "instrument", "connect", "mt5", "env", "clock", "sleep"):
        assert repr(k) not in valid, k


def test_valid_mt5_options_pass_and_are_listed_as_checked():
    cfg = load_config({"live": {"options": {"mt5": {"server_tz": "auto", "server_tz_fallback": "NY+7",
                                                    "tz_reprobe_seconds": 60, "deviation_points": 30}}}},
                      env={}, strict=True)
    rep = cfg.check(strict=True)
    assert rep.ok and "live.options.mt5 (MT5Broker keyword arguments)" in rep.checked
    plain = load_config(None, env={}).check(strict=True)          # no mt5 options: nothing listed
    assert not [c for c in plain.checked if "mt5" in c] and not [s for s in plain.skipped if "mt5" in s]


@pytest.fixture
def real_runner():
    """Import the real live package first: ``aurum.live`` imports ``aurum.live.mt5`` itself, so
    the module replacements below must not be the first import of the package."""
    return pytest.importorskip("aurum.live.runner")


def test_mt5_check_inspects_the_signature_without_constructing(monkeypatch, real_runner):
    built = []

    class FakeBroker:
        def __init__(self, symbol="XAUUSD", *, instrument=None, server_tz="auto", connect=True, mt5=None):
            built.append(1)

    fake = types.ModuleType("aurum.live.mt5")
    fake.MT5Broker = FakeBroker
    monkeypatch.setitem(sys.modules, "aurum.live.mt5", fake)
    problems = _problems({"live": {"options": {"mt5": {"deviation_points": 5}}}})
    assert problems == ["live.options.mt5.deviation_points: unknown MT5Broker setting; valid keys: ['server_tz']"]
    assert built == []

    class KwBroker:
        def __init__(self, symbol="XAUUSD", **kw):
            built.append(1)

    fake.MT5Broker = KwBroker
    rep = load_config({"live": {"options": {"mt5": {"anything": 1}}}}, env={}).check(strict=True)
    assert rep.ok and any("**kwargs" in s and "['anything']" in s for s in rep.skipped) and built == []


def test_mt5_check_is_skipped_and_listed_when_the_module_cannot_be_imported(monkeypatch, real_runner):
    monkeypatch.setitem(sys.modules, "aurum.live.mt5", None)
    cfg = load_config({"live": {"options": {"mt5": {"server_tzz": "auto"}}}}, env={}, strict=True)
    rep = cfg.check(strict=True)
    assert rep.ok and not [c for c in rep.checked if "mt5" in c]
    assert "live runner settings (aurum.live.runner.LiveConfig)" in rep.checked     # the rest still runs
    (skip,) = [s for s in rep.skipped if s.startswith("live.options.mt5")]
    assert "cannot be inspected" in skip and "keys not checked: ['server_tzz']" in skip


def test_cli_config_validate_rejects_an_mt5_typo(capsys):
    assert main(["config", "validate", "--set", "live.options.mt5.price_basiss=mid"]) == EXIT_USAGE
    assert "live.options.mt5.price_basiss: unknown MT5Broker setting (did you mean 'price_basis'?)" in \
        capsys.readouterr().err


# ---- 4b. unknown keys in live.options.desk ----------------------------------------------------------
@pytest.mark.parametrize("strict", [True, False])
def test_unknown_desk_option_keys_are_errors(strict):
    problems = _problems({"live": {"options": {"desk": {"on_eror": "hold", "max_abs_forcast": 0.5,
                                                        "foo": 1}}}}, strict=strict)
    assert problems == [
        "live.options.desk.on_eror: unknown desk setting (did you mean 'on_error'?); valid keys: ['on_error']",
        "live.options.desk.max_abs_forcast: unknown desk setting (did you mean 'max_abs_forecast'? it is set "
        "from agents.policy.max_abs_forecast, not in live.options); valid keys: ['on_error']",
        "live.options.desk.foo: unknown desk setting; valid keys: ['on_error']",
    ]                                                            # each once: no "live (runner settings)" echo


def test_valid_desk_options_still_pass():
    cfg = load_config({"live": {"options": {"desk": {"on_error": "flat"}}}}, env={}, strict=True)
    assert cfg.live_runner_mapping()["desk"]["on_error"] == "flat"
    # the typed desk keys stay "set it in its own field" problems, not "unknown"
    (p,) = _problems({"live": {"options": {"desk": {"mode": "discretionary"}}}})
    assert p.startswith("live.options.desk.mode: overrides a value the typed config already sets")


def test_runner_live_config_refuses_unknown_desk_keys():
    runner = pytest.importorskip("aurum.live.runner")
    assert set(runner.DESK_KEYS) == {"enabled", "mode", "on_failure", "on_error", "max_abs_forecast",
                                     "min_confidence", "config"}
    with pytest.raises(ValueError, match=r"desk: unknown key\(s\) \['on_eror'\].*'on_eror' -> 'on_error'"):
        runner.LiveConfig(desk={"on_eror": "hold"})
    # everything the typed config hands the runner is accepted
    m = load_config({"live": {"use_desk": True, "options": {"desk": {"on_error": "hold"}}},
                     "agents": {"enabled": True}}, env={}, strict=True).live_runner_mapping()
    assert runner.LiveConfig.from_mapping(m).desk["on_error"] == "hold"
