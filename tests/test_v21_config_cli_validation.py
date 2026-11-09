"""v2.1 config validation: every object a command builds is checked at load time and every
problem is reported at once (strategies via the registry, the feature pipeline, instrument
ranges, the live runner's own settings, rl.params via rl_config_from_aurum)."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import yaml

from aurum.core.config import ConfigError, ValidationReport, load_config

REPO = Path(__file__).resolve().parents[1]

#: the RL stack (aurum[rl]) is optional: the core and minimum-versions CI jobs do not install it
HAS_RL = all(importlib.util.find_spec(m) is not None for m in ("gymnasium", "stable_baselines3", "torch"))


def _problems(data: dict, *, strict: bool = True) -> list[str]:
    with pytest.raises(ConfigError) as ei:
        load_config(data, env={}, strict=strict)
    return ei.value.problems


# ---- all at once ---------------------------------------------------------------------------------
def test_every_problem_is_reported_at_once():
    bad = {
        "instrument": {"triple_swap_weekday": 5, "rollover_hour_utc": 24},
        "features": {"groups": ["returns", "trnd"], "overrides": {"returns": {"horizon": [1]}}, "warmup": -1},
        "strategies": [{"name": "tsmom", "params": {"horizon": [24]}}, {"name": "emacross"}],
        "live": {"options": {"on_eror": "hold"}},
        "rl": {"params": {"n_env": 2}},
        "walkforward": {"pbo_splits": 3},
    }
    problems = _problems(bad)
    text = "\n".join(problems)
    for needle in ("instrument.triple_swap_weekday", "instrument.rollover_hour_utc", "'trnd'", "did you mean 'trend'",
                   "features.overrides.returns.horizon", "features.warmup", "strategies[0].params (tsmom)",
                   "'horizon' -> 'horizons'", "strategies[1].name: unknown strategy 'emacross'", "ema_cross",
                   "live.options.on_eror", "did you mean 'on_error'", "walkforward.pbo_splits"):
        assert needle in text, needle
    if HAS_RL:                       # rl.params are checked through aurum.rl.train (skipped without it)
        assert "rl.params" in text and "n_env" in text
    assert len(problems) >= 10 if HAS_RL else len(problems) >= 9


def test_check_returns_a_report_without_raising():
    cfg = load_config(REPO / "configs" / "fast.yaml", env={})
    rep = cfg.check(strict=True)
    assert isinstance(rep, ValidationReport) and rep.ok and rep.problems == []
    joined = " | ".join(rep.checked)
    for part in ("instrument", "features (FeaturePipeline", "strategies (7 of 7", "live runner settings"):
        assert part in joined, part
    if HAS_RL:
        assert "rl.params" in joined
    else:                            # listed as skipped, with the extra to install
        assert any(x.startswith("rl.params") and "aurum[rl]" in x for x in rep.skipped), rep.skipped
    assert rep.raise_if_problems() is rep
    broken = load_config({"strategies": [{"name": "tsmom", "params": {"nope": 1}}]}, env={}, validate=False)
    rep2 = broken.check()
    assert not rep2.ok and "nope" in rep2.problems[0]
    with pytest.raises(ConfigError):
        rep2.raise_if_problems()


@pytest.mark.parametrize("path", sorted((REPO / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_pass_strict_validation(path):
    rep = load_config(path, env={}).check(strict=True)
    assert rep.ok, rep.problems
    assert not [w for w in rep.warnings if "unknown strategy" in w]


# ---- strategies ------------------------------------------------------------------------------------
def test_unknown_strategy_name_is_a_warning_unless_strict(caplog):
    data = {"strategies": [{"name": "my_custom_strategy"}]}
    with caplog.at_level(logging.WARNING, logger="aurum.core.config"):
        cfg = load_config(data, env={})                      # library load: custom strategies may register later
    assert "unknown strategy 'my_custom_strategy'" in caplog.text
    assert any("my_custom_strategy" in w for w in cfg.check().warnings)
    problems = _problems(data, strict=True)                  # CLI / `aurum config validate`
    assert problems == [p for p in problems if "my_custom_strategy" in p] and len(problems) == 1
    with pytest.raises(ConfigError, match="unknown strategy 'my_custom_strategy'"):
        cfg.build_strategies()


def test_unknown_strategy_params_are_errors_even_when_not_strict():
    # zscore_fade itself silently accepted unknown params before: the config now refuses the typo
    problems = _problems({"strategies": [{"name": "zscore_fade", "params": {"entry_z": 2.5, "exitz": 0.4}}]},
                         strict=False)
    assert len(problems) == 1 and "'exitz' -> 'exit_z'" in problems[0] and "valid:" in problems[0]


def test_constructor_errors_and_disabled_entries_are_checked():
    problems = _problems({"strategies": [
        {"name": "tsmom"},
        {"name": "intraday_seasonality", "enabled": False, "params": {"bucket_minutes": 5000}},
        {"name": "orb", "params": {"sessions": ["tokyo"]}},
    ]})
    text = "\n".join(problems)
    assert "strategies[1] (intraday_seasonality): bucket_minutes must be in [1, 1440]" in text
    assert "strategies[2] (orb)" in text and "tokyo" in text
    assert len(problems) == 2


@pytest.mark.skipif(not HAS_RL, reason="needs the optional RL stack (aurum[rl])")
def test_missing_rl_artifact_is_only_a_warning(tmp_path):
    data = {"strategies": [{"name": "rl_ppo", "params": {"artifact_dir": str(tmp_path / "not_trained_yet")}}]}
    cfg = load_config(data, env={}, strict=True)              # `aurum rl train` may create it from this config
    rep = cfg.check(strict=True)
    assert rep.ok and any("not_trained_yet" in w and "fail until" in w for w in rep.warnings)


def test_build_strategies_reports_the_config_index_and_unknown_params():
    cfg = load_config({"strategies": [{"name": "tsmom", "enabled": False}, {"name": "ema_cross"},
                                      {"name": "zscore_fade", "params": {"exitz": 1}}]}, env={}, validate=False)
    with pytest.raises(ConfigError) as ei:
        cfg.build_strategies()
    assert ei.value.problems == [p for p in ei.value.problems if p.startswith("strategies[2]")]
    built = load_config({"strategies": [{"name": "tsmom", "enabled": False}, {"name": "ema_cross"}]},
                        env={}).build_strategies()
    assert list(built) == ["ema_cross"]


# ---- features -----------------------------------------------------------------------------------------
def test_feature_pipeline_problems_all_at_once():
    problems = _problems({"features": {"groups": ["returns", "returns", "volatilty"],
                                       "overrides": {"regime": {"rank_years": 1}, "returns": {"horizonz": [1]},
                                                     "nope": {}},
                                       "warmup": -3}})
    text = "\n".join(problems)
    assert "duplicate groups ['returns']" in text
    assert "unknown feature group 'volatilty' (did you mean 'volatility'?)" in text
    assert "features.overrides.regime: group 'regime' is not in features.groups" in text
    assert "features.overrides.returns.horizonz" in text and "did you mean 'horizons'" in text
    assert "features.overrides.nope: unknown feature group" in text
    assert "features.warmup must be >= 0" in text
    assert len(problems) == 6


def test_valid_feature_overrides_pass():
    cfg = load_config({"features": {"groups": None, "overrides": {"regime": {"rank_years": 0.5}}, "warmup": 0}},
                      env={}, strict=True)
    assert cfg.features.build().overrides == {"regime": {"rank_years": 0.5}}


# ---- instrument ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("field, value, needle", [
    ("triple_swap_weekday", 5, "0..4"), ("triple_swap_weekday", -1, "0..4"), ("rollover_hour_utc", 24, "0..23"),
    ("margin_rate", 1.5, "(0, 1]"), ("margin_rate", 0.0, "positive"), ("tick_size", 0.0, "positive"),
    ("min_lot", 60.0, "must not exceed instrument.max_lot"), ("commission_per_lot", -1.0, ">= 0"),
    ("contract_size", -100.0, "positive"), ("symbol", " ", "non-empty"),
])
def test_instrument_ranges(field, value, needle):
    problems = _problems({"instrument": {field: value}})
    assert any(f"instrument.{field}" in p and needle in p for p in problems), problems


@pytest.mark.parametrize("weekday", [0, 4])
def test_instrument_boundaries_accepted(weekday):
    cfg = load_config({"instrument": {"triple_swap_weekday": weekday, "rollover_hour_utc": 0}}, env={}, strict=True)
    assert cfg.instrument.build().triple_swap_weekday == weekday


# ---- live runner settings -------------------------------------------------------------------------------
def test_live_options_unknown_keys_and_runner_value_checks():
    problems = _problems({"live": {"options": {"on_eror": "hold", "spread_source": "tick"}}})
    text = "\n".join(problems)
    assert "live.options.on_eror: unknown live-runner setting (did you mean 'on_error'?)" in text
    assert "valid keys:" in text and "'allow_live_real'" not in text.split("valid keys:")[1]
    assert "live (runner settings): spread_source must be 'bar' or 'quote'" in text
    ok = load_config({"live": {"options": {"on_error": "flatten", "paper": {"warmup_bars": 10}}}}, env={}, strict=True)
    assert ok.live_runner_mapping()["on_error"] == "flatten"


def test_runner_value_checks_are_strict_problems_and_lenient_warnings():
    """A value only the runner's LiveConfig refuses (here an OMS switch the runner owns) is an
    error for the CLI and `config validate`; a lenient library load reports it as a warning,
    and the runner still refuses the mapping when it starts (the guard is never weakened)."""
    data = {"live": {"options": {"oms": {"dry_run": False}}}}
    problems = _problems(data, strict=True)
    assert len(problems) == 1 and problems[0].startswith("live (runner settings): oms section may not set")
    cfg = load_config(data, env={})
    assert any("oms section" in w and "refuses" in w for w in cfg.check().warnings)
    runner = pytest.importorskip("aurum.live.runner")
    with pytest.raises(ValueError, match="oms section"):
        runner.LiveConfig.from_mapping(cfg.live_runner_mapping())


def test_live_use_desk_requires_agents_enabled():
    problems = _problems({"live": {"use_desk": True}})
    assert any("live.use_desk" in p and "agents.enabled" in p and "desk_overlay.yaml" in p for p in problems)
    ok = load_config({"live": {"use_desk": True}, "agents": {"enabled": True}}, env={}, strict=True)
    assert ok.live_runner_mapping()["desk"]["enabled"] is True


def test_live_check_is_skipped_without_the_runner(monkeypatch):
    monkeypatch.setitem(sys.modules, "aurum.live.runner", None)
    cfg = load_config({"live": {"options": {"on_eror": "hold"}}}, env={}, strict=True)   # cannot be checked
    rep = cfg.check(strict=True)
    assert rep.ok and any("aurum.live.runner" in s for s in rep.skipped)
    stub = types.ModuleType("aurum.live.runner")                    # a stub without LiveConfig
    monkeypatch.setitem(sys.modules, "aurum.live.runner", stub)
    assert any("LiveConfig.from_mapping" in s for s in cfg.check().skipped)


# ---- rl.params --------------------------------------------------------------------------------------------
def test_rl_params_go_through_rl_config_from_aurum(monkeypatch):
    calls = []
    fake = types.ModuleType("aurum.rl.train")

    def rl_config_from_aurum(cfg):
        calls.append(cfg)
        if cfg.rl.params.get("bad"):
            raise ValueError("unknown RLTrainConfig keys: ['bad']")
        return object()

    fake.rl_config_from_aurum = rl_config_from_aurum
    monkeypatch.setitem(sys.modules, "aurum.rl.train", fake)
    cfg = load_config({"rl": {"params": {"n_envs": 2}}}, env={}, strict=True)
    assert calls and calls[-1] is cfg
    assert "rl.params + env (rl_config_from_aurum)" in cfg.check().checked
    assert _problems({"rl": {"params": {"bad": 1}}}) == ["rl.params: unknown RLTrainConfig keys: ['bad']"]

    def needs_torch(cfg):
        raise ImportError("No module named 'torch'")

    fake.rl_config_from_aurum = needs_torch
    assert any("torch" in s for s in load_config(None, env={}).check().skipped)
    monkeypatch.setitem(sys.modules, "aurum.rl.train", None)        # RL stack not installed at all
    rep = load_config({"rl": {"params": {"bad": 1}}}, env={}).check()
    assert rep.ok and any("aurum[rl]" in s for s in rep.skipped)


def test_real_rl_mapping_and_research_only_sizer(caplog):
    rl_train = pytest.importorskip("aurum.rl.train")
    if not hasattr(rl_train, "rl_config_from_aurum"):
        pytest.skip("aurum.rl.train.rl_config_from_aurum not available")
    with pytest.raises(ConfigError, match=r"rl\.params.*total_timestep"):
        load_config({"rl": {"params": {"total_timestep": 10}}}, env={})
    # a research-only sizer must not make the config unloadable (only rl.params are checked then)
    ff = load_config({"sizing": {"method": "fixed_fractional"}, "rl": {"params": {"n_envs": 2}}}, env={}, strict=True)
    assert any("env mapping not checked" in c for c in ff.check().checked)
    # build-time WARNINGs of rl_config_from_aurum belong to `aurum rl train`, not to every load
    with caplog.at_level(logging.WARNING):
        load_config({"backtest": {"stop_atr_mult": 2.0}, "rl": {"params": {"n_envs": 2}}}, env={})
    assert "protective stops" not in caplog.text


# ---- side effects ----------------------------------------------------------------------------------------
def test_validation_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load((REPO / "configs" / "live_paper.yaml").read_text())
    data["extends"] = str(REPO / "configs" / "default.yaml")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    before = sorted(x.relative_to(tmp_path) for x in tmp_path.rglob("*"))
    cfg = load_config(p, env={}, strict=True)
    cfg.check(strict=True)
    assert sorted(x.relative_to(tmp_path) for x in tmp_path.rglob("*")) == before


def test_missing_optional_package_is_named_with_its_extra(monkeypatch, tmp_path):
    """Without the RL stack an ENABLED rl_ppo entry is a problem that says what to install; a
    DISABLED one is listed as skipped (it cannot fail a run)."""
    monkeypatch.setitem(sys.modules, "gymnasium", None)          # import gymnasium -> ModuleNotFoundError
    for mod in [m for m in sys.modules if m == "aurum.rl" or m.startswith("aurum.rl.")]:
        monkeypatch.delitem(sys.modules, mod)                    # re-imported (and refused) on first use
    rl = {"name": "rl_ppo", "params": {"artifact_dir": str(tmp_path / "policy")}}   # loading it needs the RL stack
    rep = load_config({"strategies": [rl, {"name": "tsmom"}]}, env={}, validate=False).check(strict=True)
    assert any("strategies[0] (rl_ppo): needs the optional package 'gymnasium'" in p and "aurum[rl]" in p
               for p in rep.problems), rep.problems
    rep2 = load_config({"strategies": [{**rl, "enabled": False}, {"name": "tsmom"}]}, env={},
                       validate=False).check(strict=True)
    assert not [p for p in rep2.problems if "rl_ppo" in p], rep2.problems
    assert any("rl_ppo, disabled" in x and "aurum[rl]" in x for x in rep2.skipped), rep2.skipped
