"""Configuration tree: loading, inheritance, overrides, validation errors, secrets, hashing."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

from aurum.core.config import (
    RESEARCH_RISK_DEFAULTS,
    SECRET_ENV,
    AurumConfig,
    ConfigError,
    Secret,
    apply_overrides,
    config_from_dict,
    duration_to_bars,
    load_config,
    parse_duration,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))


def _write(tmp_path: Path, name: str, data: dict | str) -> Path:
    p = tmp_path / name
    p.write_text(data if isinstance(data, str) else yaml.safe_dump(data))
    return p


# ---- shipped configs ----------------------------------------------------------------------
@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_load_and_validate(path):
    cfg = load_config(path, env={})
    assert isinstance(cfg, AurumConfig)
    assert cfg.source == str(path)
    assert len(cfg.config_hash()) == 64


def test_shipped_config_expectations():
    names = {p.name for p in CONFIGS}
    assert {"default.yaml", "fast.yaml", "live_paper.yaml", "desk_overlay.yaml"} <= names
    d = load_config(REPO / "configs" / "default.yaml", env={})
    assert d.data.timeframe == "H1" and not d.agents.enabled
    assert len(d.enabled_strategies()) >= 12
    lim = d.risk.limits("research")
    assert lim.daily_loss_persistent is False and lim.max_drawdown is None
    live = d.risk.limits("live")
    assert live.daily_loss_persistent is True and live.max_drawdown == 0.20
    f = load_config(REPO / "configs" / "fast.yaml", env={})
    assert 3 <= len(f.enabled_strategies()) < len(d.enabled_strategies())
    assert f.walkforward.train == "2Y" and f.sizing.target_vol == d.sizing.target_vol  # inherited
    lp = load_config(REPO / "configs" / "live_paper.yaml", env={})
    assert lp.live.broker == "paper" and lp.live.dry_run and not lp.live.allow_live_real
    dk = load_config(REPO / "configs" / "desk_overlay.yaml", env={})
    assert dk.agents.enabled and dk.agents.policy.mode == "overlay"
    desk = dk.agents.desk_config()
    assert desk.chief.effort == "high" and desk.max_cost_usd_per_cycle == 2.0


# ---- defaults & round trip ------------------------------------------------------------------
def test_defaults_are_valid_and_round_trip():
    cfg = load_config(None, env={})
    again = config_from_dict(cfg.to_dict(), env={})
    assert again.to_dict() == cfg.to_dict()
    assert again.config_hash() == cfg.config_hash()
    y = yaml.safe_load(cfg.to_yaml())
    assert y["data"]["timeframe"] == "H1"


def test_builders_create_real_objects():
    cfg = load_config({"sizing": {"target_vol": 0.15}, "costs": {"min_spread": 0.2}}, env={})
    sizer = cfg.sizing.build()
    assert sizer.target_vol == 0.15
    assert cfg.costs.build().min_spread == 0.2
    inst = cfg.instrument.build()
    assert inst.contract_size == 100.0
    rm = cfg.risk.build("research", instrument=inst)
    assert rm.limits.daily_loss_persistent is False
    assert cfg.combiner.build().method == "sharpe_shrink"
    ff = load_config({"sizing": {"method": "fixed_fractional", "risk_per_trade": 0.01}}, env={})
    assert type(ff.sizing.build()).__name__ == "FixedFractionalSizer"
    off = load_config({"risk": {"enabled": False}}, env={})
    assert off.risk.build("research") is None


def test_research_limits_layer_over_defaults():
    cfg = load_config({"risk": {"research": {"max_spread": 1.2}}}, env={})
    lim = cfg.risk.limits("research")
    assert lim.max_spread == 1.2
    for k, v in RESEARCH_RISK_DEFAULTS.items():
        assert getattr(lim, k) == v
    assert cfg.risk.limits("live").daily_loss_persistent is True
    with pytest.raises(ValueError):
        cfg.risk.limits("paper")


# ---- extends / overrides ----------------------------------------------------------------------
def test_extends_deep_merges_and_lists_replace(tmp_path):
    _write(tmp_path, "base.yaml", {"name": "base", "sizing": {"target_vol": 0.2, "max_leverage": 1.5},
                                   "strategies": [{"name": "a"}, {"name": "b"}]})
    child = _write(tmp_path, "child.yaml", {"extends": "base.yaml", "sizing": {"target_vol": 0.05},
                                            "strategies": [{"name": "c"}]})
    cfg = load_config(child, env={})
    assert cfg.name == "base"
    assert cfg.sizing.target_vol == 0.05 and cfg.sizing.max_leverage == 1.5
    assert [s.name for s in cfg.strategies] == ["c"]


def test_extends_cycle_detected(tmp_path):
    _write(tmp_path, "a.yaml", {"extends": "b.yaml"})
    _write(tmp_path, "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml", env={})


def test_cli_style_overrides():
    raw = apply_overrides({"walkforward": {"train": "3Y"}},
                          ["walkforward.train=500", "walkforward.anchored=true", "strategies=[{name: x}]",
                           "data.synthetic.n=300"])
    assert raw["walkforward"] == {"train": 500, "anchored": True}
    assert raw["strategies"] == [{"name": "x"}]
    assert raw["data"]["synthetic"]["n"] == 300
    cfg = load_config(None, overrides=["seed=11", "combiner.method=hrp"], env={})
    assert cfg.seed == 11 and cfg.combiner.method == "hrp"
    with pytest.raises(ConfigError, match="key.path=value"):
        apply_overrides({}, ["novalue"])
    with pytest.raises(ConfigError, match="secrets cannot be overridden"):
        apply_overrides({}, ["agents.api_key=sk-123"])


# ---- validation errors -------------------------------------------------------------------------
def test_unknown_key_suggests_correction():
    with pytest.raises(ConfigError) as ei:
        load_config({"walkforwrd": {}}, env={})
    assert "did you mean 'walkforward'" in str(ei.value)
    with pytest.raises(ConfigError) as ei:
        load_config({"sizing": {"target_volatility": 0.1}}, env={})
    assert "sizing.target_volatility" in str(ei.value) and "target_vol" in str(ei.value)


def test_type_errors_are_reported_with_paths():
    with pytest.raises(ConfigError) as ei:
        load_config({"seed": "seven", "walkforward": {"anchored": "yes please"},
                     "backtest": {"initial_equity": "lots"}}, env={})
    msg = str(ei.value)
    assert "seed: expected an integer" in msg
    assert "walkforward.anchored: expected true/false" in msg
    assert "backtest.initial_equity: expected a number" in msg
    assert len(ei.value.problems) == 3


def test_semantic_errors_collected_at_once():
    bad = {
        "data": {"timeframe": "H7", "start": "2020-01-01", "end": "2019-01-01", "events": "maybe"},
        "sizing": {"target_vol": -1},
        "risk": {"research": {"max_daily_loss": 2.0}, "live": {"bogus_limit": 1}},
        "walkforward": {"train": "3 parsecs", "pbo_splits": 5, "executor": "gpu"},
        "combiner": {"method": "magic"},
        "agents": {"policy": {"mode": "yolo"}, "desk": {"max_parallel_agents": 0}},
        "live": {"broker": "ib", "allow_live_real": True},
    }
    with pytest.raises(ConfigError) as ei:
        load_config(bad, env={})
    msg = str(ei.value)
    for needle in ("data.timeframe", "data.start", "data.events", "sizing", "risk.research",
                   "risk.live", "valid keys", "walkforward.train", "walkforward.pbo_splits",
                   "walkforward.executor", "combiner.method", "agents.policy", "agents.desk", "live.broker"):
        assert needle in msg, needle
    assert len(ei.value.problems) >= 12


def test_strategy_config_rules():
    with pytest.raises(ConfigError, match="duplicate ids"):
        load_config({"strategies": [{"name": "tsmom"}, {"name": "tsmom"}]}, env={})
    ok = load_config({"strategies": [{"name": "tsmom"}, {"name": "tsmom", "id": "tsmom_fast",
                                                         "params": {"horizons": [24, 96]}}]}, env={})
    assert [s.key for s in ok.strategies] == ["tsmom", "tsmom_fast"]
    with pytest.raises(ConfigError, match="only used with combiner.method='fixed'"):
        load_config({"strategies": [{"name": "a", "weight": 2.0}]}, env={})
    fixed = load_config({"combiner": {"method": "fixed"},
                         "strategies": [{"name": "a", "weight": 2.0}, {"name": "b"}]}, env={})
    assert fixed.combiner.method == "fixed"
    with pytest.raises(ConfigError, match="required key missing"):
        load_config({"strategies": [{"params": {}}]}, env={})
    only = ok.enabled_strategies(["tsmom_fast"])
    assert [s.key for s in only] == ["tsmom_fast"]
    with pytest.raises(ConfigError, match="not in config"):
        ok.enabled_strategies(["nope"])


def test_build_strategies_unknown_name_is_clear():
    cfg = load_config({"strategies": [{"name": "definitely_not_a_strategy"}]}, env={})
    with pytest.raises(ConfigError, match="unknown strategy 'definitely_not_a_strategy'"):
        cfg.build_strategies()


def test_yaml_dates_and_numeric_strings_are_accepted(tmp_path):
    p = _write(tmp_path, "c.yaml", "data:\n  start: 2020-01-01\n  end: 2021-06-30\n"
                                   "sizing:\n  min_vol: 1e-2\n")
    cfg = load_config(p, env={})
    assert cfg.data.start == "2020-01-01" and cfg.data.end == "2021-06-30"
    assert cfg.sizing.min_vol == 0.01


def test_invalid_yaml_and_missing_file(tmp_path):
    p = _write(tmp_path, "bad.yaml", "a: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p, env={})
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", env={})
    _write(tmp_path, "list.yaml", "- 1\n- 2\n")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(tmp_path / "list.yaml", env={})


# ---- secrets -------------------------------------------------------------------------------------
def test_secrets_come_only_from_environment():
    env = {"ANTHROPIC_API_KEY": "sk-ant-secret-value", "MT5_PASSWORD": "hunter2", "SEED": "99"}
    cfg = load_config(None, env=env)
    assert isinstance(cfg.agents.api_key, Secret)
    assert cfg.agents.api_key.get() == "sk-ant-secret-value"
    assert cfg.live.mt5_password.get() == "hunter2"
    assert cfg.seed == 0  # non-secret env vars never override config
    text = repr(cfg) + cfg.to_yaml() + str(cfg.to_dict(redact=False))
    assert "sk-ant-secret-value" not in text and "hunter2" not in text
    assert "api_key" not in cfg.to_dict()
    assert load_config(None, env={}).config_hash() == cfg.config_hash()  # secrets never hashed
    assert set(SECRET_ENV.values()) == {"ANTHROPIC_API_KEY", "MT5_PASSWORD", "AURUM_ALERT_WEBHOOK_URL",
                                        "AURUM_ARTIFACT_KEY"}


def test_secret_in_yaml_is_rejected(tmp_path):
    p = _write(tmp_path, "s.yaml", {"agents": {"api_key": "sk-ant-oops"}})
    with pytest.raises(ConfigError, match=r"must not be stored in config files; set \$ANTHROPIC_API_KEY"):
        load_config(p, env={})
    ok = _write(tmp_path, "n.yaml", {"agents": {"api_key": None}})
    assert load_config(ok, env={}).agents.api_key is None


# ---- hashing ---------------------------------------------------------------------------------------
def test_hash_ignores_cosmetic_and_execution_fields():
    base = load_config(None, env={})
    cosmetic = load_config({"name": "other", "output": {"dir": "elsewhere", "run_name": "x"},
                            "walkforward": {"n_jobs": 3, "executor": "thread"},
                            "agents": {"journal_dir": "j"}}, env={})
    assert cosmetic.config_hash() == base.config_hash()
    changed = load_config({"costs": {"slippage_fixed": 0.03}}, env={})
    assert changed.config_hash() != base.config_hash()
    reordered = load_config({"sizing": {"max_leverage": 2.0, "target_vol": 0.10}}, env={})
    assert reordered.config_hash() == base.config_hash()


def test_save_writes_secret_free_yaml(tmp_path):
    cfg = load_config({"name": "s"}, env={"ANTHROPIC_API_KEY": "sk-zzz"})
    p = cfg.save(tmp_path / "run" / "config.yaml")
    txt = p.read_text()
    assert "sk-zzz" not in txt
    again = load_config(p, env={})
    assert again.config_hash() == cfg.config_hash()


# ---- durations ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("value, expected", [
    (500, (500.0, "bars")), ("500bars", (500.0, "bars")), ("250b", (250.0, "bars")), ("12", (12.0, "bars")),
    ("3Y", (3 * 365.25, "days")), ("6M", (182.625, "days")), ("2W", (14.0, "days")),
    ("10D", (10.0, "days")), ("12h", (0.5, "days")), ("90min", (90 / 1440, "days")),
    ("1.5y", (1.5 * 365.25, "days")),
])
def test_parse_duration(value, expected):
    amount, kind = parse_duration(value)
    assert kind == expected[1]
    assert amount == pytest.approx(expected[0])


@pytest.mark.parametrize("value", ["3 parsecs", "-2Y", "0", 0, "Y", True, "3X"])
def test_parse_duration_rejects(value):
    with pytest.raises(ValueError):
        parse_duration(value)


def test_duration_to_bars_uses_calendar_density():
    h1_per_day = 23 * 5 / 7  # ~16.4 bars per calendar day for 23x5 H1 gold bars
    assert duration_to_bars("1Y", h1_per_day) == round(365.25 * h1_per_day)
    assert duration_to_bars(777, h1_per_day) == 777
    assert duration_to_bars("1h", 1.0) == 1  # never zero
    with pytest.raises(ValueError):
        duration_to_bars("1Y", 0.0)


def test_data_config_paths_and_synthetic_load():
    cfg = load_config({"data": {"dir": "/x/y", "timeframe": "h4"}}, env={})
    assert cfg.data.resolved_bars_path() == Path("/x/y/xauusd_H4.parquet")
    assert cfg.data.resolved_macro_dir() == Path("/x/y/macro")
    syn = load_config({"data": {"synthetic": {"n": 400, "model": "trend", "seed": 3},
                                "end": "2020-01-20"}}, env={})
    md = syn.data.load()
    assert len(md.bars) > 100 and md.bars.index[-1] <= dt.datetime(2020, 1, 21, tzinfo=dt.timezone.utc)
    assert md.macro and md.events is not None
    none = load_config({"data": {"synthetic": {"n": 200}, "events": "none", "macro": False}}, env={})
    md2 = none.data.load()
    assert md2.events is None and md2.macro == {}


def test_missing_bars_file_message(tmp_path):
    cfg = load_config({"data": {"dir": str(tmp_path)}}, env={})
    with pytest.raises(FileNotFoundError, match="aurum data download"):
        cfg.data.load()


# ---- adversarial review: live translation ---------------------------------------------------------
def test_live_options_are_additive_only():
    cfg = load_config({"live": {"options": {"on_error": "flatten", "paper": {"warmup_bars": 10},
                                            "desk": {"on_error": "hold"}}}}, env={})
    m = cfg.live_runner_mapping()
    assert m["on_error"] == "flatten" and m["paper"] == {"warmup_bars": 10}
    assert m["desk"]["on_error"] == "hold" and m["desk"]["mode"] == "overlay"   # merged, not replaced
    assert m["allow_live_real"] is False and m["dry_run"] is True
    for opts, frag in (({"allow_live_real": True}, "live.options.allow_live_real"),
                       ({"dry_run": False}, "live.options.dry_run"),
                       ({"magic": 7}, "live.options.magic"),
                       ({"risk": {"max_drawdown": 0.9}}, "risk.live"),
                       ({"sizer": {"target_vol": 0.5}}, "sizing"),
                       ({"costs": {"min_spread": 0.0}}, "live.options.costs"),
                       ({"desk": {"enabled": True}}, "live.options.desk.enabled"),
                       ({"mt5": {"password": "x"}}, "credentials"),
                       ({"monitor": {"webhook": "https://hooks.example/abc"}}, "credentials")):
        with pytest.raises(ConfigError, match=frag.replace(".", r"\.")):
            load_config({"live": {"options": opts}}, env={})
    ok = load_config({"live": {"options": {"monitor": {"webhook": True, "webhook_style": "slack"}}}}, env={})
    assert ok.live_runner_mapping()["monitor"]["webhook"] is True   # the runner's on/off flag is not a secret


def test_live_magic_must_fit_mt5_and_sizer_must_be_vol_target():
    with pytest.raises(ConfigError, match="32-bit"):
        load_config({"live": {"magic": 2**31}}, env={})
    cfg = load_config({"sizing": {"method": "fixed_fractional"}}, env={})   # fine for research ...
    with pytest.raises(ConfigError, match="vol_target"):                     # ... but not tradable live
        cfg.live_runner_mapping()
    with pytest.raises(ConfigError, match="vol_target"):
        cfg.live_sizer_kwargs()
