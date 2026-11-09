"""v2.1 core fixes - RL configuration and parameter validation:
``rl_config_from_aurum`` (top-level costs/financing, sizing, research risk, instrument and
equity -> ``EnvConfig``; ``rl.params`` / ``rl.params.env`` override), ``EnvConfig.instrument``
(env and the RL backtest trade the configured contract, in parity), and unknown-parameter
rejection for ``ml_gbm`` / ``meta_label`` / ``rl_ppo``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import RESEARCH_RISK_DEFAULTS, load_config
from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.strategies.base import get_strategy


def _cfg(**raw):
    return load_config(raw, env={})


# ---- unknown parameters ------------------------------------------------------------------------------
@pytest.mark.parametrize("name,bad", [
    ("ml_gbm", {"max_holding_bar": 12}),
    ("ml_gbm", {"skill_gate": None}),
    ("meta_label", {"primary_param": {}}),
    ("meta_label", {"p_treshold": 0.6}),
    ("rl_ppo", {"val_frac": 0.2}),
    ("rl_ppo", {"artifact": "x"}),
])
def test_ml_and_rl_strategies_reject_misspelt_params(name, bad):
    with pytest.raises(ValueError, match="unknown parameter"):
        get_strategy(name, **bad)


def test_valid_params_still_accepted():
    s = get_strategy("ml_gbm", max_holding_bars=12, skill_gate_z=None, target="fixed_horizon")
    assert s.params["max_holding_bars"] == 12
    m = get_strategy("meta_label", primary="tsmom", primary_params={"horizons": (24, 96)}, p_threshold=0.6)
    assert m.params["p_threshold"] == 0.6
    # the transactional refit re-instantiates with the (valid) params
    from aurum.strategies.ml import MLGBMStrategy

    assert type(s)(**s.params).params == s.params and isinstance(s, MLGBMStrategy)


def test_config_build_rejects_misspelt_strategy_params():
    # rejected when the config is validated or, at the latest, when the strategy is built
    with pytest.raises(Exception, match="max_holding_bar"):
        _cfg(strategies=[{"name": "ml_gbm", "params": {"max_holding_bar": 12}}]).build_strategies()


def test_rl_ppo_retrain_resolution_and_pin_validation():
    s = get_strategy("rl_ppo")
    assert s.params["retrain"] is True and not s.is_fitted     # no artifact: trains in fit
    with pytest.raises(ValueError, match="retrain=False needs"):
        get_strategy("rl_ppo", retrain=False)
    with pytest.raises(ValueError, match="retrain"):
        get_strategy("rl_ppo", retrain="no")
    with pytest.raises(ValueError, match="artifact_sha256"):
        get_strategy("rl_ppo", artifact_sha256="xyz")
    with pytest.raises(FileNotFoundError):
        get_strategy("rl_ppo", artifact_dir="/nonexistent/rl", artifact_sha256="abcdef12")


# ---- rl_config_from_aurum ------------------------------------------------------------------------
pytest.importorskip("gymnasium")

from aurum.rl.env import EnvConfig, GoldTradingEnv, rollout  # noqa: E402
from aurum.rl.train import (  # noqa: E402
    AURUM_MAPPED_ENV_KEYS,
    RLTrainConfig,
    backtest_forecast,
    rl_config_from_aurum,
)


def test_mapping_from_top_level_sections():
    cfg = _cfg(seed=7,
               costs={"spread_multiplier": 1.5, "min_spread": 0.2, "commission_per_lot": 3.5,
                      "financing": {"mode": "rate", "markup_long": 0.03, "fallback_rate": 0.04}},
               sizing={"target_vol": 0.12, "max_leverage": 3.0, "rebalance_band": 0.05, "max_lots": 20.0},
               risk={"research": {"max_daily_loss": 0.05, "max_spread": 1.5}},
               instrument={"contract_size": 50.0, "commission_per_lot": 1.0},
               backtest={"initial_equity": 250_000.0})
    rc = rl_config_from_aurum(cfg)
    assert isinstance(rc, RLTrainConfig) and rc.seed == 7
    env = rc.env
    assert env.costs["spread_multiplier"] == 1.5 and env.costs["commission_per_lot"] == 3.5
    assert env.costs["financing"]["markup_long"] == 0.03 and env.costs["financing"]["fallback_rate"] == 0.04
    assert env.sizer["target_vol"] == 0.12 and env.sizer["max_leverage"] == 3.0
    assert env.sizer["rebalance_band"] == 0.05 and env.sizer["max_lots"] == 20.0
    assert env.sizer["drawdown_derisk"] == [[0.10, 0.5], [0.15, 0.25]]
    assert env.risk == {**RESEARCH_RISK_DEFAULTS, "max_daily_loss": 0.05, "max_spread": 1.5}
    assert env.instrument["contract_size"] == 50.0 and env.build_instrument().contract_size == 50.0
    assert env.initial_equity == 250_000.0
    # JSON round trip (what config.json stores)
    back = RLTrainConfig.from_dict(rc.to_dict())
    assert back.env.to_dict() == env.to_dict()
    assert set(AURUM_MAPPED_ENV_KEYS) == {"sizer", "costs", "risk", "instrument", "initial_equity"}


def test_defaults_match_the_research_backtest_objects():
    cfg = _cfg()
    env = rl_config_from_aurum(cfg).env
    assert env.build_instrument() == cfg.instrument.build() == XAUUSD
    from aurum.execution.costs import CostModel
    from aurum.portfolio.sizing import VolTargetSizer
    from aurum.risk.manager import RiskLimits

    assert CostModel(**env.costs) == cfg.costs.build()
    assert repr(VolTargetSizer(**env.sizer)) == repr(cfg.sizing.build())
    assert RiskLimits(**env.risk) == cfg.risk.limits("research")


def test_rl_params_and_explicit_env_keys_override():
    cfg = _cfg(seed=3, sizing={"target_vol": 0.2},
               rl={"params": {"total_timesteps": 5000, "seed": 9, "feature_groups": ["returns"],
                              "env": {"window": 3, "sizer": {"target_vol": 0.05},
                                      "risk": {"max_drawdown": 0.25}}}})
    rc = rl_config_from_aurum(cfg)
    assert rc.total_timesteps == 5000 and rc.seed == 9 and rc.feature_groups == ("returns",)
    assert rc.env.window == 3
    assert rc.env.sizer == {"target_vol": 0.05}                 # replaced, not merged
    assert rc.env.risk == {"max_drawdown": 0.25}
    assert rc.env.costs == rl_config_from_aurum(_cfg()).env.costs   # unmapped keys still mapped


def test_disabled_risk_fixed_fractional_and_unknown_keys(caplog):
    assert rl_config_from_aurum(_cfg(risk={"enabled": False})).env.risk is None
    ff = _cfg(sizing={"method": "fixed_fractional"})
    with pytest.raises(ValueError, match="VolTargetSizer"):
        rl_config_from_aurum(ff)
    ff.rl.params = {"env": {"sizer": {"target_vol": 0.1}}}
    assert rl_config_from_aurum(ff).env.sizer == {"target_vol": 0.1}
    bad = _cfg()
    bad.rl.params = {"total_timestep": 1}
    with pytest.raises(ValueError, match="unknown RLTrainConfig"):
        rl_config_from_aurum(bad)
    bad.rl.params = {"env": {"windw": 2}}
    with pytest.raises(ValueError, match="unknown EnvConfig"):
        rl_config_from_aurum(bad)
    bad.rl.params = {"env": [1, 2]}
    with pytest.raises(ValueError, match="mapping"):
        rl_config_from_aurum(bad)
    caplog.set_level(logging.WARNING, logger="aurum.rl.train")
    rl_config_from_aurum(_cfg(backtest={"stop_atr_mult": 2.0}))
    assert any("protective stops" in r.getMessage() for r in caplog.records)


# ---- EnvConfig.instrument --------------------------------------------------------------------------
def test_env_config_instrument_validation_and_legacy_dicts():
    assert EnvConfig().instrument is None and EnvConfig().build_instrument() is XAUUSD
    legacy = EnvConfig().to_dict()
    legacy.pop("instrument")                                     # config.json of older artifacts
    assert EnvConfig.from_dict(legacy).build_instrument() is XAUUSD
    inst = Instrument(contract_size=10.0)
    e = EnvConfig(instrument=inst)
    assert e.instrument["contract_size"] == 10.0 and e.build_instrument() == inst
    with pytest.raises(ValueError, match="unknown Instrument"):
        EnvConfig(instrument={"contract_sise": 10})
    with pytest.raises(ValueError, match="mapping"):
        EnvConfig(instrument=5)


def test_env_trades_the_configured_instrument_in_parity_with_the_rl_backtest():
    from aurum.features.pipeline import FeaturePipeline

    bars = make_synthetic_bars(1600, "H1", seed=3, model="regime")
    md = MarketData(bars=bars)
    pipe = FeaturePipeline(groups=["returns", "volatility"])
    raw = pipe.compute(md)
    pipe.fit(raw.iloc[pipe.max_lookback:1000])
    feats = pipe.transform(raw)
    base = EnvConfig(window=2, costs={"financing": {"mode": "fixed"}})
    small = EnvConfig(window=2, costs={"financing": {"mode": "fixed"}},
                      instrument={"contract_size": 10.0, "lot_step": 0.01, "min_lot": 0.01,
                                  "commission_per_lot": 2.0})
    rng = np.random.default_rng(5)
    actions = rng.integers(0, 5, size=len(bars))

    def run(cfg: EnvConfig):
        env = GoldTradingEnv(bars, feats, cfg, start=1000, random_start=False)
        it = iter(actions)
        return env, rollout(env, lambda obs: int(next(it)))

    env_b, rb = run(base)
    env_s, rs = run(small)
    assert env_s.instrument.contract_size == 10.0 and env_b.instrument is XAUUSD
    lots_b = rb.result.positions.abs().max()
    lots_s = rs.result.positions.abs().max()
    assert lots_s == pytest.approx(10 * lots_b, rel=0.05)        # 1/10 the contract -> ~10x the lots
    # the RL backtest (engine) trades the same contract: equity identical to the rollout
    bt = backtest_forecast(md, rs.forecast, small, vol=pd.Series(env_s._vol, index=bars.index),
                           start=rs.start, end=rs.end)
    np.testing.assert_allclose(bt.equity.to_numpy(), rs.result.equity.to_numpy(), rtol=0, atol=1e-8)
    assert bt.costs["commission"].sum() > 0
