"""aurum.rl.train: config round trip, train-only scaler fit, smoke PPO training with
validation selection, artifact save/load, determinism and early stopping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from aurum.core.types import MarketData  # noqa: E402
from aurum.data.synthetic import make_synthetic_bars  # noqa: E402
from aurum.rl.env import EnvConfig  # noqa: E402
from aurum.rl.train import (  # noqa: E402
    ARTIFACT_FILES,
    RLTrainConfig,
    evaluate_policy,
    load_artifact,
    make_predictor,
    prepare_data,
    resolve_device,
    rollout_artifact,
    train_ppo,
)

N_TRAIN, N_VAL = 2200, 700


def _bars(seed: int = 5) -> pd.DataFrame:
    # Strongly autocorrelated returns: a learnable edge so the smoke run has signal.
    return make_synthetic_bars(N_TRAIN + N_VAL, "H1", seed=seed, model="trend",
                               regime_params={"phi": 0.3})


def _split(bars: pd.DataFrame) -> tuple[MarketData, MarketData]:
    return MarketData(bars=bars.iloc[:N_TRAIN]), MarketData(bars=bars.iloc[N_TRAIN:])


def _tiny_config(**kw) -> RLTrainConfig:
    base = dict(feature_groups=("returns", "volatility"),
                env=EnvConfig(window=2, episode_length=256), total_timesteps=2048, n_envs=2,
                n_steps=256, batch_size=128, n_epochs=2, eval_freq=1024, net_arch=(32, 32),
                seed=0)
    base.update(kw)
    return RLTrainConfig(**base)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    bars = _bars()
    md_tr, md_va = _split(bars)
    out = tmp_path_factory.mktemp("rl_art")
    res = train_ppo(md_tr, md_va, config=_tiny_config(), out_dir=out)
    return bars, res


# --------------------------------------------------------------------------------------------
def test_config_json_round_trip() -> None:
    cfg = RLTrainConfig(env=EnvConfig(window=3, dd_penalty=0.5, risk=None), net_arch=(64, 32),
                        feature_groups=("returns",))
    d = cfg.to_dict()
    back = RLTrainConfig.from_dict(json.loads(json.dumps(d)))
    assert back == cfg
    assert back.env.action_levels == (-1.0, -0.5, 0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="unknown"):
        RLTrainConfig.from_dict({**d, "bogus": 1})
    with pytest.raises(ValueError):
        RLTrainConfig(batch_size=10_000)


def test_resolve_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda", "mps")


def test_scaler_is_fitted_on_training_rows_only() -> None:
    bars = _bars()
    md_tr, md_va = _split(bars)
    cfg = _tiny_config()
    a = prepare_data(md_tr, md_va, cfg)
    # completely different validation data -> identical scaler statistics and train features
    alt = make_synthetic_bars(N_TRAIN + N_VAL, "H1", seed=99, model="jump", annual_vol=0.5,
                              start_price=900.0)
    alt_val = alt.iloc[N_TRAIN:]
    b = prepare_data(md_tr, MarketData(bars=alt_val), cfg)
    pd.testing.assert_frame_equal(a.pipeline.stats, b.pipeline.stats)
    pd.testing.assert_frame_equal(a.features.iloc[:N_TRAIN], b.features.iloc[:N_TRAIN])
    assert a.n_train == N_TRAIN and a.val_range == (N_TRAIN, N_TRAIN + N_VAL - 1)


def test_validation_must_follow_training() -> None:
    bars = _bars()
    with pytest.raises(ValueError, match="time-ordered"):
        prepare_data(MarketData(bars=bars.iloc[:N_TRAIN]), MarketData(bars=bars.iloc[N_TRAIN - 5:]),
                     _tiny_config())


def test_smoke_training_writes_a_loadable_artifact(trained) -> None:
    bars, res = trained
    out = Path(res.artifact_dir)
    for f in ARTIFACT_FILES + ("history.csv",):
        assert (out / f).exists(), f
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_evals"] == len(res.history) == 2
    assert metrics["timesteps_trained"] == res.timesteps == 2048
    assert metrics["selected_timesteps"] in (1024, 2048)
    assert set(metrics["val"]) >= {"sharpe", "total_return", "max_drawdown", "n_trades"}
    assert metrics["data"]["n_train_bars"] == N_TRAIN
    dh = json.loads((out / "data_hash.json").read_text())
    assert len(dh["train_bars_sha256"]) == 64 and dh["train_bars_sha256"] != dh["val_bars_sha256"]
    # validation backtest (engine) reproduces the rollout's simulated account to the cent
    assert (res.history["engine_rollout_max_abs_diff"] <= 1e-6).all()
    assert res.val_metrics["engine_rollout_max_abs_diff"] <= 1e-6
    # the selected checkpoint is the best one in the history
    best = res.history.loc[res.history["sharpe"].idxmax()]
    assert int(best["timesteps"]) == res.selected_timesteps
    assert res.val_metrics["sharpe"] == pytest.approx(best["sharpe"])
    # on this strongly autocorrelated series even a tiny run should trade and make money
    assert res.val_metrics["n_trades"] > 0

    art = load_artifact(out)
    assert art.config == res.config
    obs = np.random.default_rng(0).normal(size=(20, art.model.observation_space.shape[0]))
    ref = make_predictor(res.model)
    for o in obs.astype(np.float32):
        assert art.predict(o) == ref(o) == res.model.predict(o, deterministic=True)[0]
    # rolling the loaded artifact over train+val with the validated (deployment) convention
    # reproduces the validation forecasts and simulated account
    ro = rollout_artifact(art, MarketData(bars=bars), start=res.val_eval.rollout.start,
                          kill_switches=False, episode_anchor=art.config.env.episode_anchor)
    pd.testing.assert_series_equal(ro.forecast, res.val_eval.rollout.forecast)
    np.testing.assert_allclose(ro.result.equity.to_numpy(),
                               res.val_eval.rollout.result.equity.to_numpy(), atol=1e-9)


def test_evaluate_policy_uses_the_engine(trained) -> None:
    _, res = trained
    ev = evaluate_policy(make_predictor(res.model), res.data, res.config.env)
    assert ev.backtest.meta["engine"] == "run_backtest"
    assert ev.backtest.equity.index[0] == res.data.md.bars.index[N_TRAIN]
    assert ev.metrics["sharpe"] == pytest.approx(res.val_metrics["sharpe"])
    assert 0.0 <= ev.metrics["frac_long"] + ev.metrics["frac_short"] <= 1.0


def test_training_is_deterministic(tmp_path: Path) -> None:
    bars = _bars(seed=8)
    md_tr, md_va = _split(bars)
    cfg = _tiny_config(total_timesteps=512, n_steps=128, batch_size=64, eval_freq=256)
    r1 = train_ppo(md_tr, md_va, config=cfg, out_dir=tmp_path / "a")
    r2 = train_ppo(md_tr, md_va, config=cfg, out_dir=tmp_path / "b")
    cols = ["timesteps", "sharpe", "total_return", "n_trades"]
    pd.testing.assert_frame_equal(r1.history[cols], r2.history[cols])
    p1 = r1.model.policy.state_dict()
    p2 = r2.model.policy.state_dict()
    assert all(np.array_equal(p1[k].numpy(), p2[k].numpy()) for k in p1)


def test_early_stopping_on_patience(tmp_path: Path) -> None:
    bars = _bars(seed=9)
    md_tr, md_va = _split(bars)
    # n_days is constant across evaluations -> never improves after the first one
    cfg = _tiny_config(total_timesteps=4096, n_steps=128, batch_size=64, eval_freq=256,
                       patience=1, select_metric="n_days")
    res = train_ppo(md_tr, md_va, config=cfg, out_dir=tmp_path / "es")
    assert res.early_stopped
    assert len(res.history) == 2 and res.timesteps < 4096
    assert res.selected_timesteps == int(res.history["timesteps"].iloc[0])


def test_rollout_kill_switches_toggle(trained) -> None:
    """Evaluation rollouts stop at the simulated kill switch; strategy rollouts
    (kill_switches=False) keep producing forecasts to the end of the data."""
    import dataclasses

    bars, res = trained
    art = load_artifact(res.artifact_dir)
    tight_env = dataclasses.replace(
        art.config.env, risk={"max_drawdown": 0.002, "max_daily_loss": None},
        sizer={"target_vol": 0.5, "max_leverage": 5.0, "drawdown_derisk": None})
    art.config = dataclasses.replace(art.config, env=tight_env)
    md = MarketData(bars=bars)
    hard = rollout_artifact(art, md)
    soft = rollout_artifact(art, md, kill_switches=False)
    assert hard.terminated and hard.end < len(bars) - 1
    assert not soft.terminated and soft.end == len(bars) - 1
    assert (soft.forecast.iloc[hard.end + 1:] != 0).any()
    n = hard.end - hard.start
    np.testing.assert_allclose(hard.result.equity.to_numpy()[:n],
                               soft.result.equity.to_numpy()[:n], atol=1e-9)


def test_continuous_action_training_smoke(tmp_path: Path) -> None:
    bars = _bars(seed=12)
    md_tr, md_va = _split(bars)
    cfg = _tiny_config(env=EnvConfig(window=2, episode_length=128, action_mode="continuous"),
                       total_timesteps=512, n_steps=128, batch_size=64, eval_freq=256)
    res = train_ppo(md_tr, md_va, config=cfg, out_dir=tmp_path / "cont")
    fc = res.val_eval.rollout.forecast.iloc[N_TRAIN:]
    assert fc.between(-1.0, 1.0).all() and fc.nunique() > 5
    art = load_artifact(res.artifact_dir)
    assert art.config.env.action_mode == "continuous"
    o = np.zeros(art.model.observation_space.shape[0], np.float32)
    a = art.predict(o)
    assert a.shape == (1,) and -1.0 <= float(a[0]) <= 1.0


# --------------------------------------------------------------------------------------------
# reviewer: adversarial tests (train/serve state skew, bar-size guard)
# --------------------------------------------------------------------------------------------
def test_strategy_rollout_keeps_agent_state_inside_the_training_distribution(trained) -> None:
    """Regression: kill_switches=False used to DISABLE the drawdown kill, so a long rollout
    drifted into drawdowns the policy never saw in training (training episodes end at the
    kill). Now the kill fires exactly as in training and the simulated account starts a new
    episode in place, so the observed drawdown never exceeds limit + one bar's loss."""
    import dataclasses

    bars, res = trained
    art = load_artifact(res.artifact_dir)
    limit = 0.02
    tight_env = dataclasses.replace(
        art.config.env, risk={"max_drawdown": limit, "max_daily_loss": None},
        sizer={"target_vol": 0.5, "max_leverage": 5.0, "drawdown_derisk": None})
    art.config = dataclasses.replace(art.config, env=tight_env)
    base = art.predict
    seen: list[float] = []

    def spy(obs: np.ndarray):
        seen.append(float(obs[-1]) / 10.0)  # state_drawdown = 10 x drawdown
        return base(obs)

    art.predict = spy  # type: ignore[method-assign]
    ro = rollout_artifact(art, MarketData(bars=bars), kill_switches=False)
    assert not ro.terminated and ro.end == len(bars) - 1 and ro.n_restarts > 0
    eq = ro.result.equity.to_numpy()
    worst_bar_loss = float(np.max(1.0 - eq[1:] / eq[:-1]))
    assert max(seen) <= limit + worst_bar_loss + 1e-9
    # the money path stays continuous (no simulator reset): equity changes = pnl
    pnl = ro.result.pnl["net"].to_numpy()
    np.testing.assert_allclose(np.diff(eq), pnl[1:], atol=1e-6)
    assert ro.info["restart_bars"] and all(ro.start < b <= ro.end for b in ro.info["restart_bars"])


def test_rollout_artifact_rejects_other_bar_size(trained) -> None:
    from aurum.features.multi_timeframe import resample_anchored

    bars, res = trained
    art = load_artifact(res.artifact_dir)
    lookback_before = art.pipeline.max_lookback
    with pytest.raises(ValueError, match="bar"):
        rollout_artifact(art, MarketData(bars=resample_anchored(bars, "H4", 0)))
    assert art.pipeline.max_lookback == lookback_before  # artifact pipeline not mutated


def test_validated_forecasts_are_the_deployed_strategy_forecasts(trained) -> None:
    """Train/serve consistency: the forecasts a checkpoint was SELECTED on (validation) are
    exactly what the rl_ppo strategy emits on those bars from the full history. Before,
    validation used a fresh continuous account at the validation start while the strategy
    used a different account convention, so selection scored a different behaviour."""
    from aurum.strategies.rl import RLPolicyStrategy

    bars, res = trained
    assert res.config.env.episode_anchor == "M"
    s = RLPolicyStrategy.from_artifact(res.artifact_dir)
    f = s.generate(MarketData(bars=bars))
    val = res.val_eval.rollout.forecast.iloc[N_TRAIN:]
    pd.testing.assert_series_equal(f.iloc[N_TRAIN:], val, check_names=False)
    assert (val != 0).any()
    # the official validation backtest starts at the first validation bar
    assert res.val_eval.backtest.equity.index[0] == bars.index[N_TRAIN]
    assert res.val_eval.rollout.start < N_TRAIN  # rolled from one anchor period earlier


def test_evaluate_policy_with_a_continuous_account_matches_the_engine_everywhere(trained) -> None:
    """episode_anchor=None: one continuous account from the validation start; without a kill
    switch the engine replay equals the simulated account on every validation bar."""
    import dataclasses

    _, res = trained
    cfg = dataclasses.replace(res.config.env, episode_anchor=None)
    ev = evaluate_policy(make_predictor(res.model), res.data, cfg)
    assert ev.rollout.start == N_TRAIN and ev.rollout.info["episode_starts"] == []
    if not ev.rollout.info["restart_bars"]:
        np.testing.assert_allclose(ev.backtest.equity.to_numpy(),
                                   ev.rollout.result.equity.to_numpy(), atol=1e-8)
    assert ev.metrics["engine_rollout_max_abs_diff"] <= 1e-6
