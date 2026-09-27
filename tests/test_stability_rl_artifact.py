"""``rl_ppo`` artifact portability: a pickled strategy (the live trading artifact's
``strategies.pkl``) embeds its policy, so it keeps working after the live artifact directory
is copied elsewhere and the RL training directory is gone."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.strategies.base import get_strategy
from aurum.strategies.rl import RLPolicyStrategy

pytest.importorskip("stable_baselines3")

N = 2600


def _config() -> dict:
    return {
        "feature_groups": ["returns", "volatility", "session"],
        "env": {"window": 2, "episode_length": 256},
        "total_timesteps": 512, "n_envs": 1, "n_steps": 256, "batch_size": 128,
        "n_epochs": 1, "eval_freq": 512, "net_arch": [16], "seed": 3,
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[RLPolicyStrategy, MarketData, pd.Series, Path]:
    bars = make_synthetic_bars(N, "H1", seed=4, model="trend", regime_params={"phi": 0.25})
    md = MarketData(bars=bars)
    rl_out = tmp_path_factory.mktemp("rl_training_runs")
    s = get_strategy("rl_ppo", config=_config(), val_fraction=0.25, out_dir=str(rl_out))
    s.fit(md)
    return s, md, s.generate(md), rl_out


def test_live_artifact_survives_copy_and_deleted_rl_dir(trained, tmp_path: Path) -> None:
    from aurum.live.runner import load_artifact, save_artifact

    s, md, forecast, rl_out = trained
    assert s.has_embedded_policy
    src = save_artifact(tmp_path / "live" / "artifact", strategies={"rl": s}, timeframe="H1")
    moved = tmp_path / "elsewhere" / "deployed"
    shutil.copytree(src, moved)
    shutil.rmtree(src)
    # the RL training directory is gone too (e.g. a temp dir on the research machine)
    backup = tmp_path / "rl_backup"
    shutil.move(str(s.artifact_dir_), backup)
    try:
        assert not s.artifact_dir_.exists()
        art = load_artifact(moved)
        strat = art.strategies["rl"]
        assert isinstance(strat, RLPolicyStrategy) and strat._artifact is None
        pd.testing.assert_series_equal(strat.generate(md), forecast)
        assert art.max_lookback >= strat.warmup_bars > 0
        # a live-style sliding window reproduces the research forecast too
        n_hist = 3 * strat.warmup_bars
        live = strat.generate(MarketData(bars=md.bars.iloc[-n_hist:]))
        assert live.iloc[-1] == forecast.iloc[-1]
    finally:
        shutil.move(str(backup), s.artifact_dir_)


def test_pickle_is_self_contained_and_fingerprinted(trained, tmp_path: Path) -> None:
    s, md, forecast, _ = trained
    blob = pickle.dumps(s)
    assert len(blob) < 5_000_000
    restored = pickle.loads(blob)
    assert restored.artifact_fingerprint_ == s.artifact_fingerprint_
    assert set(restored._bundle) >= {"policy.zip", "pipeline.json", "config.json"}
    # identical bytes to the files on disk at attach time
    for name, data in restored._bundle.items():
        assert data == (s.artifact_dir_ / name).read_bytes(), name
    moved = tmp_path / "moved_rl"
    shutil.copytree(s.artifact_dir_, moved)
    reloaded = RLPolicyStrategy.from_artifact(moved)
    assert reloaded.artifact_fingerprint_ == s.artifact_fingerprint_
    pd.testing.assert_series_equal(reloaded.generate(md), forecast)
    broken = pickle.loads(blob)
    broken._bundle["config.json"] = broken._bundle["config.json"].replace(b'"seed": 3', b'"seed": 4')
    with pytest.raises(RuntimeError, match="fingerprint"):
        broken.generate(md)


def test_legacy_pickle_without_embedded_policy_uses_the_directory(trained) -> None:
    """Strategies pickled before embedding existed carry no bytes: they still load from the
    directory (fingerprint-checked) when it exists."""
    s, md, forecast, _ = trained
    state = s.__getstate__()
    state.pop("_bundle")
    legacy = RLPolicyStrategy.__new__(RLPolicyStrategy)
    legacy.__setstate__(dict(state))
    assert not legacy.has_embedded_policy
    pd.testing.assert_series_equal(legacy.generate(md), forecast)
