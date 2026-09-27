"""rl_ppo strategy adapter (aurum.strategies.rl): torch-free import, registry, fit/load,
valid causal forecast series, clone/pickle behaviour."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.schema import validate_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.strategies.base import get_strategy, list_strategies
from aurum.strategies.rl import RLPolicyStrategy

N = 2600
LEVELS = {-1.0, -0.5, 0.0, 0.5, 1.0}


def test_import_does_not_load_torch_or_sb3() -> None:
    code = (
        "import sys\n"
        "import aurum.strategies.rl\n"
        "from aurum.strategies.base import get_strategy\n"
        "s = get_strategy('rl_ppo')\n"
        "heavy = [m for m in ('torch', 'stable_baselines3', 'gymnasium') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_registered_trainable_and_requires_fit() -> None:
    assert list_strategies()["rl_ppo"] is RLPolicyStrategy
    s = get_strategy("rl_ppo")
    assert s.trainable and not s.is_fitted and s.warmup_bars == 0
    md = MarketData(bars=make_synthetic_bars(300, "H1", seed=0))
    with pytest.raises(RuntimeError, match="not fitted"):
        s.generate(md)
    with pytest.raises(FileNotFoundError):
        RLPolicyStrategy(artifact_dir="/nonexistent/rl_artifact")


# --------------------------------------------------------------------------------------------
pytest.importorskip("stable_baselines3")


def _config() -> dict:
    return {
        "feature_groups": ["returns", "volatility", "session"],
        "env": {"window": 2, "episode_length": 256},
        "total_timesteps": 1024, "n_envs": 2, "n_steps": 256, "batch_size": 128,
        "n_epochs": 2, "eval_freq": 512, "net_arch": [32, 32], "seed": 1,
    }


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> tuple[RLPolicyStrategy, MarketData]:
    bars = make_synthetic_bars(N, "H1", seed=4, model="trend", regime_params={"phi": 0.25})
    md = MarketData(bars=bars)
    out = tmp_path_factory.mktemp("rl_strategy")
    s = get_strategy("rl_ppo", config=_config(), val_fraction=0.25, out_dir=str(out))
    s.fit(md)
    return s, md


def test_fit_then_generate_valid_forecast(fitted) -> None:
    s, md = fitted
    assert s.is_fitted and s.artifact_dir_ is not None
    assert s.train_summary_["n_evals"] == 2
    f = s.generate(md)
    assert f.index.equals(md.bars.index) and f.name == "rl_ppo"
    assert np.isfinite(f.to_numpy()).all()
    assert set(np.unique(f.to_numpy())) <= LEVELS
    assert s.warmup_bars > 0
    assert (f.iloc[: s.warmup_bars] == 0.0).all()
    assert (f.iloc[s.warmup_bars:] != 0.0).any()


def test_load_from_artifact_and_clone_reproduce(fitted) -> None:
    s, md = fitted
    f = s.generate(md)
    loaded = RLPolicyStrategy.from_artifact(s.artifact_dir_)
    assert loaded.is_fitted and loaded.warmup_bars == s.warmup_bars
    pd.testing.assert_series_equal(loaded.generate(md), f)
    clone = s.clone()
    assert clone._artifact is None  # network reloaded lazily, never shared
    pd.testing.assert_series_equal(clone.generate(md), f)
    restored = pickle.loads(pickle.dumps(s))
    pd.testing.assert_series_equal(restored.generate(md), f)
    assert copy.deepcopy(loaded).params["artifact_dir"] == str(Path(s.artifact_dir_))


@pytest.mark.parametrize("cut", [900, 1900])
def test_generate_is_causal(fitted, cut: int) -> None:
    """Forecast[t] must not change when bars after t are replaced or removed."""
    s, md = fitted
    base = s.generate(md)
    bars = md.bars
    alt = make_synthetic_bars(len(bars), "H1", seed=77 + cut, model="jump",
                              start_price=float(bars["close"].iloc[cut]) * 1.4, annual_vol=0.5,
                              spread=0.8)
    pert = pd.concat([bars.iloc[: cut + 1], alt.iloc[cut + 1:]])
    pert.attrs = dict(bars.attrs)
    validate_bars(pert)
    fp = s.generate(MarketData(bars=pert))
    pd.testing.assert_series_equal(fp.iloc[: cut + 1], base.iloc[: cut + 1])
    ft = s.generate(MarketData(bars=bars.iloc[: cut + 1]))
    pd.testing.assert_series_equal(ft, base.iloc[: cut + 1])


def test_short_history_is_flat(fitted) -> None:
    s, md = fitted
    short = MarketData(bars=md.bars.iloc[: s.warmup_bars])
    f = s.generate(short)
    assert (f == 0.0).all() and f.index.equals(short.bars.index)


def test_fit_rejects_tiny_validation_split() -> None:
    s = get_strategy("rl_ppo", config=_config(), val_fraction=0.1, min_val_bars=500)
    with pytest.raises(ValueError, match="validation bars"):
        s.fit(MarketData(bars=make_synthetic_bars(1000, "H1", seed=0)))


# --------------------------------------------------------------------------------------------
# reviewer: adversarial tests (artifact isolation, tamper detection, bar-size guard)
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_folds(tmp_path_factory: pytest.TempPathFactory):
    """Walk-forward style: ONE prototype (one out_dir) cloned and fitted on two folds."""
    bars = make_synthetic_bars(N, "H1", seed=4, model="trend", regime_params={"phi": 0.25})
    md = MarketData(bars=bars)
    out = tmp_path_factory.mktemp("rl_shared_out")
    proto = get_strategy("rl_ppo", config=_config(), val_fraction=0.25, out_dir=str(out))
    s1 = proto.clone().fit(MarketData(bars=bars.iloc[:2000]))  # fold 1: earlier data only
    f1 = s1.generate(md)
    shipped = pickle.dumps(s1)                                  # e.g. sent to a worker process
    s2 = proto.clone().fit(md)                                  # fold 2: includes fold-1's test rows
    return md, s1, f1, shipped, s2


def test_shared_out_dir_never_swaps_a_fitted_policy(two_folds) -> None:
    """Regression: every fit wrote to the SAME out_dir, and a clone/pickle of fold 1 lazily
    reloaded fold 2's policy (trained on fold 1's test period) -> silent look-ahead."""
    md, s1, f1, shipped, s2 = two_folds
    assert s1.artifact_dir_ != s2.artifact_dir_
    assert (s1.artifact_dir_ / "policy.zip").exists() and (s2.artifact_dir_ / "policy.zip").exists()
    restored = pickle.loads(shipped)
    pd.testing.assert_series_equal(restored.generate(md), f1)
    pd.testing.assert_series_equal(s1.clone().generate(md), f1)


def test_modified_artifact_is_refused(two_folds, tmp_path: Path) -> None:
    """A strategy never silently runs a policy other than the one it attached."""
    import shutil

    md, s1, _, _, s2 = two_folds
    art = tmp_path / "art"
    shutil.copytree(s1.artifact_dir_, art)
    s = RLPolicyStrategy.from_artifact(art)
    shutil.copy(s2.artifact_dir_ / "policy.zip", art / "policy.zip")  # overwritten on disk
    with pytest.raises(RuntimeError, match="changed on disk"):
        s.generate(md)


def test_generate_rejects_a_different_bar_size(fitted) -> None:
    """An H1-trained policy must not run on H4 bars (scaler stats, warm-up, vol and the
    policy itself are bar-size specific); previously it only logged a pipeline warning."""
    from aurum.features.multi_timeframe import resample_anchored

    s, md = fitted
    h4 = resample_anchored(md.bars, "H4", 0)
    with pytest.raises(ValueError, match="bar"):
        s.generate(MarketData(bars=h4))
    with pytest.raises(ValueError, match="bar"):  # also when too short for the warm-up
        s.generate(MarketData(bars=h4.iloc[:50]))


def test_generate_does_not_depend_on_the_history_start(fitted) -> None:
    """Live runners pass a sliding window of recent bars. With the default monthly episode
    anchor, every forecast from ``warmup_bars`` into the window on is the one the full-history
    backtest produced (the continuous simulated account made them path-dependent)."""
    s, md = fitted
    assert s.params["episode_anchor"] == "auto" and s.episode_anchor == "M"
    full = s.generate(md)
    for offset in (137, 311):
        sub = MarketData(bars=md.bars.iloc[offset:])
        f = s.generate(sub)
        assert (f.iloc[: s.warmup_bars] == 0).all()
        tail = f.iloc[s.warmup_bars:]
        assert len(tail) > 200 and (tail != 0).any()
        pd.testing.assert_series_equal(tail, full.loc[tail.index])
