"""v2.1 core fixes - pre-trained ``rl_ppo`` artifacts: ``config.json`` records the train /
validation periods; with ``artifact_dir`` (``retrain`` unset -> False) ``fit`` LOADS the
fingerprint-checked artifact instead of retraining, and ``generate`` refuses out-of-sample
(walk-forward TEST) bars the artifact was trained on - the look-ahead guard."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import pandas as pd
import pytest

from aurum.core.config import load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.strategies.base import Strategy, get_strategy

pytest.importorskip("stable_baselines3")

from aurum.rl.train import RLTrainConfig, artifact_data_ranges, load_artifact  # noqa: E402
from aurum.strategies.rl import RLPolicyStrategy  # noqa: E402

N = 4200
N_ART = 2600     # the artifact is trained + validated on bars[:N_ART]


def _config() -> dict:
    return {
        "feature_groups": ["returns", "volatility", "session"],
        "env": {"window": 2, "episode_length": 256},
        "total_timesteps": 1024, "n_envs": 2, "n_steps": 256, "batch_size": 128,
        "n_epochs": 2, "eval_freq": 512, "net_arch": [32, 32], "seed": 1,
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    bars = make_synthetic_bars(N, "H1", seed=4, model="trend", regime_params={"phi": 0.25})
    md = MarketData(bars=bars)
    out = tmp_path_factory.mktemp("v21_rl")
    s = get_strategy("rl_ppo", config=_config(), val_fraction=0.25, out_dir=str(out))
    s.fit(MarketData(bars=bars.iloc[:N_ART]))
    full = RLPolicyStrategy.from_artifact(s.artifact_dir_).generate(md)   # no guard: never fitted
    return s, md, Path(s.artifact_dir_), full


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# ---- recorded periods --------------------------------------------------------------------------------
def test_config_json_records_the_training_period(trained):
    s, md, art, _ = trained
    idx = md.bars.index
    raw = json.loads((art / "config.json").read_text())
    n_val = round(N_ART * 0.25)
    assert raw["data_ranges"]["train"] == [str(idx[0]), str(idx[N_ART - n_val - 1])]
    assert raw["data_ranges"]["val"] == [str(idx[N_ART - n_val]), str(idx[N_ART - 1])]
    assert raw["data_ranges"]["timeframe"] == "H1"
    r = artifact_data_ranges(art)
    assert r["start"] == idx[0] and r["end"] == idx[N_ART - 1] and r["source"] == "config.json"
    assert load_artifact(art).data_ranges == r
    assert s.artifact_data_ranges_ == r
    # the provenance key is not an RLTrainConfig field: tolerated only when reading an artifact
    assert RLTrainConfig.from_artifact_dict(raw).to_dict() == RLTrainConfig.from_dict(_config()).to_dict()
    with pytest.raises(ValueError, match="unknown RLTrainConfig keys"):
        RLTrainConfig.from_dict(raw)


def test_training_fit_keeps_retraining_and_records_its_cutoff(trained):
    s, md, art, _ = trained
    assert s.params["retrain"] is True               # resolved at construction (no artifact_dir)
    assert s.params["artifact_dir"] == str(art)      # provenance rewritten by load() ...
    assert s.clone().params["retrain"] is True       # ... without turning a refit into a load
    assert s.fit_cutoff_ == md.bars.index[N_ART - 1]
    pd.testing.assert_series_equal(s.generate(md), trained[3])   # nothing after the period refused


# ---- loading fit -------------------------------------------------------------------------------------
def test_artifact_dir_fit_loads_instead_of_retraining(trained, monkeypatch):
    _, md, art, full = trained
    import aurum.rl.train as rl_train

    def boom(*a, **k):
        raise AssertionError("train_ppo must not be called when retrain=False")

    monkeypatch.setattr(rl_train, "train_ppo", boom)
    p = RLPolicyStrategy(artifact_dir=str(art))
    assert p.params["retrain"] is False and p.is_fitted
    fp = p.artifact_fingerprint_
    p.fit(MarketData(bars=md.bars.iloc[:3000]))
    assert p.fit_cutoff_ == md.bars.index[2999] and p.artifact_fingerprint_ == fp
    assert p.artifact_dir_ == art
    pd.testing.assert_series_equal(p.generate(md), full)


def test_fit_refuses_an_artifact_trained_after_the_cutoff(trained):
    _, md, art, full = trained
    p = RLPolicyStrategy.from_artifact(art)
    for end in (2000, N_ART - 1):                     # fold cutoffs BEFORE the artifact's last bar
        with pytest.raises(ValueError, match="look-ahead") as ei:
            p.fit(MarketData(bars=md.bars.iloc[:end]))
        assert str(md.bars.index[N_ART - 1]) in str(ei.value)
    assert p.fit_cutoff_ is None                      # a refused fit records nothing
    # a cutoff at the artifact's last bar: every later bar is genuinely out of sample
    q = RLPolicyStrategy.from_artifact(art).fit(MarketData(bars=md.bars.iloc[:N_ART]))
    pd.testing.assert_series_equal(q.generate(md), full)


def test_generate_guard_refuses_out_of_sample_bars_inside_the_period(trained):
    """Defence in depth: should a fitted strategy carry a cutoff before its artifact's end
    (hand-edited state), generate still refuses the out-of-sample bars inside the period."""
    _, md, art, full = trained
    p = RLPolicyStrategy.from_artifact(art).fit(MarketData(bars=md.bars.iloc[:N_ART]))
    p.fit_cutoff_ = md.bars.index[1999]
    pd.testing.assert_series_equal(p.generate(MarketData(bars=md.bars.iloc[:2000])), full.iloc[:2000])
    for end in (2001, N_ART, N):
        with pytest.raises(ValueError, match="look-ahead"):
            p.generate(MarketData(bars=md.bars.iloc[:end]))


def test_guard_survives_pickles_and_clones(trained):
    _, md, art, _ = trained
    p = RLPolicyStrategy.from_artifact(art).fit(MarketData(bars=md.bars.iloc[:N_ART]))
    p.fit_cutoff_ = md.bars.index[1799]
    for other in (pickle.loads(pickle.dumps(p)), p.clone()):
        assert other.fit_cutoff_ == p.fit_cutoff_ and other.params["retrain"] is False
        with pytest.raises(ValueError, match="look-ahead"):
            other.generate(md)
        with pytest.raises(ValueError, match="look-ahead"):
            other.fit(MarketData(bars=md.bars.iloc[:1800]))
    # a strategy pickled before the guard existed (no cutoff / ranges) behaves as before
    legacy = pickle.loads(pickle.dumps(p))
    state = legacy.__dict__.copy()
    state.pop("fit_cutoff_")
    state.pop("artifact_data_ranges_")
    old = RLPolicyStrategy.__new__(RLPolicyStrategy)
    old.__setstate__(state)
    assert old.fit_cutoff_ is None and old.artifact_data_ranges_ is None
    assert len(old.generate(md)) == N


def test_bar_size_is_checked_by_the_loading_fit(trained):
    from aurum.features.multi_timeframe import resample_anchored

    _, md, art, _ = trained
    h4 = resample_anchored(md.bars, "H4", 0)
    with pytest.raises(ValueError, match="bar"):
        RLPolicyStrategy.from_artifact(art).fit(MarketData(bars=h4))


# ---- fingerprint checks ------------------------------------------------------------------------------
def test_loading_fit_is_fingerprint_checked(trained, tmp_path):
    _, md, art, full = trained
    md_fit = MarketData(bars=md.bars.iloc[:N_ART])
    # 1) the directory changed after the policy was attached -> refused
    a = _copy(art, tmp_path / "a")
    s = RLPolicyStrategy.from_artifact(a)
    pipe = json.loads((a / "pipeline.json").read_text())
    (a / "pipeline.json").write_text(json.dumps(pipe, indent=4), encoding="utf-8")   # same content, new bytes
    with pytest.raises(RuntimeError, match="changed on disk"):
        s.fit(md_fit)
    # 2) the directory is gone (e.g. a pickle shipped to another machine): embedded bytes are used
    b = _copy(art, tmp_path / "b")
    s2 = RLPolicyStrategy.from_artifact(b)
    shutil.rmtree(b)
    pd.testing.assert_series_equal(s2.fit(md_fit).generate(md), full)
    # 3) tampered embedded bytes -> refused
    s3 = pickle.loads(pickle.dumps(RLPolicyStrategy.from_artifact(art)))
    s3._bundle["config.json"] = s3._bundle["config.json"].replace(b'"seed": 1', b'"seed": 2')
    with pytest.raises(RuntimeError, match="fingerprint"):
        s3.fit(md_fit)
    # 4) the artifact_sha256 pin
    fp = RLPolicyStrategy.from_artifact(art).artifact_fingerprint_
    ok = RLPolicyStrategy.from_artifact(art, artifact_sha256=fp[:12].upper())
    pd.testing.assert_series_equal(ok.fit(md_fit).generate(md), full)
    with pytest.raises(RuntimeError, match="pinned"):
        RLPolicyStrategy.from_artifact(art, artifact_sha256="0" * 12)


def test_older_artifacts_fall_back_to_metrics_json_and_unknown_periods_are_refused(trained, tmp_path):
    _, md, art, full = trained
    old = _copy(art, tmp_path / "old")
    cfg = json.loads((old / "config.json").read_text())
    cfg.pop("data_ranges")
    (old / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    r = artifact_data_ranges(old)
    assert r["source"] == "metrics.json" and r["end"] == md.bars.index[N_ART - 1]
    s = RLPolicyStrategy.from_artifact(old).fit(MarketData(bars=md.bars.iloc[:N_ART]))
    pd.testing.assert_series_equal(s.generate(md), full)
    # no record at all -> a walk-forward cannot verify the test rows: refused
    m = json.loads((old / "metrics.json").read_text())
    m.pop("data")
    (old / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    assert artifact_data_ranges(old) is None
    with pytest.raises(ValueError, match="records no train/validation period"):
        RLPolicyStrategy.from_artifact(old).fit(MarketData(bars=md.bars.iloc[:N_ART]))


def test_retrain_true_with_artifact_dir_trains_a_new_policy(trained, tmp_path):
    _, md, art, _ = trained
    s = RLPolicyStrategy(artifact_dir=str(art), retrain=True, config=_config(), val_fraction=0.25,
                         out_dir=str(tmp_path))
    s.fit(MarketData(bars=md.bars.iloc[:2400]))
    assert s.artifact_dir_ != art and s.artifact_dir_.parent == tmp_path
    assert s.artifact_data_ranges_["end"] == md.bars.index[2399] == s.fit_cutoff_


# ---- walk-forward integration ------------------------------------------------------------------------
class Flat(Strategy):
    name = "v21_flat_mom"

    def generate(self, md, features=None):
        c = md.bars["close"]
        return self._finalize((c / c.shift(48) - 1).fillna(0.0) * 20, md.bars.index)


def _wf_cfg(train: int, test: int):
    return load_config({"walkforward": {"train": train, "test": test, "executor": "serial", "n_boot": 100,
                                        "pbo_splits": 2, "min_train_bars": 200, "combiner_min_obs": 10_000},
                        "output": {"save_results": False, "tearsheet": False},
                        "risk": {"research": {"max_spread": None}}}, env={})


def test_walk_forward_with_a_pretrained_artifact(trained):
    from aurum.research.walkforward import run_walk_forward

    _, md, art, full = trained
    # every test block starts after the artifact's validation period: allowed
    rep = run_walk_forward(md, _wf_cfg(2700, 500), strategies={"rl": RLPolicyStrategy.from_artifact(art),
                                                                "mom": Flat()}, write=False)
    oos = rep.oos_forecasts
    assert oos.index[0] == md.bars.index[2700] and len(rep.folds) == 3
    pd.testing.assert_series_equal(oos["rl"], full.loc[oos.index], check_names=False)
    # folds whose test rows the policy was trained on: refused (look-ahead)
    with pytest.raises(RuntimeError, match="'rl' failed during walk-forward") as ei:
        run_walk_forward(md, _wf_cfg(1500, 500), strategies={"rl": RLPolicyStrategy.from_artifact(art),
                                                             "mom": Flat()}, write=False)
    assert "look-ahead" in str(ei.value.__cause__)
