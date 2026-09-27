"""``aurum train-final``: production artifact fitted on ALL data up to a cutoff, combiner
weights from the stitched OUT-OF-SAMPLE walk-forward, feature reference + config +
provenance in the artifact, loadable by the live runner (paper mode), and point-in-time:
data after the cutoff cannot change it. Synthetic data only (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import aurum.strategies.base as sbase
from aurum.cli import EXIT_OK, EXIT_USAGE, main
from aurum.core.config import duration_to_bars, load_config
from aurum.core.types import MarketData
from aurum.data.macro import save_macro_dir
from aurum.data.store import frame_hash, load_bars, save_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro
from aurum.execution.costs import CostModel
from aurum.live.broker import SimulatedClock
from aurum.live.monitor import FeatureReference, PnLBand
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.live.runner import LiveConfig, LiveRunner, load_artifact
from aurum.portfolio.combiner import ForecastCombiner
from aurum.research.walkforward import HOLDOUT_LEDGER, _bars_per_day, read_holdout_ledger
from aurum.strategies.base import Strategy

CUT = 4400          # the artifact is trained on bars [0, CUT]; the paper run trades after it


class TfMom(Strategy):
    name = "cr_tf_mom"
    description = "test stub: short EMA momentum"

    @property
    def warmup_bars(self) -> int:
        return 50

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        s = r.ewm(span=6, adjust=False).mean() / r.ewm(span=48, adjust=False, min_periods=24).std()
        return self._finalize((1.5 * s).clip(-1, 1), md.bars.index)


class TfFeat(Strategy):
    """Trainable consumer of the FeaturePipeline (ridge on the next vol-normalised return)."""

    name = "cr_tf_feat"
    description = "test stub: ridge on pipeline features"
    trainable = True
    uses_features = True

    def fit(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        y = (r / r.ewm(span=48, adjust=False, min_periods=24).std()).shift(-1)
        x = features.fillna(0.0)
        ok = y.notna()
        a, b = x.loc[ok].to_numpy(), y.loc[ok].to_numpy()
        self.beta_ = np.linalg.solve(a.T @ a + 10.0 * np.eye(a.shape[1]), a.T @ b)
        self.cols_ = list(features.columns)
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        x = features[self.cols_].fillna(0.0).to_numpy()
        return self._finalize(pd.Series(np.tanh(3.0 * x @ self.beta_), index=md.bars.index), md.bars.index)


@pytest.fixture(scope="module", autouse=True)
def _stub_registry():
    added = []
    for cls in (TfMom, TfFeat):
        if cls.name not in sbase._STRATEGIES:
            sbase.register_strategy(cls)
            added.append(cls.name)
    yield
    for name in added:
        sbase._STRATEGIES.pop(name, None)


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    for var in ("AURUM_ALERT_WEBHOOK_URL", "ANTHROPIC_API_KEY", "AURUM_ARTIFACT_KEY"):
        monkeypatch.delenv(var, raising=False)


def _write_config(root: Path, data: Path, name: str = "tf.yaml") -> Path:
    bars = load_bars(data / "xauusd_H1.parquet")
    cfg = {
        "name": "train_final_test", "seed": 1,
        "data": {"dir": str(data), "timeframe": "H1", "events": "rule_based"},
        "strategies": [{"name": "cr_tf_mom"}, {"name": "cr_tf_feat"}],
        "features": {"enabled": "auto", "groups": ["returns", "volatility"]},
        "walkforward": {"train": "3M", "test": "1M", "embargo": 4, "executor": "serial", "n_boot": 100,
                        "pbo_splits": 4, "min_train_bars": 200, "combiner_min_obs": 300,
                        "holdout_start": str(bars.index[3800])},
        "combiner": {"max_weight": 1.0},          # weights then reflect the net Sharpe ratios
        "risk": {"research": {"max_spread": None}},
        "output": {"dir": str(root / "runs"), "tearsheet": False},
    }
    path = root / name
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("train_final")
    data = root / "data"
    bars = make_synthetic_bars(4700, "H1", seed=9, model="trend", regime_params={"phi": 0.12})
    save_bars(bars, data / "xauusd_H1.parquet")
    macro = make_synthetic_macro(bars, seed=9)
    save_macro_dir(macro, data / "macro")
    cfg = _write_config(root, data)
    cutoff = str(bars["available_at"].iloc[CUT])
    art = root / "art"
    code = main(["train-final", "--config", str(cfg), "--out", str(art), "--cutoff", cutoff])
    assert code == EXIT_OK
    return {"root": root, "data": data, "config": cfg, "bars": bars, "macro": macro, "art": art, "cutoff": cutoff}


def _run_dir(root: Path) -> Path:
    (d,) = [p for p in (root / "runs").iterdir() if p.is_dir() and p.name.startswith("walkforward_")]
    return d


def test_artifact_contents_and_oos_weights(env):
    art = load_artifact(env["art"])
    bars = env["bars"]
    assert sorted(art.strategies) == ["cr_tf_feat", "cr_tf_mom"]
    assert art.pipeline is not None and art.pipeline.is_fitted
    assert all(getattr(s, "is_fitted", True) for s in art.strategies.values())
    # feature reference: TRAIN-window transformed features (not the Gaussian approximation)
    ref = art.feature_reference
    assert isinstance(ref, FeatureReference) and ref.source == "train" and len(ref.columns) > 10
    assert set(ref.columns) <= set(art.pipeline.columns)
    # manifest: config, provenance, cutoff, OOS-based combiner
    tr = art.manifest["training"]
    assert tr["kind"] == "train_final"
    assert pd.Timestamp(tr["cutoff"]) == bars["available_at"].iloc[CUT]
    assert pd.Timestamp(tr["train_end"]) == bars.index[CUT]            # no data after the cutoff
    assert tr["config"]["name"] == "train_final_test"
    assert tr["config_hash"] == load_config(str(env["config"])).config_hash()
    assert tr["provenance"]["data"]["bars_hash"] == frame_hash(bars.iloc[:CUT + 1]) == tr["data_hash"]
    assert tr["combiner_basis"].startswith("oos_history") and tr["combiner_cost_basis"].startswith("net")
    assert art.backtest_stats["combined"]["sharpe"] is not None
    assert PnLBand.from_stats(art.backtest_stats) is not None          # the live PnL band can use it
    # combiner weights = the fold rule applied to the stitched OUT-OF-SAMPLE forecasts (research
    # folds + holdout), last `train` bars, net of the configured costs - not the in-sample fit
    run = _run_dir(env["root"])
    oos = pd.read_parquet(run / "oos_forecasts.parquet")
    hold = pd.read_parquet(run / "holdout" / "oos_forecasts.parquet")
    keys = ["cr_tf_mom", "cr_tf_feat"]
    hist = pd.concat([oos[keys], hold[keys]]).sort_index()
    hist = hist.loc[hist.index <= bars.index[CUT]]
    cfg = load_config(str(env["config"]))
    # the fold rule's window: walkforward.train in bars at the density of the data up to the cutoff
    window = duration_to_bars(cfg.walkforward.train, _bars_per_day(bars.index[:CUT + 1]))
    assert json.loads((run / "summary.json").read_text())["settings"]["train"] == pytest.approx(window, rel=0.02)
    ref_comb = ForecastCombiner(method=cfg.combiner.method, shrinkage=cfg.combiner.shrinkage,
                                max_weight=cfg.combiner.max_weight, fdm_cap=cfg.combiner.fdm_cap).fit(
        hist.iloc[-window:], bars["close"], bars=bars, costs=cfg.costs.build(), instrument=cfg.instrument.build())
    comb = art.combiner
    assert 0 < comb.weights_.min() and comb.weights_.max() < 1      # informative, not a capped 0.5/0.5
    pd.testing.assert_series_equal(comb.weights_.loc[keys], ref_comb.weights_.loc[keys])
    assert comb.fdm_ == pytest.approx(ref_comb.fdm_)
    assert comb.train_start_ >= oos.index[0] and comb.train_end_ == bars.index[CUT]
    # the in-process walk-forward evaluated the holdout once and recorded it
    (entry,) = read_holdout_ledger(env["root"] / "runs" / HOLDOUT_LEDGER)
    assert entry["config_hash"] == tr["config_hash"]


def test_live_runner_loads_the_artifact_in_paper_mode(env, tmp_path):
    bars = env["bars"]
    art = load_artifact(env["art"])
    start = CUT + 1
    feed = ReplayFeed(bars)
    clock = SimulatedClock(bars["available_at"].iloc[start])
    broker = PaperBroker(feed, costs=CostModel(), clock=clock)
    cfg = LiveConfig(dry_run=False, state_dir=str(tmp_path / "state"), calendar=None,
                     risk={"daily_loss_persistent": False, "max_daily_loss": 0.05})
    runner = LiveRunner(cfg, broker=broker, artifact=art, clock=clock)
    res = runner.run(max_cycles=12, install_signal_handlers=False)
    assert len(res) == 12
    assert not any(r.errors for r in res)
    assert all(set(r.forecasts) == {"cr_tf_mom", "cr_tf_feat"} for r in res)
    assert all(r.bar_time > pd.Timestamp(art.manifest["training"]["train_end"]) for r in res)  # out of sample
    assert any(abs(r.combined) > 0 for r in res)
    # the drift monitor uses the artifact's TRAIN reference, and the sizer the research sizing
    assert runner.monitor.drift is not None and runner.monitor.drift.reference.source == "train"
    assert runner.sizer.target_vol == pytest.approx(0.10)


def test_from_run_reuses_oos_forecasts_and_checks_the_config(env, tmp_path, capsys):
    run = _run_dir(env["root"])
    cutoff = env["cutoff"]
    art2 = tmp_path / "art2"
    assert main(["train-final", "--config", str(env["config"]), "--out", str(art2), "--cutoff", cutoff,
                 "--from-run", str(run)]) == EXIT_OK
    a, b = load_artifact(env["art"]), load_artifact(art2)
    pd.testing.assert_series_equal(a.combiner.weights_, b.combiner.weights_)
    assert b.manifest["training"]["oos_source"] == str(run)
    # still one ledger entry: --from-run evaluates nothing
    assert len(read_holdout_ledger(env["root"] / "runs" / HOLDOUT_LEDGER)) == 1
    # a different config must not silently reuse another protocol's OOS forecasts
    capsys.readouterr()
    code = main(["train-final", "--config", str(env["config"]), "--out", str(tmp_path / "art3"), "--cutoff", cutoff,
                 "--from-run", str(run), "--set", "combiner.shrinkage=0.7"])
    assert code == EXIT_USAGE and "allow-config-mismatch" in capsys.readouterr().err
    assert main(["train-final", "--config", str(env["config"]), "--out", str(tmp_path / "art3"), "--cutoff", cutoff,
                 "--from-run", str(run), "--set", "combiner.shrinkage=0.7", "--allow-config-mismatch"]) == EXIT_OK
    # existing artifact directories are never overwritten silently
    assert main(["train-final", "--config", str(env["config"]), "--out", str(art2), "--from-run", str(run)]) \
        == EXIT_USAGE


def test_data_after_the_cutoff_cannot_change_the_artifact(env, tmp_path):
    """Perturb every price and macro print after the cutoff: the artifact is identical."""
    bars = env["bars"].copy()
    rng = np.random.default_rng(123)
    f = np.ones(len(bars))
    f[CUT + 1:] = np.exp(np.cumsum(0.01 * rng.standard_normal(len(bars) - CUT - 1)))
    for c in ("open", "high", "low", "close"):
        bars[c] = bars[c].to_numpy() * f
    bars.attrs = dict(env["bars"].attrs)
    cut_t = env["bars"]["available_at"].iloc[CUT]
    macro = {}
    for k, v in env["macro"].items():
        v = v.copy()
        late = pd.to_datetime(v["available_at"], utc=True) > cut_t
        v.loc[late.to_numpy(), "value"] = v.loc[late.to_numpy(), "value"] * 1.7
        macro[k] = v
    root = tmp_path
    data = root / "data"
    save_bars(bars, data / "xauusd_H1.parquet")
    save_macro_dir(macro, data / "macro")
    cfg = _write_config(root, data)
    art2 = root / "art"
    assert main(["train-final", "--config", str(cfg), "--out", str(art2), "--cutoff", env["cutoff"]]) == EXIT_OK
    a, b = load_artifact(env["art"]), load_artifact(art2)
    pd.testing.assert_series_equal(a.combiner.weights_, b.combiner.weights_)
    assert a.combiner.fdm_ == b.combiner.fdm_
    pd.testing.assert_frame_equal(a.pipeline.stats, b.pipeline.stats)
    assert a.feature_reference.to_dict() == b.feature_reference.to_dict()
    np.testing.assert_array_equal(a.strategies["cr_tf_feat"].beta_, b.strategies["cr_tf_feat"].beta_)
    ta, tb = a.manifest["training"], b.manifest["training"]
    assert ta["data_hash"] == tb["data_hash"]
    assert ta["provenance"]["data"]["macro_hashes"] == tb["provenance"]["data"]["macro_hashes"]
    md = MarketData(bars=env["bars"].iloc[:CUT + 1])
    x = a.features(md)
    for k in a.strategies:
        np.testing.assert_array_equal(np.asarray(a.strategies[k].generate(md, x)),
                                      np.asarray(b.strategies[k].generate(md, x)))


def test_cutoff_before_the_holdout_runs_without_one(env, tmp_path, capsys):
    """A cutoff earlier than walkforward.holdout_start: no holdout exists in that data, so the
    in-process walk-forward runs without one (and records nothing in the holdout ledger)."""
    cfg = yaml.safe_load(env["config"].read_text())
    cfg["output"]["dir"] = str(tmp_path / "runs")
    path = tmp_path / "early.yaml"
    path.write_text(yaml.safe_dump(cfg))
    bars = env["bars"]
    art = tmp_path / "art"
    code = main(["train-final", "--config", str(path), "--out", str(art), "--cutoff", str(bars["available_at"].iloc[3000])])
    out = capsys.readouterr().out
    assert code == EXIT_OK and "runs without a holdout" in out
    a = load_artifact(art)
    assert pd.Timestamp(a.manifest["training"]["train_end"]) == bars.index[3000]
    assert a.manifest["training"]["combiner_basis"].startswith("oos_history")
    assert not (tmp_path / "runs" / HOLDOUT_LEDGER).exists()
