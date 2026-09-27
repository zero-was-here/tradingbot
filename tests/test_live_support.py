"""Shared fixtures for the aurum.live tests (stub strategies, artifact builder) + artifact tests.

The stub strategies live here (an importable module) so that pickled artifacts can be
re-loaded within the test session. Real rule strategies from ``aurum.strategies`` are used
by a separate test when that wave-2 module is importable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.pipeline import FeaturePipeline
from aurum.live.monitor import FeatureReference
from aurum.live.runner import ArtifactError, TradingArtifact, load_artifact, save_artifact
from aurum.portfolio.combiner import ForecastCombiner
from aurum.strategies.base import Strategy

GROUPS = ["returns", "volatility"]
OVERRIDES = {"volatility": {"long_window": 120}}


class EmaCrossStub(Strategy):
    """Continuous EMA-cross forecast (rule, not trainable)."""

    name = "ema_cross_stub"
    description = "EMA(12) - EMA(48) in units of close * 1% (test stub)"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"fast": 12, "slow": 48, "scale": 0.004}

    @property
    def warmup_bars(self) -> int:
        return int(self.params["slow"])

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        c = md.bars["close"].astype(float)
        fast = c.ewm(span=self.params["fast"], adjust=False).mean()
        slow = c.ewm(span=self.params["slow"], adjust=False).mean()
        f = np.tanh((fast - slow) / (c * self.params["scale"]))
        return self._finalize(f, md.bars.index)


class MomentumStub(Strategy):
    """Trainable: learns the sign of 24-bar momentum's relation to the next return on TRAIN."""

    name = "mom_stub"
    trainable = True

    @property
    def warmup_bars(self) -> int:
        return 25

    def _signal(self, md: MarketData) -> pd.Series:
        c = md.bars["close"].astype(float)
        r = np.log(c).diff()
        vol = r.rolling(24, min_periods=12).std()
        return (np.log(c).diff(24) / (vol * math.sqrt(24))).clip(-3, 3) / 3

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> MomentumStub:
        x = self._signal(md)
        y = np.log(md.bars["close"].astype(float)).diff().shift(-1)
        ok = x.notna() & y.notna()
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else 0.0
        self.sign_ = 1.0 if corr >= 0 else -1.0
        self.is_fitted = True
        return self

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        return self._finalize(self.sign_ * self._signal(md), md.bars.index)


class FeatureStub(Strategy):
    """Uses the TRANSFORMED pipeline features (exercises the features path)."""

    name = "feature_stub"

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        if features is None or "returns_z_24" not in features.columns:
            raise ValueError("feature_stub needs pipeline features")
        return self._finalize(-features["returns_z_24"] / 5.0, md.bars.index)


class ExplodingStub(Strategy):
    """Raises once the latest bar opens at/after ``explode_from`` (runner error handling)."""

    name = "exploding_stub"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"explode_from": None, "always": False}

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        t0 = self.params["explode_from"]
        if self.params["always"] or (t0 is not None and md.bars.index[-1] >= pd.Timestamp(t0)):
            raise RuntimeError("boom")
        return self._finalize(pd.Series(0.5, index=md.bars.index), md.bars.index)


def synthetic_bars(n: int = 900, *, seed: int = 7, weekend_gaps: bool = False, model: str = "trend") -> pd.DataFrame:
    return make_synthetic_bars(n, "H1", seed=seed, model=model, weekend_gaps=weekend_gaps)


def build_artifact(path: Path, bars: pd.DataFrame, *, n_train: int = 400, combiner: bool = True,
                   strategies: list[Strategy] | None = None, sizer: dict | None = None,
                   backtest_stats: dict | None = None, reference: bool = True, pipeline: bool = True,
                   overwrite: bool = False) -> Path:
    """Fit pipeline/strategies/combiner on the first ``n_train`` bars and save an artifact."""
    train = MarketData(bars=bars.iloc[:n_train])
    pipe = None
    feats = None
    if pipeline:
        pipe = FeaturePipeline(groups=GROUPS, overrides=OVERRIDES)
        raw = pipe.compute(train)
        pipe.fit(raw.iloc[pipe.max_lookback:])
        feats = pipe.transform(raw)
    strats = strategies if strategies is not None else [EmaCrossStub(), MomentumStub(), FeatureStub()]
    for s in strats:
        s.fit(train, feats)
    comb = None
    if combiner:
        fc = pd.DataFrame({s.name: s.generate(train, feats) for s in strats})
        comb = ForecastCombiner(method="equal", fdm_cap=1.5).fit(fc, train.bars["close"])
    ref = FeatureReference.from_frame(feats.dropna()) if (reference and feats is not None) else None
    return save_artifact(path, strategies=strats, pipeline=pipe, combiner=comb, symbol="XAUUSD", timeframe="H1",
                         sizer_config=sizer if sizer is not None else {"target_vol": 0.10, "rebalance_band": 0.10},
                         backtest_stats=backtest_stats, feature_reference=ref,
                         training={"start": bars.index[0], "end": bars.index[n_train - 1]}, overwrite=overwrite)


# ------------------------------------------------------------------------------------------------
# artifact tests
# ------------------------------------------------------------------------------------------------
def test_artifact_roundtrip(tmp_path: Path) -> None:
    bars = synthetic_bars(600)
    p = build_artifact(tmp_path / "art", bars, backtest_stats={"daily_mean": 0.0002, "daily_std": 0.006})
    art = load_artifact(p)
    assert isinstance(art, TradingArtifact)
    assert sorted(art.strategies) == ["ema_cross_stub", "feature_stub", "mom_stub"]
    assert art.timeframe == "H1" and art.symbol == "XAUUSD"
    assert art.max_lookback >= 144
    assert art.pipeline is not None and art.pipeline.is_fitted
    assert art.feature_reference is not None and art.feature_reference.columns
    assert art.backtest_stats["daily_std"] == pytest.approx(0.006)
    md = MarketData(bars=bars)
    X = art.features(md)
    fc = pd.DataFrame({n: s.generate(md, X) for n, s in art.strategies.items()})
    comb = art.combine(fc)
    assert comb.between(-1, 1).all() and comb.index.equals(bars.index)
    manifest = json.loads((p / "manifest.json").read_text())
    assert set(manifest["files"]) >= {"strategies.pkl", "combiner.pkl", "pipeline.json"}
    assert manifest["versions"]["pandas"] == pd.__version__


def test_artifact_tamper_detected(tmp_path: Path) -> None:
    p = build_artifact(tmp_path / "art", synthetic_bars(500))
    with open(p / "combiner.pkl", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(ArtifactError, match="sha256"):
        load_artifact(p)


def test_artifact_version_and_format_checks(tmp_path: Path) -> None:
    p = build_artifact(tmp_path / "art", synthetic_bars(500))
    m = json.loads((p / "manifest.json").read_text())
    m["version"] = 99
    (p / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="version"):
        load_artifact(p)
    m["version"] = 1
    m["format"] = "something.else"
    (p / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="format"):
        load_artifact(p)
    with pytest.raises(ArtifactError, match="manifest"):
        load_artifact(tmp_path / "nope")


def test_artifact_hmac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURUM_ARTIFACT_KEY", "k" * 32)
    p = build_artifact(tmp_path / "art", synthetic_bars(500))
    assert load_artifact(p).strategies
    monkeypatch.setenv("AURUM_ARTIFACT_KEY", "another-key-entirely")
    with pytest.raises(ArtifactError, match="HMAC"):
        load_artifact(p)
    monkeypatch.delenv("AURUM_ARTIFACT_KEY")
    with pytest.raises(ArtifactError, match="require_hmac"):
        load_artifact(p, require_hmac=True)


def test_artifact_refuses_unfitted_and_overwrite(tmp_path: Path) -> None:
    bars = synthetic_bars(500)
    with pytest.raises(ValueError, match="not fitted"):
        save_artifact(tmp_path / "a", strategies=[MomentumStub()])
    p = build_artifact(tmp_path / "a", bars)
    with pytest.raises(FileExistsError):
        build_artifact(p, bars)
    build_artifact(p, bars, overwrite=True, combiner=False)
    art = load_artifact(p)
    assert art.combiner is None  # equal-weight fallback
    assert not list(tmp_path.glob(".a.*")), "temporary directories must be cleaned up"


def test_artifact_tampered_hmac_field_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURUM_ARTIFACT_KEY", "k" * 32)
    p = build_artifact(tmp_path / "art", synthetic_bars(500))
    m = json.loads((p / "manifest.json").read_text())
    m["hmac"] = 12345  # not a string: must be an integrity failure, not a TypeError
    (p / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="HMAC"):
        load_artifact(p)
