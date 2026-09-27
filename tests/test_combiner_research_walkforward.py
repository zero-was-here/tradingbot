"""Walk-forward additions: point-in-time macro for Strategy.fit, net-of-cost combiner inputs,
the holdout ledger (+ DSR n_trials from prior looks), holdout OOS forecasts and the TRAIN
feature reference of a fitted quant book."""

from __future__ import annotations

import json
import logging
import threading

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.research.walkforward import (
    HOLDOUT_LEDGER,
    _pit_macro,
    _train_md,
    append_holdout_ledger,
    fit_quant_book,
    prior_holdout_looks,
    read_holdout_ledger,
    run_walk_forward,
)
from aurum.strategies.base import Strategy


class Mom(Strategy):
    name = "cr_mom"

    @property
    def warmup_bars(self) -> int:
        return 60

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        s = r.ewm(span=6, adjust=False).mean() / r.ewm(span=48, adjust=False, min_periods=24).std()
        return self._finalize((1.5 * s).clip(-1, 1), md.bars.index)


class Slow(Strategy):
    name = "cr_slow"

    @property
    def warmup_bars(self) -> int:
        return 48

    def generate(self, md, features=None):
        c = np.log(md.bars["close"])
        return self._finalize(np.sign(c - c.shift(48)) * 0.7, md.bars.index)


class MacroSpy(Strategy):
    """Trainable; records the latest macro publication time it was allowed to see in fit."""

    name = "cr_macro_spy"
    trainable = True
    seen: list = []
    lock = threading.Lock()

    def fit(self, md, features=None):
        latest = max(pd.to_datetime(f["available_at"], utc=True).max() for f in md.macro.values() if len(f))
        with self.lock:
            self.seen.append((pd.Timestamp(md.bars["available_at"].iloc[-1]), latest,
                              sum(len(f) for f in md.macro.values())))
        self.sign_ = 1.0
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        return self._finalize(self.sign_ * np.tanh(r.rolling(6).sum() / (r.rolling(96).std() * 2)), md.bars.index)


class FeatureLinear(Strategy):
    """Trainable consumer of pipeline features (ridge on the next return)."""

    name = "cr_feat"
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


def _cfg(**wf):
    base = {"train": "3M", "test": "6W", "embargo": 6, "executor": "serial", "n_boot": 100, "pbo_splits": 4,
            "min_train_bars": 200, "combiner_min_obs": 300}
    base.update(wf)
    return load_config({"walkforward": base, "output": {"save_results": False, "tearsheet": False},
                        "risk": {"research": {"max_spread": None}}}, env={})


@pytest.fixture(scope="module")
def md() -> MarketData:
    bars = make_synthetic_bars(4500, "H1", seed=4, model="trend", regime_params={"phi": 0.12})
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=4),
                      events=make_synthetic_events(bars.index[0], bars.index[-1]))


# ---- point-in-time macro for fit ------------------------------------------------------------------
def test_pit_macro_truncation_helpers(md):
    cut = md.bars["available_at"].iloc[2000]
    tm = _pit_macro(md.macro, cut)
    for k, f in tm.items():
        assert (pd.to_datetime(f["available_at"], utc=True) <= cut).all()
        full = md.macro[k]
        assert len(f) == int((pd.to_datetime(full["available_at"], utc=True) <= cut).sum()) < len(full)
    # frames without available_at fall back to their index
    legacy = {"x": md.macro["dxy"].drop(columns="available_at")}
    assert (_pit_macro(legacy, cut)["x"].index <= cut).all()
    t = _train_md(md, 100, 2001)
    assert t.bars.index.equals(md.bars.index[100:2001])
    assert all(len(t.macro[k]) == len(tm[k]) for k in tm)
    assert t.events is md.events


def test_trainable_fit_never_sees_later_macro_prints(md):
    MacroSpy.seen.clear()
    cfg = _cfg(holdout_start=str(md.bars.index[-700]))
    rep = run_walk_forward(md, cfg, strategies=[Mom(), MacroSpy()])
    assert len(MacroSpy.seen) == len(rep.folds) + 1          # every fold + the holdout fit
    full_latest = max(pd.to_datetime(f["available_at"], utc=True).max() for f in md.macro.values())
    for train_end_decision, latest, _ in MacroSpy.seen:
        assert latest <= train_end_decision < full_latest
    # the quant book fit follows the same rule
    MacroSpy.seen.clear()
    book = fit_quant_book(md, cfg, strategies=[Mom(), MacroSpy()], fit_end=3000)
    (dec, latest, _), = MacroSpy.seen
    assert latest <= dec and dec <= md.bars["available_at"].iloc[book.fit_end - 1]


def test_walk_forward_combiner_is_net_of_costs(md):
    cfg = _cfg()
    cfg.combiner.max_weight = 1.0          # let the weights reflect the (net) Sharpe ratios
    rep = run_walk_forward(md, cfg, strategies=[Mom(), Slow()])
    fitted = rep.folds.loc[rep.folds["combiner_basis"].str.startswith("oos_history")]
    assert len(fitted)
    assert not any("GROSS" in n for n in rep.notes)
    # a zero-cost model must change the weights: the cost inputs really reach the combiner
    cfg0 = _cfg()
    cfg0.combiner.max_weight = 1.0
    cfg0.costs.min_spread = 0.0
    cfg0.costs.spread_multiplier = 0.0
    cfg0.costs.slippage_fixed = 0.0
    cfg0.costs.slippage_range_frac = 0.0
    rep0 = run_walk_forward(md, cfg0, strategies=[Mom(), Slow()])
    w, w0 = rep.weights.loc[fitted["fold"]], rep0.weights.loc[fitted["fold"]]
    assert not np.allclose(w.drop(columns="fdm").to_numpy(dtype=float), w0.drop(columns="fdm").to_numpy(dtype=float))
    # the fitted book's combiner documents its basis: net of the configured costs
    book = fit_quant_book(md, cfg, strategies=[Mom(), Slow()], oos_forecasts=rep.oos_forecasts)
    ex = book.combiner.explain()
    assert book.combiner_basis.startswith("oos_history") and ex["cost_basis"].startswith("net of estimated")
    assert all(ex["train_sharpe"][k] < ex["train_sharpe_gross"][k] for k in ("cr_mom", "cr_slow"))


# ---- holdout ledger ---------------------------------------------------------------------------------
def test_holdout_ledger_records_and_counts_prior_looks(md, tmp_path, caplog):
    hs = str(md.bars.index[-700])
    strats = [Mom(), Slow()]
    runs = tmp_path / "runs"
    ledger = runs / HOLDOUT_LEDGER

    a = run_walk_forward(md, _cfg(holdout_start=hs), strategies=strats, out_dir=runs / "a", write=True)
    assert ledger.exists()
    (e1,) = read_holdout_ledger(ledger)
    assert e1["config_hash"] == a.config_hash and e1["data_hash"] == a.data_hash
    assert e1["strategies"] == ["cr_mom", "cr_slow"]
    assert pd.Timestamp(e1["holdout_start"]) == a.holdout.start and e1["run_dir"] == str(runs / "a")
    assert e1["metrics"]["combined"]["sharpe"] == pytest.approx(a.holdout.stats.loc["combined", "sharpe"])
    assert a.n_trials == 2 and a.settings["n_trials_prior_looks"] == 0
    assert not any("WARNING" in n for n in a.notes)

    # the SAME config again: a rerun, not a new trial
    b = run_walk_forward(md, _cfg(holdout_start=hs), strategies=strats, out_dir=runs / "b", write=True)
    assert b.n_trials == 2 and b.holdout.prior_looks["n_prior_same_config"] == 1
    assert not any("WARNING" in n for n in b.notes)

    # a DIFFERENT config on the same holdout window: warning, note, n_trials raised
    caplog.set_level(logging.WARNING, logger="aurum.research.walkforward")
    c = run_walk_forward(md, _cfg(holdout_start=hs, embargo=12), strategies=strats, out_dir=runs / "c", write=True)
    assert c.config_hash != a.config_hash
    assert c.holdout.prior_looks["n_prior_configs"] == 1 and c.holdout.prior_looks["n_prior_looks"] == 2
    assert c.n_trials == 3 and c.settings["n_trials_base"] == 2 and c.settings["n_trials_prior_looks"] == 1
    assert (c.stats["n_trials"].drop("benchmark") == 3).all()
    assert (c.holdout.stats["n_trials"].drop("benchmark") == 3).all()
    assert any("WARNING" in n and "evaluated before by 1 other config" in n for n in c.notes)
    assert any("evaluated before" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    s = json.loads((runs / "c" / "summary.json").read_text())
    assert s["holdout"]["prior_looks"]["n_prior_configs"] == 1
    assert json.loads((runs / "c" / "holdout" / "holdout.json").read_text())["prior_looks"]["n_prior_looks"] == 2

    # a third config sees two other configs; the deflation is monotone in the trials
    d = run_walk_forward(md, _cfg(holdout_start=hs, embargo=18), strategies=strats, out_dir=runs / "d", write=True)
    assert d.n_trials == 4 and d.holdout.prior_looks["n_prior_configs"] == 2
    assert len(read_holdout_ledger(ledger)) == 4


def test_holdout_ledger_modes(md, tmp_path):
    hs = str(md.bars.index[-700])
    strats = [Mom(), Slow()]
    # not written and no explicit ledger: not recorded, and the report says so
    r = run_walk_forward(md, _cfg(holdout_start=hs), strategies=strats, write=False)
    assert any("NOT recorded" in n for n in r.notes)
    # an explicit ledger path records even without writing the run
    led = tmp_path / "custom" / "ledger.jsonl"
    r2 = run_walk_forward(md, _cfg(holdout_start=hs), strategies=strats, write=False, holdout_ledger=led)
    assert len(read_holdout_ledger(led)) == 1 and read_holdout_ledger(led)[0]["run_dir"] is None
    assert r2.holdout.ledger_path == led
    # no holdout -> nothing to record
    run_walk_forward(md, _cfg(), strategies=strats, write=False, holdout_ledger=led)
    assert len(read_holdout_ledger(led)) == 1
    # holdout_ledger=False switches it off even when writing
    run_walk_forward(md, _cfg(holdout_start=hs), strategies=strats, out_dir=tmp_path / "w" / "x", write=True,
                     holdout_ledger=False)
    assert not (tmp_path / "w" / HOLDOUT_LEDGER).exists()


def test_prior_looks_overlap_symbol_and_corrupt_lines(tmp_path):
    led = tmp_path / HOLDOUT_LEDGER
    base = {"symbol": "XAUUSD", "holdout_start": "2024-01-01T00:00:00+00:00",
            "holdout_end": "2024-06-30T00:00:00+00:00", "timestamp": "2026-01-01T00:00:00+00:00"}
    append_holdout_ledger(led, {**base, "config_hash": "aaa"})
    append_holdout_ledger(led, {**base, "config_hash": "bbb", "holdout_start": "2025-01-01T00:00:00+00:00",
                                "holdout_end": "2025-06-30T00:00:00+00:00"})           # disjoint window
    append_holdout_ledger(led, {**base, "config_hash": "ccc", "symbol": "XAGUSD"})       # other symbol
    with open(led, "a") as fh:
        fh.write("{not json\n")
    append_holdout_ledger(led, {**base, "config_hash": "ddd", "holdout_start": "2024-06-01T00:00:00+00:00",
                                "holdout_end": "2024-12-31T00:00:00+00:00"})           # overlaps
    entries = read_holdout_ledger(led)
    assert len(entries) == 4
    p = prior_holdout_looks(entries, start="2024-03-01", end="2024-09-30", config_hash="aaa", symbol="XAUUSD")
    assert p["n_prior_looks"] == 2 and p["n_prior_configs"] == 1 and p["prior_config_hashes"] == ["ddd"]
    assert p["n_prior_same_config"] == 1


# ---- holdout forecasts & feature reference ------------------------------------------------------------
def test_holdout_forecasts_kept_and_saved(md, tmp_path):
    hs = md.bars.index[-700]
    rep = run_walk_forward(md, _cfg(holdout_start=str(hs)), strategies=[Mom(), Slow()], out_dir=tmp_path / "r",
                           write=True)
    h = rep.holdout.forecasts
    assert list(h.columns) == ["cr_mom", "cr_slow", "combined"]
    assert h.index[0] == hs and h.index[-1] == md.bars.index[-1]
    assert h.index[0] > rep.oos_forecasts.index[-1]
    saved = pd.read_parquet(tmp_path / "r" / "holdout" / "oos_forecasts.parquet")
    pd.testing.assert_frame_equal(saved, h, check_freq=False)


def test_quant_book_feature_reference_is_from_train_transformed_features(md):
    from aurum.live.monitor import FeatureReference

    cfg = _cfg()
    cfg.features.enabled = True
    cfg.features.groups = ["returns", "volatility"]
    book = fit_quant_book(md, cfg, strategies=[Mom(), FeatureLinear()], fit_end=3500, gap=False)
    ref = book.feature_reference
    assert isinstance(ref, FeatureReference) and ref.source == "train" and ref.columns
    tr0 = int(md.bars.index.get_loc(book.train_start))
    tr1 = int(md.bars.index.get_loc(book.train_end)) + 1
    x = book.pipeline.transform(book.pipeline.compute(md)).iloc[tr0:tr1]
    exp = FeatureReference.from_frame(x)
    assert ref.edges == exp.edges and ref.expected == exp.expected and ref.n_obs == tr1 - tr0
    assert set(ref.columns) <= set(book.pipeline.columns)
    # no feature consumer -> no pipeline, no reference
    assert fit_quant_book(md, _cfg(), strategies=[Mom(), Slow()], fit_end=3500).feature_reference is None
