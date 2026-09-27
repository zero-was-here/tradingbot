"""Walk-forward protocol on synthetic data with local stub strategies.

Checks the protocol invariants (no overlap, purge/embargo respected, strictly increasing
stitched OOS, holdout physically excluded, fits only see training bars), the statistics
(random walk -> combined DSR not significant; trending data -> trend strategies earn a
positive OOS Sharpe), determinism across executors and the written report.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import ConfigError, load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events
from aurum.research.walkforward import (
    FoldPlan,
    WalkForwardReport,
    fit_quant_book,
    label_horizon,
    load_summary,
    plan_folds,
    run_single_backtest,
    run_walk_forward,
)
from aurum.strategies.base import Strategy


# ---- stub strategies (causal; not registered) ---------------------------------------------------
class EmaMomentum(Strategy):
    """Vol-normalised EMA of past returns (short-horizon time-series momentum)."""

    name = "wf_ema_mom"

    @classmethod
    def default_params(cls):
        return {"span": 6}

    @property
    def warmup_bars(self) -> int:
        return 60

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        s = r.ewm(span=self.params["span"], adjust=False).mean()
        v = r.ewm(span=48, adjust=False, min_periods=24).std()
        return self._finalize((s / v * 1.5).clip(-1, 1), md.bars.index)


class Contrarian(EmaMomentum):
    name = "wf_contrarian"

    def generate(self, md, features=None):
        return self._finalize(-super().generate(md, features), md.bars.index)


class SlowTrend(Strategy):
    """Sign of a 48-bar return (a slower trend follower)."""

    name = "wf_slow_trend"

    @property
    def warmup_bars(self) -> int:
        return 48

    def generate(self, md, features=None):
        c = np.log(md.bars["close"])
        return self._finalize(np.sign(c - c.shift(48)) * 0.7, md.bars.index)


class HourNoise(Strategy):
    """Deterministic pseudo-random sign by hour of day: no information by construction."""

    name = "wf_hour_noise"

    def generate(self, md, features=None):
        h = md.bars.index.hour.to_numpy()
        return self._finalize(pd.Series(np.where((h * 7919) % 13 > 6, 0.8, -0.8), index=md.bars.index),
                              md.bars.index)


class SpyLearner(Strategy):
    """Trainable: learns the sign of the training-period lag-1 autocorrelation and records
    exactly which bars it saw in ``fit`` / ``generate`` (shared log across clones)."""

    name = "wf_spy"
    trainable = True
    log: list = []
    lock = threading.Lock()

    @classmethod
    def default_params(cls):
        return {"horizon": 12}

    def fit(self, md, features=None):
        r = np.log(md.bars["close"]).diff().dropna()
        self.sign_ = float(np.sign(r.autocorr(1)) or 1.0)
        with self.lock:
            self.log.append(("fit", md.bars.index[0], md.bars.index[-1], len(md.bars)))
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        with self.lock:
            self.log.append(("gen", md.bars.index[0], md.bars.index[-1], len(md.bars)))
        r = np.log(md.bars["close"]).diff()
        return self._finalize(self.sign_ * np.tanh(r.rolling(4).sum() / (r.rolling(96).std() * 2)), md.bars.index)


class Exploding(Strategy):
    name = "wf_exploding"
    trainable = True

    def fit(self, md, features=None):
        raise RuntimeError("fit failed on purpose")

    def generate(self, md, features=None):  # pragma: no cover - never reached
        return pd.Series(0.0, index=md.bars.index)


# ---- helpers ----------------------------------------------------------------------------------------
def _md(model: str, n: int, seed: int, **kw) -> MarketData:
    bars = make_synthetic_bars(n, "H1", seed=seed, model=model, **kw)
    return MarketData(bars=bars, events=make_synthetic_events(bars.index[0], bars.index[-1]))


def _cfg(**wf) -> object:
    base = {"train": "4M", "test": "6W", "embargo": 6, "executor": "serial", "n_boot": 200, "pbo_splits": 8,
            "min_train_bars": 200}
    base.update(wf)
    return load_config({"walkforward": base, "output": {"save_results": False, "tearsheet": False},
                        "risk": {"research": {"max_spread": None}}}, env={})


@pytest.fixture(scope="module")
def trend_md() -> MarketData:
    return _md("trend", 9000, seed=4, regime_params={"phi": 0.12})


@pytest.fixture(scope="module")
def trend_report(trend_md) -> WalkForwardReport:
    SpyLearner.log.clear()
    cfg = _cfg(holdout_start=str(trend_md.bars.index[-1200]))
    return run_walk_forward(trend_md, cfg, strategies=[EmaMomentum(), EmaMomentum(span=16), Contrarian(),
                                                       SpyLearner()])


# ---- fold planning ------------------------------------------------------------------------------------
def test_plan_folds_no_overlap_purge_embargo_and_holdout(trend_md):
    idx = trend_md.bars.index
    cfg = _cfg(purge=5, holdout_start=str(idx[-1000]))
    plans, hold, st = plan_folds(idx, cfg, {"spy": SpyLearner(horizon=30)})
    n_research = st["n_research"]
    assert n_research == len(idx) - 1000 and st["holdout_start"] == str(idx[n_research])
    assert st["purge"] == 30  # raised to the trainable strategy's label horizon
    gap = st["purge"] + st["embargo"]
    assert len(plans) >= 5
    prev_end = None
    for p in plans:
        assert isinstance(p, FoldPlan) and p.phase == "research"
        assert p.train_start < p.train_end <= p.test_start < p.test_end <= n_research
        assert p.test_start - p.train_end >= gap                       # purge + embargo gap
        assert p.n_train == st["train"]                                # rolling window
        if prev_end is not None:
            assert p.test_start == prev_end                            # contiguous, no overlap
        prev_end = p.test_end
    assert hold is not None and hold.phase == "holdout"
    assert hold.test_start == n_research and hold.test_end == len(idx)
    assert hold.train_end <= n_research - gap
    # bars per calendar day ~ 23*5/7 for synthetic H1 with weekend gaps
    assert 14 < st["bars_per_day"] < 19


def test_plan_folds_anchored_and_errors(trend_md):
    idx = trend_md.bars.index
    plans, _, _ = plan_folds(idx, _cfg(anchored=True))
    assert all(p.train_start == 0 for p in plans)
    assert [p.n_train for p in plans] == sorted(p.n_train for p in plans)
    with pytest.raises(ConfigError, match="overlapping test blocks"):
        plan_folds(idx, _cfg(step="2W"))
    with pytest.raises(ConfigError, match="no fold"):
        plan_folds(idx, _cfg(train="3Y"))
    with pytest.raises(ConfigError, match="holdout would be empty"):
        plan_folds(idx, _cfg(holdout_start="2099-01-01"))
    with pytest.raises(ConfigError, match="min_train_bars"):
        plan_folds(idx, _cfg(train=100))


def test_label_horizon_detection():
    assert label_horizon(SpyLearner(horizon=17)) == 17
    assert label_horizon(EmaMomentum()) == 0  # not trainable -> no purge needed

    class MultiH(SpyLearner):
        label_horizon = 9

    assert label_horizon(MultiH(horizon=4, max_holding_bars=[3, 40])) == 40


# ---- the protocol ------------------------------------------------------------------------------------
def test_stitched_oos_is_strictly_increasing_and_excludes_holdout(trend_md, trend_report):
    rep = trend_report
    oos = rep.oos_forecasts
    assert oos.index.is_monotonic_increasing and not oos.index.has_duplicates
    n_research = rep.settings["n_research"]
    hold_start = trend_md.bars.index[n_research]
    assert oos.index[-1] < hold_start
    expected = trend_md.bars.index[rep.folds["test_start"].map(trend_md.bars.index.get_loc).iloc[0]:n_research]
    assert oos.index.equals(expected)  # contiguous test blocks cover the OOS span exactly
    assert set(oos.columns) == {"wf_ema_mom", "wf_ema_mom#2", "wf_contrarian", "wf_spy", "combined", "fold"}
    assert oos[["wf_ema_mom", "combined"]].abs().max().max() <= 1.0
    # books span exactly the stitched OOS
    assert rep.combined.equity.index[0] == oos.index[0] and rep.combined.equity.index[-1] == oos.index[-1]
    assert rep.benchmark is not None and rep.benchmark.equity.index.equals(rep.combined.equity.index)
    # holdout reported separately, once
    h = rep.holdout
    assert h is not None and h.start == hold_start and h.end == trend_md.bars.index[-1]
    assert h.combined.equity.index[0] == hold_start
    assert set(h.stats.index) >= {"combined", "benchmark", "wf_ema_mom"}


def test_fits_only_see_training_bars_and_research_never_sees_holdout(trend_md, trend_report):
    rep = trend_report
    idx = trend_md.bars.index
    hold_start = idx[rep.settings["n_research"]]
    fits = [e for e in SpyLearner.log if e[0] == "fit"]
    gens = [e for e in SpyLearner.log if e[0] == "gen"]
    assert len(fits) == len(rep.folds) + 1  # one refit per fold + the final holdout fit
    folds = rep.folds.set_index("fold")
    by_end = {e[2]: e for e in fits}
    for _, row in folds.iterrows():
        f = by_end[row["train_end"]]
        assert f[1] == row["train_start"] and f[3] == row["n_train"]
        gap = idx.get_loc(row["test_start"]) - idx.get_loc(row["train_end"]) - 1
        assert gap >= rep.settings["purge"] + rep.settings["embargo"]
        assert f[2] < row["test_start"]
    # research generations end at the fold's last test bar, never inside the holdout
    research_gens = [g for g in gens if g[2] < hold_start]
    assert len(research_gens) == len(rep.folds)
    assert {g[2] for g in research_gens} == set(folds["test_end"])
    assert max(f[2] for f in fits[:-1]) < hold_start
    holdout_fit = max(fits, key=lambda e: e[2])
    assert holdout_fit[2] < hold_start - pd.Timedelta(hours=rep.settings["purge"] + rep.settings["embargo"])


def test_trend_strategies_positive_oos_on_trending_data(trend_report):
    # AR(1) returns: E[r(t+1) | past] = phi * r(t), so short-horizon momentum has a real edge
    st = trend_report.stats
    assert st.loc["wf_ema_mom", "sharpe"] > 0.5
    assert st.loc["wf_ema_mom#2", "sharpe"] > 0.0
    assert st.loc["wf_contrarian", "sharpe"] < 0.0
    assert st.loc["combined", "sharpe"] > 0.0
    # the combiner learned to avoid the contrarian (non-positive shrunk Sharpe) in every fold
    w = trend_report.weights
    assert (w["wf_contrarian"] <= w["wf_ema_mom"] + 1e-12).all()
    assert (w.drop(columns="fdm").sum(axis=1).round(9) == 1.0).all()
    assert (w["fdm"] >= 1.0).all()


def test_statistics_tables(trend_report):
    rep = trend_report
    assert rep.n_trials == 4
    for col in ("sharpe", "psr", "dsr", "ci_lower", "ci_upper", "kurtosis", "max_drawdown", "n_halt_episodes"):
        assert col in rep.stats.columns
    c = rep.stats.loc["combined"]
    assert c["ci_lower"] <= c["sharpe"] <= c["ci_upper"]
    assert 0.0 <= c["dsr"] <= c["psr"] <= 1.0          # deflation never raises the probability
    assert c["dsr"] > 0.5                                # real edge survives deflation by N=4 trials
    assert c["dsr_xs"] <= c["dsr"] + 1e-12               # cross-sectional variance: more conservative
    assert c["kurtosis"] > 1.0                           # Pearson (normal = 3), not excess
    assert rep.pbo is not None and 0.0 <= rep.pbo.pbo <= 1.0 and rep.pbo.n_strategies == 4
    f = rep.folds
    assert {"fold", "train_start", "test_end", "oos_sharpe", "fdm", "gap_bars"} <= set(f.columns)
    assert (f["gap_bars"] >= rep.settings["purge"] + rep.settings["embargo"]).all()
    assert rep.fold_sharpe.shape == (len(f), 4)
    costs = rep.costs
    assert {"spread", "slippage", "commission", "swap", "gross_pnl", "net_pnl"} <= set(costs.columns)
    comb = rep.combined
    assert costs.loc["combined", "net_pnl"] == pytest.approx(
        costs.loc["combined", "gross_pnl"] - costs.loc["combined", "total_costs"] + costs.loc["combined", "swap"],
        abs=1e-4)
    assert abs(comb.reconcile()["residual"]) < 1e-4
    assert rep.provenance["data"]["bars_hash"] and rep.provenance["config_hash"] == rep.config_hash
    timing = rep.timing
    assert timing["total_s"] > 0 and "strategy_detail" in timing


def test_gbm_combined_dsr_not_significant():
    md = _md("gbm", 8000, seed=21)
    cfg = _cfg()
    rep = run_walk_forward(md, cfg, strategies=[EmaMomentum(), SlowTrend(), Contrarian(), HourNoise(),
                                                EmaMomentum(span=24)])
    c = rep.stats.loc["combined"]
    assert c["dsr"] < 0.95
    assert rep.stats.drop(index=["benchmark"])["dsr"].max() < 0.95  # nothing survives deflation
    assert rep.n_trials == 5 and "wf_ema_mom#2" in rep.stats.index


def test_executors_are_deterministic(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:6000], events=trend_md.events)
    strats = [EmaMomentum(), SlowTrend(), SpyLearner()]
    a = run_walk_forward(md, _cfg(executor="serial"), strategies=strats)
    b = run_walk_forward(md, _cfg(executor="thread", n_jobs=3), strategies=strats)
    pd.testing.assert_frame_equal(a.oos_forecasts, b.oos_forecasts)
    np.testing.assert_array_equal(a.combined.equity.to_numpy(), b.combined.equity.to_numpy())
    pd.testing.assert_frame_equal(a.stats, b.stats)


def test_fixed_weights_and_dropped_strategy(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:6000], events=trend_md.events)
    cfg = load_config({"combiner": {"method": "fixed"},
                       "strategies": [{"name": "wf_ema_mom", "weight": 3.0}, {"name": "wf_slow_trend", "weight": 1.0}],
                       "walkforward": {"train": "4M", "test": "6W", "executor": "serial", "n_boot": 200,
                                       "on_strategy_error": "drop", "min_train_bars": 200},
                       "output": {"save_results": False}}, env={})
    rep = run_walk_forward(md, cfg, strategies={"wf_ema_mom": EmaMomentum(), "wf_slow_trend": SlowTrend(),
                                                "wf_exploding": Exploding()})
    assert "wf_exploding" in rep.dropped and "fit failed on purpose" in rep.dropped["wf_exploding"]
    assert "wf_exploding" not in rep.oos_forecasts.columns
    w = rep.weights.drop(columns="fdm")
    np.testing.assert_allclose(w["wf_ema_mom"], 0.75)
    np.testing.assert_allclose(w["wf_slow_trend"], 0.25)
    cfg.walkforward.on_strategy_error = "raise"
    with pytest.raises(RuntimeError, match="wf_exploding"):
        run_walk_forward(md, cfg, strategies={"wf_ema_mom": EmaMomentum(), "wf_exploding": Exploding()})


def test_report_written_and_reloadable(tmp_path, trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    cfg = _cfg(holdout_start=str(md.bars.index[-700]))
    cfg.output.save_results = True
    cfg.output.tearsheet = True
    cfg.output.dark_charts = False
    rep = run_walk_forward(md, cfg, strategies=[EmaMomentum(), SlowTrend()], out_dir=tmp_path / "wf")
    d = tmp_path / "wf"
    assert rep.out_dir == d
    for f in ("summary.json", "config.yaml", "provenance.json", "folds.csv", "weights.csv", "stats.csv",
              "costs.csv", "fold_sharpe.csv", "oos_forecasts.parquet", "tearsheet.html",
              "books/combined/timeseries.parquet", "books/wf_ema_mom/metrics.json", "books/benchmark/meta.json",
              "holdout/stats.csv", "holdout/holdout.json", "holdout/tearsheet.html"):
        assert (d / f).exists(), f
    s = load_summary(d)
    assert s["kind"] == "walkforward" and s["n_folds"] == len(rep.folds)
    assert s["combined"]["sharpe"] == pytest.approx(rep.stats.loc["combined", "sharpe"])
    assert s["holdout"]["start"] == str(rep.holdout.start)
    assert s["provenance"]["config_hash"] == rep.config_hash
    json.dumps(s)  # strictly JSON-serialisable (no NaN literals needed)
    assert "NaN" not in (d / "summary.json").read_text()
    html = (d / "tearsheet.html").read_text()
    assert "Walk-forward folds" in html
    reloaded = load_config(d / "config.yaml", env={})
    assert reloaded.config_hash() == rep.config_hash


def test_single_backtest_out_of_sample_and_in_sample(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    cfg = _cfg()
    SpyLearner.log.clear()
    start = md.bars.index[3000]
    rep = run_single_backtest(md, cfg, strategies=[EmaMomentum(), SpyLearner(horizon=10)], start=str(start))
    fit = [e for e in SpyLearner.log if e[0] == "fit"]
    assert len(fit) == 1 and fit[0][2] < start
    assert rep.combined.equity.index[0] == start and rep.kind == "backtest"
    assert not rep.settings["in_sample"]
    ins = run_single_backtest(md, cfg, strategies=[EmaMomentum()])
    assert ins.settings["in_sample"] and any("IN-SAMPLE" in n for n in ins.notes)
    with pytest.raises(ConfigError, match="min_train_bars"):
        run_single_backtest(md, cfg, strategies=[EmaMomentum()], start=str(md.bars.index[50]))


def test_fit_quant_book_respects_fit_end(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    cfg = _cfg()
    SpyLearner.log.clear()
    book = fit_quant_book(md, cfg, strategies=[EmaMomentum(), SpyLearner(horizon=10)], fit_end=4000, gap=True)
    fit = [e for e in SpyLearner.log if e[0] == "fit"][0]
    assert fit[2] == md.bars.index[4000 - 10 - 6 - 1]  # purge (label horizon) + embargo before fit_end
    assert book.fit_end == 4000 and book.train_end == fit[2]
    assert book.signals.index.equals(md.bars.index) and book.combined.abs().max() <= 1.0
    live = fit_quant_book(md, cfg, strategies=[EmaMomentum()], gap=False)
    assert live.train_end == md.bars.index[-1]


def test_single_backtest_excludes_holdout_by_default(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    hs = md.bars.index[4500]
    cfg = _cfg(holdout_start=str(hs))
    rep = run_single_backtest(md, cfg, strategies=[EmaMomentum()], start=str(md.bars.index[3000]))
    assert rep.combined.equity.index[-1] < hs
    assert any("excluded" in n for n in rep.notes)
    full = run_single_backtest(md, cfg, strategies=[EmaMomentum()], start=str(md.bars.index[3000]),
                               include_holdout=True)
    assert full.combined.equity.index[-1] == md.bars.index[-1]
    assert any("no longer an untouched holdout" in n for n in full.notes)


class Memorizer(Strategy):
    """Trainable overfitter: memorises the sign of each TRAINING bar's next return (perfect
    in-sample, pure noise out of sample). Legal under the fit contract (it only reads the
    training slice) - exactly the kind of model whose in-sample forecasts must not drive the
    combiner weights."""

    name = "wf_memorizer"
    trainable = True

    def fit(self, md, features=None):
        r = np.log(md.bars["close"]).diff().shift(-1)
        self.table_ = np.sign(r).dropna().to_dict()
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        idx = md.bars.index
        noise = np.where((idx.hour.to_numpy() * 7919 + idx.dayofyear.to_numpy() * 31) % 11 > 5, 0.9, -0.9)
        vals = [0.9 * self.table_[t] if t in self.table_ else noise[i] for i, t in enumerate(idx)]
        return self._finalize(pd.Series(vals, index=idx), idx)


def test_combiner_oos_mode_resists_in_sample_overfitting(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:7000], events=trend_md.events)
    strats = [EmaMomentum(), EmaMomentum(span=16), Memorizer()]
    train_mode = run_walk_forward(md, _cfg(combiner_fit="train"), strategies=strats)
    oos_mode = run_walk_forward(md, _cfg(), strategies=strats)
    w_train = train_mode.weights["wf_memorizer"]
    w_oos = oos_mode.weights["wf_memorizer"]
    assert (w_train >= 0.39).all()                        # in-sample fit: capped at max_weight every fold
    later = oos_mode.folds["combiner_basis"].str.startswith("oos_history").to_numpy()
    assert later.sum() >= 3
    # OOS history exposes it: it only keeps the 0.2 that max_weight=0.4 forces onto the third strategy
    assert (w_oos[later] <= 0.2 + 1e-9).all()
    assert oos_mode.stats.loc["combined", "sharpe"] > train_mode.stats.loc["combined", "sharpe"]
    assert any("IN-SAMPLE" in n for n in train_mode.notes)
    assert oos_mode.folds["combiner_basis"].iloc[0].startswith("equal")


def test_oos_combiner_uses_only_earlier_folds(trend_md):
    from aurum.portfolio.combiner import ForecastCombiner

    md = MarketData(bars=trend_md.bars.iloc[:7000], events=trend_md.events)
    cfg = _cfg(combiner_min_obs=300)
    rep = run_walk_forward(md, cfg, strategies=[EmaMomentum(), EmaMomentum(span=16), SlowTrend()])
    oos = rep.oos_forecasts
    keys = ["wf_ema_mom", "wf_ema_mom#2", "wf_slow_trend"]
    window = rep.settings["train"]
    for _, row in rep.folds.iterrows():
        if not row["combiner_basis"].startswith("oos_history"):
            continue
        hist = oos.loc[oos.index < row["test_start"], keys].iloc[-window:]
        ref = ForecastCombiner().fit(hist, md.bars["close"])
        np.testing.assert_allclose(rep.weights.loc[row["fold"], keys].to_numpy(dtype=float),
                                   ref.weights_.to_numpy(), atol=1e-12)
        assert rep.weights.loc[row["fold"], "fdm"] == pytest.approx(ref.fdm_)


def test_fit_quant_book_combiner_from_oos_history(trend_md):
    from aurum.portfolio.combiner import ForecastCombiner

    md = MarketData(bars=trend_md.bars.iloc[:7000], events=trend_md.events)
    cfg = _cfg(combiner_min_obs=300)
    strats = [EmaMomentum(), SlowTrend()]
    rep = run_walk_forward(md, cfg, strategies=strats)
    oos = rep.oos_forecasts
    fit_end = 6500
    book = fit_quant_book(md, cfg, strategies=strats, fit_end=fit_end, oos_forecasts=oos)
    keys = ["wf_ema_mom", "wf_slow_trend"]
    hist = oos.loc[oos.index < md.bars.index[fit_end], keys].iloc[-rep.settings["train"]:]
    ref = ForecastCombiner().fit(hist, md.bars["close"])
    np.testing.assert_allclose(book.combiner.weights_.to_numpy(), ref.weights_.to_numpy(), atol=1e-12)
    eq = fit_quant_book(md, cfg, strategies=strats, fit_end=fit_end)   # no history -> equal weights
    np.testing.assert_allclose(eq.combiner.weights_.to_numpy(), [0.5, 0.5])
    with pytest.raises(ConfigError, match="lack strategies"):
        fit_quant_book(md, cfg, strategies=[EmaMomentum(), Contrarian()], oos_forecasts=oos)


# ---- adversarial review: end-to-end point-in-time checks ----------------------------------------------
class FeatureRidge(Strategy):
    """Trainable consumer of the walk-forward's FeaturePipeline features: ridge regression of
    the next bar's vol-normalised return on the fold's (train-fitted, scaled) features."""

    name = "wf_feature_ridge"
    trainable = True
    uses_features = True

    def fit(self, md, features=None):
        assert features is not None and features.index.equals(md.bars.index)
        r = np.log(md.bars["close"]).diff()
        y = (r / r.ewm(span=48, adjust=False, min_periods=24).std()).shift(-1)  # label inside the slice only
        x = features.fillna(0.0)
        ok = y.notna() & features.notna().all(axis=1)
        a, b = x.loc[ok].to_numpy(), y.loc[ok].to_numpy()
        self.beta_ = np.linalg.solve(a.T @ a + 10.0 * np.eye(a.shape[1]), a.T @ b)
        self.cols_ = list(features.columns)
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        x = features[self.cols_].fillna(0.0).to_numpy()
        return self._finalize(pd.Series(np.tanh(3.0 * x @ self.beta_), index=md.bars.index), md.bars.index)


def _perturb_from(md: MarketData, k: int, seed: int = 99) -> MarketData:
    """Same timestamps; every price from bar ``k`` on follows a different random path."""
    bars = md.bars.copy()
    rng = np.random.default_rng(seed)
    f = np.ones(len(bars))
    f[k:] = np.exp(np.cumsum(0.01 * rng.standard_normal(len(bars) - k)))
    for c in ("open", "high", "low", "close"):
        bars[c] = bars[c].to_numpy() * f
    bars.attrs = dict(md.bars.attrs)
    return MarketData(bars=bars, macro=md.macro, events=md.events)


def _pit_cfg(**wf):
    cfg = _cfg(**wf)
    cfg.features.enabled = True
    cfg.features.groups = ["returns", "volatility"]
    return cfg


def _pit_strats():
    return [EmaMomentum(), SlowTrend(), SpyLearner(horizon=6), FeatureRidge()]


def test_future_prices_never_change_earlier_oos_outputs(trend_md):
    """Changing prices from bar k on must leave every stitched OOS forecast (per strategy AND
    combined, including fitted features/models/combiner weights) and the combined equity
    before k bit-identical: fits, feature scalers, combiner weights and generation are all
    point-in-time."""
    md = MarketData(bars=trend_md.bars.iloc[:6500], events=trend_md.events)
    cfg = _pit_cfg()
    base = run_walk_forward(md, cfg, strategies=_pit_strats())
    folds = base.folds
    assert len(folds) >= 4
    k = md.bars.index.get_loc(folds["test_start"].iloc[3]) + 100   # inside the 4th test block
    pert = run_walk_forward(_perturb_from(md, k), cfg, strategies=_pit_strats())
    cut = md.bars.index[k]
    a = base.oos_forecasts.loc[base.oos_forecasts.index < cut]
    b = pert.oos_forecasts.loc[pert.oos_forecasts.index < cut]
    pd.testing.assert_frame_equal(a, b)
    assert not base.oos_forecasts.loc[base.oos_forecasts.index >= cut, "wf_feature_ridge"].equals(
        pert.oos_forecasts.loc[pert.oos_forecasts.index >= cut, "wf_feature_ridge"])   # the test has teeth
    pd.testing.assert_frame_equal(base.weights.iloc[:4], pert.weights.iloc[:4])
    ea = base.combined.equity.loc[base.combined.equity.index < cut]
    eb = pert.combined.equity.loc[pert.combined.equity.index < cut]
    np.testing.assert_array_equal(ea.to_numpy(), eb.to_numpy())


def test_holdout_prices_never_reach_research_outputs(trend_md):
    """The holdout is physically excluded: perturbing ONLY holdout prices leaves every research
    output (OOS forecasts, weights, books, statistics, PBO) bit-identical."""
    md = MarketData(bars=trend_md.bars.iloc[:6500], events=trend_md.events)
    hs = md.bars.index[5500]
    cfg = _pit_cfg(holdout_start=str(hs))
    base = run_walk_forward(md, cfg, strategies=_pit_strats())
    pert = run_walk_forward(_perturb_from(md, 5500), cfg, strategies=_pit_strats())
    pd.testing.assert_frame_equal(base.oos_forecasts, pert.oos_forecasts)
    pd.testing.assert_frame_equal(base.weights, pert.weights)
    np.testing.assert_array_equal(base.combined.equity.to_numpy(), pert.combined.equity.to_numpy())
    pd.testing.assert_frame_equal(base.stats, pert.stats)
    assert base.pbo.pbo == pert.pbo.pbo
    # ... while the holdout itself does change (evaluated on the perturbed prices)
    assert base.holdout is not None and pert.holdout is not None
    assert not np.array_equal(base.holdout.combined.equity.to_numpy(), pert.holdout.combined.equity.to_numpy())


def test_fold_returns_are_attributed_to_the_folds_own_decisions(trend_report):
    """A decision at the close of t earns bar t+1: fold k's return runs from the equity at the
    close of its first test bar to the close of the bar after its last test bar, so fold
    returns chain exactly to the total OOS return (no one-bar shift between folds)."""
    eq = trend_report.combined.equity
    idx = eq.index
    total = 1.0
    for _, row in trend_report.folds.iterrows():
        i0 = idx.get_loc(row["test_start"])
        i1 = min(idx.get_loc(row["test_end"]) + 1, len(idx) - 1)
        assert row["oos_return"] == pytest.approx(eq.iloc[i1] / eq.iloc[i0] - 1.0, rel=1e-12, abs=1e-15)
        total *= 1.0 + row["oos_return"]
    assert total - 1.0 == pytest.approx(eq.iloc[-1] / eq.iloc[0] - 1.0, rel=1e-10)


def test_zero_purge_and_embargo_are_valid_everywhere(trend_md):
    """`purge: 0` / `embargo: 0` pass validation, so they must also run (they used to crash in
    duration_to_bars, which rejects zero-length windows)."""
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    cfg = _cfg(purge=0, embargo=0)
    plans, _, st = plan_folds(md.bars.index, cfg, {"m": EmaMomentum()})
    assert st["purge"] == 0 and st["embargo"] == 0 and plans[0].test_start == plans[0].train_end
    _, _, st = plan_folds(md.bars.index, cfg, {"spy": SpyLearner(horizon=7)})
    assert st["purge"] == 7                                   # still raised to the label horizon
    book = fit_quant_book(md, cfg, strategies=[EmaMomentum()], fit_end=4000)
    assert book.train_end == md.bars.index[3999]
    rep = run_single_backtest(md, cfg, strategies=[EmaMomentum()], start=str(md.bars.index[3000]))
    assert rep.settings["purge"] == 0 and rep.settings["embargo"] == 0


def test_single_backtest_honours_configured_purge(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    SpyLearner.log.clear()
    cfg = _cfg(purge=40, embargo=6)
    start = md.bars.index[3000]
    rep = run_single_backtest(md, cfg, strategies=[EmaMomentum(), SpyLearner(horizon=10)], start=str(start))
    fit = [e for e in SpyLearner.log if e[0] == "fit"][0]
    assert rep.settings["purge"] == 40                        # config purge > label horizon: honoured
    assert fit[2] == md.bars.index[3000 - 40 - 6 - 1]


def test_fit_quant_book_enforces_min_train_bars(trend_md):
    """The desk replay / live artifact must never fit (and trade) a book on a sliver of data:
    the same min_train_bars floor as every walk-forward fold applies."""
    md = MarketData(bars=trend_md.bars.iloc[:3000], events=trend_md.events)
    cfg = _cfg(min_train_bars=500)
    with pytest.raises(ConfigError, match="min_train_bars"):
        fit_quant_book(md, cfg, strategies=[SpyLearner()], fit_end=300)
    with pytest.raises(ConfigError, match="outside the data"):
        fit_quant_book(md, cfg, strategies=[SpyLearner()], fit_end=10_000)
    assert fit_quant_book(md, cfg, strategies=[SpyLearner()], fit_end=900).train_end < md.bars.index[900]


def test_process_executor_matches_serial_with_registry_strategies():
    """Spawned worker processes (registry strategies, market data shipped once per worker)
    give bit-identical results to the serial path."""
    bars = make_synthetic_bars(5000, "H1", seed=4, model="trend", regime_params={"phi": 0.12})
    md = MarketData(bars=bars, events=make_synthetic_events(bars.index[0], bars.index[-1]))

    def cfg(**kw):
        return load_config({"walkforward": {"train": "4M", "test": "6W", "embargo": 6, "n_boot": 200,
                                            "pbo_splits": 8, "min_train_bars": 200, **kw},
                            "strategies": [{"name": "tsmom"}, {"name": "ema_cross"},
                                           {"name": "intraday_seasonality"}],
                            "output": {"save_results": False, "tearsheet": False},
                            "risk": {"research": {"max_spread": None}}}, env={})

    a = run_walk_forward(md, cfg(executor="serial"))
    b = run_walk_forward(md, cfg(executor="process", n_jobs=2))
    assert b.settings["executor"] == "process"
    pd.testing.assert_frame_equal(a.oos_forecasts, b.oos_forecasts)
    np.testing.assert_array_equal(a.combined.equity.to_numpy(), b.combined.equity.to_numpy())
    pd.testing.assert_frame_equal(a.stats, b.stats)


def test_single_backtest_start_inside_holdout_is_refused(trend_md):
    md = MarketData(bars=trend_md.bars.iloc[:5000], events=trend_md.events)
    cfg = _cfg(holdout_start=str(md.bars.index[4500]))
    with pytest.raises(ConfigError, match="inside the walk-forward holdout"):
        run_single_backtest(md, cfg, strategies=[EmaMomentum()], start=str(md.bars.index[4600]))
    rep = run_single_backtest(md, cfg, strategies=[EmaMomentum()], start=str(md.bars.index[4600]),
                              include_holdout=True)
    assert rep.combined.equity.index[0] == md.bars.index[4600]


def test_default_run_dirs_never_collide(tmp_path):
    from aurum.research.walkforward import _default_out_dir

    cfg = _cfg()
    cfg.output.dir = str(tmp_path)
    a = _default_out_dir(cfg, "walkforward")
    a.mkdir()
    b = _default_out_dir(cfg, "walkforward")
    assert b != a and not b.exists()
