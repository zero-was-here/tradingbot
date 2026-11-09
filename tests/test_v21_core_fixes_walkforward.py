"""v2.1 core fixes - walk-forward / report: holdout-ledger looks of holdout-including backtests,
``record_holdout_look``, data-source-aware prior looks, benchmark-free ledger strategy Sharpes,
the tearsheet bootstrap settings and the attribution accessors (per-bar w_i * f_i * FDM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.core.config import load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events
from aurum.research import report as report_mod
from aurum.research.walkforward import (
    CLIP_COLUMN,
    HOLDOUT_LEDGER,
    append_holdout_ledger,
    data_source_kind,
    forecast_contributions,
    prior_holdout_looks,
    read_holdout_ledger,
    record_holdout_look,
    run_single_backtest,
    run_walk_forward,
)
from aurum.strategies.base import Strategy


class Mom(Strategy):
    name = "v21_mom"

    @property
    def warmup_bars(self) -> int:
        return 60

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        s = r.ewm(span=6, adjust=False).mean() / r.ewm(span=48, adjust=False, min_periods=24).std()
        return self._finalize((1.5 * s).clip(-1, 1), md.bars.index)


class Slow(Strategy):
    name = "v21_slow"

    @property
    def warmup_bars(self) -> int:
        return 48

    def generate(self, md, features=None):
        c = np.log(md.bars["close"])
        return self._finalize(np.sign(c - c.shift(48)) * 0.9, md.bars.index)


def _cfg(*, synthetic: bool = False, seed: int = 0, **wf):
    base = {"train": "3M", "test": "6W", "embargo": 6, "executor": "serial", "n_boot": 120, "pbo_splits": 4,
            "min_train_bars": 200, "combiner_min_obs": 300}
    base.update(wf)
    raw = {"walkforward": base, "seed": seed, "output": {"save_results": False, "tearsheet": False},
           "risk": {"research": {"max_spread": None}}}
    if synthetic:
        raw["data"] = {"synthetic": {"n": 100}}
    return load_config(raw, env={})


@pytest.fixture(scope="module")
def md() -> MarketData:
    bars = make_synthetic_bars(4500, "H1", seed=4, model="trend", regime_params={"phi": 0.12})
    return MarketData(bars=bars, events=make_synthetic_events(bars.index[0], bars.index[-1]))


@pytest.fixture(scope="module")
def wf_report(md):
    return run_walk_forward(md, _cfg(holdout_start=str(md.bars.index[-700])), strategies=[Mom(), Slow()],
                            write=False, holdout_ledger=False)


# ---- data source kind & prior looks -----------------------------------------------------------
def test_data_source_kind(md):
    assert data_source_kind(_cfg()) == "real"
    assert data_source_kind(_cfg(synthetic=True)) == "synthetic"
    bars = md.bars.copy()
    bars.attrs["source"] = "synthetic:gbm"
    assert data_source_kind(_cfg(), MarketData(bars=bars)) == "synthetic"
    assert data_source_kind(None, None) == "real"


def test_prior_looks_ignore_other_data_source_kinds_but_count_legacy_entries():
    win = {"holdout_start": "2025-01-01", "holdout_end": "2025-06-30", "symbol": "XAUUSD"}
    entries = [
        {**win, "config_hash": "a", "data_source": "synthetic", "timestamp": "t1"},
        {**win, "config_hash": "b", "data_source": "real", "timestamp": "t2"},
        {**win, "config_hash": "c", "timestamp": "t0"},           # written before the field existed
    ]
    kw = {"start": "2025-03-01", "end": "2025-04-01", "config_hash": "x", "symbol": "XAUUSD"}
    real = prior_holdout_looks(entries, data_source="real", **kw)
    assert real["prior_config_hashes"] == ["b", "c"] and real["n_prior_looks"] == 2
    syn = prior_holdout_looks(entries, data_source="synthetic", **kw)
    assert syn["prior_config_hashes"] == ["a", "c"]
    legacy = prior_holdout_looks(entries, **kw)                   # no data_source: old behaviour
    assert legacy["n_prior_configs"] == 3


# ---- record_holdout_look -------------------------------------------------------------------------
def test_record_holdout_look_fills_and_clips(tmp_path, md):
    cfg = _cfg(holdout_start="2020-06-01")
    p = record_holdout_look(tmp_path, {"start": "2020-05-01", "end": "2020-07-01", "kind": "desk_replay",
                                       "config_hash": "deskhash"}, cfg=cfg, md=md)
    assert p == tmp_path / HOLDOUT_LEDGER
    (e,) = read_holdout_ledger(p)
    assert e["kind"] == "desk_replay" and e["config_hash"] == "deskhash"
    assert pd.Timestamp(e["holdout_start"]) == pd.Timestamp("2020-06-01", tz="UTC")
    assert pd.Timestamp(e["holdout_end"]) == pd.Timestamp("2020-07-01", tz="UTC")
    assert pd.Timestamp(e["evaluated_start"]) == pd.Timestamp("2020-05-01", tz="UTC")
    assert e["symbol"] == "XAUUSD" and e["timeframe"] == cfg.data.timeframe and e["name"] == cfg.name
    assert e["data_source"] == "real" and len(e["data_hash"]) == 64 and e["timestamp"]
    # entirely before the holdout / no holdout configured: nothing recorded
    assert record_holdout_look(tmp_path, {"start": "2020-01-01", "end": "2020-02-01"}, cfg=cfg) is None
    assert record_holdout_look(tmp_path, {"start": "2020-01-01", "end": "2020-09-01"}, cfg=_cfg()) is None
    assert len(read_holdout_ledger(p)) == 1
    # explicit window and an explicit .jsonl ledger file; defaults without cfg
    led = tmp_path / "other" / "ledger.jsonl"
    assert record_holdout_look(led, {"holdout_start": "2021-01-01", "holdout_end": "2021-02-01"}) == led
    (e2,) = read_holdout_ledger(led)
    assert e2["kind"] == "look" and "config_hash" not in e2
    with pytest.raises(ValueError, match="holdout_start"):
        record_holdout_look(tmp_path, {"foo": 1})
    with pytest.raises(ValueError, match="before"):
        record_holdout_look(tmp_path, {"holdout_start": "2021-02-01", "holdout_end": "2021-01-01"})
    with pytest.raises(ValueError, match="cfg"):
        record_holdout_look(tmp_path, {"start": "2021-01-01", "end": "2021-02-01"})


def test_desk_look_counts_in_a_later_walk_forward(md, tmp_path):
    hs = str(md.bars.index[-700])
    cfg = _cfg(holdout_start=hs)
    record_holdout_look(tmp_path, {"start": hs, "end": str(md.bars.index[-1]), "kind": "desk_replay",
                                   "config_hash": f"{cfg.config_hash()}:desk"}, cfg=cfg, md=md)
    r = run_walk_forward(md, cfg, strategies=[Mom(), Slow()], write=False, holdout_ledger=tmp_path / HOLDOUT_LEDGER)
    assert r.holdout.prior_looks["n_prior_configs"] == 1 and r.n_trials == 3


# ---- walk-forward ledger entry -------------------------------------------------------------------
def test_walk_forward_ledger_excludes_benchmark_and_records_data_source(md, tmp_path):
    hs = str(md.bars.index[-700])
    led = tmp_path / HOLDOUT_LEDGER
    r = run_walk_forward(md, _cfg(holdout_start=hs), strategies=[Mom(), Slow()], write=False, holdout_ledger=led)
    (e,) = read_holdout_ledger(led)
    assert "benchmark" in r.holdout.stats.index
    assert set(e["metrics"]["strategy_sharpe"]) == {"v21_mom", "v21_slow"}
    assert e["metrics"]["benchmark_sharpe"] == pytest.approx(r.holdout.stats.loc["benchmark", "sharpe"])
    assert e["data_source"] == "real" and e["kind"] == "walkforward"
    # a synthetic-config run on the same window ignores that REAL look (and vice versa)
    syn = run_walk_forward(md, _cfg(holdout_start=hs, synthetic=True, embargo=12), strategies=[Mom(), Slow()],
                           write=False, holdout_ledger=led)
    assert syn.holdout.prior_looks["n_prior_looks"] == 0 and syn.n_trials == 2
    assert read_holdout_ledger(led)[-1]["data_source"] == "synthetic"
    real2 = run_walk_forward(md, _cfg(holdout_start=hs, embargo=18), strategies=[Mom(), Slow()], write=False,
                             holdout_ledger=led)
    assert real2.holdout.prior_looks["n_prior_configs"] == 1 and real2.n_trials == 3


# ---- run_single_backtest(include_holdout=True) is a look ------------------------------------------
def test_holdout_including_backtest_is_recorded_and_counted(md, tmp_path):
    hs = str(md.bars.index[-700])
    start = str(md.bars.index[-1200])
    led = tmp_path / HOLDOUT_LEDGER
    strats = [Mom(), Slow()]
    # excluding the holdout: nothing recorded
    run_single_backtest(md, _cfg(holdout_start=hs), strategies=strats, start=start, write=False, holdout_ledger=led)
    assert read_holdout_ledger(led) == []
    # a window that ends before the holdout is not a look either
    run_single_backtest(md, _cfg(holdout_start=hs), strategies=strats, start=start, end=str(md.bars.index[-800]),
                        write=False, include_holdout=True, holdout_ledger=led)
    assert read_holdout_ledger(led) == []
    # an earlier look by ANOTHER config raises n_trials of the backtest
    append_holdout_ledger(led, {"holdout_start": hs, "holdout_end": str(md.bars.index[-1]), "config_hash": "zzz",
                                "symbol": "XAUUSD", "data_source": "real", "timestamp": "t"})
    rep = run_single_backtest(md, _cfg(holdout_start=hs), strategies=strats, start=start, write=False,
                              include_holdout=True, holdout_ledger=led)
    assert rep.n_trials == 3 and rep.settings["n_trials_prior_looks"] == 1
    assert any("WARNING" in n and "evaluated before" in n for n in rep.notes)
    entries = read_holdout_ledger(led)
    assert len(entries) == 2
    e = entries[-1]
    assert e["kind"] == "backtest" and e["in_sample"] is False and e["metrics_window"] == "evaluated"
    assert pd.Timestamp(e["holdout_start"]) == md.bars.index[-700]
    assert pd.Timestamp(e["holdout_end"]) == md.bars.index[-1] and e["n_holdout_bars"] == 700
    assert pd.Timestamp(e["evaluated_start"]) == md.bars.index[-1200]
    assert set(e["metrics"]["strategy_sharpe"]) == {"v21_mom", "v21_slow"}
    assert e["metrics"]["combined"]["sharpe"] == pytest.approx(rep.stats.loc["combined", "sharpe"])
    assert e["prior_looks"]["n_prior_configs"] == 1 and e["config_hash"] == rep.config_hash
    # ... and a later walk-forward on the window counts the backtest's config as a prior look
    wf = run_walk_forward(md, _cfg(holdout_start=hs, embargo=12), strategies=strats, write=False,
                          holdout_ledger=led)
    assert wf.holdout.prior_looks["n_prior_configs"] == 2


def test_holdout_backtest_default_ledger_and_unrecorded_note(md, tmp_path):
    hs = str(md.bars.index[-700])
    strats = [Mom(), Slow()]
    rep = run_single_backtest(md, _cfg(holdout_start=hs), strategies=strats, start=str(md.bars.index[-1200]),
                              write=False, include_holdout=True)
    assert any("NOT recorded" in n for n in rep.notes)
    run_dir = tmp_path / "runs" / "bt"
    run_single_backtest(md, _cfg(holdout_start=hs), strategies=strats, start=str(md.bars.index[-1200]),
                        out_dir=run_dir, write=True, include_holdout=True)
    (e,) = read_holdout_ledger(tmp_path / "runs" / HOLDOUT_LEDGER)   # next to the run directories
    assert e["run_dir"] == str(run_dir) and e["kind"] == "backtest"


# ---- tearsheet bootstrap settings ----------------------------------------------------------------
def test_bootstrap_settings_resolution():
    f = report_mod._bootstrap_settings
    assert f(None, None, None) == (report_mod.DEFAULT_N_BOOT, report_mod.DEFAULT_SEED)
    assert f({"walkforward": {"n_boot": 321}, "seed": 9}, None, None) == (321, 9)
    assert f({"walkforward": {"n_boot": 321}, "seed": 9}, 50, 1) == (50, 1)
    assert f({"walkforward": {"n_boot": None}, "seed": True}, None, None) == (1000, 0)


def test_tearsheets_use_walkforward_n_boot_and_config_seed(md, tmp_path, monkeypatch):
    seen: list[tuple] = []

    def spy(result, path, **kw):
        seen.append((kw.get("n_boot"), kw.get("seed"), kw.get("title", "")))
        return path

    monkeypatch.setattr(report_mod, "write_tearsheet", spy)
    cfg = _cfg(holdout_start=str(md.bars.index[-700]), n_boot=137, seed=11)
    cfg.output.tearsheet = True
    run_walk_forward(md, cfg, strategies=[Mom(), Slow()], out_dir=tmp_path / "r", write=True, holdout_ledger=False)
    assert len(seen) == 2 and all(s[:2] == (137, 11) for s in seen)
    assert any("HOLDOUT" in s[2] for s in seen)


def test_tearsheet_html_reads_bootstrap_from_extra_config(wf_report, monkeypatch):
    calls: list[dict] = []
    orig = report_mod.rstats.sharpe_summary

    def spy(*a, **kw):
        calls.append(kw)
        return orig(*a, **kw)

    monkeypatch.setattr(report_mod.rstats, "sharpe_summary", spy)
    report_mod.tearsheet_html(wf_report.combined, extra={"config": {"walkforward": {"n_boot": 77}, "seed": 5}},
                              dark_charts=False)
    assert calls and calls[-1]["n_boot"] == 77 and calls[-1]["seed"] == 5
    report_mod.tearsheet_html(wf_report.combined, dark_charts=False, n_boot=60, seed=2,
                              extra={"config": {"walkforward": {"n_boot": 77}, "seed": 5}})
    assert calls[-1]["n_boot"] == 60 and calls[-1]["seed"] == 2


# ---- attribution accessors -----------------------------------------------------------------------
def test_contributions_reproduce_the_combined_forecast(wf_report):
    r = wf_report
    keys = r.strategy_keys
    assert keys == ["v21_mom", "v21_slow"]
    f = r.strategy_forecasts()
    assert list(f.columns) == keys and f.index.equals(r.oos_forecasts.index)
    fw, fdm = r.fold_weights(), r.fold_fdm()
    assert list(fw.columns) == keys and list(fw.index) == list(r.folds["fold"].astype(str))
    np.testing.assert_allclose(fdm.to_numpy(), r.folds["fdm"].to_numpy())
    bw = r.bar_weights()
    assert list(bw.columns) == [*keys, "fdm"] and bw.index.equals(f.index)
    c = r.contributions()
    assert list(c.columns) == keys
    # explicit formula: FDM_k * w_{k,i} * f_i[t]
    np.testing.assert_allclose(c.to_numpy(), f.to_numpy() * bw[keys].to_numpy() * bw[["fdm"]].to_numpy())
    combined = r.oos_forecasts["combined"].to_numpy()
    np.testing.assert_allclose(np.clip(c.sum(axis=1).to_numpy(), -1, 1), combined, atol=1e-12)
    cr = r.contributions(residual=True)
    assert list(cr.columns) == [*keys, CLIP_COLUMN]
    np.testing.assert_allclose(cr.sum(axis=1).to_numpy(), combined, atol=1e-12)
    assert (cr[CLIP_COLUMN].abs() > 0).sum() <= (np.abs(combined) >= 1 - 1e-12).sum()
    # holdout: one fold's weights
    h = r.holdout
    hc = h.contributions(residual=True)
    np.testing.assert_allclose(hc.sum(axis=1).to_numpy(), h.forecasts["combined"].to_numpy(), atol=1e-12)
    assert list(h.strategy_forecasts().columns) == keys


def test_contributions_from_saved_run_files(md, tmp_path):
    r = run_walk_forward(md, _cfg(), strategies=[Mom(), Slow()], out_dir=tmp_path / "run", write=True)
    oos = pd.read_parquet(tmp_path / "run" / "oos_forecasts.parquet")
    weights = pd.read_csv(tmp_path / "run" / "weights.csv", index_col="fold")
    from_files = forecast_contributions(oos, weights)
    pd.testing.assert_frame_equal(from_files, r.contributions(), check_freq=False)


def test_forecast_contributions_single_fold_and_errors():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    f = pd.DataFrame({"a": [1.0, -0.5, np.nan, 2.0], "b": [0.5, 0.5, 0.5, 0.5]}, index=idx)
    c = forecast_contributions(f, {"a": 0.6, "b": 0.4}, fdm=1.5)
    np.testing.assert_allclose(c["a"].to_numpy(), [0.9, -0.45, 0.0, 0.9])   # clipped to 1, NaN -> 0
    np.testing.assert_allclose(c["b"].to_numpy(), [0.3] * 4)
    c2 = forecast_contributions(f, pd.Series({"a": 0.6, "b": 0.4, "fdm": 1.5}))
    pd.testing.assert_frame_equal(c, c2)
    with pytest.raises(ValueError, match="fdm"):
        forecast_contributions(f, {"a": 1.0})
    with pytest.raises(KeyError, match="lack"):
        forecast_contributions(f, {"zz": 1.0}, fdm=1.0)
    with pytest.raises(KeyError, match="fold"):
        forecast_contributions(f, pd.DataFrame({"a": [1.0], "fdm": [1.0]}, index=["0"]))
    g = f.assign(fold=["0", "0", "1", "1"])
    with pytest.raises(KeyError, match="folds without weights"):
        forecast_contributions(g, pd.DataFrame({"a": [1.0], "b": [0.0], "fdm": [1.0]}, index=["0"]))
