"""v2.1 fix: walk-forward statistics date every equity mark by its bar's CLOSE.

``aurum.backtest.metrics.daily_returns`` dates bars by OPEN time unless ``bar_duration`` is
given. The walk-forward statistics (``book_statistics``, the per-fold Sharpes, the DSR trial
Sharpes and the PBO matrix) used to call it without one, so for New-York-close D1 bars (open
22:00 UTC the evening before each session) the Sunday-22:00 and Monday-22:00 bars both landed
on Monday: four "days" a week and a daily Sharpe ~10% too large in magnitude. H1 bars never
straddle midnight, so their dates (and every published H1 number) must not change.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.metrics import daily_returns
from aurum.backtest.result import BacktestResult
from aurum.core.config import load_config
from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.research import walkforward as wf
from aurum.research.stats import sharpe, sharpe_summary
from aurum.strategies.base import Strategy


# ---- helpers -------------------------------------------------------------------------------------
def _ny_close_bars(n: int, seed: int = 11) -> pd.DataFrame:
    """Canonical D1 bars of a New-York-close server: session d's bar opens at 22:00 UTC the
    evening before (Sun..Thu opens for the Mon..Fri sessions) and closes 22:00 UTC on d."""
    src = make_synthetic_bars(n, "D1", seed=seed, model="trend", regime_params={"phi": 0.15})
    sessions = pd.bdate_range("2016-01-04", periods=n, tz="UTC")
    df = src[["open", "high", "low", "close", "spread"]].copy()
    df.index = sessions - pd.Timedelta(hours=2)
    return make_bars(df, "D1")


def _result(equity: pd.Series, timeframe: str | None) -> BacktestResult:
    eq = equity.astype(float)
    empty = pd.DataFrame(index=eq.index)
    return BacktestResult(equity=eq, returns=eq.pct_change().fillna(0.0), positions=pd.Series(0.0, index=eq.index),
                          costs=empty.assign(spread=0.0, slippage=0.0, commission=0.0, swap=0.0),
                          trades=pd.DataFrame(), fills=pd.DataFrame(),
                          meta={"timeframe": timeframe} if timeframe else {})


def _equity(index: pd.DatetimeIndex, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(1e5 * np.cumprod(1.0 + rng.normal(2e-4, 0.01, len(index))), index=index)


class _Mom(Strategy):
    name = "dd_mom"

    @property
    def warmup_bars(self) -> int:
        return 40

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        s = r.ewm(span=5, adjust=False).mean() / r.ewm(span=30, adjust=False, min_periods=20).std()
        return self._finalize((1.5 * s).clip(-1, 1), md.bars.index)


class _Slow(Strategy):
    name = "dd_slow"

    @property
    def warmup_bars(self) -> int:
        return 30

    def generate(self, md, features=None):
        c = np.log(md.bars["close"])
        return self._finalize(np.sign(c - c.shift(30)) * 0.8, md.bars.index)


def _cfg(timeframe: str, **wfo):
    base = {"embargo": 1, "executor": "serial", "n_boot": 100, "pbo_splits": 4, "min_train_bars": 100,
            "combiner_min_obs": 120, "holdout_start": None}
    base.update(wfo)
    return load_config({"data": {"timeframe": timeframe}, "walkforward": base, "seed": 5,
                        "output": {"save_results": False, "tearsheet": False},
                        "risk": {"research": {"max_spread": None}}}, env={})


# ---- the dating itself -----------------------------------------------------------------------------
def test_ny_close_d1_gives_five_daily_returns_per_week_on_close_dates():
    bars = _ny_close_bars(60)
    assert set(bars.index.hour) == {22} and set(bars.index.weekday) == {6, 0, 1, 2, 3}
    eq = _equity(bars.index)
    dur = wf._bars_duration(bars)
    assert dur == pd.Timedelta(days=1)
    d = daily_returns(eq, bar_duration=dur)
    assert len(d) == len(eq) == 60                               # one return per session
    weeks = pd.Series(1, index=d.index).groupby(d.index.tz_localize(None).to_period("W")).sum()
    assert (weeks == 5).all()                                    # 12 full weeks x 5 days
    sessions = (bars.index + pd.Timedelta(hours=2)).normalize()  # the date each bar CLOSES on
    assert d.index.equals(pd.DatetimeIndex(sessions).as_unit(d.index.unit))
    assert list(d.index.weekday[:5]) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(d.to_numpy()[1:], eq.pct_change().to_numpy()[1:], rtol=1e-12)
    legacy = daily_returns(eq)                                   # the bug: open-time dates
    assert len(legacy) == 48                                     # Sunday + Monday bars merged: 4/week


def test_bar_duration_resolution_order():
    bars = _ny_close_bars(30)
    assert wf._bars_duration(bars) == pd.Timedelta(days=1)                  # attrs timeframe
    no_attrs = bars.copy()
    no_attrs.attrs.clear()
    assert wf._bars_duration(no_attrs, "H1") == pd.Timedelta(days=1)        # available_at beats config
    bare = no_attrs.drop(columns="available_at")
    assert wf._bars_duration(bare, "D1") == pd.Timedelta(days=1)            # config timeframe
    assert wf._bars_duration(bare) == pd.Timedelta(days=1)                  # modal index spacing
    assert wf._bars_duration(None, "H1") == pd.Timedelta(hours=1)
    assert wf._bars_duration(bare.iloc[:2]) is None                         # nothing to go on
    h1 = make_synthetic_bars(300, "H1", seed=1)
    assert wf._bars_duration(h1, "D1") == pd.Timedelta(hours=1)             # the data wins
    assert wf._result_duration(_result(_equity(bars.index), "D1")) == pd.Timedelta(days=1)
    assert wf._result_duration(_result(_equity(bars.index), None)) == pd.Timedelta(days=1)


def test_book_statistics_dates_by_bar_close():
    bars = _ny_close_bars(300)
    eq = _equity(bars.index, seed=8)
    s = wf.book_statistics(_result(eq, "D1"), n_boot=50)
    assert s["n_days"] == 300
    ref = sharpe_summary(daily_returns(eq, bar_duration="1D").dropna().to_numpy(), 252.0, n_trials=1,
                         n_boot=50, seed=0)
    for k in ("sharpe", "psr", "ci_lower", "ci_upper", "sharpe_se"):
        assert s[k] == ref[k]
    legacy = sharpe(daily_returns(eq).dropna().to_numpy(), 252.0)
    assert abs(legacy) > abs(s["sharpe"]) * 1.04                 # the ~10% overstatement is gone
    # an explicit duration wins over the meta (open-time dating reproduces the old numbers)
    old = wf.book_statistics(_result(eq, "D1"), n_boot=50, bar_duration=pd.Timedelta(hours=1))
    assert old["n_days"] == 240 and old["sharpe"] == pytest.approx(legacy, rel=1e-12)


# ---- H1 is untouched -----------------------------------------------------------------------------
def test_h1_dates_and_statistics_identical_to_open_time_dating():
    bars = make_synthetic_bars(3000, "H1", seed=2)
    assert wf._bars_duration(bars, "H1") == pd.Timedelta(hours=1)
    eq = _equity(bars.index, seed=5)
    pd.testing.assert_series_equal(daily_returns(eq, bar_duration="1h"), daily_returns(eq))
    s = wf.book_statistics(_result(eq, "H1"), n_boot=60, seed=4)
    ref = sharpe_summary(daily_returns(eq).dropna().to_numpy(), 252.0, n_trials=1, n_boot=60, seed=4)
    for k in ("sharpe", "psr", "dsr", "ci_lower", "ci_upper", "sr0", "skew", "kurtosis"):
        assert s[k] == ref[k] or (math.isnan(s[k]) and math.isnan(ref[k]))
    t0, t1 = bars.index[500], bars.index[1500]
    assert wf._window_sharpe(eq, t0, t1, pd.Timedelta(hours=1)) == wf._window_sharpe(eq, t0, t1)


@pytest.fixture(scope="module")
def h1_report():
    bars = make_synthetic_bars(4000, "H1", seed=4, model="trend", regime_params={"phi": 0.12})
    rep = wf.run_walk_forward(MarketData(bars=bars), _cfg("H1", train="3M", test="6W", embargo=6),
                              strategies=[_Mom(), _Slow()], write=False, holdout_ledger=False)
    return rep


def test_h1_walk_forward_statistics_equal_the_legacy_open_time_computation(h1_report):
    rep = h1_report
    books = {"combined": rep.combined, **rep.strategy_results}
    trial = [sharpe(daily_returns(r.equity).dropna().to_numpy(), 252.0) for r in rep.strategy_results.values()]
    for k, r in books.items():
        dr = daily_returns(r.equity).dropna()                     # legacy: open-time dates
        ref = sharpe_summary(dr.to_numpy(), 252.0, n_trials=rep.n_trials, n_boot=100, seed=5)
        for c in ("sharpe", "psr", "dsr", "ci_lower", "ci_upper", "sr0", "sharpe_se"):
            assert rep.stats.loc[k, c] == ref[c], (k, c)
        assert rep.stats.loc[k, "n_days"] == len(dr)
    legacy_xs = wf.book_statistics(rep.combined, n_trials=rep.n_trials, trial_sharpes=trial, n_boot=100, seed=5)
    assert rep.stats.loc["combined", "dsr_xs"] == legacy_xs["dsr_xs"]
    for _, f in rep.folds.iterrows():
        sr, ret = wf._window_sharpe(rep.combined.equity, f["test_start"], f["test_end"])  # no duration
        assert (f["oos_sharpe"] == sr or (math.isnan(sr) and math.isnan(f["oos_sharpe"]))) and f["oos_return"] == ret
    matrix = pd.DataFrame({k: daily_returns(r.equity) for k, r in rep.strategy_results.items()}).dropna()
    ref_pbo = wf._pbo(matrix, 4)
    assert rep.pbo is not None and rep.pbo.pbo == ref_pbo.pbo and rep.pbo.n_splits == ref_pbo.n_splits


# ---- the walk-forward statistics path uses the close dates ---------------------------------------
@pytest.fixture(scope="module")
def d1_report():
    bars = _ny_close_bars(900)
    rep = wf.run_walk_forward(MarketData(bars=bars), _cfg("D1", train="1Y", test="3M"),
                              strategies=[_Mom(), _Slow()], write=False, holdout_ledger=False)
    return rep


def test_d1_ny_close_walk_forward_statistics_use_close_dates(d1_report):
    rep = d1_report
    dur = pd.Timedelta(days=1)
    assert len(rep.folds) >= 4
    for k, r in {"combined": rep.combined, **rep.strategy_results, "benchmark": rep.benchmark}.items():
        n = len(r.equity)
        assert rep.stats.loc[k, "n_days"] == n                    # one daily return per NY session
        assert r.metrics["n_days"] == n                           # the same series compute_metrics uses
        dr = daily_returns(r.equity, bar_duration=dur).dropna()
        n_trials = 1 if k == "benchmark" else rep.n_trials
        ref = sharpe_summary(dr.to_numpy(), 252.0, n_trials=n_trials, n_boot=100, seed=5)
        for c in ("sharpe", "psr", "dsr", "ci_lower", "ci_upper", "sr0"):
            assert rep.stats.loc[k, c] == ref[c], (k, c)
        if math.isfinite(r.metrics["sharpe"]):
            assert rep.stats.loc[k, "sharpe"] == pytest.approx(r.metrics["sharpe"], rel=1e-9, abs=1e-12)
        assert len(daily_returns(r.equity).dropna()) < 0.85 * n   # what the old code counted
    # per-fold Sharpes and the PBO matrix
    for _, f in rep.folds.iterrows():
        sr, ret = wf._window_sharpe(rep.combined.equity, f["test_start"], f["test_end"], dur)
        assert (f["oos_sharpe"] == sr or (math.isnan(sr) and math.isnan(f["oos_sharpe"]))) and f["oos_return"] == ret
        for k, r in rep.strategy_results.items():
            fs = wf._window_sharpe(r.equity, f["test_start"], f["test_end"], dur)[0]
            got = rep.fold_sharpe.loc[f["fold"], k]
            assert got == fs or (math.isnan(fs) and math.isnan(got))
    matrix = pd.DataFrame({k: daily_returns(r.equity, bar_duration=dur)
                           for k, r in rep.strategy_results.items()}).dropna()
    assert len(matrix) == len(rep.combined.equity)
    ref_pbo = wf._pbo(matrix, 4)
    assert rep.pbo is not None and rep.pbo.pbo == ref_pbo.pbo


# ---- the other daily-return paths (tearsheet, stress, dashboard, sweep, rl) ------------------------
def test_resolve_bar_duration_prefers_timeframe_then_index():
    from aurum.backtest.metrics import resolve_bar_duration

    ny = _ny_close_bars(30).index
    assert resolve_bar_duration("D1", None) == pd.Timedelta(days=1)
    assert resolve_bar_duration("H1", ny) == pd.Timedelta(hours=1)          # the timeframe wins
    assert resolve_bar_duration(None, ny) == pd.Timedelta(days=1)           # modal spacing
    assert resolve_bar_duration("not-a-tf", ny) == pd.Timedelta(days=1)     # unknown name -> index
    assert resolve_bar_duration(None, None) is None


def test_tearsheet_daily_returns_date_ny_close_bars_by_close():
    from aurum.research.report import daily_returns_from_equity

    idx = _ny_close_bars(60).index
    eq = _equity(idx)
    close_dated = daily_returns_from_equity(eq, float(eq.iloc[0]), bar_duration="1D")
    open_dated = daily_returns_from_equity(eq, float(eq.iloc[0]))
    assert len(close_dated) == 60                                           # one return per session
    assert len(open_dated) < 60                                             # Sun+Mon opens collide
    assert set(close_dated.index.weekday) <= {0, 1, 2, 3, 4}
    # H1: close dating is identical to open dating
    h1 = pd.date_range("2020-01-06", periods=24 * 15, freq="1h", tz="UTC")
    eq1 = _equity(h1)
    pd.testing.assert_series_equal(daily_returns_from_equity(eq1, 1e5, bar_duration="1h"),
                                   daily_returns_from_equity(eq1, 1e5))


def test_tearsheet_statistics_use_close_dates_for_d1_runs():
    from aurum.research.report import tearsheet_html

    idx = _ny_close_bars(120).index
    eq = _equity(idx, seed=5)
    html_d1 = tearsheet_html(_result(eq, "D1"), n_boot=50, seed=0)
    expected = sharpe(daily_returns(eq, bar_duration="1D").dropna(), 252)
    legacy = sharpe(daily_returns(eq).dropna(), 252)
    assert not math.isclose(expected, legacy, rel_tol=1e-6)
    assert f"{expected:.2f}" in html_d1


def test_stress_run_dates_ny_close_books_by_close():
    from aurum.research.montecarlo import stress_run

    idx = _ny_close_bars(80).index
    eq = _equity(idx, seed=7)
    rep = stress_run(_result(eq, "D1"), n_paths=50, horizon=20, seed=0, load_data=False)
    assert len(rep.daily_returns) == len(daily_returns(eq, bar_duration="1D").dropna())
    assert len(rep.daily_returns) > len(daily_returns(eq).dropna())
    # a plain Series (no meta) is dated by its inferred daily spacing too
    rep2 = stress_run(eq, n_paths=50, horizon=20, seed=0, load_data=False)
    pd.testing.assert_series_equal(rep2.daily_returns, rep.daily_returns)


def test_dashboard_daily_helpers_date_by_close():
    from aurum.viz import dashboard as db

    idx = _ny_close_bars(60).index
    ts = pd.DataFrame({"equity": _equity(idx, seed=9)})
    assert len(db._daily_returns(ts)) == len(daily_returns(ts["equity"], bar_duration="1D"))
    assert len(db._daily_equity(ts)) == 60
    h1 = pd.date_range("2020-01-06", periods=24 * 10, freq="1h", tz="UTC")
    ts1 = pd.DataFrame({"equity": _equity(h1)})
    r1 = db._daily_returns(ts1)
    r0 = daily_returns(ts1["equity"])
    np.testing.assert_array_equal(r1.to_numpy(), r0.to_numpy())
