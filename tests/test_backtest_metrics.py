"""Metrics verified against simple, hand-checkable series."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.metrics import (
    compute_metrics,
    daily_returns,
    drawdown_series,
    max_drawdown_duration_days,
    trading_dates,
)
from aurum.backtest.result import BacktestResult


def _result(equity: pd.Series, trades: pd.DataFrame | None = None,
            positions: pd.Series | None = None) -> BacktestResult:
    idx = equity.index
    rets = equity.pct_change().fillna(0.0)
    costs = pd.DataFrame({"spread": 0.0, "slippage": 0.0, "commission": 0.0, "swap": 0.0}, index=idx)
    return BacktestResult(
        equity=equity, returns=rets,
        positions=positions if positions is not None else pd.Series(0.0, index=idx),
        costs=costs,
        trades=trades if trades is not None else pd.DataFrame(columns=["pnl", "bars_held"]),
        fills=pd.DataFrame(columns=["lots"]),
    )


def _weekday_daily_equity(daily_rets: list[float], start: str = "2024-01-08") -> pd.Series:
    """One D1-like bar per business day; equity[0] is the initial (no-PnL) bar."""
    idx = pd.bdate_range(pd.Timestamp(start, tz="UTC"), periods=len(daily_rets) + 1)
    eq = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(daily_rets)])
    return pd.Series(eq, index=idx)


def test_drawdown_and_max_drawdown() -> None:
    eq = pd.Series([100.0, 120.0, 90.0, 130.0, 117.0],
                   index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"))
    np.testing.assert_allclose(drawdown_series(eq).to_numpy(), [0, 0, -0.25, 0, -0.1])
    m = compute_metrics(_result(eq))
    assert m["max_drawdown"] == pytest.approx(-0.25)


def test_daily_sharpe_sortino_vol_against_formula() -> None:
    r = [0.01, -0.005, 0.002, 0.004, -0.012, 0.007, 0.0, 0.003, -0.002, 0.006]
    eq = _weekday_daily_equity(r)
    res = _result(eq)
    m = compute_metrics(res)
    d = daily_returns(eq).to_numpy()
    # one bar per day: the first day is the flat decision bar (return 0 by convention)
    assert len(d) == len(r) + 1 and d[0] == 0.0
    arr = np.asarray(d)
    sharpe = arr.mean() / arr.std(ddof=1) * math.sqrt(252)
    assert m["sharpe"] == pytest.approx(sharpe)
    assert m["ann_vol"] == pytest.approx(arr.std(ddof=1) * math.sqrt(252))
    dd = math.sqrt(np.mean(np.minimum(arr, 0) ** 2))
    assert m["sortino"] == pytest.approx(arr.mean() / dd * math.sqrt(252))
    assert m["best_day"] == pytest.approx(0.01)
    assert m["worst_day"] == pytest.approx(-0.012)
    assert m["total_return"] == pytest.approx(np.prod(1 + np.asarray(r)) - 1)


def test_intraday_bars_aggregate_to_daily_and_weekends_fold() -> None:
    # H1 bars Sunday 22:00 .. Tuesday 23:00; equity moves every bar
    idx = pd.date_range("2024-01-07 22:00", "2024-01-09 23:00", freq="1h", tz="UTC")
    eq = pd.Series(100.0 + np.arange(len(idx), dtype=float), index=idx)
    d = daily_returns(eq)
    # Sunday folds into Monday: only Monday and Tuesday remain
    assert list(d.index.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-09"]
    mon_close = float(eq.loc[pd.Timestamp("2024-01-08 23:00", tz="UTC")])
    assert d.iloc[0] == pytest.approx(mon_close / 100.0 - 1)
    assert d.iloc[1] == pytest.approx(eq.iloc[-1] / mon_close - 1)
    td = trading_dates(pd.DatetimeIndex(["2024-01-06 10:00", "2024-01-07 23:00"], tz="UTC"))
    assert list(td.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-08"]
    unfolded = daily_returns(eq, fold_weekends=False)
    assert len(unfolded) == 3


def test_cagr_and_calmar() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC"),
                            pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=365.25),
                            pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=2 * 365.25)])
    eq = pd.Series([100.0, 80.0, 200.0], index=idx)
    m = compute_metrics(_result(eq))
    assert m["cagr"] == pytest.approx(math.sqrt(2) - 1)
    assert m["max_drawdown"] == pytest.approx(-0.2)
    assert m["calmar"] == pytest.approx((math.sqrt(2) - 1) / 0.2)
    assert m["years"] == pytest.approx(2.0)


def test_max_drawdown_duration() -> None:
    idx = pd.date_range("2024-01-01", periods=15, freq="D", tz="UTC")
    vals = [100, 110, 105, 100, 102, 104, 106, 108, 109, 109.5, 108, 111, 111, 110, 110.5]
    eq = pd.Series(vals, index=idx, dtype=float)
    # peak 110 on day 1, recovered on day 11 -> 10 days; last episode unrecovered: day 12 -> 14 = 2
    assert max_drawdown_duration_days(eq) == pytest.approx(10.0)
    assert max_drawdown_duration_days(pd.Series([1.0, 2.0, 3.0], index=idx[:3])) == 0.0


def test_var_cvar_tail_ratio_skew_kurt() -> None:
    rets = np.linspace(-0.03, 0.05, 41)
    rng = np.random.default_rng(0)
    rng.shuffle(rets)
    eq = _weekday_daily_equity(list(rets))
    m = compute_metrics(_result(eq))
    d = daily_returns(eq).to_numpy()
    q05, q95 = np.quantile(d, [0.05, 0.95])
    assert m["var_95_daily"] == pytest.approx(-q05)
    assert m["cvar_95_daily"] == pytest.approx(-d[d <= q05].mean())
    assert m["cvar_95_daily"] >= m["var_95_daily"]
    assert m["tail_ratio"] == pytest.approx(abs(q95) / abs(q05))
    assert m["skew"] == pytest.approx(pd.Series(d).skew())
    assert m["kurtosis"] == pytest.approx(pd.Series(d).kurt())


def test_trade_statistics() -> None:
    eq = _weekday_daily_equity([0.01] * 20)
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -25.0], "bars_held": [2, 4, 6, 8]})
    pos = pd.Series(0.0, index=eq.index)
    pos.iloc[5:10] = 1.0
    m = compute_metrics(_result(eq, trades=trades, positions=pos))
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(300 / 75)
    assert m["avg_win"] == pytest.approx(150.0)
    assert m["avg_loss"] == pytest.approx(-37.5)
    assert m["expectancy"] == pytest.approx(56.25)
    assert m["avg_hold_bars"] == pytest.approx(5.0)
    assert m["exposure"] == pytest.approx(5 / 21)
    assert m["trades_per_year"] == pytest.approx(4 / m["years"])


def test_returns_series_input_and_flat_series() -> None:
    idx = pd.date_range("2024-01-08", periods=24 * 10, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0002, 0.002, len(idx)), index=idx)
    m = compute_metrics(r)
    assert m["total_return"] == pytest.approx(np.prod(1 + r.to_numpy()) - 1)
    assert m["n_trades"] == 0 and math.isnan(m["win_rate"])
    flat = compute_metrics(pd.Series(0.0, index=idx))
    assert flat["sharpe"] == 0.0 and flat["max_drawdown"] == 0.0 and flat["total_return"] == 0.0


def test_metrics_are_json_serialisable_and_complete() -> None:
    eq = _weekday_daily_equity(list(np.random.default_rng(2).normal(0, 0.01, 60)))
    m = compute_metrics(_result(eq, trades=pd.DataFrame({"pnl": [1.0], "bars_held": [3]})))
    required = {"total_return", "cagr", "ann_vol", "sharpe", "sharpe_bar", "sortino", "calmar",
                "max_drawdown", "max_dd_duration_days", "skew", "kurtosis", "var_95_daily",
                "cvar_95_daily", "tail_ratio", "n_trades", "trades_per_year", "win_rate",
                "profit_factor", "avg_win", "avg_loss", "expectancy", "avg_hold_bars", "exposure",
                "turnover_lots_per_year", "total_costs", "cost_drag_ann", "swap_total", "best_day",
                "worst_day"}
    assert required <= set(m)
    json.dumps(m)
    assert all(isinstance(v, (int, float)) for v in m.values())


# ---- reviewer: adversarial regressions ---------------------------------------------------------------
def _ny_close_daily_equity(n: int = 60, seed: int = 3) -> pd.Series:
    """D1 bars as exported by New-York-close (NY+7) MT5 servers: each session's bar opens at
    22:00 UTC the evening before (Sun..Thu opens, Mon..Fri sessions)."""
    sessions = pd.bdate_range("2024-01-08", periods=n, tz="UTC")
    opens = sessions - pd.Timedelta(hours=2)
    rng = np.random.default_rng(seed)
    return pd.Series(1e5 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n)), index=opens)


def test_ny_close_aligned_daily_bars_give_five_days_per_week() -> None:
    """Regression: dating marks by bar OPEN folded the Sunday-22:00 bar and the Monday-22:00 bar
    onto the same Monday, leaving 4 'days' per week (daily vol/Sharpe overstated ~10%)."""
    eq = _ny_close_daily_equity()
    legacy = daily_returns(eq)                                  # open-time dating (helper default)
    assert len(legacy) < len(eq)                                # the bug: bars merged
    d = daily_returns(eq, bar_duration="1D")
    assert len(d) == len(eq)
    assert list(d.index.weekday[:5]) == [0, 1, 2, 3, 4]         # session (close) days Mon..Fri
    np.testing.assert_allclose(d.to_numpy()[1:], eq.pct_change().to_numpy()[1:], rtol=1e-12)
    res = _result(eq)
    res.meta["timeframe"] = "D1"
    m = compute_metrics(res)
    assert m["n_days"] == len(eq)
    arr = d.to_numpy()
    assert m["ann_vol"] == pytest.approx(arr.std(ddof=1) * math.sqrt(252))
    # without meta the bar length is inferred from the (regular) index: same answer
    assert compute_metrics(_result(eq))["n_days"] == len(eq)
    assert compute_metrics(_result(eq), bar_duration="1D")["sharpe"] == pytest.approx(m["sharpe"])


def test_close_time_dating_is_identical_for_bars_not_straddling_midnight() -> None:
    idx = pd.date_range("2024-01-07 22:00", periods=24 * 12, freq="1h", tz="UTC")
    idx = idx[~((idx.weekday == 5) | ((idx.weekday == 4) & (idx.hour >= 21))
                | ((idx.weekday == 6) & (idx.hour < 22)))]
    rng = np.random.default_rng(4)
    eq = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.001, len(idx))), index=idx)
    pd.testing.assert_series_equal(daily_returns(eq), daily_returns(eq, bar_duration="1h"))
    daily = _weekday_daily_equity([0.01, -0.02, 0.005, 0.0, 0.003])
    pd.testing.assert_series_equal(daily_returns(daily), daily_returns(daily, bar_duration="1D"))


def test_infer_bar_duration() -> None:
    from aurum.backtest.metrics import infer_bar_duration

    h1 = pd.date_range("2024-01-08", periods=200, freq="1h", tz="UTC")
    h1 = h1[(h1.hour != 21)]                                      # daily break gaps
    assert infer_bar_duration(h1) == pd.Timedelta(hours=1)
    assert infer_bar_duration(_ny_close_daily_equity().index) == pd.Timedelta(days=1)
    irregular = pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-01 20:00", "2024-01-02 09:00",
                                  "2024-01-03 12:00"], tz="UTC")
    assert infer_bar_duration(irregular) is None                  # no dominant spacing
    weekly = pd.date_range("2024-01-01", periods=10, freq="7D", tz="UTC")
    assert infer_bar_duration(weekly) == pd.Timedelta(days=1)     # capped at one day
    assert infer_bar_duration(pd.DatetimeIndex([], tz="UTC")) is None


def test_engine_d1_ny_close_bars_end_to_end() -> None:
    from aurum.backtest.engine import buy_and_hold_benchmark
    from aurum.data.schema import make_bars

    eq_like = _ny_close_daily_equity(120, seed=8)
    close = 2000.0 * eq_like.to_numpy() / 1e5
    op = np.r_[2000.0, close[:-1]]
    bars = make_bars(pd.DataFrame({"open": op, "high": np.maximum(op, close) + 1.0,
                                   "low": np.minimum(op, close) - 1.0, "close": close,
                                   "spread": 0.3}, index=eq_like.index), "D1")
    bh = buy_and_hold_benchmark(bars, frictionless=True)
    assert bh.metrics["n_days"] == len(bars)
    # 1x-notional buy-and-hold: daily vol ~ the price's own daily vol
    px_vol = np.log(bars["close"]).diff().std() * math.sqrt(252)
    assert bh.metrics["ann_vol"] == pytest.approx(px_vol, rel=0.05)
