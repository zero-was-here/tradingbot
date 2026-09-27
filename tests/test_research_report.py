"""Tests for aurum.research.report (tearsheet generation from a hand-built BacktestResult)."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.backtest.result import BacktestResult
from aurum.core.instrument import XAUUSD
from aurum.core.types import Side
from aurum.data.synthetic import make_synthetic_bars
from aurum.research.report import daily_returns_from_equity, tearsheet_html, write_tearsheet
from aurum.research.stats import pbo_cscv


def make_result(n: int = 3000, seed: int = 3, *, with_metrics: bool = False) -> tuple[BacktestResult, pd.DataFrame]:
    """A small but internally consistent backtest built by hand (no engine dependency).

    Position decided at close of t from 24-bar momentum, held during bar t+1; PnL marked at
    closes; spread cost charged on every change of position; nightly swap on longs.
    """
    bars = make_synthetic_bars(n, "H1", seed=seed, model="trend", annual_vol=0.18)
    close = bars["close"]
    mom = np.log(close).diff(24)
    decision = np.sign(mom).fillna(0.0) * 1.0          # lots decided at close of t
    positions = decision.shift(1).fillna(0.0)           # held during bar t (filled at open)
    pnl = positions * XAUUSD.contract_size * close.diff().fillna(0.0)
    trade_size = positions.diff().abs().fillna(positions.abs())
    spread_cost = trade_size * XAUUSD.contract_size * bars["spread"] / 2.0
    slippage = trade_size * XAUUSD.contract_size * 0.02
    rollover = (bars.index.hour == 21)
    swap = np.where(rollover, np.where(positions > 0, XAUUSD.swap_long_per_lot,
                                       np.where(positions < 0, XAUUSD.swap_short_per_lot, 0.0))
                    * positions.abs(), 0.0)
    costs = pd.DataFrame({"spread": spread_cost, "slippage": slippage,
                          "commission": trade_size * 3.5, "swap": swap}, index=bars.index)
    net = pnl - costs[["spread", "slippage", "commission"]].sum(axis=1) + costs["swap"]
    equity = 100_000.0 + net.cumsum()
    returns = equity.pct_change().fillna(0.0)

    # round trips from position runs
    rows = []
    run_id = (positions != positions.shift()).cumsum()
    for _, seg in positions.groupby(run_id):
        lots = seg.iloc[0]
        if lots == 0:
            continue
        entry, exit_ = seg.index[0], seg.index[-1]
        px_in, px_out = bars.loc[entry, "open"], bars.loc[exit_, "close"]
        c = float(costs.loc[seg.index, ["spread", "slippage", "commission"]].sum().sum())
        s = float(costs.loc[seg.index, "swap"].sum())
        rows.append(dict(entry_time=entry, exit_time=exit_, side=Side.BUY if lots > 0 else Side.SELL,
                         lots=abs(lots), entry_price=px_in, exit_price=px_out,
                         pnl=lots * 100 * (px_out - px_in) - c + s, costs=c, swap=s,
                         exit_reason="signal" if len(rows) % 7 else "stop"))
    trades = pd.DataFrame(rows)
    fills = pd.DataFrame(columns=["time", "side", "lots", "price"])
    risk_events = pd.DataFrame({
        "time": bars.index[[n // 30, n // 7, n // 7 + 1, n // 3]],
        "reasons": [["max_spread 0.9 > 0.5"], ["event_blackout: NFP"], ["event_blackout: NFP"],
                    ["daily_loss", "halted"]],
        "requested": [1.0, -1.0, -1.0, 1.0], "approved": [0.0, 0.0, 0.0, 0.0],
    })
    metrics = {"sharpe": 1.23, "total_return": 0.05, "n_trades": len(trades)} if with_metrics else {}
    res = BacktestResult(equity=equity, returns=returns, positions=positions, costs=costs,
                         trades=trades, fills=fills, metrics=metrics, target=decision,
                         forecast=np.tanh(mom.fillna(0.0) * 20), risk_events=risk_events,
                         meta={"strategy": "tsmom-demo", "timeframe": "H1", "git_sha": "abc123",
                               "initial_equity": 100_000.0})
    return res, bars


@pytest.fixture(scope="module")
def result_and_bars():
    return make_result()


def _imgs(doc: str) -> list[str]:
    return re.findall(r"src='data:image/png;base64,([A-Za-z0-9+/=]+)'", doc)


def test_write_tearsheet_self_contained(tmp_path: Path, result_and_bars) -> None:
    res, bars = result_and_bars
    folds = pd.DataFrame({"train_start": ["2020-01-06"] * 4, "oos_sharpe": [0.5, -0.2, 1.1, 0.3],
                          "n_bars": [500, 500, 500, 480]})
    rng = np.random.default_rng(0)
    pbo = pbo_cscv(rng.standard_normal((400, 8)) * 0.01, n_splits=8)
    out = write_tearsheet(
        res, tmp_path / "sub" / "ts.html", benchmark=bars["close"], title="Demo <script>",
        extra={"folds": folds, "weights": {"tsmom": 0.4, "ema_cross": 0.35, "rsi2": 0.25},
               "n_trials": 25, "pbo": pbo, "notes": "Synthetic data only.",
               "config": {"target_vol": 0.1}, "custom_table": pd.DataFrame({"a": [1, 2]})},
        n_boot=200,
    )
    assert out.exists() and out.name == "ts.html"
    doc = out.read_text(encoding="utf-8")
    # self-contained: no external fetches
    assert not re.search(r"(src|href)=['\"]https?://", doc)
    assert "<script" not in doc.replace("&lt;script", "")  # title is escaped, no JS injected
    assert "Demo &lt;script&gt;" in doc
    for section in ("Equity", "Drawdown", "Rolling Sharpe", "Monthly returns", "Return distribution",
                    "Position &amp; exposure", "Cost attribution", "Trade statistics",
                    "Statistical confidence", "Walk-forward folds", "Strategy weights",
                    "Risk-manager interventions", "All metrics", "Notes &amp; provenance"):
        assert f"<h2>{section}</h2>" in doc, section
    assert "Deflated Sharpe Ratio (N = 25 trials)" in doc
    assert "Probability of backtest overfitting" in doc
    assert "Custom table" in doc
    imgs = _imgs(doc)
    assert len(imgs) >= 16  # >= 8 charts x light/dark
    for b64 in imgs[:3]:
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"
    assert "abc123" in doc  # provenance


def test_tearsheet_minimal_result_and_directory_path(tmp_path: Path) -> None:
    idx = pd.date_range("2021-01-04", periods=60, freq="1D", tz="UTC")
    eq = pd.Series(10_000.0 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 60)), index=idx)
    res = BacktestResult(equity=eq, returns=eq.pct_change().fillna(0.0),
                         positions=pd.Series(0.0, index=idx), costs=pd.DataFrame(index=idx),
                         trades=pd.DataFrame(), fills=pd.DataFrame())
    out = write_tearsheet(res, tmp_path, dark_charts=False, n_boot=100)
    assert out == tmp_path / "tearsheet.html"
    doc = out.read_text(encoding="utf-8")
    assert "No closed trades" in doc
    assert "Statistical confidence" in doc
    assert "class='dark'" not in doc


def test_metrics_precedence(result_and_bars) -> None:
    res, _ = make_result(n=800, with_metrics=True)
    doc = tearsheet_html(res, n_boot=50, dark_charts=False)
    assert "Source: result.metrics" in doc
    assert "1.230" in doc  # provided sharpe wins over fallback


def test_daily_returns_from_equity() -> None:
    idx = pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-01 20:00", "2024-01-02 09:00",
                            "2024-01-03 12:00"], tz="UTC")
    eq = pd.Series([101.0, 102.0, 99.96, 104.958], index=idx)
    r = daily_returns_from_equity(eq, 100.0)
    assert len(r) == 3
    np.testing.assert_allclose(r.to_numpy(), [0.02, -0.02, 0.05], rtol=1e-9)


def test_empty_equity_raises() -> None:
    res = BacktestResult(equity=pd.Series(dtype=float), returns=pd.Series(dtype=float),
                         positions=pd.Series(dtype=float), costs=pd.DataFrame(),
                         trades=pd.DataFrame(), fills=pd.DataFrame())
    with pytest.raises(ValueError):
        tearsheet_html(res)


@pytest.mark.parametrize("peer_available", [True, False])
def test_daily_returns_fold_weekend_bars_into_monday(monkeypatch, peer_available: bool) -> None:
    if not peer_available:
        import sys

        monkeypatch.setitem(sys.modules, "aurum.backtest.metrics", None)  # force ImportError
    idx = pd.DatetimeIndex(["2024-01-05 20:00",   # Fri
                            "2024-01-07 22:00",   # Sun reopen stub -> counts as Monday
                            "2024-01-08 10:00"],  # Mon
                           tz="UTC")
    eq = pd.Series([110.0, 111.0, 121.0], index=idx)
    r = daily_returns_from_equity(eq, 100.0)
    assert len(r) == 2
    np.testing.assert_allclose(r.to_numpy(), [0.10, 0.10], rtol=1e-12)


def test_tearsheet_without_peer_metrics_module(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "aurum.backtest.metrics", None)
    res, _ = make_result(n=600)
    doc = tearsheet_html(res, n_boot=50, dark_charts=False)
    assert "Source: fallback (aurum.research.report)" in doc
    assert "<h2>Statistical confidence</h2>" in doc


# ---------------------------------------------------------------------------- review: adversarial
def _tiny_result(n: int = 400, *, seed: int = 0, **kw) -> BacktestResult:
    idx = pd.date_range("2024-01-02", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    eq = pd.Series(100_000.0 * np.cumprod(1 + rng.normal(0.0002, 0.002, n)), index=idx)
    eq.iloc[0] = 100_000.0
    base = dict(equity=eq, returns=eq.pct_change().fillna(0.0),
                positions=pd.Series(1.0, index=idx), costs=pd.DataFrame(index=idx),
                trades=pd.DataFrame(), fills=pd.DataFrame(), meta={"initial_equity": 100_000.0})
    base.update(kw)
    return BacktestResult(**base)


def test_engine_style_joined_risk_reasons_are_all_counted() -> None:
    """aurum.backtest.engine stores a bar's reasons as ONE '; '-joined string; the table
    used to count only the first rule and lump empty strings into a blank rule."""
    res = _tiny_result()
    t = res.equity.index
    res.risk_events = pd.DataFrame({
        "time": t[[10, 20, 30]],
        "reasons": ["max_spread 0.9 > 0.5; event_blackout: NFP", "event_blackout: CPI", ""],
        "requested": [1.0, 1.0, 1.0], "approved": [0.0, 0.0, 0.5],
    })
    doc = tearsheet_html(res, n_boot=20, dark_charts=False)
    sect = doc.split("<h2>Risk-manager interventions</h2>")[1].split("</section>")[0]
    rows = dict(re.findall(r"<th scope='row'>([^<]+)</th><td class='num'>(\d+)</td>", sect))
    assert rows.get("event_blackout") == "2"
    assert rows.get("max_spread") == "1"
    assert rows.get("(unspecified)") == "1"


def test_non_numeric_cost_column_does_not_kill_report() -> None:
    res = _tiny_result()
    res.costs = pd.DataFrame({"spread": 1.0, "note": "manual"}, index=res.equity.index)
    doc = tearsheet_html(res, n_boot=20, dark_charts=False)
    assert "<h2>Cost attribution</h2>" in doc and "$400.00" in doc


def test_n_trials_kwarg_wins_and_extra_key_not_rerendered() -> None:
    res = _tiny_result(n=24 * 60)
    doc = tearsheet_html(res, n_boot=20, dark_charts=False, n_trials=5, extra={"n_trials": 7})
    assert "Deflated Sharpe Ratio (N = 5 trials)" in doc
    assert "<h3>N trials</h3>" not in doc


def test_folds_without_finite_sharpe_do_not_print_nan() -> None:
    res = _tiny_result()
    folds = pd.DataFrame({"oos_sharpe": [np.nan, np.nan], "n_bars": [100, 100]})
    doc = tearsheet_html(res, n_boot=20, dark_charts=False, extra={"folds": folds})
    assert "<h2>Walk-forward folds</h2>" in doc
    assert "nan%" not in doc


def test_lots_traded_counts_initial_entry_and_prefers_fills() -> None:
    res = _tiny_result()  # long 1 lot from the first bar, never changed
    doc = tearsheet_html(res, n_boot=20, dark_charts=False)
    assert re.search(r"Lots traded</th><td>1\.000</td>", doc)
    res.fills = pd.DataFrame({"time": res.equity.index[[0, 5, 9]], "lots": [1.0, 0.5, 0.5]})
    doc = tearsheet_html(res, n_boot=20, dark_charts=False)
    assert re.search(r"Lots traded</th><td>2\.000</td>", doc)


def test_extra_figure_canvas_is_not_rebound() -> None:
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.figure import Figure

    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [1, 0])
    canvas_before = fig.canvas
    doc = tearsheet_html(_tiny_result(), n_boot=20, dark_charts=False, extra={"my_plot": fig})
    assert fig.canvas is canvas_before and isinstance(fig.canvas, FigureCanvasBase)
    assert "<h3>My plot</h3>" in doc


def test_first_bar_wipeout_does_not_crash_initial_equity() -> None:
    res = _tiny_result(n=100)
    res.meta = {}
    res.returns = res.returns.copy()
    res.returns.iloc[0] = -1.0  # inconsistent producer: first-bar return of -100%
    doc = tearsheet_html(res, n_boot=20, dark_charts=False)
    assert "<h2>Equity</h2>" in doc


def test_tile_and_confidence_sharpe_agree_with_engine_result() -> None:
    """Integration with the real engine (skipped if the peer module is missing): the headline
    tile (metrics) and the statistical-confidence block use the same daily returns."""
    engine = pytest.importorskip("aurum.backtest.engine")
    sizing = pytest.importorskip("aurum.portfolio.sizing")
    bars = make_synthetic_bars(900, "H1", seed=2, model="trend")
    fc = np.sign(np.log(bars["close"]).diff(24)).fillna(0.0)
    res = engine.run_backtest(bars, fc, sizer=sizing.VolTargetSizer())
    doc = tearsheet_html(res, benchmark=bars, n_boot=20, dark_charts=False)
    tile = re.search(r"Sharpe \(daily\)</div><div class='tv'>([^<]+)<", doc).group(1)
    conf = re.search(r"Sharpe \(annualised, daily returns\)</th><td>([^<]+)<", doc).group(1)
    assert tile == conf
