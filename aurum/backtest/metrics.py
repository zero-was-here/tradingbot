"""Performance metrics for backtests (SPEC §8).

Conventions
-----------
* **Headline risk/return statistics use DAILY returns** annualised with 252 trading days:
  ``sharpe``, ``ann_vol``, ``sortino``, ``skew``,
  ``kurtosis``, VaR/CVaR, best/worst day. Per-bar returns are autocorrelated through the
  intraday seasonality of volatility and annualising them with ``sqrt(bars_per_year)``
  overstates precision; the per-bar Sharpe is still reported as ``sharpe_bar``
  (annualised with ``infer_bars_per_year``, SPEC §1).
* **Day of an equity mark.** ``equity[t]`` is marked at the CLOSE of bar ``t`` (open +
  bar duration). In :func:`compute_metrics` each mark is assigned to the UTC date on which
  that close falls (a close at exactly 00:00 belongs to the previous day), with the bar
  duration taken from ``result.meta["timeframe"]`` or inferred from the index spacing. For
  bars that do not straddle midnight (M1..H1, UTC-aligned H4/D1) this equals the date of the
  bar's open time. For broker D1 bars aligned to the New York close (open Sun..Thu
  21:00/22:00 UTC) the open-time date would put the Sunday and Monday 22:00 bars both on
  Monday (after weekend folding) and yield only four "days" per week, overstating daily
  vol and Sharpe by ~10%; the close-time date gives the correct five.
* Gold CFDs reopen on Sunday ~22:00 UTC. Those few Sunday bars (and any Saturday bars) are
  **folded into the following Monday** so that a two-hour Sunday stub does not count as a
  "trading day" (which would add ~52 near-zero observations per year and bias the daily
  Sharpe and volatility). Days without bars are simply absent (not zero-return days).
* Risk-free rate = 0: CFD PnL is already an excess return (financing is paid via swap).
* ``max_drawdown`` is reported as a NEGATIVE fraction (e.g. -0.18), matching
  ``drawdown_series``; ``calmar = cagr / |max_drawdown|``.
* ``var_95_daily`` / ``cvar_95_daily`` are POSITIVE loss fractions (historical simulation):
  VaR = -q_5%(r), CVaR (expected shortfall) = -mean(r | r <= q_5%) (Acerbi & Tasche 2002).
* ``kurtosis`` is EXCESS kurtosis (normal = 0), bias-corrected like ``pandas.Series.kurt``;
  ``skew`` likewise.
* ``sortino`` uses the downside deviation with target 0 over *all* observations,
  ``sqrt(mean(min(r, 0)^2))`` (Sortino & Price 1994).
* ``cost_drag_ann`` = transaction costs (spread + slippage + commission, excl. swap) per year
  as a fraction of average equity.

All returned values are plain Python ``float``/``int`` (JSON-serialisable; NaN when a
statistic is undefined, e.g. no trades or too few days).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.result import BacktestResult
from aurum.core.timeframes import get_timeframe, infer_bars_per_year

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: int = 252
_SECONDS_PER_YEAR: float = 365.25 * 24 * 3600
_NAN = float("nan")


# ---- series helpers ------------------------------------------------------------------------------
def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak: ``equity / cummax(equity) - 1`` (<= 0)."""
    eq = pd.Series(equity, dtype=float)
    peak = eq.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / peak - 1.0
    return dd.rename("drawdown")


def _as_duration(bar_duration: pd.Timedelta | str | None) -> pd.Timedelta | None:
    if bar_duration is None:
        return None
    dur = pd.Timedelta(bar_duration)
    if pd.isna(dur) or dur <= pd.Timedelta(0):
        return None
    return dur


def infer_bar_duration(index: pd.Index) -> pd.Timedelta | None:
    """Typical bar length of a regular bar index: the modal spacing, capped at one day.

    Returns None when the index is too short or irregular (no spacing accounts for at least
    half of the gaps), in which case callers fall back to open-time dates. Weekend and holiday
    gaps are minority spacings, so they do not affect the mode of a genuine bar series.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    diffs = np.diff(index.as_unit("ns").asi8)
    diffs = diffs[diffs > 0]
    if diffs.size < 2:
        return None
    vals, counts = np.unique(diffs, return_counts=True)
    k = int(np.argmax(counts))
    if counts[k] < 2 or counts[k] < 0.5 * diffs.size:
        return None
    return min(pd.Timedelta(int(vals[k]), unit="ns"), pd.Timedelta(days=1))


def trading_dates(index: pd.DatetimeIndex, *, fold_weekends: bool = True,
                  bar_duration: pd.Timedelta | str | None = None) -> pd.DatetimeIndex:
    """UTC trading date of each bar; Saturday/Sunday dates are moved to the next Monday.

    Without ``bar_duration`` the date of the timestamp itself (the bar OPEN time) is used.
    With ``bar_duration`` the date of the bar's CLOSE (``open + duration``) is used, where a
    close at exactly 00:00 UTC belongs to the day that just ended — the day on which the
    equity mark ``equity[t]`` is actually taken.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        raise ValueError("index must be tz-aware (UTC)")
    idx = idx.tz_convert("UTC")
    unit = idx.unit
    dur = _as_duration(bar_duration)
    if dur is not None:
        idx = idx.as_unit("ns") + (dur - pd.Timedelta(1, unit="ns"))
    dates = idx.normalize().as_unit(unit)  # midnight is exact in any unit: keep the caller's
    if fold_weekends:
        wd = dates.weekday.to_numpy()
        shift = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        if shift.any():
            dates = dates + pd.to_timedelta(shift, unit="D")
    return dates


def daily_equity(equity: pd.Series, *, fold_weekends: bool = True,
                 bar_duration: pd.Timedelta | str | None = None) -> pd.Series:
    """Equity at the close of the last bar of each (UTC) trading day (see
    :func:`trading_dates` for ``bar_duration``)."""
    eq = pd.Series(equity, dtype=float).dropna()
    if eq.empty:
        return eq
    dates = trading_dates(eq.index, fold_weekends=fold_weekends, bar_duration=bar_duration)
    return eq.groupby(dates).last()


def daily_returns(equity: pd.Series, *, initial: float | None = None,
                  fold_weekends: bool = True,
                  bar_duration: pd.Timedelta | str | None = None) -> pd.Series:
    """Simple daily returns of an equity curve indexed by bar OPEN time (UTC).

    ``equity[t]`` is the mark at the close of bar ``t``; the day's closing equity is the
    last mark of that trading date (open-time date, or close-time date when
    ``bar_duration`` is given — see :func:`trading_dates`). The first day's return is
    measured against ``initial`` (default ``equity.iloc[0]``, i.e. the BacktestResult
    convention that the first bar carries no PnL).
    """
    eq = pd.Series(equity, dtype=float).dropna()
    if eq.empty:
        return pd.Series(dtype=float, name="daily_returns")
    d = daily_equity(eq, fold_weekends=fold_weekends, bar_duration=bar_duration)
    base = float(eq.iloc[0]) if initial is None else float(initial)
    prev = d.shift(1)
    prev.iloc[0] = base
    with np.errstate(divide="ignore", invalid="ignore"):
        r = d / prev - 1.0
    return r.rename("daily_returns")


# ---- scalar statistics -----------------------------------------------------------------------------
def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v


def _sharpe(r: np.ndarray, periods: float) -> float:
    """Annualised Sharpe of simple returns ``r`` (ddof=1); 0 for an all-zero series."""
    r = r[np.isfinite(r)]
    if r.size < 2:
        return _NAN
    sd = r.std(ddof=1)
    mu = r.mean()
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0 if mu == 0.0 else _NAN
    return float(mu / sd * math.sqrt(periods))


def _sortino(r: np.ndarray, periods: float) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2:
        return _NAN
    dd = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    mu = float(r.mean())
    if dd == 0.0:
        return 0.0 if mu == 0.0 else _NAN
    return mu / dd * math.sqrt(periods)


def max_drawdown(equity: pd.Series) -> float:
    """Most negative drawdown fraction (<= 0)."""
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else _NAN


def max_drawdown_duration_days(equity: pd.Series) -> float:
    """Longest time (calendar days) from a peak until equity regains it (or the sample ends)."""
    eq = pd.Series(equity, dtype=float).dropna()
    if len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    at_peak = eq.to_numpy() >= peak.to_numpy() * (1.0 - 1e-12)
    if at_peak.all():
        return 0.0
    t = pd.Series(eq.index, index=eq.index)
    peak_time = t.where(at_peak).ffill()
    # ex-post statistic: the recovery time is by definition in the future of the drawdown
    rec_time = t.where(at_peak).bfill().fillna(t.iloc[-1])
    dur = (rec_time - peak_time)[~at_peak]
    if dur.empty:
        return 0.0
    return float(dur.max().total_seconds() / 86_400.0)


def _years(index: pd.Index) -> float:
    if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return _NAN
    return (index[-1] - index[0]).total_seconds() / _SECONDS_PER_YEAR


# ---- main entry point --------------------------------------------------------------------------------
def compute_metrics(
    result_or_returns: BacktestResult | pd.Series,
    *,
    bars_per_year: float | None = None,
    trades: pd.DataFrame | None = None,
    positions: pd.Series | None = None,
    costs: pd.DataFrame | None = None,
    equity: pd.Series | None = None,
    fills: pd.DataFrame | None = None,
    bar_duration: pd.Timedelta | str | None = None,
) -> dict[str, float | int]:
    """Standard metric dictionary for a :class:`BacktestResult` or a per-bar return series.

    With a return series, equity is ``cumprod(1 + r)`` starting from 1 (unless ``equity`` is
    given) and trade/position/cost metrics require the optional frames. ``bar_duration``
    dates each equity mark by its bar's close (default: ``result.meta["timeframe"]``, else
    inferred from the index spacing). See the module docstring for conventions.
    """
    initial: float
    skip_first = False  # BacktestResult convention: the first bar's return is 0 by construction
    tf_name = None
    if isinstance(result_or_returns, BacktestResult):
        res = result_or_returns
        tf_name = (res.meta or {}).get("timeframe")
        eq = res.equity if equity is None else equity
        rets = res.returns
        trades = res.trades if trades is None else trades
        positions = res.positions if positions is None else positions
        costs = res.costs if costs is None else costs
        fills = res.fills if fills is None else fills
        eq = pd.Series(eq, dtype=float)
        initial = float(eq.iloc[0]) if len(eq) else _NAN
        skip_first = True
    else:
        rets = pd.Series(result_or_returns, dtype=float).fillna(0.0)
        if equity is not None:
            eq = pd.Series(equity, dtype=float)
            initial = float(eq.iloc[0]) if len(eq) else _NAN
        else:
            eq = (1.0 + rets).cumprod()
            initial = 1.0

    out: dict[str, float | int] = {}
    n = len(eq)
    if n == 0:
        logger.warning("compute_metrics: empty equity series")
        return {"n_bars": 0}
    idx = eq.index
    years = _years(idx)
    final = float(eq.iloc[-1])

    # --- returns / risk ---------------------------------------------------------------------
    total_return = final / initial - 1.0 if initial else _NAN
    if math.isfinite(years) and years > 0 and initial > 0:
        cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    else:
        cagr = _NAN
    dur = _as_duration(bar_duration)
    if dur is None and tf_name:
        try:
            dur = get_timeframe(tf_name).delta
        except ValueError:
            dur = None
    if dur is None:
        dur = infer_bar_duration(idx)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        d = daily_returns(eq, initial=initial, bar_duration=dur).to_numpy()
    else:
        d = np.array([])
    d = d[np.isfinite(d)]
    r_bar = pd.Series(rets, dtype=float).to_numpy()
    bpy = bars_per_year
    if bpy is None and isinstance(idx, pd.DatetimeIndex) and n >= 2:
        try:
            bpy = infer_bars_per_year(idx)
        except ValueError:
            bpy = None
    mdd = max_drawdown(eq)

    out["total_return"] = _f(total_return)
    out["cagr"] = _f(cagr)
    out["ann_vol"] = _f(d.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if d.size >= 2 else _NAN
    out["sharpe"] = _sharpe(d, TRADING_DAYS_PER_YEAR)
    out["sharpe_bar"] = _sharpe(r_bar[1:] if skip_first else r_bar, bpy) if bpy else _NAN
    out["sortino"] = _sortino(d, TRADING_DAYS_PER_YEAR)
    out["max_drawdown"] = _f(mdd)
    out["calmar"] = _f(cagr / abs(mdd)) if (mdd < 0 and math.isfinite(cagr)) else _NAN
    out["max_dd_duration_days"] = max_drawdown_duration_days(eq)
    ds = pd.Series(d)
    out["skew"] = _f(ds.skew()) if d.size >= 3 else _NAN
    out["kurtosis"] = _f(ds.kurt()) if d.size >= 4 else _NAN
    if d.size >= 1:
        q05, q95 = np.quantile(d, [0.05, 0.95])
        tail = d[d <= q05]
        out["var_95_daily"] = _f(-q05)
        out["cvar_95_daily"] = _f(-tail.mean()) if tail.size else _NAN
        out["tail_ratio"] = _f(abs(q95) / abs(q05)) if q05 != 0 else _NAN
        out["best_day"] = _f(d.max())
        out["worst_day"] = _f(d.min())
    else:
        for k in ("var_95_daily", "cvar_95_daily", "tail_ratio", "best_day", "worst_day"):
            out[k] = _NAN

    # --- trades -----------------------------------------------------------------------------
    if trades is not None and "pnl" in trades.columns:
        pnl = trades["pnl"].to_numpy(dtype=float)
        nt = int(pnl.size)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        out["n_trades"] = nt
        out["trades_per_year"] = _f(nt / years) if (math.isfinite(years) and years > 0) else _NAN
        out["win_rate"] = _f(wins.size / nt) if nt else _NAN
        if nt == 0:
            pf = _NAN
        elif losses.size == 0:
            pf = math.inf if wins.size else _NAN
        else:
            pf = float(wins.sum() / -losses.sum())
        out["profit_factor"] = pf
        out["avg_win"] = _f(wins.mean()) if wins.size else _NAN
        out["avg_loss"] = _f(losses.mean()) if losses.size else _NAN
        out["expectancy"] = _f(pnl.mean()) if nt else _NAN
        out["avg_hold_bars"] = (_f(trades["bars_held"].astype(float).mean())
                                if nt and "bars_held" in trades.columns else _NAN)
    else:
        for k in ("trades_per_year", "win_rate", "profit_factor", "avg_win", "avg_loss",
                  "expectancy", "avg_hold_bars"):
            out[k] = _NAN
        out["n_trades"] = 0

    # --- positions / turnover -----------------------------------------------------------------
    if positions is not None and len(positions):
        p = pd.Series(positions, dtype=float).fillna(0.0).to_numpy()
        out["exposure"] = _f(np.mean(np.abs(p) > 1e-12))
        if fills is not None and "lots" in fills.columns:
            traded = float(fills["lots"].astype(float).abs().sum())
        else:
            traded = float(np.abs(np.diff(p, prepend=0.0)).sum())
        out["turnover_lots_per_year"] = (_f(traded / years)
                                         if (math.isfinite(years) and years > 0) else _NAN)
    else:
        out["exposure"] = _NAN
        out["turnover_lots_per_year"] = _NAN

    # --- costs ----------------------------------------------------------------------------------
    if costs is not None and len(costs):
        cc = [c for c in ("spread", "slippage", "commission") if c in costs.columns]
        total_costs = float(costs[cc].to_numpy(dtype=float).sum()) if cc else 0.0
        swap_total = float(costs["swap"].sum()) if "swap" in costs.columns else 0.0
        mean_eq = float(eq.mean())
        out["total_costs"] = total_costs
        out["cost_drag_ann"] = (_f(total_costs / mean_eq / years)
                                if (math.isfinite(years) and years > 0 and mean_eq > 0) else _NAN)
        out["swap_total"] = swap_total
    else:
        out["total_costs"] = _NAN
        out["cost_drag_ann"] = _NAN
        out["swap_total"] = _NAN

    # --- bookkeeping ------------------------------------------------------------------------------
    out["n_bars"] = int(n)
    out["n_days"] = int(d.size)
    out["years"] = _f(years)
    out["bars_per_year"] = _f(bpy) if bpy else _NAN
    out["final_equity"] = final
    return out
