"""Self-contained HTML tearsheet for a :class:`~aurum.backtest.result.BacktestResult`.

One file, no external assets: charts are PNGs rendered with matplotlib's Agg canvas (the
object-oriented ``Figure`` API - we never touch ``pyplot`` or the global backend, so the
module is safe to import in notebooks and servers) and inlined as base64 data URIs. Every
chart is rendered twice - for a light and a dark surface - and CSS shows the one matching
the reader's colour scheme (``prefers-color-scheme``, overridable with
``<html data-theme="light|dark">``).

Sections (SPEC §9): headline tiles, equity vs benchmark (log), underwater drawdown,
rolling 6-month Sharpe, monthly return table, daily-return histogram, position/exposure,
cost attribution, trade statistics, statistical confidence (PSR / DSR / bootstrap CI /
MinTRL / haircut, see :mod:`aurum.research.stats`), optional walk-forward folds, strategy
weights, risk events, the full metrics table, run provenance and notes/extra.

Headline Sharpe is computed on DAILY (UTC-date) returns with 252 periods per year, as the
SPEC requires; the per-bar Sharpe is reported separately.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import rc_context
from matplotlib import ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from aurum.backtest.result import BacktestResult
from aurum.core.timeframes import infer_bars_per_year
from aurum.research import stats as rstats

logger = logging.getLogger(__name__)

__all__ = ["write_tearsheet", "tearsheet_html", "daily_returns_from_equity"]

DAYS_PER_YEAR = 252
_MAX_PLOT_POINTS = 6000
_MAX_TABLE_ROWS = 60


# --------------------------------------------------------------------------------------
# themes (validated palette: series blue + loss red pass CVD/contrast checks in both modes)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: str
    neutral: str
    loss: str


_LIGHT = _Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
                "#2a78d6", "#898781", "#e34948")
_DARK = _Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
               "#3987e5", "#898781", "#e66767")

DrawFn = Callable[[Figure, _Theme], None]


# --------------------------------------------------------------------------------------
# series helpers
# --------------------------------------------------------------------------------------
def _utc_index(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Copy with a tz-aware UTC DatetimeIndex (naive timestamps are assumed to be UTC)."""
    out = s.copy()
    idx = pd.DatetimeIndex(out.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return out.sort_index()


def daily_returns_from_equity(equity: pd.Series, initial_equity: float | None = None) -> pd.Series:
    """Simple returns of end-of-UTC-day equity; the first day is measured from ``initial_equity``.

    Same convention as ``aurum.backtest.metrics.daily_returns`` (which is used when it is
    importable, so the tearsheet's statistics always agree with the metrics table): days
    are UTC dates of bar open times, Saturday/Sunday bars (gold reopens Sunday ~22:00 UTC)
    are folded into the following Monday, and days without bars are absent rather than
    zero-return days (which would deflate volatility and inflate the Sharpe ratio).
    """
    eq = _utc_index(equity.astype(float)).dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    try:
        from aurum.backtest.metrics import daily_returns as _peer_daily  # noqa: PLC0415

        r = _peer_daily(eq, initial=initial_equity)
        return r.rename("daily_return")
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - fall back to the local implementation
        logger.warning("aurum.backtest.metrics.daily_returns failed; using local version",
                       exc_info=True)
    dates = eq.index.normalize()
    wd = dates.weekday.to_numpy()
    shift = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
    if shift.any():
        dates = dates + pd.to_timedelta(shift, unit="D")
    daily = eq.groupby(dates).last()
    base = float(initial_equity) if initial_equity else float(eq.iloc[0])
    prev = daily.shift(1)
    prev.iloc[0] = base
    r = daily / prev - 1.0
    r.name = "daily_return"
    return r


def _x(index: pd.Index) -> np.ndarray:
    """matplotlib-friendly naive UTC datetime64 array."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy()


def _thin(s: pd.Series, max_points: int = _MAX_PLOT_POINTS) -> pd.Series:
    """Stride-subsample a long series for plotting (always keeps the last point)."""
    if len(s) <= max_points:
        return s
    step = int(math.ceil(len(s) / max_points))
    keep = np.r_[np.arange(0, len(s), step), len(s) - 1]
    return s.iloc[np.unique(keep)]


def _side_sign(v: Any) -> float:
    """Map Side enum / int / str to +1 (long) or -1 (short); NaN if unknown."""
    if v is None:
        return float("nan")
    if isinstance(v, str):
        u = v.upper()
        if "BUY" in u or "LONG" in u or u in ("1", "+1"):
            return 1.0
        if "SELL" in u or "SHORT" in u or u == "-1":
            return -1.0
        return float("nan")
    try:
        f = float(getattr(v, "value", v))
    except (TypeError, ValueError):
        return float("nan")
    return float(np.sign(f)) if f != 0 else float("nan")


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def _initial_equity(result: BacktestResult) -> float:
    meta_ie = result.meta.get("initial_equity") if isinstance(result.meta, dict) else None
    if meta_ie is not None:
        try:
            return float(meta_ie)
        except (TypeError, ValueError):
            pass
    eq = result.equity.dropna()
    if eq.empty:
        return float("nan")
    first_r = 0.0
    if result.returns is not None and len(result.returns):
        r0 = result.returns.dropna()
        first_r = float(r0.iloc[0]) if len(r0) else 0.0
    # Guard 1 + r0 <= 0 (a first-bar wipe-out): Python float division by zero would raise.
    if np.isfinite(first_r) and 1.0 + first_r > 0.0:
        return float(eq.iloc[0]) / (1.0 + first_r)
    return float(eq.iloc[0])


def _fallback_metrics(result: BacktestResult, daily: pd.Series, initial: float) -> dict[str, Any]:
    """Minimal metrics used when ``result.metrics`` is empty and the metrics module is absent.

    ``aurum.backtest.metrics.compute_metrics`` is authoritative; these are only a safety net
    so a tearsheet can always be produced.
    """
    eq = _utc_index(result.equity.astype(float)).dropna()
    out: dict[str, Any] = {}
    if eq.empty:
        return out
    final = float(eq.iloc[-1])
    tr = final / initial - 1.0 if initial else float("nan")
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
    cagr = (1 + tr) ** (1 / years) - 1 if years > 0 and (1 + tr) > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    d = daily.dropna()
    ann_vol = float(d.std(ddof=1) * math.sqrt(DAYS_PER_YEAR)) if len(d) > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(d.to_numpy(), 0.0) ** 2))) if len(d) else 0.0
    sortino = (float(d.mean()) / downside * math.sqrt(DAYS_PER_YEAR)) if downside > 0 else float("nan")
    out.update(
        total_return=tr,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=rstats.sharpe(d, DAYS_PER_YEAR),
        sortino=sortino,
        max_drawdown=mdd,
        calmar=cagr / abs(mdd) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        best_day=float(d.max()) if len(d) else float("nan"),
        worst_day=float(d.min()) if len(d) else float("nan"),
    )
    r = result.returns.dropna() if result.returns is not None else eq.pct_change().dropna()
    if len(r) > 2:
        try:
            out["sharpe_bar"] = rstats.sharpe(r, infer_bars_per_year(_utc_index(r).index))
        except ValueError:
            pass
    if result.positions is not None and len(result.positions):
        out["exposure"] = float((result.positions.fillna(0.0) != 0).mean())
    trades = result.trades if result.trades is not None else pd.DataFrame()
    out["n_trades"] = int(len(trades))
    if len(trades) and "pnl" in trades:
        pnl = trades["pnl"].astype(float)
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        out["win_rate"] = float((pnl > 0).mean())
        out["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan")
    if result.costs is not None and len(result.costs):
        cost_cols = [c for c in ("spread", "slippage", "commission") if c in result.costs]
        out["total_costs"] = float(result.costs[cost_cols].sum().sum()) if cost_cols else 0.0
        if "swap" in result.costs:
            out["swap_total"] = float(result.costs["swap"].sum())
    return out


def _resolve_metrics(
    result: BacktestResult, daily: pd.Series, initial: float
) -> tuple[dict[str, Any], str]:
    metrics = _fallback_metrics(result, daily, initial)
    source = "fallback (aurum.research.report)"
    provided = dict(result.metrics or {})
    if provided:
        source = "result.metrics"
    else:
        try:
            from aurum.backtest.metrics import compute_metrics  # noqa: PLC0415 - optional peer

            provided = dict(compute_metrics(result))
            source = "aurum.backtest.metrics.compute_metrics"
        except ImportError:
            logger.info("aurum.backtest.metrics unavailable; using fallback metrics")
        except Exception:  # noqa: BLE001 - a broken peer module must not kill the report
            logger.warning("compute_metrics failed; using fallback metrics", exc_info=True)
    metrics.update(provided)
    return metrics, source


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------
_PCT_KEYS = {
    "total_return", "cagr", "ann_vol", "max_drawdown", "win_rate", "exposure", "var_95_daily",
    "cvar_95_daily", "best_day", "worst_day", "cost_drag_ann", "benchmark_total_return",
    "benchmark_max_drawdown", "haircut",
}
_MONEY_KEYS = {
    "avg_win", "avg_loss", "expectancy", "total_costs", "swap_total", "initial_equity",
    "final_equity", "total_pnl",
}
_LABELS = {
    "total_return": "Total return", "cagr": "CAGR", "ann_vol": "Annual volatility",
    "sharpe": "Sharpe (daily, ann.)", "sharpe_bar": "Sharpe (per-bar, ann.)",
    "sortino": "Sortino", "calmar": "Calmar", "max_drawdown": "Max drawdown",
    "max_dd_duration_days": "Max DD duration (days)", "var_95_daily": "VaR 95% (daily)",
    "cvar_95_daily": "CVaR 95% (daily)", "n_trades": "Trades", "trades_per_year": "Trades / year",
    "win_rate": "Win rate", "profit_factor": "Profit factor", "avg_win": "Average win",
    "avg_loss": "Average loss", "avg_hold_bars": "Average hold (bars)", "exposure": "Exposure",
    "turnover_lots_per_year": "Turnover (lots / year)", "total_costs": "Total costs",
    "cost_drag_ann": "Cost drag (ann.)", "swap_total": "Swap (net, + received)",
    "best_day": "Best day", "worst_day": "Worst day", "tail_ratio": "Tail ratio",
    "skew": "Skewness (daily)", "kurtosis": "Excess kurtosis (daily)",
}


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _money(v: float, decimals: int = 2) -> str:
    """``$1,234.50`` / ``−$1,234.50`` (sign before the currency symbol)."""
    if v is None or not np.isfinite(v):
        return "–"
    return f"{'−' if v < 0 else ''}${abs(v):,.{decimals}f}"


def _label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " ").strip().capitalize())


def _fmt(v: Any, key: str | None = None) -> str:
    """Human formatting keyed by metric semantics (percent / money / ratio / count)."""
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M") if not pd.isna(v) else "–"
    if isinstance(v, pd.Timedelta):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return "–"
        if math.isinf(f):
            return "∞" if f > 0 else "−∞"
        if key in _PCT_KEYS:
            return f"{f * 100:,.2f}%"
        if key in _MONEY_KEYS:
            return _money(f)
        if f != 0 and abs(f) < 1e-3:
            return f"{f:.3e}"
        if abs(f) >= 1e4:
            return f"{f:,.0f}"
        return f"{f:,.3f}"
    if isinstance(v, (dict, list, tuple)):
        try:
            return json.dumps(v, default=str)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _kv_table(rows: list[tuple[str, str]], *, cls: str = "kv") -> str:
    body = "".join(f"<tr><th scope='row'>{_esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='tw'><table class='{cls}'><tbody>{body}</tbody></table></div>"


_NUMLIKE = re.compile(r"[\s$+\-−–,.%\d∞e()x]*")


def _is_numlike(v: Any) -> bool:
    """True for numbers and pre-formatted numeric strings ("$1,234.50", "12.3%", "–")."""
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    return isinstance(v, str) and bool(_NUMLIKE.fullmatch(v)) and (
        any(ch.isdigit() for ch in v) or v.strip() in ("–", "-", "∞")
    )


def _df_table(df: pd.DataFrame, *, max_rows: int = _MAX_TABLE_ROWS, index: bool = True) -> str:
    """Escape-safe HTML table; numeric columns (and their headers) are right-aligned."""
    if df is None or len(df.columns) == 0:
        return "<p class='muted'>No data.</p>"
    shown = df.head(max_rows)
    cols = list(shown.columns)
    # Column-wise extraction keeps dtypes (row-wise iteration would upcast ints to float).
    values = {c: shown[c].tolist() for c in cols}
    align = {
        c: "num" if all(_is_numlike(v) for v in values[c] if v is not None and not (
            isinstance(v, float) and math.isnan(v))) else "txt"
        for c in cols
    }
    head = ("<th></th>" if index else "") + "".join(
        f"<th scope='col' class='{align[c]}'>{_esc(c)}</th>" for c in cols)
    rows = []
    for i, idx in enumerate(shown.index):
        cells = []
        if index:
            cells.append(f"<th scope='row'>{_esc(_fmt(idx))}</th>")
        for c in cols:
            cells.append(f"<td class='{align[c]}'>{_esc(_fmt(values[c][i], str(c)))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = ""
    if len(df) > max_rows:
        note = f"<p class='muted small'>Showing {max_rows} of {len(df):,} rows.</p>"
    return (f"<div class='tw'><table class='grid'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>{note}")


# --------------------------------------------------------------------------------------
# chart rendering
# --------------------------------------------------------------------------------------
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "svg.fonttype": "none",
}


def _style_ax(ax: Any, t: _Theme, *, date_axis: bool = False) -> None:
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t.muted, labelcolor=t.ink2, length=3, width=0.6)
    ax.grid(True, axis="y", color=t.grid, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    if date_axis:
        loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _legend(ax: Any, t: _Theme, **kw: Any) -> None:
    leg = ax.legend(loc=kw.pop("loc", "upper left"), **kw)
    for txt in leg.get_texts():
        txt.set_color(t.ink2)


def _render_png(draw: DrawFn, t: _Theme, size: tuple[float, float], dpi: int) -> bytes:
    rc = {
        **_RC,
        "text.color": t.ink2,
        "axes.titlecolor": t.ink,
        "axes.labelcolor": t.ink2,
        "axes.edgecolor": t.axis,
        "xtick.color": t.muted,
        "ytick.color": t.muted,
        "xtick.labelcolor": t.ink2,
        "ytick.labelcolor": t.ink2,
    }
    with rc_context(rc):
        fig = Figure(figsize=size, dpi=dpi, facecolor=t.surface, layout="constrained")
        FigureCanvasAgg(fig)
        draw(fig, t)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=t.surface, metadata={"Software": None})
    return buf.getvalue()


class _Charts:
    """Renders a draw function once per theme and returns an ``<figure>`` block."""

    def __init__(self, dark: bool = True, dpi: int = 110) -> None:
        self.dark = dark
        self.dpi = dpi
        self.count = 0

    def html(self, draw: DrawFn, alt: str, *, size: tuple[float, float] = (10.0, 3.2),
             caption: str = "") -> str:
        try:
            light = base64.b64encode(_render_png(draw, _LIGHT, size, self.dpi)).decode("ascii")
            dark = (
                base64.b64encode(_render_png(draw, _DARK, size, self.dpi)).decode("ascii")
                if self.dark else None
            )
        except Exception:  # noqa: BLE001 - one broken chart must not kill the report
            logger.warning("chart rendering failed: %s", alt, exc_info=True)
            return f"<p class='muted'>Chart unavailable: {_esc(alt)}.</p>"
        self.count += 1
        imgs = f"<img class='light' src='data:image/png;base64,{light}' alt='{_esc(alt)}'>"
        if dark is not None:
            imgs += (f"<img class='dark' src='data:image/png;base64,{dark}' alt='{_esc(alt)}' "
                     "aria-hidden='true'>")
        cls = "chart duo" if dark is not None else "chart"
        cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        return f"<figure class='{cls}'>{imgs}{cap}</figure>"


def _money_axis(ax: Any, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _pct_axis(ax: Any, decimals: int = 0) -> None:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%"))


def _bar_limits(v: np.ndarray, span: float) -> tuple[float, float]:
    """Axis limits with room for end-of-bar labels only on the side(s) that have bars."""
    span = span if np.isfinite(span) and span > 0 else 1.0
    lo = float(np.nanmin(v)) if np.isfinite(v).any() else 0.0
    hi = float(np.nanmax(v)) if np.isfinite(v).any() else 0.0
    left = lo - 0.3 * span if lo < 0 else 0.0
    right = hi + 0.3 * span if hi > 0 else 0.05 * span
    return left, right


# ---- individual charts ----------------------------------------------------------------
def _draw_equity(eq: pd.Series, bench: pd.Series | None) -> DrawFn:
    eqp = _thin(eq)
    bp = _thin(bench) if bench is not None else None
    positive = bool((eq > 0).all() and (bench is None or (bench.dropna() > 0).all()))

    def draw(fig: Figure, t: _Theme) -> None:
        ax = fig.add_subplot(111)
        _style_ax(ax, t, date_axis=True)
        if bp is not None:
            ax.plot(_x(bp.index), bp.to_numpy(), color=t.neutral, linewidth=1.3,
                    label="Benchmark (rebased)")
        ax.plot(_x(eqp.index), eqp.to_numpy(), color=t.series, linewidth=1.8, label="Strategy")
        if positive:
            ax.set_yscale("log")
            lo = float(min(eq.min(), bench.min() if bench is not None else np.inf))
            hi = float(max(eq.max(), bench.max() if bench is not None else -np.inf))
            if hi / max(lo, 1e-12) < 20:
                ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
                ax.yaxis.set_minor_locator(mticker.NullLocator())
        _money_axis(ax)
        ax.set_ylabel("Equity (USD, log)" if positive else "Equity (USD)")
        ax.set_title("Equity curve")
        if bp is not None:
            _legend(ax, t)

    return draw


def _draw_drawdown(eq: pd.Series) -> DrawFn:
    dd = (eq / eq.cummax() - 1.0) * 100.0
    ddp = _thin(dd)
    i_min = dd.idxmin() if len(dd) else None

    def draw(fig: Figure, t: _Theme) -> None:
        ax = fig.add_subplot(111)
        _style_ax(ax, t, date_axis=True)
        x = _x(ddp.index)
        ax.fill_between(x, ddp.to_numpy(), 0.0, color=t.loss, alpha=0.22, linewidth=0)
        ax.plot(x, ddp.to_numpy(), color=t.loss, linewidth=1.1)
        ax.axhline(0.0, color=t.axis, linewidth=0.8)
        if i_min is not None and dd.loc[i_min] < 0:
            lo = float(dd.loc[i_min])
            ax.set_ylim(lo * 1.18, max(0.5, -0.04 * lo))
            xm = _x(pd.DatetimeIndex([i_min]))[0]
            ax.plot([xm], [lo], marker="o", markersize=5, color=t.loss,
                    markeredgecolor=t.surface, markeredgewidth=1.5)
            ax.annotate(f"Max drawdown {lo:.1f}%", xy=(xm, lo), xytext=(8, -3),
                        textcoords="offset points", color=t.ink2, fontsize=8.5, va="top")
        _pct_axis(ax, 0 if dd.min() < -3 else 1)
        ax.set_ylabel("Drawdown from peak")
        ax.set_title("Underwater (drawdown)")

    return draw


def _draw_rolling_sharpe(daily: pd.Series, window: int, full: float) -> DrawFn:
    mu = daily.rolling(window, min_periods=window).mean()
    sd = daily.rolling(window, min_periods=window).std(ddof=1)
    rs = (mu / sd.where(sd > 0) * math.sqrt(DAYS_PER_YEAR)).dropna()

    def draw(fig: Figure, t: _Theme) -> None:
        ax = fig.add_subplot(111)
        _style_ax(ax, t, date_axis=True)
        ax.axhline(0.0, color=t.axis, linewidth=0.9)
        if np.isfinite(full):
            ax.axhline(full, color=t.neutral, linewidth=1.0, label=f"Full period {full:.2f}")
        ax.plot(_x(rs.index), rs.to_numpy(), color=t.series, linewidth=1.6,
                label=f"Rolling {window}-day")
        ax.set_ylabel("Sharpe (annualised)")
        ax.set_title("Rolling Sharpe ratio")
        _legend(ax, t)

    return draw


def _draw_histogram(daily: pd.Series) -> DrawFn:
    r = daily.dropna().to_numpy() * 100.0
    var95 = float(np.quantile(r, 0.05)) if r.size else float("nan")

    def draw(fig: Figure, t: _Theme) -> None:
        ax = fig.add_subplot(111)
        _style_ax(ax, t)
        bins = int(np.clip(np.sqrt(r.size) * 1.5, 10, 60))
        ax.hist(r, bins=bins, color=t.series, alpha=0.9, edgecolor=t.surface, linewidth=0.8,
                density=True, label="Daily returns")
        if r.size > 2 and r.std() > 0:
            grid = np.linspace(r.min(), r.max(), 200)
            mu, sd = r.mean(), r.std(ddof=1)
            pdf = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            ax.plot(grid, pdf, color=t.ink2, linewidth=1.2, label="Normal fit")
        if np.isfinite(var95):
            ax.axvline(var95, color=t.loss, linewidth=1.2)
            ax.annotate(f"5% quantile {var95:.2f}%", xy=(var95, ax.get_ylim()[1] * 0.92),
                        xytext=(-6, 0), textcoords="offset points", ha="right",
                        color=t.ink2, fontsize=8.5)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.set_ylabel("Density")
        ax.set_title("Distribution of daily returns")
        _legend(ax, t, loc="upper right")

    return draw


def _draw_positions(pos: pd.Series, forecast: pd.Series | None) -> DrawFn:
    # Long intraday samples flip too often to read bar-by-bar: show the daily average.
    daily = len(pos) > 1500
    if daily:
        pp = pos.fillna(0.0).groupby(pos.index.floor("D")).mean()
        fp = (forecast.fillna(0.0).groupby(forecast.index.floor("D")).mean()
              if forecast is not None else None)
    else:
        pp = pos.fillna(0.0)
        fp = forecast.fillna(0.0) if forecast is not None else None
    pp = _thin(pp)
    fp = _thin(fp) if fp is not None else None
    suffix = " (daily average)" if daily else ""

    def draw(fig: Figure, t: _Theme) -> None:
        if fp is not None:
            ax1, ax2 = fig.subplots(2, 1, sharex=True, height_ratios=[3, 2])
        else:
            ax1, ax2 = fig.add_subplot(111), None
        _style_ax(ax1, t, date_axis=True)
        x = _x(pp.index)
        ax1.fill_between(x, pp.to_numpy(), 0.0, step="post", color=t.series, alpha=0.25,
                         linewidth=0)
        ax1.step(x, pp.to_numpy(), where="post", color=t.series, linewidth=1.0)
        ax1.axhline(0.0, color=t.axis, linewidth=0.8)
        ax1.set_ylabel("Lots (signed)")
        ax1.set_title("Position" + suffix)
        if ax2 is not None and fp is not None:
            _style_ax(ax2, t, date_axis=True)
            ax2.plot(_x(fp.index), fp.to_numpy(), color=t.ink2, linewidth=0.9)
            ax2.axhline(0.0, color=t.axis, linewidth=0.8)
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Forecast")
            ax2.set_title("Combined forecast [-1, 1]" + suffix)

    return draw


def _draw_costs(totals: pd.Series, cum_cost: pd.Series) -> DrawFn:
    cp = _thin(cum_cost)

    def draw(fig: Figure, t: _Theme) -> None:
        ax1, ax2 = fig.subplots(1, 2, width_ratios=[2, 3])
        _style_ax(ax1, t)
        ax1.grid(False)
        ax1.grid(True, axis="x", color=t.grid, linewidth=0.6)
        y = np.arange(len(totals))
        vals = totals.to_numpy(dtype=float)
        ax1.barh(y, vals, color=t.series, height=0.6)
        ax1.set_yticks(y, [str(i) for i in totals.index])
        ax1.invert_yaxis()
        ax1.axvline(0.0, color=t.axis, linewidth=0.8)
        span = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
        for yi, v in zip(y, vals, strict=True):
            ax1.annotate(_money(v, 0), xy=(v, yi), xytext=(4 if v >= 0 else -4, 0),
                         textcoords="offset points", va="center",
                         ha="left" if v >= 0 else "right", color=t.ink2, fontsize=8.5)
        ax1.set_xlim(*_bar_limits(vals, span))
        _money_axis(ax1, "x")
        ax1.set_title("Cost by component (USD)")
        _style_ax(ax2, t, date_axis=True)
        ax2.plot(_x(cp.index), cp.to_numpy(), color=t.series, linewidth=1.6)
        _money_axis(ax2)
        ax2.set_ylabel("USD")
        ax2.set_title("Cumulative net cost")

    return draw


def _draw_hbar(values: pd.Series, title: str, fmt: str = "{:.2f}") -> DrawFn:
    def draw(fig: Figure, t: _Theme) -> None:
        ax = fig.add_subplot(111)
        _style_ax(ax, t)
        ax.grid(False)
        ax.grid(True, axis="x", color=t.grid, linewidth=0.6)
        y = np.arange(len(values))
        v = values.to_numpy(dtype=float)
        ax.barh(y, v, color=t.series, height=0.6)
        ax.set_yticks(y, [str(i) for i in values.index])
        ax.invert_yaxis()
        ax.axvline(0.0, color=t.axis, linewidth=0.8)
        for yi, vi in zip(y, v, strict=True):
            if np.isfinite(vi):
                ax.annotate(fmt.format(vi), xy=(vi, yi), xytext=(4 if vi >= 0 else -4, 0),
                            textcoords="offset points", va="center",
                            ha="left" if vi >= 0 else "right", color=t.ink2, fontsize=8.5)
        span = float(np.nanmax(np.abs(v))) if len(v) and np.isfinite(v).any() else 1.0
        ax.set_xlim(*_bar_limits(v, span or 1.0))
        ax.set_title(title)

    return draw


def _draw_fold_bars(labels: list[str], values: np.ndarray, title: str) -> DrawFn:
    def draw(fig: Figure, t: _Theme) -> None:
        ax = fig.add_subplot(111)
        _style_ax(ax, t)
        x = np.arange(len(values))
        ax.bar(x, values, color=t.series, width=0.7)
        ax.axhline(0.0, color=t.axis, linewidth=0.9)
        step = max(1, len(labels) // 16)
        ax.set_xticks(x[::step], labels[::step], rotation=0)
        ax.set_title(title)

    return draw


# --------------------------------------------------------------------------------------
# section builders
# --------------------------------------------------------------------------------------
def _section(title: str, body: str, *, sid: str, lead: str = "") -> str:
    lead_html = f"<p class='lead'>{lead}</p>" if lead else ""
    return f"<section id='{_esc(sid)}'><h2>{_esc(title)}</h2>{lead_html}{body}</section>"


def _tiles(metrics: Mapping[str, Any], conf: Mapping[str, Any]) -> str:
    items = [
        ("Total return", _fmt(metrics.get("total_return"), "total_return")),
        ("CAGR", _fmt(metrics.get("cagr"), "cagr")),
        ("Sharpe (daily)", _fmt(metrics.get("sharpe"), "sharpe")),
        ("Max drawdown", _fmt(metrics.get("max_drawdown"), "max_drawdown")),
        ("Annual vol", _fmt(metrics.get("ann_vol"), "ann_vol")),
        ("PSR (SR > 0)", _fmt(conf.get("psr"), "win_rate") if conf.get("psr") is not None else "–"),
    ]
    cells = "".join(
        f"<div class='tile'><div class='tl'>{_esc(k)}</div><div class='tv'>{_esc(v)}</div></div>"
        for k, v in items
    )
    return f"<div class='tiles'>{cells}</div>"


def _monthly_table(daily: pd.Series) -> str:
    d = daily.dropna()
    if d.empty:
        return "<p class='muted'>No daily returns.</p>"
    idx = pd.DatetimeIndex(d.index)
    g = (1.0 + d).groupby([idx.year, idx.month]).prod() - 1.0
    yearly = (1.0 + d).groupby(idx.year).prod() - 1.0
    scale = float(np.nanquantile(np.abs(g.to_numpy()), 0.9)) if len(g) else 0.0
    scale = max(scale, 1e-4)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    head = "<th scope='col'>Year</th>" + "".join(f"<th scope='col'>{m}</th>" for m in months)
    head += "<th scope='col'>Year</th>"

    def cell(v: float | None) -> str:
        if v is None or not np.isfinite(v):
            return "<td class='m-na'></td>"
        lvl = int(min(4, math.ceil(abs(v) / scale * 4 - 1e-12))) if v != 0 else 0
        cls = "m-0" if lvl == 0 else (f"m-p{lvl}" if v > 0 else f"m-n{lvl}")
        return f"<td class='{cls}'>{v * 100:+.1f}</td>"

    rows = []
    for y in sorted(set(idx.year)):
        tds = "".join(cell(g.get((y, m))) for m in range(1, 13))
        rows.append(f"<tr><th scope='row'>{y}</th>{tds}{cell(yearly.get(y))}</tr>")
    return (
        "<div class='tw'><table class='monthly'><thead><tr>"
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='muted small'>Compounded monthly returns in %, from end-of-day (UTC) equity. "
        "Blue = gain, red = loss; intensity scales with the 90th percentile of |monthly return|.</p>"
    )


def _lots_traded(result: BacktestResult, pos: pd.Series) -> float:
    """Total |lots| traded, same convention as ``compute_metrics``' turnover.

    Uses the fills when available (they include intrabar stop / take-profit exits that the
    per-bar ``positions`` series cannot show); otherwise ``sum |diff(positions)|`` with the
    book starting FLAT, so the initial entry counts too.
    """
    fills = result.fills
    if fills is not None and len(fills) and "lots" in fills.columns:
        lots = pd.to_numeric(fills["lots"], errors="coerce")
        if lots.notna().any():
            return float(lots.abs().sum())
    p = pos.fillna(0.0).to_numpy(dtype=float)
    return float(np.abs(np.diff(p, prepend=0.0)).sum()) if p.size else 0.0


def _trade_stats(trades: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if trades is None or len(trades) == 0 or "pnl" not in trades:
        return None, None
    tr = trades.copy()
    tr["_pnl"] = pd.to_numeric(tr["pnl"], errors="coerce")
    tr["_sign"] = tr["side"].map(_side_sign) if "side" in tr else np.nan
    hold = None
    if "entry_time" in tr and "exit_time" in tr:
        try:
            hold = (pd.to_datetime(tr["exit_time"], utc=True) - pd.to_datetime(tr["entry_time"], utc=True))
            tr["_hold_h"] = hold.dt.total_seconds() / 3600.0
        except (TypeError, ValueError):
            hold = None

    def stats(sub: pd.DataFrame) -> dict[str, Any]:
        p = sub["_pnl"].dropna()
        wins, losses = p[p > 0], p[p < 0]
        gross_loss = -float(losses.sum())
        avg_win = float(wins.mean()) if len(wins) else float("nan")
        avg_loss = float(losses.mean()) if len(losses) else float("nan")
        return {
            "Trades": int(len(p)),
            "Win rate": f"{(p > 0).mean() * 100:.1f}%" if len(p) else "–",
            "Total PnL": _money(p.sum()),
            "Expectancy": _money(p.mean()) if len(p) else "–",
            "Average win": _money(avg_win) if np.isfinite(avg_win) else "–",
            "Average loss": _money(avg_loss) if np.isfinite(avg_loss) else "–",
            "Payoff ratio": _fmt(avg_win / -avg_loss) if len(wins) and len(losses) else "–",
            "Profit factor": _fmt(float(wins.sum()) / gross_loss) if gross_loss > 0 else "–",
            "Largest win": _money(p.max()) if len(p) else "–",
            "Largest loss": _money(p.min()) if len(p) else "–",
            "Avg hold (h)": _fmt(float(sub["_hold_h"].mean())) if "_hold_h" in sub and len(sub) else "–",
            "Costs": _money(pd.to_numeric(sub['costs'], errors='coerce').sum()) if "costs" in sub else "–",
            "Swap": _money(pd.to_numeric(sub['swap'], errors='coerce').sum()) if "swap" in sub else "–",
        }

    cols = {"All": stats(tr)}
    if tr["_sign"].notna().any():
        cols["Long"] = stats(tr[tr["_sign"] > 0])
        cols["Short"] = stats(tr[tr["_sign"] < 0])
    summary = pd.DataFrame(cols)
    reasons = None
    if "exit_reason" in tr:
        grp = tr.groupby(tr["exit_reason"].astype(str))["_pnl"]
        counts = grp.size().sort_values(ascending=False)
        reasons = pd.DataFrame(
            {
                "Trades": [int(counts[k]) for k in counts.index],
                "Total PnL": [_money(grp.get_group(k).sum()) for k in counts.index],
                "Win rate": [f"{(grp.get_group(k) > 0).mean() * 100:.1f}%" for k in counts.index],
            },
            index=counts.index,
        )
    return summary, reasons


def _confidence_block(conf: Mapping[str, Any], pbo: Any, years: float) -> str:
    if "sharpe" not in conf:
        return "<p class='muted'>Not enough daily observations for statistical inference.</p>"
    lvl = int(round((1 - conf.get("ci_alpha", 0.05)) * 100))
    n_trials = int(conf.get("n_trials", 1))
    rows = [
        ("Sharpe (annualised, daily returns)", _esc(_fmt(conf.get("sharpe")))),
        ("Standard error (Mertens/Opdyke, non-normal)", _esc(_fmt(conf.get("sharpe_se")))),
    ]
    if "ci_lower" in conf:
        rows.append((
            f"{lvl}% stationary-bootstrap CI",
            _esc(f"[{_fmt(conf['ci_lower'])}, {_fmt(conf['ci_upper'])}]  "
                 f"(mean block {conf.get('bootstrap_block', float('nan')):.1f} days)"),
        ))
        rows.append(("Bootstrap P[Sharpe ≤ 0]", _esc(_fmt(conf.get("bootstrap_prob_le_zero")))))
    rows += [
        ("Probabilistic Sharpe Ratio P[SR > 0]", _esc(_fmt(conf.get("psr")))),
        (f"Deflated Sharpe Ratio (N = {n_trials} trials)", _esc(_fmt(conf.get("dsr")))),
        ("Expected max Sharpe of N skill-less trials (SR₀)", _esc(_fmt(conf.get("sr0")))),
        ("Minimum track record for 95% confidence",
         _esc(f"{_fmt(conf.get('min_trl_years'))} years (have {years:.2f})")),
        ("Skewness / kurtosis (Pearson) of daily returns",
         _esc(f"{_fmt(conf.get('skew'))} / {_fmt(conf.get('kurtosis'))}")),
    ]
    if "sharpe_haircut" in conf:
        rows.append((f"Haircut Sharpe (Harvey–Liu, BHY, N = {n_trials})",
                     _esc(f"{_fmt(conf['sharpe_haircut'])} (haircut {_fmt(conf['haircut'], 'haircut')})")))
    if pbo is not None:
        rows.append(("Probability of backtest overfitting (CSCV)", _esc(_fmt(_pbo_value(pbo)))))
    dsr = conf.get("dsr", float("nan"))
    if np.isfinite(dsr):
        verdict = (
            "The Sharpe ratio is significant at the 95% level after deflating for "
            f"{n_trials} trial(s)." if dsr >= 0.95 else
            f"The Sharpe ratio is <strong>not</strong> significant at the 95% level after "
            f"deflating for {n_trials} trial(s) (DSR {dsr:.2f})."
        )
    else:
        verdict = "Deflated Sharpe ratio unavailable."
    note = ("" if n_trials > 1 else
            " Pass <code>n_trials</code> (the number of configurations tried) for a meaningful DSR;"
            " with one trial DSR equals PSR.")
    return _kv_table(rows) + f"<p class='small'>{verdict}{note}</p>"


def _pbo_value(pbo: Any) -> float:
    if isinstance(pbo, rstats.PBOResult):
        return pbo.pbo
    if isinstance(pbo, Mapping):
        return float(pbo.get("pbo", float("nan")))
    try:
        return float(pbo)
    except (TypeError, ValueError):
        return float("nan")


def _split_reasons(r: Any) -> list[str]:
    """Normalise one ``risk_events['reasons']`` cell to a list of reason strings.

    ``aurum.backtest.engine`` stores the reasons of a bar as ONE ``"; "``-joined string;
    other producers may use a list. Empty cells (e.g. a pure size change) become
    ``"(unspecified)"`` so every event is still counted.
    """
    if isinstance(r, (list, tuple, np.ndarray)):
        items = [str(x) for x in r]
    elif r is None or (pd.api.types.is_scalar(r) and pd.isna(r)):
        items = []
    else:
        items = str(r).split(";")
    items = [x.strip() for x in items if str(x).strip()]
    return items or ["(unspecified)"]


def _risk_events_table(ev: pd.DataFrame | None) -> str | None:
    if ev is None or len(ev) == 0:
        return None
    if "reasons" not in ev:
        return _df_table(ev.tail(_MAX_TABLE_ROWS), index=False)
    reasons = ev["reasons"].map(_split_reasons)
    times = pd.to_datetime(ev["time"], utc=True) if "time" in ev else None
    frame = pd.DataFrame({"reason": reasons})
    if times is not None:
        frame["time"] = times.to_numpy()
    frame = frame.explode("reason")
    # Collapse numeric details ("spread 0.9 > 0.5") to their leading rule keyword for counting.
    reason_str = frame["reason"].astype(str)
    frame["rule"] = reason_str.str.extract(r"^\s*([A-Za-z_(][\w\-)]*)", expand=False).fillna(reason_str)
    agg: dict[str, Any] = {"reason": "size"}
    if times is not None:
        agg["time"] = ["min", "max"]
    out = frame.groupby("rule").agg(agg)
    out.columns = ["Events", "First", "Last"] if times is not None else ["Events"]
    return _df_table(out.sort_values("Events", ascending=False))


def _extra_block(key: str, value: Any, charts: _Charts) -> str:
    """Generic renderer for unrecognised ``extra`` entries."""
    title = key.replace("_", " ").strip().capitalize()
    if isinstance(value, pd.DataFrame):
        body = _df_table(value)
    elif isinstance(value, pd.Series):
        body = _df_table(value.to_frame())
    elif isinstance(value, Figure):
        buf = io.BytesIO()
        # savefig switches to an Agg print method for PNG by itself; re-binding the canvas
        # (FigureCanvasAgg(value)) would detach a pyplot figure from its GUI manager.
        value.savefig(buf, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        body = f"<figure class='chart'><img src='data:image/png;base64,{b64}' alt='{_esc(title)}'></figure>"
    elif isinstance(value, Mapping):
        body = _kv_table([(str(k), _esc(_fmt(v, str(k)))) for k, v in value.items()])
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        body = "<ul>" + "".join(f"<li>{_esc(v)}</li>" for v in value) + "</ul>"
    elif isinstance(value, str):
        body = "".join(f"<p>{_esc(p)}</p>" for p in value.split("\n\n"))
    else:
        body = f"<pre>{_esc(repr(value))}</pre>"
    return f"<h3>{_esc(title)}</h3>{body}"


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def _prepare_benchmark(benchmark: Any, eq: pd.Series) -> pd.Series | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, pd.DataFrame):
        if "close" not in benchmark:
            raise ValueError("benchmark DataFrame must have a 'close' column")
        benchmark = benchmark["close"]
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pandas Series (price or equity) or a bars DataFrame")
    b = _utc_index(benchmark.astype(float)).dropna()
    b = b[~b.index.duplicated(keep="last")]
    if b.empty:
        return None
    # Last known benchmark value at or before each equity timestamp (never back-filled).
    aligned = b.reindex(b.index.union(eq.index)).ffill().reindex(eq.index)
    first = aligned.first_valid_index()
    if first is None:
        return None
    aligned = aligned.loc[first:]
    return aligned / aligned.iloc[0] * float(eq.loc[first])


def tearsheet_html(
    result: BacktestResult,
    *,
    benchmark: pd.Series | pd.DataFrame | None = None,
    title: str = "",
    extra: Mapping[str, Any] | None = None,
    n_trials: int | None = None,
    dark_charts: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> str:
    """Build the tearsheet HTML string (see :func:`write_tearsheet`)."""
    extra = dict(extra or {})
    charts = _Charts(dark=dark_charts)
    eq = _utc_index(result.equity.astype(float)).dropna()
    if eq.empty:
        raise ValueError("result.equity is empty")
    initial = _initial_equity(result)
    if not np.isfinite(initial) or initial <= 0:
        initial = float(eq.iloc[0])
    daily = daily_returns_from_equity(eq, initial)
    metrics, metrics_source = _resolve_metrics(result, daily, initial)
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)

    extra_trials = extra.pop("n_trials", None)  # always consumed: never re-rendered generically
    trials = n_trials if n_trials is not None else extra_trials
    trial_sharpes = extra.pop("trial_sharpes", None)
    pbo = extra.pop("pbo", None)
    conf: dict[str, Any] = {}
    if daily.dropna().size >= 20:
        try:
            conf = rstats.sharpe_summary(
                daily.to_numpy(), DAYS_PER_YEAR, n_trials=int(trials) if trials else None,
                trial_sharpes_ann=trial_sharpes, n_boot=n_boot, seed=seed,
            )
        except Exception:  # noqa: BLE001
            logger.warning("statistical confidence block failed", exc_info=True)

    title = title or str(result.meta.get("name") or result.meta.get("strategy") or "Backtest tearsheet")
    tf = result.meta.get("timeframe") or result.equity.attrs.get("timeframe") or ""
    period = f"{eq.index[0]:%Y-%m-%d} → {eq.index[-1]:%Y-%m-%d}"
    sub = (f"{_esc(period)} · {len(eq):,} bars{(' · ' + _esc(tf)) if tf else ''} · "
           f"{years:.2f} years · equity {_money(initial, 0)} → {_money(float(eq.iloc[-1]), 0)}")
    parts: list[str] = [f"<header><h1>{_esc(title)}</h1><p class='sub'>{sub}</p></header>",
                        _tiles(metrics, conf)]

    # --- equity & benchmark ---------------------------------------------------------
    bench = _prepare_benchmark(benchmark, eq)
    body = charts.html(_draw_equity(eq, bench), "Strategy equity curve"
                       + (" versus rebased benchmark" if bench is not None else "")
                       + " on a log scale", size=(10.0, 3.6))
    if bench is not None:
        bd = daily_returns_from_equity(bench, float(bench.iloc[0]))
        joined = pd.concat([daily.rename("s"), bd.rename("b")], axis=1).dropna()
        bstats: list[tuple[str, str]] = [
            ("Benchmark total return", _esc(_fmt(float(bench.iloc[-1] / bench.iloc[0] - 1), "total_return"))),
            ("Benchmark Sharpe (daily)", _esc(_fmt(rstats.sharpe(bd, DAYS_PER_YEAR)))),
            ("Benchmark max drawdown", _esc(_fmt(float((bench / bench.cummax() - 1).min()), "max_drawdown"))),
        ]
        if len(joined) > 5 and joined["b"].var() > 0:
            beta = float(joined.cov().loc["s", "b"] / joined["b"].var())
            bstats += [("Correlation (daily)", _esc(_fmt(float(joined.corr().loc["s", "b"])))),
                       ("Beta (daily)", _esc(_fmt(beta)))]
        body += _kv_table(bstats)
    parts.append(_section("Equity", body, sid="equity"))
    parts.append(_section("Drawdown", charts.html(_draw_drawdown(eq), "Underwater drawdown chart"),
                          sid="drawdown"))

    # --- rolling sharpe ---------------------------------------------------------------
    nd = int(daily.dropna().size)
    if nd >= 40:
        window = 126 if nd >= 126 + 20 else max(20, nd // 3)
        parts.append(_section(
            "Rolling Sharpe",
            charts.html(_draw_rolling_sharpe(daily.dropna(), window, float(metrics.get("sharpe", np.nan))),
                        f"Rolling {window}-day annualised Sharpe ratio"),
            sid="rolling",
            lead=("6-month (126 trading days) window." if window == 126 else
                  f"Short sample: {window}-day window instead of 6 months."),
        ))
    parts.append(_section("Monthly returns", _monthly_table(daily), sid="monthly"))
    if nd >= 10:
        parts.append(_section("Return distribution",
                              charts.html(_draw_histogram(daily), "Histogram of daily returns",
                                          size=(10.0, 3.0)), sid="distribution"))

    # --- positions --------------------------------------------------------------------
    if result.positions is not None and len(result.positions):
        pos = _utc_index(result.positions.astype(float))
        fc = None
        if result.forecast is not None and result.forecast.notna().any():
            fc = _utc_index(result.forecast.astype(float))
        pos_abs = pos.abs()
        prow = [
            ("Time in market", _esc(_fmt(float((pos != 0).mean()), "exposure"))),
            ("Long / short share of bars",
             _esc(f"{(pos > 0).mean() * 100:.1f}% / {(pos < 0).mean() * 100:.1f}%")),
            ("Average |lots| when invested",
             _esc(_fmt(float(pos_abs[pos_abs > 0].mean())) if (pos_abs > 0).any() else "–")),
            ("Max |lots|", _esc(_fmt(float(pos_abs.max())))),
            ("Lots traded", _esc(_fmt(_lots_traded(result, pos)))),
        ]
        parts.append(_section(
            "Position & exposure",
            charts.html(_draw_positions(pos, fc), "Signed position in lots over time"
                        + (" with the combined forecast" if fc is not None else ""),
                        size=(10.0, 4.2 if fc is not None else 3.0)) + _kv_table(prow),
            sid="positions",
        ))

    # --- costs ------------------------------------------------------------------------
    costs = result.costs
    if costs is not None and len(costs) and len(costs.columns):
        c = _utc_index(costs.apply(pd.to_numeric, errors="coerce")).fillna(0.0)
        comp = {}
        for col, lab in (("spread", "Spread"), ("slippage", "Slippage"), ("commission", "Commission")):
            if col in c:
                comp[lab] = float(c[col].sum())
        if "swap" in c:
            comp["Swap (net paid)"] = -float(c["swap"].sum())
        totals = pd.Series(comp)
        net = c.reindex(columns=["spread", "slippage", "commission"], fill_value=0.0).sum(axis=1)
        if "swap" in c:
            net = net - c["swap"]
        cum = net.cumsum()
        total_net = float(totals.sum())
        gross_pnl = float(eq.iloc[-1] - initial) + total_net
        crow = [(k, _esc(f"{_money(v)}  ({v / initial * 100:.2f}% of initial equity)"))
                for k, v in comp.items()]
        crow.append(("Net cost", _esc(f"{_money(total_net)}  ({total_net / initial * 100:.2f}%)")))
        if years > 0:
            crow.append(("Cost drag per year", _esc(f"{total_net / initial / years * 100:.2f}% of initial equity")))
        crow.append(("PnL before costs → after",
                     _esc(f"{_money(gross_pnl)} → {_money(float(eq.iloc[-1] - initial))}")))
        body = (charts.html(_draw_costs(totals, cum), "Cost attribution by component and cumulative cost",
                            size=(10.0, 3.0)) if len(totals) else "") + _kv_table(crow)
        parts.append(_section("Cost attribution", body, sid="costs",
                              lead="Costs in USD (positive = paid). Swap is shown as net paid "
                                   "(negative = financing received)."))

    # --- trades -----------------------------------------------------------------------
    summary, reasons = _trade_stats(result.trades)
    if summary is not None:
        body = _df_table(summary)
        if reasons is not None and len(reasons):
            body += "<h3>By exit reason</h3>" + _df_table(reasons)
        parts.append(_section("Trade statistics", body, sid="trades"))
    else:
        parts.append(_section("Trade statistics", "<p class='muted'>No closed trades.</p>", sid="trades"))

    # --- statistical confidence -------------------------------------------------------
    parts.append(_section(
        "Statistical confidence", _confidence_block(conf, pbo, years), sid="confidence",
        lead="Is the Sharpe ratio distinguishable from luck? PSR/DSR: Bailey & López de Prado "
             "(2012, 2014); bootstrap: Politis & Romano (1994) with Politis–White block length; "
             "haircut: Harvey & Liu (2015).",
    ))

    # --- walk-forward folds -----------------------------------------------------------
    folds = extra.pop("folds", None)
    if isinstance(folds, pd.DataFrame) and len(folds):
        body = ""
        sh_col = next((col for col in folds.columns if str(col).lower() in
                       ("sharpe", "oos_sharpe", "sharpe_oos", "test_sharpe")), None)
        if sh_col is not None:
            vals = pd.to_numeric(folds[sh_col], errors="coerce").to_numpy(dtype=float)
            labels = [str(i) for i in folds.index]
            body += charts.html(_draw_fold_bars(labels, vals, "Out-of-sample Sharpe by fold"),
                                f"Bar chart of {sh_col} per walk-forward fold", size=(10.0, 2.8))
            fin = vals[np.isfinite(vals)]
            if fin.size:
                body += (f"<p class='small'>Positive in {np.mean(fin > 0) * 100:.0f}% of "
                         f"{fin.size} fold(s) with a finite {_esc(sh_col)}.</p>")
        body += _df_table(folds)
        parts.append(_section("Walk-forward folds", body, sid="folds"))

    # --- strategy weights -------------------------------------------------------------
    weights = extra.pop("weights", None)
    if weights is not None:
        if isinstance(weights, Mapping):
            weights = pd.Series(dict(weights), dtype=float)
        if isinstance(weights, pd.Series) and len(weights):
            w = weights.astype(float).sort_values(ascending=False)
            body = charts.html(_draw_hbar(w, "Combiner weights"), "Strategy weights bar chart",
                               size=(10.0, max(1.6, 0.32 * len(w) + 0.8)))
            body += _df_table(w.rename("weight").to_frame())
        elif isinstance(weights, pd.DataFrame):
            body = _df_table(weights)
        else:
            body = f"<pre>{_esc(repr(weights))}</pre>"
        parts.append(_section("Strategy weights", body, sid="weights"))

    # --- risk events ------------------------------------------------------------------
    rtable = _risk_events_table(result.risk_events)
    if rtable is not None:
        parts.append(_section("Risk-manager interventions", rtable, sid="risk",
                              lead=f"{len(result.risk_events):,} bars where the risk manager "
                                   "modified or blocked the requested position."))

    # --- full metrics -----------------------------------------------------------------
    mrows = [(_label(k), _esc(_fmt(v, k))) for k, v in metrics.items()]
    parts.append(_section("All metrics", _kv_table(mrows, cls="kv cols"), sid="metrics",
                          lead=f"Source: {_esc(metrics_source)}."))

    # --- provenance, notes, extra -----------------------------------------------------
    notes = extra.pop("notes", None)
    config = extra.pop("config", None)
    tail = ""
    if result.meta:
        tail += "<h3>Run provenance</h3>" + _kv_table(
            [(str(k), _esc(_fmt(v, str(k)))) for k, v in result.meta.items()])
    if config is not None:
        tail += "<h3>Config</h3><pre>" + _esc(json.dumps(config, indent=2, default=str)) + "</pre>"
    if notes:
        tail += _extra_block("notes", notes, charts)
    for key, value in extra.items():
        tail += _extra_block(str(key), value, charts)
    if tail:
        parts.append(_section("Notes & provenance", tail, sid="notes"))

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"<footer>Generated {generated} by aurum.research.report · "
                 f"{charts.count} charts · Sharpe annualised with {DAYS_PER_YEAR} days/year.</footer>")
    return _PAGE.format(title=_esc(title), css=_CSS, body="\n".join(parts))


def write_tearsheet(
    result: BacktestResult,
    path: str | Path,
    *,
    benchmark: pd.Series | pd.DataFrame | None = None,
    title: str = "",
    extra: Mapping[str, Any] | None = None,
    n_trials: int | None = None,
    dark_charts: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> Path:
    """Write a single self-contained HTML tearsheet and return its path.

    Parameters
    ----------
    result      : the backtest to report. ``result.metrics`` is used when populated;
                  otherwise ``aurum.backtest.metrics.compute_metrics`` (if importable),
                  otherwise a minimal built-in fallback.
    path        : output file; a directory (or a path without suffix) gets
                  ``tearsheet.html`` inside it. Parent directories are created.
    benchmark   : optional price/equity Series (or bars frame, ``close`` is used), rebased
                  to the strategy's equity at the first common timestamp.
    extra       : optional sections. Recognised keys: ``"folds"`` (walk-forward DataFrame;
                  a ``sharpe``/``oos_sharpe`` column is charted), ``"weights"`` (dict /
                  Series / DataFrame), ``"n_trials"`` (int, number of configurations tried,
                  for DSR and haircut), ``"trial_sharpes"`` (annualised Sharpe ratios of all
                  trials - sets the DSR trial variance), ``"pbo"`` (float or
                  :class:`~aurum.research.stats.PBOResult`), ``"notes"`` (str or list of
                  str), ``"config"`` (JSON-able). Other keys are rendered generically
                  (DataFrame/Series -> table, dict -> key/value table, str -> text,
                  matplotlib Figure -> image).
    n_trials    : overrides ``extra["n_trials"]``.
    dark_charts : also embed dark-surface renders of each chart (≈2x file size).
    n_boot, seed: stationary-bootstrap replications and seed for the Sharpe CI.
    """
    out = Path(path)
    if out.is_dir() or out.suffix == "":
        out = out / "tearsheet.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = tearsheet_html(result, benchmark=benchmark, title=title, extra=extra, n_trials=n_trials,
                         dark_charts=dark_charts, n_boot=n_boot, seed=seed)
    out.write_text(doc, encoding="utf-8")
    logger.info("tearsheet written to %s (%.0f KB)", out, out.stat().st_size / 1024)
    return out


# --------------------------------------------------------------------------------------
# page template
# --------------------------------------------------------------------------------------
_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#6f6d68;--grid:#e1e0d9;--border:rgba(11,11,11,.10);--accent:#2a78d6;--stripe:rgba(11,11,11,.025);
--m0:#f0efec;--p1:#e6f0fc;--p2:#cde2fb;--p3:#9ec5f4;--p4:#6da7ec;
--n1:#fbe7e6;--n2:#f8cfce;--n3:#f2a3a2;--n4:#eb7a79}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#9a988f;--grid:#2c2c2a;--border:rgba(255,255,255,.10);
--accent:#3987e5;--stripe:rgba(255,255,255,.03);--m0:#383835;--p1:#1e2a38;--p2:#1c3a63;--p3:#1c4d8a;
--p4:#1c5cab;--n1:#3a2323;--n2:#5a2a2a;--n3:#7a2e2e;--n4:#9a3434}
:root:not([data-theme="light"]) .chart.duo img.light{display:none}
:root:not([data-theme="light"]) .chart.duo img.dark{display:block}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;
--muted:#9a988f;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5;--stripe:rgba(255,255,255,.03);
--m0:#383835;--p1:#1e2a38;--p2:#1c3a63;--p3:#1c4d8a;--p4:#1c5cab;--n1:#3a2323;--n2:#5a2a2a;
--n3:#7a2e2e;--n4:#9a3434}
:root[data-theme="dark"] .chart.duo img.light{display:none}
:root[data-theme="dark"] .chart.duo img.dark{display:block}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
header h1{font-size:24px;line-height:1.25;margin:0 0 4px;font-weight:650;letter-spacing:-.01em}
.sub{margin:0 0 16px;color:var(--ink2)}
h2{font-size:16px;margin:0 0 8px;font-weight:650}
h3{font-size:14px;margin:16px 0 6px;font-weight:600;color:var(--ink2)}
section{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:16px;margin:16px 0}
.lead{margin:0 0 10px;color:var(--ink2)}
.muted{color:var(--muted)}.small{font-size:12.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 4px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tl{font-size:12px;color:var(--ink2)}.tv{font-size:22px;font-weight:600;margin-top:2px}
figure.chart{margin:4px 0 8px;overflow-x:auto;-webkit-overflow-scrolling:touch}
figure.chart img{display:block;width:100%;height:auto;border-radius:6px}
figure.chart.duo img.dark{display:none}
@media (max-width:640px){figure.chart img{width:620px;max-width:none}}
figcaption{font-size:12px;color:var(--muted)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
table.kv{width:100%}
table.kv th{text-align:left;font-weight:500;color:var(--ink2);padding:5px 12px 5px 0;
border-bottom:1px solid var(--grid);width:55%}
table.kv td{padding:5px 0;border-bottom:1px solid var(--grid);text-align:right}
table.kv.cols{display:block}
@media (min-width:760px){table.kv.cols tbody{display:grid;grid-template-columns:1fr 1fr;column-gap:32px}
table.kv.cols tr{display:flex;justify-content:space-between;border-bottom:1px solid var(--grid)}
table.kv.cols th,table.kv.cols td{border-bottom:0;width:auto}}
table.grid{width:100%;min-width:480px}
table.grid th,table.grid td{padding:5px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
table.grid thead th{text-align:left;color:var(--ink2);font-weight:600;font-size:12px}
table.grid thead th.num{text-align:right}
table.grid tbody th{text-align:left;font-weight:500;color:var(--ink2)}
table.grid td.num{text-align:right}table.grid td.txt{text-align:left}
table.grid th.num+th.txt,table.grid td.num+td.txt{padding-left:20px}
table.grid tbody tr:nth-child(even){background:var(--stripe)}
table.monthly{width:100%;min-width:640px;table-layout:fixed}
table.monthly th,table.monthly td{padding:6px 4px;text-align:center;font-size:12.5px}
table.monthly thead th{color:var(--ink2);font-weight:600}
table.monthly tbody th{color:var(--ink2);font-weight:600}
table.monthly td{border:2px solid var(--surface);border-radius:4px}
.m-0{background:var(--m0)}.m-p1{background:var(--p1)}.m-p2{background:var(--p2)}
.m-p3{background:var(--p3)}.m-p4{background:var(--p4)}.m-n1{background:var(--n1)}
.m-n2{background:var(--n2)}.m-n3{background:var(--n3)}.m-n4{background:var(--n4)}
pre{background:var(--page);border:1px solid var(--border);border-radius:6px;padding:10px;
overflow-x:auto;font-size:12px}
code{font-size:12.5px}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:24px}
@media print{body{background:#fff}section{break-inside:avoid}}
"""

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""
