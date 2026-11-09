"""Interactive walk-forward dashboard: one self-contained Plotly HTML page per run directory.

``build_dashboard(run_dir, out_html)`` reads what :meth:`aurum.research.walkforward.
WalkForwardReport.save` wrote (``summary.json``, ``folds.csv``, ``weights.csv``,
``costs.csv``, ``oos_forecasts.parquet``, ``books/<book>/{timeseries.parquet,trades.csv}``
and the optional ``holdout/`` directory) and renders:

* a KPI header (Sharpe with its bootstrap CI, CAGR, max drawdown, DSR, PSR, PBO, volatility,
  costs) with the buy-and-hold comparison and the protocol's pass/fail badges;
* equity vs benchmark (log scale, fold boundaries), underwater drawdown, rolling Sharpe and
  per-fold OOS Sharpe;
* per-strategy OOS forecasts as small multiples, combiner weights per fold (stacked, with
  the unallocated share), and the per-book statistics table;
* cost attribution (a gross-to-net waterfall for the combined book and per-book costs);
* the monthly returns heatmap (with a table view), and the combined book's trades;
* the final holdout (if the run evaluated one) and, given ``journal_dir``, the LLM desk
  timeline (quant vs final forecast per cycle, agents consulted / created, decisions).

Every chart has hover tooltips; the page has a light/dark toggle (initially the OS setting)
that re-themes the charts with the same palette as :mod:`aurum.viz.style`, and a
responsive layout down to phone width. Text taken from run files (strategy names, notes,
LLM rationales) is HTML-escaped. Data is downsampled to one point per trading day, so a
9-year H1 run is ~0.5 MB of data plus plotly.js (~4.8 MB inline, or a CDN tag with
``include_plotlyjs="cdn"``).

Requires the optional ``plotly`` dependency (``pip install aurum[viz]``).
"""

from __future__ import annotations

import html
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.viz.style import (
    FONT_STACK_CSS,
    get_palette,
    hex_to_rgba,
    palette,
    plotly_template,
)

logger = logging.getLogger(__name__)

__all__ = ["RunData", "build_dashboard", "load_desk_journal", "load_run"]

_COMBINED = "combined"
_BENCHMARK = "benchmark"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                  "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("aurum.viz.dashboard needs plotly: pip install 'aurum[viz]' (plotly>=5.20)") from exc
    return go


# =======================================================================================
# loading
# =======================================================================================
@dataclass
class RunData:
    """Everything the dashboard reads from a run directory (missing pieces are ``None``)."""

    run_dir: Path
    summary: dict[str, Any]
    books: dict[str, pd.DataFrame] = field(default_factory=dict)        # timeseries per book
    trades: pd.DataFrame | None = None                                  # combined book trades
    forecasts: pd.DataFrame | None = None                               # stitched OOS forecasts
    folds: pd.DataFrame | None = None
    weights: pd.DataFrame | None = None
    costs: pd.DataFrame | None = None
    holdout_books: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.summary.get("name") or self.run_dir.name)

    @property
    def strategies(self) -> list[str]:
        """Strategy keys in config order (from the stats table, else the forecast columns)."""
        stats = self.summary.get("stats") or {}
        keys = [k for k in stats if k not in (_COMBINED, _BENCHMARK)]
        if not keys and self.forecasts is not None:
            keys = [c for c in self.forecasts.columns if c not in (_COMBINED, "fold")]
        return keys


def _read_csv(path: Path, **kw: Any) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("dashboard: cannot read %s: %s", path, exc)
        return None


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        logger.warning("dashboard: cannot read %s: %s", path, exc)
        return None


def _read_books(books_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if not books_dir.is_dir():
        return out
    for d in sorted(p for p in books_dir.iterdir() if p.is_dir()):
        ts = _read_parquet(d / "timeseries.parquet")
        if ts is not None and "equity" in ts.columns and len(ts):
            out[d.name] = ts
    return out


def load_run(run_dir: str | Path) -> RunData:
    """Read a walk-forward / backtest run directory written by ``WalkForwardReport.save``."""
    d = Path(run_dir)
    p = d / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (is {d} a walkforward/backtest run directory?)")
    summary = json.loads(p.read_text(encoding="utf-8"))
    books = _read_books(d / "books")
    trades = _read_csv(d / "books" / _COMBINED / "trades.csv")
    return RunData(
        run_dir=d, summary=summary, books=books, trades=trades,
        forecasts=_read_parquet(d / "oos_forecasts.parquet"),
        folds=_read_csv(d / "folds.csv"), weights=_read_csv(d / "weights.csv"),
        costs=_read_csv(d / "costs.csv", index_col=0),
        holdout_books=_read_books(d / "holdout" / "books"),
    )


def _as_obj(v: Any) -> Any:
    """Journal payloads are JSON; tolerate stringified dicts from older writers."""
    return v if isinstance(v, (dict, list)) else None


def load_desk_journal(journal_dir: str | Path) -> pd.DataFrame:
    """One row per LLM desk cycle from a directory of ``CycleJournal`` JSONL files.

    Columns: ``cycle_id, now, status, quant_forecast, final_forecast, action, confidence,
    rationale, consulted (list), created (list), n_memos, cost_usd, duration_s,
    failure_reason, file``. Malformed lines are skipped; an empty frame is returned when no
    cycle is found.
    """
    cols = ["cycle_id", "now", "status", "quant_forecast", "final_forecast", "action", "confidence",
            "rationale", "consulted", "created", "n_memos", "cost_usd", "duration_s", "failure_reason", "file"]
    root = Path(journal_dir)
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else ([root] if root.is_file() else [])
    rows: list[dict[str, Any]] = []
    for f in files:
        cycles: dict[str, dict[str, Any]] = {}
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("dashboard: cannot read journal %s: %s", f, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict) or "event" not in e:
                continue
            cid = str(e.get("cycle_id") or f.stem)
            c = cycles.setdefault(cid, {"cycle_id": cid, "consulted": [], "created": [], "file": f.name,
                                        "status": None, "now": None, "quant_forecast": None,
                                        "final_forecast": None, "action": None, "confidence": None,
                                        "rationale": "", "n_memos": 0, "cost_usd": None,
                                        "duration_s": None, "failure_reason": None})
            ev = e["event"]
            if ev == "cycle_start":
                c["now"] = e.get("now")
                c["quant_forecast"] = e.get("quant_forecast")
            elif ev == "agent_start":
                agent = str(e.get("agent") or "")
                role = str(e.get("role") or "")
                if agent and agent != "chief" and role != "adhoc" and not agent.startswith("adhoc:"):
                    if agent not in c["consulted"]:
                        c["consulted"].append(agent)
            elif ev == "agent_created":
                name = str(e.get("name") or e.get("agent") or "")
                if name and name not in c["created"]:
                    c["created"].append(name)
            elif ev == "memo":
                c["n_memos"] += 1
            elif ev == "decision":
                dec = _as_obj(e.get("decision")) or {}
                c["action"] = dec.get("action")
                c["confidence"] = dec.get("confidence")
                c["rationale"] = str(dec.get("rationale") or "")
                c["failure_reason"] = e.get("failure_reason")
            elif ev == "cycle_end":
                c["status"] = e.get("status")
                c["final_forecast"] = e.get("final_forecast")
                if c["quant_forecast"] is None:
                    c["quant_forecast"] = e.get("quant_forecast")
                c["duration_s"] = e.get("duration_s")
                usage = _as_obj(e.get("usage")) or {}
                total = usage.get("total") if isinstance(usage, dict) else None
                if isinstance(total, dict):
                    c["cost_usd"] = total.get("cost_usd")
        rows.extend(cycles.values())
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["now"] = pd.to_datetime(df["now"], utc=True, errors="coerce")
    for k in ("quant_forecast", "final_forecast", "confidence", "cost_usd", "duration_s"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["now"]).sort_values(["now", "cycle_id"]).reset_index(drop=True)
    return df


# =======================================================================================
# series helpers
# =======================================================================================
def _daily_equity(ts: pd.DataFrame) -> pd.Series:
    from aurum.backtest.metrics import daily_equity

    eq = daily_equity(ts["equity"])
    eq.index = pd.DatetimeIndex(eq.index)
    return eq


def _daily_returns(ts: pd.DataFrame) -> pd.Series:
    from aurum.backtest.metrics import daily_returns

    r = daily_returns(ts["equity"])
    r.index = pd.DatetimeIndex(r.index)
    return r


def _monthly_returns(ts: pd.DataFrame) -> pd.DataFrame:
    """Year x month table of simple returns (fractions) from the bar equity curve."""
    eq = pd.Series(ts["equity"], dtype=float).dropna()
    if eq.empty:
        return pd.DataFrame(columns=list(range(1, 13)), dtype=float)
    idx = pd.DatetimeIndex(eq.index)
    eq.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    month_end = eq.groupby([eq.index.year, eq.index.month]).last()
    prev = month_end.shift(1)
    prev.iloc[0] = float(eq.iloc[0])
    ret = month_end / prev - 1.0
    ret.index = pd.MultiIndex.from_tuples(list(ret.index), names=["year", "month"])
    table = ret.unstack("month").reindex(columns=list(range(1, 13)))
    return table


def _rolling_sharpe(r: pd.Series, window: int) -> pd.Series:
    mu = r.rolling(window, min_periods=window).mean()
    sd = r.rolling(window, min_periods=window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = mu / sd * math.sqrt(252.0)
    return out.where(sd > 0)


def _xs(index: Any) -> list[str]:
    """Compact x values: ``YYYY-MM-DD`` for daily points, ``YYYY-MM-DD HH:MM`` otherwise (UTC).

    Plotly would serialise tz-aware timestamps as 25-character ISO strings; on a 9-year
    daily series the short form roughly halves the page's data."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    if len(idx) and (idx == idx.normalize()).all():
        return list(idx.strftime("%Y-%m-%d"))
    return list(idx.strftime("%Y-%m-%d %H:%M"))


def _ys(values: Any, decimals: int) -> list[float | None]:
    """Rounded plain floats (NaN -> None): shorter than plotly's base64 float64 arrays."""
    arr = np.round(np.asarray(values, dtype=float), int(decimals))
    return [None if not math.isfinite(v) else (int(v) if decimals <= 0 else float(v)) for v in arr.tolist()]


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _fmt(v: Any, kind: str = "num", digits: int = 2) -> str:
    x = _num(v)
    if not math.isfinite(x):
        return "n/a"
    if kind == "pct":
        return f"{x * 100:,.{digits - 1 if digits > 1 else 1}f}%"
    if kind == "usd":
        sign = "−" if x < 0 else ""
        return f"{sign}${abs(x):,.0f}" if abs(x) < 1e6 else f"{sign}${abs(x) / 1e6:,.2f}M"
    if kind == "int":
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _short(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# =======================================================================================
# figure helpers (each trace carries meta.role so the theme toggle can recolour it)
# =======================================================================================
class _Figs:
    """Builds Plotly figures in the light palette with role tags for the JS theme switch."""

    def __init__(self) -> None:
        self.go = _require_plotly()
        self.pal = get_palette("light")
        self.template = plotly_template("light")

    def new(self, *, height: int, title: str | None = None, legend: bool = True, **layout: Any) -> Any:
        """Empty figure. The title is NOT drawn by plotly: it is kept in ``layout.meta`` and
        rendered as the card's HTML heading, and the legend is an HTML row of toggle buttons
        under it (:class:`_PlotEmitter`): both reflow on phones and can never collide with the
        plot, each other or the mode bar."""
        fig = self.go.Figure()
        fig.update_layout(template=self.template, height=height + 10, autosize=True,
                          margin={"l": 56, "r": 20, "t": 10, "b": 36}, barcornerradius=4,
                          meta={"title": title or "", "legend": bool(legend)})
        fig.update_layout(**layout)
        return fig

    def cat(self, slot: int) -> str:
        return self.pal.categorical[slot]

    @staticmethod
    def spikes(fig: Any) -> None:
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                         spikedash="solid", spikecolor=get_palette("light").text_muted, hoverformat="%Y-%m-%d")


def _line(go: Any, x: Any, y: Any, *, name: str, color: str, meta: dict[str, Any], fmt: str,
          fill: str | None = None, fill_alpha: float = 0.10, dash: str | None = None,
          width: float = 2.0, showlegend: bool = True) -> Any:
    line = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    kw: dict[str, Any] = {}
    if fill:
        kw = {"fill": fill, "fillcolor": hex_to_rgba(color, fill_alpha)}
        meta = {**meta, "fill_alpha": fill_alpha}
    return go.Scatter(x=x, y=y, name=_esc(name), mode="lines", line=line, meta=meta,
                      showlegend=showlegend,
                      hovertemplate=f"<b>%{{y:{fmt}}}</b>  {_esc(name)}<extra></extra>", **kw)


def _fold_shapes(folds: pd.DataFrame | None, color: str) -> list[dict[str, Any]]:
    if folds is None or "test_start" not in folds.columns:
        return []
    shapes = []
    for ts in pd.to_datetime(folds["test_start"], utc=True, errors="coerce").dropna().iloc[1:]:
        x = f"{ts:%Y-%m-%d}"
        shapes.append({"type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x, "y0": 0, "y1": 1,
                       "line": {"color": color, "width": 1}, "layer": "below", "name": "fold"})
    return shapes


def _zero_shape(color: str, *, yref: str = "y", xref: str = "paper") -> dict[str, Any]:
    return {"type": "line", "xref": xref, "yref": yref, "x0": 0, "x1": 1, "y0": 0, "y1": 0,
            "line": {"color": color, "width": 1}, "layer": "below", "name": "zero"}


def fig_equity(F: _Figs, books: Mapping[str, pd.DataFrame], folds: pd.DataFrame | None, *,
               title: str = "Equity vs buy-and-hold (log scale)") -> Any | None:
    if _COMBINED not in books:
        return None
    fig = F.new(height=340, title=title)
    eq = _daily_equity(books[_COMBINED])
    if _BENCHMARK in books:
        b = _daily_equity(books[_BENCHMARK])
        fig.add_trace(_line(F.go, _xs(b.index), _ys(b, 0), name="Buy & hold (benchmark)", color=F.pal.text_muted,
                            meta={"role": "muted"}, fmt="$,.0f", width=1.5))
    fig.add_trace(_line(F.go, _xs(eq.index), _ys(eq, 0), name="Combined book", color=F.cat(0),
                        meta={"role": "cat", "slot": 0}, fmt="$,.0f"))
    fig.update_yaxes(type="log", tickprefix="$", tickformat="~s", title_text="Equity (USD, log)")
    fig.update_layout(shapes=_fold_shapes(folds, F.pal.grid), hovermode="x unified")
    F.spikes(fig)
    return fig


def fig_drawdown(F: _Figs, books: Mapping[str, pd.DataFrame]) -> Any | None:
    from aurum.backtest.metrics import drawdown_series

    if _COMBINED not in books:
        return None
    fig = F.new(height=240, title="Drawdown from peak")
    if _BENCHMARK in books:
        b = drawdown_series(_daily_equity(books[_BENCHMARK]))
        fig.add_trace(_line(F.go, _xs(b.index), _ys(b, 5), name="Buy & hold (benchmark)", color=F.pal.text_muted,
                            meta={"role": "muted"}, fmt=".1%", width=1.5))
    dd = drawdown_series(_daily_equity(books[_COMBINED]))
    fig.add_trace(_line(F.go, _xs(dd.index), _ys(dd, 5), name="Combined book", color=F.cat(0),
                        meta={"role": "cat", "slot": 0}, fmt=".1%", fill="tozeroy"))
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    F.spikes(fig)
    return fig


def fig_rolling_sharpe(F: _Figs, books: Mapping[str, pd.DataFrame], window: int = 126) -> Any | None:
    if _COMBINED not in books:
        return None
    fig = F.new(height=240, title=f"Rolling Sharpe ({window} trading days, annualised)")
    if _BENCHMARK in books:
        rb = _rolling_sharpe(_daily_returns(books[_BENCHMARK]), window)
        fig.add_trace(_line(F.go, _xs(rb.index), _ys(rb, 3), name="Buy & hold (benchmark)", color=F.pal.text_muted,
                            meta={"role": "muted"}, fmt=".2f", width=1.5))
    rc = _rolling_sharpe(_daily_returns(books[_COMBINED]), window)
    if not rc.notna().any():
        return None
    fig.add_trace(_line(F.go, _xs(rc.index), _ys(rc, 3), name="Combined book", color=F.cat(0),
                        meta={"role": "cat", "slot": 0}, fmt=".2f"))
    fig.update_layout(shapes=[_zero_shape(F.pal.axis)], hovermode="x unified")
    fig.update_yaxes(title_text="Sharpe", zeroline=False)
    F.spikes(fig)
    return fig


def fig_fold_sharpe(F: _Figs, folds: pd.DataFrame | None) -> Any | None:
    if folds is None or "oos_sharpe" not in folds.columns or folds.empty:
        return None
    s = pd.to_numeric(folds["oos_sharpe"], errors="coerce")
    labels = [f"F{f}" for f in folds["fold"]] if "fold" in folds.columns else [f"F{i}" for i in range(len(s))]
    window = [""] * len(s)
    if "test_start" in folds.columns and "test_end" in folds.columns:
        start = pd.to_datetime(folds["test_start"], utc=True, errors="coerce")
        end = pd.to_datetime(folds["test_end"], utc=True, errors="coerce")
        window = [f"{a:%Y-%m-%d} → {b:%Y-%m-%d}" if pd.notna(a) and pd.notna(b) else ""
                  for a, b in zip(start, end, strict=True)]
    basis = folds["combiner_basis"].astype(str).map(_esc).tolist() if "combiner_basis" in folds.columns else [""] * len(s)
    colors = [F.cat(0) if (v >= 0) else F.cat(7) for v in s.fillna(0.0)]
    fig = F.new(height=260, title="Out-of-sample Sharpe per fold (combined book)", legend=False)
    fig.add_trace(F.go.Bar(
        x=labels, y=_ys(s, 3), marker={"color": colors, "line": {"width": 0}}, name="OOS Sharpe",
        meta={"role": "polarity"}, customdata=np.column_stack([window, basis]) if len(s) else None,
        hovertemplate="<b>%{y:.2f}</b>  OOS Sharpe<br>%{x}: %{customdata[0]}<br>combiner: %{customdata[1]}<extra></extra>",
        showlegend=False))
    fig.update_layout(shapes=[_zero_shape(F.pal.axis)], hovermode="closest", bargap=0.5)
    fig.update_yaxes(title_text="Sharpe")
    return fig


def fig_forecasts(F: _Figs, forecasts: pd.DataFrame | None, strategies: Sequence[str], *,
                  max_points: int = 1000) -> Any | None:
    """Small multiples: daily (weekly beyond ``max_points`` days) mean OOS forecast per
    strategy (+ combined), shared axes."""
    if forecasts is None or forecasts.empty:
        return None
    from plotly.subplots import make_subplots

    keys = [k for k in strategies if k in forecasts.columns]
    if _COMBINED in forecasts.columns:
        keys.append(_COMBINED)
    if not keys:
        return None
    f = forecasts[keys].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(f.index)
    f.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    daily = f.groupby(f.index.normalize()).mean()
    if len(daily) > max_points:   # weekly means: the panels are narrower than 1,000 px anyway
        daily = daily.resample("W-FRI").mean().dropna(how="all")
    n = len(keys)
    xs = _xs(daily.index)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.5 / max(n, 1)),
                        subplot_titles=[_esc(k) for k in keys])
    for i, k in enumerate(keys, start=1):
        slot = 1 if k == _COMBINED else 0
        fig.add_trace(_line(F.go, xs, _ys(daily[k], 3), name=k, color=F.cat(slot),
                            meta={"role": "cat", "slot": slot}, fmt="+.2f", fill="tozeroy",
                            width=1.5, showlegend=False), row=i, col=1)
        fig.update_yaxes(range=[-1.05, 1.05], tickvals=[-1, 0, 1], row=i, col=1)
    fig.update_layout(template=F.template, height=80 * n + 60, autosize=True, hovermode="x unified",
                      margin={"l": 48, "r": 16, "t": 28, "b": 32}, showlegend=False)
    for ann in fig.layout.annotations:
        ann.update(x=0.0, xanchor="left", font={"size": 11, "color": F.pal.text})
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinecolor=F.pal.axis)
    F.spikes(fig)
    return fig


def _strategy_slots(strategies: Sequence[str], weights: pd.DataFrame | None) -> tuple[list[str], dict[str, int]]:
    """Fixed colour slots for at most 8 series: <= 8 strategies keep config order; more ->
    the 7 with the largest mean weight (still in config order) + "Other"."""
    keys = list(strategies)
    if len(keys) <= 8:
        return keys, {k: i for i, k in enumerate(keys)}
    mean_w = {k: float(pd.to_numeric(weights[k], errors="coerce").abs().mean()) if weights is not None
              and k in weights.columns else 0.0 for k in keys}
    top = set(sorted(keys, key=lambda k: -mean_w[k])[:7])
    kept = [k for k in keys if k in top]
    return kept, {k: i for i, k in enumerate(kept)}


def fig_weights(F: _Figs, weights: pd.DataFrame | None, folds: pd.DataFrame | None,
                strategies: Sequence[str]) -> Any | None:
    if weights is None or weights.empty:
        return None
    keys = [k for k in strategies if k in weights.columns]
    if not keys:
        return None
    kept, slots = _strategy_slots(keys, weights)
    w = weights[keys].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = [f"F{f}" for f in weights["fold"]] if "fold" in weights.columns else [f"F{i}" for i in range(len(w))]
    fdm = pd.to_numeric(weights["fdm"], errors="coerce") if "fdm" in weights.columns else pd.Series(np.nan, index=w.index)
    window = [""] * len(w)
    if folds is not None and "test_start" in folds.columns and len(folds) == len(w):
        s = pd.to_datetime(folds["test_start"], utc=True, errors="coerce")
        window = [f"{x:%Y-%m}" if pd.notna(x) else "" for x in s]
    custom = np.column_stack([window, fdm.round(3).astype(str)])
    fig = F.new(height=330, title="Combiner weights per fold (share of the risk budget)")
    hover = "<b>%{y:.1%}</b>  {name}<br>%{x} (test from %{customdata[0]}), FDM %{customdata[1]}<extra></extra>"
    for k in kept:
        fig.add_trace(F.go.Bar(x=labels, y=_ys(w[k], 4), name=_esc(k), marker={"color": F.cat(slots[k])},
                               meta={"role": "cat", "slot": slots[k]}, customdata=custom,
                               hovertemplate=hover.replace("{name}", _esc(k))))
    other = [k for k in keys if k not in slots]
    if other:
        fig.add_trace(F.go.Bar(x=labels, y=_ys(w[other].sum(axis=1), 4), name=f"Other ({len(other)})",
                               marker={"color": F.pal.text_muted}, meta={"role": "muted"}, customdata=custom,
                               hovertemplate=hover.replace("{name}", f"Other ({len(other)})")))
    unalloc = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    if (unalloc > 1e-9).any():
        fig.add_trace(F.go.Bar(x=labels, y=_ys(unalloc, 4), name="Unallocated", marker={"color": F.pal.axis},
                               meta={"role": "unalloc"}, customdata=custom,
                               hovertemplate=hover.replace("{name}", "unallocated (held in cash)")))
    fig.update_layout(barmode="stack", bargap=0.45, hovermode="closest", barcornerradius=0)
    # a 1px surface-coloured outline on each segment = the 2px surface gap between stacked fills
    fig.update_traces(marker_line_width=1, marker_line_color=F.pal.surface)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.0], title_text="Weight")
    return fig


def fig_cost_waterfall(F: _Figs, costs: pd.DataFrame | None, book: str = _COMBINED) -> Any | None:
    if costs is None or book not in costs.index:
        return None
    row = costs.loc[book]
    gross, net = _num(row.get("gross_pnl")), _num(row.get("net_pnl"))
    if not (math.isfinite(gross) and math.isfinite(net)):
        return None
    comps = [("Spread", -_num(row.get("spread"))), ("Slippage", -_num(row.get("slippage"))),
             ("Commission", -_num(row.get("commission"))), ("Financing", _num(row.get("swap")))]
    comps = [(k, v if math.isfinite(v) else 0.0) for k, v in comps]
    x = ["Gross P&L"] + [k for k, _ in comps] + ["Net P&L"]
    y = [gross] + [v for _, v in comps] + [net]
    measure = ["absolute"] + ["relative"] * len(comps) + ["total"]
    fig = F.new(height=310, title=f"Gross to net: where the {book} book's P&L went", legend=False)
    fig.add_trace(F.go.Waterfall(
        x=x, y=y, measure=measure, meta={"role": "waterfall"}, name=_esc(book),
        increasing={"marker": {"color": F.cat(0)}}, decreasing={"marker": {"color": F.cat(7)}},
        totals={"marker": {"color": F.pal.text_secondary}}, connector={"line": {"color": F.pal.axis, "width": 1}},
        hovertemplate="<b>%{y:$,.0f}</b>  %{x}<extra></extra>", showlegend=False))
    fig.update_layout(hovermode="closest", shapes=[_zero_shape(F.pal.axis)], waterfallgap=0.45)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", title_text="USD")
    return fig


def fig_cost_books(F: _Figs, costs: pd.DataFrame | None) -> Any | None:
    if costs is None or costs.empty:
        return None
    books = [b for b in costs.index if b != _BENCHMARK]
    if not books:
        return None
    c = costs.loc[books]
    comps = [("Spread", "spread", 1.0), ("Slippage", "slippage", 1.0), ("Commission", "commission", 1.0),
             ("Financing", "swap", -1.0)]
    fig = F.new(height=40 * len(books) + 70, title="Trading costs by book (financing: + paid / − received)")
    for slot, (label, col, sign) in enumerate(comps):
        if col not in c.columns:
            continue
        vals = sign * pd.to_numeric(c[col], errors="coerce").fillna(0.0)
        fig.add_trace(F.go.Bar(y=[_esc(b) for b in books], x=_ys(vals, 2), orientation="h", name=label,
                               marker={"color": F.cat(slot), "line": {"width": 0}},
                               meta={"role": "cat", "slot": slot},
                               hovertemplate=f"<b>%{{x:$,.0f}}</b>  {label}<br>%{{y}}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.45, hovermode="closest", barcornerradius=0)
    fig.update_traces(marker_line_width=1, marker_line_color=F.pal.surface)
    fig.update_xaxes(tickprefix="$", tickformat=",.0f", showgrid=True, title_text="USD over the OOS period")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_monthly(F: _Figs, table: pd.DataFrame) -> Any | None:
    if table is None or table.empty or not np.isfinite(table.to_numpy(dtype=float)).any():
        return None
    z = np.round(table.to_numpy(dtype=float) * 100.0, 3)
    years = [str(y) for y in table.index]
    yearly = ((1.0 + table.fillna(0.0)).prod(axis=1) - 1.0) * 100.0
    custom = np.round(np.repeat(yearly.to_numpy()[:, None], 12, axis=1), 3)
    lim = float(np.nanpercentile(np.abs(z), 98)) if np.isfinite(z).any() else 1.0
    lim = max(lim, 0.1)
    scale = [[i / (len(F.pal.diverging) - 1), c] for i, c in enumerate(F.pal.diverging)]
    fig = F.new(height=28 * len(years) + 60, title="Monthly returns, combined book", legend=False)
    fig.add_trace(F.go.Heatmap(
        z=[_ys(row, 3) for row in z], x=list(_MONTHS), y=years, zmid=0.0, zmin=-lim, zmax=lim, colorscale=scale, xgap=2, ygap=2,
        customdata=custom, meta={"role": "diverging"}, hoverongaps=False,
        colorbar={"ticksuffix": "%", "thickness": 10, "outlinewidth": 0,
                  "tickfont": {"color": F.pal.text_secondary}},
        hovertemplate="<b>%{z:+.2f}%</b>  %{x} %{y}<br>year: %{customdata:+.1f}%<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category", showline=False)
    fig.update_xaxes(showgrid=False, showline=False)
    return fig


def fig_desk(F: _Figs, cycles: pd.DataFrame) -> tuple[Any, Any] | None:
    if cycles is None or cycles.empty:
        return None
    t = _xs(cycles["now"])
    text = []
    for _, r in cycles.iterrows():
        cons = ", ".join(_esc(a) for a in r["consulted"]) or "none"
        made = ", ".join(_esc(a) for a in r["created"]) or "none"
        parts = [f"status: {_esc(r['status'])}", f"action: {_esc(r['action'])}"
                 + (f" (confidence {_fmt(r['confidence'])})" if pd.notna(r["confidence"]) else ""),
                 f"consulted: {cons}", f"created: {made}"]
        if pd.notna(r["cost_usd"]):
            parts.append(f"cost: ${_num(r['cost_usd']):.2f}")
        if r["rationale"]:
            parts.append(f"<i>{_esc(_short(r['rationale'], 160))}</i>")
        if r["failure_reason"]:
            parts.append(f"failure: {_esc(_short(r['failure_reason'], 120))}")
        text.append("<br>".join(parts))
    fig = F.new(height=300, title="Desk decisions: quant forecast vs final (after the LLM overlay)")
    fig.add_trace(F.go.Scatter(x=t, y=_ys(cycles["quant_forecast"], 4), name="Quant forecast", mode="lines+markers",
                               line={"color": F.pal.text_muted, "width": 2, "shape": "hv"},
                               marker={"size": 8, "color": F.pal.text_muted, "line": {"color": F.pal.surface, "width": 2}},
                               meta={"role": "muted"}, hovertemplate="<b>%{y:+.2f}</b>  quant<extra></extra>"))
    fig.add_trace(F.go.Scatter(x=t, y=_ys(cycles["final_forecast"], 4), name="Final forecast", mode="lines+markers",
                               line={"color": F.cat(0), "width": 2, "shape": "hv"},
                               marker={"size": 9, "color": F.cat(0), "line": {"color": F.pal.surface, "width": 2}},
                               meta={"role": "cat", "slot": 0}, text=text,
                               hovertemplate="<b>%{y:+.2f}</b>  final<br>%{text}<extra></extra>"))
    fig.update_layout(hovermode="x unified", shapes=[_zero_shape(F.pal.axis)])
    fig.update_yaxes(range=[-1.05, 1.05], title_text="Forecast")
    F.spikes(fig)
    n_cons = cycles["consulted"].map(len)
    n_made = cycles["created"].map(len)
    fig2 = F.new(height=200, title="Specialists per cycle")
    fig2.add_trace(F.go.Bar(x=t, y=n_cons.tolist(), name="Consulted (predefined)", marker={"color": F.cat(0)},
                            meta={"role": "cat", "slot": 0},
                            hovertemplate="<b>%{y}</b>  consulted<extra></extra>"))
    fig2.add_trace(F.go.Bar(x=t, y=n_made.tolist(), name="Created (ad hoc)", marker={"color": F.cat(1)},
                            meta={"role": "cat", "slot": 1},
                            hovertemplate="<b>%{y}</b>  created<extra></extra>"))
    fig2.update_layout(barmode="stack", hovermode="x unified", bargap=0.4, barcornerradius=0)
    fig2.update_traces(marker_line_width=1, marker_line_color=F.pal.surface)
    fig2.update_yaxes(title_text="Agents", dtick=1, rangemode="tozero")
    return fig, fig2


# =======================================================================================
# HTML pieces
# =======================================================================================
def _tile(label: str, value: str, *, sub: str = "", delta: str = "", delta_good: bool | None = None,
          hint: str = "") -> str:
    cls = "" if delta_good is None else (" good" if delta_good else " bad")
    icon = "" if delta_good is None else ("▲ " if delta_good else "▼ ")
    d = f'<div class="delta{cls}">{icon}{_esc(delta)}</div>' if delta else ""
    s = f'<div class="tile-sub">{_esc(sub)}</div>' if sub else ""
    title = f' title="{_esc(hint)}"' if hint else ""
    return (f'<div class="tile"{title}><div class="tile-label">{_esc(label)}</div>'
            f'<div class="tile-value">{_esc(value)}</div>{s}{d}</div>')


def _kpi_tiles(stats: Mapping[str, Any], bench: Mapping[str, Any] | None, *, n_trials: Any = None,
               pbo: Any = None) -> str:
    b = bench or {}
    tiles = []
    sh, bsh = _num(stats.get("sharpe")), _num(b.get("sharpe"))
    ci = f"95% CI [{_fmt(stats.get('ci_lower'))}, {_fmt(stats.get('ci_upper'))}]"
    tiles.append(_tile("Sharpe (daily, annualised)", _fmt(sh), sub=ci,
                       delta=f"buy & hold {_fmt(bsh)}" if math.isfinite(bsh) else "",
                       delta_good=(sh >= bsh) if (math.isfinite(sh) and math.isfinite(bsh)) else None,
                       hint="Headline Sharpe on daily returns; CI from a stationary bootstrap"))
    cg, bcg = _num(stats.get("cagr")), _num(b.get("cagr"))
    tiles.append(_tile("CAGR", _fmt(cg, "pct"),
                       delta=f"buy & hold {_fmt(bcg, 'pct')}" if math.isfinite(bcg) else "",
                       delta_good=(cg >= bcg) if (math.isfinite(cg) and math.isfinite(bcg)) else None))
    dd, bdd = _num(stats.get("max_drawdown")), _num(b.get("max_drawdown"))
    tiles.append(_tile("Max drawdown", _fmt(dd, "pct"),
                       delta=f"buy & hold {_fmt(bdd, 'pct')}" if math.isfinite(bdd) else "",
                       delta_good=(dd >= bdd) if (math.isfinite(dd) and math.isfinite(bdd)) else None))
    nt = f"n_trials = {_fmt(n_trials, 'int')}" if n_trials is not None else ""
    tiles.append(_tile("Deflated Sharpe (DSR)", _fmt(stats.get("dsr"), digits=3), sub=nt,
                       hint="P(true Sharpe > best of n_trials skill-less trials); protocol needs > 0.95"))
    tiles.append(_tile("Probabilistic Sharpe (PSR)", _fmt(stats.get("psr"), digits=3),
                       hint="P(true Sharpe > 0) given the sample length, skew and kurtosis"))
    if pbo is not None and math.isfinite(_num(pbo)):
        tiles.append(_tile("PBO", _fmt(pbo, digits=2), hint="Probability of backtest overfitting (CSCV)"))
    tiles.append(_tile("Volatility", _fmt(stats.get("ann_vol"), "pct")))
    tiles.append(_tile("Total costs", _fmt(stats.get("total_costs"), "usd"),
                       sub=f"{_fmt(stats.get('n_trades'), 'int')} trades"))
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _badges(stats: Mapping[str, Any]) -> str:
    dsr, lo = _num(stats.get("dsr")), _num(stats.get("ci_lower"))
    out = []
    for ok, text in ((dsr > 0.95, f"DSR {_fmt(dsr, digits=3)} {'>' if dsr > 0.95 else '≤'} 0.95"),
                     (lo > 0, f"Sharpe CI lower bound {_fmt(lo)} {'>' if lo > 0 else '≤'} 0")):
        if not math.isfinite(dsr if "DSR" in text else lo):
            continue
        cls, icon = ("good", "✓") if ok else ("critical", "✕")
        out.append(f'<span class="badge {cls}"><span aria-hidden="true">{icon}</span> {_esc(text)}</span>')
    if not out:
        return ""
    return ('<div class="badges" aria-label="protocol criteria">' + "".join(out)
            + '<span class="badge-note">criteria for live use (docs/RESEARCH_PROTOCOL.md)</span></div>')


_STAT_COLS = [("sharpe", "Sharpe", "num"), ("ci", "95% CI", "ci"), ("psr", "PSR", "num3"), ("dsr", "DSR", "num3"),
              ("cagr", "CAGR", "pct"), ("ann_vol", "Vol", "pct"), ("max_drawdown", "Max DD", "pct"),
              ("n_trades", "Trades", "int"), ("total_costs", "Costs", "usd")]


def _stats_table(stats: Mapping[str, Mapping[str, Any]]) -> str:
    if not stats:
        return ""
    head = "".join(f"<th>{_esc(lbl)}</th>" for _, lbl, _ in _STAT_COLS)
    rows = []
    for book, s in stats.items():
        cells = []
        for key, _, kind in _STAT_COLS:
            if kind == "ci":
                v = f"[{_fmt(s.get('ci_lower'))}, {_fmt(s.get('ci_upper'))}]"
            elif kind == "num3":
                v = _fmt(s.get(key), digits=3)
            else:
                v = _fmt(s.get(key), kind)
            cells.append(f"<td>{_esc(v)}</td>")
        cls = ' class="em"' if book == _COMBINED else (' class="muted"' if book == _BENCHMARK else "")
        rows.append(f"<tr{cls}><th scope=\"row\">{_esc(book)}</th>{''.join(cells)}</tr>")
    return (f'<div class="table-wrap"><table class="num"><thead><tr><th>Book</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _monthly_table(table: pd.DataFrame) -> str:
    if table is None or table.empty:
        return ""
    yearly = (1.0 + table.fillna(0.0)).prod(axis=1) - 1.0
    head = "".join(f"<th>{m}</th>" for m in _MONTHS) + "<th>Year</th>"
    rows = []
    for y, r in table.iterrows():
        cells = "".join(f"<td>{_fmt(v, 'pct')}</td>" if pd.notna(v) else "<td></td>" for v in r.to_numpy())
        rows.append(f'<tr><th scope="row">{_esc(y)}</th>{cells}<td class="em">{_fmt(yearly.loc[y], "pct")}</td></tr>')
    return (f'<details><summary>Table view</summary><div class="table-wrap"><table class="num">'
            f'<thead><tr><th>Year</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>')


_TRADE_COLS = [("entry_time", "Entry"), ("exit_time", "Exit"), ("side", "Side"), ("lots", "Lots"),
               ("entry_price", "Entry px"), ("exit_price", "Exit px"), ("pnl", "P&L"), ("costs", "Costs"),
               ("swap", "Financing"), ("bars_held", "Bars"), ("exit_reason", "Exit reason")]


def _trades_table(trades: pd.DataFrame | None, max_rows: int) -> str:
    if trades is None or trades.empty:
        return '<p class="note">No trades recorded for the combined book.</p>'
    t = trades.copy()
    n_all = len(t)
    if "exit_time" in t.columns:
        t["_sort"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
        t = t.sort_values("_sort", ascending=False)
    t = t.head(int(max_rows))
    pnl = pd.to_numeric(t.get("pnl"), errors="coerce") if "pnl" in t.columns else pd.Series(dtype=float)
    win = float((pd.to_numeric(trades["pnl"], errors="coerce") > 0).mean()) if "pnl" in trades.columns else float("nan")
    head = "".join(f"<th>{lbl}</th>" for c, lbl in _TRADE_COLS if c in t.columns)
    rows = []
    for _, r in t.iterrows():
        cells = []
        for c, _ in _TRADE_COLS:
            if c not in t.columns:
                continue
            v = r[c]
            if c in ("entry_time", "exit_time"):
                ts = pd.to_datetime(v, utc=True, errors="coerce")
                s = f"{ts:%Y-%m-%d %H:%M}" if pd.notna(ts) else ""
            elif c == "side":
                s = "Long" if _num(v) > 0 else ("Short" if _num(v) < 0 else _esc(v))
            elif c in ("pnl", "costs", "swap"):
                x = _num(v)
                s = f"{x:+,.2f}" if (c != "costs" and math.isfinite(x)) else _fmt(v)
            elif c in ("entry_price", "exit_price"):
                s = _fmt(v, digits=2)
            elif c == "bars_held":
                s = _fmt(v, "int")
            else:
                s = _esc(v)
            cells.append(f"<td>{s}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    shown = f"the {len(t):,} most recent of {n_all:,} trades" if n_all > len(t) else f"all {n_all:,} trades"
    summary = (f'<p class="note">Showing {shown}. Win rate {_fmt(win, "pct")}; '
               f'net P&L of the rows shown {_fmt(pnl.sum(), "usd")}.</p>')
    return (summary + '<input class="filter" type="search" placeholder="Filter trades (e.g. 2020-03, Short, stop)" '
            'aria-label="Filter trades" data-target="trades-table">'
            f'<div class="table-wrap tall"><table class="num" id="trades-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _desk_table(cycles: pd.DataFrame) -> str:
    head = "<th>Time (UTC)</th><th>Status</th><th>Quant</th><th>Final</th><th>Action</th><th>Consulted</th><th>Created</th><th>Cost</th><th>Rationale</th>"
    rows = []
    for _, r in cycles.iterrows():
        rows.append(
            "<tr>"
            f"<td>{r['now']:%Y-%m-%d %H:%M}</td><td>{_esc(r['status'])}</td>"
            f"<td>{_fmt(r['quant_forecast'])}</td><td>{_fmt(r['final_forecast'])}</td>"
            f"<td>{_esc(r['action'])}</td><td>{_esc(', '.join(r['consulted']))}</td>"
            f"<td>{_esc(', '.join(r['created']))}</td>"
            f"<td>{'$' + format(_num(r['cost_usd']), '.2f') if pd.notna(r['cost_usd']) else 'n/a'}</td>"
            f'<td class="wrap">{_esc(_short(r["rationale"], 200))}</td>'
            "</tr>")
    return (f'<div class="table-wrap tall"><table class="num"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _card(title: str | None, body: str, *, wide: bool = False, caption: str = "") -> str:
    h = f"<h3>{_esc(title)}</h3>" if title else ""
    c = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="card{" wide" if wide else ""}">{h}{body}{c}</div>'


def _note(text: str) -> str:
    return f'<p class="note">{_esc(text)}</p>'


# =======================================================================================
# page
# =======================================================================================
_CSS = """
:root{color-scheme:light;--page:%(page_l)s;--surface:%(surface_l)s;--text:%(text_l)s;--text2:%(text2_l)s;
--muted:%(muted_l)s;--grid:%(grid_l)s;--axis:%(axis_l)s;--border:%(border_l)s;--good-text:%(good_l)s;
--critical:%(critical)s;--good:%(good)s;--accent:%(accent_l)s;%(cat_l)s}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:%(page_d)s;
--surface:%(surface_d)s;--text:%(text_d)s;--text2:%(text2_d)s;--muted:%(muted_d)s;--grid:%(grid_d)s;
--axis:%(axis_d)s;--border:%(border_d)s;--good-text:%(good_d)s;--accent:%(accent_d)s;%(cat_d)s}}
:root[data-theme="dark"]{color-scheme:dark;--page:%(page_d)s;--surface:%(surface_d)s;--text:%(text_d)s;
--text2:%(text2_d)s;--muted:%(muted_d)s;--grid:%(grid_d)s;--axis:%(axis_d)s;--border:%(border_d)s;
--good-text:%(good_d)s;--accent:%(accent_d)s;%(cat_d)s}
*{box-sizing:border-box}
html,body{margin:0;background:var(--page);color:var(--text);font-family:%(font)s;font-size:15px;line-height:1.45}
.wrap-all{max-width:1320px;margin:0 auto;padding:20px 16px 48px}
header.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;margin-bottom:8px}
h1{font-size:24px;line-height:1.2;margin:0 0 4px;font-weight:650}
h2{font-size:18px;margin:32px 0 12px;font-weight:650}
h3{font-size:14px;margin:0 0 8px;font-weight:600;color:var(--text)}
.sub{color:var(--text2);margin:0;font-size:14px}
.sub code{font-size:12px}
nav.toc{display:flex;flex-wrap:wrap;gap:4px 14px;margin:12px 0 4px;font-size:13px}
nav.toc a{color:var(--text2);text-decoration:none;border-bottom:1px solid var(--grid)}
nav.toc a:hover{color:var(--text)}
button.theme{font:inherit;font-size:13px;color:var(--text);background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:6px 12px;cursor:pointer}
button.theme:focus-visible,input.filter:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,140px),1fr));gap:10px;margin-top:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;min-width:0}
.tile-label{font-size:12px;color:var(--text2)}
.tile-value{font-size:24px;font-weight:600;margin-top:2px;overflow-wrap:anywhere}
.tile-sub{font-size:12px;color:var(--text2);margin-top:2px}
.delta{font-size:12px;color:var(--text2);margin-top:4px}
.delta.good{color:var(--good-text)}
.delta.bad{color:var(--critical)}
.badges{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
.badge{font-size:12px;border-radius:999px;padding:3px 10px;border:1px solid var(--border);background:var(--surface);color:var(--text)}
.badge.good span{color:var(--good)}
.badge.critical span{color:var(--critical)}
.badge-note{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,540px),1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 14px 10px;min-width:0}
.card.wide{grid-column:1/-1}
.caption{font-size:12px;color:var(--text2);margin:6px 2px 2px}
.note{font-size:13px;color:var(--text2);margin:6px 0}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%%}
.table-wrap.tall{max-height:520px;overflow-y:auto}
table{border-collapse:collapse;width:100%%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
thead th{position:sticky;top:0;background:var(--surface);color:var(--text2);font-weight:600;z-index:1}
table.num td{font-variant-numeric:tabular-nums}
tbody th{font-weight:500}
tr.em th,tr.em td,td.em{font-weight:650}
tr.muted th,tr.muted td{color:var(--text2)}
td.wrap{white-space:normal;min-width:240px}
details{margin-top:8px}
summary{cursor:pointer;color:var(--text2);font-size:13px}
input.filter{font:inherit;font-size:13px;width:min(100%%,360px);padding:6px 10px;margin:4px 0 8px;color:var(--text);
background:var(--page);border:1px solid var(--border);border-radius:8px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--text2)}
dl.prov{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:13px;margin:0}
dl.prov dt{color:var(--text2)}
dl.prov dd{margin:0;overflow-wrap:anywhere}
ul.notes{font-size:13px;color:var(--text2);padding-left:18px;margin:6px 0}
.plot{width:100%%}
.legend{display:flex;flex-wrap:wrap;gap:2px 14px;margin:0 0 2px}
.key{font:inherit;font-size:12px;color:var(--text);background:none;border:0;padding:3px 0;cursor:pointer;
display:inline-flex;align-items:center;gap:6px;border-radius:4px}
.key:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.key[aria-pressed="false"]{color:var(--muted);text-decoration:line-through}
.key[aria-pressed="false"] .sw{opacity:.3}
.sw{display:inline-block;background:var(--c);flex:none}
.sw.line{width:16px;height:2px;border-radius:1px}
.sw.box{width:10px;height:10px;border-radius:2px}
@media (max-width:600px){h1{font-size:20px}.tile-value{font-size:20px}.card{padding:10px 8px 6px}}
"""

_JS = """
(function(){
  var P = __PALETTES__;
  var KEY = 'aurum-dashboard-theme';
  function rgba(hex, a){ var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')'; }
  function roleColor(m, p){
    if (m.role === 'cat') return p.categorical[m.slot];
    if (m.role === 'muted') return p.text_muted;
    if (m.role === 'unalloc') return p.axis;
    return null; }
  function themeFigure(gd, p){
    if (!gd || !gd.data || !gd.layout || typeof Plotly === 'undefined') return;
    var lay = {paper_bgcolor: p.surface, plot_bgcolor: p.surface, 'font.color': p.text_secondary,
      'title.font.color': p.text, 'legend.font.color': p.text, 'hoverlabel.bgcolor': p.surface,
      'hoverlabel.bordercolor': p.axis, 'hoverlabel.font.color': p.text};
    Object.keys(gd.layout).forEach(function(k){
      if (/^[xy]axis[0-9]*$/.test(k)) {
        lay[k + '.gridcolor'] = p.grid; lay[k + '.linecolor'] = p.axis; lay[k + '.zerolinecolor'] = p.axis;
        lay[k + '.tickfont.color'] = p.text_secondary; lay[k + '.title.font.color'] = p.text_secondary;
        lay[k + '.spikecolor'] = p.text_muted; } });
    (gd.layout.annotations || []).forEach(function(a, i){ lay['annotations[' + i + '].font.color'] = p.text; });
    (gd.layout.shapes || []).forEach(function(s, i){
      lay['shapes[' + i + '].line.color'] = (s.name === 'zero') ? p.axis : p.grid; });
    Plotly.relayout(gd, lay);
    gd.data.forEach(function(t, i){
      var m = t.meta || {}, u = {};
      if (m.role === 'polarity') {
        var vals = (t.orientation === 'h' ? t.x : t.y) || [];
        u['marker.color'] = [Array.prototype.map.call(vals, function(v){ return v >= 0 ? p.categorical[0] : p.categorical[7]; })];
      } else if (m.role === 'diverging') {
        u.colorscale = [p.diverging.map(function(c, j, arr){ return [j / (arr.length - 1), c]; })];
        u['colorbar.tickfont.color'] = p.text_secondary;
      } else if (m.role === 'waterfall') {
        u['increasing.marker.color'] = p.categorical[0]; u['decreasing.marker.color'] = p.categorical[7];
        u['totals.marker.color'] = p.text_secondary; u['connector.line.color'] = p.axis;
      } else {
        var c = roleColor(m, p); if (!c) return;
        if (t.type === 'bar') { u['marker.color'] = c; if (t.marker && t.marker.line && t.marker.line.width) u['marker.line.color'] = p.surface; }
        else { u['line.color'] = c; u['marker.color'] = c; u['marker.line.color'] = p.surface;
          if (t.fill && t.fill !== 'none') u.fillcolor = rgba(c, m.fill_alpha || 0.1); }
      }
      Plotly.restyle(gd, u, [i]);
    });
  }
  function apply(mode, persist){
    var p = P[mode] || P.light;
    document.documentElement.setAttribute('data-theme', mode);
    var b = document.getElementById('theme-toggle');
    if (b) { b.textContent = mode === 'dark' ? 'Light mode' : 'Dark mode';
      b.setAttribute('aria-pressed', mode === 'dark' ? 'true' : 'false'); }
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd){ themeFigure(gd, p); });
    if (persist) { try { localStorage.setItem(KEY, mode); } catch (e) {} }
  }
  function initial(){
    var s = null; try { s = localStorage.getItem(KEY); } catch (e) {}
    if (s === 'light' || s === 'dark') return s;
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
  }
  window.addEventListener('load', function(){
    apply(initial(), false);
    var b = document.getElementById('theme-toggle');
    if (b) b.addEventListener('click', function(){
      apply(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark', true); });
    document.querySelectorAll('.legend').forEach(function(lg){
      var gd = document.getElementById(lg.getAttribute('data-plot'));
      lg.querySelectorAll('button.key').forEach(function(btn){
        btn.addEventListener('click', function(){
          if (!gd || typeof Plotly === 'undefined') return;
          var on = btn.getAttribute('aria-pressed') !== 'true';
          btn.setAttribute('aria-pressed', on ? 'true' : 'false');
          Plotly.restyle(gd, {visible: on ? true : 'legendonly'}, [parseInt(btn.getAttribute('data-trace'), 10)]);
        }); }); });
    document.querySelectorAll('input.filter').forEach(function(inp){
      inp.addEventListener('input', function(){
        var q = inp.value.toLowerCase(), tb = document.getElementById(inp.getAttribute('data-target'));
        if (!tb) return;
        tb.querySelectorAll('tbody tr').forEach(function(tr){
          tr.style.display = (!q || tr.textContent.toLowerCase().indexOf(q) >= 0) ? '' : 'none'; }); }); });
  });
})();
"""


def _css() -> str:
    lp, dp = get_palette("light"), get_palette("dark")
    return _CSS % {
        "page_l": lp.page, "surface_l": lp.surface, "text_l": lp.text, "text2_l": lp.text_secondary,
        "muted_l": lp.text_muted, "grid_l": lp.grid, "axis_l": lp.axis, "border_l": lp.border,
        "good_l": lp.delta_good, "accent_l": lp.categorical[0],
        "page_d": dp.page, "surface_d": dp.surface, "text_d": dp.text, "text2_d": dp.text_secondary,
        "muted_d": dp.text_muted, "grid_d": dp.grid, "axis_d": dp.axis, "border_d": dp.border,
        "good_d": dp.delta_good, "accent_d": dp.categorical[0],
        "critical": lp.status["critical"], "good": lp.status["good"], "font": FONT_STACK_CSS,
        "cat_l": ";".join(f"--cat-{i}:{c}" for i, c in enumerate(lp.categorical)),
        "cat_d": ";".join(f"--cat-{i}:{c}" for i, c in enumerate(dp.categorical)),
    }


class _PlotEmitter:
    """Serialises figures; plotly.js goes into the first one (inline or CDN tag)."""

    def __init__(self, include_plotlyjs: bool | str) -> None:
        if include_plotlyjs not in (True, False, "cdn"):
            raise ValueError("include_plotlyjs must be True, False or 'cdn'")
        self.include = include_plotlyjs
        self.n = 0

    @staticmethod
    def legend_items(fig: Any) -> list[tuple[int, str, str, str]]:
        """``(trace index, name, css colour, key shape)`` for every legend-worthy trace."""
        items = []
        for i, t in enumerate(fig.data):
            if t.showlegend is False or not t.name:
                continue
            m = t.meta if isinstance(t.meta, dict) else {}
            role = m.get("role")
            if role == "cat":
                css = f"var(--cat-{int(m.get('slot', 0))})"
            elif role == "muted":
                css = "var(--muted)"
            elif role == "unalloc":
                css = "var(--axis)"
            else:
                continue
            items.append((i, str(t.name), css, "box" if t.type == "bar" else "line"))
        return items

    def __call__(self, fig: Any) -> str:
        import plotly.io as pio

        inc = self.include if self.n == 0 else False
        self.n += 1
        div_id = f"aurum-plot-{self.n}"
        meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        items = self.legend_items(fig) if meta.get("legend", True) else []
        fig.update_layout(showlegend=False)
        div = pio.to_html(fig, full_html=False, include_plotlyjs=inc, config=_PLOTLY_CONFIG,
                          div_id=div_id, default_width="100%",
                          default_height=f"{int(fig.layout.height or 360)}px")
        title = meta.get("title") or ""
        head = f"<h3>{_esc(title)}</h3>" if title else ""
        legend = ""
        if len(items) >= 2:   # a single series is named by the title: no legend box
            keys = "".join(
                f'<button type="button" class="key" data-trace="{i}" aria-pressed="true" '
                f'title="Show / hide this series"><span class="sw {shape}" style="--c:{css}"></span>{name}</button>'
                for i, name, css, shape in items)   # names are already HTML-escaped by the builders
            legend = f'<div class="legend" data-plot="{div_id}" role="group" aria-label="series">{keys}</div>'
        return f'{head}{legend}<div class="plot">{div}</div>'


def _section(sid: str, title: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{_esc(title)}</h2>{body}</section>'


def _safe(build: Callable[[], str], what: str, problems: list[str]) -> str:
    try:
        return build()
    except Exception as exc:  # noqa: BLE001 - one broken panel must not lose the whole page
        logger.exception("dashboard: %s failed", what)
        problems.append(f"{what}: {type(exc).__name__}: {exc}")
        return _note(f"{what} unavailable ({type(exc).__name__}: {exc}).")


def build_dashboard(run_dir: str | Path, out_html: str | Path | None = None, *,
                    journal_dir: str | Path | None = None, include_plotlyjs: bool | str = True,
                    title: str | None = None, max_trades: int = 500, rolling_days: int = 126) -> Path:
    """Write a self-contained interactive dashboard for a walk-forward run directory.

    Parameters
    ----------
    run_dir : directory written by ``aurum walkforward`` / ``WalkForwardReport.save``.
    out_html : output file (default ``<run_dir>/dashboard.html``); parents are created.
    journal_dir : optional LLM-desk journal directory (``*.jsonl`` from ``CycleJournal``):
        adds the desk timeline section.
    include_plotlyjs : ``True`` embeds plotly.js (offline, ~4.8 MB), ``"cdn"`` references the
        plotly CDN (small file, needs internet to view), ``False`` omits it (the page then
        needs plotly.js from elsewhere).
    title : page title (default ``"Aurum - <run name> - <kind>"``).
    max_trades : most recent combined-book trades listed in the trades table.
    rolling_days : window of the rolling Sharpe chart (trading days).

    Returns the written path. Missing pieces of a run (no books, no holdout, no journal)
    leave out the corresponding panels with a note; a panel that fails to build is replaced
    by a note and logged, never aborting the page.
    """
    go = _require_plotly()  # noqa: F841 - fail fast with a clear message when plotly is missing
    run = load_run(run_dir)
    out = Path(out_html) if out_html is not None else run.run_dir / "dashboard.html"
    s = run.summary
    F = _Figs()
    emit = _PlotEmitter(include_plotlyjs)
    problems: list[str] = []
    strategies = run.strategies
    stats = s.get("stats") or {}
    comb = s.get("combined") or stats.get(_COMBINED) or {}
    bench = stats.get(_BENCHMARK)
    pbo = (s.get("pbo") or {}).get("pbo") if isinstance(s.get("pbo"), dict) else None
    kind = str(s.get("kind") or "run")
    page_title = title or f"Aurum · {run.name} · {kind}"
    sections: list[str] = []
    toc: list[tuple[str, str]] = []

    # ---- KPIs ------------------------------------------------------------------------
    def kpis() -> str:
        return _kpi_tiles(comb, bench, n_trials=s.get("n_trials"), pbo=pbo) + _badges(comb)

    sections.append(_section("kpis", "Out-of-sample summary (combined book)", _safe(kpis, "KPI header", problems)))
    toc.append(("kpis", "Summary"))

    # ---- performance -----------------------------------------------------------------
    def performance() -> str:
        cards = []
        f = fig_equity(F, run.books, run.folds)
        if f is None:
            return _note("No book time series in this run directory (saved with save_books=False?).")
        cards.append(_card(None, emit(f), wide=True,
                           caption="One point per trading day. Vertical hairlines mark the start of each "
                                   "walk-forward test block (every block is out-of-sample)."))
        for fn in (lambda: fig_drawdown(F, run.books), lambda: fig_rolling_sharpe(F, run.books, rolling_days),
                   lambda: fig_fold_sharpe(F, run.folds)):
            g = fn()
            if g is not None:
                cards.append(_card(None, emit(g)))
        return '<div class="grid">' + "".join(cards) + "</div>"

    sections.append(_section("performance", "Performance", _safe(performance, "performance charts", problems)))
    toc.append(("performance", "Performance"))

    # ---- strategies ------------------------------------------------------------------
    def strategy_panels() -> str:
        cards = []
        f = fig_forecasts(F, run.forecasts, strategies)
        if f is not None:
            cards.append(_card("Out-of-sample forecasts (daily or weekly mean, −1 short … +1 long)", emit(f), wide=True,
                               caption="Stitched from the test blocks of every fold: each value was produced by a "
                                       "model fitted only on data before it. The last panel is the combined "
                                       "forecast that was traded."))
        w = fig_weights(F, run.weights, run.folds, strategies)
        if w is not None:
            cards.append(_card(None, emit(w), wide=True,
                               caption="Weights are fitted net of costs on EARLIER folds' out-of-sample forecasts. "
                                       "Strategies with a non-positive net Sharpe get zero weight; the gray band "
                                       "is risk left unallocated. Hover for the forecast diversification "
                                       "multiplier (FDM)."))
        tbl = _stats_table(stats)
        if tbl:
            cards.append(_card("Per-book statistics (each strategy traded alone, and the combined book)", tbl,
                               wide=True))
        return '<div class="grid">' + "".join(cards) + "</div>" if cards else _note("No strategy data.")

    sections.append(_section("strategies", "Strategies and combiner", _safe(strategy_panels, "strategy panels",
                                                                            problems)))
    toc.append(("strategies", "Strategies"))

    # ---- costs -----------------------------------------------------------------------
    def cost_panels() -> str:
        cards = []
        wf = fig_cost_waterfall(F, run.costs)
        if wf is not None:
            cards.append(_card(None, emit(wf), caption="Gross P&L at mid prices, then spread, slippage, commission "
                                                       "and overnight financing (negative = paid)."))
        cb = fig_cost_books(F, run.costs)
        if cb is not None:
            cards.append(_card(None, emit(cb), caption="Buy-and-hold is left out (its financing bill dwarfs the "
                                                       "scale); see the statistics table."))
        return '<div class="grid">' + "".join(cards) + "</div>" if cards else _note("No cost attribution "
                                                                                    "(costs.csv) in this run.")

    sections.append(_section("costs", "Cost attribution", _safe(cost_panels, "cost attribution", problems)))
    toc.append(("costs", "Costs"))

    # ---- monthly ---------------------------------------------------------------------
    def monthly() -> str:
        if _COMBINED not in run.books:
            return _note("No combined-book time series.")
        table = _monthly_returns(run.books[_COMBINED])
        f = fig_monthly(F, table)
        if f is None:
            return _note("Not enough data for monthly returns.")
        return _card(None, emit(f) + _monthly_table(table), wide=True,
                     caption="Blue = gain, red = loss, gray = flat; colour saturates at the 98th percentile "
                             "of |monthly return|. Hover for the calendar-year return.")

    sections.append(_section("monthly", "Monthly returns", _safe(monthly, "monthly returns", problems)))
    toc.append(("monthly", "Monthly"))

    # ---- trades ----------------------------------------------------------------------
    sections.append(_section("trades", "Trades (combined book)",
                             _safe(lambda: _card(None, _trades_table(run.trades, max_trades), wide=True),
                                   "trades table", problems)))
    toc.append(("trades", "Trades"))

    # ---- holdout ---------------------------------------------------------------------
    hold = s.get("holdout")
    if isinstance(hold, dict) and hold.get("stats"):
        def holdout() -> str:
            hs = hold.get("stats") or {}
            body = (f'<p class="note">Final holdout {_esc(str(hold.get("start", ""))[:10])} → '
                    f'{_esc(str(hold.get("end", ""))[:10])}: fitted on the last training window and evaluated '
                    f'<b>once</b>. Combiner basis: {_esc(hold.get("combiner_basis", ""))}.</p>')
            body += _kpi_tiles(hs.get(_COMBINED) or {}, hs.get(_BENCHMARK), n_trials=s.get("n_trials"))
            cards = []
            f = fig_equity(F, run.holdout_books, None, title="Holdout equity vs buy-and-hold (log scale)")
            if f is not None:
                cards.append(_card(None, emit(f), wide=True))
            tbl = _stats_table(hs)
            if tbl:
                cards.append(_card("Holdout statistics", tbl, wide=True))
            return body + ('<div class="grid" style="margin-top:14px">' + "".join(cards) + "</div>" if cards else "")

        sections.append(_section("holdout", "Final holdout", _safe(holdout, "holdout section", problems)))
        toc.append(("holdout", "Holdout"))

    # ---- desk ------------------------------------------------------------------------
    if journal_dir is not None:
        def desk() -> str:
            cycles = load_desk_journal(journal_dir)
            if cycles.empty:
                return _note(f"No desk cycles found in {journal_dir}.")
            n_created = int(cycles["created"].map(len).sum())
            cost = float(pd.to_numeric(cycles["cost_usd"], errors="coerce").sum())
            changed = int((cycles["final_forecast"].round(6) != cycles["quant_forecast"].round(6)).sum())
            head = (f'<p class="note">{len(cycles):,} cycles from {cycles["now"].min():%Y-%m-%d} to '
                    f'{cycles["now"].max():%Y-%m-%d}; the desk changed the quant forecast in {changed:,} of them, '
                    f'created {n_created:,} ad-hoc specialists and spent ${cost:,.2f} (journal estimate). '
                    'The LLM overlay can only scale or veto the quant forecast in overlay mode, and its '
                    'output still goes through the same sizer and risk manager.</p>')
            figs = fig_desk(F, cycles)
            cards = []
            if figs is not None:
                cards.append(_card(None, emit(figs[0]), wide=True))
                cards.append(_card(None, emit(figs[1]), wide=True))
            cards.append(_card("Cycle log", _desk_table(cycles), wide=True))
            return head + '<div class="grid">' + "".join(cards) + "</div>"

        sections.append(_section("desk", "LLM desk timeline", _safe(desk, "desk timeline", problems)))
        toc.append(("desk", "Desk"))

    # ---- provenance ------------------------------------------------------------------
    def provenance() -> str:
        prov = s.get("provenance") or {}
        st = s.get("settings") or {}
        items = [("Config", f"{s.get('name', '')} · hash {str(s.get('config_hash', ''))[:12]}"),
                 ("Data", f"hash {str(s.get('data_hash', ''))[:12]} · {(prov.get('data') or {}).get('n_bars', '')} bars "
                          f"{(prov.get('data') or {}).get('timeframe', '')}"),
                 ("Git", f"{str(prov.get('git_sha', ''))[:10]}{' (dirty)' if prov.get('git_dirty') else ''}"),
                 ("Created", str(prov.get("created_utc", ""))[:19]),
                 ("Walk-forward", f"{st.get('n_folds', s.get('n_folds', ''))} folds · train {st.get('train', '')} / "
                                  f"test {st.get('test', '')} bars · purge {st.get('purge', '')} · "
                                  f"embargo {st.get('embargo', '')}"),
                 ("DSR trials", str(s.get("n_trials", ""))),
                 ("Command", " ".join(str(a) for a in prov.get("argv", []) or []))]
        dl = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in items if v)
        notes = s.get("notes") or []
        nl = ("<ul class=\"notes\">" + "".join(f"<li>{_esc(n)}</li>" for n in notes) + "</ul>") if notes else ""
        return _card(None, f'<dl class="prov">{dl}</dl>' + (f"<h3 style=\"margin-top:12px\">Run notes</h3>{nl}"
                                                              if nl else ""), wide=True)

    sections.append(_section("provenance", "Provenance", _safe(provenance, "provenance", problems)))
    toc.append(("provenance", "Provenance"))

    if problems:
        logger.warning("dashboard: %d panel(s) failed: %s", len(problems), "; ".join(problems))

    oos = f"OOS {str(s.get('oos_start', ''))[:10]} → {str(s.get('oos_end', ''))[:10]}"
    sub = (f"{_esc(kind)} · {_esc(oos)} · {_esc(s.get('n_folds', ''))} folds · "
           f"{_esc(len(strategies))} strategies · <code>{_esc(run.run_dir)}</code>")
    palettes = json.dumps({m: palette(m) for m in ("light", "dark")}).replace("</", "<\\/")
    nav = "".join(f'<a href="#{sid}">{_esc(lbl)}</a>' for sid, lbl in toc)
    doc = (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<meta name=\"generator\" content=\"aurum.viz.dashboard\"><title>{_esc(page_title)}</title>"
        f"<style>{_css()}</style></head><body><div class=\"wrap-all\">"
        f"<header class=\"top\"><div><h1>{_esc(page_title)}</h1><p class=\"sub\">{sub}</p></div>"
        "<button id=\"theme-toggle\" class=\"theme\" type=\"button\" aria-pressed=\"false\">Dark mode</button></header>"
        f"<nav class=\"toc\" aria-label=\"sections\">{nav}</nav><main>{''.join(sections)}</main></div>"
        f"<script>{_JS.replace('__PALETTES__', palettes)}</script></body></html>\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
