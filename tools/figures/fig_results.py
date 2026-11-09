"""Figures for docs/RESULTS.md: the pre-registered walk-forward results and the exploratory
v2.1 trials, drawn from the run directories (read-only, research OOS up to 2024-12-31).

Nothing here re-runs a walk-forward: equity curves, fold Sharpes, confidence intervals and
DSRs are read from each run's books/, folds.csv and stats.csv (the numbers of record). The
2025-26 holdout is never drawn: the loaders refuse holdout/ paths and cut run tables at the
figure cutoff.

Render with:  python tools/figures/build.py --only results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_TALL,
    ink_on,
    panels,
    register,
    run_json,
    run_label,
    run_table,
    signed,
    source_note,
)

from aurum.viz.style import (
    add_legend,
    diverging_cmap,
    figure,
    label_line_end,
    percent_axis,
    style_axes,
    zero_line,
)

PAGE = "results"

SEL_DEFAULT = "protocol/default_selection"
SEL_TREND = "protocol/trend_core_selection"
HOLDOUT_RUN = "protocol/trend_core_holdout"
D1_TREND = "v21_d1_trend"
MANAGED = "v21_managed_gold"
SWEEP = "v21_demo/sweep_ema_cross"


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _equity(name: str, book: str) -> pd.Series:
    ts = run_table(name, f"books/{book}/timeseries.parquet")
    if ts is None:
        raise FileNotFoundError(f"runs/{name}/books/{book}/timeseries.parquet")
    eq = ts["equity"].astype(float)
    daily = eq.groupby(eq.index.normalize()).last()       # end-of-UTC-day marks, for plotting
    # keep the starting equity (the first bar carries no PnL) so growth is measured from it
    return pd.concat([eq.iloc[:1], daily.loc[daily.index > eq.index[0]]])


def _stats(name: str) -> pd.DataFrame:
    st = run_table(name, "stats.csv", cut=False)
    if st is None:
        raise FileNotFoundError(f"runs/{name}/stats.csv")
    return st.set_index(st.columns[0])


def _ink(ax, x, y, text, pal, *, ha="left", va="center", size=8.5, color="text_secondary", **kw):
    return ax.annotate(text, (x, y), ha=ha, va=va, fontsize=size, color=pal[color], **kw)


def _growth_panels(pal, series, title, note):
    """Growth of 100 (log scale, top) and drawdown (bottom) for (label, equity, colour) tuples."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fig, (ax0, ax1) = panels(2, heights=(2.3, 1.0), figsize=FIGSIZE_TALL, hspace=0.06)
    for label, eq, color in series:
        g = 100.0 * eq / float(eq.iloc[0])
        line, = ax0.plot(g.index, g.values, color=color, label=label)
        label_line_end(ax0, line, text=f"{label}  {g.iloc[-1]:,.0f}")
        dd = eq / eq.cummax() - 1.0
        ax1.plot(dd.index, dd.values, color=color, linewidth=1.2)
    ax0.set_yscale("log")
    ticks = [60, 70, 80, 90, 100, 125, 150, 175, 200, 250]
    lo, hi = ax0.get_ylim()
    ax0.set_yticks([t for t in ticks if lo <= t <= hi])
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax0.yaxis.set_minor_formatter(NullFormatter())
    ax0.axhline(100.0, color=pal["axis"], linewidth=0.75, zorder=0.5)
    ax0.set_ylabel("growth of 100 (log scale)")
    ax0.set_title(title)
    percent_axis(ax1)
    ax1.set_ylabel("drawdown")
    ax1.set_xlabel("date (UTC)")
    source_note(fig, note, pal)
    return fig


# ---------------------------------------------------------------------------------------
# pre-registered protocol
# ---------------------------------------------------------------------------------------
@register(PAGE, "equity-vs-buy-and-hold", needs=(f"runs/{SEL_DEFAULT}", f"runs/{SEL_TREND}"))
def equity_vs_buy_and_hold(pal):
    """Stitched OOS equity of both combined books against buy-and-hold, with drawdowns."""
    cats = pal["categorical"]
    series = [("buy & hold", _equity(SEL_TREND, "benchmark"), pal["text_muted"]),
              ("trend_core", _equity(SEL_TREND, "combined"), cats[0]),
              ("default", _equity(SEL_DEFAULT, "combined"), cats[1])]
    fig = _growth_panels(pal, series, "Walk-forward OOS: both combined books against buy-and-hold",
                         "Data: runs/protocol/{default,trend_core}_selection, walk-forward OOS 2015-12-16 "
                         "to 2024-12-31, net of costs and financing")
    return fig


@register(PAGE, "fold-sharpe", needs=(f"runs/{SEL_DEFAULT}", f"runs/{SEL_TREND}"))
def fold_sharpe(pal):
    """OOS Sharpe of the combined book in each of the 18 folds, both configurations."""
    cats = pal["categorical"]
    fd = run_table(SEL_DEFAULT, "folds.csv", cut=False).set_index("fold")
    ft = run_table(SEL_TREND, "folds.csv", cut=False).set_index("fold")
    x = np.arange(len(ft))
    fig, ax = figure(figsize=FIGSIZE)
    w = 0.38
    ax.bar(x - w / 2, fd["oos_sharpe"].to_numpy(), width=w, color=cats[1], label="default (14 strategies)")
    ax.bar(x + w / 2, ft["oos_sharpe"].to_numpy(), width=w, color=cats[0], label="trend_core (7 strategies)")
    zero_line(ax)
    starts = pd.to_datetime(ft["test_start"], utc=True)
    ax.set_xticks(x, [f"{k}\n{t:%b %y}" for k, t in zip(ft.index, starts, strict=True)])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlim(-0.7, len(ft) - 0.3)
    ax.set_ylabel("annualised Sharpe in the fold")
    ax.set_xlabel("fold (and month its 6-month test block starts)")
    ax.set_title("Per-fold OOS Sharpe of the combined books")
    add_legend(ax, loc="lower left", fontsize=8.5)
    pos_d, pos_t = int((fd["oos_sharpe"] > 0).sum()), int((ft["oos_sharpe"] > 0).sum())
    _ink(ax, 0.99, 0.97, f"folds with a positive Sharpe: default {pos_d}/18, trend_core {pos_t}/18", pal,
         ha="right", va="top", size=8, xycoords="axes fraction")
    r15 = float(ft.loc[15, "oos_return"])
    _ink(ax, 15, float(min(fd.loc[15, "oos_sharpe"], ft.loc[15, "oos_sharpe"])),
         f"book flat: return {signed(100 * r15, '{:+.4f}')}%;\nthe Sharpe of near-zero\nreturns means nothing", pal,
         ha="center", va="top", size=7.5, xytext=(0, -4), textcoords="offset points")
    source_note(fig, "Data: folds.csv of runs/protocol/{default,trend_core}_selection (slices of the "
                "stitched OOS equity)", pal)
    return fig


def _forest(ax, rows, pal, *, dsr_header="DSR"):
    """rows: list of (label, sharpe, lo, hi, dsr_text, kind) top to bottom; kind in
    {"combined", "strategy", "benchmark", "header"}."""
    cats = pal["categorical"]
    color = {"combined": cats[1], "strategy": cats[0], "benchmark": pal["text_muted"]}
    n = len(rows)
    labels = []
    for i, (lab, sr, lo, hi, dsr, kind) in enumerate(rows):
        y = n - 1 - i
        labels.append(lab)
        if kind == "header" or sr is None or not np.isfinite(sr):
            if kind != "header":
                _ink(ax, 0.0, y, "  no trades (skill gate)", pal, size=7.5, color="text_muted")
            continue
        ax.plot([lo, hi], [y, y], color=color[kind], linewidth=1.5, solid_capstyle="butt")
        ax.plot([sr], [y], marker="o", color=color[kind], linestyle="none", markersize=5)
        ax.annotate(dsr, (1.0, y), xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=pal["text_secondary"])
    ax.annotate(dsr_header, (1.0, n - 0.35), xycoords=("axes fraction", "data"), xytext=(8, 0),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, color=pal["text"],
                fontweight="bold")
    ax.set_yticks(range(n - 1, -1, -1), labels)
    for t, (_, _, _, _, _, kind) in zip(ax.get_yticklabels(), rows, strict=True):
        if kind in ("combined", "header"):
            t.set_fontweight("bold")
            t.set_color(pal["text"])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    zero_line(ax, axis="x")
    style_axes(ax, grid="x")


def _rows_from_stats(st: pd.DataFrame, *, bench_label: str = "buy & hold") -> list:
    rows = []
    order = ["combined"] + [b for b in st.index if b not in ("combined", "benchmark")] + ["benchmark"]
    for b in order:
        if b not in st.index:
            continue
        r = st.loc[b]
        kind = "combined" if b == "combined" else "benchmark" if b == "benchmark" else "strategy"
        label = bench_label if b == "benchmark" else b
        sr = float(r["sharpe"]) if pd.notna(r["sharpe"]) else float("nan")
        dsr = "" if not np.isfinite(sr) else (f"{r['dsr']:.3f}" if kind != "benchmark" else f"{r['dsr']:.3f}*")
        rows.append((label, sr, float(r["ci_lower"]), float(r["ci_upper"]), dsr, kind))
    return rows


@register(PAGE, "forest-protocol", needs=(f"runs/{SEL_DEFAULT}", f"runs/{SEL_TREND}"))
def forest_protocol(pal):
    """Sharpe with 95% stationary-bootstrap CI and DSR for every book of both protocol runs."""
    rd = _rows_from_stats(_stats(SEL_DEFAULT))
    rt = _rows_from_stats(_stats(SEL_TREND))
    fig, (ax0, ax1) = panels(2, heights=(len(rd), len(rt)), figsize=(8.0, 9.0), hspace=0.05)
    _forest(ax0, rd, pal)
    _forest(ax1, rt, pal)
    ax0.set_title("default.yaml (14 strategies), walk-forward OOS 2016-2024", fontsize=10.5)
    ax1.set_title("trend_core.yaml (7 strategies, selected)", fontsize=10.5)
    lo = min(r[2] for r in rd + rt if np.isfinite(r[1]))
    hi = max(r[3] for r in rd + rt if np.isfinite(r[1]))
    ax1.set_xlim(lo - 0.1, hi + 0.1)
    ax1.set_xlabel("annualised daily Sharpe with 95% stationary-bootstrap CI (n_trials = 23 for DSR)")
    fig.suptitle("Every protocol book: CI lower bound below 0, DSR far below 0.95", x=0.02, ha="left",
                 fontsize=12, color=pal["text"])
    source_note(fig, "Data: stats.csv of runs/protocol/{default,trend_core}_selection. * buy & hold DSR uses "
                "1 trial (not selected)", pal)
    return fig


@register(PAGE, "monthly-returns", needs=(f"runs/{HOLDOUT_RUN}",))
def monthly_returns(pal):
    """Monthly returns of the selected configuration's combined book over its research OOS."""
    ts = run_table(HOLDOUT_RUN, "books/combined/timeseries.parquet")       # research book, cut at 2024
    eq = ts["equity"].astype(float)
    naive = eq.index.tz_convert("UTC").tz_localize(None)
    month_end = eq.groupby(naive.to_period("M").astype(str)).last()
    prev = month_end.shift(1)
    prev.iloc[0] = float(eq.iloc[0])
    mret = month_end / prev - 1.0
    per = pd.PeriodIndex(mret.index, freq="M")
    grid = pd.DataFrame({"y": per.year, "m": per.month, "r": mret.to_numpy()}).pivot(index="y", columns="m",
                                                                                   values="r")
    grid = grid.reindex(columns=range(1, 13))
    years = grid.index.to_numpy()
    year_end = eq.groupby(eq.index.year).last()
    yprev = year_end.shift(1)
    yprev.iloc[0] = float(eq.iloc[0])
    yret = (year_end / yprev - 1.0).reindex(years)

    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.0, 5.0))
    ax, axy = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": (12, 1.6)})
    cmap = diverging_cmap(pal["mode"])
    z = grid.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(z)))
    im = ax.imshow(z, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, _pct(z[i, j]), ha="center", va="center", fontsize=7.5,
                        color=ink_on(im.cmap(im.norm(z[i, j]))))
    ylim_ = float(np.nanmax(np.abs(yret.to_numpy())))
    imy = axy.imshow(yret.to_numpy()[:, None], cmap=cmap, vmin=-ylim_, vmax=ylim_, aspect="auto")
    for i, v in enumerate(yret.to_numpy()):
        axy.text(0, i, _pct(v), ha="center", va="center", fontsize=7.5,
                 color=ink_on(imy.cmap(imy.norm(v))))
    for a in (ax, axy):
        style_axes(a, grid=None)
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        a.tick_params(length=0)
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(years)), [str(y) for y in years])
    axy.set_xticks([0], ["year"])
    ax.set_title("trend_core combined book: monthly returns in % (walk-forward OOS)")
    ax.set_xlabel("month (cells: % return; colour scale symmetric at the largest month)")
    axy.set_xlabel("own scale")
    source_note(fig, "Data: " + run_label(HOLDOUT_RUN) + "; the 2025-26 holdout months are not drawn", pal)
    return fig


def _pct(v: float) -> str:
    """A monthly return in % with one decimal and a true minus sign ("0.0" for tiny values)."""
    return "0.0" if abs(100 * v) < 0.05 else signed(100 * v, "{:+.1f}")


# ---------------------------------------------------------------------------------------
# exploratory v2.1 trials
# ---------------------------------------------------------------------------------------
@register(PAGE, "forest-exploratory", needs=(f"runs/{D1_TREND}", f"runs/{MANAGED}", f"runs/{SWEEP}"))
def forest_exploratory(pal):
    """Sharpe, 95% CI and DSR of the exploratory v2.1 trials (D1 runs and the EMA sweep)."""
    from aurum.research.stats import sharpe_summary

    d1 = _rows_from_stats(_stats(D1_TREND))
    mg = [r for r in _rows_from_stats(_stats(MANAGED)) if r[0] != "tsmom"]      # same tsmom as d1_trend
    sweep = run_table(SWEEP, "sweep.csv", cut=False)
    ret = run_table(SWEEP, "returns.csv")
    sj = run_json(SWEEP, "sweep.json")
    n_trials = int(sj["n_trials"])
    ok = sweep[sweep["status"] == "ok"].sort_values("sharpe", ascending=False)
    sw = []
    for _, r in ok.iterrows():
        x = ret[r["label"]].dropna().to_numpy(float)
        dsr = sharpe_summary(x, 252.0, n_trials=n_trials, bootstrap=False)["dsr"]
        lab = f"ema_cross {int(r['fast'])}/{int(r['slow'])}" + ("  (default)" if (r["fast"], r["slow"]) == (32, 128)
                                                               else "")
        sw.append((lab, float(r["sharpe"]), float(r["ci_lower"]), float(r["ci_upper"]), f"{dsr:.3f}", "strategy"))
    rows = ([("d1_trend (D1, 2016-2024)", None, None, None, "", "header")] + d1
            + [("managed_gold (D1, 2016-2024)", None, None, None, "", "header")] + mg
            + [(f"ema_cross sweep (H1, OOS 2019-2024, N = {n_trials})", None, None, None, "", "header")] + sw)
    fig, ax = figure(figsize=(8.0, 7.6))
    _forest(ax, rows, pal)
    ax.set_xlabel("annualised daily Sharpe with 95% stationary-bootstrap CI")
    ax.set_title("Exploratory v2.1 trials: every CI includes 0, best DSR 0.19")
    source_note(fig, f"Data: runs/{D1_TREND}, runs/{MANAGED} (n_trials 30), runs/{SWEEP}; "
                "* buy & hold DSR uses 1 trial", pal)
    return fig


@register(PAGE, "exploratory-equity", needs=(f"runs/{D1_TREND}", f"runs/{MANAGED}"))
def exploratory_equity(pal):
    """Growth of 100 for the daily-bar trials against daily buy-and-hold."""
    cats = pal["categorical"]
    series = [("buy & hold", _equity(MANAGED, "benchmark"), pal["text_muted"]),
              ("managed_gold", _equity(MANAGED, "managed_gold"), cats[0]),
              ("regime_switch", _equity(MANAGED, "regime_switch"), cats[1]),
              ("d1_trend combined", _equity(D1_TREND, "combined"), cats[2])]
    return _growth_panels(pal, series, "Daily-bar trials against buy-and-hold (exploratory, not pre-registered)",
                          f"Data: runs/{MANAGED}, runs/{D1_TREND}; D1 New York close bars, walk-forward OOS "
                          "2016-01-03 to 2024-12-30, net of costs and financing")
