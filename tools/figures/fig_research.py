"""Figures for docs/research.md: walk-forward geometry, purge/embargo, OOS combiner fitting,
the stitched OOS equity, and the statistics (stationary bootstrap, PSR, DSR, MinTRL, PBO).

Every run-based figure reads the protocol runs in ``runs/protocol/`` (read-only; their
research OOS ends 2024-12-31) and recomputes nothing that needs a walk-forward. The concept
figures call aurum's own functions in :mod:`aurum.research.stats`.

Render with:  python tools/figures/build.py --only research
"""

from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_TALL,
    bars,
    panels,
    register,
    run_file,
    run_json,
    run_label,
    run_table,
    signed,
    source_label,
    source_note,
)

from aurum.backtest.metrics import daily_returns
from aurum.research.stats import (
    expected_max_sharpe,
    min_track_record_length,
    pbo_cscv,
    probabilistic_sharpe,
    stationary_bootstrap,
    stationary_bootstrap_indices,
)
from aurum.viz.style import (
    add_legend,
    figure,
    label_line_end,
    percent_axis,
    shade_span,
    style_axes,
    zero_line,
)

PAGE = "research"

SEL_DEFAULT = "protocol/default_selection"
SEL_TREND = "protocol/trend_core_selection"
HOLDOUT_RUN = "protocol/trend_core_holdout"
TREND_BOOKS = ("tsmom", "ema_cross", "donchian", "kalman_trend", "vol_squeeze", "macro_factor", "risk_off")
DAYS = 252.0


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _folds(name: str) -> pd.DataFrame:
    f = run_table(name, "folds.csv", cut=False)
    if f is None:
        raise FileNotFoundError(f"runs/{name}/folds.csv")
    for c in ("train_start", "train_end", "test_start", "test_end"):
        f[c] = pd.to_datetime(f[c], utc=True)
    return f.set_index("fold") if "fold" in f.columns else f


def _book_daily(name: str, book: str) -> pd.Series:
    ts = run_table(name, f"books/{book}/timeseries.parquet")
    if ts is None:
        raise FileNotFoundError(f"runs/{name}/books/{book}/timeseries.parquet")
    return daily_returns(ts["equity"]).dropna()


def _ink(ax, x, y, text, pal, *, ha="left", va="center", size=8.5, color="text_secondary", **kw):
    return ax.annotate(text, (x, y), ha=ha, va=va, fontsize=size, color=pal[color], **kw)


# ---------------------------------------------------------------------------------------
# walk-forward geometry
# ---------------------------------------------------------------------------------------
@register(PAGE, "wf-geometry", needs=("data_store", "runs/protocol", f"runs/{HOLDOUT_RUN}"))
def wf_geometry(pal):
    """The 18 real rolling folds of the protocol run plus its holdout fold, on a date axis."""
    folds = _folds(HOLDOUT_RUN)
    summ = run_json(HOLDOUT_RUN, "summary.json")
    st = summ["settings"]
    blue, orange = pal["categorical"][0], pal["categorical"][1]

    # the holdout fold's training window: the `train` bars ending purge+embargo bars before
    # the first holdout bar (plan_folds); positions are in the run's research index
    idx = bars("H1", start="2013-01-01").index
    if len(idx) != st["n_research"]:
        raise RuntimeError(f"bar index has {len(idx)} rows, run used {st['n_research']}")
    gap = int(st["purge"]) + int(st["embargo"])
    tr1 = st["n_research"] - gap
    tr0 = tr1 - int(st["train"])
    h_train = (idx[tr0], idx[tr1 - 1])
    h_start = pd.Timestamp(st["holdout_start"])
    ledger = run_file("protocol", "holdout_ledger.jsonl")
    h_end = None
    if ledger is not None:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("config_hash") == summ["config_hash"]:
                h_end = pd.Timestamp(rec["holdout_end"])
    h_end = h_end if h_end is not None else pd.Timestamp("2026-09-25 20:00", tz="UTC")

    fig, ax = figure(figsize=FIGSIZE_TALL)
    n = len(folds)
    h = 0.62
    for k, row in folds.iterrows():
        y = n - int(k)
        ax.barh(y, row.train_end - row.train_start, left=row.train_start, height=h, color=blue,
                label="train (3Y = 17,834 bars)" if int(k) == 0 else None)
        ax.barh(y, row.test_end - row.test_start, left=row.test_start, height=h, color=orange,
                label="test (6M = 2,972 bars)" if int(k) == 0 else None)
    ax.barh(0, h_train[1] - h_train[0], left=h_train[0], height=h, color=blue)
    shade_span(ax, h_start, h_end)
    ax.barh(0, h_end - h_start, left=h_start, height=h, color=pal["text_muted"], hatch="////",
            edgecolor=pal["surface"], linewidth=0, label="final holdout (evaluated once, not drawn)")
    _ink(ax, h_start, n + 1.3, "holdout from\n2025-01-01", pal, va="top", size=8,
         xytext=(4, 0), textcoords="offset points", color="text_muted")
    ax.set_yticks([n - int(k) for k in folds.index] + [0],
                  [f"fold {k}" for k in folds.index] + ["holdout"])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(-0.8, n + 1.4)
    ax.set_xlim(pd.Timestamp("2012-10-01", tz="UTC"), h_end + pd.Timedelta(days=60))
    ax.grid(False)
    style_axes(ax, grid="x")
    ax.set_title("Walk-forward folds of the protocol run (rolling 3Y train / 6M test)")
    ax.set_xlabel("date (UTC)")
    add_legend(ax, loc="lower left", ncols=1, fontsize=8.5)
    _ink(ax, folds.iloc[0].test_start, n + 0.85,
         f"gap of {gap} bars before every test block (purge {st['purge']} + embargo {st['embargo']}): "
         "too short to see here", pal, size=8, color="text_muted")
    source_note(fig, "Data: " + run_label(HOLDOUT_RUN) + "; XAUUSD H1 bar index", pal)
    return fig


@register(PAGE, "purge-embargo-zoom",
          needs=("data_store", f"runs/{SEL_DEFAULT}", f"runs/{SEL_TREND}"))
def purge_embargo_zoom(pal):
    """Zoom on the fold 0 -> 1 boundary: train, purge, embargo and test bars on real H1 closes."""
    close = bars("H1", start="2013-01-01")["close"]
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, axes = panels(2, figsize=(8.0, 6.2), sharex=True, hspace=0.12)
    window = (len(close), 0)
    for ax, name, title in ((axes[0], SEL_DEFAULT, "default.yaml: purge 24 (ml_gbm, meta_label label horizon) + embargo 24"),
                            (axes[1], SEL_TREND, "trend_core.yaml: no trainable label horizon, purge 0 + embargo 24")):
        st = run_json(name, "summary.json")["settings"]
        purge, emb = int(st["purge"]), int(st["embargo"])
        row = _folds(name).loc[1]
        i_test = close.index.get_loc(row.test_start)
        i_tr_end = close.index.get_loc(row.train_end)          # last training bar (inclusive)
        assert i_test - i_tr_end - 1 == purge + emb
        lo, hi = i_tr_end - 40, i_test + 36
        window = (min(lo, window[0]), max(hi, window[1]))
        x = np.arange(lo, hi) - i_test                          # bars relative to the first test bar
        y = close.iloc[lo:hi].to_numpy()
        tr = x <= i_tr_end - i_test
        te = x >= 0
        mid = ~tr & ~te
        ax.plot(x[tr], y[tr], color=blue)
        ax.plot(x[te], y[te], color=orange)
        # gap bars joined to their neighbours so the line is continuous
        gx = np.r_[x[tr][-1], x[mid], x[te][0]]
        gy = np.r_[y[tr][-1], y[mid], y[te][0]]
        ax.plot(gx, gy, color=pal["text_muted"], linestyle=(0, (2, 2)))
        p0 = i_tr_end - i_test + 0.5
        if purge:
            shade_span(ax, p0, p0 + purge)
        ymax = float(np.nanmax(y))
        ymin = float(np.nanmin(y))
        pad = (ymax - ymin) * 0.10
        ax.set_ylim(ymin - pad, ymax + 3.2 * pad)
        top = ymax + 2.6 * pad
        if purge:
            _ink(ax, p0 + purge / 2, top, f"purge\n{purge} bars", pal, ha="center", size=8)
            ax.axvline(p0 + purge, color=pal["axis"], linewidth=0.75)
        _ink(ax, p0 + purge + emb / 2, top, f"embargo\n{emb} bars", pal, ha="center", size=8)
        ax.axvline(p0, color=pal["axis"], linewidth=0.75)
        ax.axvline(-0.5, color=pal["axis"], linewidth=0.75)
        _ink(ax, x[0] + 1, top, "train (fold 1)", pal, size=8, color="text")
        _ink(ax, 1.5, top, "test (fold 1)", pal, size=8, color="text")
        _ink(ax, p0 - 1, ymin - 0.6 * pad, f"last train bar\n{row.train_end:%a %d %b %H:%M} UTC", pal,
             ha="right", va="bottom", size=7.5, color="text_muted")
        _ink(ax, 1, ymin - 0.6 * pad, f"first test bar\n{row.test_start:%a %d %b %H:%M} UTC", pal,
             ha="left", va="bottom", size=7.5, color="text_muted")
        if purge:   # the label of the last training row looks `purge` bars ahead
            yy = ymax + 1.15 * pad
            ax.annotate("", xy=(p0 + purge, yy), xytext=(p0 - 0.5, yy),
                        arrowprops={"arrowstyle": "->", "color": pal["text_secondary"], "linewidth": 1.0})
            _ink(ax, p0 - 1, yy, "label of the last train bar ends here", pal, ha="right", size=7.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("USD per ounce")
    axes[1].set_xlabel("bars relative to the first test bar (H1; the weekend has no bars)")
    shown = close.index[window[0]:window[1]]
    source_note(fig, "Data: " + source_label("H1", start=f"{shown[0]:%Y-%m-%d}", end=f"{shown[-1]:%Y-%m-%d}",
                                             years=False)
                + "; fold boundaries from runs/protocol/*_selection/folds.csv", pal)
    return fig


# ---------------------------------------------------------------------------------------
# combiner fitted on earlier OOS forecasts only
# ---------------------------------------------------------------------------------------
@register(PAGE, "combiner-oos-timeline", needs=(f"runs/{SEL_TREND}",))
def combiner_oos_timeline(pal):
    """Per fold: the earlier-OOS window the combiner was fitted on, and the weights it chose."""
    import matplotlib.dates as mdates

    folds = _folds(SEL_TREND)
    w = run_table(SEL_TREND, "weights.csv", cut=False).set_index("fold")
    oos = run_table(SEL_TREND, "oos_forecasts.parquet")
    st = run_json(SEL_TREND, "summary.json")["settings"]
    window = int(st["train"])
    cats = pal["categorical"]
    blue, orange = cats[0], cats[1]

    fig, (ax0, ax1) = panels(2, heights=(1.25, 1.0), figsize=(8.0, 7.0), hspace=0.08)
    n = len(folds)
    for k, row in folds.iterrows():
        y = n - 1 - int(k)
        hist = oos.index[oos.index < row.test_start][-window:]
        if len(hist) >= 500:
            ax0.barh(y, hist[-1] - hist[0], left=hist[0], height=0.62, color=blue,
                     label="combiner fit window: earlier folds' OOS forecasts (last 17,834 bars)" if int(k) == 1 else None)
        else:
            _ink(ax0, row.test_end, y, "  equal weights (no earlier OOS history yet)", pal, size=8)
        ax0.barh(y, row.test_end - row.test_start, left=row.test_start, height=0.62, color=orange,
                 label="test block the weights are used on" if int(k) == 0 else None)
    ax0.set_yticks([n - 1 - int(k) for k in folds.index], [f"fold {k}" for k in folds.index])
    ax0.tick_params(axis="y", labelsize=7.5)
    ax0.set_ylim(-0.7, n - 0.3)
    style_axes(ax0, grid="x")
    ax0.set_title("Each fold's combiner is fitted only on the OOS forecasts of earlier folds")
    add_legend(ax0, loc="upper right", fontsize=8.5)

    # stacked weights per fold, drawn over the fold's test block
    keys = [c for c in w.columns if c != "fdm"]
    left = mdates.date2num(folds["test_start"].dt.tz_localize(None).to_numpy())
    right = mdates.date2num(folds["test_end"].dt.tz_localize(None).to_numpy())
    width = right - left
    bottom = np.zeros(n)
    for i, key in enumerate(keys):
        vals = w.loc[folds.index.astype(int), key].to_numpy(dtype=float)
        ax1.bar(left, vals, width=width * 0.96, bottom=bottom, align="edge", color=cats[i], label=key)
        bottom += vals
    unalloc = np.clip(1.0 - bottom, 0.0, 1.0)
    ax1.bar(left, unalloc, width=width * 0.96, bottom=bottom, align="edge", color=pal["grid"],
            hatch="////", edgecolor=pal["text_muted"], linewidth=0, label="unallocated (1 - sum of weights)")
    ax1.xaxis_date()
    ax1.set_ylim(0, 1.0)
    percent_axis(ax1)
    ax1.set_ylabel("combiner weight")
    ax1.set_xlabel("date (UTC)")
    ax1.set_title("Weights chosen per fold (sharpe_shrink, net of costs, max weight 40%)", fontsize=10.5)
    add_legend(ax1, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=4, fontsize=8)
    f15 = folds.loc[15] if 15 in folds.index else None
    if f15 is not None:
        _ink(ax1, f15.test_start + (f15.test_end - f15.test_start) / 2, 0.5, "fold 15\nflat",
             pal, ha="center", size=7.5, color="text")
    source_note(fig, "Data: " + run_label(SEL_TREND) + " (folds.csv, weights.csv, oos_forecasts.parquet)", pal)
    return fig


@register(PAGE, "stitched-oos-equity", needs=(f"runs/{SEL_TREND}",))
def stitched_oos_equity(pal):
    """One continuous OOS backtest across all folds: equity (top), combined forecast (bottom)."""
    folds = _folds(SEL_TREND)
    ts = run_table(SEL_TREND, "books/combined/timeseries.parquet")
    oos = run_table(SEL_TREND, "oos_forecasts.parquet")
    blue = pal["categorical"][0]
    eq = ts["equity"]
    fc = oos["combined"].resample("W").mean().dropna()

    fig, (ax0, ax1) = panels(2, heights=(2.2, 1.0), figsize=FIGSIZE_TALL, hspace=0.06)
    for k, row in folds.iterrows():
        if int(k) % 2 == 1:
            shade_span(ax0, row.test_start, row.test_end)
            shade_span(ax1, row.test_start, row.test_end)
        if int(k) % 3 == 0 or int(k) == 15:
            _ink(ax0, row.test_start + (row.test_end - row.test_start) / 2, 1.0, str(k), pal, ha="center",
                 va="bottom", size=7.5, color="text_muted", xycoords=("data", "axes fraction"))
    line, = ax0.plot(eq.index, eq.values, color=blue)
    ax0.axhline(float(eq.iloc[0]), color=pal["axis"], linewidth=0.75, zorder=0.5)
    label_line_end(ax0, line, fmt="{:,.0f}")
    ax0.set_ylabel("equity (USD)")
    ax0.yaxis.set_major_formatter(_thousands())
    ax0.set_title("Stitched OOS equity of the combined book: one backtest, never reset per fold")
    _ink(ax0, eq.index[-1], float(eq.iloc[0]), "start 100,000 ", pal, ha="right", va="bottom", size=8)
    ax1.plot(fc.index, fc.values, color=blue)
    zero_line(ax1)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_ylabel("forecast (weekly mean)")
    ax1.set_xlabel("date (UTC); shaded bands = odd-numbered folds, numbers = fold keys")
    f15 = folds.loc[15]
    _ink(ax1, f15.test_start + (f15.test_end - f15.test_start) / 2, 0.45, "fold 15\nflat", pal,
         ha="center", size=8, color="text")
    source_note(fig, "Data: " + run_label(SEL_TREND), pal)
    return fig


def _thousands():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: f"{v:,.0f}")


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------
def _sharpe_rows(xs: np.ndarray) -> np.ndarray:
    """Row-wise per-period Sharpe (same as aurum.research.stats' vectorised statistic)."""
    mu = xs.mean(axis=1)
    sd = xs.std(axis=1, ddof=1)
    return np.divide(mu, sd, out=np.full_like(mu, np.nan), where=sd > 0)


@register(PAGE, "stationary-bootstrap", needs=(f"runs/{SEL_TREND}",))
def stationary_bootstrap_fig(pal):
    """Block resampling (left) and the bootstrap Sharpe distribution of the real combined book."""
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1) = figure(1, 2, figsize=FIGSIZE)

    # (a) one stationary-bootstrap resample of 60 days with mean block 8 (illustration)
    n = 60
    idx = stationary_bootstrap_indices(n, 1, 8.0, np.random.default_rng(11))[0]
    starts = np.r_[0, np.flatnonzero(np.diff(idx) != 1) + 1]
    ends = np.r_[starts[1:], n]
    for j, (a, b) in enumerate(zip(starts, ends, strict=True)):
        c = blue if j % 2 == 0 else orange
        ax0.plot(np.arange(a, b), idx[a:b], color=c, marker="o", markersize=3.5, markeredgewidth=0)
    ax0.set_xlim(-2, n + 1)
    ax0.set_ylim(-3, n + 3)
    ax0.set_xlabel("position in the resampled series (day)")
    ax0.set_ylabel("day drawn from the original series")
    ax0.set_title("One resample: blocks of consecutive days", fontsize=10.5)
    _ink(ax0, 1, n + 1, f"{len(starts)} blocks; a new block starts with p = 1/8\nat a uniformly random day",
         pal, va="top", size=8)
    style_axes(ax0, grid="both")

    # (b) the real combined book: 1,000 resamples (seed 7, automatic block length)
    r = _book_daily(SEL_TREND, "combined").to_numpy()
    cfg_seed = 7
    res = stationary_bootstrap(r, _sharpe_rows, n_boot=1000, seed=cfg_seed, vectorized=True,
                               return_samples=True)
    s = res["samples"] * math.sqrt(DAYS)
    lo, hi = res["lower"] * math.sqrt(DAYS), res["upper"] * math.sqrt(DAYS)
    est = res["estimate"] * math.sqrt(DAYS)
    ax1.hist(s, bins=40, color=blue, edgecolor=pal["surface"], linewidth=0.4)
    for v in (lo, hi):
        ax1.axvline(v, color=pal["text_secondary"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax1.axvline(est, color=orange, linewidth=1.5)
    zero_line(ax1, axis="x")
    ymax = ax1.get_ylim()[1]
    _ink(ax1, lo, ymax * 0.97, f"2.5%: {signed(lo)}  ", pal, ha="right", va="top", size=8)
    _ink(ax1, hi, ymax * 0.97, f"  97.5%: {signed(hi)}", pal, ha="left", va="top", size=8)
    _ink(ax1, est, ymax * 0.80, f"  estimate {signed(est)}", pal, ha="left", size=8, color="text")
    ax1.set_xlabel("annualised Sharpe of a resample")
    ax1.set_ylabel("resamples")
    ax1.set_title(f"trend_core combined: 95% CI (block {res['mean_block']:.0f} day)", fontsize=10.5)
    source_note(fig, f"Left: illustration, seed 11. Right: runs/{SEL_TREND}, {len(r):,} daily OOS returns, "
                "1,000 resamples, seed 7", pal)
    return fig


@register(PAGE, "psr-track-length")
def psr_track_length(pal):
    """PSR(0) against years of daily data, for several Sharpe ratios and return shapes."""
    cats = pal["categorical"]
    years = np.logspace(-1, np.log10(60), 400)
    fig, (ax0, ax1) = figure(1, 2, figsize=FIGSIZE)

    def crossing(sr_pp, g3, g4):
        return min_track_record_length(sr_pp, g3, g4) / DAYS        # years at which PSR = 0.95

    for i, sr_ann in enumerate((0.25, 0.5, 1.0, 1.5)):
        sr = sr_ann / math.sqrt(DAYS)
        p = [probabilistic_sharpe(sr, y * DAYS, 0.0, 3.0) for y in years]
        ax0.plot(years, p, color=cats[i])
        yc = crossing(sr, 0.0, 3.0)
        ax0.plot([yc], [0.95], marker="o", color=cats[i], linestyle="none")
        _ink(ax0, yc, 0.95, f"SR {sr_ann:g}\n{yc:.1f} y", pal, ha="center", va="top", size=8, color="text",
             xytext=(0, -8), textcoords="offset points")
    ax0.set_xscale("log")
    ax0.set_xlim(0.1, 60)
    ax0.set_ylim(0.5, 1.005)
    ax0.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50], ["0.1", "0.5", "1", "2", "5", "10", "20", "50"])
    ax0.set_title("Normal returns: years until PSR reaches 0.95", fontsize=10.5)

    shapes = (("normal: skew 0, kurtosis 3", 0.0, 3.0),
              ("trend_core combined: skew 1.23, kurtosis 10.75", 1.2345, 10.753),
              ("zscore_fade: skew -1.93, kurtosis 28.6", -1.933, 28.603))
    yrs = np.linspace(0.1, 6, 300)
    sr = 1.0 / math.sqrt(DAYS)
    for i, (lab, g3, g4) in enumerate(shapes):
        p = [probabilistic_sharpe(sr, y * DAYS, g3, g4) for y in yrs]
        ax1.plot(yrs, p, color=cats[i], label=lab)
        yc = crossing(sr, g3, g4)
        ax1.plot([yc], [0.95], marker="o", color=cats[i], linestyle="none")
        dx, dy, ha, va = ((0, -8, "center", "top"), (-6, 6, "right", "bottom"), (6, -8, "left", "top"))[i]
        _ink(ax1, yc, 0.95, f"{yc:.2f} y", pal, ha=ha, va=va, size=8, color="text",
             xytext=(dx, dy), textcoords="offset points")
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0.75, 1.005)
    ax1.set_title("Sharpe 1.0: skew and fat tails shift the curve", fontsize=10.5)
    add_legend(ax1, loc="lower right", fontsize=7.5)
    for ax in (ax0, ax1):
        ax.axhline(0.95, color=pal["text_muted"], linewidth=0.9, linestyle=(0, (3, 2)), zorder=0.6)
        ax.set_xlabel("track record (years of daily returns" + (", log scale)" if ax is ax0 else ")"))
    ax0.set_ylabel("PSR = P(true Sharpe > 0)")
    source_note(fig, "aurum.research.stats.probabilistic_sharpe; book moments from runs/protocol/*/stats.csv",
                pal)
    return fig


@register(PAGE, "dsr-null")
def dsr_null(pal):
    """Null distribution of the best of N skill-less Sharpes, and SR0 against N."""
    cats = pal["categorical"]
    blue, orange = cats[0], cats[1]
    t_h1, t_d1 = 2346, 2322        # daily returns in the H1 protocol runs and the D1 runs (close-dated)
    rng = np.random.default_rng(23)
    n_exp, n_trials = 20000, 23
    # exact sampling of the sample Sharpe of T iid normal returns with true SR 0:
    # mean ~ N(0, 1/T), (T-1) s^2 ~ chi2(T-1)
    mu = rng.standard_normal((n_exp, n_trials)) / math.sqrt(t_h1)
    sd = np.sqrt(rng.chisquare(t_h1 - 1, (n_exp, n_trials)) / (t_h1 - 1))
    best = (mu / sd).max(axis=1) * math.sqrt(DAYS)
    sr0 = expected_max_sharpe(n_trials, 1.0 / (t_h1 - 1)) * math.sqrt(DAYS)
    observed = 0.094               # highest strategy-book Sharpe of the selected trend_core run (ema_cross)

    fig, (ax0, ax1) = figure(1, 2, figsize=FIGSIZE)
    ax0.hist(best, bins=60, color=blue, edgecolor=pal["surface"], linewidth=0.3)
    ax0.axvline(sr0, color=orange, linewidth=1.5)
    ax0.axvline(observed, color=pal["text_secondary"], linewidth=0.9, linestyle=(0, (3, 2)))
    ax0.set_xlim(0, 1.5)
    ymax = ax0.get_ylim()[1]
    ax0.set_ylim(0, ymax * 1.18)
    _ink(ax0, sr0, ymax * 1.12, f"  SR0 = {sr0:.3f} (formula)", pal, va="top", size=8.5, color="text")
    _ink(ax0, observed, ymax * 0.70, "  best trend_core\n  book: ema_cross\n  0.094", pal, size=7.5)
    _ink(ax0, 0.98, 0.80, f"simulated mean {best.mean():.3f}\nmin {best.min():.3f}: every experiment\n"
         f"beats {observed}", pal, ha="right", va="top", size=7.5, xycoords="axes fraction")
    ax0.set_xlabel("highest annualised Sharpe among 23 noise strategies")
    ax0.set_ylabel("simulated experiments")
    ax0.set_title("Best of 23 skill-less strategies, 2,346 days", fontsize=10.5)

    n = np.unique(np.round(np.logspace(0, 3, 120)).astype(int))
    for i, (t, lab) in enumerate(((t_h1, "T = 2,346 days (H1 protocol runs)"),
                                  (t_d1, "T = 2,322 days (D1 runs)"))):
        y = [expected_max_sharpe(int(k), 1.0 / (t - 1)) * math.sqrt(DAYS) for k in n]
        ax1.plot(n, y, color=cats[i], label=lab)
    for nn, t, c in ((23, t_h1, blue), (30, t_d1, orange)):
        v = expected_max_sharpe(nn, 1.0 / (t - 1)) * math.sqrt(DAYS)
        ax1.plot([nn], [v], marker="o", color=c, linestyle="none")
        _ink(ax1, nn, v, f"  N={nn}: {v:.3f}", pal, va="top", size=8,
             xytext=(4, -4), textcoords="offset points")
    ax1.set_xscale("log")
    ax1.set_xlabel("number of trials N (log scale)")
    ax1.set_ylabel("SR0, annualised")
    ax1.set_title("The bar rises with the number of trials N", fontsize=10.5)
    add_legend(ax1, loc="upper left", fontsize=7.5)
    style_axes(ax1, grid="both")
    source_note(fig, "Synthetic null: 20,000 experiments, seed 23. SR0: aurum.research.stats.expected_max_sharpe, "
                "V = 1/(T-1)", pal)
    return fig


@register(PAGE, "min-trl", needs=(f"runs/{SEL_TREND}",))
def min_trl(pal):
    """Minimum track record length against the Sharpe ratio, with the real books marked."""
    blue = pal["categorical"][0]
    stats = run_table(SEL_TREND, "stats.csv", cut=False).set_index("Unnamed: 0")
    sr_ann = np.logspace(np.log10(0.008), np.log10(3.0), 300)
    fig, ax = figure(figsize=FIGSIZE)
    yrs = [min_track_record_length(s / math.sqrt(DAYS), 0.0, 3.0) / DAYS for s in sr_ann]
    ax.plot(sr_ann, yrs, color=blue)
    _ink(ax, 1.6, min_track_record_length(1.6 / math.sqrt(DAYS)) / DAYS, "normal returns", pal, va="bottom",
         size=8, xytext=(0, 6), textcoords="offset points")
    n_days = float(stats.loc["combined", "n_days"])
    oos_years = n_days / DAYS
    ax.axhline(oos_years, color=pal["text_muted"], linewidth=0.9, linestyle=(0, (3, 2)))
    _ink(ax, 0.0085, oos_years, f"OOS sample: {oos_years:.1f} years", pal, ha="left", va="bottom", size=8,
         color="text_muted", xytext=(0, 2), textcoords="offset points")
    offsets = {"benchmark": (6, 2), "ema_cross": (6, -9), "vol_squeeze": (-6, 8), "kalman_trend": (6, 0)}
    for book, (dx, dy) in offsets.items():
        s, y = float(stats.loc[book, "sharpe"]), float(stats.loc[book, "min_trl_years"])
        if not (np.isfinite(y) and s > 0.0):
            continue
        ax.plot([s], [y], marker="o", color=pal["text"], linestyle="none", markersize=5)
        name = "buy & hold" if book == "benchmark" else book
        _ink(ax, s, y, f"{name}: SR {s:.3f}, {y:,.0f} years", pal, size=8, color="text",
             ha="left" if dx > 0 else "right", xytext=(dx, dy), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3e5)
    ax.set_xlim(0.008, 3.0)
    ax.set_xlabel("annualised Sharpe ratio (log scale)")
    ax.set_ylabel("years for PSR(0) = 0.95 (log scale)")
    ax.set_title("Minimum track record length: halve the Sharpe, quadruple the wait")
    style_axes(ax, grid="both")
    source_note(fig, f"Curve: aurum.research.stats.min_track_record_length. Points: runs/{SEL_TREND}/stats.csv "
                "(each book's own skew and kurtosis)", pal)
    return fig


@register(PAGE, "pbo-cscv", needs=(f"runs/{SEL_DEFAULT}", f"runs/{SEL_TREND}"))
def pbo_figure(pal):
    """CSCV split combinatorics, logit histograms and the IS -> OOS degradation of real runs."""
    cats = pal["categorical"]
    blue, orange = cats[0], cats[1]
    res = {}
    for name in (SEL_DEFAULT, SEL_TREND):
        stats = run_table(name, "stats.csv", cut=False).set_index("Unnamed: 0")
        books = [b for b in stats.index if b not in ("combined", "benchmark")]
        m = pd.DataFrame({b: _book_daily(name, b) for b in books}).dropna()
        res[name] = (pbo_cscv(m.to_numpy(), n_splits=16), m.shape)

    fig = figure(figsize=(8.0, 6.6))[0]
    fig.clear()
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.35))
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    for ax in (ax0, ax1, ax2):
        style_axes(ax)

    # (a) six of the C(16, 8) = 12,870 IS/OOS assignments of the 16 blocks
    s = 16
    combos = list(combinations(range(s), s // 2))
    pick = [0, 1, 2, len(combos) // 2, len(combos) - 2, len(combos) - 1]
    for r, ci in enumerate(pick):
        is_set = set(combos[ci])
        for b in range(s):
            ax0.add_patch(_rect(b, len(pick) - 1 - r, blue if b in is_set else orange, pal))
        _ink(ax0, s + 0.2, len(pick) - 1 - r + 0.4, f"#{ci + 1:,}", pal, size=7.5, color="text_muted")
    ax0.set_xlim(-0.1, s + 1.6)
    ax0.set_ylim(-0.25, len(pick) + 0.05)
    ax0.set_xticks(np.arange(s) + 0.45, [str(b + 1) for b in range(s)])
    ax0.set_yticks([])
    ax0.tick_params(axis="x", labelsize=7.5, length=0)
    ax0.set_xlabel("block of consecutive days (16 equal blocks); 6 of the 12,870 combinations shown")
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.grid(False)
    ax0.set_title("CSCV: every way to call 8 of 16 blocks in-sample (blue), the rest out-of-sample (orange)",
                  fontsize=10.0)

    # (b) logits: discrete, one value per possible OOS rank of the IS winner
    for i, name in enumerate((SEL_DEFAULT, SEL_TREND)):
        r, shape = res[name]
        vals, counts = np.unique(np.round(r.logits, 6), return_counts=True)
        lab = f"{name.split('/')[-1].replace('_selection', '')}: {shape[1]} books, PBO {r.pbo:.2f}"
        ax1.bar(vals + (-0.06 if i == 0 else 0.06), counts / counts.sum(), width=0.11, color=cats[i], label=lab)
    ax1.axvline(0.0, color=pal["text_secondary"], linewidth=0.9)
    percent_axis(ax1)
    ax1.set_xlabel("logit of the IS winner's OOS rank (0 = OOS median)")
    ax1.set_ylabel("share of the 12,870 combinations")
    ax1.set_title("PBO = share of logits at or below 0", fontsize=10.0)
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(0, ymax * 1.45)
    add_legend(ax1, loc="upper right", fontsize=7.5)

    # (c) degradation: OOS vs IS Sharpe of the selected book (trend_core)
    r, _ = res[SEL_TREND]
    sc = math.sqrt(DAYS)
    ax2.scatter(r.is_perf * sc, r.oos_perf * sc, s=3, color=blue, alpha=0.25, linewidths=0)
    xs = np.linspace(np.nanmin(r.is_perf), np.nanmax(r.is_perf), 50)
    ax2.plot(xs * sc, (r.degradation_intercept + r.degradation_slope * xs) * sc, color=orange)
    zero_line(ax2)
    _ink(ax2, 0.98, 0.95, f"slope {signed(r.degradation_slope)}\nP(OOS loss) {r.prob_oos_loss:.2f}", pal,
         ha="right", va="top", size=8, color="text", xycoords="axes fraction")
    ax2.set_xlabel("IS Sharpe of the IS winner (annualised)")
    ax2.set_ylabel("its OOS Sharpe (annualised)")
    ax2.set_title("trend_core: winners degrade OOS", fontsize=10.0)
    style_axes(ax2, grid="both")
    source_note(fig, "Data: daily OOS returns of the strategy books in runs/protocol/*_selection, 2015-12 to "
                "2024-12; aurum.research.stats.pbo_cscv", pal)
    return fig


def _rect(x: float, y: float, color: str, pal):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), 0.9, 0.8, facecolor=color, edgecolor=pal["surface"], linewidth=0)


