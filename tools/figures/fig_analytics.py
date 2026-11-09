"""Figures for docs/analytics.md: parameter sweeps, Monte Carlo stress and explainability.

Real figures read run outputs that already exist (nothing here evaluates a strategy):

* ``runs/v21_demo/sweep_ema_cross`` - the EXPLORATORY ema_cross sweep (H1 2016-2024, OOS 2019-2024);
* ``runs/protocol/trend_core_selection`` - the research part of the trend_core walk-forward
  (OOS 2015-12-16 to 2024-12-31, no holdout inside), read through the cut-at-2024 loaders.

The Monte Carlo, decision and attribution figures call the same functions as ``aurum stress``
and ``aurum explain`` (seed 0, 10,000 paths, 2 years) and draw with the library's own plotting
functions in the mode being rendered. Concept figures are labelled synthetic.

Render with:  python tools/figures/build.py --only analytics
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from _common import (
    CUTOFF_END,
    FIGSIZE_TALL,
    bars,
    ink_on,
    panels,
    register,
    run_dir,
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
    percent_axis,
    shade_span,
    style_axes,
    zero_line,
)

PAGE = "analytics"
SWEEP = "v21_demo/sweep_ema_cross"
RUN = "protocol/trend_core_selection"
EXPLAIN_AT = "2017-12-26 19:00"
SYNTH = "Data: synthetic illustration (aurum.data.synthetic, seed 7 / numpy seeds in the figure code), not market data."
MINUS = "\N{MINUS SIGN}"


def _m(text: str) -> str:
    """ASCII hyphen-minus in numbers -> the true minus sign used by the tick labels."""
    return text.replace("-", MINUS)


# ---------------------------------------------------------------------------------------
# shared loads (inside draw only; cached so the figures of the page share one read)
# ---------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _sweep() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary = run_json(SWEEP, "sweep.json")
    table = run_table(SWEEP, "sweep.csv", cut=False)        # one row per grid point, no dates
    returns = run_table(SWEEP, "returns.csv")               # daily OOS returns, cut at 2024-12-31
    end = pd.Timestamp(summary["window"]["end"])
    if end >= CUTOFF_END:
        raise ValueError(f"{SWEEP} reaches {end}: past the figure cutoff")
    return summary, table, returns


@lru_cache(maxsize=1)
def _book_equity() -> pd.Series:
    ts = run_table(RUN, "books/combined/timeseries.parquet")
    return ts["equity"].astype(float)


@lru_cache(maxsize=1)
def _mc():
    """The Monte Carlo of ``aurum stress --run runs/protocol/trend_core_selection`` (defaults)."""
    from aurum.backtest.metrics import daily_returns
    from aurum.research.montecarlo import monte_carlo

    dr = daily_returns(_book_equity()).dropna()
    return monte_carlo(dr, n_paths=10_000, horizon="2Y", seed=0)


@lru_cache(maxsize=1)
def _stress():
    """``stress_run`` with the cut H1 bars (price, peak exposure and gold moves)."""
    from aurum.research.montecarlo import stress_run

    d = run_dir(RUN)
    if _book_equity().index.max() >= CUTOFF_END:
        raise ValueError(f"{RUN} reaches past the figure cutoff")
    return stress_run(d, bars=bars("H1"), n_paths=10_000, horizon="2Y", seed=0)


@lru_cache(maxsize=1)
def _attribution():
    from aurum.research.attribution import pnl_attribution

    att = pnl_attribution(run_dir(RUN))
    if att.per_bar.index.max() >= CUTOFF_END:
        raise ValueError(f"{RUN} attribution reaches past the figure cutoff")
    return att


def _run_credit() -> str:
    """e.g. 'Data: runs/protocol/trend_core_selection, walk-forward OOS 2015-12-16 to 2024-12-31 (18 folds), XAUUSD H1.'"""
    return "Data: " + run_label(RUN) + ", XAUUSD H1."


def _sweep_credit(summary: dict) -> str:
    w = summary["window"]
    return (f"Data: runs/{SWEEP} (EXPLORATORY), XAUUSD H1 {w['start'][:4]}-{w['end'][:4]}, walk-forward OOS "
            f"{w['oos_start'][:10]} to {w['oos_end'][:10]}.")


# =======================================================================================
# A. parameter sweeps
# =======================================================================================
def _toy_grid(kind: str) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    grid = {"fast": [4, 8, 16, 32, 64], "slow": [32, 64, 128, 256, 512]}
    i, j = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    if kind == "plateau":
        rng = np.random.default_rng(21)
        z = 0.62 * np.exp(-((i - 2.0) ** 2 + (j - 2.0) ** 2) / 8.0) + 0.015 * rng.standard_normal(i.shape)
        z[2, 2] = 0.62
    else:
        rng = np.random.default_rng(5)
        z = 0.08 + 0.07 * rng.standard_normal(i.shape)
        z[3, 1] = 0.62
    rows = [{"fast": grid["fast"][a], "slow": grid["slow"][b], "status": "ok", "sharpe": float(z[a, b])}
            for a in range(5) for b in range(5)]
    return pd.DataFrame(rows), grid


@register(PAGE, "plateau-vs-knife-edge")
def plateau_vs_knife_edge(pal):
    """Synthetic: two 5x5 Sharpe grids with the same best value; plateau ratio from plateau_score."""
    from matplotlib.patches import Circle, Rectangle

    from aurum.research.sweep import plateau_score

    fig, axes = figure(1, 2, figsize=(8.0, 4.3))
    cmap = diverging_cmap(pal["mode"])
    tables = {k: _toy_grid(k) for k in ("plateau", "knife")}
    lim = max(float(t["sharpe"].abs().max()) for t, _ in tables.values())
    im = None
    for ax, (table, grid) in zip(axes, tables.values(), strict=True):
        style_axes(ax, grid=None)
        z = table.pivot(index="fast", columns="slow", values="sharpe").reindex(
            index=grid["fast"], columns=grid["slow"]).to_numpy()
        im = ax.imshow(z, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", aspect="equal")
        for a in range(5):
            for b in range(5):
                label = f"{z[a, b]:.2f}"
                label = "0.00" if label == "-0.00" else label.replace("-", MINUS)
                ax.text(b, a, label, ha="center", va="center", fontsize=7.5,
                        color=ink_on(im.cmap(im.norm(z[a, b]))))
        p = plateau_score(table, grid, "sharpe")
        bi, bj = grid["fast"].index(p["best_params"]["fast"]), grid["slow"].index(p["best_params"]["slow"])
        ri, rj = (grid["fast"].index(p["robust_best_params"]["fast"]),
                  grid["slow"].index(p["robust_best_params"]["slow"]))
        x0, x1 = max(bj - 1, 0) - 0.5, min(bj + 1, 4) + 0.5
        y0, y1 = max(bi - 1, 0) - 0.5, min(bi + 1, 4) + 0.5
        ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.04), x1 - x0 - 0.08, y1 - y0 - 0.08, fill=False,
                               edgecolor=pal["text_secondary"], linewidth=1.5, linestyle=(0, (4, 3))))
        edge = ink_on(im.cmap(im.norm(z[bi, bj])))
        ax.add_patch(Rectangle((bj - 0.44, bi - 0.44), 0.88, 0.88, fill=False, edgecolor=edge, linewidth=2.0))
        ax.add_patch(Circle((rj, ri), 0.36, fill=False, edgecolor=ink_on(im.cmap(im.norm(z[ri, rj]))),
                            linewidth=1.3, linestyle="--"))
        ax.set_xticks(range(5), [str(v) for v in grid["slow"]])
        ax.set_yticks(range(5), [str(v) for v in grid["fast"]])
        ax.tick_params(axis="both", length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("slow window (bars)")
        ax.set_ylabel("fast window (bars)")
        verdict = str(p["verdict"]).replace("-", " ")
        ax.set_title(f"{verdict[:1].upper() + verdict[1:]}: plateau ratio {p['plateau_ratio']:.2f}\n"
                     f"(mean of the {p['n_neighbours']} neighbours / best)", fontsize=10.5)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.75)
    cb.set_label("OOS Sharpe (annualised)", color=pal["text_secondary"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])
    fig.suptitle("Same best Sharpe (0.62), different neighbourhoods", x=0.01, ha="left", color=pal["text"])
    source_note(fig, "box = best, dashed outline = its grid neighbours, dashed circle = plateau centre. "
                + SYNTH.replace(" (aurum.data.synthetic, seed 7 / numpy seeds in the figure code)", ""), pal)
    return fig


@register(PAGE, "ema-sweep-heatmap", needs=(f"runs/{SWEEP}",))
def ema_sweep_heatmap(pal):
    """The real ema_cross sweep heatmap, drawn by aurum.research.sweep.plot_sweep in this mode."""
    from aurum.research.sweep import plot_sweep

    s, table, _ = _sweep()
    p, b = s["plateau"], s["best"]
    sub = (f"box = best ({p['best']}), dashed circle = plateau centre ({p['robust_best']}), dashed outline = "
           f"grid neighbours. Plateau ratio {p['plateau_ratio']:.2f}: {p['verdict']}; DSR {b['dsr']:.2f} at "
           f"n_trials {s['n_trials']}; fast=64, slow=64 is invalid (fast must be below slow).")
    fig = plot_sweep(table, s["grid"], "sharpe", maximize=True, best=p["best_params"],
                     robust=p["robust_best_params"], title="ema_cross sweep: walk-forward OOS Sharpe (EXPLORATORY)",
                     subtitle=sub, mode=pal["mode"])
    ax, cax = fig.axes[0], fig.axes[1]
    ax.set_xlabel("slow EMA span (H1 bars)")
    ax.set_ylabel("fast EMA span (H1 bars)")
    cax.set_ylabel("OOS Sharpe (annualised)", color=pal["text_secondary"])
    source_note(fig, _sweep_credit(s), pal)
    return fig


@register(PAGE, "dsr-vs-trials", needs=(f"runs/{SWEEP}",))
def dsr_vs_trials(pal):
    """Expected best-of-N Sharpe (SR0) and the DSR of the sweep's best point as N grows."""
    from aurum.research.stats import deflated_sharpe, expected_max_sharpe, sharpe_summary

    s, _, returns = _sweep()
    best = s["best"]["label"]
    x = returns[best].dropna().to_numpy(float)
    t_len = x.size
    st = sharpe_summary(x, 252.0, n_trials=s["n_trials"], n_boot=200, seed=0)
    srp, skew, kurt = st["sharpe_per_period"], st["skew"], st["kurtosis"]
    ns = np.unique(np.round(np.logspace(0, math.log10(300), 80)).astype(int))
    sr0 = np.array([expected_max_sharpe(int(n), 1.0 / (t_len - 1)) for n in ns]) * math.sqrt(252.0)
    dsr = np.array([deflated_sharpe(srp, t_len, skew, kurt, n_trials=int(n)) for n in ns])
    n_sweep = int(s["n_trials"])
    dsr_sweep = deflated_sharpe(srp, t_len, skew, kurt, n_trials=n_sweep)
    sr0_sweep = expected_max_sharpe(n_sweep, 1.0 / (t_len - 1)) * math.sqrt(252.0)

    fig, (ax0, ax1) = panels(2, heights=(1, 1), figsize=FIGSIZE_TALL)
    blue = pal["categorical"][0]
    for ax in (ax0, ax1):
        style_axes(ax)
        ax.set_xscale("log")
        ax.axvline(n_sweep, color=pal["axis"], linewidth=0.9, zorder=0.5)
    ax0.plot(ns, sr0, color=blue)
    ax0.axhline(st["sharpe"], color=pal["text_secondary"], linewidth=1.0, linestyle="--")
    ax0.annotate(f"observed best Sharpe {st['sharpe']:.2f} ({best})", xy=(ns[-1], st["sharpe"]), xytext=(-4, 5),
                 textcoords="offset points", ha="right", fontsize=8.5, color=pal["text_secondary"])
    ax0.plot([n_sweep], [sr0_sweep], marker="o", color=blue, linestyle="none")
    ax0.annotate(f"N = {n_sweep} (this sweep): SR0 = {sr0_sweep:.2f}", xy=(n_sweep, sr0_sweep), xytext=(8, -14),
                 textcoords="offset points", fontsize=8.5, color=pal["text"])
    ax0.set_ylim(0.0, float(sr0.max()) * 1.12)
    ax0.set_ylabel("Sharpe (annualised)")
    ax0.set_title("SR0: the best Sharpe expected from N skill-less trials")
    ax1.plot(ns, dsr, color=blue)
    ax1.axhline(0.95, color=pal["text_secondary"], linewidth=1.0, linestyle="--")
    ax1.annotate("0.95: the protocol's pass line", xy=(ns[-1], 0.95), xytext=(-4, 5), textcoords="offset points",
                 ha="right", fontsize=8.5, color=pal["text_secondary"])
    ax1.plot([1, n_sweep], [dsr[0], dsr_sweep], marker="o", color=blue, linestyle="none")
    ax1.annotate(f"N = 1: PSR {dsr[0]:.2f}", xy=(1, dsr[0]), xytext=(8, -4), textcoords="offset points",
                 fontsize=8.5, color=pal["text"])
    ax1.annotate(f"N = {n_sweep}: DSR {dsr_sweep:.2f}", xy=(n_sweep, dsr_sweep), xytext=(8, 4),
                 textcoords="offset points", fontsize=8.5, color=pal["text"])
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("probability")
    ax1.set_title("DSR = PSR(SR0) of the best point")
    ax1.set_xlabel("number of trials N (log scale)")
    ax1.xaxis.set_major_formatter(_plain_log_formatter())
    source_note(fig, f"T = {t_len:,} daily OOS returns of {best}, V = 1/(T-1). Data: runs/{SWEEP} "
                "(EXPLORATORY), XAUUSD H1, OOS 2019-2024.", pal)
    return fig


def _plain_log_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: f"{v:g}")


@register(PAGE, "sweep-pbo", needs=(f"runs/{SWEEP}",))
def sweep_pbo(pal):
    """PBO by CSCV across the 8 valid grid points: logit histogram and IS vs OOS Sharpe."""
    from aurum.research.stats import pbo_cscv

    s, _, returns = _sweep()
    m = returns.dropna()
    n_splits = int(s["pbo"]["n_splits"])
    res = pbo_cscv(m.to_numpy(), n_splits=n_splits)
    fig, (ax0, ax1) = figure(1, 2, figsize=(8.0, 3.9))
    blue = pal["categorical"][0]
    style_axes(ax0)
    n_cfg = res.n_strategies
    ranks = np.round(res.oos_rank * (n_cfg + 1), 6)            # w = rank / (N + 1), rank N = best
    vals, counts = np.unique(ranks, return_counts=True)
    low = vals <= (n_cfg + 1) / 2.0                             # logit(w) <= 0
    ax0.bar(vals[low], counts[low], width=0.62, color=pal["categorical"][7], linewidth=0,
            label="at or below the OOS median (logit <= 0)")
    ax0.bar(vals[~low], counts[~low], width=0.62, color=blue, linewidth=0, label="above the median")
    ax0.axvline((n_cfg + 1) / 2.0, color=pal["text"], linewidth=1.0)
    ax0.annotate(f"PBO = {res.pbo:.2f}\n(share of red)", xy=((n_cfg + 1) / 2.0, 1.0),
                 xycoords=("data", "axes fraction"), xytext=(5, -4), textcoords="offset points", va="top",
                 fontsize=9, color=pal["text"])
    add_legend(ax0, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=1)
    ax0.set_xticks(range(1, n_cfg + 1), ["1\nworst"] + [str(k) for k in range(2, n_cfg)] + [f"{n_cfg}\nbest"])
    ax0.set_xlabel(f"OOS rank of the IS winner among {n_cfg} grid points")
    ax0.set_ylabel("IS/OOS splits")
    ax0.set_title(f"{res.n_combinations:,} CSCV splits", pad=34)
    style_axes(ax1, grid="both")
    ann = math.sqrt(252.0)
    ax1.scatter(res.is_perf * ann, res.oos_perf * ann, s=5, color=blue, alpha=0.18, linewidths=0)
    xs = np.linspace(float(res.is_perf.min()), float(res.is_perf.max()), 50)
    ax1.plot(xs * ann, (res.degradation_intercept + res.degradation_slope * xs) * ann, color=pal["text"],
             linewidth=1.3)
    ax1.annotate(_m(f"OLS slope {res.degradation_slope:.2f}"), xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9, color=pal["text"])
    zero_line(ax1)
    zero_line(ax1, axis="x")
    ax1.set_xlabel("IS Sharpe of the IS winner (annualised)")
    ax1.set_ylabel("its OOS Sharpe (annualised)")
    ax1.set_title("Better in sample, worse out of sample")
    fig.suptitle(f"Probability of backtest overfitting across {res.n_strategies} grid points", x=0.01, ha="left",
                 color=pal["text"])
    source_note(fig, f"S = {n_splits} blocks of {m.shape[0]:,} daily returns. " + _sweep_credit(s), pal)
    return fig


# =======================================================================================
# B. Monte Carlo and stress
# =======================================================================================
@register(PAGE, "block-bootstrap")
def block_bootstrap(pal):
    """Synthetic: the stationary bootstrap's index map and its geometric block lengths."""
    from aurum.research.montecarlo import stationary_bootstrap_paths

    b = 6.0
    n_hist, horizon = 40, 40
    idx = stationary_bootstrap_paths(n_hist, 2, horizon, b, np.random.default_rng(11))
    fig, (ax0, ax1) = figure(1, 2, figsize=(8.0, 3.9))
    style_axes(ax0, grid="both")
    t = np.arange(horizon)
    wrap_at = None
    for k, colour in zip(range(idx.shape[0]), pal["categorical"][:2], strict=True):
        row = idx[k]
        starts = [0] + [s for s in range(1, horizon) if row[s] != (row[s - 1] + 1) % n_hist]
        bounds = starts + [horizon]
        for a, z in zip(bounds[:-1], bounds[1:], strict=True):
            seg_t, seg = t[a:z], row[a:z]
            wraps = np.flatnonzero(np.diff(seg) < 0) + 1        # circular wrap inside a block
            if wraps.size and wrap_at is None:
                wrap_at = (int(seg_t[wraps[0]]), int(seg[wraps[0]]))
            for piece_t, piece in zip(np.split(seg_t, wraps), np.split(seg, wraps), strict=True):
                ax0.plot(piece_t, piece, color=colour, linewidth=1.8, solid_capstyle="round")
        ax0.plot(t[starts], row[starts], marker="o", linestyle="none", color=colour,
                 label=f"path {k + 1}: {len(starts)} blocks")
    if wrap_at is not None:
        ax0.annotate(f"wraps from {n_hist - 1} to 0:\nsame block", xy=wrap_at,
                     xytext=(wrap_at[0] + 3, wrap_at[1] + 8), fontsize=8, color=pal["text_secondary"],
                     arrowprops={"arrowstyle": "->", "color": pal["text_secondary"], "linewidth": 0.8})
    ax0.set_xlim(-1, horizon)
    ax0.set_ylim(-1, n_hist + 8)
    ax0.set_yticks(range(0, n_hist + 1, 10))
    ax0.set_xlabel("simulated day t (dot = a new block starts)")
    ax0.set_ylabel("historical day copied")
    ax0.set_title(f"Index map (n = {n_hist}, mean block b = {b:g})")
    add_legend(ax0, loc="upper left", fontsize=8, ncol=2)
    style_axes(ax1)
    big = stationary_bootstrap_paths(10_000, 2_000, 400, b, np.random.default_rng(12))
    jumps = big[:, 1:] != (big[:, :-1] + 1) % 10_000
    lengths = []
    for row in jumps:
        pos = np.flatnonzero(row) + 1
        if pos.size >= 2:
            lengths.append(np.diff(pos))                       # complete blocks only
    lengths = np.concatenate(lengths)
    ks = np.arange(1, 26)
    emp = np.array([(lengths == k).mean() for k in ks])
    p = 1.0 / b
    ax1.bar(ks, emp, color=pal["categorical"][0], width=0.8, linewidth=0, label="simulated block lengths")
    ax1.plot(ks, p * (1 - p) ** (ks - 1), color=pal["text"], marker="o", markersize=4, linewidth=1.2,
             label="Geometric(p = 1/b)")
    ax1.set_xlabel("block length (days)")
    ax1.set_ylabel("share of blocks")
    ax1.set_title(f"Block lengths: mean {lengths.mean():.2f} days")
    add_legend(ax1, loc="upper right", fontsize=8)
    fig.suptitle("Stationary bootstrap (Politis-Romano): copy runs of history, jump at random", x=0.01, ha="left",
                 color=pal["text"])
    source_note(fig, "Indices from aurum.research.montecarlo.stationary_bootstrap_paths (numpy seeds 11, 12). "
                "Synthetic illustration, not market data.", pal)
    return fig


@register(PAGE, "fan-chart", needs=(f"runs/{RUN}",))
def fan_chart(pal):
    """aurum.research.montecarlo.plot_fan on the trend_core combined book (10,000 paths, 2 years)."""
    from aurum.research.montecarlo import plot_fan

    mc = _mc()
    fig = plot_fan(mc, mode=pal["mode"], title=f"trend_core combined book: {mc.n_paths:,} bootstrap paths of 2 "
                   f"years (mean block {mc.mean_block:.1f} day)")
    source_note(fig, _run_credit(), pal)
    return fig


@register(PAGE, "mc-distributions", needs=(f"runs/{RUN}",))
def mc_distributions(pal):
    """aurum.research.montecarlo.plot_distributions on the same simulation."""
    from aurum.research.montecarlo import plot_distributions

    mc = _mc()
    fig = plot_distributions(mc, mode=pal["mode"])
    source_note(fig, f"Seed {mc.seed}. " + _run_credit(), pal)
    return fig


def _longest_underwater(dd: np.ndarray) -> tuple[int, int]:
    """(start, end) of the longest run of dd < 0 (end exclusive)."""
    best, cur = (0, 0), None
    for i, v in enumerate(dd):
        if v < -1e-12:
            cur = i if cur is None else cur
            if i + 1 - cur > best[1] - best[0]:
                best = (cur, i + 1)
        else:
            cur = None
    return best


@register(PAGE, "equity-path-anatomy", needs=(f"runs/{RUN}",))
def equity_path_anatomy(pal):
    """One simulated path with its peak, trough, recovery, underwater stretch and ruin line."""
    mc = _mc()
    m = mc.metrics
    # the first path that recovers from a deep drawdown that also took it close to the ruin line
    ok = (m["recovered"].astype(bool) & m["time_to_recover_days"].between(60, 250) & (m["min_equity"] < 0.92)
          & (m["max_drawdown"] < -0.08))
    k = int(np.flatnonzero(ok.to_numpy())[0])
    eq = mc.paths[k]
    row = m.iloc[k]
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    trough = int(row["trough_day"])
    pk = int(np.argmax(eq[:trough + 1]))
    rec = trough + int(row["time_to_recover_days"])
    ruin_level = 0.90
    touch = int(np.flatnonzero(eq <= ruin_level)[0]) if (eq <= ruin_level).any() else None
    u0, u1 = _longest_underwater(dd)
    t = np.arange(eq.size)

    fig, (ax0, ax1) = panels(2, heights=(3, 2), figsize=FIGSIZE_TALL)
    blue = pal["categorical"][0]
    style_axes(ax0)
    style_axes(ax1)
    shade_span(ax0, pk, rec, "max-drawdown episode: peak to new peak", alpha=0.6)
    shade_span(ax1, pk, rec, alpha=0.6)
    ax0.plot(t, peak, color=pal["text_muted"], linewidth=1.0, linestyle="--")
    mid = (pk + rec) // 2
    ax0.annotate("running peak", xy=(mid, peak[mid]), xytext=(0, 4), textcoords="offset points", ha="center",
                 fontsize=8, color=pal["text_muted"])
    ax0.plot(t, eq, color=blue)
    ax0.axhline(ruin_level, color=pal["text_secondary"], linewidth=1.0, linestyle=":")
    ax0.annotate("ruin line (1 \N{MINUS SIGN} L) E0, L = 10%", xy=(t[-1], ruin_level), xytext=(-4, 4),
                 textcoords="offset points", ha="right", fontsize=8, color=pal["text_secondary"])
    ax0.axhline(1.0, color=pal["axis"], linewidth=0.8)
    dot = {"marker": "o", "linestyle": "none", "color": pal["text"], "markersize": 6}
    ax0.plot([pk], [eq[pk]], **dot)
    ax0.annotate(f"peak (day {pk})", xy=(pk, eq[pk]), xytext=(-6, 8), textcoords="offset points", ha="right",
                 fontsize=8.5, color=pal["text"])
    ax0.plot([trough], [eq[trough]], **dot)
    ax0.annotate(_m(f"trough (day {trough}): max drawdown {row['max_drawdown']:.1%}"), xy=(trough, eq[trough]),
                 xytext=(8, -4), textcoords="offset points", fontsize=8.5, color=pal["text"])
    ax0.plot([rec], [eq[rec]], **dot)
    ax0.annotate(f"new peak (day {rec}): recovered {rec - trough} days after the trough", xy=(rec, eq[rec]),
                 xytext=(-6, 9), textcoords="offset points", ha="right", fontsize=8.5, color=pal["text"])
    if touch is not None:
        ax0.plot([touch], [eq[touch]], marker="x", linestyle="none", color=pal["text"], markersize=7)
        ax0.annotate(f"ruin threshold touched (day {touch})", xy=(touch, eq[touch]), xytext=(-8, -12),
                     textcoords="offset points", ha="right", fontsize=8.5, color=pal["text"])
    ax0.set_ylim(ruin_level - 0.01, float(eq.max()) + 0.014)
    ax0.set_ylabel("equity (start = 1)")
    ax0.set_title(f"One simulated path (#{k}) of the trend_core book")
    ax1.fill_between(t, dd, 0.0, color=blue, alpha=0.22, linewidth=0)
    ax1.plot(t, dd, color=blue, linewidth=1.2)
    y_arrow = float(dd.min()) * 1.12
    ax1.annotate("", xy=(u0, y_arrow), xytext=(u1, y_arrow),
                 arrowprops={"arrowstyle": "<->", "color": pal["text_secondary"], "linewidth": 1.0})
    ax1.annotate(f"longest underwater stretch: {u1 - u0} days", xy=(0.5 * (u0 + u1), y_arrow), xytext=(0, -11),
                 textcoords="offset points", ha="center", fontsize=8.5, color=pal["text_secondary"])
    ax1.set_ylim(float(dd.min()) * 1.45, 0.005)
    percent_axis(ax1)
    ax1.set_ylabel("drawdown from peak")
    ax1.set_xlabel("trading day of the simulated horizon")
    ax1.set_xlim(0, t[-1])
    source_note(fig, f"Seed {mc.seed}. " + _run_credit(), pal)
    return fig


@register(PAGE, "deterministic-stress", needs=(f"runs/{RUN}", "data_store"))
def deterministic_stress(pal):
    """Worst historical k-day windows of the book and gap shocks at its peak exposure."""
    rep = _stress()
    ww = rep.worst_windows
    gaps = rep.gap_shocks
    pk = rep.exposure.get("peak") or {}
    fig, (ax0, ax1) = figure(1, 2, figsize=(8.0, 4.2))
    red, blue = pal["categorical"][7], pal["categorical"][0]
    style_axes(ax0, grid="x")
    ys = np.arange(len(ww))
    vals = ww["worst_return"].to_numpy(float)
    ax0.barh(ys, vals, color=red, height=0.6)
    ticks = []
    for y, v, d, a, z in zip(ys, vals, ww["days"], ww["start"], ww["end"], strict=True):
        ticks.append(f"{d} d\n{a}" if a == z else f"{d} d\n{a} to {z}")
        ax0.annotate(_m(f"{v:.1%}"), xy=(v, y), xytext=(-4, 0), textcoords="offset points", va="center",
                     ha="right", fontsize=8, color=pal["text_secondary"])
    ax0.set_yticks(ys, ticks, fontsize=7.5)
    ax0.set_xlim(float(vals.min()) * 1.45, 0.0)
    percent_axis(ax0, "x", decimals=0)
    ax0.invert_yaxis()
    ax0.set_xlabel("worst compounded return of the book")
    ax0.set_title("Worst k-day windows (book returns)")
    style_axes(ax1, grid="x")
    g = gaps.loc[gaps["exposure"].str.startswith("peak")].sort_values("shock_pct")
    names = [f"{_m(f'{p:+.0%}')} {s.split(' (')[0] if '(' in s else 'gap'}" for p, s in zip(g["shock_pct"],
                                                                                            g["scenario"], strict=True)]
    yv = np.arange(len(g))
    pv = g["pnl_pct_equity"].to_numpy(float)
    ax1.barh(yv, pv, color=[blue if v >= 0 else red for v in pv], height=0.6)
    ax1.set_yticks(yv, names, fontsize=8)
    lim = float(np.abs(pv).max()) * 1.6
    ax1.set_xlim(-lim, lim)
    for y, v in zip(yv, pv, strict=True):
        ax1.annotate(_m(f"{v:+.1%}"), xy=(v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                     va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=pal["text_secondary"])
    ax1.axvline(0.0, color=pal["text_secondary"], linewidth=0.8)
    percent_axis(ax1, "x", decimals=0)
    ax1.set_xlabel("PnL in % of equity at that time")
    ax1.set_title(f"Gap shocks at peak exposure\n{pk.get('time', '')[:10]}: {pk.get('lots', 0):+.2f} lots, "
                  f"leverage {pk.get('leverage', 0):.2f}", fontsize=10)
    fig.suptitle("Deterministic stress of the trend_core combined book", x=0.01, ha="left", color=pal["text"])
    source_note(fig, _run_credit(), pal)
    return fig


# =======================================================================================
# C. explainability
# =======================================================================================
@register(PAGE, "waterfall-research", needs=(f"runs/{RUN}", "data_store"))
def waterfall_research(pal):
    """aurum.research.attribution.plot_waterfall for one real research decision (with a clip)."""
    from aurum.research.attribution import explain_decision, plot_waterfall

    ex = explain_decision(run_dir(RUN), at=EXPLAIN_AT, md=bars("H1"))
    fig = plot_waterfall(ex, mode=pal["mode"])
    source_note(fig, _run_credit(), pal)
    return fig


def _live_record() -> dict:
    """A decisions.jsonl record in the live runner's format (synthetic numbers; the sizing block
    is the real VolTargetSizer.breakdown of those numbers)."""
    from aurum.core.instrument import XAUUSD
    from aurum.portfolio.sizing import VolTargetSizer

    weights = {"tsmom": 0.35, "ema_cross": 0.30, "donchian": 0.25, "risk_off": 0.10}
    forecasts = {"tsmom": 0.90, "ema_cross": 1.00, "donchian": 0.80, "risk_off": -0.50}
    fdm = 1.45
    strategies = {k: {"forecast": forecasts[k], "weight": w, "contribution": w * forecasts[k] * fdm}
                  for k, w in weights.items()}
    raw = sum(v["contribution"] for v in strategies.values())
    combined = float(np.clip(raw, -1.0, 1.0))
    final = 0.5 * combined
    bd = VolTargetSizer().breakdown(final, 0.14, 100_000.0, 2_350.0, XAUUSD, current_lots=0.10, drawdown=0.02)
    return {
        "type": "decision", "time": "2024-06-03T15:00:05+00:00", "bar_time": "2024-06-03T14:00:00+00:00",
        "mode": "paper", "status": "ok", "combined": combined, "final_forecast": final, "current_lots": 0.10,
        "requested_lots": bd.final_lots, "approved_lots": bd.final_lots, "equity": 100_000.0,
        "desk": {"status": "decided", "action": "scale", "quant_forecast": combined, "final_forecast": final},
        "sizing": {"forecast": bd.forecast, "vol_ann": bd.vol_ann, "equity": bd.equity, "price": bd.price,
                   "drawdown": bd.drawdown, "raw_notional": bd.raw_notional, "capped_notional": bd.capped_notional,
                   "drawdown_mult": bd.drawdown_mult, "raw_lots": bd.raw_lots, "rounded_lots": bd.rounded_lots,
                   "final_lots": bd.final_lots, "kept_current": bd.kept_current, "caps_hit": bd.caps_hit,
                   "note": bd.note},
        "attribution": {"method": "sharpe_shrink", "fdm": fdm, "strategies": strategies, "raw_combined": raw,
                        "combined": combined, "clipped": abs(raw) > 1.0, "unallocated": 1.0 - sum(weights.values()),
                        "exact": True, "final_forecast": final, "desk_delta": final - combined},
    }


@register(PAGE, "waterfall-live-desk")
def waterfall_live_desk(pal):
    """Synthetic live/paper record -> explain_live_decision -> plot_waterfall with the desk step."""
    from aurum.research.attribution import explain_live_decision, plot_waterfall

    ex = explain_live_decision([_live_record()])
    fig = plot_waterfall(ex, mode=pal["mode"])
    source_note(fig, "A decisions.jsonl record in the runner's format with synthetic numbers. Synthetic "
                "illustration, not market data.", pal)
    return fig


@register(PAGE, "pnl-attribution", needs=(f"runs/{RUN}",))
def pnl_attribution_figure(pal):
    """aurum.research.attribution.plot_attribution for the trend_core combined book."""
    from aurum.research.attribution import plot_attribution

    fig = plot_attribution(_attribution(), mode=pal["mode"])
    source_note(fig, _run_credit(), pal)
    return fig


@register(PAGE, "attribution-components", needs=(f"runs/{RUN}",))
def attribution_components(pal):
    """Per-strategy attributed price PnL, costs, financing and net as small multiples."""
    att = _attribution()
    bc = att.by_component.drop(index="unattributed", errors="ignore")
    bc = bc.loc[bc["net"].sort_values().index]
    comps = [("price", "price PnL"), ("costs", "costs"), ("swap", "financing (swap)"), ("net", "net")]
    fig, axes = figure(1, 4, figsize=(8.0, 3.9), sharey=True)
    red, blue = pal["categorical"][7], pal["categorical"][0]
    lim = float(bc.abs().to_numpy().max()) * 1.55
    ys = np.arange(len(bc))
    for ax, (col, label) in zip(axes, comps, strict=True):
        style_axes(ax, grid="x")
        v = bc[col].to_numpy(float)
        ax.barh(ys, v, color=[blue if x >= 0 else red for x in v], height=0.62)
        ax.axvline(0.0, color=pal["text_secondary"], linewidth=0.8)
        for y, x in zip(ys, v, strict=True):
            ax.annotate(signed(x / 1000, "{:+.1f}k") if abs(x) >= 1000 else signed(x, "{:+.0f}"), xy=(x, y), xytext=(3 if x >= 0 else -3, 0),
                        textcoords="offset points", va="center", ha="left" if x >= 0 else "right", fontsize=7,
                        color=pal["text_secondary"])
        ax.set_xlim(-lim, lim)
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(_k_formatter())
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].set_yticks(ys, list(bc.index), fontsize=8.5)
    for ax in axes:
        ax.set_xlabel("USD")
    fig.suptitle("Attributed PnL by component: costs carry the same signed shares", x=0.01, ha="left",
                 color=pal["text"])
    source_note(fig, _run_credit(), pal)
    return fig


def _k_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: signed(v / 1000, "{:.0f}k") if v else "0")
