"""Figures for docs/portfolio-and-risk.md: the forecast combiner (sharpe_shrink, HRP, FDM), the
vol-target sizing chain, drawdowns, VaR/ES and stress tests.

Each ``@register``-ed ``draw(pal)`` becomes
docs/_assets/figures/portfolio-and-risk/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only portfolio-and-risk

Real-data figures read the walk-forward outputs in ``runs/protocol/trend_core_selection`` (never
its holdout) and the bars in ``data_store/``, all cut at 2024-12-31 by the ``_common`` loaders.
Combiner fits below are re-fits of aurum's own ``ForecastCombiner`` on those stored out-of-sample
forecasts; no walk-forward or backtest is run here.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_TALL,
    bars,
    ink_on,
    panels,
    register,
    run_file,
    run_label,
    run_table,
    signed,
    source_label,
    source_note,
)

from aurum.portfolio.combiner import ForecastCombiner, cap_weights
from aurum.portfolio.sizing import DEFAULT_DRAWDOWN_DERISK, VolTargetSizer, drawdown_multiplier
from aurum.viz.style import (
    add_legend,
    diverging_cmap,
    figure,
    label_line_end,
    percent_axis,
    sequential_cmap,
    style_axes,
    zero_line,
)

PAGE = "portfolio-and-risk"
RUN = "protocol/trend_core_selection"
RUN_NEED = f"runs/{RUN}"
STRATS = ("tsmom", "ema_cross", "donchian", "kalman_trend", "vol_squeeze", "macro_factor", "risk_off")


# ---------------------------------------------------------------------------------------
# helpers (called inside draw only)
# ---------------------------------------------------------------------------------------
def _blend(c1: str, c2: str, t: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple((1.0 - t) * a + t * b)


def _pct_formatter(decimals: int = 0):
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos):
        v = 0.0 if abs(v) < 1e-12 else v
        return f"{v * 100:.{decimals}f}%".replace("-", "\N{MINUS SIGN}")

    return FuncFormatter(fmt)


@functools.cache
def _run_config():
    from aurum.core.config import load_config

    return load_config(str(run_file(RUN, "config.yaml")), validate=False)


@functools.cache
def _oos_forecasts() -> pd.DataFrame:
    fc = run_table(RUN, "oos_forecasts.parquet")
    fc["fold"] = fc["fold"].astype(int)
    return fc


def _cost_kwargs(h1: pd.DataFrame, index: pd.Index) -> dict:
    cfg = _run_config()
    return {"bars": h1.loc[index], "costs": cfg.costs.build(), "instrument": cfg.instrument.build()}


def _combiner(method: str | None = None, **over) -> ForecastCombiner:
    c = _run_config().combiner
    kw = {k: getattr(c, k) for k in ("shrinkage", "max_weight", "fdm_cap", "vol_halflife", "min_periods",
                                      "corr_floor", "allow_unallocated", "cost_multiplier")}
    kw.update(over)
    return ForecastCombiner(method=method or c.method, **kw)


@functools.cache
def _fold_fit(fold: int) -> tuple[ForecastCombiner, pd.DataFrame]:
    """Re-fit the combiner exactly as the walk-forward did for ``fold`` (combiner_fit=oos):
    on the stitched OOS forecasts of earlier folds, the last ``walkforward.train`` bars."""
    fc = _oos_forecasts()
    window = int(pd.read_csv(run_file(RUN, "folds.csv"))["n_train"].iloc[fold])
    hist = fc.loc[fc["fold"] < fold, list(STRATS)].iloc[-window:]
    h1 = bars("H1")
    comb = _combiner().fit(hist, h1["close"].reindex(hist.index), **_cost_kwargs(h1, hist.index))
    return comb, hist


@functools.cache
def _hrp_inputs() -> dict:
    """Net unit-vol streams of the 7 trend_core strategies over the whole stored OOS period,
    and the HRP tree built from them exactly as aurum.portfolio.combiner._hrp_weights does."""
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    fc = _oos_forecasts()[list(STRATS)]
    h1 = bars("H1")
    close = h1["close"].reindex(fc.index)
    kw = _cost_kwargs(h1, fc.index)
    out = {}
    for m in ("hrp", "inverse_vol", "equal"):
        out[m] = _combiner(m, max_weight=1.0).fit(fc, close, **kw)
    hrp = out["hrp"]
    cost = hrp.turnover_cost(fc, close, **kw)
    u = hrp.unit_vol_streams(fc, close, cost_per_turnover=cost)
    cov = np.cov(u.to_numpy(), rowvar=False)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(0.5 * (dist + dist.T), checks=False), method="single")
    root = to_tree(link)

    def cluster_var(items):
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = np.ones(len(STRATS))
    splits = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            continue
        left, right = node.get_left(), node.get_right()
        c1, c2 = left.pre_order(), right.pre_order()
        v1, v2 = cluster_var(c1), cluster_var(c2)
        a = 1.0 - v1 / (v1 + v2)
        w[c1] *= a
        w[c2] *= 1.0 - a
        splits.append((node, left, right, a))
        stack.extend([left, right])
    w /= w.sum()
    assert np.allclose(w, hrp.weights_.to_numpy(), atol=1e-9), "HRP replication drifted from the combiner"
    return {"corr": corr, "root": root, "order": root.pre_order(), "splits": splits, "fits": out,
            "std": np.sqrt(np.diag(cov)), "period": (fc.index[0], fc.index[-1])}


def _book(name: str) -> pd.DataFrame:
    return run_table(RUN, f"books/{name}/timeseries.parquet")


# ---------------------------------------------------------------------------------------
# combiner
# ---------------------------------------------------------------------------------------
@register(PAGE, "combiner-fold-weights", needs=(RUN_NEED,))
def combiner_fold_weights(pal):
    """Per-fold sharpe_shrink weights (and the unallocated remainder) and the FDM, trend_core run."""
    w = pd.read_csv(run_file(RUN, "weights.csv"), index_col=0)
    folds = pd.read_csv(run_file(RUN, "folds.csv"))
    mat = w[list(STRATS)].T
    unalloc = (1.0 - mat.sum(axis=0)).clip(lower=0.0)
    full = pd.concat([mat, unalloc.rename("unallocated").to_frame().T])
    labels = [f"{t[:7]}" for t in folds["test_start"].astype(str)]

    fig, (ax0, ax1) = panels(2, heights=(4.2, 1.3), figsize=(8.0, 6.2), hspace=0.04)
    style_axes(ax0, grid=None)
    cmap = sequential_cmap(pal["mode"])
    im = ax0.imshow(full.to_numpy(), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            v = float(full.iat[i, j])
            if v < 0.005:
                txt = "0" if i < len(STRATS) else ""
            else:
                txt = f"{v * 100:.0f}"
            ax0.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color=ink_on(im.cmap(im.norm(v))))
    ax0.axhline(len(STRATS) - 0.5, color=pal["surface"], linewidth=3)
    ax0.set_yticks(range(full.shape[0]), list(full.index))
    ax0.set_title("Combiner weight per walk-forward fold (% of the risk budget)")
    ax0.tick_params(axis="x", length=0)
    cbar = fig.colorbar(im, ax=ax0, shrink=0.8, pad=0.01)
    cbar.set_label("weight", color=pal["text_secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(_pct_formatter())
    cbar.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])

    x = np.arange(len(w))
    ax1.plot(x, w["fdm"].to_numpy(), color=pal["categorical"][0], marker="o", markersize=4,
             markeredgecolor=pal["surface"])
    ax1.axhline(1.0, color=pal["axis"], linewidth=0.8)
    ax1.set_ylim(0.9, 1.8)
    ax1.set_ylabel("FDM")
    ax1.set_xticks(x, labels, rotation=90, fontsize=7.5)
    ax1.set_xlabel("fold (test block start, YYYY-MM)")
    cb1 = fig.colorbar(im, ax=ax1, shrink=0.0001, pad=0.01)   # keeps the panels aligned
    cb1.ax.set_visible(False)
    source_note(fig, "Data: " + run_label(RUN), pal)
    return fig


@register(PAGE, "sharpe-shrink-steps", needs=(RUN_NEED, "data_store"))
def sharpe_shrink_steps(pal):
    """sharpe_shrink step by step for fold 17: gross/net Sharpe, shrunk Sharpe, weights before/after cap."""
    comb, hist = _fold_fit(17)
    lam = comb.shrinkage
    net = comb.train_sharpe_.reindex(list(STRATS))
    gross = comb.train_sharpe_gross_.reindex(list(STRATS))
    sd = comb.train_vol_.reindex(list(STRATS))
    elig = net > 0
    target = float(net[elig].mean())
    shrunk = ((1 - lam) * net + lam * target).where(elig, 0.0)
    raw = (shrunk / sd).where(elig, 0.0)
    uncapped = raw / raw.sum()
    capped = pd.Series(cap_weights(raw[elig].to_numpy(), comb.max_weight, allow_unallocated=True),
                       index=raw[elig].index).reindex(list(STRATS)).fillna(0.0)
    assert np.allclose(capped.to_numpy(), comb.weights_.reindex(list(STRATS)).to_numpy(), atol=1e-9)

    y = np.arange(len(STRATS))[::-1]
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    muted = _blend(pal["grid"], pal["text_secondary"], 0.45)
    fig, axes = figure(1, 3, figsize=(8.0, 4.4), sharey=True)
    h = 0.38
    ax = axes[0]
    ax.barh(y + h / 2, gross.to_numpy(), height=h, color=muted, label="gross")
    ax.barh(y - h / 2, net.to_numpy(), height=h, color=blue, label="net of costs")
    zero_line(ax, axis="x")
    ax.set_yticks(y, list(STRATS))
    ax.set_title("1. Sharpe, gross vs net", fontsize=10)
    ax.set_xlabel("annualised Sharpe")
    add_legend(ax, loc="upper left", fontsize=8)

    ax = axes[1]
    ax.barh(y, shrunk.to_numpy(), height=0.6, color=blue)
    ax.axvline(target, color=pal["text_secondary"], linewidth=1.0, linestyle="--")
    ax.annotate(f"target {target:.2f} = mean of\nthe positive net SRs", (target, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -2), textcoords="offset points", ha="left",
                va="top", color=pal["text_secondary"], fontsize=7.5)
    for yi, s, e in zip(y, shrunk, elig, strict=True):
        ax.annotate(f"{s:.2f}" if e else "excluded", (max(s, 0), yi), xytext=(4, 0),
                    textcoords="offset points", va="center", color=pal["text"] if e else pal["text_muted"],
                    fontsize=7.5)
    ax.set_xlim(0, max(0.35, float(shrunk.max()) * 1.9))
    ax.set_title(f"2. shrunk SR, lambda = {lam:g}", fontsize=10)
    ax.set_xlabel("annualised Sharpe")

    ax = axes[2]
    ax.barh(y + h / 2, uncapped.to_numpy(), height=h, color=muted, label="SR / std, normalised")
    ax.barh(y - h / 2, capped.to_numpy(), height=h, color=orange, label=f"after cap {comb.max_weight:g}")
    ax.axvline(comb.max_weight, color=pal["text_secondary"], linewidth=1.0, linestyle="--")
    for yi, v in zip(y, capped, strict=True):
        if v > 0:
            ax.annotate(f"{v:.0%}", (v, yi - h / 2), xytext=(3, 0), textcoords="offset points", va="center",
                        color=pal["text"], fontsize=7.5)
    percent_axis(ax, axis="x")
    ax.set_xlim(0, 0.75)
    ax.set_title("3. weights", fontsize=10)
    ax.set_xlabel("share of risk budget")
    add_legend(ax, loc="lower right", fontsize=7.5)
    fig.suptitle(f"sharpe_shrink for fold 17: fitted on OOS forecasts {hist.index[0]:%Y-%m-%d} to "
                 f"{hist.index[-1]:%Y-%m-%d}, FDM {comb.fdm_:.2f}")
    source_note(fig, "Data: " + run_label(RUN) + "; costs from the run's cost model on XAUUSD H1 bars", pal)
    return fig


@register(PAGE, "hrp-correlation", needs=(RUN_NEED, "data_store"))
def hrp_correlation(pal):
    """Correlation of the net unit-vol streams in config order and in HRP (quasi-diagonal) order."""
    d = _hrp_inputs()
    corr, order = d["corr"], d["order"]
    names = list(STRATS)
    fig, axes = figure(1, 2, figsize=(8.0, 4.4))
    cmap = diverging_cmap(pal["mode"])
    for ax, idx, title in ((axes[0], list(range(len(names))), "config order"),
                           (axes[1], order, "quasi-diagonal (dendrogram leaf) order")):
        style_axes(ax, grid=None)
        m = corr[np.ix_(idx, idx)]
        im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
        for i in range(len(idx)):
            for j in range(len(idx)):
                v = 0.0 if abs(m[i, j]) < 0.005 else m[i, j]
                ax.text(j, i, signed(v, "{:.2f}"), ha="center", va="center",
                        fontsize=6.5, color=ink_on(im.cmap(im.norm(m[i, j]))))
        lab = [names[k] for k in idx]
        ax.set_xticks(range(len(idx)), lab, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(idx)), lab, fontsize=7.5)
        ax.tick_params(axis="both", length=0)
        ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.75, pad=0.02)
    cbar.set_label("correlation", color=pal["text_secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])
    fig.suptitle("Correlation of the strategies' net unit-vol return streams")
    source_note(fig, f"Data: {run_label(RUN)}; streams use XAUUSD H1 returns and the run's cost model", pal)
    return fig


@register(PAGE, "hrp-bisection", needs=(RUN_NEED, "data_store"))
def hrp_bisection(pal):
    """Single-linkage dendrogram with each node's split fraction, then HRP vs inverse-vol vs equal weights."""
    d = _hrp_inputs()
    names = list(STRATS)
    order = d["order"]
    xpos = {leaf: i for i, leaf in enumerate(order)}

    def node_x(node):
        if node.is_leaf():
            return float(xpos[node.id])
        return 0.5 * (node_x(node.get_left()) + node_x(node.get_right()))

    fig, (ax0, ax1) = panels(2, heights=(2.0, 2.4), figsize=(8.0, 6.4), hspace=0.06)
    ink = pal["text_secondary"]
    # Evenly spaced merge levels (single-linkage distances are too close together to label);
    # the real distance is printed at each node.
    level = {}
    for rank, (node, *_rest) in enumerate(sorted(d["splits"], key=lambda t: t[0].dist), start=1):
        level[node.id] = rank

    def node_y(node):
        return 0.0 if node.is_leaf() else float(level[node.id])

    for node, left, right, a in d["splits"]:
        xn, yn = node_x(node), node_y(node)
        xl, yl = node_x(left), node_y(left)
        xr, yr = node_x(right), node_y(right)
        ax0.plot([xl, xl, xr, xr], [yl, yn, yn, yr], color=ink, linewidth=1.3)
        ax0.annotate(f"{a:.0%}", (xl, yn), xytext=(-3, -3), textcoords="offset points", ha="right",
                     va="top", color=pal["text"], fontsize=7.5)
        ax0.annotate(f"{1 - a:.0%}", (xr, yn), xytext=(3, -3), textcoords="offset points", ha="left",
                     va="top", color=pal["text"], fontsize=7.5)
        ax0.plot([xn], [yn], marker="o", markersize=4, color=ink, linestyle="none")
        # right of the node, so the parent's vertical line does not run through the label
        ax0.annotate(f"d={node.dist:.2f}", (xn, yn), xytext=(4, 3), textcoords="offset points", ha="left",
                     va="bottom", color=pal["text_muted"], fontsize=7)
    ax0.set_ylim(0, len(level) + 0.7)
    ax0.set_yticks(range(1, len(level) + 1))
    ax0.set_ylabel("merge order")
    ax0.set_title("Single-linkage tree on d = sqrt((1 - rho) / 2); each node splits capital (left / right)")
    style_axes(ax0, grid=None)

    fits = d["fits"]
    x = np.arange(len(order))
    width = 0.27
    spec = (("hrp", "HRP"), ("inverse_vol", "inverse_vol"), ("equal", "equal"))
    for i, (m, lab) in enumerate(spec):
        wv = fits[m].weights_.to_numpy()[order]
        ax1.bar(x + (i - 1) * width, wv, width=width * 0.95, color=pal["categorical"][i], label=lab)
    hw = fits["hrp"].weights_.to_numpy()[order]
    for xi, v in zip(x, hw, strict=True):
        ax1.annotate(f"{v:.0%}", (xi - width, v), xytext=(0, 2), textcoords="offset points", ha="center",
                     va="bottom", color=pal["text"], fontsize=7.5)
    ax1.yaxis.set_major_formatter(_pct_formatter())
    ax1.set_xticks(x, [f"{names[k]}\n(std {d['std'][k]:.2f})" for k in order], fontsize=7.5)
    ax1.set_ylabel("weight (no cap)")
    ax1.set_title("Resulting weights (leaf order); std = per-bar std of the unit-vol stream")
    add_legend(ax1, loc="upper right", fontsize=8.5)
    source_note(fig, f"Data: {run_label(RUN)}; streams use XAUUSD H1 returns and the run's cost model", pal)
    return fig


@register(PAGE, "fdm-correlation")
def fdm_correlation(pal):
    """Analytic: FDM = 1/sqrt(w'Hw) for N equally weighted forecasts with common correlation rho."""
    rho = np.linspace(0.0, 1.0, 201)
    fig, ax = figure(figsize=FIGSIZE)
    for i, n in enumerate((2, 3, 7, 14)):
        fdm = 1.0 / np.sqrt(rho + (1.0 - rho) / n)
        ax.plot(rho, np.minimum(fdm, 2.5), color=pal["categorical"][i])
        ax.plot(rho[fdm > 2.5], fdm[fdm > 2.5], color=pal["categorical"][i], linestyle=":", linewidth=1.2)
        # direct label just above where each curve starts (rho = 0): 1.41, 1.73, 2.65, 3.74
        ax.annotate(f"N = {n}", (0.0, fdm[0]), xytext=(7, 3), textcoords="offset points",
                    ha="left", va="bottom", color=pal["text"], fontsize=9)
    ax.axhline(2.5, color=pal["text_secondary"], linewidth=1.0, linestyle="--")
    ax.annotate("fdm_cap = 2.5 (dotted: uncapped)", (1.0, 2.5), xytext=(0, 4), textcoords="offset points",
                ha="right", color=pal["text_secondary"], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 3.9)
    ax.set_title("Forecast diversification multiplier vs average forecast correlation")
    ax.set_xlabel("pairwise correlation rho of the forecasts (floored at corr_floor = 0)")
    ax.set_ylabel("FDM")
    source_note(fig, "Analytic illustration: equal weights 1/N, every pairwise correlation = rho", pal)
    return fig


# ---------------------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------------------
@register(PAGE, "sizing-curves")
def sizing_curves(pal):
    """VolTargetSizer leverage vs vol forecast (caps and floor) and the drawdown multiplier steps."""
    from aurum.core.instrument import XAUUSD

    sizer = VolTargetSizer()
    vols = np.linspace(0.005, 0.40, 400)
    fig, axes = figure(1, 2, figsize=FIGSIZE)
    ax = axes[0]
    for i, f in enumerate((1.0, 0.5, 0.25)):
        lev = [sizer.breakdown(f, v, 100_000, 2_000.0, XAUUSD).capped_notional / 100_000 for v in vols]
        ax.plot(vols, lev, color=pal["categorical"][i])
        xl = {1.0: 0.15, 0.5: 0.22, 0.25: 0.30}[f]
        ax.annotate(f"forecast {f:g}", (xl, f * 0.10 / xl), xytext=(3, 3), textcoords="offset points",
                    ha="left", va="bottom", color=pal["text"], fontsize=8.5)
    ax.axvline(sizer.min_vol, color=pal["axis"], linewidth=0.8, linestyle="--")
    ax.annotate(f"min_vol {sizer.min_vol:.0%}", (sizer.min_vol, 0.98), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", va="top", color=pal["text_muted"], fontsize=8)
    ax.axhline(sizer.max_leverage, color=pal["axis"], linewidth=0.8, linestyle="--")
    ax.annotate(f"max_leverage {sizer.max_leverage:g}x", (0.40, sizer.max_leverage), xytext=(0, 4),
                textcoords="offset points", ha="right", color=pal["text_muted"], fontsize=8)
    ax.xaxis.set_major_formatter(_pct_formatter())
    ax.set_ylim(0, 2.3)
    ax.set_xlim(0, 0.40)
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_yticks([0.05, 0.25, 0.5, 1.0, 1.5, 2.0])
    ax.set_title("Leverage = forecast x 10% / vol, capped", fontsize=10)
    ax.set_xlabel("annualised vol forecast")
    ax.set_ylabel("|notional| / equity (square-root scale)")

    ax = axes[1]
    dd = np.linspace(0.0, 0.25, 501)
    mult = [drawdown_multiplier(x, DEFAULT_DRAWDOWN_DERISK) for x in dd]
    ax.plot(dd, mult, color=pal["categorical"][0], drawstyle="steps-post")
    for thr, m in DEFAULT_DRAWDOWN_DERISK:
        ax.annotate(f"{thr:.0%} below peak: x{m:g}", (thr, m), xytext=(4, 6), textcoords="offset points",
                    color=pal["text"], fontsize=8.5)
    ax.xaxis.set_major_formatter(_pct_formatter())
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 0.25)
    ax.set_title("Drawdown multiplier (default steps)", fontsize=10)
    ax.set_xlabel("drawdown from the equity peak")
    ax.set_ylabel("exposure multiplier")
    fig.suptitle("VolTargetSizer defaults: target_vol 10%, max_leverage 2, min_vol 2%")
    source_note(fig, "Computed with VolTargetSizer.breakdown and drawdown_multiplier (equity 100k, price 2,000)",
                pal)
    return fig


@register(PAGE, "kelly-growth")
def kelly_growth(pal):
    """Expected log growth vs position volatility for two Sharpe ratios, with the sizer's targets."""
    from aurum.core.instrument import XAUUSD

    v = np.linspace(0.0, 1.3, 600)
    fig, ax = figure(figsize=FIGSIZE)
    c = pal["categorical"]
    for i, sr in enumerate((0.6, 0.1)):
        g = sr * v - 0.5 * v ** 2
        line, = ax.plot(v, g, color=c[i])
        ax.plot([sr], [0.5 * sr ** 2], marker="o", markersize=6, color=c[i],
                markeredgecolor=pal["surface"], linestyle="none")
        ax.annotate(f"Sharpe {sr:g}: full Kelly at {sr:.0%} vol, growth {0.5 * sr ** 2:.1%}/yr",
                    (sr, 0.5 * sr ** 2), xytext=(8, 8 if i == 0 else 10), textcoords="offset points",
                    color=pal["text"], fontsize=8.5)
    half = 0.30
    ax.plot([half], [0.6 * half - 0.5 * half ** 2], marker="o", markersize=7, markeredgewidth=2.0,
            markeredgecolor=c[0], markerfacecolor=pal["surface"], linestyle="none")
    ax.annotate("half Kelly (kelly_cap 0.30):\n75% of the growth at half the vol",
                (half, 0.6 * half - 0.5 * half ** 2), xytext=(10, -8), textcoords="offset points",
                ha="left", va="top", color=pal["text"], fontsize=8.5)
    ax.plot([1.2], [0.0], marker="o", markersize=6, color=c[0], markeredgecolor=pal["surface"], linestyle="none")
    ax.annotate("2x Kelly: growth back to 0", (1.2, 0.0), xytext=(-8, -8), textcoords="offset points",
                ha="right", va="top", color=pal["text_secondary"], fontsize=8.5)
    # the sizer, through its own breakdown: default target 10%, and a mis-set 40% caught by kelly_cap 0.30
    bd_default = VolTargetSizer().breakdown(1.0, 0.20, 100_000, 2_000.0, XAUUSD)
    bd_misset = VolTargetSizer(target_vol=0.40, kelly_cap=0.30).breakdown(1.0, 0.20, 100_000, 2_000.0, XAUUSD)
    v_default = bd_default.capped_notional * 0.20 / 100_000
    v_raw = bd_misset.raw_notional * 0.20 / 100_000
    v_capped = bd_misset.capped_notional * 0.20 / 100_000
    for x, txt in ((v_default, f"default target_vol {v_default:.0%}"),
                   (v_raw, f"mis-set target_vol {v_raw:.0%}"),
                   (v_capped, f"capped to {v_capped:.0%}")):
        ax.axvline(x, color=pal["axis"], linewidth=0.8, linestyle=":")
        ax.annotate(txt, (x, 0.0), xycoords=("data", "axes fraction"), xytext=(3, 4), textcoords="offset points",
                    rotation=90, va="bottom", ha="left", color=pal["text_muted"], fontsize=8)
    ax.annotate("", xy=(v_capped, -0.10), xytext=(v_raw, -0.10),
                arrowprops=dict(arrowstyle="->", color=pal["text_secondary"], linewidth=1.0))
    zero_line(ax)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.30, 0.25)
    ax.xaxis.set_major_formatter(_pct_formatter())
    ax.yaxis.set_major_formatter(_pct_formatter())
    ax.set_title("Expected log growth of a Gaussian strategy vs the volatility it is run at")
    ax.set_xlabel("position volatility (annualised, |N| x vol / equity)")
    ax.set_ylabel("expected log growth per year")
    source_note(fig, "Synthetic illustration: g = SR x v - v^2/2 for Gaussian returns; markers from "
                "VolTargetSizer.breakdown (forecast 1, vol 20%), not market data", pal)
    return fig


@register(PAGE, "realised-vs-target-vol", needs=(RUN_NEED, "data_store"))
def realised_vs_target_vol(pal):
    """Rolling realised vol vs the sizer's intended and ex-ante vol for two books of the trend_core run."""
    from aurum.models.volatility import ewma_volatility
    from aurum.risk.var import daily_returns_from_equity

    h1 = bars("H1")
    vol = ewma_volatility(h1["close"])
    target = _run_config().sizing.target_vol
    steps = tuple(tuple(s) for s in _run_config().sizing.drawdown_derisk)
    fig, axes = panels(2, figsize=FIGSIZE_TALL)
    c = pal["categorical"]
    win = 126
    for ax, book in zip(axes, ("tsmom", "combined"), strict=True):
        ts = _book(book)
        px = h1["close"].reindex(ts.index)
        dd = 1.0 - ts["equity"] / ts["equity"].cummax()
        mult = dd.map(lambda x: drawdown_multiplier(x, steps))
        intended = target * ts["forecast"].abs().clip(upper=1.0) * mult
        exante = ts["positions"].abs() * 100.0 * px * vol.reindex(ts.index) / ts["equity"]
        day = ts.index.normalize()
        daily_int = np.sqrt((intended ** 2).groupby(day).mean())
        daily_ex = np.sqrt((exante ** 2).groupby(day).mean())
        realised = daily_returns_from_equity(ts["equity"]).rolling(win).std() * np.sqrt(252.0)
        roll_int = np.sqrt((daily_int ** 2).rolling(win).mean())
        roll_ex = np.sqrt((daily_ex ** 2).rolling(win).mean())
        ax.axhline(target, color=pal["text_secondary"], linewidth=1.0, linestyle="--")
        ax.annotate(f"target_vol {target:.0%} (forecast of 1)", (realised.index[-1], target), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom", color=pal["text_secondary"], fontsize=8)
        for s, lab, colr in ((roll_int, "intended: 10% x |forecast| x dd multiplier", c[0]),
                             (roll_ex, "ex-ante: lots x price x EWMA vol / equity", c[1]),
                             (realised, f"realised: {win}-day std of daily returns", c[2])):
            ax.plot(s.index, s.to_numpy(), color=colr, label=lab)
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_ylim(0, 0.12)
        ax.set_title(f"{book} book: full-sample realised vol "
                     f"{daily_returns_from_equity(ts['equity']).std() * np.sqrt(252.0):.1%}", fontsize=10)
        ax.set_ylabel("annualised vol")
    add_legend(axes[0], loc="lower left", fontsize=8, ncol=1)
    axes[-1].set_xlabel("date (UTC); rolling windows of 126 trading days")
    fig.suptitle("The book carries far less risk than target_vol")
    source_note(fig, "Data: " + run_label(RUN) + "; XAUUSD H1 closes for the EWMA vol", pal)
    return fig


# ---------------------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------------------
@register(PAGE, "underwater", needs=(RUN_NEED,))
def underwater(pal):
    """Equity and drawdown of the combined book vs buy-and-hold, with the sizer's de-risk thresholds."""
    fig, (ax0, ax1) = panels(2, heights=(2.2, 2.0), figsize=FIGSIZE_TALL)
    c = pal["categorical"]
    for i, (book, lab) in enumerate((("combined", "combined book"), ("benchmark", "buy and hold"))):
        ts = _book(book)
        eq = ts["equity"] / ts["equity"].iloc[0] * 100.0
        lw, z = (1.6, 3) if book == "combined" else (0.9, 2)
        (ln,) = ax0.plot(eq.index, eq.to_numpy(), color=c[i], label=lab, linewidth=lw, zorder=z)
        label_line_end(ax0, ln, lab, fmt="{:.0f}")
        dd = ts["equity"] / ts["equity"].cummax() - 1.0
        mdd = f"{dd.min():.1%}".replace("-", "\N{MINUS SIGN}")
        ax1.plot(dd.index, dd.to_numpy(), color=c[i], label=f"{lab} (max {mdd})", linewidth=lw,
                 zorder=z)
        if book == "combined":
            t = dd.idxmin()
            ax1.annotate(f"{mdd} on {t:%Y-%m-%d}", (t, dd.min()), xytext=(6, -6),
                         textcoords="offset points", va="top", color=pal["text"], fontsize=8, zorder=5,
                         bbox={"boxstyle": "round,pad=0.15", "facecolor": pal["surface"], "edgecolor": "none",
                               "alpha": 0.95})
    for thr, m in DEFAULT_DRAWDOWN_DERISK:
        ax1.axhline(-thr, color=pal["axis"], linewidth=0.8, linestyle="--")
        ax1.annotate(f"\N{MINUS SIGN}{thr:.0%}: the sizer multiplies exposure by {m:g}",
                     (ax1.get_xlim()[0], -thr), xytext=(4, 3), textcoords="offset points", va="bottom",
                     color=pal["text_secondary"], fontsize=8, zorder=5,
                     bbox={"boxstyle": "round,pad=0.15", "facecolor": pal["surface"], "edgecolor": "none",
                           "alpha": 0.95})
    ax0.set_title("Equity, indexed to 100 at the first out-of-sample bar")
    ax0.set_ylabel("index")
    ax1.yaxis.set_major_formatter(_pct_formatter())
    ax1.set_title("Underwater curve: equity / running peak - 1")
    ax1.set_ylabel("drawdown")
    ax1.set_xlabel("date (UTC)")
    add_legend(ax1, loc="lower left", fontsize=8.5)
    source_note(fig, "Data: " + run_label(RUN), pal)
    return fig


@functools.cache
def _gold_daily() -> pd.Series:
    d1 = bars("D1_nyclose")
    return d1["close"].pct_change().dropna()


@register(PAGE, "var-es-gold", needs=("data_store",))
def var_es_gold(pal):
    """Daily gold returns: the left tail with VaR/ES lines, and VaR vs ES by method and level."""
    from scipy import stats

    from aurum.risk.var import VAR_METHODS, var_es

    r = _gold_daily()
    from matplotlib.figure import Figure

    fig = Figure(figsize=FIGSIZE)
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.hist(r, bins=160, range=(-0.10, 0.06), density=True, color=_blend(pal["grid"], pal["text_secondary"], 0.3),
            edgecolor=pal["surface"], linewidth=0.2)
    x = np.linspace(-0.10, 0.06, 400)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std(ddof=1)), color=pal["text_secondary"], linewidth=1.2,
            linestyle="--", label="Gaussian, same mean and sd")
    hist99 = var_es(r, 0.99, "historical")
    c = pal["categorical"]
    ax.axvline(-hist99["var"], color=c[0], linewidth=1.5, label=f"historical VaR 99% = {hist99['var']:.2%}")
    ax.axvline(-hist99["es"], color=c[1], linewidth=1.5, label=f"historical ES 99% = {hist99['es']:.2%}")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 80)
    ax.set_xlim(-0.10, 0.06)
    ax.xaxis.set_major_formatter(_pct_formatter())
    ax.set_title("Distribution of daily returns (log density)", fontsize=10)
    ax.set_xlabel("daily return")
    ax.set_ylabel("density (log scale)")
    add_legend(ax, loc="upper left", fontsize=7.5)

    ax = axes[1]
    rows = []
    for level in (0.95, 0.99):
        for m in VAR_METHODS:
            v = var_es(r, level, m)
            rows.append((f"{m.replace('_', '-')} {level:.0%}", v["var"], v["es"]))
    yy = np.arange(len(rows))[::-1]
    for yi, (_, v, e) in zip(yy, rows, strict=True):
        ax.plot([v, e], [yi, yi], color=pal["axis"], linewidth=1.2)
    ax.plot([r_[1] for r_ in rows], yy, "o", color=c[0], label="VaR")
    ax.plot([r_[2] for r_ in rows], yy, "o", color=c[1], label="ES")
    for yi, (_, _v, e) in zip(yy, rows, strict=True):
        ax.annotate(f"{e:.2%}", (e, yi), xytext=(5, 0), textcoords="offset points", va="center",
                    color=pal["text"], fontsize=7.5)
    ax.set_yticks(yy, [r_[0] for r_ in rows], fontsize=8)
    ax.xaxis.set_major_formatter(_pct_formatter(1))
    ax.set_xlim(0.0, 0.065)
    ax.set_title("1-day VaR and ES by method", fontsize=10)
    ax.set_xlabel("loss, fraction of position")
    style_axes(ax, grid="x")
    add_legend(ax, loc="upper right", fontsize=8)
    fig.suptitle("Value at Risk and Expected Shortfall of a 1x long gold position")
    source_note(fig, "Data: " + source_label("D1_nyclose") + ", close-to-close simple returns", pal)
    return fig


@register(PAGE, "var-backtest", needs=("data_store",))
def var_backtest(pal):
    """Rolling 250-day historical VaR(99%) on daily gold returns, breaches marked (Kupiec test)."""
    from aurum.risk.var import rolling_var_backtest

    r = _gold_daily()
    window, level = 250, 0.99
    q = r.rolling(window, min_periods=window).quantile(1 - level).shift(1)   # as in rolling_var_backtest
    ok = q.notna()
    breach = ok & (r < q)
    bt = rolling_var_backtest(r, level, window=window)
    assert int(breach.sum()) == bt["breaches"]
    c = pal["categorical"]
    fig, ax = figure(figsize=FIGSIZE)
    ax.vlines(r.index[ok], 0, r[ok], color=_blend(pal["grid"], pal["text_secondary"], 0.4), linewidth=0.6)
    ax.plot(q.index, q.to_numpy(), color=c[0], linewidth=1.3, label="minus VaR 99% (250 previous sessions)")
    ax.plot(r.index[breach], r[breach], "o", color=pal["status"]["serious"], markersize=4,
            markeredgecolor=pal["surface"], markeredgewidth=0.6, label=f"breach ({bt['breaches']} days)")
    zero_line(ax)
    ax.yaxis.set_major_formatter(_pct_formatter())
    ax.set_xlim(r.index[ok][0], r.index[-1])
    ax.set_title(f"VaR back-test: {bt['breaches']} breaches vs {bt['expected']:.1f} expected "
                 f"(Kupiec p = {bt['p_value']:.3f})")
    ax.set_ylabel("daily return")
    ax.set_xlabel("session date (UTC)")
    add_legend(ax, loc="lower right", fontsize=8.5)
    source_note(fig, "Data: " + source_label("D1_nyclose"), pal)
    return fig


@register(PAGE, "stress-scenarios", needs=("data_store",))
def stress_scenarios(pal):
    """DEFAULT_STRESS_SCENARIOS applied to a full-forecast VolTargetSizer position on 2024-12-31."""
    from aurum.core.instrument import XAUUSD
    from aurum.models.volatility import ewma_volatility
    from aurum.risk.var import stress_test

    h1 = bars("H1")
    vol = float(ewma_volatility(h1["close"]).iloc[-1])
    price = float(h1["close"].iloc[-1])
    equity = 100_000.0
    lots = VolTargetSizer().target_lots(1.0, vol, equity, price, XAUUSD)
    st = stress_test(lots, price, equity, spread=1.0).iloc[::-1]
    vals = st["pnl_pct_equity"].to_numpy()
    lim = float(np.abs(vals).max()) * 1.05
    cmap = diverging_cmap(pal["mode"])
    fig, ax = figure(figsize=FIGSIZE)
    style_axes(ax, grid="x")
    y = np.arange(len(st))
    ax.barh(y, vals, color=[cmap(0.5 + 0.5 * v / lim) for v in vals], height=0.65)
    for yi, v in zip(y, vals, strict=True):
        ax.annotate(signed(v * 100, "{:+.2f}%"), (v, yi), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", ha="left" if v >= 0 else "right", va="center",
                    color=pal["text"], fontsize=8)
    zero_line(ax, axis="x")
    ax.set_yticks(y, list(st["scenario"]), fontsize=8)
    ax.set_xlim(-lim * 1.3, lim * 1.3)
    ax.xaxis.set_major_formatter(_pct_formatter(1))
    ax.set_title(f"Stress test: long {lots:.2f} lots at {price:,.2f}, equity 100k")
    ax.set_xlabel("P&L as a share of equity (exit at a 1.00 USD/oz spread)")
    source_note(fig, f"Position = VolTargetSizer().target_lots(1.0, EWMA vol {vol:.1%}, ...) at the last bar "
                f"(2024-12-31). Data: " + source_label("H1"), pal)
    return fig
