"""Figures for the platform pages: docs/index.md, docs/architecture.md and docs/development.md.

One module, one figure directory per page (``index``, ``architecture``, ``development``):

* ``index/gold-research-timeline``: real XAUUSD 2012-2024 with the walk-forward folds of the
  pre-registered ``trend_core`` run and the 2025+ holdout left blank.
* ``architecture/decision-timeline``: decide at the close of bar t, fill at the open of t+1,
  on real H1 bars and the real decisions and fills of ``runs/protocol/trend_core_selection``.
* ``architecture/decision-chain``: one real decision (2020-08-11 13:00 UTC bar) from seven
  strategy forecasts through the combiner, the sizer, the risk manager and the fill, replayed
  with :func:`aurum.research.attribution.explain_decision`.
* ``development/style-palette`` and ``development/style-helpers``: the :mod:`aurum.viz.style`
  palette and helpers (no market data).

Render with:  python tools/figures/build.py --only index architecture development
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE_GRID,
    bars,
    ink_on,
    panels,
    register,
    run_table,
    signed,
    source_label,
    source_note,
)

from aurum.viz.style import (
    STATUS,
    add_legend,
    categorical,
    diverging,
    diverging_cmap,
    figure,
    label_line_end,
    percent_axis,
    sequential,
    shade_span,
    style_axes,
    zero_line,
)

RUN = "protocol/trend_core_selection"
#: the decision explained on docs/architecture.md: the close of the 2020-08-11 13:00 UTC H1 bar
#: (fold 9 of the trend_core walk-forward, out of sample), during gold's 2020-08-11 sell-off
DECISION_BAR = "2020-08-11 13:00"
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")   # configs/default.yaml walkforward.holdout_start
MINUS = "\N{MINUS SIGN}"
TIMES = "\N{MULTIPLICATION SIGN}"


def _fmt_thousands(v: float, _pos: int | None = None) -> str:
    return f"{v:,.0f}"


def _hatch_holdout(ax, pal, right, *, text: str | None = None) -> None:
    """The 2025+ holdout as an empty hatched band: it exists, but no figure may draw it."""
    ax.axvspan(HOLDOUT_START, right, facecolor="none", edgecolor=pal["axis"], hatch="///", linewidth=0,
               zorder=0)
    ax.axvline(HOLDOUT_START, color=pal["text_muted"], linewidth=0.75, zorder=1)
    if text:
        mid = HOLDOUT_START + (right - HOLDOUT_START) / 2
        ax.text(mid, 0.5, text, transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=8, color=pal["text_secondary"], linespacing=1.4,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": pal["surface"], "edgecolor": "none"})


def _dot(ax, x, y, color, pal, *, size: float = 7.0, zorder: float = 4) -> None:
    ax.plot([x], [y], marker="o", linestyle="none", color=color, markeredgecolor=pal["surface"],
            markeredgewidth=1.5, markersize=size, zorder=zorder)


# =========================================================================================
# index.md
# =========================================================================================
@register("index", "gold-research-timeline", needs=("data_store", f"runs/{RUN}"))
def gold_research_timeline(pal):
    """Gold 2012-2024 (log scale) with the 18 walk-forward folds and the untouched 2025+ holdout."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    close = bars("D1")["close"]                              # UTC days, cut at 2024-12-31
    folds = run_table(RUN, "folds.csv")
    for c in ("train_start", "train_end", "test_start", "test_end"):
        folds[c] = pd.to_datetime(folds[c], utc=True)
    oos0, oos1 = folds["test_start"].min(), folds["test_end"].max()
    left = pd.Timestamp("2012-01-01", tz="UTC")
    right = pd.Timestamp("2026-10-01", tz="UTC")
    blue = pal["categorical"][0]

    fig, (ax0, ax1) = panels(2, heights=(3.0, 1.9), figsize=(8.0, 6.0), hspace=0.06)

    # --- price -------------------------------------------------------------------------
    shade_span(ax0, oos0, oos1, f"walk-forward out of sample: {oos0:%Y-%m-%d} to {oos1:%Y-%m-%d}")
    _hatch_holdout(ax0, pal, right, text="holdout\nfrom 2025-01-01\nevaluated once,\nnever drawn")
    line, = ax0.plot(close.index, close.to_numpy(), color=blue, linewidth=1.5)
    ax0.set_yscale("log")
    ax0.yaxis.set_major_locator(FixedLocator([1000, 1250, 1500, 2000, 2500, 3000]))
    ax0.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax0.yaxis.set_minor_locator(NullLocator())
    ax0.set_ylim(950, 3300)
    # extremes come from the data, not by hand
    marks = [                                                 # (time, label, offset in points, ha, va)
        (close.loc["2015"].idxmin(), "2015 low", (8, -3), "left", "top"),
        (close.loc["2020"].idxmax(), "2020 high", (0, 9), "center", "bottom"),
        (close.loc["2012"].idxmax(), "2012 high", (-8, 9), "left", "bottom"),
    ]
    for t, what, off, ha, va in marks:
        v = float(close.loc[t])
        _dot(ax0, t, v, blue, pal, size=6)
        ax0.annotate(f"{what} {v:,.0f} ({t:%b %Y})", (t, v), xytext=off, textcoords="offset points",
                     ha=ha, va=va, fontsize=8, color=pal["text"])
    end = label_line_end(ax0, line, f"{close.iloc[-1]:,.0f} on {close.index[-1]:%Y-%m-%d}")
    end.set_bbox({"boxstyle": "round,pad=0.2", "facecolor": pal["surface"], "edgecolor": "none"})
    ax0.set_title("Spot gold 2012 to 2024: the data aurum researches on, and how it is split")
    ax0.set_ylabel("USD per ounce (log scale)")

    # --- folds ---------------------------------------------------------------------------
    _hatch_holdout(ax1, pal, right)
    muted = pal["text_muted"]
    for i, r in folds.iterrows():
        ax1.hlines(i, r["train_start"], r["train_end"], color=muted, linewidth=2.6, capstyle="butt")
        ax1.hlines(i, r["test_start"], r["test_end"], color=blue, linewidth=2.6, capstyle="butt")
    f0 = folds.iloc[0]
    ax1.annotate("train: fit every model on 3 years", (f0["train_start"], 0), xytext=(0, 7),
                 textcoords="offset points", ha="left", va="bottom", fontsize=8, color=pal["text_secondary"])
    ax1.annotate("test: trade the next 6 months, out of sample", (f0["test_end"], 0), xytext=(6, 0),
                 textcoords="offset points", ha="left", va="center", fontsize=8, color=pal["text_secondary"])
    n = len(folds)
    ax1.set_ylim(n - 0.3, -1.9)
    ax1.set_yticks([0, n - 1], [f"fold {int(folds['fold'].iloc[0])}", f"fold {int(folds['fold'].iloc[-1])}"])
    ax1.set_title(f"Walk-forward of the trend_core book: {n} folds, refit before each test window",
                  fontsize=10)
    ax1.set_xlabel("date (UTC)")
    ax1.grid(False)
    ax1.set_xlim(left, right)
    source_note(fig, "Data: " + source_label("D1") + f"; folds: runs/{RUN}", pal)
    return fig


# =========================================================================================
# architecture.md
# =========================================================================================
def _num(ts) -> float:
    import matplotlib.dates as mdates

    return float(mdates.date2num(pd.Timestamp(ts).to_pydatetime()))


@register("architecture", "decision-timeline", needs=("data_store", f"runs/{RUN}"))
def decision_timeline(pal):
    """Decide at the close of bar t, fill at the open of bar t+1 (real H1 bars and run decisions)."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    day = DECISION_BAR[:10]
    b = bars("H1", start=f"{day} 09:00", end=f"{day} 17:00")
    ts = run_table(RUN, "books/combined/timeseries.parquet")
    fills = run_table(RUN, "books/combined/fills.csv")
    t = pd.Timestamp(DECISION_BAR, tz="UTC")
    t1 = b.index[b.index.get_loc(t) + 1]                    # next bar open = decision time
    hour = 1.0 / 24.0
    up, down = pal["categorical"][0], pal["categorical"][7]
    orange = pal["categorical"][1]
    fill = fills.loc[fills.index == t1].iloc[0]
    open_mid, spread = float(b.loc[t1, "open"]), float(b.loc[t1, "spread"])
    rng = float(b.loc[t1, "high"] - b.loc[t1, "low"])
    half = max(spread * 1.0, 0.10) / 2                       # costs: spread_multiplier 1.0, min_spread 0.10
    slip = 0.02 + 0.02 * rng                                 # slippage_fixed 0.02 + range_frac 0.02 x range

    fig, (ax0, ax1, ax2) = panels(3, heights=(3.2, 1.25, 1.25), figsize=(8.0, 7.0), hspace=0.08)
    x_t, x_t1 = _num(t), _num(t1)
    for ax in (ax0, ax1, ax2):
        shade_span(ax, x_t, x_t1, "bar t" if ax is ax0 else None)
        ax.axvline(x_t1, color=pal["text_secondary"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax0.annotate("bar t+1", (x_t1, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
                 textcoords="offset points", ha="left", va="top", color=pal["text_muted"], fontsize=8)

    # --- candles -----------------------------------------------------------------------------
    for ts_, r in b.iterrows():
        x0 = _num(ts_)
        c = up if r["close"] >= r["open"] else down
        ax0.vlines(x0 + hour / 2, r["low"], r["high"], color=c, linewidth=1.0, zorder=2)
        lo, hi = sorted((r["open"], r["close"]))
        ax0.add_patch(Rectangle((x0 + 0.15 * hour, lo), 0.7 * hour, max(hi - lo, 0.15), facecolor=c,
                                edgecolor="none", zorder=2))
    ymin, ymax = float(b["low"].min()) - 12, float(b["high"].max()) + 6
    ax0.set_ylim(ymin, ymax)
    ax0.plot([x_t1], [fill["price"]], marker="v", linestyle="none", color=orange, markersize=9,
             markeredgecolor=pal["surface"], markeredgewidth=1.5, zorder=5)
    ax0.annotate(
        f"fill at the open of bar t+1:\nsell {abs(fill['lots']):.2f} lots at {fill['price']:,.2f}\n"
        f"= {open_mid:,.2f} {MINUS} {half:.2f} {MINUS} {slip:.2f}\n(open mid, spread/2, slippage)",
        (x_t1, float(fill["price"])), xytext=(5, -9), textcoords="offset points", ha="left", va="top",
        fontsize=8, color=pal["text"], linespacing=1.3)
    ax0.annotate(f"decide at {t1:%H:%M} UTC = available_at[t]:\nonly bars up to t are used",
                 (x_t1, ymin + 2), xytext=(-6, 0), textcoords="offset points", ha="right", va="bottom",
                 fontsize=8.5, color=pal["text"])
    ax0.yaxis.set_major_formatter(_fmt_thousands_axis())
    ax0.set_title(f"Decide at the close of bar t, fill at the open of bar t+1 (XAUUSD H1, {day})")
    ax0.set_ylabel("USD per ounce")

    # --- forecast decided at each close, drawn inside the bar it was computed on ---------------
    win = ts.loc[(ts.index >= b.index[0]) & (ts.index <= b.index[-1])]
    xs = np.array([_num(i) + hour / 2 for i in win.index])
    fc = win["forecast"].to_numpy(float)
    ax1.bar(xs, fc, width=0.7 * hour, color=[up if v >= 0 else down for v in fc], zorder=2)
    zero_line(ax1)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_yticks([-1, 0, 1])
    f_t = float(win.loc[t, "forecast"])
    ax1.annotate(signed(f_t), (_num(t) + hour / 2, f_t), xytext=(0, -3), textcoords="offset points",
                 ha="center", va="top", fontsize=8.5, color=pal["text"])
    ax1.set_title("combined forecast, decided at each close (always in [\N{MINUS SIGN}1, 1])", fontsize=10)
    ax1.set_ylabel("forecast")

    # --- position held during each bar (filled at that bar's open) -----------------------------
    pos = win["positions"].to_numpy(float)
    ax2.bar(xs, pos, width=0.7 * hour, color=[up if v >= 0 else down for v in pos], zorder=2)
    zero_line(ax2)
    p_t1 = float(win.loc[t1, "positions"])
    ax2.annotate(f"{signed(p_t1)} lots, sized from the {signed(f_t)} decided one bar earlier",
                 (_num(t1) + 0.15 * hour, p_t1), xytext=(0, -4), textcoords="offset points", ha="left",
                 va="top", fontsize=8.5, color=pal["text"])
    lim = max(abs(pos).max(), 0.05)
    ax2.set_ylim(-lim * 1.75, lim * 0.35)
    ax2.set_title("position held during each bar, in lots (100 oz per lot)", fontsize=10)
    ax2.set_ylabel("lots")
    ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1), tz="UTC"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax2.set_xlim(_num(b.index[0]), _num(b.index[-1]) + hour)
    ax2.set_xlabel(f"{day}, bar open time (UTC)")
    source_note(fig, "Data: " + source_label("H1", start=f"{day} 09:00", end=f"{day} 17:00")
                + f"; forecasts, positions and fills: runs/{RUN} (fold 9, out of sample)", pal)
    return fig


def _fmt_thousands_axis():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(_fmt_thousands)


@register("architecture", "decision-chain", needs=("data_store", f"runs/{RUN}"))
def decision_chain(pal):
    """One real decision through the stack: 7 forecasts -> combiner -> sizer -> band -> risk -> fill."""
    from aurum.research.attribution import explain_decision

    md_bars = bars("H1", start="2013-01-01")                # the run's data.start; cut at 2024-12-31
    ex = explain_decision(f"runs/{RUN}", DECISION_BAR, md=md_bars)
    if not ex.sizing.get("replay_matches_saved"):
        raise RuntimeError(f"sizing replay does not match the saved request: {ex.sizing} {ex.notes}")
    tab = ex.table
    sz = ex.sizing
    fills = run_table(RUN, "books/combined/fills.csv")
    t1 = pd.Timestamp(ex.decision_time)
    fill = fills.loc[fills.index == t1].iloc[0]
    blue, orange = pal["categorical"][0], pal["categorical"][1]

    fig, (ax0, ax1) = panels(2, heights=(1.25, 1.0), figsize=(8.0, 7.6), sharex=False, hspace=0.12)

    # --- 1. strategies and combiner (forecast units) ------------------------------------------
    n = len(tab)
    y = np.arange(n)
    h = 0.36
    ax0.barh(y - h / 2, tab["forecast"], height=h, color=blue, label="forecast $f_i$ (strategy output)",
             zorder=2)
    ax0.barh(y + h / 2, tab["contribution"], height=h, color=orange,
             label="contribution $c_i = w_i \\cdot \\mathrm{FDM} \\cdot f_i$", zorder=2)
    for yi, (f, c) in enumerate(zip(tab["forecast"], tab["contribution"], strict=True)):
        for val, yy, zero_note in ((f, yi - h / 2, "0: no signal"), (c, yi + h / 2, "0: weight 0")):
            if abs(val) < 1e-12:                             # say why there is no bar
                if zero_note == "0: weight 0" and abs(f) < 1e-12:
                    continue                                 # no signal already explains it
                ax0.text(0.02, yy, zero_note, ha="left", va="center", fontsize=7.5, color=pal["text_muted"])
                continue
            x, ha = val + (0.02 if val > 0 else -0.02), "left" if val > 0 else "right"
            ax0.text(x, yy, signed(val), ha=ha, va="center", fontsize=7.5, color=pal["text_secondary"])
    for yi, w in enumerate(tab["weight"]):
        ax0.annotate(f"{w:.3f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(8, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=8, color=pal["text"])
    ax0.annotate("weight $w_i$", (1.0, -0.85), xycoords=("axes fraction", "data"), xytext=(8, 0),
                 textcoords="offset points", ha="left", va="center", fontsize=8, color=pal["text_secondary"])
    zero_line(ax0, axis="x")
    raw = float(ex.raw_combined)
    ax0.axvline(raw, color=pal["text"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax0.annotate(f"combined = sum of contributions = {signed(raw, '{:+.3f}')}\n(inside [{MINUS}1, 1]: no clip)",
                 (raw, n - 0.45), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                 fontsize=8.5, color=pal["text"])
    ax0.set_yticks(y, list(tab["strategy"]))
    ax0.set_ylim(n - 0.2, -1.3)
    ax0.set_xlim(-1.12, 1.12)
    ax0.grid(True, axis="x")
    ax0.grid(False, axis="y")
    add_legend(ax0, loc="upper right", fontsize=8.5)
    fdm = float(tab["fdm"].iloc[0])
    ax0.set_title(f"1. Strategies and combiner: decision at the close of {pd.Timestamp(ex.time):%Y-%m-%d %H:%M} UTC "
                  f"(fold {ex.fold}, FDM {fdm:.3f})", fontsize=11)
    ax0.set_xlabel("forecast units (unitless, [\N{MINUS SIGN}1, 1])")

    # --- 2. sizer, band, risk, fill (lots) -------------------------------------------------------
    cur, raw_lots, rounded = float(sz["current_lots"]), float(sz["raw_lots"]), float(sz["rounded_lots"])
    final, filled = float(sz["replayed_lots"]), float(sz["filled_lots_next_bar"])
    band = 0.10                                              # sizing.rebalance_band of the run
    # keep zone around the current position: |x - cur| < band * max(|x|, |cur|) (same sign)
    keep_lo, keep_hi = cur / (1 - band), cur * (1 - band)
    approved = final if not ex.risk["interventions"] else float(ex.risk["interventions"][-1]["approved"])
    move = abs(rounded - cur)
    rows = [
        ("held at the close", cur, f"{signed(cur)} lots held when bar t closes"),
        ("sizer", raw_lots,
         f"{signed(sz['forecast'], '{:+.3f}')} {TIMES} 10% / {sz['vol_ann']:.1%} {TIMES} {sz['equity']:,.0f} USD"
         f" = {signed(sz['raw_notional'], '{:+,.0f}')} USD = {signed(raw_lots, '{:+.4f}')} lots"),
        ("rounded", rounded, f"{signed(rounded)}: toward zero on the 0.01-lot grid"),
        ("rebalance band", final,
         f"move {move:.2f} \N{GREATER-THAN OR EQUAL TO} {band:.2f} {TIMES} {max(abs(rounded), abs(cur)):.2f}: trade"
         if not sz["kept_current_band"] else "inside the band: keep the current position"),
        ("risk manager", approved, "approved unchanged (no limit binding)" if not ex.risk["interventions"]
         else "reduced by the risk manager"),
        ("filled", filled, f"sold {abs(fill['lots']):.2f} lots at {fill['price']:,.2f} at the open of bar t+1"),
    ]
    shade_span(ax1, keep_lo, keep_hi, None)
    ax1.annotate(f"no-trade band around {signed(cur)}", (keep_hi, 0.0), xycoords=("data", "axes fraction"),
                 xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom", fontsize=8,
                 color=pal["text_muted"])
    ys = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax1.plot(vals, ys, color=pal["axis"], linewidth=1.0, zorder=2)
    for yi, (_, v, note) in enumerate(rows):
        _dot(ax1, v, yi, pal["text_muted"] if yi == 0 else blue, pal, size=8)
        ax1.annotate(note, (v, yi), xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=8.5, color=pal["text"])
    ax1.set_yticks(ys, [r[0] for r in rows])
    ax1.set_ylim(len(rows) + 0.1, -0.6)
    lo = min(vals) - 0.006
    ax1.set_xlim(lo, lo + 0.075)
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    ax1.set_title(f"2. Sizer, risk and execution: vol {sz['vol_ann']:.1%}, price {sz['price']:,.2f}, "
                  f"drawdown {sz['drawdown']:.1%}", fontsize=11)
    ax1.set_xlabel("lots (negative = short; 1 lot = 100 oz)")
    source_note(fig, f"Data: runs/{RUN} (walk-forward OOS, fold {ex.fold}); sizing replayed on "
                + source_label("H1", start="2013-01-01"), pal)
    return fig


# =========================================================================================
# development.md
# =========================================================================================
@register("development", "style-palette")
def style_palette(pal):
    """The aurum.viz.style palette for one mode: categorical, sequential, diverging and status."""
    from matplotlib.patches import Rectangle

    mode = pal["mode"]
    fig, ax = figure(figsize=(8.0, 4.4))
    style_axes(ax, grid=None)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    x0, width = 2.0, 9.6

    def row(y, colours, names=None, *, label, hexes=True, h=0.62):
        k = len(colours)
        gap = 0.08
        w = (width - gap * (k - 1)) / k
        for i, c in enumerate(colours):
            x = x0 + i * (w + gap)
            ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="none"))
            if hexes:
                ax.text(x + w / 2, y + h / 2, c.lstrip("#"), ha="center", va="center",
                        fontsize=6.5 if k > 8 else 7.5, color=ink_on(c))
            if names:
                ax.text(x + w / 2, y - 0.08, names[i], ha="center", va="top", fontsize=7.5,
                        color=pal["text_secondary"])
        ax.text(x0 - 0.15, y + h / 2, label, ha="right", va="center", fontsize=9, color=pal["text"])

    cat_names = ["0 blue", "1 orange", "2 aqua", "3 yellow", "4 magenta", "5 green", "6 violet", "7 red"]
    row(3.3, categorical(8), cat_names, label="categorical\n(fixed order)")
    row(2.2, sequential(), label="sequential\n(near zero first)")
    row(1.25, diverging(7), ["negative", "", "", "midpoint", "", "", "positive"], label="diverging")
    row(0.1, [STATUS[k] for k in ("good", "warning", "serious", "critical")],
        ["good", "warning", "serious", "critical"], label="status\n(with a word)")
    ax.set_xlim(0, x0 + width + 0.1)
    ax.set_ylim(-0.35, 4.05)
    ax.set_title(f"aurum.viz.style palette, {mode} mode (surface {pal['surface']})")
    return fig


@register("development", "style-helpers")
def style_helpers(pal):
    """Synthetic illustration of the aurum.viz.style helpers in a 2x2 grid."""
    rng = np.random.default_rng(11)
    fig, axes = figure(2, 2, figsize=FIGSIZE_GRID)
    (a, b), (c, d) = axes

    # (a) direct labels instead of a legend
    x = np.arange(250)
    for i, name in enumerate(["tsmom", "ema_cross", "donchian"]):
        yv = 100 * np.exp(np.cumsum(rng.normal(0.0002 * (1 - i), 0.006, x.size)))
        ln, = a.plot(x, yv, color=pal["categorical"][i], label=name)
        label_line_end(a, ln, fmt="{:.0f}")
    a.set_title("label_line_end: ink text, coloured dot", fontsize=10)
    a.set_xlabel("days")
    a.set_ylabel("equity (start = 100)")
    a.set_xlim(0, 330)

    # (b) shade_span + zero_line + percent_axis
    r = rng.normal(0.0, 0.01, 120)
    b.bar(np.arange(r.size), r, color=[pal["categorical"][0] if v >= 0 else pal["categorical"][7] for v in r],
          width=0.8)
    shade_span(b, 80, 119.5, "holdout band")
    zero_line(b)
    b.set_ylim(-0.03, 0.036)
    percent_axis(b, decimals=0)
    b.set_title("zero_line, percent_axis, shade_span", fontsize=10)
    b.set_xlabel("days")
    b.set_ylabel("daily return")

    # (c) diverging heatmap, symmetric limits, readable cell text
    z = rng.normal(0, 0.25, (3, 4))
    style_axes(c, grid=None)
    lim = float(np.abs(z).max())
    im = c.imshow(z, cmap=diverging_cmap(pal["mode"]), vmin=-lim, vmax=lim)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            c.text(j, i, signed(z[i, j]), ha="center", va="center", fontsize=8,
                   color=ink_on(im.cmap(im.norm(z[i, j]))))
    c.set_xticks(range(4), ["4", "8", "16", "32"])         # every fast span below every slow span
    c.set_yticks(range(3), ["64", "128", "256"])
    c.set_xlabel("fast window (bars)")
    c.set_ylabel("slow window (bars)")
    c.set_title("diverging_cmap, vmin = \N{MINUS SIGN}vmax, ink_on", fontsize=10)

    # (d) two series: add_legend draws one; sequential ramp for ordered magnitude
    k = np.arange(5)
    seq = sequential(5)
    d.bar(k, [0.4, 0.9, 1.5, 2.2, 3.1], color=seq, width=0.7, label="sequential(5)")
    d.plot(k, [0.6, 1.0, 1.4, 1.9, 2.6], color=pal["categorical"][1], marker="o", label="categorical slot 1")
    add_legend(d, loc="upper left", fontsize=8)
    d.set_xticks(k, ["q1", "q2", "q3", "q4", "q5"])
    d.set_title("sequential ramp, add_legend (2 series)", fontsize=10)
    d.set_xlabel("quintile")
    d.set_ylabel("value")
    source_note(fig, "Synthetic illustration (numpy seed 11), not market data", pal)
    return fig
