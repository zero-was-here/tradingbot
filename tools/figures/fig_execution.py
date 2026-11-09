"""Figures for docs/execution-and-costs.md (page key ``execution-and-costs``).

Every chart is drawn by aurum's own execution code (``aurum.execution.costs``,
``aurum.execution.simulator``, ``aurum.live.paper``) on real XAUUSD bars and FRED fed funds,
cut at 2024-12-31 by the ``_common`` loaders, or read from the walk-forward run
``runs/protocol/trend_core_selection``. Render with::

    python tools/figures/build.py --only execution-and-costs
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_TALL,
    bars,
    macro,
    macro_label,
    panels,
    register,
    run_label,
    run_table,
    signed,
    source_label,
    source_note,
)

from aurum.backtest.engine import average_true_range
from aurum.core.instrument import XAUUSD
from aurum.core.types import Side
from aurum.execution.costs import CostModel, FinancingModel, _weekday_weights
from aurum.execution.simulator import ExecutionSimulator, intrabar_exit
from aurum.live.broker import OrderRequest, SimulatedClock, net_lots
from aurum.live.paper import PaperBroker, ReplayFeed
from aurum.viz.style import (
    add_legend,
    figure,
    label_line_end,
    percent_axis,
    shade_span,
    style_axes,
    zero_line,
)

PAGE = "execution-and-costs"
RUN = "protocol/trend_core_selection"


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _candles(ax, frame: pd.DataFrame, pal, *, width: float = 0.6, highlight: dict | None = None) -> None:
    """Draw OHLC candles at integer x positions 0..n-1 (recessive ink; optional highlight)."""
    highlight = highlight or {}
    for i, (_, row) in enumerate(frame.iterrows()):
        color = highlight.get(i, pal["text_secondary"])
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1.2, solid_capstyle="butt", zorder=2)
        lo, hi = sorted((row["open"], row["close"]))
        filled = row["close"] < row["open"]
        ax.add_patch(_rect(i - width / 2, lo, width, max(hi - lo, 1e-6), color, filled, pal))


def _rect(x, y, w, h, color, filled, pal):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, facecolor=color if filled else pal["surface"], edgecolor=color,
                     linewidth=1.2, zorder=3)


def _hour_label(ts: pd.Timestamp) -> str:
    return f"{ts:%H:%M}"


# ---------------------------------------------------------------------------------------
# 1. timing
# ---------------------------------------------------------------------------------------
@register(PAGE, "timing", needs=("data_store",))
def timing(pal):
    """Decide at the close of bar t, fill at the open of bar t+1 (real H1 candles)."""
    day = "2024-03-06"
    b = bars("H1", start=f"{day} 07:00", end=f"{day} 12:59")
    t = 4                                   # the decision bar (11:00-12:00 UTC)
    cm = CostModel()
    o1 = float(b["open"].iloc[t + 1])
    rng1 = float(b["high"].iloc[t + 1] - b["low"].iloc[t + 1])
    buy = cm.fill_price(+1, o1, float(b["spread"].iloc[t + 1]), rng1, 1.0).price

    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, ax = figure(figsize=FIGSIZE)
    shade_span(ax, -0.5, t + 0.5, "known at the decision: bars up to t", alpha=0.6)
    _candles(ax, b, pal, highlight={t: blue, t + 1: orange})
    ax.axvline(t + 0.5, color=pal["text"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ymax = float(b["high"].max())
    ymin = float(b["low"].min())
    pad = (ymax - ymin) * 0.08
    ax.annotate(f"decision time = close of bar t\n= available_at[t] = {b['available_at'].iloc[t]:%H:%M} UTC",
                (t + 0.5, ymin - 1.7 * pad), xytext=(6, 0), textcoords="offset points", ha="left",
                va="bottom", fontsize=8.5, color=pal["text"])
    ax.plot([t + 1], [buy], marker="o", linestyle="none", color=orange, markersize=7,
            markeredgecolor=pal["surface"], markeredgewidth=1.5, zorder=5)
    arrow = {"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.8}
    ax.annotate(f"buy fills at open[t+1] + spread/2 + slippage\n= {o1:,.2f} + costs = {buy:,.2f}",
                (t + 1, buy), xytext=(t + 1.45, buy - 3.5 * pad), textcoords="data", ha="left", va="center",
                fontsize=8.5, color=pal["text"], arrowprops=arrow)
    ax.annotate("bar t+1 only settles the order:\nits high/low trigger stops and size\nthe slippage, "
                "its close marks equity", (t + 1.05, float(b["high"].iloc[t + 1])),
                xytext=(t + 1.45, ymax + 1.0 * pad), textcoords="data", ha="left",
                va="bottom", fontsize=8.5, color=pal["text_secondary"], arrowprops=arrow)
    ax.annotate("bar t", (t, float(b["low"].iloc[t])), xytext=(0, -6), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=pal["text"])
    ax.annotate("bar t+1", (t + 1, float(b["low"].iloc[t + 1])), xytext=(0, -6), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=pal["text"])
    ax.set_ylim(ymin - 2.0 * pad, ymax + 3.2 * pad)
    ax.set_xlim(-0.6, len(b) + 2.3)
    ax.set_xticks(range(len(b)), [_hour_label(i) for i in b.index])
    ax.set_xlabel(f"bar open time, {day} (UTC)")
    ax.set_ylabel("USD per ounce (mid)")
    ax.set_title("Decide at the close of bar t, fill at the open of bar t+1")
    src = source_label("H1", start=day, end=day, years=False).replace(f"{day} to {day}", day)
    source_note(fig, "Data: " + src, pal)
    return fig


# ---------------------------------------------------------------------------------------
# 2. anatomy of a fill
# ---------------------------------------------------------------------------------------
_QUIET = "2024-08-28 03:00"
_FAST = "2024-07-05 12:00"


@register(PAGE, "fill-ladder", needs=("data_store",))
def fill_ladder(pal):
    """Buy and sell fill prices around the mid for a quiet and a fast real H1 bar."""
    cm = CostModel()
    b = bars("H1", start="2024-07-01", end="2024-08-31")
    blue, orange, aqua = pal["categorical"][0], pal["categorical"][1], pal["categorical"][2]
    fig, axes = figure(1, 2, figsize=FIGSIZE, sharey=True)
    top = 0.0
    for ax, when, title in ((axes[0], _QUIET, "Quiet bar: 03:00 UTC"),
                            (axes[1], _FAST, "Fast bar: US payrolls hour")):
        row = b.loc[pd.Timestamp(when, tz="UTC")]
        spread, rng = float(row["spread"]), float(row["high"] - row["low"])
        half = 0.5 * cm.effective_spread(spread)
        fixed = cm.slippage_fixed
        ranged = cm.slippage_range_frac * rng
        parts = [(half, blue, "half spread"), (fixed, orange, "fixed slippage"),
                 (ranged, aqua, "2% of bar range")]
        for x, sign in ((0, +1), (1, -1)):
            base = 0.0
            for val, col, _ in parts:
                ax.bar(x, sign * val, bottom=sign * base, width=0.55, color=col, linewidth=0)
                base += val
            fp = cm.fill_price(sign, float(row["open"]), spread, rng, 1.0)
            ax.annotate(f"{'buy' if sign > 0 else 'sell'} {fp.price:,.2f}\n({signed(sign * base, '{:+.3f}')})",
                        (x, sign * base), xytext=(0, 5 * sign), textcoords="offset points", ha="center",
                        va="bottom" if sign > 0 else "top", fontsize=8.5, color=pal["text"])
        # direct labels for the three components on the buy bar (thin fixed slippage on the left)
        base = 0.0
        for val, _col, lab in parts:
            if lab == "fixed slippage":
                ax.annotate(f"{lab} {val:.2f}", (-0.28, base + val / 2), xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7.5, color=pal["text_secondary"])
            else:
                ax.annotate(f"{lab} {val:.3f}", (0.28, base + val / 2), xytext=(4, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=7.5, color=pal["text_secondary"])
            base += val
        top = max(top, base)
        zero_line(ax)
        ax.annotate(f"mid open {float(row['open']):,.2f}", (-1.0, 0), xytext=(2, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=7.5, color=pal["text_muted"])
        ax.set_xticks([0, 1], ["buy", "sell"])
        ax.set_xlim(-1.05, 1.6)
        ax.set_title(f"{title}\nspread {spread:.2f}, range {rng:.2f} USD/oz", fontsize=10)
    axes[0].set_ylim(-1.45 * top, 1.45 * top)
    axes[0].set_ylabel("fill price minus mid open (USD/oz)")
    fig.suptitle("Anatomy of a market fill: mid + spread/2 + slippage (CostModel defaults)")
    source_note(fig, "Data: XAUUSD H1 bars of 2024-08-28 03:00 and 2024-07-05 12:00 UTC, Dukascopy bid/ask mid",
                pal)
    return fig


# ---------------------------------------------------------------------------------------
# 3. spread and per-fill cost by hour
# ---------------------------------------------------------------------------------------
@register(PAGE, "cost-by-hour", needs=("data_store",))
def cost_by_hour(pal):
    """Quoted spread by UTC hour and the median per-fill cost components (USD/oz)."""
    cm = CostModel()
    b = bars("H1")
    hour = b.index.hour
    s = b["spread"].astype(float)
    g = s.groupby(hour)
    med, q10, q90 = g.median(), g.quantile(0.10), g.quantile(0.90)
    half = (0.5 * np.maximum(s * cm.spread_multiplier, cm.min_spread)).groupby(hour).median()
    ranged = (cm.slippage_range_frac * (b["high"] - b["low"]).astype(float)).groupby(hour).median()
    fixed = pd.Series(cm.slippage_fixed, index=half.index)

    blue, orange, aqua = pal["categorical"][0], pal["categorical"][1], pal["categorical"][2]
    fig, (ax0, ax1) = panels(2, heights=(1, 1.2), figsize=FIGSIZE_TALL)
    x = med.index.to_numpy()
    ax0.fill_between(x, q10.values, q90.values, color=blue, alpha=0.18, linewidth=0)
    line, = ax0.plot(x, med.values, color=blue, marker="o", markersize=3.5)
    ax0.axhline(cm.min_spread, color=pal["text_muted"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax0.annotate(f"min_spread floor {cm.min_spread:.2f}", (0, cm.min_spread), xytext=(2, 3),
                 textcoords="offset points", fontsize=8, color=pal["text_muted"], va="bottom")
    ax0.annotate("median", (x[9], med.values[9]), xytext=(0, 8), textcoords="offset points", ha="center",
                 fontsize=8, color=pal["text"])
    ax0.annotate("10% to 90% of bars", (x[9], q90.values[9]), xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, color=pal["text_secondary"])
    ax0.set_ylim(0, float(q90.max()) * 1.18)
    ax0.set_ylabel("quoted spread (USD/oz)")
    ax0.set_title("Quoted bid/ask spread by hour of day")

    bottom = np.zeros(len(x))
    comps = [(half.values, blue, "half spread"), (fixed.values, orange, "fixed slippage 0.02"),
             (ranged.values, aqua, "2% of bar range")]
    for vals, col, lab in comps:
        ax1.bar(x, vals, bottom=bottom, width=0.75, color=col, linewidth=0, label=lab)
        bottom = bottom + vals
    # direct labels at the right end (hour 23), centred in each segment
    acc = 0.0
    for vals, _col, lab in comps:
        ax1.annotate(lab, (x[-1] + 0.45, acc + vals[-1] / 2), xytext=(4, 0), textcoords="offset points",
                     va="center", ha="left", fontsize=8, color=pal["text_secondary"], annotation_clip=False)
        acc += vals[-1]
    peak = int(np.argmax(bottom))
    ax1.annotate(f"{bottom[peak]:.3f}", (x[peak], bottom[peak]), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, color=pal["text"])
    low = int(np.argmin(bottom))
    ax1.annotate(f"{bottom[low]:.3f}", (x[low], bottom[low]), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8, color=pal["text"])
    ax1.set_xlim(-0.6, 23.6)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_ylabel("cost per fill (USD/oz)")
    ax1.set_xlabel("bar open hour (UTC)")
    ax1.set_title("Median cost of one market fill, by component")
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# 4. intrabar ambiguity
# ---------------------------------------------------------------------------------------
@register(PAGE, "intrabar-ambiguity", needs=("data_store",))
def intrabar_ambiguity(pal):
    """An ambiguous real bar (stop and take-profit both inside its range) and how often it happens."""
    b = bars("H1")
    atr = average_true_range(b, n=14).shift(1)          # ATR known at the previous close
    when = pd.Timestamp(_FAST, tz="UTC")
    row = b.loc[when]
    a = float(atr.loc[when])
    k_demo = 2.0
    o, h, lo, c = (float(row[k]) for k in ("open", "high", "low", "close"))
    sl, tp = o - k_demo * a, o + k_demo * a
    ex = intrabar_exit(+1.0, o, h, lo, sl, tp)

    blue, orange, red = pal["categorical"][0], pal["categorical"][1], pal["categorical"][7]
    from matplotlib.figure import Figure

    fig = Figure(figsize=FIGSIZE, layout="constrained")
    ax0, ax1 = fig.subplots(1, 2, gridspec_kw={"width_ratios": (1.3, 1)})
    for ax in (ax0, ax1):
        style_axes(ax)
    fig.get_layout_engine().set(wspace=0.08)
    # left: the two possible intrabar paths, then the bar itself
    xc, w = 2.0, 0.2
    ax0.axhline(sl, color=red, linewidth=1.5)
    ax0.axhline(tp, color=blue, linewidth=1.5)
    xa = np.array([0.15, 0.4, 0.65, 0.9])
    xb = xa + 0.85
    ax0.plot(xa, [o, h, lo, c], color=blue, linewidth=1.5, linestyle=(0, (3, 2)))
    ax0.plot(xb, [o, lo, h, c], color=orange, linewidth=1.5)
    ax0.annotate("A: high first\n-> take-profit", (xa[1], h), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=7.5, color=pal["text_secondary"])
    ax0.annotate("B: low first\n-> stop", (xb[1], lo), xytext=(0, -3), textcoords="offset points",
                 ha="center", va="top", fontsize=7.5, color=pal["text_secondary"])
    ax0.plot([xc, xc], [lo, h], color=pal["text_secondary"], linewidth=1.4, zorder=2)
    ax0.add_patch(_rect(xc - w, min(o, c), 2 * w, abs(c - o), pal["text_secondary"], c < o, pal))
    ax0.annotate("OHLC bar", (xc, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                 fontsize=7.5, color=pal["text_secondary"])
    ax0.annotate(f"take-profit {tp:,.2f}\n(open + 2 ATR)", (xc + w + 0.08, tp), xytext=(0, 3),
                 textcoords="offset points", ha="left", va="bottom", fontsize=8, color=pal["text"])
    ax0.annotate(f"stop {sl:,.2f}\n(open - 2 ATR)", (xc + w + 0.08, sl), xytext=(0, -3),
                 textcoords="offset points", ha="left", va="top", fontsize=8, color=pal["text"])
    ax0.annotate(f"intrabar_exit -> {ex[1]} at {ex[0]:,.2f}", (0.0, lo), xytext=(0, -34),
                 textcoords="offset points", ha="left", va="top", fontsize=8.5, color=pal["text"])
    ax0.set_xlim(-0.05, 3.55)
    pad = (h - lo) * 0.18
    ax0.set_ylim(lo - 1.9 * pad, h + 1.2 * pad)
    ax0.set_xticks([])
    ax0.set_ylabel("USD per ounce (mid)")
    ax0.set_title("Long entered at the open of the\n2024-07-05 12:00 UTC bar (US payrolls)", fontsize=10)

    # right: how often a bracket of +/- k ATR around the open is ambiguous
    ks = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    up = (b["high"] - b["open"]).to_numpy()
    dn = (b["open"] - b["low"]).to_numpy()
    av = atr.to_numpy()
    ok = np.isfinite(av)
    both = np.array([np.mean(((up >= k * av) & (dn >= k * av))[ok]) for k in ks])
    either = np.array([np.mean(((up >= k * av) | (dn >= k * av))[ok]) for k in ks])
    l1, = ax1.plot(ks, either, color=pal["text_secondary"], marker="o", markersize=4)
    l2, = ax1.plot(ks, both, color=red, marker="o", markersize=4)
    ax1.set_yscale("log")
    ax1.set_ylim(5e-5, 1.4)
    from matplotlib.ticker import FuncFormatter

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * 100:g}%"))
    ax1.annotate("bar touches a level", (ks[1], either[1]), xytext=(6, 4), textcoords="offset points",
                 fontsize=8, color=pal["text"])
    ax1.annotate("bar touches both:\nthe order is unknown", (ks[2], both[2]), xytext=(8, 2),
                 textcoords="offset points", fontsize=8, color=pal["text"], va="bottom")
    for k, v in ((1.0, both[3]), (3.0, both[-1])):
        ax1.annotate(f"{v * 100:.2g}%", (k, v), xytext=(0, -10), textcoords="offset points", ha="center",
                     va="top", fontsize=8, color=pal["text_secondary"])
    ax1.set_xlabel("bracket half-width k (multiples of ATR(14))")
    ax1.set_ylabel("share of H1 bars (log scale)")
    ax1.set_title("Stop and take-profit at open +/- k ATR:\nhow often is the bar ambiguous?", fontsize=10)
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# 5. financing: the real benchmark and the carry of one lot
# ---------------------------------------------------------------------------------------
@functools.cache
def _carry_frame() -> pd.DataFrame:
    fin = FinancingModel()
    b = bars("H1")
    curve = fin.curve(macro(["fedfunds"]))
    days = pd.date_range("2012-01-02", "2024-12-31", freq="D", tz="UTC")
    roll = days[days.weekday < 5] + pd.Timedelta(hours=XAUUSD.rollover_hour_utc)
    rns = roll.as_unit("ns").asi8
    r = fin.benchmark_ns(rns, curve)
    avail = pd.DatetimeIndex(b["available_at"]).as_unit("ns").asi8
    pos = np.searchsorted(avail, rns, side="right") - 1
    price = b["close"].to_numpy(dtype=float)[np.clip(pos, 0, None)]
    price = np.where(pos >= 0, price, np.nan)
    long_ = np.array([fin.nightly(+1.0, p, x) for p, x in zip(price, r, strict=True)])
    short = np.array([fin.nightly(-1.0, p, x) for p, x in zip(price, r, strict=True)])
    return pd.DataFrame({"rate": r, "price": price, "long": long_, "short": short}, index=roll).dropna()


@register(PAGE, "financing-carry", needs=("data_store",))
def financing_carry(pal):
    """Fed funds as of each rollover (2012-2024) and the nightly financing of one lot long and short."""
    fin = FinancingModel()
    df = _carry_frame()
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1) = panels(2, heights=(1, 1.15), figsize=FIGSIZE_TALL)
    l_r, = ax0.plot(df.index, df["rate"], color=pal["text_secondary"], label="fed funds r")
    l_l, = ax0.plot(df.index, df["rate"] + fin.markup_long, color=blue, label="long pays r + 2.5%")
    l_s, = ax0.plot(df.index, df["rate"] - fin.markup_short, color=orange, label="short earns r - 2.5%")
    zero_line(ax0)
    percent_axis(ax0, decimals=0)
    for ln in (l_r, l_l, l_s):
        label_line_end(ax0, ln, text=ln.get_label())
    ax0.set_ylabel("annual rate")
    ax0.set_title("Benchmark rate as of each 21:00 UTC rollover, and what each side pays")

    ll, = ax1.plot(df.index, df["long"], color=blue, label="1 lot long")
    ls, = ax1.plot(df.index, df["short"], color=orange, label="1 lot short")
    zero_line(ax1)
    label_line_end(ax1, ll, fmt="{:+,.0f}")
    label_line_end(ax1, ls, fmt="{:+,.0f}")
    when = pd.Timestamp("2014-01-07 21:00", tz="UTC")
    ax1.annotate("while r < 2.5% the short pays too", (when, float(df["short"].loc[when])),
                 xytext=(when - pd.Timedelta(days=500), -27), textcoords="data", fontsize=8,
                 color=pal["text_secondary"], va="top", ha="left",
                 arrowprops={"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.8})
    ax1.set_ylabel("USD per financing night\n(+ received, - paid)")
    ax1.set_xlabel("rollover date (21:00 UTC)")
    ax1.set_title("Financing of one lot (100 oz) for one night: -lots x 100 x price x rate / 360")
    ax0.set_xlim(df.index[0], df.index[-1] + pd.Timedelta(days=760))
    ax1.set_xticks([pd.Timestamp(f"{y}-01-01", tz="UTC") for y in range(2012, 2025, 2)],
                   [str(y) for y in range(2012, 2025, 2)])
    source_note(fig, "Data: " + macro_label(["fedfunds"]) + "; gold mid close at the rollover, "
                + source_label("H1"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# 6. the triple roll in one real week
# ---------------------------------------------------------------------------------------
@register(PAGE, "triple-roll", needs=("data_store",))
def triple_roll(pal):
    """Nights charged per weekday and the cumulative financing of 1 lot long over one real week."""
    b = bars("H1", start="2024-03-03", end="2024-03-11 01:00")
    costs = CostModel(spread_multiplier=0.0, min_spread=0.0, slippage_fixed=0.0, slippage_range_frac=0.0)
    sim = ExecutionSimulator(b, costs=costs, rates=macro(["fedfunds"]))
    while not sim.done:
        sim.step(1.0)
    res = sim.result(compute_metrics=False)
    swap = res.costs["swap"]
    cum = swap.cumsum()
    t_end = pd.DatetimeIndex(b["available_at"])

    from matplotlib.figure import Figure

    blue = pal["categorical"][0]
    fig = Figure(figsize=FIGSIZE, layout="constrained")
    ax0, ax1 = fig.subplots(1, 2, gridspec_kw={"width_ratios": (1, 2.6)})
    for ax in (ax0, ax1):
        style_axes(ax)
    w = _weekday_weights(XAUUSD)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax0.bar(range(7), w, color=[blue if v != 3 else pal["categorical"][1] for v in w], width=0.7, linewidth=0)
    for i, v in enumerate(w):
        ax0.annotate(f"{v:g}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=9, color=pal["text"])
    ax0.set_xticks(range(7), names)
    ax0.set_ylim(0, 3.6)
    ax0.set_ylabel("financing nights charged")
    ax0.set_title("Nights per rollover\n(21:00 UTC)", fontsize=10)

    ax1.step(t_end, cum.values, where="post", color=blue)
    steps = swap[swap != 0]
    for ts, v in steps.items():
        x = pd.Timestamp(b.loc[ts, "available_at"])
        y = float(cum.loc[ts])
        ax1.annotate(f"{signed(v, '{:+,.1f}')}\n{x:%a}", (x, y), xytext=(4, 4), textcoords="offset points",
                     fontsize=8, color=pal["text"], va="bottom")
    zero_line(ax1)
    ax1.set_ylabel("cumulative financing (USD)")
    ax1.set_xlabel("bar close time, 3 to 11 March 2024 (UTC)")
    import matplotlib.dates as mdates

    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax1.set_title(f"1 lot long held all week: {signed(cum.iloc[-1], '{:+,.1f}')} USD", fontsize=10)
    fig.suptitle("Wednesday counts three nights (T+2 settlement spans the weekend)")
    source_note(fig, "Data: " + macro_label(["fedfunds"]) + "; "
                + source_label("H1", start="2024-03-03", end="2024-03-11", years=False), pal)
    return fig


# ---------------------------------------------------------------------------------------
# 7. gross to net: a real walk-forward
# ---------------------------------------------------------------------------------------
def _waterfall(ax, items, pal, *, total_label="net"):
    pos_c = pal["diverging"][5]
    neg_c = pal["diverging"][1]
    tot_c = pal["text_secondary"]
    level = 0.0
    xs = []
    for i, (lab, val, kind) in enumerate(items):
        if kind == "total":
            bottom, height, col = 0.0, val, tot_c
            level = val
        else:
            bottom, height = level, val
            col = pos_c if val >= 0 else neg_c
            level += val
        ax.bar(i, height, bottom=bottom, width=0.62, color=col, linewidth=0)
        top = max(bottom, bottom + height)
        lo = min(bottom, bottom + height)
        txt = signed(val, "{:+,.0f}") if abs(val) < 1000 else signed(val / 1000.0, "{:+,.1f}k")
        if height >= 0:
            ax.annotate(txt, (i, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=pal["text"])
        else:
            ax.annotate(txt, (i, lo), xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                        fontsize=8, color=pal["text"])
        if i < len(items) - 1 and kind != "total":
            ax.plot([i + 0.31, i + 0.69], [level, level], color=pal["axis"], linewidth=0.9)
        xs.append(lab)
    ax.set_xticks(range(len(items)), xs, fontsize=8)
    zero_line(ax)


def _book_totals(book: str) -> dict | None:
    ts = run_table(RUN, f"books/{book}/timeseries.parquet")
    if ts is None:
        return None
    return {"gross": float(ts["pnl_price"].sum()), "spread": -float(ts["cost_spread"].sum()),
            "slippage": -float(ts["cost_slippage"].sum()), "commission": -float(ts["cost_commission"].sum()),
            "swap": float(ts["cost_swap"].sum()), "net": float(ts["pnl_net"].sum()),
            "equity0": float(ts["equity"].iloc[0])}


@register(PAGE, "cost-waterfall", needs=(f"runs/{RUN}",))
def cost_waterfall(pal):
    """Gross price P&L to net P&L for the combined trend book and buy-and-hold (OOS 2015-2024)."""
    fig, axes = figure(1, 2, figsize=FIGSIZE)
    fig.get_layout_engine().set(wspace=0.06)
    for ax, book, title in ((axes[0], "combined", "Combined trend book (7 strategies)"),
                            (axes[1], "benchmark", "Buy-and-hold benchmark (1x notional)")):
        t = _book_totals(book)
        items = [("gross\nprice P&L", t["gross"], "step"), ("spread", t["spread"], "step"),
                 ("slippage", t["slippage"], "step"), ("financing", t["swap"], "step"),
                 ("net", t["net"], "total")]
        if abs(t["commission"]) > 0:
            items.insert(3, ("commission", t["commission"], "step"))
        _waterfall(ax, items, pal)
        ax.set_title(title, fontsize=10)
        lo = min(0.0, min(np.cumsum([v for _, v, k in items if k != "total"])), t["net"])
        hi = max(np.cumsum([v for _, v, k in items if k != "total"]).max(), t["net"], 0.0)
        span = hi - lo
        ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
        from matplotlib.ticker import FuncFormatter

        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: signed(v / 1000.0, "{:g}k") if v else "0"))
    axes[0].set_ylabel("USD on 100k USD starting equity")
    fig.suptitle("From gross to net: where the money went, out of sample")
    source_note(fig, "Data: " + run_label(RUN), pal)
    return fig


# ---------------------------------------------------------------------------------------
# 8. paper broker parity
# ---------------------------------------------------------------------------------------
@functools.cache
def _parity() -> pd.DataFrame:
    b = bars("H1", start="2024-01-01", end="2024-06-30")
    rates = macro(["fedfunds"])
    costs = CostModel(commission_per_lot=3.5)
    sim = ExecutionSimulator(b, costs=costs, rates=rates)
    feed = ReplayFeed(b)
    delay = pd.Timedelta(seconds=5)
    clock = SimulatedClock(b["available_at"].iloc[0] + delay)
    pb = PaperBroker(feed, costs=costs, clock=clock, rates=rates)
    rng = np.random.default_rng(11)
    magic = 77
    target, stop = 0.0, None
    rows = []

    def send(cid, side, lots, **kw):
        r = pb.place_order(OrderRequest(client_id=cid, symbol="XAUUSD", side=Side(side), lots=lots,
                                        time=clock.now(), magic=magic, **kw))
        assert r.ok, r

    for t in range(len(b) - 1):
        clock.advance_to(b["available_at"].iloc[t] + delay)
        rows.append((b["available_at"].iloc[t], sim.equity, pb.account().equity))
        if t % 12 == 0:
            target = float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        cur = net_lots(pb.positions("XAUUSD", magic))
        if feed.quote(clock.now()) is None:             # market closed: fill at the next open
            clock.advance_to(b.index[t + 1] + delay)
        entry = target != 0 and np.sign(target) != np.sign(cur)
        sd = 8.0 if entry else None
        d = round(target - cur, 8)
        if abs(d) > 1e-9:
            if cur != 0 and (target == 0 or np.sign(target) != np.sign(cur)):
                send(f"c{t}", -int(np.sign(cur)), abs(cur), position_ticket=pb.positions("XAUUSD", magic)[0].ticket)
                if target != 0:
                    send(f"o{t}", int(np.sign(target)), abs(target), sl_distance=sd)
            else:
                send(f"o{t}", int(np.sign(d)), abs(d), sl_distance=sd if cur == 0 else None)
        keep = stop if (not entry and target != 0) else None
        res = sim.step(target, stop_distance=sd, stop_price=keep)
        stop = res.stop_price
    clock.advance_to(b["available_at"].iloc[-1] + delay)
    rows.append((b["available_at"].iloc[-1], sim.equity, pb.account().equity))
    out = pd.DataFrame(rows, columns=["time", "simulator", "paper"]).set_index("time")
    out.attrs["n_stops"] = int((sim.result(compute_metrics=False).trades["exit_reason"] == "stop").sum())
    out.attrs["n_bars"] = len(b)
    return out


@register(PAGE, "paper-parity", needs=("data_store",))
def paper_parity(pal):
    """Equity of the ExecutionSimulator and the PaperBroker on the same targets (real H1, H1 2024)."""
    df = _parity()
    diff = (df["paper"] - df["simulator"]).abs()
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1) = panels(2, heights=(2, 1), figsize=FIGSIZE)
    ax0.plot(df.index, df["simulator"], color=blue, label="ExecutionSimulator", linewidth=3.0)
    ax0.plot(df.index, df["paper"], color=orange, label="PaperBroker", linewidth=1.2)
    add_legend(ax0, loc="upper left")
    from matplotlib.ticker import StrMethodFormatter

    ax0.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax0.set_ylabel("equity (USD)")
    ax0.set_title(f"Same random targets, same costs and stops ({df.attrs['n_stops']} stop-outs): "
                  "two equity paths, one line")
    ax1.plot(df.index, diff.values, color=pal["text_secondary"], linewidth=1.0)
    ax1.set_ylim(0, max(float(diff.max()) * 1.3, 1e-12))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.annotate(f"max {diff.max():.1e} USD", (diff.idxmax(), float(diff.max())), xytext=(6, 0),
                 textcoords="offset points", va="center", fontsize=8, color=pal["text"])
    ax1.set_ylabel("|difference| (USD)")
    ax1.set_xlabel("decision time (UTC)")
    source_note(fig, "Data: " + source_label("H1", start="2024-01-01", end="2024-06-30", years=False)
                + "; " + macro_label(["fedfunds"]), pal)
    return fig

