"""Figures for docs/data.md: point-in-time timing, macro publication lags, resampling rules,
Dukascopy bid/ask construction, spreads, coverage and the event calendar.

Each ``@register``-ed ``draw(pal)`` becomes docs/_assets/figures/data/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only data
"""

from __future__ import annotations

import datetime as dt
import functools

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_SHORT,
    FIGSIZE_TALL,
    REPO,
    bars,
    ink_on,
    macro,
    panels,
    register,
    source_label,
    source_note,
)

from aurum.viz.style import (
    add_legend,
    figure,
    percent_axis,
    sequential_cmap,
    shade_span,
    style_axes,
)

PAGE = "data"

#: Raw Dukascopy minute files for the NFP release of Friday 2024-03-08 (the month in the
#: path is zero-based: 02 = March). Gitignored like data_store/; the figure is skipped
#: (committed PNGs kept) when they are missing.
NFP_DAY = dt.date(2024, 3, 8)
NFP_DIR = "cache/dukascopy/XAUUSD/2024/02/08"
NFP_FILES = (f"{NFP_DIR}/BID_candles_min_1.bi5", f"{NFP_DIR}/ASK_candles_min_1.bi5")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _hhmm(ax, *, fmt: str = "%H:%M") -> None:
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz="UTC"))


def _bar(ax, start, end, y, color, *, height=0.56, pad_minutes=4.0, alpha=1.0, hatch=None, edge=None):
    """A horizontal bar from ``start`` to ``end`` (timestamps) on row ``y``."""
    import matplotlib.dates as mdates

    x0 = mdates.date2num(start)
    w = mdates.date2num(end) - x0 - pad_minutes / 1440.0
    kw = {"facecolors": color, "alpha": alpha}
    if hatch is not None:
        kw.update(hatch=hatch, edgecolor=edge, linewidth=0.0)
    ax.broken_barh([(x0, w)], (y - height / 2, height), **kw)
    return x0, w


# ---------------------------------------------------------------------------------------
# point-in-time timing
# ---------------------------------------------------------------------------------------
@register(PAGE, "pit-timeline", needs=("data_store",))
def pit_timeline(pal):
    """H1, H4 and D1 (New York close) bars around the 2024-03-04 daily break, with available_at."""
    import matplotlib.dates as mdates

    lo, hi = _ts("2024-03-04 16:00"), _ts("2024-03-05 03:00")
    h1 = bars("H1", start=lo, end="2024-03-05 01:59")
    h4 = bars("H4", start=lo, end="2024-03-05 00:00")
    d1 = bars("D1_nyclose", start="2024-03-03 22:00", end="2024-03-04 22:00")
    blue, orange, violet = pal["categorical"][0], pal["categorical"][1], pal["categorical"][6]
    fig, ax = figure(figsize=(8.0, 4.2))
    style_axes(ax, grid=None)
    rows = {"D1 (NY close)": 0.0, "H4": 1.0, "H1": 2.0}

    # daily maintenance break: no H1 bar opens at 22:00 UTC (17:00 New York, US winter time)
    shade_span(ax, _ts("2024-03-04 22:00"), _ts("2024-03-04 23:00"), "daily break")

    for t, r in h1.iterrows():
        a = r["available_at"]
        _bar(ax, t, a, rows["H1"], blue)
        ax.text(mdates.date2num(t + (a - t) / 2) - 2 / 1440, rows["H1"], t.strftime("%H"),
                ha="center", va="center", fontsize=7.5, color=ink_on(blue))
    for frame, row, color, name in ((h4, rows["H4"], orange, "H4"), (d1, rows["D1 (NY close)"], violet, "D1")):
        for t, r in frame.iterrows():
            a = r["available_at"]
            start = max(t, lo)
            forming = a > hi
            _bar(ax, start, min(a, hi + pd.Timedelta(minutes=4)), row, color, alpha=0.45 if forming else 1.0)
            fmt = "%a %H:%M" if name == "D1" else "%H:%M"
            if forming:
                txt = f"{name} {t:{fmt}} to {a:{fmt}}\nforming: not usable yet"
            else:
                txt = f"{name} {t:{fmt}} to {a:{fmt}}\navailable {a:{fmt}}"
            ax.text(mdates.date2num(start) + 8 / 1440, row, txt, ha="left", va="center", fontsize=8,
                    color=pal["text"] if forming else ink_on(color), linespacing=1.3)

    # HTF bars become visible to the first H1 decision at or after their available_at
    for frame, row in ((h4, rows["H4"]), (d1, rows["D1 (NY close)"])):
        for _t, r in frame.iterrows():
            a = r["available_at"]
            if not (lo < a <= hi):
                continue
            seer = h1.index[pd.DatetimeIndex(h1["available_at"]) >= a]
            if not len(seer):
                continue
            ax.annotate("", xy=(mdates.date2num(a), rows["H1"] - 0.32), xytext=(mdates.date2num(a), row + 0.3),
                        arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0,
                                    "shrinkA": 0, "shrinkB": 0})

    # decide at the close of t, fill at the open of t+1 (across the break: 23:00)
    t_last = h1.index[h1.index < _ts("2024-03-04 22:00")][-1]
    a_last = h1.loc[t_last, "available_at"]
    t_next = h1.index[h1.index > t_last][0]
    ax.annotate(f"bar {t_last:%H:%M} closes at {a_last:%H:%M}: decide there,\nfill at the next open, {t_next:%H:%M}",
                xy=(mdates.date2num(t_next), rows["H1"] + 0.3), xytext=(mdates.date2num(t_next) + 1.4 / 24, 2.75),
                ha="left", va="center", fontsize=8, color=pal["text_secondary"],
                arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})

    ax.set_ylim(-0.6, 3.15)
    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(hi))
    _hhmm(ax)
    ax.set_xlabel("time (UTC), Monday 2024-03-04 to Tuesday 2024-03-05")
    ax.set_title("A bar is labelled by its open and usable from its close (available_at)")
    source_note(fig, "Data: " + source_label("H1", start=lo, end="2024-03-05", years=False), pal)
    return fig


@register(PAGE, "asof-join", needs=("data_store",))
def asof_join_step(pal):
    """The 10-year real yield as an H1 decision sees it (as-of join) vs a join on the observation date."""
    from aurum.data.pit import asof_join

    real = macro("real10y")["real10y"]
    h1 = bars("H1", start="2024-01-10 12:00", end="2024-01-18 23:59")
    t = pd.DatetimeIndex(h1["available_at"])
    pit = asof_join(h1["available_at"], real, columns=["value"])["value"].to_numpy()
    naive_frame = real[["value"]].copy()
    naive_frame["available_at"] = naive_frame.index          # WRONG on purpose: the date itself
    naive = asof_join(h1["available_at"], naive_frame, columns=["value"])["value"].to_numpy()

    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, ax = figure(figsize=(8.0, 4.0))
    shade_span(ax, _ts("2024-01-13"), _ts("2024-01-15"), "weekend")
    shade_span(ax, _ts("2024-01-15"), _ts("2024-01-16"), "MLK day")
    ax.step(t, naive, where="post", color=orange, label="joined on observation date (leaks)")
    ax.step(t, pit, where="post", color=blue, label="asof_join on available_at")
    rows = real.loc["2024-01-10":"2024-01-18"]
    for _d, r in rows.iterrows():
        a = r["available_at"]
        if a > t[-1]:
            continue
        ax.plot([a], [r["value"]], marker="o", linestyle="none", color=blue, zorder=4)
    d12 = rows.loc["2024-01-12"]
    ax.annotate(f"Fri 12 Jan print {d12['value']:.2f}%\npublished Tue 16 Jan 21:30 UTC",
                xy=(d12["available_at"], d12["value"]), xytext=(_ts("2024-01-13 08:00"), 1.725),
                fontsize=8, color=pal["text_secondary"], va="center",
                arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    ax.annotate("the leaky join uses it\nfrom Fri 12 Jan 00:00",
                xy=(_ts("2024-01-12 01:00"), d12["value"]), xytext=(_ts("2024-01-11 02:00"), 1.71),
                fontsize=8, color=pal["text_secondary"], va="center", ha="left",
                arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    add_legend(ax, loc="upper right")
    ax.set_ylabel("10-year real yield (%)")
    ax.set_xlabel("H1 decision time, available_at (UTC)")
    ax.set_title("FRED DFII10 on H1 bars: what each decision could actually know")
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz="UTC"))
    source_note(fig, "Data: FRED DFII10 (publication-lagged); "
                + source_label("H1", start="2024-01-10", end="2024-01-18", years=False), pal)
    return fig


@register(PAGE, "macro-lags", needs=("data_store",))
def macro_lags(pal):
    """Observation date vs available_at for six real macro series over the MLK-day week of 2024."""
    import matplotlib.dates as mdates

    series = [("fedfunds", "FRED DFF"), ("real10y", "FRED DFII10"), ("us10y", "Yahoo ^TNX"),
              ("spx", "Yahoo ^GSPC"), ("dxy", "Yahoo DX-Y.NYB"), ("gold_fut", "Yahoo GC=F")]
    frames = macro([s for s, _ in series])
    rule = {"fedfunds": 0, "real10y": 0, "us10y": 1, "spx": 1, "dxy": 2, "gold_fut": 2}
    rule_names = ["FRED: next US business day + 21:30", "Yahoo cash index: date + 21:30",
                  "Yahoo future / ICE: date + 22:30"]
    colors = [pal["categorical"][0], pal["categorical"][1], pal["categorical"][6]]
    fig, ax = figure(figsize=(8.0, 4.6))
    shade_span(ax, _ts("2024-01-13"), _ts("2024-01-15"))
    shade_span(ax, _ts("2024-01-15"), _ts("2024-01-16"))
    for x, txt in ((_ts("2024-01-14"), "weekend"), (_ts("2024-01-15 12:00"), "MLK day")):
        ax.text(mdates.date2num(x), -0.62, txt, ha="center", va="center", fontsize=8, color=pal["text_muted"])
    used = set()
    n = len(series)
    for i, (name, _label) in enumerate(series):
        y = n - 1 - i
        f = frames[name].loc["2024-01-10":"2024-01-18"]
        c = colors[rule[name]]
        for k, (d, r) in enumerate(f.iterrows()):
            a = r["available_at"]
            yy = y + (-0.22, 0.0, 0.22)[k % 3]
            x0, x1 = mdates.date2num(d), mdates.date2num(a)
            ax.plot([x0, x1], [yy, yy], color=c, linewidth=1.2, alpha=0.7, solid_capstyle="butt")
            ax.plot([x0], [yy], marker="|", markersize=7, color=c, markeredgecolor=c, markeredgewidth=1.5,
                    linestyle="none")
            lab = rule_names[rule[name]] if rule[name] not in used else "_nolegend_"
            used.add(rule[name])
            ax.plot([x1], [yy], marker="o", markersize=5.5, linestyle="none", color=c, label=lab, zorder=4)
    ax.set_yticks(range(n), [f"{nm}  ({lab})" for nm, lab in reversed(series)])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, n - 0.3 + 0.8)
    ax.set_xlim(mdates.date2num(_ts("2024-01-10")), mdates.date2num(_ts("2024-01-20")))
    ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz="UTC"))
    style_axes(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (UTC), January 2024.  Tick = observation date (00:00), dot = available_at")
    ax.set_title("Each macro row becomes usable only at its available_at")
    add_legend(ax, loc="upper left", ncol=3, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    source_note(fig, "Data: FRED DFF, DFII10; Yahoo ^TNX, ^GSPC, DX-Y.NYB, GC=F (as stored in data_store/macro)", pal)
    return fig


# ---------------------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------------------
@register(PAGE, "resample-head-tail", needs=("data_store",))
def resample_head_tail(pal):
    """H1 -> D1 (22:00 UTC anchor) on a window that starts and ends mid-day: head and tail rules."""
    import matplotlib.dates as mdates

    from aurum.data.resample import resample_bars

    w = bars("H1", start="2024-03-05 10:00", end="2024-03-08 15:59")
    full = resample_bars(bars("H1", start="2024-02-01", end="2024-03-31"), "D1", daily_anchor_hour_utc=22)
    new = resample_bars(w, "D1", daily_anchor_hour_utc=22)
    old = resample_bars(w, "D1", daily_anchor_hour_utc=22, complete_from="2000-01-01")  # trusts the head
    first = w.index[0]
    b0 = _ts("2024-03-04 22:00")
    buckets = [b0 + pd.Timedelta(days=i) for i in range(4)]
    blue, violet = pal["categorical"][0], pal["categorical"][6]

    fig, (ax0, ax1) = panels(2, heights=(2.4, 1.6), figsize=(8.0, 5.0))
    for b in buckets[1:]:
        for ax in (ax0, ax1):
            ax.axvline(b, color=pal["axis"], linewidth=0.75, zorder=0.5)
    for b, r in new.iterrows():
        e = b + pd.Timedelta(days=1)
        ax0.fill_between([b, e], r["low"], r["high"], color=violet, alpha=0.16, linewidth=0, zorder=0.6)
        for x0, x1, v in ((b, b + pd.Timedelta(hours=4), r["open"]), (e - pd.Timedelta(hours=4), e, r["close"])):
            ax0.plot([x0, x1], [v, v], color=violet, linewidth=2.0, solid_capstyle="butt")
    b1 = new.index[0]
    ax0.text(b1 + pd.Timedelta(hours=1), new.loc[b1, "high"] + 1.0, "D1 bar: low-high band,\nopen and close ticks",
             fontsize=8, color=pal["text_secondary"], va="bottom")
    ax0.plot(w.index, w["close"], color=blue, label="H1 close")
    ax0.set_ylabel("USD per ounce")
    ax0.set_title("Resampling H1 to D1: only complete daily buckets are emitted")

    style_axes(ax1, grid=None)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    data_end = w["available_at"].max()
    for b in buckets:
        e = b + pd.Timedelta(days=1)
        s_vis, e_vis = max(b, first), min(e, data_end)
        if b in new.index:
            _bar(ax1, b, e, 0.55, violet, height=0.5, pad_minutes=20)
            a = new.loc[b, "available_at"]
            ax1.text(mdates.date2num(b + (e - b) / 2), 0.55, f"kept, available\n{a:%a %d %b %H:%M}",
                     ha="center", va="center", fontsize=8, color=ink_on(violet))
            ax1.annotate("", xy=(mdates.date2num(a), 0.03), xytext=(mdates.date2num(a), 0.29),
                         arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
        else:
            _bar(ax1, b, e, 0.55, pal["grid"], height=0.5, pad_minutes=20)
            _bar(ax1, s_vis, e_vis, 0.55, violet, height=0.5, pad_minutes=20, alpha=0.35)
            if b < first:
                o_old, o_full = old.loc[b, "open"], full.loc[b, "open"]
                txt = (f"head: data starts {first:%H:%M}\ndropped (old rule: open {o_old:,.1f}\n"
                       f"vs full-history {o_full:,.1f})")
            else:
                txt = f"tail: data ends {data_end:%H:%M}\nforming bucket dropped"
            ax1.text(mdates.date2num(b + (e - b) / 2), 0.55, txt, ha="center", va="center", fontsize=7.5,
                     color=pal["text"])
    ax1.set_xlim(mdates.date2num(b0 - pd.Timedelta(hours=1)), mdates.date2num(buckets[-1] + pd.Timedelta(days=1, hours=1)))
    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[22], tz="UTC"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M", tz="UTC"))
    ax1.set_xlabel("D1 bucket boundaries at 22:00 UTC (New York close); arrows = available_at")
    source_note(fig, "Data: " + source_label("H1", start="2024-03-05", end="2024-03-08", years=False), pal)
    return fig


@functools.cache
def _window_check(n_windows: int = 40, days: int = 30, seed: int = 0) -> pd.DataFrame:
    """First D1 bar of random H1 windows (2014-2024) vs the full-history D1 bar, old vs new head rule."""
    from aurum.data.resample import resample_bars

    h1 = bars("H1", start="2014-01-01")
    full = resample_bars(h1, "D1", daily_anchor_hour_utc=22)
    rng = np.random.default_rng(seed)
    ks = np.sort(rng.integers(0, len(h1) - 24 * (days + 10), n_windows))
    cols = ["open", "high", "low", "close", "volume", "spread"]
    rows = []
    for k in ks:
        win = h1.iloc[k:k + 24 * days]
        out = {"start": win.index[0]}
        for rule, kw in (("new", {}), ("old", {"complete_from": "2000-01-01"})):
            d = resample_bars(win, "D1", daily_anchor_hour_utc=22, **kw)
            common = d.index.intersection(full.index)
            same = np.isclose(d.loc[common, cols].to_numpy(), full.loc[common, cols].to_numpy(), rtol=0, atol=0)
            out[f"{rule}_bad"] = int((~same.all(axis=1)).sum())
            out[f"{rule}_open_err"] = abs(d["open"].iloc[0] / full.loc[d.index[0], "open"] - 1.0)
            out[f"{rule}_n"] = len(d)
        rows.append(out)
    return pd.DataFrame(rows).set_index("start")


@register(PAGE, "resample-window-check", needs=("data_store",))
def resample_window_check(pal):
    """Head rule check: first D1 bar of 40 random 30-day H1 windows vs the full-history D1 bar."""
    df = _window_check()
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, ax = figure(figsize=FIGSIZE_SHORT)
    ax.vlines(df.index, 0, df["old_open_err"], color=orange, linewidth=1.0, alpha=0.6)
    n_old = int((df["old_bad"] > 0).sum())
    n_new = int((df["new_bad"] > 0).sum())
    ax.plot(df.index, df["old_open_err"], marker="o", linestyle="none", color=orange,
            label=f"old head rule: {n_old} of {len(df)} windows differ")
    ax.plot(df.index, df["new_open_err"], marker="o", linestyle="none", color=blue,
            label=f"new head rule: {n_new} of {len(df)} windows differ")
    percent_axis(ax, decimals=1)
    ax.set_ylim(-0.0006, df["old_open_err"].max() * 1.35)
    add_legend(ax, loc="upper left")
    ax.set_ylabel("first D1 open error")
    ax.set_xlabel("window start (H1 bar open, UTC)")
    ax.set_title("Does a D1 bar depend on where the H1 history starts?")
    source_note(fig, "Data: " + source_label("H1", start="2014-01-01"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# Dukascopy bid/ask -> mid bars and spread
# ---------------------------------------------------------------------------------------
@register(PAGE, "dukascopy-nfp-minutes", needs=NFP_FILES)
def dukascopy_nfp_minutes(pal):
    """Real Dukascopy minute bid/ask around the 2024-03-08 NFP release: mid path and spread construction."""
    from aurum.data.dukascopy import combine_bid_ask, decode_bi5_candles
    from aurum.data.resample import resample_bars
    from aurum.data.schema import make_bars

    bid = decode_bi5_candles((REPO / NFP_FILES[0]).read_bytes(), NFP_DAY, 1000)
    ask = decode_bi5_candles((REPO / NFP_FILES[1]).read_bytes(), NFP_DAY, 1000)
    mid = combine_bid_ask(bid, ask)
    m1 = make_bars(mid, "M1")
    h1 = resample_bars(m1, "H1")
    lo, hi = _ts("2024-03-08 13:00"), _ts("2024-03-08 13:59")
    win = mid.loc[lo:hi]
    j = bid.join(ask, lsuffix="_bid", rsuffix="_ask").loc[lo:hi]
    blue, orange, violet = pal["categorical"][0], pal["categorical"][1], pal["categorical"][6]

    fig, (ax0, ax1) = panels(2, heights=(2.2, 1.6), figsize=FIGSIZE_TALL)
    release = _ts("2024-03-08 13:30")
    for ax in (ax0, ax1):
        ax.axvline(release, color=pal["axis"], linewidth=0.75, zorder=0.5)
    x = win.index + pd.Timedelta(seconds=30)
    ax0.fill_between(x, win["low"], win["high"], color=pal["grid"], step=None, linewidth=0)
    ax0.plot(x, win["close"], color=blue, label="mid close")
    hb = h1.loc[lo]
    for val, name in ((hb["high"], "H1 high"), (hb["low"], "H1 low")):
        ax0.axhline(val, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 3)))
        ax0.text(hi + pd.Timedelta(minutes=1.5), val, f"{name} {val:,.2f}", va="center", fontsize=8,
                 color=pal["text_secondary"])
    ax0.text(release + pd.Timedelta(minutes=0.7), win["low"].min() + 1, "NFP 13:30 UTC", fontsize=8,
             color=pal["text_secondary"])
    ax0.set_ylabel("USD per ounce (mid)")
    ax0.set_title("Dukascopy minute candles: mid = (bid + ask) / 2, band = mid low to high")

    xs = j.index + pd.Timedelta(seconds=30)
    ax1.plot(xs, j["open_ask"] - j["open_bid"], marker="o", markersize=3.5, markeredgewidth=0,
             linestyle="none", color=orange, label="quoted spread at the minute open")
    ax1.plot(xs, j["close_ask"] - j["close_bid"], marker="o", markersize=3.5, markeredgewidth=0,
             linestyle="none", color=violet, label="quoted spread at the minute close")
    ax1.plot(x, win["spread"], color=blue, label="spread column = their average")
    ax1.axhline(hb["spread"], color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 3)))
    ax1.text(hi + pd.Timedelta(minutes=1.5), hb["spread"], f"H1 spread\n= mean {hb['spread']:.3f}",
             va="center", fontsize=8, color=pal["text_secondary"])
    add_legend(ax1, loc="upper left", fontsize=8)
    ax1.set_ylabel("ask minus bid (USD/oz)")
    ax1.set_xlabel("time (UTC), Friday 2024-03-08")
    _hhmm(ax1)
    ax1.set_xlim(lo, hi + pd.Timedelta(minutes=1))
    source_note(fig, "Data: Dukascopy XAUUSD BID/ASK minute candles (raw cache, decoded with aurum.data.dukascopy)", pal)
    return fig


# ---------------------------------------------------------------------------------------
# spreads, coverage
# ---------------------------------------------------------------------------------------
@register(PAGE, "spread-heatmap", needs=("data_store",))
def spread_heatmap(pal):
    """Median H1 spread by UTC hour and weekday, 2012-2024."""
    h1 = bars("H1")
    idx = h1.index
    tab = (h1["spread"] * 100).groupby([idx.weekday, idx.hour]).median().unstack()
    order = [6, 0, 1, 2, 3, 4]
    tab = tab.reindex(index=order, columns=range(24))
    z = tab.to_numpy(dtype=float)
    fig, ax = figure(figsize=(8.0, 3.6))
    style_axes(ax, grid=None)
    cmap = sequential_cmap(pal["mode"])
    lo, hi = np.nanmin(z), np.nanmax(z)
    im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i in range(z.shape[0]):
        for k in range(z.shape[1]):
            if np.isfinite(z[i, k]):
                ax.text(k, i, f"{z[i, k]:.0f}", ha="center", va="center", fontsize=7,
                        color=ink_on(im.cmap(im.norm(z[i, k]))))
    ax.set_yticks(range(len(order)), ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("hour of the H1 bar open (UTC)")
    ax.set_title("Median Dukascopy spread by hour and weekday (US cents per ounce)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
    cbar.set_label("median spread (US cents/oz)", color=pal["text_secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


@register(PAGE, "spread-history", needs=("data_store",))
def spread_history(pal):
    """Monthly spread distribution of H1 bars, in USD per ounce and in basis points of price."""
    h1 = bars("H1")
    s = h1["spread"]
    bps = s / h1["close"] * 1e4
    g = s.groupby(pd.Grouper(freq="MS"))
    q = pd.DataFrame({"p10": g.quantile(0.10), "p50": g.median(), "p90": g.quantile(0.90)})
    gb = bps.groupby(pd.Grouper(freq="MS"))
    qb = pd.DataFrame({"p10": gb.quantile(0.10), "p50": gb.median(), "p90": gb.quantile(0.90)})
    blue = pal["categorical"][0]
    fig, (ax0, ax1) = panels(2, heights=(1, 1), figsize=FIGSIZE_TALL)
    for ax, d, unit in ((ax0, q, "spread (USD/oz)"), (ax1, qb, "spread (bp of price)")):
        ax.fill_between(d.index, d["p10"], d["p90"], color=blue, alpha=0.18, linewidth=0)
        ln, = ax.plot(d.index, d["p50"], color=blue)
        ax.set_ylabel(unit)
        ax.set_ylim(0, None)
    peak = q["p50"].idxmax()
    ax0.annotate(f"{peak:%b %Y}: median {q.loc[peak, 'p50']:.2f}", xy=(peak, q.loc[peak, "p50"]),
                 xytext=(peak - pd.Timedelta(days=900), q.loc[peak, "p50"] + 0.05), fontsize=8,
                 color=pal["text_secondary"], arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    ax0.set_title("Monthly median H1 spread (line) and its 10th-90th percentile band")
    ax1.set_title("The same spread relative to the gold price")
    ax1.set_xlabel("month")
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


@register(PAGE, "coverage-gaps", needs=("data_store",))
def coverage_gaps(pal):
    """H1 bars per week and the intraweek gaps found by quality_report's classification."""
    from aurum.data.loaders import _classify_gap

    h1 = bars("H1")
    weekly = h1["close"].groupby(pd.Grouper(freq="W-SAT")).size()
    # weeks are labelled by their Saturday end: drop empty weeks and the week the cutoff cuts
    weekly = weekly[(weekly > 0) & (weekly.index <= h1.index[-1].normalize())]
    prev_end = pd.DatetimeIndex(h1["available_at"])[:-1]
    missing = pd.Series(h1.index[1:] - prev_end, index=h1.index[1:])
    big = missing[missing >= pd.Timedelta(hours=3)]
    kinds = pd.Series([_classify_gap(t - d, t) for t, d in big.items()], index=big.index)
    intra = big[kinds == "intraweek"]
    hours = intra.dt.total_seconds() / 3600.0
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1) = panels(2, heights=(1.0, 1.15), figsize=FIGSIZE_TALL)
    ax0.plot(weekly.index, weekly.values, color=blue, linewidth=1.0)
    full_week = int(weekly.mode().iloc[0])
    ax0.axhline(full_week, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 3)))
    ax0.text(weekly.index[len(weekly) // 2], 81, f"most weeks: {full_week} H1 bars (5 sessions of 23 h)",
             fontsize=8, color=pal["text_secondary"], va="bottom", ha="center")
    ax0.set_ylabel("H1 bars per week")
    ax0.set_ylim(78, 124)
    ax0.set_title(f"Coverage: {len(h1):,} H1 bars in {len(weekly)} complete trading weeks")
    ax1.vlines(hours.index, 0, hours.values, color=orange, linewidth=1.0)
    ax1.plot(hours.index, hours.values, marker="o", markersize=4, markeredgewidth=0, linestyle="none", color=orange)
    groups = ((50, np.inf, "Thursday close to Sunday reopen (Good Friday, Christmas, New Year)"),
              (10, 50, "single-day holidays (25 Dec, 1 Jan midweek)"),
              (0, 10, "early closes and short outages"))
    for lo_h, hi_h, txt in groups:
        sel = hours[(hours >= lo_h) & (hours < hi_h)]
        if len(sel):
            y = float(sel.max())
            ax1.text(hours.index[0], y + 2.5, f"{len(sel)} gaps of {sel.min():.0f}-{sel.max():.0f} h: {txt}",
                     fontsize=8, color=pal["text_secondary"], va="bottom", zorder=5,
                     bbox={"boxstyle": "square,pad=0.15", "facecolor": pal["surface"], "edgecolor": "none"})
    ax1.set_ylabel("gap length (hours)")
    ax1.set_title(f"{len(intra)} intraweek gaps of at least 3 bars (quality_report classification)")
    ax1.set_xlabel("gap end (UTC)")
    ax1.set_ylim(0, hours.max() * 1.22)
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# economic calendar
# ---------------------------------------------------------------------------------------
@register(PAGE, "event-study", needs=("data_store",))
def event_study(pal):
    """Median absolute M15 return around rule-based NFP and FOMC times, 2013-2024, vs other days."""
    from aurum.data.calendar import generate_rule_based_calendar

    m15 = bars("M15", start="2013-01-01")
    r = np.log(m15["close"]).diff().abs() * 1e4
    ev = generate_rule_based_calendar("2013-01-01", "2024-12-31")
    offsets = np.arange(-12, 11)  # M15 bars from -3 h to +2.5 h (FOMC + 3 h hits the daily break)
    step = pd.Timedelta(minutes=15)
    out = {}
    ev_days = set(ev["time"].dt.normalize())
    for name in ("NFP", "FOMC"):
        times = ev.loc[ev["name"] == name, "time"]
        vals = []
        for o in offsets:
            idx = pd.DatetimeIndex(times + o * step)
            vals.append(np.nanmedian(r.reindex(idx).to_numpy()))
        out[name] = np.array(vals)
    # baseline: the same UTC clock times on days without an event of that kind
    base = {}
    for name in ("NFP", "FOMC"):
        times = ev.loc[ev["name"] == name, "time"]
        vals = []
        clock = times.dt.hour * 60 + times.dt.minute
        non = r[~r.index.normalize().isin(ev_days)]
        mins = non.index.hour * 60 + non.index.minute
        for o in offsets:
            target = ((clock + o * 15) % 1440).to_numpy()
            vals.append(np.nanmedian(non[np.isin(mins, target)].to_numpy()))
        base[name] = np.array(vals)
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, ax = figure(figsize=FIGSIZE)
    x = offsets * 15
    l1, = ax.plot(x, out["NFP"], color=blue, label="NFP")
    l2, = ax.plot(x, out["FOMC"], color=orange, label="FOMC")
    ax.plot(x, base["NFP"], color=blue, linewidth=1.0, linestyle=(0, (3, 2)), alpha=0.8)
    ax.plot(x, base["FOMC"], color=orange, linewidth=1.0, linestyle=(0, (3, 2)), alpha=0.8)
    ax.plot([], [], color=pal["text_muted"], linewidth=1.0, linestyle=(0, (3, 2)),
            label="dashed: same clock time on days without an event")
    ax.axvline(0, color=pal["axis"], linewidth=0.75, zorder=0.5)
    i0 = int(np.flatnonzero(offsets == 0)[0])
    ax.annotate(f"NFP bar: {out['NFP'][i0]:.0f} bp", xy=(0, out["NFP"][i0]), xytext=(12, out["NFP"][i0]),
                fontsize=8, color=pal["text_secondary"], va="center")
    ifm = int(np.nanargmax(out["FOMC"]))
    ax.annotate(f"FOMC bar: {out['FOMC'][ifm]:.0f} bp", xy=(x[ifm], out["FOMC"][ifm]),
                xytext=(x[ifm] + 12, out["FOMC"][ifm]), fontsize=8, color=pal["text_secondary"], va="center")
    add_legend(ax, loc="upper left")
    ax.set_xticks(range(-180, 151, 30))
    ax.set_xlabel("minutes from the scheduled release (M15 bar open)")
    ax.set_ylabel("median |log return| (bp per M15 bar)")
    ax.set_title("Scheduled releases move gold: absolute M15 returns around NFP and FOMC")
    source_note(fig, "Data: " + source_label("M15", start="2013-01-01")
                + "; aurum rule-based NFP/FOMC calendar", pal)
    return fig
