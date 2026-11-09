"""Figures for docs/features.md: every feature family on real XAUUSD bars, the leakage
(future-perturbation) test, the correlation structure and the pipeline warm-up.

Each ``@register``-ed ``draw(pal)`` becomes docs/_assets/figures/features/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only features
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_SHORT,
    FIGSIZE_TALL,
    bars,
    ink_on,
    market_data,
    panels,
    register,
    signed,
    source_label,
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

PAGE = "features"
GROUP_ORDER = ("returns", "trend", "momentum", "meanrev", "range", "volatility", "microstructure",
               "session", "mtf", "macro", "calendar", "regime")


@functools.cache
def _raw_h1() -> tuple[pd.DataFrame, int, dict[str, int]]:
    """The default pipeline's raw features on real H1 bars 2012-2024 (macro + rule-based events)."""
    from aurum.features import FeaturePipeline

    md = market_data("H1")
    pipe = FeaturePipeline()
    raw = pipe.compute(md)
    return raw, pipe.max_lookback, {g: pipe.group_lookback(g) for g in GROUP_ORDER}


def _end_labels(ax, items, pal, *, min_gap, fmt="{:.2f}", dx_points=6.0):
    """Direct labels at the right end of several lines, spread apart by at least ``min_gap``
    (data units) so they never overlap. ``items`` = [(line, text), ...]."""
    ends = []
    for line, text in items:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata(), dtype=float)
        ok = np.flatnonzero(np.isfinite(y))
        if ok.size:
            i = int(ok[-1])
            ends.append([y[i], x[i], line, text])
    ends.sort(key=lambda e: e[0])
    placed = []
    for e in ends:
        y = e[0] if not placed else max(e[0], placed[-1] + min_gap)
        placed.append(y)
    shift = max(0.0, (placed[-1] - ends[-1][0]) / 2) if placed else 0.0
    from matplotlib.transforms import offset_copy

    # The identity dot sits at the LABEL's height, just right of the line end, so lines that
    # end at almost the same value still get one visible dot each (they used to overlap).
    dot_at = offset_copy(ax.transData, fig=ax.figure, x=3.0, y=0.0, units="points")
    for e, y in zip(ends, placed, strict=True):
        yv, xv, line, text = e
        ax.plot([xv], [y - shift], marker="o", linestyle="none", color=line.get_color(), zorder=5,
                markeredgecolor=pal["surface"], markersize=6, label="_nolegend_", transform=dot_at,
                clip_on=False)
        val = fmt.format(yv).replace("-", "\N{MINUS SIGN}")
        ax.annotate(f"{text}  {val}", (xv, y - shift), xytext=(dx_points + 4.0, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5, color=pal["text"],
                    annotation_clip=False)


# ---------------------------------------------------------------------------------------
# price-based families
# ---------------------------------------------------------------------------------------
@register(PAGE, "trend-features", needs=("data_store",))
def trend_features_fig(pal):
    """Gold H1 close with EMA 20/100, the EMA cross in ATR units and the 100-bar regression t-stat."""
    from aurum.features.technical import ema, trend_features

    md = market_data("H1", macro=False, events=False, start="2023-09-01", end="2024-05-31")
    f = trend_features(md)
    close = md.bars["close"]
    e20, e100 = ema(close, 20), ema(close, 100)
    show = slice("2024-02-01", "2024-05-31")
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1, ax2) = panels(3, heights=(2.2, 1, 1), figsize=(8.0, 6.8))
    ln_c, = ax0.plot(close.loc[show].index, close.loc[show], color=pal["text_muted"], linewidth=1.0)
    ln20, = ax0.plot(e20.loc[show].index, e20.loc[show], color=blue)
    ln100, = ax0.plot(e100.loc[show].index, e100.loc[show], color=orange)
    _end_labels(ax0, [(ln20, "EMA 20"), (ln100, "EMA 100")], pal, min_gap=34, fmt="{:,.0f}")
    ax0.text(close.loc[show].index[5], close.loc[show].max(), "grey: H1 close", fontsize=8,
             color=pal["text_muted"], va="top")
    ax0.set_ylabel("USD per ounce")
    ax0.set_title("Trend family: EMA distances, crosses and regression strength (ATR units)")
    x = f.loc[show]
    ax1.plot(x.index, x["trend_ema_cross_20_100"], color=blue)
    zero_line(ax1)
    ax1.set_ylabel("ATR units")
    ax1.set_title("trend_ema_cross_20_100 = (EMA 20 - EMA 100) / ATR 14", fontsize=10)
    ax2.plot(x.index, x["trend_linreg_t_100"], color=blue)
    zero_line(ax2)
    ax2.set_ylabel("t-stat")
    ax2.set_title("trend_linreg_t_100: signed t-stat of an OLS fit of log price on time", fontsize=10)
    ax2.set_xlabel("date (UTC, H1 bar open)")
    source_note(fig, "Data: " + source_label("H1", start="2024-02-01", end="2024-05-31", years=False), pal)
    return fig


@register(PAGE, "momentum-response", needs=("data_store",))
def momentum_response(pal):
    """The Baz et al. momentum response phi(z) and how often real H1 trends sit on its fading side."""
    from aurum.features.technical import tsmom_response

    raw, warm, _ = _raw_h1()
    z = raw["momentum_tsmom_240"].iloc[warm:].dropna()
    blue = pal["categorical"][0]
    fig, (ax0, ax1) = figure(1, 2, figsize=FIGSIZE_SHORT)
    zz = np.linspace(-6, 6, 601)
    ax0.plot(zz, tsmom_response(zz), color=blue)
    zero_line(ax0)
    zero_line(ax0, axis="x")
    peak = np.sqrt(2.0)
    ax0.plot([peak, -peak], [tsmom_response(peak), tsmom_response(-peak)], marker="o", linestyle="none", color=blue)
    ax0.annotate(f"peak at z = sqrt(2): {tsmom_response(peak):.2f}", xy=(peak, tsmom_response(peak)),
                 xytext=(2.4, 0.72), fontsize=8, color=pal["text_secondary"],
                 arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    ax0.set_ylim(-1.3, 1.35)
    ax0.set_xlabel("z = vol-scaled trailing return")
    ax0.set_ylabel("response phi(z)")
    ax0.set_title("momentum_tsmom_resp_h = phi(z)\nphi(z) = z exp(-z^2 / 4) / 0.89", fontsize=10.5)
    bins = np.linspace(-6, 6, 61)
    ax1.hist(z.clip(-6, 6), bins=bins, color=blue, alpha=0.85)
    for v in (-peak, peak):
        ax1.axvline(v, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 2)))
    share = float((z.abs() > peak).mean())
    ax1.set_title(f"momentum_tsmom_240 on H1 bars\n|z| > sqrt(2) on {share:.0%} of bars", fontsize=10.5)
    ax1.set_xlabel("z (240-bar horizon)")
    ax1.set_ylabel("H1 bars")
    ax1.yaxis.set_major_formatter(_thousands())
    source_note(fig, "Data: " + source_label("H1", start=raw.index[warm]), pal)
    return fig


def _thousands():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}")


@register(PAGE, "meanrev-range", needs=("data_store",))
def meanrev_range(pal):
    """Bollinger bands and the 55-bar Donchian channel on real H1 bars, with %B and channel position."""
    from aurum.features.technical import meanrev_features, range_features

    md = market_data("H1", macro=False, events=False, start="2024-09-01", end="2024-11-30")
    mr, rg = meanrev_features(md), range_features(md)
    b = md.bars
    show = slice("2024-10-14", "2024-11-15")
    c = b["close"].loc[show]
    roll = b["close"].rolling(20, min_periods=20)
    mid, sd = roll.mean().loc[show], roll.std(ddof=0).loc[show]
    hh = b["high"].rolling(55, min_periods=55).max().loc[show]
    ll = b["low"].rolling(55, min_periods=55).min().loc[show]
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1) = panels(2, heights=(1.7, 1), figsize=FIGSIZE_TALL)
    ax0.fill_between(c.index, mid - 2 * sd, mid + 2 * sd, color=blue, alpha=0.16, linewidth=0,
                     label="Bollinger bands (20 bars, 2 sd)")
    ax0.plot(hh.index, hh, color=orange, linewidth=1.2, label="55-bar Donchian high / low")
    ax0.plot(ll.index, ll, color=orange, linewidth=1.2)
    ax0.plot(c.index, c, color=pal["text"], linewidth=1.0, label="H1 close")
    add_legend(ax0, loc="lower left", fontsize=8)
    ax0.set_ylabel("USD per ounce")
    ax0.set_title("Mean reversion and range families: where the close sits in its recent range")
    l1, = ax1.plot(c.index, mr.loc[show, "meanrev_bb_pctb_20"], color=blue, linewidth=1.2)
    l2, = ax1.plot(c.index, rg.loc[show, "range_donchian_pos_55"], color=orange, linewidth=1.2)
    for v in (-0.5, 0.5):
        ax1.axhline(v, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 2)))
    zero_line(ax1)
    _end_labels(ax1, [(l1, "bb_pctb_20"), (l2, "donchian_pos_55")], pal, min_gap=0.22, fmt="{:+.2f}")
    ax1.set_ylabel("position")
    ax1.set_title("meanrev_bb_pctb_20 and range_donchian_pos_55 (dashed: band / channel edges at 0.5)",
                  fontsize=10)
    ax1.set_xlabel("date (UTC, H1 bar open)")
    source_note(fig, "Data: " + source_label("H1", start="2024-10-14", end="2024-11-15", years=False), pal)
    return fig


@register(PAGE, "volatility-estimators", needs=("data_store",))
def volatility_estimators(pal):
    """Close-close, Parkinson and Yang-Zhang 120-bar vols in 2020, the short/long vol ratio and range/cc."""
    from aurum.features.volatility import volatility_features

    md = market_data("H1", macro=False, events=False, start="2019-09-01", end="2020-12-31")
    f = volatility_features(md)
    show = slice("2020-01-01", "2020-12-31")
    x = f.loc[show]
    blue, orange, aqua = pal["categorical"][0], pal["categorical"][1], pal["categorical"][2]
    fig, (ax0, ax1, ax2) = panels(3, heights=(2.0, 1, 1), figsize=(8.0, 6.8))
    l1, = ax0.plot(x.index, x["volatility_cc_120"], color=blue, linewidth=1.5)
    l2, = ax0.plot(x.index, x["volatility_pk_120"], color=orange, linewidth=1.5)
    l3, = ax0.plot(x.index, x["volatility_yz_120"], color=aqua, linewidth=1.5)
    _end_labels(ax0, [(l1, "close-close"), (l2, "Parkinson"), (l3, "Yang-Zhang")], pal, min_gap=0.036,
                fmt="{:.0%}")
    percent_axis(ax0)
    ax0.set_ylabel("annualised volatility")
    ax0.set_title("Volatility family: three estimators of the same 120-bar volatility (H1, 2020)")
    ax1.plot(x.index, x["volatility_logratio_yz_24_120"], color=blue, linewidth=1.2)
    zero_line(ax1)
    ax1.set_ylabel("log ratio")
    ax1.set_title("volatility_logratio_yz_24_120 = ln(yz 24 / yz 120): above 0 = expanding", fontsize=10)
    ax2.plot(x.index, x["volatility_range_cc_120"], color=blue, linewidth=1.2)
    ax2.axhline(1.0, color=pal["axis"], linewidth=0.75)
    ax2.set_ylabel("ratio")
    ax2.set_title("volatility_range_cc_120 = Parkinson / close-close (about 1 for a random walk)", fontsize=10)
    ax2.set_xlabel("date (UTC, H1 bar open)")
    source_note(fig, "Data: " + source_label("H1", start="2020-01-01", end="2020-12-31", years=False), pal)
    return fig


@register(PAGE, "microstructure", needs=("data_store",))
def microstructure_fig(pal):
    """Spread cost in ATR units by UTC hour, and the distribution of the 120-bar return autocorrelation."""
    raw, warm, _ = _raw_h1()
    r = raw.iloc[warm:]
    hour = r.index.hour
    g = r["microstructure_spread_atr"].groupby(hour)
    q = pd.DataFrame({"p25": g.quantile(0.25), "p50": g.median(), "p75": g.quantile(0.75)})
    ac = r["microstructure_autocorr_120"].dropna()
    blue = pal["categorical"][0]
    fig, (ax0, ax1) = figure(1, 2, figsize=(8.0, 3.6))
    ax0.fill_between(q.index, q["p25"], q["p75"], color=blue, alpha=0.18, linewidth=0, step="mid")
    ax0.step(q.index, q["p50"], where="mid", color=blue)
    ax0.set_xticks(range(0, 24, 3), [f"{h:02d}" for h in range(0, 24, 3)])
    ax0.set_xlabel("hour of the H1 bar open (UTC)")
    ax0.set_ylabel("spread / ATR 14")
    ax0.set_ylim(0, None)
    hmax = int(q["p50"].idxmax())
    hmin = int(q["p50"].idxmin())
    ax0.set_title("microstructure_spread_atr: median, 25-75%", fontsize=11)
    ax0.annotate(f"{hmax:02d}:00 {q.loc[hmax, 'p50']:.2f}", xy=(hmax, q.loc[hmax, "p50"]),
                 xytext=(hmax - 9, q["p75"].max() * 1.0), fontsize=8, color=pal["text_secondary"], va="center",
                 arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    ax0.annotate(f"{hmin:02d}:00 {q.loc[hmin, 'p50']:.2f}", xy=(hmin, q.loc[hmin, "p50"]),
                 xytext=(hmin - 6, q["p25"].min() * 0.6), fontsize=8, color=pal["text_secondary"], va="center",
                 arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    ax1.hist(ac, bins=np.linspace(-0.5, 0.5, 51), color=blue, alpha=0.85)
    ax1.axvline(0, color=pal["axis"], linewidth=0.75)
    med = float(ac.median())
    ax1.axvline(med, color=pal["text_secondary"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax1.text(med - 0.02, ax1.get_ylim()[1] * 0.92, f"median {signed(med, '{:+.3f}')}", ha="right", fontsize=8,
             color=pal["text_secondary"])
    ax1.set_title("microstructure_autocorr_120", fontsize=11)
    ax1.set_xlabel("rolling lag-1 autocorrelation of H1 log returns")
    ax1.set_ylabel("H1 bars")
    ax1.yaxis.set_major_formatter(_thousands())
    source_note(fig, "Data: " + source_label("H1", start=raw.index[warm]), pal)
    return fig


# ---------------------------------------------------------------------------------------
# multi-timeframe, macro, calendar, session, regime
# ---------------------------------------------------------------------------------------
@register(PAGE, "mtf-alignment", needs=("data_store",))
def mtf_alignment(pal):
    """Previous-day high/low and D1/H4 RSI mapped onto H1 bars: values change only when an HTF bar completes."""
    from aurum.features.multi_timeframe import mtf_features, resample_anchored

    md = market_data("H1", macro=False, events=False, start="2023-12-01", end="2024-03-08")
    f = mtf_features(md)
    b = md.bars
    d1 = resample_anchored(b, "D1", 22)
    lo, hi = pd.Timestamp("2024-03-04 22:00", tz="UTC"), pd.Timestamp("2024-03-08 21:00", tz="UTC")
    w = b.loc[lo:hi]
    t = pd.DatetimeIndex(w["available_at"])          # features describe the decision instant
    from aurum.data.resample import align_htf

    lv = align_htf(w, d1, columns=["high", "low"])
    blue, orange, violet = pal["categorical"][0], pal["categorical"][1], pal["categorical"][6]
    fig, (ax0, ax1) = panels(2, heights=(2, 1.3), figsize=FIGSIZE_TALL)
    for day in pd.date_range(lo, hi, freq="D"):
        for ax in (ax0, ax1):
            ax.axvline(day, color=pal["axis"], linewidth=0.75, zorder=0.5)
    ax0.plot(t, w["close"], color=blue, label="H1 close")
    ax0.step(t, lv["high"], where="post", color=orange, label="previous day high (PDH)")
    ax0.step(t, lv["low"], where="post", color=violet, label="previous day low (PDL)")
    add_legend(ax0, loc="upper left", fontsize=8)
    ax0.set_ylabel("USD per ounce")
    ax0.set_title("mtf: higher-timeframe values reach an H1 bar only once the HTF bar is complete")
    x = f.loc[lo:hi]
    ax1.step(t, x["mtf_h4_rsi_14"], where="post", color=orange, label="mtf_h4_rsi_14 (steps every 4 h)")
    ax1.step(t, x["mtf_d1_rsi_14"], where="post", color=blue, label="mtf_d1_rsi_14 (steps at 22:00)")
    add_legend(ax1, loc="lower right", fontsize=8)
    ax1.set_ylabel("(RSI - 50) / 50")
    ax1.set_xlabel("H1 decision time, available_at (UTC); vertical lines = 22:00 D1 boundaries")
    import matplotlib.dates as mdates

    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[22], tz="UTC"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %H:%M", tz="UTC"))
    source_note(fig, "Data: " + source_label("H1", start="2024-03-04", end="2024-03-08", years=False), pal)
    return fig


@register(PAGE, "macro-correlations", needs=("data_store",))
def macro_correlations(pal):
    """Rolling 60-day correlation of daily gold returns with the dollar index and the 10-year real yield."""
    raw, _warm, _ = _raw_h1()
    c = raw[["macro_corr_dxy", "macro_corr_real10y"]]
    daily = c.groupby(c.index.normalize()).last().dropna(how="all")
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, ax = figure(figsize=FIGSIZE)
    l1, = ax.plot(daily.index, daily["macro_corr_dxy"], color=blue, linewidth=1.3)
    l2, = ax.plot(daily.index, daily["macro_corr_real10y"], color=orange, linewidth=1.3)
    zero_line(ax)
    _end_labels(ax, [(l1, "vs dxy"), (l2, "vs real10y")], pal, min_gap=0.09, fmt="{:+.2f}")
    ax.set_ylim(-1, 0.4)
    ax.set_ylabel("rolling 60-day correlation")
    ax.set_xlabel("date (UTC)")
    ax.set_title("macro_corr_dxy and macro_corr_real10y as an H1 decision sees them")
    source_note(fig, "Data: " + source_label("H1") + "; Yahoo DX-Y.NYB, FRED DFII10 (publication-lagged)", pal)
    return fig


@register(PAGE, "calendar-features", needs=("data_store",))
def calendar_features_fig(pal):
    """Event-proximity features around the 2024-01-31 FOMC and 2024-02-02 NFP releases."""
    raw, _warm, _ = _raw_h1()
    lo, hi = pd.Timestamp("2024-01-29", tz="UTC"), pd.Timestamp("2024-02-06 21:00", tz="UTC")
    x = raw.loc[lo:hi]
    t = pd.DatetimeIndex(bars("H1", start=lo, end="2024-02-06 21:00")["available_at"])
    from aurum.data.calendar import generate_rule_based_calendar

    ev = generate_rule_based_calendar("2024-01-25", "2024-02-20")
    ev = ev[(ev["time"] >= lo) & (ev["time"] <= hi + pd.Timedelta(hours=1))]
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1) = panels(2, heights=(1.7, 1.3), figsize=FIGSIZE_TALL)
    wk0, wk1 = pd.Timestamp("2024-02-02 22:00", tz="UTC"), pd.Timestamp("2024-02-04 23:00", tz="UTC")
    shade_span(ax0, wk0, wk1, "weekend")
    shade_span(ax1, wk0, wk1)
    for k, (_, e) in enumerate(ev.iterrows()):
        for ax in (ax0, ax1):
            ax.axvline(e["time"], color=pal["axis"], linewidth=0.75, zorder=0.5)
        ax0.text(e["time"], 76, f" {e['name']} {e['time']:%a %d %H:%M} " if k else f"{e['name']} {e['time']:%a %d %H:%M} ",
                 fontsize=8, color=pal["text_secondary"], va="bottom", ha="left" if k else "right")
    ax0.step(t, x["calendar_hours_to_next"], where="post", color=blue, label="calendar_hours_to_next")
    ax0.step(t, x["calendar_hours_since_last"], where="post", color=orange, label="calendar_hours_since_last")
    ax0.set_ylim(0, 86)
    ax0.set_ylabel("hours (capped at 72)")
    ax0.set_title("calendar: hours to the next and since the last importance-3 event")
    add_legend(ax0, loc="center right", fontsize=8)
    flags = ["calendar_pre_2h", "calendar_post_2h", "calendar_next_nfp", "calendar_next_fomc"]
    style_axes(ax1, grid=None)
    import matplotlib.dates as mdates

    for i, col in enumerate(flags):
        y = len(flags) - 1 - i
        on = x[col].to_numpy() > 0.5
        segs = []
        start = None
        for k in range(len(on)):
            if on[k] and start is None:
                start = k
            if start is not None and (not on[k] or k == len(on) - 1):
                end = k if not on[k] else k + 1
                t0 = mdates.date2num(t[start] - pd.Timedelta(hours=1))
                t1 = mdates.date2num(t[min(end, len(t) - 1)] - pd.Timedelta(hours=1)) if end < len(t) else mdates.date2num(t[-1])
                segs.append((t0, max(t1 - t0, 1 / 24)))
                start = None
        ax1.broken_barh(segs, (y - 0.32, 0.64), facecolors=blue)
    ax1.set_yticks(range(len(flags)), list(reversed(flags)))
    ax1.tick_params(axis="y", length=0)
    ax1.spines["left"].set_visible(False)
    ax1.set_ylim(-0.6, len(flags) - 0.4)
    ax1.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d", tz="UTC"))
    ax1.set_xlabel("H1 decision time (UTC), 29 Jan to 6 Feb 2024; filled = flag is 1")
    source_note(fig, "Data: aurum rule-based NFP/FOMC calendar on " + source_label(
        "H1", start="2024-01-29", end="2024-02-06", years=False), pal)
    return fig


@register(PAGE, "session-flags", needs=("data_store",))
def session_flags(pal):
    """Session flags on three Tuesdays: US and UK winter, the March weeks when only the US is on DST, summer."""
    from aurum.features.microstructure import session_features

    days = [("2024-02-20", "Tue 20 Feb 2024: both on winter time"),
            ("2024-03-12", "Tue 12 Mar 2024: New York on DST, London not yet"),
            ("2024-07-16", "Tue 16 Jul 2024: both on summer time")]
    cols = [("session_asia", "Asia (Tokyo 08-17)"), ("session_london", "London 08-17"),
            ("session_ny", "New York 08-17"), ("session_overlap", "London and NY"),
            ("session_rollover", "NY rollover 16-18")]
    colors = [pal["categorical"][i] for i in range(len(cols))]
    fig, axes = panels(3, figsize=(8.0, 6.6), hspace=0.06)
    from aurum.core.types import MarketData

    for ax, (day, title) in zip(axes, days, strict=True):
        b = bars("H1", start=f"{day} 00:00", end=f"{day} 23:59")
        f = session_features(MarketData(bars=b))
        d0 = pd.Timestamp(day, tz="UTC")
        hrs = (pd.DatetimeIndex(b["available_at"]) - d0) / pd.Timedelta(hours=1)
        style_axes(ax, grid="x")
        for i, (col, _label) in enumerate(cols):
            y = len(cols) - 1 - i
            on = f[col].to_numpy() > 0.5
            segs = [(float(hrs[k]) - 0.5, 1.0) for k in range(len(on)) if on[k]]
            ax.broken_barh(segs, (y - 0.33, 0.66), facecolors=colors[i])
        ax.set_yticks(range(len(cols)), [lab for _c, lab in reversed(cols)], fontsize=8)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(cols) - 0.4)
        ax.set_xlim(0, 24.5)
        ax.set_title(title, fontsize=10)
    axes[-1].set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    axes[-1].set_xlabel("decision instant available_at, UTC hour of day (each cell = one H1 bar)")
    fig.suptitle("session: DST-aware session flags of the decision instant", x=0.01, ha="left",
                 fontsize=12, color=pal["text"])
    source_note(fig, "Data: H1 bar times from " + source_label("H1", start="2024-02-20", end="2024-07-16",
                                                               years=False), pal)
    return fig


@register(PAGE, "regime-features", needs=("data_store",))
def regime_features_fig(pal):
    """Gold H1 in 2024 H1 with the trend state, its two inputs and the bounded vol percentile."""
    from aurum.features.regime import efficiency_ratio

    raw, _warm, _ = _raw_h1()
    b = bars("H1")
    show = slice("2024-01-01", "2024-06-30")
    x = raw.loc[show]
    close = b["close"]
    er120 = efficiency_ratio(close, 120).loc[show]
    c = close.loc[show]
    blue = pal["categorical"][0]
    pos, neg = pal["diverging"][-1], pal["diverging"][0]
    fig, (ax0, ax1, ax2, ax3) = panels(4, heights=(1.6, 1, 1, 1), figsize=(8.0, 7.6))
    st = x["regime_trend_state"]
    for val, color, name in ((1.0, pos, "regime_trend_state = +1"), (-1.0, neg, "regime_trend_state = -1")):
        on = (st == val).to_numpy()
        if not on.any():
            continue
        for ax in (ax0, ax1, ax2):
            ax.fill_between(x.index, 0, 1, where=on, transform=ax.get_xaxis_transform(), color=color,
                            alpha=0.25, linewidth=0, step="post", label=name if ax is ax0 else None)
    ax0.plot(c.index, c.values, color=pal["text"], linewidth=1.0)
    ax0.set_ylabel("USD per ounce")
    ax0.set_title("regime: the trend state and its inputs, and the bounded volatility percentile")
    add_legend(ax0, min_series=1, loc="upper left", fontsize=8)   # names the shading (often one state only)
    ax1.plot(x.index, x["regime_trend_strength"], color=blue, linewidth=1.0)
    ax1.axhline(1.0, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax1.set_ylim(0, None)
    ax1.set_ylabel("strength")
    ax1.set_title("regime_trend_strength (120 bars); the state needs it above 1", fontsize=10)
    ax2.plot(er120.index, er120.values, color=blue, linewidth=1.0)
    ax2.axhline(0.3, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax2.set_ylim(0, None)
    ax2.set_ylabel("ER")
    ax2.set_title("Kaufman efficiency ratio over 120 bars; the state needs it above 0.3", fontsize=10)
    ax3.plot(x.index, x["regime_vol_pctrank"], color=blue, linewidth=1.0)
    for v in (0.2, 0.8):
        ax3.axhline(v, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("percentile")
    ax3.set_title("regime_vol_pctrank: 24-bar vol ranked within the last year (flags at 0.8 / 0.2)",
                  fontsize=10)
    ax3.set_xlabel("date (UTC, H1 bar open)")
    source_note(fig, "Data: " + source_label("H1", start="2024-01-01", end="2024-06-30", years=False), pal)
    return fig


# ---------------------------------------------------------------------------------------
# leakage test, correlations, warm-up
# ---------------------------------------------------------------------------------------
@functools.cache
def _perturbation() -> dict:
    """Default pipeline on real H1 (2023-01 to 2024-06) and on a copy whose future after the
    cutoff bar is replaced by an unrelated path (prices, spreads, volume and macro rows)."""
    from aurum.core.types import MarketData
    from aurum.features import FeaturePipeline
    from aurum.features.volatility import atr

    md = market_data("H1", start="2023-01-01", end="2024-06-30")
    b = md.bars
    t = int(np.flatnonzero(b.index == pd.Timestamp("2024-05-15 12:00", tz="UTC"))[0])
    cutoff = b["available_at"].iloc[t]
    rng = np.random.default_rng(11)
    n_fut = len(b) - t - 1
    factor = 1.06 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n_fut)))
    alt = b.copy()
    for c in ("open", "high", "low", "close"):
        alt.iloc[t + 1:, alt.columns.get_loc(c)] = b[c].iloc[t + 1:].to_numpy() * factor
    alt.iloc[t + 1:, alt.columns.get_loc("spread")] = b["spread"].iloc[t + 1:].to_numpy() * 3.0
    alt.iloc[t + 1:, alt.columns.get_loc("volume")] = b["volume"].iloc[t + 1:].to_numpy() * 0.5
    macro = {}
    for name, frame in md.macro.items():
        g = frame.copy()
        fut = pd.DatetimeIndex(g["available_at"]) > cutoff
        vals = g["value"].to_numpy(dtype=float).copy()
        vals[fut] = rng.uniform(-1.5, 2.5, int(fut.sum())) * np.abs(vals[fut]).mean() if fut.any() else vals[fut]
        g["value"] = vals
        macro[name] = g
    md_alt = MarketData(bars=alt, macro=macro, events=md.events)
    pipe = FeaturePipeline()
    a = pipe.compute(md)
    p = pipe.compute(md_alt)
    trunc_md = MarketData(bars=b.iloc[: t + 1],
                          macro={k: v.loc[pd.DatetimeIndex(v["available_at"]) <= cutoff] for k, v in md.macro.items()},
                          events=md.events)
    tr = pipe.compute(trunc_md)
    xa, xp = a.to_numpy(dtype=float), p.to_numpy(dtype=float)
    diff = ~((xa == xp) | (np.isnan(xa) & np.isnan(xp)))
    xt = tr.to_numpy(dtype=float)
    diff_tr = ~((xa[: t + 1] == xt) | (np.isnan(xa[: t + 1]) & np.isnan(xt)))

    def leaky(bb: pd.DataFrame) -> pd.Series:   # negative control: a CENTRED 25-bar window
        mean = bb["close"].rolling(25, center=True, min_periods=25).mean()
        return (bb["close"] - mean) / atr(bb, 14)

    la, lp = leaky(b), leaky(alt)
    return {"index": b.index, "t": t, "close": b["close"], "alt_close": alt["close"], "n_cols": a.shape[1],
            "n_diff": diff.sum(axis=1), "n_diff_past": int(diff[: t + 1].sum()),
            "n_diff_trunc": int(diff_tr.sum()), "leak_diff": (la - lp).abs(),
            "causal_a": a["trend_ema_dist_20"], "causal_p": p["trend_ema_dist_20"]}


@register(PAGE, "leakage-perturbation", needs=("data_store",))
def leakage_perturbation(pal):
    """Future perturbation on real H1: every pipeline column is unchanged up to the cutoff; a centred window is not."""
    d = _perturbation()
    t, idx = d["t"], d["index"]
    lo, hi = t - 48, t + 36
    sl = slice(lo, hi + 1)
    x = np.arange(lo, hi + 1) - t                     # bars relative to the cutoff
    cut = idx[t]
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, (ax0, ax1, ax2) = panels(3, heights=(1.5, 1, 1), figsize=(8.0, 6.8))
    for ax in (ax0, ax1, ax2):
        ax.axvline(0, color=pal["text_muted"], linewidth=1.0, linestyle=(0, (3, 2)), zorder=0.5)
    ax0.plot(x, d["alt_close"].iloc[sl], color=orange, label="perturbed history (future replaced)")
    ax0.plot(x, d["close"].iloc[sl], color=blue, label="real history")
    # left of the cutoff line, where the (unchanged) real history sits low: no line crosses it
    ax0.text(-0.6, ax0.get_ylim()[1], f"cutoff t = bar opening {cut:%a %d %b %Y %H:%M} UTC ", fontsize=8,
             color=pal["text_secondary"], va="top", ha="right")
    add_legend(ax0, loc="center left", fontsize=8)
    ax0.set_ylabel("USD per ounce")
    ax0.set_title("Leakage test: change everything after bar t, then recompute every feature")
    nd = np.asarray(d["n_diff"][sl])
    ax1.step(x, nd, where="mid", color=blue)
    ax1.set_ylim(-5, d["n_cols"] * 1.08)
    ax1.set_ylabel("columns")
    ax1.set_title(f"pipeline columns (of {d['n_cols']}) that differ from the real-history value: "
                  "0 in every row up to t", fontsize=10)
    leak = np.asarray(d["leak_diff"].iloc[sl])
    ax2.plot(x, leak, color=orange)
    first = int(x[np.flatnonzero(leak > 0)[0]])
    ax2.annotate(f"first differs at bar t{signed(first, '{:+d}')}", xy=(first, 0),
                 xytext=(x[0] + 2, float(np.nanmax(leak)) * 0.6), fontsize=8, color=pal["text_secondary"],
                 arrowprops={"arrowstyle": "-|>", "color": pal["text_secondary"], "lw": 1.0})
    ax2.set_ylabel("|difference|, ATR")
    ax2.set_title("negative control: (C - centred 25-bar mean) / ATR reads 12 bars ahead", fontsize=10)
    ax2.set_xlabel("H1 bars relative to the cutoff bar t")
    source_note(fig, "Data: " + source_label("H1", start="2023-01-01", end="2024-06-30", years=False)
                + "; perturbed copy built in tools/figures/fig_features.py", pal)
    return fig


CORR_FEATURES = (
    "returns_z_24", "trend_ema_dist_20", "trend_ema_cross_20_100", "trend_linreg_t_100", "trend_adx_14",
    "momentum_rsi_14", "momentum_tsmom_240", "meanrev_z_50", "meanrev_bb_width_20", "range_donchian_pos_55",
    "volatility_yz_24", "volatility_logratio_yz_24_120", "microstructure_spread_bps", "microstructure_volume_z",
    "session_hour_cos", "mtf_d1_ret_z_5", "macro_dxy_chg_5d", "macro_real10y_chg_5d", "macro_vix_z",
    "calendar_hours_to_next", "regime_vol_pctrank", "regime_er_100",
)


@register(PAGE, "correlation-heatmap", needs=("data_store",))
def correlation_heatmap(pal):
    """Spearman correlation of 22 representative features on real H1 bars after the warm-up."""
    raw, warm, _ = _raw_h1()
    x = raw.iloc[warm:][list(CORR_FEATURES)]
    corr = x.rank().corr().to_numpy()
    n = len(CORR_FEATURES)
    fig, ax = figure(figsize=(8.0, 7.2))
    style_axes(ax, grid=None)
    im = ax.imshow(corr, cmap=diverging_cmap(pal["mode"]), vmin=-1, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{corr[i, j]:.1f}".replace("-", "\N{MINUS SIGN}").replace("0.", ".")
                        if abs(corr[i, j]) >= 0.05 else "", ha="center", va="center", fontsize=6.5,
                        color=ink_on(im.cmap(im.norm(corr[i, j]))))
    ax.set_xticks(range(n), CORR_FEATURES, rotation=90, fontsize=7.5)
    ax.set_yticks(range(n), CORR_FEATURES, fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature correlations (Spearman) on H1 bars")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Spearman correlation", color=pal["text_secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])
    source_note(fig, "Data: " + source_label("H1", start=raw.index[warm]) + f" ({len(x):,} bars after warm-up)", pal)
    return fig


@register(PAGE, "warmup-by-group", needs=("data_store",))
def warmup_by_group(pal):
    """Predicted warm-up (lookback_fn) vs the first fully defined row on real H1 bars, per group."""
    raw, max_lb, lbs = _raw_h1()
    observed = {}
    for g in GROUP_ORDER:
        cols = [c for c in raw.columns if c.startswith(g + "_")]
        ok = raw[cols].notna().all(axis=1).to_numpy()
        observed[g] = int(ok.argmax()) if ok.any() else np.nan
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    fig, ax = figure(figsize=(8.0, 4.6))
    style_axes(ax, grid="x")
    ys = np.arange(len(GROUP_ORDER))[::-1]
    pred = np.array([lbs[g] for g in GROUP_ORDER], dtype=float)
    obs = np.array([observed[g] for g in GROUP_ORDER], dtype=float)
    ax.barh(ys, pred, height=0.55, color=blue, label="predicted warm-up (lookback_fn, bars)")
    ax.plot(obs, ys, marker="D", markersize=6, linestyle="none", color=orange,
            label="first row with every column defined (real H1)")
    for y, p, o in zip(ys, pred, obs, strict=True):
        ax.text(max(p, o) + 60, y, f"{int(p):,} / {int(o):,}", va="center", fontsize=8, color=pal["text_secondary"])
    ax.axvline(max_lb, color=pal["text_muted"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(max_lb, len(GROUP_ORDER) - 0.2, f"max_lookback {max_lb:,}", ha="right", fontsize=8,
            color=pal["text_secondary"])
    ax.set_yticks(ys, list(GROUP_ORDER))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max_lb * 1.18)
    ax.set_xlabel("H1 bars from the first bar (2012-01-01 22:00 UTC)")
    ax.set_title("Warm-up per group: predicted / observed")
    add_legend(ax, loc="upper center", fontsize=8)     # clear of the bars and the max_lookback line
    source_note(fig, "Data: " + source_label("H1") + " with macro and the rule-based calendar", pal)
    return fig
