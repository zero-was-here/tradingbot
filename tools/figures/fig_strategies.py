"""Figures for docs/strategies.md: every registered strategy, drawn by its own aurum code.

Each ``@register``-ed ``draw(pal)`` becomes docs/_assets/figures/strategies/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only strategies

Conventions used throughout this module

* Strategies are always generated on the FULL available history (2012 onwards, cut at the
  2024-12-31 figure cutoff by ``_common``) and only then sliced to the display window, so the
  warm-up, the EWMA volatility and every path-dependent state (latches, positions, filters)
  are exactly what a long research run sees on those bars.
* The intermediate series (channels, bands, z-scores, ratios) are recomputed with the same
  aurum helpers the strategy calls (``log_close``, ``bar_volatility``, ``momentum_z``,
  ``rolling_zscore``, ``efficiency_ratio``, ``atr`` ...) or read from the strategy's own
  inspection methods (``components``, ``positions``, ``squeeze_on``, ``slope``,
  ``daily_signal``, ``regime_probabilities``, ``training_targets``).
* Nothing here runs a backtest. The only fitting is ``intraday_seasonality`` on one protocol
  training window and the ``regime_switch`` HMM on the training windows listed in
  ``runs/v21_managed_gold/folds.csv`` (all before 2025).
"""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_SHORT,
    FIGSIZE_TALL,
    ink_on,
    macro_label,
    market_data,
    panels,
    register,
    run_label,
    run_table,
    signed,
    source_label,
    source_note,
)
from matplotlib.figure import Figure
from scipy import signal
from scipy.stats import norm

from aurum.core.types import MarketData
from aurum.features.regime import efficiency_ratio
from aurum.features.technical import rolling_zscore, rsi, tsmom_response
from aurum.features.volatility import atr, safe_div
from aurum.strategies import get_strategy
from aurum.strategies.macro import clean_macro, macro_momentum_z
from aurum.strategies.ml import apply_dead_zone, bet_size_from_probability, discretize_bet
from aurum.strategies.regime import regime_scale
from aurum.strategies.trend import (
    EXPECTED_ABS_BAZ,
    TARGET_ABS_FORECAST,
    Z_FORECAST_SCALAR,
    bar_volatility,
    ema_spread_variance,
    log_close,
    momentum_z,
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

PAGE = "strategies"
Z = Z_FORECAST_SCALAR

#: display windows (all before the 2024-12-31 cutoff); the page says why each was chosen
COVID_YEAR = ("2020-01-01", "2020-12-31")          # crash, rally to the August high, pullback
COVID_TURN = ("2020-02-10", "2020-05-15")          # the sell-off and the V-shaped recovery
SUMMER_2020 = ("2020-06-15", "2020-09-15")         # breakout to the 2,075 high and the reversal
RANGE_2023 = ("2023-07-17", "2023-07-28")          # a quiet two-week range (reversion's home)
ORB_WEEK = ("2024-03-04", "2024-03-08")            # one ordinary week, NFP on Friday
HIKING = ("2021-07-01", "2023-06-30")              # the 2022 hiking cycle and the dollar peak
RISK_OFF = ("2020-01-15", "2020-05-29")            # COVID: VIX spike and the dash for cash

#: daily parameters of configs/managed_gold.yaml (copied, so the figure needs no config file)
MANAGED_GOLD_D1 = {"base_weight": 0.5, "tilt_weight": 0.5, "horizons": (21, 63, 126, 252),
                   "trend_halflife": 69, "vol_fast_halflife": 5, "vol_slow_halflife": 126,
                   "vol_enter": 1.25, "vol_exit": 1.0, "floor": 0.0,
                   "vol_halflife": 30, "vol_min_periods": 20}
REGIME_SWITCH_D1 = {"base": "tsmom",
                    "base_params": {"horizons": (21, 63, 126, 252), "vol_halflife": 30, "vol_min_periods": 20},
                    "returns": "bar", "p_flat": 0.5, "p_full": 0.9}

#: strategy order used by the correlation heatmap (family by family)
FAMILY_ORDER = ("tsmom", "ema_cross", "donchian", "kalman_trend", "zscore_fade", "rsi2",
                "bollinger_revert", "vol_squeeze", "orb", "macro_factor", "risk_off",
                "intraday_seasonality", "meta_label")


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------
def _date_axis(ax, *, maxticks: int = 8) -> None:
    loc = mdates.AutoDateLocator(minticks=3, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=True))


def _win(obj, window):
    return obj.loc[window[0]:window[1]]


def _title(ax, text: str, *, size: float = 10.5) -> None:
    ax.set_title(text, fontsize=size)


def _forecast(ax, f: pd.Series, pal, *, color=None, label=None, fill: bool = True,
              ylim=(-1.15, 1.15), ticks=(-1, -0.5, 0, 0.5, 1)):
    """Forecast as a step line (held from the bar it is decided at), with a light fill."""
    color = color or pal["categorical"][0]
    x, y = f.index, f.to_numpy(dtype=float)
    if fill:
        ax.fill_between(x, 0.0, y, step="post", color=color, alpha=0.18, linewidth=0)
    (line,) = ax.step(x, y, where="post", color=color, label=label)
    zero_line(ax)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(ticks))
    ax.set_ylabel("forecast")
    return line


def _hline(ax, y: float, pal, text: str | None = None, *, side: str = "right", va: str = "bottom") -> None:
    """A threshold line in secondary ink with an optional small label inside the axes."""
    ax.axhline(y, color=pal["text_secondary"], linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    if text:
        x = 0.995 if side == "right" else 0.005
        ax.annotate(text, (x, y), xycoords=("axes fraction", "data"), ha=side, va=va,
                    xytext=(0, 2 if va == "bottom" else -2), textcoords="offset points",
                    color=pal["text_secondary"], fontsize=8, zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": pal["surface"], "ec": "none", "alpha": 0.85})


def _end_labels(ax, items, pal, *, gap: float = 0.08, fmt: str | None = None) -> None:
    """Direct labels just right of the axes for several lines: an end-dot on each line and a
    thin leader from the dot to its label, labels nudged apart (``gap`` = axes fraction)."""
    lo, hi = ax.get_ylim()
    step = gap * (hi - lo)
    ends = []
    for line, text in items:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata(), dtype=float)
        ok = np.flatnonzero(np.isfinite(y))
        if ok.size == 0:
            continue
        i = int(ok[-1])
        label = text if fmt is None else f"{text}  {fmt.format(y[i])}"
        ends.append([y[i], x[i], line, label, y[i]])
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):              # push up until separated
        if ends[k][4] < ends[k - 1][4] + step:
            ends[k][4] = ends[k - 1][4] + step
    over = (ends[-1][4] - (hi - 0.02 * (hi - lo))) if ends else 0.0
    if over > 0:                               # shift the stack down if it left the axes
        for e in ends:
            e[4] -= over
    for y, x, line, label, y_text in ends:
        ax.plot([x], [y], marker="o", linestyle="none", color=line.get_color(),
                markeredgecolor=pal["surface"], markeredgewidth=1.5, markersize=6,
                zorder=line.get_zorder() + 1, label="_nolegend_", clip_on=False)
        ax.annotate(label, (x, y), xycoords="data", xytext=(1.03, y_text),
                    textcoords=("axes fraction", "data"), va="center", ha="left",
                    color=pal["text"], fontsize=8.5, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.7,
                                "shrinkA": 1, "shrinkB": 4})


def _bar_axis(ax, index: pd.DatetimeIndex) -> None:
    """Ticks for a trading-time axis (x = bar number): one per UTC day with >= 5 bars."""
    days = index.normalize()
    first = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    counts = np.diff(np.r_[first, len(index)])
    keep = first[counts >= 5]
    ax.set_xticks(keep, [index[i].strftime("%a %d") for i in keep])
    ax.set_xlim(-1, len(index))


def _positional(s: pd.Series, window) -> pd.Series:
    """``s`` on the window with a 0..n-1 index (weekends and gaps squeezed out)."""
    return s.loc[window[0]:window[1]].reset_index(drop=True)


def _spans(mask: pd.Series):
    """(start, end) pairs of the True runs of a boolean series (end = first False index)."""
    m = mask.fillna(False).astype(bool).to_numpy()
    idx = mask.index
    out = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            out.append((idx[i], idx[min(j + 1, n - 1)]))
            i = j + 1
        else:
            i += 1
    return out


def _shade(ax, mask: pd.Series, label: str | None = None) -> None:
    for k, (a, b) in enumerate(_spans(mask)):
        shade_span(ax, a, b, label if k == 0 else None)


def _price(ax, close: pd.Series, pal, *, label: str = "close"):
    (line,) = ax.plot(close.index, close.to_numpy(), color=pal["categorical"][0], label=label)
    ax.set_ylabel("USD per ounce")
    return line


def _md(tf: str = "H1", **kw) -> MarketData:
    return market_data(tf, **kw)


# ---------------------------------------------------------------------------------------
# the forecast contract and the shared scaling
# ---------------------------------------------------------------------------------------
@register(PAGE, "forecast-contract", needs=("data_store",))
def forecast_contract(pal):
    """Raw ema_cross statistic z -> x0.6267 and cap -> forecast, on real H1 gold."""
    md = _md("H1")
    s = get_strategy("ema_cross")
    p = s.params
    lc = log_close(md.bars)
    sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
    spread = lc.ewm(span=p["fast"], adjust=False).mean() - lc.ewm(span=p["slow"], adjust=False).mean()
    z = pd.Series(safe_div(spread, sigma * math.sqrt(ema_spread_variance(p["fast"], p["slow"]))),
                  index=md.bars.index).iloc[s.warmup_bars:]
    z = z[np.isfinite(z.to_numpy())]
    f = s.generate(md).iloc[s.warmup_bars:]
    blue, orange, aqua = pal["categorical"][:3]

    fig, (a0, a1, a2) = figure(1, 3, figsize=(8.0, 3.5))
    edges = np.linspace(-5, 5, 81)
    a0.hist(np.clip(z, -5, 5), bins=edges, density=True, color=blue, alpha=0.75, linewidth=0)
    g = np.linspace(-5, 5, 400)
    a0.plot(g, norm.pdf(g), color=pal["text_secondary"], linewidth=1.2, linestyle=(0, (4, 3)))
    a0.annotate(f"real: sd {z.std():.2f}\nN(0, 1): sd 1.00", (0.03, 0.97), xycoords="axes fraction",
                va="top", ha="left", fontsize=8, color=pal["text_secondary"])
    a0.set_xlabel("z (spread / random-walk sd)")
    a0.set_ylabel("density")
    _title(a0, "1. Raw statistic z")

    x = np.linspace(-4, 4, 801)
    lin = np.clip(Z * x, -1, 1)
    sgn = np.sign(x) * TARGET_ABS_FORECAST
    baz = np.clip(tsmom_response(x) * (TARGET_ABS_FORECAST / EXPECTED_ABS_BAZ), -1, 1)
    a1.plot(x, lin, color=blue)
    a1.plot(x, sgn, color=orange)
    a1.plot(x, baz, color=aqua)
    zero_line(a1)
    cap = 1.0 / Z
    for v in (-cap, cap):
        a1.axvline(v, color=pal["grid"], linewidth=1.0, zorder=0)
    a1.annotate("linear (default)", (2.05, 1.0), xytext=(0, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, color=pal["text"])
    a1.annotate("sign", (3.3, 0.5), xytext=(0, 4), textcoords="offset points", ha="left",
                va="bottom", fontsize=8, color=pal["text"])
    a1.annotate("baz", (-3.95, float(np.interp(-3.95, x, baz))), xytext=(0, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, color=pal["text"])
    a1.annotate(f"cap binds\n|z| > {cap:.2f}", (-cap, 0.85), xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=pal["text_secondary"])
    a1.set_ylim(-1.15, 1.25)
    a1.set_yticks([-1, -0.5, 0, 0.5, 1])
    a1.set_xlabel("z")
    a1.set_ylabel("forecast")
    _title(a1, "2. Scale x0.6267, cap at \N{PLUS-MINUS SIGN}1")

    w = np.full(len(f), 1.0 / len(f))
    a2.hist(f, bins=np.linspace(-1.025, 1.025, 42), weights=w, color=blue, alpha=0.75, linewidth=0)
    percent_axis(a2, decimals=0)
    a2.annotate(f"mean |f| {f.abs().mean():.3f}\nat \N{PLUS-MINUS SIGN}1: {(f.abs() == 1).mean():.1%}",
                (0.5, 0.97), xycoords="axes fraction", va="top", ha="center", fontsize=8,
                color=pal["text_secondary"])
    a2.set_xlabel("forecast")
    a2.set_ylabel("share of bars")
    _title(a2, "3. Forecast in [\N{MINUS SIGN}1, 1]")
    source_note(fig, "Data: " + source_label("H1") + "; ema_cross 32/128 after its 385-bar warm-up", pal)
    return fig


@register(PAGE, "trend-kernels")
def trend_kernels(pal):
    """The weight each past return gets in tsmom, ema_cross and kalman_trend (defaults)."""
    lags = np.arange(1700)
    ts = get_strategy("tsmom")
    w, fdm = ts.combination()
    k_ts = sum(wi * fdm * Z / math.sqrt(h) * (lags < h) for wi, h in zip(w, ts.params["horizons"], strict=True))
    ec = get_strategy("ema_cross")
    a = 1.0 - 2.0 / (ec.params["slow"] + 1.0)
    b = 1.0 - 2.0 / (ec.params["fast"] + 1.0)
    k_ema = Z * (a ** (lags + 1) - b ** (lags + 1)) / math.sqrt(ema_spread_variance(ec.params["fast"], ec.params["slow"]))
    kt = get_strategy("kalman_trend")
    bb, aa, gnorm = kt.filter_coefficients()
    k_kal = Z * signal.lfilter(bb, aa, np.ones(len(lags))) / gnorm

    fig, ax = figure(figsize=FIGSIZE)
    blue, orange, aqua = pal["categorical"][:3]
    ago = lags + 1                                   # 1 = the decision bar's own return
    ax.plot(ago, k_ts, color=blue)
    ax.plot(ago, k_ema, color=orange)
    ax.plot(ago, k_kal, color=aqua)
    zero_line(ax)
    ax.set_xscale("log")
    ax.set_xlim(1, len(lags))
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1000"])
    ax.set_ylim(-0.003, 0.088)
    peak_e, peak_k = int(np.argmax(k_ema)), int(np.argmax(k_kal))
    arrow = {"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.8}
    ax.annotate("tsmom 120/480/1440: three boxcars", (700, k_ts[699]), xytext=(0.985, 0.36),
                textcoords="axes fraction", fontsize=8.5, color=pal["text"], va="bottom", ha="right",
                arrowprops=arrow)
    ax.annotate(f"ema_cross 32/128: a hump, peak {peak_e + 1} bars ago", (peak_e + 1, k_ema[peak_e]),
                xytext=(40, 0), textcoords="offset points", fontsize=8.5, color=pal["text"], va="center",
                arrowprops=arrow)
    ax.annotate(f"kalman_trend lookback 240: near-flat to ~30 bars (peak {peak_k + 1}), then fades",
                (150, k_kal[149]), xytext=(0.985, 0.62), textcoords="axes fraction", fontsize=8.5,
                color=pal["text"], va="bottom", ha="right", arrowprops=arrow)
    energy = [float(np.sum(k ** 2)) for k in (k_ts, k_ema, k_kal)]
    ax.annotate("sum of squared weights\n(tsmom, ema_cross, kalman_trend):\n"
                + ", ".join(f"{e:.3f}" for e in energy) + f"\n= 0.6267\N{SUPERSCRIPT TWO} = {Z * Z:.3f} for all three",
                (0.01, 0.98), xycoords="axes fraction", ha="left", va="top", fontsize=8,
                color=pal["text_secondary"])
    ax.set_xlabel("bars ago (1 = the decision bar's own return), log scale")
    ax.set_ylabel("forecast per +1 sigma return")
    ax.set_title("Three trend rules as linear filters of past returns")
    return fig


# ---------------------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------------------
@register(PAGE, "tsmom", needs=("data_store",))
def tsmom(pal):
    """Close, the per-horizon tsmom terms and the clipped forecast through 2020."""
    md = _md("H1")
    s = get_strategy("tsmom")
    p = s.params
    lc = log_close(md.bars)
    sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
    w, fdm = s.combination()
    f = _win(s.generate(md), COVID_YEAR)
    close = _win(md.bars["close"], COVID_YEAR)
    hi = close.loc["2020-02-20":"2020-03-12"].idxmax()
    lo = close.loc["2020-03-12":"2020-03-25"].idxmin()

    fig, (a0, a1, a2) = panels(3, heights=(2.6, 2.2, 1.8), figsize=FIGSIZE_TALL)
    for ax in (a0, a1, a2):
        shade_span(ax, hi, lo, "sell-off" if ax is a0 else None)
    _price(a0, close, pal)
    _title(a0, "tsmom on XAUUSD H1, 2020: close")
    names = {120: "1 week (120 bars)", 480: "1 month (480)", 1440: "3 months (1440)"}
    items = []
    for i, (wi, h) in enumerate(zip(w, p["horizons"], strict=True)):
        c = _win(pd.Series(wi * fdm * Z * momentum_z(lc, sigma, h).to_numpy(), index=lc.index), COVID_YEAR)
        line, = a1.plot(c.index, c.to_numpy(), color=pal["categorical"][i])
        items.append((line, names[h]))
    zero_line(a1)
    a1.set_ylim(-1.0, 1.0)
    _end_labels(a1, items, pal, gap=0.12)
    a1.set_ylabel("forecast units")
    _title(a1, f"Per-horizon terms w x FDM x 0.6267 x z (FDM {fdm:.3f}); their sum is the raw forecast")
    _forecast(a2, f, pal)
    _title(a2, "Forecast = clip(sum, \N{MINUS SIGN}1, 1)")
    a2.set_xlabel("date (UTC)")
    _date_axis(a2)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown for 2020)", pal)
    return fig


@register(PAGE, "ema-cross", needs=("data_store",))
def ema_cross(pal):
    """Close with the EMA32/EMA128 of log price, and ema_cross forecasts at three speeds."""
    md = _md("H1")
    lc = log_close(md.bars)
    close = _win(md.bars["close"], COVID_TURN)
    fig, (a0, a1) = panels(2, heights=(3, 2.2), figsize=FIGSIZE_TALL)
    items = [(_price(a0, close, pal), "close")]
    for i, (span, name) in enumerate(((32, "EMA 32"), (128, "EMA 128")), start=1):
        e = _win(np.exp(lc.ewm(span=span, adjust=False).mean()), COVID_TURN)
        line, = a0.plot(e.index, e.to_numpy(), color=pal["categorical"][i])
        items.append((line, name))
    _end_labels(a0, items, pal, gap=0.07)
    _title(a0, "ema_cross on XAUUSD H1, Feb-May 2020: close and the default EMA pair (of log price)")
    speeds = ((16, 64), (32, 128), (64, 256))
    items = []
    for i, (fa, sl) in enumerate(speeds):
        f = _win(get_strategy("ema_cross", fast=fa, slow=sl).generate(md), COVID_TURN)
        line = _forecast(a1, f, pal, color=pal["categorical"][i], fill=False)
        items.append((line, f"{fa}/{sl}" + (" (default)" if (fa, sl) == (32, 128) else "")))
    _end_labels(a1, items, pal, gap=0.13)
    _title(a1, "Forecast at three speeds (fast/slow spans)")
    a1.set_xlabel("date (UTC)")
    _date_axis(a1)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown Feb-May 2020)", pal)
    return fig


@register(PAGE, "kalman-trend", needs=("data_store",))
def kalman_trend(pal):
    """Close, the filtered Kalman slope inside the band where the forecast caps, and the forecast."""
    md = _md("H1")
    s = get_strategy("kalman_trend")
    p = s.params
    _, _, gnorm = s.filter_coefficients()
    sigma = bar_volatility(log_close(md.bars), p["vol_halflife"], p["vol_min_periods"])
    slope = s.slope(md.bars)
    f = _win(s.generate(md), COVID_TURN)
    close = _win(md.bars["close"], COVID_TURN)

    fig, (a0, a1, a2) = panels(3, heights=(2.4, 2.0, 1.8), figsize=FIGSIZE_TALL)
    _price(a0, close, pal)
    _title(a0, "kalman_trend on XAUUSD H1, Feb-May 2020: close")
    sl = _win(slope, COVID_TURN) * 1e4
    band = _win(sigma * gnorm / Z, COVID_TURN) * 1e4        # |slope| above this: forecast capped
    a1.fill_between(band.index, -band.to_numpy(), band.to_numpy(), color=pal["grid"], linewidth=0, zorder=0)
    a1.plot(sl.index, sl.to_numpy(), color=pal["categorical"][0])
    zero_line(a1)
    a1.annotate("band: |slope| = sigma ||g|| / 0.6267, where the forecast reaches \N{PLUS-MINUS SIGN}1",
                (0.01, 0.96), xycoords="axes fraction", fontsize=8, color=pal["text_secondary"], va="top")
    a1.set_ylim(top=float(band.max()) * 1.45)    # head room for the label above the band
    a1.set_ylabel("bp per bar")
    _title(a1, "Filtered slope of log price (steady-state Kalman gain, lookback 240)")
    _forecast(a2, f, pal)
    _title(a2, "Forecast = clip(0.6267 x slope / (sigma x ||g||), \N{MINUS SIGN}1, 1)")
    a2.set_xlabel("date (UTC)")
    _date_axis(a2)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown Feb-May 2020)", pal)
    return fig


@register(PAGE, "donchian", needs=("data_store",))
def donchian(pal):
    """Close with 120-bar entry and 60-bar exit channels, the fixed entry ATR stop, and the forecast."""
    md = _md("H1")
    s = get_strategy("donchian")
    p = s.params
    b = md.bars
    pos = pd.Series(s.positions(b), index=b.index)
    upper = b["high"].rolling(p["entry_n"], min_periods=p["entry_n"]).max().shift(1)
    lower = b["low"].rolling(p["entry_n"], min_periods=p["entry_n"]).min().shift(1)
    ex_hi = b["high"].rolling(p["exit_n"], min_periods=p["exit_n"]).max().shift(1)
    ex_lo = b["low"].rolling(p["exit_n"], min_periods=p["exit_n"]).min().shift(1)
    a = atr(b, p["atr_n"])
    # the stop is fixed at entry: close_entry -/+ stop_atr x ATR_entry
    stop = pd.Series(np.nan, index=b.index)
    entry = pos.ne(pos.shift(1)) & pos.ne(0)
    level = (b["close"] - pos * float(p["stop_atr"]) * a).where(entry)
    stop = level.ffill().where(pos.ne(0))
    f = _win(s.generate(md), SUMMER_2020)
    close = _win(b["close"], SUMMER_2020)
    in_pos = _win(pos, SUMMER_2020)

    fig, (a0, a1) = panels(2, heights=(3.4, 1.6), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    a0.plot(close.index, close.to_numpy(), color=cat[0], label="close")
    for i, (u, d, name) in enumerate(((upper, lower, "entry channel (120 bars)"),
                                      (ex_hi, ex_lo, "exit channel (60 bars)"))):
        col = cat[1] if i == 0 else cat[5]
        uu, dd = _win(u, SUMMER_2020), _win(d, SUMMER_2020)
        a0.plot(uu.index, uu.to_numpy(), color=col, linewidth=1.2, label=name)
        a0.plot(dd.index, dd.to_numpy(), color=col, linewidth=1.2)
    st = _win(stop, SUMMER_2020)
    a0.plot(st.index, st.to_numpy(), color=cat[6], linewidth=1.2, linestyle=(0, (3, 2)),
            label="stop (2.5 ATR from entry)")
    a0.set_ylabel("USD per ounce")
    add_legend(a0, loc="lower right", ncol=2, fontsize=8)
    _shade(a0, in_pos.ne(0))
    _title(a0, "donchian on XAUUSD H1, Jun-Sep 2020: channels, stop, and time in a position (shaded)")
    _forecast(a1, f, pal, ticks=(-0.8, 0, 0.8), ylim=(-1.0, 1.0))
    _title(a1, "Forecast = \N{PLUS-MINUS SIGN}0.8 while in a position")
    a1.set_xlabel("date (UTC)")
    _date_axis(a1)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown Jun-Sep 2020)", pal)
    return fig


# ---------------------------------------------------------------------------------------
# mean reversion
# ---------------------------------------------------------------------------------------
@register(PAGE, "zscore-fade", needs=("data_store",))
def zscore_fade(pal):
    """Close with mean and +/-2 sd, the z-score, the previous-bar efficiency-ratio gate and the forecast."""
    md = _md("H1")
    s = get_strategy("zscore_fade")
    p = s.params
    b = md.bars
    lc = log_close(b)
    roll = lc.rolling(p["n"], min_periods=p["n"])
    mid, sd = roll.mean(), roll.std(ddof=1)
    z = rolling_zscore(lc, p["n"])
    er = efficiency_ratio(b["close"], p["er_n"]).shift(1)
    W = RANGE_2023
    idx = _win(b, W).index
    f = _positional(s.generate(md), W)
    close = _positional(b["close"], W)
    gate_closed = _positional(er >= p["er_max"], W)

    fig, (a0, a1, a2, a3) = panels(4, heights=(2.6, 1.7, 1.2, 1.6), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    for ax in (a0, a1, a2, a3):
        _shade(ax, gate_closed)        # explained in the ER panel title (a label here hit the band)
    lc_, = a0.plot(close.index, close.to_numpy(), color=cat[0])
    m = _positional(np.exp(mid), W)
    lm, = a0.plot(m.index, m.to_numpy(), color=cat[1], linewidth=1.2)
    for sign in (-1, 1):
        band = _positional(np.exp(mid + sign * p["entry_z"] * sd), W)
        lb, = a0.plot(band.index, band.to_numpy(), color=cat[1], linewidth=1.0, linestyle=(0, (3, 2)))
    a0.set_ylabel("USD per ounce")
    _end_labels(a0, [(lc_, "close"), (lm, "48-bar mean"), (lb, "\N{PLUS-MINUS SIGN}2 sd")], pal, gap=0.09)
    _title(a0, "zscore_fade on XAUUSD H1, 17-28 Jul 2023: close, mean and \N{PLUS-MINUS SIGN}2 sd of log price")
    zz = _positional(z, W)
    a1.plot(zz.index, zz.to_numpy(), color=cat[0])
    for v, t in ((p["entry_z"], "entry \N{PLUS-MINUS SIGN}2"), (-p["entry_z"], None),
                 (p["exit_z"], "exit \N{PLUS-MINUS SIGN}0.5"), (-p["exit_z"], None)):
        _hline(a1, v, pal, t)
    a1.set_ylabel("z")
    _title(a1, "z-score of log price (48 bars)")
    ee = _positional(er, W)
    a2.plot(ee.index, ee.to_numpy(), color=cat[0])
    _hline(a2, p["er_max"], pal, "er_max 0.3")
    a2.set_ylim(0, 1)
    a2.set_ylabel("ER")
    _title(a2, "Efficiency ratio (48 bars, to the previous bar); shaded: gate closed, ER \N{GREATER-THAN OR EQUAL TO} 0.3")
    _forecast(a3, f, pal)
    _title(a3, "Forecast = side x min(1, |z| / 4) while in a trade")
    a3.set_xlabel("H1 bars, weekend squeezed out (ticks at UTC day starts)")
    _bar_axis(a3, idx)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown 17-28 Jul 2023)", pal)
    return fig


@register(PAGE, "rsi2", needs=("data_store",))
def rsi2(pal):
    """Close with the SMA200 trend filter and SMA5 exit, RSI(2) with 10/90 thresholds, and the forecast."""
    md = _md("H1")
    s = get_strategy("rsi2")
    p = s.params
    close_all = md.bars["close"].astype(float)
    sma_t = close_all.rolling(p["trend_n"], min_periods=p["trend_n"]).mean()
    sma_x = close_all.rolling(p["exit_n"], min_periods=p["exit_n"]).mean()
    r = rsi(close_all, p["rsi_n"])
    W = RANGE_2023
    idx = _win(md.bars, W).index
    f = _positional(s.generate(md), W)
    close = _positional(close_all, W)

    fig, (a0, a1, a2) = panels(3, heights=(2.6, 1.8, 1.6), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    lc_, = a0.plot(close.index, close.to_numpy(), color=cat[0])
    st, sx = _positional(sma_t, W), _positional(sma_x, W)
    t_, = a0.plot(st.index, st.to_numpy(), color=cat[1], linewidth=1.4)
    x_, = a0.plot(sx.index, sx.to_numpy(), color=cat[5], linewidth=1.0)
    a0.set_ylabel("USD per ounce")
    _end_labels(a0, [(lc_, "close"), (t_, "SMA 200 (trend)"), (x_, "SMA 5 (exit)")], pal, gap=0.09)
    _title(a0, "rsi2 on XAUUSD H1, 17-28 Jul 2023: close, trend filter and exit average")
    rr = _positional(r, W)
    a1.plot(rr.index, rr.to_numpy(), color=cat[0], linewidth=1.2)
    _hline(a1, p["lower"], pal, "buy below 10 (if above SMA 200)", va="top")
    _hline(a1, p["upper"], pal, "sell above 90 (if below SMA 200)")
    a1.set_ylim(0, 100)
    a1.set_yticks([0, 10, 50, 90, 100])
    a1.set_ylabel("RSI(2)")
    _title(a1, "Wilder RSI over 2 bars")
    _forecast(a2, f, pal)
    _title(a2, "Forecast (size set by RSI depth at entry, then held to the SMA 5 exit)")
    a2.set_xlabel("H1 bars, weekend squeezed out (ticks at UTC day starts)")
    _bar_axis(a2, idx)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown 17-28 Jul 2023)", pal)
    return fig


@register(PAGE, "bollinger-revert", needs=("data_store",))
def bollinger_revert(pal):
    """Close with the SMA48 middle, +/-2 sd bands and +/-3.5 sd stop bands, and the decaying forecast."""
    md = _md("H1")
    s = get_strategy("bollinger_revert")
    p = s.params
    close_all = md.bars["close"].astype(float)
    roll = close_all.rolling(p["n"], min_periods=p["n"])
    mid, sd = roll.mean(), roll.std(ddof=0)
    W = RANGE_2023
    idx = _win(md.bars, W).index
    f = _positional(s.generate(md), W)
    close = _positional(close_all, W)

    fig, (a0, a1) = panels(2, heights=(3.2, 1.7), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    lc_, = a0.plot(close.index, close.to_numpy(), color=cat[0])
    mm = _positional(mid, W)
    lm, = a0.plot(mm.index, mm.to_numpy(), color=cat[1], linewidth=1.2)
    items = [(lc_, "close"), (lm, "middle (SMA 48)")]
    for k, ls, col, name in ((p["k"], "-", cat[1], "\N{PLUS-MINUS SIGN}2 sd band"),
                             (p["stop_k"], (0, (3, 2)), cat[6], "\N{PLUS-MINUS SIGN}3.5 sd stop")):
        for sign in (-1, 1):
            band = _positional(mid + sign * k * sd, W)
            ln, = a0.plot(band.index, band.to_numpy(), color=col, linewidth=1.0, linestyle=ls)
        items.append((ln, name))
    a0.set_ylabel("USD per ounce")
    _end_labels(a0, items, pal, gap=0.08)
    _title(a0, "bollinger_revert on XAUUSD H1, 17-28 Jul 2023: bands of the close (48 bars)")
    _forecast(a1, f, pal)
    _title(a1, "Forecast = side x min(1, |close - mid| / (2 sd)), entered after a close back inside")
    a1.set_xlabel("H1 bars, weekend squeezed out (ticks at UTC day starts)")
    _bar_axis(a1, idx)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown 17-28 Jul 2023)", pal)
    return fig


# ---------------------------------------------------------------------------------------
# breakout
# ---------------------------------------------------------------------------------------
@register(PAGE, "vol-squeeze", needs=("data_store",))
def vol_squeeze(pal):
    """Close with Bollinger and Keltner bands, squeeze spans, 20-bar momentum z and the forecast."""
    md = _md("H1")
    s = get_strategy("vol_squeeze")
    p = s.params
    b = md.bars
    close_all = b["close"].astype(float)
    roll = close_all.rolling(p["n"], min_periods=p["n"])
    mid, sd = roll.mean(), roll.std(ddof=0)
    ema = close_all.ewm(span=p["n"], adjust=False, min_periods=p["n"]).mean()
    width = p["kc_mult"] * atr(b, p["atr_n"])
    on = pd.Series(s.squeeze_on(b), index=b.index)
    lc = log_close(b)
    mom = momentum_z(lc, bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"]), p["mom_n"])
    W = RANGE_2023
    idx = _win(b, W).index
    f = _positional(s.generate(md), W)
    close = _positional(close_all, W)
    sq = _positional(on, W)

    fig, (a0, a1, a2) = panels(3, heights=(2.8, 1.7, 1.6), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    for ax in (a0, a1, a2):
        _shade(ax, sq, "squeeze on" if ax is a0 else None)
    lc_, = a0.plot(close.index, close.to_numpy(), color=cat[0])
    items = [(lc_, "close")]
    for col, up, dn, name in ((cat[1], mid + p["bb_k"] * sd, mid - p["bb_k"] * sd, "Bollinger 20, 2 sd"),
                              (cat[5], ema + width, ema - width, "Keltner 20, 1.5 ATR")):
        for band in (up, dn):
            bb_ = _positional(band, W)
            ln, = a0.plot(bb_.index, bb_.to_numpy(), color=col, linewidth=1.1)
        items.append((ln, name))
    a0.set_ylabel("USD per ounce")
    _end_labels(a0, items, pal, gap=0.08)
    _title(a0, "vol_squeeze on XAUUSD H1, 17-28 Jul 2023: Bollinger inside Keltner = squeeze (shaded)")
    mm = _positional(mom, W)
    a1.plot(mm.index, mm.to_numpy(), color=cat[0], linewidth=1.4)
    zero_line(a1)
    a1.set_ylabel("z")
    _title(a1, "20-bar momentum z: the side taken when a squeeze of 6+ bars releases")
    _forecast(a2, f, pal)
    _title(a2, "Forecast = side x min(1, 0.6267 |z|), held up to 36 bars or until z flips")
    a2.set_xlabel("H1 bars, weekend squeezed out (ticks at UTC day starts)")
    _bar_axis(a2, idx)
    source_note(fig, "Data: " + source_label("H1") + " (generated on 2012-2024, shown 17-28 Jul 2023)", pal)
    return fig


@register(PAGE, "orb", needs=("data_store",))
def orb(pal):
    """One week of H1 bars with the London and New York opening-range boxes and the orb forecast."""
    md = _md("H1")
    s = get_strategy("orb")
    b = md.bars
    f = _win(s.generate(md), ORB_WEEK)
    w = _win(b, ORB_WEEK)
    fig, (a0, a1) = panels(2, heights=(3.2, 1.6), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    a0.vlines(w.index, w["low"].to_numpy(), w["high"].to_numpy(), color=pal["text_muted"], linewidth=0.8)
    a0.plot(w.index, w["close"].to_numpy(), color=cat[0], label="close")
    table = s.session_table()
    for i, name in enumerate(s.params["sessions"]):
        tz, op, end = table[name]
        col = cat[1] if i == 0 else cat[5]
        loc = w.index.tz_convert(ZoneInfo(tz))
        h0, m0 = (int(v) for v in op.split(":"))
        h1, m1 = (int(v) for v in end.split(":"))
        first = True
        for day in sorted(set(loc.normalize())):
            or_open = day + pd.Timedelta(hours=h0, minutes=m0)
            or_end = or_open + pd.Timedelta(minutes=int(s.params["range_minutes"]))
            sess_end = day + pd.Timedelta(hours=h1, minutes=m1)
            bar = w.loc[(loc >= or_open) & (loc < or_end)]
            if bar.empty:
                continue
            hi, lo = float(bar["high"].max()), float(bar["low"].min())
            x0 = or_open.tz_convert("UTC")
            x1 = sess_end.tz_convert("UTC")
            a0.fill_between([x0, or_end.tz_convert("UTC")], lo, hi, color=col, alpha=0.35, linewidth=0)
            city = tz.split("/")[-1].replace("_", " ")
            a0.hlines([hi, lo], x0, x1, color=col, linewidth=1.3,
                      label=f"{name}: range of the {op} {city} bar, traded until {end} {city}"
                      if first else None)
            first = False
    a0.set_ylabel("USD per ounce")
    add_legend(a0, loc="upper left", fontsize=8)
    _title(a0, "orb on XAUUSD H1, 4-8 Mar 2024: bar ranges, close and the two opening ranges")
    _forecast(a1, f, pal)
    _title(a1, "Forecast = sum of \N{PLUS-MINUS SIGN}0.5 per active session")
    a1.set_xlabel("date (UTC)")
    _date_axis(a1)
    source_note(fig, "Data: " + source_label("H1") + ", week of 4 Mar 2024", pal)
    return fig


# ---------------------------------------------------------------------------------------
# macro
# ---------------------------------------------------------------------------------------
@register(PAGE, "macro-factor", needs=("data_store",))
def macro_factor(pal):
    """Gold close, 10y real yield, DXY and the macro_factor forecast through the 2022 hiking cycle."""
    md = _md("H1")
    s = get_strategy("macro_factor")
    f = _win(s.generate(md), HIKING)
    close = _win(md.bars["close"], HIKING)
    real = _win(md.macro["real10y"]["value"], HIKING)
    dxy = _win(md.macro["dxy"]["value"], HIKING)
    fig, (a0, a1, a2, a3) = panels(4, heights=(2.2, 1.5, 1.5, 1.7), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    _price(a0, close, pal)
    _title(a0, "macro_factor, Jul 2021 - Jun 2023: gold close (H1)")
    a1.plot(real.index, real.to_numpy(), color=cat[1])
    a1.set_ylabel("percent")
    _title(a1, "Input 1: 10-year real yield (FRED DFII10), sign -1")
    a2.plot(dxy.index, dxy.to_numpy(), color=cat[5])
    a2.set_ylabel("index")
    _title(a2, "Input 2: US dollar index (DXY), sign -1")
    _forecast(a3, f, pal)
    _title(a3, "Forecast: long when both have been falling (1w/1m/3m momentum)")
    a3.set_xlabel("date (UTC)")
    _date_axis(a3)
    source_note(fig, "Data: " + source_label("H1") + "; " + macro_label(["real10y", "dxy"]), pal)
    return fig


@register(PAGE, "risk-off", needs=("data_store",))
def risk_off(pal):
    """Gold close, the VIX spike z with hysteresis thresholds, the DXY filter and the long-only forecast."""
    md = _md("H1")
    s = get_strategy("risk_off")
    p = s.params
    daily = s.daily_signal(md.macro["vix"])
    g = clean_macro(md.macro["dxy"], log_values=True)
    gz = macro_momentum_z(g["x"], (p["dxy_horizon"],), p["halflife"], p["min_obs"])
    vz = pd.Series(daily["z"].to_numpy(), index=pd.DatetimeIndex(daily["available_at"]))
    dz = pd.Series(gz.to_numpy(), index=pd.DatetimeIndex(g["available_at"]))
    f = _win(s.generate(md), RISK_OFF)
    close = _win(md.bars["close"], RISK_OFF)
    fig, (a0, a1, a2, a3) = panels(4, heights=(2.2, 1.6, 1.5, 1.6), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    blocked = _win(dz > p["dxy_z_max"], RISK_OFF)
    for ax in (a0, a1, a2, a3):
        _shade(ax, blocked, "dollar squeeze: blocked" if ax is a0 else None)
    _price(a0, close, pal)
    _title(a0, "risk_off, Jan-May 2020: gold close (H1)")
    v = _win(vz, RISK_OFF)
    a1.step(v.index, v.to_numpy(), where="post", color=cat[0])
    _hline(a1, p["entry_z"], pal, "entry 1.5")
    _hline(a1, p["exit_z"], pal, "exit 0.5", va="top")
    a1.set_ylabel("z")
    _title(a1, "VIX spike z = (ln VIX - EWMA mean) / EWMA sd, both to the previous print")
    d = _win(dz, RISK_OFF)
    a2.step(d.index, d.to_numpy(), where="post", color=cat[1])
    _hline(a2, p["dxy_z_max"], pal, "dxy_z_max 2.0")
    zero_line(a2)
    a2.set_ylabel("z")
    _title(a2, "Dollar filter: 5-day DXY momentum z")
    _forecast(a3, f, pal, ylim=(-0.1, 1.1), ticks=(0, 0.5, 1))
    _title(a3, "Forecast (long only), on H1 bars at each print's available_at")
    a3.set_xlabel("date (UTC)")
    _date_axis(a3)
    source_note(fig, "Data: " + source_label("H1") + "; " + macro_label(["vix", "dxy"]), pal)
    return fig


# ---------------------------------------------------------------------------------------
# learned seasonality
# ---------------------------------------------------------------------------------------
@register(PAGE, "seasonal-profile", needs=("data_store",))
def seasonal_profile(pal):
    """Mean vol-normalised next-bar return by New York hour and weekday, with 95% CIs."""
    md = _md("H1")
    s = get_strategy("intraday_seasonality")
    tt = s.training_targets(md)
    hour = (tt["key"] % 1440) // 60
    wd = tt["key"] // 1440
    fig = Figure(figsize=FIGSIZE)
    a0, a1 = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3.2, 1.0]})
    for ax in (a0, a1):
        style_axes(ax)
    blue = pal["categorical"][0]
    for ax, key, labels in ((a0, hour, [str(h) for h in range(24)]),
                            (a1, wd, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])):
        g = tt["y"].groupby(key)
        st = pd.DataFrame({"m": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.size()), "n": g.size()})
        if ax is a1:
            st = st.reindex([6, 0, 1, 2, 3, 4]).dropna()
        xs = np.arange(len(st))
        ax.bar(xs, st["m"].to_numpy(), color=blue, width=0.75, linewidth=0)
        ax.errorbar(xs, st["m"].to_numpy(), yerr=1.96 * st["se"].to_numpy(), fmt="none",
                    ecolor=pal["text_secondary"], elinewidth=1.0, capsize=2)
        zero_line(ax)
        ax.set_xticks(xs, [labels[int(i)] for i in st.index])
        ax.set_xlim(-0.7, len(st) - 0.3)
    a0.set_ylabel("mean next-bar return / sigma")
    a0.set_xlabel("hour the held bar opens (New York time)")
    a1.set_xlabel("weekday (NY)")
    a1.tick_params(axis="y", labelleft=False)
    _title(a0, "Gold's clock: open-to-close return of the next H1 bar, in units of its sigma")
    a1.set_title("")
    source_note(fig, "Data: " + source_label("H1") + "; bars are means, whiskers 95% CIs (1.96 se)", pal)
    return fig


@register(PAGE, "seasonal-table", needs=("data_store", "runs/protocol/default_selection"))
def seasonal_table(pal):
    """Fitted hour-of-week table on fold 0's training window: frictionless vs cost-aware."""
    folds = run_table("protocol/default_selection", "folds.csv", cut=False)
    r = folds.iloc[0]
    md = _md("H1")
    tr = MarketData(bars=md.bars.loc[r["train_start"]:r["train_end"]], macro=md.macro, events=md.events)
    s = get_strategy("intraday_seasonality")
    s.fit(tr)
    fs = s.fit_summary_
    tt = s.training_targets(tr)
    g = tt.groupby("key")["y"]
    st = pd.DataFrame({"n": g.size(), "mean": g.mean()})
    st = st[st["n"] >= max(2, int(s.params["min_obs"]))]
    shrink = pd.Series(fs["shrinkage_by_key"])
    effect = shrink.reindex(st.index).to_numpy() * (st["mean"].to_numpy() - fs["centre"])
    raw = np.clip((st["mean"].to_numpy() - fs["centre"]) * fs["scalar"], -1, 1)
    friction_free = np.clip(effect * fs["scalar"], -1, 1)
    pos = np.array([s.table_[int(k)] for k in st.index])
    # x = hours since Sunday 00:00 NY, so the week reads left to right from the Sunday open
    x = ((st.index.to_numpy() // 1440 + 1) % 7) * 24 + (st.index.to_numpy() % 1440) / 60.0

    fig, ax = figure(figsize=FIGSIZE)
    cat = pal["categorical"]
    ax.bar(x, raw, width=0.9, color=pal["grid"], linewidth=0, label="raw demeaned mean x scalar (capped)")
    ax.bar(x, friction_free, width=0.9, color=cat[0], linewidth=0, label="shrunk table (frictionless)")
    order = np.argsort(x)
    ax.step(x[order], pos[order], where="post", color=cat[1], label="cost-aware position (traded)")
    zero_line(ax)
    ax.set_ylim(-1.1, 1.75)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    lo_x, hi_x = x.min() - 1, x.max() + 1
    centres = [(max(24 * i, lo_x) + min(24 * (i + 1), hi_x)) / 2 for i in range(6)]
    ax.set_xticks(centres, ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri"])
    for i in range(1, 6):
        ax.axvline(24 * i, color=pal["grid"], linewidth=0.8, zorder=0)
    ax.set_xlim(lo_x, hi_x)
    ax.set_xlabel("hour of the week (New York time) of the bar the position is held over")
    ax.set_ylabel("forecast")
    add_legend(ax, loc="upper left", fontsize=8, ncol=3)
    a, z = str(r["train_start"])[:10], str(r["train_end"])[:10]
    ax.annotate(f"Welch p = {fs['q_pvalue']:.1e}, mean shrinkage B = {fs['mean_shrinkage']:.2f}; "
                f"turnover per week: frictionless {fs['frictionless_turnover_per_week']:.1f}, "
                f"cost-aware {fs['orbit_turnover_per_week']:.2f}",
                (0.01, 0.86), xycoords="axes fraction", va="top", ha="left", fontsize=8,
                color=pal["text_secondary"])
    ax.set_title("intraday_seasonality fitted on protocol fold 0 (training window only)")
    source_note(fig, f"Data: XAUUSD H1, Dukascopy bid/ask mid, training window {a} to {z} "
                     "(fold 0 of runs/protocol/default_selection)", pal)
    return fig


# ---------------------------------------------------------------------------------------
# ML / RL forecast contract
# ---------------------------------------------------------------------------------------
@register(PAGE, "ml-forecast-maps", needs=("runs/protocol/default_selection",))
def ml_forecast_maps(pal):
    """How ml_gbm and meta_label map a probability to a forecast, and the skill gate OOS."""
    fig = Figure(figsize=FIGSIZE_TALL)
    axd = fig.subplot_mosaic([["a", "b"], ["c", "c"]], height_ratios=[1.15, 1.0])
    for ax in axd.values():
        style_axes(ax)
    cat = pal["categorical"]
    p = np.linspace(0.0, 1.0, 1001)
    a = axd["a"]
    items = []
    for i, sc in enumerate((1.0, 4.0, 12.0)):
        y = np.clip(sc * apply_dead_zone(2.0 * p - 1.0, 0.02), -1, 1)
        line, = a.plot(p, y, color=cat[i])
        items.append((line, f"scalar {sc:g}"))
    gate, = a.plot(p, np.zeros_like(p), color=cat[3])
    items.append((gate, "gate failed: 0"))
    zero_line(a)
    a.set_xlim(0, 1)
    a.set_ylim(-1.1, 1.1)
    # labels where the lines are far apart: right of scalar 1 and 4 at forecast 0.6, left of 12 at 0.85
    for (line, text), y_at, side in zip(items, (0.6, 0.6, 0.85, None), ("left", "left", "right", "left"),
                                        strict=True):
        if y_at is None:
            a.annotate(text, (0.70, 0.0), xytext=(0, -4), textcoords="offset points", fontsize=8,
                       color=pal["text"], va="top")
            continue
        yy = np.asarray(line.get_ydata())
        xi = float(p[int(np.argmax(yy >= y_at))])
        a.annotate(text, (xi, y_at), xytext=(5 if side == "left" else -5, 0), textcoords="offset points",
                   fontsize=8, color=pal["text"], va="center", ha=side)
    a.set_xlabel("calibrated P(up)")
    a.set_ylabel("forecast")
    _title(a, "ml_gbm: clip(scalar x dead-zone(2p - 1))")

    b = axd["b"]
    m_cont = np.where(p >= 0.5, np.maximum(bet_size_from_probability(p), 0.0), 0.0)
    m_step = discretize_bet(np.clip(m_cont, 0.0, 1.0), 0.1)
    l1, = b.plot(p, m_cont, color=cat[1], linewidth=1.2)
    l2, = b.step(p, m_step, where="post", color=cat[0])
    b.set_xlim(0, 1)
    b.set_ylim(-0.05, 1.08)
    b.annotate("step 0.1 (default)", (0.64, float(np.interp(0.64, p, m_step))), xytext=(6, -4),
               textcoords="offset points", fontsize=8, color=pal["text"], va="top")
    b.annotate("continuous", (0.56, float(np.interp(0.56, p, m_cont))), xytext=(-6, 4),
               textcoords="offset points", fontsize=8, color=pal["text"], va="bottom", ha="right")
    del l1, l2
    b.set_xlabel("calibrated P(primary bet succeeds)")
    b.set_ylabel("bet size m")
    _title(b, "meta_label: m = 2 Phi(z) - 1, 0 below p 0.5")

    c = axd["c"]
    fc = run_table("protocol/default_selection", "oos_forecasts.parquet")
    act = fc.groupby("fold")[["ml_gbm", "meta_label", "intraday_seasonality"]].apply(lambda d: (d != 0).mean())
    act.index = act.index.astype(int)
    act = act.sort_index()
    xs = np.arange(len(act))
    c.bar(xs - 0.2, act["meta_label"].to_numpy(), width=0.4, color=cat[0], linewidth=0, label="meta_label")
    c.bar(xs + 0.2, act["intraday_seasonality"].to_numpy(), width=0.4, color=cat[1], linewidth=0,
          label="intraday_seasonality")
    percent_axis(c)
    c.set_ylim(0, 1.05)
    c.set_xticks(xs, [str(k) for k in act.index])
    c.set_xlabel("walk-forward fold (6-month test blocks, Dec 2015 to Dec 2024)")
    c.set_ylabel("OOS bars with forecast \N{NOT EQUAL TO} 0")
    n_ml = int((act["ml_gbm"] > 0).sum())
    _title(c, f"Gates in action out of sample: ml_gbm traded in {n_ml} of {len(act)} folds")
    add_legend(c, loc="upper right", fontsize=8)
    source_note(fig, "Top: the code's mapping functions. Bottom: "
                + run_label("protocol/default_selection"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# managed exposure and regime conditioning (v2.1)
# ---------------------------------------------------------------------------------------
@register(PAGE, "managed-gold", needs=("data_store",))
def managed_gold(pal):
    """Close and EMA with de-risk spans, the 5d/126d vol ratio with its latch lines, and the forecast."""
    md = _md("D1_nyclose")
    s = get_strategy("managed_gold", **MANAGED_GOLD_D1)
    comp = s.components(md).loc["2013-01-01":]
    close = md.bars["close"].loc["2013-01-01":]
    derisk = comp["derisk"] > 0
    fig, (a0, a1, a2) = panels(3, heights=(2.8, 1.7, 1.7), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    for ax in (a0, a1, a2):
        _shade(ax, derisk, "de-risked" if ax is a0 else None)
    lc_, = a0.plot(close.index, close.to_numpy(), color=cat[0])
    le, = a0.plot(comp.index, np.exp(comp["ema"].to_numpy()), color=cat[1], linewidth=1.4)
    _end_labels(a0, [(lc_, "close"), (le, "EMA (69-day\nhalf-life)")], pal, gap=0.12)
    a0.set_ylabel("USD per ounce")
    _title(a0, f"managed_gold, daily NY-close bars 2013-2024: de-risked on {derisk.mean():.1%} of days")
    a1.plot(comp.index, comp["vol_ratio"].to_numpy(), color=cat[0], linewidth=1.2)
    _hline(a1, s.params["vol_enter"], pal, "enter 1.25")
    _hline(a1, s.params["vol_exit"], pal, "exit 1.0", va="top")
    a1.set_ylabel("ratio")
    _title(a1, "Volatility ratio: 5-day / 126-day half-life EWMA vol")
    _forecast(a2, comp["forecast"], pal, ylim=(-0.05, 1.08), ticks=(0, 0.5, 1))
    _title(a2, "Forecast = clip(0.5 + 0.5 x trend, 0, 1), cut to 0 while de-risked")
    a2.set_xlabel("date (UTC)")
    _date_axis(a2)
    source_note(fig, "Data: " + source_label("D1_nyclose", start="2013-01-01")
                + "; parameters of configs/managed_gold.yaml", pal)
    return fig


def _regime_by_fold() -> pd.DataFrame:
    """P(chaotic), scale, base and scaled forecast per OOS day, with the HMM refitted on each
    fold's training window from runs/v21_managed_gold/folds.csv (as the walk-forward did)."""
    md = _md("D1_nyclose", start="2013-01-01")
    folds = run_table("v21_managed_gold", "folds.csv", cut=False)
    idx = md.bars.index
    rows = []
    for _, r in folds.iterrows():
        tr0, tr1, te0, te1 = (pd.Timestamp(r[k]) for k in ("train_start", "train_end", "test_start", "test_end"))
        s = get_strategy("regime_switch", **REGIME_SWITCH_D1)
        s.fit(MarketData(bars=md.bars.loc[tr0:tr1], macro=md.macro, events=md.events))
        hist = MarketData(bars=md.bars.loc[idx[0]:te1], macro=md.macro, events=md.events)
        rp = s.regime_probabilities(hist).loc[te0:te1]
        base = s.base_.generate(hist).loc[te0:te1]
        f = s.generate(hist).loc[te0:te1]
        rows.append(pd.DataFrame({"p_chaotic": rp["p_chaotic"], "scale": rp["scale"], "base": base,
                                  "forecast": f, "fold": int(r["fold"])}))
    out = pd.concat(rows)
    close = md.bars["close"]
    out["ret"] = (100.0 * np.log(close / close.shift(1))).reindex(out.index)
    return out


@register(PAGE, "regime-switch", needs=("data_store", "runs/v21_managed_gold"))
def regime_switch(pal):
    d = _regime_by_fold()
    chaotic = d["p_chaotic"] > 0.5
    fig, (a0, a1, a2) = panels(3, heights=(2.2, 1.6, 2.0), figsize=FIGSIZE_TALL)
    cat = pal["categorical"]
    _shade(a0, chaotic, "shaded: P(chaotic) > 0.5")
    a0.vlines(d.index, 0.0, d["ret"].to_numpy(), color=cat[0], linewidth=0.7)
    zero_line(a0)
    a0.set_ylabel("percent")
    _title(a0, "regime_switch: daily log returns of gold (NY close), out of sample 2016-2024")
    a1.fill_between(d.index, 0.0, d["p_chaotic"].to_numpy(), step="post", color=cat[1], alpha=0.15, linewidth=0)
    a1.step(d.index, d["p_chaotic"].to_numpy(), where="post", color=cat[1], linewidth=0.7)
    _hline(a1, 1 - REGIME_SWITCH_D1["p_flat"], pal, "flat above 0.5")
    _hline(a1, 1 - REGIME_SWITCH_D1["p_full"], pal, "full size below 0.1", va="top")
    a1.set_ylim(0, 1.02)
    a1.set_ylabel("probability")
    _title(a1, "Filtered P(chaotic | returns up to today), HMM refitted on each fold's 3-year window")
    a2.step(d.index, d["base"].to_numpy(), where="post", color=cat[1], linewidth=0.9, label="base: tsmom")
    _forecast(a2, d["forecast"], pal, label="regime_switch = base x scale").set_linewidth(1.2)
    a2.set_ylim(-1.55, 1.12)                     # room under the -1 floor for the legend
    add_legend(a2, loc="lower left", fontsize=8, ncol=2)
    _title(a2, "Base forecast and the regime-scaled forecast")
    a2.set_xlabel("date (UTC)")
    _date_axis(a2)
    source_note(fig, "Data: XAUUSD D1 NY close, Dukascopy, 2016-2024; folds of runs/v21_managed_gold", pal)
    return fig


@register(PAGE, "regime-ramp")
def regime_ramp(pal):
    """The piecewise-linear regime_scale multiplier against the calm-state probability."""
    p_flat, p_full = REGIME_SWITCH_D1["p_flat"], REGIME_SWITCH_D1["p_full"]
    x = np.linspace(0, 1, 501)
    y = regime_scale(x, p_flat, p_full)
    fig, ax = figure(figsize=FIGSIZE_SHORT)
    ax.axvspan(0, p_flat, color=pal["grid"], linewidth=0, zorder=0)
    ax.plot(x, y, color=pal["categorical"][0])
    for v, t in ((p_flat, f"p_flat = {p_flat}"), (p_full, f"p_full = {p_full}")):
        ax.axvline(v, color=pal["text_secondary"], linewidth=0.9, linestyle=(0, (4, 3)))
        ax.annotate(t, (v, 1.02), xytext=(3, 0), textcoords="offset points", fontsize=8,
                    color=pal["text_secondary"], va="bottom")
    ax.annotate("flat: chaotic at least as likely as calm", (0.02, 0.5), fontsize=8.5,
                color=pal["text_muted"], va="center")
    ax.annotate("linear ramp", (0.74, 0.36), fontsize=8.5, color=pal["text"], ha="left", va="top")
    ax.annotate("full base\nforecast", (0.95, 0.94), fontsize=8.5, color=pal["text"], ha="center", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.18)
    ax.set_xlabel("p_ok = 1 - P(chaotic)  (= P(calm) with two states)")
    ax.set_ylabel("scale")
    ax.set_title("regime_switch: forecast multiplier as a function of the regime probability")
    return fig


# ---------------------------------------------------------------------------------------
# how the strategies relate
# ---------------------------------------------------------------------------------------
@register(PAGE, "forecast-correlation", needs=("runs/protocol/default_selection",))
def forecast_correlation(pal):
    """Lower-triangle heatmap of OOS hourly forecast correlations by strategy family."""
    fc = run_table("protocol/default_selection", "oos_forecasts.parquet")
    cols = [c for c in FAMILY_ORDER if c in fc.columns and fc[c].std() > 0]
    corr = fc[cols].corr()
    n = len(cols)
    mask = np.triu(np.ones((n, n), dtype=bool))
    data = np.where(mask, np.nan, corr.to_numpy())
    fig, ax = figure(figsize=(7.6, 6.6))
    style_axes(ax, grid=None)
    cmap = diverging_cmap(pal["mode"]).with_extremes(bad=pal["surface"])
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i):
            v = data[i, j]
            ax.text(j, i, signed(0.0 if abs(v) < 0.005 else v, "{:.2f}"), ha="center", va="center", fontsize=7,
                    color=ink_on(im.cmap(im.norm(v))))
    ax.set_xticks(range(n), cols, rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(range(n), cols, fontsize=8)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("correlation of hourly forecasts", color=pal["text_secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])
    ax.set_title("Out-of-sample forecast correlations, 13 strategies (ml_gbm was always 0)")
    source_note(fig, "Data: " + run_label("protocol/default_selection"), pal)
    return fig

