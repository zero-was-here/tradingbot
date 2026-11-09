"""Figures for docs/live-trading.md (page key ``live-trading``).

Drawn by aurum's own live code: ``aurum.live.monitor`` (PSI drift, heartbeat threshold),
``aurum.risk.manager`` (kill switch and its reset), ``aurum.live.mt5`` (server time-zone
rules) and the real feature pipeline on real XAUUSD H1 bars cut at 2024-12-31. Render with::

    python tools/figures/build.py --only live-trading
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_SHORT,
    FIGSIZE_TALL,
    market_data,
    panels,
    register,
    source_label,
    source_note,
)

from aurum.features.pipeline import FeaturePipeline
from aurum.live.monitor import PSI_ALERT, PSI_WARN, DriftMonitor, FeatureReference, heartbeat_max_age, psi
from aurum.live.mt5 import _tz_offset_now
from aurum.risk.manager import RiskLimits, StandardRiskManager
from aurum.viz.style import (
    add_legend,
    figure,
    label_line_end,
    percent_axis,
    shade_span,
    zero_line,
)

PAGE = "live-trading"
SYNTH = "Data: synthetic illustration (aurum.data.synthetic, seed 7), not market data."


# ---------------------------------------------------------------------------------------
# 1. heartbeat and watchdog
# ---------------------------------------------------------------------------------------
def _heartbeat_writes(*, every: float, wake: float, cycle: float, crash: float, t0: float,
                      t1: float) -> list[float]:
    """Heartbeat write times (s after a bar close) following LiveRunner.run/_sleep_until:
    one write at the end of each loop iteration, then one every ``every`` s of sleep."""
    writes: list[float] = []
    # previous iteration ended long before t0: sleeping, writing every `every` s up to the wake
    t = wake - every * np.ceil((wake - t0) / every)
    while t < wake:
        writes.append(t)
        t += every
    end_cycle = wake + cycle           # the decision cycle writes nothing, then the loop writes once
    t = end_cycle
    while t <= min(crash, t1):
        writes.append(t)
        t += every
    return writes


@register(PAGE, "heartbeat-watchdog")
def heartbeat_watchdog(pal):
    """Heartbeat writes around one H1 bar close, the heartbeat age and the watchdog threshold."""
    every = 30.0                      # heartbeat_seconds default = retry_poll_seconds = live.poll_seconds
    threshold = heartbeat_max_age(every)          # 3 * 30 + 120 = 210 s
    delay, cycle, crash = 5.0, 120.0, 600.0
    t0, t1 = -100.0, 900.0
    writes = _heartbeat_writes(every=every, wake=delay, cycle=cycle, crash=crash, t0=t0, t1=t1)
    ts = np.linspace(t0, t1, 4001)
    w = np.asarray(writes)
    last = np.array([w[w <= x].max() if (w <= x).any() else t0 - every for x in ts])
    age = ts - last
    alert_at = float(w[w <= crash].max() + threshold)

    blue, red = pal["categorical"][0], pal["categorical"][7]
    fig, (ax0, ax1) = panels(2, heights=(1, 2.2), figsize=FIGSIZE)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.spines["left"].set_visible(False)
    shade_span(ax0, delay, delay + cycle, "decision cycle: no heartbeat")
    shade_span(ax1, delay, delay + cycle)
    ax0.vlines(w, 0.15, 0.55, color=blue, linewidth=1.6)
    ax0.annotate(f"heartbeat writes: every {every:.0f} s while sleeping", (float(w[w > delay + cycle].min()), 0.55),
                 xytext=(0, 3), textcoords="offset points", fontsize=8, color=pal["text_secondary"], va="bottom")
    for x, lab, side in ((0.0, "bar close", -1), (crash, "runner dies", 1), (alert_at, "watchdog alert", 1)):
        ax0.axvline(x, color=pal["text_muted"], linewidth=0.9, linestyle=(0, (4, 3)))
        ax1.axvline(x, color=pal["text_muted"], linewidth=0.9, linestyle=(0, (4, 3)))
        ax0.annotate(lab, (x, 0.95), xytext=(3 * side, 0), textcoords="offset points", fontsize=8, va="top",
                     ha="left" if side > 0 else "right", color=pal["text"])
    ax1.plot(ts, age, color=blue)
    ax1.axhline(threshold, color=red, linewidth=1.5)
    ax1.annotate(f"watchdog threshold = 3 x {every:.0f} s + 120 s = {threshold:.0f} s", (delay + cycle, threshold),
                 xytext=(8, 4), textcoords="offset points", fontsize=8.5, color=pal["text"], va="bottom")
    peak = float(age[(ts > delay) & (ts < delay + cycle + 1)].max())
    ax1.annotate(f"peak age {peak:.0f} s:\nlast sleep write + {cycle:.0f} s cycle", (delay + cycle, peak),
                 xytext=(8, -2), textcoords="offset points", fontsize=8, color=pal["text_secondary"], va="top")
    ax1.set_ylim(0, threshold * 1.6)
    ax1.set_xlim(t0, t1)
    ax1.set_ylabel("heartbeat age (s)")
    ax1.set_xlabel("seconds after the bar close (bar_close_delay_seconds = 5, an illustrative 120 s LLM-desk cycle)")
    ax0.set_title("Heartbeats while the runner sleeps let a watchdog alert within minutes, not hours")
    source_note(fig, SYNTH.replace("synthetic illustration (aurum.data.synthetic, seed 7)",
                                   "illustration of LiveRunner's heartbeat schedule"), pal)
    return fig


# ---------------------------------------------------------------------------------------
# 2-3. PSI drift on the real feature pipeline
# ---------------------------------------------------------------------------------------
TRAIN = ("2015-01-01", "2019-12-31")
LIVE_FROM = "2020-01-01"
WINDOW, EVERY = 500, 24               # live.options.monitor drift_window / drift_every defaults
FEATURES = (
    ("returns_z_1", "returns_z_1 (1-bar return z-score)"),
    ("volatility_yz_24", "volatility_yz_24 (1-day Yang-Zhang vol)"),
    ("trend_ema_dist_200", "trend_ema_dist_200 (distance to 200-bar EMA)"),
    ("macro_real10y_z", "macro_real10y_z (US 10y real yield z-score)"),
)


@functools.cache
def _psi_data() -> dict:
    md = market_data("H1", start="2014-01-01")
    pipe = FeaturePipeline()                    # every group, robust scaler, clip 5 (the config default)
    raw = pipe.compute(md)
    pipe.fit(raw.loc[TRAIN[0]:TRAIN[1]])
    x = pipe.transform(raw, strict=False)
    cols = [c for c, _ in FEATURES]
    ref = FeatureReference.from_frame(x.loc[TRAIN[0]:TRAIN[1], cols])
    full_ref = FeatureReference.from_frame(x.loc[TRAIN[0]:TRAIN[1]])
    mon = DriftMonitor(ref)
    full_mon = DriftMonitor(full_ref)
    live = x.loc[LIVE_FROM:]
    rows, maxrows = [], []
    for end in range(WINDOW, len(live) + 1, EVERY):
        win = live.iloc[end - WINDOW:end]
        rep = mon.evaluate(win[cols])
        rows.append(pd.Series(rep.psi, name=win.index[-1]))
    for end in range(WINDOW, len(live) + 1, EVERY * 10):
        win = live.iloc[end - WINDOW:end]
        rep = full_mon.evaluate(win)
        maxrows.append((win.index[-1], rep.max_psi, len(rep.alert), len(rep.psi)))
    return {"psi": pd.DataFrame(rows), "ref": ref, "x": x[cols], "full": maxrows, "n_cols": len(full_ref.columns)}


@register(PAGE, "psi-drift", needs=("data_store",))
def psi_drift(pal):
    """PSI of four live features against their 2015-2019 training reference, 2020-2024."""
    d = _psi_data()
    p = d["psi"]
    fig, ax = figure(figsize=FIGSIZE)
    colors = pal["categorical"]
    for i, (col, _lab) in enumerate(FEATURES):
        ln, = ax.plot(p.index, p[col].clip(lower=1e-4), color=colors[i], linewidth=1.3, label=col)
        label_line_end(ax, ln, text=col)
    ax.set_yscale("log")
    for lvl, lab in ((PSI_WARN, f"psi_warn {PSI_WARN:.2f}"), (PSI_ALERT, f"psi_alert {PSI_ALERT:.2f}")):
        ax.hlines(lvl, p.index[0], p.index[-1], color=pal["text_muted"], linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(lab, (p.index[-1], lvl), xytext=(8, 0), textcoords="offset points", fontsize=8,
                    color=pal["text_secondary"], va="center", ha="left")
    ax.set_ylim(1e-3, 12)
    ax.set_xlim(p.index[0], p.index[-1] + pd.Timedelta(days=560))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01", tz="UTC") for y in range(2020, 2025)], [str(y) for y in range(2020, 2025)])
    ax.set_ylabel("PSI vs the training reference (log scale)")
    ax.set_xlabel(f"end of the trailing {WINDOW}-bar window (evaluated every {EVERY} bars)")
    ax.set_title("PSI over 500-bar windows: only the fast feature stays below the alert line")
    source_note(fig, "Data: " + source_label("H1", start="2014-01-01")
                + f"; scaler and reference fitted on {TRAIN[0][:4]}-{TRAIN[1][:4]}", pal)
    return fig


@register(PAGE, "psi-bins", needs=("data_store",))
def psi_bins(pal):
    """Reference vs live bin shares of one feature and each bin's PSI term (one high-PSI window)."""
    d = _psi_data()
    col = "volatility_yz_24"
    ref = d["ref"]
    end = pd.Timestamp("2024-02-23 13:00", tz="UTC")
    x = d["x"][col].loc[LIVE_FROM:]
    win = x.loc[:end].iloc[-WINDOW:]
    vals = win.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    edges = np.asarray(ref.edges[col])
    counts = np.bincount(np.searchsorted(edges, vals, side="right"), minlength=len(edges) + 1).astype(float)
    e = np.asarray(ref.expected[col])
    a = counts / counts.sum()
    eps = 1e-4
    ef = np.maximum(e, eps)
    af = np.maximum(a, eps)
    ef, af = ef / ef.sum(), af / af.sum()
    terms = (af - ef) * np.log(af / ef)
    total = psi(e, counts)

    blue, gray = pal["categorical"][0], pal["text_muted"]
    fig, (ax0, ax1) = panels(2, heights=(1.6, 1), figsize=FIGSIZE_TALL)
    k = np.arange(len(e))
    ax0.bar(k - 0.2, e, width=0.38, color=gray, linewidth=0, label="training reference (2015-2019)")
    ax0.bar(k + 0.2, a, width=0.38, color=blue, linewidth=0, label=f"live window ending {end:%Y-%m-%d}")
    ax0.annotate("training reference:\n10% per decile bin", (k[-1] - 0.2, e[-1]), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=pal["text_secondary"], va="bottom", ha="center")
    top = int(np.argmax(a))
    ax0.annotate(f"{a[top]:.0%} of the last {WINDOW} bars", (k[top] + 0.2, a[top]), xytext=(-6, 3),
                 textcoords="offset points", fontsize=8, color=pal["text"], ha="right", va="bottom")
    percent_axis(ax0)
    top_y = 0.05 * np.ceil(max(float(a.max()), float(e.max())) / 0.05 + 0.4)
    ax0.set_ylim(0, top_y)
    ax0.set_yticks(np.arange(0, top_y + 1e-9, 0.05))
    ax0.set_ylabel("share of observations")
    ax0.set_title(f"{col}: bin shares, training vs a live window (PSI = {total:.2f})")
    ax1.bar(k, terms, width=0.6, color=blue, linewidth=0)
    for i, v in enumerate(terms):
        ax1.annotate(f"{v:.2f}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.5, color=pal["text_secondary"])
    zero_line(ax1)
    labels = ["below\nq10"] + [f"q{10 * j}-\nq{10 * (j + 1)}" for j in range(1, len(e) - 1)] + ["above\nq90"]
    ax1.set_xticks(k, labels, fontsize=7.5)
    ax1.set_ylim(min(0.0, float(terms.min())) - 0.05, float(terms.max()) * 1.25)
    ax1.set_ylabel("(a - e) ln(a / e)")
    ax1.set_xlabel("reference bin (training deciles of the scaled feature)")
    source_note(fig, "Data: " + source_label("H1", start="2014-01-01") + "; window = 500 H1 bars to 2024-02-23 13:00",
                pal)
    return fig


# ---------------------------------------------------------------------------------------
# 4. the kill switch and its reset
# ---------------------------------------------------------------------------------------
@register(PAGE, "kill-switch-rebase")
def kill_switch_rebase(pal):
    """A drawdown halt, the flat book while halted, and what reset_halt re-bases (illustration)."""
    import logging

    rng = np.random.default_rng(7)
    days = pd.date_range("2026-03-02 21:00", periods=70, freq="B", tz="UTC")
    risk_log = logging.getLogger("aurum.risk.manager")
    level = risk_log.level
    risk_log.setLevel(logging.CRITICAL + 10)      # the halt and the reset log at CRITICAL/WARNING
    limits = RiskLimits(max_drawdown=0.10, max_daily_loss=None)
    rm = StandardRiskManager(limits)
    drift = np.r_[np.full(18, 0.006), np.full(14, -0.0085), np.zeros(38)]
    noise = rng.normal(0.0, 0.004, len(days))
    eq = 100_000.0
    rows = []
    reset_day = 46
    for i, t in enumerate(days):
        if i == reset_day:
            rm.reset_halt(confirm="RESET")
        if not rm.halted:
            ret = drift[i] + noise[i] if i < reset_day else 0.0012 + noise[i]
            eq *= 1.0 + ret
        rm.on_bar(t, eq)
        st = rm.state
        rows.append((t, eq, st.peak_equity, rm.halted))
    risk_log.setLevel(level)
    df = pd.DataFrame(rows, columns=["time", "equity", "peak", "halted"]).set_index("time")
    dd = 1.0 - df["equity"] / df["peak"]
    halted = df.index[df["halted"].to_numpy()]
    h0, h1 = halted[0], halted[-1]
    old_peak = float(df["peak"].loc[:h1].max())
    no_rebase = 1.0 - df["equity"] / df["equity"].loc[:h1].cummax().reindex(df.index).ffill().clip(lower=old_peak)

    blue, orange, red = pal["categorical"][0], pal["categorical"][1], pal["categorical"][7]
    fig, (ax0, ax1) = panels(2, heights=(1.4, 1), figsize=FIGSIZE_TALL)
    for ax in (ax0, ax1):
        shade_span(ax, h0, h1, None)
    mid_y = 0.5 * (float(df["equity"].loc[h0]) + old_peak)
    ax0.annotate("HALTED: approved lots = 0,\nthe book is flat until\nreset_halt", (h0, mid_y),
                 xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=pal["text_secondary"])
    le, = ax0.plot(df.index, df["equity"], color=blue, label="equity")
    lp, = ax0.plot(df.index, df["peak"], color=orange, label="peak_equity (risk state)")
    label_line_end(ax0, le, text="equity", fmt="{:,.0f}")
    label_line_end(ax0, lp, text="peak_equity", fmt="{:,.0f}")
    rt = df.index[reset_day]
    ax0.annotate("reset_halt(confirm='RESET'):\npeak and day start re-based\nto the last equity",
                 (rt, df["peak"].iloc[reset_day]), xytext=(rt + pd.Timedelta(days=6), mid_y + 0.25 * (old_peak - mid_y)),
                 textcoords="data", fontsize=8, color=pal["text"], va="center",
                 arrowprops={"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.8})
    from matplotlib.ticker import StrMethodFormatter

    ax0.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax0.set_ylabel("USD")
    ax0.set_title("Kill switch: a 10% drawdown limit trips, the book stays flat, a reset re-bases the peak")
    ld, = ax1.plot(df.index, dd, color=blue, label="drawdown seen by the risk manager")
    ln, = ax1.plot(df.index[reset_day:], no_rebase.iloc[reset_day:], color=pal["text_muted"],
                   linestyle=(0, (3, 2)), linewidth=1.3, label="without the re-base")
    ax1.axhline(limits.max_drawdown, color=red, linewidth=1.5)
    ax1.annotate(f"max_drawdown {limits.max_drawdown:.0%}", (df.index[0], limits.max_drawdown), xytext=(2, 3),
                 textcoords="offset points", fontsize=8, color=pal["text"], va="bottom")
    ax1.annotate("without the re-base the old peak\nwould halt the book again at once", (df.index[reset_day + 2], float(no_rebase.iloc[reset_day + 2])),
                 xytext=(0, 8), textcoords="offset points", fontsize=8, color=pal["text_secondary"], va="bottom")
    percent_axis(ax1)
    ax1.set_ylim(0, 0.16)
    ax1.set_yticks([0.0, 0.04, 0.08, 0.12, 0.16])
    import matplotlib.dates as mdates

    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.set_ylabel("drawdown from peak")
    ax1.set_xlabel("daily marks at 21:00 UTC (illustration; live defaults: max_drawdown 20%, max_daily_loss 3%)")
    source_note(fig, "Data: synthetic equity path (seeded), driven through aurum.risk.manager.StandardRiskManager",
                pal)
    return fig


# ---------------------------------------------------------------------------------------
# 5. MT5 server clocks
# ---------------------------------------------------------------------------------------
@register(PAGE, "server-tz-offsets")
def server_tz_offsets(pal):
    """UTC offset of a NY-close ("NY+7") server vs an EU-DST server and a fixed offset, 2024."""
    t = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    specs = (("NY+7", "NY+7 (New York close, US DST)"), ("Europe/Athens", "Europe/Athens (EU DST)"),
             ("UTC+02:00", "UTC+02:00 (fixed)"))
    off = {s: np.array([_tz_offset_now(s, x).total_seconds() / 3600 for x in t[::6]]) for s, _ in specs}
    tt = t[::6]
    colors = pal["categorical"]
    fig, ax = figure(figsize=FIGSIZE_SHORT)
    mism = off["NY+7"] != off["Europe/Athens"]
    starts = np.flatnonzero(np.diff(np.r_[0, mism.astype(int)]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[mism.astype(int), 0]) == -1)
    for a, b in zip(starts, ends, strict=True):
        shade_span(ax, tt[a], tt[b], f"{tt[a]:%d %b}-\n{tt[b]:%d %b}")
    for i, (s, lab) in enumerate(specs):
        y = off[s] + (0.0 if i == 0 else (0.04 if i == 1 else -0.04))
        ax.step(tt, y, where="post", color=colors[i], linewidth=1.8 if i == 0 else 1.4, label=lab)
    add_legend(ax, loc="center", frameon=False)
    ax.set_yticks([2, 3], ["UTC+2", "UTC+3"])
    ax.set_ylim(1.75, 3.35)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylabel("server clock offset")
    ax.set_xlabel("2024 (UTC); shaded: weeks where US and EU daylight saving disagree")
    ax.set_title("Why the adapter detects the offset: broker server clocks move with daylight saving")
    source_note(fig, "Data: IANA time-zone rules via aurum.live.mt5 (no market data)", pal)
    return fig

