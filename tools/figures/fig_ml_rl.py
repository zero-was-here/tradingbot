"""Figures for docs/ml-and-rl.md (page key ``ml-and-rl``).

Each ``@register``-ed ``draw(pal)`` becomes
docs/_assets/figures/ml-and-rl/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only ml-and-rl

Everything is drawn by aurum's own code (labels, uniqueness, splitters, the skill-gate
formula, the bet-size map, the RL environment and the rl_ppo look-ahead rule) on real
XAUUSD bars cut at 2024-12-31, or on clearly labelled synthetic inputs. No model is trained
here: the RL figures drive :class:`aurum.rl.env.GoldTradingEnv` with a scripted policy, and
the walk-forward figures read existing run outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_SHORT,
    FIGSIZE_TALL,
    bars,
    market_data,
    panels,
    register,
    run_label,
    run_table,
    source_label,
    source_note,
)

from aurum.viz.style import (
    figure,
    label_line_end,
    percent_axis,
    shade_span,
    style_axes,
    zero_line,
)

PAGE = "ml-and-rl"

#: ml_gbm / meta_label label defaults (``_MLStrategyBase.default_params``)
H = 24              # max_holding_bars
VOL_SPAN = 100      # vol_span
PT = SL = 1.0       # pt_mult, sl_mult


def _ml_labels(b: pd.DataFrame) -> pd.DataFrame:
    """ml_gbm's training labels on ``b``: symmetric 1-sigma-of-the-horizon barriers,
    24-bar vertical barrier, ``vertical_label="sign"``, label tail dropped."""
    from aurum.labels import drop_label_tail, ewm_vol, triple_barrier_labels

    vol = ewm_vol(b["close"], span=VOL_SPAN) * math.sqrt(H)      # barrier_vol="horizon"
    lab = triple_barrier_labels(b, pt_mult=PT, sl_mult=SL, max_holding_bars=H, vol=vol,
                                vertical_label="sign")
    return drop_label_tail(lab, len(b), H)


# =========================================================================================
# triple-barrier labels
# =========================================================================================
@register(PAGE, "triple-barrier-path", needs=("data_store",))
def triple_barrier_path(pal):
    """Three real H1 events: which of the three barriers is touched first."""
    b = bars("H1", start="2023-02-01", end="2023-04-30")
    lab = _ml_labels(b)
    win = lab.loc["2023-03-01":"2023-03-31"]
    picks = []
    for hit, lo_h, hi_h in (("pt", 8, 16), ("sl", 8, 16), ("vertical", 24, 24)):
        cand = win[(win["barrier_hit"] == hit) & win["holding_bars"].between(lo_h, hi_h)]
        if hit == "vertical":
            cand = cand[cand["ret"].abs() < 0.4 * cand["trgt"]]
        picks.append(cand.iloc[0])
    titles = {"pt": "Upper barrier first (+1)", "sl": "Lower barrier first (\N{MINUS SIGN}1)",
              "vertical": "Vertical barrier (sign of return)"}

    fig, axes = figure(1, 3, figsize=(8.0, 3.7))
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    hi_a, lo_a, c_a = (b[k].to_numpy() for k in ("high", "low", "close"))
    for ax, ev in zip(axes, picks, strict=True):
        t, t1 = int(ev["t_idx"]), int(ev["t1_idx"])
        x = np.arange(-4, H + 5)
        rows = t + x
        ax.vlines(x, lo_a[rows], hi_a[rows], color=pal["text_muted"], linewidth=1.0, zorder=1)
        ax.plot(x, c_a[rows], color=blue, zorder=2)
        entry = float(ev["entry_price"])
        up = entry * math.exp(PT * float(ev["trgt"]))
        dn = entry * math.exp(-SL * float(ev["trgt"]))
        for lvl, name in ((up, "upper"), (dn, "lower")):
            ax.hlines(lvl, 0, H, color=pal["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.2)
            ax.annotate(name, (0, lvl), xytext=(2, 3 if name == "upper" else -3), textcoords="offset points",
                        va="bottom" if name == "upper" else "top", ha="left", fontsize=8,
                        color=pal["text_secondary"])
        ax.axvline(H, color=pal["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate("vertical\n(t+24)", (H, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                    textcoords="offset points", va="top", ha="left", fontsize=8, color=pal["text_secondary"])
        ax.plot([0], [entry], marker="o", color=blue, markeredgecolor=pal["surface"], markersize=7, zorder=4)
        k = t1 - t
        ax.plot([k], [float(ev["exit_price"])], marker="D", color=orange, markeredgecolor=pal["surface"],
                markersize=8, zorder=5)
        lab_txt = {1.0: "+1", -1.0: "\N{MINUS SIGN}1", 0.0: "0"}[float(ev["label"])]
        above = ev["barrier_hit"] == "pt"
        ax.annotate(f"exit at bar {k}\nlabel {lab_txt}", (k, float(ev["exit_price"])),
                    xytext=(0, 13 if above else -13), textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=8, color=pal["text"])
        ax.set_title(titles[ev["barrier_hit"]], fontsize=10)
        ax.set_xlabel("bars after the event")
        ax.set_xlim(-4.5, H + 7.5)
        pad = 0.35 * (up - dn)
        ax.set_ylim(min(dn, lo_a[rows].min()) - pad, max(up, hi_a[rows].max()) + pad)
        stamp = pd.Timestamp(ev["t_event"]).strftime("%Y-%m-%d %H:%M")
        ax.annotate(f"event {stamp} UTC", (0.0, 0.0), xycoords="axes fraction", xytext=(2, 3),
                    textcoords="offset points", fontsize=7.5, color=pal["text_muted"])
    axes[0].set_ylabel("USD per ounce")
    fig.suptitle("Triple-barrier labels on real XAUUSD H1 bars (barriers = 1 sigma of a 24-bar move)",
                 color=pal["text"], fontsize=11)
    source_note(fig, "Data: " + source_label("H1", start="2023-02-01", end="2023-04-30", years=False), pal)
    return fig


@register(PAGE, "label-distribution", needs=("data_store",))
def label_distribution(pal):
    """Share of each barrier outcome per year, ml_gbm defaults, real H1 2012-2024."""
    b = bars("H1")
    lab = _ml_labels(b)
    groups = (("pt", "upper first (+1)"), ("sl", "lower first (\N{MINUS SIGN}1)"),
              ("vertical", "vertical (sign of return)"))
    share = (lab.groupby(lab.index.year)["barrier_hit"].value_counts(normalize=True).unstack()
             .reindex(columns=[g for g, _ in groups] + ["ambiguous"]).fillna(0.0))
    years = share.index.to_numpy()
    fig, ax = figure(figsize=FIGSIZE)
    bottom = np.zeros(len(years))
    for i, (g, name) in enumerate(groups):
        v = share[g].to_numpy()
        ax.bar(years, v, bottom=bottom, color=pal["categorical"][i], width=0.78, edgecolor=pal["surface"],
               linewidth=0.8, label=name)
        for x, y0, y in zip(years, bottom, v, strict=True):
            ax.text(x, y0 + y / 2, f"{100 * y:.0f}", ha="center", va="center", fontsize=7.5,
                    color=_ink_on_bar(pal, i))
        ax.annotate(name, (years[-1] + 0.45, bottom[-1] + v[-1] / 2), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=pal["text"])
        bottom = bottom + v
    tot = lab["barrier_hit"].value_counts(normalize=True)
    ax.set_xlim(years[0] - 0.6, years[-1] + 3.4)
    ax.set_ylim(0, 1.0)
    percent_axis(ax)
    ax.set_xticks(years, [str(y) for y in years], fontsize=8)
    ax.set_ylabel("share of events in the year")
    ax.set_xlabel("year of the event (UTC)")
    ax.set_title(f"Which barrier is touched first, per year ({len(lab):,} H1 events; overall "
                 f"{100 * tot.get('pt', 0):.0f}% upper, {100 * tot.get('sl', 0):.0f}% lower, "
                 f"{100 * tot.get('vertical', 0):.0f}% vertical)", fontsize=10)
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


def _ink_on_bar(pal, i: int) -> str:
    from _common import ink_on

    return ink_on(pal["categorical"][i])


# =========================================================================================
# overlap and uniqueness
# =========================================================================================
@register(PAGE, "label-overlap", needs=("data_store",))
def label_overlap(pal):
    """Label spans of 40 consecutive H1 events (top) and the concurrency c_t (bottom)."""
    from aurum.labels import average_uniqueness, num_concurrent_events

    b = bars("H1", start="2023-03-01", end="2023-03-31")
    lab = _ml_labels(b)
    n = len(b)
    conc = num_concurrent_events(n, lab)
    uniq = average_uniqueness(lab, n_bars=n)
    first = int(lab["t_idx"].iloc[0]) + 96                 # well inside the slice
    sel = lab[(lab["t_idx"] >= first) & (lab["t_idx"] < first + 40)]
    u_sel = uniq[(lab["t_idx"] >= first).to_numpy() & (lab["t_idx"] < first + 40).to_numpy()]
    x0, x1 = first - 2, first + 40 + H + 2

    fig, (ax0, ax1) = panels(2, heights=(3, 1.5), figsize=FIGSIZE_TALL)
    colors = {"pt": pal["categorical"][0], "sl": pal["categorical"][1], "vertical": pal["categorical"][2],
              "ambiguous": pal["text_muted"]}
    names = {"pt": "upper first", "sl": "lower first", "vertical": "vertical", "ambiguous": "ambiguous"}
    seen = set()
    for i, (_, ev) in enumerate(sel.iterrows()):
        hit = ev["barrier_hit"]
        ax0.plot([ev["t_idx"] + 0.5, ev["t1_idx"] + 0.5], [i, i], color=colors[hit], linewidth=2.6,
                 solid_capstyle="butt", label=names[hit] if hit not in seen else "_nolegend_")
        seen.add(hit)
    hi = int(np.argmin(u_sel))
    ev_lo = sel.iloc[hi]
    ax0.annotate(f"least unique label: u = {u_sel[hi]:.3f}", (ev_lo["t1_idx"] + 0.5, hi), xytext=(8, 0),
                 textcoords="offset points", va="center", fontsize=8, color=pal["text"])
    ax0.set_ylim(-1, len(sel))
    ax0.invert_yaxis()
    ax0.set_ylabel("event (one per bar)")
    ax0.set_title("Each label depends on the returns of bars (t, t1]: neighbouring labels overlap")
    from aurum.viz.style import add_legend

    add_legend(ax0, loc="lower left", ncol=3, fontsize=8)
    xs = np.arange(x0, x1)
    ax1.step(xs, conc[x0:x1], where="mid", color=pal["categorical"][0])
    ax1.set_ylabel("labels per bar")
    ax1.set_xlabel("bar position in the slice (H1 bars)")
    ax1.set_title(f"Concurrency c_t (mean {conc[H:n - H].mean():.1f}); average uniqueness of a label "
                  f"= mean of 1/c_t over its span", fontsize=10)
    ax1.set_xlim(x0, x1)
    source_note(fig, "Data: " + source_label("H1", start="2023-03-01", end="2023-03-31", years=False), pal)
    return fig


# =========================================================================================
# purged cross-validation
# =========================================================================================
@register(PAGE, "purged-cv", needs=("data_store",))
def purged_cv(pal):
    """purged_kfold with real triple-barrier label ends and a 24-bar embargo, plus the
    ml_gbm fit / validation-tail split on the same events."""
    from aurum.labels import label_end_positions
    from aurum.research.splits import purged_kfold
    from aurum.viz.style import add_legend

    b = bars("H1", start="2023-01-02", end="2023-02-28")
    lab = _ml_labels(b)
    n0 = len(b)
    le_all = label_end_positions(lab, n0)
    n = 500                                              # first 500 bars (all labelled)
    le = np.minimum(le_all[:n], n - 1)
    k, embargo = 5, 24
    splits = purged_kfold(n, k=k, purge=0, embargo=embargo, label_end=le)
    c_train, c_test = pal["categorical"][0], pal["categorical"][1]
    c_purge, c_emb = pal["categorical"][3], pal["categorical"][6]

    fig, ax = figure(figsize=(8.0, 4.2))
    style_axes(ax, grid=None)
    rows = []
    for j, (tr, te) in enumerate(splits):
        state = np.full(n, "", dtype=object)
        state[tr] = "train"
        state[te] = "test"
        a, z = int(te.min()), int(te.max())
        removed = np.setdiff1d(np.arange(n), np.r_[tr, te])
        state[removed[removed < a]] = "purged"
        state[removed[removed > z]] = "embargo"
        rows.append((f"fold {j + 1}", state))
    # ml_gbm's own validation tail on the same events (CalibratedGBM.fit, embargo_bars=0)
    t0 = np.arange(n)
    t1 = le
    n_val = int(round(0.2 * n))
    val_start = int(t0[n - n_val])
    state = np.full(n, "train", dtype=object)
    state[n - n_val:] = "test"
    state[(t0 < n - n_val) & (t1 >= val_start)] = "purged"
    rows.append(("ml_gbm fit / val tail", state))

    colors = {"train": c_train, "test": c_test, "purged": c_purge, "embargo": c_emb}
    labels = {"train": "train", "test": "test / validation", "purged": "purged (label reaches the test block)",
              "embargo": "embargoed (after the test block)"}
    seen = set()
    for r, (_, st) in enumerate(rows):
        y = len(rows) - 1 - r + (0 if r < k else -0.35)
        i = 0
        while i < n:
            j = i
            while j < n and st[j] == st[i]:
                j += 1
            kind = st[i]
            ax.broken_barh([(i, j - i)], (y - 0.36, 0.72), facecolors=colors[kind], linewidth=0,
                           label=labels[kind] if kind not in seen else "_nolegend_")
            seen.add(kind)
            i = j
    ax.set_yticks([len(rows) - 1 - r + (0 if r < k else -0.35) for r in range(len(rows))],
                  [name for name, _ in rows], fontsize=8.5)
    ax.set_xlim(0, n)
    ax.set_ylim(-1.0, len(rows) + 0.2)
    ax.set_xlabel("observation (one event per H1 bar, chronological)")
    ax.set_title(f"Purged k-fold (k={k}, embargo {embargo} bars) and ml_gbm's purged validation tail")
    add_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    source_note(fig, "Label ends: " + source_label("H1", start="2023-01-02", end="2023-02-28", years=False), pal)
    return fig


# =========================================================================================
# skill gate
# =========================================================================================
def _auc_needed(n_eff: np.ndarray, z: float = 2.0, pos_share: float = 0.5) -> np.ndarray:
    """Smallest validation AUC with (AUC - 0.5) / se >= z, se from Hanley & McNeil (1982) with
    the effective sample size (the formula of ``_MLStrategyBase._skill_gate``)."""
    n1, n0 = pos_share * n_eff, (1 - pos_share) * n_eff
    return 0.5 + z * np.sqrt((n_eff + 1.0) / (12.0 * n1 * n0))


def _fold_n_eff(b: pd.DataFrame, folds: pd.DataFrame) -> list[float]:
    """Effective size of ml_gbm's validation tail in each fold's training window (labels only)."""
    from aurum.labels import average_uniqueness

    out = []
    for _, f in folds.iterrows():
        w = b.loc[pd.Timestamp(f["train_start"]):pd.Timestamp(f["train_end"])]
        lab = _ml_labels(w)
        lab = lab.loc[lab["label"] != 0]
        n_val = int(round(0.2 * len(lab)))
        u = average_uniqueness(lab, n_bars=len(w))[-n_val:]
        out.append(float(np.sum(u)))
    return out


@register(PAGE, "skill-gate", needs=("data_store", "runs/protocol/default_selection"))
def skill_gate(pal):
    """Minimum validation AUC that passes z >= 2, against the effective validation size."""
    folds = run_table("protocol/default_selection", "folds.csv", cut=False)
    b = bars("H1")
    n_eff_f = np.array(_fold_n_eff(b, folds))
    grid = np.logspace(np.log10(20), np.log10(20000), 300)
    fig, ax = figure(figsize=FIGSIZE)
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    ax.axhspan(0.43, 0.57, color=pal["grid"], zorder=0, linewidth=0)
    ax.annotate("ml_gbm validation AUCs in the walk-forward: 0.43-0.57 (RESULTS.md)", (22, 0.43),
                xytext=(2, 4), textcoords="offset points", va="bottom", fontsize=8, color=pal["text_muted"])
    line, = ax.plot(grid, _auc_needed(grid), color=blue, label="z = 2 (default skill_gate_z)")
    ax.plot(grid, _auc_needed(grid, z=3.0), color=pal["text_muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate("z = 3", (grid[40], _auc_needed(grid[40:41], z=3.0)[0]), xytext=(4, 4), textcoords="offset points",
                fontsize=8, color=pal["text_muted"])
    need = _auc_needed(n_eff_f)
    ax.plot(n_eff_f, need, linestyle="none", marker="o", color=orange, markeredgecolor=pal["surface"],
            markersize=7, zorder=4)
    ax.annotate(f"18 walk-forward folds: n_eff {n_eff_f.min():.0f}-{n_eff_f.max():.0f},\n"
                f"AUC must reach {need.min():.3f}-{need.max():.3f}", (n_eff_f.max(), need.min()),
                xytext=(14, 18), textcoords="offset points", fontsize=8.5, color=pal["text"],
                arrowprops={"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.8})
    ex = 74.74
    ax.plot([ex], [_auc_needed(np.array([ex]))[0]], marker="o", linestyle="none", color=pal["text_secondary"],
            markeredgecolor=pal["surface"], markersize=7, zorder=4)
    ax.annotate(f"random-walk example below: n_eff 75\nneeds AUC {_auc_needed(np.array([ex]))[0]:.3f}",
                (ex, _auc_needed(np.array([ex]))[0]), xytext=(10, 6), textcoords="offset points", fontsize=8.5,
                color=pal["text"])
    label_line_end(ax, line, text="z = 2", fmt="{:.3f}")
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(20, 20000)
    ax.set_ylim(0.40, 0.90)
    ax.set_xlabel("effective validation size n_eff (sum of average uniqueness, log scale)")
    ax.set_ylabel("validation AUC needed to trade")
    ax.set_title("The skill gate: AUC must beat 0.5 by z standard errors of the effective sample")
    source_note(fig, "Fold windows: " + run_label("protocol/default_selection") + "; labels: "
                + source_label("H1"), pal)
    return fig


@register(PAGE, "ml-active-folds", needs=("runs/protocol/default_selection",))
def ml_active_folds(pal):
    """Share of out-of-sample bars with a non-zero forecast, per fold, for ml_gbm and meta_label."""
    oos = run_table("protocol/default_selection", "oos_forecasts.parquet")
    act = oos.groupby("fold")[["ml_gbm", "meta_label"]].agg(lambda s: float((s != 0).mean()))
    act.index = act.index.astype(int)
    act = act.sort_index()
    folds = act.index.to_numpy()
    fig, ax = figure(figsize=FIGSIZE_SHORT)
    w = 0.38
    ax.bar(folds - w / 2, act["ml_gbm"], width=w, color=pal["categorical"][0], label="ml_gbm")
    ax.bar(folds + w / 2, act["meta_label"], width=w, color=pal["categorical"][1], label="meta_label")
    for f, v in zip(folds, act["meta_label"], strict=True):
        if v == 0:
            ax.text(f, 0.02, "0", ha="center", va="bottom", fontsize=7, color=pal["text_muted"])
    from aurum.viz.style import add_legend

    add_legend(ax, loc="upper left", ncol=2, fontsize=8.5)
    percent_axis(ax)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(folds, [str(int(f)) for f in folds], fontsize=8)
    ax.set_xlabel("walk-forward fold key, as in folds.csv (each tests 6 months, 2015-12 to 2024-12)")
    ax.set_ylabel("OOS bars with a position")
    n_on = int((act["meta_label"] > 0).sum())
    ax.set_title(f"Gate outcome per fold: ml_gbm never traded; meta_label traded in {n_on} of {len(folds)} folds",
                 fontsize=10)
    source_note(fig, "Data: " + run_label("protocol/default_selection"), pal)
    return fig


# =========================================================================================
# meta-labelling
# =========================================================================================
@register(PAGE, "meta-label-oos", needs=("data_store", "runs/protocol/default_selection"))
def meta_label_oos(pal):
    """Real walk-forward OOS: the primary (tsmom) picks the side, meta_label sizes it."""
    oos = run_table("protocol/default_selection", "oos_forecasts.parquet")
    a, z = "2023-10-01", "2023-10-31"
    f = oos.loc[a:z]
    fold = int(pd.to_numeric(f["fold"]).mode().iloc[0])     # fold key as in folds.csv (0-based)
    px = bars("H1", start=a, end=z)["close"]
    fig, (ax0, ax1, ax2) = panels(3, heights=(2, 1.4, 1.4), figsize=FIGSIZE_TALL)
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    ax0.plot(px.index, px.values, color=blue)
    ax0.set_ylabel("USD per ounce")
    ax0.set_title(f"XAUUSD H1 close, October 2023 (walk-forward fold {fold}, folds 0-17, out of sample)")
    ax1.plot(f.index, f["tsmom"], color=blue)
    zero_line(ax1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_ylabel("forecast")
    ax1.set_title("1. Primary: tsmom forecast (its sign is the side of the bet)", fontsize=10)
    ax2.step(f.index, f["meta_label"], where="post", color=orange)
    zero_line(ax2)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("forecast")
    ax2.set_xlabel("date (UTC)")
    ax2.set_title("2. meta_label: side x bet size from P(success), in steps of 0.1 (0 = no bet)", fontsize=10)
    source_note(fig, "Data: " + run_label("protocol/default_selection") + "; prices: "
                + source_label("H1", start=a, end=z, years=False), pal)
    return fig


@register(PAGE, "bet-size")
def bet_size(pal):
    """AFML bet sizing from a success probability, then Carver scaling and 0.1 steps."""
    from aurum.strategies import get_strategy
    from aurum.strategies.ml import bet_size_from_probability

    meta = get_strategy("meta_label")                    # default params: p_threshold 0.5, step 0.1
    p = np.linspace(0.30, 0.90, 601)
    m = bet_size_from_probability(p)
    fig, ax = figure(figsize=FIGSIZE)
    blue, orange, aqua = pal["categorical"][0], pal["categorical"][1], pal["categorical"][2]
    l0, = ax.plot(p, m, color=blue, label="m = 2 Phi(z) - 1")
    l1, = ax.plot(p, meta._bet(p, 1.0), color=orange, drawstyle="steps-mid", label="bet, scalar 1")
    l2, = ax.plot(p, meta._bet(p, 3.0), color=aqua, drawstyle="steps-mid", label="bet, scalar 3")
    zero_line(ax)
    ax.axvline(0.5, color=pal["text_muted"], linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate("p_threshold 0.5:\nno bet below", (0.5, -0.55), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=8, color=pal["text_muted"])
    ax.annotate("raw size m(p), negative below p = 0.5", (0.40, float(bet_size_from_probability(0.40))),
                xytext=(6, -10), textcoords="offset points", va="top", fontsize=8.5, color=pal["text"])
    label_line_end(ax, l1, text="bet, Carver scalar 1")
    label_line_end(ax, l2, text="bet, Carver scalar 3")
    ax.set_xlim(0.30, 1.02)
    ax.set_ylim(-0.75, 1.08)
    ax.set_xlabel("calibrated probability that the primary's bet succeeds, p")
    ax.set_ylabel("bet size (fraction of a full forecast)")
    ax.set_title(r"meta_label bet size: $m = 2\Phi(z) - 1$, $z = (p - \frac{1}{2}) / \sqrt{p(1-p)}$, "
                 "then scaled, clipped, rounded", fontsize=10.5)
    source_note(fig, "Synthetic illustration: aurum.strategies.ml functions on a probability grid, not market data",
                pal)
    return fig


# =========================================================================================
# fold anatomy and fit_history_bars
# =========================================================================================
@register(PAGE, "ml-fold-anatomy", needs=("data_store", "runs/protocol/default_selection"))
def ml_fold_anatomy(pal):
    """One real walk-forward fold as ml_gbm sees it: warm-up, fit set, validation tail, gap, test."""
    from aurum.strategies import get_strategy

    folds = run_table("protocol/default_selection", "folds.csv", cut=False)
    fold_key = 8                                          # folds.csv key (0-based)
    f = folds.iloc[fold_key]
    idx = bars("H1").index
    tr_a = int(idx.searchsorted(pd.Timestamp(f["train_start"])))
    tr_z = int(idx.searchsorted(pd.Timestamp(f["train_end"])))
    te_a = int(idx.searchsorted(pd.Timestamp(f["test_start"])))
    te_z = int(idx.searchsorted(pd.Timestamp(f["test_end"])))
    warm = get_strategy("ml_gbm").fit_history_bars
    w_a = tr_a - warm
    val_a = tr_z - int(round(0.2 * (tr_z - tr_a)))       # ~ last 20% of the events
    segs = [
        ("feature warm-up\n(fit_history_bars)", w_a, tr_a, pal["text_muted"]),
        ("fit set (80% of events)", tr_a, val_a, pal["categorical"][0]),
        ("validation\ntail (20%)", val_a, tr_z, pal["categorical"][2]),
        ("purge + embargo\n(48 bars)", tr_z, te_a, pal["categorical"][3]),
        ("test (6 months)", te_a, te_z, pal["categorical"][1]),
    ]
    fig, ax = figure(figsize=(8.0, 2.9))
    style_axes(ax, grid=None)
    for i, (name, a, z, c) in enumerate(segs):
        ax.broken_barh([(idx[a], idx[min(z, len(idx) - 1)] - idx[a])], (0, 1), facecolors=c, linewidth=0)
        mid = idx[a] + (idx[min(z, len(idx) - 1)] - idx[a]) / 2
        if name.startswith("purge"):
            ax.annotate(name, (mid, 0.0), xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                        fontsize=8.5, color=pal["text"],
                        arrowprops={"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.8})
            continue
        yy, va = (1.12, "bottom") if i % 2 == 0 else (-0.12, "top")
        ax.annotate(name, (mid, yy), ha="center", va=va, fontsize=8.5, color=pal["text"])
    ax.set_ylim(-0.9, 1.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("date (UTC)")
    ax.set_title(f"Walk-forward fold {fold_key} for ml_gbm: {warm:,} warm-up bars, then train "
                 f"{pd.Timestamp(f['train_start']):%Y-%m-%d} to {pd.Timestamp(f['train_end']):%Y-%m-%d}, "
                 f"test from {pd.Timestamp(f['test_start']):%Y-%m-%d}", fontsize=10)
    source_note(fig, "Data: " + run_label("protocol/default_selection"), pal)
    return fig


# =========================================================================================
# reinforcement learning
# =========================================================================================
def _default_env_config(**changes):
    """``EnvConfig`` exactly as ``aurum rl train -c configs/default.yaml`` builds it."""
    import dataclasses

    from _common import REPO

    from aurum.core.config import load_config
    from aurum.rl.train import rl_config_from_aurum

    env = rl_config_from_aurum(load_config(str(REPO / "configs" / "default.yaml"))).env
    return dataclasses.replace(env, **changes) if changes else env


@register(PAGE, "rl-action-mapping", needs=("data_store",))
def rl_action_mapping(pal):
    """Discrete action -> forecast level -> lots via the env's VolTargetSizer, real H1 2019-2021."""
    from aurum.models.volatility import ewma_volatility
    from aurum.portfolio.sizing import VolTargetSizer

    env_cfg = _default_env_config()
    inst = env_cfg.build_instrument()
    sizer = VolTargetSizer(**{k: (tuple(map(tuple, v)) if k == "drawdown_derisk" and v else v)
                              for k, v in env_cfg.sizer.items()})
    b = bars("H1", end="2021-06-30")
    vol_all = ewma_volatility(b["close"], halflife_bars=env_cfg.vol_halflife)
    a, z = "2019-07-01", "2021-06-30"
    close, vol = b["close"].loc[a:z], vol_all.loc[a:z]
    eq = env_cfg.initial_equity
    fig, (ax0, ax1) = panels(2, heights=(1.3, 2.4), figsize=FIGSIZE_TALL)
    ax0.plot(vol.index, vol.values, color=pal["categorical"][0])
    percent_axis(ax0)
    ax0.set_ylabel("annualised vol")
    ax0.set_title(f"Causal EWMA volatility (half-life {env_cfg.vol_halflife:g} bars), the sizer's input")
    levels = list(env_cfg.action_levels)
    order = [4, 3, 2, 1, 0]                               # +1 .. -1
    for slot, ai in enumerate(order):
        f = levels[ai]
        lots = np.array([sizer.target_lots(f, v, eq, p, inst, current_lots=0.0, drawdown=0.0)
                         for v, p in zip(vol.to_numpy(), close.to_numpy(), strict=True)])
        line, = ax1.plot(vol.index, lots, color=pal["categorical"][slot], linewidth=1.6)
        txt = f"action {ai}: forecast {f:+.1f}" if f != 0 else f"action {ai}: forecast 0"
        label_line_end(ax1, line, text=txt.replace("-", "\N{MINUS SIGN}"), fmt="{:+.2f}")
    zero_line(ax1)
    ax1.set_ylabel("lots (1 lot = 100 oz)")
    ax1.set_xlabel("date (UTC)")
    ax1.set_title(f"Lots from flat at {eq:,.0f} USD equity: target vol {sizer.target_vol:.0%}, "
                  f"leverage cap {sizer.max_leverage:g}x, 0.01-lot steps", fontsize=10)
    ax1.set_xlim(close.index[0], close.index[-1] + pd.Timedelta(days=200))
    source_note(fig, "Data: " + source_label("H1", start=a, end=z, years=False)
                + "; sizer from configs/default.yaml via rl_config_from_aurum", pal)
    return fig


@register(PAGE, "rl-episode", needs=("data_store",))
def rl_episode(pal):
    """GoldTradingEnv on real H1 bars driven by a scripted (not learned) policy: action,
    lots, true equity and the shaped reward kept apart."""
    from aurum.features.pipeline import FeaturePipeline
    from aurum.rl.env import GoldTradingEnv

    md = market_data("H1", start="2019-01-01", end="2020-06-30")
    b = md.bars
    pipe = FeaturePipeline(groups=["returns", "volatility"])
    raw = pipe.compute(md)
    fit_end = int(b.index.searchsorted(pd.Timestamp("2020-01-01", tz="UTC")))
    pipe.fit(raw.iloc[pipe.max_lookback:fit_end])           # scaler on 2019 rows only
    x = pipe.transform(raw)
    cfg = _default_env_config(dd_penalty=0.5, turnover_penalty=0.05)
    start = int(b.index.searchsorted(pd.Timestamp("2020-02-03", tz="UTC")))
    end = int(b.index.searchsorted(pd.Timestamp("2020-04-30", tz="UTC")))
    env = GoldTradingEnv(b, x, cfg, events=md.events, start=start, end=end, random_start=False, rates=md.macro)
    # scripted policy: sign of the 120-bar log return, known at the close of t (no learning)
    mom = np.log(b["close"]).diff(120).to_numpy()
    obs, info = env.reset(options={"start": start})
    rec = []
    done = False
    while not done:
        t = env.index
        f = 1.0 if mom[t] > 0 else (-1.0 if mom[t] < 0 else 0.0)
        obs, reward, term, trunc, info = env.step(env.forecast_to_action(f))
        rc = info["reward_components"]
        rec.append({"time": info["time"], "forecast": info["forecast"], "lots": info["approved_lots"],
                    "equity": info["equity"], "reward": reward, "growth": rc["growth"],
                    "drawdown": rc["drawdown"], "turnover": rc["turnover"],
                    "costs": sum(info["costs"].values())})
        done = term or trunc
    r = pd.DataFrame(rec).set_index("time")
    fig, axes = panels(4, heights=(1.2, 1.2, 1.4, 1.6), figsize=(8.0, 7.4))
    blue, orange, aqua = pal["categorical"][0], pal["categorical"][1], pal["categorical"][2]
    axes[0].step(r.index, r["forecast"], where="post", color=blue)
    axes[0].set_ylim(-1.25, 1.25)
    axes[0].set_ylabel("action level")
    axes[0].set_title("Scripted policy: +1 / \N{MINUS SIGN}1 by the sign of the 120-bar return (not a trained agent)",
                      fontsize=10)
    axes[1].step(r.index, r["lots"], where="post", color=blue)
    zero_line(axes[1])
    axes[1].set_ylabel("lots")
    axes[1].set_title("Approved lots after VolTargetSizer and the research risk limits", fontsize=10)
    axes[2].plot(r.index, r["equity"], color=blue)
    axes[2].set_ylabel("USD")
    axes[2].set_title(f"True equity (net of {r['costs'].sum():,.0f} USD spread, slippage, commission "
                      "and financing)", fontsize=10)
    cum = r[["growth", "drawdown", "turnover"]].cumsum()
    lg, = axes[3].plot(cum.index, cum["growth"], color=blue)
    ld, = axes[3].plot(cum.index, cum["drawdown"], color=orange)
    lt, = axes[3].plot(cum.index, cum["turnover"], color=aqua)
    zero_line(axes[3])
    label_line_end(axes[3], lg, text="growth", fmt="{:+.1f}")
    label_line_end(axes[3], ld, text="drawdown penalty", fmt="{:+.1f}")
    label_line_end(axes[3], lt, text="turnover penalty", fmt="{:+.1f}")
    axes[3].set_ylabel("reward units")
    axes[3].set_xlabel("date (UTC)")
    axes[3].set_title("Cumulative reward terms (reward_scale 100, dd_penalty 0.5, turnover_penalty 0.05)",
                      fontsize=10)
    from matplotlib.dates import DateFormatter
    from matplotlib.ticker import FuncFormatter

    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for ax in axes:
        ax.set_xlim(r.index[0], r.index[-1] + pd.Timedelta(days=16))
        ax.xaxis.set_major_formatter(DateFormatter("%d %b %Y"))
    source_note(fig, "Data: " + source_label("H1", start="2020-02-03", end="2020-04-30", years=False)
                + "; FRED fedfunds for financing", pal)
    return fig


@register(PAGE, "ppo-clip")
def ppo_clip(pal):
    """PPO's clipped surrogate objective per sample, for a positive and a negative advantage."""
    eps = 0.2                                           # RLTrainConfig.clip_range
    r = np.linspace(0.0, 2.0, 401)
    fig, axes = figure(1, 2, figsize=(8.0, 3.8), sharey=True)
    blue = pal["categorical"][0]
    for ax, adv, title in ((axes[0], 1.0, "advantage A > 0 (action better than expected)"),
                           (axes[1], -1.0, "advantage A < 0 (action worse than expected)")):
        unclipped = r * adv
        clipped = np.clip(r, 1 - eps, 1 + eps) * adv
        obj = np.minimum(unclipped, clipped)
        ax.plot(r, unclipped, color=pal["text_muted"], linewidth=1.2, linestyle=(0, (4, 3)))
        ax.plot(r, obj, color=blue)
        shade_span(ax, 1 - eps, 1 + eps)
        ax.axvline(1.0, color=pal["axis"], linewidth=0.8)
        zero_line(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"probability ratio $\rho = \pi_{new}(a|s)\,/\,\pi_{old}(a|s)$")
        tx = 1.8 if adv > 0 else 0.45
        ax.annotate(r"$\rho A$ (unclipped)", (tx, tx * adv), xytext=(-4, 6) if adv > 0 else (6, 4),
                    textcoords="offset points", ha="right" if adv > 0 else "left", va="bottom", fontsize=8,
                    color=pal["text_muted"])
        ax.annotate(r"min($\rho A$, clip($\rho$) A)", (1.75 if adv > 0 else 0.35,
                                            (1 + eps) * adv if adv > 0 else (1 - eps) * adv),
                    xytext=(0, -8), textcoords="offset points", ha="center", va="top", fontsize=8.5,
                    color=pal["text"])
        ax.annotate(f"[1\N{MINUS SIGN}{eps:g}, 1+{eps:g}]", (1.0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=8,
                    color=pal["text_muted"])
    axes[0].set_ylabel("per-sample objective")
    fig.suptitle(f"PPO clipped surrogate (clip_range = {eps:g}): no gain from moving $\\rho$ outside the band",
                 color=pal["text"], fontsize=11)
    source_note(fig, "Synthetic illustration of the PPO objective (Schulman et al. 2017), not market data", pal)
    return fig


@register(PAGE, "rl-artifact-guard", needs=("data_store", "runs/protocol/default_selection"))
def rl_artifact_guard(pal):
    """Which walk-forward folds may evaluate a pre-trained rl_ppo artifact (retrain=False)."""
    from _common import REPO

    from aurum.core.config import load_config

    rl = load_config(str(REPO / "configs" / "default.yaml")).rl
    idx = bars("H1").index
    tr0 = idx[idx.searchsorted(pd.Timestamp(rl.train_start, tz="UTC"))]
    tr1 = idx[idx.searchsorted(pd.Timestamp(rl.train_end, tz="UTC") + pd.Timedelta(days=1)) - 1]
    va0 = idx[idx.searchsorted(tr1, side="right")]
    va1 = idx[idx.searchsorted(pd.Timestamp(rl.val_end, tz="UTC") + pd.Timedelta(days=1)) - 1]
    folds = run_table("protocol/default_selection", "folds.csv", cut=False)
    blue, orange, aqua, yellow = (pal["categorical"][i] for i in (0, 1, 2, 3))
    fig, ax = figure(figsize=(8.0, 5.6))
    style_axes(ax, grid=None)
    n = len(folds)
    y_art = n + 1.0
    ax.broken_barh([(tr0, tr1 - tr0)], (y_art - 0.4, 0.8), facecolors=aqua, linewidth=0)
    ax.broken_barh([(va0, va1 - va0)], (y_art - 0.4, 0.8), facecolors=yellow, linewidth=0)
    ax.annotate("artifact: train", (tr0 + (tr1 - tr0) / 2, y_art), ha="center", va="center", fontsize=8.5,
                color=pal["text"])
    ax.annotate("validation", (va0 + (va1 - va0) / 2, y_art), ha="center", va="center", fontsize=8.5,
                color=pal["text"])
    ax.axvline(va1, color=pal["text_secondary"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(f"artifact's last bar\n{va1:%Y-%m-%d %H:%M}", (va1, y_art + 0.55), xytext=(4, 0),
                textcoords="offset points", fontsize=8, va="bottom", color=pal["text_secondary"])
    n_ok = 0
    for i, f in folds.iterrows():
        y = n - 1 - i
        ts, te = pd.Timestamp(f["train_start"]), pd.Timestamp(f["train_end"])
        s0, s1 = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        ok = te >= va1
        n_ok += ok
        c_tr = blue if ok else pal["axis"]
        c_te = orange if ok else pal["axis"]
        ax.broken_barh([(ts, te - ts)], (y - 0.32, 0.64), facecolors=c_tr, linewidth=0)
        ax.broken_barh([(s0, s1 - s0)], (y - 0.32, 0.64), facecolors=c_te, linewidth=0)
        ax.annotate("evaluated" if ok else "fit raises ValueError", (s1, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=pal["text"] if ok else pal["text_muted"], zorder=4,
                    bbox={"facecolor": pal["surface"], "edgecolor": "none", "pad": 0.6})
    ax.set_yticks([n - 1 - i for i in range(n)] + [y_art], [f"fold {i}" for i in range(n)] + ["artifact"],
                  fontsize=7.5)
    ax.set_ylim(-1, y_art + 1.6)
    ax.set_xlim(tr0 - pd.Timedelta(days=30), pd.Timestamp("2026-04-01", tz="UTC"))
    ax.set_xlabel("date (UTC)")
    ax.set_title(f"rl_ppo with retrain=False: only {n_ok} of {n} folds train past the artifact's last bar",
                 fontsize=10.5)
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=lbl, linewidth=0)
               for c, lbl in ((blue, "fold training window (fit cutoff = its last bar)"),
                              (orange, "fold test window"), (pal["axis"], "refused fold"))]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=8, frameon=False,
              labelcolor=pal["text"])
    source_note(fig, "Folds: " + run_label("protocol/default_selection")
                + "; artifact dates: rl section of configs/default.yaml", pal)
    return fig
