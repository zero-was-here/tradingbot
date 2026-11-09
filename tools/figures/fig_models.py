"""Figures for docs/models.md: volatility estimators (EWMA, GARCH(1,1), HAR-RV), volatility
clustering, nominal annualisation and the Gaussian HMM regime model.

Each ``@register``-ed ``draw(pal)`` becomes docs/_assets/figures/models/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only models

Every model below is aurum's own code (``aurum.models.volatility``, ``aurum.models.regime``).
Fitted parameters only ever come from data BEFORE the period they are evaluated on (GARCH is
fitted on 2012-2017 and drawn on 2018-2024; HAR refits walk-forward; the HMM is refitted each
year on the previous three years). The heavy computations are cached per process so the light
and dark renders share them.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from _common import (
    FIGSIZE,
    FIGSIZE_SHORT,
    bars,
    panels,
    register,
    source_label,
    source_note,
)

from aurum.core.timeframes import index_bar_minutes, infer_bars_per_year, nominal_bars_per_year
from aurum.models.regime import GaussianHMM
from aurum.models.volatility import (
    Garch11,
    blend_vol,
    daily_realised_variance,
    ewma_volatility,
    har_rv_forecast,
)
from aurum.viz.style import (
    add_legend,
    figure,
    percent_axis,
    shade_span,
    zero_line,
)

PAGE = "models"

#: GARCH training window (the figures evaluate the fitted model from 2018 on).
GARCH_TRAIN_END = "2017-12-31"
#: HMM: refit every calendar year on the previous ``HMM_TRAIN_YEARS`` years of daily returns.
HMM_TRAIN_YEARS = 3
HMM_FIRST_YEAR = 2015


# ---------------------------------------------------------------------------------------
# cached computations (called inside draw only)
# ---------------------------------------------------------------------------------------
@functools.cache
def _daily_forecasts() -> pd.DataFrame:
    """One row per New-York-close session (22:00 UTC anchor), keyed by the session label.

    * ``ewma``  - ewma_volatility on H1 closes (the pipeline default), last value available at
      the session's last H1 bar;
    * ``garch`` - Garch11 fitted on daily NY-close log returns up to GARCH_TRAIN_END;
    * ``har``   - har_rv_forecast on daily realised variance from H1 bars (walk-forward refits);
    * ``rv_next`` / ``rv5`` - ex-post realised vol of the next session / next five sessions.
    All annualised fractions.
    """
    h1 = bars("H1")
    d1 = bars("D1_nyclose")
    ew = ewma_volatility(h1["close"])
    ew_av = pd.Series(ew.to_numpy(), index=pd.DatetimeIndex(h1["available_at"]))
    rd = np.log(d1["close"]).diff()
    garch = Garch11().fit(rd.loc[:GARCH_TRAIN_END])
    gv = garch.forecast(rd)
    # D1_nyclose bar opening 22:00 UTC on d-1 is the session labelled d.
    g_by_label = pd.Series(gv.to_numpy(), index=(gv.index + pd.Timedelta(days=1)).normalize())
    rv = daily_realised_variance(h1, anchor_hour_utc=22)
    har = har_rv_forecast(rv["rv"])
    out = pd.DataFrame({
        "ewma": ew_av.reindex(pd.DatetimeIndex(rv["available_at"]), method="ffill").to_numpy(),
        "garch": g_by_label.reindex(rv.index).to_numpy(),
        "har": har.to_numpy(),
        "rv_next": np.sqrt(252.0 * rv["rv"].shift(-1)).to_numpy(),
        "rv5": np.sqrt(252.0 * rv["rv"].rolling(5).mean().shift(-5)).to_numpy(),
    }, index=rv.index)
    return out


@functools.cache
def _garch_fits() -> dict[str, tuple[Garch11, pd.Series]]:
    """GARCH(1,1) on D1 (NY close) and on H1 log returns, both fitted on data to 2017."""
    out = {}
    for tf in ("D1_nyclose", "H1"):
        b = bars(tf)
        r = np.log(b["close"]).diff()
        g = Garch11().fit(r.loc[:GARCH_TRAIN_END])
        out[tf] = (g, g.forecast(r))
    return out


@functools.cache
def _daily_pct_returns() -> pd.Series:
    """Daily NY-close log returns in percent (the input regime_switch uses on D1 bars)."""
    d1 = bars("D1_nyclose")
    return (100.0 * np.log(d1["close"]).diff()).dropna()


@functools.cache
def _hmm_walk_forward() -> tuple[pd.DataFrame, list[dict]]:
    """Yearly refits: year Y is filtered by a 2-state GaussianHMM fitted on Y-3 .. Y-1.

    The filter runs over every return up to the end of Y (causal: row t uses rows <= t);
    only year Y's rows are kept. Returns (per-day frame, per-fit summaries).
    """
    r = _daily_pct_returns()
    rows, fits = [], []
    for year in range(HMM_FIRST_YEAR, 2025):
        train = r.loc[str(year - HMM_TRAIN_YEARS):str(year - 1)]
        hmm = GaussianHMM(n_states=2, seed=0).fit(train)
        p = hmm.filter(r.loc[:str(year)])["p_state_1"].loc[str(year)]
        dur = hmm.expected_durations_
        fits.append({"year": year, "train": f"{year - HMM_TRAIN_YEARS}-{year - 1}",
                     "sd": np.sqrt(np.ravel(hmm.vars_)), "dur": dur,
                     "stationary": hmm.stationary_distribution_})
        rows.append(pd.DataFrame({"p_turb": p, "dur_turb": dur[1], "dur_calm": dur[0]}))
    return pd.concat(rows), fits


def _hmm_fit(start: str, end: str) -> GaussianHMM:
    return GaussianHMM(n_states=2, seed=0).fit(_daily_pct_returns().loc[start:end])


def _blend(c1: str, c2: str, t: float) -> tuple[float, float, float]:
    """Linear mix of two colours (``t`` = share of ``c2``): a band one step darker than the grid."""
    from matplotlib.colors import to_rgb

    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple((1.0 - t) * a + t * b)


def _pct_formatter(decimals: int = 0):
    """Percent tick labels with a true minus sign and no "-0%"."""
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos):
        v = 0.0 if abs(v) < 1e-12 else v
        return f"{v * 100:.{decimals}f}%".replace("-", "\N{MINUS SIGN}")

    return FuncFormatter(fmt)


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    denom = float(x @ x)
    return np.array([float(x[k:] @ x[:-k]) / denom for k in range(1, nlags + 1)])


# ---------------------------------------------------------------------------------------
# annualisation
# ---------------------------------------------------------------------------------------
@register(PAGE, "bars-per-year", needs=("data_store",))
def bars_per_year(pal):
    """H1 bars per calendar year against the nominal 252 x 23 constant and the realised density."""
    h1 = bars("H1")
    counts = h1.groupby(h1.index.year).size()
    nominal = nominal_bars_per_year(index_bar_minutes(h1.index))
    realised = infer_bars_per_year(h1.index)
    blue, orange = pal["categorical"][0], pal["categorical"][1]

    fig, ax = figure(figsize=FIGSIZE_SHORT)
    ax.plot(counts.index, counts.values, color=blue, marker="o", markersize=5,
            markeredgecolor=pal["surface"], markeredgewidth=1.0)
    ax.axhline(realised, color=orange, linewidth=1.5, linestyle="--")
    ax.axhline(nominal, color=pal["text_secondary"], linewidth=1.5)
    ax.annotate(f"realised density 2012-2024 (infer_bars_per_year): {realised:,.0f}",
                (counts.index[-1], realised), xytext=(0, 5), textcoords="offset points",
                ha="right", va="bottom", color=pal["text"], fontsize=8.5)
    ax.annotate(f"nominal 252 x 23 h (nominal_bars_per_year): {nominal:,.0f}",
                (counts.index[0], nominal), xytext=(0, 5), textcoords="offset points",
                ha="left", va="bottom", color=pal["text"], fontsize=8.5)
    for x, y in counts.items():
        ax.annotate(f"{y:,}", (x, y), xytext=(0, -8), textcoords="offset points", ha="center",
                    va="top", color=pal["text_secondary"], fontsize=7)
    ax.set_xticks(counts.index, [str(v) for v in counts.index])
    ax.set_ylim(nominal - 120, counts.max() + 60)
    ax.set_xlim(counts.index[0] - 0.5, counts.index[-1] + 0.5)
    ax.set_title("H1 bars per calendar year vs the annualisation constant")
    ax.set_ylabel("H1 bars per year")
    ax.set_xlabel("calendar year")
    source_note(fig, "Data: " + source_label("H1"), pal)
    return fig


@register(PAGE, "ewma-kernel")
def ewma_kernel(pal):
    """Analytic: share of the EWMA variance carried by the last k squared returns, three half-lives."""
    k = np.arange(0, 241)
    fig, ax = figure(figsize=FIGSIZE)
    ax.axhline(0.5, color=pal["axis"], linewidth=0.8, linestyle="--")
    for i, (h, lab_k) in enumerate(((12, 22), (48, 100), (120, 150))):
        alpha = 1.0 - 0.5 ** (1.0 / h)
        cum = 1.0 - (1.0 - alpha) ** k              # weight on r^2_t .. r^2_{t-k+1}
        colr = pal["categorical"][i]
        ax.plot(k, cum, color=colr)
        ax.plot([h], [0.5], marker="o", markersize=6, color=colr, markeredgecolor=pal["surface"],
                linestyle="none")
        name = f"H = {h} bars" + (" (default)" if h == 48 else "") + f", alpha = {alpha:.4f}"
        ax.annotate(name, (lab_k, cum[lab_k]), xytext=(8, -4), textcoords="offset points",
                    ha="left", va="top", color=pal["text"], fontsize=9)
    ax.annotate("half the weight sits on the last H returns", (2, 0.5), xytext=(0, 4),
                textcoords="offset points", ha="left", va="bottom", color=pal["text_muted"], fontsize=8.5)
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 1.02)
    percent_axis(ax)
    ax.set_title("Share of the EWMA variance that comes from the last k squared returns")
    ax.set_xlabel("k (most recent bars)")
    ax.set_ylabel("cumulative weight, 1 - (1 - alpha)^k")
    source_note(fig, "Analytic illustration (no market data)", pal)
    return fig


# ---------------------------------------------------------------------------------------
# estimators on real data
# ---------------------------------------------------------------------------------------
@register(PAGE, "vol-estimators", needs=("data_store",))
def vol_estimators(pal):
    """EWMA (H1), GARCH(1,1) (D1, fitted to 2017) and HAR-RV (walk-forward) vs realised vol."""
    df = _daily_forecasts()
    full = df.loc["2018-01-01":"2024-12-31"]
    zoom = df.loc["2020-02-03":"2020-05-29"]
    c = pal["categorical"]
    spec = (("ewma", "EWMA, H1 (pipeline default)", c[0]), ("garch", "GARCH(1,1), D1", c[1]),
            ("har", "HAR-RV, daily", c[2]))

    fig, (ax0, ax1) = panels(2, heights=(3, 2.4), figsize=(8.0, 7.0), sharex=False)
    ax0.plot(full.index, full["rv5"], color=pal["text_muted"], linewidth=1.0,
             label="realised, next 5 sessions (ex post)")
    for col, lab, colr in spec:
        ax0.plot(full.index, full[col], color=colr, linewidth=1.2, label=lab)
    shade_span(ax0, zoom.index[0], zoom.index[-1], "zoom")
    percent_axis(ax0)
    ax0.set_ylim(0, 0.7)
    ax0.set_title("Three volatility forecasts on gold, 2018-2024 (one value per NY-close session)")
    ax0.set_ylabel("annualised volatility")
    add_legend(ax0, loc="upper right", ncol=2, fontsize=8.5)

    ax1.plot(zoom.index, zoom["rv_next"], color=pal["text_muted"], linewidth=1.0, marker="o",
             markersize=3, label="realised, next session (ex post)")
    offsets = {"ewma": (6, 4), "garch": (6, 10), "har": (-10, 12)}
    for col, lab, colr in spec:
        ax1.plot(zoom.index, zoom[col], color=colr, label=lab)
        t = zoom[col].idxmax()
        ax1.plot([t], [zoom[col].max()], marker="o", markersize=5, color=colr,
                 markeredgecolor=pal["surface"], linestyle="none")
        short = {"ewma": "EWMA", "garch": "GARCH", "har": "HAR-RV"}[col]
        ax1.annotate(f"{short} peak {zoom[col].max():.0%} ({t:%d %b})", (t, zoom[col].max()),
                     xytext=offsets[col], textcoords="offset points", ha="left" if offsets[col][0] > 0 else "right",
                     va="bottom", color=pal["text"], fontsize=8.5)
    percent_axis(ax1)
    ax1.set_xlim(zoom.index[0], zoom.index[-1])
    ax1.set_ylim(0, 0.85)
    ax1.set_title("February to May 2020: how fast each forecast reacts")
    ax1.set_ylabel("annualised volatility")
    ax1.set_xlabel("session date (UTC)")
    import matplotlib.dates as mdates

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz="UTC"))
    source_note(fig, "Data: XAUUSD H1 2018-2024 (EWMA, HAR, realised) and D1 NY close 2012-2024 (GARCH), "
                "Dukascopy bid/ask mid", pal)
    return fig


@register(PAGE, "vol-clustering", needs=("data_store",))
def vol_clustering(pal):
    """Daily gold returns with the causal +/-2 sigma band from the previous session's EWMA."""
    r = _daily_pct_returns() / 100.0
    df = _daily_forecasts()
    # sigma for session d = EWMA (annualised) known at the end of session d-1, per day.
    sig = (df["ewma"].shift(1) / np.sqrt(252.0))
    d1 = bars("D1_nyclose")
    label = (d1.index + pd.Timedelta(days=1)).normalize()
    r_by_label = pd.Series(r.reindex(d1.index).to_numpy(), index=label).dropna()
    both = pd.DataFrame({"r": r_by_label, "sig": sig.reindex(r_by_label.index)}).dropna()
    outside = float((both["r"].abs() > 2 * both["sig"]).mean())
    blue, orange = pal["categorical"][0], pal["categorical"][1]

    out_mask = both["r"].abs() > 2 * both["sig"]
    fig, ax = figure(figsize=FIGSIZE)
    ax.fill_between(both.index, -2 * both["sig"], 2 * both["sig"], color=_blend(pal["grid"], pal["text_secondary"], 0.22),
                    linewidth=0, step="mid", zorder=0)
    ax.vlines(both.index[~out_mask], 0, both["r"][~out_mask], color=blue, linewidth=0.6)
    ax.vlines(both.index[out_mask], 0, both["r"][out_mask], color=orange, linewidth=0.9)
    zero_line(ax)
    lo = both["r"].idxmin()
    ax.annotate(f"{lo:%Y-%m-%d}: {both['r'].min():.1%}", (lo, both["r"].min()), xytext=(8, 0),
                textcoords="offset points", va="center", color=pal["text"], fontsize=8.5)
    ax.annotate("grey band: +/- 2 sigma, sigma = EWMA vol known at the previous close / sqrt(252)",
                (0.01, 0.98), xycoords="axes fraction", va="top", color=pal["text_secondary"], fontsize=8.5)
    ax.annotate("orange: a day outside the band", (0.01, 0.92), xycoords="axes fraction", va="top",
                color=pal["text_secondary"], fontsize=8.5)
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(_pct_formatter())
    ax.set_ylim(-0.105, 0.09)
    ax.set_title(f"Volatility clusters: daily gold returns and a causal 2-sigma band ({outside:.1%} outside)")
    ax.set_ylabel("daily log return")
    ax.set_xlabel("session date (UTC)")
    source_note(fig, "Data: " + source_label("D1_nyclose") + "; band from ewma_volatility on H1 bars", pal)
    return fig


@register(PAGE, "acf-returns", needs=("data_store",))
def acf_returns(pal):
    """Autocorrelation of daily returns, absolute returns and squared returns (lags 1-60)."""
    r = _daily_pct_returns().to_numpy()
    n_lags = 60
    lags = np.arange(1, n_lags + 1)
    band = 1.96 / np.sqrt(len(r))
    series = (("r", r, "returns r"), ("abs", np.abs(r), "absolute returns |r|"),
              ("sq", r ** 2, "squared returns r^2"))
    fig, ax = figure(figsize=FIGSIZE)
    ax.axhspan(-band, band, color=pal["grid"], zorder=0, linewidth=0)
    ax.annotate("95% band for white noise, +/- 1.96 / sqrt(n)", (n_lags, -band), xytext=(0, -3),
                textcoords="offset points", ha="right", va="top", color=pal["text_muted"], fontsize=8)
    for i, (_, x, lab) in enumerate(series):
        a = _acf(x, n_lags)
        ax.plot(lags, a, color=pal["categorical"][i], marker="o", markersize=3.5, markeredgewidth=0,
                label=f"{lab} (mean of lags 1-20: {a[:20].mean():+.3f})")
    zero_line(ax)
    ax.set_xlim(0, n_lags + 1)
    ax.set_title("Returns are nearly uncorrelated, their size is not")
    ax.set_xlabel("lag (sessions)")
    ax.set_ylabel("autocorrelation")
    add_legend(ax, loc="upper right", fontsize=8.5)
    source_note(fig, "Data: " + source_label("D1_nyclose"), pal)
    return fig


@register(PAGE, "garch-term-structure", needs=("data_store",))
def garch_term_structure(pal):
    """GARCH variance forecasts revert to the long-run level at rate alpha+beta (D1 vs H1 fits)."""
    fits = _garch_fits()
    fig, axes = figure(1, 2, figsize=FIGSIZE)
    for ax, tf, k_max, unit in ((axes[0], "D1_nyclose", 250, "sessions"), (axes[1], "H1", 48, "H1 bars")):
        g, fc = fits[tf]
        p = g.params_
        bpy = g.bars_per_year_
        vl = p.unconditional_variance
        oos = fc.loc["2018-01-01":]
        starts = (("calm", oos.idxmin()), ("stressed", oos.idxmax()))
        k = np.arange(1, k_max + 1)
        for i, (name, t) in enumerate(starts):
            h1 = (fc.loc[t] ** 2) / bpy                       # per-bar variance for t+1
            hk = vl + p.persistence ** (k - 1) * (h1 - vl)    # E[sigma^2_{t+k} | F_t]
            when = (f"session {(t + pd.Timedelta(days=1)):%Y-%m-%d}" if tf.startswith("D1")
                    else f"bar {t:%Y-%m-%d %H:%M}")
            ax.plot(k, np.sqrt(hk * bpy), color=pal["categorical"][i])
            ax.annotate(f"{name}, {when}", (k[0], np.sqrt(hk[0] * bpy)), xytext=(4, 6 if i else -10),
                        textcoords="offset points", color=pal["text"], fontsize=8)
        lr = np.sqrt(vl * bpy)
        ax.axhline(lr, color=pal["text_secondary"], linewidth=1.0, linestyle="--")
        ax.annotate(f"long-run {lr:.1%}", (k_max, lr), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", color=pal["text_secondary"], fontsize=8)
        hl = p.half_life
        ax.axvline(hl, color=pal["axis"], linewidth=0.8)
        ax.annotate(f"half-life {hl:.1f} {unit}", (hl, 0.55), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", va="center", color=pal["text_muted"],
                    fontsize=8)
        from matplotlib.ticker import MultipleLocator

        ax.yaxis.set_major_locator(MultipleLocator(0.05 if tf.startswith("D1") else 0.25))
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_xlim(0, k_max)
        tfname = "D1 (NY close)" if tf.startswith("D1") else "H1"
        ax.set_title(f"{tfname}: alpha+beta = {p.persistence:.3f}", fontsize=10)
        ax.set_xlabel(f"forecast horizon k ({unit})")
    axes[0].set_ylabel("annualised volatility forecast")
    fig.suptitle("GARCH(1,1) forecasts decay to the long-run level (fits on 2012-2017)")
    source_note(fig, "Data: XAUUSD D1 (NY close) and H1, Dukascopy bid/ask mid, fits 2012-2017; "
                "starts = lowest and highest 2018-2024 forecast", pal)
    return fig


# ---------------------------------------------------------------------------------------
# hidden Markov model
# ---------------------------------------------------------------------------------------
@register(PAGE, "hmm-filtered", needs=("data_store",))
def hmm_filtered(pal):
    """Filtered P(turbulent) from yearly 3-year refits of a 2-state Gaussian HMM, 2015-2024."""
    wf, fits = _hmm_walk_forward()
    r = _daily_pct_returns().reindex(wf.index)
    turb = wf["p_turb"] > 0.5
    blue, orange = pal["categorical"][0], pal["categorical"][1]

    fig, (ax0, ax1, ax2) = panels(3, heights=(2.2, 1.6, 1.4), figsize=(8.0, 7.0))
    ax0.scatter(r.index[~turb], r[~turb], s=4, color=blue, linewidths=0, label="P(turbulent) at most 0.5")
    ax0.scatter(r.index[turb], r[turb], s=9, color=orange, linewidths=0, label="P(turbulent) above 0.5")
    zero_line(ax0)
    ax0.set_title("Daily returns, coloured by the filtered regime probability")
    ax0.set_ylabel("daily log return (%)")
    add_legend(ax0, loc="lower left", fontsize=8.5, markerscale=2)

    ax1.fill_between(wf.index, 0, wf["p_turb"], color=orange, alpha=0.35, linewidth=0, step="pre")
    ax1.plot(wf.index, wf["p_turb"], color=orange, linewidth=0.8, drawstyle="steps-pre")
    ax1.axhline(0.5, color=pal["axis"], linewidth=0.8, linestyle="--")
    ax1.set_ylim(0, 1)
    percent_axis(ax1)
    ax1.set_title("Filtered P(turbulent | returns up to today); the steps on 1 January are refits")
    ax1.set_ylabel("probability")

    years = [pd.Timestamp(f"{f['year']}-01-01", tz="UTC") for f in fits]
    ends = years[1:] + [wf.index[-1]]
    for f, a, b in zip(fits, years, ends, strict=True):
        ax2.hlines(f["sd"][0], a, b, color=blue, linewidth=2)
        ax2.hlines(f["sd"][1], a, b, color=orange, linewidth=2)
    ax2.annotate("calm state", (ends[-1], fits[-1]["sd"][0]), xytext=(4, 0), textcoords="offset points",
                 va="center", color=pal["text"], fontsize=8.5)
    ax2.annotate("turbulent state", (ends[-1], fits[-1]["sd"][1]), xytext=(4, 0), textcoords="offset points",
                 va="center", color=pal["text"], fontsize=8.5)
    ax2.set_ylim(0, None)
    ax2.set_title("Daily standard deviation of each fitted state (one fit per year)")
    ax2.set_ylabel("sd of daily return (%)")
    ax2.set_xlabel("session date (UTC); year Y uses a model fitted on years Y-3 to Y-1")
    source_note(fig, "Data: " + source_label("D1_nyclose"), pal)
    return fig


@register(PAGE, "hmm-emissions", needs=("data_store",))
def hmm_emissions(pal):
    """Fitted emission densities (weighted by the stationary distribution) over the training histogram."""
    from scipy import stats

    windows = (("2015", "2017"), ("2012", "2014"))
    fig, axes = figure(1, 2, figsize=FIGSIZE, sharey=True)
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    x = np.linspace(-5, 5, 600)
    for ax, (a, b) in zip(axes, windows, strict=True):
        r = _daily_pct_returns().loc[a:b]
        hmm = _hmm_fit(a, b)
        pi = hmm.stationary_distribution_
        mu = np.ravel(hmm.means_)
        sd = np.sqrt(np.ravel(hmm.vars_))
        dur = hmm.expected_durations_
        ax.hist(r.clip(-5, 5), bins=80, range=(-5, 5), density=True,
                color=_blend(pal["grid"], pal["text_secondary"], 0.3), edgecolor=pal["surface"], linewidth=0.3)
        mix = np.zeros_like(x)
        for k, (colr, name) in enumerate(((blue, "calm"), (orange, "turbulent"))):
            dens = pi[k] * stats.norm.pdf(x, mu[k], sd[k])
            mix += dens
            ax.plot(x, dens, color=colr, label=f"{name}: sd {sd[k]:.2f}%, share {pi[k]:.2f}, "
                                               f"stays {dur[k]:.1f} d")
        ax.plot(x, mix, color=pal["text_secondary"], linewidth=1.2, linestyle="--", label="mixture (sum)")
        ax.set_xlim(-4.5, 4.5)
        ax.set_title(f"trained on {a}-{b}", fontsize=10)
        ax.set_xlabel("daily log return (%)")
        add_legend(ax, loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("density")
    fig.suptitle("Same model, two 3-year windows: long regimes (left), short bursts (right)")
    source_note(fig, "Data: " + source_label("D1_nyclose", start="2012-01-01", end="2017-12-31"), pal)
    return fig


@register(PAGE, "blend-vol", needs=("data_store",))
def blend_vol_figure(pal):
    """blend_vol of the EWMA (H1) and GARCH (D1) forecasts, and its gap to the plain mean of vols."""
    import matplotlib.dates as mdates

    df = _daily_forecasts().loc["2018-01-01":"2024-12-31", ["ewma", "garch"]].dropna()
    blend = blend_vol(df["ewma"], df["garch"])
    gap = blend - df.mean(axis=1)
    zoom = slice("2020-02-03", "2020-05-29")
    c = pal["categorical"]

    fig, (ax0, ax1) = panels(2, heights=(3, 2), figsize=(8.0, 6.0), sharex=False)
    ax0.plot(df.loc[zoom].index, df.loc[zoom, "ewma"], color=c[0], label="EWMA, H1")
    ax0.plot(df.loc[zoom].index, df.loc[zoom, "garch"], color=c[1], label="GARCH(1,1), D1")
    ax0.plot(blend.loc[zoom].index, blend.loc[zoom], color=pal["text"], linewidth=2.0, linestyle="--",
             label="blend_vol (equal weights)")
    percent_axis(ax0)
    ax0.set_ylim(0, 0.65)
    ax0.set_xlim(df.loc[zoom].index[0], df.loc[zoom].index[-1])
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz="UTC"))
    ax0.set_title("February to May 2020: the blend sits between its inputs")
    ax0.set_ylabel("annualised volatility")
    ax0.set_xlabel("session date (UTC)")
    add_legend(ax0, loc="upper left", fontsize=8.5)

    ax1.plot(gap.index, gap.values * 100.0, color=c[0], linewidth=1.2)
    t = gap.idxmax()
    ax1.plot([t], [gap.max() * 100.0], marker="o", markersize=5, color=c[0],
             markeredgecolor=pal["surface"], linestyle="none")
    ax1.annotate(f"largest gap {gap.max() * 100:.2f} vol points ({t:%d %b %Y})", (t, gap.max() * 100.0),
                 xytext=(8, -2), textcoords="offset points", va="center", color=pal["text"], fontsize=8.5)
    ax1.set_ylim(0, max(gap.max() * 100.0 * 1.25, 0.1))
    ax1.set_title("Blend minus the plain average of the two vols, 2018-2024 (never negative)")
    ax1.set_ylabel("vol points")
    ax1.set_xlabel("session date (UTC)")
    source_note(fig, "Data: XAUUSD H1 (EWMA) and D1 NY close (GARCH fitted 2012-2017), Dukascopy bid/ask mid, "
                "2018-2024", pal)
    return fig
