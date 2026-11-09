"""Figures for docs/example.md -- TEMPLATE (copy to fig_<page>.py; this file is not discovered).

Each ``@register``-ed ``draw(pal)`` becomes docs/_assets/figures/<PAGE>/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only <PAGE>
"""

from __future__ import annotations

import numpy as np
from _common import (
    FIGSIZE_SQUARE,
    FIGSIZE_TALL,
    bars,
    ink_on,
    panels,
    register,
    signed,
    source_label,
    source_note,
)

from aurum.models.volatility import ewma_volatility
from aurum.viz.style import (
    diverging_cmap,
    figure,
    label_line_end,
    percent_axis,
    shade_span,
    style_axes,
)

PAGE = "example"


@register(PAGE, "price-and-vol", needs=("data_store",))
def price_and_vol(pal):
    """Gold close with its 200-day EMA (top) and annualised EWMA volatility (bottom)."""
    close_all = bars("D1_nyclose")["close"]            # real data, already cut at 2024-12-31
    ema_all = close_all.ewm(span=200, adjust=False).mean()
    vol_all = ewma_volatility(close_all, halflife_bars=20)   # aurum's own estimator
    show = slice("2015-01-01", None)                   # warm up on history, then display
    close, ema, vol = close_all.loc[show], ema_all.loc[show], vol_all.loc[show]
    lo = close.loc["2020-03"].idxmin()                 # spans come from the data, not by hand
    hi = close.loc["2020-07":"2020-08"].idxmax()

    fig, (ax0, ax1) = panels(2, heights=(3, 2), figsize=FIGSIZE_TALL)
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    shade_span(ax0, lo, hi, "Mar-Aug 2020 rally")    # recessive band, muted label
    shade_span(ax1, lo, hi)
    line_close, = ax0.plot(close.index, close.values, color=blue, label="close")
    line_ema, = ax0.plot(ema.index, ema.values, color=orange, label="200-day EMA")
    label_line_end(ax0, line_close, fmt="{:,.0f}")   # direct labels instead of a legend
    label_line_end(ax0, line_ema, fmt="{:,.0f}")
    ax0.set_title("XAUUSD daily close and its 200-day EMA")
    ax0.set_ylabel("USD per ounce")

    ax1.plot(vol.index, vol.values, color=blue)
    percent_axis(ax1)                                  # 0.15 -> 15%
    ax1.set_title("Annualised EWMA volatility (half-life 20 days)")
    ax1.set_ylabel("volatility")
    ax1.set_xlabel("date (UTC)")
    source_note(fig, "Data: " + source_label("D1_nyclose", start="2015-01-01"), pal)
    return fig


@register(PAGE, "toy-sharpe-grid")
def toy_sharpe_grid(pal):
    """Synthetic illustration: a smooth Sharpe surface on a fast/slow grid (diverging map)."""
    rng = np.random.default_rng(3)                      # seed everything you draw
    fast = np.array([8, 16, 32, 64])
    slow = np.array([64, 128, 256, 512])
    f, s = np.meshgrid(np.log2(fast), np.log2(slow), indexing="ij")
    z = 0.4 * np.exp(-((f - 4.2) ** 2 + (s - 7.4) ** 2) / 2.0) - 0.15 + 0.03 * rng.standard_normal(f.shape)

    fig, ax = figure(figsize=FIGSIZE_SQUARE)
    style_axes(ax, grid=None)                           # heatmaps: no grid
    lim = float(np.abs(z).max())                        # symmetric limits around 0
    im = ax.imshow(z, cmap=diverging_cmap(pal["mode"]), vmin=-lim, vmax=lim, origin="lower")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, signed(z[i, j]), ha="center", va="center", fontsize=9,
                    color=ink_on(im.cmap(im.norm(z[i, j]))))   # readable on any cell
    ax.set_xticks(range(len(slow)), [str(v) for v in slow])
    ax.set_yticks(range(len(fast)), [str(v) for v in fast])
    ax.set_xlabel("slow window (bars)")
    ax.set_ylabel("fast window (bars)")
    ax.set_title("Toy Sharpe ratio surface (synthetic)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Sharpe ratio", color=pal["text_secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=pal["axis"], labelcolor=pal["text_secondary"])
    return fig
