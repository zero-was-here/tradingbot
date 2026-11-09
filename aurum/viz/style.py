"""Shared chart style for every Aurum figure (matplotlib) and dashboard (Plotly).

One palette, two modes. The values implement the validated reference palette of the
data-visualisation method used across the project (surfaces, ink, an 8-slot categorical
order, a single-hue sequential ramp, a blue<->red diverging pair with a gray midpoint and
a reserved status set). Both modes are *selected*, not flipped: the dark column is the same
hues stepped for the dark surface. The categorical order passes the colour-vision checks in
both modes (worst adjacent CVD Delta E 9.1 light / 8.4 dark, OKLab x100); three light-mode
slots (aqua, yellow, magenta) are below 3:1 contrast on the light surface, so charts that
use them must also give direct labels or a table view (the "relief rule").

Rules this module encodes (and every figure should keep)
--------------------------------------------------------
* **2px lines** (1.5 pt: matplotlib sizes are points and 1 CSS px = 0.75 pt), round caps
  and joins; markers >= 8 px with a 2 px ring in the surface colour.
* **Recessive chrome**: hairline (1 px) solid gridlines one step off the surface, y-grid
  only, no top/right spines, muted axis lines, no tick marks on the value axis.
* **Thin marks**: bars without borders (separate touching bars with a surface gap, never a
  stroke).
* **Never dual y-axes**: two measures of different scale -> two panels or index both to a
  common base. (Nothing here creates ``twinx``; please don't either.)
* **A legend for >= 2 series** (:func:`add_legend` only draws one then); none for a single
  series, whose title names it.
* **Text wears text colours** (primary / secondary / muted), never a series colour:
  :func:`label_line_end` writes in ink and puts identity in a coloured end-dot.
* **Categorical colours in fixed order, never cycled**: :func:`categorical` refuses more
  than 8 series (fold the tail into "Other" or use small multiples).
* **Sequential = one hue light->dark; diverging = two hues + gray midpoint.** In dark mode
  the sequential ramp runs dark->light so that "near zero" recedes into the dark surface.
* **Status colours are reserved** for good/warning/serious/critical and always paired with
  an icon or a word.

Quick start
-----------
>>> from aurum.viz.style import figure, save_figure, add_legend
>>> def draw(pal):                       # pal = palette(mode) for the mode being rendered
...     fig, ax = figure()
...     ax.plot([0, 1, 2], [1, 3, 2], label="tsmom")
...     ax.plot([0, 1, 2], [2, 1, 2], label="ema_cross")
...     add_legend(ax)
...     ax.set_title("Two series")
...     return fig
>>> paths = save_figure(draw, "docs/_assets/figures/example")   # doctest: +SKIP
>>> [p.name for p in paths]                                     # doctest: +SKIP
['example-light.png', 'example-dark.png']

Helpers: :func:`figure` (styled, pyplot-free), :func:`add_legend` (only for >= 2 series),
:func:`label_line_end` (direct end label in ink), :func:`zero_line`, :func:`shade_span`
(recessive bands for holdouts / regimes), :func:`percent_axis`, :func:`style_axes` (re-apply
the chrome, e.g. ``grid=None`` for heatmaps), :func:`categorical` / :func:`sequential` /
:func:`diverging` and the matching colormaps.

The same function is rendered once per mode inside :func:`use_style`, so it must create its
figure *inside* the call (rcParams are read when artists are created). Use :func:`figure`
(an object-oriented ``matplotlib.figure.Figure``; no pyplot state, safe on headless servers)
or pyplot - both work.

:func:`plotly_template` returns the matching Plotly layout template (used by
:mod:`aurum.viz.dashboard`), so static docs figures and interactive dashboards read as one
system.
"""

from __future__ import annotations

import inspect
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AXIS",
    "BORDER",
    "CATEGORICAL",
    "DELTA_GOOD",
    "DIVERGING_MID",
    "FONT_STACK_CSS",
    "GRID",
    "LINE_WIDTH_PT",
    "MODES",
    "PAGE",
    "SEQUENTIAL",
    "SEQUENTIAL_STEPS",
    "STATUS",
    "SURFACE",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "Palette",
    "add_legend",
    "categorical",
    "current_mode",
    "diverging",
    "diverging_cmap",
    "figure",
    "get_palette",
    "hex_to_rgba",
    "label_line_end",
    "palette",
    "percent_axis",
    "plotly_template",
    "rc_params",
    "save_figure",
    "sequential",
    "sequential_cmap",
    "series_color",
    "shade_span",
    "style_axes",
    "use_style",
    "zero_line",
]

# ---------------------------------------------------------------------------------------
# palette values (the reference instance; swap these, keep the rules)
# ---------------------------------------------------------------------------------------
MODES: tuple[str, str] = ("light", "dark")

#: chart surface
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
#: page plane behind cards
PAGE = {"light": "#f9f9f7", "dark": "#0d0d0d"}
TEXT_PRIMARY = {"light": "#0b0b0b", "dark": "#f0efec"}
TEXT_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
#: muted ink (axis captions, de-emphasised labels)
TEXT_MUTED = {"light": "#898781", "dark": "#898781"}
#: hairline gridline
GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}
#: baseline / axis line
AXIS = {"light": "#c3c2b7", "dark": "#383835"}
#: hairline ring around cards / tooltips
BORDER = {"light": "rgba(11,11,11,0.10)", "dark": "rgba(255,255,255,0.10)"}
#: "delta up is good" success text (darker than status-good for text contrast in light mode)
DELTA_GOOD = {"light": "#006300", "dark": "#0ca30c"}

#: 8 categorical slots in FIXED order: blue, orange, aqua, yellow, magenta, green, violet, red
CATEGORICAL: dict[str, tuple[str, ...]] = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
}
CATEGORICAL_NAMES: tuple[str, ...] = ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")

#: single-hue (blue) sequential ramp, light -> dark
SEQUENTIAL_STEPS: tuple[int, ...] = (100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700)
SEQUENTIAL: tuple[str, ...] = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                               "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")

#: neutral gray midpoint of the blue<->red diverging scale
DIVERGING_MID = {"light": "#f0efec", "dark": "#383835"}

#: reserved status colours (mode-invariant); always pair with an icon or a label
STATUS: dict[str, str] = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

#: CSS font stack (dashboards); matplotlib uses the first installed of _MPL_FONTS
FONT_STACK_CSS = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
_MPL_FONTS = ("Inter", "SF Pro Text", "Helvetica Neue", "Helvetica", "Segoe UI", "Arial",
              "Liberation Sans", "DejaVu Sans")

#: 1 CSS px = 0.75 pt. Lines are 2 px, hairlines 1 px, markers 8 px, rings 2 px.
PX_PT = 0.75
LINE_WIDTH_PT = 2 * PX_PT
HAIRLINE_PT = 1 * PX_PT
MARKER_SIZE_PT = 8 * PX_PT
RING_PT = 2 * PX_PT

_MODE_STACK: list[str] = []


# ---------------------------------------------------------------------------------------
# colour maths (sRGB <-> OKLab), used to derive the red arm of the diverging scale
# ---------------------------------------------------------------------------------------
def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #rrggbb, got {h!r}")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{int(round(min(max(c, 0.0), 1.0) * 255)):02x}" for c in rgb)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = min(max(c, 0.0), 1.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _to_oklab(hex_color: str) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb(hex_color))
    lc = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    mc = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    sc = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (lc, mc, sc))
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _from_oklab(lab: Sequence[float]) -> str:
    L, a, b = lab
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    lc, mc, sc = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc
    g = -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc
    bb = -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc
    return _rgb_to_hex([_linear_to_srgb(v) for v in (r, g, bb)])


def _match_lightness(hex_color: str, target_hex: str) -> str:
    """Move ``hex_color`` toward black or white in OKLab until its L equals the target's.

    Mixing toward black scales (L, a, b) uniformly (stays in gamut); mixing toward white
    blends with (1, 0, 0). Hue is preserved, chroma shrinks proportionally.
    """
    L, a, b = _to_oklab(hex_color)
    Lt = _to_oklab(target_hex)[0]
    if Lt <= L:
        k = Lt / L if L > 0 else 0.0
        return _from_oklab((L * k, a * k, b * k))
    t = (Lt - L) / (1.0 - L) if L < 1.0 else 0.0
    return _from_oklab((L + t * (1.0 - L), a * (1.0 - t), b * (1.0 - t)))


def _mix(c1: str, c2: str, t: float) -> str:
    """Linear interpolation in OKLab (perceptually even steps)."""
    a, b = _to_oklab(c1), _to_oklab(c2)
    return _from_oklab(tuple(x + (y - x) * t for x, y in zip(a, b, strict=True)))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` -> ``rgba(r,g,b,alpha)`` (area fills are the series hue at ~10%)."""
    r, g, b = (int(round(c * 255)) for c in _hex_to_rgb(hex_color))
    return f"rgba({r},{g},{b},{float(alpha):g})"


# ---------------------------------------------------------------------------------------
# palette objects
# ---------------------------------------------------------------------------------------
def _check_mode(mode: str | None) -> str:
    m = current_mode() if mode is None else str(mode).lower()
    if m not in MODES:
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    return m


def current_mode() -> str:
    """Mode of the innermost active :func:`use_style` block (``"light"`` outside one)."""
    return _MODE_STACK[-1] if _MODE_STACK else "light"


def _diverging_stops(mode: str) -> tuple[str, ...]:
    """7 anchors, NEGATIVE (red) -> mid -> POSITIVE (blue).

    Light: extremes are the darkest steps (blue 650 / a red at the same OKLab L), the arm
    next to the gray midpoint the lightest (blue 200 / matched red). Dark: reversed in
    lightness so magnitude reads as contrast against the dark surface (near-mid steps
    recede: blue 600 / matched red; extremes blue 200 / matched red).
    """
    blue = CATEGORICAL[mode][0]
    red = CATEGORICAL[mode][7]
    if mode == "light":
        blue_near, blue_ext = SEQUENTIAL[2], SEQUENTIAL[11]
    else:
        blue_near, blue_ext = SEQUENTIAL[10], SEQUENTIAL[2]
    red_near = _match_lightness(red, blue_near)
    red_ext = _match_lightness(red, blue_ext)
    return (red_ext, red, red_near, DIVERGING_MID[mode], blue_near, blue, blue_ext)


@dataclass(frozen=True)
class Palette:
    """Every colour role for one mode (see the module docstring for the rules)."""

    mode: str
    surface: str
    page: str
    text: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    border: str
    delta_good: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]
    diverging: tuple[str, ...]
    diverging_mid: str
    status: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["categorical"] = list(self.categorical)
        d["sequential"] = list(self.sequential)
        d["diverging"] = list(self.diverging)
        d["status"] = dict(self.status)
        return d


def get_palette(mode: str | None = None) -> Palette:
    """The :class:`Palette` for ``mode`` (default: the active :func:`use_style` mode)."""
    m = _check_mode(mode)
    seq = SEQUENTIAL if m == "light" else tuple(reversed(SEQUENTIAL))
    return Palette(mode=m, surface=SURFACE[m], page=PAGE[m], text=TEXT_PRIMARY[m],
                   text_secondary=TEXT_SECONDARY[m], text_muted=TEXT_MUTED[m], grid=GRID[m],
                   axis=AXIS[m], border=BORDER[m], delta_good=DELTA_GOOD[m],
                   categorical=CATEGORICAL[m], sequential=seq, diverging=_diverging_stops(m),
                   diverging_mid=DIVERGING_MID[m], status=dict(STATUS))


def palette(mode: str | None = None) -> dict[str, Any]:
    """Plain-dict palette for ``mode`` (a fresh copy; safe to mutate).

    Keys: ``mode, surface, page, text, text_secondary, text_muted, grid, axis, border,
    delta_good, categorical (8), sequential (13, near-zero first), diverging (7, negative
    red -> gray -> positive blue), diverging_mid, status {good, warning, serious,
    critical}``.
    """
    return get_palette(mode).as_dict()


def categorical(n: int, mode: str | None = None) -> list[str]:
    """The first ``n`` categorical colours in fixed order. Refuses ``n > 8``.

    A 9th hue would be indistinguishable from an existing slot under colour-vision
    deficiency: fold the tail into "Other", facet into small multiples, or encode twice.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if n > 8:
        raise ValueError(f"{n} categorical series requested; the palette has 8 fixed slots - fold the "
                         "tail into 'Other' or use small multiples (never cycle or generate hues)")
    return list(CATEGORICAL[_check_mode(mode)][:n])


def series_color(i: int, mode: str | None = None) -> str:
    """Colour of categorical slot ``i`` (0-based). Raises ``IndexError`` past slot 8."""
    if not 0 <= int(i) < 8:
        raise IndexError(f"categorical slot {i} out of range 0..7 (never cycle colours)")
    return CATEGORICAL[_check_mode(mode)][int(i)]


def sequential(n: int | None = None, mode: str | None = None) -> list[str]:
    """``n`` evenly spaced steps of the blue ramp, "near zero" first (all 13 if ``None``).

    Light mode runs light -> dark, dark mode dark -> light (near zero recedes into the
    surface). For ordinal marks start no lighter than step 250 (light) / no darker than
    step 600 (dark) so the first step still clears 2:1 against the surface.
    """
    ramp = list(get_palette(mode).sequential)
    if n is None:
        return ramp
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [ramp[len(ramp) // 2]]
    return [ramp[round(i * (len(ramp) - 1) / (n - 1))] for i in range(n)]


def diverging(n: int = 7, mode: str | None = None) -> list[str]:
    """``n`` (odd) colours from negative (red) through the gray midpoint to positive (blue).

    Equal step count per arm; interpolated in OKLab between the 7 anchors.
    """
    n = int(n)
    if n < 3 or n % 2 == 0:
        raise ValueError("diverging needs an odd n >= 3 (equal arms around the gray midpoint)")
    stops = get_palette(mode).diverging
    out = []
    for i in range(n):
        x = i / (n - 1) * (len(stops) - 1)
        j = min(int(math.floor(x)), len(stops) - 2)
        out.append(_mix(stops[j], stops[j + 1], x - j))
    return out


def _cmap(name: str, colors: Sequence[str]) -> Any:
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if name in colormaps:
        return colormaps[name]
    cmap = LinearSegmentedColormap.from_list(name, list(colors), N=256)
    try:
        colormaps.register(cmap, name=name)
    except ValueError:  # registered concurrently
        pass
    return colormaps[name]


def sequential_cmap(mode: str | None = None) -> Any:
    """Matplotlib colormap of the sequential ramp (registered as ``aurum_sequential_<mode>``)."""
    m = _check_mode(mode)
    return _cmap(f"aurum_sequential_{m}", get_palette(m).sequential)


def diverging_cmap(mode: str | None = None) -> Any:
    """Matplotlib colormap red <-> gray <-> blue (``aurum_diverging_<mode>``); centre it with
    ``matplotlib.colors.TwoSlopeNorm(vcenter=0)`` or symmetric ``vmin = -vmax``."""
    m = _check_mode(mode)
    return _cmap(f"aurum_diverging_{m}", get_palette(m).diverging)


# ---------------------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------------------
def _font_list() -> list[str]:
    try:
        from matplotlib import font_manager

        have = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001 - font discovery must never break plotting
        have = set()
    fonts = [f for f in _MPL_FONTS if f in have]
    return fonts or ["DejaVu Sans"]


def _title_weight(fonts: Sequence[str]) -> str:
    """``"semibold"`` when the first font ships a 600 weight, else ``"bold"`` (asking
    matplotlib for a missing weight logs a findfont warning on every title)."""
    try:
        from matplotlib import font_manager

        first = fonts[0]
        weights = {f.weight for f in font_manager.fontManager.ttflist if f.name == first}
    except Exception:  # noqa: BLE001
        return "bold"
    return "semibold" if (600 in weights or "semibold" in weights or "demibold" in weights) else "bold"


def rc_params(mode: str | None = None) -> dict[str, Any]:
    """The matplotlib ``rcParams`` implementing the style for ``mode``."""
    from cycler import cycler

    pal = get_palette(mode)
    sequential_cmap(pal.mode)
    diverging_cmap(pal.mode)
    fonts = _font_list()
    weight = _title_weight(fonts)
    return {
        # surfaces
        "figure.facecolor": pal.surface,
        "figure.edgecolor": pal.surface,
        "savefig.facecolor": pal.surface,
        "savefig.edgecolor": pal.surface,
        "axes.facecolor": pal.surface,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.figsize": (8.0, 4.5),
        "figure.constrained_layout.use": True,
        # type: system sans, text in ink colours
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "font.size": 10.0,
        "text.color": pal.text,
        "axes.titlecolor": pal.text,
        "axes.titlesize": 12.0,
        "axes.titleweight": weight,
        "axes.titlelocation": "left",
        "axes.titlepad": 10.0,
        "axes.labelcolor": pal.text_secondary,
        "axes.labelsize": 10.0,
        "figure.titlesize": 13.0,
        "figure.titleweight": weight,
        "legend.labelcolor": pal.text,
        "legend.fontsize": 9.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "xtick.labelcolor": pal.text_secondary,
        "ytick.labelcolor": pal.text_secondary,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        # recessive chrome: hairline solid y-grid, no top/right spines, muted axis
        "axes.edgecolor": pal.axis,
        "axes.linewidth": HAIRLINE_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": pal.grid,
        "grid.linewidth": HAIRLINE_PT,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "xtick.color": pal.axis,
        "ytick.color": pal.axis,
        "xtick.major.size": 3.0,
        "xtick.major.width": HAIRLINE_PT,
        "xtick.minor.visible": False,
        "ytick.major.size": 0.0,
        "ytick.major.width": HAIRLINE_PT,
        "axes.xmargin": 0.01,
        "date.converter": "concise",
        # marks: 2px lines with round ends, 8px markers with a 2px surface ring
        "axes.prop_cycle": cycler(color=list(pal.categorical)),
        "lines.linewidth": LINE_WIDTH_PT,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.markersize": MARKER_SIZE_PT,
        "lines.markeredgewidth": RING_PT,
        "lines.markeredgecolor": pal.surface,
        "scatter.edgecolors": pal.surface,
        "patch.linewidth": 0.0,
        "patch.edgecolor": pal.surface,
        "patch.force_edgecolor": False,
        "hatch.color": pal.axis,
        "boxplot.boxprops.color": pal.text_secondary,
        "image.cmap": f"aurum_sequential_{pal.mode}",
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": False,
        "svg.fonttype": "none",
    }


@contextmanager
def use_style(mode: str = "light") -> Iterator[dict[str, Any]]:
    """Apply the Aurum matplotlib style for ``mode`` inside the block; yield :func:`palette`.

    >>> with use_style("dark") as pal:           # doctest: +SKIP
    ...     fig, ax = figure()
    ...     ax.plot(x, y, color=pal["categorical"][0])

    Restores the previous rcParams on exit (``matplotlib.rc_context``), so it nests and
    never leaks into other code.
    """
    import matplotlib as mpl

    m = _check_mode(mode)
    _MODE_STACK.append(m)
    try:
        with mpl.rc_context(rc_params(m)):
            yield palette(m)
    finally:
        _MODE_STACK.pop()


def figure(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float] | None = None,
           sharex: bool | str = False, sharey: bool | str = False, squeeze: bool = True,
           **subplot_kw: Any) -> tuple[Any, Any]:
    """A styled ``matplotlib.figure.Figure`` and its axes, WITHOUT pyplot (headless-safe).

    Call inside :func:`use_style` (or :func:`save_figure`'s callback). Returns
    ``(fig, ax)`` like ``plt.subplots``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize)
    axes = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze,
                        subplot_kw=subplot_kw or None)
    return fig, axes


def style_axes(ax: Any, mode: str | None = None, *, grid: str | None = "y") -> Any:
    """Re-apply the recessive chrome to an existing ``Axes`` (e.g. one made outside
    :func:`use_style`): surface, spines, ticks, label colours and the grid (``"y"``,
    ``"x"``, ``"both"`` or ``None``)."""
    pal = get_palette(mode)
    ax.set_facecolor(pal.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(pal.axis)
        ax.spines[side].set_linewidth(HAIRLINE_PT)
    ax.tick_params(axis="both", colors=pal.axis, labelcolor=pal.text_secondary, width=HAIRLINE_PT)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.label.set_color(pal.text_secondary)
    ax.yaxis.label.set_color(pal.text_secondary)
    ax.title.set_color(pal.text)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=pal.grid, linewidth=HAIRLINE_PT, linestyle="-")
        ax.set_axisbelow(True)
    return ax


def add_legend(ax: Any, *, mode: str | None = None, min_series: int = 2, **kwargs: Any) -> Any:
    """Draw a frameless legend in ink colours, but only when ``>= min_series`` labelled
    artists exist (a single series is named by the title instead). Returns the legend or
    ``None``."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) < int(min_series):
        return None
    pal = get_palette(mode)
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("labelcolor", pal.text)
    kwargs.setdefault("loc", "best")
    return ax.legend(handles, labels, **kwargs)


def label_line_end(ax: Any, line: Any, text: str | None = None, *, mode: str | None = None,
                   dx_points: float = 6.0, fmt: str = "{:,.2f}", dot: bool = True) -> Any:
    """Direct label at the right end of ``line`` in TEXT colour (identity via an end-dot).

    ``text`` defaults to the line's label and last value. Use selectively (endpoints,
    extremes, the series the story is about) - never on every point.
    """
    import numpy as np

    pal = get_palette(mode)
    x = np.asarray(line.get_xdata())
    y = np.asarray(line.get_ydata(), dtype=float)
    ok = np.flatnonzero(np.isfinite(y))
    if ok.size == 0:
        return None
    i = int(ok[-1])
    if dot:
        ax.plot([x[i]], [y[i]], marker="o", linestyle="none", color=line.get_color(),
                markeredgecolor=pal.surface, markeredgewidth=RING_PT, markersize=MARKER_SIZE_PT,
                zorder=line.get_zorder() + 1, label="_nolegend_")
    if text is None:
        lab = line.get_label()
        val = fmt.format(y[i])
        text = val if (not lab or lab.startswith("_")) else f"{lab}  {val}"
    return ax.annotate(text, (x[i], y[i]), xytext=(dx_points, 0), textcoords="offset points",
                       va="center", ha="left", color=pal.text, fontsize=9, annotation_clip=False)


def percent_axis(ax: Any, axis: str = "y", *, decimals: int = 0, xmax: float = 1.0) -> Any:
    """Format an axis as percentages. ``xmax`` is the data value meaning 100% (1.0 for
    fractions such as returns and drawdowns, 100 for data already in percent)."""
    from matplotlib.ticker import PercentFormatter

    fmt = PercentFormatter(xmax=float(xmax), decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    return ax


def zero_line(ax: Any, *, mode: str | None = None, axis: str = "y") -> Any:
    """A hairline baseline at 0 in the axis colour (behind the data)."""
    pal = get_palette(mode)
    kw = {"color": pal.axis, "linewidth": HAIRLINE_PT, "zorder": 0.5}
    return ax.axhline(0.0, **kw) if axis == "y" else ax.axvline(0.0, **kw)


def shade_span(ax: Any, start: Any, end: Any, label: str | None = None, *, mode: str | None = None,
               axis: str = "x", alpha: float = 1.0) -> Any:
    """Shade a region (a holdout window, a regime, a fold) in the gridline colour - a
    recessive background band, never a data colour. ``label`` is written in muted ink at
    the top-left of the band."""
    pal = get_palette(mode)
    span = (ax.axvspan if axis == "x" else ax.axhspan)(start, end, color=pal.grid, alpha=alpha, zorder=0,
                                                     linewidth=0)
    if label:
        if axis == "x":
            ax.annotate(label, (start, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
                        textcoords="offset points", va="top", ha="left", color=pal.text_muted, fontsize=8)
        else:
            ax.annotate(label, (0.0, end), xycoords=("axes fraction", "data"), xytext=(4, -4),
                        textcoords="offset points", va="top", ha="left", color=pal.text_muted, fontsize=8)
    return span


def _call_fig_fn(fig_fn: Callable[..., Any], pal: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fig_fn)
        takes_arg = any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
                        for p in sig.parameters.values())
    except (TypeError, ValueError):  # builtins / C callables: assume the palette argument
        takes_arg = True
    return fig_fn(pal) if takes_arg else fig_fn()


def _resolve_figure(obj: Any) -> Any:
    if obj is None:
        if "matplotlib.pyplot" not in sys.modules:
            raise TypeError("fig_fn returned None and pyplot is not in use: return the Figure")
        import matplotlib.pyplot as plt

        return plt.gcf()
    if hasattr(obj, "savefig"):
        return obj
    if hasattr(obj, "figure") and hasattr(obj.figure, "savefig"):   # an Axes
        return obj.figure
    try:                                                            # an array of Axes
        first = next(iter(getattr(obj, "flat", obj)))
        if hasattr(first, "figure"):
            return first.figure
    except (TypeError, StopIteration):
        pass
    raise TypeError(f"fig_fn must return a Figure or Axes, got {type(obj).__name__}")


def save_figure(fig_fn: Callable[..., Any], path_stem: str | Path, modes: Sequence[str] = ("light", "dark"),
                dpi: int = 150, *, formats: Sequence[str] = ("png",), **savefig_kw: Any) -> list[Path]:
    """Render the SAME plotting function once per mode and save ``<stem>-<mode>.<fmt>``.

    ``fig_fn(pal)`` (or ``fig_fn()``) is called inside :func:`use_style` for each mode with
    that mode's :func:`palette` dict, and must create and return its figure (a ``Figure``, an
    ``Axes`` or an array of axes; ``None`` means the current pyplot figure). Parent
    directories are created; pyplot figures are closed afterwards. Returns the written paths
    in ``modes`` x ``formats`` order.

    ``savefig_kw`` go to ``Figure.savefig`` (``facecolor`` defaults to the mode's surface,
    PNG metadata is pinned so re-renders are byte-stable for a given matplotlib version).
    """
    stem = Path(path_stem)
    if stem.suffix.lower() in (".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"):
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for mode in modes:
        m = _check_mode(mode)
        with use_style(m) as pal:
            fig = _resolve_figure(_call_fig_fn(fig_fn, pal))
            try:
                for fmt in formats:
                    path = stem.parent / f"{stem.name}-{m}.{fmt.lstrip('.')}"
                    kw = dict(savefig_kw)
                    kw.setdefault("facecolor", pal["surface"])
                    kw.setdefault("edgecolor", pal["surface"])
                    if fmt.lower().lstrip(".") == "png":
                        kw.setdefault("metadata", {"Software": None})
                    fig.savefig(path, dpi=dpi, format=fmt.lstrip("."), **kw)
                    out.append(path)
            finally:
                if "matplotlib.pyplot" in sys.modules:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------
# plotly
# ---------------------------------------------------------------------------------------
def _plotly_scale(colors: Sequence[str]) -> list[list[Any]]:
    n = len(colors)
    return [[i / (n - 1), c] for i, c in enumerate(colors)]


def plotly_template(mode: str | None = None) -> dict[str, Any]:
    """A Plotly template dict (``{"layout": {...}}``) with the same palette and chrome.

    Pure data (no plotly import): use as ``go.layout.Template(plotly_template("dark"))`` or
    ``fig.update_layout(template=plotly_template())``.
    """
    pal = get_palette(mode)
    axis = {
        "gridcolor": pal.grid, "gridwidth": 1, "linecolor": pal.axis, "linewidth": 1,
        "zerolinecolor": pal.axis, "zerolinewidth": 1, "showline": True, "ticks": "",
        "tickfont": {"color": pal.text_secondary, "size": 11},
        "title": {"font": {"color": pal.text_secondary, "size": 12}, "standoff": 8},
        "automargin": True, "showspikes": False,
    }
    return {"layout": {
        "paper_bgcolor": pal.surface,
        "plot_bgcolor": pal.surface,
        "font": {"family": FONT_STACK_CSS, "color": pal.text_secondary, "size": 12},
        "title": {"font": {"color": pal.text, "size": 15}, "x": 0.0, "xanchor": "left"},
        "colorway": list(pal.categorical),
        "xaxis": {**axis, "showgrid": False},
        "yaxis": {**axis, "showgrid": True, "showline": False},
        "legend": {"font": {"color": pal.text, "size": 11}, "bgcolor": "rgba(0,0,0,0)",
                   "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        "hoverlabel": {"bgcolor": pal.surface, "bordercolor": pal.axis,
                       "font": {"color": pal.text, "family": FONT_STACK_CSS, "size": 12}},
        "colorscale": {"sequential": _plotly_scale(pal.sequential),
                       "diverging": _plotly_scale(pal.diverging)},
        "margin": {"l": 56, "r": 16, "t": 40, "b": 40},
        "hovermode": "x unified",
        "bargap": 0.25,
    }}
