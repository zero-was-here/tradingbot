"""aurum/viz/style.py: palette values, colour rules, matplotlib rcParams, save_figure, plotly template."""

from __future__ import annotations

import json

import matplotlib as mpl
import numpy as np
import pytest

from aurum.viz import style as S
from aurum.viz.style import (
    add_legend,
    categorical,
    current_mode,
    diverging,
    figure,
    label_line_end,
    palette,
    plotly_template,
    rc_params,
    save_figure,
    sequential,
    series_color,
    use_style,
)

LIGHT_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK_CAT = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _lum(hex_color: str) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(int(hex_color[i:i + 2], 16) / 255) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ---------------------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------------------
def test_reference_palette_values_both_modes() -> None:
    lp, dp = palette("light"), palette("dark")
    assert lp["surface"] == "#fcfcfb" and dp["surface"] == "#1a1a19"
    assert lp["text"] == "#0b0b0b" and dp["text"] == "#f0efec"
    assert lp["text_secondary"] == "#52514e" and dp["text_secondary"] == "#c3c2b7"
    assert lp["categorical"] == LIGHT_CAT and dp["categorical"] == DARK_CAT
    assert lp["sequential"] == RAMP and dp["sequential"] == RAMP[::-1]
    assert lp["diverging_mid"] == "#f0efec" and dp["diverging_mid"] == "#383835"
    assert lp["status"] == {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
    assert dp["status"] == lp["status"]


def test_palette_is_a_fresh_copy() -> None:
    p = palette("light")
    p["categorical"][0] = "#000000"
    p["status"]["good"] = "#000000"
    assert palette("light")["categorical"][0] == "#2a78d6"
    assert palette("light")["status"]["good"] == "#0ca30c"


def test_invalid_mode() -> None:
    with pytest.raises(ValueError):
        palette("sepia")
    with pytest.raises(ValueError):
        with use_style("blue"):
            pass


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_text_inks_have_accessible_contrast(mode: str) -> None:
    p = palette(mode)
    assert _contrast(p["text"], p["surface"]) >= 7.0
    assert _contrast(p["text_secondary"], p["surface"]) >= 4.5
    if mode == "dark":     # dark categorical steps were chosen to clear 3:1 on the dark surface
        assert min(_contrast(c, p["surface"]) for c in p["categorical"]) >= 3.0


def test_categorical_fixed_order_never_cycles() -> None:
    assert categorical(3, "light") == LIGHT_CAT[:3]
    assert categorical(8, "dark") == DARK_CAT
    assert series_color(7, "light") == "#e34948"
    with pytest.raises(ValueError, match="Other"):
        categorical(9)
    with pytest.raises(IndexError):
        series_color(8)
    with pytest.raises(ValueError):
        categorical(-1)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_diverging_red_gray_blue(mode: str) -> None:
    stops = palette(mode)["diverging"]
    assert len(stops) == 7 and stops[3] == palette(mode)["diverging_mid"]
    assert stops[1] == palette(mode)["categorical"][7] and stops[5] == palette(mode)["categorical"][0]
    lo, hi = stops[0], stops[-1]
    assert int(lo[1:3], 16) > int(lo[5:7], 16), "negative end is red"
    assert int(hi[5:7], 16) > int(hi[1:3], 16), "positive end is blue"
    # equal arms: lightness of mirrored steps match (OKLab L within 0.02)
    L = [S._to_oklab(c)[0] for c in stops]
    for i in range(3):
        assert abs(L[i] - L[6 - i]) < 0.02 or i == 1  # pole hues keep their own steps
    # light mode: extremes darker than the arms next to the midpoint; dark mode: brighter
    if mode == "light":
        assert L[0] < L[2] and L[6] < L[4]
    else:
        assert L[0] > L[2] and L[6] > L[4]
    out = diverging(11, mode)
    assert len(out) == 11 and out[5] == stops[3] and out[0] == stops[0] and out[-1] == stops[-1]
    with pytest.raises(ValueError):
        diverging(4)


def test_sequential_sampling() -> None:
    assert sequential(None, "light") == RAMP
    s5 = sequential(5, "light")
    assert s5[0] == RAMP[0] and s5[-1] == RAMP[-1] and len(s5) == 5
    assert sequential(5, "dark")[0] == RAMP[-1]
    assert len(sequential(1)) == 1
    with pytest.raises(ValueError):
        sequential(0)


def test_oklab_roundtrip_and_rgba() -> None:
    for c in LIGHT_CAT + DARK_CAT + RAMP:
        assert S._from_oklab(S._to_oklab(c)) == c
    assert S.hex_to_rgba("#2a78d6", 0.1) == "rgba(42,120,214,0.1)"
    with pytest.raises(ValueError):
        S.hex_to_rgba("#abc", 1.0)


def test_colormaps_registered() -> None:
    for mode in ("light", "dark"):
        assert S.sequential_cmap(mode).name == f"aurum_sequential_{mode}"
        cm = S.diverging_cmap(mode)
        mid = mpl.colors.to_hex(cm(0.5))
        assert _contrast(mid, palette(mode)["diverging_mid"]) < 1.1   # gray in the middle


# ---------------------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_rc_params_are_valid_and_encode_the_rules(mode: str) -> None:
    rc = rc_params(mode)
    with mpl.rc_context(rc):        # every key/value must be accepted by matplotlib
        pass
    p = palette(mode)
    assert rc["axes.facecolor"] == rc["figure.facecolor"] == rc["savefig.facecolor"] == p["surface"]
    assert rc["lines.linewidth"] == pytest.approx(1.5)               # 2 px
    assert rc["grid.linewidth"] == pytest.approx(0.75)               # 1 px hairline
    assert rc["grid.linestyle"] == "-" and rc["axes.grid.axis"] == "y"
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    assert list(rc["axes.prop_cycle"].by_key()["color"]) == p["categorical"]
    assert rc["text.color"] == p["text"] and rc["xtick.labelcolor"] == p["text_secondary"]
    assert rc["lines.markersize"] >= 6.0 and rc["lines.markeredgecolor"] == p["surface"]
    assert rc["patch.linewidth"] == 0.0


def test_use_style_applies_restores_and_nests() -> None:
    before = mpl.rcParams["axes.facecolor"]
    assert current_mode() == "light"
    with use_style("dark") as pal:
        assert pal["mode"] == "dark" and current_mode() == "dark"
        assert mpl.rcParams["axes.facecolor"] == "#1a1a19"
        with use_style("light"):
            assert mpl.rcParams["axes.facecolor"] == "#fcfcfb" and current_mode() == "light"
        assert mpl.rcParams["axes.facecolor"] == "#1a1a19"
    assert mpl.rcParams["axes.facecolor"] == before and current_mode() == "light"


def test_figure_is_pyplot_free_and_styled() -> None:
    with use_style("dark"):
        fig, ax = figure()
        assert type(fig).__name__ == "Figure" and fig.canvas.manager is None
        assert mpl.colors.to_hex(ax.get_facecolor()) == "#1a1a19"
        fig2, axes = figure(2, 2)
        assert axes.shape == (2, 2)


def test_add_legend_only_for_two_or_more_series() -> None:
    with use_style("light"):
        fig, ax = figure()
        ax.plot([0, 1], [0, 1], label="only")
        assert add_legend(ax) is None
        ax.plot([0, 1], [1, 0], label="second")
        leg = add_legend(ax)
        assert leg is not None and [t.get_text() for t in leg.get_texts()] == ["only", "second"]
        assert all(mpl.colors.to_hex(t.get_color()) == "#0b0b0b" for t in leg.get_texts())


def test_label_line_end_uses_ink_not_series_colour() -> None:
    with use_style("dark"):
        fig, ax = figure()
        (line,) = ax.plot([0, 1, 2], [1.0, 2.0, np.nan], label="tsmom")
        ann = label_line_end(ax, line)
        assert ann.get_text() == "tsmom  2.00"
        assert mpl.colors.to_hex(ann.get_color()) == palette("dark")["text"]
        assert mpl.colors.to_hex(ann.get_color()) != mpl.colors.to_hex(line.get_color())
        (empty,) = ax.plot([0, 1], [np.nan, np.nan])
        assert label_line_end(ax, empty) is None


def _corner_rgb(path) -> str:
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path))
    return mpl.colors.to_hex(img[1, 1, :3])


def test_save_figure_renders_both_modes(tmp_path) -> None:
    seen = []

    def draw(pal):
        seen.append(pal["mode"])
        fig, ax = figure(figsize=(4, 3))
        for i in range(3):
            ax.plot(np.arange(10), np.arange(10) * (i + 1), label=f"s{i}")
        add_legend(ax)
        assert [mpl.colors.to_hex(ln.get_color()) for ln in ax.get_lines()] == pal["categorical"][:3]
        return fig

    paths = save_figure(draw, tmp_path / "figs" / "demo")
    assert [p.name for p in paths] == ["demo-light.png", "demo-dark.png"]
    assert seen == ["light", "dark"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert _corner_rgb(paths[0]) == "#fcfcfb" and _corner_rgb(paths[1]) == "#1a1a19"
    assert mpl.rcParams["axes.facecolor"] != "#1a1a19" or current_mode() == "light"


def test_save_figure_variants(tmp_path) -> None:
    def no_arg():
        fig, ax = figure(figsize=(3, 2))
        ax.bar(["a", "b"], [1, 2])
        return ax                                   # an Axes is accepted

    p = save_figure(no_arg, tmp_path / "bars.png", modes=("dark",), formats=("png", "svg"), dpi=80)
    assert [x.name for x in p] == ["bars-dark.png", "bars-dark.svg"]
    assert "<svg" in p[1].read_text()

    def grid(pal):
        fig, axes = figure(1, 2, figsize=(4, 2))
        return axes                                 # an array of Axes too

    assert len(save_figure(grid, tmp_path / "grid", modes=("light",))) == 1

    def pyplot_none(pal):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(2, 2))
        plt.plot([0, 1], [0, 1])
        return None                                 # current pyplot figure

    import matplotlib.pyplot as plt

    n_before = len(plt.get_fignums())
    assert save_figure(pyplot_none, tmp_path / "pp", modes=("light",))[0].exists()
    assert len(plt.get_fignums()) == n_before       # closed afterwards

    with pytest.raises(TypeError):
        save_figure(lambda pal: 42, tmp_path / "bad", modes=("light",))
    with pytest.raises(ValueError):
        save_figure(no_arg, tmp_path / "bad2", modes=("night",))


def test_save_figure_is_byte_stable(tmp_path) -> None:
    def draw(pal):
        fig, ax = figure(figsize=(3, 2))
        ax.plot([0, 1, 2], [0, 1, 0])
        return fig

    a = save_figure(draw, tmp_path / "a", modes=("light",))[0].read_bytes()
    b = save_figure(draw, tmp_path / "b", modes=("light",))[0].read_bytes()
    assert a == b


# ---------------------------------------------------------------------------------------
# plotly
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_plotly_template(mode: str) -> None:
    t = plotly_template(mode)
    json.dumps(t)
    lay = t["layout"]
    p = palette(mode)
    assert lay["paper_bgcolor"] == lay["plot_bgcolor"] == p["surface"]
    assert lay["colorway"] == p["categorical"]
    assert lay["yaxis"]["gridcolor"] == p["grid"] and lay["xaxis"]["showgrid"] is False
    assert lay["colorscale"]["diverging"][3][1] == p["diverging_mid"]
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure(layout={"template": t})
    assert fig.layout.template.layout.paper_bgcolor == p["surface"]


def test_lazy_package_exports() -> None:
    import aurum.viz as viz

    assert viz.palette is palette and "build_dashboard" in dir(viz)
    with pytest.raises(AttributeError):
        viz.no_such_name  # noqa: B018


def test_axis_helpers() -> None:
    with use_style("light"):
        fig, ax = figure()
        ax.plot([0, 1, 2], [-0.1, 0.05, 0.2])
        S.percent_axis(ax)
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        assert labels and all(lbl.endswith("%") for lbl in labels)
        zl = S.zero_line(ax)
        assert mpl.colors.to_hex(zl.get_color()) == palette("light")["axis"]
        band = S.shade_span(ax, 0.5, 1.5, "holdout")
        assert mpl.colors.to_hex(band.get_facecolor()) == palette("light")["grid"]
        assert any(t.get_text() == "holdout" for t in ax.texts)
        S.style_axes(ax, grid=None)
        assert not any(gl.get_visible() for gl in ax.yaxis.get_gridlines())
