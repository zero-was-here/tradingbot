"""Shared plumbing for the documentation figures (``tools/figures/fig_<page>.py``).

Figure modules import everything they need from here::

    from _common import FIGSIZE, bars, market_data, panels, register, source_label

What lives here
---------------
* **Data loaders, cut at :data:`FIGURE_CUTOFF`** (2024-12-31). :func:`bars` reads the real
  Dukascopy bars in ``data_store/`` with aurum's own :func:`aurum.data.store.load_bars` (hash
  verified) and keeps only rows whose ``available_at`` is on or before the end of the cutoff
  day, so no figure can ever show the 2025+ holdout. :func:`macro` does the same on each
  series' ``available_at`` (the publication lag aurum applies), and :func:`market_data`
  assembles an :class:`aurum.core.types.MarketData` exactly like
  :meth:`aurum.core.config.DataConfig.load` (rule-based event calendar included).
* **A deterministic synthetic fallback** (:mod:`aurum.data.synthetic`, fixed seeds) for a clone
  without ``data_store/``: ``bars(tf, synthetic=True)`` always gives it, and :func:`bars` falls
  back to it (with a warning) when the real file is missing. :func:`source_label` names what
  was actually used, so captions and in-figure notes cannot lie about the data.
* **Run outputs**: :func:`run_dir` / :func:`run_file` / :func:`run_table` read ``runs/...``
  (gitignored) and return ``None`` when a run is missing; tables are cut at the cutoff and
  ``holdout/`` sub-directories are refused.
* **The registry**: ``@register(page, name, needs=...)`` marks a ``draw(pal)`` function as a
  documentation figure. ``tools/figures/build.py`` discovers ``fig_*.py`` modules, renders
  each figure with :func:`aurum.viz.style.save_figure` into
  ``docs/_assets/figures/<page>/<name>-{light,dark}.png`` and skips (keeping the committed
  PNGs) any figure whose ``needs`` are missing.
* **Layout helpers** on top of :mod:`aurum.viz.style`: figure sizes, :func:`panels` (stacked
  panels sharing x, with height ratios) and :func:`source_note`.

Importing a figure module must never load data: loaders raise while ``build.py`` imports the
modules (see :func:`import_guard`). Load data inside ``draw(pal)`` only; the loaders cache, so
several figures of one page share one read.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import re
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("aurum.docs.figures")

# ---------------------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
REPO = TOOLS_DIR.parents[1]
DOCS = REPO / "docs"
FIGURES_DIR = DOCS / "_assets" / "figures"
DATA_DIR = REPO / "data_store"
MACRO_DIR = DATA_DIR / "macro"
RUNS_DIR = REPO / "runs"

# ---------------------------------------------------------------------------------------
# research hygiene: the cutoff
# ---------------------------------------------------------------------------------------
#: Last calendar day (UTC) any documentation figure may show. 2025-01-01 onwards is the
#: final holdout of the research protocol (configs/default.yaml walkforward.holdout_start).
FIGURE_CUTOFF = "2024-12-31"
#: End of the cutoff day. A row is visible to figures iff its ``available_at`` (bars, macro)
#: or its bar-open index (run tables) satisfies ``<= CUTOFF_END`` / ``< CUTOFF_END``.
CUTOFF_END = pd.Timestamp(FIGURE_CUTOFF, tz="UTC") + pd.Timedelta(days=1)

# ---------------------------------------------------------------------------------------
# figure geometry (inches at DPI); keep widths <= 8 in so text stays readable at 100% width
# ---------------------------------------------------------------------------------------
DPI = 150
FIGSIZE = (8.0, 4.5)           #: default single panel
FIGSIZE_SHORT = (8.0, 3.2)     #: a strip (timeline, one-line series)
FIGSIZE_TALL = (8.0, 6.5)      #: 2-3 stacked panels sharing x
FIGSIZE_SQUARE = (6.4, 5.6)    #: heatmaps, scatter plots
FIGSIZE_GRID = (8.0, 6.0)      #: small multiples (2x2, 2x3)
#: Render budget per figure (both modes). build.py warns above it.
MAX_SECONDS = 30.0

#: Deterministic seed for the synthetic fallback and for build.py's global RNG reset.
SYNTHETIC_SEED = 7

# ---------------------------------------------------------------------------------------
# data catalogue
# ---------------------------------------------------------------------------------------
#: timeframe key -> (file in data_store/, aurum timeframe, synthetic timeline start)
TIMEFRAMES: dict[str, tuple[str, str, str]] = {
    "M15": ("xauusd_M15.parquet", "M15", "2012-01-02 00:00"),
    "H1": ("xauusd_H1.parquet", "H1", "2012-01-02 00:00"),
    "H4": ("xauusd_H4.parquet", "H4", "2012-01-02 00:00"),
    "D1": ("xauusd_D1.parquet", "D1", "2012-01-02 00:00"),          # UTC-midnight days
    "D1_nyclose": ("xauusd_D1_nyclose.parquet", "D1", "2012-01-01 22:00"),  # 22:00 UTC days
}
_TF_DESCRIPTION = {
    "M15": "M15",
    "H1": "H1",
    "H4": "H4",
    "D1": "D1 (UTC days)",
    "D1_nyclose": "D1 (New York close, 22:00 UTC)",
}

_ENV_SYNTHETIC = "AURUM_FIGURES_SYNTHETIC"
_STATE: dict[str, Any] = {"force_synthetic": os.environ.get(_ENV_SYNTHETIC, "") == "1", "importing": False}
_USAGE: set[str] = set()


class FigureDataError(RuntimeError):
    """Raised when a figure asks for data it may not have (holdout, import time, ...)."""


# ---------------------------------------------------------------------------------------
# state switches used by build.py
# ---------------------------------------------------------------------------------------
def set_force_synthetic(flag: bool) -> None:
    """Make every :func:`bars` / :func:`macro` call return synthetic data (preview mode)."""
    _STATE["force_synthetic"] = bool(flag)


def force_synthetic() -> bool:
    return bool(_STATE["force_synthetic"])


@contextmanager
def import_guard() -> Iterator[None]:
    """While active, any data load raises :class:`FigureDataError` (used when importing
    ``fig_*.py`` modules: importing must not read data)."""
    prev = _STATE["importing"]
    _STATE["importing"] = True
    try:
        yield
    finally:
        _STATE["importing"] = prev


def _guard(what: str) -> None:
    if _STATE["importing"]:
        raise FigureDataError(f"{what} was loaded while a figure module was being imported; "
                              "load data inside draw(pal) only")


def reset_usage() -> None:
    _USAGE.clear()


def usage() -> set[str]:
    """What the current render touched: ``bars:H1:real``, ``bars:H1:fallback``,
    ``bars:H1:synthetic``, ``macro:real``, ``run:protocol/trend_core_selection``..."""
    return set(_USAGE)


# ---------------------------------------------------------------------------------------
# cutoff helpers
# ---------------------------------------------------------------------------------------
def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _end_bound(end: Any) -> pd.Timestamp:
    """``end`` as an inclusive bound on bar-open time (a bare date covers the whole day),
    refusing anything past the cutoff."""
    e = _utc(end)
    if isinstance(end, str) and len(end.strip()) <= 10:
        e = e + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if e >= CUTOFF_END:
        raise FigureDataError(f"end={end!r} is after the figure cutoff {FIGURE_CUTOFF} "
                              "(2025+ is the research holdout)")
    return e


def cut_at_cutoff(df: pd.DataFrame, *, on: str = "available_at") -> pd.DataFrame:
    """Rows of ``df`` visible at the cutoff.

    ``on="available_at"`` keeps ``available_at <= CUTOFF_END`` (bars, macro frames);
    ``on="index"`` keeps a bar-open-time index ``< CUTOFF_END`` (run outputs). Works on
    frames and series; anything without a datetime index/column is returned unchanged.
    """
    if on == "available_at" and isinstance(df, pd.DataFrame) and "available_at" in df.columns:
        return df.loc[pd.DatetimeIndex(df["available_at"]) <= CUTOFF_END]
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
        return df.loc[idx < CUTOFF_END]
    return df


# ---------------------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------------------
def _check_tf(tf: str) -> tuple[str, str, str]:
    try:
        return TIMEFRAMES[tf]
    except KeyError:
        raise ValueError(f"unknown timeframe {tf!r}; one of {sorted(TIMEFRAMES)}") from None


def data_available(tf: str = "H1") -> bool:
    """True when the real ``data_store/`` file for ``tf`` exists."""
    return (DATA_DIR / _check_tf(tf)[0]).exists()


def is_synthetic(tf: str = "H1", *, synthetic: bool = False) -> bool:
    """Whether :func:`bars` would return synthetic data for these arguments right now."""
    return bool(synthetic or force_synthetic() or not data_available(tf))


@functools.cache
def _real_bars(tf: str) -> pd.DataFrame:
    from aurum.data.store import load_bars

    fname, _, _ = _check_tf(tf)
    t0 = time.perf_counter()
    df = cut_at_cutoff(load_bars(DATA_DIR / fname, verify_hash=True))
    logger.debug("loaded %s real bars (%d rows) in %.2fs", tf, len(df), time.perf_counter() - t0)
    return df


def _synthetic_n(timeframe: str, start: str) -> int:
    """Number of open bars between ``start`` and the cutoff under the synthetic weekend rule."""
    from aurum.core.timeframes import get_timeframe

    tfo = get_timeframe(timeframe)
    idx = pd.date_range(_utc(start), CUTOFF_END - pd.Timedelta(microseconds=1), freq=tfo.freq)
    wd, hr = idx.weekday, idx.hour
    closed = (wd == 5) | ((wd == 4) & (hr >= 21)) | ((wd == 6) & (hr < 22))
    return int((~closed).sum())


@functools.cache
def _synthetic_bars(tf: str) -> pd.DataFrame:
    from aurum.data.synthetic import make_synthetic_bars

    _, timeframe, start = _check_tf(tf)
    n = _synthetic_n(timeframe, start)
    df = make_synthetic_bars(n, timeframe, seed=SYNTHETIC_SEED, model="gbm", start=start,
                             annual_vol=0.16, start_price=1600.0, spread=0.30)
    return cut_at_cutoff(df)


def bars(tf: str = "H1", *, start: Any = None, end: Any = None, synthetic: bool = False) -> pd.DataFrame:
    """Canonical XAUUSD bars for ``tf`` (``M15``, ``H1``, ``H4``, ``D1``, ``D1_nyclose``).

    Real data (``data_store/``, hash-verified) cut at :data:`FIGURE_CUTOFF` on
    ``available_at``; ``start`` / ``end`` further restrict the bar OPEN time (a bare-date
    ``end`` is inclusive; ``end`` past the cutoff raises). ``synthetic=True`` always returns
    the deterministic synthetic series (GBM, 16% vol, seed :data:`SYNTHETIC_SEED`); without
    it a missing file falls back to that series with a warning (build.py refuses to commit
    such a render unless the figure declared ``needs=("data_store",)``). Returns a copy.
    """
    _guard(f"bars({tf!r})")
    if is_synthetic(tf, synthetic=synthetic):
        kind = "synthetic" if (synthetic or force_synthetic()) else "fallback"
        if kind == "fallback":
            logger.warning("data_store/%s missing: using the synthetic fallback for bars(%r)",
                           _check_tf(tf)[0], tf)
        df = _synthetic_bars(tf)
    else:
        kind = "real"
        df = _real_bars(tf)
    _USAGE.add(f"bars:{tf}:{kind}")
    if start is not None:
        df = df.loc[df.index >= _utc(start)]
    if end is not None:
        df = df.loc[df.index <= _end_bound(end)]
    out = df.copy()
    out.attrs = dict(df.attrs)
    out.attrs["figure_source"] = "real" if kind == "real" else "synthetic"
    return out


# ---------------------------------------------------------------------------------------
# macro and events
# ---------------------------------------------------------------------------------------
@functools.cache
def _real_macro() -> dict[str, pd.DataFrame]:
    from aurum.data.macro import load_macro_dir

    return {k: cut_at_cutoff(v) for k, v in load_macro_dir(MACRO_DIR).items()}


@functools.cache
def _synthetic_macro(tf: str) -> dict[str, pd.DataFrame]:
    from aurum.data.synthetic import make_synthetic_macro

    return {k: cut_at_cutoff(v) for k, v in make_synthetic_macro(_synthetic_bars(tf), seed=SYNTHETIC_SEED).items()}


def macro_available() -> bool:
    return MACRO_DIR.is_dir() and any(MACRO_DIR.glob("*.parquet"))


def macro(names: str | Iterable[str] | None = None, *, synthetic: bool = False,
          tf: str = "H1") -> dict[str, pd.DataFrame]:
    """Daily macro frames ``{name: frame}`` (``value`` [+ OHLC] and ``available_at``).

    Real series come from ``data_store/macro`` via :func:`aurum.data.macro.load_macro_dir`;
    rows are kept only when ``available_at <= CUTOFF_END``, i.e. with the publication lag
    aurum applies (FRED: 1 US business day + 21:30, Yahoo: 21:30-22:30 after the date).
    Consumers must still align them on each bar's ``available_at`` with
    :func:`aurum.data.pit.asof_join`. The synthetic fallback (``dxy, spx, vix, us10y,
    real10y`` only) is built from the synthetic ``tf`` bars. Returns copies.
    """
    _guard("macro()")
    use_syn = synthetic or force_synthetic() or not macro_available()
    if use_syn:
        kind = "synthetic" if (synthetic or force_synthetic()) else "fallback"
        if kind == "fallback":
            logger.warning("data_store/macro missing: using the synthetic macro fallback")
        frames = _synthetic_macro(tf)
    else:
        kind = "real"
        frames = _real_macro()
    _USAGE.add(f"macro:{kind}")
    if names is not None:
        wanted = [names] if isinstance(names, str) else list(names)
        missing = [n for n in wanted if n not in frames]
        if missing:
            raise KeyError(f"macro series {missing} not available ({kind}); have {sorted(frames)}")
        frames = {n: frames[n] for n in wanted}
    out = {}
    for k, v in frames.items():
        c = v.copy()
        c.attrs = dict(v.attrs)
        out[k] = c
    return out


_macro_frames = macro  # market_data() has a ``macro`` flag that shadows the function


def events_for(bars_frame: pd.DataFrame, *, synthetic: bool | None = None) -> pd.DataFrame:
    """The event calendar :meth:`DataConfig.load` would attach to ``bars_frame``: the
    rule-based USD calendar (NFP, FOMC) from the first bar's day to 14 days after the last
    bar (scheduled times are public in advance), or the synthetic calendar for synthetic
    bars."""
    _guard("events")
    syn = bars_frame.attrs.get("figure_source") == "synthetic" if synthetic is None else synthetic
    if syn:
        from aurum.data.synthetic import make_synthetic_events

        return make_synthetic_events(bars_frame.index[0], bars_frame.index[-1] + pd.Timedelta(days=7))
    from aurum.data.calendar import generate_rule_based_calendar

    return generate_rule_based_calendar(bars_frame.index[0].normalize(),
                                        bars_frame.index[-1] + pd.Timedelta(days=14))


def market_data(tf: str = "H1", *, macro: bool = True, events: bool = True, start: Any = None,
                end: Any = None, synthetic: bool = False) -> Any:
    """:class:`aurum.core.types.MarketData` for ``tf``, assembled like
    :meth:`aurum.core.config.DataConfig.load` (bars + macro + rule-based events), cut at the
    figure cutoff. Feed it to features, strategies and models exactly as research does."""
    from aurum.core.types import MarketData

    b = bars(tf, start=start, end=end, synthetic=synthetic)
    syn = b.attrs.get("figure_source") == "synthetic"
    m = _macro_frames(synthetic=syn, tf=tf) if macro else {}
    ev = events_for(b, synthetic=syn) if events else None
    return MarketData(bars=b, macro=m, events=ev)


# ---------------------------------------------------------------------------------------
# source labels (captions and in-figure notes)
# ---------------------------------------------------------------------------------------
def _years(a: pd.Timestamp, b: pd.Timestamp) -> str:
    return f"{a.year}-{b.year}" if a.year != b.year else f"{a.year}"


def source_label(tf: str = "H1", *, start: Any = None, end: Any = None, synthetic: bool = False,
                 years: bool = True) -> str:
    """The data credit for a figure drawn from :func:`bars` with the same arguments, e.g.
    ``"XAUUSD H1, Dukascopy bid/ask mid, 2012-2024"`` or, for the synthetic series,
    ``"synthetic GBM H1 bars (aurum.data.synthetic, seed 7), not market data"``.
    ``years=False`` gives full dates (``2012-01-01 to 2024-12-31``)."""
    _, timeframe, _ = _check_tf(tf)
    if is_synthetic(tf, synthetic=synthetic):
        return (f"synthetic GBM {timeframe} bars (aurum.data.synthetic, seed {SYNTHETIC_SEED}), "
                "not market data")
    b = bars(tf, start=start, end=end)
    a, z = b.index[0], b.index[-1]
    period = _years(a, z) if years else f"{a.date()} to {z.date()}"
    return f"XAUUSD {_TF_DESCRIPTION[tf]}, Dukascopy bid/ask mid, {period}"


def macro_label(names: Iterable[str] | None = None) -> str:
    """Credit for real macro series, e.g. ``"FRED DFII10, Yahoo DX-Y.NYB (publication-lagged)"``."""
    if force_synthetic() or not macro_available():
        return f"synthetic macro series (aurum.data.synthetic, seed {SYNTHETIC_SEED}), not market data"
    frames = macro(names)
    fred = [v.attrs.get("source", k).split(":", 1)[-1] for k, v in frames.items()
            if str(v.attrs.get("source", "")).startswith("fred:")]
    yahoo = [v.attrs.get("source", k).split(":", 1)[-1] for k, v in frames.items()
             if str(v.attrs.get("source", "")).startswith("yahoo:")]
    parts = ([f"FRED {', '.join(fred)}"] if fred else []) + ([f"Yahoo {', '.join(yahoo)}"] if yahoo else [])
    return "; ".join(parts) + " (publication-lagged)"


def run_label(name: str) -> str:
    """Credit for a run directory, e.g. ``"runs/protocol/trend_core_selection, walk-forward
    OOS 2015-12-16 to 2024-12-31 (18 folds)"`` (from its ``summary.json``)."""
    import json

    d = run_dir(name)
    if d is None:
        return f"runs/{name}"
    p = d / "summary.json"
    if not p.exists():
        return f"runs/{name}"
    s = json.loads(p.read_text(encoding="utf-8"))
    a, z = str(s.get("oos_start", ""))[:10], str(s.get("oos_end", ""))[:10]
    if z and pd.Timestamp(z) >= pd.Timestamp(FIGURE_CUTOFF) + pd.Timedelta(days=1):
        z = FIGURE_CUTOFF
    folds = f" ({s['n_folds']} folds)" if s.get("n_folds") else ""
    return f"runs/{name}, walk-forward OOS {a} to {z}{folds}"


# ---------------------------------------------------------------------------------------
# run outputs (gitignored: always optional)
# ---------------------------------------------------------------------------------------
def _check_run_rel(rel: str) -> str:
    parts = Path(rel).parts
    if any(p == "holdout" for p in parts):
        raise FigureDataError(f"{rel!r}: figures must not read holdout outputs (data after {FIGURE_CUTOFF})")
    if any(p == ".." for p in parts) or Path(rel).is_absolute():
        raise ValueError(f"{rel!r}: give a path relative to runs/")
    return rel


def run_dir(name: str) -> Path | None:
    """``runs/<name>`` (e.g. ``"protocol/trend_core_selection"``) or ``None`` if it is missing.
    Declare it in ``needs=("runs/<name>",)`` so build.py skips the figure on a fresh clone."""
    _guard(f"run_dir({name!r})")
    d = RUNS_DIR / _check_run_rel(name)
    if not d.is_dir():
        return None
    _USAGE.add(f"run:{name}")
    return d


def run_file(name: str, rel: str) -> Path | None:
    """``runs/<name>/<rel>`` or ``None`` when missing. Paths under ``holdout/`` are refused."""
    rel = _check_run_rel(rel)          # refuse holdout paths even when the run is absent
    d = run_dir(name)
    if d is None:
        return None
    p = d / rel
    return p if p.exists() else None


def run_json(name: str, rel: str) -> Any:
    """Parse ``runs/<name>/<rel>`` as JSON (``summary.json``, ``stress.json``, ...) or return
    ``None``. Values are returned as stored: check dates against :data:`FIGURE_CUTOFF` yourself."""
    import json

    p = run_file(name, rel)
    return None if p is None else json.loads(p.read_text(encoding="utf-8"))


def run_table(name: str, rel: str, *, cut: bool = True, **read_kw: Any) -> pd.DataFrame | None:
    """Read ``runs/<name>/<rel>`` (``.parquet`` or ``.csv``) or return ``None``.

    CSVs whose first column parses as datetimes get it as a UTC index. With ``cut`` (default)
    a datetime index is cut to bar-open times before the end of :data:`FIGURE_CUTOFF`.
    """
    p = run_file(name, rel)
    if p is None:
        return None
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(p, **read_kw)
    elif suffix == ".csv":
        df = pd.read_csv(p, **read_kw)
        if "index_col" not in read_kw and len(df.columns):
            first = df.columns[0]
            col = df[first]
            if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
                try:
                    idx = pd.to_datetime(df[first], utc=True, format="ISO8601")
                except (ValueError, TypeError):
                    idx = None
                if idx is not None:
                    df = df.drop(columns=[first])
                    df.index = pd.DatetimeIndex(idx, name=first)
    else:
        raise ValueError(f"run_table reads .parquet/.csv (use run_json for .json), not {p.name}")
    return cut_at_cutoff(df, on="index") if cut else df


# ---------------------------------------------------------------------------------------
# layout helpers (call inside draw(pal); they rely on aurum.viz.style's active rcParams)
# ---------------------------------------------------------------------------------------
def panels(n: int, *, heights: Sequence[float] | None = None, figsize: tuple[float, float] | None = None,
           sharex: bool = True, hspace: float | None = None) -> tuple[Any, list[Any]]:
    """``n`` stacked panels sharing x (never a twin y-axis): returns ``(fig, [ax0, ...])``.

    ``heights`` are relative panel heights (e.g. ``(3, 1, 1)``). Uses a pyplot-free
    ``Figure`` like :func:`aurum.viz.style.figure`. Inner panels hide their x tick labels.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize or (FIGSIZE[0], 1.6 + 1.7 * n))
    gs = {"height_ratios": list(heights)} if heights else None
    axes = fig.subplots(n, 1, sharex=sharex, squeeze=False, gridspec_kw=gs)[:, 0]
    if hspace is not None:
        fig.get_layout_engine().set(hspace=hspace)
    if sharex:
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
    return fig, list(axes)


def source_note(fig: Any, text: str, pal: dict[str, Any] | None = None) -> Any:
    """A small muted credit line at the bottom-right of the figure (``"Data: ..."``).

    Implemented with ``fig.supxlabel`` so constrained layout reserves room for it; label the
    bottom axis with ``ax.set_xlabel`` rather than ``fig.supxlabel`` in figures that use it.
    """
    from aurum.viz.style import get_palette

    color = (pal or {}).get("text_muted") or get_palette().text_muted
    return fig.supxlabel(text, x=0.99, ha="right", fontsize=7.5, color=color, fontweight="normal")


def ink_on(fill: Any) -> str:
    """The primary text colour (light-mode or dark-mode ink) that stays readable on a cell
    filled with ``fill`` (any matplotlib colour, e.g. ``im.cmap(im.norm(v))``). Use it for
    numbers written inside heatmap cells or bars, in both modes."""
    from matplotlib.colors import to_rgb

    from aurum.viz.style import TEXT_PRIMARY

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in to_rgb(fill))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return TEXT_PRIMARY["light"] if luminance > 0.179 else TEXT_PRIMARY["dark"]


def signed(x: float, fmt: str = "{:+.2f}") -> str:
    """Format a number with a true minus sign (U+2212), matching matplotlib's tick labels."""
    return fmt.format(x).replace("-", "\N{MINUS SIGN}")


# ---------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class FigureSpec:
    """One registered documentation figure."""

    page: str
    name: str
    draw: Callable[..., Any] = field(compare=False, repr=False)
    needs: tuple[str, ...] = ()
    module: str = ""

    @property
    def key(self) -> str:
        return f"{self.page}/{self.name}"

    @property
    def doc(self) -> str:
        return ((self.draw.__doc__ or "").strip().splitlines() or [""])[0]

    def paths(self, root: Path | None = None) -> tuple[Path, Path]:
        base = (root or FIGURES_DIR) / self.page
        return base / f"{self.name}-light.png", base / f"{self.name}-dark.png"

    def missing_needs(self) -> list[str]:
        """``needs`` entries that do not exist (relative to the repo root)."""
        out = []
        for need in self.needs:
            p = REPO / need
            if need.rstrip("/") == "data_store":
                ok = any(p.glob("xauusd_*.parquet")) if p.is_dir() else False
            else:
                ok = p.exists()
            if not ok:
                out.append(need)
        return out


REGISTRY: dict[str, FigureSpec] = {}


def register(page: str, name: str, *, needs: Sequence[str] = ()) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``draw(pal) -> Figure`` as ``docs/_assets/figures/<page>/<name>-{light,dark}.png``.

    ``page`` and ``name`` are lowercase slugs (``a-z``, ``0-9``, ``-``). ``needs`` lists repo
    paths the figure cannot be drawn without: ``"data_store"`` for real bars/macro,
    ``"runs/protocol/trend_core_selection"`` for a run. When one is missing, build.py skips
    the figure and keeps the committed PNGs. The function is returned unchanged.
    """
    if not _SLUG.match(page):
        raise ValueError(f"page {page!r} must be a lowercase slug like 'ml-and-rl'")
    if not _SLUG.match(name):
        raise ValueError(f"figure name {name!r} must be a lowercase slug like 'managed-gold-latch'")
    needs_t = tuple(str(n).rstrip("/") for n in needs)
    for n in needs_t:
        if Path(n).is_absolute() or ".." in Path(n).parts:
            raise ValueError(f"needs entry {n!r} must be a path relative to the repository root")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = f"{page}/{name}"
        module = getattr(fn, "__module__", "")
        old = REGISTRY.get(key)
        if old is not None and (old.module, old.draw.__qualname__) != (module, fn.__qualname__):
            raise ValueError(f"figure {key} registered twice ({old.module}.{old.draw.__qualname__} "
                             f"and {module}.{fn.__qualname__})")
        REGISTRY[key] = FigureSpec(page=page, name=name, draw=fn, needs=needs_t, module=module)
        return fn

    return deco


# ---------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------
@dataclass
class RenderResult:
    spec: FigureSpec
    status: str                     # "updated" | "unchanged" | "skipped" | "error"
    seconds: float = 0.0
    message: str = ""
    paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _seeded(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any) -> Any:
        random.seed(SYNTHETIC_SEED)
        np.random.seed(SYNTHETIC_SEED)
        return fn(*args)

    return wrapper


def render(spec: FigureSpec, *, out_root: Path | None = None, preview: bool = False) -> RenderResult:
    """Render one figure in both modes and install the PNGs atomically.

    * Missing ``needs`` -> ``skipped`` (committed PNGs untouched) unless ``preview``.
    * Byte-identical output -> ``unchanged`` (the file is not rewritten).
    * A figure that silently fell back to synthetic bars without declaring
      ``needs=("data_store",)`` -> ``error`` and nothing is installed.
    """
    from aurum.viz.style import save_figure

    root = Path(out_root) if out_root is not None else FIGURES_DIR
    missing = spec.missing_needs()
    if missing and not (preview and force_synthetic() and all(m.startswith("data_store") for m in missing)):
        light, dark = spec.paths(root)
        have = "kept" if light.exists() and dark.exists() else "NOT PRESENT - render it on a machine with the data"
        return RenderResult(spec, "skipped", message=f"needs {', '.join(missing)} (missing); committed PNGs {have}")
    reset_usage()
    t0 = time.perf_counter()
    notes: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="aurum-fig-") as tmp:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                written = save_figure(_seeded(spec.draw), Path(tmp) / spec.name, dpi=DPI)
            for w in caught:
                msg = f"{w.category.__name__}: {w.message}"
                if msg not in notes:
                    notes.append(msg)
            used = usage()
            fallback = sorted(u for u in used if u.endswith(":fallback"))
            if fallback and not preview:
                return RenderResult(spec, "error", time.perf_counter() - t0, notes=notes, message=(
                    f"used the synthetic fallback ({', '.join(fallback)}) because data_store/ is missing: "
                    "declare needs=('data_store',) or ask for bars(..., synthetic=True) explicitly"))
            if "data_store" not in spec.needs and any(u.endswith(":real") for u in used):
                notes.append("reads data_store/ but does not declare needs=('data_store',): "
                             "add it so a fresh clone skips instead of falling back")
            for u in sorted(used):
                if u.startswith("run:") and f"runs/{u[4:]}" not in spec.needs:
                    notes.append(f"reads runs/{u[4:]} but does not declare it in needs")
            dest_dir = root / spec.page
            dest_dir.mkdir(parents=True, exist_ok=True)
            changed = False
            installed = []
            for src in written:
                dest = dest_dir / src.name
                data = src.read_bytes()
                if dest.exists() and dest.read_bytes() == data:
                    installed.append(dest)
                    continue
                part = dest.with_name(dest.name + ".part")
                part.write_bytes(data)
                os.replace(part, dest)
                installed.append(dest)
                changed = True
    except Exception as exc:  # noqa: BLE001 - report and continue with the other figures
        import traceback

        tb = traceback.format_exc(limit=-4)
        return RenderResult(spec, "error", time.perf_counter() - t0, message=f"{type(exc).__name__}: {exc}\n{tb}",
                            notes=notes)
    secs = time.perf_counter() - t0
    if secs > MAX_SECONDS:
        notes.append(f"took {secs:.1f}s (budget {MAX_SECONDS:.0f}s): read run outputs instead of recomputing")
    return RenderResult(spec, "updated" if changed else "unchanged", secs, paths=installed, notes=notes)


__all__ = [
    "CUTOFF_END",
    "DATA_DIR",
    "DOCS",
    "DPI",
    "FIGSIZE",
    "FIGSIZE_GRID",
    "FIGSIZE_SHORT",
    "FIGSIZE_SQUARE",
    "FIGSIZE_TALL",
    "FIGURES_DIR",
    "FIGURE_CUTOFF",
    "FigureDataError",
    "FigureSpec",
    "MAX_SECONDS",
    "REGISTRY",
    "REPO",
    "RUNS_DIR",
    "SYNTHETIC_SEED",
    "TIMEFRAMES",
    "bars",
    "cut_at_cutoff",
    "data_available",
    "events_for",
    "ink_on",
    "is_synthetic",
    "macro",
    "macro_available",
    "macro_label",
    "market_data",
    "panels",
    "register",
    "render",
    "run_dir",
    "run_file",
    "run_json",
    "run_label",
    "run_table",
    "signed",
    "source_label",
    "source_note",
]
