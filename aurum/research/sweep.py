"""Parameter-robustness sweeps: is the best parameter set a plateau or a knife edge?

A sweep evaluates ONE strategy at every point of a parameter grid with the full
walk-forward protocol (:func:`aurum.research.walkforward.run_walk_forward`), so every
number in the table is out of sample, net of costs, through the same sizer and research
risk limits as any other run. It then asks three questions about the best point:

1. **Is it significant after the search?** The Deflated Sharpe Ratio of the best point
   uses ``n_trials`` = the number of grid points evaluated (+ distinct configurations of
   earlier sweeps over an overlapping window, from the sweep ledger), because picking the
   best of ``N`` noisy configurations inflates its Sharpe (Bailey & López de Prado 2014).
   ``dsr_xs`` repeats the test with the grid's cross-sectional Sharpe variance, the
   paper's parameter-sweep setting.
2. **Is it a plateau?** The *plateau ratio* compares the metric averaged over the best
   point's grid neighbours (Chebyshev distance 1 in grid-index space: the up-to-``3^d - 1``
   surrounding cells) with the best point itself::

       plateau_ratio = mean(metric[neighbours]) / metric[best]      (metric[best] > 0)

   ``1`` means the neighbours are as good as the optimum (a robust plateau); a ratio below
   ``knife_edge_ratio`` (default 0.5) flags a **knife-edge** optimum: a small change of any
   parameter loses most of the performance, the classic signature of a fitted-to-noise
   peak. For metrics where lower is better (costs, turnover) or that are negative at best
   (max drawdown) the ratio is ``metric[best] / mean(metric[neighbours])`` so that it reads
   the same way. The *plateau centre* (``robust_best``) is the point whose own-plus-neighbour
   mean is highest: the parameter set to prefer when you must pick one.
3. **Does selection work at all?** PBO by CSCV over the grid points' daily OOS returns
   (Bailey, Borwein, López de Prado & Zhu 2017): how often the in-sample best grid point
   is below the median out of sample.

Each grid point is evaluated as a **standalone book**: a single-strategy configuration
with ``combiner.method: fixed`` and weight 1, whose forecast diversification multiplier is
1, so the combined forecast equals the strategy forecast exactly and only one backtest per
point is needed. Points whose parameters the strategy rejects (e.g. ``fast >= slow``) are
recorded with ``status="invalid"`` and are not trials.

The holdout is never touched
----------------------------
* Bars from ``walkforward.holdout_start`` on are physically removed before anything runs,
  and every point configuration has ``holdout_start: null``, so no holdout evaluation and
  no holdout-ledger entry can happen.
* ``end=`` (inclusive; a date-only value covers the whole day) truncates the data further.
* Holdout ledgers (``holdout_ledger.jsonl`` in ``output.dir`` and its immediate
  subdirectories, e.g. ``runs/protocol/``) are read: if the data reach a holdout window
  that was already evaluated for the same symbol, the data are truncated before the
  earliest such window (``respect_holdout_ledgers=False`` disables this; never do that for
  real research). Every truncation is listed in ``SweepResult.notes``.

The sweep ledger
----------------
``sweep_ledger.jsonl`` (in the parent of the output directory, next to the run
directories) records every sweep: strategy, window, metric and the config hash of every
point. A later sweep over an overlapping window adds the DISTINCT point configurations of
earlier sweeps it did not itself evaluate to ``n_trials``: re-sweeping the same data with a
slightly different grid is still searching.
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import json
import logging
import math
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.metrics import daily_returns
from aurum.core.config import AurumConfig, ConfigError, config_from_dict
from aurum.core.types import MarketData

logger = logging.getLogger(__name__)

__all__ = [
    "SWEEP_LEDGER",
    "SweepResult",
    "grid_points",
    "parse_grid",
    "plateau_score",
    "plot_sweep",
    "point_label",
    "read_sweep_ledger",
    "run_sweep",
]

#: sweep ledger file name (kept in the output root, next to the run directories)
SWEEP_LEDGER = "sweep_ledger.jsonl"
#: metrics for which a LOWER value is better (everything else is maximised)
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "total_costs", "cost_drag_ann", "turnover_lots_per_year", "n_trades", "ann_vol", "min_trl_years",
    "sharpe_se", "n_halt_episodes", "n_risk_events",
})
#: maximised metrics that are <= 0 by construction (closer to 0 is better)
NONPOSITIVE_METRICS: frozenset[str] = frozenset({"max_drawdown"})
#: statistics copied from the walk-forward stats row of every grid point
POINT_COLUMNS: tuple[str, ...] = (
    "sharpe", "ci_lower", "ci_upper", "psr", "cagr", "ann_vol", "max_drawdown", "calmar", "sortino",
    "n_trades", "turnover_lots_per_year", "total_costs", "cost_drag_ann", "exposure", "n_days",
    "n_halt_episodes",
)
_PROCESS_MIN_BARS = 20_000
_BOOK = "combined"      # the standalone book of a single-strategy fixed-weight configuration


# ============================================================================================
# grid helpers
# ============================================================================================
def _parse_value(raw: str) -> Any:
    import yaml

    raw = raw.strip()
    try:
        return yaml.safe_load(raw) if raw else None
    except yaml.YAMLError:
        return raw


def parse_grid(specs: Sequence[str] | Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
    """``["fast=16,32,64", "slow=64,128"]`` (CLI form) or a mapping -> ``{param: [values]}``.

    Values are YAML-parsed (``16`` -> int, ``0.5`` -> float, ``true`` -> bool, otherwise the
    string). A range ``a:b:step`` (inclusive of ``b`` when it falls on the step) is
    expanded, e.g. ``fast=10:40:10`` -> ``[10, 20, 30, 40]``. Numeric lists are sorted and
    de-duplicated so that grid neighbours are value neighbours.
    """
    out: dict[str, list[Any]] = {}
    if isinstance(specs, Mapping):
        items = [(str(k), list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Index)) else [v])
                 for k, v in specs.items()]
    else:
        items = []
        for spec in specs:
            if "=" not in spec:
                raise ConfigError(f"grid spec {spec!r} must look like param=v1,v2,...")
            key, vals = spec.split("=", 1)
            key = key.strip()
            if not key:
                raise ConfigError(f"grid spec {spec!r} has an empty parameter name")
            parts = [v for v in vals.split(",") if v.strip()]
            if len(parts) == 1 and parts[0].count(":") == 2:
                a, b, s = (_parse_value(x) for x in parts[0].split(":"))
                if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b, s)) or s <= 0:
                    raise ConfigError(f"grid range {parts[0]!r} must be numeric start:stop:step with step > 0")
                n = int(math.floor((b - a) / s + 1e-9)) + 1
                values = [a + i * s for i in range(max(n, 0))]
                if all(isinstance(x, int) for x in (a, b, s)):
                    values = [int(v) for v in values]
            else:
                values = [_parse_value(v) for v in parts]
            if key in dict(items):
                raise ConfigError(f"grid parameter {key!r} given twice")
            items.append((key, values))
    for key, values in items:
        if not values:
            raise ConfigError(f"grid parameter {key!r} has no values")
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        if numeric:
            values = sorted(set(values))
        else:
            seen: list[Any] = []
            for v in values:
                if v not in seen:
                    seen.append(v)
            values = seen
        out[key] = values
    if not out:
        raise ConfigError("the grid is empty (pass at least one --grid param=v1,v2,...)")
    return out


def grid_points(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid, in row-major order of the parameters as given."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def point_label(params: Mapping[str, Any]) -> str:
    """Stable human-readable id of a grid point, e.g. ``fast=16,slow=64``."""
    return ",".join(f"{k}={v}" for k, v in params.items())


def _better(metric: str, maximize: bool | None) -> bool:
    return (metric not in LOWER_IS_BETTER) if maximize is None else bool(maximize)


# ============================================================================================
# plateau / robustness
# ============================================================================================
def _vkey(v: Any) -> tuple[str, Any]:
    """Hashable comparison key: numeric values compare by float (16 == np.int64(16) == 16.0)."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
        return ("n", float(v))
    return ("o", repr(v))


def _grid_positions(table: pd.DataFrame, grid: Mapping[str, Sequence[Any]]) -> np.ndarray:
    pos = np.empty((len(table), len(grid)), dtype=float)
    for j, (k, values) in enumerate(grid.items()):
        lookup = {_vkey(v): i for i, v in enumerate(values)}
        pos[:, j] = [lookup.get(_vkey(v), np.nan) for v in table[k]]
    return pos


def _ratio(best: float, neigh: float, maximize: bool, metric: str) -> float:
    """``neighbour / best`` for maximised signed metrics with a positive optimum (0 = no edge);
    ``best / neighbour`` for minimised positive metrics and naturally non-positive ones
    (:data:`NONPOSITIVE_METRICS`); NaN otherwise (e.g. the best Sharpe is not positive)."""
    if not (math.isfinite(best) and math.isfinite(neigh)):
        return math.nan
    if metric in NONPOSITIVE_METRICS:
        return best / neigh if best < 0 and neigh < 0 else (1.0 if best == neigh == 0 else math.nan)
    if maximize and best > 0:
        return neigh / best
    if not maximize and best > 0 and neigh > 0:
        return best / neigh
    return math.nan


def plateau_score(table: pd.DataFrame, grid: Mapping[str, Sequence[Any]], metric: str = "sharpe", *,
                  maximize: bool | None = None, knife_edge_ratio: float = 0.5,
                  plateau_ratio: float = 0.75) -> dict[str, Any]:
    """Robustness of the best grid point (module docstring, question 2).

    ``table`` has one row per grid point with the parameter columns of ``grid``, ``metric``
    and optionally ``status`` (only ``"ok"`` rows with a finite metric count). Neighbours are
    the valid points at Chebyshev distance 1 in grid-index space. Returns ``best`` /
    ``robust_best`` (labels and params), ``best_value``, ``neighbour_mean``/``min``/``max``,
    ``n_neighbours``, ``plateau_ratio``, ``drop_sd`` (best minus neighbour mean, in units of
    the cross-sectional std of the metric), ``frac_neighbours_positive``, ``edge_params``
    (parameters whose best value is at the end of their grid), ``knife_edge``
    (``plateau_ratio < knife_edge_ratio``; ``None`` when the ratio is undefined, e.g. the
    best Sharpe is not positive) and a ``verdict``: ``"robust plateau"``
    (ratio >= ``plateau_ratio``), ``"moderate"``, ``"knife-edge"`` or ``"undefined"``.
    """
    mx = _better(metric, maximize)
    t = table.reset_index(drop=True)
    vals = pd.to_numeric(t[metric], errors="coerce").to_numpy(float) if metric in t else np.full(len(t), np.nan)
    ok = np.isfinite(vals)
    if "status" in t:
        ok &= (t["status"] == "ok").to_numpy()
    out: dict[str, Any] = {"metric": metric, "maximize": mx, "knife_edge_ratio": float(knife_edge_ratio),
                           "n_valid": int(ok.sum())}
    if not ok.any():
        out.update({"best": None, "verdict": "undefined", "knife_edge": None, "plateau_ratio": math.nan})
        return out
    score = np.where(ok, vals if mx else -vals, -np.inf)
    b = int(np.argmax(score))
    pos = _grid_positions(t, grid)
    cheb = np.nanmax(np.abs(pos - pos[b]), axis=1)
    neigh = ok & (cheb == 1)
    nv = vals[neigh]
    nmean = float(nv.mean()) if nv.size else math.nan
    ratio = _ratio(float(vals[b]), nmean, mx, metric)
    sd = float(np.std(vals[ok], ddof=1)) if ok.sum() >= 3 else math.nan
    s_best = vals[b] if mx else -vals[b]
    s_nm = nmean if mx else -nmean
    drop_sd = float((s_best - s_nm) / sd) if sd and math.isfinite(sd) and sd > 0 and math.isfinite(nmean) else math.nan
    # plateau centre: highest mean over itself + its valid neighbours
    smooth = np.full(len(t), -np.inf)
    for i in np.flatnonzero(ok):
        c = np.nanmax(np.abs(pos - pos[i]), axis=1)
        m = ok & (c <= 1)
        sv = vals[m] if mx else -vals[m]
        smooth[i] = float(sv.mean())
    r = int(np.argmax(smooth))
    params = list(grid)

    def _pt(i: int) -> dict[str, Any]:
        return {k: _py(t.loc[i, k]) for k in params}

    # parameters whose best value is the first/last of their (>= 2-value) grid: the optimum may
    # lie outside the searched range
    edge = [k for j, k in enumerate(params) if len(grid[k]) >= 2 and pos[b, j] in (0, len(grid[k]) - 1)]
    knife = None if not math.isfinite(ratio) else bool(ratio < knife_edge_ratio)
    if knife is None:
        verdict = "undefined"
    elif knife:
        verdict = "knife-edge"
    elif ratio >= plateau_ratio:
        verdict = "robust plateau"
    else:
        verdict = "moderate"
    out.update({
        "best": point_label(_pt(b)), "best_params": _pt(b), "best_value": float(vals[b]),
        "n_neighbours": int(neigh.sum()), "neighbour_mean": nmean,
        "neighbour_min": float(nv.min()) if nv.size else math.nan,
        "neighbour_max": float(nv.max()) if nv.size else math.nan,
        "plateau_ratio": ratio, "drop_sd": drop_sd,
        "frac_neighbours_positive": float((nv > 0).mean()) if nv.size else math.nan,
        "knife_edge": knife, "verdict": verdict, "edge_params": edge,
        "robust_best": point_label(_pt(r)), "robust_best_params": _pt(r),
        "robust_best_value": float(vals[r]),
        "robust_best_smoothed": float(smooth[r] if mx else -smooth[r]),
    })
    return out


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


# ============================================================================================
# data window (never the holdout)
# ============================================================================================
def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _end_ts(end: Any) -> pd.Timestamp:
    e = _utc(end)
    if isinstance(end, str) and len(end.strip()) <= 10:   # date-only: the whole day
        e = e + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return e


def find_holdout_ledgers(output_dir: str | Path) -> list[Path]:
    """``holdout_ledger.jsonl`` in ``output_dir`` and its immediate subdirectories."""
    from aurum.research.walkforward import HOLDOUT_LEDGER

    root = Path(output_dir)
    found = [root / HOLDOUT_LEDGER] if (root / HOLDOUT_LEDGER).exists() else []
    if root.is_dir():
        found += sorted(p for p in root.glob(f"*/{HOLDOUT_LEDGER}") if p.is_file())
    return found


def research_window(md: MarketData, cfg: AurumConfig, *, start: Any = None, end: Any = None,
                    holdout_ledgers: Sequence[str | Path] | None = None,
                    respect_holdout_ledgers: bool = True) -> tuple[MarketData, list[str]]:
    """Restrict ``md`` to research bars only (module docstring, "The holdout is never touched").

    Returns the truncated market data and notes describing every cut. Macro frames and the
    calendar are left whole (consumers align them point-in-time on ``available_at``).
    """
    from aurum.research.walkforward import read_holdout_ledger

    bars = md.bars
    notes: list[str] = []
    if start is not None:
        bars = bars.loc[bars.index >= _utc(start)]
    hs = cfg.walkforward.holdout_start
    if hs is not None:
        keep = bars.index < _utc(hs)
        if not keep.all():
            notes.append(f"bars from walkforward.holdout_start ({hs}) excluded: the holdout is never swept")
        bars = bars.loc[keep]
    if end is not None:
        e = _end_ts(end)
        if len(bars) and bars.index[-1] > e:
            notes.append(f"data truncated at end={end}")
        bars = bars.loc[bars.index <= e]
    if respect_holdout_ledgers and len(bars):
        ledgers = find_holdout_ledgers(cfg.output.dir) if holdout_ledgers is None else [Path(p) for p in
                                                                                         holdout_ledgers]
        spent: list[tuple[pd.Timestamp, Path]] = []
        for lp in ledgers:
            for e in read_holdout_ledger(lp):
                sym = e.get("symbol")
                if sym is not None and sym != cfg.data.symbol:
                    continue
                try:
                    hs0 = _utc(e["holdout_start"])
                    he0 = _utc(e.get("holdout_end") or e["holdout_start"])
                except (KeyError, TypeError, ValueError):
                    continue
                if he0 >= bars.index[0]:
                    spent.append((hs0, lp))
        if spent:
            cut, lp = min(spent, key=lambda x: x[0])
            if bars.index[-1] >= cut:
                bars = bars.loc[bars.index < cut]
                notes.append(f"data truncated before {cut}: that holdout window was already evaluated "
                             f"(holdout ledger {lp}); a sweep must not look at it")
    if len(bars) < 2:
        raise ConfigError("the sweep's research window holds fewer than two bars after excluding the holdout")
    return MarketData(bars=bars, macro=md.macro, events=md.events), notes


# ============================================================================================
# sweep ledger
# ============================================================================================
def read_sweep_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Entries of a sweep ledger (``[]`` if the file does not exist; bad lines skipped)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("%s:%d: unreadable sweep-ledger line skipped", p, i)
    return out


def _prior_sweep_trials(entries: Sequence[Mapping[str, Any]], *, start: pd.Timestamp, end: pd.Timestamp,
                        symbol: str, own: set[str]) -> dict[str, Any]:
    hashes: set[str] = set()
    n = 0
    for e in entries:
        if e.get("symbol") not in (None, symbol):
            continue
        try:
            s0, e0 = _utc(e["window_start"]), _utc(e["window_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if s0 <= end and e0 >= start:
            n += 1
            hashes.update(h for h in e.get("config_hashes", []) if h not in own)
    return {"n_prior_sweeps": n, "n_prior_configs": len(hashes)}


def _append_ledger(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ============================================================================================
# point evaluation (pool workers)
# ============================================================================================
_WORKER_MD: MarketData | None = None


def _init_worker(md: MarketData, log_level: int) -> None:  # pragma: no cover - subprocess
    global _WORKER_MD
    _WORKER_MD = md
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    logging.basicConfig(level=log_level, format="%(processName)s %(name)s %(levelname)s: %(message)s")


def _evaluate_point(label: str, cfg_dict: dict[str, Any], walkforward: bool, start: Any,
                    md: MarketData | None = None) -> dict[str, Any]:
    from aurum.research.walkforward import run_single_backtest, run_walk_forward

    md = md if md is not None else _WORKER_MD
    if md is None:  # pragma: no cover - defensive
        raise RuntimeError("sweep worker has no market data")
    t0 = time.perf_counter()
    cfg = config_from_dict(cfg_dict, env={})
    if walkforward:
        rep = run_walk_forward(md, cfg, write=False, holdout_ledger=False)
    else:
        rep = run_single_backtest(md, cfg, start=start, write=False)
    row = {k: _py(v) for k, v in rep.stats.loc[_BOOK].to_dict().items()}
    dr = daily_returns(rep.combined.equity).dropna()
    oos = rep.oos_forecasts.index
    return {"label": label, "row": row, "daily": dr, "n_folds": len(rep.folds),
            "oos_start": str(oos[0]) if len(oos) else None, "oos_end": str(oos[-1]) if len(oos) else None,
            "seconds": round(time.perf_counter() - t0, 3), "notes": list(rep.notes),
            "config_hash": cfg.config_hash()}


# ============================================================================================
# result
# ============================================================================================
@dataclass
class SweepResult:
    """What :func:`run_sweep` produced (see the module docstring for the statistics)."""

    strategy: str
    grid: dict[str, list[Any]]
    metric: str
    maximize: bool
    table: pd.DataFrame                     # one row per grid point (params, status, statistics)
    returns: pd.DataFrame                   # daily OOS returns, one column per valid point label
    best: dict[str, Any]                    # best point: label, params, metric, sharpe, DSR block
    plateau: dict[str, Any]
    n_trials: int
    prior: dict[str, Any]
    pbo: dict[str, Any] | None
    window: dict[str, Any]
    walkforward: bool
    notes: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    out_dir: Path | None = None

    @property
    def label(self) -> str:
        """``"exploratory"`` for every sweep: a sweep is a search, its best point is a trial."""
        return "exploratory"

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary (what ``sweep.json`` holds)."""
        return _jsonable({
            "kind": "sweep", "label": self.label, "strategy": self.strategy, "grid": self.grid,
            "metric": self.metric, "maximize": self.maximize, "walkforward": self.walkforward,
            "n_points": len(self.table), "n_valid": int((self.table["status"] == "ok").sum()),
            "n_trials": self.n_trials, "prior": self.prior, "best": self.best, "plateau": self.plateau,
            "pbo": self.pbo, "window": self.window, "notes": self.notes, "timing": self.timing,
            "settings": self.settings, "table": self.table.to_dict(orient="records"), "out_dir": self.out_dir,
        })

    def plot(self, *, x: str | None = None, y: str | None = None, metric: str | None = None) -> Any:
        """Heatmap (2-D grids) or line chart (1-D) of ``metric`` (default the sweep metric)."""
        m = metric or self.metric
        p = self.plateau
        sub = ""
        if m == self.metric and p.get("best"):
            ratio = p.get("plateau_ratio")
            rtxt = f"{ratio:.2f}" if isinstance(ratio, float) and math.isfinite(ratio) else "n/a"
            sub = (f"box = best ({p['best']}), circle = plateau centre ({p.get('robust_best')}); plateau ratio "
                   f"{rtxt}: {p.get('verdict')}; DSR {_num(self.best.get('dsr'))} at n_trials {self.n_trials}")
        mode = "walk-forward OOS" if self.walkforward else "single split"
        return plot_sweep(self.table, self.grid, m, maximize=_better(m, None) if m != self.metric else self.maximize,
                          x=x, y=y, best=p.get("best_params") if m == self.metric else None,
                          robust=p.get("robust_best_params") if m == self.metric else None,
                          title=f"{self.strategy} sweep, {m} ({mode}, exploratory)", subtitle=sub)

    def save(self, directory: str | Path) -> Path:
        """Write ``sweep.csv``, ``returns.csv``, ``sweep.json`` and ``sweep.png`` (+ ``sharpe.png``)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(d / "sweep.csv", index=False)
        self.returns.to_csv(d / "returns.csv")
        self.out_dir = d
        figs = {"sweep.png": self.plot()}
        if self.metric != "sharpe" and "sharpe" in self.table:
            figs["sharpe.png"] = self.plot(metric="sharpe")
        for name, fig in figs.items():
            if fig is not None:
                fig.savefig(d / name, dpi=110, facecolor=fig.get_facecolor())
        (d / "sweep.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return d


def _num(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.3f}" if math.isfinite(x) else "n/a"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ============================================================================================
# main entry point
# ============================================================================================
def _strategy_base(cfg: AurumConfig, name: str) -> tuple[str, dict[str, Any], str]:
    """(registry name, base params, column id) for ``name`` (a config id/name or a registry name)."""
    for s in cfg.strategies:
        if s.key == name or s.name == name:
            return s.name, dict(s.params), s.key
    return name, {}, name


def _point_config(cfg: AurumConfig, reg_name: str, sid: str, params: Mapping[str, Any]) -> dict[str, Any]:
    d = cfg.to_dict(redact=True)
    d.pop("source", None)
    d["strategies"] = [{"name": reg_name, "params": dict(params), "weight": 1.0, "id": sid, "enabled": True}]
    d["combiner"] = {**d.get("combiner", {}), "method": "fixed"}
    d["walkforward"] = {**d.get("walkforward", {}), "holdout_start": None, "n_trials": None,
                        "strategy_backtests": False, "executor": "serial", "n_jobs": 1}
    d["backtest"] = {**d.get("backtest", {}), "benchmark": False}
    d["output"] = {**d.get("output", {}), "save_results": False, "tearsheet": False}
    return d


def _resolve_pool(executor: str, n_jobs: int, n_bars: int, n_tasks: int) -> tuple[str, int]:
    if executor not in ("auto", "process", "thread", "serial"):
        raise ConfigError(f"executor must be auto|process|thread|serial, got {executor!r}")
    jobs = int(n_jobs) if n_jobs and n_jobs > 0 else min(os.cpu_count() or 1, 8)
    jobs = max(1, min(jobs, n_tasks))
    if executor == "auto":
        executor = "serial" if jobs == 1 else ("process" if n_bars >= _PROCESS_MIN_BARS else "thread")
    if executor == "serial":
        jobs = 1
    return executor, jobs


def run_sweep(md: MarketData, cfg: AurumConfig, strategy_name: str,
              grid: Mapping[str, Sequence[Any]] | Sequence[str], *, metric: str = "sharpe",
              maximize: bool | None = None, walkforward: bool = True, start: Any = None, end: Any = None,
              n_jobs: int = 0, executor: str = "auto", out_dir: str | Path | None = None,
              write: bool | None = None, ledger: str | Path | bool | None = None,
              holdout_ledgers: Sequence[str | Path] | None = None, respect_holdout_ledgers: bool = True,
              prior_trials: int = 0, knife_edge_ratio: float = 0.5, n_boot: int | None = None,
              seed: int | None = None) -> SweepResult:
    """Evaluate ``strategy_name`` at every point of ``grid`` (module docstring).

    Parameters
    ----------
    md, cfg       : market data and configuration; ``cfg`` supplies costs, sizing, research risk
                    limits, the walk-forward geometry and the base parameters of the strategy
                    when it appears in ``cfg.strategies`` (by id or name; otherwise registry
                    defaults).
    strategy_name : config strategy id/name or registry name.
    grid          : ``{param: [values]}`` or CLI specs ``["fast=16,32", "slow=64,128"]``.
    metric        : stats column that ranks the points (``sharpe``, ``cagr``, ``calmar``,
                    ``max_drawdown`` ...); ``maximize`` defaults to False only for
                    :data:`LOWER_IS_BETTER` columns.
    walkforward   : True (default): full walk-forward OOS per point. False: one
                    :func:`~aurum.research.walkforward.run_single_backtest` per point
                    (in-sample unless ``start`` is given; flagged in the notes).
    start, end    : research window (``end`` inclusive; with ``walkforward=False``, ``start`` is
                    the evaluation start of the single split instead). The holdout is always
                    excluded.
    n_jobs, executor : point-level parallelism (``0`` = ``min(cpu, 8)``; ``auto`` = processes
                    for >= 20k bars, threads otherwise). Each point runs its walk-forward
                    serially, so results do not depend on the executor.
    out_dir, write : output directory (default ``output.dir/sweep_<strategy>_<utc>``) and
                    whether to write it (default ``output.save_results``).
    ledger        : sweep ledger path; ``None`` = ``<out_dir>/../sweep_ledger.jsonl`` when
                    writing (not used otherwise); ``False`` = off.
    holdout_ledgers, respect_holdout_ledgers : see :func:`research_window`.
    prior_trials  : extra trials to add to ``n_trials`` (searches not in any ledger).
    knife_edge_ratio : plateau-ratio threshold of the knife-edge flag.
    n_boot, seed  : bootstrap settings of the best point's statistics (default
                    ``walkforward.n_boot`` / ``cfg.seed``).
    """
    t_all = time.perf_counter()
    g = parse_grid(grid)
    mx = _better(metric, maximize)
    reg_name, base, sid = _strategy_base(cfg, strategy_name)
    if metric not in POINT_COLUMNS:
        logger.info("sweep metric %r is not a standard column; it must exist in the walk-forward stats", metric)
    # walk-forward: ``start`` restricts the research window; single split: it is the evaluation
    # start and the strategy is fitted on the bars before it
    md_r, notes = research_window(md, cfg, start=start if walkforward else None, end=end,
                                  holdout_ledgers=holdout_ledgers, respect_holdout_ledgers=respect_holdout_ledgers)
    idx = md_r.bars.index
    window = {"start": str(idx[0]), "end": str(idx[-1]), "n_bars": len(idx),
              "symbol": cfg.data.symbol, "timeframe": cfg.data.timeframe}
    if not walkforward:
        notes.append("walkforward=False: each point is ONE train/test split"
                     + (" (IN-SAMPLE: no start given)" if start is None else f" evaluated from {start}"))

    # ---- build + validate point configurations (invalid parameters are not trials) --------
    points = grid_points(g)
    rows: list[dict[str, Any]] = []
    tasks: list[tuple[str, dict[str, Any]]] = []
    hashes: dict[str, str] = {}
    for p in points:
        label = point_label(p)
        params = {**base, **p}
        row: dict[str, Any] = {**p, "label": label, "status": "ok", "error": ""}
        try:
            pcfg_d = _point_config(cfg, reg_name, sid, params)
            pcfg = config_from_dict(pcfg_d, env={})
            pcfg.build_strategies()
            hashes[label] = pcfg.config_hash()
            tasks.append((label, pcfg_d))
        except (ConfigError, TypeError, ValueError) as exc:
            row.update(status="invalid", error=str(exc).splitlines()[0][:300])
        rows.append(row)
    if not tasks:
        raise ConfigError(f"no valid grid point for strategy {reg_name!r}: " + "; ".join(
            r["error"] for r in rows[:3]))

    # ---- evaluate ----------------------------------------------------------------------------
    kind, jobs = _resolve_pool(executor, n_jobs, len(idx), len(tasks))
    logger.info("sweep %s: %d points (%d valid), executor=%s(%d)", reg_name, len(points), len(tasks), kind, jobs)
    t0 = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    if kind == "serial":
        for label, d in tasks:
            try:
                results[label] = _evaluate_point(label, d, walkforward, start, md_r)
            except Exception as exc:  # noqa: BLE001 - one failed point must not kill the sweep
                errors[label] = f"{type(exc).__name__}: {exc}"
    else:
        if kind == "process":
            pool: cf.Executor = cf.ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker, initargs=(md_r, logging.getLogger().getEffectiveLevel()))
            submit = lambda lb, dd: pool.submit(_evaluate_point, lb, dd, walkforward, start)  # noqa: E731
        else:
            pool = cf.ThreadPoolExecutor(max_workers=jobs)
            submit = lambda lb, dd: pool.submit(_evaluate_point, lb, dd, walkforward, start, md_r)  # noqa: E731
        with pool:
            futs = {submit(label, d): label for label, d in tasks}
            for fut in cf.as_completed(futs):
                label = futs[fut]
                try:
                    results[label] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors[label] = f"{type(exc).__name__}: {exc}"
    t_eval = time.perf_counter() - t0
    for label, err in errors.items():
        logger.error("sweep point %s failed: %s", label, err)

    # ---- table ----------------------------------------------------------------------------
    daily: dict[str, pd.Series] = {}
    oos_starts, oos_ends = [], []
    for row in rows:
        label = row["label"]
        if label in errors:
            row.update(status="error", error=errors[label][:300])
        res = results.get(label)
        if res is None:
            continue
        for c in POINT_COLUMNS:
            row[c] = res["row"].get(c, math.nan)
        if metric not in row:
            row[metric] = res["row"].get(metric, math.nan)
        row["n_folds"] = res["n_folds"]
        row["seconds"] = res["seconds"]
        row["config_hash"] = res["config_hash"][:12]
        daily[label] = res["daily"]
        if res["oos_start"]:
            oos_starts.append(res["oos_start"])
            oos_ends.append(res["oos_end"])
    table = pd.DataFrame(rows)
    lead = [*g, "label", "status", metric]
    cols = lead + [c for c in POINT_COLUMNS if c != metric] + ["n_folds", "seconds", "config_hash", "error"]
    table = table.reindex(columns=list(dict.fromkeys(c for c in cols)))
    returns = pd.DataFrame(daily).sort_index() if daily else pd.DataFrame()
    if oos_starts:
        window.update({"oos_start": min(oos_starts), "oos_end": max(oos_ends)})

    # ---- statistics ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    ok = table["status"] == "ok"
    n_valid = int(ok.sum())
    write = cfg.output.save_results if write is None else write
    run_dir: Path | None = None
    if write:
        run_dir = Path(out_dir) if out_dir is not None else (
            Path(cfg.output.dir) / f"sweep_{sid}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}")
    ledger_path: Path | None = None
    if isinstance(ledger, (str, Path)):
        ledger_path = Path(ledger)
    elif ledger is None and run_dir is not None:
        ledger_path = run_dir.parent / SWEEP_LEDGER
    prior = {"n_prior_sweeps": 0, "n_prior_configs": 0, "ledger": str(ledger_path) if ledger_path else None,
             "extra": int(prior_trials)}
    if ledger_path is not None:
        prior.update(_prior_sweep_trials(read_sweep_ledger(ledger_path), start=_utc(window["start"]),
                                         end=_utc(window["end"]), symbol=cfg.data.symbol, own=set(hashes.values())))
        if prior["n_prior_configs"]:
            notes.append(f"{prior['n_prior_configs']} distinct configuration(s) from {prior['n_prior_sweeps']} "
                         f"earlier sweep(s) over an overlapping window are added to n_trials (ledger {ledger_path})")
    n_trials = max(1, n_valid + int(prior["n_prior_configs"]) + max(0, int(prior_trials)))
    plateau = plateau_score(table, g, metric, maximize=mx, knife_edge_ratio=knife_edge_ratio)
    best = _best_block(table, returns, plateau, metric, n_trials,
                       n_boot=cfg.walkforward.n_boot if n_boot is None else n_boot,
                       seed=cfg.seed if seed is None else seed)
    pbo = _grid_pbo(returns[[c for c in returns.columns if c in set(table.loc[ok, "label"])]],
                    cfg.walkforward.pbo_splits) if n_valid >= 2 else None
    if plateau.get("knife_edge"):
        msg = (f"KNIFE-EDGE optimum: its grid neighbours keep only {plateau['plateau_ratio']:.0%} of the best "
               f"{metric}")
        if plateau.get("robust_best") != plateau.get("best"):
            msg += f"; prefer the plateau centre {plateau['robust_best']}"
        else:
            msg += "; the best point is also the best-smoothed one, but treat its level as fragile"
        notes.append(msg)
    if plateau.get("edge_params"):
        notes.append(f"the best point lies on the edge of the grid for {plateau['edge_params']}: the optimum may be "
                     "outside the searched range (extending the grid is another search: more trials)")
    timing = {"evaluate_s": round(t_eval, 3), "stats_s": round(time.perf_counter() - t0, 3)}
    settings = {"executor": kind, "n_jobs": jobs, "strategy_registry_name": reg_name, "strategy_id": sid,
                "base_params": base, "metric": metric, "knife_edge_ratio": knife_edge_ratio,
                "walkforward": _jsonable({k: v for k, v in cfg.to_dict()["walkforward"].items()
                                          if k in ("train", "test", "step", "anchored", "purge", "embargo")}),
                "book": "standalone (single strategy, fixed weight 1, FDM 1)",
                "config_hash_base": cfg.config_hash()}
    result = SweepResult(strategy=sid, grid=g, metric=metric, maximize=mx, table=table, returns=returns,
                         best=best, plateau=plateau, n_trials=n_trials, prior=prior, pbo=pbo, window=window,
                         walkforward=walkforward, notes=notes, timing=timing, settings=settings)
    if run_dir is not None:
        t0 = time.perf_counter()
        result.save(run_dir)
        result.timing["write_s"] = round(time.perf_counter() - t0, 3)
    if ledger_path is not None:
        _append_ledger(ledger_path, {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "kind": "sweep", "strategy": sid,
            "registry_name": reg_name, "symbol": cfg.data.symbol, "timeframe": cfg.data.timeframe,
            "window_start": window["start"], "window_end": window["end"], "metric": metric,
            "grid": _jsonable(g), "n_points": len(points), "n_valid": n_valid, "n_trials": n_trials,
            "config_hashes": sorted(hashes[lb] for lb in table.loc[ok, "label"] if lb in hashes),
            "best": best.get("label"), "best_sharpe": _jsonable(best.get("sharpe")),
            "out_dir": str(run_dir) if run_dir else None,
        })
    result.timing["total_s"] = round(time.perf_counter() - t_all, 3)
    if result.out_dir is not None:
        (result.out_dir / "sweep.json").write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    return result


def _best_block(table: pd.DataFrame, returns: pd.DataFrame, plateau: Mapping[str, Any], metric: str,
                n_trials: int, *, n_boot: int, seed: int) -> dict[str, Any]:
    from aurum.research.stats import deflated_sharpe, sharpe_summary

    label = plateau.get("best")
    if label is None or label not in returns:
        return {"label": label, "n_trials": n_trials}
    dr = returns[label].dropna().to_numpy(float)
    row = table.loc[table["label"] == label].iloc[0]
    out: dict[str, Any] = {"label": label, "params": plateau.get("best_params"), "metric": metric,
                           "value": _py(row[metric]), "n_trials": n_trials, "n_days": int(dr.size)}
    if dr.size < 3:
        return out
    s = sharpe_summary(dr, 252.0, n_trials=n_trials, n_boot=n_boot, seed=seed)
    out.update({k: s.get(k) for k in ("sharpe", "ci_lower", "ci_upper", "psr", "dsr", "sr0", "min_trl_years",
                                      "sharpe_haircut", "skew", "kurtosis")})
    ok = table["status"] == "ok"
    trial = pd.to_numeric(table.loc[ok, "sharpe"], errors="coerce").dropna().to_numpy(float)
    if trial.size >= 2 and math.isfinite(s.get("sharpe_per_period", math.nan)):
        out["dsr_xs"] = deflated_sharpe(s["sharpe_per_period"], int(dr.size), s["skew"], s["kurtosis"],
                                        trial_srs=trial / math.sqrt(252.0), n_trials=max(n_trials, trial.size))
    else:
        out["dsr_xs"] = math.nan
    return {k: _py(v) for k, v in out.items()}


def _grid_pbo(returns: pd.DataFrame, n_splits: int) -> dict[str, Any] | None:
    from aurum.research.stats import pbo_cscv

    m = returns.dropna()
    if m.shape[1] < 2:
        return None
    s = int(n_splits) - int(n_splits) % 2
    while s >= 2 and m.shape[0] < 2 * s:
        s -= 2
    if s < 2:
        return None
    try:
        return _jsonable(pbo_cscv(m.to_numpy(), n_splits=s).to_dict())
    except ValueError as exc:  # pragma: no cover - degenerate matrices
        logger.warning("grid PBO failed: %s", exc)
        return None


# ============================================================================================
# plots
# ============================================================================================
_INK, _INK2, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _RED = "#2a78d6", "#e34948"


def _figure(size: tuple[float, float]) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, dpi=110, facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _style(ax: Any) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=_MUTED, labelcolor=_INK2, length=3, width=0.6)
    ax.title.set_color(_INK)


def plot_sweep(table: pd.DataFrame, grid: Mapping[str, Sequence[Any]], metric: str = "sharpe", *,
               maximize: bool | None = None, x: str | None = None, y: str | None = None,
               best: Mapping[str, Any] | None = None, robust: Mapping[str, Any] | None = None,
               title: str = "", subtitle: str = "") -> Any:
    """Matplotlib Figure: a heatmap of ``metric`` over two parameters (other parameters
    aggregated by their best value), or a line chart for a 1-D grid. The best point is
    boxed, the plateau centre circled; invalid / failed points are blank. Diverging colour
    scale centred at 0 for signed metrics."""
    from matplotlib.colors import TwoSlopeNorm

    mx = _better(metric, maximize)
    params = list(grid)
    t = table.copy()
    t[metric] = pd.to_numeric(t.get(metric), errors="coerce")
    if "status" in t:
        t.loc[t["status"] != "ok", metric] = np.nan
    if len(params) == 1:
        p = params[0]
        fig = _figure((7.5, 3.6))
        ax = fig.add_subplot(1, 1, 1)
        _style(ax)
        vals = [t.loc[t[p] == v, metric].max() if mx else t.loc[t[p] == v, metric].min() for v in grid[p]]
        xs = np.arange(len(grid[p]))
        ax.plot(xs, vals, color=_BLUE, marker="o", linewidth=1.8)
        ax.axhline(0.0, color=_MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(xs, [str(v) for v in grid[p]])
        ax.set_xlabel(p, color=_INK2)
        ax.set_ylabel(metric, color=_INK2)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
        for pt, mk, col in ((best, "s", _RED), (robust, "o", _INK)):
            if pt and p in pt and pt[p] in grid[p]:
                i = list(grid[p]).index(pt[p])
                if math.isfinite(vals[i]):
                    ax.scatter([i], [vals[i]], s=140, facecolors="none", edgecolors=col, marker=mk,
                               linewidths=1.6, zorder=5)
        _titles(fig, ax, title or f"{metric} by {p}", subtitle)
        return fig
    x = x or params[1]
    y = y or params[0]
    if x == y or x not in grid or y not in grid:
        raise ValueError(f"x/y must be two different grid parameters of {params}")
    agg = "max" if mx else "min"
    piv = t.pivot_table(index=y, columns=x, values=metric, aggfunc=agg, dropna=False)
    piv = piv.reindex(index=list(grid[y]), columns=list(grid[x]))
    z = piv.to_numpy(float)
    fig = _figure((max(5.5, 1.1 * len(grid[x]) + 3.0), max(3.6, 0.8 * len(grid[y]) + 2.0)))
    ax = fig.add_subplot(1, 1, 1)
    _style(ax)
    finite = z[np.isfinite(z)]
    cmap = "RdBu" if mx else "RdBu_r"
    if finite.size and finite.min() < 0 < finite.max():
        lim = float(np.abs(finite).max())
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, norm=norm, aspect="auto", origin="lower")
    else:
        im = ax.imshow(np.ma.masked_invalid(z), cmap="Blues" if mx else "Blues_r", aspect="auto", origin="lower")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(metric, color=_INK2)
    cb.ax.tick_params(colors=_MUTED, labelcolor=_INK2)
    ax.set_xticks(range(len(grid[x])), [str(v) for v in grid[x]])
    ax.set_yticks(range(len(grid[y])), [str(v) for v in grid[y]])
    ax.set_xlabel(x, color=_INK2)
    ax.set_ylabel(y, color=_INK2)
    def dark_cell(v: float) -> bool:
        """True when the cell's RENDERED colour is dark (white text/markers on it). Works for
        the diverging and the sequential scale alike (|v| alone does not: on a sequential
        scale the darkest cell can be the one closest to 0)."""
        if not math.isfinite(v):
            return False
        r, g, b, _ = im.cmap(im.norm(v))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            txt = "n/a" if not math.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark_cell(v) else (_MUTED if not math.isfinite(v) else _INK))
    from matplotlib.patches import Circle, Rectangle

    for pt, kind in ((best, "box"), (robust, "circle")):
        if not pt or x not in pt or y not in pt or pt[x] not in grid[x] or pt[y] not in grid[y]:
            continue
        j, i = list(grid[x]).index(pt[x]), list(grid[y]).index(pt[y])
        v = z[i, j]
        edge = "#ffffff" if dark_cell(v) else _INK
        if kind == "box":
            ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=edge, linewidth=2.4))
        else:
            ax.add_patch(Circle((j, i), 0.34, fill=False, edgecolor=edge, linewidth=1.4, linestyle="--"))
    extra = "" if len(params) == 2 else f" ({agg} over {', '.join(p for p in params if p not in (x, y))})"
    _titles(fig, ax, (title or f"{metric} sweep") + extra, subtitle or "box = best, dashed circle = plateau centre")
    return fig


def _titles(fig: Any, ax: Any, title: str, subtitle: str) -> None:
    import textwrap

    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=_INK)
    if subtitle:
        width = max(40, int(fig.get_figwidth() * 13))
        ax.set_title(textwrap.fill(subtitle, width), loc="left", fontsize=8.5, color=_INK2)
