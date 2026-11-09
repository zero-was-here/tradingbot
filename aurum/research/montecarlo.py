"""Stress and uncertainty: Monte Carlo equity paths and deterministic historical stress.

A backtest is ONE path through history. This module asks what else could have happened
with the same return-generating process, and what the current book would lose in the worst
historical episodes.

Monte Carlo (stationary block bootstrap)
----------------------------------------
Daily returns ``r_1..r_n`` of a book (UTC trading days, weekends folded into Monday:
:func:`aurum.backtest.metrics.daily_returns`) are resampled with the stationary bootstrap of
Politis & Romano (1994): a simulated path of ``H`` days concatenates blocks that start at a
uniformly random day and whose lengths are iid Geometric(``p = 1/b``), wrapping circularly.
Blocks keep volatility clustering and the serial dependence of overlapping positions, which
an iid bootstrap would destroy (making drawdowns look too mild). The mean block length
``b`` defaults to the Politis-White (2004) / Patton-Politis-White (2009) automatic choice
(:func:`aurum.research.stats.optimal_block_length`). This is exactly the index scheme of
:func:`aurum.research.stats.stationary_bootstrap_indices`, generalised to a horizon ``H``
that differs from the history length ``n`` (with ``H == n`` and the same generator state it
reproduces that function bit for bit).

Each path ``k`` compounds its sampled returns: ``E_k(t) = E_0 * prod_{s<=t} (1 + r_{k,s})``.
From the ``N x (H+1)`` equity matrix:

* ``CAGR_k = (E_k(H) / E_0)^(252/H) - 1``; final equity; total return.
* ``maxDD_k = min_t (E_k(t) / max_{s<=t} E_k(s) - 1)`` (negative).
* ``P(maxDD > x)`` for a list of thresholds - the drawdown exceedance table.
* **Risk of ruin** at loss threshold ``L``: ``P(min_t E_k(t) <= (1 - L) E_0)``, a loss
  measured from the STARTING equity (unlike drawdown, which is measured from the peak).
* **Time to recover**: trading days from the trough of a path's maximum drawdown until
  equity regains the peak before it (``NaN`` when not recovered within the horizon), and
  the longest stretch below the running peak (including an unfinished one at the end).
* **Fan chart**: the 5/25/50/75/95% quantiles of ``E_k(t)`` across paths at every ``t``.

Assumptions (read them before trusting a number): the future is a reshuffle of the sampled
past (same mean, volatility and dependence up to the block length; no regime the history
did not contain); the book's sizing rules are already inside the returns (so de-risking
after drawdowns is only captured as it happened historically); costs and financing are in
the returns.

Deterministic historical stress
-------------------------------
* **Worst k-day windows** of the book's own daily returns (compounded), with their dates,
  replayed on the current equity: "if the worst week of the backtest started today".
* **Gap shocks** on the current (last) and peak exposure through
  :func:`aurum.risk.var.stress_test` / :func:`~aurum.risk.var.gap_shock`: an instantaneous
  price gap of ``pct`` on the signed position, plus half the spread to exit.
* **Worst historical gold moves** (when the bars are available): the worst k-day
  close-to-close moves of gold itself applied to the current position.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.metrics import daily_returns

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DD_THRESHOLDS",
    "DEFAULT_QUANTILES",
    "DEFAULT_RUIN_THRESHOLDS",
    "DEFAULT_WINDOWS",
    "MonteCarloResult",
    "StressReport",
    "drawdown_exceedance",
    "fan_bands",
    "gap_shock_table",
    "load_book",
    "monte_carlo",
    "parse_horizon",
    "path_metrics",
    "plot_distributions",
    "plot_fan",
    "risk_of_ruin",
    "simulate_paths",
    "stationary_bootstrap_paths",
    "stress_run",
    "worst_windows",
]

TRADING_DAYS = 252
DEFAULT_DD_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DEFAULT_RUIN_THRESHOLDS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50)
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
DEFAULT_WINDOWS: tuple[int, ...] = (1, 5, 10, 21, 63)
#: paths generated per vectorised chunk (bounds peak memory; part of the seeded stream)
CHUNK_PATHS = 4096
_UNITS = {"Y": TRADING_DAYS, "M": 21, "W": 5, "D": 1}


# ============================================================================================
# helpers
# ============================================================================================
def parse_horizon(horizon: int | float | str) -> int:
    """Trading days in a horizon: an int, or ``"2Y"`` (252/yr), ``"6M"`` (21/mo), ``"4W"``,
    ``"30D"`` (case-insensitive; ``"1.5Y"`` allowed)."""
    if isinstance(horizon, bool):
        raise ValueError(f"invalid horizon {horizon!r}")
    if isinstance(horizon, (int, np.integer)):
        h = int(horizon)
    elif isinstance(horizon, float):
        h = int(round(horizon))
    else:
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([YyMmWwDd]?)\s*", str(horizon))
        if not m:
            raise ValueError(f"invalid horizon {horizon!r}: use days or e.g. 2Y, 6M, 4W, 30D")
        h = int(round(float(m.group(1)) * _UNITS[(m.group(2) or "D").upper()]))
    if h < 1:
        raise ValueError(f"horizon must be at least one trading day, got {horizon!r}")
    return h


def _clean_returns(returns: Any) -> np.ndarray:
    x = np.asarray(pd.Series(returns, dtype=float).dropna(), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least two finite daily returns")
    if (x <= -1.0).any():
        raise ValueError("a daily return <= -100% cannot be compounded (bankrupt history)")
    return x


# ============================================================================================
# simulation
# ============================================================================================
def stationary_bootstrap_paths(n_hist: int, n_paths: int, horizon: int, mean_block: float,
                               rng: np.random.Generator) -> np.ndarray:
    """``(n_paths, horizon)`` indices into a history of length ``n_hist`` (stationary bootstrap).

    Same algorithm and draw order as :func:`aurum.research.stats.stationary_bootstrap_indices`
    (a block starts at ``t`` with probability ``1/mean_block``; its first index is uniform on
    ``[0, n_hist)``; indices then advance by one, wrapping circularly), but the path length
    may differ from the history length.
    """
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    if n_hist < 1 or n_paths < 1 or horizon < 1:
        raise ValueError("n_hist, n_paths and horizon must be positive")
    new_block = rng.random((n_paths, horizon)) < 1.0 / float(mean_block)
    new_block[:, 0] = True
    starts = rng.integers(0, n_hist, size=(n_paths, horizon))
    t = np.arange(horizon)
    last = np.where(new_block, t, 0)
    np.maximum.accumulate(last, axis=1, out=last)
    start_vals = np.take_along_axis(starts, last, axis=1)
    return (start_vals + (t - last)) % n_hist


def simulate_paths(returns: Any, n_paths: int = 10_000, horizon: int | str = "2Y", *,
                   mean_block: float | None = None, seed: int = 0, initial_equity: float = 1.0
                   ) -> tuple[np.ndarray, float]:
    """``(equity, mean_block)``: an ``(n_paths, H + 1)`` matrix of simulated equity paths
    (column 0 = ``initial_equity``) from a stationary block bootstrap of daily ``returns``.

    Deterministic for a given ``seed`` (paths are generated in chunks of
    :data:`CHUNK_PATHS`, which is part of the seeded stream).
    """
    from aurum.research.stats import optimal_block_length

    x = _clean_returns(returns)
    h = parse_horizon(horizon)
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if mean_block is None:
        mean_block = float(optimal_block_length(x)["stationary"])
    mean_block = float(min(max(mean_block, 1.0), x.size))
    rng = np.random.default_rng(seed)
    logx = np.log1p(x)
    eq = np.empty((n_paths, h + 1), dtype=float)
    eq[:, 0] = 0.0
    for a in range(0, n_paths, CHUNK_PATHS):
        b = min(a + CHUNK_PATHS, n_paths)
        idx = stationary_bootstrap_paths(x.size, b - a, h, mean_block, rng)
        np.cumsum(logx[idx], axis=1, out=eq[a:b, 1:])
    np.exp(eq, out=eq)
    eq *= float(initial_equity)
    return eq, mean_block


def path_metrics(paths: np.ndarray, *, periods: float = TRADING_DAYS) -> pd.DataFrame:
    """Per-path statistics of an ``(N, H+1)`` equity matrix (column 0 = start).

    Columns: ``final_equity``, ``total_return``, ``cagr``, ``max_drawdown`` (<= 0),
    ``min_equity``, ``trough_day``, ``time_to_recover_days`` (from the max-drawdown trough to
    regaining the prior peak; NaN = not recovered within the horizon or no drawdown),
    ``recovered`` and ``longest_underwater_days``.
    """
    eq = np.asarray(paths, dtype=float)
    if eq.ndim != 2 or eq.shape[1] < 2:
        raise ValueError("paths must be an (n_paths, horizon + 1) matrix")
    e0 = eq[:, 0]
    h = eq.shape[1] - 1
    peak = np.maximum.accumulate(eq, axis=1)
    dd = eq / peak - 1.0
    trough = np.argmin(dd, axis=1)
    rows = np.arange(eq.shape[0])
    max_dd = dd[rows, trough]
    peak_at_trough = peak[rows, trough]
    t = np.arange(eq.shape[1])
    after = (t[None, :] > trough[:, None]) & (eq >= peak_at_trough[:, None] * (1 - 1e-12))
    has = after.any(axis=1)
    first = np.argmax(after, axis=1)
    ttr = np.where(has & (max_dd < 0), first - trough, np.nan).astype(float)
    under = dd < -1e-12
    last_above = np.where(~under, t[None, :], -1)
    np.maximum.accumulate(last_above, axis=1, out=last_above)
    longest = (t[None, :] - last_above).max(axis=1)
    final = eq[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = final / e0
        cagr = np.where(growth > 0, growth ** (periods / h) - 1.0, -1.0)
    return pd.DataFrame({
        "final_equity": final, "total_return": growth - 1.0, "cagr": cagr, "max_drawdown": max_dd,
        "min_equity": eq.min(axis=1), "trough_day": trough, "time_to_recover_days": ttr,
        "recovered": np.where(max_dd < 0, has, True), "longest_underwater_days": longest,
    })


def drawdown_exceedance(max_drawdown: Any, thresholds: Sequence[float] = DEFAULT_DD_THRESHOLDS) -> pd.DataFrame:
    """``P(max drawdown deeper than x)`` for each threshold ``x`` (positive fractions)."""
    dd = -np.asarray(max_drawdown, dtype=float)
    return pd.DataFrame({"threshold": [float(x) for x in thresholds],
                         "probability": [float((dd > x).mean()) for x in thresholds]})


def risk_of_ruin(paths: np.ndarray, thresholds: Sequence[float] = DEFAULT_RUIN_THRESHOLDS) -> pd.DataFrame:
    """``P(equity ever <= (1 - L) * starting equity)`` for each loss threshold ``L``, plus the
    probability of ending the horizon below it."""
    eq = np.asarray(paths, dtype=float)
    e0 = eq[:, :1]
    lo = eq.min(axis=1, keepdims=True) / e0
    end = eq[:, -1:] / e0
    return pd.DataFrame({"loss_threshold": [float(x) for x in thresholds],
                         "p_touch": [float((lo <= 1 - x).mean()) for x in thresholds],
                         "p_end_below": [float((end <= 1 - x).mean()) for x in thresholds]})


def fan_bands(paths: np.ndarray, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pd.DataFrame:
    """Quantiles of equity across paths at each step (index = trading day 0..H; columns
    ``q05``, ``q25``, ...). Monotone in the quantile at every step by construction."""
    q = sorted(float(x) for x in quantiles)
    vals = np.quantile(np.asarray(paths, dtype=float), q, axis=0)
    return pd.DataFrame(vals.T, columns=[f"q{round(x * 100):02d}" for x in q],
                        index=pd.RangeIndex(paths.shape[1], name="day"))


# ============================================================================================
# deterministic stress
# ============================================================================================
def worst_windows(returns: pd.Series, windows: Sequence[int] = DEFAULT_WINDOWS, *,
                  equity: float | None = None) -> pd.DataFrame:
    """Worst compounded ``k``-day return of a daily return series for each ``k``, with its
    first and last day and (when ``equity`` is given) the USD loss if it started now."""
    r = pd.Series(returns, dtype=float).dropna()
    rows = []
    lr = np.log1p(r.clip(lower=-0.999999))
    for k in windows:
        k = int(k)
        if k < 1 or k > len(r):
            rows.append({"days": k, "worst_return": math.nan, "start": None, "end": None})
            continue
        roll = lr.rolling(k).sum()
        i = int(np.nanargmin(roll.to_numpy()))
        w = float(np.expm1(roll.iloc[i]))
        rows.append({"days": k, "worst_return": w, "start": str(r.index[i - k + 1].date())
                     if hasattr(r.index[i - k + 1], "date") else str(r.index[i - k + 1]),
                     "end": str(r.index[i].date()) if hasattr(r.index[i], "date") else str(r.index[i])})
    out = pd.DataFrame(rows, columns=["days", "worst_return", "start", "end"])
    if equity is not None:
        out["pnl_usd_at_current_equity"] = out["worst_return"] * float(equity)
    return out


def gap_shock_table(position_lots: float, price: float, equity: float, *,
                    shocks: Mapping[str, float] | Sequence[float] | None = None, spread: float = 0.0,
                    instrument: Any = None) -> pd.DataFrame:
    """Instantaneous price gaps on a signed position (:func:`aurum.risk.var.stress_test`).

    ``shocks``: ``{name: pct}`` or a list of pcts (e.g. ``[-0.05, 0.05]``); default
    :data:`aurum.risk.var.DEFAULT_STRESS_SCENARIOS`. Rows sorted worst first.
    """
    from aurum.core.instrument import XAUUSD
    from aurum.risk.var import stress_test

    if shocks is not None and not isinstance(shocks, Mapping):
        shocks = {f"gap_{float(p):+.1%}": float(p) for p in shocks}
    return stress_test(float(position_lots), float(price), float(equity), scenarios=shocks,
                       instrument=instrument or XAUUSD, spread=float(spread))


# ============================================================================================
# results
# ============================================================================================
@dataclass
class MonteCarloResult:
    """Simulated equity paths and their statistics (module docstring)."""

    paths: np.ndarray                       # (n_paths, horizon + 1) equity
    metrics: pd.DataFrame                   # per-path statistics (path_metrics)
    fan: pd.DataFrame                       # quantile bands per day
    dd_table: pd.DataFrame
    ruin: pd.DataFrame
    horizon: int
    n_paths: int
    mean_block: float
    seed: int
    initial_equity: float
    n_history: int
    history_sharpe: float
    seconds: float = 0.0

    def distribution(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pd.DataFrame:
        """Quantiles (rows) of CAGR, max drawdown, final equity and longest underwater (cols),
        plus the mean."""
        cols = ["cagr", "max_drawdown", "final_equity", "total_return", "longest_underwater_days"]
        q = self.metrics[cols].quantile(list(quantiles))
        q.index = [f"q{round(x * 100):02d}" for x in quantiles]
        return pd.concat([q, self.metrics[cols].mean().to_frame("mean").T])

    def recovery(self) -> dict[str, float]:
        """Time-to-recover summary in trading days."""
        m = self.metrics
        had_dd = m["max_drawdown"] < 0
        rec = m.loc[had_dd & m["recovered"].astype(bool), "time_to_recover_days"]
        return {
            "p_not_recovered": float((had_dd & ~m["recovered"].astype(bool)).mean()),
            "ttr_median_days": float(rec.median()) if len(rec) else math.nan,
            "ttr_p75_days": float(rec.quantile(0.75)) if len(rec) else math.nan,
            "ttr_p95_days": float(rec.quantile(0.95)) if len(rec) else math.nan,
            "longest_underwater_median_days": float(m["longest_underwater_days"].median()),
            "longest_underwater_p95_days": float(m["longest_underwater_days"].quantile(0.95)),
        }

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        return _jsonable({
            "n_paths": self.n_paths, "horizon_days": self.horizon, "horizon_years": self.horizon / TRADING_DAYS,
            "mean_block": self.mean_block, "seed": self.seed, "initial_equity": self.initial_equity,
            "n_history_days": self.n_history, "history_sharpe": self.history_sharpe,
            "cagr_median": float(m["cagr"].median()), "cagr_q05": float(m["cagr"].quantile(0.05)),
            "cagr_q95": float(m["cagr"].quantile(0.95)), "p_loss": float((m["total_return"] < 0).mean()),
            "max_drawdown_median": float(m["max_drawdown"].median()),
            "max_drawdown_q05": float(m["max_drawdown"].quantile(0.05)),
            "drawdown_exceedance": self.dd_table.to_dict(orient="records"),
            "risk_of_ruin": self.ruin.to_dict(orient="records"), "recovery": self.recovery(),
            "seconds": self.seconds,
        })


def monte_carlo(returns: Any, *, n_paths: int = 10_000, horizon: int | str = "2Y", mean_block: float | None = None,
                seed: int = 0, initial_equity: float = 1.0,
                dd_thresholds: Sequence[float] = DEFAULT_DD_THRESHOLDS,
                ruin_thresholds: Sequence[float] = DEFAULT_RUIN_THRESHOLDS,
                quantiles: Sequence[float] = DEFAULT_QUANTILES) -> MonteCarloResult:
    """Block-bootstrap Monte Carlo of daily ``returns`` (module docstring). Vectorised:
    10,000 paths x 2 years runs in well under a second."""
    from aurum.research.stats import sharpe

    t0 = time.perf_counter()
    x = _clean_returns(returns)
    paths, b = simulate_paths(x, n_paths, horizon, mean_block=mean_block, seed=seed, initial_equity=initial_equity)
    m = path_metrics(paths)
    res = MonteCarloResult(paths=paths, metrics=m, fan=fan_bands(paths, quantiles),
                           dd_table=drawdown_exceedance(m["max_drawdown"], dd_thresholds),
                           ruin=risk_of_ruin(paths, ruin_thresholds), horizon=paths.shape[1] - 1,
                           n_paths=paths.shape[0], mean_block=b, seed=int(seed), initial_equity=float(initial_equity),
                           n_history=int(x.size), history_sharpe=float(sharpe(x, TRADING_DAYS)))
    res.seconds = round(time.perf_counter() - t0, 4)
    return res


@dataclass
class BookData:
    """A book read from a run directory (or a BacktestResult)."""

    equity: pd.Series
    positions: pd.Series | None
    timeseries: pd.DataFrame | None
    run_dir: Path | None = None
    book: str = "combined"
    config: dict[str, Any] | None = None


@dataclass
class StressReport:
    """Monte Carlo + deterministic stress of one book."""

    mc: MonteCarloResult
    daily_returns: pd.Series
    worst_windows: pd.DataFrame
    gap_shocks: pd.DataFrame
    gold_moves: pd.DataFrame | None
    exposure: dict[str, Any]
    source: str
    notes: list[str] = field(default_factory=list)
    out_dir: Path | None = None

    def summary(self) -> dict[str, Any]:
        return _jsonable({"source": self.source, "monte_carlo": self.mc.summary(),
                          "distribution": self.mc.distribution().reset_index().to_dict(orient="records"),
                          "worst_windows": self.worst_windows.to_dict(orient="records"),
                          "gap_shocks": self.gap_shocks.to_dict(orient="records"),
                          "gold_moves": None if self.gold_moves is None else self.gold_moves.to_dict(orient="records"),
                          "exposure": self.exposure, "notes": self.notes, "out_dir": self.out_dir})

    def save(self, directory: str | Path) -> Path:
        """Write ``stress.json``, CSV tables and ``fan.png`` / ``distributions.png``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.mc.distribution().to_csv(d / "distribution.csv")
        self.mc.fan.to_csv(d / "fan_bands.csv")
        self.mc.dd_table.to_csv(d / "drawdown_exceedance.csv", index=False)
        self.mc.ruin.to_csv(d / "risk_of_ruin.csv", index=False)
        self.worst_windows.to_csv(d / "worst_windows.csv", index=False)
        self.gap_shocks.to_csv(d / "gap_shocks.csv", index=False)
        if self.gold_moves is not None:
            self.gold_moves.to_csv(d / "gold_moves.csv", index=False)
        for name, fig in (("fan.png", plot_fan(self.mc)), ("distributions.png", plot_distributions(self.mc))):
            fig.savefig(d / name, dpi=110, facecolor=fig.get_facecolor())
        self.out_dir = d
        (d / "stress.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return d


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ============================================================================================
# run directories
# ============================================================================================
def load_book(run_dir: str | Path, book: str = "combined", *, phase: str = "research") -> BookData:
    """Read ``books/<book>/timeseries.parquet`` (``holdout/books/...`` for ``phase="holdout"``)
    of a walk-forward / backtest run directory."""
    d = Path(run_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"run directory {d} not found")
    base = d / "holdout" if phase == "holdout" else d
    p = base / "books" / book / "timeseries.parquet"
    if not p.exists():
        books = sorted(x.name for x in (base / "books").glob("*")) if (base / "books").is_dir() else []
        raise FileNotFoundError(f"{p} not found (books in this run: {books or 'none'}; was the run written "
                                "with save_books?)")
    ts = pd.read_parquet(p)
    if "equity" not in ts:
        raise ValueError(f"{p} has no 'equity' column")
    cfg = None
    cp = d / "config.yaml"
    if cp.exists():
        import yaml

        try:
            cfg = yaml.safe_load(cp.read_text(encoding="utf-8"))
        except yaml.YAMLError:  # pragma: no cover - corrupted file
            cfg = None
    pos = ts["position_close"] if "position_close" in ts else ts.get("positions")
    return BookData(equity=ts["equity"].astype(float), positions=pos, timeseries=ts, run_dir=d, book=book, config=cfg)


def _bars_for_run(cfg_dict: Mapping[str, Any] | None, last: pd.Timestamp) -> pd.DataFrame | None:
    """Bars of the run's configuration up to ``last`` (None if unavailable)."""
    if not cfg_dict:
        return None
    try:
        from aurum.core.config import config_from_dict

        d = dict(cfg_dict)
        d.pop("source", None)
        d.setdefault("data", {})
        cfg = config_from_dict(d, env={}, validate=False)
        if cfg.data.synthetic is None and not cfg.data.resolved_bars_path().exists():
            return None
        bars = cfg.data.load().bars
    except Exception as exc:  # noqa: BLE001 - optional enrichment
        logger.info("could not load the run's bars: %s", exc)
        return None
    return bars.loc[bars.index <= last]


def stress_run(source: Any, *, book: str = "combined", phase: str = "research", n_paths: int = 10_000,
               horizon: int | str = "2Y", seed: int = 0, mean_block: float | None = None,
               shocks: Mapping[str, float] | Sequence[float] | None = None,
               windows: Sequence[int] = DEFAULT_WINDOWS, price: float | None = None,
               bars: pd.DataFrame | None = None, load_data: bool = True, normalise: bool = True,
               instrument: Any = None) -> StressReport:
    """Stress one book: Monte Carlo + worst windows + gap shocks (module docstring).

    ``source``: a run directory (``book`` / ``phase`` select the book), a
    :class:`~aurum.backtest.result.BacktestResult`, or an equity Series. Exposure for the gap
    shocks is the position at the last bar (``position_close``) with ``price`` = the last
    close: from ``bars`` if given, else the run's own bars (``load_data``, read from its
    ``config.yaml`` and truncated at the last bar), else the mid of the last fill
    (approximate; noted). ``normalise``: Monte Carlo equity starts at 1.0 (fractions) instead
    of the book's last equity.
    """
    notes: list[str] = []
    fills = None
    if isinstance(source, (str, Path)):
        bd = load_book(source, book, phase=phase)
        src = f"{Path(source)} [{phase}/{book}]"
        fp = (Path(source) / ("holdout" if phase == "holdout" else "") / "books" / book / "fills.csv")
        if fp.exists():
            try:
                fills = pd.read_csv(fp)
            except (pd.errors.EmptyDataError, ValueError):
                fills = None
    elif isinstance(source, pd.Series):
        bd = BookData(equity=source.astype(float), positions=None, timeseries=None, book="series")
        src = "equity series"
    elif hasattr(source, "equity") and hasattr(source, "positions"):
        pos = source.position_close if getattr(source, "position_close", None) is not None else source.positions
        bd = BookData(equity=source.equity.astype(float), positions=pos, timeseries=None, book=book)
        src = "BacktestResult"
        fills = getattr(source, "fills", None)
    else:
        raise TypeError("source must be a run directory, a BacktestResult or an equity Series")
    eq = bd.equity.dropna()
    if len(eq) < 3:
        raise ValueError("the book has fewer than three equity marks")
    dr = daily_returns(eq).dropna()
    last_eq = float(eq.iloc[-1])
    mc = monte_carlo(dr, n_paths=n_paths, horizon=horizon, mean_block=mean_block, seed=seed,
                     initial_equity=1.0 if normalise else last_eq)
    ww = worst_windows(dr, windows, equity=last_eq)

    # ---- exposure -------------------------------------------------------------------------
    lots = float(bd.positions.dropna().iloc[-1]) if bd.positions is not None and len(bd.positions.dropna()) else 0.0
    if bars is None and price is None and load_data and bd.config is not None:
        bars = _bars_for_run(bd.config, eq.index[-1])
    gold_moves = None
    exposure: dict[str, Any] = {"time": str(eq.index[-1]), "equity": last_eq, "lots": lots}
    spread = 0.0
    if price is None and bars is not None and len(bars):
        b = bars.loc[bars.index <= eq.index[-1]]
        if len(b):
            price = float(b["close"].iloc[-1])
            spread = float(b["spread"].iloc[-1]) if "spread" in b else 0.0
            exposure["price_source"] = "bars (last close)"
            gold_moves = _gold_moves(b, windows, lots, price, last_eq, instrument)
            if bd.positions is not None:
                pos = bd.positions.reindex(b.index).dropna()
                if len(pos):
                    c = b["close"].reindex(pos.index)
                    e = eq.reindex(pos.index)
                    lev = (pos.abs() * _contract(instrument) * c / e).replace([np.inf, -np.inf], np.nan)
                    if lev.notna().any():
                        t_pk = lev.idxmax()
                        exposure["peak"] = {"time": str(t_pk), "lots": float(pos.loc[t_pk]),
                                            "leverage": float(lev.loc[t_pk]), "price": float(c.loc[t_pk]),
                                            "equity": float(e.loc[t_pk])}
    if price is None and fills is not None and len(fills) and "mid" in fills:
        price = float(fills["mid"].iloc[-1])
        exposure["price_source"] = "last fill mid (approximate)"
        notes.append("gap shocks priced at the mid of the last fill (bars unavailable): approximate")
    if price is None:
        exposure["price_source"] = "unavailable"
        notes.append("no price available for the gap shocks (pass price= or bars=)")
        gaps = pd.DataFrame(columns=["scenario", "shock_pct", "pnl_usd", "pnl_pct_equity", "equity_after"])
    else:
        exposure["price"] = float(price)
        exposure["notional"] = lots * _contract(instrument) * float(price)
        exposure["leverage"] = abs(exposure["notional"]) / last_eq if last_eq > 0 else math.nan
        gaps = gap_shock_table(lots, float(price), last_eq, shocks=shocks, spread=spread, instrument=instrument)
        gaps.insert(0, "exposure", "current")
        pk = exposure.get("peak")
        if pk and pk["lots"] != 0:
            g2 = gap_shock_table(pk["lots"], pk["price"], pk["equity"], shocks=shocks, instrument=instrument)
            g2.insert(0, "exposure", f"peak ({pk['time'][:10]})")
            gaps = pd.concat([gaps, g2], ignore_index=True)
    if lots == 0.0:
        notes.append("the book is flat at its last bar: current-exposure gap shocks are zero")
    return StressReport(mc=mc, daily_returns=dr, worst_windows=ww, gap_shocks=gaps, gold_moves=gold_moves,
                        exposure=exposure, source=src, notes=notes)


def _contract(instrument: Any) -> float:
    from aurum.core.instrument import XAUUSD

    return float((instrument or XAUUSD).contract_size)


def _gold_moves(bars: pd.DataFrame, windows: Sequence[int], lots: float, price: float, equity: float,
                instrument: Any) -> pd.DataFrame:
    """Worst k-day close-to-close DROP and RISE of gold (daily closes) applied to ``lots``."""
    from aurum.backtest.metrics import daily_equity
    from aurum.core.instrument import XAUUSD
    from aurum.risk.var import gap_shock

    inst = instrument or XAUUSD
    close = daily_equity(bars["close"].astype(float))
    lr = np.log(close).diff().dropna()
    rows = []
    for k in windows:
        k = int(k)
        if k < 1 or k > len(lr):
            continue
        roll = lr.rolling(k).sum()
        drop, rise = float(np.expm1(roll.min())), float(np.expm1(roll.max()))
        i_d, i_r = roll.idxmin(), roll.idxmax()
        pd_, pr_ = gap_shock(lots, price, drop, instrument=inst), gap_shock(lots, price, rise, instrument=inst)
        rows.append({"days": k, "worst_drop": drop, "drop_end": str(pd.Timestamp(i_d).date()),
                     "worst_rise": rise, "rise_end": str(pd.Timestamp(i_r).date()),
                     "pnl_usd_current_on_drop": pd_, "pnl_usd_current_on_rise": pr_,
                     "pnl_pct_equity_worst": min(pd_, pr_) / equity if equity > 0 else math.nan})
    return pd.DataFrame(rows)


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
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_fan(mc: MonteCarloResult, *, n_sample_paths: int = 12, title: str = "") -> Any:
    """Fan chart: 5-95% and 25-75% bands, the median, a few sample paths and the start level."""
    fig = _figure((9.0, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    _style(ax)
    f = mc.fan
    years = f.index.to_numpy(float) / TRADING_DAYS
    cols = list(f.columns)
    lo, hi = cols[0], cols[-1]
    inner = (cols[1], cols[-2]) if len(cols) >= 4 else None
    rng = np.random.default_rng(mc.seed + 1)
    if n_sample_paths:
        for k in rng.choice(mc.paths.shape[0], size=min(n_sample_paths, mc.paths.shape[0]), replace=False):
            ax.plot(years, mc.paths[k], color=_MUTED, linewidth=0.5, alpha=0.45)
    ax.fill_between(years, f[lo], f[hi], color=_BLUE, alpha=0.15, linewidth=0,
                    label=f"{lo[1:]}-{hi[1:]}% band")
    if inner:
        ax.fill_between(years, f[inner[0]], f[inner[1]], color=_BLUE, alpha=0.30, linewidth=0,
                        label=f"{inner[0][1:]}-{inner[1][1:]}% band")
    mid = "q50" if "q50" in f else cols[len(cols) // 2]
    ax.plot(years, f[mid], color=_BLUE, linewidth=2.0, label="median")
    ax.axhline(mc.initial_equity, color=_INK2, linewidth=0.9, linestyle="--", label="start")
    ax.set_xlabel("years ahead", color=_INK2)
    ax.set_ylabel("equity" + (" (start = 1)" if mc.initial_equity == 1.0 else ""), color=_INK2)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(_INK2)
    ax.set_title(title or (f"Monte Carlo fan chart: {mc.n_paths:,} stationary-bootstrap paths, "
                           f"mean block {mc.mean_block:.1f} days"), loc="left", fontsize=10, fontweight="bold",
                 color=_INK)
    return fig


def plot_distributions(mc: MonteCarloResult) -> Any:
    """Histograms of simulated CAGR, maximum drawdown (with their 5/50/95% quantiles) and the
    time to recover from the maximum drawdown (recovered paths; the share of paths that did not
    recover within the horizon is printed)."""
    fig = _figure((13.0, 3.6))
    m = mc.metrics
    for i, (col, label, color) in enumerate((("cagr", "CAGR over the horizon", _BLUE),
                                             ("max_drawdown", "maximum drawdown", _RED))):
        ax = fig.add_subplot(1, 3, i + 1)
        _style(ax)
        v = m[col].to_numpy(float) * 100.0
        ax.hist(v[np.isfinite(v)], bins=60, color=color, alpha=0.75, linewidth=0)
        for q, ls in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
            x = float(np.nanquantile(v, q))
            ax.axvline(x, color=_INK, linewidth=1.0, linestyle=ls)
            ax.annotate(f"q{round(q * 100):02d} {x:.1f}%", xy=(x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -4), textcoords="offset points", fontsize=8, color=_INK2, rotation=90,
                        va="top", ha="left",
                        bbox={"boxstyle": "square,pad=0.15", "facecolor": _SURFACE, "edgecolor": "none",
                              "alpha": 0.85})
        ax.set_xlabel(f"{label} (%)", color=_INK2)
        ax.set_title(label, loc="left", fontsize=10, fontweight="bold", color=_INK)
    ax = fig.add_subplot(1, 3, 3)
    _style(ax)
    rec = mc.recovery()
    ttr = m["time_to_recover_days"].to_numpy(float)
    ttr = ttr[np.isfinite(ttr)]
    if ttr.size:
        ax.hist(ttr, bins=min(60, max(5, int(ttr.max()) // 5 + 1)), color="#1f9e89", alpha=0.8, linewidth=0)
        med = float(np.median(ttr))
        ax.axvline(med, color=_INK, linewidth=1.0)
        ax.annotate(f"median {med:.0f} d", xy=(med, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -4),
                    textcoords="offset points", fontsize=8, color=_INK2, rotation=90, va="top", ha="left")
    ax.text(0.98, 0.95, f"not recovered within\nthe horizon: {rec['p_not_recovered']:.0%}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=_INK2)
    ax.set_xlabel("trading days from the trough to a new high", color=_INK2)
    ax.set_title("time to recover from the max drawdown", loc="left", fontsize=10, fontweight="bold", color=_INK)
    return fig
