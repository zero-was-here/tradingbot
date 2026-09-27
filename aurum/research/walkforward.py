"""Walk-forward research protocol (SPEC §9): refit per fold, stitch OOS, backtest ONCE.

Protocol
--------
1. **Folds** come from :func:`aurum.research.splits.walk_forward_splits` on the bar index
   of the research span. Durations ("3Y", "6M") are converted to bar counts with the
   sample's calendar density. Between the last training bar and the first test bar there
   is a gap of ``purge + embargo`` bars; ``purge`` is never below the largest label horizon
   declared by a trainable strategy (López de Prado 2018, *AFML* ch. 7), and trainable
   strategies only ever *see* the training bars when fitting, so no label can reach into
   the test block.
2. **Per fold**: the :class:`~aurum.features.pipeline.FeaturePipeline` scaler is fitted on
   the fold's training rows only; each strategy is cloned, fitted on the training bars
   (trainable ones) and generated on history ``[hist_start, test_end)`` — warm-up uses bars
   *before* the test start, which is legitimate because it is the past — and only the test
   rows are kept. Non-trainable strategies are causal by contract (``forecast[t]`` uses bars
   ``<= t``, enforced by the strategy leakage tests), so by default they are generated once
   on the research span and sliced, which is identical to per-fold generation and far
   cheaper (``regenerate_per_fold: true`` forces the strict per-fold path).
   ``Strategy.fit`` receives macro frames truncated to rows PUBLISHED by the decision time
   of the last training bar (``available_at <= train end``): defensive point-in-time, so a
   strategy that fits on whole macro frames instead of ``asof_join``-ing them still cannot
   see a later print.
   The :class:`~aurum.portfolio.combiner.ForecastCombiner` for fold ``k`` is fitted
   (default ``walkforward.combiner_fit="oos"``) on the stitched out-of-sample forecasts of
   folds ``< k`` (strictly before fold ``k``'s test block; equal weights until
   ``combiner_min_obs`` such bars exist), then combines the test rows.
   ``combiner_fit="train"`` is the literal "fit on the training-period forecasts" variant.
   Strategies are scored NET of estimated trading costs: the combiner gets the bars'
   spreads/ranges and the SAME cost model and instrument as the backtests, and charges
   ``c_t * |Δforecast|`` per bar (derivation in :mod:`aurum.portfolio.combiner`); under
   ``sharpe_shrink`` a strategy with a non-positive net Sharpe gets zero weight and the
   weights may sum to < 1 (unallocated risk, see the combiner docs).
3. **Stitching**: test blocks are contiguous and non-overlapping (``step >= test``), so the
   OOS forecasts of all folds form one strictly increasing series per strategy and for the
   combined book.
4. **One continuous backtest** per book over the stitched OOS span, through the SAME sizer,
   research risk limits and cost model the live system uses (SPEC §0.2), plus a buy-and-hold
   benchmark over the same span.
5. **Statistics** on DAILY returns: Sharpe, PSR and DSR (Bailey & López de Prado 2012,
   2014) with ``n_trials`` = strategy configurations evaluated in this run (one config x N
   strategies, or ``walkforward.n_trials``) + the number of DISTINCT other configs that
   evaluated an overlapping holdout window before (holdout ledger, step 6), stationary
   bootstrap Sharpe CI (Politis & Romano 1994), PBO by CSCV over the per-strategy OOS daily
   return matrix (Bailey, Borwein, López de Prado & Zhu 2017), per-fold results, combiner
   weights per fold and cost attribution.
6. **Holdout** (optional ``walkforward.holdout_start``): every bar from that timestamp on is
   physically removed from the research span (features, strategies, backtests of the folds
   never see it). After the folds, one final fit on the last training window is evaluated
   on the holdout ONCE and reported separately. Every holdout evaluation is appended to the
   **holdout ledger** (``holdout_ledger.jsonl`` in the output-directory root, next to the run
   directories: timestamp, config hash, data hash, strategies, window, holdout metrics). The
   ledger is read first: when the same (overlapping) window was already evaluated by a
   DIFFERENT config the run logs a WARNING, notes it in the report (and the holdout
   tearsheet) and adds those configs to the DSR ``n_trials``. The holdout OOS forecasts are
   kept (``holdout/oos_forecasts.parquet``) so the production fit can weight on them.

Known bias of ``combiner_fit="train"`` (documented, not hidden): the combiner is then
fitted on *in-sample* forecasts of trainable strategies over the training window (their
fitted models have seen those rows), which favours trainable strategies in the weights.
The OOS evaluation is still out of sample; the weights are just biased. The default
``"oos"`` mode avoids this by weighting strategies on their past OOS forecasts only
(Timmermann 2006). Even in ``"oos"`` mode the equal-weight fallback takes its activity
flags and FDM from training-window forecast *correlations* (no returns involved).

Parallelism: per-strategy work (all folds of one strategy) and the per-book backtests run
on a ``concurrent.futures`` pool. Processes are used for registry strategies on large
samples (the market data is shipped once per worker via the pool initializer); strategies
that consume pipeline features run on threads in the parent, sharing the feature matrix.
Every result is deterministic regardless of the executor.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.engine import buy_and_hold_benchmark, run_backtest
from aurum.backtest.metrics import daily_returns
from aurum.backtest.result import BacktestResult
from aurum.core.config import AurumConfig, ConfigError, duration_to_bars
from aurum.core.types import MarketData
from aurum.portfolio.combiner import ForecastCombiner
from aurum.research.splits import walk_forward_splits
from aurum.research.stats import PBOResult, pbo_cscv, sharpe, sharpe_summary
from aurum.strategies.base import Strategy

logger = logging.getLogger(__name__)

__all__ = [
    "HOLDOUT_LEDGER",
    "FoldPlan",
    "HoldoutReport",
    "QuantBook",
    "append_holdout_ledger",
    "fit_quant_book",
    "load_summary",
    "WalkForwardReport",
    "book_statistics",
    "label_horizon",
    "plan_folds",
    "prior_holdout_looks",
    "provenance",
    "read_holdout_ledger",
    "run_single_backtest",
    "run_walk_forward",
    "train_feature_reference",
]

#: strategy parameters that denote a label / holding horizon in bars (purge candidates)
HORIZON_PARAMS: tuple[str, ...] = ("label_horizon", "horizon", "max_holding", "max_holding_bars",
                                   "vertical_barrier", "hold_bars", "forward_bars")
#: minimum research bars per worker task before processes beat threads (spawn overhead)
_PROCESS_MIN_BARS = 20_000
_BENCHMARK = "benchmark"
_COMBINED = "combined"


# ============================================================================================
# plans
# ============================================================================================
@dataclass(frozen=True)
class FoldPlan:
    """Positions (in the FULL bars frame) of one fold. Ends are exclusive."""

    key: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    hist_start: int
    phase: str = "research"      # "research" (folds, sliced to the research span) | "holdout"

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


def label_horizon(strategy: Strategy) -> int:
    """Largest label/holding horizon (bars) a TRAINABLE strategy declares (0 if none).

    Looks at a ``label_horizon`` attribute/property and at horizon-like parameters
    (:data:`HORIZON_PARAMS`). Non-trainable strategies learn nothing, so they need no purge.
    """
    if not getattr(strategy, "trainable", False):
        return 0
    vals: list[int] = []
    attr = getattr(strategy, "label_horizon", None)
    if isinstance(attr, (int, np.integer)) and not isinstance(attr, bool):
        vals.append(int(attr))
    params = getattr(strategy, "params", {}) or {}
    for k in HORIZON_PARAMS:
        v = params.get(k)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            vals.append(int(v))
        elif isinstance(v, (list, tuple)):
            vals.extend(int(x) for x in v if isinstance(x, (int, np.integer)) and not isinstance(x, bool))
    return max(vals, default=0)


def _needs_features(strategy: Strategy, enabled: bool | str) -> bool:
    """Whether the walk-forward should hand this strategy the fold's pipeline features.

    A strategy may declare ``uses_features = True/False`` (or ``requires_features``).
    Otherwise: ``features.enabled: true`` feeds every TRAINABLE strategy; ``"auto"`` feeds
    none (self-contained strategies such as ``ml_gbm`` then build and fit their own
    FeaturePipeline on the training slice); ``false`` never computes features.
    """
    if enabled is False:
        return False
    declared = getattr(strategy, "uses_features", None)
    if declared is None:
        declared = getattr(strategy, "requires_features", None)
    if declared is not None:
        return bool(declared)
    return bool(enabled is True and getattr(strategy, "trainable", False))


def _bars_per_day(index: pd.DatetimeIndex) -> float:
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    if span_days <= 0:
        raise ValueError("bar index must span a positive time")
    return (len(index) - 1) / span_days


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _gap_bars(value: int | str | None, bars_per_day: float) -> int:
    """Bars of a purge / embargo setting: ``0``, ``None`` or ``"auto"`` mean no explicit gap.

    (:func:`duration_to_bars` rejects 0 because a zero-length *window* is an error; a zero
    *gap* is legitimate.)
    """
    if value is None or value == "auto" or (isinstance(value, (int, float)) and not isinstance(value, bool)
                                            and value == 0):
        return 0
    return duration_to_bars(value, bars_per_day)


def _config_purge(cfg: AurumConfig, strategies: Mapping[str, Strategy], bars_per_day: float
                  ) -> tuple[int, int, dict[str, int]]:
    """(effective purge, configured purge, per-strategy label horizons).

    The effective purge is never below the largest label horizon of a trainable strategy.
    """
    horizons = {k: label_horizon(s) for k, s in strategies.items()}
    cfg_purge = _gap_bars(cfg.walkforward.purge, bars_per_day)
    return max(cfg_purge, max(horizons.values(), default=0)), cfg_purge, horizons


def plan_folds(index: pd.DatetimeIndex, cfg: AurumConfig, strategies: Mapping[str, Strategy] | None = None
               ) -> tuple[list[FoldPlan], FoldPlan | None, dict[str, Any]]:
    """Fold plans for ``index`` (research folds, optional holdout plan, resolved settings).

    Raises :class:`ConfigError` when the configuration yields no usable fold.
    """
    wf = cfg.walkforward
    n = len(index)
    n_research = n
    if wf.holdout_start is not None:
        hs = _utc(wf.holdout_start)
        n_research = int(index.searchsorted(hs, side="left"))
        if n_research >= n:
            raise ConfigError(f"walkforward.holdout_start {wf.holdout_start} is after the last bar "
                              f"({index[-1]}): the holdout would be empty")
    if n_research < 2:
        raise ConfigError("research span before walkforward.holdout_start has fewer than two bars")
    bpd = _bars_per_day(index[:n_research])
    train = duration_to_bars(wf.train, bpd)
    test = duration_to_bars(wf.test, bpd)
    step = test if wf.step is None else duration_to_bars(wf.step, bpd)
    embargo = _gap_bars(wf.embargo, bpd)
    purge, cfg_purge, horizons = _config_purge(cfg, strategies or {}, bpd)
    if purge > cfg_purge and wf.purge != "auto":
        logger.info("walkforward: purge raised from %d to the max label horizon %d", cfg_purge, purge)
    if step < test:
        raise ConfigError(f"walkforward.step ({step} bars) < test ({test} bars): overlapping test "
                          "blocks cannot be stitched into one OOS series")
    if train < wf.min_train_bars:
        raise ConfigError(f"walkforward.train is {train} bars (< min_train_bars={wf.min_train_bars})")
    splits = walk_forward_splits(n_research, train, test, step=step, anchored=wf.anchored, purge=purge,
                                 embargo=embargo, min_test=max(1, wf.min_test_bars))
    if not splits:
        raise ConfigError(
            f"walk-forward produced no fold: research span has {n_research} bars but "
            f"train ({train}) + purge ({purge}) + embargo ({embargo}) + min_test ({wf.min_test_bars}) "
            "does not fit; shorten walkforward.train/test or load more data")
    hist_cap = wf.history_bars
    plans = []
    for k, (tr, te) in enumerate(splits):
        tr0, tr1 = int(tr[0]), int(tr[-1]) + 1
        if tr1 - tr0 != len(tr):
            raise AssertionError("walk-forward training window is expected to be contiguous")
        hist = 0 if hist_cap is None else max(0, tr0 - int(hist_cap))
        plans.append(FoldPlan(str(k), tr0, tr1, int(te[0]), int(te[-1]) + 1, hist, "research"))
    holdout = None
    if n_research < n:
        gap = purge + embargo
        tr1 = n_research - gap
        tr0 = 0 if wf.anchored else max(0, tr1 - train)
        if tr1 - tr0 < wf.min_train_bars:
            raise ConfigError("not enough research bars before walkforward.holdout_start for the final fit")
        hist = 0 if hist_cap is None else max(0, tr0 - int(hist_cap))
        holdout = FoldPlan("holdout", tr0, tr1, n_research, n, hist, "holdout")
    resolved = {"n_bars": n, "n_research": n_research, "bars_per_day": bpd, "train": train, "test": test,
                "step": step, "purge": purge, "embargo": embargo, "anchored": wf.anchored,
                "label_horizons": horizons, "n_folds": len(plans),
                "holdout_start": str(index[n_research]) if n_research < n else None,
                "stitched_gaps": step > test}
    return plans, holdout, resolved


# ============================================================================================
# execution context & tasks (module level: picklable for process pools)
# ============================================================================================
@dataclass
class _Ctx:
    md: MarketData                                  # FULL data (research span + holdout)
    n_research: int
    raw: dict[str, pd.DataFrame] = field(default_factory=dict)   # phase -> raw features
    pipes: dict[str, Any] = field(default_factory=dict)          # fold key -> fitted pipeline


@dataclass
class _BookSpec:
    sizer: Any
    risk: Any                       # RiskConfig (a fresh manager is built per book)
    instrument: Any
    costs: Any
    initial_equity: float
    stop_atr_mult: float | None
    take_profit_atr_mult: float | None
    atr_period: int
    stop_cooldown_bars: int
    event_horizon_hours: float


_WORKER_CTX: _Ctx | None = None


def _init_worker(ctx: _Ctx, log_level: int, threads: int) -> None:  # pragma: no cover - subprocess
    global _WORKER_CTX
    _WORKER_CTX = ctx
    # Avoid n_jobs x n_cores OpenMP/BLAS oversubscription (e.g. HistGradientBoosting): the
    # native thread pools read these variables when first loaded, which happens after this.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    logging.basicConfig(level=log_level, format="%(processName)s %(name)s %(levelname)s: %(message)s")


def _slice(md: MarketData, a: int, b: int) -> MarketData:
    return MarketData(bars=md.bars.iloc[a:b], macro=md.macro, events=md.events)


def _pit_macro(macro: Mapping[str, pd.DataFrame] | None, cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Macro frames restricted to rows published by ``cutoff`` (``available_at <= cutoff``).

    Frames without ``available_at`` (off-spec) fall back to ``index <= cutoff``.
    """
    out: dict[str, pd.DataFrame] = {}
    for k, v in (macro or {}).items():
        if v is None or len(v) == 0:
            out[k] = v
            continue
        if "available_at" in v.columns:
            av = pd.to_datetime(v["available_at"], utc=True)
            out[k] = v.loc[(av <= cutoff).to_numpy()]
        else:
            logger.warning("macro frame %r has no available_at column: truncated by its index", k)
            ix = pd.DatetimeIndex(v.index)
            ix = ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")
            out[k] = v.loc[(ix <= cutoff)]
    return out


def _train_md(md: MarketData, a: int, b: int) -> MarketData:
    """Training slice for ``Strategy.fit``: bars ``[a, b)`` and macro rows PUBLISHED by the
    decision time of the last training bar (defensive point-in-time: a strategy that fits on
    whole macro frames instead of ``asof_join``-ing them can still never see a later print)."""
    bars = md.bars.iloc[a:b]
    if not md.macro or len(bars) == 0:
        return MarketData(bars=bars, macro=md.macro, events=md.events)
    if "available_at" in bars.columns:
        cutoff = pd.Timestamp(bars["available_at"].iloc[-1])
    else:  # pragma: no cover - canonical bars always carry available_at
        cutoff = pd.Timestamp(bars.index[-1])
    cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
    return MarketData(bars=bars, macro=_pit_macro(md.macro, cutoff), events=md.events)


def _phase_md(ctx: _Ctx, phase: str) -> MarketData:
    if phase == "holdout" or ctx.n_research >= len(ctx.md.bars):
        return ctx.md
    return _slice(ctx.md, 0, ctx.n_research)


def _generate(strategy: Strategy, md: MarketData, features: pd.DataFrame | None) -> np.ndarray:
    out = strategy.generate(md, features)
    idx = md.bars.index
    s = out if isinstance(out, pd.Series) else pd.Series(np.asarray(out, dtype=float), index=idx)
    if not s.index.equals(idx):
        s = s.reindex(idx)
    arr = s.to_numpy(dtype=float, copy=True)
    bad = ~np.isfinite(arr)
    if bad.any():
        logger.warning("%s: %d non-finite forecasts set to 0", strategy.name, int(bad.sum()))
        arr[bad] = 0.0
    if arr.size and (arr.max() > 1.0 + 1e-9 or arr.min() < -1.0 - 1e-9):
        logger.warning("%s: forecasts outside [-1, 1] clipped", strategy.name)
    return np.clip(arr, -1.0, 1.0)


def _strategy_task(key: str, strategy: Strategy, plans: Sequence[FoldPlan], needs_features: bool,
                   regenerate: bool, ctx: _Ctx | None = None) -> dict[str, Any]:
    """All folds of ONE strategy -> {fold key: (train forecasts, test forecasts)} + timing."""
    ctx = ctx if ctx is not None else _WORKER_CTX
    if ctx is None:  # pragma: no cover - defensive
        raise RuntimeError("walk-forward worker context not initialised")
    per_fold = bool(strategy.trainable or needs_features or regenerate)
    folds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_s = gen_s = 0.0
    phase_cache: dict[str, np.ndarray] = {}
    for plan in plans:
        if not per_fold:
            f = phase_cache.get(plan.phase)
            if f is None:
                t0 = time.perf_counter()
                f = _generate(strategy.clone(), _phase_md(ctx, plan.phase), None)
                gen_s += time.perf_counter() - t0
                phase_cache[plan.phase] = f
            folds[plan.key] = (f[plan.train_start:plan.train_end].copy(),
                               f[plan.test_start:plan.test_end].copy())
            continue
        s = strategy.clone()
        x_hist = x_train = None
        if needs_features:
            raw = ctx.raw[plan.phase]
            x_hist = ctx.pipes[plan.key].transform(raw.iloc[plan.hist_start:plan.test_end])
            x_train = x_hist.iloc[plan.train_start - plan.hist_start:plan.train_end - plan.hist_start]
        t0 = time.perf_counter()
        if s.trainable:
            # Feature warm-up may reach into older PAST bars (never into any test block).
            fit_a = plan.train_start if needs_features else max(0, plan.train_start - s.fit_history_bars)
            s.fit(_train_md(ctx.md, fit_a, plan.train_end), x_train)
        t1 = time.perf_counter()
        f = _generate(s, _slice(ctx.md, plan.hist_start, plan.test_end), x_hist)
        t2 = time.perf_counter()
        fit_s += t1 - t0
        gen_s += t2 - t1
        off = plan.hist_start
        folds[plan.key] = (f[plan.train_start - off:plan.train_end - off].copy(),
                           f[plan.test_start - off:plan.test_end - off].copy())
    return {"key": key, "folds": folds, "fit_s": fit_s, "gen_s": gen_s, "per_fold": per_fold}


def _book_task(name: str, phase: str, forecast: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp,
               spec: _BookSpec, ctx: _Ctx | None = None) -> BacktestResult:
    """One continuous backtest (``forecast=None`` -> buy-and-hold benchmark)."""
    ctx = ctx if ctx is not None else _WORKER_CTX
    if ctx is None:  # pragma: no cover - defensive
        raise RuntimeError("walk-forward worker context not initialised")
    md = _phase_md(ctx, phase)
    if forecast is None:
        res = buy_and_hold_benchmark(md, instrument=spec.instrument, costs=spec.costs,
                                     initial_equity=spec.initial_equity, start=start, end=end)
    else:
        risk = spec.risk.build("research", instrument=spec.instrument) if spec.risk is not None else None
        res = run_backtest(md, forecast, sizer=spec.sizer, risk=risk, instrument=spec.instrument,
                           costs=spec.costs, initial_equity=spec.initial_equity,
                           stop_atr_mult=spec.stop_atr_mult, take_profit_atr_mult=spec.take_profit_atr_mult,
                           atr_period=spec.atr_period, stop_cooldown_bars=spec.stop_cooldown_bars,
                           event_horizon_hours=spec.event_horizon_hours, start=start, end=end)
    res.meta["book"] = name
    res.meta["phase"] = phase
    return res


class _Runner:
    """Thin wrapper over serial / thread / process execution with one shared context."""

    def __init__(self, ctx: _Ctx, kind: str, n_jobs: int) -> None:
        self.ctx = ctx
        self.kind = kind
        self.n_jobs = max(1, n_jobs)
        self._proc: cf.ProcessPoolExecutor | None = None
        self._thread: cf.ThreadPoolExecutor | None = None

    def __enter__(self) -> _Runner:
        if self.kind == "process":
            # Workers get the market data only (shipped once per worker by the initializer);
            # feature matrices stay in the parent, where feature-consuming strategies run on
            # threads.
            light = _Ctx(md=self.ctx.md, n_research=self.ctx.n_research)
            self._proc = cf.ProcessPoolExecutor(
                max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(light, logging.getLogger().level or logging.WARNING,
                          max(1, (os.cpu_count() or 1) // self.n_jobs)))
        if self.kind in ("process", "thread"):
            self._thread = cf.ThreadPoolExecutor(max_workers=self.n_jobs, thread_name_prefix="aurum-wf")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            self._proc.shutdown(wait=True, cancel_futures=True)
        if self._thread is not None:
            self._thread.shutdown(wait=True, cancel_futures=True)

    def submit(self, fn: Callable[..., Any], *args: Any, prefer_process: bool = True) -> cf.Future:
        if self._proc is not None and prefer_process:
            return self._proc.submit(fn, *args)
        if self._thread is not None:
            return self._thread.submit(fn, *args, ctx=self.ctx)
        fut: cf.Future = cf.Future()
        try:
            fut.set_result(fn(*args, ctx=self.ctx))
        except BaseException as exc:  # noqa: BLE001 - surfaced through the future
            fut.set_exception(exc)
        return fut


def _resolve_executor(cfg: AurumConfig, n_bars: int, n_tasks: int) -> tuple[str, int]:
    wf = cfg.walkforward
    n_jobs = wf.n_jobs or min(os.cpu_count() or 1, 8)
    n_jobs = max(1, min(n_jobs, max(1, n_tasks)))
    kind = wf.executor
    if kind == "auto":
        if n_jobs <= 1:
            kind = "serial"
        elif n_bars >= _PROCESS_MIN_BARS:
            kind = "process"
        else:
            kind = "thread"
    if kind == "serial":
        n_jobs = 1
    return kind, n_jobs


# ============================================================================================
# combiner
# ============================================================================================
class _FixedWeightCombiner(ForecastCombiner):
    """ForecastCombiner with user-fixed weights (``combiner.method: fixed``).

    Uses the parent's training diagnostics (forecast correlations) and Carver's FDM
    ``1 / sqrt(w' H w)`` with negative correlations floored, capped at ``fdm_cap``.
    """

    def __init__(self, weights: Mapping[str, float], **kwargs: Any) -> None:
        super().__init__(method="equal", **kwargs)
        self.fixed = {str(k): float(v) for k, v in weights.items()}

    def fit(self, forecasts: pd.DataFrame, close: pd.Series, **cost_kwargs: Any) -> _FixedWeightCombiner:
        super().fit(forecasts, close, **cost_kwargs)
        w = np.array([max(0.0, self.fixed.get(str(c), 0.0)) for c in self.columns_], dtype=float)
        if w.sum() <= 0:
            raise ValueError("fixed combiner weights sum to zero")
        w = w / w.sum()
        corr = self.corr_.to_numpy() if self.corr_ is not None else np.eye(len(w))
        h = np.maximum(corr, self.corr_floor)
        np.fill_diagonal(h, 1.0)
        quad = float(w @ h @ w)
        fdm_raw = max(1.0, 1.0 / math.sqrt(quad)) if quad > 0 else 1.0
        self.fdm_raw_ = float(fdm_raw)
        self.fdm_ = float(min(fdm_raw, self.fdm_cap))
        self.weights_ = pd.Series(w, index=self.columns_, name="weight")
        self.method = "fixed"
        return self


def _combiner_options(cfg: AurumConfig) -> dict[str, Any]:
    """Cost-aware combiner knobs (``combiner.allow_unallocated`` / ``combiner.cost_multiplier``)
    for the combiners not built by ``CombinerConfig.build`` (fixed weights, equal fallback)."""
    c = cfg.combiner
    return {"allow_unallocated": bool(c.allow_unallocated), "cost_multiplier": float(c.cost_multiplier)}


def _make_combiner(cfg: AurumConfig, keys: Sequence[str]) -> ForecastCombiner:
    c = cfg.combiner
    if c.method == "fixed":
        weights = {s.key: (1.0 if s.weight is None else s.weight) for s in cfg.strategies}
        weights = {k: weights.get(k, 1.0) for k in keys}
        return _FixedWeightCombiner(weights, shrinkage=c.shrinkage, max_weight=1.0, fdm_cap=c.fdm_cap,
                                    vol_halflife=c.vol_halflife, min_periods=c.min_periods,
                                    corr_floor=c.corr_floor, **_combiner_options(cfg))
    return c.build()


def _cost_kwargs(cfg: AurumConfig, bars: pd.DataFrame | None) -> dict[str, Any]:
    """What :meth:`ForecastCombiner.fit` needs to score strategies NET of trading costs: the
    bars (spread, high/low; the combiner restricts them to the forecasts' own rows), the SAME
    cost model and instrument the backtests use."""
    if bars is None:
        return {}
    return {"bars": bars, "costs": cfg.costs.build(), "instrument": cfg.instrument.build()}


# ============================================================================================
# statistics
# ============================================================================================
def _halt_episodes(res: BacktestResult) -> int:
    ev = res.risk_events
    if ev is None or ev.empty or "halted" not in ev.columns:
        return 0
    h = ev["halted"].astype(bool).to_numpy()
    bars = ev["bar"].to_numpy()
    episodes, prev_bar, prev_h = 0, -10, False
    for b, x in zip(bars, h, strict=True):
        if x and not (prev_h and b == prev_bar + 1):
            episodes += 1
        prev_bar, prev_h = b, x
    return episodes


def book_statistics(result: BacktestResult, *, n_trials: int = 1, trial_sharpes: Sequence[float] | None = None,
                    n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Headline statistics of one book on DAILY returns (SPEC §1).

    * ``sharpe``/``psr``/``ci_*`` from :func:`aurum.research.stats.sharpe_summary` (Pearson
      kurtosis internally; the ``kurtosis`` reported here is Pearson, normal = 3).
    * ``dsr``: Deflated Sharpe Ratio (Bailey & López de Prado 2014) against the expected
      best of ``n_trials`` skill-less strategies, with the null sampling variance of a
      Sharpe estimate (``1/(T-1)``). This is the right null for a set of DISTINCT strategy
      configurations: "could the best of N noise strategies look this good?".
    * ``dsr_xs``: the same with the cross-sectional variance of ``trial_sharpes`` (the
      annualised OOS Sharpes of all trials, when >= 2), as in the paper's parameter-sweep
      setting. It treats genuine dispersion between strategies as noise and is therefore
      (much) more conservative; reported for completeness.

    Drawdown, trade and cost metrics come from ``result.metrics``.
    """
    from aurum.research.stats import deflated_sharpe

    dr = daily_returns(result.equity).dropna()
    n_trials = max(1, int(n_trials))
    summ = sharpe_summary(dr.to_numpy(), 252.0, n_trials=n_trials, n_boot=n_boot, seed=seed) if len(dr) >= 3 else {}
    dsr_xs = math.nan
    if summ and trial_sharpes is not None:
        ts = np.array([float(x) for x in trial_sharpes if x is not None and math.isfinite(float(x))])
        if ts.size >= 2 and math.isfinite(summ.get("sharpe_per_period", math.nan)):
            dsr_xs = deflated_sharpe(summ["sharpe_per_period"], len(dr), summ["skew"], summ["kurtosis"],
                                     trial_srs=ts / math.sqrt(252.0), n_trials=max(n_trials, ts.size))
    m = result.metrics or {}
    ev = result.risk_events
    out = {
        "sharpe": summ.get("sharpe"), "psr": summ.get("psr"), "dsr": summ.get("dsr"), "dsr_xs": dsr_xs,
        "sr0": summ.get("sr0"), "sharpe_se": summ.get("sharpe_se"), "ci_lower": summ.get("ci_lower"),
        "ci_upper": summ.get("ci_upper"), "skew": summ.get("skew"), "kurtosis": summ.get("kurtosis"),
        "min_trl_years": summ.get("min_trl_years"),
        "n_days": len(dr), "total_return": m.get("total_return"), "cagr": m.get("cagr"),
        "ann_vol": m.get("ann_vol"), "sortino": m.get("sortino"), "max_drawdown": m.get("max_drawdown"),
        "calmar": m.get("calmar"), "n_trades": m.get("n_trades"), "win_rate": m.get("win_rate"),
        "exposure": m.get("exposure"), "turnover_lots_per_year": m.get("turnover_lots_per_year"),
        "total_costs": m.get("total_costs"), "cost_drag_ann": m.get("cost_drag_ann"),
        "swap_total": m.get("swap_total"),
        "n_risk_events": 0 if ev is None else len(ev),
        "n_halt_bars": 0 if ev is None or ev.empty else int(ev["halted"].astype(bool).sum()),
        "n_halt_episodes": _halt_episodes(result),
        "n_trials": n_trials,
    }
    return {k: (math.nan if v is None else v) for k, v in out.items()}


def cost_attribution(result: BacktestResult) -> dict[str, float]:
    """Gross (mid-price) PnL vs spread, slippage, commission and swap (USD)."""
    c = result.costs
    rec = {k: float(c[k].sum()) if k in c.columns else 0.0 for k in ("spread", "slippage", "commission", "swap")}
    total = rec["spread"] + rec["slippage"] + rec["commission"]
    gross = float(result.pnl["price"].sum()) if result.pnl is not None and "price" in result.pnl else math.nan
    net = float(result.equity.iloc[-1] - result.equity.iloc[0]) if len(result.equity) else 0.0
    rec.update({"total_costs": total, "gross_pnl": gross, "net_pnl": net,
                "costs_pct_gross": total / abs(gross) if gross and math.isfinite(gross) else math.nan})
    return rec


def _window_sharpe(equity: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[float, float]:
    """(annualised daily Sharpe, total return) of the decisions made at the closes of [t0, t1].

    A decision taken at the close of bar ``t`` fills at the open of ``t+1`` and earns its PnL
    over bar ``t+1`` (SPEC §1). A fold's PnL therefore runs from the equity marked at the
    close of ``t0`` (before any of the fold's own decisions have paid off) to the close of
    the bar AFTER ``t1`` (when it exists: the last bar of a backtest has no executed
    decision). Consecutive folds chain exactly, and the first bar of a fold is no longer
    credited with the previous fold's last decision.
    """
    idx = equity.index
    i0 = int(idx.searchsorted(t0, side="left"))
    i1 = int(idx.searchsorted(t1, side="right"))   # first position after t1
    if i0 >= len(idx):
        return math.nan, math.nan
    base = float(equity.iloc[i0])
    seg = equity.iloc[i0 + 1:min(i1 + 1, len(idx))]
    if len(seg) < 1 or not base:
        return math.nan, math.nan
    dr = daily_returns(seg, initial=base).dropna()
    sr = sharpe(dr.to_numpy(), 252.0) if len(dr) >= 5 else math.nan
    return float(sr), float(seg.iloc[-1] / base - 1.0)


def _pbo(matrix: pd.DataFrame, n_splits: int) -> PBOResult | None:
    if matrix.shape[1] < 2:
        return None
    s = int(n_splits) - int(n_splits) % 2
    while s >= 2 and matrix.shape[0] < 2 * s:
        s -= 2
    if s < 2:
        return None
    return pbo_cscv(matrix.to_numpy(), n_splits=s)


# ============================================================================================
# provenance
# ============================================================================================
def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(cfg: AurumConfig | None, md: MarketData | None) -> dict[str, Any]:
    """Run provenance (SPEC §0.4): config hash, data hashes, git SHA, package versions."""
    from importlib import metadata

    from aurum.data.store import frame_hash

    pkgs = {}
    for name in ("aurum", "numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib", "pyyaml",
                 "anthropic", "torch", "stable-baselines3", "gymnasium"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    out: dict[str, Any] = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": sha, "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0], "platform": platform.platform(terse=True),
        "packages": pkgs, "argv": list(sys.argv),
    }
    if cfg is not None:
        out["config_hash"] = cfg.config_hash()
        out["config_source"] = cfg.source
    if md is not None:
        b = md.bars
        out["data"] = {
            "bars_hash": frame_hash(b), "n_bars": len(b), "first_bar": str(b.index[0]), "last_bar": str(b.index[-1]),
            "timeframe": b.attrs.get("timeframe"),
            "macro_hashes": {k: frame_hash(v) for k, v in sorted((md.macro or {}).items())},
            "events_hash": frame_hash(md.events.reset_index(drop=True)) if md.events is not None else None,
        }
    return out


# ============================================================================================
# holdout ledger
# ============================================================================================
#: file name of the holdout ledger, kept in the ROOT of the output directory (the parent of the
#: run directories, e.g. ``runs/holdout_ledger.jsonl``)
HOLDOUT_LEDGER = "holdout_ledger.jsonl"


def read_holdout_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Entries of a holdout ledger (JSONL). Missing file -> []; unreadable lines are skipped
    with a warning (a corrupt line must not hide the others)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            logger.warning("holdout ledger %s: line %d is not JSON; skipped", p, i)
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def prior_holdout_looks(entries: Sequence[Mapping[str, Any]], *, start: Any, end: Any, config_hash: str,
                        symbol: str | None = None) -> dict[str, Any]:
    """Earlier evaluations of a holdout window overlapping ``[start, end]`` (same symbol).

    ``n_prior_looks`` counts every overlapping entry; ``n_prior_configs`` the DISTINCT config
    hashes other than ``config_hash`` among them — re-running the same (deterministic) config
    is not a new trial, evaluating a different one is (it is a selection opportunity).
    """
    s, e = _utc(start), _utc(end)
    looks = []
    for rec in entries:
        try:
            rs, re_ = _utc(rec["holdout_start"]), _utc(rec["holdout_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if symbol is not None and rec.get("symbol") not in (None, symbol):
            continue
        if rs <= e and s <= re_:
            looks.append(rec)
    others = sorted({str(r.get("config_hash")) for r in looks if r.get("config_hash") != config_hash})
    return {"n_prior_looks": len(looks), "n_prior_configs": len(others), "prior_config_hashes": others,
            "n_prior_same_config": sum(1 for r in looks if r.get("config_hash") == config_hash),
            "first_look": min((str(r.get("timestamp")) for r in looks), default=None)}


def append_holdout_ledger(path: str | Path, entry: Mapping[str, Any]) -> Path:
    """Append one JSON line (flushed and fsynced: the ledger is an audit trail)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(dict(entry)), sort_keys=True, default=str)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return p


def _ledger_entry(report: WalkForwardReport, cfg: AurumConfig, run_dir: Path | None) -> dict[str, Any]:
    h = report.holdout
    assert h is not None
    stats = h.stats
    keep = ("sharpe", "psr", "dsr", "total_return", "cagr", "ann_vol", "max_drawdown", "n_trades", "total_costs",
            "n_days")
    comb = {k: stats.loc[_COMBINED, k] for k in keep if _COMBINED in stats.index and k in stats.columns}
    per = {str(k): stats.loc[k, "sharpe"] for k in stats.index if k != _COMBINED and "sharpe" in stats.columns}
    prov = report.provenance or {}
    return {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "kind": report.kind,
        "name": cfg.name, "config_hash": report.config_hash, "data_hash": report.data_hash,
        "symbol": cfg.data.symbol, "timeframe": cfg.data.timeframe,
        "holdout_start": h.start, "holdout_end": h.end, "n_holdout_bars": len(h.combined.equity),
        "strategies": sorted(report.strategy_results) or [c for c in report.oos_forecasts.columns
                                                          if c not in (_COMBINED, "fold")],
        "n_trials": report.n_trials, "prior_looks": h.prior_looks,
        "metrics": {"combined": comb, "strategy_sharpe": per},
        "run_dir": run_dir, "git_sha": prov.get("git_sha"),
    }


# ============================================================================================
# reports
# ============================================================================================
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj))
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.reset_index().to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or "book"


@dataclass
class HoldoutReport:
    """Final holdout evaluated ONCE with the model fitted on the last training window."""

    start: pd.Timestamp
    end: pd.Timestamp
    combined: BacktestResult
    strategy_results: dict[str, BacktestResult]
    benchmark: BacktestResult | None
    stats: pd.DataFrame
    weights: pd.Series
    fdm: float
    costs: pd.DataFrame
    combiner_basis: str = ""
    forecasts: pd.DataFrame | None = None       # holdout OOS forecasts (strategies + combined)
    prior_looks: dict[str, Any] = field(default_factory=dict)   # holdout ledger (earlier evaluations)
    ledger_path: Path | None = None

    def summary(self) -> dict[str, Any]:
        return _jsonable({"start": self.start, "end": self.end, "stats": self.stats.to_dict(orient="index"),
                          "weights": self.weights, "fdm": self.fdm, "combiner_basis": self.combiner_basis,
                          "costs": self.costs.to_dict(orient="index"), "prior_looks": self.prior_looks,
                          "ledger": self.ledger_path})


@dataclass
class WalkForwardReport:
    """Everything a walk-forward (or single-split backtest) produced."""

    kind: str                                   # "walkforward" | "backtest"
    config: dict[str, Any]
    config_hash: str
    provenance: dict[str, Any]
    settings: dict[str, Any]                    # resolved bar counts, purge, embargo, ...
    folds: pd.DataFrame
    weights: pd.DataFrame
    fold_sharpe: pd.DataFrame
    oos_forecasts: pd.DataFrame
    combined: BacktestResult
    strategy_results: dict[str, BacktestResult]
    benchmark: BacktestResult | None
    stats: pd.DataFrame
    costs: pd.DataFrame
    pbo: PBOResult | None
    n_trials: int
    holdout: HoldoutReport | None
    timing: dict[str, float]
    notes: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    out_dir: Path | None = None

    @property
    def data_hash(self) -> str | None:
        return (self.provenance.get("data") or {}).get("bars_hash")

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary (what ``summary.json`` holds)."""
        oos = self.oos_forecasts.index
        return _jsonable({
            "kind": self.kind,
            "name": self.config.get("name"),
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "oos_start": oos[0] if len(oos) else None, "oos_end": oos[-1] if len(oos) else None,
            "n_oos_bars": len(oos),
            "settings": self.settings,
            "n_folds": len(self.folds),
            "n_strategies": len(self.strategy_results) or self.oos_forecasts.shape[1] - 2,
            "n_trials": self.n_trials,
            "combined": self.stats.loc[_COMBINED].to_dict() if _COMBINED in self.stats.index else {},
            "stats": self.stats.to_dict(orient="index"),
            "pbo": self.pbo.to_dict() if self.pbo is not None else None,
            "folds": self.folds,
            "weights": self.weights,
            "costs": self.costs.to_dict(orient="index"),
            "holdout": self.holdout.summary() if self.holdout is not None else None,
            "timing": self.timing,
            "notes": self.notes,
            "dropped": self.dropped,
            "provenance": self.provenance,
            "out_dir": self.out_dir,
        })

    def save(self, directory: str | Path, *, save_books: bool = True, tearsheet: bool = True,
             dark_charts: bool = True) -> Path:
        """Write the run directory (see module docs of :mod:`aurum.cli` ``report``)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        import yaml

        (d / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        (d / "provenance.json").write_text(json.dumps(_jsonable(self.provenance), indent=2), encoding="utf-8")
        self.folds.to_csv(d / "folds.csv", index=False)
        self.weights.to_csv(d / "weights.csv")
        self.fold_sharpe.to_csv(d / "fold_sharpe.csv")
        self.stats.to_csv(d / "stats.csv")
        self.costs.to_csv(d / "costs.csv")
        self.oos_forecasts.to_parquet(d / "oos_forecasts.parquet")
        if save_books:
            self.combined.save(d / "books" / _COMBINED)
            for k, r in self.strategy_results.items():
                r.save(d / "books" / _safe_name(k))
            if self.benchmark is not None:
                self.benchmark.save(d / "books" / _BENCHMARK)
        if tearsheet:
            self._tearsheet(d / "tearsheet.html", self.combined, self.benchmark, dark_charts,
                            title=f"{self.config.get('name', 'aurum')} - {self.kind} OOS (combined book)")
        if self.holdout is not None:
            h = d / "holdout"
            h.mkdir(exist_ok=True)
            self.holdout.stats.to_csv(h / "stats.csv")
            self.holdout.costs.to_csv(h / "costs.csv")
            if self.holdout.forecasts is not None:
                self.holdout.forecasts.to_parquet(h / "oos_forecasts.parquet")
            (h / "holdout.json").write_text(json.dumps(self.holdout.summary(), indent=2), encoding="utf-8")
            if save_books:
                self.holdout.combined.save(h / "books" / _COMBINED)
                if self.holdout.benchmark is not None:
                    self.holdout.benchmark.save(h / "books" / _BENCHMARK)
            if tearsheet:
                self._tearsheet(h / "tearsheet.html", self.holdout.combined, self.holdout.benchmark, dark_charts,
                                title=f"{self.config.get('name', 'aurum')} - FINAL HOLDOUT (evaluated once)",
                                holdout=True)
        self.out_dir = d
        self.timing["write_s"] = round(time.perf_counter() - t0, 3)
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return d

    def _tearsheet(self, path: Path, result: BacktestResult, bench: BacktestResult | None, dark: bool,
                   *, title: str, holdout: bool = False) -> None:
        from aurum.research.report import write_tearsheet

        if holdout and self.holdout is not None:
            hnotes = ["Final holdout: fitted on the last training window before holdout_start and evaluated "
                      "exactly once."]
            pl = self.holdout.prior_looks or {}
            if pl.get("n_prior_configs"):
                hnotes.append(f"WARNING: this holdout window was evaluated before by {pl['n_prior_configs']} other "
                              f"config(s) ({pl.get('n_prior_looks')} prior look(s) in the holdout ledger): it is "
                              f"no longer untouched; DSR uses n_trials={self.n_trials}.")
            extra: dict[str, Any] = {"weights": self.holdout.weights, "n_trials": self.n_trials,
                                     "holdout_stats": self.holdout.stats, "costs_by_book": self.holdout.costs,
                                     "notes": hnotes}
        else:
            extra = {"folds": self.folds, "weights": self.weights, "n_trials": self.n_trials,
                     "strategy_stats": self.stats,
                     "fold_sharpe_by_strategy": self.fold_sharpe, "costs_by_book": self.costs,
                     "notes": list(self.notes), "config": self.config}
            if self.pbo is not None:
                extra["pbo"] = self.pbo
        write_tearsheet(result, path, benchmark=bench.equity if bench is not None else None, title=title,
                        extra=extra, dark_charts=dark)


# ============================================================================================
# the evaluation core (shared by walk-forward and single-split backtests)
# ============================================================================================
def _book_spec(cfg: AurumConfig) -> _BookSpec:
    b = cfg.backtest
    return _BookSpec(sizer=cfg.sizing.build(), risk=cfg.risk if cfg.risk.enabled else None,
                     instrument=cfg.instrument.build(), costs=cfg.costs.build(),
                     initial_equity=b.initial_equity, stop_atr_mult=b.stop_atr_mult,
                     take_profit_atr_mult=b.take_profit_atr_mult, atr_period=b.atr_period,
                     stop_cooldown_bars=b.stop_cooldown_bars, event_horizon_hours=b.event_horizon_hours)


def _unique_keys(strategies: Sequence[Strategy]) -> dict[str, Strategy]:
    out: dict[str, Strategy] = {}
    for s in strategies:
        k = s.name
        i = 2
        while k in out:
            k = f"{s.name}#{i}"
            i += 1
        out[k] = s
    return out


@dataclass
class _Evaluated:
    plans: list[FoldPlan]
    holdout: FoldPlan | None
    fc: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]     # strategy -> fold -> (train, test)
    timing: dict[str, dict[str, Any]]
    dropped: dict[str, str]


def _evaluate(ctx: _Ctx, cfg: AurumConfig, strategies: dict[str, Strategy], plans: list[FoldPlan],
              holdout: FoldPlan | None, runner: _Runner, timing: dict[str, float]) -> _Evaluated:
    """Features (parent) + per-strategy fold work (pool)."""
    all_plans = plans + ([holdout] if holdout is not None else [])
    need = {k: _needs_features(s, cfg.features.enabled) for k, s in strategies.items()}
    t0 = time.perf_counter()
    if any(need.values()):
        phases = sorted({p.phase for p in all_plans})
        for phase in phases:
            pipe = cfg.features.build()
            ctx.raw[phase] = pipe.compute(_phase_md(ctx, phase))
        for p in all_plans:
            pipe = cfg.features.build()
            raw = ctx.raw[p.phase]
            pipe.fit(raw.iloc[p.train_start:p.train_end])
            ctx.pipes[p.key] = pipe
        logger.info("features: %d columns computed, %d fold pipelines fitted",
                    ctx.raw[phases[0]].shape[1], len(ctx.pipes))
    timing["features_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    regen = cfg.walkforward.regenerate_per_fold
    futures: dict[cf.Future, str] = {}
    # Trainable / feature-consuming strategies: one task per (strategy, fold) so the costly
    # fits load-balance across workers; causal rule-based strategies: one task per strategy
    # (generated once per phase and sliced). Feature consumers run on parent threads.
    for k, s in strategies.items():
        per_fold = bool(s.trainable or need[k] or regen)
        chunks = [[p] for p in all_plans] if per_fold else [all_plans]
        for chunk in chunks:
            futures[runner.submit(_strategy_task, k, s, chunk, need[k], regen, prefer_process=not need[k])] = k
    fc: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    st_timing: dict[str, dict[str, Any]] = {}
    dropped: dict[str, str] = {}
    for fut in cf.as_completed(futures):
        k = futures[fut]
        if k in dropped:
            continue
        try:
            res = fut.result()
        except Exception as exc:
            if cfg.walkforward.on_strategy_error == "raise":
                for other in futures:
                    other.cancel()
                raise RuntimeError(f"strategy {k!r} failed during walk-forward: {exc}") from exc
            logger.error("strategy %r dropped: %s", k, exc)
            dropped[k] = f"{type(exc).__name__}: {exc}"
            fc.pop(k, None)
            continue
        fc.setdefault(k, {}).update(res["folds"])
        t = st_timing.setdefault(k, {"fit_s": 0.0, "gen_s": 0.0, "per_fold": res["per_fold"], "tasks": 0})
        t["fit_s"] = round(t["fit_s"] + res["fit_s"], 3)
        t["gen_s"] = round(t["gen_s"] + res["gen_s"], 3)
        t["tasks"] += 1
    timing["strategies_s"] = round(time.perf_counter() - t0, 3)
    for k in dropped:
        fc.pop(k, None)
        st_timing.pop(k, None)
    if not fc:
        raise RuntimeError("every strategy failed; nothing to evaluate")
    missing = {k: [p.key for p in all_plans if p.key not in f] for k, f in fc.items()}
    if any(missing.values()):  # pragma: no cover - defensive
        raise AssertionError(f"missing fold results: {missing}")
    return _Evaluated(plans, holdout, fc, st_timing, dropped)


def _fit_fold_combiner(cfg: AurumConfig, keys: list[str], f_train: pd.DataFrame, close: pd.Series,
                       history: pd.DataFrame | None, window: int | None, *, bars: pd.DataFrame | None = None,
                       combiner_fit: str | None = None) -> tuple[ForecastCombiner, str]:
    """Combiner for one fold and a description of what it was fitted on.

    ``combiner_fit="oos"``: fit on ``history`` (stitched OOS forecasts of EARLIER folds,
    genuinely out-of-sample for every strategy - weights estimated from past forecast
    performance, Timmermann 2006, *Handbook of Economic Forecasting* ch. 4), restricted to the
    last ``window`` bars; with fewer than ``combiner_min_obs`` rows, EQUAL weights (DeMiguel,
    Garlappi & Uppal 2009) with activity/FDM from the training-window forecast correlations
    (no returns involved). ``"train"``: fit on the training-window forecasts (in-sample for
    trainable strategies). ``method="fixed"`` ignores returns either way.

    With ``bars`` the strategies are scored NET of estimated trading costs (spread/2 +
    slippage + commission per unit of forecast turnover, see :mod:`aurum.portfolio.combiner`);
    the combiner restricts ``bars`` to the rows it is fitted on. ``combiner_fit`` overrides
    ``walkforward.combiner_fit``.
    """
    wf = cfg.walkforward
    mode = combiner_fit or wf.combiner_fit
    kw = _cost_kwargs(cfg, bars)
    comb = _make_combiner(cfg, keys)
    if cfg.combiner.method == "fixed" or mode == "train":
        comb.fit(f_train, close.reindex(f_train.index), **kw)
        return comb, "train_window"
    h = history
    if h is not None and window is not None and len(h) > window:
        h = h.iloc[-window:]
    why = f"OOS history {0 if h is None else len(h)} < {wf.combiner_min_obs} bars"
    if h is not None and len(h) >= wf.combiner_min_obs:
        try:
            comb.fit(h[keys], close.reindex(h.index), **kw)
            return comb, f"oos_history({len(h)} bars)"
        except ValueError as exc:  # e.g. too few valid rows after warm-up
            logger.warning("combiner OOS fit failed (%s); equal weights", exc)
            why = f"OOS fit failed: {exc}"
    c = cfg.combiner
    eq = ForecastCombiner(method="equal", shrinkage=c.shrinkage, max_weight=c.max_weight, fdm_cap=c.fdm_cap,
                          vol_halflife=c.vol_halflife, min_periods=c.min_periods, corr_floor=c.corr_floor,
                          **_combiner_options(cfg))
    eq.fit(f_train, close.reindex(f_train.index), **kw)
    return eq, f"equal ({why})"


def _combine_folds(ctx: _Ctx, cfg: AurumConfig, ev: _Evaluated, plans: Sequence[FoldPlan], keys: list[str],
                   *, history: pd.DataFrame | None = None, window: int | None = None
                   ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, ForecastCombiner]]:
    """Fit the combiner per fold (see :func:`_fit_fold_combiner`); stitch OOS test forecasts.

    Folds are processed in chronological order; each fold's test forecasts join the OOS
    history available to LATER folds only. ``history`` seeds it (e.g. all research OOS for
    the holdout).
    """
    idx = ctx.md.bars.index
    close = ctx.md.bars["close"]
    parts: list[pd.DataFrame] = []
    fold_info: list[dict[str, Any]] = []
    combiners: dict[str, ForecastCombiner] = {}
    hist_parts: list[pd.DataFrame] = [] if history is None else [history[keys]]
    for p in plans:
        tr_idx = idx[p.train_start:p.train_end]
        te_idx = idx[p.test_start:p.test_end]
        f_train = pd.DataFrame({k: ev.fc[k][p.key][0] for k in keys}, index=tr_idx)
        f_test = pd.DataFrame({k: ev.fc[k][p.key][1] for k in keys}, index=te_idx)
        h = pd.concat(hist_parts) if hist_parts else None
        if h is not None:
            h = h.loc[h.index < te_idx[0]]  # defensive: strictly before this fold's test block
        comb, basis = _fit_fold_combiner(cfg, keys, f_train, close, h, window, bars=ctx.md.bars)
        f_test[_COMBINED] = comb.combine(f_test[keys]).to_numpy()
        hist_parts.append(f_test[keys])
        f_test["fold"] = p.key
        parts.append(f_test)
        combiners[p.key] = comb
        fold_info.append({"fold": p.key, "fdm": float(comb.fdm_), "weights": comb.weights_.copy(),
                          "notes": list(comb.notes_), "basis": basis})
    oos = pd.concat(parts)
    if not oos.index.is_monotonic_increasing or oos.index.has_duplicates:
        raise AssertionError("stitched OOS index is not strictly increasing (overlapping test blocks)")
    return oos, fold_info, combiners


def _run_books(runner: _Runner, spec: _BookSpec, phase: str, forecasts: pd.DataFrame, keys: Sequence[str],
               start: pd.Timestamp, end: pd.Timestamp, *, strategies: bool, benchmark: bool
               ) -> tuple[BacktestResult, dict[str, BacktestResult], BacktestResult | None]:
    jobs: dict[str, cf.Future] = {}
    jobs[_COMBINED] = runner.submit(_book_task, _COMBINED, phase, forecasts[_COMBINED], start, end, spec)
    if strategies:
        for k in keys:
            jobs[k] = runner.submit(_book_task, k, phase, forecasts[k], start, end, spec)
    if benchmark:
        jobs[_BENCHMARK] = runner.submit(_book_task, _BENCHMARK, phase, None, start, end, spec)
    results = {k: f.result() for k, f in jobs.items()}
    comb = results.pop(_COMBINED)
    bench = results.pop(_BENCHMARK, None)
    return comb, {k: results[k] for k in keys if k in results}, bench


def _stats_table(combined: BacktestResult, books: Mapping[str, BacktestResult], bench: BacktestResult | None,
                 n_trials: int, n_boot: int, seed: int) -> pd.DataFrame:
    trial = []
    for r in books.values():
        dr = daily_returns(r.equity).dropna()
        trial.append(sharpe(dr.to_numpy(), 252.0) if len(dr) >= 3 else math.nan)
    rows = {_COMBINED: book_statistics(combined, n_trials=n_trials, trial_sharpes=trial, n_boot=n_boot, seed=seed)}
    for k, r in books.items():
        rows[k] = book_statistics(r, n_trials=n_trials, trial_sharpes=trial, n_boot=n_boot, seed=seed)
    if bench is not None:
        rows[_BENCHMARK] = book_statistics(bench, n_trials=1, n_boot=n_boot, seed=seed)
    return pd.DataFrame.from_dict(rows, orient="index")


def _costs_table(combined: BacktestResult, books: Mapping[str, BacktestResult], bench: BacktestResult | None
                 ) -> pd.DataFrame:
    rows = {_COMBINED: cost_attribution(combined)}
    rows.update({k: cost_attribution(r) for k, r in books.items()})
    if bench is not None:
        rows[_BENCHMARK] = cost_attribution(bench)
    return pd.DataFrame.from_dict(rows, orient="index")


def _resolve_strategies(cfg: AurumConfig, strategies: Sequence[Strategy] | Mapping[str, Strategy] | None,
                        only: Sequence[str] | None) -> dict[str, Strategy]:
    if strategies is None:
        out = cfg.build_strategies(only)
    elif isinstance(strategies, Mapping):
        out = dict(strategies)
    else:
        out = _unique_keys(list(strategies))
    if not out:
        raise ConfigError("no enabled strategies to evaluate (config 'strategies' is empty)")
    return out


def _run(md: MarketData, cfg: AurumConfig, strategies: dict[str, Strategy], plans: list[FoldPlan],
         holdout: FoldPlan | None, settings: dict[str, Any], *, kind: str, notes: list[str],
         extra_trials: int = 0) -> WalkForwardReport:
    t_all = time.perf_counter()
    timing: dict[str, float] = {}
    wf = cfg.walkforward
    n_research = settings["n_research"]
    ctx = _Ctx(md=md, n_research=n_research)
    n_tasks = sum(len(plans) + (holdout is not None) if s.trainable else 1 for s in strategies.values())
    exec_kind, n_jobs = _resolve_executor(cfg, n_research, max(n_tasks, len(strategies) + 2))
    logger.info("%s: %d strategies, %d folds, executor=%s(%d)", kind, len(strategies), len(plans), exec_kind, n_jobs)
    spec = _book_spec(cfg)
    # DSR trials: strategy configurations evaluated in THIS run (one config x N strategies, or
    # walkforward.n_trials) + configurations evaluated before on the same holdout window
    # (holdout ledger): every earlier look is a selection opportunity the DSR must deflate.
    n_trials_base = int(wf.n_trials or len(strategies))
    n_trials = n_trials_base + max(0, int(extra_trials))
    settings = {**settings, "executor": exec_kind, "n_jobs": n_jobs, "n_trials_base": n_trials_base,
                "n_trials_prior_looks": max(0, int(extra_trials))}
    with _Runner(ctx, exec_kind, n_jobs) as runner:
        ev = _evaluate(ctx, cfg, strategies, plans, holdout, runner, timing)
        keys = [k for k in strategies if k in ev.fc]

        t0 = time.perf_counter()
        window = None if wf.anchored else int(settings["train"])
        oos, fold_info, _ = _combine_folds(ctx, cfg, ev, plans, keys, window=window)
        timing["combine_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        start, end = oos.index[0], oos.index[-1]
        comb, books, bench = _run_books(runner, spec, "research", oos, keys, start, end,
                                        strategies=wf.strategy_backtests, benchmark=cfg.backtest.benchmark)
        timing["backtests_s"] = round(time.perf_counter() - t0, 3)

        hold_report = None
        if holdout is not None:
            t0 = time.perf_counter()
            h_fc, h_info, _ = _combine_folds(ctx, cfg, ev, [holdout], keys, history=oos, window=window)
            hs, he = h_fc.index[0], h_fc.index[-1]
            h_comb, h_books, h_bench = _run_books(runner, spec, "holdout", h_fc, keys, hs, he,
                                                  strategies=wf.strategy_backtests, benchmark=cfg.backtest.benchmark)
            h_stats = _stats_table(h_comb, h_books, h_bench, n_trials, wf.n_boot, cfg.seed)
            hold_report = HoldoutReport(start=hs, end=he, combined=h_comb, strategy_results=h_books,
                                        benchmark=h_bench, stats=h_stats, weights=h_info[0]["weights"],
                                        fdm=h_info[0]["fdm"], costs=_costs_table(h_comb, h_books, h_bench),
                                        combiner_basis=h_info[0]["basis"],
                                        forecasts=h_fc.drop(columns="fold", errors="ignore"))
            timing["holdout_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    stats = _stats_table(comb, books, bench, n_trials, wf.n_boot, cfg.seed)
    matrix = pd.DataFrame({k: daily_returns(r.equity) for k, r in books.items()}).dropna()
    pbo = _pbo(matrix, wf.pbo_splits) if len(books) >= 2 else None
    idx = md.bars.index
    fold_rows, fs_rows = [], {}
    for p, info in zip(plans, fold_info, strict=True):
        t0_, t1_ = idx[p.test_start], idx[p.test_end - 1]
        sr, ret = _window_sharpe(comb.equity, t0_, t1_)
        w = info["weights"]
        fold_rows.append({
            "fold": p.key, "train_start": idx[p.train_start], "train_end": idx[p.train_end - 1],
            "test_start": t0_, "test_end": t1_, "n_train": p.n_train, "n_test": p.n_test,
            "gap_bars": p.test_start - p.train_end, "oos_sharpe": sr, "oos_return": ret,
            "fdm": info["fdm"], "n_active": int((w > 1e-12).sum()),
            "top_weight": f"{w.idxmax()}={w.max():.2f}" if len(w) else "", "combiner_basis": info["basis"],
        })
        fs_rows[p.key] = {k: _window_sharpe(r.equity, t0_, t1_)[0] for k, r in books.items()}
    folds = pd.DataFrame(fold_rows)
    fold_sharpe = pd.DataFrame.from_dict(fs_rows, orient="index")
    fold_sharpe.index.name = "fold"
    weights = pd.DataFrame({i["fold"]: i["weights"] for i in fold_info}).T
    weights["fdm"] = [i["fdm"] for i in fold_info]
    weights.index.name = "fold"
    costs = _costs_table(comb, books, bench)
    timing["stats_s"] = round(time.perf_counter() - t0, 3)
    timing["strategy_detail"] = ev.timing  # type: ignore[assignment]
    timing["total_s"] = round(time.perf_counter() - t_all, 3)

    if settings.get("stitched_gaps"):
        notes.append("step > test: bars between test blocks are not OOS and were traded flat")
    for info in fold_info:
        for n_ in info["notes"]:
            notes.append(f"fold {info['fold']} combiner: {n_}")
    n_halts = int(stats.loc[_COMBINED, "n_halt_episodes"]) if "n_halt_episodes" in stats else 0
    if n_halts:
        notes.append(f"combined book: {n_halts} risk halt episode(s) (research limits; see risk_events.csv)")
    if ev.dropped:
        notes.append(f"dropped strategies (errors): {sorted(ev.dropped)}")

    report = WalkForwardReport(
        kind=kind, config=cfg.to_dict(), config_hash=cfg.config_hash(), provenance=provenance(cfg, md),
        settings=settings, folds=folds, weights=weights, fold_sharpe=fold_sharpe, oos_forecasts=oos,
        combined=comb, strategy_results=books, benchmark=bench, stats=stats, costs=costs, pbo=pbo,
        n_trials=n_trials, holdout=hold_report, timing=timing, notes=notes, dropped=ev.dropped,
    )
    return report


def _default_out_dir(cfg: AurumConfig, kind: str) -> Path:
    o = cfg.output
    if o.run_name:
        return Path(o.dir) / o.run_name
    base = Path(o.dir) / f"{kind}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}_{cfg.config_hash()[:8]}"
    d, i = base, 2
    while d.exists():   # two runs of the same config within one second must not overwrite each other
        d = base.with_name(f"{base.name}_{i}")
        i += 1
    return d


def run_walk_forward(md: MarketData, config: AurumConfig, *,
                     strategies: Sequence[Strategy] | Mapping[str, Strategy] | None = None,
                     out_dir: str | Path | None = None, write: bool | None = None,
                     holdout_ledger: str | Path | bool | None = None) -> WalkForwardReport:
    """Run the full walk-forward protocol (module docstring) and return the report.

    Parameters
    ----------
    md       : market data (bars + macro + events). Bars from ``walkforward.holdout_start``
               on form the final holdout.
    config   : validated :class:`AurumConfig`.
    strategies : optional strategy instances (or ``{id: instance}``) instead of building
               them from ``config.strategies`` via the registry.
    out_dir  : run directory (default ``output.dir/walkforward_<utc>_<hash8>``).
    write    : write artefacts (default ``output.save_results``).
    holdout_ledger : where holdout evaluations are recorded (JSONL, :data:`HOLDOUT_LEDGER`).
               ``None`` (default): ``<run dir>/../holdout_ledger.jsonl`` when the run is written
               (i.e. ``output.dir/holdout_ledger.jsonl`` for default run directories), not
               recorded otherwise (noted in the report); a path: that file; ``False``: off.
               Before the holdout is evaluated the ledger is read: earlier evaluations of an
               overlapping window by OTHER configs trigger a WARNING + report note and are
               added to the DSR ``n_trials`` (``settings["n_trials_prior_looks"]``).
    """
    t0 = time.perf_counter()
    strats = _resolve_strategies(config, strategies, None)
    idx = pd.DatetimeIndex(md.bars.index)
    plans, holdout, settings = plan_folds(idx, config, strats)
    write = config.output.save_results if write is None else write
    run_dir = (Path(out_dir) if out_dir is not None else _default_out_dir(config, "walkforward")) if write else None
    notes: list[str] = []
    ledger: Path | None = None
    prior: dict[str, Any] = {}
    if holdout is not None:
        if isinstance(holdout_ledger, (str, Path)):
            ledger = Path(holdout_ledger)
        elif holdout_ledger is not False:
            # default location: the ROOT of the output directory, next to the run directories
            root = run_dir.parent if run_dir is not None else (Path(config.output.dir) if holdout_ledger else None)
            ledger = root / HOLDOUT_LEDGER if root is not None else None
        if ledger is not None:
            prior = prior_holdout_looks(read_holdout_ledger(ledger), start=idx[holdout.test_start],
                                        end=idx[holdout.test_end - 1], config_hash=config.config_hash(),
                                        symbol=config.data.symbol)
            if prior["n_prior_configs"]:
                msg = (f"the holdout window {idx[holdout.test_start]} -> {idx[holdout.test_end - 1]} was "
                       f"evaluated before by {prior['n_prior_configs']} other config(s) "
                       f"({prior['n_prior_looks']} prior look(s) since {prior['first_look']}, ledger {ledger}): it is "
                       f"no longer an untouched holdout; DSR n_trials is raised by {prior['n_prior_configs']}")
                logger.warning(msg)
                notes.append("WARNING: " + msg)
        else:
            notes.append("holdout evaluation NOT recorded in a holdout ledger (results not written; pass "
                         "holdout_ledger=PATH to record it)")
    report = _run(md, config, strats, plans, holdout, settings, kind="walkforward", notes=notes,
                  extra_trials=int(prior.get("n_prior_configs", 0)))
    if report.holdout is not None:
        report.holdout.prior_looks = prior
        report.holdout.ledger_path = ledger
    if (config.combiner.method != "fixed" and config.walkforward.combiner_fit == "train"
            and any(getattr(s, "trainable", False) for s in strats.values())):
        report.notes.append("combiner_fit=train: combiner weights were fitted on IN-SAMPLE forecasts of "
                            "trainable strategies (biased toward them); combiner_fit=oos avoids this")
    n_eq = sum(1 for f in report.folds["combiner_basis"] if str(f).startswith("equal"))
    if n_eq:
        report.notes.append(f"{n_eq} fold(s) used equal combiner weights (not enough earlier OOS history)")
    if run_dir is not None:
        report.save(run_dir, tearsheet=config.output.tearsheet, dark_charts=config.output.dark_charts)
    if report.holdout is not None and ledger is not None:
        # recorded AFTER the evaluation (and the write): the ledger lists looks that happened
        append_holdout_ledger(ledger, _ledger_entry(report, config, report.out_dir))
        logger.info("holdout evaluation recorded in %s", ledger)
    report.timing["wall_s"] = round(time.perf_counter() - t0, 3)
    if report.out_dir is not None:
        (report.out_dir / "summary.json").write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    logger.info("walk-forward done in %.1fs: combined OOS Sharpe %s", report.timing["wall_s"],
                report.stats.loc[_COMBINED, "sharpe"])
    return report


def run_single_backtest(md: MarketData, config: AurumConfig, *,
                        strategies: Sequence[Strategy] | Mapping[str, Strategy] | None = None,
                        only: Sequence[str] | None = None, start: Any = None, end: Any = None,
                        out_dir: str | Path | None = None, write: bool | None = None,
                        include_holdout: bool = False) -> WalkForwardReport:
    """One train/test split: fit on bars before ``start`` (minus purge/embargo), evaluate
    ``[start, end]``. Without ``start`` everything is fitted and evaluated IN-SAMPLE (the
    report says so loudly) — use :func:`run_walk_forward` for honest estimates.

    Bars from ``walkforward.holdout_start`` on are removed unless ``include_holdout`` (the
    holdout must only be looked at once, by the walk-forward's final evaluation).
    """
    t0 = time.perf_counter()
    strats = _resolve_strategies(config, strategies, only)
    start = start if start is not None else config.backtest.start
    end = end if end is not None else config.backtest.end
    bars = md.bars
    notes: list[str] = []
    hs = config.walkforward.holdout_start
    if hs is not None and not include_holdout and start is not None and _utc(start) >= _utc(hs):
        raise ConfigError(f"backtest start {start} is inside the walk-forward holdout (from {hs}); the holdout is "
                          "evaluated once by `aurum walkforward` - pass include_holdout (--include-holdout) "
                          "to spend it anyway")
    if hs is not None and not include_holdout:
        keep = bars.index < _utc(hs)
        if not keep.all():
            md = MarketData(bars=bars.loc[keep], macro=md.macro, events=md.events)
            bars = md.bars
            notes.append(f"bars from walkforward.holdout_start ({hs}) excluded; pass include_holdout "
                         "(--include-holdout) to evaluate them")
    elif hs is not None:
        notes.append(f"WARNING: the evaluation includes the holdout period (from {hs}); it is no longer "
                     "an untouched holdout")
        logger.warning("backtest includes the walk-forward holdout period (from %s)", hs)
    if end is not None:
        e = _utc(end)
        if isinstance(end, str) and len(end.strip()) <= 10:
            e = e + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        md = MarketData(bars=bars.loc[bars.index <= e], macro=md.macro, events=md.events)
        bars = md.bars
    idx = pd.DatetimeIndex(bars.index)
    n = len(idx)
    if n < 2:
        raise ConfigError("backtest window contains fewer than two bars")
    wf = config.walkforward
    bpd = _bars_per_day(idx)
    purge, _, _ = _config_purge(config, strats, bpd)       # same purge rule as the walk-forward
    embargo = _gap_bars(wf.embargo, bpd)
    if start is None:
        plan = FoldPlan("in_sample", 0, n, 0, n, 0, "research")
        notes.append("IN-SAMPLE: strategies and combiner were fitted on the evaluated period; "
                     "use `aurum walkforward` for out-of-sample estimates")
        logger.warning("backtest without --start: IN-SAMPLE evaluation")
    else:
        s0 = int(idx.searchsorted(_utc(start), side="left"))
        tr1 = s0 - purge - embargo
        if s0 >= n - 1:
            raise ConfigError(f"backtest start {start} leaves fewer than two bars")
        if tr1 < wf.min_train_bars:
            raise ConfigError(f"only {max(tr1, 0)} bars before backtest start {start} for fitting "
                              f"(need walkforward.min_train_bars={wf.min_train_bars}); load earlier data")
        plan = FoldPlan("0", 0, tr1, s0, n, 0, "research")  # fit on ALL history before start
    settings = {"n_bars": n, "n_research": n, "bars_per_day": _bars_per_day(idx), "train": plan.n_train,
                "test": plan.n_test, "step": plan.n_test, "purge": purge, "embargo": embargo,
                "anchored": True, "n_folds": 1, "holdout_start": None, "stitched_gaps": False,
                "in_sample": start is None}
    report = _run(md, config, strats, [plan], None, settings, kind="backtest", notes=notes)
    write = config.output.save_results if write is None else write
    if write:
        d = Path(out_dir) if out_dir is not None else _default_out_dir(config, "backtest")
        report.save(d, tearsheet=config.output.tearsheet, dark_charts=config.output.dark_charts)
    report.timing["wall_s"] = round(time.perf_counter() - t0, 3)
    return report


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    """Read ``summary.json`` of a run directory written by :meth:`WalkForwardReport.save`."""
    p = Path(run_dir) / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (is {run_dir} a walkforward/backtest run directory?)")
    return json.loads(p.read_text(encoding="utf-8"))


# ============================================================================================
# a fitted quant book at one point in time (LLM desk run / replay, live warm start)
# ============================================================================================
@dataclass
class QuantBook:
    """Strategies + combiner fitted on ONE training window, applied to the whole history.

    ``signals``/``combined`` cover every bar of the data they were generated on; rows inside
    the training window are in-sample, rows after ``fit_end`` are out-of-sample.
    ``feature_reference`` (when a strategy consumes pipeline features) is the PSI reference
    of the TRAIN-window features after the train-fitted scaling (what the live drift monitor
    compares live features against).
    """

    strategies: dict[str, Strategy]
    combiner: ForecastCombiner
    signals: pd.DataFrame
    combined: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    fit_end: int                      # first bar NOT available to the fit (exclusive)
    pipeline: Any = None              # fitted FeaturePipeline when a strategy consumes features
    combiner_basis: str = ""
    feature_reference: Any = None     # aurum.live.monitor.FeatureReference of the TRAIN features


def fit_quant_book(md: MarketData, cfg: AurumConfig, *,
                   strategies: Sequence[Strategy] | Mapping[str, Strategy] | None = None,
                   only: Sequence[str] | None = None, fit_end: Any = None, gap: bool = True,
                   oos_forecasts: pd.DataFrame | None = None, combiner_fit: str | None = None,
                   feature_reference: bool = True) -> QuantBook:
    """Fit every strategy and the combiner on the training window that ends before ``fit_end``.

    ``fit_end``: bar position or timestamp; bars at/after it are never used for fitting
    (default: all bars). With ``gap=True`` the window also ends ``purge + embargo`` bars
    earlier (use it when the bars after ``fit_end`` will be evaluated, e.g. a desk replay);
    the window length / anchoring follow ``walkforward.train`` / ``anchored``. Strategies
    are then generated on the whole ``md`` (causal) and combined with the fitted weights.
    Trainable strategies are fitted on macro rows published by the end of the window only.

    Combiner: with ``walkforward.combiner_fit="oos"`` (or ``combiner_fit="oos"``) the weights
    come from ``oos_forecasts`` (e.g. a walk-forward run's ``oos_forecasts.parquet``; only rows
    before ``fit_end`` are used) and fall back to EQUAL weights without such history; with
    ``"train"`` they are fitted on the training-window forecasts (in-sample for trainable
    strategies). Either way strategies are scored net of estimated trading costs.
    """
    strats = _resolve_strategies(cfg, strategies, only)
    bars = md.bars
    idx = pd.DatetimeIndex(bars.index)
    n = len(idx)
    if fit_end is None:
        end = n
    elif isinstance(fit_end, (int, np.integer)) and not isinstance(fit_end, bool):
        end = int(fit_end)
    else:
        end = int(idx.searchsorted(_utc(fit_end), side="left"))
    if not 2 <= end <= n:
        raise ConfigError(f"fit_end={fit_end!r} is outside the data (bar {end} of {n})")
    wf = cfg.walkforward
    bpd = _bars_per_day(idx[:end])
    purge, _, _ = _config_purge(cfg, strats, bpd)
    embargo = _gap_bars(wf.embargo, bpd)
    tr1 = end - (purge + embargo if gap else 0)
    tr0 = 0 if wf.anchored else max(0, tr1 - duration_to_bars(wf.train, bpd))
    if tr1 <= 0 or tr1 - tr0 < wf.min_train_bars:
        # same floor as every walk-forward fold: never fit (and trade) a book on a sliver
        raise ConfigError(f"not enough bars to fit the quant book before bar {end}: {max(tr1 - tr0, 0)} "
                          f"training bars < walkforward.min_train_bars={wf.min_train_bars}")
    need = {k: _needs_features(s, cfg.features.enabled) for k, s in strats.items()}
    x_all = None
    pipe = None
    ref = None
    if any(need.values()):
        pipe = cfg.features.build()
        raw = pipe.compute(md)
        pipe.fit(raw.iloc[tr0:tr1])
        x_all = pipe.transform(raw)
        if feature_reference:
            ref = train_feature_reference(x_all.iloc[tr0:tr1])
    fitted: dict[str, Strategy] = {}
    cols: dict[str, np.ndarray] = {}
    md_train = _train_md(md, tr0, tr1)
    for k, proto in strats.items():
        s = proto.clone()
        feats = x_all if need[k] else None
        if s.trainable:
            if feats is None and s.fit_history_bars > 0:
                s.fit(_train_md(md, max(0, tr0 - s.fit_history_bars), tr1), None)
            else:
                s.fit(md_train, None if feats is None else feats.iloc[tr0:tr1])
        cols[k] = _generate(s, md, feats)
        fitted[k] = s
    signals = pd.DataFrame(cols, index=idx)
    keys = list(signals.columns)
    hist = None
    if oos_forecasts is not None:
        missing = [k for k in keys if k not in oos_forecasts.columns]
        if missing:
            raise ConfigError(f"oos_forecasts lack strategies {missing}")
        cut = idx[end] if end < n else idx[-1] + pd.Timedelta(seconds=1)
        hist = oos_forecasts.loc[oos_forecasts.index < cut, keys]
    window = None if wf.anchored else duration_to_bars(wf.train, bpd)
    comb, basis = _fit_fold_combiner(cfg, keys, signals.iloc[tr0:tr1], bars["close"], hist, window,
                                     bars=bars.iloc[:end], combiner_fit=combiner_fit)
    logger.info("quant book combiner: %s", basis)
    combined = comb.combine(signals)
    return QuantBook(strategies=fitted, combiner=comb, signals=signals, combined=combined,
                     train_start=idx[tr0], train_end=idx[tr1 - 1], fit_end=end, pipeline=pipe,
                     combiner_basis=basis, feature_reference=ref)


def train_feature_reference(train_features: pd.DataFrame | None, *, n_bins: int = 10) -> Any:
    """:class:`aurum.live.monitor.FeatureReference` of TRAIN-window features AFTER the
    train-fitted scaling (``pipeline.transform``) - the distribution the live drift monitor
    (PSI) compares live features with. Warm-up rows (NaN) are ignored; None without features."""
    if train_features is None or train_features.shape[1] == 0:
        return None
    from aurum.live.monitor import FeatureReference

    return FeatureReference.from_frame(train_features, n_bins=n_bins)
