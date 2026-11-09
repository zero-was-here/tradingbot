"""Explainability: why the book held what it held, and which strategy earned what.

Everything here reads what a walk-forward run already wrote (``oos_forecasts.parquet``,
``weights.csv``, ``books/<book>/timeseries.parquet``, ``risk_events.csv``, ``config.yaml``)
or a :class:`~aurum.research.walkforward.WalkForwardReport` in memory. Nothing is refitted
and no new evaluation happens, so explaining a run never spends the holdout.

1. Decision explanation at a bar ``t``
--------------------------------------
The combiner's arithmetic (:meth:`aurum.portfolio.combiner.ForecastCombiner.combine`) is
replayed exactly from the saved forecasts and the weights of the fold that contains ``t``::

    c_i(t)      = w_i(fold) * f_i(t) * FDM(fold)          contribution of strategy i
    raw(t)      = sum_i c_i(t)
    combined(t) = clip(raw(t), -1, 1)                       (= the saved 'combined' column)

The waterfall chart stacks the ``c_i`` from 0 to ``raw`` and shows the clip as its own step.
The sizing chain is then replayed with the run's own sizer
(:meth:`aurum.portfolio.sizing.VolTargetSizer.breakdown`) on the run's bars, truncated at
``t`` (point-in-time): volatility forecast ``vol(t)`` (the engine's causal
:func:`~aurum.models.volatility.ewma_volatility`), equity, drawdown from the running peak,
``notional = combined * target_vol / vol * equity`` -> leverage cap -> drawdown de-risking ->
lots -> rounding -> rebalance band, and compared with the saved requested lots. Risk-manager
interventions at ``t`` come from ``risk_events.csv``; the position actually filled at the
next open is ``positions(t+1)``. Without the bars (``load_data=False`` or no data store) the
forecast part is still exact and the sizing block is reduced to the saved values.

2. PnL attribution over a run
-----------------------------
The book's PnL over bar ``t+1`` comes from the position decided at the close of ``t``. It is
allocated to strategies in proportion to their **signed** contributions at that decision::

    share_i(t)      = c_i(t) / raw(t)            (shares sum to 1; 0 when raw(t) == 0)
    pnl_i(t+1)      = share_i(t) * pnl_book(t+1)
    unattributed    = pnl_book(t+1) when raw(t) == 0 (e.g. costs of flattening, a position
                      kept by the rebalance band while the forecast is exactly 0)

so ``sum_i pnl_i + unattributed == total book PnL`` exactly. Clipping rescales all
contributions by the same factor and leaves the shares unchanged. Approximations (by
design, documented): the position is not linear in the forecast (rounding, the rebalance
band can keep an OLDER position, leverage caps, drawdown de-risking, risk-manager vetoes),
yet each bar is attributed by the forecast mix of the latest decision; a strategy that
opposes the net position gets a negative share, i.e. it is credited with the opposite sign
of the book's PnL (it was reducing the position); costs and financing are allocated with
the same shares as price PnL. The per-component split (``price``, ``costs``, ``swap``,
``net``) uses identical shares, so components add up per strategy too.

3. Desk journal summary
-----------------------
:func:`desk_journal_summary` counts what the LLM desk did in a set of journal files (one
JSONL per cycle, see :mod:`aurum.agents`): cycles and their status, Chief actions, specialist
consultations by role, ad-hoc agents created, tool calls, policy constraints and fallbacks,
token cost, and the quant -> final forecast changes.

4. Live / paper decisions
-------------------------
The live runner writes the same arithmetic into every ``type == "decision"`` record of
``<state_dir>/decisions.jsonl`` (field ``attribution``, see :mod:`aurum.live.runner`,
"Decision attribution"). :func:`live_decisions` reads them into a tidy table (one row per
decision, ``c:<strategy>`` contribution columns) and :func:`explain_live_decision` turns one
record into the same :class:`DecisionExplanation` (waterfall with the LLM-desk step, the
runner's own sizing breakdown, the risk verdict and the orders). Reading the log is
read-only and never touches the runner's state files.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DecisionExplanation",
    "DeskSummary",
    "PnLAttribution",
    "RunData",
    "contributions",
    "desk_journal_summary",
    "explain_decision",
    "explain_live_decision",
    "live_decisions",
    "load_run",
    "plot_attribution",
    "plot_waterfall",
    "pnl_attribution",
]

_COMBINED = "combined"
_EPS = 1e-12


# ============================================================================================
# loading
# ============================================================================================
@dataclass
class RunData:
    """The parts of a walk-forward run that explanations need."""

    forecasts: pd.DataFrame          # per-strategy OOS forecasts (bar open time index)
    combined: pd.Series              # saved combined forecast
    folds: pd.Series                 # fold key per bar (str)
    weights: pd.DataFrame            # fold key -> strategy weights
    fdm: pd.Series                   # fold key -> FDM
    book: pd.DataFrame | None        # timeseries of the book (equity, positions, target, pnl_*)
    risk_events: pd.DataFrame | None
    config: dict[str, Any] | None
    phase: str = "research"
    run_dir: Path | None = None
    book_name: str = _COMBINED

    @property
    def strategies(self) -> list[str]:
        return list(self.forecasts.columns)


def _fold_key(v: Any) -> str:
    """Fold keys are strings in ``oos_forecasts.parquet`` ("0", "1", ...) and ints in CSVs."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _read_book(base: Path, book: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    p = base / "books" / book / "timeseries.parquet"
    ts = pd.read_parquet(p) if p.exists() else None
    rp = base / "books" / book / "risk_events.csv"
    ev = None
    if rp.exists():
        try:
            ev = pd.read_csv(rp)
        except pd.errors.EmptyDataError:
            ev = None
    return ts, ev


def load_run(source: Any, *, phase: str = "research", book: str = _COMBINED) -> RunData:
    """Read a run directory (or a ``WalkForwardReport``) for :func:`explain_decision` /
    :func:`pnl_attribution`. ``phase="holdout"`` reads ``holdout/`` (already evaluated: reading
    it is not a new look). Raises ``FileNotFoundError`` / ``ValueError`` naming what is
    missing."""
    if phase not in ("research", "holdout"):
        raise ValueError("phase must be 'research' or 'holdout'")
    if hasattr(source, "oos_forecasts") and hasattr(source, "weights"):
        return _load_report(source, phase=phase, book=book)
    d = Path(source)
    if not d.is_dir():
        raise FileNotFoundError(f"run directory {d} not found")
    cfg = None
    if (d / "config.yaml").exists():
        import yaml

        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    if phase == "holdout":
        base = d / "holdout"
        fp = base / "oos_forecasts.parquet"
        hj = base / "holdout.json"
        if not fp.exists() or not hj.exists():
            raise FileNotFoundError(f"{d} has no holdout evaluation (missing {fp.name} or {hj.name})")
        fc = pd.read_parquet(fp)
        h = json.loads(hj.read_text(encoding="utf-8"))
        w = pd.DataFrame([h.get("weights") or {}], index=["holdout"])
        fdm = pd.Series({"holdout": float(h.get("fdm") or 1.0)})
        folds = pd.Series("holdout", index=fc.index)
    else:
        base = d
        fp = d / "oos_forecasts.parquet"
        wp = d / "weights.csv"
        missing = [p.name for p in (fp, wp) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{d} is missing {missing}: explanations need a walk-forward/backtest run "
                                    "directory written by `aurum walkforward` (not --no-write)")
        fc = pd.read_parquet(fp)
        wdf = pd.read_csv(wp)
        key_col = "fold" if "fold" in wdf.columns else wdf.columns[0]
        wdf.index = [_fold_key(v) for v in wdf[key_col]]
        wdf = wdf.drop(columns=[key_col])
        fdm = wdf.pop("fdm").astype(float) if "fdm" in wdf else pd.Series(1.0, index=wdf.index)
        w = wdf
        if "fold" not in fc:
            raise ValueError(f"{fp} has no 'fold' column")
        folds = fc["fold"].map(_fold_key)
    if _COMBINED not in fc:
        raise ValueError(f"{fp} has no '{_COMBINED}' column")
    strat = [c for c in fc.columns if c not in (_COMBINED, "fold")]
    ts, ev = _read_book(base, book)
    return RunData(forecasts=fc[strat].astype(float), combined=fc[_COMBINED].astype(float), folds=folds,
                   weights=w.reindex(columns=strat).fillna(0.0).astype(float), fdm=fdm.astype(float),
                   book=ts, risk_events=ev, config=cfg, phase=phase, run_dir=d, book_name=book)


def _book_frame(res: Any) -> pd.DataFrame:
    f = pd.DataFrame({"equity": res.equity, "positions": res.positions})
    for k in ("target", "forecast", "position_close"):
        v = getattr(res, k, None)
        if v is not None:
            f[k] = v
    if getattr(res, "pnl", None) is not None:
        f = f.join(res.pnl.add_prefix("pnl_"), how="left")
    return f


def _load_report(rep: Any, *, phase: str, book: str) -> RunData:
    if phase == "holdout":
        h = rep.holdout
        if h is None or h.forecasts is None:
            raise ValueError("the report has no holdout forecasts")
        fc = h.forecasts
        w = pd.DataFrame([h.weights.to_dict()], index=["holdout"])
        fdm = pd.Series({"holdout": float(h.fdm)})
        folds = pd.Series("holdout", index=fc.index)
        res = h.combined if book == _COMBINED else h.strategy_results.get(book)
    else:
        fc = rep.oos_forecasts
        w = rep.weights.copy()
        w.index = [_fold_key(v) for v in w.index]
        fdm = w.pop("fdm") if "fdm" in w else pd.Series(1.0, index=w.index)
        folds = fc["fold"].map(_fold_key)
        res = rep.combined if book == _COMBINED else rep.strategy_results.get(book)
    strat = [c for c in fc.columns if c not in (_COMBINED, "fold")]
    ev = getattr(res, "risk_events", None) if res is not None else None
    return RunData(forecasts=fc[strat].astype(float), combined=fc[_COMBINED].astype(float), folds=folds,
                   weights=w.reindex(columns=strat).fillna(0.0).astype(float), fdm=fdm.astype(float),
                   book=_book_frame(res) if res is not None else None, risk_events=ev, config=rep.config,
                   phase=phase, run_dir=getattr(rep, "out_dir", None), book_name=book)


def contributions(run: RunData) -> pd.DataFrame:
    """Per-bar contributions ``c_i = w_i(fold) * f_i * FDM(fold)`` plus ``raw`` (their sum),
    ``combined`` (``clip(raw, -1, 1)``) and ``saved`` (the run's combined column)."""
    f = run.forecasts.clip(-1.0, 1.0).fillna(0.0)
    unknown = sorted(set(run.folds.unique()) - set(run.weights.index))
    if unknown:
        raise ValueError(f"no combiner weights for fold(s) {unknown} (weights.csv folds: {list(run.weights.index)})")
    w = run.weights.reindex(index=run.folds.to_numpy()).to_numpy(float)
    m = run.fdm.reindex(run.folds.to_numpy()).to_numpy(float)[:, None]
    c = pd.DataFrame(f.to_numpy(float) * w * m, index=f.index, columns=f.columns)
    c["raw"] = c[run.strategies].sum(axis=1)
    c["combined"] = c["raw"].clip(-1.0, 1.0)
    c["saved"] = run.combined
    return c


# ============================================================================================
# 1. decision explanation
# ============================================================================================
@dataclass
class DecisionExplanation:
    """One decision, from strategy forecasts to the filled position (module docstring, 1)."""

    time: pd.Timestamp                  # bar open time of the decision bar
    decision_time: pd.Timestamp | None  # its close (available_at), when the timeframe is known
    fold: str
    phase: str
    table: pd.DataFrame                 # strategy, forecast, weight, fdm, contribution, share
    raw_combined: float
    combined: float
    saved_combined: float
    sizing: dict[str, Any]
    risk: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    #: forecast after the LLM desk (live decisions); ``None`` for research runs (no desk step)
    final_forecast: float | None = None
    #: what the desk did (``action``/``status``), live decisions only
    desk: dict[str, Any] | None = None

    @property
    def clip_adjustment(self) -> float:
        return self.combined - self.raw_combined

    @property
    def desk_delta(self) -> float:
        """``final_forecast - combined`` (0 without a desk step)."""
        return 0.0 if self.final_forecast is None else float(self.final_forecast) - self.combined

    def to_dict(self) -> dict[str, Any]:
        out = {
            "time": self.time, "decision_time": self.decision_time, "fold": self.fold, "phase": self.phase,
            "raw_combined": self.raw_combined, "clip_adjustment": self.clip_adjustment,
            "combined": self.combined, "saved_combined": self.saved_combined,
            "strategies": self.table.to_dict(orient="records"), "sizing": self.sizing, "risk": self.risk,
            "notes": self.notes,
        }
        if self.final_forecast is not None:
            out.update(final_forecast=self.final_forecast, desk_delta=self.desk_delta, desk=self.desk)
        return _jsonable(out)

    def plot(self) -> Any:
        return plot_waterfall(self)


def _resolve_at(index: pd.DatetimeIndex, at: Any) -> int:
    if at is None:
        return len(index) - 1
    t = pd.Timestamp(at)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    i = int(index.searchsorted(t, side="right")) - 1
    if i < 0:
        raise ValueError(f"--at {t} is before the first OOS bar ({index[0]})")
    if t > index[-1] + pd.Timedelta(days=7):
        raise ValueError(f"--at {t} is after the last OOS bar ({index[-1]}) of this run/phase")
    return i


def _run_cfg(run: RunData) -> Any:
    from aurum.core.config import config_from_dict

    d = dict(run.config or {})
    d.pop("source", None)
    return config_from_dict(d, env={}, validate=False)


def explain_decision(source: Any, at: Any = None, *, phase: str = "research", book: str = _COMBINED,
                     md: Any = None, load_data: bool = True) -> DecisionExplanation:
    """Explain the decision at the close of the last OOS bar at or before ``at`` (default: the
    last bar). ``source``: run directory or ``WalkForwardReport``; ``md``: market data to
    replay the sizing with (default: the run's own data from its ``config.yaml`` when
    ``load_data``; bars after ``t`` are never used)."""
    run = source if isinstance(source, RunData) else load_run(source, phase=phase, book=book)
    c = contributions(run)
    i = _resolve_at(pd.DatetimeIndex(c.index), at)
    t = c.index[i]
    fold = str(run.folds.iloc[i])
    notes: list[str] = []
    if at is not None:
        t_at = pd.Timestamp(at)
        t_at = t_at.tz_localize("UTC") if t_at.tz is None else t_at.tz_convert("UTC")
        if t != t_at:
            notes.append(f"no OOS bar opens at {t_at}; explaining the last bar at or before it ({t})")
    fdm = float(run.fdm.get(fold, 1.0))
    raw = float(c["raw"].iloc[i])
    rows = []
    for s in run.strategies:
        f = float(np.clip(np.nan_to_num(run.forecasts[s].iloc[i]), -1.0, 1.0))
        w = float(run.weights.loc[fold, s])
        ci = float(c[s].iloc[i]) + 0.0
        rows.append({"strategy": s, "forecast": f, "weight": w, "fdm": fdm, "contribution": ci,
                     "share": (ci / raw + 0.0) if abs(raw) > _EPS else math.nan})   # + 0.0: no "-0.0"
    table = pd.DataFrame(rows)
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    combined = float(c["combined"].iloc[i])
    saved = float(c["saved"].iloc[i])
    if abs(combined - saved) > 1e-9:
        notes.append(f"replayed combined forecast {combined:.6f} differs from the saved {saved:.6f}")
    sizing, risk, dtime = _sizing_block(run, i, t, saved, md=md, load_data=load_data, notes=notes)
    return DecisionExplanation(time=t, decision_time=dtime, fold=fold, phase=run.phase, table=table,
                               raw_combined=raw, combined=combined, saved_combined=saved, sizing=sizing,
                               risk=risk, notes=notes)


def _sizing_block(run: RunData, i: int, t: pd.Timestamp, forecast: float, *, md: Any, load_data: bool,
                  notes: list[str]) -> tuple[dict[str, Any], dict[str, Any], pd.Timestamp | None]:
    sizing: dict[str, Any] = {"forecast": forecast}
    risk: dict[str, Any] = {"interventions": []}
    dtime = None
    b = run.book
    if b is None or t not in b.index:
        notes.append(f"book '{run.book_name}' timeseries not available at {t}: sizing not explained")
        return sizing, risk, dtime
    j = int(b.index.get_loc(t))
    eq = b["equity"].astype(float)
    cur_col = "position_close" if "position_close" in b else "positions"
    equity = float(eq.iloc[j])
    peak = float(eq.iloc[:j + 1].max())
    dd = max(0.0, 1.0 - equity / peak) if peak > 0 else 0.0
    current = float(b[cur_col].iloc[j])
    sizing.update({"equity": equity, "peak_equity": peak, "drawdown": dd, "current_lots": current,
                   "saved_requested_lots": _f(b["target"].iloc[j]) if "target" in b else math.nan,
                   "filled_lots_next_bar": float(b["positions"].iloc[j + 1]) if j + 1 < len(b) else math.nan})
    if "forecast" in b and math.isfinite(_f(b["forecast"].iloc[j])):
        sizing["book_forecast"] = _f(b["forecast"].iloc[j])
    if j == len(b) - 1:
        notes.append("last bar of the book: its decision was never executed (the backtest ends at this close), so "
                     "there is no saved request or fill to compare with")
    ev = run.risk_events
    if ev is not None and len(ev) and "bar_time" in ev:
        bt = pd.to_datetime(ev["bar_time"], utc=True)
        hit = ev.loc[bt == t]
        for _, r in hit.iterrows():
            risk["interventions"].append({k: _py(r[k]) for k in ("requested", "approved", "halted", "reasons")
                                          if k in r})
    cfg = None
    try:
        cfg = _run_cfg(run) if run.config else None
    except Exception as exc:  # noqa: BLE001
        notes.append(f"run config unreadable ({exc}): sizing replay skipped")
    if cfg is not None:
        from aurum.core.timeframes import get_timeframe

        try:
            dtime = t + get_timeframe(cfg.data.timeframe).delta
        except ValueError:
            dtime = None
    bars = None
    if md is not None:
        bars = md.bars if hasattr(md, "bars") else md
    elif load_data and cfg is not None:
        try:
            if cfg.data.synthetic is not None or cfg.data.resolved_bars_path().exists():
                bars = cfg.data.load().bars
            else:
                notes.append(f"bars file {cfg.data.resolved_bars_path()} not found: sizing replay skipped")
        except Exception as exc:  # noqa: BLE001 - optional enrichment
            notes.append(f"could not load the run's data ({exc}): sizing replay skipped")
    if bars is None or cfg is None or t not in bars.index:
        if bars is not None and t not in bars.index:
            notes.append(f"{t} is not in the loaded bars: sizing replay skipped")
        return sizing, risk, dtime
    from aurum.models.volatility import ewma_volatility

    hist = bars.loc[bars.index <= t]                      # point-in-time: nothing after t
    vol = float(ewma_volatility(hist["close"].astype(float)).iloc[-1])
    price = float(hist["close"].iloc[-1])
    if "available_at" in hist:
        dtime = pd.Timestamp(hist["available_at"].iloc[-1])
    if cfg.sizing.method != "vol_target":
        notes.append(f"sizing.method={cfg.sizing.method!r}: only vol_target sizing is replayed")
        sizing.update({"vol_ann": vol, "price": price})
        return sizing, risk, dtime
    sizer = cfg.sizing.build()
    inst = cfg.instrument.build()
    bd = sizer.breakdown(forecast, vol, equity, price, inst, current_lots=current, drawdown=dd)
    sizing.update({"vol_ann": vol, "price": price, "spread": _f(hist["spread"].iloc[-1]) if "spread" in hist
                   else math.nan, "target_vol": float(sizer.target_vol), "max_leverage": float(sizer.max_leverage),
                   "raw_notional": bd.raw_notional, "capped_notional": bd.capped_notional,
                   "drawdown_mult": bd.drawdown_mult, "raw_lots": bd.raw_lots, "rounded_lots": bd.rounded_lots,
                   "kept_current_band": bd.kept_current, "caps_hit": list(bd.caps_hit), "replayed_lots": bd.final_lots,
                   "note": bd.note})
    saved_req = sizing.get("saved_requested_lots", math.nan)
    if math.isfinite(saved_req):
        sizing["replay_matches_saved"] = bool(abs(bd.final_lots - saved_req) < 1e-9)
        if not sizing["replay_matches_saved"]:
            notes.append(f"replayed requested lots {bd.final_lots} != saved {saved_req} (data or config differ "
                         "from the run's)")
    return sizing, risk, dtime


# ============================================================================================
# 2. PnL attribution
# ============================================================================================
@dataclass
class PnLAttribution:
    """Book PnL allocated to strategies (module docstring, 2)."""

    per_bar: pd.DataFrame          # USD per bar: one column per strategy + "unattributed"
    totals: pd.Series              # USD per strategy (+ unattributed); sums to total_pnl
    by_component: pd.DataFrame     # strategy x [price, costs, swap, net]
    by_year: pd.DataFrame          # calendar year x strategy (net)
    total_pnl: float
    component: str
    phase: str
    method: str = ("pnl(t+1) allocated by signed combiner contributions at the close of t: "
                   "share_i = w_i * f_i * FDM / sum_j w_j * f_j * FDM")

    def cumulative(self) -> pd.DataFrame:
        return self.per_bar.cumsum()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"total_pnl": self.total_pnl, "component": self.component, "phase": self.phase,
                          "method": self.method, "totals": self.totals.to_dict(),
                          "by_component": self.by_component.to_dict(orient="index"),
                          "by_year": {str(k): v for k, v in self.by_year.to_dict(orient="index").items()}})

    def plot(self) -> Any:
        return plot_attribution(self)


def pnl_attribution(source: Any, *, phase: str = "research", component: str = "net") -> PnLAttribution:
    """Allocate the combined book's per-bar PnL to strategies (module docstring, 2).

    ``component``: ``net`` (default), ``price``, ``costs`` or ``swap`` for ``per_bar`` and
    ``totals``; ``by_component`` always has all four."""
    if component not in ("net", "price", "costs", "swap"):
        raise ValueError("component must be net|price|costs|swap")
    run = source if isinstance(source, RunData) else load_run(source, phase=phase, book=_COMBINED)
    if run.book is None:
        raise FileNotFoundError("the combined book's timeseries is missing (books/combined/timeseries.parquet): "
                                "was the run written with save_books?")
    b = run.book
    c = contributions(run)
    strat = run.strategies
    raw = c["raw"].to_numpy(float)
    share = np.where(np.abs(raw)[:, None] > _EPS, c[strat].to_numpy(float) / np.where(np.abs(raw) > _EPS, raw, 1.0)
                     [:, None], 0.0)
    share_df = pd.DataFrame(share, index=c.index, columns=strat)
    # PnL at row t (over bar t) comes from the decision at the close of t-1
    dec = share_df.reindex(b.index).shift(1).fillna(0.0)
    has = dec.abs().sum(axis=1) > _EPS
    pnl = _pnl_components(b)
    comp_tables = {}
    for k, s in pnl.items():
        alloc = dec.mul(s, axis=0)
        alloc["unattributed"] = s.where(~has, 0.0)
        comp_tables[k] = alloc
    per_bar = comp_tables[component]
    totals = per_bar.sum()
    by_comp = pd.DataFrame({k: v.sum() for k, v in comp_tables.items()})
    by_year = per_bar.groupby(per_bar.index.year).sum()
    by_year.index.name = "year"
    total = float(pnl[component].sum())
    resid = float(totals.sum() - total)
    if abs(resid) > 1e-6 * max(1.0, abs(total)):  # pragma: no cover - identity by construction
        logger.warning("PnL attribution residual %.6f", resid)
    return PnLAttribution(per_bar=per_bar, totals=totals, by_component=by_comp, by_year=by_year,
                          total_pnl=total, component=component, phase=run.phase)


def _pnl_components(b: pd.DataFrame) -> dict[str, pd.Series]:
    if "pnl_net" in b:
        net = b["pnl_net"].astype(float).fillna(0.0)
    else:
        net = b["equity"].astype(float).diff().fillna(0.0)
    price = b["pnl_price"].astype(float).fillna(0.0) if "pnl_price" in b else net.copy()
    costs = -b["pnl_costs"].astype(float).fillna(0.0) if "pnl_costs" in b else net * 0.0
    swap = b["pnl_swap"].astype(float).fillna(0.0) if "pnl_swap" in b else net * 0.0
    return {"net": net, "price": price, "costs": costs, "swap": swap}


# ============================================================================================
# 3. desk journal summary
# ============================================================================================
@dataclass
class DeskSummary:
    """Counts of what the LLM desk did (module docstring, 3)."""

    cycles: pd.DataFrame
    totals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"totals": self.totals, "cycles": self.cycles.to_dict(orient="records")})


def _journal_files(path: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            return sorted(p.glob("*.jsonl"))
        if p.exists():
            return [p]
        raise FileNotFoundError(f"desk journal {p} not found")
    return [Path(x) for x in path]


def desk_journal_summary(path: str | Path | Iterable[str | Path]) -> DeskSummary:
    """Summarise desk journals: a directory of ``*.jsonl`` cycle files, one file, or a list."""
    files = _journal_files(path)
    if not files:
        raise FileNotFoundError(f"no *.jsonl journal files in {path}")
    cycles: dict[str, dict[str, Any]] = {}
    consults: Counter[str] = Counter()
    created: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    agent_status: Counter[str] = Counter()
    bad = 0
    for fp in files:
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            cid = str(e.get("cycle_id", fp.stem))
            cyc = cycles.setdefault(cid, {"cycle_id": cid, "file": fp.name, "now": None, "quant_forecast": math.nan,
                                          "final_forecast": math.nan, "status": None, "action": None,
                                          "constrained": False, "used_fallback": False, "n_consults": 0,
                                          "n_created": 0, "n_tool_calls": 0, "cost_usd": math.nan,
                                          "duration_s": math.nan, "failure_reason": None})
            ev = e.get("event")
            if ev == "cycle_start":
                cyc["now"] = e.get("now")
                cyc["quant_forecast"] = _f(e.get("quant_forecast"))
            elif ev == "tool_call":
                tool = str(e.get("tool"))
                tools[tool] += 1
                cyc["n_tool_calls"] += 1
                if tool == "consult_specialist":
                    cyc["n_consults"] += 1
                    consults[str((e.get("input") or {}).get("role", "?"))] += 1
            elif ev == "agent_created":
                cyc["n_created"] += 1
                created[str(e.get("name", e.get("agent", "?")))] += 1
            elif ev == "agent_end":
                agent_status[str(e.get("status"))] += 1
            elif ev == "decision":
                d = e.get("decision") or {}
                cyc["action"] = d.get("action")
                cyc["failure_reason"] = e.get("failure_reason")
            elif ev == "policy":
                o = e.get("outcome") or {}
                cyc["constrained"] = bool(o.get("constrained"))
                cyc["used_fallback"] = bool(o.get("used_fallback"))
                cyc["action"] = cyc["action"] or o.get("action")
            elif ev == "cycle_end":
                cyc["status"] = e.get("status")
                cyc["final_forecast"] = _f(e.get("final_forecast"))
                cyc["quant_forecast"] = _f(e.get("quant_forecast", cyc["quant_forecast"]))
                cyc["duration_s"] = _f(e.get("duration_s"))
                cyc["cost_usd"] = _f(((e.get("usage") or {}).get("total") or {}).get("cost_usd"))
    df = pd.DataFrame(list(cycles.values()))
    q, f = df["quant_forecast"].astype(float), df["final_forecast"].astype(float)
    both = q.notna() & f.notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (f[both] / q[both]).replace([np.inf, -np.inf], np.nan)
    totals = {
        "n_files": len(files), "n_cycles": len(df), "unreadable_lines": bad,
        "status": dict(Counter(df["status"].fillna("unfinished"))),
        "actions": dict(Counter(df["action"].fillna("none"))),
        "n_consults": int(df["n_consults"].sum()), "consults_by_role": dict(consults),
        "n_created_agents": int(df["n_created"].sum()), "created_agents": dict(created),
        "n_tool_calls": int(df["n_tool_calls"].sum()), "tool_calls": dict(tools),
        "agent_status": dict(agent_status),
        "n_constrained": int(df["constrained"].sum()), "n_fallback": int(df["used_fallback"].sum()),
        "total_cost_usd": float(df["cost_usd"].sum(min_count=1)) if df["cost_usd"].notna().any() else math.nan,
        "mean_abs_quant": float(q.abs().mean()) if q.notna().any() else math.nan,
        "mean_abs_final": float(f.abs().mean()) if f.notna().any() else math.nan,
        "median_final_over_quant": float(scale.median()) if scale.notna().any() else math.nan,
        "n_direction_flips": int(((np.sign(q) * np.sign(f)) < 0)[both].sum()),
    }
    return DeskSummary(cycles=df, totals=totals)


# ============================================================================================
# 4. live / paper decisions (the runner's decisions.jsonl)
# ============================================================================================
_LIVE_LOG = "decisions.jsonl"


def _live_log_path(source: str | Path) -> Path:
    p = Path(source)
    if p.is_dir():
        p = p / _LIVE_LOG
    if not p.exists():
        raise FileNotFoundError(f"live decision log {p} not found (pass live.state_dir or its {_LIVE_LOG})")
    return p


def _live_records(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``type == "decision"`` records of a runner log (a state dir, a JSONL file, or records)."""
    if isinstance(source, (str, Path)):
        from aurum.live.state import read_jsonl  # torn final line (crash) skipped with a warning

        recs = read_jsonl(_live_log_path(source))
    else:
        recs = [dict(r) for r in source]
    return [r for r in recs if isinstance(r, Mapping) and r.get("type") == "decision"]


def _ts(v: Any) -> pd.Timestamp | None:
    if v is None:
        return None
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if t is pd.NaT:
        return None
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def live_decisions(source: str | Path | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per live/paper decision with its attribution (module docstring, 4).

    ``source``: ``live.state_dir``, its ``decisions.jsonl`` or the parsed records. Columns:
    ``time`` (decision time), ``bar_time``, ``mode``, ``status``, ``combined``,
    ``raw_combined``, ``final_forecast``, ``desk_delta``, ``desk_action``, ``method``,
    ``fdm``, ``unallocated``, ``clipped``, ``exact``, ``requested_lots``,
    ``approved_lots``, ``current_lots``, ``equity`` and one ``c:<strategy>`` column per
    strategy (``w_i * f_i * FDM``; NaN when the record has no attribution, e.g. a skipped
    bar or a combiner without weights). Indexed by ``bar_time``."""
    rows: list[dict[str, Any]] = []
    for r in _live_records(source):
        att = r.get("attribution") or {}
        desk = r.get("desk") or {}
        row: dict[str, Any] = {
            "time": _ts(r.get("time")), "bar_time": _ts(r.get("bar_time")), "mode": r.get("mode"),
            "status": r.get("status"), "combined": _f(r.get("combined")),
            "raw_combined": _f(att.get("raw_combined")),
            "final_forecast": _f(att.get("final_forecast", r.get("final_forecast"))),
            "desk_delta": _f(att.get("desk_delta")),
            "desk_action": (desk.get("action") or desk.get("status")) if desk else None,
            "method": att.get("method"), "fdm": _f(att.get("fdm")), "unallocated": _f(att.get("unallocated")),
            "clipped": att.get("clipped"), "exact": att.get("exact"),
            "requested_lots": _f(r.get("requested_lots")), "approved_lots": _f(r.get("approved_lots")),
            "current_lots": _f(r.get("current_lots")), "equity": _f(r.get("equity")),
        }
        for name, s in (att.get("strategies") or {}).items():
            row[f"c:{name}"] = _f((s or {}).get("contribution"))
        rows.append(row)
    if not rows:
        raise ValueError("the live decision log has no decision records yet")
    df = pd.DataFrame(rows)
    df = df.set_index("bar_time", drop=False)
    df.index.name = None
    return df


def _live_sizing(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The runner's sizing breakdown (``rec["sizing"]``) in the keys :func:`plot_waterfall`
    reads, plus the risk-approved lots and the orders."""
    s: dict[str, Any] = {"forecast": _f(rec.get("final_forecast"))}
    bd = rec.get("sizing") or {}
    if bd:
        vol, fc, eq = _f(bd.get("vol_ann")), _f(bd.get("forecast")), _f(bd.get("equity"))
        raw = _f(bd.get("raw_notional"))
        s.update({"vol_ann": vol, "price": _f(bd.get("price")), "equity": eq, "drawdown": _f(bd.get("drawdown")),
                  "raw_notional": raw, "capped_notional": _f(bd.get("capped_notional")),
                  "drawdown_mult": _f(bd.get("drawdown_mult")), "raw_lots": _f(bd.get("raw_lots")),
                  "rounded_lots": _f(bd.get("rounded_lots")), "replayed_lots": _f(bd.get("final_lots")),
                  "kept_current_band": bool(bd.get("kept_current", False)),
                  "caps_hit": list(bd.get("caps_hit") or []), "note": bd.get("note", "")})
        # target_vol is not logged; it is implied by notional = forecast * target_vol / vol * equity
        if all(math.isfinite(x) for x in (vol, fc, eq, raw)) and abs(fc) > _EPS and eq > 0:
            s["target_vol"] = raw * vol / (fc * eq)
    elif math.isfinite(_f(rec.get("equity"))):
        s.update({"equity": _f(rec.get("equity")), "drawdown": _f(rec.get("drawdown")),
                  "vol_ann": math.nan})
    if "current_lots" in rec:
        s["current_lots"] = _f(rec.get("current_lots"))
        s["saved_requested_lots"] = _f(rec.get("requested_lots"))
        s["approved_lots"] = _f(rec.get("approved_lots"))
    ex = rec.get("execution") or {}
    if isinstance(ex, Mapping) and ex:
        s["orders_status"] = ex.get("status")
    return s


def explain_live_decision(source: str | Path | Iterable[Mapping[str, Any]], at: Any = None) -> DecisionExplanation:
    """Explain one live/paper decision from the runner's log (module docstring, 4): the
    decision on the last bar opening at or before ``at`` (default: the latest decision that
    carries an attribution).

    The forecast part is the runner's own arithmetic (``attribution``); the sizing part is
    its logged :class:`~aurum.portfolio.sizing.SizingBreakdown`; the risk part its verdict
    (approved lots and reasons). ``final_forecast`` / ``desk`` add the LLM-desk step."""
    recs = _live_records(source)
    usable = [r for r in recs if r.get("attribution") and _ts(r.get("bar_time")) is not None]
    if not usable:
        raise ValueError("no live decision with an attribution yet (skipped bars and failed signals have none; "
                         "logs written before v2.1 have no attribution field)")
    notes: list[str] = []
    if at is None:
        rec = usable[-1]
    else:
        t_at = _ts(at)
        if t_at is None:
            raise ValueError(f"--at {at!r} is not a timestamp")
        before = [r for r in usable if _ts(r["bar_time"]) <= t_at]
        if not before:
            raise ValueError(f"--at {t_at} is before the first attributed live decision ({usable[0]['bar_time']})")
        rec = before[-1]
        if _ts(rec["bar_time"]) != t_at:
            notes.append(f"no attributed decision on the bar opening at {t_at}; explaining the last one before it "
                         f"({_ts(rec['bar_time'])})")
    att = rec["attribution"]
    fdm = _f(att.get("fdm"))
    raw = _f(att.get("raw_combined"))
    combined = _f(att.get("combined", rec.get("combined")))
    rows = []
    for name, s in (att.get("strategies") or {}).items():
        c = _f((s or {}).get("contribution"))
        rows.append({"strategy": str(name), "forecast": _f((s or {}).get("forecast")),
                     "weight": _f((s or {}).get("weight")), "fdm": fdm, "contribution": c + 0.0,
                     "share": (c / raw + 0.0) if math.isfinite(raw) and abs(raw) > _EPS else math.nan})
    table = pd.DataFrame(rows, columns=["strategy", "forecast", "weight", "fdm", "contribution", "share"])
    if not att.get("exact", False):
        notes.append("attribution is not exact: the combiner exposed no per-strategy weights (contributions "
                     "unavailable)" if not math.isfinite(raw) else
                     "attribution is not exact: the contributions do not reproduce the combined forecast")
        if not math.isfinite(raw):
            raw = combined
            table["contribution"] = table["contribution"].fillna(0.0)
    if len(table):
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    final = _f(att.get("final_forecast", rec.get("final_forecast")))
    desk = rec.get("desk") or None
    if isinstance(desk, Mapping):
        desk = {k: desk.get(k) for k in ("status", "action", "quant_forecast", "final_forecast", "cost_usd",
                                         "rationale") if k in desk}
    risk = rec.get("risk") or {}
    interventions = []
    if risk and (risk.get("reasons") or risk.get("halted")):
        interventions.append({"requested": rec.get("requested_lots"), "approved": rec.get("approved_lots"),
                              "halted": risk.get("halted"), "reasons": risk.get("reasons")})
    if rec.get("venue_tz_confirmed") is False:
        notes.append("venue time zone UNCONFIRMED at this decision (MT5 server_tz auto on a persisted/fallback "
                     "offset): orders were deferred until a fresh tick confirmed it")
    if rec.get("status") and rec["status"] not in ("ok", "noop", "filled", "planned"):
        notes.append(f"decision status: {rec['status']}")
    return DecisionExplanation(time=_ts(rec["bar_time"]), decision_time=_ts(rec.get("time")),
                               fold=str(rec.get("mode") or "live"), phase="live", table=table,
                               raw_combined=float(raw), combined=float(combined),
                               saved_combined=_f(rec.get("combined")), sizing=_live_sizing(rec),
                               risk={"interventions": interventions, **({"verdict": dict(risk)} if risk else {})},
                               notes=notes, final_forecast=final if math.isfinite(final) else None,
                               desk=desk if isinstance(desk, Mapping) else None)


# ============================================================================================
# helpers + plots
# ============================================================================================
def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


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


_INK, _INK2, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_BLUE, _RED, _NEUTRAL = "#2a78d6", "#e34948", "#898781"
_SERIES = ("#2a78d6", "#e8a33d", "#1f9e89", "#9467bd", "#e34948", "#8c564b", "#bcbd22", "#17becf", "#e377c2",
           "#4c4c9d")


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


def plot_waterfall(ex: DecisionExplanation) -> Any:
    """Waterfall from 0 through each strategy's contribution to the raw sum, the clip step and
    the final combined forecast; the sizing chain is printed underneath. Strategies with a zero
    contribution (zero weight or a flat forecast) are listed in the caption instead of drawn."""
    import textwrap

    fig = _figure((10.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.3])
    ax = fig.add_subplot(gs[0])
    _style(ax)
    t = ex.table
    live = t.loc[t["contribution"].abs() > 1e-12]
    idle = t.loc[t["contribution"].abs() <= 1e-12]
    labels, bottoms, heights, colors = [], [], [], []
    level = 0.0
    for _, r in live.iterrows():
        labels.append(f"{r['strategy']}\nf {r['forecast']:+.2f}\nw {r['weight']:.2f}")
        bottoms.append(level)
        heights.append(r["contribution"])
        colors.append(_BLUE if r["contribution"] >= 0 else _RED)
        level += r["contribution"]
    fdm = float(t["fdm"].iloc[0]) if len(t) else 1.0
    labels.append(f"raw sum\n(FDM {fdm:.2f})")
    bottoms.append(0.0)
    heights.append(ex.raw_combined)
    colors.append(_NEUTRAL)
    clip = ex.clip_adjustment
    if abs(clip) > 1e-12:
        labels.append("clip to\n[-1, 1]")
        bottoms.append(ex.raw_combined)
        heights.append(clip)
        colors.append("#c3c2b7")
    labels.append("combined\nforecast")
    bottoms.append(0.0)
    heights.append(ex.combined)
    colors.append(_INK)
    if ex.final_forecast is not None and abs(ex.desk_delta) > 1e-12:   # live: the LLM desk's step
        action = (ex.desk or {}).get("action") or (ex.desk or {}).get("status") or "desk"
        labels.append(f"LLM desk\n({action})")
        bottoms.append(ex.combined)
        heights.append(ex.desk_delta)
        colors.append("#9467bd")
        labels.append("final\nforecast")
        bottoms.append(0.0)
        heights.append(float(ex.final_forecast))
        colors.append(_INK)
    xs = np.arange(len(labels))
    ax.bar(xs, heights, bottom=bottoms, color=colors, width=0.6, edgecolor="none")
    for k in range(len(xs) - 1):
        y = bottoms[k] + heights[k]
        ax.plot([xs[k] + 0.3, xs[k + 1] - 0.3], [y, y], color=_MUTED, linewidth=0.7, linestyle=":")
    tops = [b0 + h for b0, h in zip(bottoms, heights, strict=True)]
    lo = min(0.0, *bottoms, *tops)
    hi = max(0.0, *bottoms, *tops)
    pad = 0.12 * max(hi - lo, 0.05)
    ax.set_ylim(lo - pad, hi + pad)
    for x, b0, h in zip(xs, bottoms, heights, strict=True):
        end = b0 + h
        ax.text(x, end + (pad * 0.15 if h >= 0 else -pad * 0.15), f"{h:+.3f}", ha="center",
                va="bottom" if h >= 0 else "top", fontsize=8, color=_INK2)
    ax.axhline(0.0, color=_INK2, linewidth=0.8)
    ax.set_xticks(xs, labels, fontsize=7.5)
    ax.set_ylabel("forecast units", color=_INK2)
    where = f"{ex.fold}, {ex.phase}" if ex.phase == "live" else f"fold {ex.fold}, {ex.phase}"
    head = f"combined forecast {ex.combined:+.3f}"
    if ex.final_forecast is not None and abs(ex.desk_delta) > 1e-12:
        head += f", final {ex.final_forecast:+.3f} after the desk"
    fig.suptitle(f"Decision at the close of the {ex.time:%Y-%m-%d %H:%M} UTC bar ({where}): {head}", x=0.01,
                 ha="left", fontsize=10.5, fontweight="bold", color=_INK)
    cap = "contribution = weight x forecast x FDM; bars stack from 0 to the raw sum"
    if len(idle):
        cap += "; zero contribution: " + ", ".join(idle["strategy"])
    ax.set_title(textwrap.fill(cap, 150), loc="left", fontsize=8.5, color=_INK2)
    tx = fig.add_subplot(gs[1])
    tx.axis("off")
    s = ex.sizing
    parts = []
    if math.isfinite(_f(s.get("vol_ann"))):
        parts.append(f"inputs: vol forecast {s['vol_ann']:.1%}, price {s.get('price', math.nan):,.2f}, equity "
                     f"{s.get('equity', math.nan):,.0f}, drawdown {s.get('drawdown', 0):.1%}")
        parts.append(f"notional = {ex.combined:+.3f} x {s.get('target_vol', math.nan):.0%} / {s['vol_ann']:.1%} x "
                     f"equity = {s.get('raw_notional', math.nan):,.0f} USD; caps {s.get('caps_hit') or 'none'}; "
                     f"drawdown multiplier {s.get('drawdown_mult', 1.0):.2f}")
        parts.append(f"lots {s.get('raw_lots', math.nan):+.3f} -> rounded {s.get('rounded_lots', math.nan):+.2f}"
                     + (" -> rebalance band kept the current position" if s.get("kept_current_band") else "")
                     + f" -> requested {s.get('replayed_lots', math.nan):+.2f}")
    elif "equity" in s:
        parts.append(f"equity {s['equity']:,.0f}, drawdown {s.get('drawdown', 0):.1%} (bars unavailable: sizing "
                     "not replayed)")
    if "current_lots" in s:
        line = f"position {s['current_lots']:+.2f} lots -> requested {s.get('saved_requested_lots', math.nan):+.2f}"
        if "approved_lots" in s:
            line += f" -> risk-approved {_f(s.get('approved_lots')):+.2f}"
        if "filled_lots_next_bar" in s:
            line += f" -> filled at the next open {s.get('filled_lots_next_bar', math.nan):+.2f}"
        if s.get("orders_status"):
            line += f" (orders: {s['orders_status']})"
        parts.append(line)
    for iv in ex.risk.get("interventions", []):
        parts.append(f"risk manager: {iv.get('reasons', '')}")
    text = "\n".join(textwrap.fill(x, 128, subsequent_indent="    ") for x in parts) or "(no sizing information)"
    tx.text(0.0, 1.0, text, va="top", ha="left", fontsize=8, color=_INK2, family="monospace",
            transform=tx.transAxes)
    return fig


def plot_attribution(att: PnLAttribution, *, top: int = 10) -> Any:
    """Cumulative attributed PnL per strategy (lines) and the book total; per-strategy totals
    as horizontal bars."""
    fig = _figure((11.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0])
    ax = fig.add_subplot(gs[0])
    _style(ax)
    cum = att.per_bar.cumsum()
    order = att.totals.abs().sort_values(ascending=False).index[:top]
    k = 0
    for col in order:
        if col == "unattributed":
            ax.plot(cum.index, cum[col], linewidth=0.9, color=_MUTED, linestyle="--", label=col)
            continue
        ax.plot(cum.index, cum[col], linewidth=1.3, color=_SERIES[k % len(_SERIES)], label=col)
        k += 1
    ax.plot(cum.index, cum.sum(axis=1), color=_INK, linewidth=2.0, label="book total")
    ax.axhline(0.0, color=_INK2, linewidth=0.7)
    ax.set_ylabel(f"cumulative {att.component} PnL (USD)", color=_INK2)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(_INK2)
    ax.set_title(f"PnL attribution by strategy ({att.phase}): total {att.total_pnl:,.0f} USD", loc="left",
                 fontsize=10, fontweight="bold", color=_INK)
    bx = fig.add_subplot(gs[1])
    _style(bx)
    tot = att.totals.sort_values()
    bx.barh(range(len(tot)), tot.to_numpy(), color=[_BLUE if v >= 0 else _RED for v in tot], height=0.6)
    bx.set_yticks(range(len(tot)), list(tot.index), fontsize=8)
    bx.axvline(0.0, color=_INK2, linewidth=0.7)
    bx.grid(True, axis="x", color=_GRID, linewidth=0.6)
    bx.set_title("total by strategy (USD)", loc="left", fontsize=10, fontweight="bold", color=_INK)
    return fig
