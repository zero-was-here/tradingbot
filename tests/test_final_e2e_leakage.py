"""End-to-end leakage audit of the walk-forward with the FULL default configuration.

``configs/default.yaml`` as shipped (all 14 strategies, including the trainable
``intraday_seasonality``, ``ml_gbm`` and ``meta_label``; net-of-cost ``sharpe_shrink``
combiner; vol-target sizer; research risk limits; rate-based financing) is run end to end
through :func:`aurum.research.walkforward.run_walk_forward`. Only the walk-forward window
lengths are shortened so that a synthetic sample fits several folds and a holdout.

Future-perturbation test (the core check)
    Report A is run on bars + macro + events as they are. Report B is run on a copy in which
    everything strictly after a cutoff ``X`` (in the middle of an OOS test block) is replaced:

    * bars with ``available_at > X`` follow a different, still valid path (block-shuffled,
      sign-reversed and re-scaled returns with the bar anatomy of the shuffled source bars,
      shuffled and re-scaled spreads, shuffled volume) - timestamps are kept, so the fold
      geometry is identical;
    * macro rows with ``available_at > X`` get new values (a multiplicative random walk plus
      noise) and every 7th of them is deleted - including rows whose OBSERVATION date is
      before ``X`` but whose publication is after it (FRED's lag);
    * calendar events scheduled more than ``EVENT_MARGIN`` after ``X`` are moved, some are
      deleted and a fake one is added. Scheduled release times are public in advance
      (SPEC §3.5), so events shortly after ``X`` legitimately shape pre-``X`` decisions
      (hours-to-next-event is capped at 72 h, risk blackouts look 24 h ahead); the margin
      keeps the test about leakage, not about the calendar's legitimate look-ahead.

    Every per-strategy and combined OOS forecast, every combiner weight/FDM, every book's
    target lots, position and equity decided or marked at or before ``X`` must then be
    BIT-IDENTICAL between A and B, while the outputs after ``X`` must differ (the test has
    teeth). The ML skill gate is switched off here: on a gated fold the ML forecasts are
    identically zero and could not reveal a leak.

Also checked on the same runs: (a) the stitched OOS contains no in-sample row - fold
geometry, and every ``Strategy.fit`` slice (with the macro rows handed to it) and every
combiner fit, recorded by spies that also run inside the process-pool workers; (b)
perturbing ONLY the holdout leaves every research output bit-identical.

Slow variants (``-m slow`` or ``AURUM_SLOW_TESTS=1``): (c) the random-walk no-edge check of
the full default configuration on three seeds; deleting bars after ``X``; and the same
audit on the REAL data store (also needs ``AURUM_REAL_DATA_AUDIT=1``; bars through
2024-12-31 only - the real 2025+ holdout is never evaluated and no holdout ledger is
written).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import aurum.research.walkforward as wf_mod
from aurum.backtest.metrics import daily_returns
from aurum.core.config import AurumConfig, load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.research.walkforward import FoldPlan, WalkForwardReport, plan_folds, run_walk_forward

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "default.yaml"
DEFAULT_STRATEGIES = ["tsmom", "ema_cross", "donchian", "kalman_trend", "zscore_fade", "rsi2",
                      "bollinger_revert", "vol_squeeze", "orb", "macro_factor", "risk_off",
                      "intraday_seasonality", "ml_gbm", "meta_label"]
ML = ("ml_gbm", "meta_label")
TRAINABLE = ("intraday_seasonality", "ml_gbm", "meta_label")
#: events scheduled within this margin after the cutoff are left alone (public schedule)
EVENT_MARGIN = pd.Timedelta(days=8)
#: synthetic research span (13M training windows after the ML models' ~1-year feature
#: warm-up, 6W test blocks -> 3 folds) followed by a holdout
N_RESEARCH = 9_000
N_HOLDOUT = 700
WF_OVERRIDES = ("walkforward.train=13M", "walkforward.test=6W", "walkforward.n_boot=200",
                "walkforward.holdout_start=null", "walkforward.executor=process",
                "output.save_results=false", "output.tearsheet=false")
_FIT_LOG_ENV = "AURUM_E2E_FIT_LOG"


# ============================================================================================
# configuration & data
# ============================================================================================
def _slow_enabled(request: pytest.FixtureRequest) -> None:
    expr = request.config.getoption("markexpr", default="") or ""
    if os.environ.get("AURUM_SLOW_TESTS") == "1" or ("slow" in expr and "not slow" not in expr):
        return
    pytest.skip("slow audit test: run with `-m slow` or AURUM_SLOW_TESTS=1")


def default_config(*overrides: str, ml_skill_gate: bool = True, holdout_start: Any = None) -> AurumConfig:
    """``configs/default.yaml`` with shortened walk-forward windows and the process executor
    (execution only: results are executor-independent). ``ml_skill_gate=False`` switches off
    the ML skill gate; nothing else of the shipped configuration changes."""
    cfg = load_config(DEFAULT_CONFIG, overrides=[*WF_OVERRIDES, *overrides], env={})
    assert [s.name for s in cfg.strategies if s.enabled] == DEFAULT_STRATEGIES
    if not ml_skill_gate:
        for s in cfg.strategies:
            if s.name in ML:
                s.params["skill_gate_z"] = None
    if holdout_start is not None:
        cfg.walkforward.holdout_start = str(pd.Timestamp(holdout_start))
    return cfg


def _fedfunds(bars: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A point-in-time policy-rate series (percent, FRED-like publication lag) so the rate-based
    financing and the macro features see a ``fedfunds`` series on synthetic data too."""
    rng = np.random.default_rng(seed + 99)
    days = pd.date_range(bars.index[0].normalize() - pd.Timedelta(days=30), bars.index[-1].normalize(),
                         freq="B", tz="UTC")
    steps = np.where(rng.random(len(days)) < 0.02, rng.choice([-0.25, 0.25], len(days)), 0.0)
    f = pd.DataFrame({"value": np.clip(2.0 + np.cumsum(steps), 0.0, None)}, index=days)
    f.index.name = "date"
    f["available_at"] = f.index + pd.Timedelta(days=1, hours=21, minutes=30)
    return f


def synthetic_md(seed: int, *, n: int, model: str = "gbm", **kw: Any) -> MarketData:
    bars = make_synthetic_bars(n, "H1", seed=seed, model=model, **kw)
    macro = make_synthetic_macro(bars, seed=seed)
    macro["fedfunds"] = _fedfunds(bars, seed)
    events = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=14))
    return MarketData(bars=bars, macro=macro, events=events)


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


# ============================================================================================
# perturbation
# ============================================================================================
def perturb_bars_after(bars: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Bars with ``available_at > cutoff`` replaced by a different valid path (same timestamps)."""
    av = pd.DatetimeIndex(bars["available_at"])
    late = np.asarray(av > cutoff)
    if not late.any():
        return bars.copy()
    k = int(np.argmax(late))
    assert late[k:].all() and k > 0, "cutoff must lie inside the bars"
    rng = np.random.default_rng(seed)
    seg = bars.iloc[k:]
    m = len(seg)
    lc = np.log(bars["close"].to_numpy(dtype=float))
    r = np.diff(lc[k - 1:])                                   # r[j]: close-to-close return of bar k+j
    c = seg["close"].to_numpy(dtype=float)
    anatomy = {col: np.log(seg[col].to_numpy(dtype=float) / c) for col in ("open", "high", "low")}
    block = 48
    starts = np.arange(0, m, block)
    order = np.concatenate([np.arange(s, min(s + block, m)) for s in starts[rng.permutation(len(starts))]])
    close = np.exp(lc[k - 1] + np.cumsum(-1.25 * r[order]))  # shuffled, reversed, 25% more volatile
    out = bars.copy()
    new = {col: close * np.exp(anatomy[col][order]) for col in anatomy}   # keeps low <= o,c <= high
    new["close"] = close
    new["spread"] = seg["spread"].to_numpy(dtype=float)[order] * rng.uniform(0.6, 1.6, m)
    if "volume" in out.columns:
        new["volume"] = seg["volume"].to_numpy()[order]
    for col, v in new.items():
        out.iloc[k:, out.columns.get_loc(col)] = v
    out.attrs = dict(bars.attrs)
    return out


def perturb_macro_after(macro: dict[str, pd.DataFrame] | None, cutoff: pd.Timestamp, seed: int
                        ) -> dict[str, pd.DataFrame]:
    """Macro rows PUBLISHED after the cutoff (``available_at > cutoff``) changed / deleted."""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for name, f in (macro or {}).items():
        g = f.copy()
        late = np.asarray(pd.to_datetime(f["available_at"], utc=True) > cutoff)
        if late.any():
            n = int(late.sum())
            walk = np.exp(np.cumsum(rng.normal(0.0, 0.03, n)))
            for col in g.columns:
                if col == "available_at" or not pd.api.types.is_numeric_dtype(g[col]):
                    continue
                vals = g[col].to_numpy(dtype=float, copy=True)
                vals[late] = vals[late] * walk + rng.normal(0.0, 0.1, n)
                g[col] = vals
            drop = np.zeros(len(g), dtype=bool)
            drop[np.flatnonzero(late)[::7]] = True
            g = g.loc[~drop]
        g.attrs = dict(f.attrs)
        out[name] = g
    return out


def perturb_events_after(events: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    """Events scheduled more than ``EVENT_MARGIN`` after the cutoff moved / deleted / added."""
    if events is None or len(events) == 0:
        return events
    t = pd.to_datetime(events["time"], utc=True)
    late = np.asarray(t > cutoff + EVENT_MARGIN)
    if not late.any():
        return events.copy()
    e = events.copy()
    e["time"] = t.where(~late, t + pd.Timedelta(days=2, hours=3))
    keep = np.ones(len(e), dtype=bool)
    keep[np.flatnonzero(late)[::3]] = False
    e = e.loc[keep]
    fake = e.iloc[[-1]].copy()
    fake["time"] = cutoff + EVENT_MARGIN + pd.Timedelta(days=5, hours=5)
    return pd.concat([e, fake]).sort_values("time", kind="stable").reset_index(drop=True)


def perturb_after(md: MarketData, cutoff: Any, *, seed: int = 0) -> MarketData:
    cutoff = _utc(cutoff)
    return MarketData(bars=perturb_bars_after(md.bars, cutoff, seed),
                      macro=perturb_macro_after(md.macro, cutoff, seed + 1),
                      events=perturb_events_after(md.events, cutoff))


# ============================================================================================
# comparison
# ============================================================================================
def _books(rep: WalkForwardReport) -> dict[str, Any]:
    out = {"combined": rep.combined, **rep.strategy_results}
    if rep.benchmark is not None:
        out["benchmark"] = rep.benchmark
    return out


def pre_cutoff_diffs(a: WalkForwardReport, b: WalkForwardReport, bars: pd.DataFrame, cutoff: pd.Timestamp,
                     bars_b: pd.DataFrame | None = None) -> dict[str, float]:
    """Max |A - B| of every output decided / marked at or before ``cutoff`` (0.0 = identical;
    inf = different rows or NaN pattern). ``bars`` / ``bars_b`` are the bars A / B ran on
    (``bars_b`` defaults to ``bars``: same timestamps, perturbed values)."""
    def decision_times(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(pd.DatetimeIndex(frame["available_at"]), index=frame.index)

    av_a = decision_times(bars)
    av_b = decision_times(bars if bars_b is None else bars_b)

    def pre(s: Any, av: pd.Series, *, held: bool = False) -> Any:
        # ``held``: the position held DURING bar t was decided at the close of bar t-1
        t = av.shift(1) if held else av
        return s.loc[np.asarray(t.reindex(s.index) <= cutoff)]

    def diff(x: Any, y: Any) -> float:
        if not x.index.equals(y.index):
            return math.inf
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.shape != y.shape or (np.isnan(x) != np.isnan(y)).any():
            return math.inf
        d = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
        return float(d.max()) if d.size else 0.0

    def early_folds(rep: WalkForwardReport, av: pd.Series) -> pd.DataFrame:
        # combiner weights/FDM of every fold whose first decision is at/before the cutoff
        start = pd.DatetimeIndex(pd.to_datetime(rep.folds["test_start"], utc=True))
        keep = np.asarray(pd.DatetimeIndex(av.reindex(start)) <= cutoff)
        w = rep.weights.iloc[keep].copy()
        w.index = start[keep]
        return w

    out: dict[str, float] = {}
    fa, fb = pre(a.oos_forecasts, av_a), pre(b.oos_forecasts, av_b)
    assert 0 < len(fa) < len(a.oos_forecasts), "the cutoff must lie strictly inside the stitched OOS"
    out["fold_labels"] = 0.0 if fa.index.equals(fb.index) and fa["fold"].equals(fb["fold"]) else math.inf
    for c in a.oos_forecasts.columns.drop("fold"):
        out[f"forecast:{c}"] = diff(fa[c], fb[c])
    out["weights"] = diff(early_folds(a, av_a), early_folds(b, av_b))
    books_b = _books(b)
    for k, ra in _books(a).items():
        rb = books_b[k]
        out[f"equity:{k}"] = diff(pre(ra.equity, av_a), pre(rb.equity, av_b))
        out[f"position:{k}"] = diff(pre(ra.positions, av_a, held=True), pre(rb.positions, av_b, held=True))
        if ra.target is not None:
            out[f"target:{k}"] = diff(pre(ra.target, av_a), pre(rb.target, av_b))
    return out


def assert_identical_before(a: WalkForwardReport, b: WalkForwardReport, bars: pd.DataFrame,
                            cutoff: pd.Timestamp, bars_b: pd.DataFrame | None = None) -> dict[str, float]:
    d = pre_cutoff_diffs(a, b, bars, cutoff, bars_b)
    leaks = {k: v for k, v in d.items() if v != 0.0}
    assert not leaks, f"outputs at/before the cutoff {cutoff} changed when only the future changed: {leaks}"
    return d


def assert_research_identical(a: WalkForwardReport, b: WalkForwardReport) -> None:
    """Every research (non-holdout) output of two reports is bit-identical."""
    for attr in ("oos_forecasts", "weights", "folds", "fold_sharpe", "stats", "costs"):
        pd.testing.assert_frame_equal(getattr(a, attr), getattr(b, attr), check_exact=True, obj=attr)
    assert (a.pbo is None) == (b.pbo is None) and (a.pbo is None or a.pbo.pbo == b.pbo.pbo)
    books_b = _books(b)
    for k, ra in _books(a).items():
        rb = books_b[k]
        np.testing.assert_array_equal(ra.equity.to_numpy(), rb.equity.to_numpy(), err_msg=k)
        np.testing.assert_array_equal(ra.positions.to_numpy(), rb.positions.to_numpy(), err_msg=k)


def t_stat_daily(equity: pd.Series) -> float:
    """t-statistic of the mean DAILY return (= daily Sharpe x sqrt(days)); a flat book is 0."""
    dr = daily_returns(equity).dropna().to_numpy()
    if dr.size < 3 or not np.std(dr, ddof=1) > 0:
        return 0.0
    return float(dr.mean() / dr.std(ddof=1) * math.sqrt(dr.size))


# ============================================================================================
# spies: every Strategy.fit slice (also inside spawned workers) and every combiner fit
# ============================================================================================
def _install_fit_spy(path: str) -> None:
    """Wrap ``walkforward._train_md`` (the ONLY way training data reaches ``Strategy.fit``)
    so every call appends one JSON line to ``path``."""
    real = wf_mod._train_md
    if getattr(real, "_e2e_spy", False):
        return

    def spy(md: MarketData, a: int, b: int) -> MarketData:
        out = real(md, a, b)
        mx = [pd.to_datetime(v["available_at"], utc=True).max() for v in (out.macro or {}).values() if len(v)]
        rec = {"first_bar": str(out.bars.index[0]), "last_bar": str(out.bars.index[-1]),
               "decision_cutoff": str(out.bars["available_at"].iloc[-1]),
               "macro_max_available": str(max(mx)) if mx else None, "pid": os.getpid()}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return out

    spy._e2e_spy = True  # type: ignore[attr-defined]
    wf_mod._train_md = spy


def _recording_init_worker(ctx: Any, log_level: int, threads: int) -> None:  # pragma: no cover - subprocess
    """Process-pool initializer: the usual one + the fit spy (log path from the environment)."""
    wf_mod._init_worker(ctx, log_level, threads)
    _install_fit_spy(os.environ[_FIT_LOG_ENV])


def run_instrumented(md: MarketData, cfg: AurumConfig, log: Path
                     ) -> tuple[WalkForwardReport, list[dict[str, Any]], list[dict[str, Any]]]:
    """``run_walk_forward`` recording every ``Strategy.fit`` slice and every fold-combiner fit."""
    comb: list[dict[str, Any]] = []
    real_fit_comb = wf_mod._fit_fold_combiner
    real_train_md = wf_mod._train_md

    def spy_fit_comb(cfg_, keys, f_train, close, history, window, **kwargs):
        comb.append({"train_last": f_train.index[-1],
                     "history_last": None if history is None or len(history) == 0 else history.index[-1]})
        return real_fit_comb(cfg_, keys, f_train, close, history, window, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_FIT_LOG_ENV, str(log))
        mp.setattr(wf_mod, "_init_worker", _recording_init_worker)
        mp.setattr(wf_mod, "_fit_fold_combiner", spy_fit_comb)
        mp.setattr(wf_mod, "_train_md", real_train_md)     # restored on exit
        _install_fit_spy(str(log))                          # parent (serial / thread tasks)
        rep = run_walk_forward(md, cfg, write=False, holdout_ledger=False)
    fits = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in fits:
        for k in ("first_bar", "last_bar", "decision_cutoff", "macro_max_available"):
            r[k] = None if r[k] is None else pd.Timestamp(r[k])
    return rep, fits, comb


def check_no_in_sample_rows(rep: WalkForwardReport, md: MarketData, cfg: AurumConfig,
                            fits: list[dict[str, Any]], combiner_fits: list[dict[str, Any]]) -> None:
    """(a) The stitched OOS holds test rows only, each strictly after its own fold's training
    window by >= purge + embargo bars, and nothing fitted for a fold saw its test rows."""
    idx = md.bars.index
    plans, holdout, settings = plan_folds(idx, cfg, cfg.build_strategies())
    all_plans: list[FoldPlan] = plans + ([holdout] if holdout is not None else [])
    gap = int(settings["purge"]) + int(settings["embargo"])
    assert int(settings["purge"]) >= 24, "purge must cover the ML label horizon (24 bars)"
    assert len(rep.folds) == len(plans)
    oos = rep.oos_forecasts
    assert oos.index.is_monotonic_increasing and not oos.index.has_duplicates
    assert len(oos) == sum(p.n_test for p in plans)
    if holdout is not None:
        assert oos.index.max() < idx[holdout.test_start]
    for p, (_, row) in zip(plans, rep.folds.iterrows(), strict=True):
        assert (row["train_start"], row["train_end"]) == (idx[p.train_start], idx[p.train_end - 1])
        assert (row["test_start"], row["test_end"]) == (idx[p.test_start], idx[p.test_end - 1])
        assert p.test_start - p.train_end == gap == int(row["gap_bars"])
        rows = oos.loc[oos["fold"] == p.key]
        pos = idx.get_indexer(rows.index)
        assert len(rows) == p.n_test and (pos >= p.test_start).all() and (pos < p.test_end).all()
    # every Strategy.fit slice ends with its plan's training window (never inside the gap or a
    # test block); the macro rows it got were published by that slice's last decision
    assert len(fits) == len(TRAINABLE) * len(all_plans), fits
    for p in all_plans:
        mine = [r for r in fits if r["last_bar"] == idx[p.train_end - 1]]
        assert len(mine) == len(TRAINABLE), (p.key, fits)
        for r in mine:
            assert r["first_bar"] <= idx[p.train_start]           # warm-up history is the past
            assert r["decision_cutoff"] < idx[p.test_start]
            assert r["macro_max_available"] is None or r["macro_max_available"] <= r["decision_cutoff"], r
    # every combiner fit: its OOS history ends strictly before the fold's first test bar
    assert len(combiner_fits) == len(all_plans)
    for rec, p in zip(combiner_fits, all_plans, strict=True):
        assert rec["train_last"] == idx[p.train_end - 1]
        if rec["history_last"] is not None:
            assert rec["history_last"] < idx[p.test_start]
    assert any(r["history_last"] is not None for r in combiner_fits)


# ============================================================================================
# fixtures
# ============================================================================================
@pytest.fixture(scope="module")
def audit(tmp_path_factory) -> dict[str, Any]:
    """A (as is, instrumented), B (everything after X perturbed), C (only the holdout
    perturbed) - full default config, ML skill gate off, mildly trending synthetic data (so
    the combiner allocates and the books trade; leakage itself is data-agnostic)."""
    md = synthetic_md(3, n=N_RESEARCH + N_HOLDOUT, model="trend", regime_params={"phi": 0.12})
    hs = md.bars.index[N_RESEARCH]
    cfg = default_config(ml_skill_gate=False, holdout_start=hs)
    a, fits, comb = run_instrumented(md, cfg, tmp_path_factory.mktemp("e2e") / "fits.jsonl")
    folds = a.folds
    assert len(folds) >= 3
    k = len(folds) // 2                                    # X in the MIDDLE of a test block
    i0 = md.bars.index.get_loc(folds["test_start"].iloc[k])
    i1 = md.bars.index.get_loc(folds["test_end"].iloc[k])
    cutoff = _utc(md.bars["available_at"].iloc[(i0 + i1) // 2])
    b = run_walk_forward(perturb_after(md, cutoff, seed=11), cfg, write=False, holdout_ledger=False)
    last_research_decision = _utc(md.bars["available_at"].iloc[N_RESEARCH - 1])
    c = run_walk_forward(perturb_after(md, last_research_decision, seed=5), cfg, write=False,
                         holdout_ledger=False)
    return {"md": md, "cfg": cfg, "a": a, "b": b, "c": c, "cutoff": cutoff, "fits": fits,
            "combiner_fits": comb, "holdout_start": hs}


# ============================================================================================
# tests
# ============================================================================================
def test_perturbation_helpers_only_touch_the_future():
    """Negative control for the harness itself: the perturbed copy is identical up to the
    cutoff, different after it, and still valid market data."""
    from aurum.data.schema import validate_bars

    md = synthetic_md(1, n=3000)
    bars = md.bars
    cutoff = _utc(bars["available_at"].iloc[len(bars) // 2])
    p = perturb_after(md, cutoff, seed=1)
    validate_bars(p.bars)
    assert p.bars.index.equals(bars.index)
    early = bars["available_at"] <= cutoff
    pd.testing.assert_frame_equal(p.bars.loc[early], bars.loc[early])
    assert not np.allclose(p.bars.loc[~early, "close"], bars.loc[~early, "close"])
    for name, f in md.macro.items():
        g = p.macro[name]
        pub = pd.to_datetime(f["available_at"], utc=True) <= cutoff
        pd.testing.assert_frame_equal(g.loc[pd.to_datetime(g["available_at"], utc=True) <= cutoff], f.loc[pub],
                                      check_freq=False)
        assert len(g) < len(f) and not np.allclose(g["value"].iloc[-50:], f["value"].reindex(g.index).iloc[-50:])
    t0 = pd.to_datetime(md.events["time"], utc=True)
    t1 = pd.to_datetime(p.events["time"], utc=True)
    horizon = cutoff + EVENT_MARGIN
    pd.testing.assert_frame_equal(p.events.loc[(t1 <= horizon).to_numpy()].reset_index(drop=True),
                                  md.events.loc[(t0 <= horizon).to_numpy()].reset_index(drop=True))
    assert not t1[t1 > horizon].reset_index(drop=True).equals(t0[t0 > horizon].reset_index(drop=True))


def test_comparison_catches_a_planted_leak(audit):
    """Negative control for the comparison: a single forecast changed one bar before X is
    reported."""
    a, cutoff, bars = audit["a"], audit["cutoff"], audit["md"].bars
    import copy

    leaky = copy.copy(a)
    leaky.oos_forecasts = a.oos_forecasts.copy()
    av = bars["available_at"].reindex(a.oos_forecasts.index)
    t = a.oos_forecasts.index[np.flatnonzero((av <= cutoff).to_numpy())[-1]]
    leaky.oos_forecasts.loc[t, "ml_gbm"] += 1e-9
    d = pre_cutoff_diffs(a, leaky, bars, cutoff)
    assert d["forecast:ml_gbm"] > 0 and sum(v > 0 for v in d.values()) == 1


def test_future_perturbation_leaves_every_pre_cutoff_output_bit_identical(audit):
    """Full default config (14 strategies, ML skill gate off so ML forecasts are live): every
    OOS forecast, combiner weight, target, position and equity value decided/marked at or
    before X is bit-identical when bars, macro and events after X are replaced."""
    a, b, cutoff, md = audit["a"], audit["b"], audit["cutoff"], audit["md"]
    assert sorted(a.strategy_results) == sorted(DEFAULT_STRATEGIES) and not a.dropped
    d = assert_identical_before(a, b, md.bars, cutoff)
    assert len(d) > 3 * len(DEFAULT_STRATEGIES)
    # teeth: the ML strategies were trading before X, and the outputs moved after X
    av = md.bars["available_at"].reindex(a.oos_forecasts.index)
    pre, post = (av <= cutoff).to_numpy(), (av > cutoff).to_numpy()
    for k in ML:
        assert (a.oos_forecasts.loc[pre, k] != 0).any(), f"{k} is flat before X: the test has no teeth for it"
    for k in [*ML, "combined", "tsmom", "macro_factor"]:
        assert not np.array_equal(a.oos_forecasts.loc[post, k].to_numpy(), b.oos_forecasts.loc[post, k].to_numpy()), k
    post_eq = (md.bars["available_at"].reindex(a.combined.equity.index) > cutoff).to_numpy()
    assert (np.abs(a.combined.positions.to_numpy()[~post_eq]) > 0).any()
    assert not np.array_equal(a.combined.equity.to_numpy()[post_eq], b.combined.equity.to_numpy()[post_eq])
    assert not a.weights.iloc[-1:].equals(b.weights.iloc[-1:])     # fitted on perturbed OOS history


def test_stitched_oos_contains_no_in_sample_rows(audit):
    """(a) Fold geometry, every Strategy.fit slice (bars and the macro rows handed to it, also
    inside the process-pool workers) and every combiner fit end before the rows they are
    evaluated on - for the research folds and the holdout."""
    check_no_in_sample_rows(audit["a"], audit["md"], audit["cfg"], audit["fits"], audit["combiner_fits"])


def test_holdout_rows_never_influence_research_outputs(audit):
    """(b) Perturbing ONLY the holdout (its bars, the macro rows published in it, events well
    inside it) leaves every research output bit-identical; the holdout itself changes."""
    a, c = audit["a"], audit["c"]
    assert a.holdout is not None and c.holdout is not None
    assert a.oos_forecasts.index.max() < audit["holdout_start"] == a.holdout.start
    assert_research_identical(a, c)
    # the holdout combiner is fitted on research OOS only -> same weights; its books do change
    pd.testing.assert_series_equal(a.holdout.weights, c.holdout.weights, check_exact=True)
    assert not np.array_equal(a.holdout.benchmark.equity.to_numpy(), c.holdout.benchmark.equity.to_numpy())
    for k in ("tsmom", "macro_factor", *ML):
        fa, fc = a.holdout.forecasts[k].to_numpy(), c.holdout.forecasts[k].to_numpy()
        if k in ML and not (fa != 0).any():
            continue    # an ML model can sit inside its dead zone for a whole (short) holdout
        assert not np.array_equal(fa, fc), k


# ---- slow ---------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_walk_combined_oos_sharpe_is_insignificant(seed, request):
    """(c) Random walks (model gbm), full default config: the combined OOS book has no
    significant Sharpe, |t| < 2 (a flat book counts as t = 0)."""
    _slow_enabled(request)
    md = synthetic_md(seed, n=N_RESEARCH + 3000)
    rep = run_walk_forward(md, default_config(), write=False, holdout_ledger=False)
    t = t_stat_daily(rep.combined.equity)
    assert abs(t) < 2.0, f"seed {seed}: combined OOS t-stat {t:.2f} on a random walk"


@pytest.mark.slow
def test_deleting_future_bars_is_invisible_before_the_cutoff(audit, request):
    """Timestamp-structure look-ahead: bars after X are also DELETED (a whole day plus 5% at
    random) on top of the value perturbation. With the walk-forward windows given in BARS
    (the counts report A resolved "13M"/"6W" to) nothing decided before X may change:
    resampling completeness, bar-size inference, warm-ups and vol annualisation must not
    depend on how many bars exist after X.

    (With DURATION windows the bar count of "13M" is derived from the calendar density of
    the whole research span, so the fold boundaries themselves depend on how many bars exist
    later - a timestamp-only dependence, no price information.)"""
    _slow_enabled(request)
    a, cutoff, md = audit["a"], audit["cutoff"], audit["md"]
    p = perturb_after(md, cutoff, seed=21)
    bars = p.bars
    late = np.flatnonzero((bars["available_at"] > cutoff).to_numpy())
    rng = np.random.default_rng(0)
    drop = np.zeros(len(bars), dtype=bool)
    drop[rng.choice(late, size=len(late) // 20, replace=False)] = True
    drop |= np.asarray(bars.index.normalize() == bars.index[late[len(late) // 4]].normalize())
    assert not drop[: late[0]].any()
    kept = bars.loc[~drop].copy()
    kept.attrs = dict(bars.attrs)
    md_b = MarketData(bars=kept, macro=p.macro, events=p.events)
    cfg = default_config(f"walkforward.train={int(a.settings['train'])}",
                         f"walkforward.test={int(a.settings['test'])}", ml_skill_gate=False,
                         holdout_start=audit["holdout_start"])
    b = run_walk_forward(md_b, cfg, write=False, holdout_ledger=False)
    assert b.settings["train"] == a.settings["train"] and b.settings["purge"] == a.settings["purge"]
    assert_identical_before(a, b, md.bars, cutoff, md_b.bars)


REAL_OVERRIDES = ("data.start=2019-01-01", "data.end=2024-12-31T23:59:59Z", "walkforward.train=2Y",
                  "walkforward.test=6M", "walkforward.executor=auto")


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("AURUM_REAL_DATA_AUDIT") != "1"
                    or not (REPO / "data_store" / "xauusd_H1.parquet").exists(),
                    reason="real-data audit: set AURUM_REAL_DATA_AUDIT=1 (needs data_store/, ~10 min)")
def test_real_data_future_and_holdout_perturbation(monkeypatch, request):
    """The same audit on the real data store, 2019-2024 (the 2025+ holdout is never loaded):
    default config with the ML skill gate off and X in the middle of a test block; then a
    pseudo-holdout from 2024-07-01 perturbed alone (default config)."""
    _slow_enabled(request)
    monkeypatch.chdir(REPO)
    cfg = default_config(*REAL_OVERRIDES, ml_skill_gate=False)
    md = cfg.data.load()
    assert md.bars.index[-1] < pd.Timestamp("2025-01-01", tz="UTC")
    a = run_walk_forward(md, cfg, write=False, holdout_ledger=False)
    f = a.folds.iloc[len(a.folds) // 2]
    i0, i1 = md.bars.index.get_loc(f["test_start"]), md.bars.index.get_loc(f["test_end"])
    cutoff = _utc(md.bars["available_at"].iloc[(i0 + i1) // 2])
    b = run_walk_forward(perturb_after(md, cutoff, seed=11), cfg, write=False, holdout_ledger=False)
    assert_identical_before(a, b, md.bars, cutoff)

    hs = pd.Timestamp("2024-07-01", tz="UTC")
    cfg_h = default_config(*REAL_OVERRIDES, holdout_start=hs)
    n_research = int(md.bars.index.searchsorted(hs))
    last = _utc(md.bars["available_at"].iloc[n_research - 1])
    ha = run_walk_forward(md, cfg_h, write=False, holdout_ledger=False)
    hb = run_walk_forward(perturb_after(md, last, seed=5), cfg_h, write=False, holdout_ledger=False)
    assert_research_identical(ha, hb)
