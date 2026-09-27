"""``aurum`` command-line interface (also ``python -m aurum``).

Commands
--------
::

    aurum data download [--start --end --timeframes all|M15,H1,... --no-macro --offline]
    aurum data info [--config C | --dir DIR]
    aurum config show|validate --config C [--set key=value ...]
    aurum backtest --config C [--strategy ID ...] [--start --end] [--out DIR]
    aurum walkforward --config C [--out DIR] [--jobs N] [--executor process|thread|serial]
    aurum report --run DIR [--json]
    aurum strategies list [--json]
    aurum features list [--json]
    aurum desk demo [--journal-dir DIR]
    aurum desk run --config C [--at TIMESTAMP] [--from-run WF_DIR]
    aurum desk replay --config C --start S --end E [--every N] [--max-cost USD] [--yes] [--fake]
                      [--from-run WF_DIR]
    aurum rl train --config C
    aurum live artifact --config C [--out DIR] [--from-run WF_DIR]
    aurum live run --config C [--i-understand-real-money] [--max-cycles N]

Every command that takes ``--config`` also takes ``--set key.path=value`` (YAML-parsed,
repeatable). Secrets come only from environment variables (``ANTHROPIC_API_KEY``,
``MT5_PASSWORD``, ``AURUM_ALERT_WEBHOOK_URL``, ``AURUM_ARTIFACT_KEY``).

Exit codes: 0 success, 1 runtime error, 2 usage/configuration error, 3 optional component
unavailable (e.g. ``aurum.live.runner`` or torch missing), 4 confirmation required
(``desk replay`` without ``--yes``).

Safety: ``desk run`` / ``desk replay`` spend money on the Claude API — replay prints a loud
cost estimate and refuses to start without ``--yes``, and enforces a hard running budget.
``live run`` defaults to dry-run and never touches a real-money account unless the config
says ``live.allow_live_real: true`` AND ``--i-understand-real-money`` is passed.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("aurum.cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_CONFIRM = 4

_STAT_COLS = ["sharpe", "ci_lower", "ci_upper", "psr", "dsr", "cagr", "ann_vol", "max_drawdown",
              "n_trades", "total_costs", "n_halt_episodes"]


class CLIError(Exception):
    """User-facing failure with an exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _load_cfg(args: argparse.Namespace) -> Any:
    from aurum.core.config import load_config

    return load_config(getattr(args, "config", None), overrides=getattr(args, "set", None) or ())


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)


def _table(rows: dict[str, dict[str, Any]], cols: Sequence[str], label: str = "book") -> str:
    """Plain fixed-width table (no pandas styling dependency)."""
    names = list(rows)
    if not names:
        return "(empty)"
    width0 = max(len(label), *(len(n) for n in names))
    cells = {n: [_fmt(rows[n].get(c)) for c in cols] for n in names}
    widths = [max(len(c), *(len(cells[n][i]) for n in names)) for i, c in enumerate(cols)]
    head = label.ljust(width0) + "  " + "  ".join(c.rjust(w) for c, w in zip(cols, widths, strict=True))
    lines = [head, "-" * len(head)]
    for n in names:
        lines.append(n.ljust(width0) + "  " + "  ".join(x.rjust(w) for x, w in zip(cells[n], widths, strict=True)))
    return "\n".join(lines)


def _stats_rows(stats: Any) -> dict[str, dict[str, Any]]:
    import pandas as pd

    if isinstance(stats, pd.DataFrame):
        return {str(k): {c: (None if pd.isna(v) else v) for c, v in row.items()} for k, row in stats.iterrows()}
    return {str(k): dict(v) for k, v in (stats or {}).items()}


def _print_report(summary: dict[str, Any], *, out: Any = None) -> None:
    stream = out if out is not None else sys.stdout
    p = lambda *a: print(*a, file=stream)  # noqa: E731
    kind = summary.get("kind", "run")
    p(f"== {kind}: {summary.get('name')}  config {str(summary.get('config_hash'))[:12]}  "
      f"data {str(summary.get('data_hash'))[:12]}")
    st = summary.get("settings") or {}
    p(f"OOS {summary.get('oos_start')} -> {summary.get('oos_end')}  ({summary.get('n_oos_bars')} bars, "
      f"{summary.get('n_folds')} fold(s), train {st.get('train')} / test {st.get('test')} bars, "
      f"purge {st.get('purge')}, embargo {st.get('embargo')}, n_trials {summary.get('n_trials')})")
    p("")
    p(_table(_stats_rows(summary.get("stats")), _STAT_COLS))
    pbo = summary.get("pbo")
    if pbo:
        p(f"\nPBO (CSCV, {int(pbo.get('n_splits') or 0)} splits, {int(pbo.get('n_strategies') or 0)} strategies): "
          f"{_fmt(pbo.get('pbo'))}  prob OOS loss of IS-best {_fmt(pbo.get('prob_oos_loss'))}")
    hold = summary.get("holdout")
    if hold:
        p(f"\n== FINAL HOLDOUT (evaluated once) {hold.get('start')} -> {hold.get('end')}")
        p(_table(_stats_rows(hold.get("stats")), _STAT_COLS))
    notes = summary.get("notes") or []
    if notes:
        p("\nnotes:")
        for n in notes[:20]:
            p(f"  - {n}")
    timing = summary.get("timing") or {}
    if timing:
        keys = [k for k in ("features_s", "strategies_s", "combine_s", "backtests_s", "holdout_s", "stats_s",
                            "write_s", "wall_s") if k in timing]
        p("\ntiming: " + ", ".join(f"{k[:-2]} {timing[k]:.1f}s" for k in keys))
    if summary.get("out_dir"):
        d = Path(summary["out_dir"])
        p(f"\nreport dir: {d}")
        if (d / "tearsheet.html").exists():
            p(f"tearsheet:  {d / 'tearsheet.html'}")


def _utc(ts: str) -> Any:
    import pandas as pd

    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


# ----------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------
def cmd_data_download(args: argparse.Namespace) -> int:
    """Download Dukascopy bid/ask history and macro series into the data store (network)."""
    import datetime as dt

    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = args.end or (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
    t0 = time.time()
    if not args.no_bars:
        tfs = [t.strip().upper() for t in args.timeframes.split(",")] if args.timeframes != "all" else ["ALL"]
        if tfs == ["ALL"]:
            from aurum.data.dukascopy import build_dataset, prefetch_cache

            if not args.offline:
                rep = prefetch_cache(args.symbol, args.start, end, cache_dir=args.cache, max_workers=args.workers,
                                     time_budget_s=args.time_budget)
                print(f"prefetch: {rep.fetched} fetched, {rep.already_cached} cached, {rep.failed} failed, "
                      f"{rep.not_attempted} not attempted in {rep.elapsed_s:.0f}s"
                      + (" (time budget reached: re-run to resume)" if rep.stopped_early else ""))
            manifest = build_dataset(out, symbol=args.symbol, start=args.start, end=end, cache_dir=args.cache,
                                     max_workers=args.workers, offline=args.offline)
            for name, info in (manifest.get("files") or {}).items():
                print(f"  {name}: {info.get('rows')} rows {info.get('start')} -> {info.get('end')} "
                      f"sha256 {str(info.get('sha256'))[:12]}")
        else:
            from aurum.core.timeframes import get_timeframe
            from aurum.data.dukascopy import download_dukascopy
            from aurum.data.store import save_bars

            for tf in tfs:
                get_timeframe(tf)
                bars = download_dukascopy(args.symbol, args.start, end, timeframe=tf, cache_dir=args.cache,
                                          max_workers=args.workers, offline=args.offline)
                path = save_bars(bars, out / f"{args.symbol.lower()}_{tf}.parquet")
                print(f"  {tf}: {len(bars)} bars {bars.index[0]} -> {bars.index[-1]} -> {path}")
    if not args.no_macro:
        from aurum.data.macro import fetch_fred, fetch_yahoo_daily, save_macro_dir

        frames = fetch_yahoo_daily(start=args.macro_start, end=end, cache_dir=args.macro_cache)
        frames.update(fetch_fred(start=args.macro_start, end=end, cache_dir=args.macro_cache))
        path = save_macro_dir(frames, out / "macro")
        for k, v in sorted(frames.items()):
            print(f"  macro {k}: {len(v)} rows {v.index[0].date() if len(v) else '-'} -> "
                  f"{v.index[-1].date() if len(v) else '-'}")
        print(f"macro saved to {path}")
    print(f"done in {time.time() - t0:.0f}s ({pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC)")
    return EXIT_OK


def cmd_data_info(args: argparse.Namespace) -> int:
    """Summarise the bar files and macro series in the data store."""
    import pandas as pd

    from aurum.data.store import frame_hash, load_bars

    if args.config:
        cfg = _load_cfg(args)
        d = Path(cfg.data.dir)
        macro_dir = cfg.data.resolved_macro_dir()
    else:
        d = Path(args.dir)
        macro_dir = d / "macro"
    if not d.exists():
        raise CLIError(f"data directory {d} does not exist (run `aurum data download`)", EXIT_USAGE)
    files = sorted(d.glob("*.parquet"))
    if not files:
        print(f"no bar files in {d}")
    rows = {}
    for f in files:
        try:
            b = load_bars(f, verify_hash=True)
            ok = "ok"
        except ValueError as exc:
            b = load_bars(f, verify_hash=False)
            ok = f"HASH MISMATCH ({exc})"
        rows[f.name] = {"timeframe": b.attrs.get("timeframe"), "rows": len(b), "first": str(b.index[0]),
                        "last": str(b.index[-1]), "median_spread": float(b["spread"].median()),
                        "hash": frame_hash(b)[:12], "check": ok}
    if rows:
        print(_table(rows, ["timeframe", "rows", "first", "last", "median_spread", "hash", "check"], "file"))
    if macro_dir.exists():
        from aurum.data.macro import load_macro_dir

        macro = load_macro_dir(macro_dir)
        mrows = {k: {"rows": len(v), "first": str(v.index[0].date()) if len(v) else "-",
                     "last": str(v.index[-1].date()) if len(v) else "-",
                     "last_available_at": str(pd.Timestamp(v["available_at"].iloc[-1])) if len(v) else "-"}
                 for k, v in sorted(macro.items())}
        print("\nmacro:")
        print(_table(mrows, ["rows", "first", "last", "last_available_at"], "series"))
    manifest = d / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(f"\nmanifest: built {m.get('built_utc')} via {m.get('build_path')} from {m.get('source')}")
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------
def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    if args.action == "show":
        print(cfg.to_yaml())
    print(f"# config OK: {cfg.source or '<defaults>'}  hash {cfg.config_hash()}")
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# research
# ----------------------------------------------------------------------------------------
def _apply_output_args(cfg: Any, args: argparse.Namespace) -> None:
    if getattr(args, "no_write", False):
        cfg.output.save_results = False
    if getattr(args, "no_tearsheet", False):
        cfg.output.tearsheet = False


def cmd_backtest(args: argparse.Namespace) -> int:
    """Single-split backtest: fit before --start, evaluate [--start, --end]."""
    from aurum.research.walkforward import run_single_backtest

    cfg = _load_cfg(args)
    _apply_output_args(cfg, args)
    if args.jobs is not None:
        cfg.walkforward.n_jobs = args.jobs
    t0 = time.time()
    md = cfg.data.load()
    print(f"loaded {len(md.bars)} {cfg.data.timeframe} bars {md.bars.index[0]} -> {md.bars.index[-1]} "
          f"({time.time() - t0:.1f}s)")
    rep = run_single_backtest(md, cfg, only=args.strategy or None, start=args.start, end=args.end,
                              out_dir=args.out, include_holdout=args.include_holdout)
    _print_report(rep.summary())
    return EXIT_OK


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Walk-forward with per-fold refits, stitched OOS backtest, DSR/PBO, optional holdout."""
    from aurum.research.walkforward import run_walk_forward

    cfg = _load_cfg(args)
    _apply_output_args(cfg, args)
    if args.jobs is not None:
        cfg.walkforward.n_jobs = args.jobs
    if args.executor:
        cfg.walkforward.executor = args.executor
    t0 = time.time()
    md = cfg.data.load()
    print(f"loaded {len(md.bars)} {cfg.data.timeframe} bars {md.bars.index[0]} -> {md.bars.index[-1]} "
          f"({time.time() - t0:.1f}s); {len(cfg.enabled_strategies())} strategies")
    rep = run_walk_forward(md, cfg, out_dir=args.out)
    _print_report(rep.summary())
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from aurum.research.walkforward import load_summary

    try:
        summary = load_summary(args.run)
    except FileNotFoundError as exc:
        raise CLIError(str(exc), EXIT_USAGE) from exc
    summary.setdefault("out_dir", str(args.run))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# registries
# ----------------------------------------------------------------------------------------
def cmd_strategies_list(args: argparse.Namespace) -> int:
    from aurum.strategies.base import list_strategies

    rows = {}
    for name, cls in sorted(list_strategies().items()):
        try:
            warm = cls().warmup_bars
        except Exception:  # noqa: BLE001 - a listing must not fail on one strategy
            warm = None
        rows[name] = {"trainable": bool(cls.trainable), "warmup_bars": warm,
                      "description": (cls.description or (cls.__doc__ or "").strip().split("\n")[0])[:90],
                      "params": cls.default_params()}
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return EXIT_OK
    print(_table(rows, ["trainable", "warmup_bars", "description"], "strategy"))
    return EXIT_OK


def cmd_features_list(args: argparse.Namespace) -> int:
    from aurum.features.base import list_features
    from aurum.features.pipeline import CANONICAL_GROUP_ORDER

    order = {g: i for i, g in enumerate(CANONICAL_GROUP_ORDER)}
    specs = sorted(list_features(), key=lambda s: (order.get(s.name, 99), s.name))
    rows = {s.name: {"family": s.family, "lookback": s.lookback,
                     "requires": ",".join([*s.requires_macro, *(["events"] if s.requires_events else [])]) or "-",
                     "doc": (s.doc.split("\n")[0] if s.doc else "")[:80]} for s in specs}
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    print(_table(rows, ["family", "lookback", "requires", "doc"], "group"))
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# LLM desk
# ----------------------------------------------------------------------------------------
def _print_desk_result(res: Any) -> None:
    d = res.decision
    print(f"status:          {res.status}" + (f" ({res.failure_reason})" if res.failure_reason else ""))
    print(f"decision time:   {res.now}")
    print(f"quant forecast:  {res.quant_forecast:+.3f}")
    if d is not None:
        print(f"decision:        action={d.action} scale={d.scale} forecast={d.forecast} "
              f"confidence={d.confidence} horizon={d.horizon_bars} bars")
        print(f"rationale:       {d.rationale}")
        if d.key_risks:
            print(f"key risks:       {'; '.join(d.key_risks)}")
    print(f"final forecast:  {res.final_forecast:+.3f}  (policy {res.policy.mode}: {res.policy.action})")
    print(f"memos:           {len(res.memos)} " + ", ".join(f"{m.agent_id}={m.stance}" for m in res.memos))
    total = (res.usage or {}).get("total", {})
    print(f"cost:            ${res.cost_usd:.4f}  tokens in={total.get('input_tokens', 0)} "
          f"out={total.get('output_tokens', 0)} cache_read={total.get('cache_read_input_tokens', 0)}")
    if res.journal_path:
        print(f"journal:         {res.journal_path}")


def cmd_desk_demo(args: argparse.Namespace) -> int:
    """Offline desk cycle with a scripted fake Claude client (no network, no key)."""
    from aurum.agents import demo

    res = demo(journal_dir=args.journal_dir)
    print("LLM desk demo (offline scripted client, synthetic anonymised data)")
    _print_desk_result(res)
    return EXIT_OK


def _require_api_key(cfg: Any) -> None:
    if not cfg.agents.api_key:
        raise CLIError("ANTHROPIC_API_KEY is not set: the desk needs Claude API credentials "
                       "(export ANTHROPIC_API_KEY=...; never put keys in config files)", EXIT_USAGE)


def _desk_provider(md: Any, book: Any, cfg: Any, *, anonymise: bool, state: dict[str, Any] | None = None) -> Any:
    from aurum.agents import HistoricalDeskDataProvider
    from aurum.models.volatility import ewma_volatility

    replay = state is not None
    state = state if state is not None else {}

    def risk_status(now: Any) -> dict[str, Any]:
        # scale-free on purpose (no price levels): safe to show under anonymisation
        return {"drawdown_from_peak": -float(state.get("drawdown", 0.0)), "halted": bool(state.get("halted", False)),
                "equity_vs_initial_pct": 100.0 * (float(state.get("equity", cfg.backtest.initial_equity))
                                                  / cfg.backtest.initial_equity - 1.0),
                "book": "replay book (research risk limits)" if replay else "analysis only: flat, no open book"}

    def positions(now: Any) -> dict[str, Any]:
        lots = float(state.get("position", 0.0))
        return {"lots": lots, "side": "long" if lots > 0 else ("short" if lots < 0 else "flat")}

    return HistoricalDeskDataProvider(
        md, signals=book.signals, combined=book.combined, vol=ewma_volatility(md.bars["close"]),
        anonymise=anonymise, lookback_bars=cfg.agents.lookback_bars,
        backtest_stats={"combiner": book.combiner.explain().get("weights"),
                        "fit_window": f"{book.train_start} -> {book.train_end}"} if not anonymise else None,
        risk_status_fn=risk_status, positions_fn=positions)


def cmd_desk_run(args: argparse.Namespace) -> int:
    """One live-API desk cycle on the latest (or --at) bar close."""
    from aurum.agents import TradingDesk
    from aurum.core.types import MarketData
    from aurum.research.walkforward import fit_quant_book

    cfg = _load_cfg(args)
    _require_api_key(cfg)
    md = cfg.data.load()
    if args.at:
        at = _utc(args.at)
        keep = md.bars["available_at"] <= at
        if not keep.any():
            raise CLIError(f"no bar closed at or before {at}", EXIT_USAGE)
        md = MarketData(bars=md.bars.loc[keep], macro=md.macro, events=md.events)
    now = md.bars["available_at"].iloc[-1]
    print(f"fitting the quant book on data up to {now} ...")
    book = fit_quant_book(md, cfg, gap=False, oos_forecasts=_oos_history(args.from_run))
    q = float(book.combined.iloc[-1])
    desk_cfg = cfg.agents.desk_config()
    print(f"quant forecast {q:+.3f}; running ONE desk cycle (budget cap ${desk_cfg.max_cost_usd_per_cycle})")
    desk = TradingDesk(_desk_provider(md, book, cfg, anonymise=False), client=None, config=desk_cfg,
                       policy=cfg.agents.policy.build(), journal_dir=args.journal_dir or cfg.agents.journal_dir)
    res = desk.run_cycle(now, q, context={"mode": "manual desk run", "account": "none (analysis only)"})
    _print_desk_result(res)
    sizer = cfg.sizing.build()
    from aurum.models.volatility import ewma_volatility

    vol = float(ewma_volatility(md.bars["close"]).iloc[-1])
    lots = sizer.target_lots(res.final_forecast, vol, cfg.backtest.initial_equity, float(md.bars["close"].iloc[-1]),
                             cfg.instrument.build())
    print(f"sizing (before risk): {lots:+.2f} lots at equity {cfg.backtest.initial_equity:,.0f}, vol {vol:.1%}")
    return EXIT_OK


def _fake_desk_client() -> Any:
    """Deterministic offline client: the Chief halves the quant forecast every cycle."""
    from aurum.agents.testing import FakeAnthropicClient, decision_call, message

    def chief(kwargs: dict) -> Any:
        return message(decision_call(action="scale", scale=0.5, forecast=0.0, confidence=0.6,
                                     rationale="offline replay: halve exposure"))

    return FakeAnthropicClient(default=chief)


def cmd_desk_replay(args: argparse.Namespace) -> int:
    """Replay the LLM desk over history through the SAME sizer / risk / simulator."""
    import pandas as pd

    from aurum.agents import TradingDesk
    from aurum.backtest.engine import run_backtest
    from aurum.research.report import write_tearsheet
    from aurum.research.walkforward import book_statistics, fit_quant_book, provenance

    cfg = _load_cfg(args)
    every = int(args.every if args.every is not None else cfg.agents.replay_every)
    if every < 1:
        raise CLIError("--every must be >= 1", EXIT_USAGE)
    max_cost = float(args.max_cost if args.max_cost is not None else cfg.agents.replay_max_cost_usd)
    if not (math.isfinite(max_cost) and max_cost > 0):
        raise CLIError("--max-cost must be a positive number of USD", EXIT_USAGE)
    md = cfg.data.load()
    idx = md.bars.index
    s0 = int(idx.searchsorted(_utc(args.start), side="left"))
    e = _utc(args.end)
    if len(args.end.strip()) <= 10:
        e = e + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    s1 = int(idx.searchsorted(e, side="right")) - 1
    if s1 - s0 < 1:
        raise CLIError(f"replay window [{args.start}, {args.end}] has fewer than two bars", EXIT_USAGE)
    hs = cfg.walkforward.holdout_start
    if hs is not None and idx[s1] >= _utc(hs):
        print(f"WARNING: the replay window reaches into the walk-forward holdout (from {hs}); "
              "evaluating the desk there spends the holdout")
    n_cycles = (s1 - s0 - 1) // every + 1          # decisions are taken at bars s0 .. s1-1
    desk_cfg = cfg.agents.desk_config()
    per_cycle_cap = desk_cfg.max_cost_usd_per_cycle or float("inf")
    expected = n_cycles * cfg.agents.expected_cost_per_cycle_usd
    worst = min(n_cycles * per_cycle_cap, max_cost)  # each cycle is capped at the budget left
    banner = "=" * 78
    print(banner)
    print(f"LLM DESK REPLAY {'(OFFLINE FAKE CLIENT - no API calls)' if args.fake else '- THIS CALLS THE PAID CLAUDE API'}")
    print(f"window {idx[s0]} -> {idx[s1]}: {s1 - s0 + 1} bars, one desk cycle every {every} bars = {n_cycles} cycles")
    if not args.fake:
        print(f"estimated cost ~${expected:,.2f} (at ~${cfg.agents.expected_cost_per_cycle_usd:.2f}/cycle); "
              f"worst case ${worst:,.2f} (per-cycle cap ${per_cycle_cap}, capped by the budget)")
        print(f"hard budget for this replay: ${max_cost:,.2f} - every cycle may only spend what is left of it "
              "(an API call already in flight can overshoot by its own cost); once it is spent the desk is "
              "no longer called and the quant forecast is used for the remaining bars")
    print(banner)
    if not args.yes:
        print("refusing to start without --yes")
        return EXIT_CONFIRM
    if not args.fake:
        _require_api_key(cfg)

    book = fit_quant_book(md, cfg, fit_end=s0, gap=True, oos_forecasts=_oos_history(args.from_run))
    print(f"quant book fitted on {book.train_start} -> {book.train_end} (OOS from {idx[s0]})")
    state: dict[str, Any] = {}
    policy = cfg.agents.policy.build()
    desk = TradingDesk(_desk_provider(md, book, cfg, anonymise=cfg.agents.anonymise, state=state),
                       client=_fake_desk_client() if args.fake else None, config=desk_cfg,
                       policy=policy, journal_dir=args.journal_dir or cfg.agents.journal_dir)
    ledger = {"cost": 0.0, "cycles": 0, "budget_hit_at": None}
    decisions: list[dict[str, Any]] = []
    # The desk's standing view between cycles: its last DeskResult (None = no view: follow the
    # quant book) and the final forecast actually used on the previous bar ("hold" semantics).
    view: dict[str, Any] = {"res": None, "final": None}

    def between_cycles(q: float) -> float:
        """The last decision stands, RE-APPLIED by the same DecisionPolicy to the CURRENT quant
        forecast (what the live runner, which consults the desk at every bar, would do). Holding
        the last *forecast* instead would let an overlay desk keep a position the quant book has
        since reversed or closed, i.e. flip direction / add risk, which overlay mode forbids."""
        res = view["res"]
        if res is None or res.status == "skipped":
            return q
        return float(policy.apply(q, res.decision, previous_forecast=view["final"]))

    def hook(bar: int, now: Any, q: float, st: dict) -> float:
        state.update(st)
        if (bar - s0) % every:
            final = between_cycles(q)
        elif ledger["cost"] >= max_cost:
            if ledger["budget_hit_at"] is None:
                ledger["budget_hit_at"] = str(now)
                logger.warning("desk replay budget $%.2f exhausted at %s: quant forecast from here", max_cost, now)
            view["res"] = None
            final = q
            decisions.append({"time": now, "bar": bar, "quant": q, "final": q, "status": "budget_exhausted",
                              "action": None, "cost_usd": 0.0, "halted": st.get("halted")})
        else:
            # hard budget: this cycle may spend at most what is left of the replay budget
            remaining = max_cost - ledger["cost"]
            desk.config = dataclasses.replace(desk_cfg, max_cost_usd_per_cycle=min(per_cycle_cap, remaining))
            res = desk.run_cycle(now, q, previous_forecast=view["final"])
            ledger["cost"] += res.cost_usd
            ledger["cycles"] += 1
            view["res"] = res
            final = float(res.final_forecast)
            decisions.append({"time": now, "bar": bar, "quant": q, "final": final, "status": res.status,
                              "action": res.decision.action if res.decision else None, "cost_usd": res.cost_usd,
                              "halted": st.get("halted")})
            if ledger["cycles"] % 10 == 0:
                print(f"  {ledger['cycles']}/{n_cycles} cycles, spent ${ledger['cost']:.2f}")
        view["final"] = final
        return final

    b = cfg.backtest
    common = dict(sizer=cfg.sizing.build(), instrument=cfg.instrument.build(), costs=cfg.costs.build(),
                  initial_equity=b.initial_equity, stop_atr_mult=b.stop_atr_mult,
                  take_profit_atr_mult=b.take_profit_atr_mult, atr_period=b.atr_period,
                  stop_cooldown_bars=b.stop_cooldown_bars, event_horizon_hours=b.event_horizon_hours,
                  start=s0, end=s1)
    t0 = time.time()
    # hook at EVERY bar: the desk is consulted every `every` bars, its policy applied at each bar
    desk_res = run_backtest(md, book.combined, risk=cfg.risk.build("research", instrument=common["instrument"]),
                            forecast_hook=hook, hook_every=1, **common)
    quant_res = run_backtest(md, book.combined, risk=cfg.risk.build("research", instrument=common["instrument"]),
                             **common)
    rows = {"desk": book_statistics(desk_res, n_boot=300, seed=cfg.seed),
            "quant_only": book_statistics(quant_res, n_boot=300, seed=cfg.seed)}
    print(f"\nreplay done in {time.time() - t0:.1f}s: {ledger['cycles']} desk cycles, total cost ${ledger['cost']:.4f}"
          + (f", budget exhausted at {ledger['budget_hit_at']}" if ledger["budget_hit_at"] else ""))
    print(_table(rows, _STAT_COLS))
    out = Path(args.out) if args.out else Path(cfg.output.dir) / (
        f"desk_replay_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}_{cfg.config_hash()[:8]}")
    out.mkdir(parents=True, exist_ok=True)
    dec = pd.DataFrame(decisions)
    dec.to_csv(out / "desk_decisions.csv", index=False)
    desk_res.save(out / "books" / "desk")
    quant_res.save(out / "books" / "quant_only")
    summary = {"kind": "desk_replay", "fake_client": bool(args.fake), "every": every, "n_cycles": ledger["cycles"],
               "cost_usd": ledger["cost"], "budget_usd": max_cost, "budget_hit_at": ledger["budget_hit_at"],
               "window": [str(idx[s0]), str(idx[s1])], "stats": rows,
               "provenance": provenance(cfg, md), "anonymised": cfg.agents.anonymise}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if cfg.output.tearsheet:
        write_tearsheet(desk_res, out / "tearsheet.html", benchmark=quant_res.equity,
                        title="LLM desk replay (benchmark = quant-only book)",
                        extra={"desk_decisions": dec.tail(200),
                               "notes": [f"{ledger['cycles']} desk cycles every {every} bars; cost ${ledger['cost']:.4f}",
                                         "LLMs may have memorised historical prices; anonymisation mitigates "
                                         "but cannot eliminate this look-ahead (SPEC §10)."]},
                        dark_charts=cfg.output.dark_charts)
    print(f"\nreport dir: {out}")
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# RL / live (delegate to modules owned elsewhere; imported lazily, fail gracefully)
# ----------------------------------------------------------------------------------------
def cmd_rl_train(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    try:
        rl_train = importlib.import_module("aurum.rl.train")
    except ImportError as exc:
        raise CLIError(f"RL training is unavailable ({exc}); install the extras: pip install 'aurum[rl]'",
                       EXIT_UNAVAILABLE) from exc
    entry = getattr(rl_train, "train_from_config", None)
    if callable(entry):
        result = entry(cfg)
        print(result if not hasattr(result, "summary") else result.summary())
        return EXIT_OK
    import pandas as pd

    from aurum.core.types import MarketData

    md = cfg.data.load()
    r = cfg.rl
    day = pd.Timedelta(days=1)
    bars = md.bars
    if r.train_start:
        bars = bars.loc[bars.index >= _utc(r.train_start)]
    tr = bars.loc[bars.index < _utc(r.train_end) + day]
    va = bars.loc[(bars.index >= _utc(r.train_end) + day) & (bars.index < _utc(r.val_end) + day)]
    if len(tr) < 1000 or len(va) < 100:
        raise CLIError(f"rl: too little data (train {len(tr)} bars, validation {len(va)} bars)", EXIT_USAGE)
    params = {"seed": cfg.seed, **r.params}
    try:
        rl_cfg = rl_train.RLTrainConfig.from_dict(params)
        print(f"PPO training: train {tr.index[0]} -> {tr.index[-1]} ({len(tr)} bars), "
              f"validation {va.index[0]} -> {va.index[-1]} ({len(va)} bars) -> {r.out_dir}")
        res = rl_train.train_ppo(MarketData(bars=tr, macro=md.macro, events=md.events),
                                 MarketData(bars=va, macro=md.macro, events=md.events),
                                 config=rl_cfg, out_dir=r.out_dir)
    except ImportError as exc:
        raise CLIError(f"RL training needs torch/stable-baselines3 ({exc}): pip install 'aurum[rl]'",
                       EXIT_UNAVAILABLE) from exc
    except (TypeError, ValueError) as exc:
        raise CLIError(f"rl.params: {exc}", EXIT_USAGE) from exc
    print(f"artifact: {getattr(res, 'artifact_dir', r.out_dir)}")
    val = getattr(res, "val_eval", None)
    if val is not None and getattr(val, "metrics", None):
        print(json.dumps({k: v for k, v in val.metrics.items() if isinstance(v, (int, float))}, indent=2))
    return EXIT_OK


def _oos_history(run_dir: str | None) -> Any:
    """``oos_forecasts.parquet`` of a walk-forward run (combiner history), or None."""
    if not run_dir:
        return None
    import pandas as pd

    p = Path(run_dir) / "oos_forecasts.parquet"
    if not p.exists():
        raise CLIError(f"{p} not found (--from-run must be a walkforward run directory)", EXIT_USAGE)
    return pd.read_parquet(p)


def cmd_live_artifact(args: argparse.Namespace) -> int:
    """Fit the quant book on the latest training window and write the live trading artifact."""
    import pandas as pd

    from aurum.core.types import MarketData
    from aurum.data.store import frame_hash
    from aurum.research.walkforward import fit_quant_book

    cfg = _load_cfg(args)
    try:
        runner = importlib.import_module("aurum.live.runner")
    except ModuleNotFoundError as exc:
        raise CLIError(f"aurum.live.runner is not available ({exc})", EXIT_UNAVAILABLE) from exc
    md = cfg.data.load()
    if args.at:
        keep = md.bars["available_at"] <= _utc(args.at)
        md = MarketData(bars=md.bars.loc[keep], macro=md.macro, events=md.events)
    out = Path(args.out or cfg.live.artifact_dir or "artifacts/live")
    print(f"fitting {len(cfg.enabled_strategies())} strategies + combiner on data up to {md.bars.index[-1]} ...")
    sizer = cfg.live_sizer_kwargs()   # the live runner's sizer == the research sizer (or refuse)
    book = fit_quant_book(md, cfg, gap=False, oos_forecasts=_oos_history(args.from_run))
    stats = None
    if args.from_run:
        from aurum.research.walkforward import load_summary

        summ = load_summary(args.from_run)
        stats = {"source_run": str(args.from_run), "combined": summ.get("combined"),
                 "config_hash": summ.get("config_hash")}
    path = runner.save_artifact(
        out, strategies=book.strategies, pipeline=book.pipeline, combiner=book.combiner,
        symbol=cfg.live.symbol or cfg.instrument.symbol, timeframe=cfg.data.timeframe, sizer_config=sizer,
        backtest_stats=stats, overwrite=args.overwrite,
        training={"train_start": str(book.train_start), "train_end": str(book.train_end),
                  "config_hash": cfg.config_hash(), "data_hash": frame_hash(md.bars),
                  "created": pd.Timestamp.now(tz="UTC").isoformat()},
        notes=f"aurum live artifact from {cfg.source or 'defaults'}")
    w = book.combiner.explain()
    print(f"artifact: {path}  (train {book.train_start} -> {book.train_end}, fdm {w.get('fdm', float('nan')):.2f})")
    print("weights: " + ", ".join(f"{k}={v:.2f}" for k, v in (w.get("weights") or {}).items()))
    return EXIT_OK


def cmd_live_run(args: argparse.Namespace) -> int:
    """Run :mod:`aurum.live.runner` with this config (dry-run unless the config says otherwise)."""
    import yaml

    cfg = _load_cfg(args)
    lv = cfg.live
    # the single research -> live translation (raises ConfigError, i.e. exit 2, on a sizer the
    # runner cannot reproduce or on live.options that try to override typed/safety fields)
    mapping = cfg.live_runner_mapping()
    real_flag = bool(args.i_understand_real_money)
    if real_flag and not lv.allow_live_real:
        print("note: --i-understand-real-money has no effect because live.allow_live_real is false")
    if lv.allow_live_real and not real_flag:
        print("note: live.allow_live_real is true but --i-understand-real-money was not given: "
              "real-money accounts will be refused")
    print(f"live: broker={lv.broker} dry_run={lv.dry_run} allow_live_real={lv.allow_live_real and real_flag} "
          f"magic={lv.magic} artifact={lv.artifact_dir}")
    try:
        runner = importlib.import_module("aurum.live.runner")
    except ModuleNotFoundError as exc:
        if exc.name in ("aurum.live.runner", "aurum.live"):
            raise CLIError("aurum.live.runner is not available in this build (the live runner module "
                           "has not been installed yet)", EXIT_UNAVAILABLE) from exc
        raise CLIError(f"live runner dependency missing: {exc}", EXIT_UNAVAILABLE) from exc
    fn = getattr(runner, "run_from_config", None)
    if callable(fn):
        out = fn(cfg, i_understand_real_money=real_flag)
        return int(out) if isinstance(out, int) else EXIT_OK
    main_fn = getattr(runner, "main", None)
    live_cls = getattr(runner, "LiveConfig", None)
    if not callable(main_fn) or live_cls is None or not hasattr(live_cls, "from_mapping"):
        raise CLIError("aurum.live.runner exposes no known entry point (expected run_from_config(cfg, *, "
                       "i_understand_real_money) or main(argv) + LiveConfig.from_mapping)", EXIT_UNAVAILABLE)
    try:
        live_cls.from_mapping(mapping)  # validate before writing anything
    except (TypeError, ValueError) as exc:
        raise CLIError(f"live runner rejected the configuration: {exc}", EXIT_USAGE) from exc
    state = Path(lv.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    resolved = state / "aurum_live_config.yaml"
    resolved.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")  # no secrets in it
    argv = ["--config", str(resolved)]
    if real_flag:
        argv.append("--i-understand-real-money")
    if args.max_cycles is not None:
        argv += ["--max-cycles", str(args.max_cycles)]
    if args.until:
        argv += ["--until", args.until]
    print(f"resolved runner config: {resolved}")
    out = main_fn(argv)
    return int(out) if isinstance(out, int) else EXIT_OK


# ----------------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------------
def _add_config(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    p.add_argument("--config", "-c", required=required, help="YAML config file (see configs/)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set walkforward.test=3M (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aurum", description="Aurum v2: XAUUSD systematic research & trading.")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v INFO, -vv DEBUG logging")
    ap.add_argument("--traceback", action="store_true", help="show full tracebacks on errors")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # data
    data = sub.add_parser("data", help="download / inspect market data")
    dsub = data.add_subparsers(dest="action", metavar="ACTION")
    dsub.required = True
    dl = dsub.add_parser("download", help="Dukascopy bars + Yahoo/FRED macro into the data store (network)")
    dl.add_argument("--out", default="data_store")
    dl.add_argument("--symbol", default="XAUUSD")
    dl.add_argument("--start", default="2012-01-01")
    dl.add_argument("--end", default=None, help="inclusive end date (default: yesterday UTC)")
    dl.add_argument("--timeframes", default="all",
                    help="'all' (M15,H1,H4,D1,D1_nyclose with manifest) or a list like H1,H4")
    dl.add_argument("--cache", default="cache/dukascopy")
    dl.add_argument("--workers", type=int, default=4)
    dl.add_argument("--time-budget", type=float, default=None, help="seconds for the prefetch pass")
    dl.add_argument("--offline", action="store_true", help="build only from the local cache")
    dl.add_argument("--no-bars", action="store_true")
    dl.add_argument("--no-macro", action="store_true")
    dl.add_argument("--macro-start", default="2011-01-01")
    dl.add_argument("--macro-cache", default="cache/macro")
    dl.set_defaults(func=cmd_data_download)
    di = dsub.add_parser("info", help="summarise bar files and macro series")
    di.add_argument("--dir", default="data_store")
    _add_config(di, required=False)
    di.set_defaults(func=cmd_data_info)

    # config
    cp = sub.add_parser("config", help="show or validate a configuration")
    cp.add_argument("action", choices=["show", "validate"])
    _add_config(cp, required=False)
    cp.set_defaults(func=cmd_config)

    # backtest
    bt = sub.add_parser("backtest", help="single-split backtest (fit before --start, test after)")
    _add_config(bt)
    bt.add_argument("--strategy", action="append", default=[], metavar="ID", help="restrict to these strategies")
    bt.add_argument("--start", default=None, help="evaluation start (fit on earlier data); omit = IN-SAMPLE")
    bt.add_argument("--end", default=None)
    bt.add_argument("--out", default=None, help="report directory")
    bt.add_argument("--jobs", type=int, default=None)
    bt.add_argument("--include-holdout", action="store_true",
                    help="also evaluate bars after walkforward.holdout_start (spends the holdout)")
    bt.add_argument("--no-write", action="store_true", help="do not write the report directory")
    bt.add_argument("--no-tearsheet", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    # walkforward
    wf = sub.add_parser("walkforward", help="walk-forward research run -> report directory")
    _add_config(wf)
    wf.add_argument("--out", default=None, help="report directory")
    wf.add_argument("--jobs", type=int, default=None, help="parallel workers (0 = auto)")
    wf.add_argument("--executor", choices=["auto", "process", "thread", "serial"], default=None)
    wf.add_argument("--no-write", action="store_true")
    wf.add_argument("--no-tearsheet", action="store_true")
    wf.set_defaults(func=cmd_walkforward)

    # report
    rp = sub.add_parser("report", help="print the summary of a run directory")
    rp.add_argument("--run", required=True, help="run directory (contains summary.json)")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=cmd_report)

    # registries
    st = sub.add_parser("strategies", help="strategy registry")
    ssub = st.add_subparsers(dest="action", metavar="ACTION")
    ssub.required = True
    sl = ssub.add_parser("list", help="list registered strategies")
    sl.add_argument("--json", action="store_true")
    sl.set_defaults(func=cmd_strategies_list)
    ft = sub.add_parser("features", help="feature registry")
    fsub = ft.add_subparsers(dest="action", metavar="ACTION")
    fsub.required = True
    fl = fsub.add_parser("list", help="list registered feature groups")
    fl.add_argument("--json", action="store_true")
    fl.set_defaults(func=cmd_features_list)

    # desk
    dk = sub.add_parser("desk", help="LLM trading desk (Claude API)")
    ksub = dk.add_subparsers(dest="action", metavar="ACTION")
    ksub.required = True
    kd = ksub.add_parser("demo", help="offline demo cycle with a scripted fake client")
    kd.add_argument("--journal-dir", default=None)
    kd.set_defaults(func=cmd_desk_demo)
    kr = ksub.add_parser("run", help="ONE live-API desk cycle on the latest bar (needs ANTHROPIC_API_KEY)")
    _add_config(kr)
    kr.add_argument("--at", default=None, help="decision time (UTC); default: latest bar close")
    kr.add_argument("--journal-dir", default=None)
    kr.add_argument("--from-run", default=None, help="walk-forward run dir: combiner weights from its OOS forecasts")
    kr.set_defaults(func=cmd_desk_run)
    kp = ksub.add_parser("replay", help="replay the desk over history through the same sizer/risk (PAID)")
    _add_config(kp)
    kp.add_argument("--start", required=True)
    kp.add_argument("--end", required=True)
    kp.add_argument("--every", type=int, default=None, help="desk cycle every N bars (default agents.replay_every)")
    kp.add_argument("--max-cost", type=float, default=None, help="hard USD budget (default agents.replay_max_cost_usd)")
    kp.add_argument("--yes", action="store_true", help="confirm the cost estimate and start")
    kp.add_argument("--fake", action="store_true", help="offline scripted client (no API calls, for testing)")
    kp.add_argument("--journal-dir", default=None)
    kp.add_argument("--from-run", default=None, help="walk-forward run dir: combiner weights from its OOS forecasts")
    kp.add_argument("--out", default=None)
    kp.set_defaults(func=cmd_desk_replay)

    # rl
    rl = sub.add_parser("rl", help="reinforcement learning")
    rsub = rl.add_subparsers(dest="action", metavar="ACTION")
    rsub.required = True
    rt = rsub.add_parser("train", help="train a PPO policy (config section 'rl')")
    _add_config(rt)
    rt.set_defaults(func=cmd_rl_train)

    # live
    lv = sub.add_parser("live", help="live / paper trading")
    lsub = lv.add_subparsers(dest="action", metavar="ACTION")
    lsub.required = True
    la = lsub.add_parser("artifact", help="fit strategies + combiner on the latest data -> trading artifact")
    _add_config(la)
    la.add_argument("--out", default=None, help="artifact directory (default live.artifact_dir)")
    la.add_argument("--at", default=None, help="fit on bars closed at or before this UTC time")
    la.add_argument("--from-run", default=None,
                    help="walk-forward run dir: combiner weights from its OOS forecasts; OOS stats embedded")
    la.add_argument("--overwrite", action="store_true")
    la.set_defaults(func=cmd_live_artifact)
    lr = lsub.add_parser("run", help="run the live loop (dry-run by default)")
    _add_config(lr)
    lr.add_argument("--i-understand-real-money", action="store_true",
                    help="required (with live.allow_live_real: true) to trade a real-money account")
    lr.add_argument("--max-cycles", type=int, default=None)
    lr.add_argument("--until", default=None, help="stop at this UTC time (ISO)")
    lr.set_defaults(func=cmd_live_run)
    return ap


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root.setLevel(level)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``aurum`` / ``python -m aurum``; returns the process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse: --help (0) or usage error (2)
        return int(exc.code or 0) if isinstance(exc.code, int) else EXIT_USAGE
    _setup_logging(args.verbose)
    from aurum.core.config import ConfigError

    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"aurum: configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except CLIError as exc:
        print(f"aurum: {exc}", file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"aurum: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("aurum: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level: report and exit non-zero
        if args.traceback or args.verbose >= 2:
            raise
        logger.debug("unhandled error", exc_info=True)
        print(f"aurum: error: {type(exc).__name__}: {exc} (use --traceback for details)", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
