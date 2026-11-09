"""``aurum`` command-line interface (also ``python -m aurum``).

Commands
--------
::

    aurum [-v|-vv] [--traceback] [--version] COMMAND ...
    aurum data download [--out DIR] [--start D] [--end D] [--timeframes all|M15,H1,...] [--offline]
                        [--no-bars] [--no-macro] [--time-budget S] [--workers N] [--cache DIR]
    aurum data info [--dir DIR | --config C]
    aurum config show|validate [--config C] [--set key=value ...]
    aurum backtest --config C [--strategy ID ...] [--start TS] [--end TS] [--out DIR] [--jobs N]
                   [--include-holdout] [--no-write] [--no-tearsheet]
    aurum walkforward --config C [--out DIR] [--jobs N] [--executor auto|process|thread|serial]
                      [--no-write] [--no-tearsheet]
    aurum train-final --config C --out DIR [--cutoff TS] [--from-run WF_DIR] [--overwrite]
                      [--allow-config-mismatch] [--jobs N] [--executor E] [--no-write] [--no-tearsheet]
    aurum report --run DIR [--json]
    aurum strategies list [--json]
    aurum features list [--json]
    aurum desk demo [--journal-dir DIR]
    aurum desk run --config C [--at TS] [--journal-dir DIR] [--from-run WF_DIR]
    aurum desk replay --config C --start TS --end TS [--every N] [--max-cost USD] [--yes] [--fake]
                      [--journal-dir DIR] [--from-run WF_DIR] [--out DIR]
    aurum rl train --config C
    aurum live artifact --config C [--out DIR] [--at TS] [--from-run WF_DIR] [--overwrite]
    aurum live run --config C [--i-understand-real-money] [--max-cycles N] [--until TS]
    aurum live status --config C [--json]
    aurum live reset-halt --config C [--confirm RESET] [--reason TEXT]

Extension commands (``aurum.cli_ext.register_all(subparsers)``, e.g. ``sweep``, ``stress``,
``explain``, ``dashboard``) are added when that module is importable; a broken extension
only logs a warning and never affects the core commands above.

Every command that takes ``--config`` also takes ``--set key.path=value`` (YAML-parsed,
repeatable) and validates the config STRICTLY before doing any work: every strategy is
built from the registry with its params, the feature pipeline, the instrument ranges, the
live runner's settings and ``rl.params`` are checked, and all problems are reported at once
(exit 2). Secrets come only from environment variables (``ANTHROPIC_API_KEY``,
``MT5_PASSWORD``, ``AURUM_ALERT_WEBHOOK_URL``, ``AURUM_ARTIFACT_KEY``).

Exit codes: 0 success, 1 runtime error, 2 usage/configuration error (also: a refused
``live reset-halt``), 3 optional component unavailable (e.g. ``aurum.live.runner`` or torch
missing), 4 confirmation required (``desk replay`` without ``--yes``, ``live reset-halt``
without ``--confirm RESET``).

Safety: ``desk run`` / ``desk replay`` spend money on the Claude API and need
``agents.enabled: true`` (see ``configs/desk_overlay.yaml``) — replay prints a loud cost
estimate, refuses to start without ``--yes`` and enforces a hard running budget (``--fake``
replays are offline and free). ``live run`` defaults to dry-run and never touches a
real-money account unless the config says ``live.allow_live_real: true`` AND
``--i-understand-real-money`` is passed. ``live reset-halt`` is the ONLY CLI path that clears
the risk manager's kill switch: it refuses while a runner holds the state directory, needs
the literal ``--confirm RESET`` and logs who/when/why to the state directory.
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
def _load_cfg(args: argparse.Namespace, *, validate: bool = True) -> Any:
    """The command's config: ``--config`` + ``--set``, validated STRICTLY (unknown strategy
    names are errors too), so a command fails before it loads data or fits anything."""
    from aurum.core.config import load_config

    return load_config(getattr(args, "config", None), overrides=getattr(args, "set", None) or (),
                       validate=validate, strict=True)


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
    st = summary.get("settings") or {}
    if st.get("in_sample"):
        kind = f"{kind} (IN-SAMPLE: fitted and evaluated on the same bars)"
    p(f"== {kind}: {summary.get('name')}  config {str(summary.get('config_hash'))[:12]}  "
      f"data {str(summary.get('data_hash'))[:12]}")
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
        import importlib.util

        from aurum.data.macro import fetch_fred, fetch_yahoo_daily, save_macro_dir

        if importlib.util.find_spec("yfinance") is None:
            print("WARNING: yfinance is not installed (pip install 'aurum[data]'): the Yahoo series "
                  "(dxy, us10y, vix, spx, silver, gold_fut, oil) are skipped unless already cached; "
                  "macro_factor and risk_off need them")
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
    """Strict validation of a config: report EVERY problem at once, what was built, and which
    checks were skipped (optional components not installed)."""
    from aurum.core.config import ConfigError

    cfg = _load_cfg(args, validate=False)      # keys/types (structural problems raise here)
    rep = cfg.check(strict=True)
    if args.action == "show" and rep.ok:
        print(cfg.to_yaml())
    for w in rep.warnings:
        print(f"# warning: {w}")
    for s in rep.skipped:
        print(f"# skipped: {s}")
    if not rep.ok:
        raise ConfigError(rep.problems)
    print(f"# checked: {', '.join(rep.checked)}")
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
    if args.start is None and cfg.backtest.start is None:
        print("== IN-SAMPLE backtest: no --start (and no backtest.start), so strategies and combiner are "
              "fitted and evaluated on the SAME bars. The numbers below are not an estimate of future "
              "performance; pass --start or run `aurum walkforward`.")
    t0 = time.time()
    md = cfg.data.load()
    print(f"loaded {len(md.bars)} {cfg.data.timeframe} bars {md.bars.index[0]} -> {md.bars.index[-1]} "
          f"({time.time() - t0:.1f}s)")
    rep = run_single_backtest(md, cfg, only=args.strategy or None, start=args.start, end=args.end,
                              out_dir=args.out, include_holdout=args.include_holdout)
    _print_report(rep.summary())
    return EXIT_OK


def _cli_holdout_ledger(cfg: Any) -> Path | None:
    """Holdout ledger for a CLI walk-forward: the default (next to the run directories) when
    results are written; with ``--no-write`` the holdout is still SHOWN to a human, so the look
    is recorded in ``output.dir/holdout_ledger.jsonl`` all the same."""
    if cfg.output.save_results:
        return None
    from aurum.research.walkforward import HOLDOUT_LEDGER

    return Path(cfg.output.dir) / HOLDOUT_LEDGER


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
    rep = run_walk_forward(md, cfg, out_dir=args.out, holdout_ledger=_cli_holdout_ledger(cfg))
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
    def requires(s: Any) -> str:
        macro = list(s.requires_macro)
        # a group that reads md.macro but needs no particular series declares it on its
        # function (FeatureSpec.requires_macro names series the pipeline would enforce)
        if not macro and getattr(s.fn, "requires_macro", None) == "any":
            macro = ["macro(any)"]
        return ",".join([*macro, *(["events"] if s.requires_events else [])]) or "-"

    rows = {s.name: {"family": s.family, "lookback": s.lookback, "requires": requires(s),
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


_DESK_DISABLED = ("agents.enabled is false in this config: the LLM desk is opt-in because it calls the paid "
                  "Claude API. Use a config that enables it (configs/desk_overlay.yaml sets agents.enabled: true "
                  "with an overlay policy and a per-cycle budget) or pass --set agents.enabled=true")
_NO_API_KEY = ("ANTHROPIC_API_KEY is not set: the desk needs Claude API credentials "
               "(export ANTHROPIC_API_KEY=...; never put keys in config files)")


def _desk_preflight(cfg: Any, *, need_key: bool = True) -> None:
    """A paid desk command needs ``agents.enabled: true`` AND an API key: report both at once."""
    problems = []
    if not cfg.agents.enabled:
        problems.append(_DESK_DISABLED)
    if need_key and not cfg.agents.api_key:
        problems.append(_NO_API_KEY)
    if len(problems) == 1:
        raise CLIError(problems[0], EXIT_USAGE)
    if problems:
        raise CLIError("the desk cannot run:\n  - " + "\n  - ".join(problems), EXIT_USAGE)


def _warn_equal_weights(book: Any, cfg: Any, from_run: str | None, what: str) -> None:
    """Say loudly when the combiner fell back to equal weights (no OOS history)."""
    basis = str(getattr(book, "combiner_basis", "") or "")
    if not basis.startswith("equal") or cfg.combiner.method in ("equal", "fixed"):
        return
    why = ("no --from-run was given, so there is no out-of-sample forecast history to fit weights on"
           if not from_run else f"the OOS history of {from_run} is shorter than walkforward.combiner_min_obs="
                                f"{cfg.walkforward.combiner_min_obs} bars")
    print(f"WARNING: {what}: the combiner fell back to EQUAL weights ({basis}): {why}. Pass --from-run "
          "<walk-forward run dir> (or use `aurum train-final`) for weights fitted out of sample.")


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
    _desk_preflight(cfg)
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
    _warn_equal_weights(book, cfg, args.from_run, "desk run")
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
              "evaluating the desk there spends the holdout (the look is recorded in the holdout ledger "
              "next to the report directory and counts as a separate trial in later walk-forwards)")
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
        if expected > max_cost:
            print(f"NOTE: the estimate exceeds the budget: expect the budget to run out after about "
                  f"{int(max_cost // max(cfg.agents.expected_cost_per_cycle_usd, 1e-9))} of {n_cycles} cycles "
                  "(use a larger --every, a shorter window or a higher --max-cost)")
        if not cfg.agents.enabled:
            print("NOTE: agents.enabled is false in this config: the replay will refuse to start "
                  "(see configs/desk_overlay.yaml)")
    print(banner)
    if not args.yes:
        print("refusing to start without --yes")
        return EXIT_CONFIRM
    if not args.fake:
        _desk_preflight(cfg)
    elif not cfg.agents.enabled:
        print("note: agents.enabled is false; allowed for this --fake replay only (offline, no API calls)")

    book = fit_quant_book(md, cfg, fit_end=s0, gap=True, oos_forecasts=_oos_history(args.from_run))
    _warn_equal_weights(book, cfg, args.from_run, "desk replay")
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
    out = Path(args.out) if args.out else Path(cfg.output.dir) / (
        f"desk_replay_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}_{cfg.config_hash()[:8]}")
    t0 = time.time()
    # hook at EVERY bar: the desk is consulted every `every` bars, its policy applied at each bar
    desk_res = run_backtest(md, book.combined, risk=cfg.risk.build("research", instrument=common["instrument"]),
                            forecast_hook=hook, hook_every=1, **common)
    quant_res = run_backtest(md, book.combined, risk=cfg.risk.build("research", instrument=common["instrument"]),
                             **common)
    rows = {"desk": book_statistics(desk_res, n_boot=300, seed=cfg.seed),
            "quant_only": book_statistics(quant_res, n_boot=300, seed=cfg.seed)}
    # A replay that evaluates bars inside the walk-forward holdout is a LOOK at it: record it
    # before anything is shown, in the ledger a later walk-forward reads (next to the report
    # directory, like a walk-forward's own), under its own config hash so the desk counts as a
    # separate trial (DSR n_trials) when the holdout is evaluated again.
    look = _record_desk_replay_look(cfg, md, out, idx[s0], idx[s1], fake=bool(args.fake), rows=rows,
                                    n_cycles=ledger["cycles"])
    print(f"\nreplay done in {time.time() - t0:.1f}s: {ledger['cycles']} desk cycles, total cost ${ledger['cost']:.4f}"
          + (f", budget exhausted at {ledger['budget_hit_at']}" if ledger["budget_hit_at"] else ""))
    print(_table(rows, _STAT_COLS))
    out.mkdir(parents=True, exist_ok=True)
    dec = pd.DataFrame(decisions)
    dec.to_csv(out / "desk_decisions.csv", index=False)
    desk_res.save(out / "books" / "desk")
    quant_res.save(out / "books" / "quant_only")
    summary = {"kind": "desk_replay", "fake_client": bool(args.fake), "every": every, "n_cycles": ledger["cycles"],
               "cost_usd": ledger["cost"], "budget_usd": max_cost, "budget_hit_at": ledger["budget_hit_at"],
               "window": [str(idx[s0]), str(idx[s1])], "stats": rows,
               "provenance": provenance(cfg, md), "anonymised": cfg.agents.anonymise,
               "holdout_look": str(look) if look is not None else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if cfg.output.tearsheet:
        write_tearsheet(desk_res, out / "tearsheet.html", benchmark=quant_res.equity,
                        title="LLM desk replay (benchmark = quant-only book)",
                        extra={"desk_decisions": dec.tail(200),
                               "notes": [f"{ledger['cycles']} desk cycles every {every} bars; cost ${ledger['cost']:.4f}",
                                         "LLMs may have memorised historical prices; anonymisation mitigates "
                                         "but cannot eliminate this look-ahead (SPEC §10)."]},
                        dark_charts=cfg.output.dark_charts)
    if look is not None:
        print(f"holdout look recorded in {look}")
    print(f"\nreport dir: {out}")
    return EXIT_OK


def _record_desk_replay_look(cfg: Any, md: Any, out: Path, start: Any, end: Any, *, fake: bool,
                             rows: dict[str, Any], n_cycles: int) -> Path | None:
    """Record a desk replay's look at the walk-forward holdout (None if the window ends before
    ``walkforward.holdout_start`` or there is none). The ledger sits next to the report
    directory (``<out>/../holdout_ledger.jsonl``, i.e. ``output.dir`` by default), where a
    walk-forward writing there reads it. The config hash gets a ``:desk`` suffix (``:desk-fake``
    for the offline scripted client) so the replay is a distinct trial, not a re-run of the
    quant config."""
    if cfg.walkforward.holdout_start is None:
        return None
    from aurum.research.walkforward import record_holdout_look

    def sharpe(k: str) -> Any:
        r = rows.get(k)
        return r.get("sharpe") if isinstance(r, dict) else None

    entry = {"start": start, "end": end, "kind": "desk_replay",
             "config_hash": f"{cfg.config_hash()}:desk{'-fake' if fake else ''}", "fake_client": fake,
             "run_dir": str(out), "n_desk_cycles": int(n_cycles),
             "metrics": {"desk_sharpe": sharpe("desk"), "quant_only_sharpe": sharpe("quant_only")}}
    return record_holdout_look(out.parent, entry, cfg=cfg, md=md)


# ----------------------------------------------------------------------------------------
# RL / live (delegate to modules owned elsewhere; imported lazily, fail gracefully)
# ----------------------------------------------------------------------------------------
def _rl_train_config(rl_train: Any, cfg: Any) -> Any:
    """The PPO run's :class:`RLTrainConfig`: the environment comes from the top-level
    ``costs`` / ``sizing`` / ``risk.research`` / ``instrument`` / ``backtest.initial_equity``
    (the SAME settings the research backtests use) via
    ``aurum.rl.train.rl_config_from_aurum``; explicit ``rl.params.env`` keys override."""
    build = getattr(rl_train, "rl_config_from_aurum", None)
    if callable(build):
        return build(cfg)
    logger.warning("aurum.rl.train has no rl_config_from_aurum: the environment uses RLTrainConfig defaults, "
                   "NOT the config's costs/sizing/risk")
    return rl_train.RLTrainConfig.from_dict({"seed": cfg.seed, **cfg.rl.params})


def cmd_rl_train(args: argparse.Namespace) -> int:
    """PPO on the shared simulator; env costs/sizing/risk from the config's top-level sections."""
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

    try:   # before loading data: a bad rl.params / env mapping is a configuration error
        rl_cfg = _rl_train_config(rl_train, cfg)
    except (TypeError, ValueError) as exc:
        raise CLIError(f"rl.params: {exc}", EXIT_USAGE) from exc
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
    env = getattr(rl_cfg, "env", None)
    over = sorted((r.params.get("env") or {}).keys()) if isinstance(r.params.get("env"), dict) else []
    if env is not None:
        sizer = getattr(env, "sizer", {}) or {}
        src = {k: ("rl.params.env" if k in over else "config") for k in ("sizer", "risk", "costs")}
        risk = "off" if getattr(env, "risk", None) is None else f"limits from {src['risk']}"
        print(f"env: sizer from {src['sizer']} (target_vol {sizer.get('target_vol')}, max_leverage "
              f"{sizer.get('max_leverage')}), risk {risk}, costs from {src['costs']}"
              + (f"; rl.params.env overrides {over}" if over else ""))
    try:
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
    """Stitched OOS forecasts of a walk-forward run (combiner history), or None.

    ``oos_forecasts.parquet`` (research folds) plus, when the run evaluated a holdout,
    ``holdout/oos_forecasts.parquet`` (forecasts of the model fitted before the holdout:
    also genuinely out-of-sample). Consumers only use rows before their own fit end.
    """
    if not run_dir:
        return None
    import pandas as pd

    p = Path(run_dir) / "oos_forecasts.parquet"
    if not p.exists():
        raise CLIError(f"{p} not found (--from-run must be a walkforward run directory)", EXIT_USAGE)
    oos = pd.read_parquet(p)
    h = Path(run_dir) / "holdout" / "oos_forecasts.parquet"
    if h.exists():
        ho = pd.read_parquet(h)
        cols = [c for c in oos.columns if c in ho.columns and c != "fold"]
        oos = pd.concat([oos[cols], ho[cols]]).sort_index()
        oos = oos.loc[~oos.index.duplicated(keep="first")]
    return oos


def cmd_live_artifact(args: argparse.Namespace) -> int:
    """Fit the quant book on the latest training window and write the live trading artifact."""
    import pandas as pd

    from aurum.core.types import MarketData
    from aurum.data.store import frame_hash
    from aurum.research.walkforward import fit_quant_book

    cfg = _load_cfg(args)
    sizer = cfg.live_sizer_kwargs()   # the live runner's sizer == the research sizer (or refuse)
    out = Path(args.out or cfg.live.artifact_dir or "artifacts/live")
    if out.exists() and not args.overwrite:   # refuse BEFORE loading data and fitting anything
        raise CLIError(f"artifact {out} already exists (pass --overwrite to replace it)", EXIT_USAGE)
    try:
        runner = importlib.import_module("aurum.live.runner")
    except ModuleNotFoundError as exc:
        raise CLIError(f"aurum.live.runner is not available ({exc})", EXIT_UNAVAILABLE) from exc
    oos = _oos_history(args.from_run)
    md = cfg.data.load()
    if args.at:
        keep = md.bars["available_at"] <= _utc(args.at)
        if not keep.any():
            raise CLIError(f"no bar closed at or before {args.at}", EXIT_USAGE)
        md = MarketData(bars=md.bars.loc[keep], macro=md.macro, events=md.events)
    print(f"fitting {len(cfg.enabled_strategies())} strategies + combiner on data up to {md.bars.index[-1]} ...")
    book = fit_quant_book(md, cfg, gap=False, oos_forecasts=oos)
    _warn_equal_weights(book, cfg, args.from_run, "live artifact")
    stats = None
    if args.from_run:
        from aurum.research.walkforward import load_summary

        summ = load_summary(args.from_run)
        stats = {"source_run": str(args.from_run), "combined": summ.get("combined"),
                 "config_hash": summ.get("config_hash")}
    path = runner.save_artifact(
        out, strategies=book.strategies, pipeline=book.pipeline, combiner=book.combiner,
        symbol=cfg.live.symbol or cfg.instrument.symbol, timeframe=cfg.data.timeframe, sizer_config=sizer,
        backtest_stats=stats, overwrite=args.overwrite, feature_reference=book.feature_reference,
        training={"train_start": str(book.train_start), "train_end": str(book.train_end),
                  "config_hash": cfg.config_hash(), "data_hash": frame_hash(md.bars),
                  "combiner_basis": book.combiner_basis,
                  "created": pd.Timestamp.now(tz="UTC").isoformat()},
        notes=f"aurum live artifact from {cfg.source or 'defaults'}")
    w = book.combiner.explain()
    print(f"artifact: {path}  (train {book.train_start} -> {book.train_end}, fdm {w.get('fdm', float('nan')):.2f})")
    print("weights: " + ", ".join(f"{k}={v:.2f}" for k, v in (w.get("weights") or {}).items()))
    return EXIT_OK


def cmd_train_final(args: argparse.Namespace) -> int:
    """Production trading artifact: everything fitted on ALL data up to ``--cutoff``.

    * Data: bars whose close (``available_at``) is at or before the cutoff (default: the last
      bar); macro rows published by then. Nothing after the cutoff is read.
    * Combiner weights come from the stitched OUT-OF-SAMPLE walk-forward forecasts (never the
      in-sample training-window forecasts): from ``--from-run`` (its research OOS plus holdout
      OOS forecasts; its config hash must match unless ``--allow-config-mismatch``), or from a
      walk-forward run on the same data executed now (written like ``aurum walkforward``,
      including the holdout evaluation and its ledger entry). The fit follows the fold rule
      (last ``walkforward.train`` bars of OOS history, all when anchored; equal weights when
      fewer than ``combiner_min_obs``), net of the configured costs.
    * Pipeline scaler and trainable strategies: fitted on the training window that ENDS at
      the cutoff (no purge/holdout held back; its length follows ``walkforward.train`` /
      ``anchored``, i.e. exactly the protocol the walk-forward validated).
    * ``feature_reference.json``: PSI reference of the TRAIN-window transformed features (live
      drift monitor); the resolved config, provenance and OOS stats go into the manifest /
      ``backtest.json``. The directory loads with :func:`aurum.live.runner.load_artifact`.
    """
    import pandas as pd

    from aurum.core.types import MarketData
    from aurum.data.store import frame_hash
    from aurum.research.walkforward import (
        _jsonable,
        _pit_macro,
        fit_quant_book,
        load_summary,
        provenance,
        run_walk_forward,
    )

    cfg = _load_cfg(args)
    _apply_output_args(cfg, args)
    if args.jobs is not None:
        cfg.walkforward.n_jobs = args.jobs
    if args.executor:
        cfg.walkforward.executor = args.executor
    sizer = cfg.live_sizer_kwargs()   # refuse early: the live runner could not size this book
    try:
        runner = importlib.import_module("aurum.live.runner")
    except ModuleNotFoundError as exc:
        raise CLIError(f"aurum.live.runner is not available ({exc})", EXIT_UNAVAILABLE) from exc
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise CLIError(f"artifact {out} exists (pass --overwrite)", EXIT_USAGE)

    md = cfg.data.load()
    avail = pd.DatetimeIndex(md.bars["available_at"])
    cutoff = _utc(args.cutoff) if args.cutoff else avail[-1]
    keep = avail <= cutoff
    if keep.sum() < 2:
        raise CLIError(f"fewer than two bars closed at or before the cutoff {cutoff}", EXIT_USAGE)
    bars = md.bars.loc[keep]
    decision_cutoff = pd.Timestamp(bars["available_at"].iloc[-1])
    md = MarketData(bars=bars, macro=_pit_macro(md.macro, decision_cutoff), events=md.events)
    print(f"train-final: {len(bars)} {cfg.data.timeframe} bars {bars.index[0]} -> {bars.index[-1]} "
          f"(last close {decision_cutoff}); {len(cfg.enabled_strategies())} strategies")

    # ---- combiner history: stitched OUT-OF-SAMPLE forecasts ---------------------------------
    keys = [s.key for s in cfg.enabled_strategies()]
    if args.from_run:
        summ = load_summary(args.from_run)
        if summ.get("config_hash") != cfg.config_hash():
            msg = (f"--from-run {args.from_run} was produced by config {str(summ.get('config_hash'))[:12]}, this "
                   f"config is {cfg.config_hash()[:12]}: its OOS forecasts validate a different protocol")
            if not args.allow_config_mismatch:
                raise CLIError(msg + " (pass --allow-config-mismatch to use them anyway)", EXIT_USAGE)
            print(f"WARNING: {msg}")
        hist = _oos_history(args.from_run)
        oos_source = str(args.from_run)
        oos_stats = summ.get("combined") or {}
        wf_hash = summ.get("config_hash")
    else:
        print("no --from-run: running the walk-forward on the same data for OUT-OF-SAMPLE combiner weights ...")
        wf_cfg = cfg
        hs = cfg.walkforward.holdout_start
        if hs is not None and _utc(hs) > bars.index[-1]:
            # the cutoff precedes the configured holdout: there is no holdout in this data
            print(f"note: walkforward.holdout_start {hs} is after the cutoff; the walk-forward runs without a holdout")
            wf_cfg = dataclasses.replace(cfg, walkforward=dataclasses.replace(cfg.walkforward, holdout_start=None))
        rep = run_walk_forward(md, wf_cfg, holdout_ledger=_cli_holdout_ledger(wf_cfg))
        _print_report(rep.summary())
        parts = [rep.oos_forecasts]
        if rep.holdout is not None and rep.holdout.forecasts is not None:
            parts.append(rep.holdout.forecasts)
        cols = [k for k in keys if all(k in p.columns for p in parts)]
        hist = pd.concat([p[cols] for p in parts]).sort_index()
        oos_source = str(rep.out_dir) if rep.out_dir is not None else "in-process walk-forward (not written)"
        oos_stats = _jsonable(rep.stats.loc["combined"].to_dict()) if "combined" in rep.stats.index else {}
        wf_hash = rep.config_hash
    missing = [k for k in keys if k not in hist.columns]
    if missing:
        raise CLIError(f"OOS forecasts lack strategies {missing} (was the walk-forward run with this config?)",
                       EXIT_USAGE)
    hist = hist.loc[hist.index <= bars.index[-1], keys]
    if cfg.walkforward.combiner_fit != "oos" and cfg.combiner.method != "fixed":
        print("note: walkforward.combiner_fit=train is ignored here - production weights always come from the "
              "stitched out-of-sample forecasts")

    # ---- the final fit ----------------------------------------------------------------------
    book = fit_quant_book(md, cfg, gap=False, oos_forecasts=hist, combiner_fit="oos")
    if book.combiner_basis.startswith("equal"):
        print(f"WARNING: combiner fell back to {book.combiner_basis}: the OOS history is too short for "
              f"fitted weights (walkforward.combiner_min_obs={cfg.walkforward.combiner_min_obs})")
    ex = book.combiner.explain()
    backtest_stats = {"source_run": oos_source, "combined": oos_stats, "config_hash": wf_hash,
                      "basis": "stitched out-of-sample walk-forward (combined book)"}
    training = _jsonable({
        "kind": "train_final", "cutoff": decision_cutoff, "last_bar": bars.index[-1],
        "train_start": book.train_start, "train_end": book.train_end,
        "config_hash": cfg.config_hash(), "data_hash": frame_hash(bars),
        "combiner_basis": book.combiner_basis, "combiner_cost_basis": ex.get("cost_basis"),
        "oos_source": oos_source, "oos_rows": len(hist),
        "oos_start": hist.index[0] if len(hist) else None, "oos_end": hist.index[-1] if len(hist) else None,
        "config": cfg.to_dict(), "provenance": provenance(cfg, md),
        "created": pd.Timestamp.now(tz="UTC"),
    })
    path = runner.save_artifact(
        out, strategies=book.strategies, pipeline=book.pipeline, combiner=book.combiner,
        symbol=cfg.live.symbol or cfg.instrument.symbol, timeframe=cfg.data.timeframe, sizer_config=sizer,
        backtest_stats=_jsonable(backtest_stats), feature_reference=book.feature_reference,
        training=training, overwrite=args.overwrite,
        notes=f"aurum train-final from {cfg.source or 'defaults'} (cutoff {decision_cutoff})")
    ref = book.feature_reference
    print(f"\nartifact: {path}")
    print(f"  fit window {book.train_start} -> {book.train_end}; combiner {book.combiner_basis} "
          f"[{ex.get('cost_basis')}]")
    print("  weights: " + ", ".join(f"{k}={v:.2f}" for k, v in (ex.get("weights") or {}).items())
          + f"  (sum {ex.get('weights_sum', float('nan')):.2f}, fdm {ex.get('fdm', float('nan')):.2f})")
    print(f"  feature reference: {len(ref.columns) if ref is not None else 0} columns; OOS source: {oos_source}")
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
          f"magic={lv.magic} artifact={lv.artifact_dir} desk={'on' if lv.use_desk else 'off'}")
    try:
        runner = importlib.import_module("aurum.live.runner")
    except ModuleNotFoundError as exc:
        if exc.name in ("aurum.live.runner", "aurum.live"):
            raise CLIError("aurum.live.runner is not available in this build (the live runner module "
                           "has not been installed yet)", EXIT_UNAVAILABLE) from exc
        raise CLIError(f"live runner dependency missing: {exc}", EXIT_UNAVAILABLE) from exc
    main_fn = getattr(runner, "main", None)
    live_cls = getattr(runner, "LiveConfig", None)
    if not callable(main_fn) or live_cls is None or not hasattr(live_cls, "from_mapping"):
        raise CLIError("aurum.live.runner exposes no known entry point (expected main(argv) + "
                       "LiveConfig.from_mapping)", EXIT_UNAVAILABLE)
    try:
        live_cfg = live_cls.from_mapping(mapping)  # validate before writing anything
    except (TypeError, ValueError) as exc:
        raise CLIError(f"live runner rejected the configuration: {exc}", EXIT_USAGE) from exc
    # the runner's own environment check (LLM desk enabled: anthropic importable and
    # ANTHROPIC_API_KEY set) also runs before anything is written or connected
    check = getattr(runner, "check_desk_prerequisites", None)
    if callable(check):
        try:
            check(live_cfg)
        except ValueError as exc:
            raise CLIError(f"live runner cannot start: {exc}", EXIT_USAGE) from exc
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
    # One INFO line per bar decision is the operator's view of the loop: show the runner's
    # INFO records at the default verbosity (the root logger stays at WARNING for the rest).
    runner_log = logging.getLogger("aurum.live.runner")
    level = runner_log.level
    if runner_log.getEffectiveLevel() > logging.INFO:
        runner_log.setLevel(logging.INFO)
    try:
        out = main_fn(argv)
    finally:
        runner_log.setLevel(level)
    return int(out) if isinstance(out, int) else EXIT_OK


#: files the live runner keeps in ``live.state_dir``
_RISK_STATE = "risk_state.json"
_RUNNER_STATE = "runner_state.json"
_OMS_STATE = "oms_state.json"
_HEARTBEAT = "heartbeat.json"
_DECISIONS = "decisions.jsonl"
_ALERTS = "alerts.jsonl"
_LOCK = "runner.lock"


def _probe_runner_lock(path: Path) -> dict[str, Any]:
    """READ-ONLY check whether a runner holds ``runner.lock`` (never creates or writes it).

    ``held`` is True/False on POSIX (a non-blocking ``flock`` probe on a read-only handle,
    released at once) and None when it cannot be determined (no file -> False). ``pid`` is
    the last holder's pid as written by the runner (informational; it is not cleared on exit).
    """
    info: dict[str, Any] = {"path": str(path), "exists": path.exists(), "held": False, "pid": None}
    if not info["exists"]:
        return info
    try:
        with open(path, encoding="utf-8") as fh:
            info["pid"] = fh.read().strip() or None
            if sys.platform.startswith("win"):  # pragma: no cover - msvcrt needs a writable handle
                info["held"] = None
                return info
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                info["held"] = True
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                info["held"] = False
    except OSError as exc:
        info["held"] = None
        info["error"] = str(exc)
    return info


def _read_state_json(path: Path) -> tuple[Any, str | None]:
    """``(data, error)`` for a state file: ``(None, None)`` when it does not exist."""
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"


def _tail_jsonl(path: Path, n: int, *, where: Any = None) -> list[dict[str, Any]]:
    """The last ``n`` records of a JSONL log (optionally filtered); torn lines are skipped."""
    from collections import deque

    if not path.exists():
        return []
    out: deque[dict[str, Any]] = deque(maxlen=max(n, 1))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and (where is None or where(rec)):
                out.append(rec)
    return list(out)


def _risk_summary(risk: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(risk, dict):
        return None
    peak, last, day0 = risk.get("peak_equity"), risk.get("last_equity"), risk.get("day_start_equity")
    num = (int, float)
    dd = (1.0 - last / peak) if isinstance(peak, num) and isinstance(last, num) and peak > 0 else None
    day = (last / day0 - 1.0) if isinstance(day0, num) and isinstance(last, num) and day0 > 0 else None
    return {"halted": risk.get("halted"), "halt_kind": risk.get("halt_kind"), "halt_reason": risk.get("halt_reason"),
            "halted_at": risk.get("halted_at"), "peak_equity": peak, "last_equity": last, "drawdown": dd,
            "day_start_equity": day0, "day_return": day, "trades_today": risk.get("trades_today"),
            "last_time": risk.get("last_time")}


def _live_status(cfg: Any) -> dict[str, Any]:
    """Everything ``aurum live status`` shows, read from ``live.state_dir`` without writing."""
    import pandas as pd

    lv = cfg.live
    state = Path(lv.state_dir)
    out: dict[str, Any] = {
        "config": cfg.source, "state_dir": str(state), "state_dir_exists": state.is_dir(),
        "live": {"broker": lv.broker, "dry_run": lv.dry_run, "allow_live_real": lv.allow_live_real,
                 "magic": lv.magic, "symbol": lv.symbol or cfg.instrument.symbol, "artifact_dir": lv.artifact_dir,
                 "use_desk": lv.use_desk},
        "now": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    out["runner_lock"] = _probe_runner_lock(state / _LOCK)
    hb, hb_err = _read_state_json(state / _HEARTBEAT)
    if isinstance(hb, dict) and hb.get("time"):
        try:
            hb["age_seconds"] = round((pd.Timestamp.now(tz="UTC") - pd.Timestamp(hb["time"])).total_seconds(), 1)
        except (TypeError, ValueError):
            pass
    out["heartbeat"] = hb if hb_err is None else {"error": hb_err}
    risk, risk_err = _read_state_json(state / _RISK_STATE)
    if risk_err is not None:
        out["risk"] = {"error": risk_err, "note": "the runner starts HALTED on an unreadable risk state"}
    else:
        out["risk"] = _risk_summary(risk)
    rs, rs_err = _read_state_json(state / _RUNNER_STATE)
    out["runner"] = ({"error": rs_err} if rs_err else None if rs is None else
                     {k: rs.get(k) for k in ("last_bar", "last_avail", "n_decisions", "prev_final_forecast",
                                             "cooldown_left", "pending", "updated")})
    oms, oms_err = _read_state_json(state / _OMS_STATE)
    if oms_err is not None:
        out["oms"] = {"error": oms_err, "note": "the OMS refuses to trade on an unreadable state file"}
    elif isinstance(oms, dict):
        dec = oms.get("decisions") if isinstance(oms.get("decisions"), dict) else {}
        last_key = max(dec) if dec else None
        last = dec.get(last_key) if last_key else None
        open_ = sorted(k for k, v in dec.items() if isinstance(v, dict)
                       and v.get("status") not in ("complete", "rejected", "superseded", "conflict"))
        out["oms"] = {"magic": oms.get("magic"), "symbol": oms.get("symbol"), "n_decisions": len(dec),
                      "last_decision": last_key, "last_status": last.get("status") if isinstance(last, dict) else None,
                      "last_target": last.get("target") if isinstance(last, dict) else None,
                      "not_terminal": open_[-5:]}
    else:
        out["oms"] = None
    paper, paper_err = _read_state_json(state / "paper_broker.json")
    if paper_err is not None:
        out["paper_broker"] = {"error": paper_err}
    elif isinstance(paper, dict):
        pos = paper.get("positions") if isinstance(paper.get("positions"), list) else []
        out["paper_broker"] = {"balance": paper.get("balance"), "n_orders": paper.get("n_orders"),
                               "net_lots": round(sum(float(p.get("lots", 0.0) or 0.0) for p in pos
                                                     if isinstance(p, dict)), 6),
                               "n_positions": len(pos)}
    last = _tail_jsonl(state / _DECISIONS, 1, where=lambda r: r.get("type") == "decision")
    if last:
        d = last[0]
        out["last_decision"] = {k: d.get(k) for k in ("time", "bar_time", "status", "combined", "final_forecast",
                                                      "requested_lots", "approved_lots", "current_lots", "equity",
                                                      "halted") if k in d}
    out["last_halt_reset"] = (_tail_jsonl(state / _DECISIONS, 1, where=lambda r: r.get("type") == "halt_reset")
                              or [None])[0]
    out["recent_alerts"] = [{k: a.get(k) for k in ("time", "level", "kind", "message")}
                            for a in _tail_jsonl(state / _ALERTS, 5)]
    return out


def _print_live_status(s: dict[str, Any]) -> None:
    lv = s["live"]
    print(f"live status: {s['state_dir']}  (config {s['config'] or '<defaults>'})")
    print(f"  config:      broker={lv['broker']} dry_run={lv['dry_run']} allow_live_real={lv['allow_live_real']} "
          f"magic={lv['magic']} symbol={lv['symbol']} desk={'on' if lv['use_desk'] else 'off'}")
    if not s["state_dir_exists"]:
        print("  (state directory does not exist: no runner has used it yet)")
        return
    lock = s["runner_lock"]
    running = {True: "RUNNING (lock held)", False: "not running", None: "unknown"}[lock["held"]]
    print(f"  runner:      {running}" + (f", last pid {lock['pid']}" if lock.get("pid") else ""))
    hb = s.get("heartbeat")
    if isinstance(hb, dict) and "error" not in hb:
        print(f"  heartbeat:   {hb.get('status')} at {hb.get('time')} (age {hb.get('age_seconds', '?')} s), "
              f"mode {hb.get('mode')}, last bar {hb.get('last_bar')}, {hb.get('n_decisions')} decisions"
              + (f", pending {hb['pending']}" if hb.get("pending") else ""))
    elif isinstance(hb, dict):
        print(f"  heartbeat:   {hb['error']}")
    risk = s.get("risk")
    if risk is None:
        print("  risk:        no risk state file")
    elif "error" in risk:
        print(f"  risk:        {risk['error']} ({risk['note']})")
    else:
        state = "HALTED" if risk.get("halted") else "active"
        print(f"  risk:        {state}" + (f" [{risk.get('halt_kind')}] {risk.get('halt_reason')} since "
                                           f"{risk.get('halted_at')}" if risk.get("halted") else ""))
        print(f"               equity {_fmt(risk.get('last_equity'))} / peak {_fmt(risk.get('peak_equity'))} "
              f"(drawdown {_fmt(risk.get('drawdown'))}), day return {_fmt(risk.get('day_return'))}, "
              f"trades today {risk.get('trades_today')}, as of {risk.get('last_time')}")
        if risk.get("halted"):
            print("               clear it (runner stopped) with: aurum live reset-halt -c CONFIG --confirm RESET "
                  "--reason '...'")
    oms = s.get("oms")
    if isinstance(oms, dict) and "error" in oms:
        print(f"  oms:         {oms['error']} ({oms['note']})")
    elif isinstance(oms, dict):
        print(f"  oms:         {oms['n_decisions']} decisions, last {oms['last_decision']} -> {oms['last_status']} "
              f"(target {_fmt(oms['last_target'])})"
              + (f"; not terminal: {oms['not_terminal']}" if oms["not_terminal"] else ""))
    else:
        print("  oms:         no OMS state file")
    pb = s.get("paper_broker")
    if isinstance(pb, dict):
        print("  paper:       " + (pb["error"] if "error" in pb else
                                   f"balance {_fmt(pb.get('balance'))}, net {pb.get('net_lots')} lots in "
                                   f"{pb.get('n_positions')} position(s), {pb.get('n_orders')} orders"))
    run = s.get("runner")
    if isinstance(run, dict) and "error" not in run:
        print(f"  last bar:    {run.get('last_bar')} ({run.get('n_decisions')} decisions), previous final "
              f"forecast {_fmt(run.get('prev_final_forecast'))}")
    d = s.get("last_decision")
    if d:
        print(f"  decision:    {d.get('time')} status={d.get('status')} combined {_fmt(d.get('combined'))} final "
              f"{_fmt(d.get('final_forecast'))} -> approved {_fmt(d.get('approved_lots'))} lots "
              f"(current {_fmt(d.get('current_lots'))}), equity {_fmt(d.get('equity'))}")
    r = s.get("last_halt_reset")
    if r:
        print(f"  last reset:  {r.get('time')} by {r.get('user')}@{r.get('host')}: {r.get('reason')}")
    for a in s.get("recent_alerts") or []:
        print(f"  alert:       {a.get('time')} {a.get('level')} {a.get('kind')}: {a.get('message')}")


def cmd_live_status(args: argparse.Namespace) -> int:
    """Read-only summary of the runner, risk (kill switch), OMS and paper-broker state."""
    cfg = _load_cfg(args, validate=False)   # status must work even while a config has problems
    s = _live_status(cfg)
    if args.json:
        print(json.dumps(s, indent=2, default=str))
    else:
        _print_live_status(s)
    return EXIT_OK


def cmd_live_reset_halt(args: argparse.Namespace) -> int:
    """Clear the live risk manager's kill switch: the deliberate, logged human action.

    Order: show the halt state -> refuse while a runner holds ``runner.lock`` (it keeps its own
    in-memory state and rewrites the file every bar) -> require the literal ``--confirm RESET``
    -> take the runner lock (so no runner can start mid-reset) -> re-read the state under the
    lock -> ``StandardRiskManager.reset_halt(confirm="RESET")`` (re-bases peak/day-start
    equity to the last observed equity) -> append who/when/why to ``alerts.jsonl`` and
    ``decisions.jsonl`` in the state directory.
    """
    import getpass
    import os
    import socket

    import pandas as pd

    cfg = _load_cfg(args, validate=False)
    state = Path(cfg.live.state_dir)
    risk_path = state / _RISK_STATE
    if not state.is_dir():
        raise CLIError(f"state directory {state} does not exist: nothing to reset (live.state_dir)", EXIT_USAGE)
    if not risk_path.exists():
        prev = [n for n in (_RUNNER_STATE, _OMS_STATE) if (state / n).exists()]
        hint = (f" (but {' and '.join(prev)} exist: the runner will start HALTED because the kill-switch state "
                "is unknown; start it once so it records that halt, stop it, then reset)") if prev else ""
        raise CLIError(f"no {_RISK_STATE} in {state}: nothing to reset{hint}", EXIT_USAGE)

    from aurum.risk.manager import RiskLimits, StandardRiskManager

    try:
        limits = cfg.risk.limits("live")
    except (TypeError, ValueError):
        limits = RiskLimits()   # the limits do not matter for a reset; the state file does
    risk_log = logging.getLogger("aurum.risk.manager")
    level = risk_log.level
    risk_log.setLevel(logging.ERROR)   # display-only read: the state is printed below (re-read under the lock)
    try:
        before = StandardRiskManager(limits, cfg.instrument.build(), state_path=risk_path).snapshot()
    finally:
        risk_log.setLevel(level)
    _, read_err = _read_state_json(risk_path)
    print(f"risk state {risk_path}:")
    print(f"  halted:      {before['halted']}" + (f" [{before['halt_kind']}]" if before["halt_kind"] else ""))
    print(f"  reason:      {before['halt_reason'] or '-'}")
    print(f"  halted at:   {before['halted_at'] or '-'}")
    print(f"  equity:      last {_fmt(before['last_equity'])}, peak {_fmt(before['peak_equity'])}, "
          f"drawdown {_fmt(before['drawdown'])}")
    if read_err:
        print(f"  NOTE: the file is {read_err}; a reset rewrites it with a fresh state (peak/day start re-based "
              "at the next bar)")
    if not before["halted"]:
        print("the risk manager is not halted: nothing to reset")
        return EXIT_OK
    lock = _probe_runner_lock(state / _LOCK)
    if lock["held"]:
        raise CLIError(f"refusing to reset: a runner holds {state / _LOCK} (pid {lock.get('pid') or '?'}). A running "
                       "runner keeps its own risk state and rewrites the file every bar; stop it first", EXIT_USAGE)
    if args.confirm != "RESET":
        print("refusing to reset without the literal confirmation: re-run with --confirm RESET "
              "(and --reason '...' for the log)")
        return EXIT_CONFIRM

    try:
        runner = importlib.import_module("aurum.live.runner")
        lock_fn, unlock_fn = runner._lock_state_dir, runner._unlock_state_dir   # noqa: SLF001 - same lock
        locked_error = runner.RunnerLockedError
    except (ImportError, AttributeError) as exc:
        raise CLIError(f"aurum.live.runner is not available to take the state-dir lock ({exc})",
                       EXIT_UNAVAILABLE) from exc
    try:
        fh = lock_fn(state / _LOCK)
    except locked_error as exc:   # a runner started between the probe and now
        raise CLIError(f"refusing to reset: {exc}", EXIT_USAGE) from exc
    try:
        rm = StandardRiskManager(limits, cfg.instrument.build(), state_path=risk_path)   # re-read under the lock
        before = rm.snapshot()
        if not before["halted"]:
            print("the risk manager is no longer halted: nothing to reset")
            return EXIT_OK
        rm.reset_halt(confirm="RESET")
        after = rm.snapshot()
        try:
            user = getpass.getuser()
        except Exception:  # noqa: BLE001 - no login name (containers): still log the reset
            user = "unknown"
        now = pd.Timestamp.now(tz="UTC")
        reason = (args.reason or "").strip() or "no reason given"
        record = {"type": "halt_reset", "time": now, "user": user, "host": socket.gethostname(),
                  "pid": os.getpid(), "reason": reason, "config": cfg.source,
                  "previous": {k: before[k] for k in ("halt_kind", "halt_reason", "halted_at", "peak_equity",
                                                      "last_equity", "drawdown")},
                  "rebased_equity": after["peak_equity"]}
        from aurum.live.state import append_jsonl

        append_jsonl(state / _DECISIONS, record, fsync=True)
        append_jsonl(state / _ALERTS, {"level": "critical", "kind": "halt_reset", "time": now,
                                       "message": f"kill switch reset by {user}@{record['host']}: {reason} "
                                                  f"(was: {before['halt_reason']})",
                                       "data": record}, fsync=True)
    finally:
        unlock_fn(fh)
    print(f"kill switch RESET by {user}: {reason}")
    print(f"  peak / day-start equity re-based to {_fmt(after['peak_equity'])}; logged to {state / _DECISIONS} "
          f"and {state / _ALERTS}")
    print("  the runner resumes trading at its next start (still dry-run unless the config says otherwise)")
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------------
_EXECUTORS = ["auto", "process", "thread", "serial"]


def _add_config(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    p.add_argument("--config", "-c", required=required, metavar="PATH",
                   help="YAML config file (see configs/; `extends:` inheritance allowed)"
                        + ("" if required else "; default: the built-in defaults"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value by dotted path, YAML-parsed, e.g. --set walkforward.test=3M "
                        "(repeatable; secrets cannot be set)")


class _GuardedSubparsers:
    """What ``aurum.cli_ext.register_all`` receives: the real subparsers action, except that
    a core command name cannot be replaced and every parser it adds is remembered, so a
    failing extension can be rolled back without leaving half-registered commands."""

    def __init__(self, action: Any) -> None:
        self._action = action
        self._core = set(action.choices)
        self.added: list[str] = []

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        names = [name, *kwargs.get("aliases", ())]
        taken = sorted(set(names) & (self._core | set(self.added)))
        if taken:
            raise ValueError(f"extension command {taken} clashes with an existing aurum command")
        parser = self._action.add_parser(name, **kwargs)
        self.added.extend(names)
        return parser

    def rollback(self) -> None:
        for n in self.added:
            self._action._name_parser_map.pop(n, None)   # noqa: SLF001 - argparse has no removal API
        self._action._choices_actions[:] = [a for a in self._action._choices_actions  # noqa: SLF001
                                            if getattr(a, "dest", None) not in self.added]
        self.added.clear()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._action, item)


def _register_extensions(sub: Any) -> list[str]:
    """Add the subcommands of ``aurum.cli_ext.register_all(subparsers)`` if that module
    exists. Imported lazily and fully guarded: a missing module is silent, a broken one logs a
    WARNING and is rolled back, so the core commands always work. Returns the added names."""
    import importlib.util

    name = "aurum.cli_ext"
    if name in sys.modules:
        if sys.modules[name] is None:      # explicitly blocked
            return []
    else:
        try:
            if importlib.util.find_spec(name) is None:
                return []
        except (ImportError, ValueError):
            return []
    guard = _GuardedSubparsers(sub)
    try:
        mod = importlib.import_module(name)
        register = getattr(mod, "register_all", None)
        if not callable(register):
            logger.warning("aurum.cli_ext has no register_all(subparsers): extension commands not added")
            return []
        register(guard)
    except Exception as exc:  # noqa: BLE001 - an extension must never break the core CLI
        guard.rollback()
        logger.warning("aurum CLI extensions (aurum.cli_ext) failed to load and were skipped: %s: %s",
                       type(exc).__name__, exc)
        return []
    return list(guard.added)


def build_parser() -> argparse.ArgumentParser:
    from aurum import __version__

    ap = argparse.ArgumentParser(
        prog="aurum", description="Aurum v2: XAUUSD systematic research & trading.",
        epilog="Exit codes: 0 ok, 1 runtime error, 2 usage/configuration error, 3 optional component missing, "
               "4 confirmation required. Docs: docs/cli.md")
    ap.add_argument("--version", action="version", version=f"aurum {__version__}",
                    help="print the installed aurum version and exit")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v INFO, -vv DEBUG logging (-vv also shows full tracebacks)")
    ap.add_argument("--traceback", action="store_true", help="show full tracebacks on unexpected errors")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # data
    data = sub.add_parser("data", help="download / inspect market data")
    dsub = data.add_subparsers(dest="action", metavar="ACTION")
    dsub.required = True
    dl = dsub.add_parser("download", help="Dukascopy bars + Yahoo/FRED macro into the data store (network)")
    dl.add_argument("--out", default="data_store", metavar="DIR", help="data store directory (default: data_store)")
    dl.add_argument("--symbol", default="XAUUSD", help="Dukascopy symbol (default: XAUUSD)")
    dl.add_argument("--start", default="2012-01-01", metavar="DATE", help="first day of bars (default: 2012-01-01)")
    dl.add_argument("--end", default=None, metavar="DATE", help="inclusive end date (default: yesterday UTC)")
    dl.add_argument("--timeframes", default="all",
                    help="'all' (M15,H1,H4,D1,D1_nyclose with manifest) or a list like H1,H4")
    dl.add_argument("--cache", default="cache/dukascopy", metavar="DIR",
                    help="raw Dukascopy file cache (default: cache/dukascopy)")
    dl.add_argument("--workers", type=int, default=4, metavar="N", help="download concurrency (default: 4)")
    dl.add_argument("--time-budget", type=float, default=None, metavar="SECONDS",
                    help="time limit for the prefetch pass (re-run to resume)")
    dl.add_argument("--offline", action="store_true",
                    help="build the bars only from the local cache (macro is still fetched unless --no-macro)")
    dl.add_argument("--no-bars", action="store_true", help="skip the bar data")
    dl.add_argument("--no-macro", action="store_true", help="skip the Yahoo/FRED macro series")
    dl.add_argument("--macro-start", default="2011-01-01", metavar="DATE",
                    help="first day of macro data (default: 2011-01-01)")
    dl.add_argument("--macro-cache", default="cache/macro", metavar="DIR",
                    help="macro download cache (default: cache/macro)")
    dl.set_defaults(func=cmd_data_download)
    di = dsub.add_parser("info", help="summarise bar files and macro series")
    di.add_argument("--dir", default="data_store", metavar="DIR",
                    help="data store to inspect (default: data_store; ignored with --config)")
    _add_config(di, required=False)
    di.set_defaults(func=cmd_data_info)

    # config
    cp = sub.add_parser("config", help="show or validate a configuration (strict: all problems at once)")
    cp.add_argument("action", choices=["show", "validate"],
                    help="show: print the resolved YAML; validate: only check it")
    _add_config(cp, required=False)
    cp.set_defaults(func=cmd_config)

    # backtest
    bt = sub.add_parser("backtest", help="single-split backtest (fit before --start, test after)")
    _add_config(bt)
    bt.add_argument("--strategy", action="append", default=[], metavar="ID",
                    help="restrict to these strategy ids/names (repeatable)")
    bt.add_argument("--start", default=None, metavar="TS",
                    help="evaluation start (fit on earlier data); omit = IN-SAMPLE (default: backtest.start)")
    bt.add_argument("--end", default=None, metavar="TS",
                    help="evaluation end; a date-only value includes the whole day (default: backtest.end)")
    bt.add_argument("--out", default=None, metavar="DIR",
                    help="report directory (default: output.dir/backtest_<UTC>_<hash8>)")
    bt.add_argument("--jobs", type=int, default=None, metavar="N", help="parallel workers (0 = auto)")
    bt.add_argument("--include-holdout", action="store_true",
                    help="also evaluate bars after walkforward.holdout_start (spends the holdout)")
    bt.add_argument("--no-write", action="store_true", help="do not write the report directory")
    bt.add_argument("--no-tearsheet", action="store_true", help="do not write tearsheet.html")
    bt.set_defaults(func=cmd_backtest)

    # walkforward
    wf = sub.add_parser("walkforward", help="walk-forward research run -> report directory")
    _add_config(wf)
    wf.add_argument("--out", default=None, metavar="DIR",
                    help="run directory (default: output.dir/walkforward_<UTC>_<hash8>)")
    wf.add_argument("--jobs", type=int, default=None, metavar="N", help="parallel workers (0 = auto)")
    wf.add_argument("--executor", choices=_EXECUTORS, default=None,
                    help="worker pool (default: walkforward.executor); results do not depend on it")
    wf.add_argument("--no-write", action="store_true",
                    help="do not write the run directory (a holdout look is still recorded in the ledger)")
    wf.add_argument("--no-tearsheet", action="store_true", help="do not write tearsheet.html")
    wf.set_defaults(func=cmd_walkforward)

    # train-final
    tf = sub.add_parser("train-final", help="production artifact: fit on ALL data up to a cutoff, combiner "
                                            "weights from the stitched OOS walk-forward")
    _add_config(tf)
    tf.add_argument("--out", required=True, metavar="DIR", help="artifact directory (loadable by aurum.live.runner)")
    tf.add_argument("--cutoff", default=None, metavar="TS",
                    help="use bars closed at or before this UTC time (default: the last bar)")
    tf.add_argument("--from-run", default=None, metavar="WF_DIR",
                    help="walk-forward run dir whose OOS forecasts give the combiner weights "
                         "(default: run the walk-forward now)")
    tf.add_argument("--allow-config-mismatch", action="store_true",
                    help="accept a --from-run produced by a different config hash")
    tf.add_argument("--overwrite", action="store_true", help="replace an existing --out (checked before fitting)")
    tf.add_argument("--jobs", type=int, default=None, metavar="N", help="walk-forward workers (0 = auto)")
    tf.add_argument("--executor", choices=_EXECUTORS, default=None, help="walk-forward worker pool")
    tf.add_argument("--no-write", action="store_true", help="do not write the walk-forward run directory")
    tf.add_argument("--no-tearsheet", action="store_true", help="no tearsheet for the walk-forward run")
    tf.set_defaults(func=cmd_train_final)

    # report
    rp = sub.add_parser("report", help="print the summary of a run directory")
    rp.add_argument("--run", required=True, metavar="DIR", help="run directory (contains summary.json)")
    rp.add_argument("--json", action="store_true", help="print the raw summary.json")
    rp.set_defaults(func=cmd_report)

    # registries
    st = sub.add_parser("strategies", help="strategy registry")
    ssub = st.add_subparsers(dest="action", metavar="ACTION")
    ssub.required = True
    sl = ssub.add_parser("list", help="list registered strategies")
    sl.add_argument("--json", action="store_true", help="JSON with default params per strategy")
    sl.set_defaults(func=cmd_strategies_list)
    ft = sub.add_parser("features", help="feature registry")
    fsub = ft.add_subparsers(dest="action", metavar="ACTION")
    fsub.required = True
    fl = fsub.add_parser("list", help="list registered feature groups")
    fl.add_argument("--json", action="store_true", help="JSON output")
    fl.set_defaults(func=cmd_features_list)

    # desk
    dk = sub.add_parser("desk", help="LLM trading desk (Claude API)")
    ksub = dk.add_subparsers(dest="action", metavar="ACTION")
    ksub.required = True
    kd = ksub.add_parser("demo", help="offline demo cycle with a scripted fake client")
    kd.add_argument("--journal-dir", default=None, metavar="DIR", help="write the cycle's JSONL journal here")
    kd.set_defaults(func=cmd_desk_demo)
    kr = ksub.add_parser("run", help="ONE paid desk cycle on the latest bar (needs agents.enabled and "
                                     "ANTHROPIC_API_KEY)")
    _add_config(kr)
    kr.add_argument("--at", default=None, metavar="TS", help="decision time (UTC); default: latest bar close")
    kr.add_argument("--journal-dir", default=None, metavar="DIR", help="journal directory (default: agents.journal_dir)")
    kr.add_argument("--from-run", default=None, metavar="WF_DIR",
                    help="walk-forward run dir: combiner weights from its OOS forecasts (else equal weights)")
    kr.set_defaults(func=cmd_desk_run)
    kp = ksub.add_parser("replay", help="replay the desk over history through the same sizer/risk (PAID "
                                        "unless --fake)")
    _add_config(kp)
    kp.add_argument("--start", required=True, metavar="TS", help="first decision bar")
    kp.add_argument("--end", required=True, metavar="TS", help="last bar; a date-only value includes the whole day")
    kp.add_argument("--every", type=int, default=None, metavar="N",
                    help="desk cycle every N bars (default: agents.replay_every)")
    kp.add_argument("--max-cost", type=float, default=None, metavar="USD",
                    help="hard USD budget (default: agents.replay_max_cost_usd)")
    kp.add_argument("--yes", action="store_true", help="confirm the cost estimate and start")
    kp.add_argument("--fake", action="store_true", help="offline scripted client (no API calls, for testing)")
    kp.add_argument("--journal-dir", default=None, metavar="DIR", help="journal directory (default: agents.journal_dir)")
    kp.add_argument("--from-run", default=None, metavar="WF_DIR",
                    help="walk-forward run dir: combiner weights from its OOS forecasts")
    kp.add_argument("--out", default=None, metavar="DIR",
                    help="replay report directory (default: output.dir/desk_replay_<UTC>_<hash8>)")
    kp.set_defaults(func=cmd_desk_replay)

    # rl
    rl = sub.add_parser("rl", help="reinforcement learning")
    rsub = rl.add_subparsers(dest="action", metavar="ACTION")
    rsub.required = True
    rt = rsub.add_parser("train", help="train a PPO policy (config section 'rl'; env costs/sizing/risk from the "
                                       "top-level config)")
    _add_config(rt)
    rt.set_defaults(func=cmd_rl_train)

    # live
    lv = sub.add_parser("live", help="live / paper trading")
    lsub = lv.add_subparsers(dest="action", metavar="ACTION")
    lsub.required = True
    la = lsub.add_parser("artifact", help="fit strategies + combiner on the latest data -> trading artifact")
    _add_config(la)
    la.add_argument("--out", default=None, metavar="DIR", help="artifact directory (default: live.artifact_dir)")
    la.add_argument("--at", default=None, metavar="TS", help="fit on bars closed at or before this UTC time")
    la.add_argument("--from-run", default=None, metavar="WF_DIR",
                    help="walk-forward run dir: combiner weights from its OOS forecasts; OOS stats embedded")
    la.add_argument("--overwrite", action="store_true", help="replace an existing artifact (checked before fitting)")
    la.set_defaults(func=cmd_live_artifact)
    lr = lsub.add_parser("run", help="run the live loop (dry-run by default)")
    _add_config(lr)
    lr.add_argument("--i-understand-real-money", action="store_true",
                    help="required (with live.allow_live_real: true) to trade a real-money account")
    lr.add_argument("--max-cycles", type=int, default=None, metavar="N", help="stop after N bar decisions")
    lr.add_argument("--until", default=None, metavar="TS", help="stop at this UTC time (ISO)")
    lr.set_defaults(func=cmd_live_run)
    ls = lsub.add_parser("status", help="read-only summary of the runner / risk / OMS state in live.state_dir")
    _add_config(ls)
    ls.add_argument("--json", action="store_true", help="machine-readable JSON")
    ls.set_defaults(func=cmd_live_status)
    lh = lsub.add_parser("reset-halt", help="clear the risk kill switch (runner stopped; logged; needs "
                                            "--confirm RESET)")
    _add_config(lh)
    lh.add_argument("--confirm", default=None, metavar="RESET",
                    help="must be the literal word RESET; without it the halt state is shown and nothing changes")
    lh.add_argument("--reason", default=None, metavar="TEXT",
                    help="why the halt is being cleared (written to the state directory's logs)")
    lh.set_defaults(func=cmd_live_reset_halt)

    _register_extensions(sub)
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

    func = getattr(args, "func", None)
    if not callable(func):   # e.g. an extension subcommand registered without set_defaults(func=...)
        parser.print_usage(sys.stderr)
        print(f"aurum: command {args.command!r} has nothing to run", file=sys.stderr)
        return EXIT_USAGE
    try:
        return int(func(args))
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
