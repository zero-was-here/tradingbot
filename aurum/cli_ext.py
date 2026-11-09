"""Research-analytics commands for the ``aurum`` CLI: ``sweep``, ``stress`` and ``explain``.

``register_all(subparsers)`` adds them to the ``aurum`` parser (``aurum.cli.build_parser``
calls it); it also registers ``aurum dashboard`` from :mod:`aurum.viz.cli` when that module
exists. Registration is idempotent (commands already present are skipped).

::

    aurum sweep   -c CONFIG --strategy NAME --grid 'param=v1,v2,...' [--grid ...] [--metric sharpe]
                  [--start DATE] [--end DATE] [--out DIR] [--jobs N] [--executor ...] [--single-split]
                  [--no-ledger] [--set KEY=VALUE ...]
    aurum stress  --run DIR [--book combined] [--phase research|holdout] [--paths N] [--horizon 2Y]
                  [--seed S] [--block B] [--shock PCT ...] [--no-data] [--out DIR]
    aurum explain --run DIR [--at TIMESTAMP] [--phase research|holdout] [--attribution]
                  [--desk-journal DIR] [--no-data] [--out DIR]
    aurum explain --live STATE_DIR [--at TIMESTAMP] [--out DIR]

Every command prints a concise table and writes PNG/CSV/JSON files. ``stress`` and
``explain`` only READ a run directory (default output: ``<run>/analysis/<command>``);
``explain --live`` reads a live/paper runner's ``decisions.jsonl`` (the per-decision
``attribution`` the runner logs) and writes to ``runs/analysis/explain_live``, never into the
runner's state directory; a
``sweep`` never evaluates bars from ``walkforward.holdout_start`` on, nor any holdout window
already recorded in a holdout ledger (see :mod:`aurum.research.sweep`).

``python -m aurum.cli_ext ...`` runs the full ``aurum`` CLI with these commands registered.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["COMMANDS", "main", "register_all"]

COMMANDS = ("sweep", "stress", "explain")


# ----------------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------------
def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        x = float(v)
        if not math.isfinite(x):
            return "nan"
        if pct:
            return f"{x:+.1%}"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        if float(x).is_integer() and abs(x) < 1e6 and isinstance(v, int):
            return str(int(x))
        return f"{x:.3f}"
    return str(v)


_INT_COLS = frozenset({"n_trades", "n_days", "n_folds", "days", "n_neighbours", "longest_underwater_days"})


def _cell(v: Any, col: str, pct: Sequence[str]) -> str:
    if col in _INT_COLS and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) \
            and float(v).is_integer():
        return f"{int(v):,}"
    return _fmt(v, col in pct)


def _table(rows: Sequence[dict[str, Any]], cols: Sequence[str], *, pct: Sequence[str] = ()) -> str:
    if not rows:
        return "(empty)"
    cells = [[_cell(r.get(c), c, pct) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    head = "  ".join(c.rjust(w) for c, w in zip(cols, widths, strict=True))
    lines = [head, "-" * len(head)]
    lines += ["  ".join(x.rjust(w) for x, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(lines)


def _records(df: Any) -> list[dict[str, Any]]:
    import pandas as pd

    out = []
    for r in df.to_dict(orient="records"):
        out.append({k: (None if (isinstance(v, float) and not math.isfinite(v)) or v is pd.NaT else v)
                    for k, v in r.items()})
    return out


def _cli_error(msg: str, code: int | None = None) -> Exception:
    from aurum.cli import EXIT_USAGE, CLIError

    return CLIError(msg, EXIT_USAGE if code is None else code)


# ----------------------------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------------------------
def cmd_sweep(args: argparse.Namespace) -> int:
    from aurum.cli import EXIT_OK, _load_cfg
    from aurum.research.sweep import parse_grid, run_sweep

    if not args.grid:
        raise _cli_error("sweep needs at least one --grid param=v1,v2,...")
    cfg = _load_cfg(args)
    grid = parse_grid(args.grid)
    t0 = time.perf_counter()
    md = cfg.data.load()
    print(f"loaded {len(md.bars)} {cfg.data.timeframe} bars {md.bars.index[0]} -> {md.bars.index[-1]} "
          f"({time.perf_counter() - t0:.1f}s)")
    n = math.prod(len(v) for v in grid.values())
    print(f"== sweep (EXPLORATORY: every grid point is a trial): {args.strategy}  {n} point(s)  "
          f"metric {args.metric}  {'walk-forward OOS' if not args.single_split else 'single split'}")
    res = run_sweep(md, cfg, args.strategy, grid, metric=args.metric, walkforward=not args.single_split,
                    start=args.start, end=args.end, n_jobs=args.jobs or 0, executor=args.executor,
                    out_dir=args.out, write=True, ledger=False if args.no_ledger else None)
    w = res.window
    print(f"research window {w['start']} -> {w['end']} ({w['n_bars']} bars); OOS {w.get('oos_start')} -> "
          f"{w.get('oos_end')}")
    print()
    t = res.table.sort_values(res.metric, ascending=not res.maximize, na_position="last")
    cols = [*res.grid, "status", "sharpe", "cagr", "max_drawdown", "n_trades", "turnover_lots_per_year",
            "total_costs", "seconds"]
    if res.metric not in cols:
        cols.insert(len(res.grid) + 1, res.metric)
    print(_table(_records(t), cols, pct=("cagr", "max_drawdown")))
    b, p = res.best, res.plateau
    print()
    if b.get("label"):
        print(f"best: {b['label']}  {res.metric}={_fmt(b.get('value'))}  Sharpe {_fmt(b.get('sharpe'))} "
              f"[{_fmt(b.get('ci_lower'))}, {_fmt(b.get('ci_upper'))}]  PSR {_fmt(b.get('psr'))}  "
              f"DSR {_fmt(b.get('dsr'))} (n_trials {res.n_trials})  DSR_xs {_fmt(b.get('dsr_xs'))}")
        print(f"plateau: {p.get('verdict')}  ratio {_fmt(p.get('plateau_ratio'))} over {p.get('n_neighbours')} "
              f"neighbour(s) (mean {_fmt(p.get('neighbour_mean'))}, min {_fmt(p.get('neighbour_min'))});  "
              f"plateau centre {p.get('robust_best')} ({res.metric} {_fmt(p.get('robust_best_value'))})")
    if res.pbo:
        print(f"PBO across grid points: {_fmt(res.pbo.get('pbo'))}  "
              f"(P[OOS loss of IS-best] {_fmt(res.pbo.get('prob_oos_loss'))})")
    for n_ in res.notes:
        print(f"note: {n_}")
    bad = res.table.loc[res.table["status"] != "ok", ["label", "status", "error"]]
    for _, r in bad.iterrows():
        print(f"skipped {r['label']}: {r['status']} ({r['error']})")
    print(f"\ntiming: evaluate {res.timing.get('evaluate_s')}s, total {res.timing.get('total_s')}s "
          f"({res.settings['executor']} x{res.settings['n_jobs']})")
    print(f"outputs: {res.out_dir}/sweep.csv, sweep.json, sweep.png")
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# stress
# ----------------------------------------------------------------------------------------
def cmd_stress(args: argparse.Namespace) -> int:
    from aurum.cli import EXIT_OK
    from aurum.research.montecarlo import stress_run

    run = Path(args.run)
    if not run.is_dir():
        raise FileNotFoundError(f"run directory {run} not found")
    shocks = [float(s) for s in args.shock] if args.shock else None
    rep = stress_run(run, book=args.book, phase=args.phase, n_paths=args.paths, horizon=args.horizon,
                     seed=args.seed, mean_block=args.block, shocks=shocks, load_data=not args.no_data)
    out = Path(args.out) if args.out else run / "analysis" / "stress"
    rep.save(out)
    mc = rep.mc
    print(f"== stress: {rep.source}")
    print(f"history: {mc.n_history} daily returns, Sharpe {_fmt(mc.history_sharpe)};  Monte Carlo: {mc.n_paths:,} "
          f"paths x {mc.horizon} days, stationary bootstrap mean block {mc.mean_block:.1f} days, seed {mc.seed} "
          f"({mc.seconds:.2f}s)")
    print()
    dist = mc.distribution()
    rows = [{"stat": idx, **{k: float(v) for k, v in r.items()}} for idx, r in dist.iterrows()]
    print(_table(rows, ["stat", "cagr", "max_drawdown", "total_return", "final_equity", "longest_underwater_days"],
                 pct=("cagr", "max_drawdown", "total_return")))
    print()
    dd = mc.dd_table.assign(threshold=lambda d: d["threshold"].map(lambda x: f">{x:.0%}"))
    ru = mc.ruin.assign(loss_threshold=lambda d: d["loss_threshold"].map(lambda x: f"-{x:.0%}"))
    print("P(max drawdown deeper than x)          risk of ruin (equity touches start - x)")
    left = _table(_records(dd), ["threshold", "probability"]).splitlines()
    right = _table(_records(ru), ["loss_threshold", "p_touch", "p_end_below"]).splitlines()
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        print(f"{a:<38} {b}")
    rec = mc.recovery()
    def days(x: float) -> str:
        return f"{x:,.0f}" if isinstance(x, float) and math.isfinite(x) else "-"

    print(f"\nrecovery: P(not recovered from the max drawdown within the horizon) {rec['p_not_recovered']:.1%}; "
          f"time to recover median {days(rec['ttr_median_days'])} / p95 {days(rec['ttr_p95_days'])} trading days; "
          f"longest underwater median {days(rec['longest_underwater_median_days'])} days")
    print("\nworst historical k-day windows of the book (replayed on current equity):")
    print(_table(_records(rep.worst_windows), ["days", "worst_return", "start", "end", "pnl_usd_at_current_equity"],
                 pct=("worst_return",)))
    ex = rep.exposure
    print(f"\nexposure at {ex.get('time')}: {ex.get('lots', 0):+.2f} lots, price {_fmt(ex.get('price'))} "
          f"({ex.get('price_source')}), leverage {_fmt(ex.get('leverage'))}")
    if len(rep.gap_shocks):
        print(_table(_records(rep.gap_shocks), ["exposure", "scenario", "shock_pct", "pnl_usd", "pnl_pct_equity"],
                     pct=("shock_pct", "pnl_pct_equity")))
    if rep.gold_moves is not None and len(rep.gold_moves):
        print("\nworst historical gold moves applied to the current position:")
        print(_table(_records(rep.gold_moves), ["days", "worst_drop", "drop_end", "worst_rise", "rise_end",
                                                "pnl_pct_equity_worst"],
                     pct=("worst_drop", "worst_rise", "pnl_pct_equity_worst")))
    for n_ in rep.notes:
        print(f"note: {n_}")
    print(f"\noutputs: {out}/stress.json, fan.png, distributions.png, *.csv")
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# explain
# ----------------------------------------------------------------------------------------
def _explain_live(args: argparse.Namespace, out: Path) -> list[str]:
    """``explain --live``: one live/paper decision from the runner's ``decisions.jsonl`` plus a
    per-strategy summary over the whole log (read-only)."""
    from aurum.research.attribution import explain_live_decision, live_decisions

    log = live_decisions(args.live)
    ex = explain_live_decision(args.live, args.at)
    tag = f"{ex.time:%Y%m%dT%H%M}"
    print(f"== explain (live log): {args.live} [{ex.fold}] decision at the close of {ex.time}"
          + (f", decided {ex.decision_time}" if ex.decision_time is not None else ""))
    print()
    print(_table(_records(ex.table), ["strategy", "forecast", "weight", "fdm", "contribution", "share"],
                 pct=("share",)))
    line = (f"\nraw sum {ex.raw_combined:+.4f}  clip {ex.clip_adjustment:+.4f}  -> combined {ex.combined:+.4f}")
    if ex.final_forecast is not None:
        act = (ex.desk or {}).get("action") or (ex.desk or {}).get("status") or "off"
        line += f"  -> desk {act} {ex.desk_delta:+.4f} -> final {ex.final_forecast:+.4f}"
    print(line)
    s = ex.sizing
    if math.isfinite(float(s.get("vol_ann", math.nan) or math.nan)):
        print(f"sizing: vol {s['vol_ann']:.1%}, price {_fmt(s.get('price'))}, equity {_fmt(s.get('equity'))}, "
              f"drawdown {_fmt(s.get('drawdown'), pct=True)} -> notional {_fmt(s.get('raw_notional'))} -> caps "
              f"{s.get('caps_hit') or 'none'}, drawdown x{_fmt(s.get('drawdown_mult'))} -> {_fmt(s.get('raw_lots'))} "
              f"lots -> target {_fmt(s.get('replayed_lots'))}"
              + (" (rebalance band kept the current position)" if s.get("kept_current_band") else ""))
    if "current_lots" in s:
        print(f"position {_fmt(s['current_lots'])} -> target {_fmt(s.get('saved_requested_lots'))} -> risk-approved "
              f"{_fmt(s.get('approved_lots'))} lots" + (f" (orders: {s['orders_status']})"
                                                         if s.get("orders_status") else ""))
    for iv in ex.risk.get("interventions", []):
        print(f"risk manager: requested {iv.get('requested')} approved {iv.get('approved')} "
              f"halted {iv.get('halted')}: {iv.get('reasons')}")
    for n_ in ex.notes:
        print(f"note: {n_}")
    contrib = log[[c for c in log.columns if c.startswith("c:")]]
    n_att = int(contrib.notna().any(axis=1).sum())
    if n_att:
        raw_abs = contrib.abs().sum(axis=1).replace(0.0, math.nan)
        rows = [{"strategy": c[2:], "mean_contribution": float(contrib[c].mean()),
                 "mean_abs_contribution": float(contrib[c].abs().mean()),
                 "share_of_abs": float((contrib[c].abs() / raw_abs).mean())}
                for c in contrib.abs().mean().sort_values(ascending=False).index]
        print(f"\n== live log summary: {len(log)} decisions, {n_att} with an attribution "
              f"({log['time'].min()} -> {log['time'].max()})")
        print(_table(rows, ["strategy", "mean_contribution", "mean_abs_contribution", "share_of_abs"],
                     pct=("share_of_abs",)))
        if log["desk_delta"].notna().any():
            moved = log["desk_delta"].fillna(0.0).abs() > 1e-12
            print(f"LLM desk changed the forecast on {int(moved.sum())} of {len(log)} decisions "
                  f"(mean |delta| {log.loc[moved, 'desk_delta'].abs().mean() if moved.any() else 0.0:.3f})")
    (out / f"decision_live_{tag}.json").write_text(json.dumps(ex.to_dict(), indent=2), encoding="utf-8")
    ex.table.to_csv(out / f"decision_live_{tag}.csv", index=False)
    fig = ex.plot()
    fig.savefig(out / f"decision_live_{tag}.png", dpi=110, facecolor=fig.get_facecolor())
    log.to_csv(out / "live_decisions.csv", index=False)
    return [f"decision_live_{tag}.json", f"decision_live_{tag}.csv", f"decision_live_{tag}.png",
            "live_decisions.csv"]


def cmd_explain(args: argparse.Namespace) -> int:
    from aurum.cli import EXIT_OK
    from aurum.research.attribution import desk_journal_summary, explain_decision, load_run, pnl_attribution

    live = getattr(args, "live", None)
    if args.out:
        out = Path(args.out)
    elif args.run:
        out = Path(args.run) / "analysis" / "explain"
    elif live:   # never write into the runner's state directory
        out = Path("runs") / "analysis" / "explain_live"
    else:
        out = Path("runs") / "analysis" / "explain"
    if not args.run and not args.desk_journal and not live:
        raise _cli_error("explain needs --run DIR, --live STATE_DIR and/or --desk-journal DIR")
    if args.run and live:
        raise _cli_error("explain: give --run (a research run) or --live (a runner's decision log), not both")
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if live:
        written += _explain_live(args, out)
    if args.run:
        run = load_run(args.run, phase=args.phase)
        ex = explain_decision(run, args.at, load_data=not args.no_data)
        tag = f"{ex.time:%Y%m%dT%H%M}"
        print(f"== explain: {args.run} [{ex.phase}] decision at the close of {ex.time} (fold {ex.fold})"
              + (f", decided {ex.decision_time}" if ex.decision_time is not None else ""))
        print()
        print(_table(_records(ex.table), ["strategy", "forecast", "weight", "fdm", "contribution", "share"],
                     pct=("share",)))
        print(f"\nraw sum {ex.raw_combined:+.4f}  clip {ex.clip_adjustment:+.4f}  -> combined {ex.combined:+.4f} "
              f"(saved {ex.saved_combined:+.4f})")
        s = ex.sizing
        if "vol_ann" in s:
            print(f"sizing: vol {s['vol_ann']:.1%}, price {_fmt(s.get('price'))}, equity {_fmt(s.get('equity'))}, "
                  f"drawdown {s.get('drawdown', 0):.1%} -> notional {_fmt(s.get('raw_notional'))} -> caps "
                  f"{s.get('caps_hit') or 'none'}, drawdown x{_fmt(s.get('drawdown_mult'))} -> "
                  f"{_fmt(s.get('raw_lots'))} lots -> requested {_fmt(s.get('replayed_lots'))}"
                  + (" (rebalance band kept the current position)" if s.get("kept_current_band") else "")
                  + (f"; matches saved: {s.get('replay_matches_saved')}" if "replay_matches_saved" in s else ""))
        if "current_lots" in s:
            print(f"position {s['current_lots']:+.2f} -> requested {_fmt(s.get('saved_requested_lots'))} -> filled "
                  f"at the next open {_fmt(s.get('filled_lots_next_bar'))} lots")
        for iv in ex.risk.get("interventions", []):
            print(f"risk manager: requested {iv.get('requested')} approved {iv.get('approved')} "
                  f"halted {iv.get('halted')}: {iv.get('reasons')}")
        for n_ in ex.notes:
            print(f"note: {n_}")
        (out / f"decision_{tag}.json").write_text(json.dumps(ex.to_dict(), indent=2), encoding="utf-8")
        ex.table.to_csv(out / f"decision_{tag}.csv", index=False)
        fig = ex.plot()
        fig.savefig(out / f"decision_{tag}.png", dpi=110, facecolor=fig.get_facecolor())
        written += [f"decision_{tag}.json", f"decision_{tag}.csv", f"decision_{tag}.png"]
        if args.attribution:
            att = pnl_attribution(run)
            print(f"\n== PnL attribution ({att.phase}, {att.component}); method: {att.method}")
            rows = [{"strategy": k, "net": float(att.by_component.loc[k, "net"]),
                     "price": float(att.by_component.loc[k, "price"]),
                     "costs": float(att.by_component.loc[k, "costs"]),
                     "swap": float(att.by_component.loc[k, "swap"]),
                     "share_of_total": float(att.totals[k] / att.total_pnl) if att.total_pnl else math.nan}
                    for k in att.totals.sort_values().index]
            print(_table(rows, ["strategy", "net", "price", "costs", "swap", "share_of_total"],
                         pct=("share_of_total",)))
            print(f"total {att.total_pnl:,.2f} USD = sum of the rows ({att.totals.sum():,.2f})")
            att.by_component.to_csv(out / "attribution_totals.csv")
            att.by_year.to_csv(out / "attribution_by_year.csv")
            att.per_bar.resample("1D").sum().loc[lambda d: d.abs().sum(axis=1) > 0].to_csv(
                out / "attribution_daily.csv")
            (out / "attribution.json").write_text(json.dumps(att.to_dict(), indent=2), encoding="utf-8")
            fig = att.plot()
            fig.savefig(out / "attribution.png", dpi=110, facecolor=fig.get_facecolor())
            written += ["attribution.png", "attribution_totals.csv", "attribution_by_year.csv",
                        "attribution_daily.csv", "attribution.json"]
    if args.desk_journal:
        ds = desk_journal_summary(args.desk_journal)
        t = ds.totals
        print(f"\n== desk journal: {args.desk_journal}: {t['n_cycles']} cycle(s) in {t['n_files']} file(s)")
        for k in ("status", "actions", "consults_by_role", "created_agents", "tool_calls", "agent_status"):
            print(f"  {k}: {t[k]}")
        print(f"  consults {t['n_consults']}, created agents {t['n_created_agents']}, tool calls {t['n_tool_calls']}, "
              f"constrained {t['n_constrained']}, fallbacks {t['n_fallback']}, cost ${_fmt(t['total_cost_usd'])}, "
              f"median final/quant {_fmt(t['median_final_over_quant'])}, direction flips {t['n_direction_flips']}")
        (out / "desk_summary.json").write_text(json.dumps(ds.to_dict(), indent=2), encoding="utf-8")
        ds.cycles.to_csv(out / "desk_cycles.csv", index=False)
        written += ["desk_summary.json", "desk_cycles.csv"]
    print(f"\noutputs: {out}/" + ", ".join(written))
    return EXIT_OK


# ----------------------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------------------
def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="YAML config file (see configs/)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set data.start=2016-01-01 (repeatable)")


def _add_sweep(sub: Any) -> None:
    sp = sub.add_parser("sweep", help="parameter-robustness sweep: walk-forward OOS per grid point, DSR, plateau "
                                      "score, heatmap (never touches the holdout)")
    _add_config(sp)
    sp.add_argument("--strategy", required=True, help="config strategy id/name or registry name")
    sp.add_argument("--grid", action="append", default=[], metavar="PARAM=V1,V2,...",
                    help="grid values of one parameter (repeatable; 'a:b:step' ranges allowed)")
    sp.add_argument("--metric", default="sharpe", help="ranking metric (sharpe, cagr, calmar, max_drawdown, ...)")
    sp.add_argument("--start", default=None, help="research window start (inclusive)")
    sp.add_argument("--end", default=None, help="research window end (inclusive; date = whole day)")
    sp.add_argument("--out", default=None, help="output directory (default output.dir/sweep_<strategy>_<utc>)")
    sp.add_argument("--jobs", type=int, default=None, help="parallel grid points (0 = auto)")
    sp.add_argument("--executor", choices=["auto", "process", "thread", "serial"], default="auto",
                    help="how grid points run in parallel (results are identical across executors)")
    sp.add_argument("--single-split", action="store_true",
                    help="one train/test split per point instead of the walk-forward (in-sample without --start)")
    sp.add_argument("--no-ledger", action="store_true", help="do not read/append the sweep ledger")
    sp.set_defaults(func=cmd_sweep)


def _add_stress(sub: Any) -> None:
    sp = sub.add_parser("stress", help="Monte Carlo (block bootstrap) + historical stress of a run's book")
    sp.add_argument("--run", required=True, help="walk-forward/backtest run directory")
    sp.add_argument("--book", default="combined", help="book under books/ (default combined)")
    sp.add_argument("--phase", choices=["research", "holdout"], default="research",
                    help="stitched walk-forward OOS book (research) or the already-evaluated holdout")
    sp.add_argument("--paths", type=int, default=10_000, help="Monte Carlo paths (default 10000)")
    sp.add_argument("--horizon", default="2Y", help="trading days or 2Y / 6M / 4W / 30D (default 2Y)")
    sp.add_argument("--seed", type=int, default=0, help="random seed of the bootstrap (default 0)")
    sp.add_argument("--block", type=float, default=None, help="mean block length in days (default automatic)")
    sp.add_argument("--shock", action="append", default=[], metavar="PCT",
                    help="gap shock as a fraction, e.g. --shock=-0.05 (repeatable; default scenario set)")
    sp.add_argument("--no-data", action="store_true", help="do not load the run's bars (price from the last fill)")
    sp.add_argument("--out", default=None, help="output directory (default <run>/analysis/stress)")
    sp.set_defaults(func=cmd_stress)


def _add_explain(sub: Any) -> None:
    sp = sub.add_parser("explain", help="explain a decision (forecast waterfall + sizing) and attribute PnL "
                                        "to strategies")
    sp.add_argument("--run", default=None, help="walk-forward/backtest run directory")
    sp.add_argument("--live", default=None, metavar="STATE_DIR",
                    help="explain a live/paper decision from the runner's decision log (live.state_dir or its "
                         "decisions.jsonl; read-only) instead of a research run")
    sp.add_argument("--at", default=None,
                    help="bar open time (UTC) to explain; default the last OOS bar (or the last live decision)")
    sp.add_argument("--phase", choices=["research", "holdout"], default="research",
                    help="explain the stitched walk-forward OOS (research) or the already-evaluated holdout")
    sp.add_argument("--attribution", action="store_true", help="also attribute the combined book's PnL")
    sp.add_argument("--desk-journal", default=None, help="desk journal directory/file to summarise")
    sp.add_argument("--no-data", action="store_true", help="do not load the run's bars (no sizing replay)")
    sp.add_argument("--out", default=None,
                    help="output directory (default <run>/analysis/explain; runs/analysis/explain_live for --live)")
    sp.set_defaults(func=cmd_explain)


def register_all(subparsers: Any) -> list[str]:
    """Add ``sweep``, ``stress``, ``explain`` (and ``dashboard`` from :mod:`aurum.viz.cli` when
    installed) to an argparse sub-parsers action. Idempotent; returns the names added."""
    def names() -> set[str]:
        return set(getattr(subparsers, "choices", None) or {})

    added: list[str] = []
    for name, fn in (("sweep", _add_sweep), ("stress", _add_stress), ("explain", _add_explain)):
        if name in names():
            continue
        fn(subparsers)
        added.append(name)
    try:
        from aurum.viz.cli import register as viz_register
    except ImportError:
        viz_register = None
    if viz_register is not None and "dashboard" not in names():
        before = names()
        viz_register(subparsers)
        added += sorted(names() - before)
    return added


def _subparsers(parser: argparse.ArgumentParser) -> Any:
    for a in parser._actions:  # noqa: SLF001 - argparse has no public accessor
        if isinstance(a, argparse._SubParsersAction):  # noqa: SLF001
            return a
    raise RuntimeError("parser has no sub-commands")


def build_parser() -> argparse.ArgumentParser:
    """The ``aurum`` parser with these commands registered (idempotent)."""
    from aurum.cli import build_parser as base

    parser = base()
    register_all(_subparsers(parser))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m aurum.cli_ext``: the ``aurum`` CLI with the analytics commands registered,
    with the same logging, error handling and exit codes as :func:`aurum.cli.main`."""
    from aurum.cli import EXIT_ERROR, EXIT_USAGE, CLIError, _setup_logging
    from aurum.core.config import ConfigError

    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0) if isinstance(exc.code, int) else EXIT_USAGE
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"aurum: configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except CLIError as exc:
        print(f"aurum: {exc}", file=sys.stderr)
        return exc.code
    except (FileNotFoundError, ValueError) as exc:
        print(f"aurum: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("aurum: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if args.traceback or args.verbose >= 2:
            raise
        print(f"aurum: error: {type(exc).__name__}: {exc} (use --traceback for details)", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
