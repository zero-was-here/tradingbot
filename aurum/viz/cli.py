"""``aurum dashboard``: build the interactive HTML dashboard of a walk-forward run.

::

    aurum dashboard --run runs/protocol/trend_core_holdout [--journal runs/desk_journal]
                    [--out FILE] [--cdn] [--max-trades N] [--title TEXT]

Wiring: :func:`register` adds the sub-command to an ``argparse`` sub-parser group (the main
``aurum`` parser calls ``aurum.viz.cli.register(sub)``). The module also runs on its own:
``python -m aurum.viz.cli --run DIR``.

Exit codes follow :mod:`aurum.cli`: 0 success, 1 runtime error (e.g. unreadable run
directory), 2 usage error, 3 optional component unavailable (plotly not installed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3

__all__ = ["add_arguments", "cmd_dashboard", "main", "register"]

_DESCRIPTION = ("Build a self-contained interactive HTML dashboard (KPIs, equity, drawdown, rolling Sharpe, "
                "forecasts, combiner weights, costs, monthly returns, trades, holdout and optional LLM-desk "
                "timeline) from a walk-forward run directory.")


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments of ``aurum dashboard`` (shared by :func:`register` and :func:`main`)."""
    p.add_argument("--run", required=True, help="walk-forward run directory (contains summary.json)")
    p.add_argument("--journal", default=None, help="LLM desk journal directory (*.jsonl): adds the desk timeline")
    p.add_argument("--out", default=None, help="output HTML file (default: <run>/dashboard.html)")
    p.add_argument("--cdn", action="store_true",
                   help="reference plotly.js from its CDN instead of embedding it (~4.8 MB smaller; "
                        "viewing then needs internet)")
    p.add_argument("--max-trades", type=int, default=500, help="most recent trades listed (default 500)")
    p.add_argument("--title", default=None, help="page title")
    return p


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Build the dashboard; print the output path and size."""
    run = Path(args.run)
    if not (run / "summary.json").exists():
        print(f"aurum: {run / 'summary.json'} not found (is {run} a walkforward/backtest run directory?)",
              file=sys.stderr)
        return EXIT_ERROR
    if args.journal is not None and not Path(args.journal).exists():
        print(f"aurum: journal directory {args.journal} not found", file=sys.stderr)
        return EXIT_ERROR
    if args.max_trades < 0:
        print("aurum: --max-trades must be >= 0", file=sys.stderr)
        return EXIT_USAGE
    try:
        import plotly  # noqa: F401

        from aurum.viz.dashboard import build_dashboard
    except ImportError as exc:
        print(f"aurum: the dashboard needs plotly ({exc}); install it with: pip install 'aurum[viz]'",
              file=sys.stderr)
        return EXIT_UNAVAILABLE
    out = build_dashboard(run, args.out, journal_dir=args.journal,
                          include_plotlyjs="cdn" if args.cdn else True,
                          title=args.title, max_trades=args.max_trades)
    size = out.stat().st_size
    print(f"dashboard: {out} ({size / 1e6:.2f} MB{', plotly.js from CDN' if args.cdn else ''})")
    return EXIT_OK


def register(subparsers: Any) -> argparse.ArgumentParser:
    """Add ``dashboard`` to an ``argparse`` sub-parser group; returns the new parser.

    The parser's ``func`` default is :func:`cmd_dashboard`, matching how ``aurum.cli``
    dispatches (``args.func(args)``).
    """
    p = subparsers.add_parser("dashboard", help="interactive HTML dashboard of a walk-forward run directory",
                              description=_DESCRIPTION)
    add_arguments(p)
    p.set_defaults(func=cmd_dashboard)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point (``python -m aurum.viz.cli``)."""
    ap = argparse.ArgumentParser(prog="aurum dashboard", description=_DESCRIPTION)
    add_arguments(ap)
    try:
        args = ap.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0) if isinstance(exc.code, int) else EXIT_USAGE
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return cmd_dashboard(args)
    except Exception as exc:  # noqa: BLE001 - top-level: report and exit non-zero
        print(f"aurum: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
