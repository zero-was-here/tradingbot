"""aurum/viz/dashboard.py + aurum/viz/cli.py: build the interactive dashboard from a real
walk-forward run directory made on synthetic data (with and without a holdout / desk journal)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from aurum.core.config import load_config  # noqa: E402
from aurum.core.types import MarketData  # noqa: E402
from aurum.data.synthetic import (  # noqa: E402
    make_synthetic_bars,
    make_synthetic_events,
    make_synthetic_macro,
)
from aurum.research.walkforward import run_walk_forward  # noqa: E402
from aurum.viz import cli as viz_cli  # noqa: E402
from aurum.viz.dashboard import (  # noqa: E402
    _Figs,
    _monthly_returns,
    _strategy_slots,
    build_dashboard,
    fig_weights,
    load_desk_journal,
    load_run,
)

SECTIONS = ("kpis", "performance", "strategies", "costs", "monthly", "trades", "provenance")


def _config(holdout: str | None) -> object:
    return load_config({
        "name": "viz_test",
        "strategies": [
            {"name": "tsmom", "params": {"horizons": [24, 96, 240], "vol_halflife": 48, "vol_min_periods": 24}},
            {"name": "ema_cross", "params": {"fast": 8, "slow": 32, "vol_halflife": 48, "vol_min_periods": 24}},
            {"name": "donchian", "params": {"entry_n": 48, "exit_n": 24, "atr_n": 24}},
        ],
        "walkforward": {"train": "4M", "test": "6W", "embargo": 6, "executor": "serial", "n_boot": 200,
                        "pbo_splits": 8, "min_train_bars": 200, "holdout_start": holdout, "n_trials": 5,
                        "combiner_min_obs": 200},
        "risk": {"research": {"max_spread": None}},
        "output": {"save_results": True, "tearsheet": False},
    }, env={})


@pytest.fixture(scope="module")
def market() -> MarketData:
    bars = make_synthetic_bars(9000, "H1", seed=4, model="trend", regime_params={"phi": 0.12})
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=4),
                      events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=14)))


@pytest.fixture(scope="module")
def run_dir(market: MarketData, tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("wf") / "run"
    run_walk_forward(market, _config(None), out_dir=d, write=True, holdout_ledger=False)
    return d


@pytest.fixture(scope="module")
def holdout_run_dir(market: MarketData, tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("wfh") / "run"
    run_walk_forward(market, _config(str(market.bars.index[-1500])), out_dir=d, write=True, holdout_ledger=False)
    return d


@pytest.fixture(scope="module")
def journal_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from aurum.agents import demo

    d = tmp_path_factory.mktemp("journal")
    demo(journal_dir=d)                         # offline scripted client: no network, no key
    return d


def _plot_count(text: str) -> int:
    return len(re.findall(r'id="aurum-plot-\d+"', text))


# ---------------------------------------------------------------------------------------
def test_load_run_reads_the_run_directory(run_dir: Path) -> None:
    run = load_run(run_dir)
    assert run.name == "viz_test"
    assert run.strategies == ["tsmom", "ema_cross", "donchian"]
    assert {"combined", "benchmark", "tsmom"} <= set(run.books)
    assert run.trades is not None and run.forecasts is not None and run.weights is not None
    assert run.costs is not None and "combined" in run.costs.index
    with pytest.raises(FileNotFoundError):
        load_run(run_dir.parent / "nope")


def test_dashboard_has_every_section_and_no_failed_panel(run_dir: Path, tmp_path: Path) -> None:
    out = build_dashboard(run_dir, tmp_path / "d" / "dash.html", include_plotlyjs="cdn")
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    for sid in SECTIONS:
        assert f'id="{sid}"' in html, sid
    assert 'id="desk"' not in html and 'id="holdout"' not in html     # neither given nor present
    assert "unavailable" not in html                                    # no panel failed
    for label in ("Sharpe (daily, annualised)", "Deflated Sharpe (DSR)", "Probabilistic Sharpe (PSR)",
                  "Max drawdown", "CAGR", "Equity vs buy-and-hold (log scale)", "Drawdown from peak",
                  "Rolling Sharpe", "Combiner weights per fold", "Gross to net", "Trading costs by book",
                  "Monthly returns, combined book", "Per-book statistics", "Table view"):
        assert label in html, label
    # equity, drawdown, rolling Sharpe, fold Sharpe, forecasts, weights, waterfall, costs, monthly
    assert _plot_count(html) == 9
    assert html.count('class="key"') >= 2                              # HTML legends with toggles
    assert 'id="theme-toggle"' in html and "prefers-color-scheme" in html
    palettes = json.loads(re.search(r"var P = (\{.*?\});", html, flags=re.S).group(1))
    assert palettes["light"]["surface"] == "#fcfcfb" and palettes["dark"]["surface"] == "#1a1a19"
    assert 'src="https://cdn.plot.ly' in html and out.stat().st_size < 1_500_000
    assert 'id="trades-table"' in html and "hovertemplate" in html
    assert '<meta name="viewport"' in html


def test_dashboard_default_output_and_inline_plotly(run_dir: Path) -> None:
    out = build_dashboard(run_dir)
    try:
        assert out == run_dir / "dashboard.html"
        assert out.stat().st_size > 3_000_000                          # plotly.js embedded: works offline
        assert 'src="https://cdn.plot.ly' not in out.read_text(encoding="utf-8")
    finally:
        out.unlink()


def test_dashboard_with_desk_journal(run_dir: Path, journal_dir: Path, tmp_path: Path) -> None:
    cycles = load_desk_journal(journal_dir)
    assert len(cycles) == 1
    row = cycles.iloc[0]
    assert row["status"] == "decided" and row["consulted"] and row["created"]
    assert np.isfinite(row["final_forecast"]) and np.isfinite(row["quant_forecast"])
    html = build_dashboard(run_dir, tmp_path / "desk.html", journal_dir=journal_dir,
                           include_plotlyjs="cdn").read_text(encoding="utf-8")
    assert 'id="desk"' in html and "Specialists per cycle" in html and "Cycle log" in html
    for agent in row["consulted"] + row["created"]:
        assert agent in html
    assert _plot_count(html) == 11


def test_desk_journal_parsing_is_robust(tmp_path: Path) -> None:
    assert load_desk_journal(tmp_path / "missing").empty
    j = tmp_path / "j"
    j.mkdir()
    (j / "bad.jsonl").write_text('not json\n{"event": "cycle_start", "cycle_id": "c1", "now": '
                                 '"2024-01-02T10:00:00+00:00", "quant_forecast": 0.4}\n{"oops": 1}\n'
                                 '{"event": "cycle_end", "cycle_id": "c1", "status": "failed", '
                                 '"final_forecast": 0.4}\n', encoding="utf-8")
    df = load_desk_journal(j)
    assert len(df) == 1 and df.loc[0, "status"] == "failed" and df.loc[0, "consulted"] == []


def test_holdout_section(holdout_run_dir: Path, tmp_path: Path) -> None:
    html = build_dashboard(holdout_run_dir, tmp_path / "h.html", include_plotlyjs="cdn").read_text(encoding="utf-8")
    assert 'id="holdout"' in html and "Holdout equity vs buy-and-hold" in html and "evaluated <b>once</b>" in html
    assert "unavailable" not in html


def test_untrusted_text_is_escaped(run_dir: Path, tmp_path: Path) -> None:
    d = tmp_path / "evil"
    shutil.copytree(run_dir, d)
    s = json.loads((d / "summary.json").read_text())
    s["notes"] = ["<script>alert('x')</script> & <b>bold</b>"]
    s["name"] = "<img src=x onerror=alert(1)>"
    (d / "summary.json").write_text(json.dumps(s))
    html = build_dashboard(d, tmp_path / "evil.html", include_plotlyjs="cdn").read_text(encoding="utf-8")
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html


def test_missing_books_degrade_gracefully(run_dir: Path, tmp_path: Path) -> None:
    d = tmp_path / "nobooks"
    shutil.copytree(run_dir, d, ignore=shutil.ignore_patterns("books"))
    html = build_dashboard(d, tmp_path / "nb.html", include_plotlyjs="cdn").read_text(encoding="utf-8")
    assert "No book time series" in html and 'id="kpis"' in html and "Combiner weights per fold" in html
    assert "unavailable" not in html


def test_bad_include_option(run_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_dashboard(run_dir, tmp_path / "x.html", include_plotlyjs="maybe")


def test_monthly_returns_math() -> None:
    idx = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h", tz="UTC")
    eq = pd.Series(100.0, index=idx)
    eq[idx >= "2024-02-01"] = 110.0      # +10% in February
    eq[idx >= "2024-03-01"] = 99.0       # -10% in March
    table = _monthly_returns(pd.DataFrame({"equity": eq}))
    np.testing.assert_allclose(table.loc[2024, [1, 2, 3]].to_numpy(dtype=float), [0.0, 0.10, -0.10])
    assert table.loc[2024, 4:].isna().all()


def test_never_more_than_eight_colour_slots() -> None:
    names = [f"s{i}" for i in range(12)]
    w = pd.DataFrame({"fold": range(4), **{n: [0.01 * (i + 1)] * 4 for i, n in enumerate(names)}, "fdm": 1.2})
    kept, slots = _strategy_slots(names, w)
    assert len(kept) == 7 and sorted(slots.values()) == list(range(7))
    assert kept == names[5:]                          # the 7 largest weights, in config order
    fig = fig_weights(_Figs(), w, None, names)
    trace_names = [t.name for t in fig.data]
    assert "Other (5)" in trace_names and len(trace_names) == 9   # 7 + Other + Unallocated
    small, small_slots = _strategy_slots(names[:3], None)
    assert small == names[:3] and small_slots == {"s0": 0, "s1": 1, "s2": 2}


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def test_cli_register_and_run(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ap = argparse.ArgumentParser(prog="aurum")
    sub = ap.add_subparsers(dest="command")
    viz_cli.register(sub)
    out = tmp_path / "cli.html"
    args = ap.parse_args(["dashboard", "--run", str(run_dir), "--out", str(out), "--cdn", "--max-trades", "20"])
    assert args.func is viz_cli.cmd_dashboard
    assert args.func(args) == 0
    assert out.exists() and "dashboard:" in capsys.readouterr().out
    assert "20 most recent" in out.read_text(encoding="utf-8")


def test_cli_main_errors(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert viz_cli.main(["--run", str(tmp_path / "missing")]) == 1
    assert "summary.json" in capsys.readouterr().err
    assert viz_cli.main(["--run", str(run_dir), "--journal", str(tmp_path / "nojournal")]) == 1
    assert viz_cli.main(["--run", str(run_dir), "--max-trades", "-1"]) == 2
    assert viz_cli.main([]) == 2                                     # argparse usage error
    out = tmp_path / "main.html"
    assert viz_cli.main(["--run", str(run_dir), "--out", str(out), "--cdn"]) == 0 and out.exists()
