"""configs/d1_trend.yaml and configs/managed_gold.yaml: validation, holdout protection, the DSR
trial count, and an end-to-end walk-forward of their strategy sets on synthetic daily bars."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.cli import main as aurum_main
from aurum.core.config import load_config
from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.research.walkforward import run_walk_forward

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"d1_trend": ROOT / "configs" / "d1_trend.yaml", "managed_gold": ROOT / "configs" / "managed_gold.yaml"}
#: last allowed bar OPEN: the 2024-12-31 NY session opens 2024-12-30 22:00 UTC; the next bar
#: (opening 2024-12-31 22:00 UTC) is the first session of the used-up 2025+ holdout.
HOLDOUT_EDGE = pd.Timestamp("2024-12-31 22:00", tz="UTC")


@pytest.mark.parametrize("name", list(CONFIGS))
def test_config_validates_and_builds(name: str) -> None:
    cfg = load_config(CONFIGS[name], env={}, strict=True)
    assert cfg.name == name
    assert cfg.data.timeframe == "D1"
    assert cfg.data.bars_path.endswith("xauusd_D1_nyclose.parquet")
    strategies = cfg.build_strategies()
    expected = {"d1_trend": ["tsmom", "ema_cross", "donchian"],
                "managed_gold": ["managed_gold", "regime_switch", "tsmom"]}[name]
    assert list(strategies) == expected
    for s in strategies.values():
        assert s.warmup_bars <= 260, "daily params: warm-up of about one year at most"
    # costs / financing / sizing are the default (H1 protocol) ones
    base = load_config(ROOT / "configs" / "default.yaml", env={})
    assert cfg.costs == base.costs and cfg.sizing == base.sizing and cfg.combiner == base.combiner


@pytest.mark.parametrize("name", list(CONFIGS))
def test_config_cannot_touch_the_used_up_holdout(name: str) -> None:
    cfg = load_config(CONFIGS[name], env={})
    assert cfg.walkforward.holdout_start is None
    end = pd.Timestamp(cfg.data.end)
    assert end < HOLDOUT_EDGE, f"data.end {end} would include a 2025 session"
    assert cfg.walkforward.embargo == 1


@pytest.mark.parametrize("name", list(CONFIGS))
def test_dsr_counts_the_exploratory_trials(name: str) -> None:
    """23 pre-registered + 5 new strategy configurations + 2 combined books."""
    assert load_config(CONFIGS[name], env={}).walkforward.n_trials == 30


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cli_config_validate(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert aurum_main(["config", "validate", "-c", str(CONFIGS[name])]) == 0
    assert "config OK" in capsys.readouterr().out


def test_daily_tsmom_horizons_are_monthly() -> None:
    cfg = load_config(CONFIGS["d1_trend"], env={})
    tsmom = cfg.build_strategies()["tsmom"]
    assert tsmom.params["horizons"] == (21, 63, 126, 252)
    mg = load_config(CONFIGS["managed_gold"], env={}).build_strategies()
    assert mg["managed_gold"].params["horizons"] == (21, 63, 126, 252)
    assert mg["regime_switch"].params["returns"] == "bar"
    assert mg["regime_switch"].base_.params == tsmom.params        # same trend follower underneath
    assert mg["tsmom"].params == tsmom.params                       # identical trial, counted once


@pytest.fixture(scope="module")
def synthetic_daily() -> MarketData:
    bars = make_synthetic_bars(2600, "D1", seed=9, model="regime")
    return MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=9),
                      events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=14)))


def test_managed_gold_config_walk_forward_end_to_end(synthetic_daily: MarketData, tmp_path: Path) -> None:
    """The config's strategy set runs through the real walk-forward engine (per-fold HMM
    refits, combiner, one stitched backtest per book) and writes a complete run directory."""
    cfg = load_config(CONFIGS["managed_gold"], env={}, overrides=[
        "walkforward.train=2Y", "walkforward.test=6M", "walkforward.executor=serial", "walkforward.n_boot=200",
        "walkforward.pbo_splits=8", "output.tearsheet=false", "risk.research.max_spread=null",
        "walkforward.combiner_min_obs=100"])
    rep = run_walk_forward(synthetic_daily, cfg, out_dir=tmp_path / "run", write=True)
    assert set(rep.stats.index) >= {"combined", "managed_gold", "regime_switch", "tsmom", "benchmark"}
    assert rep.n_trials == 30
    assert rep.settings["n_folds"] >= 4
    f = rep.oos_forecasts
    assert f["managed_gold"].between(0, 1).all() and f["managed_gold"].mean() > 0.1
    assert (f["regime_switch"].abs() <= f["tsmom"].abs() + 1e-12).all(), "the regime filter only scales down"
    assert np.isfinite(rep.stats.loc["combined", "sharpe"])
    for p in ("summary.json", "folds.csv", "weights.csv", "costs.csv", "oos_forecasts.parquet",
              "books/combined/timeseries.parquet", "books/managed_gold/trades.csv"):
        assert (tmp_path / "run" / p).exists(), p


def test_strategies_pickle_for_process_workers() -> None:
    """The walk-forward process executor pickles strategies into spawned workers."""
    for s in load_config(CONFIGS["managed_gold"], env={}).build_strategies().values():
        clone = pickle.loads(pickle.dumps(s))
        assert type(clone) is type(s) and clone.params == s.params
