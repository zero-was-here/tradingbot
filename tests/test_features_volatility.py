"""Volatility helpers & the ``volatility`` group: estimator accuracy on simulated Brownian
OHLC (known true vol), drift/gap robustness, ATR mechanics and standalone importability."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.volatility import (
    atr,
    bar_minutes,
    close_to_close_vol,
    default_bars_per_year,
    garman_klass_vol,
    parkinson_vol,
    rogers_satchell_vol,
    safe_div,
    true_range,
    volatility_features,
    yang_zhang_vol,
)

BPY = 6000.0
SIGMA = 0.20


def brownian_bars(n: int = 2000, steps: int = 400, *, seed: int = 0, drift_bar_sig: float = 0.0,
                  gap_sig_frac: float = 0.0) -> pd.DataFrame:
    """OHLC sampled from a fine Brownian path with annual vol SIGMA at BPY bars/year.

    ``drift_bar_sig``: drift per bar in units of per-bar sigma; ``gap_sig_frac``: opening
    gap st.dev. per bar in units of per-bar sigma (true total vol = SIGMA*sqrt(1+frac^2)).
    """
    rng = np.random.default_rng(seed)
    s_bar = SIGMA / np.sqrt(BPY)
    inc = rng.normal(drift_bar_sig * s_bar / steps, s_bar / np.sqrt(steps), (n, steps))
    gaps = rng.normal(0.0, gap_sig_frac * s_bar, n)
    gaps[0] = 0.0
    within = np.cumsum(inc, axis=1)
    start = np.log(1800.0) + np.cumsum(gaps + np.r_[0.0, within[:-1, -1]])
    lp = start[:, None] + within
    o, c = np.exp(start), np.exp(lp[:, -1])
    h = np.maximum(o, np.exp(lp.max(axis=1)))
    lo = np.minimum(o, np.exp(lp.min(axis=1)))
    idx = pd.date_range("2021-01-04", periods=n, freq="h", tz="UTC")
    return make_bars(pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx), "H1",
                     default_spread=0.2)


ESTIMATORS = [close_to_close_vol, parkinson_vol, garman_klass_vol, rogers_satchell_vol, yang_zhang_vol]


@pytest.mark.parametrize("fn", ESTIMATORS, ids=lambda f: f.__name__)
def test_estimators_recover_true_vol(fn) -> None:
    bars = brownian_bars(seed=0)
    est = fn(bars, len(bars) - 1, BPY).iloc[-1]
    # Range estimators are biased ~-3% by discrete sampling of the extremes (400 steps).
    assert est == pytest.approx(SIGMA, rel=0.08)


def test_drift_robustness() -> None:
    bars = brownian_bars(seed=1, drift_bar_sig=1.0)
    n = len(bars) - 1
    rs = rogers_satchell_vol(bars, n, BPY).iloc[-1]
    yz = yang_zhang_vol(bars, n, BPY).iloc[-1]
    pk = parkinson_vol(bars, n, BPY).iloc[-1]
    assert rs == pytest.approx(SIGMA, rel=0.08)
    assert yz == pytest.approx(SIGMA, rel=0.08)
    assert pk > 1.1 * rs  # Parkinson is inflated by drift (Rogers & Satchell 1991)


def test_yang_zhang_captures_opening_gaps() -> None:
    bars = brownian_bars(seed=2, gap_sig_frac=1.0)
    n = len(bars) - 1
    total = SIGMA * np.sqrt(2.0)
    assert yang_zhang_vol(bars, n, BPY).iloc[-1] == pytest.approx(total, rel=0.08)
    assert close_to_close_vol(bars, n, BPY).iloc[-1] == pytest.approx(total, rel=0.08)
    # Pure range estimators ignore the gap component.
    assert rogers_satchell_vol(bars, n, BPY).iloc[-1] == pytest.approx(SIGMA, rel=0.08)


def test_true_range_and_atr_mechanics() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    close = np.full(40, 100.0)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=idx)
    df.iloc[20, df.columns.get_loc("open")] = 104.0  # gap up bar
    df.iloc[20, df.columns.get_loc("high")] = 105.0
    df.iloc[20, df.columns.get_loc("close")] = 104.0
    df.iloc[20, df.columns.get_loc("low")] = 103.5
    bars = make_bars(df, "H1", default_spread=0.1)
    tr = true_range(bars)
    assert tr.iloc[0] == pytest.approx(2.0)          # no previous close: H - L
    assert tr.iloc[20] == pytest.approx(5.0)         # |H - prev C| = 105 - 100
    assert tr.iloc[21] == pytest.approx(5.0)         # |L - prev C| = 104 - 99
    a = atr(bars, 14)
    assert a.iloc[:13].isna().all() and a.iloc[13] == pytest.approx(2.0)
    # Wilder recursion reference
    ref = np.empty(40)
    ref[0] = tr.iloc[0]
    for i in range(1, 40):
        ref[i] = ref[i - 1] + (tr.iloc[i] - ref[i - 1]) / 14.0
    np.testing.assert_allclose(a.iloc[13:].to_numpy(), ref[13:], rtol=1e-12)


def test_bars_per_year_is_timeframe_based() -> None:
    h1 = make_synthetic_bars(50, "H1")
    d1 = make_synthetic_bars(50, "D1", weekend_gaps=False)
    assert bar_minutes(h1) == 60 and default_bars_per_year(h1) == pytest.approx(252 * 23)
    assert default_bars_per_year(d1) == pytest.approx(252)
    h1_noattrs = h1.copy()
    h1_noattrs.attrs = {}
    assert bar_minutes(h1_noattrs) == 60  # falls back to available_at - open


def test_volatility_group_columns_and_values() -> None:
    bars = make_synthetic_bars(3000, "H1", seed=4, annual_vol=0.16)
    out = volatility_features(MarketData(bars=bars))
    assert out.index.equals(bars.index)
    for est in ("cc", "pk", "gk", "rs", "yz"):
        for n in (24, 120):
            assert f"volatility_{est}_{n}" in out.columns
    assert {"volatility_cc_480", "volatility_yz_480", "volatility_ewma", "volatility_volofvol",
            "volatility_atr_pct", "volatility_logratio_yz_24_120"} <= set(out.columns)
    assert np.isfinite(out.to_numpy()[~np.isnan(out.to_numpy())]).all()
    assert out["volatility_cc_24"].iloc[:24].isna().all() and out["volatility_cc_24"].iloc[24:].notna().all()
    # close-close vol of the synthetic GBM ≈ annual_vol * sqrt(252*23 / realised bars/year)
    realised_bpy = 119 * 52.18
    expected = 0.16 * np.sqrt(252 * 23 / realised_bpy)
    assert out["volatility_cc_480"].iloc[-1] == pytest.approx(expected, rel=0.15)


def test_safe_div_never_returns_inf() -> None:
    out = safe_div(np.array([1.0, 1.0, np.nan, 2.0]), np.array([0.0, np.inf, 1.0, 4.0]))
    assert np.isnan(out[:3]).all() and out[3] == 0.5


def test_volatility_helpers_import_standalone() -> None:
    """Strategies/sizing import these helpers: they must not drag in other feature modules."""
    code = (
        "import sys\n"
        "from aurum.features.volatility import atr, true_range, yang_zhang_vol, parkinson_vol, "
        "garman_klass_vol, rogers_satchell_vol\n"
        "bad = [m for m in ('aurum.features.technical', 'aurum.features.pipeline', "
        "'aurum.features.macro') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
