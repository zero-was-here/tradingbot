"""Random-walk sanity check: on a driftless GBM nothing is predictable, so no feature may be
significantly correlated with the NEXT bar's return.

A feature that is (accidentally) built with ``shift(-1)``, a centred window, or data joined
before it was available shows up as a highly significant correlation with ``r[t+1]``.

Statistics: Spearman rank correlation (robust to the fat tails of ratio features) between
each feature ``x_t`` and ``r_{t+1}`` on post-warm-up rows. Under H0 (r iid and independent
of the past) ``atanh(rho) * sqrt(n-3)`` is ~N(0,1) even if ``x`` is serially correlated
(``x_t r_{t+1}`` is a martingale-difference sequence). Bonferroni-corrected family-wise
alpha = 1% over all columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events, make_synthetic_macro
from aurum.features.pipeline import FeaturePipeline

N_BARS = 20_000
FAMILY_ALPHA = 0.01


@pytest.fixture(scope="module")
def rw_features() -> tuple[pd.DataFrame, pd.Series, int]:
    bars = make_synthetic_bars(N_BARS, "H1", seed=11, model="gbm")
    md = MarketData(bars=bars, macro=make_synthetic_macro(bars, seed=11),
                    events=make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=7)))
    pipe = FeaturePipeline()
    raw = pipe.compute(md)
    # Target for the TEST ONLY: the next bar's log return (shift(-1) is fine here).
    nxt = np.log(bars["close"]).diff().shift(-1)
    return raw, nxt, pipe.max_lookback


def _significant(raw: pd.DataFrame, target: pd.Series, start: int) -> tuple[pd.DataFrame, float]:
    x = raw.iloc[start:-1]
    y = target.iloc[start:-1]
    y_rank = y.rank().to_numpy()
    rows = []
    for c in x.columns:
        v = x[c].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(y.to_numpy())
        n = int(ok.sum())
        if n < 500 or np.nanstd(v[ok]) == 0:
            continue
        xr = pd.Series(v[ok]).rank().to_numpy()
        yr = pd.Series(y_rank[ok]).rank().to_numpy()
        rho = float(np.corrcoef(xr, yr)[0, 1])
        z = np.arctanh(np.clip(rho, -0.999999, 0.999999)) * np.sqrt(n - 3)
        rows.append({"column": c, "rho": rho, "n": n, "p": 2 * stats.norm.sf(abs(z))})
    res = pd.DataFrame(rows).set_index("column")
    return res, FAMILY_ALPHA / max(len(res), 1)


def test_no_feature_predicts_next_bar_on_random_walk(rw_features) -> None:
    raw, nxt, warmup = rw_features
    res, alpha = _significant(raw, nxt, warmup)
    assert len(res) >= 150, f"too few testable columns ({len(res)})"
    bad = res[res["p"] < alpha]
    assert bad.empty, f"features predictive of r[t+1] on a random walk (look-ahead?):\n{bad}"
    # Effect sizes must also be economically negligible.
    assert res["rho"].abs().max() < 0.04


def test_randomwalk_check_detects_shift_minus_one(rw_features) -> None:
    """Negative control: accidental shift(-1) / centred windows are caught."""
    raw, nxt, warmup = rw_features
    r = nxt.shift(1)  # r[t], the current bar's return
    noise = np.random.default_rng(0).normal(0.0, 4.0 * float(r.std()), len(r))
    leaky = pd.DataFrame({
        "leak_shift": r.shift(-1),                          # = r[t+1]
        "leak_centred": r.rolling(9, center=True).mean(),   # includes r[t+1..t+4]
        "leak_noisy": r.shift(-1) + noise,                  # weak leak buried in noise
    }, index=raw.index)
    res, _ = _significant(leaky, nxt, warmup)
    alpha = FAMILY_ALPHA / (len(raw.columns) + len(leaky.columns))
    assert (res["p"] < alpha).all(), res
