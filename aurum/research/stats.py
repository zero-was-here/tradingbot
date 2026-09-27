"""Statistical inference for backtests: is this Sharpe ratio real?

A backtest Sharpe ratio is an *estimate* with sampling error, computed on non-normal returns,
and usually the *maximum* over many tried configurations. This module quantifies all three
problems. Pure numpy/scipy; every function is deterministic given its ``seed``.

Units convention (important)
----------------------------
Unless a function takes a ``periods`` argument, Sharpe ratios are **per-period** (not
annualised): ``SR = mean(r) / std(r)`` of the return series actually observed (e.g. daily).
Annualise with ``SR_ann = SR * sqrt(periods)``. Kurtosis is **Pearson (non-excess)**
kurtosis: 3 for a normal distribution. Skewness/kurtosis are the (biased) sample moments,
which guarantee ``kurt >= 1 + skew**2`` and hence a non-negative variance term below. A
``kurt`` below that bound cannot be Pearson kurtosis (it is almost surely EXCESS kurtosis, the
``aurum.backtest.metrics`` convention) and triggers a ``RuntimeWarning``.

Formulas
--------
Let ``g3`` = skewness, ``g4`` = kurtosis, ``T`` = number of observations.

* Sharpe standard error (Mertens 2002; Opdyke 2007), valid for non-normal iid returns::

      V[SR_hat] = (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)

* Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) - probability that the true
  SR exceeds a benchmark ``SR*``::

      PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

* Expected maximum of ``N`` iid ``N(mu, V)`` Sharpe estimates (Bailey & López de Prado
  2014, using the Gumbel/extreme-value approximation with the Euler-Mascheroni constant
  ``gamma ~ 0.5772``)::

      E[max SR] ~ mu + sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))

* Deflated Sharpe Ratio (Bailey & López de Prado 2014): ``DSR = PSR(SR0)`` with
  ``SR0 = E[max SR]`` over the ``N`` trials - the probability that the selected strategy's
  true SR is positive after accounting for selection among ``N`` trials and non-normality.

* Minimum Track Record Length (Bailey & López de Prado 2012)::

      MinTRL = 1 + (1 - g3*SR + (g4 - 1)/4 * SR^2) * (Phi^-1(p) / (SR - SR*))^2

* Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu 2017):
  see :func:`pbo_cscv`.

* Stationary bootstrap (Politis & Romano 1994) with automatic mean block length (Politis &
  White 2004, corrected by Patton, Politis & White 2009): see :func:`stationary_bootstrap`.

* Multiple-testing haircuts (Harvey & Liu 2015): see :func:`haircut_sharpe`.

References
----------
* A. Lo (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4).
* C. Mertens (2002), "The Sharpe ratio and the information ratio" (working note).
* J. Opdyke (2007), "Comparing Sharpe ratios: so where are the p-values?", *J. Asset Mgmt* 8.
* D. Bailey, M. López de Prado (2012), "The Sharpe Ratio Efficient Frontier", *J. Risk* 15(2).
* D. Bailey, M. López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality", *J. Portfolio Management* 40(5).
* D. Bailey, J. Borwein, M. López de Prado, Q. Zhu (2017), "The Probability of Backtest
  Overfitting", *J. Computational Finance* 20(4).
* D. Politis, J. Romano (1994), "The Stationary Bootstrap", *JASA* 89(428).
* D. Politis, H. White (2004), "Automatic Block-Length Selection for the Dependent
  Bootstrap", *Econometric Reviews* 23(1); A. Patton, D. Politis, H. White (2009),
  correction, *Econometric Reviews* 28(4).
* C. Harvey, Y. Liu (2015), "Backtesting", *J. Portfolio Management* 42(1).
* Y. Benjamini, D. Yekutieli (2001), "The control of the false discovery rate in multiple
  testing under dependency", *Annals of Statistics* 29(4).
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

EULER_MASCHERONI = 0.5772156649015329

__all__ = [
    "EULER_MASCHERONI",
    "PBOResult",
    "adjust_pvalues",
    "annualize_sharpe",
    "deannualize_sharpe",
    "deflated_sharpe",
    "expected_max_sharpe",
    "haircut_sharpe",
    "min_track_record_length",
    "optimal_block_length",
    "pbo_cscv",
    "probabilistic_sharpe",
    "return_moments",
    "sharpe",
    "sharpe_ci",
    "sharpe_pvalue",
    "sharpe_std",
    "sharpe_summary",
    "stationary_bootstrap",
    "stationary_bootstrap_indices",
]

ArrayLike = np.ndarray | pd.Series | Sequence[float]


# --------------------------------------------------------------------------------------
# basic helpers
# --------------------------------------------------------------------------------------
def _clean(returns: ArrayLike) -> np.ndarray:
    """1-D float array with NaN/inf removed."""
    x = np.asarray(returns, dtype=float).ravel()
    return x[np.isfinite(x)]


def sharpe(returns: ArrayLike, periods: float = 252.0, *, ddof: int = 1) -> float:
    """Annualised Sharpe ratio ``mean / std * sqrt(periods)`` (risk-free rate = 0).

    Pass ``periods=1`` for the per-period ratio used by the inference functions below.
    Returns NaN for fewer than two observations or zero dispersion.
    """
    x = _clean(returns)
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=ddof)
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan")
    return float(x.mean() / sd * math.sqrt(periods))


def annualize_sharpe(sr: float, periods: float) -> float:
    """Per-period -> annual: ``SR * sqrt(periods)`` (assumes iid returns)."""
    return float(sr) * math.sqrt(periods)


def deannualize_sharpe(sr_ann: float, periods: float) -> float:
    """Annual -> per-period: ``SR_ann / sqrt(periods)``."""
    return float(sr_ann) / math.sqrt(periods)


def return_moments(returns: ArrayLike) -> tuple[float, float]:
    """(skewness, Pearson kurtosis) using biased sample moments (normal -> (0, 3))."""
    x = _clean(returns)
    if x.size < 3 or x.std() == 0:
        return 0.0, 3.0
    return float(sps.skew(x, bias=True)), float(sps.kurtosis(x, fisher=False, bias=True))


def _check_pearson_kurt(skew: float, kurt: float) -> None:
    """Warn when ``kurt`` cannot be a PEARSON kurtosis.

    For any distribution (and for biased sample moments) Pearson kurtosis satisfies
    ``kurt >= 1 + skew**2`` (Pearson 1916). A smaller value almost surely means EXCESS
    kurtosis (normal = 0) was passed - e.g. ``aurum.backtest.metrics.compute_metrics``'s
    ``kurtosis`` or ``pandas.Series.kurt()`` - which understates the Sharpe variance term by
    ``3/4 * SR^2`` and overstates PSR/DSR. Warn (not raise) because bias-corrected
    estimators can dip slightly below the bound in tiny samples.
    """
    if np.isfinite(skew) and np.isfinite(kurt) and kurt < (1.0 + skew * skew) * (1.0 - 1e-9):
        warnings.warn(
            f"kurt={kurt:.4g} < 1 + skew^2 = {1.0 + skew * skew:.4g} is impossible for PEARSON "
            "kurtosis (normal = 3); this looks like EXCESS kurtosis - add 3 before calling "
            "aurum.research.stats (or use return_moments / sharpe_summary).",
            RuntimeWarning,
            stacklevel=4,
        )


def _sr_var_term(sr: float, skew: float, kurt: float) -> float:
    """``1 - g3*SR + (g4-1)/4 * SR^2`` floored at a tiny positive number."""
    _check_pearson_kurt(skew, kurt)
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if v <= 0:
        logger.debug("non-positive Sharpe variance term %.4g (skew=%.3g kurt=%.3g); flooring", v,
                     skew, kurt)
    return max(v, 1e-12)


def sharpe_std(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-period Sharpe estimate (Mertens 2002 / Opdyke 2007)."""
    if n < 2:
        return float("nan")
    return math.sqrt(_sr_var_term(sr, skew, kurt) / (n - 1))


# --------------------------------------------------------------------------------------
# PSR / DSR / MinTRL
# --------------------------------------------------------------------------------------
def probabilistic_sharpe(
    sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_star: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio ``P[true SR > sr_star]`` (Bailey & López de Prado 2012).

    ``sr`` and ``sr_star`` are per-period; ``kurt`` is Pearson kurtosis. Negative skew and
    fat tails (``kurt > 3``) widen the sampling distribution and lower the PSR of a
    positive Sharpe ratio.
    """
    if not (np.isfinite(sr) and np.isfinite(sr_star)) or n < 2:
        return float("nan")
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(_sr_var_term(sr, skew, kurt))
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var: float, mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` iid ``N(mean, var)`` Sharpe estimates.

    Uses the extreme-value approximation of Bailey & López de Prado (2014), eq. (6)::

        E[max] ~ mean + sqrt(var) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    with ``g`` the Euler-Mascheroni constant. ``n_trials <= 1`` returns ``mean`` (no
    selection). ``var`` is the cross-sectional variance of the trials' (per-period) Sharpe
    ratios - use the *effective* number of independent trials when trials are correlated.
    """
    if var < 0:
        raise ValueError("var must be non-negative")
    n = float(n_trials)
    if n <= 1:
        return float(mean)
    g = EULER_MASCHERONI
    z = (1.0 - g) * sps.norm.ppf(1.0 - 1.0 / n) + g * sps.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(mean + math.sqrt(var) * z)


def deflated_sharpe(
    sr: float,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    trial_srs: ArrayLike | None = None,
    *,
    n_trials: int | None = None,
    trial_var: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    ``DSR = PSR(SR0)`` where ``SR0 = expected_max_sharpe(N, V)`` is the Sharpe ratio one
    would expect from the best of ``N`` skill-less trials. All Sharpe ratios per-period.

    The trial statistics can be given as

    * ``trial_srs`` - the per-period Sharpe ratios of all trials (``N = len``,
      ``V = var(trial_srs, ddof=1)``), optionally with ``n_trials`` overriding ``N`` (e.g.
      an effective number of independent trials after clustering), or
    * ``n_trials`` and ``trial_var``, or
    * ``n_trials`` alone: ``V`` then defaults to ``1 / (n - 1)``, the sampling variance of
      a Sharpe estimate when the true SR is 0 - i.e. "the best of N pure-noise strategies".
      This is the *least* conservative choice; real parameter sweeps usually have larger
      dispersion.

    With ``N <= 1`` the DSR equals the PSR against 0.
    """
    if trial_srs is not None:
        t = _clean(trial_srs)
        n_eff = int(n_trials) if n_trials is not None else int(t.size)
        var = float(t.var(ddof=1)) if t.size > 1 else 0.0
        if trial_var is not None:
            var = float(trial_var)
    else:
        n_eff = int(n_trials) if n_trials is not None else 1
        var = float(trial_var) if trial_var is not None else (1.0 / (n - 1) if n > 1 else 0.0)
    sr0 = expected_max_sharpe(n_eff, var) if n_eff > 1 else 0.0
    return probabilistic_sharpe(sr, n, skew, kurt, sr_star=sr0)


def min_track_record_length(
    sr: float, skew: float = 0.0, kurt: float = 3.0, sr_star: float = 0.0, prob: float = 0.95
) -> float:
    """Minimum number of observations for ``PSR(sr_star) >= prob`` (Bailey & LdP 2012).

    ``MinTRL = 1 + (1 - g3*SR + (g4-1)/4*SR^2) * (Phi^-1(prob) / (SR - SR*))^2`` in units of
    the return periodicity (e.g. days). Returns ``inf`` when ``sr <= sr_star``.
    """
    if not 0 < prob < 1:
        raise ValueError("prob must be in (0, 1)")
    if not np.isfinite(sr) or sr <= sr_star:
        return float("inf")
    z = sps.norm.ppf(prob)
    return float(1.0 + _sr_var_term(sr, skew, kurt) * (z / (sr - sr_star)) ** 2)


# --------------------------------------------------------------------------------------
# multiple testing / haircuts
# --------------------------------------------------------------------------------------
def sharpe_pvalue(
    sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_star: float = 0.0
) -> float:
    """One-sided p-value of ``H0: SR <= sr_star`` = ``1 - PSR(sr_star)`` (per-period SR)."""
    psr = probabilistic_sharpe(sr, n, skew, kurt, sr_star)
    return float("nan") if not np.isfinite(psr) else 1.0 - psr


def adjust_pvalues(pvalues: ArrayLike, method: str = "holm") -> np.ndarray:
    """Family-wise / false-discovery-rate adjusted p-values (returned in input order).

    ``method``: ``"bonferroni"`` (FWER), ``"sidak"`` (FWER, independent tests), ``"holm"``
    (FWER step-down, Holm 1979), ``"bh"`` (FDR, Benjamini-Hochberg 1995) or ``"bhy"`` (FDR
    under arbitrary dependence, Benjamini-Yekutieli 2001 - the variant recommended by Harvey
    & Liu 2015 for correlated strategy searches).
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = p.size
    if m == 0:
        return p
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    method = method.lower()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method == "sidak":
        return 1.0 - (1.0 - p) ** m
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ranks = np.arange(1, m + 1)
    if method == "holm":
        adj = np.maximum.accumulate((m - ranks + 1) * ps)
    elif method in ("bh", "bhy"):
        c = np.sum(1.0 / ranks) if method == "bhy" else 1.0
        adj = np.minimum.accumulate((ps * m * c / ranks)[::-1])[::-1]
    else:
        raise ValueError(f"unknown method {method!r}")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def haircut_sharpe(
    sr_ann: float,
    n_obs: int,
    n_trials: int,
    *,
    periods: float = 252.0,
    method: str = "bonferroni",
) -> dict[str, float]:
    """Harvey & Liu (2015) multiple-testing haircut of an annualised Sharpe ratio.

    1. ``t = SR_per_period * sqrt(T)`` and two-sided ``p = 2 (1 - Phi(|t|))``.
    2. Adjust for ``M = n_trials`` tests: Bonferroni ``min(M p, 1)``; Sidak
       ``1 - (1 - p)^M``; Holm - identical to Bonferroni for the *best* strategy (the one
       with the smallest p-value, which is the one being reported); BHY - uses the
       conservative bound ``min(M c(M) p, 1)`` with ``c(M) = sum_{i<=M} 1/i`` since the
       other trials' p-values are unknown.
    3. Invert: ``t_adj = Phi^-1(1 - p_adj / 2)``, ``SR_adj = t_adj / sqrt(T) * sqrt(periods)``.

    Returns ``{"sr", "sr_haircut", "haircut", "t_stat", "p_value", "p_adjusted"}``;
    ``haircut`` is the fractional reduction ``1 - SR_adj / SR`` (1.0 = fully discounted).
    A non-finite ``sr_ann`` yields NaN statistics (never a fake 0.0 haircut Sharpe).
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    m = max(int(n_trials), 1)
    if not np.isfinite(sr_ann):
        nan = float("nan")
        return {"sr": float(sr_ann), "sr_haircut": nan, "haircut": nan, "t_stat": nan,
                "p_value": nan, "p_adjusted": nan, "n_trials": float(m)}
    sr_pp = deannualize_sharpe(sr_ann, periods)
    t = sr_pp * math.sqrt(n_obs)
    p = float(2.0 * sps.norm.sf(abs(t)))
    method = method.lower()
    if method in ("bonferroni", "holm"):
        p_adj = min(p * m, 1.0)
    elif method == "sidak":
        p_adj = 1.0 - (1.0 - p) ** m
    elif method == "bhy":
        c = float(np.sum(1.0 / np.arange(1, m + 1)))
        p_adj = min(p * m * c, 1.0)
    else:
        raise ValueError(f"unknown method {method!r}")
    t_adj = float(sps.norm.isf(p_adj / 2.0)) if p_adj < 1.0 else 0.0
    sr_adj = math.copysign(t_adj / math.sqrt(n_obs) * math.sqrt(periods), sr_ann)
    haircut = 1.0 - sr_adj / sr_ann if sr_ann != 0 else float("nan")
    return {
        "sr": float(sr_ann),
        "sr_haircut": float(sr_adj),
        "haircut": float(haircut),
        "t_stat": float(t),
        "p_value": p,
        "p_adjusted": float(p_adj),
        "n_trials": float(m),
    }


# --------------------------------------------------------------------------------------
# PBO via CSCV
# --------------------------------------------------------------------------------------
@dataclass
class PBOResult:
    """Output of :func:`pbo_cscv`.

    pbo               : probability of backtest overfitting = share of IS/OOS combinations in
                        which the IS-optimal configuration ranks at or below the OOS median.
    logits            : ``lambda_c = ln(w_c / (1 - w_c))`` of the OOS relative rank ``w_c``.
    selected          : index of the IS-best configuration per combination.
    is_perf, oos_perf : IS and OOS performance of the selected configuration.
    prob_oos_loss     : share of combinations with OOS performance < 0.
    degradation_slope / degradation_intercept : OLS fit of ``oos_perf`` on ``is_perf``;
                        a negative slope is the classic overfitting signature.
    """

    pbo: float
    logits: np.ndarray
    selected: np.ndarray
    is_perf: np.ndarray
    oos_perf: np.ndarray
    oos_rank: np.ndarray
    n_combinations: int
    n_splits: int
    n_strategies: int
    prob_oos_loss: float
    degradation_slope: float
    degradation_intercept: float
    meta: dict = field(default_factory=dict)

    def __float__(self) -> float:
        """``float(result)`` is the PBO itself (convenient where a scalar is expected)."""
        return float(self.pbo)

    def to_dict(self) -> dict[str, float]:
        return {
            "pbo": self.pbo,
            "logit_mean": float(np.mean(self.logits)) if self.logits.size else float("nan"),
            "prob_oos_loss": self.prob_oos_loss,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "n_combinations": float(self.n_combinations),
            "n_splits": float(self.n_splits),
            "n_strategies": float(self.n_strategies),
        }


def _combo_sharpe(
    cnt: np.ndarray, s1: np.ndarray, s2c: np.ndarray, center: np.ndarray, metric: str
) -> np.ndarray:
    """Per-combination performance from aggregated block sums.

    ``s1`` = sum of returns, ``s2c`` = sum of squared *centred* returns (centred on the
    full-sample column mean ``center`` for numerical stability). Zero-variance columns
    score 0 (they cannot be preferred on a risk-adjusted basis).
    """
    c = cnt[:, None]
    mean = s1 / c
    if metric == "mean":
        return mean
    var = (s2c - c * (mean - center) ** 2) / np.maximum(c - 1.0, 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 1e-15)


def pbo_cscv(
    perf_matrix: np.ndarray | pd.DataFrame,
    n_splits: int = 16,
    *,
    metric: str | Callable[[np.ndarray], np.ndarray] = "sharpe",
    max_combinations: int | None = None,
    seed: int = 0,
) -> PBOResult:
    """Probability of Backtest Overfitting by Combinatorially Symmetric CV (Bailey et al.).

    ``perf_matrix`` is ``T x N``: per-period returns of ``N`` configurations (columns) over
    the same ``T`` periods. Algorithm (Bailey, Borwein, López de Prado & Zhu 2017, sec. 2):

    1. Cut the rows into ``S = n_splits`` (even) contiguous blocks.
    2. For each of the ``C(S, S/2)`` ways to pick half the blocks as in-sample (IS) - the
       complement being out-of-sample (OOS) - compute every configuration's performance
       (default: per-period Sharpe) IS and OOS.
    3. Select ``n* = argmax IS``. Let ``w`` be the relative OOS rank of ``n*``,
       ``rank / (N + 1)`` (average rank on ties, rank ``N`` = best) and
       ``lambda = logit(w)``.
    4. ``PBO = share of combinations with lambda <= 0``: how often the IS winner is no
       better than the OOS median.

    PBO ~ 0 means the IS winner reliably stays good OOS; PBO ~ 0.5 or more means the
    selection procedure has no skill (for pure noise PBO is typically ~0.5, somewhat above
    0.5 for small ``T`` because IS and OOS halves of one finite sample are negatively
    related). ``metric`` may be ``"sharpe"``, ``"mean"`` or a callable mapping a
    ``(rows x N)`` array to ``N`` scores (slower, looped). ``max_combinations`` randomly
    samples that many combinations (seeded) when ``C(S, S/2)`` is too large.
    """
    m = np.asarray(perf_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T periods x N configurations)")
    t_len, n_cfg = m.shape
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even integer >= 2")
    if n_cfg < 2:
        raise ValueError("need at least 2 configurations")
    if t_len < 2 * n_splits:
        raise ValueError(f"need at least {2 * n_splits} rows for n_splits={n_splits}, got {t_len}")
    if not np.all(np.isfinite(m)):
        logger.warning("pbo_cscv: non-finite values replaced by 0")
        m = np.where(np.isfinite(m), m, 0.0)

    blocks = np.array_split(np.arange(t_len), n_splits)
    half = n_splits // 2
    total = math.comb(n_splits, half)
    if max_combinations is not None and int(max_combinations) < 1:
        raise ValueError("max_combinations must be >= 1")
    if max_combinations is not None and max_combinations < total:
        # Sample DISTINCT combinations (without replacement): duplicates would over-weight
        # some IS/OOS partitions in the PBO average.
        rng = np.random.default_rng(seed)
        chosen: dict[tuple[int, ...], None] = {}
        while len(chosen) < int(max_combinations):
            chosen.setdefault(tuple(np.sort(rng.permutation(n_splits)[:half]).tolist()), None)
        combos_arr = np.array(list(chosen), dtype=np.int64)
    else:
        combos_arr = np.array(list(combinations(range(n_splits), half)))
    n_comb = combos_arr.shape[0]
    ind = np.zeros((n_comb, n_splits), dtype=float)
    ind[np.repeat(np.arange(n_comb), half), combos_arr.ravel()] = 1.0

    if isinstance(metric, str):
        if metric not in ("sharpe", "mean"):
            raise ValueError("metric must be 'sharpe', 'mean' or a callable")
        center = m.mean(axis=0)
        b_cnt = np.array([len(b) for b in blocks], dtype=float)
        b_s1 = np.vstack([m[b].sum(axis=0) for b in blocks])
        b_s2c = np.vstack([((m[b] - center) ** 2).sum(axis=0) for b in blocks])
        is_cnt, oos_cnt = ind @ b_cnt, (1.0 - ind) @ b_cnt
        is_perf_all = _combo_sharpe(is_cnt, ind @ b_s1, ind @ b_s2c, center, metric)
        oos_perf_all = _combo_sharpe(
            oos_cnt, (1.0 - ind) @ b_s1, (1.0 - ind) @ b_s2c, center, metric
        )
    else:
        is_perf_all = np.empty((n_comb, n_cfg))
        oos_perf_all = np.empty((n_comb, n_cfg))
        for c in range(n_comb):
            mask = ind[c].astype(bool)
            is_rows = np.concatenate([blocks[j] for j in np.flatnonzero(mask)])
            oos_rows = np.concatenate([blocks[j] for j in np.flatnonzero(~mask)])
            is_perf_all[c] = np.asarray(metric(m[is_rows]), dtype=float)
            oos_perf_all[c] = np.asarray(metric(m[oos_rows]), dtype=float)
        is_perf_all = np.nan_to_num(is_perf_all, nan=-np.inf)
        oos_perf_all = np.nan_to_num(oos_perf_all, nan=-np.inf)

    sel = np.argmax(is_perf_all, axis=1)
    rows = np.arange(n_comb)
    oos_sel = oos_perf_all[rows, sel]
    less = (oos_perf_all < oos_sel[:, None]).sum(axis=1)
    equal = (oos_perf_all == oos_sel[:, None]).sum(axis=1)
    rank = less + (equal + 1) / 2.0  # average rank in 1..N, N = best
    w = rank / (n_cfg + 1.0)
    logits = np.log(w / (1.0 - w))
    pbo = float(np.mean(logits <= 0.0))

    is_sel = is_perf_all[rows, sel]
    finite = np.isfinite(is_sel) & np.isfinite(oos_sel)
    if finite.sum() >= 2 and np.ptp(is_sel[finite]) > 0:
        slope, intercept = np.polyfit(is_sel[finite], oos_sel[finite], 1)
    else:
        slope, intercept = float("nan"), float("nan")
    res = PBOResult(
        pbo=pbo,
        logits=logits,
        selected=sel,
        is_perf=is_sel,
        oos_perf=oos_sel,
        oos_rank=w,
        n_combinations=int(n_comb),
        n_splits=int(n_splits),
        n_strategies=int(n_cfg),
        prob_oos_loss=float(np.mean(oos_sel < 0.0)),
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        meta={"metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "fn")},
    )
    logger.debug("pbo_cscv: T=%d N=%d S=%d combos=%d -> PBO=%.3f", t_len, n_cfg, n_splits,
                 n_comb, pbo)
    return res


# --------------------------------------------------------------------------------------
# stationary bootstrap
# --------------------------------------------------------------------------------------
def _flat_top(t: np.ndarray) -> np.ndarray:
    """Politis-Romano trapezoidal flat-top lag window: 1 on |t|<=1/2, linear to 0 at |t|=1."""
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(returns: ArrayLike) -> dict[str, float]:
    """Automatic block length for the stationary and circular block bootstraps.

    Politis & White (2004) with the Patton, Politis & White (2009) correction:

    1. ``m_hat`` = smallest lag after which ``K_N = max(5, ceil(sqrt(log10 n)))``
       consecutive autocorrelations are insignificant (``|rho| < 2 sqrt(log10(n)/n)``);
       bandwidth ``M = min(2 m_hat, ceil(sqrt n) + K_N)``.
    2. Flat-top kernel estimates ``g = sum_{|k|<=M} lam(k/M) R(k)`` (spectral density at 0,
       times 2 pi) and ``G = sum_{|k|<=M} lam(k/M) |k| R(k)``.
    3. ``b_SB = (2 G^2 / D_SB)^(1/3) n^(1/3)`` with ``D_SB = 2 g^2``;
       ``b_CB`` uses ``D_CB = 4/3 g^2``. Both clipped to ``[1, ceil(min(3 sqrt n, n/3))]``.

    For iid data ``G ~ 0`` and the block length collapses to 1 (the ordinary bootstrap);
    persistent series (e.g. overlapping-horizon strategy returns) get long blocks.
    Returns ``{"stationary", "circular", "m_hat", "bandwidth"}``.
    """
    x = _clean(returns)
    n = x.size
    if n < 8:
        return {"stationary": 1.0, "circular": 1.0, "m_hat": 0.0, "bandwidth": 0.0}
    e = x - x.mean()
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    b_max = float(math.ceil(min(3.0 * math.sqrt(n), n / 3.0)))
    max_lag = min(m_max + kn, n - 1)
    acov = np.array([e[k:] @ e[: n - k] for k in range(max_lag + 1)]) / n
    if acov[0] <= 0:
        return {"stationary": 1.0, "circular": 1.0, "m_hat": 0.0, "bandwidth": 0.0}
    rho = acov / acov[0]
    thr = 2.0 * math.sqrt(math.log10(n) / n)
    insig = np.abs(rho[1:]) < thr  # insig[j] <-> lag j+1
    m_hat = m_max
    for m in range(0, max_lag - kn + 1):
        if insig[m : m + kn].all():
            m_hat = m
            break
    bw = int(min(2 * m_hat, m_max, max_lag))
    if bw == 0:
        g0 = acov[0]
        big_g = 0.0
    else:
        k = np.arange(1, bw + 1)
        lam = _flat_top(k / bw)
        g0 = acov[0] + 2.0 * np.sum(lam * acov[k])
        big_g = 2.0 * np.sum(lam * k * acov[k])
    if g0 <= 0:
        # Negative spectral estimate at 0 (strong negative dependence): fall back to n^(1/3).
        b = float(np.clip(n ** (1.0 / 3.0), 1.0, b_max))
        return {"stationary": b, "circular": b, "m_hat": float(m_hat), "bandwidth": float(bw)}
    ratio = big_g**2 / g0**2
    b_sb = (ratio) ** (1.0 / 3.0) * n ** (1.0 / 3.0)       # (2 G^2 / (2 g^2))^(1/3)
    b_cb = (1.5 * ratio) ** (1.0 / 3.0) * n ** (1.0 / 3.0)  # (2 G^2 / (4/3 g^2))^(1/3)
    return {
        "stationary": float(np.clip(b_sb, 1.0, b_max)),
        "circular": float(np.clip(b_cb, 1.0, b_max)),
        "m_hat": float(m_hat),
        "bandwidth": float(bw),
    }


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n)`` resampling indices of the stationary bootstrap (Politis & Romano 1994).

    Each resample concatenates blocks that start at a uniformly random position and whose
    lengths are iid Geometric(``p = 1/mean_block``), wrapping circularly. The resampled
    series is itself strictly stationary, unlike fixed-length block bootstraps.
    Vectorised: a new block starts at ``t`` with probability ``p``; position within the
    block is ``t - (last block start)``.
    """
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / float(mean_block)
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    t = np.arange(n)
    last = np.where(new_block, t, 0)
    np.maximum.accumulate(last, axis=1, out=last)
    start_vals = np.take_along_axis(starts, last, axis=1)
    return (start_vals + (t - last)) % n


def stationary_bootstrap(
    returns: ArrayLike,
    stat_fn: Callable[[np.ndarray], Any],
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int = 0,
    *,
    alpha: float = 0.05,
    vectorized: bool = False,
    ci_method: str = "percentile",
    return_samples: bool = False,
) -> dict[str, Any]:
    """Stationary-bootstrap confidence interval for any statistic of a return series.

    Block resampling preserves the serial dependence (volatility clustering, overlapping
    positions) that an iid bootstrap would destroy - destroying it typically makes CIs of
    the Sharpe ratio too narrow. ``mean_block=None`` selects the block length automatically
    with :func:`optimal_block_length`.

    Parameters
    ----------
    stat_fn    : ``f(sample_1d) -> float``; with ``vectorized=True`` it instead receives a
                 ``(chunk, n)`` array and must return ``(chunk,)`` (much faster).
    ci_method  : ``"percentile"`` (default) or ``"basic"`` (``2 theta - q``).

    Returns a dict with ``estimate, lower, upper, alpha, std_error, bias, mean_block,
    n_boot, n, ci_method`` and ``prob_le_zero`` (share of resampled statistics <= 0 - a
    bootstrap p-value for "the statistic is positive"), plus ``samples`` if requested.
    """
    x = _clean(returns)
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 finite observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if mean_block is None:
        mean_block = optimal_block_length(x)["stationary"]
    mean_block = float(min(max(mean_block, 1.0), n))
    rng = np.random.default_rng(seed)
    estimate = float(stat_fn(x[None, :])[0]) if vectorized else float(stat_fn(x))
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    samples = np.empty(n_boot)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = stationary_bootstrap_indices(n, size, mean_block, rng)
        xs = x[idx]
        if vectorized:
            samples[done : done + size] = np.asarray(stat_fn(xs), dtype=float)
        else:
            samples[done : done + size] = [float(stat_fn(row)) for row in xs]
        done += size
    ok = samples[np.isfinite(samples)]
    if ok.size < n_boot:
        logger.warning("stationary_bootstrap: %d non-finite statistics dropped", n_boot - ok.size)
    if ok.size == 0:
        lo = hi = se = bias = p0 = float("nan")
    else:
        q_lo, q_hi = np.quantile(ok, [alpha / 2.0, 1.0 - alpha / 2.0])
        if ci_method == "percentile":
            lo, hi = float(q_lo), float(q_hi)
        elif ci_method == "basic":
            lo, hi = float(2 * estimate - q_hi), float(2 * estimate - q_lo)
        else:
            raise ValueError("ci_method must be 'percentile' or 'basic'")
        se = float(ok.std(ddof=1)) if ok.size > 1 else float("nan")
        bias = float(ok.mean() - estimate)
        p0 = float(np.mean(ok <= 0.0))
    out: dict[str, Any] = {
        "estimate": estimate,
        "lower": lo,
        "upper": hi,
        "alpha": float(alpha),
        "std_error": se,
        "bias": bias,
        "prob_le_zero": p0,
        "mean_block": mean_block,
        "n_boot": int(n_boot),
        "n": int(n),
        "ci_method": ci_method,
    }
    if return_samples:
        out["samples"] = samples
    return out


def _sharpe_rows(xs: np.ndarray) -> np.ndarray:
    """Row-wise per-period Sharpe of a 2-D array (NaN for zero dispersion)."""
    mu = xs.mean(axis=1)
    sd = xs.std(axis=1, ddof=1)
    return np.divide(mu, sd, out=np.full_like(mu, np.nan), where=sd > 0)


def sharpe_ci(
    returns: ArrayLike,
    periods: float = 252.0,
    *,
    alpha: float = 0.05,
    method: str = "bootstrap",
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Confidence interval for the **annualised** Sharpe ratio.

    ``method="bootstrap"`` - stationary bootstrap (robust to autocorrelation and fat tails);
    ``method="analytic"`` - normal approximation with the Mertens/Opdyke standard error
    (robust to non-normality, assumes iid). Output keys as :func:`stationary_bootstrap`
    (``estimate``/``lower``/``upper``/``std_error`` annualised) plus ``method``.
    """
    x = _clean(returns)
    scale = math.sqrt(periods)
    if method == "analytic":
        sr = sharpe(x, periods=1.0)
        g3, g4 = return_moments(x)
        se = sharpe_std(sr, x.size, g3, g4)
        z = float(sps.norm.ppf(1.0 - alpha / 2.0))
        return {
            "estimate": float(sr * scale),
            "lower": float((sr - z * se) * scale),
            "upper": float((sr + z * se) * scale),
            "alpha": float(alpha),
            "std_error": float(se * scale),
            "prob_le_zero": 1.0 - probabilistic_sharpe(sr, x.size, g3, g4, 0.0),
            "n": int(x.size),
            "method": "analytic",
        }
    if method != "bootstrap":
        raise ValueError("method must be 'bootstrap' or 'analytic'")
    res = stationary_bootstrap(
        x, _sharpe_rows, n_boot=n_boot, mean_block=mean_block, seed=seed, alpha=alpha,
        vectorized=True,
    )
    for key in ("estimate", "lower", "upper", "std_error", "bias"):
        res[key] = res[key] * scale
    res["method"] = "bootstrap"
    return res


# --------------------------------------------------------------------------------------
# one-stop summary
# --------------------------------------------------------------------------------------
def sharpe_summary(
    returns: ArrayLike,
    periods: float = 252.0,
    *,
    n_trials: int | None = None,
    trial_sharpes_ann: ArrayLike | None = None,
    sr_star_ann: float = 0.0,
    alpha: float = 0.05,
    n_boot: int = 1000,
    bootstrap: bool = True,
    seed: int = 0,
    prob: float = 0.95,
) -> dict[str, float]:
    """All the "is it real?" statistics of one return series in one dict.

    Inputs and Sharpe outputs are **annualised** (``periods`` per year); track-record
    lengths are reported both in periods and years. ``n_trials``/``trial_sharpes_ann``
    feed the Deflated Sharpe Ratio and the multiple-testing haircut; without them DSR is
    computed with ``N = 1`` (equal to PSR).
    """
    x = _clean(returns)
    n = int(x.size)
    out: dict[str, float] = {"n_obs": float(n), "periods_per_year": float(periods)}
    if n < 3:
        logger.info("sharpe_summary: fewer than 3 observations")
        return out
    sr = sharpe(x, periods=1.0)
    g3, g4 = return_moments(x)
    sr_star = deannualize_sharpe(sr_star_ann, periods)
    trial_pp = None
    if trial_sharpes_ann is not None:
        trial_pp = _clean(trial_sharpes_ann) / math.sqrt(periods)
    n_eff = int(n_trials) if n_trials is not None else (
        int(trial_pp.size) if trial_pp is not None else 1
    )
    out.update(
        {
            "sharpe": sr * math.sqrt(periods) if np.isfinite(sr) else float("nan"),
            "sharpe_per_period": sr,
            "skew": g3,
            "kurtosis": g4,
            "sharpe_se": sharpe_std(sr, n, g3, g4) * math.sqrt(periods),
            "psr": probabilistic_sharpe(sr, n, g3, g4, sr_star),
            "n_trials": float(n_eff),
        }
    )
    if trial_pp is not None and trial_pp.size > 1:
        var = float(trial_pp.var(ddof=1))
    else:
        var = 1.0 / (n - 1)
    sr0 = expected_max_sharpe(n_eff, var) if n_eff > 1 else 0.0
    out["sr0"] = sr0 * math.sqrt(periods)
    out["dsr"] = probabilistic_sharpe(sr, n, g3, g4, sr_star=sr0)
    mtrl = min_track_record_length(sr, g3, g4, sr_star, prob)
    out["min_trl"] = mtrl
    out["min_trl_years"] = mtrl / periods if np.isfinite(mtrl) else float("inf")
    if np.isfinite(sr) and sr != 0:
        hc = haircut_sharpe(out["sharpe"], n, n_eff, periods=periods, method="bhy")
        out["sharpe_haircut"] = hc["sr_haircut"]
        out["haircut"] = hc["haircut"]
    if bootstrap:
        ci = sharpe_ci(x, periods, alpha=alpha, n_boot=n_boot, seed=seed)
        out["ci_lower"] = ci["lower"]
        out["ci_upper"] = ci["upper"]
        out["ci_alpha"] = float(alpha)
        out["bootstrap_block"] = ci["mean_block"]
        out["bootstrap_prob_le_zero"] = ci["prob_le_zero"]
    return out
