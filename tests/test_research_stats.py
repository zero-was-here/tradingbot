"""Tests for aurum.research.stats: PSR/DSR/MinTRL, PBO (CSCV), stationary bootstrap, haircuts."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from aurum.data.synthetic import make_synthetic_bars
from aurum.research import stats as st


# ---------------------------------------------------------------------------- Sharpe basics
def test_sharpe_and_annualisation() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 5000)
    sr = st.sharpe(r, periods=1)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1))
    assert st.sharpe(r, 252) == pytest.approx(sr * math.sqrt(252))
    assert st.annualize_sharpe(st.deannualize_sharpe(1.3, 252), 252) == pytest.approx(1.3)
    assert math.isnan(st.sharpe([0.01]))
    assert math.isnan(st.sharpe(np.zeros(10)))
    assert st.sharpe([0.01, np.nan, 0.02, -0.01], 1) == pytest.approx(st.sharpe([0.01, 0.02, -0.01], 1))


def test_return_moments_normal_and_pearson_convention() -> None:
    rng = np.random.default_rng(1)
    g3, g4 = st.return_moments(rng.standard_normal(200_000))
    assert abs(g3) < 0.03 and abs(g4 - 3.0) < 0.05
    g3t, g4t = st.return_moments(rng.standard_t(5, 200_000))
    assert g4t > 4.0  # fat tails -> Pearson kurtosis well above 3


# ---------------------------------------------------------------------------- PSR / DSR
def test_psr_matches_normal_formula_and_monotonicity() -> None:
    sr, n = 0.1, 500
    expected = sps.norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr**2))
    assert st.probabilistic_sharpe(sr, n) == pytest.approx(expected)
    # PSR at the benchmark is exactly 0.5
    assert st.probabilistic_sharpe(0.07, 300, -0.5, 6, sr_star=0.07) == pytest.approx(0.5)
    # increasing in SR and in n (for SR > SR*)
    srs = np.linspace(-0.1, 0.3, 21)
    vals = [st.probabilistic_sharpe(s, 250) for s in srs]
    assert np.all(np.diff(vals) > 0)
    ns = [20, 50, 100, 500, 2000]
    vals_n = [st.probabilistic_sharpe(0.05, n) for n in ns]
    assert np.all(np.diff(vals_n) > 0)
    # negative skew and fat tails reduce confidence in a positive SR
    assert st.probabilistic_sharpe(0.1, 500, -2.0, 10.0) < st.probabilistic_sharpe(0.1, 500)
    # limits
    assert st.probabilistic_sharpe(0.2, 100_000) > 0.999999
    assert st.probabilistic_sharpe(-0.2, 100_000) < 1e-6


def test_expected_max_sharpe_properties() -> None:
    assert st.expected_max_sharpe(1, 0.5) == 0.0
    assert st.expected_max_sharpe(1, 0.5, mean=0.2) == 0.2
    vals = [st.expected_max_sharpe(n, 1.0) for n in (2, 5, 10, 100, 1000, 10_000)]
    assert np.all(np.diff(vals) > 0)
    # scales with the standard deviation of the trials
    assert st.expected_max_sharpe(50, 4.0) == pytest.approx(2 * st.expected_max_sharpe(50, 1.0))
    # close to the Monte-Carlo expected maximum of N standard normals
    rng = np.random.default_rng(2)
    mc = rng.standard_normal((20_000, 100)).max(axis=1).mean()
    assert st.expected_max_sharpe(100, 1.0) == pytest.approx(mc, rel=0.03)
    with pytest.raises(ValueError):
        st.expected_max_sharpe(10, -1.0)


def test_dsr_reproduces_bailey_lopez_de_prado_2014_example() -> None:
    """Numerical example of "The Deflated Sharpe Ratio" (JPM 2014).

    Annualised SR 2.5 over T = 1250 daily observations (5 years, 250 days/year), skewness
    -3, kurtosis 10, N = 100 independent trials with annualised Sharpe variance 1/2. The
    paper reports SR0 ~ 0.1132 (daily) and DSR ~ 0.9004.
    """
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250
    sr0 = st.expected_max_sharpe(100, var)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    dsr = st.deflated_sharpe(sr, 1250, -3.0, 10.0, n_trials=100, trial_var=var)
    assert dsr == pytest.approx(0.9004, abs=1e-3)


def test_dsr_monotonicity_and_limits() -> None:
    sr, n = 0.08, 1000
    psr = st.probabilistic_sharpe(sr, n)
    assert st.deflated_sharpe(sr, n, n_trials=1) == pytest.approx(psr)
    dsrs = [st.deflated_sharpe(sr, n, n_trials=k) for k in (1, 2, 10, 100, 1000)]
    assert np.all(np.diff(dsrs) < 0)  # more trials -> more deflation
    assert dsrs[0] > 0.99 and dsrs[-1] < dsrs[0]
    # trial_srs path equals explicit (N, V)
    rng = np.random.default_rng(3)
    trials = rng.normal(0, 0.03, 40)
    a = st.deflated_sharpe(sr, n, 0.1, 4.0, trials)
    b = st.deflated_sharpe(sr, n, 0.1, 4.0, n_trials=40, trial_var=float(trials.var(ddof=1)))
    assert a == pytest.approx(b)
    # larger trial dispersion -> more deflation
    assert st.deflated_sharpe(sr, n, n_trials=50, trial_var=1e-4) > st.deflated_sharpe(
        sr, n, n_trials=50, trial_var=1e-2)


def test_min_track_record_length_consistency() -> None:
    sr, g3, g4 = 0.1, -0.5, 5.0
    mtrl = st.min_track_record_length(sr, g3, g4, prob=0.95)
    # at exactly MinTRL observations the PSR equals the target probability
    assert st.probabilistic_sharpe(sr, mtrl, g3, g4) == pytest.approx(0.95, abs=1e-9)
    assert st.min_track_record_length(0.2) < st.min_track_record_length(0.1)
    assert st.min_track_record_length(0.1, prob=0.99) > st.min_track_record_length(0.1, prob=0.9)
    assert math.isinf(st.min_track_record_length(0.0))
    assert math.isinf(st.min_track_record_length(0.05, sr_star=0.1))


# ---------------------------------------------------------------------------- haircuts
def test_adjust_pvalues_known_values() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(st.adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
    np.testing.assert_allclose(st.adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(st.adjust_pvalues(p, "bh"), [0.02, 0.04, 0.04, 0.02])
    c = 1 + 1 / 2 + 1 / 3 + 1 / 4
    np.testing.assert_allclose(st.adjust_pvalues(p, "bhy"), np.minimum(np.array([0.02, 0.04, 0.04, 0.02]) * c, 1))
    assert np.all(st.adjust_pvalues(p, "sidak") <= st.adjust_pvalues(p, "bonferroni"))
    with pytest.raises(ValueError):
        st.adjust_pvalues(p, "nope")


def test_haircut_sharpe() -> None:
    one = st.haircut_sharpe(1.5, 252 * 10, 1)
    assert one["sr_haircut"] == pytest.approx(1.5, rel=1e-6) and abs(one["haircut"]) < 1e-6
    hs = [st.haircut_sharpe(1.5, 252 * 10, m)["sr_haircut"] for m in (1, 10, 100, 1000)]
    assert np.all(np.diff(hs) < 0) and hs[-1] > 0
    bhy = st.haircut_sharpe(1.5, 252 * 10, 100, method="bhy")["sr_haircut"]
    sidak = st.haircut_sharpe(1.5, 252 * 10, 100, method="sidak")["sr_haircut"]
    assert bhy < hs[2] <= sidak + 1e-12
    weak = st.haircut_sharpe(0.3, 252, 50)
    assert weak["sr_haircut"] == 0.0 and weak["haircut"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------- PBO / CSCV
def test_pbo_pure_noise_is_around_one_half() -> None:
    vals = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        m = rng.standard_normal((800, 20)) * 0.01
        vals.append(st.pbo_cscv(m, n_splits=10).pbo)
    assert 0.3 < float(np.mean(vals)) < 0.7


def test_pbo_near_zero_when_one_configuration_has_real_edge() -> None:
    rng = np.random.default_rng(7)
    m = rng.standard_normal((1000, 30)) * 0.01
    m[:, 11] += 0.004  # per-period Sharpe 0.4 vs 0 for the others
    res = st.pbo_cscv(m, n_splits=16)
    assert res.n_combinations == math.comb(16, 8)
    assert res.pbo < 0.05
    assert np.mean(res.selected == 11) > 0.95
    assert res.prob_oos_loss < 0.05
    d = res.to_dict()
    assert set(d) >= {"pbo", "prob_oos_loss", "degradation_slope", "n_combinations"}


def test_pbo_callable_metric_and_sampling_match_fast_path() -> None:
    rng = np.random.default_rng(8)
    m = rng.standard_normal((240, 6)) * 0.01
    fast = st.pbo_cscv(m, n_splits=6, metric="mean")
    slow = st.pbo_cscv(m, n_splits=6, metric=lambda x: x.mean(axis=0))
    assert fast.pbo == pytest.approx(slow.pbo)
    np.testing.assert_allclose(fast.logits, slow.logits)
    sampled = st.pbo_cscv(m, n_splits=6, max_combinations=10, seed=1)
    assert sampled.n_combinations == 10
    with pytest.raises(ValueError):
        st.pbo_cscv(m, n_splits=5)
    with pytest.raises(ValueError):
        st.pbo_cscv(m[:, :1], n_splits=4)


# ---------------------------------------------------------------------------- bootstrap
def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_optimal_block_length_iid_vs_persistent() -> None:
    iid = st.optimal_block_length(np.random.default_rng(4).standard_normal(2000))
    assert iid["stationary"] <= 2.0
    persistent = st.optimal_block_length(_ar1(2000, 0.7, 5))
    # theory for AR(1): b_SB = (2 phi / (1 - phi^2))^(2/3) n^(1/3) ~ 24.7 for phi=0.7, n=2000
    assert 10 < persistent["stationary"] < 45
    assert persistent["circular"] > persistent["stationary"]
    assert st.optimal_block_length(_ar1(2000, 0.9, 6))["stationary"] > persistent["stationary"]
    assert st.optimal_block_length([1.0, 2.0])["stationary"] == 1.0


def test_stationary_bootstrap_indices_block_structure() -> None:
    rng = np.random.default_rng(9)
    idx = st.stationary_bootstrap_indices(1000, 50, 10.0, rng)
    assert idx.shape == (50, 1000) and idx.min() >= 0 and idx.max() < 1000
    cont = (np.diff(idx, axis=1) % 1000) == 1  # continuation of a block (with wrap-around)
    mean_len = 1.0 / (1.0 - cont.mean())
    assert 8.0 < mean_len < 12.0  # geometric blocks with mean 10
    iid = st.stationary_bootstrap_indices(1000, 20, 1.0, rng)
    assert ((np.diff(iid, axis=1) % 1000) == 1).mean() < 0.01


def test_stationary_bootstrap_ci_deterministic_and_covers() -> None:
    rng = np.random.default_rng(10)
    r = rng.normal(0.0005, 0.01, 1500)
    a = st.stationary_bootstrap(r, np.mean, n_boot=500, seed=3)
    b = st.stationary_bootstrap(r, np.mean, n_boot=500, seed=3)
    assert a == b
    assert a["lower"] < a["estimate"] < a["upper"]
    se_theory = r.std(ddof=1) / math.sqrt(r.size)
    assert a["std_error"] == pytest.approx(se_theory, rel=0.2)
    v = st.stationary_bootstrap(r, lambda x: x.mean(axis=1), n_boot=500, seed=3, vectorized=True)
    assert v["estimate"] == pytest.approx(a["estimate"])
    basic = st.stationary_bootstrap(r, np.mean, n_boot=500, seed=3, ci_method="basic")
    assert basic["upper"] - basic["lower"] == pytest.approx(a["upper"] - a["lower"])


def test_bootstrap_widens_ci_under_autocorrelation() -> None:
    x = _ar1(1500, 0.8, 11) * 0.01 + 0.001
    blocky = st.stationary_bootstrap(x, np.mean, n_boot=400, seed=0)
    iid = st.stationary_bootstrap(x, np.mean, n_boot=400, seed=0, mean_block=1.0)
    assert blocky["mean_block"] > 5
    assert (blocky["upper"] - blocky["lower"]) > 1.8 * (iid["upper"] - iid["lower"])


def test_sharpe_ci_bootstrap_and_analytic_agree_for_iid() -> None:
    rng = np.random.default_rng(12)
    r = rng.normal(0.0006, 0.01, 1260)
    boot = st.sharpe_ci(r, 252, n_boot=1000, seed=0)
    ana = st.sharpe_ci(r, 252, method="analytic")
    assert boot["estimate"] == pytest.approx(st.sharpe(r, 252))
    assert ana["estimate"] == pytest.approx(boot["estimate"])
    assert boot["std_error"] == pytest.approx(ana["std_error"], rel=0.2)
    assert boot["lower"] < boot["estimate"] < boot["upper"]
    assert boot["method"] == "bootstrap" and ana["method"] == "analytic"


def test_sharpe_summary_on_synthetic_bars() -> None:
    bars = make_synthetic_bars(3000, "H1", seed=1)
    daily = bars["close"].resample("1D").last().dropna().pct_change().dropna()
    out = st.sharpe_summary(daily, 252, n_trials=20, n_boot=300, seed=0)
    for key in ("sharpe", "psr", "dsr", "sr0", "min_trl", "ci_lower", "ci_upper", "skew",
                "kurtosis", "sharpe_se", "bootstrap_block"):
        assert key in out
    assert out["dsr"] <= out["psr"] + 1e-12  # deflation never helps
    assert out["ci_lower"] < out["sharpe"] < out["ci_upper"]
    assert out["sr0"] > 0
    short = st.sharpe_summary([0.01, -0.01], 252)
    assert "sharpe" not in short


# ---------------------------------------------------------------------------- review: adversarial
def test_excess_kurtosis_is_flagged() -> None:
    """Pearson kurtosis >= 1 + skew^2 always. Passing EXCESS kurtosis (normal = 0, as
    aurum.backtest.metrics reports it) silently overstated PSR/DSR; it now warns."""
    import warnings

    with pytest.warns(RuntimeWarning, match="EXCESS kurtosis"):
        st.probabilistic_sharpe(0.1, 500, 0.0, 0.0)
    with pytest.warns(RuntimeWarning):
        st.deflated_sharpe(0.1, 500, -0.5, 0.5, n_trials=10)
    with pytest.warns(RuntimeWarning):
        st.min_track_record_length(0.1, 0.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        st.probabilistic_sharpe(0.1, 500, 0.0, 3.0)
        st.probabilistic_sharpe(0.1, 500, -3.0, 10.0)  # boundary kurt == 1 + skew^2 is legal
        g3, g4 = st.return_moments(np.random.default_rng(0).standard_t(4, 300))
        st.probabilistic_sharpe(0.1, 300, g3, g4)
    # the understated variance is material: PSR with excess (wrong) > PSR with Pearson (right)
    with pytest.warns(RuntimeWarning):
        wrong = st.probabilistic_sharpe(0.3, 60, 0.0, 0.0)
    assert wrong > st.probabilistic_sharpe(0.3, 60, 0.0, 3.0)


def test_haircut_nan_sharpe_propagates_nan() -> None:
    """Regression: a NaN Sharpe used to come back as a 0.0 haircut Sharpe (looks like data)."""
    out = st.haircut_sharpe(float("nan"), 500, 10)
    assert math.isnan(out["sr_haircut"]) and math.isnan(out["haircut"])
    assert math.isnan(out["p_adjusted"])


def test_pbo_sign_convention_deterministic() -> None:
    """Hand-built 2-block example: the IS winner is always the OOS loser -> PBO = 1; the IS
    winner is always the OOS winner -> PBO = 0 (checks rank direction and logit sign)."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((200, 3)) * 1e-4
    flip = noise.copy()
    flip[:100, 0] += 0.01   # A best in first half, worst in second
    flip[:100, 1] -= 0.01
    flip[100:, 0] -= 0.01
    flip[100:, 1] += 0.01   # B best in second half, worst in first
    res = st.pbo_cscv(flip, n_splits=2, metric="mean")
    assert res.n_combinations == 2
    assert res.pbo == 1.0 and np.all(res.logits < 0)
    assert res.prob_oos_loss == 1.0
    stable = noise.copy()
    stable[:, 2] += 0.01    # C best everywhere
    res2 = st.pbo_cscv(stable, n_splits=2, metric="mean")
    assert res2.pbo == 0.0 and np.all(res2.selected == 2) and np.all(res2.logits > 0)
    assert float(res2) == 0.0  # PBOResult usable as a scalar


def test_pbo_sampled_combinations_are_distinct_and_deterministic() -> None:
    rng = np.random.default_rng(5)
    m = rng.standard_normal((320, 6)) * 0.01
    total = math.comb(8, 4)
    a = st.pbo_cscv(m, n_splits=8, max_combinations=total - 1, seed=3)
    b = st.pbo_cscv(m, n_splits=8, max_combinations=total - 1, seed=3)
    assert a.n_combinations == total - 1
    assert a.pbo == b.pbo and np.array_equal(a.selected, b.selected)
    # distinct: the (IS perf, OOS perf) pairs of all sampled combinations are unique
    pairs = {(round(float(x), 12), round(float(y), 12)) for x, y in zip(a.is_perf, a.oos_perf, strict=True)}
    assert len(pairs) == total - 1
    with pytest.raises(ValueError):
        st.pbo_cscv(m, n_splits=8, max_combinations=0)


def test_pbo_nonfinite_inputs_do_not_crash() -> None:
    rng = np.random.default_rng(6)
    m = rng.standard_normal((200, 4)) * 0.01
    m[5, 1] = np.nan
    m[7, 2] = np.inf
    res = st.pbo_cscv(m, n_splits=4)
    assert 0.0 <= res.pbo <= 1.0 and np.isfinite(res.logits).all()


def test_bootstrap_block_longer_than_sample_is_clipped() -> None:
    r = np.random.default_rng(1).normal(0, 0.01, 50)
    out = st.stationary_bootstrap(r, np.mean, n_boot=50, mean_block=500.0, seed=0)
    assert out["mean_block"] == 50.0 and np.isfinite(out["lower"])
    with pytest.raises(ValueError):
        st.stationary_bootstrap([0.01], np.mean)
