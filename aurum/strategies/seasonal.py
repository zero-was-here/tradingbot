"""Seasonal strategy: ``intraday_seasonality`` (learned hour-of-week drift, shrunk, cost-aware).

Economic rationale
------------------
Gold's order flow has a strong clock: Asian physical buying (Shanghai/Hong Kong/India),
the London open and the LBMA auctions (10:30 and 15:00 London), the COMEX open and the
08:30 ET US data window, the 17:00 ET rollover, and pre-weekend de-risking on Friday
afternoons. Recurring, predictable liquidity demand at the same time of day produces
periodic return patterns: Heston, Korajczyk & Sadka (2010) show that returns at a given
half-hour predict returns at the same half-hour on subsequent days for weeks, driven by
systematic institutional trading and liquidity provision; Cai, Cheung & Wong (2001)
document pronounced intraday periodicity in COMEX gold. Such patterns are small relative
to noise, so the estimates MUST be shrunk hard to avoid fitting noise — and they are small
relative to a retail CFD's spread, so the positions MUST be chosen net of costs.

Estimator
---------
``fit`` (TRAIN data only) computes, for every decision bar ``t``, the vol-normalised
return of the bar it will hold, ``y_t = ln(C_{t+1}/O_{t+1}) / sigma_t`` (open-to-close of
bar ``t+1``: the part of the next bar's move a position decided at the close of ``t`` and
filled at the next open actually earns), and groups it by the local hour-of-week bucket of
the decision time ``available_at[t]`` (DST-aware ``zoneinfo`` clock, New York by default).
Bucket means are shrunk towards the precision-weighted mean ``mu`` with an empirical-Bayes
(James–Stein / normal-normal random-effects) estimator: ``post_b = mu + B_b (m_b - mu)``,
``B_b = tau^2 / (tau^2 + se_b^2)``, where ``se_b^2 = var_b / n_b`` uses each bucket's OWN
variance. Gold's hourly volatility differs several-fold across the week (Asia vs the
London/NY opens and US data), so a pooled variance would make volatile hours look like mean
effects. The between-bucket variance ``tau^2`` is the DerSimonian & Laird (1986) moment
estimator on inverse-variance weights, floored at 0. Because that estimate is positive
about half the time even without any seasonality, a homogeneity pre-test guards it:
unless Welch's (1951) heteroskedastic one-way ANOVA rejects "all bucket means equal" at
level ``significance``, the table is flat. (A pooled-variance Cochran Q rejected 15% of the
time at a nominal 5% on real 2012-15 gold labels under a sign-flip null; Welch rejected
6.5%.) With ``demean=False`` the null is "all means zero" and ``Q = sum m_b^2 / se_b^2 ~
chi2(k)``. A fixed ``shrinkage`` (prior strength in observations, ``B_b = n_b / (n_b +
lambda)``) is also supported (the pre-test still applies unless ``significance=None``).

Cost-aware positions (``cost_aware=True``, default)
----------------------------------------------------
The shrunk effects ``alpha_b`` (vol units per bar) flip sign from one hour to the next,
and a bar's edge is typically a few hundredths of a sigma while a retail gold CFD costs
~0.1 sigma PER SIDE on H1 (half-spread + slippage). Trading the raw table therefore turns
the book over thousands of times a year and pays far more in costs than it can earn (on
2016-19 gold, fitted on 2012-15, frictionless table: ~2,550 trades/yr and ~$46k of costs on
$100k for a gross daily Sharpe of ~0.5 — net Sharpe about -7).

``fit`` therefore chooses the positions by solving, on the TRAINING estimates, the periodic
mean-variance problem with proportional transaction costs over the weekly cycle of buckets
(dynamic programming on a position grid, average-reward relative value iteration)::

    max  sum_b [ f_b alpha_b - (gamma / 2) f_b^2 - lambda kappa_b |f_b - f_{b-1}| ],  |f| <= 1

* ``kappa_b`` — expected cost of trading one unit of forecast at bucket ``b``, in the same
  vol units as ``alpha_b``: ``(half effective spread + slippage + commission) / price /
  sigma`` of the EXECUTION bar, averaged over the bucket's TRAINING rows with the ``costs``
  model (``aurum.execution.costs.CostModel``, default parameters unless given). Sizing is
  vol-targeted, so edge and cost scale identically with the position: the comparison is
  sizing-invariant.
* ``gamma`` — set so that WITHOUT costs the solution ``f_b = alpha_b / gamma`` is exactly
  the classic table (Carver-scaled to an average |forecast| of 0.5, capped at 1): with zero
  costs (or ``cost_aware=False``) the strategy is unchanged.
* ``lambda`` = ``cost_multiplier`` (default 2): the margin by which the expected gross edge
  of a trade must exceed its expected round-trip cost (``lambda = 2`` <=> a 50% haircut on
  the in-sample edge; McLean & Pontiff (2016) find published anomaly returns ~58% lower
  post-publication, and spreads widen around the data releases seasonal effects cluster on).

What the solution looks like (Constantinides 1986; Davis & Norman 1990; Garleanu &
Pedersen 2013 for the discrete-time analogue): a no-trade band around the frictionless
position. A position CHANGE is made only if the edge it adds over the run of buckets it is
held for exceeds ``lambda`` x its cost — so (a) a lone bucket gets a round trip only if
``|alpha_b| > lambda (kappa_b + kappa_{b+1})``, i.e. it clears the round-trip cost per bar
by the margin (``fit_summary_["n_buckets_single_bar_feasible"]``); (b) adjacent same-sign
buckets are held as ONE position (minimum holding), and small opposite-sign buckets inside
a run are held through; (c) the implied dead zone is scaled to costs
(``dead_zone="auto"``): a lone bucket's forecast is soft-thresholded by ``lambda x round
trip`` (``fit_summary_["dead_zone_single_bar"]``, in forecast units). Because a small
position costs little risk (quadratic) but a lot to flatten and rebuild (linear), the band
around zero can carry a small position through edge-free stretches instead of flattening
it. Turnover is bounded by construction: along the fitted weekly orbit, ``lambda x`` the
expected costs never exceed the risk-adjusted expected gross edge (``fit_summary_`` reports
turnover, gross and cost per week, and the frictionless turnover for comparison).

The fitted positions form the steady-state weekly orbit of the optimal policy from a flat
start, stored as a table keyed by bucket: ``generate`` is still a pure function of the bar
timestamps and the fitted table (no price input), so it is trivially point-in-time and
live/backtest parity is exact. A bucket not seen in training (a holiday-shifted session)
holds the position of the preceding bucket of the cycle (the optimal action at zero edge
when trading costs something). Financing (swap) is NOT modelled in the fit.

References
----------
* Heston, S., Korajczyk, R. & Sadka, R. (2010). "Intraday Patterns in the Cross-section
  of Stock Returns". J. Finance 65(4), 1369-1407.
* Cai, J., Cheung, Y.-L. & Wong, M. (2001). "What moves the gold market?". J. Futures
  Markets 21(3), 257-278.
* Efron, B. & Morris, C. (1975). "Data Analysis Using Stein's Estimator and its
  Generalizations". JASA 70(350).
* DerSimonian, R. & Laird, N. (1986). "Meta-analysis in clinical trials". Controlled
  Clinical Trials 7(3) — method-of-moments between-group variance.
* Welch, B. L. (1951). "On the Comparison of Several Mean Values: An Alternative
  Approach". Biometrika 38(3/4), 330-336 — heteroskedastic one-way ANOVA.
* Constantinides, G. (1986). "Capital Market Equilibrium with Transaction Costs". JPE 94(4);
  Davis, M. & Norman, A. (1990). "Portfolio Selection with Transaction Costs". Math. OR
  15(4) — no-trade regions under proportional costs.
* Garleanu, N. & Pedersen, L. H. (2013). "Dynamic Trading with Predictable Returns and
  Transaction Costs". J. Finance 68(6).
* McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return
  Predictability?". J. Finance 71(1).
* Carver, R. (2015). *Systematic Trading* — forecast scaling, trading-cost "speed limits".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats as sps

from aurum.core.instrument import XAUUSD
from aurum.core.types import MarketData
from aurum.features.volatility import safe_div
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import TARGET_ABS_FORECAST, bar_volatility, check_params, log_close

logger = logging.getLogger(__name__)

__all__ = ["IntradaySeasonality", "bucket_keys", "periodic_cost_aware_positions"]

#: Position grid of the cost-aware dynamic programme (step 1/40 = 0.025 of a full forecast).
_GRID = np.linspace(-1.0, 1.0, 81)
_MAX_SWEEPS = 400


def bucket_keys(times: pd.DatetimeIndex | pd.Series, tz: str, bucket_minutes: int) -> np.ndarray:
    """Local hour-of-week bucket id: ``weekday * 1440 + floor(minute_of_day / b) * b``.

    ``tz_convert`` with a ``ZoneInfo`` applies the zone's historical DST rules, so a bucket
    always refers to the same LOCAL wall-clock time (e.g. 08:00 New York is 12:00 UTC in
    summer and 13:00 UTC in winter).
    """
    loc = pd.DatetimeIndex(times).tz_convert(ZoneInfo(tz))
    minute = (loc.hour * 60 + loc.minute).to_numpy()
    b = int(bucket_minutes)
    return np.asarray(loc.weekday).astype(np.int64) * 1440 + (minute // b) * b


def periodic_cost_aware_positions(alpha: np.ndarray, kappa: np.ndarray, gamma: float,
                                  grid: np.ndarray = _GRID) -> tuple[np.ndarray, dict[str, Any]]:
    """Steady-state positions of ``max sum_b f_b a_b - gamma/2 f_b^2 - k_b |f_b - f_{b-1}|``
    over a CYCLE of slots (``b = 0..L-1``, slot ``L`` = slot 0), ``f`` on ``grid``.

    Average-reward relative value iteration: one sweep runs the Bellman recursion backwards
    once around the cycle; it stops when the policy of a full sweep repeats. The returned
    positions are the orbit of that policy from a FLAT start at slot 0, iterated until the
    weekly orbit repeats (a unique orbit unless costs make several positions equally good,
    in which case flat-start is the conservative choice). Ties prefer the smaller |f|.
    """
    a = np.asarray(alpha, dtype=float)
    k = np.asarray(kappa, dtype=float)
    g = np.asarray(grid, dtype=float)
    n_slots, n_grid = len(a), len(g)
    i0 = int(np.argmin(np.abs(g)))
    if n_slots == 0:
        return np.zeros(0), {"sweeps": 0, "converged": True, "orbit_cycles": 0}
    move = np.abs(g[None, :] - g[:, None])                    # [prev i, new j]
    tiny = 1e-12 * (np.abs(g) + 1e-3 * move)                  # tie-break: small |f|, then stay
    reward = a[:, None] * g[None, :] - 0.5 * gamma * g[None, :] ** 2   # [slot, j]
    v_next = np.zeros(n_grid)
    policy = np.zeros((n_slots, n_grid), dtype=np.int64)
    prev_policy = None
    sweeps, converged = 0, False
    while sweeps < _MAX_SWEEPS and not converged:
        sweeps += 1
        for b in range(n_slots - 1, -1, -1):
            q = reward[b][None, :] - k[b] * move - tiny + v_next[None, :]
            policy[b] = np.argmax(q, axis=1)
            v_next = q[np.arange(n_grid), policy[b]]
        v_next = v_next - v_next[i0]                          # relative values stay bounded
        converged = prev_policy is not None and np.array_equal(policy, prev_policy)
        prev_policy = policy.copy()
    # orbit from flat at slot 0, iterated until one week repeats the previous one
    i = i0
    orbit_prev = None
    orbit = np.zeros(n_slots, dtype=np.int64)
    cycles, repeated = 0, False
    while cycles < 50 and not repeated:
        cycles += 1
        for b in range(n_slots):
            i = int(policy[b, i])
            orbit[b] = i
        repeated = orbit_prev is not None and np.array_equal(orbit, orbit_prev)
        orbit_prev = orbit.copy()
    return g[orbit], {"sweeps": int(sweeps), "converged": bool(converged), "orbit_cycles": int(cycles),
                      "orbit_periodic": bool(repeated)}


def _cost_model(spec: Any) -> Any:
    from aurum.execution.costs import CostModel

    if spec is None:
        return CostModel()
    if isinstance(spec, CostModel):
        return spec
    if isinstance(spec, Mapping):
        return CostModel(**dict(spec))
    raise ValueError("costs must be None, a CostModel or a mapping of CostModel parameters")


@register_strategy
class IntradaySeasonality(Strategy):
    """Learned hour-of-week mean returns, empirical-Bayes shrunk, traded net of costs.

    Rationale: recurring time-of-day liquidity demand (Asian physical flows, London fixes,
    COMEX open / US data, rollover, weekend de-risking) produces periodic drift in gold
    returns; Heston, Korajczyk & Sadka (2010) show such intraday periodicity is persistent.
    The effects are tiny relative to noise, so bucket means are shrunk with a James–Stein /
    empirical-Bayes estimator (Efron & Morris 1975) and vanish when the data show no
    between-bucket dispersion beyond sampling error; they are also tiny relative to costs,
    so positions solve a mean-variance problem with proportional costs over the weekly
    cycle (no-trade bands; hold across adjacent buckets). See the module docstring.

    Parameters: ``tz`` (clock for the buckets, default America/New_York — the broker day
    and US data are on NY time), ``bucket_minutes`` (60), ``shrinkage``
    (``"empirical_bayes"`` or a prior strength in observations), ``demean`` (remove the
    training-period average drift so the rule is a pure timing signal, default True),
    ``min_obs`` (buckets seen fewer times in training forecast 0 / are held through),
    ``significance`` (level of the Welch heterogeneity pre-test; ``None`` disables it),
    ``cost_aware`` (default True: cost-aware positions; False = the frictionless table),
    ``costs`` (``CostModel`` or its kwargs used to estimate costs from the TRAINING bars'
    spreads and ranges; default ``CostModel()``), ``cost_multiplier`` (margin ``lambda``
    of edge over cost, default 2), ``dead_zone`` (``"auto"`` = the cost-implied soft
    threshold only; a number in [0, 1) additionally zeroes |forecast| below it, the only
    turnover control when ``cost_aware=False``) and the volatility normaliser (EWMA
    half-life 240 H1 bars).

    References: Heston, Korajczyk & Sadka (2010) JF 65(4); Cai, Cheung & Wong (2001) JFM
    21(3); Efron & Morris (1975) JASA 70; Garleanu & Pedersen (2013) JF 68(6); McLean &
    Pontiff (2016) JF 71(1); Carver (2015).
    """

    name = "intraday_seasonality"
    description = ("Hour-of-week (New York clock) drift learned on training data with "
                   "empirical-Bayes shrinkage; positions chosen net of trading costs "
                   "(no-trade bands over the weekly cycle).")
    trainable = True

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "tz": "America/New_York",
            "bucket_minutes": 60,
            "shrinkage": "empirical_bayes",
            "demean": True,
            "min_obs": 30,
            "significance": 0.05,
            "cost_aware": True,
            "costs": None,
            "cost_multiplier": 2.0,
            "dead_zone": "auto",
            "vol_halflife": 240.0,
            "vol_min_periods": 120,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        ZoneInfo(str(p["tz"]))
        p["bucket_minutes"] = int(p["bucket_minutes"])
        if not 1 <= p["bucket_minutes"] <= 1440:
            raise ValueError("bucket_minutes must be in [1, 1440]")
        shrink = p["shrinkage"]
        if shrink != "empirical_bayes" and not (isinstance(shrink, (int, float)) and shrink >= 0):
            raise ValueError("shrinkage must be 'empirical_bayes' or a non-negative number")
        sig = p["significance"]
        if sig is not None and not 0.0 < float(sig) < 1.0:
            raise ValueError("significance must be in (0, 1) or None")
        dz = p["dead_zone"]
        if dz != "auto" and not (isinstance(dz, (int, float)) and 0.0 <= float(dz) < 1.0):
            raise ValueError("dead_zone must be 'auto' or a number in [0, 1)")
        if not float(p["cost_multiplier"]) >= 0.0:
            raise ValueError("cost_multiplier must be >= 0")
        _cost_model(p["costs"])
        self.table_: dict[int, float] = {}
        self.fit_summary_: dict[str, Any] = {}
        self._cycle_keys: np.ndarray = np.zeros(0, dtype=np.int64)
        self._cycle_pos: np.ndarray = np.zeros(0)

    @property
    def warmup_bars(self) -> int:
        return 0   # the forecast is a function of the clock only

    # ------------------------------------------------------------------------------------
    def training_targets(self, md: MarketData) -> pd.DataFrame:
        """Rows ``(key, y, cost)`` used by :meth:`fit`: bucket of ``available_at[t]``, the
        vol-normalised open-to-close return of bar ``t+1`` (a TRAINING label) and the cost
        of trading one unit of forecast at ``t`` (filled at the open of ``t+1``) in the same
        vol units: ``(half effective spread + slippage + commission per oz) / close_t /
        sigma_t`` with the spread and range of bar ``t+1``."""
        p = self.params
        bars = md.bars
        lc = log_close(bars)
        sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
        oc = lc - np.log(bars["open"].to_numpy(dtype=float))
        y = pd.Series(safe_div(oc.shift(-1), sigma), index=bars.index)  # label: needs bar t+1
        keys = bucket_keys(bars["available_at"], p["tz"], p["bucket_minutes"])
        cm = _cost_model(p["costs"])
        spread = bars["spread"].to_numpy(dtype=float) if "spread" in bars else np.zeros(len(bars))
        spread = np.where(np.isfinite(spread) & (spread >= 0), spread, 0.0)
        eff = np.maximum(spread * cm.spread_multiplier, cm.min_spread)
        rng = (bars["high"].to_numpy(dtype=float) - bars["low"].to_numpy(dtype=float))
        rng = np.where(np.isfinite(rng) & (rng >= 0), rng, 0.0)
        comm = cm.commission_per_lot if cm.commission_per_lot is not None else XAUUSD.commission_per_lot
        per_side = 0.5 * eff + cm.slippage_fixed + cm.slippage_range_frac * rng
        per_side_next = pd.Series(per_side, index=bars.index).shift(-1) + float(comm) / XAUUSD.contract_size
        cost = safe_div(per_side_next / bars["close"].astype(float), sigma)
        frame = pd.DataFrame({"key": keys, "y": y.to_numpy(), "cost": np.asarray(cost, dtype=float)},
                             index=bars.index)
        return frame[np.isfinite(frame["y"].to_numpy())]

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> IntradaySeasonality:
        p = self.params
        data = self.training_targets(md)
        self.table_ = {}
        self._cycle_keys = np.zeros(0, dtype=np.int64)
        self._cycle_pos = np.zeros(0)
        if len(data) < 2:
            logger.warning("intraday_seasonality: not enough training rows (%d); flat table", len(data))
            self.fit_summary_ = {"n_obs": int(len(data)), "tau2": 0.0, "scalar": 0.0}
            self.is_fitted = True
            return self
        g = data.groupby("key")["y"]
        stats = pd.DataFrame({"n": g.size(), "mean": g.mean(), "var": g.var(ddof=1)})
        if "cost" in data.columns:
            c = data["cost"].where(np.isfinite(data["cost"]))
            stats["cost"] = c.groupby(data["key"]).mean()
        else:  # a custom label table without costs: frictionless
            stats["cost"] = 0.0
        stats = stats[stats["n"] >= max(2, int(p["min_obs"]))]
        if stats.empty:
            logger.warning("intraday_seasonality: no bucket has >= %s observations", p["min_obs"])
            self.fit_summary_ = {"n_obs": int(len(data)), "tau2": 0.0, "scalar": 0.0}
            self.is_fitted = True
            return self
        n = stats["n"].to_numpy(dtype=float)
        m = stats["mean"].to_numpy(dtype=float)
        var = stats["var"].to_numpy(dtype=float)
        dof = (n - 1.0).sum()
        s2 = float(((n - 1.0) * np.nan_to_num(var)).sum() / dof) if dof > 0 else 1.0
        if not s2 > 0:
            s2 = 1.0
        # Per-bucket standard errors: gold's hourly volatility varies ~3x (p90/p10) across
        # the week, so a pooled variance makes volatile hours look like mean effects (the
        # pooled test rejected 15% of the time at a nominal 5% on real 2012-15 labels under
        # a sign-flip null). A degenerate (zero/NaN) variance falls back to the pooled one.
        var_b = np.where(np.isfinite(var) & (var > 0), var, s2)
        se2 = var_b / n
        w = 1.0 / se2
        k = len(m)
        grand_mean = float((n * m).sum() / n.sum())
        if p["demean"]:
            # H0: all bucket means equal. Welch (1951) heteroskedastic one-way ANOVA; the
            # centre is the precision-weighted mean (the random-effects prior mean).
            mu = float((w * m).sum() / w.sum())
            q_stat = float((w * (m - mu) ** 2).sum())
            if k >= 2:
                lam = float((((1.0 - w / w.sum()) ** 2) / (n - 1.0)).sum())
                f_stat = (q_stat / (k - 1)) / (1.0 + 2.0 * (k - 2) / (k * k - 1.0) * lam)
                p_value = float(sps.f.sf(f_stat, k - 1, (k * k - 1.0) / (3.0 * lam)))
                # DerSimonian & Laird (1986) moment estimator of the between-bucket variance
                tau2 = max(0.0, (q_stat - (k - 1)) / float(w.sum() - (w * w).sum() / w.sum()))
            else:
                p_value, tau2 = 1.0, 0.0
            test = "welch"
        else:
            # H0: every bucket mean is zero (known centre): Q ~ chi2(k)
            mu = 0.0
            q_stat = float((w * m * m).sum())
            p_value = float(sps.chi2.sf(q_stat, k))
            tau2 = max(0.0, (q_stat - k) / float(w.sum()))
            test = "chi2"
        heterogeneous = p["significance"] is None or p_value < float(p["significance"])
        if not heterogeneous:
            shrink = np.zeros(k)
        elif p["shrinkage"] == "empirical_bayes":
            # posterior mean mu + B_b (m_b - mu), B_b = tau2 / (tau2 + se_b^2): noisy
            # (volatile or rarely observed) buckets are shrunk harder
            shrink = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros(k)
        else:
            shrink = n / (n + float(p["shrinkage"]))
        effect = shrink * (m - mu)
        avg_abs = float((n * np.abs(effect)).sum() / n.sum())
        scalar = TARGET_ABS_FORECAST / avg_abs if avg_abs > 0 else 0.0
        frictionless = np.clip(effect * scalar, -1.0, 1.0)
        kappa = np.nan_to_num(stats["cost"].to_numpy(dtype=float), nan=0.0, posinf=0.0)
        kappa = np.maximum(kappa, 0.0)
        lam_c = float(p["cost_multiplier"])
        cost_aware = bool(p["cost_aware"]) and lam_c > 0.0 and scalar > 0.0 and bool(np.any(kappa > 0))
        dp_info: dict[str, Any] = {}
        if cost_aware:
            gamma = 1.0 / scalar
            fc, dp_info = periodic_cost_aware_positions(effect, lam_c * kappa, gamma)
        else:
            fc = frictionless
        dz = p["dead_zone"]
        if dz != "auto" and float(dz) > 0.0:
            fc = np.where(np.abs(fc) < float(dz), 0.0, fc)
        keys = stats.index.to_numpy(dtype=np.int64)
        self.table_ = {int(key): float(v) for key, v in zip(keys, fc, strict=True)}
        self._cycle_keys = keys
        self._cycle_pos = np.asarray(fc, dtype=float)
        # diagnostics along the fitted weekly orbit (in-sample expectations, vol units)
        dpos = np.abs(np.diff(np.r_[fc[-1], fc]))
        per_cycle = n.sum() / max(k, 1)                   # training observations per slot
        rt_next = kappa + np.roll(kappa, -1)
        self.fit_summary_ = {
            "n_obs": int(len(data)),
            "n_buckets": int(k),
            "grand_mean": grand_mean,
            "centre": mu,
            "pooled_var": s2,
            "tau2": float(tau2),
            "test": test,
            "cochran_q": q_stat,
            "q_pvalue": p_value,
            "heterogeneous": bool(heterogeneous),
            "mean_shrinkage": float(np.mean(shrink)),
            "shrinkage_by_key": {int(key): float(b) for key, b in zip(keys, shrink, strict=True)},
            "scalar": float(scalar),
            "cost_aware": bool(cost_aware),
            "cost_multiplier": lam_c,
            "cost_per_side_median": float(np.median(kappa)) if k else 0.0,
            "n_active": int(np.count_nonzero(fc)),
            "n_buckets_single_bar_feasible": int(np.count_nonzero(np.abs(effect) > lam_c * rt_next)),
            "dead_zone_single_bar": float(np.median(lam_c * rt_next) * scalar) if k else 0.0,
            "orbit_turnover_per_week": float(dpos.sum()),
            "orbit_expected_gross_per_week": float((fc * effect).sum()),
            "orbit_expected_cost_per_week": float((kappa * dpos).sum()),
            "frictionless_turnover_per_week": float(np.abs(np.diff(np.r_[frictionless[-1], frictionless])).sum()),
            "obs_per_slot": float(per_cycle),
            "dp": dp_info,
            "train_start": str(md.bars.index[0]) if len(md.bars) else None,
            "train_end": str(md.bars.index[-1]) if len(md.bars) else None,
        }
        logger.info("intraday_seasonality fitted: %d buckets, tau2=%.3g, Q p-value=%.3g, mean B=%.3f, "
                    "cost-aware=%s, %d active buckets, turnover/week %.2f (frictionless %.2f)",
                    k, tau2, p_value, float(np.mean(shrink)), cost_aware, int(np.count_nonzero(fc)),
                    self.fit_summary_["orbit_turnover_per_week"],
                    self.fit_summary_["frictionless_turnover_per_week"])
        self.is_fitted = True
        return self

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("intraday_seasonality must be fit() on training data before generate()")
        p = self.params
        bars = md.bars
        keys = bucket_keys(bars["available_at"], p["tz"], p["bucket_minutes"])
        cyc_keys = getattr(self, "_cycle_keys", np.zeros(0, dtype=np.int64))
        if self.fit_summary_.get("cost_aware") and len(cyc_keys):
            # buckets unseen in training hold the preceding cycle bucket's position (no trade)
            j = np.searchsorted(cyc_keys, keys, side="right") - 1      # -1 wraps to the last
            fc = self._cycle_pos[j]
        else:
            fc = pd.Series(keys).map(self.table_).to_numpy(dtype=float)
        return self._finalize(pd.Series(fc, index=bars.index), bars.index)
