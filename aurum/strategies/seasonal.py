"""Seasonal strategy: ``intraday_seasonality`` (learned hour-of-week drift, shrunk).

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
to noise, so the estimates MUST be shrunk hard to avoid fitting noise.

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
lambda)``) is also supported (the pre-test still applies unless ``significance=None``). The
(demeaned) effects are then Carver-scaled to an average |forecast| of 0.5 on the training
distribution and capped. Note that this rescaling undoes the *overall* level of
shrinkage: shrinkage sets the relative sizes of the buckets, and the pre-test decides
whether there is a table at all.

``generate`` is a pure function of the bar timestamps and the fitted table (no price
input), so it is trivially point-in-time.

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
* Carver, R. (2015). *Systematic Trading* — forecast scaling.
"""

from __future__ import annotations

import logging
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats as sps

from aurum.core.types import MarketData
from aurum.features.volatility import safe_div
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import TARGET_ABS_FORECAST, bar_volatility, check_params, log_close

logger = logging.getLogger(__name__)

__all__ = ["IntradaySeasonality", "bucket_keys"]


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


@register_strategy
class IntradaySeasonality(Strategy):
    """Learned hour-of-week mean returns with empirical-Bayes shrinkage (trainable).

    Rationale: recurring time-of-day liquidity demand (Asian physical flows, London fixes,
    COMEX open / US data, rollover, weekend de-risking) produces periodic drift in gold
    returns; Heston, Korajczyk & Sadka (2010) show such intraday periodicity is persistent.
    The effects are tiny relative to noise, so bucket means are shrunk with a James–Stein /
    empirical-Bayes estimator (Efron & Morris 1975) and vanish when the data show no
    between-bucket dispersion beyond sampling error. See the module docstring for the
    estimator.

    Parameters: ``tz`` (clock for the buckets, default America/New_York — the broker day
    and US data are on NY time), ``bucket_minutes`` (60), ``shrinkage``
    (``"empirical_bayes"`` or a prior strength in observations), ``demean`` (remove the
    training-period average drift so the rule is a pure timing signal, default True),
    ``min_obs`` (buckets seen fewer times in training forecast 0), ``significance`` (level
    of the Welch heterogeneity pre-test; ``None`` disables it), ``dead_zone`` (buckets
    whose scaled |forecast| is below it are set to 0 — a turnover control, default 0 = off)
    and the volatility normaliser (EWMA half-life 240 H1 bars).

    Costs: an hour-of-week table changes sign from one hour to the next, so the standalone
    rule turns over its whole position many times a day; its per-bar edge is far below the
    round-trip cost of a retail gold CFD. It is meant as a (netted) input to the
    ``ForecastCombiner``; standalone use needs a ``dead_zone`` or a slower bucket.

    References: Heston, Korajczyk & Sadka (2010) JF 65(4); Cai, Cheung & Wong (2001) JFM
    21(3); Efron & Morris (1975) JASA 70; Carver (2015).
    """

    name = "intraday_seasonality"
    description = ("Hour-of-week (New York clock) drift learned on training data with "
                   "empirical-Bayes shrinkage; forecasts the bar about to be held.")
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
            "dead_zone": 0.0,
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
        if not 0.0 <= float(p["dead_zone"]) < 1.0:
            raise ValueError("dead_zone must be in [0, 1)")
        self.table_: dict[int, float] = {}
        self.fit_summary_: dict[str, Any] = {}

    @property
    def warmup_bars(self) -> int:
        return 0   # the forecast is a function of the clock only

    # ------------------------------------------------------------------------------------
    def training_targets(self, md: MarketData) -> pd.DataFrame:
        """Rows ``(key, y)`` used by :meth:`fit`: bucket of ``available_at[t]`` and the
        vol-normalised open-to-close return of bar ``t+1`` (a TRAINING label)."""
        p = self.params
        bars = md.bars
        lc = log_close(bars)
        sigma = bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"])
        oc = lc - np.log(bars["open"].to_numpy(dtype=float))
        y = pd.Series(safe_div(oc.shift(-1), sigma), index=bars.index)  # label: needs bar t+1
        keys = bucket_keys(bars["available_at"], p["tz"], p["bucket_minutes"])
        frame = pd.DataFrame({"key": keys, "y": y.to_numpy()}, index=bars.index)
        return frame[np.isfinite(frame["y"].to_numpy())]

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> IntradaySeasonality:
        p = self.params
        data = self.training_targets(md)
        self.table_ = {}
        if len(data) < 2:
            logger.warning("intraday_seasonality: not enough training rows (%d); flat table", len(data))
            self.fit_summary_ = {"n_obs": int(len(data)), "tau2": 0.0, "scalar": 0.0}
            self.is_fitted = True
            return self
        g = data.groupby("key")["y"]
        stats = pd.DataFrame({"n": g.size(), "mean": g.mean(), "var": g.var(ddof=1)})
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
        fc = np.clip(effect * scalar, -1.0, 1.0)
        fc = np.where(np.abs(fc) < float(p["dead_zone"]), 0.0, fc)
        self.table_ = {int(key): float(v) for key, v in zip(stats.index, fc, strict=True)}
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
            "shrinkage_by_key": {int(key): float(b) for key, b in zip(stats.index, shrink, strict=True)},
            "scalar": float(scalar),
            "train_start": str(md.bars.index[0]) if len(md.bars) else None,
            "train_end": str(md.bars.index[-1]) if len(md.bars) else None,
        }
        logger.info("intraday_seasonality fitted: %d buckets, tau2=%.3g, Q p-value=%.3g, mean B=%.3f",
                    k, tau2, p_value, float(np.mean(shrink)))
        self.is_fitted = True
        return self

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("intraday_seasonality must be fit() on training data before generate()")
        p = self.params
        bars = md.bars
        keys = bucket_keys(bars["available_at"], p["tz"], p["bucket_minutes"])
        fc = pd.Series(keys).map(self.table_).to_numpy(dtype=float)
        return self._finalize(pd.Series(fc, index=bars.index), bars.index)
