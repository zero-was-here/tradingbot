"""Volatility forecasting.

``ewma_volatility`` is the default causal forecaster used by the backtest engine and live
runner. Additional models (GARCH, HAR-RV, range-based blends) live in this module too and
must keep the same causal contract: value at row t uses returns up to and including t.

Models
------
* :func:`ewma_volatility` — RiskMetrics-style exponentially weighted variance
  (J.P. Morgan/Reuters, *RiskMetrics Technical Document*, 1996).
* :class:`Garch11` — GARCH(1,1) of Bollerslev (1986) fitted by Gaussian (quasi) maximum
  likelihood with the stationarity constraints ``omega > 0, alpha, beta >= 0,
  alpha + beta < 1``. The Gaussian likelihood is a QMLE: parameter estimates remain
  consistent under fat-tailed innovations (Bollerslev & Wooldridge, 1992).
* :class:`HarRV` / :func:`har_rv_forecast` — Heterogeneous AutoRegressive model of realised
  variance (Corsi, 2009, *J. Financial Econometrics* 7(2)), regressing next-day realised
  variance on daily, weekly and monthly averages of past realised variance. The expanding
  re-estimation in :func:`har_rv_forecast` only ever uses (regressor, target) pairs whose
  target was observed before the forecast origin.
* :func:`daily_realised_variance` — sum of squared intraday log returns per day, with the
  ``available_at`` of the last bar in the day (point-in-time, SPEC §0).
* :func:`blend_vol` — forecast combination in *variance* space.

Point-in-time notes
-------------------
Every ``forecast``/``predict`` method returns, at row ``t``, the forecast for the NEXT
period made with information up to and including ``t``. That is exactly the quantity the
sizer needs at the decision time ``available_at[t]`` (SPEC §1). Model parameters are
estimated with ``fit`` on TRAINING data only; applying a fitted model to later data is
causal because the recursions only run forward.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, signal

from aurum.core.timeframes import index_bar_minutes, infer_bars_per_year, nominal_bars_per_year

logger = logging.getLogger(__name__)

_LOG_2PI = math.log(2.0 * math.pi)


def ewma_volatility(
    close: pd.Series,
    *,
    halflife_bars: float = 48.0,
    bars_per_year: float | None = None,
    min_periods: int = 20,
    floor: float = 0.03,
    cap: float = 2.0,
) -> pd.Series:
    """Annualised EWMA volatility of log returns, causal (uses returns up to t).

    Returns a fraction (0.15 == 15% annualised), clipped to [floor, cap]. Warm-up rows get a
    conservative constant 0.20 (never back-filled from future estimates).
    """
    # Nominal, timeframe-based annualisation: never depends on how dense FUTURE data is.
    bpy = bars_per_year or nominal_bars_per_year(index_bar_minutes(close.index))
    r = np.log(close).diff()
    var = (r**2).ewm(halflife=halflife_bars, min_periods=min_periods, adjust=False).mean()
    vol = np.sqrt(var * bpy)
    vol = vol.clip(lower=floor, upper=cap)
    return vol.fillna(0.20).rename("vol_ann")


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _as_array(x: pd.Series | np.ndarray | Sequence[float]) -> tuple[np.ndarray, pd.Index | None]:
    """Return a float64 1-D array and the pandas index (if any)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("expected a single return series, got a multi-column DataFrame")
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, na_value=np.nan), x.index
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D return series")
    return arr, None


def _resolve_bpy(bars_per_year: float | None, index: pd.Index | None) -> float:
    if bars_per_year is not None:
        if bars_per_year <= 0:
            raise ValueError("bars_per_year must be positive")
        return float(bars_per_year)
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        return float(nominal_bars_per_year(index_bar_minutes(index)))
    raise ValueError("bars_per_year is required when returns carry no DatetimeIndex")


# ----------------------------------------------------------------------------------------
# GARCH(1,1)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GarchParams:
    """GARCH(1,1) parameters in the units of the fitted returns (per bar).

    ``r_t = mu + eps_t``, ``eps_t = sigma_t z_t``,
    ``sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2``.
    """

    mu: float
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        """``alpha + beta``: the rate at which variance shocks decay."""
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """Long-run variance ``omega / (1 - alpha - beta)`` (per bar)."""
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1.0 else math.inf

    @property
    def half_life(self) -> float:
        """Bars for a variance shock to decay by half: ``ln 0.5 / ln(alpha + beta)``."""
        p = self.persistence
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return math.inf
        return math.log(0.5) / math.log(p)

    def to_dict(self) -> dict[str, float]:
        return {
            "mu": self.mu,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.persistence,
            "unconditional_variance": self.unconditional_variance,
            "half_life_bars": self.half_life,
        }


def _garch_filter(e2: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """One-step-ahead variances for a fully observed ``e2``.

    Returns ``hnext`` with ``hnext[t] = omega + alpha * e2[t] + beta * hnext[t-1]`` and
    ``hnext[-1] := h0``. The recursion is a first-order IIR filter, evaluated with
    ``scipy.signal.lfilter`` (vectorised, O(n)).
    """
    x = omega + alpha * e2
    out, _ = signal.lfilter([1.0], [1.0, -beta], x, zi=np.array([beta * h0]))
    return out


def _garch_filter_nan(e2: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """Like :func:`_garch_filter` but tolerant of missing observations.

    When ``e2[t]`` is missing we replace it by its conditional expectation ``h_t`` so that
    ``h_{t+1} = omega + (alpha + beta) h_t`` — the correct multi-step forecast.
    """
    finite = np.isfinite(e2)
    n = e2.shape[0]
    if finite.all():
        return _garch_filter(e2, omega, alpha, beta, h0)
    out = np.empty(n)
    if not finite.any():
        h = h0
        for t in range(n):
            h = omega + (alpha + beta) * h
            out[t] = h
        return out
    first = int(np.argmax(finite))
    h = h0
    for t in range(first):
        h = omega + (alpha + beta) * h
        out[t] = h
    if finite[first:].all():
        out[first:] = _garch_filter(e2[first:], omega, alpha, beta, h)
        return out
    for t in range(first, n):
        if finite[t]:
            h = omega + alpha * e2[t] + beta * h
        else:
            h = omega + (alpha + beta) * h
        out[t] = h
    return out


def simulate_garch11(
    n: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    mu: float = 0.0,
    seed: int = 0,
    burn: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Gaussian GARCH(1,1). Returns ``(returns, conditional_variances)``.

    Used for tests and Monte-Carlo research. ``burn`` initial draws are discarded so the
    process starts from its stationary distribution.
    """
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        raise ValueError("parameters violate omega>0, alpha,beta>=0, alpha+beta<1")
    rng = np.random.default_rng(seed)
    total = n + burn
    z = rng.standard_normal(total)
    h = np.empty(total)
    r = np.empty(total)
    h_prev = omega / (1.0 - alpha - beta)
    e_prev2 = h_prev
    for t in range(total):
        h_t = omega + alpha * e_prev2 + beta * h_prev
        e = math.sqrt(h_t) * z[t]
        h[t] = h_t
        r[t] = mu + e
        h_prev, e_prev2 = h_t, e * e
    return r[burn:], h[burn:]


class Garch11:
    """GARCH(1,1) volatility model (Bollerslev, 1986) with Gaussian QMLE.

    Estimation
        The mean is either zero (``mean="zero"``, the usual choice for intraday returns
        whose mean is swamped by noise) or the training-sample mean (``mean="constant"``,
        two-step estimator). Returns are standardised by their training standard deviation
        before optimisation so the problem is well scaled (``omega`` is mapped back
        afterwards; ``alpha`` and ``beta`` are scale invariant). The likelihood recursion is
        started at the training-sample variance (the classical "backcast"), which is stored
        in ``h0_`` and reused for forecasting so later data never influence the start.
        Optimisation uses SLSQP with bounds and the inequality ``alpha + beta <= 1 - 1e-6``
        from several starting points; the best likelihood wins.

    Forecasting (causal)
        ``forecast(returns)`` returns at row ``t`` the forecast of volatility over the next
        ``horizon`` bars made with returns up to and including ``t``, annualised with
        ``bars_per_year`` (inferred from a DatetimeIndex if not given). ``floor`` / ``cap``
        are annualised fractions and are applied to annualised forecasts only.
    """

    def __init__(
        self,
        *,
        mean: str = "zero",
        bars_per_year: float | None = None,
        floor: float | None = None,
        cap: float | None = None,
        max_iter: int = 500,
        min_obs: int = 250,
    ) -> None:
        if mean not in ("zero", "constant"):
            raise ValueError("mean must be 'zero' or 'constant'")
        self.mean = mean
        self.bars_per_year = bars_per_year
        self.floor = floor
        self.cap = cap
        self.max_iter = int(max_iter)
        self.min_obs = int(min_obs)
        self.params_: GarchParams | None = None
        self.h0_: float = math.nan
        self.loglik_: float = math.nan
        self.n_obs_: int = 0
        self.converged_: bool = False
        self.bars_per_year_: float | None = None

    # ---- estimation -------------------------------------------------------------------
    def fit(self, returns: pd.Series | np.ndarray) -> Garch11:
        """Estimate parameters by Gaussian (quasi) maximum likelihood on TRAINING returns."""
        r, index = _as_array(returns)
        r = r[np.isfinite(r)]
        n = r.shape[0]
        if n < self.min_obs:
            raise ValueError(f"need at least {self.min_obs} finite returns to fit GARCH, got {n}")
        mu = float(r.mean()) if self.mean == "constant" else 0.0
        eps = r - mu
        scale = float(np.sqrt(np.mean(eps**2)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("returns have zero variance; cannot fit GARCH")
        z2 = (eps / scale) ** 2
        h0 = float(z2.mean())  # == 1 by construction; kept explicit for clarity

        def nll(theta: np.ndarray) -> float:
            omega, alpha, beta = (float(v) for v in theta)
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
                return 1e10
            h = np.empty(n)
            h[0] = h0
            h[1:] = _garch_filter(z2[:-1], omega, alpha, beta, h0)
            if not np.all(np.isfinite(h)) or np.any(h <= 0):
                return 1e10
            return 0.5 * float(np.mean(_LOG_2PI + np.log(h) + z2 / h))

        starts = [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95), (0.15, 0.70), (0.10, 0.40)]
        bounds = [(1e-8, 10.0 * h0), (0.0, 1.0), (0.0, 1.0)]
        cons = [{"type": "ineq", "fun": lambda th: 1.0 - 1e-6 - th[1] - th[2]}]
        best: optimize.OptimizeResult | None = None
        for a0, b0 in starts:
            x0 = np.array([h0 * (1.0 - a0 - b0), a0, b0])
            try:
                res = optimize.minimize(
                    nll,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"maxiter": self.max_iter, "ftol": 1e-10},
                )
            except (ValueError, FloatingPointError) as exc:  # pragma: no cover - defensive
                logger.debug("GARCH start %s failed: %s", (a0, b0), exc)
                continue
            if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
                best = res
        if best is None or not np.isfinite(best.fun) or best.fun >= 1e9:
            raise RuntimeError("GARCH(1,1) optimisation failed from every starting point")
        omega_z, alpha, beta = (float(v) for v in best.x)
        # Numerical safety: enforce constraints exactly after the optimiser.
        alpha = max(alpha, 0.0)
        beta = max(beta, 0.0)
        if alpha + beta >= 1.0:
            s = (1.0 - 1e-6) / (alpha + beta)
            alpha, beta = alpha * s, beta * s
        self.params_ = GarchParams(mu=mu, omega=max(omega_z, 1e-12) * scale**2, alpha=alpha, beta=beta)
        self.h0_ = h0 * scale**2
        self.loglik_ = -float(best.fun) * n - n * math.log(scale)  # log-lik in original units
        self.n_obs_ = n
        self.converged_ = bool(best.success)
        if isinstance(index, pd.DatetimeIndex) and self.bars_per_year is None and len(index) >= 2:
            self.bars_per_year_ = float(infer_bars_per_year(index))
        else:
            self.bars_per_year_ = self.bars_per_year
        if not self.converged_:
            logger.warning("GARCH(1,1) optimiser did not report convergence: %s", best.message)
        logger.debug("GARCH(1,1) fitted: %s", self.params_)
        return self

    def _check(self) -> GarchParams:
        if self.params_ is None:
            raise RuntimeError("Garch11 is not fitted; call fit(train_returns) first")
        return self.params_

    @property
    def aic(self) -> float:
        return 2 * (3 + (self.mean == "constant")) - 2 * self.loglik_

    @property
    def bic(self) -> float:
        k = 3 + (self.mean == "constant")
        return k * math.log(max(self.n_obs_, 1)) - 2 * self.loglik_

    # ---- filtering / forecasting ------------------------------------------------------
    def conditional_variance(self, returns: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        """In-sample conditional variance ``sigma_t^2`` (per bar) — row t uses returns < t."""
        p = self._check()
        r, index = _as_array(returns)
        e2 = (r - p.mu) ** 2
        hnext = _garch_filter_nan(e2, p.omega, p.alpha, p.beta, self.h0_)
        h = np.empty_like(hnext)
        if h.size:
            h[0] = self.h0_
            h[1:] = hnext[:-1]
        return pd.Series(h, index=index, name="garch_var") if index is not None else h

    def forecast(
        self,
        returns: pd.Series | np.ndarray,
        *,
        horizon: int = 1,
        annualise: bool = True,
    ) -> pd.Series:
        """Causal volatility forecast: row ``t`` = vol over bars ``t+1..t+horizon`` given r[:t+1].

        For ``horizon > 1`` the average variance over the horizon is used,
        ``VL + (h_{t+1} - VL) (1 - p^H) / ((1 - p) H)`` with ``p = alpha + beta`` and long-run
        variance ``VL``. Annualised with ``sqrt(bars_per_year)`` when ``annualise``.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        p = self._check()
        r, index = _as_array(returns)
        e2 = (r - p.mu) ** 2
        hnext = _garch_filter_nan(e2, p.omega, p.alpha, p.beta, self.h0_)
        if horizon > 1:
            pers = p.persistence
            vl = p.unconditional_variance
            if pers <= 0:
                factor = 1.0 / horizon
            else:
                factor = (1.0 - pers**horizon) / ((1.0 - pers) * horizon)
            hnext = vl + (hnext - vl) * factor
        vol = np.sqrt(np.maximum(hnext, 0.0))
        if annualise:
            # Prefer the explicit value, then the density measured on the TRAINING index
            # (robust), and only then the (possibly short) index being forecast.
            bpy = self.bars_per_year if self.bars_per_year is not None else self.bars_per_year_
            if bpy is None:
                bpy = _resolve_bpy(None, index)
            vol = vol * math.sqrt(bpy)
            # ``floor``/``cap`` are ANNUALISED fractions, so they only apply to annualised
            # output (clipping a per-bar vol at e.g. 0.03 would be a unit error).
            if self.floor is not None or self.cap is not None:
                vol = np.clip(vol, self.floor if self.floor is not None else -np.inf,
                              self.cap if self.cap is not None else np.inf)
        name = "garch_vol_ann" if annualise else "garch_vol"
        return pd.Series(vol, index=index if index is not None else pd.RangeIndex(len(vol)), name=name)

    def summary(self) -> dict:
        """JSON-friendly description for reports and agents."""
        p = self._check()
        out = p.to_dict()
        out.update(
            loglik=self.loglik_,
            aic=self.aic,
            bic=self.bic,
            n_obs=self.n_obs_,
            converged=self.converged_,
        )
        bpy = self.bars_per_year or self.bars_per_year_
        if bpy:
            out["unconditional_vol_ann"] = math.sqrt(p.unconditional_variance * bpy)
        return out

    def __repr__(self) -> str:
        return f"Garch11(mean={self.mean!r}, params={self.params_})"


# ----------------------------------------------------------------------------------------
# Realised variance & HAR-RV
# ----------------------------------------------------------------------------------------
def _session_end(label: pd.Timestamp, anchor_hour_utc: int) -> pd.Timestamp:
    """End of the trading session labelled ``label`` (see :func:`daily_realised_variance`)."""
    if anchor_hour_utc == 0:
        return label + pd.Timedelta(days=1)
    return label + pd.Timedelta(hours=anchor_hour_utc)


def daily_realised_variance(
    bars: pd.DataFrame,
    *,
    anchor_hour_utc: int = 0,
    complete_only: bool = True,
    fold_weekends: bool = True,
) -> pd.DataFrame:
    """Daily realised variance from intraday bars (Andersen, Bollerslev, Diebold & Labys, 2003).

    ``rv[D] = sum of squared close-to-close log returns of the bars that OPEN in session D``.
    Sessions run from ``anchor_hour_utc`` to ``anchor_hour_utc`` the next day (e.g. 21 for a
    broker/New-York-close day). A session is labelled by the date on which it ENDS for
    ``anchor_hour_utc > 0`` (FX/NY-close trade-date convention: the session opening Sunday
    21:00 UTC is Monday's) and by its own date for the midnight anchor. The first return of
    a session spans the overnight/weekend gap, so RV includes gap risk.

    ``fold_weekends`` (default) merges sessions labelled Saturday or Sunday into the
    following Monday — the same rule as ``aurum.backtest.metrics.trading_dates``. Gold CFDs
    reopen on Sunday ~22:00 UTC, so with a midnight anchor the two Sunday bars would
    otherwise form a stub "day" whose RV is an order of magnitude below a full day's; fed to
    a daily model such as HAR-RV it drags the lags down and produces a far too LOW vol
    forecast for Monday (i.e. oversized positions).

    Returns a frame indexed by session label (tz-aware UTC midnight) with columns ``rv``,
    ``n_obs`` and ``available_at`` (the ``available_at`` of the session's last bar), so it can
    be aligned onto trading bars with :func:`aurum.data.pit.asof_join`.

    ``complete_only`` drops the final session when its last bar ends before the session
    boundary — a partial day's RV is not comparable to full days and must not be fed to a
    daily model.
    """
    if "close" not in bars.columns or "available_at" not in bars.columns:
        raise ValueError("bars must contain 'close' and 'available_at'")
    if not 0 <= int(anchor_hour_utc) <= 23:
        raise ValueError("anchor_hour_utc must be an hour in [0, 23]")
    anchor_hour_utc = int(anchor_hour_utc)
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        raise ValueError("bars index must be tz-aware UTC")
    idx = idx.tz_convert("UTC")
    r = np.log(bars["close"].astype(float)).diff()
    offset = pd.Timedelta(hours=anchor_hour_utc)
    label = (idx - offset).normalize()
    if anchor_hour_utc > 0:
        label = label + pd.Timedelta(days=1)
    if fold_weekends and len(label):
        wd = label.weekday.to_numpy()
        shift = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        if shift.any():
            label = label + pd.to_timedelta(shift, unit="D")
    frame = pd.DataFrame(
        {"r2": (r**2).to_numpy(), "avail": pd.DatetimeIndex(bars["available_at"]).tz_convert("UTC")},
        index=label,
    )
    g = frame.groupby(level=0, sort=True)
    out = pd.DataFrame(
        {
            "rv": g["r2"].sum(min_count=1),
            "n_obs": g["r2"].count(),
            "available_at": g["avail"].max(),
        }
    )
    out.index.name = "date"
    out = out.dropna(subset=["rv"])
    if complete_only and len(out):
        day_end = _session_end(out.index[-1], anchor_hour_utc)
        if out["available_at"].iloc[-1] < day_end:
            out = out.iloc[:-1]
    return out


class HarRV:
    """HAR-RV model of Corsi (2009).

    ``RV_{t+1} = b0 + b_d RV_t + b_w mean(RV_{t-4..t}) + b_m mean(RV_{t-21..t}) + e_{t+1}``.

    With ``log=True`` the regression is run on logs (Andersen et al. 2007 show log-HAR fits
    better and cannot produce negative forecasts); predictions are mapped back with the
    log-normal bias correction ``exp(mu + s^2/2)``.

    Zero or tiny RV values (holidays, flat sessions) are floored before taking logs and
    level forecasts are floored at the same value. The floor (``1e-6 x`` the median positive
    RV) is estimated on the TRAINING sample in :meth:`fit` and stored in ``floor_``; it is
    never recomputed from the series passed to :meth:`predict`, which may contain data
    after the forecast origin (a whole-sample floor would leak future RV levels into the
    regressors of every row whose window contains a zero-RV day).
    """

    def __init__(self, lags: Sequence[int] = (1, 5, 22), *, log: bool = False) -> None:
        lags = tuple(int(v) for v in lags)
        if not lags or min(lags) < 1:
            raise ValueError("lags must be positive integers")
        self.lags = lags
        self.log = log
        self.coef_: pd.Series | None = None
        self.resid_var_: float = math.nan
        self.r2_: float = math.nan
        self.n_obs_: int = 0
        self.floor_: float | None = None

    @staticmethod
    def _floor(rv: pd.Series | np.ndarray) -> float:
        """Scale-free positive floor: 1e-6 x median positive RV of the given sample."""
        arr = np.asarray(rv, dtype=float)
        pos = arr[np.isfinite(arr) & (arr > 0)]
        return float(np.median(pos) * 1e-6) if pos.size else 1e-18

    def _resolve_floor(self, rv: pd.Series, floor: float | None) -> float:
        if floor is not None:
            return float(floor)
        if self.floor_ is not None:
            return self.floor_
        return self._floor(rv)

    def _levels(self, daily_rv: pd.Series) -> pd.DataFrame:
        """Trailing means of RV (levels) over each lag window ending at t — causal."""
        rv = daily_rv.astype(float)
        return pd.DataFrame(
            {f"rv_{lag}": rv.rolling(lag, min_periods=lag).mean() for lag in self.lags}, index=rv.index
        )

    def design(self, daily_rv: pd.Series, *, floor: float | None = None) -> pd.DataFrame:
        """Causal regressors at row t: trailing means of RV over each lag window ending at t.

        In log mode the means are floored at ``floor`` (default: the fitted ``floor_``, or —
        before fitting — the floor of ``daily_rv`` itself, which is only legitimate when
        ``daily_rv`` is the training sample).
        """
        x = self._levels(daily_rv)
        if self.log:
            x = np.log(x.clip(lower=self._resolve_floor(daily_rv, floor)))
        return x

    def _target(self, daily_rv: pd.Series, *, floor: float | None = None) -> pd.Series:
        rv = daily_rv.astype(float)
        y = np.log(rv.clip(lower=self._resolve_floor(rv, floor))) if self.log else rv
        # Next-period target for TRAINING pairs only (forecast origin t, realisation t+1).
        return y.shift(-1)

    @staticmethod
    def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        xx = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(xx, y, rcond=None)
        resid = y - xx @ coef
        dof = max(len(y) - xx.shape[1], 1)
        s2 = float(resid @ resid / dof)
        tss = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else math.nan
        return coef, s2, r2

    def fit(self, daily_rv: pd.Series) -> HarRV:
        """OLS on TRAINING daily RV (pairs (X_t, RV_{t+1}) with both inside the sample)."""
        floor = self._floor(daily_rv)
        x = self.design(daily_rv, floor=floor)
        y = self._target(daily_rv, floor=floor)
        ok = x.notna().all(axis=1) & y.notna()
        n = int(ok.sum())
        if n < len(self.lags) + 10:
            raise ValueError(f"not enough observations to fit HAR-RV ({n})")
        coef, s2, r2 = self._ols(x.loc[ok].to_numpy(), y.loc[ok].to_numpy())
        self.coef_ = pd.Series(coef, index=["const", *x.columns])
        self.resid_var_ = s2
        self.r2_ = r2
        self.n_obs_ = n
        self.floor_ = floor
        return self

    def predict(self, daily_rv: pd.Series) -> pd.Series:
        """Forecast of RV_{t+1} made at t (daily variance units). Causal given fitted coef."""
        if self.coef_ is None or self.floor_ is None:
            raise RuntimeError("HarRV is not fitted")
        x = self.design(daily_rv, floor=self.floor_)
        pred = self.coef_.iloc[0] + x.to_numpy() @ self.coef_.iloc[1:].to_numpy()
        pred = pd.Series(pred, index=x.index, name="har_rv_forecast")
        if self.log:
            pred = np.exp(pred + 0.5 * self.resid_var_)
        else:
            pred = pred.clip(lower=self.floor_)
        return pred


def har_rv_forecast(
    daily_rv: pd.Series,
    *,
    lags: Sequence[int] = (1, 5, 22),
    log: bool = False,
    min_train: int = 250,
    refit_every: int = 21,
    window: int | None = None,
    output: str = "vol",
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Walk-forward HAR-RV forecast of next-day realised variance/volatility.

    At each refit origin ``t0`` (every ``refit_every`` rows after ``min_train``) the model is
    estimated on pairs ``(X_s, RV_{s+1})`` with ``s + 1 <= t0`` (expanding, or the last
    ``window`` pairs), then used for rows ``t0 .. t0 + refit_every - 1``. Row ``t`` therefore
    depends only on ``daily_rv[:t+1]`` — strictly point-in-time.

    ``output="vol"`` returns annualised volatility ``sqrt(252 * RV_hat)``; ``"variance"``
    returns the daily variance forecast. Rows before the first fit are NaN.

    The positivity floor used for logs (and to floor level forecasts) is re-estimated at
    each refit origin from ``daily_rv[:t0+1]`` only, so even the floor is point-in-time.
    """
    if output not in ("vol", "variance"):
        raise ValueError("output must be 'vol' or 'variance'")
    rv = daily_rv.astype(float)
    model = HarRV(lags, log=log)
    levels = model._levels(rv).to_numpy()   # causal trailing means (floor-independent)
    rv_arr = rv.to_numpy()
    y_lvl = np.append(rv_arr[1:], np.nan)   # y[s] = RV_{s+1}: only used for s + 1 <= t0
    n = len(rv)
    k = levels.shape[1]
    pred = np.full(n, np.nan)
    start = max(int(min_train), max(model.lags) + 1)
    for t0 in range(start, n, max(int(refit_every), 1)):
        lo = 0 if window is None else max(0, t0 - int(window))
        hi = min(t0 + int(refit_every), n)
        floor = HarRV._floor(rv_arr[: t0 + 1])  # information available at the origin t0
        if log:  # np.maximum propagates NaN (warm-up rows stay NaN)
            x = np.log(np.maximum(levels[lo:hi], floor))
            y = np.log(np.maximum(y_lvl[lo:t0], floor))
        else:
            x = levels[lo:hi]
            y = y_lvl[lo:t0]
        xs = x[: t0 - lo]  # rows s <= t0-1  =>  target index s+1 <= t0
        ok = np.isfinite(xs).all(axis=1) & np.isfinite(y)
        if ok.sum() < max(3 * (k + 1), 30):
            continue
        coef, s2, _ = HarRV._ols(xs[ok], y[ok])
        seg = coef[0] + x[t0 - lo:] @ coef[1:]
        seg = np.exp(seg + 0.5 * s2) if log else np.maximum(seg, floor)
        pred[t0:hi] = seg
    out = pd.Series(pred, index=rv.index)
    if output == "vol":
        return np.sqrt(out * periods_per_year).rename("har_vol_ann")
    return out.rename("har_rv_forecast")


# ----------------------------------------------------------------------------------------
# Blending
# ----------------------------------------------------------------------------------------
def blend_vol(
    *series: pd.Series,
    weights: Sequence[float] | None = None,
    name: str = "vol_blend",
) -> pd.Series:
    """Combine volatility forecasts by weighted averaging of VARIANCES.

    Averaging variances (not vols) is the natural linear pooling for second moments and
    avoids the Jensen downward bias of averaging square roots. Series are outer-aligned on
    their index; where some inputs are missing, weights are renormalised over those present.
    """
    if not series:
        raise ValueError("blend_vol needs at least one series")
    k = len(series)
    if weights is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (k,):
            raise ValueError("weights must have one entry per series")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        w = w / w.sum()
    frame = pd.concat([pd.Series(s, dtype=float).rename(i) for i, s in enumerate(series)], axis=1)
    var = frame.to_numpy() ** 2
    mask = np.isfinite(var)
    wm = np.where(mask, w[None, :], 0.0)
    denom = wm.sum(axis=1)
    num = np.where(mask, var, 0.0) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        blended = np.where(denom > 0, np.sqrt(num / np.where(denom > 0, denom, 1.0)), np.nan)
    return pd.Series(blended, index=frame.index, name=name)
