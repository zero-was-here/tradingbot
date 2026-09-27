"""Forecast combination (SPEC §7).

Strategies emit forecasts in [-1, 1]. The combiner learns, on TRAINING data only, a
weight per strategy and a *forecast diversification multiplier* (FDM), then produces

    combined[t] = clip(FDM * sum_i w_i * f_i[t], -1, 1)

Evaluation stream
    For each strategy the combiner measures the *unit-volatility return stream*
    ``u_i[t] = f_i[t] * r[t+1] / vol[t]`` where ``r[t+1]`` is the next bar's log return and
    ``vol[t]`` a causal EWMA per-bar volatility. This is exactly the return a vol-targeted
    position (see ``aurum.portfolio.sizing``) earns per unit of target volatility, so the
    statistics are comparable across calm and turbulent periods. Using ``r[t+1]`` is only
    legitimate because ``fit`` receives training data: the close series is restricted to
    the forecasts' own index, so the last training row has no forward return and is
    dropped — nothing beyond the training window is touched.

Weighting methods
    * ``equal``          — 1/N (DeMiguel, Garlappi & Uppal, 2009: hard to beat out of sample).
    * ``inverse_vol``    — ``w_i ∝ 1 / std(u_i)``: naive risk parity.
    * ``sharpe_shrink``  — diagonal mean-variance ``w_i ∝ SR_i^shrunk / std(u_i)`` where
      ``SR^shrunk = (1 - λ) SR_i + λ mean(SR)`` shrinks noisy in-sample Sharpe ratios toward
      the cross-sectional mean (James-Stein style; cf. Jorion, 1986). Negative shrunk
      Sharpes get zero weight. If no strategy has a positive shrunk Sharpe the method falls
      back to inverse-vol (and says so in ``explain()``).
    * ``hrp``            — Hierarchical Risk Parity (López de Prado, 2016, "Building
      Diversified Portfolios that Outperform Out of Sample", J. Portfolio Management):
      single-linkage clustering on the correlation distance ``sqrt((1 - ρ)/2)``,
      and top-down inverse-variance splitting along the dendrogram.

    Weights are then capped at ``max_weight`` with proportional redistribution of the
    excess (the cap is relaxed to 1/N if it is infeasible). Strategies whose forecast is
    constant over the training window carry no information and get zero weight.

Forecast diversification multiplier (Carver, 2015, *Systematic Trading*, ch. 8)
    Averaging imperfectly correlated forecasts shrinks their dispersion; the FDM restores
    it: ``FDM = 1 / sqrt(w' H w)`` with ``H`` the correlation matrix of the (active) training
    forecasts, negative correlations floored at 0 as Carver recommends (conservative), and
    the result capped at ``fdm_cap``. Carver's derivation assumes forecasts share a common
    scale; strategies here all live on [-1, 1] but may differ in typical magnitude, which
    the combiner reports (``avg_abs_forecast``) rather than silently rescaling.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from aurum.core.timeframes import infer_bars_per_year

logger = logging.getLogger(__name__)

METHODS = ("sharpe_shrink", "equal", "inverse_vol", "hrp")


def cap_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Normalise non-negative weights to sum to 1 with every weight <= ``max_weight``.

    The excess above the cap is redistributed proportionally to the uncapped weights (or
    evenly if they are all zero). If ``max_weight * N < 1`` the cap is infeasible and is
    relaxed to ``1/N``.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    n = w.size
    if n == 0:
        return w
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n)
    w = w / w.sum()
    cap = max(float(max_weight), 1.0 / n)
    for _ in range(n + 2):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = w < cap - 1e-12
        if not free.any():
            break
        mass = float(w[free].sum())
        if mass > 1e-15:
            w[free] += excess * w[free] / mass
        else:
            w[free] += excess / free.sum()
    return w / w.sum()


def _hrp_weights(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights (López de Prado, 2016).

    Single-linkage tree on the correlation distance ``sqrt((1 - ρ) / 2)``; capital is then
    split top-down between the two children of every dendrogram node in inverse proportion
    to their inverse-variance cluster variances. Following the tree (Pfitzinger & Katzke,
    2019) rather than halving the quasi-diagonal leaf order (the original recursive
    bisection) guarantees that a natural cluster is never cut in two.
    """
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    link = linkage(squareform(dist, checks=False), method="single")

    def cluster_var(items: list[int]) -> float:
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.maximum(np.diag(sub), 1e-18)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = np.ones(n)
    stack = [to_tree(link)]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            continue
        left, right = node.get_left(), node.get_right()
        c1, c2 = left.pre_order(), right.pre_order()
        v1, v2 = cluster_var(c1), cluster_var(c2)
        a = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
        w[c1] *= a
        w[c2] *= 1.0 - a
        stack.extend([left, right])
    return w / w.sum()


class ForecastCombiner:
    """Blend strategy forecasts into one forecast in [-1, 1] (SPEC §7).

    Parameters
    ----------
    method        : "sharpe_shrink" | "equal" | "inverse_vol" | "hrp".
    shrinkage     : λ in [0, 1] for "sharpe_shrink" (1 = ignore Sharpe differences).
    max_weight    : per-strategy weight cap.
    fdm_cap       : cap on the forecast diversification multiplier.
    vol_halflife  : half-life (bars) of the causal EWMA volatility used for unit-vol streams.
    min_periods   : warm-up bars for that volatility.
    corr_floor    : floor on pairwise forecast correlations in the FDM (Carver uses 0).
    bars_per_year : annualisation for the reported Sharpe ratios (inferred if None).
    """

    def __init__(
        self,
        method: str = "sharpe_shrink",
        shrinkage: float = 0.5,
        max_weight: float = 0.4,
        fdm_cap: float = 2.5,
        *,
        vol_halflife: float = 48.0,
        min_periods: int = 20,
        corr_floor: float = 0.0,
        bars_per_year: float | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        if not fdm_cap >= 1.0:
            raise ValueError("fdm_cap must be >= 1")
        self.method = method
        self.shrinkage = float(shrinkage)
        self.max_weight = float(max_weight)
        self.fdm_cap = float(fdm_cap)
        self.vol_halflife = float(vol_halflife)
        self.min_periods = int(min_periods)
        self.corr_floor = float(corr_floor)
        self.bars_per_year = bars_per_year
        # fitted state
        self.weights_: pd.Series | None = None
        self.fdm_: float = math.nan
        self.fdm_raw_: float = math.nan
        self.train_sharpe_: pd.Series | None = None
        self.train_vol_: pd.Series | None = None
        self.train_mean_: pd.Series | None = None
        self.avg_abs_forecast_: pd.Series | None = None
        self.corr_: pd.DataFrame | None = None
        self.stream_corr_: pd.DataFrame | None = None
        self.n_obs_: int = 0
        self.notes_: list[str] = []
        self.columns_: list[str] = []
        self.train_start_: pd.Timestamp | None = None
        self.train_end_: pd.Timestamp | None = None

    # ------------------------------------------------------------------------------------
    def unit_vol_streams(self, forecasts: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """``f_i[t] * r[t+1] / vol[t]`` on the rows of ``forecasts`` (warm-up/last row dropped).

        ``close`` is restricted to ``forecasts.index`` so no price after the last forecast row
        is ever used.
        """
        if not isinstance(forecasts, pd.DataFrame) or forecasts.shape[1] == 0:
            raise ValueError("forecasts must be a non-empty DataFrame (one column per strategy)")
        # ``shift(-1)`` means "next bar" only on a strictly increasing index; on an unsorted
        # or duplicated index it would silently pair forecasts with the wrong returns.
        if not forecasts.index.is_monotonic_increasing or forecasts.index.has_duplicates:
            raise ValueError("forecasts index must be strictly increasing (sorted, no duplicates)")
        px = pd.Series(close, dtype=float)
        if px.index.has_duplicates:
            raise ValueError("close index has duplicate timestamps")
        px = px.reindex(forecasts.index)
        if px.isna().all():
            raise ValueError("close has no prices on the forecasts' index")
        logret = np.log(px).diff()
        fwd = logret.shift(-1)  # r[t+1] within the given window only; last row -> NaN
        vol = np.sqrt(
            (logret**2).ewm(halflife=self.vol_halflife, min_periods=self.min_periods, adjust=False).mean()
        )
        scale = fwd / vol
        valid = scale.notna() & np.isfinite(scale) & (vol > 0)
        f = forecasts.astype(float).clip(-1.0, 1.0).fillna(0.0)
        return f.loc[valid].mul(scale.loc[valid], axis=0)

    def fit(self, forecasts: pd.DataFrame, close: pd.Series) -> ForecastCombiner:
        """Learn weights and FDM from TRAINING forecasts and the matching close prices."""
        streams = self.unit_vol_streams(forecasts, close)
        cols = [str(c) for c in forecasts.columns]
        if len(set(cols)) != len(cols):
            raise ValueError("forecast column names must be unique")
        n_obs = len(streams)
        if n_obs < 30:
            raise ValueError(f"not enough training observations to fit the combiner ({n_obs})")
        self.notes_ = []
        self.columns_ = list(forecasts.columns)
        self.train_start_ = forecasts.index[0] if len(forecasts) else None
        self.train_end_ = forecasts.index[-1] if len(forecasts) else None

        if self.bars_per_year is not None:
            bpy = float(self.bars_per_year)
        elif isinstance(forecasts.index, pd.DatetimeIndex) and len(forecasts.index) >= 2:
            bpy = float(infer_bars_per_year(forecasts.index))
        else:
            bpy = 252.0
            self.notes_.append("no DatetimeIndex: Sharpe annualised with 252 periods/year")

        u = streams.to_numpy()
        mean = u.mean(axis=0)
        sd = u.std(axis=0, ddof=1)
        active = sd > 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            sr_bar = np.where(active, mean / np.where(active, sd, 1.0), 0.0)
        sharpe = sr_bar * math.sqrt(bpy)
        f_train = forecasts.astype(float).clip(-1.0, 1.0).fillna(0.0).loc[streams.index]
        f_sd = f_train.to_numpy().std(axis=0)
        active &= f_sd > 1e-12
        if not active.all():
            dead = [c for c, a in zip(self.columns_, active, strict=True) if not a]
            self.notes_.append(f"inactive in train (zero weight): {dead}")
            logger.info("ForecastCombiner: strategies inactive in train get zero weight: %s", dead)

        raw = self._raw_weights(u, sd, sr_bar, active)
        w = np.zeros(len(cols))
        if active.any():
            n_act = int(active.sum())
            cap = self.max_weight
            if cap * n_act < 1.0 - 1e-12:
                cap = 1.0 / n_act
                self.notes_.append(
                    f"max_weight {self.max_weight} infeasible with {n_act} active strategies; relaxed to {cap:.4f}"
                )
            w[active] = cap_weights(raw[active], cap)
            forced = [c for c, wi, ri in zip(self.columns_, w, raw, strict=True) if wi > 1e-12 and ri <= 0]
            if forced:
                msg = (
                    f"max_weight={cap:.3f} forced {w[[c in forced for c in self.columns_]].sum():.1%} of "
                    f"weight onto strategies the method scored at zero (e.g. non-positive shrunk Sharpe): "
                    f"{forced}; consider pruning them or raising max_weight"
                )
                self.notes_.append(msg)
                logger.warning("ForecastCombiner: %s", msg)
        else:
            w = np.full(len(cols), 1.0 / len(cols))
            self.notes_.append("no active strategy in train: equal weights")

        corr = f_train.corr().to_numpy()
        corr = np.where(np.isfinite(corr), corr, 0.0)
        np.fill_diagonal(corr, 1.0)
        h = np.maximum(corr, self.corr_floor)
        np.fill_diagonal(h, 1.0)
        wa = np.where(active, w, 0.0)
        quad = float(wa @ h @ wa)
        wsum = wa.sum()
        if quad > 0 and wsum > 0:
            # Normalise by the active mass so inactive (always-zero) forecasts do not inflate it.
            fdm_raw = wsum / math.sqrt(quad)
        else:
            fdm_raw = 1.0
        fdm_raw = max(fdm_raw, 1.0)
        self.fdm_raw_ = float(fdm_raw)
        self.fdm_ = float(min(fdm_raw, self.fdm_cap))
        if fdm_raw > self.fdm_cap:
            self.notes_.append(f"FDM {fdm_raw:.3f} capped at {self.fdm_cap}")

        self.weights_ = pd.Series(w, index=self.columns_, name="weight")
        self.train_sharpe_ = pd.Series(sharpe, index=self.columns_, name="train_sharpe")
        # Per-bar moments of the unit-vol stream (in units of one bar's volatility; the std is
        # ~ the RMS forecast when forecasts and returns are independent).
        self.train_vol_ = pd.Series(sd, index=self.columns_, name="train_stream_std")
        self.train_mean_ = pd.Series(mean, index=self.columns_, name="train_stream_mean")
        self.avg_abs_forecast_ = pd.Series(np.abs(f_train.to_numpy()).mean(axis=0), index=self.columns_)
        self.corr_ = pd.DataFrame(corr, index=self.columns_, columns=self.columns_)
        self.stream_corr_ = streams.corr()
        self.n_obs_ = int(n_obs)
        logger.info("ForecastCombiner(%s) fitted on %d bars: weights=%s fdm=%.3f",
                    self.method, n_obs, np.round(w, 4).tolist(), self.fdm_)
        return self

    def _raw_weights(self, u: np.ndarray, sd: np.ndarray, sr: np.ndarray, active: np.ndarray) -> np.ndarray:
        n = u.shape[1]
        raw = np.zeros(n)
        if not active.any():
            return np.ones(n)
        inv_vol = np.where(active, 1.0 / np.where(active, sd, 1.0), 0.0)
        if self.method == "equal":
            raw = active.astype(float)
        elif self.method == "inverse_vol":
            raw = inv_vol
        elif self.method == "sharpe_shrink":
            sr_mean = sr[active].mean()
            shrunk = (1.0 - self.shrinkage) * sr + self.shrinkage * sr_mean
            raw = np.where(active, np.clip(shrunk, 0.0, None) * inv_vol, 0.0)
            if raw.sum() <= 0:
                self.notes_.append("no positive shrunk Sharpe in train: fell back to inverse_vol")
                logger.warning("ForecastCombiner: no positive shrunk Sharpe; falling back to inverse-vol")
                raw = inv_vol
        elif self.method == "hrp":
            idx = np.flatnonzero(active)
            sub = u[:, idx]
            cov = np.cov(sub, rowvar=False).reshape(len(idx), len(idx))
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.sqrt(np.diag(cov))
                corr = cov / np.outer(d, d)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            np.fill_diagonal(corr, 1.0)
            raw[idx] = _hrp_weights(cov, corr)
        return raw

    # ------------------------------------------------------------------------------------
    def combine(self, forecasts: pd.DataFrame) -> pd.Series:
        """``clip(FDM * sum_i w_i f_i, -1, 1)`` — row-wise, hence causal."""
        if self.weights_ is None:
            raise RuntimeError("ForecastCombiner is not fitted; call fit(train_forecasts, train_close)")
        missing = [c for c in self.columns_ if c not in forecasts.columns]
        if missing:
            raise KeyError(f"forecasts missing fitted strategies: {missing}")
        extra = [c for c in forecasts.columns if c not in self.columns_]
        if extra:
            logger.debug("ForecastCombiner.combine ignoring unfitted columns: %s", extra)
        f = forecasts[self.columns_].astype(float).clip(-1.0, 1.0).fillna(0.0)
        out = (f.to_numpy() @ self.weights_.to_numpy()) * self.fdm_
        return pd.Series(np.clip(out, -1.0, 1.0), index=forecasts.index, name="combined")

    def fit_combine(self, forecasts: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Fit and combine on the SAME (training) data — in-sample, for diagnostics only."""
        return self.fit(forecasts, close).combine(forecasts)

    def explain(self) -> dict:
        """JSON-friendly summary for reports and LLM agents."""
        if self.weights_ is None:
            return {"fitted": False, "method": self.method}

        def _d(s: pd.Series | None) -> dict[str, float]:
            return {} if s is None else {str(k): float(v) for k, v in s.items()}

        off = self.corr_.to_numpy()[~np.eye(len(self.columns_), dtype=bool)] if self.corr_ is not None else []
        return {
            "fitted": True,
            "method": self.method,
            "weights": _d(self.weights_),
            "train_sharpe": _d(self.train_sharpe_),
            "sharpe_basis": "per-bar unit-vol stream f[t]*r[t+1]/vol[t] on train, annualised",
            "train_stream_std": _d(self.train_vol_),
            "avg_abs_forecast": _d(self.avg_abs_forecast_),
            "fdm": float(self.fdm_),
            "fdm_raw": float(self.fdm_raw_),
            "fdm_cap": self.fdm_cap,
            "avg_forecast_correlation": float(np.mean(off)) if len(off) else float("nan"),
            "shrinkage": self.shrinkage,
            "max_weight": self.max_weight,
            "n_obs": self.n_obs_,
            "train_start": str(self.train_start_) if self.train_start_ is not None else None,
            "train_end": str(self.train_end_) if self.train_end_ is not None else None,
            "notes": list(self.notes_),
        }

    def __repr__(self) -> str:
        return (
            f"ForecastCombiner(method={self.method!r}, shrinkage={self.shrinkage}, "
            f"max_weight={self.max_weight}, fdm_cap={self.fdm_cap})"
        )
