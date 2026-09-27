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

Net of trading costs
    Strategies are scored on their stream NET of an estimated execution cost per unit of
    forecast turnover::

        net_i[t] = f_i[t] * r[t+1] / vol[t]  -  c_t * |f_i[t] - f_i[t-1]|

    Derivation of ``c_t`` (same units as ``u``). The vol-target sizer holds a notional of
    ``f * (σ*/σ_ann) * E`` (``σ*`` the target vol, ``E`` equity); with per-bar vols
    ``σ*_b = σ*/sqrt(B)`` and ``vol[t] = σ_ann/sqrt(B)`` (``B`` bars/year) the position as a
    fraction of equity is ``f[t] * σ*_b / vol[t]``, so its bar return is
    ``σ*_b * f[t] * r[t+1] / vol[t] = σ*_b * u[t]``. Moving from ``f[t-1]`` to ``f[t]``
    trades ``σ*_b * |Δf[t]| / vol[t]`` of equity, and every unit of notional crossed costs
    ``k_t / close[t]`` with ``k_t`` the per-ounce price concession of one fill (half the
    effective spread + slippage + commission per ounce). Dividing the cost
    ``σ*_b * |Δf| / vol[t] * k_t / close[t]`` by ``σ*_b`` gives it in unit-vol units::

        c_t = (spread_eff[t] / 2 + slippage[t] + commission_per_oz) / (close[t] * vol[t])

    with ``spread_eff = max(spread * spread_multiplier, min_spread)`` and
    ``slippage = slippage_fixed + slippage_range_frac * (high - low)`` from
    :class:`aurum.execution.costs.CostModel` (the same model the simulator charges; bar
    ``t``'s spread/range stand in for the execution bar ``t+1``'s, and the ``sqrt(lots)``
    impact term is ignored because lots are unknown here). ``c_t`` is target-vol free: the
    target cancels. Approximations (documented, deliberately simple): trades caused by
    vol changes at a constant forecast are ignored; the sizer's rebalance band suppresses
    small trades (so jittery forecasts are over-charged); and each strategy is charged its
    own turnover, whereas in the blend opposite trades of different strategies net out —
    both make the estimate conservative for the combined book, which is the right side to
    err on when deciding whether a strategy earns its keep. Swap/financing (a holding, not
    a turnover cost) is not included. An explicit ``cost_per_turnover`` (scalar or series,
    in these units) overrides the estimate; without any cost information the combiner
    scores GROSS streams and says so in ``explain()``.

Weighting methods (all on the net streams when costs are known)
    * ``equal``          — 1/N (DeMiguel, Garlappi & Uppal, 2009: hard to beat out of sample).
    * ``inverse_vol``    — ``w_i ∝ 1 / std(u_i)``: naive risk parity.
    * ``sharpe_shrink``  — diagonal mean-variance ``w_i ∝ SR_i^shrunk / std(u_i)`` where
      ``SR^shrunk = (1 - λ) SR_i + λ mean(SR_j : SR_j > 0)`` shrinks the noisy in-sample
      Sharpe ratios of the candidate strategies toward their cross-sectional mean
      (James-Stein style; cf. Jorion, 1986). A strategy whose NET Sharpe is not positive gets
      zero weight and does not enter the shrinkage target (one heavy loser must not drag the
      candidates below zero). Legacy mode (``allow_unallocated=False``) shrinks toward the
      mean over ALL active strategies and zeroes non-positive SHRUNK Sharpes.
    * ``hrp``            — Hierarchical Risk Parity (López de Prado, 2016, "Building
      Diversified Portfolios that Outperform Out of Sample", J. Portfolio Management):
      single-linkage clustering on the correlation distance ``sqrt((1 - ρ)/2)``,
      and top-down inverse-variance splitting along the dendrogram.

    Weights are then capped at ``max_weight`` with proportional redistribution of the
    excess among the strategies the method scored positively (the cap is relaxed to
    ``1/N_active`` when it is infeasible for the configured strategy set). Strategies whose
    forecast is constant over the training window carry no information and get zero weight.

Unallocated risk (deliberate deviation from "weights sum to 1", default)
    With ``allow_unallocated=True`` (default) the cap never pushes weight onto strategies
    the method scored at zero: when the positively scored strategies cannot absorb the whole
    budget under ``max_weight`` (e.g. one skilled strategy and a 0.4 cap), the remainder is
    left UNALLOCATED — the weights sum to less than 1 and the book holds proportionally less
    risk — and when no strategy has a positive net Sharpe, ``sharpe_shrink`` leaves
    the book flat. The alternative, forcing weight onto strategies with negative expected
    net returns just to satisfy a concentration cap, adds risk with negative expected
    reward. ``allow_unallocated=False`` restores the previous behaviour exactly: weights
    always sum to 1 (the excess is redistributed onto zero-scored strategies, flagged in the
    notes), ``sharpe_shrink`` scores by shrunk Sharpe only and falls back to inverse-vol when
    no shrunk Sharpe is positive. ``inverse_vol``/``hrp``/``equal`` score every active
    strategy positively, so for them both settings give weights summing to 1.

Forecast diversification multiplier (Carver, 2015, *Systematic Trading*, ch. 8)
    Averaging imperfectly correlated forecasts shrinks their dispersion; the FDM restores
    it: ``FDM = sum(w) / sqrt(w' H w)`` with ``H`` the correlation matrix of the (active)
    training forecasts, negative correlations floored at 0 as Carver recommends
    (conservative), and the result capped at ``fdm_cap``. Normalising by ``sum(w)`` makes the
    FDM a property of the allocated sub-portfolio, so unallocated weight still reduces the
    combined forecast (less risk) instead of being scaled back up. Carver's derivation
    assumes forecasts share a common scale; strategies here all live on [-1, 1] but may
    differ in typical magnitude, which the combiner reports (``avg_abs_forecast``) rather
    than silently rescaling.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.timeframes import infer_bars_per_year

logger = logging.getLogger(__name__)

METHODS = ("sharpe_shrink", "equal", "inverse_vol", "hrp")


def cap_weights(weights: np.ndarray, max_weight: float, *, allow_unallocated: bool = False) -> np.ndarray:
    """Normalise non-negative weights with every weight <= ``max_weight``.

    ``allow_unallocated=False`` (the original contract): the result sums to 1. The excess
    above the cap is redistributed proportionally to the uncapped weights (or evenly if
    they are all zero, i.e. onto zero-scored entries). If ``max_weight * N < 1`` the cap is
    infeasible and is relaxed to ``1/N``.

    ``allow_unallocated=True``: only POSITIVE entries can receive weight (water-filling
    under the cap, never relaxed); when they cannot absorb the whole unit budget the
    remainder stays unallocated and the result sums to less than 1 (all zeros when no entry
    is positive).
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    n = w.size
    if n == 0:
        return w
    if allow_unallocated:
        if not np.isfinite(w).all():
            w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            return np.zeros(n)
        w = w / w.sum()
        cap = float(max_weight)
        for _ in range(n + 2):
            over = w > cap + 1e-12
            if not over.any():
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            free = (w > 1e-15) & (w < cap - 1e-12)
            if not free.any():
                break  # every positive entry is at the cap: the rest stays unallocated
            w[free] += excess * w[free] / float(w[free].sum())
        return np.minimum(w, cap)
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
    allow_unallocated : True (default) = the weight cap never forces weight onto strategies
                    scored at zero and ``sharpe_shrink`` drops non-positive net Sharpes, so the
                    weights may sum to < 1 (less risk); False = the previous sum-to-1 behaviour
                    (see the module docstring).
    cost_multiplier : scales the estimated turnover cost ``c_t`` (1 = the cost model's
                    estimate; > 1 stress-tests; 0 = score gross). Ignored for an explicit
                    ``cost_per_turnover``.
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
        allow_unallocated: bool = True,
        cost_multiplier: float = 1.0,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        if not fdm_cap >= 1.0:
            raise ValueError("fdm_cap must be >= 1")
        if not (math.isfinite(cost_multiplier) and cost_multiplier >= 0.0):
            raise ValueError("cost_multiplier must be finite and >= 0")
        self.method = method
        self.shrinkage = float(shrinkage)
        self.max_weight = float(max_weight)
        self.fdm_cap = float(fdm_cap)
        self.vol_halflife = float(vol_halflife)
        self.min_periods = int(min_periods)
        self.corr_floor = float(corr_floor)
        self.bars_per_year = bars_per_year
        self.allow_unallocated = bool(allow_unallocated)
        self.cost_multiplier = float(cost_multiplier)
        # fitted state
        self.weights_: pd.Series | None = None
        self.fdm_: float = math.nan
        self.fdm_raw_: float = math.nan
        self.train_sharpe_: pd.Series | None = None
        self.train_sharpe_gross_: pd.Series | None = None
        self.train_vol_: pd.Series | None = None
        self.train_mean_: pd.Series | None = None
        self.train_turnover_: pd.Series | None = None
        self.train_cost_: pd.Series | None = None
        self.avg_abs_forecast_: pd.Series | None = None
        self.corr_: pd.DataFrame | None = None
        self.stream_corr_: pd.DataFrame | None = None
        self.n_obs_: int = 0
        self.notes_: list[str] = []
        self.columns_: list[str] = []
        self.train_start_: pd.Timestamp | None = None
        self.train_end_: pd.Timestamp | None = None
        self.cost_basis_: str = "gross (not fitted)"
        self.avg_cost_per_turnover_: float = math.nan

    # ------------------------------------------------------------------------------------
    def _checked(self, forecasts: pd.DataFrame, close: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """(clean forecasts, close on their index, per-bar vol[t]) — shared by streams and costs."""
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
        vol = np.sqrt(
            (logret**2).ewm(halflife=self.vol_halflife, min_periods=self.min_periods, adjust=False).mean()
        )
        f = forecasts.astype(float).clip(-1.0, 1.0).fillna(0.0)
        return f, px, vol

    def unit_vol_streams(self, forecasts: pd.DataFrame, close: pd.Series, *,
                         cost_per_turnover: pd.Series | float | None = None) -> pd.DataFrame:
        """``f_i[t] * r[t+1] / vol[t]`` on the rows of ``forecasts`` (warm-up/last row dropped),
        minus ``c_t * |f_i[t] - f_i[t-1]|`` when ``cost_per_turnover`` (``c_t``, unit-vol units)
        is given.

        ``close`` is restricted to ``forecasts.index`` so no price after the last forecast row
        is ever used.
        """
        gross, net, _ = self._streams(forecasts, close, self._as_cost_series(cost_per_turnover, forecasts.index))
        return gross if net is None else net

    def _streams(self, forecasts: pd.DataFrame, close: pd.Series, cost: pd.Series | None
                 ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """(gross streams, net streams or None, turnover |Δf| or None) on the valid rows."""
        f, px, vol = self._checked(forecasts, close)
        fwd = np.log(px).diff().shift(-1)  # r[t+1] within the given window only; last row -> NaN
        scale = fwd / vol
        valid = scale.notna() & np.isfinite(scale) & (vol > 0)
        gross = f.loc[valid].mul(scale.loc[valid], axis=0)
        if cost is None:
            return gross, None, None
        # |f[t] - f[t-1]| within the window (the first row has no previous forecast: 0)
        turnover = f.diff().abs().fillna(0.0).loc[valid]
        c = cost.reindex(f.index).loc[valid]
        if c.isna().any():
            # rows without a cost estimate are charged the window's median (never zero-cost)
            c = c.fillna(float(c.median()) if c.notna().any() else 0.0)
        net = gross - turnover.mul(c, axis=0)
        return gross, net, turnover

    @staticmethod
    def _as_cost_series(cost: pd.Series | float | None, index: pd.Index) -> pd.Series | None:
        if cost is None:
            return None
        if isinstance(cost, pd.Series):
            if cost.index.has_duplicates:
                raise ValueError("cost_per_turnover index has duplicate timestamps")
            s = cost.astype(float).reindex(index)
        else:
            v = float(cost)
            s = pd.Series(v, index=index, dtype=float)
        if (s.dropna() < 0).any() or not np.isfinite(s.dropna()).all():
            raise ValueError("cost_per_turnover must be finite and >= 0")
        return s

    def turnover_cost(self, forecasts: pd.DataFrame, close: pd.Series, *, bars: pd.DataFrame | None = None,
                      spread: pd.Series | None = None, costs: Any = None, instrument: Any = None
                      ) -> pd.Series:
        """Estimated cost ``c_t`` per unit of forecast turnover, in unit-vol units (module docs).

        ``bars`` (a bars frame with ``spread`` and ideally ``high``/``low``) or ``spread`` (full
        spread in price units) supply the spread; both are restricted to the forecasts' index.
        ``costs`` is the :class:`~aurum.execution.costs.CostModel` (default ``CostModel()``),
        ``instrument`` gives the commission per ounce (default XAUUSD).
        """
        from aurum.core.instrument import XAUUSD
        from aurum.execution.costs import CostModel

        f, px, vol = self._checked(forecasts, close)
        idx = f.index
        if bars is not None:
            if "spread" not in bars.columns:
                raise ValueError("bars passed to the combiner need a 'spread' column")
            b = bars.reindex(idx)
            spr = b["spread"].astype(float)
            rng = (b["high"].astype(float) - b["low"].astype(float)) if {"high", "low"} <= set(b.columns) \
                else pd.Series(0.0, index=idx)
        elif spread is not None:
            spr = pd.Series(spread, dtype=float).reindex(idx)
            rng = pd.Series(0.0, index=idx)
        else:
            raise ValueError("turnover_cost needs bars or spread")
        cm = costs if costs is not None else CostModel()
        inst = instrument if instrument is not None else XAUUSD
        s = np.where(np.isfinite(spr.to_numpy()) & (spr.to_numpy() >= 0), spr.to_numpy(), 0.0)
        eff = np.maximum(s * float(cm.spread_multiplier), float(cm.min_spread))
        r = rng.to_numpy(dtype=float)
        r = np.where(np.isfinite(r) & (r >= 0), r, 0.0)
        slip = float(cm.slippage_fixed) + float(cm.slippage_range_frac) * r
        comm = float(cm.commission(1.0, instrument=inst)) / float(inst.contract_size)
        k = 0.5 * eff + slip + comm                                  # USD/oz per fill
        with np.errstate(invalid="ignore", divide="ignore"):
            c = k / (px.to_numpy() * vol.to_numpy())
        c = np.where(np.isfinite(c) & (c >= 0), c, np.nan) * self.cost_multiplier
        return pd.Series(c, index=idx, name="cost_per_turnover")

    def fit(self, forecasts: pd.DataFrame, close: pd.Series, *, bars: pd.DataFrame | None = None,
            spread: pd.Series | None = None, costs: Any = None, instrument: Any = None,
            cost_per_turnover: pd.Series | float | None = None) -> ForecastCombiner:
        """Learn weights and FDM from TRAINING forecasts and the matching close prices.

        Cost information (optional, all restricted to the forecasts' index): ``bars`` or
        ``spread`` with the ``costs`` model/``instrument`` -> estimated ``c_t``; or an explicit
        ``cost_per_turnover`` (scalar or series in unit-vol units per unit of |Δforecast|),
        which takes precedence. Without any, streams are scored gross (noted).
        """
        cols = [str(c) for c in forecasts.columns] if isinstance(forecasts, pd.DataFrame) else []
        if len(set(cols)) != len(cols):
            raise ValueError("forecast column names must be unique")
        if cost_per_turnover is not None:
            cost = self._as_cost_series(cost_per_turnover, forecasts.index)
            basis = "net of explicit cost_per_turnover"
        elif (bars is not None or spread is not None) and self.cost_multiplier > 0:
            cost = self.turnover_cost(forecasts, close, bars=bars, spread=spread, costs=costs,
                                      instrument=instrument)
            basis = ("net of estimated costs c_t=(spread_eff/2+slippage+commission)/(close*vol)"
                     + (f" x{self.cost_multiplier:g}" if self.cost_multiplier != 1.0 else ""))
        else:
            cost = None
            basis = "gross (no cost information passed to fit)"
        gross, net, turnover = self._streams(forecasts, close, cost)
        streams = gross if net is None else net
        n_obs = len(streams)
        if n_obs < 30:
            raise ValueError(f"not enough training observations to fit the combiner ({n_obs})")
        self.notes_ = []
        self.columns_ = list(forecasts.columns)
        self.train_start_ = forecasts.index[0] if len(forecasts) else None
        self.train_end_ = forecasts.index[-1] if len(forecasts) else None
        self.cost_basis_ = basis
        if cost is None:
            self.notes_.append("scored on GROSS unit-vol streams (no spread/cost information given)")

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
        g = gross.to_numpy()
        g_sd = g.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe_gross = np.where(g_sd > 1e-12, g.mean(axis=0) / np.where(g_sd > 1e-12, g_sd, 1.0), 0.0) \
                * math.sqrt(bpy)
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
            if self.allow_unallocated:
                pos = active & (raw > 0)
                if pos.any():
                    w[pos] = cap_weights(raw[pos], cap, allow_unallocated=True)
                unalloc = 1.0 - float(w.sum())
                if unalloc > 1e-9:
                    zero = [c for c, a, p in zip(self.columns_, active, pos, strict=True) if a and not p]
                    msg = (f"{unalloc:.1%} of the risk budget left UNALLOCATED: max_weight={cap:.3f} cannot be "
                           f"met by the {int(pos.sum())} positively scored strategies"
                           + (f"; zero weight (non-positive net/shrunk Sharpe): {zero}" if zero else ""))
                    self.notes_.append(msg)
                    logger.info("ForecastCombiner: %s", msg)
            else:
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
            # Normalise by the allocated mass: inactive (always-zero) forecasts must not inflate
            # it, and unallocated weight must not be scaled back up.
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
        self.train_sharpe_gross_ = pd.Series(sharpe_gross, index=self.columns_, name="train_sharpe_gross")
        # Per-bar moments of the (net) unit-vol stream (in units of one bar's volatility; the
        # std is ~ the RMS forecast when forecasts and returns are independent).
        self.train_vol_ = pd.Series(sd, index=self.columns_, name="train_stream_std")
        self.train_mean_ = pd.Series(mean, index=self.columns_, name="train_stream_mean")
        if turnover is not None:
            self.train_turnover_ = pd.Series(turnover.to_numpy().mean(axis=0), index=self.columns_,
                                             name="train_turnover")
            self.train_cost_ = pd.Series((g - u).mean(axis=0), index=self.columns_, name="train_cost_per_bar")
            c_valid = cost.reindex(streams.index)
            self.avg_cost_per_turnover_ = float(c_valid.mean()) if c_valid.notna().any() else math.nan
        else:
            self.train_turnover_ = None
            self.train_cost_ = None
            self.avg_cost_per_turnover_ = math.nan
        self.avg_abs_forecast_ = pd.Series(np.abs(f_train.to_numpy()).mean(axis=0), index=self.columns_)
        self.corr_ = pd.DataFrame(corr, index=self.columns_, columns=self.columns_)
        self.stream_corr_ = streams.corr()
        self.n_obs_ = int(n_obs)
        logger.info("ForecastCombiner(%s) fitted on %d bars (%s): weights=%s fdm=%.3f", self.method, n_obs,
                    "net" if cost is not None else "gross", np.round(w, 4).tolist(), self.fdm_)
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
            if self.allow_unallocated:
                # A strategy with a non-positive NET Sharpe is expected to lose money after
                # costs: it gets no weight, and it does not enter the shrinkage target either
                # (one heavy loser must not drag the candidates' shrunk Sharpes below zero).
                eligible = active & (sr > 0)
                if eligible.any():
                    shrunk = (1.0 - self.shrinkage) * sr + self.shrinkage * sr[eligible].mean()
                    raw = np.where(eligible, shrunk * inv_vol, 0.0)
                else:
                    msg = ("no strategy with a positive net Sharpe in train: all weights 0, the combined "
                           "book stays flat")
                    self.notes_.append(msg)
                    logger.warning("ForecastCombiner: %s", msg)
            else:
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

    def fit_combine(self, forecasts: pd.DataFrame, close: pd.Series, **cost_kwargs: Any) -> pd.Series:
        """Fit and combine on the SAME (training) data — in-sample, for diagnostics only."""
        return self.fit(forecasts, close, **cost_kwargs).combine(forecasts)

    def explain(self) -> dict:
        """JSON-friendly summary for reports and LLM agents."""
        if self.weights_ is None:
            return {"fitted": False, "method": self.method}

        def _d(s: pd.Series | None) -> dict[str, float]:
            return {} if s is None else {str(k): float(v) for k, v in s.items()}

        off = self.corr_.to_numpy()[~np.eye(len(self.columns_), dtype=bool)] if self.corr_ is not None else []
        wsum = float(self.weights_.sum())
        # getattr: combiners pickled by an earlier version lack the cost-aware attributes
        avg_c = getattr(self, "avg_cost_per_turnover_", math.nan)
        return {
            "fitted": True,
            "method": self.method,
            "weights": _d(self.weights_),
            "weights_sum": wsum,
            "unallocated": max(0.0, 1.0 - wsum),
            "allow_unallocated": bool(getattr(self, "allow_unallocated", False)),
            "train_sharpe": _d(self.train_sharpe_),
            "train_sharpe_gross": _d(getattr(self, "train_sharpe_gross_", None)),
            "sharpe_basis": "per-bar unit-vol stream f[t]*r[t+1]/vol[t] - c_t*|f[t]-f[t-1]| on train, annualised",
            "cost_basis": getattr(self, "cost_basis_", "gross"),
            "avg_cost_per_turnover": float(avg_c) if avg_c is not None and math.isfinite(avg_c) else None,
            "train_turnover": _d(getattr(self, "train_turnover_", None)),
            "train_cost_per_bar": _d(getattr(self, "train_cost_", None)),
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
            f"max_weight={self.max_weight}, fdm_cap={self.fdm_cap}, "
            f"allow_unallocated={getattr(self, 'allow_unallocated', False)})"
        )
