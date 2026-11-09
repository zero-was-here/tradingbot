"""Regime-conditioned trend following: ``regime_switch`` (trainable).

Economic rationale
------------------
Trend following on gold was flat to negative through the choppy 2016-2024 sample and
strongly positive in the 2025-26 trend (docs/RESULTS.md): its returns are *regime
dependent*. Markets alternate between calm, persistent phases and turbulent, mean-reverting
ones whose volatility differs by a factor of 2-4 (Hamilton 1989; Ang & Timmermann 2012).
Trend rules make their money when moves are orderly and lose it in violent, whipsawing
markets where stops and reversals pile up (Daniel & Moskowitz 2016 "Momentum crashes";
Garg, Goulding, Harvey & Mazzoleni 2021 "Momentum turning points"). A two-state Gaussian
hidden Markov model of returns separates the two phases without hand-picked thresholds;
conditioning the trend forecast on the *filtered* probability of the calm state keeps the
trend exposure when it is likely to work and steps aside in the chaotic state. This is a
risk-management overlay of a known premium (as in Nystrup, Madsen & Lindström 2017's
regime-based asset allocation), not a new source of alpha.

Method
------
``fit`` (TRAINING bars only)
    Daily log returns in percent (from the bars, resampled point-in-time with
    :func:`aurum.data.resample.resample_bars`; or per-bar returns for daily bars / with
    ``returns="bar"``) are fitted by Baum-Welch EM (:class:`aurum.models.regime.GaussianHMM`,
    ``n_states=2``). States are ordered by variance: state 0 is calm, the last state the
    high-volatility "chaotic" regime. A trainable base strategy is fitted on the same slice.

``generate``
    The forward filter gives ``p_t = P(s_t = chaotic | r_0..r_t)`` — causal by
    construction (no smoother, no Viterbi). Each daily probability becomes usable at that
    day's ``available_at`` and is mapped onto the trading bars with
    :func:`aurum.data.pit.asof_join`. With ``p_ok = 1 - p_t`` (= ``P(calm)`` for two
    states)::

        scale_t = clip((p_ok_t - p_flat) / (p_full - p_flat), 0, 1)
        f_t     = base_forecast_t * scale_t

    so the forecast is flat when the chaotic state is at least as likely as ``1 - p_flat``
    (default: more likely than not), passes the base forecast in full when the calm state
    has probability >= ``p_full`` and is linear in between (no jump at the threshold).

The base forecast defaults to ``tsmom`` with its own defaults (``base`` / ``base_params``
select any registered non-recursive strategy). Expose the regime for charts with
:meth:`RegimeSwitch.regime_probabilities` (per bar) or
:meth:`RegimeSwitch.filtered_probabilities` (per return observation).

Point-in-time notes
-------------------
* The filtered probability of day ``d`` uses returns up to ``d``; the in-progress day is
  never emitted by ``resample_bars`` (or is emitted with an ``available_at`` after every
  intraday decision of that day), so intraday bars only see completed days.
* ``fit`` uses the smoother inside the training window — fine for estimation, never used
  for decisions (SPEC §5).
* The filter forgets its initial condition geometrically (rate = the second eigenvalue of
  the transition matrix), so a live run on a shorter history agrees with research after a
  few expected regime durations; ``warmup`` (bars) defaults to the base strategy's warm-up
  (``tsmom``: 1,441 H1 bars ≈ 3 months), far longer than that.

References
----------
* Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time
  Series and the Business Cycle". Econometrica 57(2).
* Rabiner, L. R. (1989). "A tutorial on hidden Markov models". Proc. IEEE 77(2).
* Ang, A. & Timmermann, A. (2012). "Regime Changes and Financial Markets". Annual Review of
  Financial Economics 4.
* Nystrup, P., Madsen, H. & Lindström, E. (2017). "Long memory of financial time series and
  hidden Markov models with time-varying parameters". J. Forecasting 36(8); and Nystrup et
  al. (2015) "Regime-Based Versus Static Asset Allocation". JPM 42(1).
* Daniel, K. & Moskowitz, T. (2016). "Momentum crashes". JFE 122(2).
* Garg, A., Goulding, C., Harvey, C. R. & Mazzoleni, M. (2021). "Momentum Turning Points".
  JFE 141(3).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.data.pit import asof_join
from aurum.data.resample import resample_bars
from aurum.data.schema import BAR_COLS
from aurum.features.volatility import bar_minutes
from aurum.models.regime import GaussianHMM
from aurum.strategies.base import Strategy, get_strategy, register_strategy
from aurum.strategies.trend import check_params

logger = logging.getLogger(__name__)

__all__ = ["RegimeSwitch", "regime_scale"]

_RETURN_MODES = ("auto", "daily", "bar")


def regime_scale(p_ok: np.ndarray | pd.Series, p_flat: float, p_full: float) -> np.ndarray:
    """Piecewise-linear map of ``P(not chaotic)`` to a forecast multiplier in [0, 1].

    0 at or below ``p_flat``, 1 at or above ``p_full``, linear in between; NaN stays NaN.
    """
    p = np.asarray(p_ok, dtype=float)
    return np.clip((p - float(p_flat)) / (float(p_full) - float(p_flat)), 0.0, 1.0)


@register_strategy
class RegimeSwitch(Strategy):
    """Base trend forecast scaled by the HMM-filtered probability of the calm regime.

    See the module docstring for the rationale, the method and references.

    Parameters
    ----------
    base, base_params : registry name and parameters of the base forecast (``tsmom``, {}).
    returns     : ``"daily"`` (resample to D1 point-in-time, recommended for intraday bars),
                  ``"bar"`` (per-bar returns) or ``"auto"`` (daily below D1, bar otherwise).
    daily_anchor_hour_utc : D1 bucket boundary for ``returns="daily"`` (22 = New York close).
    n_states    : HMM states (default 2); the highest-variance state is "chaotic".
    p_flat, p_full : thresholds of :func:`regime_scale` on ``P(not chaotic)`` (0.5 / 0.9).
    min_train_obs : minimum finite training returns to fit the HMM; with fewer the strategy
                  abstains (forecast 0) and logs a warning (default 60).
    seed, n_init : HMM restarts (deterministic).
    warmup      : warm-up in bars (default ``None`` = the base strategy's warm-up).
    """

    name = "regime_switch"
    description = ("Trend forecast (tsmom) scaled by a 2-state Gaussian HMM's causal filtered "
                   "P(calm) on daily returns; flat in the high-vol chaotic regime. Trainable.")
    trainable = True

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "base": "tsmom",
            "base_params": {},
            "returns": "auto",
            "daily_anchor_hour_utc": 22,
            "n_states": 2,
            "p_flat": 0.5,
            "p_full": 0.9,
            "min_train_obs": 60,
            "seed": 0,
            "n_init": 3,
            "warmup": None,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        if not isinstance(p["base"], str) or not p["base"]:
            raise ValueError("base must be a registered strategy name")
        if p["base"] == self.name:
            raise ValueError("regime_switch cannot use itself as its base strategy")
        p["base_params"] = dict(p["base_params"] or {})
        if p["returns"] not in _RETURN_MODES:
            raise ValueError(f"returns must be one of {_RETURN_MODES}, got {p['returns']!r}")
        p["daily_anchor_hour_utc"] = int(p["daily_anchor_hour_utc"])
        if not 0 <= p["daily_anchor_hour_utc"] < 24:
            raise ValueError("daily_anchor_hour_utc must be in [0, 24)")
        p["n_states"] = int(p["n_states"])
        if p["n_states"] < 2:
            raise ValueError("n_states must be >= 2")
        p["p_flat"], p["p_full"] = float(p["p_flat"]), float(p["p_full"])
        if not 0.0 <= p["p_flat"] < p["p_full"] <= 1.0:
            raise ValueError("need 0 <= p_flat < p_full <= 1")
        p["min_train_obs"] = int(p["min_train_obs"])
        if p["min_train_obs"] < 5 * p["n_states"]:
            raise ValueError(f"min_train_obs must be >= {5 * p['n_states']} (5 per state)")
        p["seed"], p["n_init"] = int(p["seed"]), int(p["n_init"])
        if p["warmup"] is not None:
            p["warmup"] = int(p["warmup"])
            if p["warmup"] < 0:
                raise ValueError("warmup must be >= 0 or None")
        self.base_: Strategy = get_strategy(p["base"], **p["base_params"])
        self.hmm_: GaussianHMM | None = None
        self.fit_note_: str = ""

    # ------------------------------------------------------------------------------------
    @property
    def warmup_bars(self) -> int:
        w = self.params["warmup"]
        return int(self.base_.warmup_bars if w is None else max(w, 0))

    @property
    def label_horizon(self) -> int:
        """Purge horizon of a trainable base strategy (the HMM itself labels nothing)."""
        if not getattr(self.base_, "trainable", False):
            return 0
        from aurum.research.walkforward import label_horizon

        return int(label_horizon(self.base_))

    @property
    def fit_history_bars(self) -> int:
        return int(self.base_.fit_history_bars)

    def return_mode(self, bars: pd.DataFrame) -> str:
        """``"daily"`` or ``"bar"`` for these bars (``auto`` decides from the bar length,
        which :func:`bar_minutes` reads from ``attrs`` or the PREFIX of the frame)."""
        mode = self.params["returns"]
        if mode != "auto":
            return mode
        minutes = bar_minutes(bars) if len(bars) else float("nan")
        return "bar" if (math.isfinite(minutes) and minutes >= 1440) else "daily"

    def observations(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return observations the HMM sees: ``r`` (log return, percent) + ``available_at``.

        Daily mode resamples to complete D1 buckets point-in-time (``resample_bars``); a
        day's return is usable from that day's ``available_at`` (the bucket end).
        """
        if len(bars) == 0:
            return pd.DataFrame({"r": pd.Series(dtype=float),
                                 "available_at": pd.Series(dtype="datetime64[ns, UTC]")})
        if self.return_mode(bars) == "daily":
            src = resample_bars(bars[BAR_COLS], "D1",
                                daily_anchor_hour_utc=self.params["daily_anchor_hour_utc"])
        else:
            src = bars
        close = src["close"].to_numpy(dtype=float)
        r = np.full(len(close), np.nan)
        if len(close) > 1:
            r[1:] = 100.0 * np.diff(np.log(close))
        return pd.DataFrame({"r": r, "available_at": src["available_at"].to_numpy()}, index=src.index)

    # ------------------------------------------------------------------------------------
    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> RegimeSwitch:
        """Fit the HMM on the TRAINING slice's returns (and a trainable base strategy)."""
        p = self.params
        if getattr(self.base_, "trainable", False):
            self.base_.fit(md, features)
        obs = self.observations(md.bars)
        x = obs["r"].to_numpy(dtype=float)
        n_ok = int(np.isfinite(x).sum())
        self.hmm_ = None
        if n_ok < p["min_train_obs"]:
            self.fit_note_ = (f"only {n_ok} finite training returns (< min_train_obs="
                              f"{p['min_train_obs']}): abstaining (forecast 0)")
            logger.warning("regime_switch: %s", self.fit_note_)
        else:
            try:
                self.hmm_ = GaussianHMM(n_states=p["n_states"], seed=p["seed"], n_init=p["n_init"]).fit(x)
                self.fit_note_ = f"HMM fitted on {n_ok} {self.return_mode(md.bars)} returns"
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                self.fit_note_ = f"HMM fit failed ({exc}): abstaining (forecast 0)"
                logger.warning("regime_switch: %s", self.fit_note_)
        self.is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("regime_switch must be fit() on training data before generate()")

    def filtered_probabilities(self, md: MarketData) -> pd.DataFrame:
        """Causal filtered state probabilities per return observation.

        Columns ``p_state_0..K-1`` (0 = calm), ``p_calm`` (= ``p_state_0``), ``p_chaotic``
        (highest-variance state), ``p_ok`` (``1 - p_chaotic``), ``scale`` and
        ``available_at``. Empty (all-NaN probabilities) when the HMM could not be fitted.
        """
        self._check_fitted()
        obs = self.observations(md.bars)
        k = self.params["n_states"]
        cols = [f"p_state_{i}" for i in range(k)]
        if self.hmm_ is None or len(obs) == 0:
            out = pd.DataFrame(np.nan, index=obs.index, columns=cols)
        else:
            out = self.hmm_.filter(obs["r"].to_numpy(dtype=float))
            out.index = obs.index
        out["p_calm"] = out["p_state_0"]
        out["p_chaotic"] = out[f"p_state_{k - 1}"]
        out["p_ok"] = 1.0 - out["p_chaotic"]
        out["scale"] = regime_scale(out["p_ok"], self.params["p_flat"], self.params["p_full"])
        out["available_at"] = obs["available_at"].to_numpy()
        return out

    def regime_probabilities(self, md: MarketData) -> pd.DataFrame:
        """:meth:`filtered_probabilities` mapped onto ``md.bars`` point-in-time (asof on
        ``available_at``). NaN before the first usable observation."""
        filt = self.filtered_probabilities(md)
        cols = [c for c in filt.columns if c != "available_at"]
        if len(md.bars) == 0:
            return pd.DataFrame(columns=cols, index=md.bars.index, dtype=float)
        return asof_join(md.bars["available_at"], filt, columns=cols)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        self._check_fitted()
        bars = md.bars
        if self.hmm_ is None:
            return self._finalize(pd.Series(0.0, index=bars.index), bars.index)
        base = self.base_.generate(md, features).reindex(bars.index).to_numpy(dtype=float)
        scale = self.regime_probabilities(md)["scale"].to_numpy(dtype=float)
        return self._finalize(pd.Series(base * scale, index=bars.index), bars.index)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly description of the fitted regime model (for reports / agents)."""
        out: dict[str, Any] = {"base": self.params["base"], "returns": self.params["returns"],
                               "p_flat": self.params["p_flat"], "p_full": self.params["p_full"],
                               "fit_note": self.fit_note_, "hmm": None}
        if self.hmm_ is not None:
            out["hmm"] = self.hmm_.summary()
        return out
