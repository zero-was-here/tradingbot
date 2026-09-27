"""Machine-learning strategies: ``ml_gbm`` and ``meta_label`` (SPEC §6).

Both are *trainable*: ``fit(md)`` learns from the TRAINING slice it is given and
``generate(md)`` applies the frozen model row by row to any (later, longer) history.

``ml_gbm`` - direction classifier
---------------------------------
Economic rationale: gold's short-horizon direction is close to a random walk, but weak,
state-dependent regularities exist (trend persistence in low-vol regimes, mean reversion
after stretched moves, session and event effects, macro co-movement). A gradient-boosted
tree ensemble can combine many such weak, non-linear conditional effects that single-rule
strategies express one at a time. Everything that makes ML dangerous in finance is handled
explicitly (López de Prado, *Advances in Financial Machine Learning* (AFML), 2018):

* **Targets** - triple-barrier labels (AFML ch. 3) with volatility-scaled barriers, or
  fixed-horizon vol-normalised forward returns; computed ONLY from the training bars and
  only for events whose outcome is decided before the training slice ends; the last
  ``max_holding_bars`` events are purged regardless (no selection bias toward fast exits).
* **Overlap** - labels overlap in time, so observations are weighted by average uniqueness
  (AFML ch. 4), optionally times return attribution and time decay.
* **Validation** - a time-ordered validation TAIL of the training events (never shuffled),
  purged so that no fit-set label resolves inside it, drives early stopping (best
  boosting iteration by weighted log-loss), probability calibration (Platt or isotonic,
  monotone non-decreasing - calibration can shrink a signal but never flip it) and
  permutation feature importance (AUC drop, Breiman 2001; stored for reports/agents).
* **Class balance** - by default the classes are re-weighted to balance, separately within
  the fit set and the validation tail, so ``p = 0.5`` means "no conditional information"
  and neither the model nor the calibrator can simply bet on a period's drift.
* **Non-stationary features** - level-type features (e.g. 250-day z-scores of macro
  levels) can act as *time stamps* that let trees memorise regimes of overlapping labels
  (high in-sample, ~0.5 validation AUC). ``drop_features`` (regex list) removes such
  columns; the permutation importances in ``fit_report_``/``explain()`` reveal them.

Forecast: ``f = 2 p - 1`` (calibrated ``p``), then a continuous dead-zone
``sign(f) * max(|f| - dz, 0) / (1 - dz)`` (no churn on coin flips), then Carver-style
scaling (R. Carver, *Systematic Trading*, 2015, ch. 7): a scalar fixed at fit time makes
the average absolute forecast on the validation tail equal ``target_abs_forecast`` (0.5 -
Carver's 10 on a +/-20 scale), then clip to [-1, 1]. Because that scaling would blow pure
noise up to full size, a **skill gate** (``skill_gate_z``) sets the scalar to 0 - the
strategy stays flat - unless the validation AUC exceeds 0.5 by ``skill_gate_z`` standard
errors, with the standard error computed from the uniqueness-adjusted effective sample size
(Hanley & McNeil 1982). Both strategies apply the gate.

``meta_label`` - meta-labelling a primary strategy
--------------------------------------------------
Economic rationale (AFML 3.6): a simple primary model (default ``tsmom``) decides the
SIDE; a secondary ML model learns WHEN the primary is likely to be right (e.g. trend
signals in trending, low-noise regimes) and sizes the bet. Label = did the primary's side
hit its profit-take before its stop (side-aware triple barrier)? Timeouts at the vertical
barrier count as a success when the bet's return is positive (``meta_vertical=
"return_sign"``, AFML ``getBins``). The strict alternative ``"fail"`` (timeout = 0) is
available but NOT the default: it makes the label partly a *barrier-touch* indicator,
which is predictable from the volatility state and from weekend gaps even on a random walk
(validation AUC ~0.64 on a synthetic GBM path during development) without any directional
edge. The
classifier predicts ``p = P(success)`` and the bet size follows AFML ch. 10:
``z = (p - 1/2) / sqrt(p (1-p))``,
``m = 2 Phi(z) - 1``, zero when ``p < p_threshold`` (never reverses the primary),
optionally Carver-scaled to a target average size, then discretised to ``step_size``
(AFML snippet 10.3) to avoid over-trading on tiny probability changes.
``forecast = sign(primary) * m``.

Point-in-time contract
----------------------
* ``fit`` reads only ``md`` / ``features`` passed in (rows of ``features`` outside
  ``md.bars.index`` are ignored). Labels (forward-looking) live only inside ``fit``.
* ``features`` (optional, both methods) is any numeric frame indexed like the bars - raw
  ``FeaturePipeline.compute`` output or already-transformed fold features from the
  walk-forward engine; the strategy fits its own robust scaler on the training rows either
  way (harmless on pre-scaled inputs; trees are scale-invariant). A strategy fitted on
  external features must be given them at ``generate`` time too.
* ``generate`` recomputes causal features on the ``md`` it is given (or uses ``features``),
  then applies the FROZEN scaler, model, calibrator and scalar row by row - forecast ``t``
  depends only on information at the close of ``t``.
* External ``features`` keep their leading warm-up NaNs as "not yet defined": a row with
  a column still inside its warm-up is never a training event and gets a zero forecast
  (NaNs after a column's first valid value become the neutral 0, as in the pipeline).
* EWM features depend weakly on where history starts; compute live features on a long
  history (>= 3 x ``warmup_bars``) for research/live parity.
* **Trainable primaries** (``primary`` of ``meta_label``, ``primary_features`` of
  ``ml_gbm``) are fitted on the first ``primary_oos_frac`` of the training slice only, and
  the ML model learns from events after that point. Their in-sample forecasts would
  otherwise be over-fitted (e.g. hour-of-week means estimated on the very bars being
  labelled), and the second-stage model would learn to trust them far more than their
  out-of-sample quality warrants - the stacking leak of Wolpert (1992), "Stacked
  generalization". ``primary_oos_frac=0`` restores in-sample fitting (logged as a warning).
* ``fit`` is transactional: everything is built on a fresh instance and committed only on
  success, so a failed refit leaves the previous fit intact and self-consistent (never a
  new scaler with an old model).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

from aurum.core.types import MarketData
from aurum.features.pipeline import CANONICAL_GROUP_ORDER, FeaturePipeline
from aurum.labels.triple_barrier import (
    average_uniqueness,
    cusum_filter,
    drop_label_tail,
    ewm_vol,
    fixed_horizon_labels,
    meta_labels,
    return_attribution_weights,
    time_decay_weights,
    triple_barrier_labels,
    uniqueness_weights,
)
from aurum.strategies.base import Strategy, get_strategy, register_strategy

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PRICE_GROUPS",
    "CalibratedGBM",
    "MLGBMStrategy",
    "MetaLabelStrategy",
    "apply_dead_zone",
    "bet_size_from_probability",
    "carver_scalar",
    "discretize_bet",
]

#: Price/time-only feature groups used by default; ``macro`` / ``calendar`` are added
#: automatically when the fit-time ``MarketData`` carries macro frames / events.
DEFAULT_PRICE_GROUPS: tuple[str, ...] = (
    "returns", "trend", "momentum", "meanrev", "range", "volatility", "microstructure",
    "session", "mtf", "regime",
)
_EPS = 1e-6
#: Warm-up given to the scaling-only pipeline of EXTERNAL features: never "past warm-up",
#: so leading NaNs (not yet defined) stay NaN until each column's first valid value.
_EXTERNAL_WARMUP = 2**62


# ---------------------------------------------------------------------------------------
# forecast mapping helpers (pure functions, unit-tested)
# ---------------------------------------------------------------------------------------
def apply_dead_zone(f: np.ndarray | float, dead_zone: float) -> np.ndarray:
    """Continuous dead-zone: ``sign(f) * max(|f| - dz, 0) / (1 - dz)``.

    Values within ``dz`` of zero become exactly 0 and the map stays continuous and maps
    +/-1 to +/-1, so crossing the threshold does not cause a jump in position.
    """
    if not 0.0 <= dead_zone < 1.0:
        raise ValueError("dead_zone must be in [0, 1)")
    x = np.asarray(f, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - dead_zone, 0.0) / (1.0 - dead_zone)


def carver_scalar(f: np.ndarray, target_abs: float | None, *, max_scalar: float = 50.0) -> float:
    """Scalar making ``mean(|scalar * f|) == target_abs`` (Carver 2015, ch. 7), capped.

    Returns 1.0 when ``target_abs`` is None and 0.0 when ``f`` is identically zero (the
    model expresses no view on the calibration sample - nothing to scale).
    """
    if target_abs is None:
        return 1.0
    if target_abs <= 0:
        raise ValueError("target_abs must be > 0 or None")
    x = np.asarray(f, dtype=float)
    x = x[np.isfinite(x)]
    avg = float(np.mean(np.abs(x))) if x.size else 0.0
    if avg <= 1e-12:
        return 0.0
    return float(min(max_scalar, target_abs / avg))


def bet_size_from_probability(p: np.ndarray | float, *, num_classes: int = 2) -> np.ndarray:
    """AFML snippet 10.1: ``m = 2 Phi(z) - 1`` with ``z = (p - 1/K) / sqrt(p (1 - p))``.

    The test statistic ``z`` measures how far the predicted probability is from the
    no-information level ``1/K``; mapping it through the normal CDF gives a bet size in
    (-1, 1) that grows quickly for confident predictions and is ~0 near ``p = 1/K``.
    """
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    z = (q - 1.0 / num_classes) / np.sqrt(q * (1.0 - q))
    return 2.0 * norm.cdf(z) - 1.0


def discretize_bet(m: np.ndarray | float, step: float) -> np.ndarray:
    """AFML snippet 10.3: round bet sizes to multiples of ``step`` (0 disables), clip to [-1, 1]."""
    x = np.asarray(m, dtype=float)
    if step and step > 0:
        x = np.round(x / step) * step
    return np.clip(x, -1.0, 1.0)


def _weighted_logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    ll = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
    return float(np.sum(w * ll) / np.sum(w))


def _weighted_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


# ---------------------------------------------------------------------------------------
# classifier with a purged, time-ordered validation tail
# ---------------------------------------------------------------------------------------
class CalibratedGBM:
    """``HistGradientBoostingClassifier`` + time-ordered validation tail + calibration.

    The last ``val_frac`` of the (chronologically sorted) training events form the
    validation tail. Fit-set events whose label resolves at or after the first validation
    event (``t1_idx >= t_idx[val_start] - embargo``) are purged, so the tail is a genuine
    pseudo-out-of-sample block. On the tail we choose the boosting iteration count
    (weighted log-loss), calibrate probabilities, and (optionally) compute permutation
    importances. Deterministic for a fixed ``seed``.
    """

    DEFAULT_MODEL: dict[str, Any] = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "max_depth": 4,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "max_features": 0.5,
        "max_bins": 255,
    }
    RESERVED_MODEL_KEYS: frozenset[str] = frozenset(
        {"class_weight", "early_stopping", "random_state", "validation_fraction", "n_iter_no_change"})

    def __init__(
        self,
        *,
        model: dict[str, Any] | None = None,
        val_frac: float = 0.2,
        embargo_bars: int = 0,
        early_stopping: bool = True,
        calibration: str = "platt",
        class_weight: str | None = "balanced",
        min_events: int = 200,
        seed: int = 0,
        n_threads: int | None = None,
    ) -> None:
        if not 0.0 < val_frac < 0.9:
            raise ValueError("val_frac must be in (0, 0.9)")
        if calibration not in ("platt", "isotonic", "none"):
            raise ValueError("calibration must be 'platt', 'isotonic' or 'none'")
        if class_weight not in ("balanced", None):
            raise ValueError("class_weight must be 'balanced' or None")
        reserved = sorted(set(model or {}) & self.RESERVED_MODEL_KEYS)
        if reserved:
            # class_weight would silently re-weight on top of our fit/validation balancing;
            # early stopping / seeding are handled here (time-ordered tail, ``seed``)
            raise ValueError(f"model params {reserved} are managed by CalibratedGBM "
                             "(use class_weight=, early_stopping=, seed=)")
        self.model_params = {**self.DEFAULT_MODEL, **(model or {})}
        self.val_frac = float(val_frac)
        self.embargo_bars = int(embargo_bars)
        self.early_stopping = bool(early_stopping)
        self.calibration = calibration
        self.class_weight = class_weight
        self.min_events = int(min_events)
        self.seed = int(seed)
        self.n_threads = n_threads
        self.model_: Any = None
        self.calibrator_: dict[str, Any] = {"kind": "none"}
        self.feature_names_: list[str] = []
        self.report_: dict[str, Any] = {}
        self.feature_importances_: pd.Series | None = None
        self.feature_importances_std_: pd.Series | None = None
        # validation tail kept between fit() and release_validation()
        self._val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ---- threading ----------------------------------------------------------------------
    def _threads(self):
        if self.n_threads is None:
            return nullcontext()
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(self.n_threads), user_api="openmp")

    # ---- fit ------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray, t_idx: np.ndarray,
            t1_idx: np.ndarray) -> CalibratedGBM:
        from sklearn.ensemble import HistGradientBoostingClassifier

        y = np.asarray(y, dtype=float)
        w = np.asarray(sample_weight, dtype=float)
        t0 = np.asarray(t_idx, dtype=np.int64)
        t1 = np.asarray(t1_idx, dtype=np.int64)
        if not (len(X) == len(y) == len(w) == len(t0) == len(t1)):
            raise ValueError("X, y, sample_weight, t_idx, t1_idx must have equal length")
        if np.any(np.diff(t0) < 0):
            raise ValueError("events must be in chronological order (sorted t_idx)")
        n = len(y)
        n_val = int(round(self.val_frac * n))
        if n - n_val < self.min_events or n_val < max(20, self.min_events // 10):
            raise ValueError(f"not enough training events ({n}; need >= {self.min_events} to fit "
                             f"and a validation tail)")
        pos = np.arange(n)
        val_mask = pos >= n - n_val
        val_start = int(t0[n - n_val])
        fit_mask = ~val_mask & (t1 < val_start - self.embargo_bars)
        n_purged = int((~val_mask).sum() - fit_mask.sum())
        y_fit, y_val = y[fit_mask], y[val_mask]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            raise ValueError("both classes must be present in the fit set and the validation tail")
        if not (np.all(np.isfinite(w)) and np.all(w >= 0) and w[fit_mask].sum() > 0
                and w[val_mask].sum() > 0):
            raise ValueError("sample weights must be finite, >= 0 and have a positive sum in both "
                             "the fit set and the validation tail (is time_decay zeroing the fit set?)")
        # Class balance is applied SEPARATELY within the fit set and within the validation
        # tail: balancing val with fit-set shares would let the calibrator's intercept
        # re-learn the tail period's unconditional drift (e.g. a 2016-17 gold rally after a
        # 2013-15 bear market), turning the "balanced" probability into a constant bet.
        cw = np.ones(n)
        if self.class_weight == "balanced":
            for part in (fit_mask, val_mask):
                wp, yp = w[part], y[part]
                tot = wp.sum()
                for cls in (0.0, 1.0):
                    share = wp[yp == cls].sum() / tot
                    cw[part & (y == cls)] = 0.5 / max(share, 1e-12)
        w_all = w * cw
        Xa = np.ascontiguousarray(X.to_numpy(dtype=float))
        X_fit, X_val = Xa[fit_mask], Xa[val_mask]
        w_fit, w_val = w_all[fit_mask], w_all[val_mask]
        w_fit = w_fit / w_fit.mean()
        w_val = w_val / w_val.mean()
        self.feature_names_ = [str(c) for c in X.columns]

        params = dict(self.model_params)
        max_iter = int(params.pop("max_iter"))

        def make(n_iter: int) -> Any:
            return HistGradientBoostingClassifier(max_iter=n_iter, early_stopping=False,
                                                  random_state=self.seed, **params)

        base_rate = float(np.sum(w_val * y_val) / np.sum(w_val))
        base_loss = _weighted_logloss(y_val, np.full(len(y_val), base_rate), w_val)
        with self._threads():
            model = make(max_iter).fit(X_fit, y_fit, sample_weight=w_fit)
            best_iter = max_iter
            losses: list[float] = []
            if self.early_stopping:
                for proba in model.staged_predict_proba(X_val):
                    losses.append(_weighted_logloss(y_val, proba[:, 1], w_val))
                best_iter = int(np.argmin(losses)) + 1
                if best_iter < max_iter:
                    # same seed and data -> identical first ``best_iter`` trees
                    model = make(best_iter).fit(X_fit, y_fit, sample_weight=w_fit)
            p_val_raw = model.predict_proba(X_val)[:, 1]
        self.model_ = model
        self.calibrator_ = self._fit_calibrator(p_val_raw, y_val, w_val)
        p_val = self._calibrate(p_val_raw)
        self._val = (X_val, y_val, w_val)
        with self._threads():
            p_fit_raw = model.predict_proba(X_fit)[:, 1]
        self.report_ = {
            "n_events": n,
            "n_fit": int(fit_mask.sum()),
            "n_val": int(n_val),
            "n_purged_before_val": n_purged,
            "fit_pos_rate": float(y_fit.mean()),
            "val_pos_rate": float(y_val.mean()),
            "best_iter": int(best_iter),
            "max_iter": max_iter,
            "fit_auc": _weighted_auc(y_fit, p_fit_raw, w_fit),
            "val_auc": _weighted_auc(y_val, p_val_raw, w_val),
            "val_logloss_base": base_loss,
            "val_logloss_model": _weighted_logloss(y_val, p_val_raw, w_val),
            "val_logloss_calibrated": _weighted_logloss(y_val, p_val, w_val),
            "val_p_mean": float(np.mean(p_val)),
            "val_p_std": float(np.std(p_val)),
            "calibration": self.calibrator_.get("kind"),
            "calibration_params": {k: v for k, v in self.calibrator_.items()
                                   if k in ("a", "b", "clamped")},
            "n_features": X.shape[1],
        }
        logger.info("CalibratedGBM: %d fit / %d val events, best_iter=%d, val AUC=%.4f, "
                    "logloss base=%.5f model=%.5f cal=%.5f", self.report_["n_fit"], n_val, best_iter,
                    self.report_["val_auc"], base_loss, self.report_["val_logloss_model"],
                    self.report_["val_logloss_calibrated"])
        return self

    # ---- calibration ------------------------------------------------------------------------
    def _fit_calibrator(self, p_raw: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, Any]:
        if self.calibration == "none":
            return {"kind": "none"}
        if self.calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, increasing=True,
                                     out_of_bounds="clip").fit(p_raw, y, sample_weight=w)
            return {"kind": "isotonic", "model": iso}
        # Platt scaling on the logit of the raw probability: p = sigmoid(a * logit(p_raw) + b),
        # a >= 0 (a negative slope would flip the model's ranking on a noisy tail).
        s = logit(np.clip(p_raw, _EPS, 1 - _EPS))
        wn = w / w.sum()

        def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = theta
            q = np.clip(expit(a * s + b), _EPS, 1 - _EPS)
            val = -np.sum(wn * (y * np.log(q) + (1 - y) * np.log(1 - q)))
            g = q - y
            return float(val), np.array([np.sum(wn * g * s), np.sum(wn * g)])

        res = minimize(nll, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 50.0), (-20.0, 20.0)])
        a, b = (float(v) for v in res.x)
        return {"kind": "platt", "a": a, "b": b, "clamped": bool(a <= 1e-9)}

    def _calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        kind = self.calibrator_.get("kind", "none")
        if kind == "isotonic":
            return np.asarray(self.calibrator_["model"].predict(p_raw), dtype=float)
        if kind == "platt":
            s = logit(np.clip(p_raw, _EPS, 1 - _EPS))
            return expit(self.calibrator_["a"] * s + self.calibrator_["b"])
        return np.asarray(p_raw, dtype=float)

    # ---- inference --------------------------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CalibratedGBM is not fitted")
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        if Xa.shape[0] == 0:
            return np.empty(0)
        with self._threads():
            return self.model_.predict_proba(Xa)[:, 1]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Calibrated ``P(y = 1)``."""
        return self._calibrate(self.predict_raw(X))

    def validation_proba(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(calibrated p, y, w) on the validation tail (only until ``release_validation``)."""
        if self._val is None:
            raise RuntimeError("validation tail not available (released or not fitted)")
        X_val, y_val, w_val = self._val
        return self.predict_proba(X_val), y_val, w_val

    # ---- importance -------------------------------------------------------------------------
    def compute_permutation_importance(self, *, n_repeats: int = 3, max_rows: int | None = 5000,
                                       seed: int | None = None) -> pd.Series:
        """Mean drop in validation AUC when each feature is permuted (Breiman 2001).

        Uses the raw model (calibration is monotone, so AUC is unaffected). Rows are a
        deterministic random subsample of the validation tail when it exceeds ``max_rows``.
        """
        from sklearn.inspection import permutation_importance

        if self._val is None:
            raise RuntimeError("validation tail not available (released or not fitted)")
        X_val, y_val, w_val = self._val
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if max_rows is not None and len(y_val) > max_rows:
            sel = np.sort(rng.choice(len(y_val), size=int(max_rows), replace=False))
            X_val, y_val, w_val = X_val[sel], y_val[sel], w_val[sel]
        with self._threads():
            res = permutation_importance(self.model_, X_val, y_val, scoring="roc_auc",
                                         n_repeats=int(n_repeats), random_state=int(rng.integers(2**31 - 1)),
                                         sample_weight=w_val)
        imp = pd.Series(res.importances_mean, index=self.feature_names_, name="importance")
        std = pd.Series(res.importances_std, index=self.feature_names_, name="importance_std")
        order = imp.sort_values(ascending=False).index
        self.feature_importances_ = imp.loc[order]
        self.feature_importances_std_ = std.loc[order]
        return self.feature_importances_

    def release_validation(self) -> None:
        """Free the validation arrays (keeps clones/pickles small)."""
        self._val = None


# ---------------------------------------------------------------------------------------
# shared machinery for the two strategies
# ---------------------------------------------------------------------------------------
def _make_strategy(spec: Any, params: dict[str, Any] | None = None) -> Strategy:
    """Instantiate a primary strategy from a name, ``(name, params)``, class or instance."""
    if isinstance(spec, Strategy):
        return spec.clone()
    if isinstance(spec, type) and issubclass(spec, Strategy):
        return spec(**(params or {}))
    if isinstance(spec, tuple | list) and len(spec) == 2 and isinstance(spec[0], str):
        return get_strategy(spec[0], **{**dict(spec[1]), **(params or {})})
    if isinstance(spec, str):
        try:
            return get_strategy(spec, **(params or {}))
        except KeyError as exc:
            raise KeyError(f"primary strategy {spec!r} is not registered (is its module "
                           f"implemented?): {exc}") from exc
    raise TypeError(f"cannot build a strategy from {spec!r}")


def _head(md: MarketData, n: int) -> MarketData:
    """The first ``n`` bars of ``md`` (macro/events untouched: consumers align them by
    ``available_at``, exactly as for any training slice)."""
    return MarketData(bars=md.bars.iloc[:n], macro=md.macro, events=md.events)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Strategy):
        return {"strategy": obj.name, "params": _jsonable(obj.params)}
    if isinstance(obj, type):
        return getattr(obj, "name", obj.__name__)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class _MLStrategyBase(Strategy):
    """Feature handling, label weights, classifier fitting, skill gate and reporting shared
    by the ML strategies. Subclasses implement ``_fit`` (labels -> classifier -> forecast
    scalar; wrapped by the transactional :meth:`fit`) and ``generate`` (frozen features ->
    probability -> forecast)."""

    trainable = True
    #: accepted ``event_filter`` values (subclasses may add their own)
    _EVENT_FILTERS: ClassVar[tuple[str, ...]] = ("all", "cusum")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            # trainable primaries are fitted on this leading fraction of the training slice;
            # the ML model learns only from later events (0 = in-sample, not recommended)
            "primary_oos_frac": 0.5,
            # features
            "feature_groups": None,        # None = DEFAULT_PRICE_GROUPS (+macro/calendar if present)
            "feature_overrides": {},
            "drop_features": (),           # regex patterns of raw feature columns to exclude
            "include_macro": True,
            "include_calendar": True,
            "scaler": "robust",
            # labels
            "max_holding_bars": 24,
            "pt_mult": 1.0,
            "sl_mult": 1.0,
            "barrier_vol": "horizon",      # barrier unit: per-bar sigma * sqrt(h) ("horizon") or sigma ("bar")
            "vol_span": 100,
            "event_filter": "all",         # "all" | "cusum" (| "flip" for meta_label)
            "cusum_mult": 2.0,
            "event_stride": 1,
            "sample_weight": "uniqueness",  # "uniqueness" | "return_attribution" | "uniqueness_x_return" | "none"
            "time_decay": 1.0,             # AFML 4.11 last_weight (1 = no decay)
            # classifier
            "model": {},
            "val_frac": 0.2,
            "embargo_bars": 0,
            "early_stopping": True,
            "calibration": "platt",
            "min_train_events": 200,
            "importance": "permutation",   # "permutation" | "none"
            "importance_repeats": 3,
            "importance_max_rows": 5000,
            # forecast mapping
            "target_abs_forecast": 0.5,
            "max_forecast_scalar": 20.0,
            "skill_gate_z": 2.0,           # stay flat unless val AUC beats 0.5 by this many s.e. (None = off)
            "seed": 0,
            "n_threads": None,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._pipe: FeaturePipeline | None = None
        self._external_features: bool = False
        self._clf: CalibratedGBM | None = None
        self._feature_columns: list[str] = []
        self.forecast_scalar_: float = 1.0
        self.fit_report_: dict[str, Any] = {}
        self.train_labels_: pd.DataFrame | None = None
        self.feature_importances_: pd.Series | None = None
        #: first training position whose trainable-primary forecasts are out of sample
        self._primary_oos_start: int = 0

    # ---- fit (transactional) --------------------------------------------------------------
    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> _MLStrategyBase:
        """Train on ``md`` (the TRAINING slice) only; labels never extend past its last bar.

        Transactional: the fit runs on a fresh instance and is committed only when it
        succeeds. A failed (re)fit raises and leaves this object exactly as it was - a
        previously fitted strategy keeps its complete, self-consistent model, an unfitted
        one stays unfitted.
        """
        self._validate_params()
        work = type(self)(**self.params)
        work._fit(md, features)
        self.__dict__.update(work.__dict__)
        return self

    def _fit(self, md: MarketData, features: pd.DataFrame | None) -> None:
        raise NotImplementedError

    def _validate_params(self) -> None:
        """Reject invalid options up front (before any expensive feature computation)."""
        p = self.params
        checks = {
            "event_filter": (p["event_filter"], self._EVENT_FILTERS),
            "barrier_vol": (p["barrier_vol"], ("horizon", "bar")),
            "sample_weight": (p["sample_weight"],
                              ("uniqueness", "return_attribution", "uniqueness_x_return", "none")),
            "importance": (p["importance"], ("permutation", "none")),
            "calibration": (p["calibration"], ("platt", "isotonic", "none")),
        }
        for key, (value, allowed) in checks.items():
            if value not in allowed:
                raise ValueError(f"{self.name}: {key} must be one of {allowed}, got {value!r}")
        if not 0.0 <= float(p["primary_oos_frac"]) < 1.0:
            raise ValueError(f"{self.name}: primary_oos_frac must be in [0, 1)")

    def _fit_primary(self, strat: Strategy, md: MarketData) -> None:
        """Fit a trainable primary on the leading ``primary_oos_frac`` of ``md`` and move
        ``_primary_oos_start`` so the ML model only learns from its out-of-sample forecasts."""
        if not strat.trainable:
            return
        frac = float(self.params["primary_oos_frac"])
        if frac <= 0.0:
            logger.warning("%s: trainable primary %r is fitted on the whole training slice; its "
                           "in-sample forecasts are optimistic features/sides for the ML model "
                           "(set primary_oos_frac > 0)", self.name, strat.name)
            strat.fit(md)
            return
        n_p = int(round(frac * len(md.bars)))
        strat.fit(_head(md, n_p))
        self._primary_oos_start = max(self._primary_oos_start, n_p)
        logger.info("%s: trainable primary %r fitted on the first %d training bars; ML events "
                    "start after them", self.name, strat.name, n_p)

    # ---- horizons -----------------------------------------------------------------------
    @property
    def label_horizon(self) -> int:
        """Longest label horizon in bars (read by the walk-forward engine to size its purge;
        ``fit`` additionally never builds a label that resolves after its training slice)."""
        return int(self.params["max_holding_bars"])

    def _pipe_warmup(self) -> int:
        """Warm-up of the fitted feature source: the pipeline's ``max_lookback``, or 0 for
        external features (their warm-up is data-defined: leading NaNs stay NaN)."""
        if self._pipe is None or self._external_features:
            return 0
        return int(self._pipe.max_lookback)

    @property
    def fit_history_bars(self) -> int:
        """Own-pipeline models need ``warmup_bars`` of pre-training history so their first
        training rows have fully warmed-up features (external features arrive warmed up)."""
        if self._external_features:
            return 0
        return self.warmup_bars

    @property
    def warmup_bars(self) -> int:
        if self._pipe is not None:
            return self._pipe_warmup()
        groups = self.params["feature_groups"] or DEFAULT_PRICE_GROUPS
        try:
            return int(FeaturePipeline(groups=list(groups),
                                       overrides=self._overrides_for(list(groups))).max_lookback)
        except Exception:  # noqa: BLE001 - informational only before fit
            return 0

    # ---- features -----------------------------------------------------------------------
    def _groups_for(self, md: MarketData) -> list[str]:
        g = self.params["feature_groups"]
        if g is not None:
            return list(g)
        groups = set(DEFAULT_PRICE_GROUPS)
        if self.params["include_macro"] and md.macro:
            groups.add("macro")
        if self.params["include_calendar"] and md.events is not None:
            groups.add("calendar")
        ordered = [x for x in CANONICAL_GROUP_ORDER if x in groups]
        return ordered + sorted(groups - set(ordered))

    def _overrides_for(self, groups: list[str]) -> dict[str, dict]:
        ov = self.params.get("feature_overrides") or {}
        return {k: dict(v) for k, v in ov.items() if k in groups}

    def _base_raw(self, md: MarketData, features: pd.DataFrame | None, *, fitting: bool) -> pd.DataFrame:
        """Raw (unscaled) features for the rows of ``md.bars``."""
        if fitting:
            self._external_features = features is not None
            if features is None:
                groups = self._groups_for(md)
                self._pipe = FeaturePipeline(groups=groups, overrides=self._overrides_for(groups),
                                             scaler=self.params["scaler"])
            else:
                # scaling only; leading NaNs of external features are warm-up (kept NaN)
                self._pipe = FeaturePipeline(groups=[], scaler=self.params["scaler"],
                                             warmup=_EXTERNAL_WARMUP)
        assert self._pipe is not None
        if features is not None:
            raw = features.reindex(md.bars.index)
            raw = raw.select_dtypes(include=[np.number, "bool"]).astype(float)
            raw = raw.replace([np.inf, -np.inf], np.nan)
        elif self._external_features:
            raise ValueError(f"{self.name} was fitted on externally supplied features; pass "
                             "`features` to generate() as well")
        else:
            raw = self._pipe.compute(md)
        return self._drop_columns(raw)

    def _drop_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        pats = [re.compile(p) for p in (self.params.get("drop_features") or ())]
        if not pats:
            return raw
        drop = [c for c in raw.columns if any(p.search(str(c)) for p in pats)]
        return raw.drop(columns=drop)

    def _transform(self, raw: pd.DataFrame, *, fitting: bool, train_rows: np.ndarray | None = None
                   ) -> pd.DataFrame:
        assert self._pipe is not None
        if fitting:
            rows = raw if train_rows is None else raw.iloc[train_rows]
            self._pipe.fit(rows)
            self._feature_columns = list(self._pipe.columns)
        return self._pipe.transform(raw, strict=False)

    # ---- labels & weights ---------------------------------------------------------------
    def _label_vol(self, close: pd.Series) -> pd.Series:
        vol = ewm_vol(close, span=int(self.params["vol_span"]))
        if self.params["barrier_vol"] == "horizon":
            vol = vol * math.sqrt(int(self.params["max_holding_bars"]))
        elif self.params["barrier_vol"] != "bar":
            raise ValueError("barrier_vol must be 'horizon' or 'bar'")
        return vol

    def _event_positions(self, close: pd.Series, candidates: np.ndarray) -> np.ndarray:
        """Subset of candidate positions per ``event_filter`` / ``event_stride``."""
        pos = np.asarray(candidates, dtype=np.int64)
        if self.params["event_filter"] not in self._EVENT_FILTERS:
            raise ValueError(f"{self.name}: event_filter must be one of {self._EVENT_FILTERS}, "
                             f"got {self.params['event_filter']!r}")
        if self.params["event_filter"] == "cusum":
            thr = float(self.params["cusum_mult"]) * ewm_vol(close, span=int(self.params["vol_span"]))
            ev = close.index.get_indexer(cusum_filter(close, thr))
            pos = np.intersect1d(pos, ev)
        stride = int(self.params["event_stride"])
        if stride > 1:
            pos = pos[::stride]
        return pos

    def _sample_weights(self, labels: pd.DataFrame, n_bars: int, close: pd.Series) -> np.ndarray:
        kind = self.params["sample_weight"]
        if kind == "none":
            w = np.ones(len(labels))
        elif kind == "uniqueness":
            w = uniqueness_weights(labels, n_bars=n_bars).to_numpy()
        elif kind == "return_attribution":
            w = return_attribution_weights(labels, close).to_numpy()
        elif kind == "uniqueness_x_return":
            w = (uniqueness_weights(labels, n_bars=n_bars).to_numpy()
                 * return_attribution_weights(labels, close).to_numpy())
        else:
            raise ValueError(f"unknown sample_weight {kind!r}")
        decay = float(self.params["time_decay"])
        if decay < 1.0:
            w = w * time_decay_weights(uniqueness_weights(labels, n_bars=n_bars).to_numpy(), decay)
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            raise ValueError("all sample weights are zero")
        return w / w.mean()

    # ---- classifier ---------------------------------------------------------------------
    def _new_classifier(self, class_weight: str | None) -> CalibratedGBM:
        return CalibratedGBM(
            model=self.params["model"],
            val_frac=float(self.params["val_frac"]),
            embargo_bars=int(self.params["embargo_bars"]),
            early_stopping=bool(self.params["early_stopping"]),
            calibration=self.params["calibration"],
            class_weight=class_weight,
            min_events=int(self.params["min_train_events"]),
            seed=int(self.params["seed"]),
            n_threads=self.params["n_threads"],
        )

    def _fit_classifier(self, X: pd.DataFrame, y: np.ndarray, labels: pd.DataFrame, n_bars: int,
                        close: pd.Series, class_weight: str | None) -> CalibratedGBM:
        w = self._sample_weights(labels, n_bars, close)
        clf = self._new_classifier(class_weight)
        clf.fit(X, y, w, labels["t_idx"].to_numpy(), labels["t1_idx"].to_numpy())
        return clf

    def _skill_gate(self, clf: CalibratedGBM, labels: pd.DataFrame, y: np.ndarray, n_bars: int
                    ) -> dict[str, Any]:
        """Is the validation AUC significantly above 0.5?

        Under H0 (no skill) the AUC has variance ``(n1 + n0 + 1) / (12 n1 n0)`` (Hanley &
        McNeil 1982). Overlapping labels are not independent, so ``n`` is the EFFECTIVE
        size of the validation tail: the sum of its labels' average uniqueness (AFML ch. 4).
        The tail also chose the iteration count and the calibration, so the test is mildly
        optimistic - it is a guard against scaling pure noise up to full size (Carver
        scaling normalises any forecast's average magnitude), not a proof of skill.
        """
        n_val = int(clf.report_["n_val"])
        u = average_uniqueness(labels, n_bars=n_bars)[-n_val:]
        pos_share = float(np.mean(y[-n_val:]))
        n_eff = float(np.sum(u))
        n1, n0 = pos_share * n_eff, (1.0 - pos_share) * n_eff
        auc = float(clf.report_["val_auc"])
        if n1 > 0 and n0 > 0 and math.isfinite(auc):
            se = math.sqrt((n_eff + 1.0) / (12.0 * n1 * n0))
            z = (auc - 0.5) / se
        else:
            se, z = float("nan"), float("nan")
        z_min = self.params["skill_gate_z"]
        passed = True if z_min is None else bool(math.isfinite(z) and z >= float(z_min))
        if not passed:
            logger.warning("%s: validation AUC %.4f (z=%.2f, n_eff=%.0f) below the skill gate "
                           "z >= %s - forecasts will be zero", self.name, auc, z, n_eff, z_min)
        return {"val_n_eff": n_eff, "val_auc_se": se, "val_auc_z": z, "skill_gate_passed": passed}

    def _finish_fit(self, clf: CalibratedGBM, labels: pd.DataFrame, md: MarketData,
                    extra: dict[str, Any]) -> None:
        if self.params["importance"] == "permutation":
            clf.compute_permutation_importance(n_repeats=int(self.params["importance_repeats"]),
                                               max_rows=self.params["importance_max_rows"])
            self.feature_importances_ = clf.feature_importances_
        elif self.params["importance"] != "none":
            raise ValueError("importance must be 'permutation' or 'none'")
        clf.release_validation()
        self._clf = clf
        self.train_labels_ = labels
        bars = md.bars
        self.fit_report_ = {
            "strategy": self.name,
            "train_start": bars.index[0].isoformat(),
            "train_end": bars.index[-1].isoformat(),
            "n_bars": len(bars),
            "n_labels": len(labels),
            "label_first": labels.index[0].isoformat() if len(labels) else None,
            "label_last_t1": labels["t1"].max().isoformat() if len(labels) else None,
            "barrier_counts": {str(k): int(v) for k, v in labels["barrier_hit"].value_counts().items()},
            "mean_holding_bars": float(labels["holding_bars"].mean()) if len(labels) else float("nan"),
            "n_features": len(self._feature_columns),
            "feature_groups": list(self._pipe.groups) if self._pipe is not None else [],
            "external_features": self._external_features,
            "forecast_scalar": self.forecast_scalar_,
            **clf.report_,
            **extra,
        }
        self.is_fitted = True

    def _require_fitted(self) -> None:
        if not self.is_fitted or self._clf is None:
            raise RuntimeError(f"{self.name}: call fit() on training data before generate()")

    # ---- reporting ----------------------------------------------------------------------
    def importance_by_group(self) -> pd.Series:
        """Permutation importance summed by feature family (column prefix before ``_``)."""
        if self.feature_importances_ is None:
            return pd.Series(dtype=float)
        fam = self.feature_importances_.index.str.split("_").str[0]
        return self.feature_importances_.groupby(fam).sum().sort_values(ascending=False)

    def explain(self, top: int = 15) -> dict[str, Any]:
        """JSON-serialisable summary for tearsheets and the LLM desk."""
        out: dict[str, Any] = {
            "strategy": self.name,
            "description": self.description,
            "fitted": self.is_fitted,
            "params": _jsonable({k: v for k, v in self.params.items() if k != "model"}),
            "model": _jsonable(self._clf.model_params if self._clf else self.params["model"]),
            "fit_report": _jsonable(self.fit_report_),
        }
        if self.feature_importances_ is not None:
            out["top_features"] = {str(k): float(v) for k, v in self.feature_importances_.head(top).items()}
            out["importance_by_group"] = {str(k): float(v) for k, v in self.importance_by_group().items()}
        return out


# ---------------------------------------------------------------------------------------
# ml_gbm
# ---------------------------------------------------------------------------------------
@register_strategy
class MLGBMStrategy(_MLStrategyBase):
    """Gradient-boosted direction classifier on FeaturePipeline features (see module doc)."""

    name = "ml_gbm"
    description = ("HistGradientBoosting classifier on causal features predicting the "
                   "triple-barrier (or vol-normalised forward-return) direction; purged, "
                   "uniqueness-weighted training, calibrated 2p-1 with dead-zone and Carver scaling.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            **super().default_params(),
            "target": "triple_barrier",   # "triple_barrier" | "fixed_horizon"
            "fixed_threshold": 0.0,       # |z| dead band for fixed-horizon labels
            "primary_features": (),       # strategies whose forecasts become features
            "class_weight": "balanced",
            "dead_zone": 0.02,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._primaries: list[tuple[str, Strategy]] = []

    # ---- primary forecasts as features ------------------------------------------------
    def _primary_frame(self, md: MarketData, *, fitting: bool) -> pd.DataFrame | None:
        specs: Sequence[Any] = self.params["primary_features"] or ()
        if not specs:
            return None
        if fitting:
            self._primaries = []
            for spec in specs:
                strat = _make_strategy(spec)
                self._fit_primary(strat, md)
                col = f"primary_{strat.name}"
                while col in {c for c, _ in self._primaries}:
                    col += "_"
                self._primaries.append((col, strat))
        cols = {col: strat.generate(md).reindex(md.bars.index).astype(float)
                for col, strat in self._primaries}
        return pd.DataFrame(cols, index=md.bars.index)

    def _raw(self, md: MarketData, features: pd.DataFrame | None, *, fitting: bool) -> pd.DataFrame:
        raw = self._base_raw(md, features, fitting=fitting)
        prim = self._primary_frame(md, fitting=fitting)
        if prim is not None:
            raw = pd.concat([raw.drop(columns=[c for c in prim.columns if c in raw.columns]), prim], axis=1)
        return raw

    # ---- fit ----------------------------------------------------------------------------
    def _validate_params(self) -> None:
        super()._validate_params()
        if self.params["target"] not in ("triple_barrier", "fixed_horizon"):
            raise ValueError(f"unknown target {self.params['target']!r}")

    def _fit(self, md: MarketData, features: pd.DataFrame | None) -> None:
        """Train on ``md`` (the TRAINING slice) only. Labels never extend past its last bar."""
        bars = md.bars
        n = len(bars)
        h = int(self.params["max_holding_bars"])
        raw = self._raw(md, features, fitting=True)
        # events need every feature defined: pipeline AND primary-forecast warm-ups, and
        # out-of-sample forecasts of trainable primaries
        warm = max(self.warmup_bars, self._primary_oos_start)
        # Scaler statistics from post-warm-up TRAIN rows only.
        X_all = self._transform(raw, fitting=True, train_rows=np.arange(min(warm, n - 1), n))
        close = bars["close"]
        vol = self._label_vol(close)
        valid_x = X_all.notna().all(axis=1).to_numpy()
        cand = np.flatnonzero(valid_x & (np.arange(n) >= warm))
        pos = self._event_positions(close, cand)
        if self.params["target"] == "triple_barrier":
            labels = triple_barrier_labels(bars, pt_mult=self.params["pt_mult"],
                                           sl_mult=self.params["sl_mult"], max_holding_bars=h,
                                           vol=vol, t_events=pos, vertical_label="sign")
        elif self.params["target"] == "fixed_horizon":
            labels = fixed_horizon_labels(bars, h, vol=ewm_vol(close, span=int(self.params["vol_span"])),
                                          threshold=float(self.params["fixed_threshold"]), t_events=pos)
        else:
            raise ValueError(f"unknown target {self.params['target']!r}")
        labels = drop_label_tail(labels, n, h)          # purge: nothing resolves past train end
        labels = labels.loc[labels["label"] != 0]       # binary target: up vs down
        if labels.empty:
            raise ValueError(f"{self.name}: no usable training labels")
        assert int(labels["t1_idx"].max()) <= n - 1
        X = X_all.iloc[labels["t_idx"].to_numpy()]
        y = (labels["label"].to_numpy() > 0).astype(float)
        clf = self._fit_classifier(X, y, labels, n, close, self.params["class_weight"])
        # Carver scalar on the validation tail, fixed from now on.
        p_val, _, _ = clf.validation_proba()
        f_val = apply_dead_zone(2.0 * p_val - 1.0, float(self.params["dead_zone"]))
        gate = self._skill_gate(clf, labels, y, n)
        self.forecast_scalar_ = carver_scalar(f_val, self.params["target_abs_forecast"],
                                              max_scalar=float(self.params["max_forecast_scalar"]))
        if not gate["skill_gate_passed"]:
            self.forecast_scalar_ = 0.0
        elif self.forecast_scalar_ == 0.0:
            logger.warning("%s: calibrated forecasts are inside the dead-zone on the whole "
                           "validation tail - the strategy will stay flat", self.name)
        self._finish_fit(clf, labels, md, {
            **gate,
            "target": self.params["target"],
            "first_event_bar": int(warm),
            "val_abs_forecast": float(np.mean(np.abs(np.clip(self.forecast_scalar_ * f_val, -1, 1)))),
            "val_frac_active": float(np.mean(f_val != 0)),
        })

    # ---- generate -----------------------------------------------------------------------
    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        self._require_fitted()
        assert self._clf is not None
        raw = self._raw(md, features, fitting=False)
        X = self._transform(raw, fitting=False)
        ok = X.notna().all(axis=1).to_numpy()
        out = np.zeros(len(X))
        if ok.any():
            p = self._clf.predict_proba(X.to_numpy()[ok])
            f = apply_dead_zone(2.0 * p - 1.0, float(self.params["dead_zone"]))
            out[ok] = np.clip(self.forecast_scalar_ * f, -1.0, 1.0)
        return self._finalize(pd.Series(out, index=md.bars.index), md.bars.index)

    def predict_proba(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        """Calibrated ``P(up)`` per bar (NaN in warm-up) - diagnostics / agents."""
        self._require_fitted()
        assert self._clf is not None
        X = self._transform(self._raw(md, features, fitting=False), fitting=False)
        ok = X.notna().all(axis=1).to_numpy()
        p = np.full(len(X), np.nan)
        if ok.any():
            p[ok] = self._clf.predict_proba(X.to_numpy()[ok])
        return pd.Series(p, index=md.bars.index, name="p_up")

    @property
    def warmup_bars(self) -> int:
        base = super().warmup_bars
        prim = max((s.warmup_bars for _, s in self._primaries), default=0)
        return max(base, prim)


# ---------------------------------------------------------------------------------------
# meta_label
# ---------------------------------------------------------------------------------------
@register_strategy
class MetaLabelStrategy(_MLStrategyBase):
    """Meta-labelling of a primary strategy with AFML bet sizing (see module doc)."""

    name = "meta_label"
    description = ("Meta-labelling: an ML classifier predicts whether the primary strategy's "
                   "bet (default tsmom) hits its profit-take before its stop; forecast = "
                   "primary side x AFML bet size from the calibrated probability.")
    _EVENT_FILTERS = ("all", "cusum", "flip")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            **super().default_params(),
            "primary": "tsmom",           # name | (name, params) | Strategy instance/class
            "primary_params": {},
            "min_primary_abs": 0.0,       # |primary forecast| must exceed this to be a bet
            "meta_vertical": "return_sign",  # vertical barrier -> sign of the bet's return, or "fail" (0)
            "p_threshold": 0.5,           # no bet when P(success) < threshold
            "step_size": 0.1,             # bet-size discretisation (AFML 10.3); 0 disables
            "scale_by_primary": False,    # multiply by |primary forecast| as well
            "class_weight": None,         # keep true success probabilities (bet sizing needs them)
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._primary: Strategy | None = None

    # ---- primary ------------------------------------------------------------------------
    def _primary_signal(self, md: MarketData) -> tuple[pd.Series, pd.Series]:
        assert self._primary is not None
        f = self._primary.generate(md).reindex(md.bars.index).astype(float).fillna(0.0)
        side = np.sign(f).where(f.abs() > float(self.params["min_primary_abs"]), 0.0)
        return f, side

    def _raw(self, md: MarketData, features: pd.DataFrame | None, f_primary: pd.Series,
             side: pd.Series, *, fitting: bool) -> pd.DataFrame:
        raw = self._base_raw(md, features, fitting=fitting)
        extra = pd.DataFrame({"meta_primary_forecast": f_primary, "meta_primary_side": side},
                             index=md.bars.index)
        return pd.concat([raw.drop(columns=[c for c in extra.columns if c in raw.columns]), extra],
                         axis=1)

    def _bet(self, p: np.ndarray, scalar: float) -> np.ndarray:
        m = np.maximum(bet_size_from_probability(p), 0.0)
        m = np.where(p >= float(self.params["p_threshold"]), m, 0.0)
        m = np.clip(scalar * m, 0.0, 1.0)
        return discretize_bet(m, float(self.params["step_size"]))

    # ---- fit ----------------------------------------------------------------------------
    def _validate_params(self) -> None:
        super()._validate_params()
        if self.params["meta_vertical"] not in ("return_sign", "fail"):
            raise ValueError("meta_vertical must be 'return_sign' or 'fail'")

    def _fit(self, md: MarketData, features: pd.DataFrame | None) -> None:
        """Fit the primary (if trainable, out of sample - see module doc) and the meta-model
        on the TRAINING slice ``md``."""
        bars = md.bars
        n = len(bars)
        h = int(self.params["max_holding_bars"])
        self._primary = _make_strategy(self.params["primary"], self.params["primary_params"])
        self._fit_primary(self._primary, md)
        f_primary, side = self._primary_signal(md)
        raw = self._raw(md, features, f_primary, side, fitting=True)
        assert self._pipe is not None
        warm = max(self.warmup_bars, self._primary_oos_start)
        X_all = self._transform(raw, fitting=True, train_rows=np.arange(min(warm, n - 1), n))
        close = bars["close"]
        valid_x = X_all.notna().all(axis=1).to_numpy()
        s = side.to_numpy()
        cand = valid_x & (np.arange(n) >= warm) & (s != 0)
        if self.params["event_filter"] == "flip":
            prev = np.r_[0.0, s[:-1]]
            cand &= s != prev
        pos = self._event_positions(close, np.flatnonzero(cand))
        labels = triple_barrier_labels(bars, pt_mult=self.params["pt_mult"], sl_mult=self.params["sl_mult"],
                                       max_holding_bars=h, vol=self._label_vol(close), side=side,
                                       t_events=pos)
        labels = drop_label_tail(labels, n, h)
        y_s = meta_labels(labels, vertical=self.params["meta_vertical"])
        keep = y_s.notna().to_numpy()
        labels, y = labels.loc[keep], y_s.to_numpy()[keep]
        if labels.empty:
            raise ValueError(f"{self.name}: no usable meta-labels (does the primary ever take a side?)")
        assert int(labels["t1_idx"].max()) <= n - 1
        X = X_all.iloc[labels["t_idx"].to_numpy()]
        clf = self._fit_classifier(X, y, labels, n, close, self.params["class_weight"])
        p_val, _, _ = clf.validation_proba()
        gate = self._skill_gate(clf, labels, y, n)
        # Carver scaling on the (un-discretised) bet sizes of the validation tail
        m_raw = np.where(p_val >= float(self.params["p_threshold"]),
                         np.maximum(bet_size_from_probability(p_val), 0.0), 0.0)
        self.forecast_scalar_ = carver_scalar(m_raw, self.params["target_abs_forecast"],
                                              max_scalar=float(self.params["max_forecast_scalar"]))
        if not gate["skill_gate_passed"]:
            self.forecast_scalar_ = 0.0
        m_val = self._bet(p_val, self.forecast_scalar_)
        self._finish_fit(clf, labels, md, {
            **gate,
            "primary": _jsonable(self._primary),
            "primary_trainable": bool(self._primary.trainable),
            "first_event_bar": int(warm),
            "meta_vertical": self.params["meta_vertical"],
            "success_rate": float(np.mean(y)),
            "val_abs_bet": float(np.mean(np.abs(m_val))),
            "val_frac_active": float(np.mean(m_val > 0)),
        })

    # ---- generate -----------------------------------------------------------------------
    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        self._require_fitted()
        assert self._clf is not None
        f_primary, side = self._primary_signal(md)
        X = self._transform(self._raw(md, features, f_primary, side, fitting=False), fitting=False)
        s = side.to_numpy()
        ok = X.notna().all(axis=1).to_numpy() & (s != 0)
        out = np.zeros(len(X))
        if ok.any():
            p = self._clf.predict_proba(X.to_numpy()[ok])
            size = self._bet(p, self.forecast_scalar_)
            if self.params["scale_by_primary"]:
                size = size * np.minimum(np.abs(f_primary.to_numpy()[ok]), 1.0)
            out[ok] = s[ok] * size
        return self._finalize(pd.Series(out, index=md.bars.index), md.bars.index)

    def predict_proba(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        """Calibrated ``P(primary bet succeeds)`` per bar (NaN where no bet / warm-up)."""
        self._require_fitted()
        assert self._clf is not None
        f_primary, side = self._primary_signal(md)
        X = self._transform(self._raw(md, features, f_primary, side, fitting=False), fitting=False)
        ok = X.notna().all(axis=1).to_numpy() & (side.to_numpy() != 0)
        p = np.full(len(X), np.nan)
        if ok.any():
            p[ok] = self._clf.predict_proba(X.to_numpy()[ok])
        return pd.Series(p, index=md.bars.index, name="p_success")

    @property
    def warmup_bars(self) -> int:
        base = super().warmup_bars
        return max(base, int(self._primary.warmup_bars) if self._primary is not None else 0)
