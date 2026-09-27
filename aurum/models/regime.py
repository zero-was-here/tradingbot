"""Hidden Markov regime model (Gaussian emissions, diagonal covariance).

Financial returns alternate between calm, trending and turbulent periods whose volatility
differs by a factor of 2-4. A Markov-switching model (Hamilton, 1989) captures this with a
latent state ``s_t`` following a Markov chain and state-dependent return distributions.

Estimation is by the Baum-Welch EM algorithm (Baum et al., 1970) with the per-step
*scaling* of Rabiner (1989, "A tutorial on hidden Markov models", Proc. IEEE 77(2), §V.A)
so the forward/backward recursions never underflow. Emission log-densities are shifted by
their per-row maximum before exponentiation, which makes the recursions robust to extreme
observations as well.

Point-in-time contract (SPEC §5)
    * ``fit`` must be called on TRAINING data only (it uses the forward-backward
      smoother, i.e. future observations inside the training window — that is fine for
      estimation, never for decisions).
    * ``filter(x)`` returns the forward-filtered probabilities ``P(s_t | x_0..x_t)``: row t
      depends only on ``x[:t+1]`` and the fitted parameters.
    * ``predict_next(x)`` returns ``P(s_{t+1} | x_0..x_t) = filter_t @ A``.
    * States are ordered by (scale-free) variance, so state 0 is the calmest regime.

Missing observations (NaN) contribute no likelihood (the filter just propagates the prior
through the transition matrix); for multivariate inputs, missing components are
marginalised out, which is exact for diagonal Gaussians.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOG_2PI = math.log(2.0 * math.pi)


class GaussianHMM:
    """Gaussian hidden Markov model with diagonal covariances.

    Parameters
    ----------
    n_states  : number of latent regimes (default 2: calm / turbulent).
    seed      : seed for the random restarts (deterministic).
    n_iter    : maximum EM iterations per restart.
    tol       : convergence threshold on the absolute log-likelihood improvement.
    n_init    : number of EM restarts (the first is a deterministic quantile split, the
                others perturb it randomly); the best training likelihood is kept.
    var_floor : minimum variance as a fraction of the training variance of each feature
                (prevents a state collapsing onto a single observation).
    init_stay : initial diagonal of the transition matrix (regimes are persistent).

    Fitted attributes: ``startprob_`` (K,), ``transmat_`` (K,K), ``means_`` (K,D),
    ``vars_`` (K,D), ``loglik_``, ``n_iter_``, ``converged_``.
    """

    def __init__(
        self,
        n_states: int = 2,
        seed: int = 0,
        *,
        n_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 3,
        var_floor: float = 1e-3,
        init_stay: float = 0.95,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if not 0.0 < init_stay < 1.0 and n_states > 1:
            raise ValueError("init_stay must be in (0, 1)")
        self.n_states = int(n_states)
        self.seed = int(seed)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.n_init = max(int(n_init), 1)
        self.var_floor = float(var_floor)
        self.init_stay = float(init_stay)
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.loglik_: float = math.nan
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.columns_: list[str] | None = None
        self._named_columns: bool = False
        self._scale: np.ndarray | None = None
        self._floor: np.ndarray | None = None

    # ------------------------------------------------------------------------------------
    # data handling
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _prepare(x: pd.Series | pd.DataFrame | np.ndarray) -> tuple[np.ndarray, pd.Index]:
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=float, na_value=np.nan)[:, None], x.index
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=float, na_value=np.nan), x.index
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("x must be 1-D or 2-D")
        return arr, pd.RangeIndex(arr.shape[0])

    def _check_fitted(self) -> None:
        if self.transmat_ is None:
            raise RuntimeError("GaussianHMM is not fitted; call fit(train_x) first")

    def _prepare_fitted(self, x: pd.Series | pd.DataFrame | np.ndarray) -> tuple[np.ndarray, pd.Index]:
        """Like :meth:`_prepare` but aligned to the fitted feature layout.

        Emission parameters are positional, so a DataFrame whose columns come in a different
        order (e.g. a live feature pipeline) would silently be scored against the wrong
        means/variances, and a narrower input would broadcast against them. DataFrames are
        therefore re-ordered by the fitted column names (when the model was fitted on a
        DataFrame; arrays and Series are positional), and any width mismatch raises.
        """
        self._check_fitted()
        n_feat = self.means_.shape[1]
        if isinstance(x, pd.DataFrame) and self._named_columns and self.columns_ is not None:
            names = [str(c) for c in x.columns]
            if names != self.columns_:
                missing = [c for c in self.columns_ if c not in names]
                if missing:
                    raise KeyError(f"x lacks fitted feature columns {missing}; fitted on {self.columns_}")
                pos = {c: i for i, c in enumerate(names)}
                x = x.iloc[:, [pos[c] for c in self.columns_]]
        arr, index = self._prepare(x)
        if arr.shape[1] != n_feat:
            raise ValueError(f"x has {arr.shape[1]} feature(s) but the model was fitted on {n_feat}")
        return arr, index

    # ------------------------------------------------------------------------------------
    # core recursions
    # ------------------------------------------------------------------------------------
    def _log_emission(self, x: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
        """log N(x_t | mean_k, diag var_k), missing components marginalised (T x K)."""
        obs = np.isfinite(x)
        xf = np.where(obs, x, 0.0)
        diff = xf[:, None, :] - means[None, :, :]
        term = -0.5 * (_LOG_2PI + np.log(var)[None, :, :] + diff * diff / var[None, :, :])
        term = np.where(obs[:, None, :], term, 0.0)
        return term.sum(axis=2)

    @staticmethod
    def _forward(logb: np.ndarray, startprob: np.ndarray, transmat: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Scaled forward pass. Returns (alpha_hat, c, b, loglik).

        ``alpha_hat[t] = P(s_t | x_0..t)`` (normalised), ``c[t]`` the scaling constants with
        ``log L = sum log c + sum max_k logb`` and ``b = exp(logb - max_k logb)``.
        """
        t_len, k = logb.shape
        m = logb.max(axis=1)
        b = np.exp(logb - m[:, None])
        alpha = np.empty((t_len, k))
        c = np.empty(t_len)
        a = startprob * b[0]
        s = a.sum()
        if not s > 0:
            a, s = startprob.copy(), 1e-300
        alpha[0] = a / a.sum()
        c[0] = s
        at = transmat
        if k == 2:
            # Unrolled scalar loop: ~10x faster than tiny numpy ops for the common 2-state case.
            a00, a01, a10, a11 = at[0, 0], at[0, 1], at[1, 0], at[1, 1]
            p0, p1 = alpha[0, 0], alpha[0, 1]
            b0 = b[:, 0].tolist()
            b1 = b[:, 1].tolist()
            out0 = [p0]
            out1 = [p1]
            cl = [c[0]]
            for t in range(1, t_len):
                q0 = p0 * a00 + p1 * a10
                q1 = p0 * a01 + p1 * a11
                n0 = q0 * b0[t]
                n1 = q1 * b1[t]
                s = n0 + n1
                if s > 0:
                    p0, p1 = n0 / s, n1 / s
                else:  # every state has ~zero density: fall back to the prediction
                    tot = q0 + q1
                    p0, p1 = q0 / tot, q1 / tot
                    s = 1e-300
                out0.append(p0)
                out1.append(p1)
                cl.append(s)
            alpha[:, 0] = out0
            alpha[:, 1] = out1
            c[:] = cl
        else:
            for t in range(1, t_len):
                q = alpha[t - 1] @ at
                a = q * b[t]
                s = a.sum()
                if s > 0:
                    alpha[t] = a / s
                else:
                    alpha[t] = q / q.sum()
                    s = 1e-300
                c[t] = s
        loglik = float(np.log(c).sum() + m.sum())
        return alpha, c, b, loglik

    @staticmethod
    def _backward(b: np.ndarray, c: np.ndarray, transmat: np.ndarray) -> np.ndarray:
        t_len, k = b.shape
        beta = np.empty((t_len, k))
        beta[-1] = 1.0
        if k == 2:
            a00, a01, a10, a11 = transmat[0, 0], transmat[0, 1], transmat[1, 0], transmat[1, 1]
            b0 = b[:, 0].tolist()
            b1 = b[:, 1].tolist()
            cl = c.tolist()
            r0, r1 = 1.0, 1.0
            out0 = [0.0] * t_len
            out1 = [0.0] * t_len
            out0[-1], out1[-1] = 1.0, 1.0
            for t in range(t_len - 2, -1, -1):
                u0 = b0[t + 1] * r0
                u1 = b1[t + 1] * r1
                ct = cl[t + 1]
                r0 = (a00 * u0 + a01 * u1) / ct
                r1 = (a10 * u0 + a11 * u1) / ct
                out0[t] = r0
                out1[t] = r1
            beta[:, 0] = out0
            beta[:, 1] = out1
        else:
            for t in range(t_len - 2, -1, -1):
                beta[t] = transmat @ (b[t + 1] * beta[t + 1]) / c[t + 1]
        # Guard against overflow when c was floored at 1e-300 (pathological data).
        beta = np.nan_to_num(beta, nan=1.0, posinf=1e300)
        return beta

    def _e_step(self, x: np.ndarray, startprob: np.ndarray, transmat: np.ndarray,
                means: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        logb = self._log_emission(x, means, var)
        alpha, c, b, loglik = self._forward(logb, startprob, transmat)
        beta = self._backward(b, c, transmat)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        if x.shape[0] > 1:
            w = b[1:] * beta[1:] / c[1:, None]
            xi_sum = transmat * (alpha[:-1].T @ w)
        else:
            xi_sum = np.zeros_like(transmat)
        return gamma, xi_sum, loglik

    def _m_step(self, x: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray, floor: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = gamma.shape[1]
        startprob = gamma[0] + 1e-8
        startprob /= startprob.sum()
        if k > 1:
            transmat = xi_sum + 1e-10
            transmat /= transmat.sum(axis=1, keepdims=True)
        else:
            transmat = np.ones((1, 1))
        means, var = self._weighted_moments(x, gamma, floor)
        return startprob, transmat, means, var

    @staticmethod
    def _weighted_moments(x: np.ndarray, gamma: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.isfinite(x).astype(float)
        xf = np.where(obs > 0, x, 0.0)
        w = gamma.T @ obs                                   # K x D effective counts
        w = np.maximum(w, 1e-12)
        means = (gamma.T @ xf) / w
        diff = xf[:, None, :] - means[None, :, :]           # T x K x D
        sq = diff * diff * obs[:, None, :]
        var = np.einsum("tk,tkd->kd", gamma, sq) / w
        var = np.maximum(var, floor[None, :])
        return means, var

    # ------------------------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------------------------
    def _initial_params(self, x: np.ndarray, rng: np.random.Generator, restart: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_len, d = x.shape
        k = self.n_states
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        z = (x - mu) / sd
        energy = np.nansum(z * z, axis=1)
        # Deterministic split on the "energy" quantiles: low energy -> calm state.
        ranks = np.argsort(np.argsort(energy, kind="stable"), kind="stable")
        labels = np.minimum((ranks * k) // max(t_len, 1), k - 1)
        resp = np.zeros((t_len, k))
        resp[np.arange(t_len), labels] = 1.0
        if restart > 0:
            noise = rng.dirichlet(np.ones(k), size=t_len)
            mix = min(0.3 + 0.2 * restart, 0.9)
            resp = (1.0 - mix) * resp + mix * noise
        means, var = self._weighted_moments(x, resp, self._floor)
        if restart > 0:
            means = means + rng.normal(0.0, 0.1, size=means.shape) * sd[None, :]
            var = np.maximum(var * np.exp(rng.normal(0.0, 0.3, size=var.shape)), self._floor[None, :])
        if k > 1:
            transmat = np.full((k, k), (1.0 - self.init_stay) / (k - 1))
            np.fill_diagonal(transmat, self.init_stay)
        else:
            transmat = np.ones((1, 1))
        startprob = np.full(k, 1.0 / k)
        return startprob, transmat, means, var

    def fit(self, x: pd.Series | pd.DataFrame | np.ndarray) -> GaussianHMM:
        """Baum-Welch EM on TRAINING observations (T,) or (T, D)."""
        arr, _ = self._prepare(x)
        if isinstance(x, pd.DataFrame):
            self.columns_ = [str(c) for c in x.columns]
            self._named_columns = True
        elif isinstance(x, pd.Series):
            self.columns_ = [str(x.name) if x.name is not None else "x"]
            self._named_columns = False
        else:
            self.columns_ = [f"x{i}" for i in range(arr.shape[1])]
            self._named_columns = False
        finite_rows = np.isfinite(arr).any(axis=1)
        if finite_rows.sum() < 5 * self.n_states:
            raise ValueError("not enough finite observations to fit the HMM")
        overall_var = np.nanvar(arr, axis=0)
        overall_var = np.where(overall_var > 0, overall_var, 1.0)
        self._scale = overall_var
        self._floor = self.var_floor * overall_var
        rng = np.random.default_rng(self.seed)
        best: tuple | None = None
        for restart in range(self.n_init):
            params = self._initial_params(arr, rng, restart)
            startprob, transmat, means, var = params
            prev = -np.inf
            converged = False
            it = 0
            for it in range(1, self.n_iter + 1):
                gamma, xi_sum, ll = self._e_step(arr, startprob, transmat, means, var)
                if not np.isfinite(ll):
                    logger.debug("HMM restart %d produced non-finite log-likelihood", restart)
                    break
                if ll - prev < self.tol and it > 1:
                    converged = True
                    break
                prev = ll
                startprob, transmat, means, var = self._m_step(arr, gamma, xi_sum, self._floor)
            final_ll = self._forward(self._log_emission(arr, means, var), startprob, transmat)[3]
            logger.debug("HMM restart %d: loglik=%.4f iters=%d converged=%s", restart, final_ll, it, converged)
            if np.isfinite(final_ll) and (best is None or final_ll > best[0]):
                best = (final_ll, startprob, transmat, means, var, it, converged)
        if best is None:
            raise RuntimeError("HMM fitting failed in every restart")
        ll, startprob, transmat, means, var, it, converged = best
        # Order states by scale-free variance: state 0 = calm.
        order = np.argsort((var / self._scale[None, :]).sum(axis=1), kind="stable")
        self.startprob_ = startprob[order]
        self.transmat_ = transmat[np.ix_(order, order)]
        self.means_ = means[order]
        self.vars_ = var[order]
        self.loglik_ = float(ll)
        self.n_iter_ = int(it)
        self.converged_ = bool(converged)
        if not converged:
            logger.info("HMM EM stopped at n_iter=%d before reaching tol=%g", it, self.tol)
        return self

    def _columns(self, prefix: str) -> list[str]:
        return [f"{prefix}{k}" for k in range(self.n_states)]

    def filter(self, x: pd.Series | pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Forward-filtered ``P(s_t = k | x_0..x_t)`` — causal (row t uses x[:t+1] only)."""
        arr, index = self._prepare_fitted(x)
        if arr.shape[0] == 0:
            return pd.DataFrame(columns=self._columns("p_state_"), index=index, dtype=float)
        logb = self._log_emission(arr, self.means_, self.vars_)
        alpha, *_ = self._forward(logb, self.startprob_, self.transmat_)
        return pd.DataFrame(alpha, index=index, columns=self._columns("p_state_"))

    def predict_next(self, x: pd.Series | pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """One-step-ahead regime probabilities ``P(s_{t+1} = k | x_0..x_t)`` (causal)."""
        filt = self.filter(x)
        nxt = filt.to_numpy() @ self.transmat_
        return pd.DataFrame(nxt, index=filt.index, columns=self._columns("p_next_state_"))

    def filtered_state(self, x: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series:
        """Most likely CURRENT state under the filtered (causal) distribution."""
        filt = self.filter(x)
        return pd.Series(filt.to_numpy().argmax(axis=1), index=filt.index, name="regime")

    def score(self, x: pd.Series | pd.DataFrame | np.ndarray) -> float:
        """Log-likelihood of ``x`` under the fitted model."""
        arr, _ = self._prepare_fitted(x)
        logb = self._log_emission(arr, self.means_, self.vars_)
        return self._forward(logb, self.startprob_, self.transmat_)[3]

    @property
    def stationary_distribution_(self) -> np.ndarray:
        """Left eigenvector of the transition matrix for eigenvalue 1."""
        self._check_fitted()
        vals, vecs = np.linalg.eig(self.transmat_.T)
        i = int(np.argmin(np.abs(vals - 1.0)))
        v = np.real(vecs[:, i])
        v = np.abs(v) / np.abs(v).sum()
        return v

    @property
    def expected_durations_(self) -> np.ndarray:
        """Expected regime duration in observations: ``1 / (1 - A_kk)``."""
        self._check_fitted()
        stay = np.diag(self.transmat_)
        with np.errstate(divide="ignore"):
            return np.where(stay < 1.0, 1.0 / (1.0 - stay), np.inf)

    def summary(self) -> dict:
        """JSON-friendly description (for reports / LLM agents)."""
        self._check_fitted()
        return {
            "n_states": self.n_states,
            "columns": self.columns_,
            "means": self.means_.tolist(),
            "vars": self.vars_.tolist(),
            "transmat": self.transmat_.tolist(),
            "startprob": self.startprob_.tolist(),
            "stationary": self.stationary_distribution_.tolist(),
            "expected_durations": [float(v) for v in self.expected_durations_],
            "loglik": self.loglik_,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
        }

    def __repr__(self) -> str:
        return f"GaussianHMM(n_states={self.n_states}, seed={self.seed}, fitted={self.transmat_ is not None})"
