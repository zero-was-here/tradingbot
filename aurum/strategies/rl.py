"""``rl_ppo``: forecasts from a PPO policy trained in :mod:`aurum.rl` (SPEC §6).

Economic rationale
------------------
Rule-based strategies hard-code one hypothesis each (trend persistence, mean reversion,
macro sensitivity). A reinforcement-learning policy instead learns a direct mapping from
the causal feature state (multi-horizon returns, trend, volatility and regime descriptors,
session clock, higher-timeframe context) plus its own position to a target exposure,
maximising log wealth net of the SAME transaction costs, swaps, vol-targeted sizing and
risk limits used everywhere else (Moody & Saffell, 2001, "Learning to trade via direct
reinforcement"; Deng et al., 2017, "Deep direct reinforcement learning for financial signal
representation and trading"). Because it observes its own position and unrealised P&L it
can learn cost-aware holding behaviour that a stateless forecast cannot express.

The prior for a large edge is weak: financial rewards have a very low signal-to-noise
ratio, so PPO readily overfits the training period. Validation-based checkpoint selection
and the out-of-sample protocol of ``aurum.research`` are therefore essential; the artifact's
``metrics.json`` records how many validation evaluations were used for selection.

Forecast semantics
------------------
``generate`` rolls the deterministic policy forward bar by bar with its OWN simulated
account (ExecutionSimulator + the training sizer/costs/risk limits), because the policy's
observation includes its current position, unrealised P&L, time in trade and drawdown.
The forecast at ``t`` is the policy's action on the observation at the close of ``t``
(discrete levels, default ``{-1, -0.5, 0, 0.5, 1}``). The simulated account runs the
TRAINING sizer, costs and risk limits and mirrors how training episodes start and end:

* it starts flat at the first bar with a full observation window and, with the default
  ``episode_anchor="M"``, starts a FRESH episode (flat, initial equity) at the first bar of
  every month. A forecast therefore depends only on the bars since the last anchor (plus
  the feature warm-up), not on where the history window starts: a live runner fed a
  sliding window reproduces the backtest (a continuous account made 2-25% of 2020-21 H1
  forecasts depend on the window start). Forecasts before ``warmup_bars`` (feature warm-up
  + one anchor period) are 0.
* when a kill switch (or ruin) would end a training episode, it fires and flattens as in
  training, then the account starts a new episode in place (drawdown peak and risk state
  re-based, :meth:`aurum.rl.env.GoldTradingEnv.rebase_episode`). Forecasts continue after
  a simulated drawdown, yet the policy never observes a drawdown deeper than the ones that
  ended its training episodes (disabling the kills instead fed it >20% drawdowns — never
  seen in training — on 42% of 2012-2021 H1 bars).

The downstream risk manager is the one that may halt real trading. The simulated position
is the policy's own bookkeeping; a downstream backtest reproduces it only while it uses the
training sizer/risk settings from an anchor on and until the first simulated kill switch.

``features`` passed to ``fit``/``generate`` are ignored: the policy must see exactly the
feature columns and scaling it was trained with, so the artifact's own
:class:`~aurum.features.pipeline.FeaturePipeline` recomputes them from ``md`` (causal).
``md`` must have the bar size the policy was trained on (``ValueError`` otherwise).

Artifact integrity and portability: every ``fit`` writes a NEW directory (a unique
subdirectory of ``out_dir``), so clones fitted on different walk-forward folds can never
overwrite each other's policy. When a strategy attaches an artifact (``fit`` or
:meth:`RLPolicyStrategy.load`) it reads the files that define the policy (``policy.zip``,
``pipeline.json``, ``config.json``, plus ``metrics.json``) ONCE and embeds their bytes and
SHA-256 fingerprint in its own state. Clones and pickles (the live trading artifact's
``strategies.pkl``, walk-forward worker processes) are therefore self-contained: the
artifact directory is provenance only and may be moved, copied elsewhere or deleted, and
the policy is rebuilt from the embedded bytes. A strategy never silently trades a different
policy: overwriting the directory later has no effect (the attached bytes are used), and
embedded bytes that no longer match the fingerprint are refused. The torch network itself
is still rebuilt lazily and never pickled. Strategies pickled before embedding existed fall
back to the directory (which must then exist, unchanged).

Pre-trained artifacts in a walk-forward (``retrain=False``)
---------------------------------------------------------
With ``params.artifact_dir`` the strategy attaches that artifact at construction and, by
default (``retrain`` unset), ``fit`` does NOT train: it re-verifies the attached policy (the
embedded bytes against their SHA-256 fingerprint, the directory - when it still exists -
against the same fingerprint, and the optional ``artifact_sha256`` pin) and the bar size,
and records the fit window's last bar as the *training cutoff*. The artifact's
``config.json`` records the train/validation periods it was trained on
(:func:`aurum.rl.train.artifact_data_ranges`; older artifacts: ``metrics.json``; an
artifact with no record is refused).

Look-ahead guard. Every walk-forward TEST row lies after its fold's training cutoff, so a
policy that has seen any bar after the cutoff would forecast test rows it was trained (or
checkpoint-selected) on. Hence:

* ``fit`` raises ``ValueError`` when the artifact's last training/validation bar is AFTER
  the fit window's last bar - the same contract every trainable strategy obeys (``fit``
  sees training data only; this also covers the walk-forward's purge/embargo gap). It
  fails at build time (a fold, ``aurum train-final``, a desk replay), never later in a live
  loop;
* ``generate`` (defence in depth, e.g. hand-edited state) refuses any bar after
  ``fit_cutoff_`` that is not strictly after the artifact's last bar.

In a walk-forward this means a pre-trained policy can only be evaluated on folds whose
training window ends at or after its validation period; earlier folds raise.
``retrain=True`` restores per-fold training (the attached artifact is then only the
initial state). Without ``artifact_dir`` the strategy trains in ``fit`` as before.

Importing this module does NOT import torch, stable-baselines3 or gymnasium; they are
loaded inside ``fit``/``generate`` (optional ``rl`` extra).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from aurum.core.types import MarketData
from aurum.rl import episode_anchor_span_bars
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import check_params

logger = logging.getLogger(__name__)

__all__ = ["RLPolicyStrategy"]

#: Files whose content defines a policy's behaviour (network, feature scaling, env config).
_FINGERPRINT_FILES = ("policy.zip", "pipeline.json", "config.json")


def _artifact_fingerprint(path: Path) -> str:
    """SHA-256 over the files that define the policy (detects in-place overwrites)."""
    h = hashlib.sha256()
    for name in _FINGERPRINT_FILES:
        h.update(name.encode("utf-8") + b"\0")
        with open(path / name, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _bytes_fingerprint(files: Mapping[str, bytes]) -> str:
    """:func:`_artifact_fingerprint` of in-memory file contents (same digest)."""
    h = hashlib.sha256()
    for name in _FINGERPRINT_FILES:
        h.update(name.encode("utf-8") + b"\0")
        h.update(files[name])
    return h.hexdigest()


@register_strategy
class RLPolicyStrategy(Strategy):
    """Adapter from a trained :mod:`aurum.rl` PPO artifact to the Strategy interface.

    Parameters (``params``)
    -----------------------
    artifact_dir : load a trained artifact (``policy.zip``, ``pipeline.json``, ``config.json``)
                   instead of training. Its files are embedded in the strategy (pickles are
                   self-contained); the path is kept for provenance only.
    retrain      : ``None`` (default) = ``False`` when ``artifact_dir`` is given at
                   construction, ``True`` otherwise (resolved once, in ``__init__``).
                   ``False``: ``fit`` loads/verifies the attached artifact instead of training
                   and ``generate`` refuses out-of-sample bars the artifact was trained on
                   (see "Pre-trained artifacts" in the module doc). ``True``: ``fit`` trains
                   a new policy on the data it is given.
    artifact_sha256 : optional pin (full SHA-256 or a prefix of >= 8 hex digits) of the
                   artifact fingerprint; a different policy is refused at construction/fit.
    config       : :class:`aurum.rl.train.RLTrainConfig` or its dict form, used by ``fit``.
    val_fraction : trailing fraction of the ``fit`` data held out (time-ordered) for
                   validation-based checkpoint selection / early stopping.
    min_val_bars : minimum validation bars; ``fit`` raises if the split leaves fewer.
    out_dir      : parent directory under which every ``fit`` creates its OWN new artifact
                   subdirectory (``rl_ppo_<last fit bar>_<random>``); default: a new
                   temporary directory. Reusing one ``out_dir`` across folds is safe.
    device       : torch device for inference (``"cpu"`` default for determinism).
    episode_anchor : ``"auto"`` (default: the artifact's ``config.env.episode_anchor``, i.e.
                   the convention its checkpoint was validated with — monthly by default),
                   ``"W"``, ``"M"``, ``"Q"`` or ``None``. The simulated account starts a fresh
                   training-style episode (flat, initial equity) at the first bar of every
                   calendar period, so a forecast depends only on the bars since the last
                   anchor (plus the feature warm-up) — NOT on where the history window
                   starts: a live runner fed a sliding window reproduces the backtest.
                   ``None`` = one continuous simulated account (path-dependent). Overriding
                   the validated convention changes the policy's behaviour (see module doc).
    """

    name = "rl_ppo"
    description = ("PPO policy (stable-baselines3) trained on the shared simulator with "
                   "vol-targeted sizing and full costs; forecast = deterministic action, "
                   "rolled forward with its own simulated position.")
    trainable = True

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"artifact_dir": None, "retrain": None, "artifact_sha256": None, "config": None,
                "val_fraction": 0.25, "min_val_bars": 500, "out_dir": None, "device": "cpu",
                "episode_anchor": "auto"}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        self._artifact: Any = None          # aurum.rl.train.RLArtifact (lazy, holds torch)
        self._pipeline: Any = None          # FeaturePipeline (JSON, cheap)
        self._config: Any = None            # RLTrainConfig
        self._bundle: dict[str, bytes] | None = None   # embedded artifact files (pickled)
        self.artifact_dir_: Path | None = None
        self.artifact_fingerprint_: str | None = None
        #: train/validation periods recorded in the attached artifact (None if unknown)
        self.artifact_data_ranges_: dict[str, Any] | None = None
        #: last bar of the data the latest ``fit`` received (bars after it are out of sample)
        self.fit_cutoff_: pd.Timestamp | None = None
        self.train_summary_: dict[str, Any] = {}
        retrain = self.params.get("retrain")
        if retrain is None:
            retrain = not self.params.get("artifact_dir")
        if not isinstance(retrain, bool):
            raise ValueError(f"rl_ppo: retrain must be true, false or null, got {retrain!r}")
        if not retrain and not self.params.get("artifact_dir"):
            raise ValueError("rl_ppo: retrain=False needs params.artifact_dir (there is no policy to load)")
        # Resolved ONCE: ``load`` rewrites params["artifact_dir"] (provenance) after every
        # training fit, which must not turn a training strategy into a loading one.
        self.params["retrain"] = retrain
        pin = self.params.get("artifact_sha256")
        if pin is not None and not (isinstance(pin, str) and len(pin) >= 8
                                    and all(c in "0123456789abcdef" for c in pin.lower())):
            raise ValueError("rl_ppo: artifact_sha256 must be a hex SHA-256 (or a prefix of >= 8 digits)")
        if self.params.get("artifact_dir"):
            self.load(self.params["artifact_dir"])

    # ---- persistence ---------------------------------------------------------------------
    def load(self, artifact_dir: str | Path) -> RLPolicyStrategy:
        """Attach a trained artifact: its files are read ONCE and embedded (bytes + SHA-256
        fingerprint), so this strategy and its pickles no longer need the directory. The
        policy network is rebuilt lazily on the first :meth:`generate` (keeps construction
        torch-free)."""
        from aurum.features.pipeline import FeaturePipeline
        from aurum.rl.train import RLTrainConfig, artifact_data_ranges, read_artifact_bytes

        p = Path(artifact_dir)
        missing = [f for f in _FINGERPRINT_FILES if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(f"RL artifact {p} is missing {missing}")
        files = read_artifact_bytes(p)
        fingerprint = _bytes_fingerprint(files)
        self._check_pin(fingerprint, p)
        self._config = RLTrainConfig.from_artifact_dict(json.loads(files["config.json"].decode("utf-8")))
        self._pipeline = FeaturePipeline.from_dict(json.loads(files["pipeline.json"].decode("utf-8")))
        self._artifact = None
        self._bundle = dict(files)
        self.artifact_dir_ = p
        self.artifact_fingerprint_ = fingerprint
        self.artifact_data_ranges_ = artifact_data_ranges(files)
        self.params["artifact_dir"] = str(p)
        self.train_summary_ = {}
        if "metrics.json" in files:
            m = json.loads(files["metrics.json"].decode("utf-8"))
            self.train_summary_ = {k: m.get(k) for k in (
                "val", "selected_timesteps", "n_evals", "timesteps_trained", "early_stopped")}
        self.is_fitted = True
        return self

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, **params: Any) -> RLPolicyStrategy:
        return cls(artifact_dir=str(artifact_dir), **params)

    @property
    def has_embedded_policy(self) -> bool:
        """Whether the policy files travel inside this object (pickles are self-contained)."""
        return bool(getattr(self, "_bundle", None))

    def _get_artifact(self) -> Any:
        if self._artifact is None:
            expected = getattr(self, "artifact_fingerprint_", None)
            bundle = getattr(self, "_bundle", None)
            if bundle:
                if expected is None or _bytes_fingerprint(bundle) != expected:
                    raise RuntimeError(
                        "the RL policy embedded in this strategy does not match its fingerprint "
                        f"({(expected or '?')[:12]}...); refusing to run a different policy. "
                        "Call load(artifact_dir) explicitly to attach new files.")
                from aurum.rl.train import load_artifact_bytes

                self._artifact = load_artifact_bytes(bundle, device=self.params["device"],
                                                     path=self.artifact_dir_)
                return self._artifact
            # legacy state (pickled before policies were embedded): the directory must exist
            if self.artifact_dir_ is None:
                raise RuntimeError("rl_ppo is not fitted: call fit() or load(artifact_dir)")
            if expected is not None and _artifact_fingerprint(self.artifact_dir_) != expected:
                raise RuntimeError(
                    f"RL artifact {self.artifact_dir_} changed on disk since it was attached "
                    f"(fingerprint {expected[:12]}...); refusing to run a different policy. "
                    "Call load(artifact_dir) explicitly to attach the new files.")
            from aurum.rl.train import load_artifact

            self._artifact = load_artifact(self.artifact_dir_, device=self.params["device"])
        return self._artifact

    def __getstate__(self) -> dict[str, Any]:
        # The torch network is rebuilt on demand from the EMBEDDED policy bytes, so clones and
        # pickles are self-contained (no dependency on ``artifact_dir_``) and never share
        # mutable network state.
        state = self.__dict__.copy()
        state["_artifact"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("_bundle", None)   # strategies pickled before embedding
        state.setdefault("artifact_data_ranges_", None)   # ... or before the look-ahead guard
        state.setdefault("fit_cutoff_", None)
        self.__dict__.update(state)

    # ---- verification ------------------------------------------------------------------------
    def _check_pin(self, fingerprint: str, where: Any) -> None:
        pin = self.params.get("artifact_sha256")
        if pin and not fingerprint.startswith(str(pin).lower()):
            raise RuntimeError(f"RL artifact {where} has fingerprint {fingerprint[:16]}..., not the "
                               f"pinned artifact_sha256={pin}; refusing to run a different policy")

    def _verify_attached(self) -> None:
        """Re-check the attached policy before a loading ``fit`` (see module doc): embedded
        bytes vs their fingerprint, the directory (if it still exists) vs the same
        fingerprint, and the ``artifact_sha256`` pin."""
        bundle = getattr(self, "_bundle", None)
        expected = getattr(self, "artifact_fingerprint_", None)
        if not bundle:
            # legacy pickle (no embedded bytes) or nothing attached yet: attach from the directory
            src = self.artifact_dir_ or self.params.get("artifact_dir")
            if not src:
                raise RuntimeError("rl_ppo: retrain=False but no artifact is attached")
            if expected is not None and _artifact_fingerprint(Path(src)) != expected:
                raise RuntimeError(f"RL artifact {src} changed on disk since it was attached "
                                   f"(fingerprint {expected[:12]}...); refusing to run a different policy")
            self.load(src)
            return
        actual = _bytes_fingerprint(bundle)
        if expected is None or actual != expected:
            raise RuntimeError("the RL policy embedded in this strategy does not match its fingerprint "
                               f"({(expected or '?')[:12]}...); refusing to run a different policy")
        d = self.artifact_dir_
        if d is not None and all((d / f).exists() for f in _FINGERPRINT_FILES):
            on_disk = _artifact_fingerprint(d)
            if on_disk != expected:
                raise RuntimeError(
                    f"RL artifact {d} changed on disk since it was attached (fingerprint "
                    f"{expected[:12]}... -> {on_disk[:12]}...); refusing to fit with an ambiguous "
                    "policy. Call load(artifact_dir) explicitly to attach the new files.")
        self._check_pin(actual, d or "(embedded)")

    def _check_lookahead(self, index: pd.DatetimeIndex) -> None:
        """Refuse out-of-sample bars (after ``fit_cutoff_``) the attached policy has seen."""
        cutoff = getattr(self, "fit_cutoff_", None)
        ranges = getattr(self, "artifact_data_ranges_", None)
        if cutoff is None or not ranges or len(index) == 0:
            return
        oos = index[index > cutoff]
        if len(oos) == 0 or oos[0] > ranges["end"]:
            return
        bad = oos[oos <= ranges["end"]]
        raise ValueError(
            f"rl_ppo look-ahead guard: the attached policy ({self.artifact_dir_ or 'embedded'}) was "
            f"trained/validated on bars {ranges['start']} -> {ranges['end']} (recorded in "
            f"{ranges.get('source')}), but it is asked to forecast {len(bad)} out-of-sample bar(s) "
            f"{bad[0]} -> {bad[-1]} after its fit cutoff {cutoff}. Evaluate it only after "
            f"{ranges['end']}, use an artifact trained before {oos[0]}, or set retrain=True.")

    # ---- Strategy interface ----------------------------------------------------------------
    @property
    def _policy_warmup(self) -> int:
        """First bar with a full observation window (where the simulated account starts)."""
        if self._pipeline is None or self._config is None:
            return 0
        return int(self._pipeline.max_lookback + self._config.env.window - 1)

    @property
    def episode_anchor(self) -> str | None:
        """The resolved episode anchor (``"auto"`` -> the artifact's validated convention)."""
        anchor = self.params.get("episode_anchor", "auto")
        if anchor == "auto":
            return None if self._config is None else self._config.env.episode_anchor
        return anchor

    @property
    def warmup_bars(self) -> int:
        """Feature warm-up plus, with an ``episode_anchor``, one (conservative) anchor period:
        from here on every forecast lies after an anchor that every history window of at
        least this length contains, so the forecast does not depend on the window start."""
        warm = self._policy_warmup
        if warm == 0 or self._pipeline is None:
            return warm
        minutes = self._pipeline.bar_minutes
        span = episode_anchor_span_bars(self.episode_anchor, float(minutes) if minutes else 60.0)
        return warm + span

    def _train_config(self) -> Any:
        from aurum.rl.train import RLTrainConfig

        cfg = self.params.get("config")
        if cfg is None:
            return RLTrainConfig()
        if isinstance(cfg, RLTrainConfig):
            return cfg
        return RLTrainConfig.from_dict(cfg)

    def fit(self, md: MarketData, features: pd.DataFrame | None = None) -> RLPolicyStrategy:
        """Train PPO on ``md`` with a trailing time-ordered validation split, or - with
        ``retrain=False`` - verify and keep the attached artifact (module doc), refusing
        (``ValueError``) an artifact trained on any bar after the last bar of ``md``.

        Either way the last bar of ``md`` becomes ``fit_cutoff_``: later bars are out of
        sample, and ``generate`` refuses those the attached policy was trained on."""
        if len(md.bars) == 0:
            raise ValueError("rl_ppo.fit: no bars")
        if not self.params.get("retrain", True):
            return self._fit_from_artifact(md)
        from aurum.rl.train import train_ppo

        cfg = self._train_config()
        n = len(md.bars)
        frac = float(self.params["val_fraction"])
        if not 0.0 < frac < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        n_val = int(round(n * frac))
        if n_val < int(self.params["min_val_bars"]) or n - n_val < 2:
            raise ValueError(f"rl_ppo.fit: {n} bars leave {n_val} validation bars "
                             f"(< min_val_bars={self.params['min_val_bars']})")
        split = n - n_val
        md_tr = MarketData(bars=md.bars.iloc[:split], macro=md.macro, events=md.events)
        md_va = MarketData(bars=md.bars.iloc[split:], macro=md.macro, events=md.events)
        # A NEW directory per fit: clones fitted on different folds must never share (and
        # overwrite) one artifact, or a lazily reloaded policy could be a later fold's.
        prefix = f"rl_ppo_{pd.Timestamp(md.bars.index[-1]):%Y%m%dT%H%M}_"
        parent = self.params.get("out_dir")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
            out = tempfile.mkdtemp(prefix=prefix, dir=str(parent))
        else:
            out = tempfile.mkdtemp(prefix=prefix)
        logger.info("rl_ppo.fit: %d train / %d validation bars -> %s", split, n_val, out)
        res = train_ppo(md_tr, md_va, config=cfg, out_dir=out)
        self.load(res.artifact_dir)
        self.fit_cutoff_ = pd.Timestamp(md.bars.index[-1])
        return self

    def _fit_from_artifact(self, md: MarketData) -> RLPolicyStrategy:
        """``retrain=False``: keep the attached artifact after re-verifying it; record the
        training cutoff for the look-ahead guard."""
        from aurum.rl.train import check_bar_size

        self._verify_attached()
        if not self.artifact_data_ranges_:
            raise ValueError(
                f"rl_ppo: artifact {self.artifact_dir_} records no train/validation period "
                "(config.json data_ranges / metrics.json data), so a walk-forward cannot verify that "
                "its test rows are out of sample; retrain it with this version or set retrain=True")
        check_bar_size(self._pipeline, md.bars)
        cutoff = pd.Timestamp(md.bars.index[-1])
        start, end = self.artifact_data_ranges_["start"], self.artifact_data_ranges_["end"]
        if end > cutoff:
            raise ValueError(
                f"rl_ppo look-ahead guard: the attached policy ({self.artifact_dir_ or 'embedded'}) was "
                f"trained/validated on bars {start} -> {end} (recorded in "
                f"{self.artifact_data_ranges_.get('source')}), AFTER this fit's training cutoff {cutoff}: "
                f"it would forecast out-of-sample bars it has seen. Use it only where the training "
                f"window ends at or after {end}, use an artifact trained before {cutoff}, or set "
                "retrain=True.")
        if self.artifact_data_ranges_.get("source") != "config.json":
            logger.info("rl_ppo: training period of %s read from metrics.json (not fingerprinted)",
                        self.artifact_dir_)
        self.fit_cutoff_ = cutoff
        self.is_fitted = True
        logger.info("rl_ppo.fit: loaded artifact %s (fingerprint %s..., trained %s -> %s) instead of "
                    "retraining; cutoff %s", self.artifact_dir_, (self.artifact_fingerprint_ or "?")[:12],
                    start, end, cutoff)
        return self

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        """Deterministic policy forecasts in the policy's discrete levels (or [-1, 1])."""
        if not self.is_fitted or (self.artifact_dir_ is None and not self.has_embedded_policy):
            raise RuntimeError("rl_ppo is not fitted: call fit() or load(artifact_dir)")
        from aurum.rl.train import check_bar_size, rollout_artifact

        check_bar_size(self._pipeline, md.bars)
        index = md.bars.index
        self._check_lookahead(pd.DatetimeIndex(index))
        warm = self.warmup_bars
        if len(index) < warm + 2:
            logger.info("rl_ppo.generate: %d bars do not cover the %d-bar warm-up; flat",
                        len(index), warm)
            return self._finalize(pd.Series(0.0, index=index), index)
        ro = rollout_artifact(self._get_artifact(), md, start=self._policy_warmup,
                              kill_switches=False,
                              episode_anchor=self.episode_anchor)
        return self._finalize(ro.forecast, index)  # zero before warmup_bars
