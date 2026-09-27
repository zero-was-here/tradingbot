"""Live runner: the bar-close decision loop of the production path (SPEC §11).

One cycle per CLOSED bar, the same chain as the research backtest (SPEC §0.2-0.3)::

    broker.latest_bars (closed only) + macro (point-in-time) + calendar
      -> FeaturePipeline.compute/transform       (fitted on TRAIN, loaded from the artifact)
      -> strategy forecasts -> ForecastCombiner   (fitted, loaded from the artifact)
      -> optional LLM desk (TradingDesk; failures fall back per policy)
      -> VolTargetSizer -> StandardRiskManager (LIVE limits, persistent kill switch)
      -> OrderManager (idempotent, reconciles against the venue position)

Timing mirrors the simulator: the decision time is the bar's ``available_at``; orders are
sent right after it (``bar_close_delay_seconds`` later, default 5 s, so the venue has
published the bar). When the market is closed at that moment (daily maintenance break,
weekend) the OMS reports ``deferred`` and the runner retries the SAME decision when quotes
return — the simulator's "fill at the next open" — unless a newer bar supersedes it or it
would ADD risk and is older than ``max_defer_seconds`` (risk reductions never go stale: a
Friday-close flatten fills at the reopen). A deferred intent retried while the risk manager
is halted is replaced by a flatten. ``stop_cooldown_bars`` reproduces the engine's
post-stop re-entry block (the stop is recognised by replaying the engine's intrabar rule on
the bars since the last decision, because venues do not report why a position vanished).

Safety
------
* ``dry_run=True`` by default: intended orders are planned and logged, nothing is sent.
* Real-money guard: a non-demo account is refused unless the config sets
  ``live.allow_live_real: true`` AND the operator passes ``--i-understand-real-money``;
  a loud banner is logged/printed when real trading is enabled.
* Kill switch: the risk manager's state file persists halts across restarts; only
  ``StandardRiskManager.reset_halt(confirm="RESET")`` clears them.
* Idempotency: the OMS never executes the same bar twice; the runner also remembers the
  last processed bar. One runner per ``state_dir``: an OS lock (``runner.lock``) refuses a
  second process (two runners would both trade every bar).
* Stale data: a new bar older than ``max_bar_age_seconds`` is skipped (and the risk manager
  blocks new risk above ``stale_data_seconds``); insufficient history refuses to trade. A
  HALTED book is still flattened on such bars (the kill switch must not depend on data).
  Bars the venue returns with ``available_at > now`` are dropped and alerted (never decide
  on a forming bar, whatever the adapter does).
* Non-finite sizes HOLD the position (as in the backtest engine) instead of being rounded
  to 0 lots (which would liquidate).
* Errors in features/strategies: ``on_error="hold"`` (default) keeps the position and alerts;
  ``"flatten"`` closes it through risk and the OMS.
* Graceful shutdown on SIGINT/SIGTERM; a heartbeat file is rewritten every loop.
* Every decision is appended to ``decisions.jsonl`` (forecasts per strategy, combined, desk
  decision, sizing, risk, orders, fills, equity).

Trading artifact
----------------
A directory written by the final-train step (:func:`save_artifact`)::

    manifest.json           format/version, versions, sha256 of every file, optional HMAC
    pipeline.json           FeaturePipeline.save (config + TRAIN scaler statistics)
    strategies.pkl          {name: fitted Strategy}      (pickle)
    strategies.json         names, classes, params (human-readable)
    combiner.pkl            fitted ForecastCombiner      (pickle, optional: equal weights)
    feature_reference.json  PSI reference bins of the TRAIN features (optional)
    backtest.json           expected daily mean/std, OOS stats (optional; PnL band)

Pickle executes code on load: only load artifacts you produced. :func:`load_artifact`
verifies every file's sha256 against the manifest BEFORE unpickling (integrity) and, when
``AURUM_ARTIFACT_KEY`` is set, an HMAC-SHA256 of the manifest (authenticity).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import importlib.metadata
import json
import logging
import math
import os
import pickle
import platform
import shutil
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.backtest.engine import OUTCOME_COLUMNS, average_true_range
from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.interfaces import RiskContext
from aurum.core.timeframes import get_timeframe
from aurum.core.types import MarketData
from aurum.execution.costs import CostModel
from aurum.live.broker import Broker, BrokerError, Clock, SimulatedClock, SystemClock, net_lots
from aurum.live.monitor import (
    AlertManager,
    DriftMonitor,
    FeatureReference,
    JsonlAlertSink,
    LiveMonitor,
    LogAlertSink,
    PnLBand,
    SlippageTracker,
    WebhookAlertSink,
)
from aurum.live.oms import ExecutionReport, OrderManager
from aurum.live.paper import _protective_exit
from aurum.live.state import (
    StateCorruptError,
    append_jsonl,
    atomic_write_json,
    read_json,
    utc,
    write_heartbeat,
)
from aurum.models.volatility import ewma_volatility
from aurum.portfolio.sizing import VolTargetSizer
from aurum.risk.manager import RiskLimits, StandardRiskManager

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "ArtifactError",
    "CycleResult",
    "LiveConfig",
    "LiveDeskDataProvider",
    "LiveRunner",
    "RealMoneyGuardError",
    "RunnerLockedError",
    "TradingArtifact",
    "check_real_money_guard",
    "load_artifact",
    "main",
    "real_money_banner",
    "save_artifact",
]

ARTIFACT_FORMAT = "aurum.live.artifact"
ARTIFACT_VERSION = 1
_EPS = 1e-9


# =============================================================================================
# artifact
# =============================================================================================
class ArtifactError(RuntimeError):
    """The trading artifact is missing, incompatible or fails its integrity checks."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    """Commit of the working tree (read-only ``git rev-parse``), for provenance only."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent, check=False)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _manifest_mac(manifest: Mapping[str, Any], key: bytes) -> str:
    body = {k: v for k, v in manifest.items() if k != "hmac"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


@dataclass
class TradingArtifact:
    """Everything the live runner needs from research, loaded from an artifact directory."""

    strategies: dict[str, Any]
    pipeline: Any | None = None
    combiner: Any | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    feature_reference: FeatureReference | None = None
    backtest_stats: dict[str, Any] = field(default_factory=dict)
    sizer_config: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def timeframe(self) -> str | None:
        return self.manifest.get("timeframe")

    @property
    def symbol(self) -> str | None:
        return self.manifest.get("symbol")

    @property
    def max_lookback(self) -> int:
        """Warm-up bars: the pipeline's and every strategy's."""
        lb = int(self.pipeline.max_lookback) if self.pipeline is not None else 0
        for s in self.strategies.values():
            lb = max(lb, int(getattr(s, "warmup_bars", 0) or 0))
        return lb

    def features(self, md: MarketData) -> pd.DataFrame | None:
        """TRANSFORMED features (fitted scaler) — what strategies were trained on."""
        if self.pipeline is None:
            return None
        return self.pipeline.transform(self.pipeline.compute(md))

    def combine(self, forecasts: pd.DataFrame) -> pd.Series:
        """Combined forecast in [-1, 1]; equal-weight mean when no combiner was fitted."""
        if self.combiner is not None:
            return self.combiner.combine(forecasts)
        return forecasts.mean(axis=1).clip(-1.0, 1.0).rename("combined")


def save_artifact(
    path: str | Path,
    *,
    strategies: Mapping[str, Any] | list[Any],
    pipeline: Any | None = None,
    combiner: Any | None = None,
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    sizer_config: Mapping[str, Any] | None = None,
    backtest_stats: Mapping[str, Any] | None = None,
    feature_reference: FeatureReference | None = None,
    training: Mapping[str, Any] | None = None,
    notes: str = "",
    overwrite: bool = False,
) -> Path:
    """Write a trading artifact directory (see module docstring). Returns its path.

    ``strategies`` must be FITTED (``is_fitted``); names come from the mapping keys (or each
    strategy's ``name`` for a list) and must match the combiner's fitted columns.
    The directory is written to a temporary sibling and swapped in, so a crash never leaves
    a half-written artifact where the runner would load it.
    """
    if isinstance(strategies, Mapping):
        strat = {str(k): v for k, v in strategies.items()}
    else:
        strat = {}
        for s in strategies:
            name = str(getattr(s, "name", type(s).__name__))
            if name in strat:
                raise ValueError(f"duplicate strategy name {name!r}; pass a mapping with unique keys")
            strat[name] = s
    if not strat:
        raise ValueError("an artifact needs at least one strategy")
    for name, s in strat.items():
        if not callable(getattr(s, "generate", None)):
            raise TypeError(f"strategy {name!r} has no generate(md, features) method")
        if not getattr(s, "is_fitted", True):
            raise ValueError(f"strategy {name!r} is not fitted")
    if pipeline is not None and not getattr(pipeline, "is_fitted", False):
        raise ValueError("the feature pipeline must be fitted (on TRAIN) before saving")
    if combiner is not None:
        cols = list(getattr(combiner, "columns_", []) or [])
        if cols and sorted(cols) != sorted(strat):
            raise ValueError(f"combiner was fitted on {sorted(cols)}, strategies are {sorted(strat)}")
    get_timeframe(timeframe)

    dest = Path(path)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"artifact {dest} exists (pass overwrite=True)")
    tmp = dest.with_name(f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        files: dict[str, str] = {}
        if pipeline is not None:
            pipeline.save(tmp / "pipeline.json")
        with open(tmp / "strategies.pkl", "wb") as fh:
            pickle.dump(strat, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (tmp / "strategies.json").write_text(json.dumps({
            n: {"class": f"{type(s).__module__}.{type(s).__qualname__}",
                "params": _jsonable_params(getattr(s, "params", {})),
                "warmup_bars": int(getattr(s, "warmup_bars", 0) or 0),
                "trainable": bool(getattr(s, "trainable", False))} for n, s in strat.items()},
            indent=2, sort_keys=True))
        if combiner is not None:
            with open(tmp / "combiner.pkl", "wb") as fh:
                pickle.dump(combiner, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if feature_reference is not None:
            feature_reference.save(tmp / "feature_reference.json")
        if backtest_stats:
            atomic_write_json(tmp / "backtest.json", dict(backtest_stats))
        for f in sorted(tmp.iterdir()):
            files[f.name] = _sha256(f)
        lookback = TradingArtifact(strategies=strat, pipeline=pipeline).max_lookback
        manifest: dict[str, Any] = {
            "format": ARTIFACT_FORMAT, "version": ARTIFACT_VERSION,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "symbol": symbol, "timeframe": get_timeframe(timeframe).name,
            "strategies": sorted(strat), "combiner": type(combiner).__name__ if combiner is not None else None,
            "max_lookback": lookback,
            "n_features": len(pipeline.columns) if pipeline is not None else 0,
            "sizer": dict(sizer_config or {}), "training": _jsonable_params(dict(training or {})),
            "notes": notes, "files": files,
            "versions": {"aurum": _pkg_version("aurum"), "python": platform.python_version(),
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": _pkg_version("scikit-learn")},
            "git_sha": _git_sha(),
        }
        key = os.environ.get("AURUM_ARTIFACT_KEY")
        if key:
            manifest["hmac"] = _manifest_mac(manifest, key.encode())
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        backup = None
        if dest.exists():
            backup = dest.with_name(f".{dest.name}.old-{uuid.uuid4().hex[:8]}")
            os.replace(dest, backup)
        os.replace(tmp, dest)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info("trading artifact written to %s (%d strategies, lookback %d)", dest, len(strat), lookback)
    return dest


def _jsonable_params(d: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[str(k)] = v
        except (TypeError, ValueError):
            out[str(k)] = repr(v)
    return out


def load_artifact(path: str | Path, *, verify: bool = True, require_hmac: bool = False) -> TradingArtifact:
    """Load and verify a trading artifact (format/version, file hashes, optional HMAC)."""
    root = Path(path)
    mpath = root / "manifest.json"
    if not mpath.exists():
        raise ArtifactError(f"{root} is not a trading artifact (no manifest.json)")
    try:
        manifest = json.loads(mpath.read_text())
    except ValueError as exc:
        raise ArtifactError(f"unreadable manifest in {root}: {exc}") from exc
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError(f"{root}: unknown artifact format {manifest.get('format')!r}")
    version = int(manifest.get("version", -1))
    if not 1 <= version <= ARTIFACT_VERSION:
        raise ArtifactError(f"{root}: artifact version {version} not supported (max {ARTIFACT_VERSION})")
    vers = manifest.get("versions", {})
    here = _pkg_version("aurum")
    if vers.get("aurum") and here and vers["aurum"].split(".")[0] != here.split(".")[0]:
        raise ArtifactError(f"artifact built with aurum {vers['aurum']}, running {here}: major versions differ")
    if vers.get("python") and vers["python"].rsplit(".", 1)[0] != platform.python_version().rsplit(".", 1)[0]:
        logger.warning("artifact pickled with Python %s, running %s: pickle compatibility not guaranteed",
                       vers["python"], platform.python_version())
    key = os.environ.get("AURUM_ARTIFACT_KEY")
    if key:
        mac = manifest.get("hmac")
        if not isinstance(mac, str) or not hmac.compare_digest(mac, _manifest_mac(manifest, key.encode())):
            raise ArtifactError(f"{root}: HMAC verification failed (artifact not signed with AURUM_ARTIFACT_KEY)")
    elif require_hmac:
        raise ArtifactError("require_hmac=True but AURUM_ARTIFACT_KEY is not set")
    files: dict[str, str] = manifest.get("files", {})
    if verify:
        for name, digest in files.items():
            f = root / name
            if not f.exists():
                raise ArtifactError(f"{root}: file {name} listed in the manifest is missing")
            if _sha256(f) != digest:
                raise ArtifactError(f"{root}: sha256 mismatch for {name} (modified or corrupt)")
        for f in ("strategies.pkl", "combiner.pkl", "pipeline.json"):
            if (root / f).exists() and f not in files:
                raise ArtifactError(f"{root}: {f} is not covered by the manifest")
    if "strategies.pkl" not in files:
        raise ArtifactError(f"{root}: no strategies.pkl")

    pipeline = None
    if "pipeline.json" in files:
        from aurum.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline.load(root / "pipeline.json")
        if not pipeline.is_fitted:
            raise ArtifactError("artifact pipeline is not fitted")
    with open(root / "strategies.pkl", "rb") as fh:
        strategies = pickle.load(fh)  # noqa: S301 - integrity verified above; own artifacts only
    if not isinstance(strategies, dict) or not strategies:
        raise ArtifactError("strategies.pkl must hold a non-empty {name: strategy} mapping")
    if sorted(strategies) != sorted(manifest.get("strategies", [])):
        raise ArtifactError("strategies.pkl does not match the manifest's strategy list")
    for n, s in strategies.items():
        if not callable(getattr(s, "generate", None)):
            raise ArtifactError(f"strategy {n!r} has no generate() method")
    combiner = None
    if "combiner.pkl" in files:
        with open(root / "combiner.pkl", "rb") as fh:
            combiner = pickle.load(fh)  # noqa: S301
        if not callable(getattr(combiner, "combine", None)):
            raise ArtifactError("combiner.pkl has no combine() method")
    ref = FeatureReference.load(root / "feature_reference.json") if "feature_reference.json" in files else None
    stats = read_json(root / "backtest.json") if "backtest.json" in files else {}
    art = TradingArtifact(strategies=strategies, pipeline=pipeline, combiner=combiner, manifest=manifest,
                          feature_reference=ref, backtest_stats=stats or {},
                          sizer_config=dict(manifest.get("sizer") or {}), path=root)
    logger.info("artifact %s loaded: %s on %s %s, lookback %d", root, sorted(strategies), art.symbol,
                art.timeframe, art.max_lookback)
    return art


# =============================================================================================
# configuration
# =============================================================================================
_SECTIONS = ("risk", "sizer", "costs", "desk", "monitor", "paper", "mt5", "oms")


@dataclass
class LiveConfig:
    """Live-runner configuration (YAML: a ``live:`` section plus the sub-sections below).

    Sub-sections (plain dicts): ``risk`` (RiskLimits kwargs — LIVE limits), ``sizer``
    (VolTargetSizer kwargs, override the artifact's), ``costs`` (CostModel kwargs used by the
    paper broker and the slippage monitor), ``desk`` (``enabled``, ``mode``, ``on_failure``,
    ``on_error``, ``max_abs_forecast``, ``min_confidence``, ``config`` = DeskConfig kwargs),
    ``monitor`` (``webhook``, ``webhook_style``, ``drift_every``, ``drift_window``,
    ``psi_warn``, ``psi_alert``, ``z_warn``, ``z_alert``, ``alert_cooldown_seconds``),
    ``paper`` (``data``: ``"replay"`` (default: ``bars_path`` replayed on a simulated clock
    from ``start`` or after ``warmup_bars``, until ``end``) or ``"mt5"`` (paper fills on live MT5
    data, wall clock); ``initial_equity``, ``hedging``), ``mt5``
    (MT5Broker kwargs: ``server_tz``, ``deviation_points``...), ``oms`` (OrderManager kwargs).
    """

    artifact_dir: str | None = None
    symbol: str = "XAUUSD"
    timeframe: str = "H1"
    magic: int = 20260926
    broker: str = "paper"
    dry_run: bool = True
    allow_live_real: bool = False
    state_dir: str = "runs/live"
    bar_close_delay_seconds: float = 5.0
    history_bars: int | None = None
    history_multiple: float = 3.0
    min_history_bars: int = 300
    max_bar_age_seconds: float | None = None
    retry_poll_seconds: float = 30.0
    retry_poll_max_seconds: float = 300.0
    max_defer_seconds: float | None = None
    stop_atr_mult: float | None = None
    take_profit_atr_mult: float | None = None
    atr_period: int = 14
    stop_cooldown_bars: int = 0
    spread_source: str = "bar"
    on_error: str = "hold"
    flatten_on_shutdown: bool = False
    macro_dir: str | None = None
    macro_refresh_hours: float = 24.0
    macro_max_age_days: float = 7.0
    calendar: str | None = "rule_based"
    calendar_csv: str | None = None
    risk: dict[str, Any] = field(default_factory=dict)
    sizer: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    desk: dict[str, Any] = field(default_factory=dict)
    monitor: dict[str, Any] = field(default_factory=dict)
    paper: dict[str, Any] = field(default_factory=dict)
    mt5: dict[str, Any] = field(default_factory=dict)
    oms: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        get_timeframe(self.timeframe)
        if self.spread_source not in ("bar", "quote"):
            raise ValueError("spread_source must be 'bar' or 'quote'")
        if self.on_error not in ("hold", "flatten"):
            raise ValueError("on_error must be 'hold' or 'flatten'")
        if self.broker not in ("paper", "mt5"):
            raise ValueError("broker must be 'paper' or 'mt5'")
        if self.calendar not in (None, "rule_based", "none"):
            raise ValueError("calendar must be 'rule_based' or None (add CSV events with calendar_csv)")
        if not 0 <= self.bar_close_delay_seconds <= 600:
            raise ValueError("bar_close_delay_seconds must be in [0, 600]")
        if self.history_multiple < 1:
            raise ValueError("history_multiple must be >= 1")
        for k in ("stop_atr_mult", "take_profit_atr_mult"):
            v = getattr(self, k)
            if v is not None and not v > 0:
                raise ValueError(f"{k} must be positive or None")
        if not 0 < int(self.magic) < 2**31:
            raise ValueError("magic must be a positive 32-bit integer")
        if int(self.stop_cooldown_bars) < 0:
            raise ValueError("stop_cooldown_bars must be >= 0")

    @property
    def tf_seconds(self) -> float:
        return get_timeframe(self.timeframe).delta.total_seconds()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LiveConfig:
        """Flat mapping or ``{"live": {...}, "risk": {...}, ...}``. Unknown keys raise."""
        d = dict(data)
        base = dict(d.pop("live", {}) or {})
        for sec in _SECTIONS:
            if sec in d:
                base[sec] = dict(d.pop(sec) or {})
        base.update(d)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(base) - known)
        if unknown:
            raise ValueError(f"unknown live config keys: {unknown}")
        return cls(**base)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LiveConfig:
        import yaml

        return cls.from_mapping(yaml.safe_load(Path(path).read_text()) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================================
# real-money guard
# =============================================================================================
class RealMoneyGuardError(RuntimeError):
    """Refusal to trade a non-demo account without both explicit opt-ins."""


def check_real_money_guard(is_demo: bool, *, allow_live_real: bool, i_understand_real_money: bool) -> bool:
    """Raise unless the account is a demo or BOTH opt-ins are given. Returns True for real money."""
    if is_demo:
        return False
    if not allow_live_real or not i_understand_real_money:
        missing = []
        if not allow_live_real:
            missing.append("config live.allow_live_real: true")
        if not i_understand_real_money:
            missing.append("CLI flag --i-understand-real-money")
        raise RealMoneyGuardError("broker account is NOT a demo account; refusing to run. Real-money trading "
                                  "requires " + " AND ".join(missing))
    return True


class RunnerLockedError(RuntimeError):
    """Another runner process already owns this ``state_dir`` (two runners = double orders)."""


def _lock_state_dir(path: Path) -> Any:
    """Exclusive, non-blocking OS lock on ``path`` (``fcntl.flock`` / ``msvcrt.locking``).

    The OS drops the lock when the process dies, so a crash never leaves a stale lock. Two
    runners on one state directory would each decide every bar; their OMS states are separate
    in memory, so a race between reading positions and sending could double the position.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fh = os.fdopen(fd, "r+")
    try:
        if os.name == "nt":  # pragma: no cover - Windows (the MT5 platform)
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise RunnerLockedError(f"{path} is locked by another running LiveRunner; refusing to start a second "
                                f"one on the same state directory") from exc
    try:  # informational only (who holds the lock); never fail the start over it
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
    except OSError:  # pragma: no cover - platform specific
        pass
    return fh


def _unlock_state_dir(fh: Any) -> None:
    if fh is None:
        return
    try:
        if os.name == "nt":  # pragma: no cover
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:  # pragma: no cover
        pass
    finally:
        fh.close()


def real_money_banner(account: Any, *, dry_run: bool) -> str:
    bar = "!" * 78
    lines = [bar, "!!!  REAL-MONEY ACCOUNT  -- live orders will move real funds".ljust(75) + "!!!",
             f"!!!  equity {getattr(account, 'equity', float('nan')):,.2f} {getattr(account, 'currency', '')}"
             f"  server {getattr(account, 'server', '?')}".ljust(75) + "!!!",
             ("!!!  dry_run=True: orders are NOT sent (planning only)" if dry_run
              else "!!!  dry_run=False: ORDERS WILL BE SENT").ljust(75) + "!!!", bar]
    return "\n".join(lines)


# =============================================================================================
# desk data provider (live)
# =============================================================================================
class LiveDeskDataProvider:
    """:class:`~aurum.agents.providers.DeskDataProvider` over the runner's CURRENT window.

    Re-pointed every cycle (:meth:`refresh`) at a point-in-time
    :class:`~aurum.agents.providers.HistoricalDeskDataProvider` built from the live bars,
    macro, calendar, per-strategy forecasts and hooks into the live risk manager/positions.
    """

    def __init__(self) -> None:
        self._inner: Any = None

    def refresh(self, md: MarketData, *, signals: pd.DataFrame | None, combined: pd.Series | None,
                vol: pd.Series | None, backtest_stats: Mapping[str, Any] | None,
                risk_status_fn: Callable[[pd.Timestamp], Mapping[str, Any]] | None,
                positions_fn: Callable[[pd.Timestamp], Mapping[str, Any]] | None) -> None:
        from aurum.agents.providers import HistoricalDeskDataProvider

        self._inner = HistoricalDeskDataProvider(md, signals=signals, combined=combined, vol=vol,
                                                 backtest_stats=dict(backtest_stats or {}),
                                                 risk_status_fn=risk_status_fn, positions_fn=positions_fn)

    def _call(self, name: str, now: pd.Timestamp) -> dict[str, Any]:
        if self._inner is None:
            return {"available": False, "note": "no live data yet"}
        return getattr(self._inner, name)(now)

    def as_of(self, now: pd.Timestamp) -> str:
        return utc(now).isoformat() if self._inner is None else self._inner.as_of(now)

    def market_snapshot(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("market_snapshot", now)

    def quant_signals(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("quant_signals", now)

    def risk_status(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("risk_status", now)

    def macro_snapshot(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("macro_snapshot", now)

    def calendar(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("calendar", now)

    def backtest_stats(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("backtest_stats", now)

    def positions(self, now: pd.Timestamp) -> dict[str, Any]:
        return self._call("positions", now)


# =============================================================================================
# runner
# =============================================================================================
@dataclass
class CycleResult:
    """Outcome of one decision (also written to the decision log)."""

    time: pd.Timestamp
    bar_time: pd.Timestamp
    status: str
    forecasts: dict[str, float] = field(default_factory=dict)
    combined: float | None = None
    final_forecast: float | None = None
    desk: dict[str, Any] | None = None
    vol_ann: float | None = None
    equity: float | None = None
    current_lots: float | None = None
    requested_lots: float | None = None
    approved_lots: float | None = None
    halted: bool = False
    risk_reasons: list[str] = field(default_factory=list)
    report: ExecutionReport | None = None
    errors: list[str] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Pending:
    """A decision the OMS could not execute because the market was closed."""

    decision_time: pd.Timestamp
    target: float
    kwargs: dict[str, Any]
    reference: dict[str, float]
    next_try: pd.Timestamp
    attempts: int = 0


def _sign(x: float) -> int:
    return 1 if x > _EPS else (-1 if x < -_EPS else 0)


def _is_outcome_column(name: object) -> bool:
    low = str(name).strip().lower()
    return low in OUTCOME_COLUMNS or low.startswith(("actual", "surprise"))


def _finite(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class LiveRunner:
    """Bar-close scheduler driving artifact -> desk -> sizer -> risk -> OMS on a broker.

    Parameters
    ----------
    config        : :class:`LiveConfig`.
    broker        : venue (:class:`~aurum.live.paper.PaperBroker`, :class:`~aurum.live.mt5.MT5Broker`).
    artifact      : a loaded :class:`TradingArtifact` (default: ``load_artifact(config.artifact_dir)``).
    clock         : waiting/scheduling clock; a :class:`~aurum.live.broker.SimulatedClock` makes
                    replays instant. Default: the paper broker's clock or the system clock.
    instrument    : contract specification.
    desk          : an already-built ``TradingDesk`` (tests); otherwise built when
                    ``config.desk["enabled"]`` with ``desk_client`` (None = anthropic from env).
    macro_provider / events_provider : ``now -> frames`` hooks (default: ``config.macro_dir`` and
                    the rule-based calendar). Frames are filtered point-in-time by the runner.
    alert_sinks   : extra alert sinks (log, JSONL and the env webhook are added by default).
    i_understand_real_money : the CLI flag (second opt-in of the real-money guard).
    """

    def __init__(
        self,
        config: LiveConfig,
        *,
        broker: Broker,
        artifact: TradingArtifact | None = None,
        clock: Clock | None = None,
        instrument: Instrument = XAUUSD,
        desk: Any | None = None,
        desk_client: Any | None = None,
        macro_provider: Callable[[pd.Timestamp], Mapping[str, pd.DataFrame]] | None = None,
        events_provider: Callable[[pd.Timestamp], pd.DataFrame | None] | None = None,
        alert_sinks: list[Any] | None = None,
        i_understand_real_money: bool = False,
    ) -> None:
        self.config = config
        self.broker = broker
        self.instrument = instrument
        self.clock: Clock = clock or getattr(broker, "clock", None) or SystemClock()
        self.tf = get_timeframe(config.timeframe)
        self.state_dir = Path(config.state_dir)
        self._i_understand = bool(i_understand_real_money)
        self._artifact = artifact
        self._desk = desk
        self._desk_client = desk_client
        self._macro_provider = macro_provider
        self._events_provider = events_provider
        self._extra_sinks = list(alert_sinks or [])
        self._stop = threading.Event()
        if isinstance(self.clock, SystemClock) and self.clock.stop_event is None:
            self.clock.stop_event = self._stop
        self._started = False
        self._pending: _Pending | None = None
        self._lock_fh: Any = None
        self._cooldown_left = 0
        self._cooldown_side = 0
        self._last_exec: dict[str, Any] | None = None
        self._last_bar: pd.Timestamp | None = None
        self._last_avail: pd.Timestamp | None = None
        self._macro_cache: tuple[pd.Timestamp, dict[str, pd.DataFrame]] | None = None
        self._events_cache: tuple[pd.Timestamp, pd.DataFrame | None] | None = None
        self._macro_stale_alerted: str | None = None
        self._halt_alerted = False
        self._risk_events_obj: Any = None
        self._hb_last: tuple[float, str] | None = None
        self.heartbeat_interval_s = 5.0
        self.max_consecutive_failures = 5
        self.n_decisions = 0
        self.real_money = False
        self.mode = "unknown"
        self.cycles: list[CycleResult] = []
        self.keep_cycles = 1000

    # ------------------------------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------------------------------
    @property
    def decision_log_path(self) -> Path:
        return self.state_dir / "decisions.jsonl"

    @property
    def heartbeat_path(self) -> Path:
        return self.state_dir / "heartbeat.json"

    @property
    def runner_state_path(self) -> Path:
        return self.state_dir / "runner_state.json"

    @property
    def artifact(self) -> TradingArtifact:
        if self._artifact is None:
            if not self.config.artifact_dir:
                raise ArtifactError("no artifact given and config.artifact_dir is not set")
            self._artifact = load_artifact(self.config.artifact_dir)
        return self._artifact

    # ------------------------------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------------------------------
    def start(self) -> None:
        """Safety checks and construction of risk/sizer/OMS/monitor/desk (idempotent)."""
        if self._started:
            return
        cfg = self.config
        account = self.broker.account()
        is_demo = bool(self.broker.is_demo()) and bool(account.is_demo)
        self.real_money = check_real_money_guard(is_demo, allow_live_real=cfg.allow_live_real,
                                                 i_understand_real_money=self._i_understand)
        if self.real_money:
            banner = real_money_banner(account, dry_run=cfg.dry_run)
            for line in banner.splitlines():
                logger.critical(line)
        if account.currency and account.currency.upper() != "USD":
            logger.warning("account currency %s: sizing and PnL assume USD", account.currency)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock_fh = _lock_state_dir(self.state_dir / "runner.lock")
        try:
            self._start_locked(account)
        except BaseException:
            _unlock_state_dir(self._lock_fh)
            self._lock_fh = None
            raise

    def _start_locked(self, account: Any) -> None:
        cfg = self.config
        self.mode = ("DRY-RUN " if cfg.dry_run else "") + ("REAL" if self.real_money else
                                                          "PAPER" if type(self.broker).__name__ == "PaperBroker"
                                                          else "DEMO")
        art = self.artifact
        if art.timeframe and get_timeframe(art.timeframe).name != self.tf.name:
            raise ArtifactError(f"artifact was built for {art.timeframe} bars, config trades {self.tf.name}")
        if art.symbol and art.symbol != cfg.symbol:
            logger.warning("artifact symbol %s differs from venue symbol %s (suffix?)", art.symbol, cfg.symbol)

        lookback = art.max_lookback
        self.min_bars = max(lookback + 1, 30)
        self.n_history = int(cfg.history_bars) if cfg.history_bars else max(
            int(math.ceil(cfg.history_multiple * max(lookback, 1))), cfg.min_history_bars, self.min_bars)
        if self.n_history < self.min_bars:
            raise ValueError(f"history_bars={self.n_history} < required warm-up {self.min_bars}")

        self.state_dir.mkdir(parents=True, exist_ok=True)
        limits = {"stale_data_seconds": max(120.0, 0.5 * cfg.tf_seconds) + cfg.bar_close_delay_seconds,
                  **cfg.risk}
        self.risk = StandardRiskManager(RiskLimits(**limits), self.instrument,
                                        state_path=self.state_dir / "risk_state.json")
        if self.risk.halted:
            logger.critical("risk manager starts HALTED (%s): the runner will only flatten until "
                            "reset_halt(confirm='RESET')", self.risk.halt_reason)
        self.sizer = VolTargetSizer(**{**art.sizer_config, **cfg.sizer})
        self.costs = CostModel(**cfg.costs) if cfg.costs else getattr(self.broker, "costs", CostModel())
        oms_kw = dict(cfg.oms)
        self.oms = OrderManager(self.broker, self.instrument, cfg.magic, symbol=cfg.symbol,
                                state_path=self.state_dir / "oms_state.json", dry_run=cfg.dry_run,
                                sleep=self.clock.sleep, **oms_kw)
        self._build_monitor()
        self._build_desk()
        self._load_runner_state()
        self._started = True
        self._log({"type": "start", "time": self.broker.server_time(), "mode": self.mode,
                   "dry_run": cfg.dry_run, "artifact": str(art.path) if art.path else None,
                   "artifact_created": art.manifest.get("created_at"), "strategies": sorted(art.strategies),
                   "n_history": self.n_history, "min_bars": self.min_bars, "account": account.to_dict(),
                   "risk_halted": self.risk.halted})
        logger.info("LiveRunner started [%s] %s %s magic=%d history=%d bars (warm-up %d)", self.mode,
                    cfg.symbol, self.tf.name, cfg.magic, self.n_history, self.min_bars)

    def _build_monitor(self) -> None:
        mc = self.config.monitor
        sinks: list[Any] = [LogAlertSink(), JsonlAlertSink(self.state_dir / "alerts.jsonl")]
        if mc.get("webhook", True):
            wh = WebhookAlertSink(style=mc.get("webhook_style", "generic"),
                                  min_level=mc.get("webhook_min_level", "warning"))
            if wh.enabled:
                sinks.append(wh)
        sinks += self._extra_sinks
        alerts = AlertManager(sinks, cooldown_seconds=float(mc.get("alert_cooldown_seconds", 900.0)))
        art = self.artifact
        ref = art.feature_reference
        if ref is None and art.pipeline is not None:
            ref = FeatureReference.from_pipeline(art.pipeline)
        drift = (DriftMonitor(ref, warn=float(mc.get("psi_warn", 0.10)), alert=float(mc.get("psi_alert", 0.25)),
                              min_obs=int(mc.get("drift_min_obs", 100)))
                 if ref is not None and ref.columns else None)
        pnl = PnLBand.from_stats(art.backtest_stats, z_warn=float(mc.get("z_warn", -2.0)),
                                 z_alert=float(mc.get("z_alert", -3.0)))
        self.monitor = LiveMonitor(alerts=alerts, drift=drift, slippage=SlippageTracker(self.costs, self.instrument),
                                   pnl=pnl, drift_every=int(mc.get("drift_every", 24)),
                                   drift_window=int(mc.get("drift_window", 500)))

    def _build_desk(self) -> None:
        dc = self.config.desk
        if self._desk is None and not dc.get("enabled", False):
            self.desk = None
            self.desk_provider = None
            return
        if self._desk is not None:
            self.desk = self._desk
            prov = getattr(self.desk, "provider", None)
            self.desk_provider = prov if isinstance(prov, LiveDeskDataProvider) else None
            return
        from aurum.agents import DecisionPolicy, DeskConfig, TradingDesk

        self.desk_provider = LiveDeskDataProvider()
        policy = DecisionPolicy(mode=dc.get("mode", "overlay"), on_failure=dc.get("on_failure", "follow_quant"),
                                max_abs_forecast=float(dc.get("max_abs_forecast", 1.0)),
                                min_confidence=float(dc.get("min_confidence", 0.0)))
        self.desk = TradingDesk(self.desk_provider, client=self._desk_client,
                                config=DeskConfig(**dict(dc.get("config", {}))), policy=policy,
                                journal_dir=self.state_dir / "desk_journal")

    def _load_runner_state(self) -> None:
        try:
            st = read_json(self.runner_state_path, default={}) or {}
        except StateCorruptError:
            logger.error("runner state %s unreadable; starting without it (OMS idempotency still applies)",
                         self.runner_state_path)
            st = {}
        if st.get("last_bar"):
            self._last_bar = utc(st["last_bar"])
            self._last_avail = utc(st["last_avail"]) if st.get("last_avail") else self._last_bar + self.tf.delta
        self.n_decisions = int(st.get("n_decisions", 0))
        self._cooldown_left = int(st.get("cooldown_left", 0) or 0)
        self._cooldown_side = int(st.get("cooldown_side", 0) or 0)
        le = st.get("last_exec")
        if isinstance(le, dict) and le.get("time"):
            try:
                self._last_exec = {"time": utc(le["time"]), "side": int(le.get("side", 0)),
                                   "sl": _finite(le.get("sl")) if le.get("sl") is not None else None,
                                   "tp": _finite(le.get("tp")) if le.get("tp") is not None else None}
            except (TypeError, ValueError) as exc:
                logger.error("ignoring unreadable last_exec in runner state: %s", exc)
        if self.monitor.pnl is not None:
            self.monitor.pnl.load_state(st.get("pnl_band"))
        pend = st.get("pending")
        if pend:
            try:
                self._pending = _Pending(decision_time=utc(pend["decision_time"]), target=float(pend["target"]),
                                         kwargs=dict(pend.get("kwargs") or {}),
                                         reference={k: float(v) for k, v in (pend.get("reference") or {}).items()},
                                         next_try=utc(pend["next_try"]), attempts=int(pend.get("attempts", 0)))
                logger.warning("restored deferred order intent for %s (target %+.2f)",
                               self._pending.decision_time, self._pending.target)
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("ignoring unreadable deferred intent in runner state: %s", exc)

    def _save_runner_state(self) -> None:
        p = self._pending
        atomic_write_json(self.runner_state_path, {
            "last_bar": self._last_bar, "last_avail": self._last_avail, "n_decisions": self.n_decisions,
            "cooldown_left": self._cooldown_left, "cooldown_side": self._cooldown_side,
            "last_exec": self._last_exec,
            "pnl_band": self.monitor.pnl.to_state() if self.monitor.pnl is not None else None,
            "pending": None if p is None else {"decision_time": p.decision_time, "target": p.target,
                                               "kwargs": p.kwargs, "reference": p.reference,
                                               "next_try": p.next_try, "attempts": p.attempts},
            "updated": self.broker.server_time()})

    # ------------------------------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------------------------------
    def _macro(self, now: pd.Timestamp, decision_time: pd.Timestamp) -> dict[str, pd.DataFrame]:
        cfg = self.config
        frames: Mapping[str, pd.DataFrame] | None = None
        if self._macro_provider is not None:
            frames = self._macro_provider(now)
        elif cfg.macro_dir:
            refresh = pd.Timedelta(hours=cfg.macro_refresh_hours)
            if self._macro_cache is None or now - self._macro_cache[0] >= refresh:
                from aurum.data.macro import load_macro_dir

                try:
                    self._macro_cache = (now, load_macro_dir(cfg.macro_dir))
                    logger.info("macro data reloaded from %s (%d series)", cfg.macro_dir, len(self._macro_cache[1]))
                except Exception as exc:
                    logger.error("macro reload failed (%s); keeping previous data", exc)
                    if self._macro_cache is None:
                        self._macro_cache = (now, {})
            frames = self._macro_cache[1]
        if not frames:
            return {}
        out: dict[str, pd.DataFrame] = {}
        newest = None
        for name, f in frames.items():
            if f is None or len(f) == 0 or "available_at" not in f.columns:
                continue
            avail = pd.DatetimeIndex(f["available_at"])
            g = f.loc[avail <= decision_time]  # point in time: nothing published after the decision
            if len(g):
                out[name] = g
                last = pd.Timestamp(g["available_at"].iloc[-1])
                newest = last if newest is None else max(newest, last)
        if newest is not None and decision_time - newest > pd.Timedelta(days=cfg.macro_max_age_days):
            day = f"{decision_time:%Y-%m-%d}"
            if self._macro_stale_alerted != day:
                self._macro_stale_alerted = day
                self.monitor.alerts.alert("warning", "macro_stale",
                                          f"newest macro observation is from {newest} (> {cfg.macro_max_age_days} days)",
                                          time=decision_time)
        return out

    def _events(self, now: pd.Timestamp, decision_time: pd.Timestamp) -> pd.DataFrame | None:
        cfg = self.config
        if self._events_provider is not None:
            ev = self._events_provider(now)
        else:
            if cfg.calendar in (None, "none") and not cfg.calendar_csv:
                return None
            if self._events_cache is None or now - self._events_cache[0] >= pd.Timedelta(hours=24):
                frames = []
                try:
                    if cfg.calendar == "rule_based":
                        from aurum.data.calendar import generate_rule_based_calendar

                        frames.append(generate_rule_based_calendar(now - pd.Timedelta(days=120),
                                                                   now + pd.Timedelta(days=45)))
                    if cfg.calendar_csv:
                        from aurum.data.calendar import load_calendar_csv

                        frames.append(load_calendar_csv(cfg.calendar_csv))
                    if len(frames) > 1:
                        from aurum.data.calendar import merge_calendars

                        ev_all = merge_calendars(*frames)
                    else:
                        ev_all = frames[0] if frames else None
                except Exception as exc:
                    logger.error("calendar refresh failed (%s); keeping previous events", exc)
                    ev_all = self._events_cache[1] if self._events_cache is not None else None
                self._events_cache = (now, ev_all)
            ev = self._events_cache[1]
        if ev is not self._risk_events_obj:
            # scheduled times drive the risk manager's event blackouts (public in advance)
            self._risk_events_obj = ev
            self.risk.set_events(ev)
        if ev is None:
            return None
        ev = ev.copy()
        if len(ev) and "time" in ev.columns:
            future = pd.to_datetime(ev["time"], utc=True) > decision_time
            for c in ev.columns:
                if _is_outcome_column(c):
                    ev[c] = ev[c].astype(object).where(~future, None)
        return ev

    # ------------------------------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------------------------------
    def request_stop(self) -> None:
        """Ask the loop to finish the current cycle and exit (signal-safe)."""
        self._stop.set()

    def run(self, *, until: pd.Timestamp | None = None, max_cycles: int | None = None,
            install_signal_handlers: bool = True) -> list[CycleResult]:
        """Run until ``until`` (broker time), ``max_cycles`` decisions, or a stop request."""
        self.start()
        previous: dict[int, Any] = {}
        if install_signal_handlers and threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    previous[sig] = signal.signal(sig, self._on_signal)
                except (ValueError, OSError):  # pragma: no cover - platform specific
                    pass
        n = 0
        results: list[CycleResult] = []
        status = "running"
        failures = 0
        try:
            while not self._stop.is_set():
                try:
                    now = self.broker.server_time()
                    if until is not None and now >= until:
                        break
                    res = self.run_once(now)
                    failures = 0
                except (BrokerError, OSError) as exc:
                    # Venue/transport trouble (terminal disconnected, file system hiccup): alert,
                    # back off, try to reconnect; programming errors still crash loudly below.
                    failures += 1
                    level = "critical" if failures >= self.max_consecutive_failures else "warning"
                    logger.error("broker/IO failure %d in the runner loop: %s", failures, exc)
                    self.monitor.alerts.alert(level, "broker_error", f"{type(exc).__name__}: {exc}",
                                              time=self.clock.now())
                    self._heartbeat("degraded", force=True)
                    reconnect = getattr(self.broker, "reconnect", None)
                    if callable(reconnect):
                        try:
                            reconnect()
                        except Exception as rexc:  # pragma: no cover - venue specific
                            logger.error("reconnect failed: %s", rexc)
                    backoff = min(self.config.retry_poll_seconds * 2 ** min(failures - 1, 6),
                                  self.config.retry_poll_max_seconds)
                    self._sleep_until(self.clock.now() + pd.Timedelta(seconds=backoff))
                    continue
                if res is not None:
                    results.append(res)
                    n += 1
                    if max_cycles is not None and n >= max_cycles:
                        break
                self._heartbeat("running")
                wake = self._next_wake(self.broker.server_time())
                if until is not None:
                    wake = min(wake, until)
                self._sleep_until(wake)
        except BaseException as exc:
            status = f"error: {type(exc).__name__}"
            logger.exception("runner loop terminated by an exception")
            try:
                self.monitor.alerts.alert("critical", "runner_crash", f"runner stopped: {type(exc).__name__}: {exc}")
            except Exception:  # pragma: no cover
                pass
            raise
        finally:
            for sig, h in previous.items():
                try:
                    signal.signal(sig, h)
                except (ValueError, OSError):  # pragma: no cover
                    pass
            self.shutdown(status="stopped" if status == "running" else status)
        return results

    def _on_signal(self, signum: int, frame: Any) -> None:  # pragma: no cover - exercised manually
        if self._stop.is_set():
            logger.warning("second signal %d: forcing exit", signum)
            raise KeyboardInterrupt
        logger.warning("signal %d received: finishing the current cycle and shutting down", signum)
        self._stop.set()

    def _sleep_until(self, t: pd.Timestamp) -> None:
        while not self._stop.is_set():
            now = self.clock.now()
            if now >= t:
                return
            remaining = (t - now).total_seconds()
            self.clock.sleep(remaining if self.clock.simulated else min(remaining, 1.0))

    def _next_wake(self, now: pd.Timestamp) -> pd.Timestamp:
        delay = pd.Timedelta(seconds=self.config.bar_close_delay_seconds)
        tf = self.tf.delta
        if self._last_avail is not None:
            nxt = self._last_avail + tf
            if nxt + delay <= now:
                k = int((now - delay - nxt) / tf) + 1
                nxt = nxt + k * tf
        else:
            nxt = now.floor(tf) + tf if self.tf.minutes < 1440 else now.normalize() + pd.Timedelta(days=1)
        wake = nxt + delay
        if self._pending is not None:
            wake = min(wake, max(self._pending.next_try, now + pd.Timedelta(seconds=1)))
        return max(wake, now + pd.Timedelta(seconds=1))

    def shutdown(self, *, status: str = "stopped") -> None:
        """Flush logs and heartbeat; optionally flatten (``flatten_on_shutdown``)."""
        if not self._started:
            return
        try:
            now = self.broker.server_time()
        except Exception:  # never mask the exception that brought us here
            logger.exception("broker time unavailable during shutdown")
            now = self.clock.now()
        if self.config.flatten_on_shutdown and not self.config.dry_run:
            try:
                rep = self.oms.flatten(now.floor("min"), reason="shutdown")
                self._log({"type": "shutdown_flatten", "time": now, "execution": rep.to_dict()})
            except Exception:
                logger.exception("flatten on shutdown failed")
        self._log({"type": "stop", "time": now, "status": status, "n_decisions": self.n_decisions})
        try:
            self._heartbeat(status, force=True)
        except Exception:  # pragma: no cover
            logger.exception("final heartbeat failed")
        if hasattr(self.broker, "shutdown") and not isinstance(self.clock, SimulatedClock):
            try:
                self.broker.shutdown()
            except Exception:  # pragma: no cover
                logger.exception("broker shutdown failed")
        self._started = False
        _unlock_state_dir(self._lock_fh)
        self._lock_fh = None
        logger.info("LiveRunner stopped (%s) after %d decisions", status, self.n_decisions)

    def _heartbeat(self, status: str, *, force: bool = False) -> None:
        """Rewrite the heartbeat file (throttled to one write per ``heartbeat_interval_s`` of
        WALL time unless the status changes — a watchdog measures real time)."""
        mono = time.monotonic()
        if (not force and self._hb_last is not None and self._hb_last[1] == status
                and mono - self._hb_last[0] < self.heartbeat_interval_s):
            return
        self._hb_last = (mono, status)
        try:
            write_heartbeat(self.heartbeat_path, status=status, mode=self.mode, broker_time=self.broker.server_time(),
                            last_bar=self._last_bar, n_decisions=self.n_decisions,
                            pending=self._pending.decision_time if self._pending else None,
                            halted=self.risk.halted if hasattr(self, "risk") else None)
        except OSError as exc:  # pragma: no cover
            logger.error("heartbeat write failed: %s", exc)

    def _log(self, record: Mapping[str, Any]) -> None:
        try:
            append_jsonl(self.decision_log_path, record)
        except (OSError, TypeError, ValueError) as exc:
            logger.error("decision log write failed: %s", exc)

    # ------------------------------------------------------------------------------------------
    # one cycle
    # ------------------------------------------------------------------------------------------
    def run_once(self, now: pd.Timestamp | None = None) -> CycleResult | None:
        """Process a newly closed bar if there is one (else retry a deferred order)."""
        self.start()
        now = self.broker.server_time() if now is None else now
        bars = self.broker.latest_bars(self.config.symbol, self.tf.name, self.n_history)
        if len(bars):
            closed = pd.DatetimeIndex(bars["available_at"]) <= now
            if not closed.all():
                # Defence in depth: a venue adapter / clock bug must never let a decision see a
                # bar that has not closed yet (SPEC §0-1).
                msg = f"broker returned {int((~closed).sum())} bar(s) not closed at {now}: dropped"
                logger.error(msg)
                self.monitor.alerts.alert("critical", "forming_bar", msg, time=now)
                bars = bars.loc[closed]
        if len(bars) and (self._last_bar is None or bars.index[-1] > self._last_bar):
            return self._decide(bars, now)
        if self._pending is not None:
            self._try_pending(now)
        return None

    def _decide(self, bars: pd.DataFrame, now: pd.Timestamp) -> CycleResult:
        cfg = self.config
        art = self.artifact
        t_dec = utc(bars["available_at"].iloc[-1])
        bar_time = utc(bars.index[-1])
        age = (now - t_dec).total_seconds()
        close = float(bars["close"].iloc[-1])
        bar_spread = float(bars["spread"].iloc[-1])
        res = CycleResult(time=t_dec, bar_time=bar_time, status="pending")
        rec: dict[str, Any] = {"type": "decision", "time": t_dec, "bar_time": bar_time, "now": now,
                               "mode": self.mode, "dry_run": cfg.dry_run, "price": close, "bar_spread": bar_spread,
                               "data_age_s": age, "n_bars": len(bars)}
        if self._pending is not None:
            rec["superseded_pending"] = self._pending.decision_time
            self._pending = None

        max_age = cfg.max_bar_age_seconds if cfg.max_bar_age_seconds is not None else (
            cfg.tf_seconds + cfg.bar_close_delay_seconds + 120.0)
        if len(bars) < self.min_bars:
            res.status = "insufficient_history"
            res.errors.append(f"{len(bars)} closed bars < warm-up {self.min_bars}")
            self.monitor.alerts.alert("warning", "insufficient_history", res.errors[-1], time=t_dec)
            self._flatten_if_halted(res, rec, bars, t_dec, now)
            return self._finish(res, rec, bar_time, t_dec)
        if age > max_age:
            res.status = "stale"
            res.errors.append(f"latest bar closed {age:.0f}s ago (> {max_age:.0f}s): not trading on it")
            self.monitor.alerts.alert("warning", "stale_data", res.errors[-1], time=now)
            self._flatten_if_halted(res, rec, bars, t_dec, now)
            return self._finish(res, rec, bar_time, t_dec)

        # ---- signals -------------------------------------------------------------------
        forecasts: dict[str, float] = {}
        frames: dict[str, pd.Series] = {}
        feats: pd.DataFrame | None = None
        md: MarketData | None = None
        signal_error: str | None = None
        try:
            md = MarketData(bars=bars, macro=self._macro(now, t_dec), events=self._events(now, t_dec))
            feats = art.features(md)
            for name, strat in art.strategies.items():
                try:
                    f = strat.generate(md, feats)
                    frames[name] = pd.Series(f, index=bars.index, dtype=float).reindex(bars.index)
                    forecasts[name] = float(np.clip(_finite(frames[name].iloc[-1], 0.0), -1.0, 1.0))
                except Exception as exc:
                    logger.exception("strategy %s failed", name)
                    res.errors.append(f"strategy {name}: {type(exc).__name__}: {exc}")
                    frames[name] = pd.Series(0.0, index=bars.index)
                    forecasts[name] = 0.0
                    self.monitor.alerts.alert("critical", "strategy_error", res.errors[-1], time=t_dec)
            if len(res.errors) == len(art.strategies):
                raise RuntimeError("every strategy failed")
            fc_frame = pd.DataFrame(frames, index=bars.index).fillna(0.0).clip(-1.0, 1.0)
            combined_series = art.combine(fc_frame)
            combined = float(np.clip(_finite(combined_series.iloc[-1], 0.0), -1.0, 1.0))
        except Exception as exc:
            logger.exception("signal computation failed at %s", t_dec)
            signal_error = f"{type(exc).__name__}: {exc}"
            res.errors.append(signal_error)
            self.monitor.alerts.alert("critical", "signal_error", f"signal computation failed: {signal_error}",
                                      time=t_dec)
            combined_series = None
            combined = 0.0
        res.forecasts = forecasts
        res.combined = combined
        rec.update(forecasts=forecasts, combined=combined)

        # ---- account & risk bookkeeping --------------------------------------------------
        account = self.broker.account()
        positions = self.broker.positions(cfg.symbol, cfg.magic)
        current = net_lots(positions)
        equity = float(account.equity)
        self.risk.on_bar(t_dec, equity)
        peak = self.risk.state.peak_equity or equity
        dd = max(0.0, 1.0 - equity / peak) if peak > 0 else 0.0
        vol_series = ewma_volatility(bars["close"].astype(float))
        vol = _finite(vol_series.iloc[-1], 0.20)
        res.vol_ann, res.equity, res.current_lots = vol, equity, current
        rec.update(equity=equity, balance=account.balance, current_lots=current, drawdown=dd, vol_ann=vol,
                   positions=[p.to_dict() for p in positions])
        exit_reason = self._detect_protective_exit(bars, current, t_dec)
        if exit_reason is not None:
            rec["protective_exit"] = exit_reason

        # ---- LLM desk --------------------------------------------------------------------
        final = combined
        if signal_error is None and self.desk is not None:
            final, desk_info = self._run_desk(md, t_dec, combined, fc_frame, combined_series, vol_series,
                                              positions, account)
            res.desk = desk_info
            rec["desk"] = desk_info
        res.final_forecast = final
        rec["final_forecast"] = final

        # ---- sizing ----------------------------------------------------------------------
        if signal_error is not None:
            if cfg.on_error == "flatten":
                requested = 0.0
                rec["on_error"] = "flatten"
            else:
                requested = current
                rec["on_error"] = "hold"
        else:
            bd = self.sizer.breakdown(final, vol, equity, close, self.instrument, current_lots=current, drawdown=dd)
            rec["sizing"] = bd.to_dict()
            raw_lots = _finite(bd.final_lots)
            if math.isnan(raw_lots):
                # Same rule as the backtest engine: a non-finite size HOLDS the position.
                # (Instrument.round_lots maps NaN to 0.0, which would silently liquidate.)
                msg = f"sizer returned non-finite lots ({bd.final_lots!r}); holding {current:+.2f}"
                logger.error(msg)
                res.errors.append(msg)
                self.monitor.alerts.alert("critical", "sizing_error", msg, time=t_dec)
                requested = current
            else:
                requested = self.instrument.round_lots(raw_lots)
        res.requested_lots = requested
        rec["requested_lots"] = requested
        # Post-stop cooldown is part of the ORDER (as in the backtest engine), applied before
        # risk so that stateful limits evaluate what will actually be sent.
        order = requested
        if self._cooldown_left > 0:
            if abs(order) > _EPS and _sign(order) == self._cooldown_side:
                rec["stop_cooldown"] = f"stop_cooldown ({self._cooldown_left} bars left)"
                order = 0.0
            self._cooldown_left -= 1
        rec["order_lots"] = order

        # ---- risk ------------------------------------------------------------------------
        spread = bar_spread
        quote = None
        if cfg.spread_source == "quote":
            quote = self.broker.quote(cfg.symbol)
            if quote is not None:
                spread = quote.spread
        ctx = RiskContext(time=t_dec, equity=equity, current_lots=current, target_lots=order, price=close,
                          spread=spread, vol_ann=vol, data_age_seconds=age,
                          extra={"drawdown": dd, "bar_time": bar_time, "forecast": final, "mode": self.mode})
        decision = self.risk.evaluate(ctx)
        approved = 0.0 if decision.halted else float(decision.approved_lots)
        lo, hi = min(0.0, order), max(0.0, order)
        if not (lo - _EPS <= approved <= hi + _EPS):
            logger.error("risk approval %.4f outside [0, requested %.4f]: clamped", approved, order)
            approved = min(max(approved, lo), hi)
        approved = self.instrument.round_lots(approved)
        res.approved_lots, res.halted, res.risk_reasons = approved, bool(decision.halted), list(decision.reasons)
        rec.update(approved_lots=approved, risk={"halted": bool(decision.halted), "reasons": list(decision.reasons),
                                                 "snapshot": self.risk.snapshot()})
        if decision.halted and not self._halt_alerted:
            self._halt_alerted = True
            self.monitor.alerts.alert("critical", "risk_halt", f"risk kill switch engaged: {self.risk.halt_reason}",
                                      time=t_dec)
        elif not decision.halted:
            self._halt_alerted = False

        # ---- orders ----------------------------------------------------------------------
        kwargs = self._protective_kwargs(bars, approved, current, positions)
        rec["protective"] = kwargs
        if signal_error is not None and cfg.on_error == "hold" and abs(approved - current) < _EPS:
            res.status = "error_hold"
            rec["execution"] = None
        else:
            report = self.oms.reconcile(approved, t_dec, reason="risk" if decision.halted else "signal", **kwargs)
            res.report = report
            rec["execution"] = report.to_dict()
            self._after_execution(report, reference={"mid": close, "spread": bar_spread,
                                                     "bar_range": float(bars["high"].iloc[-1] - bars["low"].iloc[-1])},
                                  time=now)
            if report.status == "deferred":
                self._pending = _Pending(decision_time=t_dec, target=approved, kwargs=kwargs,
                                         reference={"mid": close, "spread": bar_spread, "bar_range": 0.0},
                                         next_try=now + pd.Timedelta(seconds=cfg.retry_poll_seconds))
            res.status = ("error_" if signal_error else "") + report.status
        self._record_exec(t_dec)
        after = self.broker.account()
        rec["equity_after"] = after.equity
        self.monitor.on_decision(t_dec, equity=equity, features=feats)
        snap = self.monitor.snapshot()
        if self.monitor.n_decisions % self.monitor.drift_every != 0:
            snap.pop("drift", None)  # only log the drift report on the cycles that computed it
        rec["monitor"] = snap
        return self._finish(res, rec, bar_time, t_dec)

    def _finish(self, res: CycleResult, rec: dict[str, Any], bar_time: pd.Timestamp,
                t_dec: pd.Timestamp) -> CycleResult:
        rec["status"] = res.status
        rec["errors"] = list(res.errors)
        res.record = rec
        self._log(rec)
        self._last_bar = bar_time
        self._last_avail = t_dec
        self.n_decisions += 1
        self._save_runner_state()
        self.cycles.append(res)
        if len(self.cycles) > self.keep_cycles:
            del self.cycles[: len(self.cycles) - self.keep_cycles]
        logger.info("[%s] %s decision %s: combined %+.3f final %+.3f -> requested %+.2f approved %+.2f "
                    "(current %+.2f) status=%s", self.mode, self.config.symbol, t_dec,
                    res.combined if res.combined is not None else float("nan"),
                    res.final_forecast if res.final_forecast is not None else float("nan"),
                    res.requested_lots if res.requested_lots is not None else float("nan"),
                    res.approved_lots if res.approved_lots is not None else float("nan"),
                    res.current_lots if res.current_lots is not None else float("nan"), res.status)
        return res

    def _flatten_if_halted(self, res: CycleResult, rec: dict[str, Any], bars: pd.DataFrame,
                           t_dec: pd.Timestamp, now: pd.Timestamp) -> None:
        """On a bar skipped for data reasons, a HALTED book is still flattened: closing needs
        no signal, and the kill switch must not depend on the data feed being healthy."""
        if not self.risk.halted:
            return
        cfg = self.config
        current = net_lots(self.broker.positions(cfg.symbol, cfg.magic))
        if abs(current) < _EPS:
            return
        logger.critical("risk halted and bar %s skipped (%s): flattening %+.2f lots", t_dec, res.status, current)
        report = self.oms.reconcile(0.0, t_dec, reason="risk")
        res.report, res.halted, res.approved_lots, res.current_lots = report, True, 0.0, current
        rec["execution"] = report.to_dict()
        rec["halt_flatten"] = True
        ref = {"mid": float(bars["close"].iloc[-1]), "spread": float(bars["spread"].iloc[-1]), "bar_range": 0.0}
        self._after_execution(report, reference=ref, time=now)
        if report.status == "deferred":
            self._pending = _Pending(decision_time=t_dec, target=0.0, kwargs={}, reference=ref,
                                     next_try=now + pd.Timedelta(seconds=cfg.retry_poll_seconds))
        self._record_exec(t_dec)

    def _record_exec(self, t: pd.Timestamp) -> None:
        """Side and protective levels of our position after executing the decision at ``t``
        (input of the post-stop cooldown; skipped when no cooldown is configured)."""
        if int(self.config.stop_cooldown_bars) <= 0:
            return
        cfg = self.config
        pos = self.broker.positions(cfg.symbol, cfg.magic)
        side = _sign(net_lots(pos))
        same = [p for p in pos if _sign(p.lots) == side] if side else []
        self._last_exec = {"time": utc(t), "side": side,
                           "sl": next((float(p.sl) for p in same if p.sl), None),
                           "tp": next((float(p.tp) for p in same if p.tp), None)}

    def _detect_protective_exit(self, bars: pd.DataFrame, current: float, t_dec: pd.Timestamp) -> str | None:
        """Was our position closed by its broker-side stop since the last decision?

        The venue does not tell us why a position vanished, so the backtest engine's rule is
        replayed on the bars closed since then (:func:`aurum.live.paper._protective_exit`: gap
        through the level at the open, else stop before take-profit). A stop starts the
        ``stop_cooldown_bars`` cooldown on that side, exactly like ``run_backtest``.
        """
        k = int(self.config.stop_cooldown_bars)
        le = self._last_exec
        if k <= 0 or le is None or not le.get("side") or abs(current) > _EPS:
            return None
        if le.get("sl") is None and le.get("tp") is None:
            return None
        avail = pd.DatetimeIndex(bars["available_at"])
        win = bars.loc[(avail > le["time"]) & (avail <= t_dec)]
        for o, h, lo in zip(win["open"].to_numpy(float), win["high"].to_numpy(float),
                            win["low"].to_numpy(float), strict=True):
            ex = _protective_exit(float(le["side"]), o, h, lo, le.get("sl"), le.get("tp"))
            if ex is None:
                continue
            if ex[1] == "stop":
                self._cooldown_left, self._cooldown_side = k, int(le["side"])
                logger.info("protective stop detected since %s: %d-bar cooldown on side %+d", le["time"], k,
                            self._cooldown_side)
            self._last_exec = None
            return ex[1]
        return None

    def _protective_kwargs(self, bars: pd.DataFrame, approved: float, current: float,
                           positions: list[Any]) -> dict[str, Any]:
        """Broker-side SL/TP like the backtest engine: fresh ATR distances on ENTRY (anchored
        at the fill), existing levels kept for adds (hedging tickets inherit them)."""
        cfg = self.config
        if cfg.stop_atr_mult is None and cfg.take_profit_atr_mult is None:
            return {}
        s = _sign(approved)
        if s == 0:
            return {}
        if s != _sign(current):
            atr = float(average_true_range(bars, cfg.atr_period).iloc[-1])
            kw: dict[str, Any] = {}
            if cfg.stop_atr_mult is not None and atr > 0:
                kw["sl_distance"] = cfg.stop_atr_mult * atr
            if cfg.take_profit_atr_mult is not None and atr > 0:
                kw["tp_distance"] = cfg.take_profit_atr_mult * atr
            return kw
        same = [p for p in positions if _sign(p.lots) == s]
        kw = {}
        sl = next((p.sl for p in same if p.sl), None)
        tp = next((p.tp for p in same if p.tp), None)
        if sl is not None:
            kw["stop_loss"] = float(sl)
        if tp is not None:
            kw["take_profit"] = float(tp)
        return kw

    def _after_execution(self, report: ExecutionReport, *, reference: Mapping[str, float], time: pd.Timestamp) -> None:
        for lr in report.legs:
            if lr.fill is not None:
                self.monitor.on_fill(lr.fill, reference_mid=reference["mid"], spread=reference["spread"],
                                     bar_range=reference.get("bar_range", 0.0), time=time)
        if report.status in ("rejected", "conflict", "partial", "error", "unknown"):
            self.monitor.alerts.alert("critical" if report.status in ("conflict", "unknown") else "warning",
                                      f"execution_{report.status}",
                                      f"execution {report.status} at {report.decision_time}: {'; '.join(report.errors)[:400]}",
                                      time=time)

    def _try_pending(self, now: pd.Timestamp) -> None:
        p = self._pending
        assert p is not None
        cfg = self.config
        if now < p.next_try:
            return
        max_defer = cfg.max_defer_seconds if cfg.max_defer_seconds is not None else 1.5 * cfg.tf_seconds
        if (now - p.decision_time).total_seconds() > max_defer:
            cur = net_lots(self.broker.positions(cfg.symbol, cfg.magic))
            reduces = abs(p.target) < abs(cur) - _EPS and (abs(p.target) < _EPS or _sign(p.target) == _sign(cur))
            if not reduces:
                logger.warning("deferred decision %s expired after %.0fs without an open market", p.decision_time,
                               (now - p.decision_time).total_seconds())
                self._log({"type": "deferred_expired", "time": now, "decision_time": p.decision_time,
                           "target": p.target, "attempts": p.attempts})
                self._pending = None
                self._save_runner_state()
                return
            # Only NEW risk goes stale. A reduction (kill switch, blackout flatten, smaller
            # target) waits for the market: dropping it would carry the exposure past the reopen.
            if p.attempts == 0 or not p.reference.get("kept_past_expiry"):
                logger.warning("deferred decision %s (target %+.2f vs %+.2f held) reduces risk: kept past "
                               "max_defer_seconds until the market reopens", p.decision_time, p.target, cur)
                p.reference["kept_past_expiry"] = 1.0
        p.attempts += 1
        backoff = min(cfg.retry_poll_seconds * (2 ** min(p.attempts, 10)), cfg.retry_poll_max_seconds)
        if self.broker.quote(cfg.symbol) is None:
            p.next_try = now + pd.Timedelta(seconds=backoff)
            return
        if self.risk.halted and (abs(p.target) > _EPS or p.kwargs):
            # The intent was approved BEFORE the kill switch engaged (or it was restored from
            # disk into a halted process). A halted book may only go flat (SPEC §0.3).
            logger.critical("risk halted (%s): deferred decision %s (target %+.2f) replaced by a flatten",
                            self.risk.halt_reason, p.decision_time, p.target)
            self._log({"type": "deferred_halted", "time": now, "decision_time": p.decision_time,
                       "original_target": p.target, "halt_reason": self.risk.halt_reason})
            p.target, p.kwargs = 0.0, {}
        report = self.oms.reconcile(p.target, p.decision_time, **p.kwargs)
        if report.status == "deferred":
            p.next_try = now + pd.Timedelta(seconds=backoff)
            return
        self._pending = None
        self._record_exec(p.decision_time)
        self._save_runner_state()
        self._after_execution(report, reference=p.reference, time=now)
        self._log({"type": "deferred_execution", "time": now, "decision_time": p.decision_time,
                   "attempts": p.attempts, "execution": report.to_dict(),
                   "equity_after": self.broker.account().equity})

    def _run_desk(self, md: MarketData | None, t_dec: pd.Timestamp, combined: float, fc_frame: pd.DataFrame,
                  combined_series: pd.Series | None, vol_series: pd.Series, positions: list[Any],
                  account: Any) -> tuple[float, dict[str, Any]]:
        """LLM desk cycle; any failure falls back per ``desk.on_error`` (default follow_quant)."""
        dc = self.config.desk
        on_error = dc.get("on_error", "follow_quant")
        info: dict[str, Any] = {"quant_forecast": combined}
        try:
            if self.desk_provider is not None and md is not None:
                cs = combined_series if combined_series is not None else pd.Series(combined, index=md.bars.index)
                eq = float(account.equity)
                cs_contract = self.instrument.contract_size
                self.desk_provider.refresh(
                    md, signals=fc_frame, combined=cs.reindex(md.bars.index), vol=vol_series,
                    backtest_stats=self.artifact.backtest_stats,
                    risk_status_fn=lambda now: self.risk.snapshot(),
                    positions_fn=lambda now: {"net_lots": net_lots(positions),
                                              "exposure_pct_equity": 100.0 * net_lots(positions) * cs_contract
                                              * float(md.bars["close"].iloc[-1]) / eq if eq > 0 else None,
                                              "n_tickets": len(positions)})
            out = self.desk.run_cycle(t_dec, combined, context={"mode": self.mode, "dry_run": self.config.dry_run})
            final = float(np.clip(_finite(out.final_forecast, combined), -1.0, 1.0))
            info.update(status=out.status, final_forecast=final, failure_reason=out.failure_reason,
                        action=out.policy.action if out.policy is not None else None,
                        decision=out.decision.to_dict() if out.decision is not None else None,
                        cost_usd=out.cost_usd, cycle_id=out.cycle_id,
                        journal=str(out.journal_path) if out.journal_path else None)
            if out.status == "failed":
                self.monitor.alerts.alert("warning", "desk_failed", f"desk cycle failed: {out.failure_reason}",
                                          time=t_dec)
            return final, info
        except Exception as exc:
            logger.exception("LLM desk raised; falling back to %s", on_error)
            if on_error == "flat":
                final = 0.0
            elif on_error == "hold":
                prev = getattr(self.desk, "last_final_forecast", None)
                final = float(prev) if prev is not None else 0.0
                # overlay semantics: holding may not add risk relative to the quant forecast
                if _sign(final) != _sign(combined) or abs(final) > abs(combined):
                    final = 0.0 if _sign(final) != _sign(combined) else combined
            else:
                final = combined
            info.update(status="error", final_forecast=final, failure_reason=f"{type(exc).__name__}: {exc}",
                        fallback=on_error)
            self.monitor.alerts.alert("warning", "desk_error", f"desk error ({type(exc).__name__}); fallback "
                                      f"{on_error}", time=t_dec)
            return final, info


# =============================================================================================
# CLI
# =============================================================================================
def _as_utc(value: Any) -> pd.Timestamp:
    """Config/CLI time -> UTC Timestamp: naive values are UTC, aware ones are converted (YAML
    turns ``2026-09-01T00:00:00Z`` into an aware datetime, which ``Timestamp(v, tz=...)`` rejects)."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _build_broker(cfg: LiveConfig, instrument: Instrument) -> tuple[Broker, Clock, pd.Timestamp | None]:
    if cfg.broker == "mt5":
        from aurum.live.mt5 import MT5Broker

        broker = MT5Broker(cfg.symbol, instrument=instrument, **cfg.mt5)
        return broker, broker.clock, None
    from aurum.data.store import load_bars
    from aurum.live.paper import BrokerDataFeed, PaperBroker, ReplayFeed

    p = cfg.paper
    costs = CostModel(**cfg.costs)
    if p.get("data") == "mt5":  # paper execution on live MT5 quotes/bars (real time)
        from aurum.live.mt5 import MT5Broker

        data = MT5Broker(cfg.symbol, instrument=instrument, **cfg.mt5)
        wall = SystemClock()
        broker = PaperBroker(BrokerDataFeed(data, cfg.symbol, cfg.timeframe), instrument, costs,
                             float(p.get("initial_equity", 100_000.0)), clock=wall, symbol=cfg.symbol,
                             hedging=bool(p.get("hedging", False)),
                             state_path=Path(cfg.state_dir) / "paper_broker.json")
        return broker, wall, None
    if not p.get("bars_path"):
        raise ValueError("paper broker needs paper.bars_path (a stored bars parquet to replay)")
    bars = load_bars(p["bars_path"])
    if p.get("end"):
        bars = bars.loc[: _as_utc(p["end"])]
    feed = ReplayFeed(bars, cfg.timeframe)
    if p.get("start"):
        start = _as_utc(p["start"])
    else:
        start = feed.decision_time(min(len(bars) - 2, int(p.get("warmup_bars", 2000))))
    clock = SimulatedClock(start + pd.Timedelta(seconds=cfg.bar_close_delay_seconds - 1))
    broker = PaperBroker(feed, instrument, costs, float(p.get("initial_equity", 100_000.0)),
                         clock=clock, symbol=cfg.symbol, hedging=bool(p.get("hedging", False)),
                         state_path=Path(cfg.state_dir) / "paper_broker.json")
    return broker, clock, feed.end + pd.Timedelta(seconds=cfg.bar_close_delay_seconds + 1)


def main(argv: list[str] | None = None) -> int:
    """``python -m aurum.live.runner --config live.yaml [--no-dry-run] [--i-understand-real-money]``."""
    ap = argparse.ArgumentParser(prog="aurum-live", description="Aurum live/paper trading runner (SPEC §11)")
    ap.add_argument("--config", required=True, help="YAML config with a 'live:' section")
    ap.add_argument("--no-dry-run", action="store_true", help="send orders (default: plan and log only)")
    ap.add_argument("--i-understand-real-money", action="store_true",
                    help="second opt-in required (with live.allow_live_real) to trade a non-demo account")
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--until", default=None, help="stop at this UTC time (ISO)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = LiveConfig.from_yaml(args.config)
    if args.no_dry_run:
        cfg.dry_run = False
    broker, clock, replay_end = _build_broker(cfg, XAUUSD)
    runner = LiveRunner(cfg, broker=broker, clock=clock, i_understand_real_money=args.i_understand_real_money)
    try:
        runner.start()
    except RealMoneyGuardError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if runner.real_money:
        print(real_money_banner(broker.account(), dry_run=cfg.dry_run), file=sys.stderr)
    print(f"aurum live runner [{runner.mode}] {cfg.symbol} {cfg.timeframe} magic={cfg.magic} "
          f"state={cfg.state_dir}", file=sys.stderr)
    until = _as_utc(args.until) if args.until else replay_end
    runner.run(until=until, max_cycles=args.max_cycles)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
