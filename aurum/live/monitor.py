"""Live monitoring: feature drift, execution quality, PnL vs expectation, alerts (SPEC §11).

What is watched and why
-----------------------
* **Feature drift (PSI).** Models and scalers were fitted on a training window; if the live
  feature distribution moves away from it, forecasts silently degrade. The Population
  Stability Index ``PSI = sum_i (a_i - e_i) ln(a_i / e_i)`` over reference bins (training
  quantiles) is the standard credit-scoring drift statistic (Siddiqi, 2006, *Credit Risk
  Scorecards*; Yurdakul, 2018, "Statistical properties of the population stability index").
  Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
  The reference is stored in the trading artifact (:class:`FeatureReference.from_frame` on
  the TRANSFORMED training features). Without one, :meth:`FeatureReference.from_pipeline`
  builds a Gaussian approximation from the pipeline's fitted statistics — robust/standard
  scaling maps training data to roughly N(0, 1) — which is cruder (skewed features show
  spurious PSI) and is labelled as such.
* **Slippage.** Every fill's implementation shortfall versus the decision mid
  (``side * (fill_price - mid)`` in USD/oz) is compared with what the backtest cost model
  charged for the same order (``effective_spread/2 + slippage``). Live costs persistently
  above the model mean the backtest overstated net returns (Perold, 1988, "The
  implementation shortfall: paper versus reality", *J. Portfolio Management*).
* **PnL vs backtest band.** Under the backtest, daily log returns are ~iid(mu, sigma); after
  ``n`` live days ``z = (sum r - n mu) / (sigma sqrt n)``. A strongly negative ``z`` flags
  that live performance is inconsistent with the research expectation (a one-sided test;
  the default -3 has a ~0.1% false-alarm rate per check under normality, conservative
  because the same z is checked daily).
* **Heartbeat.** The runner rewrites a heartbeat file every loop; :func:`check_heartbeat`
  is for an external watchdog (cron, systemd timer) that alerts when it goes stale.

Alerts go to sinks: the log (always), an optional JSONL file, and an optional webhook
(generic JSON POST accepted by Slack/Discord incoming webhooks and similar; Telegram bot
API supported). The webhook URL comes from the environment and is never logged; payloads are
scrubbed of anything that looks like a credential.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import stats as _st

from aurum.core.instrument import XAUUSD, Instrument
from aurum.core.types import Fill
from aurum.execution.costs import CostModel
from aurum.live.state import append_jsonl, atomic_write_json, read_json, to_jsonable

logger = logging.getLogger(__name__)

__all__ = [
    "PSI_ALERT",
    "PSI_WARN",
    "Alert",
    "AlertManager",
    "AlertSink",
    "DriftMonitor",
    "DriftReport",
    "FeatureReference",
    "JsonlAlertSink",
    "LiveMonitor",
    "LogAlertSink",
    "PnLBand",
    "SlippageTracker",
    "WebhookAlertSink",
    "check_heartbeat",
    "psi",
    "redact",
]

PSI_WARN = 0.10
PSI_ALERT = 0.25
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}
_SECRET_KEY = re.compile(r"pass(word|wd)?|secret|token|api[_-]?key|authori[sz]ation|credential|webhook|login",
                         re.IGNORECASE)
_SECRET_ENV = ("MT5_PASSWORD", "MT5_LOGIN", "ANTHROPIC_API_KEY", "AURUM_ALERT_WEBHOOK_URL",
               "AURUM_ARTIFACT_KEY", "AURUM_ALERT_TELEGRAM_CHAT_ID")


# =============================================================================================
# PSI
# =============================================================================================
def psi(expected: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray, *, eps: float = 1e-4) -> float:
    """Population Stability Index between two binned distributions (proportions or counts).

    Empty bins are floored at ``eps`` (otherwise ``ln(0)``); both vectors are renormalised.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.ndim != 1 or e.size == 0:
        raise ValueError("expected and actual must be 1-D arrays of the same length")
    if e.sum() <= 0 or a.sum() <= 0:
        return math.nan
    e = np.maximum(e / e.sum(), eps)
    a = np.maximum(a / a.sum(), eps)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _bin_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right")
    return np.bincount(idx, minlength=len(edges) + 1).astype(float)


@dataclass
class FeatureReference:
    """Per-feature reference histogram: interior bin ``edges`` and ``expected`` proportions."""

    edges: dict[str, list[float]]
    expected: dict[str, list[float]]
    n_obs: int = 0
    source: str = "train"

    @property
    def columns(self) -> list[str]:
        return list(self.edges)

    @classmethod
    def from_frame(cls, features: pd.DataFrame, *, n_bins: int = 10, min_obs: int = 50) -> FeatureReference:
        """Quantile bins of the (transformed) TRAINING features. Columns with few distinct
        values (flags, clipped states) get one bin per value."""
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        edges: dict[str, list[float]] = {}
        expected: dict[str, list[float]] = {}
        for c in features.columns:
            x = features[c].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size < min_obs:
                continue
            uniq = np.unique(x)
            if uniq.size < 2:
                continue
            if uniq.size <= n_bins:
                e = (uniq[:-1] + uniq[1:]) / 2.0
            else:
                e = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
            counts = _bin_counts(x, e)
            edges[str(c)] = [float(v) for v in e]
            expected[str(c)] = [float(v) for v in counts / counts.sum()]
        return cls(edges=edges, expected=expected, n_obs=len(features), source="train")

    @classmethod
    def from_pipeline(cls, pipeline: Any, *, n_bins: int = 10) -> FeatureReference:
        """Gaussian approximation from a fitted :class:`~aurum.features.pipeline.FeaturePipeline`:
        robust/standard-scaled training features are ~N(0,1), so equal-probability normal bins
        are the reference. Discrete/unscaled columns are skipped (no distribution stored)."""
        st = pipeline.stats
        qs = _st.norm.ppf(np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        edges, expected = {}, {}
        for col, kind in zip(st.index, st["kind"], strict=True):
            if kind in ("robust", "robust_std", "standard"):
                edges[str(col)] = [float(v) for v in qs]
                expected[str(col)] = [1.0 / n_bins] * n_bins
        return cls(edges=edges, expected=expected, n_obs=0, source="gaussian_approx")

    def to_dict(self) -> dict[str, Any]:
        return {"format": "aurum.live.FeatureReference", "version": 1, "source": self.source,
                "n_obs": self.n_obs, "edges": self.edges, "expected": self.expected}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FeatureReference:
        if d.get("format") != "aurum.live.FeatureReference":
            raise ValueError("not a FeatureReference payload")
        return cls(edges={k: list(v) for k, v in d["edges"].items()},
                   expected={k: list(v) for k, v in d["expected"].items()},
                   n_obs=int(d.get("n_obs", 0)), source=str(d.get("source", "train")))

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.to_dict(), indent=None, convert=False)  # exact floats

    @classmethod
    def load(cls, path: str | Path) -> FeatureReference:
        return cls.from_dict(read_json(path))


@dataclass
class DriftReport:
    time: pd.Timestamp | None
    psi: dict[str, float]
    n_obs: int
    warn: list[str]
    alert: list[str]
    missing: list[str]
    source: str

    @property
    def max_psi(self) -> float:
        vals = [v for v in self.psi.values() if math.isfinite(v)]
        return max(vals) if vals else math.nan

    def top(self, k: int = 5) -> dict[str, float]:
        items = sorted(((c, v) for c, v in self.psi.items() if math.isfinite(v)), key=lambda t: -t[1])
        return dict(items[:k])

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "n_obs": self.n_obs, "max_psi": self.max_psi, "top": self.top(),
                "n_warn": len(self.warn), "n_alert": len(self.alert), "alert": self.alert[:20],
                "missing": self.missing[:20], "source": self.source}


class DriftMonitor:
    """PSI of recent live features against a :class:`FeatureReference`."""

    def __init__(self, reference: FeatureReference, *, warn: float = PSI_WARN, alert: float = PSI_ALERT,
                 min_obs: int = 100) -> None:
        if not 0 < warn <= alert:
            raise ValueError("need 0 < warn <= alert")
        self.reference = reference
        self.warn = float(warn)
        self.alert = float(alert)
        self.min_obs = int(min_obs)

    def evaluate(self, features: pd.DataFrame, *, time: pd.Timestamp | None = None) -> DriftReport | None:
        """PSI per referenced column over the rows of ``features`` (None if too few rows)."""
        if len(features) < self.min_obs:
            return None
        out: dict[str, float] = {}
        missing: list[str] = []
        for c in self.reference.columns:
            if c not in features.columns:
                missing.append(c)
                continue
            x = features[c].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size < self.min_obs:
                continue
            e = np.asarray(self.reference.edges[c], dtype=float)
            out[c] = psi(self.reference.expected[c], _bin_counts(x, e))
        warn = sorted(c for c, v in out.items() if self.warn <= v < self.alert)
        alert = sorted(c for c, v in out.items() if v >= self.alert)
        return DriftReport(time=time, psi=out, n_obs=len(features), warn=warn, alert=alert, missing=missing,
                           source=self.reference.source)


# =============================================================================================
# execution quality
# =============================================================================================
@dataclass
class SlippageRecord:
    time: pd.Timestamp
    client_id: str
    side: int
    lots: float
    fill_price: float
    reference_mid: float
    realized_per_oz: float
    expected_per_oz: float

    @property
    def excess_per_oz(self) -> float:
        return self.realized_per_oz - self.expected_per_oz


class SlippageTracker:
    """Live implementation shortfall vs the backtest cost model (USD per ounce)."""

    def __init__(self, costs: CostModel | None = None, instrument: Instrument = XAUUSD, *, window: int = 50,
                 alert_ratio: float = 2.0, abs_tolerance: float = 0.05, min_fills: int = 10) -> None:
        self.costs = costs if costs is not None else CostModel()
        self.instrument = instrument
        self.records: deque[SlippageRecord] = deque(maxlen=int(window))
        self.alert_ratio = float(alert_ratio)
        self.abs_tolerance = float(abs_tolerance)
        self.min_fills = int(min_fills)
        self.n_total = 0

    def expected_per_oz(self, spread: float, bar_range: float, lots: float) -> float:
        return 0.5 * self.costs.effective_spread(spread) + self.costs.slippage(bar_range, lots)

    def record(self, fill: Fill, *, reference_mid: float, spread: float, bar_range: float = 0.0) -> SlippageRecord:
        side = int(fill.side)
        rec = SlippageRecord(time=pd.Timestamp(fill.time), client_id=fill.client_id, side=side, lots=float(fill.lots),
                             fill_price=float(fill.price), reference_mid=float(reference_mid),
                             realized_per_oz=side * (float(fill.price) - float(reference_mid)),
                             expected_per_oz=self.expected_per_oz(spread, bar_range, fill.lots))
        self.records.append(rec)
        self.n_total += 1
        return rec

    def summary(self) -> dict[str, Any]:
        n = len(self.records)
        if n == 0:
            return {"n": 0, "n_total": self.n_total}
        real = np.array([r.realized_per_oz for r in self.records])
        exp = np.array([r.expected_per_oz for r in self.records])
        lots = np.array([r.lots for r in self.records])
        cs = self.instrument.contract_size
        return {"n": n, "n_total": self.n_total, "mean_realized_per_oz": float(real.mean()),
                "mean_expected_per_oz": float(exp.mean()),
                "ratio": float(real.mean() / exp.mean()) if exp.mean() > 0 else math.nan,
                "excess_usd": float(np.sum((real - exp) * lots * cs))}

    def check(self, time: pd.Timestamp | None = None) -> list[Alert]:
        s = self.summary()
        if s["n"] < self.min_fills:
            return []
        if s["mean_realized_per_oz"] > self.alert_ratio * s["mean_expected_per_oz"] + self.abs_tolerance:
            return [Alert("warning", "slippage",
                          f"live execution cost {s['mean_realized_per_oz']:.3f} USD/oz vs model "
                          f"{s['mean_expected_per_oz']:.3f} over the last {s['n']} fills", time=time, data=s)]
        return []


# =============================================================================================
# PnL vs backtest expectation
# =============================================================================================
class PnLBand:
    """Cumulative live log return vs the backtest's daily return distribution.

    Days are UTC trading dates with Saturday/Sunday folded into the following Monday — the
    convention of :func:`aurum.backtest.metrics.daily_returns` that produced ``daily_mean`` /
    ``daily_std``. Counting gold's Sunday-evening session as a day of its own would inflate
    ``n`` by ~20% (six "days" per week) and bias ``z`` downward (spurious alerts).
    """

    def __init__(self, daily_mean: float, daily_std: float, *, z_warn: float = -2.0, z_alert: float = -3.0,
                 min_days: int = 5, fold_weekends: bool = True) -> None:
        if not (math.isfinite(daily_std) and daily_std > 0):
            raise ValueError("daily_std must be positive")
        if not z_alert <= z_warn < 0:
            raise ValueError("need z_alert <= z_warn < 0")
        self.mu = float(daily_mean)
        self.sigma = float(daily_std)
        self.z_warn = float(z_warn)
        self.z_alert = float(z_alert)
        self.min_days = int(min_days)
        self.fold_weekends = bool(fold_weekends)
        self.start_equity: float | None = None
        self.day_close: dict[str, float] = {}

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any] | None, **kw: Any) -> PnLBand | None:
        """From artifact backtest stats: ``daily_mean``/``daily_std`` (daily LOG returns), or
        annualised ``sharpe`` + ``ann_vol`` (252 days; top level or under ``"combined"``, the
        layout of a walk-forward summary — use OUT-OF-SAMPLE stats, in-sample ones bias the band).

        With ``SR = mean_d / sd_d * sqrt(252)`` and ``sd_d = ann_vol / sqrt(252)`` the daily
        simple-return mean is ``SR * ann_vol / 252``; the log-return mean subtracts ``sd_d**2 / 2``.
        """
        if not stats:
            return None
        mu, sd = stats.get("daily_mean"), stats.get("daily_std")
        if mu is None or sd is None:
            src: Mapping[str, Any] = stats
            if stats.get("sharpe") is None or stats.get("ann_vol") is None:
                nested = stats.get("combined")
                src = nested if isinstance(nested, Mapping) else {}
            sr, vol = src.get("sharpe"), src.get("ann_vol")
            try:
                sr, vol = float(sr), float(vol)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            if not (math.isfinite(sr) and math.isfinite(vol)):
                return None
            sd = vol / math.sqrt(252.0)
            mu = sr * vol / 252.0 - 0.5 * sd * sd
        try:
            return cls(float(mu), float(sd), **kw)
        except ValueError:
            return None

    def update(self, time: pd.Timestamp, equity: float) -> None:
        if not (math.isfinite(equity) and equity > 0):
            return
        if self.start_equity is None:
            self.start_equity = float(equity)
        day = pd.Timestamp(time).tz_convert("UTC").normalize()
        if self.fold_weekends and day.weekday() >= 5:
            day = day + pd.Timedelta(days=7 - day.weekday())
        self.day_close[f"{day:%Y-%m-%d}"] = float(equity)

    def status(self) -> dict[str, Any]:
        n = max(0, len(self.day_close) - 1) if self.start_equity is not None else 0
        if self.start_equity is None or not self.day_close:
            return {"n_days": 0}
        last = self.day_close[max(self.day_close)]
        cum = math.log(last / self.start_equity)
        if n == 0:
            return {"n_days": 0, "cum_log_return": cum}
        exp = n * self.mu
        sd = self.sigma * math.sqrt(n)
        return {"n_days": n, "cum_log_return": cum, "expected": exp, "z": (cum - exp) / sd,
                "band_lo": exp + self.z_alert * sd, "band_hi": exp - self.z_alert * sd}

    def check(self, time: pd.Timestamp | None = None) -> list[Alert]:
        s = self.status()
        if s.get("n_days", 0) < self.min_days or "z" not in s:
            return []
        z = s["z"]
        if z <= self.z_alert:
            return [Alert("critical", "pnl_band", f"live PnL z={z:.2f} below the backtest band after "
                          f"{s['n_days']} days (cum {s['cum_log_return']:+.2%} vs expected {s['expected']:+.2%})",
                          time=time, data=s)]
        if z <= self.z_warn:
            return [Alert("warning", "pnl_band", f"live PnL z={z:.2f} after {s['n_days']} days", time=time, data=s)]
        return []

    def to_state(self) -> dict[str, Any]:
        return {"start_equity": self.start_equity, "day_close": dict(self.day_close)}

    def load_state(self, d: Mapping[str, Any] | None) -> None:
        if not d:
            return
        self.start_equity = d.get("start_equity")
        self.day_close = {str(k): float(v) for k, v in (d.get("day_close") or {}).items()}


# =============================================================================================
# alerts
# =============================================================================================
@dataclass
class Alert:
    level: str
    kind: str
    message: str
    time: pd.Timestamp | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in _LEVELS:
            raise ValueError(f"alert level must be one of {sorted(_LEVELS)}")

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "kind": self.kind, "message": self.message, "time": self.time,
                "data": self.data}


def redact(obj: Any, *, extra_secrets: Iterable[str] = ()) -> Any:
    """Drop credential-looking keys and scrub known secret values (env) from strings."""
    secrets = [v for v in (os.environ.get(k) for k in _SECRET_ENV) if v and len(v) >= 4]
    secrets += [s for s in extra_secrets if s and len(s) >= 4]

    def scrub(x: Any) -> Any:
        if isinstance(x, Mapping):
            return {k: ("***" if _SECRET_KEY.search(str(k)) else scrub(v)) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [scrub(v) for v in x]
        if isinstance(x, str):
            for s in secrets:
                x = x.replace(s, "***")
            return x
        return x

    return scrub(to_jsonable(obj))


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class LogAlertSink:
    """Alerts to the Python log (level mapped)."""

    def send(self, alert: Alert) -> None:
        logger.log(_LEVELS[alert.level], "ALERT [%s] %s", alert.kind, alert.message)


class JsonlAlertSink:
    """Alerts appended to a JSONL file (redacted)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def send(self, alert: Alert) -> None:
        append_jsonl(self.path, redact(alert.to_dict()))


class WebhookAlertSink:
    """POST alerts as JSON to a webhook (Slack/Discord incoming webhooks, generic receivers,
    or the Telegram bot API). The URL is read from ``env_var`` unless given and is never
    logged. Delivery failures are logged (without the URL) and swallowed.

    ``style``: ``"generic"`` sends ``{"text", "content", "level", "kind", "time", "data"}``
    (Slack reads ``text``, Discord ``content``); ``"telegram"`` sends
    ``{"chat_id", "text"}`` with ``chat_id`` from ``AURUM_ALERT_TELEGRAM_CHAT_ID``.
    """

    def __init__(self, url: str | None = None, *, env_var: str = "AURUM_ALERT_WEBHOOK_URL",
                 style: str = "generic", min_level: str = "warning", timeout: float = 5.0,
                 post: Callable[[str, bytes, float], None] | None = None, prefix: str = "aurum") -> None:
        if style not in ("generic", "telegram"):
            raise ValueError("style must be 'generic' or 'telegram'")
        if min_level not in _LEVELS:
            raise ValueError(f"min_level must be one of {sorted(_LEVELS)}")
        self._url = url if url is not None else os.environ.get(env_var)
        self.style = style
        self.min_level = min_level
        self.timeout = float(timeout)
        self._post = post or self._urllib_post
        self.prefix = prefix
        self.n_sent = 0
        self.n_failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    @staticmethod
    def _urllib_post(url: str, body: bytes, timeout: float) -> None:
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "aurum-live"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured URL
            resp.read(256)

    def payload(self, alert: Alert) -> dict[str, Any]:
        text = f"[{self.prefix}] {alert.level.upper()} {alert.kind}: {alert.message}"
        safe = redact({"text": text, "level": alert.level, "kind": alert.kind, "time": alert.time,
                       "data": alert.data}, extra_secrets=[self._url or ""])
        if self.style == "telegram":
            return {"chat_id": os.environ.get("AURUM_ALERT_TELEGRAM_CHAT_ID", ""), "text": safe["text"]}
        return {**safe, "content": safe["text"]}

    def send(self, alert: Alert) -> None:
        if not self.enabled or _LEVELS[alert.level] < _LEVELS[self.min_level]:
            return
        body = json.dumps(self.payload(alert), allow_nan=False).encode("utf-8")
        try:
            self._post(self._url, body, self.timeout)  # type: ignore[arg-type]
            self.n_sent += 1
        except Exception as exc:  # never let alerting break trading
            self.n_failed += 1
            msg = str(exc).replace(self._url or "\0", "<webhook>")
            logger.warning("webhook alert delivery failed: %s: %s", type(exc).__name__, msg[:200])


class AlertManager:
    """Fan-out to sinks with de-duplication: the same (kind, level) is re-sent at most once
    per ``cooldown_seconds`` of alert time (flapping checks do not spam the channel)."""

    def __init__(self, sinks: Sequence[AlertSink] | None = None, *, cooldown_seconds: float = 900.0) -> None:
        self.sinks: list[AlertSink] = list(sinks) if sinks is not None else [LogAlertSink()]
        self.cooldown = pd.Timedelta(seconds=float(cooldown_seconds))
        self._last: dict[tuple[str, str], pd.Timestamp] = {}
        self.history: deque[Alert] = deque(maxlen=500)

    def send(self, alert: Alert) -> bool:
        t = alert.time if alert.time is not None else pd.Timestamp.now(tz="UTC")
        alert.time = t
        key = (alert.kind, alert.level)
        prev = self._last.get(key)
        if prev is not None and t - prev < self.cooldown:
            return False
        self._last[key] = t
        self.history.append(alert)
        for sink in self.sinks:
            try:
                sink.send(alert)
            except Exception:  # pragma: no cover - a broken sink must not stop the others
                logger.exception("alert sink %r failed", sink)
        return True

    def alert(self, level: str, kind: str, message: str, *, time: pd.Timestamp | None = None,
              **data: Any) -> bool:
        return self.send(Alert(level, kind, message, time=time, data=data))


def check_heartbeat(path: str | Path, *, max_age_seconds: float, now: pd.Timestamp | None = None) -> Alert | None:
    """Watchdog helper: an alert if the heartbeat file is missing, unreadable or stale."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    try:
        hb = read_json(path)
    except Exception as exc:
        return Alert("critical", "heartbeat", f"heartbeat unreadable: {type(exc).__name__}", time=now)
    if hb is None:
        return Alert("critical", "heartbeat", "heartbeat file missing", time=now)
    try:
        t = pd.Timestamp(hb["time"])
        if t is pd.NaT:
            raise ValueError("NaT")
        t = t.tz_localize("UTC") if t.tz is None else t  # a watchdog must not crash on a naive stamp
    except (KeyError, ValueError, TypeError):
        return Alert("critical", "heartbeat", "heartbeat has no valid time", time=now)
    age = (now - t).total_seconds()
    if age > max_age_seconds:
        return Alert("critical", "heartbeat", f"runner heartbeat is {age:.0f}s old (status {hb.get('status')})",
                     time=now, data={"age_seconds": age, "status": hb.get("status")})
    return None


# =============================================================================================
# façade used by the runner
# =============================================================================================
class LiveMonitor:
    """Bundles drift, slippage and PnL-band checks and routes their alerts.

    ``drift_every``: run the (comparatively expensive) PSI check every N decisions on the
    last ``drift_window`` feature rows.
    """

    def __init__(self, *, alerts: AlertManager | None = None, drift: DriftMonitor | None = None,
                 slippage: SlippageTracker | None = None, pnl: PnLBand | None = None, drift_every: int = 24,
                 drift_window: int = 500) -> None:
        self.alerts = alerts or AlertManager()
        self.drift = drift
        self.slippage = slippage
        self.pnl = pnl
        self.drift_every = max(1, int(drift_every))
        self.drift_window = int(drift_window)
        self.n_decisions = 0
        self.last_drift: DriftReport | None = None

    def on_decision(self, time: pd.Timestamp, *, equity: float | None = None,
                    features: pd.DataFrame | None = None) -> list[Alert]:
        self.n_decisions += 1
        out: list[Alert] = []
        if self.pnl is not None and equity is not None:
            self.pnl.update(time, equity)
            out += self.pnl.check(time)
        if self.drift is not None and features is not None and self.n_decisions % self.drift_every == 0:
            rep = self.drift.evaluate(features.iloc[-self.drift_window:], time=time)
            if rep is not None:
                self.last_drift = rep
                if rep.alert:
                    out.append(Alert("warning", "feature_drift",
                                     f"{len(rep.alert)} feature(s) with PSI >= {self.drift.alert:.2f} "
                                     f"(max {rep.max_psi:.2f}; {rep.source} reference)", time=time,
                                     data=rep.to_dict()))
                if rep.missing:
                    out.append(Alert("warning", "feature_missing",
                                     f"{len(rep.missing)} reference feature(s) missing live", time=time,
                                     data={"missing": rep.missing[:20]}))
        for a in out:
            self.alerts.send(a)
        return out

    def on_fill(self, fill: Fill, *, reference_mid: float, spread: float, bar_range: float = 0.0,
                time: pd.Timestamp | None = None) -> list[Alert]:
        if self.slippage is None:
            return []
        self.slippage.record(fill, reference_mid=reference_mid, spread=spread, bar_range=bar_range)
        out = self.slippage.check(time)
        for a in out:
            self.alerts.send(a)
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_decisions": self.n_decisions,
            "drift": self.last_drift.to_dict() if self.last_drift is not None else None,
            "slippage": self.slippage.summary() if self.slippage is not None else None,
            "pnl": self.pnl.status() if self.pnl is not None else None,
        }
