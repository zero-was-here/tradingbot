"""Production execution path (SPEC §11): brokers, order management, live runner, monitoring.

* :mod:`aurum.live.broker`  — ``Broker`` protocol, account/position/order value types, clocks.
* :mod:`aurum.live.paper`   — ``PaperBroker`` (simulator-identical fills) and ``ReplayFeed``.
* :mod:`aurum.live.mt5`     — ``MT5Broker`` (MetaTrader 5; the package is imported lazily).
* :mod:`aurum.live.oms`     — ``OrderManager``: idempotent target-position reconciliation.
* :mod:`aurum.live.runner`  — ``LiveRunner`` bar-close loop, trading artifact, CLI.
* :mod:`aurum.live.monitor` — PSI feature drift, slippage, PnL band, alert sinks.
* :mod:`aurum.live.state`   — atomic JSON state, JSONL logs, heartbeats.

Safe by default (SPEC §0.5): the runner is dry-run unless told otherwise and refuses real
accounts without both ``live.allow_live_real`` and ``--i-understand-real-money``.
Importing this package needs neither ``MetaTrader5`` nor ``anthropic``. The runner's names
(``LiveRunner``, ``LiveConfig``, ``load_artifact``...) are imported LAZILY on first access
(PEP 562), so ``python -m aurum.live.runner`` does not find its own module pre-imported by
the package (which made runpy emit a ``RuntimeWarning``) and ``import aurum.live`` stays light.
"""

import importlib
from typing import TYPE_CHECKING, Any

from aurum.live.broker import (
    AccountInfo,
    Broker,
    BrokerDeal,
    BrokerError,
    BrokerPosition,
    Clock,
    OrderRequest,
    OrderResult,
    OrderStatusCode,
    Quote,
    SimulatedClock,
    SystemClock,
    net_lots,
)
from aurum.live.monitor import (
    Alert,
    AlertManager,
    DriftMonitor,
    FeatureReference,
    JsonlAlertSink,
    LiveMonitor,
    LogAlertSink,
    PnLBand,
    SlippageTracker,
    WebhookAlertSink,
    check_heartbeat,
    heartbeat_max_age,
    psi,
)
from aurum.live.mt5 import MT5Broker, MT5UnavailableError
from aurum.live.oms import ExecutionReport, OrderManager, make_client_id
from aurum.live.paper import BrokerDataFeed, PaperBroker, ReplayFeed
from aurum.live.state import StateCorruptError

if TYPE_CHECKING:  # pragma: no cover - static analysis only; resolved lazily at runtime
    from aurum.live.runner import (
        ArtifactError,
        CycleResult,
        LiveConfig,
        LiveDeskDataProvider,
        LiveRunner,
        RealMoneyGuardError,
        RunnerConfigError,
        RunnerLockedError,
        TradingArtifact,
        check_desk_prerequisites,
        check_real_money_guard,
        load_artifact,
        save_artifact,
    )

#: names served from :mod:`aurum.live.runner` on first attribute access
_RUNNER_EXPORTS = frozenset({
    "ArtifactError", "CycleResult", "LiveConfig", "LiveDeskDataProvider", "LiveRunner",
    "RealMoneyGuardError", "RunnerConfigError", "RunnerLockedError", "TradingArtifact",
    "check_desk_prerequisites", "check_real_money_guard", "load_artifact", "save_artifact",
})


def __getattr__(name: str) -> Any:
    """PEP 562: import :mod:`aurum.live.runner` only when one of its names is first used."""
    if name in _RUNNER_EXPORTS or name == "runner":
        module = importlib.import_module("aurum.live.runner")
        value = module if name == "runner" else getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RUNNER_EXPORTS)


__all__ = [
    "AccountInfo",
    "Alert",
    "AlertManager",
    "ArtifactError",
    "Broker",
    "BrokerDataFeed",
    "BrokerDeal",
    "BrokerError",
    "BrokerPosition",
    "Clock",
    "CycleResult",
    "DriftMonitor",
    "ExecutionReport",
    "FeatureReference",
    "JsonlAlertSink",
    "LiveConfig",
    "LiveDeskDataProvider",
    "LiveMonitor",
    "LiveRunner",
    "LogAlertSink",
    "MT5Broker",
    "MT5UnavailableError",
    "OrderManager",
    "OrderRequest",
    "OrderResult",
    "OrderStatusCode",
    "PaperBroker",
    "PnLBand",
    "Quote",
    "RealMoneyGuardError",
    "ReplayFeed",
    "RunnerConfigError",
    "RunnerLockedError",
    "SimulatedClock",
    "SlippageTracker",
    "StateCorruptError",
    "SystemClock",
    "TradingArtifact",
    "WebhookAlertSink",
    "check_desk_prerequisites",
    "check_heartbeat",
    "check_real_money_guard",
    "heartbeat_max_age",
    "load_artifact",
    "make_client_id",
    "net_lots",
    "psi",
    "save_artifact",
]
