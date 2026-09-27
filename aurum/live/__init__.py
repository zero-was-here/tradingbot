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
Importing this package needs neither ``MetaTrader5`` nor ``anthropic``.
"""

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
    psi,
)
from aurum.live.mt5 import MT5Broker, MT5UnavailableError
from aurum.live.oms import ExecutionReport, OrderManager, make_client_id
from aurum.live.paper import BrokerDataFeed, PaperBroker, ReplayFeed
from aurum.live.runner import (
    ArtifactError,
    CycleResult,
    LiveConfig,
    LiveDeskDataProvider,
    LiveRunner,
    RealMoneyGuardError,
    RunnerLockedError,
    TradingArtifact,
    check_real_money_guard,
    load_artifact,
    save_artifact,
)
from aurum.live.state import StateCorruptError

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
    "RunnerLockedError",
    "SimulatedClock",
    "SlippageTracker",
    "StateCorruptError",
    "SystemClock",
    "TradingArtifact",
    "WebhookAlertSink",
    "check_heartbeat",
    "check_real_money_guard",
    "load_artifact",
    "make_client_id",
    "net_lots",
    "psi",
    "save_artifact",
]
