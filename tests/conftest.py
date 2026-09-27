"""Suite-wide guards that keep the test suite hermetic on a fresh clone, in CI and on a
developer machine that holds real credentials.

* **Secrets are scrubbed for the whole session** (before any fixture, including module-scoped
  ones, runs): a real ``ANTHROPIC_API_KEY``, MT5 login or alert webhook in the developer's shell
  can never reach a test. Tests that need a value set it with ``monkeypatch.setenv``. Set
  ``AURUM_TEST_KEEP_SECRETS=1`` to opt out (e.g. for a hand-run ``-m network`` check).
* **No internet without ``@pytest.mark.network``** (SPEC §0.6): outbound TCP/UDP connections
  to non-loopback addresses are refused, and a test that tried one fails even if the library
  swallowed the error and fell back (a silent fallback would otherwise hide the call).
  Loopback and Unix sockets (local test servers, process pools) are unaffected.
* Headless plotting: ``MPLBACKEND=Agg`` unless the caller chose a backend.

Optional dependencies (torch, gymnasium, stable-baselines3, anthropic, yfinance) are handled
per test with ``pytest.importorskip``; local market data (``data_store/``, ``cache/``, ``runs/``
are gitignored) is never required: tests use ``aurum.data.synthetic`` or skip.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

#: environment variables that carry credentials or reach real services (aurum + Anthropic SDK)
SECRET_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "MT5_PATH",
    "AURUM_ALERT_WEBHOOK_URL",
    "AURUM_ALERT_TELEGRAM_CHAT_ID",
    "AURUM_ARTIFACT_KEY",
)


class NetworkBlockedError(ConnectionError):
    """Raised for an outbound internet connection from a test without ``@pytest.mark.network``."""


_net: dict[str, Any] = {"allow": True, "attempts": []}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_local(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        return True                                   # AF_UNIX path or exotic family
    host = address[0].decode() if isinstance(address[0], bytes) else str(address[0])
    if host in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False                                  # unresolved hostname: treat as remote


def _check(sock: socket.socket, address: Any) -> None:
    if _net["allow"] or sock.family not in (socket.AF_INET, socket.AF_INET6) or _is_local(address):
        return
    _net["attempts"].append(address)
    raise NetworkBlockedError(f"test tried to connect to {address!r}; mark it @pytest.mark.network")


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _check(self, address)
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _check(self, address)
    return _real_connect_ex(self, address)


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("AURUM_TEST_KEEP_SECRETS") != "1":
        for var in SECRET_ENV_VARS:
            os.environ.pop(var, None)
    socket.socket.connect = _guarded_connect          # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex    # type: ignore[method-assign]


def pytest_unconfigure(config: pytest.Config) -> None:
    socket.socket.connect = _real_connect             # type: ignore[method-assign]
    socket.socket.connect_ex = _real_connect_ex       # type: ignore[method-assign]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[Any]:
    # covers the item's setup too, so module-scoped fixtures built for it are guarded as well
    _net["allow"] = item.get_closest_marker("network") is not None
    _net["attempts"] = []
    try:
        return (yield)
    finally:
        _net["allow"] = True


@pytest.fixture(autouse=True)
def _fail_on_blocked_network() -> Iterator[None]:
    yield
    if _net["attempts"]:
        pytest.fail(f"outbound connection(s) without @pytest.mark.network: {_net['attempts'][:3]}",
                    pytrace=False)
