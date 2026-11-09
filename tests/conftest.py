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
* **Nothing is written into the repository's ``data_store/``** (real market data on a
  developer machine): the aurum.data writers (``store.save_frame`` / ``save_bars``,
  ``macro.save_macro_dir`` and macro's own ``save_frame`` binding, ``dukascopy.build_dataset``)
  and the generic parquet writers (``pyarrow.parquet.write_table``, ``DataFrame.to_parquet``)
  raise :class:`DataStoreWriteError` when the resolved target is inside it, before touching
  the disk. They are wrapped in ``pytest_configure``, i.e. before collection, so a test module
  that does ``from aurum.data.store import save_bars`` at import time gets the guarded writer,
  and module-scoped fixtures are covered. A test that tried such a write fails even if the
  error was swallowed. Writes anywhere else (``tmp_path``) are unaffected. Code running in a
  child process (process pools) is not wrapped.
* Headless plotting: ``MPLBACKEND=Agg`` unless the caller chose a backend.

Optional dependencies (torch, gymnasium, stable-baselines3, anthropic, yfinance) are handled
per test with ``pytest.importorskip``; local market data (``data_store/``, ``cache/``, ``runs/``
are gitignored) is never required: tests use ``aurum.data.synthetic`` or skip.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import ipaddress
import os
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
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


REPO_ROOT = Path(__file__).resolve().parents[1]


class DataStoreWriteError(PermissionError):
    """Raised when a test tries to write into the repository's ``data_store/``."""


class DataStoreGuard:
    """Refuses writes whose resolved target is ``root`` or inside it (see the module doc)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.attempts: list[str] = []
        self.error = DataStoreWriteError

    def contains(self, target: Any) -> bool:
        if not isinstance(target, (str, bytes, os.PathLike)):
            return False                                  # file object, buffer, None
        root = os.path.realpath(self.root)
        path = os.path.realpath(os.fsdecode(target))      # relative paths: against the cwd
        return path == root or path.startswith(root.rstrip(os.sep) + os.sep)

    def check(self, target: Any, writer: str) -> None:
        if self.contains(target):
            self.attempts.append(f"{writer} -> {os.fsdecode(target)}")
            raise DataStoreWriteError(f"{writer}: a test tried to write {os.fsdecode(target)!r} inside "
                                      f"{self.root}; write to tmp_path instead (tests/conftest.py)")


_data_store = DataStoreGuard(REPO_ROOT / "data_store")

#: (module, attribute, name of the argument that holds the target path or directory)
DATA_WRITERS = (
    ("aurum.data.store", "save_frame", "path"),
    ("aurum.data.store", "save_bars", "path"),
    ("aurum.data.macro", "save_frame", "path"),          # macro's own binding (save_macro_dir, Yahoo cache)
    ("aurum.data.macro", "save_macro_dir", "path"),
    ("aurum.data.dukascopy", "build_dataset", "out_dir"),
    ("pyarrow.parquet", "write_table", "where"),
)
_restore: list[tuple[Any, str, Any]] = []


def _guarded(writer: str, func: Callable[..., Any], param: str) -> Callable[..., Any]:
    sig = inspect.signature(func)

    @functools.wraps(func)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            target = sig.bind_partial(*args, **kwargs).arguments.get(param)
        except TypeError:
            target = None                                 # let the real function report bad arguments
        _data_store.check(target, writer)
        return func(*args, **kwargs)

    guarded.__aurum_data_store_guard__ = True             # type: ignore[attr-defined]
    return guarded


def _wrap(owner: Any, attr: str, writer: str, param: str) -> None:
    func = getattr(owner, attr)
    if getattr(func, "__aurum_data_store_guard__", False):
        return
    _restore.append((owner, attr, func))
    setattr(owner, attr, _guarded(writer, func, param))


def _install_data_store_guard() -> None:
    import pandas as pd

    import aurum.data

    for module_name, attr, param in DATA_WRITERS:
        _wrap(importlib.import_module(module_name), attr, f"{module_name}.{attr}", param)
    for name in ("save_frame", "save_bars", "save_macro_dir", "build_dataset"):
        if name in vars(aurum.data):                     # lazy re-exports already resolved
            _wrap(aurum.data, name, f"aurum.data.{name}", "out_dir" if name == "build_dataset" else "path")
    _wrap(pd.DataFrame, "to_parquet", "pandas.DataFrame.to_parquet", "path")


def _uninstall_data_store_guard() -> None:
    while _restore:
        owner, attr, func = _restore.pop()
        setattr(owner, attr, func)


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("AURUM_TEST_KEEP_SECRETS") != "1":
        for var in SECRET_ENV_VARS:
            os.environ.pop(var, None)
    socket.socket.connect = _guarded_connect          # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex    # type: ignore[method-assign]
    _install_data_store_guard()


def pytest_unconfigure(config: pytest.Config) -> None:
    socket.socket.connect = _real_connect             # type: ignore[method-assign]
    socket.socket.connect_ex = _real_connect_ex       # type: ignore[method-assign]
    _uninstall_data_store_guard()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[Any]:
    # covers the item's setup too, so module-scoped fixtures built for it are guarded as well
    _net["allow"] = item.get_closest_marker("network") is not None
    _net["attempts"] = []
    _data_store.attempts = []
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


@pytest.fixture(autouse=True)
def _fail_on_data_store_write() -> Iterator[None]:
    yield
    if _data_store.attempts:
        pytest.fail(f"write(s) into data_store/ refused: {_data_store.attempts[:3]}", pytrace=False)


@pytest.fixture
def data_store_guard() -> DataStoreGuard:
    """The suite-wide ``data_store/`` write guard (for the guard's own tests)."""
    return _data_store
