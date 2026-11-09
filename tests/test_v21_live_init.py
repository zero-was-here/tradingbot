"""v2.1: ``aurum.live`` imports the runner lazily, so ``python -m aurum.live.runner`` runs without
runpy's "found in sys.modules ... prior to execution" RuntimeWarning (fresh interpreters)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _py(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT, timeout=120)


def test_module_entry_point_has_no_runtime_warning() -> None:
    out = _py("-W", "error::RuntimeWarning", "-m", "aurum.live.runner", "--help")
    assert out.returncode == 0, out.stderr
    assert "RuntimeWarning" not in out.stderr and "usage: aurum-live" in out.stdout


def test_package_import_is_lazy_and_names_still_resolve() -> None:
    code = (
        "import sys, aurum.live as L\n"
        "assert 'aurum.live.runner' not in sys.modules, 'runner imported eagerly'\n"
        "assert 'LiveRunner' in dir(L) and 'RunnerConfigError' in L.__all__\n"
        "from aurum.live import LiveRunner, LiveConfig, RunnerConfigError, check_desk_prerequisites\n"
        "import aurum.live.runner as R\n"
        "assert LiveRunner is R.LiveRunner and LiveConfig is R.LiveConfig and L.runner is R\n"
        "assert issubclass(RunnerConfigError, ValueError) and check_desk_prerequisites is R.check_desk_prerequisites\n"
        "from aurum.live import *  # noqa\n"
        "assert load_artifact is R.load_artifact\n"
        "try:\n"
        "    L.no_such_name\n"
        "except AttributeError as exc:\n"
        "    assert 'no_such_name' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('missing attribute did not raise')\n"
        "print('ok')\n"
    )
    out = _py("-W", "error::RuntimeWarning", "-c", code)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_in_process_attribute_access() -> None:
    import aurum.live as live
    import aurum.live.runner as runner

    for name in ("LiveRunner", "LiveConfig", "TradingArtifact", "load_artifact", "save_artifact", "ArtifactError",
                 "RealMoneyGuardError", "RunnerLockedError", "CycleResult", "LiveDeskDataProvider",
                 "check_real_money_guard", "RunnerConfigError", "check_desk_prerequisites"):
        assert getattr(live, name) is getattr(runner, name)
        assert name in live.__all__
