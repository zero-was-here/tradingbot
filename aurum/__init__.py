"""Aurum v2: point-in-time systematic research and (paper) trading for spot gold (XAUUSD).

Importing the package is cheap: it only resolves :data:`__version__`. The subpackages
(``aurum.data``, ``aurum.features``, ``aurum.strategies``, ``aurum.research``,
``aurum.live``, ...) are imported on demand; the CLI entry point is :func:`aurum.cli.main`.
"""

from __future__ import annotations

__all__ = ["__version__"]


def _resolve_version() -> str:
    """The installed distribution's version (``importlib.metadata``); in a source checkout
    that was never installed, the ``[project] version`` of the adjacent ``pyproject.toml``;
    otherwise ``"0+unknown"``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("aurum")
    except PackageNotFoundError:
        pass
    import re
    from pathlib import Path

    try:
        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "0+unknown"
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    found = re.search(r'^version\s*=\s*"([^"]+)"', project.group(1), re.M) if project else None
    return found.group(1) if found else "0+unknown"


__version__: str = _resolve_version()
