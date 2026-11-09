"""Visualisation: one shared chart style and an interactive walk-forward dashboard.

* :mod:`aurum.viz.style` - the palette (light AND dark), matplotlib ``rcParams``,
  :func:`~aurum.viz.style.use_style`, :func:`~aurum.viz.style.save_figure` (renders one
  plotting function to ``<stem>-light.png`` / ``<stem>-dark.png``) and the matching Plotly
  template. Depends on matplotlib only.
* :mod:`aurum.viz.dashboard` - :func:`~aurum.viz.dashboard.build_dashboard`: a
  self-contained interactive HTML page for a walk-forward run directory (needs the optional
  ``plotly`` dependency: ``pip install aurum[viz]``).
* :mod:`aurum.viz.cli` - ``aurum dashboard --run DIR [--journal DIR] [--out FILE] [--cdn]``.

Names are re-exported lazily (PEP 562), so ``import aurum.viz`` never imports matplotlib or
plotly by itself.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "use_style": "aurum.viz.style",
    "palette": "aurum.viz.style",
    "get_palette": "aurum.viz.style",
    "save_figure": "aurum.viz.style",
    "figure": "aurum.viz.style",
    "categorical": "aurum.viz.style",
    "sequential": "aurum.viz.style",
    "diverging": "aurum.viz.style",
    "plotly_template": "aurum.viz.style",
    "build_dashboard": "aurum.viz.dashboard",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
