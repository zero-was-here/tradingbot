"""``python -m aurum ...`` -> :func:`aurum.cli.main`."""

from __future__ import annotations

import sys

from aurum.cli import main

if __name__ == "__main__":
    sys.exit(main())
