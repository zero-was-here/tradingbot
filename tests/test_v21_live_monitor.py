"""v2.1 watchdog: check_heartbeat can derive a SHORT threshold from the interval the runner's
heartbeat advertises (it is rewritten every poll period while the runner sleeps)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aurum.live.monitor import check_heartbeat, heartbeat_max_age
from aurum.live.state import write_heartbeat

T0 = pd.Timestamp("2026-09-28 10:00", tz="UTC")


def test_heartbeat_max_age_formula_and_validation() -> None:
    assert heartbeat_max_age(30) == 3 * 30 + 120
    assert heartbeat_max_age(60, periods=2, grace_seconds=600) == 720
    for bad in (0, -1, float("nan"), float("inf"), None, "x"):
        with pytest.raises((TypeError, ValueError)):
            heartbeat_max_age(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        heartbeat_max_age(30, periods=0.5)
    with pytest.raises(ValueError):
        heartbeat_max_age(30, grace_seconds=-1)


def test_derived_threshold_from_the_heartbeat(tmp_path: Path) -> None:
    hb = tmp_path / "heartbeat.json"
    write_heartbeat(hb, time=T0, status="running", heartbeat_every_s=30.0, next_wake=T0 + pd.Timedelta(minutes=50))
    # 3 * 30 + 120 = 210 s: fresh at +200 s, stale at +220 s (an H1 runner used to need > 1 h)
    assert check_heartbeat(hb, now=T0 + pd.Timedelta(seconds=200)) is None
    a = check_heartbeat(hb, now=T0 + pd.Timedelta(seconds=220))
    assert a is not None and a.level == "critical" and "status running" in a.message
    # a longer grace for slow decision cycles (LLM desk)
    assert check_heartbeat(hb, now=T0 + pd.Timedelta(seconds=600), grace_seconds=600) is None
    # an explicit threshold still wins
    assert check_heartbeat(hb, max_age_seconds=3600, now=T0 + pd.Timedelta(seconds=1800)) is None


def test_heartbeat_without_interval_alerts_instead_of_crashing(tmp_path: Path) -> None:
    hb = tmp_path / "heartbeat.json"
    write_heartbeat(hb, time=T0, status="running")                      # an older runner's heartbeat
    a = check_heartbeat(hb, now=T0 + pd.Timedelta(seconds=5))
    assert a is not None and a.level == "critical" and "heartbeat_every_s" in a.message
    write_heartbeat(hb, time=T0, status="running", heartbeat_every_s="soon")
    assert "heartbeat_every_s" in check_heartbeat(hb, now=T0).message
    assert check_heartbeat(hb, max_age_seconds=60, now=T0) is None      # explicit threshold: fine
    with pytest.raises(ValueError):                                      # a watchdog SETUP error raises
        check_heartbeat(hb, now=T0, periods=0)
    assert check_heartbeat(tmp_path / "missing.json").message == "heartbeat file missing"
