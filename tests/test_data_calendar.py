"""Tests for aurum.data.calendar (rule-based NFP/FOMC calendar, CSV import) and synthetic events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aurum.data.calendar import (
    EVENT_COLUMNS,
    FOMC_STATEMENTS,
    fomc_statements,
    generate_rule_based_calendar,
    load_calendar_csv,
    merge_calendars,
    nfp_release_date,
    validate_events,
)
from aurum.data.synthetic import make_synthetic_events


def test_nfp_first_friday_dst_aware():
    ev = generate_rule_based_calendar("2024-01-01", "2024-12-31", include=("NFP",))
    validate_events(ev)
    assert list(ev.columns[:6]) == EVENT_COLUMNS
    assert len(ev) == 12
    ny = ev["time"].dt.tz_convert("America/New_York")
    assert (ny.dt.dayofweek == 4).all()
    assert (ny.dt.day <= 7).all()
    assert ((ny.dt.hour == 8) & (ny.dt.minute == 30)).all()
    t = dict(zip(ev["time"].dt.strftime("%Y-%m"), ev["time"], strict=True))
    assert t["2024-01"] == pd.Timestamp("2024-01-05 13:30", tz="UTC")  # EST → 13:30 UTC
    assert t["2024-07"] == pd.Timestamp("2024-07-05 12:30", tz="UTC")  # EDT → 12:30 UTC
    # March 2024: US DST started 10 Mar → first Friday (1 Mar) still EST
    assert t["2024-03"] == pd.Timestamp("2024-03-01 13:30", tz="UTC")
    assert ev["approximate"].all()


def test_nfp_holiday_adjustments_and_bls_rule():
    assert str(nfp_release_date(2021, 1)) == "2021-01-08"   # 1 Jan holiday → next Friday
    assert str(nfp_release_date(2025, 7)) == "2025-07-03"   # 4 Jul Friday → Thursday
    # BLS reference-week rule gets the months the first-Friday rule misses
    assert str(nfp_release_date(2023, 12, "bls")) == "2023-12-08"
    assert str(nfp_release_date(2024, 3, "bls")) == "2024-03-08"
    assert str(nfp_release_date(2024, 2, "bls")) == "2024-02-02"
    with pytest.raises(ValueError):
        nfp_release_date(2024, 1, "whenever")


def test_fomc_verified_statements():
    ev = fomc_statements()
    validate_events(ev)
    assert len(ev) == len(FOMC_STATEMENTS) == 128
    ny = ev["time"].dt.tz_convert("America/New_York")
    # Wednesdays except the documented Thursday (2012-09-13, 2015-09-17, 2018-11-08,
    # 2020-11-05, 2024-11-07) and the one-day Tuesday meeting of 2012-03-13
    assert (ny.dt.dayofweek == 2).sum() == 122
    verified = ev[~ev["approximate"]]
    assert (verified["time"].dt.tz_convert("America/New_York").dt.hour == 14).all()
    assert verified["time"].min() == pd.Timestamp("2016-01-27 19:00", tz="UTC")
    # DST: January statement at 19:00 UTC, June at 18:00 UTC
    t = set(ev["time"])
    assert pd.Timestamp("2024-01-31 19:00", tz="UTC") in t
    assert pd.Timestamp("2024-06-12 18:00", tz="UTC") in t
    # emergency / unscheduled actions are NOT in the calendar (not known in advance)
    days = set(ev["time"].dt.strftime("%Y-%m-%d"))
    assert "2020-03-03" not in days and "2020-03-15" not in days
    assert "2025-09-17" in days and "2026-09-16" in days


def test_rule_based_calendar_range_and_sorting():
    ev = generate_rule_based_calendar("2023-06-01", "2023-08-31")
    assert ev["time"].is_monotonic_increasing
    assert set(ev["name"]) == {"NFP", "FOMC"}
    assert (ev["time"] >= pd.Timestamp("2023-06-01", tz="UTC")).all()
    assert (ev["time"] <= pd.Timestamp("2023-08-31", tz="UTC")).all()
    assert (ev["name"] == "FOMC").sum() == 2  # 14 Jun, 26 Jul
    with pytest.raises(ValueError):
        generate_rule_based_calendar("2023-06-01", "2023-08-31", include=("CPI",))


def test_rule_based_calendar_outside_fomc_coverage_logs(caplog):
    with caplog.at_level("WARNING"):
        ev = generate_rule_based_calendar("2010-01-01", "2010-12-31")
    assert (ev["name"] == "FOMC").sum() == 0
    assert "FOMC" in caplog.text


def test_load_calendar_csv_and_merge(tmp_path: Path):
    p = tmp_path / "cal.csv"
    p.write_text(
        "Time,Name,Currency,Importance,Actual,Forecast\n"
        "2024-01-11 08:30,CPI,usd,high,3.4,3.2\n"
        "2024-01-05 13:30:00+00:00,NFP,USD,3,216,170\n"
        "2024-01-10 10:00,ISM,USD,low,,\n"
    )
    ev = load_calendar_csv(p, tz="America/New_York")
    validate_events(ev)
    assert list(ev["name"]) == ["NFP", "ISM", "CPI"]
    assert ev.loc[ev["name"] == "CPI", "time"].iloc[0] == pd.Timestamp("2024-01-11 13:30", tz="UTC")
    assert ev.loc[ev["name"] == "CPI", "importance"].iloc[0] == 3
    assert ev.loc[ev["name"] == "ISM", "importance"].iloc[0] == 1
    assert ev["currency"].eq("USD").all() and not ev["approximate"].any()
    assert ev.loc[ev["name"] == "NFP", "actual"].iloc[0] == 216
    rule = generate_rule_based_calendar("2024-01-01", "2024-01-31")
    merged = merge_calendars(ev, rule)
    assert (merged["name"] == "NFP").sum() == 1       # duplicate (time, name) removed
    assert merged.loc[merged["name"] == "NFP", "source"].iloc[0].startswith("csv:")  # first wins
    assert isinstance(merged.index, pd.RangeIndex)


def test_synthetic_events_shape_and_weekdays():
    ev = make_synthetic_events("2022-01-01", "2023-12-31")
    validate_events(ev)
    ny = ev["time"].dt.tz_convert("America/New_York")
    fomc = ny[ev["name"] == "FOMC"]
    assert len(fomc) == 16
    assert (fomc.dt.dayofweek == 2).all() and (fomc.dt.hour == 14).all()
    assert fomc.dt.day.between(15, 21).all()
    nfp = ny[ev["name"] == "NFP"]
    assert (nfp.dt.dayofweek == 4).all() and ((nfp.dt.hour == 8) & (nfp.dt.minute == 30)).all()
    cpi = ny[ev["name"] == "CPI"]
    assert (cpi.dt.dayofweek < 5).all()


def test_load_calendar_csv_parses_approximate_flags(tmp_path: Path):
    p = tmp_path / "flags.csv"
    p.write_text(
        "time,name,importance,approximate\n"
        "2024-01-05 13:30,NFP,3,False\n"
        "2024-01-11 13:30,CPI,3,yes\n"
    )
    ev = load_calendar_csv(p)
    assert ev["approximate"].tolist() == [False, True]  # bool("False") would be True
    p.write_text("time,name,approximate\n2024-01-05 13:30,NFP,maybe\n")
    with pytest.raises(ValueError, match="boolean"):
        load_calendar_csv(p)


# ------------------------------------------------------------------------------------ review
@pytest.mark.parametrize("year", [2015, 2020, 2026])
@pytest.mark.parametrize("rule", ["first_friday", "bls"])
def test_nfp_moves_off_observed_independence_day(year: int, rule: str):
    """4 Jul on a Saturday → Friday 3 Jul is the federal holiday; BLS released the June
    report on Thursday 2 Jul (2015, 2020). The old rule put NFP on the holiday, so the risk
    blackout would have guarded a day without a release and missed the real one."""
    d = nfp_release_date(year, 7, rule)
    assert str(d) == f"{year}-07-02" and d.weekday() == 3
    ev = generate_rule_based_calendar(f"{year}-07-01", f"{year}-07-31", include=("NFP",), nfp_rule=rule)
    assert ev["time"].iloc[0] == pd.Timestamp(f"{year}-07-02 12:30", tz="UTC")


def test_bare_date_end_includes_the_whole_day():
    """Regression: end="2024-01-05" meant the instant 00:00 and dropped that day's NFP."""
    ev = generate_rule_based_calendar("2024-01-01", "2024-01-05", include=("NFP",))
    assert ev["time"].tolist() == [pd.Timestamp("2024-01-05 13:30", tz="UTC")]
    ev2 = generate_rule_based_calendar("2024-01-01", pd.Timestamp("2024-01-31", tz="UTC"))
    assert pd.Timestamp("2024-01-31 19:00", tz="UTC") in set(ev2["time"])  # FOMC on the end date
    assert len(fomc_statements("2024-01-31", "2024-01-31")) == 1
    # an explicit time-of-day is an exact (inclusive) bound
    early = generate_rule_based_calendar("2024-01-01", "2024-01-05 13:29", include=("NFP",))
    assert len(early) == 0
    exact = generate_rule_based_calendar("2024-01-01", "2024-01-05 13:30", include=("NFP",))
    assert len(exact) == 1


def test_fomc_table_is_sorted_unique_and_statement_days_are_weekdays():
    days = [d for d, _, _ in FOMC_STATEMENTS]
    assert days == sorted(days) and len(set(days)) == len(days)
    ev = fomc_statements()
    ny = ev["time"].dt.tz_convert("America/New_York")
    assert (ny.dt.dayofweek < 5).all()
    # ~8 scheduled statements per year
    per_year = ny.dt.year.value_counts()
    assert per_year.between(8, 8).all()
