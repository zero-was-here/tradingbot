"""Economic-event calendar (SPEC §3.5).

Event frame format (``RangeIndex``; one row per scheduled release):

  time        tz-aware UTC scheduled release time
  name        e.g. "NFP", "CPI", "FOMC"
  currency    e.g. "USD"
  importance  1..3 (3 = market-moving for gold)
  source      provenance string
  approximate bool — True when the *time or date* is inferred by rule / not verified
  [actual, forecast, previous]  optional outcome columns (CSV imports only)

Point-in-time rules. Scheduled release times are published well in advance, so using the
*future schedule* (e.g. "hours until the next FOMC statement") is not look-ahead. Event
*outcomes* (actual, surprise) may only be used from ``time`` onward. Unscheduled events
(emergency FOMC cuts) are NOT known in advance and are therefore excluded from the
rule-based calendar — including them would leak.

FOMC statement dates
--------------------
Hard-coded from the Federal Reserve Board's calendars, retrieved 2026-09-26:

* https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (2021-2027)
* https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm (2012-2020)

The statement is released on the last day of each scheduled meeting. Release times were
verified from each statement's press release ("For release at 2:00 p.m. EST/EDT") for
2016-01-27 .. 2026-09-16 (``approximate=False``). Press releases before 2016 only say "For
immediate release", so for 2012-2015 the time follows the Fed's documented practice
(2012: 12:30 ET on press-conference meetings, 14:15 ET otherwise; 2013-2015: 14:00 ET) and
is flagged ``approximate=True``. Future meetings (2026-10-28 onward) come from the official
schedule, assume 14:00 ET and are flagged ``approximate=True``. The 2020-03-17/18 meeting was
*scheduled* (hence known in advance) but superseded by the unscheduled 2020-03-15 action;
it is kept (``approximate=True``) because dropping it would use hindsight. Unscheduled
actions (2013-10-16, 2014-03-04, 2019-10-04, 2020-03-03, 2020-03-15) and notation votes are
excluded.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aurum.data.loaders import parse_timestamps

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"
EVENT_COLUMNS = ["time", "name", "currency", "importance", "source", "approximate"]
OUTCOME_COLUMNS = ["actual", "forecast", "previous"]

FOMC_SOURCE = "federalreserve.gov/monetarypolicy/fomccalendars.htm (retrieved 2026-09-26)"
FOMC_HISTORICAL_SOURCE = "federalreserve.gov/monetarypolicy/fomchistorical{year}.htm (retrieved 2026-09-26)"

#: FOMC statement release: (date, New York local time "HH:MM", time_verified).
#: See module docstring for provenance and the meaning of the flag.
FOMC_STATEMENTS: tuple[tuple[str, str, bool], ...] = (
    # 2012 — dates verified (fomchistorical2012.htm); times per Fed practice, unverified
    ("2012-01-25", "12:30", False), ("2012-03-13", "14:15", False), ("2012-04-25", "12:30", False),
    ("2012-06-20", "12:30", False), ("2012-08-01", "14:15", False), ("2012-09-13", "12:30", False),
    ("2012-10-24", "14:15", False), ("2012-12-12", "12:30", False),
    # 2013-2015 — dates verified; 14:00 ET per Fed practice, unverified
    ("2013-01-30", "14:00", False), ("2013-03-20", "14:00", False), ("2013-05-01", "14:00", False),
    ("2013-06-19", "14:00", False), ("2013-07-31", "14:00", False), ("2013-09-18", "14:00", False),
    ("2013-10-30", "14:00", False), ("2013-12-18", "14:00", False),
    ("2014-01-29", "14:00", False), ("2014-03-19", "14:00", False), ("2014-04-30", "14:00", False),
    ("2014-06-18", "14:00", False), ("2014-07-30", "14:00", False), ("2014-09-17", "14:00", False),
    ("2014-10-29", "14:00", False), ("2014-12-17", "14:00", False),
    ("2015-01-28", "14:00", False), ("2015-03-18", "14:00", False), ("2015-04-29", "14:00", False),
    ("2015-06-17", "14:00", False), ("2015-07-29", "14:00", False), ("2015-09-17", "14:00", False),
    ("2015-10-28", "14:00", False), ("2015-12-16", "14:00", False),
    # 2016 onward — date and 14:00 ET time verified from each statement press release
    ("2016-01-27", "14:00", True), ("2016-03-16", "14:00", True), ("2016-04-27", "14:00", True),
    ("2016-06-15", "14:00", True), ("2016-07-27", "14:00", True), ("2016-09-21", "14:00", True),
    ("2016-11-02", "14:00", True), ("2016-12-14", "14:00", True),
    ("2017-02-01", "14:00", True), ("2017-03-15", "14:00", True), ("2017-05-03", "14:00", True),
    ("2017-06-14", "14:00", True), ("2017-07-26", "14:00", True), ("2017-09-20", "14:00", True),
    ("2017-11-01", "14:00", True), ("2017-12-13", "14:00", True),
    ("2018-01-31", "14:00", True), ("2018-03-21", "14:00", True), ("2018-05-02", "14:00", True),
    ("2018-06-13", "14:00", True), ("2018-08-01", "14:00", True), ("2018-09-26", "14:00", True),
    ("2018-11-08", "14:00", True), ("2018-12-19", "14:00", True),
    ("2019-01-30", "14:00", True), ("2019-03-20", "14:00", True), ("2019-05-01", "14:00", True),
    ("2019-06-19", "14:00", True), ("2019-07-31", "14:00", True), ("2019-09-18", "14:00", True),
    ("2019-10-30", "14:00", True), ("2019-12-11", "14:00", True),
    ("2020-01-29", "14:00", True),
    ("2020-03-18", "14:00", False),  # scheduled 17-18 Mar meeting, superseded on 15 Mar (see doc)
    ("2020-04-29", "14:00", True), ("2020-06-10", "14:00", True), ("2020-07-29", "14:00", True),
    ("2020-09-16", "14:00", True), ("2020-11-05", "14:00", True), ("2020-12-16", "14:00", True),
    ("2021-01-27", "14:00", True), ("2021-03-17", "14:00", True), ("2021-04-28", "14:00", True),
    ("2021-06-16", "14:00", True), ("2021-07-28", "14:00", True), ("2021-09-22", "14:00", True),
    ("2021-11-03", "14:00", True), ("2021-12-15", "14:00", True),
    ("2022-01-26", "14:00", True), ("2022-03-16", "14:00", True), ("2022-05-04", "14:00", True),
    ("2022-06-15", "14:00", True), ("2022-07-27", "14:00", True), ("2022-09-21", "14:00", True),
    ("2022-11-02", "14:00", True), ("2022-12-14", "14:00", True),
    ("2023-02-01", "14:00", True), ("2023-03-22", "14:00", True), ("2023-05-03", "14:00", True),
    ("2023-06-14", "14:00", True), ("2023-07-26", "14:00", True), ("2023-09-20", "14:00", True),
    ("2023-11-01", "14:00", True), ("2023-12-13", "14:00", True),
    ("2024-01-31", "14:00", True), ("2024-03-20", "14:00", True), ("2024-05-01", "14:00", True),
    ("2024-06-12", "14:00", True), ("2024-07-31", "14:00", True), ("2024-09-18", "14:00", True),
    ("2024-11-07", "14:00", True), ("2024-12-18", "14:00", True),
    ("2025-01-29", "14:00", True), ("2025-03-19", "14:00", True), ("2025-05-07", "14:00", True),
    ("2025-06-18", "14:00", True), ("2025-07-30", "14:00", True), ("2025-09-17", "14:00", True),
    ("2025-10-29", "14:00", True), ("2025-12-10", "14:00", True),
    ("2026-01-28", "14:00", True), ("2026-03-18", "14:00", True), ("2026-04-29", "14:00", True),
    ("2026-06-17", "14:00", True), ("2026-07-29", "14:00", True), ("2026-09-16", "14:00", True),
    # scheduled, not yet held at retrieval time — official dates, assumed 14:00 ET
    ("2026-10-28", "14:00", False), ("2026-12-09", "14:00", False),
    ("2027-01-27", "14:00", False), ("2027-03-17", "14:00", False), ("2027-04-28", "14:00", False),
    ("2027-06-09", "14:00", False), ("2027-07-28", "14:00", False), ("2027-09-15", "14:00", False),
    ("2027-10-27", "14:00", False), ("2027-12-08", "14:00", False),
)

#: First and last calendar years covered by :data:`FOMC_STATEMENTS`.
FOMC_COVERAGE = (2012, 2027)


# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------
def _to_utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _before_end(times: pd.Series, end: Any) -> pd.Series:
    """Mask ``times <= end`` where a bare-date ``end`` covers its WHOLE day.

    A bare date (or a timestamp exactly at UTC midnight) means the whole day, as in
    ``aurum.data.dukascopy``: ``end="2024-01-05"`` must include that day's 13:30 UTC NFP.
    Treating it as the instant 00:00 silently dropped the last day's events — e.g. a live
    runner asking for "events through today" would lose today's release and its risk
    blackout. Including extra *scheduled* events is never look-ahead; dropping them is
    the unsafe direction.
    """
    e = _to_utc(end)
    if e == e.normalize():
        return times < e + pd.Timedelta(days=1)
    return times <= e


def ny_to_utc(day: dt.date | str, hhmm: str) -> pd.Timestamp:
    """New York wall-clock time on ``day`` → UTC (DST-aware via the IANA database)."""
    local = pd.Timestamp(f"{pd.Timestamp(day).date()} {hhmm}")
    return local.tz_localize(NY_TZ).tz_convert("UTC")


def empty_events() -> pd.DataFrame:
    """An empty frame with the canonical event columns and dtypes."""
    return pd.DataFrame(
        {
            "time": pd.Series([], dtype="datetime64[ns, UTC]"),
            "name": pd.Series([], dtype=object),
            "currency": pd.Series([], dtype=object),
            "importance": pd.Series([], dtype="int64"),
            "source": pd.Series([], dtype=object),
            "approximate": pd.Series([], dtype=bool),
        }
    )


def validate_events(events: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``events`` violates the SPEC §3.5 format."""
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"event frame missing columns {missing}")
    t = events["time"]
    if not isinstance(t.dtype, pd.DatetimeTZDtype):
        raise ValueError("event 'time' must be tz-aware (UTC)")
    if t.isna().any():
        raise ValueError("event 'time' has missing values")
    imp = events["importance"]
    if len(imp) and ((imp < 1) | (imp > 3)).any():
        raise ValueError("importance must be in 1..3")
    if not isinstance(events.index, pd.RangeIndex):
        raise ValueError("event frame must have a RangeIndex")


def _finalize(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return empty_events()
    df["time"] = pd.DatetimeIndex(df["time"]).tz_convert("UTC").as_unit("ns")
    df["importance"] = df["importance"].astype("int64")
    df["approximate"] = df["approximate"].astype(bool)
    extra = [c for c in df.columns if c not in EVENT_COLUMNS]
    df = df[EVENT_COLUMNS + extra].sort_values(["time", "name"], kind="stable").reset_index(drop=True)
    validate_events(df)
    return df


# ------------------------------------------------------------------------------------
# rule-based calendar
# ------------------------------------------------------------------------------------
def _first_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def _bls_friday(year: int, month: int) -> dt.date:
    """BLS convention: the Employment Situation for month M-1 is released on the third
    Friday after the end of the reference week (the Sun-Sat week containing the 12th)."""
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    twelfth = dt.date(py, pm, 12)
    week_end = twelfth + dt.timedelta(days=(5 - twelfth.weekday()) % 7)  # Saturday
    return week_end + dt.timedelta(days=6 + 14)  # 3rd Friday after the Saturday


def nfp_release_date(year: int, month: int, rule: str = "first_friday") -> dt.date:
    """Approximate Employment Situation (NFP) release date in ``month``.

    ``rule="first_friday"`` (SPEC default) or ``"bls"`` (reference-week rule, which gets
    months such as Dec-2023 → 8 Dec right). Federal-holiday adjustments are applied: a
    release falling on 1 Jan moves one week later; one falling on Independence Day moves to
    the Thursday before — that is 4 Jul itself *and* Friday 3 Jul, the observed holiday when
    4 Jul is a Saturday (BLS released on Thu 2 Jul in 2015 and 2020; 2026 is the same case).
    Remaining exceptions (government shutdowns, ad-hoc BLS moves) are why rows are flagged
    ``approximate=True``.
    """
    if rule == "first_friday":
        d = _first_friday(year, month)
    elif rule == "bls":
        d = _bls_friday(year, month)
    else:
        raise ValueError("rule must be 'first_friday' or 'bls'")
    if d.month == 1 and d.day == 1:
        d += dt.timedelta(days=7)
    elif d.month == 7 and d.day in (3, 4):  # d is a Friday: 4 Jul, or 3 Jul observed holiday
        d -= dt.timedelta(days=1)
    return d


def fomc_statements(start: Any = None, end: Any = None) -> pd.DataFrame:
    """Scheduled FOMC statement releases in [start, end] (UTC) as an event frame.

    A bare-date ``end`` includes the whole day (see :func:`_before_end`)."""
    rows = []
    for day, hhmm, verified in FOMC_STATEMENTS:
        rows.append(
            {
                "time": ny_to_utc(day, hhmm),
                "name": "FOMC",
                "currency": "USD",
                "importance": 3,
                "source": FOMC_HISTORICAL_SOURCE.format(year=day[:4]) if day < "2021" else FOMC_SOURCE,
                "approximate": not verified,
            }
        )
    ev = _finalize(rows)
    if start is not None:
        ev = ev.loc[ev["time"] >= _to_utc(start)]
    if end is not None:
        ev = ev.loc[_before_end(ev["time"], end)]
    return ev.reset_index(drop=True)


def generate_rule_based_calendar(
    start: Any,
    end: Any,
    *,
    include: tuple[str, ...] = ("NFP", "FOMC"),
    nfp_rule: str = "first_friday",
) -> pd.DataFrame:
    """Rule-based USD calendar for [start, end] (UTC, inclusive; a bare-date ``end`` covers
    the whole day, like ``download_dukascopy``).

    * NFP: 08:30 America/New_York on the date from :func:`nfp_release_date` (12:30 UTC in
      US summer time, 13:30 UTC in winter), ``approximate=True``.
    * FOMC: statement times from :data:`FOMC_STATEMENTS` (2012-2027 only; outside that
      range no FOMC rows are produced and a warning is logged).

    CPI is not rule-based (BLS schedule varies) — import it with :func:`load_calendar_csv`.
    """
    s, e = _to_utc(start), _to_utc(end)
    if e < s:
        raise ValueError("end before start")
    unknown = set(include) - {"NFP", "FOMC"}
    if unknown:
        raise ValueError(f"rule-based calendar cannot generate {sorted(unknown)}")
    rows: list[dict] = []
    if "NFP" in include:
        for m in pd.date_range(s.tz_localize(None).normalize().replace(day=1), e.tz_localize(None), freq="MS"):
            d = nfp_release_date(m.year, m.month, nfp_rule)
            rows.append(
                {
                    "time": ny_to_utc(d, "08:30"),
                    "name": "NFP",
                    "currency": "USD",
                    "importance": 3,
                    "source": f"rule:{nfp_rule}",
                    "approximate": True,
                }
            )
    frames = [_finalize(rows)] if rows else []
    if "FOMC" in include:
        lo, hi = FOMC_COVERAGE
        if s.year < lo or e.year > hi:
            logger.warning(
                "FOMC dates only verified for %d-%d; no FOMC rows outside that range", lo, hi
            )
        frames.append(fomc_statements())
    if not frames:
        return empty_events()
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.loc[(ev["time"] >= s) & _before_end(ev["time"], end)]
    return _finalize(ev.reset_index(drop=True))


# ------------------------------------------------------------------------------------
# CSV import / merge
# ------------------------------------------------------------------------------------
_TRUE = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE = {"false", "f", "no", "n", "0", "0.0", "", "nan", "none"}


def _parse_bool(value: Any) -> bool:
    """CSV flag → bool. ``bool("False")`` is True, so strings are parsed explicitly."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean flag")


def load_calendar_csv(path: str | Path, *, tz: str = "UTC", source: str | None = None) -> pd.DataFrame:
    """Load a generic calendar CSV: ``time,name,currency,importance[,actual,forecast,previous]``.

    Column names are case-insensitive. Naive times are interpreted in ``tz`` (IANA name,
    fixed offset or ``"NY+7"``); times with explicit offsets are converted to UTC. ``importance`` accepts 1..3 or
    ``low/medium/high``. Missing ``currency`` defaults to "USD"; ``approximate`` defaults
    to False; ``source`` defaults to ``"csv:<file name>"``.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("time", "name"):
        if c not in df.columns:
            raise ValueError(f"calendar CSV lacks required column {c!r}")
    t = parse_timestamps(df["time"].astype(str), tz)
    out = pd.DataFrame({"time": t, "name": df["name"].astype(str).str.strip()})
    out["currency"] = df["currency"].astype(str).str.strip().str.upper() if "currency" in df.columns else "USD"
    if "importance" in df.columns:
        imp = df["importance"]
        mapping = {"low": 1, "medium": 2, "med": 2, "high": 3}
        imp = imp.map(lambda v: mapping.get(str(v).strip().lower(), v))
        out["importance"] = pd.to_numeric(imp, errors="coerce").fillna(1).clip(1, 3).astype("int64")
    else:
        out["importance"] = 2
    out["source"] = df["source"].astype(str) if "source" in df.columns else (source or f"csv:{Path(str(path)).name}")
    out["approximate"] = df["approximate"].map(_parse_bool) if "approximate" in df.columns else False
    for c in OUTCOME_COLUMNS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return _finalize(out)


def merge_calendars(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate event frames, dropping duplicates of (time, name) — first frame wins."""
    frames = tuple(f for f in frames if f is not None and len(f))
    if not frames:
        return empty_events()
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.drop_duplicates(subset=["time", "name"], keep="first")
    return _finalize(ev.reset_index(drop=True))


__all__ = [
    "EVENT_COLUMNS",
    "FOMC_COVERAGE",
    "FOMC_STATEMENTS",
    "empty_events",
    "fomc_statements",
    "generate_rule_based_calendar",
    "load_calendar_csv",
    "merge_calendars",
    "nfp_release_date",
    "ny_to_utc",
    "validate_events",
]
