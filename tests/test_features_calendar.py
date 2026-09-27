"""``calendar`` group: event proximity relative to the decision time ``available_at``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events
from aurum.features.calendar import calendar_features, classify_event


def _bars(start: str = "2024-03-04 00:00", periods: int = 24 * 7, freq: str = "h", tf: str = "H1"):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    px = np.full(periods, 2000.0)
    df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px}, index=idx)
    return make_bars(df, tf, default_spread=0.3)


def _events(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    ev = pd.DataFrame(rows, columns=["time", "name", "importance", "currency"])
    ev["time"] = pd.to_datetime(ev["time"], utc=True)
    ev["source"] = "test"
    ev["approximate"] = False
    return ev


@pytest.fixture
def events() -> pd.DataFrame:
    return _events([
        ("2024-03-05 13:30", "CPI y/y", 3, "USD"),
        ("2024-03-05 13:30", "Core CPI m/m", 3, "USD"),
        ("2024-03-06 15:00", "JOLTS", 2, "USD"),              # below importance threshold
        ("2024-03-07 12:00", "ECB Rate Decision", 3, "EUR"),
        ("2024-03-08 13:30", "Non-Farm Payrolls", 3, "USD"),
    ])


def test_hours_to_next_and_since_last(events: pd.DataFrame) -> None:
    bars = _bars()
    out = calendar_features(MarketData(bars=bars, events=events))
    t = pd.DatetimeIndex(bars["available_at"])
    at = pd.Series(np.arange(len(bars)), index=t)
    # decision 2024-03-05 13:00 -> CPI in 0.5h; last event: none -> capped 72
    r = out.iloc[at[pd.Timestamp("2024-03-05 13:00", tz="UTC")]]
    assert r["calendar_hours_to_next"] == pytest.approx(0.5)
    assert r["calendar_hours_since_last"] == pytest.approx(72.0)
    assert r["calendar_in_30m"] == 1 and r["calendar_pre_2h"] == 1 and r["calendar_post_2h"] == 0
    assert r["calendar_next_cpi"] == 1 and r["calendar_next_nfp"] == 0
    # decision 2024-03-05 15:00 -> CPI 1.5h ago; next = ECB 2024-03-07 12:00 (45h)
    r = out.iloc[at[pd.Timestamp("2024-03-05 15:00", tz="UTC")]]
    assert r["calendar_hours_since_last"] == pytest.approx(1.5)
    assert r["calendar_hours_to_next"] == pytest.approx(45.0)
    assert r["calendar_in_30m"] == 0 and r["calendar_in_2h"] == 1 and r["calendar_post_2h"] == 1
    assert r[["calendar_next_cpi", "calendar_next_nfp", "calendar_next_fomc"]].sum() == 0
    # JOLTS (importance 2) is ignored
    r = out.iloc[at[pd.Timestamp("2024-03-06 15:00", tz="UTC")]]
    assert r["calendar_in_30m"] == 0
    # decision 2024-03-08 13:00 -> NFP next
    r = out.iloc[at[pd.Timestamp("2024-03-08 13:00", tz="UTC")]]
    assert r["calendar_next_nfp"] == 1 and r["calendar_hours_to_next"] == pytest.approx(0.5)


def test_event_at_decision_time_is_upcoming() -> None:
    bars = _bars(periods=48, freq="30min", tf="M30")
    ev = _events([("2024-03-04 13:30", "FOMC Statement", 3, "USD")])
    out = calendar_features(MarketData(bars=bars, events=ev))
    t = pd.DatetimeIndex(bars["available_at"])
    i = int(np.flatnonzero(t == pd.Timestamp("2024-03-04 13:30", tz="UTC"))[0])
    assert out["calendar_hours_to_next"].iloc[i] == 0.0
    assert out["calendar_next_fomc"].iloc[i] == 1
    assert out["calendar_hours_since_last"].iloc[i] == 72.0
    assert out["calendar_hours_since_last"].iloc[i + 1] == pytest.approx(0.5)


def test_currency_filter_and_count(events: pd.DataFrame) -> None:
    bars = _bars()
    usd = calendar_features(MarketData(bars=bars, events=events), currencies=("USD",))
    alle = calendar_features(MarketData(bars=bars, events=events))
    t = pd.DatetimeIndex(bars["available_at"])
    i = int(np.flatnonzero(t == pd.Timestamp("2024-03-07 11:00", tz="UTC"))[0])
    assert alle["calendar_hours_to_next"].iloc[i] == pytest.approx(1.0)       # ECB
    assert usd["calendar_hours_to_next"].iloc[i] == pytest.approx(26.5)       # NFP
    j = int(np.flatnonzero(t == pd.Timestamp("2024-03-05 00:00", tz="UTC"))[0])
    assert alle["calendar_n_next_24h"].iloc[j] == 1  # the two CPI prints share one timestamp


def test_no_events_gives_empty_frame() -> None:
    bars = _bars()
    out = calendar_features(MarketData(bars=bars, events=None))
    assert out.shape == (len(bars), 0) and out.index.equals(bars.index)


def test_empty_calendar_is_capped() -> None:
    bars = _bars()
    out = calendar_features(MarketData(bars=bars, events=_events([])))
    assert (out["calendar_hours_to_next"] == 72).all() and (out["calendar_in_2h"] == 0).all()


def test_classify_event_patterns() -> None:
    kinds = classify_event(pd.Series(["Nonfarm Payrolls", "CPI m/m", "FOMC Minutes", "GDP"]))
    assert kinds["nfp"].tolist() == [True, False, False, False]
    assert kinds["cpi"].tolist() == [False, True, False, False]
    assert kinds["fomc"].tolist() == [False, False, True, False]


def test_synthetic_calendar_runs() -> None:
    bars = make_synthetic_bars(1000, seed=2)
    ev = make_synthetic_events(bars.index[0], bars.index[-1] + pd.Timedelta(days=5))
    out = calendar_features(MarketData(bars=bars, events=ev))
    assert out.notna().all().all()
    assert out["calendar_hours_to_next"].between(0, 72).all()


# ---- independent review ---------------------------------------------------------------------------
def test_missing_event_times_are_ignored(events: pd.DataFrame) -> None:
    bars = _bars()
    bad = pd.concat([events, _events([("2024-03-06 10:00", "NFP", 3, "USD")]).assign(time=pd.NaT)],
                    ignore_index=True)
    ref = calendar_features(MarketData(bars=bars, events=events))
    out = calendar_features(MarketData(bars=bars, events=bad))
    pd.testing.assert_frame_equal(out, ref)
