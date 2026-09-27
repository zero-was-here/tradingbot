"""Tests for aurum.data.macro (frame contract, availability lags, FRED parsing, caching)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.data import macro as mc
from aurum.data.pit import asof_join

FRED_TEXT = "observation_date,DFII10\n2024-01-02,1.80\n2024-01-03,.\n2024-01-04,1.85\n"


def test_parse_fred_csv_handles_missing_and_header_variants():
    s = mc.parse_fred_csv(FRED_TEXT, "DFII10")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]
    assert s.iloc[1] == pytest.approx(1.85)
    old = "DATE,DFF\n2024-01-01,5.33\n"
    assert mc.parse_fred_csv(old).iloc[0] == pytest.approx(5.33)


def test_to_macro_frame_contract():
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [1.5, 2.5, 3.5], "Low": [0.5, 1.5, 2.5], "Close": [1.2, np.nan, 3.2],
         "Volume": [0, 0, 0]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], tz="America/New_York"),
    )
    f = mc.to_macro_frame(raw, availability_lag=pd.Timedelta(hours=21, minutes=30), source="yahoo:X")
    mc.validate_macro_frame(f)
    assert f.index.name == "date" and str(f.index.tz) == "UTC"
    # exchange-local calendar date is kept (not shifted by the NY→UTC offset)
    assert f.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert len(f) == 2  # NaN close dropped
    assert f["available_at"].iloc[0] == pd.Timestamp("2024-01-02 21:30", tz="UTC")
    assert {"value", "open", "high", "low", "volume", "available_at"} <= set(f.columns)


def test_default_lags_are_conservative():
    assert mc.default_yahoo_lag("^GSPC") == pd.Timedelta(hours=21, minutes=30)
    assert mc.default_yahoo_lag("^VIX") == pd.Timedelta(hours=21, minutes=30)
    # futures / ICE trade until 17:00 New York = 22:00 UTC in winter
    for t in ("GC=F", "SI=F", "CL=F", "DX-Y.NYB"):
        assert mc.default_yahoo_lag(t) == pd.Timedelta(hours=22, minutes=30)
    assert mc.FRED_LAG == pd.Timedelta(days=1, hours=21, minutes=30)


def test_fetch_fred_uses_cache_and_lag(tmp_path: Path, monkeypatch):
    calls = []

    def fake_text(sid, start, end, timeout=60.0):
        calls.append(sid)
        return FRED_TEXT

    monkeypatch.setattr(mc, "_fred_text", fake_text)
    out = mc.fetch_fred({"real10y": "DFII10"}, "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    f = out["real10y"]
    mc.validate_macro_frame(f)
    assert f["available_at"].iloc[0] == pd.Timestamp("2024-01-03 21:30", tz="UTC")
    assert f.attrs["source"] == "fred:DFII10"
    mc.fetch_fred({"real10y": "DFII10"}, "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert calls == ["DFII10"]  # second call served from cache
    # per-series lag override
    out2 = mc.fetch_fred({"real10y": "DFII10"}, "2024-01-01", "2024-01-31", cache_dir=tmp_path,
                         availability_lag={"real10y": pd.Timedelta(days=2)})
    assert out2["real10y"]["available_at"].iloc[0] == pd.Timestamp("2024-01-04", tz="UTC")


def test_fetch_yahoo_with_stubbed_history(tmp_path: Path, monkeypatch):
    def fake_hist(ticker, start, end_incl):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
        raw = pd.DataFrame({"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
                            "volume": [5.0, 6.0]}, index=idx)
        raw.index = mc._date_index(raw.index)
        if ticker == "BAD":
            raise ValueError("no data")
        return raw

    monkeypatch.setattr(mc, "_yahoo_history", fake_hist)
    out = mc.fetch_yahoo_daily({"gold_fut": "GC=F", "spx": "^GSPC", "bad": "BAD"}, "2024-01-01", "2024-01-31",
                               cache_dir=tmp_path)
    assert set(out) == {"gold_fut", "spx"}  # failing ticker skipped (logged)
    assert out["gold_fut"]["available_at"].iloc[0] == pd.Timestamp("2024-01-02 22:30", tz="UTC")
    assert out["spx"]["available_at"].iloc[0] == pd.Timestamp("2024-01-02 21:30", tz="UTC")
    assert (tmp_path / "yahoo").exists()
    with pytest.raises(ValueError):
        mc.fetch_yahoo_daily({"bad": "BAD"}, "2024-01-01", "2024-01-31", cache_dir=tmp_path, skip_errors=False)


def test_macro_frame_is_point_in_time_with_asof_join():
    f = mc.to_macro_frame(pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"])),
                          availability_lag=mc.FRED_LAG)
    decisions = pd.DatetimeIndex(["2024-01-03 21:00", "2024-01-03 21:30", "2024-01-04 22:00"], tz="UTC")
    got = asof_join(decisions, f, columns=["value"])["value"]
    assert np.isnan(got.iloc[0])  # 2 Jan value not yet published at 3 Jan 21:00
    assert got.iloc[1] == 1.0 and got.iloc[2] == 2.0


@pytest.mark.network
def test_real_fred_and_yahoo(tmp_path: Path):
    fred = mc.fetch_fred({"fedfunds": "DFF"}, "2024-01-01", "2024-02-01", cache_dir=tmp_path, skip_errors=False)
    assert 5.0 < fred["fedfunds"]["value"].median() < 5.6
    y = mc.fetch_yahoo_daily({"spx": "^GSPC"}, "2024-01-01", "2024-02-01", cache_dir=tmp_path, skip_errors=False)
    assert 4500 < y["spx"]["value"].median() < 5200


# ------------------------------------------------------------------------------------ review
DFF_WEEKEND = (
    "observation_date,DFF\n"
    "2024-01-11,5.33\n2024-01-12,5.33\n2024-01-13,5.33\n2024-01-14,5.33\n"
    "2024-01-15,5.33\n2024-01-16,5.33\n2024-01-17,5.33\n"
)


def test_fred_publication_lag_counts_us_business_days(tmp_path: Path, monkeypatch):
    """Regression (look-ahead): with calendar days a FRIDAY print became usable on Saturday,
    i.e. from the Sunday-evening open, although H.15 / the NY Fed publish it on the next
    business day (FRED on a Saturday ends on Thursday for DFII10). 2024-01-15 is MLK day."""
    monkeypatch.setattr(mc, "_fred_text", lambda sid, s, e, timeout=60.0: DFF_WEEKEND)
    f = mc.fetch_fred({"fedfunds": "DFF"}, "2024-01-01", "2024-01-31", cache_dir=tmp_path)["fedfunds"]
    mc.validate_macro_frame(f)
    av = dict(zip(f.index.strftime("%Y-%m-%d"), f["available_at"], strict=True))
    assert av["2024-01-11"] == pd.Timestamp("2024-01-12 21:30", tz="UTC")  # Thu → Fri
    tue = pd.Timestamp("2024-01-16 21:30", tz="UTC")                      # Mon is a holiday
    for d in ("2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15"):   # Fri, Sat, Sun, MLK
        assert av[d] == tue, d
    assert av["2024-01-16"] == pd.Timestamp("2024-01-17 21:30", tz="UTC")
    # at the Sunday-evening open and all through (holiday) Monday the latest visible print
    # is Thursday's — Friday's is not published yet
    decisions = pd.DatetimeIndex(["2024-01-14 23:00", "2024-01-15 20:00", "2024-01-16 21:29"], tz="UTC")
    seen = asof_join(decisions, f.assign(obs=f.index), columns=["obs"])["obs"]
    assert (seen == pd.Timestamp("2024-01-11", tz="UTC")).all()


def test_us_business_day_offset_rolls_back_then_forward():
    d = pd.DatetimeIndex(["2024-07-03", "2024-07-04", "2024-07-05", "2024-07-06", "2024-12-24"], tz="UTC")
    out = mc.us_business_day_offset(d, 1)
    assert out.strftime("%Y-%m-%d").tolist() == ["2024-07-05", "2024-07-05", "2024-07-08", "2024-07-08",
                                                 "2024-12-26"]
    assert mc.us_business_day_offset(d, 0).equals(d)
    lag = mc.lagged_available_at(d[:1], mc.FRED_LAG, business_days=True)
    assert lag[0] == pd.Timestamp("2024-07-05 21:30", tz="UTC")
    assert mc.lagged_available_at(d[:1], mc.FRED_LAG)[0] == pd.Timestamp("2024-07-04 21:30", tz="UTC")


def test_macro_frames_reject_zero_or_negative_lag():
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-02"]))
    for lag in (pd.Timedelta(0), pd.Timedelta(hours=-1)):
        with pytest.raises(ValueError, match="positive"):
            mc.to_macro_frame(s, availability_lag=lag)
    same_day = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"))
    same_day["available_at"] = same_day.index  # "known at 00:00 of its own date" → leak
    with pytest.raises(ValueError, match="not after"):
        mc.validate_macro_frame(same_day)
    nat = same_day.assign(available_at=pd.Series([pd.NaT], index=same_day.index, dtype="datetime64[ns, UTC]"))
    with pytest.raises(ValueError, match="missing"):
        mc.validate_macro_frame(nat)


def test_yahoo_provisional_bar_is_dropped_and_not_cached(tmp_path: Path, monkeypatch):
    """A fetch during the session returns today's still-moving bar; it must neither be used
    (its 'close' is an intraday snapshot) nor frozen into the raw cache as a final close."""
    calls = []

    def fake_hist(ticker, start, end_incl):
        calls.append(ticker)
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
        raw = pd.DataFrame({"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
                            "close": [1.0, 2.0], "volume": [5.0, 6.0]}, index=idx)
        raw.index = mc._date_index(raw.index)
        return raw

    monkeypatch.setattr(mc, "_yahoo_history", fake_hist)
    monkeypatch.setattr(mc, "_utcnow", lambda: pd.Timestamp("2024-01-03 18:00", tz="UTC"))
    out = mc.fetch_yahoo_daily({"spx": "^GSPC"}, "2024-01-01", "2024-01-03", cache_dir=tmp_path)
    assert out["spx"].index.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    assert not list(tmp_path.rglob("*.parquet"))
    # after the close the same request is final: both rows, and now it is cached
    monkeypatch.setattr(mc, "_utcnow", lambda: pd.Timestamp("2024-01-04 01:00", tz="UTC"))
    out2 = mc.fetch_yahoo_daily({"spx": "^GSPC"}, "2024-01-01", "2024-01-03", cache_dir=tmp_path)
    assert len(out2["spx"]) == 2 and len(list((tmp_path / "yahoo").glob("*.parquet"))) == 1
    mc.fetch_yahoo_daily({"spx": "^GSPC"}, "2024-01-01", "2024-01-03", cache_dir=tmp_path)
    assert len(calls) == 2  # third call served from cache


def test_save_macro_dir_harmonises_units(tmp_path: Path):
    f = mc.to_macro_frame(pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"])),
                          availability_lag=mc.YAHOO_CASH_LAG)
    f.index = f.index.as_unit("s")
    f["available_at"] = pd.DatetimeIndex(f["available_at"]).as_unit("us")
    back = mc.load_macro_dir(mc.save_macro_dir({"x": f}, tmp_path))["x"]
    assert back.index.unit == pd.DatetimeIndex(back["available_at"]).unit
    assert back["available_at"].iloc[0] == pd.Timestamp("2024-01-02 21:30", tz="UTC")
