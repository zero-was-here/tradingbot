"""Tests for aurum.data.loaders (MT5 / generic CSV parsing, tz handling, quality report)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.data.loaders import bars_from_ohlc, load_csv, load_mt5_csv, quality_report, to_utc_index
from aurum.data.schema import SchemaError, validate_bars
from aurum.data.synthetic import make_synthetic_bars

MT5_TAB = (
    "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
    "2024.01.15\t00:00:00\t2050.10\t2051.00\t2049.50\t2050.80\t1234\t0\t25\n"
    "2024.01.15\t01:00:00\t2050.80\t2052.20\t2050.30\t2051.90\t987\t0\t18\n"
    "2024.07.15\t00:00:00\t2410.00\t2412.00\t2408.00\t2411.00\t1500\t0\t30\n"
)


def test_mt5_tab_with_spread_etc_gmt_minus_2(tmp_path: Path):
    p = tmp_path / "XAUUSD_H1.csv"
    p.write_text(MT5_TAB)
    bars = load_mt5_csv(p, "H1", server_tz="Etc/GMT-2", point_size=0.01)
    validate_bars(bars)
    # Etc/GMT-2 == fixed UTC+2 → subtract 2h all year
    assert bars.index[0] == pd.Timestamp("2024-01-14 22:00", tz="UTC")
    assert bars.index[2] == pd.Timestamp("2024-07-14 22:00", tz="UTC")
    assert bars["spread"].tolist() == pytest.approx([0.25, 0.18, 0.30])
    assert bars["volume"].tolist() == [1234.0, 987.0, 1500.0]
    assert bars["available_at"].iloc[0] == pd.Timestamp("2024-01-14 23:00", tz="UTC")
    assert bars.attrs["timeframe"] == "H1"


def test_mt5_ny_plus_7_is_dst_aware():
    bars = load_mt5_csv(io.StringIO(MT5_TAB), "H1", server_tz="NY+7")
    # winter: server = UTC+2 ; summer (US DST): server = UTC+3
    assert bars.index[0] == pd.Timestamp("2024-01-14 22:00", tz="UTC")
    assert bars.index[1] == pd.Timestamp("2024-01-14 23:00", tz="UTC")
    assert bars.index[2] == pd.Timestamp("2024-07-14 21:00", tz="UTC")


def test_ny_plus_7_follows_us_not_eu_dst():
    # 2024-03-11 (after US DST switch on 10 Mar, before EU switch on 31 Mar)
    idx = to_utc_index(pd.DatetimeIndex(["2024-03-11 12:00", "2024-03-08 12:00"]), "NY+7")
    assert idx[0] == pd.Timestamp("2024-03-11 09:00", tz="UTC")  # UTC+3
    assert idx[1] == pd.Timestamp("2024-03-08 10:00", tz="UTC")  # UTC+2
    eu = to_utc_index(pd.DatetimeIndex(["2024-03-11 12:00"]), "Europe/Athens")
    assert eu[0] == pd.Timestamp("2024-03-11 10:00", tz="UTC")  # EET still UTC+2


@pytest.mark.parametrize(
    "spec,expected",
    [("UTC", "2024-01-15 12:00"), ("UTC+2", "2024-01-15 10:00"), ("-05:00", "2024-01-15 17:00"),
     ("GMT+03:30", "2024-01-15 08:30"), ("Etc/GMT+5", "2024-01-15 17:00")],
)
def test_to_utc_index_specs(spec: str, expected: str):
    out = to_utc_index(pd.DatetimeIndex(["2024-01-15 12:00"]), spec)
    assert out[0] == pd.Timestamp(expected, tz="UTC")


def test_to_utc_index_rejects_garbage_and_passes_aware():
    with pytest.raises(ValueError):
        to_utc_index(pd.DatetimeIndex(["2024-01-15"]), "Mars/Olympus")
    aware = pd.DatetimeIndex(["2024-01-15 12:00"], tz="America/New_York")
    assert to_utc_index(aware, "Etc/GMT-2")[0] == pd.Timestamp("2024-01-15 17:00", tz="UTC")


def test_mt5_dst_fallback_ambiguity_does_not_crash():
    # NY fall-back 2024-11-03 01:00-02:00 local happens twice → server 08:00-09:00 (Sunday)
    idx = to_utc_index(pd.DatetimeIndex(["2024-11-03 08:30", "2024-11-03 10:00"]), "NY+7")
    assert idx.is_monotonic_increasing
    # spring-forward non-existent hour is shifted, not an error
    idx2 = to_utc_index(pd.DatetimeIndex(["2024-03-10 09:30"]), "NY+7")
    assert idx2.tz is not None


def test_mt5_comma_d1_without_time_and_no_spread():
    text = (
        "<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>\n"
        "2024.01.15,2050.1,2060.0,2040.0,2055.0,50000,0\n"
        "2024.01.16,2055.0,2058.0,2020.0,2028.0,60000,0\n"
    )
    with pytest.raises(SchemaError):
        load_mt5_csv(io.StringIO(text), "D1", server_tz="UTC")
    bars = load_mt5_csv(io.StringIO(text), "D1", server_tz="UTC", default_spread=0.35)
    assert bars.index[1] == pd.Timestamp("2024-01-16", tz="UTC")
    assert (bars["spread"] == 0.35).all()
    assert bars["available_at"].iloc[0] == pd.Timestamp("2024-01-16", tz="UTC")


def test_mt4_headerless_and_duplicates():
    text = (
        "2024.01.15,00:00,2050.10,2051.00,2049.50,2050.80,1234\n"
        "2024.01.15,00:00,2050.10,2051.00,2049.50,2050.90,1234\n"   # duplicate → keep last
        "2024.01.15,00:15,2050.80,2052.20,2050.30,2051.90,987\n"
    )
    bars = load_mt5_csv(io.StringIO(text), "M15", server_tz="UTC", default_spread=0.2)
    assert len(bars) == 2
    assert bars["close"].iloc[0] == 2050.90


def test_mt5_utf16_export(tmp_path: Path):
    p = tmp_path / "utf16.csv"
    p.write_bytes(MT5_TAB.encode("utf-16"))
    bars = load_mt5_csv(p, "H1", server_tz="UTC")
    assert len(bars) == 3


def test_load_csv_generic(tmp_path: Path):
    p = tmp_path / "bars.csv"
    p.write_text(
        "Time,Open,High,Low,Close,Volume\n"
        "2024-01-15 00:00,2050,2051,2049,2050.5,10\n"
        "2024-01-15 01:00,2050.5,2052,2050,2051,12\n"
    )
    bars = load_csv(p, "H1", default_spread=0.3)
    assert bars.index[0] == pd.Timestamp("2024-01-15", tz="UTC")
    assert (bars["spread"] == 0.3).all()
    ny = load_csv(p, "H1", tz="America/New_York", default_spread=0.3)
    assert ny.index[0] == pd.Timestamp("2024-01-15 05:00", tz="UTC")
    p2 = tmp_path / "offsets.csv"
    p2.write_text("time,open,high,low,close,spread\n2024-07-01T00:00:00+03:00,1,2,0.5,1.5,0.1\n")
    b2 = load_csv(p2, "H1")
    assert b2.index[0] == pd.Timestamp("2024-06-30 21:00", tz="UTC")


def test_bars_from_ohlc():
    idx = pd.date_range("2024-01-15", periods=3, freq="h")
    df = pd.DataFrame({"Open": [1.0, 2, 3], "High": [2.0, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5]}, index=idx)
    bars = bars_from_ohlc(df, "H1", 0.2)
    validate_bars(bars)
    assert str(bars.index.tz) == "UTC"
    with pytest.raises(SchemaError):
        bars_from_ohlc(df.drop(columns=["High"]).reset_index(drop=True), "H1", 0.2)


def test_quality_report_flags_gaps_zero_spreads_and_outliers():
    bars = make_synthetic_bars(3000, "H1", seed=7)
    bars = bars.drop(bars.index[1000:1010])  # a 10h intra-week hole
    close = bars["close"].to_numpy().copy()
    close[2000] *= 1.08  # a single spike
    bars = bars.assign(close=close,
                       high=np.maximum(bars["high"].to_numpy(), close),
                       low=np.minimum(bars["low"].to_numpy(), close))
    bars.iloc[5:15, bars.columns.get_loc("spread")] = 0.0
    bars.attrs["timeframe"] = "H1"
    rep = quality_report(bars)
    assert rep["n_rows"] == len(bars)
    assert rep["n_intraweek_gaps"] == 1
    assert rep["largest_intraweek_gaps"][0]["hours"] == 10.0
    assert rep["n_weekend_gaps"] >= 10 and rep["n_daily_break_gaps"] == 0
    assert rep["zero_spread_frac"] == pytest.approx(10 / len(bars))
    assert rep["n_return_outliers"] >= 1
    assert any(o["time"].startswith(str(bars.index[2000])[:13]) for o in rep["top_return_outliers"])
    assert set(rep["median_spread_by_year"]) <= {2020, 2021}


def test_quality_report_recognises_daily_break():
    idx = pd.date_range("2024-07-15", "2024-07-17 23:45", freq="15min", tz="UTC")
    idx = idx[~(idx.hour == 21)]  # 21:00-22:00 UTC maintenance break (US summer)
    n = len(idx)
    df = pd.DataFrame({"open": np.full(n, 2400.0), "high": np.full(n, 2401.0), "low": np.full(n, 2399.0),
                       "close": np.full(n, 2400.0), "spread": np.full(n, 0.3)}, index=idx)
    rep = quality_report(bars_from_ohlc(df, "M15"))
    assert rep["n_daily_break_gaps"] == 3 and rep["n_intraweek_gaps"] == 0
    assert rep["flat_bar_frac"] == 0.0


def test_load_csv_partial_spread_and_bad_volume(tmp_path: Path):
    p = tmp_path / "partial.csv"
    p.write_text(
        "time,open,high,low,close,volume,spread\n"
        "2024-01-15 00:00,2050,2051,2049,2050.5,10,0.25\n"
        "2024-01-15 01:00,2050.5,2052,2050,2051,,\n"
    )
    with pytest.raises(SchemaError, match="spread"):
        load_csv(p, "H1")
    bars = load_csv(p, "H1", default_spread=0.4)
    assert bars["spread"].tolist() == pytest.approx([0.25, 0.4])
    assert bars["volume"].tolist() == [10.0, 0.0]


# ------------------------------------------------------------------------------------ review
def test_load_csv_drops_unparseable_timestamps(tmp_path: Path, caplog):
    """A blank/garbage time used to surface as a misleading 'index must be strictly
    increasing' SchemaError for the whole file."""
    p = tmp_path / "gaps.csv"
    p.write_text(
        "time,open,high,low,close,spread\n"
        "2024-01-15 00:00,1,2,0.5,1.5,0.1\n"
        ",1,2,0.5,1.5,0.1\n"
        "2024-01-15 02:00,1,2,0.5,1.5,0.1\n"
    )
    with caplog.at_level("WARNING"):
        bars = load_csv(p, "H1")
    assert len(bars) == 2 and "timestamps" in caplog.text
    validate_bars(bars)


def test_mt5_ny_plus_7_d1_bars_start_at_new_york_close():
    """NY-close servers stamp D1 bars at server midnight = 17:00 New York the previous day."""
    text = (
        "<DATE>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.07.15\t2400\t2410\t2390\t2405\t1\t0\t20\n"
        "2024.01.15\t2000\t2010\t1990\t2005\t1\t0\t20\n"   # unsorted input is sorted
    )
    bars = load_mt5_csv(io.StringIO(text), "D1", server_tz="NY+7")
    assert bars.index.tolist() == [pd.Timestamp("2024-01-14 22:00", tz="UTC"), pd.Timestamp("2024-07-14 21:00", tz="UTC")]
    assert (bars["available_at"] - bars.index == pd.Timedelta(days=1)).all()


def test_quality_report_single_and_empty_bars():
    b = make_synthetic_bars(5, "H1", seed=1)
    assert quality_report(b.iloc[:0]) == {"n_rows": 0, "timeframe": "H1"}
    rep = quality_report(b.iloc[:1])
    assert rep["n_gaps"] == 0 and rep["n_return_outliers"] == 0
