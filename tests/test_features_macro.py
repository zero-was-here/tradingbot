"""``macro`` group: availability timing, units, missing data, and correlation/beta sanity."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro
from aurum.features.macro import macro_features, session_date
from aurum.features.multi_timeframe import resample_anchored


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_synthetic_bars(3000, "H1", seed=21)


def _daily(values: np.ndarray, dates: pd.DatetimeIndex, lag: pd.Timedelta) -> pd.DataFrame:
    f = pd.DataFrame({"value": values}, index=dates)
    f.index.name = "date"
    f["available_at"] = f.index + lag
    return f


def test_no_macro_returns_empty_frame(bars: pd.DataFrame) -> None:
    out = macro_features(MarketData(bars=bars))
    assert out.shape == (len(bars), 0) and out.index.equals(bars.index)


def test_columns_units_and_missing_series(bars: pd.DataFrame, caplog) -> None:
    macro = make_synthetic_macro(bars, seed=21)
    macro.pop("real10y")
    with caplog.at_level(logging.INFO, logger="aurum.features.macro"):
        out = macro_features(MarketData(bars=bars, macro=macro))
    assert "macro_corr_dxy" in out.columns and "macro_corr_real10y" not in out.columns
    assert any("real10y" in r.getMessage() for r in caplog.records)
    for s in ("dxy", "spx", "vix", "us10y"):
        assert {f"macro_{s}_chg_1d", f"macro_{s}_chg_5d", f"macro_{s}_chg_20d", f"macro_{s}_z"} <= set(out.columns)
    # us10y is a yield: bp changes; dxy is a price: log returns.
    t = len(bars) - 1
    dec = bars["available_at"].iloc[t]
    y = macro["us10y"].loc[macro["us10y"]["available_at"] <= dec, "value"]
    d = macro["dxy"].loc[macro["dxy"]["available_at"] <= dec, "value"]
    assert out["macro_us10y_chg_1d"].iloc[t] == pytest.approx((y.iloc[-1] - y.iloc[-2]) * 100)
    assert out["macro_dxy_chg_5d"].iloc[t] == pytest.approx(np.log(d.iloc[-1] / d.iloc[-6]))


def test_values_only_visible_after_available_at(bars: pd.DataFrame) -> None:
    macro = make_synthetic_macro(bars, seed=21)
    out = macro_features(MarketData(bars=bars, macro=macro), series=["dxy"], beta_series=())
    f = macro["dxy"]
    avail = pd.DatetimeIndex(bars["available_at"])
    k = next(i for i in range(30, len(f)) if f.index[i].dayofweek == 2)  # a Wednesday
    day = f.index[k]
    # decision at 21:00 UTC (< 21:30 availability): still sees the previous day's change
    i_before = int(np.flatnonzero(avail == day + pd.Timedelta(hours=21))[0])
    i_after = int(np.flatnonzero(avail == day + pd.Timedelta(hours=22))[0])
    chg = np.log(f["value"]).diff()
    assert out["macro_dxy_chg_1d"].iloc[i_before] == pytest.approx(chg.iloc[k - 1])
    assert out["macro_dxy_chg_1d"].iloc[i_after] == pytest.approx(chg.iloc[k])


def test_non_monotone_availability_uses_running_max(bars: pd.DataFrame) -> None:
    dates = pd.date_range(bars.index[0].normalize(), periods=60, freq="B", tz="UTC")
    f = _daily(np.linspace(100, 110, 60), dates, pd.Timedelta(hours=21, minutes=30))
    # observation 10 is published late (after observation 11)
    f.iloc[10, f.columns.get_loc("available_at")] = dates[11] + pd.Timedelta(hours=23)
    out = macro_features(MarketData(bars=bars, macro={"dxy": f}), beta_series=())
    avail = pd.DatetimeIndex(bars["available_at"])
    i = int(np.flatnonzero(avail == dates[11] + pd.Timedelta(hours=22))[0])
    chg9 = np.log(f["value"].iloc[9] / f["value"].iloc[8])
    # obs 11 is "available" at 21:30 but depends on the change vs obs 10 -> not yet usable
    assert out["macro_dxy_chg_1d"].iloc[i] == pytest.approx(chg9)


def test_stale_series_becomes_nan(bars: pd.DataFrame) -> None:
    dates = pd.date_range(bars.index[0].normalize(), periods=20, freq="B", tz="UTC")
    f = _daily(np.linspace(100, 101, 20), dates, pd.Timedelta(hours=21, minutes=30))
    out = macro_features(MarketData(bars=bars, macro={"dxy": f}), beta_series=(), stale_days=5)
    last_avail = f["available_at"].iloc[-1]
    stale = pd.DatetimeIndex(bars["available_at"]) > last_avail + pd.Timedelta(days=5)
    assert out.loc[stale, "macro_dxy_chg_1d"].isna().all()
    assert out.loc[~stale, "macro_dxy_chg_1d"].iloc[-1] == pytest.approx(np.log(101 / f["value"].iloc[-2]))


def test_corr_and_beta_recover_exact_relation(bars: pd.DataFrame) -> None:
    d1 = resample_anchored(bars, "D1", 22)
    g = np.log(d1["close"]).diff().fillna(0.0).to_numpy()
    dates = session_date(d1.index, 22)
    dxy = _daily(100 * np.exp(np.cumsum(-0.5 * g)), dates, pd.Timedelta(hours=21, minutes=30))
    out = macro_features(MarketData(bars=bars, macro={"dxy": dxy}))
    tail = out.iloc[-200:]
    np.testing.assert_allclose(tail["macro_corr_dxy"], -1.0, atol=1e-9)
    np.testing.assert_allclose(tail["macro_beta_dxy"], -2.0, rtol=1e-9)  # g = -2 * dlog(dxy)


def test_series_kind_from_attrs(bars: pd.DataFrame) -> None:
    dates = pd.date_range(bars.index[0].normalize(), periods=40, freq="B", tz="UTC")
    f = _daily(np.linspace(-0.5, 0.5, 40), dates, pd.Timedelta(hours=21, minutes=30))
    f.attrs["kind"] = "spread"
    out = macro_features(MarketData(bars=bars, macro={"t10y2y_custom": f}), beta_series=())
    assert out["macro_t10y2y_custom_chg_1d"].dropna().gt(0).all()  # bp diffs, not NaN logs


# ---- reviewer: adversarial cases -------------------------------------------------------------
def test_broker_daily_bars_pair_same_trading_date(bars: pd.DataFrame) -> None:
    """MT5 'NY+7' D1 bars converted to UTC open at 21:00/22:00 UTC of the PREVIOUS calendar
    day. Gold's day-d return must be paired with the macro print of day d (not d-1)."""
    d1 = resample_anchored(bars, "D1", 22)                   # index = d-1 22:00 UTC
    assert (d1.index.hour == 22).all()
    g = np.log(d1["close"]).diff().fillna(0.0).to_numpy()
    dates = session_date(d1.index, 22)
    dxy = _daily(100 * np.exp(np.cumsum(-0.5 * g)), dates, pd.Timedelta(hours=21, minutes=30))
    out = macro_features(MarketData(bars=d1, macro={"dxy": dxy}))
    tail = out["macro_corr_dxy"].dropna().iloc[-20:]
    assert len(tail) == 20
    np.testing.assert_allclose(tail, -1.0, atol=1e-9)
    np.testing.assert_allclose(out["macro_beta_dxy"].dropna().iloc[-20:], -2.0, rtol=1e-9)


def test_column_set_does_not_depend_on_history_length(bars: pd.DataFrame) -> None:
    """A live runner computing on a short window must get the same schema as research,
    otherwise ``FeaturePipeline.transform`` raises on the 'missing' columns."""
    macro = make_synthetic_macro(bars, seed=21)
    full = macro_features(MarketData(bars=bars, macro=macro))
    for n in (1, 5, 30):
        short = bars.iloc[:n]
        cutoff = short["available_at"].iloc[-1]
        visible = {k: v.loc[pd.DatetimeIndex(v["available_at"]) <= cutoff] for k, v in macro.items()}
        out = macro_features(MarketData(bars=short, macro=visible))
        assert list(out.columns) == list(full.columns), n
        pd.testing.assert_frame_equal(out, full.iloc[:n])


def test_empty_bars_do_not_crash(bars: pd.DataFrame) -> None:
    macro = make_synthetic_macro(bars, seed=21)
    empty = bars.iloc[:0]
    out = macro_features(MarketData(bars=empty, macro=macro))
    full = macro_features(MarketData(bars=bars, macro=macro))
    assert out.shape == (0, full.shape[1]) and list(out.columns) == list(full.columns)


# ---- independent review ---------------------------------------------------------------------------
def test_utc_midnight_daily_bars_pair_same_trading_date() -> None:
    """Daily gold bars labelled at UTC midnight (e.g. futures settlement series) pair with
    the macro print of the SAME date (the midpoint rule works for any daily anchor)."""
    d1 = make_synthetic_bars(400, "D1", seed=5)
    assert (d1.index.hour == 0).all()
    g = np.log(d1["close"]).diff().fillna(0.0).to_numpy()
    dxy = _daily(100 * np.exp(np.cumsum(-0.5 * g)), d1.index, pd.Timedelta(hours=21, minutes=30))
    out = macro_features(MarketData(bars=d1, macro={"dxy": dxy}))
    np.testing.assert_allclose(out["macro_corr_dxy"].dropna().iloc[-50:], -1.0, atol=1e-9)


def test_series_without_visible_rows_keeps_nan_columns(bars: pd.DataFrame) -> None:
    macro = make_synthetic_macro(bars, seed=21)
    empty_dxy = macro["dxy"].iloc[:0]
    out = macro_features(MarketData(bars=bars, macro={**macro, "dxy": empty_dxy}))
    ref = macro_features(MarketData(bars=bars, macro=macro))
    assert list(out.columns) == list(ref.columns)
    dxy_cols = [c for c in out.columns if "dxy" in c]
    assert out[dxy_cols].isna().all().all()
    other = [c for c in out.columns if "dxy" not in c]
    pd.testing.assert_frame_equal(out[other], ref[other])


def test_macro_lookback_scales_with_bar_size() -> None:
    from aurum.features.macro import MACRO_LOOKBACK, macro_lookback

    assert macro_lookback({}, 60.0) == MACRO_LOOKBACK
    assert macro_lookback({}, 15.0) == 4 * MACRO_LOOKBACK
    assert macro_lookback({}, 1440.0) * 24 == MACRO_LOOKBACK
    assert macro_lookback({"z_min_periods": 250}, 60.0) > MACRO_LOOKBACK
