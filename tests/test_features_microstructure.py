"""``session`` (DST-aware clocks) and ``microstructure`` groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurum.core.types import MarketData
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.features.microstructure import microstructure_features, session_features


def _hourly_bars(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    px = np.full(periods, 2000.0)
    df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px}, index=idx)
    return make_bars(df, "H1", default_spread=0.3)


def _session_at(decision_utc: str) -> pd.Series:
    """Session features for the bar whose close (decision time) is ``decision_utc``."""
    t = pd.Timestamp(decision_utc, tz="UTC")
    bars = _hourly_bars(str(t - pd.Timedelta(hours=1)).replace("+00:00", ""), 1)
    out = session_features(MarketData(bars=bars))
    return out.iloc[0]


@pytest.mark.parametrize(
    ("decision", "london", "ny", "overlap"),
    [
        # Winter (GMT / EST): London 08-17 UTC, NY 13-22 UTC.
        ("2024-01-15 07:00", 0, 0, 0),
        ("2024-01-15 08:00", 1, 0, 0),
        ("2024-01-15 12:00", 1, 0, 0),
        ("2024-01-15 13:00", 1, 1, 1),
        ("2024-01-15 17:00", 0, 1, 0),
        ("2024-01-15 22:00", 0, 0, 0),
        # Summer (BST / EDT): London 07-16 UTC, NY 12-21 UTC.
        ("2024-07-15 06:00", 0, 0, 0),
        ("2024-07-15 07:00", 1, 0, 0),
        ("2024-07-15 12:00", 1, 1, 1),
        ("2024-07-15 16:00", 0, 1, 0),
        ("2024-07-15 21:00", 0, 0, 0),
        # DST mismatch week: US on EDT since 10 Mar, UK still on GMT until 31 Mar 2024.
        ("2024-03-12 08:00", 1, 0, 0),
        ("2024-03-12 12:00", 1, 1, 1),
        ("2024-03-12 16:00", 1, 1, 1),
        ("2024-03-12 17:00", 0, 1, 0),
        # Autumn mismatch: UK back on GMT (27 Oct), US still EDT until 3 Nov 2024.
        ("2024-10-29 12:00", 1, 1, 1),
        ("2024-10-29 07:00", 0, 0, 0),
        # Weekend
        ("2024-01-13 13:00", 0, 0, 0),
    ],
)
def test_session_flags_follow_local_dst(decision: str, london: int, ny: int, overlap: int) -> None:
    row = _session_at(decision)
    assert (row["session_london"], row["session_ny"], row["session_overlap"]) == (london, ny, overlap)


def test_asia_session_tokyo_hours() -> None:
    assert _session_at("2024-01-15 23:00")["session_asia"] == 1  # Tue 08:00 JST
    assert _session_at("2024-01-15 07:00")["session_asia"] == 1  # Mon 16:00 JST
    assert _session_at("2024-01-15 08:00")["session_asia"] == 0  # 17:00 JST
    assert _session_at("2024-07-15 23:00")["session_asia"] == 1  # no DST in Japan


def test_week_progress_and_rollover() -> None:
    # Sunday 17:00 New York = week open; winter = 22:00 UTC, summer = 21:00 UTC.
    assert _session_at("2024-01-14 22:00")["session_week_progress"] == pytest.approx(0.0)
    assert _session_at("2024-07-14 21:00")["session_week_progress"] == pytest.approx(0.0)
    assert _session_at("2024-01-19 22:00")["session_week_progress"] == pytest.approx(1.0)
    assert _session_at("2024-01-17 10:00")["session_week_progress"] == pytest.approx((24 * 2 + 12) / 120)
    assert _session_at("2024-01-16 22:00")["session_rollover"] == 1  # 17:00 EST
    assert _session_at("2024-07-16 21:00")["session_rollover"] == 1  # 17:00 EDT
    assert _session_at("2024-07-16 23:00")["session_rollover"] == 0  # 19:00 EDT
    assert _session_at("2024-01-19 18:00")["session_friday_late"] == 1


def test_session_uses_decision_time_and_cyclical_encoding() -> None:
    bars = _hourly_bars("2024-01-15 00:00", 48)
    out = session_features(MarketData(bars=bars))
    # bar opening 00:00 is decided at 01:00 UTC
    assert out["session_hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 1 / 24))
    radius = out["session_hour_sin"] ** 2 + out["session_hour_cos"] ** 2
    np.testing.assert_allclose(radius, 1.0)
    assert out.index.equals(bars.index)


def test_microstructure_columns_and_values() -> None:
    bars = make_synthetic_bars(1500, "H1", seed=9)
    out = microstructure_features(MarketData(bars=bars))
    assert out.index.equals(bars.index)
    np.testing.assert_allclose(out["microstructure_spread_bps"], bars["spread"] / bars["close"] * 1e4)
    # weekend re-open detection: synthetic data resumes Sunday 22:00 after Friday 20:00
    reopen = (bars.index.dayofweek == 6) & (bars.index.hour == 22)
    assert (out.loc[reopen, "microstructure_after_gap"] == 1).all()
    assert out["microstructure_after_gap"].iloc[1:].sum() == reopen.sum()
    assert out[f"microstructure_autocorr_{120}"].dropna().between(-1, 1).all()
    gap = (bars["open"] - bars["close"].shift(1)).iloc[500]
    from aurum.features.volatility import atr

    assert out["microstructure_gap_atr"].iloc[500] == pytest.approx(gap / atr(bars, 14).iloc[499])


def test_microstructure_handles_zero_volume() -> None:
    bars = _hourly_bars("2024-01-15 00:00", 300)  # volume column all zeros
    out = microstructure_features(MarketData(bars=bars))
    assert out["microstructure_volume_z"].isna().all()  # undefined, not +/-inf
    assert not np.isinf(out.to_numpy()).any()
