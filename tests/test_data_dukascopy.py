"""Tests for aurum.data.dukascopy (decoder, bid/ask merge, download orchestration)."""

from __future__ import annotations

import datetime as dt
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.data import dukascopy as dk
from aurum.data.schema import validate_bars


def _blob(records: list[tuple[int, int, int, int, int, float]]) -> bytes:
    """Build a Dukascopy-style candle file: >IIIIIf records (t, O, C, L, H, vol), LZMA-alone."""
    arr = np.array(records, dtype=dk.CANDLE_DTYPE)
    return lzma.compress(arr.tobytes(), format=lzma.FORMAT_ALONE)


def _day_records(base: int, *, spread_pts: int = 0, active_minutes: range = range(0, 1440),
                 seed: int = 0) -> list[tuple[int, int, int, int, int, float]]:
    """1440 minute records; minutes outside ``active_minutes`` are flat zero-volume fillers."""
    rng = np.random.default_rng(seed)
    recs = []
    px = base
    for m in range(1440):
        if m in active_minutes:
            o = px
            c = px + int(rng.integers(-300, 301))
            lo = min(o, c) - int(rng.integers(0, 100))
            hi = max(o, c) + int(rng.integers(0, 100))
            recs.append((m * 60, o + spread_pts, c + spread_pts, lo + spread_pts, hi + spread_pts, 0.01))
            px = c
        else:
            recs.append((m * 60, px + spread_pts, px + spread_pts, px + spread_pts, px + spread_pts, 0.0))
    return recs


DAY = dt.date(2024, 1, 15)


def test_decode_field_order_and_scale():
    # O=2047525 C=2046935 L=2046815 H=2047675 -> note O, C, L, H record order
    raw = _blob([(0, 2047525, 2046935, 2046815, 2047675, 0.01064), (60, 2046855, 2046915, 2046695, 2047005, 0.00878)])
    df = dk.decode_bi5_candles(raw, DAY, 1000)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2024-01-15 00:00", tz="UTC")
    assert df.index[1] == pd.Timestamp("2024-01-15 00:01", tz="UTC")
    row = df.iloc[0]
    assert row["open"] == pytest.approx(2047.525)
    assert row["close"] == pytest.approx(2046.935)
    assert row["low"] == pytest.approx(2046.815)
    assert row["high"] == pytest.approx(2047.675)
    assert row["volume"] == pytest.approx(0.01064, rel=1e-6)


def test_decode_autodetects_price_scale_for_gold():
    raw = _blob(_day_records(2_050_000))
    df = dk.decode_bi5_candles(raw, DAY, None)
    assert 1900 < df["close"].median() < 2200
    # after 2020 gold > 2000: x/1000 and x/10000 are both "plausible"; default scale must win
    assert dk.detect_price_scale(np.array([4_450_000.0]), "XAUUSD") == 1000
    assert dk.detect_price_scale(np.array([1_571_500.0]), "XAUUSD") == 1000
    # a feed scaled differently is still recognised (x/100 would give 1571.5)
    assert dk.detect_price_scale(np.array([157_150.0]), "XAUUSD") == 100
    assert dk.detect_price_scale(np.array([108_123.0]), "EURUSD") == 100_000
    with pytest.raises(ValueError):
        dk.detect_price_scale(np.array([], dtype=float))


def test_decode_empty_and_malformed():
    assert dk.decode_bi5_candles(b"", DAY, 1000).empty
    recs = [(0, 2000000, 2000100, 1999900, 2000200, 1.0),
            (60, 2000000, 2000100, 2000150, 2000200, 1.0),  # low above close -> invalid
            (90_000, 2000000, 2000000, 2000000, 2000000, 1.0)]  # outside the day
    df = dk.decode_bi5_candles(_blob(recs), DAY, 1000)
    assert len(df) == 1
    # trailing partial record is ignored
    raw = lzma.compress(np.array(recs[:1], dtype=dk.CANDLE_DTYPE).tobytes() + b"\x00" * 7, format=lzma.FORMAT_ALONE)
    assert len(dk.decode_bi5_candles(raw, DAY, 1000)) == 1


def test_combine_bid_ask_mid_spread_and_filler_removal():
    active = range(0, 1320)  # 22:00-24:00 flat (market closed)
    bid = dk.decode_bi5_candles(_blob(_day_records(2_050_000, active_minutes=active)), DAY, 1000)
    ask = dk.decode_bi5_candles(_blob(_day_records(2_050_000, spread_pts=300, active_minutes=active)), DAY, 1000)
    out = dk.combine_bid_ask(bid, ask)
    assert len(out) == 1320
    assert out.index.max() == pd.Timestamp("2024-01-15 21:59", tz="UTC")
    np.testing.assert_allclose(out["spread"], 0.300, atol=1e-9)
    np.testing.assert_allclose(out["close"], bid["close"].iloc[:1320] + 0.15, atol=1e-9)
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()


def test_combine_crossed_quotes_floor_to_zero():
    bid = dk.decode_bi5_candles(_blob([(0, 2000500, 2000500, 2000400, 2000600, 1.0)]), DAY, 1000)
    ask = dk.decode_bi5_candles(_blob([(0, 2000300, 2000300, 2000200, 2000400, 1.0)]), DAY, 1000)
    out = dk.combine_bid_ask(bid, ask)
    assert out["spread"].iloc[0] == 0.0


def test_urls_and_cache_paths_use_zero_based_month(tmp_path: Path):
    url = dk.datafeed_url("xauusd", DAY, "bid")
    assert url == "https://datafeed.dukascopy.com/datafeed/XAUUSD/2024/00/15/BID_candles_min_1.bi5"
    url_h = dk.datafeed_url("XAUUSD", dt.date(2024, 12, 1), "ASK", kind="hour")
    assert url_h == "https://datafeed.dukascopy.com/datafeed/XAUUSD/2024/11/ASK_candles_hour_1.bi5"
    p = dk.cache_path(tmp_path, "XAUUSD", DAY, "ASK")
    assert p.parts[-5:] == ("XAUUSD", "2024", "00", "15", "ASK_candles_min_1.bi5")


class _Resp:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code = status
        self.content = content


class FakeSession:
    """Serves synthetic files; optionally throttles the first N requests with 503."""

    def __init__(self, files: dict[str, bytes], throttle_first: int = 0):
        self.files = files
        self.calls: list[str] = []
        self.throttle_first = throttle_first

    def get(self, url: str, timeout: float = 0) -> _Resp:
        self.calls.append(url)
        if len(self.calls) <= self.throttle_first:
            return _Resp(503)
        if url in self.files:
            return _Resp(200, self.files[url])
        return _Resp(404)


def _files_for(days: list[dt.date], base: int = 2_050_000) -> dict[str, bytes]:
    files = {}
    for i, d in enumerate(days):
        active = range(0, 1260) if d.weekday() == 4 else range(0, 1440)  # Friday: close 21:00
        if d.weekday() == 6:
            active = range(1380, 1440)  # Sunday: opens 23:00
        files[dk.datafeed_url("XAUUSD", d, "BID")] = _blob(_day_records(base, active_minutes=active, seed=i))
        files[dk.datafeed_url("XAUUSD", d, "ASK")] = _blob(
            _day_records(base, spread_pts=250, active_minutes=active, seed=i))
    return files


def test_download_with_fake_session_and_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dk.time, "sleep", lambda s: None)
    monkeypatch.setattr(dk, "THROTTLE_COOLDOWN_S", 0.0)
    days = [dt.date(2024, 1, 11) + dt.timedelta(days=k) for k in range(5)]  # Thu..Mon
    sess = FakeSession(_files_for(days), throttle_first=2)
    today = dt.date(2024, 2, 1)
    bars = dk.download_dukascopy("XAUUSD", "2024-01-11", "2024-01-15", "M1", cache_dir=tmp_path,
                                 session=sess, max_workers=2, today=today)
    validate_bars(bars)
    assert bars.attrs["timeframe"] == "M1" and bars.attrs["price_scale"] == 1000
    # Saturday skipped; no bars between Friday 21:00 and Sunday 23:00
    assert not any(url.endswith("/13/BID_candles_min_1.bi5") for url in sess.calls)
    fri_close = pd.Timestamp("2024-01-12 21:00", tz="UTC")
    sun_open = pd.Timestamp("2024-01-14 23:00", tz="UTC")
    assert not ((bars.index >= fri_close) & (bars.index < sun_open)).any()
    np.testing.assert_allclose(bars["spread"], 0.25, atol=1e-9)
    assert bars.index[-1] == pd.Timestamp("2024-01-15 23:59", tz="UTC")
    # second call is served entirely from the cache
    n_calls = len(sess.calls)
    again = dk.download_dukascopy("XAUUSD", "2024-01-11", "2024-01-15", "H1", cache_dir=tmp_path,
                                  session=sess, max_workers=2, today=today)
    assert len(sess.calls) == n_calls
    assert again.attrs["timeframe"] == "H1"
    validate_bars(again)
    # Friday 20:00 H1 bar exists (last full hour), no weekend bars, trailing hour complete
    assert pd.Timestamp("2024-01-12 20:00", tz="UTC") in again.index
    assert again.index[-1] == pd.Timestamp("2024-01-15 23:00", tz="UTC")


def _hour_month_files(month: dt.date, px: int = 2_050_000) -> tuple[dict[str, bytes], int]:
    """Monthly hour files (Saturdays flat) with a 300-point (0.30) spread; returns last price."""
    n_hours = 31 * 24
    rng = np.random.default_rng(1)
    recs_b, recs_a = [], []
    for h in range(n_hours):
        wd = (month + dt.timedelta(days=h // 24)).weekday()
        if wd == 5:
            recs_b.append((h * 3600, px, px, px, px, 0.0))
            recs_a.append((h * 3600, px + 300, px + 300, px + 300, px + 300, 0.0))
            continue
        c = px + int(rng.integers(-2000, 2001))
        lo, hi = min(px, c) - 500, max(px, c) + 500
        recs_b.append((h * 3600, px, c, lo, hi, 1.0))
        recs_a.append((h * 3600, px + 300, c + 300, lo + 300, hi + 300, 1.0))
        px = c
    files = {
        dk.datafeed_url("XAUUSD", month, "BID", kind="hour"): _blob(recs_b),
        dk.datafeed_url("XAUUSD", month, "ASK", kind="hour"): _blob(recs_a),
    }
    return files, px


def test_download_hour_resolution_with_trailing_minute_month(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dk.time, "sleep", lambda s: None)
    today = dt.date(2024, 2, 3)  # January complete (>=3 days ago), February incomplete
    jan = dt.date(2024, 1, 1)
    files, px = _hour_month_files(jan)
    files.update(_files_for([dt.date(2024, 2, 1), dt.date(2024, 2, 2)], base=px))
    sess = FakeSession(files)
    h1 = dk.download_dukascopy("XAUUSD", "2024-01-01", "2024-02-02", "H1", cache_dir=tmp_path, session=sess,
                               source_resolution="H1", max_workers=1, today=today)
    validate_bars(h1)
    assert h1.attrs["source_resolution"] == "H1"
    assert h1.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert pd.Timestamp("2024-01-31 23:00", tz="UTC") in h1.index
    assert pd.Timestamp("2024-02-02 20:00", tz="UTC") in h1.index  # from minute files
    assert not (h1.index.dayofweek == 5).any()
    assert (h1.loc[:"2024-01-31", "spread"].round(6) == 0.3).all()
    # complete months come from hour files: no daily minute files requested for January
    assert not any("/2024/00/" in u and "candles_min" in u for u in sess.calls)
    d1 = dk.download_dukascopy("XAUUSD", "2024-01-01", "2024-02-02", "D1", cache_dir=tmp_path, session=sess,
                               source_resolution="H1", max_workers=1, today=today)
    validate_bars(d1)
    assert d1.attrs["timeframe"] == "D1"
    with pytest.raises(ValueError):
        dk.download_dukascopy("XAUUSD", "2024-01-01", "2024-01-02", "M15", source_resolution="H1",
                              session=sess, cache_dir=tmp_path, today=today)


def test_fetch_file_does_not_cache_fresh_empty(tmp_path: Path):
    sess = FakeSession({})
    today = dt.date(2024, 1, 16)
    assert dk.fetch_file("XAUUSD", dt.date(2024, 1, 15), "BID", cache_dir=tmp_path, session=sess, today=today) == b""
    assert not dk.cache_path(tmp_path, "XAUUSD", dt.date(2024, 1, 15), "BID").exists()
    dk.fetch_file("XAUUSD", dt.date(2023, 1, 15), "BID", cache_dir=tmp_path, session=sess, today=today)
    assert dk.cache_path(tmp_path, "XAUUSD", dt.date(2023, 1, 15), "BID").exists()


def test_fetch_file_gives_up_after_retries(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dk.time, "sleep", lambda s: None)

    class Broken:
        def get(self, url, timeout=0):
            raise ConnectionError("boom")

    with pytest.raises(dk.DukascopyError):
        dk.fetch_file("XAUUSD", DAY, "BID", cache_dir=tmp_path, session=Broken(), retries=2, today=dt.date(2025, 1, 1))


def test_adaptive_limiter_halves_and_recovers():
    lim = dk.AdaptiveLimiter(max_concurrency=4, cooldown=0.0)
    lim.acquire()
    lim.release(throttled=True)
    assert lim.limit == 2
    for _ in range(8):
        lim.acquire()
        lim.release(throttled=False)
    assert lim.limit == 3


@pytest.mark.network
def test_real_dukascopy_day(tmp_path: Path):
    bars = dk.download_dukascopy("XAUUSD", "2024-01-15", "2024-01-15", cache_dir=tmp_path, max_workers=2)
    assert 2000 < bars["close"].median() < 2100
    assert 0.05 < bars["spread"].median() < 1.0


def test_offline_mode_never_hits_network(tmp_path: Path):
    class NoNetwork:
        def get(self, url, timeout=0):  # pragma: no cover - must not be called
            raise AssertionError("network used in offline mode")

    with pytest.raises(dk.DukascopyError, match="offline"):
        dk.download_dukascopy("XAUUSD", "2024-01-15", "2024-01-15", cache_dir=tmp_path, session=NoNetwork(),
                              offline=True, today=dt.date(2024, 2, 1))
    files = _files_for([DAY])
    for side in dk.SIDES:
        p = dk.cache_path(tmp_path, "XAUUSD", DAY, side)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(files[dk.datafeed_url("XAUUSD", DAY, side)])
    bars = dk.download_dukascopy("XAUUSD", "2024-01-15", "2024-01-15", "D1", cache_dir=tmp_path,
                                 session=NoNetwork(), offline=True, today=dt.date(2024, 2, 1))
    assert len(bars) == 1 and bars.index[0] == pd.Timestamp("2024-01-15", tz="UTC")


def test_build_dataset_offline(tmp_path: Path):
    import json

    from aurum.data.store import load_bars

    cache = tmp_path / "cache"
    jan = dt.date(2024, 1, 1)
    files, _ = _hour_month_files(jan)
    files.update(_files_for([dt.date(2024, 1, d) for d in (28, 29, 30, 31)]))
    for url, blob in files.items():
        parts = url.split("/datafeed/XAUUSD/")[1].split("/")
        p = cache.joinpath("XAUUSD", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    man = dk.build_dataset(tmp_path / "store", start="2024-01-01", end="2024-01-31", cache_dir=cache, offline=True)
    assert set(man["files"]) == {"xauusd_H1", "xauusd_H4", "xauusd_D1", "xauusd_D1_nyclose", "xauusd_M15"}
    m15 = load_bars(tmp_path / "store" / "xauusd_M15.parquet")
    assert m15.index[0] == pd.Timestamp("2024-01-28 23:00", tz="UTC")  # Sunday open, contiguous cache
    d1 = load_bars(tmp_path / "store" / "xauusd_D1.parquet")
    assert d1.index[-1] == pd.Timestamp("2024-01-31", tz="UTC")  # last day kept (complete_until)
    ny = load_bars(tmp_path / "store" / "xauusd_D1_nyclose.parquet")
    assert (ny.index.hour == 22).all()
    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest["files"]["xauusd_H1"]["rows"] == len(load_bars(tmp_path / "store" / "xauusd_H1.parquet"))
    assert "median_spread_by_year" in manifest["quality"]["xauusd_M15"]


class _HeaderResp(_Resp):
    def __init__(self, status: int, content: bytes = b"", headers: dict | None = None):
        super().__init__(status, content)
        self.headers = headers or {}


def test_fetch_file_honours_retry_after_and_deadline(tmp_path: Path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(dk.time, "sleep", lambda s: sleeps.append(s))

    class ThrottleOnce:
        def __init__(self):
            self.n = 0

        def get(self, url, timeout=0):
            self.n += 1
            if self.n == 1:
                return _HeaderResp(429, headers={"Retry-After": "7"})
            return _HeaderResp(200, b"")

    dk.fetch_file("XAUUSD", DAY, "BID", cache_dir=tmp_path, session=ThrottleOnce(), today=dt.date(2025, 1, 1))
    assert len(sleeps) == 1 and sleeps[0] >= 7.0

    class AlwaysThrottled:
        def get(self, url, timeout=0):
            return _HeaderResp(429, headers={"Retry-After": "120"})

    sleeps.clear()
    with pytest.raises(dk.DukascopyError, match="time budget"):
        dk.fetch_file("XAUUSD", DAY, "ASK", cache_dir=tmp_path, session=AlwaysThrottled(),
                      today=dt.date(2025, 1, 1), deadline=dk.time.monotonic() + 5.0)
    assert sleeps == []  # gave up instead of sleeping past the deadline
    assert not dk.cache_path(tmp_path, "XAUUSD", DAY, "ASK").exists()


def test_prefetch_cache_newest_first_budget_resume_and_gaps(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dk.time, "sleep", lambda s: None)
    monkeypatch.setattr(dk, "THROTTLE_COOLDOWN_S", 0.0)
    days = [dt.date(2024, 1, 8) + dt.timedelta(days=k) for k in range(8)]  # Mon 8 .. Mon 15
    today = dt.date(2024, 2, 1)
    sess = FakeSession(_files_for(days))
    kw = dict(cache_dir=tmp_path, session=sess, today=today, max_workers=1)

    rep0 = dk.prefetch_cache("XAUUSD", "2024-01-08", "2024-01-15", time_budget_s=0.0, **kw)
    assert rep0.planned == 14 and rep0.fetched == 0 and rep0.not_attempted == 14 and rep0.stopped_early
    assert rep0.contiguous_start is None and sess.calls == []

    rep = dk.prefetch_cache("XAUUSD", "2024-01-08", "2024-01-15", **kw)
    assert "/2024/00/15/" in sess.calls[0]  # newest first
    assert rep.fetched == 14 and rep.failed == 0 and not rep.stopped_early
    assert rep.contiguous_start == dt.date(2024, 1, 8)
    n_calls = len(sess.calls)
    again = dk.prefetch_cache("XAUUSD", "2024-01-08", "2024-01-15", **kw)
    assert again.already_cached == 14 and again.fetched == 0 and len(sess.calls) == n_calls

    # a day that keeps failing breaks the contiguous run; later days are still cached
    class Flaky(FakeSession):
        def get(self, url, timeout=0):
            if "/2024/00/10/" in url:
                self.calls.append(url)
                return _Resp(429)
            return super().get(url, timeout)

    other = tmp_path / "other"
    rep2 = dk.prefetch_cache("XAUUSD", "2024-01-08", "2024-01-15", cache_dir=other, session=Flaky(_files_for(days)),
                             today=today, max_workers=2)
    assert rep2.failed == 2 and rep2.fetched == 12
    assert rep2.contiguous_start == dt.date(2024, 1, 11)


def test_adaptive_limiter_exponential_cooldown_and_deadline():
    lim = dk.AdaptiveLimiter(max_concurrency=2, cooldown=10.0, max_cooldown=40.0)
    t0 = dk.time.monotonic()
    for _ in range(4):  # consecutive throttles: pauses 10, 20, 40, 40 (x jitter 0.5..1.5)
        lim._active += 1
        lim.release(throttled=True)
    assert lim.limit == 1
    assert lim._resume_at - t0 >= 0.5 * 40.0 - 1e-6
    # a slot is not granted during the cool-down; the deadline makes acquire give up quickly
    assert lim.acquire(deadline=dk.time.monotonic() + 0.05) is False
    lim._resume_at = 0.0
    assert lim.acquire(deadline=dk.time.monotonic() + 0.05) is True
    lim.release(throttled=False)
    assert lim._throttle_streak == 0


def test_build_dataset_m1_path_when_minute_cache_is_complete(tmp_path: Path):
    import json

    from aurum.data.resample import resample_bars
    from aurum.data.store import load_bars

    cache = tmp_path / "cache"
    days = [dt.date(2024, 1, 10) + dt.timedelta(days=k) for k in range(6)]  # Wed 10 .. Mon 15
    for url, blob in _files_for([d for d in days if d.weekday() != 5]).items():
        parts = url.split("/datafeed/XAUUSD/")[1].split("/")
        p = cache.joinpath("XAUUSD", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    with pytest.raises(ValueError):
        dk.build_dataset(tmp_path / "x", start="2024-01-10", end="2024-01-15", cache_dir=cache, h1_source="M5")
    man = dk.build_dataset(tmp_path / "store", start="2024-01-10", end="2024-01-15", cache_dir=cache, offline=True)
    assert man["build_path"] == "M1"
    assert json.loads((tmp_path / "store" / "manifest.json").read_text())["minute_cache_contiguous_from"] == "2024-01-10"
    m15 = load_bars(tmp_path / "store" / "xauusd_M15.parquet")
    h1 = load_bars(tmp_path / "store" / "xauusd_H1.parquet")
    d1 = load_bars(tmp_path / "store" / "xauusd_D1.parquet")
    assert h1.attrs["source_resolution"] == "M1" and d1.attrs["daily_anchor_hour_utc"] == 0
    assert m15.index[0] == pd.Timestamp("2024-01-10", tz="UTC") and h1.index[-1] == pd.Timestamp("2024-01-15 23:00", tz="UTC")
    # every file is aggregated straight from M1 → consistent across timeframes
    h1_from_m15 = resample_bars(m15, "H1")
    common = h1.index.intersection(h1_from_m15.index)
    assert len(common) > 50
    for c in ("open", "high", "low", "close", "volume"):
        np.testing.assert_allclose(h1.loc[common, c], h1_from_m15.loc[common, c])
    assert d1.index[-1] == pd.Timestamp("2024-01-15", tz="UTC")
    assert not (d1.index.dayofweek == 5).any()


def test_fetch_file_rejects_truncated_payload(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dk.time, "sleep", lambda s: None)
    good = _blob(_day_records(2_050_000))

    class Truncating:
        def __init__(self):
            self.n = 0

        def get(self, url, timeout=0):
            self.n += 1
            return _Resp(200, good[: len(good) // 2] if self.n == 1 else good)

    sess = Truncating()
    data = dk.fetch_file("XAUUSD", DAY, "BID", cache_dir=tmp_path, session=sess, today=dt.date(2025, 1, 1))
    assert sess.n == 2 and data == good
    assert dk.cache_path(tmp_path, "XAUUSD", DAY, "BID").read_bytes() == good


# ------------------------------------------------------------------------------------ review
def _write_cache(cache: Path, files: dict[str, bytes]) -> None:
    for url, blob in files.items():
        parts = url.split("/datafeed/XAUUSD/")[1].split("/")
        p = cache.joinpath("XAUUSD", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)


def test_download_drops_bucket_cut_by_the_requested_start(tmp_path: Path):
    """Regression: a mid-week start with 22:00-anchored D1 (or an intraday start with H4)
    produced a first bar that silently lacked the part of its bucket before ``start`` —
    wrong open/high/low/volume presented as a complete bar."""
    days = [dt.date(2024, 1, 9) + dt.timedelta(days=k) for k in range(4)]  # Tue 9 .. Fri 12
    _write_cache(tmp_path, _files_for(days))
    kw = {"cache_dir": tmp_path, "offline": True, "today": dt.date(2024, 2, 1)}
    d1 = dk.download_dukascopy("XAUUSD", "2024-01-10", "2024-01-12", "D1", daily_anchor_hour_utc=22, **kw)
    assert d1.index[0] == pd.Timestamp("2024-01-10 22:00", tz="UTC")  # not the cut 01-09 22:00 bucket
    h4 = dk.download_dukascopy("XAUUSD", "2024-01-10 10:00", "2024-01-12", "H4", **kw)
    assert h4.index[0] == pd.Timestamp("2024-01-10 12:00", tz="UTC")
    # the first kept bar is a genuine full bucket
    m1 = dk.download_dukascopy("XAUUSD", "2024-01-09", "2024-01-12", "M1", **kw)
    full = m1.loc["2024-01-10 12:00":"2024-01-10 15:59"]
    assert h4["open"].iloc[0] == full["open"].iloc[0] and h4["high"].iloc[0] == full["high"].max()
    assert h4["volume"].iloc[0] == pytest.approx(full["volume"].sum())


def test_build_dataset_first_nyclose_bar_is_not_truncated(tmp_path: Path):
    cache = tmp_path / "cache"
    days = [dt.date(2024, 1, 10) + dt.timedelta(days=k) for k in range(6)]  # Wed 10 .. Mon 15
    _write_cache(cache, _files_for([d for d in days if d.weekday() != 5]))
    dk.build_dataset(tmp_path / "store", start="2024-01-10", end="2024-01-15", cache_dir=cache, offline=True)
    from aurum.data.store import load_bars

    ny = load_bars(tmp_path / "store" / "xauusd_D1_nyclose.parquet")
    assert ny.index[0] == pd.Timestamp("2024-01-10 22:00", tz="UTC")
    for name in ("M15", "H1", "H4", "D1", "D1_nyclose"):
        b = load_bars(tmp_path / "store" / f"xauusd_{name}.parquet")
        assert b.index[0] >= pd.Timestamp("2024-01-10", tz="UTC")
        assert b.index.unit == pd.DatetimeIndex(b["available_at"]).unit
