"""Dukascopy historical candle downloader and decoder (SPEC §3.2).

Dukascopy Bank publishes free historical *bid* and *ask* candles for its ECN feed:

* one-minute candles, one file per UTC day and side::

    https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM-1:02d}/{DD:02d}/{BID|ASK}_candles_min_1.bi5

* one-hour candles, one file per UTC month and side (used by ``source_resolution="H1"``)::

    https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM-1:02d}/{BID|ASK}_candles_hour_1.bi5

Format facts (verified against the live feed on 2026-09-26):

* the month is **zero-based** (January = ``00``);
* the payload is LZMA-compressed in the legacy ``.lzma`` ("LZMA alone") container;
* the decompressed payload is a sequence of 24-byte big-endian records ``>IIIIIf``:
  ``seconds-from-period-start (UTC), open, close, low, high, volume`` — note the unusual
  **O, C, L, H** order. The period is the day (minute files) or the month (hour files);
* prices are integers in "points": ``price = int / price_scale``. For XAUUSD the scale is
  1000 — 2024-01-15 decodes to ~2047-2058 USD/oz with a median ~0.32 USD spread;
* every file carries every minute (hour) of its period. Periods without any tick
  (weekends, the daily 21:00/22:00 UTC break, holidays) are *flat filler candles* with zero
  volume repeating the last price. They are not tradeable bars and are dropped.

Rate limiting: the free feed throttles aggressively (observed 2026-09: >4 concurrent requests
get HTTP 503; sustained use is tarpitted to ~10-15 s per file and then answered with HTTP 429
"Too Many Requests" for extended periods — Dukascopy directs bulk history users to a
requester-pays S3 export instead, see https://www.dukascopy.com/wiki/en/development/data-export/).
Hence the on-disk cache, the retry/backoff logic (honouring ``Retry-After`` when sent), the
hard cap of 8 workers with an adaptive (AIMD) concurrency limiter, the time-budgeted,
resumable, newest-first :func:`prefetch_cache`, and the ``source_resolution="H1"`` option,
which needs ~24x fewer requests than minute files for H1/H4/D1 research data.

The canonical bars built here are MID prices, ``mid_x = (bid_x + ask_x) / 2`` for each of
O/H/L/C. The mid high/low is a slight *outer bound* approximation: the bid high and the
ask high need not print on the same tick, so ``(bid_high + ask_high)/2`` can exceed the true
mid-path maximum by at most half the intrabar change in spread. The bar ``spread`` is the
average of the opening and closing quoted spreads ``0.5*((ask_o-bid_o)+(ask_c-bid_c))``,
floored at 0: a robust "typical" spread not dominated by one wide print, which is what an
execution model charging ``spread/2`` per fill wants. ``volume`` is the bid-side Dukascopy
volume (Dukascopy's own units; use it for relative activity only).

All timestamps are UTC; bars are labelled by their OPEN time and ``available_at`` is
``open + timeframe`` (SPEC §1). Coarser timeframes are produced with the leak-free
``aurum.data.resample.resample_bars``.
"""

from __future__ import annotations

import calendar as _cal
import datetime as dt
import logging
import lzma
import os
import random
import tempfile
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from aurum.core.timeframes import Timeframe, get_timeframe
from aurum.data.resample import resample_bars
from aurum.data.schema import make_bars

logger = logging.getLogger(__name__)

DATAFEED_ROOT = "https://datafeed.dukascopy.com/datafeed"
DATAFEED_URL = DATAFEED_ROOT + "/{symbol}/{year:04d}/{month0:02d}/{day:02d}/{side}_candles_min_1.bi5"
DATAFEED_HOUR_URL = DATAFEED_ROOT + "/{symbol}/{year:04d}/{month0:02d}/{side}_candles_hour_1.bi5"
SIDES = ("BID", "ASK")
_KINDS = {"min": "candles_min_1", "hour": "candles_hour_1"}

#: 24-byte big-endian candle record, O-C-L-H order (Dukascopy convention).
CANDLE_DTYPE = np.dtype(
    [
        ("t", ">u4"),
        ("open", ">u4"),
        ("close", ">u4"),
        ("low", ">u4"),
        ("high", ">u4"),
        ("volume", ">f4"),
    ]
)

#: Global pause (seconds, jittered) imposed on all workers after a throttled (503/429) reply.
THROTTLE_COOLDOWN_S = 5.0

#: Candidate power-of-ten point scales tried by :func:`detect_price_scale`.
CANDIDATE_SCALES: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


@dataclass(frozen=True)
class _ScaleHint:
    default: float      # scale Dukascopy uses for this instrument
    lo: float           # plausible price range (quote currency per unit)
    hi: float
    ref: float          # typical price, used to break ties between plausible scales


# Plausible ranges are wide (decades of history) but a wrong power of ten must still fall
# outside them *when the default scale is wrong*; the default is always tried first, which
# removes the ambiguity once gold trades above 2000 (x/1000 and x/10000 both "plausible").
_SCALE_HINTS: dict[str, _ScaleHint] = {
    "XAUUSD": _ScaleHint(1_000.0, 200.0, 20_000.0, 1_800.0),
    "XAGUSD": _ScaleHint(1_000.0, 3.0, 300.0, 22.0),
    "XPTUSD": _ScaleHint(1_000.0, 200.0, 10_000.0, 1_000.0),
    "XPDUSD": _ScaleHint(1_000.0, 100.0, 10_000.0, 1_200.0),
}


class _HTTPResponse(Protocol):  # minimal subset of requests.Response we rely on
    status_code: int
    content: bytes
    # ``headers`` (mapping) is used when present, e.g. to honour ``Retry-After``.


class _HTTPSession(Protocol):
    def get(self, url: str, timeout: float = ...) -> _HTTPResponse: ...  # pragma: no cover


class DukascopyError(RuntimeError):
    """Raised when a datafeed file cannot be fetched after all retries."""


# ------------------------------------------------------------------------------------
# pure decoding
# ------------------------------------------------------------------------------------
def _scale_hint(symbol: str) -> _ScaleHint:
    sym = symbol.upper()
    if sym in _SCALE_HINTS:
        return _SCALE_HINTS[sym]
    if sym.endswith("JPY"):
        return _ScaleHint(1_000.0, 20.0, 400.0, 110.0)
    # Generic FX major/minor: Dukascopy quotes 5 decimals.
    return _ScaleHint(100_000.0, 0.05, 50.0, 1.2)


def detect_price_scale(raw_prices: Iterable[float] | np.ndarray, symbol: str = "XAUUSD") -> float:
    """Infer the integer->price divisor from raw Dukascopy integer prices.

    The symbol's known Dukascopy scale is accepted if it maps the median raw price into the
    symbol's plausible range. Otherwise every power of ten in :data:`CANDIDATE_SCALES` whose
    implied price is plausible is considered and the one closest (in log distance) to the
    symbol's typical price wins (logged as a warning). Raises ``ValueError`` if nothing is
    plausible: then the file is not what we think it is and guessing would corrupt prices.
    """
    arr = np.asarray(raw_prices if isinstance(raw_prices, np.ndarray) else list(raw_prices), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError("cannot detect price scale from an empty / non-positive sample")
    med = float(np.median(arr))
    hint = _scale_hint(symbol)
    if hint.lo <= med / hint.default <= hint.hi:
        return hint.default
    plausible = [s for s in CANDIDATE_SCALES if hint.lo <= med / s <= hint.hi]
    if not plausible:
        raise ValueError(
            f"no candidate price scale maps median raw price {med:g} into the plausible range "
            f"[{hint.lo}, {hint.hi}] for {symbol}"
        )
    best = min(plausible, key=lambda s: abs(np.log(med / s / hint.ref)))
    logger.warning("Dukascopy %s: price scale auto-detected as %g (expected %g)", symbol, best, hint.default)
    return best


def decode_bi5_candles(
    raw: bytes,
    day: dt.date,
    price_scale: float | None,
    *,
    symbol: str = "XAUUSD",
    period_seconds: int = 86_400,
) -> pd.DataFrame:
    """Decode one ``*_candles_*.bi5`` file (pure function, no I/O).

    Parameters
    ----------
    raw : the file bytes exactly as served (LZMA-alone compressed). Empty bytes (Dukascopy
          serves 0-byte files for periods without data) give an empty frame.
    day : UTC start of the file's period (the day for minute files, the 1st of the month
          for hour files); record times are seconds from its midnight.
    price_scale : integer divisor (1000 for XAUUSD). ``None`` auto-detects with
          :func:`detect_price_scale` using ``symbol``.
    period_seconds : length of the file's period; records at/after it are rejected
          (86400 for daily minute files, ``days_in_month*86400`` for monthly hour files).

    Returns
    -------
    DataFrame indexed by the UTC candle OPEN time (``time``) with float columns
    ``open, high, low, close, volume`` — every record, including flat zero-volume filler
    candles (:func:`combine_bid_ask` removes those). Structurally invalid records (time
    outside the period, non-positive prices, high < low, O/C outside [L, H]) are dropped
    with a warning rather than propagated.
    """
    cols = ["open", "high", "low", "close", "volume"]
    if not raw:
        return _empty_frame(cols)
    payload = lzma.decompress(raw)  # FORMAT_AUTO handles the .lzma "alone" container
    rem = len(payload) % CANDLE_DTYPE.itemsize
    if rem:
        logger.warning(
            "Dukascopy %s: payload length %d not a multiple of %d; truncating trailing bytes",
            day, len(payload), CANDLE_DTYPE.itemsize,
        )
        payload = payload[: len(payload) - rem]
    rec = np.frombuffer(payload, dtype=CANDLE_DTYPE)
    if rec.size == 0:
        return _empty_frame(cols)

    t = rec["t"].astype(np.int64)
    o = rec["open"].astype(np.float64)
    c = rec["close"].astype(np.float64)
    lo = rec["low"].astype(np.float64)
    hi = rec["high"].astype(np.float64)
    vol = rec["volume"].astype(np.float64)

    scale = float(price_scale) if price_scale is not None else detect_price_scale(c, symbol)
    if not scale > 0:
        raise ValueError("price_scale must be positive")

    ok = (t >= 0) & (t < period_seconds) & (o > 0) & (c > 0) & (lo > 0) & (hi > 0)
    ok &= (hi >= lo) & (hi >= np.maximum(o, c)) & (lo <= np.minimum(o, c))
    ok &= np.isfinite(vol) & (vol >= 0)
    if not ok.all():
        logger.warning("Dukascopy %s: dropping %d malformed candle records", day, int((~ok).sum()))
    origin = pd.Timestamp(day)
    origin = origin.tz_localize("UTC") if origin.tz is None else origin.tz_convert("UTC")
    index = pd.DatetimeIndex(origin + pd.to_timedelta(t[ok], unit="s"), name="time")
    out = pd.DataFrame(
        {
            "open": o[ok] / scale,
            "high": hi[ok] / scale,
            "low": lo[ok] / scale,
            "close": c[ok] / scale,
            "volume": vol[ok],
        },
        index=index,
    )
    if not out.index.is_monotonic_increasing or out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _empty_frame(cols: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in cols}, index=idx)


def combine_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Merge decoded bid & ask candles into mid OHLC + spread for *active* candles only.

    A candle is active if either side printed a tick: non-zero volume or a non-degenerate
    range. Flat zero-volume filler candles (market closed) are dropped — they would create
    fake zero-return, zero-range bars that bias volatility and cost estimates downward.

    ``spread = max(0, ((ask_o - bid_o) + (ask_c - bid_c)) / 2)``; crossed quotes (negative
    spread, rare feed glitches) are floored at 0.
    """
    cols = ["open", "high", "low", "close", "volume", "spread"]
    j = bid.join(ask, how="inner", lsuffix="_bid", rsuffix="_ask")
    if j.empty:
        return _empty_frame(cols)
    active = (
        (j["volume_bid"].to_numpy() > 0)
        | (j["volume_ask"].to_numpy() > 0)
        | (j["high_bid"].to_numpy() > j["low_bid"].to_numpy())
        | (j["high_ask"].to_numpy() > j["low_ask"].to_numpy())
    )
    j = j.loc[active]
    data: dict[str, np.ndarray] = {}
    for f in ("open", "high", "low", "close"):
        data[f] = 0.5 * (j[f"{f}_bid"].to_numpy() + j[f"{f}_ask"].to_numpy())
    data["volume"] = j["volume_bid"].to_numpy()
    raw_spread = 0.5 * (
        (j["open_ask"].to_numpy() - j["open_bid"].to_numpy())
        + (j["close_ask"].to_numpy() - j["close_bid"].to_numpy())
    )
    n_neg = int((raw_spread < 0).sum())
    if n_neg:
        logger.debug("combine_bid_ask: %d crossed-quote candles floored to zero spread", n_neg)
    data["spread"] = np.maximum(raw_spread, 0.0)
    out = pd.DataFrame(data, index=j.index)
    out.index.name = "time"
    return out


# ------------------------------------------------------------------------------------
# fetching & caching
# ------------------------------------------------------------------------------------
def datafeed_url(symbol: str, day: dt.date, side: str, kind: str = "min") -> str:
    """URL of one candle file. ``kind="min"``: daily minute file; ``"hour"``: monthly hour file."""
    side = side.upper()
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if kind == "min":
        return DATAFEED_URL.format(symbol=symbol.upper(), year=day.year, month0=day.month - 1, day=day.day, side=side)
    if kind == "hour":
        return DATAFEED_HOUR_URL.format(symbol=symbol.upper(), year=day.year, month0=day.month - 1, side=side)
    raise ValueError(f"kind must be one of {sorted(_KINDS)}")


def cache_path(cache_dir: str | Path, symbol: str, day: dt.date, side: str, kind: str = "min") -> Path:
    """On-disk location mirroring the datafeed layout (zero-based month)."""
    base = Path(cache_dir) / symbol.upper() / f"{day.year:04d}" / f"{day.month - 1:02d}"
    if kind == "min":
        return base / f"{day.day:02d}" / f"{side.upper()}_candles_min_1.bi5"
    if kind == "hour":
        return base / f"{side.upper()}_candles_hour_1.bi5"
    raise ValueError(f"kind must be one of {sorted(_KINDS)}")


_thread_local = threading.local()


def _default_session(pool: int) -> Any:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        import requests
        from requests.adapters import HTTPAdapter

        sess = requests.Session()
        sess.headers["User-Agent"] = "Mozilla/5.0 (compatible; aurum-research/2.0)"
        adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
        sess.mount("https://", adapter)
        _thread_local.session = sess
    return sess


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _period_end(day: dt.date, kind: str) -> dt.date:
    if kind == "hour":
        return dt.date(day.year, day.month, _cal.monthrange(day.year, day.month)[1])
    return day


class AdaptiveLimiter:
    """Shared AIMD concurrency limiter for a rate-limited HTTP endpoint.

    The free Dukascopy feed rejects excess concurrent requests with 503/429. Workers call
    :meth:`acquire` before each request and :meth:`release` after it. A throttled response
    halves the allowed concurrency (multiplicative decrease) and imposes a global cool-down
    that doubles with every consecutive throttled reply (``cooldown * 2**k``, capped at
    ``max_cooldown``, jittered) so a sustained block is probed ever more rarely instead of
    being prolonged; every ``4 * limit`` consecutive successes raise the limit by one
    (additive increase), up to ``max_concurrency``. This converges to whatever the server
    tolerates without manual tuning and without hammering it.
    """

    def __init__(self, max_concurrency: int = 4, cooldown: float = 5.0, max_cooldown: float = 300.0) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.limit = self.max_concurrency
        self.cooldown = float(cooldown)
        self.max_cooldown = float(max_cooldown)
        self._active = 0
        self._ok_streak = 0
        self._throttle_streak = 0
        self._resume_at = 0.0
        self._cond = threading.Condition()

    def acquire(self, deadline: float | None = None) -> bool:
        """Block until a request slot is free; ``False`` if ``deadline`` (monotonic) passes first."""
        with self._cond:
            while True:
                now = time.monotonic()
                wait = self._resume_at - now
                if self._active < self.limit and wait <= 0:
                    self._active += 1
                    return True
                if deadline is not None and now >= deadline:
                    return False
                timeout = max(wait, 0.05) if wait > 0 else 0.5
                if deadline is not None:
                    timeout = min(timeout, max(deadline - now, 0.01))
                self._cond.wait(timeout=timeout)

    def release(self, *, throttled: bool) -> None:
        with self._cond:
            self._active -= 1
            if throttled:
                self.limit = max(1, self.limit // 2)
                self._ok_streak = 0
                pause = min(self.max_cooldown, self.cooldown * 2 ** min(self._throttle_streak, 16))
                self._throttle_streak += 1
                self._resume_at = max(self._resume_at, time.monotonic() + pause * (0.5 + random.random()))
            else:
                self._throttle_streak = 0
                self._ok_streak += 1
                if self._ok_streak >= 4 * self.limit and self.limit < self.max_concurrency:
                    self.limit += 1
                    self._ok_streak = 0
            self._cond.notify_all()


def _valid_payload(body: bytes) -> bool:
    """True for an empty body or a complete LZMA stream of whole 24-byte candle records.

    A connection dropped mid-transfer can still surface as HTTP 200 with a short body; caching
    it would silently corrupt every later offline build, so it is rejected and re-fetched.
    """
    if not body:
        return True
    try:
        payload = lzma.decompress(body)
    except (lzma.LZMAError, EOFError):
        return False
    return len(payload) % CANDLE_DTYPE.itemsize == 0


def _retry_after_seconds(resp: Any) -> float | None:
    """Seconds requested by a ``Retry-After`` header (delta-seconds form only), if any."""
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("Retry-After")
    except Exception:  # pragma: no cover - exotic header containers
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None  # HTTP-date form: fall back to our own backoff


def fetch_file(
    symbol: str,
    day: dt.date,
    side: str,
    *,
    kind: str = "min",
    cache_dir: str | Path | None = "cache/dukascopy",
    session: _HTTPSession | None = None,
    retries: int = 6,
    backoff: float = 1.0,
    timeout: float = 90.0,
    today: dt.date | None = None,
    limiter: AdaptiveLimiter | None = None,
    throttle_retries: int = 40,
    offline: bool = False,
    deadline: float | None = None,
) -> bytes:
    """Return the raw bytes of one candle file, using/filling the on-disk cache.

    * HTTP 200 → bytes (possibly empty) are cached atomically.
    * HTTP 404 → treated as "no data" (empty bytes).
    * HTTP 503/429 (throttling) → back off (``Retry-After`` if sent, capped at 5 min; via
      ``limiter`` if given) and retry, up to ``throttle_retries`` times; other errors →
      exponential backoff with jitter, up to ``retries`` times; then ``DukascopyError``.

    Empty results for periods ending within 3 days of ``today`` are NOT cached: Dukascopy
    publishes a period's file only after it is complete, so "empty" may mean "not yet".
    ``offline=True`` never touches the network: a cache miss raises ``DukascopyError``.
    ``deadline`` (a ``time.monotonic()`` instant) makes the call give up instead of sleeping
    past it — used by time-budgeted bulk fetches.
    """
    today = today or dt.datetime.now(dt.timezone.utc).date()
    path = cache_path(cache_dir, symbol, day, side, kind) if cache_dir is not None else None
    if path is not None and path.exists():
        return path.read_bytes()
    url = datafeed_url(symbol, day, side, kind)
    if offline:
        raise DukascopyError(f"offline and not cached: {url}")
    sess = session if session is not None else _default_session(8)
    last_err: Exception | None = None
    data: bytes | None = None
    n_err = n_throttle = 0
    while data is None:
        throttled = False
        retry_after: float | None = None
        if limiter is not None and not limiter.acquire(deadline=deadline):
            last_err = DukascopyError(f"time budget exhausted waiting for the rate limiter ({url})")
            break
        try:
            resp = sess.get(url, timeout=timeout)
            status = int(resp.status_code)
            if status == 200:
                body = bytes(resp.content)
                if _valid_payload(body):
                    data = body
                else:  # truncated / corrupt transfer: never cache it, retry
                    last_err = DukascopyError(f"corrupt LZMA payload ({len(body)} bytes) for {url}")
            elif status == 404:
                data = b""
            else:
                throttled = status in (429, 503)
                retry_after = _retry_after_seconds(resp) if throttled else None
                last_err = DukascopyError(f"HTTP {status} for {url}")
        except Exception as exc:  # network errors from requests/urllib3
            last_err = exc
        finally:
            if limiter is not None:
                limiter.release(throttled=throttled)
        if data is not None:
            break
        if throttled:
            n_throttle += 1
            if n_throttle > throttle_retries:
                break
            sleep = min(30.0, backoff * (1 + n_throttle)) * (0.5 + random.random())
            if retry_after is not None:
                sleep = max(sleep, min(retry_after, 300.0))
        else:
            n_err += 1
            if n_err > retries:
                break
            sleep = min(60.0, backoff * (2 ** (n_err - 1))) * (0.5 + random.random())
        if deadline is not None and time.monotonic() + sleep > deadline:
            last_err = DukascopyError(f"time budget exhausted while retrying {url} (last: {last_err})")
            break
        logger.debug("retry for %s in %.2fs (%s)", url, sleep, last_err)
        time.sleep(sleep)
    if data is None:
        raise DukascopyError(
            f"failed to fetch {url} ({n_err} errors, {n_throttle} throttled): {last_err}"
        ) from last_err

    fresh = (today - _period_end(day, kind)).days < 3
    if path is not None and not (fresh and not data):
        _atomic_write(path, data)
    return data


def _fetch_many(
    symbol: str,
    tasks: list[tuple[dt.date, str, str]],
    *,
    cache_dir: str | Path | None,
    session: _HTTPSession | None,
    workers: int,
    retries: int,
    today: dt.date,
    passes: int = 3,
    pass_pause: float = 20.0,
    offline: bool = False,
) -> dict[tuple[dt.date, str, str], bytes]:
    """Fetch many files with a shared adaptive limiter; failed files are retried in later
    passes (after a pause) instead of aborting the whole download on the first failure."""
    limiter = AdaptiveLimiter(max_concurrency=workers, cooldown=THROTTLE_COOLDOWN_S)
    out: dict[tuple[dt.date, str, str], bytes] = {}
    pending = list(tasks)
    errors: dict[tuple[dt.date, str, str], Exception] = {}
    progress = {"n": 0}
    lock = threading.Lock()

    def _job(task: tuple[dt.date, str, str]) -> None:
        d, k, s = task
        sess = session if session is not None else _default_session(workers)
        try:
            out[task] = fetch_file(
                symbol, d, s, kind=k, cache_dir=cache_dir, session=sess, retries=retries,
                today=today, limiter=limiter, offline=offline,
            )
            errors.pop(task, None)
        except DukascopyError as exc:
            errors[task] = exc
        with lock:
            progress["n"] += 1
            if progress["n"] % 200 == 0:
                logger.info("Dukascopy %s: %d/%d files (limit=%d)", symbol, progress["n"], len(tasks), limiter.limit)

    for p in range(1 if offline else passes):
        if not pending:
            break
        if p:
            logger.warning("Dukascopy %s: pass %d retrying %d failed files", symbol, p + 1, len(pending))
            time.sleep(pass_pause)
        if workers == 1 or len(pending) <= 2:
            for t in pending:
                _job(t)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dukascopy") as ex:
                list(ex.map(_job, pending))
        pending = [t for t in pending if t not in out]
    if pending:
        example = errors.get(pending[0])
        raise DukascopyError(f"{len(pending)} Dukascopy files could not be fetched, e.g. {example}")
    return out


def fetch_day_file(symbol: str, day: dt.date, side: str, **kwargs: Any) -> bytes:
    """Backward-compatible alias of :func:`fetch_file` for daily minute files."""
    return fetch_file(symbol, day, side, kind="min", **kwargs)


def _to_utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _drop_truncated_head(bars: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Drop aggregated bars whose bucket OPENS before ``start``.

    The requested range cuts such a bucket: its data before ``start`` was never loaded, so
    its open/high/low/volume describe only the tail of the period while ``resample_bars``
    (which cannot tell "not requested" from "market closed") emits it as complete. Example:
    ``start="2024-01-10"`` with 22:00-anchored D1 bars would produce a 2024-01-09 22:00 bar
    missing its first two hours.
    """
    head = bars.index < start
    if head.any():
        logger.debug("dropping %d aggregated bar(s) that start before %s", int(head.sum()), start)
        bars = bars.loc[~head]
    return bars


def _resolve_range(start: Any, end: Any, today: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start is None:
        raise ValueError("start is required")
    start_ts = _to_utc(start)
    if end is None:
        end_excl = pd.Timestamp(today, tz="UTC")
    else:
        end_ts = _to_utc(end)
        # a bare date is an inclusive calendar day; a timestamp with a time is exclusive
        end_excl = end_ts + pd.Timedelta(days=1) if end_ts == end_ts.normalize() else end_ts
    if end_excl <= start_ts:
        raise ValueError(f"empty date range: start={start_ts} end={end}")
    return start_ts, end_excl


def _plan_periods(
    start: pd.Timestamp, end_excl: pd.Timestamp, resolution: str, skip_saturday: bool, today: dt.date
) -> list[tuple[dt.date, str]]:
    """List of (period_start, kind) files to fetch.

    ``resolution="M1"``: one minute file per day. ``"H1"``: one hour file per *complete*
    month (ended >= 3 days before ``today``); the trailing incomplete month falls back to
    minute files so the most recent days are still covered.
    """
    last_day = (end_excl - pd.Timedelta(microseconds=1)).date()
    first_day = start.date()
    plan: list[tuple[dt.date, str]] = []

    def add_days(d0: dt.date, d1: dt.date) -> None:
        d = d0
        while d <= d1:
            if d < today and not (skip_saturday and d.weekday() == 5):
                plan.append((d, "min"))
            d += dt.timedelta(days=1)

    if resolution == "M1":
        add_days(first_day, last_day)
        return plan
    if resolution != "H1":
        raise ValueError("source_resolution must be 'M1' or 'H1'")
    m = dt.date(first_day.year, first_day.month, 1)
    while m <= last_day:
        m_end = _period_end(m, "hour")
        if (today - m_end).days >= 3:
            plan.append((m, "hour"))
        else:
            add_days(max(m, first_day), min(m_end, last_day))
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return plan


def download_dukascopy(
    symbol: str = "XAUUSD",
    start: Any = None,
    end: Any = None,
    timeframe: str | Timeframe = "M1",
    *,
    cache_dir: str | Path | None = "cache/dukascopy",
    price_scale: float | None = None,
    max_workers: int = 8,
    session: _HTTPSession | None = None,
    source_resolution: str = "M1",
    skip_saturday: bool = True,
    daily_anchor_hour_utc: int = 0,
    retries: int = 6,
    today: dt.date | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Download (or load from cache) Dukascopy bid/ask candles and build canonical bars.

    Parameters
    ----------
    start, end : date-like, interpreted in UTC. A bare date for ``end`` is *inclusive* (the
        whole day); a timestamp with a time component is an exclusive bound on bar open time.
        ``end=None`` means "up to the last complete UTC day".
    timeframe : output timeframe; anything coarser than the source resolution is aggregated
        with ``resample_bars`` (trailing incomplete buckets are dropped — see there).
    price_scale : integer->price divisor; ``None`` auto-detects once from the data (logged).
    max_workers : concurrent HTTP requests, capped at 8 (the free feed throttles; 4 is the
        practical sweet spot — more just produces 503s that are retried with backoff).
    source_resolution : ``"M1"`` (SPEC default: daily minute files) or ``"H1"`` (monthly
        hour files — ~24x fewer requests, for H1/H4/D1 research). ``timeframe`` must not be
        finer than the source resolution.
    skip_saturday : Saturday-UTC minute files only contain flat filler candles for
        metals/FX; skipping them saves ~14% of requests. Set False for 7-day markets.
    daily_anchor_hour_utc : forwarded to ``resample_bars`` for H4/D1 buckets.
    offline : use the on-disk cache only (a missing file raises ``DukascopyError``).

    Returns canonical bars (``aurum.data.schema``) with ``attrs`` ``source``, ``symbol``,
    ``price_scale`` and ``source_resolution``.
    """
    tf = get_timeframe(timeframe)
    src_tf = get_timeframe(source_resolution)
    if tf.minutes < src_tf.minutes:
        raise ValueError(f"timeframe {tf.name} is finer than source_resolution {src_tf.name}")
    today = today or dt.datetime.now(dt.timezone.utc).date()
    start_ts, end_excl = _resolve_range(start, end, today)
    plan = _plan_periods(start_ts, end_excl, src_tf.name, skip_saturday, today)
    workers = max(1, min(int(max_workers), 8))
    logger.info(
        "Dukascopy %s: %d periods (%s) %s..%s with %d workers",
        symbol, len(plan), src_tf.name, start_ts, end_excl, workers,
    )
    tasks = [(d, k, s) for d, k in plan for s in SIDES]
    t0 = time.monotonic()
    raw = _fetch_many(symbol, tasks, cache_dir=cache_dir, session=session, workers=workers,
                      retries=retries, today=today, offline=offline)
    logger.info("Dukascopy %s: fetched %d files in %.1fs", symbol, len(tasks), time.monotonic() - t0)

    # Decode at scale 1 (raw integer points) so the scale is detected once, globally.
    frames: dict[str, list[pd.DataFrame]] = {"min": [], "hour": []}
    scale = price_scale
    n_empty = 0
    for d, k in plan:
        bid_raw, ask_raw = raw[(d, k, "BID")], raw[(d, k, "ASK")]
        if not bid_raw or not ask_raw:
            n_empty += 1
            continue
        period = 86_400 if k == "min" else _cal.monthrange(d.year, d.month)[1] * 86_400
        bid = decode_bi5_candles(bid_raw, d, 1.0, period_seconds=period)
        ask = decode_bi5_candles(ask_raw, d, 1.0, period_seconds=period)
        frame = combine_bid_ask(bid, ask)
        if frame.empty:
            n_empty += 1
            continue
        if scale is None:
            scale = detect_price_scale(frame["close"].to_numpy(), symbol)
            logger.info("Dukascopy %s: using price scale %g", symbol, scale)
        frames[k].append(frame)
    if not frames["min"] and not frames["hour"]:
        raise DukascopyError(f"no Dukascopy data for {symbol} in [{start_ts}, {end_excl})")
    assert scale is not None

    def _finish(parts: list[pd.DataFrame], tf_name: str) -> pd.DataFrame | None:
        if not parts:
            return None
        df = pd.concat(parts)
        for c in ("open", "high", "low", "close", "spread"):
            df[c] = df[c].to_numpy() / scale
        df = df.loc[(df.index >= start_ts) & (df.index < end_excl)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return make_bars(df, tf_name) if len(df) else None

    # Whole past UTC days are final, so buckets ending by min(end, today 00:00) are complete
    # even when the market closed early inside them (e.g. the Friday of the last week).
    complete_until = min(end_excl, pd.Timestamp(today, tz="UTC"))
    m1 = _finish(frames["min"], "M1")
    h1 = _finish(frames["hour"], "H1")
    logger.info(
        "Dukascopy %s: %d M1 + %d H1 active bars (%d empty/closed periods skipped)",
        symbol, 0 if m1 is None else len(m1), 0 if h1 is None else len(h1), n_empty,
    )
    empty_msg = f"no Dukascopy bars for {symbol} in [{start_ts}, {end_excl})"
    if src_tf.name == "M1":
        if m1 is None:
            raise DukascopyError(empty_msg)
        bars = m1 if tf.name == "M1" else _drop_truncated_head(resample_bars(
            m1, tf, daily_anchor_hour_utc=daily_anchor_hour_utc, complete_until=complete_until
        ), start_ts)
    else:
        parts = [h1] if h1 is not None else []
        if m1 is not None:  # trailing incomplete month came from minute files
            m1_h1 = resample_bars(m1, "H1", complete_until=complete_until)
            if h1 is not None:
                m1_h1 = m1_h1.loc[m1_h1.index > h1.index[-1]]
            parts.append(m1_h1)
        parts = [p for p in parts if len(p)]
        if not parts:
            raise DukascopyError(empty_msg)
        bars = pd.concat(parts) if len(parts) > 1 else parts[0]
        bars = _drop_truncated_head(make_bars(bars.drop(columns=["available_at"]), "H1"), start_ts)
        if tf.name != "H1":
            bars = _drop_truncated_head(resample_bars(
                bars, tf, daily_anchor_hour_utc=daily_anchor_hour_utc, complete_until=complete_until
            ), start_ts)
    bars.attrs.update(
        {
            "source": "dukascopy",
            "symbol": symbol.upper(),
            "price_scale": float(scale),
            "source_resolution": src_tf.name,
        }
    )
    return bars


# ------------------------------------------------------------------------------------
# research dataset
# ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PrefetchReport:
    """Outcome of :func:`prefetch_cache` (file counts are BID+ASK files, not days)."""

    planned: int                 # files in the requested range
    already_cached: int          # of which were on disk before the call
    fetched: int                 # downloaded (or confirmed empty) during the call
    failed: int                  # gave up after retries / throttling
    not_attempted: int           # skipped because the time budget ran out
    elapsed_s: float
    contiguous_start: dt.date | None   # earliest day of the fully-cached run ending at ``end``
    stopped_early: bool


def prefetch_cache(
    symbol: str = "XAUUSD",
    start: Any = None,
    end: Any = None,
    *,
    cache_dir: str | Path = "cache/dukascopy",
    source_resolution: str = "M1",
    max_workers: int = 8,
    time_budget_s: float | None = None,
    newest_first: bool = True,
    skip_saturday: bool = True,
    session: _HTTPSession | None = None,
    today: dt.date | None = None,
    retries: int = 6,
) -> PrefetchReport:
    """Fill the on-disk cache (no decoding) within an optional wall-clock budget.

    Designed for the throttled free feed: files are fetched **newest first** so that, if the
    budget runs out (or the feed starts answering 429), the fully-cached range still grows
    contiguously backward from ``end`` — exactly what :func:`build_dataset` consumes offline
    via :func:`contiguous_cached_start`. The call is resumable (cached files are skipped),
    never raises on individual failures (they are counted and can be retried later), and
    shares one :class:`AdaptiveLimiter` across workers so concurrency shrinks under
    throttling instead of hammering the server.
    """
    today = today or dt.datetime.now(dt.timezone.utc).date()
    start_ts, end_excl = _resolve_range(start, end, today)
    src = get_timeframe(source_resolution).name
    plan = _plan_periods(start_ts, end_excl, src, skip_saturday, today)
    if newest_first:
        plan = plan[::-1]
    tasks = [(d, k, s) for d, k in plan for s in SIDES]
    todo = [t for t in tasks if not cache_path(cache_dir, symbol, t[0], t[2], t[1]).exists()]
    workers = max(1, min(int(max_workers), 8))
    limiter = AdaptiveLimiter(max_concurrency=workers, cooldown=THROTTLE_COOLDOWN_S)
    t0 = time.monotonic()
    deadline = t0 + float(time_budget_s) if time_budget_s is not None else None
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    lock = threading.Lock()
    logger.info("Dukascopy prefetch %s: %d files planned, %d to fetch, %d workers, budget=%s s",
                symbol, len(tasks), len(todo), workers, time_budget_s)

    def _job(task: tuple[dt.date, str, str]) -> None:
        d, k, s = task
        if deadline is not None and time.monotonic() >= deadline:
            outcome = "skipped"
        else:
            sess = session if session is not None else _default_session(workers)
            try:
                fetch_file(symbol, d, s, kind=k, cache_dir=cache_dir, session=sess, retries=retries,
                           today=today, limiter=limiter, deadline=deadline)
                outcome = "ok"
            except DukascopyError as exc:
                logger.debug("prefetch failed for %s %s %s: %s", d, k, s, exc)
                outcome = "failed"
        with lock:
            counts[outcome] += 1
            done = counts["ok"] + counts["failed"]
            if outcome != "skipped" and done % 100 == 0:
                logger.info("Dukascopy prefetch %s: %d/%d files (%d failed, limit=%d, %.0fs, at %s)",
                            symbol, done, len(todo), counts["failed"], limiter.limit, time.monotonic() - t0, d)

    if workers == 1 or len(todo) <= 2:
        for t in todo:
            _job(t)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dukascopy-prefetch") as ex:
            list(ex.map(_job, todo))
    last_day = (end_excl - pd.Timedelta(microseconds=1)).date()
    last_day = min(last_day, today - dt.timedelta(days=1))
    earliest = start_ts.date()
    contiguous = (
        contiguous_cached_start(symbol, cache_dir, last_day, skip_saturday=skip_saturday, earliest=earliest)
        if src == "M1" else None
    )
    report = PrefetchReport(
        planned=len(tasks), already_cached=len(tasks) - len(todo), fetched=counts["ok"],
        failed=counts["failed"], not_attempted=counts["skipped"], elapsed_s=time.monotonic() - t0,
        contiguous_start=contiguous,
        stopped_early=counts["skipped"] > 0 or (deadline is not None and time.monotonic() >= deadline),
    )
    logger.info("Dukascopy prefetch %s done: %s", symbol, report)
    return report


def contiguous_cached_start(
    symbol: str,
    cache_dir: str | Path,
    end: dt.date,
    *,
    skip_saturday: bool = True,
    earliest: dt.date = dt.date(2003, 5, 5),
) -> dt.date | None:
    """Earliest day ``d`` such that BID and ASK minute files for every day in ``[d, end]``
    (Saturdays excluded) are in the cache — i.e. the range an ``offline`` build can cover."""
    first: dt.date | None = None
    d = end
    while d >= earliest:
        if not (skip_saturday and d.weekday() == 5):
            if not all(cache_path(cache_dir, symbol, d, s, "min").exists() for s in SIDES):
                break
            first = d
        d -= dt.timedelta(days=1)
    return first


def build_dataset(
    out_dir: str | Path,
    *,
    symbol: str = "XAUUSD",
    start: Any = "2012-01-01",
    end: Any = "2026-09-25",
    cache_dir: str | Path = "cache/dukascopy",
    m15_start: Any = None,
    max_workers: int = 4,
    offline: bool = False,
    ny_close_anchor_hour: int = 22,
    h1_source: str = "auto",
) -> dict[str, Any]:
    """Build the canonical research bar files from Dukascopy and write a manifest.

    Files: ``{sym}_M15``, ``{sym}_H1``, ``{sym}_H4``, ``{sym}_D1`` (SPEC default UTC-midnight
    anchor, so Sunday-evening sessions form short "Sunday" bars) and ``{sym}_D1_nyclose``
    (anchored at ``ny_close_anchor_hour`` UTC: one bar per trading session, no stubs).

    Two build paths, chosen by ``h1_source``:

    * ``"M1"`` — the SPEC §3.2 path: every file is aggregated *directly* from the M1 mid bars
      of the daily minute files over the whole ``[start, end]``.
    * ``"H1"`` — H1 from monthly hour files (~24x fewer requests on the throttled free
      feed), H4/D1 resampled from it, and M15 from minute files over ``[m15_start, end]``
      (default: the contiguous fully-cached run ending at ``end``). Hour-file mid highs/lows
      are ``(bid_high + ask_high)/2`` over the whole hour, a slightly looser outer bound than
      the minute-level construction (observed: O/C identical, H/L wider by <= ~$2.5 on ~5%
      of 2026 bars); spreads are the hour's open/close quoted spread instead of the mean of
      minute spreads (same median).
    * ``"auto"`` (default) — ``"M1"`` if the minute-file cache already covers the whole range,
      else ``"H1"``. It never triggers a full minute download by itself: fill the cache with
      :func:`prefetch_cache` first.

    Returns ``{"files": {...}, "quality": {...}, ...}`` and writes ``manifest.json`` (row
    counts, ranges, content hashes, sources, data-quality report) next to the parquet files.
    """
    import json

    from aurum.data.loaders import quality_report
    from aurum.data.store import frame_hash, save_bars

    if h1_source not in ("auto", "M1", "H1"):
        raise ValueError("h1_source must be 'auto', 'M1' or 'H1'")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sym = symbol.lower()
    today = dt.datetime.now(dt.timezone.utc).date()
    start_ts, end_excl = _resolve_range(start, end, today)
    cu = min(end_excl, pd.Timestamp(today, tz="UTC"))
    end_day = min((end_excl - pd.Timedelta(microseconds=1)).date(), today - dt.timedelta(days=1))
    start_day = start_ts.date()
    first_needed = start_day + dt.timedelta(days=1) if start_day.weekday() == 5 else start_day
    m1_first = contiguous_cached_start(symbol, cache_dir, end_day, earliest=start_day)
    m1_complete = m1_first is not None and m1_first <= first_needed
    source = "M1" if h1_source == "M1" or (h1_source == "auto" and m1_complete) else "H1"
    logger.info("build_dataset %s: minute cache contiguous from %s (complete=%s) -> %s path",
                symbol, m1_first, m1_complete, source)
    manifest: dict[str, Any] = {
        "symbol": symbol.upper(), "source": "dukascopy datafeed", "build_path": source,
        "minute_cache_contiguous_from": str(m1_first) if m1_first else None,
        "files": {}, "quality": {}, "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    def _save(bars: pd.DataFrame, name: str, note: str, parent: pd.DataFrame, anchor: int | None = None) -> None:
        bars.attrs.update({k: v for k, v in parent.attrs.items() if k != "timeframe"})
        if anchor is not None:
            bars.attrs["daily_anchor_hour_utc"] = anchor
        path = save_bars(bars, out / f"{name}.parquet", metadata={"note": note})
        manifest["files"][name] = {
            "path": path.name, "timeframe": bars.attrs.get("timeframe"), "rows": len(bars),
            "start": str(bars.index[0]), "end": str(bars.index[-1]), "sha256": frame_hash(bars), "note": note,
        }
        manifest["quality"][name] = quality_report(bars)

    daily = (("H4", 0, ""), ("D1", 0, ""), ("D1", ny_close_anchor_hour, "_nyclose"))
    if source == "M1":
        m1 = download_dukascopy(symbol, start, end, "M1", cache_dir=cache_dir, source_resolution="M1",
                                max_workers=max_workers, offline=offline)
        note = "Dukascopy daily minute candles (bid/ask -> mid) aggregated from M1"
        for tf in ("M15", "H1"):
            _save(_drop_truncated_head(resample_bars(m1, tf, complete_until=cu), start_ts), f"{sym}_{tf}", note, m1)
        for tf, anchor, suffix in daily:
            bars = _drop_truncated_head(resample_bars(m1, tf, daily_anchor_hour_utc=anchor, complete_until=cu),
                                        start_ts)
            _save(bars, f"{sym}_{tf}{suffix}", f"{note}, daily_anchor_hour_utc={anchor}", m1, anchor)
        del m1
    else:
        h1 = download_dukascopy(symbol, start, end, "H1", cache_dir=cache_dir, source_resolution="H1",
                                max_workers=max_workers, offline=offline)
        _save(h1, f"{sym}_H1", "Dukascopy monthly hour candles (bid/ask -> mid); trailing month from minute files",
              h1)
        for tf, anchor, suffix in daily:
            bars = _drop_truncated_head(resample_bars(h1, tf, daily_anchor_hour_utc=anchor, complete_until=cu),
                                        start_ts)
            _save(bars, f"{sym}_{tf}{suffix}", f"resampled from {sym}_H1 with daily_anchor_hour_utc={anchor}",
                  h1, anchor)
        if m15_start is None:
            m15_start = m1_first
        if m15_start is not None:
            m15 = download_dukascopy(symbol, m15_start, end, "M15", cache_dir=cache_dir, source_resolution="M1",
                                     max_workers=max_workers, offline=offline)
            _save(m15, f"{sym}_M15", "Dukascopy daily minute candles (bid/ask -> mid) resampled to M15", m15)
        else:
            logger.warning("no cached minute data ending at %s: M15 file not built", end)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


__all__ = [
    "AdaptiveLimiter",
    "CANDLE_DTYPE",
    "DATAFEED_HOUR_URL",
    "DATAFEED_URL",
    "DukascopyError",
    "PrefetchReport",
    "build_dataset",
    "cache_path",
    "combine_bid_ask",
    "contiguous_cached_start",
    "datafeed_url",
    "decode_bi5_candles",
    "detect_price_scale",
    "download_dukascopy",
    "fetch_day_file",
    "fetch_file",
    "prefetch_cache",
]
