"""Breakout strategies: ``vol_squeeze`` and ``orb`` (session opening-range breakout).

Economic rationale
------------------
Volatility clusters (Mandelbrot 1963; Engle 1982) and is mean-reverting: periods of
unusually low realised range are followed by range expansion. When compressed ranges
resolve, the first directional move tends to carry because resting stop orders and
option-hedging flows cluster just outside the range and are triggered together
(Osler 2003, 2005 documents stop-loss clustering and the resulting price cascades in FX).
Session opens are a special case: in a 23-hour market like gold, liquidity and
information arrival are concentrated at the London open/fixes and the New York
open/US data releases, so the range formed right after an open is where overnight order
flow gets priced; a decisive break of it often signals the session's direction
(Crabel 1990; Zarattini & Aziz 2023 document ORB profitability in equity index futures).

Both rules are stateful loops over bars (~20 ms per 100k bars), evaluated at each bar's
close with data up to that close only.

References
----------
* Osler, C. (2003). "Currency Orders and Exchange Rate Dynamics: An Explanation for the
  Predictive Success of Technical Analysis". J. Finance 58(5); Osler (2005). "Stop-loss
  orders and price cascades in currency markets". J. Int. Money & Finance 24(2).
* Crabel, T. (1990). *Day Trading with Short Term Price Patterns and Opening Range
  Breakout*. Traders Press.
* Carter, J. (2005/2012). *Mastering the Trade*, ch. 11 ("the squeeze"). McGraw-Hill.
* Zarattini, C. & Aziz, A. (2023). "Can Day Trading Really Be Profitable?". SSRN 4416622.
* Engle, R. (1982). "Autoregressive Conditional Heteroscedasticity". Econometrica 50(4).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.volatility import atr, bar_minutes
from aurum.strategies.base import Strategy, register_strategy
from aurum.strategies.trend import (
    Z_FORECAST_SCALAR,
    bar_volatility,
    check_params,
    log_close,
    momentum_z,
)

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_SESSIONS", "OpeningRangeBreakout", "VolatilitySqueeze"]


# ---------------------------------------------------------------------------------------
# vol_squeeze
# ---------------------------------------------------------------------------------------
@register_strategy
class VolatilitySqueeze(Strategy):
    """Bollinger-inside-Keltner "squeeze" release, traded in the direction of momentum.

    Rationale: Bollinger bands (``SMA ± bb_k·sd``) contracting inside Keltner channels
    (``EMA ± kc_mult·ATR``) mean close-to-close dispersion is unusually low relative to the
    bar ranges — a volatility compression. Volatility mean-reverts, and the expansion that
    ends a squeeze tends to be directional as clustered stops are run (Carter 2005;
    Osler 2005). The trade is taken in the direction of the prevailing momentum when the
    squeeze releases.

    Rules: the squeeze is ON when both Bollinger bands lie inside the Keltner channel. When
    it turns OFF after having been on for at least ``min_squeeze`` bars, enter in the
    direction of the vol-normalised ``mom_n``-bar momentum ``z``. Hold for at most
    ``hold`` bars; exit early if momentum flips sign. Forecast while in a trade:
    ``sign * min(1, 0.627 |z|)`` (Carver scaling of an ~N(0,1) statistic). Defaults (H1):
    20-bar bands, bb_k 2.0, kc_mult 1.5, ATR(20), squeeze ≥ 6 bars, 20-bar momentum,
    hold ≤ 36 bars.

    References: Carter (2005) *Mastering the Trade*, ch. 11; Osler (2005) JIMF 24(2);
    Bollinger (2001).
    """

    name = "vol_squeeze"
    description = ("TTM-style squeeze: when Bollinger(20,2) exits Keltner(20,1.5 ATR) after a "
                   "compression, trade the 20-bar momentum direction for up to 36 bars.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n": 20, "bb_k": 2.0, "kc_mult": 1.5, "atr_n": 20, "min_squeeze": 6,
                "mom_n": 20, "hold": 36, "vol_halflife": 120.0, "vol_min_periods": 60}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        for k in ("n", "atr_n", "min_squeeze", "mom_n", "hold", "vol_min_periods"):
            p[k] = int(p[k])
        if p["n"] < 2 or p["atr_n"] < 1 or p["mom_n"] < 1 or p["hold"] < 1 or p["min_squeeze"] < 1:
            raise ValueError("window parameters must be positive (n >= 2)")
        if not (float(p["bb_k"]) > 0 and float(p["kc_mult"]) > 0):
            raise ValueError("bb_k and kc_mult must be positive")

    @property
    def warmup_bars(self) -> int:
        p = self.params
        return max(p["n"], p["atr_n"], p["mom_n"], p["vol_min_periods"]) + p["min_squeeze"] + 1

    def squeeze_on(self, bars: pd.DataFrame) -> np.ndarray:
        """Boolean array: Bollinger bands strictly inside the Keltner channel at each close."""
        p = self.params
        close = bars["close"].astype(float)
        roll = close.rolling(p["n"], min_periods=p["n"])
        mid, sd = roll.mean(), roll.std(ddof=0)
        ema = close.ewm(span=p["n"], adjust=False, min_periods=p["n"]).mean()
        width = float(p["kc_mult"]) * atr(bars, p["atr_n"])
        bb_k = float(p["bb_k"])
        inside = ((mid + bb_k * sd) < (ema + width)) & ((mid - bb_k * sd) > (ema - width))
        return inside.to_numpy(dtype=bool)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        p = self.params
        bars = md.bars
        on = self.squeeze_on(bars).tolist()
        lc = log_close(bars)
        mom = momentum_z(lc, bar_volatility(lc, p["vol_halflife"], p["vol_min_periods"]),
                         p["mom_n"]).tolist()
        out = [0.0] * len(on)
        state, held, run = 0, 0, 0
        hold, min_sq = p["hold"], p["min_squeeze"]
        for t, is_on in enumerate(on):
            z = mom[t]
            valid = z == z  # not NaN
            if state != 0:
                held += 1
                if held >= hold or not valid or z * state < 0:
                    state = 0
            if not is_on and run >= min_sq and valid and z != 0.0:
                state, held = (1 if z > 0 else -1), 0
            run = run + 1 if is_on else 0
            if state != 0:
                out[t] = state * min(1.0, Z_FORECAST_SCALAR * abs(z))
        return self._finalize(pd.Series(out, index=bars.index), bars.index)


# ---------------------------------------------------------------------------------------
# orb
# ---------------------------------------------------------------------------------------
#: session name -> (IANA time zone, local open "HH:MM", local end-of-trading "HH:MM").
#: London: the LBMA/OTC day starts ~08:00 London; New York: 08:00 ET brackets the
#: COMEX pre-open and the 08:30 ET US data releases (whole-hour boundaries keep the
#: opening range aligned with H1 bars in both DST regimes).
DEFAULT_SESSIONS: dict[str, tuple[str, str, str]] = {
    "london": ("Europe/London", "08:00", "16:00"),
    "new_york": ("America/New_York", "08:00", "16:00"),
}


def _minutes(hhmm: str) -> int:
    hh, mm = str(hhmm).split(":")
    m = int(hh) * 60 + int(mm)
    if not 0 <= m <= 24 * 60:
        raise ValueError(f"bad time {hhmm!r}")
    return m


def _hhmm(minutes: int) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def _local_parts(times: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(local day number, local minute of day, local weekday) — DST-aware via zoneinfo."""
    loc = times.tz_convert(ZoneInfo(tz))
    naive = loc.tz_localize(None)
    day = naive.normalize().as_unit("s").asi8 // 86400
    minute = (loc.hour * 60 + loc.minute).to_numpy()
    return np.asarray(day), np.asarray(minute), np.asarray(loc.weekday)


@register_strategy
class OpeningRangeBreakout(Strategy):
    """Session opening-range breakout (London and New York opens, DST-aware).

    Rationale: overnight order flow and news are priced in the first minutes after a major
    session opens; a close outside that opening range means one side has won the
    auction and the session's liquidity is likely to extend the move (Crabel 1990;
    Zarattini & Aziz 2023). Gold's two main liquidity windows are the London open (LBMA,
    European physical/OTC flow) and the New York open (COMEX, 08:30 ET US data).

    Rules, per session and local trading day (weekdays only):
      * opening range (OR) = high/low of bars whose OPEN lies in
        ``[open, open + range_minutes)`` local time and that END by the OR end (bars longer
        than the range cannot form one — use H1 or faster);
      * from the OR end until the local session end, at each close: enter long on the first
        close above the OR high (short below the OR low), at most one entry per session;
      * exit on a close back beyond the opposite side of the OR (``allow_reversal`` flips
        instead) and always flat from the session end;
      * optional filter: skip the session when the OR width is outside
        ``[min_range_atr, max_range_atr] × ATR(atr_n)`` at decision time.
    Session boundaries are converted from local wall-clock time with ``zoneinfo``, so they
    follow London and New York DST independently. Each active session contributes
    ``±level``; overlapping sessions add up and are capped at ±1.

    Defaults: sessions london 08:00-16:00 Europe/London and new_york 08:00-16:00
    America/New_York, 60-minute OR, level 0.5, no width filter.

    References: Crabel (1990); Zarattini & Aziz (2023) SSRN 4416622; Osler (2005) JIMF.
    """

    name = "orb"
    description = ("Opening-range breakout for the London and New York opens (60-min range, "
                   "DST-aware local times); one trade per session, flat at 16:00 local.")

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "sessions": ("london", "new_york"),
            "session_times": None,            # override/extend DEFAULT_SESSIONS
            "range_minutes": 60,
            "level": 0.5,
            "allow_reversal": False,
            "min_range_atr": 0.0,
            "max_range_atr": None,
            "atr_n": 24,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        check_params(self)
        p = self.params
        # a bare string is one session name (tuple("london") would split it into letters)
        p["sessions"] = (p["sessions"],) if isinstance(p["sessions"], str) else tuple(p["sessions"])
        self._warned_coarse = False
        self._warned_sessions: set[str] = set()
        table = self.session_table()
        missing = [s for s in p["sessions"] if s not in table]
        if missing or not p["sessions"]:
            raise ValueError(f"unknown/empty sessions {missing}; known: {sorted(table)}")
        for s in p["sessions"]:
            tz, op, end = table[s]
            ZoneInfo(tz)
            if not _minutes(op) + int(p["range_minutes"]) < _minutes(end):
                raise ValueError(f"session {s!r}: opening range must end before the session end")
        if int(p["range_minutes"]) < 1:
            raise ValueError("range_minutes must be >= 1")
        if not 0.0 < float(p["level"]) <= 1.0:
            raise ValueError("level must be in (0, 1]")
        p["atr_n"] = int(p["atr_n"])

    def session_table(self) -> dict[str, tuple[str, str, str]]:
        table = dict(DEFAULT_SESSIONS)
        extra: Mapping[str, Sequence[str]] | None = self.params.get("session_times")
        if extra:
            table.update({k: (str(v[0]), str(v[1]), str(v[2])) for k, v in extra.items()})
        return table

    @property
    def warmup_bars(self) -> int:
        p = self.params
        uses_atr = float(p["min_range_atr"] or 0.0) > 0 or p["max_range_atr"] is not None
        return p["atr_n"] + 1 if uses_atr else 0

    def session_positions(self, bars: pd.DataFrame, session: str) -> np.ndarray:
        """Position state in {-1, 0, +1} for one session (causal loop)."""
        p = self.params
        tz, op, end = self.session_table()[session]
        or_start, or_end, sess_end = _minutes(op), _minutes(op) + int(p["range_minutes"]), _minutes(end)
        n = len(bars)
        if n == 0:
            return np.zeros(0)
        dur = bar_minutes(bars)
        o_day, o_min, o_wd = _local_parts(pd.DatetimeIndex(bars.index), tz)
        d_day, d_min, d_wd = _local_parts(pd.DatetimeIndex(bars["available_at"]), tz)
        or_bar = (o_min >= or_start) & (o_min + dur <= or_end) & (o_wd < 5)
        window = (d_min >= or_end) & (d_min < sess_end) & (d_wd < 5)
        if not or_bar.any() and bars.index[-1] - bars.index[0] >= pd.Timedelta(days=7):
            # e.g. H1 bars stamped at :30 never fit inside an 08:00-09:00 range: say so
            # instead of silently returning a flat forecast
            warned: set[str] = getattr(self, "_warned_sessions", set())   # absent in old pickles
            if session not in warned:
                logger.warning("orb: no bar fits the %s opening range %s-%s %s in %d bars spanning "
                               "a week or more (misaligned bar timestamps?); forecast is 0",
                               session, op, _hhmm(or_end), tz, n)
                warned.add(session)
                self._warned_sessions = warned
        lo_mult = float(p["min_range_atr"] or 0.0)
        hi_mult = math.inf if p["max_range_atr"] is None else float(p["max_range_atr"])
        use_atr = lo_mult > 0 or math.isfinite(hi_mult)
        a = atr(bars, p["atr_n"]).tolist() if use_atr else [math.nan] * n
        high, low = bars["high"].tolist(), bars["low"].tolist()
        close = bars["close"].tolist()
        or_bar_l, window_l = or_bar.tolist(), window.tolist()
        o_day_l, d_day_l = o_day.tolist(), d_day.tolist()
        reverse = bool(p["allow_reversal"])
        out = [0.0] * n
        or_day, or_hi, or_lo = None, -math.inf, math.inf
        trade_day, state, traded = None, 0, False
        for t in range(n):
            if or_bar_l[t]:
                if o_day_l[t] != or_day:
                    or_day, or_hi, or_lo = o_day_l[t], high[t], low[t]
                else:
                    or_hi, or_lo = max(or_hi, high[t]), min(or_lo, low[t])
            if not (window_l[t] and d_day_l[t] == or_day):
                state = 0
                continue
            if d_day_l[t] != trade_day:
                trade_day, state, traded = d_day_l[t], 0, False
            c = close[t]
            if state == 1 and c < or_lo:
                state = -1 if reverse else 0
            elif state == -1 and c > or_hi:
                state = 1 if reverse else 0
            elif state == 0 and not traded:
                ok = True
                if use_atr:
                    width, at = or_hi - or_lo, a[t]
                    ok = at > 0 and lo_mult * at <= width <= hi_mult * at
                if ok and c > or_hi:
                    state, traded = 1, True
                elif ok and c < or_lo:
                    state, traded = -1, True
            out[t] = float(state)
        return np.asarray(out)

    def generate(self, md: MarketData, features: pd.DataFrame | None = None) -> pd.Series:
        bars = md.bars
        total = np.zeros(len(bars))
        if len(bars) and bar_minutes(bars) > int(self.params["range_minutes"]):
            # no bar fits inside the opening range (e.g. H4/D1): the rule cannot trade
            if not getattr(self, "_warned_coarse", False):   # attr absent in old pickles
                logger.warning("orb: %.0f-minute bars are longer than the %d-minute opening "
                               "range; no opening range can form and the forecast is 0 "
                               "(use H1 or faster bars)", bar_minutes(bars),
                               int(self.params["range_minutes"]))
                self._warned_coarse = True
            return self._finalize(pd.Series(total, index=bars.index), bars.index)
        for s in self.params["sessions"]:
            total += self.session_positions(bars, s) * float(self.params["level"])
        return self._finalize(pd.Series(np.clip(total, -1.0, 1.0), index=bars.index), bars.index)
