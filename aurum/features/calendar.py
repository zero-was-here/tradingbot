"""Economic-calendar proximity features (``calendar``).

Scheduled US macro releases (NFP, CPI, FOMC) produce the largest intraday gold moves and
spread blow-outs; volatility compresses before them and expands after (e.g. Andersen,
Bollerslev, Diebold & Vega 2003, "Micro Effects of Macro Announcements", AER 93(1)). These
features let models and the risk layer condition on "event proximity".

Leakage note (SPEC §3.5): release TIMES are scheduled and published well in advance, so using
the time of the *next* event is legitimate. OUTCOMES (actual/forecast/surprise) are not used
here at all. Caveat: unscheduled announcements (emergency FOMC cuts) appear in historical
calendars but could not have been known in advance — such rows should be flagged in the
calendar source (e.g. ``importance`` or a separate name) if that matters.

All times are relative to the decision instant ``available_at[t]``. An event whose
scheduled time equals the decision time counts as *upcoming* (its outcome is not yet known).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.base import register_feature

logger = logging.getLogger(__name__)

__all__ = ["EVENT_PATTERNS", "calendar_features", "classify_event"]

_NS_PER_HOUR = 3_600_000_000_000

#: Upper-cased substrings that map a calendar ``name`` onto an event type.
EVENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "nfp": ("NFP", "NON-FARM", "NONFARM", "NON FARM"),
    "cpi": ("CPI", "CONSUMER PRICE"),
    "fomc": ("FOMC", "FED INTEREST RATE", "FEDERAL FUNDS RATE", "FED FUNDS RATE"),
}


def classify_event(names: pd.Series) -> dict[str, np.ndarray]:
    """Boolean membership of each event name in each ``EVENT_PATTERNS`` type."""
    up = names.astype(str).str.upper()
    return {k: np.logical_or.reduce([up.str.contains(p, regex=False).to_numpy() for p in pats])
            for k, pats in EVENT_PATTERNS.items()}


def _ns(values) -> np.ndarray:
    idx = pd.DatetimeIndex(values)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.as_unit("ns").asi8


@register_feature("calendar", family="calendar", lookback=0, requires_events=True)
def calendar_features(
    md: MarketData,
    *,
    min_importance: int = 3,
    cap_hours: float = 72.0,
    currencies: Sequence[str] | None = None,
    near_minutes: float = 30.0,
    wide_hours: float = 2.0,
    count_horizon_hours: float = 24.0,
) -> pd.DataFrame:
    """Event-proximity features for events with ``importance >= min_importance``.

    * ``calendar_hours_to_next`` / ``calendar_hours_since_last``: hours to the next
      (``time >= T``) / since the last (``time < T``) event, capped at ``cap_hours`` (also
      used when no such event exists in the calendar).
    * ``calendar_in_30m`` / ``calendar_in_2h``: an event within ±``near_minutes`` /
      ±``wide_hours`` of T; ``calendar_pre_2h`` / ``calendar_post_2h``: one-sided versions.
    * ``calendar_next_{nfp,cpi,fomc}``: one-hot type of the next event (any of the events
      sharing the next timestamp), zero when the next event is beyond ``cap_hours``.
    * ``calendar_n_next_24h``: number of qualifying event timestamps in ``[T, T+24h)``.

    Returns an empty frame (bars index) when ``md.events`` is None.
    """
    bars = md.bars
    if md.events is None:
        logger.info("calendar: md.events is None; returning no calendar features")
        return pd.DataFrame(index=bars.index)
    ev = md.events
    missing = [c for c in ("time", "name") if c not in ev.columns]
    if missing:
        raise KeyError(f"events frame lacks columns {missing}")
    mask = np.ones(len(ev), dtype=bool)
    if "importance" in ev.columns:
        mask &= pd.to_numeric(ev["importance"], errors="coerce").fillna(0).to_numpy() >= min_importance
    if currencies is not None and "currency" in ev.columns:
        mask &= ev["currency"].astype(str).str.upper().isin([c.upper() for c in currencies]).to_numpy()
    # Unparseable / missing release times would become the int64 minimum ("year 1677").
    mask &= pd.DatetimeIndex(pd.to_datetime(ev["time"], utc=True, errors="coerce")).notna()
    ev = ev.loc[mask]
    t_ns = _ns(bars["available_at"])
    n = t_ns.size

    if len(ev):
        e_ns = _ns(ev["time"])
        kinds = classify_event(ev["name"])
        order = np.argsort(e_ns, kind="stable")
        e_sorted = e_ns[order]
        uniq, first = np.unique(e_sorted, return_index=True)
        # Type flags per unique timestamp: OR over events sharing that time.
        type_flags = {k: np.logical_or.reduceat(v[order], first) for k, v in kinds.items()}
    else:
        uniq = np.empty(0, dtype=np.int64)
        type_flags = {k: np.empty(0, dtype=bool) for k in EVENT_PATTERNS}

    nxt = np.searchsorted(uniq, t_ns, side="left")          # first event with time >= T
    has_next = nxt < uniq.size
    has_last = nxt > 0
    safe_next = np.minimum(nxt, max(uniq.size - 1, 0))
    safe_last = np.maximum(nxt - 1, 0)
    h_next = np.full(n, np.inf)
    h_last = np.full(n, np.inf)
    if uniq.size:
        h_next = np.where(has_next, (uniq[safe_next] - t_ns) / _NS_PER_HOUR, np.inf)
        h_last = np.where(has_last, (t_ns - uniq[safe_last]) / _NS_PER_HOUR, np.inf)
    near = near_minutes / 60.0
    within_cap = h_next <= cap_hours
    cols: dict[str, np.ndarray] = {
        "calendar_hours_to_next": np.minimum(h_next, cap_hours),
        "calendar_hours_since_last": np.minimum(h_last, cap_hours),
        "calendar_in_30m": ((h_next <= near) | (h_last <= near)).astype(float),
        "calendar_in_2h": ((h_next <= wide_hours) | (h_last <= wide_hours)).astype(float),
        "calendar_pre_2h": (h_next <= wide_hours).astype(float),
        "calendar_post_2h": (h_last <= wide_hours).astype(float),
    }
    for k in EVENT_PATTERNS:
        flag = np.zeros(n, dtype=bool)
        if uniq.size:
            flag = type_flags[k][safe_next] & has_next & within_cap
        cols[f"calendar_next_{k}"] = flag.astype(float)
    horizon = int(count_horizon_hours * _NS_PER_HOUR)
    end = np.searchsorted(uniq, t_ns + horizon, side="left")
    cols[f"calendar_n_next_{int(count_horizon_hours)}h"] = (end - nxt).astype(float)
    if uniq.size and t_ns.size and t_ns[-1] > uniq[-1]:
        logger.info("calendar: bars extend beyond the last calendar event (%s); "
                    "hours_to_next is capped there", pd.Timestamp(int(uniq[-1]), tz="UTC"))
    return pd.DataFrame(cols, index=bars.index, dtype=float)


def _no_lookback(p: object, bar_minutes: float = 60.0) -> int:
    """Event proximity uses the (public) schedule only: no bar history needed."""
    return 0


calendar_features.lookback_fn = _no_lookback  # type: ignore[attr-defined]
