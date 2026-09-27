"""Triple-barrier, meta- and fixed-horizon labels plus sample-uniqueness weights.

.. warning::

   **EVERY LABEL IN THIS MODULE LOOKS INTO THE FUTURE BY CONSTRUCTION.**

   A label for the event at bar ``t`` is a function of prices in bars ``t+1 .. t1``. Labels
   are *training targets only*: they may be computed inside a strategy's ``fit()`` on the
   TRAINING slice it was given, and nowhere else. Never join a label (or anything derived
   from one - ``ret``, ``t1``, ``barrier_hit``, uniqueness weights) onto a feature frame,
   a forecast, a sizing input or a risk input. The walk-forward engine hands strategies
   training slices only, and this module never reads beyond the end of the frame it is
   given: an event whose outcome is not yet decided at the last bar is dropped (or marked
   ``"incomplete"``), so labels computed on a training slice can never encode post-train
   prices.

Triple-barrier method (López de Prado, *Advances in Financial Machine Learning* (AFML),
2018, ch. 3)
------------------------------------------------------------------------------------------
For an event at the close of bar ``t`` (entry price ``close[t]``) we place

* an upper horizontal barrier at ``entry * exp(+u * trgt[t])``,
* a lower horizontal barrier at ``entry * exp(-d * trgt[t])``,
* a vertical barrier ``max_holding_bars`` bars later,

where ``trgt[t]`` is a volatility estimate KNOWN AT ``t`` (e.g. the per-bar EWM standard
deviation of log returns, :func:`ewm_vol`). Without a side, ``u = pt_mult`` and
``d = sl_mult``: the upper barrier is labelled +1, the lower -1. With a side (meta-labelling,
AFML 3.6) the profit-taking barrier ``pt_mult`` lies in the direction of the bet and the
stop-loss ``sl_mult`` against it, and labels are +1 (PT first), -1 (SL first), 0 (vertical).

Barrier touches are detected on the HIGH/LOW of each subsequent bar - a close-only check
misses intrabar stops and is optimistic. Because the path inside a bar is unknown:

* if a bar OPENS beyond a barrier (a gap), that barrier was hit first, at the open price;
* if both barriers are inside one bar's range, the conservative convention of
  :mod:`aurum.execution.simulator` applies: the **stop is assumed first** (label -1). With
  no side there is no "stop", so the event is labelled 0 with ``barrier_hit="ambiguous"``.

Exit prices are the barrier level (or the gapped open) for horizontal touches and the close
of the vertical-barrier bar otherwise; ``ret = side * log(exit / entry)``.

Sample uniqueness (AFML ch. 4)
------------------------------
Triple-barrier labels overlap in time, so neighbouring observations are not independent:
treating them as IID over-weights long, overlapping episodes and inflates apparent
sample size. Label ``i`` depends on the returns of bars ``(t_idx_i, t1_idx_i]``; the
*concurrency* ``c_t`` is the number of labels depending on bar ``t``'s return, the
*uniqueness* of label ``i`` at ``t`` is ``1 / c_t`` and its *average uniqueness* is the mean
over its lifespan (AFML snippets 4.1-4.2). Using average uniqueness as a sample weight
(optionally times return attribution, AFML 4.10, and time decay, AFML 4.11) is the
standard remedy.

Conventions
-----------
* ``t_idx`` / ``t1_idx`` are integer POSITIONS in the bars frame the labels were built on
  (use them as ``label_end`` for :mod:`aurum.research.splits`); ``t_event`` / ``t1`` are
  the corresponding bar OPEN times (the bars-index convention of SPEC §1). The outcome of
  a label is known at ``bars["available_at"].iloc[t1_idx]``.
* The returned frame is indexed by ``t_event`` (index name ``"time"``, like bars) so that
  it aligns with feature frames via ``.loc``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "LABEL_COLUMNS",
    "BARRIERS",
    "ewm_vol",
    "cusum_filter",
    "get_events",
    "apply_triple_barrier",
    "triple_barrier_labels",
    "fixed_horizon_labels",
    "meta_labels",
    "label_end_positions",
    "drop_label_tail",
    "num_concurrent_events",
    "average_uniqueness",
    "uniqueness_weights",
    "return_attribution_weights",
    "time_decay_weights",
]

#: Columns of every label frame produced by this module (index = ``t_event``, named "time").
LABEL_COLUMNS: list[str] = [
    "t_event", "t1", "label", "ret", "barrier_hit", "t_idx", "t1_idx", "side", "trgt",
    "holding_bars", "entry_price", "exit_price",
]
#: Possible ``barrier_hit`` values.
BARRIERS: tuple[str, ...] = ("pt", "sl", "vertical", "ambiguous", "incomplete")

#: Max elements of one (events x horizon) window matrix; bounds peak memory (~16 bytes each).
_CHUNK_ELEMENTS = 2_000_000
_EVENT_COLUMNS = ["t_idx", "t1_max_idx", "trgt", "side"]


# ---------------------------------------------------------------------------------------
# volatility target and event sampling (both causal)
# ---------------------------------------------------------------------------------------
def ewm_vol(close: pd.Series, span: int = 100, *, min_periods: int | None = None) -> pd.Series:
    """Per-bar EWM standard deviation of log returns (AFML snippet 3.1 ``getDailyVol``).

    Causal: row ``t`` uses returns up to and including bar ``t``. NOT annualised - it is the
    natural unit for barrier widths (a barrier ``k * vol`` is ``k`` one-bar sigmas away).
    Scale by ``sqrt(h)`` for barriers meant to be ``k`` sigmas of an ``h``-bar horizon.
    """
    if span < 2:
        raise ValueError("span must be >= 2")
    mp = max(2, span // 4) if min_periods is None else int(min_periods)
    r = np.log(close.astype(float)).diff()
    vol = r.ewm(span=span, min_periods=mp, adjust=True).std()
    vol.name = "vol"
    return vol


def cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns (AFML snippet 2.4).

    Emits an event at bar ``t`` when the cumulative up- or down-drift since the last reset
    exceeds ``threshold`` (a scalar or a per-bar Series, e.g. ``2 * ewm_vol(close)``). It
    samples bars where *something happened* instead of every bar, which reduces label
    redundancy. Causal: the decision at ``t`` uses returns up to ``t`` and the threshold at
    ``t`` only. Bars with a NaN threshold never trigger (and keep accumulating).
    """
    lr = np.log(close.astype(float)).diff().to_numpy()
    if isinstance(threshold, pd.Series):
        h = threshold.reindex(close.index).to_numpy(dtype=float)
    else:
        h = np.full(len(close), float(threshold))
    s_pos = s_neg = 0.0
    out: list[int] = []
    for i in range(1, len(lr)):
        r = lr[i]
        if not np.isfinite(r):
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        hi = h[i]
        if not np.isfinite(hi) or hi <= 0:
            continue
        if s_neg < -hi:
            s_neg = 0.0
            out.append(i)
        elif s_pos > hi:
            s_pos = 0.0
            out.append(i)
    return close.index[np.asarray(out, dtype=np.int64)]


# ---------------------------------------------------------------------------------------
# input coercion
# ---------------------------------------------------------------------------------------
def _per_bar(values: pd.Series | np.ndarray | float | None, index: pd.DatetimeIndex, name: str,
             default: float | None = None) -> np.ndarray:
    if values is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(len(index), float(default))
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            values = values.reindex(index)
        return values.to_numpy(dtype=float)
    if np.ndim(values) == 0:
        return np.full(len(index), float(values))  # type: ignore[arg-type]
    arr = np.asarray(values, dtype=float)
    if arr.shape != (len(index),):
        raise ValueError(f"{name} must be a scalar or have one value per bar, got shape {arr.shape}")
    return arr


def _event_positions(index: pd.DatetimeIndex,
                     t_events: pd.Index | Sequence | np.ndarray | pd.Series | None) -> np.ndarray:
    """Event positions from None (every bar), timestamps, a boolean mask or int positions."""
    n = len(index)
    if t_events is None:
        return np.arange(n, dtype=np.int64)
    if isinstance(t_events, pd.Series) and t_events.dtype == bool:
        return np.flatnonzero(t_events.reindex(index, fill_value=False).to_numpy())
    if isinstance(t_events, pd.Index) and t_events.dtype.kind in "iub":
        t_events = t_events.to_numpy()  # integer positions / mask, not epoch nanoseconds
    arr = t_events if isinstance(t_events, pd.Index) else np.asarray(t_events)
    if isinstance(arr, np.ndarray) and arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError("boolean t_events must have one value per bar")
        return np.flatnonzero(arr)
    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.integer):
        pos = np.unique(arr.astype(np.int64))
        if pos.size and (pos[0] < 0 or pos[-1] >= n):
            raise ValueError("integer t_events out of range")
        return pos
    times = pd.DatetimeIndex(arr)
    pos = index.get_indexer(times)
    if np.any(pos < 0):
        bad = times[pos < 0][:3].tolist()
        raise KeyError(f"t_events not found in the bars index: {bad}")
    return np.unique(pos.astype(np.int64))


def _side_array(side: pd.Series | np.ndarray | float | None, index: pd.DatetimeIndex) -> np.ndarray | None:
    if side is None:
        return None
    s = _per_bar(side, index, "side")
    return np.sign(np.nan_to_num(s, nan=0.0))


# ---------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------
def get_events(
    bars: pd.DataFrame,
    *,
    vol: pd.Series | np.ndarray | float,
    max_holding_bars: int,
    t_events: pd.Index | Sequence | np.ndarray | pd.Series | None = None,
    side: pd.Series | np.ndarray | float | None = None,
    min_target: float = 0.0,
) -> pd.DataFrame:
    """Candidate events with their barrier target and vertical barrier (AFML ``getEvents``).

    Parameters
    ----------
    bars             : canonical bars frame (only its index is used here).
    vol              : barrier unit per bar, known at ``t`` (Series aligned to ``bars``,
                       array or scalar) - typically :func:`ewm_vol`.
    max_holding_bars : vertical barrier distance in bars (> 0).
    t_events         : event bars (None = every bar; timestamps, boolean mask or positions).
    side             : optional position side per bar (sign is used). Events whose side is
                       0/NaN are dropped - there is no bet to meta-label.
    min_target       : drop events whose ``trgt`` is below this (AFML ``minRet``).

    Returns a frame indexed by event time with ``t_idx``, ``t1_max_idx`` (vertical barrier
    position; may exceed ``len(bars) - 1`` for events near the end), ``trgt`` and ``side``
    (+1 when no side is given). ``attrs["side_given"]`` records whether ``side`` was
    passed, so :func:`apply_triple_barrier` treats the sides as real bets by default.
    """
    if int(max_holding_bars) != max_holding_bars or max_holding_bars < 1:
        raise ValueError("max_holding_bars must be a positive integer")
    h = int(max_holding_bars)
    index = pd.DatetimeIndex(bars.index)
    trgt = _per_bar(vol, index, "vol")
    pos = _event_positions(index, t_events)
    s_all = _side_array(side, index)
    tr = trgt[pos]
    keep = np.isfinite(tr) & (tr > 0) & (tr >= min_target)
    if s_all is not None:
        keep &= s_all[pos] != 0
    pos = pos[keep]
    ev = pd.DataFrame(
        {
            "t_idx": pos,
            "t1_max_idx": pos + h,
            "trgt": trgt[pos],
            "side": (s_all[pos] if s_all is not None else np.ones(len(pos))).astype(float),
        },
        index=index[pos],
    )
    ev.index.name = "time"
    ev.attrs["side_given"] = s_all is not None
    return ev


def _assemble(index: pd.DatetimeIndex, t_idx: np.ndarray, t1_idx: np.ndarray, *, label: np.ndarray,
              ret: np.ndarray, barrier: np.ndarray, side: np.ndarray, trgt: np.ndarray,
              entry: np.ndarray, exit_px: np.ndarray) -> pd.DataFrame:
    """Build a label frame with :data:`LABEL_COLUMNS` (works for zero events too)."""
    t_idx = np.asarray(t_idx, dtype=np.int64)
    t1_idx = np.asarray(t1_idx, dtype=np.int64)
    return pd.DataFrame(
        {
            "t_event": index[t_idx],
            "t1": index[t1_idx],
            "label": np.asarray(label, dtype=float),
            "ret": np.asarray(ret, dtype=float),
            "barrier_hit": np.asarray(barrier, dtype=object),
            "t_idx": t_idx,
            "t1_idx": t1_idx,
            "side": np.asarray(side, dtype=float),
            "trgt": np.asarray(trgt, dtype=float),
            "holding_bars": (t1_idx - t_idx).astype(np.int64),
            "entry_price": np.asarray(entry, dtype=float),
            "exit_price": np.asarray(exit_px, dtype=float),
        },
        index=pd.DatetimeIndex(index[t_idx], name="time"),
    )


def _no_labels(index: pd.DatetimeIndex) -> pd.DataFrame:
    e = np.empty(0)
    i = np.empty(0, dtype=np.int64)
    return _assemble(index, i, i, label=e, ret=e, barrier=np.empty(0, dtype=object), side=e,
                     trgt=e, entry=e, exit_px=e)


def apply_triple_barrier(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pt_mult: float | None,
    sl_mult: float | None,
    side_given: bool | None = None,
    vertical_label: str = "zero",
    min_ret: float = 0.0,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Find the first barrier touch for each event (vectorised over events x horizon).

    ``events`` is the output of :func:`get_events`. ``pt_mult`` / ``sl_mult`` of ``None``,
    0 or ``inf`` disable that barrier (AFML convention for 0). ``side_given`` says whether
    ``events["side"]`` is a real bet direction. Default (None): ``events.attrs["side_given"]``
    as recorded by :func:`get_events`; for a hand-built frame without it, the sides count
    as bets when any of them differs from +1 (an all-long frame then needs
    ``side_given=True`` to get the stop-first tie rule).

    ``vertical_label``: ``"zero"`` (label 0 at the vertical barrier) or ``"sign"`` (sign of
    ``ret``, 0 when ``|ret| <= min_ret``) - AFML's two variants.

    Events whose vertical barrier lies beyond the last bar and that have not touched a
    horizontal barrier by then are undecided: dropped (``drop_incomplete=True``) or kept
    with ``label = NaN`` and ``barrier_hit = "incomplete"``.
    """
    if vertical_label not in ("zero", "sign"):
        raise ValueError("vertical_label must be 'zero' or 'sign'")
    index = pd.DatetimeIndex(bars.index)
    n = len(index)
    if events.empty or n < 2:
        return _no_labels(index)
    if side_given is None:
        recorded = events.attrs.get("side_given")
        side_given = (bool(recorded) if recorded is not None
                      else bool(np.any(events["side"].to_numpy(dtype=float) != 1.0)))
    side_given = bool(side_given)
    pt = _barrier_mult(pt_mult, "pt_mult")
    sl = _barrier_mult(sl_mult, "sl_mult")
    o = bars["open"].to_numpy(dtype=float)
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)

    t_idx = events["t_idx"].to_numpy(dtype=np.int64)
    h_arr = (events["t1_max_idx"].to_numpy(dtype=np.int64) - t_idx)
    if np.any(h_arr < 1):
        raise ValueError("t1_max_idx must be > t_idx")
    h = int(h_arr.max())
    trgt = events["trgt"].to_numpy(dtype=float)
    s = events["side"].to_numpy(dtype=float) if side_given else np.ones(len(t_idx))
    entry = c[t_idx]
    up_mult = np.where(s > 0, pt, sl)
    dn_mult = np.where(s > 0, sl, pt)
    with np.errstate(over="ignore", invalid="ignore"):
        upper = np.where(np.isfinite(up_mult), entry * np.exp(up_mult * trgt), np.inf)
        lower = np.where(np.isfinite(dn_mult), entry * np.exp(-dn_mult * trgt), -np.inf)

    m = len(t_idx)
    k_up = np.full(m, h, dtype=np.int64)
    k_dn = np.full(m, h, dtype=np.int64)
    chunk = max(1, _CHUNK_ELEMENTS // h)
    steps = np.arange(1, h + 1, dtype=np.int64)
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        P = t_idx[a:b, None] + steps[None, :]
        valid = (P <= n - 1) & (steps[None, :] <= h_arr[a:b, None])
        Pc = np.minimum(P, n - 1)
        hit_u = valid & (hi[Pc] >= upper[a:b, None])
        hit_d = valid & (lo[Pc] <= lower[a:b, None])
        any_u = hit_u.any(axis=1)
        any_d = hit_d.any(axis=1)
        k_up[a:b] = np.where(any_u, hit_u.argmax(axis=1), h)
        k_dn[a:b] = np.where(any_d, hit_d.argmax(axis=1), h)

    k = np.minimum(k_up, k_dn)
    touched = k < h
    pos = np.where(touched, t_idx + 1 + k, np.minimum(t_idx + h_arr, n - 1))
    complete_vertical = ~touched & (t_idx + h_arr <= n - 1)
    incomplete = ~touched & ~complete_vertical

    o_at = o[pos]
    first_up = touched & (k_up < k_dn)
    first_dn = touched & (k_dn < k_up)
    both = touched & (k_up == k_dn)
    # a bar that opens beyond a barrier hit it first, at the open
    gap_up = both & (o_at >= upper)
    gap_dn = both & ~gap_up & (o_at <= lower)
    tie = both & ~gap_up & ~gap_dn
    up_hit = first_up | gap_up
    dn_hit = first_dn | gap_dn
    if side_given:
        # conservative: the stop is assumed to fill first when both are inside one bar
        stop_is_up = s < 0
        up_hit = up_hit | (tie & stop_is_up)
        dn_hit = dn_hit | (tie & ~stop_is_up)
        tie = np.zeros(m, dtype=bool)

    exit_px = c[pos].copy()
    exit_px[up_hit] = np.maximum(upper[up_hit], o_at[up_hit])
    exit_px[dn_hit] = np.minimum(lower[dn_hit], o_at[dn_hit])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = s * np.log(exit_px / entry)

    # barrier names relative to the bet: PT is in the direction of ``s``
    is_pt = (up_hit & (s > 0)) | (dn_hit & (s < 0))
    is_sl = (up_hit & (s < 0)) | (dn_hit & (s > 0))
    barrier = np.full(m, "vertical", dtype=object)
    barrier[is_pt] = "pt"
    barrier[is_sl] = "sl"
    barrier[tie] = "ambiguous"
    barrier[incomplete] = "incomplete"

    label = np.zeros(m, dtype=float)
    label[is_pt] = 1.0
    label[is_sl] = -1.0
    if vertical_label == "sign":
        vert = complete_vertical
        label[vert] = np.where(np.abs(ret[vert]) > min_ret, np.sign(ret[vert]), 0.0)
    label[incomplete] = np.nan
    ret = np.where(incomplete, np.nan, ret)
    exit_px = np.where(incomplete, np.nan, exit_px)

    t1_idx = np.where(incomplete, n - 1, pos).astype(np.int64)
    out = _assemble(index, t_idx, t1_idx, label=label, ret=ret, barrier=barrier,
                    side=s if side_given else np.zeros(m), trgt=trgt, entry=entry, exit_px=exit_px)
    if drop_incomplete:
        n_inc = int(incomplete.sum())
        if n_inc:
            logger.debug("triple barrier: dropped %d undecided events at the end of the data", n_inc)
        out = out.loc[~incomplete]
    return out


def _barrier_mult(x: float | None, name: str) -> float:
    if x is None or x == 0 or not np.isfinite(x):
        return np.inf
    if x < 0:
        raise ValueError(f"{name} must be >= 0")
    return float(x)


def triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    pt_mult: float | None,
    sl_mult: float | None,
    max_holding_bars: int,
    vol: pd.Series | np.ndarray | float,
    side: pd.Series | np.ndarray | float | None = None,
    t_events: pd.Index | Sequence | np.ndarray | pd.Series | None = None,
    vertical_label: str = "zero",
    min_ret: float = 0.0,
    min_target: float = 0.0,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Triple-barrier labels (AFML ch. 3) - **FORWARD-LOOKING: training targets only.**

    Parameters
    ----------
    bars             : canonical bars (mid OHLC). Only these bars are read: labels never
                       extend beyond ``bars.index[-1]``.
    pt_mult, sl_mult : barrier widths in units of ``vol`` (None/0 disables a barrier). With
                       ``side=None`` they are the upper/lower barriers; with a side they are
                       profit-take (with the bet) and stop-loss (against it).
    max_holding_bars : vertical barrier, in bars after the event.
    vol              : per-bar barrier unit known at the event (e.g. :func:`ewm_vol`).
    side             : optional bet direction per bar (meta-labelling); events with side 0
                       or NaN are skipped.
    t_events         : which bars are events (default every bar with a valid ``vol``).
    vertical_label   : ``"zero"`` or ``"sign"`` (see :func:`apply_triple_barrier`).
    min_ret          : dead band for ``vertical_label="sign"``.
    min_target       : minimum ``vol`` for an event (AFML ``minRet``).
    drop_incomplete  : drop events still undecided at the last bar (default) instead of
                       returning them with a NaN label.

    Returns
    -------
    DataFrame indexed by event time (``"time"``) with :data:`LABEL_COLUMNS`:
    ``t_event``, ``t1`` (open time of the bar in which the first barrier was touched, or the
    vertical-barrier bar), ``label`` in {-1, 0, 1}, ``ret`` (side-adjusted log return to the
    exit), ``barrier_hit`` in {"pt", "sl", "vertical", "ambiguous", "incomplete"},
    positions ``t_idx``/``t1_idx``, ``side`` (0 when no side was given), ``trgt``,
    ``holding_bars``, ``entry_price`` (``close[t]``) and ``exit_price``.
    """
    events = get_events(bars, vol=vol, max_holding_bars=max_holding_bars, t_events=t_events,
                        side=side, min_target=min_target)
    return apply_triple_barrier(bars, events, pt_mult=pt_mult, sl_mult=sl_mult,
                                side_given=side is not None, vertical_label=vertical_label,
                                min_ret=min_ret, drop_incomplete=drop_incomplete)


def fixed_horizon_labels(
    bars: pd.DataFrame,
    horizon: int,
    *,
    vol: pd.Series | np.ndarray | float | None = None,
    threshold: float = 0.0,
    t_events: pd.Index | Sequence | np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Fixed-horizon, volatility-normalised forward-return labels - **FORWARD-LOOKING.**

    ``ret = log(close[t+h] / close[t])``; with ``vol`` (per-bar sigma known at ``t``) the
    return is normalised to ``z = ret / (vol[t] * sqrt(h))`` and ``label = sign(z)`` when
    ``|z| > threshold`` else 0 (without ``vol``, ``threshold`` applies to ``ret``). The
    normalisation makes a single threshold meaningful across volatility regimes - the main
    flaw of raw fixed-horizon labels noted in AFML 3.2. Events with ``t + h`` beyond the
    last bar are dropped. Same columns as :func:`triple_barrier_labels` plus ``ret_norm``;
    ``barrier_hit`` is always ``"vertical"``.
    """
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    h = int(horizon)
    index = pd.DatetimeIndex(bars.index)
    n = len(index)
    pos = _event_positions(index, t_events)
    pos = pos[pos + h <= n - 1]
    c = bars["close"].to_numpy(dtype=float)
    v = _per_bar(vol, index, "vol")[pos] if vol is not None else np.ones(len(pos))
    if vol is not None:
        ok = np.isfinite(v) & (v > 0)
        pos, v = pos[ok], v[ok]
    ret = np.log(c[pos + h] / c[pos])
    z = ret / (v * np.sqrt(h)) if vol is not None else ret
    label = np.where(np.abs(z) > threshold, np.sign(z), 0.0)
    t1_idx = pos + h
    out = _assemble(index, pos, t1_idx, label=label, ret=ret,
                    barrier=np.full(len(pos), "vertical", dtype=object), side=np.zeros(len(pos)),
                    trgt=v if vol is not None else np.full(len(pos), np.nan), entry=c[pos],
                    exit_px=c[t1_idx])
    out["ret_norm"] = z
    return out


def meta_labels(labels: pd.DataFrame, *, vertical: str = "fail", min_ret: float = 0.0) -> pd.Series:
    """Binary meta-labels (AFML 3.6) from side-aware triple-barrier labels.

    1 = the primary bet hit its profit-take first, 0 = it hit the stop first.
    ``vertical`` decides the timeouts: ``"fail"`` (0 - strictly "PT before SL") or
    ``"return_sign"`` (1 if the side-adjusted return at the vertical barrier exceeds
    ``min_ret``; AFML's ``getBins``). Ambiguous/incomplete rows are NaN.
    """
    if vertical not in ("fail", "return_sign"):
        raise ValueError("vertical must be 'fail' or 'return_sign'")
    hit = labels["barrier_hit"]
    y = pd.Series(np.nan, index=labels.index, name="meta_label")
    y[hit == "pt"] = 1.0
    y[hit == "sl"] = 0.0
    vert = hit == "vertical"
    if vertical == "fail":
        y[vert] = 0.0
    else:
        y[vert] = (labels.loc[vert, "ret"] > min_ret).astype(float)
    return y


# ---------------------------------------------------------------------------------------
# purging helpers
# ---------------------------------------------------------------------------------------
def label_end_positions(labels: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Inclusive label-end position per BAR for :mod:`aurum.research.splits` ``label_end``.

    Bars carrying a label get ``max(t1_idx)`` over labels starting there; other bars get
    their own position (no forward reach).
    """
    le = np.arange(n_bars, dtype=np.int64)
    if len(labels):
        t0 = labels["t_idx"].to_numpy(dtype=np.int64)
        t1 = labels["t1_idx"].to_numpy(dtype=np.int64)
        if t0.max() >= n_bars or t1.max() >= n_bars:
            raise ValueError("labels reference positions beyond n_bars")
        np.maximum.at(le, t0, t1)
    return le


def drop_label_tail(labels: pd.DataFrame, n_bars: int, max_holding_bars: int) -> pd.DataFrame:
    """Keep only events that start at least ``max_holding_bars`` before the last bar.

    Dropping every label that *could* have reached beyond the data end (not just those that
    actually did) avoids a selection bias: near the end only quickly-resolved events would
    survive, over-representing fast barrier touches.
    """
    return labels.loc[labels["t_idx"].to_numpy() <= n_bars - 1 - int(max_holding_bars)]


# ---------------------------------------------------------------------------------------
# uniqueness / sample weights (AFML ch. 4)
# ---------------------------------------------------------------------------------------
def _spans(labels_or_t_idx, t1_idx) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(labels_or_t_idx, pd.DataFrame):
        return (labels_or_t_idx["t_idx"].to_numpy(dtype=np.int64),
                labels_or_t_idx["t1_idx"].to_numpy(dtype=np.int64))
    if t1_idx is None:
        raise ValueError("t1_idx is required when t_idx is an array")
    t0 = np.asarray(labels_or_t_idx, dtype=np.int64)
    t1 = np.asarray(t1_idx, dtype=np.int64)
    if t0.shape != t1.shape:
        raise ValueError("t_idx and t1_idx must have the same shape")
    if np.any(t1 < t0):
        raise ValueError("t1_idx must be >= t_idx")
    return t0, t1


def num_concurrent_events(n_bars: int, t_idx, t1_idx=None) -> np.ndarray:
    """Concurrency ``c_t``: number of labels whose return span ``(t_idx, t1_idx]`` covers bar
    ``t`` (AFML snippet 4.1, attributing bar ``t``'s return ``close[t]/close[t-1]`` to bar
    ``t``). O(n) via a difference array. ``t_idx`` may be a label frame."""
    t0, t1 = _spans(t_idx, t1_idx)
    diff = np.zeros(int(n_bars) + 1, dtype=np.int64)
    np.add.at(diff, np.minimum(t0 + 1, n_bars), 1)
    np.add.at(diff, np.minimum(t1 + 1, n_bars), -1)
    return np.cumsum(diff)[: int(n_bars)]


def average_uniqueness(t_idx, t1_idx=None, *, n_bars: int | None = None) -> np.ndarray:
    """Average uniqueness of each label over its lifespan (AFML snippet 4.2), in (0, 1].

    ``u_i = mean_{t in (t_idx_i, t1_idx_i]} 1 / c_t``; a label with an empty span
    (``t1 == t0``) gets 1. ``n_bars`` defaults to ``max(t1_idx) + 1``.
    """
    t0, t1 = _spans(t_idx, t1_idx)
    if t0.size == 0:
        return np.empty(0)
    n = int(t1.max()) + 1 if n_bars is None else int(n_bars)
    c = num_concurrent_events(n, t0, t1)
    inv = np.where(c > 0, 1.0 / np.maximum(c, 1), 0.0)
    cs = np.r_[0.0, np.cumsum(inv)]  # cs[k] = sum inv[0..k-1]
    length = (t1 - t0).astype(float)
    total = cs[t1 + 1] - cs[t0 + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(length > 0, total / np.maximum(length, 1.0), 1.0)
    return u


def uniqueness_weights(labels: pd.DataFrame, *, n_bars: int | None = None) -> pd.Series:
    """Average uniqueness as a Series aligned with ``labels`` (normalised to mean 1)."""
    u = average_uniqueness(labels, n_bars=n_bars)
    if u.size and u.mean() > 0:
        u = u / u.mean()
    return pd.Series(u, index=labels.index, name="uniqueness")


def return_attribution_weights(labels: pd.DataFrame, close: pd.Series) -> pd.Series:
    """AFML snippet 4.10: ``w_i = |sum_{t in span_i} r_t / c_t|`` normalised to mean 1.

    Labels spanning large absolute (concurrency-shared) log returns get more weight, so the
    classifier focuses on economically meaningful outcomes. ``close`` must be the price
    series the labels were built on.
    """
    n = len(close)
    t0, t1 = _spans(labels, None)
    if t0.size == 0:
        return pd.Series(np.empty(0), index=labels.index, name="return_attribution")
    c = num_concurrent_events(n, t0, t1)
    r = np.log(close.to_numpy(dtype=float))
    r = np.r_[0.0, np.diff(r)]
    contrib = np.where(c > 0, r / np.maximum(c, 1), 0.0)
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(contrib))]
    w = np.abs(cs[t1 + 1] - cs[t0 + 1])
    if w.mean() > 0:
        w = w / w.mean()
    return pd.Series(w, index=labels.index, name="return_attribution")


def time_decay_weights(avg_uniqueness: np.ndarray | pd.Series, last_weight: float = 1.0) -> np.ndarray:
    """Piecewise-linear time decay over cumulative uniqueness (AFML snippet 4.11).

    The newest observation has weight 1 and the oldest ``last_weight`` (``c`` in AFML):
    ``c = 1`` no decay, ``0 < c < 1`` linear decay, ``c = 0`` decays to zero, ``-1 < c < 0``
    zeroes the oldest ``|c|`` fraction. Input must be in chronological order.
    """
    u = np.asarray(avg_uniqueness, dtype=float)
    if u.size == 0:
        return u
    if not -1.0 < last_weight <= 1.0:
        raise ValueError("last_weight must be in (-1, 1]")
    x = np.cumsum(u)
    total = x[-1]
    if last_weight >= 0:
        slope = (1.0 - last_weight) / total
    else:
        slope = 1.0 / ((last_weight + 1.0) * total)
    const = 1.0 - slope * total
    return np.maximum(0.0, const + slope * x)
