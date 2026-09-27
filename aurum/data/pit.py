"""Point-in-time alignment helpers.

The single rule of the codebase: a value may be used at decision time ``T`` only if its
``available_at <= T``. ``asof_join`` implements that rule for aligning any secondary frame
(higher timeframe bars, daily macro series, event tables) onto the trading bars.
"""

from __future__ import annotations

import pandas as pd


def asof_join(
    decision_times: pd.Series | pd.DatetimeIndex,
    right: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    available_col: str = "available_at",
    tolerance: pd.Timedelta | None = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """For each decision time, take the latest ``right`` row with ``available_at <= time``.

    Parameters
    ----------
    decision_times : times at which decisions are made — normally ``bars["available_at"]``.
    right          : frame with an ``available_at`` column (tz-aware UTC).
    columns        : columns of ``right`` to return (default: all except ``available_at``).
    tolerance      : optional max staleness; older matches become NaN.
    index          : index for the result (default: ``bars`` index if a Series was passed).

    Returns a frame aligned row-for-row with ``decision_times``. Rows with no eligible
    ``right`` row are NaN (never back-filled); so are rows whose decision time is NaT, and
    every row when ``right`` is empty.
    """
    if available_col not in right.columns:
        raise KeyError(f"right frame lacks {available_col!r}")
    if isinstance(decision_times, pd.Series):
        out_index = index if index is not None else decision_times.index
        # DatetimeIndex(Series) keeps the tz even for an empty Series (``to_numpy()`` of a
        # tz-aware Series is an object array, which loses the tz when empty) and is faster.
        times = pd.DatetimeIndex(decision_times)
    else:
        times = pd.DatetimeIndex(decision_times)
        out_index = index if index is not None else times
    if times.tz is None:
        if len(times):
            raise ValueError("decision_times must be tz-aware")
        times = times.tz_localize("UTC")  # nothing to align; accept an untyped empty input
    cols = [c for c in (columns or right.columns) if c != available_col]
    missing = [c for c in cols if c not in right.columns]
    if missing:
        raise KeyError(f"right frame lacks columns {missing}")

    r = right[[available_col] + cols].copy()
    avail = pd.DatetimeIndex(r[available_col])
    if avail.tz is None and len(avail.dropna()):
        raise ValueError(f"right[{available_col!r}] must be tz-aware")
    r[available_col] = avail.tz_localize("UTC") if avail.tz is None else avail.tz_convert("UTC")
    r = r.dropna(subset=[available_col]).sort_values(available_col, kind="stable")
    # If two rows become available at the same instant keep the last one (latest info).
    r = r.drop_duplicates(subset=[available_col], keep="last")
    r = r.rename(columns={available_col: "_avail"})
    r["_avail"] = r["_avail"].astype("datetime64[ns, UTC]")

    left = pd.DataFrame({"_t": times.tz_convert("UTC").astype("datetime64[ns, UTC]")})
    left["_pos"] = range(len(left))
    # merge_asof rejects null keys: decision rows without a time (NaT) get NaN results.
    valid = left["_t"].notna().to_numpy()
    left_sorted = left.loc[valid].sort_values("_t", kind="stable")
    if len(r) == 0 or len(left_sorted) == 0:
        result = pd.DataFrame({c: right[c].iloc[:0].reindex(range(len(left))) for c in cols})
        result.index = out_index
        return result
    merged = pd.merge_asof(
        left_sorted,
        r,
        left_on="_t",
        right_on="_avail",
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )
    merged = merged.set_index("_pos").reindex(range(len(left)))
    result = merged[cols].reset_index(drop=True)
    result.index = out_index
    return result
