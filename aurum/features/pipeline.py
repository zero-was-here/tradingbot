"""Feature pipeline (SPEC §4.1): compute registered groups, fit a scaler on TRAIN only,
transform consistently in research, backtest and live.

Scaling
-------
``scaler="robust"`` (default) standardises each column with the median and
``IQR * 0.7413`` (``1/1.349 = 1/(2 Phi^-1(0.75))``, which equals sigma for a Gaussian), so
fat-tailed financial features are not dominated by a few crash bars (Huber 1981, "Robust
Statistics"). If the IQR is degenerate but the column is not constant (e.g. a spike-y
series that is zero > 75% of the time) the standard deviation is used instead. Columns
whose finite training values are all in {-1, 0, 1} (flags, sign states) are passed through
unscaled. Values are then clipped to ``±clip`` (winsorisation in sigma units).

Point-in-time
-------------
* ``fit`` sees ONLY ``raw_train``: location/scale are plain constants afterwards, so
  ``transform`` is row-wise and cannot leak across time.
* Columns constant (or entirely NaN) in train are dropped; columns missing at transform
  time raise ``FeatureSchemaError``; so do unseen extra columns (``strict=True``), which
  would indicate that live and research feature configs have diverged.
* NaN -> 0 (= the training median after scaling, i.e. a neutral value) happens only AFTER
  the warm-up. A NaN is still "warm-up" (kept NaN) if it occurs before the column's first
  valid value in the frame AND within the first ``max_lookback`` rows; later NaNs (a stale
  macro feed, a zero-range bar) become 0. Both rules only look at the past, so this is
  causal too. Downstream code can drop warm-up rows with ``dropna()``. The rule is
  relative to the FRAME passed in: ``transform(raw.iloc[a:])`` equals
  ``transform(raw).iloc[a:]`` whenever every column has a valid value in its first row.

Warm-up length
--------------
``max_lookback`` is computed per group from the group's EFFECTIVE parameters (signature
defaults + overrides) by a ``lookback_fn(params, bar_minutes)`` attached to the feature
function, so overriding a window moves it while non-window parameters (``bars_per_year``,
``cap_hours``, thresholds) do not. Daily-based groups (``mtf``, ``macro``) need a number of
DAYS whatever the bar size, so their bar count scales with ``bar_minutes`` (recorded by
``compute``, settable in the constructor, persisted in JSON; H1 is assumed until known).
It is the bars needed until values are *defined*; EWM-based features (EMAs, EWMA vol, the
D1 EWM vol in ``mtf``) keep a decaying dependence on their seed — a live runner must compute
on a long history (and check ``parity_report`` against research) for bit-level parity. The
``regime`` group is fully window-bounded (its bounded vol rank is NaN until its window of
``rank_years`` is full, so its warm-up — about one year of bars, e.g. 5,820 H1 bars — is also
the history after which its values no longer depend on where the history starts); the
legacy expanding rank (``expanding=True``) is not.

Group order
-----------
The default group order is canonical (SPEC §4 table order, then other registered groups by
name), NOT the registry insertion order, which depends on which feature module a process
happened to import first (e.g. a strategy importing ``aurum.features.volatility``).

Persistence
-----------
``save``/``load`` use JSON (groups, overrides, scaler settings, columns, per-column
statistics, format version) — human-diffable and safe to ship to the live runner.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.features.base import FeatureSpec, get_feature, list_features
from aurum.features.volatility import bar_minutes as _bar_minutes

logger = logging.getLogger(__name__)

__all__ = ["CANONICAL_GROUP_ORDER", "FORMAT_VERSION", "FeaturePipeline", "FeatureSchemaError",
           "IQR_TO_SIGMA"]

FORMAT_VERSION = 1
#: IQR -> sigma for a normal distribution (1 / 1.34898).
IQR_TO_SIGMA = 0.7413
_SCALERS = ("robust", "standard", "none")
_FORMAT_NAME = "aurum.features.FeaturePipeline"
#: Bar size assumed for ``max_lookback`` until ``compute`` has seen the bars.
_DEFAULT_BAR_MINUTES = 60.0
#: SPEC §4 table order; other registered groups follow, sorted by name.
CANONICAL_GROUP_ORDER: tuple[str, ...] = (
    "returns", "trend", "momentum", "meanrev", "range", "volatility", "microstructure",
    "session", "mtf", "macro", "calendar", "regime",
)


class FeatureSchemaError(ValueError):
    """Raised when a raw feature frame does not match the fitted pipeline's columns."""


def _max_int(obj: Any) -> int:
    """Largest INTEGER found in an overrides value — fallback warm-up estimate for groups
    without a ``lookback_fn``. Floats (``bars_per_year``, hours, thresholds) and booleans
    are ignored: windows are integer bar counts."""
    if isinstance(obj, bool | np.bool_):
        return 0
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, Mapping):
        return max((_max_int(v) for v in obj.values()), default=0)
    if isinstance(obj, Iterable) and not isinstance(obj, str | bytes):
        return max((_max_int(v) for v in obj), default=0)
    return 0


def _effective_params(spec: FeatureSpec, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Keyword defaults of the feature function, then registry params, then overrides."""
    defaults = {k: p.default for k, p in inspect.signature(spec.fn).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return {**defaults, **spec.params, **overrides}


def _ordered_groups() -> list[str]:
    """All registered groups in canonical order (independent of import order)."""
    names = {s.name for s in list_features()}
    head = [g for g in CANONICAL_GROUP_ORDER if g in names]
    return head + sorted(names - set(head))


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list | set | frozenset):
        items = sorted(obj) if isinstance(obj, set | frozenset) else obj
        return [_to_jsonable(v) for v in items]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _from_jsonable(obj: Any) -> Any:
    """Lists -> tuples (feature functions take sequences; tuples are hashable/immutable)."""
    if isinstance(obj, dict):
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(_from_jsonable(v) for v in obj)
    return obj


class FeaturePipeline:
    """Compute -> fit (train only) -> transform, with JSON persistence.

    Parameters
    ----------
    groups    : registry names to compute (default: every registered group, in the canonical
                ``CANONICAL_GROUP_ORDER``). Unknown names raise ``KeyError`` immediately.
    overrides : ``{group: {param: value}}`` forwarded to the group's feature function.
    scaler    : ``"robust"`` (median / IQR*0.7413), ``"standard"`` (mean / std) or ``"none"``.
    clip      : clip scaled values to ``[-clip, clip]``; ``None`` disables clipping.
    warmup    : explicit warm-up in bars (overrides the ``max_lookback`` estimate).
    bar_minutes : bar size used to convert day-based warm-ups into bars; recorded by
                ``compute`` from the bars (H1 is assumed until known).
    """

    def __init__(
        self,
        groups: list[str] | None = None,
        overrides: dict[str, dict] | None = None,
        scaler: str = "robust",
        clip: float | None = 5.0,
        *,
        warmup: int | None = None,
        bar_minutes: float | None = None,
    ) -> None:
        if scaler not in _SCALERS:
            raise ValueError(f"scaler must be one of {_SCALERS}, got {scaler!r}")
        if clip is not None and not (clip > 0 and math.isfinite(clip)):
            raise ValueError("clip must be a positive finite number or None")
        if warmup is not None and warmup < 0:
            raise ValueError("warmup must be >= 0")
        if bar_minutes is not None and not (bar_minutes > 0 and math.isfinite(bar_minutes)):
            raise ValueError("bar_minutes must be a positive finite number or None")
        names = _ordered_groups() if groups is None else list(groups)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate groups in {names}")
        for g in names:
            get_feature(g)  # raises KeyError for unknown names
        overrides = {k: dict(v) for k, v in (overrides or {}).items()}
        unknown = sorted(set(overrides) - set(names))
        if unknown:
            raise ValueError(f"overrides given for groups not in the pipeline: {unknown}")
        self.groups: list[str] = names
        self.overrides: dict[str, dict] = overrides
        self.scaler = scaler
        self.clip = None if clip is None else float(clip)
        self.warmup = None if warmup is None else int(warmup)
        self.bar_minutes: float | None = None if bar_minutes is None else float(bar_minutes)
        self._columns: list[str] = []
        self._dropped: list[str] = []
        self._loc: np.ndarray = np.empty(0)
        self._scale: np.ndarray = np.empty(0)
        self._kind: list[str] = []
        self._raw_columns: list[str] = []
        self.is_fitted: bool = False

    # ---- introspection -----------------------------------------------------------------
    @property
    def max_lookback(self) -> int:
        """Bars of history needed before every column of every group is defined.

        ``warmup`` if given; otherwise the max over groups of the group's
        ``lookback_fn(effective_params, bar_minutes)`` (see module doc). Groups without a
        ``lookback_fn`` fall back to their registered ``lookback`` and, when overridden, the
        largest integer override + 1.
        """
        if self.warmup is not None:
            return self.warmup
        return max((self.group_lookback(g) for g in self.groups), default=0)

    def group_lookback(self, group: str) -> int:
        """Warm-up in bars of one group for the pipeline's overrides and bar size."""
        spec = get_feature(group)
        ov = self.overrides.get(group, {})
        minutes = self.bar_minutes or _DEFAULT_BAR_MINUTES
        fn = getattr(spec.fn, "lookback_fn", None)
        if callable(fn):
            return int(fn(_effective_params(spec, ov), minutes))
        lb = int(spec.lookback)
        if ov:
            lb = max(lb, _max_int(ov) + 1)
        return lb

    @property
    def columns(self) -> list[str]:
        """Output columns of ``transform`` (fitted, non-constant). Before ``fit``: the raw
        columns of the last ``compute`` call (empty if none)."""
        return list(self._columns) if self.is_fitted else list(self._raw_columns)

    @property
    def dropped_columns(self) -> list[str]:
        """Raw columns dropped at fit time (constant or all-NaN in train)."""
        return list(self._dropped)

    @property
    def stats(self) -> pd.DataFrame:
        """Per-column scaler statistics (``loc``, ``scale``, ``kind``)."""
        return pd.DataFrame({"loc": self._loc, "scale": self._scale, "kind": self._kind},
                            index=pd.Index(self._columns, name="column"))

    # ---- compute -----------------------------------------------------------------------
    def compute(self, md: MarketData) -> pd.DataFrame:
        """Raw (unscaled) concatenation of all groups; causal row by row.

        Groups whose declared data requirements are missing (``requires_events`` without
        ``md.events``; ``requires_macro`` names absent from ``md.macro``) are skipped with
        an INFO log — the fitted-column check in ``transform`` will then flag the gap.
        """
        self._record_bar_minutes(md.bars)
        frames: list[pd.DataFrame] = []
        macro = md.macro or {}
        for g in self.groups:
            spec = get_feature(g)
            if spec.requires_events and md.events is None:
                logger.info("feature group %r skipped: requires events", g)
                continue
            missing = [m for m in spec.requires_macro if m not in macro]
            if missing:
                logger.info("feature group %r skipped: missing macro series %s", g, missing)
                continue
            out = spec.compute(md, **self.overrides.get(g, {}))
            frames.append(out)
        if frames:
            raw = pd.concat(frames, axis=1)
        else:
            raw = pd.DataFrame(index=md.bars.index)
        dup = raw.columns[raw.columns.duplicated()].unique().tolist()
        if dup:
            raise ValueError(f"duplicate feature columns across groups: {dup}")
        raw = raw.astype(float).replace([np.inf, -np.inf], np.nan)
        self._raw_columns = list(raw.columns)
        return raw

    def _record_bar_minutes(self, bars: pd.DataFrame) -> None:
        """Remember the bar size (drives day-based warm-ups); warn if it changes after fit —
        the fitted statistics are specific to one bar size."""
        try:
            minutes = float(_bar_minutes(bars))
        except ValueError:
            return
        if not (math.isfinite(minutes) and minutes > 0):
            return
        if self.bar_minutes is not None and self.is_fitted and minutes != self.bar_minutes:
            logger.warning("FeaturePipeline fitted on %.0f-minute bars is computing %.0f-minute "
                           "bars: scaler statistics and warm-up will not match",
                           self.bar_minutes, minutes)
        self.bar_minutes = minutes

    # ---- fit / transform ---------------------------------------------------------------
    def fit(self, raw_train: pd.DataFrame) -> FeaturePipeline:
        """Estimate per-column location/scale from ``raw_train`` ONLY."""
        if raw_train.shape[1] == 0:
            raise ValueError("raw_train has no columns")
        x = raw_train.to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        cols = list(raw_train.columns)
        n_valid = np.sum(~np.isnan(x), axis=0)
        # All-NaN columns trigger "All-NaN slice" RuntimeWarnings; they are dropped below.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cmin = np.nanmin(x, axis=0)
            cmax = np.nanmax(x, axis=0)
            q25, med, q75 = np.nanquantile(x, [0.25, 0.5, 0.75], axis=0)
            mean = np.nanmean(x, axis=0)
            std = np.nanstd(x, axis=0)
        discrete = np.all(np.isnan(x) | (x == 0.0) | (x == 1.0) | (x == -1.0), axis=0)
        keep, loc, scale, kind, dropped = [], [], [], [], []
        for i, c in enumerate(cols):
            if n_valid[i] == 0:
                dropped.append(c)
                logger.info("feature %r dropped: no finite values in train", c)
                continue
            if cmax[i] - cmin[i] <= 0.0:
                dropped.append(c)
                logger.info("feature %r dropped: constant in train", c)
                continue
            if discrete[i]:
                lo, sc, k = 0.0, 1.0, "discrete"
            elif self.scaler == "none":
                lo, sc, k = 0.0, 1.0, "none"
            elif self.scaler == "standard":
                lo, sc, k = float(mean[i]), float(std[i]), "standard"
            else:
                lo, sc, k = float(med[i]), float((q75[i] - q25[i]) * IQR_TO_SIGMA), "robust"
                if not (sc > 1e-12 * max(1.0, abs(lo))):
                    sc, k = float(std[i]), "robust_std"
            if not (sc > 0 and math.isfinite(sc)):
                dropped.append(c)
                logger.info("feature %r dropped: degenerate scale in train", c)
                continue
            keep.append(c)
            loc.append(lo)
            scale.append(sc)
            kind.append(k)
        if not keep:
            raise ValueError("every feature column is constant or empty in raw_train")
        self._columns, self._dropped = keep, dropped
        self._loc, self._scale, self._kind = np.array(loc), np.array(scale), kind
        self.is_fitted = True
        logger.info("FeaturePipeline fitted on %d rows: %d columns kept, %d dropped",
                    len(raw_train), len(keep), len(dropped))
        return self

    def transform(self, raw: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
        """Scale with the fitted statistics, clip, and fill post-warm-up NaN with 0.

        Raises ``FeatureSchemaError`` if a fitted column is missing, or (``strict``) if
        ``raw`` has columns never seen at fit time.

        The warm-up rule treats ``raw.iloc[0]`` as the start of history. For walk-forward
        folds prefer ``transform(raw_full).iloc[test_idx]`` (transform is row-wise and
        causal, so this is leak-free); ``transform(raw_full.iloc[test_idx])`` is identical
        only when every column has a valid value in the fold's first row — otherwise a
        column that is NaN there (e.g. a stale macro feed) stays NaN until its first valid
        value or ``max_lookback`` rows into the fold.
        """
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline is not fitted; call fit() or load() first")
        missing = [c for c in self._columns if c not in raw.columns]
        if missing:
            raise FeatureSchemaError(
                f"{len(missing)} fitted feature column(s) missing at transform time: {missing[:10]}")
        if strict:
            known = set(self._columns) | set(self._dropped)
            unseen = [c for c in raw.columns if c not in known]
            if unseen:
                raise FeatureSchemaError(
                    f"{len(unseen)} column(s) not seen at fit time: {unseen[:10]} "
                    "(pass strict=False to ignore)")
        x = raw[self._columns].to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        x = (x - self._loc) / self._scale
        if self.clip is not None:
            np.clip(x, -self.clip, self.clip, out=x)
        isnan = np.isnan(x)
        if isnan.any():
            seen = np.logical_or.accumulate(~isnan, axis=0)
            past_warmup = (np.arange(x.shape[0]) >= self.max_lookback)[:, None]
            x[isnan & (seen | past_warmup)] = 0.0
        return pd.DataFrame(x, index=raw.index, columns=list(self._columns))

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw).transform(raw)

    # ---- persistence ---------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": _FORMAT_NAME,
            "version": FORMAT_VERSION,
            "groups": list(self.groups),
            "overrides": _to_jsonable(self.overrides),
            "scaler": self.scaler,
            "clip": self.clip,
            "warmup": self.warmup,
            "bar_minutes": self.bar_minutes,
            "max_lookback": self.max_lookback,
            "fitted": self.is_fitted,
            "columns": list(self._columns),
            "dropped": list(self._dropped),
            "stats": {c: {"loc": float(lo), "scale": float(sc), "kind": k}
                      for c, lo, sc, k in zip(self._columns, self._loc, self._scale, self._kind,
                                              strict=True)},
        }

    def save(self, path: str | Path) -> Path:
        """Write the pipeline (config + fitted statistics) as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeaturePipeline:
        if payload.get("format") != _FORMAT_NAME:
            raise ValueError(f"not a FeaturePipeline payload (format={payload.get('format')!r})")
        version = int(payload.get("version", -1))
        if version > FORMAT_VERSION or version < 1:
            raise ValueError(f"unsupported FeaturePipeline format version {version}")
        pipe = cls(
            groups=list(payload["groups"]),
            overrides=_from_jsonable(payload.get("overrides") or {}),
            scaler=payload.get("scaler", "robust"),
            clip=payload.get("clip"),
            warmup=payload.get("warmup"),
            bar_minutes=payload.get("bar_minutes"),
        )
        if payload.get("fitted"):
            cols = list(payload["columns"])
            stats = payload["stats"]
            pipe._columns = cols
            pipe._dropped = list(payload.get("dropped", []))
            pipe._loc = np.array([float(stats[c]["loc"]) for c in cols])
            pipe._scale = np.array([float(stats[c]["scale"]) for c in cols])
            pipe._kind = [str(stats[c]["kind"]) for c in cols]
            pipe.is_fitted = True
        saved_lb = payload.get("max_lookback")
        if saved_lb is not None and int(saved_lb) != pipe.max_lookback:
            logger.warning("max_lookback changed since save (%s -> %s): feature registry differs",
                           saved_lb, pipe.max_lookback)
        return pipe

    @classmethod
    def load(cls, path: str | Path) -> FeaturePipeline:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- diagnostics -----------------------------------------------------------------------
    def parity_report(self, a: pd.DataFrame, b: pd.DataFrame, atol: float = 1e-8) -> pd.DataFrame:
        """Column-by-column comparison of two feature frames on their common index.

        Typical use: batch (research) features vs features recomputed incrementally by the
        live runner. NaN == NaN counts as a match. Returns one row per column (union, ``a``
        order first) with ``in_a``, ``in_b``, ``n_rows``, ``max_abs_diff``, ``n_mismatch``
        (|diff| > atol or NaN in only one frame), ``n_nan_mismatch`` and ``ok``.
        """
        cols = list(a.columns) + [c for c in b.columns if c not in a.columns]
        idx = a.index.intersection(b.index)
        rows = []
        for c in cols:
            in_a, in_b = c in a.columns, c in b.columns
            rec: dict[str, Any] = {"column": c, "in_a": in_a, "in_b": in_b, "n_rows": len(idx),
                                   "max_abs_diff": np.nan, "n_mismatch": len(idx),
                                   "n_nan_mismatch": 0, "ok": False}
            if in_a and in_b:
                x = a.loc[idx, c].to_numpy(dtype=float)
                y = b.loc[idx, c].to_numpy(dtype=float)
                nx, ny = np.isnan(x), np.isnan(y)
                nan_mis = nx ^ ny
                both = ~nx & ~ny
                diff = np.abs(x[both] - y[both])
                n_bad = int(np.sum(diff > atol) + np.sum(nan_mis))
                rec.update(max_abs_diff=float(diff.max()) if diff.size else 0.0,
                           n_mismatch=n_bad, n_nan_mismatch=int(nan_mis.sum()), ok=n_bad == 0)
            rows.append(rec)
        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame(
            columns=["in_a", "in_b", "n_rows", "max_abs_diff", "n_mismatch", "n_nan_mismatch", "ok"])

    def snapshot(self, frame: pd.DataFrame, at: pd.Timestamp | None = None,
                 columns: list[str] | None = None) -> dict[str, float | None]:
        """JSON-safe ``{column: value}`` for one row (default: the last) — e.g. for the LLM
        desk's data provider. NaN/inf become ``None``."""
        if frame.empty:
            return {}
        row = frame.iloc[-1] if at is None else frame.loc[at]
        if columns is not None:
            row = row[columns]
        out: dict[str, float | None] = {}
        for k, v in row.items():
            fv = float(v)
            out[str(k)] = fv if math.isfinite(fv) else None
        return out

    def __repr__(self) -> str:
        state = f"fitted, {len(self._columns)} cols" if self.is_fitted else "unfitted"
        return (f"FeaturePipeline(groups={self.groups}, scaler={self.scaler!r}, clip={self.clip}, "
                f"{state})")
