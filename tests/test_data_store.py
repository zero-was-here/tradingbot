"""Tests for aurum.data.store (parquet round-trip, frame hashing) and macro persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurum.data.macro import load_macro_dir, save_macro_dir
from aurum.data.schema import validate_bars
from aurum.data.store import frame_hash, load_bars, load_frame, save_bars, save_frame
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro


def test_save_load_bars_round_trip(tmp_path: Path):
    bars = make_synthetic_bars(500, "H1", seed=1)
    bars.attrs.update({"source": "synthetic", "price_scale": 1000.0})
    p = save_bars(bars, tmp_path / "sub" / "xau_H1.parquet")
    out = load_bars(p)
    validate_bars(out)
    pd.testing.assert_frame_equal(out, bars, check_freq=False)
    assert str(out.index.tz) == "UTC" and out.index.name == "time"
    assert isinstance(out["available_at"].dtype, pd.DatetimeTZDtype)
    assert out.attrs["timeframe"] == "H1"
    assert out.attrs["source"] == "synthetic" and out.attrs["price_scale"] == 1000.0
    assert out.attrs["_meta"]["n_rows"] == 500
    assert frame_hash(out) == frame_hash(bars)


def test_load_bars_detects_tampering(tmp_path: Path):
    bars = make_synthetic_bars(50, "M15", seed=2)
    p = save_bars(bars, tmp_path / "b.parquet")
    df = load_frame(p)
    df.iloc[3, df.columns.get_loc("close")] += 0.01
    meta = df.attrs.pop("_meta")
    # rewrite with the OLD hash in metadata → load must refuse
    save_frame(df, p, metadata=meta)
    with pytest.raises(ValueError, match="hash"):
        load_bars(p)
    assert load_bars(p, verify_hash=False)["close"].iloc[3] == pytest.approx(bars["close"].iloc[3] + 0.01)


def test_frame_hash_stable_and_sensitive():
    bars = make_synthetic_bars(200, "H1", seed=3)
    h = frame_hash(bars)
    assert h == frame_hash(bars.copy())
    # independent of datetime resolution (us vs ns) — pandas 3 defaults to us
    b_ns = bars.copy()
    b_ns.index = b_ns.index.as_unit("ns")
    b_ns["available_at"] = pd.DatetimeIndex(b_ns["available_at"]).as_unit("ns")
    assert frame_hash(b_ns) == h
    # -0.0 vs 0.0 and NaN payloads are canonical
    a = pd.DataFrame({"x": [0.0, np.nan]})
    b = pd.DataFrame({"x": [-0.0, float("nan")]})
    assert frame_hash(a) == frame_hash(b)
    # sensitive to values, index, column names/order
    changed = bars.copy()
    changed.iloc[10, 0] += 1e-9
    assert frame_hash(changed) != h
    assert frame_hash(bars.rename(columns={"open": "o"})) != h
    assert frame_hash(bars[list(reversed(bars.columns))]) != h
    shifted = bars.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    assert frame_hash(shifted) != h
    # object/text columns are supported
    assert frame_hash(pd.DataFrame({"s": ["a", None]})) != frame_hash(pd.DataFrame({"s": ["a", "b"]}))


def test_macro_dir_round_trip(tmp_path: Path):
    bars = make_synthetic_bars(1000, "H1", seed=4)
    macro = make_synthetic_macro(bars, seed=4)
    save_macro_dir(macro, tmp_path / "macro")
    back = load_macro_dir(tmp_path / "macro")
    assert set(back) == set(macro)
    for k in macro:
        pd.testing.assert_frame_equal(back[k], macro[k], check_freq=False)
        assert str(back[k].index.tz) == "UTC"
        assert isinstance(back[k]["available_at"].dtype, pd.DatetimeTZDtype)


def test_save_macro_dir_rejects_bad_frames(tmp_path: Path):
    bad = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
    with pytest.raises(ValueError):
        save_macro_dir({"x": bad}, tmp_path)
    early = bad.assign(available_at=pd.DatetimeIndex(["2023-12-31"], tz="UTC"))
    with pytest.raises(ValueError):
        save_macro_dir({"x": early}, tmp_path)


# ------------------------------------------------------------------------------------ review
def test_save_bars_non_canonical_layout_does_not_trip_hash_check(tmp_path: Path):
    """Regression: the hash was taken on the caller's layout but verified on the canonical
    one, so a valid frame with ``spread`` before ``volume`` or an unnamed index was
    reported as corrupted on reload."""
    bars = make_synthetic_bars(120, "H1", seed=5)
    reordered = bars[["open", "high", "low", "close", "spread", "volume", "available_at"]]
    reordered.attrs = dict(bars.attrs)
    out = load_bars(save_bars(reordered, tmp_path / "a.parquet"))
    pd.testing.assert_frame_equal(out, bars, check_freq=False)
    unnamed = bars.rename_axis(None)
    unnamed.attrs = dict(bars.attrs)
    out2 = load_bars(save_bars(unnamed, tmp_path / "b.parquet"))
    assert out2.index.name == "time"
    pd.testing.assert_frame_equal(out2, bars, check_freq=False)
    # the caller's frame is not mutated
    assert unnamed.index.name is None and list(reordered.columns)[4] == "spread"


def test_save_bars_harmonises_mixed_datetime_units(tmp_path: Path):
    """A decoder can yield a seconds-resolution index next to a us ``available_at``;
    stored bars must come back with ONE unit so ``merge_asof`` on them works."""
    bars = make_synthetic_bars(60, "M15", seed=6)
    mixed = bars.copy()
    mixed.index = mixed.index.as_unit("s")
    mixed["available_at"] = pd.DatetimeIndex(mixed["available_at"]).as_unit("us")
    h = frame_hash(mixed)
    out = load_bars(save_bars(mixed, tmp_path / "m.parquet"))
    assert out.index.unit == pd.DatetimeIndex(out["available_at"]).unit
    assert frame_hash(out) == h  # values unchanged
    left = pd.DataFrame({"t": out.index})
    right = pd.DataFrame({"t": out["available_at"].to_numpy(), "c": out["close"].to_numpy()})
    merged = pd.merge_asof(left, right, on="t")  # raised "incompatible merge keys" before
    assert merged["c"].iloc[1] == out["close"].iloc[0]


def test_save_frame_does_not_warn_on_rich_attrs(tmp_path: Path):
    import warnings

    bars = make_synthetic_bars(20, "H1", seed=7)
    bars.attrs.update({"built": pd.Timestamp("2024-01-01", tz="UTC"), "arr": np.arange(3)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pyarrow warned "Could not serialize attrs"
        out = load_bars(save_bars(bars, tmp_path / "w.parquet"))
    assert out.attrs["built"] == "2024-01-01 00:00:00+00:00" and out.attrs["timeframe"] == "H1"


def test_empty_bars_round_trip(tmp_path: Path):
    bars = make_synthetic_bars(10, "H1", seed=1).iloc[:0]
    bars.attrs = {"timeframe": "H1"}
    out = load_bars(save_bars(bars, tmp_path / "e.parquet"))
    assert len(out) == 0 and str(out.index.tz) == "UTC" and out.attrs["timeframe"] == "H1"
