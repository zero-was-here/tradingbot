"""The suite-wide data_store/ write guard of tests/conftest.py fires, and only there.

The writers are wrapped in ``pytest_configure`` (before collection), so even the names this
module imports at import time are the guarded ones. Every refusal happens before the disk is
touched: the probe paths below never come into existence.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import aurum.data
from aurum.data import dukascopy, macro, store
from aurum.data.macro import save_macro_dir
from aurum.data.store import load_bars, save_bars, save_frame
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frames():
    bars = make_synthetic_bars(48, "H1", seed=3)
    return bars, make_synthetic_macro(bars, seed=3)


def _writers(bars, macro_frames):
    """Every guarded writer as ``name -> f(target)``."""
    table = pa.Table.from_pandas(bars[["open", "close"]])
    return {
        "save_frame (imported at module import)": lambda t: save_frame(bars, t / "f.parquet"),
        "save_bars (imported at module import)": lambda t: save_bars(bars, t / "b.parquet"),
        "save_macro_dir (imported at module import)": lambda t: save_macro_dir(macro_frames, t / "macro"),
        "aurum.data.store.save_bars": lambda t: store.save_bars(bars, path=t / "b.parquet"),
        "aurum.data.macro.save_frame": lambda t: macro.save_frame(bars, t / "m.parquet"),
        "aurum.data.save_bars (lazy re-export)": lambda t: aurum.data.save_bars(bars, t / "b.parquet"),
        "aurum.data.dukascopy.build_dataset": lambda t: dukascopy.build_dataset(t / "store", offline=True,
                                                                                cache_dir=t / "cache"),
        "pyarrow.parquet.write_table": lambda t: pq.write_table(table, str(t / "t.parquet")),
        "DataFrame.to_parquet": lambda t: bars.to_parquet(t / "p.parquet"),
    }


def test_guard_refuses_writes_inside_the_store_and_allows_them_elsewhere(data_store_guard, frames, tmp_path,
                                                                          monkeypatch):
    bars, macro_frames = frames
    fake = tmp_path / "data_store"
    monkeypatch.setattr(data_store_guard, "root", fake)
    for name, write in _writers(bars, macro_frames).items():
        with pytest.raises(data_store_guard.error, match="write to tmp_path"):
            write(fake / "sub")
        assert not fake.exists(), f"{name} touched the disk before refusing"
    # relative paths resolve against the cwd; a symlink into the store is resolved too
    monkeypatch.chdir(tmp_path)
    with pytest.raises(data_store_guard.error):
        save_bars(bars, "data_store/rel.parquet")
    (tmp_path / "link").symlink_to(fake, target_is_directory=True)
    with pytest.raises(PermissionError):                 # DataStoreWriteError is a PermissionError
        bars.to_parquet(tmp_path / "link" / "x.parquet")
    assert not fake.exists()
    assert len(data_store_guard.attempts) == len(_writers(bars, macro_frames)) + 2
    data_store_guard.attempts.clear()                    # expected refusals: do not fail this test

    # a sibling whose name only starts with "data_store", and tmp_path itself, are fine
    out = save_bars(bars, tmp_path / "data_store2" / "ok.parquet")
    pd.testing.assert_frame_equal(load_bars(out), bars, check_freq=False)
    assert sorted(macro.load_macro_dir(save_macro_dir(macro_frames, tmp_path / "macro"))) == sorted(macro_frames)
    bars.to_parquet(tmp_path / "plain.parquet")
    assert not data_store_guard.attempts


def test_guard_protects_the_repository_data_store(data_store_guard, monkeypatch):
    """The suite-wide guard points at the repository's real data_store/. No writer is aimed at
    it here (if the guard ever regressed, that would write real files): the test above already
    proves that every writer consults this guard object before touching the disk."""
    root = REPO / "data_store"
    assert Path(data_store_guard.root) == root
    inside = [root, root / "_conftest_guard_probe", root / "macro" / "x.parquet", str(root / "a.parquet"),
              os.fsencode(root / "b.parquet"), root / "sub" / ".." / "c.parquet"]
    for target in inside:
        assert data_store_guard.contains(target), target
        with pytest.raises(data_store_guard.error, match="write to tmp_path"):
            data_store_guard.check(target, "probe")
    monkeypatch.chdir(REPO)                              # relative paths resolve against the cwd
    assert data_store_guard.contains("data_store/xauusd_H1.parquet")
    for target in (REPO / "data_store2" / "x.parquet", REPO / "runs" / "x.parquet", REPO, None, b"", 3):
        assert not data_store_guard.contains(target), target
    assert len(data_store_guard.attempts) == len(inside)
    data_store_guard.attempts.clear()                    # expected refusals: do not fail this test
    assert not (root / "_conftest_guard_probe").exists()


def test_guarded_writers_keep_their_signatures():
    for func in (save_frame, save_bars, save_macro_dir, macro.save_frame, dukascopy.build_dataset,
                 pq.write_table, pd.DataFrame.to_parquet):
        assert getattr(func, "__aurum_data_store_guard__", False), func
        assert inspect.signature(func) == inspect.signature(func.__wrapped__)
