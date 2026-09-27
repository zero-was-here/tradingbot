"""Tests for aurum.labels.triple_barrier (AFML ch. 3-4 labels and weights).

Hand-built price paths pin down every barrier rule (PT, SL, both-in-one-bar -> stop,
gaps, vertical barrier, side-awareness, undecided tail); a loop-based reference
implementation cross-checks the vectorised code on random bars; property tests check that
labels never read past the data end (truncation invariance) and that the uniqueness
weights match brute force.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import aurum.labels.triple_barrier as tb
from aurum.data.schema import make_bars
from aurum.data.synthetic import make_synthetic_bars
from aurum.labels import (
    LABEL_COLUMNS,
    average_uniqueness,
    cusum_filter,
    drop_label_tail,
    ewm_vol,
    fixed_horizon_labels,
    get_events,
    label_end_positions,
    meta_labels,
    num_concurrent_events,
    return_attribution_weights,
    time_decay_weights,
    triple_barrier_labels,
    uniqueness_weights,
)
from aurum.research.splits import walk_forward_splits

V = 0.01  # constant per-bar barrier unit (log return) for the hand-built paths
UP = 100.0 * math.exp(V)    # 101.005...
DN = 100.0 * math.exp(-V)   # 99.004...


def hand_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Bars from (open, high, low, close) tuples on a gap-free hourly grid."""
    idx = pd.date_range("2024-01-02 00:00", periods=len(rows), freq="h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    return make_bars(df, "H1", default_spread=0.3)


def only_first(bars: pd.DataFrame, **kw) -> pd.Series:
    """Label row of the event at bar 0 (barrier unit V)."""
    kw.setdefault("pt_mult", 1.0)
    kw.setdefault("sl_mult", 1.0)
    kw.setdefault("max_holding_bars", 3)
    lab = triple_barrier_labels(bars, vol=V, t_events=[0], **kw)
    assert len(lab) == 1
    return lab.iloc[0]


# ---------------------------------------------------------------------------------------
# hand-built paths
# ---------------------------------------------------------------------------------------
def test_profit_take_hit_first():
    bars = hand_bars([(100, 100, 100, 100), (100, 100.5, 99.6, 100.2), (100.2, 101.2, 100.0, 101.0),
                      (101, 101.1, 98.0, 98.5)])
    row = only_first(bars)
    assert row["barrier_hit"] == "pt"
    assert row["label"] == 1
    assert row["t1"] == bars.index[2] and row["t1_idx"] == 2
    assert row["ret"] == pytest.approx(V)
    assert row["exit_price"] == pytest.approx(UP)
    assert row["holding_bars"] == 2
    assert row["entry_price"] == 100.0


def test_stop_loss_hit_first():
    bars = hand_bars([(100, 100, 100, 100), (100, 100.5, 99.6, 100.2), (100.2, 100.4, 98.9, 99.0),
                      (99, 102.0, 98.9, 101.5)])
    row = only_first(bars)
    assert row["barrier_hit"] == "sl"
    assert row["label"] == -1
    assert row["t1_idx"] == 2
    assert row["ret"] == pytest.approx(-V)


def test_both_barriers_in_one_bar():
    bars = hand_bars([(100, 100, 100, 100), (100.1, 101.5, 98.5, 100.0), (100, 100.1, 99.9, 100.0),
                      (100, 100.1, 99.9, 100.0)])
    # no side: the order inside the bar is unknown -> ambiguous, label 0
    row = only_first(bars)
    assert row["barrier_hit"] == "ambiguous" and row["label"] == 0 and row["t1_idx"] == 1
    # long: conservative -> the stop (lower barrier) is assumed first
    row = only_first(bars, side=1.0)
    assert row["barrier_hit"] == "sl" and row["label"] == -1
    assert row["ret"] == pytest.approx(-V)
    assert row["exit_price"] == pytest.approx(DN)
    # short: the stop is the UPPER barrier
    row = only_first(bars, side=-1.0)
    assert row["barrier_hit"] == "sl" and row["label"] == -1
    assert row["ret"] == pytest.approx(-V)
    assert row["exit_price"] == pytest.approx(UP)


def test_gap_through_a_barrier_hits_it_first_at_the_open():
    # bar 1 opens above the upper barrier; its low also crosses the lower one
    bars = hand_bars([(100, 100, 100, 100), (101.5, 101.8, 98.5, 99.0), (99, 99.1, 98.9, 99.0),
                      (99, 99.1, 98.9, 99.0)])
    row = only_first(bars)
    assert row["barrier_hit"] == "pt" and row["label"] == 1
    assert row["exit_price"] == pytest.approx(101.5)
    assert row["ret"] == pytest.approx(math.log(1.015))
    # for a short the upper barrier is the stop: gap-through fills at the (worse) open
    row = only_first(bars, side=-1.0)
    assert row["barrier_hit"] == "sl" and row["label"] == -1
    assert row["ret"] == pytest.approx(-math.log(1.015))
    assert row["ret"] < -V


def test_vertical_barrier():
    bars = hand_bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100.3), (100.3, 100.8, 99.8, 100.6),
                      (100.6, 100.9, 100.1, 100.7), (100.7, 110, 90, 105)])
    row = only_first(bars)
    assert row["barrier_hit"] == "vertical"
    assert row["label"] == 0
    assert row["t1_idx"] == 3  # max_holding_bars = 3; bar 4's huge range is never seen
    assert row["ret"] == pytest.approx(math.log(1.007))
    assert row["exit_price"] == pytest.approx(100.7)
    row = only_first(bars, vertical_label="sign")
    assert row["label"] == 1
    row = only_first(bars, vertical_label="sign", min_ret=0.01)
    assert row["label"] == 0  # |ret| = 0.7% inside the dead band


def test_side_aware_short_profit_take():
    bars = hand_bars([(100, 100, 100, 100), (100, 100.3, 99.5, 99.7), (99.7, 99.8, 98.8, 98.9),
                      (98.9, 99, 98, 98.5)])
    row = only_first(bars, side=-1.0)
    assert row["barrier_hit"] == "pt" and row["label"] == 1
    assert row["ret"] == pytest.approx(V)
    assert row["side"] == -1
    # asymmetric widths: a short's PT is pt_mult below, SL is sl_mult above
    row = only_first(bars, side=-1.0, pt_mult=3.0, sl_mult=0.5)
    assert row["barrier_hit"] == "vertical"


def test_disabled_barriers_and_zero_side_events():
    bars = hand_bars([(100, 100, 100, 100), (100, 100.3, 98.0, 98.5), (98.5, 102.0, 98.4, 101.8),
                      (101.8, 102, 101, 101.5)])
    row = only_first(bars, sl_mult=None)
    assert row["barrier_hit"] == "pt" and row["t1_idx"] == 2
    row = only_first(bars, pt_mult=0.0, sl_mult=0.0)
    assert row["barrier_hit"] == "vertical"
    side = pd.Series([0.0, 1.0, np.nan, -1.0], index=bars.index)
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=2, vol=V, side=side,
                                drop_incomplete=False)
    assert lab["t_idx"].tolist() == [1, 3]  # side 0 / NaN are not bets
    with pytest.raises(ValueError):
        triple_barrier_labels(bars, pt_mult=-1, sl_mult=1, max_holding_bars=2, vol=V)


def test_undecided_tail_is_dropped_or_flagged():
    bars = hand_bars([(100, 100, 100, 100), (100, 100.2, 99.8, 100.0), (100, 100.2, 99.8, 100.1)])
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=5, vol=V)
    assert lab.empty
    assert list(lab.columns) == LABEL_COLUMNS
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=5, vol=V, drop_incomplete=False)
    assert (lab["barrier_hit"] == "incomplete").all()
    assert lab["label"].isna().all() and lab["ret"].isna().all()
    assert (lab["t1_idx"] <= len(bars) - 1).all()
    # an event near the end that DID touch before the end is decided and kept
    bars2 = hand_bars([(100, 100, 100, 100), (100, 101.5, 99.9, 101.2), (101.2, 101.3, 101.0, 101.1)])
    lab2 = triple_barrier_labels(bars2, pt_mult=1, sl_mult=1, max_holding_bars=5, vol=V, t_events=[0])
    assert lab2["barrier_hit"].tolist() == ["pt"]


# ---------------------------------------------------------------------------------------
# reference implementation and properties on random bars
# ---------------------------------------------------------------------------------------
def reference_labels(bars, vol, pt, sl, h, side=None):
    """Plain-loop triple barrier (same conventions) for cross-checking."""
    o, hi, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(bars)
    out = {}
    for t in range(n):
        s = 1.0 if side is None else float(np.sign(side[t]))
        if s == 0 or not np.isfinite(vol[t]):
            continue
        up_m, dn_m = (pt, sl) if s > 0 else (sl, pt)
        upper, lower = c[t] * math.exp(up_m * vol[t]), c[t] * math.exp(-dn_m * vol[t])
        res = None
        for j in range(t + 1, min(t + h, n - 1) + 1):
            hu, hd = hi[j] >= upper, lo[j] <= lower
            if not (hu or hd):
                continue
            if hu and hd:
                if o[j] >= upper:
                    hd = False
                elif o[j] <= lower:
                    hu = False
                elif side is None:
                    res = (j, 0.0, "ambiguous")
                    break
                else:
                    hu, hd = (s < 0), (s > 0)  # stop first
            if hu:
                px = max(upper, o[j])
            else:
                px = min(lower, o[j])
            r = s * math.log(px / c[t])
            is_pt = (hu and s > 0) or (hd and s < 0)
            res = (j, 1.0 if is_pt else -1.0, "pt" if is_pt else "sl", r)
            break
        if res is None:
            if t + h > n - 1:
                continue  # undecided
            res = (t + h, 0.0, "vertical", s * math.log(c[t + h] / c[t]))
        elif res[2] == "ambiguous":
            res = (res[0], 0.0, "ambiguous", math.log(c[res[0]] / c[t]))
        out[t] = res
    return out


@pytest.mark.parametrize("use_side", [False, True])
def test_vectorised_matches_reference_on_random_bars(use_side, monkeypatch):
    bars = make_synthetic_bars(900, "H1", seed=11, model="jump")
    vol = ewm_vol(bars["close"], span=50).to_numpy() * 2.0
    rng = np.random.default_rng(3)
    side = rng.choice([-1.0, 0.0, 1.0], size=len(bars)) if use_side else None
    monkeypatch.setattr(tb, "_CHUNK_ELEMENTS", 37)  # force many small chunks
    lab = triple_barrier_labels(bars, pt_mult=1.5, sl_mult=0.7, max_holding_bars=8, vol=vol,
                                side=side)
    ref = reference_labels(bars, vol, 1.5, 0.7, 8, side)
    assert sorted(ref) == lab["t_idx"].tolist()
    got = lab.set_index("t_idx")
    for t, (j, label, hit, r) in ref.items():
        row = got.loc[t]
        assert row["t1_idx"] == j and row["label"] == label and row["barrier_hit"] == hit, t
        assert row["ret"] == pytest.approx(r, abs=1e-12)
    assert set(lab["barrier_hit"]) >= {"pt", "sl", "vertical"}


def test_labels_never_read_past_the_data_end():
    """Labels computed on a prefix equal the full-sample labels of the same events: a label
    depends only on bars up to its t1 (the future beyond the frame is never needed)."""
    bars = make_synthetic_bars(1500, "H1", seed=4, model="regime")
    vol = ewm_vol(bars["close"]) * math.sqrt(12)
    full = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=12, vol=vol)
    for m in (300, 777, 1200):
        part = triple_barrier_labels(bars.iloc[:m], pt_mult=1, sl_mult=1, max_holding_bars=12,
                                     vol=vol.iloc[:m])
        assert (part["t1_idx"] <= m - 1).all()
        assert (part["t1"] <= bars.index[m - 1]).all()
        pd.testing.assert_frame_equal(part, full.loc[part.index])
        # everything decided inside the prefix is present
        decided = full[full["t1_idx"] <= m - 1]
        assert decided.index.equals(part.index)


def test_get_events_and_min_target():
    bars = make_synthetic_bars(300, "H1", seed=1)
    vol = ewm_vol(bars["close"], span=20)
    ev = get_events(bars, vol=vol, max_holding_bars=5)
    assert ev["trgt"].notna().all()
    assert (ev["t1_max_idx"] - ev["t_idx"] == 5).all()
    assert (ev["side"] == 1).all()
    ev2 = get_events(bars, vol=vol, max_holding_bars=5, min_target=float(vol.median()))
    assert len(ev2) < len(ev)
    mask = pd.Series(False, index=bars.index)
    mask.iloc[[50, 60]] = True
    assert get_events(bars, vol=vol, max_holding_bars=5, t_events=mask)["t_idx"].tolist() == [50, 60]
    assert get_events(bars, vol=vol, max_holding_bars=5,
                      t_events=bars.index[[70, 50]])["t_idx"].tolist() == [50, 70]
    with pytest.raises(KeyError):
        get_events(bars, vol=vol, max_holding_bars=5, t_events=[pd.Timestamp("1999-01-01", tz="UTC")])
    with pytest.raises(ValueError):
        get_events(bars, vol=vol, max_holding_bars=0)


# ---------------------------------------------------------------------------------------
# fixed horizon, meta labels, purging helpers
# ---------------------------------------------------------------------------------------
def test_fixed_horizon_labels():
    bars = make_synthetic_bars(400, "H1", seed=2)
    vol = ewm_vol(bars["close"], span=30)
    lab = fixed_horizon_labels(bars, 6, vol=vol, threshold=0.5)
    c = bars["close"].to_numpy()
    t = lab["t_idx"].to_numpy()
    np.testing.assert_allclose(lab["ret"], np.log(c[t + 6] / c[t]))
    np.testing.assert_allclose(lab["ret_norm"], lab["ret"] / (vol.to_numpy()[t] * math.sqrt(6)))
    assert ((lab["label"] == 0) == (lab["ret_norm"].abs() <= 0.5)).all()
    assert (lab["t1_idx"] == t + 6).all() and lab["t1_idx"].max() <= len(bars) - 1
    assert (lab["barrier_hit"] == "vertical").all()
    raw = fixed_horizon_labels(bars, 1)
    assert len(raw) == len(bars) - 1


def test_meta_labels_mapping():
    lab = pd.DataFrame({"barrier_hit": ["pt", "sl", "vertical", "vertical", "ambiguous"],
                        "ret": [0.01, -0.01, 0.002, -0.001, 0.0]})
    np.testing.assert_array_equal(meta_labels(lab).to_numpy(), [1, 0, 0, 0, np.nan])
    np.testing.assert_array_equal(meta_labels(lab, vertical="return_sign").to_numpy(), [1, 0, 1, 0, np.nan])
    with pytest.raises(ValueError):
        meta_labels(lab, vertical="nope")


def test_label_end_positions_feed_walk_forward_purging():
    bars = make_synthetic_bars(600, "H1", seed=8)
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=10,
                                vol=ewm_vol(bars["close"]) * math.sqrt(10))
    le = label_end_positions(lab, len(bars))
    assert le.shape == (len(bars),)
    assert (le >= np.arange(len(bars))).all()
    assert (le[lab["t_idx"].to_numpy()] == lab["t1_idx"].to_numpy()).all()
    splits = walk_forward_splits(len(bars), 300, 100, label_end=le)
    for tr, te in splits:
        assert (le[tr] < te[0]).all()  # no training label resolves inside the test block
    tail = drop_label_tail(lab, len(bars), 10)
    assert tail["t_idx"].max() <= len(bars) - 11


# ---------------------------------------------------------------------------------------
# uniqueness and weights
# ---------------------------------------------------------------------------------------
def test_uniqueness_hand_example():
    t0, t1 = np.array([0, 1, 6]), np.array([3, 4, 8])
    c = num_concurrent_events(9, t0, t1)
    np.testing.assert_array_equal(c, [0, 1, 2, 2, 1, 0, 0, 1, 1])
    u = average_uniqueness(t0, t1, n_bars=9)
    np.testing.assert_allclose(u, [2 / 3, 2 / 3, 1.0])


def test_uniqueness_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 200
    t0 = np.sort(rng.integers(0, n - 20, 80))
    t1 = t0 + rng.integers(1, 20, 80)
    conc = np.zeros(n)
    for a, b in zip(t0, t1, strict=True):
        conc[a + 1: b + 1] += 1
    expect = np.array([np.mean(1.0 / conc[a + 1: b + 1]) for a, b in zip(t0, t1, strict=True)])
    np.testing.assert_allclose(average_uniqueness(t0, t1, n_bars=n), expect, rtol=1e-12)
    np.testing.assert_array_equal(num_concurrent_events(n, t0, t1), conc)


def test_weight_series_helpers():
    bars = make_synthetic_bars(500, "H1", seed=5)
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=8,
                                vol=ewm_vol(bars["close"]) * math.sqrt(8))
    u = uniqueness_weights(lab, n_bars=len(bars))
    assert u.index.equals(lab.index) and u.mean() == pytest.approx(1.0)
    w = return_attribution_weights(lab, bars["close"])
    assert w.mean() == pytest.approx(1.0) and (w >= 0).all()
    # brute-force return attribution
    t0, t1 = lab["t_idx"].to_numpy(), lab["t1_idx"].to_numpy()
    r = np.r_[0.0, np.diff(np.log(bars["close"].to_numpy()))]
    c = num_concurrent_events(len(bars), t0, t1)
    raw = np.array([abs(np.sum(r[a + 1: b + 1] / c[a + 1: b + 1])) for a, b in zip(t0, t1, strict=True)])
    np.testing.assert_allclose(w.to_numpy(), raw / raw.mean(), rtol=1e-10)


def test_time_decay_weights():
    u = np.full(10, 0.5)
    np.testing.assert_allclose(time_decay_weights(u, 1.0), np.ones(10))
    d = time_decay_weights(u, 0.5)
    assert d[-1] == pytest.approx(1.0) and np.all(np.diff(d) > 0)
    assert d[0] == pytest.approx(0.5 + 0.5 * 0.1)  # linear in cumulative uniqueness
    z = time_decay_weights(u, -0.5)
    assert (z[:5] == 0).all() and z[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        time_decay_weights(u, 1.5)


# ---------------------------------------------------------------------------------------
# causal helpers
# ---------------------------------------------------------------------------------------
def test_ewm_vol_and_cusum_are_causal():
    bars = make_synthetic_bars(800, "H1", seed=9, model="regime")
    full_v = ewm_vol(bars["close"], span=40)
    part_v = ewm_vol(bars["close"].iloc[:500], span=40)
    pd.testing.assert_series_equal(part_v, full_v.iloc[:500])
    thr = 2.0 * full_v
    ev_full = cusum_filter(bars["close"], thr)
    ev_part = cusum_filter(bars["close"].iloc[:500], thr.iloc[:500])
    assert ev_part.equals(ev_full[ev_full <= bars.index[499]])
    assert 0 < len(ev_full) < len(bars) / 2


def test_cusum_hand_path():
    close = pd.Series(np.exp(np.cumsum([0, 0.004, 0.004, 0.004, -0.001, -0.009, -0.003, 0.0])),
                      index=pd.date_range("2024-01-02", periods=8, freq="h", tz="UTC"))
    ev = cusum_filter(close, 0.01)
    # +0.012 at bar 3 triggers (reset), then -0.013 cumulative down at bar 5
    assert list(close.index.get_indexer(ev)) == [3, 5]


# ---------------------------------------------------------------------------------------
# adversarial review
# ---------------------------------------------------------------------------------------
def test_two_step_afml_api_keeps_the_side():
    """get_events(side=...) -> apply_triple_barrier(...) must label the given bets exactly
    like the one-step call (it used to silently drop the side: ~70% of short labels wrong)."""
    bars = make_synthetic_bars(600, "H1", seed=3)
    vol = ewm_vol(bars["close"]) * 2.0
    rng = np.random.default_rng(1)
    side = pd.Series(rng.choice([-1.0, 1.0], len(bars)), index=bars.index)
    one = triple_barrier_labels(bars, pt_mult=1.5, sl_mult=0.7, max_holding_bars=4, vol=vol, side=side)
    ev = get_events(bars, vol=vol, max_holding_bars=4, side=side)
    assert ev.attrs["side_given"] is True
    pd.testing.assert_frame_equal(tb.apply_triple_barrier(bars, ev, pt_mult=1.5, sl_mult=0.7), one)
    # a hand-built events frame (no attrs) with short sides is recognised as bets too
    bare = pd.DataFrame(ev.to_dict("list"), index=ev.index)
    assert not bare.attrs
    pd.testing.assert_frame_equal(tb.apply_triple_barrier(bars, bare, pt_mult=1.5, sl_mult=0.7), one)
    # without a side the events carry +1 and are unsided labels
    ev0 = get_events(bars, vol=vol, max_holding_bars=4)
    assert ev0.attrs["side_given"] is False
    assert (tb.apply_triple_barrier(bars, ev0, pt_mult=1, sl_mult=1)["side"] == 0).all()


def test_integer_pandas_index_events_are_positions():
    bars = make_synthetic_bars(300, "H1", seed=1)
    vol = ewm_vol(bars["close"], span=20)
    got = get_events(bars, vol=vol, max_holding_bars=5, t_events=pd.Index([70, 50]))
    assert got["t_idx"].tolist() == [50, 70]
