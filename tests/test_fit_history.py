"""Walk-forward fit slices may reach back into older PAST bars for feature warm-up
(``Strategy.fit_history_bars``) but must never include any bar at/after the fold's test start."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_events
from aurum.research.walkforward import run_walk_forward
from aurum.strategies.base import Strategy
from aurum.strategies.ml import MLGBMStrategy
from tests.test_research_walkforward import _cfg


class WarmupSpy(Strategy):
    name = "wf_warmup_spy"
    trainable = True
    log: list = []
    lock = threading.Lock()

    @property
    def fit_history_bars(self) -> int:
        return 300

    def fit(self, md, features=None):
        with self.lock:
            self.log.append((md.bars.index[0], md.bars.index[-1]))
        self.is_fitted = True
        return self

    def generate(self, md, features=None):
        r = np.log(md.bars["close"]).diff()
        return self._finalize(np.tanh(r.rolling(24).sum() * 50), md.bars.index)


def test_fit_history_extends_back_but_never_into_test():
    bars = make_synthetic_bars(6000, "H1", seed=3, model="trend")
    md = MarketData(bars=bars, events=make_synthetic_events(bars.index[0], bars.index[-1]))
    WarmupSpy.log.clear()
    rep = run_walk_forward(md, _cfg(), strategies=[WarmupSpy()])
    idx = bars.index
    folds = rep.folds.reset_index(drop=True)
    assert len(WarmupSpy.log) >= len(folds)
    fits = sorted(WarmupSpy.log)[-len(folds):]
    for (first, last), (_, row) in zip(fits, folds.iterrows(), strict=True):
        tr0 = idx.get_loc(pd.Timestamp(row["train_start"]))
        assert last == pd.Timestamp(row["train_end"])              # ends exactly at train end
        assert first == idx[max(0, tr0 - 300)]                      # warm-up prepended
        assert last < pd.Timestamp(row["test_start"])               # never touches the test block


def test_ml_fit_history_equals_own_pipeline_warmup():
    s = MLGBMStrategy()
    assert s.fit_history_bars == s.warmup_bars > 0
