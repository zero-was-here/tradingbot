"""v2.1 core fixes - loaders (deterministic ``attrs['source']`` for buffers, server_tz default
warning), GARCH nominal annualisation, ``risk_report`` stress price (``last_price`` / ``bars``),
the FixedFractionalSizer stop contract, and the macro feature group's requirement metadata."""

from __future__ import annotations

import io
import logging
import math

import numpy as np
import pandas as pd
import pytest

from aurum.core.timeframes import index_bar_minutes, infer_bars_per_year, nominal_bars_per_year
from aurum.core.types import MarketData
from aurum.data.loaders import BUFFER_SOURCE_NAME, load_csv, load_mt5_csv, source_name
from aurum.data.synthetic import make_synthetic_bars, make_synthetic_macro
from aurum.models.volatility import Garch11, ewma_volatility
from aurum.risk.var import risk_report

MT5 = ("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
       "2024.01.02\t01:00:00\t2063.10\t2064.00\t2062.50\t2063.80\t1200\t0\t25\n"
       "2024.01.02\t02:00:00\t2063.80\t2065.20\t2063.10\t2064.90\t1300\t0\t24\n"
       "2024.01.02\t03:00:00\t2064.90\t2066.00\t2064.00\t2065.50\t1100\t0\t26\n")
CSV = ("time,open,high,low,close,spread\n"
       "2024-01-02 01:00,2063.1,2064.0,2062.5,2063.8,0.3\n"
       "2024-01-02 02:00,2063.8,2065.2,2063.1,2064.9,0.3\n")


# ---- loaders -----------------------------------------------------------------------------------------
def test_buffer_sources_are_deterministic(tmp_path):
    a = load_mt5_csv(io.StringIO(MT5), "H1", server_tz="NY+7")
    b = load_mt5_csv(io.StringIO(MT5), "H1", server_tz="NY+7")
    assert a.attrs["source"] == b.attrs["source"] == f"mt5:{BUFFER_SOURCE_NAME}"
    assert "0x" not in a.attrs["source"]
    assert load_mt5_csv(io.BytesIO(MT5.encode()), "H1", server_tz="UTC").attrs["source"] == "mt5:<buffer>"
    c = load_csv(io.StringIO(CSV), "H1")
    assert c.attrs["source"] == "csv:<buffer>"
    # named file objects and paths keep their file name
    p = tmp_path / "XAUUSD_H1.csv"
    p.write_text(MT5, encoding="utf-8")
    assert load_mt5_csv(p, "H1", server_tz="UTC").attrs["source"] == "mt5:XAUUSD_H1.csv"
    with open(p, encoding="utf-8") as fh:
        assert load_mt5_csv(fh, "H1", server_tz="UTC").attrs["source"] == "mt5:XAUUSD_H1.csv"
    named = io.StringIO(CSV)
    named.name = "/some/dir/export.csv"
    assert load_csv(named, "H1").attrs["source"] == "csv:export.csv"
    assert source_name(str(p)) == source_name(p) == "XAUUSD_H1.csv"


def test_default_server_tz_warns_but_is_unchanged(caplog):
    caplog.set_level(logging.WARNING, logger="aurum.data.loaders")
    default = load_mt5_csv(io.StringIO(MT5), "H1")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("NY+7" in m and "server_tz" in m for m in msgs)
    # identical to an explicit fixed UTC+2 (the documented default) ...
    caplog.clear()
    explicit = load_mt5_csv(io.StringIO(MT5), "H1", server_tz="Etc/GMT-2")
    pd.testing.assert_frame_equal(default, explicit)
    assert default.index[0] == pd.Timestamp("2024-01-01 23:00", tz="UTC")
    # ... and an explicit choice (even the same zone) does not warn
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    load_mt5_csv(io.StringIO(MT5), "H1", server_tz="NY+7")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---- GARCH annualisation -------------------------------------------------------------------------
def test_garch_annualises_like_ewma_volatility():
    bars = make_synthetic_bars(3000, "H1", seed=2)
    r = np.log(bars["close"]).diff().dropna()
    g = Garch11().fit(r)
    nominal = nominal_bars_per_year(index_bar_minutes(r.index))
    assert g.bars_per_year_ == pytest.approx(nominal) == pytest.approx(5796.0)
    assert g.bars_per_year_ != pytest.approx(infer_bars_per_year(r.index), rel=1e-3)
    ann = g.forecast(r)
    per_bar = g.forecast(r, annualise=False)
    np.testing.assert_allclose(ann.to_numpy(), per_bar.to_numpy() * math.sqrt(5796.0))
    assert g.summary()["unconditional_vol_ann"] == pytest.approx(
        math.sqrt(g.params_.unconditional_variance * 5796.0))
    # same level convention as the engine's default EWMA vol (both nominal): ratio ~1
    ew = ewma_volatility(bars["close"]).reindex(ann.index)
    assert 0.7 < float((ann / ew).iloc[500:].median()) < 1.4
    # an explicit bars_per_year still wins; a training subset gives the same constant
    assert Garch11(bars_per_year=252).fit(r).bars_per_year_ == 252
    assert Garch11().fit(r.iloc[:800]).bars_per_year_ == pytest.approx(5796.0)
    d1 = make_synthetic_bars(600, "D1", seed=3)
    rd = np.log(d1["close"]).diff().dropna()
    assert Garch11().fit(rd).bars_per_year_ == pytest.approx(252.0)


# ---- risk_report stress price ---------------------------------------------------------------------
def _result(bars: pd.DataFrame, lots: float = 0.5, last_fill: float | None = None):
    from aurum.backtest.result import BacktestResult

    idx = bars.index
    eq = pd.Series(100_000 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, len(idx)))), index=idx)
    pos = pd.Series(lots, index=idx)
    fills = pd.DataFrame({"price": [float(bars["close"].iloc[0]) if last_fill is None else last_fill]})
    return BacktestResult(equity=eq, returns=eq.pct_change().fillna(0.0), positions=pos,
                          costs=pd.DataFrame(index=idx), trades=pd.DataFrame(), fills=fills, meta={})


def test_risk_report_price_resolution():
    bars = make_synthetic_bars(800, "H1", seed=6)
    res = _result(bars, last_fill=1234.5)
    # no price information beyond the fills: the (stale) last fill, noted
    rep = risk_report(res)
    assert rep["position"]["price"] == 1234.5 and rep["position"]["price_source"] == "last_fill"
    assert any("last fill" in n for n in rep["notes"])
    # bars -> last close at/before the last equity mark (also through MarketData)
    rb = risk_report(res, bars=bars)
    assert rb["position"]["price"] == pytest.approx(float(bars["close"].iloc[-1]))
    assert rb["position"]["price_source"] == "bars_close"
    assert risk_report(res, bars=MarketData(bars=bars))["position"]["price"] == rb["position"]["price"]
    # bars that extend past the result are read as of the last equity mark (never later)
    short = _result(bars.iloc[:500], last_fill=1234.5)
    rs = risk_report(short, bars=bars)
    assert rs["position"]["price"] == pytest.approx(float(bars["close"].iloc[499]))
    # explicit last_price wins; the legacy alias still works; conflicting values raise
    rl = risk_report(res, bars=bars, last_price=2000.0)
    assert rl["position"]["price"] == 2000.0 and rl["position"]["price_source"] == "argument"
    assert risk_report(res, price=1999.0)["position"]["price"] == 1999.0
    assert risk_report(res, price=2000.0, last_price=2000.0)["position"]["price"] == 2000.0
    with pytest.raises(ValueError, match="disagree"):
        risk_report(res, price=1.0, last_price=2.0)
    # meta["last_price"] comes before bars
    res.meta["last_price"] = 1500.0
    rm = risk_report(res, bars=bars)
    assert rm["position"]["price"] == 1500.0 and rm["position"]["price_source"] == "meta"
    # notional / stress follow the chosen price
    assert rl["position"]["notional"] == pytest.approx(0.5 * 100 * 2000.0)


def test_risk_report_without_any_price_omits_the_stress_test():
    bars = make_synthetic_bars(300, "H1", seed=1)
    res = _result(bars)
    res.fills = pd.DataFrame(columns=["price"])
    rep = risk_report(res)
    assert "position" not in rep and any("no price" in n for n in rep["notes"])


# ---- sizing contract (documentation) --------------------------------------------------------------
def test_fixed_fractional_sizer_documents_the_external_stop():
    from aurum.portfolio.sizing import FixedFractionalSizer

    doc = FixedFractionalSizer.__doc__ or ""
    assert "stop_atr_mult" in doc and "external" in doc.lower()
    # behaviour unchanged: lots from risk_per_trade / (stop_atr * ATR)
    from aurum.core.instrument import XAUUSD

    s = FixedFractionalSizer(risk_per_trade=0.01, stop_atr=2.0, max_leverage=50.0, rebalance_band=0.0)
    lots = s.target_lots(1.0, 0.15, 100_000.0, 2000.0, XAUUSD, atr=10.0)
    assert lots == pytest.approx(0.01 * 100_000 / (2.0 * 10.0 * 100.0), abs=0.01)


# ---- macro feature group metadata -----------------------------------------------------------------
def test_macro_group_declares_its_requirement_without_changing_values():
    from aurum.features.base import get_feature
    from aurum.features.macro import MACRO_REQUIREMENT, macro_features
    from aurum.features.pipeline import FeaturePipeline

    spec = get_feature("macro")
    assert spec.fn is macro_features and spec.fn.requires_macro == MACRO_REQUIREMENT == "any"
    assert spec.requires_macro == ()   # enforced names: none (see MACRO_REQUIREMENT)
    bars = make_synthetic_bars(2500, "H1", seed=3)
    macro = make_synthetic_macro(bars, seed=3)
    # a FRED-only set (no dxy, as without yfinance) still produces its columns via the pipeline
    partial = {k: macro[k] for k in ("real10y",)}
    for m in (macro, partial):
        md = MarketData(bars=bars, macro=m)
        direct = macro_features(md)
        via = FeaturePipeline(groups=["macro"]).compute(md)
        assert len(direct.columns) > 0
        pd.testing.assert_frame_equal(via, direct.astype(float), check_freq=False)
    # no macro at all: no columns (unchanged)
    assert FeaturePipeline(groups=["macro"]).compute(MarketData(bars=bars)).shape[1] == 0
