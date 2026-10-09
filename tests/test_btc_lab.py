"""Backtest-lab correctness: causality, cost monotonicity, no future info."""
from __future__ import annotations

import numpy as np
import pytest

from backtest import btc_lab as lab
from backtest import btc_strategies as strat


@pytest.fixture
def px(synth_ohlc):
    d = synth_ohlc
    return {k: d[k].to_numpy(float) for k in ("open", "high", "low", "close")}, d


def test_strategy_entries_are_causal(px):
    p, d = px
    c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    e_full, wu = strat.build_trend(c, h, l, fast=20, slow=50, trend=200, adx_min=0.0, atr_rank_max=1.0)
    cut = 500
    e_part, _ = strat.build_trend(c[:cut], h[:cut], l[:cut], fast=20, slow=50, trend=200,
                                  adx_min=0.0, atr_rank_max=1.0)
    # the entry decision for bar cut-1 must not use bars >= cut
    assert e_full[cut - 1] == e_part[cut - 1]


def test_backtest_runs_and_is_deterministic(px):
    p, d = px
    c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    entries, wu = strat.build_trend(c, h, l, adx_min=0.0, atr_rank_max=1.0)
    atr = strat.base_atr(h, l, c, 14)
    r1 = lab.run_backtest(p, atr, entries, wu + 5, len(c), wu, lab.Costs(), lab.RiskCfg())
    r2 = lab.run_backtest(p, atr, entries, wu + 5, len(c), wu, lab.Costs(), lab.RiskCfg())
    assert r1["summary"] == r2["summary"]


def test_higher_costs_never_improve_pnl(px):
    p, d = px
    c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    entries, wu = strat.build_breakout(c, h, l)
    atr = strat.base_atr(h, l, c, 14)
    cheap = lab.run_backtest(p, atr, entries, wu + 5, len(c), wu,
                             lab.Costs(spread=5, slippage_per_side=0.2), lab.RiskCfg())["summary"]
    dear = lab.run_backtest(p, atr, entries, wu + 5, len(c), wu,
                            lab.Costs(spread=60, slippage_per_side=4), lab.RiskCfg())["summary"]
    if cheap.get("trades", 0) and dear.get("trades", 0):
        assert dear["net_pl"] <= cheap["net_pl"] + 1e-6


def test_entry_bar_not_stop_checked(px):
    """A position opened at open[i] must not be stopped by bar i's own low/high."""
    p, d = px
    n = len(d)
    entries = np.zeros(n, np.int8)
    entries[300] = 1
    atr = np.full(n, 50.0)
    # craft: make bar 300 a huge down bar so its low would breach any nearby stop
    o = p["open"].copy(); h = p["high"].copy(); l = p["low"].copy(); c = p["close"].copy()
    l[300] = o[300] - 5000
    pp = {"open": o, "high": h, "low": l, "close": c}
    r = lab.run_backtest(pp, atr, entries, 260, n, 250, lab.Costs(), lab.RiskCfg())
    tr = r["trades"]
    if len(tr["entry_i"]):
        # if it entered at 300, the exit must be >= 301 (not stopped on the entry bar)
        assert tr["exit_i"][0] >= tr["entry_i"][0] + 1


def test_walk_forward_reports_every_fold(px):
    p, d = px
    c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    entries, wu = strat.build_trend(c, h, l, adx_min=0.0, atr_rank_max=1.0)
    atr = strat.base_atr(h, l, c, 14)
    wf = lab.walk_forward(p, atr, entries, wu, lab.Costs(), lab.RiskCfg(), n_folds=5)
    assert len(wf["folds"]) == 5
    assert all("range" in f for f in wf["folds"])


def test_monte_carlo_shape(px):
    p, d = px
    c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    entries, wu = strat.build_momentum(c, h, l)
    atr = strat.base_atr(h, l, c, 14)
    r = lab.run_backtest(p, atr, entries, wu + 5, len(c), wu, lab.Costs(), lab.RiskCfg())
    mc = lab.monte_carlo(r["trades"], lab.RiskCfg(), n=200)
    assert "n_trials" in mc or "note" in mc


_RISK = lab.RiskCfg(initial_balance=5000.0, risk_per_trade=0.01, max_risk_per_trade=0.02)


def _flat(n=400, px=50_000.0):
    o = np.full(n, px); h = o + 10; l = o - 10; c = o.copy()
    return {"open": o, "high": h, "low": l, "close": c}


def test_no_reentry_on_the_bar_a_trade_closed_inside():
    """Regression (2026-10-07): a trade that hits its target INSIDE bar i was still open at open[i];
    live trades one position per symbol, so no new trade may open at open[i]."""
    p = _flat()
    n = len(p["open"])
    atr = np.full(n, 100.0)
    entries = np.zeros(n, np.int8)
    entries[300] = 1
    entries[310] = 1                                  # signal on the bar where trade 1 exits
    p["high"][310] = p["open"][310] + 2000            # target hit inside bar 310
    r = lab.run_backtest(p, atr, entries, 260, n, 250, lab.Costs(), _RISK)
    tr = r["trades"]
    assert list(tr["entry_i"]) == [300] and list(tr["exit_i"]) == [310]


def test_reentry_allowed_when_previous_closed_at_the_open():
    """A gap through the stop exits AT open[i] -- the slot is free, as live."""
    p = _flat()
    n = len(p["open"])
    atr = np.full(n, 100.0)
    entries = np.zeros(n, np.int8)
    entries[300] = 1
    entries[310] = 1
    for k in ("open", "high", "low", "close"):
        p[k][310:] -= 1000                            # gap down through the stop at bar 310's open
    r = lab.run_backtest(p, atr, entries, 260, n, 250, lab.Costs(), _RISK)
    assert list(r["trades"]["entry_i"]) == [300, 310]
