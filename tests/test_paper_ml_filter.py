"""Signal-quality filter: feature code, model maths and the paper tracker (no MT5, no orders)."""
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from execution import paper_ml_filter as pm
from strategy import ml_filter as mf


def _h1(n=3200, seed=1, end="2026-10-09 12:00"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    t = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n, freq="h")
    return pd.DataFrame({"time": t, "open": np.r_[c[0], c[:-1]], "high": c * 1.002, "low": c * 0.998, "close": c})


def _d1(n=600, end="2026-10-09"):
    c = np.linspace(80, 120, n)
    return pd.DataFrame({"time": pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n, freq="D"), "close": c})


class FakeGW:
    def __init__(self, h1, d1):
        self.h1, self.d1, self.orders = h1, d1, 0

    def get_rates(self, symbol, timeframe, count=5000):
        return (self.h1 if timeframe == "H1" else self.d1).tail(count).reset_index(drop=True)


def test_shipped_model_matches_the_feature_list():
    m = mf.load_model()
    assert m is not None and m["features"] == mf.FEATS and len(m["coef"]) == len(mf.FEATS)
    assert m["coef"][mf.FEATS.index("vol_rel")] < 0 < m["coef"][mf.FEATS.index("rsi_s")]      # as in the study


def test_ridge_fit_recovers_a_linear_relation_and_threshold_is_the_median():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, len(mf.FEATS)))
    y = 0.5 * x[:, 0] - 0.3 * x[:, 3] + rng.normal(0, 0.1, 4000)
    m = mf.fit(x, y, alpha=1.0)
    assert abs(m["coef"][0] - 0.5) < 0.02 and abs(m["coef"][3] + 0.3) < 0.02
    p = mf.predict(m, x)
    assert abs((p > m["threshold"]).mean() - 0.5) < 0.01


def test_features_are_direction_signed_and_need_warm_up():
    bars = mf.bar_frame(_h1())
    last = bars.iloc[-1]
    buy = mf.features(last, 1, 110.0, 100.0, 13)
    sell = mf.features(last, -1, 110.0, 100.0, 13)
    assert buy["rsi_s"] == -sell["rsi_s"] and buy["trend_ok"] == 1.0 and sell["trend_ok"] == 0.0
    assert buy["vol_rel"] == sell["vol_rel"] > 0
    assert mf.features(bars.iloc[100], 1, 110.0, 100.0, 13) is None          # no volatility reference yet


def test_score_uses_only_bars_up_to_the_signal_bar():
    h1, d1, m = _h1(), _d1(), mf.load_model()
    sig = h1.time.iloc[-30]
    now = datetime(2026, 10, 9, 13, tzinfo=timezone.utc)
    a, note = pm.score_trade(FakeGW(h1, d1), m, "BTCUSD", sig.isoformat(), "BUY", now)
    later = h1.copy()
    later.loc[later.time > sig, ["open", "high", "low", "close"]] *= 3.0     # the future must not matter
    b, _ = pm.score_trade(FakeGW(later, d1), m, "BTCUSD", sig.isoformat(), "BUY", now)
    assert note == "ok" and a["score"] == b["score"]
    none, why = pm.score_trade(FakeGW(h1, d1), m, "BTCUSD", "2020-01-01T00:00:00+00:00", "BUY", now)
    assert none is None and "signal bar" in why


def test_paper_pass_scores_live_trades_once_and_reports_groups(tmp_path):
    h1, d1 = _h1(), _d1()
    live = tmp_path / "multi_live.sqlite"
    c = sqlite3.connect(live)
    c.execute("""CREATE TABLE trades(ticket INTEGER PRIMARY KEY, symbol TEXT, grp TEXT, opened_utc TEXT, signal_bar_utc TEXT,
        direction TEXT, volume REAL, entry REAL, sl REAL, tp REAL, atr REAL, risk_usd REAL, status TEXT, closed_utc TEXT,
        exit REAL, exit_reason TEXT, pnl_usd REAL, r_multiple REAL)""")
    for k, (d, st, r) in enumerate([("BUY", "CLOSED", 3.0), ("SELL", "CLOSED", -1.0), ("BUY", "OPEN", None)]):
        c.execute("INSERT INTO trades(ticket, symbol, signal_bar_utc, direction, status, r_multiple) VALUES(?,?,?,?,?,?)",
                  (k + 1, "BTCUSD", h1.time.iloc[-40 + 5 * k].isoformat(), d, st, r))
    c.commit()
    c.close()
    gw, now = FakeGW(h1, d1), datetime(2026, 10, 9, 13, tzinfo=timezone.utc)
    kw = dict(live_db=live, db=tmp_path / "p.sqlite", status=tmp_path / "s.json")
    s = pm.run_once(gw, now, **kw)
    assert s["scored"] == 3 and s["open_scored"] == 1 and s["taken"]["closed"] + s["skipped"]["closed"] == 2
    assert s["verdict"].startswith("keep watching") and gw.orders == 0
    first = sqlite3.connect(tmp_path / "p.sqlite").execute("SELECT ticket, score, scored_utc FROM scores ORDER BY ticket").fetchall()
    pm.run_once(gw, datetime(2026, 10, 9, 14, tzinfo=timezone.utc), **kw)       # a later pass keeps the first score
    assert sqlite3.connect(tmp_path / "p.sqlite").execute("SELECT ticket, score, scored_utc FROM scores ORDER BY ticket").fetchall() == first
    assert "verdict" in pm.report(tmp_path / "s.json") and json.loads((tmp_path / "s.json").read_text())["live_trades"] == 3


def test_verdict_needs_enough_trades_then_compares_the_groups():
    n = pm.MIN_CLOSED
    trades = pd.DataFrame({"ticket": range(n), "symbol": "BTCUSD", "signal_bar_utc": "x", "direction": "BUY",
                           "status": "CLOSED", "r_multiple": [1.0 if k % 2 else -1.0 for k in range(n)], "closed_utc": "y"})
    scores = pd.DataFrame({"ticket": range(n), "score": 0.1, "taken": [k % 2 for k in range(n)], "note": "ok"})
    now, m = datetime(2026, 11, 1, tzinfo=timezone.utc), {"threshold": 0.0}
    assert "helps" in pm.standing(trades, scores, now, m)["verdict"]
    assert "NOT" in pm.standing(trades, scores.assign(taken=1 - scores.taken), now, m)["verdict"]
    assert pm.standing(trades.head(10), scores.head(10), now, m)["verdict"].startswith("keep watching")
