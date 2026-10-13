"""Wiring of the calm-market filter and the position-value cap in execution/live_multi.py (added 2026-10-13).
Skipped until the trader has them (MULTI_CALM_ONLY_SYMBOLS / MULTI_NOTIONAL_CAP_SYMBOLS)."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from execution import live_multi as lm

pytestmark = pytest.mark.skipif(not hasattr(lm, "CALM_ONLY_SYMBOLS"), reason="trader patch not applied yet")


class Gw:
    """H1 bars at 84,000 whose last `burst` bars have `burst_range` wide candles."""
    def __init__(self, burst_range=100.0, short=False, positions=()):
        self.burst_range, self.short, self._pos, self.asked = burst_range, short, list(positions), []

    def get_rates(self, s, tf, n):
        self.asked.append((tf, n))
        n = min(n, 150) if self.short else n
        t = pd.date_range(end="2026-10-12 12:00", periods=n, freq="h", tz="UTC")
        r = np.r_[np.full(n - 20, 100.0), np.full(20, self.burst_range)]
        return pd.DataFrame({"time": t, "open": 84000.0, "high": 84000 + r / 2, "low": 84000 - r / 2, "close": 84000.0})

    def get_tick(self, s):
        return {"bid": 84000.0, "ask": 84001.0, "spread": 1.0, "age_seconds": 1}

    def calc_value_per_unit(self, s, d, px):
        return 1.0

    def account_info(self):
        return {"currency": "USD"}

    def positions(self):
        return self._pos


def _call(monkeypatch, gw, *, calm=frozenset({"BTCUSD"}), news_blocks=True, **kw):
    for k in ("LONG_ONLY_SYMBOLS", "REGIME_SYMBOLS", "STRONG_ONLY_SYMBOLS"):
        monkeypatch.setattr(lm, k, set())
    monkeypatch.setattr(lm, "CALM_ONLY_SYMBOLS", set(calm))
    monkeypatch.setattr(lm, "CALM_MAX_RATIO", 1.2)
    monkeypatch.setattr(lm.btc_h1_signal, "generate", lambda s, done: NS(decision="BUY", signal_bar_utc="sig"))
    news = NS(blocking_event=lambda s, now: {"event": "CPI"}, is_blocked=lambda s, now: news_blocks)
    bars = gw.get_rates("BTCUSD", "H1", 400)
    due = bars.time.iloc[-2].isoformat()
    now = (bars.time.iloc[-1] + pd.Timedelta(minutes=5)).to_pydatetime()
    return lm._evaluate_symbol(gw, None, kw.pop("c", None), "BTCUSD", kw.pop("spec", object()), "Crypto", due, now,
                               kw.pop("equity", 15000.0), news, kw.pop("gst", None), False, {}, **kw)


def test_jumpy_market_blocks_the_entry(monkeypatch):
    gw = Gw(burst_range=400.0)
    res = _call(monkeypatch, gw)
    assert res["decision"] == "blocked" and res["final"] is True and res["reason"].startswith("jumpy market")
    assert ("H1", lm.calm_filter.H1_BARS) in gw.asked                       # the filter reads the longer history


def test_calm_market_passes_to_the_next_check(monkeypatch):
    assert "news" in _call(monkeypatch, Gw(burst_range=100.0)).get("reason", "")


def test_other_symbols_are_not_filtered(monkeypatch):
    gw = Gw(burst_range=400.0)
    assert "news" in _call(monkeypatch, gw, calm=frozenset({"ETHUSD"})).get("reason", "")
    assert ("H1", lm.calm_filter.H1_BARS) not in gw.asked


def test_too_little_history_is_retried_not_traded(monkeypatch):
    res = _call(monkeypatch, Gw(short=True))
    assert res["decision"] == "retry" and not res.get("final") and "calm filter" in res["reason"]


def _spec():
    return NS(stops_level_price=0.0, value_per_price_unit_per_lot=1.0, currency_profit="USD", contract_size=1.0,
              volume_min=0.01, volume_step=0.01, volume_max=100.0)


def _sized(monkeypatch, tmp_path, *, cap, cap_used=0.0, equity=15000.0):
    """A dry-run pass through sizing: the ATR is about 100, so the stop is 200 away and 0.35 % of $15k wants
    0.26 BTC = $21,840 of position."""
    monkeypatch.setattr(lm, "NOTIONAL_CAP_SYMBOLS", {"BTCUSD", "ETHUSD"} if cap else set())
    monkeypatch.setattr(lm, "NOTIONAL_CAP", cap)
    monkeypatch.setattr(lm, "STATE", tmp_path)
    c = lm._conn(tmp_path / "t.sqlite")
    now = pd.Timestamp("2026-10-12 12:05", tz="UTC").to_pydatetime()
    gst = lm.guard_state(c, [], {}, equity, equity, now)
    try:
        return _call(monkeypatch, Gw(), calm=frozenset(), news_blocks=False, c=c, spec=_spec(), gst=gst, equity=equity,
                     risk=0.0035, cap_used=cap_used)
    finally:
        c.close()


def test_position_value_is_capped_and_the_risk_shrinks_with_it(monkeypatch, tmp_path):
    free = _sized(monkeypatch, tmp_path, cap=0.0)
    assert free["decision"] == "dry_run" and free["plan"]["volume"] == pytest.approx(0.26, abs=0.011)
    capped = _sized(monkeypatch, tmp_path, cap=0.6)
    assert capped["decision"] == "dry_run" and capped["plan"]["volume"] == 0.10          # $9,000 / $84,001
    assert capped["plan"]["risk_usd"] == pytest.approx(free["plan"]["risk_usd"] * 0.10 / free["plan"]["volume"], rel=0.01)


def test_open_positions_count_against_the_cap(monkeypatch, tmp_path):
    assert _sized(monkeypatch, tmp_path, cap=0.6, cap_used=5000.0)["plan"]["volume"] == 0.04   # $4,000 left
    full = _sized(monkeypatch, tmp_path, cap=0.6, cap_used=8990.0)
    assert full["decision"] == "skip" and full["final"] is True and "position value cap" in full["reason"]


# ---- reversal exit (MULTI_REVERSAL_EXIT_SYMBOLS) -------------------------------------------------
class RevGw:
    """Quiet bars at 84,000 (ATR about 100) whose last completed bar is `last` (open, high, low, close)."""
    def __init__(self, last):
        self.last = last

    def get_rates(self, s, tf, n):
        t = pd.date_range(end="2026-10-12 12:00", periods=n, freq="h", tz="UTC")           # 12:00 is still forming
        df = pd.DataFrame({"time": t, "open": 84000.0, "high": 84050.0, "low": 83950.0, "close": 84000.0})
        df.loc[n - 2, ["open", "high", "low", "close"]] = self.last
        return df


def _rev(monkeypatch, tmp_path, last, close_ok=True, entry=83700.0, atr=100.0):
    monkeypatch.setattr(lm, "STATE", tmp_path)
    monkeypatch.setattr(lm, "_REVERSAL_LOGGED", set())
    closed = []
    monkeypatch.setattr(lm, "_close", lambda gw, spec, p, reason, c: closed.append((p["ticket"], reason)) or True)
    c = lm._conn(tmp_path / "t.sqlite")
    c.execute("INSERT OR REPLACE INTO trades(ticket, symbol, opened_utc, direction, volume, entry, sl, tp, atr) VALUES(?,?,?,?,?,?,?,?,?)",
              (7, "BTCUSD", "2026-10-12T05:00:10+00:00", "BUY", 0.05, entry, entry - 2 * atr, entry + 6 * atr, atr))
    c.commit()
    now = pd.Timestamp("2026-10-12 12:00:20", tz="UTC").to_pydatetime()
    pos = {"ticket": 7, "symbol": "BTCUSD", "type": "BUY", "volume": 0.05}
    try:
        out = [lm._reversal_exit(RevGw(last), c, pos, NS(symbol="BTCUSD"), now, close_ok) for _ in range(2)]
        events = [r[0] for r in c.execute("SELECT kind FROM events")]
    finally:
        c.close()
    return out, closed, events


def test_reversal_exit_closes_a_winner_on_a_big_candle_against(monkeypatch, tmp_path):
    out, closed, _ = _rev(monkeypatch, tmp_path, (84150.0, 84160.0, 83950.0, 83960.0))     # 210 tall, down, 2.6 ATR up
    assert out[0] is True and closed[0] == (7, "reversal")


def test_reversal_exit_leaves_other_trades_alone(monkeypatch, tmp_path):
    assert _rev(monkeypatch, tmp_path, (84000.0, 84050.0, 83950.0, 84000.0))[1] == []      # an ordinary candle
    assert _rev(monkeypatch, tmp_path, (84150.0, 84160.0, 83950.0, 83960.0), entry=83900.0)[1] == []   # only 0.6 ATR up


def test_reversal_exit_in_a_dry_run_is_logged_once_and_closes_nothing(monkeypatch, tmp_path):
    out, closed, events = _rev(monkeypatch, tmp_path, (84150.0, 84160.0, 83950.0, 83960.0), close_ok=False)
    assert out == [False, False] and closed == [] and events.count("would_close") == 1

