"""
Data layer for the local dashboard.  READ-ONLY: reads status files, trade DBs (sqlite mode=ro),
reports and MT5 price history (copy_rates).  Never sends, modifies or closes orders.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.btc_strategies import _ema, _rsi, base_atr
from strategy.btc_h1_signal import PARAMS

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
S = ROOT / "state"
FILES = dict(btc_status=R / "btc_live_status.json", multi_status=R / "multi_live_status.json",
             challenge=R / "challenge_status.json", edge=R / "edge_monitor_multi.json",
             btc_db=S / "btc_live_H1.sqlite", multi_db=S / "multi_live.sqlite",
             notify_log=ROOT / "logs" / "notify.log", kill=S / "KILL_SWITCH")
SL_ATR, TP_ATR = 2.0, 3.0
_cache: dict = {}


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_min(st):
    try:
        return round((datetime.now(timezone.utc) - datetime.fromisoformat(st["generated_utc"])).total_seconds() / 60, 1)
    except Exception:
        return None


def _rows(db: Path, sql: str, args=()) -> list[dict]:
    if not db.exists():
        return []
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        cur = c.execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.Error:
        return []
    finally:
        c.close()


# -- signal readiness (pure, tested) -------------------------------------------------
def signal_readiness(df: pd.DataFrame) -> dict:
    """What the frozen rule says about the NEXT H1 bar, from COMPLETED bars only.

    BUY  needs RSI14 >= rsi_buy  AND 8-bar momentum > 0 AND close > EMA(ema_htf x 4)
    SELL needs RSI14 <= rsi_sell AND 8-bar momentum < 0 AND close < EMA(ema_htf x 4)
    """
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    rsi = float(_rsi(c, 14)[-1])
    mom = float(pd.Series(c).diff(PARAMS["mom_win"]).iloc[-1])
    ema = float(_ema(c, PARAMS["ema_htf"] * 4)[-1])
    atr = float(base_atr(h, l, c, 14)[-1])
    close = float(c[-1])
    buy = {"rsi": rsi >= PARAMS["rsi_buy"], "momentum": mom > 0, "trend": close > ema}
    sell = {"rsi": rsi <= PARAMS["rsi_sell"], "momentum": mom < 0, "trend": close < ema}
    decision = "BUY" if all(buy.values()) else "SELL" if all(sell.values()) else "HOLD"
    # which side is closer to triggering
    side = "BUY" if sum(buy.values()) >= sum(sell.values()) else "SELL"
    conds = buy if side == "BUY" else sell
    need = []
    if not conds["rsi"]:
        need.append(f"RSI {'≥' if side == 'BUY' else '≤'} {PARAMS['rsi_buy'] if side == 'BUY' else PARAMS['rsi_sell']:.0f} (now {rsi:.1f})")
    if not conds["momentum"]:
        need.append(f"8-bar momentum {'up' if side == 'BUY' else 'down'} (now {mom:+.5g})")
    if not conds["trend"]:
        need.append(f"close {'above' if side == 'BUY' else 'below'} EMA{PARAMS['ema_htf'] * 4} {ema:.5g} (now {close:.5g})")
    plan = None
    if atr > 0:
        d = 1 if side == "BUY" else -1
        plan = {"side": side, "entry": close, "sl": close - d * SL_ATR * atr, "tp": close + d * TP_ATR * atr}
    return {"decision": decision, "closest_side": side, "conditions": conds, "met": sum(conds.values()),
            "missing": need, "rsi": round(rsi, 1), "momentum": mom, "ema": ema, "atr": atr, "close": close,
            "signal_bar_utc": str(df["time"].iloc[-1]), "plan": plan}


def next_entry_window(now: datetime) -> dict:
    start = pd.Timestamp(now).floor("1h")
    into = (now - start.to_pydatetime()).total_seconds()
    if into <= 900:
        return {"open_now": True, "closes_utc": (start + pd.Timedelta(minutes=15)).isoformat()}
    nxt = start + pd.Timedelta(hours=1)
    return {"open_now": False, "opens_utc": nxt.isoformat(), "in_min": round((nxt - pd.Timestamp(now)).total_seconds() / 60, 1)}


# -- MT5 (read-only, cached) ----------------------------------------------------------
def market(symbols: list[str], bars: int = 300, ttl: float = 30.0) -> dict:
    key = ("market", tuple(symbols))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    out = {"connected": False, "symbols": {}}
    try:
        from mt5.gateway import MT5Gateway
        gw = MT5Gateway()
        if gw.connect():
            try:
                mt5 = gw.raw()
                a, t = mt5.account_info(), mt5.terminal_info()
                out.update(connected=True, account=dict(
                    login=a.login, server=a.server, type={0: "demo", 1: "contest", 2: "real"}.get(a.trade_mode, "?"),
                    balance=a.balance, equity=a.equity, margin_free=a.margin_free, currency=a.currency,
                    algo_terminal=bool(t.trade_allowed), algo_account=bool(a.trade_allowed)))
                now = datetime.now(timezone.utc)
                for s in symbols:
                    df = gw.get_rates(s, "H1", bars)
                    if df is None or df.empty:
                        out["symbols"][s] = {"error": "no rates"}
                        continue
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                    done = df[df["time"] < pd.Timestamp(now).floor("1h")].reset_index(drop=True)
                    tick = gw.get_tick(s) or {}
                    out["symbols"][s] = {
                        "candles": [{"time": int(r.time.timestamp()), "open": r.open, "high": r.high, "low": r.low,
                                     "close": r.close} for r in df.itertuples()],
                        "readiness": signal_readiness(done) if len(done) > 120 else {"error": "not enough bars"},
                        "bid": tick.get("bid"), "ask": tick.get("ask"), "spread": tick.get("spread"),
                        "tick_age_s": tick.get("age_seconds")}
            finally:
                gw.shutdown()
    except Exception as e:
        out["error"] = str(e)
    _cache[key] = (time.time(), out)
    return out


def _first_balance(ch, ms, closed) -> float:
    """Account balance before the first closed trade: current balance minus realised P/L."""
    now_bal = float((ms or {}).get("balance") or (ms or {}).get("equity") or (ch or {}).get("phase_start_balance") or 0)
    return now_bal - sum(t.get("pnl_usd") or 0 for t in closed)


def equity_curve(closed: list[dict], *, start_balance: float) -> list[dict]:
    """Realised balance after each closed trade (pure, tested)."""
    run, out = float(start_balance), []
    for t in sorted(closed, key=lambda x: x["closed_utc"]):
        run += t.get("pnl_usd") or 0
        out.append({"time": int(datetime.fromisoformat(t["closed_utc"]).timestamp()), "value": round(run, 2)})
    return out


# -- everything ---------------------------------------------------------------------------
def processes() -> dict:
    try:
        import psutil
        found = {"live_multi": False, "btc_live": False, "watchdog": False, "mt5": False}
        for p in psutil.process_iter(["name", "cmdline"]):
            n, c = (p.info["name"] or "").lower(), " ".join(p.info["cmdline"] or []).lower()
            found["live_multi"] |= "live_multi" in c
            found["btc_live"] |= "run_live.py" in c
            found["watchdog"] |= "tools.watchdog" in c
            found["mt5"] |= n == "terminal64.exe"
        return found
    except Exception as e:
        return {"error": str(e)}


def gather() -> dict:
    now = datetime.now(timezone.utc)
    ms, bs = _json(FILES["multi_status"]), _json(FILES["btc_status"])
    symbols = (ms or {}).get("symbols") or ["BTCUSD.vx", "XAUUSD.vx", "DAX40.vx"]
    trades = _rows(FILES["multi_db"], "SELECT * FROM trades ORDER BY opened_utc")
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("closed_utc")]
    ch = _json(FILES["challenge"])
    eq = equity_curve(closed, start_balance=(ch or {}).get("started_balance") or _first_balance(ch, ms, closed))
    try:
        from news.filter import NewsFilter
        news = [{"time_utc": e["time"].isoformat(), "currency": e["currency"], "event": e["event"]}
                for e in NewsFilter().upcoming(hours=72)]
    except Exception:
        news = []
    try:
        from tools import edge_monitor as em
        ref = R / "multi_reference_trades.csv"
        edge = em.evaluate(em.load_live_R(FILES["multi_db"]), em.load_reference_R(ref)) if ref.exists() else None
    except Exception:
        edge = None
    alerts = []
    if FILES["notify_log"].exists():
        with FILES["notify_log"].open(encoding="utf-8", errors="replace") as f:
            alerts = [ln.strip().split("NOTIFY ", 1)[-1][:300] for ln in deque(f, maxlen=15) if "NOTIFY" in ln][::-1]
    return {
        "generated_utc": now.isoformat(),
        "processes": processes(),
        "kill_switch": FILES["kill"].read_text(encoding="utf-8").strip()[:200] if FILES["kill"].exists() else None,
        "multi": {"status": ms, "age_min": _age_min(ms)},
        "btc": {"status": bs, "age_min": _age_min(bs)},
        "challenge": ch,
        "entry_window": next_entry_window(now),
        "market": market(symbols),
        "trades": trades[-100:],
        "equity_curve": eq,
        "news": news,
        "edge": edge,
        "alerts": alerts,
    }
