"""
Data layer for the local dashboard.  READ-ONLY: reads status files, trade DBs (sqlite mode=ro),
reports and MT5 price history (copy_rates).  Never sends, modifies or closes orders.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.btc_strategies import _ema, _rsi, base_atr
from strategy import regime_filter
from strategy.btc_h1_signal import PARAMS

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
S = ROOT / "state"
FILES = dict(btc_status=R / "btc_live_status.json", multi_status=R / "multi_live_status.json",
             challenge=R / "challenge_status.json", edge=R / "edge_monitor_multi.json",
             btc_db=S / "btc_live_H1.sqlite", multi_db=S / "multi_live.sqlite",
             notify_log=ROOT / "logs" / "notify.log", kill=S / "KILL_SWITCH")
SL_ATR, TP_ATR = 2.0, 6.0          # must match execution/live_multi.py (target 3 -> 6 ATR on 2026-10-08)
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
def _needs(side: str, conds: dict, rsi: float, mom: float, ema: float, close: float) -> list[str]:
    """The conditions still missing for `side`, in words."""
    need = []
    if not conds["rsi"]:
        need.append(f"RSI {'≥' if side == 'BUY' else '≤'} {PARAMS['rsi_buy'] if side == 'BUY' else PARAMS['rsi_sell']:.0f} (now {rsi:.1f})")
    if not conds["momentum"]:
        need.append(f"8-bar momentum {'up' if side == 'BUY' else 'down'} (now {mom:+.5g})")
    if not conds["trend"]:
        need.append(f"close {'above' if side == 'BUY' else 'below'} EMA{PARAMS['ema_htf'] * 4} {ema:.5g} (now {close:.5g})")
    return need


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
    needs = {"BUY": _needs("BUY", buy, rsi, mom, ema, close), "SELL": _needs("SELL", sell, rsi, mom, ema, close)}
    plan = None
    if atr > 0:
        d = 1 if side == "BUY" else -1
        plan = {"side": side, "entry": close, "sl": close - d * SL_ATR * atr, "tp": close + d * TP_ATR * atr}
    return {"decision": decision, "closest_side": side, "conditions": conds, "met": sum(conds.values()),
            "missing": needs[side], "needs": needs, "rsi": round(rsi, 1), "momentum": mom, "ema": ema, "atr": atr, "close": close,
            "buy": buy, "sell": sell, "signal_bar_utc": str(df["time"].iloc[-1]), "plan": plan}


def forming_preview(df_all: pd.DataFrame, now: datetime) -> dict | None:
    """The same rule applied as if the still-forming H1 bar closed at the current price.
    Display only: the bot decides on completed bars, so this can change before the hour ends."""
    t = pd.to_datetime(df_all["time"], utc=True)
    if not len(df_all) or t.iloc[-1] < pd.Timestamp(now).floor("1h"):
        return None
    r = signal_readiness(df_all)
    return {k: r[k] for k in ("decision", "rsi", "momentum", "ema", "close", "buy", "sell")}


def order_blocks(symbol: str, decision: str | None, *, open_positions: list[dict], groups: dict,
                 max_per_group: int, regime: dict | None, long_only: bool = False) -> list[str]:
    """Why the trader would not place `decision` (BUY / SELL) on `symbol` right now -- the same
    checks as execution/live_multi.py: one position per pair, the per-group cap and the daily
    EMA200 trend filter.  Empty list = nothing blocks it."""
    out = []
    pos = next((p for p in open_positions if p.get("symbol") == symbol), None)
    if pos:
        out.append(f"{pos.get('type')} already open (one per pair)")
    grp = groups.get(symbol, "")
    others = [p["symbol"] for p in open_positions if p.get("symbol") != symbol and grp and groups.get(p.get("symbol")) == grp]
    if max_per_group and len(others) >= max_per_group:
        out.append(f"group limit: {', '.join(others)} already open in {grp} (max {max_per_group})")
    if regime:
        only = regime.get("only")
        if only is None:
            out.append(f"trend filter: {regime.get('why', 'no daily data')}")
        elif decision in ("BUY", "SELL") and decision != only:
            out.append(f"trend filter: {only.lower()}s only ({regime.get('why')})")
    if long_only and decision == "SELL":
        out.append("buy-only symbol (MULTI_LONG_ONLY_SYMBOLS)")
    return out


MARKET_CLOSED_S = 900          # no new price for 15 min = market closed (weekend, daily break, holiday)


def next_trade(symbol: str, d: dict, *, now: datetime, open_positions: list[dict], groups: dict,
               max_per_group: int, long_only: bool) -> dict:
    """When the bot next looks at `symbol`, which side it may take and what is still missing for it.
    The bot decides once per hour on the candle that just closed (first 15 min of the hour)."""
    nxt = pd.Timestamp(now).floor("1h") + pd.Timedelta(hours=1)
    out = {"next_check_utc": nxt.isoformat(), "side": None, "needs": [], "waiting": []}
    r = d.get("readiness") or {}
    if d.get("error") or r.get("error"):
        out["status"] = d.get("error") or r.get("error")
        return out
    sides = ["BUY"] if long_only else ["BUY", "SELL"]
    only = (d.get("regime") or {}).get("only", "") if d.get("regime") else ""
    if d.get("regime") and only is None:
        out["status"] = f"trend filter has no daily data ({d['regime'].get('why')})"
        return out
    if only:
        sides = [x for x in sides if x == only]
    if not sides:
        out["status"] = "no side allowed (buy-only symbol, daily trend down)"
        return out
    met = {x: sum((r.get(x.lower()) or {}).values()) for x in sides}
    side = r["decision"] if r.get("decision") in sides else max(sides, key=lambda x: met[x])
    kw = dict(open_positions=open_positions, groups=groups, max_per_group=max_per_group, regime=None)
    out.update(side=side, needs=(r.get("needs") or {}).get(side, []), met=met[side],
               waiting=order_blocks(symbol, side, **kw))
    age = d.get("tick_age_s")
    if age is not None and age > MARKET_CLOSED_S:
        out["status"] = "market closed -- checks resume on the first full hour after it reopens"
    elif r.get("decision") == side and not out["waiting"]:
        out["status"] = f"{side} signal on the last candle"
    elif out["waiting"]:
        out["status"] = "waiting: " + "; ".join(out["waiting"])
    else:
        out["status"] = f"{side} needs " + "; ".join(out["needs"])
    return out


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
                    m1 = gw.get_rates(s, "M1", 240)
                    info = mt5.symbol_info(s)
                    regime = None
                    if s in regime_filter.parse_symbols(os.getenv("MULTI_REGIME_SYMBOLS", "")):
                        ok, why = regime_filter.side200("BUY", gw.get_rates(s, "D1", regime_filter.D1_BARS), now)
                        regime = {"only": None if ok is None else "BUY" if ok else "SELL", "why": why}
                    out["symbols"][s] = {
                        "candles": [{"time": int(r.time.timestamp()), "open": r.open, "high": r.high, "low": r.low,
                                     "close": r.close} for r in df.itertuples()],
                        "candles_m1": [] if m1 is None or m1.empty else [
                            {"time": int(pd.Timestamp(r.time).timestamp()), "open": r.open, "high": r.high,
                             "low": r.low, "close": r.close} for r in m1.itertuples()],
                        "readiness": signal_readiness(done) if len(done) > 120 else {"error": "not enough bars"},
                        "forming": forming_preview(df, now) if len(done) > 120 else None,
                        "bid": tick.get("bid"), "ask": tick.get("ask"), "spread": tick.get("spread"),
                        "tick_age_s": tick.get("age_seconds"),
                        "group": (info.path or "").split("\\")[0] if info else "", "regime": regime}
            finally:
                gw.shutdown()
    except Exception as e:
        out["error"] = str(e)
    _cache[key] = (time.time(), out)
    return out


# -- rules panel (pure, tested) --------------------------------------------------------
ALLOWED_SYMBOLS = {"BTCUSD.vx", "XAUUSD.vx", "XAUEUR.vx", "DAX40.vx"}


def _r(group, rule, status, value, limit=""):
    return {"group": group, "rule": rule, "status": status, "value": value, "limit": limit}


def rules_check(*, challenge: dict | None, account: dict, multi_status: dict | None, trades: list[dict],
                processes: dict, kill_switch: str | None, edge: dict | None, news_events: list[dict],
                now: datetime) -> list[dict]:
    """Every FundingPips and own rule with status pass / warn / fail / info."""
    out: list[dict] = []
    eq = float(account.get("equity") or 0.0)
    ch = challenge or {}
    # ---- FundingPips 2-Step Standard (judged as the firm would) --------------------------
    if ch:
        res = ch.get("result")
        out.append(_r("FundingPips", f"Challenge result (phase {ch.get('phase')})",
                      "fail" if res == "failed" else "pass" if res == "passed" else "info", res))
        tgt, base = ch.get("target_usd"), ch.get("phase_start_balance")
        out.append(_r("FundingPips", f"Profit target phase {ch.get('phase')}",
                      "pass" if tgt and eq >= tgt else "info", f"{ch.get('profit_pct')}%",
                      f"{'+8' if ch.get('phase') == 1 else '+5'}% (${tgt:,.0f})" if tgt else ""))
        days = len(ch.get("trading_days") or [])
        out.append(_r("FundingPips", "Minimum trading days", "pass" if days >= 3 else "info", days, "≥ 3"))
        dl = float(ch.get("daily_loss_pct") or 0)
        out.append(_r("FundingPips", "Daily loss (day starts 21:00 UTC)", "fail" if dl >= 5 else "warn" if dl >= 3.5 else "pass",
                      f"{dl:.2f}% (room ${ch.get('daily_headroom_usd')})", "< 5% of higher of opening balance/equity"))
        tl = 100 * (base - eq) / base if base and eq else 0.0
        out.append(_r("FundingPips", "Max loss (static from start)", "fail" if tl >= 10 else "warn" if tl >= 7 else "pass",
                      f"{max(tl, 0):.2f}% (room ${ch.get('total_headroom_usd')})", f"< 10% (never below ${base * 0.9:,.0f})" if base else "< 10%"))
    else:
        out.append(_r("FundingPips", "Challenge tracking", "warn", "not started",
                      "set PROP_CHALLENGE and run tools.challenge_tracker --reset"))
    open_tr = [t for t in trades if t.get("status") == "OPEN"]
    worst = max(((t.get("risk_usd") or 0) / eq * 100 for t in open_tr), default=0.0) if eq else 0.0
    out.append(_r("FundingPips", "Risk per trade (funded max 3%)", "fail" if worst > 3 else "pass", f"{worst:.2f}% largest open", "≤ 3%"))
    near_news = []
    for t in trades:
        for key in ("opened_utc", "closed_utc"):
            if not t.get(key):
                continue
            ts = datetime.fromisoformat(t[key])
            for e in news_events:
                if abs((ts - datetime.fromisoformat(e["time_utc"])).total_seconds()) <= 300:
                    near_news.append(f"#{t.get('ticket')} {key[:-4]} near {e['event']}")
    phase = ch.get("phase")
    funded = phase not in (1, 2)
    out.append(_r("FundingPips", "News window (funded only: ±5 min news / ±10 min speeches; trades opened 5 h+ before are exempt)",
                  ("warn" if near_news else "pass") if funded else "info",
                  ("; ".join(near_news[:3]) or "none this week") if funded else "no news restriction during evaluation",
                  "profit deducted, not a breach"))
    idle = ch.get("days_since_last_closed_trade")
    if idle is not None:
        out.append(_r("FundingPips", "Inactivity (a trade must close within 30 days)",
                      "fail" if idle >= 30 else "warn" if idle >= 20 else "pass", f"{idle:.0f} days since last closed trade", "< 30 days"))
    out.append(_r("FundingPips", "Weekend / overnight holding", "info", "Swing add-on bought", "required for 14-96 h trades"))
    # ---- own rules (RULES.md) -----------------------------------------------------------------
    ms = multi_status or {}
    risk = float(ms.get("risk_per_trade") or 0.005) * 100
    out.append(_r("Own", "Risk per trade", "pass" if risk <= 0.5 + 1e-9 else "warn", f"{risk:.2f}%", "0.5% (hard cap 1%)"))
    out.append(_r("Own", "Max open positions", "fail" if len(open_tr) > 6 else "pass", len(open_tr), "≤ 6"))
    tot = sum((t.get("risk_usd") or 0) for t in open_tr) / eq * 100 if eq else 0.0
    out.append(_r("Own", "Total open risk", "fail" if tot > 3 else "warn" if tot > 2.5 else "pass", f"{tot:.2f}%", "≤ 3%"))
    groups: dict[str, int] = {}
    for t in open_tr:
        groups[t.get("grp") or "?"] = groups.get(t.get("grp") or "?", 0) + 1
    gmax = max(groups.values(), default=0)
    out.append(_r("Own", "Positions per market group", "fail" if gmax > 2 else "pass", gmax, "≤ 2"))
    syms = set(ms.get("symbols") or [])
    bad = sorted(syms - ALLOWED_SYMBOLS)
    out.append(_r("Own", "Only scan-approved symbols", "fail" if bad else "pass", ", ".join(sorted(syms)) or "–",
                  "BTC, XAU(USD or EUR), DAX"))
    gold = syms & {"XAUUSD.vx", "XAUEUR.vx"}
    out.append(_r("Own", "Only one gold symbol", "warn" if len(gold) > 1 else "pass", ", ".join(sorted(gold)) or "none", "1"))
    out.append(_r("Own", "BTC-only bot NOT on the same account", "fail" if processes.get("btc_live") else "pass",
                  "running" if processes.get("btc_live") else "not running", "not running"))
    acct_type = account.get("type")
    out.append(_r("Own", "Demo until go-live checklist done", "pass" if acct_type == "demo" else "warn", acct_type or "?", "demo"))
    algo = account.get("algo_terminal") and account.get("algo_account")
    out.append(_r("Own", "Algo trading enabled", "pass" if algo else "fail", "on" if algo else "OFF", "on"))
    out.append(_r("Own", "Kill switch clear", "fail" if kill_switch else "pass", kill_switch or "clear", "clear"))
    for name, key in (("Trader running", "live_multi"), ("Watchdog running", "watchdog"), ("MT5 running", "mt5")):
        out.append(_r("Own", name, "pass" if processes.get(key) else "fail", "yes" if processes.get(key) else "no", "yes"))
    if edge:
        out.append(_r("Own", "Edge monitor (live vs backtest)", "fail" if edge.get("retire") else "pass",
                      f"{edge.get('live_trades')} trades, mean {edge.get('live_mean_R')} R",
                      f"stop below {edge.get('sample_floor_R')} R" if edge.get("sample_floor_R") is not None else "starts after 20 trades"))
        out.append(_r("Own", "Losing streak", "fail" if (edge.get("live_worst_streak") or 0) > (edge.get("streak_limit") or 99) else "pass",
                      edge.get("live_worst_streak"), f"≤ {edge.get('streak_limit')}"))
    prop = ms.get("prop")
    if prop is None:
        out.append(_r("Own", "Prop controls active in trader", "warn", "not reported (PROP_CHALLENGE off or trader not running)", "active"))
    else:
        out.append(_r("Own", "Prop controls active in trader", "pass" if prop.get("allow_entries") else "info",
                      "; ".join(prop.get("reasons") or []) or f"entries allowed, risk {100 * (prop.get('risk_cap') or 0):.2f}%",
                      "stop -3.5%/day, flatten -4.25%/day, flatten+kill -8.5% total, close idea at -0.9%, 10 min post-loss cooldown"))
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


def _week_news() -> list[dict]:
    """All high-impact events in this week's calendar (for the news-window rule)."""
    try:
        from news.filter import NewsFilter
        return [{"time_utc": e["time"].isoformat(), "event": e["event"], "currency": e["currency"]}
                for e in NewsFilter().get_events() if e["impact"] == "HIGH"]
    except Exception:
        return []


def symbol_strength(trades: list[dict], top: int = 12) -> list[dict]:
    """Backtest scan ranking (reports/multi_symbol_scan.json) joined with each symbol's live record."""
    rep = _json(R / "multi_symbol_scan.json") or {}
    live: dict[str, dict] = {}
    for t in trades:
        if t.get("status") != "CLOSED" or t.get("pnl_usd") is None:
            continue
        s = live.setdefault(t["symbol"], {"trades": 0, "wins": 0, "net_usd": 0.0, "sum_R": 0.0})
        s["trades"] += 1
        s["wins"] += t["pnl_usd"] > 0
        s["net_usd"] += t["pnl_usd"]
        s["sum_R"] += t.get("r_multiple") or 0.0
    rows = []
    for sym, r in (rep.get("results") or {}).items():
        if r.get("expectancy_R") is None:
            continue
        lv = live.get(sym, {})
        rows.append({"symbol": sym, "group": r.get("group"), "years": r.get("years"),
                     "trades_per_month": r.get("trades_per_month"), "expectancy_R": r.get("expectancy_R"),
                     "profit_factor": r.get("profit_factor"), "wf_positive": r.get("wf_positive"),
                     "passes": bool(r.get("passes")), "live_trades": lv.get("trades", 0),
                     "live_wins": lv.get("wins", 0), "live_net_usd": round(lv.get("net_usd", 0.0), 2),
                     "live_avg_R": round(lv["sum_R"] / lv["trades"], 3) if lv.get("trades") else None})
    rows.sort(key=lambda x: (not x["passes"], -x["expectancy_R"]))
    return rows[:top]


def add_blocks(mkt: dict, open_positions: list[dict]) -> None:
    """Attach `blocks` (closed-bar signal), `blocks_forming` and `next` (next check) to every symbol in a market() result."""
    syms = mkt.get("symbols") or {}
    groups = {s: d.get("group", "") for s, d in syms.items()}
    cap = int(float(os.getenv("MULTI_MAX_PER_GROUP", 2)))
    long_only = regime_filter.parse_symbols(os.getenv("MULTI_LONG_ONLY_SYMBOLS", ""))
    now = datetime.now(timezone.utc)
    for s, d in syms.items():
        kw = dict(open_positions=open_positions, groups=groups, max_per_group=cap, regime=d.get("regime"),
                  long_only=s in long_only)
        d["blocks"] = order_blocks(s, (d.get("readiness") or {}).get("decision"), **kw)
        d["blocks_forming"] = order_blocks(s, (d.get("forming") or {}).get("decision"), **kw)
        d["next"] = next_trade(s, d, now=now, open_positions=open_positions, groups=groups, max_per_group=cap,
                               long_only=s in long_only)


def tests_table(reports: Path = R, now: datetime | None = None) -> dict | None:
    """The live trader and every paper test in one table (tools/paper_report.py); None if it cannot be built."""
    try:
        from tools import paper_report
        return paper_report.collect(reports, now)
    except Exception:
        return None


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
    procs = processes()
    mkt = market(symbols)
    add_blocks(mkt, (ms or {}).get("open_positions") or [])
    kill = FILES["kill"].read_text(encoding="utf-8").strip()[:200] if FILES["kill"].exists() else None
    try:
        rules = rules_check(challenge=ch, account=mkt.get("account") or {}, multi_status=ms, trades=trades,
                            processes=procs, kill_switch=kill, edge=edge, news_events=_week_news(), now=now)
    except Exception as e:
        rules = [_r("Own", "Rules panel", "warn", f"error: {e}")]
    return {
        "generated_utc": now.isoformat(),
        "rules": rules,
        "processes": procs,
        "kill_switch": kill,
        "multi": {"status": ms, "age_min": _age_min(ms)},
        "btc": {"status": bs, "age_min": _age_min(bs)},
        "challenge": ch,
        "entry_window": next_entry_window(now),
        "market": mkt,
        "trades": trades[-100:],
        "equity_curve": eq,
        "news": news,
        "edge": edge,
        "alerts": alerts,
        "strength": symbol_strength(trades),
        "tests": tests_table(now=now),
    }
