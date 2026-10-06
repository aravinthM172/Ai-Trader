"""
Go-live preflight: catches, BEFORE money is lost, the failures seen in the first live week
(DAX40 sized 11x too big, MT5 down 9 h with missed signals, XAUUSD + XAUEUR losing together),
plus what usually breaks when switching to a new / real account.

    python -m tools.go_live_preflight                    # check the account MT5 is logged in to
    python -m tools.go_live_preflight --balance 2000     # also size every symbol for a planned deposit
    python -m tools.go_live_preflight --notify           # Telegram the summary when anything FAILs

Read-only: never sends, modifies or closes orders.  Exit code 1 if any check FAILs.
Writes reports/go_live_preflight.json.

Checks
  account      account type matches LIVE_ACCOUNT_MODE, Algo Trading on, currency, balance
  config       effective guard (one position per correlated group), risk per trade, running trader
               restarted after the last .env change
  sizing       per symbol: the volume live_multi would send now, and the $ loss at the stop from the
               broker's own profit calculator -- must match the plan and stay under the max-risk cap
  min lot      the balance needed so the minimum lot fits under the max-risk cap
  spread       current spread vs the live entry filter
  uptime       trader heartbeat gaps in the last 7 days
  missed       real signals on bars the trader never evaluated while flat in that symbol
  history      closed trades that lost more than 1.3 x planned risk; entry slippage
  safety       kill switch, news calendar freshness
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from execution import live_multi as lm
from execution import safety
from strategy import btc_h1_signal
from backtest.btc_strategies import base_atr
from common import notify

REPORTS = ROOT / "reports"
OUT = REPORTS / "go_live_preflight.json"
CONSOLE = ROOT / "logs" / "multi_console.log"
CALENDAR = ROOT / "data" / "calendar_cache.json"
LOSS_TOLERANCE = 1.30          # a stop-out may cost at most 1.3 x planned risk (gaps/slippage)
SIZING_TOLERANCE = 1.25        # broker-calculated loss at SL vs planned risk
HEARTBEAT_GAP_MIN = 15
# Known, fixed incidents (ticket -> why).  Listed so history checks stay meaningful.
ACKNOWLEDGED = {20257630511: "DAX40 sized 11x (tick value bug) -- fixed in 14007d7 on 2026-10-06"}


class Report:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, area: str, status: str, msg: str, **data):
        self.items.append(dict(area=area, status=status, msg=msg, **data))

    @property
    def fails(self):
        return [i for i in self.items if i["status"] == "FAIL"]


def check_account(gw, rep: Report) -> dict:
    acct = gw.account_info() or {}
    term = gw.raw().terminal_info()
    mode = lm._ACCOUNT_MODES.get(int(acct.get("trade_mode", -1)), "unknown")
    want = os.getenv("LIVE_ACCOUNT_MODE", "demo").strip().lower()
    rep.add("account", "PASS" if mode == want else "FAIL",
            f"logged in to a {mode.upper()} account (server {acct.get('server')}); LIVE_ACCOUNT_MODE={want}")
    rep.add("account", "PASS" if term and term.trade_allowed else "FAIL",
            "MT5 Algo Trading button is " + ("ON" if term and term.trade_allowed else "OFF -- orders will be rejected"))
    rep.add("account", "INFO", f"balance {acct.get('balance')} {acct.get('currency')}, equity {acct.get('equity')}, "
                                f"leverage 1:{acct.get('leverage')}")
    ok, why = lm.live_send_allowed(acct)
    rep.add("account", "INFO", f"trader would send real orders now: {ok} ({why})")
    return acct


def check_config(rep: Report, syms: list[str], groups: dict[str, str]):
    g = lm.GUARD
    by_group: dict[str, list[str]] = {}
    for s in syms:
        by_group.setdefault(groups.get(s, ""), []).append(s)
    shared = {k: v for k, v in by_group.items() if len(v) > 1}
    if shared and g.max_per_group > 1:
        rep.add("config", "FAIL", f"max_per_group={g.max_per_group}: correlated symbols can open together {shared} "
                                  "-- set MULTI_MAX_PER_GROUP=1 in .env")
    else:
        rep.add("config", "PASS", f"max_per_group={g.max_per_group}; correlated groups {shared or 'none'}")
    rep.add("config", "INFO", f"symbols {syms}; risk/trade {lm.RISK_PER_TRADE:.2%} (cap {lm.MAX_RISK_PER_TRADE:.2%}); "
                              f"max open {g.max_open_positions}, total open risk {g.max_total_open_risk:.1%}")
    try:
        import psutil
        env_m = datetime.fromtimestamp((ROOT / ".env").stat().st_mtime, timezone.utc)
        procs = [p for p in psutil.process_iter(["cmdline", "create_time"])
                 if "execution.live_multi" in (p.info["cmdline"] or [])]      # the python process, not its .bat launcher
        if not procs:
            rep.add("config", "WARN", "live_multi trader process is not running")
        else:
            started = datetime.fromtimestamp(min(p.info["create_time"] for p in procs), timezone.utc)
            if started < env_m:
                rep.add("config", "FAIL", f"running trader started {started:%Y-%m-%d %H:%M} UTC, BEFORE the last .env change "
                                          f"({env_m:%Y-%m-%d %H:%M} UTC) -- it is not using the current settings; restart it")
            else:
                rep.add("config", "PASS", f"running trader started {started:%Y-%m-%d %H:%M} UTC, after the last .env change")
    except Exception as e:
        rep.add("config", "WARN", f"could not inspect the trader process: {e}")


def check_sizing(gw, rep: Report, syms, specs, equity: float, label: str, planned_balance: float | None = None):
    mt5 = gw.raw()
    ccy = (gw.account_info() or {}).get("currency", "USD")
    bal = planned_balance or equity
    for s in syms:
        spec = specs.get(s)
        df = gw.get_rates(s, "H1", 400)
        tick = gw.get_tick(s)
        if spec is None or df is None or df.empty or not tick:
            rep.add("sizing", "FAIL", f"{s}: no spec / bars / tick -- symbol missing on this account?", symbol=s)
            continue
        atr = float(base_atr(df.high.to_numpy(float), df.low.to_numpy(float), df.close.to_numpy(float), 14)[-1])
        worst = None
        for d, otype, px in (("BUY", mt5.ORDER_TYPE_BUY, tick["ask"]), ("SELL", mt5.ORDER_TYPE_SELL, tick["bid"])):
            sl, tp, sl_d = lm.stop_plan(d, px, atr, spec.stops_level_price)
            vpu, note = lm.loss_value_per_unit(spec, gw.calc_value_per_unit(s, d, px), ccy)
            if vpu is None:
                rep.add("sizing", "FAIL", f"{s} {d}: {note}", symbol=s)
                continue
            vol, risk_usd, why = lm.size_volume(equity=bal, risk_frac=lm.symbol_risk(s), max_risk_frac=lm.MAX_RISK_PER_TRADE,
                                                sl_dist=sl_d, value_per_unit=vpu, vmin=spec.volume_min,
                                                vstep=spec.volume_step, vmax=spec.volume_max)
            if vol <= 0:
                need = spec.volume_min * sl_d * vpu / lm.MAX_RISK_PER_TRADE
                rep.add("min lot", "WARN", f"{s}: {why} -> trades SKIPPED at {bal:,.0f}; needs a balance of ~{need:,.0f}",
                        symbol=s, min_balance=round(need))
                break
            loss = mt5.order_calc_profit(otype, s, vol, px, sl)
            actual = abs(loss) if loss is not None else None
            ratio = actual / risk_usd if actual and risk_usd else None
            worst = max(worst or 0, ratio or 0)
            status = "PASS"
            if actual is None:
                status = "WARN"
            elif ratio > SIZING_TOLERANCE or actual > bal * lm.MAX_RISK_PER_TRADE * 1.05:
                status = "FAIL"
            rep.add("sizing", status, f"{label} {s} {d}: {vol} lot, stop {sl_d:.5g} away -> planned ${risk_usd:.2f} "
                                      f"({risk_usd / bal:.2%}), broker says ${actual if actual is None else round(actual, 2)}"
                                      f"{'' if ratio is None else f' (x{ratio:.2f})'}{'' if note == 'ok' else ' | ' + note}",
                    symbol=s, volume=vol, planned=round(risk_usd, 2), broker_loss=actual)
        if not planned_balance:
            ratio_sp = tick["spread"] / atr if atr > 0 else np.inf
            rep.add("spread", "PASS" if ratio_sp <= lm.MAX_SPREAD_ATR else "WARN",
                    f"{s}: spread {tick['spread']:.5g} = {ratio_sp:.2f} x ATR (entry filter {lm.MAX_SPREAD_ATR})")


def check_uptime(rep: Report, days: int = 7) -> list[tuple]:
    if not CONSOLE.exists():
        rep.add("uptime", "WARN", "no trader console log")
        return []
    ts = []
    with CONSOLE.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\] (LIVE|DRY)", line)
            if m:
                ts.append(datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc))
    now = datetime.now(timezone.utc)
    ts = [t for t in ts if t >= now - timedelta(days=days)] + [now]
    gaps = [(a, b) for a, b in zip(ts, ts[1:]) if (b - a) > timedelta(minutes=HEARTBEAT_GAP_MIN)]
    down = sum(((b - a) for a, b in gaps), timedelta())
    longest = max(((b - a) for a, b in gaps), default=timedelta())
    status = "PASS" if longest < timedelta(hours=1) else ("WARN" if longest < timedelta(hours=3) else "FAIL")
    rep.add("uptime", status, f"last {days} days: trader down {down.total_seconds() / 3600:.1f} h in {len(gaps)} gaps, "
                              f"longest {longest.total_seconds() / 3600:.1f} h"
                              + "".join(f"\n      {a:%a %d %b %H:%M} -> {b:%H:%M} UTC" for a, b in gaps if b - a > timedelta(hours=1)))
    return gaps


def check_missed(gw, rep: Report, syms, gaps, trades: pd.DataFrame):
    """Signals on bars that fell inside downtime while the bot had no position in that symbol."""
    missed = []
    for s in syms:
        last = None                                       # a run of consecutive signals = one missed trade
        df = gw.get_rates(s, "H1", 600)
        if df is None or df.empty:
            continue
        df["time"] = pd.to_datetime(df.time, utc=True)
        df = df.iloc[:-1].reset_index(drop=True)
        t_s = trades[trades.symbol == s]
        for a, b in gaps:
            for i in np.flatnonzero((df.time + pd.Timedelta(hours=1) > a) & (df.time + pd.Timedelta(hours=1) < b)):
                bar_close = df.time.iloc[i] + pd.Timedelta(hours=1)
                busy = ((t_s.opened <= bar_close) & (t_s.closed.isna() | (t_s.closed > bar_close))).any()
                if busy or i < 300:
                    continue
                sig = btc_h1_signal.generate(s, df.iloc[: i + 1])
                if sig.decision in ("BUY", "SELL") and sig.decision != last:
                    missed.append(f"{s} {sig.decision} @ {bar_close:%a %d %b %H:%M} UTC")
                last = sig.decision if sig.decision in ("BUY", "SELL") else None
    rep.add("missed", "PASS" if not missed else "FAIL",
            f"{len(missed)} trades missed while the trader was down" + "".join(f"\n      {m}" for m in missed[:15]))


def check_history(rep: Report) -> pd.DataFrame:
    c = sqlite3.connect(f"file:{lm.DB}?mode=ro", uri=True)
    t = pd.read_sql("select * from trades", c)
    ev = pd.read_sql("select kind, payload from events where kind='order_send'", c)
    t["opened"] = pd.to_datetime(t.opened_utc, utc=True, format="ISO8601")
    t["closed"] = pd.to_datetime(t.closed_utc, utc=True, format="ISO8601")
    closed = t[t.status == "CLOSED"]
    bad = closed[closed.pnl_usd < -LOSS_TOLERANCE * closed.risk_usd]
    for _, r in bad.iterrows():
        ack = ACKNOWLEDGED.get(int(r.ticket))
        rep.add("history", "INFO" if ack else "FAIL",
                f"#{r.ticket} {r.symbol} lost ${-r.pnl_usd:.2f} vs planned ${r.risk_usd:.2f} (x{-r.pnl_usd / r.risk_usd:.2f})"
                + (f" -- known: {ack}" if ack else " -- sizing or stop problem"))
    if bad.empty:
        rep.add("history", "PASS", f"{len(closed)} closed trades, none lost more than {LOSS_TOLERANCE} x planned risk")
    slips = []
    for _, e in ev.iterrows():
        p = json.loads(e.payload)
        req = p.get("request", {})
        row = t[(t.symbol == p.get("symbol")) & np.isclose(t.sl, req.get("sl", np.nan), rtol=1e-6)]
        if len(row) and req.get("price"):
            d = 1 if req.get("type") == 0 else -1
            slips.append(d * (float(row.entry.iloc[0]) - req["price"]) / float(row.atr.iloc[0]))
    if slips:
        rep.add("history", "PASS" if np.mean(slips) < 0.05 else "WARN",
                f"entry slippage over {len(slips)} orders: mean {np.mean(slips):+.3f} ATR, worst {max(slips):+.3f} ATR")
    return t


def check_safety(rep: Report):
    rep.add("safety", "INFO" if not safety.kill_switch_active() else "WARN",
            "emergency stop is " + ("ACTIVE -- no new entries" if safety.kill_switch_active() else "clear"))
    if CALENDAR.exists():
        age = (datetime.now() - datetime.fromtimestamp(CALENDAR.stat().st_mtime)).total_seconds() / 3600
        rep.add("safety", "PASS" if age < 24 else "WARN", f"news calendar cache is {age:.1f} h old")
    else:
        rep.add("safety", "WARN", "no news calendar cache")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=float, help="also size every symbol for this planned account balance")
    ap.add_argument("--notify", action="store_true", help="Telegram the summary if anything FAILs")
    a = ap.parse_args()
    from mt5.gateway import MT5Gateway
    rep = Report()
    gw = MT5Gateway()
    if not gw.connect():
        print("FAIL: MT5 connection failed -- start the terminal and log in")
        return 1
    try:
        acct = check_account(gw, rep)
        syms = lm.symbols()
        mt5 = gw.raw()
        groups = {s: ((mt5.symbol_info(s).path or "").split("\\")[0] if mt5.symbol_info(s) else "") for s in syms}
        specs = {s: gw.get_spec(s) for s in syms}
        check_config(rep, syms, groups)
        equity = float(acct.get("equity") or acct.get("balance") or 0.0)
        check_sizing(gw, rep, syms, specs, equity, "now")
        if a.balance:
            check_sizing(gw, rep, syms, specs, equity, f"at {a.balance:,.0f}", planned_balance=a.balance)
        gaps = check_uptime(rep)
        trades = check_history(rep)
        check_missed(gw, rep, syms, gaps, trades)
        check_safety(rep)
    finally:
        gw.shutdown()

    order = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
    print(f"\nGO-LIVE PREFLIGHT  {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n")
    for i in rep.items:
        print(f"[{i['status']:4}] {i['area']:8} {i['msg']}")
    n = {k: sum(i["status"] == k for i in rep.items) for k in order}
    verdict = "NOT READY" if n["FAIL"] else ("READY WITH WARNINGS" if n["WARN"] else "READY")
    print(f"\n{verdict}: {n['FAIL']} fail, {n['WARN']} warn, {n['PASS']} pass")
    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   verdict=verdict, counts=n, items=rep.items), indent=2, default=str))
    if a.notify and rep.fails:
        notify.send(f"[preflight] {verdict}: " + "; ".join(i["msg"].split("\n")[0] for i in rep.fails)[:3500])
    return 1 if rep.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
