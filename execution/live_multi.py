"""
MULTI-SYMBOL live execution of the frozen H1 momentum rule (momentum_rsi_mtf) on the
symbols that passed backtest/multi_symbol_scan.py.  Separate from execution/live.py
(which keeps trading BTCUSD.vx unchanged); BTCUSD.vx is excluded here by default.

Signal: strategy/btc_h1_signal.generate() -- the same validated builder, verbatim,
on the last COMPLETED H1 bar of each symbol; entry at the start of the next bar,
only inside the entry window (a late start skips the bar, never chases).

Per entry, in order (any failure = no order):
  1. kill switch (state/KILL_SWITCH, shared with live.py)
  2. news filter (high-impact event in the symbol's currencies)
  3. portfolio guard (risk/portfolio_guard.py): max positions, total open risk,
     per-group cap, daily loss limit, drawdown pause
  4. spread <= MAX_SPREAD_ATR x ATR, tick fresh
  5. size = risk % of equity / (2 ATR stop x $ per price unit per lot), broker steps;
     skipped if the minimum lot would risk more than MAX_RISK_PER_TRADE
  6. broker order_check
Then it SENDS only if ALL of:  MULTI_LIVE_TRADING=true, LIVE_TRADING=true, account
mode == LIVE_ACCOUNT_MODE (default demo).  Otherwise: dry run, logged as WOULD SEND.

After a fill: broker SL (2 ATR) / TP (3 ATR) re-anchored at the fill; SL verified,
else the position is closed.  Time exit after 96 bars.  One position per symbol.

    python -m execution.live_multi              # one pass
    python -m execution.live_multi --loop 30    # production loop

State: state/multi_live.sqlite, state/multi_live_state.json
Status: reports/multi_live_status.json        Log: logs/multi_live.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from backtest.btc_strategies import base_atr
from common.logging_setup import get_logger
from execution import safety
from execution.order_validator import build_request
from news.filter import NewsFilter
from risk import portfolio_guard as pg
from risk import prop_controls
from strategy import btc_h1_signal

load_dotenv()
log = get_logger("multi.live", filename="multi_live.log")

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
REPORTS = ROOT / "reports"
DB = STATE / "multi_live.sqlite"
LS = STATE / "multi_live_state.json"
COMMENT = "mom-multi"
MAGIC_BASE = 26_100_000
SL_ATR, TP_ATR = 2.0, 3.0
MAX_HOLD_H = 96
DEVIATION = 50
_DONE = {10008, 10009, 10010}
_RETRY = {10004, 10020, 10021}
_ACCOUNT_MODES = {0: "demo", 1: "contest", 2: "real"}


def _env_f(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_b(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


RISK_PER_TRADE = _env_f("MULTI_RISK_PER_TRADE", 0.005)
MAX_RISK_PER_TRADE = _env_f("MULTI_MAX_RISK_PER_TRADE", 0.01)
MAX_SPREAD_ATR = _env_f("MULTI_MAX_SPREAD_ATR", 0.35)
MAX_TICK_AGE_S = _env_f("MULTI_MAX_TICK_AGE_S", 120)
ENTRY_WINDOW_S = _env_f("MULTI_ENTRY_WINDOW_SECONDS", 900)
GUARD = pg.GuardConfig(
    max_open_positions=int(_env_f("MULTI_MAX_OPEN_POSITIONS", 6)),
    max_total_open_risk=_env_f("MULTI_MAX_TOTAL_OPEN_RISK", 0.03),
    max_per_group=int(_env_f("MULTI_MAX_PER_GROUP", 2)),
    daily_loss_limit=_env_f("MULTI_DAILY_LOSS_LIMIT", 0.04),
    max_drawdown_pause=_env_f("MULTI_MAX_DRAWDOWN_PAUSE", 0.08),
    dd_pause_hours=_env_f("MULTI_DD_PAUSE_HOURS", 72),
)


# -- configuration -------------------------------------------------------------
def symbols() -> list[str]:
    env = os.getenv("MULTI_SYMBOLS", "").strip()
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    try:
        passers = json.loads((REPORTS / "multi_symbol_scan.json").read_text())["passers"]
    except Exception:
        return []
    return [s for s in passers if s != "BTCUSD.vx"]          # live.py owns BTCUSD.vx


def magic_for(symbol: str) -> int:
    return MAGIC_BASE + zlib.crc32(symbol.encode()) % 100_000


def live_send_allowed(account: dict) -> tuple[bool, str]:
    if not _env_b("MULTI_LIVE_TRADING"):
        return False, "MULTI_LIVE_TRADING is off"
    if not safety.live_trading_enabled():
        return False, "LIVE_TRADING is off"
    if safety.kill_switch_active():
        return False, "kill switch active"
    mode = _ACCOUNT_MODES.get(int(account.get("trade_mode", -1)), "unknown")
    allowed = os.getenv("LIVE_ACCOUNT_MODE", "demo").strip().lower()
    if mode != allowed:
        return False, f"account is {mode}, LIVE_ACCOUNT_MODE={allowed}"
    return True, "ok"


# -- pure helpers (unit-tested) ----------------------------------------------------
def completed_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop the still-forming H1 bar (time >= start of the current hour)."""
    cur = pd.Timestamp(now).floor("1h")
    t = pd.to_datetime(df["time"], utc=True)
    return df[t < cur].reset_index(drop=True)


def size_volume(*, equity: float, risk_frac: float, max_risk_frac: float, sl_dist: float,
                value_per_unit: float, vmin: float, vstep: float, vmax: float) -> tuple[float, float, str]:
    """(volume, risk $, note).  volume 0 = skip."""
    if sl_dist <= 0 or value_per_unit <= 0 or equity <= 0:
        return 0.0, 0.0, "bad inputs"
    per_lot = sl_dist * value_per_unit
    raw = equity * risk_frac / per_lot
    vol = math.floor(raw / vstep + 1e-9) * vstep
    if vol < vmin:
        if vmin * per_lot <= equity * max_risk_frac:
            vol = vmin
        else:
            return 0.0, vmin * per_lot, f"min lot risks ${vmin * per_lot:.2f} > {max_risk_frac:.1%} of equity"
    vol = min(vol, vmax)
    return round(vol, 8), vol * per_lot, "ok"


def stop_plan(direction: str, price: float, atr: float, stops_level: float) -> tuple[float, float, float]:
    sl_d = max(SL_ATR * atr, stops_level * 1.15)
    tp_d = max(TP_ATR * atr, stops_level * 1.15)
    sl = price - sl_d if direction == "BUY" else price + sl_d
    tp = price + tp_d if direction == "BUY" else price - tp_d
    return sl, tp, sl_d


# -- persistence -----------------------------------------------------------------
def _conn(path: Path = DB) -> sqlite3.Connection:
    STATE.mkdir(exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        ticket INTEGER PRIMARY KEY, symbol TEXT, grp TEXT, opened_utc TEXT, signal_bar_utc TEXT,
        direction TEXT, volume REAL, entry REAL, sl REAL, tp REAL, atr REAL, risk_usd REAL,
        status TEXT DEFAULT 'OPEN', closed_utc TEXT, exit REAL, exit_reason TEXT, pnl_usd REAL, r_multiple REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, kind TEXT, payload TEXT)""")
    return c


def _event(c, kind: str, payload: dict) -> None:
    c.execute("INSERT INTO events(ts_utc,kind,payload) VALUES(?,?,?)",
              (datetime.now(timezone.utc).isoformat(), kind, json.dumps(payload, default=str)))
    c.commit()


def _load_ls() -> dict:
    try:
        return json.loads(LS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ls(s: dict) -> None:
    LS.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def guard_state(c, open_pos: list[dict], groups: dict, equity: float, peak: float, now: datetime) -> pg.GuardState:
    """Rebuild the guard state from MT5 positions + the trade DB (restart-safe)."""
    st = pg.GuardState(peak_equity=peak)
    rows = {r[0]: r for r in c.execute("SELECT ticket, symbol, risk_usd FROM trades").fetchall()}
    for p in open_pos:
        r = rows.get(p["ticket"])
        risk = (r[2] / equity) if r and r[2] and equity > 0 else RISK_PER_TRADE
        st.open.append(pg.OpenPos(p["symbol"], groups.get(p["symbol"], ""), risk))
    day = now.strftime("%Y-%m-%d")
    realised = c.execute("SELECT COALESCE(SUM(pnl_usd),0) FROM trades WHERE status='CLOSED' AND substr(closed_utc,1,10)=?",
                         (day,)).fetchone()[0]
    st.day, st.day_realised = day, float(realised)
    st.day_start_equity = equity - float(realised)
    for sym, t in c.execute("SELECT symbol, MAX(closed_utc) FROM trades WHERE status='CLOSED' GROUP BY symbol"):
        if t:
            st.last_exit[sym] = datetime.fromisoformat(t)
    return st


# -- broker actions -----------------------------------------------------------------
def _close(gw, spec, p, reason, c) -> bool:
    mt5 = gw.raw()
    rc = -1
    for _ in range(3):
        tick = gw.get_tick(spec.symbol)
        r = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "symbol": spec.symbol, "volume": p["volume"],
            "type": mt5.ORDER_TYPE_SELL if p["type"] == "BUY" else mt5.ORDER_TYPE_BUY,
            "position": p["ticket"], "price": tick["bid"] if p["type"] == "BUY" else tick["ask"],
            "deviation": DEVIATION, "magic": p["magic"], "comment": f"close:{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": gw.preferred_filling(spec)})
        rc = int(getattr(r, "retcode", -1)) if r is not None else -1
        _event(c, "close_send", {"ticket": p["ticket"], "symbol": spec.symbol, "reason": reason, "retcode": rc})
        if rc in _DONE:
            return True
        if rc not in _RETRY:
            break
    log.error("FAILED to close #%s %s (%s) retcode %s", p["ticket"], spec.symbol, reason, rc)
    return False


def _ensure_sl(gw, spec, p, sl, tp, c) -> bool:
    if p["sl"] > 0 and p["tp"] > 0:
        return True
    mt5 = gw.raw()
    r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": spec.symbol, "position": p["ticket"],
                        "sl": round(sl, spec.digits), "tp": round(tp, spec.digits), "magic": p["magic"]})
    ok = (int(getattr(r, "retcode", -1)) if r is not None else -1) in _DONE
    _event(c, "sltp_repair", {"ticket": p["ticket"], "symbol": spec.symbol, "ok": ok})
    return ok


def _sync_closed(gw, c, magics: set[int]) -> None:
    mt5 = gw.raw()
    now = datetime.now(timezone.utc)
    for d in mt5.history_deals_get(now - timedelta(days=45), now + timedelta(days=1)) or []:
        if int(d.magic) not in magics or int(d.entry) not in (1, 3):
            continue
        row = c.execute("SELECT status, risk_usd FROM trades WHERE ticket=?", (int(d.position_id),)).fetchone()
        if not row or row[0] != "OPEN":
            continue
        pnl = float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0))
        reason = {4: "stop", 5: "target"}.get(int(getattr(d, "reason", -1)), "manual/time")
        r_mult = pnl / row[1] if row[1] else None
        c.execute("""UPDATE trades SET status='CLOSED', closed_utc=?, exit=?, exit_reason=?, pnl_usd=?, r_multiple=?
                     WHERE ticket=?""", (gw.to_utc(int(d.time)).isoformat(), float(d.price), reason, round(pnl, 2),
                                         None if r_mult is None else round(r_mult, 3), int(d.position_id)))
        _event(c, "closed", {"ticket": int(d.position_id), "symbol": d.symbol, "reason": reason, "pnl": round(pnl, 2)})
        log.info("TRADE CLOSED #%d %s %s pnl $%.2f", d.position_id, d.symbol, reason, pnl)
    c.commit()


def _send(gw, spec, direction, volume, sl_d, tp_d, magic, c) -> dict:
    mt5 = gw.raw()
    rc, r, px = -1, None, 0.0
    for attempt in range(3):
        tick = gw.get_tick(spec.symbol)
        px = tick["ask"] if direction == "BUY" else tick["bid"]
        sl = px - sl_d if direction == "BUY" else px + sl_d
        tp = px + tp_d if direction == "BUY" else px - tp_d
        req = build_request(gateway=gw, spec=spec, direction=direction, entry=px, volume=volume,
                            sl=sl, tp=tp, magic=magic, comment=COMMENT, deviation=DEVIATION)
        chk = gw.order_check(req)
        if not chk or chk.get("retcode") != 0:
            return {"ok": False, "error": f"order_check failed: {chk}"}
        r = mt5.order_send(req)
        rc = int(getattr(r, "retcode", -1)) if r is not None else -1
        _event(c, "order_send", {"symbol": spec.symbol, "attempt": attempt, "retcode": rc, "request": req})
        if rc in _DONE or rc not in _RETRY:
            break
    if rc not in _DONE:
        return {"ok": False, "retcode": rc}
    mine = [p for p in gw.positions(spec.symbol) if p["magic"] == magic]
    if not mine:
        return {"ok": True, "warning": "filled but position not found"}
    pos = mine[0]
    if not _ensure_sl(gw, spec, pos, sl, tp, c):
        _close(gw, spec, pos, "no_sl", c)
        return {"ok": False, "error": "SL could not be set -- closed"}
    return {"ok": True, "ticket": pos["ticket"], "fill": pos["price_open"], "sl": pos["sl"], "tp": pos["tp"]}


# -- one pass ---------------------------------------------------------------------------
def run_once(*, now: datetime | None = None) -> dict:
    from mt5.gateway import MT5Gateway
    now = now or datetime.now(timezone.utc)
    syms = symbols()
    ls = _load_ls()
    c = _conn()
    gw = MT5Gateway()
    status = {"generated_utc": now.isoformat(), "symbols": syms, "risk_per_trade": RISK_PER_TRADE,
              "kill_switch_active": safety.kill_switch_active(), "entries": {}}
    if not syms:
        status["error"] = "no symbols (run backtest.multi_symbol_scan or set MULTI_SYMBOLS)"
        _write_status(status)
        return status
    if not gw.connect():
        status["error"] = "MT5 connection failed"
        _write_status(status)
        return status
    try:
        mt5 = gw.raw()
        acct = gw.account_info()
        equity = float(acct.get("equity") or acct.get("balance") or 0.0)
        ls["peak_equity"] = max(float(ls.get("peak_equity") or 0.0), equity)
        send_ok, send_why = live_send_allowed(acct)
        status.update(mode="LIVE" if send_ok else "DRY_RUN", send_block_reason=None if send_ok else send_why,
                      equity=equity, balance=float(acct.get("balance") or equity), peak_equity=ls["peak_equity"])
        magics = {magic_for(s): s for s in syms}
        _sync_closed(gw, c, set(magics))
        groups, specs = {}, {}
        for s in syms:
            info = mt5.symbol_info(s)
            groups[s] = (info.path or "").split("\\")[0] if info else ""
            specs[s] = gw.get_spec(s)
        mine = [p for p in gw.positions() if p["magic"] in magics]

        # manage: time exit + missing-SL repair
        for p in mine:
            spec = specs.get(p["symbol"])
            if spec is None:
                continue
            held_h = (now - gw.to_utc(p["time"])).total_seconds() / 3600.0
            if held_h >= MAX_HOLD_H:
                _close(gw, spec, p, "time", c)
            elif p["sl"] <= 0:
                row = c.execute("SELECT sl, tp FROM trades WHERE ticket=?", (p["ticket"],)).fetchone()
                if not (row and _ensure_sl(gw, spec, p, row[0], row[1], c)):
                    _close(gw, spec, p, "no_sl", c)
        mine = [p for p in gw.positions() if p["magic"] in magics]

        # prop-firm controls (only when a challenge rehearsal / prop account is configured)
        prop = _prop_controls(gw, c, ls, mine, specs, equity, float(acct.get("balance") or equity), now, send_ok)
        if prop is not None:
            status["prop"] = prop.to_dict()
            mine = [p for p in gw.positions() if p["magic"] in magics]
        status["open_positions"] = mine

        # entries
        into_bar = (now - pd.Timestamp(now).floor("1h").to_pydatetime()).total_seconds()
        due = (pd.Timestamp(now).floor("1h") - pd.Timedelta(hours=1)).isoformat()
        evaluated = ls.setdefault("evaluated", {})
        if safety.kill_switch_active():
            status["entries"]["*"] = "kill switch active"
        elif into_bar > ENTRY_WINDOW_S:
            status["entries"]["*"] = f"{into_bar:.0f}s into the bar > window {ENTRY_WINDOW_S:.0f}s"
        elif prop is not None and not prop.allow_entries:
            status["entries"]["*"] = "prop controls: " + "; ".join(prop.reasons)
        else:
            news = NewsFilter(fail_closed=_env_b("MULTI_NEWS_FAIL_CLOSED"))
            gst = guard_state(c, mine, groups, equity, ls["peak_equity"], now)
            for s in syms:
                key = f"{s}|{due}"
                if key in evaluated:
                    continue
                status["entries"][s] = res = _evaluate_symbol(
                    gw, mt5, c, s, specs[s], groups[s], due, now, equity, news, gst, send_ok, ls,
                    risk=min(RISK_PER_TRADE, prop.risk_cap) if prop is not None else RISK_PER_TRADE)
                if res.get("final"):
                    evaluated[key] = res.get("decision", "done")
        ls["evaluated"] = dict(sorted(evaluated.items())[-2000:])
        _save_ls(ls)
        summ = c.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usd),0), AVG(r_multiple) FROM trades WHERE status='CLOSED'").fetchone()
        status.update(closed_trades=summ[0], net_pl_usd=round(summ[1], 2),
                      expectancy_R=None if summ[2] is None else round(summ[2], 4))
        _write_status(status)
        return status
    finally:
        gw.shutdown()
        c.close()


def _evaluate_symbol(gw, mt5, c, s, spec, group, due, now, equity, news, gst, send_ok, ls,
                     risk: float = RISK_PER_TRADE) -> dict:
    if spec is None:
        return {"decision": "skip", "reason": "no symbol spec", "final": True}
    df = gw.get_rates(s, "H1", 400)
    if df is None or df.empty:
        return {"decision": "retry", "reason": "no rates"}
    done = completed_bars(df, now)
    if done.empty or pd.Timestamp(done["time"].iloc[-1]).isoformat() != pd.Timestamp(due).isoformat():
        return {"decision": "retry", "reason": f"last completed bar {None if done.empty else done['time'].iloc[-1]} != {due}"}
    sig = btc_h1_signal.generate(s, done)
    if sig.decision == "HOLD":
        return {"decision": "hold", "final": True}
    out = {"decision": sig.decision, "signal_bar_utc": sig.signal_bar_utc}
    ev = news.blocking_event(s, now)
    if news.is_blocked(s, now):
        return {**out, "decision": "blocked", "reason": f"news: {ev['event'] if ev else 'calendar unavailable'}", "final": True}
    atr = float(base_atr(done["high"].to_numpy(float), done["low"].to_numpy(float), done["close"].to_numpy(float), 14)[-1])
    tick = gw.get_tick(s)
    if not tick or (tick.get("age_seconds") or 1e9) > MAX_TICK_AGE_S:
        return {**out, "decision": "retry", "reason": "stale tick"}
    if atr <= 0 or tick["spread"] > MAX_SPREAD_ATR * atr:
        return {**out, "decision": "retry", "reason": f"spread {tick['spread']:.5g} > {MAX_SPREAD_ATR} x ATR {atr:.5g}"}
    px = tick["ask"] if sig.decision == "BUY" else tick["bid"]
    sl, tp, sl_d = stop_plan(sig.decision, px, atr, spec.stops_level_price)
    vol, risk_usd, note = size_volume(equity=equity, risk_frac=risk, max_risk_frac=MAX_RISK_PER_TRADE,
                                      sl_dist=sl_d, value_per_unit=spec.value_per_price_unit_per_lot,
                                      vmin=spec.volume_min, vstep=spec.volume_step, vmax=spec.volume_max)
    if vol <= 0:
        return {**out, "decision": "skip", "reason": note, "final": True}
    ok, why = pg.allow_entry(GUARD, gst, symbol=s, group=group, risk_frac=risk_usd / equity, now=now, equity=equity)
    if not ok:
        return {**out, "decision": "blocked", "reason": f"guard: {why}", "final": True}
    plan = {"entry": px, "sl": sl, "tp": tp, "volume": vol, "risk_usd": round(risk_usd, 2), "atr": atr}
    if not send_ok:
        _event(c, "would_send", {"symbol": s, "direction": sig.decision, **plan})
        log.info("DRY RUN -- would %s %s %.2f @ ~%.5g SL %.5g TP %.5g risk $%.2f", sig.decision, s, vol, px, sl, tp, risk_usd)
        pg.on_open(gst, symbol=s, group=group, risk_frac=risk_usd / equity)      # count it against caps this pass
        return {**out, "decision": "dry_run", "plan": plan, "final": True}
    magic = magic_for(s)
    res = _send(gw, spec, sig.decision, vol, sl_d, abs(tp - px), magic, c)
    if res.get("ok") and res.get("ticket"):
        c.execute("""INSERT OR REPLACE INTO trades(ticket, symbol, grp, opened_utc, signal_bar_utc, direction, volume,
                     entry, sl, tp, atr, risk_usd) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (res["ticket"], s, group, now.isoformat(), sig.signal_bar_utc, sig.decision, vol, res["fill"],
                   res["sl"], res["tp"], atr, round(risk_usd, 2)))
        c.commit()
        pg.on_open(gst, symbol=s, group=group, risk_frac=risk_usd / equity)
        log.warning("LIVE OPEN #%s %s %s %.2f @ %.5g SL %.5g TP %.5g risk $%.2f", res["ticket"], s, sig.decision, vol,
                    res["fill"], res["sl"], res["tp"], risk_usd)
    return {**out, "decision": "sent" if res.get("ok") else "send_failed", "result": res, "final": True}


def _prop_controls(gw, c, ls, mine, specs, equity, balance, now, send_ok):
    """Apply risk/prop_controls when a prop challenge is configured (PROP_CHALLENGE in .env)."""
    from tools import challenge_tracker as ct
    if not ct.enabled():
        return None
    try:
        ch = json.loads(ct.OUT.read_text(encoding="utf-8"))
    except Exception:
        ch = None
    rules = ct.FUNDINGPIPS_2STEP_STANDARD
    day = ct.trading_day(now, rules.day_reset_utc_hour)
    if ls.get("prop_day") != day:                       # firm baseline: higher of balance / equity at day start
        ls["prop_day"], ls["prop_day_ref"] = day, max(balance, equity)
    if ch is not None:
        ch = dict(ch, day_ref=ls["prop_day_ref"])
    row = c.execute("SELECT MAX(closed_utc) FROM trades WHERE status='CLOSED' AND pnl_usd < 0").fetchone()
    last_loss = datetime.fromisoformat(row[0]) if row and row[0] else None
    d = prop_controls.decide(challenge=ch, equity=equity, positions=mine, last_losing_close=last_loss, now=now,
                             trading_day=day, halted_day=ls.get("prop_halt_day"), base_risk=RISK_PER_TRADE)
    if d.halt_today:
        ls["prop_halt_day"] = day
    for p in [p for p in mine if d.flatten_all or p["ticket"] in d.close_tickets]:
        spec = specs.get(p["symbol"])
        if spec is None:
            continue
        reason = "prop_flatten" if d.flatten_all else "idea_loss"
        if send_ok:
            _close(gw, spec, p, reason, c)
        else:
            _event(c, "would_close", {"ticket": p["ticket"], "symbol": p["symbol"], "reason": reason})
        log.warning("PROP CONTROL %s #%s %s (P/L %.2f): %s", "CLOSE" if send_ok else "WOULD CLOSE", p["ticket"],
                    p["symbol"], p.get("profit") or 0.0, "; ".join(d.reasons))
    if d.kill and send_ok and not safety.kill_switch_active():
        reasons = "; ".join(d.reasons)
        (STATE / "KILL_SWITCH").write_text(f"{now.isoformat()}  prop controls: {reasons}\n", encoding="utf-8")
        log.critical("KILL SWITCH TRIPPED by prop controls: %s", "; ".join(d.reasons))
    return d


def _write_status(st: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "multi_live_status.json").write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    print(f"MULTI-SYMBOL LIVE  MULTI_LIVE_TRADING={_env_b('MULTI_LIVE_TRADING')}  LIVE_TRADING={safety.live_trading_enabled()}  "
          f"KILL={'ACTIVE' if safety.kill_switch_active() else 'clear'}  symbols={symbols()}")
    while True:
        try:
            st = run_once()
            print(f"[{st['generated_utc'][:19]}] {st.get('mode')} eq ${st.get('equity')} open {len(st.get('open_positions') or [])} "
                  f"closed {st.get('closed_trades')} net ${st.get('net_pl_usd')} | " +
                  json.dumps({k: v.get('decision') if isinstance(v, dict) else v for k, v in st.get('entries', {}).items()})[:400])
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            log.exception("multi pass failed: %s", e)
        if a.loop <= 0:
            return 0
        try:
            time.sleep(max(10, a.loop))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
