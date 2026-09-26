"""
LIVE execution -- BTCUSD.vx H1 `momentum_rsi_mtf` (frozen).  The ONLY module in the
project allowed to call order_send().

Mirrors the forward-tested replay (execution/paper_replay.py) exactly:
  * signal on the last COMPLETED H1 bar (execution.pipeline -> strategy.btc_h1_signal)
  * enter at the start of the next bar  -> only within LIVE_ENTRY_WINDOW_SECONDS of the
    bar open; a late start (PC asleep, restart) SKIPS the bar instead of chasing price
  * broker-side SL = 2 ATR, TP = 3 ATR (validated geometry, re-anchored at the fill price)
  * time exit after 96 H1 bars
  * one position at a time, one trade per signal bar

Sends ONLY when ALL hold (anything else = dry run: full pipeline + order_check, no send):
  LIVE_TRADING=true, no KILL_SWITCH, terminal + account algo trading allowed,
  every pipeline stage + safety gate passed, broker order_check OK.

Extra live guards on top of execution/safety.py:
  * drawdown breaker: balance < (1 - LIVE_MAX_DRAWDOWN_FRAC) x peak -> writes
    state/KILL_SWITCH (manual reset: delete the file)
  * position without a broker SL after the fill -> SL re-applied, else closed at once
  * clock sanity: a signal bar that looks in the future (bad server offset) is skipped

    python run_live.py            # one pass
    python run_live.py --loop 30  # production

State : state/btc_live_H1.sqlite, state/btc_live_state.json
Status: reports/btc_live_status.json      Log: logs/btc_live.log
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from common.logging_setup import get_logger
from config.assets import get_asset_config
from execution import safety, state as state_store
from execution.order_validator import build_request
from execution.pipeline import run_pipeline
from mt5.gateway import MT5Gateway

log = get_logger("btc.live", filename="btc_live.log")

_ROOT = Path(__file__).resolve().parents[1]
_STATE = _ROOT / "state"
_REPORTS = _ROOT / "reports"
_STATE.mkdir(exist_ok=True)
_REPORTS.mkdir(exist_ok=True)
_DB = _STATE / "btc_live_H1.sqlite"
_LIVE_STATE = _STATE / "btc_live_state.json"
_KILL_FILE = _STATE / "KILL_SWITCH"

ASSET, TF = "BTC", "H1"
MAGIC = 26092601
COMMENT = "btc-h1-live"
MAX_HOLD_H = 96                               # == paper_replay max_hold (H1 bars)
DEVIATION = 50                                # points
_DONE = {10008, 10009, 10010}                 # placed / done / done partial
_RETRY = {10004, 10020, 10021}                # requote / price changed / price off


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def entry_window_s() -> float:
    return _env_f("LIVE_ENTRY_WINDOW_SECONDS", 900)


def max_drawdown_frac() -> float:
    return _env_f("LIVE_MAX_DRAWDOWN_FRAC", 0.35)


_ACCOUNT_MODES = {0: "demo", 1: "contest", 2: "real"}


def allowed_account_mode() -> str:
    """Orders go ONLY to this kind of account.  Default demo; real needs LIVE_ACCOUNT_MODE=real."""
    return os.getenv("LIVE_ACCOUNT_MODE", "demo").strip().lower()


def account_mode_ok(account: dict) -> tuple[bool, str]:
    mode = _ACCOUNT_MODES.get(int(account.get("trade_mode", -1)), "unknown")
    return mode == allowed_account_mode(), mode


# -- persistence -----------------------------------------------------------
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        ticket INTEGER PRIMARY KEY, opened_utc TEXT, signal_bar_utc TEXT, direction TEXT,
        volume REAL, entry REAL, sl REAL, tp REAL, atr REAL, risk_usd REAL,
        status TEXT DEFAULT 'OPEN', closed_utc TEXT, exit REAL, exit_reason TEXT,
        pnl_usd REAL, r_multiple REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, kind TEXT, payload TEXT)""")
    return c


def _event(c, kind: str, payload: dict) -> None:
    c.execute("INSERT INTO events(ts_utc,kind,payload) VALUES(?,?,?)",
              (datetime.now(timezone.utc).isoformat(), kind, json.dumps(payload, default=str)))
    c.commit()


def _load_state() -> dict:
    try:
        return json.loads(_LIVE_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(s: dict) -> None:
    _LIVE_STATE.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


# -- helpers ---------------------------------------------------------------
def _my_positions(gw, symbol: str) -> list[dict]:
    return [p for p in gw.positions(symbol) if p["magic"] == MAGIC]


def _retcode(r) -> int:
    return int(getattr(r, "retcode", -1)) if r is not None else -1


def _close_position(gw, spec, p: dict, reason: str, c) -> bool:
    mt5 = gw.raw()
    for _ in range(3):
        tick = gw.get_tick(spec.symbol)
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": spec.symbol, "volume": p["volume"],
            "type": mt5.ORDER_TYPE_SELL if p["type"] == "BUY" else mt5.ORDER_TYPE_BUY,
            "position": p["ticket"],
            "price": tick["bid"] if p["type"] == "BUY" else tick["ask"],
            "deviation": DEVIATION, "magic": MAGIC, "comment": f"close:{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": gw.preferred_filling(spec),
        }
        r = mt5.order_send(req)
        rc = _retcode(r)
        _event(c, "close_send", {"ticket": p["ticket"], "reason": reason, "retcode": rc,
                                 "comment": getattr(r, "comment", None)})
        if rc in _DONE:
            log.warning("CLOSED #%d (%s) retcode %d", p["ticket"], reason, rc)
            return True
        if rc not in _RETRY:
            break
    log.error("FAILED to close #%d (%s) -- last retcode %d %s", p["ticket"], reason, rc,
              mt5.last_error())
    return False


def _ensure_stops(gw, spec, p: dict, sl: float, tp: float, c) -> bool:
    if p["sl"] > 0 and p["tp"] > 0:
        return True
    mt5 = gw.raw()
    r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": spec.symbol,
                        "position": p["ticket"], "sl": round(sl, spec.digits),
                        "tp": round(tp, spec.digits), "magic": MAGIC})
    ok = _retcode(r) in _DONE
    _event(c, "sltp_repair", {"ticket": p["ticket"], "retcode": _retcode(r), "ok": ok})
    return ok


def _sync_closed(gw, c) -> None:
    """Pull closing deals of our positions from MT5 history; update DB + daily realised P/L."""
    mt5 = gw.raw()
    now = datetime.now(timezone.utc)
    # MT5 filters history in server time; a wide window makes the offset irrelevant
    deals = mt5.history_deals_get(now - timedelta(days=45), now + timedelta(days=1)) or []
    today = now.strftime("%Y-%m-%d")
    realised_today = 0.0
    for d in deals:
        if int(d.magic) != MAGIC or int(d.entry) not in (1, 3):      # DEAL_ENTRY_OUT / OUT_BY
            continue
        pnl = float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0))
        t_utc = gw.to_utc(int(d.time))
        if t_utc.strftime("%Y-%m-%d") == today:
            realised_today += pnl
        row = c.execute("SELECT status, risk_usd FROM trades WHERE ticket=?", (int(d.position_id),)).fetchone()
        if row and row[0] == "OPEN":
            reason = {4: "stop", 5: "target"}.get(int(getattr(d, "reason", -1)), "manual/time")
            r_mult = pnl / row[1] if row[1] else None
            c.execute("""UPDATE trades SET status='CLOSED', closed_utc=?, exit=?, exit_reason=?,
                         pnl_usd=?, r_multiple=? WHERE ticket=?""",
                      (t_utc.isoformat(), float(d.price), reason, round(pnl, 2),
                       round(r_mult, 3) if r_mult is not None else None, int(d.position_id)))
            _event(c, "closed", {"ticket": int(d.position_id), "reason": reason, "pnl": round(pnl, 2)})
            log.info("TRADE CLOSED #%d %s pnl $%.2f R %s", d.position_id, reason, pnl,
                     None if r_mult is None else round(r_mult, 2))
    c.commit()
    st = state_store.load()                     # feeds safety.py's daily-loss guard
    state_store.day_bucket(st)["realised_pl"] = round(realised_today, 2)
    state_store.save(st)


def _trip_kill(reason: str) -> None:
    _KILL_FILE.write_text(f"{datetime.now(timezone.utc).isoformat()}  {reason}\n", encoding="utf-8")
    log.critical("KILL SWITCH TRIPPED: %s", reason)


# -- one pass --------------------------------------------------------------
def manage(c, ls: dict) -> dict:
    """Reconcile open positions, time exits, drawdown breaker.  Returns a snapshot."""
    cfg = get_asset_config(ASSET, timeframe=TF)
    gw = MT5Gateway()
    if not gw.connect():
        return {"error": "MT5 connection failed"}
    try:
        spec = gw.get_spec(cfg.symbol)
        acct = gw.account_info()
        term = gw.terminal_info()
        _sync_closed(gw, c)
        bal = float(acct.get("balance", 0.0))
        peaks = ls.setdefault("peak_balance_by_login", {})       # demo and real tracked apart
        key = str(acct.get("login"))
        peaks[key] = max(float(peaks.get(key) or 0.0), bal)
        ls["peak_balance"] = peaks[key]
        floor = (1.0 - max_drawdown_frac()) * ls["peak_balance"]
        if bal > 0 and bal < floor and not safety.kill_switch_active():
            _trip_kill(f"drawdown breaker: balance ${bal:.2f} < ${floor:.2f} "
                       f"({max_drawdown_frac():.0%} below peak ${ls['peak_balance']:.2f})")

        mine = _my_positions(gw, cfg.symbol)
        now = datetime.now(timezone.utc)
        for p in mine:
            held_h = (now - gw.to_utc(p["time"])).total_seconds() / 3600.0
            if held_h >= MAX_HOLD_H:
                _close_position(gw, spec, p, "time", c)
            elif p["sl"] <= 0:
                row = c.execute("SELECT sl, tp FROM trades WHERE ticket=?", (p["ticket"],)).fetchone()
                if not (row and _ensure_stops(gw, spec, p, row[0], row[1], c)):
                    _close_position(gw, spec, p, "no_sl", c)
        return {"balance": bal, "equity": acct.get("equity"), "peak_balance": ls["peak_balance"],
                "drawdown_floor": round(floor, 2),
                "terminal_trade_allowed": term.get("trade_allowed"),
                "account_trade_allowed": acct.get("trade_allowed"),
                "open_positions": _my_positions(gw, cfg.symbol)}
    finally:
        gw.shutdown()


def _signal_bar_due(now: datetime) -> str:
    """The completed bar whose signal is actionable now (UTC hour label)."""
    return (pd.Timestamp(now).floor("1h") - pd.Timedelta(hours=1)).isoformat()


def maybe_enter(c, ls: dict) -> dict:
    now = datetime.now(timezone.utc)
    due = _signal_bar_due(now)
    evaluated = ls.setdefault("evaluated_bars", {})
    if due in evaluated:
        return {"skipped": f"bar {due} already {evaluated[due]}"}
    into_bar = (now - pd.Timestamp(now).floor("1h").to_pydatetime()).total_seconds()
    if into_bar > entry_window_s():
        evaluated[due] = "missed_window"
        return {"skipped": f"{into_bar:.0f}s into the bar > entry window {entry_window_s():.0f}s"}

    res = run_pipeline(ASSET, timeframe=TF)
    sb = res.get("signal_bar_utc")
    out = {"signal_bar_utc": sb, "signal": (res.get("signal") or {}).get("decision"),
           "decision": res.get("decision"), "reason": res.get("decision_reason"),
           "error": res.get("error")}
    if res.get("error"):
        return out                                           # transient -> retry next pass
    sb_ts = pd.Timestamp(sb) if sb else None
    if sb_ts is None or sb_ts.isoformat() != pd.Timestamp(due).isoformat():
        # the feed's last completed bar isn't the one we expect: clock offset or feed lag
        out["skipped"] = f"signal bar {sb} != expected {due} (clock/feed check)"
        return out
    if res.get("decision") == "HOLD":
        evaluated[due] = "hold"
        return out
    if res.get("decision") != "APPROVED_DRY_RUN":
        return out                                           # a gate blocked -> retry within window
    if c.execute("SELECT 1 FROM trades WHERE signal_bar_utc=?", (sb,)).fetchone():
        evaluated[due] = "already_traded"
        return out

    live = safety.live_trading_enabled() and not safety.kill_switch_active()
    allow_send = bool(res.get("safety", {}).get("allow_live_send"))
    mode_ok, mode = account_mode_ok(res.get("account") or {})
    out["account_mode"] = mode
    if not (live and allow_send and mode_ok):
        evaluated[due] = "dry_run_approved"
        out["dry_run"] = True
        _event(c, "would_send", {"signal_bar": sb, "direction": res.get("evaluated_direction"),
                                 "plan": res.get("stop_plan"), "volume": res["sizing"]["volume"],
                                 "live_trading": safety.live_trading_enabled(),
                                 "allow_live_send": allow_send, "account_mode": mode,
                                 "allowed_account_mode": allowed_account_mode()})
        if live and allow_send and not mode_ok:
            log.warning("NOT SENT: account is %s but LIVE_ACCOUNT_MODE=%s", mode, allowed_account_mode())
        log.info("DRY RUN -- would %s %.2f @ ~%.2f SL %.2f TP %.2f (signal bar %s)",
                 res.get("evaluated_direction"), res["sizing"]["volume"], res["stop_plan"]["entry"],
                 res["stop_plan"]["sl"], res["stop_plan"]["tp"], sb)
        return out

    evaluated[due] = "send_attempted"                        # never double-send a bar
    _save_state(ls)
    out["sent"] = _send(c, res)
    evaluated[due] = "sent" if out["sent"].get("ok") else "send_failed"
    return out


def _send(c, res: dict) -> dict:
    cfg = get_asset_config(ASSET, timeframe=TF)
    direction = res["evaluated_direction"]
    plan, sz = res["stop_plan"], res["sizing"]
    sl_d = abs(plan["entry"] - plan["sl"])
    tp_d = abs(plan["tp"] - plan["entry"])
    gw = MT5Gateway()
    if not gw.connect():
        return {"ok": False, "error": "MT5 connection failed"}
    try:
        mt5 = gw.raw()
        spec = gw.get_spec(cfg.symbol)
        if _my_positions(gw, cfg.symbol):
            return {"ok": False, "error": "position already open"}
        mode_ok, mode = account_mode_ok(gw.account_info())      # re-check right before sending
        if not mode_ok:
            return {"ok": False, "error": f"account is {mode}, LIVE_ACCOUNT_MODE={allowed_account_mode()}"}
        rc, r = -1, None
        for attempt in range(3):
            tick = gw.get_tick(cfg.symbol)
            px = tick["ask"] if direction == "BUY" else tick["bid"]
            sl = px - sl_d if direction == "BUY" else px + sl_d
            tp = px + tp_d if direction == "BUY" else px - tp_d
            req = build_request(gateway=gw, spec=spec, direction=direction, entry=px,
                                volume=sz["volume"], sl=sl, tp=tp, magic=MAGIC,
                                comment=COMMENT, deviation=DEVIATION)
            chk = gw.order_check(req)
            if not chk or chk["retcode"] != 0:
                _event(c, "order_check_failed", {"check": chk})
                return {"ok": False, "error": f"order_check failed: {chk}"}
            r = mt5.order_send(req)
            rc = _retcode(r)
            _event(c, "order_send", {"attempt": attempt, "retcode": rc,
                                     "comment": getattr(r, "comment", None), "request": req})
            if rc in _DONE or rc not in _RETRY:
                break
        if rc not in _DONE:
            log.error("ORDER FAILED retcode %d %s %s", rc, getattr(r, "comment", None), mt5.last_error())
            return {"ok": False, "retcode": rc, "comment": getattr(r, "comment", None)}

        fill = float(getattr(r, "price", 0.0)) or px
        mine = _my_positions(gw, cfg.symbol)
        pos = mine[0] if mine else None
        if pos is None:
            log.error("order DONE (retcode %d) but no position found -- check the terminal", rc)
            return {"ok": True, "warning": "position not found after fill", "retcode": rc}
        if not _ensure_stops(gw, spec, pos, sl, tp, c):
            _close_position(gw, spec, pos, "no_sl", c)
            return {"ok": False, "error": "broker SL could not be set -- position closed"}
        risk_usd = abs(pos["price_open"] - pos["sl"]) * spec.value_per_price_unit_per_lot * pos["volume"]
        c.execute("""INSERT OR REPLACE INTO trades(ticket, opened_utc, signal_bar_utc, direction, volume,
                     entry, sl, tp, atr, risk_usd) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                  (pos["ticket"], datetime.now(timezone.utc).isoformat(), res["signal_bar_utc"],
                   direction, pos["volume"], pos["price_open"], pos["sl"], pos["tp"],
                   res.get("atr"), round(risk_usd, 2)))
        c.commit()
        st = state_store.load()
        state_store.record_intent(st, cfg.symbol)            # safety.py cooldown
        state_store.save(st)
        log.warning("LIVE OPEN #%d %s %.2f @ %.2f SL %.2f TP %.2f risk $%.2f (signal bar %s)",
                    pos["ticket"], direction, pos["volume"], pos["price_open"], pos["sl"], pos["tp"],
                    risk_usd, res["signal_bar_utc"])
        return {"ok": True, "ticket": pos["ticket"], "fill": fill, "sl": pos["sl"], "tp": pos["tp"],
                "risk_usd": round(risk_usd, 2)}
    finally:
        gw.shutdown()


def _summary(c) -> dict:
    rows = c.execute("SELECT pnl_usd, r_multiple FROM trades WHERE status='CLOSED'").fetchall()
    pnl = [r[0] for r in rows if r[0] is not None]
    R = [r[1] for r in rows if r[1] is not None]
    return {"closed_trades": len(pnl), "net_pl_usd": round(sum(pnl), 2),
            "expectancy_R": round(sum(R) / len(R), 4) if R else None,
            "win_rate": round(sum(1 for x in pnl if x > 0) / len(pnl), 4) if pnl else None}


def run_once() -> dict:
    c = _conn()
    ls = _load_state()
    try:
        snap = manage(c, ls)
        entry = {"skipped": "kill switch active"} if safety.kill_switch_active() else (
            {"skipped": "position open"} if snap.get("open_positions") else maybe_enter(c, ls))
        # keep the evaluated-bar memory small
        ev = ls.get("evaluated_bars", {})
        ls["evaluated_bars"] = dict(sorted(ev.items())[-200:])
        _save_state(ls)
        status = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "LIVE" if safety.live_trading_enabled() else "DRY_RUN",
            "allowed_account_mode": allowed_account_mode(),
            "kill_switch_active": safety.kill_switch_active(),
            "strategy": "momentum_rsi_mtf", "timeframe": TF, "magic": MAGIC,
            "account": snap, "last_entry_check": entry, **_summary(c),
        }
        (_REPORTS / "btc_live_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        return status
    finally:
        c.close()
