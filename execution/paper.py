"""
PHASE 10 / H1 forward test -- paper mode.  NEVER calls order_send().

Reads live BTCUSD.vx candles for the SELECTED timeframe, runs the decision
pipeline (real Valetax spread, every safety gate), and if a trade would be
placed it records a HYPOTHETICAL position, then monitors and closes it using
live bid/ask.  Every event is persisted to a per-timeframe SQLite DB + JSON.

    python run_btc.py --paper                       # M5 (default, placeholder rule strategy)
    python run_btc.py --paper --timeframe H1        # validated momentum_rsi_mtf on H1
    python run_btc.py --paper --timeframe H1 --loop 300

State : state/btc_paper_<TF>.sqlite
Status: reports/btc_paper_status_<TF>.json   (+ reports/btc_paper_status.json = latest)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from common.logging_setup import get_logger
from config.assets import get_asset_config
from execution.pipeline import run_pipeline
from mt5.gateway import MT5Gateway

log = get_logger("btc.paper", filename="btc_paper.log")

_ROOT = Path(__file__).resolve().parents[1]
_STATE = _ROOT / "state"
_REPORTS = _ROOT / "reports"
_STATE.mkdir(exist_ok=True)

VALUE_PER_UNIT_PER_LOT = 1.0
SLIPPAGE_PER_SIDE = 1.0
_MAX_HOLD_HOURS = {"M5": 24, "M15": 48, "H1": 96, "H4": 240}


def _db_path(tf: str) -> Path:
    return _STATE / f"btc_paper_{tf.upper()}.sqlite"


def _status_path(tf: str) -> Path:
    return _REPORTS / f"btc_paper_status_{tf.upper()}.json"


def _conn(tf: str) -> sqlite3.Connection:
    c = sqlite3.connect(_db_path(tf))
    c.execute("""CREATE TABLE IF NOT EXISTS positions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        opened_utc TEXT, timeframe TEXT, strategy TEXT, symbol TEXT,
        direction TEXT, signal_bar_utc TEXT,
        entry REAL, sl REAL, tp REAL, volume REAL,
        atr REAL, spread REAL, est_risk_usd REAL, est_cost_usd REAL,
        reason_entry TEXT, note TEXT,
        status TEXT DEFAULT 'OPEN',
        closed_utc TEXT, exit REAL, exit_reason TEXT, pnl_usd REAL,
        r_multiple REAL, bars_held INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, timeframe TEXT, kind TEXT, payload TEXT)""")
    return c


def _event(c, tf, kind, payload):
    c.execute("INSERT INTO events(ts_utc,timeframe,kind,payload) VALUES(?,?,?,?)",
              (datetime.now(timezone.utc).isoformat(), tf, kind, json.dumps(payload, default=str)))


def _open_positions(c):
    cur = c.execute("SELECT * FROM positions WHERE status='OPEN'")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _monitor(c, tf, gw):
    tick = gw.get_tick("BTCUSD.vx")
    closed = []
    if not tick:
        return closed
    bid, ask = tick["bid"], tick["ask"]
    now = datetime.now(timezone.utc)
    max_hold_h = _MAX_HOLD_HOURS.get(tf.upper(), 24)
    for p in _open_positions(c):
        d = 1 if p["direction"] == "BUY" else -1
        reason = exit_px = None
        if d == 1:
            if bid <= p["sl"]:
                reason, exit_px = "stop", p["sl"] - SLIPPAGE_PER_SIDE
            elif bid >= p["tp"]:
                reason, exit_px = "target", p["tp"]
        else:
            if ask >= p["sl"]:
                reason, exit_px = "stop", p["sl"] + SLIPPAGE_PER_SIDE
            elif ask <= p["tp"]:
                reason, exit_px = "target", p["tp"]
        held_h = (now - datetime.fromisoformat(p["opened_utc"])).total_seconds() / 3600.0
        if reason is None and held_h >= max_hold_h:
            reason, exit_px = "time", (bid if d == 1 else ask) - d * SLIPPAGE_PER_SIDE
        if reason:
            pnl = (exit_px - p["entry"]) * d * VALUE_PER_UNIT_PER_LOT * p["volume"]
            r_mult = (pnl / p["est_risk_usd"]) if p["est_risk_usd"] else None
            step_h = {"M5": 1 / 12, "M15": 0.25, "H1": 1, "H4": 4}.get(tf.upper(), 1 / 12)
            c.execute("""UPDATE positions SET status='CLOSED', closed_utc=?, exit=?, exit_reason=?,
                         pnl_usd=?, r_multiple=?, bars_held=? WHERE id=?""",
                      (now.isoformat(), round(exit_px, 2), reason, round(pnl, 4),
                       round(r_mult, 4) if r_mult is not None else None,
                       int(held_h / step_h), p["id"]))
            _event(c, tf, "paper_close",
                   {"id": p["id"], "reason_exit": reason, "exit": round(exit_px, 2),
                    "pnl_usd": round(pnl, 4), "r_multiple": r_mult})
            log.info("[%s] PAPER CLOSE #%d %s %s @ %.2f  pnl $%.2f  R %.2f",
                     tf, p["id"], p["direction"], reason, exit_px, pnl, r_mult or 0.0)
            closed.append({"id": p["id"], "reason_exit": reason, "pnl_usd": round(pnl, 4),
                           "r_multiple": round(r_mult, 4) if r_mult is not None else None})
    return closed


def _maybe_open(c, tf, res, ignore_spread):
    plan = res.get("stop_plan")
    sz = res.get("sizing")
    direction = res.get("evaluated_direction")
    sig = res.get("signal", {})
    if not (plan and sz and direction):
        return None
    if not plan.get("accepted") or not sz.get("accepted") or sz.get("volume", 0) <= 0:
        return None
    safe = res.get("safety", {})
    blocking = [r for r in safe.get("reasons", []) if not (ignore_spread and "spread" in r)]
    if blocking:
        return None
    if _open_positions(c):
        return None
    # one hypothetical trade per signal bar -- never re-enter a bar we've already
    # traded (e.g. after it stopped out earlier in the same still-active bar).
    sb = res.get("signal_bar_utc")
    if sb and c.execute(
            "SELECT 1 FROM positions WHERE signal_bar_utc=? LIMIT 1", (sb,)).fetchone():
        return None
    now = datetime.now(timezone.utc).isoformat()
    cost = (plan["spread_price"] + 2 * SLIPPAGE_PER_SIDE) * VALUE_PER_UNIT_PER_LOT * sz["volume"]
    row = dict(
        opened_utc=now, timeframe=tf.upper(), strategy=res.get("strategy"), symbol="BTCUSD.vx",
        direction=direction, signal_bar_utc=res.get("signal_bar_utc"),
        entry=plan["entry"], sl=plan["sl"], tp=plan["tp"], volume=sz["volume"],
        atr=res.get("atr"), spread=plan["spread_price"],
        est_risk_usd=sz["est_loss_at_sl"], est_cost_usd=round(cost, 4),
        reason_entry=sig.get("reason"),
        note=("spread_filter_ignored" if ignore_spread else "normal"),
    )
    c.execute(f"""INSERT INTO positions({','.join(row)}) VALUES({','.join('?' for _ in row)})""",
              tuple(row.values()))
    pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    _event(c, tf, "paper_open", {"id": pid, **row})
    log.info("[%s] PAPER OPEN #%d %s [%s] vol %.2f entry %.2f SL %.2f TP %.2f  signal_bar %s",
             tf, pid, direction, res.get("strategy"), sz["volume"], plan["entry"],
             plan["sl"], plan["tp"], res.get("signal_bar_utc"))
    return {"id": pid, "direction": direction, "signal_bar_utc": res.get("signal_bar_utc")}


def _report(c, tf, strategy=None):
    cur = c.execute("SELECT * FROM positions WHERE status='CLOSED' ORDER BY id")
    cols = [d[0] for d in cur.description]
    closed = [dict(zip(cols, r)) for r in cur.fetchall()]
    opn = _open_positions(c)
    pnls = [p["pnl_usd"] for p in closed if p["pnl_usd"] is not None]
    Rs = [p["r_multiple"] for p in closed if p["r_multiple"] is not None]
    wins = [x for x in pnls if x > 0]
    now = datetime.now(timezone.utc)

    def since(days):
        cut = (now - timedelta(days=days)).isoformat()
        xs = [p["pnl_usd"] for p in closed if p["closed_utc"] and p["closed_utc"] >= cut]
        return dict(trades=len(xs), net_usd=round(sum(xs), 2))

    try:
        db_str = str(_db_path(tf).relative_to(_ROOT))
    except ValueError:
        db_str = str(_db_path(tf))
    return {
        "generated_utc": now.isoformat(),
        "timeframe": tf.upper(),
        "strategy": (closed[-1]["strategy"] if closed else (opn[0]["strategy"] if opn else strategy)),
        "db": db_str,
        "open_positions": opn,
        "closed_trades": len(closed),
        "net_pl_usd": round(sum(pnls), 2),
        "expectancy_R": round(sum(Rs) / len(Rs), 4) if Rs else None,
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else None,
        "avg_win_usd": round(sum(wins) / len(wins), 4) if wins else 0.0,
        "avg_loss_usd": round(sum(x for x in pnls if x <= 0) / max(1, len(pnls) - len(wins)), 4) if pnls else 0.0,
        "largest_loss_usd": round(min(pnls), 4) if pnls else None,
        "daily": since(1),
        "weekly": since(7),
        "trade_log": closed[-25:],
        "NOTE": "PAPER ONLY -- order_send() is never called.",
    }


def run_once(*, timeframe: str = "M5", ignore_spread: bool = False,
             bars: int | None = None, force_direction=None,
             sim_balance: float | None = None) -> dict:
    tf = timeframe.upper()
    cfg = get_asset_config("BTC", timeframe=tf)
    c = _conn(tf)
    gw = MT5Gateway()
    if not gw.connect():
        c.close()
        return {"error": "MT5 connection failed", "timeframe": tf}
    try:
        closed = _monitor(c, tf, gw)
        res = run_pipeline("BTC", timeframe=tf, history_bars=bars,
                           force_direction=force_direction, ignore_spread_filter=ignore_spread,
                           paper_sim_balance=sim_balance)
        # HARD GUARD: the pipeline must have actually used this timeframe
        if res.get("timeframe", "").upper() != tf:
            _event(c, tf, "paper_error", {"expected_tf": tf, "got_tf": res.get("timeframe")})
            c.commit(); c.close()
            raise RuntimeError(f"pipeline ran on {res.get('timeframe')} not {tf} -- aborting paper pass")
        opened = _maybe_open(c, tf, res, ignore_spread)
        _event(c, tf, "paper_pass", {
            "timeframe": res.get("timeframe"), "strategy": res.get("strategy"),
            "history_bars_returned": res.get("history_bars_returned"),
            "signal": res.get("signal", {}).get("decision"),
            "signal_bar_utc": res.get("signal_bar_utc"),
            "decision": res.get("decision"),
            "spread_ok": res.get("spread_filter", {}).get("ok"),
            "opened": opened, "closed": closed})
        c.commit()
        status = _report(c, tf, res.get("strategy"))
        status["last_pass"] = {
            "timeframe": res.get("timeframe"), "strategy": res.get("strategy"),
            "sizing_balance": res.get("sizing_balance"),
            "real_account_balance": res.get("real_account_balance"),
            "history_bars_returned": res.get("history_bars_returned"),
            "signal": res.get("signal", {}).get("decision"),
            "signal_confidence": res.get("signal", {}).get("confidence"),
            "signal_reason": res.get("signal", {}).get("reason"),
            "signal_bar_utc": res.get("signal_bar_utc"),
            "atr": res.get("atr"),
            "decision": res.get("decision"),
            "spread_filter_ok": res.get("spread_filter", {}).get("ok"),
            "spread_to_atr": res.get("spread_filter", {}).get("spread_to_atr"),
            "opened": opened, "closed": closed, "ignore_spread": ignore_spread,
            "live_trading_enabled": res.get("live_trading_enabled"),
            "kill_switch_active": res.get("kill_switch_active"),
        }
        _status_path(tf).write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        (_REPORTS / "btc_paper_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
        return status
    finally:
        gw.shutdown()
        c.close()
