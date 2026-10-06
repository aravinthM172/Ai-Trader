"""
Paper (dry-run) forward test of the two round-3 watch-list ideas.  NEVER sends orders.

Sleeves (frozen, exactly as tested in backtest/strategy_round3_daily.py):
  rsi2_indices    Connors RSI(2) pullback, long only, on the 10 index CFDs
  trend_gold_btc  Turtle 55-day breakout AND Clenow 50/100 EMA trend, on XAUUSD and BTCUSD

How it works: every run pulls the broker's completed D1 bars, re-runs the frozen backtest
code on them (a "shadow backtest"), and books every trade whose entry is on or after the
paper start date as a paper trade.  Same fills, costs and swap as the backtest, so the paper
result is a clean out-of-sample test of the EDGE (not of live execution quality).
It also reports what the strategy would do at the next bar (enter / exit / stop levels) and
sends a Telegram note when that changes or a paper trade opens or closes.

PROMOTION RULE (pre-registered): a sleeve may be proposed for real money only when it has
>= 30 closed paper trades AND mean R > 0 AND the owner approves.  Expect ~10 months for
rsi2_indices (~36 trades/yr) and ~1.5 years for trend_gold_btc (~20 trades/yr).

    python -m execution.paper_daily            # one pass
    python -m execution.paper_daily --loop 3600

Writes reports/paper_daily_status.json, reports/paper_daily_trades.csv,
reports/paper_daily_state.json.  Read-only on MT5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from backtest import strategy_round3_daily as r3
from backtest.multi_symbol_scan import spec_of
from common import notify
from common.logging_setup import get_logger

load_dotenv()
log = get_logger("paper.daily")
REPORTS = ROOT / "reports"
STATUS = REPORTS / "paper_daily_status.json"
TRADES = REPORTS / "paper_daily_trades.csv"
STATE = REPORTS / "paper_daily_state.json"
MIN_PAPER_TRADES = 30

INDICES = ["DAX40.vx", "NAS100.vx", "SP500.vx", "US30.vx", "JPN225.vx",
           "HK50.vx", "UK100.vx", "FRA40.vx", "EU50.vx", "AUS200.vx"]
SLEEVES = {
    "rsi2_indices": [("connors_rsi2", s) for s in INDICES],
    "trend_gold_btc": [(st, s) for st in ("turtle55", "clenow_trend") for s in ("XAUUSD.vx", "BTCUSD.vx")],
}


def fetch(gw, symbol: str) -> pd.DataFrame | None:
    d = gw.get_rates(symbol, "D1", 3000)
    if d is None or len(d) < 260:
        return None
    d = d.iloc[:-1]                                   # last row is the forming bar
    d = d.assign(date=(pd.to_datetime(d["time"], utc=True) + pd.Timedelta(hours=6)).dt.tz_convert(None).dt.normalize())
    return r3._clean(d.set_index("date")[["open", "high", "low", "close"]])


def next_action(st: str, s: dict, last_close: float) -> str:
    if s.get("pos"):
        side = "LONG" if s["pos"] > 0 else "SHORT"
        if s.get("exit_next_open"):
            return f"{side} open -> EXIT at next open"
        if "stop" in s:
            return f"{side} open, stop {s['stop']:.5g}"
        return f"{side} open, exit if a close crosses {s['trail_close']:.5g}"
    if st == "turtle55":
        return f"flat; buy stop {s['buy_stop']:.5g} / sell stop {s['sell_stop']:.5g}"
    if s.get("enter_next_open"):
        side = "LONG" if s["enter_next_open"] > 0 else "SHORT"
        return f"ENTER {side} at next open (stop distance {s['risk_if_entered']:.5g})"
    return "flat; no signal"


def run_once(gw, st_file: dict) -> dict:
    mt5 = gw.raw()
    start = pd.Timestamp(st_file.setdefault("paper_start", datetime.now(timezone.utc).strftime("%Y-%m-%d")))
    expected = {}
    rep3 = REPORTS / "strategy_round3.json"
    if rep3.exists():
        res = json.loads(rep3.read_text())["results"]
        expected = {f"{st}|{sym}": r.get("broker_R") for st, rs in res.items() for sym, r in rs.items()}

    pairs, rows, alerts = {}, [], []
    for sleeve, members in SLEEVES.items():
        for st, sym in members:
            key = f"{st}|{sym}"
            info = mt5.symbol_info(sym)
            d = fetch(gw, sym)
            if info is None or d is None:
                pairs[key] = dict(sleeve=sleeve, error="no data")
                continue
            sp = spec_of(info)
            sp["price"] = float(info.bid or d.close.iloc[-1])
            sf = r3.broker_spread_frac(sp)                # same cost model as the backtest
            s: dict = {}
            t = r3.trade_table(d, r3.STRATEGIES[st], sf, sp["swap_long"], sp["swap_short"], state=s)
            paper = t[t.entry >= start]
            for _, x in paper.iterrows():
                rows.append(dict(sleeve=sleeve, strategy=st, symbol=sym, entry=x.entry.date(), exit=x.exit.date(),
                                 dir=int(x.dir), R=round(float(x.R), 4)))
            open_R = None
            if s.get("pos") and d.index[s["entry_i"]] >= start:
                cost = sf * s["entry_px"] * (1 + 2 * r3.SLIP_FRAC)
                open_R = round((s["pos"] * (d.close.iloc[-1] - s["entry_px"]) - cost) / s["risk"], 3)
            act = next_action(st, s, float(d.close.iloc[-1]))
            if s.get("pos") and d.index[s["entry_i"]] < start:
                act += " (opened before paper start -- not counted)"
            pairs[key] = dict(sleeve=sleeve, last_bar=str(d.index[-1].date()), last_close=float(d.close.iloc[-1]),
                              next_action=act, open_paper_R=open_R, paper_trades=int(len(paper)),
                              paper_R=round(float(paper.R.sum()), 3), backtest_broker_meanR=expected.get(key))
            prev = st_file.setdefault("last_action", {}).get(key)
            if prev != act and ("ENTER" in act or "EXIT" in act or (prev and "open" in act and "open" not in prev)):
                alerts.append(f"[paper D1] {key}: {act}")
            st_file["last_action"][key] = act

    tr = pd.DataFrame(rows, columns=["sleeve", "strategy", "symbol", "entry", "exit", "dir", "R"])
    known = set(st_file.get("known_trades", []))
    for _, x in tr.iterrows():
        k = f"{x.strategy}|{x.symbol}|{x.entry}"
        if k not in known:
            alerts.append(f"[paper D1] CLOSED {x.strategy} {x.symbol} {'BUY' if x.dir > 0 else 'SELL'} "
                          f"{x.entry} -> {x.exit}: {x.R:+.2f} R")
            known.add(k)
    st_file["known_trades"] = sorted(known)

    sleeves = {}
    for sleeve in SLEEVES:
        R = tr.loc[tr.sleeve == sleeve, "R"]
        sleeves[sleeve] = dict(closed=int(len(R)), mean_R=round(float(R.mean()), 4) if len(R) else None,
                               total_R=round(float(R.sum()), 3), win_rate=round(float((R > 0).mean()), 3) if len(R) else None,
                               ready_for_review=bool(len(R) >= MIN_PAPER_TRADES and R.mean() > 0))
    status = dict(generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), mode="PAPER (no orders)",
                  paper_start=str(start.date()), promotion_rule=f">= {MIN_PAPER_TRADES} closed paper trades, mean R > 0, owner approval",
                  sleeves=sleeves, pairs=pairs)
    REPORTS.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2))
    tr.to_csv(TRADES, index=False)
    for a in alerts:
        log.info(a)
        notify.send(a)
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=float, default=0, help="seconds between passes (0 = run once)")
    a = ap.parse_args()
    from mt5.gateway import MT5Gateway
    while True:
        st_file = json.loads(STATE.read_text()) if STATE.exists() else {}
        gw = MT5Gateway()
        try:
            if not gw.connect():
                log.error("MT5 connection failed")
            else:
                s = run_once(gw, st_file)
                STATE.write_text(json.dumps(st_file, indent=2))
                log.info("pass done: %s", {k: (v["closed"], v["total_R"]) for k, v in s["sleeves"].items()})
        except Exception as e:
            log.exception("pass failed: %s", e)
        finally:
            gw.shutdown()
        if not a.loop:
            return 0
        time.sleep(a.loop)


if __name__ == "__main__":
    raise SystemExit(main())
