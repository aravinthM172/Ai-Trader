"""
Live-vs-backtest twin check: is a bad live run bad luck, or is the live trader not doing what the
backtest does?

Replays the frozen H1 rule (strategy/btc_h1_signal.generate, the same builder live_multi calls) on
every completed H1 bar since --since, opens at the next bar's open with the live stop plan
(2 ATR stop / 3 ATR target, 96-bar time exit, one position per symbol) and walks the bars to the
exit.  Then pairs each backtest trade with the live trade on the same symbol + signal bar.

    python -m tools.twin_check                         # since the first live trade
    python -m tools.twin_check --since 2026-10-04T16:00

Verdict per trade:
  match        same signal bar, same direction, same exit type (target / stop / time)
  exit differs same entry, different exit (spread, broker prices, manual / prop close)
  missed       backtest traded, live did not (bot down, guard / news block, account switch)
  extra        live traded, backtest did not -- the one that points at a bug
Archived Valetax trades are replayed on the FundingPips feed (DAX40.vx -> GER40); XAUEUR.vx has no
FundingPips twin and is reported as not replayable.

Read-only.  Writes reports/twin_check.json.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.btc_strategies import base_atr
from execution.live_multi import MAX_HOLD_H, SL_ATR, TP_ATR, completed_bars
from strategy import btc_h1_signal

ROOT = Path(__file__).resolve().parents[1]
DBS = {"valetax": ROOT / "state" / "archive_valetax_2026-10-06" / "multi_live.sqlite",
       "fundingpips": ROOT / "state" / "multi_live.sqlite"}
OUT = ROOT / "reports" / "twin_check.json"
FEED = {"BTCUSD.vx": "BTCUSD", "XAUUSD.vx": "XAUUSD", "DAX40.vx": "GER40", "XAUEUR.vx": None,
        "BTCUSD": "BTCUSD", "XAUUSD": "XAUUSD", "GER40": "GER40"}
WARMUP = 400                       # live_multi feeds generate() the last 400 bars


# -- pure helpers (unit-tested) ---------------------------------------------------------
def simulate_exit(direction: str, entry: float, atr: float, bars: pd.DataFrame) -> tuple[str, float, int]:
    """(exit reason, R, bars held) for a trade opened at `entry` on bars[0].  Stop checked before
    target inside a bar (conservative, as the backtest engine does)."""
    sl_d, tp_d = SL_ATR * atr, TP_ATR * atr
    sign = 1 if direction == "BUY" else -1
    for i, (h, l) in enumerate(zip(bars["high"].to_numpy(float), bars["low"].to_numpy(float))):
        if i >= MAX_HOLD_H:
            px = float(bars["open"].iloc[i])
            return "time", sign * (px - entry) / sl_d, i
        adverse, favour = (l, h) if sign == 1 else (h, l)
        if sign * (adverse - entry) <= -sl_d:
            return "stop", -1.0, i
        if sign * (favour - entry) >= tp_d:
            return "target", tp_d / sl_d, i
    return "open", sign * (float(bars["close"].iloc[-1]) - entry) / sl_d, len(bars)


def backtest_trades(symbol: str, df: pd.DataFrame, since: pd.Timestamp,
                    raw: dict | None = None) -> list[dict]:
    """Every trade the frozen rule takes on `df` (completed bars) with a signal bar >= since.
    `raw`, if given, collects the signal on EVERY bar ({signal bar iso: BUY/SELL}), including bars
    the backtest skipped because it was already in a position."""
    out, busy_until = [], None
    times = pd.to_datetime(df["time"], utc=True)
    for k in range(WARMUP, len(df) - 1):
        if times.iloc[k] < since:
            continue
        busy = busy_until is not None and k < busy_until
        if busy and raw is None:
            continue
        window = df.iloc[k - WARMUP + 1:k + 1].reset_index(drop=True)
        sig = btc_h1_signal.generate(symbol, window)
        if sig.decision == "HOLD":
            continue
        if raw is not None:
            raw[times.iloc[k].isoformat()] = sig.decision
        if busy:
            continue
        atr = float(base_atr(window["high"].to_numpy(float), window["low"].to_numpy(float),
                             window["close"].to_numpy(float), 14)[-1])
        entry = float(df["open"].iloc[k + 1])
        reason, r, held = simulate_exit(sig.decision, entry, atr, df.iloc[k + 1:].reset_index(drop=True))
        busy_until = k + 1 + held + 1
        out.append({"symbol": symbol, "signal_bar_utc": times.iloc[k].isoformat(), "direction": sig.decision,
                    "entry": entry, "atr": atr, "exit_reason": reason, "r": round(r, 3)})
    return out


def _exit_kind(live_reason: str | None) -> str:
    r = (live_reason or "").lower()
    return "target" if "target" in r else "stop" if "stop" in r else "time" if "time" in r else r or "open"


def pair(bt: list[dict], live: list[dict], raw: dict | None = None) -> list[dict]:
    """Join backtest and live trades on (feed symbol, signal bar).  `raw` = {feed: {bar: BUY/SELL}}
    splits 'extra' into 'extra (backtest in a trade)' -- live was flat when the backtest was not,
    e.g. after a missed signal or an account switch -- and 'extra (no signal)', the real red flag."""
    key = lambda t: (t["feed"], pd.Timestamp(t["signal_bar_utc"]).isoformat())
    bmap = {key(t): t for t in bt}
    lmap = {key(t): t for t in live if t.get("signal_bar_utc")}
    rows = []
    for k in sorted(set(bmap) | set(lmap), key=lambda x: x[1]):
        b, l = bmap.get(k), lmap.get(k)
        if b and l:
            same_dir = b["direction"] == l["direction"]
            lk = _exit_kind(l.get("exit_reason"))
            verdict = ("direction differs" if not same_dir else
                       "match" if lk == b["exit_reason"] or "open" in (lk, b["exit_reason"]) else "exit differs")
        elif b:
            verdict = "missed"
        else:
            sig = (raw or {}).get(k[0], {}).get(k[1])
            verdict = ("extra (backtest in a trade)" if sig == l["direction"] else
                       "extra (direction differs)" if sig else "extra (no signal)")
        rows.append({"symbol": k[0], "signal_bar_utc": k[1], "verdict": verdict,
                     "bt_direction": b and b["direction"], "bt_exit": b and b["exit_reason"], "bt_r": b and b["r"],
                     "live_symbol": l and l["symbol"], "live_direction": l and l["direction"],
                     "live_exit": l and _exit_kind(l.get("exit_reason")), "live_r": l and l.get("r_multiple"),
                     "account": l and l.get("account")})
    return rows


# -- IO --------------------------------------------------------------------------------------
def live_trades() -> list[dict]:
    out = []
    for acct, path in DBS.items():
        if not path.exists():
            continue
        c = sqlite3.connect(path)
        c.row_factory = sqlite3.Row
        for r in c.execute("SELECT * FROM trades ORDER BY opened_utc"):
            t = dict(r)
            t["account"], t["feed"] = acct, FEED.get(t["symbol"])
            out.append(t)
        c.close()
    return out


def run(since: str | None = None) -> dict:
    from mt5.gateway import MT5Gateway
    live = live_trades()
    not_replayable = [t for t in live if not t["feed"]]
    live = [t for t in live if t["feed"]]
    start = pd.Timestamp(since, tz="UTC") if since else min(pd.Timestamp(t["signal_bar_utc"]) for t in live)
    gw = MT5Gateway()
    if not gw.connect():
        raise SystemExit("MT5 connection failed")
    try:
        now = datetime.now(timezone.utc)
        bt, raw = [], {}
        for feed in sorted({t["feed"] for t in live} | {"BTCUSD", "XAUUSD", "GER40"}):
            df = gw.get_rates(feed, "H1", 1500)
            if df is None or df.empty:
                continue
            for t in backtest_trades(feed, completed_bars(df, now), start, raw=raw.setdefault(feed, {})):
                bt.append({**t, "feed": feed})
    finally:
        gw.shutdown()
    rows = pair(bt, live, raw)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    paired = [r for r in rows if r["live_r"] is not None and r["bt_r"] is not None and r["live_exit"] != "open"]
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "since_utc": start.isoformat(),
              "counts": counts, "rows": rows,
              "backtest_avg_r": round(float(np.mean([b["r"] for b in bt if b["exit_reason"] != "open"])), 3)
              if any(b["exit_reason"] != "open" for b in bt) else None,
              "paired_live_avg_r": round(float(np.mean([r["live_r"] for r in paired])), 3) if paired else None,
              "paired_bt_avg_r": round(float(np.mean([r["bt_r"] for r in paired])), 3) if paired else None,
              "not_replayable": [{"symbol": t["symbol"], "signal_bar_utc": t["signal_bar_utc"],
                                  "r": t.get("r_multiple")} for t in not_replayable]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--since", help="UTC start, e.g. 2026-10-04T16:00 (default: first live trade)")
    rep = run(ap.parse_args().since)
    print(f"twin check since {rep['since_utc']}: {rep['counts']}")
    for r in rep["rows"]:
        print(f"  {r['signal_bar_utc'][:16]} {r['symbol']:7} {r['verdict']:27} "
              f"backtest {r['bt_direction'] or '-':4} {r['bt_exit'] or '-':6} {r['bt_r'] if r['bt_r'] is not None else '':>6}  "
              f"live {r['live_direction'] or '-':4} {r['live_exit'] or '-':6} {r['live_r'] if r['live_r'] is not None else '':>6} "
              f"{r['account'] or ''}")
    print(f"avg R -- all backtest trades {rep['backtest_avg_r']} | paired: live {rep['paired_live_avg_r']} "
          f"vs backtest {rep['paired_bt_avg_r']}")
    if rep["not_replayable"]:
        print(f"not replayable (no FundingPips feed): {rep['not_replayable']}")


if __name__ == "__main__":
    main()
