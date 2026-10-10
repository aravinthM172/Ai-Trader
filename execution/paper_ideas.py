"""
Paper (dry-run) forward test of the two research ideas of 2026-10-10 -- NEVER sends orders.

    python -m execution.paper_ideas            one pass (the hourly scheduled task "GoldAI Paper Ideas")
    python -m execution.paper_ideas --report   print the current standing

Ideas (rules exactly as backtest/strategy_round5.py, strategy_round6.py and strategy_round7.py; the functions here
are self-contained so the VPS needs no research data, and tests/test_paper_ideas.py checks them against the backtest):
  breakout_nextday   BTCUSD, ETHUSD   first H1 close beyond the day's open +/- 0.5 x yesterday's range: trade the
                                      break at the next open, stop at the day's open, flat at the next day's first bar.
  breakout_trail     BTCUSD, ETHUSD   same entry; no next-day exit, the stop moves to the previous day's low / high
                                      at each new day, 10 days at most.
  gold_asia          XAUUSD           buy the 19:00 New York bar's open (one hour after the daily reopening), close at
                                      07:00 UTC, protective stop 3 ATR.
Day = broker day (rollover 21:00 UTC).  1 R = distance from entry to the first stop.  Costs: the symbol's spread at
the time of the pass x 1.5 (spread + slippage); swaps and commission are not included.

Every pass reloads the H1 candles from one fixed start (PAPER_START - WARMUP_DAYS), so the trade list is the same on
every run and only grows.  Trades entered before PAPER_START are warm-up and are not counted.
Writes state/paper_ideas.sqlite (table trades) and reports/paper_ideas_status.json.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.logging_setup import get_logger
from strategy.btc_features import _atr

log = get_logger("paper.ideas", filename="paper_ideas.log")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "state" / "paper_ideas.sqlite"
STATUS = ROOT / "reports" / "paper_ideas_status.json"
PAPER_START = pd.Timestamp("2026-10-10 10:00", tz="UTC")       # first hour after the rules were fixed
WARMUP_DAYS = 40
IDEAS = {"breakout_nextday": ("BTCUSD", "ETHUSD"), "breakout_trail": ("BTCUSD", "ETHUSD"), "gold_asia": ("XAUUSD",)}
MIN_RISK_ATR, MAX_RISK_ATR = 0.5, 6.0
USD_PER_R = 12.5                                               # $5k x 0.25 %, for the dollar column only
COLS = ["entry", "exit", "dir", "entry_px", "exit_px", "stop_px", "risk_px", "reason", "status", "R_gross"]


def _prev_group(per: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Per-group value -> for each bar the value of the PREVIOUS group."""
    uniq = pd.unique(key)
    pos = pd.Series(np.arange(len(uniq)), index=uniq).reindex(key).to_numpy()
    return np.concatenate([[np.nan], per[:-1]])[pos]


def _simulate(t, o, h, l, c, atr, d, stop, flat, lvl_l, lvl_s, hold) -> pd.DataFrame:
    """One position at a time.  d[j] != 0: enter at the open of bar j + 1 with stop[j].  flat[k]: close at the open
    of bar k.  lvl_l / lvl_s: stop levels known at the open of bar k (the stop only tightens).  A trade that is
    still running at the last bar is reported as OPEN, marked at the last close."""
    n, rows, i = len(c), [], 1
    while i < n:
        j = i - 1
        di, a = int(d[j]), atr[j]
        if di == 0 or not (a > 0):
            i += 1
            continue
        ep, sp = o[i], stop[j]
        risk = di * (ep - sp)
        if not (risk >= MIN_RISK_ATR * a) or risk > MAX_RISK_ATR * a:
            i += 1
            continue
        x, xp, why, at_open = -1, 0.0, "time", False
        for k in range(i, min(n, i + hold)):
            if k > i:
                if flat[k]:
                    x, xp, why, at_open = k, o[k], "flat", True
                    break
                lv = lvl_l[k] if di > 0 else lvl_s[k]
                if np.isfinite(lv) and di * (lv - sp) > 0:
                    sp = lv
                if di * (o[k] - sp) <= 0:
                    x, xp, why, at_open = k, o[k], "stop", True
                    break
            if (di > 0 and l[k] <= sp) or (di < 0 and h[k] >= sp):
                x, xp, why = k, sp, "stop"
                break
        status = "CLOSED"
        if x < 0:
            if i + hold <= n - 1:
                x, xp, at_open = i + hold, o[i + hold], True
            else:
                x, xp, why, status = n - 1, c[n - 1], "running", "OPEN"
        rows.append((t[i], t[x], di, ep, xp, sp, risk, why, status, di * (xp - ep) / risk))
        if status == "OPEN":
            break
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=COLS)


def _arrays(df: pd.DataFrame):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return df["time"].to_numpy(), o, h, l, c, _atr(h, l, c, 14)[0]


def breakout_trades(df: pd.DataFrame, trail: bool) -> pd.DataFrame:
    t, o, h, l, c, atr = _arrays(df)
    n = len(c)
    day = (df["time"] + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize().to_numpy()
    g = pd.DataFrame(dict(o=o, h=h, l=l)).groupby(day, sort=False)
    pdh, pdl = _prev_group(g.h.max().to_numpy(), day), _prev_group(g.l.min().to_numpy(), day)
    dopen = g.o.transform("first").to_numpy()
    up, dn = c > dopen + 0.5 * (pdh - pdl), c < dopen - 0.5 * (pdh - pdl)
    hit = up | dn
    first = hit & (pd.Series(hit.astype(int)).groupby(day, sort=False).cumsum().to_numpy() == 1)
    d = np.where(first & up, 1, np.where(first & dn, -1, 0))
    newday = np.concatenate([[True], day[1:] != day[:-1]])
    nan = np.full(n, np.nan)
    if trail:
        return _simulate(t, o, h, l, c, atr, d, dopen, np.zeros(n, bool), pdl, pdh, 240)
    return _simulate(t, o, h, l, c, atr, d, dopen, newday, nan, nan, 30)


def gold_asia_trades(df: pd.DataFrame) -> pd.DataFrame:
    t, o, h, l, c, atr = _arrays(df)
    n = len(c)
    enter = df["time"].dt.tz_convert("America/New_York").dt.hour.to_numpy() == 19
    d = np.zeros(n, int)
    d[:-1] = enter[1:].astype(int)                               # the order is placed on the bar before
    nan = np.full(n, np.nan)
    return _simulate(t, o, h, l, c, atr, d, c - 3 * atr, df["time"].dt.hour.to_numpy() == 7, nan, nan, 120)


RUN = {"breakout_nextday": lambda df: breakout_trades(df, False), "breakout_trail": lambda df: breakout_trades(df, True),
       "gold_asia": gold_asia_trades}


# -- one pass ------------------------------------------------------------------------------------
def candles(gw, symbol: str, now: datetime) -> pd.DataFrame | None:
    start = PAPER_START - pd.Timedelta(days=WARMUP_DAYS)
    d = gw.get_rates(symbol, "H1", int((pd.Timestamp(now) - start).total_seconds() // 3600) + 100)
    if d is None or len(d) < 200:
        return None
    d = d[(d["time"] >= start) & (d["time"] + pd.Timedelta(hours=1) <= pd.Timestamp(now))]      # closed bars only
    return d[["time", "open", "high", "low", "close"]].reset_index(drop=True)


def run_once(gw, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    frames, data = [], {}
    for idea, syms in IDEAS.items():
        for s in syms:
            if s not in data:
                data[s] = candles(gw, s, now)
            df = data[s]
            if df is None:
                log.warning("%s: no candles", s)
                continue
            spec = gw.get_spec(s)
            cost_px = 1.5 * float(spec.spread_points) * float(spec.point) if spec else 0.0
            tr = RUN[idea](df)
            tr = tr[pd.to_datetime(tr["entry"], utc=True) >= PAPER_START]
            frames.append(tr.assign(idea=idea, symbol=s, cost_R=cost_px / tr["risk_px"] if len(tr) else []))
    tr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS + ["idea", "symbol", "cost_R"])
    if len(tr):
        tr["R"] = tr["R_gross"] - tr["cost_R"]
        tr["entry"], tr["exit"] = (pd.to_datetime(tr[k], utc=True).dt.strftime("%Y-%m-%d %H:%M") for k in ("entry", "exit"))
    else:
        tr["R"] = []
    return save(tr, now)


def save(tr: pd.DataFrame, now: datetime) -> dict:
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("CREATE TABLE IF NOT EXISTS trades(idea TEXT, symbol TEXT, entry TEXT, exit TEXT, dir INTEGER, entry_px REAL, "
                  "exit_px REAL, stop_px REAL, risk_px REAL, reason TEXT, status TEXT, R_gross REAL, cost_R REAL, R REAL, "
                  "first_seen TEXT, PRIMARY KEY(idea, symbol, entry))")
        for r in tr.itertuples(index=False):
            # a closed trade is final: its cost is the spread of the pass that first saw it closed
            old = c.execute("SELECT status, cost_R, first_seen FROM trades WHERE idea=? AND symbol=? AND entry=?",
                            (r.idea, r.symbol, r.entry)).fetchone()
            if old and old[0] == "CLOSED":
                continue
            c.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (r.idea, r.symbol, r.entry, r.exit, int(r.dir), r.entry_px, r.exit_px, r.stop_px, r.risk_px, r.reason,
                       r.status, r.R_gross, r.cost_R, r.R, old[2] if old else now.isoformat(timespec="seconds")))
        rows = pd.read_sql("SELECT * FROM trades", c)
    status = dict(generated_utc=now.isoformat(timespec="seconds"), paper_start_utc=str(PAPER_START), usd_per_R=USD_PER_R, ideas={})
    for (idea, sym), g in rows.groupby(["idea", "symbol"]):
        done = g[g.status == "CLOSED"]
        status["ideas"][f"{idea}|{sym}"] = dict(
            closed=len(done), open=int((g.status == "OPEN").sum()), wins=int((done.R > 0).sum()),
            total_R=round(float(done.R.sum()), 2), avg_R=round(float(done.R.mean()), 3) if len(done) else None,
            usd=round(float(done.R.sum()) * USD_PER_R, 2), worst_R=round(float(done.R.min()), 2) if len(done) else None,
            last_entry=g.entry.max() if len(g) else None)
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2))
    return status


def report() -> str:
    if not STATUS.exists():
        return "no paper pass has run yet"
    s = json.loads(STATUS.read_text())
    lines = [f"paper ideas since {s['paper_start_utc']} (updated {s['generated_utc']}); $ at ${s['usd_per_R']} per R"]
    for k, v in s["ideas"].items():
        lines.append(f"  {k:28s} closed {v['closed']:3d}  open {v['open']}  wins {v['wins']:3d}  total {v['total_R']:+7.2f}R  "
                     f"${v['usd']:+8.2f}  avg {v['avg_R']}  worst {v['worst_R']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    if ap.parse_args().report:
        print(report())
        return 0
    from dotenv import load_dotenv
    load_dotenv()
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    try:
        if not gw.connect():
            log.error("MT5 connection failed")
            return 1
        s = run_once(gw)
        log.info("pass done: %s", {k: (v["closed"], v["total_R"]) for k, v in s["ideas"].items()})
    except Exception as e:
        log.exception("pass failed: %s", e)
        return 1
    finally:
        gw.shutdown()
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
