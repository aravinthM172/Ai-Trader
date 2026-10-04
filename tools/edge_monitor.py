"""
Automatic retirement: stop the live strategy when live results show the edge is gone,
without waiting for a human or for the 35 % drawdown breaker.

Rules (fixed in advance; RULES.md explains them).  Any one trips state/KILL_SWITCH:
  1. SAMPLE TEST  -- after >= MIN_TRADES live trades, the live mean R is below the
     2.5th percentile of the mean of the same number of trades drawn from the backtest,
     after first cutting the backtest edge by HAIRCUT_R (the decay already seen).
  2. STREAK       -- a losing streak longer than the backtest's worst + STREAK_MARGIN.
  3. DECAY        -- after >= 100 live trades, the last 100 have mean R <= 0.

Reference trades: reports/btc_paper_replay_trades.csv (r_multiple).
Live trades:      state/btc_live_H1.sqlite (closed rows, r_multiple).

    python -m tools.edge_monitor             # report + enforce (writes KILL_SWITCH on a trip)
    python -m tools.edge_monitor --dry       # report only

Writes reports/edge_monitor.json.  Never sends or closes orders; a trip only blocks NEW
entries (open positions keep their broker SL/TP), exactly like the manual kill switch.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE = ROOT / "reports" / "btc_paper_replay_trades.csv"
LIVE_DB = ROOT / "state" / "btc_live_H1.sqlite"
KILL_FILE = ROOT / "state" / "KILL_SWITCH"
OUT = ROOT / "reports" / "edge_monitor.json"

MIN_TRADES = 20
HAIRCUT_R = 0.10
PCTILE = 2.5
STREAK_MARGIN = 3
DECAY_WINDOW = 100
N_BOOT = 20_000


def worst_streak(R) -> int:
    s = m = 0
    for r in R:
        s = s + 1 if r < 0 else 0
        m = max(m, s)
    return m


def evaluate(live_R, ref_R, *, seed: int = 7) -> dict:
    live_R = np.asarray(live_R, float)
    ref_R = np.asarray(ref_R, float)
    n = len(live_R)
    out = {"live_trades": n, "live_mean_R": round(float(live_R.mean()), 4) if n else None,
           "reference_trades": int(len(ref_R)), "reference_mean_R": round(float(ref_R.mean()), 4),
           "haircut_R": HAIRCUT_R, "trips": []}

    ref_worst = worst_streak(ref_R)
    live_worst = worst_streak(live_R)
    out.update(reference_worst_streak=ref_worst, live_worst_streak=live_worst,
               streak_limit=ref_worst + STREAK_MARGIN)
    if live_worst > ref_worst + STREAK_MARGIN:
        out["trips"].append(f"losing streak {live_worst} > backtest worst {ref_worst} + {STREAK_MARGIN}")

    if n >= MIN_TRADES:
        rng = np.random.default_rng(seed)
        boot = rng.choice(ref_R - HAIRCUT_R, size=(N_BOOT, n), replace=True).mean(axis=1)
        floor = float(np.percentile(boot, PCTILE))
        out["sample_floor_R"] = round(floor, 4)
        if live_R.mean() < floor:
            out["trips"].append(f"live mean {live_R.mean():+.3f} R over {n} trades < "
                                f"{PCTILE}th pct {floor:+.3f} R of the backtest (edge cut by {HAIRCUT_R} R)")
    else:
        out["sample_floor_R"] = None
        out["note"] = f"sample test starts after {MIN_TRADES} live trades"

    if n >= DECAY_WINDOW:
        last = float(live_R[-DECAY_WINDOW:].mean())
        out["last_100_mean_R"] = round(last, 4)
        if last <= 0:
            out["trips"].append(f"last {DECAY_WINDOW} trades mean {last:+.3f} R <= 0")

    out["retire"] = bool(out["trips"])
    return out


def load_live_R(db: Path = LIVE_DB) -> np.ndarray:
    if not db.exists():
        return np.array([])
    c = sqlite3.connect(db)
    try:
        rows = c.execute("SELECT r_multiple FROM trades WHERE status='CLOSED' AND r_multiple IS NOT NULL "
                         "ORDER BY closed_utc").fetchall()
    finally:
        c.close()
    return np.array([r[0] for r in rows], float)


def load_reference_R(path: Path = REFERENCE) -> np.ndarray:
    return pd.read_csv(path)["r_multiple"].dropna().to_numpy(float)


def trip(reason: str, kill_file: Path = KILL_FILE) -> None:
    kill_file.parent.mkdir(exist_ok=True)
    kill_file.write_text(f"{datetime.now(timezone.utc).isoformat()}  edge_monitor: {reason}\n", encoding="utf-8")


def run(*, enforce: bool = True, db: Path = LIVE_DB, ref: Path = REFERENCE,
        kill_file: Path = KILL_FILE, out: Path = OUT) -> dict:
    res = evaluate(load_live_R(db), load_reference_R(ref))
    res["checked_utc"] = datetime.now(timezone.utc).isoformat()
    res["enforced"] = False
    if res["retire"] and enforce and not kill_file.exists():
        trip("; ".join(res["trips"]), kill_file)
        res["enforced"] = True
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="report only, never write KILL_SWITCH")
    a = ap.parse_args()
    r = run(enforce=not a.dry)
    print(json.dumps({k: r[k] for k in ("live_trades", "live_mean_R", "sample_floor_R", "live_worst_streak",
                                        "streak_limit", "retire", "trips", "enforced")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
