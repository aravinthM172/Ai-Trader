"""
Score the AI shadow opinions (execution/ai_shadow.py): does the AI's SKIP pick out losing signals?

Every logged signal is replayed on the broker's H1 bars with the live stop plan (entry at the next
bar's open, 2 ATR stop / 3 ATR target, 96-bar time exit) -- whether or not the live bot traded it,
so blocked signals count too.  Then mean R of AI-TAKE vs AI-SKIP.

    python -m tools.ai_shadow_report

Promotion rule (same as the paper strategies): at least MIN_SIGNALS scored signals AND the SKIP
group's mean R below the TAKE group's by at least MIN_GAP_R.  Even then a veto is a separate,
owner-approved change to the locked live trader.  Read-only.  Writes reports/ai_shadow_report.json.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from backtest.btc_strategies import base_atr
from execution.ai_shadow import DB
from execution.live_multi import completed_bars
from tools.twin_check import simulate_exit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ai_shadow_report.json"
MIN_SIGNALS = 30
MIN_GAP_R = 0.15


def summarise(rows: list[dict]) -> dict:
    """rows: dicts with verdict and r (None while the trade is still open)."""
    done = [r for r in rows if r.get("r") is not None and r["verdict"] in {"TAKE", "SKIP"}]
    grp = {}
    for v in ("TAKE", "SKIP"):
        rs = [r["r"] for r in done if r["verdict"] == v]
        grp[v] = {"n": len(rs), "mean_r": round(sum(rs) / len(rs), 3) if rs else None,
                  "win_rate": round(sum(x > 0 for x in rs) / len(rs), 3) if rs else None}
    t, s = grp["TAKE"]["mean_r"], grp["SKIP"]["mean_r"]
    gap = round(t - s, 3) if t is not None and s is not None else None
    passed = len(done) >= MIN_SIGNALS and gap is not None and gap >= MIN_GAP_R
    return {"scored": len(done), "groups": grp, "take_minus_skip_r": gap,
            "all_signals_mean_r": round(sum(r["r"] for r in done) / len(done), 3) if done else None,
            "pass": passed,
            "status": ("PASS -- AI skip filters out losers; consider a veto (owner approval)" if passed else
                       f"collecting ({len(done)}/{MIN_SIGNALS} scored)" if len(done) < MIN_SIGNALS else
                       "FAIL -- AI opinion does not separate winners from losers")}


def main() -> int:
    if not DB.exists():
        print("no AI shadow opinions yet")
        return 0
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    ops = [dict(r) for r in c.execute("SELECT * FROM opinions ORDER BY signal_bar_utc")]
    c.close()
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        print("MT5 connection failed")
        return 1
    rows, now = [], pd.Timestamp.now(tz="UTC")
    try:
        for o in ops:
            df = gw.get_rates(o["symbol"], "H1", 1500)
            r = dict(o, r=None, exit=None)
            if df is not None and not df.empty:
                df = completed_bars(df, now.to_pydatetime()).reset_index(drop=True)
                times = pd.to_datetime(df["time"], utc=True)
                k = times.searchsorted(pd.Timestamp(o["signal_bar_utc"]))
                if k < len(df) - 1 and times.iloc[k] == pd.Timestamp(o["signal_bar_utc"]):
                    w = df.iloc[:k + 1]
                    atr = float(base_atr(w["high"].to_numpy(float), w["low"].to_numpy(float),
                                         w["close"].to_numpy(float), 14)[-1])
                    reason, rr, _ = simulate_exit(o["direction"], float(df["open"].iloc[k + 1]), atr,
                                                  df.iloc[k + 1:].reset_index(drop=True))
                    r["exit"] = reason
                    r["r"] = None if reason == "open" else round(rr, 3)
            rows.append(r)
    finally:
        gw.shutdown()
    summary = {**summarise(rows), "logged": len(ops),
               "errors": sum(o["verdict"] == "ERROR" for o in ops), "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    for r in rows:
        conf = "" if r["confidence"] is None else f"{r['confidence']:.2f}"
        res = "" if r["r"] is None else r["r"]
        print(f"{r['signal_bar_utc'][:16]} {r['symbol']:8} {r['direction']:4} AI {r['verdict']:5} {conf:4}  "
              f"{r['exit'] or '-':6} {res}")
    print({k: v for k, v in summary.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
