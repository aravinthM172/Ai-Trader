"""
GOLD: ICT-style Asian-session liquidity sweep with EMA side (the owner's idea, 2026-10-10).  Backtest only.

The idea as described: session killzone boxes and deviations; EMA 20 / 50 / 100 / 200 -- above or below the EMAs
decides the side; the Asian-session liquidity sweep is the main setup; target 1:2; after a target, read the EMAs
again and be ready for the other side.

Fixed before the first run (New York time, the ICT clock):
  Asian box   high / low of 20:00-00:00
  Killzones   London 02:00-05:00, New York 07:00-10:00 -- entries only here
  Sweep buy   a bar trades below the Asian low and closes back above it; sweep sell mirrored at the Asian high.
              First one per killzone and side.
  Stop        beyond the sweep bar's extreme by 0.1 ATR.   Entry: next bar's open.   Out after 24 hours at the latest.
  Sides       ema200: buys only above the 200 EMA, sells only below it (EMAs on the trading chart)
              stack : buys only with 20 > 50 > 100 > 200 and price above the 20; sells mirrored
              noema : both sides, no EMA filter (to see whether the EMAs matter)
  Targets     rr2 : 2 x the risk (1:2)
              box : the other side of the Asian box
              dev1: the other side of the box plus one box height (1 deviation)
  Variants    ema200_rr2 (the idea as stated), stack_rr2, noema_rr2, ema200_box, ema200_dev1 on M15 (FundingPips
              2022-, the chart this method is traded on) and H1 (Dukascopy 2005-); ema200_rr2 also on M5 (2025-).
After a target the next signal is judged on the EMAs at that moment, so the side flips when price crosses them.

Judged with r7.judge (same rule as every round).  M15 / M5 have under 10 years: they cannot "pass", only inform.

    python -m backtest.gold_ict_sweep [--ledger]
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import gold_round8 as g8
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("ema200", "rr2"), ("stack", "rr2"), ("noema", "rr2"), ("ema200", "box"), ("ema200", "dev1")]
BARS_24H = {"M5": 288, "M15": 96, "H1": 24}
N_TESTS = 2 * len(VARIANTS) + 1


def sweep(p: pd.DataFrame, cost: dict, side: str, target: str, tf: str) -> pd.DataFrame:
    o, h, l, c = g8._arr(p)
    n, A = len(c), base_atr(h, l, c, 14)
    ny = p["time"].dt.tz_convert("America/New_York")
    hr = ny.dt.hour.to_numpy()
    day = (ny + pd.Timedelta(hours=4)).dt.tz_localize(None).dt.normalize().to_numpy()     # the box at 20:00 starts the day
    asia = hr >= 20
    gb = pd.DataFrame(dict(h=np.where(asia, h, np.nan), l=np.where(asia, l, np.nan))).groupby(day, sort=False)
    ah, al = gb.h.transform("max").to_numpy(), gb.l.transform("min").to_numpy()            # read in the killzones only
    zone = np.where((hr >= 2) & (hr < 5), 1, np.where((hr >= 7) & (hr < 10), 2, 0))
    ok = (zone > 0) & np.isfinite(ah) & np.isfinite(al) & (ah > al)
    key = pd.factorize(pd.Series(day).astype(str) + "|" + pd.Series(zone).astype(str))[0]
    buy = r5._first_of_group(ok & (l < al) & (c > al), key)
    sell = r5._first_of_group(ok & (h > ah) & (c < ah), key)
    e20, e50, e100, e200 = (r7.ema(c, k) for k in (20, 50, 100, 200))
    warm = np.arange(n) >= 200
    if side == "ema200":
        buy, sell = buy & warm & (c > e200), sell & warm & (c < e200)
    elif side == "stack":
        buy = buy & warm & (c > e20) & (e20 > e50) & (e50 > e100) & (e100 > e200)
        sell = sell & warm & (c < e20) & (e20 < e50) & (e50 < e100) & (e100 < e200)
    sl_b, sl_s = l - 0.1 * A, h + 0.1 * A
    box = ah - al
    if target == "rr2":
        tg_b, tg_s = c + 2 * (c - sl_b), c - 2 * (sl_s - c)
    elif target == "box":
        tg_b, tg_s = ah, al
    else:
        tg_b, tg_s = ah + box, al - box
    s = r5._put(r5._sig(n), buy, sell, sl_b, sl_s, tg_b, tg_s)
    return r5.simulate_piece(p, dict(s, hold=BARS_24H[tf]), cost)


def lookahead_ok(df: pd.DataFrame, tf: str, cost: dict) -> bool:
    p = max(fr.segments(df), key=len)
    na = min(len(p) - 3000, 30000)
    cut = p.time.iloc[na - 400]
    x, y = (sweep(p.iloc[:k].reset_index(drop=True), cost, "ema200", "rr2", tf) for k in (na, na + 3000))
    x, y = x[x.exit < cut].reset_index(drop=True), y[y.exit < cut].reset_index(drop=True)
    return len(x) == len(y) and bool(np.allclose(x.R.to_numpy(), y.R.to_numpy()))


def main() -> int:
    warnings.filterwarnings("ignore")
    cost = r6.cost("XAUUSD")
    data = {"M15": g8._fp("gold/fp_XAUUSD_M15.csv"), "H1": lh.h1("XAUUSD")[0], "M5": g8._fp("gold/fp_XAUUSD_M5.csv")}
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + N_TESTS
    res, look = {}, {}
    for tf, df in data.items():
        yrs = g8._years(df)
        look[tf] = lookahead_ok(df, tf, cost)
        for side, target in (VARIANTS if tf != "M5" else VARIANTS[:1]):
            t = g8._over(df, lambda p: sweep(p, cost, side, target, tf))
            r = r7.judge(t, yrs, n_trials)
            if "checks" in r:
                r["target_hit_pct"] = round(100 * float((t.reason == "target").mean()), 1)
                r["stop_hit_pct"] = round(100 * float((t.reason == "stop").mean()), 1)
            r.update(years=round(yrs, 1), trades_per_year=round(len(t) / yrs, 1))
            res[f"{tf}|{side}_{target}"] = r
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "--ledger" in sys.argv:
        r3.append_ledger([dict(tested_utc=tested, name="ict_" + k.replace("|", "_"), params="{}", lookahead_ok=look[k.split("|")[0]],
                               n_trials=n_trials, deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")),
                               useful=bool(r.get("passes"))) for k, r in res.items()])
    (ROOT / "reports" / "gold_ict_sweep.json").write_text(json.dumps(dict(meta=dict(n_trials=n_trials, generated=tested, lookahead_ok=look, cost=cost),
                                                                         results=res), indent=2, default=str))
    rows = ["| chart | variant | years | trades/yr | win % | target hit % | mean R | t | R/yr | buys R | sells R | first 70 % | last 30 % | last 2 y | slices | 2x cost | max DD (R) | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        tf, v = k.split("|")
        if "checks" not in r:
            rows.append(f"| {tf} | {v} | {r['years']} | {r['trades_per_year']} | too few trades | | | | | | | | | | | | | - |")
            continue
        rows.append(f"| {tf} | {v} | {r['years']} | {r['trades_per_year']} | {round(100 * r['win_rate'], 1)} | {r['target_hit_pct']} | {r['mean_R']} | {r['t']} | "
                    f"{r['R_per_year']} | {r['long_R']} | {r['short_R']} | {r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | "
                    f"{r['cost2x_R']} | {r['max_dd_R']} | {r['verdict'] if r['verdict'] != 'no' else '-'} |")
    md = (f"# Gold: Asian liquidity sweep with EMA side (ICT style)\n\nN for the deflated Sharpe: {n_trials}. Look-ahead check: {look}.\n\n"
          + "\n".join(rows) + "\n")
    (ROOT / "GOLD_ICT_SWEEP.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
