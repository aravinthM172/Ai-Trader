"""
GOLD: the way gold "calls" are made (signal channels, TradingView ideas), as fixed rules.  Backtest only.

What the published calls have in common (sources in GOLD_ZONE_CALLS.md):
  1. side from the 4-hour chart; 2. a supply / demand zone on the 4-hour chart; 3. wait for price to come back
  into the zone; 4. enter on a small-chart confirmation (break of the last swing); 5. stop beyond the zone;
  6. TP1 / TP2 / TP3, part closed at each, stop to entry after TP1.

Fixed before the first run:
  Side         buys only when the last closed H4 candle is above its 50 EMA, sells only below ("bias");
               a control without it ("nobias").
  Demand zone  the H4 candle before an up-impulse (H4 up candle, range >= 1.5 ATR, close above the previous high);
               zone = that candle's low..high, at most 2 ATR tall.  Supply mirrored.  Only the newest zone of each
               kind is kept; a zone is used once and dies when a small-chart candle closes through it.
  Entry        after price has touched the zone: first small-chart close above the highest high of the previous
               3 candles, within 16 candles of the touch; buy at the next open.  Sells mirrored.
  Stop         0.1 ATR beyond the lower of the zone edge and the lowest low since the touch.
  Exits        "tp2":  one target at 2 x risk.
               "tp123": a third at 1R, 2R and 3R; stop to entry after TP1.     Out after 5 days at the latest.
  Charts       M15 (FundingPips 2022-) and H1 (Dukascopy 2005-).  H4 candles are built from the same data.
  Tests        bias_tp2, bias_tp123, nobias_tp2 on each chart = 6.
In a candle that could hit both, the stop counts first.  Costs: FundingPips spread + slippage + swap, as in every round.

    python -m backtest.gold_zone_calls [--ledger]
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
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
HOLD = {"M15": 480, "H1": 120}
N_TESTS = 6


def h4_context(p: pd.DataFrame) -> pd.DataFrame:
    """Per small-chart bar: bias and the newest demand / supply zone, all from H4 candles already closed."""
    g = p.set_index("time").resample("4h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    pa, ph, pl = np.roll(A, 1), np.roll(h, 1), np.roll(l, 1)
    tall = (ph - pl) <= 2 * pa
    up = (c > o) & ((h - l) >= 1.5 * pa) & (c > ph) & tall
    dn = (c < o) & ((h - l) >= 1.5 * pa) & (c < pl) & tall
    up[:15], dn[:15] = False, False
    z = pd.DataFrame(index=g.index + pd.Timedelta(hours=4))                      # known when the H4 candle has closed
    z["bias"] = np.sign(c - r7.ema(c, 50))
    z.loc[z.index[:50], "bias"] = 0
    z["d_lo"], z["d_hi"], z["d_id"] = np.where(up, pl, np.nan), np.where(up, ph, np.nan), np.where(up, np.arange(len(c)), np.nan)
    z["s_lo"], z["s_hi"], z["s_id"] = np.where(dn, pl, np.nan), np.where(dn, ph, np.nan), np.where(dn, np.arange(len(c)), np.nan)
    z = z.ffill()
    out = pd.merge_asof(p[["time"]], z.reset_index().rename(columns={"index": "time"}), on="time", direction="backward")
    return out


def calls(p: pd.DataFrame, cost: dict, tf: str, bias: bool, exits: str) -> pd.DataFrame:
    o, h, l, c = g8._arr(p)
    n, A = len(c), base_atr(h, l, c, 14)
    z = h4_context(p)
    bz = z["bias"].to_numpy(float)
    zones = {1: (z["d_lo"].to_numpy(float), z["d_hi"].to_numpy(float), z["d_id"].to_numpy(float)),
             -1: (z["s_lo"].to_numpy(float), z["s_hi"].to_numpy(float), z["s_id"].to_numpy(float))}
    hi3 = pd.Series(h).rolling(3).max().shift(1).to_numpy()
    lo3 = pd.Series(l).rolling(3).min().shift(1).to_numpy()
    parts = [(2.0, 1.0)] if exits == "tp2" else [(1.0, 1 / 3), (2.0, 1 / 3), (3.0, 1 / 3)]
    used = {1: -1.0, -1: -1.0}                    # zone id already traded or broken
    touch = {1: -1, -1: -1}                       # bar of the first touch of the current zone
    cur = {1: -1.0, -1: -1.0}
    ext = {1: np.nan, -1: np.nan}                 # extreme since the touch
    rows, j, t = [], 20, p["time"]
    while j < n - 2:
        entered = False
        for d in (1, -1):
            lo, hi, zid = zones[d][0][j], zones[d][1][j], zones[d][2][j]
            if not np.isfinite(zid) or zid == used[d]:
                continue
            if zid != cur[d]:
                cur[d], touch[d], ext[d] = zid, -1, np.nan
            if (d > 0 and c[j] < lo) or (d < 0 and c[j] > hi):          # closed through the zone: dead
                used[d] = zid
                continue
            in_zone = (l[j] <= hi) if d > 0 else (h[j] >= lo)
            if touch[d] < 0:
                if not in_zone:
                    continue
                touch[d], ext[d] = j, (l[j] if d > 0 else h[j])
            ext[d] = min(ext[d], l[j]) if d > 0 else max(ext[d], h[j])
            if j - touch[d] > 16:
                used[d] = zid
                continue
            confirm = (c[j] > hi3[j]) if d > 0 else (c[j] < lo3[j])
            if not confirm or (bias and bz[j] != d) or not (A[j] > 0):
                continue
            i = j + 1
            ep = o[i]
            sp = (min(lo, ext[d]) - 0.1 * A[j]) if d > 0 else (max(hi, ext[d]) + 0.1 * A[j])
            risk = d * (ep - sp)
            used[d] = zid
            if not (0.5 * A[j] <= risk <= 6 * A[j]):
                continue
            # manage
            stop, left, got, k, reason = sp, 1.0, 0.0, i, "time"
            nxt = 0
            last = min(n - 1, i + HOLD[tf])
            while k < last and left > 1e-9:
                if k > i and d * (o[k] - stop) <= 0:
                    got += left * d * (o[k] - ep) / risk; left = 0.0; reason = "stop" if nxt == 0 else "breakeven"; break
                if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
                    got += left * d * (stop - ep) / risk; left = 0.0; reason = "stop" if nxt == 0 else "breakeven"; break
                while nxt < len(parts) and ((d > 0 and h[k] >= ep + parts[nxt][0] * risk) or (d < 0 and l[k] <= ep - parts[nxt][0] * risk)):
                    got += parts[nxt][1] * parts[nxt][0]; left -= parts[nxt][1]; nxt += 1
                    if exits == "tp123":
                        stop = ep
                    reason = "target"
                if left <= 1e-9:
                    break
                k += 1
            if left > 1e-9:
                k = last
                got += left * d * (o[k] - ep) / risk
            nights = ((t.iloc[k] + pd.Timedelta(hours=3)).normalize() - (t.iloc[i] + pd.Timedelta(hours=3)).normalize()).days
            cost_R = cost["cost_frac"] * ep / risk
            swap = ep * (cost["swap_long"] if d > 0 else cost["swap_short"]) / 365 * nights / risk
            rows.append((t.iloc[i], t.iloc[k], d, reason, got + swap - cost_R, cost_R))
            j = k
            entered = True
            break
        if not entered:
            j += 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "R", "cost_R"])


def main() -> int:
    warnings.filterwarnings("ignore")
    cost = r6.cost("XAUUSD")
    data = {"M15": g8._fp("gold/fp_XAUUSD_M15.csv"), "H1": lh.h1("XAUUSD")[0]}
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + N_TESTS
    res = {}
    for tf, df in data.items():
        yrs = g8._years(df)
        for name, bias, exits in (("bias_tp2", True, "tp2"), ("bias_tp123", True, "tp123"), ("nobias_tp2", False, "tp2")):
            t = r7._cat([calls(p, cost, tf, bias, exits) for p in fr.segments(df)])
            r = r7.judge(t, yrs, n_trials)
            if "checks" in r:
                r["reasons"] = {k: round(100 * v, 1) for k, v in t.reason.value_counts(normalize=True).items()}
            r.update(years=round(yrs, 1), trades_per_year=round(len(t) / yrs, 1))
            res[f"{tf}|{name}"] = r
    # look-ahead: trades closed early must not change when later bars are added
    p = max(fr.segments(data["H1"]), key=len)
    na = min(len(p) - 3000, 30000)
    x, y = (calls(p.iloc[:k].reset_index(drop=True), cost, "H1", True, "tp123") for k in (na, na + 3000))
    cut = p.time.iloc[na - 400]
    x, y = x[x.exit < cut], y[y.exit < cut]
    look = len(x) == len(y) and bool(np.allclose(x.R.to_numpy(), y.R.to_numpy()))
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "--ledger" in sys.argv:
        r3.append_ledger([dict(tested_utc=tested, name="zone_" + k.replace("|", "_"), params="{}", lookahead_ok=look, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                          for k, r in res.items()])
    (ROOT / "reports" / "gold_zone_calls.json").write_text(json.dumps(dict(meta=dict(n_trials=n_trials, generated=tested, lookahead_ok=look),
                                                                          results=res), indent=2, default=str))
    rows = ["| chart | variant | years | trades/yr | win % | exits (%) | mean R | t | R/yr | buys R | sells R | first 70 % | last 30 % | last 2 y | slices | 2x cost | max DD (R) | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        tf, v = k.split("|")
        if "checks" not in r:
            rows.append(f"| {tf} | {v} | {r['years']} | {r['trades_per_year']} | too few trades | | | | | | | | | | | | | - |")
            continue
        rows.append(f"| {tf} | {v} | {r['years']} | {r['trades_per_year']} | {round(100 * r['win_rate'], 1)} | {r['reasons']} | {r['mean_R']} | {r['t']} | "
                    f"{r['R_per_year']} | {r['long_R']} | {r['short_R']} | {r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | "
                    f"{r['cost2x_R']} | {r['max_dd_R']} | {r['verdict'] if r['verdict'] != 'no' else '-'} |")
    md = f"# Gold: zone calls (H4 zone, small-chart confirmation, TP1 / TP2 / TP3)\n\nN for the deflated Sharpe: {n_trials}. Look-ahead check ok: {look}.\n\n" + "\n".join(rows) + "\n"
    (ROOT / "GOLD_ZONE_CALLS.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
