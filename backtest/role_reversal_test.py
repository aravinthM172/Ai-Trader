"""
Owner idea 2026-10-22: "once price moves above resistance and makes it support, then take the trade" (role reversal /
breakout-retest) -- all five markets, H1 / H4 / D1, verified zip data, FundedNext costs.

    python -m backtest.role_reversal_test

PRE-REGISTERED (mirror for sells)
  resistance  highest high of the previous 50 bars
  breakout    a close above it arms the setup for N bars (H1 24, H4 12, D1 10); the broken level is remembered
  retest      a later bar's low comes within 0.25 ATR of the level (or below it) and the bar CLOSES above the level -> the old
              resistance held as support -> BUY at the next open
  stop        the lowest low since the breakout - 0.1 ATR, kept between 0.5 and 3 ATR from the entry
  target      2R / 3R / 4R; time exit 96 (H1), 48 (H4), 20 (D1) bars
  variants    plain, and "trend" = only in the direction of the daily close vs its EMA200 (last completed day)
Judge: 60 / 40 split; ROBUST = first 60 % > 0, unseen > 0 with t >= 2, unseen > 0 at 2x costs.  Read-only.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from backtest import all_pairs_verified as ap
from backtest import full_reassessment as fr
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ARM = {"H1": 24, "H4": 12, "D1": 10}


def daily_side(p: pd.DataFrame) -> np.ndarray:
    dc = p.set_index("time").close.resample("1D").last().dropna()
    side = np.sign(dc - dc.ewm(span=200, adjust=False).mean()).shift(1)
    return side.reindex(p.time.dt.floor("1D")).to_numpy()


def signals(p: pd.DataFrame, tf: str, trend: bool):
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    A = base_atr(h, l, c, 14)
    hi50, lo50 = r7.prev(r7.hi(h, 50)), r7.prev(r7.lo(l, 50))
    side = daily_side(p) if trend else np.ones(n)
    d, stop = np.zeros(n, np.int64), np.full(n, np.nan)
    armed = None                                         # (dir, level, start, extreme)
    for i in range(55, n):
        if armed and i - armed[2] > ARM[tf]:
            armed = None
        if armed:
            dr, lev, st, ext = armed
            ext = min(ext, l[i]) if dr > 0 else max(ext, h[i])
            armed = (dr, lev, st, ext)
            if dr > 0 and l[i] <= lev + 0.25 * A[i] and c[i] > lev:
                d[i], stop[i] = 1, ext - 0.1 * A[i]; armed = None
            elif dr < 0 and h[i] >= lev - 0.25 * A[i] and c[i] < lev:
                d[i], stop[i] = -1, ext + 0.1 * A[i]; armed = None
            continue
        if c[i] > hi50[i] and (not trend or side[i] > 0):
            armed = (1, hi50[i], i, l[i])
        elif c[i] < lo50[i] and (not trend or side[i] < 0):
            armed = (-1, lo50[i], i, h[i])
    risk = np.where(d != 0, d * (c - stop), np.nan)
    risk = np.clip(risk, 0.5 * A, 3.0 * A)
    return d, c - d * risk, risk


def trades(df, tf, trend, tp, cf):
    out = []
    for p in ([df] if tf == "D1" else [p for p in fr.segments(df) if len(p) > 300]):
        p = p.reset_index(drop=True)
        c = p.close.to_numpy(float)
        d, stop, risk = signals(p, tf, trend)
        s = r5._sig(len(c))
        ok = d != 0
        s["d"][ok], s["stop"][ok], s["tgt"][ok] = d[ok], stop[ok], (c + d * tp * risk)[ok]
        t = r5.simulate_piece(p, dict(s, hold=ap.HOLD[tf]), dict(cost_frac=cf, swap_long=0.0, swap_short=0.0))
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    rows = []
    for sym in ap.SYMS:
        for tf in ("H1", "H4", "D1"):
            df = ap.data(sym, tf)
            cut, yrs = gs.split_at(df), (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
            for trend in (False, True):
                for tp in (2.0, 3.0, 4.0):
                    t = trades(df, tf, trend, tp, ap.COST[sym])
                    s = ap.summarise(t, cut, yrs)
                    if s:
                        rows.append({"pair": sym, "tf": tf, "filter": "daily trend" if trend else "none", "target": f"{tp:g}R", **s})
        print(f"{sym} done", flush=True)
    P = pd.DataFrame(rows)
    P["ROBUST"] = (P["first60 R"] > 0) & (P["unseen R"] > 0) & (P["unseen t"] >= 2) & (P["2x cost unseen R"] > 0)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 200):
        print(P.to_string(index=False))
        print("\nshare of variants profitable on unseen data:", round(100 * float((P["unseen R"] > 0).mean()), 1), "%; robust:", int(P.ROBUST.sum()), "of", len(P))
    P.to_csv(ap.ROOT / "reports" / "role_reversal_test.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
