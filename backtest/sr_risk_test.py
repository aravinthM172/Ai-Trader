"""
Owner 2026-10-22: "strong risk management based on support and resistance".  S/R as entries (pullback limits) and as targets
already lost (pullback_entry_test, structure-tp study).  Two S/R uses that are about RISK, tested on the live rule:

    python -m backtest.sr_risk_test

  base        live rule, stop 2 ATR, target 4.5 ATR (today)
  struct      stop = lowest low of the last 10 H1 bars - 0.1 ATR (support), kept between 1 and 3 ATR; target = 2.25 x that risk
  room        base, but skip a BUY when the highest high of the last 50 bars (resistance) is less than 1.5 ATR above the
              entry price (mirror for sells: support within 1.5 ATR below)
  both        struct + room
Live rules via tp_weekday_study.signals (strong-signal gold / NDX, weekend-flat crypto), daily EMA200 filter gold / NDX / USDJPY,
NDX / USDJPY buy-only; verified zip H1 data; FundedNext costs; one position per market; from 2018.  Read-only.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from backtest import all_pairs_verified as ap
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest import tp_weekday_study as tw
from backtest.btc_strategies import base_atr
from tools import zip_history as zh

warnings.filterwarnings("ignore")
FROM = pd.Timestamp("2018-01-01", tz="UTC")


def run(sym: str, variant: str) -> pd.DataFrame:
    df = zh.resample(zh.load_m1(sym), "1h")
    reg = lh.daily_regime(df, True) if sym in lh.LIVE_FILTER else None
    out = []
    for p in fr.segments(df):
        if len(p) < 400:
            continue
        p = p.reset_index(drop=True)
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        ent, _ = tw.signals(p, sym, True)
        d = np.zeros(len(c), np.int64)
        d[:-1] = ent[1:]
        if reg is not None:
            day = (p.time + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
            side = np.r_[reg.side.reindex(day, method="ffill").to_numpy()[1:], np.nan]
            d[(d != 0) & (d != side)] = 0
        if sym in ("NDX100", "USDJPY"):
            d[d < 0] = 0
        if variant in ("room", "both"):
            res, sup = r7.hi(h, 50), r7.lo(l, 50)
            d[(d > 0) & (res - c < 1.5 * A)] = 0
            d[(d < 0) & (c - sup < 1.5 * A)] = 0
        if variant in ("struct", "both"):
            lo10, hi10 = r7.lo(l, 10) - 0.1 * A, r7.hi(h, 10) + 0.1 * A
            risk = np.where(d > 0, c - lo10, hi10 - c)
            risk = np.clip(np.nan_to_num(risk, nan=2.0 * A), 1.0 * A, 3.0 * A)
        else:
            risk = 2.0 * A
        s = r5._sig(len(c))
        ok = d != 0
        s["d"][ok] = d[ok]
        s["stop"] = np.where(ok, c - d * risk, np.nan)
        s["tgt"] = np.where(ok, c + d * 2.25 * risk, np.nan)
        t = r5.simulate_piece(p, dict(s, hold=96), dict(cost_frac=ap.COST[sym], swap_long=0.0, swap_short=0.0))
        if len(t):
            out.append(t)
    t = pd.concat(out, ignore_index=True)
    return t[t.entry >= FROM].assign(sym=sym)


def main() -> int:
    rows = []
    for v in ("base", "struct", "room", "both"):
        allt = pd.concat([run(s, v) for s in tw.SYMS], ignore_index=True).sort_values("exit")
        yrs = (allt.exit.max() - FROM).days / 365.25
        half = FROM + (allt.exit.max() - FROM) / 2
        usd = allt.R * 0.0035 * 15000
        eq = usd.cumsum()
        streak = (allt.R < 0).astype(int).groupby((allt.R >= 0).cumsum()).sum().max()
        r = {"variant": v, "trades/yr": round(len(allt) / yrs), "win %": round(100 * float((allt.R > 0).mean()), 1),
             "avg R": round(float(allt.R.mean()), 3), "1st half": round(float(allt[allt.entry < half].R.mean()), 3),
             "2nd half": round(float(allt[allt.entry >= half].R.mean()), 3), "$/yr": round(float(usd.sum()) / yrs),
             "max drop $": round(float((eq.cummax() - eq).max())), "longest losing streak": int(streak)}
        for s in tw.SYMS:
            x = allt[allt.sym == s]
            r[s] = round(float(x.R.mean()), 3)
        rows.append(r)
        print(f"{v} done", flush=True)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
