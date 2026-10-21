"""
Owner question 2026-10-21: the live bot buys at the next open right after a strong signal candle ("hanging in the air").
Would waiting for a pullback toward support (limit order) do better?

    python -m backtest.pullback_entry_test

Live rules (tp_weekday_study.signals: momentum H1 + strong-signal gold / NDX + weekend-flat crypto entries; daily EMA200 filter
for gold / NDX / USDJPY; NDX / USDJPY buy-only), stop 2 ATR and target 4.5 ATR measured from the ACTUAL entry price, time
exit 96 bars.  Entry variants:
  market     next open (today's bot)
  lim_0.5    limit 0.5 ATR better than the signal close, valid 4 bars, else skipped
  lim_1.0    limit 1.0 ATR better, valid 4 bars
  lim_ema20  limit at the H1 EMA20 (nearest dynamic support / resistance), valid 4 bars (skipped if already beyond it)
Data 2018-03 .. 2026 (common period of the 5 markets), costs = FundingPips spread + slippage, $ at 0.35 % of $15K.
Read-only.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest import tp_weekday_study as tw
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
FROM = pd.Timestamp("2018-03-01", tz="UTC")
SL, TP = 2.0, 4.5


def run(df: pd.DataFrame, sym: str, variant: str) -> pd.DataFrame:
    cost = fb.cost_of(lh.SPECS[sym])
    cost = dict(cost, swap_long=cost["swap_long"], swap_short=cost["swap_short"])
    reg = lh.daily_regime(df, True) if sym in lh.LIVE_FILTER else None
    out = []
    for p in fr.segments(df):
        if len(p) < 400:
            continue
        p = p.reset_index(drop=True)
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        ent, _ = tw.signals(p, sym, True)                       # ent[i] = enter at bar i -> signal bar i - 1
        d = np.zeros(len(c), np.int64)
        d[:-1] = ent[1:]
        if reg is not None:
            day = (p.time + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
            side = reg.side.reindex(day, method="ffill").to_numpy()
            nxt = np.r_[side[1:], np.nan]
            d[(d != 0) & (d != nxt)] = 0
        if sym in ("NDX100", "USDJPY"):
            d[d < 0] = 0
        e20 = r7.ema(c, 20)
        if variant == "market":
            ref = c.copy(); limit = np.full(len(c), np.nan)
        elif variant == "lim_ema20":
            ref = e20.copy(); limit = np.where(d * (c - e20) > 0, e20, np.nan)
        else:
            k = float(variant.split("_")[1])
            ref = c - d * k * A; limit = ref.copy()
        s = r5._sig(len(c))
        ok = (d != 0) & (np.isfinite(limit) | (variant == "market"))
        s["d"][ok] = d[ok]
        s["stop"] = np.where(ok, ref - d * SL * A, np.nan)
        s["tgt"] = np.where(ok, ref + d * TP * A, np.nan)
        t = r5.simulate_piece(p, dict(s, hold=96, limit=np.where(ok, limit, np.nan), valid=4), cost)
        if len(t):
            out.append(t)
    t = pd.concat(out, ignore_index=True)
    return t[t.entry >= FROM].assign(sym=sym)


def main() -> int:
    data = {s: lh.h1(s)[0] for s in tw.SYMS}
    rows = []
    for v in ("market", "lim_0.5", "lim_1.0", "lim_ema20"):
        allt = pd.concat([run(data[s], s, v) for s in tw.SYMS], ignore_index=True)
        yrs = (allt.exit.max() - FROM).days / 365.25
        half = FROM + (allt.exit.max() - FROM) / 2
        r = {"entry": v, "trades/yr": round(len(allt) / yrs), "win %": round(100 * float((allt.R > 0).mean()), 1),
             "avg R": round(float(allt.R.mean()), 3), "1st half R": round(float(allt[allt.entry < half].R.mean()), 3),
             "2nd half R": round(float(allt[allt.entry >= half].R.mean()), 3),
             "$/yr @0.35% 15K": round(float(allt.R.sum()) / yrs * 0.0035 * 15000)}
        for s in tw.SYMS:
            x = allt[allt.sym == s]
            r[s] = round(float(x.R.mean()), 3) if len(x) else None
        rows.append(r)
        print(f"{v} done", flush=True)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
