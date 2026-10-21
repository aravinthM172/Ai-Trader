"""
ACY "Day Trading Gold with Smart Money Concepts" model (article shared by the owner 2026-10-21), coded as written.

    python -m backtest.gold_smc_test

Sequence (bullish; bearish is the mirror):
  1 levels      previous day high / low, previous week high / low, Asian session (00:00-06:00 UTC) high / low
  2 sweep       inside a kill zone an M5 bar trades below one of the LOW levels
  3 displacement + MSB  within the next 6 bars a bullish fair value gap forms (low of bar k > high of bar k-2) whose
                middle candle body is >= 1 ATR, and price closes above the highest high of the 12 bars before the sweep
  4 entry       limit order in the FVG, valid 12 bars: variant "50%" = middle of the gap, variant "3rd" = top edge of the
                gap (low of the 3rd candle); stop = sweep low - 0.1 ATR
  5 exit        variant "liquidity" = the nearest opposite level above entry (PDH / Asian high / PWH); "2R"; "3R";
                time exit 48 M5 bars (4 h)
  kill zones    London 02:00-05:00 and New York 08:30-11:30, New York local time; 1-hour bias = last completed hour's
                close above its EMA50 for buys (below for sells)
Data: FundedNext gold M5 (2025-05-30 .. 2026-10-21, spread per bar) and the other broker's M5 as a second source.
Costs: median spread x 1.5 + $6.5 / lot commission; also 2x.  Judge: 60 / 40 split, unseen mean R > 0 with t >= 2,
> 0 at 2x costs and on the other broker.  Read-only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import gold_cheatsheet_test as cs
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
HOLD = 48


def setups(m: pd.DataFrame) -> list[dict]:
    """Every completed sweep -> displacement -> FVG sequence: signal bar index, direction, FVG edges, sweep extreme,
    liquidity target."""
    o, h, l, c = (m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    A = base_atr(h, l, c, 14)
    ny = m.time.dt.tz_convert("America/New_York")
    mins = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    kz = ((mins >= 120) & (mins < 300)) | ((mins >= 510) & (mins < 690))
    h1 = m.set_index("time").close.resample("1h").last().dropna()
    bias = np.sign(h1 - h1.ewm(span=50, adjust=False).mean())
    bias.index = bias.index + pd.Timedelta(hours=1)
    bias = bias.reindex(m.time, method="ffill").fillna(0).to_numpy()
    day = m.time.dt.normalize()
    pdh = m.groupby(day).high.max().shift(1).reindex(day).to_numpy()
    pdl = m.groupby(day).low.min().shift(1).reindex(day).to_numpy()
    wk = m.time.dt.to_period("W").astype(str)
    pwh = m.groupby(wk).high.max().shift(1).reindex(wk).to_numpy()
    pwl = m.groupby(wk).low.min().shift(1).reindex(wk).to_numpy()
    hr = m.time.dt.hour.to_numpy()
    asia = m[hr < 6].groupby(day[hr < 6])
    ah = np.where(hr >= 6, asia.high.max().reindex(day).to_numpy(), np.nan)
    al = np.where(hr >= 6, asia.low.min().reindex(day).to_numpy(), np.nan)
    hi12, lo12 = r7.prev(r7.hi(h, 12)), r7.prev(r7.lo(l, 12))
    out = []
    i = 14
    while i < n - 8:
        if not kz[i]:
            i += 1
            continue
        lows = [x for x in (pdl[i], pwl[i], al[i]) if np.isfinite(x)]
        highs = [x for x in (pdh[i], pwh[i], ah[i]) if np.isfinite(x)]
        d = 0
        if bias[i] > 0 and any(l[i] < x for x in lows):
            d = 1
        elif bias[i] < 0 and any(h[i] > x for x in highs):
            d = -1
        if d == 0:
            i += 1
            continue
        found = False
        for k in range(i + 2, min(n, i + 8)):
            body = abs(c[k - 1] - o[k - 1])
            if d > 0 and l[k] > h[k - 2] and body >= A[k] and c[k - 1] > o[k - 1] and c[k] > hi12[i]:
                ext = l[i:k + 1].min()
                tgt = min([x for x in highs if x > l[k]], default=np.nan)
                out.append(dict(sig=k, d=1, top=l[k], bot=h[k - 2], stop=ext - 0.1 * A[k], liq=tgt))
                found = True
                break
            if d < 0 and h[k] < l[k - 2] and body >= A[k] and c[k - 1] < o[k - 1] and c[k] < lo12[i]:
                ext = h[i:k + 1].max()
                tgt = max([x for x in lows if x < h[k]], default=np.nan)
                out.append(dict(sig=k, d=-1, top=l[k - 2], bot=h[k], stop=ext + 0.1 * A[k], liq=tgt))
                found = True
                break
        i = (out[-1]["sig"] + 1) if found else i + 1
    return out


def trades(m: pd.DataFrame, entry: str, exit_: str, cost_mult: float = 1.0) -> pd.DataFrame:
    n = len(m)
    s = r5._sig(n)
    limit = np.full(n, np.nan)
    for x in setups(m):
        k, d = x["sig"], x["d"]
        lim = (x["top"] + x["bot"]) / 2 if entry == "50%" else (x["top"] if d > 0 else x["bot"])
        risk = d * (lim - x["stop"])
        if risk <= 0:
            continue
        tgt = x["liq"] if exit_ == "liquidity" else lim + d * float(exit_[:-1]) * risk
        if not np.isfinite(tgt) or d * (tgt - lim) <= 0:
            continue
        s["d"][k], s["stop"][k], s["tgt"][k], limit[k] = d, x["stop"], tgt, lim
    cost = dict(cost_frac=(m.spr.median() * 1.5 + cs.COMM_PER_OZ) * cost_mult / m.close.median(), swap_long=0.0, swap_short=0.0)
    return r5.simulate_piece(m, dict(s, hold=HOLD, limit=limit, valid=12), cost)


def main() -> int:
    fn = cs.load_m5(ROOT / "data" / "gold" / "fn_XAUUSD_M5.csv")
    ob = cs.load_m5(ROOT / "data" / "xauusd_m5_5y.csv")
    cut, cut_b = gs.split_at(fn), gs.split_at(ob)
    months = (fn.time.iloc[-1] - fn.time.iloc[0]).days / 30.44
    print(f"FundedNext M5 {fn.time.iloc[0].date()} .. {fn.time.iloc[-1].date()}: {len(setups(fn))} complete sweep -> "
          f"displacement -> FVG sequences in kill zones")
    rows = []
    for entry in ("50%", "3rd"):
        for exit_ in ("liquidity", "2R", "3R"):
            t, t2, tb = trades(fn, entry, exit_), trades(fn, entry, exit_, 2.0), trades(ob, entry, exit_)
            d_, c_ = t[t.entry < cut], t[t.entry >= cut]
            rows.append({"entry": entry, "exit": exit_, "trades/month": round(len(t) / months, 1),
                         "win %": round(100 * float((t.R > 0).mean()), 1) if len(t) else None,
                         "avg R": round(float(t.R.mean()), 3) if len(t) else None,
                         "first 60 %": round(float(d_.R.mean()), 3) if len(d_) else None,
                         "unseen 40 %": round(float(c_.R.mean()), 3) if len(c_) else None,
                         "unseen t": round(float(c_.R.mean() / c_.R.std() * np.sqrt(len(c_))), 2) if len(c_) > 2 else None,
                         "2x cost unseen": round(float(t2[t2.entry >= cut].R.mean()), 3) if (t2.entry >= cut).any() else None,
                         "other broker unseen": round(float(tb[tb.entry >= cut_b].R.mean()), 3) if (tb.entry >= cut_b).any() else None,
                         "$/month @0.35% 15K": round(float(t.R.sum()) * 0.0035 * 15000 / months)})
    P = pd.DataFrame(rows)
    P["PASS"] = (P["unseen 40 %"] > 0) & (P["unseen t"] >= 2) & (P["2x cost unseen"] > 0) & (P["other broker unseen"] > 0)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(P.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
