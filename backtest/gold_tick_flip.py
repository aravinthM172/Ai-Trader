"""
Owner idea 2026-10-21, tick-accurate: flip a bulk 15 x 0.01 gold lot every time the FORMING candle changes colour.
Long while bid > the candle's open (first bid of the candle, as MT5 builds candles), short while bid < it; flip on
every cross; the position carries into the next candle until that one turns the other colour.

    python -m backtest.gold_tick_flip data/dukascopy_ticks/XAUUSD_20260801_20260831.csv

Fills on real Dukascopy bid / ask: a buy pays the ask, a sell gets the bid (spread included), + $6.5 / lot commission
round turn.  "lag 0" fills on the tick that crossed (impossible in practice), "lag 1" on the next tick (a fast VPS EA).
Flat across data gaps > 30 min.  Reports FundedNext's Quick Strike numbers: share of trades closed within 30 s and the
share of profit from them (FundedNext withholds the target / payouts above 30 %).  Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

LOTS, COMM = 0.15, 6.5
USD_PER_1 = 100 * LOTS


@numba.njit(cache=True)
def _sim(tms, bid, ask, tf_ms, lag):
    n = len(tms)
    pnl = np.empty(n)
    dur = np.empty(n)
    end = np.empty(n, np.int64)
    k = 0
    pos, entry, t0 = 0, 0.0, 0
    cur, X = -1, 0.0
    for i in range(n - lag):
        if i > 0 and tms[i] - tms[i - 1] > 1_800_000 and pos != 0:           # gap: flat at the last tick
            j = i - 1
            ex = bid[j] if pos > 0 else ask[j]
            pnl[k], dur[k], end[k] = pos * (ex - entry), (tms[j] - t0) / 1000.0, tms[j]
            k += 1
            pos = 0
        c = tms[i] // tf_ms
        if c != cur:
            cur, X = c, bid[i]
            continue
        want = 1 if bid[i] > X else (-1 if bid[i] < X else pos)
        if want != pos:
            j = i + lag
            if pos != 0:
                ex = bid[j] if pos > 0 else ask[j]
                pnl[k], dur[k], end[k] = pos * (ex - entry), (tms[j] - t0) / 1000.0, tms[j]
                k += 1
            pos, t0 = want, tms[j]
            entry = ask[j] if want > 0 else bid[j]
    return pnl[:k], dur[:k], end[:k]


def run(d: pd.DataFrame, tf_min: int, lag: int) -> dict:
    tms = d.time.dt.tz_convert(None).to_numpy().astype("datetime64[ms]").astype("int64")   # ms, any pandas unit
    p, dur, end = _sim(tms, d.bid.to_numpy(), d.ask.to_numpy(), tf_min * 60_000, lag)
    usd = p * USD_PER_1 - COMM * LOTS
    days = pd.to_datetime(end, unit="ms", utc=True).normalize()
    day = pd.Series(usd).groupby(days).sum()
    n_days = d.time.dt.normalize().nunique()
    quick = dur < 30
    prof = usd.sum()
    qs = float(usd[quick].sum() / prof * 100) if prof > 0 else float("nan")
    return {"candle": f"M{tf_min}", "fill": "same tick" if lag == 0 else f"{lag} tick later",
            "trades/day": round(len(usd) / n_days), "win %": round(100 * float((usd > 0).mean()), 1),
            "avg $/trade": round(float(usd.mean()), 2), "net $ month": round(float(prof)),
            "worst day $": round(float(day.min())), "days < -$750": int((day < -750).sum()),
            "days +": f"{int((day > 0).sum())}/{len(day)}",
            "trades < 30 s %": round(100 * float(quick.mean()), 1),
            "Quick Strike profit share %": None if qs != qs else round(qs, 1)}


def main() -> int:
    path = Path(sys.argv[1])
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], utc=True, format="ISO8601")
    d = d[(d.ask > d.bid) & (d.bid > 0)].sort_values("time").reset_index(drop=True)
    sp = d.ask - d.bid
    print(f"{path.name}: {len(d)} ticks {d.time.iloc[0]} -> {d.time.iloc[-1]}; spread median ${sp.median():.3f}, "
          f"mean ${sp.mean():.3f}; ticks per minute {len(d) / (d.time.iloc[-1] - d.time.iloc[0]).total_seconds() * 60:.0f}")
    rows = [run(d, tf, lag) for tf in (5, 1) for lag in (0, 1)]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
