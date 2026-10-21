"""
SUPERSEDED by backtest/gold_tick_flip.py: with only 4 prices per minute this badly undercounts the crosses of the
candle open and fills exactly at the open, so it shows large fake profits.  Kept only to document why ticks are needed.

Owner idea 2026-10-21, second form: trade the FORMING candle.  As soon as the current gold candle is green (price above
its open) hold a bulk long of 15 x 0.01 lots; the moment it turns red (price below its open) close all and sell; flip
again on every cross.  A position carries into the next candle until that candle turns the other colour.

    python -m backtest.gold_forming_flip

Path inside each 1-minute bar: open -> low -> high -> close for a green minute, open -> high -> low -> close for a red one
(Dukascopy 1-minute BID, data/dukascopy_M1/XAUUSD.csv).  For the M5 test the candle opens and the crosses come from that
1-minute path -- close to real.  For the M1 test the path inside a minute is only 4 points, so the number of crosses is
a LOWER bound (real ticks cross more often) -- the M1 result is optimistic.
Fills at the open level when it is crossed; a buy pays the spread (broker M5 spread column, data/xauusd_m5_5y.csv, per
5-minute slot; $0.15 when missing) + 0.25 x spread slippage per side + $6.5 / lot commission round turn.  Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOTS, COMM, SLIP = 0.15, 6.5, 0.25
USD_PER_1 = 100 * LOTS


def minute_path(m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """4 price points per minute in time order, and the minute index each point belongs to."""
    o, h, l, c = (m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    up = c >= o
    pts = np.column_stack([o, np.where(up, l, h), np.where(up, h, l), c]).ravel()
    return pts, np.repeat(np.arange(len(m)), 4)


@numba.njit(cache=True)
def _flip(pts, bar_of_pt, bar_open, spread_of_pt, new_session):
    """Walk the path; long above the current candle's open, short below; returns trade P/L in price units and count."""
    pos, entry, n_tr = 0, 0.0, 0
    out = np.empty(len(pts))
    at = np.empty(len(pts), np.int64)
    prev_bar = -1
    prev_px = pts[0]
    for k in range(len(pts)):
        b = bar_of_pt[k]
        px = pts[k]
        X = bar_open[b]
        s = spread_of_pt[k]
        if b != prev_bar:
            if new_session[b] and pos != 0:                  # data gap (weekend / daily break): flat at the last price
                g = pos * (prev_px - entry)
                out[n_tr], at[n_tr] = g - (1 + 2 * SLIP) * s, k
                n_tr += 1
                pos = 0
            prev_bar = b
            prev_px = X
        # did the move prev_px -> px cross the candle open X (or start exactly on it and leave it)?
        want = pos
        if px > X:
            want = 1
        elif px < X:
            want = -1
        if want != pos and want != 0:
            fill = X if (prev_px - X) * (px - X) <= 0 else px     # crossed the open: fill at the open level
            if pos != 0:
                g = pos * (fill - entry)
                out[n_tr], at[n_tr] = g - (1 + 2 * SLIP) * s, k
                n_tr += 1
            pos, entry = want, fill
        prev_px = px
    return out[:n_tr], at[:n_tr]


def run(m: pd.DataFrame, spread: pd.Series, tf_min: int) -> dict:
    pts, minute_of_pt = minute_path(m)
    grp = m.time.dt.floor(f"{tf_min}min")
    bar_id = pd.factorize(grp)[0]
    bar_open = m.groupby(bar_id).open.first().to_numpy()
    bar_of_pt = bar_id[minute_of_pt]
    gap_min = m.time.diff().dt.total_seconds().fillna(60).to_numpy() > 1800
    new_session = pd.Series(gap_min).groupby(bar_id).max().to_numpy()
    s5 = spread.reindex(m.time.dt.floor("5min")).ffill().fillna(0.15).to_numpy()
    spread_of_pt = s5[minute_of_pt]
    pnl, at = _flip(pts, bar_of_pt, bar_open, spread_of_pt, new_session)
    usd = pnl * USD_PER_1 - COMM * LOTS
    days = m.time.dt.normalize().nunique()
    months = (m.time.iloc[-1] - m.time.iloc[0]).days / 30.44
    gross_usd = (pnl + (1 + 2 * SLIP) * np.median(spread_of_pt)) * USD_PER_1
    day_pnl = pd.Series(usd).groupby(m.time.dt.normalize().to_numpy()[minute_of_pt[at]]).sum()
    return {"timeframe": f"M{tf_min}", "trades/day": round(len(usd) / days, 1), "win %": round(100 * float((usd > 0).mean()), 1),
            "gross $/trade (before costs)": round(float(gross_usd.mean()), 2),
            "cost $/trade": round(float((1 + 2 * SLIP) * np.median(spread_of_pt) * USD_PER_1 + COMM * LOTS), 2),
            "net $ total": round(float(usd.sum())), "net $/month": round(float(usd.sum()) / months),
            "net $/day": round(float(usd.sum()) / days), "worst day $": round(float(day_pnl.min())),
            "days < -$750": int((day_pnl < -750).sum()), "days profitable %": round(100 * float((day_pnl > 0).mean()), 1)}


def main() -> int:
    m = pd.read_csv(ROOT / "data" / "dukascopy_M1" / "XAUUSD.csv")
    m["time"] = pd.to_datetime(m["time"], utc=True)
    m = m.sort_values("time").reset_index(drop=True)
    b5 = pd.read_csv(ROOT / "data" / "xauusd_m5_5y.csv")
    b5["time"] = pd.to_datetime(b5["time"], utc=True)
    spread = b5.set_index("time").spread / 100.0
    print(f"1-minute data {m.time.iloc[0]} -> {m.time.iloc[-1]} ({len(m)} minutes); broker spread median "
          f"${spread[spread.index >= m.time.iloc[0]].median():.2f}")
    rows = [run(m, spread, 5), run(m, spread, 1)]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    print("M1 row is optimistic: inside a minute only 4 price points are known, real ticks cross the open more often.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
