"""
Owner idea 2026-10-21: gold M5 / M1 candle-colour flip with a bulk of 0.01 lots.
  Green candle closes -> buy 10-15 x 0.01 lots.  First red candle -> close all and sell (and the mirror).

    python -m backtest.gold_candle_flip [file]          (default data/xauusd_m5_5y.csv, broker M5 with spread column)

Fills at the next bar's open; buy pays the ask (bid + that bar's recorded spread), sells get the bid; + 0.25 x spread
slippage per side; commission $1.30 per lot round turn (FundedNext gold, from the live trade history: $0.13 on 0.02 x
... scaled is $6.5/lot, so we use the higher of the two below).  15 x 0.01 = 0.15 lot = $15 per $1 move.
Variants: A the owner's rule (always in, flip on every colour change); B buy-only (green -> buy, red -> flat);
C need 2 candles of the new colour to flip; D A but only with the H1 EMA200 trend; E A only 07:00-20:00 UTC.
Reports per variant: trades/day, average move captured, spread + commission paid, net $, worst day, days that would
break the FundedNext $750 daily limit, net per month.  Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOTS = 0.15
USD_PER_1 = 100 * LOTS                        # contract 100 oz
COMM_PER_LOT = 6.5                            # FundedNext metals, round turn (0.02 lot paid $0.13 one way in the live history)
SLIP = 0.25


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["spr"] = d["spread"] / 100.0 if "spread" in d else 0.15        # points -> $ (2-digit gold)
    return d.sort_values("time").reset_index(drop=True)


def h1_trend(d: pd.DataFrame) -> np.ndarray:
    h = d.set_index("time").close.resample("1h").last().dropna()
    side = np.sign(h - h.ewm(span=200, adjust=False).mean()).shift(1)          # last COMPLETED hour
    return side.reindex(d.time.dt.floor("1h")).to_numpy()


def run(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    o, c, spr = d.open.to_numpy(), d.close.to_numpy(), d.spr.to_numpy()
    col = np.sign(c - o)                                                # +1 green, -1 red, 0 doji
    trend = h1_trend(d) if variant == "D" else None
    hrs = d.time.dt.hour.to_numpy()
    gap = d.time.diff().dt.total_seconds().fillna(300).to_numpy() > 3600   # weekend / daily break: go flat
    pos, entry, ent_i, rows = 0, 0.0, 0, []

    def close(i, px_bid):
        nonlocal pos
        exit_px = px_bid if pos > 0 else px_bid + spr[i]                # long sells at bid, short buys at ask
        slip = SLIP * spr[i] * 2
        pnl = (pos * (exit_px - entry) - slip) * USD_PER_1 - COMM_PER_LOT * LOTS
        rows.append((d.time.iloc[ent_i], d.time.iloc[i], pos, pnl, abs(exit_px - entry)))
        pos = 0

    for i in range(2, len(d)):
        if gap[i] and pos:
            close(i - 1, c[i - 1])                                       # flat into the break at the last close
        want = col[i - 1]
        if variant == "C" and col[i - 2] != col[i - 1]:
            want = pos                                                   # need two candles of the new colour
        if want == 0:
            want = pos
        if variant == "B" and want < 0:
            want = 0
        if variant == "D" and trend[i] == trend[i] and want != 0 and want != trend[i]:
            want = 0
        if variant == "E" and not (7 <= hrs[i] < 20):
            want = 0
        if want != pos:
            if pos:
                close(i, o[i])
            if want:
                pos, ent_i = int(want), i
                entry = o[i] + spr[i] if want > 0 else o[i]              # buy at ask, sell at bid
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "pnl", "move"])


def report(t: pd.DataFrame, d: pd.DataFrame, name: str) -> dict:
    days = d.time.dt.normalize().nunique()
    day = t.groupby(t.exit.dt.normalize()).pnl.sum()
    eq = t.pnl.cumsum()
    months = max(1.0, (d.time.iloc[-1] - d.time.iloc[0]).days / 30.44)
    spread_cost = float((d.spr.mean() * (1 + 2 * SLIP)) * USD_PER_1 + COMM_PER_LOT * LOTS)
    return {"variant": name, "trades/day": round(len(t) / days, 1), "win %": round(100 * float((t.pnl > 0).mean()), 1),
            "avg $/trade": round(float(t.pnl.mean()), 2), "cost $/trade ~": round(spread_cost, 2),
            "net $": round(float(t.pnl.sum())), "net $/month": round(float(t.pnl.sum()) / months),
            "worst day $": round(float(day.min())), "best day $": round(float(day.max())),
            "days < -$750": int((day < -750).sum()), "max drop $": round(float((eq.cummax() - eq).max()))}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "xauusd_m5_5y.csv"
    d = load(path)
    print(f"data {path.name}: {d.time.iloc[0]} -> {d.time.iloc[-1]}, {len(d)} bars, median spread ${d.spr.median():.2f}")
    names = {"A": "owner rule: flip on every colour", "B": "buy-only (green buy, red flat)",
             "C": "2 candles to flip", "D": "flip, only with H1 EMA200 trend", "E": "flip, 07-20 UTC only"}
    out = [report(run(d, k), d, f"{k} {v}") for k, v in names.items()]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
