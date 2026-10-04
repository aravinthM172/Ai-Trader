"""
Swap / overnight-financing sensitivity for momentum_rsi_mtf (ADD_LIST Phase 0).

The backtest cost model has spread + slippage only.  Real CFD positions pay financing
at each daily rollover.  Valetax swap rates are read by tools/fetch_swap.py when MT5
is running; until then this answers: at what ANNUAL financing rate does the edge vanish?

Model (conservative):
  * financing charged on BOTH long and short, every calendar night (crypto CFDs trade 7 days)
  * rollover at ROLLOVER_UTC_HOUR (Valetax server ~ EET -> 21:00/22:00 UTC)
  * cost per night = price x annual_rate / 365 per unit; in R this is
        price x rate / 365 x nights / stop_distance        (lot size cancels)
    with stop_distance = sl_atr_mult x ATR at entry (the engine's stop).

    python -m backtest.swap_sensitivity

Writes reports/swap_sensitivity.json.  Research only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import btc_h1_validation as bv
from backtest import xau_h1_validation as xv
from backtest import btc_strategies as strat
from backtest.research_strategies import LIVE_MOMENTUM

ROLLOVER_UTC_HOUR = 21
RATES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)


def nights_held(entry_t: pd.Series, exit_t: pd.Series) -> np.ndarray:
    """Number of rollover instants in (entry, exit]."""
    shift = pd.Timedelta(hours=ROLLOVER_UTC_HOUR)
    a = (entry_t - shift).dt.floor("D")
    b = (exit_t - shift).dt.floor("D")
    return ((b - a).dt.days).clip(lower=0).to_numpy()


def swap_table(name, df, evaluate, sl_mult):
    e = evaluate(name, strat.build_momentum, df, LIVE_MOMENTUM)
    t = e["_trades"]
    times = df["time"].reset_index(drop=True)
    R = np.where(t["risk"] > 0, t["pnl"] / t["risk"], 0.0)
    n = nights_held(times.iloc[t["entry_i"]].reset_index(drop=True),
                    times.iloc[t["exit_i"]].reset_index(drop=True))
    stop = sl_mult * e["_atr"][t["entry_i"] - 1]
    per_rate = t["entry_px"] / 365.0 * n / np.where(stop > 0, stop, np.nan)     # R cost per 1.0 annual rate
    per_rate = np.nan_to_num(per_rate)
    rows = {f"{r:.0%}": round(float((R - r * per_rate).mean()), 4) for r in RATES}
    breakeven = float(R.mean() / per_rate.mean()) if per_rate.mean() > 0 else None
    return dict(trades=int(len(R)), mean_nights=round(float(n.mean()), 2),
                share_held_overnight=round(float((n > 0).mean()), 3),
                expectancy_R_by_annual_rate=rows,
                breakeven_annual_rate=round(breakeven, 3) if breakeven else None)


def main() -> int:
    rep = dict(model=__doc__.split("Model (conservative):")[1].split("python -m")[0].strip(),
               btc_valetax=swap_table("btc", bv.load(), bv.evaluate, bv.EDGE_RISK.sl_atr_mult),
               xau_valetax=swap_table("xau", xv.load(), xv.evaluate, xv.EDGE_RISK.sl_atr_mult))
    out = ROOT / "reports" / "swap_sensitivity.json"
    out.write_text(json.dumps(rep, indent=2))
    for k in ("btc_valetax", "xau_valetax"):
        r = rep[k]
        print(f"\n{k}: {r['trades']} trades, {r['share_held_overnight']:.0%} held past a rollover, "
              f"avg {r['mean_nights']} nights")
        for rate, v in r["expectancy_R_by_annual_rate"].items():
            print(f"   annual financing {rate:>4}: expectancy {v:+.4f} R")
        print(f"   edge disappears at ~{r['breakeven_annual_rate']:.0%} a year (both sides charged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
