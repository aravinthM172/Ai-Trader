"""
Stop-loss x take-profit grid for the live H1 momentum, on ~20 years of verified data.

    python -m backtest.sl_tp_grid

PRE-REGISTERED (owner request 2026-10-07, written before the first run):
  Grid: stop 1.5 / 2 / 2.5 / 3 / 4 ATR(14) x target 3 / 4 / 6 / 8 ATR  (20 combinations; 2 / 3 = live).
  Simulator: backtest/full_reassessment rules (next-open entry, entry bar counts, stop before target,
    gaps at the open, one position, 96-bar time exit, holes split, FundingPips spread + swap) with the
    stop multiple as a parameter.  R = P/L / stop distance, so risk per trade is the same in every cell.
  Data: gold H1 = Dukascopy 2005-2009 + FundingPips 2010-today (both verified, seam checked: median
    close difference $0.32 in 2010); BTC H1 = Bitstamp 2014-today.
  CHOOSE on the discovery years only -- gold 2005-2015, BTC 2014-2019 -- by the average of the two
    symbols' mean R.  CONFIRM on 2016- (gold) / 2020- (BTC), untouched until the choice is made.
  PASS: the chosen cell's confirmation mean R beats the live 2 / 3 cell on both symbols and is > 0 on both.
Writes reports/sl_tp_grid.json and SL_TP_GRID.md.  Read-only; no orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import strategy_round3_daily as r3
from backtest.btc_strategies import base_atr, build_momentum
from strategy.btc_h1_signal import PARAMS
from tools import fetch_dukascopy as fd

ROOT = Path(__file__).resolve().parents[1]
STOPS = (1.5, 2.0, 2.5, 3.0, 4.0)
TARGETS = (3.0, 4.0, 6.0, 8.0)
SPLIT = {"XAUUSD": pd.Timestamp("2016-01-01", tz="UTC"), "BTCUSD": pd.Timestamp("2020-01-01", tz="UTC")}


def gold_h1() -> pd.DataFrame:
    parts = [fd._decode(p.read_bytes(), int(p.stem[:4]), int(p.stem[5:]))
             for p in sorted(fd.RAW.joinpath("XAUUSD").glob("200[5-9]_*.bi5")) if p.stat().st_size > 0]
    dk = pd.concat(parts).sort_values("time")
    dk[["open", "high", "low", "close"]] /= 1000.0
    dk = dk[dk.time < "2010-01-01"][["time", "open", "high", "low", "close"]]
    fp = fr._read(ROOT / "data" / "gold" / "fp_XAUUSD_H1.csv")
    fp = fp[fp.time >= "2010-01-01"]
    return pd.concat([dk, fp], ignore_index=True).sort_values("time").reset_index(drop=True)


def simulate(df: pd.DataFrame, sl_atr: float, tp_atr: float, cost: dict) -> pd.DataFrame:
    out = [_piece(p, sl_atr, tp_atr, cost) for p in fr.segments(df)]
    out = [x for x in out if len(x)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["entry", "R"])


def _piece(df, sl_atr, tp_atr, cost):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    ent, _ = build_momentum(c, h, l, **PARAMS)
    A = base_atr(h, l, c, 14)
    rows, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep, sl_d, tp_d = o[i], sl_atr * a, tp_atr * a
        stop, tgt = ep - d * sl_d, ep + d * tp_d
        x, xp, at_open = None, None, False
        for k in range(i, min(n, i + fr.HOLD)):
            if k > i and d * (o[k] - stop) <= 0:
                x, xp, at_open = k, o[k], True; break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop):
                x, xp = k, stop; break
            if k > i and d * (o[k] - tgt) >= 0:
                x, xp, at_open = k, o[k], True; break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                x, xp = k, tgt; break
        if x is None:
            x = min(n - 1, i + fr.HOLD); xp, at_open = o[x], True
        nt = fr.nights(t.iloc[i], t.iloc[x])
        rate = cost["swap_long"] if d > 0 else cost["swap_short"]
        rows.append((t.iloc[i], (d * (xp - ep) - cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "R"])


def main() -> int:
    cst = fr.costs()
    data = {"XAUUSD": gold_h1(), "BTCUSD": fr._dense_from(fr._read(ROOT / "data" / "btcusd_bitstamp_H1.csv"))}
    grid = {}
    for sl in STOPS:
        for tp in TARGETS:
            cell = {}
            for s, d in data.items():
                t = simulate(d, sl, tp, cst[s])
                disc, conf = t[t.entry < SPLIT[s]], t[t.entry >= SPLIT[s]]
                cell[s] = dict(trades=len(t), disc_R=round(float(disc.R.mean()), 4), conf_R=round(float(conf.R.mean()), 4),
                               all_R=round(float(t.R.mean()), 4), win=round(float((t.R > 0).mean()), 3),
                               R_per_year=round(float(t.R.sum()) / ((t.entry.max() - t.entry.min()).days / 365.25), 1),
                               **fr.risk_stats(t))
            cell["disc_avg"] = round((cell["XAUUSD"]["disc_R"] + cell["BTCUSD"]["disc_R"]) / 2, 4)
            grid[f"{sl}/{tp}"] = cell
            print(f"SL {sl} TP {tp}: disc avg {cell['disc_avg']:+.3f} | gold disc {cell['XAUUSD']['disc_R']:+.3f} conf {cell['XAUUSD']['conf_R']:+.3f} | "
                  f"BTC disc {cell['BTCUSD']['disc_R']:+.3f} conf {cell['BTCUSD']['conf_R']:+.3f}", flush=True)
    chosen = max(grid, key=lambda k: grid[k]["disc_avg"])
    live = grid["2.0/3.0"]
    ch = grid[chosen]
    passes = all(ch[s]["conf_R"] > live[s]["conf_R"] and ch[s]["conf_R"] > 0 for s in data)
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_trials = r3.ledger_n() + len(grid)
    r3.append_ledger([dict(tested_utc=tested, name=f"sltp_{k.replace('/', '_')}", params=json.dumps({"sl_tp_atr": k}),
                           lookahead_ok=True, xau_exp_R=v["XAUUSD"]["all_R"], btc_exp_R=v["BTCUSD"]["all_R"],
                           n_trials=n_trials, passes=bool(k == chosen and passes), useful=bool(k == chosen and passes))
                      for k, v in grid.items()])
    rep = dict(chosen=chosen, passes=passes, generated=tested, grid=grid,
               spans={s: f"{d.time.min().date()}..{d.time.max().date()}" for s, d in data.items()})
    (ROOT / "reports" / "sl_tp_grid.json").write_text(json.dumps(rep, indent=2))
    md = _md(rep)
    (ROOT / "SL_TP_GRID.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep) -> str:
    g = rep["grid"]
    L = ["# Stop-loss x take-profit grid -- live H1 momentum, ~20 years of verified data", "",
         f"Gold {rep['spans']['XAUUSD']} (choose 2005-2015, confirm 2016-), BTC {rep['spans']['BTCUSD']} (choose 2014-2019, "
         "confirm 2020-).  Pre-registered in `backtest/sl_tp_grid.py`.", "",
         f"**Chosen on discovery years: stop / target = {rep['chosen']} ATR -- confirmation "
         f"{'PASSED' if rep['passes'] else 'FAILED'}** (must beat the live 2 / 3 on both symbols and be > 0).", ""]
    for label, key in (("Discovery (choose) -- mean R", "disc_R"), ("Confirmation (unseen) -- mean R", "conf_R"),
                       ("Whole period -- R per year", "R_per_year"), ("Whole period -- max drawdown R", "max_dd_R"),
                       ("Whole period -- worst losing streak", "worst_streak")):
        for s in ("XAUUSD", "BTCUSD"):
            L += ["", f"### {s}: {label}", "", "| stop \\ target | " + " | ".join(f"{t:g} ATR" for t in TARGETS) + " |",
                  "|---|" + "--:|" * len(TARGETS)]
            for sl in STOPS:
                L.append(f"| {sl:g} ATR | " + " | ".join(
                    (f"**{g[f'{sl}/{tp}'][s][key]}**" if f"{sl}/{tp}" == rep["chosen"] else str(g[f"{sl}/{tp}"][s][key]))
                    for tp in TARGETS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
