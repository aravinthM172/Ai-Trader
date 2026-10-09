"""
Take-profit at previous swing highs / lows ("structure") instead of a fixed ATR multiple.

    python -m backtest.structure_tp_study

Same entries and stop as the live H1 momentum (backtest/full_reassessment.simulate rules: next-open
entry, 2 ATR stop, entry bar counts, one position, 96-bar time exit, holes split, FundingPips costs).
Only the target changes.  PRE-REGISTERED (before the first run):
  tp3   3 ATR (live)                 tp6   6 ATR (proposed 2026-10-07)
  S1    nearest confirmed swing high (BUY) / low (SELL) at least 0.5 ATR beyond the entry; none -> 6 ATR
  S2    nearest confirmed swing giving >= 1 R (2 ATR); none -> 6 ATR
  S3    highest high / lowest low of the previous 100 H1 bars if >= 1 R away; else 6 ATR
Swings: zigzag with a 2 x ATR(14) reversal (strategy_round4_intermarket.zigzag), only swings CONFIRMED
on or before the signal bar.  Data: FundingPips XAUUSD H1 2010-, Bitstamp BTCUSD H1 2014-.
Added 2026-10-07 (owner request, before running): STOP from structure too --
  stop = last confirmed swing low (BUY) / high (SELL) beyond the entry, -0.2 ATR buffer; < 0.5 ATR -> 0.5 ATR;
  > 4 ATR -> trade skipped.  Targets: SS_swing (nearest swing >= 1 R of that stop, none -> 3 R), SS_2R, SS_3R.
  Also gold H4 2004- (22 years) with the same rules (96 bars = 16 days).
Rule (vs tp3, per symbol): mean R higher on the full sample, first 70 % and last 30 %.
Writes reports/structure_tp_study.json and STRUCTURE_TP_STUDY.md.  Read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import strategy_round4_intermarket as r4
from backtest.btc_strategies import base_atr, build_momentum
from strategy.btc_h1_signal import PARAMS

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("tp3", "tp6", "S1", "S2", "S3", "SS_swing", "SS_2R", "SS_3R")


def target_distance(kind: str, d: int, ep: float, a: float, i: int, h, l, swings) -> float:
    """Distance from entry to the target, in price.  swings: [(price, confirm_i, kind)] confirmed <= i-1."""
    if kind == "tp3":
        return 3 * a
    if kind == "tp6":
        return 6 * a
    if kind == "S3":
        lvl = h[max(0, i - 100):i].max() if d > 0 else l[max(0, i - 100):i].min()
        dist = d * (lvl - ep)
        return dist if dist >= 2 * a else 6 * a
    need = 0.5 * a if kind == "S1" else 2 * a
    want = 1 if d > 0 else -1                                   # swing highs for BUY, lows for SELL
    cands = [d * (p - ep) for p, cf, k in swings if k == want and d * (p - ep) >= need]
    return min(cands) if cands else 6 * a


def simulate(df: pd.DataFrame, kind: str, cost: dict) -> pd.DataFrame:
    parts = [p for p in fr.segments(df)]
    out = [_piece(p, kind, cost) for p in parts]
    out = [x for x in out if len(x)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["entry", "R", "tp_R"])


def _piece(df, kind, cost):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    ent, _ = build_momentum(c, h, l, **PARAMS)
    A = base_atr(h, l, c, 14)
    sw = r4.zigzag(h, l, A)                                    # (extreme_i, price, kind, confirm_i)
    rows, i, n, s_ptr, known = [], 1, len(c), 0, []
    while i < n - 1:
        while s_ptr < len(sw) and sw[s_ptr][3] <= i - 1:
            known.append((sw[s_ptr][1], sw[s_ptr][3], sw[s_ptr][2]))
            s_ptr += 1
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep, sl_d = o[i], 2 * a
        if kind.startswith("SS"):
            want = -1 if d > 0 else 1                           # swing lows protect a BUY, highs a SELL
            below = [d * (ep - p) for p, cf, k in known[-40:] if k == want and d * (ep - p) > 0]
            if not below:
                i += 1
                continue
            sl_d = max(min(below) + 0.2 * a, 0.5 * a)
            if sl_d > 4 * a:
                i += 1
                continue
            if kind == "SS_swing":
                tops = [d * (p - ep) for p, cf, k in known[-40:] if k == -want and d * (p - ep) >= sl_d]
                tp_d = min(tops) if tops else 3 * sl_d
            else:
                tp_d = (2 if kind == "SS_2R" else 3) * sl_d
        else:
            tp_d = target_distance(kind, d, ep, a, i, h, l, known[-40:])
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
        R = (d * (xp - ep) - cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d
        rows.append((t.iloc[i], R, tp_d / sl_d))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "R", "tp_R"])


def main() -> int:
    cst = fr.costs()
    gold = fr._read(ROOT / "data" / "gold" / "fp_XAUUSD_H1.csv")
    gold = gold[gold.time >= "2010-01-01"].reset_index(drop=True)
    btc = fr._dense_from(fr._read(ROOT / "data" / "btcusd_bitstamp_H1.csv"))
    h4 = fr._read(ROOT / "data" / "gold" / "fp_XAUUSD_H4.csv")
    rep = {}
    for sym, d in (("XAUUSD", gold), ("BTCUSD", btc), ("XAUUSD H4 2004-", h4)):
        rows = {}
        for v in VARIANTS:
            t = simulate(d, v, cst[sym.split()[0]])
            cut = t.entry.min() + (t.entry.max() - t.entry.min()) * 0.7
            rows[v] = dict(trades=len(t), mean_R=round(float(t.R.mean()), 4), win=round(float((t.R > 0).mean()), 3),
                           first70=round(float(t.R[t.entry < cut].mean()), 4), last30=round(float(t.R[t.entry >= cut].mean()), 4),
                           total_R=round(float(t.R.sum()), 1), median_tp_R=round(float(t.tp_R.median()), 2),
                           **fr.risk_stats(t))
        base = rows["tp3"]
        for v, r in rows.items():
            r["beats_tp3"] = bool(v != "tp3" and r["mean_R"] > base["mean_R"] and r["first70"] > base["first70"]
                                  and r["last30"] > base["last30"])
        rep[sym] = rows
    (ROOT / "reports" / "structure_tp_study.json").write_text(json.dumps(rep, indent=2))
    L = ["# Take-profit at previous swing highs / lows vs fixed ATR targets", "",
         "Same entries and 2 ATR stop as the live bot; verified data; rules pre-registered in `backtest/structure_tp_study.py`.", ""]
    for sym, rows in rep.items():
        L += [f"## {sym}", "", "| stop / target | trades | mean R | win % | first 70 % | last 30 % | total R | median target (R) | max DD R | streak | beats tp3 |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
        for v, r in rows.items():
            L.append(f"| {v} | {r['trades']} | {r['mean_R']} | {100 * r['win']:.0f} | {r['first70']} | {r['last30']} | {r['total_R']} | "
                     f"{r['median_tp_R']} | {r['max_dd_R']} | {r['worst_streak']} | {'**yes**' if r['beats_tp3'] else ''} |")
        L.append("")
    md = "\n".join(L)
    (ROOT / "STRUCTURE_TP_STUDY.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
