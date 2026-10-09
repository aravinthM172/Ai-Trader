"""
Timeframe study: would the live strategy make more money on 15- or 30-minute candles?  (2026-10-09)

Two ways to "check more often":
  A  same rules, faster candles   RSI14 / 8-bar momentum / EMA96 / ATR14 / 96-bar hold on M15 or M30 bars
                                  -- a genuinely faster strategy (smaller stops, more trades, costs weigh more)
  B  same strategy, checked more  every length scaled so it spans the same time as on H1 (M15: RSI56, 32-bar
     often                        momentum, EMA384, ATR56, 384-bar hold) -- the H1 strategy looked at every 15 / 30 min

Everything else as live: 2 ATR stop, 6 ATR target, FundingPips costs and swaps (fb.cost_of), daily EMA200 filter on
XAUUSD / NDX100 / USDJPY, NDX100 + USDJPY buy-only, $12.50 per R (ETH / USDJPY $6.25).
Data: FundingPips MT5 candles (data/fp/<SYM>_<TF>.csv, M15 ~3.3 years, M30 5-8 years; tools: scratchpad fetch).
All three timeframes are compared on the SAME window (from the first M15 bar) so the numbers are like for like.

    python -m backtest.timeframe_study           -> TIMEFRAME_STUDY.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest.btc_strategies import _shift1, base_atr
from strategy.btc_features import _ema, _rsi

ROOT = Path(__file__).resolve().parents[1]
SYMS = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100"]
FILTER, LONG_ONLY = {"XAUUSD", "NDX100", "USDJPY"}, {"NDX100", "USDJPY"}
RISK_USD = {"ETHUSD": 6.25, "USDJPY": 6.25}
BARS_PER_H = {"H1": 1, "M30": 2, "M15": 4}
BASE = dict(rsi=14, mom=8, ema=96, atr=14, hold=96)          # live H1 lengths in bars


def load(sym: str, tf: str) -> pd.DataFrame:
    if tf == "H1":
        return fb.load_h1(sym)
    d = pd.read_csv(fb.FP / f"{sym}_{tf}.csv")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d[["time", "open", "high", "low", "close"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)


def simulate(df: pd.DataFrame, cost: dict, *, rsi: int, mom: int, ema: int, atr: int, hold: int,
             tp_atr: float = 6.0, sl_atr: float = fr.SL_ATR) -> pd.DataFrame:
    """fr.simulate_piece with the indicator lengths as parameters (signal on the closed bar, entry at next open)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    r, m, e = _rsi(c, rsi), pd.Series(c).diff(mom).to_numpy(), _ema(c, ema)
    r1, m1, up1 = _shift1(r), _shift1(m), _shift1((c > e).astype(float))
    ent = np.where((r1 >= 60) & (m1 > 0) & (up1 > 0), 1, np.where((r1 <= 40) & (m1 < 0) & (up1 < 0.5), -1, 0))
    A = base_atr(h, l, c, atr)
    rows, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep = o[i]
        sl_d = sl_atr * a
        stop, tgt = ep - d * sl_d, ep + d * tp_atr * a
        x, xp, why, at_open = None, None, "time", False
        for k in range(i, min(n, i + hold)):
            if k > i and d * (o[k] - stop) <= 0:
                x, xp, why, at_open = k, o[k], "stop", True; break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop):
                x, xp, why = k, stop, "stop"; break
            if k > i and d * (o[k] - tgt) >= 0:
                x, xp, why, at_open = k, o[k], "target", True; break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                x, xp, why = k, tgt, "target"; break
        if x is None:
            x = min(n - 1, i + hold)
            xp, at_open = o[x], True
        nt = fr.nights(t.iloc[i], t.iloc[x])
        rate = cost["swap_long"] if d > 0 else cost["swap_short"]
        R = (d * (xp - ep) - cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d
        rows.append((t.iloc[i], t.iloc[x], d, why, nt, R))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "nights", "R"])


def live_trades(df: pd.DataFrame, sym: str, params: dict) -> pd.DataFrame:
    t = simulate(df, fb.cost_of(lh.SPECS[sym]), **params)
    if not len(t):
        return t.assign(sym=sym)
    reg = lh.daily_regime(df, True)
    day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    side = reg.side.reindex(day, method="ffill").to_numpy()
    if sym in FILTER:
        t = t[side == t.dir.to_numpy()]
    if sym in LONG_ONLY:
        t = t[t.dir > 0]
    return t.assign(sym=sym)


def portfolio(tr: pd.DataFrame, since: pd.Timestamp) -> dict:
    p = tr[tr.entry >= since].sort_values("exit")
    usd = p.R * p.sym.map(lambda s: RISK_USD.get(s, 12.5))
    eq = usd.cumsum()
    yrs = (p.exit.max() - since).days / 365.25
    ps = fr.prop_sim(p[["exit"]].assign(R=usd.to_numpy() / 5000), 1.0)
    return {"trades/yr": round(len(p) / yrs), "avg R": round(p.R.mean(), 3), "$ per year": round(usd.sum() / yrs),
            "max DD $": round(float((eq.cummax().clip(lower=0) - eq).max())),
            "prop pass": ps["outcomes"].get("pass", 0),
            "prop fail": round(ps["outcomes"].get("daily", 0) + ps["outcomes"].get("max_loss", 0), 3),
            "median days": ps["median_days_to_pass"]}


def main() -> int:
    data = {(s, tf): load(s, tf) for s in SYMS for tf in BARS_PER_H}
    since = max(data[(s, "M15")].time.min() for s in SYMS) + pd.Timedelta(days=30)   # indicator warm-up
    variants = {"H1 (live)": ("H1", BASE)}
    for tf in ("M30", "M15"):
        k = BARS_PER_H[tf]
        variants[f"{tf} A: same rules"] = (tf, BASE)
        variants[f"{tf} B: H1 rules checked every {60 // k} min"] = (tf, {n: v * k for n, v in BASE.items()})
    per_sym, port, long_m30 = {}, {}, {}
    for name, (tf, params) in variants.items():
        trs = []
        for s in SYMS:
            t = live_trades(data[(s, tf)], s, params)
            trs.append(t)
            per_sym[(s, name)] = lh.stats(t[t.entry >= since])
            if tf in ("M30", "H1"):
                long_m30[(s, name)] = lh.stats(t[t.entry >= data[(s, "M30")].time.min() + pd.Timedelta(days=30)])
        tr = pd.concat(trs, ignore_index=True)
        port[name] = portfolio(tr, since)
        print(f"{name}: {len(tr)} trades", file=sys.stderr)

    # check: the generic simulator reproduces the live H1 simulation exactly
    df = data[("BTCUSD", "H1")]
    ref = fr.simulate_piece(df, 6.0, fb.cost_of(lh.SPECS["BTCUSD"]))
    mine = simulate(df, fb.cost_of(lh.SPECS["BTCUSD"]), **BASE)
    same = len(ref) == len(mine) and np.allclose(ref.R.to_numpy(), mine.R.to_numpy())

    ps = pd.DataFrame(per_sym).T
    ps.index.names = ["symbol", "version"]
    lm = pd.DataFrame(long_m30).T
    lm.index.names = ["symbol", "version"]
    pt = pd.DataFrame(port).T
    pt.index.name = "version"
    out = ["# Timeframe study: H1 vs M30 vs M15 (2026-10-09)", "",
           f"Common window: {since.date()} to {max(d.time.max() for d in data.values()).date()} (FundingPips MT5 candles). "
           "Live filters, costs and risk as on the VPS.",
           f"Check: generic simulator = live H1 simulation on BTCUSD: **{'yes' if same else 'NO'}**.", "",
           "## Portfolio (5 live symbols, $5k account)", "", lh._md(pt), "",
           "## Per symbol, common window", "", lh._md(ps), "",
           "## Per symbol, longer M30 window (5-8 years) vs H1 on the same years", "", lh._md(lm), ""]
    (ROOT / "TIMEFRAME_STUDY.md").write_text("\n".join(out), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(f"generic simulator == live H1: {same}\n", pt.to_string(), "\n", ps.to_string(), "\n", lm.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
