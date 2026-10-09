"""
Exit-rule study for the live multi bot (momentum_rsi_mtf, H1, 2 ATR stop / 6 ATR target / 96 h).

Question (2026-10-09, after a BTC short got 90 % of the way to its target and turned back to the stop):
would either of these have helped over the long history?

  BE n     break-even stop: once price has moved n ATR in the trade's favour, the stop moves to the entry price
           (from the NEXT bar -- inside one H1 bar the order of high and low is unknown, so no credit is taken)
  RSI50    exit at the next bar's open once RSI(14) closes back across 50 against the trade
           (SELL: RSI >= 50, BUY: RSI <= 50) -- the signal's reason has gone

Everything else is the live setup: same entries, costs, swaps, data and filters as backtest/long_history_backtest.py
(daily EMA200 filter on XAUUSD / NDX100 / USDJPY, NDX100 + USDJPY buy-only, ETH + USDJPY at half risk).

    python -m backtest.exit_rules_study          -> EXIT_RULES_STUDY.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest.btc_strategies import base_atr, build_momentum
from strategy.btc_features import _rsi
from strategy.btc_h1_signal import PARAMS

ROOT = Path(__file__).resolve().parents[1]
LIVE = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100"]
FILTER = {"XAUUSD", "NDX100", "USDJPY"}                 # MULTI_REGIME_SYMBOLS
LONG_ONLY = {"NDX100", "USDJPY"}                       # MULTI_LONG_ONLY_SYMBOLS
RISK_USD = {"ETHUSD": 6.25, "USDJPY": 6.25}            # $5k x 0.25 % = $12.50 per R, half risk on ETH / USDJPY
VARIANTS = {"live (no change)": {}, "BE 2 ATR": {"be": 2.0}, "BE 3 ATR": {"be": 3.0}, "BE 4 ATR": {"be": 4.0},
            "RSI50 exit": {"rsi_exit": True}, "BE 3 + RSI50": {"be": 3.0, "rsi_exit": True}}


def simulate(df: pd.DataFrame, tp_atr: float, cost: dict, cost_mult: float = 1.0, *, be: float | None = None,
             rsi_exit: bool = False) -> pd.DataFrame:
    """fr.simulate_piece plus the two optional exit rules.  With be=None and rsi_exit=False it is identical."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    ent, _ = build_momentum(c, h, l, **PARAMS)
    A = base_atr(h, l, c, 14)
    rsi = _rsi(c, 14)
    rows, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep = o[i]
        sl_d, tp_d = fr.SL_ATR * a, tp_atr * a
        stop, tgt = ep - d * sl_d, ep + d * tp_d
        x, xp, why, at_open, moved, leave = None, None, "time", False, False, False
        for k in range(i, min(n, i + fr.HOLD)):
            if leave:                                              # RSI turned on the previous close
                x, xp, why, at_open = k, o[k], "rsi50", True; break
            if k > i and d * (o[k] - stop) <= 0:
                x, xp, why, at_open = k, o[k], "breakeven" if moved else "stop", True; break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop):
                x, xp, why = k, stop, "breakeven" if moved else "stop"; break
            if k > i and d * (o[k] - tgt) >= 0:
                x, xp, why, at_open = k, o[k], "target", True; break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                x, xp, why = k, tgt, "target"; break
            if be is not None and not moved and d * ((h[k] if d > 0 else l[k]) - ep) >= be * a:
                stop, moved = ep, True                             # active from the next bar
            if rsi_exit and np.isfinite(rsi[k]) and d * (rsi[k] - 50) <= 0:
                leave = True
        if x is None:
            x = min(n - 1, i + fr.HOLD)
            xp, at_open = o[x], True
        nt = fr.nights(t.iloc[i], t.iloc[x])
        rate = cost["swap_long"] if d > 0 else cost["swap_short"]
        R = (d * (xp - ep) - cost_mult * cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d
        rows.append((t.iloc[i], t.iloc[x], d, why, nt, R))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "nights", "R"])


def live_trades(t: pd.DataFrame, s: str) -> pd.DataFrame:
    if s in FILTER:
        t = t[t.trend_ok]
    if s in LONG_ONLY:
        t = t[t.dir > 0]
    return t


def portfolio(tr: pd.DataFrame, since: str) -> dict:
    p = tr[tr.exit >= since].sort_values("exit")
    usd = p.R * p.sym.map(lambda s: RISK_USD.get(s, 12.5))
    eq = usd.cumsum()
    yrs = (p.exit.max() - pd.Timestamp(since, tz="UTC")).days / 365.25
    yr = usd.groupby(p.exit.dt.year).sum()
    ps = fr.prop_sim(p[["exit"]].assign(R=usd.to_numpy() / 5000), 1.0)
    return {"trades": len(p), "$ per year": round(usd.sum() / yrs), "max DD $": round(float((eq.cummax().clip(lower=0) - eq).max())),
            "years up": f"{(yr > 0).sum()}/{len(yr)}", "prop pass": ps["outcomes"].get("pass", 0),
            "prop fail": round(ps["outcomes"].get("daily", 0) + ps["outcomes"].get("max_loss", 0), 3),
            "median days": ps["median_days_to_pass"]}


def main() -> int:
    data = {s: lh.h1(s) for s in LIVE}
    per_sym, port, reasons, alltr = {}, {}, {}, {}
    real = fr.simulate_piece
    try:
        for name, kw in VARIANTS.items():
            fr.simulate_piece = lambda p, tp, cost, kw=kw: simulate(p, tp, cost, **kw)
            trs = []
            for s, (df, src) in data.items():
                t = live_trades(lh.run(df, s), s)
                per_sym[(s, name)] = lh.stats(t)
                trs.append(t)
            tr = pd.concat(trs, ignore_index=True)
            alltr[name] = tr
            reasons[name] = (tr.reason.value_counts(normalize=True) * 100).round(1)
            for since in ("2014-01-01", "2019-01-01"):
                port[(name, since[:4])] = portfolio(tr, since)
            print(f"{name}: {len(tr)} trades", file=sys.stderr)
    finally:
        fr.simulate_piece = real

    # the base simulation must reproduce the live backtest exactly (same entries, same R)
    base = pd.concat([live_trades(lh.run(df, s), s) for s, (df, _) in data.items()], ignore_index=True)
    same = np.allclose(base.sort_values(["sym", "entry"]).R.to_numpy(),
                       alltr["live (no change)"].sort_values(["sym", "entry"]).R.to_numpy())

    pd.set_option("display.width", 250)
    ps = pd.DataFrame(per_sym).T
    ps.index.names = ["symbol", "exit rule"]
    pt = pd.DataFrame(port).T
    pt.index.names = ["exit rule", "since"]
    rs = pd.DataFrame(reasons).T.fillna(0)
    out = ["# Exit-rule study: break-even stop and RSI-50 exit (2026-10-09)", "",
           "Live setup unchanged except the exit. Same data, costs and filters as LONG_HISTORY_BACKTEST.md.",
           f"Check: the 'live (no change)' run reproduces the live backtest exactly: **{'yes' if same else 'NO'}**.", "",
           "## Per symbol (R per trade, after costs)", "", lh._md(ps), "",
           "## Portfolio (live 5 symbols, $5k account, $12.50 per R, ETH / USDJPY $6.25)", "", lh._md(pt), "",
           "## How trades end (% of trades)", "", lh._md(rs), ""]
    (ROOT / "EXIT_RULES_STUDY.md").write_text("\n".join(out), encoding="utf-8")
    print(f"same as live backtest: {same}")
    print(ps.to_string(), "\n", pt.to_string(), "\n", rs.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
