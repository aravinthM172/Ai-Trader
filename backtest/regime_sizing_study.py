"""
    python -m backtest.regime_sizing_study      (writes REGIME_SIZING_STUDY.md)

Two backtests of the live H1 momentum (2026-10-08), on backtest/full_reassessment's simulator.

TEST 1 -- daily-EMA regime filter (pre-registered before running).
  Daily bars from H1 (UTC days); EMA20/50/200 of daily closes, using only the LAST COMPLETED day.
    stack : BUY only if EMA20 > EMA50 > EMA200, SELL only if EMA20 < EMA50 < EMA200, else skip
    side200: BUY only if close > EMA200, SELL only if close < EMA200
  The filter masks entries BEFORE simulating (a skipped signal leaves the slot free, as live would).
  Primary: gold, Dukascopy H1 2005-2026, exit tp3 (live).  Also tp6; second source FundingPips 2010-;
  BTC / GER40 as out-of-sample checks.
  PASS (primary) = filtered mean R > unfiltered mean R on the full period AND on first 70 % AND last 30 %,
  >= 6 of 8 time folds better, second source better, and filtered total R not lower than 80 % of unfiltered.

TEST 2 -- position sizing, portfolio XAU + BTC + GER40, 2014-2026 (common period).
  Each trade's stop as a fraction of price is replayed at TODAY's prices on a $10,000 account so min-lot
  limits bite as they do now.  Variants:
    live      : target 0.25 %; lots floored to 0.01; if 0 -> 0.01 lot when <= 1 % (allowance), else skip
    equal_X   : every trade risks exactly X % (as if lots were continuous) -- the "fix" ideal
    skip_over : target 0.25 %, no allowance: skip when 0.01 lot risks > 0.25 % x 1.25
  Report $ P/L, $ max drawdown, $ / R ratio, prop pass rate (FP 2-step, daily -3 %, total -10 %).
"""
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from backtest import full_reassessment as fr
from backtest.btc_strategies import base_atr, build_momentum
from strategy.btc_h1_signal import PARAMS

OUT = REPO
SL_ATR, HOLD = fr.SL_ATR, fr.HOLD


# ---- simulator with an entry mask (copy of fr.simulate_piece + `allow`) --------------------
def sim_piece(df, tp_atr, cost, allow=None, cost_mult=1.0):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    ent, _ = build_momentum(c, h, l, **PARAMS)
    ent = ent.astype(int).copy()
    if allow is not None:
        ent = np.where((ent > 0) & (allow[0] > 0) | (ent < 0) & (allow[1] > 0), ent, 0)
    A = base_atr(h, l, c, 14)
    rows, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1; continue
        ep = o[i]; sl_d, tp_d = SL_ATR * a, tp_atr * a
        stop, tgt = ep - d * sl_d, ep + d * tp_d
        x, xp, at_open = None, None, False
        for k in range(i, min(n, i + HOLD)):
            if k > i and d * (o[k] - stop) <= 0: x, xp, at_open = k, o[k], True; break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop): x, xp = k, stop; break
            if k > i and d * (o[k] - tgt) >= 0: x, xp, at_open = k, o[k], True; break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt): x, xp = k, tgt; break
        if x is None:
            x = min(n - 1, i + HOLD); xp, at_open = o[x], True
        nt = fr.nights(t.iloc[i], t.iloc[x])
        rate = cost["swap_long"] if d > 0 else cost["swap_short"]
        R = (d * (xp - ep) - cost_mult * cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d
        rows.append((t.iloc[i], t.iloc[x], d, R, sl_d / ep))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "R", "sl_frac"])


def regime_masks(df, kind):
    d = df.set_index("time")["close"].resample("D").last().dropna()
    e20, e50, e200 = (d.ewm(span=n, adjust=False).mean() for n in (20, 50, 200))
    if kind == "stack":
        up, dn = (e20 > e50) & (e50 > e200), (e20 < e50) & (e50 < e200)
    else:
        up, dn = d > e200, d < e200
    warm = pd.Series(np.arange(len(d)) >= 200, index=d.index)
    up, dn = (up & warm).shift(1, fill_value=False), (dn & warm).shift(1, fill_value=False)  # last completed day
    day = df["time"].dt.floor("D")
    return up.reindex(day).fillna(False).to_numpy(int), dn.reindex(day).fillna(False).to_numpy(int)


def simulate(df, tp_atr, cost, filt=None):
    parts = []
    for p in fr.segments(df):
        allow = regime_masks(p, filt) if filt else None
        parts.append(sim_piece(p, tp_atr, cost, allow))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True)


def folds_better(a, b, start, end, k=8):
    edges = pd.date_range(start, end, periods=k + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ma = a[(a.entry >= lo) & (a.entry < hi)].R.mean(); mb = b[(b.entry >= lo) & (b.entry < hi)].R.mean()
        out.append(bool(ma > mb) if np.isfinite(ma) and np.isfinite(mb) else False)
    return sum(out)


def summary(t):
    return dict(n=len(t), meanR=round(t.R.mean(), 4), totR=round(t.R.sum(), 1), **fr.risk_stats(t))


lines = []
P = lambda *s: (print(*s), lines.append(" ".join(map(str, s))))

data = fr.datasets()
cst = fr.costs()
duka_x = fr._read(REPO / "data" / "dukascopy_H1" / "XAUUSD.vx.csv")
fp_x = data["XAUUSD"][0]
sets = {"XAUUSD duka 2005-": (duka_x, "XAUUSD"), "XAUUSD FP 2010-": (fp_x, "XAUUSD"),
        "BTCUSD bitstamp 2014-": (data["BTCUSD"][0], "BTCUSD"), "GER40 duka 2014-": (data["GER40"][0], "GER40")}

P("# TEST 1 -- daily EMA regime filter\n")
P("| data | exit | filter | trades | mean R | total R | max DD R | first70 | last30 | folds better | per-year R>0 |")
P("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
res = {}
for name, (df, sym) in sets.items():
    for ex, tp in fr.EXITS.items():
        base = simulate(df, tp, cst[sym])
        cut = base.entry.iloc[0] + 0.7 * (base.entry.iloc[-1] - base.entry.iloc[0])
        for filt in (None, "stack", "side200"):
            t = base if filt is None else simulate(df, tp, cst[sym], filt)
            s = summary(t)
            f70, l30 = t[t.entry < cut].R.mean(), t[t.entry >= cut].R.mean()
            fb = "-" if filt is None else f"{folds_better(t, base, base.entry.iloc[0], base.entry.iloc[-1])}/8"
            yrs = t.groupby(t.entry.dt.year).R.sum()
            res[(name, ex, filt)] = dict(**s, f70=f70, l30=l30, t=t)
            P(f"| {name} | {ex} | {filt or 'none (live)'} | {s['n']} | {s['meanR']:+.4f} | {s['totR']:+.1f} | "
              f"{s['max_dd_R']} | {f70:+.4f} | {l30:+.4f} | {fb} | {(yrs > 0).sum()}/{len(yrs)} |")

P("\nPass check (pre-registered, gold Dukascopy, both exits):")
for ex in fr.EXITS:
    for filt in ("stack", "side200"):
        a, b = res[("XAUUSD duka 2005-", ex, filt)], res[("XAUUSD duka 2005-", ex, None)]
        s2a, s2b = res[("XAUUSD FP 2010-", ex, filt)], res[("XAUUSD FP 2010-", ex, None)]
        fb = folds_better(a["t"], b["t"], b["t"].entry.iloc[0], b["t"].entry.iloc[-1])
        checks = dict(full=a["meanR"] > b["meanR"], first70=a["f70"] > b["f70"], last30=a["l30"] > b["l30"],
                      folds=fb >= 6, second_src=s2a["meanR"] > s2b["meanR"], total_kept=a["totR"] >= 0.8 * b["totR"])
        P(f"- {ex} {filt}: {'PASS' if all(checks.values()) else 'fail'}  {checks}")

# regime breakdown of the unfiltered gold trades: where do they make / lose money?
P("\nGold (Dukascopy, tp3, live) trades split by the daily regime at entry:")
df = duka_x
t = res[("XAUUSD duka 2005-", "tp3", None)]["t"]
d = df.set_index("time")["close"].resample("D").last().dropna()
e20, e50, e200 = (d.ewm(span=n, adjust=False).mean() for n in (20, 50, 200))
reg = pd.Series("tangled", index=d.index)
reg[(e20 > e50) & (e50 > e200)] = "stacked up"; reg[(e20 < e50) & (e50 < e200)] = "stacked down"
reg = reg.shift(1).reindex(t.entry.dt.floor("D")).to_numpy()
t = t.assign(reg=reg, side=np.where(t.dir > 0, "BUY", "SELL"))
P(t.groupby(["reg", "side"]).R.agg(["count", "mean", "sum"]).round(3).to_string())

# ---- TEST 2 -- sizing -----------------------------------------------------------------------
P("\n# TEST 2 -- sizing, XAU + BTC + GER40 portfolio, live exit tp3, 2014-2026, $10,000 account at today's prices\n")
EQ = 10_000.0
spec = {"XAUUSD": dict(px=4134.58, vpu=100.0), "BTCUSD": dict(px=84213.1, vpu=1.0),
        "GER40": dict(px=25214.78, vpu=25 * 1.16)}          # GER40: 25 EUR / point / lot x EURUSD
port = []
for sym, (df, _) in {"XAUUSD": (fp_x, 0), "BTCUSD": (data["BTCUSD"][0], 0), "GER40": (data["GER40"][0], 0)}.items():
    t = simulate(df, 3.0, cst[sym]); t = t[t.entry >= "2014-01-01"].assign(sym=sym)
    t["loss_per_lot"] = t.sl_frac * spec[sym]["px"] * spec[sym]["vpu"]       # $ lost at the stop per 1.00 lot
    port.append(t)
port = pd.concat(port, ignore_index=True).sort_values("entry").reset_index(drop=True)
P("today's min-lot (0.01) risk as % of $10k, by symbol (median / p90):")
for sym, g in port.groupby("sym"):
    m = 100 * 0.01 * g.loss_per_lot / EQ
    P(f"  {sym}: {m.median():.2f} % / {m.quantile(0.9):.2f} %")


def risk_usd(t, mode, target):
    lpl = t.loss_per_lot.to_numpy()
    if mode == "equal":
        return np.full(len(t), target * EQ)
    lots = np.floor(target * EQ / lpl / 0.01 + 1e-9) * 0.01
    r = lots * lpl
    minr = 0.01 * lpl
    if mode == "live":
        return np.where(lots > 0, r, np.where(minr <= 0.01 * EQ, minr, 0.0))
    if mode == "skip_over":
        return np.where(lots > 0, r, np.where(minr <= 1.25 * target * EQ, minr, 0.0))


P("\n| sizing | trades taken | total R | total $ | $ per R | max DD $ | worst day $ | prop pass / fail | XAU / BTC / GER40 avg risk $ |")
P("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for label, mode, tg in [("live 0.25 % + min-lot allowance", "live", 0.0025), ("skip if min lot > 0.31 %", "skip_over", 0.0025),
                        ("equal 0.25 % (ideal)", "equal", 0.0025), ("equal 0.40 %", "equal", 0.004)]:
    ru = risk_usd(port, mode, tg)
    k = ru > 0
    tt = port[k].assign(usd=port.R[k] * ru[k])
    eq = tt.usd.cumsum(); dd = (eq - eq.cummax()).min()
    wd = tt.groupby(tt.exit.dt.floor("D")).usd.sum().min()
    prop = fr.prop_sim(tt.assign(R=tt.usd / (tg * EQ)), tg)       # R rescaled to the target so $ is exact
    po = prop["outcomes"]
    avg = " / ".join(f"{ru[k][(tt.sym == s).to_numpy()].mean():.0f}" for s in ("XAUUSD", "BTCUSD", "GER40"))
    P(f"| {label} | {k.sum()} | {tt.R.sum():+.1f} | {tt.usd.sum():+,.0f} | {tt.usd.sum() / tt.R.sum():.1f} | {dd:,.0f} | {wd:,.0f} | "
      f"{po.get('pass', 0):.0%} / {po.get('daily', 0) + po.get('max_loss', 0):.0%} | {avg} |")

P("\nPer symbol, live exit tp3, 2014-2026: mean R")
P(port.groupby("sym").R.agg(["count", "mean", "sum"]).round(3).to_string())
P("\nNOTE (2026-10-08): TEST 2 assumes a $10,000 account.  The live FundingPips account is $5,000, where the "
  "0.01 lot of gold risks ~0.4-0.7 % at current volatility, so the 'skip if min lot > 0.31 %' rule would block "
  "nearly every gold trade.  The real-account combined test (scratchpad combined_bt.py) favoured the 1 % cap "
  "+ 6 ATR target + gold side200 + no GER40.  Read TEST 2 as: GER40 at the minimum lot is what hurts.")
(OUT / "REGIME_SIZING_STUDY.md").write_text("\n".join(lines), encoding="utf-8")
