"""
Strategy round 4 -- intermarket (DXY), pairs trading and chart patterns, from the classic books.

    python -m backtest.strategy_round4_intermarket            # downloads DXY once (Yahoo), then cached
    python -m backtest.strategy_round4_intermarket --offline

Same data, cost model and per-strategy PASS RULE as round 3 (backtest/strategy_round3_daily.py,
`evaluate`): >= 10 years and >= 100 trades, mean R > 0 on full / first 70 % / last 30 %, >= 6 of 8
time folds positive, positive at 2x costs, deflated Sharpe >= 0.95 with N = every idea in
research/ledger.csv + this round, positive on the broker's own D1 history.  Costs and swaps are the
round-3 broker specs (reports/round3_specs.json).  Every tested idea goes into the ledger.

PRE-REGISTERED (written before the first run; book parameters, nothing tuned):

A. Intermarket -- John Murphy, "Intermarket Analysis": gold and the non-USD currencies move against
   the US dollar index.  DXY regime = DXY close vs its 50-day SMA, as of the PREVIOUS day's close.
   A1 dxy_filter_live_xau  The live H1 momentum (strategy/btc_h1_signal PARAMS, 2 ATR stop / 3 ATR
                           target / 96-bar exit, one position) on broker XAUUSD H1, 2009-today.
                           Filter: gold longs only when DXY < SMA50, shorts only when DXY > SMA50.
                           FILTER PASS RULE (all): keeps >= 40 % of trades; kept-trade mean R above
                           removed-trade mean R on the full sample, first 70 % and last 30 %;
                           kept minus all >= 0 in >= 6 of 8 folds; Welch t (kept vs removed) >= 2.
   A2 dxy_<turtle55|clenow_trend>  Round-3 daily trend systems on XAUUSD and the six USD majors,
                           trading only WITH the dollar regime (XXXUSD/gold long when DXY < SMA50,
                           USDXXX long when DXY > SMA50).  Trades against the regime are dropped
                           (post-filter: the unfiltered system's other trades are unchanged).
                           Round-3 pass rule per symbol and pooled.
B. Pairs trading -- Ernie Chan, "Algorithmic Trading" ch. 2-3; Ganapathy Vidyamurthy, "Pairs
   Trading".  spread = log A - beta log B, beta = rolling 252-day OLS (closes <= today);
   z = (spread - 60-day mean) / 60-day std.  |z| > 2 at the close -> trade the spread back
   (short the rich leg, long the cheap one, beta-weighted) at the next opens.  Exit at the next
   open after |z| < 0.5, |z| > 4 (stop), or 20 bars.  1 R = 2 x the 60-day spread std at entry.
   Costs: both legs' broker spread + slippage, both legs' swap.
   Pairs: EURUSD/GBPUSD, AUDUSD/NZDUSD, EURJPY/GBPJPY, AUDJPY/NZDJPY, EURUSD/USDCHF, XAUUSD/XAGUSD,
   NAS100/SP500, US30/SP500, DAX40/EU50, BTCUSD/ETHUSD.  Per pair and pooled.
C. Chart patterns -- Thomas Bulkowski, "Encyclopedia of Chart Patterns"; detection as in Lo,
   Mamaysky & Wang (2000), "Foundations of Technical Analysis" (mechanical, no hindsight).
   Swing points: zigzag with a 2 x ATR(20) reversal; a swing is known only on the bar that
   confirms it.  Entry: first close beyond the confirmation line within 40 bars after the last
   swing is confirmed, filled at the next open.  Measure rule target (pattern height from the
   line), stop 0.25 ATR beyond the pattern's last extreme, time exit 60 bars.  Intraday
   stop/target ambiguity -> stop.
   C1 double_top_bottom  two highs (lows) within 1 ATR, 10-120 bars apart; line = the swing
                         low (high) between them.
   C2 head_shoulders     five swings S-x-H-x-S, head beyond both shoulders by >= 0.5 ATR,
                         shoulders within 1.5 ATR; line = neckline through the two troughs
                         (peaks), extended to the trigger bar.
   Every round-3 symbol; per symbol and pooled.

Writes reports/strategy_round4.json and STRATEGY_ROUND4.md.  Read-only; no orders.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import strategy_round3_daily as r3
from backtest.btc_strategies import build_momentum
from common.logging_setup import get_logger
from strategy.btc_h1_signal import PARAMS as LIVE_PARAMS

log = get_logger("round4", filename="round4.log")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DXY_TICKER = "DX-Y.NYB"
DXY_SMA = 50
USD_MAJORS = ["EURUSD.vx", "GBPUSD.vx", "AUDUSD.vx", "NZDUSD.vx", "USDCAD.vx", "USDCHF.vx", "USDJPY.vx"]
PAIRS = [("EURUSD.vx", "GBPUSD.vx"), ("AUDUSD.vx", "NZDUSD.vx"), ("EURJPY.vx", "GBPJPY.vx"),
         ("AUDJPY.vx", "NZDJPY.vx"), ("EURUSD.vx", "USDCHF.vx"), ("XAUUSD.vx", "XAGUSD.vx"),
         ("NAS100.vx", "SP500.vx"), ("US30.vx", "SP500.vx"), ("DAX40.vx", "EU50.vx"), ("BTCUSD.vx", "ETHUSD.vx")]
PAIR_BETA, PAIR_Z, PAIR_IN, PAIR_OUT, PAIR_STOP, PAIR_MAX = 252, 60, 2.0, 0.5, 4.0, 20
ZZ_ATR, PAT_WINDOW, PAT_HOLD = 2.0, 40, 60


# -- A. dollar regime ------------------------------------------------------------------
def dxy_regime(dxy: pd.DataFrame) -> pd.Series:
    """+1 dollar strong / -1 weak, indexed by the date it becomes KNOWN (the next day)."""
    c = dxy["close"]
    reg = np.sign(c - c.rolling(DXY_SMA).mean()).dropna()
    reg.index = reg.index + pd.Timedelta(days=1)
    return reg


def usd_sign(symbol: str) -> int:
    """+1 if the symbol rises when the dollar falls (XXXUSD, gold), -1 if it rises with it (USDXXX)."""
    return -1 if symbol.startswith("USD") else 1


def regime_at(reg: pd.Series, dates) -> np.ndarray:
    """Dollar regime known at each date (last value on or before it); 0 if unknown."""
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    i = reg.index.searchsorted(idx, side="right") - 1
    v = reg.to_numpy()
    return np.where(i >= 0, v[np.clip(i, 0, None)], 0)


def dxy_filtered(fn, reg: pd.Series, sign: int):
    """Wrap a round-3 strategy: keep only trades in the dollar-regime direction."""
    def wrapped(o, h, l, c, dates):
        out = fn(o, h, l, c, dates)
        if not out:
            return out
        r = regime_at(reg, dates[[t[0] for t in out]])
        return [t for t, rg in zip(out, r) if rg != 0 and t[2] == -sign * rg]
    return wrapped


def live_xau_trades(h1: pd.DataFrame, cost_px: float) -> pd.DataFrame:
    """The live H1 momentum on gold: entries at the next bar's open, 2/3 ATR, 96 bars, one position."""
    from backtest.btc_strategies import base_atr
    o, h, l, c = (h1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    entries, _ = build_momentum(c, h, l, **LIVE_PARAMS)
    A = base_atr(h, l, c, 14)
    out, i, n = [], 1, len(c)
    while i < n - 1:
        d = int(entries[i])
        a = A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep, sl, tp = o[i], 2 * a, 3 * a
        r, j = None, i
        for j in range(i, min(n, i + 96)):
            adverse, favour = (l[j], h[j]) if d > 0 else (h[j], l[j])
            if d * (adverse - ep) <= -sl:
                r = -1.0; break
            if d * (favour - ep) >= tp:
                r = 1.5; break
        if r is None:
            j = min(n - 1, i + 96)
            r = d * (o[j] - ep) / sl
        out.append((h1.index[i], d, r - cost_px * ep / sl))
        i = j + 1
    return pd.DataFrame(out, columns=["entry", "dir", "R"])


def filter_test(t: pd.DataFrame, keep: np.ndarray) -> dict:
    R, e = t["R"].to_numpy(), t["entry"]
    k, x = R[keep], R[~keep]
    cut = e.min() + (e.max() - e.min()) * 0.7
    early, late = (e < cut).to_numpy(), (e >= cut).to_numpy()
    edges = pd.date_range(e.min(), e.max(), periods=9)
    folds = 0
    for a, b in zip(edges[:-1], edges[1:]):
        m = ((e >= a) & (e <= b)).to_numpy()
        if (m & keep).any() and R[m & keep].mean() - R[m].mean() >= 0:
            folds += 1
    welch = (k.mean() - x.mean()) / np.sqrt(k.var(ddof=1) / len(k) + x.var(ddof=1) / len(x))
    checks = dict(keeps_40pct=keep.mean() >= 0.4, kept_gt_removed_full=k.mean() > x.mean(),
                  kept_gt_removed_first70=R[keep & early].mean() > R[~keep & early].mean(),
                  kept_gt_removed_last30=R[keep & late].mean() > R[~keep & late].mean(),
                  folds_6of8=folds >= 6, welch_t_2=welch >= 2)
    return dict(trades=len(R), kept=int(keep.sum()), all_R=round(R.mean(), 4), kept_R=round(k.mean(), 4),
                removed_R=round(x.mean(), 4), kept_R_last30=round(R[keep & late].mean(), 4),
                removed_R_last30=round(R[~keep & late].mean(), 4), folds_positive=folds,
                welch_t=round(float(welch), 2), checks=checks, passes=all(checks.values()))


# -- B. pairs ----------------------------------------------------------------------------
def pair_trades(a: pd.DataFrame, b: pd.DataFrame, cost_a: float, cost_b: float, swaps: tuple,
                cost_mult: float = 1.0) -> pd.DataFrame:
    j = a.join(b, how="inner", rsuffix="_b").dropna()
    la, lb = np.log(j["close"].to_numpy()), np.log(j["close_b"].to_numpy())
    oa, ob = np.log(j["open"].to_numpy()), np.log(j["open_b"].to_numpy())
    sa, sb = pd.Series(la), pd.Series(lb)
    beta = (sa.rolling(PAIR_BETA).cov(sb) / sb.rolling(PAIR_BETA).var()).to_numpy()
    spread = la - beta * lb
    sp = pd.Series(spread)
    mu, sd = sp.rolling(PAIR_Z).mean().to_numpy(), sp.rolling(PAIR_Z).std().to_numpy()
    z = (spread - mu) / sd
    dates, n, out, i = j.index, len(j), [], PAIR_BETA + PAIR_Z
    (sla, ssa), (slb, ssb) = swaps
    while i < n - 2:
        if not np.isfinite(z[i]) or abs(z[i]) <= PAIR_IN or sd[i] <= 0:
            i += 1
            continue
        d, be, risk, e = (-1 if z[i] > 0 else 1), beta[i], 2 * sd[i], i + 1
        x = None
        for k in range(e, min(n - 1, i + PAIR_MAX)):
            zk = (la[k] - be * lb[k] - mu[k]) / sd[k]
            if abs(zk) < PAIR_OUT or d * zk < -PAIR_STOP:
                x = k + 1; break
        x = x or min(n - 1, i + PAIR_MAX + 1)
        pnl = d * ((oa[x] - oa[e]) - be * (ob[x] - ob[e]))
        cost = cost_mult * (cost_a + abs(be) * cost_b) * (1 + 2 * r3.SLIP_FRAC)
        nights = (dates[x] - dates[e]).days
        long_a = d > 0                                    # long spread = long A, short beta x B
        rate = ((sla if long_a else ssa) or 0) + abs(be) * (((ssb if long_a == (be > 0) else slb)) or 0)
        out.append((dates[e], dates[x], d, (pnl - cost + rate / 365 * nights) / risk))
        i = x
    return pd.DataFrame(out, columns=["entry", "exit", "dir", "R"])


# -- C. chart patterns ---------------------------------------------------------------------
def zigzag(h, l, A):
    """Swings as (extreme_i, price, kind +1 high / -1 low, confirm_i); a swing is known at confirm_i,
    the first bar that has moved ZZ_ATR x ATR back from the extreme."""
    sw, trend, hi_i, lo_i = [], 0, 0, 0
    for i in range(len(h)):
        a = A[i]
        if not np.isfinite(a):
            hi_i, lo_i = i, i
            continue
        if trend >= 0:
            if h[i] >= h[hi_i]:
                hi_i = i
            elif l[i] <= h[hi_i] - ZZ_ATR * a:
                sw.append((hi_i, h[hi_i], 1, i)); trend, lo_i = -1, i
                continue
        if trend <= 0:
            if l[i] <= l[lo_i]:
                lo_i = i
            elif h[i] >= l[lo_i] + ZZ_ATR * a:
                sw.append((lo_i, l[lo_i], -1, i)); trend, hi_i = 1, i
    return sw


def _pattern_trades(o, h, l, c, A, setups):
    """setups: (start_i, dir, line(i) -> price, stop, height) -- start_i = the confirmation bar."""
    out, busy, n = [], -1, len(c)
    for s, d, line, stop, height in sorted(setups, key=lambda x: x[0]):
        if s <= busy:
            continue
        for i in range(s, min(n - 1, s + PAT_WINDOW)):
            if d * (c[i] - line(i)) > 0:
                e = i + 1
                ep = o[e]
                risk = d * (ep - stop)
                if risk <= 0:
                    break
                tgt = ep + d * height
                x, xp = None, None
                for k in range(e, min(n, e + PAT_HOLD)):
                    hit_stop = (l[k] <= stop) if d > 0 else (h[k] >= stop)
                    if hit_stop:
                        x, xp = k, (o[k] if d * (o[k] - stop) <= 0 else stop); break
                    if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                        x, xp = k, (o[k] if d * (o[k] - tgt) >= 0 else tgt); break
                if x is None:
                    x = min(n - 1, e + PAT_HOLD); xp = o[x]
                out.append((e, x, d, ep, xp, risk))
                busy = x
                break
            if d * (c[i] - stop) <= 0:                    # invalidated before the trigger
                break
    return out


def s_double_top_bottom(o, h, l, c, dates):
    A = r3.atr(h, l, c, 20)
    sw = zigzag(h, l, A)
    setups = []
    for k in range(2, len(sw)):
        (i1, p1, k1, _), (im, pm, _, _), (i2, p2, k2, cf) = sw[k - 2], sw[k - 1], sw[k]
        a = A[cf]
        if k1 != k2 or not (10 <= i2 - i1 <= 120) or abs(p1 - p2) > a:
            continue
        d = -k2                                           # double top -> short
        ext = max(p1, p2) if d < 0 else min(p1, p2)
        setups.append((cf, d, (lambda i, v=pm: v), ext - d * 0.25 * a, abs(ext - pm)))
    return _pattern_trades(o, h, l, c, A, setups)


def s_head_shoulders(o, h, l, c, dates):
    A = r3.atr(h, l, c, 20)
    sw = zigzag(h, l, A)
    setups = []
    for k in range(4, len(sw)):
        (iL, pL, kL, _), (t1, q1, _, _), (iH, pH, kH, _), (t2, q2, _, _), (iR, pR, kR, cf) = sw[k - 4:k + 1]
        if not (kL == kH == kR):
            continue
        a, s = A[cf], kH                                  # s = +1 top, -1 inverse
        if s * (pH - (max(pL, pR) if s > 0 else min(pL, pR))) < 0.5 * a:
            continue
        if abs(pL - pR) > 1.5 * a:
            continue
        slope = (q2 - q1) / (t2 - t1)
        neck = (lambda i, q=q1, t=t1, m=slope: q + m * (i - t))
        height = abs(pH - neck(iH))
        setups.append((cf, -s, neck, pR + s * 0.25 * a, height))
    return _pattern_trades(o, h, l, c, A, setups)


PATTERNS = {"double_top_bottom": s_double_top_bottom, "head_shoulders": s_head_shoulders}


# -- driver ---------------------------------------------------------------------------------
def _years(t: pd.DataFrame) -> float:
    return (t.entry.max() - t.entry.min()).days / 365.25 if len(t) else 0.0


def load_data(offline: bool):
    specs = json.loads(r3.SPECS.read_text())
    data, broker = {}, {}
    for sym in sorted(set(r3.YAHOO) | r3.BROKER_PRIMARY):
        if sym not in specs:
            continue
        broker[sym] = r3.broker_d1(None, sym)
        data[sym] = broker[sym] if sym in r3.BROKER_PRIMARY else r3.yahoo(r3.YAHOO[sym], offline)
    for sym, d in list(data.items()):
        if d is not None and len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9) < r3.MIN_BARS_PER_YEAR:
            data[sym] = None
    return specs, data, broker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    specs, data, broker = load_data(a.offline)
    dxy = r3.yahoo(DXY_TICKER, a.offline)
    if dxy is None:
        raise SystemExit("no DXY data")
    reg = dxy_regime(dxy)
    cross = {"XAUUSD.vx": r3.yahoo("GC=F", a.offline)}       # gold: broker D1 is primary, futures 2nd source

    def second(s):
        return cross.get(s) if s in r3.BROKER_PRIMARY else broker.get(s)
    sfrac = {s: r3.broker_spread_frac(specs[s]) for s in data if s in specs}
    live = r3.live_daily_R()

    a2 = [(st, s) for st in ("turtle55", "clenow_trend") for s in ["XAUUSD.vx"] + USD_MAJORS if data.get(s) is not None]
    pairs = [(x, y) for x, y in PAIRS if data.get(x) is not None and data.get(y) is not None]
    pats = [(st, s) for st in PATTERNS for s in data if data[s] is not None]
    pooled_n = 2 + 1 + len(PATTERNS)
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + 1 + len(a2) + len(pairs) + len(pats) + pooled_n
    log.info("round 4: %d A2, %d pairs, %d pattern tests; deflated-Sharpe N=%d", len(a2), len(pairs), len(pats), n_trials)

    def corr_live(t):
        if not len(t):
            return None
        dr = t.groupby(pd.to_datetime(t.exit).dt.normalize()).R.sum()
        j = pd.concat([dr, live], axis=1, join="inner").fillna(0.0)
        return round(float(j.corr().iloc[0, 1]), 3) if len(j) > 50 else None

    res, pool, ledger = {}, {}, []

    # A1 -- DXY filter on the live H1 momentum, gold
    h1 = pd.read_csv(r3.H1CACHE / "XAUUSD.vx.csv", parse_dates=["time"]).set_index("time")
    lt = live_xau_trades(h1, sfrac["XAUUSD.vx"] * (1 + 2 * r3.SLIP_FRAC))
    rg = regime_at(reg, lt.entry)
    keep = (rg != 0) & (lt["dir"].to_numpy() == -rg)
    res["A1_dxy_filter_live_xau"] = filter_test(lt, keep)
    log.info("A1 %s", {k: v for k, v in res["A1_dxy_filter_live_xau"].items() if k != "checks"})

    # A2 -- daily trend systems traded only with the dollar regime
    for st, s in a2:
        sp = specs[s]
        fn = dxy_filtered(r3.STRATEGIES[st], reg, usd_sign(s))
        t = r3.trade_table(data[s], fn, sfrac[s], sp["swap_long"], sp["swap_short"])
        t2 = r3.trade_table(data[s], fn, sfrac[s], sp["swap_long"], sp["swap_short"], cost_mult=2.0)
        base = r3.trade_table(data[s], r3.STRATEGIES[st], sfrac[s], sp["swap_long"], sp["swap_short"])
        b, bR = second(s), None
        if b is not None and (b.index[-1] - b.index[0]).days >= 3 * 365:
            tb = r3.trade_table(b, fn, sfrac[s], sp["swap_long"], sp["swap_short"])
            bR = float(tb.R.mean()) if len(tb) else None
        r = r3.evaluate(t, t2, (data[s].index[-1] - data[s].index[0]).days / 365.25, n_trials, bR)
        r.update(unfiltered_R=round(float(base.R.mean()), 4) if len(base) else None, corr_live=corr_live(t))
        res.setdefault(f"A2_dxy_{st}", {})[s] = r
        p = pool.setdefault(f"A2_dxy_{st}", dict(t=[], t2=[], b=[]))
        p["t"].append(t); p["t2"].append(t2)
        if bR is not None:
            p["b"].append(tb.R)
        log.info("A2 %-13s %-10s trades=%s R=%s (unfiltered %s) PASS=%s", st, s, r.get("trades"), r.get("mean_R"),
                 r["unfiltered_R"], r.get("passes"))

    # B -- pairs
    for x, y in pairs:
        cx, cy = sfrac[x], sfrac[y]
        sw = ((specs[x]["swap_long"], specs[x]["swap_short"]), (specs[y]["swap_long"], specs[y]["swap_short"]))
        t = pair_trades(data[x], data[y], cx, cy, sw)
        t2 = pair_trades(data[x], data[y], cx, cy, sw, cost_mult=2.0)
        bx, by, bR = second(x), second(y), None
        if bx is not None and by is not None and min(bx.index[-1] - bx.index[0], by.index[-1] - by.index[0]).days >= 3 * 365:
            tb = pair_trades(bx, by, cx, cy, sw)
            bR = float(tb.R.mean()) if len(tb) else None
        r = r3.evaluate(t, t2, _years(t), n_trials, bR)
        r["corr_live"] = corr_live(t)
        res.setdefault("B_pairs", {})[f"{x}/{y}"] = r
        p = pool.setdefault("B_pairs", dict(t=[], t2=[], b=[]))
        p["t"].append(t); p["t2"].append(t2)
        if bR is not None:
            p["b"].append(tb.R)
        log.info("B  %-22s trades=%s R=%s last30=%s WF=%s PASS=%s", f"{x}/{y}", r.get("trades"), r.get("mean_R"),
                 r.get("last30_R"), r.get("wf_positive"), r.get("passes"))

    # C -- chart patterns
    for st, s in pats:
        sp = specs[s]
        fn = PATTERNS[st]
        t = r3.trade_table(data[s], fn, sfrac[s], sp["swap_long"], sp["swap_short"])
        t2 = r3.trade_table(data[s], fn, sfrac[s], sp["swap_long"], sp["swap_short"], cost_mult=2.0)
        b, bR = second(s), None
        if b is not None and (b.index[-1] - b.index[0]).days >= 3 * 365:
            tb = r3.trade_table(b, fn, sfrac[s], sp["swap_long"], sp["swap_short"])
            bR = float(tb.R.mean()) if len(tb) else None
        r = r3.evaluate(t, t2, (data[s].index[-1] - data[s].index[0]).days / 365.25, n_trials, bR)
        r["corr_live"] = corr_live(t)
        res.setdefault(f"C_{st}", {})[s] = r
        p = pool.setdefault(f"C_{st}", dict(t=[], t2=[], b=[]))
        p["t"].append(t); p["t2"].append(t2)
        if bR is not None:
            p["b"].append(tb.R)

    pooled = {}
    for name, p in pool.items():
        t = pd.concat([x for x in p["t"] if len(x)], ignore_index=True)
        t2 = pd.concat([x for x in p["t2"] if len(x)], ignore_index=True)
        bR = float(pd.concat(p["b"]).mean()) if p["b"] else None
        r = r3.evaluate(t, t2, _years(t), n_trials, bR)
        r.update(corr_live=corr_live(t), members=len(p["t"]))
        r["useful"] = bool(r.get("passes") and (r["corr_live"] is None or r["corr_live"] < 0.5))
        pooled[name] = r
        log.info("POOLED %-22s trades=%s R=%s last30=%s WF=%s/8 DSR=%s PASS=%s", name, r.get("trades"), r.get("mean_R"),
                 r.get("last30_R"), r.get("wf_positive"), r.get("dsr"), r.get("passes"))

    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    a1 = res["A1_dxy_filter_live_xau"]
    ledger.append(dict(tested_utc=tested, name="r4_dxy_filter_live_xau", params=json.dumps({"sma": DXY_SMA}),
                       lookahead_ok=True, xau_exp_R=a1["kept_R"], n_trials=n_trials, passes=a1["passes"],
                       useful=a1["passes"]))
    for name, rs in res.items():
        if name.startswith("A1"):
            continue
        for key, r in rs.items():
            ledger.append(dict(tested_utc=tested, name=f"r4_{name}", params=json.dumps({"symbol": key, "tf": "D1"}),
                               lookahead_ok=True, corr_with_live=r.get("corr_live", ""), n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")),
                               useful=bool(r.get("passes") and (r.get("corr_live") or 0) < 0.5)))
    for name, r in pooled.items():
        ledger.append(dict(tested_utc=tested, name=f"r4_{name}_pooled", params=json.dumps({"members": r["members"]}),
                           lookahead_ok=True, corr_with_live=r.get("corr_live", ""), n_trials=n_trials,
                           deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=r["useful"]))
    r3.append_ledger(ledger)

    rep = dict(meta=dict(n_trials=n_trials, dxy_sma=DXY_SMA, runtime_s=round(time.perf_counter() - t0, 1),
                         rule="pre-registered in module docstring"), pooled=pooled, results=res)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "strategy_round4.json").write_text(json.dumps(rep, indent=2, default=str))
    (ROOT / "STRATEGY_ROUND4.md").write_text(_md(rep), encoding="utf-8")
    print(_md(rep))
    return 0


def _row(name, r):
    return (f"| {name} | {r.get('trades')} | {r.get('trades_per_year', '')} | {r.get('mean_R')} | "
            f"{'' if r.get('win_rate') is None else round(100 * r['win_rate'])} | {r.get('profit_factor', '')} | "
            f"{r.get('first70_R', '')} | {r.get('last30_R', '')} | {r.get('wf_positive', '')}/8 | {r.get('cost2x_R', '')} | "
            f"{r.get('broker_R', '')} | {r.get('dsr', '')} | {r.get('corr_live', '')} | {'**yes**' if r.get('passes') else 'no'} |")


def _md(rep) -> str:
    m, res = rep["meta"], rep["results"]
    a1 = res["A1_dxy_filter_live_xau"]
    hdr = ["| test | trades | /yr | mean R | win % | PF | first 70 % | last 30 % | WF+ | 2x cost | broker D1 | DSR | corr live | PASS |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    L = ["# Strategy round 4 -- intermarket (DXY), pairs trading, chart patterns", "",
         f"Deflated-Sharpe N = {m['n_trials']}.  Round-3 data, broker costs and pass rule; rules pre-registered in "
         "`backtest/strategy_round4_intermarket.py`.", "",
         f"## A1. DXY filter on the live H1 momentum, gold (DXY vs {m['dxy_sma']}-day SMA)", "",
         f"- {a1['trades']} trades; filter keeps {a1['kept']} ({100 * a1['kept'] / a1['trades']:.0f} %)",
         f"- mean R: all {a1['all_R']}, kept {a1['kept_R']}, removed {a1['removed_R']} "
         f"(last 30 %: kept {a1['kept_R_last30']}, removed {a1['removed_R_last30']})",
         f"- folds where filtering helped: {a1['folds_positive']}/8, Welch t = {a1['welch_t']}",
         f"- checks: {', '.join(k + ('' if v else ' FAIL') for k, v in a1['checks'].items())}",
         f"- **PASS: {'yes' if a1['passes'] else 'no'}**", "", "## Pooled per idea", ""] + hdr
    for name, r in rep["pooled"].items():
        L.append(_row(f"{name} ({r['members']})", r))
    for name, rs in res.items():
        if name.startswith("A1"):
            continue
        rows = sorted(rs.items(), key=lambda kv: -(kv[1].get("mean_R") or -9))
        L += ["", f"## {name}" + (" (top 15 by mean R)" if len(rows) > 15 else ""), ""] + hdr
        for key, r in rows[:15]:
            extra = f" (unfiltered {r['unfiltered_R']})" if "unfiltered_R" in r else ""
            L.append(_row(key + extra, r))
    passers = [f"{n}|{k}" for n, rs in res.items() if not n.startswith("A1") for k, r in rs.items() if r.get("passes")]
    passers += [f"{n} (pooled)" for n, r in rep["pooled"].items() if r.get("passes")]
    L += ["", f"**Passing ({len(passers)}):** {', '.join(passers) or 'none'}", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
