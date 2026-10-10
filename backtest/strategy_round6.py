"""
Strategy round 6 -- the six "worth testing" methods left after round 5.  (2026-10-10)

    python -m backtest.strategy_round6             -> STRATEGY_ROUND6.md, reports/strategy_round6.json
    python -m backtest.strategy_round6 --ledger    same, and record every test in research/ledger.csv

PRE-REGISTERED (written before the first run; one fixed textbook parameter set per idea, nothing tuned).
Costs everywhere: FundingPips 1-year median spread + 0.25 x spread slippage per side + FundingPips swap.

A  ATR trailing stop
   A1 on the live H1 momentum (RSI14 >= 60, 8-bar momentum, EMA96, 2 ATR stop, 96-bar time exit, live daily
      EMA200 filter and buy-only markets): after every bar the stop moves up to (highest high since entry
      - 3 ATR), never back.  Two versions: with the 6 ATR target kept, and with no target.
   A2 on the round-5 volatility breakout: no exit at the next day; from each new day the stop moves to the
      previous day's low (high for sells), never back; 10 days at most.
   Each version is compared with its own untrailed baseline on the same market and years.
B  Relative strength rotation, daily, all FundingPips markets with a D1 file: every week rank by the
   L-day return divided by the L-day volatility, buy the top 3 and sell the bottom 3 at the week's first
   open, close at the next week's first open.  1 R = 2 x ATR20.  L = 60 and L = 250.
C  IBS mean reversion on stock indices, daily: close in the lowest 20 % of the day's range and above the
   200-day average: buy the next open; sell at the open after a close in the upper half of the range, or
   after 5 days; protective stop 3 x ATR10 (= 1 R).  Long-run Yahoo cash index; FundingPips D1 as the
   second source.
D  Lead-lag, H1: leader's last hourly move >= 2 standard deviations (100 bars) while the follower moved
   < 1: trade the follower in the leader's direction at the next open, stop 1.5 ATR, target 1.5 ATR, 4 bars.
   BTC -> ETH and ETH -> BTC (both Binance, same clock), NDX100 (Dukascopy) -> BTC (Binance).
E  Outside filters on the live trades (r4.filter_test: keeps >= 40 %, kept > removed on full / first 70 % /
   last 30 %, helps in >= 6 of 8 folds, Welch t >= 2):
   E1 NDX100 buys only while VIX is under its 50-day average (yesterday's close).
   E2 gold buys only while the 10-year real yield (FRED DFII10, 2 days late) is under its 50-day average,
      sells only while it is above.
F  Crypto funding rate (Binance perpetual, every 8 h; "24 h funding" = mean of the last 3 prints; crowded
   = beyond the 90th / 10th percentile of the previous 180 days):
   F1 fade: 24 h funding crosses into crowded-long -> sell, into crowded-short -> buy; stop 2 ATR,
      target 4 ATR, 72 bars.
   F2 filter on the live BTC / ETH trades: skip buys while crowded-long, sells while crowded-short.

PASS RULE (r3.evaluate, unchanged) for A / B / C / D / F1; LEAD = every check except years_10 and dsr_95,
and t >= 2.75.  Deflated-Sharpe N = ledger + every test here.  Read-only; no orders.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round4_intermarket as r4
from backtest import strategy_round5 as r5
from backtest.btc_strategies import base_atr
from strategy.btc_features import _ema, _rsi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LIVE = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100"]
REGIME, LONG_ONLY = {"XAUUSD", "NDX100", "USDJPY"}, {"NDX100", "USDJPY"}
INDICES = {"NDX100": "^NDX", "SPX500": "^GSPC", "DJI30": "^DJI", "GER40": "^GDAXI", "JP225": "^N225", "FTSE100": "^FTSE"}
N_TESTS = 3 * len(LIVE) + 2 + len(INDICES) + 1 + 3 + 2 + 4
INF = float("inf")


def cost(sym: str) -> dict:
    return fb.cost_of(lh.SPECS[sym])


def judge(t: pd.DataFrame, t2: pd.DataFrame, years: float, n_trials: int, broker_R: float | None = None) -> dict:
    r = r3.evaluate(t, t2, years, n_trials, broker_R)
    if "checks" in r:
        r["t"] = r5.t_stat(t.R.to_numpy())
        r["last2y_R"] = round(float(t.R[t.entry >= t.entry.max() - pd.Timedelta(days=730)].mean()), 4)
        r["lead"] = bool(all(v for k, v in r["checks"].items() if k not in ("years_10", "dsr_95")) and (r["t"] or 0) >= r5.LEAD_T)
    return r


def judge_c(t: pd.DataFrame, years: float, n_trials: int) -> dict:
    """Trade table with a cost_R column: 2x cost = one more cost."""
    return judge(t, t.assign(R=t.R - t.cost_R), years, n_trials)


def years_of(df: pd.DataFrame) -> float:
    return sum((p.time.iloc[-1] - p.time.iloc[0]).days for p in fr.segments(df)) / 365.25


# -- A. trailing stops ----------------------------------------------------------------------------
@njit(cache=True)
def _run_trail(o, h, l, c, A, d, sl, tp, rel, chand, lvl_l, lvl_s, flat, hold, min_risk, max_risk):
    """Market entry at the open after signal bar j.  rel: sl / tp are distances from the entry price.
    chand > 0: after each bar stop = best price since entry -/+ chand x entry ATR.  lvl_l / lvl_s: stop levels
    known at the open of bar k (previous day's low / high).  The stop only ever tightens."""
    n = len(c)
    ei, xi, dd, why_ = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    ep_, xp_, rk_ = np.zeros(n), np.zeros(n), np.zeros(n)
    m, i = 0, 1
    while i < n - 1:
        j = i - 1
        di, a = int(d[j]), A[j]
        if di == 0 or not (a > 0):
            i += 1
            continue
        ep = o[i]
        if rel:
            sp, tg = ep - di * sl[j], ep + di * tp[j]
        else:
            sp, tg = sl[j], tp[j]
        risk = di * (ep - sp)
        if not (risk >= min_risk * a) or risk > max_risk * a:
            i += 1
            continue
        ext, x, xp, why, at_open = ep, -1, 0.0, 0, False
        for k in range(i, min(n, i + hold)):
            if k > i:
                if flat[k]:
                    x, xp, why, at_open = k, o[k], 3, True
                    break
                lv = lvl_l[k] if di > 0 else lvl_s[k]
                if not np.isnan(lv) and di * (lv - sp) > 0:
                    sp = lv
                if di * (o[k] - sp) <= 0:
                    x, xp, why, at_open = k, o[k], 1, True
                    break
            if (di > 0 and l[k] <= sp) or (di < 0 and h[k] >= sp):
                x, xp, why = k, sp, 1
                break
            if k > i and di * (o[k] - tg) >= 0:
                x, xp, why, at_open = k, o[k], 2, True
                break
            if (di > 0 and h[k] >= tg) or (di < 0 and l[k] <= tg):
                x, xp, why = k, tg, 2
                break
            if chand > 0:
                ext = max(ext, h[k]) if di > 0 else min(ext, l[k])
                ns = ext - di * chand * a
                if di * (ns - sp) > 0:
                    sp = ns
        if x < 0:
            x = min(n - 1, i + hold)
            xp, at_open = o[x], True
        ei[m], xi[m], dd[m], ep_[m], xp_[m], rk_[m], why_[m] = i, x, di, ep, xp, risk, why
        m += 1
        i = x if at_open else x + 1
    return ei[:m], xi[:m], dd[:m], ep_[:m], xp_[:m], rk_[:m], why_[:m]


def _table(p: pd.DataFrame, res, c: dict) -> pd.DataFrame:
    ei, xi, dd, ep, xp, rk, why = res
    if not len(ei):
        return pd.DataFrame(columns=["entry", "exit", "dir", "reason", "R", "cost_R"])
    t = p["time"]
    ent, ext = t.iloc[ei].reset_index(drop=True), t.iloc[xi].reset_index(drop=True)
    nt = ((ext + pd.Timedelta(hours=3)).dt.normalize() - (ent + pd.Timedelta(hours=3)).dt.normalize()).dt.days.to_numpy()
    rate = np.where(dd > 0, c["swap_long"], c["swap_short"])
    cost_R = c["cost_frac"] * ep / rk
    return pd.DataFrame(dict(entry=ent, exit=ext, dir=dd, reason=np.array(["time", "stop", "target", "flat"])[why],
                             R=(dd * (xp - ep) + ep * rate / 365 * nt) / rk - cost_R, cost_R=cost_R))


def momentum_piece(p: pd.DataFrame, c: dict, tp_atr: float, chand: float) -> pd.DataFrame:
    o, h, l, cl = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(cl)
    r, m, e, A = _rsi(cl, 14), pd.Series(cl).diff(8).to_numpy(), _ema(cl, 96), base_atr(h, l, cl, 14)
    d = np.where((r >= 60) & (m > 0) & (cl > e), 1, np.where((r <= 40) & (m < 0) & ~(cl > e), -1, 0)).astype(np.int8)
    nan, no = np.full(n, np.nan), np.zeros(n, bool)
    return _table(p, _run_trail(o, h, l, cl, A, d, 2 * A, np.full(n, tp_atr) * A if np.isfinite(tp_atr) else np.full(n, INF),
                                True, chand, nan, nan, no, 96, 0.0, INF), c)


def live_filter(t: pd.DataFrame, df: pd.DataFrame, sym: str) -> pd.DataFrame:
    if not len(t):
        return t
    if sym in REGIME:
        reg = lh.daily_regime(df, True)
        day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
        t = t[reg.side.reindex(day, method="ffill").to_numpy() == t.dir.to_numpy()]
    return t[t.dir > 0] if sym in LONG_ONLY else t


def momentum(df: pd.DataFrame, sym: str, tp_atr: float, chand: float) -> pd.DataFrame:
    parts = [momentum_piece(p, cost(sym), tp_atr, chand) for p in fr.segments(df)]
    return live_filter(pd.concat(parts, ignore_index=True), df, sym).reset_index(drop=True)


def breakout_piece(p: pd.DataFrame, c: dict, trail: bool) -> pd.DataFrame:
    o, h, l, cl = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(cl)
    s = r5.build(p.assign(vol=np.nan))["vbo_day"]
    D = r5.prev_day_ohlc(o, h, l, cl, r5.broker_day(p["time"]))
    nan = np.full(n, np.nan)
    res = _run_trail(o, h, l, cl, base_atr(h, l, cl, 14), s["d"], s["stop"], s["tgt"], False, 0.0,
                     D["pdl"] if trail else nan, D["pdh"] if trail else nan,
                     np.zeros(n, bool) if trail else s["flat"], 240 if trail else 30, r5.MIN_RISK_ATR, r5.MAX_RISK_ATR)
    return _table(p, res, c)


def breakout(df: pd.DataFrame, sym: str, trail: bool) -> pd.DataFrame:
    return pd.concat([breakout_piece(p, cost(sym), trail) for p in fr.segments(df)], ignore_index=True)


def versus(base: pd.DataFrame, var: pd.DataFrame) -> dict:
    """Baseline vs variant on the same market: R per year is what the account earns at fixed risk."""
    yrs = max((base.entry.max() - base.entry.min()).days / 365.25, 0.1)
    half = base.entry.min() + (base.entry.max() - base.entry.min()) / 2
    last = base.entry.max() - pd.Timedelta(days=730)
    per = lambda t, m: round(float(t.R[m(t)].sum()) / yrs, 1)
    dd = lambda t: round(float((t.sort_values("exit").R.cumsum().cummax() - t.sort_values("exit").R.cumsum()).max()), 1)
    out = {}
    for name, t in (("base", base), ("var", var)):
        out[name] = dict(trades=len(t), mean_R=round(float(t.R.mean()), 4), R_per_year=per(t, lambda x: x.entry >= x.entry.min()),
                         first_half=round(float(t.R[t.entry < half].sum()), 1), second_half=round(float(t.R[t.entry >= half].sum()), 1),
                         last2y=round(float(t.R[t.entry >= last].sum()), 1), max_dd_R=dd(t))
    b, v = out["base"], out["var"]
    out["better"] = bool(v["R_per_year"] > b["R_per_year"] and v["first_half"] > b["first_half"]
                         and v["second_half"] > b["second_half"] and v["last2y"] > b["last2y"])
    return out


# -- B. relative strength rotation ----------------------------------------------------------------
def rotation(L: int) -> pd.DataFrame:
    syms = [s for s in lh.SPECS if (fb.FP / f"{s}_D1.csv").exists()]
    D = {s: fb.load_d1(s) for s in syms}
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in D.values()])))
    O = pd.DataFrame({s: D[s].open for s in syms}).reindex(idx)
    score, atr20 = {}, {}
    for s, d in D.items():
        c = d.close
        score[s] = ((c / c.shift(L) - 1) / (c.pct_change().rolling(L).std() * np.sqrt(L))).reindex(idx).ffill(limit=5)
        atr20[s] = pd.Series(r3.atr(d.high.to_numpy(), d.low.to_numpy(), c.to_numpy(), 20), index=d.index).reindex(idx).ffill(limit=5)
    S, A = pd.DataFrame(score), pd.DataFrame(atr20)
    iso = idx.isocalendar()
    first = np.flatnonzero(~pd.Series(list(zip(iso.year, iso.week))).duplicated().to_numpy())
    rows = []
    for a, b in zip(first[1:-1], first[2:]):
        sc = S.iloc[a - 1]
        ok = sc.notna() & O.iloc[a].notna() & O.iloc[b].notna() & (A.iloc[a - 1] > 0)
        if ok.sum() < 10:
            continue
        rank = sc[ok].sort_values()
        for s, d in [(s, -1) for s in rank.index[:3]] + [(s, 1) for s in rank.index[-3:]]:
            ep, xp, risk, c = O[s].iloc[a], O[s].iloc[b], 2 * A[s].iloc[a - 1], cost(s)
            rate = c["swap_long"] if d > 0 else c["swap_short"]
            cost_R = c["cost_frac"] * ep / risk
            rows.append((idx[a], idx[b], d, s, (d * (xp - ep) + ep * rate / 365 * (idx[b] - idx[a]).days) / risk - cost_R, cost_R))
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "sym", "R", "cost_R"])


# -- C. IBS on indices ----------------------------------------------------------------------------
def s_ibs(o, h, l, c, dates):
    n, rng = len(c), h - l
    ibs = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5)
    s200, A = r3._roll(c, 200, "mean"), r3.atr(h, l, c, 10)
    out, i = [], 201
    while i < n - 1:
        if not (ibs[i] < 0.2 and c[i] > s200[i] and A[i] > 0):
            i += 1
            continue
        e, risk = i + 1, 3 * A[i]
        ep, x = o[e], None
        for k in range(e, min(n - 1, e + 5)):
            hit = r3._stop_hit(1, o[k], l[k], h[k], ep - risk)
            if hit is not None:
                out.append((e, k, 1, ep, hit, risk)); x = k
                break
            if ibs[k] > 0.5:
                out.append((e, k + 1, 1, ep, o[k + 1], risk)); x = k + 1
                break
        if x is None:
            x = min(n - 1, e + 5)
            out.append((e, x, 1, ep, o[x], risk))
        i = x
    return out


def ibs(sym: str, d: pd.DataFrame, mult: float = 1.0) -> pd.DataFrame:
    c = cost(sym)
    return r3.trade_table(d, s_ibs, c["cost_frac"] / (1 + 2 * r3.SLIP_FRAC), c["swap_long"], c["swap_short"], cost_mult=mult)


# -- D. lead-lag ----------------------------------------------------------------------------------
def _z(c: pd.Series) -> pd.Series:
    r = c.pct_change()
    return r / r.rolling(100).std()


def lead_lag(leader: pd.DataFrame, follower: pd.DataFrame, sym: str) -> tuple[pd.DataFrame, dict]:
    j = follower.merge(leader[["time", "close"]].rename(columns={"close": "lead"}), on="time", how="inner").sort_values("time").reset_index(drop=True)
    parts = []
    for p in fr.segments(j):
        zl, zf = _z(p.lead).to_numpy(), _z(p.close).to_numpy()
        h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
        A = base_atr(h, l, c, 14)
        s = r5._put(r5._sig(len(c)), (zl >= 2) & (np.abs(zf) < 1), (zl <= -2) & (np.abs(zf) < 1),
                    c - 1.5 * A, c + 1.5 * A, c + 1.5 * A, c - 1.5 * A)
        parts.append(r5.simulate_piece(p, dict(s, hold=4), cost(sym)))
    rl, rf = j.lead.pct_change(), j.close.pct_change()
    clock = {f"lag {k:+d}": round(float(rl.shift(k).corr(rf)), 3) for k in (-1, 0, 1)}   # +1 = leader one bar earlier
    return pd.concat(parts, ignore_index=True), clock


# -- E / F. outside data ----------------------------------------------------------------------------
def fred(series: str) -> pd.Series:
    p = DATA / "fred" / f"{series}.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=60) as r:
            p.write_bytes(r.read())
    d = pd.read_csv(p)
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0])).dropna()


def funding(pair: str) -> pd.Series:
    p = DATA / "binance_funding" / f"{pair}.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        rows, start = [], int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
        while True:
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}&limit=1000&startTime={start}"
            with urllib.request.urlopen(url, timeout=30) as r:
                k = json.loads(r.read())
            rows += [(x["fundingTime"], float(x["fundingRate"])) for x in k]
            if len(k) < 1000:
                break
            start = k[-1]["fundingTime"] + 1
            time.sleep(0.3)
        pd.DataFrame(rows, columns=["time", "rate"]).to_csv(p, index=False)
    d = pd.read_csv(p)
    t = pd.to_datetime(d["time"], unit="ms", utc=True).dt.round("h")
    return pd.Series(d["rate"].to_numpy(float), index=t).groupby(level=0).last()


def crowding(pair: str) -> pd.DataFrame:
    """Per funding print: 24 h funding and whether it is crowded (vs the PREVIOUS 180 days)."""
    f24 = funding(pair).rolling(3).mean()
    hi, lo = f24.rolling(540).quantile(0.9).shift(1), f24.rolling(540).quantile(0.1).shift(1)
    return pd.DataFrame(dict(f24=f24, long=(f24 > hi) & (f24 > 0), short=(f24 < lo) & (f24 < 0))).dropna(subset=["f24"])[hi.notna()]


def funding_fade(pair: str, sym: str) -> pd.DataFrame:
    px = r5.binance_h1(pair)[["time", "open", "high", "low", "close"]]
    cr = crowding(pair)
    enter_l = cr.long & ~cr.long.shift(1, fill_value=False)        # crossed into crowded-long -> sell
    enter_s = cr.short & ~cr.short.shift(1, fill_value=False)
    px = px[px.time >= cr.index.min()].reset_index(drop=True)
    parts = []
    for p in fr.segments(px):
        sig_t = p.time + pd.Timedelta(hours=1)                      # the bar closing at the funding time
        sell, buy = enter_l.reindex(sig_t).fillna(False).to_numpy(bool), enter_s.reindex(sig_t).fillna(False).to_numpy(bool)
        h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
        A = base_atr(h, l, c, 14)
        s = r5._put(r5._sig(len(c)), buy, sell, c - 2 * A, c + 2 * A, c + 4 * A, c - 4 * A)
        parts.append(r5.simulate_piece(p, dict(s, hold=72), cost(sym)))
    return pd.concat(parts, ignore_index=True)


def state_at(series: pd.Series, when: pd.Series) -> np.ndarray:
    """Last value of `series` at or before each timestamp (tz-aware index)."""
    return series.reindex(pd.DatetimeIndex(when), method="ffill").to_numpy()


def live_trades(sym: str) -> pd.DataFrame:
    t = lh.live_rows(lh.run(lh.h1(sym)[0], sym))
    return (t[t.dir > 0] if sym in LONG_ONLY else t).reset_index(drop=True)


# -- run ------------------------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore")
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + N_TESTS
    res, notes = {}, {}

    # A
    A = {}
    for s in LIVE:
        df = lh.h1(s)[0]
        yrs = years_of(df)
        base = momentum(df, s, 6.0, 0.0)
        for name, tp in (("A1_mom_trail3_tp6", 6.0), ("A1_mom_trail3_notp", INF)):
            v = momentum(df, s, tp, 3.0)
            res[f"{name}|{s}"] = judge_c(v, yrs, n_trials)
            A[f"{name}|{s}"] = versus(base, v)
        b0, b1 = breakout(df, s, False), breakout(df, s, True)
        res[f"A2_vbo_daytrail|{s}"] = judge_c(b1, yrs, n_trials)
        A[f"A2_vbo_daytrail|{s}"] = versus(b0, b1)
        if s == "BTCUSD":                                            # the new simulator reproduces the known ones
            p = fr.segments(df)[-1]
            ref = fr.simulate_piece(p, 6.0, cost(s))
            mine = momentum_piece(p, cost(s), 6.0, 0.0)
            ref5 = r5.simulate(df.assign(vol=np.nan), cost(s))["vbo_day"]
            notes["check_momentum"] = bool(len(ref) == len(mine) and np.allclose(ref.R.to_numpy(), mine.R.to_numpy()))
            notes["check_breakout"] = bool(len(ref5) == len(b0) and np.allclose(ref5.R.to_numpy(), b0.R.to_numpy()))
        print("A", s, {k.split("|")[0]: v["var"]["R_per_year"] - v["base"]["R_per_year"] for k, v in A.items() if k.endswith(s)}, file=sys.stderr)

    # B
    for L in (60, 250):
        t = rotation(L)
        r = judge_c(t, (t.exit.max() - t.entry.min()).days / 365.25, n_trials)
        r.update(long_R=round(float(t.R[t.dir > 0].mean()), 4), short_R=round(float(t.R[t.dir < 0].mean()), 4))
        res[f"B_rotation_{L}d|all"] = r
        print("B", L, r.get("mean_R"), file=sys.stderr)

    # C
    pool, pool2 = [], []
    for s, tick in INDICES.items():
        y = r3.yahoo(tick, offline=False)
        if y is None or len(y) < 1000:
            notes[f"C_{s}"] = "no Yahoo data"
            continue
        t, t2 = ibs(s, y), ibs(s, y, 2.0)
        tb = ibs(s, fb.load_d1(s))
        res[f"C_ibs|{s}"] = judge(t, t2, (y.index[-1] - y.index[0]).days / 365.25, n_trials, float(tb.R.mean()) if len(tb) >= 30 else None)
        res[f"C_ibs|{s}"]["fp_trades"] = len(tb)
        pool.append(t.assign(sym=s)); pool2.append(t2)
    if pool:
        t = pd.concat(pool, ignore_index=True).sort_values("entry").reset_index(drop=True)
        res["C_ibs|pooled"] = judge(t, pd.concat(pool2, ignore_index=True), (t.exit.max() - t.entry.min()).days / 365.25, n_trials)
        ibs_trades = t
    print("C", {k: v.get("mean_R") for k, v in res.items() if k.startswith("C_")}, file=sys.stderr)

    # D
    bt, et = (r5.binance_h1(p)[["time", "open", "high", "low", "close"]] for p in ("BTCUSDT", "ETHUSDT"))
    ndx = lh.h1("NDX100")[0]
    for name, (lead, fol, sym) in {"D_btc_leads_eth": (bt, et, "ETHUSD"), "D_eth_leads_btc": (et, bt, "BTCUSD"),
                                   "D_ndx_leads_btc": (ndx, bt, "BTCUSD")}.items():
        t, clock = lead_lag(lead, fol, sym)
        res[f"{name}|{sym}"] = judge_c(t, (t.exit.max() - t.entry.min()).days / 365.25, n_trials)
        notes[f"{name}_clock"] = clock
    print("D", {k: v.get("mean_R") for k, v in res.items() if k.startswith("D_")}, file=sys.stderr)

    # E
    filt = {}
    vix = r3.yahoo("^VIX", offline=False)
    if vix is not None:
        calm = (vix.close < vix.close.rolling(50).mean()).shift(1).dropna()
        t = live_trades("NDX100")
        day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
        k = calm.reindex(pd.DatetimeIndex(day), method="ffill")
        ok = k.notna().to_numpy()
        filt["E1_vix_ndx100"] = r4.filter_test(t[ok].reset_index(drop=True), k[ok].to_numpy(bool))
    else:
        notes["E1"] = "no VIX data"
    ry = fred("DFII10")
    falling = (ry < ry.rolling(50).mean())
    falling.index = falling.index + pd.Timedelta(days=2)
    t = live_trades("XAUUSD")
    day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    k = falling.reindex(pd.DatetimeIndex(day), method="ffill")
    ok = k.notna().to_numpy()
    kk = k[ok].to_numpy(bool)
    tt = t[ok].reset_index(drop=True)
    filt["E2_real_yield_gold"] = r4.filter_test(tt, np.where(tt.dir.to_numpy() > 0, kk, ~kk))

    # F
    for pair, s in (("BTCUSDT", "BTCUSD"), ("ETHUSDT", "ETHUSD")):
        t = funding_fade(pair, s)
        res[f"F1_funding_fade|{s}"] = judge_c(t, (t.exit.max() - t.entry.min()).days / 365.25, n_trials)
        cr = crowding(pair)
        lt = live_trades(s)
        lt = lt[lt.entry >= cr.index.min() + pd.Timedelta(days=1)].reset_index(drop=True)
        cl, cs = state_at(cr.long.astype(float), lt.entry) > 0.5, state_at(cr.short.astype(float), lt.entry) > 0.5
        filt[f"F2_funding_filter|{s}"] = r4.filter_test(lt, ~np.where(lt.dir.to_numpy() > 0, cl, cs))
    print("F", {k: v.get("mean_R") for k, v in res.items() if k.startswith("F1")}, file=sys.stderr)

    # what a passing idea would add next to the live bot
    extra = {}
    if res.get("C_ibs|pooled", {}).get("mean_R", 0) > 0:
        live = pd.concat([live_trades(s)[["entry", "exit", "R"]].assign(w=0.5 if s in ("ETHUSD", "USDJPY") else 1.0) for s in LIVE])
        since = pd.Timestamp("2018-03-01", tz="UTC")
        add = ibs_trades.assign(entry=ibs_trades.entry.dt.tz_localize("UTC"), exit=ibs_trades.exit.dt.tz_localize("UTC"), w=1.0)
        for name, parts in (("live now", [live]), ("live + IBS on 6 indices at 0.25 %", [live, add[["entry", "exit", "R", "w"]]])):
            p = pd.concat(parts, ignore_index=True)
            p = p[p.entry >= since].sort_values("exit")
            usd = p.R * p.w * 12.5
            eq = usd.cumsum()
            ps = fr.prop_sim(p[["exit"]].assign(R=(p.R * p.w).to_numpy()), 0.0025)
            o = ps["outcomes"]
            yrs = (p.exit.max() - since).days / 365.25
            extra[name] = {"trades/yr": round(len(p) / yrs), "$ / yr": round(usd.sum() / yrs), "max DD $": round(float((eq.cummax() - eq).max())),
                           "pass": o.get("pass", 0), "fail": round(o.get("daily", 0) + o.get("max_loss", 0), 3), "median days": ps["median_days_to_pass"]}

    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "--ledger" in sys.argv:
        r3.append_ledger([dict(tested_utc=tested, name=f"r6_{k.replace('|', '_')}", params="{}", lookahead_ok=True, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                          for k, r in {**res, **filt}.items()])
    rep = dict(meta=dict(n_trials=n_trials, generated=tested, notes=notes, ledger="--ledger" in sys.argv),
               results=res, trailing=A, filters=filt, portfolio=extra)
    (ROOT / "reports" / "strategy_round6.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "STRATEGY_ROUND6.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep: dict) -> str:
    m, res = rep["meta"], rep["results"]
    L = ["# Strategy round 6 -- trailing stops, rotation, IBS, lead-lag, outside filters, funding rate", "",
         f"Deflated-Sharpe N = {m['n_trials']}.  FundingPips costs and swaps.  Rules pre-registered in `backtest/strategy_round6.py`.",
         f"Simulator checks: momentum baseline = live simulation **{m['notes'].get('check_momentum')}**, "
         f"breakout baseline = round 5 **{m['notes'].get('check_breakout')}**.", "",
         "## A. Trailing stops vs the untrailed baseline (R per year at fixed risk)", "",
         "| test | market | base R/yr | trailed R/yr | base mean R | trailed mean R | 1st half | 2nd half | last 2 y | base DD | trailed DD | better everywhere |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for k, v in rep["trailing"].items():
        n, s = k.split("|")
        b, x = v["base"], v["var"]
        L.append(f"| {n} | {s} | {b['R_per_year']} | {x['R_per_year']} | {b['mean_R']} | {x['mean_R']} | "
                 f"{b['first_half']} -> {x['first_half']} | {b['second_half']} -> {x['second_half']} | {b['last2y']} -> {x['last2y']} | "
                 f"{b['max_dd_R']} | {x['max_dd_R']} | {'**yes**' if v['better'] else 'no'} |")
    L += ["", "## Stand-alone tests", "",
          "| test | market | trades | /yr | mean R | t | win % | PF | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | 2nd src | DSR | PASS | lead | failed checks |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|---|"]
    for k, r in res.items():
        n, s = k.split("|")
        if "checks" not in r:
            L.append(f"| {n} | {s} | {r.get('trades')} |  |  |  |  |  |  |  |  |  |  |  |  | no |  | too few trades |")
            continue
        bad = ", ".join(c for c, v in r["checks"].items() if not v)
        L.append(f"| {n} | {s} | {r['trades']} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {round(100 * r['win_rate'])} | "
                 f"{r['profit_factor']} | {r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | "
                 f"{r.get('broker_R')} | {r['dsr']} | {'**yes**' if r['passes'] else 'no'} | {'**lead**' if r['lead'] else ''} | {bad} |")
    L += ["", "## E / F2. Filters on the live trades", "",
          "| filter | trades | kept | all R | kept R | removed R | kept R last 30 % | removed R last 30 % | folds helped | Welch t | PASS | failed checks |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|---|"]
    for k, f in rep["filters"].items():
        bad = ", ".join(c for c, v in f["checks"].items() if not v)
        L.append(f"| {k} | {f['trades']} | {f['kept']} | {f['all_R']} | {f['kept_R']} | {f['removed_R']} | {f['kept_R_last30']} | "
                 f"{f['removed_R_last30']} | {f['folds_positive']}/8 | {f['welch_t']} | {'**yes**' if f['passes'] else 'no'} | {bad} |")
    clocks = {k: v for k, v in m["notes"].items() if k.endswith("_clock")}
    if clocks:
        L += ["", "Lead-lag clock check (correlation of hourly returns; the largest must be lag 0):"]
        L += [f"- {k[:-6]}: {v}" for k, v in clocks.items()]
    if rep["portfolio"]:
        L += ["", "## What IBS would add on the $5k account (2018+, 0.25 % per trade)", "", "```",
              pd.DataFrame(rep["portfolio"]).T.to_string(), "```"]
    passing = [k for k, r in res.items() if r.get("passes")] + [k for k, f in rep["filters"].items() if f.get("passes")]
    leads = [k for k, r in res.items() if r.get("lead")]
    better = [k for k, v in rep["trailing"].items() if v["better"]]
    L += ["", f"**Passing ({len(passing)}):** {', '.join(passing) or 'none'}", "",
          f"**Leads ({len(leads)}):** {', '.join(leads) or 'none'}", "",
          f"**Trailing stop better everywhere ({len(better)}):** {', '.join(better) or 'none'}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
