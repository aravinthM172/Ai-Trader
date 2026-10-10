"""
GOLD: re-test of the strategies named in the blogs / forum threads read on 2026-10-10.  Backtest only.

Each rule is taken from its source as written; where the source left something open, the choice made here is
marked (mine).  Fixed before the first run.

  Daily chart (gold futures 2000-)
    ema21_50_d1     EMA 21 / 50 cross, both sides, out on the opposite cross          [quant-signals]
    ema9_21_d1      EMA 9 / 21 cross, same                                             [quant-signals: loses after costs]
    donchian20_d1   close above the 20-day high, out below the 10-day low; mirrored   [PineForge]
    rsi_30_70_d1    buy RSI 14 < 30, out > 50; sell > 70, out < 50                     [quant-signals: struggles]
  4-hour chart (FundingPips 2004-)
    stoch_h4        long only: slow %K (14,3,3) crosses up through 20; stop 2 %, target 7 %   [backtrex]
    donchian20_h4   as the daily Donchian                                              [PineForge]
  1-hour chart (Dukascopy 2005-)
    ema9_21_h1_long long only: EMA 9 crosses above 21, out on the cross back, stop 2 ATR      [PineForge: "74 % wins"]
    bb_rsi_h1       long only: close under the lower Bollinger band (20, 2) with RSI < 30; out above the 20 SMA;
                    stop 1 ATR under the band                                           [PineForge]
    rsi_ma_h1       long only: price above EMA 50 and RSI crosses up through 30; out at RSI > 70 or under EMA 50;
                    stop 2 ATR (mine)                                                   [PineForge]
    pdh_pdl_sweep_h1  takes out yesterday's high / low and closes back: fade, stop beyond the bar, target 1R (mine) [GitHub "70 % wins"]
    day_breakout_h1 long only, New York 09:00-15:00: buy stop at the previous bar's high + 0.5 ATR, stop 1 ATR,
                    2 ATR trailing stop, out at 15:00 (multiples mine -- the source's are paywalled)  [Alpha Algo]
  15-minute chart (FundingPips 2022-)
    day_breakout_m15  the same rule on the chart the source uses
    ny_orb_m15      range 08:00-09:30 New York; a close beyond it in the first 30 minutes after 09:30: follow, stop at
                    the far side, target 2R, flat 12:00 (stop / target mine; the fair-value-gap condition is left out) [ForexFactory]
    pdh_pdl_sweep_m15
Not testable as written: the "40-pip" structure method, trendline scalping, the hedging method, the Gold Box
product, the 2011 end-of-day system (rules behind a login).

Stops inside a candle count before targets.  R = the first stop distance.  Costs: FundingPips spread + slippage + swap.
Judged with r7.judge.  Extra columns: mean R before 2019 and from 2019 (gold's bull run flatters buy-side rules).

    python -m backtest.gold_sources_retest [--ledger]
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import gold_round8 as g8
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
INF = np.inf
N_TESTS = 14
F = lambda a: np.nan_to_num(np.asarray(a, float)) > 0          # boolean, NaN -> False


def run(p, cost, *, le=None, se=None, lx=None, sx=None, stop_l=None, stop_s=None, tgt_l=None, tgt_s=None,
        hold=10 ** 9, flat=None) -> pd.DataFrame:
    """Signals on bar j act at the open of bar j + 1.  le / se: entries; lx / sx: exits; stop_* / tgt_*: price
    levels set on the signal bar; flat[k]: out at the open of bar k (clock exits)."""
    o, h, l, c = g8._arr(p)
    n, t = len(c), p["time"]
    z = np.zeros(n, bool)
    le, se, lx, sx = (z if a is None else F(a) for a in (le, se, lx, sx))
    flat = z if flat is None else flat
    rows, pos, i = [], 0, 1
    ep = sp = tp = risk = 0.0
    e_i = 0
    while i < n:
        j = i - 1
        if pos != 0:
            out, xp, why = False, 0.0, ""
            if flat[i] or (lx[j] if pos > 0 else sx[j]) or i - e_i >= hold:
                out, xp, why = True, o[i], "exit"
            elif pos * (o[i] - sp) <= 0:
                out, xp, why = True, o[i], "stop"
            elif (pos > 0 and l[i] <= sp) or (pos < 0 and h[i] >= sp):
                out, xp, why = True, sp, "stop"
            elif (pos > 0 and h[i] >= tp) or (pos < 0 and l[i] <= tp):
                out, xp, why = True, tp, "target"
            if out:
                nights = ((t.iloc[i] + pd.Timedelta(hours=3)).normalize() - (t.iloc[e_i] + pd.Timedelta(hours=3)).normalize()).days
                cR = cost["cost_frac"] * ep / risk
                swap = ep * (cost["swap_long"] if pos > 0 else cost["swap_short"]) / 365 * nights / risk
                rows.append((t.iloc[e_i], t.iloc[i], pos, why, pos * (xp - ep) / risk + swap - cR, cR))
                at_open = why == "exit" or xp == o[i]
                pos = 0
                if not at_open:
                    i += 1
                    continue
        if pos == 0 and not flat[i]:
            d = 1 if le[j] else (-1 if se[j] else 0)
            if d != 0:
                s_ = (stop_l if d > 0 else stop_s)[j]
                g_ = (tgt_l if d > 0 else tgt_s)[j] if (tgt_l is not None) else (INF if d > 0 else -INF)
                r_ = d * (o[i] - s_)
                if np.isfinite(s_) and r_ > 0 and d * (g_ - o[i]) > 0:
                    pos, ep, sp, tp, risk, e_i = d, o[i], s_, g_, r_, i
                    if (d > 0 and l[i] <= sp) or (d < 0 and h[i] >= sp):          # stopped in the entry bar
                        cR = cost["cost_frac"] * ep / risk
                        rows.append((t.iloc[i], t.iloc[i], d, "stop", -1.0 - cR, cR))
                        pos = 0
        i += 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "R", "cost_R"])


def _ind(p):
    o, h, l, c = g8._arr(p)
    return o, h, l, c, base_atr(h, l, c, 14)


def ema_cross(p, cost, fast, slow, long_only=False, stop_atr=3.0):
    o, h, l, c, A = _ind(p)
    a, b = r7.ema(c, fast), r7.ema(c, slow)
    up, dn = r7.cross_up(a, b), r7.cross_up(b, a)
    warm = np.arange(len(c)) >= slow * 3
    return run(p, cost, le=up & warm, se=None if long_only else dn & warm, lx=dn, sx=up, stop_l=c - stop_atr * A, stop_s=c + stop_atr * A)


def donchian(p, cost):
    o, h, l, c, A = _ind(p)
    hi20, lo20, hi10, lo10 = r7.prev(r7.hi(h, 20)), r7.prev(r7.lo(l, 20)), r7.prev(r7.hi(h, 10)), r7.prev(r7.lo(l, 10))
    return run(p, cost, le=c > hi20, se=c < lo20, lx=c < lo10, sx=c > hi10, stop_l=c - 3 * A, stop_s=c + 3 * A)


def rsi_30_70(p, cost):
    o, h, l, c, A = _ind(p)
    r = r7.rsi(c, 14)
    return run(p, cost, le=r < 30, se=r > 70, lx=r > 50, sx=r < 50, stop_l=c - 3 * A, stop_s=c + 3 * A)


def stoch(p, cost):
    o, h, l, c, A = _ind(p)
    hh, ll = r7.hi(h, 14), r7.lo(l, 14)
    k = r7.sma(100 * (c - ll) / np.where(hh > ll, hh - ll, np.nan), 3)
    return run(p, cost, le=(k > 20) & (r7.prev(k) <= 20), stop_l=c * 0.98, tgt_l=c * 1.07, stop_s=c, tgt_s=c)


def bb_rsi(p, cost):
    o, h, l, c, A = _ind(p)
    ma, sd = r7.sma(c, 20), pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    low = ma - 2 * sd
    return run(p, cost, le=(c < low) & (r7.rsi(c, 14) < 30), lx=c > ma, stop_l=low - A)


def rsi_ma(p, cost):
    o, h, l, c, A = _ind(p)
    r, e50 = r7.rsi(c, 14), r7.ema(c, 50)
    return run(p, cost, le=(c > e50) & (r > 30) & (r7.prev(r) <= 30), lx=(r > 70) | (c < e50), stop_l=c - 2 * A)


def pd_sweep(p, cost):
    o, h, l, c, A = _ind(p)
    D = r5.prev_day_ohlc(o, h, l, c, r5.broker_day(p["time"]))
    sell, buy = (h > D["pdh"]) & (c < D["pdh"]), (l < D["pdl"]) & (c > D["pdl"])
    sl_b, sl_s = l - 0.1 * A, h + 0.1 * A
    return run(p, cost, le=buy, se=sell, stop_l=sl_b, stop_s=sl_s, tgt_l=c + (c - sl_b), tgt_s=c - (sl_s - c), hold=24 if len(p) and (p.time.iloc[1] - p.time.iloc[0]) >= pd.Timedelta(hours=1) else 96)


def day_breakout(p, cost):
    """Buy stop above the previous bar; the stop trails 2 ATR under the best high (updated at each close)."""
    o, h, l, c, A = _ind(p)
    n, t = len(c), p["time"]
    ny = t.dt.tz_convert("America/New_York").dt.hour.to_numpy()
    rows, i, pos = [], 15, False
    ep = sp = risk = best = 0.0
    e_i = 0
    while i < n:
        if pos:
            xp = None
            if ny[i] >= 15 or ny[i] < 9:
                xp, why = o[i], "exit"
            elif o[i] <= sp:
                xp, why = o[i], "stop"
            elif l[i] <= sp:
                xp, why = sp, "stop"
            if xp is not None:
                cR = cost["cost_frac"] * ep / risk
                rows.append((t.iloc[e_i], t.iloc[i], 1, why, (xp - ep) / risk - cR, cR))
                pos = False
                if why == "stop" and xp != o[i]:
                    i += 1
                    continue
            else:
                best = max(best, h[i])
                sp = max(sp, best - 2 * A[i])
        if not pos and 9 <= ny[i] < 15 and A[i - 1] > 0:
            lvl = h[i - 1] + 0.5 * A[i - 1]
            if h[i] >= lvl:
                ep = max(o[i], lvl)
                risk = A[i - 1]
                sp, best, e_i = ep - risk, h[i], i
                if l[i] <= sp and c[i] < ep:                       # broke out and fell back through the stop in the same bar
                    cR = cost["cost_frac"] * ep / risk
                    rows.append((t.iloc[i], t.iloc[i], 1, "stop", -1.0 - cR, cR))
                else:
                    pos = True
                    sp = max(sp, best - 2 * A[i])
        i += 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "R", "cost_R"])


def ny_orb(p, cost):
    o, h, l, c, A = _ind(p)
    ny = p["time"].dt.tz_convert("America/New_York")
    m = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    day = ny.dt.tz_localize(None).dt.normalize().to_numpy()
    inr = (m >= 480) & (m < 570)
    g = pd.DataFrame(dict(h=np.where(inr, h, np.nan), l=np.where(inr, l, np.nan))).groupby(day, sort=False)
    rh, rl = g.h.transform("max").to_numpy(), g.l.transform("min").to_numpy()
    win = (m >= 570) & (m < 600) & np.isfinite(rh) & (rh > rl)
    brk = r5._first_of_group(win & ((c > rh) | (c < rl)), day)
    return run(p, cost, le=brk & (c > rh), se=brk & (c < rl), stop_l=rl, stop_s=rh, tgt_l=c + 2 * (c - rl), tgt_s=c - 2 * (rh - c), flat=m == 720)


def all_trades() -> tuple[dict, dict]:
    cost = r6.cost("XAUUSD")
    d1, h4 = r7.d1("XAUUSD"), g8._fp("gold/fp_XAUUSD_H4.csv")
    h1, m15 = lh.h1("XAUUSD")[0], g8._fp("gold/fp_XAUUSD_M15.csv")
    yd = (d1.time.iloc[-1] - d1.time.iloc[0]).days / 365.25
    over = lambda df, f: r7._cat([f(p) for p in fr.segments(df)])
    T = {
        "ema21_50_d1": (ema_cross(d1, cost, 21, 50), yd, "D1"),
        "ema9_21_d1": (ema_cross(d1, cost, 9, 21), yd, "D1"),
        "donchian20_d1": (donchian(d1, cost), yd, "D1"),
        "rsi_30_70_d1": (rsi_30_70(d1, cost), yd, "D1"),
        "stoch_h4": (over(h4, lambda p: stoch(p, cost)), g8._years(h4), "H4"),
        "donchian20_h4": (over(h4, lambda p: donchian(p, cost)), g8._years(h4), "H4"),
        "ema9_21_h1_long": (over(h1, lambda p: ema_cross(p, cost, 9, 21, True, 2.0)), g8._years(h1), "H1"),
        "bb_rsi_h1": (over(h1, lambda p: bb_rsi(p, cost)), g8._years(h1), "H1"),
        "rsi_ma_h1": (over(h1, lambda p: rsi_ma(p, cost)), g8._years(h1), "H1"),
        "pdh_pdl_sweep_h1": (over(h1, lambda p: pd_sweep(p, cost)), g8._years(h1), "H1"),
        "day_breakout_h1": (over(h1, lambda p: day_breakout(p, cost)), g8._years(h1), "H1"),
        "day_breakout_m15": (over(m15, lambda p: day_breakout(p, cost)), g8._years(m15), "M15"),
        "ny_orb_m15": (over(m15, lambda p: ny_orb(p, cost)), g8._years(m15), "M15"),
        "pdh_pdl_sweep_m15": (over(m15, lambda p: pd_sweep(p, cost)), g8._years(m15), "M15"),
    }
    assert len(T) == N_TESTS
    live = r6.live_trades("XAUUSD")
    bench = {"live_h1_rule": (live, g8._years(h1), "H1")}
    return T, bench


def summarise(t: pd.DataFrame, yrs: float, n_trials: int, chart: str) -> dict:
    r = r7.judge(t, yrs, n_trials)
    r.update(chart=chart, years=round(yrs, 1), trades_per_year=round(len(t) / yrs, 1))
    if len(t):
        a, b = t.R[t.entry < "2019-01-01"], t.R[t.entry >= "2019-01-01"]
        r["pre2019_R"] = round(float(a.mean()), 4) if len(a) >= 20 else None
        r["from2019_R"] = round(float(b.mean()), 4) if len(b) >= 20 else None
        e = t.sort_values("exit")
        cum = e.R.cumsum().to_numpy()
        k = max(1, len(e) // 300)
        r["curve"] = [[str(x.date()), round(float(y), 2)] for x, y in zip(e.exit.iloc[::k], cum[::k])] + [[str(e.exit.iloc[-1].date()), round(float(cum[-1]), 2)]]
    return r


def main() -> int:
    warnings.filterwarnings("ignore")
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + N_TESTS
    T, bench = all_trades()
    res = {k: summarise(t, y, n_trials, ch) for k, (t, y, ch) in T.items()}
    ben = {k: summarise(t.assign(cost_R=t.get("cost_R", 0.0)), y, n_trials, ch) for k, (t, y, ch) in bench.items()}
    # look-ahead: early trades must not change when later bars are added
    cost = r6.cost("XAUUSD")
    p = max(fr.segments(lh.h1("XAUUSD")[0]), key=len)
    na = min(len(p) - 3000, 30000)
    bad = []
    for name, f in dict(ema=lambda q: ema_cross(q, cost, 9, 21, True, 2.0), bb=lambda q: bb_rsi(q, cost), rsi_ma=lambda q: rsi_ma(q, cost),
                        sweep=lambda q: pd_sweep(q, cost), brk=lambda q: day_breakout(q, cost), donch=lambda q: donchian(q, cost)).items():
        x, y = f(p.iloc[:na].reset_index(drop=True)), f(p.iloc[:na + 3000].reset_index(drop=True))
        cut = p.time.iloc[na - 400]
        x, y = x[x.exit < cut], y[y.exit < cut]
        if len(x) != len(y) or not np.allclose(x.R.to_numpy(), y.R.to_numpy()):
            bad.append(name)
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "--ledger" in sys.argv:
        r3.append_ledger([dict(tested_utc=tested, name="src_" + k, params="{}", lookahead_ok=not bad, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes"))) for k, r in res.items()])
    (ROOT / "reports" / "gold_sources_retest.json").write_text(json.dumps(dict(meta=dict(n_trials=n_trials, generated=tested, lookahead_failed=bad),
                                                                              results=res, benchmark=ben), indent=1, default=str))
    rows = ["| idea | chart | years | trades/yr | win % | mean R | t | R/yr | before 2019 | from 2019 | last 2 y | slices | 2x cost | max DD (R) | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in {**res, **ben}.items():
        if "checks" not in r:
            rows.append(f"| {k} | {r['chart']} | {r['years']} | {r['trades_per_year']} | too few trades | | | | | | | | | | - |")
            continue
        rows.append(f"| {k} | {r['chart']} | {r['years']} | {r['trades_per_year']} | {round(100 * r['win_rate'], 1)} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | "
                    f"{r.get('pre2019_R')} | {r.get('from2019_R')} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {r['max_dd_R']} | "
                    f"{r['verdict'] if r['verdict'] != 'no' else '-'} |")
    md = f"# Gold: re-test of the strategies from blogs and forums\n\nN for the deflated Sharpe: {n_trials}. Look-ahead check failed for: {bad or 'none'}.\n\n" + "\n".join(rows) + "\n"
    (ROOT / "GOLD_SOURCES_RETEST.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
