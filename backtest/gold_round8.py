"""
GOLD ROUND 8 -- what gold traders do that rounds 1-7 had not tested.  Backtest only, on already-formed candles.

Already tested on gold (not repeated here): hourly momentum (live), Turtle / Donchian / Supertrend / 43 indicator
rules on H1 and D1, volume profile, Asian range breakout, liquidity sweeps, fair value gaps, fades (Bollinger,
VWAP, pivots), Ichimoku, daily volatility breakout, Asian-hours long, London-hours short, pre-FOMC, turn of
month, DXY and real-yield filters, COT, weekday / month effects (descriptive), stop / target grids.

Pre-registered ideas (fixed before the first run; sources in GOLD_ROUND8.md):

  Sessions (H1, Dukascopy 2005-)
    pmfix_long            buy 15:00 London (afternoon fix), close 08:00 London next morning       [Zurich study]
    asia_uptrend          the round-7 reopen-safe Asian rule (buy 19:00 New York, close 07:00 UTC),
                          only when yesterday's daily close is above its 200-day EMA
    ny_cont_1r / _run     first H1 close beyond the London range (07-12 UTC) during 13-16 UTC: follow, stop at the
                          far side of the range, target 1 x range / no target, flat 21:00 UTC      [popular]
  Round numbers (H1)
    round50_break / round100_break   close through a $50 / $100 level: follow, 2 ATR stop, 6 ATR target
    round50_fade  / round100_fade    touch of the level from >= 1 ATR away that closes back: fade, stop 1 ATR
                                     beyond the level, target 2 ATR
  Weekend gap (H1)
    gap_fade / gap_follow  first bar of the week opens >= 1 ATR from Friday's close; enter one bar later
  Other markets leading gold (H1, FundingPips 2010-, same broker clock)
    silver_lead / eur_lead  silver / EURUSD moved >= 2 sd in the hour while gold moved < 1 sd: follow in gold
  Slower chart
    h4_live / h4_live_up   the live rule on 4-hour candles (2004-), all trades / buys above the 200-day EMA only
  News (H1)
    nfp_follow / fomc_follow  after the announcement hour closes, follow its direction (to 21:00 UTC / 24 h)
  Daily lead-lag and calendar (D1 gold futures 2000-)
    tnx_fade      yesterday's 10-year yield up -> sell gold for one day, down -> buy
    silver_fade   fade yesterday's silver direction for one day
    gold_reversal fade yesterday's gold direction for one day
    january_long / friday_long
  Scalping (what most retail gold traders do)
    scalp_m5 / scalp_m15   EMA 9 / 21 cross with RSI 14 on its side, London and New York opening hours only,
                           1.5 ATR stop, 3 ATR target  (FundingPips M5 1.4 years, M15 4.2 years: too short to pass)

Judged with the round-7 rule (r7.judge): pass = >= 10 years, >= 100 trades, mean R > 0 on the whole sample, the
first 70 % and the last 30 %, >= 6 of 8 slices, > 0 at double cost, deflated Sharpe >= 0.95 with N = ledger + these.

    python -m backtest.gold_round8            # report only
    python -m backtest.gold_round8 --ledger   # also record the tests in research/ledger.csv
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
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INF = np.inf
N_TESTS = 23


# -- data ----------------------------------------------------------------------------------------
def _fp(path: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / path)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d[["time", "open", "high", "low", "close"]].dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _daily(path: str) -> pd.Series:
    d = pd.read_csv(DATA / path)
    return pd.Series(d["close"].to_numpy(float), index=pd.to_datetime(d["date"]).dt.normalize()).dropna()


def _years(df: pd.DataFrame) -> float:
    return sum((p.time.iloc[-1] - p.time.iloc[0]).days for p in fr.segments(df)) / 365.25


def _over(df: pd.DataFrame, f) -> pd.DataFrame:
    return r7._cat([f(p) for p in fr.segments(df)])


def _arr(p):
    return tuple(p[k].to_numpy(float) for k in ("open", "high", "low", "close"))


def _clock(p, tz=None):
    t = p["time"] if tz is None else p["time"].dt.tz_convert(tz)
    return t.dt.hour.to_numpy()


def _one(n, v=1):
    return np.full(n, v, np.int8)


# -- sessions ------------------------------------------------------------------------------------
def pmfix_long(p, cost):
    hl = _clock(p, "Europe/London")
    return r7.window(p, cost, hl == 15, _one(len(p)), hl == 8)


def asia_uptrend(p, cost):
    c = p["close"].to_numpy(float)
    day = r5.broker_day(p["time"])
    dclose = pd.Series(c).groupby(day, sort=False).last()
    ema200 = dclose.ewm(span=200, adjust=False).mean()
    ema200[:200] = np.nan
    prev = r5.prev_group(dict(c=dclose.to_numpy(), e=ema200.to_numpy()), day)
    up = prev["c"] > prev["e"]
    return r7.window(p, cost, (_clock(p, "America/New_York") == 19) & up, _one(len(p)), _clock(p) == 7)


def ny_cont(p, cost, target: bool):
    o, h, l, c = _arr(p)
    hr = _clock(p)
    day = p["time"].dt.normalize().to_numpy()
    inr = (hr >= 7) & (hr < 12)
    g = pd.DataFrame(dict(h=np.where(inr, h, np.nan), l=np.where(inr, l, np.nan))).groupby(day, sort=False)
    rh, rl = g.h.transform("max").to_numpy(), g.l.transform("min").to_numpy()        # read from 13:00 only
    win = (hr >= 13) & (hr <= 16) & np.isfinite(rh) & np.isfinite(rl)
    brk = r5._first_of_group(win & ((c > rh) | (c < rl)), day)
    rng = rh - rl
    s = r5._put(r5._sig(len(c)), brk & (c > rh), brk & (c < rl), rl, rh,
                c + rng if target else np.full(len(c), INF), c - rng if target else np.full(len(c), -INF))
    return r5.simulate_piece(p, dict(s, hold=10, flat=hr == 21), cost)


# -- round numbers ---------------------------------------------------------------------------------
def round_break(p, cost, step: float):
    o, h, l, c = _arr(p)
    A, pc = base_atr(h, l, c, 14), r5._prev(c)
    up_lvl, dn_lvl = (np.floor(pc / step) + 1) * step, (np.ceil(pc / step) - 1) * step
    s = r5._put(r5._sig(len(c)), c >= up_lvl, c <= dn_lvl, c - 2 * A, c + 2 * A, c + 6 * A, c - 6 * A)
    return r5.simulate_piece(p, dict(s, hold=96), cost)


def round_fade(p, cost, step: float):
    o, h, l, c = _arr(p)
    A, pc = base_atr(h, l, c, 14), r5._prev(c)
    up_lvl, dn_lvl = (np.floor(pc / step) + 1) * step, (np.ceil(pc / step) - 1) * step
    short = (h >= up_lvl) & (c < up_lvl) & (up_lvl - pc >= A)
    long = (l <= dn_lvl) & (c > dn_lvl) & (pc - dn_lvl >= A)
    s = r5._put(r5._sig(len(c)), long, short, dn_lvl - A, up_lvl + A, c + 2 * A, c - 2 * A)
    return r5.simulate_piece(p, dict(s, hold=24), cost)


# -- weekend gap -----------------------------------------------------------------------------------
def weekend_gap(p, cost, fade: bool):
    o, h, l, c = _arr(p)
    A = base_atr(h, l, c, 14)
    first = np.concatenate([[False], np.diff(p["time"].to_numpy()) > np.timedelta64(40, "h")])
    pc, pa = r5._prev(c), r5._prev(A)
    gap = o - pc
    big = first & (np.abs(gap) >= pa)                    # known when the first bar of the week has closed
    if fade:
        s = r5._put(r5._sig(len(c)), big & (gap < 0), big & (gap > 0), c - 2 * A, c + 2 * A, pc, pc)
    else:
        s = r5._put(r5._sig(len(c)), big & (gap > 0), big & (gap < 0), c - 2 * A, c + 2 * A, c + 2 * A, c - 2 * A)
    return r5.simulate_piece(p, dict(s, hold=24), cost)


# -- another market moves first ------------------------------------------------------------------
def lead(gold: pd.DataFrame, other: pd.DataFrame, cost) -> pd.DataFrame:
    m = gold.merge(other[["time", "close"]].rename(columns={"close": "x"}), on="time", how="inner")

    def f(p):
        o, h, l, c = _arr(p)
        A = base_atr(h, l, c, 14)
        z = lambda v: (np.log(v).diff() / np.log(v).diff().rolling(100).std().shift(1)).to_numpy()
        zx, zg = z(p["x"]), z(p["close"])
        ok = (np.abs(zx) >= 2) & (np.abs(zg) < 1)
        s = r5._put(r5._sig(len(c)), ok & (zx > 0), ok & (zx < 0), c - 2 * A, c + 2 * A, c + 2 * A, c - 2 * A)
        return r5.simulate_piece(p, dict(s, hold=24), cost)
    return _over(m, f)


# -- 4-hour chart ----------------------------------------------------------------------------------
def h4_live(p, cost, up_only: bool):
    o, h, l, c = _arr(p)
    A = base_atr(h, l, c, 14)
    buy, sell, _ = r7.rules(o, h, l, c)["live_momentum_rsi60"]
    buy, sell = np.nan_to_num(buy.astype(float)) > 0, np.nan_to_num(sell.astype(float)) > 0
    buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
    if up_only:
        buy, sell = buy & (c > r7.ema(c, 1200)), np.zeros(len(c), bool)             # 1200 x 4 h = 200 days
    s = r5._put(r5._sig(len(c)), buy, sell, c - 2 * A, c + 2 * A, c + 6 * A, c - 6 * A)
    return r5.simulate_piece(p, dict(s, hold=96), cost)


# -- news ------------------------------------------------------------------------------------------
def news_follow(p, cost, when: pd.DatetimeIndex, to_2100: bool):
    o, h, l, c = _arr(p)
    A = base_atr(h, l, c, 14)
    ev = pd.DatetimeIndex(p["time"]).isin(when)
    move = c - o
    ok = ev & (np.abs(move) >= 0.5 * A)
    s = r5._put(r5._sig(len(c)), ok & (move > 0), ok & (move < 0), c - 2 * A, c + 2 * A, np.full(len(c), INF), np.full(len(c), -INF))
    flat = (_clock(p) == 21) if to_2100 else np.zeros(len(c), bool)
    return r5.simulate_piece(p, dict(s, hold=12 if to_2100 else 24, flat=flat), cost)


# -- daily -----------------------------------------------------------------------------------------
def one_day(d: pd.DataFrame, cost, direction: np.ndarray) -> pd.DataFrame:
    """direction[i]: side to hold during day i (decided before its open); out at the next open."""
    direction = np.nan_to_num(direction).astype(np.int8)
    return r7.window(d, cost, direction != 0, direction, np.ones(len(d), bool), hold=3)


def daily_ideas(d: pd.DataFrame, cost) -> dict:
    date = d["time"].dt.tz_localize(None).dt.normalize()
    c = d["close"].to_numpy(float)
    chg = lambda s: np.sign(s.reindex(date).ffill().diff().to_numpy())                # change on day i (known at its close)
    y = r5._prev(chg(_daily("gold/yahoo_tnx_D1.csv")))                               # yesterday's change, for today
    sv = r5._prev(chg(_daily("gold/yahoo_silver_futures_D1.csv")))
    g = r5._prev(np.sign(np.diff(c, prepend=np.nan)))
    return {
        "tnx_fade": one_day(d, cost, -y),
        "silver_fade": one_day(d, cost, -sv),
        "gold_reversal": one_day(d, cost, -g),
        "january_long": one_day(d, cost, (date.dt.month == 1).to_numpy().astype(float)),
        "friday_long": one_day(d, cost, (date.dt.dayofweek == 4).to_numpy().astype(float)),
    }


# -- scalping ----------------------------------------------------------------------------------------
def scalp(p, cost):
    o, h, l, c = _arr(p)
    A = base_atr(h, l, c, 14)
    e9, e21, r14 = r7.ema(c, 9), r7.ema(c, 21), r7.rsi(c, 14)
    hr = _clock(p)
    sess = np.isin(hr, (7, 8, 9, 13, 14, 15))
    up, dn = r7.cross_up(e9, e21), r7.cross_up(e21, e9)
    s = r5._put(r5._sig(len(c)), sess & up & (r14 > 50), sess & dn & (r14 < 50), c - 1.5 * A, c + 1.5 * A, c + 3 * A, c - 3 * A)
    return r5.simulate_piece(p, dict(s, hold=48), cost)


# -- run -------------------------------------------------------------------------------------------
def run_all() -> tuple[dict, dict]:
    cost = r6.cost("XAUUSD")
    duka, _ = lh.h1("XAUUSD")
    fp_g, fp_s, fp_e = _fp("gold/fp_XAUUSD_H1.csv"), _fp("gold/fp_XAGUSD_H1.csv"), _fp("fp/EURUSD_H1.csv")
    h4, m5, m15 = _fp("gold/fp_XAUUSD_H4.csv"), _fp("gold/fp_XAUUSD_M5.csv"), _fp("gold/fp_XAUUSD_M15.csv")
    d = r7.d1("XAUUSD")
    ev = pd.read_csv(DATA / "events_history.csv")
    when = lambda k: pd.DatetimeIndex(pd.to_datetime(ev[ev.event == k].time_utc, utc=True)).floor("h")
    yh1, yfp = _years(duka), _years(fp_g)
    yd = (d.time.iloc[-1] - d.time.iloc[0]).days / 365.25

    T = {}                                                # name -> (trades, years, data)
    H = lambda name, f: T.__setitem__(name, (_over(duka, f), yh1, "H1 Dukascopy"))
    H("pmfix_long", lambda p: pmfix_long(p, cost))
    H("asia_uptrend", lambda p: asia_uptrend(p, cost))
    H("ny_cont_1r", lambda p: ny_cont(p, cost, True))
    H("ny_cont_run", lambda p: ny_cont(p, cost, False))
    for step in (50, 100):
        H(f"round{step}_break", lambda p, s=step: round_break(p, cost, s))
        H(f"round{step}_fade", lambda p, s=step: round_fade(p, cost, s))
    H("gap_fade", lambda p: weekend_gap(p, cost, True))
    H("gap_follow", lambda p: weekend_gap(p, cost, False))
    T["silver_lead"] = (lead(fp_g, fp_s, cost), yfp, "H1 FundingPips")
    T["eur_lead"] = (lead(fp_g, fp_e, cost), yfp, "H1 FundingPips")
    T["h4_live"] = (_over(h4, lambda p: h4_live(p, cost, False)), _years(h4), "H4 FundingPips")
    T["h4_live_up"] = (_over(h4, lambda p: h4_live(p, cost, True)), _years(h4), "H4 FundingPips")
    H("nfp_follow", lambda p: news_follow(p, cost, when("NFP"), True))
    H("fomc_follow", lambda p: news_follow(p, cost, when("FOMC"), False))
    for k, t in daily_ideas(d, cost).items():
        T[k] = (t, yd, "D1 futures")
    T["scalp_m5"] = (_over(m5, lambda p: scalp(p, cost)), _years(m5), "M5 FundingPips")
    T["scalp_m15"] = (_over(m15, lambda p: scalp(p, cost)), _years(m15), "M15 FundingPips")
    assert len(T) == N_TESTS, len(T)
    meta = dict(h1_from=str(duka.time.iloc[0].date()), h1_years=round(yh1, 1), fp_years=round(yfp, 1),
                h4_years=round(_years(h4), 1), d1_from=str(d.time.iloc[0].date()), d1_years=round(yd, 1),
                m5_years=round(_years(m5), 1), m15_years=round(_years(m15), 1), cost=cost)
    return T, meta


def lookahead_ok() -> list:
    """Trades entered in the first part must not change when later bars are added."""
    cost = r6.cost("XAUUSD")
    p = max(fr.segments(lh.h1("XAUUSD")[0]), key=len)
    na = min(len(p) - 3000, 30000)
    a, b = p.iloc[:na].reset_index(drop=True), p.iloc[:na + 3000].reset_index(drop=True)
    fs = dict(pmfix_long=lambda q: pmfix_long(q, cost), asia_uptrend=lambda q: asia_uptrend(q, cost),
              ny_cont_1r=lambda q: ny_cont(q, cost, True), round50_break=lambda q: round_break(q, cost, 50),
              round50_fade=lambda q: round_fade(q, cost, 50), gap_fade=lambda q: weekend_gap(q, cost, True),
              h4_live=lambda q: h4_live(q, cost, True), scalp=lambda q: scalp(q, cost))
    bad = []
    for k, f in fs.items():
        x, y = f(a), f(b)
        cut = a.time.iloc[na - 200]
        x, y = x[x.exit < cut].reset_index(drop=True), y[y.exit < cut].reset_index(drop=True)
        if len(x) != len(y) or not np.allclose(x.R.to_numpy(), y.R.to_numpy()):
            bad.append(k)
    return bad


def main() -> int:
    warnings.filterwarnings("ignore")
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + N_TESTS
    T, meta = run_all()
    bad = lookahead_ok()
    res = {}
    for k, (t, yrs, src) in T.items():
        r = r7.judge(t, yrs, n_trials)
        r.update(data=src, years=round(yrs, 1), trades_per_year=round(len(t) / yrs, 1))
        res[k] = r
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "--ledger" in sys.argv:
        r3.append_ledger([dict(tested_utc=tested, name="g8_" + k, params="{}", lookahead_ok=k not in bad, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                          for k, r in res.items()])
    rep = dict(meta=dict(meta, n_trials=n_trials, generated=tested, lookahead_failed=bad), results=res)
    (ROOT / "reports" / "gold_round8.json").write_text(json.dumps(rep, indent=2, default=str))
    rows = ["| idea | data | years | trades/yr | mean R | t | R/yr | first 70 % | last 30 % | last 2 y | slices | 2x cost | max DD (R) | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "checks" not in r:
            rows.append(f"| {k} | {r['data']} | {r['years']} | {r['trades_per_year']} | too few trades ({r.get('trades')}) | | | | | | | | | - |")
            continue
        rows.append(f"| {k} | {r['data']} | {r['years']} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | "
                    f"{r.get('first70_R')} | {r.get('last30_R')} | {r['last2y_R']} | {r.get('wf_positive')}/8 | {r.get('cost2x_R')} | "
                    f"{r['max_dd_R']} | {r['verdict'] if r['verdict'] != 'no' else '-'} |")
    md = "# Gold round 8\n\n" + f"N for the deflated Sharpe: {n_trials}. Look-ahead check failed for: {bad or 'none'}.\n\n" + "\n".join(rows) + "\n"
    (ROOT / "GOLD_ROUND8.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
