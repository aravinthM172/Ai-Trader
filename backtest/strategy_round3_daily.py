"""
Strategy round 3 -- DAILY-chart strategies from the classic trading books, tested on
10-20+ years of history on every Valetax symbol that has a long public price record.

Rounds 1-2 tested H1 ideas on ~7 years of BTC/XAU.  This round asks a different question:
does a slower, book-documented edge exist on the daily chart, on ANY symbol we can trade,
over two decades -- and is it different enough from the live H1 momentum to add to it?

    python -m backtest.strategy_round3_daily            # download (cached) + test
    python -m backtest.strategy_round3_daily --offline  # cached data only

PRE-REGISTERED (written before the first run; book parameters, nothing tuned):

Candidates
  turtle55      Curtis Faith, "Way of the Turtle" (Dennis/Eckhardt System 2): stop-entry at the
                55-day high/low, 2N hard stop (N = 20-day ATR), exit at the 20-day opposite
                extreme.  Long + short.  No pyramiding.
  clenow_trend  Andreas Clenow, "Following the Trend": 50 EMA vs 100 EMA trend filter, enter on a
                50-day closing high (low), 3 x ATR(100) trailing stop from the best close.  L + S.
  tsmom12       Moskowitz-Ooi-Pedersen (2012) time-series momentum: at each month end hold the sign
                of the 12-month (252-bar) return; 3 x ATR(20) protective stop.  L + S.
  connors_rsi2  Larry Connors, "Short Term Trading Strategies That Work": long when close > SMA200
                and RSI(2) < 5, exit when close > SMA5; 3 x ATR(10) protective stop.  Long only.
                INDICES ONLY (the book's market).
  williams_vbo  Larry Williams, "Long-Term Secrets to Short-Term Trading": stop-entry at
                open +/- 0.6 x yesterday's range, stop at the opposite level, exit at the close.
                Day trade (no swap).  L + S.
  crabel_nr7    Toby Crabel, "Day Trading with Short Term Price Patterns": after an NR7 day, stop-
                entry at its high/low, stop at the other side, exit at the close.  Day trade.
  turn_of_month Lakonishok & Smidt (1988) / Ariel (1987): long from the close of the 2nd-to-last
                trading day of the month to the close of the 3rd trading day; 3 x ATR(20)
                protective stop.  Long only.  INDICES ONLY.

Universe: every Valetax symbol with a Yahoo daily proxy (cash index, FX spot, futures for
  energy/silver/platinum/palladium, BTC/ETH spot).  XAUUSD uses the broker's own D1 (2004-).
  XAUEUR / XAGEUR are synthetic (metal / EURUSD).  History from 2005-01-01.

Execution model (honest, no look-ahead):
  - Close-based signals fill at the NEXT bar's open.  Stop-entries fill at the level, or at the
    open if the market gaps through it.  Stops: open if gapped through, else the level.
  - Intraday ambiguity is always resolved against us: a bar that touches both the entry and the
    stop (or both breakout levels) is booked as a full stop-out.
  - Costs: broker spread (median recorded, floor 0.5 x current) scaled to each bar's price level,
    plus slippage 0.25 x spread per side.  Swap: broker's current long/short rate per night held.
  - R = P/L in units of the initial stop distance, after costs and swap.

PASS RULE per (strategy, symbol) -- ALL must hold:
  1. >= 10 years of data and >= 100 trades
  2. mean R > 0 on the full sample, on the first 70 % and on the last 30 % (time split)
  3. >= 6 of 8 equal time folds with mean R > 0
  4. mean R > 0 with DOUBLE costs
  5. deflated Sharpe (per trade) >= 0.95 with N = every idea in research/ledger.csv + 14 prior
     + every (strategy, symbol) pair tested in this round
  6. mean R > 0 on the broker's own D1 history (different data source, CFD session), when the
     broker has >= 3 years of it
USEFUL if it passes AND its daily R correlation with the live H1 momentum is < 0.5.

Added before the first run (after the data download and look-ahead check, before any result):
  - Data eligibility: a symbol is used only if it averages >= 200 bars a year since its start
    (Yahoo's PA=F / PL=F / SI=F futures series have long gaps).
  - POOLED test per strategy: the books trade these systems as diversified portfolios, and slow
    trend systems cannot reach 100 trades on one symbol in 20 years.  Each strategy's trades on
    ALL its eligible symbols are pooled (1 R each) and checked with the same rules 1-6
    (broker check = pooled broker trades).  Counted as 7 more trials.

Every pair is appended to research/ledger.csv, pass or fail, so N only grows.
Writes reports/strategy_round3.json and STRATEGY_ROUND3.md.  Read-only on MT5; no orders.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from backtest.multi_symbol_scan import deflated_sharpe, spec_of

log = get_logger("round3")
YCACHE = ROOT / "data" / "yahoo_D1"
BCACHE = ROOT / "data" / "mt5_D1"
H1CACHE = ROOT / "data" / "mt5_H1"
REPORTS = ROOT / "reports"
LEDGER = ROOT / "research" / "ledger.csv"
SPECS = REPORTS / "round3_specs.json"
START = "2005-01-01"
SLIP_FRAC = 0.25
MIN_YEARS, MIN_TRADES = 10.0, 100
MIN_BARS_PER_YEAR = 200
PRIOR_TRIALS = 14
INDEX_ONLY = {"connors_rsi2", "turn_of_month"}

YAHOO = {
    "DAX40.vx": "^GDAXI", "NAS100.vx": "^NDX", "SP500.vx": "^GSPC", "US30.vx": "^DJI", "JPN225.vx": "^N225",
    "HK50.vx": "^HSI", "UK100.vx": "^FTSE", "FRA40.vx": "^FCHI", "EU50.vx": "^STOXX50E", "AUS200.vx": "^AXJO",
    "XAGUSD.vx": "SI=F", "XPTUSD.vx": "PL=F", "XPDUSD.vx": "PA=F", "XTIUSD.vx": "CL=F", "XBRUSD.vx": "BZ=F",
    "XNGUSD.vx": "NG=F", "BTCUSD.vx": "BTC-USD", "ETHUSD.vx": "ETH-USD",
}
FX = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD",
      "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
      "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
YAHOO.update({f"{p}.vx": f"{p}=X" for p in FX})
SYNTHETIC = {"XAUEUR.vx": ("XAUUSD.vx", "EURUSD.vx"), "XAGEUR.vx": ("XAGUSD.vx", "EURUSD.vx")}
BROKER_PRIMARY = {"XAUUSD.vx"}


# -- data -------------------------------------------------------------------------
def _clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[~((d.high == d.low) & (d.open == d.close))]                       # flat / missing bars
    d = d.assign(high=d[["open", "high", "close"]].max(axis=1), low=d[["open", "low", "close"]].min(axis=1))
    rng = d.high - d.low
    med = rng.rolling(20, min_periods=5).median().shift(1)
    d = d[~(rng > 8 * med)]                                                  # bad ticks
    return d.loc[d.index >= START].copy()


def yahoo(ticker: str, offline: bool) -> pd.DataFrame | None:
    YCACHE.mkdir(parents=True, exist_ok=True)
    p = YCACHE / f"{ticker.replace('^', '_').replace('=', '_')}.csv"
    if not p.exists() and not offline:
        import yfinance as yf
        d = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)[["open", "high", "low", "close"]]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d.to_csv(p, index_label="date")
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["date"], index_col="date")
    return _clean(d)


def broker_d1(gw, symbol: str) -> pd.DataFrame | None:
    BCACHE.mkdir(parents=True, exist_ok=True)
    p = BCACHE / f"{symbol}.csv"
    if not p.exists() and gw is not None:
        d = gw.get_rates(symbol, "D1", 9000)
        if d is None or d.empty:
            return None
        d.to_csv(p, index=False)
    if not p.exists():
        return None
    d = pd.read_csv(p)
    # D1 bars open at broker midnight (UTC+2/+3) = 21:00/22:00 UTC the day before
    d["date"] = (pd.to_datetime(d["time"], utc=True) + pd.Timedelta(hours=6)).dt.tz_convert(None).dt.normalize()
    return _clean(d.set_index("date")[["open", "high", "low", "close"]])


def synthetic(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    j = num.join(den, how="inner", rsuffix="_d")
    out = pd.DataFrame(index=j.index)
    out["open"] = j.open / j.open_d
    out["close"] = j.close / j.close_d
    out["high"] = np.maximum(j.high / j.close_d, out[["open", "close"]].max(axis=1))
    out["low"] = np.minimum(j.low / j.close_d, out[["open", "close"]].min(axis=1))
    return out


def broker_spread_frac(sp: dict) -> float:
    """Spread as a fraction of price (median recorded H1 spread, floor 0.5 x current)."""
    spread = sp["spread_now"]
    p = H1CACHE / f"{sp['symbol']}.csv"
    if p.exists():
        h = pd.read_csv(p, usecols=["close", "spread"])
        if h["spread"].median() > 0:
            spread = max(float(h["spread"].median()) * sp["point"], 0.5 * sp["spread_now"])
        return spread / float(h["close"].iloc[-1])
    return spread / sp["price"]


# -- indicators (value at bar i uses bars <= i) -----------------------------------
def _wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def atr(h, l, c, n):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return _wilder(tr, n)


def rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    up, dn = _wilder(np.where(d > 0, d, 0.0), n), _wilder(np.where(d < 0, -d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + up / dn)


def _roll(x, n, f):
    return getattr(pd.Series(x).rolling(n), f)().to_numpy()


def _shift1(x):
    return np.r_[np.nan, x[:-1]]


# -- strategies: each returns a list of (entry_i, exit_i, dir, entry_px, exit_px, risk_px) --
def _stop_hit(d, o, l, h, stop):
    """Exit price if today's bar hits the stop (gap -> open), else None."""
    if d > 0:
        return o if o <= stop else (stop if l <= stop else None)
    return o if o >= stop else (stop if h >= stop else None)


def s_turtle55(o, h, l, c, dates, state=None):
    n = len(c)
    N = atr(h, l, c, 20)
    hi55, lo55 = _shift1(_roll(h, 55, "max")), _shift1(_roll(l, 55, "min"))
    hi20, lo20 = _shift1(_roll(h, 20, "max")), _shift1(_roll(l, 20, "min"))
    out, pos = [], 0
    for i in range(56, n):
        if pos:
            lvl = max(hard, lo20[i]) if pos > 0 else min(hard, hi20[i])
            x = _stop_hit(pos, o[i], l[i], h[i], lvl)
            if x is not None:
                out.append((ei, i, pos, ep, x, risk)); pos = 0
            continue
        if np.isnan(N[i - 1]) or np.isnan(hi55[i]):
            continue
        up, dn = h[i] >= hi55[i], l[i] <= lo55[i]
        if up and dn:
            continue
        if up or dn:
            pos = 1 if up else -1
            ep = max(o[i], hi55[i]) if up else min(o[i], lo55[i])
            risk, ei = 2 * N[i - 1], i
            hard = ep - pos * risk
            lvl = max(hard, lo20[i]) if pos > 0 else min(hard, hi20[i])
            if (pos > 0 and l[i] <= lvl) or (pos < 0 and h[i] >= lvl):      # same-bar stop: assume worst
                out.append((ei, i, pos, ep, lvl, risk)); pos = 0
    if state is not None:                          # orders for the NEXT bar (paper trading)
        if pos:
            state.update(pos=pos, entry_i=ei, entry_px=ep, risk=risk,
                         stop=max(hard, l[-20:].min()) if pos > 0 else min(hard, h[-20:].max()))
        else:
            state.update(pos=0, buy_stop=float(h[-55:].max()), sell_stop=float(l[-55:].min()), N=float(N[-1]))
    return out


def s_clenow_trend(o, h, l, c, dates, state=None):
    n = len(c)
    e50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    e100 = pd.Series(c).ewm(span=100, adjust=False).mean().to_numpy()
    A = atr(h, l, c, 100)
    hc, lc = _roll(c, 50, "max"), _roll(c, 50, "min")
    out, pos, pend = [], 0, 0
    for i in range(101, n):
        if pend and not pos:                       # entry signalled at yesterday's close
            pos, ep, ei = pend, o[i], i
            risk, best = 3 * A[i - 1], o[i]
            pend = 0
        elif pend and pos:                         # exit signalled at yesterday's close
            out.append((ei, i, pos, ep, o[i], risk)); pos, pend = 0, 0
        if pos:
            best = max(best, c[i]) if pos > 0 else min(best, c[i])
            if (pos > 0 and c[i] < best - 3 * A[i]) or (pos < 0 and c[i] > best + 3 * A[i]):
                pend = pos
        elif np.isfinite(A[i]):
            if e50[i] > e100[i] and c[i] >= hc[i]:
                pend = 1
            elif e50[i] < e100[i] and c[i] <= lc[i]:
                pend = -1
    if state is not None:
        if pos:
            state.update(pos=pos, entry_i=ei, entry_px=ep, risk=risk, exit_next_open=bool(pend),
                         trail_close=best - 3 * A[-1] if pos > 0 else best + 3 * A[-1])
        else:
            state.update(pos=0, enter_next_open=int(pend), risk_if_entered=float(3 * A[-1]))
    return out


def _month_end(dates):
    m = pd.DatetimeIndex(dates).month
    return np.r_[m[1:] != m[:-1], False]


def s_tsmom12(o, h, l, c, dates):
    n = len(c)
    A = atr(h, l, c, 20)
    me = _month_end(dates)
    out, pos, pend = [], 0, 0
    for i in range(253, n):
        if pend:
            if pos:
                out.append((ei, i, pos, ep, o[i], risk)); pos = 0
            pos, ep, ei, risk = pend, o[i], i, 3 * A[i - 1]
            stop, pend = ep - pend * risk, 0
        if pos:
            x = _stop_hit(pos, o[i], l[i], h[i], stop)
            if x is not None:
                out.append((ei, i, pos, ep, x, risk)); pos = 0
        if me[i]:
            sig = int(np.sign(c[i] / c[i - 252] - 1))
            if sig and sig != pos:
                pend = sig
    return out


def s_connors_rsi2(o, h, l, c, dates, state=None):
    n = len(c)
    r2, s200, s5, A = rsi(c, 2), _roll(c, 200, "mean"), _roll(c, 5, "mean"), atr(h, l, c, 10)
    out, pos, pend = [], 0, 0
    for i in range(201, n):
        if pend == 1 and not pos:
            pos, ep, ei, risk = 1, o[i], i, 3 * A[i - 1]
            stop, pend = ep - risk, 0
        elif pend == -1 and pos:
            out.append((ei, i, pos, ep, o[i], risk)); pos, pend = 0, 0
        if pos:
            x = _stop_hit(1, o[i], l[i], h[i], stop)
            if x is not None:
                out.append((ei, i, pos, ep, x, risk)); pos = 0
            elif c[i] > s5[i]:
                pend = -1
        elif c[i] > s200[i] and r2[i] < 5:
            pend = 1
    if state is not None:
        if pos:
            state.update(pos=1, entry_i=ei, entry_px=ep, risk=risk, stop=stop, exit_next_open=pend == -1)
        else:
            state.update(pos=0, enter_next_open=int(pend == 1), risk_if_entered=float(3 * A[-1]),
                         rsi2=float(r2[-1]), above_sma200=bool(c[-1] > s200[-1]))
    return out


def _range_breakout_day(i, o, h, l, c, up, dn, out):
    """Stop-entry at up/dn on bar i, stop at the opposite level, exit at the close."""
    hit_up, hit_dn = h[i] >= up, l[i] <= dn
    if not (hit_up or hit_dn):
        return
    if hit_up and hit_dn:                          # both levels touched: book as a full stop-out
        ep = max(o[i], up)
        out.append((i, i, 1, ep, dn, ep - dn)); return
    d = 1 if hit_up else -1
    ep = max(o[i], up) if d > 0 else min(o[i], dn)
    stop = dn if d > 0 else up
    risk = abs(ep - stop)
    if risk > 0:
        out.append((i, i, d, ep, c[i], risk))


def s_williams_vbo(o, h, l, c, dates):
    out = []
    for i in range(2, len(c)):
        rng = h[i - 1] - l[i - 1]
        if rng > 0:
            _range_breakout_day(i, o, h, l, c, o[i] + 0.6 * rng, o[i] - 0.6 * rng, out)
    return out


def s_crabel_nr7(o, h, l, c, dates):
    rng = h - l
    nr7 = rng <= _roll(rng, 7, "min")
    out = []
    for i in range(8, len(c)):
        if nr7[i - 1] and rng[i - 1] > 0:
            _range_breakout_day(i, o, h, l, c, h[i - 1], l[i - 1], out)
    return out


def s_turn_of_month(o, h, l, c, dates):
    """Enter at the close of day m-1 (m = last trading day of the month), exit at the close of m+3."""
    n = len(c)
    A = atr(h, l, c, 20)
    me = _month_end(dates)
    out = []
    for m in np.flatnonzero(me):
        ei, xi = m - 1, m + 3
        if ei < 21 or xi >= n or not np.isfinite(A[ei]):
            continue
        ep, risk = c[ei], 3 * A[ei]
        stop, exit_i, exit_px = ep - risk, xi, c[xi]
        for j in range(ei + 1, xi + 1):
            x = _stop_hit(1, o[j], l[j], h[j], stop)
            if x is not None:
                exit_i, exit_px = j, x
                break
        out.append((ei, exit_i, 1, ep, exit_px, risk))
    return out


STRATEGIES = {
    "turtle55": s_turtle55, "clenow_trend": s_clenow_trend, "tsmom12": s_tsmom12,
    "connors_rsi2": s_connors_rsi2, "williams_vbo": s_williams_vbo, "crabel_nr7": s_crabel_nr7,
    "turn_of_month": s_turn_of_month,
}


# -- evaluation ----------------------------------------------------------------------
def trade_table(d: pd.DataFrame, fn, spread_frac: float, swap_long: float, swap_short: float,
                cost_mult: float = 1.0, state: dict | None = None) -> pd.DataFrame:
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    dates = d.index
    t = fn(o, h, l, c, dates) if state is None else fn(o, h, l, c, dates, state=state)
    if not t:
        return pd.DataFrame(columns=["entry", "exit", "dir", "R"])
    a = np.array(t, dtype=float)
    ei, xi, dr, ep, xp, rk = a.T
    ei, xi = ei.astype(int), xi.astype(int)
    cost = cost_mult * spread_frac * ep * (1 + 2 * SLIP_FRAC)
    nights = (dates[xi] - dates[ei]).days.to_numpy()
    rate = np.where(dr > 0, swap_long or 0.0, swap_short or 0.0)
    swap = ep * rate / 365.0 * nights
    R = (dr * (xp - ep) - cost + swap) / rk
    return pd.DataFrame(dict(entry=dates[ei], exit=dates[xi], dir=dr.astype(int), R=R))


def evaluate(t: pd.DataFrame, t2: pd.DataFrame, years: float, n_trials: int, broker_R: float | None) -> dict:
    R = t["R"].to_numpy()
    if len(R) < 30:
        return dict(trades=len(R), passes=False, note="too few trades")
    t0, t1 = t.entry.min(), t.entry.max()
    cut = t0 + (t1 - t0) * 0.7
    first, last = R[t.entry < cut], R[t.entry >= cut]
    edges = pd.date_range(t0, t1, periods=9)
    folds = [R[(t.entry >= a) & (t.entry <= b)] for a, b in zip(edges[:-1], edges[1:])]
    wf = sum(1 for f in folds if len(f) and f.mean() > 0)
    dsr = deflated_sharpe(R, n_trials)
    yr = t.assign(y=t.exit.dt.year).groupby("y").R.sum()
    checks = dict(years_10=years >= MIN_YEARS, trades_100=len(R) >= MIN_TRADES, full_pos=R.mean() > 0,
                  first70_pos=len(first) > 0 and first.mean() > 0, last30_pos=len(last) > 0 and last.mean() > 0,
                  wf_6of8=wf >= 6, cost2x_pos=len(t2) > 0 and t2.R.mean() > 0, dsr_95=dsr >= 0.95,
                  broker_pos=broker_R is None or broker_R > 0)
    wins, losses = R[R > 0].sum(), -R[R < 0].sum()
    return dict(years=round(years, 1), trades=int(len(R)), trades_per_year=round(len(R) / max(years, 1e-9), 1),
                mean_R=round(float(R.mean()), 4), win_rate=round(float((R > 0).mean()), 3),
                profit_factor=round(float(wins / losses), 3) if losses > 0 else None,
                first70_R=round(float(first.mean()), 4) if len(first) else None,
                last30_R=round(float(last.mean()), 4) if len(last) else None, wf_positive=wf,
                cost2x_R=round(float(t2.R.mean()), 4) if len(t2) else None, dsr=round(dsr, 3),
                broker_R=None if broker_R is None else round(broker_R, 4),
                years_positive=f"{int((yr > 0).sum())}/{len(yr)}", total_R=round(float(R.sum()), 1),
                checks=checks, passes=all(checks.values()))


def live_daily_R() -> pd.Series:
    r = pd.read_csv(REPORTS / "multi_reference_trades.csv")
    d = pd.to_datetime(r.exit_time, utc=True).dt.tz_convert(None).dt.normalize()
    return r.assign(d=d).groupby("d").r_multiple.sum()


def ledger_n() -> int:
    if not LEDGER.exists():
        return 0
    with LEDGER.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def append_ledger(rows: list[dict]) -> None:
    with LEDGER.open(newline="", encoding="utf-8") as f:
        fields = csv.DictReader(f).fieldnames
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_specs(offline: bool):
    gw = None
    if not offline:
        from mt5.gateway import MT5Gateway
        gw = MT5Gateway()
        if not gw.connect():
            raise SystemExit("MT5 connection failed (broker specs/swaps come from the terminal)")
        mt5 = gw.raw()
        specs = {}
        for s in mt5.symbols_get() or []:
            if s.trade_mode != 0:
                sp = spec_of(s)
                sp["price"] = float(s.bid or s.last or 0.0)
                specs[s.name] = sp
        SPECS.write_text(json.dumps(specs, indent=2))
    return gw, json.loads(SPECS.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    gw, specs = load_specs(a.offline)
    try:
        data, broker = {}, {}
        for sym in sorted(set(YAHOO) | BROKER_PRIMARY):
            if sym not in specs:
                continue
            broker[sym] = broker_d1(gw, sym)
            data[sym] = broker[sym] if sym in BROKER_PRIMARY else yahoo(YAHOO[sym], a.offline)
        for sym, (num, den) in SYNTHETIC.items():
            broker[sym] = broker_d1(gw, sym)
            if data.get(num) is not None and data.get(den) is not None:
                data[sym] = synthetic(data[num], data[den])
        cross = {"XAUUSD.vx": yahoo("GC=F", a.offline)}       # gold: Yahoo futures is the 2nd source
        for sym, d in list(data.items()):
            if d is not None and len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9) < MIN_BARS_PER_YEAR:
                log.info("%s excluded: %d bars since %s (gappy data)", sym, len(d), d.index[0].date())
                data[sym] = None
    finally:
        if gw is not None:
            gw.shutdown()

    pairs = [(st, sym) for st in STRATEGIES for sym in data
             if data[sym] is not None and (st not in INDEX_ONLY or (specs[sym]["path"] or "").startswith("Indexes"))]
    n_trials = ledger_n() + PRIOR_TRIALS + len(pairs) + len(STRATEGIES)
    log.info("%d symbols with data, %d (strategy, symbol) pairs, deflated-Sharpe N=%d", len(data), len(pairs), n_trials)
    live = live_daily_R()

    res, daily, pool = {}, {}, {}
    for st, sym in pairs:
        d, sp = data[sym], specs[sym]
        fn = STRATEGIES[st]
        sf = broker_spread_frac(sp)
        t = trade_table(d, fn, sf, sp["swap_long"], sp["swap_short"])
        t2 = trade_table(d, fn, sf, sp["swap_long"], sp["swap_short"], cost_mult=2.0)
        b = cross.get(sym) if sym in BROKER_PRIMARY else broker.get(sym)
        bR = None
        if b is not None and (b.index[-1] - b.index[0]).days >= 3 * 365:
            tb = trade_table(b, fn, sf, sp["swap_long"], sp["swap_short"])
            bR = float(tb.R.mean()) if len(tb) else None
        years = (d.index[-1] - d.index[0]).days / 365.25
        r = evaluate(t, t2, years, n_trials, bR)
        p = pool.setdefault(st, dict(t=[], t2=[], b=[], years=0.0))
        p["t"].append(t); p["t2"].append(t2); p["years"] = max(p["years"], years)
        if bR is not None:
            p["b"].append(tb.R)
        r["group"] = (sp["path"] or "").split("\\")[0]
        r["spread_pct"] = round(100 * sf, 4)
        if len(t):
            dr = t.groupby(t.exit.dt.normalize()).R.sum()
            j = pd.concat([dr, live], axis=1, join="inner").fillna(0.0)
            r["corr_live"] = round(float(j.corr().iloc[0, 1]), 3) if len(j) > 50 else None
            r["useful"] = bool(r.get("passes") and (r["corr_live"] is None or r["corr_live"] < 0.5))
            daily[f"{st}|{sym}"] = dr
        res.setdefault(st, {})[sym] = r
        log.info("%-14s %-11s trades=%s meanR=%s last30=%s WF=%s/8 DSR=%s PASS=%s", st, sym, r.get("trades"),
                 r.get("mean_R"), r.get("last30_R"), r.get("wf_positive"), r.get("dsr"), r.get("passes"))

    pooled = {}
    for st, p in pool.items():
        t = pd.concat([x for x in p["t"] if len(x)], ignore_index=True)
        t2 = pd.concat([x for x in p["t2"] if len(x)], ignore_index=True)
        bR = float(pd.concat(p["b"]).mean()) if p["b"] else None
        r = evaluate(t, t2, p["years"], n_trials, bR)
        dr = t.groupby(t.exit.dt.normalize()).R.sum()
        j = pd.concat([dr, live], axis=1, join="inner").fillna(0.0)
        r["corr_live"] = round(float(j.corr().iloc[0, 1]), 3)
        r["symbols"] = len(p["t"])
        r["useful"] = bool(r["passes"] and r["corr_live"] < 0.5)
        pooled[st] = r
        log.info("POOLED %-14s trades=%s meanR=%s last30=%s WF=%s/8 DSR=%s PASS=%s", st, r.get("trades"),
                 r.get("mean_R"), r.get("last30_R"), r.get("wf_positive"), r.get("dsr"), r.get("passes"))

    passers = [f"{st}|{sym}" for st in res for sym, r in res[st].items() if r.get("passes")]
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    append_ledger([dict(tested_utc=tested, name=f"r3_{st}", params=json.dumps({"symbol": sym, "tf": "D1"}),
                        lookahead_ok=True, xau_exp_R=r.get("mean_R") if sym == "XAUUSD.vx" else "",
                        corr_with_live=r.get("corr_live", ""), n_trials=n_trials, deflated_sharpe=r.get("dsr", ""),
                        passes=bool(r.get("passes")), useful=bool(r.get("useful")))
                   for st in res for sym, r in res[st].items()]
                  + [dict(tested_utc=tested, name=f"r3_{st}_pooled", params=json.dumps({"symbols": r["symbols"], "tf": "D1"}),
                          lookahead_ok=True, corr_with_live=r.get("corr_live", ""), n_trials=n_trials,
                          deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("useful")))
                     for st, r in pooled.items()])
    corr = pd.concat({k: daily[k] for k in passers}, axis=1).fillna(0.0).corr().round(2) if len(passers) > 1 else None
    rep = dict(meta=dict(start=START, n_symbols=len(data), n_pairs=len(pairs), n_trials=n_trials,
                         rule="pre-registered in module docstring", runtime_s=round(time.perf_counter() - t0, 1)),
               passers=passers, pooled=pooled, correlation=None if corr is None else corr.to_dict(), results=res)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "strategy_round3.json").write_text(json.dumps(rep, indent=2, default=str))
    (ROOT / "STRATEGY_ROUND3.md").write_text(_md(rep), encoding="utf-8")
    print(_md(rep))
    return 0


def _md(rep) -> str:
    m = rep["meta"]
    L = ["# Strategy round 3 -- classic book strategies on the DAILY chart, 2005-today", "",
         f"{m['n_symbols']} symbols, {m['n_pairs']} (strategy, symbol) pairs, deflated-Sharpe N = {m['n_trials']}. "
         "Broker spread + slippage + swap. Pass rule pre-registered in `backtest/strategy_round3_daily.py`.", "",
         "## Pooled per strategy (all eligible symbols, 1 R per trade)", "",
         "| strategy | symbols | trades/yr | mean R | win % | PF | first 70 % | last 30 % | WF+ | 2x cost | broker D1 | yrs + | DSR | corr live | PASS |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for st, r in rep["pooled"].items():
        L.append(f"| {st} | {r['symbols']} | {r['trades_per_year']} | {r['mean_R']} | {100 * r['win_rate']:.0f} | "
                 f"{r['profit_factor']} | {r['first70_R']} | {r['last30_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | "
                 f"{r['broker_R']} | {r['years_positive']} | {r['dsr']} | {r['corr_live']} | "
                 f"{'**yes**' if r['passes'] else 'no'} |")
    L += ["", "## Summary per strategy", "",
         "| strategy | symbols tested | mean R (avg over symbols) | symbols with mean R > 0 | PASS |",
         "|---|--:|--:|--:|--:|"]
    for st, rs in rep["results"].items():
        v = [r["mean_R"] for r in rs.values() if r.get("mean_R") is not None]
        L.append(f"| {st} | {len(rs)} | {np.mean(v):.4f} | {sum(x > 0 for x in v)} | "
                 f"{sum(1 for r in rs.values() if r.get('passes'))} |")
    rows = [(st, sym, r) for st, rs in rep["results"].items() for sym, r in rs.items() if r.get("mean_R") is not None]
    rows.sort(key=lambda x: -(x[2].get("dsr") or 0) * 10 - x[2]["mean_R"])
    L += ["", "## Top 30 pairs (by deflated Sharpe, then mean R)", "",
          "| strategy | symbol | years | trades/yr | mean R | win % | PF | first 70 % | last 30 % | WF+ | 2x cost | broker D1 | yrs + | DSR | corr live | PASS |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for st, sym, r in rows[:30]:
        L.append(f"| {st} | {sym} | {r['years']} | {r['trades_per_year']} | {r['mean_R']} | {100 * r['win_rate']:.0f} | "
                 f"{r['profit_factor']} | {r['first70_R']} | {r['last30_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | "
                 f"{r['broker_R']} | {r['years_positive']} | {r['dsr']} | {r.get('corr_live')} | "
                 f"{'**yes**' if r['passes'] else 'no'} |")
    L += ["", f"**Passing pairs ({len(rep['passers'])}):** {', '.join(rep['passers']) or 'none'}", ""]
    near = [(st, sym, r) for st, sym, r in rows if not r["passes"]
            and sum(not v for v in r.get("checks", {}).values()) == 1]
    if near:
        L += ["Failed exactly one check:", ""]
        for st, sym, r in near[:15]:
            L.append(f"- {st} / {sym}: {', '.join(k for k, v in r['checks'].items() if not v)}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
