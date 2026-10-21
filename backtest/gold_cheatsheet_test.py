"""
Owner's "XAUUSD Gold Movement Cheat Sheet" (2026-10-21): setups A / B / C on M15 entries with an H1 trend filter.

    python -m backtest.gold_cheatsheet_test

Data: FundedNext gold M5 2025-05-30 .. 2026-10-21 (MT5 copy_rates, spread per bar) built into M15 and H1; second source
the other broker's M5 2025-03-24 .. 2026-08-28.  Costs: the bar's own spread x 1.5 (slippage) + $6.5 / lot commission.
H1 trend (filter) = last COMPLETED hour: close above EMA50 and EMA50 above EMA200 (bullish); mirror bearish.
Session filter: entries only 07:00-20:00 UTC (London + New York), as the sheet suggests.

PRE-REGISTERED rules (written before the first run; nothing tuned afterwards)
  A breakout + retest   resistance = highest high of the previous 48 M15 bars (12 h).  An M15 close above it with a
                        bullish H1 trend arms the setup for 12 bars.  Retest = a later bar whose low comes within 0.1 ATR
                        of the level and closes above it as a bullish candle -> BUY next open.  Stop = lowest low since
                        the breakout - 0.1 ATR.  Mirror for sells.
  B liquidity sweep     levels = previous UTC day high / low and the Asian session (00:00-07:00 UTC) high / low.
                        Sweep = an M15 bar trades above a level and closes back below it.  Within the next 8 bars a close
                        below the sweep bar's low (structure break) with a bearish H1 trend -> SELL next open.  Stop =
                        sweep high + 0.1 ATR.  Mirror for buys.
  C trend continuation  bullish H1 trend; the M15 low touches the M15 EMA20 while staying above the 20-bar M15 swing low
                        (structure intact); confirmation = a bullish M15 candle closing above the previous bar's high
                        -> BUY next open.  Stop = the pullback low (lowest of the last 3 bars) - 0.1 ATR.  Mirror.
  Targets   fixed R-multiples 1.5R, 2R, 3R (the sheet's "predefined R-multiple"); time exit 32 M15 bars (8 h).
  Stops narrower than 0.5 ATR or wider than 6 ATR are skipped (strategy_round5 limits).
  Judge     60 / 40 split; pass = unseen mean R > 0 with t >= 2, > 0 at 2x costs, > 0 on the second broker.
Read-only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
COMM_PER_OZ = 0.065
TARGETS = (1.5, 2.0, 3.0)
HOLD = 32


def server_to_utc(t: pd.Series) -> pd.Series:
    """MT5 broker server time (FundedNext, Valetax) = New York time + 7 h (UTC+3 summer / UTC+2 winter) -> real UTC.
    Checked 2026-10-21 against Dukascopy UTC: median close difference $9.65 before, $0.27 after."""
    naive = t.dt.tz_localize(None) - pd.Timedelta(hours=7)
    return naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def load_m5(path: Path) -> pd.DataFrame:
    """Broker MT5 history (FundedNext / Valetax M1 or M5 exports); timestamps converted from server time to UTC."""
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], unit="s", utc=True) if pd.api.types.is_numeric_dtype(d["time"]) \
        else pd.to_datetime(d["time"], utc=True)
    d["time"] = server_to_utc(d["time"])
    d = d.dropna(subset=["time"])
    d["spr"] = d["spread"] / 100.0
    return d.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = d.groupby(d.time.dt.floor(rule))
    return pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                         "low": g.low.min().values, "close": g.close.last().values, "spr": g.spr.median().values})


def h1_trend(m15: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    c = h1.close
    side = np.where((c > c.ewm(span=50, adjust=False).mean()) & (c.ewm(span=50, adjust=False).mean() > c.ewm(span=200, adjust=False).mean()), 1,
                    np.where((c < c.ewm(span=50, adjust=False).mean()) & (c.ewm(span=50, adjust=False).mean() < c.ewm(span=200, adjust=False).mean()), -1, 0))
    s = pd.Series(side, index=h1.time + pd.Timedelta(hours=1))            # known once the hour is complete
    return s.reindex(m15.time, method="ffill").fillna(0).to_numpy()


def signals(m: pd.DataFrame, trend: np.ndarray) -> dict:
    o, h, l, c = (m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    A = base_atr(h, l, c, 14)
    hour = m.time.dt.hour.to_numpy()
    sess = (hour >= 7) & (hour < 20)
    out = {k: (np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)) for k in "ABC"}

    def put(k, i, d, stop):
        bl, sl, sb, ss = out[k]
        if d > 0:
            bl[i], sb[i] = True, stop
        else:
            sl[i], ss[i] = True, stop

    # A breakout + retest
    hi48, lo48 = r7.prev(r7.hi(h, 48)), r7.prev(r7.lo(l, 48))
    armed = None                                             # (direction, level, start index, extreme since)
    for i in range(50, n):
        if armed and i - armed[2] > 12:
            armed = None
        if armed:
            d, lev, st, ext = armed
            ext = min(ext, l[i]) if d > 0 else max(ext, h[i])
            armed = (d, lev, st, ext)
            if d > 0 and l[i] <= lev + 0.1 * A[i] and c[i] > lev and c[i] > o[i] and sess[i]:
                put("A", i, 1, ext - 0.1 * A[i]); armed = None
            elif d < 0 and h[i] >= lev - 0.1 * A[i] and c[i] < lev and c[i] < o[i] and sess[i]:
                put("A", i, -1, ext + 0.1 * A[i]); armed = None
            continue
        if c[i] > hi48[i] and trend[i] > 0:
            armed = (1, hi48[i], i, l[i])
        elif c[i] < lo48[i] and trend[i] < 0:
            armed = (-1, lo48[i], i, h[i])

    # B liquidity sweep -> structure break
    day = m.time.dt.normalize()
    dh = m.groupby(day).high.max().shift(1).reindex(day).to_numpy()
    dl = m.groupby(day).low.min().shift(1).reindex(day).to_numpy()
    asia = m[(hour < 7)].groupby(day[hour < 7])
    ah = asia.high.max().reindex(day).to_numpy()
    al = asia.low.min().reindex(day).to_numpy()
    ah = np.where(hour >= 7, ah, np.nan)
    al = np.where(hour >= 7, al, np.nan)
    pend = None                                              # (direction, sweep index, sweep extreme, trigger level)
    for i in range(1, n):
        if pend and i - pend[1] > 8:
            pend = None
        if pend:
            d, si, ext, trig = pend
            if d < 0 and c[i] < trig and trend[i] < 0 and sess[i]:
                put("B", i, -1, ext + 0.1 * A[i]); pend = None; continue
            if d > 0 and c[i] > trig and trend[i] > 0 and sess[i]:
                put("B", i, 1, ext - 0.1 * A[i]); pend = None; continue
        for lev in (dh[i], ah[i]):
            if np.isfinite(lev) and h[i] > lev and c[i] < lev:
                pend = (-1, i, h[i], l[i])
        for lev in (dl[i], al[i]):
            if np.isfinite(lev) and l[i] < lev and c[i] > lev:
                pend = (1, i, l[i], h[i])

    # C trend continuation
    e20 = r7.ema(c, 20)
    swl, swh = r7.prev(r7.lo(l, 20)), r7.prev(r7.hi(h, 20))
    pl, ph = r7.lo(l, 3), r7.hi(h, 3)
    touch_up = pd.Series(l <= e20).rolling(3).max().to_numpy() > 0
    touch_dn = pd.Series(h >= e20).rolling(3).max().to_numpy() > 0
    for i in range(25, n):
        if not sess[i]:
            continue
        if trend[i] > 0 and touch_up[i] and pl[i] > swl[i] and c[i] > o[i] and c[i] > h[i - 1]:
            put("C", i, 1, pl[i] - 0.1 * A[i])
        elif trend[i] < 0 and touch_dn[i] and ph[i] < swh[i] and c[i] < o[i] and c[i] < l[i - 1]:
            put("C", i, -1, ph[i] + 0.1 * A[i])
    return out


def trades(m: pd.DataFrame, trend: np.ndarray, cost_mult: float = 1.0, zero_cost: bool = False) -> dict:
    c = m.close.to_numpy(float)
    cost = dict(cost_frac=0.0 if zero_cost else (m.spr.median() * 1.5 * cost_mult + COMM_PER_OZ * cost_mult) / m.close.median(),
                swap_long=0.0, swap_short=0.0)
    res = {}
    for k, (bl, sl_, sb, ss) in signals(m, trend).items():
        for tp in TARGETS:
            stop = np.where(bl, sb, np.where(sl_, ss, np.nan))
            risk = np.abs(c - stop)
            tl, ts = c + tp * risk, c - tp * risk
            s = r5._put(r5._sig(len(c)), bl, sl_, sb, ss, tl, ts)
            res[(k, tp)] = r5.simulate_piece(m, dict(s, hold=HOLD), cost)
    return res


def run(path: Path) -> dict:
    d = load_m5(path)
    m15, h1 = resample(d, "15min"), resample(d, "1h")
    return {"m15": m15, "trend": h1_trend(m15, h1)}


def main() -> int:
    fn = run(ROOT / "data" / "gold" / "fn_XAUUSD_M5.csv")
    ob = run(ROOT / "data" / "xauusd_m5_5y.csv")
    cut = gs.split_at(fn["m15"])
    base, x2, zero = trades(fn["m15"], fn["trend"]), trades(fn["m15"], fn["trend"], 2.0), trades(fn["m15"], fn["trend"], zero_cost=True)
    alt = trades(ob["m15"], ob["trend"])
    cut_b = gs.split_at(ob["m15"])
    months = (fn["m15"].time.iloc[-1] - fn["m15"].time.iloc[0]).days / 30.44
    names = {"A": "A breakout + retest", "B": "B liquidity sweep", "C": "C trend continuation"}
    rows = []
    for (k, tp), t in base.items():
        d_, c_ = t[t.entry < cut], t[t.entry >= cut]
        a = alt[(k, tp)]
        ac = a[a.entry >= cut_b]
        z = zero[(k, tp)]
        rows.append({"setup": names[k], "target": f"{tp:g}R", "trades/month": round(len(t) / months, 1),
                     "win %": round(100 * float((t.R > 0).mean()), 1) if len(t) else None,
                     "no-cost R": round(float(z.R.mean()), 3) if len(z) else None,
                     "first 60 % R": round(float(d_.R.mean()), 3) if len(d_) else None,
                     "unseen R": round(float(c_.R.mean()), 3) if len(c_) else None,
                     "unseen t": round(float(c_.R.mean() / c_.R.std() * np.sqrt(len(c_))), 2) if len(c_) > 2 else None,
                     "2x cost unseen": round(float(x2[(k, tp)][x2[(k, tp)].entry >= cut].R.mean()), 3) if len(x2[(k, tp)]) else None,
                     "other broker unseen": round(float(ac.R.mean()), 3) if len(ac) else None,
                     "$/month @0.35% 15K": round(float(t.R.sum()) * 0.0035 * 15000 / months)})
    P = pd.DataFrame(rows)
    P["PASS"] = (P["unseen R"] > 0) & (P["unseen t"] >= 2) & (P["2x cost unseen"] > 0) & (P["other broker unseen"] > 0)
    print(f"FundedNext M15 {fn['m15'].time.iloc[0].date()} .. {fn['m15'].time.iloc[-1].date()}, cost "
          f"{100 * (fn['m15'].spr.median() * 1.5 + COMM_PER_OZ) / fn['m15'].close.median():.4f} % of price per trade")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(P.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
