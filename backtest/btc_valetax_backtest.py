"""
BTCUSD.vx (Valetax) backtest -- uses the REAL broker characteristics.

  * downloads live BTCUSD.vx history from MT5 (M5 ~6 months, chunked)
  * contract 1.0, tick 0.01, $1 P/L per $1 move per 1.0 lot, volume 0.01 step
  * broker minimum stop = 2976 points = $29.76  (enforced on every SL/TP)
  * spread from the historical 'spread' column (points -> $), floored at a
    configurable minimum, plus per-side slippage
  * strategy = the same causal EMA/RSI/trend signal as strategy/btc_signal.py
    (NOT optimised -- this measures whether it is even viable after real costs)
  * strictly causal: signal from bar i-1, fill at open[i]
  * chronological TRAIN / VALIDATION / FINAL split + rolling walk-forward
  * reports: trades, win rate, PF, expectancy ($ and R), max DD, avg win/loss,
    Sharpe/Sortino (per-trade), avg hold, largest loss, max consec losses,
    exposure, and the trading-cost impact

    python -m backtest.btc_valetax_backtest [--download] [--bars 50000]

NO LIVE ORDER IS SENT.
"""
from __future__ import annotations

import argparse
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
from config.assets import btc_config
from strategy.btc_features import _atr, _ema, _rsi

log = get_logger("btc.backtest")

DATA_CSV = ROOT / "data" / "btcusd_vx_m5.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# real Valetax BTCUSD.vx contract
VALUE_PER_UNIT_PER_LOT = 1.0        # $ per $1 move per 1.0 lot
VOLUME_MIN = 0.01
VOLUME_STEP = 0.01
VOLUME_MAX = 50.0
BROKER_STOPS_LEVEL = 2976 * 0.01   # $29.76
INITIAL_BALANCE = 100.0
RISK_PER_TRADE = 0.01
MAX_RISK_PER_TRADE = 0.02
EQUITY_STOP_FRAC = 0.30
SLIPPAGE = 1.0                      # $ per side (conservative for a $78k instrument)
# Valetax reports a FIXED $29.76 spread in BTCUSD.vx M5 history (2976 pts on
# 49969/50000 bars).  Floor at that -- never below the broker's own value.
DEFAULT_SPREAD_FLOOR = 29.76

FAST, SLOW, TREND = 20, 50, 200
RSI_P, ATR_P = 14, 14
SL_ATR, TP_ATR = 2.0, 3.0
MIN_RR = 1.3


def download(bars: int) -> pd.DataFrame:
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        raise SystemExit("MT5 connection failed")
    try:
        df = gw.get_rates("BTCUSD.vx", "M5", bars)
        if df is None or df.empty:
            raise SystemExit("no history returned")
        df.to_csv(DATA_CSV, index=False)
        log.info("downloaded %d BTCUSD.vx M5 bars -> %s (%s -> %s)",
                 len(df), DATA_CSV.name, df["time"].iloc[0], df["time"].iloc[-1])
        return df
    finally:
        gw.shutdown()


def load() -> pd.DataFrame:
    if not DATA_CSV.exists():
        return download(50000)
    df = pd.read_csv(DATA_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _sim(o, h, l, c, spread, ef, es, et, rv, av, i0, i1, *, warmup):
    """Causal EMA/RSI trend strategy with real Valetax costs. One position at a time."""
    bal = INITIAL_BALANCE
    peak = bal
    eq_stop = INITIAL_BALANCE * EQUITY_STOP_FRAC
    pos = 0
    entry = sl = tp = lots = 0.0
    ent_i = 0
    trades = []
    bars_in_pos = 0
    start = max(i0, warmup, 1)
    for i in range(start, i1):
        hs = spread[i] * 0.5
        # manage
        if pos != 0:
            bars_in_pos += 1
            ex = None
            if pos == 1:
                if o[i] - hs <= sl: ex = o[i] - hs - SLIPPAGE
                elif l[i] - hs <= sl: ex = sl - SLIPPAGE
                elif o[i] - hs >= tp: ex = o[i] - hs
                elif h[i] - hs >= tp: ex = tp
            else:
                if o[i] + hs >= sl: ex = o[i] + hs + SLIPPAGE
                elif h[i] + hs >= sl: ex = sl + SLIPPAGE
                elif o[i] + hs <= tp: ex = o[i] + hs
                elif l[i] + hs <= tp: ex = tp
            if ex is None and i == i1 - 1:
                ex = (c[i] - hs - SLIPPAGE) if pos == 1 else (c[i] + hs + SLIPPAGE)
            if ex is not None:
                pnl = (ex - entry) * pos * VALUE_PER_UNIT_PER_LOT * lots
                bal += pnl
                trades.append(dict(entry_i=ent_i, exit_i=i, dir=pos, lots=lots,
                                   entry=entry, exit=ex, pnl=pnl,
                                   risk=abs(entry - sl) * VALUE_PER_UNIT_PER_LOT * lots,
                                   bars=bars_in_pos))
                pos = 0
        eq = bal
        peak = max(peak, eq)
        if eq <= eq_stop:
            break
        if pos != 0:
            continue
        # signal from i-1
        a = av[i - 1]
        if not (a > 0) or np.isnan(ef[i - 1]) or np.isnan(es[i - 1]) or np.isnan(et[i - 1]):
            continue
        up = ef[i - 1] > es[i - 1] > et[i - 1] and rv[i - 1] >= 55
        dn = ef[i - 1] < es[i - 1] < et[i - 1] and rv[i - 1] <= 45
        if not (up or dn):
            continue
        d = 1 if up else -1
        entry_fill = o[i] + hs if d == 1 else o[i] - hs
        ref = o[i] - hs if d == 1 else o[i] + hs
        sl_ref = max(a * SL_ATR, BROKER_STOPS_LEVEL * 1.15)
        sl_px = ref - d * sl_ref
        risk_px = abs(entry_fill - sl_px)
        tp_px = entry_fill + d * max(a * TP_ATR, risk_px * MIN_RR, BROKER_STOPS_LEVEL * 1.15)
        rr = abs(tp_px - entry_fill) / risk_px if risk_px > 0 else 0
        if rr < MIN_RR:
            continue
        if (spread[i] + 2 * SLIPPAGE) / risk_px > 0.5:      # cost dominates
            continue
        # sizing
        risk_d = bal * RISK_PER_TRADE
        loss_per_lot = risk_px * VALUE_PER_UNIT_PER_LOT
        raw = risk_d / loss_per_lot if loss_per_lot > 0 else 0
        v = np.floor(raw / VOLUME_STEP + 1e-9) * VOLUME_STEP
        if v < VOLUME_MIN:
            if VOLUME_MIN * loss_per_lot / bal <= MAX_RISK_PER_TRADE:
                v = VOLUME_MIN
            else:
                continue
        v = min(v, VOLUME_MAX)
        pos, entry, sl, tp, lots, ent_i, bars_in_pos = d, entry_fill, sl_px, tp_px, v, i, 0
    return bal, trades, start


def metrics(bal, trades, n_bars, tag):
    if not trades:
        return dict(split=tag, trades=0, final_balance=round(bal, 2), note="no trades")
    df = pd.DataFrame(trades)
    pnl = df["pnl"].to_numpy()
    R = np.where(df["risk"].to_numpy() > 0, pnl / df["risk"].to_numpy(), 0.0)
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    eq = INITIAL_BALANCE + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[INITIAL_BALANCE], eq]))
    dd = peak - np.concatenate([[INITIAL_BALANCE], eq])
    # consec losses
    cl = mx = 0
    for x in pnl:
        cl = 0 if x > 0 else cl + 1
        mx = max(mx, cl)
    downside = R[R < 0]
    gross_cost = float((df["bars"].sum() * 0))  # cost already inside pnl; report spread impact separately
    spread_cost = float(len(df) * (DEFAULT_SPREAD_FLOOR + 2 * SLIPPAGE) * VALUE_PER_UNIT_PER_LOT * df["lots"].mean())
    return dict(
        split=tag,
        trades=int(len(df)),
        final_balance=round(float(bal), 2),
        net_pl=round(float(pnl.sum()), 2),
        return_pct=round(100.0 * (bal - INITIAL_BALANCE) / INITIAL_BALANCE, 2),
        win_rate=round(float((pnl > 0).mean()), 4),
        profit_factor=round(float(wins.sum() / -losses.sum()), 4) if losses.sum() < 0 else None,
        expectancy_dollar=round(float(pnl.mean()), 4),
        expectancy_R=round(float(R.mean()), 4),
        avg_win=round(float(wins.mean()), 4) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 4) if len(losses) else 0.0,
        largest_loss=round(float(pnl.min()), 4),
        max_consecutive_losses=int(mx),
        max_drawdown_abs=round(float(dd.max()), 2),
        max_drawdown_pct=round(float((dd / peak).max()) * 100, 2),
        sharpe_per_trade=round(float(R.mean() / R.std(ddof=1)), 3) if len(R) > 1 and R.std(ddof=1) > 0 else None,
        sortino_per_trade=round(float(R.mean() / downside.std(ddof=1)), 3) if len(downside) > 1 and downside.std(ddof=1) > 0 else None,
        avg_holding_bars=round(float(df["bars"].mean()), 1),
        exposure_frac=round(float(df["bars"].sum() / max(1, n_bars)), 4),
        est_total_spread_slippage_cost=round(spread_cost, 2),
        cost_vs_gross_profit=round(spread_cost / max(1e-9, wins.sum()), 3) if len(wins) else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--bars", type=int, default=50000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = download(args.bars) if args.download else load()
    o = df.open.to_numpy(float); h = df.high.to_numpy(float)
    l = df.low.to_numpy(float); c = df.close.to_numpy(float)
    N = len(c)
    if "spread" in df.columns:
        sp = np.maximum(df["spread"].to_numpy(float) * 0.01, DEFAULT_SPREAD_FLOOR)
    else:
        sp = np.full(N, DEFAULT_SPREAD_FLOOR)

    ef = _ema(c, FAST); es = _ema(c, SLOW); et = _ema(c, TREND)
    rv = _rsi(c, RSI_P); av, _ = _atr(h, l, c, ATR_P)
    warmup = max(FAST, SLOW, TREND, RSI_P, ATR_P) + 5

    T1, V1 = int(N * 0.70), int(N * 0.85)
    splits = [("TRAIN", 0, T1), ("VALIDATION", T1, V1), ("FINAL", V1, N)]

    out = {"meta": dict(rows=int(N), span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
                        symbol="BTCUSD.vx", broker="Valetax",
                        contract=dict(value_per_unit_per_lot=VALUE_PER_UNIT_PER_LOT,
                                      volume_min=VOLUME_MIN, broker_stops_level=BROKER_STOPS_LEVEL),
                        costs=dict(spread_floor=DEFAULT_SPREAD_FLOOR, slippage_per_side=SLIPPAGE),
                        strategy=dict(ema=[FAST, SLOW, TREND], rsi=RSI_P, atr=ATR_P,
                                      sl_atr=SL_ATR, tp_atr=TP_ATR, min_rr=MIN_RR),
                        note="NOT optimised; measures viability of a simple trend rule after real costs")}
    all_tr = []
    for tag, i0, i1 in splits:
        bal, tr, start = _sim(o, h, l, c, sp, ef, es, et, rv, av, i0, i1, warmup=warmup)
        out[tag] = metrics(bal, tr, i1 - start, tag)
        all_tr += tr
        m = out[tag]
        log.info("[%s] trades=%s net=%s pf=%s expR=%s dd%%=%s", tag, m.get("trades"),
                 m.get("net_pl"), m.get("profit_factor"), m.get("expectancy_R"),
                 m.get("max_drawdown_pct"))

    # rolling walk-forward: 8 folds, train context ignored (rule is fixed) -> just OOS chunks
    fold = N // 10
    wf = []
    for k in range(2, 10):
        a, b = k * fold, (k + 1) * fold
        bal, tr, start = _sim(o, h, l, c, sp, ef, es, et, rv, av, a, b, warmup=warmup)
        wf.append(metrics(bal, tr, b - start, f"WF{k}"))
    out["walk_forward"] = wf
    wf_expR = [w["expectancy_R"] for w in wf if w.get("trades", 0) > 0]
    out["walk_forward_summary"] = dict(
        folds=len(wf), folds_with_trades=len(wf_expR),
        mean_expectancy_R=round(float(np.mean(wf_expR)), 4) if wf_expR else None,
        folds_positive=int(sum(1 for x in wf_expR if x > 0)),
    )

    comb = metrics(INITIAL_BALANCE + sum(t["pnl"] for t in all_tr), all_tr, N, "COMBINED")
    out["COMBINED"] = comb
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)

    (REPORTS / "btc_valetax_backtest.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 88)
    print("BTCUSD.vx (Valetax) BACKTEST  --  real contract + real costs  --  NO LIVE ORDER")
    print("=" * 88)
    print(f"rows {N}  {df.time.iloc[0]} -> {df.time.iloc[-1]}  ({(df.time.iloc[-1]-df.time.iloc[0]).days} days)")
    print(f"contract: $1/$1move/lot  vol step {VOLUME_STEP}  broker min stop ${BROKER_STOPS_LEVEL:.2f}")
    print(f"costs: spread floor ${DEFAULT_SPREAD_FLOOR}  slippage ${SLIPPAGE}/side  |  strategy NOT optimised")
    print("-" * 88)
    for tag in ("TRAIN", "VALIDATION", "FINAL", "COMBINED"):
        m = out[tag]
        print(f"{tag:11s} trades {m.get('trades',0):4d}  net ${m.get('net_pl',0):+9.2f} "
              f"({m.get('return_pct',0):+.1f}%)  win {m.get('win_rate',0)*100:5.1f}%  "
              f"PF {m.get('profit_factor')}  expR {m.get('expectancy_R')}  "
              f"maxDD {m.get('max_drawdown_pct')}%  hold {m.get('avg_holding_bars')}b  "
              f"maxConsecL {m.get('max_consecutive_losses')}")
    print("-" * 88)
    ws = out["walk_forward_summary"]
    print(f"walk-forward: {ws['folds_with_trades']}/{ws['folds']} folds traded, "
          f"mean expR {ws['mean_expectancy_R']}, {ws['folds_positive']} positive")
    c0 = out["COMBINED"]
    if c0.get("trades", 0) > 0:
        print(f"cost impact  : est spread+slippage ${c0.get('est_total_spread_slippage_cost')} "
              f"vs gross profit -> ratio {c0.get('cost_vs_gross_profit')}")
    print("=" * 88)
    print("report: reports/btc_valetax_backtest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
