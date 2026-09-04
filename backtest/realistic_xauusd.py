"""
Realistic XAUUSD MT5 walk-forward backtest (TRAIN / VALIDATION / FINAL TEST).

This is a corrected rewrite. The previous version silently produced almost no
trades (3 / 0 / 0). Root causes and fixes are summarised at the bottom of this
file under "BUGFIXES".

Design rules enforced here:
  * Signal is decided from bar (i-1) data only; the order is filled at the
    OPEN of bar i. Nothing in the decision or the fill uses bar i's close/
    high/low.
  * Dollar P/L uses the MT5 profit identity
        profit = (exit - entry) / TICK_SIZE * TICK_VALUE * volume
    i.e. value_per_point = TICK_VALUE / TICK_SIZE.  CONTRACT_SIZE is used only
    for notional / margin, never for P/L.
  * Position size is rounded DOWN to the lot step. If the honest size is below
    the minimum lot the trade is SKIPPED - it is never forced to 0.01.
  * Costs: full spread (half on entry, half on exit), stop-order slippage,
    optional round-turn commission.
  * Indicators are computed ONCE over the whole series (they are strictly
    causal, so this is not look-ahead) and the three windows trade disjoint,
    chronological bar ranges - no data crosses a split boundary.

--------------------------------------------------------------------------------
NOTE (engine_core migration): the entire trade simulation - indicators, signal,
execution, sizing, risk, drawdown - now lives in ``backtest/engine_core.py`` and
is shared verbatim with ``optimizer_realistic_numba.py``.  This file keeps only
the training-chosen STRATEGY parameters and the CLI / CSV reporting.  Numerical
results are unchanged from the pre-migration standalone.
"""

import os
import argparse
import numpy as np
import pandas as pd

try:                                                # `python backtest/realistic_xauusd.py`
    import engine_core as ec
except ImportError:                                 # `python -m backtest.realistic_xauusd`
    from backtest import engine_core as ec


# ---------------------------------------------------------------------------
# FILES
# ---------------------------------------------------------------------------
DATA = ec.DATA
REPORT = "reports/xauusd_realistic_results.csv"


# ---------------------------------------------------------------------------
# ACCOUNT / CONTRACT / COST CONSTANTS  --  single source: engine_core.py
# (re-exported here so the print/report code below is unchanged)
# ---------------------------------------------------------------------------
INITIAL_BALANCE      = ec.INITIAL_BALANCE
RISK_PER_TRADE       = ec.RISK_PER_TRADE
EQUITY_STOP_FRACTION = ec.EQUITY_STOP_FRACTION
MAX_MARGIN_FRACTION  = ec.MAX_MARGIN_FRACTION
LEVERAGE             = ec.LEVERAGE

CONTRACT_SIZE        = ec.CONTRACT_SIZE
TICK_SIZE            = ec.TICK_SIZE
TICK_VALUE           = ec.TICK_VALUE
VALUE_PER_POINT      = ec.VALUE_PER_POINT

MIN_LOT              = ec.MIN_LOT
LOT_STEP             = ec.LOT_STEP
MAX_LOT              = ec.MAX_LOT

DEFAULT_SPREAD       = ec.DEFAULT_SPREAD
SLIPPAGE             = ec.SLIPPAGE
COMMISSION_PER_LOT   = ec.COMMISSION_PER_LOT
MIN_STOP_TICKS       = ec.MIN_STOP_TICKS


# ---------------------------------------------------------------------------
# STRATEGY PARAMS  (these were chosen on TRAINING data only)
# ---------------------------------------------------------------------------
FAST = 5
SLOW = 80
TREND = 200

RSI_PERIOD = 20
RSI_BUY = 60.0
RSI_SELL = 40.0

ATR_PERIOD = 20

SL_MULT = 2.5
TP_MULT = 3.5


# ===========================================================================
# ENGINE  --  imported verbatim from the shared source of truth
# ===========================================================================
# ema / rsi / atr / the full _backtest event loop used to be defined here.
# They now live in backtest/engine_core.py and are shared, unchanged, with
# optimizer_realistic_numba.py.  main() below calls ec.run_window_fast(),
# which wraps engine_core._backtest with the engine_core cost/risk constants.
from engine_core import (  # noqa: E402  (re-exported for callers/tests)
    ema, rsi, atr, compute_indicators, prepare_spread, load_ohlc,
    run_window_fast, run_window_logged, summary_dict,
)


# ===========================================================================
# DRIVER
# ===========================================================================
def _load():
    # Identical implementation, single source: engine_core.load_ohlc
    return load_ohlc(DATA)


def main():
    parser = argparse.ArgumentParser(
        description="Realistic XAUUSD MT5 backtest"
    )
    parser.add_argument("--fast", type=int, default=FAST)
    parser.add_argument("--slow", type=int, default=SLOW)
    parser.add_argument("--trend", type=int, default=TREND)
    parser.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    parser.add_argument("--rsi-buy", type=float, default=RSI_BUY)
    parser.add_argument("--rsi-sell", type=float, default=RSI_SELL)
    parser.add_argument("--atr-period", type=int, default=ATR_PERIOD)
    parser.add_argument("--sl", type=float, default=SL_MULT)
    parser.add_argument("--tp", type=float, default=TP_MULT)
    parser.add_argument("--spread", type=float, default=ec.DEFAULT_SPREAD,
                        help="Spread floor in dollars")
    parser.add_argument("--slippage", type=float, default=ec.SLIPPAGE,
                        help="Stop-fill slippage in dollars")

    args = parser.parse_args()

    ec.DEFAULT_SPREAD = args.spread
    ec.SLIPPAGE = args.slippage
    globals()["DEFAULT_SPREAD"] = args.spread
    globals()["SLIPPAGE"] = args.slippage

    fast = args.fast
    slow = args.slow
    trend = args.trend
    rsi_period = args.rsi_period
    rsi_buy = args.rsi_buy
    rsi_sell = args.rsi_sell
    atr_period = args.atr_period
    sl_mult = args.sl
    tp_mult = args.tp
    print("=" * 72)
    print("REALISTIC XAUUSD MT5 BACKTEST  (train / validation / final test)")
    print("=" * 72)

    df = _load()

    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    n = c.shape[0]

    # per-bar spread (points -> $, floored at DEFAULT_SPREAD); single source.
    sp = prepare_spread(df)

    # indicators: computed ONCE over the whole series (causal -> no look-ahead)
    ef, es, et, rv, av, warmup = compute_indicators(
        o, h, l, c, fast, slow, trend, rsi_period, atr_period
    )

    # 70 / 15 / 15 chronological, disjoint
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = [
        ("TRAINING", 0, train_end),
        ("VALIDATION", train_end, val_end),
        ("FINAL TEST", val_end, n),
    ]

    print()
    print(f"Candles          : {n:,}")
    if "time" in df.columns and df["time"].notna().any():
        print(f"Range            : {df['time'].iloc[0]}  ->  {df['time'].iloc[-1]}")
    print(f"Initial balance  : ${INITIAL_BALANCE:,.2f}")
    print(f"Risk / trade     : {RISK_PER_TRADE * 100:.2f}%")
    print(f"Spread floor     : ${DEFAULT_SPREAD:.2f}   slippage ${SLIPPAGE:.2f}"
          f"   commission ${COMMISSION_PER_LOT:.2f}/lot")
    print(f"Value per $1move : ${VALUE_PER_POINT:.2f} / lot   "
          f"(TICK_VALUE / TICK_SIZE)")
    if abs(VALUE_PER_POINT - CONTRACT_SIZE) > 1e-9:
        print(f"  NOTE: CONTRACT_SIZE={CONTRACT_SIZE:.0f} implies "
              f"${CONTRACT_SIZE:.2f}/lot per $1 move, but the tick spec implies "
              f"${VALUE_PER_POINT:.2f}. P/L follows the tick spec (MT5 formula). "
              f"If your broker's tick value is really $1.00, set TICK_VALUE=1.0.")
    print()
    print(f"Strategy         : EMA {fast}/{slow}/{trend}  |  "
          f"RSI {rsi_period} buy>={rsi_buy:.0f} sell<={rsi_sell:.0f}  |  "
          f"ATR {atr_period}  SL x{sl_mult}  TP x{tp_mult}")

    print()
    print("Compiling / warming up ...")
    run_window_fast(
        o, h, l, c, ef, es, et, rv, av, sp,
        0, min(2000, n), warmup,
        rsi_buy, rsi_sell, sl_mult, tp_mult,
        args.slippage,
    )
    print("Ready.")

    rows = []
    for name, i0, i1 in splits:
        res = run_window_fast(
            o, h, l, c, ef, es, et, rv, av, sp,
            i0, i1, warmup,
            rsi_buy, rsi_sell, sl_mult, tp_mult,
            args.slippage,
        )
        (trades, balance, profit, wr, pf,
         dd_abs, dd_pct, wins, losses, comm,
         _gross_profit, _gross_loss, _n_rec) = res

        trades = int(round(trades))
        wins = int(round(wins))
        losses = int(round(losses))

        print()
        print("=" * 72)
        print(f"{name}    bars [{i0:,} : {i1:,})   ({i1 - i0:,} candles)")
        if "time" in df.columns and df["time"].notna().any():
            print(f"  {df['time'].iloc[i0]}  ->  {df['time'].iloc[min(i1, n) - 1]}")
        print("=" * 72)
        print(f"  Trades            : {trades}")
        print(f"  Wins / Losses     : {wins} / {losses}")
        print(f"  Final balance     : ${balance:,.2f}")
        print(f"  Net profit        : ${profit:,.2f}  "
              f"({(balance / INITIAL_BALANCE - 1.0) * 100:.2f}%)")
        print(f"  Win rate          : {wr * 100:.2f}%")
        print(f"  Profit factor     : {pf:.3f}")
        print(f"  Max drawdown      : ${dd_abs:,.2f}  ({dd_pct * 100:.2f}%)")
        print(f"  Commission paid   : ${comm:,.2f}")

        rows.append({
            "period": name,
            "bar_start": i0,
            "bar_end": i1,
            "candles": i1 - i0,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "final_balance": round(balance, 2),
            "net_profit": round(profit, 2),
            "return_pct": round((balance / INITIAL_BALANCE - 1.0) * 100, 2),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4) if np.isfinite(pf) else "inf",
            "max_drawdown_abs": round(dd_abs, 2),
            "max_drawdown_pct": round(dd_pct * 100, 2),
            "commission_paid": round(comm, 2),
        })

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)

    print()
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()


# ===========================================================================
# BUGFIXES  (what was wrong in the previous version)
# ===========================================================================
#
#  1. POSITION SIZING
#     Old: raw_lots = risk / (stop_distance * CONTRACT_SIZE)  with CONTRACT_SIZE
#     = 100. Combined with a $100 account this made the honest lot size ~10x
#     larger than the account could take, so np.floor(...) collapsed it to 0
#     and every trade was skipped. Fixed to size off VALUE_PER_POINT
#     (= TICK_VALUE / TICK_SIZE = 10) and to include spread + slippage +
#     commission in the worst-case loss. Still rounds DOWN, still skips when
#     below MIN_LOT (never forces 0.01). Added a margin cap.
#
#  2. XAUUSD DOLLAR P/L
#     Old: pnl = (exit - entry) * CONTRACT_SIZE * lots  -> overstated every
#     P/L by 10x versus the supplied tick spec. MT5 actually computes
#     profit = (exit - entry) / TICK_SIZE * TICK_VALUE * volume. Fixed to
#     use VALUE_PER_POINT. CONTRACT_SIZE now only feeds the margin cap.
#
#  3. BID / ASK EXECUTION
#     Old: entry = close[i] (the signal bar's own close), same mid used for
#     buy and sell, and shorts were stopped/taken off the raw (bid) high/low
#     with no ask adjustment. Fixed: fill at open[i]; buy at ask (mid + s/2),
#     sell at bid (mid - s/2); long exits at bid, short exits at ask
#     (mid +/- s/2 on the high/low that triggers them).
#
#  4. SPREAD
#     Old: a single hard-coded 0.35 applied once. Fixed: per-bar spread from
#     the data's own spread column (points -> $), floored at the observed
#     0.35; half charged on entry and half on exit (full spread round trip);
#     plus explicit slippage on stop fills and optional commission.
#
#  5. ATR
#     Old: Wilder recursion seeded at 0.0, so ATR was far too small for the
#     first ~100 bars -> tiny stops -> distorted sizing (this is exactly why
#     TRAINING got a few trades near the seed region and the later windows
#     got none). Fixed: seed with the SMA of the first `period` true ranges,
#     standard Wilder smoothing thereafter.
#
#  6. EMA / RSI
#     EMA old: seeded with a single price. RSI old: seeded at 0 and, worse,
#     returned 50 whenever there were no losses in the window - so strong
#     up-legs never reached the RSI_BUY = 60.0 threshold and buy signals were
#     suppressed. Fixed: SMA seeding for EMA; RSI seeded from the first
#     `period` deltas and returns 100 (not 50) when there are no losses.
#
#  7. LOOK-AHEAD BIAS
#     Old: decided on bar i-1 but entered at bar i's close (a price not known
#     at decision time). Fixed: fill at bar i's OPEN. Also: no entry on the
#     same bar a position was just closed (old code could exit mid-bar i and
#     immediately re-enter at open[i], a price that preceded the exit).
#     Indicators are read at [i-1] only.
#
#  8. SL / TP EXECUTION
#     Old: always filled exactly at the SL/TP level, ignoring opening gaps,
#     and unconditionally counted every TP hit as a win. Fixed: stop orders
#     fill at min(sl, open)/max(sl, open) plus slippage (gap-aware, worse
#     fill); targets are limit orders and fill at the level; the stop is
#     checked before the target within a bar; win/loss is classified by the
#     sign of realised P/L.
#
#  9. DRAWDOWN
#     Old: measured only on realised balance (updated only when a trade
#     closed), so intratrade equity excursions were invisible, and reported
#     in $ only. Fixed: equity is marked to market every bar including the
#     open position; peak/drawdown tracked on equity; reported in $ and %.
#     Added a 20%-of-start equity kill-switch.
#
# 10. TRAIN / VALIDATION / TEST LEAKAGE
#     Old: indicators were recomputed from scratch on each sliced window,
#     which (a) gave every window a fresh un-converged warm-up and (b) still
#     shared the same over-fit hard-coded params. Fixed: indicators are
#     computed once over the full causal series and each window trades a
#     disjoint chronological bar range [i0, i1); no bar's data crosses a
#     boundary, and each window resets balance/peak. The params are marked
#     as training-chosen; re-optimise only on TRAINING and never touch
#     FINAL TEST until the end.
#
# 11. MINIMUM-LOT HANDLING
#     Kept the "skip the trade if the honest size < MIN_LOT" behaviour (it is
#     the conservative choice and was requested), but fixed the floating
#     point in the lot-step rounding (np.floor(x / step + 1e-9) * step) so
#     e.g. 0.03 does not collapse to 0.02, and added MAX_LOT + margin caps.
#
# 12. OTHER
#     * open[] is now actually read from the CSV (the old code never loaded it).
#     * profit_factor returns inf (not 0.0) when there are no losses.
#     * end-of-window open positions are closed and booked instead of vanishing.
#     * column names are normalised; missing columns raise clearly.
#     * numba is optional - a pure-python fallback keeps the file runnable.
#
# ===========================================================================






