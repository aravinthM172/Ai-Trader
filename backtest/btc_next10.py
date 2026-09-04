from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from backtest import btc_engine_core as ec

DATA = ROOT / "data" / "btc_m5_history.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

print("=" * 90)
print("BTC — NEXT 10 RESEARCH STEPS")
print("=" * 90)

# ---------------------------------------------------------------------------
# 01. VERIFY ENGINE IMPORT
# ---------------------------------------------------------------------------
print("\n[01/10] ENGINE")
print("Version        :", ec.ENGINE_VERSION)
print("Tick size      :", ec.TICK_SIZE)
print("Tick value     :", ec.TICK_VALUE)
print("Value/price    :", ec.VALUE_PER_POINT)
print("Contract size  :", ec.CONTRACT_SIZE)
print("Min lot        :", ec.MIN_LOT)
print("Lot step       :", ec.LOT_STEP)
print("Spread floor   :", ec.DEFAULT_SPREAD)
print("Slippage       :", ec.SLIPPAGE)

# ---------------------------------------------------------------------------
# 02. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[02/10] DATA")
df = pd.read_csv(DATA)

o = np.ascontiguousarray(df.open.to_numpy(np.float64))
h = np.ascontiguousarray(df.high.to_numpy(np.float64))
l = np.ascontiguousarray(df.low.to_numpy(np.float64))
c = np.ascontiguousarray(df.close.to_numpy(np.float64))

spread = ec.prepare_spread(df)

print("Rows           :", len(df))
print("From           :", df.time.iloc[0])
print("To             :", df.time.iloc[-1])

# ---------------------------------------------------------------------------
# 03. DATA SANITY
# ---------------------------------------------------------------------------
print("\n[03/10] DATA SANITY")
bad = (
    (df.high < df.low) |
    (df.open < df.low) | (df.open > df.high) |
    (df.close < df.low) | (df.close > df.high)
)

print("NaN OHLC       :", int(df[["open","high","low","close"]].isna().any(axis=1).sum()))
print("Duplicate time :", int(df.time.duplicated().sum()))
print("Bad OHLC       :", int(bad.sum()))

if int(bad.sum()) > 0:
    raise RuntimeError("Bad OHLC data detected.")

# ---------------------------------------------------------------------------
# 04. SPREAD SANITY
# ---------------------------------------------------------------------------
print("\n[04/10] SPREAD")
print(pd.Series(spread).describe().to_string())
print("\nRaw spread points:")
print(df.spread.describe().to_string())

# ---------------------------------------------------------------------------
# 05. INDICATOR SANITY
# ---------------------------------------------------------------------------
print("\n[05/10] INDICATORS")

FAST, SLOW, TREND = 20, 60, 150
RSI_PERIOD, ATR_PERIOD = 20, 20
SL, TP = 2.0, 3.5

ef, es, et, rv, av, warmup = ec.compute_indicators(
    o, h, l, c,
    FAST, SLOW, TREND,
    RSI_PERIOD, ATR_PERIOD
)

print("Warmup        :", warmup)
print("EMA20 latest  :", float(ef[-1]))
print("EMA60 latest  :", float(es[-1]))
print("EMA150 latest :", float(et[-1]))
print("RSI latest    :", float(rv[-1]))
print("ATR latest    :", float(av[-1]))

# ---------------------------------------------------------------------------
# 06. FULL TRADE LOG
# ---------------------------------------------------------------------------
print("\n[06/10] TRADE SANITY")

res, trades = ec.run_window_logged(
    o, h, l, c,
    np.ascontiguousarray(ef),
    np.ascontiguousarray(es),
    np.ascontiguousarray(et),
    np.ascontiguousarray(rv),
    np.ascontiguousarray(av),
    spread,
    0, len(c), warmup,
    60.0, 40.0,
    SL, TP,
)

print("Trades        :", int(res[0]))
print("Net profit    :", float(res[2]))
print("Win rate      :", float(res[3]))
print("Profit factor :", float(res[4]))
print("Max DD        :", float(res[5]))

n = len(trades["pnl"])
print("Recorded      :", n)

if n == 0:
    raise RuntimeError("No trades recorded.")

x = pd.DataFrame(trades)

x["entry_time"] = pd.to_datetime(df.time.iloc[x.entry_i].to_numpy())
x["exit_time"] = pd.to_datetime(df.time.iloc[x.exit_i].to_numpy())

print("\nFIRST 15 TRADES")
print(
    x[
        [
            "entry_time","exit_time","dir","lots",
            "entry_px","exit_px","pnl"
        ]
    ].head(15).to_string(index=False)
)

# ---------------------------------------------------------------------------
# 07. PNL / LOT SANITY
# ---------------------------------------------------------------------------
print("\n[07/10] PNL / LOT SANITY")

print("\nLot statistics:")
print(x.lots.describe().to_string())

print("\nPnL statistics:")
print(x.pnl.describe().to_string())

wins = x[x.pnl > 0]
losses = x[x.pnl <= 0]

print("\nWins         :", len(wins))
print("Losses       :", len(losses))

if len(wins):
    print("Average win  :", float(wins.pnl.mean()))
if len(losses):
    print("Average loss :", float(losses.pnl.mean()))

print(
    "\nExpected BTC P/L identity:",
    "price_move × VALUE_PER_POINT × lots"
)

# ---------------------------------------------------------------------------
# 08. ENTRY / EXIT PRICE SANITY
# ---------------------------------------------------------------------------
print("\n[08/10] EXECUTION SANITY")

# Buy should enter above corresponding bar open;
# sell should enter below it.
entry_reference = df.open.iloc[x.entry_i].to_numpy()

buy_bad = (
    (x.dir.to_numpy() == 1) &
    (x.entry_px.to_numpy() < entry_reference)
).sum()

sell_bad = (
    (x.dir.to_numpy() == -1) &
    (x.entry_px.to_numpy() > entry_reference)
).sum()

print("Bad BUY entries :", int(buy_bad))
print("Bad SELL entries:", int(sell_bad))

if buy_bad or sell_bad:
    raise RuntimeError("Execution-side sanity check failed.")

# ---------------------------------------------------------------------------
# 09. SAVE SANITY REPORT
# ---------------------------------------------------------------------------
print("\n[09/10] SAVE")

summary = pd.DataFrame([{
    "symbol": "BTC",
    "rows": len(df),
    "warmup": warmup,
    "trades": int(res[0]),
    "profit": float(res[2]),
    "winrate": float(res[3]),
    "pf": float(res[4]),
    "dd": float(res[5]),
    "tick_size": ec.TICK_SIZE,
    "tick_value": ec.TICK_VALUE,
    "value_per_price": ec.VALUE_PER_POINT,
    "contract_size": ec.CONTRACT_SIZE,
    "min_lot": ec.MIN_LOT,
    "lot_step": ec.LOT_STEP,
    "spread_floor": ec.DEFAULT_SPREAD,
    "slippage": ec.SLIPPAGE,
}])

summary_path = REPORTS / "btc_engine_sanity.csv"
summary.to_csv(summary_path, index=False)

trade_path = REPORTS / "btc_baseline_trade_log.csv"
x.to_csv(trade_path, index=False)

print("Saved:", summary_path)
print("Saved:", trade_path)

# ---------------------------------------------------------------------------
# 10. SMALL OPTIMIZER SMOKE — ONLY IF SANITY PASSED
# ---------------------------------------------------------------------------
print("\n[10/10] BTC OPTIMIZER SMOKE")

configs = [
    (5,30,100,7,10,2.0,3.5),
    (10,50,150,10,14,2.0,3.5),
    (15,60,150,14,20,2.0,3.5),
    (20,60,150,20,20,2.0,3.5),
    (20,80,200,20,20,2.5,3.5),
    (30,80,200,20,20,2.5,3.5),
    (15,80,200,20,20,2.5,3.5),
    (30,50,150,20,20,2.0,3.5),
    (20,50,150,20,14,2.0,3.5),
    (30,60,150,20,20,2.0,3.5),
]

rows = []

for cfg in configs:
    fast, slow, trend, rp, ap, sl, tp = cfg

    ef2, es2, et2, rv2, av2, wu2 = ec.compute_indicators(
        o, h, l, c,
        fast, slow, trend, rp, ap
    )

    train_end = int(len(c) * 0.70)
    val_end = int(len(c) * 0.85)

    tr = ec.run_window_fast(
        o, h, l, c,
        np.ascontiguousarray(ef2),
        np.ascontiguousarray(es2),
        np.ascontiguousarray(et2),
        np.ascontiguousarray(rv2),
        np.ascontiguousarray(av2),
        spread,
        0, train_end, wu2,
        60.0, 40.0,
        sl, tp,
        ec.SLIPPAGE,
        ec.COMMISSION_PER_LOT,
    )

    va = ec.run_window_fast(
        o, h, l, c,
        np.ascontiguousarray(ef2),
        np.ascontiguousarray(es2),
        np.ascontiguousarray(et2),
        np.ascontiguousarray(rv2),
        np.ascontiguousarray(av2),
        spread,
        train_end, val_end, wu2,
        60.0, 40.0,
        sl, tp,
        ec.SLIPPAGE,
        ec.COMMISSION_PER_LOT,
    )

    rows.append({
        "ema_fast": fast,
        "ema_slow": slow,
        "ema_trend": trend,
        "rsi_period": rp,
        "atr_period": ap,
        "sl": sl,
        "tp": tp,
        "train_trades": int(tr[0]),
        "train_profit": float(tr[2]),
        "train_pf": float(tr[4]),
        "train_dd": float(tr[5]),
        "val_trades": int(va[0]),
        "val_profit": float(va[2]),
        "val_pf": float(va[4]),
        "val_dd": float(va[5]),
    })

smoke = pd.DataFrame(rows).sort_values(
    ["val_pf","val_profit"],
    ascending=False
)

smoke_path = REPORTS / "btc_optimizer_smoke.csv"
smoke.to_csv(smoke_path, index=False)

print("\n" + "=" * 90)
print("BTC OPTIMIZER SMOKE RESULTS")
print("=" * 90)
print(smoke.to_string(index=False))

print("\nSaved:", smoke_path)
print("\nBTC 10-STEP CHECK COMPLETE")
print("NO LIVE ORDERS WERE SENT.")
print("=" * 90)
