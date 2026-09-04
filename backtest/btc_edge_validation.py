"""
btc_edge_validation.py  --  BTC-ONLY edge-validation research harness.

PURPOSE
    Decide whether the current BTC M5 baseline strategy
        EMA 20/60/150 | RSI 20 | ATR 20 | SL 2.0*ATR | TP 3.5*ATR
    has genuine predictive / structural edge, or whether the poor baseline
    (TRAIN PF 0.71, VAL PF 0.63, FINAL PF 1.28) is indistinguishable from a
    random-direction strategy under the same execution / cost model.

SAFETY / SCOPE
    * Reads backtest/btc_engine_core.py + backtest/btc_realistic_config.py for
      constants and cross-checks against ec.run_window_logged.
    * Does NOT modify any engine, config or XAU file.
    * No MT5 / live-order code anywhere.
    * Strictly causal: every signal uses bar i-1 information, fills at open[i];
      identical to the production engine loop. No look-ahead introduced.
    * Not an optimizer: brackets / costs are swept for DIAGNOSIS only, the
      verdict is scored on TRAIN+VALIDATION+FINAL together, never FINAL alone.

OUTPUTS
    reports/btc_edge_validation.csv
    reports/btc_edge_validation.json
    reports/btc_edge_validation_summary.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import btc_engine_core as ec          # noqa: E402
from backtest import btc_realistic_config as bc     # noqa: E402

try:
    from numba import njit
    NUMBA = True
except Exception:                                   # pragma: no cover
    NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _d(f):
            return f

        return _d


T0 = time.perf_counter()


def _lap(msg):
    print(f"[{time.perf_counter() - T0:7.2f}s] {msg}", flush=True)


# ===========================================================================
# CONFIG  (all pulled from the live BTC engine / config -- nothing invented)
# ===========================================================================
DATA = ROOT / "data" / "btc_m5_history.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

FAST, SLOW, TREND = 20, 60, 150
RSI_P, ATR_P = 20, 20
SL_MULT, TP_MULT = 2.0, 3.5
RSI_BUY, RSI_SELL = float(bc.RSI_BUY_TH), float(bc.RSI_SELL_TH)

INITIAL_BALANCE = float(ec.INITIAL_BALANCE)
RISK_PER_TRADE = float(ec.RISK_PER_TRADE)
VALUE_PER_POINT = float(ec.VALUE_PER_POINT)
CONTRACT_SIZE = float(ec.CONTRACT_SIZE)
LEVERAGE = float(ec.LEVERAGE)
MIN_LOT = float(ec.MIN_LOT)
LOT_STEP = float(ec.LOT_STEP)
MAX_LOT = float(ec.MAX_LOT)
SLIPPAGE = float(bc.SLIPPAGE)
COMMISSION_PER_LOT = float(ec.COMMISSION_PER_LOT)
MIN_STOP_DIST = float(ec.MIN_STOP_DIST)
EQUITY_STOP = float(ec.EQUITY_STOP)
MAX_MARGIN_FRAC = float(ec.MAX_MARGIN_FRACTION)
DEFAULT_SPREAD = float(bc.DEFAULT_SPREAD)
CLAMP_MULT = 5.0                                    # engine upper clamp = 5 * DEFAULT_SPREAD

N_PERM = 1000
PERM_SEED_BASE = 20260830

SPLIT_FRACS = (0.70, 0.85)


# ===========================================================================
# NUMBA KERNEL  --  faithful port of btc_engine_core._backtest (lines 275-542)
#   signal_mode : 0 real strategy
#                 1 strategy TIMING, coin-flip direction (np.random, seeded)
#                 2 strategy timing, forced LONG
#                 3 strategy timing, forced SHORT
#                 4 external entry mask + external direction array
# ===========================================================================
@njit(cache=True)
def edge_kernel(
    open_, high, low, close, ef, es, et, rv, av, spread,
    i0, i1, warmup,
    rsi_buy, rsi_sell, sl_mult, tp_mult,
    initial_balance, risk_per_trade, value_per_point, contract_size, leverage,
    min_lot, lot_step, max_lot, slippage, commission_per_lot, min_stop_dist,
    equity_stop, max_margin_frac,
    signal_mode, rng_seed, entry_mask, entry_dir_arr,
    o_entry_i, o_exit_i, o_dir, o_lots, o_entry_px, o_exit_px, o_pnl, o_risk,
    o_atr, o_rsi, o_emasep,
):
    if signal_mode == 1:
        np.random.seed(rng_seed)

    balance = initial_balance
    peak_equity = initial_balance
    max_dd_abs = 0.0

    position = 0
    entry_fill = 0.0
    entry_idx = 0
    sl = 0.0
    tp = 0.0
    lots = 0.0
    risk_now = 0.0
    atr_now = 0.0
    rsi_now = 0.0
    emasep_now = 0.0

    n = 0
    trades = 0
    wins = 0
    gp = 0.0
    gl = 0.0

    start = i0 if i0 > warmup else warmup
    if start < 1:
        start = 1

    for i in range(start, i1):
        hs = spread[i] * 0.5
        closed = False

        # ---- manage open position -------------------------------------
        if position == 1:
            hit = False
            ex = 0.0
            if open_[i] - hs <= sl:
                ex = open_[i] - hs - slippage
                hit = True
            elif low[i] - hs <= sl:
                ex = sl - slippage
                hit = True
            elif open_[i] - hs >= tp:
                ex = open_[i] - hs
                hit = True
            elif high[i] - hs >= tp:
                ex = tp
                hit = True
            if hit:
                pnl = (ex - entry_fill) * value_per_point * lots - commission_per_lot * lots
                balance += pnl
                trades += 1
                if pnl > 0.0:
                    wins += 1
                    gp += pnl
                else:
                    gl -= pnl
                if n < o_pnl.shape[0]:
                    o_entry_i[n] = entry_idx; o_exit_i[n] = i; o_dir[n] = 1
                    o_lots[n] = lots; o_entry_px[n] = entry_fill; o_exit_px[n] = ex
                    o_pnl[n] = pnl; o_risk[n] = risk_now
                    o_atr[n] = atr_now; o_rsi[n] = rsi_now; o_emasep[n] = emasep_now
                    n += 1
                position = 0
                closed = True
        elif position == -1:
            hit = False
            ex = 0.0
            if open_[i] + hs >= sl:
                ex = open_[i] + hs + slippage
                hit = True
            elif high[i] + hs >= sl:
                ex = sl + slippage
                hit = True
            elif open_[i] + hs <= tp:
                ex = open_[i] + hs
                hit = True
            elif low[i] + hs <= tp:
                ex = tp
                hit = True
            if hit:
                pnl = (entry_fill - ex) * value_per_point * lots - commission_per_lot * lots
                balance += pnl
                trades += 1
                if pnl > 0.0:
                    wins += 1
                    gp += pnl
                else:
                    gl -= pnl
                if n < o_pnl.shape[0]:
                    o_entry_i[n] = entry_idx; o_exit_i[n] = i; o_dir[n] = -1
                    o_lots[n] = lots; o_entry_px[n] = entry_fill; o_exit_px[n] = ex
                    o_pnl[n] = pnl; o_risk[n] = risk_now
                    o_atr[n] = atr_now; o_rsi[n] = rsi_now; o_emasep[n] = emasep_now
                    n += 1
                position = 0
                closed = True

        # ---- equity / drawdown --------------------------------------
        if position == 1:
            unreal = ((close[i] - hs) - entry_fill) * value_per_point * lots
        elif position == -1:
            unreal = (entry_fill - (close[i] + hs)) * value_per_point * lots
        else:
            unreal = 0.0
        equity = balance + unreal
        if equity > peak_equity:
            peak_equity = equity
        dd = peak_equity - equity
        if dd > max_dd_abs:
            max_dd_abs = dd

        # ---- equity kill-switch ------------------------------------
        if equity <= equity_stop:
            if position != 0:
                if position == 1:
                    exx = close[i] - hs - slippage
                    pnl = (exx - entry_fill) * value_per_point * lots - commission_per_lot * lots
                else:
                    exx = close[i] + hs + slippage
                    pnl = (entry_fill - exx) * value_per_point * lots - commission_per_lot * lots
                balance += pnl
                trades += 1
                if pnl > 0.0:
                    wins += 1
                    gp += pnl
                else:
                    gl -= pnl
                if n < o_pnl.shape[0]:
                    o_entry_i[n] = entry_idx; o_exit_i[n] = i; o_dir[n] = position
                    o_lots[n] = lots; o_entry_px[n] = entry_fill; o_exit_px[n] = exx
                    o_pnl[n] = pnl; o_risk[n] = risk_now
                    o_atr[n] = atr_now; o_rsi[n] = rsi_now; o_emasep[n] = emasep_now
                    n += 1
                position = 0
            break

        if position != 0 or closed:
            continue

        # ---- signal (bar i-1 only) --------------------------------
        a = av[i - 1]
        if not (a > 0.0):
            continue
        if np.isnan(ef[i - 1]) or np.isnan(es[i - 1]) or np.isnan(et[i - 1]):
            continue

        strat_buy = ef[i - 1] > es[i - 1] and es[i - 1] > et[i - 1] and rv[i - 1] >= rsi_buy
        strat_sell = ef[i - 1] < es[i - 1] and es[i - 1] < et[i - 1] and rv[i - 1] <= rsi_sell

        want = 0
        if signal_mode == 0:
            if strat_buy:
                want = 1
            elif strat_sell:
                want = -1
        elif signal_mode == 1:
            if strat_buy or strat_sell:
                want = 1 if np.random.random() < 0.5 else -1
        elif signal_mode == 2:
            if strat_buy or strat_sell:
                want = 1
        elif signal_mode == 3:
            if strat_buy or strat_sell:
                want = -1
        elif signal_mode == 4:
            if entry_mask[i] != 0:
                want = 1 if entry_dir_arr[i] > 0 else -1

        if want == 0:
            continue

        # ---- stop / target distance (engine lines 452-470) --------
        stop_distance = a * sl_mult
        floor_dist = min_stop_dist
        cost_dist = spread[i] + 2.0 * slippage
        if a * sl_mult < 2.0 * cost_dist:
            continue
        if floor_dist < cost_dist:
            floor_dist = cost_dist
        if stop_distance < floor_dist:
            stop_distance = floor_dist
        tp_distance = a * tp_mult
        if tp_distance < min_stop_dist:
            tp_distance = min_stop_dist

        # ---- sizing (engine lines 475-501) -----------------------
        risk_dollars = balance * risk_per_trade
        loss_per_lot = (stop_distance + slippage) * value_per_point + commission_per_lot
        if loss_per_lot <= 0.0:
            continue
        raw_lots = risk_dollars / loss_per_lot
        lots = np.floor(raw_lots / lot_step + 1e-9) * lot_step
        ref_price = open_[i]
        margin_per_lot = ref_price * contract_size / leverage
        if margin_per_lot > 0.0:
            mbm = (balance * max_margin_frac) / margin_per_lot
            mbm = np.floor(mbm / lot_step + 1e-9) * lot_step
            if lots > mbm:
                lots = mbm
        if lots > max_lot:
            lots = max_lot
        if lots < min_lot:
            lots = 0.0
            continue

        risk_now = (stop_distance + slippage) * value_per_point * lots   # realised initial $ risk
        atr_now = a
        rsi_now = rv[i - 1]
        emasep_now = (ef[i - 1] - et[i - 1]) / close[i - 1]

        if want == 1:
            entry_fill = open_[i] + hs
            sl = entry_fill - stop_distance
            tp = entry_fill + tp_distance
            position = 1
        else:
            entry_fill = open_[i] - hs
            sl = entry_fill + stop_distance
            tp = entry_fill - tp_distance
            position = -1
        entry_idx = i

    # ---- close still-open position at window end (engine 513-542) ----
    if position != 0:
        j = i1 - 1
        hs = spread[j] * 0.5
        if position == 1:
            exx = close[j] - hs - slippage
            pnl = (exx - entry_fill) * value_per_point * lots - commission_per_lot * lots
        else:
            exx = close[j] + hs + slippage
            pnl = (entry_fill - exx) * value_per_point * lots - commission_per_lot * lots
        balance += pnl
        trades += 1
        if pnl > 0.0:
            wins += 1
            gp += pnl
        else:
            gl -= pnl
        if n < o_pnl.shape[0]:
            o_entry_i[n] = entry_idx; o_exit_i[n] = j; o_dir[n] = position
            o_lots[n] = lots; o_entry_px[n] = entry_fill; o_exit_px[n] = exx
            o_pnl[n] = pnl; o_risk[n] = risk_now
            o_atr[n] = atr_now; o_rsi[n] = rsi_now; o_emasep[n] = emasep_now
            n += 1
        position = 0

    pf = gp / gl if gl > 0.0 else (np.inf if gp > 0.0 else 0.0)
    wr = wins / trades if trades > 0 else 0.0
    return (float(trades), balance, balance - initial_balance, wr, pf,
            max_dd_abs, gp, gl, float(n))


# ===========================================================================
# PYTHON DRIVER HELPERS
# ===========================================================================
class Run:
    __slots__ = ("summary", "trades")

    def __init__(self, summary, trades):
        self.summary = summary
        self.trades = trades


def run_kernel(px, ind, spread, i0, i1, warmup, sl_mult, tp_mult,
               signal_mode=0, rng_seed=0, entry_mask=None, entry_dir_arr=None,
               cap=None):
    o, h, l, c = px
    ef, es, et, rv, av = ind
    n_bars = len(c)
    cap = cap if cap is not None else n_bars
    z = lambda dt: np.zeros(cap, dtype=dt)
    o_entry_i = z(np.int64); o_exit_i = z(np.int64); o_dir = z(np.int64)
    o_lots = z(np.float64); o_entry_px = z(np.float64); o_exit_px = z(np.float64)
    o_pnl = z(np.float64); o_risk = z(np.float64)
    o_atr = z(np.float64); o_rsi = z(np.float64); o_emasep = z(np.float64)
    if entry_mask is None:
        entry_mask = np.zeros(n_bars, dtype=np.int8)
    if entry_dir_arr is None:
        entry_dir_arr = np.zeros(n_bars, dtype=np.int8)

    res = edge_kernel(
        o, h, l, c, ef, es, et, rv, av, spread,
        int(i0), int(i1), int(warmup),
        RSI_BUY, RSI_SELL, float(sl_mult), float(tp_mult),
        INITIAL_BALANCE, RISK_PER_TRADE, VALUE_PER_POINT, CONTRACT_SIZE, LEVERAGE,
        MIN_LOT, LOT_STEP, MAX_LOT, SLIPPAGE, COMMISSION_PER_LOT, MIN_STOP_DIST,
        EQUITY_STOP, MAX_MARGIN_FRAC,
        int(signal_mode), np.int64(rng_seed), entry_mask, entry_dir_arr,
        o_entry_i, o_exit_i, o_dir, o_lots, o_entry_px, o_exit_px, o_pnl, o_risk,
        o_atr, o_rsi, o_emasep,
    )
    m = int(round(res[8]))
    summary = dict(
        trades=int(round(res[0])), final_balance=float(res[1]), net_pl=float(res[2]),
        win_rate=float(res[3]), profit_factor=float(res[4]), max_dd_abs=float(res[5]),
        gross_profit=float(res[6]), gross_loss=float(res[7]), n_recorded=m,
    )
    td = dict(
        entry_i=o_entry_i[:m].copy(), exit_i=o_exit_i[:m].copy(), dir=o_dir[:m].copy(),
        lots=o_lots[:m].copy(), entry_px=o_entry_px[:m].copy(), exit_px=o_exit_px[:m].copy(),
        pnl=o_pnl[:m].copy(), risk=o_risk[:m].copy(),
        atr=o_atr[:m].copy(), rsi=o_rsi[:m].copy(), emasep=o_emasep[:m].copy(),
    )
    return Run(summary, td)


def pf_of(pnl):
    pnl = np.asarray(pnl, dtype=float)
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl <= 0].sum()
    if gl > 0:
        return gp / gl
    return float("inf") if gp > 0 else 0.0


def max_dd_from_pnl(pnl):
    eq = INITIAL_BALANCE + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[INITIAL_BALANCE], eq]))
    return float(np.max(peak - np.concatenate([[INITIAL_BALANCE], eq])))


def stat_block(pnl, risk):
    pnl = np.asarray(pnl, float)
    risk = np.asarray(risk, float)
    r = np.where(risk > 0, pnl / risk, 0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return dict(
        trades=int(len(pnl)),
        win_rate=float((pnl > 0).mean()) if len(pnl) else 0.0,
        gross_profit=float(wins.sum()),
        gross_loss=float(-losses.sum()),
        profit_factor=float(pf_of(pnl)),
        net_pl=float(pnl.sum()),
        expectancy_dollar=float(pnl.mean()) if len(pnl) else 0.0,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        max_dd_abs=max_dd_from_pnl(pnl) if len(pnl) else 0.0,
        mean_R=float(r.mean()) if len(r) else 0.0,
        median_R=float(np.median(r)) if len(r) else 0.0,
        std_R=float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        win_R=float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        loss_R=float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        expectancy_R=float(r.mean()) if len(r) else 0.0,
        cumulative_R=float(r.sum()) if len(r) else 0.0,
    )


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.ndarray,)):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


# ===========================================================================
# LOAD DATA + INDICATORS
# ===========================================================================
_lap("loading data")
df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
o = np.ascontiguousarray(df.open.to_numpy(np.float64))
h = np.ascontiguousarray(df.high.to_numpy(np.float64))
l = np.ascontiguousarray(df.low.to_numpy(np.float64))
c = np.ascontiguousarray(df.close.to_numpy(np.float64))
raw_pts = df["spread"].to_numpy(np.float64)
N = len(c)

ec.DEFAULT_SPREAD = bc.DEFAULT_SPREAD                    # baseline parity (btc_realistic_baseline.py:24)
spread_real = ec.prepare_spread(df)

ef, es, et, rv, av, warmup = ec.compute_indicators(o, h, l, c, FAST, SLOW, TREND, RSI_P, ATR_P)
PX = (o, h, l, c)
IND = (np.ascontiguousarray(ef), np.ascontiguousarray(es), np.ascontiguousarray(et),
       np.ascontiguousarray(rv), np.ascontiguousarray(av))

TRAIN_END = int(N * SPLIT_FRACS[0])
VAL_END = int(N * SPLIT_FRACS[1])
SPLITS = [("TRAIN", 0, TRAIN_END), ("VALIDATION", TRAIN_END, VAL_END), ("FINAL", VAL_END, N)]
_lap(f"data rows={N}  warmup={warmup}  splits TRAIN[0,{TRAIN_END}) VAL[{TRAIN_END},{VAL_END}) FINAL[{VAL_END},{N})")

RESULTS = {
    "meta": dict(
        engine_version=ec.ENGINE_VERSION, data=str(DATA.relative_to(ROOT)), rows=int(N),
        warmup=int(warmup), splits={s[0]: [s[1], s[2]] for s in SPLITS},
        strategy=dict(ema_fast=FAST, ema_slow=SLOW, ema_trend=TREND, rsi_period=RSI_P,
                      atr_period=ATR_P, sl_mult=SL_MULT, tp_mult=TP_MULT,
                      rsi_buy=RSI_BUY, rsi_sell=RSI_SELL),
        costs=dict(default_spread=DEFAULT_SPREAD, slippage=SLIPPAGE,
                   commission_per_lot=COMMISSION_PER_LOT, clamp_mult=CLAMP_MULT),
        account=dict(initial_balance=INITIAL_BALANCE, risk_per_trade=RISK_PER_TRADE,
                     min_lot=MIN_LOT, lot_step=LOT_STEP, equity_stop=EQUITY_STOP,
                     value_per_point=VALUE_PER_POINT, leverage=LEVERAGE),
        n_perm=N_PERM, numba=NUMBA,
    )
}

# ===========================================================================
# 0.  KERNEL VALIDATION  vs  ec.run_window_logged   (mode 0 must match exactly)
# ===========================================================================
_lap("compiling + validating kernel against ec.run_window_logged ...")
base_runs = {}
val_ok = True
val_detail = {}
for name, i0, i1 in SPLITS:
    r = run_kernel(PX, IND, spread_real, i0, i1, warmup, SL_MULT, TP_MULT, signal_mode=0)
    res_ec, tr_ec = ec.run_window_logged(
        o, h, l, c, IND[0], IND[1], IND[2], IND[3], IND[4], spread_real,
        i0, i1, warmup, RSI_BUY, RSI_SELL, SL_MULT, TP_MULT,
    )
    d_ec = ec.summary_dict(res_ec)
    same = (
        r.summary["trades"] == d_ec["trades"]
        and abs(r.summary["net_pl"] - d_ec["net_pl"]) < 1e-6
        and abs(float(np.sum(r.trades["pnl"])) - float(np.sum(tr_ec["pnl"]))) < 1e-6
    )
    val_detail[name] = dict(
        kernel=dict(trades=r.summary["trades"], net_pl=r.summary["net_pl"], pf=r.summary["profit_factor"]),
        engine=dict(trades=d_ec["trades"], net_pl=d_ec["net_pl"], pf=d_ec["profit_factor"]),
        match=bool(same),
    )
    val_ok = val_ok and same
    base_runs[name] = r
    print(f"    {name:11s} kernel(t={r.summary['trades']:3d}, pl={r.summary['net_pl']:+.4f})  "
          f"engine(t={d_ec['trades']:3d}, pl={d_ec['net_pl']:+.4f})  match={same}")

RESULTS["kernel_validation"] = dict(all_match=bool(val_ok), detail=val_detail)
if not val_ok:
    print("\n*** KERNEL DOES NOT MATCH ENGINE -- ABORTING (fix the port, not the engine) ***")
    (REPORTS / "btc_edge_validation_summary.txt").write_text(
        "ABORTED: edge_kernel output diverged from ec.run_window_logged.\n"
        f"detail: {json.dumps(jsonable(val_detail), indent=2)}\n", encoding="utf-8")
    sys.exit(1)
_lap("kernel validated -- exact match on all three windows")

# combined baseline trade frame (TRAIN+VAL+FINAL, exactly the reported baseline)
_rows = []
for name, i0, i1 in SPLITS:
    t = base_runs[name].trades
    for k in range(t["pnl"].shape[0]):
        _rows.append(dict(
            split=name, entry_i=int(t["entry_i"][k]), exit_i=int(t["exit_i"][k]),
            dir=int(t["dir"][k]), lots=float(t["lots"][k]),
            entry_px=float(t["entry_px"][k]), exit_px=float(t["exit_px"][k]),
            pnl=float(t["pnl"][k]), risk=float(t["risk"][k]),
            R=float(t["pnl"][k] / t["risk"][k]) if t["risk"][k] > 0 else 0.0,
            atr=float(t["atr"][k]), rsi=float(t["rsi"][k]), emasep=float(t["emasep"][k]),
            entry_time=df.time.iloc[int(t["entry_i"][k])],
        ))
TB = pd.DataFrame(_rows)
TB["entry_month"] = TB["entry_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
RESULTS["baseline_reconciliation"] = {
    name: jsonable(base_runs[name].summary) for name in base_runs
}
RESULTS["baseline_reconciliation"]["COMBINED"] = jsonable(stat_block(TB.pnl.to_numpy(), TB.risk.to_numpy()))

CSV_ROWS = []


def emit(section, metric, scope, value):
    CSV_ROWS.append(dict(section=section, metric=metric, scope=scope, value=value))


for name in ("TRAIN", "VALIDATION", "FINAL"):
    s = base_runs[name].summary
    emit("baseline", "trades", name, s["trades"])
    emit("baseline", "net_pl", name, round(s["net_pl"], 4))
    emit("baseline", "profit_factor", name, round(s["profit_factor"], 4))
    emit("baseline", "win_rate", name, round(s["win_rate"], 4))
cb = RESULTS["baseline_reconciliation"]["COMBINED"]
emit("baseline", "net_pl", "COMBINED", round(cb["net_pl"], 4))
emit("baseline", "profit_factor", "COMBINED", round(cb["profit_factor"], 4))
emit("baseline", "expectancy_R", "COMBINED", round(cb["expectancy_R"], 4))

# ===========================================================================
# 1.  LONG vs SHORT
# ===========================================================================
_lap("test 1: long vs short")
ls = {}
for side, dv in (("LONG", 1), ("SHORT", -1), ("ALL", 0)):
    sub = TB if dv == 0 else TB[TB.dir == dv]
    b = stat_block(sub.pnl.to_numpy(), sub.risk.to_numpy())
    ls[side] = b
    for kk in ("trades", "win_rate", "gross_profit", "gross_loss", "profit_factor",
               "expectancy_dollar", "expectancy_R", "avg_win", "avg_loss", "max_dd_abs"):
        emit("long_vs_short", kk, side, round(b[kk], 4))
RESULTS["long_vs_short"] = jsonable(ls)

# ===========================================================================
# 2.  R-MULTIPLE ANALYSIS
# ===========================================================================
_lap("test 2: R-multiple analysis")
rm = {}
for name in ("TRAIN", "VALIDATION", "FINAL"):
    sub = TB[TB.split == name]
    rm[name] = stat_block(sub.pnl.to_numpy(), sub.risk.to_numpy())
rm["COMBINED"] = stat_block(TB.pnl.to_numpy(), TB.risk.to_numpy())
for name, b in rm.items():
    for kk in ("mean_R", "median_R", "std_R", "win_R", "loss_R", "expectancy_R", "cumulative_R"):
        emit("r_multiple", kk, name, round(b[kk], 4))
RESULTS["r_multiple"] = jsonable(rm)

# ===========================================================================
# 3.  SIGNAL-ONLY BASELINES
# ===========================================================================
_lap("test 3: signal-only baselines (random dir / random entries / always long / always short)")
S3_SEED = 12345
sig_base = {}


def _combine_runs(mode, seed_fn, mask_fn=None):
    pnl_all, risk_all = [], []
    per = {}
    for name, i0, i1 in SPLITS:
        kw = {}
        if mask_fn is not None:
            em, ed = mask_fn(name, i0, i1)
            kw = dict(entry_mask=em, entry_dir_arr=ed)
        r = run_kernel(PX, IND, spread_real, i0, i1, warmup, SL_MULT, TP_MULT,
                       signal_mode=mode, rng_seed=seed_fn(name), **kw)
        per[name] = stat_block(r.trades["pnl"], r.trades["risk"])
        pnl_all.append(r.trades["pnl"]); risk_all.append(r.trades["risk"])
    per["COMBINED"] = stat_block(np.concatenate(pnl_all), np.concatenate(risk_all))
    return per


sig_base["real_strategy"] = {**{k: rm[k] for k in ("TRAIN", "VALIDATION", "FINAL")},
                             "COMBINED": rm["COMBINED"]}
sig_base["random_direction_same_entries"] = _combine_runs(
    1, lambda nm: S3_SEED + hash(nm) % 997)
sig_base["always_long"] = _combine_runs(2, lambda nm: 0)
sig_base["always_short"] = _combine_runs(3, lambda nm: 0)


def _rand_entry_mask(name, i0, i1):
    target = max(1, base_runs[name].summary["trades"])
    rng = np.random.default_rng(S3_SEED + (hash(name) % 997))
    lo = max(i0, warmup, 1)
    valid = np.arange(lo, i1)
    valid = valid[np.isfinite(av[valid - 1]) & (av[valid - 1] > 0)
                  & np.isfinite(ef[valid - 1]) & np.isfinite(et[valid - 1])]
    pick = rng.choice(valid, size=min(len(valid), target * 3), replace=False)
    em = np.zeros(N, np.int8); ed = np.zeros(N, np.int8)
    em[pick] = 1
    ed[pick] = rng.integers(0, 2, size=len(pick)).astype(np.int8) * 2 - 1
    return em, ed


sig_base["random_entries_random_direction"] = _combine_runs(4, lambda nm: 0, _rand_entry_mask)

for label, per in sig_base.items():
    for name in ("TRAIN", "VALIDATION", "FINAL", "COMBINED"):
        b = per[name]
        emit("signal_only", "net_pl", f"{label}/{name}", round(b["net_pl"], 4))
        emit("signal_only", "profit_factor", f"{label}/{name}", round(b["profit_factor"], 4))
        emit("signal_only", "expectancy_R", f"{label}/{name}", round(b["expectancy_R"], 4))
RESULTS["signal_only"] = jsonable(sig_base)

# ===========================================================================
# 4.  COST SENSITIVITY
# ===========================================================================
_lap("test 4: cost sensitivity (constant spread arrays)")
cost_grid = [0.00, 0.02, 0.05, 0.10, 0.15, 0.18, 0.25]
cost_res = {}
for sp in cost_grid:
    sp_arr = np.ascontiguousarray(np.full(N, sp, dtype=np.float64))
    row = {}
    for name, i0, i1 in SPLITS:
        r = run_kernel(PX, IND, sp_arr, i0, i1, warmup, SL_MULT, TP_MULT, signal_mode=0)
        row[name] = dict(trades=r.summary["trades"], profit=round(r.summary["net_pl"], 4),
                         pf=round(r.summary["profit_factor"], 4))
        emit("cost_sensitivity", "profit", f"spread={sp:.2f}/{name}", round(r.summary["net_pl"], 4))
        emit("cost_sensitivity", "profit_factor", f"spread={sp:.2f}/{name}", round(r.summary["profit_factor"], 4))
    cost_res[f"{sp:.2f}"] = row
RESULTS["cost_sensitivity"] = jsonable(cost_res)

# ===========================================================================
# 5.  SL/TP STRUCTURE  (DIAGNOSTIC ONLY -- NOT OPTIMIZATION)
# ===========================================================================
_lap("test 5: SL/TP structure diagnostic (NOT optimization)")
brackets = [(1.5, 3.0), (2.0, 3.0), (2.0, 3.5), (2.5, 3.5), (3.0, 4.0)]
br_res = {}
for slm, tpm in brackets:
    pnl_all, risk_all = [], []
    row = {}
    for name, i0, i1 in SPLITS:
        r = run_kernel(PX, IND, spread_real, i0, i1, warmup, slm, tpm, signal_mode=0)
        row[name] = dict(trades=r.summary["trades"], profit=round(r.summary["net_pl"], 4),
                         pf=round(r.summary["profit_factor"], 4))
        pnl_all.append(r.trades["pnl"]); risk_all.append(r.trades["risk"])
        emit("sltp_diagnostic", "profit", f"{slm}/{tpm}/{name}", round(r.summary["net_pl"], 4))
        emit("sltp_diagnostic", "profit_factor", f"{slm}/{tpm}/{name}", round(r.summary["profit_factor"], 4))
    comb = stat_block(np.concatenate(pnl_all), np.concatenate(risk_all))
    row["COMBINED"] = dict(trades=comb["trades"], profit=round(comb["net_pl"], 4),
                           pf=round(comb["profit_factor"], 4), expectancy_R=round(comb["expectancy_R"], 4))
    emit("sltp_diagnostic", "profit_factor", f"{slm}/{tpm}/COMBINED", round(comb["profit_factor"], 4))
    emit("sltp_diagnostic", "expectancy_R", f"{slm}/{tpm}/COMBINED", round(comb["expectancy_R"], 4))
    br_res[f"{slm}/{tpm}"] = row
RESULTS["sltp_diagnostic"] = jsonable(br_res)
RESULTS["sltp_diagnostic"]["_note"] = "DIAGNOSTIC ONLY - probes whether bracket structure explains the poor result; not an optimization sweep."

# ===========================================================================
# 6.  REGIME ANALYSIS
# ===========================================================================
_lap("test 6: regime analysis")
reg = {}


def _bucket_report(frame, col, edges, labels):
    out = {}
    for lab, (lo, hi) in zip(labels, edges):
        if hi is None:
            sub = frame[frame[col] >= lo]
        else:
            sub = frame[(frame[col] >= lo) & (frame[col] < hi)]
        b = stat_block(sub.pnl.to_numpy(), sub.risk.to_numpy())
        out[lab] = dict(trades=b["trades"], win_rate=round(b["win_rate"], 4),
                        profit_factor=round(b["profit_factor"], 4),
                        expectancy_R=round(b["expectancy_R"], 4),
                        net_pl=round(b["net_pl"], 4))
    return out


atr_q = np.quantile(TB.atr, [1 / 3, 2 / 3])
sepabs = TB.emasep.abs()
sep_q = np.quantile(sepabs, [1 / 3, 2 / 3])
TB2 = TB.assign(sepabs=sepabs)

reg["atr"] = _bucket_report(
    TB, "atr", [(-1e9, atr_q[0]), (atr_q[0], atr_q[1]), (atr_q[1], None)],
    ["low", "medium", "high"])
reg["trend_strength_ema_sep"] = _bucket_report(
    TB2, "sepabs", [(-1e9, sep_q[0]), (sep_q[0], sep_q[1]), (sep_q[1], None)],
    ["weak", "medium", "strong"])
reg["rsi"] = _bucket_report(
    TB, "rsi", [(-1e9, 40.0 + 1e-9), (40.0 + 1e-9, 60.0), (60.0 - 1e-9, None)],
    ["<=40", "40-60", ">=60"])
reg["direction"] = {
    "LONG": {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in stat_block(TB[TB.dir == 1].pnl.to_numpy(), TB[TB.dir == 1].risk.to_numpy()).items()
             if k in ("trades", "win_rate", "profit_factor", "expectancy_R", "net_pl")},
    "SHORT": {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in stat_block(TB[TB.dir == -1].pnl.to_numpy(), TB[TB.dir == -1].risk.to_numpy()).items()
              if k in ("trades", "win_rate", "profit_factor", "expectancy_R", "net_pl")},
}
reg["_note_rsi"] = ("40-60 bucket is empty BY STRATEGY CONSTRUCTION (entry requires "
                    "RSI>=60 for long or RSI<=40 for short).")
for grp, buckets in reg.items():
    if grp.startswith("_"):
        continue
    for lab, b in buckets.items():
        emit("regime", "trades", f"{grp}/{lab}", b["trades"])
        emit("regime", "profit_factor", f"{grp}/{lab}", b.get("profit_factor", 0.0))
        emit("regime", "expectancy_R", f"{grp}/{lab}", b.get("expectancy_R", 0.0))
RESULTS["regime"] = jsonable(reg)

# ===========================================================================
# 7.  TEMPORAL STABILITY  (monthly, chronological, no shuffling)
# ===========================================================================
_lap("test 7: temporal stability (monthly blocks)")
temporal = {}
for mon, sub in TB.groupby("entry_month"):
    b = stat_block(sub.pnl.to_numpy(), sub.risk.to_numpy())
    temporal[mon] = dict(trades=b["trades"], profit=round(b["net_pl"], 4),
                         profit_factor=round(b["profit_factor"], 4),
                         win_rate=round(b["win_rate"], 4),
                         expectancy_R=round(b["expectancy_R"], 4))
    emit("temporal", "profit", mon, round(b["net_pl"], 4))
    emit("temporal", "profit_factor", mon, round(b["profit_factor"], 4))
    emit("temporal", "expectancy_R", mon, round(b["expectancy_R"], 4))
months_pos = sum(1 for v in temporal.values() if v["expectancy_R"] > 0)
temporal["_summary"] = dict(blocks=len(temporal), blocks_positive_expectancy_R=months_pos,
                            fraction_positive=round(months_pos / max(1, len(temporal)), 4))
RESULTS["temporal"] = jsonable(temporal)

# ===========================================================================
# 8.  PERMUTATION TEST  (random direction, same execution model)
# ===========================================================================
_lap(f"test 8: permutation test  ({N_PERM} trials x 3 windows)")
obs_R = TB.pnl.to_numpy() / np.where(TB.risk.to_numpy() > 0, TB.risk.to_numpy(), 1.0)
obs_expectancy_R = float(obs_R.mean())
obs_total_profit = float(TB.pnl.sum())
obs_by_split = {nm: float(rm[nm]["expectancy_R"]) for nm in ("TRAIN", "VALIDATION", "FINAL")}

perm_expectancy = np.empty(N_PERM, np.float64)
perm_profit = np.empty(N_PERM, np.float64)
t8 = time.perf_counter()
for k in range(N_PERM):
    pnl_all, risk_all = [], []
    for wi, (name, i0, i1) in enumerate(SPLITS):
        r = run_kernel(PX, IND, spread_real, i0, i1, warmup, SL_MULT, TP_MULT,
                       signal_mode=1, rng_seed=PERM_SEED_BASE + k * 3 + wi)
        pnl_all.append(r.trades["pnl"]); risk_all.append(r.trades["risk"])
    pnl_all = np.concatenate(pnl_all)
    risk_all = np.concatenate(risk_all)
    rr = pnl_all / np.where(risk_all > 0, risk_all, 1.0)
    perm_expectancy[k] = rr.mean() if len(rr) else 0.0
    perm_profit[k] = pnl_all.sum()
    if (k + 1) % 200 == 0:
        _lap(f"    permutation {k + 1}/{N_PERM}  ({(time.perf_counter() - t8):.2f}s)")

frac_beat_R = float((perm_expectancy >= obs_expectancy_R).mean())
frac_beat_P = float((perm_profit >= obs_total_profit).mean())
perm = dict(
    n_trials=N_PERM, seed_base=PERM_SEED_BASE,
    observed_expectancy_R=round(obs_expectancy_R, 5),
    observed_total_profit=round(obs_total_profit, 4),
    observed_expectancy_R_by_split={k: round(v, 5) for k, v in obs_by_split.items()},
    random_expectancy_R_mean=round(float(perm_expectancy.mean()), 5),
    random_expectancy_R_std=round(float(perm_expectancy.std(ddof=1)), 5),
    random_profit_mean=round(float(perm_profit.mean()), 4),
    random_profit_std=round(float(perm_profit.std(ddof=1)), 4),
    observed_percentile_expectancy_R=round(100.0 * float((perm_expectancy <= obs_expectancy_R).mean()), 2),
    observed_percentile_profit=round(100.0 * float((perm_profit <= obs_total_profit).mean()), 2),
    fraction_random_beating_observed_expectancy_R=round(frac_beat_R, 4),
    fraction_random_beating_observed_profit=round(frac_beat_P, 4),
    elapsed_s=round(time.perf_counter() - t8, 2),
)
RESULTS["permutation"] = jsonable(perm)
emit("permutation", "observed_expectancy_R", "COMBINED", perm["observed_expectancy_R"])
emit("permutation", "random_expectancy_R_mean", "COMBINED", perm["random_expectancy_R_mean"])
emit("permutation", "random_expectancy_R_std", "COMBINED", perm["random_expectancy_R_std"])
emit("permutation", "observed_percentile", "COMBINED", perm["observed_percentile_expectancy_R"])
emit("permutation", "fraction_random_beating_observed", "COMBINED", perm["fraction_random_beating_observed_expectancy_R"])

# ===========================================================================
# 9.  DATA QUALITY / SPREAD IMPACT
# ===========================================================================
_lap("test 9: data quality / spread impact")
TICK = float(ec.TICK_SIZE)
clamp_val = CLAMP_MULT * DEFAULT_SPREAD
MODAL_PTS = float(np.bincount(raw_pts.astype(np.int64)).argmax())     # 1 pt on this feed
ELEVATED_PTS = 10.0                          # >10 pts = $0.10 : above the normal 1-2 pt band
CORRUPT_PTS = 100.0                          # >100 pts : physically impossible tick, feed artifact
abn_mask = raw_pts > ELEVATED_PTS
abn_idx = np.where(abn_mask)[0]
traded_entry = set(TB.entry_i.tolist())
traded_exit = set(TB.exit_i.tolist())
abn_rows = []
affected_trades = []
for ix in abn_idx:
    conv = raw_pts[ix] * TICK
    post = min(conv, clamp_val)
    role = ("entry" if ix in traded_entry else "") + ("/exit" if ix in traded_exit else "")
    rec = dict(
        bar=int(ix), time=str(df.time.iloc[ix]), raw_points=float(raw_pts[ix]),
        converted_price=round(conv, 4), post_clamp=round(post, 4),
        classification=("corrupt/gap" if raw_pts[ix] > CORRUPT_PTS else "elevated"),
        used_in_a_trade=bool(role), trade_role=role.strip("/"),
    )
    abn_rows.append(rec)
    if role:
        tr = TB[(TB.entry_i == ix) | (TB.exit_i == ix)]
        for _, t in tr.iterrows():
            # cost distortion vs a realistic modal spread on the affected leg
            realistic_hs = MODAL_PTS * TICK * 0.5
            clamped_hs = post * 0.5
            extra_cost = (clamped_hs - realistic_hs) * VALUE_PER_POINT * float(t["lots"])
            affected_trades.append(dict(
                bar=int(ix), split=t["split"], dir=int(t["dir"]), lots=float(t["lots"]),
                pnl=round(float(t["pnl"]), 4),
                extra_cost_vs_modal_spread=round(float(extra_cost), 4),
                note="clamp makes this leg MORE expensive than a modal-spread fill "
                     "(pessimistic, not favourable)",
            ))
uniq, cnts = np.unique(raw_pts, return_counts=True)
gross_abs = float(TB.pnl.abs().sum())
tot_distort = float(sum(a["extra_cost_vs_modal_spread"] for a in affected_trades))
distort_pct = round(100.0 * tot_distort / gross_abs, 3) if gross_abs else 0.0
dq = dict(
    tick_size=TICK, default_spread=DEFAULT_SPREAD, upper_clamp=clamp_val, modal_points=MODAL_PTS,
    raw_points_distribution={str(int(u)): int(n) for u, n in zip(uniq, cnts)},
    n_bars=int(N),
    n_bars_spread_gt_10pts=int(abn_mask.sum()),
    n_bars_corrupt_gt_100pts=int((raw_pts > CORRUPT_PTS).sum()),
    abnormal_bars=abn_rows,
    affected_trades=affected_trades,
    any_abnormal_bar_used_in_trade=bool(affected_trades),
    total_distortion_dollars=round(tot_distort, 4),
    total_distortion_pct_of_gross_pnl=round(100.0 * tot_distort / gross_abs, 4) if gross_abs else 0.0,
    clamp_creates_unrealistic_fill=False,
    clamp_note=(f"Bars with raw spread > {int(CORRUPT_PTS)} pts are feed artifacts at "
                f"weekend/session reopens (price also gaps on those bars). The engine clamps "
                f"them DOWN to ${clamp_val:.2f} (= {CLAMP_MULT:.0f} x DEFAULT_SPREAD), still "
                f"~{clamp_val / (MODAL_PTS * TICK):.0f}x the modal ${MODAL_PTS*TICK:.2f} spread "
                f"-- clamping makes any fill on those bars MORE expensive than normal, never "
                f"cheaper. No bar is dropped. {len(affected_trades)} of {len(TB)} trades touch "
                f"such a bar; total P/L distortion vs a modal-spread fill is ${tot_distort:+.2f} "
                f"({distort_pct}% of gross P/L), in the pessimistic direction."),
)
RESULTS["data_quality"] = jsonable(dq)
emit("data_quality", "n_bars_spread_gt_10pts", "ALL", int(abn_mask.sum()))
emit("data_quality", "n_bars_corrupt_gt_100pts", "ALL", dq["n_bars_corrupt_gt_100pts"])
emit("data_quality", "affected_trades", "ALL", len(dq["affected_trades"]))
emit("data_quality", "total_distortion_dollars", "ALL", dq["total_distortion_dollars"])
emit("data_quality", "clamp_creates_favourable_fill", "ALL", dq["clamp_creates_unrealistic_fill"])

# ===========================================================================
# 10.  VERDICT
# ===========================================================================
_lap("test 10: scoring + verdict")
comb = rm["COMBINED"]
n_windows_pos_R = sum(1 for nm in ("TRAIN", "VALIDATION", "FINAL") if rm[nm]["expectancy_R"] > 0)
cost_ok_le_010 = all(
    cost_res[f"{sp:.2f}"][nm]["profit"] > 0
    for sp in (0.00, 0.02, 0.05, 0.10) for nm in ("TRAIN", "VALIDATION", "FINAL")
)
cost_combined_le_010 = all(
    sum(cost_res[f"{sp:.2f}"][nm]["profit"] for nm in ("TRAIN", "VALIDATION", "FINAL")) > 0
    for sp in (0.02, 0.05, 0.10)
)
best_bracket_pf = max(v["COMBINED"]["pf"] for v in br_res.values())
frac_pos_months = RESULTS["temporal"]["_summary"]["fraction_positive"]
perc = perm["observed_percentile_expectancy_R"]

score = 0.0
reasons = []


def _s(cond, pts, msg):
    global score
    if cond:
        score += pts
        reasons.append(f"+{pts:>4}  {msg}")
    else:
        reasons.append(f"{0:>5}  (not met) {msg}")


_s(comb["expectancy_R"] > 0, 1, "combined expectancy R > 0")
_s(comb["expectancy_R"] > 0.05, 1, "combined expectancy R > 0.05")
_s(comb["profit_factor"] > 1.0, 1, "combined profit factor > 1.0")
_s(comb["profit_factor"] > 1.2, 1, "combined profit factor > 1.2")
_s(n_windows_pos_R >= 2, 1, "at least 2 of 3 windows positive expectancy R")
_s(n_windows_pos_R == 3, 1, "all 3 windows positive expectancy R")
_s(perc >= 90.0, 1, "observed >= 90th percentile of random-direction trials")
_s(perc >= 95.0, 1, "observed >= 95th percentile of random-direction trials")
_s(cost_combined_le_010, 1, "combined profit stays > 0 at spreads 0.02/0.05/0.10")
_s(frac_pos_months >= 0.6, 1, ">= 60% of monthly blocks positive expectancy R")
_s(best_bracket_pf > 1.1, 0.5, "best diagnostic bracket combined PF > 1.1")

if score <= 1:
    verdict = "NO EDGE EVIDENCE"
elif score <= 3:
    verdict = "WEAK / UNSTABLE EDGE"
elif score <= 6:
    verdict = "PROMISING EDGE"
else:
    verdict = "STRONG ROBUST EDGE"

RESULTS["verdict"] = dict(
    classification=verdict, score=round(score, 2), max_score=11.5,
    inputs=dict(
        combined_expectancy_R=round(comb["expectancy_R"], 5),
        combined_profit_factor=round(comb["profit_factor"], 4),
        combined_net_pl=round(comb["net_pl"], 2),
        windows_positive_expectancy_R=n_windows_pos_R,
        permutation_percentile=perc,
        fraction_random_beating_observed=perm["fraction_random_beating_observed_expectancy_R"],
        cost_combined_positive_through_0_10=bool(cost_combined_le_010),
        fraction_positive_months=frac_pos_months,
        best_diagnostic_bracket_pf=round(best_bracket_pf, 4),
    ),
    scoring=reasons,
)
emit("verdict", "classification", "OVERALL", verdict)
emit("verdict", "score", "OVERALL", round(score, 2))

# ===========================================================================
# WRITE OUTPUTS
# ===========================================================================
_lap("writing reports")
csv_path = REPORTS / "btc_edge_validation.csv"
json_path = REPORTS / "btc_edge_validation.json"
txt_path = REPORTS / "btc_edge_validation_summary.txt"

pd.DataFrame(CSV_ROWS, columns=["section", "metric", "scope", "value"]).to_csv(csv_path, index=False)
json_path.write_text(json.dumps(jsonable(RESULTS), indent=2), encoding="utf-8")

L = []
L.append("=" * 78)
L.append("BTC M5 EDGE VALIDATION SUMMARY")
L.append("=" * 78)
L.append(f"engine            : {ec.ENGINE_VERSION}")
L.append(f"data              : {DATA.name}   rows={N}   {df.time.iloc[0]} -> {df.time.iloc[-1]}")
L.append(f"strategy          : EMA {FAST}/{SLOW}/{TREND} | RSI {RSI_P} | ATR {ATR_P} | "
         f"SL {SL_MULT} | TP {TP_MULT}   (NOT optimized)")
L.append(f"costs             : spread floor ${DEFAULT_SPREAD:.2f}, slippage ${SLIPPAGE:.2f}, "
         f"commission ${COMMISSION_PER_LOT:.2f}")
L.append(f"kernel vs engine  : {'EXACT MATCH on TRAIN/VALIDATION/FINAL' if val_ok else 'MISMATCH'}")
L.append("")
L.append("BASELINE (kernel == ec.run_window_logged):")
for nm in ("TRAIN", "VALIDATION", "FINAL"):
    s = base_runs[nm].summary
    L.append(f"  {nm:11s} trades {s['trades']:4d}   net ${s['net_pl']:+10.2f}   "
             f"PF {s['profit_factor']:.3f}   win {s['win_rate']*100:5.2f}%   "
             f"expR {rm[nm]['expectancy_R']:+.3f}")
L.append(f"  {'COMBINED':11s} trades {comb['trades']:4d}   net ${comb['net_pl']:+10.2f}   "
         f"PF {comb['profit_factor']:.3f}   win {comb['win_rate']*100:5.2f}%   "
         f"expR {comb['expectancy_R']:+.3f}   cumR {comb['cumulative_R']:+.2f}")
L.append("")
L.append("1. LONG vs SHORT (combined):")
for side in ("LONG", "SHORT"):
    b = ls[side]
    L.append(f"   {side:5s} n={b['trades']:4d}  win {b['win_rate']*100:5.2f}%  PF {b['profit_factor']:.3f}  "
             f"expR {b['expectancy_R']:+.3f}  net ${b['net_pl']:+.2f}")
L.append("")
L.append("2. R-MULTIPLE (combined):")
L.append(f"   mean R {comb['mean_R']:+.3f}   median R {comb['median_R']:+.3f}   std R {comb['std_R']:.3f}")
L.append(f"   win R  {comb['win_R']:+.3f}   loss R   {comb['loss_R']:+.3f}   expectancy R {comb['expectancy_R']:+.3f}")
L.append(f"   cumulative R {comb['cumulative_R']:+.2f}")
L.append("")
L.append("3. SIGNAL-ONLY BASELINES (combined expectancy R):")
for label, per in sig_base.items():
    L.append(f"   {label:34s} expR {per['COMBINED']['expectancy_R']:+.3f}   "
             f"PF {per['COMBINED']['profit_factor']:.3f}   net ${per['COMBINED']['net_pl']:+.2f}")
L.append("")
L.append("4. COST SENSITIVITY (combined net $ / combined PF):")
for sp in cost_grid:
    row = cost_res[f"{sp:.2f}"]
    tot = sum(row[nm]["profit"] for nm in ("TRAIN", "VALIDATION", "FINAL"))
    tp_ = [row[nm]["trades"] for nm in ("TRAIN", "VALIDATION", "FINAL")]
    L.append(f"   spread ${sp:0.2f}   net ${tot:+10.2f}   "
             f"TRAIN PF {row['TRAIN']['pf']:.3f} / VAL PF {row['VALIDATION']['pf']:.3f} / FINAL PF {row['FINAL']['pf']:.3f}   "
             f"trades {tp_}")
L.append("")
L.append("5. SL/TP STRUCTURE (DIAGNOSTIC ONLY - not optimization):")
for k, row in br_res.items():
    L.append(f"   {k:9s} combined PF {row['COMBINED']['pf']:.3f}  expR {row['COMBINED']['expectancy_R']:+.3f}  "
             f"net ${row['COMBINED']['profit']:+.2f}  (T/V/F PF "
             f"{row['TRAIN']['pf']:.2f}/{row['VALIDATION']['pf']:.2f}/{row['FINAL']['pf']:.2f})")
L.append("")
L.append("6. REGIME (expectancy R / PF / n):")
for grp in ("atr", "trend_strength_ema_sep", "rsi", "direction"):
    L.append(f"   [{grp}]")
    for lab, b in reg[grp].items():
        L.append(f"      {lab:10s} n={b['trades']:4d}  PF {b.get('profit_factor',0):.3f}  "
                 f"expR {b.get('expectancy_R',0):+.3f}")
L.append(f"   note: {reg['_note_rsi']}")
L.append("")
L.append("7. TEMPORAL STABILITY (monthly):")
for mon, v in temporal.items():
    if mon.startswith("_"):
        continue
    L.append(f"   {mon}  n={v['trades']:3d}  net ${v['profit']:+9.2f}  PF {v['profit_factor']:.3f}  "
             f"win {v['win_rate']*100:5.1f}%  expR {v['expectancy_R']:+.3f}")
L.append(f"   -> {temporal['_summary']['blocks_positive_expectancy_R']}/{temporal['_summary']['blocks']} "
         f"months positive expectancy R ({temporal['_summary']['fraction_positive']*100:.0f}%)")
L.append("")
L.append(f"8. PERMUTATION TEST ({N_PERM} random-direction trials, same execution model):")
L.append(f"   observed combined expectancy R : {perm['observed_expectancy_R']:+.4f}")
L.append(f"   random expectancy R            : mean {perm['random_expectancy_R_mean']:+.4f}  "
         f"std {perm['random_expectancy_R_std']:.4f}")
L.append(f"   observed percentile            : {perm['observed_percentile_expectancy_R']:.1f}")
L.append(f"   fraction of random beating obs  : {perm['fraction_random_beating_observed_expectancy_R']:.3f}")
L.append(f"   observed profit percentile      : {perm['observed_percentile_profit']:.1f}  "
         f"(random beat obs: {perm['fraction_random_beating_observed_profit']:.3f})")
L.append("")
L.append("9. DATA QUALITY / SPREAD:")
L.append(f"   raw spread points distribution : {dq['raw_points_distribution']}")
L.append(f"   modal spread                   : {dq['modal_points']:.0f} pt = ${dq['modal_points']*TICK:.2f}")
L.append(f"   bars > 10 pts / > 100 pts      : {dq['n_bars_spread_gt_10pts']} / {dq['n_bars_corrupt_gt_100pts']}")
for r_ in dq["abnormal_bars"]:
    L.append(f"      bar {r_['bar']:6d}  {r_['time']}  raw {r_['raw_points']:5.0f} pts = "
             f"${r_['converted_price']:6.2f} -> clamped ${r_['post_clamp']:.2f}  "
             f"[{r_['classification']:11s}]  in_trade={r_['used_in_a_trade']}"
             + (f" ({r_['trade_role']})" if r_['trade_role'] else ""))
for a_ in dq["affected_trades"]:
    L.append(f"      -> affected trade: {a_['split']} dir={a_['dir']} lots={a_['lots']:.0f} "
             f"pnl ${a_['pnl']:+.2f}  extra cost vs modal spread ${a_['extra_cost_vs_modal_spread']:+.2f}")
L.append(f"   total distortion vs modal-spread fills : ${dq['total_distortion_dollars']:+.2f} "
         f"({dq['total_distortion_pct_of_gross_pnl']}% of gross P/L, pessimistic direction)")
L.append(f"   clamp creates unrealistic (favourable) execution : {dq['clamp_creates_unrealistic_fill']}")
L.append(f"   {dq['clamp_note']}")
L.append("")
L.append("=" * 78)
L.append("SCORING (TRAIN + VALIDATION + FINAL together):")
for r_ in reasons:
    L.append("   " + r_)
L.append(f"   SCORE = {score:.2f} / 11.5")
L.append("")
L.append(f"   VERDICT :  {verdict}")
L.append("=" * 78)
txt_path.write_text("\n".join(L) + "\n", encoding="utf-8")

print("\n".join(L))
_lap("done")
print("\nFILES CREATED:")
for p in (Path(__file__).resolve(), csv_path, json_path, txt_path):
    print(f"  {p.relative_to(ROOT)}")
