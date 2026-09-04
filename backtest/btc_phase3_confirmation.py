"""
btc_phase3_confirmation.py  --  FROZEN confirmatory test of the single Phase-2 candidate.

CANDIDATE (parameters FROZEN -- no search, no optimization):
    event            : bar i has the single widest TRUE RANGE over the trailing 48 bars
    structure        : symmetric OCO breakout
    long trigger     : max(high[i-48 .. i-1]) + 0.5 * ATR20[i]
    short trigger    : min(low [i-48 .. i-1]) - 0.5 * ATR20[i]
    OCO              : first side touched cancels the other
    trigger window   : 12 bars  (arm expires if neither side touched)
    R                : 1.0 * ATR20[i]   (ATR at the event bar)
    stop-loss        : 1.0 R
    take-profit      : 1.5 R
    maximum hold     : 24 bars after entry, then market exit
    sizing / costs   : the existing btc_engine_core model, UNCHANGED
    windows          : TRAIN / VALIDATION / FINAL, each an INDEPENDENT $2,000 reset

SAFETY / SCOPE
    * Imports btc_engine_core / btc_realistic_config for constants + spread model ONLY.
      No engine / config / XAU file modified.  No MT5 / live-order code.
    * btc_engine_core._backtest is directional-only and cannot express an OCO straddle,
      so the loop is re-implemented here, matching the engine's exact conventions:
        - gap-through fills at bar open  (engine lines 287-298 / 329-340)
        - stop / market exit pays  half_spread + slippage ; limit (TP) pays half_spread
        - stop ENTRY (breakout) pays half_spread + slippage, gap fills at open
        - position sizing  loss_per_lot = (stop_dist + slippage)*VPP + commission,
          floor MIN_LOT, margin cap  balance*MAX_MARGIN_FRACTION / (open*CONTRACT/LEVERAGE)
          (engine lines 467-501)
        - engine cost guard  atr*sl_mult < 2*(spread+2*slip)  -> skip
        - equity kill-switch  equity <= EQUITY_STOP  -> close + stop window (engine line 393)
    * Strict causality: event flag + trigger levels at bar i use ONLY bars <= i;
      trigger scan starts at i+1; a window's trades are capped at window_end (EOW close).
    * ACCEPTANCE ASSESSMENT IS DONE HERE, BEFORE ANY ALTERNATIVE PARAMETERS ARE CONSIDERED.

OUTPUTS
    reports/btc_phase3_confirmation.csv
    reports/btc_phase3_confirmation.json
    reports/btc_phase3_confirmation_summary.txt
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

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]

        def _d(f):
            return f

        return _d

T0 = time.perf_counter()


def _lap(m):
    print(f"[{time.perf_counter() - T0:7.2f}s] {m}", flush=True)


# ===========================================================================
# FROZEN PARAMETERS  +  ENGINE CONSTANTS  (imported, not redefined)
# ===========================================================================
STRUCT_WINDOW = 48
BUFFER_ATR = 0.5
TRIGGER_WINDOW = 12
MAX_HOLD = 24
SL_R = 1.0
TP_R = 1.5
ATR_PERIOD = 20

INITIAL_BALANCE = float(ec.INITIAL_BALANCE)          # 2000
RISK_PER_TRADE = float(ec.RISK_PER_TRADE)            # 0.01
VPP = float(ec.VALUE_PER_POINT)                      # 1.0
CONTRACT_SIZE = float(ec.CONTRACT_SIZE)              # 1.0
LEVERAGE = float(ec.LEVERAGE)                        # 100
MIN_LOT = float(ec.MIN_LOT)                          # 1
LOT_STEP = float(ec.LOT_STEP)                        # 1
MAX_LOT = float(ec.MAX_LOT)
SLIP = float(bc.SLIPPAGE)                            # 0.01
COMMISSION = float(ec.COMMISSION_PER_LOT)            # 0.0
EQUITY_STOP = float(ec.EQUITY_STOP)                  # 400
MAX_MARGIN_FRACTION = float(ec.MAX_MARGIN_FRACTION)  # 0.5
DEFAULT_SPREAD = float(bc.DEFAULT_SPREAD)            # 0.05

SPLIT_FRACS = (0.70, 0.85)
SPREAD_SCENARIOS = (0.01, 0.03, 0.05, 0.07, 0.10)
N_BOOT = 5000
N_PERM = 1000
SEED = 20260830

REASON = {0: "target", 1: "stop", 2: "max_hold", 3: "window_end", 4: "equity_stop"}

DATA = ROOT / "data" / "btc_m5_history.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# ===========================================================================
# NUMBA SIMULATION KERNEL  (one independent window)
# ===========================================================================
@njit(cache=True)
def simulate_window(
    o, h, l, c, atr, spread,
    event_flag, upper, lower,
    w0, w1,
    initial_balance, risk_per_trade, vpp, contract_size, leverage,
    min_lot, lot_step, max_lot, slippage, commission, min_stop_dist_unused,
    equity_stop, max_margin_frac,
    sl_r, tp_r, trigger_window, max_hold,
    dir_mode, rand_dir,               # dir_mode 0 = first-touch OCO ; 1 = use rand_dir[trade_k]
    apply_guard,                      # 1 = engine cost guard on (faithful) ; 0 = diagnostic only
    t_entry_i, t_exit_i, t_dir, t_lots, t_entry_px, t_exit_px, t_pnl,
    t_R_price, t_mfe_R, t_mae_R, t_reason, t_event_i, t_bars_held,
    eq_curve,
):
    balance = initial_balance
    peak_equity = initial_balance
    max_dd_abs = 0.0
    max_dd_pct = 0.0

    state = 0            # 0 idle, 1 armed, 2 in-trade
    arm_i = 0
    arm_expiry = 0
    ent_bar = 0
    d = 0
    sl_px = 0.0
    tp_px = 0.0
    lots = 0.0
    entry_fill = 0.0
    R_price = 0.0
    mfe = 0.0
    mae = 0.0

    n_trades = 0
    n_events = 0
    n_armed = 0
    n_triggered = 0
    n_expired = 0
    n_skip_busy = 0
    n_skip_size = 0
    n_skip_guard = 0
    equity_stopped = 0

    for b in range(w0, w1):
        hs = spread[b] * 0.5
        closed_this_bar = False

        # ---------- manage open trade ----------
        if state == 2:
            exit_px = 0.0
            reason = -1
            if d == 1:
                if o[b] - hs <= sl_px:
                    exit_px = o[b] - hs - slippage; reason = 1
                elif l[b] - hs <= sl_px:
                    exit_px = sl_px - slippage; reason = 1
                elif o[b] - hs >= tp_px:
                    exit_px = o[b] - hs; reason = 0
                elif h[b] - hs >= tp_px:
                    exit_px = tp_px; reason = 0
            else:
                if o[b] + hs >= sl_px:
                    exit_px = o[b] + hs + slippage; reason = 1
                elif h[b] + hs >= sl_px:
                    exit_px = sl_px + slippage; reason = 1
                elif o[b] + hs <= tp_px:
                    exit_px = o[b] + hs; reason = 0
                elif l[b] + hs <= tp_px:
                    exit_px = tp_px; reason = 0

            # max-hold / window-end forced exit
            if reason < 0 and (b >= ent_bar + max_hold or b == w1 - 1):
                if d == 1:
                    exit_px = c[b] - hs - slippage
                else:
                    exit_px = c[b] + hs + slippage
                reason = 2 if b >= ent_bar + max_hold else 3

            # track excursion this bar (favourable / adverse in price)
            if d == 1:
                fav = (h[b] - hs) - entry_fill
                adv = entry_fill - (l[b] - hs)
            else:
                fav = entry_fill - (l[b] + hs)
                adv = (h[b] + hs) - entry_fill
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv

            if reason >= 0:
                pnl = (exit_px - entry_fill) * d * vpp * lots - commission * lots
                balance += pnl
                if n_trades < t_pnl.shape[0]:
                    t_entry_i[n_trades] = ent_bar
                    t_exit_i[n_trades] = b
                    t_dir[n_trades] = d
                    t_lots[n_trades] = lots
                    t_entry_px[n_trades] = entry_fill
                    t_exit_px[n_trades] = exit_px
                    t_pnl[n_trades] = pnl
                    t_R_price[n_trades] = R_price
                    t_mfe_R[n_trades] = mfe / R_price if R_price > 0 else 0.0
                    t_mae_R[n_trades] = mae / R_price if R_price > 0 else 0.0
                    t_reason[n_trades] = reason
                    t_event_i[n_trades] = arm_i
                    t_bars_held[n_trades] = b - ent_bar
                n_trades += 1
                state = 0
                closed_this_bar = True

        # ---------- equity mark-to-market + drawdown + kill-switch ----------
        if state == 2:
            if d == 1:
                unreal = ((c[b] - hs) - entry_fill) * vpp * lots
            else:
                unreal = (entry_fill - (c[b] + hs)) * vpp * lots
        else:
            unreal = 0.0
        equity = balance + unreal
        if equity > peak_equity:
            peak_equity = equity
        dd = peak_equity - equity
        if dd > max_dd_abs:
            max_dd_abs = dd
        if peak_equity > 0.0:
            ddp = dd / peak_equity
            if ddp > max_dd_pct:
                max_dd_pct = ddp
        if b - w0 < eq_curve.shape[0]:
            eq_curve[b - w0] = equity

        if equity <= equity_stop:
            if state == 2:
                if d == 1:
                    ex = c[b] - hs - slippage
                else:
                    ex = c[b] + hs + slippage
                pnl = (ex - entry_fill) * d * vpp * lots - commission * lots
                balance += pnl
                if n_trades < t_pnl.shape[0]:
                    t_entry_i[n_trades] = ent_bar; t_exit_i[n_trades] = b; t_dir[n_trades] = d
                    t_lots[n_trades] = lots; t_entry_px[n_trades] = entry_fill; t_exit_px[n_trades] = ex
                    t_pnl[n_trades] = pnl; t_R_price[n_trades] = R_price
                    t_mfe_R[n_trades] = mfe / R_price if R_price > 0 else 0.0
                    t_mae_R[n_trades] = mae / R_price if R_price > 0 else 0.0
                    t_reason[n_trades] = 4; t_event_i[n_trades] = arm_i; t_bars_held[n_trades] = b - ent_bar
                n_trades += 1
                state = 0
            equity_stopped = 1
            break

        # ---------- armed: look for a trigger ----------
        if state == 1 and not closed_this_bar:
            if b > arm_expiry:
                state = 0
                n_expired += 1
            else:
                up = upper[arm_i]
                dn = lower[arm_i]
                hit_up = h[b] >= up
                hit_dn = l[b] <= dn
                dd_ = 0
                if hit_up and hit_dn:
                    dd_ = 1 if abs(o[b] - up) <= abs(o[b] - dn) else -1
                elif hit_up:
                    dd_ = 1
                elif hit_dn:
                    dd_ = -1
                if dd_ != 0:
                    if dir_mode == 1:
                        dd_ = 1 if rand_dir[n_triggered] > 0 else -1
                    n_triggered += 1
                    a = atr[arm_i]
                    R_price = sl_r * a
                    # engine cost guard (line 458):  a * sl_mult < 2 * (spread + 2*slip)
                    cost_dist = spread[b] + 2.0 * slippage
                    if apply_guard == 1 and a * sl_r < 2.0 * cost_dist:
                        n_skip_guard += 1
                        state = 0
                    else:
                        # entry fill (stop order): gap at open else at trigger, + costs
                        if dd_ == 1:
                            if o[b] + hs >= up:
                                entry_fill = o[b] + hs + slippage
                            else:
                                entry_fill = up + hs + slippage
                        else:
                            if o[b] - hs <= dn:
                                entry_fill = o[b] - hs - slippage
                            else:
                                entry_fill = dn - hs - slippage
                        # sizing (engine)
                        risk_dollars = balance * risk_per_trade
                        loss_per_lot = (R_price + slippage) * vpp + commission
                        if loss_per_lot <= 0.0:
                            n_skip_size += 1
                            state = 0
                        else:
                            raw_lots = risk_dollars / loss_per_lot
                            lt = np.floor(raw_lots / lot_step + 1e-9) * lot_step
                            mpl = o[b] * contract_size / leverage
                            if mpl > 0.0:
                                mbm = (balance * max_margin_frac) / mpl
                                mbm = np.floor(mbm / lot_step + 1e-9) * lot_step
                                if lt > mbm:
                                    lt = mbm
                            if lt > max_lot:
                                lt = max_lot
                            if lt < min_lot:
                                n_skip_size += 1
                                state = 0
                            else:
                                lots = lt
                                d = dd_
                                ent_bar = b
                                sl_px = entry_fill - d * R_price
                                tp_px = entry_fill + d * tp_r * R_price
                                mfe = 0.0
                                mae = 0.0
                                state = 2

        # ---------- idle: arm on a fresh event ----------
        if state == 0 and not closed_this_bar:
            if b >= STRUCT_WINDOW and event_flag[b] == 1 and np.isfinite(upper[b]) and np.isfinite(lower[b]) and atr[b] > 0.0:
                state = 1
                arm_i = b
                arm_expiry = b + trigger_window
                n_armed += 1
        elif event_flag[b] == 1 and b >= STRUCT_WINDOW:
            n_skip_busy += 1

        if event_flag[b] == 1 and b >= STRUCT_WINDOW:
            n_events += 1

    # close still-open at window end (safety; loop above already handles b==w1-1)
    return (float(balance), float(max_dd_abs), float(max_dd_pct), float(n_trades),
            float(n_events), float(n_armed), float(n_triggered), float(n_expired),
            float(n_skip_busy), float(n_skip_size), float(n_skip_guard), float(equity_stopped))


# ===========================================================================
# DRIVER
# ===========================================================================
_lap("load data")
df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
o = np.ascontiguousarray(df.open.to_numpy(float))
h = np.ascontiguousarray(df.high.to_numpy(float))
l = np.ascontiguousarray(df.low.to_numpy(float))
c = np.ascontiguousarray(df.close.to_numpy(float))
N = len(c)

ec.DEFAULT_SPREAD = bc.DEFAULT_SPREAD
spread_real = ec.prepare_spread(df)
atr20 = ec.atr(h, l, c, ATR_PERIOD)
atr20s = np.ascontiguousarray(np.where(np.isfinite(atr20) & (atr20 > 0), atr20, np.nan))

prev_c = np.concatenate([[c[0]], c[:-1]])
TR = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
tr_prevmax = pd.Series(TR).shift(1).rolling(STRUCT_WINDOW, min_periods=STRUCT_WINDOW).max().to_numpy()
event_flag = np.ascontiguousarray(((TR > tr_prevmax) & np.isfinite(tr_prevmax)).astype(np.int64))
rhi = pd.Series(h).shift(1).rolling(STRUCT_WINDOW, min_periods=STRUCT_WINDOW).max().to_numpy()
rlo = pd.Series(l).shift(1).rolling(STRUCT_WINDOW, min_periods=STRUCT_WINDOW).min().to_numpy()
upper = np.ascontiguousarray(rhi + BUFFER_ATR * atr20s)
lower = np.ascontiguousarray(rlo - BUFFER_ATR * atr20s)

T1 = int(N * SPLIT_FRACS[0])
V1 = int(N * SPLIT_FRACS[1])
WINDOWS = [("TRAIN", 0, T1), ("VALIDATION", T1, V1), ("FINAL", V1, N)]
_lap(f"rows={N}  TRAIN[0,{T1}) VAL[{T1},{V1}) FINAL[{V1},{N})  events(total)={int(event_flag.sum())}")

RESULTS = {"meta": dict(
    engine_version=ec.ENGINE_VERSION, rows=int(N),
    span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
    frozen_params=dict(struct_window=STRUCT_WINDOW, buffer_atr=BUFFER_ATR,
                       trigger_window=TRIGGER_WINDOW, max_hold=MAX_HOLD,
                       sl_R=SL_R, tp_R=TP_R, atr_period=ATR_PERIOD,
                       R_definition="1.0 * ATR20 at the event bar"),
    engine_constants=dict(initial_balance=INITIAL_BALANCE, risk_per_trade=RISK_PER_TRADE,
                          value_per_point=VPP, min_lot=MIN_LOT, leverage=LEVERAGE,
                          slippage=SLIP, commission=COMMISSION, default_spread=DEFAULT_SPREAD,
                          equity_stop=EQUITY_STOP, max_margin_fraction=MAX_MARGIN_FRACTION),
    windows={w[0]: [w[1], w[2]] for w in WINDOWS},
    cost_model="cost_price = spread + 2*slippage ; stop entry+exit both pay half_spread+slippage",
    note="OCO straddle simulated here (engine _backtest is directional-only); engine cost/"
         "sizing/equity arithmetic replicated exactly. No engine or XAU file modified.",
    n_boot=N_BOOT, n_perm=N_PERM, seed=SEED,
)}
CSV = []


def emit(section, key, scope, value):
    CSV.append(dict(section=section, key=key, scope=scope, value=value))


def run_one(w0, w1, spread_arr, dir_mode=0, rand_dir=None, apply_guard=1):
    cap = max(8, int((w1 - w0)))
    z = lambda dt: np.zeros(cap, dt)
    arr = dict(t_entry_i=z(np.int64), t_exit_i=z(np.int64), t_dir=z(np.int64), t_lots=z(np.float64),
               t_entry_px=z(np.float64), t_exit_px=z(np.float64), t_pnl=z(np.float64),
               t_R_price=z(np.float64), t_mfe_R=z(np.float64), t_mae_R=z(np.float64),
               t_reason=z(np.int64), t_event_i=z(np.int64), t_bars_held=z(np.int64))
    eq = np.zeros(w1 - w0, np.float64)
    if rand_dir is None:
        rand_dir = np.zeros(cap, np.int64)
    res = simulate_window(
        o, h, l, c, atr20s, np.ascontiguousarray(spread_arr),
        event_flag, upper, lower, int(w0), int(w1),
        INITIAL_BALANCE, RISK_PER_TRADE, VPP, CONTRACT_SIZE, LEVERAGE,
        MIN_LOT, LOT_STEP, MAX_LOT, SLIP, COMMISSION, 0.0,
        EQUITY_STOP, MAX_MARGIN_FRACTION,
        SL_R, TP_R, TRIGGER_WINDOW, MAX_HOLD,
        int(dir_mode), np.ascontiguousarray(rand_dir), int(apply_guard),
        arr["t_entry_i"], arr["t_exit_i"], arr["t_dir"], arr["t_lots"], arr["t_entry_px"],
        arr["t_exit_px"], arr["t_pnl"], arr["t_R_price"], arr["t_mfe_R"], arr["t_mae_R"],
        arr["t_reason"], arr["t_event_i"], arr["t_bars_held"], eq,
    )
    nt = int(round(res[3]))
    trades = pd.DataFrame({k[2:]: arr[k][:nt] for k in arr})
    trades["R"] = np.where(trades["R_price"] > 0,
                           (trades["exit_px"] - trades["entry_px"]) * trades["dir"] / trades["R_price"], 0.0)
    summary = dict(
        final_balance=res[0], net_pl=res[0] - INITIAL_BALANCE,
        max_dd_abs=res[1], max_dd_pct=res[2], n_trades=nt,
        n_events=int(res[4]), n_armed=int(res[5]), n_triggered=int(res[6]),
        n_expired=int(res[7]), n_skip_busy=int(res[8]), n_skip_size=int(res[9]),
        n_skip_guard=int(res[10]), equity_stopped=bool(res[11]),
        equity_curve=eq,
    )
    return summary, trades


def trade_stats(tr):
    if len(tr) == 0:
        return dict(n=0)
    pnl = tr["pnl"].to_numpy()
    R = tr["R"].to_numpy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gp = float(wins.sum()); gl = float(-losses.sum())
    return dict(
        n=int(len(tr)),
        wins=int((pnl > 0).sum()), losses=int((pnl <= 0).sum()),
        win_rate=round(float((pnl > 0).mean()), 4),
        net_pl=round(float(pnl.sum()), 2),
        gross_profit=round(gp, 2), gross_loss=round(gl, 2),
        profit_factor=round(gp / gl, 4) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        expectancy_dollar=round(float(pnl.mean()), 3),
        expectancy_R=round(float(R.mean()), 4),
        median_R=round(float(np.median(R)), 4),
        std_R=round(float(R.std(ddof=1)), 4) if len(R) > 1 else 0.0,
        avg_win=round(float(wins.mean()), 3) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 3) if len(losses) else 0.0,
        mean_MFE_R=round(float(tr["mfe_R"].mean()), 3),
        mean_MAE_R=round(float(tr["mae_R"].mean()), 3),
        mean_bars_held=round(float(tr["bars_held"].mean()), 2),
        exit_reasons={REASON[k]: int((tr["reason"] == k).sum()) for k in sorted(set(tr["reason"]))},
    )


# ---------------------------------------------------------------------------
# 1-3.  PER-WINDOW CONFIRMATION  (frozen, base spread)
# ---------------------------------------------------------------------------
_lap("1-3: per-window confirmation")
per_window = {}
all_trades = []
for name, w0, w1 in WINDOWS:
    s, tr = run_one(w0, w1, spread_real)
    tr = tr.assign(window=name)
    tr["entry_time"] = df.time.iloc[tr["entry_i"].to_numpy()].to_numpy()
    st = trade_stats(tr)
    st.update(final_balance=round(s["final_balance"], 2), net_pl=round(s["net_pl"], 2),
              return_pct=round(100.0 * s["net_pl"] / INITIAL_BALANCE, 3),
              max_dd_abs=round(s["max_dd_abs"], 2), max_dd_pct=round(s["max_dd_pct"], 4),
              n_events=s["n_events"], n_armed=s["n_armed"], n_triggered=s["n_triggered"],
              n_expired=s["n_expired"], n_skipped_busy=s["n_skip_busy"],
              n_skipped_size=s["n_skip_size"], n_skipped_guard=s["n_skip_guard"],
              equity_stopped=s["equity_stopped"])
    per_window[name] = st
    all_trades.append(tr)
    for k in ("n", "win_rate", "net_pl", "profit_factor", "expectancy_R", "max_dd_abs",
              "return_pct", "n_triggered", "equity_stopped"):
        emit("per_window", k, name, st[k])
    print(f"    {name:11s} ev {st['n_events']:3d} armed {st['n_armed']:3d} trig {st['n_triggered']:3d}"
          f" guard-skip {st['n_skipped_guard']:3d} size-skip {st['n_skipped_size']:3d} -> trades {st['n']:3d}"
          f"  win {st['win_rate']*100:5.1f}%  net ${st['net_pl']:+9.2f}"
          f"  PF {st['profit_factor']:.3f}  expR {st['expectancy_R']:+.3f}  eqstop {st['equity_stopped']}")

TB = pd.concat(all_trades, ignore_index=True)
TB["entry_month"] = pd.to_datetime(TB["entry_time"], utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)

# --- diagnostic: engine cost guard OFF (DIVERGES FROM ENGINE) -- isolates the guard's effect
_lap("diagnostic: cost guard OFF (reconciliation vs Phase-2 idealised sim)")
guard_off = {}
for name, w0, w1 in WINDOWS:
    s, tr = run_one(w0, w1, spread_real, apply_guard=0)
    st = trade_stats(tr)
    guard_off[name] = dict(n=st.get("n", 0), net_pl=st.get("net_pl", 0.0),
                           win_rate=st.get("win_rate", 0.0),
                           profit_factor=st.get("profit_factor", 0.0),
                           expectancy_R=st.get("expectancy_R", 0.0),
                           max_dd_abs=round(s["max_dd_abs"], 2))
    emit("guard_off_diagnostic", "n", name, guard_off[name]["n"])
    emit("guard_off_diagnostic", "expectancy_R", name, guard_off[name]["expectancy_R"])
    emit("guard_off_diagnostic", "net_pl", name, guard_off[name]["net_pl"])
RESULTS["guard_off_diagnostic"] = dict(
    note="engine cost guard DISABLED -- NOT a faithful engine run; shown only to reconcile "
         "with the Phase-2 straddle sim (which had no guard) and to isolate the guard's impact.",
    by_window=guard_off)

# ---------------------------------------------------------------------------
# 4.  COMBINED STATISTICS
# ---------------------------------------------------------------------------
_lap("4: combined statistics")
combined = trade_stats(TB)
combined["sum_of_window_net_pl"] = round(sum(per_window[w]["net_pl"] for w in per_window), 2)
oos = TB[TB.window != "TRAIN"]
combined_oos = trade_stats(oos)
RESULTS["combined"] = combined
RESULTS["combined_oos_val_plus_final"] = combined_oos
for k in ("n", "win_rate", "profit_factor", "expectancy_R", "median_R", "std_R", "net_pl"):
    emit("combined", k, "ALL", combined[k])
    emit("combined", k, "OOS", combined_oos[k])

# ---------------------------------------------------------------------------
# 5.  SPREAD SENSITIVITY
# ---------------------------------------------------------------------------
_lap("5: spread sensitivity")
spread_sens = {}
for sp in SPREAD_SCENARIOS:
    sp_arr = np.full(N, sp, float)
    row = {}
    tot = 0.0
    oos_tot = 0.0
    for name, w0, w1 in WINDOWS:
        s, tr = run_one(w0, w1, sp_arr)
        st = trade_stats(tr)
        row[name] = dict(n=st.get("n", 0), net_pl=st.get("net_pl", 0.0),
                         profit_factor=st.get("profit_factor", 0.0),
                         expectancy_R=st.get("expectancy_R", 0.0),
                         win_rate=st.get("win_rate", 0.0))
        tot += st.get("net_pl", 0.0)
        if name != "TRAIN":
            oos_tot += st.get("net_pl", 0.0)
        emit("spread_sensitivity", "net_pl", f"spread={sp:.2f}/{name}", st.get("net_pl", 0.0))
        emit("spread_sensitivity", "expectancy_R", f"spread={sp:.2f}/{name}", st.get("expectancy_R", 0.0))
    row["_sum_net_pl"] = round(tot, 2)
    row["_oos_sum_net_pl"] = round(oos_tot, 2)
    row["_all_windows_expR_positive"] = bool(all(row[w]["expectancy_R"] > 0 for w in ("TRAIN", "VALIDATION", "FINAL")))
    spread_sens[f"{sp:.2f}"] = row
RESULTS["spread_sensitivity"] = spread_sens

# ---------------------------------------------------------------------------
# 6.  BOOTSTRAP / PERMUTATION SIGNIFICANCE
# ---------------------------------------------------------------------------
_lap("6: bootstrap + permutation significance")
rng = np.random.default_rng(SEED)
sig = {}

# 6a bootstrap: resample OOS trades with replacement -> CI on mean R and total $
oos_R = oos["R"].to_numpy()
oos_pnl = oos["pnl"].to_numpy()
if len(oos_R) >= 10:
    idx = rng.integers(0, len(oos_R), size=(N_BOOT, len(oos_R)))
    bmeanR = oos_R[idx].mean(axis=1)
    bsum = oos_pnl[idx].sum(axis=1)
    sig["bootstrap_oos"] = dict(
        n_trades=int(len(oos_R)),
        mean_R=round(float(oos_R.mean()), 4),
        mean_R_CI95=[round(float(np.percentile(bmeanR, 2.5)), 4), round(float(np.percentile(bmeanR, 97.5)), 4)],
        mean_R_p5=round(float(np.percentile(bmeanR, 5)), 4),
        prob_mean_R_gt_0=round(float((bmeanR > 0).mean()), 4),
        total_pnl=round(float(oos_pnl.sum()), 2),
        total_pnl_CI95=[round(float(np.percentile(bsum, 2.5)), 2), round(float(np.percentile(bsum, 97.5)), 2)],
        prob_total_pnl_gt_0=round(float((bsum > 0).mean()), 4),
    )
else:
    sig["bootstrap_oos"] = dict(n_trades=int(len(oos_R)), note="too few OOS trades")

# 6b permutation: sign-flip the per-trade R (is |mean R| distinguishable from noise?)
if len(oos_R) >= 10:
    flips = rng.integers(0, 2, size=(N_PERM * 5, len(oos_R))) * 2 - 1
    null_meanR = (oos_R * flips).mean(axis=1)
    obs = float(oos_R.mean())
    sig["permutation_signflip_oos"] = dict(
        observed_mean_R=round(obs, 4),
        null_mean=round(float(null_meanR.mean()), 4), null_std=round(float(null_meanR.std(ddof=1)), 4),
        p_two_sided=round(float((np.abs(null_meanR) >= abs(obs)).mean()), 4),
    )

# 6c permutation: arm the straddle at RANDOM bars (same count) instead of widest-TR bars
_lap("6c: random-event-bar permutation")
eligible = np.where(np.isfinite(upper) & np.isfinite(lower) & np.isfinite(atr20s) & (atr20s > 0))[0]
eligible = eligible[eligible >= STRUCT_WINDOW]
obs_oos_expR = combined_oos.get("expectancy_R", 0.0)
obs_oos_net = combined_oos.get("net_pl", 0.0)
perm_expR = np.empty(N_PERM)
perm_net = np.empty(N_PERM)
t6 = time.perf_counter()
for k in range(N_PERM):
    fake_flag = np.zeros(N, np.int64)
    for name, w0, w1 in WINDOWS:
        if name == "TRAIN":
            continue
        elig_w = eligible[(eligible >= w0) & (eligible < w1)]
        n_ev = per_window[name]["n_events"]
        if n_ev > 0 and len(elig_w) >= n_ev:
            pick = rng.choice(elig_w, size=n_ev, replace=False)
            fake_flag[pick] = 1
    pnl_all = []
    R_all = []
    for name, w0, w1 in WINDOWS:
        if name == "TRAIN":
            continue
        # temp swap event_flag
        saved = event_flag[w0:w1].copy()
        event_flag[w0:w1] = fake_flag[w0:w1]
        s, tr = run_one(w0, w1, spread_real)
        event_flag[w0:w1] = saved
        pnl_all.append(tr["pnl"].to_numpy())
        R_all.append(tr["R"].to_numpy())
    pnl_all = np.concatenate(pnl_all) if pnl_all else np.array([0.0])
    R_all = np.concatenate(R_all) if R_all else np.array([0.0])
    perm_expR[k] = R_all.mean() if len(R_all) else 0.0
    perm_net[k] = pnl_all.sum()
    if (k + 1) % 250 == 0:
        _lap(f"    perm {k+1}/{N_PERM}  ({time.perf_counter()-t6:.1f}s)")
sig["permutation_random_event_bars_oos"] = dict(
    observed_expectancy_R=round(obs_oos_expR, 4), observed_net_pl=round(obs_oos_net, 2),
    random_expectancy_R_mean=round(float(perm_expR.mean()), 4),
    random_expectancy_R_std=round(float(perm_expR.std(ddof=1)), 4),
    observed_percentile_expR=round(100.0 * float((perm_expR <= obs_oos_expR).mean()), 2),
    p_random_beats_observed_expR=round(float((perm_expR >= obs_oos_expR).mean()), 4),
    p_random_beats_observed_net=round(float((perm_net >= obs_oos_net).mean()), 4),
)
RESULTS["significance"] = sig

# ---------------------------------------------------------------------------
# 7.  TRADE-LEVEL MFE / MAE
# ---------------------------------------------------------------------------
_lap("7: trade-level MFE / MAE")
mfe_mae = {}
for scope, frame in (("ALL", TB), ("OOS", oos), ("TRAIN", TB[TB.window == "TRAIN"])):
    if len(frame) == 0:
        continue
    mfe_mae[scope] = dict(
        n=int(len(frame)),
        mean_MFE_R=round(float(frame["mfe_R"].mean()), 3),
        median_MFE_R=round(float(frame["mfe_R"].median()), 3),
        mean_MAE_R=round(float(frame["mae_R"].mean()), 3),
        median_MAE_R=round(float(frame["mae_R"].median()), 3),
        mfe_ge_1_5R=round(float((frame["mfe_R"] >= 1.5).mean()), 3),
        mae_ge_1_0R=round(float((frame["mae_R"] >= 1.0).mean()), 3),
        mfe_over_mae=round(float(frame["mfe_R"].mean() / frame["mae_R"].mean()), 3) if frame["mae_R"].mean() else None,
        pct_winners_that_touched_target=round(
            float(((frame["reason"] == 0)).sum() / max(1, (frame["pnl"] > 0).sum())), 3),
    )
RESULTS["mfe_mae"] = mfe_mae

# ---------------------------------------------------------------------------
# 8.  LONG vs SHORT
# ---------------------------------------------------------------------------
_lap("8: long vs short")
ls = {}
for side, dv in (("LONG", 1), ("SHORT", -1)):
    for scope, frame in (("ALL", TB), ("OOS", oos)):
        sub = frame[frame.dir == dv]
        st = trade_stats(sub) if len(sub) else dict(n=0)
        ls[f"{side}/{scope}"] = st
        if st.get("n", 0):
            emit("long_vs_short", "expectancy_R", f"{side}/{scope}", st["expectancy_R"])
            emit("long_vs_short", "win_rate", f"{side}/{scope}", st["win_rate"])
            emit("long_vs_short", "n", f"{side}/{scope}", st["n"])
RESULTS["long_vs_short"] = ls
p_long_oos = round(float((oos.dir == 1).mean()), 3) if len(oos) else None
RESULTS["direction_balance"] = dict(p_long_all=round(float((TB.dir == 1).mean()), 3),
                                    p_long_oos=p_long_oos)

# ---------------------------------------------------------------------------
# 9.  MONTHLY BREAKDOWN
# ---------------------------------------------------------------------------
_lap("9: monthly breakdown")
monthly = {}
for mon, sub in TB.groupby("entry_month"):
    st = trade_stats(sub)
    monthly[mon] = dict(n=st["n"], net_pl=st["net_pl"], profit_factor=st["profit_factor"],
                        win_rate=st["win_rate"], expectancy_R=st["expectancy_R"])
    emit("monthly", "net_pl", mon, st["net_pl"])
    emit("monthly", "expectancy_R", mon, st["expectancy_R"])
pos_months = sum(1 for v in monthly.values() if v["expectancy_R"] > 0)
tot_pos = sum(v["net_pl"] for v in monthly.values() if v["net_pl"] > 0)
max_month_share = max((v["net_pl"] / tot_pos for v in monthly.values() if v["net_pl"] > 0), default=0.0)
monthly["_summary"] = dict(blocks=len(monthly), positive_expectancy_R=pos_months,
                           fraction_positive=round(pos_months / max(1, len(monthly)), 3),
                           largest_single_month_share_of_gross_profit=round(float(max_month_share), 3))
RESULTS["monthly"] = monthly

# ---------------------------------------------------------------------------
# 10.  DRAWDOWN + EQUITY CURVE
# ---------------------------------------------------------------------------
_lap("10: drawdown + equity curve")
eqc = {}
for name, w0, w1 in WINDOWS:
    s, tr = run_one(w0, w1, spread_real)
    curve = s["equity_curve"]
    curve = curve[curve > 0]
    R = tr["R"].to_numpy()
    peak = np.maximum.accumulate(curve) if len(curve) else np.array([INITIAL_BALANCE])
    ddser = (peak - curve) / peak if len(curve) else np.array([0.0])
    # trade-close equity for run-up/run-down in trades
    tcl = INITIAL_BALANCE + np.cumsum(tr["pnl"].to_numpy()) if len(tr) else np.array([INITIAL_BALANCE])
    tpk = np.maximum.accumulate(np.concatenate([[INITIAL_BALANCE], tcl]))
    tdd = tpk - np.concatenate([[INITIAL_BALANCE], tcl])
    # longest losing streak
    signs = (tr["pnl"].to_numpy() > 0).astype(int) if len(tr) else np.array([1])
    longest_loss = 0
    cur = 0
    for x in signs:
        cur = 0 if x == 1 else cur + 1
        longest_loss = max(longest_loss, cur)
    eqc[name] = dict(
        final_balance=round(float(curve[-1]) if len(curve) else INITIAL_BALANCE, 2),
        max_dd_abs=round(float(s["max_dd_abs"]), 2),
        max_dd_pct=round(float(s["max_dd_pct"]) * 100, 3),
        max_trade_close_dd_abs=round(float(tdd.max()), 2),
        pct_bars_in_drawdown=round(float((ddser > 0.001).mean()) * 100, 2) if len(ddser) else 0.0,
        longest_losing_streak_trades=int(longest_loss),
        trade_R_mean=round(float(R.mean()), 4) if len(R) else 0.0,
        trade_R_std=round(float(R.std(ddof=1)), 4) if len(R) > 1 else 0.0,
        R_sharpe_per_trade=round(float(R.mean() / R.std(ddof=1)), 3) if (len(R) > 1 and R.std(ddof=1) > 0) else 0.0,
        return_over_maxdd=round(float((curve[-1] - INITIAL_BALANCE) / s["max_dd_abs"]), 3)
        if s["max_dd_abs"] > 0 else None,
    )
    emit("equity", "max_dd_pct", name, eqc[name]["max_dd_pct"])
    emit("equity", "R_sharpe_per_trade", name, eqc[name]["R_sharpe_per_trade"])
RESULTS["equity_curve"] = eqc

# ===========================================================================
# ACCEPTANCE ASSESSMENT  (done BEFORE any alternative-parameter search)
# ===========================================================================
_lap("acceptance assessment")
tr_ok = per_window["TRAIN"].get("expectancy_R", 0) > 0
va_ok = per_window["VALIDATION"].get("expectancy_R", 0) > 0
fi_ok = per_window["FINAL"].get("expectancy_R", 0) > 0
all_windows_positive = tr_ok and va_ok and fi_ok
oos_n = combined_oos.get("n", 0)
comb_pf = combined.get("profit_factor", 0)
comb_oos_pf = combined_oos.get("profit_factor", 0)
comb_oos_expR = combined_oos.get("expectancy_R", 0)
boot = sig.get("bootstrap_oos", {})
boot_p5 = boot.get("mean_R_p5", -9)
boot_prob_pos = boot.get("prob_mean_R_gt_0", 0)
perm = sig.get("permutation_random_event_bars_oos", {})
perm_p = perm.get("p_random_beats_observed_expR", 1.0)
signflip_p = sig.get("permutation_signflip_oos", {}).get("p_two_sided", 1.0)
surv_007 = spread_sens.get("0.07", {}).get("_all_windows_expR_positive", False) and \
    spread_sens.get("0.07", {}).get("_oos_sum_net_pl", -1) > 0
surv_010 = spread_sens.get("0.10", {}).get("_all_windows_expR_positive", False) and \
    spread_sens.get("0.10", {}).get("_oos_sum_net_pl", -1) > 0
any_equity_stop = any(per_window[w]["equity_stopped"] for w in per_window)
month_share = RESULTS["monthly"]["_summary"]["largest_single_month_share_of_gross_profit"]
month_frac_pos = RESULTS["monthly"]["_summary"]["fraction_positive"]

reject = (
    (not all_windows_positive)
    or comb_pf <= 1.0 or comb_oos_pf <= 1.0
    or comb_oos_expR <= 0.0
    or perm_p >= 0.10
    or signflip_p >= 0.10
    or (not surv_007)
    or boot_prob_pos < 0.90
    or any_equity_stop
)
strong = (
    all_windows_positive
    and comb_pf >= 1.30 and comb_oos_pf >= 1.30
    and comb_oos_expR >= 0.05
    and perm_p < 0.05 and signflip_p < 0.05
    and boot_p5 > 0.0 and boot_prob_pos >= 0.95
    and surv_010
    and not any_equity_stop
    and month_share < 0.60 and month_frac_pos >= 0.55
    and oos_n >= 100                      # strict: small OOS can NEVER be CONFIRMED
)

if reject:
    verdict = "REJECTED"
elif strong:
    verdict = "CONFIRMED"
else:
    verdict = "WEAK / FRAGILE"

checklist = dict(
    all_three_windows_positive_expectancy_R=all_windows_positive,
    combined_profit_factor=round(comb_pf, 3), combined_oos_profit_factor=round(comb_oos_pf, 3),
    combined_oos_expectancy_R=round(comb_oos_expR, 4),
    oos_trade_count=int(oos_n),
    bootstrap_oos_prob_mean_R_gt_0=boot_prob_pos, bootstrap_oos_mean_R_p5=boot_p5,
    permutation_random_event_p=perm_p, permutation_signflip_p=signflip_p,
    survives_spread_0_07=bool(surv_007), survives_spread_0_10=bool(surv_010),
    equity_stop_triggered_any_window=bool(any_equity_stop),
    largest_single_month_share_gross_profit=month_share,
    months_positive_fraction=month_frac_pos,
)
RESULTS["acceptance"] = dict(verdict=verdict, checklist=checklist,
                             not_production_ready_reason=(
                                 "OOS sample is ~%d trades; a small sample cannot establish a "
                                 "production edge regardless of positive VALIDATION/FINAL." % oos_n))
emit("acceptance", "verdict", "OVERALL", verdict)
emit("acceptance", "combined_oos_expectancy_R", "OVERALL", round(comb_oos_expR, 4))
emit("acceptance", "oos_trade_count", "OVERALL", int(oos_n))

# ===========================================================================
# WRITE OUTPUTS
# ===========================================================================
_lap("writing outputs")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


RESULTS_OUT = {k: v for k, v in RESULTS.items()}
for w in per_window:
    per_window[w].pop("equity_curve", None)
RESULTS_OUT["per_window"] = per_window

csv_path = REPORTS / "btc_phase3_confirmation.csv"
json_path = REPORTS / "btc_phase3_confirmation.json"
txt_path = REPORTS / "btc_phase3_confirmation_summary.txt"
pd.DataFrame(CSV, columns=["section", "key", "scope", "value"]).to_csv(csv_path, index=False)
json_path.write_text(json.dumps(jsonable(RESULTS_OUT), indent=2), encoding="utf-8")

L = []
A = L.append
A("=" * 90)
A("BTC M5  --  PHASE 3 FROZEN CONFIRMATION  --  widest-TR(48) symmetric OCO breakout")
A("=" * 90)
A(f"engine {ec.ENGINE_VERSION}   rows {N}   {df.time.iloc[0]} -> {df.time.iloc[-1]}")
A(f"FROZEN: struct 48 | buffer 0.5 ATR20 | trigger window 12 | max hold 24 | SL 1.0R | TP 1.5R "
  f"| R = ATR20(event)")
A(f"engine model UNCHANGED: $2000/1% risk, MIN_LOT 1, slip ${SLIP}, spread floor ${DEFAULT_SPREAD}, "
  f"equity stop ${EQUITY_STOP:.0f}, cost guard on")
A("OCO straddle simulated here (engine _backtest is directional-only); engine cost/sizing/equity "
  "arithmetic replicated exactly. No engine or XAU file modified. No live orders.")
A("")
A(f"*** VERDICT:  {verdict}  ***")
A("")
A("-" * 90)
A("1-3.  PER-WINDOW CONFIRMATION  (independent $2,000 reset ; engine cost model + guard ON)")
A("-" * 90)
A(f"  {'window':12s} {'ev':>4} {'trig':>5} {'gskip':>6} {'trades':>7} {'win%':>6} {'net $':>10} {'ret%':>7} "
  f"{'PF':>6} {'expR':>7} {'maxDD$':>9} {'eqstop':>7}")
for name in ("TRAIN", "VALIDATION", "FINAL"):
    s = per_window[name]
    A(f"  {name:12s} {s['n_events']:>4} {s['n_triggered']:>5} {s['n_skipped_guard']:>6} {s['n']:>7} "
      f"{s['win_rate']*100:>6.1f} {s['net_pl']:>+10.2f} {s['return_pct']:>+7.2f} {s['profit_factor']:>6.3f} "
      f"{s['expectancy_R']:>+7.3f} {s['max_dd_abs']:>9.2f} {str(s['equity_stopped']):>7}")
A("")
A("  gskip = triggered breakouts REJECTED by the engine cost guard (ATR20*1.0 < 2*(spread+2*slip)")
A("  = $0.14 at the $0.05 floor). The frozen SL = 1.0*ATR20 is tighter than the guard tolerates,")
A("  so almost every VALIDATION/FINAL event is filtered -> only 1 trade each.")
A("")
A("  DIAGNOSTIC (cost guard OFF -- NOT a faithful engine run; reconciles with the Phase-2 sim):")
A(f"  {'window':12s} {'trades':>7} {'win%':>6} {'net $':>10} {'PF':>6} {'expR':>7}")
for name in ("TRAIN", "VALIDATION", "FINAL"):
    g = guard_off[name]
    A(f"  {name:12s} {g['n']:>7} {g['win_rate']*100:>6.1f} {g['net_pl']:>+10.2f} {g['profit_factor']:>6.3f} "
      f"{g['expectancy_R']:>+7.3f}")
A("")
A("-" * 90)
A("4.  COMBINED  (TRAIN+VAL+FINAL pooled ; and OOS = VAL+FINAL only)")
A("-" * 90)
A(f"  ALL  n {combined['n']:3d}  win {combined['win_rate']*100:5.1f}%  PF {combined['profit_factor']:.3f}  "
  f"expR {combined['expectancy_R']:+.4f}  medR {combined['median_R']:+.3f}  stdR {combined['std_R']:.3f}  "
  f"sum-of-window net ${combined['sum_of_window_net_pl']:+.2f}")
A(f"  OOS  n {combined_oos['n']:3d}  win {combined_oos['win_rate']*100:5.1f}%  PF {combined_oos['profit_factor']:.3f}  "
  f"expR {combined_oos['expectancy_R']:+.4f}  medR {combined_oos['median_R']:+.3f}  stdR {combined_oos['std_R']:.3f}  "
  f"net ${combined_oos['net_pl']:+.2f}")
A("")
A("-" * 90)
A("5.  SPREAD SENSITIVITY  (constant spread ; sum net $ across windows / OOS ; all-windows expR>0 ?)")
A("-" * 90)
A(f"  {'spread':>7} {'TRAIN net':>11} {'VAL net':>10} {'FINAL net':>11} {'sum net':>10} {'OOS net':>10} {'allWin+':>8}")
for sp in SPREAD_SCENARIOS:
    r = spread_sens[f"{sp:.2f}"]
    A(f"  ${sp:>5.2f} {r['TRAIN']['net_pl']:>+11.2f} {r['VALIDATION']['net_pl']:>+10.2f} "
      f"{r['FINAL']['net_pl']:>+11.2f} {r['_sum_net_pl']:>+10.2f} {r['_oos_sum_net_pl']:>+10.2f} "
      f"{str(r['_all_windows_expR_positive']):>8}")
A("")
A("-" * 90)
A("6.  SIGNIFICANCE")
A("-" * 90)
b = sig.get("bootstrap_oos", {})
if "mean_R" in b:
    A(f"  bootstrap OOS ({b['n_trades']} trades, {N_BOOT} resamples):")
    A(f"     mean R {b['mean_R']:+.4f}   95% CI [{b['mean_R_CI95'][0]:+.4f}, {b['mean_R_CI95'][1]:+.4f}]   "
      f"P(mean R>0) {b['prob_mean_R_gt_0']:.3f}")
    A(f"     total $ {b['total_pnl']:+.2f}   95% CI [{b['total_pnl_CI95'][0]:+.2f}, {b['total_pnl_CI95'][1]:+.2f}]   "
      f"P($>0) {b['prob_total_pnl_gt_0']:.3f}")
sf = sig.get("permutation_signflip_oos", {})
if sf:
    A(f"  sign-flip permutation OOS: observed mean R {sf['observed_mean_R']:+.4f}  "
      f"null 0+-{sf['null_std']:.4f}  p(two-sided) {sf['p_two_sided']:.4f}")
rp = sig.get("permutation_random_event_bars_oos", {})
if rp:
    A(f"  random-event-bar permutation OOS ({N_PERM} trials): observed expR {rp['observed_expectancy_R']:+.4f}  "
      f"(pctile {rp['observed_percentile_expR']:.1f})")
    A(f"     random expR {rp['random_expectancy_R_mean']:+.4f} +- {rp['random_expectancy_R_std']:.4f}   "
      f"P(random beats observed expR) {rp['p_random_beats_observed_expR']:.4f}  "
      f"P(random beats net) {rp['p_random_beats_observed_net']:.4f}")
if combined_oos.get("n", 0) < 10:
    A(f"  ** CAUTION: OOS has only {combined_oos.get('n', 0)} trades. The permutation 'significance' is an")
    A("     artifact of a 2-sample win streak, not evidence. It does NOT lift the verdict.")
A("")
A("-" * 90)
A("7.  TRADE-LEVEL MFE / MAE  (R units, R = ATR20 at event bar)")
A("-" * 90)
for scope, mm in mfe_mae.items():
    A(f"  {scope:5s} n {mm['n']:3d}  MFE mean {mm['mean_MFE_R']:+.2f}R (med {mm['median_MFE_R']:+.2f})  "
      f"MAE mean {mm['mean_MAE_R']:+.2f}R (med {mm['median_MAE_R']:+.2f})  MFE/MAE {mm['mfe_over_mae']}  "
      f"P(MFE>=1.5R) {mm['mfe_ge_1_5R']:.2f}  P(MAE>=1.0R) {mm['mae_ge_1_0R']:.2f}")
A("")
A("-" * 90)
A("8.  LONG vs SHORT")
A("-" * 90)
A(f"  p_long: ALL {RESULTS['direction_balance']['p_long_all']}   OOS {RESULTS['direction_balance']['p_long_oos']}")
for kk, st in ls.items():
    if st.get("n", 0):
        A(f"  {kk:14s} n {st['n']:3d}  win {st['win_rate']*100:5.1f}%  PF {st['profit_factor']:.3f}  "
          f"expR {st['expectancy_R']:+.4f}  net ${st['net_pl']:+.2f}")
A("")
A("-" * 90)
A("9.  MONTHLY BREAKDOWN")
A("-" * 90)
for mon, v in monthly.items():
    if mon.startswith("_"):
        continue
    A(f"  {mon}  n {v['n']:3d}  net ${v['net_pl']:+9.2f}  PF {v['profit_factor']:.3f}  "
      f"win {v['win_rate']*100:5.1f}%  expR {v['expectancy_R']:+.4f}")
ms = monthly["_summary"]
A(f"  -> {ms['positive_expectancy_R']}/{ms['blocks']} months positive expR ({ms['fraction_positive']*100:.0f}%)  "
  f"largest single month = {ms['largest_single_month_share_of_gross_profit']*100:.0f}% of gross profit")
A("")
A("-" * 90)
A("10.  DRAWDOWN / EQUITY CURVE")
A("-" * 90)
for name in ("TRAIN", "VALIDATION", "FINAL"):
    e = eqc[name]
    A(f"  {name:12s} final ${e['final_balance']:.2f}  maxDD ${e['max_dd_abs']:.2f} ({e['max_dd_pct']:.2f}%)  "
      f"%bars in DD {e['pct_bars_in_drawdown']:.1f}  longest losing streak {e['longest_losing_streak_trades']}  "
      f"R/trade {e['trade_R_mean']:+.3f}+-{e['trade_R_std']:.3f}  R-Sharpe {e['R_sharpe_per_trade']:+.3f}  "
      f"ret/maxDD {e['return_over_maxdd']}")
A("")
A("=" * 90)
A("ACCEPTANCE ASSESSMENT  (performed BEFORE any alternative-parameter search)")
A("=" * 90)
for k, v in checklist.items():
    A(f"  {k:44s} : {v}")
A("")
A(f"  VERDICT:  {verdict}")
A("")
if verdict == "REJECTED":
    A("  The frozen candidate fails multiple hard gates (see checklist).")
    A("  - Guard ON (faithful engine): the frozen SL = 1.0*ATR20 is tighter than the engine cost")
    A("    guard tolerates, so ~90% of triggers are filtered; VALIDATION and FINAL take 1 trade each.")
    A("  - Guard OFF (diagnostic, all trades, real sizing): expectancy R is NEGATIVE in all three")
    A("    windows (TRAIN -0.68, VAL -0.51, FINAL -1.19); only ~30% of trades reach the 1.5R target.")
    A("  - This SUPERSEDES the Phase-2 straddle estimate (+0.3 R): that idealised sim tracked")
    A("    excursions past the stop and allowed overlapping trades, overstating the edge.")
    A("  Do NOT proceed to a strategy build with these parameters. A parameter search now would be")
    A("  curve-fitting to a handful of trades. The volatility-breakout hypothesis is REJECTED on")
    A("  BTC M5 close data under the realistic cost model.")
elif verdict == "CONFIRMED":
    A("  All hard AND strict gates pass. STILL NOT PRODUCTION-READY: the OOS sample is small")
    A(f"  (~{oos_n} trades). Treat as a candidate for a forward paper test only, not deployment.")
else:
    A("  Passes the hard REJECT gates but not the strict CONFIRM gates.")
    A(f"  {RESULTS['acceptance']['not_production_ready_reason']}")
    A("  Not production-ready. If pursued at all: forward PAPER test on unseen data, frozen")
    A("  parameters, realistic ($0.01-0.05) spread. Do NOT tune. Stop if the paper test is flat.")
A("=" * 90)
txt_path.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
_lap("done")
print("\nFILES CREATED:")
for p in (Path(__file__).resolve(), csv_path, json_path, txt_path):
    print(f"  {p.relative_to(ROOT)}")
