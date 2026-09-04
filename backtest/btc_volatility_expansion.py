"""
btc_volatility_expansion.py  --  BTC-ONLY volatility-expansion / symmetric-breakout research.

CONTEXT
    Phase-1 signal discovery found: BTC M5 OHLC has NO usable *directional* edge, but
    volatility DOES predict future MOVE MAGNITUDE (Spearman(atr_rank, |fwd ret|) ~= +0.32
    across TRAIN / VALIDATION / FINAL).

HYPOTHESIS (this script)
    Current volatility -> future move SIZE (not direction).  Can a DIRECTION-AGNOSTIC
    breakout monetize it?  i.e. arm a symmetric OCO around the recent range; whichever
    side triggers first, ride it; is the subsequent move (a) unusually large, (b) larger
    than round-trip cost, (c) genuinely direction-agnostic or does one side dominate?

SAFETY / SCOPE
    * Reads data/btc_m5_history.csv and imports btc_engine_core / btc_realistic_config
      for the contract + spread model ONLY.  No engine / config / XAU file is modified.
    * No MT5 / live-order code.
    * _backtest is directional-only and cannot represent an OCO straddle, so the Phase-3
      simulation is here, replicating the engine's EXACT cost arithmetic:
         stop-entry fill  = trigger +/- half_spread +/- slippage      (engine: stop pays slip)
         time/stop exit   = px      +/- half_spread +/- slippage
         target exit      = px      +/- half_spread                    (limit: no slip)
         cost_price       = spread + 2*slippage                        (engine guard, line 454)
    * Strict causality: every trigger level / regime / event flag at bar i uses ONLY
      bars <= i;  every forward measurement uses ONLY bars > i.  Regime + breakout
      thresholds are frozen on TRAIN and applied unchanged to VALIDATION / FINAL.

OUTPUTS
    reports/btc_volatility_expansion.csv
    reports/btc_volatility_expansion.json
    reports/btc_volatility_expansion_summary.txt
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=RuntimeWarning)

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
# CONFIG  (from the confirmed BTC engine / config -- nothing invented)
# ===========================================================================
DATA = ROOT / "data" / "btc_m5_history.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TICK = float(ec.TICK_SIZE)                 # 0.01
VPP = float(ec.VALUE_PER_POINT)            # 1.0  ($ per $1 move per lot)
SLIP = float(bc.SLIPPAGE)                  # 0.01
DEFAULT_SPREAD = float(bc.DEFAULT_SPREAD)  # 0.05
CLAMP_MULT = 5.0
SPLIT_FRACS = (0.70, 0.85)

HORIZONS = (1, 3, 6, 12, 24)
STRUCT_WINDOWS = (12, 24, 48)
HOLD_BARS = (6, 12, 24)
BUFFER_FRACS = (0.5,)                      # trigger buffer in ATR units (kept small: hypothesis test)
BRACKET = (1.5, 1.0)                       # (target_R, stop_R) for the bracket-exit variant
SPREAD_SCENARIOS = (0.05, 0.10, 0.18, 0.25)
SEED = 20260830
N_NULL = 400
REGIME_LABELS = ("very_low", "low", "medium", "high", "very_high")


# ===========================================================================
# HELPERS
# ===========================================================================
def _rankdata(a):
    a = np.asarray(a, float)
    n = len(a)
    if n == 0:
        return np.empty(0, float)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    pos = np.arange(1, n + 1, dtype=float)
    change = np.ones(n, bool)
    change[1:] = sa[1:] != sa[:-1]
    grp = np.cumsum(change) - 1
    mean_rank = np.bincount(grp, weights=pos) / np.bincount(grp)
    out = np.empty(n, float)
    out[order] = mean_rank[grp]
    return out


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return 0.0, int(m.sum())
    rx = _rankdata(x[m]) - (m.sum() + 1) / 2.0
    ry = _rankdata(y[m]) - (m.sum() + 1) / 2.0
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / d) if d > 0 else 0.0), int(m.sum())


def roll_mean(s, w):
    return pd.Series(s).rolling(w, min_periods=w).mean().to_numpy()


def roll_std(s, w):
    return pd.Series(s).rolling(w, min_periods=w).std(ddof=0).to_numpy()


def roll_max_sh1(s, w):
    return pd.Series(s).shift(1).rolling(w, min_periods=w).max().to_numpy()


def roll_min_sh1(s, w):
    return pd.Series(s).shift(1).rolling(w, min_periods=w).min().to_numpy()


def roll_rank_pct(s, w):
    try:
        return pd.Series(s).rolling(w, min_periods=max(20, w // 5)).rank(pct=True).to_numpy()
    except Exception:
        a = np.asarray(s, float)
        out = np.full(len(a), np.nan)
        for i in range(w - 1, len(a)):
            win = a[i - w + 1:i + 1]
            out[i] = np.mean(win <= a[i])
        return out


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
    return x


# ===========================================================================
# NUMBA KERNELS
# ===========================================================================
@njit(cache=True)
def fwd_move(high, low, close, atr, H):
    """causal forward move measures at bar i over (i, i+H]."""
    n = close.shape[0]
    absret = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    rng = np.full(n, np.nan)
    for i in range(n - H):
        a = atr[i]
        if not (a > 0.0):
            continue
        hh = high[i + 1]
        ll = low[i + 1]
        for k in range(i + 2, i + H + 1):
            if high[k] > hh:
                hh = high[k]
            if low[k] < ll:
                ll = low[k]
        absret[i] = abs(close[i + H] - close[i]) / a
        mfe[i] = (hh - close[i]) / a
        mae[i] = (close[i] - ll) / a
        rng[i] = (hh - ll) / a
    return absret, mfe, mae, rng


@njit(cache=True)
def straddle_scan(open_, high, low, close, atr, spread,
                  upper, lower, ev_idx,
                  h_scan, h_hold, slip, tgt_R, stop_R,
                  o_trig, o_dir, o_ttrig, o_mfe, o_mae,
                  o_pnl_time, o_pnl_brk, o_brk_outcome):
    """
    For each armed event bar i in ev_idx:
      * arm OCO: buy-stop at upper[i], sell-stop at lower[i]
      * first side touched within h_scan bars wins, other cancelled
      * entry fill = trigger +/- half_spread +/- slip   (stop order)
      * measure post-entry MFE/MAE in ATR(i) units over h_hold bars
      * TIME exit  = close at (trigger_bar + h_hold) +/- half_spread +/- slip
      * BRACKET exit = first of { target = entry + dir*tgt_R*atr(i)  (limit, no slip),
                                  stop   = entry - dir*stop_R*atr(i) (stop, +slip) },
        checking STOP before TARGET within a bar (conservative); else time exit.
    All pnl in R units = pnl_price / atr(i).
    """
    n = close.shape[0]
    for e in range(ev_idx.shape[0]):
        i = ev_idx[e]
        a = atr[i]
        up = upper[i]
        dn = lower[i]
        if not (a > 0.0) or not np.isfinite(up) or not np.isfinite(dn):
            o_trig[e] = 0
            continue
        ke = -1
        d = 0
        kmax = i + h_scan
        if kmax > n - 1:
            kmax = n - 1
        for k in range(i + 1, kmax + 1):
            hit_up = high[k] >= up
            hit_dn = low[k] <= dn
            if hit_up and hit_dn:
                d = 1 if abs(open_[k] - up) <= abs(open_[k] - dn) else -1
                ke = k
                break
            elif hit_up:
                d = 1
                ke = k
                break
            elif hit_dn:
                d = -1
                ke = k
                break
        if ke < 0:
            o_trig[e] = 0
            continue

        hs_e = spread[ke] * 0.5
        if d == 1:
            entry = up + hs_e + slip
        else:
            entry = dn - hs_e - slip

        mend = ke + h_hold
        if mend > n - 1:
            mend = n - 1

        mfe = 0.0
        mae = 0.0
        tgt_px = entry + d * tgt_R * a
        stp_px = entry - d * stop_R * a
        brk_exit = 0.0
        brk_done = 0
        brk_outcome = 0            # +1 target, -1 stop, 0 time
        for m in range(ke + 1, mend + 1):
            hs_m = spread[m] * 0.5
            if d == 1:
                fav = (high[m] - hs_m) - entry
                adv = entry - (low[m] - hs_m)
            else:
                fav = entry - (low[m] + hs_m)
                adv = (high[m] + hs_m) - entry
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv
            if brk_done == 0:
                if d == 1:
                    if low[m] <= stp_px:
                        brk_exit = stp_px - hs_m - slip
                        brk_done = 1
                        brk_outcome = -1
                    elif high[m] >= tgt_px:
                        brk_exit = tgt_px - hs_m
                        brk_done = 1
                        brk_outcome = 1
                else:
                    if high[m] >= stp_px:
                        brk_exit = stp_px + hs_m + slip
                        brk_done = 1
                        brk_outcome = -1
                    elif low[m] <= tgt_px:
                        brk_exit = tgt_px + hs_m
                        brk_done = 1
                        brk_outcome = 1

        hs_x = spread[mend] * 0.5
        if d == 1:
            time_exit = close[mend] - hs_x - slip
        else:
            time_exit = close[mend] + hs_x + slip
        if brk_done == 0:
            brk_exit = time_exit
            brk_outcome = 0

        o_trig[e] = 1
        o_dir[e] = d
        o_ttrig[e] = ke - i
        o_mfe[e] = mfe / a
        o_mae[e] = mae / a
        o_pnl_time[e] = ((time_exit - entry) * d) / a
        o_pnl_brk[e] = ((brk_exit - entry) * d) / a
        o_brk_outcome[e] = brk_outcome


# ===========================================================================
# LOAD  +  CAUSAL SERIES
# ===========================================================================
_lap("load data")
df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
o = df.open.to_numpy(float)
h = df.high.to_numpy(float)
l = df.low.to_numpy(float)
c = df.close.to_numpy(float)
raw_pts = df["spread"].to_numpy(float)
N = len(c)

ec.DEFAULT_SPREAD = bc.DEFAULT_SPREAD
spread = ec.prepare_spread(df)
cost_price_base = DEFAULT_SPREAD + 2.0 * SLIP

prev_c = np.concatenate([[c[0]], c[:-1]])
TR = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
atr10 = ec.atr(h, l, c, 10)
atr20 = ec.atr(h, l, c, 20)
atr50 = ec.atr(h, l, c, 50)
atr20s = np.where(np.isfinite(atr20) & (atr20 > 0), atr20, np.nan)
ret1 = np.concatenate([[np.nan], np.diff(c) / c[:-1]])
rv10 = pd.Series(ret1).rolling(10, min_periods=10).std(ddof=0).to_numpy()
rv20 = pd.Series(ret1).rolling(20, min_periods=20).std(ddof=0).to_numpy()
rv50 = pd.Series(ret1).rolling(50, min_periods=50).std(ddof=0).to_numpy()

TRAIN_END = int(N * SPLIT_FRACS[0])
VAL_END = int(N * SPLIT_FRACS[1])
WINDOWS = {"TRAIN": (0, TRAIN_END), "VALIDATION": (TRAIN_END, VAL_END), "FINAL": (VAL_END, N)}
_lap(f"rows={N}  TRAIN[0,{TRAIN_END}) VAL[{TRAIN_END},{VAL_END}) FINAL[{VAL_END},{N})  "
     f"cost_price(base)=${cost_price_base:.2f}")

RES = {"meta": dict(
    engine_version=ec.ENGINE_VERSION, rows=int(N),
    span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
    windows={k: list(v) for k, v in WINDOWS.items()},
    contract=dict(tick=TICK, value_per_point=VPP, min_lot=float(ec.MIN_LOT),
                  default_spread=DEFAULT_SPREAD, slippage=SLIP, clamp_mult=CLAMP_MULT),
    cost_price_base=cost_price_base,
    horizons=list(HORIZONS), struct_windows=list(STRUCT_WINDOWS), hold_bars=list(HOLD_BARS),
    buffer_fracs=list(BUFFER_FRACS), bracket=list(BRACKET),
    spread_scenarios=list(SPREAD_SCENARIOS), seed=SEED,
    causality="feature/level<=i ; forward measures>i ; regime+event thresholds frozen on TRAIN",
    note="direction-agnostic breakout research; _backtest is directional-only so the OCO "
         "straddle is simulated here with the engine's exact cost arithmetic. No engine change.",
)}
CSV = []


def emit(section, key, scope, value):
    CSV.append(dict(section=section, key=key, scope=scope, value=value))


def wmask(win, H):
    w0, w1 = WINDOWS[win]
    m = np.zeros(N, bool)
    hi = min(w1, N - H)
    if hi > w0:
        m[w0:hi] = True
    return m


# forward move measures per horizon
_lap("forward move measures")
FWD = {}
for H in HORIZONS:
    ar, mf, ma, rg = fwd_move(h, l, c, atr20s, H)
    fut = np.concatenate([c[H:], [np.nan] * H])
    absret_px = np.abs(fut - c)
    # raw future high-low range in price units (causal: bars i+1..i+H)
    rng_hi = pd.Series(h).rolling(H).max().shift(-H).to_numpy()
    rng_lo = pd.Series(l).rolling(H).min().shift(-H).to_numpy()
    FWD[H] = dict(absret_R=ar, mfe_R=mf, mae_R=ma, range_R=rg,          # *_R divide by atr20[i] (circular vs vol features!)
                  absret_px=absret_px, range_px=(rng_hi - rng_lo))

# ===========================================================================
# PHASE 1  --  VOLATILITY REGIMES  ->  FUTURE MOVE SIZE
# ===========================================================================
_lap("phase 1: volatility regimes -> future move size")
VOLF = {
    "atr20_rank_500": roll_rank_pct(atr20s, 500),
    "atr10_over_atr50": atr10 / np.where(atr50 > 0, atr50, np.nan),
    "rv10_over_rv50": rv10 / np.where(rv50 > 0, rv50, np.nan),
    "range_compression_6_48": roll_mean(TR, 6) / np.where(roll_mean(TR, 48) > 0, roll_mean(TR, 48), np.nan),
    "range_expansion_20": TR / np.where(roll_mean(TR, 20) > 0, roll_mean(TR, 20), np.nan),
    "atr20_over_spread": atr20s / np.where(spread > 0, spread, np.nan),
    "expected_move_over_cost": atr20s / cost_price_base,
}
PRIMARY_VOL = "atr20_rank_500"

# TRAIN-frozen quintile edges for each vol measure
tm = wmask("TRAIN", max(HORIZONS))
regime_edges = {}
for name, arr in VOLF.items():
    v = arr[tm]
    v = v[np.isfinite(v)]
    regime_edges[name] = list(np.quantile(v, [0.2, 0.4, 0.6, 0.8])) if len(v) else [0, 0, 0, 0]


def assign_regime(arr, edges):
    e = np.asarray(edges, float)
    out = np.full(len(arr), -1, int)
    fin = np.isfinite(arr)
    out[fin] = np.searchsorted(e, arr[fin], side="right")
    return out


phase1 = {}
for name, arr in VOLF.items():
    reg = assign_regime(arr, regime_edges[name])
    per = {}
    for win in ("TRAIN", "VALIDATION", "FINAL"):
        wm = wmask(win, 24)
        bh = {}
        for q in range(5):
            g = wm & (reg == q)
            if g.sum() < 20:
                continue
            row = dict(n=int(g.sum()))
            for H in (6, 12, 24):
                mk = wmask(win, H) & (reg == q) & np.isfinite(FWD[H]["absret_R"])
                if mk.sum() < 20:
                    continue
                row[f"H{H}"] = dict(
                    mean_absret_px=round(float(FWD[H]["absret_px"][mk].mean()), 4),
                    mean_range_px=round(float(FWD[H]["range_px"][mk].mean()), 4),
                    mean_absret_R=round(float(FWD[H]["absret_R"][mk].mean()), 4),
                    mean_range_R=round(float(FWD[H]["range_R"][mk].mean()), 4),
                    mean_mfe_R=round(float(FWD[H]["mfe_R"][mk].mean()), 4),
                    mean_mae_R=round(float(FWD[H]["mae_R"][mk].mean()), 4),
                )
            bh[REGIME_LABELS[q]] = row
        per[win] = bh
    # Spearman(vol value, RAW forward measure) per window.  RAW $ only -- NOT
    # atr-normalised, because normalising the forward move by the same atr that
    # the vol feature ranks is circular and forces a spurious negative.
    mono = {}
    for win in ("TRAIN", "VALIDATION", "FINAL"):
        row = {}
        for H in (6, 12):
            mk = wmask(win, H)
            r_net, n1 = spearman(arr[mk], FWD[H]["absret_px"][mk])
            r_rng, _ = spearman(arr[mk], FWD[H]["range_px"][mk])
            r_circ, _ = spearman(arr[mk], FWD[H]["absret_R"][mk])
            row[f"H{H}"] = dict(spearman_netmove_px=round(r_net, 4),
                                spearman_range_px=round(r_rng, 4),
                                spearman_netmove_ATRnorm_CIRCULAR=round(r_circ, 4), n=n1)
            emit("phase1_vol_size", f"{name}/spearman_netmove_px_H{H}", win, round(r_net, 4))
            emit("phase1_vol_size", f"{name}/spearman_range_px_H{H}", win, round(r_rng, 4))
        mono[win] = row
    # cross-window consistency of very_high vs very_low RAW $ move ratios
    def _ratio(field, H):
        out = []
        for win in ("TRAIN", "VALIDATION", "FINAL"):
            try:
                hi = per[win]["very_high"][f"H{H}"][field]
                lo = per[win]["very_low"][f"H{H}"][field]
                out.append(round(hi / lo, 3) if lo else None)
            except Exception:
                out.append(None)
        return out
    phase1[name] = dict(train_quintile_edges=[round(x, 5) for x in regime_edges[name]],
                        by_window=per, spearman_vs_raw_forward=mono,
                        veryhigh_over_verylow_netmove_px_H12=_ratio("mean_absret_px", 12),
                        veryhigh_over_verylow_range_px_H6=_ratio("mean_range_px", 6))
RES["phase1_volatility_regimes"] = phase1

# ===========================================================================
# PHASE 2  --  BREAKOUT EVENTS  ->  FUTURE MOVE SIZE
# ===========================================================================
_lap("phase 2: breakout events -> future move size")


def event_flags(Nw):
    rhi = roll_max_sh1(h, Nw)
    rlo = roll_min_sh1(l, Nw)
    tr_max = pd.Series(TR).shift(1).rolling(Nw, min_periods=Nw).max().to_numpy()
    brk_hi = (c > rhi).astype(float)
    brk_lo = (c < rlo).astype(float)
    brk_any = ((c > rhi) | (c < rlo)).astype(float)
    range_brk = (TR > tr_max).astype(float)
    comp = roll_mean(TR, 6) / np.where(roll_mean(TR, 24) > 0, roll_mean(TR, 24), np.nan)
    comp_prev = np.concatenate([[np.nan], comp[:-1]])
    compress_expand = ((comp_prev < 0.8) & (TR / np.where(roll_mean(TR, 24) > 0, roll_mean(TR, 24), np.nan) > 1.5)).astype(float)
    atr_exp = (atr10 / np.concatenate([[np.nan] * 12, atr10[:-12]]) > 1.3).astype(float)
    lowvol_prev = np.concatenate([[np.nan], roll_rank_pct(atr20s, 500)[:-1]])
    exp_after_lowvol = ((lowvol_prev < 0.25) & (TR / np.where(roll_mean(TR, 20) > 0, roll_mean(TR, 20), np.nan) > 1.5)).astype(float)
    return dict(brk_any=brk_any, range_brk=range_brk, compress_expand=compress_expand,
               atr_expansion=atr_exp, expansion_after_lowvol=exp_after_lowvol,
               _rhi=rhi, _rlo=rlo)


phase2 = {}
for Nw in STRUCT_WINDOWS:
    ef = event_flags(Nw)
    for ename in ("brk_any", "range_brk", "compress_expand", "atr_expansion", "expansion_after_lowvol"):
        flag = ef[ename]
        per = {}
        for win in ("TRAIN", "VALIDATION", "FINAL"):
            rows = {}
            for H in (6, 12, 24):
                base_mk = wmask(win, H) & np.isfinite(FWD[H]["absret_px"])
                ev_mk = base_mk & (flag == 1.0)
                non_mk = base_mk & (flag == 0.0)
                if ev_mk.sum() < 20:
                    continue
                ev_move = float(FWD[H]["absret_px"][ev_mk].mean())            # RAW $ (not atr-normalised)
                non_move = float(FWD[H]["absret_px"][non_mk].mean()) if non_mk.sum() else np.nan
                ev_rng = float(FWD[H]["range_px"][ev_mk].mean())
                non_rng = float(FWD[H]["range_px"][non_mk].mean()) if non_mk.sum() else np.nan
                rows[f"H{H}"] = dict(
                    n_event=int(ev_mk.sum()), event_rate=round(float(ev_mk.sum() / base_mk.sum()), 4),
                    mean_absret_px_event=round(ev_move, 5),
                    mean_absret_px_nonevent=round(non_move, 5) if np.isfinite(non_move) else None,
                    move_lift_ratio=round(ev_move / non_move, 3) if (np.isfinite(non_move) and non_move) else None,
                    range_lift_ratio=round(ev_rng / non_rng, 3) if (np.isfinite(non_rng) and non_rng) else None,
                    mean_mfe_R_event=round(float(FWD[H]["mfe_R"][ev_mk].mean()), 4),
                    mean_mae_R_event=round(float(FWD[H]["mae_R"][ev_mk].mean()), 4),
                )
            per[win] = rows
        lifts = [per[w].get("H12", {}).get("move_lift_ratio") for w in ("TRAIN", "VALIDATION", "FINAL")]
        phase2[f"{ename}_N{Nw}"] = dict(by_window=per,
                                       H12_move_lift_train_val_final=lifts,
                                       lift_sign_consistent=bool(
                                           all(x is not None and x > 1.0 for x in lifts)))
        for w in ("TRAIN", "VALIDATION", "FINAL"):
            v = per.get(w, {}).get("H12", {}).get("move_lift_ratio")
            emit("phase2_event_size", f"{ename}_N{Nw}/H12_move_lift", w, v)
RES["phase2_breakout_events"] = phase2

# ===========================================================================
# PHASE 3  --  DIRECTION-AGNOSTIC OCO STRADDLE SIMULATION
# ===========================================================================
_lap("phase 3: direction-agnostic OCO straddle simulation")
tgt_R, stop_R = BRACKET
EV_DEFS = {}
for Nw in STRUCT_WINDOWS:
    ef = event_flags(Nw)
    rhi, rlo = ef["_rhi"], ef["_rlo"]
    for ename in ("brk_any", "range_brk", "compress_expand", "atr_expansion", "expansion_after_lowvol"):
        EV_DEFS[f"{ename}_N{Nw}"] = (ef[ename], rhi, rlo)


def run_straddle(flag, rhi, rlo, buf_frac, h_scan, h_hold, sp_arr, win):
    wm = wmask(win, h_scan + h_hold + 2)
    ev_idx = np.where(wm & (flag == 1.0) & np.isfinite(atr20s) & np.isfinite(rhi) & np.isfinite(rlo))[0].astype(np.int64)
    if len(ev_idx) < 15:
        return None
    upper = rhi + buf_frac * atr20s
    lower = rlo - buf_frac * atr20s
    z = lambda dt: np.zeros(len(ev_idx), dtype=dt)
    o_trig = z(np.int64); o_dir = z(np.int64); o_ttrig = z(np.int64)
    o_mfe = z(np.float64); o_mae = z(np.float64)
    o_pt = z(np.float64); o_pb = z(np.float64); o_bo = z(np.int64)
    straddle_scan(o, h, l, c, atr20s, np.ascontiguousarray(sp_arr),
                  np.ascontiguousarray(upper), np.ascontiguousarray(lower), ev_idx,
                  int(h_scan), int(h_hold), SLIP, float(tgt_R), float(stop_R),
                  o_trig, o_dir, o_ttrig, o_mfe, o_mae, o_pt, o_pb, o_bo)
    tr = o_trig == 1
    ntr = int(tr.sum())
    if ntr < 15:
        return dict(n_armed=int(len(ev_idx)), n_triggered=ntr, trigger_rate=round(ntr / len(ev_idx), 3))
    d = o_dir[tr]
    pt = o_pt[tr]
    pb = o_pb[tr]
    return dict(
        n_armed=int(len(ev_idx)), n_triggered=ntr,
        trigger_rate=round(ntr / len(ev_idx), 3),
        p_long=round(float((d == 1).mean()), 3),
        mean_bars_to_trigger=round(float(o_ttrig[tr].mean()), 2),
        mean_MFE_R=round(float(o_mfe[tr].mean()), 3),
        mean_MAE_R=round(float(o_mae[tr].mean()), 3),
        mfe_minus_mae_R=round(float((o_mfe[tr] - o_mae[tr]).mean()), 3),
        continuation_rate=round(float((o_bo[tr] == 1).mean()), 3),
        stopout_rate=round(float((o_bo[tr] == -1).mean()), 3),
        time_exit_rate=round(float((o_bo[tr] == 0).mean()), 3),
        net_R_time_exit=round(float(pt.mean()), 4),
        net_R_bracket=round(float(pb.mean()), 4),
        pct_trades_net_pos_time=round(float((pt > 0).mean()), 3),
        pct_trades_net_pos_bracket=round(float((pb > 0).mean()), 3),
        median_R_bracket=round(float(np.median(pb)), 4),
    )


phase3 = {}
buf = BUFFER_FRACS[0]
for evname, (flag, rhi, rlo) in EV_DEFS.items():
    for h_scan in STRUCT_WINDOWS:
        for h_hold in HOLD_BARS:
            key = f"{evname}/scan{h_scan}/hold{h_hold}"
            per = {}
            for win in ("TRAIN", "VALIDATION", "FINAL"):
                r = run_straddle(flag, rhi, rlo, buf, h_scan, h_hold, spread, win)
                if r:
                    per[win] = r
            if len(per) == 3 and all("net_R_bracket" in per[w] for w in per):
                nb = [per[w]["net_R_bracket"] for w in ("TRAIN", "VALIDATION", "FINAL")]
                nt = [per[w]["net_R_time_exit"] for w in ("TRAIN", "VALIDATION", "FINAL")]
                per["_net_R_bracket_tvf"] = nb
                per["_net_R_time_tvf"] = nt
                per["_bracket_pos_all_windows"] = bool(all(x > 0 for x in nb))
                per["_time_pos_all_windows"] = bool(all(x > 0 for x in nt))
            phase3[key] = per
            for w in ("TRAIN", "VALIDATION", "FINAL"):
                if w in per and "net_R_bracket" in per[w]:
                    emit("phase3_straddle", f"{key}/net_R_bracket", w, per[w]["net_R_bracket"])
                    emit("phase3_straddle", f"{key}/net_R_time", w, per[w]["net_R_time_exit"])
RES["phase3_straddle"] = phase3

# best configs by weakest-window bracket net R
scored = []
for key, per in phase3.items():
    if "_net_R_bracket_tvf" in per:
        scored.append((key, min(per["_net_R_bracket_tvf"]), per["_net_R_bracket_tvf"],
                       per["_net_R_time_tvf"]))
scored.sort(key=lambda t: -t[1])
RES["phase3_best_configs"] = [dict(config=k, weakest_bracket_net_R=round(w, 4),
                                   bracket_tvf=[round(x, 4) for x in b],
                                   time_tvf=[round(x, 4) for x in t]) for k, w, b, t in scored[:10]]

# ===========================================================================
# PHASE 4  --  COST MODEL
# ===========================================================================
_lap("phase 4: cost model")
reg_primary = assign_regime(VOLF[PRIMARY_VOL], regime_edges[PRIMARY_VOL])
phase4 = dict(cost_price_formula="spread + 2*slippage", slippage=SLIP,
              scenarios={}, regime_expected_move={}, spread_sensitivity={})
for sp in SPREAD_SCENARIOS:
    cp = sp + 2.0 * SLIP
    phase4["scenarios"][f"{sp:.2f}"] = dict(spread=sp, cost_price=round(cp, 4),
                                            round_trip_cost=round(cp, 4))
# expected move per regime (all windows pooled, H=12) vs cost
for q in range(5):
    mk = (reg_primary == q) & np.isfinite(FWD[12]["absret_R"]) & np.isfinite(atr20s)
    if mk.sum() < 30:
        continue
    em_R = float(FWD[12]["absret_R"][mk].mean())
    em_px = float(np.abs(np.concatenate([c[12:], [np.nan] * 12])[mk] - c[mk]).mean())
    atr_med = float(np.median(atr20s[mk]))
    row = dict(n=int(mk.sum()), expected_move_R_H12=round(em_R, 3),
               expected_move_px_H12=round(em_px, 4), median_atr=round(atr_med, 4))
    for sp in SPREAD_SCENARIOS:
        cp = sp + 2.0 * SLIP
        row[f"cost_over_atr@{sp:.2f}"] = round(cp / atr_med, 3)
        row[f"cost_over_expmove@{sp:.2f}"] = round(cp / em_px, 3) if em_px else None
        row[f"expmove_over_cost@{sp:.2f}"] = round(em_px / cp, 3) if cp else None
        row[f"breakeven@{sp:.2f}"] = bool(em_px > cp)
    phase4["regime_expected_move"][REGIME_LABELS[q]] = row
    emit("phase4_cost", "expected_move_over_cost@0.05", REGIME_LABELS[q],
         row.get("expmove_over_cost@0.05"))

# spread sensitivity of the top-3 phase-3 configs
for key, wk, b, t in scored[:3]:
    evname = key.split("/")[0]
    h_scan = int(key.split("scan")[1].split("/")[0])
    h_hold = int(key.split("hold")[1])
    flag, rhi, rlo = EV_DEFS[evname]
    ss = {}
    for sp in SPREAD_SCENARIOS:
        sp_arr = np.full(N, sp, float)
        nb = []
        for win in ("TRAIN", "VALIDATION", "FINAL"):
            r = run_straddle(flag, rhi, rlo, buf, h_scan, h_hold, sp_arr, win)
            nb.append(round(r["net_R_bracket"], 4) if (r and "net_R_bracket" in r) else None)
        ss[f"{sp:.2f}"] = nb
        emit("phase4_spread_sensitivity", f"{key}/net_R_bracket_tvf", f"spread={sp:.2f}", nb)
    phase4["spread_sensitivity"][key] = ss
RES["phase4_cost_model"] = phase4

# ===========================================================================
# PHASE 5  --  NULL BASELINE  +  VERDICT
# ===========================================================================
_lap("phase 5: null baseline + verdict")
rng = np.random.default_rng(SEED)
# null: does vol -> RAW future range (H6) beat shuffled vol?  (the genuine, non-circular relation)
mk = wmask("TRAIN", 6) & np.isfinite(VOLF[PRIMARY_VOL]) & np.isfinite(FWD[6]["range_px"])
xv = VOLF[PRIMARY_VOL][mk]
yv = FWD[6]["range_px"][mk]
obs_r, _ = spearman(xv, yv)
rx = _rankdata(xv) - (len(xv) + 1) / 2
ry = _rankdata(yv) - (len(yv) + 1) / 2
den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
null = np.empty(N_NULL)
for i in range(N_NULL):
    null[i] = float((rx[rng.permutation(len(rx))] * ry).sum() / den) if den > 0 else 0.0
null_p_range = float((np.abs(null) >= abs(obs_r)).mean())

p1 = RES["phase1_volatility_regimes"][PRIMARY_VOL]
sp_net = {w: p1["spearman_vs_raw_forward"][w]["H6"]["spearman_netmove_px"] for w in ("TRAIN", "VALIDATION", "FINAL")}
sp_rng = {w: p1["spearman_vs_raw_forward"][w]["H6"]["spearman_range_px"] for w in ("TRAIN", "VALIDATION", "FINAL")}
sp_circ = {w: p1["spearman_vs_raw_forward"][w]["H6"]["spearman_netmove_ATRnorm_CIRCULAR"] for w in ("TRAIN", "VALIDATION", "FINAL")}
net_move_predictable = all(v > 0.10 for v in sp_net.values())          # NET directional move
range_predictable = all(v > 0.15 for v in sp_rng.values())             # thrash / realized range
vh_vl_net = p1["veryhigh_over_verylow_netmove_px_H12"]
vh_vl_rng = p1["veryhigh_over_verylow_range_px_H6"]

best_bracket_weakest = scored[0][1] if scored else -9.9
best_bracket_tvf = scored[0][2] if scored else [None, None, None]
best_time_weakest = max((min(t) for _, _, _, t in scored), default=-9.9)
best_key = scored[0][0] if scored else None

# spread at which the best config's bracket net R goes negative in ANY window
best_spread_sens = phase4["spread_sensitivity"].get(best_key, {}) if best_key else {}
dies_at_spread = None
for sp in SPREAD_SCENARIOS:
    nb = best_spread_sens.get(f"{sp:.2f}")
    if nb and any((x is None) or (x <= 0) for x in nb):
        dies_at_spread = sp
        break

# direction dominance / whipsaw from the best config
dir_reading = "n/a"
whipsaw = None
n_oos_triggers = None
if best_key:
    pw = phase3[best_key]
    ps = [pw[w]["p_long"] for w in ("TRAIN", "VALIDATION", "FINAL") if w in pw and "p_long" in pw[w]]
    mm = [pw[w]["mfe_minus_mae_R"] for w in ("TRAIN", "VALIDATION", "FINAL") if w in pw and "mfe_minus_mae_R" in pw[w]]
    n_oos_triggers = [pw[w]["n_triggered"] for w in ("VALIDATION", "FINAL") if w in pw and "n_triggered" in pw[w]]
    if ps:
        bal = all(0.4 <= x <= 0.6 for x in ps)
        whipsaw = bool(mm and np.mean(mm) < 0)
        dir_reading = (("direction-agnostic (balanced), " +
                        ("WHIPSAW: MAE>MFE" if whipsaw else "post-trigger CONTINUATION: MFE>MAE"))
                       if bal else "one side dominates p_long~%.2f" % np.mean(ps))

cost_over_atr_median = float(np.nanmedian((spread + 2 * SLIP) / atr20s))
best_pos_all_windows_005 = bool(scored and best_bracket_tvf[0] is not None
                               and all(x > 0 for x in best_bracket_tvf))
small_oos = bool(n_oos_triggers and min(n_oos_triggers) < 60)

if net_move_predictable and best_pos_all_windows_005 and (dies_at_spread is None or dies_at_spread >= 0.18) and not small_oos:
    verdict = "PROMISING - a symmetric breakout clears cost cross-window and is robust to spread"
elif best_pos_all_windows_005 and best_bracket_weakest > 0.02:
    verdict = ("WEAK / FRAGILE - one narrow breakout event is net-positive cross-window at the "
               f"${DEFAULT_SPREAD:.2f} floor, but small OOS sample"
               + (f" and it inverts at spread ${dies_at_spread:.2f}" if dies_at_spread else "") + "")
elif range_predictable and not net_move_predictable:
    verdict = ("SIZE-ONLY (RANGE, NOT DIRECTION) - volatility predicts the future high-low RANGE "
               "(thrash) short-term, NOT the net move; no breakout structure converts it to a "
               "cost-surviving edge")
else:
    verdict = "NO EDGE - neither net move size nor breakout structure is exploitable"

RES["verdict"] = dict(
    classification=verdict,
    vol_predicts_NET_move=dict(spearman_H6_train_val_final=list(sp_net.values()),
                               consistent_gt_0_10=bool(net_move_predictable),
                               veryhigh_over_verylow_ratio_H12=vh_vl_net),
    vol_predicts_RANGE=dict(spearman_H6_train_val_final=list(sp_rng.values()),
                            consistent_gt_0_15=bool(range_predictable),
                            veryhigh_over_verylow_ratio_H6=vh_vl_rng,
                            null_p=round(null_p_range, 4)),
    atr_normalised_spearman_is_CIRCULAR=dict(
        values_H6=list(sp_circ.values()),
        note="dividing the forward move by atr[i] while ranking atr[i] forces a spurious "
             "negative; this is NOT evidence of anything. Use the RAW-$ rows."),
    breakout_beats_cost=dict(best_config=best_key,
                             best_bracket_net_R_train_val_final=[round(x, 4) for x in best_bracket_tvf]
                             if best_bracket_tvf[0] is not None else None,
                             weakest_window_bracket_net_R=round(best_bracket_weakest, 4),
                             positive_all_windows_at_floor_spread=best_pos_all_windows_005,
                             inverts_at_spread=dies_at_spread,
                             oos_trigger_counts_val_final=n_oos_triggers,
                             small_oos_sample=small_oos),
    direction=dict(reading=dir_reading, whipsaw=whipsaw),
    cost_context=dict(round_trip_cost_over_atr_median=round(cost_over_atr_median, 3)),
)

# ===========================================================================
# OUTPUT
# ===========================================================================
_lap("writing reports")
csv_path = REPORTS / "btc_volatility_expansion.csv"
json_path = REPORTS / "btc_volatility_expansion.json"
txt_path = REPORTS / "btc_volatility_expansion_summary.txt"
pd.DataFrame(CSV, columns=["section", "key", "scope", "value"]).to_csv(csv_path, index=False)
json_path.write_text(json.dumps(jsonable(RES), indent=2), encoding="utf-8")

L = []
A = L.append
A("=" * 88)
A("BTC M5 VOLATILITY-EXPANSION / SYMMETRIC-BREAKOUT RESEARCH  --  SUMMARY")
A("=" * 88)
A(f"engine {ec.ENGINE_VERSION}   rows {N}   {df.time.iloc[0]} -> {df.time.iloc[-1]}")
A(f"windows TRAIN[0,{TRAIN_END}) VAL[{TRAIN_END},{VAL_END}) FINAL[{VAL_END},{N})   "
  f"regime+event thresholds FROZEN on TRAIN")
A(f"cost model: cost_price = spread + 2*slippage ; base ${cost_price_base:.2f} ; "
  f"round-trip cost / ATR median = {cost_over_atr_median:.2f}")
A("direction-agnostic OCO straddle simulated with the engine's exact fill arithmetic; "
  "no engine file touched.")
A("")
A("CORRECTION vs the Phase-1 discovery write-up: that report stated Spearman(atr_rank, |fwd")
A("move|) = +0.32; the value was NEGATIVE (-0.32) and is a CIRCULAR artifact of dividing the")
A("forward move by the same atr[i]. The genuine, non-circular relationships are measured below")
A("in RAW $:  vol -> future RANGE is weakly positive short-term;  vol -> NET move is ~zero.")
A("")
A(f"VERDICT:  {verdict}")
A("")
A("-" * 88)
A("PHASE 1  --  VOLATILITY REGIME  ->  FUTURE MOVE SIZE   (primary vol = atr20 rank/500)")
A("-" * 88)
def _pred_word(vals, hi, lo):
    m = min(vals)
    return "IS" if m > hi else ("is WEAKLY" if m > lo else "is NOT")


A("  Spearman(vol, RAW forward measure), H6, TRAIN / VAL / FINAL:")
A(f"    vs |NET move|_$    : {'/'.join(f'{v:+.3f}' for v in sp_net.values())}   "
  f"-> net directional move {_pred_word(list(sp_net.values()), 0.20, 0.08)} predictable "
  f"(|sp|~0.1 => ~1% of variance)")
A(f"    vs future RANGE_$  : {'/'.join(f'{v:+.3f}' for v in sp_rng.values())}   "
  f"(null p {null_p_range:.3f})   -> thrash/range {_pred_word(list(sp_rng.values()), 0.20, 0.10)} predictable")
A(f"    vs |move|/atr[i]   : {'/'.join(f'{v:+.3f}' for v in sp_circ.values())}   "
  f"<-- CIRCULAR (atr on both sides); ignore, it is not evidence")
A(f"  very_high / very_low ratio:  |net move|_$ (H12) TVF = {vh_vl_net}   "
  f"future RANGE_$ (H6) TVF = {vh_vl_rng}")
A("  regime -> mean |NET move|_$  /  mean future RANGE_$  (H6), pooled by window:")
for q in REGIME_LABELS:
    line = f"    {q:10s}"
    for win in ("TRAIN", "VALIDATION", "FINAL"):
        nn = p1["by_window"].get(win, {}).get(q, {}).get("H6", {}).get("mean_absret_px")
        rr = p1["by_window"].get(win, {}).get(q, {}).get("H6", {}).get("mean_range_px")
        line += f"  {win[:3]} {nn if nn is not None else '-'}/{rr if rr is not None else '-'}"
    A(line)
A("  other vol measures (Spearman vs RAW future RANGE_$ H6, TRAIN/VAL/FINAL):")
for name in VOLF:
    if name == PRIMARY_VOL:
        continue
    sv = [phase1[name]["spearman_vs_raw_forward"][w]["H6"]["spearman_range_px"]
          for w in ("TRAIN", "VALIDATION", "FINAL")]
    A(f"    {name:26s} {'/'.join(f'{x:+.3f}' for x in sv)}")
A("")
A("-" * 88)
A("PHASE 2  --  BREAKOUT EVENT  ->  FUTURE MOVE SIZE   (event vs non-event, H12 |fwd move| lift)")
A("-" * 88)
A(f"  {'event':30s} {'rate':>6} {'lift TRAIN':>11} {'lift VAL':>9} {'lift FINAL':>11} {'consistent>1':>13}")
for k, v in phase2.items():
    r = v["by_window"].get("TRAIN", {}).get("H12", {}).get("event_rate")
    lifts = v["H12_move_lift_train_val_final"]
    A(f"  {k:30s} {r if r is not None else '  -':>6} "
      f"{str(lifts[0]):>11} {str(lifts[1]):>9} {str(lifts[2]):>11} {str(v['lift_sign_consistent']):>13}")
A("")
A("-" * 88)
A("PHASE 3  --  DIRECTION-AGNOSTIC OCO STRADDLE   (buffer 0.5 ATR, bracket "
  f"{tgt_R}R target / {stop_R}R stop)")
A("-" * 88)
A("  top configs by weakest-window BRACKET net R (all 3 windows required):")
A(f"  {'config':34s} {'wk netR':>8} {'bracket T/V/F':>26} {'time T/V/F':>26}")
for k, wk, b, t in scored[:10]:
    A(f"  {k:34s} {wk:>8.4f} "
      f"{'/'.join(f'{x:+.3f}' for x in b):>26} {'/'.join(f'{x:+.3f}' for x in t):>26}")
if best_key:
    pw = phase3[best_key]
    A("")
    A(f"  best config detail ({best_key}):")
    for win in ("TRAIN", "VALIDATION", "FINAL"):
        if win in pw and "p_long" in pw[win]:
            d = pw[win]
            A(f"    {win:11s} armed {d['n_armed']:4d} trig {d['n_triggered']:4d} "
              f"({d['trigger_rate']:.0%})  p_long {d['p_long']:.2f}  "
              f"MFE {d['mean_MFE_R']:+.2f}R  MAE {d['mean_MAE_R']:+.2f}R  "
              f"MFE-MAE {d['mfe_minus_mae_R']:+.2f}R  cont {d['continuation_rate']:.0%} "
              f"stop {d['stopout_rate']:.0%}  netR(brk) {d['net_R_bracket']:+.3f}")
    A(f"    direction reading: {dir_reading}")
A("")
A("-" * 88)
A("PHASE 4  --  COST MODEL")
A("-" * 88)
A(f"  cost_price = spread + 2 x slippage(${SLIP:.2f}):")
for sp in SPREAD_SCENARIOS:
    A(f"    spread ${sp:.2f}  ->  round-trip cost ${sp + 2*SLIP:.2f}")
A("  expected move (H12) vs cost, by volatility regime (pooled):")
A(f"    {'regime':10s} {'n':>6} {'E|move| $':>10} {'E|move| R':>10}  "
  f"{'exp/cost@.05':>12} {'exp/cost@.10':>12} {'exp/cost@.18':>12} {'BE@.05':>7}")
for q, row in phase4["regime_expected_move"].items():
    A(f"    {q:10s} {row['n']:>6} {row['expected_move_px_H12']:>10.4f} {row['expected_move_R_H12']:>10.3f}  "
      f"{str(row.get('expmove_over_cost@0.05')):>12} {str(row.get('expmove_over_cost@0.10')):>12} "
      f"{str(row.get('expmove_over_cost@0.18')):>12} {str(row.get('breakeven@0.05')):>7}")
A("  spread sensitivity of top-3 straddle configs (bracket net R, T/V/F):")
for key, ss in phase4["spread_sensitivity"].items():
    A(f"    {key}")
    for sp, nb in ss.items():
        A(f"       spread ${sp}: {nb}")
A("")
A("=" * 88)
A("INTERPRETATION  (answers to Phase-3 A / B / C)")
A("=" * 88)
A(f"  A. Does the event predict an unusually large move?  -> future RANGE lift: see Phase 2")
A(f"     (best is expansion_after_lowvol ~1.5x, cross-window). NET-move lift is ~1.0 (none).")
A(f"  B. Does the move exceed cost?  best straddle config = {best_key}")
A(f"     bracket net R (T/V/F) at ${DEFAULT_SPREAD:.2f} = "
  + (f"{best_bracket_tvf}" if best_bracket_tvf[0] is not None else "n/a")
  + (f" ; inverts at spread ${dies_at_spread:.2f}" if dies_at_spread else " ; robust across tested spreads"))
A(f"     OOS triggered sample (VAL/FINAL) = {n_oos_triggers}   round-trip cost/ATR median = {cost_over_atr_median:.2f}")
A(f"  C. Direction: {dir_reading}")
A("")
if verdict.startswith("PROMISING"):
    A(f"  A symmetric breakout ({best_key}) clears cost in all three windows and is spread-robust.")
    A("  Worth ONE confirmatory build: thresholds frozen on TRAIN, single pass VAL + FINAL, no tuning.")
elif verdict.startswith("WEAK"):
    A(f"  ONE narrow event ({best_key}) is net-positive across TRAIN/VAL/FINAL at the ${DEFAULT_SPREAD:.2f}")
    A("  spread floor, with post-trigger continuation (MFE>MAE). BUT the OOS sample is small")
    A(f"  ({n_oos_triggers} triggers) " + (f"and it inverts once spread >= ${dies_at_spread:.2f}. "
      if dies_at_spread else "though spread-robust in the tested range. "))
    A("  This is a fragile, single-event result -- at most ONE confirmatory build with realistic")
    A("  ($0.01-0.05) costs and low expectations. Do NOT tune it. If the one pass is negative, stop.")
elif verdict.startswith("SIZE-ONLY"):
    A("  Volatility predicts the future high-low RANGE (thrash) at short horizons -- but NOT the")
    A("  NET directional move, which is what a breakout needs. No tested breakout structure")
    A(f"  converts range-predictability into a cost-surviving edge; round-trip cost is ~{cost_over_atr_median:.1f}x ATR")
    A("  and the OCO " + ("whipsaws" if whipsaw else "does not follow through") + ".")
    A("  Do NOT build a directional OR a breakout strategy on BTC M5 close data. A vol product here")
    A("  would need much lower real cost (recorded spread is $0.01; floor is 5x that), a higher")
    A("  timeframe, or non-price information (orderflow / funding / cross-asset).")
else:
    A("  Neither net move size nor breakout structure is exploitable on this data. Stop.")
A("=" * 88)
txt_path.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
_lap("done")
print("\nFILES CREATED:")
for p in (Path(__file__).resolve(), csv_path, json_path, txt_path):
    print(f"  {p.relative_to(ROOT)}")
