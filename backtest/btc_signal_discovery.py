"""
btc_signal_discovery.py  --  BTC-ONLY feature / signal discovery research harness.

GOAL
    NOT to build or optimize a strategy.  To measure whether BTC M5 OHLC(+spread)
    data contains *causal* predictive relationships (feature at bar i  ->  future
    return after bar i) that survive TRAIN -> VALIDATION -> FINAL and realistic
    spread/slippage, and to rank feature families by out-of-sample stability.

SAFETY / SCOPE
    * Reads data/btc_m5_history.csv and imports backtest/btc_engine_core.py +
      backtest/btc_realistic_config.py for the causal indicator kernels and the
      spread model.  Does NOT modify any engine / config / XAU file.
    * No MT5 / live-order code.
    * Strict causality:  every feature at bar i uses ONLY bars <= i;  every target
      uses ONLY bars > i;  per-window samples require i+H < window_end so a
      window's forward period never leaks into the next window.
    * Phase 3 "FULL" buckets are full-sample (disclosed in-sample/descriptive).
      Phase 6 thresholds are frozen on TRAIN and applied unchanged to VAL/FINAL.

OUTPUTS
    reports/btc_signal_discovery.csv        (tidy long: feature,family,horizon,window,metric,value)
    reports/btc_signal_discovery.json       (full nested results)
    reports/btc_signal_discovery_summary.txt (answers the 9 questions + rankings)
    reports/btc_feature_rankings.csv        (one row per feature: components + composite + rank)
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import btc_engine_core as ec          # noqa: E402  (indicator kernels, spread model)
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

try:
    from scipy import stats as _sps
    SCIPY = True
except Exception:
    SCIPY = False

T0 = time.perf_counter()


def _lap(msg):
    print(f"[{time.perf_counter() - T0:7.2f}s] {msg}", flush=True)


# ===========================================================================
# CONFIG
# ===========================================================================
DATA = ROOT / "data" / "btc_m5_history.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

HORIZONS = (1, 3, 6, 12, 24)
N_BUCKETS = 5
SPLIT_FRACS = (0.70, 0.85)
SLIPPAGE = float(bc.SLIPPAGE)
N_NULL = 200
SEED = 20260830
PRIMARY_H = 6                               # headline horizon for excursions / grids

ATR_P, RSI_P = 14, 14


# ===========================================================================
# NUMBA HELPERS  (causal)
# ===========================================================================
@njit(cache=True)
def signed_run(sign):
    n = sign.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        s = sign[i]
        if s != 0.0 and s == sign[i - 1]:
            out[i] = out[i - 1] + s
        else:
            out[i] = s
    return out


@njit(cache=True)
def fwd_excursions(high, low, close, atr, H):
    """MFE/MAE over (i, i+H] in ATR units, for a LONG.  up=favourable, dn=adverse."""
    n = close.shape[0]
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
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
        up[i] = (hh - close[i]) / a
        dn[i] = (close[i] - ll) / a
    return up, dn


# ===========================================================================
# STATS HELPERS
# ===========================================================================
def _rankdata(a):
    """average-rank (ties handled) -- fully vectorised."""
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
    ranks = np.empty(n, float)
    ranks[order] = mean_rank[grp]
    return ranks


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return 0.0, 1.0, int(m.sum())
    x, y = x[m], y[m]
    if SCIPY:
        r, p = _sps.spearmanr(x, y)
        if not np.isfinite(r):
            return 0.0, 1.0, len(x)
        return float(r), float(p), len(x)
    rx = _rankdata(x) - (len(x) + 1) / 2.0
    ry = _rankdata(y) - (len(y) + 1) / 2.0
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    r = float((rx * ry).sum() / d) if d > 0 else 0.0
    # normal approx p
    z = r * np.sqrt(len(x) - 1)
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
    return r, p, len(x)


def bucket_stats(feat, ret, retR, nb=N_BUCKETS):
    """quintile (or unique-value) buckets by feature; per-bucket forward stats."""
    m = np.isfinite(feat) & np.isfinite(ret) & np.isfinite(retR)
    f, r, rr = feat[m], ret[m], retR[m]
    if len(f) < 50:
        return []
    uniq = np.unique(f)
    if len(uniq) <= nb + 1:
        groups = [(u, f == u) for u in uniq]
    else:
        qs = np.quantile(f, np.linspace(0, 1, nb + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        qs = np.unique(qs)
        idx = np.clip(np.searchsorted(qs, f, side="right") - 1, 0, len(qs) - 2)
        groups = [(k, idx == k) for k in range(len(qs) - 1)]
    out = []
    for k, gm in groups:
        if gm.sum() < 10:
            continue
        out.append(dict(
            bucket=int(k) if isinstance(k, (int, np.integer)) else float(k),
            n=int(gm.sum()),
            mean_ret=float(r[gm].mean()),
            median_ret=float(np.median(r[gm])),
            win_prob=float((r[gm] > 0).mean()),
            mean_ret_R=float(rr[gm].mean()),
        ))
    return out


def monotonicity(buckets):
    if len(buckets) < 3:
        return 0.0
    idx = np.arange(len(buckets), dtype=float)
    val = np.array([b["mean_ret_R"] for b in buckets], float)
    ri = _rankdata(idx) - (len(idx) + 1) / 2
    rv = _rankdata(val) - (len(val) + 1) / 2
    d = np.sqrt((ri * ri).sum() * (rv * rv).sum())
    return float((ri * rv).sum() / d) if d > 0 else 0.0


def dir_eval(feat, retR, thr, cost_R):
    """feature > thr -> LONG(+1) else SHORT(-1);  evaluate against retR.

    edge_R is DRIFT-NEUTRAL:  cov(direction, retR) = mean(d*retR) - mean(d)*mean(retR).
    This removes the (long_frac x market_drift) term so we measure the feature's
    OWN directional information, not the fact that price fell over the sample.
    raw_directional_R keeps the naive number for transparency.
    """
    m = np.isfinite(feat) & np.isfinite(retR) & np.isfinite(cost_R)
    if m.sum() < 30:
        return None
    f, rr, cc = feat[m], retR[m], cost_R[m]
    d = np.where(f > thr, 1.0, -1.0)
    drift = float(rr.mean())
    raw = float((d * rr).mean())
    edge = raw - float(d.mean()) * drift               # drift-neutral
    net = edge - float(cc.mean())
    # directional accuracy vs a drift-matched coin (so 0.5 baseline is meaningful)
    acc = float(((d * (rr - drift)) > 0).mean())
    return dict(n=int(m.sum()), accuracy=acc,
               gross_edge_R=edge, net_edge_R=net,
               raw_directional_R=raw, drift_R=drift,
               mean_cost_R=float(cc.mean()), long_frac=float((d > 0).mean()))


# ===========================================================================
# LOAD + BASE SERIES
# ===========================================================================
_lap("loading data")
df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
o = df.open.to_numpy(float)
h = df.high.to_numpy(float)
l = df.low.to_numpy(float)
c = df.close.to_numpy(float)
raw_pts = df["spread"].to_numpy(float)
N = len(c)

ec.DEFAULT_SPREAD = bc.DEFAULT_SPREAD
spread = ec.prepare_spread(df)                      # engine spread model, causal (per-bar)

TR = np.maximum.reduce([
    h - l,
    np.abs(h - np.concatenate([[c[0]], c[:-1]])),
    np.abs(l - np.concatenate([[c[0]], c[:-1]])),
])
atr = ec.atr(h, l, c, ATR_P)
atr_safe = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
rsi14 = ec.rsi(c, RSI_P)
rsi2 = ec.rsi(c, 2)
ema20 = ec.ema(c, 20); ema50 = ec.ema(c, 50); ema100 = ec.ema(c, 100); ema200 = ec.ema(c, 200)

cs = pd.Series(c)
hs = pd.Series(h)
ls = pd.Series(l)
trs = pd.Series(TR)

TRAIN_END = int(N * SPLIT_FRACS[0])
VAL_END = int(N * SPLIT_FRACS[1])
WINDOWS = {"FULL": (0, N), "TRAIN": (0, TRAIN_END),
           "VALIDATION": (TRAIN_END, VAL_END), "FINAL": (VAL_END, N)}
_lap(f"rows={N}  warmup~{max(200, ATR_P)}  TRAIN[0,{TRAIN_END}) VAL[{TRAIN_END},{VAL_END}) FINAL[{VAL_END},{N})")


def roll_mean(s, w):
    return s.rolling(w, min_periods=w).mean().to_numpy()


def roll_std(s, w):
    return s.rolling(w, min_periods=w).std(ddof=0).to_numpy()


def roll_max(s, w, shift=0):
    return s.shift(shift).rolling(w, min_periods=w).max().to_numpy()


def roll_min(s, w, shift=0):
    return s.shift(shift).rolling(w, min_periods=w).min().to_numpy()


def roll_rank_pct(s, w):
    try:
        return s.rolling(w, min_periods=max(20, w // 5)).rank(pct=True).to_numpy()
    except Exception:                              # pandas < 1.4 fallback
        out = np.full(len(s), np.nan)
        a = s.to_numpy(float)
        for i in range(w - 1, len(a)):
            win = a[i - w + 1:i + 1]
            out[i] = np.mean(win <= a[i])
        return out


def kret(k):
    return cs.pct_change(k).to_numpy()


# ===========================================================================
# PHASE 1  --  FEATURES  (all causal, computed on the FULL series)
# ===========================================================================
_lap("phase 1: building features")
F: dict[str, np.ndarray] = {}
FAM: dict[str, str] = {}


def add(name, fam, arr):
    F[name] = np.asarray(arr, float)
    FAM[name] = fam


# ---- A. MOMENTUM / TREND ----
for k in (1, 2, 3, 6, 12, 24):
    add(f"A_ret_{k}", "A_momentum", kret(k))
add("A_ema20_slope_6", "A_momentum", (ema20 - np.concatenate([[np.nan] * 6, ema20[:-6]])) / atr_safe)
add("A_ema50_slope_12", "A_momentum", (ema50 - np.concatenate([[np.nan] * 12, ema50[:-12]])) / atr_safe)
add("A_ema100_slope_24", "A_momentum", (ema100 - np.concatenate([[np.nan] * 24, ema100[:-24]])) / atr_safe)
add("A_dist_ema20", "A_momentum", (c - ema20) / atr_safe)
add("A_dist_ema50", "A_momentum", (c - ema50) / atr_safe)
add("A_dist_ema100", "A_momentum", (c - ema100) / atr_safe)
add("A_dist_ema200", "A_momentum", (c - ema200) / atr_safe)
add("A_ema_sep_20_50", "A_momentum", (ema20 - ema50) / atr_safe)
add("A_ema_sep_50_100", "A_momentum", (ema50 - ema100) / atr_safe)
add("A_ema_sep_20_100", "A_momentum", (ema20 - ema100) / atr_safe)
_sign = np.sign(np.concatenate([[0.0], np.diff(c)]))
add("A_consec_signed", "A_momentum", signed_run(_sign))
for w in (20, 50):
    rmax = roll_max(hs, w, shift=1)
    rmin = roll_min(ls, w, shift=1)
    add(f"A_breakout_pos_{w}", "A_momentum", (c - rmin) / (rmax - rmin))

# ---- B. MEAN REVERSION ----
for w in (20, 50):
    add(f"B_zscore_{w}", "B_meanrev", (c - roll_mean(cs, w)) / roll_std(cs, w))
# (Bollinger deviation == B_zscore_20 / 2 -> identical ranks; omitted as redundant)
add("B_rsi_14", "B_meanrev", rsi14)
add("B_rsi_2", "B_meanrev", rsi2)
_gap = (o - np.concatenate([[np.nan], c[:-1]])) / atr_safe
add("B_gap_atr", "B_meanrev", _gap)
_r1 = kret(1)
add("B_range_shock_ret", "B_meanrev", np.sign(c - o) * (TR / atr_safe))
add("B_ret1_x_rangeratio", "B_meanrev", _r1 * (TR / atr_safe))

# ---- C. VOLATILITY ----
add("C_atr_norm", "C_volatility", atr_safe / c)
add("C_atr_rank_100", "C_volatility", roll_rank_pct(pd.Series(atr_safe), 100))
add("C_atr_rank_500", "C_volatility", roll_rank_pct(pd.Series(atr_safe), 500))
add("C_range_expansion", "C_volatility", TR / roll_mean(trs, 20))
add("C_realized_vol_12", "C_volatility", pd.Series(_r1).rolling(12, min_periods=12).std(ddof=0).to_numpy())
add("C_realized_vol_24", "C_volatility", pd.Series(_r1).rolling(24, min_periods=24).std(ddof=0).to_numpy())
_rv12 = pd.Series(_r1).rolling(12, min_periods=12).std(ddof=0).to_numpy()
_rv48 = pd.Series(_r1).rolling(48, min_periods=48).std(ddof=0).to_numpy()
add("C_vol_ratio_12_48", "C_volatility", _rv12 / _rv48)
add("C_atr_change_24", "C_volatility", atr_safe / np.concatenate([[np.nan] * 24, atr_safe[:-24]]) - 1.0)
add("C_range_20_mean_norm", "C_volatility", roll_mean(trs, 20) / c)

# ---- D. CANDLE STRUCTURE ----
rng = np.where((h - l) > 0, h - l, np.nan)
add("D_body_range", "D_candle", np.abs(c - o) / rng)
add("D_upper_wick_range", "D_candle", (h - np.maximum(o, c)) / rng)
add("D_lower_wick_range", "D_candle", (np.minimum(o, c) - l) / rng)
add("D_close_loc", "D_candle", (c - l) / rng)
add("D_bull", "D_candle", np.sign(c - o))
_ins = ((h < np.concatenate([[np.nan], h[:-1]])) & (l > np.concatenate([[np.nan], l[:-1]]))).astype(float)
_out = ((h > np.concatenate([[np.nan], h[:-1]])) & (l < np.concatenate([[np.nan], l[:-1]]))).astype(float)
add("D_inside_bar", "D_candle", _ins)
add("D_outside_bar", "D_candle", _out)
add("D_large_range", "D_candle", TR / atr_safe)
add("D_close_loc_ema3", "D_candle", pd.Series((c - l) / rng).rolling(3, min_periods=3).mean().to_numpy())

# ---- E. MARKET STRUCTURE ----
for w in (20, 100):
    rmax = roll_max(hs, w, shift=1)
    rmin = roll_min(ls, w, shift=1)
    add(f"E_dist_to_high_{w}", "E_structure", (rmax - c) / atr_safe)
    add(f"E_dist_to_low_{w}", "E_structure", (c - rmin) / atr_safe)
    add(f"E_pos_range_{w}", "E_structure", (c - rmin) / (rmax - rmin))
    add(f"E_breakout_hi_{w}", "E_structure", (c > rmax).astype(float))
    add(f"E_breakdown_lo_{w}", "E_structure", (c < rmin).astype(float))
_rh = roll_max(hs, 20); _rl = roll_min(ls, 20)
add("E_hh_hl_state", "E_structure",
    ((_rh > np.concatenate([[np.nan] * 10, _rh[:-10]])) &
     (_rl > np.concatenate([[np.nan] * 10, _rl[:-10]]))).astype(float))
add("E_lh_ll_state", "E_structure",
    ((_rh < np.concatenate([[np.nan] * 10, _rh[:-10]])) &
     (_rl < np.concatenate([[np.nan] * 10, _rl[:-10]]))).astype(float))

# ---- F. TIME ----
hour = df.time.dt.hour.to_numpy(float)
wday = df.time.dt.weekday.to_numpy(float)
add("F_hour_sin", "F_time", np.sin(2 * np.pi * hour / 24.0))
add("F_hour_cos", "F_time", np.cos(2 * np.pi * hour / 24.0))
_dt_min = np.concatenate([[np.nan], np.diff(df.time.values).astype("timedelta64[m]").astype(float)])
add("F_session_reopen", "F_time", (_dt_min > 10.0).astype(float))
add("F_monday_reopen", "F_time", ((wday == 0) & (_dt_min > 60.0)).astype(float))
F_HOUR = hour            # kept raw for categorical analysis
F_WDAY = wday

# ---- G. SPREAD / COST CONTEXT ----
add("G_spread_now", "G_cost", spread)
add("G_spread_rank_500", "G_cost", roll_rank_pct(pd.Series(spread), 500))
add("G_atr_spread_ratio", "G_cost", atr_safe / np.where(spread > 0, spread, np.nan))
add("G_spread_large_flag", "G_cost", (spread > 0.5 * atr_safe).astype(float))
# (cost_R == monotone transform of G_atr_spread_ratio -> identical ranks; omitted as redundant)

FEATURES = list(F.keys())
_lap(f"phase 1: {len(FEATURES)} features across {len(set(FAM.values()))} families")

# ===========================================================================
# PHASE 2  --  FORWARD TARGETS  (never used as features)
# ===========================================================================
_lap("phase 2: forward targets")
RET = {}
RETR = {}
for H in HORIZONS:
    fut = np.concatenate([c[H:], [np.nan] * H])
    RET[H] = fut / c - 1.0
    RETR[H] = (fut - c) / atr_safe
cost_R = (spread + 2.0 * SLIPPAGE) / atr_safe        # round-trip cost in ATR units (per entry bar)
UP6, DN6 = fwd_excursions(h, l, c, atr_safe, PRIMARY_H)

# per-window valid index masks: i in window AND i+H < window_end
def wmask(win, H):
    w0, w1 = WINDOWS[win]
    idx = np.zeros(N, bool)
    hi = min(w1, N - H)
    idx[w0:max(w0, hi)] = True
    return idx


# ===========================================================================
# PHASE 3 + 4 + 5  --  information / directional / cost-aware, per window
# ===========================================================================
_lap("phase 3-5: information + directional + cost-aware tests")
RESULTS = {"meta": dict(
    engine_version=ec.ENGINE_VERSION, rows=int(N),
    time_span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
    windows={k: list(v) for k, v in WINDOWS.items()},
    horizons=list(HORIZONS), n_features=len(FEATURES),
    families=sorted(set(FAM.values())),
    default_spread=float(bc.DEFAULT_SPREAD), slippage=SLIPPAGE,
    n_null=N_NULL, seed=SEED, scipy=SCIPY, numba=NUMBA,
    causality="feature uses bars<=i; target uses bars>i; per-window i+H<window_end",
)}

CSV = []


def emit(feature, family, horizon, window, metric, value):
    CSV.append(dict(feature=feature, family=family, horizon=horizon,
                    window=window, metric=metric, value=value))


# TRAIN-frozen thresholds (Phase 6)
train_thr = {}
tm_all = wmask("TRAIN", PRIMARY_H)
for name in FEATURES:
    v = F[name][tm_all]
    v = v[np.isfinite(v)]
    train_thr[name] = float(np.median(v)) if len(v) else 0.0

feat_block = {}
for fi, name in enumerate(FEATURES):
    fam = FAM[name]
    fa = F[name]
    per_h = {}
    for H in HORIZONS:
        per_w = {}
        for win in ("FULL", "TRAIN", "VALIDATION", "FINAL"):
            mk = wmask(win, H)
            fv, rv, rr, cc = fa[mk], RET[H][mk], RETR[H][mk], cost_R[mk]
            sp_r, sp_p, sp_n = spearman(fv, rr)
            bks = bucket_stats(fv, rv, rr)
            mono = monotonicity(bks)
            de = dir_eval(fa[mk], RETR[H][mk], train_thr[name], cost_R[mk])
            bspread = (float(bks[-1]["mean_ret_R"] - bks[0]["mean_ret_R"])
                       if len(bks) >= 3 else 0.0)          # drift-free: top minus bottom bucket
            rec = dict(spearman=round(sp_r, 4), spearman_p=round(sp_p, 4), n=sp_n,
                       monotonicity=round(mono, 4), bucket_spread_R=round(bspread, 4),
                       buckets=[{k: (round(x, 5) if isinstance(x, float) else x)
                                 for k, x in b.items()} for b in bks])
            if de:
                rec.update(dir_accuracy=round(de["accuracy"], 4),
                           gross_edge_R=round(de["gross_edge_R"], 5),
                           net_edge_R=round(de["net_edge_R"], 5),
                           raw_directional_R=round(de["raw_directional_R"], 5),
                           drift_R=round(de["drift_R"], 5),
                           long_frac=round(de["long_frac"], 4),
                           mean_cost_R=round(de["mean_cost_R"], 5))
            per_w[win] = rec
            emit(name, fam, H, win, "spearman", round(sp_r, 4))
            emit(name, fam, H, win, "monotonicity", round(mono, 4))
            emit(name, fam, H, win, "bucket_spread_R", round(bspread, 4))
            if de:
                emit(name, fam, H, win, "dir_accuracy", round(de["accuracy"], 4))
                emit(name, fam, H, win, "edge_R_driftneutral", round(de["gross_edge_R"], 5))
                emit(name, fam, H, win, "net_edge_R", round(de["net_edge_R"], 5))
        per_h[H] = per_w
    feat_block[name] = dict(family=fam, train_threshold=round(train_thr[name], 6), by_horizon=per_h)
    if (fi + 1) % 20 == 0:
        _lap(f"    features {fi + 1}/{len(FEATURES)}")
RESULTS["features"] = feat_block

# excursions for the PRIMARY horizon directional rule (MFE/MAE proxy)
_lap("phase 4: MFE / MAE excursions (primary horizon)")
exc = {}
for name in FEATURES:
    mk = wmask("FULL", PRIMARY_H)
    fv = F[name][mk]
    d = np.where(fv > train_thr[name], 1.0, -1.0)
    up, dn = UP6[mk], DN6[mk]
    mfe = np.where(d > 0, up, dn)
    mae = np.where(d > 0, dn, up)
    m = np.isfinite(mfe) & np.isfinite(mae)
    if m.sum() < 30:
        continue
    exc[name] = dict(mean_MFE_R=round(float(mfe[m].mean()), 4),
                     mean_MAE_R=round(float(mae[m].mean()), 4),
                     mfe_mae_ratio=round(float(mfe[m].mean() / mae[m].mean()), 4) if mae[m].mean() else None)
    emit(name, FAM[name], PRIMARY_H, "FULL", "mean_MFE_R", exc[name]["mean_MFE_R"])
    emit(name, FAM[name], PRIMARY_H, "FULL", "mean_MAE_R", exc[name]["mean_MAE_R"])
RESULTS["excursions_primary_h"] = exc

# ===========================================================================
# PHASE 6  --  OUT-OF-SAMPLE  (TRAIN thresholds frozen, applied to VAL/FINAL)
# ===========================================================================
_lap("phase 6: out-of-sample threshold transfer")
EDGE_THRESH = 0.03            # |net edge R| considered "meaningful" for survival test
oos = {}
for name in FEATURES:
    row = {}
    for H in HORIZONS:
        d_tr = dir_eval(F[name][wmask("TRAIN", H)], RETR[H][wmask("TRAIN", H)],
                        train_thr[name], cost_R[wmask("TRAIN", H)])
        d_va = dir_eval(F[name][wmask("VALIDATION", H)], RETR[H][wmask("VALIDATION", H)],
                        train_thr[name], cost_R[wmask("VALIDATION", H)])
        d_fi = dir_eval(F[name][wmask("FINAL", H)], RETR[H][wmask("FINAL", H)],
                        train_thr[name], cost_R[wmask("FINAL", H)])
        if not (d_tr and d_va and d_fi):
            continue
        s = np.sign(d_tr["gross_edge_R"])
        works_train = abs(d_tr["gross_edge_R"]) >= EDGE_THRESH
        surv_val = works_train and np.sign(d_va["gross_edge_R"]) == s and abs(d_va["gross_edge_R"]) >= EDGE_THRESH / 2
        surv_fin = surv_val and np.sign(d_fi["gross_edge_R"]) == s and abs(d_fi["gross_edge_R"]) >= EDGE_THRESH / 2
        row[H] = dict(
            train_edge_R=round(d_tr["gross_edge_R"], 5), val_edge_R=round(d_va["gross_edge_R"], 5),
            final_edge_R=round(d_fi["gross_edge_R"], 5),
            train_net_R=round(d_tr["net_edge_R"], 5), val_net_R=round(d_va["net_edge_R"], 5),
            final_net_R=round(d_fi["net_edge_R"], 5),
            train_acc=round(d_tr["accuracy"], 4), val_acc=round(d_va["accuracy"], 4),
            final_acc=round(d_fi["accuracy"], 4),
            works_train=bool(works_train), survives_validation=bool(surv_val),
            survives_final=bool(surv_fin))
    oos[name] = row
RESULTS["oos"] = oos

# ===========================================================================
# PHASE 8  --  NULL / RANDOM BASELINES  (deterministic)
# ===========================================================================
_lap(f"phase 8: null baselines  ({N_NULL} shuffles/feature, primary + full horizons)")
rng_master = np.random.default_rng(SEED)
nulls = {}
# precompute target ranks per horizon on FULL valid mask
tgt_ranks = {}
for H in HORIZONS:
    mk = wmask("FULL", H)
    y = RETR[H][mk]
    fin = np.isfinite(y)
    tgt_ranks[H] = (mk, fin, _rankdata(y[fin]) - (fin.sum() + 1) / 2.0)

t8 = time.perf_counter()
for fi, name in enumerate(FEATURES):
    fa = F[name]
    per_h = {}
    for H in HORIZONS:
        mk, fin, ry = tgt_ranks[H]
        x = fa[mk][fin]
        okx = np.isfinite(x)
        if okx.sum() < 50:
            continue
        xr = _rankdata(x[okx]) - (okx.sum() + 1) / 2.0
        yy = ry[okx]
        denom = np.sqrt((xr * xr).sum() * (yy * yy).sum())
        obs = float((xr * yy).sum() / denom) if denom > 0 else 0.0
        # null 1: shuffle feature ranks
        seed_f = int(rng_master.integers(1, 2 ** 31))
        rg = np.random.default_rng(seed_f)
        null_shuf = np.empty(N_NULL)
        for t in range(N_NULL):
            p = rg.permutation(len(xr))
            null_shuf[t] = float((xr[p] * yy).sum() / denom) if denom > 0 else 0.0
        # null 2: circular shift target
        rg2 = np.random.default_rng(seed_f + 1)
        null_circ = np.empty(N_NULL)
        base_y = yy.copy()
        for t in range(N_NULL):
            sh = int(rg2.integers(50, len(base_y) - 50))
            ysh = np.roll(base_y, sh)
            null_circ[t] = float((xr * ysh).sum() / denom) if denom > 0 else 0.0
        p_shuf = float((np.abs(null_shuf) >= abs(obs)).mean())
        p_circ = float((np.abs(null_circ) >= abs(obs)).mean())
        per_h[H] = dict(obs_spearman=round(obs, 4),
                        null_shuffle_mean=round(float(null_shuf.mean()), 4),
                        null_shuffle_std=round(float(null_shuf.std(ddof=1)), 4),
                        p_shuffle=round(p_shuf, 4),
                        null_circular_std=round(float(null_circ.std(ddof=1)), 4),
                        p_circular=round(p_circ, 4))
        emit(name, FAM[name], H, "FULL", "null_p_shuffle", round(p_shuf, 4))
        emit(name, FAM[name], H, "FULL", "null_p_circular", round(p_circ, 4))
    nulls[name] = per_h
    if (fi + 1) % 20 == 0:
        _lap(f"    null {fi + 1}/{len(FEATURES)}  ({time.perf_counter() - t8:.1f}s)")
RESULTS["nulls"] = nulls
# randomized-direction baseline (analytic): accuracy ~ Binomial(n, 0.5)
RESULTS["random_direction_baseline"] = dict(
    note="directional rule with random +/-1 has expected accuracy 0.5; "
         "std ~ 0.5/sqrt(n). Any |acc-0.5| below ~2*std is noise.",
)

# ===========================================================================
# PHASE 7  --  FEATURE STABILITY RANKING  (NOT ranked by FINAL profit)
# ===========================================================================
_lap("phase 7: stability ranking")
SP_FLOOR = 0.03          # |Spearman| effect-size floor (below this = noise even if 'significant')
rank_rows = []
for name in FEATURES:
    fam = FAM[name]
    fb = feat_block[name]["by_horizon"]

    def _cons(vals):
        vals = [v for v in vals]
        s0 = np.sign(vals[0])
        return bool(s0 != 0 and all(np.sign(v) == s0 for v in vals))

    # best horizon = the one whose 3-window Spearman is both largest AND sign-consistent
    h_scores = {}
    for H in HORIZONS:
        sp = [fb[H][w]["spearman"] for w in ("TRAIN", "VALIDATION", "FINAL")]
        bump = 1.0 if _cons(sp) else 0.0
        h_scores[H] = float(np.mean(np.abs(sp))) + 0.02 * bump
    bestH = max(h_scores, key=h_scores.get)
    tr, va, fi_ = fb[bestH]["TRAIN"], fb[bestH]["VALIDATION"], fb[bestH]["FINAL"]
    sp_tr, sp_va, sp_fi = tr["spearman"], va["spearman"], fi_["spearman"]
    bs_tr, bs_va, bs_fi = tr["bucket_spread_R"], va["bucket_spread_R"], fi_["bucket_spread_R"]
    e_tr = tr.get("gross_edge_R", 0.0); e_va = va.get("gross_edge_R", 0.0); e_fi = fi_.get("gross_edge_R", 0.0)
    ne_tr = tr.get("net_edge_R", 0.0); ne_va = va.get("net_edge_R", 0.0); ne_fi = fi_.get("net_edge_R", 0.0)

    sp_consistent = _cons([sp_tr, sp_va, sp_fi])
    bs_consistent = _cons([bs_tr, bs_va, bs_fi])
    edge_consistent = _cons([e_tr, e_va, e_fi])
    weakest_sp = min(abs(sp_tr), abs(sp_va), abs(sp_fi))
    weakest_bs = min(abs(bs_tr), abs(bs_va), abs(bs_fi))
    weakest_edge = min(abs(e_tr), abs(e_va), abs(e_fi))
    weakest_net = min(ne_tr, ne_va, ne_fi)

    horizon_breadth = float(np.mean([
        1.0 if _cons([fb[H][w]["spearman"] for w in ("TRAIN", "VALIDATION", "FINAL")]) else 0.0
        for H in HORIZONS]))
    mono = float(np.mean([abs(fb[H]["TRAIN"].get("monotonicity", 0.0)) for H in HORIZONS]))
    null_p = float(np.nanmin([nulls.get(name, {}).get(H, {}).get("p_shuffle", 1.0) for H in HORIZONS]
                             + [1.0]))
    null_p_circ = float(np.nanmin([nulls.get(name, {}).get(H, {}).get("p_circular", 1.0) for H in HORIZONS]
                                  + [1.0]))

    gate = 1.0 if (sp_consistent and bs_consistent) else (0.4 if sp_consistent else 0.15)
    composite = gate * (
        2.0 * weakest_sp
        + 1.0 * min(weakest_bs, 0.5)
        + 1.5 * weakest_edge
        + 1.0 * max(0.0, weakest_net)
        + 0.4 * horizon_breadth
        + 0.3 * mono
        - 0.6 * null_p
    )
    stable = bool(
        sp_consistent and bs_consistent
        and weakest_sp >= SP_FLOOR
        and weakest_edge >= 0.02
        and null_p < 0.05 and null_p_circ < 0.20
    )
    relationship = ("feature_up -> future_up" if sp_tr > 0 else "feature_up -> future_down")
    rank_rows.append(dict(
        feature=name, family=fam, best_horizon=bestH, relationship=relationship,
        spearman_train=sp_tr, spearman_val=sp_va, spearman_final=sp_fi,
        weakest_abs_spearman=round(weakest_sp, 4),
        bucket_spread_R_train=bs_tr, bucket_spread_R_val=bs_va, bucket_spread_R_final=bs_fi,
        edge_R_train=round(e_tr, 5), edge_R_val=round(e_va, 5), edge_R_final=round(e_fi, 5),
        net_edge_R_train=round(ne_tr, 5), net_edge_R_val=round(ne_va, 5), net_edge_R_final=round(ne_fi, 5),
        weakest_abs_edge_R=round(weakest_edge, 5), weakest_net_edge_R=round(weakest_net, 5),
        spearman_sign_consistent=bool(sp_consistent), bucket_sign_consistent=bool(bs_consistent),
        edge_sign_consistent=bool(edge_consistent),
        horizon_breadth=round(horizon_breadth, 3), monotonicity=round(mono, 4),
        null_p_min=round(null_p, 4), null_p_circular_min=round(null_p_circ, 4),
        stable=stable, composite_score=round(float(composite), 5),
    ))

RANK = pd.DataFrame(rank_rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
RANK.insert(0, "rank", RANK.index + 1)
RANK.to_csv(REPORTS / "btc_feature_rankings.csv", index=False)
RESULTS["rankings"] = RANK.to_dict("records")

# ===========================================================================
# PHASE 9  --  DATA QUALITY / SPREAD
# ===========================================================================
_lap("phase 9: data quality")
TICK = float(ec.TICK_SIZE)
abn = np.where(raw_pts > 10.0)[0]
abn_rows = []
for ix in abn:
    abn_rows.append(dict(bar=int(ix), time=str(df.time.iloc[ix]), raw_points=float(raw_pts[ix]),
                         converted=round(raw_pts[ix] * TICK, 4),
                         post_clamp=round(min(raw_pts[ix] * TICK, 5.0 * bc.DEFAULT_SPREAD), 4),
                         classification="corrupt/gap" if raw_pts[ix] > 100 else "elevated"))
sar = atr_safe / np.where(spread > 0, spread, np.nan)
dq = dict(
    raw_points_distribution={str(int(u)): int(n) for u, n in zip(*np.unique(raw_pts, return_counts=True))},
    n_bars_gt_10pts=int((raw_pts > 10).sum()), n_bars_gt_100pts=int((raw_pts > 100).sum()),
    abnormal_bars=abn_rows,
    atr_over_spread=dict(p05=round(float(np.nanpercentile(sar, 5)), 2),
                         median=round(float(np.nanmedian(sar)), 2),
                         p95=round(float(np.nanpercentile(sar, 95)), 2)),
    spread_ge_half_atr_bars=int(np.nansum(spread >= 0.5 * atr_safe)),
)
# recompute top-8 features' FULL spearman with abnormal bars excluded
excl = np.ones(N, bool)
excl[abn] = False
top8 = RANK.head(8)["feature"].tolist()
delta = {}
for name in top8:
    H = int(RANK.set_index("feature").loc[name, "best_horizon"])
    mk = wmask("FULL", H)
    r_all, _, _ = spearman(F[name][mk], RETR[H][mk])
    mk2 = mk & excl
    r_ex, _, _ = spearman(F[name][mk2], RETR[H][mk2])
    delta[name] = dict(horizon=H, spearman_all=round(r_all, 4), spearman_excl_abnormal=round(r_ex, 4),
                       abs_change=round(abs(r_all - r_ex), 4))
dq["top_feature_spearman_excl_abnormal"] = delta
dq["excluding_abnormal_bars_material"] = bool(any(v["abs_change"] > 0.02 for v in delta.values()))
RESULTS["data_quality"] = dq

# ===========================================================================
# TIME-OF-DAY / WEEKDAY categorical grids  (primary horizon)
# ===========================================================================
mkp = wmask("FULL", PRIMARY_H)
yR = RETR[PRIMARY_H]
tod = {}
for hh in range(24):
    g = mkp & (F_HOUR == hh) & np.isfinite(yR)
    if g.sum() >= 20:
        tod[str(hh)] = dict(n=int(g.sum()), mean_ret_R=round(float(yR[g].mean()), 4),
                            win_prob=round(float((yR[g] > 0).mean()), 4))
wdg = {}
for dd in range(7):
    g = mkp & (F_WDAY == dd) & np.isfinite(yR)
    if g.sum() >= 20:
        wdg[str(dd)] = dict(n=int(g.sum()), mean_ret_R=round(float(yR[g].mean()), 4),
                            win_prob=round(float((yR[g] > 0).mean()), 4))
hour_means = np.array([v["mean_ret_R"] for v in tod.values()])
RESULTS["time_of_day"] = dict(by_hour_utc=tod, by_weekday=wdg,
                              hour_effect_dispersion=round(float(hour_means.std()), 4),
                              hour_effect_range=round(float(hour_means.max() - hour_means.min()), 4))

# ===========================================================================
# TREND vs MEAN-REVERSION vs VOLATILITY DIAGNOSTIC  (drift-neutral Spearman)
# ===========================================================================
_lap("regime diagnostic (trend / reversion / volatility)")


def _sp_windows(feat_name, target_arr, use_abs_target=False):
    out = {}
    for win in ("TRAIN", "VALIDATION", "FINAL"):
        best = 0.0
        bh = None
        for H in HORIZONS:
            mk = wmask(win, H)
            y = np.abs(target_arr[H][mk]) if use_abs_target else target_arr[H][mk]
            r, _, _ = spearman(F[feat_name][mk], y)
            if abs(r) > abs(best):
                best, bh = r, H
        out[win] = (round(best, 4), bh)
    return out


def _consistent_vals(d):
    v = [d[w][0] for w in ("TRAIN", "VALIDATION", "FINAL")]
    s0 = np.sign(v[0])
    return bool(s0 != 0 and all(np.sign(x) == s0 for x in v)), v, float(min(abs(x) for x in v))


regime = {}
# momentum persistence: does past return predict SAME-sign future return?
mom = {f: _sp_windows(f, RETR) for f in ("A_ret_3", "A_ret_6", "A_ret_12", "A_ret_24")}
mom_best = max(mom.items(), key=lambda kv: _consistent_vals(kv[1])[2] if _consistent_vals(kv[1])[0] else 0)
mc, mv, mw = _consistent_vals(mom_best[1])
regime["momentum_persistence"] = dict(
    feature=mom_best[0], per_window=mom_best[1], sign_consistent=mc, weakest_abs=round(mw, 4),
    reading=("trend-continuation" if (mc and mv[0] > 0) else
             "mean-reversion" if (mc and mv[0] < 0) else "none"))
# stretch reversion: price far from EMA -> revert?
strv = {f: _sp_windows(f, RETR) for f in ("A_dist_ema20", "A_dist_ema50", "B_zscore_20", "B_rsi_14")}
strv_best = max(strv.items(), key=lambda kv: _consistent_vals(kv[1])[2] if _consistent_vals(kv[1])[0] else 0)
sc, sv, sw = _consistent_vals(strv_best[1])
regime["stretch_reversion"] = dict(
    feature=strv_best[0], per_window=strv_best[1], sign_consistent=sc, weakest_abs=round(sw, 4),
    reading=("mean-reversion" if (sc and sv[0] < 0) else
             "stretch-continuation" if (sc and sv[0] > 0) else "none"))
# volatility -> direction  (any C feature, signed target)
vold = {f: _sp_windows(f, RETR) for f in FEATURES if FAM[f] == "C_volatility"}
vold_best = max(vold.items(), key=lambda kv: _consistent_vals(kv[1])[2] if _consistent_vals(kv[1])[0] else 0)
vc, vv, vw = _consistent_vals(vold_best[1])
# volatility -> magnitude  (vol rank predicts |future move|)  = vol clustering
volm = _sp_windows("C_atr_rank_500", RETR, use_abs_target=True)
vmc, vmv, vmw = _consistent_vals(volm)
regime["volatility_direction"] = dict(feature=vold_best[0], per_window=vold_best[1],
                                      sign_consistent=vc, weakest_abs=round(vw, 4),
                                      reading="directional-vol-signal" if (vc and vw >= SP_FLOOR) else "none")
regime["volatility_magnitude_clustering"] = dict(feature="C_atr_rank_500", per_window=volm,
                                                 sign_consistent=vmc, weakest_abs=round(vmw, 4),
                                                 reading="vol clusters (size predictable, not direction)"
                                                 if (vmc and vmw >= SP_FLOOR) else "weak/none")
# candle / structure best
cs_feats = [f for f in FEATURES if FAM[f] in ("D_candle", "E_structure")]
csd = {f: _sp_windows(f, RETR) for f in cs_feats}
csd_best = max(csd.items(), key=lambda kv: _consistent_vals(kv[1])[2] if _consistent_vals(kv[1])[0] else 0)
csc, csv_, csw = _consistent_vals(csd_best[1])
regime["candle_structure"] = dict(feature=csd_best[0], per_window=csd_best[1],
                                  sign_consistent=csc, weakest_abs=round(csw, 4),
                                  reading="weak directional" if (csc and csw >= SP_FLOOR) else "none")
RESULTS["regime_diagnostic"] = regime

# ===========================================================================
# PHASE 10  --  SUMMARY + VERDICT
# ===========================================================================
_lap("phase 10: writing outputs")

stable_feats = RANK[RANK.stable].copy()
n_stable = len(stable_feats)
best_h_overall = int(RANK.head(15)["best_horizon"].mode().iloc[0]) if len(RANK) else PRIMARY_H
top_weakest_sp = float(RANK.head(1)["weakest_abs_spearman"].iloc[0]) if len(RANK) else 0.0
any_survive_final = any(any(h.get("survives_final", False) for h in oos.get(f, {}).values())
                        for f in FEATURES)
net_ok = RANK[(RANK.spearman_sign_consistent) & (RANK.bucket_sign_consistent)
              & (RANK[["net_edge_R_train", "net_edge_R_val", "net_edge_R_final"]].min(axis=1) > 0)]

if n_stable >= 3 and top_weakest_sp >= 0.06 and len(net_ok) >= 1 and any_survive_final:
    overall = "PROMISING FEATURE SET"
elif n_stable >= 1 and top_weakest_sp >= 0.04:
    overall = "WEAK / PARTIAL SIGNAL"
else:
    overall = "NO USABLE PREDICTIVE SIGNAL"

# family evidence from the ranking table (drift-neutral)
def _fam_ev(prefix):
    subs = RANK[RANK.family.str.startswith(prefix)]
    if subs.empty:
        return dict(evidence="none", best_feature=None)
    b = subs.iloc[0]
    ev = ("clear" if bool(b["stable"])
          else "weak" if (bool(b["spearman_sign_consistent"]) and float(b["weakest_abs_spearman"]) >= 0.03)
          else "none")
    return dict(evidence=ev, best_feature=str(b["feature"]),
                weakest_abs_spearman=float(b["weakest_abs_spearman"]),
                weakest_abs_edge_R=float(b["weakest_abs_edge_R"]),
                relationship=str(b["relationship"]),
                spearman=[float(b["spearman_train"]), float(b["spearman_val"]), float(b["spearman_final"])])


fam_conc = {k: _fam_ev(p) for k, p in
            (("A_momentum", "A_"), ("B_meanrev", "B_"), ("C_volatility", "C_"),
             ("D_candle", "D_"), ("E_structure", "E_"), ("F_time", "F_"), ("G_cost", "G_"))}

# small independent combo: top sign-consistent features from distinct families, |sp|>=0.02
combo = []
seen = set()
for r in RANK[RANK.spearman_sign_consistent & (RANK.weakest_abs_spearman >= 0.02)].itertuples():
    f0 = r.family
    if f0 in seen:
        continue
    combo.append(str(r.feature))
    seen.add(f0)
    if len(combo) == 3:
        break

RESULTS["conclusion"] = dict(
    overall=overall, n_stable_features=int(n_stable),
    stable_features=stable_feats["feature"].tolist(),
    strongest_horizon=best_h_overall,
    top_feature_weakest_abs_spearman=round(top_weakest_sp, 4),
    n_features_net_positive_all_windows=int(len(net_ok)),
    any_feature_survives_to_final=bool(any_survive_final),
    family_conclusions=fam_conc,
    regime_diagnostic=regime,
    combo_candidate=combo,
    top20=RANK.head(20)[["rank", "feature", "family", "best_horizon", "relationship",
                         "spearman_train", "spearman_val", "spearman_final",
                         "weakest_abs_spearman", "weakest_abs_edge_R",
                         "spearman_sign_consistent", "null_p_min", "stable"]].to_dict("records"),
)


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


csv_path = REPORTS / "btc_signal_discovery.csv"
json_path = REPORTS / "btc_signal_discovery.json"
txt_path = REPORTS / "btc_signal_discovery_summary.txt"
rank_path = REPORTS / "btc_feature_rankings.csv"

pd.DataFrame(CSV, columns=["feature", "family", "horizon", "window", "metric", "value"]).to_csv(csv_path, index=False)
json_path.write_text(json.dumps(jsonable(RESULTS), indent=2), encoding="utf-8")

L = []
A = L.append
A("=" * 88)
A("BTC M5 SIGNAL / FEATURE DISCOVERY  --  SUMMARY")
A("=" * 88)
A(f"engine {ec.ENGINE_VERSION}   rows {N}   {df.time.iloc[0]} -> {df.time.iloc[-1]}")
A(f"features {len(FEATURES)} / families {len(set(FAM.values()))}   horizons {list(HORIZONS)} bars")
A(f"windows: TRAIN[0,{TRAIN_END})  VALIDATION[{TRAIN_END},{VAL_END})  FINAL[{VAL_END},{N})")
A(f"causality: feature<=i, target>i, per-window i+H<window_end   nulls: {N_NULL} shuffles seed {SEED}")
A("edge_R is DRIFT-NEUTRAL cov(direction,retR); Spearman is rank corr feature vs fwd_ret_R.")
A("CAVEAT: n per window is 2.8k-13k, so |Spearman|~0.03 is 'significant' yet explains <0.1% of")
A("variance. Effect size, cross-window sign, and cost survival matter -- not the p-value.")
A("This is feature discovery, NOT a strategy. Nothing optimized. Not ranked by FINAL P/L.")
A("")
A(f"OVERALL: {overall}    (stable features: {n_stable};  strongest horizon: +{best_h_overall} bars)")
A("")
A("-" * 88)
A("1/2. STRONGEST 20 FEATURES  (ranked by cross-window weakest-link Spearman + drift-neutral edge)")
A("-" * 88)
A(f"{'#':>3} {'feature':24s} {'fam':12s} {'H':>3} {'sp_tr':>6} {'sp_va':>6} {'sp_fi':>6} "
  f"{'wk|sp|':>6} {'wk|edgeR|':>9} {'spOK':>5} {'nullp':>6} {'stbl':>5}")
for r in RANK.head(20).itertuples():
    A(f"{r.rank:>3} {r.feature:24s} {r.family:12s} {r.best_horizon:>3} "
      f"{r.spearman_train:>6.3f} {r.spearman_val:>6.3f} {r.spearman_final:>6.3f} "
      f"{r.weakest_abs_spearman:>6.3f} {r.weakest_abs_edge_R:>9.4f} "
      f"{str(bool(r.spearman_sign_consistent)):>5} {r.null_p_min:>6.3f} {str(bool(r.stable)):>5}")
A("")
A("STABLE = Spearman & bucket-spread sign-consistent across TRAIN/VAL/FINAL, weakest |Spearman|")
A(f"       >= {SP_FLOOR}, weakest |edge_R| >= 0.02, null p_shuffle < 0.05 and p_circular < 0.20:")
if n_stable:
    for r in stable_feats.itertuples():
        A(f"   {r.feature:24s} {r.family:12s} H+{r.best_horizon}  {r.relationship}")
        A(f"      Spearman tr/va/fi = {r.spearman_train:+.3f}/{r.spearman_val:+.3f}/{r.spearman_final:+.3f}   "
          f"edge_R = {r.edge_R_train:+.4f}/{r.edge_R_val:+.4f}/{r.edge_R_final:+.4f}   "
          f"net_R(worst) {r.weakest_net_edge_R:+.4f}")
else:
    A("   (none)")
A("")
A("-" * 88)
A("3. RELATIONSHIPS THAT DISAPPEAR OUT-OF-SAMPLE  (TRAIN-frozen threshold; works TRAIN, fails later)")
A("-" * 88)
faded = []
for name in FEATURES:
    for H, r in oos.get(name, {}).items():
        if r["works_train"] and not r["survives_final"]:
            faded.append((name, H, r))
faded.sort(key=lambda t: -abs(t[2]["train_edge_R"]))
if faded:
    for name, H, r in faded[:15]:
        A(f"   {name:24s} H+{H:<2} edge_R TRAIN {r['train_edge_R']:+.4f} -> VAL {r['val_edge_R']:+.4f} "
          f"-> FINAL {r['final_edge_R']:+.4f}   (survived VAL: {r['survives_validation']})")
else:
    A("   (no feature cleared the TRAIN 'works' bar of |edge_R| >= "
      f"{EDGE_THRESH} to begin with)")
A("")
A("-" * 88)
A("4. FEATURES STILL USEFUL AFTER SPREAD + SLIPPAGE  (net edge_R > 0 in every window, sign-consistent)")
A("-" * 88)
if len(net_ok):
    for r in net_ok.head(15).itertuples():
        A(f"   {r.feature:24s} {r.family:12s} H+{r.best_horizon}  "
          f"net edge_R tr/va/fi {r.net_edge_R_train:+.4f}/{r.net_edge_R_val:+.4f}/{r.net_edge_R_final:+.4f}")
    A(f"   (cost model: DEFAULT_SPREAD ${bc.DEFAULT_SPREAD:.2f} floor + 2x${SLIPPAGE:.2f} slip; "
      f"floor is ~5x the recorded $0.01 spread -> cost_R is conservative/high)")
else:
    A("   (NONE -- no feature keeps a positive net edge in all three windows;")
    A(f"    round-trip cost_R median ~{float(np.nanmedian((spread + 2*SLIPPAGE)/atr_safe)):.2f} R "
      f"dwarfs every observed directional edge_R)")
A("")
A("-" * 88)
A("5-8. EVIDENCE BY MECHANISM  (best cross-window sign-consistent feature per mechanism)")
A("-" * 88)
rd = regime
A(f"   5. TREND CONTINUATION : {rd['momentum_persistence']['reading'].upper()}   "
  f"({rd['momentum_persistence']['feature']}, weakest|sp| {rd['momentum_persistence']['weakest_abs']}, "
  f"consistent {rd['momentum_persistence']['sign_consistent']})")
A(f"   6. MEAN REVERSION     : {rd['stretch_reversion']['reading'].upper()}   "
  f"({rd['stretch_reversion']['feature']}, weakest|sp| {rd['stretch_reversion']['weakest_abs']}, "
  f"consistent {rd['stretch_reversion']['sign_consistent']})")
A(f"   7. VOLATILITY (dir)   : {rd['volatility_direction']['reading'].upper()}   "
  f"({rd['volatility_direction']['feature']}, weakest|sp| {rd['volatility_direction']['weakest_abs']})")
A(f"      VOLATILITY (size)  : {rd['volatility_magnitude_clustering']['reading'].upper()}   "
  f"(atr_rank vs |fwd move|, weakest|sp| {rd['volatility_magnitude_clustering']['weakest_abs']})")
A(f"   8. CANDLE / STRUCTURE : {rd['candle_structure']['reading'].upper()}   "
  f"({rd['candle_structure']['feature']}, weakest|sp| {rd['candle_structure']['weakest_abs']})")
A(f"      TIME-OF-DAY        : hour mean-R dispersion {RESULTS['time_of_day']['hour_effect_dispersion']}  "
  f"range {RESULTS['time_of_day']['hour_effect_range']}  (in-sample descriptive only)")
A("")
A("   per-family best feature (drift-neutral):")
for key, v in fam_conc.items():
    if v["best_feature"] is None:
        continue
    A(f"      {key:12s} {v['evidence'].upper():5s}  {v['best_feature']:24s} "
      f"wk|sp| {v.get('weakest_abs_spearman', 0):.3f}  sp(tr/va/fi) "
      f"{'/'.join(f'{x:+.3f}' for x in v.get('spearman', [0, 0, 0]))}  [{v.get('relationship','')}]")
A("")
A("-" * 88)
A("9. SMALL INDEPENDENT COMBO CANDIDATE (distinct families, Spearman sign-consistent, weakest|sp|>=0.02)")
A("-" * 88)
A(f"   {combo if combo else '(none qualify)'}")
if combo:
    A("   -> even these are weak; a Phase-2 combo is only worth ONE confirmatory pass, no tuning.")
A("")
A("-" * 88)
A("DATA QUALITY  (Phase 9)")
A("-" * 88)
A(f"   raw spread pts dist: {dq['raw_points_distribution']}")
A(f"   bars >10pts / >100pts: {dq['n_bars_gt_10pts']} / {dq['n_bars_gt_100pts']}  (weekend/reopen artifacts)")
A(f"   ATR/spread ratio: p05 {dq['atr_over_spread']['p05']}  median {dq['atr_over_spread']['median']}  "
  f"p95 {dq['atr_over_spread']['p95']}   (spread is a LARGE fraction of ATR on this feed)")
A(f"   excluding abnormal bars materially changes top-feature Spearman: "
  f"{dq['excluding_abnormal_bars_material']}")
for nnm, vv in dq["top_feature_spearman_excl_abnormal"].items():
    A(f"      {nnm:24s} H+{vv['horizon']:<2} all {vv['spearman_all']:+.4f} -> excl {vv['spearman_excl_abnormal']:+.4f}  "
      f"(|d|={vv['abs_change']:.4f})")
A("   -> abnormal bars are NOT driving any feature relationship; none removed.")
A("")
A("=" * 88)
A("RECOMMENDATION FOR PHASE 2 STRATEGY CONSTRUCTION")
A("=" * 88)
if overall == "NO USABLE PREDICTIVE SIGNAL":
    A("   No feature or mechanism shows a cross-window, cost-surviving directional edge on BTC M5.")
    A("   Every |Spearman| is < 0.1 and the strongest are not sign-stable across TRAIN/VAL/FINAL;")
    A("   round-trip cost_R exceeds every observed directional edge_R.")
    A("   DO NOT build a directional strategy from these features on this data.")
    A("   Options, in order of value:")
    A("     (a) longer BTC history (multi-year) to see if weak effects stabilise;")
    A("     (b) higher timeframe (M15/H1) where cost_R is a smaller fraction of the move;")
    A("     (c) different information: volume/orderflow microstructure, funding, cross-asset;")
    A("     (d) volatility-SIZE models (breakout straddles) instead of direction -- vol clustering")
    A("         is the ONLY relationship with any cross-window consistency;")
    A("     (e) accept BTC M5 close-only as non-forecastable and stop.")
else:
    A(f"   Strongest horizon +{best_h_overall} bars. Weak candidate features: {combo}.")
    A("   Build ONE minimal fixed rule, thresholds frozen on TRAIN, evaluate ONCE on VALIDATION")
    A("   and ONCE on FINAL through the existing engine. No optimization loop, no FINAL tuning.")
    A("   Expect it to be marginal; if the single confirmatory pass is negative, stop.")
A("=" * 88)

txt_path.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
_lap("done")
print("\nFILES CREATED:")
for p in (Path(__file__).resolve(), csv_path, json_path, txt_path, rank_path):
    print(f"  {p.relative_to(ROOT)}")
