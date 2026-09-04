"""
Engine parity test.

Verifies that the optimizer (optimizer_realistic_numba.py) and the standalone
(realistic_xauusd.py) run the *same* simulation, now that both route through
backtest/engine_core.py.

Fixed configuration (task Phase 5):

    FAST=10  SLOW=80  TREND=200  RSI_PERIOD=7  RSI_BUY=60  RSI_SELL=40
    ATR_PERIOD=20  SL=1.5  TP=2.0

Window: training only, bars [0, 70000).

Three code paths are compared:

  A  standalone wiring  -- engine_core.compute_indicators + prepare_spread,
                           engine_core.run_window_logged  (per-trade log)
  B  optimizer wiring   -- optimizer.build_indicator_matrices (shared-indicator
                           matrices) + row indexing, engine_core.run_window_logged
  C  optimizer kernel   -- optimizer.run_chunk (the actual @njit(parallel) prange
                           kernel) restricted to this one config

A vs B  proves the optimizer's matrix/indicator plumbing is identical.
A vs C  proves the compiled prange kernel is identical.

Tolerance is 1e-9 (absolute).  There is no arithmetic difference between the
paths -- they call the identical engine_core._backtest -- so anything above
floating-point formatting noise is a real divergence and is reported, not hidden.

Run directly for a readable report:   python tests/test_engine_parity.py
Run under pytest:                     pytest -q tests/test_engine_parity.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))

import engine_core as ec                      # noqa: E402
import optimizer_realistic_numba as opt       # noqa: E402

CFG = dict(
    fast=10, slow=80, trend=200,
    rsi_period=7, rsi_buy=60.0, rsi_sell=40.0,
    atr_period=20, sl=1.5, tp=2.0,
)
I0, I1 = 0, 70000
ATOL = 1e-9

SUMMARY_KEYS = [
    "trades", "final_balance", "net_pl", "win_rate", "profit_factor",
    "max_dd_abs", "max_dd_pct", "wins", "losses", "commission",
    "gross_profit", "gross_loss",
]
TRADE_KEYS = ["entry_i", "exit_i", "dir", "lots", "entry_px", "exit_px", "pnl"]


# --------------------------------------------------------------------------- #
# shared data
# --------------------------------------------------------------------------- #
def _load():
    df = ec.load_ohlc(ec.DATA)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    sp = ec.prepare_spread(df)
    return df, o, h, l, c, sp


# --------------------------------------------------------------------------- #
# the three paths
# --------------------------------------------------------------------------- #
def path_A_standalone(o, h, l, c, sp):
    ef, es, et, rv, av, warmup = ec.compute_indicators(
        o, h, l, c, CFG["fast"], CFG["slow"], CFG["trend"],
        CFG["rsi_period"], CFG["atr_period"],
    )
    res, trades = ec.run_window_logged(
        o, h, l, c, ef, es, et, rv, av, sp,
        I0, I1, warmup, CFG["rsi_buy"], CFG["rsi_sell"], CFG["sl"], CFG["tp"],
    )
    return ec.summary_dict(res), trades


def path_B_optimizer_matrices(o, h, l, c, sp):
    (ema_mat, rsi_mat, atr_mat,
     ema_row, rsi_row, atr_row) = opt.build_indicator_matrices(o, h, l, c)

    ef = ema_mat[ema_row[CFG["fast"]]]
    es = ema_mat[ema_row[CFG["slow"]]]
    et = ema_mat[ema_row[CFG["trend"]]]
    rv = rsi_mat[rsi_row[CFG["rsi_period"]]]
    av = atr_mat[atr_row[CFG["atr_period"]]]
    warmup = max(CFG["fast"], CFG["slow"], CFG["trend"],
                 CFG["rsi_period"], CFG["atr_period"]) + 5

    res, trades = ec.run_window_logged(
        o, h, l, c, ef, es, et, rv, av, sp,
        I0, I1, warmup, opt.RSI_BUY_TH, opt.RSI_SELL_TH, CFG["sl"], CFG["tp"],
    )
    return ec.summary_dict(res), trades


def path_C_optimizer_kernel(o, h, l, c, sp):
    (ema_mat, rsi_mat, atr_mat,
     ema_row, rsi_row, atr_row) = opt.build_indicator_matrices(o, h, l, c)

    target = (CFG["fast"], CFG["slow"], CFG["trend"],
              CFG["rsi_period"], CFG["atr_period"], CFG["sl"], CFG["tp"])
    assert target in opt.build_configs(), "parity config not in the optimizer grid"

    (fast_row, slow_row, trend_row, warmup_arr,
     rsi_r, atr_r, slm, tpm) = opt.configs_to_arrays(
        [target], ema_row, rsi_row, atr_row)

    out = opt.run_chunk(
        np.ascontiguousarray(o), np.ascontiguousarray(h),
        np.ascontiguousarray(l), np.ascontiguousarray(c), np.ascontiguousarray(sp),
        ema_mat, rsi_mat, atr_mat,
        fast_row, slow_row, trend_row, warmup_arr,
        rsi_r, atr_r, slm, tpm,
        opt.RSI_BUY_TH, opt.RSI_SELL_TH,
        I1, 85000,           # train_end = I1 ; val_end unused for the train row
    )
    r = out[0]
    # run_chunk train columns: 0 trades, 1 win_rate, 2 net_pl, 3 pf, 4 dd_abs
    return {
        "trades": int(round(r[0])),
        "win_rate": r[1],
        "net_pl": r[2],
        "profit_factor": r[3],
        "max_dd_abs": r[4],
    }


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def _cmp_summary(a, b, keys):
    diffs = []
    for k in keys:
        va, vb = a[k], b[k]
        if isinstance(va, float) and (np.isinf(va) or np.isinf(vb)):
            ok = (va == vb)
        else:
            ok = abs(float(va) - float(vb)) <= ATOL
        if not ok:
            diffs.append((k, va, vb, float(vb) - float(va)))
    return diffs


def _first_trade_divergence(ta, tb):
    na, nb = len(ta["pnl"]), len(tb["pnl"])
    n = min(na, nb)
    for i in range(n):
        for k in TRADE_KEYS:
            va, vb = ta[k][i], tb[k][i]
            if abs(float(va) - float(vb)) > ATOL:
                return i, k, va, vb
    if na != nb:
        return n, "<trade count>", na, nb
    return None


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #
_CACHE = {}


def _run_all():
    if "done" in _CACHE:
        return _CACHE
    df, o, h, l, c, sp = _load()
    sA, tA = path_A_standalone(o, h, l, c, sp)
    sB, tB = path_B_optimizer_matrices(o, h, l, c, sp)
    sC = path_C_optimizer_kernel(o, h, l, c, sp)
    _CACHE.update(done=True, sA=sA, tA=tA, sB=sB, tB=tB, sC=sC)
    return _CACHE


def test_summary_A_vs_B():
    r = _run_all()
    diffs = _cmp_summary(r["sA"], r["sB"], SUMMARY_KEYS)
    assert not diffs, f"A vs B summary divergence: {diffs}"


def test_trades_A_vs_B():
    r = _run_all()
    div = _first_trade_divergence(r["tA"], r["tB"])
    assert div is None, f"A vs B first trade divergence: index={div[0]} field={div[1]} A={div[2]} B={div[3]}"


def test_summary_A_vs_C_kernel():
    r = _run_all()
    a, cc = r["sA"], r["sC"]
    diffs = _cmp_summary(
        {k: a[k] for k in cc}, cc,
        ["trades", "win_rate", "net_pl", "profit_factor", "max_dd_abs"],
    )
    assert not diffs, f"A vs C (run_chunk kernel) divergence: {diffs}"


def test_trade_count_matches_recorded():
    r = _run_all()
    assert r["sA"]["trades"] == len(r["tA"]["pnl"])


# --------------------------------------------------------------------------- #
# readable CLI report
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, float):
        if np.isinf(v):
            return "inf"
        return f"{v:.10g}"
    return str(v)


def main():
    print("=" * 78)
    print("ENGINE PARITY  --  EMA 10/80/200  RSI 7 (60/40)  ATR 20  SL 1.5  TP 2.0")
    print(f"window: training bars [{I0}, {I1})   tolerance: {ATOL:g} abs")
    print("=" * 78)

    r = _run_all()
    sA, sB, sC, tA, tB = r["sA"], r["sB"], r["sC"], r["tA"], r["tB"]

    print(f"\n{'metric':<16}{'A standalone':>20}{'B opt-matrices':>20}{'C opt-kernel':>18}")
    print("-" * 78)
    for k in SUMMARY_KEYS:
        ck = _fmt(sC[k]) if k in sC else ""
        print(f"{k:<16}{_fmt(sA[k]):>20}{_fmt(sB[k]):>20}{ck:>18}")

    d_ab = _cmp_summary(sA, sB, SUMMARY_KEYS)
    d_ac = _cmp_summary(
        {k: sA[k] for k in sC}, sC,
        ["trades", "win_rate", "net_pl", "profit_factor", "max_dd_abs"],
    )
    tdiv = _first_trade_divergence(tA, tB)

    print("\n" + "-" * 78)
    print(f"A vs B summary : {'MATCH' if not d_ab else 'DIVERGENCE ' + str(d_ab)}")
    print(f"A vs C kernel  : {'MATCH' if not d_ac else 'DIVERGENCE ' + str(d_ac)}")
    print(f"A vs B trades  : {len(tA['pnl'])} vs {len(tB['pnl'])} recorded | "
          f"{'MATCH' if tdiv is None else 'FIRST DIVERGENCE ' + str(tdiv)}")

    if tdiv is None and len(tA["pnl"]):
        i = 0
        print(f"\nfirst trade (sanity): entry_i={tA['entry_i'][0]} exit_i={tA['exit_i'][0]} "
              f"dir={tA['dir'][0]} lots={tA['lots'][0]:.2f} "
              f"entry={tA['entry_px'][0]:.5f} exit={tA['exit_px'][0]:.5f} pnl={tA['pnl'][0]:.6f}")
        print(f"last  trade (sanity): entry_i={tA['entry_i'][-1]} exit_i={tA['exit_i'][-1]} "
              f"dir={tA['dir'][-1]} lots={tA['lots'][-1]:.2f} "
              f"entry={tA['entry_px'][-1]:.5f} exit={tA['exit_px'][-1]:.5f} pnl={tA['pnl'][-1]:.6f}")

    ok = (not d_ab) and (not d_ac) and (tdiv is None)
    print("\n" + "=" * 78)
    print("PARITY: PASS" if ok else "PARITY: FAIL")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
