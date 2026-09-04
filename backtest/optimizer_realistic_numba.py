
import os
import sys
import time
import math
import json
import hashlib
import argparse
import itertools
import datetime as _dt
import numpy as np
import pandas as pd

try:
    from numba import njit, prange, get_num_threads
    NUMBA = True
except ImportError:  # pragma: no cover - fallback path
    NUMBA = False

    def njit(*args, **kwargs):
        """No-op @njit replacement so the file still runs without numba."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(func):
            return func

        return deco

    prange = range

    def get_num_threads():
        return 1


try:                       # `python backtest/optimizer_realistic_numba.py`
    import engine_core as ec
    from engine_core import (
        ema, rsi, atr, load_ohlc, prepare_spread, run_window_fast, warmup_for,
    )
except ImportError:        # `python -m backtest.optimizer_realistic_numba`
    from backtest import engine_core as ec
    from backtest.engine_core import (
        ema, rsi, atr, load_ohlc, prepare_spread, run_window_fast, warmup_for,
    )

_ENGINE_CORE_FILE = os.path.abspath(ec.__file__)


DATA = "data/xauusd_m5_5y.csv"
OUT = "reports/xauusd_realistic_optimization.csv"
FINAL_OUT = "reports/xauusd_realistic_final_test.csv"
CHECKPOINT = "reports/xauusd_realistic_optimization.checkpoint.csv"
# Sidecar written next to the checkpoint.  Binds the checkpoint rows to the exact
# engine / grid / RSI / split / cost / risk / data configuration that produced
# them.  A checkpoint with no sidecar, or a sidecar that does not match the
# current configuration, is NOT resumable (see check_checkpoint_fingerprint).
FINGERPRINT = "reports/xauusd_realistic_optimization.checkpoint.fingerprint.json"

# Account / contract / friction constants  --  SINGLE SOURCE: engine_core.py.
# The numba kernel run_window_fast() bakes engine_core's values in at compile
# time; these names are re-exported here only for logging / back-compat.
INITIAL_BALANCE = ec.INITIAL_BALANCE
RISK_PCT = ec.RISK_PER_TRADE
TICK_VALUE = ec.TICK_VALUE
TICK_SIZE = ec.TICK_SIZE
VALUE_PER_PRICE_PER_LOT = ec.VALUE_PER_POINT
MIN_LOT = ec.MIN_LOT
LOT_STEP = ec.LOT_STEP
MAX_LOT = ec.MAX_LOT
DEFAULT_SPREAD = ec.DEFAULT_SPREAD
SLIPPAGE = ec.SLIPPAGE            # 0.01  (this file used a divergent 0.02 before)

# Chronological split (indices into the candle array).
TRAIN_END = 70000
VAL_END = 85000
TOTAL = 100000

FAST = [5, 10, 15, 20, 30]
SLOW = [30, 40, 50, 60, 80]
TREND = [100, 150, 200]
RSI_PERIOD = [7, 10, 14, 20]
RSI_BUY = [55, 60]
RSI_SELL = [40, 45]
ATR_PERIOD = [10, 14, 20]
SL = [1.0, 1.5, 2.0, 2.5]
TP = [1.5, 2.0, 2.5, 3.0, 3.5]

# Strategy entry thresholds.
# PINNED for the first aligned run (task Phase 4): RSI_BUY / RSI_SELL are NOT an
# optimization dimension yet.  The previous code used RSI_BUY[0] = 55 here, which
# (a) never matched the standalone's 60 and (b) left the advertised RSI_BUY /
# RSI_SELL grids as dead code.  RSI_PERIOD stays a real optimization dimension.
RSI_BUY_TH = 60.0
RSI_SELL_TH = 40.0

# Filters applied to a configuration after backtesting.
MIN_TRAIN_TRADES = 30
MIN_TRAIN_PF = 1.05
MIN_VAL_TRADES = 20
MIN_VAL_PF = 1.05

CHUNK = 500  # configs per checkpoint / progress step


# --------------------------------------------------------------------------- #
# Simulation engine  --  imported verbatim from the shared source of truth.
#
# ema / rsi / atr / floor_lot / the whole trade loop used to be re-implemented
# here (and had drifted: wrong ATR seed, spread not floored, SLIPPAGE 0.02,
# close>trend filter, RSI_BUY 55, no min-stop / margin cap / equity kill-switch,
# half the round-trip spread).  They are now engine_core._backtest, reached via
# engine_core.run_window_fast(), exactly as realistic_xauusd.py uses it.
#
# This file's ONLY remaining jobs: build the shared indicator matrices, fan the
# parameter grid across prange, checkpoint, filter, and run the untouched final
# test.  Optimization dimension is now fully separated from simulation semantics.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Parallel driver: one independent configuration per prange iteration.
# --------------------------------------------------------------------------- #
@njit(cache=True, parallel=True)
def run_chunk(
    op,
    hi,
    lo,
    cl,
    spread,
    ema_mat,
    rsi_mat,
    atr_mat,
    fast_row,
    slow_row,
    trend_row,
    warmup_arr,
    rsi_row,
    atr_row,
    slm_arr,
    tpm_arr,
    rsi_buy,
    rsi_sell,
    train_end,
    val_end,
):
    m = fast_row.shape[0]
    out = np.zeros((m, 10), dtype=np.float64)

    for k in prange(m):
        ef_arr = ema_mat[fast_row[k]]
        es_arr = ema_mat[slow_row[k]]
        et_arr = ema_mat[trend_row[k]]
        r_arr = rsi_mat[rsi_row[k]]
        a_arr = atr_mat[atr_row[k]]

        wu = warmup_arr[k]
        slm = slm_arr[k]
        tpm = tpm_arr[k]

        # Train and validation windows, identical engine to realistic_xauusd.py.
        t = run_window_fast(
            op, hi, lo, cl, ef_arr, es_arr, et_arr, r_arr, a_arr, spread,
            0, train_end, wu, rsi_buy, rsi_sell, slm, tpm,
        )
        v = run_window_fast(
            op, hi, lo, cl, ef_arr, es_arr, et_arr, r_arr, a_arr, spread,
            train_end, val_end, wu, rsi_buy, rsi_sell, slm, tpm,
        )

        # _backtest return layout: 0 trades, 2 net_pl, 3 win_rate, 4 pf,
        #                          5 max_dd_abs
        out[k, 0] = t[0]
        out[k, 1] = t[3]
        out[k, 2] = t[2]
        out[k, 3] = t[4]
        out[k, 4] = t[5]
        out[k, 5] = v[0]
        out[k, 6] = v[3]
        out[k, 7] = v[2]
        out[k, 8] = v[4]
        out[k, 9] = v[5]

    return out


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data():
    # Single source: engine_core.load_ohlc (time sort + dedupe) and
    # engine_core.prepare_spread (points -> price, floored at DEFAULT_SPREAD,
    # NaN / non-finite / <= floor -> DEFAULT_SPREAD).  Identical to the arrays
    # realistic_xauusd.py feeds its engine.
    df = load_ohlc(DATA)
    spread = prepare_spread(df)

    op = np.ascontiguousarray(df["open"].to_numpy(np.float64))
    hi = np.ascontiguousarray(df["high"].to_numpy(np.float64))
    lo = np.ascontiguousarray(df["low"].to_numpy(np.float64))
    cl = np.ascontiguousarray(df["close"].to_numpy(np.float64))
    spread = np.ascontiguousarray(spread)

    return op, hi, lo, cl, spread


# --------------------------------------------------------------------------- #
# Indicator matrices (contiguous, one row per distinct period)
# --------------------------------------------------------------------------- #
def build_indicator_matrices(op, hi, lo, cl):
    print("Preparing shared indicators...", flush=True)

    ema_periods = sorted(set(FAST + SLOW + TREND))
    rsi_periods = sorted(set(RSI_PERIOD))
    atr_periods = sorted(set(ATR_PERIOD))

    ema_mat = np.ascontiguousarray(
        np.stack([ema(cl, p) for p in ema_periods])
    )
    rsi_mat = np.ascontiguousarray(
        np.stack([rsi(cl, p) for p in rsi_periods])
    )
    atr_mat = np.ascontiguousarray(
        np.stack([atr(hi, lo, cl, p) for p in atr_periods])
    )

    ema_row = {p: i for i, p in enumerate(ema_periods)}
    rsi_row = {p: i for i, p in enumerate(rsi_periods)}
    atr_row = {p: i for i, p in enumerate(atr_periods)}

    return ema_mat, rsi_mat, atr_mat, ema_row, rsi_row, atr_row


def build_configs():
    configs = list(
        itertools.product(FAST, SLOW, TREND, RSI_PERIOD, ATR_PERIOD, SL, TP)
    )
    configs = [x for x in configs if x[0] < x[1] and x[1] < x[2]]
    return configs


def configs_to_arrays(configs, ema_row, rsi_row, atr_row):
    fast_row = np.array([ema_row[c[0]] for c in configs], dtype=np.int64)
    slow_row = np.array([ema_row[c[1]] for c in configs], dtype=np.int64)
    trend_row = np.array([ema_row[c[2]] for c in configs], dtype=np.int64)
    rsi_r = np.array([rsi_row[c[3]] for c in configs], dtype=np.int64)
    atr_r = np.array([atr_row[c[4]] for c in configs], dtype=np.int64)
    slm = np.array([c[5] for c in configs], dtype=np.float64)
    tpm = np.array([c[6] for c in configs], dtype=np.float64)
    # standalone warmup rule -- single source: engine_core.warmup_for
    warmup_arr = np.array(
        [warmup_for(c[0], c[1], c[2], c[3], c[4]) for c in configs],
        dtype=np.int64,
    )
    return (
        fast_row, slow_row, trend_row, warmup_arr, rsi_r, atr_r, slm, tpm,
    )


# --------------------------------------------------------------------------- #
# Numba warm-up
# --------------------------------------------------------------------------- #
def warmup():
    print("Compiling / warming Numba engine...", flush=True)
    t0 = time.time()

    x = np.ascontiguousarray(np.linspace(1000.0, 1001.0, 64))
    o = x.copy()
    h = np.ascontiguousarray(x + 0.05)
    l = np.ascontiguousarray(x - 0.05)
    c = x.copy()
    sp = np.ascontiguousarray(np.full(64, DEFAULT_SPREAD))

    ema(c, 3)
    rsi(c, 3)
    atr(h, l, c, 3)

    ema_mat = np.ascontiguousarray(np.stack([ema(c, 3), ema(c, 4)]))
    rsi_mat = np.ascontiguousarray(np.stack([rsi(c, 3)]))
    atr_mat = np.ascontiguousarray(np.stack([atr(h, l, c, 3)]))

    one = np.array([0], dtype=np.int64)
    two = np.array([1], dtype=np.int64)
    per = np.array([9], dtype=np.int64)   # warmup_arr for the warm-up call
    fone = np.array([1.5], dtype=np.float64)
    ftwo = np.array([2.0], dtype=np.float64)

    run_chunk(
        o, h, l, c, sp,
        ema_mat, rsi_mat, atr_mat,
        one, two, two, per,
        one, one, fone, ftwo,
        RSI_BUY_TH, RSI_SELL_TH,
        20, 30,
    )

    print(
        f"Numba compilation complete. ({time.time() - t0:.1f}s)",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
CP_COLUMNS = [
    "ema_fast", "ema_slow", "ema_trend", "rsi_period", "atr_period", "sl", "tp",
    "train_trades", "train_winrate", "train_profit", "train_pf", "train_dd",
    "val_trades", "val_winrate", "val_profit", "val_pf", "val_dd",
]


def load_checkpoint(path):
    if not os.path.exists(path):
        return 0, []
    try:
        cp = pd.read_csv(path)
    except Exception:
        return 0, []
    rows = [tuple(r) for r in cp[CP_COLUMNS].to_numpy()]
    return len(rows), rows


def append_checkpoint(path, rows):
    write_header = not os.path.exists(path)
    df = pd.DataFrame(rows, columns=CP_COLUMNS)
    df.to_csv(path, mode="a", header=write_header, index=False)


def fmt_eta(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# --------------------------------------------------------------------------- #
# Checkpoint fingerprinting
#
# The checkpoint stores only (config -> metrics) rows.  Resuming is safe ONLY if
# every input that can move those metrics is identical to when the rows were
# written: the engine semantics, the parameter grid, the pinned RSI thresholds,
# the train/val/test boundaries, the execution-cost and risk/lot/margin
# constants, and the price data itself.  All of that is hashed into a sidecar
# JSON.  On ANY mismatch the optimizer refuses to resume and tells the user to
# run --fresh; the stale checkpoint is archived with a timestamp, NEVER deleted,
# and old and new results are NEVER mixed in one file.
# --------------------------------------------------------------------------- #
def _sha256_file(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_bufsize), b""):
            h.update(block)
    return h.hexdigest()


def _data_identity(path):
    """Prefer SHA-256; fall back to size + mtime only if the file can't be read."""
    d = {"path": path.replace("\\", "/")}
    try:
        d["sha256"] = _sha256_file(path)
    except OSError:
        st = os.stat(path)
        d["sha256"] = None
        d["size_bytes"] = st.st_size
        d["mtime_ns"] = st.st_mtime_ns
    return d


def build_fingerprint(n_bars, train_end, val_end):
    """Canonical dict of everything that can change a checkpoint metric + its sha."""
    payload = {
        "fingerprint_version": 1,
        "engine": {
            "engine_version": ec.ENGINE_VERSION,
            "engine_core_sha256": _sha256_file(_ENGINE_CORE_FILE),
        },
        "grid": {
            "FAST": list(FAST), "SLOW": list(SLOW), "TREND": list(TREND),
            "RSI_PERIOD": list(RSI_PERIOD), "ATR_PERIOD": list(ATR_PERIOD),
            "SL": list(SL), "TP": list(TP),
            "filter": "fast < slow < trend",
            "config_count": len(build_configs()),
        },
        "rsi_thresholds": {
            "RSI_BUY_TH": RSI_BUY_TH,
            "RSI_SELL_TH": RSI_SELL_TH,
        },
        "splits": {
            "TRAIN_END": int(train_end),
            "VAL_END": int(val_end),
            "TOTAL": int(TOTAL),
            "n_bars": int(n_bars),
        },
        "execution_costs": {
            "DEFAULT_SPREAD": ec.DEFAULT_SPREAD,
            "SLIPPAGE": ec.SLIPPAGE,
            "COMMISSION_PER_LOT": ec.COMMISSION_PER_LOT,
            "MIN_STOP_TICKS": ec.MIN_STOP_TICKS,
            "MIN_STOP_DIST": ec.MIN_STOP_DIST,
            "TICK_SIZE": ec.TICK_SIZE,
            "TICK_VALUE": ec.TICK_VALUE,
            "VALUE_PER_POINT": ec.VALUE_PER_POINT,
        },
        "risk": {
            "INITIAL_BALANCE": ec.INITIAL_BALANCE,
            "RISK_PER_TRADE": ec.RISK_PER_TRADE,
            "EQUITY_STOP_FRACTION": ec.EQUITY_STOP_FRACTION,
            "EQUITY_STOP": ec.EQUITY_STOP,
            "MAX_MARGIN_FRACTION": ec.MAX_MARGIN_FRACTION,
            "LEVERAGE": ec.LEVERAGE,
            "CONTRACT_SIZE": ec.CONTRACT_SIZE,
            "MIN_LOT": ec.MIN_LOT,
            "LOT_STEP": ec.LOT_STEP,
            "MAX_LOT": ec.MAX_LOT,
        },
        "checkpoint_schema": list(CP_COLUMNS),
        "data": _data_identity(DATA),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return payload, payload_sha


def _fp_flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_fp_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def write_fingerprint(payload, payload_sha):
    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    doc = {
        "fingerprint_sha256": payload_sha,
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "payload": payload,
    }
    with open(FINGERPRINT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")


def archive_stale_checkpoint():
    """Move an existing checkpoint (+ sidecar) aside with a UTC timestamp.

    NEVER deletes.  Returns the list of (src, dst) moves performed.
    """
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = []
    for src in (CHECKPOINT, FINGERPRINT):
        if not os.path.exists(src):
            continue
        root, ext = os.path.splitext(src)
        dst = f"{root}.superseded-{ts}{ext}"
        i = 1
        while os.path.exists(dst):
            dst = f"{root}.superseded-{ts}.{i}{ext}"
            i += 1
        os.rename(src, dst)
        moved.append((src, dst))
    return moved


def check_checkpoint_fingerprint(current_payload, current_sha):
    """(ok, message).  ok=True -> safe to resume.  Only call when CHECKPOINT exists."""
    if not os.path.exists(FINGERPRINT):
        return False, (
            "the checkpoint has NO fingerprint sidecar "
            f"({os.path.basename(FINGERPRINT)} is missing).\n"
            "  It predates checkpoint fingerprinting, or was written by a "
            "different\n"
            "  engine / grid / data configuration.  It cannot be verified as "
            "compatible\n"
            "  with the current engine and MUST NOT be resumed."
        )
    try:
        with open(FINGERPRINT, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return False, f"the fingerprint sidecar is unreadable ({exc})."

    if doc.get("fingerprint_sha256") == current_sha:
        return True, "fingerprint matches the current configuration."

    stored_flat = _fp_flatten(doc.get("payload", {}))
    cur_flat = _fp_flatten(current_payload)
    diffs = []
    for k in sorted(set(stored_flat) | set(cur_flat)):
        sv = stored_flat.get(k, "<absent>")
        cv = cur_flat.get(k, "<absent>")
        if sv != cv:
            diffs.append(f"      {k}:  checkpoint={sv!r}   current={cv!r}")
    body = "\n".join(diffs) if diffs else (
        f"      sha differs: {doc.get('fingerprint_sha256')} != {current_sha}"
    )
    return False, "fingerprint MISMATCH - the configuration changed:\n" + body


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        nargs="?",
        type=int,
        const=10,
        default=None,
        help="Run only the first N configurations (default 10) as a smoke test.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a NEW sweep from configuration 0. Any existing checkpoint "
             "(and fingerprint) is archived with a UTC timestamp - never "
             "deleted, never resumed, never mixed with the new results.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("REALISTIC XAUUSD NUMBA RESEARCH OPTIMIZER")
    print("=" * 70)

    if not os.path.exists(DATA):
        raise RuntimeError(f"Missing {DATA}")

    op, hi, lo, cl, spread = load_data()
    n = len(cl)

    val_end = min(VAL_END, n)
    train_end = min(TRAIN_END, val_end)

    configs = build_configs()

    print(f"CPU cores:          {os.cpu_count()}")
    print(f"Configurations:     {len(configs):,}")
    print(f"Training candles:   {train_end:,}")
    print(f"Validation candles: {val_end - train_end:,}")
    print(f"Final-test candles: {n - val_end:,}")
    if NUMBA:
        print(f"Numba status:       ENABLED (threads={get_num_threads()})")
    else:
        print("Numba status:       DISABLED (pure-Python fallback, slow)")
    print("=" * 70, flush=True)

    (
        ema_mat, rsi_mat, atr_mat, ema_row, rsi_row, atr_row
    ) = build_indicator_matrices(op, hi, lo, cl)

    (
        fast_row, slow_row, trend_row, warmup_arr,
        rsi_r, atr_r, slm, tpm,
    ) = configs_to_arrays(configs, ema_row, rsi_row, atr_row)

    warmup()

    # Prime the parallel dispatch / thread pool on the REAL full-size arrays
    # so the first real chunk is not the one that pays that cost.
    _t = time.time()
    run_chunk(
        op, hi, lo, cl, spread,
        ema_mat, rsi_mat, atr_mat,
        np.ascontiguousarray(fast_row[:2]),
        np.ascontiguousarray(slow_row[:2]),
        np.ascontiguousarray(trend_row[:2]),
        np.ascontiguousarray(warmup_arr[:2]),
        np.ascontiguousarray(rsi_r[:2]),
        np.ascontiguousarray(atr_r[:2]),
        np.ascontiguousarray(slm[:2]),
        np.ascontiguousarray(tpm[:2]),
        RSI_BUY_TH, RSI_SELL_TH,
        train_end, val_end,
    )
    print(f"Engine primed on full dataset. ({time.time() - _t:.1f}s)", flush=True)

    smoke = args.smoke is not None
    total = args.smoke if smoke else len(configs)
    total = min(total, len(configs))

    checkpoint_path = CHECKPOINT
    if smoke:
        checkpoint_path = None  # smoke NEVER reads or writes the real checkpoint

    # Fingerprint the CURRENT engine / grid / RSI / split / cost / risk / data
    # configuration.  This is what a resumable checkpoint must match.
    fp_payload, fp_sha = build_fingerprint(n, train_end, val_end)
    print()
    print("Engine / configuration fingerprint")
    print("-" * 70)
    print(f"  engine_version      : {ec.ENGINE_VERSION}")
    print(f"  engine_core sha256  : {fp_payload['engine']['engine_core_sha256']}")
    print(f"  RSI thresholds      : buy >= {RSI_BUY_TH:.0f}   "
          f"sell <= {RSI_SELL_TH:.0f}   (PINNED - not a grid dimension)")
    print(f"  RSI_PERIOD grid     : {RSI_PERIOD}")
    print(f"  config space        : {fp_payload['grid']['config_count']:,} "
          f"(fast<slow<trend)")
    print(f"  splits (train/val)  : {train_end:,} / {val_end:,}   total {n:,}")
    print(f"  spread floor / slip : ${ec.DEFAULT_SPREAD:.2f} / ${ec.SLIPPAGE:.2f}"
          f"   commission ${ec.COMMISSION_PER_LOT:.2f}/lot")
    print(f"  risk / margin / stop: {ec.RISK_PER_TRADE:.0%} risk   "
          f"{ec.MAX_MARGIN_FRACTION:.0%} margin cap   "
          f"${ec.EQUITY_STOP:.2f} equity stop")
    print(f"  data sha256         : {fp_payload['data'].get('sha256')}")
    print(f"  FINGERPRINT sha256  : {fp_sha}")
    print("-" * 70, flush=True)

    done = 0
    results_rows = []

    if smoke:
        print(
            "SMOKE MODE: the real checkpoint and its fingerprint are left "
            "untouched;\n"
            "           not resuming, not writing them, not running the full "
            "sweep.",
            flush=True,
        )
    elif args.fresh:
        moved = archive_stale_checkpoint()
        for src, dst in moved:
            print(f"--fresh: archived  {src}\n"
                  f"              ->   {dst}   (kept on disk, NOT deleted)",
                  flush=True)
        if not moved:
            print("--fresh: no existing checkpoint to archive.", flush=True)
        print("--fresh: starting a NEW sweep from configuration 0.", flush=True)
        write_fingerprint(fp_payload, fp_sha)
        print(f"--fresh: wrote fingerprint {FINGERPRINT}", flush=True)
    elif os.path.exists(checkpoint_path):
        ok, msg = check_checkpoint_fingerprint(fp_payload, fp_sha)
        if not ok:
            print()
            print("=" * 70)
            print("REFUSING TO RESUME FROM THE EXISTING CHECKPOINT")
            print("=" * 70)
            print(f"  checkpoint : {checkpoint_path}")
            print(f"  reason     : {msg}")
            print()
            print("  The existing checkpoint has NOT been read, modified or "
                  "deleted.")
            print("  Old and new results will never be mixed in one file.")
            print()
            print("  To start a brand-new sweep from configuration 0 (the "
                  "existing")
            print("  checkpoint is ARCHIVED with a UTC timestamp, never "
                  "deleted), run:")
            print()
            print("      python backtest/optimizer_realistic_numba.py --fresh")
            print()
            print("=" * 70, flush=True)
            sys.exit(2)
        print(f"Checkpoint fingerprint OK: {msg}", flush=True)
        done, results_rows = load_checkpoint(checkpoint_path)
        if done:
            print(
                f"Resuming from checkpoint: {done:,}/{len(configs):,} "
                f"configurations already done.",
                flush=True,
            )
        if done >= len(configs):
            print("Checkpoint already complete - proceeding to reporting.",
                  flush=True)
    else:
        # No checkpoint on disk -> this run starts a fresh sweep; stamp it.
        write_fingerprint(fp_payload, fp_sha)
        print(f"New sweep: wrote fingerprint {FINGERPRINT}", flush=True)

    if done < total:
        print(
            f"Running {total - done:,} remaining configurations "
            f"(chunk size {CHUNK}, progress every {CHUNK})...",
            flush=True,
        )

    start_time = time.time()

    idx = done
    while idx < total:
        # Align chunk boundaries to multiples of CHUNK so progress prints
        # land on 500 / 1000 / 1500 ... even after a resume.
        hi_idx = min(((idx // CHUNK) + 1) * CHUNK, total)
        sl = slice(idx, hi_idx)

        out = run_chunk(
            op, hi, lo, cl, spread,
            ema_mat, rsi_mat, atr_mat,
            np.ascontiguousarray(fast_row[sl]),
            np.ascontiguousarray(slow_row[sl]),
            np.ascontiguousarray(trend_row[sl]),
            np.ascontiguousarray(warmup_arr[sl]),
            np.ascontiguousarray(rsi_r[sl]),
            np.ascontiguousarray(atr_r[sl]),
            np.ascontiguousarray(slm[sl]),
            np.ascontiguousarray(tpm[sl]),
            RSI_BUY_TH, RSI_SELL_TH,
            train_end, val_end,
        )

        chunk_rows = []
        for j in range(hi_idx - idx):
            ef, es, et, rp, ap, slm_v, tpm_v = configs[idx + j]
            r = out[j]
            chunk_rows.append((
                ef, es, et, rp, ap, slm_v, tpm_v,
                r[0], r[1], r[2], r[3], r[4],
                r[5], r[6], r[7], r[8], r[9],
            ))

        results_rows.extend(chunk_rows)
        if checkpoint_path:
            append_checkpoint(checkpoint_path, chunk_rows)

        idx = hi_idx

        elapsed = time.time() - start_time
        processed = idx - done
        speed = processed / elapsed if elapsed > 0 else 0.0
        remaining = total - idx
        eta = remaining / speed if speed > 0 else 0.0

        print(
            f"Progress: {idx:,}/{total:,} "
            f"({idx / total * 100:.1f}%) | "
            f"elapsed {fmt_eta(elapsed)} | "
            f"{speed:.1f} cfg/sec | "
            f"ETA {fmt_eta(eta)}",
            flush=True,
        )

    total_elapsed = time.time() - start_time
    if total - done > 0:
        print(
            f"Backtesting done: {total - done:,} configs in "
            f"{fmt_eta(total_elapsed)} "
            f"({(total - done) / total_elapsed if total_elapsed else 0:.1f} "
            f"cfg/sec).",
            flush=True,
        )

    if smoke:
        print()
        print("=" * 70)
        print("SMOKE TEST COMPLETE - progress printing confirmed.")
        print("=" * 70)
        df = pd.DataFrame(results_rows[:total], columns=CP_COLUMNS)
        print(df.to_string(index=False))
        return

    report(results_rows, op, hi, lo, cl, spread,
           ema_mat, rsi_mat, atr_mat, ema_row, rsi_row, atr_row,
           train_end, val_end, n)


# --------------------------------------------------------------------------- #
# Reporting (validation selection + untouched final test)
# --------------------------------------------------------------------------- #
def report(results_rows, op, hi, lo, cl, spread,
           ema_mat, rsi_mat, atr_mat, ema_row, rsi_row, atr_row,
           train_end, val_end, n):

    raw = pd.DataFrame(results_rows, columns=CP_COLUMNS)

    valid = raw[
        (raw.train_trades >= MIN_TRAIN_TRADES)
        & (raw.train_pf >= MIN_TRAIN_PF)
        & (raw.val_trades >= MIN_VAL_TRADES)
        & (raw.val_profit > 0)
        & (raw.val_pf >= MIN_VAL_PF)
    ].copy()

    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)

    if len(valid) == 0:
        print("NO VALID STRATEGIES FOUND.")
        raw.to_csv(OUT, index=False)
        return

    valid = valid.sort_values(["val_pf", "val_profit"], ascending=False)
    valid.to_csv(OUT, index=False)

    print()
    print("=" * 70)
    print("VALIDATION CANDIDATES (selected WITHOUT final-test data)")
    print("=" * 70)
    print(valid.head(20).to_string(index=False))

    # --- Untouched final test, ONLY on the top validation candidates ---
    top = valid.head(10)
    final_rows = []
    for row in top.itertuples(index=False):
        ef = int(row.ema_fast)
        es = int(row.ema_slow)
        et = int(row.ema_trend)
        rp = int(row.rsi_period)
        ap = int(row.atr_period)

        wu = warmup_for(ef, es, et, rp, ap)
        res = run_window_fast(
            op, hi, lo, cl,
            ema_mat[ema_row[ef]],
            ema_mat[ema_row[es]],
            ema_mat[ema_row[et]],
            rsi_mat[rsi_row[rp]],
            atr_mat[atr_row[ap]],
            spread,
            val_end, n, wu,
            RSI_BUY_TH, RSI_SELL_TH, float(row.sl), float(row.tp),
        )
        # _backtest return layout: 0 trades, 2 net_pl, 3 win_rate, 4 pf, 5 dd_abs
        final_rows.append([
            ef, es, et, rp, ap, row.sl, row.tp,
            res[0], res[2], res[3], res[4], res[5],
        ])

    final_df = pd.DataFrame(
        final_rows,
        columns=[
            "ema_fast", "ema_slow", "ema_trend", "rsi_period", "atr_period",
            "sl", "tp",
            "test_trades", "test_profit", "test_winrate", "test_pf", "test_dd",
        ],
    )
    final_df.to_csv(FINAL_OUT, index=False)

    print()
    print("=" * 70)
    print("UNTOUCHED FINAL TEST")
    print("=" * 70)
    print(final_df.to_string(index=False))

    print()
    print("Saved:")
    print(" ", OUT)
    print(" ", FINAL_OUT)
    print(" ", CHECKPOINT, "(checkpoint kept; use --fresh to rerun)")
    print(" ", FINGERPRINT, "(binds the checkpoint to this engine/grid/data)")


if __name__ == "__main__":
    main()
