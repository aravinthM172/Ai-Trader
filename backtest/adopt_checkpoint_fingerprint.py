"""
SAFE checkpoint-adoption operation.

Generates ONLY the fingerprint sidecar

    reports/xauusd_realistic_optimization.checkpoint.fingerprint.json

for the existing, already-complete checkpoint

    reports/xauusd_realistic_optimization.checkpoint.csv

so that a bare `python backtest/optimizer_realistic_numba.py` run will accept the
checkpoint as resumable / complete.

This script:
  * does NOT run the optimizer, does NOT run --fresh
  * does NOT modify, delete, rename or archive the checkpoint CSV
  * writes exactly one new file (the sidecar) and nothing else
  * refuses to overwrite an existing sidecar

Guards (any failure => print the mismatch, write nothing, exit non-zero):
  1. checkpoint exists and has EXACTLY 17,280 data rows
  2. checkpoint column schema == engine CP_COLUMNS
  3. checkpoint config keys (ema_fast..tp) match build_configs() EXACTLY,
     in the same positional order (the checkpoint is position-keyed)
  4. build_configs() / fingerprint grid still describe 17,280 configs
  5. checkpoint SHA-256 is identical before and after

The fingerprint is built from the CURRENT optimizer configuration via the
optimizer's own build_fingerprint() / write_fingerprint(), i.e. the same code
path a real run uses:
    engine_core.py source hash + ENGINE_VERSION, the full parameter grid and
    config_count, pinned RSI_BUY_TH/RSI_SELL_TH, TRAIN_END/VAL_END/TOTAL/n_bars,
    execution costs, risk/lot/margin constants, the checkpoint column schema,
    and the data file SHA-256.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import engine_core as ec  # noqa: F401  (kept for parity with optimizer imports)
import optimizer_realistic_numba as opt

EXPECT_ROWS = 17_280
KEY_COLS = [
    "ema_fast", "ema_slow", "ema_trend", "rsi_period", "atr_period", "sl", "tp",
]


def _sha256_file(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_bufsize), b""):
            h.update(block)
    return h.hexdigest()


def _stop(msg):
    print()
    print("=" * 70)
    print("STOP - nothing written.")
    print("=" * 70)
    print(msg)
    sys.exit(1)


def main():
    cp = opt.CHECKPOINT
    fp = opt.FINGERPRINT

    print("=" * 70)
    print("SAFE CHECKPOINT ADOPTION - fingerprint sidecar only")
    print("=" * 70)
    print(f"checkpoint : {cp}")
    print(f"sidecar    : {fp}")
    print()

    if not os.path.exists(cp):
        _stop(f"checkpoint CSV not found: {cp}")

    if os.path.exists(fp):
        _stop(
            f"a fingerprint sidecar already exists: {fp}\n"
            "Refusing to overwrite it. If you really mean to regenerate it, move\n"
            "it aside yourself first."
        )

    sha_before = _sha256_file(cp)
    print(f"checkpoint sha256 (before) : {sha_before}")

    # --- 1/2. load + schema ------------------------------------------------- #
    df = pd.read_csv(cp)

    if list(df.columns) != list(opt.CP_COLUMNS):
        _stop(
            "checkpoint column schema does not match the engine:\n"
            f"  checkpoint : {list(df.columns)}\n"
            f"  engine     : {list(opt.CP_COLUMNS)}"
        )

    if len(df) != EXPECT_ROWS:
        _stop(
            f"checkpoint row count is {len(df):,}, expected exactly "
            f"{EXPECT_ROWS:,}.\n"
            "The checkpoint is not a complete 17,280-config sweep; not adopting."
        )

    # --- 3/4. config-key match against build_configs() -------------------- #
    configs = opt.build_configs()

    if len(configs) != EXPECT_ROWS:
        _stop(
            f"build_configs() now returns {len(configs):,} configs, expected "
            f"{EXPECT_ROWS:,}.\nThe current parameter grid differs from the one "
            "that produced this checkpoint."
        )

    cp_keys = [
        (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]),
         float(r[5]), float(r[6]))
        for r in df[KEY_COLS].to_numpy()
    ]
    want_keys = [
        (int(c[0]), int(c[1]), int(c[2]), int(c[3]), int(c[4]),
         float(c[5]), float(c[6]))
        for c in configs
    ]

    positional = [
        (i, cp_keys[i], want_keys[i])
        for i in range(EXPECT_ROWS)
        if cp_keys[i] != want_keys[i]
    ]
    if positional:
        lines = [
            f"{len(positional):,} checkpoint rows are not in build_configs() "
            "order (checkpoint is position-keyed):"
        ]
        for i, got, want in positional[:20]:
            lines.append(f"  row {i}: checkpoint={got}   build_configs={want}")
        if len(positional) > 20:
            lines.append(f"  ... and {len(positional) - 20:,} more")
        _stop("\n".join(lines))

    if set(cp_keys) != set(want_keys):
        only_cp = sorted(set(cp_keys) - set(want_keys))[:20]
        only_bc = sorted(set(want_keys) - set(cp_keys))[:20]
        _stop(
            "checkpoint config-key SET differs from build_configs():\n"
            f"  only in checkpoint    : {only_cp}\n"
            f"  only in build_configs : {only_bc}"
        )

    print(f"rows                       : {len(df):,}  (OK, == {EXPECT_ROWS:,})")
    print("schema                     : matches engine CP_COLUMNS  (OK)")
    print("config keys                : exact positional match to "
          "build_configs()  (OK)")

    # --- build the fingerprint from the CURRENT configuration ------------- #
    op_, hi_, lo_, cl_, spread_ = opt.load_data()
    n = len(cl_)
    val_end = min(opt.VAL_END, n)
    train_end = min(opt.TRAIN_END, val_end)

    payload, payload_sha = opt.build_fingerprint(n, train_end, val_end)

    if payload["grid"]["config_count"] != EXPECT_ROWS:
        _stop(
            "fingerprint grid.config_count is "
            f"{payload['grid']['config_count']:,}, expected {EXPECT_ROWS:,}."
        )
    if payload["checkpoint_schema"] != list(df.columns):
        _stop("fingerprint checkpoint_schema disagrees with the checkpoint file.")

    opt.write_fingerprint(payload, payload_sha)

    print()
    print(f"wrote sidecar              : {fp}")
    print(f"fingerprint sha256         : {payload_sha}")
    print(f"engine_version             : {payload['engine']['engine_version']}")
    print(f"engine_core sha256         : "
          f"{payload['engine']['engine_core_sha256']}")
    print(f"data sha256                : {payload['data'].get('sha256')}")
    print(f"splits train/val/total/n   : {train_end:,} / {val_end:,} / "
          f"{opt.TOTAL:,} / {n:,}")
    print(f"RSI thresholds             : buy>={payload['rsi_thresholds']['RSI_BUY_TH']:.0f}"
          f"  sell<={payload['rsi_thresholds']['RSI_SELL_TH']:.0f}  (pinned)")

    # --- 5. checkpoint must be byte-for-byte unchanged ------------------- #
    sha_after = _sha256_file(cp)
    print()
    print(f"checkpoint sha256 (after)  : {sha_after}")
    if sha_after != sha_before:
        print()
        print("FATAL: checkpoint SHA-256 CHANGED during this operation.")
        sys.exit(2)
    print("checkpoint preserved byte-for-byte  (OK)")

    # --- end-to-end: engine's own resume check must now pass ------------- #
    ok, msg = opt.check_checkpoint_fingerprint(payload, payload_sha)
    print()
    print(f"engine resume check        : {'OK' if ok else 'FAIL'} - {msg}")
    if not ok:
        sys.exit(3)

    print()
    print("=" * 70)
    print("DONE - checkpoint adopted. A bare optimizer run will now resume/accept")
    print("it (17,280/17,280 complete -> proceeds straight to reporting).")
    print("=" * 70)


if __name__ == "__main__":
    main()
