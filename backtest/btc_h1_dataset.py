"""
STEP 1 -- BTCUSD.vx (Valetax) H1 research dataset.

Downloads the maximum H1 history the terminal provides, identifies the
CONTIGUOUS dense window (Valetax only keeps real hourly bars from ~2024;
2019-2023 are daily snapshots), validates it, and writes:

    data/btcusd_vx_H1.csv          (full pull)
    data/btcusd_vx_H1_dense.csv    (contiguous 1h window used for research)
    reports/btc_h1_dataset.json

Historical spread is read from MT5's own ``spread`` column -- never fabricated.

    python -m backtest.btc_h1_dataset            # download + validate
    python -m backtest.btc_h1_dataset --validate # re-validate cached CSV only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from config.assets import btc_config
from data.quality import validate_ohlc

log = get_logger("btc.h1data")
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

FULL = DATA / "btcusd_vx_H1.csv"
DENSE = DATA / "btcusd_vx_H1_dense.csv"
STEP = 3600
POINT = 0.01


def download() -> dict:
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        raise SystemExit("MT5 connection failed")
    try:
        spec = gw.get_spec("BTCUSD.vx")
        df = gw.get_rates("BTCUSD.vx", "H1", 200_000)
        if df is None or df.empty:
            raise SystemExit("no H1 data")
        df.to_csv(FULL, index=False)
        log.info("downloaded %d H1 bars %s -> %s", len(df), df.time.iloc[0], df.time.iloc[-1])
        return {
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "server_utc_offset_hours": gw.server_utc_offset_seconds // 3600,
            "symbol_spec": {
                "point": spec.point, "tick_size": spec.tick_size, "tick_value": spec.tick_value,
                "contract_size": spec.contract_size,
                "value_per_unit_per_lot": spec.value_per_price_unit_per_lot,
                "volume_min": spec.volume_min, "volume_step": spec.volume_step,
                "volume_max": spec.volume_max, "stops_level_points": spec.stops_level_points,
                "stops_level_price": spec.stops_level_price, "spread_points_now": spec.spread_points,
                "trade_exemode": spec.trade_exemode, "filling_mode": spec.filling_mode,
                "trade_mode": spec.trade_mode,
            },
            "full_bars": int(len(df)),
        }
    finally:
        gw.shutdown()


def dense_window(df: pd.DataFrame, *, min_run=500, max_gap_h=72.0) -> tuple[pd.DataFrame, dict]:
    """
    Keep the real hourly history: start at the first sustained hourly run
    (drops the 2019-2023 daily snapshots), keep everything after it, tolerating
    isolated gaps up to ``max_gap_h`` (weekend/maintenance) which are recorded.
    """
    t = df["time"].to_numpy()
    dt_h = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
    ok = (dt_h >= 0.9) & (dt_h <= 1.1)
    # first index where an hourly run of >= min_run bars begins
    run = start = 0
    first = None
    for i, v in enumerate(ok):
        if v:
            if run == 0:
                start = i
            run += 1
            if run >= min_run:
                first = start
                break
        else:
            run = 0
    if first is None:
        first = 0
    dense = df.iloc[first:].reset_index(drop=True)
    # record (but keep) internal gaps
    d2 = dense["time"].diff().dt.total_seconds().dropna() / 3600.0
    gaps = []
    for j in np.where((d2 > 1.5).to_numpy())[0]:
        idx = d2.index[j]
        gaps.append({"after": str(dense["time"].iloc[idx - 1]),
                     "before": str(dense["time"].iloc[idx]),
                     "gap_hours": round(float(d2.iloc[j]), 1)})
    big_gaps = [g for g in gaps if g["gap_hours"] > max_gap_h]
    info = {
        "dense_start": str(dense["time"].iloc[0]),
        "dense_end": str(dense["time"].iloc[-1]),
        "dense_bars": int(len(dense)),
        "dense_span_days": int((dense["time"].iloc[-1] - dense["time"].iloc[0]).days),
        "full_bars": int(len(df)),
        "dropped_leading_sparse_bars": int(first),
        "internal_gap_count": len(gaps),
        "internal_gaps_over_72h": len(big_gaps),
        "largest_gaps": sorted(gaps, key=lambda g: -g["gap_hours"])[:6],
        "note": "2019-2023 in the full pull are ~daily snapshots (dropped). The dense "
                "hourly window keeps isolated weekend/maintenance gaps (recorded above); "
                "no bar is fabricated to fill them.",
    }
    return dense, info


def validate(dense: pd.DataFrame, cfg) -> dict:
    c = btc_config()
    c.timeframe = "H1"
    c.min_history_bars = 1000
    c.stale_seconds = STEP * 8
    c.max_candle_gap_multiplier = 3.0
    clean, rep = validate_ohlc(dense, symbol="BTCUSD.vx", timeframe="H1", cfg=c)

    sp = dense["spread"].to_numpy(float) * POINT if "spread" in dense.columns else None
    spread_info = None
    if sp is not None:
        nz = sp[sp > 0]
        spread_info = {
            "available": True, "fabricated": False, "source": "MT5 spread column (points)",
            "dollars_min": round(float(nz.min()), 2) if len(nz) else None,
            "dollars_median": round(float(np.median(sp)), 2),
            "dollars_p95": round(float(np.percentile(sp, 95)), 2),
            "dollars_max": round(float(sp.max()), 2),
            "zero_spread_bars": int((sp <= 0).sum()),
            "mostly_constant_29_76": bool(float(np.median(nz)) == 29.76 if len(nz) else False),
        }
    vol = dense["tick_volume"] if "tick_volume" in dense.columns else None
    rvol = dense["real_volume"] if "real_volume" in dense.columns else None
    return {
        "csv": DENSE.name,
        "candles": int(len(dense)),
        "start": str(dense["time"].iloc[0]),
        "end": str(dense["time"].iloc[-1]),
        "span_days": int((dense["time"].iloc[-1] - dense["time"].iloc[0]).days),
        "span_years": round((dense["time"].iloc[-1] - dense["time"].iloc[0]).days / 365.25, 2),
        "expected_step_seconds": STEP,
        "missing_candles": rep.missing_candles,
        "gap_events": len(rep.gap_events),
        "largest_gaps": rep.gap_events[:5],
        "duplicates": rep.duplicate_timestamps,
        "ohlc_anomalies": rep.issues,
        "rejected_rows": rep.n_rejected,
        "zero_or_negative_price_bars": int(rep.issues.get("non_positive_price", 0)),
        "tick_volume_zero_bars": int((vol <= 0).sum()) if vol is not None else None,
        "real_volume_all_zero": bool((rvol == 0).all()) if rvol is not None else None,
        "spread": spread_info,
        "timezone": "MT5 timestamps are broker-server local (+3h); converted to true UTC on load",
        "quality_ok_for_research": bool(rep.n_clean >= 1000 and rep.n_rejected / max(1, len(dense)) < 0.02),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    cfg = btc_config()

    meta = {"generated_utc": datetime.now(timezone.utc).isoformat()}
    if not args.validate:
        meta.update(download())
    if not FULL.exists():
        raise SystemExit("run without --validate first")

    df = pd.read_csv(FULL)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    dense, dinfo = dense_window(df)
    dense.to_csv(DENSE, index=False)
    meta["dense_window"] = dinfo
    meta["validation"] = validate(dense, cfg)
    (REPORTS / "btc_h1_dataset.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    v = meta["validation"]
    sp = v["spread"] or {}
    print("=" * 94)
    print("BTCUSD.vx (Valetax) H1 DATASET  --  NO LIVE ORDER")
    print("=" * 94)
    if "symbol_spec" in meta:
        s = meta["symbol_spec"]
        print(f"contract: ${s['value_per_unit_per_lot']}/$1move/lot  vol {s['volume_min']}/{s['volume_step']}  "
              f"min stop ${s['stops_level_price']:.2f}  exemode {s['trade_exemode']}  fill {s['filling_mode']}  "
              f"server +{meta.get('server_utc_offset_hours')}h")
    print(f"full pull       : {df.shape[0]:,} bars {df.time.iloc[0]} -> {df.time.iloc[-1]}")
    print(f"                  (2019-2023 are daily snapshots, not hourly -- dropped)")
    print(f"dense H1 window : {v['candles']:,} bars  {v['start'][:16]} -> {v['end'][:16]}  "
          f"({v['span_days']}d / {v['span_years']}y)")
    print(f"quality         : missing {v['missing_candles']}  dup {v['duplicates']}  "
          f"anomalies {sum(v['ohlc_anomalies'].values()) if v['ohlc_anomalies'] else 0}  "
          f"rejected {v['rejected_rows']}  ok={v['quality_ok_for_research']}")
    print(f"spread (broker) : median ${sp.get('dollars_median')}  p95 ${sp.get('dollars_p95')}  "
          f"max ${sp.get('dollars_max')}  zero-bars {sp.get('zero_spread_bars')}  (NOT fabricated)")
    print(f"volume          : real_volume all-zero {v['real_volume_all_zero']} (crypto CFD); "
          f"tick_volume zero-bars {v['tick_volume_zero_bars']}")
    print("report: reports/btc_h1_dataset.json   dataset: data/btcusd_vx_H1_dense.csv")
    print("=" * 94)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
