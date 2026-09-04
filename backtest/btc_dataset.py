"""
PHASE 2 -- reproducible BTCUSD.vx (Valetax) research dataset.

Downloads M5 / M15 / H1 history from the live terminal, validates it, and writes:
    data/btcusd_vx_M5.csv  data/btcusd_vx_M15.csv  data/btcusd_vx_H1.csv
    reports/btc_dataset.json

Historical spread is NOT fabricated: it is read from MT5's own ``spread`` column
(points -> $).  Valetax reports a fixed $29.76 spread; that fact is recorded, and
cost sensitivity (btc_cost_sensitivity.json) models scenarios around it.

    python -m backtest.btc_dataset              # download + validate
    python -m backtest.btc_dataset --validate   # re-validate cached CSVs only
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

log = get_logger("btc.dataset")
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TIMEFRAMES = ("M5", "M15", "H1")
TARGET_BARS = {"M5": 200_000, "M15": 200_000, "H1": 200_000}   # ask for a lot; take what we get


def _csv(tf: str) -> Path:
    return DATA / f"btcusd_vx_{tf}.csv"


def download() -> dict:
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        raise SystemExit("MT5 connection failed")
    meta = {"server_utc_offset_hours": gw.server_utc_offset_seconds // 3600,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(), "timeframes": {}}
    try:
        spec = gw.get_spec("BTCUSD.vx")
        meta["symbol_spec"] = {
            "point": spec.point, "tick_size": spec.tick_size, "tick_value": spec.tick_value,
            "contract_size": spec.contract_size, "value_per_unit_per_lot": spec.value_per_price_unit_per_lot,
            "volume_min": spec.volume_min, "volume_step": spec.volume_step, "volume_max": spec.volume_max,
            "stops_level_points": spec.stops_level_points, "stops_level_price": spec.stops_level_price,
            "spread_points_now": spec.spread_points, "spread_float": spec.spread_float,
            "trade_exemode": spec.trade_exemode, "filling_mode": spec.filling_mode,
        }
        for tf in TIMEFRAMES:
            df = gw.get_rates("BTCUSD.vx", tf, TARGET_BARS[tf])
            if df is None or df.empty:
                meta["timeframes"][tf] = {"error": "no data"}
                continue
            df.to_csv(_csv(tf), index=False)
            log.info("[%s] %d bars %s -> %s", tf, len(df), df["time"].iloc[0], df["time"].iloc[-1])
            meta["timeframes"][tf] = {"csv": _csv(tf).name, "bars": int(len(df))}
        return meta
    finally:
        gw.shutdown()


def _step_seconds(tf: str) -> int:
    return {"M5": 300, "M15": 900, "H1": 3600}[tf]


def validate_tf(tf: str, cfg) -> dict:
    p = _csv(tf)
    if not p.exists():
        return {"error": f"{p.name} not found -- run without --validate first"}
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    cfg2 = btc_config()
    cfg2.timeframe = tf
    cfg2.min_history_bars = 500
    cfg2.stale_seconds = _step_seconds(tf) * 6
    clean, rep = validate_ohlc(df, symbol="BTCUSD.vx", timeframe=tf, cfg=cfg2)

    sp_col = df["spread"] if "spread" in df.columns else None
    spread_info = None
    if sp_col is not None:
        sp_dollars = sp_col.to_numpy(float) * cfg.point
        nz = sp_dollars[sp_dollars > 0]
        spread_info = {
            "available": True, "source": "MT5 spread column (points)",
            "fabricated": False,
            "constant": bool(len(np.unique(np.round(nz, 2))) == 1),
            "dollars_min": round(float(sp_dollars.min()), 2),
            "dollars_median": round(float(np.median(sp_dollars)), 2),
            "dollars_p95": round(float(np.percentile(sp_dollars, 95)), 2),
            "dollars_max": round(float(sp_dollars.max()), 2),
            "zero_spread_bars": int((sp_dollars <= 0).sum()),
        }
    else:
        spread_info = {"available": False, "note": "no historical spread; use cost scenarios"}

    vol = df["tick_volume"] if "tick_volume" in df.columns else None
    rvol = df["real_volume"] if "real_volume" in df.columns else None

    return {
        "csv": p.name,
        "candles": int(len(df)),
        "start": str(df["time"].iloc[0]),
        "end": str(df["time"].iloc[-1]),
        "span_days": int((df["time"].iloc[-1] - df["time"].iloc[0]).days),
        "expected_step_seconds": _step_seconds(tf),
        "missing_candles": rep.missing_candles,
        "gap_events": len(rep.gap_events),
        "largest_gaps": rep.gap_events[:5],
        "duplicates": rep.duplicate_timestamps,
        "ohlc_anomalies": rep.issues,
        "rejected_rows": rep.n_rejected,
        "tick_volume_zero_bars": int((vol <= 0).sum()) if vol is not None else None,
        "tick_volume_median": float(vol.median()) if vol is not None else None,
        "real_volume_all_zero": bool((rvol == 0).all()) if rvol is not None else None,
        "spread": spread_info,
        "timezone": "MT5 timestamps are broker-server local; converted to true UTC on load "
                    f"(offset stored in reports/btc_dataset.json)",
        "quality_ok_for_research": bool(rep.n_clean >= 500 and rep.n_rejected / max(1, len(df)) < 0.02),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="re-validate cached CSVs, no download")
    args = ap.parse_args()
    cfg = btc_config()

    meta = {"validated_utc": datetime.now(timezone.utc).isoformat()}
    if not args.validate:
        meta.update(download())

    meta["validation"] = {tf: validate_tf(tf, cfg) for tf in TIMEFRAMES}
    (REPORTS / "btc_dataset.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("=" * 92)
    print("BTCUSD.vx (Valetax) RESEARCH DATASET  --  NO LIVE ORDER")
    print("=" * 92)
    if "symbol_spec" in meta:
        s = meta["symbol_spec"]
        print(f"contract: $ {s['value_per_unit_per_lot']}/$1move/lot  vol {s['volume_min']}/{s['volume_step']}  "
              f"broker min stop ${s['stops_level_price']:.2f}  exemode {s['trade_exemode']}  fill {s['filling_mode']}")
        print(f"server UTC offset: {meta.get('server_utc_offset_hours')} h")
    print("-" * 92)
    for tf, v in meta["validation"].items():
        if "error" in v:
            print(f"{tf:4s} : {v['error']}")
            continue
        sp = v["spread"]
        sp_s = (f"${sp['dollars_median']} const" if sp.get("constant")
                else f"${sp.get('dollars_min')}-{sp.get('dollars_max')}") if sp["available"] else "n/a"
        print(f"{tf:4s} : {v['candles']:>7,} candles  {v['start'][:10]} -> {v['end'][:10]} "
              f"({v['span_days']}d)  missing {v['missing_candles']}  dup {v['duplicates']}  "
              f"anomalies {sum(v['ohlc_anomalies'].values()) if v['ohlc_anomalies'] else 0}  "
              f"spread {sp_s}  ok={v['quality_ok_for_research']}")
    print("-" * 92)
    print("historical spread is broker-reported (NOT fabricated); it is fixed at $29.76 on this feed.")
    print("cost sensitivity (Phase 4) models optimistic/normal/conservative/stressed around it.")
    print("report: reports/btc_dataset.json")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
