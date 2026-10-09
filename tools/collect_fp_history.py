"""
Download the FundingPips (MetaQuotes) feed for every tradable CFD -- H1 and D1, max history -- plus each
symbol's spread and swap, into data/fp/.  This is the verified-clean source (2026-10-07: matches
TradingView OANDA and Dukascopy; the old Valetax H1 files had daily bars mixed in and were sparse
before 2024).  Dated futures (e.g. GCM26) are skipped: they expire and have short histories.

    python -m tools.collect_fp_history

Quality report per file (data/fp/QUALITY.json): first dense year, bars per year, the largest gap,
and the count of 'daily bars mixed into H1' (bars spanning > 0.8 of their broker day's range on
the same clock hour every day -- the Valetax defect).
Swap: annual fraction of price, computed from the raw broker value -- points mode as
points x point x 365 / price (NOT through tick value, which FundingPips reports 100x off for gold).
Read-only on MT5; no orders.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.logging_setup import get_logger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fp"
log = get_logger("fp.history", filename="fp_history.log")
MIN_FREE_RAM_GB = 1.5
# Symbols the live trader / paper trader / dashboard use: their MT5 history cache is never cleared.
KEEP_CACHE = {"BTCUSD", "XAUUSD", "GER40", "NDX100", "SPX500", "DJI30", "JP225", "FTSE100"}
PAUSE_S = 20                                                 # between symbols (shared terminal)
DATED = re.compile(r"^[A-Z0-9]{2,4}[FGHJKMNQUVXZ]\d{2}$")      # futures month codes, e.g. GCM26, FT1M26


def _free_ram_gb() -> float:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return 99.0


def clear_cache(m, sym: str, hist_dir: Path) -> None:
    """After the CSV is safe on D:, drop the symbol from Market Watch and delete its MT5 history cache on C:
    (the terminal re-downloads it if anything ever asks).  Tested 2026-10-07 with the live trader running."""
    if sym in KEEP_CACHE or not (OUT / f"{sym}_H1.csv").exists():
        return
    import shutil
    m.symbol_select(sym, False)
    shutil.rmtree(hist_dir / sym, ignore_errors=True)


def swap_annual(info, side: float, price: float) -> float | None:
    mode, px = int(info.swap_mode), float(price or 0)
    if px <= 0:
        return None
    if mode == 1:                                               # points per night
        return side * float(info.point) * 365 / px
    if mode in (5, 6):                                          # annual interest %
        return side / 100.0
    if mode in (2, 3, 4):                                       # money per lot per night (approx. USD)
        return side / float(info.trade_contract_size or 1) * 365 / px
    return None


def quality(d: pd.DataFrame) -> dict:
    t = d["time"]
    per = t.dt.year.value_counts().sort_index()
    ref = per.iloc[-2] if len(per) > 1 else per.iloc[-1]
    dense = [int(y) for y, n in per.items() if n >= 0.7 * ref]
    day = (t + pd.Timedelta(hours=3)).dt.normalize()
    drng = (d.high.groupby(day).transform("max") - d.low.groupby(day).transform("min"))
    tv = d.tick_volume
    # calibrated 2026-10-07: clean FP gold 22, corrupt Valetax gold 1307 -- range AND volume of a whole day
    big = ((d.high - d.low) > 0.8 * drng) & (tv > 5 * tv.rolling(500, min_periods=50).median())
    hr = t[big].dt.hour.value_counts()
    return dict(bars=len(d), start=str(t.min()), end=str(t.max()), first_dense_year=dense[0] if dense else None,
                bars_per_year=int(ref), largest_gap_days=round(t.diff().max().total_seconds() / 86400, 1),
                daily_bar_suspects=int(hr.iloc[0]) if len(hr) else 0,
                daily_bar_suspect_hour=int(hr.index[0]) if len(hr) else None)


def main() -> int:
    import sys
    from_disk = "--from-disk" in sys.argv          # no history requests: specs + quality for files already on disk
    from mt5.gateway import MT5Gateway
    OUT.mkdir(parents=True, exist_ok=True)
    gw = MT5Gateway()
    if not gw.connect():
        print("MT5 connection failed")
        return 1
    m = gw.raw()
    hist_dir = Path(m.terminal_info().data_path) / "bases" / m.account_info().server / "history"
    for f in OUT.glob("*_H1.csv"):                      # symbols saved by an earlier run
        clear_cache(m, f.name[:-7], hist_dir)
    specs, qual = {}, {}
    try:
        syms = [s for s in m.symbols_get() if s.trade_mode != 0 and not DATED.match(s.name)]
        for s in sorted(syms, key=lambda x: x.name):
            m.symbol_select(s.name, True)
            info = m.symbol_info(s.name)
            h1 = None
            for tf, n in (("H1", 200_000), ("D1", 20_000)):
                p = OUT / f"{s.name}_{tf}.csv"
                if from_disk and not p.exists():
                    continue
                if p.exists() and (from_disk or (time.time() - p.stat().st_mtime) < 86400):
                    d = pd.read_csv(p)
                    d["time"] = pd.to_datetime(d["time"], utc=True)
                else:
                    d = gw.get_rates(s.name, tf, n)
                    if d is None or d.empty:
                        log.warning("%s %s: no data", s.name, tf)
                        continue
                    d.to_csv(p, index=False)
                if tf == "H1":
                    h1 = d
                    qual[s.name] = quality(d)
            clear_cache(m, s.name, hist_dir)
            if not from_disk:
                time.sleep(PAUSE_S)                    # give the live trader turns on the shared MT5 terminal
            while _free_ram_gb() < MIN_FREE_RAM_GB:      # never push the PC (and the live trader) out of memory
                log.warning("free RAM %.1f GB < %.1f GB -- waiting", _free_ram_gb(), MIN_FREE_RAM_GB)
                time.sleep(60)
            if h1 is None:
                continue
            px = float(h1.close.iloc[-1])
            last_year = h1[h1.time >= h1.time.max() - pd.Timedelta(days=365)]
            spread_pts = float(last_year.spread[last_year.spread > 0].median()) if (last_year.spread > 0).any() else float(info.spread)
            specs[s.name] = dict(path=info.path, point=info.point, digits=info.digits, price=px,
                                 spread_points_median_1y=spread_pts, spread_price=spread_pts * info.point,
                                 contract_size=info.trade_contract_size, swap_mode=info.swap_mode,
                                 swap_long_raw=info.swap_long, swap_short_raw=info.swap_short,
                                 swap_long=swap_annual(info, info.swap_long, px),
                                 swap_short=swap_annual(info, info.swap_short, px),
                                 currency_profit=info.currency_profit)
            q = qual.get(s.name, {})
            log.info("%-8s H1 %s..%s dense from %s, gap %s d, daily-bar suspects %s | spread %s | swap L %s S %s",
                     s.name, q.get("start", "")[:10], q.get("end", "")[:10], q.get("first_dense_year"),
                     q.get("largest_gap_days"), q.get("daily_bar_suspects"), specs[s.name]["spread_price"],
                     specs[s.name]["swap_long"], specs[s.name]["swap_short"])
    finally:
        gw.shutdown()
    (OUT / "SPECS.json").write_text(json.dumps(specs, indent=2))
    (OUT / "QUALITY.json").write_text(json.dumps(qual, indent=2))
    for k, q in qual.items():
        print(f"{k:8} {q['start'][:10]}..{q['end'][:10]} dense from {q['first_dense_year']}  "
              f"{q['bars_per_year']:5}/yr  gap {q['largest_gap_days']:5} d  daily-bar suspects {q['daily_bar_suspects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
