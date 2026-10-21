"""
Download free Dukascopy 1-minute BID candles for one instrument and date range (added 2026-10-21 for the owner's
"enter while the candle forms" gold test).

    python -m tools.fetch_dukascopy_m1 XAUUSD 2026-03-01 2026-08-31

One LZMA file per instrument-day (datafeed.dukascopy.com/datafeed/<INST>/<YYYY>/<MM0>/<DD>/BID_candles_min_1.bi5),
cached raw in data/dukascopy_raw/<INST>_M1/, merged into data/dukascopy_M1/<INST>.csv (time,open,high,low,close,volume).
Gold prices are stored x1000 by Dukascopy (same scale as the hourly files).  Zero-volume minutes are dropped.
"""
from __future__ import annotations

import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
SCALE = {"XAUUSD": 1000.0}
PAUSE_S = 1.5


def get_day(inst: str, day: pd.Timestamp) -> bytes | None:
    p = ROOT / "data" / "dukascopy_raw" / f"{inst}_M1" / f"{day:%Y-%m-%d}.bi5"
    if p.exists():
        return p.read_bytes()
    url = URL.format(inst=inst, y=day.year, m=day.month - 1, d=day.day)
    for attempt in range(5):
        time.sleep(PAUSE_S)
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            return raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")
                return b""
            time.sleep(60 if e.code in (429, 503) else 5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def decode(raw: bytes, day: pd.Timestamp, scale: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    a = np.array(list(struct.iter_unpack(">5if", lzma.decompress(raw))))
    d = pd.DataFrame(dict(time=day + pd.to_timedelta(a[:, 0], unit="s"), open=a[:, 1] / scale, close=a[:, 2] / scale,
                          low=a[:, 3] / scale, high=a[:, 4] / scale, volume=a[:, 5]))
    return d[d.volume > 0]


def main() -> int:
    inst, a, b = sys.argv[1], pd.Timestamp(sys.argv[2], tz="UTC"), pd.Timestamp(sys.argv[3], tz="UTC")
    parts, missing = [], 0
    days = [d for d in pd.date_range(a, b, freq="D") if d.weekday() != 5]          # no Saturday sessions
    for k, day in enumerate(days):
        raw = get_day(inst, day)
        if raw is None:
            missing += 1
            continue
        parts.append(decode(raw, day, SCALE.get(inst, 1000.0)))
        if k % 20 == 0:
            print(f"{day.date()} ({k + 1}/{len(days)})", flush=True)
    d = pd.concat([x for x in parts if len(x)], ignore_index=True).sort_values("time").drop_duplicates("time")
    out = ROOT / "data" / "dukascopy_M1" / f"{inst}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"saved {out} : {len(d)} minutes, {d.time.iloc[0]} -> {d.time.iloc[-1]}, failed days {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
