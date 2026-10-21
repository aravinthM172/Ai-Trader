"""
Download free Dukascopy tick data (bid + ask) for one instrument and date range (added 2026-10-21 for the owner's
"flip while the candle forms" gold test).

    python -m tools.fetch_dukascopy_ticks XAUUSD 2026-08-01 2026-08-31

One LZMA file per instrument-hour (datafeed.dukascopy.com/datafeed/<INST>/<YYYY>/<MM0>/<DD>/<HH>h_ticks.bi5), cached raw
in data/dukascopy_raw/<INST>_ticks/, merged into data/dukascopy_ticks/<INST>_<from>_<to>.parquet|csv
(time, bid, ask).  Gold prices are stored x1000.  Record = ms in hour, ask, bid, ask volume, bid volume.
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
URL = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
SCALE = {"XAUUSD": 1000.0}
PAUSE_S = 2.0


def get_hour(inst: str, t: pd.Timestamp) -> bytes | None:
    p = ROOT / "data" / "dukascopy_raw" / f"{inst}_ticks" / f"{t:%Y-%m-%d_%H}.bi5"
    if p.exists():
        return p.read_bytes()
    url = URL.format(inst=inst, y=t.year, m=t.month - 1, d=t.day, h=t.hour)
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


def decode(raw: bytes, t: pd.Timestamp, scale: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    a = np.array(list(struct.iter_unpack(">3i2f", lzma.decompress(raw))))
    return pd.DataFrame(dict(time=t + pd.to_timedelta(a[:, 0], unit="ms"), ask=a[:, 1] / scale, bid=a[:, 2] / scale))


def main() -> int:
    inst, a, b = sys.argv[1], pd.Timestamp(sys.argv[2], tz="UTC"), pd.Timestamp(sys.argv[3], tz="UTC") + pd.Timedelta(hours=23)
    hours = [h for h in pd.date_range(a, b, freq="h") if h.weekday() != 5]
    parts, failed = [], 0
    for k, h in enumerate(hours):
        raw = get_hour(inst, h)
        if raw is None:
            failed += 1
            continue
        parts.append(decode(raw, h, SCALE.get(inst, 1000.0)))
        if k % 100 == 0:
            print(f"{h} ({k + 1}/{len(hours)})", flush=True)
    d = pd.concat([x for x in parts if len(x)], ignore_index=True).sort_values("time")
    out = ROOT / "data" / "dukascopy_ticks" / f"{inst}_{a:%Y%m%d}_{b:%Y%m%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"saved {out}: {len(d)} ticks, {d.time.iloc[0]} -> {d.time.iloc[-1]}, failed hours {failed}, "
          f"median spread {float((d.ask - d.bid).median()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
