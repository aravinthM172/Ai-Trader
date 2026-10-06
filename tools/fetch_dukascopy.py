"""
Download free Dukascopy H1 history (BID, UTC) for every Valetax symbol it carries, 2005 -> today.

    python -m tools.fetch_dukascopy                 # all symbols, resumes where it left off
    python -m tools.fetch_dukascopy EURUSD.vx XAUUSD.vx

One LZMA file per instrument-month (datafeed.dukascopy.com/datafeed/<INST>/<YYYY>/<MM0>/BID_candles_hour_1.bi5),
cached raw in data/dukascopy_raw/, merged into data/dukascopy_H1/<symbol>.csv with columns time,open,high,low,close,volume.
Price scale is found by matching the Yahoo daily close (data/yahoo_D1).  Zero-volume (closed-market) hours are dropped.
"""
from __future__ import annotations

import lzma
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger

log = get_logger("dukascopy")
RAW = ROOT / "data" / "dukascopy_raw"
OUT = ROOT / "data" / "dukascopy_H1"
URL = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/BID_candles_hour_1.bi5"
START_YEAR = 2005
PAUSE_S = 3.0
BLOCKED_SLEEP_S = 300                                  # on HTTP 503 (rate limited) wait 5 minutes
# most useful first: what the live bot trades, then indices/commodities/crypto, then FX
PRIORITY = ["DAX40.vx", "XAUUSD.vx", "BTCUSD.vx", "XAGUSD.vx", "NAS100.vx", "SP500.vx", "US30.vx", "JPN225.vx",
            "UK100.vx", "FRA40.vx", "EU50.vx", "HK50.vx", "AUS200.vx", "ETHUSD.vx", "XTIUSD.vx", "XBRUSD.vx", "XNGUSD.vx"]

FX = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD",
      "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
      "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
INSTRUMENTS = {f"{p}.vx": p for p in FX}
INSTRUMENTS.update({
    "XAUUSD.vx": "XAUUSD", "XAGUSD.vx": "XAGUSD",
    "DAX40.vx": "DEUIDXEUR", "SP500.vx": "USA500IDXUSD", "NAS100.vx": "USATECHIDXUSD", "US30.vx": "USA30IDXUSD",
    "JPN225.vx": "JPNIDXJPY", "UK100.vx": "GBRIDXGBP", "FRA40.vx": "FRAIDXEUR", "HK50.vx": "HKGIDXHKD",
    "AUS200.vx": "AUSIDXAUD", "EU50.vx": "EUSIDXEUR",
    "XTIUSD.vx": "LIGHTCMDUSD", "XBRUSD.vx": "BRENTCMDUSD", "XNGUSD.vx": "GASCMDUSD",
    "BTCUSD.vx": "BTCUSD", "ETHUSD.vx": "ETHUSD",
})
YAHOO_REF = {"XAUUSD.vx": "GC_F", "XAGUSD.vx": "SI_F", "DAX40.vx": "_GDAXI", "SP500.vx": "_GSPC", "NAS100.vx": "_NDX",
             "US30.vx": "_DJI", "JPN225.vx": "_N225", "UK100.vx": "_FTSE", "FRA40.vx": "_FCHI", "HK50.vx": "_HSI",
             "AUS200.vx": "_AXJO", "EU50.vx": "_STOXX50E", "XTIUSD.vx": "CL_F", "XBRUSD.vx": "BZ_F", "XNGUSD.vx": "NG_F",
             "BTCUSD.vx": "BTC-USD", "ETHUSD.vx": "ETH-USD"}
YAHOO_REF.update({f"{p}.vx": f"{p}_X" for p in FX})


def _get(inst: str, y: int, m: int) -> bytes | None:
    p = RAW / inst / f"{y}_{m:02d}.bi5"
    now = datetime.now(timezone.utc)
    current = (y, m) == (now.year, now.month - 1)
    if p.exists() and not current:
        return p.read_bytes()
    url = URL.format(inst=inst, y=y, m=m)
    for attempt in range(6):
        time.sleep(PAUSE_S)                              # Dukascopy throttles fast clients hard
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            return raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")                       # no data for this month: remember it
                return b""
            if e.code in (429, 503):
                log.info("rate limited (%s) at %s %d-%02d -> sleeping %ds", e.code, inst, y, m + 1, BLOCKED_SLEEP_S)
                time.sleep(BLOCKED_SLEEP_S)
                continue
            time.sleep(10 * (attempt + 1))
        except Exception:
            time.sleep(10 * (attempt + 1))
    return None


def _decode(raw: bytes, y: int, m: int) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    data = lzma.decompress(raw)
    a = np.array(list(struct.iter_unpack(">5if", data)))
    base = pd.Timestamp(year=y, month=m + 1, day=1, tz="UTC")
    d = pd.DataFrame(dict(time=base + pd.to_timedelta(a[:, 0], unit="s"), open=a[:, 1], close=a[:, 2],
                          low=a[:, 3], high=a[:, 4], volume=a[:, 5]))
    return d[d.volume > 0]


def _scale(symbol: str, d: pd.DataFrame) -> float:
    ref = ROOT / "data" / "yahoo_D1" / f"{YAHOO_REF.get(symbol, '')}.csv"
    if not ref.exists():
        raise RuntimeError(f"no Yahoo reference for {symbol}")
    y = pd.read_csv(ref, parse_dates=["date"], index_col="date")["close"].dropna()
    daily = d.set_index("time")["close"].resample("1D").last().dropna()
    daily.index = daily.index.tz_localize(None)
    j = pd.concat([daily, y], axis=1, join="inner").dropna()
    ratio = float(np.median(j.iloc[:, 0] / j.iloc[:, 1]))
    return 10.0 ** round(np.log10(ratio))


def fetch_symbol(symbol: str, workers: int = 1) -> int:
    inst = INSTRUMENTS[symbol]
    now = datetime.now(timezone.utc)
    months = [(y, m) for y in range(START_YEAR, now.year + 1) for m in range(12) if (y, m) <= (now.year, now.month - 1)]
    with ThreadPoolExecutor(workers) as ex:
        raws = list(ex.map(lambda ym: _get(inst, *ym), months))
    missing = sum(r is None for r in raws)
    parts = [_decode(r, y, m) for r, (y, m) in zip(raws, months) if r]
    if not parts:
        log.warning("%s (%s): no data", symbol, inst)
        return 0
    d = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    s = _scale(symbol, d)
    for k in ("open", "high", "low", "close"):
        d[k] = d[k] / s
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"{symbol}.csv", index=False)
    log.info("%-11s %-14s %7d bars %s -> %s  scale 1/%g  failed months %d", symbol, inst, len(d),
             d.time.iloc[0].date(), d.time.iloc[-1].date(), s, missing)
    return len(d)


def load(symbol: str) -> pd.DataFrame | None:
    p = OUT / f"{symbol}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d


def main() -> int:
    syms = sys.argv[1:] or PRIORITY + [s for s in INSTRUMENTS if s not in PRIORITY]
    for rnd in range(3):                                 # later rounds only re-request failed months
        for s in syms:
            try:
                fetch_symbol(s)
            except Exception as e:
                log.exception("%s failed: %s", s, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
