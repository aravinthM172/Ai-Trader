"""
Read the verified zip history (tools/fetch_zip_history.py) into clean UTC 1-minute candles, and audit it.

    python -m tools.zip_history audit          # time zone check vs Dukascopy, gaps, bad candles for all five markets

load_m1(symbol) -> DataFrame time (UTC), open, high, low, close   (cached as data/zips/m1_<symbol>.pkl)
resample(df, "5min" / "15min" / "1h" / ...) -> candles on the same UTC grid
Sources: BTCUSD / ETHUSD = Binance BTCUSDT / ETHUSDT (official, SHA-256 checked); XAUUSD / USDJPY / NDX100 = HistData
XAUUSD / USDJPY / NSXUSD.  HistData timestamps are converted with HISTDATA_TZ (found by the audit, not assumed).
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Z = ROOT / "data" / "zips"
SRC = {"BTCUSD": ("binance", "BTCUSDT"), "ETHUSD": ("binance", "ETHUSDT"), "XAUUSD": ("histdata", "XAUUSD"),
       "USDJPY": ("histdata", "USDJPY"), "NDX100": ("histdata", "NSXUSD")}
HISTDATA_TZ = "America/New_York"   # found by audit(): New York local time WITH daylight saving (not fixed UTC-5)


def _binance(name: str) -> pd.DataFrame:
    parts = []
    for f in sorted((Z / "binance" / name).glob("*.zip")):
        with zipfile.ZipFile(f) as zf:
            raw = zf.read(zf.namelist()[0])
        d = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 1, 2, 3, 4], names=["t", "open", "high", "low", "close"])
        if not np.issubdtype(d["t"].dtype, np.number):            # some months carry a header row
            d = d[pd.to_numeric(d["t"], errors="coerce").notna()].astype(float)
        unit = "us" if d["t"].iloc[0] > 1e14 else "ms"
        d["time"] = pd.to_datetime(d["t"].astype("int64"), unit=unit, utc=True)
        parts.append(d[["time", "open", "high", "low", "close"]])
    return pd.concat(parts, ignore_index=True)


def _histdata(name: str, tz: str = HISTDATA_TZ) -> pd.DataFrame:
    parts = []
    for f in sorted((Z / "histdata" / name).glob("*.zip")):
        with zipfile.ZipFile(f) as zf:
            csv = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            raw = zf.read(csv)
        d = pd.read_csv(io.BytesIO(raw), sep=";", header=None, names=["ts", "open", "high", "low", "close", "v"])
        d["time"] = pd.to_datetime(d["ts"], format="%Y%m%d %H%M%S")
        parts.append(d[["time", "open", "high", "low", "close"]])
    d = pd.concat(parts, ignore_index=True)
    d["time"] = d["time"].dt.tz_localize(tz).dt.tz_convert("UTC")
    return d


def load_m1(sym: str, refresh: bool = False) -> pd.DataFrame:
    cache = Z / f"m1_{sym}.pkl"
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    src, name = SRC[sym]
    d = _binance(name) if src == "binance" else _histdata(name)
    d = d.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    d = d[(d.high >= d.low) & (d.close > 0)]
    d.to_pickle(cache)
    return d


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = d.groupby(d.time.dt.floor(rule))
    return pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                         "low": g.low.min().values, "close": g.close.last().values}).reset_index(drop=True)


def _ref_h1(sym: str) -> pd.Series | None:
    """Independent UTC reference closes: Dukascopy (gold, Nasdaq 100) or Bitstamp (BTC)."""
    from backtest import full_reassessment as fr
    p = {"XAUUSD": "dukascopy_H1/XAUUSD.vx.csv", "NDX100": "dukascopy_H1/NAS100.vx.csv", "BTCUSD": "btcusd_bitstamp_H1.csv",
         "ETHUSD": "dukascopy_H1/ETHUSD.vx.csv"}.get(sym)
    if not p or not (ROOT / "data" / p).exists():
        return None
    d = fr._read(ROOT / "data" / p)
    return d.set_index("time").close


def audit() -> None:
    for sym, (src, name) in SRC.items():
        raw = _binance(name) if src == "binance" else _histdata(name, "UTC")      # HistData read as-is first
        ref = _ref_h1(sym)
        line = f"{sym:7} {src:8} {len(raw):>9,} minutes {raw.time.iloc[0].date()}..{raw.time.iloc[-1].date()}"
        if ref is not None:
            h = raw.groupby(raw.time.dt.floor("1h")).close.last()
            best = []
            for sh in range(-7, 8):
                x = h.copy(); x.index = x.index + pd.Timedelta(hours=sh)
                j = pd.concat([x, ref], axis=1, join="inner").dropna()
                if len(j) > 100:
                    rel = ((j.iloc[:, 0] - j.iloc[:, 1]).abs() / j.iloc[:, 1]).median()
                    best.append((rel, sh, len(j), j))
            rel, sh, n, j = min(best, key=lambda b: b[0])
            summer = j[j.index.month.isin([6, 7, 8])]; winter = j[j.index.month.isin([12, 1, 2])]
            f = lambda q: ((q.iloc[:, 0] - q.iloc[:, 1]).abs() / q.iloc[:, 1]).median() * 100
            line += f" | vs reference: best shift {sh:+d} h, median diff {100 * rel:.3f} % ({n:,} hours; summer {f(summer):.3f} %, winter {f(winter):.3f} %)"
        print(line, flush=True)
    print("\nAfter conversion to UTC (HistData as", HISTDATA_TZ, "):")
    for sym in SRC:
        d = load_m1(sym, refresh=True)
        t = d.time
        gap = t.diff().dt.total_seconds().div(60).fillna(1)
        crypto = sym in ("BTCUSD", "ETHUSD")
        if crypto:
            missing = int((gap - 1).clip(lower=0).sum())
            big = gap[gap > 60]
            print(f"{sym:7} {len(d):>9,} min | missing minutes {missing:,} ({100 * missing / (len(d) + missing):.3f} %) | gaps > 1 h: "
                  f"{len(big)} (longest {gap.max() / 60:.1f} h on {t[gap.idxmax()].date()})")
        else:
            wk = t.dt.weekday
            inside = gap[(gap > 1) & (gap < 60 * 40)]                      # holes inside the trading week (not weekends)
            hours_holes = gap[(gap > 90) & (gap < 60 * 40)]
            days = t[gap.index.isin(hours_holes.index)].dt.date
            print(f"{sym:7} {len(d):>9,} min | in-week holes > 90 min: {len(hours_holes)} (on {days.nunique()} days, "
                  f"most are the daily 1-h break or holidays) | longest in-week hole {inside.max() / 60:.1f} h")
        rng = d.high - d.low
        spikes = int((rng > 25 * rng.rolling(500, min_periods=50).median()).sum())
        print(f"        bad candles {int((d.high < d.low).sum())} | spikes (> 25x median range) {spikes}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        audit()
