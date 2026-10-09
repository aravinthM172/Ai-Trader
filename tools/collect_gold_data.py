"""
Collect everything for a dedicated XAUUSD study into data/gold/ -- prices on every timeframe from
several sources, plus the macro drivers of gold.  Re-runnable: each source is re-downloaded only if
missing or older than a day.  Writes data/gold/MANIFEST.json (rows, span, gaps per file).

    python -m tools.collect_gold_data

Sources
  FundingPips MT5 (the feed the bot trades): XAUUSD M5 / M15 / H1 / H4 / D1, max history, with spread;
      XAGUSD H1 / D1
  Broker Valetax archive (already on disk): XAUUSD H1 2009-, D1 2004-, M5 2025-
  Dukascopy (tools/fetch_dukascopy): XAUUSD H1 2005- (when complete)
  Yahoo: GC=F gold futures, SI=F silver, DX-Y.NYB dollar index, ^TNX 10y yield, ^VIX, GLD (volume)
  FRED (public CSV, no key): DFII10 10y real yield, DGS10 10y nominal, T10YIE 10y breakeven inflation,
      DTWEXBGS broad trade-weighted dollar
  CFTC: gold futures Commitments of Traders (disaggregated, COMEX gold 088691), weekly, 2010-
  Events: FOMC + NFP history (data/events_history.csv, news/history.py)
Read-only on MT5; no orders.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.logging_setup import get_logger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gold"
log = get_logger("gold.data", filename="gold_data.log")
UA = {"User-Agent": "Mozilla/5.0 (research; gold-ai-trader)"}
FRED = {"DFII10": "real_yield_10y", "DGS10": "yield_10y", "T10YIE": "breakeven_10y", "DTWEXBGS": "dollar_broad"}
YAHOO = {"GC=F": "gold_futures", "SI=F": "silver_futures", "DX-Y.NYB": "dxy", "^TNX": "tnx", "^VIX": "vix", "GLD": "gld"}
MT5_TF = {"M5": 200_000, "M15": 200_000, "H1": 200_000, "H4": 50_000, "D1": 20_000}


def _fresh(p: Path, hours: float = 24) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < hours * 3600


def _http(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def mt5_feed() -> None:
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        log.error("MT5 connection failed -- FundingPips feed skipped")
        return
    try:
        for sym, tfs in (("XAUUSD", MT5_TF), ("XAGUSD", {"H1": 200_000, "D1": 20_000})):
            gw.raw().symbol_select(sym, True)
            for tf, n in tfs.items():
                p = OUT / f"fp_{sym}_{tf}.csv"
                if _fresh(p):
                    continue
                d = gw.get_rates(sym, tf, n)
                if d is None or d.empty:
                    log.warning("FundingPips %s %s: no data", sym, tf)
                    continue
                d.to_csv(p, index=False)
                log.info("FundingPips %s %s: %d bars %s -> %s", sym, tf, len(d), d.time.iloc[0], d.time.iloc[-1])
    finally:
        gw.shutdown()


def yahoo() -> None:
    import yfinance as yf
    for tk, name in YAHOO.items():
        p = OUT / f"yahoo_{name}_D1.csv"
        if _fresh(p):
            continue
        d = yf.download(tk, period="max", interval="1d", progress=False, auto_adjust=False)
        if d is None or d.empty:
            log.warning("Yahoo %s: no data", tk)
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d.to_csv(p, index_label="date")
        log.info("Yahoo %s: %d days from %s", tk, len(d), d.index[0].date())


def fred() -> None:
    for sid, name in FRED.items():
        p = OUT / f"fred_{name}.csv"
        if _fresh(p):
            continue
        raw = _http(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
        d = pd.read_csv(io.BytesIO(raw))
        d.columns = ["date", name]
        d[name] = pd.to_numeric(d[name], errors="coerce")
        d.dropna().to_csv(p, index=False)
        log.info("FRED %s: %d rows from %s", sid, len(d), d.date.iloc[0])


def cot() -> None:
    p = OUT / "cot_gold.csv"
    if _fresh(p, 24 * 7):
        return
    frames = []
    for y in range(2010, datetime.now(timezone.utc).year + 1):
        try:
            z = zipfile.ZipFile(io.BytesIO(_http(f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip")))
            d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        except Exception as e:
            log.warning("CFTC %d: %s", y, e)
            continue
        code = next(c for c in d.columns if c.strip().startswith("CFTC_Contract_Market_Code"))
        d = d[d[code].astype(str).str.strip() == "088691"]
        keep = {"Report_Date_as_YYYY-MM-DD": "date", "Open_Interest_All": "open_interest",
                "M_Money_Positions_Long_All": "mm_long", "M_Money_Positions_Short_All": "mm_short",
                "Prod_Merc_Positions_Long_All": "prod_long", "Prod_Merc_Positions_Short_All": "prod_short",
                "Swap_Positions_Long_All": "swap_long", "Swap__Positions_Short_All": "swap_short"}
        cols = {c: keep[c.strip()] for c in d.columns if c.strip() in keep}
        frames.append(d[list(cols)].rename(columns=cols))
        time.sleep(1)
    if frames:
        d = pd.concat(frames).sort_values("date").drop_duplicates("date")
        d["mm_net"] = d.mm_long - d.mm_short
        d.to_csv(p, index=False)
        log.info("CFTC gold COT: %d weeks %s -> %s", len(d), d.date.iloc[0], d.date.iloc[-1])


def manifest() -> dict:
    out = {}
    for p in sorted(OUT.glob("*.csv")):
        d = pd.read_csv(p)
        tc = "time" if "time" in d.columns else "date"
        t = pd.to_datetime(d[tc], utc=True, errors="coerce").dropna()
        gaps = t.diff()
        out[p.name] = dict(rows=len(d), start=str(t.min()), end=str(t.max()),
                           biggest_gap_days=round(gaps.max().total_seconds() / 86400, 1) if len(t) > 1 else None)
    # existing broker / Dukascopy gold files, for reference
    for name, f in {"valetax_XAUUSD_H1": ROOT / "data" / "mt5_H1" / "XAUUSD.vx.csv",
                    "valetax_XAUUSD_D1": ROOT / "data" / "mt5_D1" / "XAUUSD.vx.csv",
                    "valetax_XAUUSD_M5": ROOT / "data" / "xauusd_m5_5y.csv",
                    "dukascopy_XAUUSD_H1": ROOT / "data" / "dukascopy_H1" / "XAUUSD.vx.csv"}.items():
        if f.exists():
            t = pd.to_datetime(pd.read_csv(f, usecols=["time"]).time, utc=True)
            out[name] = dict(rows=len(t), start=str(t.min()), end=str(t.max()),
                             biggest_gap_days=round(t.diff().max().total_seconds() / 86400, 1), path=str(f.relative_to(ROOT)))
    (OUT / "MANIFEST.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (mt5_feed, yahoo, fred, cot):
        try:
            step()
        except Exception as e:
            log.exception("%s failed: %s", step.__name__, e)
    for k, v in manifest().items():
        print(f"{k:34} {v['rows']:>8} rows  {v['start'][:10]} .. {v['end'][:10]}  biggest gap {v['biggest_gap_days']} d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
