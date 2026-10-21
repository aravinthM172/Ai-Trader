"""
Download Indian market history from Yahoo Finance (owner request 2026-10-21): daily candles as far back as Yahoo has
them and hourly candles for the last ~2 years (Yahoo's limit), for the main indices and the Nifty 50 stocks.

    python -m tools.fetch_yahoo_india

Writes data/india/D1/<ticker>.csv and data/india/H1/<ticker>.csv (time UTC, open, high, low, close, volume).
Nifty 50 list as of 2026 (Yahoo .NS tickers); stocks that were not in the index decades ago still have their own history,
so the stock set is SURVIVOR-BIASED (today's winners) -- results on stocks are optimistic, indices are not.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "india"
INDICES = ["^NSEI", "^NSEBANK", "^BSESN", "^CNXIT", "^NSEMDCP50", "^CNXFMCG", "^CNXAUTO", "^CNXPHARMA", "^CNXMETAL"]
NIFTY50 = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
           "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
           "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
           "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
           "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
           "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"]


def save(df: pd.DataFrame, path: Path) -> int:
    if df is None or df.empty:
        return 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])
    idx = df.index if df.index.tz is not None else df.index.tz_localize("Asia/Kolkata")
    df.index = idx.tz_convert("UTC")
    df = df[(df.high >= df.low) & (df.close > 0)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.rename_axis("time").reset_index().to_csv(path, index=False)
    return len(df)


def main() -> int:
    tickers = INDICES + [f"{s}.NS" for s in NIFTY50] + ["^BSESN"]
    rows = []
    for t in dict.fromkeys(tickers):
        name = t.replace("^", "_")
        n1 = save(yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False), OUT / "D1" / f"{name}.csv")
        time.sleep(0.5)
        nh = save(yf.download(t, period="730d", interval="1h", auto_adjust=True, progress=False), OUT / "H1" / f"{name}.csv")
        time.sleep(0.5)
        first = pd.read_csv(OUT / "D1" / f"{name}.csv", nrows=1).time.iloc[0][:10] if n1 else "-"
        rows.append((t, n1, first, nh))
        print(f"{t:16} D1 {n1:6} from {first}   H1 {nh:5}", flush=True)
    ok = sum(1 for r in rows if r[1])
    print(f"done: {ok}/{len(rows)} tickers with daily data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
