"""
Daily-trend regime filter for the multi trader (added 2026-10-08).

side200: a BUY is allowed only when the last COMPLETED daily close is above that day's EMA200,
a SELL only when it is below.  Applied per symbol from MULTI_REGIME_SYMBOLS (default: none = off).

Evidence (scratchpad regime_sizing_bt.py + combined_bt.py, 2026-10-08): on gold, Dukascopy H1
2005-2026, live exit 3 ATR: +0.028R -> +0.069R per trade, total R +174 -> +221, max DD 64R -> 51R,
better in 7 of 8 time folds and on the FundingPips feed 2010-.  It did not hold up on GER40 and
adds little on BTC, so it is meant for XAUUSD only.  ~550 ideas sit in research/ledger.csv:
treat it as a strong lead, not a proven edge.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

EMA_LEN = 200
D1_BARS = 1000            # history fetched for the EMA (ewm warm-up: 1000 bars ~ the full-history value)
MIN_BARS = 400            # fewer completed daily bars than this -> no decision (fail closed)


def parse_symbols(raw: str) -> set[str]:
    """'XAUUSD, BTCUSD' -> {'XAUUSD', 'BTCUSD'}."""
    return {s.strip() for s in (raw or "").split(",") if s.strip()}


def completed_daily(d1: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop the still-forming daily bar: a D1 bar is complete once 24 h have passed since it opened."""
    t = pd.to_datetime(d1["time"], utc=True)
    return d1[t + pd.Timedelta(days=1) <= pd.Timestamp(now)].reset_index(drop=True)


def side200(direction: str, d1: pd.DataFrame | None, now: datetime) -> tuple[bool | None, str]:
    """(allowed, reason).  allowed None = not enough daily data to decide (caller retries)."""
    if d1 is None or d1.empty:
        return None, "no daily bars"
    done = completed_daily(d1, now)
    if len(done) < MIN_BARS:
        return None, f"only {len(done)} completed daily bars (< {MIN_BARS})"
    close = done["close"].astype(float)
    ema = float(close.ewm(span=EMA_LEN, adjust=False).mean().iloc[-1])
    last = float(close.iloc[-1])
    side = "above" if last > ema else "below"
    allowed = (direction == "BUY" and last > ema) or (direction == "SELL" and last < ema)
    return allowed, f"daily close {last:.5g} {side} EMA{EMA_LEN} {ema:.5g}"
