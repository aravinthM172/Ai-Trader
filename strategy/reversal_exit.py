"""
Reversal exit for the multi trader (added 2026-10-14): bank a trade that is well in profit when a big hourly
candle goes against it.

The trade is closed at the start of the next hour when the last completed H1 bar
  - closed against the trade (a down candle for a BUY, an up candle for a SELL),
  - is at least CANDLE_ATR (1.5) x ATR14 from high to low, and
  - still closed at least MIN_PROFIT_ATR (2.0) x the trade's entry ATR in profit (the stop is 2 ATR, so +1R).
Applied per symbol from MULTI_REVERSAL_EXIT_SYMBOLS (default: none = off).  Stop and target are unchanged.

Evidence (scratchpad reversal.py, 2026-10-14, live rule 2 / 6 ATR, long H1 history, first half + second half):
BTCUSD 61 + 25 R per year against 62 + 25 without it, worst drawdown 30R against 41R, win rate 37.5 % against
33.5 %; ETHUSD 64 + 21 against 68 + 17, drawdown 28R against 36R.  About 15 % of trades end this way, at +1.4R on
average.  On the 18 live BTC / ETH trades to 2026-10-13 it closed two losers at +1.4R and touched no winner.
Sixteen variants of eight signals were tried, so read it as "same profit, smoother", not as an edge; it did not
help on gold or USDJPY and acting from +1 ATR cost profit everywhere.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from backtest.btc_strategies import base_atr

ATR_LEN = 14
MIN_PROFIT_ATR = 2.0
CANDLE_ATR = 1.5
H1_BARS = 120             # enough for the ATR to settle
MIN_BARS = 40


def check(done: pd.DataFrame | None, *, direction: str, entry: float, atr_entry: float, opened: datetime,
          min_profit_atr: float = MIN_PROFIT_ATR, candle_atr: float = CANDLE_ATR) -> tuple[bool, str]:
    """(close now?, reason).  done = completed H1 bars, oldest first; only the last one is judged."""
    if done is None or len(done) < MIN_BARS or atr_entry <= 0 or entry <= 0:
        return False, "not enough data"
    last = done.iloc[-1]
    if pd.Timestamp(last["time"]) < pd.Timestamp(opened).floor("1h"):
        return False, "no completed bar since the entry"
    atr = float(base_atr(done["high"].to_numpy(float), done["low"].to_numpy(float), done["close"].to_numpy(float), ATR_LEN)[-1])
    if not atr > 0:
        return False, "no ATR"
    d = 1.0 if direction == "BUY" else -1.0
    o, h, l, c = (float(last[k]) for k in ("open", "high", "low", "close"))
    up = d * (c - entry) / atr_entry
    if up < min_profit_atr:
        return False, f"{up:.1f} ATR in profit (needs {min_profit_atr:g})"
    if d * (c - o) >= 0:
        return False, "last candle went with the trade"
    size = (h - l) / atr
    if size < candle_atr:
        return False, f"candle against is {size:.1f} ATR (needs {candle_atr:g})"
    return True, f"reversal: {size:.1f} ATR candle against while {up:.1f} ATR in profit"
