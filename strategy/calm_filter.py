"""
Calm-market entry filter for the multi trader (added 2026-10-13).

A signal is skipped when the market is "jumpy": ATR14 as a share of price on the signal bar is at least
MAX_RATIO times its median over the last 30 days of H1 bars.  Applied per symbol from MULTI_CALM_ONLY_SYMBOLS
(default: none = off).

Evidence (scratchpad volf.py + wf.py, 2026-10-13, live rule 2 / 6 ATR, long H1 history): BTCUSD entries with a
ratio below 1.2 average +0.23R, the rest +0.06R (calm better in 11 of 13 years); ETHUSD +0.25R against +0.02R
(8 of 9 years).  In a FundedNext $15K replay it cut the worst drawdown from about $2,000 to $1,270 for 5 % less
profit.  It was found by looking at one week of live losses and it does NOT hold on XAUUSD, USDJPY or NDX100, so
it is meant for BTCUSD and ETHUSD only: a strong lead, not a proven edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.btc_strategies import base_atr

ATR_LEN = 14
MEDIAN_BARS = 720         # 30 days of H1 bars
MIN_BARS = 200            # fewer ATR values than this in the window -> no decision (caller retries)
H1_BARS = 800             # history to fetch: the median window plus the ATR warm-up
MAX_RATIO = 1.2


def ratio(done: pd.DataFrame) -> float | None:
    """ATR % of price on the last completed bar / its median over the last MEDIAN_BARS completed bars.
    None when there is too little history."""
    if done is None or len(done) < MIN_BARS + ATR_LEN:
        return None
    h, l, c = (done[k].to_numpy(float) for k in ("high", "low", "close"))
    rel = (base_atr(h, l, c, ATR_LEN) / c)[-MEDIAN_BARS:]
    ok = rel[np.isfinite(rel) & (rel > 0)]
    if len(ok) < MIN_BARS or not np.isfinite(rel[-1]) or rel[-1] <= 0:
        return None
    return float(rel[-1] / np.median(ok))


def allowed(done: pd.DataFrame, max_ratio: float = MAX_RATIO) -> tuple[bool | None, str]:
    """(allowed, reason).  allowed None = not enough bars to decide."""
    r = ratio(done)
    if r is None:
        return None, "not enough H1 bars"
    return r < max_ratio, f"volatility {r:.2f} x its 30-day normal (limit {max_ratio:g})"
