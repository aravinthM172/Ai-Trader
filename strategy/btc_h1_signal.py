"""
Live adapter for the VALIDATED H1 strategy `momentum_rsi_mtf`.

It calls backtest/btc_strategies.build_momentum() VERBATIM (same code path the
walk-forward / cost-sensitivity / Monte-Carlo validation used) so the live
signal cannot drift from the backtested one.  Parameters are FROZEN:

    rsi_buy = 60, rsi_sell = 40, mom_win = 8, ema_htf = 24  (rsi_period 14)

Methodology (as validated):
  * the signal is evaluated on the last COMPLETED H1 bar
  * the position is opened at the OPEN of the next H1 bar (== the live fill price)

build_momentum() places "decision-from-bar-k" at array index k+1.  With the last
completed bar at index L, that decision would sit at L+1 (one past the end), so
we append a single synthetic trailing bar (a copy of bar L -- it feeds only
*later* indices of the causal EMA/RSI kernels, never index L) and read
entries[L+1].  No look-ahead: entries[L+1] depends only on bars 0..L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.btc_strategies import build_momentum, _ema, _rsi, warmup_for

STRATEGY_NAME = "momentum_rsi_mtf"
PARAMS = dict(rsi_buy=60.0, rsi_sell=40.0, mom_win=8, ema_htf=24)   # FROZEN -- do not tune


@dataclass
class Signal:
    symbol: str
    decision: str          # BUY | SELL | HOLD
    confidence: float
    reason: str
    features: dict
    strategy: str = STRATEGY_NAME
    signal_bar_utc: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _min_bars() -> int:
    return warmup_for(PARAMS["ema_htf"] * 4, PARAMS["mom_win"], 14) + 5


def generate(symbol: str, df_completed: pd.DataFrame, *, min_confidence: float = 0.70) -> Signal:
    """
    df_completed : DataFrame of COMPLETED H1 bars (the still-forming bar removed
                   by the caller), ascending time, columns open/high/low/close/time.
    """
    if df_completed is None or len(df_completed) < _min_bars():
        return Signal(symbol, "HOLD", 0.0,
                      f"insufficient completed H1 bars ({0 if df_completed is None else len(df_completed)} < {_min_bars()})",
                      {})
    c = df_completed["close"].to_numpy(float)
    h = df_completed["high"].to_numpy(float)
    l = df_completed["low"].to_numpy(float)

    ext_c = np.append(c, c[-1])
    ext_h = np.append(h, h[-1])
    ext_l = np.append(l, l[-1])
    entries, _ = build_momentum(ext_c, ext_h, ext_l, **PARAMS)
    d = int(entries[-1])                          # decision from the LAST completed bar

    # condition values on the signal bar (for the trade record 'reason')
    rsi = _rsi(c, 14)
    mom = pd.Series(c).diff(PARAMS["mom_win"]).to_numpy()
    ema_htf = _ema(c, PARAMS["ema_htf"] * 4)
    sig_bar_time = str(df_completed["time"].iloc[-1]) if "time" in df_completed.columns else None
    reason = (f"{STRATEGY_NAME}: signal_bar={sig_bar_time} "
              f"rsi14={rsi[-1]:.1f} mom{PARAMS['mom_win']}={mom[-1]:+.1f} "
              f"close>{PARAMS['ema_htf']*4}EMA={bool(c[-1] > ema_htf[-1])} -> "
              f"{'BUY' if d == 1 else 'SELL' if d == -1 else 'HOLD'}")
    feats = {
        "rsi14": float(rsi[-1]),
        f"mom{PARAMS['mom_win']}": float(mom[-1]),
        "close": float(c[-1]),
        "ema96": float(ema_htf[-1]) if np.isfinite(ema_htf[-1]) else None,
        "close_above_htf_ema": bool(c[-1] > ema_htf[-1]) if np.isfinite(ema_htf[-1]) else None,
    }
    if d == 1:
        return Signal(symbol, "BUY", 1.0, reason, feats, signal_bar_utc=sig_bar_time)
    if d == -1:
        return Signal(symbol, "SELL", 1.0, reason, feats, signal_bar_utc=sig_bar_time)
    return Signal(symbol, "HOLD", 0.0, reason, feats, signal_bar_utc=sig_bar_time)
