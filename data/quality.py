"""
Market-data quality validation.  Asset-agnostic; driven by AssetConfig.

Never silently repairs data.  Bad rows are separated out and logged.
Returns (clean_df, report).  ``report.ok`` is False when the data cannot be
trusted for trading decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common.logging_setup import get_logger

log = get_logger("btc.data")

_OHLC = ("open", "high", "low", "close")


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    n_input: int = 0
    n_clean: int = 0
    n_rejected: int = 0
    earliest: str | None = None
    latest: str | None = None
    expected_step_seconds: int = 0
    issues: dict[str, int] = field(default_factory=dict)
    rejected_samples: list[dict] = field(default_factory=list)
    missing_candles: int = 0
    duplicate_timestamps: int = 0
    gap_events: list[dict] = field(default_factory=list)
    stale: bool = False
    stale_age_seconds: float | None = None
    insufficient_history: bool = False
    zero_volume_bars: int = 0
    has_spread_column: bool = False
    ok: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _step_seconds(tf: str) -> int:
    return {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
            "H1": 3600, "H4": 14400, "D1": 86400}.get(tf.upper(), 300)


def validate_ohlc(df: pd.DataFrame, *, symbol: str, timeframe: str, cfg) -> tuple[pd.DataFrame, QualityReport]:
    rep = QualityReport(symbol=symbol, timeframe=timeframe)
    rep.expected_step_seconds = _step_seconds(timeframe)

    if df is None or len(df) == 0:
        rep.ok = False
        rep.reasons.append("no data")
        rep.insufficient_history = True
        log.error("[%s %s] no data", symbol, timeframe)
        return pd.DataFrame(), rep

    d = df.copy().reset_index(drop=True)
    for c in _OHLC:
        if c not in d.columns:
            rep.ok = False
            rep.reasons.append(f"missing column '{c}'")
            return pd.DataFrame(), rep
    if "time" not in d.columns:
        rep.ok = False
        rep.reasons.append("missing column 'time'")
        return pd.DataFrame(), rep

    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    rep.n_input = len(d)
    rep.has_spread_column = "spread" in d.columns
    vol_col = "tick_volume" if "tick_volume" in d.columns else ("volume" if "volume" in d.columns else None)

    bad = pd.Series(False, index=d.index)

    def flag(mask: pd.Series, key: str):
        nonlocal bad
        n = int(mask.sum())
        if n:
            rep.issues[key] = rep.issues.get(key, 0) + n
            for _, r in d[mask].head(3).iterrows():
                rep.rejected_samples.append({
                    "issue": key, "time": str(r["time"]),
                    **{c: (None if pd.isna(r[c]) else float(r[c])) for c in _OHLC},
                })
        bad = bad | mask

    flag(d["time"].isna(), "nan_timestamp")
    flag(d[list(_OHLC)].isna().any(axis=1), "nan_ohlc")
    flag((d[list(_OHLC)] <= 0).any(axis=1), "non_positive_price")
    flag(d["high"] < d["low"], "high_lt_low")
    flag((d["close"] > d["high"]) | (d["close"] < d["low"]), "close_outside_range")
    flag((d["open"] > d["high"]) | (d["open"] < d["low"]), "open_outside_range")

    # duplicate timestamps
    dup = d["time"].duplicated(keep="first")
    rep.duplicate_timestamps = int(dup.sum())
    flag(dup, "duplicate_timestamp")

    # impossible jumps: |close-to-close return| > max_bar_jump_atr * ATR
    d_sorted = d.sort_values("time")
    prev_close = d_sorted["close"].shift(1)
    tr = np.maximum.reduce([
        (d_sorted["high"] - d_sorted["low"]).to_numpy(),
        (d_sorted["high"] - prev_close).abs().to_numpy(),
        (d_sorted["low"] - prev_close).abs().to_numpy(),
    ])
    atr = pd.Series(tr, index=d_sorted.index).rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
    jump = (d_sorted["close"] - prev_close).abs()
    impossible = (jump > cfg.max_bar_jump_atr * atr) & atr.notna()
    impossible = impossible.reindex(d.index).fillna(False)
    flag(impossible, "impossible_jump")

    clean = d[~bad].sort_values("time").reset_index(drop=True)
    rep.n_rejected = int(bad.sum())
    rep.n_clean = len(clean)

    if len(clean):
        rep.earliest = str(clean["time"].iloc[0])
        rep.latest = str(clean["time"].iloc[-1])
        # missing candles (gaps) -- weekend gaps for BTC 24/7 are still flagged, informational
        deltas = clean["time"].diff().dt.total_seconds().dropna()
        step = rep.expected_step_seconds
        missing = ((deltas / step).round() - 1).clip(lower=0)
        rep.missing_candles = int(missing.sum())
        gap_mask = deltas > step * cfg.max_candle_gap_multiplier
        for i in np.where(gap_mask.to_numpy())[0][:20]:
            idx = deltas.index[i]
            rep.gap_events.append({
                "after": str(clean["time"].iloc[idx - 1]),
                "before": str(clean["time"].iloc[idx]),
                "gap_bars": int(round(deltas.iloc[i] / step)) - 1,
            })
        if vol_col:
            rep.zero_volume_bars = int((clean[vol_col] <= 0).sum())
        # staleness of the most recent bar
        age = (datetime.now(timezone.utc) - clean["time"].iloc[-1]).total_seconds()
        rep.stale_age_seconds = float(age)
        rep.stale = age > cfg.stale_seconds

    rep.insufficient_history = rep.n_clean < cfg.min_history_bars

    if rep.insufficient_history:
        rep.ok = False
        rep.reasons.append(f"insufficient history ({rep.n_clean} < {cfg.min_history_bars})")
    if rep.stale:
        rep.ok = False
        rep.reasons.append(f"stale data (last bar {rep.stale_age_seconds:.0f}s old > {cfg.stale_seconds}s)")
    # hard-fail issues (structural corruption in the *clean* set should be zero, but bad ratio matters)
    if rep.n_input and rep.n_rejected / rep.n_input > 0.02:
        rep.ok = False
        rep.reasons.append(f"rejected ratio {rep.n_rejected / rep.n_input:.1%} > 2%")

    lvl = log.warning if not rep.ok else log.info
    lvl("[%s %s] clean=%d rejected=%d issues=%s missing=%d dup=%d stale=%s ok=%s",
        symbol, timeframe, rep.n_clean, rep.n_rejected, rep.issues,
        rep.missing_candles, rep.duplicate_timestamps, rep.stale, rep.ok)
    if rep.n_rejected:
        log.warning("[%s %s] rejected samples: %s", symbol, timeframe, rep.rejected_samples[:5])

    return clean, rep
