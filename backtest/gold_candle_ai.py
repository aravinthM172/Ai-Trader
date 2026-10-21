"""
Owner idea 2026-10-21: let an AI model call the next gold M5 candle green or red, and trade the bulk 0.15 lot only
when it is confident.

    python -m backtest.gold_candle_ai

Data: broker M5 with recorded spread (data/xauusd_m5_5y.csv).  Features use only CLOSED candles before the one being
predicted: last 12 candle returns / bodies / wicks (in ATR units), RSI 14, 3 / 12 / 48-bar momentum, ATR %, distance
from EMA 20 / 50 / 200, H1 EMA200 side (last completed hour), hour of day, weekday, spread.
Target: next candle green (close > open).
Models: logistic regression and LightGBM.  Train on the first 60 %, test on the last 40 % (never seen).  Also a
walk-forward: retrain every month on all data before it, predict the next month.
Trading: enter at the next open in the predicted colour when P(green) >= th or <= 1 - th, exit at that candle's close.
Cost per trade = the bar's spread + 0.25 x spread slippage per side + $6.5/lot commission, 0.15 lot.  Read-only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
LOTS, USD_PER_1, COMM, SLIP = 0.15, 15.0, 6.5, 0.25
THRESH = (0.50, 0.52, 0.55, 0.58, 0.60, 0.65)


def features(d: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = d.open, d.high, d.low, d.close
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f = pd.DataFrame(index=d.index)
    for k in range(12):                                           # candle k bars ago (0 = the last closed one)
        f[f"ret{k}"] = ((c - c.shift()) / atr).shift(k)
        f[f"body{k}"] = ((c - o) / atr).shift(k)
        f[f"uw{k}"] = ((h - np.maximum(o, c)) / atr).shift(k)
        f[f"lw{k}"] = ((np.minimum(o, c) - l) / atr).shift(k)
    dlt = c.diff()
    up, dn = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    for n in (3, 12, 48):
        f[f"mom{n}"] = (c - c.shift(n)) / atr
    f["atr_pct"] = atr / c
    for n in (20, 50, 200):
        f[f"ema{n}"] = (c - c.ewm(span=n, adjust=False).mean()) / atr
    hc = d.set_index("time").close.resample("1h").last().dropna()
    side = np.sign(hc - hc.ewm(span=200, adjust=False).mean()).shift(1)
    f["h1_side"] = side.reindex(d.time.dt.floor("1h")).to_numpy()
    f["hour"], f["wday"], f["spread"] = d.time.dt.hour, d.time.dt.weekday, d.spr
    return f


def trade_pnl(d: pd.DataFrame, idx: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Enter at the open of bar idx (ask for buys, bid for sells), exit at that bar's close."""
    o, c, s = d.open.to_numpy()[idx], d.close.to_numpy()[idx], d.spr.to_numpy()[idx]
    move = side * (c - o) - s - 2 * SLIP * s
    return move * USD_PER_1 - COMM * LOTS


def evaluate(d, p, test_idx, label):
    y = (d.close.to_numpy()[test_idx] > d.open.to_numpy()[test_idx]).astype(int)
    acc = float(((p >= 0.5) == y).mean())
    rows = []
    months = max(1.0, (d.time.iloc[test_idx[-1]] - d.time.iloc[test_idx[0]]).days / 30.44)
    for th in THRESH:
        m = (p >= th) | (p <= 1 - th)
        if m.sum() < 50:
            continue
        side = np.where(p[m] >= 0.5, 1, -1)
        pnl = trade_pnl(d, test_idx[m], side)
        hit = float((np.sign(d.close.to_numpy()[test_idx[m]] - d.open.to_numpy()[test_idx[m]]) == side).mean())
        gross = pnl + (d.spr.to_numpy()[test_idx[m]] * (1 + 2 * SLIP)) * USD_PER_1 + COMM * LOTS
        rows.append({"model": label, "confidence >=": th, "trades/day": round(m.sum() / (months * 21.7), 1),
                     "right %": round(100 * hit, 1), "gross $/trade": round(float(gross.mean()), 2),
                     "net $/trade": round(float(pnl.mean()), 2), "net $/month": round(float(pnl.sum()) / months)})
    print(f"{label}: accuracy on every unseen candle {100 * acc:.2f} % (n={len(y)})")
    return rows


def main() -> int:
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = pd.read_csv(ROOT / "data" / "xauusd_m5_5y.csv")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    from backtest.gold_cheatsheet_test import server_to_utc
    d["time"] = server_to_utc(d["time"])          # broker server time -> UTC
    d = d.dropna(subset=["time"])
    d["spr"] = d["spread"] / 100.0
    d = d.sort_values("time").reset_index(drop=True)
    X = features(d).shift(1)                                  # features known at the open of the predicted bar
    y = (d.close > d.open).astype(int)
    ok = X.notna().all(axis=1).to_numpy() & (d.time.diff().dt.total_seconds().fillna(300).to_numpy() <= 600)
    idx = np.flatnonzero(ok)
    cut = idx[int(len(idx) * 0.6)]
    tr, te = idx[idx < cut], idx[idx >= cut]
    print(f"train {d.time.iloc[tr[0]].date()}..{d.time.iloc[tr[-1]].date()} ({len(tr)}), "
          f"test {d.time.iloc[te[0]].date()}..{d.time.iloc[te[-1]].date()} ({len(te)}); "
          f"base rate green in test {100 * y.iloc[te].mean():.1f} %")
    rows = []
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    lr.fit(X.iloc[tr], y.iloc[tr])
    rows += evaluate(d, lr.predict_proba(X.iloc[te])[:, 1], te, "logistic 60/40")
    gb = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    gb.fit(X.iloc[tr], y.iloc[tr])
    rows += evaluate(d, gb.predict_proba(X.iloc[te])[:, 1], te, "lightgbm 60/40")
    # walk-forward: retrain monthly on everything before the month
    month = d.time.dt.to_period("M").to_numpy()
    pw, iw = [], []
    for m in sorted(set(month[te])):
        test_m = te[month[te] == m]
        train_m = idx[idx < test_m[0]]
        g = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        g.fit(X.iloc[train_m], y.iloc[train_m])
        pw.append(g.predict_proba(X.iloc[test_m])[:, 1])
        iw.append(test_m)
    rows += evaluate(d, np.concatenate(pw), np.concatenate(iw), "lightgbm walk-forward monthly")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
