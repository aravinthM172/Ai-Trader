"""
Chop-filter study: would skipping the live H1 momentum's entries in sideways markets have helped?

    python -m backtest.chop_filter_study

Same entries as the live trader (strategy/btc_h1_signal PARAMS, entry at the next bar's open,
2 ATR stop, one position at a time, 96-bar time exit), on broker H1 XAUUSD, BTCUSD, DAX40, plus
Bitstamp BTC 2014-2018 as unseen confirmation.  Each filter is applied to that fixed entry set
(post-filter: a skipped trade does not free the slot for a later one).

PRE-REGISTERED (written before the first run; book thresholds, nothing tuned).  All measured on
the H1 bars up to and including the signal bar (the bar before entry) -- no look-ahead:
  adx25     J. Welles Wilder, "New Concepts in Technical Trading Systems": ADX(14) >= 25 = trending
  chop618   E.W. Dreiss, Choppiness Index(14) < 61.8 = not choppy
  er030     Perry Kaufman, "Trading Systems and Methods": efficiency ratio(20) >= 0.30 = trending

Primary exit = the live one (target 3 ATR).  The proposed 6 ATR target is tested too; both count
as trials in research/ledger.csv.

PASS RULE per (filter, exit), on the three broker symbols pooled (r4.filter_test):
  keeps >= 40 % of trades; kept mean R > removed mean R on the full sample, first 70 % and last 30 %;
  filtering helps in >= 6 of 8 time folds; Welch t (kept vs removed) >= 2
AND on Bitstamp BTC 2014-2018 (never used here): kept mean R > removed mean R.

Writes reports/chop_filter_study.json and CHOP_FILTER_STUDY.md.  Read-only; no orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import strategy_round3_daily as r3
from backtest import strategy_round4_intermarket as r4
from backtest.btc_strategies import base_atr, build_momentum
from common.logging_setup import get_logger
from strategy.btc_h1_signal import PARAMS

log = get_logger("chop", filename="chop_filter.log")
ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ["XAUUSD.vx", "BTCUSD.vx", "DAX40.vx"]
BITSTAMP = ROOT / "data" / "btcusd_bitstamp_H1.csv"
EXITS = {"tp3": 3.0, "tp6": 6.0}


# -- indicators: value at bar i uses bars <= i ----------------------------------------------
def _wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def adx(h, l, c, n=14):
    up, dn = np.diff(h, prepend=np.nan), -np.diff(l, prepend=np.nan)
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr_, pdi_, ndi_ = _wilder(tr, n), _wilder(pdm, n), _wilder(ndm, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, ndi = 100 * pdi_ / atr_, 100 * ndi_ / atr_
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    return _wilder(np.nan_to_num(dx, nan=0.0), n)


def chop(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    s = pd.Series(tr).rolling(n).sum().to_numpy()
    rng = pd.Series(h).rolling(n).max().to_numpy() - pd.Series(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * np.log10(s / rng) / np.log10(n)


def efficiency_ratio(c, n=20):
    net = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    path = pd.Series(np.abs(np.diff(c, prepend=np.nan))).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return net / path


FILTERS = {
    "adx25": lambda h, l, c: adx(h, l, c) >= 25,
    "chop618": lambda h, l, c: chop(h, l, c) < 61.8,
    "er030": lambda h, l, c: efficiency_ratio(c) >= 0.30,
}


# -- trades ---------------------------------------------------------------------------------
def momentum_trades(df: pd.DataFrame, cost_frac: float, tp_atr: float) -> pd.DataFrame:
    """Live entries; returns entry time, signal bar index, direction, R after costs."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ent, _ = build_momentum(c, h, l, **PARAMS)
    A = base_atr(h, l, c, 14)
    out, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep, sl, r, j = o[i], 2 * a, None, i
        for j in range(i, min(n, i + 96)):
            adverse, favour = ((ep - l[j]), (h[j] - ep)) if d > 0 else ((h[j] - ep), (ep - l[j]))
            if adverse >= sl:
                r = -1.0; break
            if favour >= tp_atr * a:
                r = tp_atr / 2; break
        if r is None:
            j = min(n - 1, i + 96)
            r = d * (o[j] - ep) / sl
        out.append((df["time"].iloc[i], i - 1, d, r - cost_frac * ep / sl))
        i = j + 1
    return pd.DataFrame(out, columns=["entry", "sig_i", "dir", "R"])


def keep_mask(df: pd.DataFrame, t: pd.DataFrame, name: str) -> np.ndarray:
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    ok = FILTERS[name](h, l, c)
    return np.asarray(ok[t["sig_i"].to_numpy()], dtype=bool)


def _load(sym):
    return pd.read_csv(r3.H1CACHE / f"{sym}.csv", parse_dates=["time"])


def main() -> int:
    specs = json.loads(r3.SPECS.read_text())
    data = {s: _load(s) for s in SYMBOLS}
    cost = {s: r3.broker_spread_frac(specs[s]) * (1 + 2 * r3.SLIP_FRAC) for s in SYMBOLS}
    bs = pd.read_csv(BITSTAMP)
    bs["time"] = pd.to_datetime(bs["time"], utc=True)
    bs = bs[(bs.time >= "2014-01-01") & (bs.time < "2019-01-01")].reset_index(drop=True)
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + len(FILTERS) * len(EXITS)

    res = {}
    for ex, tp in EXITS.items():
        trades = {s: momentum_trades(data[s], cost[s], tp) for s in SYMBOLS}
        tb = momentum_trades(bs, cost["BTCUSD.vx"], tp)
        for f in FILTERS:
            pooled = pd.concat([t.assign(sym=s, keep=keep_mask(data[s], t, f)) for s, t in trades.items()],
                               ignore_index=True).sort_values("entry").reset_index(drop=True)
            r = r4.filter_test(pooled, pooled["keep"].to_numpy(dtype=bool))
            kb = keep_mask(bs, tb, f)
            r["bitstamp_unseen"] = dict(trades=len(tb), kept=int(kb.sum()), all_R=round(tb.R.mean(), 4),
                                        kept_R=round(tb.R[kb].mean(), 4), removed_R=round(tb.R[~kb].mean(), 4))
            r["checks"]["bitstamp_kept_gt_removed"] = bool(tb.R[kb].mean() > tb.R[~kb].mean())
            r["passes"] = all(r["checks"].values())
            per = {}
            for s in SYMBOLS:
                k = keep_mask(data[s], trades[s], f)
                R = trades[s].R
                per[s] = dict(all_R=round(R.mean(), 4), kept_R=round(R[k].mean(), 4), removed_R=round(R[~k].mean(), 4),
                              kept_pct=round(100 * k.mean()))
            r["per_symbol"] = per
            res[f"{f}|{ex}"] = r
            log.info("%s|%s kept %d/%d all %.3f kept %.3f removed %.3f t=%.2f bitstamp kept %.3f removed %.3f PASS=%s",
                     f, ex, r["kept"], r["trades"], r["all_R"], r["kept_R"], r["removed_R"], r["welch_t"],
                     r["bitstamp_unseen"]["kept_R"], r["bitstamp_unseen"]["removed_R"], r["passes"])

    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r3.append_ledger([dict(tested_utc=tested, name=f"chop_{k.replace('|', '_')}", params=json.dumps({"filter": k}),
                           lookahead_ok=True, xau_exp_R=r["per_symbol"]["XAUUSD.vx"]["kept_R"],
                           bitstamp_exp_R=r["bitstamp_unseen"]["kept_R"], n_trials=n_trials,
                           passes=r["passes"], useful=r["passes"]) for k, r in res.items()])
    rep = dict(meta=dict(n_trials=n_trials, rule="pre-registered in module docstring"), results=res)
    (ROOT / "reports" / "chop_filter_study.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "CHOP_FILTER_STUDY.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep) -> str:
    L = ["# Chop-filter study -- skip the live momentum's entries in sideways markets?", "",
         f"Pre-registered in `backtest/chop_filter_study.py`; ledger N = {rep['meta']['n_trials']}.  "
         "Pooled = broker H1 gold + BTC + DAX40; unseen = Bitstamp BTC 2014-2018.", "",
         "| filter | exit | kept | mean R all | kept | removed | last 30 % kept / removed | folds | Welch t | unseen kept / removed | PASS |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for k, r in rep["results"].items():
        f, ex = k.split("|")
        b = r["bitstamp_unseen"]
        L.append(f"| {f} | {ex} | {100 * r['kept'] / r['trades']:.0f} % | {r['all_R']} | {r['kept_R']} | {r['removed_R']} | "
                 f"{r['kept_R_last30']} / {r['removed_R_last30']} | {r['folds_positive']}/8 | {r['welch_t']} | "
                 f"{b['kept_R']} / {b['removed_R']} | {'**yes**' if r['passes'] else 'no'} |")
    L += ["", "Per symbol (mean R: all / kept / removed, % kept):", ""]
    for k, r in rep["results"].items():
        L.append(f"- {k}: " + "; ".join(f"{s.split('.')[0]} {v['all_R']} / {v['kept_R']} / {v['removed_R']} ({v['kept_pct']} %)"
                                       for s, v in r["per_symbol"].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
