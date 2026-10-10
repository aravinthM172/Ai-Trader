"""
Machine-learning signal filter for the live momentum rule (2026-10-10).  Offline, no MT5.

Every signal of the live rule (6 markets, longest H1 history, as backtest/long_history_backtest.py) gets the
features of strategy/ml_filter.py.  A ridge model is trained on past years only (expanding walk-forward, 5-day
embargo) to predict the trade's R; the next year is scored out-of-sample.  Rule fixed in advance: take the trade
when its score is above the median score of the training signals.

Checks: score quintiles, a permutation test against random selection of the same size inside each market-year,
and a bootstrap interval on (taken - skipped).  Costs: spread + slippage + swap + standard commission.
Limit: selection is post-hoc -- a skipped trade does not free the slot for a later signal.

    python -m backtest.ml_filter_study            # validate, write ML_FILTER_STUDY.md
    python -m backtest.ml_filter_study --freeze   # also fit on all signals and write strategy/ml_filter_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import long_history_backtest as lh
from strategy import ml_filter as mf

ROOT = Path(__file__).resolve().parents[1]
SYMS = ["BTCUSD", "ETHUSD", "XAUUSD", "NDX100", "USDJPY", "GER40"]
STRONG = {"XAUUSD", "NDX100"}                       # MULTI_STRONG_ONLY_SYMBOLS on the VPS
FIRST_TEST_YEAR, EMBARGO_DAYS, N_SIM = 2016, 5, 2000
Y_CLIP = (-1.5, 3.5)


def commission_R(sym: str, px: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """Standard-account round-turn commission in R (stop = 2 ATR): crypto 0.04 %, metals / forex $5 per lot."""
    if sym in ("BTCUSD", "ETHUSD"):
        c = 0.0004 * px
    elif sym == "XAUUSD":
        c = np.full_like(px, 0.05)
    elif sym == "USDJPY":
        c = 5.0 / 100000.0 * px
    else:
        c = np.zeros_like(px)
    return c / (2.0 * atr)


def dataset() -> pd.DataFrame:
    frames = []
    for sym in SYMS:
        df, _ = lh.h1(sym)
        t = lh.run(df, sym)
        bars = mf.bar_frame(df)
        i = pd.Index(df.time).get_indexer(t.entry) - 1                     # the signal bar
        rows = [mf.features(bars.iloc[k], int(d), float(dc), float(de), int(e.hour)) if k >= 0 else None
                for k, d, dc, de, e in zip(i, t.dir, t.d_close, t.d_ema200, t.entry)]
        f = pd.DataFrame([r or {k: np.nan for k in mf.FEATS} for r in rows])
        f["sym"], f["entry"], f["exit"], f["year"] = sym, t.entry.to_numpy(), t.exit.to_numpy(), t.entry.dt.year.to_numpy()
        f["R"] = t.R.to_numpy(float) - commission_R(sym, t.entry_px.to_numpy(float), t.atr.to_numpy(float))
        live = np.ones(len(f), bool)
        if sym in lh.LIVE_FILTER:
            live &= f.trend_ok.to_numpy() > 0
        if sym in STRONG:
            live &= f.rsi_s.to_numpy() >= 15
        f["live"] = live
        frames.append(f)
    T = pd.concat(frames, ignore_index=True).dropna(subset=mf.FEATS + ["R"])
    T["entry"], T["exit"] = pd.to_datetime(T.entry, utc=True), pd.to_datetime(T.exit, utc=True)
    return T.sort_values("entry").reset_index(drop=True)


def walk_forward(T: pd.DataFrame) -> pd.DataFrame:
    T = T.assign(score=np.nan, taken=False)
    for yr in range(FIRST_TEST_YEAR, int(T.year.max()) + 1):
        tr = T[T.exit < pd.Timestamp(f"{yr}-01-01", tz="UTC") - pd.Timedelta(days=EMBARGO_DAYS)]
        te = T.index[T.year == yr]
        if len(tr) < 1500 or not len(te):
            continue
        m = mf.fit(tr[mf.FEATS].to_numpy(float), tr.R.clip(*Y_CLIP).to_numpy(float))
        s = mf.predict(m, T.loc[te, mf.FEATS].to_numpy(float))
        T.loc[te, "score"], T.loc[te, "taken"] = s, s > m["threshold"]
    return T[T.score.notna()].copy()


def row(label: str, x: pd.DataFrame) -> dict:
    if len(x) < 20:
        return {"set": label, "trades": len(x)}
    return {"set": label, "trades": len(x), "avg R": round(x.R.mean(), 3), "total R": round(x.R.sum(), 1),
            "t": round(x.R.mean() / x.R.std() * np.sqrt(len(x)), 2), "win %": round(100 * (x.R > 0).mean(), 1),
            "years +": f"{(x.groupby('year').R.sum() > 0).sum()}/{x.year.nunique()}"}


def significance(S: pd.DataFrame, rng) -> dict:
    """Taken set vs random sets of the same size inside each market-year, and a bootstrap on taken - skipped."""
    obs = S[S.taken].R.mean()
    grp = [(g.R.to_numpy(), int(g.taken.sum())) for _, g in S.groupby(["sym", "year"])]
    n = sum(k for _, k in grp)
    sims = np.array([sum(rng.choice(r, k, replace=False).sum() for r, k in grp if k) / n for _ in range(N_SIM)])
    a, b = S[S.taken].R.to_numpy(), S[~S.taken].R.to_numpy()
    diff = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_SIM)])
    return {"taken avg R": round(obs, 3), "random same size": round(sims.mean(), 3),
            "p": round((np.sum(sims >= obs) + 1) / (N_SIM + 1), 4), "taken - skipped": round(a.mean() - b.mean(), 3),
            "95 % low": round(np.percentile(diff, 2.5), 3), "95 % high": round(np.percentile(diff, 97.5), 3)}


def _md(rows: list[dict]) -> str:
    d = pd.DataFrame(rows).fillna("")
    return "\n".join(["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.to_numpy()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="fit on all signals and write strategy/ml_filter_model.json")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    T = dataset()
    O = walk_forward(T)
    L = O[O.live]
    summary = [row("all signals", O), row("live rule", L), row("live rule + filter: taken", L[L.taken]),
               row("live rule + filter: skipped", L[~L.taken]), row("all signals + filter: taken", O[O.taken]),
               row("all signals + filter: skipped", O[~O.taken])]
    last = L[L.year >= L.year.max() - 2]
    recent = [row("live rule", last), row("live rule + filter: taken", last[last.taken]),
              row("live rule + filter: skipped", last[~last.taken])]
    per = [{"market": s, **{k: v for k, v in row("", L[(L.sym == s)]).items() if k in ("trades", "avg R")},
            "taken": int(L[(L.sym == s)].taken.sum()), "taken avg R": round(L[(L.sym == s) & L.taken].R.mean(), 3),
            "skipped avg R": round(L[(L.sym == s) & ~L.taken].R.mean(), 3)} for s in SYMS]
    q = pd.qcut(O.score, 5, labels=False)
    quint = [{"score fifth": int(k) + 1, **{c: v for c, v in row("", g).items() if c != "set"}} for k, g in O.groupby(q)]
    sig = [{"scope": "all signals", **significance(O, rng)}, {"scope": "inside the live rule", **significance(L, rng)}]
    final = mf.fit(T[mf.FEATS].to_numpy(float), T.R.clip(*Y_CLIP).to_numpy(float))
    coefs = sorted(zip(mf.FEATS, final["coef"]), key=lambda z: -abs(z[1]))
    out = ["# Machine-learning signal filter -- study of 2026-10-10", "",
           f"Signals: {len(T)} ({T.entry.min().date()} to {T.entry.max().date()}), markets {', '.join(SYMS)}. "
           f"Out-of-sample years {O.year.min()}-{O.year.max()}: {len(O)} signals. Model: ridge on {len(mf.FEATS)} features "
           "(strategy/ml_filter.py). Costs include standard commission. Selection is post-hoc (a skipped trade does not "
           "free the slot). PAPER ONLY: execution/paper_ml_filter.py scores the live trades.", "",
           "## Out-of-sample result", "", _md(summary), "", "## Last 3 years, inside the live rule", "", _md(recent), "",
           "## Per market, inside the live rule", "", _md(per), "", "## Score fifths (1 = lowest score), all signals", "",
           _md(quint), "", "## Is it luck?", "", _md(sig), "", "## Final model (all signals), standardised coefficients", "",
           ", ".join(f"{k} {v:+.3f}" for k, v in coefs), f"; threshold {final['threshold']:+.4f}", ""]
    text = "\n".join(out)
    (ROOT / "ML_FILTER_STUDY.md").write_text(text, encoding="utf-8")
    print(text)
    if a.freeze:
        final["trained_to"], final["signals"] = str(T.entry.max().date()), len(T)
        mf.MODEL_PATH.write_text(json.dumps(final, indent=1), encoding="utf-8")
        print(f"model written: {mf.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
