"""
Calendar + news study: how every Valetax market moves by weekday, month, turn of month and around
FOMC / NFP, on 20 years of daily data (2005 -> today).

    python -m backtest.calendar_news_study

PRE-REGISTERED (fixed before the first run):
  Return       r_t = ln(close_t / close_t-1); z_t = r_t / (stdev of r over the previous 60 days).
  Effects      weekday (Mon..Fri), month (Jan..Dec), turn of month (last 1 + first 3 trading days),
               FOMC day -1 / 0 / +1, NFP day 0 / +1 (event date in New York time; the day the
               bar's date equals it).  Each effect = mean z on its days vs all other days.
  DISCOVERY    2005-01-01 .. 2018-12-31: a (symbol, effect) is a candidate if |t| >= 3.
  VALIDATION   2019-01-01 .. today: confirmed if the mean has the SAME sign and |t| >= 2.
  Volatility   the same split for |z| (event days are expected to be MORE volatile).
  Tradeable    mean |return| of the effect on validation days vs the broker round-trip cost.
Chance check: with N tests at |t| >= 3 about 0.27 % pass by luck; the report prints how many
candidates are expected by chance next to how many were found.

Writes reports/calendar_news_study.json and CALENDAR_NEWS_STUDY.md.  Research only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import strategy_round3_daily as r3
from news import history as nh

REPORTS = ROOT / "reports"
SPLIT = pd.Timestamp("2019-01-01")
T_DISC, T_VAL = 3.0, 2.0
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MO = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_all() -> dict[str, pd.DataFrame]:
    specs = json.loads(r3.SPECS.read_text())
    data = {}
    for sym in sorted(set(r3.YAHOO) | r3.BROKER_PRIMARY):
        d = r3.broker_d1(None, sym) if sym in r3.BROKER_PRIMARY else r3.yahoo(r3.YAHOO[sym], True)
        if d is not None and len(d) / max((d.index[-1] - d.index[0]).days / 365.25, 1e-9) >= r3.MIN_BARS_PER_YEAR:
            data[sym] = d
    data["XAUEUR.vx"] = r3.synthetic(data["XAUUSD.vx"], data["EURUSD.vx"])
    return {k: v for k, v in data.items() if k in specs}


def effect_masks(idx: pd.DatetimeIndex, fomc: set, nfp: set) -> dict[str, np.ndarray]:
    m = {f"weekday_{WD[i]}": idx.weekday == i for i in range(5)}
    m |= {f"month_{MO[i]}": idx.month == i + 1 for i in range(12)}
    mon = idx.month
    last = np.r_[mon[1:] != mon[:-1], False]
    first = np.r_[True, mon[1:] != mon[:-1]]
    k = np.zeros(len(idx), bool)
    for j in np.flatnonzero(first):
        k[j:j + 3] = True                       # first 3 trading days
    m["turn_of_month"] = last | k
    days = idx.normalize()
    pos = {d: i for i, d in enumerate(days)}
    for name, dates, offs in (("FOMC", fomc, (-1, 0, 1)), ("NFP", nfp, (0, 1))):
        hit = [pos[d] for d in dates if d in pos]
        for o in offs:
            a = np.zeros(len(idx), bool)
            ii = np.array(hit) + o
            a[ii[(ii >= 0) & (ii < len(idx))]] = True
            m[f"{name}_day{o:+d}"] = a
    return m


def _t(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan, len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def study_symbol(d: pd.DataFrame, fomc: set, nfp: set, cost_frac: float) -> list[dict]:
    r = np.log(d.close).diff()
    vol = r.rolling(60, min_periods=40).std().shift(1)
    z = (r / vol).to_numpy()
    disc = (d.index < SPLIT)
    rows = []
    for name, mask in effect_masks(d.index, fomc, nfp).items():
        row = dict(effect=name)
        for part, sel in (("disc", disc), ("val", ~disc)):
            on, off = z[mask & sel], z[~mask & sel]
            mu, t, n = _t(on)
            mu_off = np.nanmean(off) if len(off) else np.nan
            # difference vs other days (Welch t)
            on_, off_ = on[np.isfinite(on)], off[np.isfinite(off)]
            if len(on_) >= 20 and len(off_) >= 20:
                diff = on_.mean() - off_.mean()
                td = diff / np.sqrt(on_.var(ddof=1) / len(on_) + off_.var(ddof=1) / len(off_))
            else:
                diff = td = np.nan
            a_on, a_off = np.abs(on_), np.abs(off_)
            vr = a_on.mean() / a_off.mean() if len(a_on) >= 20 and len(a_off) else np.nan
            row |= {f"{part}_n": n, f"{part}_mean_z": mu, f"{part}_diff_z": diff, f"{part}_t": td, f"{part}_vol_ratio": vr}
        val_ret = np.abs(r.to_numpy()[mask & ~disc])
        row["val_mean_abs_ret_bp"] = float(np.nanmean(val_ret) * 1e4) if len(val_ret) else np.nan
        row["val_signed_ret_bp"] = float(np.nanmean(r.to_numpy()[mask & ~disc]) * 1e4) if mask[~disc].any() else np.nan
        row["cost_bp"] = cost_frac * 1e4 * (1 + 2 * r3.SLIP_FRAC)
        row["candidate"] = bool(np.isfinite(row["disc_t"]) and abs(row["disc_t"]) >= T_DISC)
        row["confirmed"] = bool(row["candidate"] and np.isfinite(row["val_t"]) and np.sign(row["val_t"]) == np.sign(row["disc_t"])
                                and abs(row["val_t"]) >= T_VAL)
        rows.append(row)
    return rows


def main() -> int:
    specs = json.loads(r3.SPECS.read_text())
    if not nh.OUT.exists():
        nh.build()
    ev = nh.load()
    ny = ev.assign(d=ev.time_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    fomc, nfp = set(ny[ny.event == "FOMC"].d), set(ny[ny.event == "NFP"].d)
    data = load_all()
    rows = []
    for sym, d in data.items():
        sp = specs[sym]
        sp.setdefault("price", float(d.close.iloc[-1]))
        for row in study_symbol(d, fomc, nfp, r3.broker_spread_frac(sp)):
            rows.append(dict(symbol=sym, group=(sp["path"] or "").split("\\")[0]) | row)
    df = pd.DataFrame(rows)
    n_tests = int(df.disc_t.notna().sum())
    rep = dict(meta=dict(symbols=len(data), tests=n_tests, expected_by_chance=round(n_tests * 0.0027, 1),
                         candidates=int(df.candidate.sum()), confirmed=int(df.confirmed.sum()),
                         split=str(SPLIT.date()), rule="pre-registered in module docstring"),
               rows=df.round(4).to_dict(orient="records"))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "calendar_news_study.json").write_text(json.dumps(rep, indent=1, default=str))
    md = _md(df, rep["meta"])
    (ROOT / "CALENDAR_NEWS_STUDY.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(df: pd.DataFrame, m: dict) -> str:
    L = ["# Calendar + news study -- 20 years of daily data, every Valetax market", "",
         f"{m['symbols']} symbols, {m['tests']} (symbol, effect) tests. Discovery 2005-2018 (|t| >= {T_DISC}), "
         f"validation {m['split']}..today (same sign, |t| >= {T_VAL}). Pre-registered in `backtest/calendar_news_study.py`.", "",
         f"**Direction candidates: {m['candidates']}** (about {m['expected_by_chance']} expected by pure chance) -- "
         f"**confirmed on unseen years: {m['confirmed']}**", ""]
    c = df[df.candidate].sort_values("disc_t", key=np.abs, ascending=False)
    if len(c):
        L += ["## Direction effects found in discovery", "",
              "| symbol | effect | disc t | disc mean z | val t | val mean z | val avg return (bp) | cost (bp) | confirmed |",
              "|---|---|--:|--:|--:|--:|--:|--:|:-:|"]
        for _, r in c.iterrows():
            L.append(f"| {r.symbol} | {r.effect} | {r.disc_t:.2f} | {r.disc_diff_z:+.3f} | {r.val_t:.2f} | {r.val_diff_z:+.3f} | "
                     f"{r.val_signed_ret_bp:+.1f} | {r.cost_bp:.1f} | {'**yes**' if r.confirmed else 'no'} |")
    L += ["", "## Volatility on news days (|move| vs a normal day; 1.00 = no difference)", "",
          "| group | FOMC -1 | FOMC 0 | FOMC +1 | NFP 0 | NFP +1 |", "|---|--:|--:|--:|--:|--:|"]
    v = df[df.effect.str.startswith(("FOMC", "NFP"))].pivot_table(index="group", columns="effect", values="val_vol_ratio")
    for g, r in v.iterrows():
        L.append(f"| {g} | " + " | ".join(f"{r.get(k, np.nan):.2f}" for k in
                                          ("FOMC_day-1", "FOMC_day+0", "FOMC_day+1", "NFP_day+0", "NFP_day+1")) + " |")
    L += ["", "## Weekday / turn-of-month direction by group (validation years, mean z vs other days; t in brackets)", "",
          "| group | Mon | Tue | Wed | Thu | Fri | turn of month |", "|---|--:|--:|--:|--:|--:|--:|"]
    for g, gd in df.groupby("group"):
        cells = []
        for e in [f"weekday_{w}" for w in WD] + ["turn_of_month"]:
            x = gd[gd.effect == e]
            cells.append(f"{x.val_diff_z.mean():+.3f} ({x.val_t.mean():+.1f})")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
