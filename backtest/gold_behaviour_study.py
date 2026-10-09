"""
How does XAUUSD actually move?  A descriptive study to design a gold-specific strategy from.

    python -m backtest.gold_behaviour_study

Data: FundingPips XAUUSD H1 2010- (data/gold, tools/collect_gold_data), daily macro series.
Every effect is measured on DISCOVERY 2010-2017 and CONFIRMATION 2018-today separately; an effect
counts as real only if it has the same sign in both halves and |t| >= 2 in each (no tuning here --
this study only describes; strategies built from it are tested separately with the honest
simulator in backtest/full_reassessment.py).

Sections
  1 volatility and direction by hour (UTC) and session
  2 day of week
  3 trend or mean reversion?  autocorrelation of returns at 1h, 4h, 1d, 1w; variance ratios
  4 breakouts: previous-day high/low, Asian range broken in London -- does price follow through?
  5 news: FOMC / NFP hours -- volatility and direction
  6 macro drivers (daily): same-day and next-day links with DXY, 10y real yield, silver, VIX; COT
  7 the live momentum's trades (honest simulator, TP 3 and 6 ATR) split by entry session,
    volatility regime, DXY trend, real-yield trend, daily trend
Writes reports/gold_behaviour.json and GOLD_BEHAVIOUR.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "data" / "gold"
SPLIT = pd.Timestamp("2018-01-01", tz="UTC")
SESSIONS = [("Asia 00-07", 0, 7), ("London 07-12", 7, 12), ("London+NY 12-16", 12, 16),
            ("NY 16-21", 16, 21), ("Rollover 21-24", 21, 24)]


def h1() -> pd.DataFrame:
    d = pd.read_csv(G / "fp_XAUUSD_H1.csv")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d[d.time >= "2010-01-01"].reset_index(drop=True)
    d["ret"] = np.log(d.close / d.close.shift(1))
    d.loc[d.time.diff() > pd.Timedelta(hours=1), "ret"] = np.nan        # no returns across gaps/weekends
    d["absret"] = d.ret.abs()
    d["hour"], d["dow"] = d.time.dt.hour, d.time.dt.dayofweek
    return d


def daily_macro() -> pd.DataFrame:
    def rd(name, col, date="date"):
        x = pd.read_csv(G / name)
        x[date] = pd.to_datetime(x[date]).dt.tz_localize(None).dt.normalize()
        return x.set_index(date)[col]
    gold = rd("yahoo_gold_futures_D1.csv", "close").rename("gold")
    m = pd.concat([gold, rd("yahoo_dxy_D1.csv", "close").rename("dxy"),
                   rd("yahoo_silver_futures_D1.csv", "close").rename("silver"),
                   rd("yahoo_vix_D1.csv", "close").rename("vix"),
                   rd("fred_real_yield_10y.csv", "real_yield_10y").rename("ry"),
                   rd("fred_yield_10y.csv", "yield_10y").rename("y10")], axis=1).sort_index()
    m = m[m.index >= "2010-01-01"].ffill().dropna()
    out = pd.DataFrame(index=m.index)
    for c in ("gold", "dxy", "silver"):
        out[c] = np.log(m[c]).diff()
    out["vix"] = m.vix.diff()
    out["ry"], out["y10"] = m.ry.diff(), m.y10.diff()                     # yield changes in % points
    out["ry_level"], out["vix_level"] = m.ry, m.vix
    return out.dropna()


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def both_halves(df: pd.DataFrame, col: str, time_col: str = "time") -> dict:
    a, b = df[df[time_col] < SPLIT][col], df[df[time_col] >= SPLIT][col]
    ta, tb = tstat(a), tstat(b)
    return dict(disc_mean=a.mean(), disc_t=round(ta, 2), conf_mean=b.mean(), conf_t=round(tb, 2),
                real=bool(np.sign(ta) == np.sign(tb) and abs(ta) >= 2 and abs(tb) >= 2))


# -- sections --------------------------------------------------------------------------------
def by_hour(d):
    rows = []
    for hr, g in d.groupby("hour"):
        bh = both_halves(g, "ret")
        rows.append(dict(hour=int(hr), vol_bp=round(1e4 * g.absret.mean(), 1),
                         share_of_daily_range=None, mean_ret_bp=round(1e4 * g.ret.mean(), 2),
                         disc_t=bh["disc_t"], conf_t=bh["conf_t"], real=bh["real"]))
    tot = sum(r["vol_bp"] for r in rows)
    for r in rows:
        r["share_of_daily_range"] = round(100 * r["vol_bp"] / tot, 1)
    sess = []
    for name, a, b in SESSIONS:
        g = d[(d.hour >= a) & (d.hour < b)]
        bh = both_halves(g, "ret")
        sess.append(dict(session=name, vol_bp=round(1e4 * g.absret.mean(), 1), mean_ret_bp=round(1e4 * g.ret.mean(), 2),
                         disc_t=bh["disc_t"], conf_t=bh["conf_t"], real=bh["real"]))
    return rows, sess


def by_dow(d):
    day = d.dropna(subset=["ret"]).groupby(d.time.dt.normalize()).agg(ret=("ret", "sum"), time=("time", "first"))
    day["dow"] = day.time.dt.dayofweek
    out = []
    for k, g in day.groupby("dow"):
        bh = both_halves(g, "ret")
        out.append(dict(day=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k], n=len(g),
                        mean_ret_bp=round(1e4 * g.ret.mean(), 1), vol_bp=round(1e4 * g.ret.abs().mean(), 1),
                        disc_t=bh["disc_t"], conf_t=bh["conf_t"], real=bh["real"]))
    return out


def persistence(d):
    """Autocorrelation of non-overlapping returns: > 0 = trends continue, < 0 = they reverse."""
    s = d.set_index("time").close
    out = []
    for name, rule in (("1 hour", "1h"), ("4 hours", "4h"), ("1 day", "1D"), ("1 week", "W")):
        r = np.log(s.resample(rule).last().dropna()).diff().dropna()
        x = pd.DataFrame(dict(time=r.index[1:], p=(r.shift(1) * r).iloc[1:].to_numpy() /
                              r.var()))          # lag-1 autocorrelation contribution per period
        bh = both_halves(x, "p")
        out.append(dict(horizon=name, autocorr_disc=round(float(bh["disc_mean"]), 3), t_disc=bh["disc_t"],
                        autocorr_conf=round(float(bh["conf_mean"]), 3), t_conf=bh["conf_t"], real=bh["real"]))
    vr = {}
    r1 = d.dropna(subset=["ret"]).set_index("time").ret
    for q in (4, 24, 120):
        for tag, part in (("2010-17", r1[r1.index < SPLIT]), ("2018-26", r1[r1.index >= SPLIT])):
            rq = part.rolling(q).sum().iloc[::q].dropna()
            vr[f"VR({q}h) {tag}"] = round(float(rq.var() / (q * part.var())), 3)
    return out, vr


def breakouts(d):
    """After price takes out yesterday's high (low), where does it close the day vs the break level?"""
    day = d.assign(date=(d.time + pd.Timedelta(hours=3)).dt.normalize())          # broker day starts 21:00 UTC
    agg = day.groupby("date").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"),
                                  time=("time", "first"))
    agg["ph"], agg["pl"] = agg.high.shift(1), agg.low.shift(1)
    atr = (agg.high - agg.low).rolling(20).mean().shift(1)
    up = agg[agg.high > agg.ph]
    dn = agg[agg.low < agg.pl]
    res = {}
    for name, x, lvl, sign in (("break of yesterday's HIGH", up, "ph", 1), ("break of yesterday's LOW", dn, "pl", -1)):
        follow = sign * (x.close - x[lvl]) / atr[x.index]
        f = pd.DataFrame(dict(time=x.time, f=follow)).dropna()
        bh = both_halves(f, "f")
        res[name] = dict(days=len(f), close_beyond_pct=round(100 * float((f.f > 0).mean()), 1),
                         mean_followthrough_atr=round(float(f.f.mean()), 3), disc_t=bh["disc_t"], conf_t=bh["conf_t"],
                         real=bh["real"])
    # Asian range (00-07 UTC) broken during London (07-12): close at 16 UTC vs the break level
    rows = []
    for dt, g in d.groupby(d.time.dt.normalize()):
        a = g[g.hour < 7]
        lon = g[(g.hour >= 7) & (g.hour < 12)]
        end = g[g.hour == 16]
        if len(a) < 5 or lon.empty or end.empty:
            continue
        hi, lo, rng = a.high.max(), a.low.min(), a.high.max() - a.low.min()
        if rng <= 0:
            continue
        up_i = lon.index[lon.high > hi]
        dn_i = lon.index[lon.low < lo]
        first_up = up_i[0] if len(up_i) else None
        first_dn = dn_i[0] if len(dn_i) else None
        if first_up is not None and (first_dn is None or first_up < first_dn):
            rows.append((g.time.iloc[0], (end.close.iloc[0] - hi) / rng))
        elif first_dn is not None:
            rows.append((g.time.iloc[0], (lo - end.close.iloc[0]) / rng))
    f = pd.DataFrame(rows, columns=["time", "f"])
    bh = both_halves(f, "f")
    res["Asian range broken in London (first side)"] = dict(days=len(f), close_beyond_pct=round(100 * float((f.f > 0).mean()), 1),
                                                         mean_followthrough_atr=round(float(f.f.mean()), 3),
                                                         disc_t=bh["disc_t"], conf_t=bh["conf_t"], real=bh["real"],
                                                         note="follow-through in units of the Asian range")
    return res


def news(d):
    ev = pd.read_csv(ROOT / "data" / "events_history.csv")
    ev["time_utc"] = pd.to_datetime(ev.time_utc, utc=True).dt.floor("h")
    base = d.groupby("hour").absret.mean()
    out = {}
    for kind, g in ev.groupby("event"):
        hits = d[d.time.isin(g.time_utc)]
        after = d[d.time.isin(g.time_utc + pd.Timedelta(hours=1))]
        out[kind] = dict(events=len(hits), vol_multiple_event_hour=round(float((hits.absret / hits.hour.map(base)).mean()), 2),
                         vol_multiple_next_hour=round(float((after.absret / after.hour.map(base)).mean()), 2),
                         mean_ret_event_hour_bp=round(1e4 * float(hits.ret.mean()), 1), t=round(tstat(hits.ret), 2),
                         continuation_next_hour_pct=round(100 * float((np.sign(hits.ret.to_numpy()[:len(after)]) ==
                                                                       np.sign(after.ret.to_numpy())).mean()), 1))
    return out


def macro():
    m = daily_macro().reset_index().rename(columns={"index": "date"})
    m["time"] = pd.to_datetime(m.date).dt.tz_localize("UTC")
    out = {"same_day_correlation": {}, "next_day_prediction": {}}
    for c in ("dxy", "ry", "y10", "silver", "vix"):
        for tag, part in (("2010-17", m[m.time < SPLIT]), ("2018-26", m[m.time >= SPLIT])):
            out["same_day_correlation"][f"{c} {tag}"] = round(float(part.gold.corr(part[c])), 3)
        m["x"] = m[c].shift(1) * np.sign(1)
        # next-day: does yesterday's move in c predict today's gold?  slope t-stat via sign product
        m["p"] = np.sign(m[c].shift(1)) * m.gold
        bh = both_halves(m.dropna(subset=["p"]), "p")
        out["next_day_prediction"][c] = dict(disc_bp=round(1e4 * float(bh["disc_mean"]), 2), disc_t=bh["disc_t"],
                                             conf_bp=round(1e4 * float(bh["conf_mean"]), 2), conf_t=bh["conf_t"],
                                             real=bh["real"], meaning=f"gold return today x sign of yesterday's {c} change")
    m["p"] = np.sign(m.gold.shift(1)) * m.gold
    bh = both_halves(m.dropna(subset=["p"]), "p")
    out["next_day_prediction"]["gold itself"] = dict(disc_bp=round(1e4 * float(bh["disc_mean"]), 2), disc_t=bh["disc_t"],
                                                     conf_bp=round(1e4 * float(bh["conf_mean"]), 2), conf_t=bh["conf_t"],
                                                     real=bh["real"], meaning="1-day momentum")
    # COT: managed-money net position change vs next week's gold return
    cot = pd.read_csv(G / "cot_gold.csv")
    cot["date"] = pd.to_datetime(cot.date)
    gw = np.log(pd.read_csv(G / "yahoo_gold_futures_D1.csv", parse_dates=["date"]).set_index("date").close)
    rows = []
    for _, r in cot.iterrows():
        a, b = r.date + pd.Timedelta(days=4), r.date + pd.Timedelta(days=11)        # released Friday, trade next week
        x = gw[(gw.index >= a) & (gw.index <= b)]
        if len(x) > 2:
            rows.append((r.date, r.mm_net, x.iloc[-1] - x.iloc[0]))
    c = pd.DataFrame(rows, columns=["date", "mm_net", "fwd"])
    c["chg"] = c.mm_net.diff()
    c["time"] = c.date.dt.tz_localize("UTC")
    c["p"] = np.sign(c.chg) * c.fwd
    c["z"] = (c.mm_net - c.mm_net.rolling(52).mean()) / c.mm_net.rolling(52).std()
    c["p_extreme"] = -np.sign(c.z.where(c.z.abs() > 1.5)) * c.fwd                  # contrarian at crowded extremes
    out["cot_next_week"] = {"follow managed-money change": both_halves(c.dropna(subset=["p"]), "p"),
                            "fade managed-money extremes (|z|>1.5)": both_halves(c.dropna(subset=["p_extreme"]), "p_extreme")}
    for k in out["cot_next_week"].values():
        k["disc_mean"], k["conf_mean"] = round(1e4 * float(k["disc_mean"]), 1), round(1e4 * float(k["conf_mean"]), 1)
    return out


def momentum_context(d):
    cst = fr.costs()["XAUUSD"]
    m = daily_macro()
    dd = pd.read_csv(G / "yahoo_dxy_D1.csv", parse_dates=["date"]).set_index("date").close
    ry = pd.read_csv(G / "fred_real_yield_10y.csv", parse_dates=["date"]).set_index("date").real_yield_10y
    gd = pd.read_csv(G / "fp_XAUUSD_D1.csv")
    gd["date"] = (pd.to_datetime(gd.time, utc=True) + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    gd = gd.set_index("date").close
    known = lambda s, n: (s > s.rolling(n).mean()).shift(1).dropna()                  # as of the previous day
    ctx = {"DXY above 50d avg": known(dd, 50), "real yield above 50d avg": known(ry, 50),
           "gold above 200d avg": known(gd, 200)}
    A = base_atr(d.high.to_numpy(float), d.low.to_numpy(float), d.close.to_numpy(float), 14)
    atr_pct = pd.Series(A, index=d.time).rolling(24 * 250, min_periods=24 * 60).rank(pct=True)
    out = {}
    for ex, tp in fr.EXITS.items():
        t = fr.simulate(d[["time", "open", "high", "low", "close"]], tp, cst)
        t["hour"] = t.entry.dt.hour
        t["day"] = t.entry.dt.tz_localize(None).dt.normalize()
        rows = {}
        for name, a, b in SESSIONS:
            rows[f"entry {name}"] = t[(t.hour >= a) & (t.hour < b)]
        vp = atr_pct.reindex(t.entry - pd.Timedelta(hours=1), method="ffill").to_numpy()
        rows["volatility low third"], rows["volatility mid third"], rows["volatility high third"] = \
            t[vp < 1 / 3], t[(vp >= 1 / 3) & (vp < 2 / 3)], t[vp >= 2 / 3]
        for name, s in ctx.items():
            flag = s.reindex(t.day, method="ffill").to_numpy()
            same = np.where(name.startswith("gold"), flag == (t.dir > 0), flag != (t.dir > 0))
            rows[f"with {name.split(' above')[0]} (trade in gold-friendly direction)"] = t[same.astype(bool)]
            rows[f"against {name.split(' above')[0]}"] = t[~same.astype(bool)]
        rows["longs"], rows["shorts"] = t[t.dir > 0], t[t.dir < 0]
        res = {}
        for k, x in rows.items():
            x = x.rename(columns={"entry": "time"})
            bh = both_halves(x, "R")
            res[k] = dict(trades=len(x), mean_R=round(float(x.R.mean()), 3), disc_R=round(float(bh["disc_mean"]), 3),
                          conf_R=round(float(bh["conf_mean"]), 3), disc_t=bh["disc_t"], conf_t=bh["conf_t"])
        out[ex] = dict(all=dict(trades=len(t), mean_R=round(float(t.R.mean()), 3)), splits=res)
    return out


def main() -> int:
    d = h1()
    hours, sess = by_hour(d)
    pers, vr = persistence(d)
    rep = dict(hours=hours, sessions=sess, weekdays=by_dow(d), persistence=pers, variance_ratio=vr,
               breakouts=breakouts(d), news=news(d), macro=macro(), momentum=momentum_context(d))
    (ROOT / "reports" / "gold_behaviour.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "GOLD_BEHAVIOUR.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _tbl(rows: list[dict]) -> list[str]:
    keys = list(rows[0])
    return ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + \
           ["| " + " | ".join(("**yes**" if v is True else "no" if v is False else
                               f"{v:.3g}" if isinstance(v, float) else str(v)) for v in r.values()) + " |" for r in rows]


def _md(r) -> str:
    L = ["# How XAUUSD moves -- FundingPips H1 2010-today", "",
         "Discovery 2010-2017 vs confirmation 2018-today; **real** = same sign and |t| >= 2 in both.  "
         "Descriptive only -- nothing here is a tested strategy.", "",
         "## 1. By session (UTC)", ""] + _tbl(r["sessions"]) + ["", "By hour:", ""] + _tbl(r["hours"])
    L += ["", "## 2. Day of week", ""] + _tbl(r["weekdays"])
    L += ["", "## 3. Trend or reversal?  (lag-1 autocorrelation; > 0 = moves continue)", ""] + _tbl(r["persistence"])
    L += ["", "Variance ratios (> 1 = trending, < 1 = mean-reverting): " +
          ", ".join(f"{k} {v}" for k, v in r["variance_ratio"].items())]
    L += ["", "## 4. Breakouts", ""] + _tbl([dict(setup=k, **v) for k, v in r["breakouts"].items()])
    L += ["", "## 5. News hours (volatility vs a normal same hour)", ""] + _tbl([dict(event=k, **v) for k, v in r["news"].items()])
    L += ["", "## 6. Macro drivers (daily)", "", "Same-day correlation with gold: " +
          ", ".join(f"{k} {v}" for k, v in r["macro"]["same_day_correlation"].items()), "",
          "Does yesterday's move predict today's gold?", ""] + \
         _tbl([dict(driver=k, **v) for k, v in r["macro"]["next_day_prediction"].items()]) + \
         ["", "COT (managed money) -> next week's gold return (bp):", ""] + \
         _tbl([dict(rule=k, **{a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
               for k, v in r["macro"]["cot_next_week"].items()])
    for ex, m in r["momentum"].items():
        L += ["", f"## 7. Live momentum trades by context -- exit {ex} (all: {m['all']['trades']} trades, "
                  f"{m['all']['mean_R']} R)", ""] + _tbl([dict(context=k, **v) for k, v in m["splits"].items()])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
