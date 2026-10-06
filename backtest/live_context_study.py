"""
Live H1 momentum in context: results by weekday, 4-hour UTC block and news proximity (FOMC / NFP),
then a portfolio check of the filter that passed.  Research only.

    python -m backtest.live_context_study

PRE-REGISTERED: discovery = entries < 2022-01-01, validation = 2022+. 14 buckets (5 weekdays, 6 four-hour UTC blocks,
3 news states). A bucket is a FILTER candidate if disc Welch t (bucket vs rest) <= -2.5, val t <= -1.5 (same sign),
and dropping it raises validation mean R.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from backtest import multi_symbol_scan as ms
from news import history as nh
specs = json.loads(open("reports/round3_specs.json").read())
ev = nh.load().time_utc.sort_values().to_numpy()
LIVE = ["BTCUSD.vx", "XAUUSD.vx", "XAUEUR.vx", "DAX40.vx"]
def trades(sym):
    d = ms.load(sym); t = ms.symbol_trades(d, specs[sym])
    t["entry_time"] = pd.to_datetime(t.entry_time, utc=True); t["exit_time"] = pd.to_datetime(t.exit_time, utc=True)
    return t
T = pd.concat([trades(s) for s in LIVE], ignore_index=True)
e = T.entry_time.to_numpy(); x = T.exit_time.to_numpy()
nxt = np.searchsorted(ev, e)                       # first event at/after entry
prv = nxt - 1
hrs_to = (ev[np.minimum(nxt, len(ev) - 1)] - e) / np.timedelta64(1, "h")
hrs_since = (e - ev[np.maximum(prv, 0)]) / np.timedelta64(1, "h")
spans = ev[np.minimum(nxt, len(ev) - 1)] <= x
T["news"] = np.where(hrs_since <= 2, "entered <=2h after news", np.where(spans, "trade spans news", "no news"))
T["weekday"] = T.entry_time.dt.day_name().str[:3]
T["hour_block"] = (T.entry_time.dt.hour // 4 * 4).map(lambda h: f"{h:02d}-{h+4:02d} UTC")
T["part"] = np.where(T.entry_time < pd.Timestamp("2022-01-01", tz="UTC"), "disc", "val")
print(f"trades {len(T)} (disc {(T.part=='disc').sum()}, val {(T.part=='val').sum()}); overall mean R disc {T[T.part=='disc'].R.mean():+.3f} val {T[T.part=='val'].R.mean():+.3f}\n")
def welch(a, b):
    if len(a) < 2 or len(b) < 2: return float("nan")
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
rows = []
for col in ("weekday", "hour_block", "news"):
    for b in sorted(T[col].unique()):
        r = dict(feature=col, bucket=b)
        for p in ("disc", "val"):
            P = T[T.part == p]; on, off = P[P[col] == b].R, P[P[col] != b].R
            r[f"{p}_n"], r[f"{p}_R"], r[f"{p}_t"] = len(on), round(on.mean(), 3), round(welch(on, off), 2)
        V = T[T.part == "val"]; r["val_R_if_dropped"] = round(V[V[col] != b].R.mean(), 3)
        r["FILTER"] = bool(r["disc_t"] <= -2.5 and r["val_t"] <= -1.5 and r["val_R_if_dropped"] > V.R.mean())
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nper symbol, trades spanning news vs not (all years):")
print(T.groupby(["symbol", "news"]).R.agg(["count", "mean"]).round(3).unstack().to_string())

# -- portfolio check: does skipping the flagged hours make more money, not just a higher mean R? --
from risk import portfolio_guard as pg
from backtest import portfolio_sim as ps
T["group"] = T.symbol.map({"BTCUSD.vx": "Crypto", "XAUUSD.vx": "Metals", "XAUEUR.vx": "Metals", "DAX40.vx": "Indexes"})
cfg = pg.GuardConfig(max_open_positions=6, max_total_open_risk=0.03, max_per_group=1)
for period, P in (("validation 2022+", T[T.part == "val"]), ("all years", T)):
    for name, tr in (("live rule as is", P), ("skip entries 12-16 UTC", P[P.hour_block != "12-16 UTC"])):
        r = ps.simulate(tr, cfg); pc = ps.prop_challenge(r.pop("_daily_ret"))
        print(f"{period:17} {name:24} taken {r['trades_taken']:5} meanR {r['mean_R_taken']:+.3f}  CAGR {r['cagr_pct']:6}%  maxDD {r['max_drawdown_pct']:6}%  "
              f"worst month {r['worst_month_pct']:5}%  phase1 {pc['phase1_10pct']['pass_rate']} ({pc['phase1_10pct']['median_days']}d)  all-3 {pc['all_three']}")
