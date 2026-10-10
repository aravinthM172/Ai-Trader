"""
Every H1 strategy of rounds 5 and 6, plus the live momentum, market by market.  (2026-10-10)

    python -m backtest.strategy_by_market             -> STRATEGY_BY_MARKET.md
    python -m backtest.strategy_by_market --ledger    same, and record the GBPUSD tests (new market)

Markets: BTCUSD, ETHUSD, NDX100, XAUUSD, GER40, EURUSD, GBPUSD.  Same rules, data and costs as
backtest/strategy_round5.py and strategy_round6.py (GBPUSD: FundingPips H1, full dense history).
Verdict per (strategy, market):
  works  mean R > 0 on full / first 70 % / last 30 %, >= 6 of 8 folds, > 0 at 2x costs, >= 100 trades, t >= 2.75
  weak   mean R > 0 and t >= 1.5
  no     everything else
Read-only; no orders.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6

ROOT = Path(__file__).resolve().parents[1]
MARKETS = ["BTCUSD", "ETHUSD", "NDX100", "XAUUSD", "GER40", "EURUSD", "GBPUSD"]
NAMES = {"momentum": "Live momentum (2 ATR stop, 6 ATR target)", "momentum_trend": "Live momentum + daily EMA200 filter",
         "momentum_trail": "Live momentum, 3 ATR trailing stop, no target", "vbo_day": "Volatility breakout, next-day exit",
         "vbo_trail": "Volatility breakout, daily trailing stop", "nr7_vbo": "Breakout only after a quiet (NR7) day",
         "ichimoku": "Ichimoku cloud", "vp_break_d": "Volume profile breakout", "vp_fade_d": "Volume profile fade (daily)",
         "vp_fade_w": "Volume profile fade (weekly)", "asia_break": "Asian range breakout", "sweep_rev": "Liquidity sweep reversal",
         "fvg_retrace": "Fair value gap retrace", "bb_fade": "Bollinger band fade", "vwap_fade": "VWAP fade",
         "pivot_bounce": "Pivot point bounce"}


def load(sym: str) -> pd.DataFrame:
    if sym != "GBPUSD":
        return r5.load(sym)[0]
    df = fb.load_h1(sym)
    return df.assign(vol=r5.volume_of(sym)[0].reindex(df.time).to_numpy())


def trend_filter(t: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    reg = lh.daily_regime(df, True)
    day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    return t[reg.side.reindex(day, method="ffill").to_numpy() == t.dir.to_numpy()].reset_index(drop=True)


def all_trades(sym: str, df: pd.DataFrame) -> dict:
    c = r6.cost(sym)
    seg = fr.segments(df)
    cat = lambda f: pd.concat([f(p) for p in seg], ignore_index=True)
    mom = cat(lambda p: r6.momentum_piece(p, c, 6.0, 0.0))
    out = {"momentum": mom, "momentum_trend": trend_filter(mom, df),
           "momentum_trail": cat(lambda p: r6.momentum_piece(p, c, r6.INF, 3.0)),
           "vbo_trail": cat(lambda p: r6.breakout_piece(p, c, True))}
    out.update(r5.simulate(df, c))
    return out


def verdict(r: dict) -> str:
    if "checks" not in r:
        return "no"
    if r["lead"] and r["checks"]["trades_100"]:
        return "works"
    return "weak" if r["mean_R"] > 0 and (r["t"] or 0) >= 1.5 else "no"


def main() -> int:
    warnings.filterwarnings("ignore")
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + len(NAMES)
    res = {}
    for s in MARKETS:
        df = load(s)
        years = r6.years_of(df)
        for k, t in all_trades(s, df).items():
            r = r5.judge(t, years, n_trials)
            r["verdict"] = verdict(r)
            if "checks" in r:
                r["R_per_year"] = round(float(t.R.sum()) / years, 1)
                eq = t.sort_values("exit").R.cumsum()
                r["max_dd_R"] = round(float((eq.cummax() - eq).max()), 1)
            res[(s, k)] = r
        print(s, round(years, 1), {k: res[(s, k)]["verdict"] for k in NAMES}, file=sys.stderr)
    if "--ledger" in sys.argv:
        tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r3.append_ledger([dict(tested_utc=tested, name=f"bymarket_{k}_GBPUSD", params="{}", lookahead_ok=True, n_trials=n_trials,
                               deflated_sharpe=res[("GBPUSD", k)].get("dsr", ""), passes=bool(res[("GBPUSD", k)].get("passes")),
                               useful=bool(res[("GBPUSD", k)].get("passes"))) for k in NAMES])

    L = ["# Every strategy, market by market (2026-10-10)", "",
         "H1 data, FundingPips costs and swaps, rules as in `backtest/strategy_round5.py` / `strategy_round6.py`.  "
         "R = one unit of risk (the stop distance).  Verdict rule in `backtest/strategy_by_market.py`.", "",
         "## Verdict grid", "", "| strategy | " + " | ".join(MARKETS) + " |", "|---|" + ":-:|" * len(MARKETS)]
    mark = {"works": "**works**", "weak": "weak", "no": "-"}
    for k, name in NAMES.items():
        L.append(f"| {name} | " + " | ".join(mark[res[(s, k)]["verdict"]] for s in MARKETS) + " |")
    L += ["", "## Average R per trade", "", "| strategy | " + " | ".join(MARKETS) + " |", "|---|" + "--:|" * len(MARKETS)]
    for k, name in NAMES.items():
        L.append(f"| {name} | " + " | ".join(str(res[(s, k)].get("mean_R", "")) for s in MARKETS) + " |")
    for s in MARKETS:
        L += ["", f"## {s}", "",
              "| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
        for k in sorted(NAMES, key=lambda k: -(res[(s, k)].get("R_per_year") or -999)):
            r = res[(s, k)]
            if "checks" not in r:
                L.append(f"| {NAMES[k]} |  |  |  |  |  |  |  |  |  |  | no |")
                continue
            L.append(f"| {NAMES[k]} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | {r['max_dd_R']} | "
                     f"{r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {r['verdict']} |")
    md = "\n".join(L) + "\n"
    (ROOT / "STRATEGY_BY_MARKET.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
