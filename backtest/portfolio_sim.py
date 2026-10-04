"""
Portfolio simulation: ONE account trading the frozen momentum rule on many symbols at
once (nautilus-style shared account), with the portfolio guard (risk/portfolio_guard.py,
freqtrade-style protections) deciding which entries may open.

Inputs: symbols that PASSED backtest/multi_symbol_scan.py (reports/multi_symbol_scan.json),
their cached H1 data (data/mt5_H1/) and live broker specs (read-only MT5).

    python -m backtest.portfolio_sim

Compares guard configurations at the same risk per trade.  PRE-REGISTERED adoption rule:
a protection set replaces the baseline only if return / max-drawdown improves by >= 10 %
AND total return stays >= 90 % of the baseline's.

Also replays the portfolio's daily returns through a 2-step prop-firm challenge
(+10 % / +5 %, 5 % daily loss, 10 % max loss, then +4 % funded payout).

Writes reports/portfolio_sim.json and PORTFOLIO_SIM.md.  Research only -- NO LIVE ORDER.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk import portfolio_guard as pg
from backtest import multi_symbol_scan as ms

REPORTS = ROOT / "reports"
START_EQUITY = 10_000.0
RISK_PER_TRADE = 0.005

CONFIGS = {
    "caps_only": pg.GuardConfig(max_open_positions=6, max_total_open_risk=0.03, max_per_group=2),
    "caps+stoploss_guard": pg.GuardConfig(max_open_positions=6, max_total_open_risk=0.03, max_per_group=2,
                                          sl_count=3, sl_lookback_hours=48, sl_pause_hours=24),
    "caps+daily4%+dd8%": pg.GuardConfig(max_open_positions=6, max_total_open_risk=0.03, max_per_group=2,
                                        daily_loss_limit=0.04, max_drawdown_pause=0.08, dd_pause_hours=72),
    "all_protections": pg.GuardConfig(max_open_positions=6, max_total_open_risk=0.03, max_per_group=2,
                                      sl_count=3, sl_lookback_hours=48, sl_pause_hours=24,
                                      daily_loss_limit=0.04, max_drawdown_pause=0.08, dd_pause_hours=72),
}


def simulate(trades: pd.DataFrame, cfg: pg.GuardConfig, risk: float = RISK_PER_TRADE,
             start_equity: float = START_EQUITY) -> dict:
    """Event-driven: process exits before each entry, size entries off current equity."""
    tr = trades.sort_values("entry_time").reset_index(drop=True)
    st = pg.GuardState()
    eq = start_equity
    open_pos: list[tuple[pd.Timestamp, int, float]] = []        # (exit_time, row, $risk)
    curve, taken, skipped, reasons = [], [], 0, {}

    def close_until(t):
        nonlocal eq
        open_pos.sort(key=lambda x: x[0])
        while open_pos and open_pos[0][0] <= t:
            xt, i, r_usd = open_pos.pop(0)
            row = tr.iloc[i]
            pnl = float(row.R) * r_usd
            eq += pnl
            pg.on_close(cfg, st, symbol=row.symbol, now=xt.to_pydatetime(), pnl=pnl, equity_after=eq,
                        stopped=bool(row.stopped))
            curve.append((xt, eq))

    for i, row in tr.iterrows():
        et = pd.Timestamp(row.entry_time)
        close_until(et)
        ok, why = pg.allow_entry(cfg, st, symbol=row.symbol, group=row.group, risk_frac=risk,
                                 now=et.to_pydatetime(), equity=eq)
        if not ok:
            skipped += 1
            k = why.split(" ")[0] if not why.startswith("paused") else "paused"
            reasons[k] = reasons.get(k, 0) + 1
            continue
        pg.on_open(st, symbol=row.symbol, group=row.group, risk_frac=risk)
        open_pos.append((pd.Timestamp(row.exit_time), i, risk * eq))
        taken.append(i)
    close_until(pd.Timestamp.max.tz_localize("UTC"))

    c = pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")["equity"]
    daily = c.resample("1D").last().ffill()
    daily = pd.concat([pd.Series([start_equity], index=[daily.index[0] - pd.Timedelta(days=1)]), daily])
    peak = daily.cummax()
    dd = (daily - peak) / peak
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    total = daily.iloc[-1] / start_equity - 1
    monthly = daily.resample("ME").last().pct_change().dropna()
    under = (dd < 0).astype(int)
    longest = int(under.groupby((under != under.shift()).cumsum()).sum().max())
    t_taken = tr.iloc[taken]
    return dict(
        trades_taken=len(taken), trades_skipped=skipped, skip_reasons=reasons,
        trades_per_month=round(len(taken) / max(years * 12, 1e-9), 1),
        mean_R_taken=round(float(t_taken.R.mean()), 4) if len(t_taken) else None,
        total_return_pct=round(100 * total, 1),
        cagr_pct=round(100 * ((1 + total) ** (1 / years) - 1), 1) if years > 0 and total > -1 else None,
        max_drawdown_pct=round(100 * float(dd.min()), 1),
        longest_underwater_days=longest,
        return_over_dd=round(total / abs(dd.min()), 2) if dd.min() < 0 else None,
        months_positive_pct=round(100 * float((monthly > 0).mean()), 1),
        worst_month_pct=round(100 * float(monthly.min()), 1),
        years=round(years, 1),
        per_symbol_R={k: round(float(v), 1) for k, v in t_taken.groupby("symbol").R.sum().sort_values(ascending=False).items()},
        _daily_ret=daily.pct_change().dropna(),
    )


def prop_challenge(daily_ret: pd.Series) -> dict:
    """Rolling-start replay: P1 +10 %, P2 +5 %, funded +4 %; 5 % daily / 10 % static max loss."""
    r = daily_ret.to_numpy()

    def run(s, target):
        eq = 1.0
        for k in range(s, len(r)):
            day_start = eq
            eq *= 1 + r[k]
            if eq / day_start - 1 <= -0.05 or eq - 1 <= -0.10:
                return False, k - s + 1
            if eq - 1 >= target:
                return True, k - s + 1
        return None, len(r) - s

    out = {}
    for name, tg in (("phase1_10pct", 0.10), ("phase2_5pct", 0.05), ("funded_4pct", 0.04)):
        res = [run(s, tg) for s in range(0, max(1, len(r) - 30), 3)]
        done = [x for x in res if x[0] is not None]
        days = [x[1] for x in done if x[0]]
        out[name] = dict(pass_rate=round(float(np.mean([x[0] for x in done])), 3) if done else None,
                         median_days=int(np.median(days)) if days else None)
    vals = [out[k]["pass_rate"] or 0 for k in out]
    out["all_three"] = round(float(np.prod(vals)), 3)
    return out


def main() -> int:
    scan = json.loads((REPORTS / "multi_symbol_scan.json").read_text())
    passers = scan.get("passers", [])
    if not passers:
        raise SystemExit("no passing symbols in reports/multi_symbol_scan.json")
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        raise SystemExit("MT5 connection failed (broker specs needed)")
    try:
        mt5 = gw.raw()
        specs = {s: ms.spec_of(mt5.symbol_info(s)) for s in passers}
    finally:
        gw.shutdown()
    trades = pd.concat([ms.symbol_trades(ms.load(s), specs[s]) for s in passers], ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    # reference distribution for the multi-symbol edge monitor (live_multi excludes BTCUSD.vx)
    ref = trades[trades.symbol != "BTCUSD.vx"].sort_values("exit_time")
    ref.rename(columns={"R": "r_multiple"})[["symbol", "exit_time", "r_multiple"]].to_csv(
        REPORTS / "multi_reference_trades.csv", index=False)

    res = {}
    for name, cfg in CONFIGS.items():
        r = simulate(trades, cfg)
        r["prop_challenge"] = prop_challenge(r.pop("_daily_ret"))
        r["config"] = asdict(cfg)
        res[name] = r
        print(f"{name:22} ret {r['total_return_pct']:>7}%  CAGR {r['cagr_pct']}%  maxDD {r['max_drawdown_pct']}%  "
              f"ret/DD {r['return_over_dd']}  trades/mo {r['trades_per_month']}  prop all-3 {r['prop_challenge']['all_three']}")

    base = res["caps_only"]
    for name, r in res.items():
        if name == "caps_only":
            continue
        better = (r["return_over_dd"] or 0) >= 1.10 * (base["return_over_dd"] or 0) and \
                 r["total_return_pct"] >= 0.90 * base["total_return_pct"]
        r["adopt_over_baseline"] = bool(better)
    rep = dict(meta=dict(symbols=passers, start_equity=START_EQUITY, risk_per_trade=RISK_PER_TRADE,
                         rule="pre-registered in module docstring", swap_included=True), results=res)
    (REPORTS / "portfolio_sim.json").write_text(json.dumps(rep, indent=2, default=str))
    (ROOT / "PORTFOLIO_SIM.md").write_text(_md(rep), encoding="utf-8")
    print(_md(rep))
    return 0


def _md(rep) -> str:
    m = rep["meta"]
    L = ["# Portfolio simulation -- frozen momentum on all passing symbols, one account", "",
         f"Symbols ({len(m['symbols'])}): {', '.join(m['symbols'])}", "",
         f"Start ${m['start_equity']:,.0f}, risk {m['risk_per_trade']:.1%} of equity per trade, swap included.", "",
         "| guard config | trades/mo | total return | CAGR | max DD | longest underwater | ret/DD | months + | worst month | prop pass (all 3) | adopt? |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for n, r in rep["results"].items():
        L.append(f"| {n} | {r['trades_per_month']} | {r['total_return_pct']}% | {r['cagr_pct']}% | {r['max_drawdown_pct']}% | "
                 f"{r['longest_underwater_days']} d | {r['return_over_dd']} | {r['months_positive_pct']}% | {r['worst_month_pct']}% | "
                 f"{r['prop_challenge']['all_three']:.0%} | {'baseline' if n == 'caps_only' else ('yes' if r.get('adopt_over_baseline') else 'no')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
