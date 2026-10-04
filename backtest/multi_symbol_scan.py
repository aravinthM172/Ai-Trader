"""
Multi-symbol scan: the FROZEN live rule (momentum_rsi_mtf, H1, 2 ATR stop / 3 ATR target)
on every tradable Valetax symbol, each with its own broker spread, contract size,
minimum stop distance and swap.  Nothing is tuned per symbol.

    python -m backtest.multi_symbol_scan            # download (cached) + scan
    python -m backtest.multi_symbol_scan --offline  # cached data only

PRE-REGISTERED PASS RULE per symbol (fixed before the first run):
  - expectancy_R after swap > 0 on the combined sample AND on the FINAL (last 15 %) split
  - walk-forward >= 6/8 folds positive
  - expectancy_R > 0 with DOUBLE the spread (cost stress)
  - >= 200 trades
  - deflated Sharpe (per-trade, N = number of symbols scanned) >= 0.95

Writes reports/multi_symbol_scan.json and MULTI_SYMBOL_SCAN.md.  Read-only on MT5; no orders.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from backtest import btc_lab as lab
from backtest import btc_strategies as strat
from backtest.research_strategies import LIVE_MOMENTUM
from backtest.swap_sensitivity import nights_held

log = get_logger("multi.scan")
CACHE = ROOT / "data" / "mt5_H1"
REPORTS = ROOT / "reports"
RISK = lab.RiskCfg(initial_balance=100_000.0)       # large nominal balance: min-lot never binds; R is scale-free
SLIP_FRAC = 0.25                                     # slippage per side as a fraction of the spread
MIN_TRADES = 200
_N = NormalDist()


# -- data / specs ------------------------------------------------------------
def fetch(gw, symbol: str, bars: int = 100_000) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{symbol}.csv"
    d = gw.get_rates(symbol, "H1", bars)
    if d is None or d.empty:
        return None
    d.to_csv(p, index=False)
    return d


def load(symbol: str) -> pd.DataFrame | None:
    p = CACHE / f"{symbol}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def spec_of(info) -> dict:
    value = float(info.trade_tick_value) / float(info.trade_tick_size) if info.trade_tick_size else 0.0
    from tools.fetch_swap import annualise
    return dict(symbol=info.name, point=float(info.point), value_per_unit_per_lot=value,
                volume_min=float(info.volume_min), volume_step=float(info.volume_step),
                volume_max=float(info.volume_max), stops_level=float(info.trade_stops_level) * float(info.point),
                spread_now=float(info.spread) * float(info.point),
                swap_long=annualise(info, float(info.swap_long)), swap_short=annualise(info, float(info.swap_short)),
                path=info.path)


# -- evaluation ----------------------------------------------------------------
def deflated_sharpe(R: np.ndarray, n_trials: int) -> float:
    T = len(R)
    sd = R.std(ddof=1)
    if T < 30 or sd <= 0:
        return 0.0
    sr = R.mean() / sd
    g = 0.5772156649
    sr0 = np.sqrt(1 / (T - 1)) * ((1 - g) * _N.inv_cdf(1 - 1 / n_trials) + g * _N.inv_cdf(1 - 1 / (n_trials * np.e)))
    z = (R - R.mean()) / R.std()
    s, k = float((z ** 3).mean()), float((z ** 4).mean())
    den = 1 - s * sr + (k - 1) / 4 * sr ** 2
    return float(_N.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(den))) if den > 0 else 0.0


def swap_R(trades, df, atr, sp) -> np.ndarray:
    times = df["time"]
    n = nights_held(times.iloc[trades["entry_i"]].reset_index(drop=True),
                    times.iloc[trades["exit_i"]].reset_index(drop=True))
    rate = np.where(trades["dir"] > 0, sp["swap_long"] or 0.0, sp["swap_short"] or 0.0)   # negative = cost
    stop = RISK.sl_atr_mult * atr[trades["entry_i"] - 1]
    return np.nan_to_num(trades["entry_px"] * rate / 365.0 * n / np.where(stop > 0, stop, np.nan))


def costs_contract(df: pd.DataFrame, sp: dict) -> tuple[lab.Costs, lab.Contract, float]:
    """Broker-realistic costs (median recorded spread, floor 0.5 x current) and contract spec."""
    spread = float(np.median(df["spread"])) * sp["point"] if "spread" in df and df["spread"].median() > 0 else sp["spread_now"]
    spread = max(spread, sp["spread_now"] * 0.5)
    costs = lab.Costs(spread=spread, slippage_per_side=SLIP_FRAC * spread)
    contract = lab.Contract(value_per_unit_per_lot=sp["value_per_unit_per_lot"], volume_min=sp["volume_min"],
                            volume_step=sp["volume_step"], volume_max=sp["volume_max"],
                            broker_stops_level=sp["stops_level"], symbol=sp["symbol"])
    return costs, contract, spread


def symbol_trades(df: pd.DataFrame, sp: dict) -> pd.DataFrame:
    """Per-trade table (times, direction, R after swap, stopped flag) for the frozen rule."""
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    px = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    costs, contract, _ = costs_contract(df, sp)
    ent, wu = strat.build_momentum(c, h, l, **LIVE_MOMENTUM)
    atr = strat.base_atr(h, l, c, 14)
    t = lab.run_backtest(px, atr, ent, wu + 30, len(c), wu, costs, RISK, contract)["trades"]
    R = np.where(t["risk"] > 0, t["pnl"] / t["risk"], 0.0) + swap_R(t, df, atr, sp)
    return pd.DataFrame(dict(symbol=sp["symbol"], group=(sp["path"] or "").split("\\")[0],
                             entry_time=df["time"].iloc[t["entry_i"]].to_numpy(),
                             exit_time=df["time"].iloc[t["exit_i"]].to_numpy(),
                             direction=t["dir"], R=R, stopped=t["reason"] == 1))


def scan_symbol(df: pd.DataFrame, sp: dict, n_trials: int) -> dict:
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    px = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    costs, contract, spread = costs_contract(df, sp)
    ent, wu = strat.build_momentum(c, h, l, **LIVE_MOMENTUM)
    atr = strat.base_atr(h, l, c, 14)
    N = len(c)
    start = wu + 30
    comb = lab.run_backtest(px, atr, ent, start, N, wu, costs, RISK, contract)
    t = comb["trades"]
    if len(t["pnl"]) < 30:
        return dict(trades=int(len(t["pnl"])), passes=False, note="too few trades")
    R = np.where(t["risk"] > 0, t["pnl"] / t["risk"], 0.0) + swap_R(t, df, atr, sp)
    fin_i0 = int(N * 0.85)
    fin = R[t["entry_i"] >= fin_i0]
    wf = lab.walk_forward(px, atr, ent, wu, costs, RISK, n_folds=8, contract=contract)["summary"]
    stress = lab.run_backtest(px, atr, ent, start, N, wu, replace(costs, spread=2 * spread), RISK, contract)
    ts = stress["trades"]
    R2 = np.where(ts["risk"] > 0, ts["pnl"] / ts["risk"], 0.0) + swap_R(ts, df, atr, sp)
    dsr = deflated_sharpe(R, n_trials)
    years = (df["time"].iloc[-1] - df["time"].iloc[start]).days / 365.25
    checks = dict(exp_pos=R.mean() > 0, final_pos=len(fin) > 0 and fin.mean() > 0,
                  wf_6of8=(wf.get("folds_positive_expectancy") or 0) >= 6,
                  stress_pos=len(R2) > 0 and R2.mean() > 0, min_trades=len(R) >= MIN_TRADES, dsr_95=dsr >= 0.95)
    return dict(
        group=(sp["path"] or "").split("\\")[0], years=round(years, 1), trades=int(len(R)),
        trades_per_month=round(len(R) / max(years * 12, 1e-9), 1),
        spread=round(spread, 6), swap_long=sp["swap_long"], swap_short=sp["swap_short"],
        expectancy_R=round(float(R.mean()), 4), expectancy_R_no_swap=comb["summary"].get("expectancy_R"),
        final_expectancy_R=round(float(fin.mean()), 4) if len(fin) else None,
        wf_positive=wf.get("folds_positive_expectancy"), wf_worst=wf.get("min_expectancy_R"),
        stress_expectancy_R=round(float(R2.mean()), 4) if len(R2) else None,
        profit_factor=comb["summary"].get("profit_factor"), deflated_sharpe=round(dsr, 3),
        checks=checks, passes=all(checks.values()),
        _daily=pd.Series(R, index=df["time"].iloc[t["exit_i"]].dt.floor("D").to_numpy()).groupby(level=0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    if not gw.connect():
        raise SystemExit("MT5 connection failed (specs/swaps come from the terminal)")
    try:
        mt5 = gw.raw()
        infos = [s for s in (mt5.symbols_get() or []) if s.trade_mode != 0]
        specs = {s.name: spec_of(s) for s in infos}
        data = {}
        for name in specs:
            d = load(name) if a.offline else None
            if d is None:
                fetch(gw, name)
                d = load(name)
            if d is not None and len(d) > 3000:
                data[name] = d
            log.info("%-12s %s bars", name, None if d is None else len(d))
    finally:
        gw.shutdown()

    n = len(data)
    res = {}
    for name, d in data.items():
        try:
            res[name] = scan_symbol(d, specs[name], n)
        except Exception as e:
            log.exception("%s failed: %s", name, e)
            res[name] = dict(passes=False, note=f"error: {e}")
        r = res[name]
        log.info("%-12s trades=%s expR=%s final=%s WF=%s/8 stress=%s DSR=%s PASS=%s", name, r.get("trades"),
                 r.get("expectancy_R"), r.get("final_expectancy_R"), r.get("wf_positive"),
                 r.get("stress_expectancy_R"), r.get("deflated_sharpe"), r.get("passes"))

    passers = [k for k, r in res.items() if r.get("passes")]
    daily = pd.concat({k: res[k]["_daily"] for k in passers}, axis=1, sort=True).fillna(0.0) if passers else pd.DataFrame()
    corr = daily.corr().round(2) if len(passers) > 1 else pd.DataFrame()
    for r in res.values():
        r.pop("_daily", None)
    port = None
    if len(passers) > 0:
        p = daily.sum(axis=1)                                 # equal risk (1 R unit) per symbol trade
        port = dict(symbols=len(passers), trades_per_month=round(sum(res[k]["trades_per_month"] for k in passers), 1),
                    mean_daily_R=round(float(p.mean()), 4),
                    worst_day_R=round(float(p.min()), 2),
                    avg_pairwise_corr=round(float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()), 3)
                    if len(passers) > 1 else None)
    rep = dict(meta=dict(n_symbols_scanned=n, rule="pre-registered in module docstring", strategy="momentum_rsi_mtf (frozen)",
                         slippage_per_side=f"{SLIP_FRAC} x spread", runtime_s=round(time.perf_counter() - t0, 1)),
               passers=passers, portfolio=port, correlation=corr.to_dict() if len(corr) else {}, results=res)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "multi_symbol_scan.json").write_text(json.dumps(rep, indent=2, default=str))
    (ROOT / "MULTI_SYMBOL_SCAN.md").write_text(_md(rep), encoding="utf-8")
    print(_md(rep))
    return 0


def _md(rep) -> str:
    L = ["# Multi-symbol scan -- frozen momentum_rsi_mtf on every Valetax symbol", "",
         f"{rep['meta']['n_symbols_scanned']} symbols. Realistic broker spread, slippage {rep['meta']['slippage_per_side']}, "
         "swap included. Pass rule pre-registered in `backtest/multi_symbol_scan.py`.", "",
         "| symbol | group | years | trades/mo | exp R (after swap) | FINAL | WF+ | 2x spread | PF | DSR | PASS |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    rows = sorted(rep["results"].items(), key=lambda kv: -(kv[1].get("expectancy_R") or -9))
    for k, r in rows:
        L.append(f"| {k} | {r.get('group', '')} | {r.get('years', '')} | {r.get('trades_per_month', '')} | "
                 f"{r.get('expectancy_R', r.get('note', ''))} | {r.get('final_expectancy_R', '')} | {r.get('wf_positive', '')}/8 | "
                 f"{r.get('stress_expectancy_R', '')} | {r.get('profit_factor', '')} | {r.get('deflated_sharpe', '')} | "
                 f"{'**yes**' if r.get('passes') else 'no'} |")
    if rep.get("portfolio"):
        p = rep["portfolio"]
        L += ["", f"**Passing symbols ({p['symbols']}):** {', '.join(rep['passers'])}",
              f"Combined: ~{p['trades_per_month']} trades/month, average pairwise daily correlation {p['avg_pairwise_corr']}, "
              f"worst day {p['worst_day_R']} R at 1 R per trade."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
