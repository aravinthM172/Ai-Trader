"""
btc_lab.py -- authoritative BTCUSD.vx backtest + validation engine.

One audited simulator; every strategy in btc_strategies.py runs through it.

Causality contract:
  * a strategy emits ``entries[i]`` in {-1, 0, +1} decided ONLY from bars <= i-1
  * the engine fills that entry at ``open[i]`` (+/- half-spread +/- slippage)
  * an open position is managed from bar i+1 onward (entry bar's own H/L never
    trigger its stop)  -- matches backtest/btc_engine_core.py POINT M
  * ATR / indicator arrays are computed once over the full series with causal
    (SMA-seeded Wilder) kernels, then sliced

Costs (all in price $):  entry pays half_spread + slippage (stop-style breakout
fill is conservative); stop exit pays half_spread + slippage; target exit pays
half_spread only; time / equity-stop exit pays half_spread + slippage.
Round-trip cost modelled by the engine == spread + 2*slippage.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

try:
    from numba import njit
except Exception:                                    # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

# Valetax BTCUSD.vx real contract (module defaults -- kept for backward compatibility)
VALUE_PER_UNIT_PER_LOT = 1.0
VOLUME_MIN = 0.01
VOLUME_STEP = 0.01
VOLUME_MAX = 50.0
BROKER_STOPS_LEVEL = 29.76
BROKER_STOP_BUFFER = 1.15


@dataclass
class Contract:
    """Instrument spec used by the simulator.  Defaults = Valetax BTCUSD.vx."""
    value_per_unit_per_lot: float = 1.0     # $ P/L per 1.0 price move per 1.0 lot
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 50.0
    broker_stops_level: float = 29.76       # $ minimum SL/TP distance
    symbol: str = "BTCUSD.vx"


BTC_CONTRACT = Contract()
XAU_CONTRACT = Contract(value_per_unit_per_lot=100.0, volume_min=0.01, volume_step=0.01,
                        volume_max=50.0, broker_stops_level=0.31, symbol="XAUUSD.vx")


@dataclass
class Costs:
    spread: float = 29.76           # $ per bar (Valetax reports this fixed)
    slippage_per_side: float = 1.0  # $
    commission_per_lot: float = 0.0 # $ round-turn

    def round_trip(self) -> float:
        return self.spread + 2.0 * self.slippage_per_side


@dataclass
class RiskCfg:
    initial_balance: float = 100.0
    risk_per_trade: float = 0.01
    max_risk_per_trade: float = 0.02
    equity_stop_frac: float = 0.30
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    min_rr: float = 1.3
    max_stop_frac_price: float = 0.05
    allow_min_lot_over_target: bool = True


@njit(cache=True)
def _simulate(o, h, l, c, atr, entries, i0, i1, warmup,
              spread, slippage, commission_per_lot,
              init_bal, risk_pt, max_risk_pt, eq_stop_frac,
              sl_mult, tp_mult, min_rr, max_stop_frac, allow_min_lot,
              value_per_unit, vmin, vstep, vmax, broker_min,
              t_ei, t_xi, t_dir, t_lots, t_epx, t_xpx, t_pnl, t_risk, t_bars, t_reason,
              eq_curve):
    bal = init_bal
    peak = init_bal
    eq_stop = init_bal * eq_stop_frac
    max_dd = 0.0
    pos = 0
    entry = sl = tp = lots = risk_px = 0.0
    ent_i = 0
    n = 0
    bars_held = 0
    n_signals = 0
    n_skip_rr = 0
    n_skip_cost = 0
    n_skip_size = 0
    hs = spread * 0.5
    start = i0 if i0 > warmup else warmup
    if start < 1:
        start = 1
    equity_stopped = 0

    for i in range(start, i1):
        # ---- manage ----
        if pos != 0:
            bars_held += 1
            ex = 0.0
            reason = -1
            if pos == 1:
                if o[i] - hs <= sl:
                    ex = o[i] - hs - slippage; reason = 1
                elif l[i] - hs <= sl:
                    ex = sl - slippage; reason = 1
                elif o[i] - hs >= tp:
                    ex = o[i] - hs; reason = 0
                elif h[i] - hs >= tp:
                    ex = tp; reason = 0
            else:
                if o[i] + hs >= sl:
                    ex = o[i] + hs + slippage; reason = 1
                elif h[i] + hs >= sl:
                    ex = sl + slippage; reason = 1
                elif o[i] + hs <= tp:
                    ex = o[i] + hs; reason = 0
                elif l[i] + hs <= tp:
                    ex = tp; reason = 0
            if reason < 0 and i == i1 - 1:
                ex = (c[i] - hs - slippage) if pos == 1 else (c[i] + hs + slippage)
                reason = 2
            if reason >= 0:
                pnl = (ex - entry) * pos * value_per_unit * lots - commission_per_lot * lots
                bal += pnl
                if n < t_pnl.shape[0]:
                    t_ei[n] = ent_i; t_xi[n] = i; t_dir[n] = pos; t_lots[n] = lots
                    t_epx[n] = entry; t_xpx[n] = ex; t_pnl[n] = pnl
                    t_risk[n] = risk_px * value_per_unit * lots
                    t_bars[n] = bars_held; t_reason[n] = reason
                n += 1
                pos = 0

        # ---- equity / DD ----
        if pos == 1:
            eq = bal + ((c[i] - hs) - entry) * value_per_unit * lots
        elif pos == -1:
            eq = bal + (entry - (c[i] + hs)) * value_per_unit * lots
        else:
            eq = bal
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
        if i - start < eq_curve.shape[0]:
            eq_curve[i - start] = eq
        if eq <= eq_stop:
            if pos != 0:
                ex = (c[i] - hs - slippage) if pos == 1 else (c[i] + hs + slippage)
                pnl = (ex - entry) * pos * value_per_unit * lots - commission_per_lot * lots
                bal += pnl
                if n < t_pnl.shape[0]:
                    t_ei[n] = ent_i; t_xi[n] = i; t_dir[n] = pos; t_lots[n] = lots
                    t_epx[n] = entry; t_xpx[n] = ex; t_pnl[n] = pnl
                    t_risk[n] = risk_px * value_per_unit * lots
                    t_bars[n] = bars_held; t_reason[n] = 3
                n += 1
                pos = 0
            equity_stopped = 1
            break

        if pos != 0:
            continue

        # ---- entry (decided from <= i-1, filled at open[i]) ----
        d = entries[i]
        if d == 0:
            continue
        a = atr[i - 1]
        if not (a > 0.0):
            continue
        n_signals += 1
        entry_fill = o[i] + hs if d == 1 else o[i] - hs
        ref = o[i] - hs if d == 1 else o[i] + hs
        sl_ref = a * sl_mult
        bmin = broker_min * BROKER_STOP_BUFFER
        if sl_ref < bmin:
            sl_ref = bmin
        sl_px = ref - d * sl_ref
        risk = entry_fill - sl_px if d == 1 else sl_px - entry_fill
        if risk <= 0:
            continue
        tp_dist = a * tp_mult
        need = risk * min_rr
        if tp_dist < need:
            tp_dist = need
        if tp_dist < bmin:
            tp_dist = bmin
        tp_px = entry_fill + d * tp_dist
        rr = (tp_dist) / risk
        if rr < min_rr - 1e-9:
            n_skip_rr += 1
            continue
        if (spread + 2.0 * slippage) / risk > 0.5:
            n_skip_cost += 1
            continue
        if risk > max_stop_frac * ((o[i] + o[i]) * 0.5):
            n_skip_cost += 1
            continue
        # sizing
        risk_d = bal * risk_pt
        loss_per_lot = risk * value_per_unit + commission_per_lot
        raw = risk_d / loss_per_lot if loss_per_lot > 0 else 0.0
        v = np.floor(raw / vstep + 1e-9) * vstep
        if v < vmin:
            if allow_min_lot > 0 and bal > 0 and (vmin * loss_per_lot / bal) <= max_risk_pt + 1e-9:
                v = vmin
            else:
                n_skip_size += 1
                continue
        if v > vmax:
            v = vmax
        pos = d
        entry = entry_fill
        sl = sl_px
        tp = tp_px
        lots = v
        risk_px = risk
        ent_i = i
        bars_held = 0

    win = 0
    for k in range(n):
        if t_pnl[k] > 0.0:
            win += 1
    return (bal, max_dd, float(n), float(win), float(n_signals),
            float(n_skip_rr), float(n_skip_cost), float(n_skip_size), float(equity_stopped))


def run_backtest(px: dict, atr: np.ndarray, entries: np.ndarray,
                 i0: int, i1: int, warmup: int, costs: Costs, risk: RiskCfg,
                 contract: Contract = BTC_CONTRACT) -> dict:
    o = np.ascontiguousarray(px["open"], float)
    h = np.ascontiguousarray(px["high"], float)
    l = np.ascontiguousarray(px["low"], float)
    c = np.ascontiguousarray(px["close"], float)
    atr = np.ascontiguousarray(atr, float)
    entries = np.ascontiguousarray(entries.astype(np.int64))
    cap = max(8, int(i1 - i0))
    z = lambda dt: np.zeros(cap, dt)
    t_ei, t_xi, t_dir = z(np.int64), z(np.int64), z(np.int64)
    t_lots, t_epx, t_xpx, t_pnl, t_risk = z(np.float64), z(np.float64), z(np.float64), z(np.float64), z(np.float64)
    t_bars, t_reason = z(np.int64), z(np.int64)
    eq = np.zeros(max(1, i1 - max(i0, warmup, 1)), np.float64)

    res = _simulate(
        o, h, l, c, atr, entries, int(i0), int(i1), int(warmup),
        costs.spread, costs.slippage_per_side, costs.commission_per_lot,
        risk.initial_balance, risk.risk_per_trade, risk.max_risk_per_trade, risk.equity_stop_frac,
        risk.sl_atr_mult, risk.tp_atr_mult, risk.min_rr, risk.max_stop_frac_price,
        1 if risk.allow_min_lot_over_target else 0,
        contract.value_per_unit_per_lot, contract.volume_min, contract.volume_step,
        contract.volume_max, contract.broker_stops_level,
        t_ei, t_xi, t_dir, t_lots, t_epx, t_xpx, t_pnl, t_risk, t_bars, t_reason, eq)

    n = int(round(res[2]))
    trades = dict(
        entry_i=t_ei[:n].copy(), exit_i=t_xi[:n].copy(), dir=t_dir[:n].copy(),
        lots=t_lots[:n].copy(), entry_px=t_epx[:n].copy(), exit_px=t_xpx[:n].copy(),
        pnl=t_pnl[:n].copy(), risk=t_risk[:n].copy(), bars=t_bars[:n].copy(), reason=t_reason[:n].copy(),
    )
    summ = metrics(trades, eq[:max(1, i1 - max(i0, warmup, 1))], i1 - max(i0, warmup, 1),
                   costs, risk, contract)
    summ.update(final_balance=round(float(res[0]), 2),
                max_dd_abs_engine=round(float(res[1]), 2),
                n_signals=int(res[4]), skipped_rr=int(res[5]),
                skipped_cost=int(res[6]), skipped_size=int(res[7]),
                equity_stopped=bool(res[8]))
    return {"summary": summ, "trades": trades}


def metrics(trades: dict, equity: np.ndarray, n_bars: int, costs: Costs, risk: RiskCfg,
            contract: Contract = BTC_CONTRACT) -> dict:
    pnl = trades["pnl"]
    if len(pnl) == 0:
        return dict(trades=0, net_pl=0.0, note="no trades")
    rk = trades["risk"]
    R = np.where(rk > 0, pnl / rk, 0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    eqc = risk.initial_balance + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[risk.initial_balance], eqc]))
    dd = peak - np.concatenate([[risk.initial_balance], eqc])
    ddp = dd / peak
    cl = mx = 0
    for x in pnl:
        cl = 0 if x > 0 else cl + 1
        mx = max(mx, cl)
    downside = R[R < 0]
    rtc = costs.round_trip()
    total_cost = float(np.sum(rtc * contract.value_per_unit_per_lot * trades["lots"]))
    gp = float(wins.sum()); gl = float(-losses.sum())
    return dict(
        trades=int(len(pnl)),
        net_pl=round(float(pnl.sum()), 2),
        return_pct=round(100.0 * float(pnl.sum()) / risk.initial_balance, 2),
        win_rate=round(float((pnl > 0).mean()), 4),
        profit_factor=round(gp / gl, 4) if gl > 0 else (None if gp == 0 else float("inf")),
        expectancy_dollar=round(float(pnl.mean()), 4),
        expectancy_R=round(float(R.mean()), 4),
        avg_win=round(float(wins.mean()), 4) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 4) if len(losses) else 0.0,
        largest_loss=round(float(pnl.min()), 4),
        largest_win=round(float(pnl.max()), 4),
        max_consecutive_losses=int(mx),
        max_drawdown_abs=round(float(dd.max()), 2),
        max_drawdown_pct=round(float(ddp.max()) * 100, 2),
        return_over_maxdd=round(float(pnl.sum() / dd.max()), 3) if dd.max() > 0 else None,
        sharpe_per_trade=round(float(R.mean() / R.std(ddof=1)), 3) if len(R) > 1 and R.std(ddof=1) > 0 else None,
        sortino_per_trade=round(float(R.mean() / downside.std(ddof=1)), 3) if len(downside) > 1 and downside.std(ddof=1) > 0 else None,
        avg_holding_bars=round(float(trades["bars"].mean()), 1),
        exposure_frac=round(float(trades["bars"].sum() / max(1, n_bars)), 4),
        transaction_cost_total=round(total_cost, 2),
        spread_cost_component=round(float(np.sum(costs.spread * contract.value_per_unit_per_lot * trades["lots"])), 2),
        slippage_cost_component=round(float(np.sum(2 * costs.slippage_per_side * contract.value_per_unit_per_lot * trades["lots"])), 2),
        cost_vs_gross_profit=round(total_cost / gp, 3) if gp > 0 else None,
        exit_reasons={0: "target", 1: "stop", 2: "time", 3: "equity_stop"},
        exit_reason_counts={int(k): int((trades["reason"] == k).sum()) for k in (0, 1, 2, 3)},
    )


# ---------------------------------------------------------------------------
def walk_forward(px, atr, entries, warmup, costs, risk, *, n_folds=8, oos_frac=0.15, contract=BTC_CONTRACT):
    N = len(px["close"])
    usable0 = warmup + 50
    step = (N - usable0) // n_folds
    folds = []
    for k in range(n_folds):
        a = usable0 + k * step
        b = a + step if k < n_folds - 1 else N
        r = run_backtest(px, atr, entries, a, b, warmup, costs, risk, contract)["summary"]
        r["fold"] = k
        r["range"] = [int(a), int(b)]
        folds.append(r)
    traded = [f for f in folds if f.get("trades", 0) > 0]
    exp = [f["expectancy_R"] for f in traded]
    nets = [f["net_pl"] for f in folds]
    total_net = sum(nets)
    best_share = (max(nets) / total_net) if total_net > 0 and max(nets) > 0 else None
    return {
        "folds": folds,
        "summary": dict(
            n_folds=n_folds, folds_traded=len(traded),
            folds_positive_expectancy=int(sum(1 for x in exp if x > 0)),
            mean_expectancy_R=round(float(np.mean(exp)), 4) if exp else None,
            min_expectancy_R=round(float(np.min(exp)), 4) if exp else None,
            total_net_pl=round(total_net, 2),
            single_fold_profit_share=round(best_share, 3) if best_share is not None else None,
            all_folds_positive=bool(exp and all(x > 0 for x in exp)),
            majority_folds_positive=bool(exp and sum(1 for x in exp if x > 0) > len(exp) / 2),
        ),
    }


def cost_sensitivity(px, atr, entries, i0, i1, warmup, risk, scenarios: dict, contract=BTC_CONTRACT):
    out = {}
    for name, c in scenarios.items():
        s = run_backtest(px, atr, entries, i0, i1, warmup, c, risk, contract)["summary"]
        out[name] = dict(spread=c.spread, slippage=c.slippage_per_side,
                         round_trip=round(c.round_trip(), 2),
                         trades=s.get("trades", 0), net_pl=s.get("net_pl", 0.0),
                         profit_factor=s.get("profit_factor"),
                         expectancy_R=s.get("expectancy_R"),
                         max_drawdown_pct=s.get("max_drawdown_pct"))
    return out


def monte_carlo(trades: dict, risk: RiskCfg, *, n=5000, seed=20260830,
                slippage_jitter_R=0.05, sim_balance=None, sim_risk_frac=None):
    """
    Bootstrap the sequence of realised R-multiples and re-simulate as a
    FIXED-FRACTIONAL account (risk ``sim_risk_frac`` of running balance per trade),
    so results are honest for the target account size regardless of what balance
    the trades were generated on.
    """
    pnl = trades["pnl"].astype(float)
    rk = trades["risk"].astype(float)
    R = np.where(rk > 0, pnl / rk, 0.0)
    if len(R) < 10:
        return {"note": f"only {len(R)} trades, MC not meaningful", "n_trades": int(len(R))}
    ib = float(sim_balance if sim_balance is not None else risk.initial_balance)
    rf = float(sim_risk_frac if sim_risk_frac is not None else risk.risk_per_trade)
    stop_level = ib * risk.equity_stop_frac
    rng = np.random.default_rng(seed)
    finals = np.empty(n)
    maxdd_abs = np.empty(n)
    maxdd_pct = np.empty(n)
    ruin = over50 = 0
    for k in range(n):
        seq_R = R[rng.integers(0, len(R), len(R))].copy()
        seq_R += rng.normal(0, slippage_jitter_R, len(seq_R))      # execution jitter
        bal = ib
        peak = ib
        mdd_abs = mdd_pct = 0.0
        blown = False
        for r in seq_R:
            bal += rf * bal * r
            if bal > peak:
                peak = bal
            d = peak - bal
            if d > mdd_abs:
                mdd_abs = d
            if peak > 0 and d / peak > mdd_pct:
                mdd_pct = d / peak
            if bal <= stop_level:
                blown = True
                break
        finals[k] = bal
        maxdd_abs[k] = mdd_abs
        maxdd_pct[k] = mdd_pct
        if blown:
            ruin += 1
        if mdd_pct > 0.5:
            over50 += 1
    return {
        "n_trials": n, "n_trades": int(len(R)),
        "sim_balance": ib, "sim_risk_frac": rf,
        "prob_negative_final": round(float((finals < ib).mean()), 4),
        "prob_drawdown_over_50pct": round(over50 / n, 4),
        "prob_ruin_equity_stop": round(ruin / n, 4),
        "median_final_balance": round(float(np.median(finals)), 2),
        "p05_final_balance": round(float(np.percentile(finals, 5)), 2),
        "p95_final_balance": round(float(np.percentile(finals, 95)), 2),
        "expected_max_drawdown_abs": round(float(np.mean(maxdd_abs)), 2),
        "p95_max_drawdown_abs": round(float(np.percentile(maxdd_abs, 95)), 2),
        "p95_max_drawdown_pct": round(float(np.percentile(maxdd_pct, 95)) * 100, 2),
        "worst_losing_streak": int(_worst_streak(R)),
    }


def _worst_streak(pnl):
    cl = mx = 0
    for x in pnl:
        cl = 0 if x > 0 else cl + 1
        mx = max(mx, cl)
    return mx
