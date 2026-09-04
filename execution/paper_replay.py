"""
MT5-FREE H1 paper replay -- for Oracle Cloud (Linux, no MetaTrader5).

Reads a candle export produced by tools/export_h1.py and replays every completed
H1 bar through the FROZEN, validated `momentum_rsi_mtf` strategy using the same
causal fill / stop / sizing logic as backtest/btc_lab.py.  Uses the REAL per-bar
Valetax spread from the export (floored at the broker minimum -- never below it,
never fabricated).  order_send() is impossible here (MT5 not imported).

Deterministic: re-run it whenever a fresh export arrives; the trade log grows.
Trades are split into IN-SAMPLE (entry <= validation cutoff) and FORWARD / OOS
(entry after).  The FORWARD subset is the actual forward test.

    python run_btc.py --paper --timeframe H1 --from-csv data/export/btcusd_vx_H1_export.csv [--paper-balance 1500]
    python run_btc.py --paper --timeframe H1 --asset XAU \
        --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500

BTC   state : state/btc_paper_H1_replay.sqlite   status: reports/btc_h1_paper_status.json
XAU   state : state/xau_paper_H1_replay.sqlite   status: reports/xau_h1_paper_status.json
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.logging_setup import get_logger
from backtest.btc_strategies import build_momentum   # numba-free (pandas/numpy only)
from strategy.btc_h1_signal import PARAMS, STRATEGY_NAME
from strategy.stops import build_stop_plan, spread_filter
from risk.sizing import _normalise_volume
from config.assets import get_asset_config
from execution import safety

log = get_logger("h1.replay", filename="btc_paper.log")

_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _AssetSpec:
    asset: str                    # "BTC" / "XAU"
    cfg_key: str                  # get_asset_config() key
    symbol: str
    slippage: float               # $ per side
    broker_min_spread: float      # $ -- spread floor, never fabricated below this
    broker_stops_level: float     # $ -- broker minimum stop distance
    value_per_unit: float         # $ P/L per 1.0 unit move per 1.0 lot
    dense_csv: str                # cutoff = last bar of this validated dataset
    db: str
    status: str
    trades_csv: str
    default_export: str
    fallback_cutoff: str


_SPECS: dict[str, _AssetSpec] = {
    "BTC": _AssetSpec(
        asset="BTC", cfg_key="BTC", symbol="BTCUSD.vx",
        slippage=1.0, broker_min_spread=29.76, broker_stops_level=29.76, value_per_unit=1.0,
        dense_csv="data/btcusd_vx_H1_dense.csv",
        db="state/btc_paper_H1_replay.sqlite",
        status="reports/btc_h1_paper_status.json",     # the filename tools/btc_h1_live_gate.py reads
        trades_csv="reports/btc_paper_replay_trades.csv",
        default_export="data/export/btcusd_vx_H1_export.csv",
        fallback_cutoff="2026-08-30T10:00:00Z"),
    "XAU": _AssetSpec(
        asset="XAU", cfg_key="XAU", symbol="XAUUSD.vx",
        slippage=0.10, broker_min_spread=0.30, broker_stops_level=0.31, value_per_unit=100.0,
        dense_csv="data/xauusd_vx_H1.csv",
        db="state/xau_paper_H1_replay.sqlite",
        status="reports/xau_h1_paper_status.json",     # the filename tools/xau_h1_live_gate.py reads
        trades_csv="reports/xau_paper_replay_trades.csv",
        default_export="data/export/xauusd_vx_H1_export.csv",
        fallback_cutoff="2026-08-28T00:00:00Z"),
}

# ---- BTC module-level defaults (kept for backward compat + test monkeypatching) ----
_SPEC = _SPECS["BTC"]
_DB = _ROOT / _SPEC.db
_STATUS = _ROOT / _SPEC.status
_TRADES_CSV = _ROOT / _SPEC.trades_csv
_DENSE = _ROOT / _SPEC.dense_csv
_DB.parent.mkdir(exist_ok=True)

SLIPPAGE = _SPEC.slippage
BROKER_MIN_SPREAD = _SPEC.broker_min_spread
VALUE_PER_UNIT = _SPEC.value_per_unit


def _spec(asset: str) -> _AssetSpec:
    try:
        return _SPECS[asset.strip().upper()]
    except KeyError:
        raise KeyError(f"unknown replay asset '{asset}' (expected BTC or XAU)")


def _validation_cutoff() -> pd.Timestamp:
    """BTC cutoff -- last bar of the validated dense H1 set.  Tests monkeypatch this."""
    if _DENSE.exists():
        d = pd.read_csv(_DENSE, usecols=["time"])
        return pd.to_datetime(d["time"].iloc[-1], utc=True)
    return pd.Timestamp(_SPEC.fallback_cutoff)


def _cutoff_for(spec: _AssetSpec) -> pd.Timestamp:
    if spec.asset == "BTC":
        return _validation_cutoff()          # honour the monkeypatch in the BTC tests
    p = _ROOT / spec.dense_csv
    if p.exists():
        d = pd.read_csv(p, usecols=["time"])
        return pd.to_datetime(d["time"].iloc[-1], utc=True)
    return pd.Timestamp(spec.fallback_cutoff)


def _conn(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY, timestamp_utc TEXT, timeframe TEXT, strategy TEXT,
        sample TEXT, direction TEXT, signal_bar_utc TEXT,
        entry REAL, exit REAL, spread REAL, atr REAL, stop REAL, target REAL,
        volume REAL, est_risk_usd REAL, pnl_usd REAL, r_multiple REAL,
        bars_held INTEGER, reason_entry TEXT, reason_exit TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ran_utc TEXT, export_utc TEXT,
        csv_last_bar TEXT, n_bars INTEGER, n_trades INTEGER, n_forward INTEGER)""")
    return c


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    meta = {}
    mp = csv_path.with_suffix(".json")
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return df, meta


def _simulate(df: pd.DataFrame, balance: float, cutoff: pd.Timestamp,
              spec: _AssetSpec = _SPECS["BTC"]) -> list[dict]:
    """Bar-stepping replay -- mirrors backtest/btc_lab._simulate fill/stop/sizing,
    but with per-bar real spread and full trade records."""
    slippage = spec.slippage
    min_spread = spec.broker_min_spread
    value_per_unit = spec.value_per_unit

    o = df.open.to_numpy(float); h = df.high.to_numpy(float)
    l = df.low.to_numpy(float); c = df.close.to_numpy(float)
    t = df["time"].to_numpy()
    if "spread" in df.columns:
        sp = np.maximum(df["spread"].to_numpy(float) * 0.01, min_spread)
    else:
        sp = np.full(len(c), min_spread)

    from strategy.btc_features import _atr
    atr, _ = _atr(h, l, c, 14)

    entries, warmup = build_momentum(c, h, l, **PARAMS)
    cfg = get_asset_config(spec.cfg_key, timeframe="H1")

    bal = balance
    eq_stop = balance * 0.30
    pos = 0
    entry = sl = tp = lots = risk_px = 0.0
    ent_i = 0
    sig_reason = ""
    max_hold = 96
    trades: list[dict] = []
    start = max(warmup + 5, 1)
    fwd_reset_done = False   # forward trades must be sized off `balance`, not the
                            # compounded in-sample equity -- reset once, when flat.

    for i in range(start, len(c)):
        hs = sp[i] * 0.5
        # ---- manage open position (engine fill logic) ----
        if pos != 0:
            ex = None
            reason_exit = None
            if pos == 1:
                if o[i] - hs <= sl: ex, reason_exit = o[i] - hs - slippage, "stop"
                elif l[i] - hs <= sl: ex, reason_exit = sl - slippage, "stop"
                elif o[i] - hs >= tp: ex, reason_exit = o[i] - hs, "target"
                elif h[i] - hs >= tp: ex, reason_exit = tp, "target"
            else:
                if o[i] + hs >= sl: ex, reason_exit = o[i] + hs + slippage, "stop"
                elif h[i] + hs >= sl: ex, reason_exit = sl + slippage, "stop"
                elif o[i] + hs <= tp: ex, reason_exit = o[i] + hs, "target"
                elif l[i] + hs <= tp: ex, reason_exit = tp, "target"
            if ex is None and i - ent_i >= max_hold:
                ex = (c[i] - hs - slippage) if pos == 1 else (c[i] + hs + slippage)
                reason_exit = "time"
            if ex is None and i == len(c) - 1:
                ex, reason_exit = None, None      # leave last position OPEN (still forming)
            if ex is not None:
                pnl = (ex - entry) * pos * value_per_unit * lots
                bal += pnl
                risk_usd = risk_px * value_per_unit * lots
                trades.append(dict(
                    id=len(trades) + 1,
                    timestamp_utc=str(pd.Timestamp(t[i]).tz_convert("UTC")),
                    timeframe="H1", strategy=STRATEGY_NAME,
                    sample=("forward" if pd.Timestamp(t[ent_i]).tz_convert("UTC") > cutoff else "in_sample"),
                    direction="BUY" if pos == 1 else "SELL",
                    signal_bar_utc=str(pd.Timestamp(t[ent_i - 1]).tz_convert("UTC")),
                    entry=round(entry, 2), exit=round(ex, 2), spread=round(sp[ent_i], 2),
                    atr=round(float(atr[ent_i - 1]), 2), stop=round(sl, 2), target=round(tp, 2),
                    volume=round(lots, 2), est_risk_usd=round(risk_usd, 4),
                    pnl_usd=round(pnl, 4),
                    r_multiple=round(pnl / risk_usd, 4) if risk_usd else None,
                    bars_held=int(i - ent_i), reason_entry=sig_reason, reason_exit=reason_exit,
                ))
                pos = 0

        if bal <= eq_stop:
            break
        if pos != 0:
            continue

        # ---- at the validation seam: restart the paper account so FORWARD trades
        #      are sized off the intended balance, not the in-sample compounded equity ----
        if not fwd_reset_done and pd.Timestamp(t[i]).tz_convert("UTC") > cutoff:
            bal = balance
            eq_stop = balance * 0.30
            fwd_reset_done = True

        # ---- entry (decided from completed bar i-1, fill at open[i]) ----
        d = int(entries[i])
        if d == 0:
            continue
        av = atr[i - 1]
        if not (av > 0):
            continue
        price = o[i]
        sp_ok, _ = spread_filter(sp[i], av, price, cfg)
        if not sp_ok:
            continue
        plan = build_stop_plan(direction="BUY" if d == 1 else "SELL",
                               bid=o[i] - sp[i] * 0.5, ask=o[i] + sp[i] * 0.5,
                               atr=av, cfg=cfg, broker_stops_level_price=spec.broker_stops_level,
                               slippage_price=slippage)
        if not plan.accepted:
            continue
        risk = abs(plan.entry - plan.sl)
        risk_d = bal * cfg.risk_per_trade
        loss_per_lot = risk * value_per_unit
        raw = risk_d / loss_per_lot if loss_per_lot > 0 else 0.0
        v = _normalise_volume(raw, 0.01, 0.01, 50.0)
        if v < 0.01:
            if (0.01 * loss_per_lot / bal) <= cfg.max_risk_per_trade:
                v = 0.01
            else:
                continue
        pos = d
        entry, sl, tp, lots, risk_px, ent_i = plan.entry, plan.sl, plan.tp, v, risk, i
        sig_reason = (f"{STRATEGY_NAME}: signal_bar={pd.Timestamp(t[i-1]).tz_convert('UTC')} "
                      f"atr={av:.1f} -> {'BUY' if d == 1 else 'SELL'}")
    return trades


def run_from_csv(csv_path: str, *, asset: str = "BTC", sim_balance: float | None = None) -> dict:
    spec = _spec(asset)
    p = Path(csv_path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        return {"error": f"export file not found: {p}", "timeframe": "H1"}
    if safety.kill_switch_active():
        return {"error": "KILL_SWITCH active -- replay aborted", "timeframe": "H1"}

    # BTC keeps the module-level paths so the existing tests' monkeypatch still bites.
    if spec.asset == "BTC":
        db, status_p, trades_p = _DB, _STATUS, _TRADES_CSV
    else:
        db = _ROOT / spec.db
        status_p = _ROOT / spec.status
        trades_p = _ROOT / spec.trades_csv
    db.parent.mkdir(parents=True, exist_ok=True)
    status_p.parent.mkdir(parents=True, exist_ok=True)

    df, meta = _load(p)
    balance = float(sim_balance) if sim_balance else 100.0
    cutoff = _cutoff_for(spec)
    trades = _simulate(df, balance, cutoff, spec)

    c = _conn(db)
    c.execute("DELETE FROM trades")
    for tr in trades:
        cols = ",".join(tr)
        c.execute(f"INSERT INTO trades({cols}) VALUES({','.join('?' for _ in tr)})", tuple(tr.values()))
    fwd = [t for t in trades if t["sample"] == "forward"]
    c.execute("INSERT INTO runs(ran_utc,export_utc,csv_last_bar,n_bars,n_trades,n_forward) VALUES(?,?,?,?,?,?)",
              (datetime.now(timezone.utc).isoformat(), meta.get("export_utc"),
               str(df["time"].iloc[-1]), len(df), len(trades), len(fwd)))
    c.commit(); c.close()

    pd.DataFrame(trades).to_csv(trades_p, index=False)

    def stats(ts):
        if not ts:
            return {"trades": 0}
        pnl = np.array([t["pnl_usd"] for t in ts], float)
        R = np.array([t["r_multiple"] for t in ts if t["r_multiple"] is not None], float)
        wins = pnl[pnl > 0]
        eq = balance + np.cumsum(pnl)
        peak = np.maximum.accumulate(np.concatenate([[balance], eq]))
        dd = (peak - np.concatenate([[balance], eq]))
        cl = mx = 0
        for x in pnl:
            cl = 0 if x > 0 else cl + 1
            mx = max(mx, cl)
        return {
            "trades": len(ts), "net_pl_usd": round(float(pnl.sum()), 2),
            "win_rate": round(float((pnl > 0).mean()), 4),
            "profit_factor": round(float(wins.sum() / -pnl[pnl <= 0].sum()), 4) if (pnl <= 0).any() and pnl[pnl <= 0].sum() < 0 else None,
            "expectancy_R": round(float(R.mean()), 4) if len(R) else None,
            "avg_win_usd": round(float(wins.mean()), 4) if len(wins) else 0.0,
            "avg_loss_usd": round(float(pnl[pnl <= 0].mean()), 4) if (pnl <= 0).any() else 0.0,
            "largest_loss_usd": round(float(pnl.min()), 4),
            "max_drawdown_usd": round(float(dd.max()), 2),
            "max_consecutive_losses": int(mx),
        }

    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "csv_replay (MT5-free, Oracle-Cloud-friendly)",
        "asset": spec.asset, "symbol": spec.symbol,
        "timeframe": "H1", "strategy": STRATEGY_NAME, "strategy_params": PARAMS,
        "sim_balance": balance,
        "export_utc": meta.get("export_utc"),
        "csv": str(p.relative_to(_ROOT)) if p.is_relative_to(_ROOT) else str(p),
        "candles": int(len(df)),
        "candle_span": [str(df["time"].iloc[0]), str(df["time"].iloc[-1])],
        "validation_cutoff_utc": str(cutoff),
        "data_source": meta.get("source", "MT5 export (Valetax)"),
        "spread_source": (meta.get("spread_note")
                          or f"MT5 export column (broker), floored at ${spec.broker_min_spread:.2f} -- NOT fabricated"),
        "in_sample": stats([t for t in trades if t["sample"] == "in_sample"]),
        "forward_out_of_sample": stats(fwd),
        "forward_trade_log": fwd[-30:],
        "live_trading_enabled": safety.live_trading_enabled(),
        "kill_switch_active": safety.kill_switch_active(),
        "NOTE": "PAPER REPLAY -- MetaTrader5 is not imported here; order_send() cannot be called.",
    }
    status_p.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    log.info("replay[%s]: %d candles -> %d trades (%d forward)  forward expR=%s net=$%s",
             spec.asset, len(df), len(trades), len(fwd),
             status["forward_out_of_sample"].get("expectancy_R"),
             status["forward_out_of_sample"].get("net_pl_usd"))
    return status
