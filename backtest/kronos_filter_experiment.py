"""
Research: does a Kronos forecast (HF NeoQuasar/Kronos-small) improve the frozen
momentum_rsi_mtf BTC H1 trades if used as an agree/disagree filter?

Runs in the SEPARATE research venv (torch), never in the live bot's venv:
    D:\\Downloads\\Project\\kronos-research\\venv\\Scripts\\python -m backtest.kronos_filter_experiment
    ... --analyse-only        (re-analyse the cached forecasts)

Clean window only: Kronos was published 2025-08 and pretrained on multi-exchange
candles that very likely include older BTC history, so trades before
CLEAN_FROM would be judged by a model that may have seen their outcome.

Trades = the real-feed forward replay (reports/btc_paper_replay_trades.csv,
Valetax export, broker spread) -- exactly the trades the live bot would take.
For each trade: context = the 512 completed H1 bars up to the signal bar,
forecast = next PRED_LEN bars (mean of SAMPLES paths).  Filter = keep the trade
only if the forecast close at horizon h moves in the trade's direction.

Trade-level subset analysis: skipping a trade could free the slot for another
signal in reality; that knock-on is ignored here (first-pass test).

Writes reports/kronos_forecasts.csv (cache, resumable) and reports/kronos_filter.json.
Research only -- NO LIVE ORDER, nothing here touches execution/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KRONOS_REPO = ROOT.parent / "kronos-research" / "Kronos"

from backtest.btc_bitstamp_h1_dataset import valetax_true_utc

TRADES = ROOT / "reports" / "btc_paper_replay_trades.csv"
BARS = ROOT / "data" / "export" / "btcusd_vx_H1_export.csv"
CACHE = ROOT / "reports" / "kronos_forecasts.csv"
OUT = ROOT / "reports" / "kronos_filter.json"
CLEAN_FROM = pd.Timestamp("2025-09-01", tz="UTC")
CONTEXT, PRED_LEN, SAMPLES = 512, 8, 3
HORIZONS = (2, 4, 8)


def load_bars() -> pd.DataFrame:
    b = pd.read_csv(BARS)
    b["time"] = pd.to_datetime(b["time"], utc=True)
    b = b.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    b["kronos_time"] = valetax_true_utc(b["time"])        # undo the winter 1h label shift
    b["kronos_time"] = b["kronos_time"].fillna(b["time"]).dt.tz_localize(None)
    return b


def load_trades() -> pd.DataFrame:
    t = pd.read_csv(TRADES)
    for c in ("timestamp_utc", "signal_bar_utc"):
        t[c] = pd.to_datetime(t[c], utc=True)
    return t[t["signal_bar_utc"] >= CLEAN_FROM].reset_index(drop=True)


def forecast(trades: pd.DataFrame, bars: pd.DataFrame) -> None:
    import torch
    sys.path.insert(0, str(KRONOS_REPO))
    from model import Kronos, KronosTokenizer, KronosPredictor
    torch.set_num_threads(3)                 # leave a core for the live bot + MT5
    torch.manual_seed(0)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=CONTEXT)

    done = set(pd.read_csv(CACHE)["id"]) if CACHE.exists() else set()
    idx = {t: i for i, t in enumerate(bars["time"])}
    todo = trades[~trades["id"].isin(done)]
    print(f"{len(trades)} clean trades, {len(done)} cached, {len(todo)} to forecast", flush=True)
    t0 = time.time()
    for n, tr in enumerate(todo.itertuples(), 1):
        s = idx.get(tr.signal_bar_utc)
        if s is None or s + 1 < CONTEXT:
            continue
        x = bars.iloc[s + 1 - CONTEXT:s + 1]
        step = pd.Timedelta(hours=1)
        y_ts = pd.Series([x["kronos_time"].iloc[-1] + step * (k + 1) for k in range(PRED_LEN)])
        xd = x[["open", "high", "low", "close"]].astype(float).reset_index(drop=True)
        xd["volume"] = x["tick_volume"].astype(float).to_numpy()
        p = pred.predict(df=xd, x_timestamp=x["kronos_time"].reset_index(drop=True), y_timestamp=y_ts,
                         pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=SAMPLES, verbose=False)
        last = float(x["close"].iloc[-1])
        row = {"id": tr.id, "signal_bar_utc": tr.signal_bar_utc, "direction": tr.direction,
               "r_multiple": tr.r_multiple, "last_close": last,
               "pred_high_max": float(p["high"].max()), "pred_low_min": float(p["low"].min())}
        for h in HORIZONS:
            row[f"pred_ret_h{h}"] = float(p["close"].iloc[h - 1]) / last - 1.0
        pd.DataFrame([row]).to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el / n:.1f}s each  ETA {el / n * (len(todo) - n) / 60:.0f} min", flush=True)


def _stats(R: np.ndarray) -> dict:
    if len(R) == 0:
        return {"trades": 0}
    se = R.std(ddof=1) / np.sqrt(len(R)) if len(R) > 1 else np.nan
    return {"trades": int(len(R)), "expectancy_R": round(float(R.mean()), 4),
            "total_R": round(float(R.sum()), 2), "t_stat": round(float(R.mean() / se), 2) if se > 0 else None,
            "win_rate": round(float((R > 0).mean()), 4)}


def analyse() -> dict:
    f = pd.read_csv(CACHE)
    d = np.where(f["direction"] == "BUY", 1.0, -1.0)
    R = f["r_multiple"].to_numpy(float)
    rng = np.random.default_rng(7)
    rep = {"clean_from": str(CLEAN_FROM.date()), "model": "NeoQuasar/Kronos-small",
           "context": CONTEXT, "pred_len": PRED_LEN, "samples": SAMPLES,
           "all_trades": _stats(R), "filters": {}}
    for h in HORIZONS:
        agree = d * f[f"pred_ret_h{h}"].to_numpy() > 0
        keep, drop = R[agree], R[~agree]
        # null: random subsets of the same size -- how often does chance do as well?
        null = np.array([rng.choice(R, size=len(keep), replace=False).mean() for _ in range(5000)]) \
            if 0 < len(keep) < len(R) else np.array([R.mean()])
        rep["filters"][f"agree_h{h}"] = {
            "kept": _stats(keep), "rejected": _stats(drop),
            "direction_hit_rate": round(float((np.sign(f[f"pred_ret_h{h}"]) == d).mean()), 4),
            "p_value_vs_random_subset": round(float((null >= keep.mean()).mean()), 4) if len(keep) else None,
        }
    best = max(rep["filters"].items(), key=lambda kv: kv[1]["kept"].get("expectancy_R") or -9)
    k, a = best[1]["kept"], rep["all_trades"]
    rep["verdict"] = (
        "PROMISING -> paper-test next to the demo bot"
        if (best[1]["p_value_vs_random_subset"] or 1) < 0.05 and k["trades"] >= 0.5 * a["trades"]
        and k["total_R"] >= a["total_R"] else
        "NO CLEAR BENEFIT -> keep the strategy unfiltered")
    rep["best_filter"] = best[0]
    OUT.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    if not a.analyse_only:
        forecast(load_trades(), load_bars())
    rep = analyse()
    al = rep["all_trades"]
    print("=" * 92)
    print(f"KRONOS FILTER on momentum_rsi_mtf BTC H1  (clean window from {rep['clean_from']})  -- research only")
    print("=" * 92)
    print(f"all trades         : n={al['trades']}  expR={al['expectancy_R']}  total {al['total_R']}R  win {al['win_rate']}")
    for k, v in rep["filters"].items():
        kp, rj = v["kept"], v["rejected"]
        print(f"{k:18s} : kept n={kp['trades']} expR={kp.get('expectancy_R')} total {kp.get('total_R')}R  | "
              f"rejected n={rj['trades']} expR={rj.get('expectancy_R')}  | hit {v['direction_hit_rate']}  "
              f"p={v['p_value_vs_random_subset']}")
    print(f"verdict: {rep['verdict']}  (best {rep['best_filter']})")
    print(f"report: {OUT.relative_to(ROOT)}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
