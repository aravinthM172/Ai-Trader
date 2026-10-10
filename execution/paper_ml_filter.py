"""
Paper test of the signal-quality filter (strategy/ml_filter.py) on the LIVE trades -- NEVER sends orders and
never changes what the trader does.

Every trade the multi trader opened (state/multi_live.sqlite) is scored once, from the candles as they stood when
its signal bar closed: "taken" (score above the model threshold) or "skipped".  When the trade closes, its real R
goes into one of the two groups.  The question the test answers: do the trades the filter would have skipped do
worse than the ones it would have taken?

VERDICT RULE (fixed 2026-10-10, before any live trade was scored): the filter may be proposed for the trader only
when >= MIN_CLOSED closed trades are scored, with >= MIN_PER_GROUP in each group, AND the taken group's average R
is above the skipped group's.  Until then the answer is "keep watching".

Runs inside the hourly paper pass (execution/paper_ideas.py) or alone:
    python -m execution.paper_ml_filter            one pass
    python -m execution.paper_ml_filter --report   print the current standing
Writes state/paper_ml_filter.sqlite (table scores) and reports/paper_ml_filter_status.json.  Read-only on MT5.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.logging_setup import get_logger
from strategy import ml_filter as mf
from strategy import regime_filter

log = get_logger("paper.ml_filter", filename="paper_ideas.log")
ROOT = Path(__file__).resolve().parents[1]
LIVE_DB = ROOT / "state" / "multi_live.sqlite"
DB = ROOT / "state" / "paper_ml_filter.sqlite"
STATUS = ROOT / "reports" / "paper_ml_filter_status.json"
MIN_CLOSED, MIN_PER_GROUP = 60, 20


def _open(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE IF NOT EXISTS scores(
        ticket INTEGER PRIMARY KEY, symbol TEXT, signal_bar_utc TEXT, direction TEXT, score REAL, taken INTEGER,
        features TEXT, note TEXT, scored_utc TEXT)""")
    return c


def live_trades(db: Path) -> pd.DataFrame:
    cols = ["ticket", "symbol", "signal_bar_utc", "direction", "status", "r_multiple", "closed_utc"]
    if not db.exists():
        return pd.DataFrame(columns=cols)
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(f"SELECT {', '.join(cols)} FROM trades", c)
    finally:
        c.close()


def score_trade(gw, model: dict, symbol: str, signal_bar_utc: str, direction: str, now: datetime) -> tuple[dict | None, str]:
    """(result, note).  result = {score, taken, features} from the bars up to the signal bar; None = cannot score."""
    sig = pd.Timestamp(signal_bar_utc)
    sig = sig.tz_localize("UTC") if sig.tzinfo is None else sig.tz_convert("UTC")
    entry = sig + pd.Timedelta(hours=1)
    need = mf.H1_BARS + int(max((pd.Timestamp(now) - sig).total_seconds(), 0) // 3600) + 50
    h1 = gw.get_rates(symbol, "H1", need)
    if h1 is None or h1.empty:
        return None, "no H1 candles"
    h1 = h1[h1["time"] <= sig].reset_index(drop=True)
    if not len(h1) or h1["time"].iloc[-1] != sig:
        return None, "signal bar not in the candle history"
    if len(h1) < mf.VOL_MIN + 100:
        return None, f"only {len(h1)} H1 bars before the signal"
    d1 = gw.get_rates(symbol, "D1", regime_filter.D1_BARS)
    if d1 is None or d1.empty:
        return None, "no daily candles"
    done = regime_filter.completed_daily(d1, entry.to_pydatetime())
    if len(done) < regime_filter.MIN_BARS:
        return None, f"only {len(done)} completed daily bars"
    close = done["close"].astype(float)
    ema = float(close.ewm(span=regime_filter.EMA_LEN, adjust=False).mean().iloc[-1])
    feats = mf.features(mf.bar_frame(h1).iloc[-1], 1 if direction == "BUY" else -1, float(close.iloc[-1]), ema, entry.hour)
    if feats is None:
        return None, "a feature is missing (warm-up)"
    s, taken = mf.score(model, feats)
    return {"score": s, "taken": bool(taken), "features": feats}, "ok"


def run_once(gw, now: datetime | None = None, live_db: Path = LIVE_DB, db: Path = DB, status: Path = STATUS,
             model: dict | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    model = model or mf.load_model()
    if model is None:
        log.warning("no usable model file (%s) -- nothing scored", mf.MODEL_PATH.name)
        return {"error": "no model"}
    trades = live_trades(live_db)
    c = _open(db)
    try:
        done = {r[0] for r in c.execute("SELECT ticket FROM scores WHERE score IS NOT NULL")}
        for t in trades[~trades.ticket.isin(done)].itertuples():
            if not t.signal_bar_utc or t.direction not in ("BUY", "SELL"):
                continue
            try:
                res, note = score_trade(gw, model, t.symbol, t.signal_bar_utc, t.direction, now)
            except Exception as e:                                  # one bad symbol must not stop the pass
                res, note = None, f"error: {e}"
                log.exception("scoring %s #%s failed", t.symbol, t.ticket)
            c.execute("INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?,?)",
                      (int(t.ticket), t.symbol, t.signal_bar_utc, t.direction, res and res["score"],
                       res and int(res["taken"]), json.dumps(res["features"]) if res else None, note, now.isoformat()))
            if res:
                log.info("scored %s #%s %s: %+.3f -> %s", t.symbol, t.ticket, t.direction, res["score"],
                         "taken" if res["taken"] else "skipped")
        c.commit()
        scores = pd.read_sql_query("SELECT ticket, score, taken, note FROM scores", c)
    finally:
        c.close()
    out = standing(trades, scores, now, model)
    status.parent.mkdir(exist_ok=True)
    status.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _group(x: pd.DataFrame) -> dict:
    r = x.r_multiple.dropna()
    return {"closed": int(len(r)), "avg_R": round(float(r.mean()), 3) if len(r) else None,
            "total_R": round(float(r.sum()), 2), "win_pct": round(100 * float((r > 0).mean()), 1) if len(r) else None}


def standing(trades: pd.DataFrame, scores: pd.DataFrame, now: datetime, model: dict) -> dict:
    m = trades.merge(scores, on="ticket", how="left")
    ok = m[m.score.notna()]
    closed = ok[(ok.status == "CLOSED") & ok.r_multiple.notna()]
    tk, sk = closed[closed.taken == 1], closed[closed.taken == 0]
    g_t, g_s = _group(tk), _group(sk)
    enough = len(closed) >= MIN_CLOSED and min(len(tk), len(sk)) >= MIN_PER_GROUP
    if not enough:
        verdict = f"keep watching ({len(closed)} of {MIN_CLOSED} closed trades, each group needs {MIN_PER_GROUP})"
    elif g_t["avg_R"] > g_s["avg_R"]:
        verdict = "filter helps so far: may be proposed to the owner"
    else:
        verdict = "filter does NOT help so far: do not use"
    return {"generated_utc": now.isoformat(), "model_trained_to": model.get("trained_to"), "threshold": model["threshold"],
            "live_trades": int(len(trades)), "scored": int(len(ok)), "not_scored": int(len(m) - len(ok)),
            "open_scored": int((ok.status != "CLOSED").sum()), "taken": g_t, "skipped": g_s, "all_scored": _group(closed),
            "by_symbol": {s: {"taken": _group(g[g.taken == 1]), "skipped": _group(g[g.taken == 0])}
                          for s, g in closed.groupby("symbol")},
            "verdict": verdict}


def report(status: Path = STATUS) -> str:
    try:
        s = json.loads(status.read_text(encoding="utf-8"))
    except Exception:
        return "no paper ML-filter status yet"
    if "error" in s:
        return f"paper ML filter: {s['error']}"
    line = lambda k: f"{s[k]['closed']} closed, avg R {s[k]['avg_R']}, total R {s[k]['total_R']}"  # noqa: E731
    return "\n".join([f"paper ML filter @ {s['generated_utc'][:16]}  scored {s['scored']} of {s['live_trades']} live trades "
                      f"({s['open_scored']} still open, {s['not_scored']} not scored)",
                      f"  taken:   {line('taken')}", f"  skipped: {line('skipped')}", f"  verdict: {s['verdict']}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    if ap.parse_args().report:
        print(report())
        return 0
    from dotenv import load_dotenv
    load_dotenv()
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    try:
        if not gw.connect():
            log.error("MT5 connection failed")
            return 1
        run_once(gw)
    except Exception as e:
        log.exception("pass failed: %s", e)
        return 1
    finally:
        gw.shutdown()
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
