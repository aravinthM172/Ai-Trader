"""
AI second opinion in SHADOW mode: for every BUY/SELL the frozen H1 rule produces, ask an OpenAI
model "take or skip?" and record the answer.  The answer NEVER changes what the bot does -- no
veto, no sizing, no delay.  The point is to measure it first: tools/ai_shadow_report.py replays
every logged signal with the live stop plan and compares the mean R of AI-TAKE vs AI-SKIP signals.
Only if SKIP clearly loses more (30+ signals) is it worth wiring in as a veto -- the same promotion
rule as every other strategy here.

Why not backtest it instead: an LLM has seen the historical prices and news in its training data,
so asking it about a past chart leaks the future.  Only signals after it went live are honest.

The call runs in a daemon thread, off the trading path; any error (no key, no credit, timeout) is
logged and stored as verdict ERROR.  One call per (symbol, signal bar), however many times the
trader re-evaluates that bar.

    AI_SHADOW=true|false     default true when OPENAI_API_KEY is set
    AI_MODEL=gpt-5.5         any chat-completions model the key can use
    AI_REASONING=medium      reasoning effort for reasoning models (low|medium|high)

Records: state/ai_shadow.sqlite (table opinions).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.logging_setup import get_logger

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "state" / "ai_shadow.sqlite"
URL = "https://api.openai.com/v1/chat/completions"
BARS_IN_PROMPT = 72                     # last 3 days of H1 bars
TIMEOUT_S = 120

log = get_logger("ai.shadow", filename="ai_shadow.log")
_lock = threading.Lock()
_asked: set[tuple[str, str]] = set()

SYSTEM = """You are a discretionary risk reviewer sitting next to a systematic H1 momentum strategy.
The strategy has fired a signal. Your job: say whether a careful discretionary trader would TAKE it
or SKIP it, judging only from the data given (trend quality, chop vs clean trend, how extended the
move already is, nearby swing levels, volatility).
The trade plan is fixed: stop 2 ATR, target 3 ATR, time exit after 96 bars. You cannot change it.
Be calibrated: most signals of a profitable system should be TAKE; SKIP only what looks clearly poor.
Return ONLY JSON: {"verdict": "TAKE" | "SKIP", "confidence": 0.0-1.0, "reason": "<= 2 short sentences"}"""


def enabled() -> bool:
    raw = os.getenv("AI_SHADOW")
    if raw is not None and raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def _conn(path: Path | None = None) -> sqlite3.Connection:
    path = path or DB
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS opinions(
        symbol TEXT, signal_bar_utc TEXT, direction TEXT, asked_utc TEXT, model TEXT,
        verdict TEXT, confidence REAL, reason TEXT, latency_s REAL, error TEXT,
        PRIMARY KEY(symbol, signal_bar_utc))""")
    return c


# -- pure helpers (unit-tested) ---------------------------------------------------------
def build_prompt(symbol: str, direction: str, features: dict, done: pd.DataFrame, atr: float) -> str:
    bars = done.tail(BARS_IN_PROMPT)
    rows = "\n".join(f"{pd.Timestamp(t):%Y-%m-%d %H:%M},{o:.6g},{h:.6g},{l:.6g},{c:.6g}"
                     for t, o, h, l, c in zip(bars["time"], bars["open"], bars["high"], bars["low"], bars["close"]))
    px = float(done["close"].iloc[-1])
    sign = 1 if direction == "BUY" else -1
    return (f"Symbol: {symbol}\nSignal: {direction} at the open of the next H1 bar (last close {px:.6g})\n"
            f"ATR(14, H1): {atr:.6g}  ->  stop {px - sign * 2 * atr:.6g}, target {px + sign * 3 * atr:.6g}\n"
            f"Strategy features on the signal bar: {json.dumps(features)}\n\n"
            f"Last {len(bars)} completed H1 bars, UTC (time,open,high,low,close):\n{rows}")


def parse_reply(text: str) -> tuple[str, float | None, str]:
    """(verdict, confidence, reason); verdict ERROR if the reply is not the requested JSON."""
    try:
        d = json.loads(text)
        v = str(d.get("verdict", "")).strip().upper()
        if v not in {"TAKE", "SKIP"}:
            return "ERROR", None, f"bad verdict: {text[:200]}"
        conf = d.get("confidence")
        conf = min(1.0, max(0.0, float(conf))) if conf is not None else None
        return v, conf, str(d.get("reason", ""))[:500]
    except (ValueError, TypeError, AttributeError):
        return "ERROR", None, f"not JSON: {text[:200]}"


# -- the call ---------------------------------------------------------------------------
def _ask(model: str, prompt: str) -> str:
    body = {"model": model, "response_format": {"type": "json_object"}, "max_completion_tokens": 4000,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    effort = os.getenv("AI_REASONING", "medium")
    if effort and model.startswith(("gpt-5", "o")):
        body["reasoning_effort"] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY", ""), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _work(symbol, bar, direction, model, prompt) -> None:
    t0 = datetime.now(timezone.utc)
    verdict, conf, reason, err = "ERROR", None, "", None
    try:
        verdict, conf, reason = parse_reply(_ask(model, prompt))
        if verdict == "ERROR":
            err, reason = reason, ""
    except urllib.error.HTTPError as e:
        try:
            err = f"HTTP {e.code}: {json.loads(e.read()).get('error', {}).get('message', '')[:300]}"
        except Exception:
            err = f"HTTP {e.code}"
    except Exception as e:                          # timeout, DNS, ... -- never reaches the trader
        err = f"{type(e).__name__}: {e}"[:300]
    lat = (datetime.now(timezone.utc) - t0).total_seconds()
    try:
        with _lock:
            c = _conn()
            c.execute("INSERT OR REPLACE INTO opinions VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (symbol, bar, direction, t0.isoformat(), model, verdict, conf, reason, round(lat, 1), err))
            c.commit()
            c.close()
    except Exception as e:
        log.error("ai shadow: could not store opinion for %s %s: %s", symbol, bar, e)
    if err:
        log.warning("AI SHADOW %s %s %s -> ERROR %s", symbol, direction, bar, err)
    else:
        log.info("AI SHADOW %s %s %s -> %s (%.2f) %s", symbol, direction, bar, verdict, conf or 0, reason)


def submit(symbol: str, direction: str, signal_bar_utc: str | None, features: dict,
           done: pd.DataFrame, atr: float) -> bool:
    """Fire-and-forget.  Returns True if a request was started.  Never raises."""
    try:
        if not enabled() or not signal_bar_utc:
            return False
        bar = pd.Timestamp(signal_bar_utc).isoformat()
        key = (symbol, bar)
        with _lock:
            if key in _asked:
                return False
            _asked.add(key)
            c = _conn()
            seen = c.execute("SELECT verdict FROM opinions WHERE symbol=? AND signal_bar_utc=?", key).fetchone()
            c.close()
        if seen and seen[0] != "ERROR":
            return False
        model = os.getenv("AI_MODEL", "gpt-5.5")
        prompt = build_prompt(symbol, direction, features, done, atr)
        threading.Thread(target=_work, args=(symbol, bar, direction, model, prompt),
                         name=f"ai-shadow-{symbol}", daemon=True).start()
        return True
    except Exception as e:
        log.error("ai shadow submit failed for %s: %s", symbol, e)
        return False
