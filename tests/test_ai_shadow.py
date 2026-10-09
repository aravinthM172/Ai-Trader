import json
import time

import pandas as pd

from execution import ai_shadow
from tools.ai_shadow_report import MIN_SIGNALS, summarise


def _bars(n=100):
    t = pd.date_range("2026-10-01", periods=n, freq="h", tz="UTC")
    c = pd.Series(range(n), dtype=float) + 100
    return pd.DataFrame({"time": t, "open": c, "high": c + 1, "low": c - 1, "close": c})


def test_parse_reply():
    assert ai_shadow.parse_reply('{"verdict":"take","confidence":0.7,"reason":"clean trend"}') == ("TAKE", 0.7, "clean trend")
    assert ai_shadow.parse_reply('{"verdict":"SKIP","confidence":3}')[:2] == ("SKIP", 1.0)
    assert ai_shadow.parse_reply('{"verdict":"MAYBE"}')[0] == "ERROR"
    assert ai_shadow.parse_reply("not json")[0] == "ERROR"


def test_prompt_has_plan_and_last_bars():
    p = ai_shadow.build_prompt("XAUUSD", "BUY", {"rsi14": 60.0}, _bars(), 2.0)
    assert "stop 195" in p and "target 205" in p          # last close 199, ATR 2
    assert p.count("\n2026-10-") == ai_shadow.BARS_IN_PROMPT
    assert '"rsi14": 60.0' in p


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not ai_shadow.enabled()
    assert ai_shadow.submit("XAUUSD", "BUY", "2026-10-05T04:00:00+00:00", {}, _bars(), 2.0) is False
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("AI_SHADOW", "false")
    assert not ai_shadow.enabled()


def test_submit_asks_once_per_bar_and_stores(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("AI_SHADOW", raising=False)
    monkeypatch.setattr(ai_shadow, "DB", tmp_path / "ai.sqlite")
    monkeypatch.setattr(ai_shadow, "_asked", set())
    calls = []
    monkeypatch.setattr(ai_shadow, "_ask", lambda m, p: calls.append(m) or json.dumps(
        {"verdict": "SKIP", "confidence": 0.6, "reason": "choppy"}))
    bar = "2026-10-05 04:00:00+00:00"
    assert ai_shadow.submit("XAUUSD", "SELL", bar, {}, _bars(), 2.0) is True
    assert ai_shadow.submit("XAUUSD", "SELL", bar, {}, _bars(), 2.0) is False     # same bar: no second call
    for _ in range(50):
        if ai_shadow.DB.exists() and ai_shadow._conn().execute("SELECT COUNT(*) FROM opinions").fetchone()[0]:
            break
        time.sleep(0.05)
    row = ai_shadow._conn().execute("SELECT verdict, confidence, reason, error FROM opinions").fetchone()
    assert row == ("SKIP", 0.6, "choppy", None) and len(calls) == 1


def test_api_error_is_stored_not_raised(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(ai_shadow, "DB", tmp_path / "ai.sqlite")

    def boom(m, p):
        raise TimeoutError("slow")
    monkeypatch.setattr(ai_shadow, "_ask", boom)
    ai_shadow._work("BTCUSD", "2026-10-05T04:00:00+00:00", "BUY", "gpt-5.5", "p")
    v, err = ai_shadow._conn().execute("SELECT verdict, error FROM opinions").fetchone()
    assert v == "ERROR" and "slow" in err


def test_summarise_pass_rule():
    rows = [{"verdict": "TAKE", "r": 1.5}] * 20 + [{"verdict": "SKIP", "r": -1.0}] * 10
    s = summarise(rows)
    assert s["scored"] == MIN_SIGNALS and s["take_minus_skip_r"] == 2.5 and s["pass"]
    assert not summarise(rows[:10])["pass"]                               # too few
    assert not summarise([{"verdict": "TAKE", "r": 0.2}] * 20 + [{"verdict": "SKIP", "r": 0.2}] * 10)["pass"]
    assert summarise([{"verdict": "TAKE", "r": None}])["scored"] == 0      # open trades not scored
