"""A blocked or skipped signal says which side it was and why, in the status line and in multi_live_state.json
("why"), so "why no trade?" has an exact answer.  "evaluated" keeps its old values (dashboard / MCP read them)."""
import json
from datetime import timedelta

from execution import live_multi as lm
from test_live_multi_drill import T0, drill  # noqa: F401  (fixture)


def test_blocked_signal_keeps_side_and_reason(drill, monkeypatch):
    monkeypatch.setattr(lm, "LONG_ONLY_SYMBOLS", {"USDJPY"})
    drill.signal = {"USDJPY": "SELL", "BTCUSD": "BUY"}
    st = drill.run(T0)
    e = st["entries"]["USDJPY"]
    assert e["decision"] == "blocked" and e["side"] == "SELL"
    assert lm._entry_text(e) == "blocked: SELL buy-only symbol"
    assert lm._entry_text(st["entries"]["BTCUSD"]) == "sent" and lm._entry_text("kill switch active") == "kill switch active"
    ls = json.loads((drill.dir / "ls.json").read_text())
    key = next(k for k in ls["evaluated"] if k.startswith("USDJPY|"))
    assert ls["evaluated"][key] == "blocked"                       # unchanged format
    assert ls["why"] == {key: "SELL buy-only symbol"}              # only blocked / skipped bars get a reason


def test_reasons_are_kept_for_the_last_500_bars(drill, monkeypatch):
    monkeypatch.setattr(lm, "LONG_ONLY_SYMBOLS", {"USDJPY"})
    drill.signal = {"USDJPY": "SELL"}
    ls = {"why": {f"X|2020-01-01T{i:05d}": "old" for i in range(600)}}
    (drill.dir / "ls.json").write_text(json.dumps(ls))
    drill.run(T0)
    why = json.loads((drill.dir / "ls.json").read_text())["why"]
    assert len(why) == 500 and any(k.startswith("USDJPY|") for k in why)


def test_hold_has_no_reason(drill):
    drill.signal = {}
    st = drill.run(T0 + timedelta(hours=1))
    assert all(lm._entry_text(v) == "hold" for v in st["entries"].values())
    assert not json.loads((drill.dir / "ls.json").read_text()).get("why")


def test_the_cap_drops_the_oldest_bars_not_one_symbol(drill):
    old = {f"{s}|2026-09-{d:02d}T{h:02d}:00:00+00:00": "hold"
           for s in ("ETHUSD", "XAUUSD", "ZZZ") for d in range(1, 29) for h in range(24)}
    (drill.dir / "ls.json").write_text(json.dumps({"evaluated": old}))   # 2016 old bars, all sorting after "BTCUSD"
    drill.run(T0)                                                         # + 3 new bars -> over the 2000 cap
    ev = json.loads((drill.dir / "ls.json").read_text())["evaluated"]
    assert any(k.startswith("BTCUSD|") for k in ev)


def test_newest_keeps_by_time():
    d = {"BTCUSD|2026-10-14T05:00:00+00:00": "hold", "XAUUSD|2026-01-01T00:00:00+00:00": "hold",
         "ZZZ|2025-01-01T00:00:00+00:00": "hold"}
    assert list(lm.newest(d, 2)) == ["XAUUSD|2026-01-01T00:00:00+00:00", "BTCUSD|2026-10-14T05:00:00+00:00"]
