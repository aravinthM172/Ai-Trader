"""Combined report of the live trader and every paper test (read-only on the status files)."""
import json
from datetime import datetime, timezone

from tools import paper_report as pr

NOW = datetime(2026, 10, 10, 18, tzinfo=timezone.utc)


def _write(d, name, obj):
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def test_all_tests_appear_in_one_table(tmp_path):
    _write(tmp_path, "multi_live_status.json", {"generated_utc": "2026-10-10T17:07:55", "closed_trades": 15,
                                                "open_positions": [1], "expectancy_R": 0.0165, "net_pl_usd": -10.02})
    _write(tmp_path, "paper_ideas_status.json", {"generated_utc": "2026-10-10T17:00:05", "ideas": {
        "breakout_nextday|BTCUSD": {"closed": 2, "open": 1, "avg_R": 0.4, "total_R": 0.8}}})
    _write(tmp_path, "paper_daily_status.json", {"generated_utc": "2026-10-10T16:40:20", "sleeves": {
        "rsi2_indices": {"closed": 3, "mean_R": -0.1, "total_R": -0.3, "ready_for_review": False}},
        "pairs": {"connors_rsi2|GER40": {"sleeve": "rsi2_indices", "open_paper_R": 0.2},
                  "connors_rsi2|NDX100": {"sleeve": "rsi2_indices", "open_paper_R": None}}})
    _write(tmp_path, "paper_ml_filter_status.json", {"generated_utc": "2026-10-10T17:00:09", "open_scored": 1,
                                                     "taken": {"closed": 9, "avg_R": 0.2, "total_R": 1.8},
                                                     "skipped": {"closed": 6, "avg_R": -0.1, "total_R": -0.6},
                                                     "verdict": "keep watching (15 of 60 closed trades, each group needs 20)"})
    s = pr.write(tmp_path, now=NOW)
    assert [r["group"] for r in s["rows"]] == ["LIVE", "paper hourly", "paper daily", "paper ML filter", "paper ML filter"]
    assert s["missing"] == [] and s["rows"][2]["open"] == 1
    assert json.loads((tmp_path / "paper_all_status.json").read_text())["rows"] == s["rows"]
    t = pr.text(s)
    assert "multi trader (real orders)" in t and "breakout_nextday BTCUSD" in t and "keep watching" in t


def test_missing_or_broken_files_are_listed_not_fatal(tmp_path):
    (tmp_path / "paper_ideas_status.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path, "paper_ml_filter_status.json", {"error": "no model"})
    s = pr.collect(tmp_path, NOW)
    assert s["rows"] == [] and len(s["missing"]) == 4
    assert "no status file yet" in pr.text(s)
