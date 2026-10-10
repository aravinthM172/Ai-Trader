"""
All paper tests in one place (added 2026-10-10).  Read-only: it only reads the status files the paper passes write.

    python -m tools.paper_report          print one table and write reports/paper_all_status.json

Sources:
  reports/multi_live_status.json        the LIVE trader, shown first for comparison
  reports/paper_ideas_status.json       hourly ideas (breakout, gold ideas)        execution/paper_ideas.py
  reports/paper_daily_status.json       daily ideas (RSI-2 indices, gold/BTC trend) execution/paper_daily.py
  reports/paper_ml_filter_status.json   signal-quality filter on the live trades    execution/paper_ml_filter.py
The hourly paper pass refreshes the combined file, so it is never more than about an hour old.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "paper_all_status.json"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row(group: str, name: str, closed, open_, avg_R, total_R, updated, note: str = "") -> dict:
    return {"group": group, "test": name, "closed": closed, "open": open_, "avg_R": avg_R, "total_R": total_R,
            "updated_utc": (updated or "")[:16], "note": note}


def collect(reports: Path = REPORTS, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rows, missing = [], []
    live = _read(reports / "multi_live_status.json")
    if live:
        rows.append(_row("LIVE", "multi trader (real orders)", live.get("closed_trades"), len(live.get("open_positions") or []),
                         live.get("expectancy_R"), None, live.get("generated_utc"), f"net ${live.get('net_pl_usd')}"))
    else:
        missing.append("multi_live_status.json")
    ideas = _read(reports / "paper_ideas_status.json")
    if ideas:
        for key, v in ideas.get("ideas", {}).items():
            rows.append(_row("paper hourly", key.replace("|", " "), v.get("closed"), v.get("open"), v.get("avg_R"),
                             v.get("total_R"), ideas.get("generated_utc")))
    else:
        missing.append("paper_ideas_status.json")
    daily = _read(reports / "paper_daily_status.json")
    if daily:
        for key, v in daily.get("sleeves", {}).items():
            open_n = sum(1 for p in daily.get("pairs", {}).values() if p.get("sleeve") == key and p.get("open_paper_R") is not None)
            rows.append(_row("paper daily", key, v.get("closed"), open_n, v.get("mean_R"), v.get("total_R"),
                             daily.get("generated_utc"), "ready for review" if v.get("ready_for_review") else ""))
    else:
        missing.append("paper_daily_status.json")
    ml = _read(reports / "paper_ml_filter_status.json")
    if ml and "error" not in ml:
        for k in ("taken", "skipped"):
            rows.append(_row("paper ML filter", f"live trades the filter {k}", ml[k]["closed"],
                             ml.get("open_scored") if k == "taken" else None, ml[k]["avg_R"], ml[k]["total_R"],
                             ml.get("generated_utc"), ml.get("verdict", "") if k == "taken" else ""))
    else:
        missing.append("paper_ml_filter_status.json")
    return {"generated_utc": now.isoformat(timespec="seconds"), "rows": rows, "missing": missing}


def text(s: dict) -> str:
    f = lambda v, spec: "" if v is None else format(v, spec)  # noqa: E731
    lines = [f"ALL TESTS @ {s['generated_utc'][:16]} UTC   (R = one unit of risk; paper tests send no orders)",
             f"{'group':<16}{'test':<34}{'closed':>7}{'open':>6}{'avg R':>8}{'total R':>9}  {'updated':<17}note"]
    for r in s["rows"]:
        lines.append(f"{r['group']:<16}{r['test'][:33]:<34}{f(r['closed'], 'd'):>7}{f(r['open'], 'd'):>6}"
                     f"{f(r['avg_R'], '+.3f'):>8}{f(r['total_R'], '+.2f'):>9}  {r['updated_utc']:<17}{r['note']}")
    if s["missing"]:
        lines.append("no status file yet: " + ", ".join(s["missing"]))
    return "\n".join(lines)


def write(reports: Path = REPORTS, out: Path | None = None, now: datetime | None = None) -> dict:
    s = collect(reports, now)
    out = out or reports / OUT.name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def main() -> int:
    print(text(write()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
