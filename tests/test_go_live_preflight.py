from datetime import datetime, timedelta, timezone

from tools import go_live_preflight as pf


def test_uptime_finds_long_gap(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    lines, t = [], now - timedelta(hours=12)
    while t < now:
        if not (now - timedelta(hours=8) < t < now - timedelta(hours=3)):        # 5 h outage
            lines.append(f"[{t.replace(tzinfo=None).isoformat()}] LIVE eq $5000 open 0 closed 0 net $0 | {{}}")
        t += timedelta(seconds=30)
    log = tmp_path / "console.log"
    log.write_text("\n".join(lines))
    monkeypatch.setattr(pf, "CONSOLE", log)
    rep = pf.Report()
    gaps = pf.check_uptime(rep)
    assert len(gaps) == 1 and rep.items[0]["status"] == "FAIL"
    assert 4.9 <= (gaps[0][1] - gaps[0][0]).total_seconds() / 3600 <= 5.1


def test_no_gap_passes(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    lines = [f"[{(now - timedelta(seconds=30 * k)).replace(tzinfo=None).isoformat()}] LIVE eq" for k in range(2000, 0, -1)]
    log = tmp_path / "console.log"
    log.write_text("\n".join(lines))
    monkeypatch.setattr(pf, "CONSOLE", log)
    rep = pf.Report()
    assert pf.check_uptime(rep) == [] and rep.items[0]["status"] == "PASS"
