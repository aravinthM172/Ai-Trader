import sqlite3

import numpy as np
import pandas as pd

from tools import edge_monitor as em


def _ref(seed=1, n=1300):
    rng = np.random.default_rng(seed)
    # backtest-like: 45 % wins of ~+1.45 R, losses of -1 R  -> mean ~ +0.10..0.2 R
    return np.where(rng.random(n) < 0.5, 1.45, -1.0)


def test_healthy_edge_does_not_retire():
    ref = _ref()
    live = _ref(seed=2, n=60)
    r = em.evaluate(live, ref)
    assert r["retire"] is False, r["trips"]


def test_dead_edge_retires_on_sample_test():
    ref = _ref()
    live = np.where(np.random.default_rng(3).random(60) < 0.25, 1.45, -1.0)   # mean ~ -0.4 R
    r = em.evaluate(live, ref)
    assert r["retire"] and any("pct" in t for t in r["trips"])


def test_sample_test_waits_for_min_trades():
    r = em.evaluate([-1.0] * (em.MIN_TRADES - 1), _ref())
    assert r["sample_floor_R"] is None
    assert not any("pct" in t for t in r["trips"])


def test_long_losing_streak_retires():
    ref = _ref()
    limit = em.worst_streak(ref) + em.STREAK_MARGIN
    r = em.evaluate([1.45] * 5 + [-1.0] * (limit + 1), ref)
    assert any("losing streak" in t for t in r["trips"])


def test_decay_window():
    ref = _ref()
    live = np.tile([1.45, -1.0, -1.0], 40)[:em.DECAY_WINDOW + 5]            # mean ~ -0.18 R
    r = em.evaluate(live, ref)
    assert any(f"last {em.DECAY_WINDOW}" in t for t in r["trips"])


def _db(path, R):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE trades(ticket INTEGER PRIMARY KEY, status TEXT, closed_utc TEXT, r_multiple REAL)")
    for i, r in enumerate(R):
        c.execute("INSERT INTO trades VALUES(?,?,?,?)", (i, "CLOSED", f"2026-10-{1 + i // 10:02d}T{i % 10:02d}", r))
    c.commit()
    c.close()


def test_run_dry_never_writes_kill_file(tmp_path):
    db, ref, kill, out = tmp_path / "l.sqlite", tmp_path / "r.csv", tmp_path / "KILL", tmp_path / "o.json"
    _db(db, [-1.0] * 40)
    pd.DataFrame({"r_multiple": _ref()}).to_csv(ref, index=False)
    r = em.run(enforce=False, db=db, ref=ref, kill_file=kill, out=out)
    assert r["retire"] and not r["enforced"] and not kill.exists()


def test_run_enforce_writes_kill_file(tmp_path):
    db, ref, kill, out = tmp_path / "l.sqlite", tmp_path / "r.csv", tmp_path / "KILL", tmp_path / "o.json"
    _db(db, [-1.0] * 40)
    pd.DataFrame({"r_multiple": _ref()}).to_csv(ref, index=False)
    r = em.run(enforce=True, db=db, ref=ref, kill_file=kill, out=out)
    assert r["enforced"] and kill.exists() and "edge_monitor" in kill.read_text()
