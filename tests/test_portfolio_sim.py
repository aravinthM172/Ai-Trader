import pandas as pd

from backtest import portfolio_sim as ps
from risk import portfolio_guard as pg


def _trades(rows):
    df = pd.DataFrame(rows, columns=["symbol", "group", "entry_time", "exit_time", "direction", "R", "stopped"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def test_equity_compounds_on_exit_and_caps_block_overlap():
    t = _trades([
        ("A", "G1", "2026-01-01 00:00", "2026-01-01 05:00", 1, 2.0, False),
        ("B", "G1", "2026-01-01 01:00", "2026-01-01 06:00", 1, -1.0, True),
        ("C", "G1", "2026-01-01 02:00", "2026-01-01 07:00", 1, 1.0, False),   # 3rd in group -> blocked
        ("C", "G1", "2026-01-02 00:00", "2026-01-02 05:00", 1, 1.0, False),
    ])
    cfg = pg.GuardConfig(max_open_positions=6, max_total_open_risk=0.03, max_per_group=2)
    r = ps.simulate(t, cfg, risk=0.01, start_equity=10_000)
    assert r["trades_taken"] == 3 and r["trades_skipped"] == 1
    # +2R on $100, -1R on $100, then +1R on 1% of 10,100
    assert abs(r["total_return_pct"] - round(100 * (10_000 + 200 - 100 + 101 - 10_000) / 10_000, 1)) < 1e-9


def test_prop_challenge_detects_daily_breach():
    good = pd.Series([0.01] * 40, index=pd.date_range("2026-01-01", periods=40, freq="D"))
    out = ps.prop_challenge(good)
    assert out["phase1_10pct"]["pass_rate"] == 1.0
    bad = pd.Series([-0.06] + [0.0] * 39, index=good.index)
    assert ps.prop_challenge(bad)["phase1_10pct"]["pass_rate"] < 1.0
