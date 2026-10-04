from datetime import datetime, timedelta, timezone

from risk import portfolio_guard as pg

T0 = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)


def _allow(cfg, st, sym="EURUSD.vx", group="FX Majors", risk=0.005, now=T0, eq=10_000):
    return pg.allow_entry(cfg, st, symbol=sym, group=group, risk_frac=risk, now=now, equity=eq)


def test_max_positions_and_group_and_risk_caps():
    cfg = pg.GuardConfig(max_open_positions=3, max_total_open_risk=0.012, max_per_group=2)
    st = pg.GuardState()
    for s in ("EURUSD.vx", "GBPUSD.vx"):
        assert _allow(cfg, st, sym=s)[0]
        pg.on_open(st, symbol=s, group="FX Majors", risk_frac=0.005)
    ok, why = _allow(cfg, st, sym="USDJPY.vx", risk=0.001)
    assert not ok and "max_per_group" in why
    ok, why = _allow(cfg, st, sym="NAS100.vx", group="Indexes")
    assert not ok and "total open risk" in why                  # 0.010 + 0.005 > 0.012
    assert _allow(cfg, st, sym="NAS100.vx", group="Indexes", risk=0.002)[0]


def test_same_symbol_twice_blocked():
    cfg, st = pg.GuardConfig(), pg.GuardState()
    pg.on_open(st, symbol="EURUSD.vx", group="FX Majors", risk_frac=0.005)
    assert not _allow(cfg, st)[0]


def test_stoploss_guard_pauses_symbol():
    cfg = pg.GuardConfig(sl_count=2, sl_lookback_hours=48, sl_pause_hours=24)
    st = pg.GuardState()
    for k in range(2):
        pg.on_open(st, symbol="EURUSD.vx", group="FX Majors", risk_frac=0.005)
        pg.on_close(cfg, st, symbol="EURUSD.vx", now=T0 + timedelta(hours=k), pnl=-50, equity_after=10_000 - 50 * (k + 1), stopped=True)
    ok, why = _allow(cfg, st, now=T0 + timedelta(hours=5))
    assert not ok and "paused" in why
    assert _allow(cfg, st, now=T0 + timedelta(hours=26))[0]
    assert _allow(cfg, st, sym="GBPUSD.vx", now=T0 + timedelta(hours=5))[0]   # other symbols unaffected


def test_daily_loss_limit_resets_next_day():
    cfg = pg.GuardConfig(daily_loss_limit=0.04)
    st = pg.GuardState()
    pg.new_day_if_needed(st, T0, 10_000)
    pg.on_open(st, symbol="EURUSD.vx", group="FX Majors", risk_frac=0.005)
    pg.on_close(cfg, st, symbol="EURUSD.vx", now=T0, pnl=-400, equity_after=9_600, stopped=True)
    assert not _allow(cfg, st, sym="GBPUSD.vx", eq=9_600)[0]
    assert _allow(cfg, st, sym="GBPUSD.vx", eq=9_600, now=T0 + timedelta(days=1))[0]


def test_drawdown_pause_blocks_all_then_expires():
    cfg = pg.GuardConfig(max_drawdown_pause=0.08, dd_pause_hours=72)
    st = pg.GuardState()
    pg.new_day_if_needed(st, T0, 10_000)
    ok, why = _allow(cfg, st, eq=9_100)
    assert not ok and "drawdown" in why
    assert not _allow(cfg, st, sym="NAS100.vx", group="Indexes", eq=9_100, now=T0 + timedelta(hours=10))[0]
    assert _allow(cfg, st, sym="NAS100.vx", group="Indexes", eq=9_100, now=T0 + timedelta(hours=73))[0]


def test_cooldown():
    cfg = pg.GuardConfig(cooldown_hours=6)
    st = pg.GuardState()
    pg.on_open(st, symbol="EURUSD.vx", group="FX Majors", risk_frac=0.005)
    pg.on_close(cfg, st, symbol="EURUSD.vx", now=T0, pnl=20, equity_after=10_020, stopped=False)
    assert not _allow(cfg, st, now=T0 + timedelta(hours=3))[0]
    assert _allow(cfg, st, now=T0 + timedelta(hours=7))[0]
