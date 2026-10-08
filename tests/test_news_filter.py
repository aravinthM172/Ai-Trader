from datetime import datetime, timedelta, timezone

from news.filter import NewsFilter, symbol_currencies

NOW = datetime(2026, 10, 7, 12, 25, tzinfo=timezone.utc)
FEED = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-10-07T08:30:00-04:00", "impact": "High"},     # 12:30 UTC
    {"title": "German Ifo", "country": "EUR", "date": "2026-10-07T08:00:00+00:00", "impact": "High"},
    {"title": "Retail Sales", "country": "GBP", "date": "2026-10-07T12:30:00+00:00", "impact": "Medium"},
]


def _nf(tmp_path, fetcher=lambda: FEED, **kw):
    return NewsFilter(cache_path=tmp_path / "cal.json", fetcher=fetcher, **kw)


def test_symbol_currency_mapping():
    assert symbol_currencies("EURUSD.vx") == {"EUR", "USD"}
    assert symbol_currencies("NAS100.vx") == {"USD"}
    assert symbol_currencies("DAX40.vx") == {"EUR"}
    assert symbol_currencies("GER40") == {"EUR"}           # FundingPips name
    assert symbol_currencies("NDX100") == {"USD"}
    assert symbol_currencies("JP225") == {"JPY"}
    assert symbol_currencies("FTSE100") == {"GBP"}
    assert symbol_currencies("USDJPY") == {"USD", "JPY"}
    assert symbol_currencies("XAUUSD.vx") == {"USD"}
    assert symbol_currencies("BTCUSD.vx") == {"USD"}
    assert symbol_currencies("GBPJPY.vx") == {"GBP", "JPY"}
    assert symbol_currencies("XTIUSD.vx") == {"USD"}


def test_blocks_only_affected_symbols_in_window(tmp_path):
    nf = _nf(tmp_path)
    assert nf.is_blocked("EURUSD.vx", NOW)                   # USD CPI in 5 min
    assert nf.is_blocked("BTCUSD.vx", NOW)
    assert not nf.is_blocked("GBPJPY.vx", NOW)               # GBP event is only Medium
    assert not nf.is_blocked("DAX40.vx", NOW)                # EUR event was 4 h ago
    assert not nf.is_blocked("EURUSD.vx", NOW - timedelta(hours=2))
    assert nf.is_blocked("EURUSD.vx", NOW + timedelta(minutes=30))     # 25 min after CPI
    assert not nf.is_blocked("EURUSD.vx", NOW + timedelta(minutes=40))


def test_disabled_never_blocks(tmp_path):
    assert not _nf(tmp_path, enabled=False).is_blocked("EURUSD.vx", NOW)


def _boom():
    raise OSError("offline")


def test_feed_down_without_cache_fail_open_and_closed(tmp_path):
    assert not _nf(tmp_path, fetcher=_boom, fail_closed=False).is_blocked("EURUSD.vx", NOW)
    assert _nf(tmp_path, fetcher=_boom, fail_closed=True).is_blocked("EURUSD.vx", NOW)


def test_uses_fresh_cache_when_feed_down(tmp_path):
    _nf(tmp_path).get_events(NOW)                            # populate cache
    nf = _nf(tmp_path, fetcher=_boom, fail_closed=True)
    assert nf.is_blocked("EURUSD.vx", NOW + timedelta(hours=1)) is False   # cache fresh, CPI window over
    assert nf.is_blocked("EURUSD.vx", NOW)                   # still knows about CPI from cache
