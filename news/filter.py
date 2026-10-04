"""
Economic-calendar news filter.

Source: ForexFactory weekly calendar feed (https://nfs.faireconomy.media/ff_calendar_thisweek.json),
cached in data/calendar_cache.json and refreshed every REFRESH_HOURS.

A symbol is blocked from opening new positions from `before_minutes` before until
`after_minutes` after a HIGH-impact event in any currency that moves it
(EURUSD -> EUR, USD;  NAS100 -> USD;  XAUUSD -> USD;  BTCUSD -> USD ...).

If the feed cannot be fetched and the cache is older than STALE_HOURS:
  fail_closed=True  -> block everything (use for prop-firm accounts with news rules)
  fail_closed=False -> allow (no news protection; logged)

    python -m news.filter            # print upcoming high-impact events + blocked symbols

Note: the feed covers the current week only, so this filter cannot be backtested
historically from this source.  It is a live risk control, off in research.
"""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common.logging_setup import get_logger

log = get_logger("news")
FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE = Path(__file__).resolve().parents[1] / "data" / "calendar_cache.json"
REFRESH_HOURS = 6
STALE_HOURS = 24 * 7

_INDEX_CCY = {"NAS100": "USD", "SP500": "USD", "US30": "USD", "DXY": "USD", "DAX40": "EUR", "EU50": "EUR",
              "FRA40": "EUR", "UK100": "GBP", "JPN225": "JPY", "AUS200": "AUD", "HK50": "CNY"}
_CCYS = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CNY"}


def symbol_currencies(symbol: str) -> set[str]:
    """Currencies whose high-impact news should block this symbol."""
    s = re.sub(r"\..*$", "", symbol.upper())          # strip broker suffix (.vx)
    if s in ("BTC", "XAU", "GOLD"):
        return {"USD"}
    if s in _INDEX_CCY:
        return {_INDEX_CCY[s]}
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return {"USD", s[3:6]} & _CCYS or {"USD"}
    if s.startswith(("XTI", "XBR", "XNG")):
        return {"USD"}
    if len(s) == 6 and s[:3] in _CCYS and s[3:] in _CCYS:
        return {s[:3], s[3:]}
    if s.endswith(("USD", "EUR")):                    # crypto: BTCUSD, ETHUSD, BTCEUR ...
        return {s[-3:]}
    return {"USD"}


def _parse(raw: list[dict]) -> list[dict]:
    out = []
    for e in raw:
        try:
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        except Exception:
            continue
        out.append({"time": t, "currency": e.get("country", ""), "impact": (e.get("impact") or "").upper(),
                    "event": e.get("title", "")})
    return out


def _fetch(timeout: float = 15.0) -> list[dict]:
    req = urllib.request.Request(FEED, headers={"User-Agent": "gold-ai-trader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class NewsFilter:

    def __init__(self, enabled=True, before_minutes=15, after_minutes=30, *, fail_closed=False,
                 cache_path: Path = CACHE, fetcher=_fetch):
        self.enabled = enabled
        self.before_minutes = before_minutes
        self.after_minutes = after_minutes
        self.fail_closed = fail_closed
        self.cache_path = cache_path
        self.fetcher = fetcher
        self._stale = False

    # -- data ----------------------------------------------------------------
    def _load_cache(self) -> dict | None:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_events(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        cache = self._load_cache()
        age_h = None
        if cache:
            age_h = (now - datetime.fromisoformat(cache["fetched_utc"])).total_seconds() / 3600
        if cache is None or age_h is None or age_h > REFRESH_HOURS:
            try:
                raw = self.fetcher()
                cache = {"fetched_utc": now.isoformat(), "events": raw}
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(cache), encoding="utf-8")
                age_h = 0.0
            except Exception as e:
                log.warning("calendar fetch failed (%s); using cache aged %s h", e,
                            None if age_h is None else round(age_h, 1))
        self._stale = cache is None or age_h is None or age_h > STALE_HOURS
        return _parse(cache["events"]) if cache else []

    # -- decisions -------------------------------------------------------------
    def blocking_event(self, symbol: str, now: datetime | None = None) -> dict | None:
        """The high-impact event currently blocking `symbol`, or None."""
        now = now or datetime.now(timezone.utc)
        ccys = symbol_currencies(symbol)
        for ev in self.get_events(now):
            if ev["impact"] != "HIGH" or not (ev["currency"] in ccys or ev["currency"] == "ALL"):
                continue
            mins = (ev["time"] - now).total_seconds() / 60
            if -self.after_minutes <= mins <= self.before_minutes:
                return ev
        return None

    def is_blocked(self, asset, now: datetime | None = None) -> bool:
        """True if new entries on `asset` (symbol or asset key) must wait."""
        if not self.enabled:
            return False
        ev = self.blocking_event(str(asset), now)
        if self._stale:
            log.warning("calendar data stale/unavailable -> %s", "BLOCK (fail_closed)" if self.fail_closed else "allow")
            return self.fail_closed
        return ev is not None

    def upcoming(self, hours: float = 48, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        return [e for e in self.get_events(now)
                if e["impact"] == "HIGH" and now - timedelta(minutes=self.after_minutes) <= e["time"] <= now + timedelta(hours=hours)]


if __name__ == "__main__":
    nf = NewsFilter()
    up = nf.upcoming(hours=24 * 7)
    print(f"{len(up)} high-impact events in the next 7 days:")
    for e in up:
        print(f"  {e['time']:%a %d %b %H:%M} UTC  {e['currency']:4} {e['event']}")
