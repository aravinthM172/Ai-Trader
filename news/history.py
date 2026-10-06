"""
Historical high-impact US event times, 2005 -> 2026, for research (event studies, filter backtests).

  FOMC  rate decisions, parsed from federalreserve.gov (fomchistorical<YYYY>.htm for <= 2020,
        fomccalendars.htm after).  Scheduled meetings only; decision = last meeting day,
        statement 14:00 ET (14:15 ET before 2013 -- same H1 bar).
  NFP   US Employment Situation, 08:30 ET.  BLS blocks scripted access to its release archive, so
        dates follow the BLS rule: the third Friday after the reference week (the Sun-Sat week
        containing the 12th).  Holiday/shutdown exceptions are NOT modelled -- validate() measures
        how often the computed hour really shows a volatility spike.

    python -m news.history          # build data/events_history.csv and print the NFP validation

Research only; the live news filter is news/filter.py.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "events_history.csv"
ET = ZoneInfo("America/New_York")
_H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}
_M = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
      "November", "December"]
_MI = {m: i + 1 for i, m in enumerate(_M)} | {m[:3]: i + 1 for i, m in enumerate(_M)} | {"Sept": 9}


def _get(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_H), timeout=30).read().decode("utf-8", "ignore")


def fomc_dates(first=2005, last=2026) -> list[date]:
    out = set()
    months = "|".join(_M + [m[:3] for m in _M] + ["Sept"])      # "Feb 31-1 Meeting" = Jan 31 - Feb 1
    for y in range(first, 2021):
        h = _get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
        for m1, d1, m2, d2 in re.findall(rf"({months}) (\d{{1,2}})(?:-(?:({months}) )?(\d{{1,2}}))? Meeting - {y}", h):
            out.add(date(y, _MI[m2 or m1], int(d2 or d1)))
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "|", _get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")))
    for part in re.split(r"(?=\b\d{4} FOMC Meetings)", txt):
        ym = re.match(r"(\d{4}) FOMC Meetings", part)
        if not ym or not (2021 <= int(ym.group(1)) <= last):
            continue
        y = int(ym.group(1))
        for mon, d1, d2 in re.findall(r"\|([A-Z][a-z]+(?:/[A-Z][a-z]+)?)\|\| \|(\d{1,2})(?:-(\d{1,2}))?\*?\|", part):
            m = mon.split("/")[-1]
            if m in _MI:
                out.add(date(y, _MI[m], int(d2 or d1)))
    return sorted(d for d in out if first <= d.year <= last)


def nfp_dates(first=2005, last=2026) -> list[date]:
    out = []
    for y in range(first, last + 1):
        for m in range(1, 13):
            d12 = date(y, m, 12)
            sat = d12 + timedelta(days=(5 - d12.weekday()) % 7)          # end of the Sun-Sat reference week
            fri = sat + timedelta(days=6) + timedelta(days=14)            # third Friday after it
            out.append(fri)
    return out


def build() -> pd.DataFrame:
    rows = [dict(event="FOMC", time_utc=datetime.combine(d, datetime.min.time().replace(hour=14), ET)) for d in fomc_dates()]
    rows += [dict(event="NFP", time_utc=datetime.combine(d, datetime.min.time().replace(hour=8, minute=30), ET)) for d in nfp_dates()]
    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df.time_utc, utc=True)
    df = df[df.time_utc <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=90)].sort_values("time_utc")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT)
    d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
    return d


def validate(h1: pd.DataFrame, events: pd.DataFrame, kind: str) -> dict:
    """Share of events whose H1 bar range is > 1.5x the median range of that UTC hour on the same weekday."""
    h = h1.assign(rng=h1.high - h1.low, hr=h1.time.dt.hour, wd=h1.time.dt.weekday).set_index("time")
    base = h.groupby(["wd", "hr"]).rng.median()
    ev = events[events.event == kind].time_utc.dt.floor("h")
    ev = ev[(ev >= h.index[0]) & (ev <= h.index[-1])]
    ratio = [h.rng.get(t) / base.get((t.weekday(), t.hour)) for t in ev if t in h.index]
    s = pd.Series(ratio).dropna()
    return dict(event=kind, n=len(s), spike_share=round(float((s > 1.5).mean()), 3), median_ratio=round(float(s.median()), 2))


if __name__ == "__main__":
    ev = build()
    print(ev.event.value_counts().to_dict(), ev.time_utc.min(), "->", ev.time_utc.max())
    h = pd.read_csv(ROOT / "data" / "mt5_H1" / "EURUSD.vx.csv")
    h["time"] = pd.to_datetime(h.time, utc=True)
    for k in ("NFP", "FOMC"):
        print(validate(h, ev, k))
