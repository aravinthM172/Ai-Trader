"""
Download verified 1-minute history from zip archives for the five live markets (owner request 2026-10-21: "data from a
verified source in zip files, not Valetax or FundingPips").

    python -m tools.fetch_zip_history binance BTCUSDT ETHUSDT      # data.binance.vision monthly klines, SHA-256 checked
    python -m tools.fetch_zip_history histdata XAUUSD USDJPY NSXUSD # histdata.com yearly / monthly M1 ASCII zips

Raw zips are kept in data/zips/<source>/<symbol>/ (resumable: existing, verified files are skipped).
Binance: every monthly zip is checked against its official .CHECKSUM (SHA-256); a mismatch deletes the file.
HistData: the free download is a small form (token from the year / month page, then POST to get.php).
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "zips"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get(url: str, data: bytes | None = None, headers: dict | None = None, tries: int = 4) -> bytes | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def binance(sym: str) -> None:
    d = OUT / "binance" / sym
    d.mkdir(parents=True, exist_ok=True)
    base = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/"
    months = pd.period_range("2017-08", pd.Timestamp.utcnow().to_period("M") - 1, freq="M")
    ok = bad = miss = 0
    for m in months:
        name = f"{sym}-1m-{m}.zip"
        f = d / name
        if f.exists() and (d / (name + ".ok")).exists():
            ok += 1
            continue
        z = get(base + name)
        if z is None:
            miss += 1
            continue
        chk = get(base + name + ".CHECKSUM")
        want = chk.decode().split()[0] if chk else None
        if want and hashlib.sha256(z).hexdigest() != want:
            bad += 1
            print(f"  CHECKSUM MISMATCH {name} -- discarded", flush=True)
            continue
        f.write_bytes(z)
        (d / (name + ".ok")).write_text(want or "no checksum published")
        ok += 1
        time.sleep(0.3)
    print(f"binance {sym}: {ok} months verified, {bad} checksum failures, {miss} not published", flush=True)


def histdata(sym: str) -> None:
    d = OUT / "histdata" / sym
    d.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.utcnow().year
    targets = [(y, None) for y in range(2000, this_year)] + [(this_year, m) for m in range(1, pd.Timestamp.utcnow().month + 1)]
    got = miss = 0
    for y, m in targets:
        name = f"HISTDATA_COM_ASCII_{sym}_M1{y}{'' if m is None else f'{m:02d}'}.zip"
        f = d / name
        if f.exists() and f.stat().st_size > 1000:
            got += 1
            continue
        page_url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{sym.lower()}/{y}" + \
                   ("" if m is None else f"/{m}")
        page = get(page_url)
        if not page:
            miss += 1
            continue
        html = page.decode("utf-8", "replace")
        form = re.search(r'<form[^>]*id="file_down"[^>]*>(.*?)</form>', html, re.S)
        if not form:
            miss += 1                                           # no file for that year / month
            continue
        fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', form.group(1)))
        z = get("https://www.histdata.com/get.php", urllib.parse.urlencode(fields).encode(),
                {"Referer": page_url, "Content-Type": "application/x-www-form-urlencoded"})
        if not z or z[:2] != b"PK":
            miss += 1
            continue
        f.write_bytes(z)
        got += 1
        time.sleep(1.0)
    print(f"histdata {sym}: {got} zip files, {miss} periods not available", flush=True)


def main() -> int:
    src, syms = sys.argv[1], sys.argv[2:]
    for s in syms:
        (binance if src == "binance" else histdata)(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
