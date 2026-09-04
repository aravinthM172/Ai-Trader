"""
Runs ON THE ORACLE CLOUD BOX (no MT5, no PC).  Pulls fresh H1 candles for BTC and
XAU from free public sources and writes them in the tools/export_h1.py format, so
execution/paper_replay.py consumes them unchanged.

Use this when your Windows PC (the only machine with MT5) is off for the whole
forward-test window.

    python -m tools.fetch_h1_public --asset BTC
    python -m tools.fetch_h1_public --asset XAU
    python -m tools.fetch_h1_public --asset ALL          # both

How it works
------------
* The IN-SAMPLE prefix stays the real Valetax data (data/<sym>_H1*.csv, scp'd up
  once).  Only the bars AFTER the validation cutoff -- the actual forward test --
  come from the public feed.
* Public H1 OHLC for BTCUSD / XAUUSD tracks the broker feed closely.  The venue
  price *level* can differ (crypto exchange vs broker, gold futures vs spot), so
  the public block is shifted by a single additive constant to match the last
  real close at the seam -- every bar-to-bar move (what the strategy uses) is
  preserved, the basis jump is removed.  The offset is logged.
* Spread is not published anywhere, so the `spread` column is set to the Valetax
  broker floor (BTC 2976 pt / $29.76 fixed; XAU 30 pt / $0.30) -- the same
  conservative number the validation used.  Nothing is ever set below it.

Only stdlib + pandas/numpy -- no new dependency, no API key.
"""
from __future__ import annotations

import argparse
import json
import sys
import http.cookiejar
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.logging_setup import get_logger

log = get_logger("h1.fetch_public")
OUT = ROOT / "data" / "export"
OUT.mkdir(parents=True, exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

_OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
_yahoo_primed = False


def _prime_yahoo() -> None:
    """Yahoo's chart API 429s hard without a session cookie; one warm-up GET fixes it."""
    global _yahoo_primed
    if _yahoo_primed:
        return
    try:
        _OPENER.open(urllib.request.Request("https://finance.yahoo.com/", headers=_UA), timeout=15).read()
    except Exception as e:              # noqa: BLE001
        log.warning("yahoo cookie prime failed (continuing anyway): %s", e)
    _yahoo_primed = True

_ASSETS = {
    "BTC": dict(
        slug="btcusd_vx", symbol="BTCUSD.vx",
        dense="data/btcusd_vx_H1_dense.csv",
        spread_points=2976,          # $29.76 fixed Valetax BTC spread
        sources=["binance:BTCUSDT", "yahoo:BTC-USD"]),
    "XAU": dict(
        slug="xauusd_vx", symbol="XAUUSD.vx",
        dense="data/xauusd_vx_H1.csv",
        spread_points=30,            # $0.30 conservative gold floor
        # PAXG (Pax Gold) is redeemable 1:1 for a troy oz of gold -> PAXGUSDT ~= XAUUSD,
        # trades 24/7 on Binance, no key.  The level-match offset absorbs the tiny basis.
        sources=["binance:PAXGUSDT", "yahoo:XAUUSD=X", "yahoo:GC=F"]),
}

_HL = ("open", "high", "low", "close")


def _get(url: str, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with _OPENER.open(req, timeout=30) as r:
                return r.read()
        except Exception as e:              # noqa: BLE001 - report and retry
            last = e
            log.warning("GET failed (%d/%d) %s -- %s", k + 1, tries, url.split("?")[0], e)
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"all {tries} attempts failed for {url.split('?')[0]}: {last}")


def _fetch_binance(pair: str, start_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cur = start_ms
    now_ms = int(time.time() * 1000)
    while cur < now_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol={pair}"
               f"&interval=1h&startTime={cur}&limit=1000")
        batch = json.loads(_get(url))
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 3_600_000
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame(columns=["time", *_HL, "tick_volume"])
    d = pd.DataFrame(rows, columns=["ot", "o", "h", "l", "c", "v", *range(6)])
    return pd.DataFrame({
        "time": pd.to_datetime(d["ot"].astype("int64"), unit="ms", utc=True),
        "open": d["o"].astype(float), "high": d["h"].astype(float),
        "low": d["l"].astype(float), "close": d["c"].astype(float),
        "tick_volume": d["v"].astype(float).round().astype("int64"),
    })


def _fetch_yahoo(ticker: str, start_ms: int) -> pd.DataFrame:
    _prime_yahoo()
    span_days = max(7, int((time.time() * 1000 - start_ms) / 86_400_000) + 3)
    rng = f"{min(span_days, 720)}d"
    err = None
    for host in ("query1", "query2"):
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(ticker)}?interval=1h&range={rng}")
        try:
            j = json.loads(_get(url))
            res = j["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            d = pd.DataFrame({
                "time": pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True),
                "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                "tick_volume": q.get("volume", [0] * len(ts)),
            }).dropna(subset=list(_HL)).reset_index(drop=True)
            d["tick_volume"] = pd.to_numeric(d["tick_volume"], errors="coerce").fillna(0).round().astype("int64")
            # snap to the top of the hour (yahoo sometimes returns :00:01 etc.)
            d["time"] = d["time"].dt.floor("h")
            return d.drop_duplicates("time")
        except Exception as e:              # noqa: BLE001
            err = e
            log.warning("yahoo %s via %s failed: %s", ticker, host, e)
    raise RuntimeError(f"yahoo fetch failed for {ticker}: {err}")


def _fetch(source: str, start_ms: int) -> pd.DataFrame:
    kind, sym = source.split(":", 1)
    if kind == "binance":
        return _fetch_binance(sym, start_ms)
    if kind == "yahoo":
        return _fetch_yahoo(sym, start_ms)
    raise ValueError(f"unknown source '{source}'")


def build_export(asset: str) -> dict:
    a = _ASSETS[asset]
    dense_p = ROOT / a["dense"]
    if not dense_p.exists():
        raise SystemExit(f"missing {a['dense']} -- scp the validated dataset to the cloud box once")

    dense = pd.read_csv(dense_p, usecols=lambda c: c in ("time", *_HL, "tick_volume"))
    dense["time"] = pd.to_datetime(dense["time"], utc=True)
    dense = dense.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    cutoff = dense["time"].iloc[-1]
    if "tick_volume" not in dense.columns:
        dense["tick_volume"] = 0
    log.info("[%s] validation cutoff = %s (%d in-sample bars)", asset, cutoff, len(dense))

    start_ms = int(cutoff.timestamp() * 1000) - 7 * 3_600_000   # small overlap for level-matching
    pub = pd.DataFrame()
    used = None
    for src in a["sources"]:
        try:
            pub = _fetch(src, start_ms)
        except Exception as e:              # noqa: BLE001
            log.warning("[%s] source %s unusable: %s", asset, src, e)
            continue
        if len(pub) and pub["time"].max() > cutoff:
            used = src
            break
    if used is None or pub.empty:
        raise SystemExit(f"[{asset}] no public source returned usable H1 data past {cutoff}")

    pub = pub[["time", *_HL, "tick_volume"]].drop_duplicates("time").sort_values("time").reset_index(drop=True)

    # ---- level-match: shift the public block to meet the last real close ----
    overlap = pub[pub["time"] <= cutoff]
    ref_pub = overlap["close"].iloc[-1] if len(overlap) else pub["close"].iloc[0]
    offset = float(dense["close"].iloc[-1] - ref_pub)
    for col in _HL:
        pub[col] = pub[col].astype(float) + offset
    log.info("[%s] source=%s  level offset applied = %+.2f  (public %.2f -> broker %.2f)",
             asset, used, offset, ref_pub, ref_pub + offset)

    fwd = pub[pub["time"] > cutoff].reset_index(drop=True)
    if fwd.empty:
        raise SystemExit(f"[{asset}] public feed has no bars after {cutoff} yet -- try again later")

    combined = pd.concat([dense[["time", *_HL, "tick_volume"]], fwd], ignore_index=True)
    combined = combined.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    combined["spread"] = a["spread_points"]

    csv_path = OUT / f"{a['slug']}_H1_export.csv"
    combined.to_csv(csv_path, index=False)

    meta = {
        "export_utc": datetime.now(timezone.utc).isoformat(),
        "asset": asset, "symbol": a["symbol"],
        "source": f"PUBLIC FEED ({used}) -- PC/MT5 was unavailable",
        "public_source": used,
        "level_offset_applied": round(offset, 4),
        "bars": int(len(combined)),
        "in_sample_bars": int((combined["time"] <= cutoff).sum()),
        "forward_bars": int((combined["time"] > cutoff).sum()),
        "validation_cutoff_utc": str(cutoff),
        "first_bar_utc": str(combined["time"].iloc[0]),
        "last_bar_utc": str(combined["time"].iloc[-1]),
        "spread_synthetic": True,
        "spread_note": (f"public feed has no spread; column set to the Valetax broker floor "
                        f"({a['spread_points']} pt) -- conservative, never below the broker minimum"),
    }
    (OUT / f"{a['slug']}_H1_export.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("=" * 84)
    print(f"{a['symbol']} H1 PUBLIC EXPORT  --  NO MT5, NO ORDER SENT")
    print("=" * 84)
    print(f"source      : {used}   (level offset {offset:+.2f})")
    print(f"bars        : {len(combined):,}   {combined['time'].iloc[0]} -> {combined['time'].iloc[-1]}")
    print(f"cutoff      : {cutoff}")
    print(f"forward     : {meta['forward_bars']} bars past the validation cutoff  <-- the real test")
    print(f"spread      : {a['spread_points']} pt (synthetic broker floor; public feeds carry no spread)")
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    print(f"files       : {_rel(csv_path)}")
    print(f"              {_rel(OUT / (a['slug'] + '_H1_export.json'))}")
    print()
    ra = "" if asset == "BTC" else " --asset XAU"
    print("now replay it:")
    print(f"  python run_btc.py --paper --timeframe H1{ra} "
          f"--from-csv data/export/{a['slug']}_H1_export.csv --paper-balance 1500")
    print("=" * 84)
    log.info("[%s] wrote %s  (%d bars, %d forward)", asset, csv_path.name, len(combined), meta["forward_bars"])
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["BTC", "XAU", "ALL"], default="ALL")
    args = ap.parse_args()
    assets = ["BTC", "XAU"] if args.asset == "ALL" else [args.asset]
    rc = 0
    for a in assets:
        try:
            build_export(a)
        except SystemExit as e:
            print(f"[{a}] {e}")
            rc = 1
        except Exception as e:              # noqa: BLE001
            log.exception("[%s] failed: %s", a, e)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
