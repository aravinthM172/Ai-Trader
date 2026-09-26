"""
Long-history BTC/USD H1 research dataset from Bitstamp 1-minute data (2012 -> today).

Valetax BTCUSD.vx only keeps real hourly bars from 2024-01 (see btc_h1_dataset.py),
which is ONE macro cycle.  This builds ~12 years of H1 from the public Bitstamp
minute history so the frozen strategy can be tested across 2014-2023 regimes it
never saw (2014-15 bear, 2017 bubble, 2018 crash, 2020-21 bull, 2022 bear).

Source (no login): github.com/ff137/bitstamp-btcusd-minute-data, updated daily.
    data/raw/btcusd_bitstamp_1min_2012-2025.csv.gz   (Kaggle-derived, CC BY-SA 4.0:
        "Zielak (mczielinski), Bitcoin Historical Data, Kaggle")
    data/raw/btcusd_bitstamp_1min_latest.csv         (API updates 2025-01-07 ->)
    data/raw/btcusd_bitstamp_provenance.csv          (flagged outages)
Data stays local (CC BY-SA ShareAlike) -- data/raw/ is gitignored.

    python -m backtest.btc_bitstamp_h1_dataset            # download + build + cross-check
    python -m backtest.btc_bitstamp_h1_dataset --no-download

Writes:
    data/btcusd_bitstamp_H1.csv      (same columns as btcusd_vx_H1_dense.csv)
    reports/btc_bitstamp_h1_dataset.json

Bars are never fabricated: the upstream fills missing minutes with flat zero-volume
candles, so any hour with ZERO traded volume is dropped (leaves a gap, like the
Valetax weekend gaps).  spread = 0 (Bitstamp has no broker spread) -- the
long-history validation applies a price-proportional Valetax spread instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger

log = get_logger("btc.bitstamp")
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
OUT = ROOT / "data" / "btcusd_bitstamp_H1.csv"
VALETAX = ROOT / "data" / "btcusd_vx_H1_dense.csv"

BASE = "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/data"
FILES = {
    "btcusd_bitstamp_1min_2012-2025.csv.gz": f"{BASE}/historical/btcusd_bitstamp_1min_2012-2025.csv.gz",
    "btcusd_bitstamp_1min_latest.csv": f"{BASE}/updates/btcusd_bitstamp_1min_latest.csv",
    "btcusd_bitstamp_provenance.csv": f"{BASE}/provenance/btcusd_bitstamp_1min.csv",
}
# 2012-13 is too illiquid on Bitstamp (upstream README); start once 30-min dead spells are rare
START = "2014-01-01"


def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = RAW / name
        # historical bulk file is frozen upstream; the update file grows daily
        if dst.exists() and not force and "latest" not in name:
            continue
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, dst)


def load_minutes() -> pd.DataFrame:
    parts = [pd.read_csv(RAW / n) for n in ("btcusd_bitstamp_1min_2012-2025.csv.gz",
                                            "btcusd_bitstamp_1min_latest.csv")]
    m = pd.concat(parts, ignore_index=True)
    m = m.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    m["time"] = pd.to_datetime(m["timestamp"], unit="s", utc=True)
    return m


def to_h1(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m = m[m["time"] >= pd.Timestamp(START, tz="UTC")]
    traded = m["volume"] > 0
    g = m.set_index("time").groupby(pd.Grouper(freq="1h", label="left", closed="left"))
    h1 = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(),
        "tick_volume": traded.groupby(m["time"].dt.floor("1h")).sum(),   # minutes with trades
        "real_volume": g["volume"].sum(),                                  # BTC traded
    })
    h1["spread"] = 0
    n_all = len(h1)
    dead = h1["real_volume"] <= 0
    h1 = h1[~dead].dropna(subset=["open", "close"])
    # the last hour is only a finished bar if the source reaches its final minute
    last = m["time"].iloc[-1]
    if last.minute != 59:
        h1 = h1[h1.index < last.floor("1h")]
    h1 = h1.reset_index()[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    h1["tick_volume"] = h1["tick_volume"].astype(int)
    bad = int(((h1.high < h1[["open", "close"]].max(axis=1)) |
               (h1.low > h1[["open", "close"]].min(axis=1)) | (h1.low <= 0)).sum())
    gaps = h1["time"].diff().dt.total_seconds().div(3600).fillna(1)
    info = dict(
        start=str(h1.time.iloc[0]), end=str(h1.time.iloc[-1]), bars=int(len(h1)),
        span_years=round((h1.time.iloc[-1] - h1.time.iloc[0]).days / 365.25, 2),
        hours_in_span=int(n_all), dropped_zero_volume_hours=int(dead.sum()),
        gaps_over_1h=int((gaps > 1.5).sum()), largest_gap_hours=float(gaps.max()),
        ohlc_anomalies=bad,
        bars_per_year={int(y): int(n) for y, n in h1.groupby(h1.time.dt.year).size().items()},
        partial_hours_thin=int((h1.tick_volume < 10).sum()),
    )
    return h1, info


def valetax_true_utc(t: pd.Series) -> pd.Series:
    """
    The Valetax CSVs were converted with ONE server offset (+3h, detected at connect in
    summer), but the server runs EET/EEST (+2h winter / +3h summer, EU switch dates), so
    winter bars are labelled 1h early.  Undo that: label + 3h = server wall clock, then
    localise it as Europe/Athens.  Hours that don't exist or repeat at the clock change -> NaT.
    """
    server = (t + pd.Timedelta(hours=3)).dt.tz_localize(None)
    return (server.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
                  .dt.tz_convert("UTC"))


def _compare(v: pd.DataFrame, b: pd.DataFrame) -> dict:
    j = v[["close"]].join(b[["close"]], rsuffix="_b", how="inner")
    r = np.log(j).diff()
    rel = j.close / j.close_b - 1.0
    return dict(overlap_bars=int(len(j)), hourly_return_corr=round(float(r.close.corr(r.close_b)), 4),
                close_diff_pct_median=round(float(rel.median() * 100), 4),
                close_diff_pct_abs_p95=round(float(rel.abs().quantile(0.95) * 100), 4),
                close_diff_pct_abs_max=round(float(rel.abs().max() * 100), 4))


def cross_check(h1: pd.DataFrame) -> dict:
    """Compare against the Valetax feed where both exist (after undoing its DST mislabel)."""
    if not VALETAX.exists():
        return {"available": False}
    v = pd.read_csv(VALETAX)
    v["time"] = pd.to_datetime(v["time"], utc=True)
    b = h1.set_index("time")
    raw = _compare(v.set_index("time"), b)
    v["time"] = valetax_true_utc(v["time"])
    fixed = _compare(v.dropna(subset=["time"]).set_index("time"), b)
    return dict(
        available=True, overlap=[str(v.time.min()), str(v.time.max())],
        valetax_labels_as_stored=raw,
        valetax_labels_dst_corrected=fixed,
        valetax_dst_mislabel_detected=bool(fixed["hourly_return_corr"] > raw["hourly_return_corr"] + 0.2),
        same_market=bool(fixed["hourly_return_corr"] > 0.95 and abs(fixed["close_diff_pct_median"]) < 0.5),
        note="Valetax BTCUSD.vx is a CFD quoted off spot; small level differences vs Bitstamp are expected.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    if not args.no_download:
        download()
    m = load_minutes()
    log.info("minutes=%d  %s -> %s", len(m), m.time.iloc[0], m.time.iloc[-1])
    h1, info = to_h1(m)
    h1.to_csv(OUT, index=False)
    xc = cross_check(h1)
    rep = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
               source="github.com/ff137/bitstamp-btcusd-minute-data (Bitstamp BTC/USD 1-min)",
               license="pre-2025-01-07 rows CC BY-SA 4.0: Zielak (mczielinski), Bitcoin Historical Data, Kaggle",
               minutes=int(len(m)), h1=info, valetax_cross_check=xc)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "btc_bitstamp_h1_dataset.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    print("=" * 94)
    print("BTC/USD Bitstamp H1 LONG-HISTORY DATASET  --  research only, NO LIVE ORDER")
    print("=" * 94)
    print(f"H1 bars    : {info['bars']:,}  {info['start'][:16]} -> {info['end'][:16]}  ({info['span_years']}y)")
    print(f"dropped    : {info['dropped_zero_volume_hours']} zero-volume hours (not fabricated)  "
          f"gaps>1h {info['gaps_over_1h']}  largest {info['largest_gap_hours']:.0f}h  "
          f"OHLC anomalies {info['ohlc_anomalies']}")
    print(f"per year   : {info['bars_per_year']}")
    if xc.get("available"):
        for lbl, k in (("as stored", "valetax_labels_as_stored"), ("DST-fixed", "valetax_labels_dst_corrected")):
            x = xc[k]
            print(f"vs Valetax ({lbl:9s}): {x['overlap_bars']:,} bars  return corr {x['hourly_return_corr']}  "
                  f"close diff median {x['close_diff_pct_median']}%  |p95| {x['close_diff_pct_abs_p95']}%")
        print(f"             same market={xc['same_market']}  Valetax DST mislabel={xc['valetax_dst_mislabel_detected']}")
    print(f"dataset: {OUT.relative_to(ROOT)}   report: reports/btc_bitstamp_h1_dataset.json")
    print("=" * 94)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
