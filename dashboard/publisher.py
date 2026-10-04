"""
Runs on the TRADING PC.  Every PUSH_SECONDS it collects a read-only snapshot
(dashboard/data.gather) and POSTs it to the cloud dashboard.  Outbound only -- nothing in the
cloud can reach this PC, MT5 or the bot.  If the cloud is down, the push fails quietly and
the bot is unaffected.

.env (or .env.dashboard, git-ignored) on the PC:
    DASHBOARD_URL=https://<your-host>          (e.g. https://141-1-2-3.sslip.io)
    DASHBOARD_PUSH_TOKEN=<long random secret>  (same value on the cloud server)

    python -m dashboard.publisher            # loop
    python -m dashboard.publisher --once     # one push (test)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from common.logging_setup import get_logger
from dashboard.data import gather

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[1] / ".env.dashboard")   # dashboard-only settings (git-ignored)
log = get_logger("dashboard.pub", filename="dashboard_publisher.log")
PUSH_SECONDS = int(float(os.getenv("DASHBOARD_PUSH_SECONDS", 60)))
TICK_SECONDS = float(os.getenv("DASHBOARD_TICK_SECONDS", 3))
TICK_SYMBOLS = [x.strip() for x in os.getenv("MULTI_SYMBOLS", "BTCUSD.vx,XAUUSD.vx,DAX40.vx").split(",") if x.strip()]


def push(snapshot: dict, *, url: str, token: str, timeout: float = 20.0, path: str = "/api/push") -> int:
    body = gzip.compress(json.dumps(snapshot, default=str).encode())
    req = urllib.request.Request(url.rstrip("/") + path, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json", "Content-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    url, token = os.getenv("DASHBOARD_URL", "").strip(), os.getenv("DASHBOARD_PUSH_TOKEN", "").strip()
    if not (url and token):
        print("set DASHBOARD_URL and DASHBOARD_PUSH_TOKEN in .env")
        return 1
    gw, last_full = None, 0.0
    while True:
        if time.time() - last_full >= PUSH_SECONDS or a.once:
            last_full = time.time()
            try:
                if gw is not None:                      # gather() opens its own connection
                    gw.shutdown()
                    gw = None
                code = push(gather(), url=url, token=token)
                log.info("pushed snapshot -> %s", code)
                if a.once:
                    print(f"pushed -> HTTP {code}")
            except Exception as e:
                log.warning("push failed: %s", e)
                if a.once:
                    print(f"push failed: {e}")
            if a.once:
                return 0
        try:
            if gw is None:
                from mt5.gateway import MT5Gateway
                gw = MT5Gateway()
                if not gw.connect():
                    gw = None
            if gw is not None:
                push(ticks(gw), url=url, token=token, path="/api/tick", timeout=8.0)
        except Exception as e:
            log.debug("tick push failed: %s", e)
            gw = None
        time.sleep(max(1.0, TICK_SECONDS))


def ticks(gw) -> dict:
    """Latest bid/ask per symbol (small, sent every few seconds for the live chart)."""
    out = {"time": time.time(), "symbols": {}}
    try:
        acct = gw.account_info()
        pos = gw.positions()
        out["account"] = {"equity": acct.get("equity"), "balance": acct.get("balance"),
                          "floating": round(sum(p["profit"] for p in pos), 2), "open": len(pos)}
        out["positions"] = [{"symbol": p["symbol"], "type": p["type"], "volume": p["volume"],
                             "price_open": p["price_open"], "sl": p["sl"], "tp": p["tp"], "profit": p["profit"]} for p in pos]
    except Exception:
        pass
    for s in TICK_SYMBOLS:
        t = gw.get_tick(s)
        if t and t.get("bid"):
            out["symbols"][s] = {"bid": t["bid"], "ask": t["ask"], "time_utc": t.get("time_utc"),
                                 "age_s": t.get("age_seconds")}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
