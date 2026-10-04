"""
Runs on the TRADING PC.  Every PUSH_SECONDS it collects a read-only snapshot
(dashboard/data.gather) and POSTs it to the cloud dashboard.  Outbound only -- nothing in the
cloud can reach this PC, MT5 or the bot.  If the cloud is down, the push fails quietly and
the bot is unaffected.

.env on the PC:
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

from dotenv import load_dotenv

from common.logging_setup import get_logger
from dashboard.data import gather

load_dotenv()
log = get_logger("dashboard.pub", filename="dashboard_publisher.log")
PUSH_SECONDS = int(float(os.getenv("DASHBOARD_PUSH_SECONDS", 60)))


def push(snapshot: dict, *, url: str, token: str, timeout: float = 20.0) -> int:
    body = gzip.compress(json.dumps(snapshot, default=str).encode())
    req = urllib.request.Request(url.rstrip("/") + "/api/push", data=body, method="POST", headers={
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
    while True:
        try:
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
        time.sleep(max(15, PUSH_SECONDS))


if __name__ == "__main__":
    raise SystemExit(main())
