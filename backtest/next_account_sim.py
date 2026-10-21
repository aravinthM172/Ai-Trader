"""
Which sizing protects the next account, and which account to buy (owner request 2026-10-21, after the FundedNext
15K failed on manual gold trades).  Pending items 1 (loss-limit sizing), 10 (one crypto trade at a time) and 13
(new-account simulation) of reports/Complete trading technology map.md.

    python -m backtest.next_account_sim

Trade stream: the live bot's rules on the longest verified H1 data (backtest/tp_weekday_study: BTC ETH gold USDJPY
NDX100, target 6 ATR, stop 2 ATR, strong-signal + trend filters, NDX100 / USDJPY buy-only, weekend-flat entries),
live caps: max 4 open (1.4 % at 0.35 %), max 2 per group; crypto group 2 (today) or 1 ("one crypto at a time").
Event replay from 2018-03: every trade is sized at ENTRY from the balance at that moment, P/L lands at exit.
Limits use closed P/L only (floating losses are not seen), so breaches are slightly UNDER-counted -- compare variants.
Attempts start on the first day of every month and run up to 365 days.

Sizing variants
  S0  fixed 0.35 %, 2 crypto                       (the bot until today)
  S1  fixed 0.35 %, 1 crypto
  S2  auto risk 0.45 % (dashboard cushion: x clip((bal - floor) / room, 0.3, 1.5)), 2 crypto
  S3  auto risk 0.45 % + half risk within 25 % of the phase target + 1 crypto
  S4  auto risk 0.35 % + half risk near target + 1 crypto
Accounts (rules read first-hand 2026-10-13 .. 10-21)
  FN-2step-15K   reset $107.99: daily 5 % of initial, max 10 % static, targets 8 % then 5 %, 80 % split after
  FN-Lite-10K    $59.99: daily 4 % of initial, max 8 % static, targets 8 % then 4 %, 80 % split after
  CTI-Direct-20K $999: no daily limit, 6 % static from initial, funded now, 70 % split
  FTUK-Inst-10K  price not published: daily 5 %, 6 % trailing from the highest balance, funded now, 65 % split
Read-only.  Writes reports/next_account_sim.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import long_history_backtest as lh
from backtest import tp_weekday_study as tw

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
GROUP = {"BTCUSD": "crypto", "ETHUSD": "crypto", "XAUUSD": "comm", "NDX100": "idx", "USDJPY": "fx"}
FROM = pd.Timestamp("2018-03-01", tz="UTC")

SIZING = {
    "S0 fixed 0.35%, 2 crypto": dict(base=0.0035, cushion=False, taper=False, crypto=2),
    "S1 fixed 0.35%, 1 crypto": dict(base=0.0035, cushion=False, taper=False, crypto=1),
    "S2 auto 0.45%, 2 crypto": dict(base=0.0045, cushion=True, taper=False, crypto=2),
    "S3 auto 0.45% + taper + 1 crypto": dict(base=0.0045, cushion=True, taper=True, crypto=1),
    "S4 auto 0.35% + taper + 1 crypto": dict(base=0.0035, cushion=True, taper=True, crypto=1),
}
ACCOUNTS = {
    "FN-2step-15K": dict(size=15000, fee=107.99, daily=0.05, max_loss=0.10, trailing=False, targets=(0.08, 0.05), split=0.80),
    "FN-Lite-10K": dict(size=10000, fee=59.99, daily=0.04, max_loss=0.08, trailing=False, targets=(0.08, 0.04), split=0.80),
    "CTI-Direct-20K": dict(size=20000, fee=999.0, daily=None, max_loss=0.06, trailing=False, targets=(), split=0.70),
    "FTUK-Inst-10K": dict(size=10000, fee=None, daily=0.05, max_loss=0.06, trailing=True, targets=(), split=0.65),
}


def stream() -> pd.DataFrame:
    parts = []
    for s in tw.SYMS:
        t = tw.limit_open(tw.all_trades(lh.h1(s)[0], s, 6.0), 1)
        if s in ("NDX100", "USDJPY"):
            t = t[t.dir > 0]
        parts.append(t[["entry", "exit", "R", "at_open", "sym"]])
    t = pd.concat(parts, ignore_index=True)
    return t[t.entry >= FROM].sort_values(["entry", "sym"]).reset_index(drop=True)


def capped(t: pd.DataFrame, crypto: int, max_total: int = 4) -> pd.DataFrame:
    lim = {"crypto": crypto, "comm": 2, "idx": 2, "fx": 2}
    keep, open_ = [], []
    for r in t.itertuples():
        open_ = [o for o in open_ if o[1] > r.entry or (o[1] == r.entry and not o[2])]
        if any(o[0] == r.sym for o in open_) or len(open_) >= max_total:
            continue
        if sum(GROUP[o[0]] == GROUP[r.sym] for o in open_) >= lim[GROUP[r.sym]]:
            continue
        keep.append(r.Index)
        open_.append((r.sym, r.exit, r.at_open))
    return t.loc[keep].reset_index(drop=True)


def attempt(t: pd.DataFrame, start: pd.Timestamp, acc: dict, sz: dict) -> dict:
    sub = t[(t.entry >= start) & (t.entry < start + pd.Timedelta(days=365))]
    ev = sorted([(r.entry, 1, i) for i, r in enumerate(sub.itertuples())] +
                [(r.exit, 0, i) for i, r in enumerate(sub.itertuples())], key=lambda x: (x[0], x[1]))
    init = acc["size"]
    bal, peak, phase, base = init, init, 0, init
    day, day_start = None, init
    risk_usd = {}
    R = sub.R.to_numpy()
    for when, kind, i in ev:
        d = when.normalize()
        if d != day:
            day, day_start = d, bal
        if kind == 1:                                            # entry: size from the balance now
            room = init * acc["max_loss"]
            floor = (peak if acc["trailing"] else init) - room
            m = min(1.5, max(0.3, (bal - floor) / room)) if sz["cushion"] else 1.0
            if sz["taper"] and acc["targets"] and phase < len(acc["targets"]):
                tgt = base * acc["targets"][phase]
                if bal - base >= 0.75 * tgt:
                    m *= 0.5
            risk_usd[i] = bal * min(sz["base"] * m, 0.01)
            continue
        if i not in risk_usd:
            continue
        bal += R[i] * risk_usd.pop(i)
        peak = max(peak, bal)
        if acc["daily"] is not None and day_start - bal >= acc["daily"] * init:
            return {"outcome": "fail daily", "days": (when - start).days, "profit": bal - init}
        floor = (peak if acc["trailing"] else init) - init * acc["max_loss"]
        if bal <= floor:
            return {"outcome": "fail max loss", "days": (when - start).days, "profit": bal - init}
        if acc["targets"] and phase < len(acc["targets"]) and bal >= base * (1 + acc["targets"][phase]):
            phase += 1
            if phase == len(acc["targets"]):
                return {"outcome": "pass", "days": (when - start).days, "profit": 0.0}
            bal = peak = base = day_start = init                 # phase 2 starts on a fresh balance (and day)
            risk_usd.clear()
    if acc["targets"]:
        return {"outcome": "not finished", "days": 365, "profit": bal - init}
    return {"outcome": "survived 12 m", "days": 365, "profit": bal - init}


def main() -> int:
    raw = stream()
    starts = pd.date_range(FROM, raw.exit.max() - pd.Timedelta(days=365), freq="MS")
    print(f"trades {len(raw)} from {raw.entry.min().date()} to {raw.exit.max().date()}, {len(starts)} monthly starts")
    streams = {c: capped(raw, c) for c in (1, 2)}
    res, rows = {}, []
    for an, acc in ACCOUNTS.items():
        for sn, sz in SIZING.items():
            out = pd.DataFrame([attempt(streams[sz["crypto"]], s, acc, sz) for s in starts])
            vc = out.outcome.value_counts(normalize=True)
            r = {"account": an, "sizing": sn,
                 "pass or survive %": round(100 * float(vc.get("pass", 0) + vc.get("survived 12 m", 0)), 1),
                 "fail %": round(100 * float(vc.get("fail daily", 0) + vc.get("fail max loss", 0)), 1),
                 "of which daily %": round(100 * float(vc.get("fail daily", 0)), 1),
                 "unfinished %": round(100 * float(vc.get("not finished", 0)), 1)}
            if acc["targets"]:
                p = out[out.outcome == "pass"].days
                r["median days to pass"] = int(p.median()) if len(p) else None
            else:
                ok = out[out.outcome == "survived 12 m"].profit
                r["median 12 m payout $"] = round(float(np.median(np.maximum(out.profit, 0) * acc["split"]))) \
                    if len(out) else None
                r["12 m payout if survived $"] = round(float(ok.median() * acc["split"])) if len(ok) else None
            rows.append(r)
            res.setdefault(an, {})[sn] = r
        print(f"{an} done", flush=True)
    (ROOT / "reports" / "next_account_sim.json").write_text(json.dumps(res, indent=1, default=str))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
