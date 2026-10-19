"""
Loss review of the live multi trader: what kind of loss was each one, and is live still in line with the
backtest?  The "learn from errors" step -- it only reports; nothing it finds changes the trader by itself.

Each closed trade gets one label:
  win            closed at the target
  normal stop    stopped out between -1.25R and -0.75R: the strategy working as designed, nothing to learn
  slipped stop   stopped out worse than -1.25R: gap, spread or slippage -> an execution problem
  manual         closed by hand, not by the bot
  time exit      the bot's 96-hour exit
  bot exit       another close by the bot's own rules (reversal, no stop, prop / emergency flatten)
  closed by bot  a bot close with no recorded reason
Entry slippage: the fill against the price the order was sent at (order_send event), in R (negative = worse).

Live against backtest, per market: the live average R against the backtest of the live rules (REFERENCE) with a
95 % band for that many trades.  Outside the band is a finding; inside it is noise, however it looks.

    python -m tools.loss_review              # print + write reports/loss_review.json

The watchdog runs it once a UTC day (tools/watchdog.py) and sends any new finding to Telegram.

Read-only: opens the trade DB read-only and never touches MT5.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "state" / "multi_live.sqlite"
OUT = ROOT / "reports" / "loss_review.json"

# Backtest of the live rules per market (2 ATR stop / 6 ATR target, 96 h, daily-EMA200 side for XAU / NDX / USDJPY,
# buy-only NDX / USDJPY, strong RSI for XAU / NDX; calm filter not modelled), longest history per market, costs in.
# Source: scale-out / pyramid study 2026-10-14, variant A (9,259 trades): symbol -> (trades, mean R, std R).
REFERENCE = {
    "BTCUSD": (3234, 0.171, 1.762),
    "ETHUSD": (2299, 0.139, 1.753),
    "NDX100": (820, 0.161, 1.800),
    "USDJPY": (1124, 0.189, 1.783),
    "XAUUSD": (1782, 0.117, 1.757),
}
MIN_FOR_VERDICT = 10            # live trades in a market before its average is judged at all
SLIPPED_R = -1.25
BOT_EXITS = {"reversal", "no_sl", "prop_flatten", "idea_loss", "emergency", "weekend"}
DONE = {10008, 10009, 10010}


def label(exit_reason: str | None, r: float | None, bot_reason: str | None) -> str:
    if exit_reason == "target":
        return "win"
    if exit_reason == "stop":
        return "slipped stop" if r is not None and r < SLIPPED_R else "normal stop"
    if exit_reason == "manual":
        return "manual"
    if bot_reason == "time":
        return "time exit"
    if bot_reason in BOT_EXITS:
        return "bot exit"
    return "closed by bot"


def band(symbol: str, n: int) -> tuple[float, float] | None:
    """95 % range of the average R of n backtest trades for this market."""
    ref = REFERENCE.get(symbol)
    if ref is None or n <= 0:
        return None
    half = 1.96 * ref[2] / math.sqrt(n)
    return ref[1] - half, ref[1] + half


def _events(c) -> tuple[dict, dict]:
    """ticket -> bot close reason; symbol -> list of sent prices in time order."""
    closes, sent = {}, {}
    for kind, payload in c.execute("SELECT kind, payload FROM events WHERE kind IN ('close_send','order_send') "
                                   "ORDER BY id"):
        try:
            p = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if kind == "close_send" and p.get("retcode") in DONE:
            closes[int(p["ticket"])] = p.get("reason")
        elif kind == "order_send" and p.get("retcode") in DONE:
            req = p.get("request") or {}
            sent.setdefault(p.get("symbol"), []).append(float(req.get("price") or 0.0))
    return closes, sent


def review(db: Path = DB) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        closes, sent = _events(c)
        rows = c.execute("""SELECT ticket, symbol, direction, opened_utc, closed_utc, entry, sl, exit, exit_reason,
                            pnl_usd, r_multiple, status FROM trades ORDER BY opened_utc""").fetchall()
    finally:
        c.close()
    trades, open_n = [], 0
    used = Counter()
    for (ticket, sym, d, opened, closed, entry, sl, exit_px, why, pnl, r, status) in rows:
        slip_r = None
        prices = sent.get(sym) or []
        if entry and sl and used[sym] < len(prices) and prices[used[sym]]:
            sign = 1.0 if d == "BUY" else -1.0
            slip_r = round(sign * (prices[used[sym]] - entry) / abs(entry - sl), 3)
        used[sym] += 1
        if status != "CLOSED":
            open_n += 1
            continue
        trades.append(dict(ticket=ticket, symbol=sym, direction=d, opened=opened, closed=closed,
                           r=None if r is None else round(r, 3), pnl=pnl, entry_slip_r=slip_r,
                           kind=label(why, r, closes.get(int(ticket)))))
    by_sym = {}
    for sym in sorted({t["symbol"] for t in trades} | set(REFERENCE)):
        g = [t for t in trades if t["symbol"] == sym and t["r"] is not None]
        if not g:
            continue
        avg = sum(t["r"] for t in g) / len(g)
        b = band(sym, len(g))
        if len(g) < MIN_FOR_VERDICT or b is None:
            verdict = f"too early ({len(g)} of {MIN_FOR_VERDICT} trades)"
        elif avg < b[0]:
            verdict = "BELOW the backtest range -- look at this market"
        elif avg > b[1]:
            verdict = "above the backtest range (luck or a better market; do not raise risk)"
        else:
            verdict = "in line with the backtest"
        by_sym[sym] = dict(trades=len(g), avg_R=round(avg, 3), total_R=round(sum(t["r"] for t in g), 2),
                           wins=sum(t["kind"] == "win" for t in g),
                           backtest_avg_R=REFERENCE.get(sym, (None, None))[1],
                           band_95=None if b is None else [round(b[0], 3), round(b[1], 3)], verdict=verdict)
    kinds = Counter(t["kind"] for t in trades)
    slips = [t["entry_slip_r"] for t in trades if t["entry_slip_r"] is not None]
    findings = []
    if kinds["slipped stop"]:
        findings.append(f"{kinds['slipped stop']} stop(s) filled worse than {SLIPPED_R}R: check gaps / spread at "
                        "those times (execution, not the strategy)")
    if slips and sum(slips) / len(slips) < -0.05:
        findings.append(f"entries fill {sum(slips) / len(slips):+.3f}R worse than sent on average: costs are above plan")
    if kinds["manual"]:
        findings.append(f"{kinds['manual']} trade(s) closed by hand: they are counted, but they are not the strategy")
    findings += [f"{s}: {v['verdict']}" for s, v in by_sym.items() if v["verdict"].startswith(("BELOW", "above"))]
    if not findings:
        findings.append("nothing to act on: every loss so far is the strategy working as designed or too few trades")
    rs = [t["r"] for t in trades if t["r"] is not None]
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), closed=len(trades),
                open=open_n, avg_R=None if not rs else round(sum(rs) / len(rs), 3),
                total_R=round(sum(rs), 2), net_usd=round(sum(t["pnl"] or 0.0 for t in trades), 2),
                kinds=dict(kinds), avg_entry_slip_R=None if not slips else round(sum(slips) / len(slips), 3),
                by_symbol=by_sym, findings=findings, trades=trades[-50:])


def render(rep: dict) -> str:
    out = [f"Loss review {rep['generated_utc']}: {rep['closed']} closed, {rep['open']} open, "
           f"avg {rep['avg_R']}R, total {rep['total_R']}R, net ${rep['net_usd']}",
           "Kinds: " + (", ".join(f"{k} {v}" for k, v in sorted(rep["kinds"].items())) or "none"),
           f"Average entry slippage: {rep['avg_entry_slip_R']}R"]
    for s, v in rep["by_symbol"].items():
        out.append(f"  {s:7s} {v['trades']:3d} trades  avg {v['avg_R']:+.3f}R (backtest {v['backtest_avg_R']}, "
                   f"95 % band {v['band_95']})  -> {v['verdict']}")
    out.append("Findings:")
    out += [f"  - {f}" for f in rep["findings"]]
    return "\n".join(out)


def run(db: Path = DB, out: Path = OUT) -> dict:
    rep = review(db)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DB)
    a = ap.parse_args()
    print(render(run(a.db)))


if __name__ == "__main__":
    main()
