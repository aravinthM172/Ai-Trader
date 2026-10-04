"""
READ-ONLY MCP server for the gold-ai-trader bots.

Lets Claude answer "how is the bot doing?" from status files, trade databases, logs and
reports.  It NEVER connects to MT5, never writes a file, never sends or closes an order,
and has no tool that runs arbitrary code (lesson from tradingview-mcp's ui_evaluate).
SQLite databases are opened read-only (mode=ro).

Register with Claude Code: see .mcp.json in the project root.
    venv/Scripts/python -m mcp_server.trader_mcp       # stdio
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer

BOTS = {
    "btc": dict(status=ROOT / "reports" / "btc_live_status.json", db=ROOT / "state" / "btc_live_H1.sqlite",
                log=ROOT / "logs" / "btc_live.log", ref=ROOT / "reports" / "btc_paper_replay_trades.csv"),
    "multi": dict(status=ROOT / "reports" / "multi_live_status.json", db=ROOT / "state" / "multi_live.sqlite",
                  log=ROOT / "logs" / "multi_live.log", ref=ROOT / "reports" / "multi_reference_trades.csv"),
}
REPORTS = {   # whitelisted, read-only
    "rules": ROOT / "RULES.md", "multi_symbol_scan": ROOT / "MULTI_SYMBOL_SCAN.md",
    "portfolio_sim": ROOT / "PORTFOLIO_SIM.md", "strategy_round1": ROOT / "STRATEGY_ROUND1.md",
    "swap_rates": ROOT / "reports" / "swap_rates.json", "edge_monitor": ROOT / "reports" / "edge_monitor.json",
    "edge_monitor_multi": ROOT / "reports" / "edge_monitor_multi.json", "research_ledger": ROOT / "research" / "ledger.csv",
    "add_list": ROOT / "research" / "github_algo_trading" / "ADD_LIST.md",
}
KILL = ROOT / "state" / "KILL_SWITCH"

server = MCPServer(
    name="gold-ai-trader",
    instructions="""Read-only view of the gold-ai-trader live bots. Nothing here can trade.

Bots: "btc" = BTCUSD.vx H1 momentum (execution/live.py); "multi" = same rule on the symbols that
passed the 55-symbol scan (execution/live_multi.py).

Which tool when:
- "Is everything OK?"                      -> health  (call first)
- "How is the btc / multi bot doing?"      -> bot_status, then live_vs_backtest
- "What trades happened?"                  -> recent_trades (default 20; keep it small)
- "Why didn't it trade?"                   -> bot_status (last entry decisions), then recent_log
- "Is news blocking trading?"              -> upcoming_news
- "What are the rules / what passed?"      -> read_report (rules, multi_symbol_scan, portfolio_sim ...)
To stop trading the user must use Telegram /kill or create state/KILL_SWITCH themselves.""",
)


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_min(st: dict | None):
    try:
        return round((datetime.now(timezone.utc) - datetime.fromisoformat(st["generated_utc"])).total_seconds() / 60, 1)
    except Exception:
        return None


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


@server.tool()
def health() -> dict:
    """One-call overview: each bot's mode, status age, balance/equity, open positions, kill switch."""
    out = {"kill_switch": KILL.read_text(encoding="utf-8").strip()[:300] if KILL.exists() else None}
    for name, p in BOTS.items():
        st = _json(p["status"])
        if st is None:
            out[name] = "no status file"
            continue
        acct = st.get("account") or {}
        out[name] = {"mode": st.get("mode"), "status_age_min": _age_min(st),
                     "stale": (_age_min(st) or 1e9) > 5,
                     "balance_or_equity": acct.get("balance", st.get("equity")),
                     "open_positions": len(acct.get("open_positions") or st.get("open_positions") or []),
                     "closed_trades": st.get("closed_trades"), "net_pl_usd": st.get("net_pl_usd"),
                     "expectancy_R": st.get("expectancy_R"), "error": acct.get("error") or st.get("error")}
    return out


@server.tool()
def bot_status(bot: str = "btc") -> dict:
    """Full status JSON of one bot ("btc" or "multi"), including last entry decisions."""
    p = BOTS.get(bot)
    if not p:
        return {"error": f"unknown bot {bot!r}; use 'btc' or 'multi'"}
    st = _json(p["status"])
    return {"status_age_min": _age_min(st), **(st or {"error": "no status file"})}


@server.tool()
def recent_trades(bot: str = "btc", n: int = 20) -> dict:
    """Last n trades (open and closed) from the bot's trade database, newest first."""
    p = BOTS.get(bot)
    if not p or not p["db"].exists():
        return {"error": "no trade database yet"}
    n = max(1, min(int(n), 200))
    c = _ro(p["db"])
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(trades)")]
        rows = c.execute(f"SELECT * FROM trades ORDER BY opened_utc DESC LIMIT {n}").fetchall()
    finally:
        c.close()
    return {"trades": [dict(zip(cols, r)) for r in rows]}


@server.tool()
def live_vs_backtest(bot: str = "btc") -> dict:
    """Edge-monitor comparison of live trades vs the backtest distribution (report only, never trips)."""
    from tools import edge_monitor as em
    p = BOTS.get(bot)
    if not p or not p["ref"].exists():
        return {"error": "no reference trades for this bot"}
    return em.evaluate(em.load_live_R(p["db"]), em.load_reference_R(p["ref"]))


@server.tool()
def recent_log(bot: str = "btc", lines: int = 40) -> dict:
    """Last lines of the bot's log file (max 200)."""
    p = BOTS.get(bot)
    if not p or not p["log"].exists():
        return {"error": "no log file"}
    with p["log"].open(encoding="utf-8", errors="replace") as f:
        return {"lines": list(deque(f, maxlen=max(1, min(int(lines), 200))))}


@server.tool()
def upcoming_news(hours: int = 48) -> dict:
    """High-impact economic events in the next `hours` (these block trading in affected symbols)."""
    from news.filter import NewsFilter
    ev = NewsFilter().upcoming(hours=max(1, min(int(hours), 168)))
    return {"events": [{"time_utc": e["time"].isoformat(), "currency": e["currency"], "event": e["event"]} for e in ev]}


@server.tool()
def read_report(name: str = "rules") -> dict:
    """Read a whitelisted report: rules, multi_symbol_scan, portfolio_sim, strategy_round1, swap_rates,
    edge_monitor, edge_monitor_multi, research_ledger, add_list."""
    p = REPORTS.get(name)
    if not p:
        return {"error": f"unknown report; choose from {sorted(REPORTS)}"}
    if not p.exists():
        return {"error": "report not generated yet"}
    return {"name": name, "content": p.read_text(encoding="utf-8", errors="replace")[:20_000]}


if __name__ == "__main__":
    server.run("stdio")
