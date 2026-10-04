"""
Telegram notifications (stdlib only).  Without TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in
.env, messages go to logs/notify.log instead, so nothing breaks before Telegram is set up.

Setup: talk to @BotFather -> /newbot -> token; send your bot any message, then open
https://api.telegram.org/bot<TOKEN>/getUpdates to read your chat id.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from common.logging_setup import get_logger

load_dotenv()
log = get_logger("notify", filename="notify.log")
_API = "https://api.telegram.org/bot{token}/{method}"


def _cfg() -> tuple[str, str]:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip()


def enabled() -> bool:
    t, c = _cfg()
    return bool(t and c)


def _call(method: str, params: dict, timeout: float = 15.0) -> dict | None:
    token, _ = _cfg()
    if not token:
        return None
    data = urllib.parse.urlencode(params).encode()
    try:
        with urllib.request.urlopen(_API.format(token=token, method=method), data=data, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:                                   # never let alerting crash the caller
        log.warning("telegram %s failed: %s", method, e)
        return None


def send(text: str) -> bool:
    """Send a message; always also logged.  Returns True if Telegram accepted it."""
    log.info("NOTIFY %s", text.replace("\n", " | "))
    token, chat = _cfg()
    if not (token and chat):
        return False
    r = _call("sendMessage", {"chat_id": chat, "text": text[:4000], "disable_web_page_preview": "true"})
    return bool(r and r.get("ok"))


def poll_commands(offset: int, timeout: int = 0) -> tuple[list[str], int]:
    """Return (commands from the configured chat only, next offset)."""
    _, chat = _cfg()
    r = _call("getUpdates", {"offset": offset, "timeout": timeout}, timeout=timeout + 15)
    cmds = []
    if not (r and r.get("ok")):
        return cmds, offset
    for u in r.get("result", []):
        offset = max(offset, int(u["update_id"]) + 1)
        m = u.get("message") or {}
        if str((m.get("chat") or {}).get("id")) == chat and (m.get("text") or "").startswith("/"):
            cmds.append(m["text"].split()[0].split("@")[0].lower())
    return cmds, offset
