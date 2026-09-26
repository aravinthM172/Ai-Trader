"""Central logging.  Console + rotating file in logs/.  Never logs secrets."""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = Path(os.getenv("GOLD_AI_LOG_DIR") or (_ROOT / "logs"))   # tests redirect this
_LOG_DIR.mkdir(exist_ok=True)

_SECRET_KEYS = ("password", "api_key", "anthropic", "token", "secret", "login")
_SECRET_RE = re.compile(
    r"(?i)\b(" + "|".join(_SECRET_KEYS) + r")\b\s*[:=]\s*\S+"
)


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if _SECRET_RE.search(msg):
                record.msg = _SECRET_RE.sub(r"\1=***", msg)
                record.args = ()
        except Exception:
            pass
        return True


_CONFIGURED: set[str] = set()


def get_logger(name: str = "btc", *, filename: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    redact = _RedactFilter()

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.addFilter(redact)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        _LOG_DIR / (filename or f"{name}.log"),
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.addFilter(redact)
    logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger
