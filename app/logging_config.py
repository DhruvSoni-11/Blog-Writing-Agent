"""File logging with automatic redaction of secrets."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
_KEY_PATTERNS = [
    re.compile(r"sk-or-[A-Za-z0-9\-_]{8,}"),  # OpenRouter keys
    re.compile(r"tvly-[A-Za-z0-9\-_]{8,}"),  # Tavily keys
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-_.=]{8,}"),
    re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^\s'\",]+"),
]


class RedactingFormatter(logging.Formatter):
    """Formatter that removes configured secrets and key-like strings (including tracebacks)."""

    def __init__(self, fmt: str, secrets: list[str] | None = None) -> None:
        super().__init__(fmt)
        self.secrets = [s for s in (secrets or []) if s and len(s) >= 4]

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "***REDACTED***")
        for pattern in _KEY_PATTERNS:
            text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "***REDACTED***", text)
        return text

    def format(self, record: logging.LogRecord) -> str:
        return self.redact(super().format(record))


def setup_logging(level: str = "INFO", secrets: list[str] | None = None, log_dir: Path = LOG_DIR) -> Path:
    """Log to ``logs/blogforge.log`` only, so Rich terminal output stays clean."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "blogforge.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", secrets))
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party HTTP libraries are noisy at INFO and may log request details.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "primp", "ddgs", "tavily", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file
