"""Uniform logging for the platform."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Install a single stderr handler at the configured level (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    from config.settings import get_settings

    resolved = (level or get_settings().log_level).upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("afya")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (`afya.<name>`)."""
    configure_logging()
    return logging.getLogger(f"afya.{name}" if not name.startswith("afya") else name)
