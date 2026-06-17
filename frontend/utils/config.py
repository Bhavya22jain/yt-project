"""
frontend/utils/config.py
────────────────────────────────────────────────────────────────────────────
Frontend configuration — reads from environment variables and .env file.

Keeps all magic strings and tunable values in one place so no page or
component module imports os.environ directly.

Usage:
    from utils.config import cfg

    base = cfg.BACKEND_URL          # "http://localhost:8000"
    timeout = cfg.REQUEST_TIMEOUT   # 120
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env from the project root (two levels up from frontend/utils/)
_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _ROOT / ".env"

if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)
    except ImportError:
        pass  # python-dotenv not installed; rely on process environment


@dataclass(frozen=True)
class _Config:
    """Immutable frontend configuration object."""

    # Backend connection
    BACKEND_URL: str
    REQUEST_TIMEOUT: int        # seconds; long enough for AI summarization
    SUMMARIZE_ENDPOINT: str
    CHAT_ENDPOINT: str
    VIDEOS_ENDPOINT: str
    HEALTH_ENDPOINT: str

    # UI behaviour
    MAX_CHAT_HISTORY: int       # turns kept in session before pruning
    POLL_INTERVAL_MS: int       # not currently used; reserved for streaming

    # App metadata
    APP_VERSION: str


def _load() -> _Config:
    base = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
    return _Config(
        BACKEND_URL=base,
        REQUEST_TIMEOUT=int(os.getenv("REQUEST_TIMEOUT", "120")),
        SUMMARIZE_ENDPOINT=f"{base}/api/v1/summarize",
        CHAT_ENDPOINT=f"{base}/api/v1/chat",
        VIDEOS_ENDPOINT=f"{base}/api/v1/videos",
        HEALTH_ENDPOINT=f"{base}/api/v1/health",
        MAX_CHAT_HISTORY=int(os.getenv("MAX_CHAT_HISTORY", "20")),
        POLL_INTERVAL_MS=int(os.getenv("POLL_INTERVAL_MS", "500")),
        APP_VERSION=os.getenv("APP_VERSION", "1.0.0"),
    )


# Module-level singleton — import and use directly
cfg: _Config = _load()
