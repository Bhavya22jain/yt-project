"""
core/logging.py
───────────────
Configures Loguru as the application logger.
Call `setup_logging()` once at application startup (main.py).
All other modules simply do: from loguru import logger
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """
    Remove the default Loguru handler and configure:
    - A coloured stdout handler (development)
    - A rotating file handler (all environments)
    """
    logger.remove()  # Remove Loguru's default stderr handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # ── Console handler ──────────────────────────────────────────
    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=settings.debug,
        diagnose=settings.debug,
    )

    # ── File handler ─────────────────────────────────────────────
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path,
        format=log_format,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=False,      # Never write locals to file (security)
        enqueue=True,        # Thread-safe async-friendly writes
    )

    logger.info(
        f"Logging configured | level={settings.log_level} | "
        f"env={settings.app_env} | file={settings.log_file}"
    )
