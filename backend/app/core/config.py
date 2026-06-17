"""
core/config.py
──────────────
Centralised application settings loaded from environment variables.
A single `settings` singleton is imported throughout the codebase —
never read os.environ directly outside this module.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration values, with sensible defaults for development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = Field(default="AI YouTube Summarizer")
    app_version: str = Field(default="1.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)

    # ── Backend ───────────────────────────────────────────────────
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    backend_reload: bool = Field(default=True)

    # ── Frontend ──────────────────────────────────────────────────
    frontend_port: int = Field(default=8501)
    backend_url: str = Field(default="http://localhost:8000")

    # ── Anthropic ─────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-opus-4-20250514")
    anthropic_max_tokens: int = Field(default=4096)
    anthropic_temperature: float = Field(default=0.3)

    # ── Transcript ────────────────────────────────────────────────
    max_transcript_length: int = Field(default=50000)
    transcript_languages: str = Field(default="en,en-US,en-GB")

    # ── Database ──────────────────────────────────────────────────
    database_url: str = Field(
        default="",
        description=(
            "Full SQLAlchemy URL. Leave blank to use the default "
            "SQLite file at backend/data/yt_summarizer.db"
        ),
    )

    # ── Summary ───────────────────────────────────────────────────
    summary_cache_ttl: int = Field(default=3600)

    # ── CORS ──────────────────────────────────────────────────────
    allowed_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501"
    )

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")

    # ── Derived helpers ───────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse the comma-separated ALLOWED_ORIGINS string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def transcript_languages_list(self) -> List[str]:
        """Parse the comma-separated TRANSCRIPT_LANGUAGES string into a list."""
        return [lang.strip() for lang in self.transcript_languages.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    Use this in FastAPI dependency injection:

        from app.core.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()
