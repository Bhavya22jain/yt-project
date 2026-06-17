"""
database/database.py
─────────────────────────────────────────────────────────────────────────────
Database connection layer — SQLAlchemy 2.0 style.

Responsibilities:
  • Create the SQLite engine (configurable path via settings)
  • Provide a session factory (SessionLocal)
  • Expose `get_db()` as a FastAPI dependency for request-scoped sessions
  • Expose `init_db()` to create all tables at application startup
  • Expose `engine` for direct use in tests (in-memory SQLite)

Architecture note:
  Sessions are request-scoped. One session is opened when a request arrives
  and committed/closed when it finishes. Errors trigger an automatic rollback.
  Never import `SessionLocal` directly in services — always use `get_db()`.

Usage (FastAPI endpoint):
    from app.database.database import get_db
    from sqlalchemy.orm import Session

    @router.post("/summarize")
    def summarize(db: Session = Depends(get_db)):
        ...

Usage (startup):
    from app.database.database import init_db
    init_db()
"""

from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# All ORM models inherit from this. Defining it here avoids circular imports.
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all ORM table models.
    Import this in db_models.py, not the other way around.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Database URL
# ─────────────────────────────────────────────────────────────────────────────

def _build_database_url() -> str:
    """
    Construct the SQLite connection URL.

    • In production / staging: reads DATABASE_URL from settings (if set).
    • Otherwise: creates a `data/` directory next to the backend package
      and stores `yt_summarizer.db` there.
    • Tests can override by setting DATABASE_URL=sqlite:///:memory: in .env.test
    """
    db_url: str = getattr(settings, "database_url", "")

    if db_url:
        return db_url

    # Default: file-based SQLite under backend/data/
    db_dir = Path(__file__).resolve().parent.parent.parent / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "yt_summarizer.db"
    return f"sqlite:///{db_path}"


DATABASE_URL: str = _build_database_url()


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

engine = create_engine(
    DATABASE_URL,
    #
    # connect_args — SQLite-specific:
    #   check_same_thread=False   required because FastAPI uses a thread pool;
    #                             SQLAlchemy's connection pool handles safety.
    connect_args={"check_same_thread": False},
    #
    # echo — logs every SQL statement in debug mode. Very useful during
    # development; set DEBUG=false in .env to silence it in production.
    echo=settings.debug,
    #
    # pool_pre_ping — test connections before use to handle stale connections.
    pool_pre_ping=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Enable WAL mode for SQLite
# WAL (Write-Ahead Logging) allows concurrent reads during writes — important
# when Streamlit and FastAPI both hit the same file.
# ─────────────────────────────────────────────────────────────────────────────

@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record) -> None:  # type: ignore[type-arg]
    """Enable WAL mode and foreign-key enforcement on every new connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


# ─────────────────────────────────────────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────────────────────────────────────────

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,   # We manage transactions explicitly (commit/rollback)
    autoflush=False,    # Prevent implicit flushes; flush manually before queries
    expire_on_commit=False,  # Keep ORM objects usable after session.commit()
)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency — request-scoped session
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session for a single request.

    Pattern:
        - Open a new session before the request handler runs.
        - Yield it to the handler (and any downstream dependencies).
        - Always close the session in the `finally` block, even on errors.
        - Let CRUD functions commit; the dependency never commits itself.

    Example:
        @router.get("/videos")
        def list_videos(db: Session = Depends(get_db)):
            return crud.get_all_videos(db)
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Table initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables defined via the ORM (if they don't already exist).
    Call this once at application startup from main.py's lifespan hook.

    Uses CREATE TABLE IF NOT EXISTS semantics — safe to call multiple times.
    Does NOT run migrations; use Alembic for schema evolution in production.
    """
    # Import db_models here (not at module top) to avoid circular imports.
    # The import registers each model's metadata with Base.
    from app.database import db_models  # noqa: F401

    logger.info(f"Initialising database | url={DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified ✓")


def drop_all_tables() -> None:
    """
    Drop every table. DESTRUCTIVE — test environments only.
    Guards against accidental use in production.
    """
    if settings.is_production:
        raise RuntimeError("drop_all_tables() must never run in production.")

    from app.database import db_models  # noqa: F401

    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped.")


def health_check_db() -> bool:
    """
    Verify the database connection is alive.
    Returns True if reachable, False otherwise.
    Used by the /health endpoint.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(f"Database health check failed: {exc}")
        return False
