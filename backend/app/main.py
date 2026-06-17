"""
main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application factory — the composition root of the backend.

Responsibilities:
  • Build and configure the FastAPI app instance.
  • Register middleware (CORS, request-ID, timing).
  • Register global exception handlers (AppBaseException, Pydantic, generic).
  • Hook lifecycle events (startup: logging + DB init; shutdown: cleanup).
  • Mount API routers.

Architecture notes:
  • Use create_app() factory rather than a module-level `app = FastAPI()`
    so that tests can call create_app() with different settings and get
    an isolated instance.
  • The module-level `app` at the bottom is what uvicorn loads.
  • Middleware runs in reverse registration order (last registered = outermost).

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.core.exceptions import AppBaseException
from app.core.logging import setup_logging
from app.api.v1.endpoints.video import router as video_router
from app.database.database import init_db, health_check_db


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup & shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs setup before `yield` (startup) and cleanup after `yield` (shutdown).

    Startup:
      1. Configure Loguru logging.
      2. Initialise the SQLite database (CREATE TABLE IF NOT EXISTS).
      3. Verify DB connectivity and log the result.

    Shutdown:
      1. Log a clean shutdown message.
      (Add connection pool draining, cache flushing here as the project grows.)
    """
    # ── Startup ───────────────────────────────────────────────────────────
    setup_logging()

    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info("=" * 60)
    logger.info(f"  Environment : {settings.app_env}")
    logger.info(f"  Debug mode  : {settings.debug}")
    logger.info(f"  API docs    : http://{settings.backend_host}:{settings.backend_port}/docs")

    init_db()
    db_status = "✓ connected" if health_check_db() else "✗ UNREACHABLE — check logs"
    logger.info(f"  Database    : {db_status}")
    logger.info("=" * 60)
    logger.info("Server ready.")

    yield  # ← application is live here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info(f"{settings.app_name} shutting down cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and return a fully configured FastAPI application.

    Called once at module load (bottom of this file) to produce the `app`
    object uvicorn imports. Tests can also call create_app() directly.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "## AI YouTube Video Summarizer & Chat Assistant\n\n"
            "Paste a YouTube URL to get an AI-generated structured summary.\n\n"
            "### Pipeline\n"
            "`URL` → `transcript extraction` → `Claude AI summarization` "
            "→ `SQLite persistence` → `structured JSON response`\n\n"
            "### Endpoints\n"
            "- **POST /api/v1/summarize** — Run the full pipeline\n"
            "- **GET  /api/v1/videos**    — List processed videos\n"
            "- **GET  /api/v1/videos/{id}** — Fetch a single video\n"
            "- **DELETE /api/v1/videos/{id}** — Remove a video\n"
            "- **POST /api/v1/chat**      — Chat with a video\n"
            "- **GET  /api/v1/health**    — Liveness check\n"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # Show request validation errors in full in debug mode
        debug=settings.debug,
    )

    # ── Middleware (outermost first) ───────────────────────────────────────

    # CORS — allow Streamlit frontend and local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # ── Request-ID + timing middleware ────────────────────────────────────
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        """
        Attach a unique X-Request-ID to every request and response,
        and add X-Process-Time (ms) to every response header.

        This makes individual requests traceable across log lines, error
        responses, and client-side debugging.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(elapsed_ms)

        logger.debug(
            f"{request.method} {request.url.path} → "
            f"{response.status_code} [{elapsed_ms}ms] rid={request_id}"
        )
        return response

    # ── Exception handlers ────────────────────────────────────────────────

    @app.exception_handler(AppBaseException)
    async def app_exception_handler(
        request: Request, exc: AppBaseException
    ) -> JSONResponse:
        """
        Convert all typed application exceptions to structured JSON errors.
        These are expected errors (invalid URL, transcript not found, etc.)
        logged at WARNING rather than ERROR.
        """
        rid = getattr(request.state, "request_id", "unknown")
        code_map = {
            "InvalidYouTubeURLError": "INVALID_URL",
            "TranscriptNotAvailableError": "TRANSCRIPT_NOT_AVAILABLE",
            "TranscriptFetchError": "TRANSCRIPT_FETCH_FAILED",
            "SummarizationError": "SUMMARIZATION_FAILED",
            "AIProviderError": "AI_PROVIDER_ERROR",
            "ChatError": "CHAT_FAILED",
            "ValidationError": "VALIDATION_ERROR",
        }
        code = code_map.get(type(exc).__name__, "APPLICATION_ERROR")

        logger.warning(
            f"[{rid}] {type(exc).__name__} [{exc.status_code}]: {exc.message}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.message,
                "code": code,
                "request_id": rid,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Convert Pydantic validation errors (e.g. missing fields, wrong types)
        into a structured ErrorResponse with per-field detail.

        HTTP 422 Unprocessable Entity.
        """
        rid = getattr(request.state, "request_id", "unknown")

        details = []
        for error in exc.errors():
            field_path = " → ".join(str(loc) for loc in error["loc"] if loc != "body")
            details.append({
                "field": field_path or None,
                "message": error["msg"],
            })

        logger.warning(
            f"[{rid}] Request validation error on {request.url.path}: "
            f"{len(details)} field(s) failed"
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Request validation failed. Check the 'details' field.",
                "code": "VALIDATION_ERROR",
                "details": details,
                "request_id": rid,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for any unhandled exception.
        Logged at ERROR with full traceback; client receives a generic message
        (no internal detail exposed in production).
        """
        rid = getattr(request.state, "request_id", "unknown")
        logger.exception(
            f"[{rid}] Unhandled exception on {request.method} {request.url.path}: {exc}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "An unexpected internal error occurred.",
                "code": "INTERNAL_ERROR",
                "request_id": rid,
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(video_router, prefix="/api/v1")

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Module-level app instance — uvicorn entry point
# ─────────────────────────────────────────────────────────────────────────────
app: FastAPI = create_app()
