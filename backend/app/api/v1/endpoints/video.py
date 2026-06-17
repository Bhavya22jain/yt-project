"""
api/v1/endpoints/video.py  —  routes.py
────────────────────────────────────────────────────────────────────────────
FastAPI router — all video-related HTTP endpoints.

The router layer has exactly three responsibilities:
  1. Parse and validate incoming requests (via Pydantic schemas + Depends).
  2. Orchestrate service calls and database writes.
  3. Serialise and return typed responses.

No business logic lives here. No raw SQL. No direct Anthropic calls.

Endpoints:
  POST   /api/v1/summarize            Full URL→Transcript→AI→DB→Response flow
  GET    /api/v1/videos               Paginated list of processed videos
  GET    /api/v1/videos/{video_id}    Single video + summary by YT video ID
  DELETE /api/v1/videos/{video_id}    Remove a video and all its data
  GET    /api/v1/health               Liveness + DB connectivity check
  POST   /api/v1/chat                 Chat with a video via its transcript

Request-scoped DB sessions are injected via Depends(get_db) — each request
gets its own session; it is committed by CRUD functions and closed in the
finally block of get_db(), never by this module.

Request ID middleware: every request is tagged with a UUID (X-Request-ID)
so errors can be correlated across logs, responses, and the client.
"""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import (
    AIProviderError,
    AppBaseException,
    SummarizationError,
    TranscriptFetchError,
    TranscriptNotAvailableError,
)
from app.database.database import get_db, health_check_db
from app.database import crud
from app.schemas.video import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    ErrorResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
    TimestampItem,
    VideoDetailResponse,
    VideoListResponse,
    VideoRecord,
    VideoSummary,
)
from app.services.chat_service import ChatService
from app.services.summary_service import SummaryService
from app.services.transcript_service import TranscriptService

router = APIRouter()

# ── Service singletons ────────────────────────────────────────────────────────
# One instance shared across all requests — services are stateless after init.
# In a production system these would be injected via a DI framework.
_transcript_service = TranscriptService()
_summary_service = SummaryService()
_chat_service = ChatService()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_request_id() -> str:
    """Generate a short unique request identifier for log correlation."""
    return uuid.uuid4().hex[:12]


def _video_record_from_db(
    video_db,
    summary_db=None,
) -> VideoRecord:
    """
    Convert a (VideoDB, Optional[SummaryDB]) pair into a VideoRecord schema.

    All field mappings are explicit so that adding a column to the ORM model
    doesn't silently change the API response shape.
    """
    summary_schema: Optional[VideoSummary] = None

    if summary_db is not None:
        summary_schema = VideoSummary(
            video_id=video_db.video_id,
            title=video_db.title,
            duration=video_db.metadata_duration,
            executive_summary=summary_db.executive_summary,
            detailed_summary=summary_db.detailed_summary,
            key_points=crud.get_key_points(summary_db),
            action_items=crud.get_action_items(summary_db),
            important_timestamps=[
                TimestampItem(time=t["time"], description=t["description"])
                for t in crud.get_important_timestamps(summary_db)
                if t.get("time") and t.get("description")
            ],
        )

    return VideoRecord(
        db_id=video_db.id,
        video_id=video_db.video_id,
        youtube_url=video_db.youtube_url,
        title=video_db.title,
        channel=video_db.channel,
        duration=video_db.metadata_duration,
        language=video_db.language,
        is_processed=video_db.is_processed,
        summarize_count=video_db.summarize_count or 0,
        word_count=video_db.transcript_word_count,
        transcript=video_db.transcript_text,
        created_at=video_db.created_at,
        updated_at=video_db.updated_at,
        summary=summary_schema,
    )


def _raise_http(exc: AppBaseException, request_id: str) -> None:
    """
    Convert a typed AppBaseException into an HTTPException with a structured
    JSON body that matches ErrorResponse.

    We raise HTTPException (not return ErrorResponse) because FastAPI needs to
    set the HTTP status code, not just serialize the body.
    """
    # Map application exception types to machine-readable error codes
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
    raise HTTPException(
        status_code=exc.status_code,
        detail={
            "success": False,
            "error": exc.message,
            "code": code,
            "request_id": request_id,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /summarize  — the core pipeline endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize a YouTube video",
    description="""
Full pipeline: URL validation → transcript extraction → AI summarization
→ database persistence → structured response.

**Caching**: if the video has been summarized before and `force_refresh` is
False, the cached summary is returned immediately (processing_ms will be low
and `cached` will be True).

**Re-summarization**: pass `force_refresh: true` to bypass the cache and
generate a fresh summary — useful when a video's auto-captions have improved.
    """,
    tags=["Video"],
    responses={
        200: {"description": "Summary generated or retrieved from cache"},
        422: {"model": ErrorResponse, "description": "Invalid YouTube URL"},
        404: {"model": ErrorResponse, "description": "No transcript available for this video"},
        502: {"model": ErrorResponse, "description": "Transcript fetch or AI provider error"},
        503: {"model": ErrorResponse, "description": "AI provider unavailable"},
    },
)
async def summarize_video(
    request: SummarizeRequest,
    db: Session = Depends(get_db),
) -> SummarizeResponse:
    """
    Core pipeline:
      1. Validate the YouTube URL (Pydantic, already done by schema).
      2. Extract the video ID.
      3. Check the DB cache — return cached summary if fresh and not forced.
      4. Fetch the transcript from YouTube.
      5. Persist the video + transcript to the DB.
      6. Call the AI summarization service.
      7. Persist the summary to the DB.
      8. Return the structured SummarizeResponse.

    Every step that can fail raises a typed AppBaseException which is caught
    here and converted to an HTTP error with an ErrorResponse body.
    """
    rid = _new_request_id()
    t_start = time.monotonic()

    logger.info(
        f"[{rid}] POST /summarize | url={request.youtube_url!r} "
        f"force_refresh={request.force_refresh}"
    )

    # ── Step 1: Extract video ID ──────────────────────────────────────────
    try:
        video_id = _transcript_service.extract_video_id(request.youtube_url)
    except AppBaseException as exc:
        logger.warning(f"[{rid}] URL validation failed: {exc.message}")
        _raise_http(exc, rid)

    logger.debug(f"[{rid}] video_id={video_id!r}")

    # ── Step 2: Check DB cache ────────────────────────────────────────────
    if not request.force_refresh:
        cached_result = crud.get_video_with_summary(db, video_id=video_id)
        if cached_result is not None:
            video_db, summary_db = cached_result
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(
                f"[{rid}] Cache HIT video_id={video_id!r} | "
                f"db_id={video_db.id} | elapsed_ms={elapsed_ms}"
            )
            summary_schema = VideoSummary(
                video_id=video_db.video_id,
                title=video_db.title,
                duration=video_db.metadata_duration,
                executive_summary=summary_db.executive_summary,
                detailed_summary=summary_db.detailed_summary,
                key_points=crud.get_key_points(summary_db),
                action_items=crud.get_action_items(summary_db),
                important_timestamps=[
                    TimestampItem(time=t["time"], description=t["description"])
                    for t in crud.get_important_timestamps(summary_db)
                    if t.get("time") and t.get("description")
                ],
            )
            return SummarizeResponse(
                success=True,
                cached=True,
                processing_ms=elapsed_ms,
                data=summary_schema,
            )

    logger.debug(f"[{rid}] Cache MISS — proceeding with full pipeline")

    # ── Step 3: Fetch transcript ──────────────────────────────────────────
    try:
        transcript = await _transcript_service.get_transcript(request.youtube_url)
    except AppBaseException as exc:
        logger.warning(f"[{rid}] Transcript fetch failed: {exc.message}")
        # Persist the failure so we don't hammer YouTube on retries
        _persist_failed_video(db, video_id, request.youtube_url, str(exc.message))
        _raise_http(exc, rid)

    logger.debug(
        f"[{rid}] Transcript fetched | segments={len(transcript.segments)} "
        f"words={transcript.word_count} lang={transcript.language!r}"
    )

    # ── Step 4: Persist video + transcript to DB ──────────────────────────
    try:
        video_db, created = crud.get_or_create_video(
            db,
            video_id=video_id,
            youtube_url=request.youtube_url,
            language=transcript.language,
            transcript_text=transcript.full_text,
            transcript_word_count=transcript.word_count,
        )
        if not created:
            # Update transcript data on existing record (language / text may differ
            # if a better-quality caption track was found this time)
            video_db = crud.update_video(
                db,
                video_db_id=video_db.id,
                language=transcript.language,
                transcript_text=transcript.full_text,
                transcript_word_count=transcript.word_count,
            )
        logger.debug(
            f"[{rid}] Video persisted | db_id={video_db.id} created={created}"
        )
    except Exception as exc:
        logger.exception(f"[{rid}] DB write (video) failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "Database write failed.", "request_id": rid},
        )

    # ── Step 5: AI summarization ──────────────────────────────────────────
    try:
        summary = await _summary_service.summarize(transcript)
    except AppBaseException as exc:
        logger.warning(f"[{rid}] Summarization failed: {exc.message}")
        crud.mark_video_failed(
            db, video_db_id=video_db.id, error_message=exc.message
        )
        _raise_http(exc, rid)

    logger.debug(
        f"[{rid}] Summary generated | "
        f"key_points={len(summary.key_points)} "
        f"action_items={len(summary.action_items)} "
        f"timestamps={len(summary.important_timestamps)}"
    )

    # ── Step 6: Persist summary to DB ─────────────────────────────────────
    try:
        crud.upsert_summary(
            db,
            video_db_id=video_db.id,
            model_name=settings.anthropic_model,
            executive_summary=summary.executive_summary,
            detailed_summary=summary.detailed_summary,
            key_points=summary.key_points,
            action_items=summary.action_items,
            important_timestamps=[
                {"time": ts.time, "description": ts.description}
                for ts in summary.important_timestamps
            ],
        )
        crud.mark_video_processed(db, video_db_id=video_db.id)
        logger.debug(f"[{rid}] Summary persisted to DB for db_id={video_db.id}")
    except Exception as exc:
        logger.exception(f"[{rid}] DB write (summary) failed: {exc}")
        # The summary was generated — return it even if DB write fails.
        # Log prominently so ops can investigate.
        logger.error(
            f"[{rid}] ⚠ Summary generated but NOT persisted. "
            "Returning response anyway — check DB health."
        )

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        f"[{rid}] Summarization complete | video_id={video_id!r} | "
        f"elapsed_ms={elapsed_ms}"
    )

    return SummarizeResponse(
        success=True,
        cached=False,
        processing_ms=elapsed_ms,
        data=summary,
    )


def _persist_failed_video(
    db: Session, video_id: str, youtube_url: str, error_message: str
) -> None:
    """
    Record a failed processing attempt in the DB so we can track
    which videos fail and why without blocking the error response.
    Errors here are swallowed — we never let a DB write block an HTTP response.
    """
    try:
        video_db, _ = crud.get_or_create_video(
            db, video_id=video_id, youtube_url=youtube_url
        )
        crud.mark_video_failed(
            db, video_db_id=video_db.id, error_message=error_message[:1024]
        )
    except Exception as inner_exc:
        logger.warning(f"Failed to persist error state for {video_id!r}: {inner_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /videos — list all processed videos
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/videos",
    response_model=VideoListResponse,
    status_code=status.HTTP_200_OK,
    summary="List summarized videos",
    description="Returns a paginated list of videos that have been processed.",
    tags=["Video"],
)
async def list_videos(
    skip: int = Query(default=0, ge=0, description="Records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    processed_only: bool = Query(default=True, description="Only return processed videos"),
    db: Session = Depends(get_db),
) -> VideoListResponse:
    """
    Returns a paginated list of videos with their embedded summaries.
    Use `skip` and `limit` for pagination. Default page size is 20.
    """
    rid = _new_request_id()
    logger.info(f"[{rid}] GET /videos | skip={skip} limit={limit}")

    videos = crud.get_all_videos(db, skip=skip, limit=limit, processed_only=processed_only)
    # Total count for pagination metadata (unsliced query)
    total_videos = crud.get_all_videos(db, skip=0, limit=10_000, processed_only=processed_only)
    total = len(total_videos)

    items = []
    for v in videos:
        try:
            items.append(_video_record_from_db(v, v.summary))
        except Exception as exc:
            # A single malformed record should not break the whole list
            logger.warning(f"[{rid}] Skipping malformed video db_id={v.id}: {exc}")

    return VideoListResponse(success=True, total=total, skip=skip, limit=limit, items=items)


# ─────────────────────────────────────────────────────────────────────────────
# GET /videos/{video_id} — single video detail
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/videos/{video_id}",
    response_model=VideoDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a video and its summary",
    description="Fetch a single video and its AI-generated summary by YouTube video ID.",
    tags=["Video"],
    responses={
        404: {"model": ErrorResponse, "description": "Video not found in database"},
    },
)
async def get_video(
    video_id: str,
    db: Session = Depends(get_db),
) -> VideoDetailResponse:
    """
    Fetch a previously summarized video by its 11-character YouTube video ID.
    Returns 404 if the video has not been processed yet.
    """
    rid = _new_request_id()
    logger.info(f"[{rid}] GET /videos/{video_id}")

    video_db = crud.get_video_by_video_id(db, video_id=video_id)
    if video_db is None:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"Video '{video_id}' has not been summarized yet.",
                "code": "VIDEO_NOT_FOUND",
                "request_id": rid,
            },
        )

    return VideoDetailResponse(
        success=True,
        data=_video_record_from_db(video_db, video_db.summary),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /videos/{video_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/videos/{video_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a video and all its data",
    description=(
        "Remove a video record, its summary, and all chat sessions from the database. "
        "This action is irreversible."
    ),
    tags=["Video"],
    responses={
        404: {"model": ErrorResponse, "description": "Video not found"},
    },
)
async def delete_video(
    video_id: str,
    db: Session = Depends(get_db),
) -> DeleteResponse:
    rid = _new_request_id()
    logger.info(f"[{rid}] DELETE /videos/{video_id}")

    video_db = crud.get_video_by_video_id(db, video_id=video_id)
    if video_db is None:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"Video '{video_id}' not found.",
                "code": "VIDEO_NOT_FOUND",
                "request_id": rid,
            },
        )

    crud.delete_video(db, video_db_id=video_db.id)
    logger.info(f"[{rid}] Deleted video_id={video_id!r} db_id={video_db.id}")

    return DeleteResponse(
        success=True,
        message=f"Video '{video_id}' and all associated data have been deleted.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with a video",
    description="""
Answer questions about a YouTube video using its transcript as context.

The video must have been summarized first (POST /summarize) so that its
transcript is stored in the database. Pass `session_token` to persist the
chat history across requests.
    """,
    tags=["Video"],
    responses={
        404: {"model": ErrorResponse, "description": "Video not found or not yet summarized"},
        422: {"model": ErrorResponse, "description": "Invalid request"},
        502: {"model": ErrorResponse, "description": "AI provider error"},
    },
)
async def chat_about_video(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Flow:
      1. Validate the URL and extract video_id.
      2. Load the transcript from DB (avoids re-fetching YouTube).
      3. If session_token provided, get-or-create a ChatSession in DB.
      4. Call ChatService with transcript text + chat history.
      5. Persist user message + assistant reply to DB (if session_token set).
      6. Return ChatResponse.
    """
    rid = _new_request_id()
    logger.info(
        f"[{rid}] POST /chat | url={request.youtube_url!r} "
        f"question_len={len(request.question)} "
        f"session={request.session_token!r}"
    )

    # ── Step 1: Extract video ID ──────────────────────────────────────────
    try:
        video_id = _transcript_service.extract_video_id(request.youtube_url)
    except AppBaseException as exc:
        _raise_http(exc, rid)

    # ── Step 2: Load transcript from DB ───────────────────────────────────
    video_db = crud.get_video_by_video_id(db, video_id=video_id)
    if video_db is None or not video_db.transcript_text:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": (
                    f"Video '{video_id}' has not been summarized yet. "
                    "Call POST /summarize first."
                ),
                "code": "VIDEO_NOT_FOUND",
                "request_id": rid,
            },
        )

    # Reconstruct a lightweight VideoTranscript from stored text
    # (no segments needed — ChatService works on full_text)
    from app.models.video import TranscriptSegment, VideoTranscript
    transcript = VideoTranscript(
        video_id=video_id,
        language=video_db.language or "en",
        segments=[
            TranscriptSegment(
                text=video_db.transcript_text,
                start=0.0,
                duration=0.0,
            )
        ],
    )

    # ── Step 3: Resolve chat history ──────────────────────────────────────
    chat_history = list(request.chat_history)  # from request body (client state)

    db_session = None
    if request.session_token:
        try:
            db_session, _ = crud.get_or_create_chat_session(
                db,
                video_db_id=video_db.id,
                session_token=request.session_token,
            )
            # If DB has history, use it (more authoritative than client state)
            db_history = crud.get_chat_history_dicts(db, session_id=db_session.id)
            if db_history:
                chat_history = db_history
                logger.debug(
                    f"[{rid}] Loaded {len(chat_history)} messages from DB session"
                )
        except Exception as exc:
            logger.warning(f"[{rid}] Chat session DB error (non-fatal): {exc}")

    # ── Step 4: AI chat ───────────────────────────────────────────────────
    try:
        chat_response = await _chat_service.chat(
            transcript=transcript,
            question=request.question,
            chat_history=chat_history,
        )
    except AppBaseException as exc:
        logger.warning(f"[{rid}] Chat service error: {exc.message}")
        _raise_http(exc, rid)

    # ── Step 5: Persist messages to DB ────────────────────────────────────
    if db_session is not None:
        try:
            crud.add_chat_message(
                db, session_id=db_session.id, role="user", content=request.question
            )
            crud.add_chat_message(
                db, session_id=db_session.id, role="assistant", content=chat_response.answer
            )
        except Exception as exc:
            logger.warning(f"[{rid}] Chat message persist failed (non-fatal): {exc}")

    logger.info(f"[{rid}] Chat response generated | answer_len={len(chat_response.answer)}")

    return ChatResponse(
        success=True,
        answer=chat_response.answer,
        sources=chat_response.sources,
        session_token=request.session_token,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns application status, version, and database connectivity.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Liveness probe endpoint. Used by load balancers and monitoring systems.
    Includes a live DB connectivity check so infra knows when the DB is down.
    """
    db_ok = health_check_db()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
        database="connected" if db_ok else "unavailable",
    )
