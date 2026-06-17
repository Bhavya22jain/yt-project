"""
schemas/video.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 request/response schemas for the Video Summarizer API.

These are the serialisation contracts — not DB models (those live in
database/db_models.py) and not domain objects (models/video.py).

Design:
  • Every request field is validated at the boundary; bad data never
    reaches a service or database layer.
  • Every response field is typed and documented so Swagger auto-docs
    are useful without additional annotation.
  • Fields that appear in multiple schemas are defined once and reused.
  • Sensitive internal fields (DB ids, raw SQL) never appear in responses.

Sections:
  1. Shared validators / helpers
  2. Request schemas
  3. Core domain response models
  4. Endpoint response envelopes
  5. Error schema
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Shared validators / helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches all common YouTube URL formats:
#   https://www.youtube.com/watch?v=XXXXXXXXXXX
#   https://youtu.be/XXXXXXXXXXX
#   https://youtube.com/watch?v=XXXXXXXXXXX&t=30s
#   https://www.youtube.com/embed/XXXXXXXXXXX
#   https://m.youtube.com/watch?v=XXXXXXXXXXX
_YOUTUBE_PATTERN = re.compile(
    r"(?:https?://)?(?:(?:www|m)\.)?(?:"
    r"youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/)"
    r"|youtu\.be/"
    r")([a-zA-Z0-9_-]{11})"
)


def _validate_youtube_url(v: str) -> str:
    """
    Shared YouTube URL validator used by multiple request schemas.

    Accepts all standard YouTube URL shapes, strips leading/trailing
    whitespace, and raises ValueError (→ HTTP 422) if the URL is invalid.
    """
    v = v.strip()
    if not v:
        raise ValueError("YouTube URL must not be empty.")
    if not _YOUTUBE_PATTERN.search(v):
        raise ValueError(
            f"'{v}' is not a recognised YouTube URL. "
            "Accepted formats: youtube.com/watch?v=…, youtu.be/…, "
            "youtube.com/shorts/…, youtube.com/embed/…"
        )
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 2. Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    """
    Payload for POST /api/v1/summarize.

    Attributes:
        youtube_url:  Any valid YouTube video URL.
        force_refresh: If True, bypass cached summary and re-summarize.
                       Useful when a video's auto-captions improve over time.
        language_preference: BCP-47 language codes in priority order.
                             Falls back to any available transcript if none match.
    """

    youtube_url: str = Field(
        ...,
        description="Full YouTube video URL (any standard format)",
        examples=[
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
        ],
        min_length=10,
        max_length=2048,
    )
    force_refresh: bool = Field(
        default=False,
        description=(
            "Set true to re-fetch the transcript and regenerate the AI summary "
            "even if a cached result already exists in the database."
        ),
    )
    language_preference: List[str] = Field(
        default_factory=list,
        description=(
            "Preferred transcript language codes in priority order, e.g. ['en', 'es']. "
            "Falls back to any available language if no preference matches."
        ),
        max_length=10,
    )

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        return _validate_youtube_url(v)

    @field_validator("language_preference")
    @classmethod
    def validate_language_codes(cls, v: List[str]) -> List[str]:
        """Normalise language codes to lowercase and strip whitespace."""
        return [lang.strip().lower() for lang in v if lang.strip()]

    model_config = {"json_schema_extra": {
        "examples": [{
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "force_refresh": False,
            "language_preference": ["en"],
        }]
    }}


class ChatRequest(BaseModel):
    """
    Payload for POST /api/v1/chat.

    Attributes:
        youtube_url:   YouTube URL identifying which video to chat about.
        question:      The user's current question (3–1000 chars).
        chat_history:  Previous turns for multi-turn conversations.
        session_token: Client UUID that groups messages into a chat session.
                       Generate once per browser session and reuse.
    """

    youtube_url: str = Field(
        ...,
        description="YouTube URL of the video to chat about",
        min_length=10,
        max_length=2048,
    )
    question: str = Field(
        ...,
        description="User's question about the video content",
        min_length=3,
        max_length=1000,
    )
    chat_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Previous conversation turns. Each item must have 'role' "
            "('user' or 'assistant') and 'content' (non-empty string)."
        ),
    )
    session_token: Optional[str] = Field(
        default=None,
        description=(
            "Client-generated UUID that groups messages into a persistent "
            "chat session stored in the database. If omitted, no session is saved."
        ),
        max_length=64,
    )

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        return _validate_youtube_url(v)

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        """Strip whitespace; enforce min length after stripping."""
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question must be at least 3 characters after trimming.")
        return v

    @field_validator("chat_history")
    @classmethod
    def validate_chat_history(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate that every history item has the expected shape.
        Silently drops malformed items rather than rejecting the whole request —
        a corrupted history item should not block the current question.
        """
        valid = []
        for item in v:
            if (
                isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
                and item["content"].strip()
            ):
                valid.append({"role": item["role"], "content": item["content"]})
        return valid

    @field_validator("session_token")
    @classmethod
    def clean_session_token(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else None

    model_config = {"json_schema_extra": {
        "examples": [{
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "question": "What are the main points of this video?",
            "chat_history": [],
            "session_token": "550e8400-e29b-41d4-a716-446655440000",
        }]
    }}


class VideoHistoryParams(BaseModel):
    """
    Query parameters for GET /api/v1/videos.
    Defined as a schema for documentation; parsed manually in the route.
    """
    skip: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum records to return")
    processed_only: bool = Field(
        default=True,
        description="If true, only return successfully processed videos",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core domain response models
# ─────────────────────────────────────────────────────────────────────────────

class TimestampItem(BaseModel):
    """A single important moment in the video."""
    time: str = Field(..., description="Timestamp in M:SS or H:MM:SS format, e.g. '4:32'")
    description: str = Field(..., description="What is being discussed at this moment")


class VideoSummary(BaseModel):
    """
    Structured AI-generated summary of a YouTube video.
    Produced by SummaryService and stored in the summaries table.
    """
    video_id: str = Field(..., description="YouTube 11-character video ID")
    title: Optional[str] = Field(None, description="Video title from YouTube metadata")
    duration: Optional[str] = Field(None, description="Video duration as M:SS or H:MM:SS")
    executive_summary: str = Field(
        ...,
        description="2–3 sentence TL;DR of the entire video",
    )
    detailed_summary: str = Field(
        ...,
        description="Thorough paragraph-level breakdown of all major sections",
    )
    key_points: List[str] = Field(
        ...,
        description="5–10 standalone key takeaways from the video",
    )
    action_items: List[str] = Field(
        ...,
        description="Concrete next steps a viewer can act on",
    )
    important_timestamps: List[TimestampItem] = Field(
        default_factory=list,
        description="Notable moments with their timestamps",
    )


class VideoRecord(BaseModel):
    """
    A video as stored in the database — returned by list/get endpoints.
    Adds persistence metadata (db_id, processed state, timestamps) to
    the core summary data.
    """
    db_id: int = Field(..., description="Internal database primary key")
    video_id: str = Field(..., description="YouTube 11-character video ID")
    youtube_url: str = Field(..., description="Original URL submitted by the user")
    title: Optional[str] = None
    channel: Optional[str] = None
    duration: Optional[str] = None
    language: Optional[str] = None
    is_processed: bool = Field(..., description="True once summarization completed")
    summarize_count: int = Field(..., description="How many times this video was summarized")
    word_count: Optional[int] = Field(None, description="Approximate transcript word count")
    transcript: Optional[str] = Field(
        None,
        description="Full transcript text, present when available",
    )
    created_at: datetime
    updated_at: datetime
    summary: Optional[VideoSummary] = Field(
        None,
        description="Embedded summary, present when is_processed=True",
    )

    model_config = {"from_attributes": True}


class ChatMessage(BaseModel):
    """A single turn in a chat conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Endpoint response envelopes
# ─────────────────────────────────────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    """
    Response envelope for POST /api/v1/summarize.

    Attributes:
        success:       Always True on 200 responses.
        cached:        True if the result was served from the database
                       rather than freshly generated.
        processing_ms: Wall-clock time in milliseconds (0 if cached).
        data:          The full structured summary.
    """
    success: bool = True
    cached: bool = Field(
        default=False,
        description="True if this result was fetched from the DB cache",
    )
    processing_ms: int = Field(
        default=0,
        description="End-to-end processing time in milliseconds",
    )
    data: VideoSummary


class VideoListResponse(BaseModel):
    """Response envelope for GET /api/v1/videos."""
    success: bool = True
    total: int = Field(..., description="Total matching records (before pagination)")
    skip: int
    limit: int
    items: List[VideoRecord]


class VideoDetailResponse(BaseModel):
    """Response envelope for GET /api/v1/videos/{video_id}."""
    success: bool = True
    data: VideoRecord


class DeleteResponse(BaseModel):
    """Response envelope for DELETE /api/v1/videos/{video_id}."""
    success: bool = True
    message: str


class ChatResponse(BaseModel):
    """Response envelope for POST /api/v1/chat."""
    success: bool = True
    answer: str = Field(..., description="AI answer grounded in the transcript")
    sources: List[str] = Field(
        default_factory=list,
        description="Relevant transcript excerpts the answer is based on",
    )
    session_token: Optional[str] = Field(
        None,
        description="Echoed back so the client can persist the session token",
    )


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""
    status: str = "ok"
    version: str
    environment: str
    database: str = Field(
        ...,
        description="'connected' or 'unavailable'",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error schema
# ─────────────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """A single validation error detail item."""
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    """
    Standardised error envelope returned on all 4xx / 5xx responses.

    Attributes:
        success: Always False.
        error:   Human-readable summary of what went wrong.
        code:    Machine-readable error code for client-side handling.
        details: Optional list of field-level validation errors.
        request_id: Echoed from X-Request-ID header for log correlation.
    """
    success: bool = False
    error: str
    code: Optional[str] = Field(
        None,
        description="Machine-readable error code, e.g. 'TRANSCRIPT_NOT_AVAILABLE'",
    )
    details: List[ErrorDetail] = Field(default_factory=list)
    request_id: Optional[str] = None
