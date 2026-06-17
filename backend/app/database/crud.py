"""
database/crud.py
─────────────────────────────────────────────────────────────────────────────
CRUD (Create, Read, Update, Delete) functions for every database table.

Design principles:
  • Every function accepts a `db: Session` as its first argument.
    The session is provided by the `get_db()` FastAPI dependency and is
    request-scoped; CRUD functions never open their own sessions.

  • Functions raise ValueError for business-rule violations (e.g. not found)
    so callers can convert them to HTTP 404/422 at the router layer.

  • JSON list columns (key_points, action_items, timestamps) are serialised /
    deserialised transparently using `_dumps` / `_loads` helpers.

  • Every write function commits and returns the refreshed ORM object, so
    callers always get up-to-date data (including server-set timestamps).

  • No business logic here — CRUD only reads/writes database rows.

Sections:
  1. Helpers
  2. Video CRUD
  3. Summary CRUD
  4. Chat Session CRUD
  5. Chat Message CRUD
  6. Composite helpers (cross-table convenience)
"""

import json
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.db_models import (
    ChatMessageDB,
    ChatSessionDB,
    SummaryDB,
    VideoDB,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dumps(obj: Any) -> str:
    """Serialise a Python object to a compact JSON string."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _loads(raw: str) -> Any:
    """Deserialise a JSON string; returns an empty list on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        preview = repr(raw)[:80]
        logger.warning(f"JSON decode failed for value: {preview}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Video CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_video(
    db: Session,
    *,
    video_id: str,
    youtube_url: str,
    title: Optional[str] = None,
    channel: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    thumbnail_url: Optional[str] = None,
    language: Optional[str] = None,
    transcript_text: Optional[str] = None,
    transcript_word_count: Optional[int] = None,
) -> VideoDB:
    """
    Insert a new video row.

    Raises:
        ValueError: A video with this video_id already exists.
                    Use `get_or_create_video` when upsert semantics are needed.
    """
    existing = get_video_by_video_id(db, video_id=video_id)
    if existing:
        raise ValueError(
            f"Video with video_id={video_id!r} already exists (id={existing.id}). "
            "Use get_or_create_video() for upsert semantics."
        )

    video = VideoDB(
        video_id=video_id,
        youtube_url=youtube_url,
        title=title,
        channel=channel,
        duration_seconds=duration_seconds,
        thumbnail_url=thumbnail_url,
        language=language,
        transcript_text=transcript_text,
        transcript_word_count=transcript_word_count,
        is_processed=False,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    logger.debug(f"Created VideoDB id={video.id} video_id={video_id!r}")
    return video


def get_video_by_id(db: Session, *, video_db_id: int) -> Optional[VideoDB]:
    """Fetch a video by its internal integer primary key."""
    return db.get(VideoDB, video_db_id)


def get_video_by_video_id(db: Session, *, video_id: str) -> Optional[VideoDB]:
    """
    Fetch a video by YouTube video ID (the 11-char string).
    Returns None if not found — does NOT raise.
    """
    stmt = select(VideoDB).where(VideoDB.video_id == video_id)
    return db.execute(stmt).scalar_one_or_none()


def get_all_videos(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 100,
    processed_only: bool = False,
) -> list[VideoDB]:
    """
    Return a paginated list of videos, newest first.

    Args:
        skip:           Number of rows to skip (for pagination).
        limit:          Maximum rows to return.
        processed_only: If True, only return videos where is_processed=True.
    """
    stmt = select(VideoDB).order_by(VideoDB.created_at.desc())
    if processed_only:
        stmt = stmt.where(VideoDB.is_processed.is_(True))
    stmt = stmt.offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def get_or_create_video(
    db: Session,
    *,
    video_id: str,
    youtube_url: str,
    **kwargs: Any,
) -> tuple[VideoDB, bool]:
    """
    Return (video, created) — fetch existing or insert new.

    This is the primary entry point for the summarize endpoint so that
    re-summarizing the same video updates rather than duplicates it.

    Returns:
        (VideoDB, True)  if a new row was inserted.
        (VideoDB, False) if an existing row was returned.
    """
    video = get_video_by_video_id(db, video_id=video_id)
    if video:
        logger.debug(f"Found existing VideoDB id={video.id} video_id={video_id!r}")
        return video, False

    video = create_video(db, video_id=video_id, youtube_url=youtube_url, **kwargs)
    return video, True


def update_video(
    db: Session,
    *,
    video_db_id: int,
    **fields: Any,
) -> VideoDB:
    """
    Partially update a video row with the supplied keyword fields.

    Raises:
        ValueError: No video found with this id.

    Example:
        update_video(db, video_db_id=3, is_processed=True, title="My Video")
    """
    video = get_video_by_id(db, video_db_id=video_db_id)
    if not video:
        raise ValueError(f"Video with id={video_db_id} not found.")

    _ALLOWED_VIDEO_FIELDS = {
        "title", "channel", "duration_seconds", "thumbnail_url",
        "language", "transcript_text", "transcript_word_count",
        "is_processed", "process_error", "summarize_count", "youtube_url",
    }
    for key, value in fields.items():
        if key not in _ALLOWED_VIDEO_FIELDS:
            raise ValueError(f"update_video: field '{key}' is not updatable.")
        setattr(video, key, value)

    video.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(video)
    logger.debug(f"Updated VideoDB id={video_db_id} fields={list(fields.keys())}")
    return video


def mark_video_processed(
    db: Session,
    *,
    video_db_id: int,
) -> VideoDB:
    """
    Convenience wrapper: set is_processed=True and increment summarize_count.
    Called by SummaryService after a successful summarization.
    """
    video = get_video_by_id(db, video_db_id=video_db_id)
    if not video:
        raise ValueError(f"Video with id={video_db_id} not found.")

    video.is_processed = True
    video.process_error = None
    video.summarize_count = (video.summarize_count or 0) + 1
    video.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(video)
    return video


def mark_video_failed(
    db: Session,
    *,
    video_db_id: int,
    error_message: str,
) -> VideoDB:
    """
    Set is_processed=False and record the error message.
    Called by SummaryService when summarization fails.
    """
    return update_video(
        db,
        video_db_id=video_db_id,
        is_processed=False,
        process_error=error_message[:1024],
    )


def delete_video(db: Session, *, video_db_id: int) -> bool:
    """
    Delete a video (and its summary + chat sessions via CASCADE).

    Returns:
        True  if the row was deleted.
        False if no row with this id existed.
    """
    video = get_video_by_id(db, video_db_id=video_db_id)
    if not video:
        return False

    db.delete(video)
    db.commit()
    logger.info(f"Deleted VideoDB id={video_db_id}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Summary CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_summary(
    db: Session,
    *,
    video_db_id: int,
    model_name: str,
    executive_summary: str,
    detailed_summary: str,
    key_points: list[str],
    action_items: list[str],
    important_timestamps: list[dict],
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    generation_time_ms: Optional[int] = None,
) -> SummaryDB:
    """
    Insert a new summary row.

    List arguments (key_points, action_items, important_timestamps) are
    automatically JSON-serialised before storage.

    Raises:
        ValueError: A summary already exists for this video_db_id.
                    Use upsert_summary() when re-summarization is expected.
    """
    existing = get_summary_by_video_db_id(db, video_db_id=video_db_id)
    if existing:
        raise ValueError(
            f"Summary for video_db_id={video_db_id} already exists. "
            "Use upsert_summary() to replace it."
        )

    summary = SummaryDB(
        video_db_id=video_db_id,
        model_name=model_name,
        executive_summary=executive_summary,
        detailed_summary=detailed_summary,
        key_points_json=_dumps(key_points),
        action_items_json=_dumps(action_items),
        important_timestamps_json=_dumps(important_timestamps),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        generation_time_ms=generation_time_ms,
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    logger.debug(f"Created SummaryDB id={summary.id} for video_db_id={video_db_id}")
    return summary


def upsert_summary(
    db: Session,
    *,
    video_db_id: int,
    model_name: str,
    executive_summary: str,
    detailed_summary: str,
    key_points: list[str],
    action_items: list[str],
    important_timestamps: list[dict],
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    generation_time_ms: Optional[int] = None,
) -> SummaryDB:
    """
    Insert a new summary or replace an existing one for the same video.

    This is the main entry point for SummaryService — handles both first-time
    summarization and re-summarization of the same video.

    Returns the final SummaryDB object.
    """
    existing = get_summary_by_video_db_id(db, video_db_id=video_db_id)

    if existing:
        # Update in place
        existing.model_name = model_name
        existing.executive_summary = executive_summary
        existing.detailed_summary = detailed_summary
        existing.key_points_json = _dumps(key_points)
        existing.action_items_json = _dumps(action_items)
        existing.important_timestamps_json = _dumps(important_timestamps)
        existing.prompt_tokens = prompt_tokens
        existing.completion_tokens = completion_tokens
        existing.generation_time_ms = generation_time_ms
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        logger.debug(f"Updated SummaryDB id={existing.id} for video_db_id={video_db_id}")
        return existing

    return create_summary(
        db,
        video_db_id=video_db_id,
        model_name=model_name,
        executive_summary=executive_summary,
        detailed_summary=detailed_summary,
        key_points=key_points,
        action_items=action_items,
        important_timestamps=important_timestamps,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        generation_time_ms=generation_time_ms,
    )


def get_summary_by_id(db: Session, *, summary_id: int) -> Optional[SummaryDB]:
    """Fetch a summary by its internal primary key."""
    return db.get(SummaryDB, summary_id)


def get_summary_by_video_db_id(
    db: Session, *, video_db_id: int
) -> Optional[SummaryDB]:
    """Fetch the summary for a given internal video id. Returns None if absent."""
    stmt = select(SummaryDB).where(SummaryDB.video_db_id == video_db_id)
    return db.execute(stmt).scalar_one_or_none()


def get_summary_by_youtube_id(
    db: Session, *, video_id: str
) -> Optional[SummaryDB]:
    """
    Fetch a summary by YouTube video ID (the 11-char string).
    Joins videos → summaries in one query.
    """
    stmt = (
        select(SummaryDB)
        .join(VideoDB, VideoDB.id == SummaryDB.video_db_id)
        .where(VideoDB.video_id == video_id)
    )
    return db.execute(stmt).scalar_one_or_none()


def get_key_points(summary: SummaryDB) -> list[str]:
    """Deserialise the key_points_json column into a Python list."""
    return _loads(summary.key_points_json)


def get_action_items(summary: SummaryDB) -> list[str]:
    """Deserialise the action_items_json column into a Python list."""
    return _loads(summary.action_items_json)


def get_important_timestamps(summary: SummaryDB) -> list[dict]:
    """Deserialise the important_timestamps_json column."""
    return _loads(summary.important_timestamps_json)


def delete_summary(db: Session, *, summary_id: int) -> bool:
    """Delete a summary by primary key. Returns True if deleted, False if absent."""
    summary = get_summary_by_id(db, summary_id=summary_id)
    if not summary:
        return False
    db.delete(summary)
    db.commit()
    logger.info(f"Deleted SummaryDB id={summary_id}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chat Session CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_chat_session(
    db: Session,
    *,
    video_db_id: int,
    session_token: str,
) -> ChatSessionDB:
    """
    Create a new chat session for a (video, client-session) pair.

    Args:
        video_db_id:   Internal integer FK to videos.id.
        session_token: UUID generated by the Streamlit frontend.

    Raises:
        ValueError: A session with this token already exists for this video.
    """
    existing = get_chat_session_by_token(
        db, video_db_id=video_db_id, session_token=session_token
    )
    if existing:
        raise ValueError(
            f"Chat session already exists for video_db_id={video_db_id} "
            f"token={session_token!r}."
        )

    session = ChatSessionDB(
        video_db_id=video_db_id,
        session_token=session_token,
        message_count=0,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.debug(f"Created ChatSessionDB id={session.id} token={session_token!r}")
    return session


def get_or_create_chat_session(
    db: Session,
    *,
    video_db_id: int,
    session_token: str,
) -> tuple[ChatSessionDB, bool]:
    """
    Return (session, created).
    Main entry point for the chat endpoint — idempotent across page refreshes.
    """
    session = get_chat_session_by_token(
        db, video_db_id=video_db_id, session_token=session_token
    )
    if session:
        return session, False

    session = create_chat_session(
        db, video_db_id=video_db_id, session_token=session_token
    )
    return session, True


def get_chat_session_by_id(
    db: Session, *, session_id: int
) -> Optional[ChatSessionDB]:
    """Fetch a chat session by primary key."""
    return db.get(ChatSessionDB, session_id)


def get_chat_session_by_token(
    db: Session,
    *,
    video_db_id: int,
    session_token: str,
) -> Optional[ChatSessionDB]:
    """Fetch a chat session by (video_db_id, session_token) pair."""
    stmt = select(ChatSessionDB).where(
        ChatSessionDB.video_db_id == video_db_id,
        ChatSessionDB.session_token == session_token,
    )
    return db.execute(stmt).scalar_one_or_none()


def get_chat_sessions_for_video(
    db: Session, *, video_db_id: int
) -> list[ChatSessionDB]:
    """Return all chat sessions for a given video, newest first."""
    stmt = (
        select(ChatSessionDB)
        .where(ChatSessionDB.video_db_id == video_db_id)
        .order_by(ChatSessionDB.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def delete_chat_session(db: Session, *, session_id: int) -> bool:
    """Delete a chat session and all its messages (CASCADE). Returns True if deleted."""
    session = get_chat_session_by_id(db, session_id=session_id)
    if not session:
        return False
    db.delete(session)
    db.commit()
    logger.info(f"Deleted ChatSessionDB id={session_id}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chat Message CRUD
# ─────────────────────────────────────────────────────────────────────────────

def add_chat_message(
    db: Session,
    *,
    session_id: int,
    role: str,
    content: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    response_time_ms: Optional[int] = None,
) -> ChatMessageDB:
    """
    Append a new message to a chat session.

    Position is auto-computed from the current message_count on the session
    (no gap, no collision). Also increments session.message_count.

    Args:
        session_id:   FK to chat_sessions.id.
        role:         'user' or 'assistant'.
        content:      Message text.

    Raises:
        ValueError: Unknown session_id or invalid role.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}.")

    session = get_chat_session_by_id(db, session_id=session_id)
    if not session:
        raise ValueError(f"ChatSession id={session_id} not found.")

    position = session.message_count  # 0-based

    message = ChatMessageDB(
        session_id=session_id,
        role=role,
        content=content,
        position=position,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=response_time_ms,
    )
    db.add(message)

    # Keep the denormalised counter in sync
    session.message_count = position + 1
    session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)
    logger.debug(
        f"Added ChatMessageDB id={message.id} role={role!r} "
        f"pos={position} session_id={session_id}"
    )
    return message


def get_messages_for_session(
    db: Session, *, session_id: int
) -> list[ChatMessageDB]:
    """Return all messages for a session in turn order (position ASC)."""
    stmt = (
        select(ChatMessageDB)
        .where(ChatMessageDB.session_id == session_id)
        .order_by(ChatMessageDB.position.asc())
    )
    return list(db.execute(stmt).scalars().all())


def get_chat_history_dicts(
    db: Session, *, session_id: int
) -> list[dict[str, str]]:
    """
    Return the chat history as a list of dicts compatible with the
    Anthropic API messages format: [{"role": "user", "content": "..."}].

    This is the format consumed by ChatService.
    """
    messages = get_messages_for_session(db, session_id=session_id)
    return [{"role": m.role, "content": m.content} for m in messages]


def get_last_n_messages(
    db: Session,
    *,
    session_id: int,
    n: int = 10,
) -> list[ChatMessageDB]:
    """
    Return the last N messages for a session (most recent turns).
    Useful for trimming context before sending to the AI.
    """
    stmt = (
        select(ChatMessageDB)
        .where(ChatMessageDB.session_id == session_id)
        .order_by(ChatMessageDB.position.desc())
        .limit(n)
    )
    # Reverse to restore chronological order
    return list(reversed(db.execute(stmt).scalars().all()))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Composite helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_video_with_summary(
    db: Session, *, video_id: str
) -> Optional[tuple[VideoDB, SummaryDB]]:
    """
    Fetch a (VideoDB, SummaryDB) pair by YouTube video ID in a single round-trip.
    Returns None if the video doesn't exist or has no summary yet.
    """
    video = get_video_by_video_id(db, video_id=video_id)
    if not video or not video.summary:
        return None
    return video, video.summary


def get_recent_videos_with_summaries(
    db: Session,
    *,
    limit: int = 10,
) -> list[tuple[VideoDB, SummaryDB]]:
    """
    Return the N most recently processed videos with their summaries.
    Useful for a "recently summarized" sidebar in the UI.
    """
    stmt = (
        select(VideoDB)
        .where(VideoDB.is_processed.is_(True))
        .order_by(VideoDB.updated_at.desc())
        .limit(limit)
    )
    videos = list(db.execute(stmt).scalars().all())
    return [(v, v.summary) for v in videos if v.summary is not None]
