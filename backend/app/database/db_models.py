"""
database/db_models.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy 2.0 ORM table definitions.

Tables:
  • videos        — one row per unique YouTube video processed
  • summaries     — one-to-one with videos; stores the AI-generated summary
  • chat_sessions — groups chat messages for a (video, user_session) pair
  • chat_messages — individual turns within a chat session

Design decisions:
  • All primary keys are auto-incrementing integers (simple, fast for SQLite).
  • video_id (YouTube's 11-char ID) has a unique index — used as the lookup key.
  • JSON columns (key_points, action_items, timestamps) stored as TEXT + JSON
    serialisation in CRUD. SQLite has no native JSON column; this is idiomatic.
  • Timestamps use UTC; `onupdate` keeps `updated_at` current automatically.
  • Soft relationships: summaries.video_id FK → videos.id with CASCADE DELETE.
    Deleting a video row removes its summary automatically.

Naming convention:
  ORM classes:   VideoDB, SummaryDB, ChatSessionDB, ChatMessageDB
  ("DB" suffix distinguishes from the domain dataclasses in models/video.py)
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# VideoDB
# ─────────────────────────────────────────────────────────────────────────────

class VideoDB(Base):
    """
    Represents a YouTube video that has been processed by the application.

    One row is created the first time a video URL is summarized.
    Re-summarizing the same video updates the existing row rather than
    inserting a duplicate (enforced by the unique index on `video_id`).
    """

    __tablename__ = "videos"

    # ── Primary key ───────────────────────────────────────────────
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Internal surrogate key",
    )

    # ── YouTube identity ──────────────────────────────────────────
    video_id: Mapped[str] = mapped_column(
        String(11),
        nullable=False,
        unique=True,
        index=True,
        comment="YouTube 11-character video ID, e.g. 'dQw4w9WgXcQ'",
    )
    youtube_url: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original URL as submitted by the user",
    )

    # ── Video metadata (may be None if fetch failed) ───────────────
    title: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="Video title from YouTube",
    )
    channel: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Channel / author name",
    )
    duration_seconds: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Video length in seconds",
    )
    thumbnail_url: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )
    language: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        comment="Transcript language code, e.g. 'en'",
    )

    # ── Transcript ────────────────────────────────────────────────
    transcript_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Full transcript as a single string (cached for chat reuse)",
    )
    transcript_word_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ── Processing state ──────────────────────────────────────────
    is_processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True once summarization completed successfully",
    )
    process_error: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="Last error message if processing failed",
    )
    summarize_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="How many times this video has been re-summarized",
    )

    # ── Timestamps ────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────
    # back_populates wires the two sides of the relationship together.
    # uselist=False makes it one-to-one (a video has at most one summary).
    summary: Mapped[Optional["SummaryDB"]] = relationship(
        "SummaryDB",
        back_populates="video",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="select",
    )
    chat_sessions: Mapped[list["ChatSessionDB"]] = relationship(
        "ChatSessionDB",
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="select",
    )

    @property
    def metadata_duration(self) -> Optional[str]:
        """Return duration_seconds as a human-readable M:SS / H:MM:SS string."""
        if self.duration_seconds is None:
            return None
        h = self.duration_seconds // 3600
        m = (self.duration_seconds % 3600) // 60
        s = self.duration_seconds % 60
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def __repr__(self) -> str:
        return (
            f"<VideoDB id={self.id} video_id={self.video_id!r} "
            f"title={self.title!r} processed={self.is_processed}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SummaryDB
# ─────────────────────────────────────────────────────────────────────────────

class SummaryDB(Base):
    """
    Stores the AI-generated structured summary for a video.

    One-to-one with VideoDB. List fields (key_points, action_items,
    important_timestamps) are stored as JSON strings and deserialised
    by CRUD helpers — SQLite has no native array type.
    """

    __tablename__ = "summaries"

    # ── Primary key ───────────────────────────────────────────────
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ── Foreign key → videos ──────────────────────────────────────
    video_db_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,        # enforces one-to-one
        index=True,
        comment="FK to videos.id",
    )

    # ── AI model used ─────────────────────────────────────────────
    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Anthropic model that generated this summary",
    )

    # ── Summary content ───────────────────────────────────────────
    executive_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="2–3 sentence TL;DR",
    )
    detailed_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Paragraph-level content breakdown",
    )

    # JSON-serialised lists — use crud.py helpers to read/write
    key_points_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        comment='JSON array of strings, e.g. ["Point A", "Point B"]',
    )
    action_items_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        comment='JSON array of strings',
    )
    important_timestamps_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        comment='JSON array of {time, description} objects',
    )

    # ── Generation metadata ───────────────────────────────────────
    prompt_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Tokens in the prompt sent to the AI",
    )
    completion_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Tokens in the AI's response",
    )
    generation_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Wall-clock time for the AI call in milliseconds",
    )

    # ── Timestamps ────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationship ──────────────────────────────────────────────
    video: Mapped["VideoDB"] = relationship(
        "VideoDB",
        back_populates="summary",
    )

    def __repr__(self) -> str:
        return (
            f"<SummaryDB id={self.id} video_db_id={self.video_db_id} "
            f"model={self.model_name!r}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ChatSessionDB
# ─────────────────────────────────────────────────────────────────────────────

class ChatSessionDB(Base):
    """
    Groups a sequence of chat messages for a (video, browser-session) pair.

    A new session is created each time a user starts chatting about a video.
    `session_token` is a client-generated UUID stored in Streamlit session state
    and sent with every chat request so messages can be grouped correctly.
    """

    __tablename__ = "chat_sessions"
    __table_args__ = (
        UniqueConstraint("video_db_id", "session_token", name="uq_video_session"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    video_db_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Client-provided UUID that ties Streamlit session → DB session
    session_token: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="UUID generated by the Streamlit frontend",
    )

    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Denormalised counter for quick display; kept in sync by CRUD",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────
    video: Mapped["VideoDB"] = relationship("VideoDB", back_populates="chat_sessions")
    messages: Mapped[list["ChatMessageDB"]] = relationship(
        "ChatMessageDB",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessageDB.position",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<ChatSessionDB id={self.id} token={self.session_token!r} "
            f"messages={self.message_count}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ChatMessageDB
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessageDB(Base):
    """
    A single turn in a chat session.
    role is either 'user' or 'assistant' — matches the Anthropic API convention.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="'user' or 'assistant'",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Message text",
    )
    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="0-based turn index within the session; used for ordering",
    )

    # Token usage for the assistant's turn (null for user messages)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationship ──────────────────────────────────────────────
    session: Mapped["ChatSessionDB"] = relationship(
        "ChatSessionDB", back_populates="messages"
    )

    def __repr__(self) -> str:
        preview = self.content[:40].replace("\n", " ")
        return f"<ChatMessageDB id={self.id} role={self.role!r} pos={self.position} '{preview}…'>"
