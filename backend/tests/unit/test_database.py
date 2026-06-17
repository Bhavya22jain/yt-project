"""
tests/unit/test_database.py
────────────────────────────
Day 2: Unit tests for the full database layer.

Strategy:
  • Uses an in-memory SQLite database (sqlite:///:memory:) via a pytest
    fixture that overrides the module-level `engine` before any table
    is created. Each test function gets a fresh, isolated database.
  • Tests cover: table creation, every CRUD function, edge cases,
    JSON serialisation helpers, and CASCADE behaviour.
"""

import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.database import Base
from app.database.db_models import VideoDB, SummaryDB, ChatSessionDB, ChatMessageDB
from app.database import crud


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db() -> Session:
    """
    Provide a fresh in-memory SQLite session for each test.
    Tables are created before the test and dropped after.
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


@pytest.fixture
def sample_video(db: Session) -> VideoDB:
    """Pre-inserted video row for tests that need existing data."""
    video, _ = crud.get_or_create_video(
        db,
        video_id="dQw4w9WgXcQ",
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        channel="Rick Astley",
        duration_seconds=212,
        language="en",
        transcript_text="We're no strangers to love you know the rules and so do I.",
        transcript_word_count=14,
    )
    return video


@pytest.fixture
def sample_summary(db: Session, sample_video: VideoDB) -> SummaryDB:
    """Pre-inserted summary row for tests that need existing data."""
    return crud.upsert_summary(
        db,
        video_db_id=sample_video.id,
        model_name="claude-opus-4-20250514",
        executive_summary="A classic 80s pop song about devotion.",
        detailed_summary="Rick Astley promises never to give up on the listener.",
        key_points=["Devotion", "Classic 80s sound", "Iconic music video"],
        action_items=["Listen to the full album", "Watch the music video"],
        important_timestamps=[{"time": "0:10", "description": "Intro beat drops"}],
        prompt_tokens=500,
        completion_tokens=200,
        generation_time_ms=1800,
    )


@pytest.fixture
def sample_session(db: Session, sample_video: VideoDB) -> ChatSessionDB:
    """Pre-inserted chat session."""
    session, _ = crud.get_or_create_chat_session(
        db,
        video_db_id=sample_video.id,
        session_token="test-token-abc-123",
    )
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Table creation
# ─────────────────────────────────────────────────────────────────────────────

class TestTableCreation:

    def test_all_tables_exist(self, db: Session):
        """All four ORM tables must be present after init."""
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        assert "videos" in tables
        assert "summaries" in tables
        assert "chat_sessions" in tables
        assert "chat_messages" in tables

    def test_videos_columns(self, db: Session):
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        cols = {c["name"] for c in inspector.get_columns("videos")}
        required = {
            "id", "video_id", "youtube_url", "title", "channel",
            "is_processed", "transcript_text", "created_at", "updated_at",
        }
        assert required.issubset(cols)

    def test_summaries_columns(self, db: Session):
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        cols = {c["name"] for c in inspector.get_columns("summaries")}
        required = {
            "id", "video_db_id", "model_name", "executive_summary",
            "detailed_summary", "key_points_json", "action_items_json",
            "important_timestamps_json",
        }
        assert required.issubset(cols)


# ─────────────────────────────────────────────────────────────────────────────
# Video CRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoCRUD:

    def test_create_video(self, db: Session):
        video = crud.create_video(
            db,
            video_id="abc12345678",
            youtube_url="https://youtube.com/watch?v=abc12345678",
            title="Test Video",
        )
        assert video.id is not None
        assert video.video_id == "abc12345678"
        assert video.title == "Test Video"
        assert video.is_processed is False

    def test_create_duplicate_video_raises(self, db: Session, sample_video: VideoDB):
        with pytest.raises(ValueError, match="already exists"):
            crud.create_video(
                db,
                video_id=sample_video.video_id,
                youtube_url=sample_video.youtube_url,
            )

    def test_get_video_by_id(self, db: Session, sample_video: VideoDB):
        fetched = crud.get_video_by_id(db, video_db_id=sample_video.id)
        assert fetched is not None
        assert fetched.video_id == sample_video.video_id

    def test_get_video_by_video_id(self, db: Session, sample_video: VideoDB):
        fetched = crud.get_video_by_video_id(db, video_id="dQw4w9WgXcQ")
        assert fetched is not None
        assert fetched.id == sample_video.id

    def test_get_nonexistent_video_returns_none(self, db: Session):
        result = crud.get_video_by_video_id(db, video_id="xxxxxxxxxxx")
        assert result is None

    def test_get_or_create_returns_existing(self, db: Session, sample_video: VideoDB):
        video, created = crud.get_or_create_video(
            db,
            video_id=sample_video.video_id,
            youtube_url=sample_video.youtube_url,
        )
        assert created is False
        assert video.id == sample_video.id

    def test_get_or_create_inserts_new(self, db: Session):
        video, created = crud.get_or_create_video(
            db,
            video_id="newvideo1234",
            youtube_url="https://youtube.com/watch?v=newvideo1234",
        )
        assert created is True
        assert video.id is not None

    def test_get_all_videos_empty(self, db: Session):
        assert crud.get_all_videos(db) == []

    def test_get_all_videos_returns_all(self, db: Session, sample_video: VideoDB):
        crud.create_video(
            db, video_id="second00001", youtube_url="https://youtube.com/watch?v=second00001"
        )
        videos = crud.get_all_videos(db)
        assert len(videos) == 2

    def test_get_all_videos_processed_only_filter(self, db: Session, sample_video: VideoDB):
        crud.mark_video_processed(db, video_db_id=sample_video.id)
        crud.create_video(
            db, video_id="unprocessed1", youtube_url="https://youtube.com/watch?v=unprocessed1"
        )
        processed = crud.get_all_videos(db, processed_only=True)
        assert all(v.is_processed for v in processed)
        assert len(processed) == 1

    def test_update_video_fields(self, db: Session, sample_video: VideoDB):
        updated = crud.update_video(
            db, video_db_id=sample_video.id, title="Updated Title", channel="New Channel"
        )
        assert updated.title == "Updated Title"
        assert updated.channel == "New Channel"

    def test_update_video_invalid_field_raises(self, db: Session, sample_video: VideoDB):
        with pytest.raises(ValueError, match="not updatable"):
            crud.update_video(db, video_db_id=sample_video.id, nonexistent_field="x")

    def test_update_nonexistent_video_raises(self, db: Session):
        with pytest.raises(ValueError, match="not found"):
            crud.update_video(db, video_db_id=9999, title="Ghost")

    def test_mark_video_processed(self, db: Session, sample_video: VideoDB):
        assert sample_video.is_processed is False
        video = crud.mark_video_processed(db, video_db_id=sample_video.id)
        assert video.is_processed is True
        assert video.summarize_count == 1
        assert video.process_error is None

    def test_mark_video_processed_increments_count(self, db: Session, sample_video: VideoDB):
        crud.mark_video_processed(db, video_db_id=sample_video.id)
        crud.mark_video_processed(db, video_db_id=sample_video.id)
        video = crud.get_video_by_id(db, video_db_id=sample_video.id)
        assert video.summarize_count == 2

    def test_mark_video_failed(self, db: Session, sample_video: VideoDB):
        video = crud.mark_video_failed(
            db, video_db_id=sample_video.id, error_message="Transcript not available"
        )
        assert video.is_processed is False
        assert "Transcript" in video.process_error

    def test_delete_video(self, db: Session, sample_video: VideoDB):
        deleted = crud.delete_video(db, video_db_id=sample_video.id)
        assert deleted is True
        assert crud.get_video_by_id(db, video_db_id=sample_video.id) is None

    def test_delete_nonexistent_video_returns_false(self, db: Session):
        assert crud.delete_video(db, video_db_id=9999) is False


# ─────────────────────────────────────────────────────────────────────────────
# Summary CRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryCRUD:

    def test_create_summary(self, db: Session, sample_video: VideoDB):
        summary = crud.create_summary(
            db,
            video_db_id=sample_video.id,
            model_name="claude-opus-4-20250514",
            executive_summary="Short overview.",
            detailed_summary="Longer breakdown.",
            key_points=["Point 1", "Point 2"],
            action_items=["Do thing A"],
            important_timestamps=[{"time": "1:00", "description": "Key moment"}],
        )
        assert summary.id is not None
        assert summary.video_db_id == sample_video.id
        assert summary.executive_summary == "Short overview."

    def test_create_duplicate_summary_raises(self, db: Session, sample_summary: SummaryDB):
        with pytest.raises(ValueError, match="already exists"):
            crud.create_summary(
                db,
                video_db_id=sample_summary.video_db_id,
                model_name="model",
                executive_summary="x",
                detailed_summary="y",
                key_points=[],
                action_items=[],
                important_timestamps=[],
            )

    def test_upsert_creates_when_absent(self, db: Session, sample_video: VideoDB):
        summary = crud.upsert_summary(
            db,
            video_db_id=sample_video.id,
            model_name="test-model",
            executive_summary="First",
            detailed_summary="First detailed",
            key_points=["A"],
            action_items=["B"],
            important_timestamps=[],
        )
        assert summary.id is not None

    def test_upsert_updates_when_present(self, db: Session, sample_summary: SummaryDB):
        updated = crud.upsert_summary(
            db,
            video_db_id=sample_summary.video_db_id,
            model_name="new-model",
            executive_summary="Updated overview",
            detailed_summary="New detail",
            key_points=["New point"],
            action_items=[],
            important_timestamps=[],
        )
        assert updated.id == sample_summary.id  # same row
        assert updated.executive_summary == "Updated overview"
        assert updated.model_name == "new-model"

    def test_get_summary_by_video_db_id(self, db: Session, sample_summary: SummaryDB):
        fetched = crud.get_summary_by_video_db_id(
            db, video_db_id=sample_summary.video_db_id
        )
        assert fetched is not None
        assert fetched.id == sample_summary.id

    def test_get_summary_by_youtube_id(self, db: Session, sample_summary: SummaryDB):
        fetched = crud.get_summary_by_youtube_id(db, video_id="dQw4w9WgXcQ")
        assert fetched is not None
        assert fetched.id == sample_summary.id

    def test_key_points_json_roundtrip(self, db: Session, sample_summary: SummaryDB):
        points = crud.get_key_points(sample_summary)
        assert isinstance(points, list)
        assert "Devotion" in points

    def test_action_items_json_roundtrip(self, db: Session, sample_summary: SummaryDB):
        items = crud.get_action_items(sample_summary)
        assert isinstance(items, list)
        assert len(items) == 2

    def test_timestamps_json_roundtrip(self, db: Session, sample_summary: SummaryDB):
        timestamps = crud.get_important_timestamps(sample_summary)
        assert isinstance(timestamps, list)
        assert timestamps[0]["time"] == "0:10"

    def test_delete_summary(self, db: Session, sample_summary: SummaryDB):
        deleted = crud.delete_summary(db, summary_id=sample_summary.id)
        assert deleted is True
        assert crud.get_summary_by_id(db, summary_id=sample_summary.id) is None

    def test_delete_video_cascades_to_summary(
        self, db: Session, sample_video: VideoDB, sample_summary: SummaryDB
    ):
        """Deleting a video must delete its summary via CASCADE."""
        summary_id = sample_summary.id
        crud.delete_video(db, video_db_id=sample_video.id)
        assert crud.get_summary_by_id(db, summary_id=summary_id) is None


# ─────────────────────────────────────────────────────────────────────────────
# Chat Session CRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestChatSessionCRUD:

    def test_create_chat_session(self, db: Session, sample_video: VideoDB):
        session = crud.create_chat_session(
            db, video_db_id=sample_video.id, session_token="token-001"
        )
        assert session.id is not None
        assert session.session_token == "token-001"
        assert session.message_count == 0

    def test_create_duplicate_session_raises(
        self, db: Session, sample_session: ChatSessionDB
    ):
        with pytest.raises(ValueError, match="already exists"):
            crud.create_chat_session(
                db,
                video_db_id=sample_session.video_db_id,
                session_token=sample_session.session_token,
            )

    def test_get_or_create_returns_existing(
        self, db: Session, sample_session: ChatSessionDB
    ):
        session, created = crud.get_or_create_chat_session(
            db,
            video_db_id=sample_session.video_db_id,
            session_token=sample_session.session_token,
        )
        assert created is False
        assert session.id == sample_session.id

    def test_get_or_create_inserts_new(self, db: Session, sample_video: VideoDB):
        session, created = crud.get_or_create_chat_session(
            db, video_db_id=sample_video.id, session_token="brand-new-token"
        )
        assert created is True
        assert session.id is not None

    def test_get_chat_sessions_for_video(
        self, db: Session, sample_video: VideoDB, sample_session: ChatSessionDB
    ):
        crud.create_chat_session(
            db, video_db_id=sample_video.id, session_token="second-token"
        )
        sessions = crud.get_chat_sessions_for_video(db, video_db_id=sample_video.id)
        assert len(sessions) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Chat Message CRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestChatMessageCRUD:

    def test_add_user_message(self, db: Session, sample_session: ChatSessionDB):
        msg = crud.add_chat_message(
            db, session_id=sample_session.id, role="user", content="What is this video about?"
        )
        assert msg.role == "user"
        assert msg.position == 0
        assert msg.content == "What is this video about?"

    def test_add_assistant_message_increments_position(
        self, db: Session, sample_session: ChatSessionDB
    ):
        crud.add_chat_message(db, session_id=sample_session.id, role="user", content="Q1")
        msg2 = crud.add_chat_message(
            db, session_id=sample_session.id, role="assistant", content="A1"
        )
        assert msg2.position == 1

    def test_session_message_count_updated(
        self, db: Session, sample_session: ChatSessionDB
    ):
        crud.add_chat_message(db, session_id=sample_session.id, role="user", content="Hi")
        crud.add_chat_message(db, session_id=sample_session.id, role="assistant", content="Hello!")
        session = crud.get_chat_session_by_id(db, session_id=sample_session.id)
        assert session.message_count == 2

    def test_invalid_role_raises(self, db: Session, sample_session: ChatSessionDB):
        with pytest.raises(ValueError, match="role must be"):
            crud.add_chat_message(
                db, session_id=sample_session.id, role="system", content="bad"
            )

    def test_get_messages_for_session_ordered(
        self, db: Session, sample_session: ChatSessionDB
    ):
        for i, (role, content) in enumerate([
            ("user", "Q1"), ("assistant", "A1"), ("user", "Q2"), ("assistant", "A2")
        ]):
            crud.add_chat_message(db, session_id=sample_session.id, role=role, content=content)

        messages = crud.get_messages_for_session(db, session_id=sample_session.id)
        assert [m.position for m in messages] == [0, 1, 2, 3]

    def test_get_chat_history_dicts(self, db: Session, sample_session: ChatSessionDB):
        crud.add_chat_message(db, session_id=sample_session.id, role="user", content="Hello")
        crud.add_chat_message(db, session_id=sample_session.id, role="assistant", content="Hi there!")
        history = crud.get_chat_history_dicts(db, session_id=sample_session.id)
        assert history == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_get_last_n_messages(self, db: Session, sample_session: ChatSessionDB):
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            crud.add_chat_message(db, session_id=sample_session.id, role=role, content=f"msg{i}")

        last3 = crud.get_last_n_messages(db, session_id=sample_session.id, n=3)
        assert len(last3) == 3
        assert last3[0].position == 3   # oldest of the last 3
        assert last3[-1].position == 5  # most recent

    def test_cascade_delete_session_removes_messages(
        self, db: Session, sample_session: ChatSessionDB
    ):
        crud.add_chat_message(db, session_id=sample_session.id, role="user", content="Hi")
        session_id = sample_session.id
        crud.delete_chat_session(db, session_id=session_id)
        messages = crud.get_messages_for_session(db, session_id=session_id)
        assert messages == []


# ─────────────────────────────────────────────────────────────────────────────
# Composite helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeHelpers:

    def test_get_video_with_summary(
        self, db: Session, sample_video: VideoDB, sample_summary: SummaryDB
    ):
        result = crud.get_video_with_summary(db, video_id="dQw4w9WgXcQ")
        assert result is not None
        video, summary = result
        assert video.video_id == "dQw4w9WgXcQ"
        assert summary.executive_summary == "A classic 80s pop song about devotion."

    def test_get_video_with_summary_returns_none_when_missing(self, db: Session):
        result = crud.get_video_with_summary(db, video_id="notexist123")
        assert result is None

    def test_get_recent_videos_with_summaries(
        self, db: Session, sample_video: VideoDB, sample_summary: SummaryDB
    ):
        crud.mark_video_processed(db, video_db_id=sample_video.id)
        results = crud.get_recent_videos_with_summaries(db, limit=5)
        assert len(results) == 1
        video, summary = results[0]
        assert video.id == sample_video.id

    def test_get_recent_videos_excludes_unprocessed(self, db: Session):
        crud.create_video(
            db,
            video_id="unprocessed1",
            youtube_url="https://youtube.com/watch?v=unprocessed1",
        )
        results = crud.get_recent_videos_with_summaries(db, limit=5)
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonHelpers:

    def test_dumps_loads_roundtrip(self):
        data = [{"time": "1:23", "description": "Something happens"}, "hello", 42]
        assert crud._loads(crud._dumps(data)) == data

    def test_loads_bad_json_returns_empty_list(self):
        assert crud._loads("not-json!!!") == []

    def test_loads_none_returns_empty_list(self):
        assert crud._loads(None) == []  # type: ignore[arg-type]
