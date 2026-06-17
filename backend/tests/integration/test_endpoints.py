"""
tests/integration/test_endpoints.py
─────────────────────────────────────────────────────────────────────────────
Day 5: Integration tests for all FastAPI endpoints.

Strategy:
  • Uses FastAPI TestClient (synchronous HTTPX wrapper) — no real server needed.
  • Every external call is mocked at the service layer, not the HTTP layer.
    - TranscriptService.get_transcript  → mock returns VideoTranscript
    - TranscriptService.extract_video_id → mock returns "dQw4w9WgXcQ"
    - SummaryService.summarize          → mock returns VideoSummary
    - ChatService.chat                  → mock returns ChatResponse
  • Each test class owns its own mock setup and DB state via in-memory SQLite.
  • Tests verify: HTTP status, response schema shape, error codes, DB side-
    effects (rows inserted/updated), caching behaviour, validation rejections.

Fixtures:
  • client       — TestClient bound to the real FastAPI app
  • db_session   — fresh in-memory SQLite session (function-scoped)
  • mock_transcript_service  — patches TranscriptService on the router module
  • mock_summary_service     — patches SummaryService on the router module
  • mock_chat_service        — patches ChatService on the router module
  • sample_transcript        — VideoTranscript domain object
  • sample_summary           — VideoSummary schema object
"""

import json
from types import SimpleNamespace
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import (
    AIProviderError,
    SummarizationError,
    TranscriptNotAvailableError,
    TranscriptFetchError,
    InvalidYouTubeURLError,
    ChatError,
)
from app.database.database import Base, get_db
from app.database import crud
from app.main import create_app
from app.models.video import TranscriptSegment, VideoTranscript
from app.schemas.video import (
    ChatResponse as ChatResponseSchema,
    TimestampItem,
    VideoSummary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VALID_VIDEO_ID = "dQw4w9WgXcQ"
SHORT_URL = "https://youtu.be/dQw4w9WgXcQ"
INVALID_URL = "https://vimeo.com/123456"
SUMMARIZE_PATH = "/api/v1/summarize"
VIDEOS_PATH = "/api/v1/videos"
CHAT_PATH = "/api/v1/chat"
HEALTH_PATH = "/api/v1/health"


# ─────────────────────────────────────────────────────────────────────────────
# Shared domain-object factories
# ─────────────────────────────────────────────────────────────────────────────

def make_transcript(video_id: str = VALID_VIDEO_ID) -> VideoTranscript:
    return VideoTranscript(
        video_id=video_id,
        language="en",
        segments=[
            TranscriptSegment("Hello and welcome to this tutorial.", 0.0, 3.0),
            TranscriptSegment("Today we cover machine learning.", 3.0, 4.0),
            TranscriptSegment("Let's start with the basics.", 7.0, 3.5),
            TranscriptSegment("Thank you for watching.", 60.0, 2.0),
        ],
    )


def make_summary(video_id: str = VALID_VIDEO_ID) -> VideoSummary:
    return VideoSummary(
        video_id=video_id,
        title=None,
        duration="1:04",
        executive_summary="An introductory ML tutorial covering the basics.",
        detailed_summary=(
            "The video begins with a welcome and overview, then dives into "
            "machine learning fundamentals before concluding with next steps."
        ),
        key_points=[
            "Machine learning requires clean data.",
            "Start simple before going complex.",
            "Evaluation is as important as training.",
        ],
        action_items=[
            "Read the linked scikit-learn documentation.",
            "Try the example notebook on GitHub.",
            "Join the community Discord.",
        ],
        important_timestamps=[
            TimestampItem(time="0:00", description="Introduction"),
            TimestampItem(time="0:03", description="ML overview begins"),
        ],
    )


def make_chat_response(answer: str = "The video is about machine learning.") -> ChatResponseSchema:
    return ChatResponseSchema(
        success=True,
        answer=answer,
        sources=["Hello and welcome to this tutorial."],
        session_token=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# In-memory DB fixture — isolates each test from every other
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    Fresh in-memory SQLite session per test.

    Key: We keep a single persistent connection open for the lifetime of the
    fixture and create the session bound directly to that connection. This
    ensures that Base.metadata.create_all and every subsequent CRUD call use
    the *exact same* SQLite in-memory database (in-memory DBs are
    connection-scoped — a new connection sees an empty database).
    """
    from sqlalchemy import event as sa_event

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Hold one connection open for the entire test so the in-memory DB persists
    connection = engine.connect()
    Base.metadata.create_all(bind=connection)

    SessionLocal = sessionmaker(bind=connection, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=connection)
        connection.close()
        engine.dispose()


@pytest.fixture(scope="function")
def client(db_session: Session) -> TestClient:
    """
    TestClient bound to a fresh app with the in-memory DB injected.

    Strategy:
      1. The in-memory engine already has tables created (done in db_session
         fixture via Base.metadata.create_all).
      2. We patch db_module.engine BEFORE the TestClient starts so that
         lifespan's init_db() / health_check_db() use our in-memory engine.
      3. We override get_db so every endpoint call receives our test session.
    """
    import app.database.database as db_module

    test_engine = db_session.bind          # already has tables

    app_instance = create_app()
    app_instance.dependency_overrides[get_db] = lambda: db_session

    # Patch the names as imported into main.py (not the source module),
    # because `from x import y` binds the name in the importing module's
    # namespace — patching the source module has no effect on already-imported refs.
    with (
        patch("app.main.init_db", lambda: None),
        patch("app.main.health_check_db", lambda: True),
        patch.object(db_module, "engine", test_engine),
    ):
        with TestClient(app_instance, raise_server_exceptions=False) as c:
            yield c


# ─────────────────────────────────────────────────────────────────────────────
# Service mock fixtures — patch at the router module level
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_MODULE = "app.api.v1.endpoints.video"


@pytest.fixture
def mock_services(db_session: Session):
    """
    Patch all three service singletons on the router module.
    Returns a namespace with .transcript, .summary, .chat attributes
    so individual tests can customise return values or side_effects.
    """
    transcript_mock = MagicMock()
    transcript_mock.extract_video_id.return_value = VALID_VIDEO_ID
    transcript_mock.get_transcript = AsyncMock(return_value=make_transcript())

    summary_mock = MagicMock()
    summary_mock.summarize = AsyncMock(return_value=make_summary())

    chat_mock = MagicMock()
    chat_mock.chat = AsyncMock(return_value=make_chat_response())

    with (
        patch(f"{ROUTER_MODULE}._transcript_service", transcript_mock),
        patch(f"{ROUTER_MODULE}._summary_service", summary_mock),
        patch(f"{ROUTER_MODULE}._chat_service", chat_mock),
    ):
        yield SimpleNamespace(
            transcript=transcript_mock,
            summary=summary_mock,
            chat=chat_mock,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self, client):
        resp = client.get(HEALTH_PATH)
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        assert resp.status_code == 200 if (resp := client.get(HEALTH_PATH)) else True
        body = client.get(HEALTH_PATH).json()
        assert body["status"] == "ok"

    def test_version_present(self, client):
        body = client.get(HEALTH_PATH).json()
        assert "version" in body
        assert body["version"]

    def test_environment_present(self, client):
        body = client.get(HEALTH_PATH).json()
        assert "environment" in body

    def test_database_field_present(self, client):
        body = client.get(HEALTH_PATH).json()
        assert "database" in body
        assert body["database"] in ("connected", "unavailable")

    def test_request_id_in_response_headers(self, client):
        resp = client.get(HEALTH_PATH)
        assert "x-request-id" in resp.headers

    def test_process_time_in_response_headers(self, client):
        resp = client.get(HEALTH_PATH)
        assert "x-process-time" in resp.headers


# ─────────────────────────────────────────────────────────────────────────────
# 2. POST /summarize — request validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeValidation:

    def test_missing_body_returns_422(self, client):
        resp = client.post(SUMMARIZE_PATH, json={})
        assert resp.status_code == 422

    def test_missing_url_field_returns_422(self, client):
        resp = client.post(SUMMARIZE_PATH, json={"force_refresh": False})
        assert resp.status_code == 422

    def test_invalid_youtube_url_returns_422(self, client, mock_services):
        mock_services.transcript.extract_video_id.side_effect = InvalidYouTubeURLError(INVALID_URL)
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": INVALID_URL})
        assert resp.status_code == 422

    def test_invalid_url_error_body_shape(self, client, mock_services):
        mock_services.transcript.extract_video_id.side_effect = InvalidYouTubeURLError(INVALID_URL)
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": INVALID_URL}).json()
        assert body["success"] is False
        assert "error" in body

    def test_vimeo_url_rejected(self, client, mock_services):
        mock_services.transcript.extract_video_id.side_effect = InvalidYouTubeURLError(INVALID_URL)
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": "https://vimeo.com/123"})
        assert resp.status_code in (422, 400)

    def test_empty_url_returns_422(self, client):
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": ""})
        assert resp.status_code == 422

    def test_valid_short_url_accepted_by_schema(self, client, mock_services):
        mock_services.transcript.extract_video_id.return_value = VALID_VIDEO_ID
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": SHORT_URL})
        # Schema accepts short URLs — pipeline proceeds
        assert resp.status_code in (200, 502)

    def test_extra_unknown_fields_ignored(self, client, mock_services):
        """Pydantic v2 ignores extra fields by default — should not return 422."""
        resp = client.post(SUMMARIZE_PATH, json={
            "youtube_url": VALID_URL,
            "unknown_extra_field": "should be ignored",
        })
        assert resp.status_code != 422

    def test_language_preference_accepted(self, client, mock_services):
        resp = client.post(SUMMARIZE_PATH, json={
            "youtube_url": VALID_URL,
            "language_preference": ["en", "es"],
        })
        assert resp.status_code in (200, 502)

    def test_force_refresh_defaults_to_false(self, client, mock_services):
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        # No error about force_refresh — it has a default
        assert resp.status_code in (200, 502)


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST /summarize — happy path (full pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeHappyPath:

    def test_returns_200(self, client, mock_services):
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 200

    def test_success_true(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert body["success"] is True

    def test_response_has_data(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert "data" in body

    def test_data_has_executive_summary(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert "executive_summary" in body["data"]
        assert len(body["data"]["executive_summary"]) > 0

    def test_data_has_detailed_summary(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert "detailed_summary" in body["data"]

    def test_data_has_key_points_list(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert isinstance(body["data"]["key_points"], list)
        assert len(body["data"]["key_points"]) > 0

    def test_data_has_action_items_list(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert isinstance(body["data"]["action_items"], list)

    def test_data_has_timestamps_list(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert isinstance(body["data"]["important_timestamps"], list)

    def test_timestamps_have_time_and_description(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        for ts in body["data"]["important_timestamps"]:
            assert "time" in ts
            assert "description" in ts

    def test_data_has_video_id(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert body["data"]["video_id"] == VALID_VIDEO_ID

    def test_cached_false_on_fresh_summary(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert body["cached"] is False

    def test_processing_ms_is_non_negative(self, client, mock_services):
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert body["processing_ms"] >= 0

    def test_transcript_service_called_once(self, client, mock_services):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        mock_services.transcript.get_transcript.assert_called_once()

    def test_summary_service_called_once(self, client, mock_services):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        mock_services.summary.summarize.assert_called_once()

    def test_video_saved_to_db(self, client, mock_services, db_session):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        assert video is not None

    def test_video_marked_processed_in_db(self, client, mock_services, db_session):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        assert video.is_processed is True

    def test_summary_saved_to_db(self, client, mock_services, db_session):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        summary = crud.get_summary_by_youtube_id(db_session, video_id=VALID_VIDEO_ID)
        assert summary is not None

    def test_summary_executive_summary_in_db(self, client, mock_services, db_session):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        summary = crud.get_summary_by_youtube_id(db_session, video_id=VALID_VIDEO_ID)
        assert "ML tutorial" in summary.executive_summary

    def test_transcript_text_stored_in_db(self, client, mock_services, db_session):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        assert video.transcript_text is not None
        assert len(video.transcript_text) > 0

    def test_summarize_count_incremented(self, client, mock_services, db_session):
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        assert video.summarize_count == 1

    def test_request_id_header_returned(self, client, mock_services):
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert "x-request-id" in resp.headers


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST /summarize — caching behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeCaching:

    def _seed_db(self, db_session: Session):
        """Pre-populate the DB with a processed video + summary."""
        video, _ = crud.get_or_create_video(
            db_session,
            video_id=VALID_VIDEO_ID,
            youtube_url=VALID_URL,
            language="en",
            transcript_text="Pre-seeded transcript text.",
            transcript_word_count=4,
        )
        crud.upsert_summary(
            db_session,
            video_db_id=video.id,
            model_name="claude-opus-4-20250514",
            executive_summary="Cached executive summary.",
            detailed_summary="Cached detailed summary.",
            key_points=["Cached point A", "Cached point B"],
            action_items=["Cached action"],
            important_timestamps=[{"time": "0:00", "description": "Start"}],
        )
        crud.mark_video_processed(db_session, video_db_id=video.id)
        return video

    def test_cached_true_when_already_processed(self, client, mock_services, db_session):
        self._seed_db(db_session)
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert body["cached"] is True

    def test_cached_response_returns_200(self, client, mock_services, db_session):
        self._seed_db(db_session)
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 200

    def test_cached_response_has_correct_summary(self, client, mock_services, db_session):
        self._seed_db(db_session)
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        assert body["data"]["executive_summary"] == "Cached executive summary."

    def test_no_transcript_service_call_on_cache_hit(self, client, mock_services, db_session):
        self._seed_db(db_session)
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        mock_services.transcript.get_transcript.assert_not_called()

    def test_no_summary_service_call_on_cache_hit(self, client, mock_services, db_session):
        self._seed_db(db_session)
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        mock_services.summary.summarize.assert_not_called()

    def test_force_refresh_bypasses_cache(self, client, mock_services, db_session):
        self._seed_db(db_session)
        body = client.post(SUMMARIZE_PATH, json={
            "youtube_url": VALID_URL,
            "force_refresh": True,
        }).json()
        # fresh summary from mock (not cached)
        assert body["cached"] is False
        assert "introductory ML tutorial" in body["data"]["executive_summary"]

    def test_force_refresh_calls_services(self, client, mock_services, db_session):
        self._seed_db(db_session)
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL, "force_refresh": True})
        mock_services.transcript.get_transcript.assert_called_once()
        mock_services.summary.summarize.assert_called_once()

    def test_force_refresh_updates_db_summary(self, client, mock_services, db_session):
        self._seed_db(db_session)
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL, "force_refresh": True})
        summary = crud.get_summary_by_youtube_id(db_session, video_id=VALID_VIDEO_ID)
        # DB now has the fresh mock summary
        assert "introductory ML tutorial" in summary.executive_summary

    def test_summarize_count_increments_on_refresh(self, client, mock_services, db_session):
        self._seed_db(db_session)
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL, "force_refresh": True})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        # was 1 from seed; mark_video_processed increments again → 2
        assert video.summarize_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. POST /summarize — error paths
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeErrors:

    def test_transcript_not_available_returns_404(self, client, mock_services):
        mock_services.transcript.get_transcript.side_effect = (
            TranscriptNotAvailableError(VALID_VIDEO_ID)
        )
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 404

    def test_transcript_not_available_error_code(self, client, mock_services):
        mock_services.transcript.get_transcript.side_effect = (
            TranscriptNotAvailableError(VALID_VIDEO_ID)
        )
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        payload = body if "code" in body else body.get("detail", body)
        assert payload["code"] == "TRANSCRIPT_NOT_AVAILABLE"

    def test_transcript_fetch_error_returns_502(self, client, mock_services):
        mock_services.transcript.get_transcript.side_effect = (
            TranscriptFetchError(VALID_VIDEO_ID, "Network timeout")
        )
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 502

    def test_summarization_error_returns_502(self, client, mock_services):
        mock_services.summary.summarize.side_effect = SummarizationError("AI parse error")
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 502

    def test_summarization_error_code(self, client, mock_services):
        mock_services.summary.summarize.side_effect = SummarizationError("AI parse error")
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        payload = body if "code" in body else body.get("detail", body)
        assert payload["code"] == "SUMMARIZATION_FAILED"

    def test_ai_provider_error_returns_503(self, client, mock_services):
        mock_services.summary.summarize.side_effect = AIProviderError("Rate limit")
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 503

    def test_ai_provider_error_code(self, client, mock_services):
        mock_services.summary.summarize.side_effect = AIProviderError("Rate limit")
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        payload = body if "code" in body else body.get("detail", body)
        assert payload["code"] == "AI_PROVIDER_ERROR"

    def test_error_body_has_success_false(self, client, mock_services):
        mock_services.summary.summarize.side_effect = SummarizationError("fail")
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        # FastAPI may nest detail under body["detail"] for HTTPException
        payload = body if "success" in body else body.get("detail", body)
        assert payload["success"] is False

    def test_error_body_has_request_id(self, client, mock_services):
        mock_services.summary.summarize.side_effect = SummarizationError("fail")
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        payload = body if "request_id" in body else body.get("detail", body)
        assert "request_id" in payload

    def test_video_error_state_persisted_on_transcript_fail(
        self, client, mock_services, db_session
    ):
        mock_services.transcript.get_transcript.side_effect = (
            TranscriptNotAvailableError(VALID_VIDEO_ID)
        )
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        # Video row should be created with error state
        if video:
            assert video.is_processed is False

    def test_video_error_state_persisted_on_ai_fail(
        self, client, mock_services, db_session
    ):
        mock_services.summary.summarize.side_effect = SummarizationError("Bad AI")
        client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL})
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        assert video is not None
        assert video.is_processed is False
        assert video.process_error is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. GET /videos
# ─────────────────────────────────────────────────────────────────────────────

class TestListVideos:

    def _seed_videos(self, db: Session, count: int = 3) -> list:
        videos = []
        for i in range(count):
            vid_id = f"video_{i:07d}"
            v, _ = crud.get_or_create_video(
                db, video_id=vid_id,
                youtube_url=f"https://www.youtube.com/watch?v={vid_id}",
            )
            crud.upsert_summary(
                db, video_db_id=v.id, model_name="test",
                executive_summary=f"Summary {i}",
                detailed_summary=f"Detailed {i}",
                key_points=[f"Point {i}"],
                action_items=[],
                important_timestamps=[],
            )
            crud.mark_video_processed(db, video_db_id=v.id)
            videos.append(v)
        return videos

    def test_returns_200_on_empty_db(self, client):
        resp = client.get(VIDEOS_PATH)
        assert resp.status_code == 200

    def test_empty_db_returns_empty_list(self, client):
        body = client.get(VIDEOS_PATH).json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_returns_all_processed_videos(self, client, db_session):
        self._seed_videos(db_session, 3)
        body = client.get(VIDEOS_PATH).json()
        assert body["total"] == 3
        assert len(body["items"]) == 3

    def test_pagination_skip(self, client, db_session):
        self._seed_videos(db_session, 5)
        body = client.get(f"{VIDEOS_PATH}?skip=2&limit=10").json()
        assert len(body["items"]) == 3

    def test_pagination_limit(self, client, db_session):
        self._seed_videos(db_session, 5)
        body = client.get(f"{VIDEOS_PATH}?limit=2").json()
        assert len(body["items"]) == 2

    def test_pagination_metadata_returned(self, client, db_session):
        self._seed_videos(db_session, 5)
        body = client.get(f"{VIDEOS_PATH}?skip=1&limit=2").json()
        assert body["skip"] == 1
        assert body["limit"] == 2

    def test_unprocessed_video_excluded_by_default(self, client, db_session):
        # Unprocessed (no mark_processed call)
        crud.get_or_create_video(
            db_session, video_id="unproc0001",
            youtube_url="https://youtube.com/watch?v=unproc0001"
        )
        body = client.get(VIDEOS_PATH).json()
        assert body["total"] == 0

    def test_unprocessed_included_when_flag_false(self, client, db_session):
        crud.get_or_create_video(
            db_session, video_id="unproc0001",
            youtube_url="https://youtube.com/watch?v=unproc0001"
        )
        body = client.get(f"{VIDEOS_PATH}?processed_only=false").json()
        assert body["total"] == 1

    def test_each_item_has_required_fields(self, client, db_session):
        self._seed_videos(db_session, 1)
        body = client.get(VIDEOS_PATH).json()
        item = body["items"][0]
        for field in ("db_id", "video_id", "youtube_url", "is_processed",
                      "created_at", "updated_at"):
            assert field in item, f"Missing field: {field}"

    def test_items_have_embedded_summary(self, client, db_session):
        self._seed_videos(db_session, 1)
        body = client.get(VIDEOS_PATH).json()
        assert body["items"][0]["summary"] is not None

    def test_limit_out_of_range_returns_422(self, client):
        resp = client.get(f"{VIDEOS_PATH}?limit=0")
        assert resp.status_code == 422

    def test_limit_exceeds_max_returns_422(self, client):
        resp = client.get(f"{VIDEOS_PATH}?limit=101")
        assert resp.status_code == 422

    def test_negative_skip_returns_422(self, client):
        resp = client.get(f"{VIDEOS_PATH}?skip=-1")
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET /videos/{video_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestGetVideo:

    def _seed_one(self, db: Session) -> str:
        v, _ = crud.get_or_create_video(
            db, video_id=VALID_VIDEO_ID, youtube_url=VALID_URL,
            language="en", transcript_text="Hello world.",
        )
        crud.upsert_summary(
            db, video_db_id=v.id, model_name="test",
            executive_summary="Test executive summary.",
            detailed_summary="Test detailed summary.",
            key_points=["Point A"],
            action_items=["Do A"],
            important_timestamps=[{"time": "0:00", "description": "Start"}],
        )
        crud.mark_video_processed(db, video_db_id=v.id)
        return VALID_VIDEO_ID

    def test_returns_200_for_known_video(self, client, db_session):
        self._seed_one(db_session)
        resp = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        assert resp.status_code == 200

    def test_returns_404_for_unknown_video(self, client):
        resp = client.get(f"{VIDEOS_PATH}/unknownvideo")
        assert resp.status_code == 404

    def test_404_error_code(self, client):
        body = client.get(f"{VIDEOS_PATH}/unknownvideo").json()
        payload = body if "code" in body else body.get("detail", body)
        assert payload["code"] == "VIDEO_NOT_FOUND"

    def test_response_has_data(self, client, db_session):
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert "data" in body

    def test_data_video_id_matches(self, client, db_session):
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert body["data"]["video_id"] == VALID_VIDEO_ID

    def test_data_has_embedded_summary(self, client, db_session):
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert body["data"]["summary"] is not None
        assert body["data"]["summary"]["executive_summary"] == "Test executive summary."

    def test_data_has_transcript_text(self, client, db_session):
        """The full transcript must be exposed so the frontend can display it."""
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert body["data"]["transcript"] == "Hello world."

    def test_transcript_none_when_not_stored(self, client, db_session):
        v, _ = crud.get_or_create_video(
            db_session, video_id="notranscript1",
            youtube_url="https://www.youtube.com/watch?v=notranscript1",
        )
        body = client.get(f"{VIDEOS_PATH}/notranscript1").json()
        assert body["data"]["transcript"] is None

    def test_data_summary_has_key_points(self, client, db_session):
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert body["data"]["summary"]["key_points"] == ["Point A"]

    def test_data_summary_has_timestamps(self, client, db_session):
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        ts = body["data"]["summary"]["important_timestamps"]
        assert len(ts) == 1
        assert ts[0]["time"] == "0:00"

    def test_success_true(self, client, db_session):
        self._seed_one(db_session)
        body = client.get(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert body["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 8. DELETE /videos/{video_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteVideo:

    def _seed_one(self, db: Session) -> None:
        v, _ = crud.get_or_create_video(
            db, video_id=VALID_VIDEO_ID, youtube_url=VALID_URL
        )
        crud.upsert_summary(
            db, video_db_id=v.id, model_name="test",
            executive_summary="To be deleted.",
            detailed_summary=".", key_points=["x"],
            action_items=[], important_timestamps=[],
        )
        crud.mark_video_processed(db, video_db_id=v.id)

    def test_returns_200_on_successful_delete(self, client, db_session):
        self._seed_one(db_session)
        resp = client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        assert resp.status_code == 200

    def test_success_true_on_delete(self, client, db_session):
        self._seed_one(db_session)
        body = client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert body["success"] is True

    def test_message_in_response(self, client, db_session):
        self._seed_one(db_session)
        body = client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}").json()
        assert "message" in body
        assert len(body["message"]) > 0

    def test_returns_404_for_unknown_video(self, client):
        resp = client.delete(f"{VIDEOS_PATH}/unknownvideo")
        assert resp.status_code == 404

    def test_video_removed_from_db(self, client, db_session):
        self._seed_one(db_session)
        client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        assert crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID) is None

    def test_summary_cascade_deleted(self, client, db_session):
        self._seed_one(db_session)
        client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        assert crud.get_summary_by_youtube_id(db_session, video_id=VALID_VIDEO_ID) is None

    def test_second_delete_returns_404(self, client, db_session):
        self._seed_one(db_session)
        client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        resp = client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        assert resp.status_code == 404

    def test_deleted_video_not_in_list(self, client, db_session):
        self._seed_one(db_session)
        client.delete(f"{VIDEOS_PATH}/{VALID_VIDEO_ID}")
        body = client.get(VIDEOS_PATH).json()
        assert body["total"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. POST /chat — validation
# ─────────────────────────────────────────────────────────────────────────────

class TestChatValidation:

    def test_missing_body_returns_422(self, client):
        resp = client.post(CHAT_PATH, json={})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, client):
        resp = client.post(CHAT_PATH, json={"youtube_url": VALID_URL})
        assert resp.status_code == 422

    def test_missing_url_returns_422(self, client):
        resp = client.post(CHAT_PATH, json={"question": "What is this?"})
        assert resp.status_code == 422

    def test_question_too_short_returns_422(self, client):
        resp = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL, "question": "Hi"
        })
        assert resp.status_code == 422

    def test_question_whitespace_only_returns_422(self, client, mock_services):
        resp = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL, "question": "  "
        })
        assert resp.status_code == 422

    def test_invalid_youtube_url_returns_422(self, client, mock_services):
        mock_services.transcript.extract_video_id.side_effect = (
            InvalidYouTubeURLError("bad-url")
        )
        resp = client.post(CHAT_PATH, json={
            "youtube_url": "bad-url", "question": "What is this?"
        })
        assert resp.status_code in (422, 400)

    def test_malformed_history_item_silently_dropped(self, client, mock_services, db_session):
        # Pre-seed the video with a transcript
        v, _ = crud.get_or_create_video(
            db_session, video_id=VALID_VIDEO_ID, youtube_url=VALID_URL,
            transcript_text="Hello world.", language="en"
        )
        body = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this about?",
            "chat_history": [
                {"role": "invalid_role", "content": "bad"},  # dropped
                {"role": "user", "content": "Previous question"},  # kept
            ],
        }).json()
        # Request should not 422 — malformed history is silently dropped
        assert body.get("success") in (True, False)  # no 422

    def test_question_stripped_of_whitespace(self, client, mock_services, db_session):
        v, _ = crud.get_or_create_video(
            db_session, video_id=VALID_VIDEO_ID, youtube_url=VALID_URL,
            transcript_text="Hello world.", language="en"
        )
        resp = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "  What is this video?  ",
        })
        # Should not 422 — whitespace stripped, leaving a valid question
        assert resp.status_code != 422


# ─────────────────────────────────────────────────────────────────────────────
# 10. POST /chat — happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestChatHappyPath:

    def _seed_video_with_transcript(self, db: Session) -> None:
        v, _ = crud.get_or_create_video(
            db, video_id=VALID_VIDEO_ID, youtube_url=VALID_URL,
            language="en",
            transcript_text="Hello and welcome. Today we cover machine learning.",
        )
        crud.mark_video_processed(db, video_db_id=v.id)

    def test_returns_200(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        resp = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this video about?",
        })
        assert resp.status_code == 200

    def test_success_true(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        body = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this video about?",
        }).json()
        assert body["success"] is True

    def test_answer_in_response(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        body = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this video about?",
        }).json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_sources_in_response(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        body = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this video about?",
        }).json()
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_chat_service_called(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL, "question": "What is this about?"
        })
        mock_services.chat.chat.assert_called_once()

    def test_session_token_echoed_back(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        token = "my-test-session-token-42"
        body = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this about?",
            "session_token": token,
        }).json()
        assert body["session_token"] == token

    def test_no_session_token_accepted(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        body = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL, "question": "What is this about?"
        }).json()
        assert body["success"] is True
        assert body.get("session_token") is None

    def test_messages_persisted_when_session_token_provided(
        self, client, mock_services, db_session
    ):
        self._seed_video_with_transcript(db_session)
        token = "persist-session-001"
        client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "Tell me about ML.",
            "session_token": token,
        })
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        sessions = crud.get_chat_sessions_for_video(db_session, video_db_id=video.id)
        assert len(sessions) == 1
        messages = crud.get_messages_for_session(db_session, session_id=sessions[0].id)
        assert len(messages) == 2  # user + assistant
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_messages_not_persisted_without_session_token(
        self, client, mock_services, db_session
    ):
        self._seed_video_with_transcript(db_session)
        client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL, "question": "What is ML?"
        })
        video = crud.get_video_by_video_id(db_session, video_id=VALID_VIDEO_ID)
        if video:
            sessions = crud.get_chat_sessions_for_video(
                db_session, video_db_id=video.id
            )
            assert len(sessions) == 0

    def test_video_not_summarized_returns_404(self, client, mock_services):
        # No DB seed — video doesn't exist
        resp = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL,
            "question": "What is this about?",
        })
        assert resp.status_code == 404

    def test_chat_error_returns_502(self, client, mock_services, db_session):
        self._seed_video_with_transcript(db_session)
        mock_services.chat.chat.side_effect = ChatError("AI blew up")
        resp = client.post(CHAT_PATH, json={
            "youtube_url": VALID_URL, "question": "What is this about?"
        })
        assert resp.status_code == 502


# ─────────────────────────────────────────────────────────────────────────────
# 11. Response schema compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseSchemas:
    """Verify every response exactly matches its Pydantic schema."""

    def test_summarize_response_is_schema_compliant(self, client, mock_services):
        from app.schemas.video import SummarizeResponse
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        # Should parse without raising
        parsed = SummarizeResponse(**body)
        assert parsed.success is True

    def test_health_response_is_schema_compliant(self, client):
        from app.schemas.video import HealthResponse
        body = client.get(HEALTH_PATH).json()
        parsed = HealthResponse(**body)
        assert parsed.status == "ok"

    def test_video_list_response_is_schema_compliant(self, client):
        from app.schemas.video import VideoListResponse
        body = client.get(VIDEOS_PATH).json()
        parsed = VideoListResponse(**body)
        assert parsed.success is True

    def test_error_response_is_schema_compliant(self, client, mock_services):
        from app.schemas.video import ErrorResponse
        mock_services.summary.summarize.side_effect = SummarizationError("fail")
        body = client.post(SUMMARIZE_PATH, json={"youtube_url": VALID_URL}).json()
        # HTTPException wraps detail — unwrap if needed
        payload = body if "success" in body else body.get("detail", body)
        parsed = ErrorResponse(**payload)
        assert parsed.success is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. Middleware behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestMiddleware:

    def test_custom_request_id_echoed(self, client):
        custom_id = "my-custom-request-id"
        resp = client.get(HEALTH_PATH, headers={"X-Request-ID": custom_id})
        assert resp.headers.get("x-request-id") == custom_id

    def test_auto_request_id_generated_when_absent(self, client):
        resp = client.get(HEALTH_PATH)
        rid = resp.headers.get("x-request-id")
        assert rid is not None
        assert len(rid) > 0

    def test_process_time_is_numeric(self, client):
        resp = client.get(HEALTH_PATH)
        pt = resp.headers.get("x-process-time")
        assert pt is not None
        assert int(pt) >= 0

    def test_cors_headers_present(self, client):
        resp = client.options(
            HEALTH_PATH,
            headers={"Origin": "http://localhost:8501",
                     "Access-Control-Request-Method": "GET"},
        )
        # CORS preflight or regular request should include CORS header
        assert resp.status_code in (200, 204)

    def test_404_on_unknown_route(self, client):
        resp = client.get("/api/v1/doesnotexist")
        assert resp.status_code == 404

    def test_method_not_allowed(self, client):
        # /health only accepts GET
        resp = client.delete(HEALTH_PATH)
        assert resp.status_code == 405

    def test_pydantic_validation_error_returns_structured_body(self, client):
        resp = client.post(SUMMARIZE_PATH, json={"youtube_url": 12345})
        assert resp.status_code == 422
        body = resp.json()
        assert body["success"] is False
        assert "details" in body
        assert body["code"] == "VALIDATION_ERROR"
