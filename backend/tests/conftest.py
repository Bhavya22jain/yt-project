"""
tests/conftest.py
──────────────────
Shared pytest fixtures available to all tests.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.video import TranscriptSegment, VideoTranscript, VideoMetadata


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI test client — session-scoped for speed."""
    return TestClient(app)


@pytest.fixture
def sample_transcript() -> VideoTranscript:
    """A minimal VideoTranscript fixture for unit tests."""
    return VideoTranscript(
        video_id="dQw4w9WgXcQ",
        language="en",
        segments=[
            TranscriptSegment(text="Hello and welcome.", start=0.0, duration=2.5),
            TranscriptSegment(text="Today we discuss AI.", start=2.5, duration=3.0),
            TranscriptSegment(text="Thank you for watching.", start=55.0, duration=2.0),
        ],
    )


@pytest.fixture
def sample_metadata() -> VideoMetadata:
    """A minimal VideoMetadata fixture."""
    return VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Sample Video",
        channel="Test Channel",
        duration_seconds=60,
    )
