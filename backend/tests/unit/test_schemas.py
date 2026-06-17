"""
tests/unit/test_schemas.py
───────────────────────────
Day 1: Unit tests for Pydantic schemas — URL validation, field constraints.
"""

import pytest
from pydantic import ValidationError

from app.schemas.video import SummarizeRequest, ChatRequest


class TestSummarizeRequest:

    def test_valid_standard_url(self):
        req = SummarizeRequest(youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert "dQw4w9WgXcQ" in req.youtube_url

    def test_valid_short_url(self):
        req = SummarizeRequest(youtube_url="https://youtu.be/dQw4w9WgXcQ")
        assert req.youtube_url

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            SummarizeRequest(youtube_url="https://vimeo.com/123456")

    def test_non_url_raises(self):
        with pytest.raises(ValidationError):
            SummarizeRequest(youtube_url="not a url at all")


class TestChatRequest:

    def test_valid_chat_request(self):
        req = ChatRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            question="What is this video about?",
        )
        assert req.question == "What is this video about?"
        assert req.chat_history == []

    def test_question_too_short_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                question="Hi",
            )

    def test_question_is_stripped(self):
        req = ChatRequest(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            question="  What is this video about?  ",
        )
        assert req.question == "What is this video about?"
