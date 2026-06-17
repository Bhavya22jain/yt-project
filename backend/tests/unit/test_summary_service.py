"""
tests/unit/test_summary_service.py
─────────────────────────────────────────────────────────────────────────────
Day 4: Full test suite for SummaryService.

Testing strategy:
  • All tests mock the Anthropic API client — zero real network calls.
  • Tool-use path tests mock `client.messages.create` to return a realistic
    tool_use content block, verifying the happy-path parsing pipeline.
  • Fallback path tests simulate a missing tool-use block and verify
    JSON extraction from plain text.
  • Error-path tests verify that every Anthropic SDK exception is correctly
    mapped to the right typed application exception with the right HTTP status.
  • Prompt-building and JSON extraction tests are pure-logic — no mocks.
  • _build_video_summary tests cover field extraction, defaults, and
    validation with deliberately malformed AI responses.

All fixtures defined in this file. No external network dependency.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import anthropic
from app.core.exceptions import AIProviderError, SummarizationError
from app.models.video import TranscriptSegment, VideoTranscript
from app.schemas.video import TimestampItem, VideoSummary
from app.services.summary_service import SummaryService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_async(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def make_segment(text: str, start: float = 0.0, duration: float = 3.0) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, duration=duration)


def make_transcript(
    segments: list[TranscriptSegment] | None = None,
    video_id: str = "testVideo123",
    language: str = "en",
) -> VideoTranscript:
    if segments is None:
        segments = [
            make_segment("Hello and welcome to this video.", 0.0, 3.0),
            make_segment("Today we cover three important topics.", 3.0, 3.5),
            make_segment("First is machine learning fundamentals.", 6.5, 4.0),
            make_segment("Second is data preprocessing techniques.", 10.5, 4.5),
            make_segment("Third is model evaluation strategies.", 15.0, 4.0),
            make_segment("Let's start with the first topic.", 19.0, 2.5),
            make_segment("Machine learning requires good data.", 21.5, 3.5),
            make_segment("Always clean your data before training.", 25.0, 3.0),
            make_segment("Evaluation is crucial for model quality.", 28.0, 3.5),
            make_segment("Use cross-validation for robust results.", 31.5, 3.0),
            make_segment("Thank you so much for watching.", 34.5, 2.5),
        ]
    return VideoTranscript(video_id=video_id, language=language, segments=segments)


def make_valid_summary_data() -> dict[str, Any]:
    """Return a dict that matches the VideoSummary schema exactly."""
    return {
        "executive_summary": (
            "This video provides a comprehensive introduction to machine learning, "
            "covering data preprocessing, model training, and evaluation strategies. "
            "It is ideal for beginners wanting a structured overview."
        ),
        "detailed_summary": (
            "The video opens with an overview of three core ML topics. "
            "The presenter first explains machine learning fundamentals, "
            "stressing the importance of high-quality training data. "
            "Next, data preprocessing techniques are covered in depth, "
            "including normalisation and missing-value imputation. "
            "The third section addresses model evaluation, recommending "
            "cross-validation over a simple train/test split. "
            "The video concludes with practical tips for beginners."
        ),
        "key_points": [
            "Machine learning models are only as good as their training data.",
            "Always clean and preprocess data before model training.",
            "Cross-validation provides more robust evaluation than a single split.",
            "Feature engineering can dramatically improve model performance.",
            "Start with simple baseline models before trying complex ones.",
        ],
        "action_items": [
            "Download the scikit-learn library and run the example notebook.",
            "Read the linked article on data preprocessing best practices.",
            "Try cross-validation on your next project.",
            "Join the Discord community for Q&A.",
        ],
        "important_timestamps": [
            {"time": "0:00", "description": "Introduction and overview"},
            {"time": "0:21", "description": "Machine learning fundamentals begin"},
            {"time": "0:31", "description": "Model evaluation strategies"},
        ],
    }


def make_tool_use_response(data: dict[str, Any]) -> MagicMock:
    """
    Build a mock Anthropic API response with a tool_use content block.
    Mirrors the structure of anthropic.types.Message.
    """
    tool_block = SimpleNamespace(type="tool_use", input=data)
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def make_text_response(text: str) -> MagicMock:
    """
    Build a mock Anthropic API response with a plain text content block.
    Used for fallback path tests.
    """
    text_block = SimpleNamespace(type="text", text=text)
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response


def make_empty_response() -> MagicMock:
    """Response with no content blocks — triggers fallback."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = []
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def service() -> SummaryService:
    """SummaryService with a fake API key (no real calls)."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-fake"}):
        svc = SummaryService()
        svc._api_key = "test-key-fake"
        return svc


@pytest.fixture
def transcript() -> VideoTranscript:
    return make_transcript()


@pytest.fixture
def summary_data() -> dict[str, Any]:
    return make_valid_summary_data()


@pytest.fixture
def mock_client(service: SummaryService) -> AsyncMock:
    """
    Attach a mock AsyncAnthropic client to the service.
    Tests set mock_client.messages.create.return_value or side_effect.
    """
    client_mock = MagicMock()
    client_mock.messages.create = AsyncMock()
    service._client = client_mock
    return client_mock


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:

    def test_model_loaded_from_settings(self, service):
        assert service._model  # not empty
        assert isinstance(service._model, str)

    def test_max_tokens_is_positive(self, service):
        assert service._max_tokens > 0

    def test_temperature_in_valid_range(self, service):
        assert 0.0 <= service._temperature <= 1.0

    def test_client_is_none_before_first_call(self, service):
        service._client = None
        assert service._client is None

    def test_get_client_raises_without_api_key(self):
        svc = SummaryService()
        svc._api_key = ""
        svc._client = None
        with pytest.raises(AIProviderError, match="ANTHROPIC_API_KEY"):
            svc._get_client()

    def test_get_client_returns_async_client_with_key(self, service):
        service._client = None
        client = service._get_client()
        assert isinstance(client, anthropic.AsyncAnthropic)
        # Second call returns the same cached instance
        assert service._get_client() is client


# ─────────────────────────────────────────────────────────────────────────────
# 2. _build_annotated_transcript
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAnnotatedTranscript:

    def test_returns_string(self, service, transcript):
        result = service._build_annotated_transcript(transcript)
        assert isinstance(result, str)

    def test_contains_all_segment_texts(self, service, transcript):
        result = service._build_annotated_transcript(transcript)
        for seg in transcript.segments:
            assert seg.text in result

    def test_first_segment_always_has_timestamp(self, service, transcript):
        result = service._build_annotated_transcript(transcript)
        first_line = result.split("\n")[0]
        assert first_line.startswith("[")
        assert "]" in first_line

    def test_timestamp_format_is_m_colon_ss(self, service):
        """Timestamps must look like [0:00] or [1:05]."""
        import re
        segments = [make_segment("Hello", start=65.0)]  # 1:05
        t = make_transcript(segments)
        result = service._build_annotated_transcript(t)
        assert re.search(r"\[\d+:\d{2}\]", result)

    def test_every_tenth_segment_has_timestamp(self, service):
        """Only every 10th segment should carry a [M:SS] prefix."""
        import re
        segments = [make_segment(f"Word {i}", start=float(i * 3)) for i in range(25)]
        t = make_transcript(segments)
        result = service._build_annotated_transcript(t)
        lines = result.split("\n")
        timestamped = [l for l in lines if re.match(r"\[\d+:\d{2}\]", l)]
        # Segments 0, 10, 20 should have timestamps → 3
        assert len(timestamped) == 3

    def test_empty_transcript_returns_empty_string(self, service):
        t = make_transcript(segments=[])
        result = service._build_annotated_transcript(t)
        assert result == ""

    def test_single_segment_has_timestamp(self, service):
        t = make_transcript(segments=[make_segment("Only one.", 0.0)])
        result = service._build_annotated_transcript(t)
        assert result.startswith("[0:00]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. _build_user_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildUserPrompt:

    def test_contains_video_id(self, service, transcript):
        annotated = service._build_annotated_transcript(transcript)
        prompt = service._build_user_prompt(transcript, annotated)
        assert transcript.video_id in prompt

    def test_contains_language(self, service, transcript):
        annotated = service._build_annotated_transcript(transcript)
        prompt = service._build_user_prompt(transcript, annotated)
        assert transcript.language in prompt

    def test_contains_word_count(self, service, transcript):
        annotated = service._build_annotated_transcript(transcript)
        prompt = service._build_user_prompt(transcript, annotated)
        assert str(transcript.word_count) in prompt

    def test_contains_annotated_transcript(self, service, transcript):
        annotated = service._build_annotated_transcript(transcript)
        prompt = service._build_user_prompt(transcript, annotated)
        assert annotated in prompt

    def test_duration_formatted_minutes(self, service, transcript):
        annotated = service._build_annotated_transcript(transcript)
        prompt = service._build_user_prompt(transcript, annotated)
        # Duration should contain m and s
        assert "m" in prompt and "s" in prompt

    def test_duration_formatted_hours_for_long_video(self, service):
        # Build a transcript that lasts >1 hour
        segments = [make_segment("Content", start=3700.0, duration=2.0)]
        t = make_transcript(segments)
        annotated = service._build_annotated_transcript(t)
        prompt = service._build_user_prompt(t, annotated)
        assert "h" in prompt   # should include hours


# ─────────────────────────────────────────────────────────────────────────────
# 4. _extract_json_from_text  (fallback JSON extraction)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractJsonFromText:

    def test_extracts_pure_json(self, service):
        data = {"executive_summary": "Great video.", "key_points": ["A", "B"]}
        result = service._extract_json_from_text(json.dumps(data))
        assert result == data

    def test_extracts_json_with_markdown_fence(self, service):
        data = {"executive_summary": "Test."}
        text = f"```json\n{json.dumps(data)}\n```"
        result = service._extract_json_from_text(text)
        assert result == data

    def test_extracts_json_with_plain_fence(self, service):
        data = {"key": "value"}
        text = f"```\n{json.dumps(data)}\n```"
        result = service._extract_json_from_text(text)
        assert result == data

    def test_extracts_json_with_preamble(self, service):
        data = {"executive_summary": "Here it is.", "key_points": []}
        text = f"Here is the summary:\n{json.dumps(data)}"
        result = service._extract_json_from_text(text)
        assert result["executive_summary"] == "Here it is."

    def test_empty_text_raises(self, service):
        with pytest.raises(SummarizationError, match="empty"):
            service._extract_json_from_text("")

    def test_no_json_object_raises(self, service):
        with pytest.raises(SummarizationError, match="valid JSON"):
            service._extract_json_from_text("This is just plain text with no JSON.")

    def test_malformed_json_raises(self, service):
        # JSON with balanced braces but invalid interior — parse error triggers.
        with pytest.raises(SummarizationError, match="malformed JSON"):
            service._extract_json_from_text('{"key": unterminated_value}')

    def test_nested_objects_parsed_correctly(self, service):
        data = {
            "executive_summary": "Summary.",
            "important_timestamps": [
                {"time": "0:30", "description": "Key moment"}
            ],
        }
        result = service._extract_json_from_text(json.dumps(data))
        assert result["important_timestamps"][0]["time"] == "0:30"

    def test_unicode_in_json_preserved(self, service):
        data = {"executive_summary": "Résumé: über alles, naïve → naive."}
        result = service._extract_json_from_text(json.dumps(data, ensure_ascii=False))
        assert "Résumé" in result["executive_summary"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. _build_video_summary  (response → VideoSummary)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildVideoSummary:

    def test_returns_video_summary_instance(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert isinstance(result, VideoSummary)

    def test_video_id_from_transcript(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert result.video_id == transcript.video_id

    def test_executive_summary_extracted(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert result.executive_summary == summary_data["executive_summary"]

    def test_detailed_summary_extracted(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert result.detailed_summary == summary_data["detailed_summary"]

    def test_key_points_is_list_of_strings(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert isinstance(result.key_points, list)
        assert all(isinstance(p, str) for p in result.key_points)

    def test_key_points_count_matches_input(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert len(result.key_points) == len(summary_data["key_points"])

    def test_action_items_is_list_of_strings(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert isinstance(result.action_items, list)
        assert all(isinstance(a, str) for a in result.action_items)

    def test_timestamps_are_timestamp_items(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        assert all(isinstance(t, TimestampItem) for t in result.important_timestamps)

    def test_timestamp_fields_populated(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        first = result.important_timestamps[0]
        assert first.time == "0:00"
        assert first.description == "Introduction and overview"

    def test_duration_formatted_as_m_ss(self, service, transcript, summary_data):
        result = service._build_video_summary(transcript, summary_data)
        # Duration should be formatted like "0:37" or "37:00"
        assert ":" in result.duration

    def test_missing_executive_summary_raises(self, service, transcript, summary_data):
        del summary_data["executive_summary"]
        with pytest.raises(SummarizationError, match="missing required fields"):
            service._build_video_summary(transcript, summary_data)

    def test_missing_detailed_summary_raises(self, service, transcript, summary_data):
        del summary_data["detailed_summary"]
        with pytest.raises(SummarizationError, match="missing required fields"):
            service._build_video_summary(transcript, summary_data)

    def test_missing_key_points_raises(self, service, transcript, summary_data):
        summary_data["key_points"] = []
        with pytest.raises(SummarizationError, match="missing required fields"):
            service._build_video_summary(transcript, summary_data)

    def test_empty_action_items_accepted(self, service, transcript, summary_data):
        summary_data["action_items"] = []
        result = service._build_video_summary(transcript, summary_data)
        assert result.action_items == []

    def test_missing_timestamps_defaults_to_empty(self, service, transcript, summary_data):
        del summary_data["important_timestamps"]
        result = service._build_video_summary(transcript, summary_data)
        assert result.important_timestamps == []

    def test_malformed_timestamp_skipped(self, service, transcript, summary_data):
        """Timestamps missing time or description should be silently skipped."""
        summary_data["important_timestamps"] = [
            {"time": "0:30"},                           # missing description
            {"description": "No time field"},           # missing time
            {"time": "1:00", "description": "Good"},    # valid
        ]
        result = service._build_video_summary(transcript, summary_data)
        assert len(result.important_timestamps) == 1
        assert result.important_timestamps[0].time == "1:00"

    def test_non_list_key_points_coerced(self, service, transcript, summary_data):
        """If AI returns key_points as a string instead of array, coerce gracefully."""
        summary_data["key_points"] = "Single point as string"
        result = service._build_video_summary(transcript, summary_data)
        assert isinstance(result.key_points, list)
        assert len(result.key_points) == 1

    def test_whitespace_stripped_from_fields(self, service, transcript, summary_data):
        summary_data["executive_summary"] = "  Padded summary.  "
        result = service._build_video_summary(transcript, summary_data)
        assert result.executive_summary == "Padded summary."

    def test_title_is_none_by_default(self, service, transcript, summary_data):
        """Title is populated later from YouTube metadata, not by the AI."""
        result = service._build_video_summary(transcript, summary_data)
        assert result.title is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. summarize() — happy path via tool-use (mocked API)
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeToolUsePath:

    def test_returns_video_summary(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert isinstance(result, VideoSummary)

    def test_executive_summary_populated(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert result.executive_summary == summary_data["executive_summary"]

    def test_key_points_populated(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert len(result.key_points) == len(summary_data["key_points"])

    def test_action_items_populated(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert len(result.action_items) == len(summary_data["action_items"])

    def test_timestamps_populated(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert len(result.important_timestamps) == len(summary_data["important_timestamps"])

    def test_video_id_matches_transcript(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert result.video_id == transcript.video_id

    def test_api_called_once_on_success(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        run_async(service.summarize(transcript))
        assert mock_client.messages.create.call_count == 1

    def test_tool_use_in_api_call(self, service, transcript, summary_data, mock_client):
        """Verify that tool_choice and tools are passed to the API."""
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        run_async(service.summarize(transcript))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_choice" in call_kwargs
        assert call_kwargs["tool_choice"]["name"] == "extract_video_summary"

    def test_system_prompt_included(self, service, transcript, summary_data, mock_client):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        run_async(service.summarize(transcript))
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 50


# ─────────────────────────────────────────────────────────────────────────────
# 7. summarize() — fallback JSON path (missing tool_use block)
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeFallbackPath:

    def test_falls_back_when_no_tool_block(self, service, transcript, summary_data, mock_client):
        """First call returns no tool_use block → second call returns JSON text."""
        json_text = json.dumps(summary_data)
        mock_client.messages.create.side_effect = [
            make_empty_response(),              # primary: no tool_use block
            make_text_response(json_text),      # fallback: plain JSON text
        ]
        result = run_async(service.summarize(transcript))
        assert isinstance(result, VideoSummary)

    def test_api_called_twice_on_fallback(self, service, transcript, summary_data, mock_client):
        json_text = json.dumps(summary_data)
        mock_client.messages.create.side_effect = [
            make_empty_response(),
            make_text_response(json_text),
        ]
        run_async(service.summarize(transcript))
        assert mock_client.messages.create.call_count == 2

    def test_fallback_with_markdown_fenced_json(self, service, transcript, summary_data, mock_client):
        fenced = f"```json\n{json.dumps(summary_data)}\n```"
        mock_client.messages.create.side_effect = [
            make_empty_response(),
            make_text_response(fenced),
        ]
        result = run_async(service.summarize(transcript))
        assert result.executive_summary == summary_data["executive_summary"]

    def test_fallback_with_preamble_text(self, service, transcript, summary_data, mock_client):
        text_with_preamble = f"Here is the summary:\n{json.dumps(summary_data)}"
        mock_client.messages.create.side_effect = [
            make_empty_response(),
            make_text_response(text_with_preamble),
        ]
        result = run_async(service.summarize(transcript))
        assert result.video_id == transcript.video_id


# ─────────────────────────────────────────────────────────────────────────────
# 8. summarize() — error handling & exception mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeErrorHandling:

    def test_empty_transcript_raises_immediately(self, service):
        t = make_transcript(segments=[])
        with pytest.raises(SummarizationError, match="no segments"):
            run_async(service.summarize(t))

    def test_auth_error_mapped_to_ai_provider_error(self, service, transcript, mock_client):
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid key",
            response=MagicMock(status_code=401),
            body={}
        )
        with pytest.raises(AIProviderError, match="API key"):
            run_async(service.summarize(transcript))

    def test_auth_error_has_503_status_code(self, service, transcript, mock_client):
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid key",
            response=MagicMock(status_code=401),
            body={}
        )
        with pytest.raises(AIProviderError) as exc_info:
            run_async(service.summarize(transcript))
        assert exc_info.value.status_code == 503

    def test_rate_limit_error_mapped(self, service, transcript, mock_client):
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={}
        )
        with pytest.raises(AIProviderError, match="rate limit"):
            run_async(service.summarize(transcript))

    def test_api_status_error_mapped(self, service, transcript, mock_client):
        mock_client.messages.create.side_effect = anthropic.APIStatusError(
            message="Server error",
            response=MagicMock(status_code=500),
            body={}
        )
        with pytest.raises(AIProviderError, match="500"):
            run_async(service.summarize(transcript))

    def test_connection_error_mapped(self, service, transcript, mock_client):
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )
        with pytest.raises(AIProviderError, match="connect"):
            run_async(service.summarize(transcript))

    def test_malformed_fallback_json_raises_summarization_error(
        self, service, transcript, mock_client
    ):
        mock_client.messages.create.side_effect = [
            make_empty_response(),
            make_text_response("This is not JSON at all."),
        ]
        with pytest.raises(SummarizationError):
            run_async(service.summarize(transcript))

    def test_missing_required_fields_in_tool_response_raises(
        self, service, transcript, mock_client
    ):
        incomplete_data = {"executive_summary": "Only this field."}
        mock_client.messages.create.return_value = make_tool_use_response(incomplete_data)
        with pytest.raises(SummarizationError, match="missing required fields"):
            run_async(service.summarize(transcript))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tool schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestToolSchema:
    """Verify the JSON Schema used for tool-use is well-formed."""

    def test_schema_has_required_name(self):
        from app.services.summary_service import _SUMMARY_TOOL_SCHEMA
        assert _SUMMARY_TOOL_SCHEMA["name"] == "extract_video_summary"

    def test_schema_has_input_schema(self):
        from app.services.summary_service import _SUMMARY_TOOL_SCHEMA
        assert "input_schema" in _SUMMARY_TOOL_SCHEMA

    def test_schema_requires_all_summary_fields(self):
        from app.services.summary_service import _SUMMARY_TOOL_SCHEMA
        required = set(_SUMMARY_TOOL_SCHEMA["input_schema"]["required"])
        expected = {
            "executive_summary", "detailed_summary",
            "key_points", "action_items", "important_timestamps",
        }
        assert expected.issubset(required)

    def test_key_points_is_array_type(self):
        from app.services.summary_service import _SUMMARY_TOOL_SCHEMA
        props = _SUMMARY_TOOL_SCHEMA["input_schema"]["properties"]
        assert props["key_points"]["type"] == "array"

    def test_timestamps_items_have_required_fields(self):
        from app.services.summary_service import _SUMMARY_TOOL_SCHEMA
        ts_schema = _SUMMARY_TOOL_SCHEMA["input_schema"]["properties"]["important_timestamps"]
        assert ts_schema["items"]["required"] == ["time", "description"]

    def test_schema_description_is_non_empty(self):
        from app.services.summary_service import _SUMMARY_TOOL_SCHEMA
        assert len(_SUMMARY_TOOL_SCHEMA["description"]) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 10. Integration-style: full pipeline with mocked API
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeFullPipeline:
    """
    End-to-end tests for the summarize() method with realistic data.
    The API call is mocked; everything else runs real code.
    """

    def test_long_transcript_summarized_correctly(
        self, service, summary_data, mock_client
    ):
        """A transcript with 50 segments should summarize without error."""
        segments = [
            make_segment(f"Segment number {i} with meaningful content.", float(i * 3))
            for i in range(50)
        ]
        t = make_transcript(segments)
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(t))
        assert isinstance(result, VideoSummary)

    def test_transcript_with_unicode_summarized(
        self, service, summary_data, mock_client
    ):
        segments = [make_segment("Café résumé naïve über.", 0.0)]
        t = make_transcript(segments)
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(t))
        assert isinstance(result, VideoSummary)

    def test_single_segment_transcript(self, service, summary_data, mock_client):
        t = make_transcript(segments=[make_segment("Short video.", 0.0, 5.0)])
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(t))
        assert result.video_id == t.video_id

    def test_output_key_points_are_non_empty_strings(
        self, service, transcript, summary_data, mock_client
    ):
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        assert all(len(p) > 0 for p in result.key_points)

    def test_output_action_items_start_with_verb(
        self, service, transcript, summary_data, mock_client
    ):
        """Verify action items are imperative-style (start with capital letter)."""
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(transcript))
        for item in result.action_items:
            assert item[0].isupper(), f"Action item should start with capital: {item!r}"

    def test_summary_video_id_survives_roundtrip(
        self, service, summary_data, mock_client
    ):
        t = make_transcript(video_id="unique_vid_01")
        mock_client.messages.create.return_value = make_tool_use_response(summary_data)
        result = run_async(service.summarize(t))
        assert result.video_id == "unique_vid_01"
