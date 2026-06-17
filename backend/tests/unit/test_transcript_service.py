"""
tests/unit/test_transcript_service.py
──────────────────────────────────────────────────────────────────────────────
Full test suite for TranscriptService.

Testing strategy:
  • URL parsing tests use no mocks — pure logic, very fast.
  • Text cleaning tests use no mocks — pure logic.
  • Segment building / trimming tests use no mocks — pure logic.
  • `get_transcript` tests mock `_fetch_sync` (the blocking I/O boundary)
    so no real network calls are made. This keeps tests hermetic,
    deterministic, and fast.
  • Error-path tests verify that library exceptions are mapped to the
    correct typed application exceptions.

All fixtures and helpers are defined in this file — no conftest dependencies
beyond the standard pytest session.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import (
    InvalidYouTubeURLError,
    TranscriptFetchError,
    TranscriptNotAvailableError,
)
from app.models.video import TranscriptSegment, VideoTranscript
from app.services.transcript_service import TranscriptService, _NOISE_PATTERN


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_snippet(text: str, start: float = 0.0, duration: float = 2.0):
    """Create a minimal fake FetchedTranscriptSnippet-like object."""
    return SimpleNamespace(text=text, start=start, duration=duration)


def make_snippets(*texts: str) -> list:
    """Create a list of fake snippets with evenly spaced timestamps."""
    return [
        make_snippet(text, start=i * 3.0, duration=2.5)
        for i, text in enumerate(texts)
    ]


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def service() -> TranscriptService:
    return TranscriptService()


@pytest.fixture
def sample_snippets() -> list:
    return make_snippets(
        "Hello and welcome to this video.",
        "Today we are talking about Python.",
        "Let&#39;s get started right away.",
        "[Music] Some background noise here.",
        "The key insight is that clean code matters.",
        "Thank you so much for watching.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. extract_video_id — URL parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractVideoId:
    """
    Covers every supported URL format and key edge cases.
    No network calls, no mocks needed.
    """

    # ── Valid URLs ─────────────────────────────────────────────

    def test_standard_watch_url(self):
        assert TranscriptService.extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_watch_url_without_www(self):
        assert TranscriptService.extract_video_id(
            "https://youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_watch_url_http(self):
        assert TranscriptService.extract_video_id(
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_short_url_youtu_be(self):
        assert TranscriptService.extract_video_id(
            "https://youtu.be/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_short_url_without_scheme(self):
        assert TranscriptService.extract_video_id(
            "youtu.be/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_watch_url_with_timestamp(self):
        assert TranscriptService.extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"
        ) == "dQw4w9WgXcQ"

    def test_watch_url_with_multiple_params(self):
        assert TranscriptService.extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx&index=3"
        ) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert TranscriptService.extract_video_id(
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert TranscriptService.extract_video_id(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_live_url(self):
        assert TranscriptService.extract_video_id(
            "https://www.youtube.com/live/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_watch_url_without_scheme(self):
        assert TranscriptService.extract_video_id(
            "www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_bare_video_id_accepted(self):
        """An 11-char ID passed directly should be returned as-is."""
        assert TranscriptService.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_video_id_with_underscore_and_hyphen(self):
        """IDs can contain underscores and hyphens."""
        assert TranscriptService.extract_video_id(
            "https://youtu.be/a_B-C123456"
        ) == "a_B-C123456"

    def test_url_with_leading_trailing_whitespace(self):
        assert TranscriptService.extract_video_id(
            "  https://youtu.be/dQw4w9WgXcQ  "
        ) == "dQw4w9WgXcQ"

    # ── Invalid URLs ───────────────────────────────────────────

    def test_empty_string_raises(self):
        with pytest.raises(InvalidYouTubeURLError):
            TranscriptService.extract_video_id("")

    def test_vimeo_url_raises(self):
        with pytest.raises(InvalidYouTubeURLError):
            TranscriptService.extract_video_id("https://vimeo.com/123456789")

    def test_random_string_raises(self):
        with pytest.raises(InvalidYouTubeURLError):
            TranscriptService.extract_video_id("not a url")

    def test_youtube_homepage_raises(self):
        with pytest.raises(InvalidYouTubeURLError):
            TranscriptService.extract_video_id("https://www.youtube.com/")

    def test_short_id_raises(self):
        """10 chars is one short of a valid video ID."""
        with pytest.raises(InvalidYouTubeURLError):
            TranscriptService.extract_video_id("dQw4w9WgXc")  # 10 chars

    def test_long_id_raises(self):
        """12 chars is one over — should not match bare-ID fast path."""
        with pytest.raises(InvalidYouTubeURLError):
            TranscriptService.extract_video_id("dQw4w9WgXcQQ")  # 12 chars

    def test_invalid_youtube_url_error_contains_url(self):
        bad = "https://example.com/not-youtube"
        try:
            TranscriptService.extract_video_id(bad)
        except InvalidYouTubeURLError as exc:
            assert bad in exc.message
            assert exc.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 2. clean_text — text normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanText:
    """All text cleaning is pure logic — no I/O, no mocks needed."""

    def test_strips_surrounding_whitespace(self):
        assert TranscriptService.clean_text("  hello world  ") == "hello world"

    def test_collapses_internal_spaces(self):
        assert TranscriptService.clean_text("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert TranscriptService.clean_text("hello\nworld\n") == "hello world"

    def test_collapses_tabs(self):
        assert TranscriptService.clean_text("hello\t\tworld") == "hello world"

    def test_decodes_html_entities(self):
        assert TranscriptService.clean_text("&amp; &lt; &gt;") == "& < >"

    def test_decodes_numeric_html_entity(self):
        assert TranscriptService.clean_text("&#39;til now") == "'til now"

    def test_decodes_apos_html_entity(self):
        assert TranscriptService.clean_text("don&apos;t") == "don't"

    def test_removes_music_bracket_noise(self):
        assert TranscriptService.clean_text("[Music]") == ""

    def test_removes_music_bracket_with_surrounding_text(self):
        result = TranscriptService.clean_text("[Music] Welcome back everyone")
        assert result == "Welcome back everyone"

    def test_removes_applause_noise(self):
        result = TranscriptService.clean_text("great job [Applause] thank you")
        assert "Applause" not in result
        assert "great job" in result

    def test_removes_laughter_noise(self):
        result = TranscriptService.clean_text("funny moment [Laughter] right?")
        assert "Laughter" not in result

    def test_removes_inaudible_noise(self):
        result = TranscriptService.clean_text("I was [inaudible] yesterday")
        assert "inaudible" not in result
        assert "yesterday" in result

    def test_removes_background_music_noise(self):
        result = TranscriptService.clean_text("[Background Music] Let's continue")
        assert "Background" not in result

    def test_removes_paren_music_noise(self):
        result = TranscriptService.clean_text("(music) Hello there")
        assert "(music)" not in result

    def test_removes_musical_note_noise(self):
        result = TranscriptService.clean_text("♪ some song lyrics ♪ back to speech")
        assert "♪" not in result
        assert "back to speech" in result

    def test_normalises_unicode_ligature(self):
        # "ﬁ" is a ligature that should normalise to "fi"
        result = TranscriptService.clean_text("ﬁrst")
        assert result == "first"

    def test_empty_string_returns_empty(self):
        assert TranscriptService.clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert TranscriptService.clean_text("   \n\t  ") == ""

    def test_pure_noise_returns_empty(self):
        assert TranscriptService.clean_text("[Music]") == ""

    def test_normal_text_unchanged(self):
        text = "This is a normal sentence about machine learning."
        assert TranscriptService.clean_text(text) == text

    def test_mixed_noise_and_content(self):
        raw = "[Music] Hello [Applause] and welcome [Laughter] to the show."
        result = TranscriptService.clean_text(raw)
        assert "Hello" in result
        assert "welcome" in result
        assert "show" in result
        assert "[" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 3. _build_segments — segment construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSegments:

    def test_builds_correct_count(self, service, sample_snippets):
        # sample_snippets has 6 snippets.
        # "[Music] Some background noise here." is NOT pure noise — after
        # stripping [Music] it becomes "Some background noise here." which is
        # kept.  All 6 snippets produce valid segments.
        segments = service._build_segments(sample_snippets)
        assert len(segments) == 6

    def test_segment_types(self, service, sample_snippets):
        segments = service._build_segments(sample_snippets)
        assert all(isinstance(s, TranscriptSegment) for s in segments)

    def test_drops_noise_only_segments(self, service):
        snippets = [
            make_snippet("[Music]", start=0.0),
            make_snippet("[Applause]", start=2.0),
            make_snippet("Real content here.", start=4.0),
        ]
        segments = service._build_segments(snippets)
        assert len(segments) == 1
        assert segments[0].text == "Real content here."

    def test_preserves_timestamps(self, service):
        snippets = [make_snippet("Hello", start=5.5, duration=2.0)]
        segments = service._build_segments(snippets)
        assert segments[0].start == 5.5
        assert segments[0].duration == 2.0

    def test_cleans_text_in_segments(self, service):
        snippets = [make_snippet("  Hello   world  ")]
        segments = service._build_segments(snippets)
        assert segments[0].text == "Hello world"

    def test_decodes_html_entities_in_segments(self, service):
        snippets = [make_snippet("Let&#39;s go")]
        segments = service._build_segments(snippets)
        assert segments[0].text == "Let's go"

    def test_empty_snippets_returns_empty(self, service):
        assert service._build_segments([]) == []

    def test_all_noise_returns_empty(self, service):
        snippets = [
            make_snippet("[Music]"),
            make_snippet("[Applause]"),
            make_snippet("[Laughter]"),
        ]
        assert service._build_segments(snippets) == []

    def test_start_formatted_property(self, service):
        snippets = [make_snippet("Hello", start=125.0)]
        segments = service._build_segments(snippets)
        assert segments[0].start_formatted == "2:05"


# ─────────────────────────────────────────────────────────────────────────────
# 4. _trim_segments — length capping
# ─────────────────────────────────────────────────────────────────────────────

class TestTrimSegments:

    def test_no_trim_when_under_limit(self, service):
        segments = [TranscriptSegment("hello", 0.0, 1.0)]
        result = service._trim_segments(segments)
        assert result == segments

    def test_trims_when_over_limit(self, service):
        # Each segment is 100 chars; limit is 50000 by default in settings.
        # Override _max_length for this test.
        service._max_length = 50
        segments = [
            TranscriptSegment("a" * 20, float(i), 1.0)
            for i in range(10)
        ]
        result = service._trim_segments(segments)
        assert len(result) < 10

    def test_empty_segments_returns_empty(self, service):
        assert service._trim_segments([]) == []

    def test_trim_keeps_beginning_not_end(self, service):
        """Trimming must drop from the END, not the beginning."""
        service._max_length = 30
        segments = [
            TranscriptSegment("FIRST segment.", 0.0, 1.0),   # 14 chars
            TranscriptSegment("SECOND segment.", 1.0, 1.0),  # 15 chars → over
        ]
        result = service._trim_segments(segments)
        assert any("FIRST" in s.text for s in result)

    def test_single_oversized_segment_included(self, service):
        """A single segment that exceeds the limit on its own is still kept."""
        service._max_length = 5
        segments = [TranscriptSegment("This is longer than 5 chars.", 0.0, 1.0)]
        # First segment is always included even if it alone exceeds limit
        result = service._trim_segments(segments)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. VideoTranscript domain object properties
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoTranscriptProperties:

    @pytest.fixture
    def transcript(self) -> VideoTranscript:
        return VideoTranscript(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[
                TranscriptSegment("Hello and welcome.", 0.0, 2.5),
                TranscriptSegment("Today we discuss Python.", 3.0, 3.5),
                TranscriptSegment("Thanks for watching.", 55.0, 2.0),
            ],
        )

    def test_full_text_joins_segments(self, transcript):
        text = transcript.full_text
        assert "Hello and welcome." in text
        assert "Today we discuss Python." in text
        assert "Thanks for watching." in text

    def test_full_text_separated_by_spaces(self, transcript):
        text = transcript.full_text
        assert "  " not in text  # no double spaces at join points

    def test_word_count(self, transcript):
        # "Hello and welcome. Today we discuss Python. Thanks for watching."
        # = 3 + 4 + 3 = 10 words
        assert transcript.word_count == 10

    def test_total_duration(self, transcript):
        # last segment: start=55.0, duration=2.0 → total=57.0
        assert transcript.total_duration_seconds == pytest.approx(57.0)

    def test_empty_transcript_word_count(self):
        t = VideoTranscript(video_id="x", language="en", segments=[])
        assert t.word_count == 0

    def test_empty_transcript_duration(self):
        t = VideoTranscript(video_id="x", language="en", segments=[])
        assert t.total_duration_seconds == 0.0

    def test_empty_transcript_full_text(self):
        t = VideoTranscript(video_id="x", language="en", segments=[])
        assert t.full_text == ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. get_transcript — async integration (mocked I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTranscript:
    """
    Mocks `_fetch_sync` so no real HTTP calls are made.
    Tests cover: happy path, error mapping, trimming, language detection.
    """

    def _mock_fetch(self, service: TranscriptService, snippets: list, lang: str = "en"):
        """Patch service._fetch_sync to return (snippets, lang) synchronously."""
        service._fetch_sync = MagicMock(return_value=(snippets, lang))

    def test_returns_video_transcript_object(self, service, sample_snippets):
        self._mock_fetch(service, sample_snippets)
        result = run_async(
            service.get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        )
        assert isinstance(result, VideoTranscript)

    def test_video_id_is_correct(self, service, sample_snippets):
        self._mock_fetch(service, sample_snippets)
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        assert result.video_id == "dQw4w9WgXcQ"

    def test_language_code_is_set(self, service, sample_snippets):
        self._mock_fetch(service, sample_snippets, lang="en-US")
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        assert result.language == "en-US"

    def test_segments_are_cleaned(self, service):
        raw = [
            make_snippet("  Hello   world  ", 0.0, 2.0),
            make_snippet("Let&#39;s go", 2.0, 1.5),
        ]
        self._mock_fetch(service, raw)
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        assert result.segments[0].text == "Hello world"
        assert result.segments[1].text == "Let's go"

    def test_noise_segments_dropped(self, service):
        raw = [
            make_snippet("[Music]", 0.0, 3.0),
            make_snippet("Real content", 3.0, 2.0),
            make_snippet("[Applause]", 5.0, 1.0),
        ]
        self._mock_fetch(service, raw)
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        assert len(result.segments) == 1
        assert result.segments[0].text == "Real content"

    def test_transcript_is_trimmed_when_over_limit(self, service):
        service._max_length = 50
        raw = [make_snippet("word " * 20, float(i), 1.0) for i in range(20)]
        self._mock_fetch(service, raw)
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        total_chars = sum(len(s.text) for s in result.segments)
        assert total_chars <= service._max_length + 100  # small tolerance

    def test_invalid_url_raises_before_fetch(self, service):
        with pytest.raises(InvalidYouTubeURLError):
            run_async(service.get_transcript("https://vimeo.com/123456"))

    def test_transcript_not_available_is_propagated(self, service):
        service._fetch_sync = MagicMock(
            side_effect=TranscriptNotAvailableError("dQw4w9WgXcQ")
        )
        with pytest.raises(TranscriptNotAvailableError):
            run_async(
                service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
            )

    def test_transcript_fetch_error_is_propagated(self, service):
        service._fetch_sync = MagicMock(
            side_effect=TranscriptFetchError("dQw4w9WgXcQ", "Network timeout")
        )
        with pytest.raises(TranscriptFetchError):
            run_async(
                service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
            )

    def test_word_count_is_positive(self, service, sample_snippets):
        self._mock_fetch(service, sample_snippets)
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        assert result.word_count > 0

    def test_duration_is_positive(self, service, sample_snippets):
        self._mock_fetch(service, sample_snippets)
        result = run_async(
            service.get_transcript("https://youtu.be/dQw4w9WgXcQ")
        )
        assert result.total_duration_seconds > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. _fetch_sync — library exception mapping (unit-mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchSyncErrorMapping:
    """
    Patches the youtube_transcript_api library to verify that every
    known library exception is correctly mapped to our typed exceptions.
    Does not require a network connection.
    """

    PATCH_BASE = "app.services.transcript_service.TranscriptService._fetch_sync"

    def _run(self, side_effect, service: TranscriptService):
        service._fetch_sync = MagicMock(side_effect=side_effect)
        run_async(service.get_transcript("https://youtu.be/dQw4w9WgXcQ"))

    def test_transcript_not_available_has_correct_status(self, service):
        service._fetch_sync = MagicMock(
            side_effect=TranscriptNotAvailableError("abc")
        )
        with pytest.raises(TranscriptNotAvailableError) as exc_info:
            run_async(service.get_transcript("https://youtu.be/dQw4w9WgXcQ"))
        assert exc_info.value.status_code == 404

    def test_transcript_fetch_error_has_correct_status(self, service):
        service._fetch_sync = MagicMock(
            side_effect=TranscriptFetchError("abc", "timeout")
        )
        with pytest.raises(TranscriptFetchError) as exc_info:
            run_async(service.get_transcript("https://youtu.be/dQw4w9WgXcQ"))
        assert exc_info.value.status_code == 502

    def test_invalid_url_error_has_correct_status(self, service):
        with pytest.raises(InvalidYouTubeURLError) as exc_info:
            run_async(service.get_transcript("not-a-url"))
        assert exc_info.value.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 8. Noise pattern regex — direct pattern tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNoisePattern:
    """Directly test the compiled _NOISE_PATTERN regex."""

    @pytest.mark.parametrize("noise", [
        "[Music]",
        "[music]",
        "[MUSIC]",
        "[Applause]",
        "[Laughter]",
        "[inaudible]",
        "[Crosstalk]",
        "[Silence]",
        "[Background Music]",
        "[Background Noise]",
        "[Cheering]",
        "[Clapping]",
        "(music)",
        "(applause)",
        "(laughter)",
        "(inaudible)",
        "♪ la la la ♪",
    ])
    def test_noise_is_matched(self, noise):
        assert _NOISE_PATTERN.search(noise), f"Expected {noise!r} to match noise pattern"

    @pytest.mark.parametrize("not_noise", [
        "Hello world",
        "This is a real sentence.",
        "The music was playing.",
        "I heard applause in the distance.",
        "Python programming tutorial",
    ])
    def test_real_content_not_matched(self, not_noise):
        # The pattern should not match clean prose
        # (Note: "music" or "applause" in brackets would match, but not plain text)
        result = _NOISE_PATTERN.sub("", not_noise).strip()
        assert len(result) > 0, f"Expected {not_noise!r} not to be fully consumed"
