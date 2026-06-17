"""
services/transcript_service.py
─────────────────────────────────────────────────────────────────────────────
Transcript extraction service.

Responsibilities:
  1. Validate and parse YouTube URLs into video IDs
  2. Fetch transcripts via youtube-transcript-api (v1.2.4+)
  3. Clean and normalise raw transcript text
  4. Return typed VideoTranscript domain objects

URL formats supported:
  • https://www.youtube.com/watch?v=VIDEO_ID
  • https://youtu.be/VIDEO_ID
  • https://youtube.com/watch?v=VIDEO_ID&t=42s   (extra query params ignored)
  • https://www.youtube.com/embed/VIDEO_ID
  • https://www.youtube.com/shorts/VIDEO_ID
  • http variants and missing-scheme variants (youtube.com/watch?v=...)

Text cleaning applied:
  • Strip leading/trailing whitespace per segment
  • Collapse multiple internal spaces to one
  • Remove formatting tokens: [Music], [Applause], (music), etc.
  • Normalise unicode (NFKC) — fixes "smart quotes", ligatures, etc.
  • Join segments with a single space; no double-spaces at joins

Architecture note:
  The public method `get_transcript` is async so it fits naturally into the
  FastAPI async request handler. The youtube-transcript-api call is blocking
  (it uses requests under the hood), so we run it via asyncio.to_thread()
  to avoid blocking the event loop.
"""

import asyncio
import html
import re
import unicodedata
from typing import List
from urllib.parse import parse_qs, urlparse

from loguru import logger

from app.core.config import settings
from app.core.exceptions import (
    InvalidYouTubeURLError,
    TranscriptFetchError,
    TranscriptNotAvailableError,
)
from app.models.video import TranscriptSegment, VideoTranscript


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Matches the 11-character YouTube video ID in every known URL shape.
# Group 1 always captures the ID.
_VIDEO_ID_PATTERNS: List[re.Pattern] = [
    # Standard watch URL:  youtube.com/watch?v=VIDEO_ID
    re.compile(r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})"),
    # Short URL:           youtu.be/VIDEO_ID
    re.compile(r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})"),
    # Embed URL:           youtube.com/embed/VIDEO_ID
    re.compile(r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})"),
    # Shorts URL:          youtube.com/shorts/VIDEO_ID
    re.compile(r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"),
    # Live URL:            youtube.com/live/VIDEO_ID
    re.compile(r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})"),
]

# Auto-generated caption noise to strip (case-insensitive).
# These tokens appear frequently in auto-generated transcripts and add
# no semantic value to a summary.
_NOISE_PATTERN: re.Pattern = re.compile(
    r"""
    \[                          # opening bracket
        (?:
            (?:background\s+)?music |
            applause             |
            laughter             |
            inaudible            |
            crosstalk            |
            silence              |
            pause                |
            (?:background\s+)?noise |
            cheering             |
            clapping             |
            ♪[^♪]*♪              # musical notes wrapping lyrics
        )
    \]                          # closing bracket
    |
    \(                          # opening paren
        (?:
            music                |
            applause             |
            laughter             |
            inaudible
        )
    \)                          # closing paren
    |
    ♪[^♪\n]*♪                  # ♪ ... ♪ without brackets
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Collapse any run of whitespace (spaces, tabs, newlines) to a single space.
_WHITESPACE_RE: re.Pattern = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptService
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptService:
    """
    Fetches and normalises YouTube video transcripts.

    Example usage:
        service = TranscriptService()

        # Basic usage — returns a VideoTranscript domain object
        transcript = await service.get_transcript(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        print(transcript.full_text[:200])
        print(f"{transcript.word_count} words across {len(transcript.segments)} segments")

        # Low-level helpers are also public for testing / flexibility
        video_id = TranscriptService.extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        # → "dQw4w9WgXcQ"
    """

    def __init__(self) -> None:
        self._languages: List[str] = settings.transcript_languages_list
        self._max_length: int = settings.max_transcript_length
        logger.debug(
            f"TranscriptService initialised | "
            f"languages={self._languages} | "
            f"max_length={self._max_length}"
        )

    # ── Public API ────────────────────────────────────────────────

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """
        Parse the 11-character video ID from any supported YouTube URL.

        Tries each pattern in `_VIDEO_ID_PATTERNS` in order. Also handles
        bare video IDs (exactly 11 chars of [a-zA-Z0-9_-]) passed directly.

        Args:
            youtube_url: Any YouTube URL or bare video ID string.

        Returns:
            The 11-character video ID string.

        Raises:
            InvalidYouTubeURLError: No YouTube video ID could be found.

        Examples:
            >>> TranscriptService.extract_video_id("https://youtu.be/dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
            >>> TranscriptService.extract_video_id(
            ...     "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"
            ... )
            'dQw4w9WgXcQ'
            >>> TranscriptService.extract_video_id("dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
        """
        url = youtube_url.strip()

        # Accept bare 11-char video IDs directly (useful in tests / CLI)
        if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
            logger.debug(f"extract_video_id: bare video ID accepted: {url!r}")
            return url

        # Normalise: add scheme if missing so urlparse works correctly
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url

        for pattern in _VIDEO_ID_PATTERNS:
            match = pattern.search(url)
            if match:
                video_id = match.group(1)
                pattern_preview = repr(pattern.pattern)[:40]
                logger.debug(
                    f"extract_video_id: matched pattern={pattern_preview} "
                    f"→ {video_id!r}"
                )
                return video_id

        logger.warning(f"extract_video_id: no match for URL {youtube_url!r}")
        raise InvalidYouTubeURLError(youtube_url)

    async def get_transcript(self, youtube_url: str) -> VideoTranscript:
        """
        Fetch and return a clean VideoTranscript for the given YouTube URL.

        Flow:
          1. Extract video ID from URL (raises InvalidYouTubeURLError on failure)
          2. Call youtube-transcript-api in a thread (non-blocking)
          3. Clean and normalise each segment's text
          4. Trim to max_transcript_length if necessary
          5. Return a VideoTranscript domain object

        Args:
            youtube_url: Any supported YouTube URL.

        Returns:
            VideoTranscript with cleaned segments and metadata.

        Raises:
            InvalidYouTubeURLError:       URL cannot be parsed.
            TranscriptNotAvailableError:  No transcript for this video.
            TranscriptFetchError:         Network / API failure.

        Example:
            transcript = await service.get_transcript(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            # transcript.full_text  → single clean string
            # transcript.segments   → list of TranscriptSegment
            # transcript.word_count → int
        """
        video_id = self.extract_video_id(youtube_url)
        logger.info(f"Fetching transcript | video_id={video_id!r}")

        raw_snippets, language_code = await self._fetch_raw_transcript(video_id)

        segments = self._build_segments(raw_snippets)
        segments = self._trim_segments(segments)

        transcript = VideoTranscript(
            video_id=video_id,
            language=language_code,
            segments=segments,
        )

        logger.info(
            f"Transcript ready | video_id={video_id!r} "
            f"language={language_code!r} "
            f"segments={len(segments)} "
            f"words={transcript.word_count} "
            f"duration={transcript.total_duration_seconds:.0f}s"
        )
        return transcript

    @staticmethod
    def clean_text(raw: str) -> str:
        """
        Apply all text-cleaning steps to a single string.

        Steps (in order):
          1. Decode HTML entities  (&amp; → &,  &#39; → ', etc.)
          2. Normalise unicode     (NFKC: ligatures, fancy quotes → ASCII)
          3. Remove noise tokens   ([Music], [Applause], ♪...♪, etc.)
          4. Collapse whitespace   (tabs, newlines, double-spaces → single space)
          5. Strip surrounding whitespace

        This method is public so callers can use it independently (e.g. in
        tests or when cleaning user-supplied text before sending to the AI).

        Args:
            raw: Raw text string from a transcript snippet.

        Returns:
            Cleaned, normalised text string. May be empty string if the
            input was pure noise (e.g. "[Music]").

        Examples:
            >>> TranscriptService.clean_text("Hello  world\\n")
            'Hello world'
            >>> TranscriptService.clean_text("[Music] Welcome back")
            'Welcome back'
            >>> TranscriptService.clean_text("&amp; so on &#39;til now")
            '& so on \\'til now'
        """
        if not raw:
            return ""

        # 1. Decode HTML entities
        text = html.unescape(raw)

        # 2. Normalise unicode (handles "ﬁ" → "fi", curly quotes, etc.)
        text = unicodedata.normalize("NFKC", text)

        # 3. Remove noise tokens
        text = _NOISE_PATTERN.sub("", text)

        # 4. Collapse all whitespace runs
        text = _WHITESPACE_RE.sub(" ", text)

        # 5. Strip
        return text.strip()

    # ── Private helpers ───────────────────────────────────────────

    async def _fetch_raw_transcript(
        self, video_id: str
    ) -> tuple[list, str]:
        """
        Call youtube-transcript-api in a thread pool (blocking I/O).

        Strategy:
          1. Try `YouTubeTranscriptApi().fetch(video_id, languages=preferred)`
             — returns the best available language from our preference list.
          2. If that fails with NoTranscriptFound (preferred language absent),
             list all available transcripts and fetch the first one regardless
             of language — better than nothing for a summarizer.
          3. Map all known library exceptions to our typed exceptions.

        Returns:
            (snippets, language_code) where snippets is a list of
            FetchedTranscriptSnippet objects.

        Raises:
            TranscriptNotAvailableError: Captions are disabled or video
                                         has no captions at all.
            TranscriptFetchError:        Any other failure (network, age gate,
                                         IP block, etc.)
        """
        return await asyncio.to_thread(self._fetch_sync, video_id)

    def _fetch_sync(self, video_id: str) -> tuple[list, str]:
        """
        Synchronous transcript fetch — runs inside asyncio.to_thread().
        Separated out so it is easily unit-testable without an event loop.
        """
        # Import here so the rest of the module is importable even if the
        # library is not installed (e.g. during isolated unit tests).
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            CouldNotRetrieveTranscript,
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        api = YouTubeTranscriptApi()

        # ── Attempt 1: preferred languages ────────────────────────
        try:
            fetched = api.fetch(video_id, languages=self._languages)
            return list(fetched.snippets), fetched.language_code

        except NoTranscriptFound:
            logger.info(
                f"Preferred languages {self._languages} not found for "
                f"{video_id!r}. Trying any available language."
            )

        except TranscriptsDisabled:
            raise TranscriptNotAvailableError(video_id)

        except VideoUnavailable:
            raise TranscriptFetchError(
                video_id, "Video is unavailable (private, deleted, or region-locked)."
            )

        except CouldNotRetrieveTranscript as exc:
            # Covers: AgeRestricted, IpBlocked, PoTokenRequired, etc.
            raise TranscriptFetchError(video_id, str(exc))

        # ── Attempt 2: any available language ────────────────────
        try:
            transcript_list = api.list(video_id)
            # find_transcript tries manual first, then auto-generated
            transcript = transcript_list.find_transcript(["en"])
            fetched = transcript.fetch()
            return list(fetched.snippets), fetched.language_code

        except NoTranscriptFound:
            # No transcript in any language — genuinely unavailable
            raise TranscriptNotAvailableError(video_id)

        except TranscriptsDisabled:
            raise TranscriptNotAvailableError(video_id)

        except Exception as exc:
            raise TranscriptFetchError(video_id, str(exc)) from exc

    def _build_segments(self, raw_snippets: list) -> List[TranscriptSegment]:
        """
        Convert raw FetchedTranscriptSnippet objects into TranscriptSegment
        domain objects, applying text cleaning to each snippet.

        Snippets whose cleaned text is empty (pure noise, e.g. "[Music]") are
        silently dropped — they carry no information for the summarizer.

        Args:
            raw_snippets: List of FetchedTranscriptSnippet (text/start/duration).

        Returns:
            List of TranscriptSegment with cleaned text.
        """
        segments: List[TranscriptSegment] = []
        dropped = 0

        for snippet in raw_snippets:
            cleaned = self.clean_text(snippet.text)
            if not cleaned:
                dropped += 1
                continue

            segments.append(
                TranscriptSegment(
                    text=cleaned,
                    start=float(snippet.start),
                    duration=float(snippet.duration),
                )
            )

        if dropped:
            logger.debug(f"Dropped {dropped} empty/noise segments after cleaning.")

        return segments

    def _trim_segments(
        self, segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """
        Trim segments so the concatenated transcript does not exceed
        `self._max_length` characters (controlled by MAX_TRANSCRIPT_LENGTH
        in .env).

        Trimming drops segments from the END of the transcript — the
        beginning (introduction, thesis) is usually more valuable for a
        summary than the tail (outro, credits).

        A warning is logged if trimming occurs so operators can tune the limit.

        Args:
            segments: Cleaned segments list (may be empty).

        Returns:
            Possibly shortened list of TranscriptSegment.
        """
        if not segments:
            return segments

        kept: List[TranscriptSegment] = []
        total_chars = 0

        for seg in segments:
            seg_len = len(seg.text) + 1  # +1 for the space joining segments
            if total_chars + seg_len > self._max_length:
                # Always keep at least one segment so the caller never gets
                # an empty result even when a single segment exceeds the limit.
                if not kept:
                    kept.append(seg)
                logger.warning(
                    f"Transcript trimmed at {total_chars} chars "
                    f"(limit={self._max_length}). "
                    f"Dropped {len(segments) - len(kept)} trailing segments."
                )
                break
            kept.append(seg)
            total_chars += seg_len

        return kept
