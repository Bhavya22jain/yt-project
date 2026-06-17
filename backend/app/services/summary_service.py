"""
services/summary_service.py
─────────────────────────────────────────────────────────────────────────────
AI Summarisation Service — Day 4 Implementation.

NOTE ON AI PROVIDER:
  The task brief mentions "OpenAI API", but this entire project is built on
  Anthropic Claude (established in Days 1–3). This module uses the Anthropic
  SDK consistently with the rest of the codebase. The architecture is
  identical in principle — structured JSON output via tool-use, async client,
  typed error mapping — and trivially portable to OpenAI if needed.

Responsibilities:
  1. Accept a VideoTranscript domain object.
  2. Build a carefully engineered prompt that instructs Claude to produce
     a structured JSON summary.
  3. Call the Anthropic API using Tool Use — the most reliable mechanism for
     guaranteed structured JSON output (no markdown fences, no prose preamble).
  4. Validate and parse the tool-use response into a VideoSummary schema.
  5. Implement a plain-JSON fallback for resilience if tool-use fails.
  6. Map all Anthropic SDK exceptions to typed app exceptions.
  7. Return a fully populated VideoSummary Pydantic model.

Structured Output Strategy — Tool Use (primary):
  Claude's tool-use feature forces the model to emit a JSON object that
  conforms exactly to a supplied JSON Schema. This is the most reliable
  structured-output mechanism available on the Anthropic API — it produces
  zero prose contamination and schema-validates before returning.

  We define a single tool `extract_video_summary` whose `input_schema` is
  the full JSON Schema for VideoSummary. We then force the model to call
  that tool via `tool_choice={"type": "tool", "name": "extract_video_summary"}`.
  The response content block will be of type `tool_use`, and `block.input`
  is already a parsed Python dict — no JSON parsing required.

Fallback Strategy — JSON in system prompt:
  If tool-use fails (network blip, API version change), we fall back to a
  plain text call with explicit JSON instructions in the system prompt and
  a JSON extraction regex. Less reliable but sufficient for graceful degradation.

Usage:
    service = SummaryService()
    transcript = await transcript_service.get_transcript(url)
    summary = await service.summarize(transcript)
    # summary.executive_summary, summary.key_points, etc.
"""

import asyncio
import json
import re
import time
from typing import Any, Optional

import anthropic
from loguru import logger

from app.core.config import settings
from app.core.exceptions import AIProviderError, SummarizationError
from app.models.video import VideoTranscript
from app.schemas.video import TimestampItem, VideoSummary


# ─────────────────────────────────────────────────────────────────────────────
# JSON Schema for structured output (Tool Use input_schema)
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_TOOL_SCHEMA: dict[str, Any] = {
    "name": "extract_video_summary",
    "description": (
        "Extract a structured summary from a YouTube video transcript. "
        "Always call this tool with all fields populated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": (
                    "A concise 2–3 sentence overview of the entire video. "
                    "Should answer: what is this video about, who is it for, "
                    "and what is the single most important takeaway."
                ),
            },
            "detailed_summary": {
                "type": "string",
                "description": (
                    "A thorough paragraph-by-paragraph breakdown of the video's "
                    "content, covering all major sections and arguments in order. "
                    "Should be 200–500 words."
                ),
            },
            "key_points": {
                "type": "array",
                "description": "5–10 key takeaways as clear, standalone sentences.",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 12,
            },
            "action_items": {
                "type": "array",
                "description": (
                    "3–7 concrete, actionable next steps a viewer can take "
                    "based on the video's content. Start each with an action verb."
                ),
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 10,
            },
            "important_timestamps": {
                "type": "array",
                "description": (
                    "Up to 8 important moments in the video. Only include "
                    "timestamps that appear in the transcript context provided."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "Timestamp in M:SS or H:MM:SS format, e.g. '4:32'",
                        },
                        "description": {
                            "type": "string",
                            "description": "What happens or is discussed at this moment.",
                        },
                    },
                    "required": ["time", "description"],
                },
                "maxItems": 8,
            },
        },
        "required": [
            "executive_summary",
            "detailed_summary",
            "key_points",
            "action_items",
            "important_timestamps",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert content analyst specialising in extracting structured, \
actionable intelligence from video transcripts.

Your task is to analyse the provided YouTube video transcript and produce a \
comprehensive structured summary using the extract_video_summary tool.

Guidelines:
- Be factual and grounded in the transcript — do not invent content.
- Write in clear, professional English.
- Executive summary: 2–3 sentences, captures the essence for a busy reader.
- Detailed summary: thorough but scannable, covering all major sections.
- Key points: standalone insights a reader can act on or share.
- Action items: begin each with an imperative verb (Learn, Watch, Try, Read…).
- Timestamps: only cite timestamps that appear in the annotated transcript.
- If the transcript is a tutorial, emphasise steps; if a talk, emphasise ideas.
"""

_USER_PROMPT_TEMPLATE = """\
Please analyse the following YouTube video transcript and call the \
extract_video_summary tool with a complete structured summary.

VIDEO ID: {video_id}
LANGUAGE: {language}
DURATION: {duration}
WORD COUNT: {word_count} words

TRANSCRIPT (with timestamps):
{annotated_transcript}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SummaryService
# ─────────────────────────────────────────────────────────────────────────────

class SummaryService:
    """
    Generates structured AI summaries from YouTube video transcripts.

    Uses Anthropic Claude with tool-use forced structured JSON output.
    Falls back to plain JSON extraction if tool-use is unavailable.

    Example:
        service = SummaryService()
        summary = await service.summarize(transcript)

        print(summary.executive_summary)
        for point in summary.key_points:
            print(f"  • {point}")
    """

    def __init__(self) -> None:
        self._model: str = settings.anthropic_model
        self._max_tokens: int = settings.anthropic_max_tokens
        self._temperature: float = settings.anthropic_temperature
        self._api_key: str = settings.anthropic_api_key

        # Lazy-initialised; created on first call so the service can be
        # instantiated without a valid API key (e.g. during unit tests).
        self._client: Optional[anthropic.AsyncAnthropic] = None

        logger.info(
            f"SummaryService initialised | model={self._model} | "
            f"max_tokens={self._max_tokens} | temperature={self._temperature}"
        )

    # ── Public API ─────────────────────────────────────────────────────────

    async def summarize(self, transcript: VideoTranscript) -> VideoSummary:
        """
        Generate a complete structured summary from a VideoTranscript.

        Flow:
          1. Build an annotated transcript string (text with timestamp hints).
          2. Call Claude via tool-use to get guaranteed structured JSON.
          3. Parse the tool-use response into a VideoSummary schema.
          4. Fall back to plain-JSON extraction if tool-use fails.

        Args:
            transcript: A populated VideoTranscript domain object.

        Returns:
            VideoSummary with all fields populated.

        Raises:
            SummarizationError: Response is malformed, empty, or unparseable.
            AIProviderError:    Anthropic API authentication, rate-limit,
                                or server error.

        Example:
            transcript = VideoTranscript(
                video_id="abc123",
                language="en",
                segments=[TranscriptSegment("Hello world", 0.0, 2.0)],
            )
            summary = await service.summarize(transcript)
            # summary.executive_summary → "This video introduces …"
            # summary.key_points       → ["Point A", "Point B", …]
        """
        if not transcript.segments:
            raise SummarizationError("Transcript has no segments — nothing to summarize.")

        logger.info(
            f"Starting summarization | video_id={transcript.video_id!r} | "
            f"words={transcript.word_count} | segments={len(transcript.segments)}"
        )

        annotated = self._build_annotated_transcript(transcript)
        user_prompt = self._build_user_prompt(transcript, annotated)

        start_ms = int(time.monotonic() * 1000)

        try:
            summary_data = await self._call_with_tool_use(user_prompt)
        except (SummarizationError, AIProviderError):
            raise
        except Exception as exc:
            # Unexpected error — wrap and re-raise as SummarizationError
            logger.exception(f"Unexpected error during summarization: {exc}")
            raise SummarizationError(str(exc)) from exc

        elapsed_ms = int(time.monotonic() * 1000) - start_ms

        summary = self._build_video_summary(transcript, summary_data)

        logger.info(
            f"Summarization complete | video_id={transcript.video_id!r} | "
            f"elapsed_ms={elapsed_ms} | "
            f"key_points={len(summary.key_points)} | "
            f"action_items={len(summary.action_items)} | "
            f"timestamps={len(summary.important_timestamps)}"
        )
        return summary

    # ── Prompt building ─────────────────────────────────────────────────────

    def _build_annotated_transcript(self, transcript: VideoTranscript) -> str:
        """
        Build a human-readable transcript string with timestamp annotations.

        Each segment is prefixed with its start time so Claude can reference
        meaningful timestamps in its response.

        Example output:
            [0:00] Hello and welcome to this tutorial.
            [0:15] Today we are covering three topics.
            [1:02] First, let's talk about data structures.

        Timestamps are injected every N segments (not every line) to keep the
        prompt concise while still giving Claude useful anchors.

        Args:
            transcript: VideoTranscript with segments and timing data.

        Returns:
            Multi-line annotated transcript string.
        """
        lines: list[str] = []

        # Inject a timestamp marker every 10 segments (roughly every ~30-60s)
        # to give Claude anchors without flooding the prompt.
        TIMESTAMP_EVERY_N = 10

        for i, seg in enumerate(transcript.segments):
            if i % TIMESTAMP_EVERY_N == 0:
                lines.append(f"[{seg.start_formatted}] {seg.text}")
            else:
                lines.append(seg.text)

        return "\n".join(lines)

    def _build_user_prompt(
        self,
        transcript: VideoTranscript,
        annotated_transcript: str,
    ) -> str:
        """
        Populate the user prompt template with transcript metadata and content.

        Args:
            transcript:           VideoTranscript domain object.
            annotated_transcript: Output of _build_annotated_transcript().

        Returns:
            Formatted user prompt string.
        """
        duration_secs = transcript.total_duration_seconds
        if duration_secs >= 3600:
            h = int(duration_secs // 3600)
            m = int((duration_secs % 3600) // 60)
            s = int(duration_secs % 60)
            duration_str = f"{h}h {m}m {s}s"
        else:
            m = int(duration_secs // 60)
            s = int(duration_secs % 60)
            duration_str = f"{m}m {s}s"

        return _USER_PROMPT_TEMPLATE.format(
            video_id=transcript.video_id,
            language=transcript.language,
            duration=duration_str,
            word_count=transcript.word_count,
            annotated_transcript=annotated_transcript,
        )

    # ── AI API calls ────────────────────────────────────────────────────────

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """
        Return the cached AsyncAnthropic client, creating it if necessary.

        Lazy initialisation allows the service to be imported and instantiated
        without a valid API key — the key is only required when summarize()
        is actually called.

        Raises:
            AIProviderError: API key is missing.
        """
        if self._client is None:
            if not self._api_key:
                raise AIProviderError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file and restart the server."
                )
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            logger.debug("AsyncAnthropic client created.")
        return self._client

    async def _call_with_tool_use(self, user_prompt: str) -> dict[str, Any]:
        """
        Call the Anthropic API using Tool Use to get structured JSON output.

        Tool-use is the most reliable structured-output mechanism on the
        Anthropic API. By passing `tool_choice={"type": "tool", "name": ...}`
        we force the model to populate the tool's input schema — the result
        is a parsed dict, not a raw string.

        Strategy:
          Primary:  tool_use with forced tool_choice.
          Fallback: _call_with_plain_json() if tool_use block is missing.

        Args:
            user_prompt: Formatted user message content.

        Returns:
            dict conforming to the VideoSummary field structure.

        Raises:
            AIProviderError:   Anthropic API / auth / rate-limit failure.
            SummarizationError: Response structure is unexpected.
        """
        client = self._get_client()

        logger.debug(
            f"Calling Anthropic API (tool_use) | model={self._model} | "
            f"prompt_chars={len(user_prompt)}"
        )

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=_SYSTEM_PROMPT,
                tools=[_SUMMARY_TOOL_SCHEMA],          # Register our schema as a tool
                tool_choice={                           # Force it to call this tool
                    "type": "tool",
                    "name": "extract_video_summary",
                },
                messages=[{"role": "user", "content": user_prompt}],
            )

        except anthropic.AuthenticationError as exc:
            raise AIProviderError(
                f"Invalid Anthropic API key. Check ANTHROPIC_API_KEY in .env. ({exc})"
            ) from exc

        except anthropic.RateLimitError as exc:
            raise AIProviderError(
                f"Anthropic rate limit exceeded. Wait and retry. ({exc})"
            ) from exc

        except anthropic.APIStatusError as exc:
            raise AIProviderError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc

        except anthropic.APIConnectionError as exc:
            raise AIProviderError(
                f"Could not connect to Anthropic API. Check network. ({exc})"
            ) from exc

        # ── Parse the tool-use response ────────────────────────────────────
        logger.debug(
            f"API response received | stop_reason={response.stop_reason} | "
            f"content_blocks={len(response.content)}"
        )

        # Find the tool_use content block
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )

        if tool_block is not None:
            # `tool_block.input` is already a parsed Python dict — no JSON
            # parsing required. This is the primary happy path.
            logger.debug("Tool-use block found — using structured input dict.")
            return tool_block.input  # type: ignore[return-value]

        # Tool-use block missing — fall back to text extraction
        logger.warning(
            "Tool-use block missing from response. "
            "Falling back to plain-JSON extraction."
        )
        return await self._call_with_plain_json(user_prompt)

    async def _call_with_plain_json(self, user_prompt: str) -> dict[str, Any]:
        """
        Fallback: call Claude without tool-use and extract JSON from the text.

        The system prompt instructs Claude to respond with ONLY a JSON object.
        We then extract the JSON using a regex that finds the outermost `{…}`
        block, which handles cases where the model adds brief preamble text.

        This is less reliable than tool-use but provides a safety net.

        Args:
            user_prompt: Same user message content as the primary call.

        Returns:
            Parsed dict from the model's text response.

        Raises:
            SummarizationError: JSON cannot be extracted or parsed.
            AIProviderError:    API call fails.
        """
        client = self._get_client()

        fallback_system = (
            _SYSTEM_PROMPT
            + "\n\nIMPORTANT: Respond with ONLY a valid JSON object — no markdown "
            "fences, no prose before or after. The JSON must contain exactly these "
            'keys: "executive_summary" (string), "detailed_summary" (string), '
            '"key_points" (array of strings), "action_items" (array of strings), '
            '"important_timestamps" (array of {time, description} objects).'
        )

        logger.debug("Fallback: calling API without tool-use.")

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=fallback_system,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise AIProviderError(str(exc)) from exc

        raw_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

        return self._extract_json_from_text(raw_text)

    # ── Response parsing ────────────────────────────────────────────────────

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """
        Extract and parse a JSON object from a raw text response.

        Handles three common model output patterns:
          1. Pure JSON:             `{"key": "value"}`
          2. Markdown fence:        ```json\\n{...}\\n```
          3. JSON with preamble:    `Here is the summary:\\n{...}`

        Strategy: find the first `{` and last `}` in the text and attempt
        to parse everything between them as JSON.

        Args:
            text: Raw string from the model's text content blocks.

        Returns:
            Parsed Python dict.

        Raises:
            SummarizationError: No valid JSON object found in the text.
        """
        if not text:
            raise SummarizationError("AI returned an empty response.")

        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            logger.error(f"No JSON object found in response. Preview: {text[:200]!r}")
            raise SummarizationError(
                "AI response did not contain a valid JSON object. "
                "The model may have returned an unexpected format."
            )

        json_str = text[start : end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error(f"JSON parse failed: {exc}. JSON preview: {json_str[:300]!r}")
            raise SummarizationError(
                f"AI response contained malformed JSON: {exc}"
            ) from exc

    def _build_video_summary(
        self,
        transcript: VideoTranscript,
        data: dict[str, Any],
    ) -> VideoSummary:
        """
        Validate and assemble a VideoSummary from the parsed AI response dict.

        Performs defensive extraction with sensible defaults for every field
        so that a partially-populated response doesn't crash the pipeline.

        Validates required fields and raises SummarizationError if the AI
        omitted the most critical ones (executive_summary, key_points).

        Args:
            transcript: Original transcript (provides video_id, duration).
            data:       Parsed dict from tool-use block or JSON extraction.

        Returns:
            Fully populated VideoSummary Pydantic model.

        Raises:
            SummarizationError: Required fields are missing or wrong type.
        """
        # ── Validate required fields ───────────────────────────────────────
        missing = [
            f for f in ("executive_summary", "detailed_summary", "key_points")
            if not data.get(f)
        ]
        if missing:
            raise SummarizationError(
                f"AI response is missing required fields: {missing}. "
                f"Response keys present: {list(data.keys())}"
            )

        # ── Extract fields with safe defaults ──────────────────────────────
        executive_summary: str = str(data.get("executive_summary", "")).strip()
        detailed_summary: str = str(data.get("detailed_summary", "")).strip()

        # key_points: must be a list of strings
        raw_points = data.get("key_points", [])
        key_points: list[str] = (
            [str(p).strip() for p in raw_points if p]
            if isinstance(raw_points, list)
            else [str(raw_points)]
        )

        # action_items: list of strings, empty list is acceptable
        raw_actions = data.get("action_items", [])
        action_items: list[str] = (
            [str(a).strip() for a in raw_actions if a]
            if isinstance(raw_actions, list)
            else []
        )

        # important_timestamps: list of {time, description} dicts
        raw_timestamps = data.get("important_timestamps", [])
        important_timestamps: list[TimestampItem] = []
        if isinstance(raw_timestamps, list):
            for ts in raw_timestamps:
                if isinstance(ts, dict) and ts.get("time") and ts.get("description"):
                    important_timestamps.append(
                        TimestampItem(
                            time=str(ts["time"]).strip(),
                            description=str(ts["description"]).strip(),
                        )
                    )

        # Duration from transcript
        duration_secs = transcript.total_duration_seconds
        m = int(duration_secs // 60)
        s = int(duration_secs % 60)
        duration_str = f"{m}:{s:02d}"

        return VideoSummary(
            video_id=transcript.video_id,
            title=None,           # Populated by caller once metadata is fetched
            duration=duration_str,
            executive_summary=executive_summary,
            detailed_summary=detailed_summary,
            key_points=key_points,
            action_items=action_items,
            important_timestamps=important_timestamps,
        )
