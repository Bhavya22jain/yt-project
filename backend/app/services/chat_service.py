"""
services/chat_service.py
─────────────────────────
Responsible for:
  1. Accepting a user question + chat history + transcript
  2. Calling Claude with the transcript as context
  3. Returning a grounded answer

Day 1: Interface + stub only. No business logic yet.
"""

from typing import List

from loguru import logger

from app.core.config import settings
from app.models.video import VideoTranscript
from app.schemas.video import ChatResponse


class ChatService:
    """
    Answers user questions about a YouTube video using the transcript as context.

    Usage:
        service = ChatService()
        response = await service.chat(transcript, question, history)
    """

    def __init__(self) -> None:
        self._model = settings.anthropic_model
        self._max_tokens = settings.anthropic_max_tokens
        logger.debug(f"ChatService initialised | model={self._model}")

    async def chat(
        self,
        transcript: VideoTranscript,
        question: str,
        chat_history: List[dict],
    ) -> ChatResponse:
        """
        Answer a question about the video using the transcript as context.

        Args:
            transcript:   Full VideoTranscript object.
            question:     The user's current question.
            chat_history: List of previous turns (role/content dicts).

        Returns:
            ChatResponse with answer and optional source excerpts.

        Raises:
            ChatError: AI response is empty or malformed.
            AIProviderError: Anthropic API call failed.

        TODO (Day 4): Implement context window management, prompt, API call.
        """
        raise NotImplementedError("ChatService.chat — Day 4")

    def _build_system_prompt(self, transcript_text: str) -> str:
        """
        Build the system prompt that grounds the AI to the transcript.

        TODO (Day 4): Implement.
        """
        raise NotImplementedError("ChatService._build_system_prompt — Day 4")

    def _trim_transcript_to_context(self, transcript_text: str) -> str:
        """
        Trim the transcript to fit within the model's context window,
        preserving the most relevant portions.

        TODO (Day 4): Implement smart truncation.
        """
        raise NotImplementedError("ChatService._trim_transcript_to_context — Day 4")
