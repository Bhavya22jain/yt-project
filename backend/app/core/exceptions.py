"""
core/exceptions.py
──────────────────
Custom exception hierarchy for the application.
Raising a typed exception carries the HTTP status and a human-readable
message all the way to the global error handler in main.py.
"""


class AppBaseException(Exception):
    """Root exception for all application errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ── YouTube / Transcript ──────────────────────────────────────

class InvalidYouTubeURLError(AppBaseException):
    """Raised when the provided URL is not a valid YouTube URL."""

    def __init__(self, url: str) -> None:
        super().__init__(
            message=f"Invalid YouTube URL: '{url}'",
            status_code=422,
        )


class TranscriptNotAvailableError(AppBaseException):
    """Raised when no transcript exists for the requested video."""

    def __init__(self, video_id: str) -> None:
        super().__init__(
            message=f"No transcript available for video '{video_id}'. "
                    "The video may have captions disabled.",
            status_code=404,
        )


class TranscriptFetchError(AppBaseException):
    """Raised when the transcript API call fails unexpectedly."""

    def __init__(self, video_id: str, reason: str = "") -> None:
        super().__init__(
            message=f"Failed to fetch transcript for '{video_id}'. {reason}".strip(),
            status_code=502,
        )


# ── AI / Summarisation ────────────────────────────────────────

class SummarizationError(AppBaseException):
    """Raised when the AI model fails to produce a summary."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message=f"Summarization failed. {reason}".strip(),
            status_code=502,
        )


class AIProviderError(AppBaseException):
    """Raised on Anthropic API errors (auth, rate-limit, server error)."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message=f"AI provider error. {reason}".strip(),
            status_code=503,
        )


# ── Chat ──────────────────────────────────────────────────────

class ChatError(AppBaseException):
    """Raised when a chat response cannot be generated."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(
            message=f"Chat error. {reason}".strip(),
            status_code=502,
        )


# ── Validation ────────────────────────────────────────────────

class ValidationError(AppBaseException):
    """Raised when incoming request data fails business-level validation."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            status_code=422,
        )
