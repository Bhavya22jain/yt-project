"""
models/video.py
───────────────
Internal domain models — pure Python dataclasses / Pydantic models
used within service and repository layers.

These are NOT API schemas (those live in schemas/).
They represent the domain objects the application reasons about.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class TranscriptSegment:
    """
    A single segment from the YouTube transcript API.
    Each segment has text, a start time (seconds), and a duration.
    """
    text: str
    start: float       # seconds from video start
    duration: float    # seconds this segment lasts

    @property
    def start_formatted(self) -> str:
        """Return start time as MM:SS string."""
        minutes = int(self.start // 60)
        seconds = int(self.start % 60)
        return f"{minutes}:{seconds:02d}"


@dataclass
class VideoTranscript:
    """
    The full transcript for a YouTube video, plus metadata.
    Created by TranscriptService, consumed by SummaryService.
    """
    video_id: str
    language: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def full_text(self) -> str:
        """Concatenate all segments into a single string."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def total_duration_seconds(self) -> float:
        """Approximate total video duration from transcript."""
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.start + last.duration

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


@dataclass
class VideoMetadata:
    """
    Basic metadata about a YouTube video.
    Populated by the YouTube service (no transcript needed).
    """
    video_id: str
    title: Optional[str] = None
    channel: Optional[str] = None
    duration_seconds: Optional[int] = None
    thumbnail_url: Optional[str] = None
    published_at: Optional[str] = None

    @property
    def duration_formatted(self) -> Optional[str]:
        """Return duration as H:MM:SS or MM:SS."""
        if self.duration_seconds is None:
            return None
        h = self.duration_seconds // 3600
        m = (self.duration_seconds % 3600) // 60
        s = self.duration_seconds % 60
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


@dataclass
class ProcessedVideo:
    """
    Aggregates transcript + metadata into a single object
    passed into the summarisation pipeline.
    """
    transcript: VideoTranscript
    metadata: VideoMetadata

    @property
    def video_id(self) -> str:
        return self.transcript.video_id
