"""
frontend/pages/summarizer.py
────────────────────────────────────────────────────────────────────────────
Summarizer page — the main user-facing workflow.

Flow:
  1. User pastes a YouTube URL and clicks "Generate Summary".
  2. A loading spinner shows while the backend pipeline runs
     (transcript extraction → AI summarization → DB persistence).
  3. Results are displayed in five labelled sections:
       Executive Summary · Detailed Summary · Key Points ·
       Action Items · Important Timestamps
  4. Metadata strip shows cache status, processing time, word count.
  5. "Chat about this video" button jumps to the Chat page.
  6. Recent history sidebar lets users reload past summaries instantly.

Session state used:
  current_url   str         — last URL the user submitted
  summary       dict|None   — full API response body from /summarize
  page          str         — used by app.py for routing

This module contains NO network code — all HTTP calls go through
utils/api_client.py.
"""

import re
import time
from typing import Optional

import streamlit as st

from utils.api_client import (
    delete_video,
    get_video,
    get_video_history,
    health_check,
    summarize_video,
)
from utils.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_YT_PATTERN = re.compile(
    r"(?:https?://)?(?:(?:www|m)\.)?(?:"
    r"youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/)|youtu\.be/"
    r")([a-zA-Z0-9_-]{11})"
)

_YT_THUMBNAIL = "https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Called by app.py when the Summarizer page is active."""
    _render_page_header()
    _render_url_input_section()

    if st.session_state.summary:
        _render_summary_results(st.session_state.summary)


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

def _render_page_header() -> None:
    st.title("🎬 YouTube Video Summarizer")
    st.markdown(
        "Paste any YouTube URL to get an AI-generated structured summary — "
        "executive overview, key points, action items, and important timestamps."
    )
    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# URL input section
# ─────────────────────────────────────────────────────────────────────────────

def _render_url_input_section() -> None:
    """URL text input + Generate button + advanced options expander."""

    # ── URL row ───────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        url = st.text_input(
            label="YouTube URL",
            value=st.session_state.get("current_url", ""),
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="url_input",
        )

    with col_btn:
        submitted = st.button(
            "▶ Summarize",
            type="primary",
            use_container_width=True,
        )

    # ── Advanced options ──────────────────────────────────────────────────
    with st.expander("⚙️ Options", expanded=False):
        force_refresh = st.checkbox(
            "Force refresh",
            value=False,
            help="Re-fetch transcript and regenerate summary even if cached",
        )
        lang_input = st.text_input(
            "Preferred transcript languages (comma-separated)",
            value="en",
            help="BCP-47 codes in priority order, e.g. 'en, es, fr'",
        )
        language_pref = [l.strip() for l in lang_input.split(",") if l.strip()]
    
    # ── Inline URL validation hint ────────────────────────────────────────
    if url and not _YT_PATTERN.search(url):
        st.warning(
            "⚠️ That doesn't look like a YouTube URL. "
            "Try: `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`"
        )
        return

    # ── Trigger ───────────────────────────────────────────────────────────
    if submitted:
        if not url.strip():
            st.error("Please enter a YouTube URL.")
            return

        st.session_state.current_url = url.strip()
        # Clear previous summary so the spinner renders cleanly
        st.session_state.summary = None
        st.rerun()

    # ── Detect a pending URL (set by rerun above) and run the pipeline ────
    if (
        st.session_state.get("current_url")
        and st.session_state.summary is None
        and submitted is False   # don't re-trigger on every rerun
    ):
        _run_summarize_pipeline(
            st.session_state.current_url,
            force_refresh=force_refresh,
            language_pref=language_pref,
        )

    # ── Always run when button pressed ────────────────────────────────────
    if submitted and url.strip() and _YT_PATTERN.search(url):
        _run_summarize_pipeline(
            url.strip(),
            force_refresh=force_refresh,
            language_pref=language_pref,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_summarize_pipeline(
    url: str,
    force_refresh: bool = False,
    language_pref: Optional[list] = None,
) -> None:
    """
    Call the backend, show a progress spinner, store the result in session
    state, and trigger a rerun to render results cleanly.
    """
    video_id = _extract_video_id(url)

    # ── Progress display ──────────────────────────────────────────────────
    progress_placeholder = st.empty()

    with progress_placeholder.container():
        # Show thumbnail while processing (if we can derive the video ID)
        if video_id:
            thumb_col, info_col = st.columns([1, 3])
            with thumb_col:
                st.image(
                    _YT_THUMBNAIL.format(video_id=video_id),
                    use_container_width=True,
                )
            with info_col:
                st.markdown(f"**Video ID:** `{video_id}`")
                st.markdown(f"**URL:** {url}")

        steps = st.status(
            "Processing video…",
            expanded=True,
            state="running",
        )
        with steps:
            st.write("📡 Validating URL and extracting video ID…")
            t0 = time.monotonic()

    # ── API call ──────────────────────────────────────────────────────────
    result, error = summarize_video(
        youtube_url=url,
        force_refresh=force_refresh,
        language_preference=language_pref,
    )
    elapsed = time.monotonic() - t0
    progress_placeholder.empty()

    # ── Handle error ──────────────────────────────────────────────────────
    if error:
        _render_error(error, url)
        return

    # ── Store and render ──────────────────────────────────────────────────
    st.session_state.summary = result
    st.session_state.current_url = url

    # Show a brief success banner with metadata
    cached = result.get("cached", False)
    processing_ms = result.get("processing_ms", 0)
    if cached:
        st.success(f"⚡ Loaded from cache in {processing_ms}ms")
    else:
        st.success(
            f"✅ Summary generated in {processing_ms / 1000:.1f}s "
            f"(~{elapsed:.1f}s total)"
        )


def _render_error(error: str, url: str) -> None:
    """Show a contextual error box with actionable guidance."""
    # Pick the right icon and heading based on common error patterns
    if "TRANSCRIPT_NOT_AVAILABLE" in error or "transcript" in error.lower():
        st.error(
            "### 📭 No Transcript Available\n\n"
            f"{error}\n\n"
            "**Try:** Look for a version of this video with auto-generated captions, "
            "or check if the video has manual subtitles enabled."
        )
    elif "INVALID_URL" in error or "recognised YouTube URL" in error:
        st.error(
            f"### 🔗 Invalid URL\n\n{error}\n\n"
            "**Try:** Copy the URL directly from your browser's address bar."
        )
    elif "AI_PROVIDER" in error or "rate limit" in error.lower():
        st.error(
            f"### 🤖 AI Provider Error\n\n{error}\n\n"
            "The AI service is temporarily unavailable. Please try again in a moment."
        )
    elif "offline" in error.lower() or "Cannot reach" in error:
        st.error(error)
        st.code("uvicorn app.main:app --reload --port 8000", language="bash")
    else:
        st.error(f"### ❌ Error\n\n{error}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary result rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary_results(response: dict) -> None:
    """
    Render the full summary from the API response envelope.

    The `response` dict is the full SummarizeResponse body:
      { success, cached, processing_ms, data: VideoSummary }
    """
    data: dict = response.get("data", {})
    if not data:
        st.warning("Received an empty summary. Try regenerating.")
        return

    video_id = data.get("video_id", "")
    title = data.get("title")
    duration = data.get("duration")
    cached = response.get("cached", False)

    # ── Header row — thumbnail + title + meta ─────────────────────────────
    st.divider()
    if video_id:
        img_col, meta_col = st.columns([1, 3])
        with img_col:
            st.image(
                _YT_THUMBNAIL.format(video_id=video_id),
                use_container_width=True,
            )
        with meta_col:
            if title:
                st.subheader(title)
            st.markdown(f"🔗 `{st.session_state.current_url}`")

            badge_cols = st.columns(4)
            with badge_cols[0]:
                st.metric("Video ID", video_id)
            with badge_cols[1]:
                st.metric("Duration", duration or "—")
            with badge_cols[2]:
                st.metric("Source", "Cache ⚡" if cached else "Fresh ✨")
            with badge_cols[3]:
                processing_ms = response.get("processing_ms", 0)
                st.metric("Time", f"{processing_ms}ms")
    else:
        st.subheader(title or "Video Summary")

    st.divider()

    # ── Action buttons ─────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 6])
    with btn_col1:
        if st.button("💬 Chat about this video", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
    with btn_col2:
        if st.button("🔄 Re-summarize", use_container_width=True):
            st.session_state.summary = None
            # Trigger fresh summarization
            _run_summarize_pipeline(
                st.session_state.current_url,
                force_refresh=True,
            )
            return

    st.divider()

    # ── Six content sections ─────────────────────────────────────────────
    _render_executive_summary(data)
    _render_detailed_summary(data)
    _render_key_points(data)
    _render_action_items(data)
    _render_timestamps(data)
    _render_transcript(data)

    # ── Copy-friendly text export ──────────────────────────────────────────
    with st.expander("📋 Export plain text", expanded=False):
        _render_text_export(data)


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_executive_summary(data: dict) -> None:
    """Section 1: 2–3 sentence TL;DR."""
    text = data.get("executive_summary", "").strip()
    if not text:
        return

    st.markdown("### 📌 Executive Summary")
    st.info(text)
    st.markdown("")


def _render_detailed_summary(data: dict) -> None:
    """Section 2: Full paragraph breakdown."""
    text = data.get("detailed_summary", "").strip()
    if not text:
        return

    st.markdown("### 📄 Detailed Summary")
    # Render paragraphs separated by blank lines
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for para in paragraphs:
        st.markdown(para)
    st.markdown("")


def _render_key_points(data: dict) -> None:
    """Section 3: Bullet-point key takeaways in two columns."""
    points: list = data.get("key_points", [])
    if not points:
        return

    st.markdown("### 💡 Key Points")

    # Split into two columns for readability
    mid = (len(points) + 1) // 2
    left_points = points[:mid]
    right_points = points[mid:]

    col_l, col_r = st.columns(2)
    with col_l:
        for point in left_points:
            st.markdown(f"- {point}")
    with col_r:
        for point in right_points:
            st.markdown(f"- {point}")

    st.markdown("")


def _render_action_items(data: dict) -> None:
    """Section 4: Actionable next steps as interactive checklist."""
    actions: list = data.get("action_items", [])
    if not actions:
        return

    st.markdown("### ✅ Action Items")

    for i, action in enumerate(actions):
        # Use a checkbox widget — purely cosmetic (state not persisted)
        st.checkbox(action, key=f"action_{i}_{hash(action)}", value=False)

    st.markdown("")


def _render_timestamps(data: dict) -> None:
    """Section 5: Important timestamps as a clickable table."""
    timestamps: list = data.get("important_timestamps", [])
    if not timestamps:
        return

    st.markdown("### ⏱ Important Timestamps")

    video_id = data.get("video_id", "")

    for ts in timestamps:
        time_str = ts.get("time", "")
        description = ts.get("description", "")
        if not time_str or not description:
            continue

        col_time, col_desc = st.columns([1, 5])
        with col_time:
            # Build a clickable YouTube deep-link to the timestamp
            if video_id and time_str:
                seconds = _timestamp_to_seconds(time_str)
                yt_link = f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
                st.markdown(f"[**`{time_str}`**]({yt_link})")
            else:
                st.markdown(f"**`{time_str}`**")
        with col_desc:
            st.markdown(description)

    st.markdown("")


def _render_transcript(data: dict) -> None:
    """
    Section 6: Full video transcript, fetched lazily on demand.

    The /summarize response does not include the transcript (only the AI
    summary), so we fetch it via GET /videos/{video_id} only when the user
    actually opens the expander — avoids an extra API call on every render.
    """
    video_id = data.get("video_id", "")
    if not video_id:
        return

    st.markdown("### 📜 Full Transcript")

    with st.expander("Show transcript", expanded=False):
        cache_key = f"_transcript_cache_{video_id}"

        if cache_key not in st.session_state:
            with st.spinner("Loading transcript…"):
                record, error = get_video(video_id)
            if error:
                st.error(f"Could not load transcript: {error}")
                return
            st.session_state[cache_key] = record.get("transcript") if record else None

        transcript_text = st.session_state.get(cache_key)

        if not transcript_text:
            st.caption("No transcript text stored for this video.")
            return

        word_count = len(transcript_text.split())
        st.caption(f"~{word_count:,} words")
        st.text_area(
            label="Transcript",
            value=transcript_text,
            height=300,
            label_visibility="collapsed",
            disabled=True,
        )

    st.markdown("")


def _render_text_export(data: dict) -> None:
    """Render a copyable plain-text version of the summary."""
    title = data.get("title") or f"Video {data.get('video_id', '')}"
    lines = [
        f"# {title}",
        f"URL: {st.session_state.get('current_url', '')}",
        "",
        "## Executive Summary",
        data.get("executive_summary", ""),
        "",
        "## Detailed Summary",
        data.get("detailed_summary", ""),
        "",
        "## Key Points",
    ]
    for p in data.get("key_points", []):
        lines.append(f"- {p}")
    lines += ["", "## Action Items"]
    for a in data.get("action_items", []):
        lines.append(f"- [ ] {a}")
    lines += ["", "## Important Timestamps"]
    for ts in data.get("important_timestamps", []):
        lines.append(f"- {ts.get('time', '')} — {ts.get('description', '')}")

    st.code("\n".join(lines), language="markdown")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_video_id(url: str) -> Optional[str]:
    m = _YT_PATTERN.search(url)
    return m.group(1) if m else None


def _timestamp_to_seconds(time_str: str) -> int:
    """
    Convert "M:SS" or "H:MM:SS" to total seconds for YouTube deep-links.

    Examples:
        "4:32"    → 272
        "1:04:15" → 3855
        "0:30"    → 30
    """
    try:
        parts = [int(p) for p in time_str.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (ValueError, IndexError):
        pass
    return 0
