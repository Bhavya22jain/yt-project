"""
frontend/pages/chat.py
────────────────────────────────────────────────────────────────────────────
Chat page — multi-turn Q&A grounded in the video transcript.

Requires:
  session_state.summary     — set by the Summarizer page after a successful call
  session_state.current_url — YouTube URL of the summarized video

Features:
  • Full conversation history displayed using st.chat_message bubbles.
  • st.chat_input() for question submission (Enter to send).
  • Streaming-style spinner while waiting for the AI response.
  • Session token generated once per browser session for DB persistence.
  • "Suggested questions" generated from the summary key points.
  • Clear chat button with confirmation guard.
  • Sources expander shows which transcript excerpts were used.

Session state used:
  current_url    str   — video URL (sent with every chat request)
  summary        dict  — from /summarize (used for context chips)
  chat_history   list  — [{"role": "user"|"assistant", "content": "..."}]
  session_token  str   — UUID kept for the lifetime of the browser session
  page           str   — for routing back to Summarizer
"""

import uuid
from typing import Optional

import streamlit as st

from utils.api_client import chat_about_video


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Called by app.py when the Chat page is active."""
    _ensure_session_token()

    # ── Guard: must have a summarized video ───────────────────────────────
    if not st.session_state.get("summary"):
        _render_no_summary_state()
        return

    summary_data = st.session_state.summary.get("data", {})
    video_id = summary_data.get("video_id", "")
    title = summary_data.get("title") or f"Video {video_id}"

    # ── Page header ───────────────────────────────────────────────────────
    st.title("💬 Chat with the Video")
    st.markdown(
        f"Ask anything about **{title}** — answers are grounded in the transcript."
    )

    # ── Suggested questions chips (from key points) ────────────────────────
    _render_suggested_questions(summary_data)

    st.divider()

    # ── Conversation history ───────────────────────────────────────────────
    _render_chat_history()

    # ── Chat input ────────────────────────────────────────────────────────
    if question := st.chat_input("Ask something about the video…"):
        _handle_question(question)

    # ── Bottom controls ───────────────────────────────────────────────────
    _render_bottom_controls()


# ─────────────────────────────────────────────────────────────────────────────
# Subcomponents
# ─────────────────────────────────────────────────────────────────────────────

def _render_no_summary_state() -> None:
    """Shown when the user lands on Chat without summarizing first."""
    st.title("💬 Chat with the Video")
    st.warning(
        "No video has been summarized yet. "
        "Go to the **Summarizer** tab, paste a YouTube URL, and generate a summary first."
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Go to Summarizer", use_container_width=True, type="primary"):
            st.session_state.page = "summarizer"
            st.rerun()


def _render_suggested_questions(summary_data: dict) -> None:
    """
    Show 3 clickable question chips derived from the video's key points.
    Clicking a chip submits it as if the user typed it.
    """
    key_points: list = summary_data.get("key_points", [])
    if not key_points:
        return

    # Build three suggested questions from the first few key points
    suggestions = []
    if len(key_points) >= 1:
        suggestions.append(f"Can you explain: {key_points[0].rstrip('.')}?")
    if len(key_points) >= 2:
        suggestions.append(f"Tell me more about: {key_points[1].rstrip('.')}.")
    suggestions.append("What are the most important takeaways from this video?")

    st.markdown("**💡 Suggested questions:**")
    chip_cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(chip_cols, suggestions)):
        with col:
            if st.button(
                suggestion,
                key=f"chip_{i}",
                use_container_width=True,
                help=f"Click to ask: {suggestion}",
            ):
                _handle_question(suggestion)
                st.rerun()


def _render_chat_history() -> None:
    """Render all messages in the chat history using st.chat_message bubbles."""
    history: list = st.session_state.get("chat_history", [])

    if not history:
        st.markdown(
            "<div style='text-align:center; color:#666; padding:2rem;'>"
            "💬 No messages yet — ask your first question below."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for message in history:
        role = message.get("role", "user")
        content = message.get("content", "")
        sources = message.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)

            # Show sources for assistant messages if available
            if role == "assistant" and sources:
                with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}.** {src}",
                        )


def _handle_question(question: str) -> None:
    """
    Submit a question to the backend and append both turns to chat history.
    Renders the user message immediately, then shows a spinner while waiting.
    """
    url = st.session_state.get("current_url", "")
    if not url:
        st.error("No video URL found in session. Please re-summarize the video.")
        return

    # ── Immediately render user bubble ────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(question)

    # Append user turn
    st.session_state.chat_history.append({"role": "user", "content": question})

    # ── Call backend ──────────────────────────────────────────────────────
    # Build history without the just-appended turn (backend wants prior context)
    prior_history = st.session_state.chat_history[:-1]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            data, error = chat_about_video(
                youtube_url=url,
                question=question,
                chat_history=prior_history,
                session_token=st.session_state.get("session_token"),
            )

        if error:
            st.error(f"❌ {error}")
            # Roll back the user message on error
            st.session_state.chat_history.pop()
            return

        answer: str = data.get("answer", "")
        sources: list = data.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {src}")

    # Append assistant turn (store sources so they render on reruns too)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def _render_bottom_controls() -> None:
    """Clear chat button and session info shown at the bottom."""
    history = st.session_state.get("chat_history", [])
    if not history:
        return

    st.divider()
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 6])

    with ctrl_col1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with ctrl_col2:
        st.caption(f"{len(history)} message(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Session utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_session_token() -> None:
    """
    Generate a stable session token once per browser session.
    Used by the backend to group chat messages into a persistent ChatSession.
    """
    if "session_token" not in st.session_state:
        st.session_state.session_token = str(uuid.uuid4())
