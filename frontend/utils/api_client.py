"""
frontend/utils/api_client.py
────────────────────────────────────────────────────────────────────────────
HTTP client for the FastAPI backend.

All network calls go through this module. Pages and components never
import httpx or requests directly — they call functions here.

Design:
  • Every function returns a typed Result tuple: (data | None, error_str | None)
    so callers can handle errors without try/except boilerplate.
  • Timeouts are enforced on every call (default from cfg.REQUEST_TIMEOUT).
  • The error string is always human-readable and safe to display in the UI.
  • The health_check() function is used by the sidebar to show backend status.

Response shape assumptions (matches backend schemas/video.py):
  POST /api/v1/summarize  →  { success, cached, processing_ms, data: VideoSummary }
  POST /api/v1/chat       →  { success, answer, sources, session_token }
  GET  /api/v1/videos     →  { success, total, skip, limit, items: [VideoRecord] }
  GET  /api/v1/health     →  { status, version, environment, database }
  Error responses         →  { success: false, error, code, request_id }

Usage:
    from utils.api_client import summarize_video, chat_about_video

    data, err = summarize_video(url)
    if err:
        st.error(err)
    else:
        st.write(data["executive_summary"])
"""

from typing import Any, Optional, Tuple

import httpx

from utils.config import cfg

# Convenience type alias
Result = Tuple[Optional[dict], Optional[str]]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _error_message(response: httpx.Response) -> str:
    """
    Extract a user-friendly error string from any non-2xx response.

    Handles three body shapes:
      1. Our structured ErrorResponse:   { "error": "...", "code": "..." }
      2. FastAPI HTTPException detail:   { "detail": "..." } or { "detail": { "error": "..." } }
      3. Unstructured / non-JSON:        raw status line
    """
    try:
        body = response.json()
    except Exception:
        return f"Backend returned HTTP {response.status_code}."

    # Shape 1: our ErrorResponse envelope
    if isinstance(body.get("error"), str):
        msg = body["error"]
        code = body.get("code")
        return f"{msg} [{code}]" if code else msg

    # Shape 2a: FastAPI's default HTTPException with dict detail
    detail = body.get("detail")
    if isinstance(detail, dict):
        if isinstance(detail.get("error"), str):
            return detail["error"]
        return str(detail)

    # Shape 2b: FastAPI default string detail
    if isinstance(detail, str):
        return detail

    return f"HTTP {response.status_code}: unexpected response format."


def _friendly_connection_error(url: str) -> str:
    return (
        "Cannot reach the backend server.\n\n"
        f"Make sure FastAPI is running at **{cfg.BACKEND_URL}**:\n"
        "```\nuvicorn app.main:app --reload --port 8000\n```"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def summarize_video(
    youtube_url: str,
    force_refresh: bool = False,
    language_preference: Optional[list[str]] = None,
) -> Result:
    """
    Call POST /api/v1/summarize.

    Args:
        youtube_url:         Full YouTube URL.
        force_refresh:       Bypass cache and re-summarize.
        language_preference: Preferred transcript language codes.

    Returns:
        (summary_dict, None)  on success — summary_dict is data: VideoSummary
        (None, error_str)     on any failure

    Example:
        data, err = summarize_video("https://youtube.com/watch?v=dQw4w9WgXcQ")
        # data keys: video_id, executive_summary, detailed_summary,
        #            key_points, action_items, important_timestamps
    """
    payload: dict[str, Any] = {
        "youtube_url": youtube_url,
        "force_refresh": force_refresh,
        "language_preference": language_preference or [],
    }

    try:
        response = httpx.post(
            cfg.SUMMARIZE_ENDPOINT,
            json=payload,
            timeout=cfg.REQUEST_TIMEOUT,
        )
    except httpx.ConnectError:
        return None, _friendly_connection_error(cfg.SUMMARIZE_ENDPOINT)
    except httpx.TimeoutException:
        return None, (
            f"Request timed out after {cfg.REQUEST_TIMEOUT}s. "
            "The video may be very long — try again or use a shorter video."
        )
    except httpx.RequestError as exc:
        return None, f"Network error: {exc}"

    if response.status_code == 200:
        body = response.json()
        # Return the full envelope so callers can check body["cached"]
        return body, None

    return None, _error_message(response)


def chat_about_video(
    youtube_url: str,
    question: str,
    chat_history: Optional[list[dict]] = None,
    session_token: Optional[str] = None,
) -> Result:
    """
    Call POST /api/v1/chat.

    Args:
        youtube_url:   YouTube URL (identifies which video to chat about).
        question:      User's current question.
        chat_history:  Previous turns [{"role": "user"|"assistant", "content": "..."}].
        session_token: UUID for persistent chat sessions.

    Returns:
        (chat_response_dict, None) on success
        (None, error_str)          on failure

    Example:
        data, err = chat_about_video(url, "What is the main topic?")
        # data keys: answer, sources, session_token
    """
    payload: dict[str, Any] = {
        "youtube_url": youtube_url,
        "question": question,
        "chat_history": chat_history or [],
    }
    if session_token:
        payload["session_token"] = session_token

    try:
        response = httpx.post(
            cfg.CHAT_ENDPOINT,
            json=payload,
            timeout=cfg.REQUEST_TIMEOUT,
        )
    except httpx.ConnectError:
        return None, _friendly_connection_error(cfg.CHAT_ENDPOINT)
    except httpx.TimeoutException:
        return None, f"Chat request timed out after {cfg.REQUEST_TIMEOUT}s."
    except httpx.RequestError as exc:
        return None, f"Network error: {exc}"

    if response.status_code == 200:
        return response.json(), None

    return None, _error_message(response)


def get_video(video_id: str) -> Result:
    """
    Call GET /api/v1/videos/{video_id}.

    Used to fetch the full transcript text for display, since the
    /summarize response does not include it (only the AI summary).

    Returns:
        (video_record_dict, None) on success — includes 'transcript' key
        (None, error_str)         on failure (e.g. 404 if not summarized yet)
    """
    try:
        response = httpx.get(
            f"{cfg.VIDEOS_ENDPOINT}/{video_id}",
            timeout=30,
        )
    except httpx.ConnectError:
        return None, _friendly_connection_error(cfg.VIDEOS_ENDPOINT)
    except httpx.RequestError as exc:
        return None, f"Network error: {exc}"

    if response.status_code == 200:
        return response.json().get("data"), None

    return None, _error_message(response)


def get_video_history(
    skip: int = 0,
    limit: int = 20,
) -> Result:
    """
    Call GET /api/v1/videos to fetch previously summarized videos.

    Returns:
        (list_response_dict, None) — dict has keys: total, skip, limit, items
        (None, error_str)          on failure
    """
    try:
        response = httpx.get(
            cfg.VIDEOS_ENDPOINT,
            params={"skip": skip, "limit": limit, "processed_only": True},
            timeout=30,
        )
    except httpx.ConnectError:
        return None, _friendly_connection_error(cfg.VIDEOS_ENDPOINT)
    except httpx.RequestError as exc:
        return None, f"Network error: {exc}"

    if response.status_code == 200:
        return response.json(), None

    return None, _error_message(response)


def delete_video(video_id: str) -> Result:
    """
    Call DELETE /api/v1/videos/{video_id}.

    Returns:
        ({"success": True, "message": "..."}, None) on success
        (None, error_str)                           on failure
    """
    try:
        response = httpx.delete(
            f"{cfg.VIDEOS_ENDPOINT}/{video_id}",
            timeout=30,
        )
    except httpx.ConnectError:
        return None, _friendly_connection_error(cfg.VIDEOS_ENDPOINT)
    except httpx.RequestError as exc:
        return None, f"Network error: {exc}"

    if response.status_code == 200:
        return response.json(), None

    return None, _error_message(response)


def health_check() -> Tuple[bool, str]:
    """
    Ping GET /api/v1/health.

    Returns:
        (True,  "connected")   if the backend is reachable and DB is up
        (False, reason_str)    otherwise
    """
    try:
        response = httpx.get(cfg.HEALTH_ENDPOINT, timeout=5)
    except httpx.ConnectError:
        return False, f"Backend offline — is FastAPI running at {cfg.BACKEND_URL}?"
    except httpx.TimeoutException:
        return False, "Health check timed out."
    except httpx.RequestError as exc:
        return False, f"Network error: {exc}"

    if response.status_code == 200:
        body = response.json()
        db_status = body.get("database", "unknown")
        if db_status == "connected":
            return True, "connected"
        return False, f"Backend up but database is {db_status}"

    return False, f"Backend returned HTTP {response.status_code}"
