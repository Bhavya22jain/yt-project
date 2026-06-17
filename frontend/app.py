"""
frontend/app.py
────────────────────────────────────────────────────────────────────────────
Streamlit application entry point — composition root of the frontend.

Responsibilities:
  • Page config (title, icon, layout) — must be the first Streamlit call.
  • Session state initialisation with sensible defaults.
  • Sidebar: branding, navigation, backend health indicator, history panel.
  • Top-level routing: delegates rendering to pages/summarizer.py or pages/chat.py.

Architecture:
  • No business logic, API calls, or rendering of summary content here.
  • All HTTP calls live in utils/api_client.py.
  • All page content lives in pages/*.py.
  • Config lives in utils/config.py.

Run with:
    streamlit run frontend/app.py
    # or from the project root:
    streamlit run frontend/app.py --server.port 8501
"""

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Add the frontend directory to sys.path so that `from utils.x import y`
# and `from pages.x import y` resolve correctly regardless of the working
# directory Streamlit is launched from.
_FRONTEND_DIR = Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

# ── Page config — MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="AI YouTube Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourname/yt-summarizer",
        "Report a bug": "https://github.com/yourname/yt-summarizer/issues",
        "About": "AI YouTube Summarizer — powered by Claude & FastAPI",
    },
)

# ── Imports (after path setup) ────────────────────────────────────────────────
from utils.api_client import get_video_history, delete_video, health_check
from utils.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Session state — initialise once, persist across reruns
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "page": "summarizer",           # active page: "summarizer" | "chat"
    "current_url": "",              # last YouTube URL submitted
    "summary": None,                # full SummarizeResponse body dict | None
    "chat_history": [],             # [{"role": "...", "content": "...", "sources": [...]}]
    "session_token": None,          # UUID for chat DB persistence
    "history_loaded": False,        # flag: have we fetched video history yet?
}

for _key, _default in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    # ── Branding ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding: 0.5rem 0 1rem;'>
            <div style='font-size:2.5rem;'>🎬</div>
            <div style='font-size:1.2rem; font-weight:700; line-height:1.2;'>
                AI YouTube<br>Summarizer
            </div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.25rem;'>
                Powered by Claude AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────
    nav_options = ["📋 Summarizer", "💬 Chat"]
    current_index = 0 if st.session_state.page == "summarizer" else 1

    selected = st.radio(
        "Navigation",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed",
    )
    new_page = "summarizer" if selected == "📋 Summarizer" else "chat"

    if new_page != st.session_state.page:
        # Clear chat state when switching away from chat to keep things clean
        if new_page == "summarizer":
            st.session_state.chat_history = []
        st.session_state.page = new_page
        st.rerun()

    st.divider()

    # ── Backend health indicator ───────────────────────────────────────────
    st.markdown("**Backend Status**")
    is_healthy, status_msg = health_check()

    if is_healthy:
        st.success(f"✅ {status_msg}", icon=None)
    else:
        st.error(f"❌ Offline")
        with st.expander("Details"):
            st.caption(status_msg)
            st.code(
                "uvicorn app.main:app --reload --port 8000",
                language="bash",
            )

    st.divider()

    # ── Recent video history ───────────────────────────────────────────────
    st.markdown("**Recent Videos**")

    if st.button("🔄 Refresh history", use_container_width=True):
        st.session_state.history_loaded = False

    if not st.session_state.history_loaded:
        with st.spinner("Loading…"):
            history_data, history_err = get_video_history(limit=8)
        st.session_state.history_loaded = True

        if history_err or not history_data:
            st.caption("History unavailable" if not is_healthy else "No videos yet")
            history_items = []
        else:
            history_items = history_data.get("items", [])
            st.session_state["_history_items"] = history_items
    else:
        history_items = st.session_state.get("_history_items", [])

    if history_items:
        for item in history_items:
            vid_url = item.get("youtube_url", "")
            vid_title = (
                item.get("title")
                or item.get("video_id", "Unknown")
            )
            summarize_count = item.get("summarize_count", 1)

            # Truncate long titles for the sidebar
            display_title = vid_title[:32] + "…" if len(vid_title) > 32 else vid_title

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    f"▶ {display_title}",
                    key=f"hist_{item.get('video_id')}",
                    use_container_width=True,
                    help=vid_url,
                ):
                    # Load this video into the summarizer
                    st.session_state.current_url = vid_url
                    st.session_state.page = "summarizer"
                    # Trigger a fresh load from cache
                    st.session_state.summary = None
                    st.session_state.history_loaded = False
                    st.rerun()

            with col_del:
                vid_id = item.get("video_id", "")
                if st.button(
                    "🗑",
                    key=f"del_{vid_id}",
                    help=f"Delete '{display_title}'",
                ):
                    _, del_err = delete_video(vid_id)
                    if del_err:
                        st.error(del_err)
                    else:
                        # Clear current summary if we just deleted it
                        if st.session_state.current_url == vid_url:
                            st.session_state.summary = None
                            st.session_state.current_url = ""
                        st.session_state.history_loaded = False
                        st.rerun()
    else:
        st.caption("No summarized videos yet.")

    st.divider()

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:0.7rem; color:#999; text-align:center;'>"
        f"v{cfg.APP_VERSION} · "
        f"<a href='{cfg.BACKEND_URL}/docs' target='_blank'>API Docs</a>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.page == "summarizer":
    from pages.summarizer import render
    render()
else:
    from pages.chat import render
    render()
