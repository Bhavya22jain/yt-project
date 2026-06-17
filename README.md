# 🎬 AI YouTube Video Summarizer & Chat Assistant

> Paste a YouTube URL. Get a structured summary in seconds. Then *chat* with the video.

Powered by **Claude AI** (Anthropic) · Built with **FastAPI** + **Streamlit**

---

## ✨ Features

| Feature | Description |
|---|---|
| **Executive Summary** | 2–3 sentence TL;DR of the entire video |
| **Detailed Summary** | Paragraph-level breakdown of the content |
| **Key Points** | Bullet-point takeaways |
| **Action Items** | Actionable next steps extracted from the video |
| **Timestamps** | Important moments with their timecodes |
| **Chat Interface** | Ask any question — the AI answers from the transcript |

---

## 🏗 Architecture

```
yt-summarizer/
├── backend/                  # FastAPI REST API
│   └── app/
│       ├── api/v1/endpoints/ # HTTP layer (thin, no business logic)
│       ├── core/             # Config, logging, exceptions
│       ├── models/           # Internal domain objects
│       ├── schemas/          # Pydantic request/response contracts
│       ├── services/         # Business logic (transcript, summary, chat)
│       ├── repositories/     # Data access (cache, storage)
│       └── utils/            # Shared helpers
│
├── frontend/                 # Streamlit UI
│   ├── app.py                # Entry point + navigation
│   ├── pages/                # Summarizer + Chat pages
│   ├── components/           # Reusable UI widgets
│   └── utils/                # API client, formatters
│
├── docs/                     # Architecture diagrams, API docs
├── scripts/                  # Setup and utility scripts
├── requirements.txt
├── .env.example
└── README.md
```

### Clean Architecture Principles

- **Separation of concerns** — each layer has one job
- **Dependency direction** — API → Services → Models (never reverse)
- **Schemas ≠ Models** — API contracts are separate from domain objects
- **No logic in routers** — validate, delegate, return
- **Config as singleton** — `settings` object, never raw `os.environ`

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourname/yt-summarizer.git
cd yt-summarizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Run the frontend

```bash
cd frontend
streamlit run app.py
```

Open: http://localhost:8501

---

## 📅 Development Status

| Day | Focus | Status |
|---|---|---|
| **Day 1** | Project foundation, architecture, folder structure | ✅ Done |
| **Day 2** | Database layer — SQLAlchemy ORM, 4 tables, full CRUD | ✅ Done (52 tests) |
| **Day 3** | Transcript service — URL parsing + transcript fetching | ✅ Done (99 tests) |
| **Day 4** | Summary service — Claude tool-use structured output | ✅ Done (80 tests) |
| **Day 5** | FastAPI endpoints — full pipeline, caching, error handling | ✅ Done (118 tests) |
| **Day 6** | Streamlit UI — full frontend integration with backend | ✅ Code complete |
| **Day 7** | Real end-to-end run, deployment | ⬜ Not started |

**361 automated tests passing.** Every AI call in the test suite is mocked —
no test has ever made a real call to the Anthropic API.

---

## ⚠️ Known Limitations

Read this before assuming the project is "done." These are real gaps, not
hypothetical ones:

1. **Never run end-to-end with a real API key.** No `.env` with a live
   `ANTHROPIC_API_KEY` has been used in this repository. The summarization
   and chat services have only been exercised against mocked responses.
   Before relying on this for a demo or resume claim, run it once for real:
   paste a YouTube URL, watch the transcript get fetched, watch Claude
   generate a summary, confirm it lands in `videos.db`.

2. **Frontend has zero automated tests.** All 361 tests are backend-only.
   The Streamlit app has been syntax-checked (`ast.parse`) but never
   actually launched and clicked through by a human.

3. **AI provider is Anthropic Claude, not OpenAI.** If your assignment or
   rubric specifically requires the OpenAI API, this codebase does not
   satisfy that — swapping providers is a deliberate follow-up task, not
   a trivial config change, since the structured-output strategy
   (Claude tool-use) has no direct OpenAI equivalent without rewriting
   `summary_service.py`.

4. **No deployment configuration.** No Dockerfile, no CI/CD, no production
   WSGI/ASGI process manager config. This runs via `uvicorn --reload` and
   `streamlit run` only — fine for local development and demos, not
   production-ready as-is.

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | ✗ | `claude-opus-4-20250514` | Model to use |
| `BACKEND_URL` | ✗ | `http://localhost:8000` | Backend URL for Streamlit |
| `MAX_TRANSCRIPT_LENGTH` | ✗ | `50000` | Max transcript chars |
| `LOG_LEVEL` | ✗ | `INFO` | Logging verbosity |

See `.env.example` for the full list.

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
```

Current status: **361 passed**, 0 failed.

---

## 🛠 Tech Stack

- **FastAPI** — async REST API
- **Streamlit** — rapid UI
- **Anthropic Claude** — AI summarisation & chat (tool-use structured output)
- **SQLAlchemy 2.0 + SQLite (WAL mode)** — persistence
- **youtube-transcript-api** — transcript extraction
- **Pydantic v2** — data validation
- **Loguru** — structured logging
- **pytest** — testing (361 tests, unit + integration)

---

## 📄 License

MIT © 2025
