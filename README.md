# VidyaTech AI

VidyaTech AI is a full-stack academic platform for students and faculty. It combines a personal AI study assistant (RAG over your own uploaded notes/PDFs), an adaptive planner, health/burnout tracking, faculty analytics, class communities, and AI-generated quizzes — all served from a single FastAPI backend with a single-file vanilla JS/HTML frontend.

## What it does

**For students**
- Upload notes/PDFs/DOCX and chat with a multi-agent AI (Tutor, Quiz Master, Summarizer) that answers strictly from *your own* uploaded material (RAG with a relevance cutoff, so off-topic questions are refused instead of hallucinated).
- Import a syllabus and get an adaptive day-by-day study schedule (harder topics scheduled earlier, last day reserved for revision).
- See a live syllabus heatmap (🟢/🟡/🔴 per topic) and an overall Exam Readiness score, based on how well your notes actually cover each scheduled topic.
- Get a 60-second spoken recap of any topic (browser Web Speech API, no audio files/TTS backend).
- Take a "reverse quiz": explain a topic in your own words and get it graded against your source material, with missing points and misconceptions called out.
- Generate a one-page cheat sheet (definitions, formulas, likely exam questions) and export any AI response as PDF or DOCX.
- Track burnout risk / study density from your real scheduled workload, overdue tasks, and query volume — not hardcoded labels.
- Join faculty-created communities with a 6-digit code, receive shared tasks and notes, and take faculty-assigned quizzes.
- Track topic mastery on a progress tree and use Career tools (skill-gap matrix, resume bullets, scholarship matching/essays).

**For faculty**
- See department-level "doubt cluster" analytics derived from the actual text of student queries, plus resource-utilization stats.
- Assign tasks to one student or an entire department/community.
- Create communities, share notes (auto-summarized by AI on upload), and track member progress.
- Generate AI-authored MCQ quizzes from a topic (grounded in your own uploaded reference material), assign them, and view per-student results.

**Security/data model note:** every student's and faculty member's uploads live in their own isolated FAISS index (`scope = user:<id>`), so nobody's queries or documents are ever retrievable by another user. The one deliberate exception is community notes, which are visible to every member of that community by design.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, SQLAlchemy, Alembic |
| Database | SQLite by default (falls back automatically), PostgreSQL supported via `DATABASE_URL` |
| Vector search | FAISS (`faiss-cpu`) + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Document parsing | `pypdf`, `python-docx` |
| Export | `reportlab` (PDF), `python-docx` (DOCX) |
| Frontend | Single-file `index.html` — vanilla JS, Chart.js, marked.js + DOMPurify (Markdown), KaTeX (math rendering). No build step. |

## Prerequisites

- **Python 3.10+** and `pip`
- A free **Groq API key** — sign up at [console.groq.com](https://console.groq.com) and create a key. This powers every AI feature (chat agents, quiz generation, summaries, grading, community note summaries).
- SQLite needs no setup and is used by default. PostgreSQL is optional (see below).

## Setup — running it locally

**1. Get the project into a folder and open a terminal there.**

```bash
cd Test4.0
```

**2. Create and activate a virtual environment** (recommended so this project's packages don't clash with anything else on your machine).

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies.**

```bash
pip install -r requirements.txt
```

The first install pulls `sentence-transformers` and its embedding model (~100MB) plus `faiss-cpu` — this can take a few minutes the first time. The embedding model is downloaded automatically on the app's first run (needs an internet connection once; it's cached locally after that).

**4. Configure your environment variables.**

Create a file named `.env` in the project root (same folder as `main.py`):

```env
GROQ_API_KEY=your_real_groq_api_key
DATABASE_URL=sqlite:///./vidyatech.db
```

- `GROQ_API_KEY` is **required** — without it, every AI feature (chat, quizzes, summaries, recap, cheat sheets, career tools) will fail.
- `DATABASE_URL` is optional. If you omit it entirely, the app falls back to a local SQLite file (`vidyatech.db`) automatically — good enough to run the whole project with zero extra setup. To use PostgreSQL instead:
  ```env
  DATABASE_URL=postgresql://user:password@localhost:5432/vidyatech_db
  ```
  (make sure the database itself already exists — the app/Alembic creates tables, not the database).

`.env` is read automatically at startup (`python-dotenv`) — don't commit it; add it to `.gitignore`.

**5. Create the database tables.**

The simplest path for local/demo use — tables are created automatically the first time the server starts (`init_db()` runs on startup), so you can usually skip straight to step 6.

If you'd rather manage schema changes properly with Alembic (recommended if you'll be editing the models in `database.py`):

```bash
mkdir versions          # one-time: alembic needs this folder to exist
alembic revision --autogenerate -m "baseline"
alembic upgrade head
```

Whenever you change a model afterward:
```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

**6. Run the backend.**

```bash
uvicorn main:app --reload
```

The API is now running at `http://127.0.0.1:8000` (interactive docs at `http://127.0.0.1:8000/docs`).

**7. Open the frontend.**

Just open `index.html` directly in your browser (double-click it, or right-click → Open With → your browser). It's hardcoded to talk to `http://127.0.0.1:8000`, so make sure the backend from step 6 is running first. An internet connection is needed the first time you load the page (it pulls Chart.js, marked.js, DOMPurify, and KaTeX from a CDN).

That's it — register an account (choose Student or Faculty), and you're in.

## Project structure

```
Test4.0/
├── main.py                    # FastAPI app: all routes wired up here
├── database.py                # SQLAlchemy models + DB connection/engine
├── rag_service.py             # FAISS indexing, retrieval, Groq multi-agent chat
├── planner_service.py         # Syllabus parsing → adaptive schedule
├── health_service.py          # Burnout/stress snapshot computation
├── analytics_service.py       # Faculty doubt-cluster & resource analytics
├── export_service.py          # PDF / DOCX export of AI responses
├── community_service.py       # Community join-code + membership helpers
├── heatmap_service.py         # Syllabus coverage heatmap scoring
├── recap_service.py           # 60-second topic recap generation
├── reverse_quiz_service.py    # "Explain it back" prompt + grading
├── cheatsheet_service.py      # One-page revision sheet builder
├── quiz_service.py            # Faculty MCQ quiz generation/grading
├── progress_service.py        # Topic mastery / progress tree
├── career_service.py          # Skill-gap matrix, resume bullets, scholarships
├── env.py, script.py.mako     # Alembic migration environment
├── alembic.ini                # Alembic config (reads DATABASE_URL, no secrets)
├── requirements.txt
└── index.html                 # Entire frontend (no build step)
```

## API overview

Full interactive reference is auto-generated at `http://127.0.0.1:8000/docs` once the server is running. Broad groups of endpoints:

- **Auth** — `/register`, `/login`, `DELETE /users/{id}`
- **Documents / RAG** — `/upload`, `/documents`, `/agent-chat`, `/clear`
- **Planner** — `/planner/import-syllabus`, `/planner/schedule/{user_id}`, `/planner/tasks`, `/planner/heatmap/{user_id}`, `/planner/recap`, `/planner/cheatsheet`
- **Study Hub** — `/chat-history/{user_id}`, `/studyhub/reverse-quiz/prompt`, `/studyhub/reverse-quiz/evaluate`, `/export/pdf`, `/export/docx`
- **Health** — `/student/health/{user_id}`, `/student/health/{user_id}/trend`
- **Communities** — `/communities`, `/communities/join`, `/communities/{id}`, `/communities/{id}/assign-task`, `/communities/{id}/notes`
- **Quizzes** — `/quizzes/generate`, `/quizzes/{id}/assign`, `/quizzes/{id}/take`, `/quizzes/{id}/submit`, `/quizzes/{id}/results`
- **Faculty** — `/faculty/analytics/{department}`, `/faculty/assign-task`
- **Progress & Career** — `/progress/tree/{user_id}`, `/progress/sprint/start`, `/progress/sprint/grade`, `/career/gap-matrix`, `/career/resume-bullets`, `/career/scholarships`

## Known limitations

- Login is email-only with no password/session mechanism — fine for a local demo, but replace with real authentication before deploying this anywhere public.
- The syllabus topic extractor is heuristic (line/heading based); dense prose syllabi are parsed less reliably than line-per-topic ones.
- Faculty-only endpoints check role by looking up the caller's `user_id` in the database rather than a real session/JWT — again, fine for local/demo use only.
- Community note files are stored on local disk (`./community_notes/...`); swap for object storage (S3, etc.) before running multiple server instances or on ephemeral disks.
- CORS is currently wide open (`allow_origins=["*"]`) to make local development easy — tighten this before deploying.

## Troubleshooting

- **"Failed to fetch" in the browser / requests hanging** — the backend isn't running, or isn't on port 8000. Check the `uvicorn` terminal for errors.
- **AI features return errors** — almost always a missing/invalid `GROQ_API_KEY` in `.env`, or the free-tier Groq quota being exhausted.
- **First request is slow** — the embedding model download/load happens on first use; subsequent requests are fast.
- **`alembic` command not found / import errors** — make sure your virtual environment is activated before running Alembic commands.
