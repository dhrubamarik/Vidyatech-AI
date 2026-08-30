# VidyaTech AI — Updated Project

## Latest pass: per-user data isolation

**The bug:** `rag_service.py` kept a single, process-wide FAISS index.
Every student's uploaded documents were embedded into that *one* shared
index, and `/agent-chat`, `/planner/recap`, `/studyhub/reverse-quiz/*`,
and `/planner/cheatsheet` all searched it with no per-user filtering —
so one student's questions could surface another student's private
notes. `/documents` listed every user's files, `/clear` wiped
everyone's knowledge base at once, and two users uploading a
same-named file could overwrite each other's file on disk.

**The fix:**
- `RAGPipeline` now holds a dict of *isolated* per-scope FAISS indices
  (`stores: Dict[scope, index]`) instead of one shared index. Every
  retrieval/indexing call takes an explicit `scope` — in practice
  `user:<id>` — so one person's uploads are never reachable from
  another person's queries. This is enforced by `main.py`'s
  `user_scope()` helper on every route that touches the knowledge base:
  `/upload`, `/agent-chat`, `/documents` (list + delete),
  `/planner/heatmap`, `/planner/recap`, `/planner/cheatsheet`,
  `/studyhub/reverse-quiz/prompt` and `/evaluate`.
- `/documents` now requires `user_id` and only returns that user's own
  documents. `DELETE /documents/{id}` verifies the caller actually
  uploaded that document before deleting it.
- Uploaded files are stored on disk as `<user_id>_<filename>`, so two
  users can't collide on the same filename.
- `/clear` was a global "wipe everyone's knowledge base" endpoint; it's
  now `POST /clear?user_id=` and only clears the caller's own uploads.
- Faculty-generated quizzes (`/quizzes/generate`) draw from the
  generating faculty member's own uploaded reference material, not a
  shared pool.
- Communities remain the one deliberate exception: faculty-shared
  community notes (`/communities/{id}/notes`) are visible to every
  member of that community, by design — that's the "unity should only
  be within communities" behavior. `GET
  /communities/notes/{note_id}/download` now checks the requester is
  actually a member (or the creator) of that community first.

**New delete/remove options**, beyond the existing per-document delete
and full account deletion:
- `POST /clear?user_id=` — wipe just your uploaded documents/knowledge
  base, without deleting your account.
- `DELETE /chat-history/{user_id}` — clear your Study Hub chat history.
- Both are wired up in the UI: a "Clear My Knowledge Base" button on
  the Knowledge Base page, and a "Clear My Chat History" button on the
  Study Hub page.

## What changed (previous pass)

**Security fix:** `database.py` no longer has a real-looking Postgres
password hardcoded as a default. It now falls back to a local SQLite file
(`vidyatech.db`) if `DATABASE_URL` isn't set, so the project still runs
out of the box, but nothing sensitive is committed. Set your real
connection string as an environment variable instead:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/vidyatech_db"
```

Do the same for your Groq key:

```bash
export GROQ_API_KEY="your_real_key"
```

## New backend modules

- `planner_service.py` — parses uploaded syllabus text into topics and
  spreads them into an adaptive day-by-day schedule ahead of an exam
  date (harder topics earlier, last day reserved for revision).
- `health_service.py` — computes a real burnout/stress snapshot from a
  student's actual scheduled workload, overdue tasks, and agent-query
  volume (no hardcoded "LOW/OPTIMAL").
- `analytics_service.py` — derives faculty "doubt clusters" from the
  actual text of student queries (word/bigram frequency), and a
  resource-utilization figure from real query and chunk counts.

`main.py` now exposes planner (`/planner/...`), health
(`/student/health/...`), and richer faculty analytics endpoints on top
of the original auth/upload/agent-chat routes.

## Frontend

`index.html` is a full rebuild matching the mockup: a landing page,
sidebar-navigation app shell, and six workspaces — Dashboard, Planner,
Knowledge Base, Study Hub (multi-agent chat), Health Tracker, and
Faculty Analytics. It's still a single vanilla-JS/HTML file (no build
step); it loads Chart.js from a CDN for the workload and stress-trend
charts, so an internet connection is needed the first time it's opened.

## Running it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `index.html` in a browser (it talks to `http://127.0.0.1:8000`).

## What's new in this pass

- **Rendering fix:** agent responses were showing raw `**bold**` and
  `\(LaTeX\)` syntax. Responses are now parsed as Markdown (via
  marked.js, sanitized with DOMPurify) and math delimiters are rendered
  with KaTeX — both loaded from a CDN, so an internet connection is
  needed the first time the page loads.
- **Download responses:** every agent response in Study Hub has PDF and
  DOCX buttons. These call new `/export/pdf` and `/export/docx`
  endpoints (`export_service.py`, using `reportlab` and `python-docx`)
  and trigger a browser download of the generated file.
- **Delete documents & accounts:** a ✕ button on each document in the
  Knowledge Base removes it from FAISS *and* the database
  (`DELETE /documents/{id}`). A "Delete Account" button in the sidebar
  wipes a user and everything that references them — tasks, chat
  history, health snapshots, uploaded documents (and their FAISS
  chunks), and any communities they created — via `DELETE /users/{id}`.
  Note: `rag_service.py`'s FAISS index now tracks which chunks belong to
  which filename so a single document's vectors can be surgically
  removed without wiping everyone else's.
- **Faculty task assignment:** the Department Analytics page has an
  "Assign" button per student and an "Assign Task to All" button, both
  backed by `POST /faculty/assign-task`. Assigned tasks show up
  automatically in that student's Planner and Dashboard (they're just
  `StudyTask` rows with `source="faculty"`).
- **Communities:** faculty create a community (`POST /communities`) and
  get a unique 6-digit join code; students join with that code
  (`POST /communities/join`). Inside a community, faculty can see member
  progress, assign shared tasks (`POST /communities/{id}/assign-task`),
  and upload notes (`POST /communities/{id}/notes`) — each note is
  text-extracted and summarized by the LLM automatically
  (`rag_engine.summarize_text`), and members can download the original
  file or read the summary from their side.

## Database migrations (Alembic)

The project now includes an Alembic scaffold (`alembic.ini`,
`alembic/env.py`) wired to read `DATABASE_URL` the same way the app
does, so no credentials live in these files either. I couldn't run
database commands from this environment, so you'll need to run the
first migration yourself:

```bash
# One-time setup, from the project root, with your venv active and
# DATABASE_URL / .env already set:

# 1. If you already have tables in your DB from running the app before,
#    generate a migration and mark it as already applied (baseline),
#    so Alembic doesn't try to recreate existing tables:
alembic revision --autogenerate -m "baseline"
alembic stamp head

# 2. From now on, whenever you change a model in database.py:
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

If you're starting from a brand-new empty database instead, skip the
`stamp head` step — just run `alembic upgrade head` after the first
`revision --autogenerate` and it'll create everything.

## Bug fixes (this pass)

- **RAG answered off-topic questions using whatever was nearest.**
  `retrieve_context()` had no relevance cutoff — it always returned the
  top-k nearest chunks even if none were actually related to the
  query, and the Summarizer's prompt just said "summarize the
  context," so asking something unrelated to your uploaded material
  (e.g. a recipe question against a math PDF) got answered from the
  math PDF anyway. Fixed with a distance threshold in
  `rag_service.py` (`RELEVANCE_THRESHOLD`) — queries with no
  sufficiently close match now correctly fall back to "outside the
  scope of your uploaded document."
- **Markdown/LaTeX responses showed raw `**bold**` and `\binom{n}{0}`
  instead of rendering.** Standard Markdown treats `\(`, `\)`, `\[`,
  `\]` as *escapable* characters, so running the Markdown parser first
  silently stripped those backslashes before KaTeX ever saw the
  delimiters. Fixed in `index.html`'s `renderMarkdown()`: math segments
  are now extracted and rendered with KaTeX *first*, swapped for
  plain-text placeholders Markdown can't touch, and substituted back in
  after Markdown parsing.

## New features (this pass)

All of the below were added as new files/endpoints/views — existing
code was only touched where the bug fixes above required it.

1. **Live Syllabus Heatmap & Exam Readiness Score** — `GET
   /planner/heatmap/{user_id}` scores each of your scheduled topics
   against how well your uploaded documents actually cover it (via
   FAISS distance), buckets into 🟢/🟡/🔴, and rolls that into a single
   Exam Readiness %. Shown on the Planner page.
2. **60-Second Audio Recap** — `POST /planner/recap` generates a short
   spoken-style summary of a topic from your notes; the browser's
   built-in Web Speech API reads it aloud client-side (no TTS backend
   or audio files needed). Also on the Planner page.
3. **Proof-of-Understanding Reverse Quiz** — `POST
   /studyhub/reverse-quiz/prompt` asks you to explain a topic in your
   own words; `POST /studyhub/reverse-quiz/evaluate` grades your
   explanation against the source material and returns an accuracy
   score, missing points, and misconceptions. On the Study Hub page.
4. **One-Click Cheat Sheet Builder** — `POST /planner/cheatsheet`
   builds a one-page revision sheet (definitions, formulas, likely exam
   questions) from your scheduled topics, downloadable as PDF/DOCX via
   the existing export endpoints. On the Planner page.
5. **Faculty MCQ Quizzes** — new `Quiz` / `QuizQuestion` /
   `QuizAssignment` / `QuizSubmission` tables. Faculty generate a quiz
   from a topic (`POST /quizzes/generate`, AI-authored from the
   knowledge base), assign it to a department or a specific community
   (`POST /quizzes/{id}/assign`), and students take it
   (`GET /quizzes/{id}/take`, `POST /quizzes/{id}/submit` —
   auto-graded, with correct answers revealed after submission).
   Faculty see per-student results (`GET /quizzes/{id}/results`). New
   "Quizzes" nav item for both roles.

Since new tables were added again (`quizzes`, `quiz_questions`,
`quiz_assignments`, `quiz_submissions`), the same schema note applies
as before: use the Alembic commands above, or drop/recreate if you're
still just testing locally.

- Swap the email-only login for real password/session auth before any
  real deployment.
- The syllabus topic extractor is a heuristic (line/heading based) — it
  works well on syllabi with one topic per line, less well on dense
  prose; an LLM-based extraction pass would be more robust if Groq
  quota allows it.
- Faculty-only endpoints (`/communities`, `/faculty/assign-task`, etc.)
  check the caller's role by looking up `faculty_id`/`user_id` in the
  database — reasonable for a demo without real auth, but not a
  substitute for actual session-based authorization.
- Community note files are stored on local disk
  (`./community_notes/<community_id>/...`) — fine for local use, but
  swap for object storage (S3, etc.) before deploying anywhere with
  multiple server instances or ephemeral disks.
