import os
import shutil
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import (
    init_db, SessionLocal, User, DocumentRecord, ChatHistory, HealthMetrics,
    StudyTask, SyllabusImport, UserRole, Community, CommunityMembership, CommunityNote,
    Quiz, QuizQuestion, QuizAssignment, QuizSubmission
)
from rag_service import rag_engine
import planner_service
import health_service
import analytics_service
import export_service
import community_service
import heatmap_service
import recap_service
import reverse_quiz_service
import cheatsheet_service
import quiz_service
from database import TopicMastery
import progress_service
import career_service

app = FastAPI(title="VidyaTech Multi-Role Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./temp_uploads"
NOTES_DIR = "./community_notes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)


def user_scope(user_id: int) -> str:
    """Every user's uploads/queries live in their own isolated FAISS
    scope, so nothing one person uploads or asks about can ever surface
    in another person's retrieval -- the only shared context is inside
    communities, and even that (community notes) is summarized rather
    than mixed into anyone's personal retrieval index."""
    return f"user:{user_id}"


def doc_disk_path(doc: "DocumentRecord") -> str:
    """Where a given document's raw file lives on disk. Prefixed with
    the uploader's id so two different users uploading a file with the
    same name never collide or overwrite each other's file."""
    return os.path.join(UPLOAD_DIR, f"{doc.uploaded_by}_{doc.filename}")


def require_faculty(user_id: int, db: Session) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.role != UserRole.FACULTY:
        raise HTTPException(status_code=403, detail="Only faculty accounts can do this.")
    return user


@app.on_event("startup")
def startup_event():
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Pydantic Schemas ---
class UserRegister(BaseModel):
    name: str
    email: str
    role: UserRole
    department: str = "Computer Science"


class UserLogin(BaseModel):
    email: str


class AgentQueryModel(BaseModel):
    user_id: int
    query: str
    agent_type: str = "doubt_solver"


class TaskCreate(BaseModel):
    user_id: int
    title: str
    subject: Optional[str] = None
    scheduled_date: date
    duration_minutes: int = 60
    priority: str = "MEDIUM"


class ExportRequest(BaseModel):
    title: str = "VidyaTech AI Response"
    content: str


# --- Career Intelligence & Autonomous Scholarship Finder ---
class GapMatrixRequest(BaseModel):
    user_id: int
    target_role: str


class ResumeBulletsRequest(BaseModel):
    user_id: int
    target_role: Optional[str] = None
    focus_area: Optional[str] = None


class ScholarshipMatchRequest(BaseModel):
    user_id: int
    gpa: float
    department: Optional[str] = None
    year: Optional[str] = None
    interests: Optional[str] = None


class ScholarshipEssayRequest(BaseModel):
    user_id: int
    scholarship_name: Optional[str] = None
    essay_prompt: str


class FacultyAssignTask(BaseModel):
    faculty_id: int
    student_ids: List[int]
    title: str
    subject: Optional[str] = None
    scheduled_date: date
    duration_minutes: int = 60
    priority: str = "MEDIUM"


class CommunityCreate(BaseModel):
    faculty_id: int
    name: str
    description: Optional[str] = None
    department: Optional[str] = None


class CommunityJoin(BaseModel):
    user_id: int
    join_code: str


class CommunityAssignTask(BaseModel):
    faculty_id: int
    title: str
    scheduled_date: date
    duration_minutes: int = 60
    priority: str = "MEDIUM"
    student_ids: Optional[List[int]] = None  # omit to assign to every member


class RecapRequest(BaseModel):
    topic: str
    user_id: int


class CheatsheetRequest(BaseModel):
    user_id: int


class ReverseQuizPromptRequest(BaseModel):
    topic: str
    user_id: int


class ReverseQuizEvalRequest(BaseModel):
    topic: str
    explanation: str
    user_id: int


class SprintStartRequest(BaseModel):
    topic: str
    user_id: int


class SprintGradeRequest(BaseModel):
    topic: str
    explanation: str
    user_id: int


class QuizGenerateRequest(BaseModel):
    faculty_id: int
    title: str
    subject: Optional[str] = None
    topic: str
    num_questions: int = 5
    community_id: Optional[int] = None


class QuizAssignRequest(BaseModel):
    faculty_id: int
    student_ids: Optional[List[int]] = None
    community_id: Optional[int] = None


class QuizSubmitRequest(BaseModel):
    student_id: int
    answers: Dict[str, str]


# --- Authentication & User Management ---
@app.post("/register", summary="1. Register New User")
def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="An account with this email already exists. Please log in instead.")

    new_user = User(
        name=user_data.name,
        email=user_data.email,
        role=user_data.role,
        department=user_data.department
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "status": "success",
        "user_id": new_user.id,
        "name": new_user.name,
        "role": new_user.role,
        "department": new_user.department
    }


@app.post("/login", summary="2. User Login")
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found. Please register first.")

    return {
        "status": "success",
        "user_id": user.id,
        "name": user.name,
        "role": user.role,
        "department": user.department
    }


@app.delete("/users/{user_id}", summary="2b. Delete Account")
def delete_account(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Clean up everything owned by this user so we don't leave orphaned rows.
    db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
    db.query(HealthMetrics).filter(HealthMetrics.user_id == user_id).delete()
    db.query(StudyTask).filter(StudyTask.user_id == user_id).delete()
    db.query(SyllabusImport).filter(SyllabusImport.user_id == user_id).delete()
    db.query(CommunityMembership).filter(CommunityMembership.user_id == user_id).delete()

    # Documents this user uploaded to their own knowledge base: remove
    # the files from disk and drop their entire personal vector-store
    # scope in one shot (nobody else's scope is touched).
    owned_docs = db.query(DocumentRecord).filter(DocumentRecord.uploaded_by == user_id).all()
    for doc in owned_docs:
        file_path = doc_disk_path(doc)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.delete(doc)
    rag_engine.clear_vector_store(user_scope(user_id))

    # If this user is faculty, remove communities they created (and
    # everything inside them) rather than leaving orphaned groups.
    owned_communities = db.query(Community).filter(Community.created_by == user_id).all()
    for community in owned_communities:
        db.query(CommunityMembership).filter(CommunityMembership.community_id == community.id).delete()
        db.query(StudyTask).filter(StudyTask.community_id == community.id).delete()
        notes = db.query(CommunityNote).filter(CommunityNote.community_id == community.id).all()
        for note in notes:
            if os.path.exists(note.file_path):
                os.remove(note.file_path)
            db.delete(note)
        db.delete(community)

    db.delete(user)
    db.commit()
    return {"status": "success", "message": "Account and associated data deleted."}


# --- RAG / Knowledge Base Endpoints ---
# Every document is indexed into the uploader's own private FAISS scope
# (see user_scope()). Nobody else's agent-chat, recap, cheat sheet,
# reverse quiz, or heatmap query can ever retrieve it.
@app.post("/upload", summary="3. Upload a document into YOUR private knowledge base")
async def upload_document(file: UploadFile = File(...), user_id: int = 1, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Create the DB row first so we have a stable, per-document id to use
    # as the FAISS chunk-owner label -- avoids two different documents
    # (or two different users' same-named files) ever colliding.
    doc_record = DocumentRecord(filename=file.filename, chunks_indexed=0, uploaded_by=user_id)
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    file_path = doc_disk_path(doc_record)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        num_chunks = rag_engine.process_and_index_document(
            file_path, scope=user_scope(user_id), owner=str(doc_record.id)
        )
        doc_record.chunks_indexed = num_chunks
        db.commit()
        db.refresh(doc_record)

        return {
            "status": "success",
            "id": doc_record.id,
            "filename": file.filename,
            "chunks_indexed": num_chunks
        }
    except ValueError as err:
        db.delete(doc_record)
        db.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        db.delete(doc_record)
        db.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Unexpected upload error: {str(err)}")


@app.delete("/documents/{doc_id}", summary="3c. Remove a Document from YOUR Knowledge Base")
def delete_document(doc_id: int, user_id: int, db: Session = Depends(get_db)):
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.uploaded_by != user_id:
        raise HTTPException(status_code=403, detail="You can only remove documents you uploaded yourself.")

    rag_engine.remove_document(user_scope(user_id), owner=str(doc.id))
    file_path = doc_disk_path(doc)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.delete(doc)
    db.commit()
    return {"status": "success", "message": f"Removed {doc.filename} and its indexed chunks."}


@app.get("/documents", summary="3b. List YOUR Indexed Documents")
def list_documents(user_id: int, db: Session = Depends(get_db)):
    docs = db.query(DocumentRecord).filter(DocumentRecord.uploaded_by == user_id) \
        .order_by(DocumentRecord.uploaded_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "chunks_indexed": d.chunks_indexed,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in docs
    ]


@app.post("/agent-chat", summary="4. Query YOUR Knowledge Base & Route to Selected AI Agent")
async def agent_chat(payload: AgentQueryModel, db: Session = Depends(get_db)):
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = rag_engine.generate_agent_response(payload.query, user_scope(payload.user_id), payload.agent_type)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["response"])

    chat_record = ChatHistory(
        user_id=payload.user_id,
        agent_type=payload.agent_type,
        query=payload.query,
        response=result["response"],
        has_context=str(result["has_context"])
    )
    db.add(chat_record)
    db.commit()

    return result


@app.get("/chat-history/{user_id}", summary="4b. Recent Chat History for a Student")
def get_chat_history(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(ChatHistory).filter(ChatHistory.user_id == user_id) \
        .order_by(ChatHistory.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "agent_type": r.agent_type,
            "query": r.query,
            "response": r.response,
            "has_context": r.has_context == "True",
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in reversed(rows)
    ]


# --- Export agent responses ---
@app.delete("/chat-history/{user_id}", summary="4b2. Clear YOUR Chat History")
def clear_chat_history(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    deleted = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
    db.commit()
    return {"status": "success", "deleted": deleted}


@app.post("/export/pdf", summary="4c. Export a Response as PDF")
def export_pdf(payload: ExportRequest):
    pdf_bytes = export_service.generate_pdf_bytes(payload.title, payload.content)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="response.pdf"'}
    )


@app.post("/export/docx", summary="4d. Export a Response as DOCX")
def export_docx(payload: ExportRequest):
    docx_bytes = export_service.generate_docx_bytes(payload.title, payload.content)
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="response.docx"'}
    )


# --- AI Planner & Adaptive Scheduler ---
@app.post("/planner/import-syllabus", summary="5. Import Syllabus & Generate Adaptive Schedule")
async def import_syllabus(
    file: UploadFile = File(...),
    user_id: int = 1,
    exam_date: str = None,
    db: Session = Depends(get_db),
):
    if not exam_date:
        raise HTTPException(status_code=400, detail="exam_date (YYYY-MM-DD) is required to build a schedule.")
    try:
        target_date = datetime.strptime(exam_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="exam_date must be in YYYY-MM-DD format.")
    if target_date <= date.today():
        raise HTTPException(status_code=400, detail="exam_date must be in the future.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    raw_text = rag_engine.extract_text_from_file(file_path)
    if not raw_text:
        raise HTTPException(status_code=400, detail="No extractable text found in the syllabus file.")

    topics = planner_service.extract_topics(raw_text)
    if not topics:
        raise HTTPException(status_code=400, detail="Could not detect distinct topics in this file. Try a syllabus with one topic per line.")

    schedule = planner_service.build_adaptive_schedule(topics, date.today(), target_date)

    created_tasks = []
    for item in schedule:
        task = StudyTask(
            user_id=user_id,
            title=item["title"],
            subject=file.filename.rsplit(".", 1)[0],
            scheduled_date=item["scheduled_date"],
            duration_minutes=item["duration_minutes"],
            priority=item["priority"],
            source="syllabus",
        )
        db.add(task)
        created_tasks.append(task)

    db.add(SyllabusImport(
        user_id=user_id,
        filename=file.filename,
        topics_found=len(topics),
        exam_date=target_date,
    ))
    db.commit()
    for t in created_tasks:
        db.refresh(t)

    return {
        "status": "success",
        "topics_found": len(topics),
        "tasks_created": len(created_tasks),
        "schedule": [
            {
                "id": t.id,
                "title": t.title,
                "scheduled_date": t.scheduled_date.isoformat(),
                "duration_minutes": t.duration_minutes,
                "priority": t.priority,
            }
            for t in created_tasks
        ],
    }


@app.get("/planner/schedule/{user_id}", summary="5b. Get Student's Full Schedule")
def get_schedule(user_id: int, db: Session = Depends(get_db)):
    tasks = db.query(StudyTask).filter(StudyTask.user_id == user_id) \
        .order_by(StudyTask.scheduled_date.asc()).all()

    total_hours_week = round(sum(
        t.duration_minutes for t in tasks
        if not t.is_completed and date.today() <= t.scheduled_date <= date.today() + timedelta(days=6)
    ) / 60.0, 1)

    return {
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "subject": t.subject,
                "scheduled_date": t.scheduled_date.isoformat(),
                "duration_minutes": t.duration_minutes,
                "priority": t.priority,
                "source": t.source,
                "is_completed": t.is_completed,
            }
            for t in tasks
        ],
        "workload_hours_next_7d": total_hours_week,
    }


@app.post("/planner/tasks", summary="5c. Add a Manual Task")
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    task = StudyTask(
        user_id=payload.user_id,
        title=payload.title,
        subject=payload.subject,
        scheduled_date=payload.scheduled_date,
        duration_minutes=payload.duration_minutes,
        priority=payload.priority,
        source="manual",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"status": "success", "id": task.id}


@app.patch("/planner/tasks/{task_id}/complete", summary="5d. Toggle Task Completion")
def toggle_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(StudyTask).filter(StudyTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    task.is_completed = not task.is_completed
    db.commit()
    return {"status": "success", "is_completed": task.is_completed}


@app.delete("/planner/tasks/{task_id}", summary="5e. Delete a Task")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(StudyTask).filter(StudyTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    db.delete(task)
    db.commit()
    return {"status": "success"}


# --- Faculty Department Analytics Dashboard ---
@app.get("/faculty/analytics/{department}", summary="6. Faculty Aggregated Analytics")
def get_faculty_analytics(department: str, db: Session = Depends(get_db)):
    students = db.query(User).filter(
        User.role == UserRole.STUDENT,
        User.department == department
    ).all()
    student_ids = [s.id for s in students]

    dept_chats = db.query(ChatHistory).filter(ChatHistory.user_id.in_(student_ids)).all() if student_ids else []
    total_queries = len(dept_chats)

    doc_rows = db.query(DocumentRecord).all()
    total_chunks = sum(d.chunks_indexed for d in doc_rows) or 0

    doubt_clusters = analytics_service.top_doubt_clusters([c.query for c in dept_chats])
    utilization = analytics_service.resource_utilization(total_queries, total_chunks)

    student_progress = []
    for s in students:
        s_tasks = db.query(StudyTask).filter(StudyTask.user_id == s.id).all()
        done = sum(1 for t in s_tasks if t.is_completed)
        pct = round((done / len(s_tasks)) * 100) if s_tasks else 0
        student_progress.append({"user_id": s.id, "name": s.name, "progress_pct": pct})

    return {
        "department": department,
        "active_students": len(students),
        "total_doubts_resolved": total_queries,
        "doubt_clusters": doubt_clusters or ["Not enough activity yet"],
        "resource_utilization": utilization,
        "student_progress": student_progress,
    }


@app.post("/faculty/assign-task", summary="6b. Faculty Assigns a Task to Students")
def faculty_assign_task(payload: FacultyAssignTask, db: Session = Depends(get_db)):
    require_faculty(payload.faculty_id, db)
    if not payload.student_ids:
        raise HTTPException(status_code=400, detail="Select at least one student.")

    created = []
    for sid in payload.student_ids:
        task = StudyTask(
            user_id=sid,
            title=payload.title,
            subject=payload.subject,
            scheduled_date=payload.scheduled_date,
            duration_minutes=payload.duration_minutes,
            priority=payload.priority,
            source="faculty",
            assigned_by=payload.faculty_id,
        )
        db.add(task)
        created.append(task)
    db.commit()
    return {"status": "success", "tasks_created": len(created)}


# --- Communities ---
@app.post("/communities", summary="9. Faculty Creates a Community")
def create_community(payload: CommunityCreate, db: Session = Depends(get_db)):
    faculty = require_faculty(payload.faculty_id, db)
    code = community_service.generate_unique_join_code(db)
    community = Community(
        name=payload.name,
        description=payload.description,
        join_code=code,
        created_by=faculty.id,
        department=payload.department or faculty.department,
    )
    db.add(community)
    db.commit()
    db.refresh(community)
    return {
        "status": "success",
        "id": community.id,
        "name": community.name,
        "join_code": community.join_code,
    }


@app.post("/communities/join", summary="9b. Student Joins a Community by Code")
def join_community(payload: CommunityJoin, db: Session = Depends(get_db)):
    community = db.query(Community).filter(Community.join_code == payload.join_code.strip()).first()
    if not community:
        raise HTTPException(status_code=404, detail="No community found with that code.")

    existing = db.query(CommunityMembership).filter(
        CommunityMembership.community_id == community.id,
        CommunityMembership.user_id == payload.user_id,
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="You've already joined this community.")

    db.add(CommunityMembership(community_id=community.id, user_id=payload.user_id))
    db.commit()
    return {"status": "success", "community_id": community.id, "name": community.name}


@app.get("/communities/mine/{user_id}", summary="9c. Communities for a User (created or joined)")
def my_communities(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.role == UserRole.FACULTY:
        communities = db.query(Community).filter(Community.created_by == user_id).all()
    else:
        memberships = db.query(CommunityMembership).filter(CommunityMembership.user_id == user_id).all()
        community_ids = [m.community_id for m in memberships]
        communities = db.query(Community).filter(Community.id.in_(community_ids)).all() if community_ids else []

    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "join_code": c.join_code if user.role == UserRole.FACULTY else None,
            "department": c.department,
            "member_count": db.query(CommunityMembership).filter(CommunityMembership.community_id == c.id).count(),
        }
        for c in communities
    ]


@app.get("/communities/{community_id}", summary="9d. Community Detail (members, progress, notes)")
def get_community(community_id: int, db: Session = Depends(get_db)):
    community = db.query(Community).filter(Community.id == community_id).first()
    if not community:
        raise HTTPException(status_code=404, detail="Community not found.")

    notes = db.query(CommunityNote).filter(CommunityNote.community_id == community_id) \
        .order_by(CommunityNote.uploaded_at.desc()).all()

    return {
        "id": community.id,
        "name": community.name,
        "description": community.description,
        "join_code": community.join_code,
        "department": community.department,
        "members": community_service.member_progress(db, community_id),
        "notes": [
            {
                "id": n.id,
                "filename": n.filename,
                "ai_summary": n.ai_summary,
                "uploaded_at": n.uploaded_at.isoformat() if n.uploaded_at else None,
            }
            for n in notes
        ],
    }


@app.post("/communities/{community_id}/assign-task", summary="9e. Faculty Assigns a Task to a Community")
def assign_community_task(community_id: int, payload: CommunityAssignTask, db: Session = Depends(get_db)):
    community = db.query(Community).filter(Community.id == community_id).first()
    if not community:
        raise HTTPException(status_code=404, detail="Community not found.")
    require_faculty(payload.faculty_id, db)
    if community.created_by != payload.faculty_id:
        raise HTTPException(status_code=403, detail="Only the faculty member who created this community can assign tasks here.")

    if payload.student_ids:
        target_ids = payload.student_ids
    else:
        memberships = db.query(CommunityMembership).filter(CommunityMembership.community_id == community_id).all()
        target_ids = [m.user_id for m in memberships]

    if not target_ids:
        raise HTTPException(status_code=400, detail="This community has no members yet.")

    for sid in target_ids:
        db.add(StudyTask(
            user_id=sid,
            title=payload.title,
            subject=community.name,
            scheduled_date=payload.scheduled_date,
            duration_minutes=payload.duration_minutes,
            priority=payload.priority,
            source="faculty",
            assigned_by=payload.faculty_id,
            community_id=community_id,
        ))
    db.commit()
    return {"status": "success", "tasks_created": len(target_ids)}


@app.post("/communities/{community_id}/notes", summary="9f. Faculty Shares a Note (auto-summarized)")
async def upload_community_note(
    community_id: int,
    file: UploadFile = File(...),
    faculty_id: int = 1,
    db: Session = Depends(get_db),
):
    community = db.query(Community).filter(Community.id == community_id).first()
    if not community:
        raise HTTPException(status_code=404, detail="Community not found.")
    require_faculty(faculty_id, db)
    if community.created_by != faculty_id:
        raise HTTPException(status_code=403, detail="Only the faculty member who created this community can share notes here.")

    community_dir = os.path.join(NOTES_DIR, str(community_id))
    os.makedirs(community_dir, exist_ok=True)
    file_path = os.path.join(community_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    raw_text = rag_engine.extract_text_from_file(file_path)
    ai_summary = rag_engine.summarize_text(raw_text) if raw_text else "No extractable text to summarize."

    note = CommunityNote(
        community_id=community_id,
        filename=file.filename,
        file_path=file_path,
        uploaded_by=faculty_id,
        ai_summary=ai_summary,
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return {
        "status": "success",
        "id": note.id,
        "filename": note.filename,
        "ai_summary": note.ai_summary,
    }


@app.get("/communities/notes/{note_id}/download", summary="9g. Download a Shared Note")
def download_community_note(note_id: int, user_id: int, db: Session = Depends(get_db)):
    note = db.query(CommunityNote).filter(CommunityNote.id == note_id).first()
    if not note or not os.path.exists(note.file_path):
        raise HTTPException(status_code=404, detail="Note not found.")

    community = db.query(Community).filter(Community.id == note.community_id).first()
    is_member = db.query(CommunityMembership).filter(
        CommunityMembership.community_id == note.community_id,
        CommunityMembership.user_id == user_id,
    ).first()
    if not (community and (community.created_by == user_id or is_member)):
        raise HTTPException(status_code=403, detail="Only members of this community can download its notes.")

    return FileResponse(note.file_path, filename=note.filename)


# --- AI Health & Burnout Portal ---
@app.get("/student/health/{user_id}", summary="7. Student Health Metrics (recomputed live)")
def get_health_metrics(user_id: int, db: Session = Depends(get_db)):
    snapshot = health_service.compute_health_snapshot(db, user_id)
    health_service.save_snapshot(db, user_id, snapshot)
    return snapshot


@app.get("/student/health/{user_id}/trend", summary="7b. Health History for Trend Chart")
def get_health_trend(user_id: int, limit: int = 14, db: Session = Depends(get_db)):
    rows = db.query(HealthMetrics).filter(HealthMetrics.user_id == user_id) \
        .order_by(HealthMetrics.recorded_at.desc()).limit(limit).all()
    return [
        {
            "stress_level": r.stress_level,
            "workload_hours": r.workload_hours,
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
        }
        for r in reversed(rows)
    ]


@app.post("/clear", summary="8. Clear YOUR Knowledge Base (your uploaded documents only)")
async def clear_store(user_id: int, db: Session = Depends(get_db)):
    """Bulk-remove every document this user has uploaded, without
    touching their account, tasks, chat history, or anyone else's
    documents. For a full account wipe, use DELETE /users/{user_id}
    instead."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    owned_docs = db.query(DocumentRecord).filter(DocumentRecord.uploaded_by == user_id).all()
    removed = 0
    for doc in owned_docs:
        file_path = doc_disk_path(doc)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.delete(doc)
        removed += 1
    db.commit()

    rag_engine.clear_vector_store(user_scope(user_id))
    return {
        "status": "success",
        "message": f"Cleared your knowledge base ({removed} document(s) removed)."
    }


# =====================================================================
# NEW: hackathon feature set (heatmap, audio recap, reverse quiz,
# cheat sheet) + faculty-assigned MCQ quizzes. All additive endpoints.
# =====================================================================

def _student_topics(db: Session, user_id: int) -> List[str]:
    """Unique, non-revision topic titles currently on a student's
    schedule -- used as the topic list for the heatmap and cheat sheet."""
    tasks = db.query(StudyTask).filter(StudyTask.user_id == user_id).all()
    seen = []
    for t in tasks:
        if t.title not in seen and "revision" not in t.title.lower():
            seen.append(t.title)
    return seen


# --- 1. Live Syllabus Heatmap & Exam Readiness Score ---
@app.get("/planner/heatmap/{user_id}", summary="10. Syllabus Coverage Heatmap & Exam Readiness Score")
def get_heatmap(user_id: int, db: Session = Depends(get_db)):
    topics = _student_topics(db, user_id)
    if not topics:
        return {"coverage": [], "exam_readiness_score": 0, "message": "No syllabus topics yet — import one in Planner."}
    coverage = rag_engine.coverage_for_topics(user_scope(user_id), topics)
    score = heatmap_service.exam_readiness_score(coverage)
    return {"coverage": coverage, "exam_readiness_score": score}


# --- 2. 60-Second Audio Flashcard Recap (text only; spoken client-side) ---
@app.post("/planner/recap", summary="11. Generate a 60-Second Spoken Recap for a Topic")
def generate_recap(payload: RecapRequest):
    if not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    return recap_service.build_recap(payload.topic, user_scope(payload.user_id))


# --- 3. Interactive "Proof-of-Understanding" Reverse Quiz ---
@app.post("/studyhub/reverse-quiz/prompt", summary="12. Get a 'Explain This Back' Prompt for a Topic")
def reverse_quiz_prompt(payload: ReverseQuizPromptRequest):
    if not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    return reverse_quiz_service.generate_prompt(payload.topic, user_scope(payload.user_id))


@app.post("/studyhub/reverse-quiz/evaluate", summary="12b. Grade the Student's Explanation")
def reverse_quiz_evaluate(payload: ReverseQuizEvalRequest):
    if not payload.explanation.strip():
        raise HTTPException(status_code=400, detail="Explanation cannot be empty.")
    return reverse_quiz_service.evaluate_explanation(payload.topic, payload.explanation, user_scope(payload.user_id))


# --- 4. One-Click Cheat Sheet Builder ---
@app.post("/planner/cheatsheet", summary="13. Generate a One-Page Exam Cheat Sheet")
def generate_cheatsheet(payload: CheatsheetRequest, db: Session = Depends(get_db)):
    topics = _student_topics(db, payload.user_id)
    content = cheatsheet_service.build_cheatsheet(topics, user_scope(payload.user_id))
    return {"content": content}


# --- Faculty: AI-Generated MCQ Quizzes ---
@app.post("/quizzes/generate", summary="14. Faculty Generates a Quiz from a Topic")
def generate_quiz(payload: QuizGenerateRequest, db: Session = Depends(get_db)):
    require_faculty(payload.faculty_id, db)
    result = quiz_service.generate_questions(payload.topic, user_scope(payload.faculty_id), payload.num_questions)
    if not result["questions"]:
        raise HTTPException(status_code=400, detail=result.get("message") or "Could not generate questions for this topic.")

    quiz = Quiz(
        title=payload.title,
        subject=payload.subject or payload.topic,
        created_by=payload.faculty_id,
        community_id=payload.community_id,
    )
    db.add(quiz)
    db.commit()
    db.refresh(quiz)

    created_questions = []
    for q in result["questions"]:
        question = QuizQuestion(
            quiz_id=quiz.id,
            prompt=q["prompt"],
            option_a=q["option_a"],
            option_b=q["option_b"],
            option_c=q["option_c"],
            option_d=q["option_d"],
            correct_option=q["correct_option"],
            explanation=q.get("explanation", ""),
        )
        db.add(question)
        created_questions.append(question)
    db.commit()
    for q in created_questions:
        db.refresh(q)

    return {
        "status": "success",
        "quiz_id": quiz.id,
        "title": quiz.title,
        "questions": [
            {
                "id": q.id, "prompt": q.prompt,
                "option_a": q.option_a, "option_b": q.option_b,
                "option_c": q.option_c, "option_d": q.option_d,
                "correct_option": q.correct_option, "explanation": q.explanation,
            }
            for q in created_questions
        ],
    }


@app.post("/quizzes/{quiz_id}/assign", summary="14b. Faculty Assigns a Quiz to Students or a Community")
def assign_quiz(quiz_id: int, payload: QuizAssignRequest, db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    require_faculty(payload.faculty_id, db)
    if quiz.created_by != payload.faculty_id:
        raise HTTPException(status_code=403, detail="Only the faculty member who created this quiz can assign it.")

    if payload.community_id:
        memberships = db.query(CommunityMembership).filter(CommunityMembership.community_id == payload.community_id).all()
        target_ids = [m.user_id for m in memberships]
    else:
        target_ids = payload.student_ids or []

    if not target_ids:
        raise HTTPException(status_code=400, detail="No students to assign to.")

    created = 0
    for sid in target_ids:
        exists = db.query(QuizAssignment).filter(QuizAssignment.quiz_id == quiz_id, QuizAssignment.student_id == sid).first()
        if not exists:
            db.add(QuizAssignment(quiz_id=quiz_id, student_id=sid))
            created += 1
    db.commit()
    return {"status": "success", "students_assigned": created}


@app.get("/quizzes/mine/{user_id}", summary="14c. Quizzes for a User (created or assigned)")
def my_quizzes(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.role == UserRole.FACULTY:
        quizzes = db.query(Quiz).filter(Quiz.created_by == user_id).order_by(Quiz.created_at.desc()).all()
        out = []
        for q in quizzes:
            submissions = db.query(QuizSubmission).filter(QuizSubmission.quiz_id == q.id).all()
            avg_score = round(sum(s.score_pct for s in submissions) / len(submissions)) if submissions else None
            out.append({
                "id": q.id, "title": q.title, "subject": q.subject,
                "question_count": db.query(QuizQuestion).filter(QuizQuestion.quiz_id == q.id).count(),
                "assigned_count": db.query(QuizAssignment).filter(QuizAssignment.quiz_id == q.id).count(),
                "submission_count": len(submissions),
                "average_score": avg_score,
            })
        return out

    assignments = db.query(QuizAssignment).filter(QuizAssignment.student_id == user_id).all()
    out = []
    for a in assignments:
        quiz = db.query(Quiz).filter(Quiz.id == a.quiz_id).first()
        if not quiz:
            continue
        submission = db.query(QuizSubmission).filter(
            QuizSubmission.quiz_id == quiz.id, QuizSubmission.student_id == user_id
        ).first()
        out.append({
            "id": quiz.id, "title": quiz.title, "subject": quiz.subject,
            "question_count": db.query(QuizQuestion).filter(QuizQuestion.quiz_id == quiz.id).count(),
            "completed": submission is not None,
            "score_pct": submission.score_pct if submission else None,
        })
    return out


@app.get("/quizzes/{quiz_id}/take", summary="14d. Student-Facing Quiz (no answers revealed)")
def take_quiz(quiz_id: int, db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    questions = db.query(QuizQuestion).filter(QuizQuestion.quiz_id == quiz_id).all()
    return {
        "id": quiz.id, "title": quiz.title, "subject": quiz.subject,
        "questions": [
            {
                "id": q.id, "prompt": q.prompt,
                "option_a": q.option_a, "option_b": q.option_b,
                "option_c": q.option_c, "option_d": q.option_d,
            }
            for q in questions
        ],
    }


@app.post("/quizzes/{quiz_id}/submit", summary="14e. Student Submits Quiz Answers (auto-graded)")
def submit_quiz(quiz_id: int, payload: QuizSubmitRequest, db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    questions = db.query(QuizQuestion).filter(QuizQuestion.quiz_id == quiz_id).all()
    question_dicts = [
        {"id": q.id, "correct_option": q.correct_option, "explanation": q.explanation}
        for q in questions
    ]
    graded = quiz_service.grade_submission(question_dicts, payload.answers)

    import json as _json
    existing = db.query(QuizSubmission).filter(
        QuizSubmission.quiz_id == quiz_id, QuizSubmission.student_id == payload.student_id
    ).first()
    if existing:
        existing.answers_json = _json.dumps(payload.answers)
        existing.score_pct = graded["score_pct"]
        existing.submitted_at = datetime.utcnow()
    else:
        db.add(QuizSubmission(
            quiz_id=quiz_id, student_id=payload.student_id,
            answers_json=_json.dumps(payload.answers), score_pct=graded["score_pct"],
        ))
    db.commit()
    return graded


@app.get("/quizzes/{quiz_id}/results", summary="14f. Faculty Views Per-Student Results")
def quiz_results(quiz_id: int, db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    submissions = db.query(QuizSubmission).filter(QuizSubmission.quiz_id == quiz_id).all()
    out = []
    for s in submissions:
        student = db.query(User).filter(User.id == s.student_id).first()
        out.append({
            "student_name": student.name if student else "Unknown",
            "score_pct": s.score_pct,
            "submitted_at": s.submitted_at.isoformat() if s.submitted_at else None,
        })
    return {"quiz_title": quiz.title, "results": out}


# =====================================================================
# NEW: Accelerated Syllabus Engine (Progress Tracker). Purely additive --
# reuses the existing syllabus topics (StudyTask), RAG coverage
# (rag_engine.coverage_for_topics, same source as the heatmap), and the
# existing recap/reverse-quiz services for Speed Learning Mode. Nothing
# above this point was changed to build it.
# =====================================================================

def _student_topics_by_subject(db: Session, user_id: int) -> Dict[str, List[str]]:
    """Same underlying data as _student_topics, grouped by the syllabus
    file each topic came from (StudyTask.subject) so the tracker can
    show a Module -> Topic tree."""
    tasks = db.query(StudyTask).filter(StudyTask.user_id == user_id).all()
    grouped: Dict[str, List[str]] = {}
    for t in tasks:
        if "revision" in t.title.lower():
            continue
        subject = t.subject or "General"
        bucket = grouped.setdefault(subject, [])
        if t.title not in bucket:
            bucket.append(t.title)
    return grouped


@app.get("/progress/tree/{user_id}", summary="15. Progress Tracker — Visual Syllabus Tree")
def get_progress_tree(user_id: int, db: Session = Depends(get_db)):
    grouped = _student_topics_by_subject(db, user_id)
    all_topics = [t for topics in grouped.values() for t in topics]
    if not all_topics:
        return {
            "modules": [], "percent_complete": 0, "topics_total": 0, "topics_mastered": 0,
            "message": "No syllabus topics yet — import one in Planner.",
        }
    coverage = rag_engine.coverage_for_topics(user_scope(user_id), all_topics)
    mastery_rows = db.query(TopicMastery).filter(TopicMastery.user_id == user_id).all()
    return progress_service.build_tree(grouped, coverage, mastery_rows)


@app.post("/progress/sprint/start", summary="15b. Speed Learning Mode — Core Summary + Recall Prompt")
def start_sprint(payload: SprintStartRequest):
    if not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    scope = user_scope(payload.user_id)
    summary = recap_service.build_recap(payload.topic, scope)
    prompt = reverse_quiz_service.generate_prompt(payload.topic, scope)
    return {
        "topic": payload.topic,
        "core_summary": summary.get("recap"),
        "recall_prompt": prompt.get("prompt"),
        "has_context": summary.get("has_context", False),
    }


@app.post("/progress/sprint/grade", summary="15c. Speed Learning Mode — Instant Gap Calibration")
def grade_sprint(payload: SprintGradeRequest, db: Session = Depends(get_db)):
    if not payload.explanation.strip():
        raise HTTPException(status_code=400, detail="Explanation cannot be empty.")
    result = reverse_quiz_service.evaluate_explanation(
        payload.topic, payload.explanation, user_scope(payload.user_id)
    )
    score = int(result.get("accuracy_score", 0) or 0)
    mastered = score >= progress_service.MASTERY_THRESHOLD
    if mastered:
        db.add(TopicMastery(user_id=payload.user_id, topic=payload.topic, accuracy_score=score))
        db.commit()
    result["new_status"] = "MASTERED" if mastered else "IN_PROGRESS"
    result["mastery_threshold"] = progress_service.MASTERY_THRESHOLD
    return result


@app.get("/progress/analytics/{user_id}", summary="15d. Accelerated Velocity & Retention Analytics")
def get_progress_analytics(user_id: int, db: Session = Depends(get_db)):
    grouped = _student_topics_by_subject(db, user_id)
    topics_total = len({t for topics in grouped.values() for t in topics})
    mastery_rows = db.query(TopicMastery).filter(TopicMastery.user_id == user_id) \
        .order_by(TopicMastery.mastered_at.asc()).all()
    distinct_mastered = len({m.topic for m in mastery_rows})
    velocity = progress_service.compute_velocity(mastery_rows)
    return {
        "study_velocity_topics_per_hour": velocity,
        "topics_mastered": distinct_mastered,
        "topics_total": topics_total,
        "predicted_completion": progress_service.predict_completion(topics_total, distinct_mastered, velocity),
        "memory_decay_warnings": progress_service.memory_decay_warnings(mastery_rows),
    }


# --- Career Intelligence & Autonomous Scholarship Finder ---
# Reuses each student's existing private RAG scope (their resume/notes,
# already indexed via the normal /upload endpoint) -- no new upload path,
# table, or vector store is introduced. See career_service.py.
@app.post("/career/gap-matrix", summary="16. Resume vs. Career Roadmap Gap Matrix")
def career_gap_matrix(payload: GapMatrixRequest):
    result = career_service.build_gap_matrix(user_scope(payload.user_id), payload.target_role)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/career/resume-bullets", summary="16b. One-Click Resume Bullet Synthesizer")
def career_resume_bullets(payload: ResumeBulletsRequest):
    return career_service.synthesize_resume_bullets(
        user_scope(payload.user_id), payload.target_role, payload.focus_area
    )


@app.get("/career/scholarships", summary="17. List All Scholarships")
def career_list_scholarships():
    return {"scholarships": career_service.list_scholarships()}


@app.post("/career/scholarships/match", summary="17b. Autonomous Scholarship Match & Eligibility Scoring")
def career_match_scholarships(payload: ScholarshipMatchRequest):
    matches = career_service.match_scholarships(
        payload.gpa, payload.department, payload.year, payload.interests
    )
    return {"matches": matches}


@app.post("/career/scholarship-essay", summary="17c. RAG-Driven Scholarship Essay Assistant")
def career_scholarship_essay(payload: ScholarshipEssayRequest):
    result = career_service.draft_scholarship_essay(
        user_scope(payload.user_id), payload.scholarship_name, payload.essay_prompt
    )
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result
