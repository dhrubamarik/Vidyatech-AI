import os
from datetime import datetime, date
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime, Date,
    ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker
import enum

# Loads DATABASE_URL / GROQ_API_KEY from a local .env file if one exists
# (and is silently a no-op if it doesn't) -- this runs first so every
# module that reads os.environ afterwards sees the values.
load_dotenv()

# NOTE: no credentials are hardcoded here. Set DATABASE_URL in your
# environment (or a local .env file that is NOT committed) before running
# the app. Falls back to a local SQLite file so the project still runs
# out of the box for development/demo purposes.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./vidyatech.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserRole(str, enum.Enum):
    STUDENT = "student"
    FACULTY = "faculty"


class TaskPriority(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TaskSource(str, enum.Enum):
    SYLLABUS = "syllabus"
    MANUAL = "manual"
    FACULTY = "faculty"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(String(20), default=UserRole.STUDENT, nullable=False)
    department = Column(String(100), nullable=True)  # Useful for Faculty View
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    chunks_indexed = Column(Integer, nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    agent_type = Column(String(50), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    has_context = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class HealthMetrics(Base):
    """A point-in-time snapshot of a student's computed workload/burnout
    state. New rows are written each time the health endpoint recomputes
    metrics, which is what powers the trend chart on the Health portal."""
    __tablename__ = "health_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stress_level = Column(Integer)          # 0-100
    burnout_risk = Column(String(20))       # LOW, MODERATE, HIGH
    study_density = Column(String(20))      # LIGHT, OPTIMAL, HEAVY
    workload_hours = Column(Float, default=0.0)   # scheduled hours, next 7 days
    query_count_7d = Column(Integer, default=0)   # agent queries, last 7 days
    overdue_tasks = Column(Integer, default=0)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class StudyTask(Base):
    """A single item on a student's adaptive schedule/planner, either
    imported from a parsed syllabus or added manually."""
    __tablename__ = "study_tasks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    subject = Column(String(150), nullable=True)
    scheduled_date = Column(Date, nullable=False, default=date.today)
    duration_minutes = Column(Integer, default=60)
    priority = Column(String(10), default=TaskPriority.MEDIUM)
    source = Column(String(20), default=TaskSource.MANUAL)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=True)   # faculty who assigned it, if any
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=True)


class SyllabusImport(Base):
    """Tracks each syllabus upload so the planner can group/regenerate
    the tasks it produced."""
    __tablename__ = "syllabus_imports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    topics_found = Column(Integer, default=0)
    exam_date = Column(Date, nullable=True)
    imported_at = Column(DateTime, default=datetime.utcnow)


class Community(Base):
    """A faculty-created study group. Students join with a 6-digit code;
    faculty use it to track member progress, assign shared tasks, and
    share notes."""
    __tablename__ = "communities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    description = Column(String(255), nullable=True)
    join_code = Column(String(6), unique=True, nullable=False, index=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    department = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CommunityMembership(Base):
    __tablename__ = "community_memberships"

    id = Column(Integer, primary_key=True, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)


class CommunityNote(Base):
    """A note/resource shared by faculty inside a community. Text is
    extracted and summarized by the AI at upload time."""
    __tablename__ = "community_notes"

    id = Column(Integer, primary_key=True, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    ai_summary = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class Quiz(Base):
    """A quiz built by a faculty member, either from AI-generated questions
    (from a topic) or entered manually, then assigned to students."""
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    subject = Column(String(150), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    option_a = Column(String(500), nullable=False)
    option_b = Column(String(500), nullable=False)
    option_c = Column(String(500), nullable=False)
    option_d = Column(String(500), nullable=False)
    correct_option = Column(String(1), nullable=False)  # "A" | "B" | "C" | "D"
    explanation = Column(Text, nullable=True)


class QuizAssignment(Base):
    __tablename__ = "quiz_assignments"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)


class QuizSubmission(Base):
    __tablename__ = "quiz_submissions"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    answers_json = Column(Text, nullable=False)   # {"<question_id>": "A", ...}
    score_pct = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)


class TopicMastery(Base):
    """Records that a student passed the Speed Learning Mode diagnostic
    (an 'explain it back' check graded against their own notes) on a
    topic. This is the signal that promotes a syllabus node on the
    Progress Tracker from Unseen/In-Progress to Mastered -- it's kept as
    its own append-only table (rather than a field on StudyTask) so a
    topic's mastery history/velocity can be reconstructed for the
    analytics dashboard.
    """
    __tablename__ = "topic_mastery"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String(255), nullable=False)
    accuracy_score = Column(Integer, default=0)
    mastered_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
