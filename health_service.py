"""
Burnout / workload scoring.

Computes a student's stress score, burnout risk band, and study density
from real signals already in the database: how many study hours are
scheduled in the coming week, how many tasks are overdue, and how often
they've been leaning on the AI agents. No hardcoded "LOW / OPTIMAL"
placeholders -- every number here is derived from that student's data.
"""
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from database import ChatHistory, StudyTask, HealthMetrics


def compute_health_snapshot(db: Session, user_id: int) -> Dict[str, Any]:
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    today = date.today()

    query_count_7d = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.timestamp >= week_ago
    ).count()

    upcoming_tasks = db.query(StudyTask).filter(
        StudyTask.user_id == user_id,
        StudyTask.scheduled_date >= today,
        StudyTask.scheduled_date <= today + timedelta(days=6),
        StudyTask.is_completed == False,  # noqa: E712
    ).all()
    workload_hours = round(sum(t.duration_minutes for t in upcoming_tasks) / 60.0, 1)

    overdue_tasks = db.query(StudyTask).filter(
        StudyTask.user_id == user_id,
        StudyTask.scheduled_date < today,
        StudyTask.is_completed == False,  # noqa: E712
    ).count()

    # --- Scoring ---
    stress_score = min(
        100,
        round(query_count_7d * 2.5 + workload_hours * 4 + overdue_tasks * 8)
    )

    if stress_score < 35:
        burnout_risk = "LOW"
    elif stress_score < 70:
        burnout_risk = "MODERATE"
    else:
        burnout_risk = "HIGH"

    avg_daily_hours = workload_hours / 7.0
    if avg_daily_hours < 1.5:
        study_density = "LIGHT"
    elif avg_daily_hours <= 4:
        study_density = "OPTIMAL"
    else:
        study_density = "HEAVY"

    insights = _build_insights(query_count_7d, workload_hours, overdue_tasks, burnout_risk)

    return {
        "stress_score": stress_score,
        "burnout_risk": burnout_risk,
        "study_density": study_density,
        "workload_hours": workload_hours,
        "query_count_7d": query_count_7d,
        "overdue_tasks": overdue_tasks,
        "insights": insights,
    }


def _build_insights(query_count_7d: int, workload_hours: float, overdue_tasks: int, burnout_risk: str) -> List[str]:
    insights = []
    if overdue_tasks > 0:
        insights.append(f"You have {overdue_tasks} overdue task(s) — reschedule or clear these first to cut stress fastest.")
    if workload_hours > 20:
        insights.append(f"{workload_hours}h is scheduled this week. Consider moving low-priority topics to next week.")
    if query_count_7d > 25:
        insights.append("Heavy agent usage this week — good sign you're engaging, just make sure it's not last-minute cramming.")
    if burnout_risk == "LOW" and not insights:
        insights.append("Workload looks manageable. Good time to get ahead on upcoming topics.")
    if not insights:
        insights.append("Keep an eye on your task list — staying a day ahead keeps stress from compounding.")
    return insights


def save_snapshot(db: Session, user_id: int, snapshot: Dict[str, Any]) -> HealthMetrics:
    record = HealthMetrics(
        user_id=user_id,
        stress_level=snapshot["stress_score"],
        burnout_risk=snapshot["burnout_risk"],
        study_density=snapshot["study_density"],
        workload_hours=snapshot["workload_hours"],
        query_count_7d=snapshot["query_count_7d"],
        overdue_tasks=snapshot["overdue_tasks"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
