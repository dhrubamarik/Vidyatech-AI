"""
Faculty-created study communities: unique 6-digit join codes, member
progress, and shared notes with an auto-generated AI summary.
"""
import random
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from database import Community, CommunityMembership, StudyTask, User


def generate_unique_join_code(db: Session) -> str:
    """A 6-digit numeric code, retried until it doesn't collide with an
    existing community (the search space is 900,000 codes, so collisions
    are rare, but we check anyway)."""
    for _ in range(50):
        code = str(random.randint(100000, 999999))
        exists = db.query(Community).filter(Community.join_code == code).first()
        if not exists:
            return code
    raise RuntimeError("Could not generate a unique join code — try again.")


def member_progress(db: Session, community_id: int) -> List[Dict[str, Any]]:
    members = db.query(CommunityMembership).filter(CommunityMembership.community_id == community_id).all()
    result = []
    for m in members:
        student = db.query(User).filter(User.id == m.user_id).first()
        if not student:
            continue
        tasks = db.query(StudyTask).filter(
            StudyTask.user_id == student.id,
            StudyTask.community_id == community_id,
        ).all()
        done = sum(1 for t in tasks if t.is_completed)
        pct = round((done / len(tasks)) * 100) if tasks else 0
        result.append({
            "user_id": student.id,
            "name": student.name,
            "tasks_assigned": len(tasks),
            "tasks_completed": done,
            "progress_pct": pct,
        })
    return result
