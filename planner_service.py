"""
Adaptive planner logic.

Given raw syllabus text, this module extracts a list of study topics and
spreads them across the days between today and a target exam/deadline
date, front-loading denser/harder topics and reserving the final day(s)
for revision. This is intentionally a transparent heuristic (not an LLM
call) so the schedule is fast, deterministic, and free to generate.
"""
import re
from datetime import date, timedelta
from typing import List, Dict, Any

HARD_KEYWORDS = (
    "advanced", "exam", "project", "design", "architecture", "proof",
    "optimization", "algorithm", "theorem", "derivation", "analysis"
)

# Lines that are clearly not topics: empty, boilerplate headers, page numbers.
NOISE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^page \d+", re.I),
    re.compile(r"^table of contents", re.I),
    re.compile(r"^(course|syllabus|semester|credits?)\s*[:\-]", re.I),
]

# A line that looks like a topic heading: numbered/bulleted, or a "Unit/
# Module/Chapter/Week N:" style header, or a short standalone line.
TOPIC_LINE = re.compile(
    r"^\s*(?:(?:unit|module|chapter|week|lecture|topic)\s*\d+\s*[:\-–]?\s*)?"
    r"(?:[\d]+[\.\)]\s*|[-•*]\s*)?(.{4,110})$",
    re.I
)


def extract_topics(raw_text: str, max_topics: int = 24) -> List[str]:
    """Pull a deduplicated, ordered list of candidate study topics out of
    free-form syllabus text."""
    topics: List[str] = []
    seen = set()

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(p.search(line) for p in NOISE_PATTERNS):
            continue
        if len(line) > 130:
            # Likely a prose paragraph, not a topic heading -> skip.
            continue

        m = TOPIC_LINE.match(line)
        if not m:
            continue
        candidate = m.group(1).strip(" :-–\t")
        if len(candidate) < 4:
            continue

        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        topics.append(candidate)

        if len(topics) >= max_topics:
            break

    return topics


def _priority_for(topic: str) -> str:
    low = topic.lower()
    if any(k in low for k in HARD_KEYWORDS):
        return "HIGH"
    if len(topic.split()) <= 3:
        return "LOW"
    return "MEDIUM"


def _duration_for(priority: str) -> int:
    return {"HIGH": 90, "MEDIUM": 60, "LOW": 45}.get(priority, 60)


def build_adaptive_schedule(
    topics: List[str],
    start_date: date,
    exam_date: date,
    max_topics_per_day: int = 3,
) -> List[Dict[str, Any]]:
    """Spread `topics` across the days between start_date and exam_date.

    - Reserves the final day before the exam for revision only.
    - Harder topics (per _priority_for) are scheduled earlier where possible.
    - Caps how many topics land on any single day.
    """
    if not topics:
        return []

    days_available = max((exam_date - start_date).days, 1)
    revision_day = exam_date - timedelta(days=1) if days_available > 1 else exam_date
    study_days = max(days_available - 1, 1) if days_available > 1 else 1

    # Harder topics first so they get earlier, less rushed slots.
    ordered = sorted(topics, key=lambda t: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[_priority_for(t)])

    topics_per_day = max(1, min(max_topics_per_day, -(-len(ordered) // study_days)))

    schedule = []
    for i, topic in enumerate(ordered):
        day_offset = min(i // topics_per_day, study_days - 1)
        scheduled_date = start_date + timedelta(days=day_offset)
        priority = _priority_for(topic)
        schedule.append({
            "title": topic,
            "scheduled_date": scheduled_date,
            "duration_minutes": _duration_for(priority),
            "priority": priority,
        })

    if days_available > 1:
        schedule.append({
            "title": "Final revision & practice questions",
            "scheduled_date": revision_day,
            "duration_minutes": 90,
            "priority": "HIGH",
        })

    return schedule
