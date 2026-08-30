"""
Accelerated Syllabus Engine.

Turns signals that already exist elsewhere in the app -- syllabus topics
(planner_service), RAG coverage (rag_service.coverage_for_topics), and
Speed Learning Mode diagnostic results (graded via reverse_quiz_service)
-- into the three things the progress tracker needs:

1. A visual Module -> Topic tree with a red/yellow/green status per node
   and an overall percent-complete.
2. Study velocity + a predicted syllabus-completion date.
3. Memory-decay warnings for topics that were mastered a while ago (or
   only barely passed) and are due for a refresher.

Deliberately built on top of existing services rather than new tracking
plumbing, so it's accurate starting from a student's very first sprint.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Iterable

MASTERY_THRESHOLD = 70   # accuracy_score needed to flag a topic Mastered
DECAY_DAYS = 5           # days after mastery before we flag a refresher
DECAY_SCORE_FLOOR = 85   # mastered, but barely -- flag it even if recent


def node_status(topic: str, coverage_status: str, mastered_topics: Iterable[str]) -> str:
    """Merge RAG-coverage (from the existing heatmap) with explicit
    mastery events into the 3-state node the tracker shows. Passing the
    Speed Learning diagnostic always wins over coverage alone, since
    it's proof of active recall rather than just "notes exist"."""
    if topic in mastered_topics:
        return "MASTERED"
    if coverage_status == "RED":
        return "UNSEEN"
    return "IN_PROGRESS"  # GREEN/YELLOW coverage: notes cover it, not yet verified


def build_tree(tasks_by_subject: Dict[str, List[str]], coverage: List[Dict[str, Any]],
                mastery_rows: List[Any]) -> Dict[str, Any]:
    coverage_map = {c["topic"]: c["status"] for c in coverage}
    mastered_topics = {m.topic for m in mastery_rows}

    modules, total, mastered = [], 0, 0
    for subject, topics in tasks_by_subject.items():
        nodes = []
        for t in topics:
            status = node_status(t, coverage_map.get(t, "RED"), mastered_topics)
            nodes.append({"topic": t, "status": status})
            total += 1
            if status == "MASTERED":
                mastered += 1
        modules.append({"module": subject, "topics": nodes})

    percent = round((mastered / total) * 100) if total else 0
    return {
        "modules": modules,
        "percent_complete": percent,
        "topics_total": total,
        "topics_mastered": mastered,
    }


def _first_mastery_times(mastery_rows: List[Any]) -> List[datetime]:
    """One timestamp per distinct topic -- the earliest time it was
    mastered -- so redoing a sprint on the same topic doesn't distort
    the velocity/date math below."""
    first: Dict[str, datetime] = {}
    for m in mastery_rows:
        if m.topic not in first or m.mastered_at < first[m.topic]:
            first[m.topic] = m.mastered_at
    return sorted(first.values())


def compute_velocity(mastery_rows: List[Any]) -> float:
    """Topics mastered per hour, from the span between the first and
    most recent distinct-topic mastery. Needs at least two mastered
    topics to mean anything; 0 otherwise."""
    times = _first_mastery_times(mastery_rows)
    if len(times) < 2:
        return 0.0
    span_hours = (times[-1] - times[0]).total_seconds() / 3600
    if span_hours <= 0:
        return round(len(times), 2)
    return round((len(times) - 1) / span_hours, 2)


def predict_completion(topics_total: int, topics_mastered: int, velocity: float) -> str:
    remaining = topics_total - topics_mastered
    if topics_total == 0:
        return "Import a syllabus in Planner to get a prediction."
    if remaining <= 0:
        return "Syllabus complete — nice work."
    if velocity <= 0:
        return "Not enough sprint data yet — finish a couple more Speed Learning sprints for an estimate."
    hours_needed = remaining / velocity
    eta = datetime.utcnow() + timedelta(hours=hours_needed)
    return f"At your current rate, you'll finish by {eta.strftime('%A, %b %d')}."


def memory_decay_warnings(mastery_rows: List[Any]) -> List[Dict[str, Any]]:
    """Flags topics whose most recent mastery is stale (>= DECAY_DAYS
    old) or was only a borderline pass, so they surface for a quick
    refresher before they're forgotten."""
    latest: Dict[str, Any] = {}
    for m in mastery_rows:
        if m.topic not in latest or m.mastered_at > latest[m.topic].mastered_at:
            latest[m.topic] = m

    now = datetime.utcnow()
    warnings = []
    for m in latest.values():
        days_since = (now - m.mastered_at).days
        stale = days_since >= DECAY_DAYS
        borderline = m.accuracy_score < DECAY_SCORE_FLOOR
        if stale or borderline:
            warnings.append({
                "topic": m.topic,
                "days_since_mastery": days_since,
                "accuracy_score": m.accuracy_score,
                "reason": "time" if stale else "borderline score",
            })
    return warnings
