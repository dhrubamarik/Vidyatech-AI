"""
Turns per-topic RAG coverage (from rag_service.coverage_for_topics) into
an aggregate "Exam Readiness Score" — the number the syllabus heatmap
UI leads with.
"""
from typing import List, Dict, Any

STATUS_WEIGHT = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0}


def exam_readiness_score(coverage: List[Dict[str, Any]]) -> int:
    if not coverage:
        return 0
    total = sum(STATUS_WEIGHT.get(c["status"], 0) for c in coverage)
    return round((total / len(coverage)) * 100)
