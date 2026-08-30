"""
Career Intelligence & Autonomous Scholarship Finder.

This is purely additive on top of the existing RAG pipeline -- it does
NOT introduce a separate upload path, database table, or vector store.
A student's resume is just a normal document uploaded through the
existing `/upload` endpoint (already indexed into their private
per-user FAISS scope). Everything here retrieves from that same scope,
so resume content, notes, and project write-ups are all searchable
together. Scholarship data is a small in-memory seed list; swapping it
for a real DB table or external feed later would not require changing
any of the matching logic below.
"""
from typing import List, Dict, Any, Optional
from rag_service import rag_engine

# --- Seed scholarship data --------------------------------------------------
# "departments"/"years": ["Any"] means open to everyone. "tags" are matched
# against the student's free-text interests for the keyword-overlap part of
# the score.
SCHOLARSHIPS: List[Dict[str, Any]] = [
    {
        "id": "sch_001", "name": "STEM Excellence Scholarship", "provider": "National STEM Foundation",
        "amount": "$5,000", "min_gpa": 3.5, "departments": ["Computer Science", "Engineering", "Any"],
        "years": ["3", "4", "Any"], "deadline": "2026-11-15",
        "tags": ["research", "ai", "machine learning", "engineering", "innovation"],
        "description": "For high-achieving STEM students with a demonstrated research or project track record.",
    },
    {
        "id": "sch_002", "name": "First-Generation Scholar Award", "provider": "Horizon Education Trust",
        "amount": "$3,000", "min_gpa": 3.0, "departments": ["Any"], "years": ["Any"],
        "deadline": "2026-10-01",
        "tags": ["first-generation", "community", "leadership", "financial need"],
        "description": "Supports first-generation college students demonstrating leadership and community involvement.",
    },
    {
        "id": "sch_003", "name": "Women in Technology Grant", "provider": "TechForward Alliance",
        "amount": "$4,000", "min_gpa": 3.2, "departments": ["Computer Science", "Engineering", "Data Science"],
        "years": ["2", "3", "4"], "deadline": "2026-12-05",
        "tags": ["women in tech", "coding", "hackathon", "software", "diversity"],
        "description": "For women pursuing degrees in technology fields with an interest in closing the industry gender gap.",
    },
    {
        "id": "sch_004", "name": "Community Impact Fellowship", "provider": "Civic Futures Foundation",
        "amount": "$2,500", "min_gpa": 2.8, "departments": ["Any"], "years": ["Any"],
        "deadline": "2026-09-30",
        "tags": ["volunteering", "social impact", "nonprofit", "community service"],
        "description": "For students who have led or contributed significantly to a community service initiative.",
    },
    {
        "id": "sch_005", "name": "Future Innovators Hackathon Grant", "provider": "BuildNext Labs",
        "amount": "$1,500", "min_gpa": 3.0, "departments": ["Computer Science", "Engineering", "Design"],
        "years": ["1", "2", "3", "4"], "deadline": "2026-11-20",
        "tags": ["hackathon", "prototype", "startup", "product", "rag", "llm"],
        "description": "For students actively building projects or prototypes, especially ones shown at hackathons.",
    },
    {
        "id": "sch_006", "name": "Merit Scholars Award", "provider": "State University Foundation",
        "amount": "$6,000", "min_gpa": 3.7, "departments": ["Any"], "years": ["Any"],
        "deadline": "2026-10-25",
        "tags": ["academic excellence", "honors", "gpa"],
        "description": "Purely merit-based award for students with an outstanding overall academic record.",
    },
    {
        "id": "sch_007", "name": "Data Science & AI Futures Award", "provider": "Prescient Analytics Foundation",
        "amount": "$4,500", "min_gpa": 3.3, "departments": ["Computer Science", "Data Science", "Statistics"],
        "years": ["2", "3", "4"], "deadline": "2026-12-15",
        "tags": ["data science", "ai", "machine learning", "analytics", "rag", "llm"],
        "description": "For students specializing in data science, AI, or applied machine learning.",
    },
    {
        "id": "sch_008", "name": "Regional Access Scholarship", "provider": "Statewide Opportunity Fund",
        "amount": "$2,000", "min_gpa": 2.5, "departments": ["Any"], "years": ["1", "2"],
        "deadline": "2026-09-15",
        "tags": ["financial need", "access", "underrepresented"],
        "description": "Need-based award aimed at improving access for early-year undergraduates.",
    },
]

GAP_MATRIX_SYSTEM = (
    "You are a Career Readiness Analyst. Compare a student's resume/notes "
    "against the typical requirements of their target role. Respond in "
    "Markdown with exactly these sections: '## Matched Skills' (bullets, "
    "only skills actually evidenced in the source material), '## Skill "
    "Gaps' (bullets, things the role typically needs that are NOT evidenced "
    "in the source material), '## Recommended Next Steps' (3 short, "
    "concrete bullets — a course, project type, or certification), and "
    "'## Readiness Score' (a single line: a percentage 0-100 followed by "
    "one sentence of justification). Be honest and specific — never invent "
    "experience the student doesn't actually have."
)

BULLET_SYSTEM = (
    "You are a Resume Writer. From the student's notes/project material "
    "below, write 4-6 resume-ready bullet points using the STAR method "
    "(Situation/Task, Action, Result) condensed into one punchy line each. "
    "Start every bullet with a strong action verb, quantify impact where "
    "the source material supports it, and keep each bullet under 25 words. "
    "Only use accomplishments that are actually evidenced in the source "
    "material — never fabricate metrics or outcomes that aren't there. "
    "Output as a Markdown bullet list, nothing else."
)

ESSAY_SYSTEM = (
    "You are a Scholarship Essay Assistant. Draft a compelling, authentic "
    "scholarship essay (300-400 words) that directly answers the essay "
    "prompt, grounded ONLY in the achievements, experiences, and context "
    "given below. Never invent facts, awards, or experiences that aren't "
    "in the source material — if the material is thin, write a shorter, "
    "honest essay rather than padding it with fabricated details. Write "
    "in first person, with a clear narrative arc (a specific moment, what "
    "it taught the student, how it connects to their goals). Output plain "
    "prose paragraphs, no headers."
)


def _resume_context(scope: str, query: str, top_k: int = 8) -> Dict[str, Any]:
    """Pulls the nearest chunks from the student's own knowledge base for
    this scope, WITHOUT rag_engine's strict relevance-distance cutoff.
    That cutoff exists in retrieve_context() to stop an unrelated question
    being answered from whatever's nearest -- the right call for Q&A, but
    wrong here: these features summarize/build from everything a student
    has already uploaded to their own private scope, so "found" should
    just mean "you've uploaded something," not "this one query happened
    to land within a tight embedding-distance radius of a chunk." A long,
    composite query like "skills, experience, projects, and certifications
    relevant to X" often sits just outside that radius even when the
    resume chunks are exactly what's needed."""
    store = rag_engine.stores.get(scope)
    if store is None or store.index.ntotal == 0 or not store.documents:
        return {"context": "", "found": False}

    top_k = min(top_k, store.index.ntotal)
    query_vec = rag_engine.embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = store.index.search(query_vec, top_k)
    chunks = [store.documents[i] for i in indices[0] if 0 <= i < len(store.documents)]

    if not chunks:
        return {"context": "", "found": False}
    return {"context": "\n\n".join(chunks), "found": True}


def build_gap_matrix(scope: str, target_role: str) -> Dict[str, Any]:
    if not target_role.strip():
        return {"status": "error", "message": "Target role is required."}

    retrieval = _resume_context(
        scope, f"skills, experience, projects, and certifications relevant to {target_role}"
    )
    if not retrieval["found"]:
        return {
            "status": "no_resume",
            "message": "Upload your resume (or project notes) in the Knowledge Base first — "
                       "the Gap Matrix compares that material against your target role.",
        }

    user_prompt = f"Target role: {target_role}\n\nResume / notes excerpt:\n{retrieval['context']}\n\nBuild the gap matrix now."
    content = rag_engine.raw_completion(GAP_MATRIX_SYSTEM, user_prompt, temperature=0.2)
    return {"status": "success", "target_role": target_role, "content": content}


def synthesize_resume_bullets(scope: str, target_role: Optional[str], focus_area: Optional[str]) -> Dict[str, Any]:
    query_bits = [b for b in [target_role, focus_area, "projects, achievements, and measurable results"] if b]
    retrieval = _resume_context(scope, " ".join(query_bits))
    if not retrieval["found"]:
        return {
            "status": "no_resume",
            "message": "Upload your resume or study/project notes in the Knowledge Base first — "
                       "bullets are synthesized from that material.",
        }

    context_line = f"Target role: {target_role}\n" if target_role else ""
    focus_line = f"Focus area: {focus_area}\n" if focus_area else ""
    user_prompt = f"{context_line}{focus_line}Source material:\n{retrieval['context']}\n\nWrite the bullets now."
    content = rag_engine.raw_completion(BULLET_SYSTEM, user_prompt, temperature=0.3)
    return {"status": "success", "content": content}


def list_scholarships() -> List[Dict[str, Any]]:
    return SCHOLARSHIPS


def match_scholarships(gpa: float, department: str, year: str, interests: Optional[str]) -> List[Dict[str, Any]]:
    """Deterministic, explainable rule-based scoring -- no LLM call needed,
    so this stays fast and free to run on every keystroke-adjacent action."""
    interest_words = set()
    if interests:
        interest_words = {w.strip().lower() for w in interests.replace(",", " ").split() if w.strip()}

    results = []
    for sch in SCHOLARSHIPS:
        score = 0
        reasons = []

        if gpa >= sch["min_gpa"]:
            score += 40
            reasons.append(f"GPA meets the {sch['min_gpa']} minimum")
        elif gpa >= sch["min_gpa"] - 0.3:
            score += 20
            reasons.append(f"GPA is close to the {sch['min_gpa']} minimum")
        else:
            reasons.append(f"GPA is below the {sch['min_gpa']} minimum")

        dept_ok = "Any" in sch["departments"] or (department and department in sch["departments"])
        if dept_ok:
            score += 30
            reasons.append("Department is eligible")

        year_ok = "Any" in sch["years"] or (year and year in sch["years"])
        if year_ok:
            score += 15
            reasons.append("Year of study is eligible")

        overlap = interest_words & set(sch["tags"])
        if interest_words:
            overlap_score = round(15 * (len(overlap) / max(1, len(interest_words))))
            score += overlap_score
            if overlap:
                reasons.append(f"Matches your interests: {', '.join(sorted(overlap))}")
        score = min(100, score)

        results.append({
            **sch,
            "match_score": score,
            "eligible": dept_ok and year_ok and gpa >= sch["min_gpa"],
            "reasons": reasons,
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results


def draft_scholarship_essay(scope: str, scholarship_name: str, essay_prompt: str) -> Dict[str, Any]:
    if not essay_prompt.strip():
        return {"status": "error", "message": "Essay prompt is required."}

    retrieval = _resume_context(
        scope, f"achievements, leadership, challenges, and experiences relevant to: {essay_prompt}"
    )
    if not retrieval["found"]:
        return {
            "status": "no_resume",
            "message": "Upload your resume or notes in the Knowledge Base first — the essay is "
                       "grounded in your own documented achievements, not generic filler.",
        }

    user_prompt = (
        f"Scholarship: {scholarship_name or 'General scholarship'}\n"
        f"Essay prompt: {essay_prompt}\n\n"
        f"Student's documented achievements/experience:\n{retrieval['context']}\n\n"
        f"Draft the essay now."
    )
    content = rag_engine.raw_completion(ESSAY_SYSTEM, user_prompt, temperature=0.4)
    return {"status": "success", "content": content}
