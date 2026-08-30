"""
Builds a compact, one-page "cheat sheet" -- key formulas, definitions,
and likely exam questions -- from the material behind a student's
current planner topics. Reuses the same Markdown-ish output shape as
chat responses, so it can go straight through export_service.py.
"""
from typing import List
from rag_service import rag_engine

CHEATSHEET_SYSTEM = (
    "You build a compact one-page exam cheat sheet from source material. "
    "Structure it in Markdown with a '## Key Definitions' section, a "
    "'## Key Formulas' section (if any exist in the material), and a "
    "'## Likely Exam Questions' section with 3 short predicted questions "
    "and one-line answers. Be dense and compact -- short bullets, no "
    "filler, no long paragraphs. Only use what's in the source material."
)


def build_cheatsheet(topics: List[str], scope: str, max_chars: int = 6000) -> str:
    if not topics:
        return "No topics on your schedule yet — import a syllabus or add tasks in Planner first."

    collected = []
    seen_chunks = set()
    for topic in topics:
        retrieval = rag_engine.retrieve_context(topic, scope, top_k=3)
        if retrieval["found"] and retrieval["context"] not in seen_chunks:
            seen_chunks.add(retrieval["context"])
            collected.append(f"### {topic}\n{retrieval['context']}")

    if not collected:
        return "Your uploaded documents don't seem to cover your scheduled topics yet — upload notes in the Knowledge Base first."

    combined = "\n\n".join(collected)[:max_chars]
    user_prompt = f"Source material, organized by topic:\n\n{combined}\n\nBuild the cheat sheet now."
    return rag_engine.raw_completion(CHEATSHEET_SYSTEM, user_prompt, temperature=0.25)
