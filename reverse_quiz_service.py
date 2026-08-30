"""
"Proof-of-Understanding" reverse quiz: instead of picking an answer, the
student explains a concept in their own words and the AI grades that
explanation against the actual source material.
"""
import json
import re
from typing import Dict, Any
from rag_service import rag_engine

PROMPT_SYSTEM = (
    "You write a single short active-recall prompt for a student, in the "
    "form 'Explain <specific concept> in your own words (2-3 sentences).' "
    "Base it tightly on the given source material. Respond with ONLY the "
    "prompt sentence, nothing else."
)

EVAL_SYSTEM = (
    "You are grading a student's spoken/written explanation of a concept "
    "against reference source material. Respond with ONLY a JSON object "
    "(no markdown fences, no commentary) in exactly this shape: "
    '{"accuracy_score": <0-100 integer>, "missing_points": ["...", "..."], '
    '"misconceptions": ["...", "..."], "feedback": "one or two encouraging '
    'sentences on how to improve"}. missing_points and misconceptions may '
    "be empty arrays if there are none."
)


def _safe_json(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {
            "accuracy_score": 0,
            "missing_points": [],
            "misconceptions": [],
            "feedback": "Couldn't automatically score this one — the raw model output was: " + text[:300],
        }


def generate_prompt(topic: str, scope: str) -> Dict[str, Any]:
    retrieval = rag_engine.retrieve_context(topic, scope)
    if not retrieval["found"]:
        return {
            "topic": topic,
            "prompt": None,
            "has_context": False,
            "message": f"No material on \"{topic}\" found in your uploaded documents yet.",
        }
    user_prompt = f"Topic: {topic}\n\nSource material:\n{retrieval['context']}"
    question = rag_engine.raw_completion(PROMPT_SYSTEM, user_prompt)
    return {"topic": topic, "prompt": question.strip(), "has_context": True}


def evaluate_explanation(topic: str, explanation: str, scope: str) -> Dict[str, Any]:
    retrieval = rag_engine.retrieve_context(topic, scope)
    if not retrieval["found"]:
        return {
            "accuracy_score": 0,
            "missing_points": [],
            "misconceptions": [],
            "feedback": f"No material on \"{topic}\" is indexed, so this can't be graded against your notes.",
        }
    user_prompt = (
        f"Topic: {topic}\n\nSource material:\n{retrieval['context']}\n\n"
        f"Student's explanation:\n{explanation}\n\nGrade it now."
    )
    raw = rag_engine.raw_completion(EVAL_SYSTEM, user_prompt, temperature=0.2)
    return _safe_json(raw)
