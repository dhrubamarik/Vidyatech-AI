"""
Faculty-facing quiz generation (AI-authored MCQs grounded in the
uploaded knowledge base) and submission grading.
"""
import json
import re
from typing import List, Dict, Any
from rag_service import rag_engine

QUIZ_GEN_SYSTEM = (
    "You write multiple-choice exam questions strictly from the given "
    "source material. Respond with ONLY a JSON array (no markdown fences, "
    "no commentary), where each item has exactly this shape: "
    '{"prompt": "...", "option_a": "...", "option_b": "...", '
    '"option_c": "...", "option_d": "...", "correct_option": "A", '
    '"explanation": "one sentence on why this is correct"}. '
    "correct_option must be one of \"A\", \"B\", \"C\", \"D\". "
    "Make questions unambiguous with exactly one correct option."
)


def _safe_json_list(text: str) -> List[Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def generate_questions(topic: str, scope: str, num_questions: int = 5) -> Dict[str, Any]:
    retrieval = rag_engine.retrieve_context(topic, scope, top_k=5)
    if not retrieval["found"]:
        return {"questions": [], "has_context": False, "message": f'No material on "{topic}" found in the knowledge base yet.'}

    user_prompt = (
        f"Topic: {topic}\nNumber of questions: {num_questions}\n\n"
        f"Source material:\n{retrieval['context']}\n\nGenerate the questions now."
    )
    raw = rag_engine.raw_completion(QUIZ_GEN_SYSTEM, user_prompt, temperature=0.4)
    questions = _safe_json_list(raw)

    clean_questions = []
    for q in questions[:max(num_questions, 1)]:
        if not all(k in q for k in ("prompt", "option_a", "option_b", "option_c", "option_d", "correct_option")):
            continue
        if q["correct_option"].upper() not in ("A", "B", "C", "D"):
            continue
        clean_questions.append({
            "prompt": q["prompt"],
            "option_a": q["option_a"],
            "option_b": q["option_b"],
            "option_c": q["option_c"],
            "option_d": q["option_d"],
            "correct_option": q["correct_option"].upper(),
            "explanation": q.get("explanation", ""),
        })

    return {"questions": clean_questions, "has_context": True, "message": "" if clean_questions else "The model didn't return usable questions — try again or a different topic."}


def grade_submission(questions: List[Dict[str, Any]], answers: Dict[str, str]) -> Dict[str, Any]:
    """`questions` is a list of dicts with at least id and correct_option.
    `answers` maps str(question_id) -> chosen letter."""
    if not questions:
        return {"score_pct": 0, "results": []}

    results = []
    correct_count = 0
    for q in questions:
        chosen = answers.get(str(q["id"]))
        is_correct = chosen == q["correct_option"]
        if is_correct:
            correct_count += 1
        results.append({
            "question_id": q["id"],
            "chosen": chosen,
            "correct_option": q["correct_option"],
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
        })

    score_pct = round((correct_count / len(questions)) * 100)
    return {"score_pct": score_pct, "results": results}
