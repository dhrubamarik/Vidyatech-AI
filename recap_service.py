"""
Generates a short, spoken-style recap of a topic from the student's own
uploaded material. The text comes back to the client, which reads it
aloud with the browser's built-in Web Speech API -- no audio file or
TTS service needed server-side.
"""
from typing import Dict, Any
from rag_service import rag_engine

RECAP_SYSTEM_PROMPT = (
    "You are creating a spoken audio recap for a student to listen to while "
    "reviewing. Write roughly 130-160 words (about 60 seconds spoken aloud), "
    "in plain conversational sentences with no markdown, no bullet points, "
    "and no headers -- just flowing spoken-style prose a text-to-speech "
    "engine can read naturally."
)


def build_recap(topic: str, scope: str) -> Dict[str, Any]:
    retrieval = rag_engine.retrieve_context(topic, scope)
    if not retrieval["found"]:
        return {
            "topic": topic,
            "recap": f"I couldn't find material on \"{topic}\" in your uploaded documents yet. "
                     f"Upload notes covering this topic in the Knowledge Base to generate a recap.",
            "has_context": False,
        }

    user_prompt = f"Topic: {topic}\n\nSource material:\n{retrieval['context']}\n\nWrite the spoken recap now."
    recap_text = rag_engine.raw_completion(RECAP_SYSTEM_PROMPT, user_prompt)
    return {"topic": topic, "recap": recap_text, "has_context": True}
