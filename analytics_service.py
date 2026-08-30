"""
Faculty department analytics.

Turns raw ChatHistory rows into the numbers a faculty member actually
wants: how many students are active, how many doubts the agents have
resolved, which topics keep coming up (derived from real query text,
not a fixed list), and how heavily the knowledge base is being used
relative to how much content has been uploaded for it.
"""
import re
from collections import Counter
from typing import List, Dict, Any

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "does", "do", "did", "can", "could", "would", "should",
    "explain", "please", "help", "with", "for", "and", "or", "to", "of",
    "in", "on", "this", "that", "it", "me", "my", "i", "you", "your", "we",
    "about", "between", "difference", "give", "tell", "define", "meaning",
}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")


def top_doubt_clusters(queries: List[str], top_n: int = 3) -> List[str]:
    """Extract the most frequent meaningful terms/phrases across all
    student queries, as a stand-in for topic clustering."""
    counts: Counter = Counter()

    for q in queries:
        words = [w.lower() for w in WORD_RE.findall(q) if w.lower() not in STOPWORDS]
        # unigrams
        counts.update(words)
        # bigrams read a bit more like a "topic" than single words
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            counts[bigram] += 1

    if not counts:
        return []

    # Prefer bigrams (more descriptive) when they're reasonably frequent,
    # falling back to unigrams to fill out the list.
    ranked = counts.most_common(20)
    bigrams = [(term, c) for term, c in ranked if " " in term]
    unigrams = [(term, c) for term, c in ranked if " " not in term]

    picks = []
    for term, _ in bigrams + unigrams:
        label = term.title()
        if label not in picks:
            picks.append(label)
        if len(picks) >= top_n:
            break

    return picks


def resource_utilization(total_queries: int, total_chunks_indexed: int) -> str:
    """A simple utilization ratio: queries served per indexed chunk,
    expressed as a percentage capped at 100. More queries relative to a
    small knowledge base -> higher utilization."""
    if total_chunks_indexed == 0:
        return "0%"
    ratio = (total_queries / total_chunks_indexed) * 100
    return f"{min(round(ratio), 100)}%"
