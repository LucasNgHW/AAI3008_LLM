"""
rag/reflection.py
-----------------
Minimal retrieval reflection helpers.

This supports one simple retry loop:
  1. Observe the first retrieval result
  2. Decide whether it is empty or weak
  3. Rewrite the query into a clearer standalone search string
  4. Retrieve once more
"""

from __future__ import annotations

import re

RETRY_SCORE_THRESHOLD = 0.35

_LEADING_FILLER_PATTERNS = (
    r"^(can|could|would)\s+you\s+",
    r"^please\s+",
    r"^help\s+me\s+understand\s+",
    r"^i\s+want\s+to\s+know\s+about\s+",
    r"^tell\s+me\s+about\s+",
)


def top_score(chunks: list[dict]) -> float | None:
    """Return the best available score from the first retrieved chunk."""
    if not chunks:
        return None

    first = chunks[0]
    for key in ("rerank_score", "score"):
        value = first.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def should_retry_retrieval(chunks: list[dict], min_score: float = RETRY_SCORE_THRESHOLD) -> bool:
    """
    Retry when no chunks were found, or when the first-stage top score is weak.
    """
    score = top_score(chunks)
    return score is None or score < min_score


def _recent_user_context(conversation_history: list[dict] | None, max_turns: int = 2) -> str:
    if not conversation_history:
        return ""

    turns = [
        turn["content"].strip()
        for turn in conversation_history
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]
    return " ".join(turns[-max_turns:])


def rewrite_query_for_retry(query: str, conversation_history: list[dict] | None = None) -> str:
    """
    Turn a vague or conversational question into a cleaner standalone retrieval
    query using local heuristics only.
    """
    cleaned = " ".join(query.strip().split())
    lowered = cleaned.lower()

    for pattern in _LEADING_FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip(" ?")
    if not cleaned:
        cleaned = query.strip()

    history_context = _recent_user_context(conversation_history)
    if history_context and history_context.lower() not in lowered:
        return f"{history_context}. {cleaned}"

    return cleaned
