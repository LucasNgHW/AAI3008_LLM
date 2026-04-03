"""
rag/router.py
-------------
Lightweight routing for deciding whether a user query should:
  - use the RAG pipeline, or
  - be answered directly without retrieval.

The router is intentionally conservative: unless a query is clearly a greeting,
help request, or app-capability question, it defaults to RAG.
"""

from __future__ import annotations


_DIRECT_PREFIXES = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks",
    "what can you do",
    "how can you help",
    "how do i use this",
    "how should i use this",
    "who are you",
    "help me use",
)

_DIRECT_EXACT = {
    "help",
    "thanks",
    "thank you",
}

_COURSE_HINTS = {
    "lecture", "lectures", "slide", "slides", "pdf", "pdfs", "material",
    "materials", "course", "notes", "chapter", "topic", "bert", "gpt",
    "transformer", "transformers", "tokenisation", "tokenization",
    "embedding", "embeddings", "attention", "sentiment", "nlp",
}

_QUESTION_STARTERS = (
    "what is",
    "what are",
    "explain",
    "define",
    "compare",
    "summarise",
    "summarize",
    "how does",
    "why does",
    "tell me about",
)


def route_query(
    query: str,
    selected_source: str | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Return 'direct' for obvious conversational/help queries, else 'rag'.

    If the user has selected a specific course material filter, always route to
    RAG because that expresses explicit intent to search the knowledge base.
    """
    if selected_source:
        return "rag"

    normalized = " ".join(query.lower().strip().split())
    if not normalized:
        return "direct"

    if normalized in _DIRECT_EXACT:
        return "direct"

    if normalized.startswith(_DIRECT_PREFIXES):
        for starter in _QUESTION_STARTERS:
            if normalized.startswith(starter):
                return "rag"
        return "direct"

    tokens = set(normalized.replace("?", " ").replace(",", " ").split())
    if tokens & _COURSE_HINTS:
        return "rag"

    if any(normalized.startswith(starter) for starter in _QUESTION_STARTERS):
        return "rag"

    # Very short social turns like "ok", "cool", or "nice" should not trigger retrieval.
    if len(tokens) <= 3 and not tokens & _COURSE_HINTS:
        return "direct"

    return "rag"
