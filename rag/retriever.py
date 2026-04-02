"""
rag/retriever.py
----------------
Retrieves the top-k most relevant chunks from Qdrant for a given query.

Supports optional metadata filters:
  - topic_filter:      e.g. "transformers", "tokenisation"
  - difficulty_filter: "beginner" | "intermediate" | "advanced"
  - week_filter:       int, e.g. 3
  - source_filter:     exact source label stored in Qdrant

Results are returned as a list of flat dicts with text, score, and all payload fields.

Design note on difficulty filtering
------------------------------------
`retrieve_with_context` does NOT auto-inject a difficulty filter from the user
profile. The profile difficulty is for *prompt personalisation* only (generator.py).
Auto-filtering difficulty at retrieval time silently collapses recall — a beginner
asking about an advanced topic simply gets no results. Difficulty filtering is only
applied when the caller explicitly passes `difficulty_filter`.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from pipeline.embedder import embed_query
from pipeline.indexer  import get_client, COLLECTION_NAME

# Maximum characters from recent history appended to the query vector
_HISTORY_QUERY_CHARS = 200


def retrieve(
    query: str,
    top_k: int = 5,
    topic_filter: str | None = None,
    difficulty_filter: str | None = None,
    week_filter: int | None = None,
    source_filter: str | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Search the Qdrant collection for the most relevant chunks.

    Args:
        query:             Text to embed and search.
        top_k:             Number of results to return.
        topic_filter:      Restrict to a specific topic label.
        difficulty_filter: Restrict to a specific difficulty level.
        week_filter:       Restrict to content from a specific week number.
        source_filter:     Restrict to a specific stored source label.
        score_threshold:   Minimum cosine similarity (0–1). None = no threshold.

    Returns:
        List of dicts with keys: text, score, topic, difficulty,
        content_type, week, source, slide.
    """
    client  = get_client()
    qvector = embed_query(query).tolist()

    # Build compound metadata filter
    conditions = []
    if topic_filter:
        conditions.append(FieldCondition(key="topic",      match=MatchValue(value=topic_filter)))
    if difficulty_filter:
        conditions.append(FieldCondition(key="difficulty", match=MatchValue(value=difficulty_filter)))
    if week_filter is not None:
        conditions.append(FieldCondition(key="week",       match=MatchValue(value=week_filter)))
    if source_filter:
        conditions.append(FieldCondition(key="source",     match=MatchValue(value=source_filter)))

    qdrant_filter = Filter(must=conditions) if conditions else None

    # Use query_points (qdrant-client ≥1.7); fall back to legacy search if the
    # server/client version only exposes the older API.
    try:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        points = getattr(result, "points", result)
    except Exception:
        # query_points not available, fall back to legacy client.search
        try:
            points = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=qvector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    return [
        {
            "text":         p.payload.get("text", ""),
            "score":        round(p.score, 4),
            "topic":        p.payload.get("topic"),
            "difficulty":   p.payload.get("difficulty"),
            "content_type": p.payload.get("content_type"),
            "week":         p.payload.get("week"),
            "source":       p.payload.get("source"),
            "slide":        p.payload.get("slide"),
        }
        for p in points
    ]


def retrieve_with_context(
    query: str,
    conversation_history: list[dict] | None = None,
    user_profile: dict | None = None,
    top_k: int = 5,
    **filter_kwargs,
) -> list[dict]:
    """
    Retrieval with lightweight query augmentation from recent user turns.

    The last few user messages are prepended to the query string before
    embedding so that follow-up questions ("tell me more", "what about BERT")
    carry implicit context from the conversation.

    NOTE: `user_profile` is accepted for API compatibility but is NOT used to
    inject a difficulty filter — see module docstring for rationale. Callers
    that want difficulty-filtered results should pass `difficulty_filter`
    explicitly via filter_kwargs.

    Args:
        query:                The student's current question.
        conversation_history: Full conversation as [{"role": ..., "content": ...}].
        user_profile:         Accepted but unused for filtering (see note above).
        top_k:                Number of results to return.
        **filter_kwargs:      Forwarded verbatim to retrieve().

    Returns:
        Same structure as retrieve().
    """
    augmented_query = query
    if conversation_history:
        recent = conversation_history[-3:]
        context_str = " ".join(
            turn["content"] for turn in recent if turn.get("content")
        )
        augmented_query = f"{context_str} {query}"

    return retrieve(augmented_query, top_k=top_k, **filter_kwargs)
