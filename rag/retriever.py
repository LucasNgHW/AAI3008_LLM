"""
rag/retriever.py
----------------
Retrieves the top-k most relevant chunks from Qdrant for a given query.

Supports optional metadata filters:
  - topic_filter:      e.g. "transformers", "tokenisation"
  - difficulty_filter: "beginner" | "intermediate" | "advanced"
  - week_filter:       int, e.g. 3

Results are returned as a list of dicts with text, score, and all payload fields.
"""

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from pipeline.embedder import embed_query
from pipeline.indexer import get_client, COLLECTION_NAME


def retrieve(
    query: str,
    top_k: int = 5,
    topic_filter: str | None = None,
    difficulty_filter: str | None = None,
    week_filter: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Search the Qdrant collection for the most relevant chunks.

    Args:
        query:             Student's question, encoded at query time.
        top_k:             Number of results to return.
        topic_filter:      Restrict to a specific topic label.
        difficulty_filter: Restrict to a specific difficulty level.
        week_filter:       Restrict to content from a specific week.
        score_threshold:   Minimum cosine similarity score (0–1).

    Returns:
        List of dicts, each containing:
          - text, score, topic, difficulty, content_type, week, source
    """
    client  = get_client()
    qvector = embed_query(query).tolist()

    # Build compound metadata filter
    conditions = []
    if topic_filter:
        conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic_filter)))
    if difficulty_filter:
        conditions.append(FieldCondition(key="difficulty", match=MatchValue(value=difficulty_filter)))
    if week_filter is not None:
        conditions.append(FieldCondition(key="week", match=MatchValue(value=week_filter)))

    qdrant_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvector,
        limit=top_k,
        query_filter=qdrant_filter,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [
        {
            "text":         r.payload.get("text", ""),
            "score":        round(r.score, 4),
            "topic":        r.payload.get("topic"),
            "difficulty":   r.payload.get("difficulty"),
            "content_type": r.payload.get("content_type"),
            "week":         r.payload.get("week"),
            "source":       r.payload.get("source"),
            "slide":        r.payload.get("slide"),
        }
        for r in results
    ]


def retrieve_with_context(
    query: str,
    conversation_history: list[dict] | None = None,
    user_profile: dict | None = None,
    top_k: int = 5,
    **filter_kwargs,
) -> list[dict]:
    """
    Augmented retrieval that prepends conversation context to the query
    and applies profile-based difficulty filtering when no explicit filter
    is provided.

    Args:
        query:                The student's current question.
        conversation_history: Last N turns as [{"role": ..., "content": ...}].
        user_profile:         Loaded user profile dict from UserProfile.
        top_k:                Number of results to return.
        **filter_kwargs:      Passed through to retrieve().

    Returns:
        Same as retrieve().
    """
    # Build an augmented query that includes recent conversation context
    augmented_query = query
    if conversation_history:
        recent = conversation_history[-3:]  # last 3 turns for query augmentation
        context_str = " ".join(
            turn["content"] for turn in recent if turn.get("content")
        )
        augmented_query = f"{context_str} {query}"

    # Apply profile-based difficulty filter if no explicit one is set
    if user_profile and "difficulty_filter" not in filter_kwargs:
        preferred = user_profile.get("preferred_difficulty")
        if preferred:
            filter_kwargs.setdefault("difficulty_filter", preferred)

    return retrieve(augmented_query, top_k=top_k, **filter_kwargs)
