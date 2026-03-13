"""
rag/reranker.py
---------------
Cross-encoder reranker with LRU result cache to eliminate redundant forward
passes on repeated queries.

Performance budget
------------------
Without cache : ~600 ms per call on CPU (15 query-passage pairs, 6-layer MiniLM)
With cache    : ~0 ms on repeated identical queries within the same session

Cache design
------------
- Key   : (query_text, tuple(chunk["text"] for chunk in chunks), top_n)
- Value : list of reranked chunk dicts (copies, so callers cannot mutate cache)
- Size  : 128 entries (covers a full conversation session; ~10 MB max)
- Scope : process-level singleton — shared across Streamlit reruns in the same
          worker process

Warmup
------
Call `warmup()` at app startup to load the CrossEncoder weights before the
first user query.  Load time is ~2-3 s cold; negligible once warm.
"""

import hashlib
from functools import lru_cache
from sentence_transformers.cross_encoder import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CACHE_SIZE    = 128   # LRU slots

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def warmup() -> None:
    """
    Load the CrossEncoder and run one dummy prediction so the first real
    query does not pay the cold-start penalty.  Call once at app startup.
    """
    re = get_reranker()
    re.predict([("warmup query", "warmup passage")])
    print(f"Reranker warm ({RERANKER_MODEL})")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(query: str, chunks: list[dict], top_n: int) -> tuple:
    """Build a hashable cache key from the rerank inputs."""
    texts = tuple(c.get("text", "") for c in chunks)
    return (query, texts, top_n)


# We use a module-level dict instead of functools.lru_cache so we can store
# arbitrary objects (lists of dicts) and inspect/clear the cache if needed.
_cache: dict[tuple, list[dict]] = {}
_cache_order: list[tuple] = []   # insertion-order queue for LRU eviction


def _cache_get(key: tuple) -> list[dict] | None:
    return _cache.get(key)


def _cache_put(key: tuple, value: list[dict]) -> None:
    if key in _cache:
        _cache_order.remove(key)
    elif len(_cache) >= _CACHE_SIZE:
        oldest = _cache_order.pop(0)
        del _cache[oldest]
    _cache[key] = value
    _cache_order.append(key)


def clear_cache() -> None:
    """Empty the rerank cache (useful between eval runs)."""
    _cache.clear()
    _cache_order.clear()


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-score retrieved chunks with the cross-encoder and return the top_n.

    Results are cached by (query, chunk texts, top_n).  On a cache hit the
    forward pass is skipped entirely (~0 ms).

    The input list is NOT mutated; each returned dict is a shallow copy with
    a new "rerank_score" key alongside the original "score".

    Args:
        query:   The student's original question (not the augmented query).
        chunks:  First-stage retrieved chunks from retriever.retrieve().
        top_n:   Number of chunks to return after reranking.

    Returns:
        List of top_n chunk dicts sorted by descending rerank_score.
    """
    if not chunks:
        return []

    key    = _cache_key(query, chunks, top_n)
    cached = _cache_get(key)
    if cached is not None:
        # Return copies so callers cannot accidentally mutate cached state
        return [dict(c) for c in cached]

    reranker = get_reranker()
    pairs    = [(query, chunk["text"]) for chunk in chunks]
    scores   = reranker.predict(pairs)

    scored = []
    for chunk, score in zip(chunks, scores):
        entry = dict(chunk)
        entry["rerank_score"] = float(score)
        scored.append(entry)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    result = scored[:top_n]

    _cache_put(key, result)
    return [dict(c) for c in result]
