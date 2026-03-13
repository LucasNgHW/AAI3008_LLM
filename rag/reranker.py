"""
rag/reranker.py
---------------
Cross-encoder reranker that performs a second-pass scoring of first-stage
retrieved chunks using full query-passage attention.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - Takes (query, passage) pairs and outputs a relevance logit
  - More accurate than bi-encoder similarity; ~3–5× slower on CPU

Typical usage:
    chunks   = retriever.retrieve(query, top_k=20)      # over-retrieve
    reranked = reranker.rerank(query, chunks, top_n=5)  # prune to top 5

Latency note: ~600 ms/query on CPU with the 6-layer MiniLM model.
"""

from sentence_transformers.cross_encoder import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-score retrieved chunks with the cross-encoder and return the top_n.

    The input list is NOT mutated; each returned dict is a shallow copy with
    a new "rerank_score" key added alongside the original "score".

    Args:
        query:   The student's original question (not the augmented query).
        chunks:  First-stage retrieved chunks from retriever.retrieve().
        top_n:   Number of chunks to return after reranking.

    Returns:
        List of top_n chunk dicts sorted by descending rerank_score.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs    = [(query, chunk["text"]) for chunk in chunks]
    scores   = reranker.predict(pairs)   # numpy array of floats

    # Build copies so we never mutate the caller's list
    scored = []
    for chunk, score in zip(chunks, scores):
        entry = dict(chunk)
        entry["rerank_score"] = float(score)
        scored.append(entry)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_n]
