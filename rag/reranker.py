"""
rag/reranker.py
---------------
Cross-encoder reranker that performs a second-pass scoring of first-stage
retrieved chunks using full query-passage attention.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - Takes (query, passage) pairs as input and outputs a relevance score
  - Significantly more accurate than bi-encoder similarity but ~3–5x slower

Usage:
  chunks  = retriever.retrieve(query, top_k=20)   # over-retrieve
  reranked = reranker.rerank(query, chunks, top_n=5)  # then rerank to top 5

Latency note (Week 8): adds ~600 ms per query on CPU. Mitigations being explored:
  - Reduce top_k passed to reranker (e.g. 10 → 5)
  - Cache reranker scores for repeated queries
  - Switch to a lighter bi-encoder rescoring model if latency remains unacceptable
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
    Re-score a list of retrieved chunks using the cross-encoder and return
    the top_n highest-scoring results.

    Args:
        query:   The student's original question.
        chunks:  Retrieved chunks from retriever.retrieve() (first-stage).
        top_n:   Number of chunks to return after reranking.

    Returns:
        List of top_n chunk dicts, sorted by descending reranker score.
        Each chunk gains a "rerank_score" field alongside the original "score".
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Cross-encoder expects (query, passage) pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)  # returns numpy array of floats

    # Attach reranker scores
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort descending by reranker score and return top_n
    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]
