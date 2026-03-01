"""
evaluate.py
-----------
Measures retrieval quality against a hand-labelled QA evaluation set.

Metrics:
  - MRR (Mean Reciprocal Rank): measures how highly the first relevant chunk
    is ranked. MRR=1.0 means the correct chunk is always ranked first.
  - Recall@k: fraction of queries where at least one relevant chunk appears
    in the top-k results.

Evaluation set format (JSON):
  [
    {
      "question": "What is byte-pair encoding?",
      "relevant_sources": ["lecture_02_tokenisation.pdf"],  // filename substrings
      "relevant_topics":  ["tokenisation"]
    },
    ...
  ]

Usage:
    python evaluate.py --eval data/eval_set.json --top_k 5
    python evaluate.py --eval data/eval_set.json --top_k 5 --rerank
"""

import json
import argparse
from rag.retriever import retrieve
from rag.reranker  import rerank


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def is_relevant(chunk: dict, item: dict) -> bool:
    """
    Determine whether a retrieved chunk is relevant to an eval item.
    Relevance is satisfied if:
      - The chunk's source filename contains any of the relevant_sources substrings, OR
      - The chunk's topic matches any of the relevant_topics
    """
    source = (chunk.get("source") or "").lower()
    topic  = (chunk.get("topic") or "").lower()

    for rel_src in item.get("relevant_sources", []):
        if rel_src.lower() in source:
            return True
    for rel_topic in item.get("relevant_topics", []):
        if rel_topic.lower() == topic:
            return True
    return False


def evaluate(
    eval_set: list[dict],
    top_k: int = 5,
    use_rerank: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the evaluation and return a dict of metrics.

    Args:
        eval_set:   Loaded evaluation items.
        top_k:      Number of chunks to retrieve per query.
        use_rerank: Whether to apply cross-encoder reranking before scoring.
        verbose:    Print per-query results.

    Returns:
        {"mrr": float, f"recall@{top_k}": float, "n": int}
    """
    reciprocal_ranks = []
    hits             = []

    for i, item in enumerate(eval_set):
        question = item["question"]

        # Retrieve
        first_k = top_k * 3 if use_rerank else top_k
        chunks  = retrieve(question, top_k=first_k)
        if use_rerank and chunks:
            chunks = rerank(question, chunks, top_n=top_k)

        # Score
        rr   = 0.0
        hit  = 0
        for rank, chunk in enumerate(chunks, start=1):
            if is_relevant(chunk, item):
                if rr == 0.0:
                    rr = 1.0 / rank  # reciprocal rank of first relevant result
                hit = 1
                break

        reciprocal_ranks.append(rr)
        hits.append(hit)

        if verbose:
            status = "✓" if hit else "✗"
            print(f"[{status}] Q{i+1:02d} RR={rr:.2f} | {question[:70]}")

    mrr      = sum(reciprocal_ranks) / len(reciprocal_ranks)
    recall_k = sum(hits) / len(hits)
    n        = len(eval_set)

    print(f"\n{'='*50}")
    print(f"Evaluation Results  (n={n}, top_k={top_k}, rerank={use_rerank})")
    print(f"  MRR:          {mrr:.4f}")
    print(f"  Recall@{top_k}:    {recall_k:.4f}")
    print(f"{'='*50}\n")

    return {"mrr": mrr, f"recall@{top_k}": recall_k, "n": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("--eval",   required=True,        help="Path to eval set JSON file")
    parser.add_argument("--top_k",  type=int, default=5,  help="Number of results to retrieve")
    parser.add_argument("--rerank", action="store_true",  help="Apply cross-encoder reranking")
    parser.add_argument("--quiet",  action="store_true",  help="Suppress per-query output")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval)
    evaluate(eval_set, top_k=args.top_k, use_rerank=args.rerank, verbose=not args.quiet)
