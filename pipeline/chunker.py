"""
pipeline/chunker.py
-------------------
Splits parsed documents into retrievable chunks and enriches each with metadata:
  - topic:        inferred via keyword matching against an NLP vocabulary
  - difficulty:   estimated via jargon-density heuristic
  - content_type: preserved from the parser ("pdf", "slide", "code", "markdown")
  - week:         inferred from source filename or passed explicitly

Import compatibility
--------------------
RecursiveCharacterTextSplitter moved from `langchain` to `langchain-text-splitters`
in LangChain ≥0.1. We try the modern path first and fall back to the legacy one.
"""

import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "RecursiveCharacterTextSplitter not found. "
            "Install with: pip install langchain-text-splitters"
        ) from exc

# ── Topic taxonomy ─────────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "tokenisation": [
        "token", "tokeniz", "tokenis", "bpe", "wordpiece", "subword",
        "byte pair", "sentencepiece", "vocabulary",
    ],
    "transformers": [
        "transformer", "attention", "self-attention", "multi-head",
        "positional encoding", "encoder", "decoder", "bert", "gpt", "t5",
    ],
    "language_models": [
        "language model", "n-gram", "perplexity", "lm", "neural lm",
        "autoregressive", "masked lm", "causal lm",
    ],
    "sentiment": [
        "sentiment", "opinion mining", "polarity", "emotion",
        "aspect-based", "subjectivity", "vader",
    ],
    "embeddings": [
        "embedding", "word2vec", "glove", "fasttext", "dense vector",
        "representation", "latent space", "semantic similarity",
    ],
    "named_entity": [
        "named entity", "ner", "entity recognition", "entity extraction",
        "sequence labelling", "iob", "bio tagging",
    ],
    "parsing": [
        "dependency", "constituency", "parse tree", "syntactic",
        "pos tag", "part of speech", "chunking",
    ],
    "text_classification": [
        "classification", "categorisation", "naive bayes",
        "logistic regression", "text classifier", "softmax",
    ],
}

BEGINNER_TERMS = [
    "example", "introduction", "overview", "definition",
    "basic", "simple", "what is",
]
ADVANCED_TERMS = [
    "loss function", "gradient", "softmax", "backpropagation", "cross-entropy",
    "attention mechanism", "layer normalisation", "residual connection",
    "fine-tuning", "kullback-leibler", "variational", "monte carlo",
    "expectation-maximisation", "beam search", "contrastive loss",
    "negative sampling",
]

# ── Inference helpers ──────────────────────────────────────────────────────────

def infer_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {
        topic: sum(1 for kw in keywords if kw in text_lower)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def infer_difficulty(text: str) -> str:
    text_lower     = text.lower()
    advanced_count = sum(1 for t in ADVANCED_TERMS if t in text_lower)
    beginner_count = sum(1 for t in BEGINNER_TERMS if t in text_lower)

    if advanced_count >= 2:
        return "advanced"
    if advanced_count >= 1:
        return "intermediate"
    if beginner_count >= 2:
        return "beginner"
    return "intermediate"


_WEEK_PATTERNS = [
    r"week[_\-]?(\d+)",
    r"lecture[_\-]?(\d+)",
    r"wk[_\-]?(\d+)",
    r"w(\d{1,2})[-_]",
]


def infer_week(source: str) -> int | None:
    lower = source.lower()
    for pat in _WEEK_PATTERNS:
        m = re.search(pat, lower)
        if m:
            return int(m.group(1))
    return None


# ── Main chunking functions ────────────────────────────────────────────────────

def chunk_document(
    doc: dict,
    week: int | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Split a single parsed document dict into retrievable chunks.

    Args:
        doc:          Output dict from pipeline/parsers.py.
        week:         Override week number; inferred from filename if None.
        chunk_size:   Target character length per chunk.
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        List of chunk dicts, each with "text" and "metadata" keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ": ", "; ", " ", ""],
    )

    raw_text = doc.get("text", "").strip()
    if not raw_text:
        return []

    splits        = splitter.split_text(raw_text)
    resolved_week = week if week is not None else infer_week(doc.get("source", ""))

    chunks = []
    for i, split_text in enumerate(splits):
        if not split_text.strip():
            continue
        chunks.append({
            "text": split_text,
            "metadata": {
                "source":       doc.get("source"),
                "topic":        infer_topic(split_text),
                "difficulty":   infer_difficulty(split_text),
                "content_type": doc.get("content_type", "text"),
                "week":         resolved_week,
                "slide":        doc.get("slide"),
                "cell_type":    doc.get("cell_type"),
                "chunk_index":  i,
            },
        })
    return chunks


def chunk_documents(docs: list[dict], **kwargs) -> list[dict]:
    """Chunk a list of parsed document dicts."""
    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, **kwargs))
    print(f"Chunking complete: {len(docs)} documents → {len(all_chunks)} chunks")
    return all_chunks
