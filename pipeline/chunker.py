"""
pipeline/chunker.py
-------------------
Splits parsed documents into retrievable chunks and enriches each with metadata:
  - topic:         inferred via keyword matching against an NLP vocabulary
  - difficulty:    estimated via jargon-density heuristic
  - content_type:  preserved from the parser ("pdf", "slide", "code", "markdown")
  - week:          inferred from source filename or passed explicitly
  - section_title: inferred heading/section label when available

Chunking strategy
-----------------
We prefer a simple structure-aware strategy first:
  1. split the document into paragraphs/blocks
  2. detect heading-like blocks
  3. build chunks that stay within section boundaries where possible

If the text is dense or contains very long blocks, we fall back to
RecursiveCharacterTextSplitter so we still respect size limits.

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


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def _looks_like_heading(paragraph: str) -> bool:
    """
    Heuristic heading detector for slide/PDF text.

    We treat short standalone lines as headings when they look title-like,
    are in all caps, or end with a colon.
    """
    compact = " ".join(paragraph.split())
    if not compact:
        return False

    if "\n" in paragraph and len(compact) > 90:
        return False

    words = compact.split()
    if len(words) > 12 or len(compact) > 90:
        return False

    if compact.endswith(":"):
        return True
    if compact.isupper() and len(words) >= 2:
        return True
    if compact.endswith((".", "?", "!")):
        return False

    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not alpha_words:
        return False

    title_like = sum(1 for w in alpha_words if w[:1].isupper())
    return title_like / len(alpha_words) >= 0.6


def _recursive_splits(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ": ", "; ", " ", ""],
    )
    return [split.strip() for split in splitter.split_text(text) if split.strip()]


def _split_large_paragraph(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]
    return _recursive_splits(paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _overlap_seed(parts: list[str], target_chars: int) -> list[str]:
    """Carry a small amount of trailing context into the next chunk."""
    seed: list[str] = []
    total = 0
    for part in reversed(parts):
        seed.insert(0, part)
        total += len(part) + 2
        if total >= target_chars:
            break
    return seed


def _infer_initial_section(paragraphs: list[str]) -> str | None:
    if paragraphs and _looks_like_heading(paragraphs[0]):
        return " ".join(paragraphs[0].split()).rstrip(":")
    return None


def _structure_aware_chunks(
    raw_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Chunk text by paragraphs and heading-like boundaries.

    Returns lightweight chunk specs with text + section metadata; the caller
    adds the rest of the standard metadata fields.
    """
    paragraphs = _paragraphs(raw_text)
    if not paragraphs:
        return []

    section_title = _infer_initial_section(paragraphs)
    buffer_parts: list[str] = []
    chunk_specs: list[dict] = []

    def flush_buffer() -> None:
        nonlocal buffer_parts
        if not buffer_parts:
            return
        text = "\n\n".join(buffer_parts).strip()
        if text:
            chunk_specs.append({
                "text": text,
                "section_title": section_title,
                "paragraph_count": len(buffer_parts),
            })
        buffer_parts = _overlap_seed(buffer_parts, chunk_overlap)

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph):
            flush_buffer()
            section_title = " ".join(paragraph.split()).rstrip(":")
            continue

        for piece in _split_large_paragraph(paragraph, chunk_size, chunk_overlap):
            candidate_parts = buffer_parts + [piece]
            candidate_text = "\n\n".join(candidate_parts)
            if buffer_parts and len(candidate_text) > chunk_size:
                flush_buffer()
            buffer_parts.append(piece)

    flush_buffer()
    return chunk_specs

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
    raw_text = doc.get("text", "").strip()
    if not raw_text:
        return []

    resolved_week = week if week is not None else infer_week(doc.get("source", ""))
    structured_splits = _structure_aware_chunks(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if structured_splits:
        split_specs = structured_splits
    else:
        split_specs = [
            {"text": split_text, "section_title": None, "paragraph_count": 1}
            for split_text in _recursive_splits(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ]

    chunks = []
    for i, split_spec in enumerate(split_specs):
        split_text = split_spec["text"]
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
                "section_title": split_spec.get("section_title"),
                "paragraph_count": split_spec.get("paragraph_count", 1),
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
