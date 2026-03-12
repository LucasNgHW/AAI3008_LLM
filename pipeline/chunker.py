"""
pipeline/chunker.py
-------------------
Splits parsed documents into retrievable chunks and enriches each with metadata:
  - topic:        inferred via keyword matching against an NLP vocabulary
  - difficulty:   estimated via jargon-density heuristic
  - content_type: preserved from the parser ("pdf", "slide", "code", "markdown")
  - week:         passed in manually or inferred from source filename
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "tokenisation": ["token", "tokeniz", "tokenis", "bpe", "wordpiece", "subword", "byte pair", "sentencepiece", "vocabulary"],
    "transformers": ["transformer", "attention", "self-attention", "multi-head", "positional encoding", "encoder", "decoder", "bert", "gpt", "t5"],
    "language_models": ["language model", "n-gram", "perplexity", "lm", "neural lm", "autoregressive", "masked lm", "causal lm"],
    "sentiment": ["sentiment", "opinion mining", "polarity", "emotion", "aspect-based", "subjectivity", "vader"],
    "embeddings": ["embedding", "word2vec", "glove", "fasttext", "dense vector", "representation", "latent space", "semantic similarity"],
    "named_entity": ["named entity", "ner", "entity recognition", "entity extraction", "sequence labelling", "iob", "bio tagging"],
    "parsing": ["dependency", "constituency", "parse tree", "syntactic", "pos tag", "part of speech", "chunking"],
    "text_classification": ["classification", "categorisation", "naive bayes", "logistic regression", "text classifier", "softmax"],
}

BEGINNER_TERMS = ["example", "introduction", "overview", "definition", "basic", "simple", "what is"]
ADVANCED_TERMS = [
    "loss function", "gradient", "softmax", "backpropagation", "cross-entropy",
    "attention mechanism", "layer normalisation", "residual connection", "fine-tuning",
    "kullback-leibler", "variational", "monte carlo", "expectation-maximisation",
    "beam search", "contrastive loss", "negative sampling",
]


def infer_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    return max(scores, key=scores.get) if scores else "general"


def infer_difficulty(text: str) -> str:
    text_lower = text.lower()
    advanced_count = sum(1 for t in ADVANCED_TERMS if t in text_lower)
    beginner_count = sum(1 for t in BEGINNER_TERMS if t in text_lower)

    if advanced_count >= 3:
        return "advanced"
    if advanced_count >= 1:
        return "intermediate"
    if beginner_count >= 2:
        return "beginner"
    return "intermediate"


def infer_week(source: str) -> int | None:
    import re
    patterns = [r"week[_\-]?(\d+)", r"lecture[_\-]?(\d+)", r"wk[_\-]?(\d+)", r"w(\d{1,2})[-_]"]
    lower = source.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return int(m.group(1))
    return None


def chunk_document(doc: dict, week: int | None = None, chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ": ", "; ", " ", ""],
    )

    raw_text = doc.get("text", "").strip()
    if not raw_text:
        return []

    splits = splitter.split_text(raw_text)
    resolved_week = week if week is not None else infer_week(doc.get("source", ""))

    chunks = []
    for i, split_text in enumerate(splits):
        if not split_text.strip():
            continue
        chunks.append({
            "text": split_text,
            "metadata": {
                "source": doc.get("source"),
                "topic": infer_topic(split_text),
                "difficulty": infer_difficulty(split_text),
                "content_type": doc.get("content_type", "text"),
                "week": resolved_week,
                "slide": doc.get("slide"),
                "cell_type": doc.get("cell_type"),
                "chunk_index": i,
            }
        })
    return chunks


def chunk_documents(docs: list[dict], **kwargs) -> list[dict]:
    """Chunk a list of parsed document dicts."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, **kwargs)
        all_chunks.extend(chunks)
    print(f"Chunking complete: {len(docs)} documents → {len(all_chunks)} chunks")
    return all_chunks