"""
pipeline/embedder.py
--------------------
Generates dense vector embeddings for text chunks using
sentence-transformers/all-MiniLM-L6-v2 (384-dimensional, local, no API cost).

The model is loaded once at module level and reused across calls.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load once; reused for both indexing and query-time encoding
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised 384-dim embeddings.

    Args:
        texts:         Strings to encode.
        batch_size:    Sentences processed per forward pass.
        show_progress: Print tqdm progress bar.

    Returns:
        np.ndarray of shape (len(texts), 384).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine sim == dot product after L2 norm
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string. Returns shape (384,)."""
    return embed_texts([query], show_progress=False)[0]


def embed_chunks(chunks: list[dict], **kwargs) -> list[dict]:
    """
    Add an 'embedding' key to each chunk dict in-place.

    Args:
        chunks: Output from chunker.chunk_documents().

    Returns:
        Same list with 'embedding' added to each dict.
    """
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts, **kwargs)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    print("Embedding complete.")
    return chunks
