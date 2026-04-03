"""
pipeline/embedder.py
--------------------
Generates dense vector embeddings using sentence-transformers/all-MiniLM-L6-v2
(384-dimensional, local, no API cost).

Performance notes
-----------------
The model singleton is loaded once and reused.  Call `warmup()` at application
start (ingest.py and ui.py both do this) so the ~1-2 s model-load cost is paid
upfront rather than on the first real query.

embed_query() is the hot path: ~5 ms on CPU once warm.  batch_size=1 skips
padding overhead for single-query calls; show_progress_bar=False removes tqdm.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def warmup() -> None:
    """
    Load the model and run one dummy encode so the first real query
    does not pay the cold-start penalty.  Call once at app startup.
    """
    model = get_model()
    model.encode(["warmup"], show_progress_bar=False, convert_to_numpy=True)
    print(f"Embedding model warm ({MODEL_NAME})")


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised 384-dim embeddings.
    Returns np.ndarray of shape (len(texts), 384).
    """
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string.  Returns shape (384,).
    batch_size=1 skips padding; show_progress_bar=False removes tqdm overhead.
    """
    model = get_model()
    return model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


def embed_chunks(
    chunks: list[dict],
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[dict]:
    """Add an embedding key to each chunk dict in-place."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(
        texts,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    print("Embedding complete.")
    return chunks
