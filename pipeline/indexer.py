"""
pipeline/indexer.py
-------------------
Indexes embedded chunks into a local Qdrant collection.

Collection schema:
  - Vector:  384-dim cosine similarity (from all-MiniLM-L6-v2)
  - Payload: text, source, topic, difficulty, content_type, week, slide, cell_type, chunk_index

Qdrant is run in local (on-disk) mode during development.
To switch to a hosted Qdrant Cloud instance, replace QdrantClient(path=...)
with QdrantClient(url="https://...", api_key="...").
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

COLLECTION_NAME = "nlp_course"
VECTOR_DIM      = 384
STORAGE_PATH    = "./qdrant_storage"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=STORAGE_PATH)
    return _client


def setup_collection(recreate: bool = False) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Args:
        recreate: If True, drop and recreate the collection (wipes existing data).
    """
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if recreate and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")
        existing = []

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        # Create payload indices to speed up filtered search
        for field, schema in [
            ("topic",        PayloadSchemaType.KEYWORD),
            ("difficulty",   PayloadSchemaType.KEYWORD),
            ("content_type", PayloadSchemaType.KEYWORD),
            ("week",         PayloadSchemaType.INTEGER),
        ]:
            client.create_payload_index(COLLECTION_NAME, field, schema)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def index_chunks(chunks: list[dict], batch_size: int = 256) -> None:
    """
    Upsert a list of embedded chunk dicts into Qdrant.

    Each chunk must have:
      - "text":      str
      - "embedding": list[float]  (length 384)
      - "metadata":  dict

    Args:
        chunks:     Output of embedder.embed_chunks().
        batch_size: Number of points per upsert call.
    """
    client = get_client()

    points = []
    for chunk in chunks:
        payload = {
            "text":         chunk["text"],
            "source":       chunk["metadata"].get("source"),
            "topic":        chunk["metadata"].get("topic"),
            "difficulty":   chunk["metadata"].get("difficulty"),
            "content_type": chunk["metadata"].get("content_type"),
            "week":         chunk["metadata"].get("week"),
            "slide":        chunk["metadata"].get("slide"),
            "cell_type":    chunk["metadata"].get("cell_type"),
            "chunk_index":  chunk["metadata"].get("chunk_index"),
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["embedding"],
            payload=payload,
        ))

    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Indexed {min(i + batch_size, total)}/{total} points")

    print(f"Indexing complete: {total} points in '{COLLECTION_NAME}'")


def collection_info() -> dict:
    """Return basic stats about the collection."""
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)

    vectors_count = getattr(info, "vectors_count", None)
    points_count  = getattr(info, "points_count", None)

    return {
        "name": COLLECTION_NAME,
        "vectors_count": vectors_count if vectors_count is not None else points_count,
        "status": getattr(info, "status", None),
    }
