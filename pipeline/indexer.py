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
    Filter,
    FieldCondition,
    MatchValue,
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from project_paths import QDRANT_STORAGE_DIR

COLLECTION_NAME = "nlp_course"
VECTOR_DIM      = 384
STORAGE_PATH    = str(QDRANT_STORAGE_DIR)

KEYWORD_SCHEMA = getattr(PayloadSchemaType, "KEYWORD", "keyword")
INTEGER_SCHEMA = getattr(PayloadSchemaType, "INTEGER", "integer")

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
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

    # Local Qdrant ignores payload indexes, but server Qdrant benefits from them.
    for field, schema in [
        ("topic", KEYWORD_SCHEMA),
        ("difficulty", KEYWORD_SCHEMA),
        ("content_type", KEYWORD_SCHEMA),
        ("week", INTEGER_SCHEMA),
        ("material_id", INTEGER_SCHEMA),
        ("source", KEYWORD_SCHEMA),
    ]:
        try:
            client.create_payload_index(COLLECTION_NAME, field, schema)
        except Exception:
            pass


def index_chunks(chunks: list[dict], batch_size: int = 256) -> None:
    if not chunks:
        print("No chunks to index.")
        return
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
            "material_id":  chunk["metadata"].get("material_id"),
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
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name":          COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "status":        str(info.status),
        }
    except Exception:
        return {"name": COLLECTION_NAME, "vectors_count": 0, "status": "not found"}


def collection_exists() -> bool:
    """Return True if the Qdrant collection currently exists."""
    client = get_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception:
        return False
    return COLLECTION_NAME in existing


def _material_filter(material_id: int, source_label: str | None = None) -> Filter:
    should_conditions = [FieldCondition(key="material_id", match=MatchValue(value=material_id))]
    if source_label:
        should_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_label)))
    return Filter(should=should_conditions)


def count_material_chunks(material_id: int, source_label: str | None = None) -> int:
    """
    Count Qdrant points for one material.

    Returns 0 if the collection does not exist.
    """
    if not collection_exists():
        return 0

    client = get_client()
    material_filter = _material_filter(material_id, source_label)

    try:
        result = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=material_filter,
            exact=True,
        )
        return int(getattr(result, "count", result))
    except Exception:
        try:
            result = client.count(
                collection_name=COLLECTION_NAME,
                filter=material_filter,
                exact=True,
            )
            return int(getattr(result, "count", result))
        except Exception:
            return 0


def delete_material_chunks(material_id: int, source_label: str | None = None) -> bool:
    """
    Delete Qdrant points for one material.

    We match both `material_id` and `source` so this works for newly indexed
    chunks and also older DB-ingested chunks that only carried the source label.
    """
    if not collection_exists():
        return True

    client = get_client()
    material_filter = _material_filter(material_id, source_label)
    before_count = count_material_chunks(material_id, source_label)
    if before_count == 0:
        return True

    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=material_filter,
        )
    except Exception:
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=material_filter,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to delete Qdrant chunks for material {material_id}") from exc

    return count_material_chunks(material_id, source_label) == 0
