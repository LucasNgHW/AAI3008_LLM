"""
pipeline/material_ingestion.py
------------------------------
Minimal helpers to ingest PDFs stored in SQLite into Qdrant.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pipeline.chunker import chunk_documents
from pipeline.embedder import embed_chunks
from pipeline.indexer import (
    COLLECTION_NAME,
    collection_exists,
    delete_material_chunks,
    get_client,
    index_chunks,
    setup_collection,
)
from pipeline.parsers import parse_file
from storage.materials_db import (
    count_materials,
    delete_all_materials,
    delete_material,
    get_material,
    list_materials,
)


def _db_source_label(material: dict) -> str:
    return f"db://materials/{material['id']}/{material['filename']}"


def ingest_material(material_id: int, recreate: bool = False) -> int:
    """
    Ingest one stored PDF into Qdrant.

    Returns the number of chunks indexed.
    """
    material = get_material(material_id)
    if material is None:
        raise ValueError(f"Material id {material_id} not found")

    suffix = Path(material["filename"]).suffix or ".pdf"
    temp_path = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(material["content"])
            temp_path = tmp.name

        docs = parse_file(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    for doc in docs:
        doc["source"] = _db_source_label(material)

    chunks = chunk_documents(docs)
    for chunk in chunks:
        chunk["metadata"]["material_id"] = material["id"]
    chunks = embed_chunks(chunks, show_progress=False)

    setup_collection(recreate=recreate)
    index_chunks(chunks)
    return len(chunks)


def ingest_all_materials(recreate: bool = False) -> int:
    """
    Ingest all stored PDFs into Qdrant.

    Returns the total number of chunks indexed.
    """
    materials = list_materials()
    total_chunks = 0
    first = True

    for material in materials:
        total_chunks += ingest_material(
            material["id"],
            recreate=(recreate and first),
        )
        first = False

    return total_chunks


def sync_qdrant_with_db() -> dict:
    """
    Rebuild Qdrant from the current SQLite contents.

    This is the safest way to keep retrieval aligned with the database after
    uploads or deletions because it removes any stale vectors left behind by
    older indexing runs.
    """
    material_count = count_materials()

    if material_count == 0:
        client = get_client()
        if collection_exists():
            client.delete_collection(COLLECTION_NAME)
        return {
            "ok": True,
            "material_count": 0,
            "chunk_count": 0,
            "message": "Qdrant cleared because the materials database is empty.",
        }

    chunk_count = ingest_all_materials(recreate=True)
    return {
        "ok": True,
        "material_count": material_count,
        "chunk_count": chunk_count,
        "message": f"Rebuilt Qdrant from {material_count} stored material(s).",
    }


def delete_material_everywhere(material_id: int) -> dict:
    """
    Delete one material from both SQLite and Qdrant.

    Returns a status dict so the UI can report whether Qdrant cleanup was
    verified before the SQLite row was removed.
    """
    material = get_material(material_id)
    if material is None:
        return {
            "deleted": False,
            "filename": None,
            "reason": "missing",
            "message": "Material not found in the database.",
        }

    filename = material["filename"]
    source_label = _db_source_label(material)

    try:
        qdrant_deleted = delete_material_chunks(material_id, source_label=source_label)
    except Exception as exc:
        return {
            "deleted": False,
            "filename": filename,
            "reason": "qdrant_error",
            "message": f"Qdrant deletion failed for {filename}: {exc}",
        }

    if not qdrant_deleted:
        return {
            "deleted": False,
            "filename": filename,
            "reason": "qdrant_verify_failed",
            "message": f"Could not verify Qdrant cleanup for {filename}; the database record was kept.",
        }

    deleted_in_db = delete_material(material_id)
    if not deleted_in_db:
        return {
            "deleted": False,
            "filename": filename,
            "reason": "db_delete_failed",
            "message": f"Qdrant chunks were removed but the database row for {filename} could not be deleted.",
        }

    return {
        "deleted": True,
        "filename": filename,
        "reason": "ok",
        "message": f"Deleted {filename} from the database and Qdrant.",
    }


def delete_all_materials_everywhere() -> dict:
    """
    Delete all materials from SQLite and clear the Qdrant collection.

    Returns a status dict so the UI can report whether Qdrant cleanup was
    verified before database rows were removed.
    """
    client = get_client()
    if collection_exists():
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception as exc:
            return {
                "deleted": False,
                "deleted_count": 0,
                "reason": "qdrant_error",
                "message": f"Failed to clear the Qdrant collection: {exc}",
            }

        if collection_exists():
            return {
                "deleted": False,
                "deleted_count": 0,
                "reason": "qdrant_verify_failed",
                "message": "Could not verify that the Qdrant collection was cleared; database rows were kept.",
            }

    deleted_count = delete_all_materials()

    return {
        "deleted": True,
        "deleted_count": deleted_count,
        "reason": "ok",
        "message": f"Deleted {deleted_count} material(s) and cleared the Qdrant collection.",
    }
