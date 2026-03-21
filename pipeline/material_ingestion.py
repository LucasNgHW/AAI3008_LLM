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
from pipeline.indexer import COLLECTION_NAME, delete_material_chunks, get_client, index_chunks, setup_collection
from pipeline.parsers import parse_file
from storage.materials_db import delete_all_materials, delete_material, get_material, list_materials


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


def delete_material_everywhere(material_id: int) -> bool:
    """
    Delete one material from both SQLite and Qdrant.
    """
    material = get_material(material_id)
    if material is None:
        return False

    delete_material_chunks(material_id, source_label=_db_source_label(material))
    return delete_material(material_id)


def delete_all_materials_everywhere() -> int:
    """
    Delete all materials from SQLite and clear the Qdrant collection.

    Returns the number of SQLite material rows deleted.
    """
    deleted_count = delete_all_materials()

    client = get_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return deleted_count
