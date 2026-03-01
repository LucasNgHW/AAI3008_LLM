"""
ingest.py
---------
Top-level script to run the full data ingestion pipeline:

  1. Parse all documents in ./data/raw/
  2. Chunk and tag with metadata
  3. Generate embeddings
  4. Index into Qdrant

Usage:
    python ingest.py                        # full ingest from ./data/raw
    python ingest.py --dir path/to/files    # custom directory
    python ingest.py --recreate             # drop and rebuild the collection

Requirements:
    pip install unstructured[pdf] python-pptx nbformat sentence-transformers qdrant-client langchain
"""

import argparse
import time

from pipeline.parsers  import parse_directory
from pipeline.chunker  import chunk_documents
from pipeline.embedder import embed_chunks
from pipeline.indexer  import setup_collection, index_chunks, collection_info


def run_ingestion(data_dir: str = "./data/raw", recreate: bool = False) -> None:
    start = time.time()
    print(f"\n{'='*60}")
    print(f"NLP Assistant — Data Ingestion Pipeline")
    print(f"Source directory: {data_dir}")
    print(f"{'='*60}\n")

    # Step 1: Parse
    print("Step 1: Parsing documents...")
    docs = parse_directory(data_dir)
    print(f"  → {len(docs)} document sections extracted\n")

    # Step 2: Chunk
    print("Step 2: Chunking and tagging metadata...")
    chunks = chunk_documents(docs)
    print(f"  → {len(chunks)} chunks created\n")

    # Step 3: Embed
    print("Step 3: Generating embeddings...")
    chunks = embed_chunks(chunks)
    print(f"  → Embeddings complete\n")

    # Step 4: Index
    print("Step 4: Indexing into Qdrant...")
    setup_collection(recreate=recreate)
    index_chunks(chunks)

    # Summary
    info    = collection_info()
    elapsed = round(time.time() - start, 1)
    print(f"\n{'='*60}")
    print(f"Ingestion complete in {elapsed}s")
    print(f"Collection '{info['name']}': {info['vectors_count']} vectors indexed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NLP Assistant ingestion pipeline.")
    parser.add_argument("--dir",      default="./data/raw", help="Directory containing raw course files")
    parser.add_argument("--recreate", action="store_true",  help="Drop and recreate the Qdrant collection")
    args = parser.parse_args()

    run_ingestion(data_dir=args.dir, recreate=args.recreate)
