# Personalized NLP Learning Assistant using RAG

A Streamlit-based learning assistant for NLP course materials.  
The app stores uploaded PDFs in SQLite, indexes chunk embeddings in Qdrant, retrieves relevant course content, and generates grounded answers with Gemini.

## Overview

This project was built for **AAI3008 Large Language Models** under the idea:

> **Idea 3: Personalized NLP Learning Assistant using RAG**

The assistant is designed to:

- store course materials in a structured database
- retrieve relevant content from a vector database
- support multi-turn course-related dialogue
- adapt answer style based on a simple student profile

## Features

- **Structured material storage**
  Uploaded PDFs are stored in a local SQLite database.
- **Vector retrieval with Qdrant**
  Course content is chunked, embedded, and indexed for semantic search.
- **Reranking**
  Retrieved chunks can be reranked with a cross-encoder for better relevance.
- **Multi-turn dialogue**
  Follow-up questions can use recent conversation context.
- **Simple personalization**
  The app tracks recent interaction difficulty and topic history to adjust explanation style.
- **Material management**
  Stored PDFs can be viewed and deleted from the UI, with matching cleanup in Qdrant.
- **Retry / reflection step**
  If the first retrieval is weak, the app rewrites the query once and retries retrieval.
- **Structure-aware chunking**
  Chunking now respects paragraph and heading boundaries where possible, instead of using only fixed character windows.

## Architecture

```text
PDF Upload -> SQLite -> Parsing -> Chunking -> Embeddings -> Qdrant
                                                |
User Query -> Routing -> Retrieval -> Rerank -> Gemini Answer
                          ^                    |
                          |---- retry once ----|
```

## Tech Stack

| Component | Choice |
|---|---|
| UI | Streamlit |
| LLM | Gemini (`google-genai`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | Qdrant (local mode) |
| Material Store | SQLite |

## Project Structure

```text
AAI3008_LLM/
├── app/
│   ├── ui.py              # Main Streamlit page
│   ├── sidebar.py         # Sidebar rendering and actions
│   ├── onboarding.py      # First-run upload flow
│   └── ui_helpers.py      # Shared UI helpers
├── data/
│   ├── raw/               # Optional raw course files for CLI ingest
│   └── materials.db       # SQLite database for uploaded PDFs
├── personalisation/
│   └── user_profile.py    # Per-student profile and difficulty tracking
├── pipeline/
│   ├── parsers.py         # PDF / PPTX / notebook parsing
│   ├── chunker.py         # Structure-aware chunking + metadata
│   ├── embedder.py        # Embedding model loading and encoding
│   ├── indexer.py         # Qdrant collection setup and indexing
│   └── material_ingestion.py
│                          # SQLite -> parser -> chunk -> embed -> Qdrant
├── rag/
│   ├── router.py          # Direct vs RAG vs out-of-scope routing
│   ├── retriever.py       # Retrieval with filters and follow-up context
│   ├── reranker.py        # Cross-encoder reranking
│   ├── reflection.py      # One-step retry / query rewrite
│   └── generator.py       # Prompt assembly and streaming generation
├── storage/
│   └── materials_db.py    # SQLite helpers for stored course materials
├── qdrant_storage/        # Local Qdrant data
├── profiles/              # Per-user JSON profiles
├── ingest.py              # CLI ingestion entry point
├── evaluate.py            # Retrieval evaluation script
├── project_paths.py       # Shared project-root paths
└── requirements.txt
```

## Setup

### 1. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
```
#### For Windows:
```bash
python -m venv .venv
source .venv/Scripts/activate
```

Or use Conda if preferred.

### 2. Install dependencies (Might take a while to download)

```bash
python -m pip install -r requirements.txt
```

### 3. Set your Gemini API key

```bash
export GEMINI_API_KEY=your_key_here
```

## Run the App

```bash
streamlit run app/ui.py
```

On first launch, the app will ask you to upload course PDFs.  
Those files are:

- stored in **SQLite**
- parsed and chunked
- embedded locally
- indexed into **Qdrant**

After indexing completes, the chat interface becomes available.

## CLI Ingestion

You can still ingest from the raw data folder:

```bash
python ingest.py
python ingest.py --recreate
python ingest.py --dir path/to/files
```

## Retrieval Evaluation

If you prepare an evaluation set in `data/eval_set.json`, you can run:

```bash
python evaluate.py --eval data/eval_set.json --top_k 5
python evaluate.py --eval data/eval_set.json --top_k 5 --rerank
```

## How the Assistant Responds

The app uses a small router before answering:

- **Direct**: greetings or app-help questions
- **RAG**: course questions grounded in uploaded materials
- **Out of scope**: unrelated questions are refused instead of answered loosely

For course questions, the flow is:

1. Retrieve relevant chunks from Qdrant
2. Retry once with a rewritten query if retrieval looks weak
3. Optionally rerank the chunks
4. Generate an answer grounded in the retrieved material

## Personalization

Each student has a small profile stored in `profiles/<student_id>.json`.

The profile currently tracks:

- recent query history
- topic frequency
- a simple rolling difficulty estimate

That difficulty level is injected into the answer prompt so the assistant can respond in a more beginner-, intermediate-, or advanced-friendly style.

## Notes

- Storage is fully local:
  - SQLite for original PDFs
  - Qdrant for embeddings
  - JSON for student profiles
- The project is designed for **course-grounded answers**, not general chatbot use.
- If you delete a material from the UI, the app also attempts to remove its vectors from Qdrant.

## Future Improvements

Some natural next steps are:

- stronger personalization based on actual learning progress
- page-level citations for PDFs
- richer material metadata and deduplication
- hybrid retrieval or more advanced query rewriting
- more detailed evaluation beyond retrieval metrics alone

## Author Notes

This project was built as an applied RAG system for an academic NLP learning use case.  
The implementation emphasizes:

- grounded answers
- simple local deployment
- clear modular structure
- enough extensibility for future agentic features
