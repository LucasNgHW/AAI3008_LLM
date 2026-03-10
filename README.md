# NLP Learning Assistant (RAG)

A personalised NLP learning assistant that retrieves relevant course material and generates grounded, difficulty-adapted answers via Claude.

## Project Structure

```
nlp-assistant/
├── pipeline/
│   ├── parsers.py       # PDF, PPTX, notebook extraction
│   ├── chunker.py       # RecursiveCharacterTextSplitter + metadata tagging
│   ├── embedder.py      # all-MiniLM-L6-v2 batch encoding
│   └── indexer.py       # Qdrant collection setup and upsert
├── rag/
│   ├── retriever.py     # Qdrant similarity search with metadata filters
│   ├── reranker.py      # Cross-encoder second-pass reranking
│   └── generator.py     # Claude API prompt assembly and generation
├── personalisation/
│   └── user_profile.py  # Per-student JSON profile + difficulty inference
├── app/
│   └── ui.py            # Streamlit chat interface
├── profiles/            # Auto-created; one JSON per student
├── data/raw/            # Place your course files here
├── ingest.py            # Pipeline runner
├── evaluate.py          # MRR + Recall@k evaluation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
```

## Usage

### 1. Ingest course materials

Place PDFs, PPTX files, and Jupyter notebooks in `data/raw/`, then run:

```bash
python ingest.py                      # standard ingest
python ingest.py --recreate           # wipe and rebuild the index
python ingest.py --dir path/to/files  # custom source directory
```

### 2. Run the chat interface

```bash
streamlit run app/ui.py
```

### 3. Evaluate retrieval quality

Prepare an evaluation set at `data/eval_set.json` (see `evaluate.py` for schema), then:

```bash
python evaluate.py --eval data/eval_set.json --top_k 5
python evaluate.py --eval data/eval_set.json --top_k 5 --rerank
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Local, free to re-index, 384-dim |
| Vector DB | Qdrant | Compound metadata filter API |
| Chunking | RecursiveCharacterTextSplitter → SemanticChunker (Wk 8) | Baseline first, then upgrade post-eval |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder for higher accuracy |
| LLM | Claude API | Context window + instruction following |
