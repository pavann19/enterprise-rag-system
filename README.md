# Enterprise RAG System — Ollama Edition

A minimal, modular **Retrieval-Augmented Generation (RAG)** pipeline that runs entirely on local hardware via [Ollama](https://ollama.ai). No cloud APIs, no API keys, no external data transmission.

---

## Problem Statement

Large language models (LLMs) generate answers from knowledge encoded during training. This creates two practical problems:

1. **Staleness** — the model cannot answer questions about events or documents after its training cutoff.
2. **Hallucination** — without a grounding source, the model may produce plausible but incorrect answers.

RAG addresses both by retrieving relevant passages from a controlled knowledge base at query time and injecting them into the prompt. The model generates an answer grounded in your documents rather than in its parametric memory.

This project demonstrates that full pipeline using only local Ollama models, making it suitable for environments where data privacy, offline operation, or zero API cost are requirements.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  INGESTION  (runs once per knowledge base)                          │
│                                                                     │
│  data/sample.txt                                                    │
│       │                                                             │
│       ▼                                                             │
│  chunker.py          word-boundary splitting with overlap           │
│       │                                                             │
│       ▼                                                             │
│  embedder.py  ───► Ollama /api/embeddings (nomic-embed-text)        │
│       │                                                             │
│       ▼                                                             │
│  corpus_embeddings   np.ndarray in memory                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  QUERY  (runs per user question)                                    │
│                                                                     │
│  user question                                                      │
│       │                                                             │
│       ▼                                                             │
│  embedder.py  ───► Ollama /api/embeddings (nomic-embed-text)        │
│       │                                                             │
│       ▼                                                             │
│  retriever.py        cosine similarity → top-k passages             │
│       │                                                             │
│       ▼                                                             │
│  generator.py ───► Ollama /api/generate  (mistral / llama3)         │
│       │                                                             │
│       ▼                                                             │
│  json_validator.py   schema check → RAGResponse TypedDict           │
│       │                                                             │
│       ▼                                                             │
│  { query, answer, sources, model }                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Map

| Module | Role |
|---|---|
| `rag/_http.py` | Shared Ollama HTTP transport (single point for all API calls) |
| `rag/chunker.py` | Word-boundary text splitting with configurable size and overlap |
| `rag/embedder.py` | Calls `/api/embeddings`; returns `np.ndarray` |
| `rag/retriever.py` | Cosine similarity ranking; returns top-k `(text, score)` tuples |
| `rag/generator.py` | Builds RAG prompt; calls `/api/generate`; returns answer string |
| `validator/json_validator.py` | `RAGResponse` TypedDict + `ValidationError`; raises on schema failure |
| `app.py` | Orchestrates `ingest()` → `query_pipeline()`; CLI entry point |

---

## Pipeline Flow

**Step 1 — Ingestion**
The knowledge base (`data/sample.txt`) is read and split into overlapping chunks by `chunker.py`. Overlap preserves sentence context across chunk boundaries.

**Step 2 — Corpus Embedding**
Each chunk is sent to Ollama's `/api/embeddings` endpoint via `embedder.py`. The response is a dense float vector. All vectors are stacked into a `(n_chunks, dim)` numpy array.

**Step 3 — Query Embedding**
The user's question is embedded using the same model and endpoint, producing a `(dim,)` query vector.

**Step 4 — Retrieval**
`retriever.py` computes cosine similarity between the query vector and every row in the corpus array. The top-k highest-scoring chunks are returned as candidate passages.

**Step 5 — Generation**
`generator.py` formats a prompt containing a system instruction, the numbered passages, and the question. This prompt is sent to Ollama's `/api/generate`. The model is instructed to answer using only the provided context.

**Step 6 — Validation**
The response dict is passed to `validator/json_validator.py`, which checks it against the `RAGResponse` TypedDict schema. If any field is missing, has the wrong type, or is blank, a `ValidationError` is raised before the response is returned to the caller.

---

## Design Decisions

**Shared HTTP transport (`rag/_http.py`)**
Both `embedder.py` and `generator.py` call Ollama via a single `ollama_post()` helper. This avoids duplicated `urllib` boilerplate and provides one place to adjust timeouts, headers, or authentication if the setup changes.

**Retrieval and generation are independent modules**
`retriever.py` only computes similarity scores — it does not know about Ollama. `generator.py` only generates text — it does not know how passages were selected. This makes each module individually testable and replaceable.

**Structured output via TypedDict and raising validator**
Rather than returning `(bool, str)` tuples, `validate()` raises `ValidationError` on failure and returns a typed `RAGResponse` on success. Callers use standard exception handling and get IDE-visible type information on the return value.

**Private helpers, minimal public API**
`_cosine_similarity()` and `_build_prompt()` are module-private. Each module exposes only what its callers need: one or two public functions.

**No vector database**
Cosine similarity over a numpy array is sufficient for small corpora and requires no infrastructure. The tradeoff — O(n) scan per query — is documented under Limitations.

**Stdlib-only HTTP**
`urllib` is used instead of `requests` or `httpx` so the project has no runtime HTTP dependency. The only runtime dependency is `numpy`.

---

## Setup

### Prerequisites

```bash
# 1. Install Ollama — https://ollama.ai/download

# 2. Pull required models
ollama pull nomic-embed-text   # embedding model
ollama pull mistral            # generation model (or: llama3, phi3, gemma)

# 3. Start the Ollama server
ollama serve
```

### Install Python dependencies

```bash
pip install -r requirements.txt   # numpy only
```

### Run

```bash
cd enterprise-rag-system

python app.py                                          # default question
python app.py "What is the role of overlap in chunking?"   # custom question
```

### Sample output

```
==============================================================
  Question : What are the four stages of a RAG pipeline?
  Embed    : nomic-embed-text  |  Generate : mistral
==============================================================

[ingest]   5 chunks created from sample.txt
[ingest]   Embedding with 'nomic-embed-text'…
[ingest]   Corpus shape: (5, 768)
[retrieve] Top-3 passages. Scores: [0.9211, 0.8934, 0.8512]
[generate] Calling 'mistral'…
[generate] Done.
[validate] ✓ Schema check passed.

──────────────────────────────────────────────────────────────
Answer:
A RAG pipeline has four stages: Ingestion, Embedding, Retrieval, and Generation…

──────────────────────────────────────────────────────────────
Structured Response (JSON):
{
  "query": "What are the four stages of a RAG pipeline?",
  "answer": "…",
  "sources": ["…", "…", "…"],
  "model": "mistral"
}
```

---

## Production Considerations

This project is a portfolio demonstration, not a production deployment. The table below maps current design choices to what a production system would require.

| Concern | Current approach | Production approach |
|---|---|---|
| **Privacy** | Fully local — data never leaves the machine | Same: run Ollama on-premise or in a private VPC |
| **API cost** | Zero — local inference only | Zero variable cost once hardware is provisioned |
| **Latency** | ~1–8 s on CPU; ~0.2–1 s on GPU | Serve Ollama on a GPU-backed server; use a faster model |
| **Corpus size** | In-memory numpy, recomputed each run | Pre-compute embeddings, persist with `np.save` or a vector DB |
| **Concurrency** | Single-threaded, synchronous | Async workers; Ollama supports parallel requests |
| **Observability** | `print()` statements | Structured logging, request tracing, latency metrics |
| **Deployment** | `python app.py` | FastAPI wrapper, Docker container, process manager |

---

## Limitations

- **In-memory only** — embeddings are recomputed on every run. Not suitable for corpora larger than a few thousand chunks.
- **Single file ingestion** — the pipeline reads one `.txt` file. Multi-document ingestion requires additional code to walk a directory.
- **O(n) retrieval** — cosine similarity scans all chunks linearly. Performance degrades as the corpus grows.
- **No streaming** — generation is buffered (`"stream": false`). Interactive UIs would require streaming support.
- **CPU embedding is slow** — `nomic-embed-text` can take several seconds per chunk on CPU hardware.
- **No re-ranking** — retrieved passages are ranked by embedding similarity only. A cross-encoder re-ranker would improve precision but adds latency and complexity.
- **No conversation history** — each query is stateless. Multi-turn Q&A is not supported.

---

## Future Improvements

| Area | Change |
|---|---|
| **Persistence** | Save/load corpus embeddings with `np.save` / `np.load` to skip re-embedding on startup |
| **Scale** | Replace numpy cosine scan with FAISS or Qdrant for approximate nearest-neighbour search |
| **Multi-document** | Walk `data/` directory; support `.pdf` via `pypdf` |
| **Streaming** | Pass `"stream": true` to Ollama and yield tokens progressively |
| **Web UI** | Thin Streamlit or FastAPI layer over `query_pipeline()` |
| **Re-ranking** | Add a cross-encoder pass after retrieval to improve passage precision |
| **Evaluation** | Integrate RAGAS metrics: faithfulness, answer relevancy, context recall |
| **Testing** | `pytest` unit tests for `chunker`, `retriever`, and `validator` modules |
