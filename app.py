"""
app.py
------
Orchestration layer for the Enterprise RAG pipeline.

Coordinates two strictly separated execution phases:

  INGESTION  — document loading, segmentation, and vector encoding (run once)
    document → chunk_text() → embed_texts() → corpus_embeddings (np.ndarray)

  QUERY      — retrieval, generation, and schema-validated output (run per request)
    query → embed_texts() → retrieve() → generate_answer() → validate() → RAGResponse

All inference is local via Ollama. The validated output conforms to
the RAGResponse TypedDict contract, ensuring reliable integration with
downstream enterprise APIs and audit pipelines.
"""

import json
import sys
from pathlib import Path

from rag.chunker  import chunk_text
from rag.embedder import embed_texts
from rag.retriever import retrieve
from rag.generator import generate_answer
from validator.json_validator import validate, ValidationError, RAGResponse

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE     = Path(__file__).parent / "data" / "sample.txt"
EMBED_MODEL   = "nomic-embed-text"   # ollama pull nomic-embed-text
GEN_MODEL     = "mistral"            # ollama pull mistral
CHUNK_SIZE    = 300                  # characters per chunk (approx.)
CHUNK_OVERLAP = 50                   # overlap between consecutive chunks
TOP_K         = 3                    # passages to inject into the prompt
# ──────────────────────────────────────────────────────────────────────────────


# ── Phase 1: Ingestion ─────────────────────────────────────────────────────────

def ingest(embed_model: str = EMBED_MODEL) -> tuple:
    """
    Loads and embeds the knowledge base. Returns (chunks, corpus_embeddings).

    This phase is independent of any user query and could be cached
    or pre-computed in a production system.

    Args:
        embed_model: Ollama embedding model to use.

    Returns:
        Tuple of (List[str] chunks, np.ndarray corpus_embeddings).
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Knowledge base not found: {DATA_FILE}")

    raw_text = DATA_FILE.read_text(encoding="utf-8")
    chunks   = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"[ingest]   {len(chunks)} chunks created from {DATA_FILE.name}")

    print(f"[ingest]   Embedding with '{embed_model}'…")
    corpus_embeddings = embed_texts(chunks, model=embed_model)
    print(f"[ingest]   Corpus shape: {corpus_embeddings.shape}")

    return chunks, corpus_embeddings


# ── Phase 2: Query ─────────────────────────────────────────────────────────────

def query_pipeline(
    question: str,
    chunks: list,
    corpus_embeddings,
    embed_model: str = EMBED_MODEL,
    gen_model:   str = GEN_MODEL,
    top_k:       int = TOP_K,
) -> RAGResponse:
    """
    Runs retrieval + generation for a single question, then validates the output.

    Args:
        question:          The user's question.
        chunks:            Pre-ingested text chunks.
        corpus_embeddings: Pre-computed chunk embeddings.
        embed_model:       Ollama model for query embedding.
        gen_model:         Ollama model for answer generation.
        top_k:             Number of passages to retrieve.

    Returns:
        Validated RAGResponse dict.

    Raises:
        ValidationError: If the pipeline output fails schema validation.
        ConnectionError: If Ollama is unreachable.
    """
    # Retrieval
    query_embedding = embed_texts([question], model=embed_model)[0]
    results         = retrieve(query_embedding, corpus_embeddings, chunks, top_k=top_k)
    passages        = [chunk for chunk, _ in results]
    scores          = [round(score, 4) for _, score in results]
    print(f"[retrieve] Top-{top_k} passages. Scores: {scores}")

    # Generation
    print(f"[generate] Calling '{gen_model}'…")
    answer = generate_answer(question, passages, model=gen_model)
    print("[generate] Done.")

    # Structured output + validation
    response = RAGResponse(
        query   = question,
        answer  = answer,
        sources = passages,
        model   = gen_model,
    )
    validated = validate(response)          # raises ValidationError on failure
    print("[validate] ✓ Schema check passed.")

    return validated


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the four stages of a RAG pipeline?"
    )

    print(f"\n{'='*62}")
    print(f"  Question : {question}")
    print(f"  Embed    : {EMBED_MODEL}  |  Generate : {GEN_MODEL}")
    print(f"{'='*62}\n")

    try:
        chunks, corpus_embeddings = ingest()
        response = query_pipeline(question, chunks, corpus_embeddings)

    except (ConnectionError, FileNotFoundError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    except ValidationError as exc:
        print(f"\n[VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    # Display results
    print(f"\n{'─'*62}")
    print("Answer:")
    print(response["answer"])

    print(f"\n{'─'*62}")
    print("Structured Response (JSON):")
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
