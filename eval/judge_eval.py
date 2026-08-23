"""
eval/judge_eval.py
-------------------
Answer-quality evaluation via an LLM judge — the item eval/README.md
previously listed as deliberately deferred ("needs an LLM judge... either
running a second local model or calling a cloud API").

Resolution: use whichever GEN_BACKEND is already configured as the judge
too. This project already depends on a generation backend to answer
questions in the first place (Ollama locally, Groq/Anthropic for the hosted
demo) — reusing it for judging adds no new dependency or air-gap
compromise beyond what generation itself already requires. If GEN_BACKEND
is "ollama", the judge call stays fully local like everything else in that
mode; if it's "groq"/"anthropic", both the answer and the judgment happen
on the same already-accepted cloud call surface.

Scores each golden-set query's real generated answer against its source
passage on two axes, 1-5 each:
  faithfulness — is the answer supported by the passage, no invented facts?
  relevancy    — does the answer actually address the question asked?

This is explicitly a **judge model's opinion**, not ground truth like
run_eval.py's rank-based retrieval metrics — a small/local judge model can
misjudge subtle cases. Report these numbers as "a judge model's assessment
of N answers", never as a fully verified accuracy figure.

Usage:
    python -m eval.judge_eval
    python -m eval.judge_eval --gen-backend groq --gen-model llama-3.1-8b-instant
"""

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from app import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_BACKEND,
    EMBED_MODEL,
    GEN_BACKEND,
    GEN_MODEL,
    VECTOR_BACKEND,
    query_pipeline,
)
from rag.generator import generate_answer
from rag.ingestion import ingest
from rag.logging_config import get_logger
from validator.json_validator import ValidationError

log = get_logger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"

_JUDGE_PROMPT = """You are grading an AI assistant's answer to a question, using ONLY the \
source passage below as ground truth.

Source passage:
{passage}

Question: {question}

Assistant's answer: {answer}

Score the answer on two axes, each an integer from 1 (worst) to 5 (best):
- faithfulness: is every claim in the answer actually supported by the source passage \
(no invented facts, no unsupported claims)?
- relevancy: does the answer actually address the question asked?

Respond with ONLY a JSON object, no other text: {{"faithfulness": <int>, "relevancy": <int>}}"""


def _load_golden_set() -> list:
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def _parse_judge_response(raw: str) -> dict:
    """Extracts the {"faithfulness": N, "relevancy": N} object from the judge's raw text."""
    match = re.search(r"\{[^{}]*\}", raw)
    if not match:
        raise ValueError(f"Judge response did not contain a JSON object: {raw!r}")
    parsed = json.loads(match.group(0))
    faithfulness = int(parsed["faithfulness"])
    relevancy = int(parsed["relevancy"])
    if not (1 <= faithfulness <= 5) or not (1 <= relevancy <= 5):
        raise ValueError(f"Judge scores out of the 1-5 range: {parsed}")
    return {"faithfulness": faithfulness, "relevancy": relevancy}


def judge_answer(question: str, answer: str, passage: str, gen_backend: str, gen_model: str) -> dict:
    """Calls the configured generation backend as a judge, returns {"faithfulness": int, "relevancy": int}."""
    prompt_as_passages = [_JUDGE_PROMPT.format(passage=passage, question=question, answer=answer)]
    raw = generate_answer(
        query="Grade the answer above.", passages=prompt_as_passages, model=gen_model, backend=gen_backend
    )
    return _parse_judge_response(raw)


def run(gen_backend: str = GEN_BACKEND, gen_model: str = GEN_MODEL) -> dict:
    """
    For each golden query: runs the real pipeline to get a generated answer,
    then asks the judge to score it against the query's known-correct source
    document's full text. Returns {"config": {...}, "summary": {...}, "per_query": [...]}.
    """
    golden_queries = _load_golden_set()
    log.info("Loaded %d golden queries for answer-quality judging", len(golden_queries))

    chunks, metadata, vector_store = ingest(
        data_dir=DATA_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model=EMBED_MODEL,
        embed_backend=EMBED_BACKEND,
        backend=VECTOR_BACKEND,
        cache_dir=CACHE_DIR,
    )

    per_query = []
    for item in golden_queries:
        response = query_pipeline(
            query=item["query"],
            chunks=chunks,
            metadata=metadata,
            vector_store=vector_store,
            gen_model=gen_model,
            gen_backend=gen_backend,
        )
        top_passage = response["sources"][0]["text"] if response["sources"] else ""

        try:
            scores = judge_answer(
                question=item["query"],
                answer=response["answer"],
                passage=top_passage,
                gen_backend=gen_backend,
                gen_model=gen_model,
            )
        except ValueError as exc:
            log.warning("Judge response unparseable for query id=%s: %s", item["id"], exc)
            scores = {"faithfulness": None, "relevancy": None}

        per_query.append(
            {
                "id": item["id"],
                "query": item["query"],
                "answer": response["answer"],
                **scores,
            }
        )

    scored = [q for q in per_query if q["faithfulness"] is not None]
    summary = {
        "n_queries": len(golden_queries),
        "n_scored": len(scored),
        "n_unparseable": len(golden_queries) - len(scored),
        "mean_faithfulness": (
            round(sum(q["faithfulness"] for q in scored) / len(scored), 3) if scored else None
        ),
        "mean_relevancy": round(sum(q["relevancy"] for q in scored) / len(scored), 3) if scored else None,
    }

    return {
        "config": {
            "gen_backend": gen_backend,
            "gen_model": gen_model,
            "judge": "same backend/model as generation",
        },
        "summary": summary,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge answer-quality evaluation")
    parser.add_argument("--gen-backend", default=GEN_BACKEND, choices=["ollama", "anthropic", "groq"])
    parser.add_argument("--gen-model", default=GEN_MODEL)
    args = parser.parse_args()

    try:
        report = run(gen_backend=args.gen_backend, gen_model=args.gen_model)
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as exc:
        print(f"[VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(report["summary"], indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"judge_eval_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":  # pragma: no cover — trivial guard; main() itself is tested directly
    main()
