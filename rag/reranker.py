"""
rag/reranker.py
----------------
Optional cross-encoder re-ranking of retrieved passages.

rag/retriever.py ranks by cosine similarity between independently-computed
query/passage embeddings (a bi-encoder) — fast, but the query and passage
never actually attend to each other. A cross-encoder scores each
(query, passage) pair jointly, which is slower (one forward pass per
candidate, not one for the whole corpus) but more precise — worth it only
over the small top-N retrieve() already narrowed things down to, never over
the full corpus.

Opt-in via RERANK_ENABLED=1 (default off): it adds latency and a model
download on first use, uncommented `sentence-transformers` already being a
dependency is what makes this free to add here. Not applied unless a caller
asks for it (app.py's query_pipeline), so existing behavior is unchanged by
default.
"""

import os

from rag.logging_config import get_logger

log = get_logger(__name__)

RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "").lower() in ("1", "true", "yes")
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model_cache: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import CrossEncoder

        log.info("Loading cross-encoder reranker model '%s' (first use only, cached after).", model_name)
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def rerank(
    query: str,
    results: list[dict[str, object]],
    model_name: str = DEFAULT_RERANK_MODEL,
) -> list[dict[str, object]]:
    """
    Re-scores and re-sorts retrieve()'s output using a cross-encoder.

    Args:
        query:      The user's question.
        results:    Output of rag.retriever.retrieve() —
                    [{"text": str, "score": float, "source": str}, ...]
        model_name: HuggingFace cross-encoder model identifier.

    Returns:
        The same list, re-sorted by cross-encoder relevance (descending).
        Each dict gains a "rerank_score" key; the original "score" (cosine
        similarity from retrieval) is preserved for comparison.

    Raises:
        ImportError: If sentence-transformers isn't installed.
    """
    if not results:
        return results

    model = _get_model(model_name)
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    reranked = [
        {**r, "rerank_score": round(float(score), 4)} for r, score in zip(results, scores, strict=True)
    ]
    reranked.sort(key=lambda r: r["rerank_score"], reverse=True)

    log.debug(
        "rerank() reordered %d passages — top source now '%s' (rerank_score=%.4f)",
        len(reranked),
        reranked[0]["source"],
        reranked[0]["rerank_score"],
    )
    return reranked
