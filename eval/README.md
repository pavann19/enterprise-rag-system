# Retrieval Evaluation Harness

Scores the pipeline's **retrieval** quality — does the right document come back for
a given question — against a small, hand-labeled golden set (`golden_set.json`, 15
queries, 5 per corpus document). Metrics: MRR, hit-rate@k, precision@k
(`metrics.py`, unit-tested in `tests/test_eval_metrics.py` with synthetic data, no
Ollama required).

## Scope — what this is and isn't

This is **not** RAGAS. RAGAS-style faithfulness/answer-relevancy scoring needs an
LLM judge to read the generated answer and grade it — that means either running a
second local model or calling a cloud API. The second option contradicts this
project's air-gapped design; the first was cut for now to avoid overclaiming
metrics no one has actually validated for local-judge reliability at this project's
current maturity. What's here instead is deliberately smaller and needs zero
judge: given a query with one known-correct source document, did retrieval surface
it, and at what rank. That's answerable with plain set/rank arithmetic against a
label, so every number it produces is trustworthy by construction — no separate
verification step required.

**Answer-quality evaluation (faithfulness, relevancy) is implemented** in
`judge_eval.py`, using whichever GEN_BACKEND is already configured as the
judge — see that file's docstring for why reusing the existing generation
backend, rather than adding a second one, was the resolution here. Treat its
output as **a judge model's opinion**, not ground truth like the rank-based
metrics above: a small/local judge can misjudge subtle cases, so report
these numbers as "N answers as scored by <model>", not as verified accuracy.

## Running it

Requires a running Ollama instance (embeds the 15 queries — no generation call is
made, so `GEN_MODEL`/mistral is not needed for this):

```bash
ollama serve
python -m eval.run_eval
```

Writes a timestamped full report to `eval/results/` (gitignored — regenerate
rather than commit) and prints the summary to stdout.

## Actual measured result (2026-08-20)

Run against the real corpus with `nomic-embed-text` via Ollama, `top_k=5`,
`backend=numpy` — not a stub, not illustrative:

```json
{
  "n_queries": 15,
  "mrr": 1.0,
  "hit_rate@1": 1.0,
  "precision@1": 1.0,
  "hit_rate@3": 1.0,
  "precision@3": 0.8666666666666667,
  "hit_rate@5": 1.0,
  "precision@5": 0.7466666666666666
}
```

Every one of the 15 golden queries retrieved its expected source document at
rank 1 (MRR = 1.0). Precision drops as `k` grows because the corpus documents
share vocabulary (all three are finance-policy text), so top-3/top-5 pull in
some correct-topic-but-wrong-file passages alongside the right one — expected
retrieval behavior, not a bug.

**Read this number honestly, not generously:** 15 hand-labeled queries against
a 3-document corpus is a real measurement, not a production-scale claim.
Perfect retrieval here says the pipeline's cosine-similarity ranking works
correctly on well-separated documents with distinct vocabulary — it says
nothing about performance on a larger, noisier, or more ambiguous corpus. Full
per-query detail (which passages ranked where, for which query) lives in the
timestamped report this run produced; re-run it yourself to reproduce or
challenge these numbers rather than taking them on faith.

## Extending the golden set

Add entries to `golden_set.json`: `{"id", "query", "expected_source"}`. Keep each
query traceable to a specific passage in `data/*.txt` you've actually read — a
golden label that doesn't match the source content measures nothing.

## Other scripts in this directory

This file covers `run_eval.py` specifically (retrieval *quality*). Three more
scripts here measure other things:

- `judge_eval.py` — answer *quality* (faithfulness/relevancy) via an LLM
  judge, see above.
- `benchmark_scale.py` — ingestion throughput and query latency against a
  synthetic corpus of hundreds of documents (never touches `data/` or this
  golden set). See the main
  [README's Production Scale section](../README.md#-production-scale) for
  the actual measured numbers rather than duplicating them here.
- `load_test.py` — real concurrent HTTP load against a running `service/api.py`.
