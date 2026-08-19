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

**Answer-quality evaluation (faithfulness, relevancy, hallucination rate) is not
implemented.** See the main README roadmap.

## Running it

Requires a running Ollama instance (embeds the 15 queries — no generation call is
made, so `GEN_MODEL`/mistral is not needed for this):

```bash
ollama serve
python -m eval.run_eval
```

Writes a timestamped full report to `eval/results/` (gitignored — regenerate
rather than commit) and prints the summary to stdout, e.g.:

```json
{
  "n_queries": 15,
  "mrr": 0.93,
  "hit_rate@1": 0.87,
  "precision@1": 0.87,
  "hit_rate@3": 1.0,
  "precision@3": 0.33
}
```

**No run has been captured yet in this repository** — the harness has been built
and its metric functions are unit-tested, but end-to-end numbers require a local
Ollama instance this environment doesn't have running. Treat any numbers you see
elsewhere as illustrative only until a real `eval/results/*.json` exists.

## Extending the golden set

Add entries to `golden_set.json`: `{"id", "query", "expected_source"}`. Keep each
query traceable to a specific passage in `data/*.txt` you've actually read — a
golden label that doesn't match the source content measures nothing.
