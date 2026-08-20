# Evals

The evaluation harness runs a set of query/expected-answer cases through
Tessera's full query path (route → retrieve → generate) and reports
retrieval, generation, and routing metrics. It's built now, in Phase 1,
even though real test cases arrive later — see CLAUDE.md constraint #4:
this becomes the CI quality gate in Phase 5.

## Running it

```
set -a; source .env; set +a   # GEMINI_API_KEY must be set
uv run tessera eval           # from the repo root
```

This is the same harness `tessera eval` wraps — it's the one CLI
composition root (see `cli.py`), so prefer it. `python -m evals.harness`
still works as a no-install fallback and builds its own temporary
(non-persisted) index over `data/corpus/` rather than reading the
persisted one at `data/vectorstore/`; `tessera eval` requires
`tessera ingest` to have run first and reads that persisted index.
Either way, every case in `evals/cases/*.yaml` runs and a metrics report
prints. Each
case that reaches generation costs up to 3 live LLM calls (route,
generate, judge) — B/D cases that terminate at routing cost just 1.
Gemini's free tier caps `gemini-3.6-flash` at **20 requests/day**, so
budget accordingly before running the full case set, especially
alongside any other live-LLM work the same day.

## Case schema

Cases live in `evals/cases/*.yaml`, one list of entries per file:

```yaml
- id: q001
  query: "Do we have a framework for market entry analysis?"
  archetype: A
  relevant_sources: ["methodology/market-entry-overview.md"]
  ideal_answer: "Points to the market entry methodology page, summarises the key steps, cites the source."
```

- `archetype`: one of `A`/`B`/`C`/`D`. Only `A` (lookup) and `C`
  (synthesis) reach retrieval and generation; `B`/`D` cases exist to
  check that routing and the terminal refusal/not-yet-supported
  messages are correct, and should leave `relevant_sources` empty.
- `relevant_sources`: corpus-relative paths (relative to
  `data/corpus/`, e.g. `"methodology/pricing-strategy-overview.md"`) —
  used for recall@k/precision@k/MRR. Leave empty for `B`/`D` cases.
- `ideal_answer`: a free-text description of what a good answer should
  cover, fed to the LLM-judge for groundedness/relevance scoring. Leave
  empty to skip judging (e.g. for `B`/`D` cases, or any case where you
  only want the routing/retrieval metrics).

`evals/cases/placeholder.yaml` currently holds the 8 workshop queries
from Discovery Findings §7 (two per archetype) — useful for proving the
harness runs end-to-end, but explicitly **not representative** of real
consultant query patterns.

## Populating with the real query log

When the real query log arrives (target: 20–30 query/ideal-answer
pairs, actual consultant wording):

1. Add a new file under `evals/cases/` (e.g. `evals/cases/query_log.yaml`)
   rather than overwriting `placeholder.yaml` — keeping the placeholder
   set around preserves a known-good smoke-test case set independent of
   the real data.
2. For each real query: classify its archetype by hand (or from
   whatever context the log carries), identify the actual corpus
   document(s) it should point to for `relevant_sources`, and write a
   short `ideal_answer` description of what a good answer covers — not
   a full reference answer, just enough for the LLM-judge to grade
   against.
3. Only `A`/`C` queries get meaningful retrieval/groundedness/relevance
   numbers. If the real log includes expertise-finding or comparative
   queries, they're still worth adding as `B`/`D` cases to keep routing
   accuracy meaningful, just without `relevant_sources`/`ideal_answer`.
4. Re-run `python -m evals.harness` and compare against the placeholder
   baseline. Per the build plan: **numbers may be poor at this stage —
   tuning happens in Phase 2, against this real log.** The harness
   working end-to-end is Phase 1's deliverable, not the scores.

## What's in here

- `harness.py` — loads cases, runs each through `route()` → `retrieve()`
  → `generate_answer()` (the same functions `pipeline.answer_query()`
  composes — called directly here because the harness needs the full
  ranked retrieval and the exact chunks shown to the model, which
  `answer_query()`'s collapsed return type doesn't expose), and
  aggregates metrics. `main()` is the only impure part — it builds a
  real embedder/index/LLM client and prints; everything else is pure
  and unit-tested with fakes (see `tests/test_harness.py`).
- `metrics.py` — recall@k, precision@k, MRR (pure, deterministic,
  directly unit-tested) and the LLM-as-judge groundedness/relevance
  scorer (only its response-parsing is unit-tested — see
  `tests/test_metrics.py` and CLAUDE.md's "do not over-test LLM
  outputs" convention).
- `cases/` — the YAML case files described above.
