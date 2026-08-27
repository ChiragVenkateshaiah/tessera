# Evals

The evaluation harness runs a set of query/expected-answer cases through
Tessera's full query path (route → retrieve → generate) and reports
retrieval, generation, and routing metrics. It's built now, in Phase 1,
even though real test cases arrive later — see CLAUDE.md constraint #4:
this becomes the CI quality gate in Phase 5.

## Running it

```
set -a; source .env; set +a   # NVIDIA_API_KEY must be set
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
NVIDIA NIM's free tier allows up to 40 requests/minute and 10,000
requests/day, so a full sweep — even against a real query log of a few
dozen cases — fits comfortably without special pacing.

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

`evals/cases/query_log.yaml` (25 cases) is populated — but read this
before trusting it as real data. Meridian Advisory and its stakeholders
(including "Priya," who Discovery Findings §10 names as the owner of
this deliverable) are fictional; this is a portfolio project, not an
engagement with an actual client, so a genuine consultant-authored query
log will never arrive. These 25 cases are Claude-synthesized to match
the real-world usage patterns Discovery Findings §7 describes
(archetype distribution, situational/time-pressured wording), grounded
against the actual pilot corpus rather than guessed — every
`relevant_sources` path was verified against a real file under
`data/corpus/` by reading the document, not assumed. See the file's own
header comment and checkpoint.md's 2026-08-27 entry for the full
context. If this project ever becomes a template for a real engagement,
replace `query_log.yaml`'s contents with the actual log using the same
process that built it:

1. Add a new file under `evals/cases/` (or replace `query_log.yaml`)
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
4. Re-run `uv run tessera eval` (or `python -m evals.harness` as the
   no-install fallback) and compare against the placeholder baseline.
   Per the build plan: **numbers may be poor at this stage — tuning
   happens in Phase 2, against this real log.** The harness working
   end-to-end is Phase 1's deliverable, not the scores.

**First full sweep (33 cases: 8 placeholder + 25 query_log, 2026-08-27,
against live NVIDIA NIM):** routing accuracy 93.9%, mean recall 0.74,
mean precision 0.49, mean MRR 0.80, mean groundedness 4.86, mean
relevance 4.76 (all 1-5 scale except recall/precision/MRR). Two
genuinely misrouted cases (both archetype-C queries phrased as short
direct requests — "what do we have," "what's our standard approach" —
routed to archetype A instead) and a couple of retrieval misses on
single-document A cases are real findings, not noise: this is exactly
what Phase 2 tuning is for. Full per-case report in checkpoint.md's
2026-08-27 entry.

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
