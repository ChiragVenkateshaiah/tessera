# Evals

The evaluation harness runs a set of query/expected-answer cases through
Tessera's full query path (route → retrieve → generate) and reports
retrieval, generation, and routing metrics. It's built now, in Phase 1,
even though real test cases arrive later — see CLAUDE.md constraint #4:
this becomes the CI quality gate in Phase 5.

## Running it

```
set -a; source .env; set +a   # NVIDIA_API_KEY must be set
uv run tessera ingest         # once: the document index
uv run tessera index-people   # once: the people index (archetype B)
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
generate, judge) — D cases that terminate at routing cost just 1. A
scored B case costs 3 (route, generate, judge); a B no-match case costs
just 1, because the no-match path makes no generation call.
NVIDIA NIM's free tier allows up to 40 requests/minute and 10,000
requests/day, so a full sweep — even against a real query log of a few
dozen cases — fits on paper. In practice NIM throttled far below that
on 2026-09-29, so `tessera eval` wraps the LLM in `RetryingLLMClient`
(calls ≥3 s apart; 429/5xx retried with 15–120 s exponential backoff,
`Retry-After` honoured, up to 6 attempts) and prints `[n/total] case ok`
progress and any "retrying in Ns" notices to stderr — the report itself
stays on stdout. A 55-case sweep can still take 1–1.5 hours when NIM is
unhealthy; that is the backoff working, not a hang.

## Case schema

Cases live in `evals/cases/*.yaml`, one list of entries per file:

```yaml
- id: q001
  query: "Do we have a framework for market entry analysis?"
  archetype: A
  relevant_sources: ["methodology/market-entry-overview.md"]
  ideal_answer: "Points to the market entry methodology page, summarises the key steps, cites the source."
```

- `archetype`: one of `A`/`B`/`C`/`D`. `A` (lookup) and `C` (synthesis)
  run document retrieval and generation; `B` (expertise) runs expertise
  retrieval (`find_experts`) and generation; `D` cases exist to check
  that routing and the terminal refusal are correct, and should leave
  `relevant_sources` empty.
- `relevant_sources`: corpus-relative paths (relative to
  `data/corpus/`, e.g. `"methodology/pricing-strategy-overview.md"`) —
  used for recall@k/precision@k/MRR. Leave empty for `B`/`D` cases.
- `ideal_answer`: a free-text description of what a good answer should
  cover, fed to the LLM-judge for groundedness/relevance scoring. Leave
  empty to skip judging (e.g. for `B`/`D` cases, or any case where you
  only want the routing/retrieval metrics).
- `relevant_people` (archetype B): `person_id`s from `data/expertise/`
  who belong in the shortlist. Used for person recall@k / precision@k /
  MRR. Person recall is scored against `min(|relevant_people|, k)`
  (`metrics.shortlist_recall_at_k`), not `|relevant_people|`: a query with
  20 genuine experts and a 5-person shortlist would otherwise cap at 0.25
  however good the ranking. Sets are derived mechanically from the raw
  records (project / authored evidence on the query's topic, plus
  industry or role/recency where the query says so) and audited against
  retrieval's top 20 — see the comment on each case. A test asserts every
  id resolves to a real record.
- `expect_no_match` (archetype B): the query asks for expertise the firm
  doesn't have. Correct = the fixed no-match message **and** zero
  generation LLM calls (counted, not assumed). These live in
  `expertise_nomatch.yaml`, are not judged, and don't contribute to
  recall/MRR; they are scored only by the no-match refusal rate.

`evals/cases/placeholder.yaml` holds the 8 workshop queries from
Discovery Findings §7 (two per archetype). Since P2-2 it is **held out
as an overfitting check-set** — retrieval tuning (P2-3) works against
`query_log.yaml` only, and `placeholder.yaml` is the independent check
that a tuned config didn't just fit the tuning set. It is explicitly
**not representative** of real consultant query patterns.

## Populating with the real query log

`evals/cases/query_log.yaml` (42 cases as of P2-2) is populated — but
read this before trusting it as real data. Meridian Advisory and its
stakeholders (including "Priya," who Discovery Findings §10 names as the
owner of this deliverable) are fictional; this is a portfolio project,
not an engagement with an actual client, so a genuine consultant-authored
query log will never arrive. These cases are Claude-synthesized to match
the real-world usage patterns Discovery Findings §7 describes
(archetype distribution, situational/time-pressured wording), grounded
against the actual pilot corpus rather than guessed — every
`relevant_sources` path was verified against a real file under
`data/corpus/` by reading the document, not assumed. See the file's own
header comment and checkpoint.md's 2026-08-27 / 2026-09-04 entries for
the full context. If this project ever becomes a template for a real
engagement,
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

**First full sweep (33 cases, 2026-08-27, against live NVIDIA NIM):**
routing accuracy 93.9%, mean recall 0.74, mean precision 0.49, mean MRR
0.80, mean groundedness 4.86, mean relevance 4.76. Two genuinely
misrouted cases (both archetype-C queries phrased as short direct
requests — "what do we have," "what's our standard approach" — routed
to archetype A instead) and two recall=0.00 retrieval misses on
single-document A cases were real findings, not noise — exactly what
Phase 2 tuning is for, and both were fixed same-session:

- **Retrieval fix**: chunk embeddings were computed from body text
  alone — `chunker.py`'s new `chunk_embedding_text()` prepends the
  document title and heading path, so a query echoing a document's title
  (a common lookup pattern, e.g. "do we have a checklist for X") can
  match even when the body prose doesn't repeat those words. Both misses
  were confirmed root-caused this way (the target doc's exact query
  terms appeared only in its title) and confirmed fixed after
  re-`tessera ingest`-ing with the new embedding text.
  `generation/answer.py`'s `RELEVANCE_THRESHOLD = 0.35` was re-checked
  against the new embedding scores and still holds — the on-corpus/
  off-corpus separation margin actually widened.
- **Routing fix**: `ROUTER_SYSTEM_PROMPT` (`generation/prompts.py`)
  gained an explicit A-vs-C disambiguation note — the decisive signal
  for C is the situation (a deadline, a new staffing, an upcoming
  meeting), not the trailing question's wording, since both misroutes
  had lookup-shaped endings ("what do we have") on top of a genuine
  onboarding/deadline situation. Verified fixed live, and a 7-query
  regression check across all four archetypes confirmed no
  overcorrection of genuine A/B/D queries into C.

**Post-tuning sweep (same 33 cases): 100% routing accuracy, mean recall
0.87, mean precision 0.71, mean MRR 0.95, mean groundedness 4.95, mean
relevance 4.81** (recall/precision/MRR figures on the 21 archetype-A/C
cases with `relevant_sources`; the other 12 are B/D routing-only cases).
Full per-case detail and the debugging story (including several
transient `503`s from NVIDIA's API, unrelated to this tuning, handled
correctly by Task 7's per-case error isolation) in checkpoint.md's
2026-08-27 entry.

**P2-2 baseline sweep (50 cases, 2026-09-04): routing accuracy 96.0%,
mean recall 0.88, mean precision 0.79, mean MRR 0.97, mean groundedness
5.00, mean relevance 4.91** — `tessera eval --check` → `=> PASS (gated
thresholds)`. The eval set grew 33 → 50 and a label-completeness audit
tightened four under-labeled A/C cases (see checkpoint.md's 2026-09-04
entry). Two of the 17 new archetype-C cases (`ql035`, `ql038`) misroute
to A — lookup-shaped phrasing over synthesis intent — a P2-4 router
prompt target; routing still clears the 95% gate.

**P2-3 (retrieval-constant grid search, `evals/tune_retrieval.py`):**
the 72-point grid over `LOOKUP_TOP_K`/`SYNTHESIS_CANDIDATE_K`/
`SYNTHESIS_MAX_RESULTS`/`SYNTHESIS_MAX_PER_DOCUMENT` found no config
that beats the current constants by more than the plan's 0.02
keep-current tiebreak — **the constants are unchanged**.
`RELEVANCE_THRESHOLD` was re-probed against the live index (on-corpus
min 0.576, adjacent-but-absent 0.243–0.306, off-corpus max 0.113) and
also holds unchanged at 0.35. Confirmation sweep: same 96.0%/0.88/0.79/
0.97, groundedness 4.94, relevance 4.91 — `=> PASS`, no regression from
P2-2. Run `python -m evals.tune_retrieval` (or `uv run` equivalent) to
reproduce the grid search — it's retrieval-only, no LLM calls, seconds
to run.

**P2-4 (close the two known bar gaps):** the P2-3 grid confirmed the
retrieval constants can't lift the multi-source-A recall floor, so
`retriever.retrieve()`'s A path now diversifies to one chunk per
document — it returns the top `LOOKUP_TOP_K` *distinct* documents from a
30-candidate pool, the same mechanism C uses. `q001`/`ql003`/`ql004`
recall 0.40/0.50 → 1.00; mean recall 0.88 → 0.95. `ROUTER_SYSTEM_PROMPT`
gained a second A-vs-C note for queries describing a live client need
("client wants help with X … what's our approach") — `ql035`/`ql038`
now route C, routing 96% → 100%. `LOOKUP_ANSWER_SYSTEM_PROMPT` was
tightened so single-target lookups don't pad their answers with the
now-larger neighbour set. `evals/tune_retrieval.py`'s lookup branch
mirrors the new A path.

**P2-5 / Phase 2 exit sweep (50 cases, 2026-09-06, clean 50/50):**
routing 100%, mean recall 0.95, mean precision 0.42, mean MRR 0.97, mean
groundedness 4.77, mean relevance 4.60 — `tessera eval --check` →
`=> PASS (gated thresholds)`. Relevance clears by only 0.10: the P2-4
A-diversification means narrow single-target lookups now return 5
same-family docs, and the judge marks a few down for breadth
(`ql007`/`ql004`/`ql027`/`ql028`). Carried forward as the first
post-Phase-2 tuning item — see `checkpoint.md` "Notes / open flags"
(candidate lever: adaptive `k` for A). Precision fell 0.72 → 0.42 with
the same change and stays ungated (`QUALITY_BAR.md`).

**Phase 3 (archetype B, P3-5 → P3-6):** P3-5 added the B metrics
(person recall/precision/MRR against `relevant_people`, a separate B
judge, the 6-case `expertise_nomatch.yaml` refusal set) and the B bar
rows, provisional at first because person recall scored 0.89. A
retrieval pass (PR #51) read "led/ran" and "recently" from the query,
lifted person recall to 0.91, and gated the B rows. Use
`python -m evals.diagnose_expertise` to inspect B ranking without LLM
calls.

**P3-6 / Phase 3 exit sweep (55 cases, 2026-10-01, clean 55/55):**
routing 100%; A/C recall 0.95, precision 0.42, MRR 0.97, groundedness
4.83, relevance 4.60; B person recall 0.91, precision 0.91, MRR 1.00,
groundedness 5.00, relevance 4.89, no-match refusal 100% —
`tessera eval --check` → `=> PASS (gated thresholds)`. Thin margins:
A/C relevance (+0.10, same narrow-A cause as P2-5) and B person recall
(+0.01; `ql019` 0.60, `ql041`/`ql042` 0.80). See `QUALITY_BAR.md`.

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
- `diagnose_expertise.py` — retrieval-only view of archetype-B ranking
  (candidates, evidence, scores) for a query; no LLM calls.
- `tune_retrieval.py` — retrieval-only grid search over the A/C
  retrieval constants (P2-3); no LLM calls.
- `cases/` — the YAML case files described above.
