# Tessera — Phase 3 Plan (Claude Code Brief)

**Status: DRAFT — pending review.** Not yet adopted. Phase 2 is complete
and tagged `v0.2.0`; all 5 Phase 1 exit criteria and all 6 gated Phase 2
bar thresholds hold (see `checkpoint.md`). Companion documents
(`Tessera_Discovery_Findings.md`, `Tessera_Solution_Design.md`) carry the
problem context and full architecture; `Tessera_Phase2_Plan.md` is the
immediate predecessor and the model for this document's shape.

---

## 0. Context in one paragraph

Phases 1–2 built and evaluated the document side of Tessera: archetypes A
(lookup) and C (synthesis) retrieve from the pilot corpus and generate
grounded, cited answers, held to a documented quality bar. Archetype B —
**expertise-finding** ("who at the firm knows about X") — has been a
routed "not yet supported" response since Phase 1 because it is not
document retrieval at all: the answer is a *person*, sourced from
structured HR/staffing/expertise data, and Discovery left the location
and structure of that data as an open question (Discovery Findings §9.8,
Solution Design §1.2B). Phase 3 builds B: synthesize the expertise
dataset, stand up a distinct retrieval path over it behind a swappable
port, generate person-answers whose every claim traces to a profile
record, wire B through the router and pipeline, and extend the eval
harness and quality bar to cover it.

## 1. Objective and boundaries

**Objective:** archetype B answered end to end — a synthesized firm
expertise dataset, a distinct `ExpertiseStore`-backed retrieval path,
grounded person-answers with mandatory evidence citations, wired through
`router` / `pipeline` / `cli`, and measured by an extended eval harness
against an extended quality bar.

**Exit gate:** on the full eval set (now including populated B cases), all
gated thresholds pass — the existing A/C thresholds unchanged, plus the
new B thresholds (§4) — reproducibly, on a clean sweep.

**In scope for Phase 3**
- A synthesized firm expertise dataset (`data/expertise/`, committed) and
  its documented schema
- An `ExpertiseStore` port + a local implementation, reusing the existing
  `Embedder` port for profile-text semantic matching
- An expertise retrieval path: query → ranked people, each with the
  evidence that ranked them
- B generation: a person-answer prompt + generator that names people,
  cites the profile evidence for each, and surfaces the staleness /
  self-reported-vs-evidenced distinction
- Router change: B stops short-circuiting and routes into the expertise
  path; the "not yet supported" message is retained only as the
  no-match fallback
- `pipeline.answer_query()` and `cli.py` handle B
- Eval harness: B cases get real metrics (person recall@k, an
  answer-quality judge); the 10 existing B cases rewritten with
  `relevant_people` + `ideal_answer`
- `evals/QUALITY_BAR.md`: B thresholds added
- Phase 3 exit: README, checkpoint, clean full sweep, tag `v0.3.0`

**Explicitly NOT in Phase 3 — unchanged from the Phase 1–2 do-not-build
list except where noted**
- Archetype D — remains a refusal guardrail only.
- Any AWS deployment, Terraform, CI/CD, monitoring — Phases 4–5.
- PowerPoint / deck ingestion — not in the pilot corpus.
- **Real HR-system integration** (Workday, an SFDC people-index, an SSO
  directory) — Phase 4+. Phase 3's `ExpertiseStore` is deliberately a
  local implementation behind the port, exactly as `ChromaVectorStore`
  is for documents; the production data-source swap is a later phase.
- **A staleness / update mechanism** beyond a per-record `last_updated`
  timestamp that the answer surfaces. Live sync from a source system is
  Phase 4+ (Discovery Findings §9.8 asks for the "update mechanism"; the
  pilot answer is "a dated static snapshot, and the answer tells the
  user how old its evidence is").
- **Access-control enforcement** — the expertise dataset is a skills
  directory scoped to be low-sensitivity by construction (§2.3), the
  same posture the pilot corpus takes. This sidesteps the confidentiality
  problem for B; it does not solve it.
- A web UI — CLI is sufficient.

**The synthesized-data note.** Meridian Advisory and its staff are
fictional (confirmed with the user; see `CLAUDE.md`). The HR expertise
data Discovery Findings §9.8 describes will never arrive. `data/expertise/`
is a deliberately, transparently synthesized stand-in, built against
realistic firm structure and grounded in the existing pilot corpus (a
consultant's "authored thought leadership" points at real files in
`data/corpus/`). This is the same convention the synthetic corpus
(Phase 1) and the synthesized query log (Phase 2) already follow — the
dataset's own header and `data/expertise/README.md` state it plainly.

## 2. The expertise data model

### 2.1 Shape

One record per consultant. Proposed fields (final schema is P3-1's
deliverable):

| Field | Type | Notes |
|---|---|---|
| `person_id` | str | stable slug, e.g. `c0142` |
| `name` | str | synthesized, non-real |
| `title` | str | Analyst / Consultant / Engagement Manager / Principal / Partner |
| `office` | str | a small fixed set (London, NYC, Singapore, …) |
| `practice` | str | one of the corpus's practice areas (pricing, cost transformation, M&A, operating model, …) |
| `skills` | list of `{topic, level, basis}` | `topic` from the corpus topic vocabulary; `level` self-assessed 1–5; `basis` = `self_reported` or `evidenced` |
| `project_history` | list of `{industry, topic, role, year}` | **no client names** — industry + topic only, mirroring the corpus's own anonymization posture (§2.3) |
| `authored` | list of corpus paths | thought-leadership / methodology docs this person is credited on — verified against real `data/corpus/` filenames |
| `languages` | list of str | |
| `last_updated` | date | per-record snapshot date; the B answer surfaces it |

### 2.2 Evidenced vs. claimed — the core modelling decision

Discovery Findings §1.2B names the primary quality risk as "stale or
self-reported expertise data." The model draws that line explicitly:

- **Evidenced** expertise = backed by `project_history` entries and/or
  `authored` corpus documents. Strong signal.
- **Claimed** expertise = a `skills` entry with `basis: self_reported`
  and no corroborating project/authorship. Weaker signal, and the B
  answer must label it as self-reported rather than presenting it as
  equivalent.

Retrieval ranking (§3.2) weights evidenced over claimed; generation
(§3.3) never presents a purely-claimed match without flagging it.

### 2.3 Confidentiality posture

The expertise dataset is scoped to be low-sensitivity the same way the
pilot corpus is:

- `project_history` records **industry + topic + role + year**, never a
  client name or an identifying engagement detail. "Retail pricing, 2023,
  workstream lead" — not "Project Atlas for [retailer]".
- No compensation, performance-rating, tenure, or personal data. This is
  a *skills directory*, not an HR record.
- `authored` points only at the already-public pilot corpus.

This is a deliberate sequencing choice, not a claim that expertise data
is generally safe — the same honesty the engagement takes everywhere
else (Discovery Findings §8: the prior tool burned trust with naive
security promises).

### 2.4 Size

Proposed: **~120–150 synthesized consultants**, not the full 600.
Rationale: enough that retrieval has to genuinely discriminate (multiple
plausible people per query, a long tail of weak matches), few enough to
generate with real variation and to label eval cases against by hand.
The firm is described as 600-strong; the dataset header notes it models a
representative slice. **Open decision for review** — go to the full 600
if the extra realism is worth the generation and labeling cost.

## 3. Architecture

### 3.1 The `ExpertiseStore` port

A thin interface, per design constraint #1 (swappable ports), sibling to
`VectorStore`:

```
class ExpertiseStore(ABC):
    def add(self, people: list[Person], embeddings: list[list[float]]) -> None: ...
    def search(self, embedding: list[float], k: int,
               where: dict | None = None) -> list[PersonMatch]: ...
    def count(self) -> int: ...
```

`PersonMatch` carries the `Person` record, a similarity `score`, and
(populated by the retrieval layer, not the store) the matched evidence.
Phase 3 ships one implementation — a local Chroma collection **separate
from the document collection**, holding one embedded profile summary per
person plus the structured fields as metadata. Phase 4 swaps this for a
real people-index behind the same interface.

Profile summaries are embedded with the **existing `Embedder` port** —
no new embedding dependency. The summary text is generated from the
structured record (`chunk_embedding_text()`'s analogue for people).

### 3.2 Expertise retrieval (`src/tessera/retrieval/expertise.py`)

`find_experts(query, embedder, store, where=None) -> ExpertiseResult` —
pure with respect to infrastructure per constraint #6 (ports injected,
returns data, no I/O / env / print).

1. Embed the query, semantic-search the profile collection for a
   candidate pool.
2. Re-rank by an evidence-strength score: matched `project_history` count
   and recency, matched `authored` docs, `skills.level` — with
   `evidenced` weighted above `self_reported`.
3. Return the top *N* people, each with the specific evidence that
   surfaced them (so generation can cite it and the eval can check it).

Structured pre-filtering via `where` (e.g. `practice`, `office`) is the
B-path analogue of A's metadata filtering — available from the start
here, since the query often names a practice or location.

### 3.3 B generation (`src/tessera/generation/`)

- `EXPERTISE_ANSWER_SYSTEM_PROMPT` in `prompts.py`: answer only from the
  provided person records; name the people; for each, cite the concrete
  evidence (`[project: retail pricing, 2023]`, `[authored:
  pricing-strategy-value-based-pricing.md]`); state when a match rests on
  self-reported skills only; surface the snapshot date; if no one clears
  a relevance floor, say so plainly (the retained "not yet supported"
  message's spirit, now "we don't have an obvious expert on that").
- `generate_expertise_answer(result, llm) -> GeneratedAnswer` in a new
  `generation/expertise.py` (or folded into `answer.py` — P3-4's call),
  mirroring `generate_answer()`'s shape: a relevance floor that
  short-circuits to a fixed no-match message with **zero LLM calls**
  when nothing qualifies.

### 3.4 Router and pipeline

- `router.terminal_response_for()` no longer returns a message for
  `Archetype.EXPERTISE` — only D stays terminal. `NOT_YET_SUPPORTED_
  MESSAGE` is removed or repurposed as the no-match fallback.
- `pipeline.answer_query()` gains a B branch: `route` → `find_experts` →
  `generate_expertise_answer`, returning the same `AnswerResult` shape as
  A/C (one shape regardless of archetype — the existing contract).
- `retriever.retrieve()` keeps raising `ValueError` for B (B has its own
  path now; it must never reach the document retriever).
- `cli.py`: `tessera query` handles B with no special-casing —
  `answer_query()` already returns one shape. A new `tessera index-people`
  (or a flag on `ingest`) builds the profile collection.

## 4. Evaluation

### 4.1 B metrics

- **Person recall@k** — of the `relevant_people` labelled for a case, how
  many appear in the top-k returned. Direct analogue of document
  recall@k.
- **Person MRR** — rank of the first relevant person.
- **Answer groundedness (LLM-judge, 1–5)** — every named person and every
  cited evidence item traces to a provided record; no invented people or
  projects.
- **Answer relevance (LLM-judge, 1–5)** — the people named actually fit
  the question, and self-reported-only matches are flagged.
- Precision@k tracked, reported, not gated — same labeling-completeness
  confound as the A/C side (`QUALITY_BAR.md`).

### 4.2 How the bar extends

`evals/QUALITY_BAR.md` gains a B section. **Proposed thresholds
(provisional — see below):** person recall@k ≥ 0.80, person MRR ≥ 0.90,
B groundedness / relevance ≥ 4.5, per-case person-recall > 0.00.

B thresholds enter **provisional (reported, not gated)** for the first
Phase 3 sweep, then are gated once a label-completeness audit
(§4.3) confirms the `relevant_people` sets are sound — the same staged
approach P2-1 → P2-2 took with precision. `--check` gates only what is
marked gated at any given time.

### 4.3 The B eval cases

- The 10 existing B cases (`ql018`–`ql022`, `ql040`–`ql042`, `q003`,
  `q004`) currently have empty `relevant_sources` / `ideal_answer`.
  Rewrite each with `relevant_people` (verified `person_id`s from the
  synthesized dataset) and an `ideal_answer` for the judge.
- Add a `relevant_people` field to the case schema and `evals/README.md`.
- Consider 2–3 **no-match** B cases (a query for expertise the firm
  genuinely doesn't have) to exercise the zero-LLM-call fallback — the
  B-side analogue of the off-corpus A refusal.
- `placeholder.yaml`'s 2 B cases (`q003`/`q004`) stay in the held-out
  check-set.

## 5. Task sequence

Work in order. Each task: branch off `main`, small commits, PR, merge
commit (`CLAUDE.md` git workflow). Run `pytest` and — for any task
touching retrieval / prompts / the eval set — a fresh
`tessera eval --check` sweep, pasted into the PR body. Stop after each
task and report against its acceptance check.

### P3-1 — Expertise dataset + schema
- Commit this document.
- Define the record schema (`data/expertise/README.md` + a
  `Person` dataclass / loader in `src/tessera/ingestion/`).
- Synthesize ~120–150 consultant records (`data/expertise/people.yaml`
  or one file per person — P3-1's call). Realistic distribution across
  practice / office / seniority; deliberate overlap so multiple people
  plausibly match a query; a weak-signal long tail.
- Every `authored` path mechanically verified against real
  `data/corpus/` filenames. No client names anywhere in
  `project_history`.
- Loader validates the schema at load time (mirrors `loader.py`).

**Acceptance:** dataset loads, every record validates, every `authored`
path resolves to a real corpus file, `project_history` contains no
client names; a spot-check confirms realistic variation.

### P3-2 — `ExpertiseStore` port + local implementation
- `store/base.py` (or a new module): `ExpertiseStore` interface +
  `Person` / `PersonMatch` dataclasses.
- Local Chroma implementation, a **separate collection** from documents.
- Profile-summary text generation from the structured record; embed with
  the existing `Embedder`.
- `tessera index-people` CLI command (or `ingest --people`).
- Tests: interface swap proven with a fake, same pattern as
  `test_indexing.py`.

**Acceptance:** all ~120–150 people indexed; a manual query returns
plausible people; swapping the implementation needs no change outside the
store module.

### P3-3 — Expertise retrieval path
- `retrieval/expertise.py`: `find_experts()` — semantic candidate pool →
  evidence-strength re-rank → top-N with per-person evidence. Pure per
  constraint #6.
- `where` pre-filtering on `practice` / `office`.
- Tests against fake `Embedder` / `ExpertiseStore`: ranking prefers
  evidenced over claimed; `where` filter passes through; evidence is
  attached to each result.

**Acceptance:** for a hand-checked query ("who knows pharma pricing"),
the people with real pharma-pricing project history and/or authored docs
rank above people who only self-tagged the skill.

### P3-4 — B generation + router/pipeline/CLI wiring
- `EXPERTISE_ANSWER_SYSTEM_PROMPT` + `generate_expertise_answer()` with a
  zero-LLM-call no-match fallback.
- `router.terminal_response_for()`: B no longer terminal.
- `pipeline.answer_query()`: B branch, same `AnswerResult` shape.
- `cli.py`: `tessera query` handles B; no-match path verified.
- Tests: fake-LLM unit tests for the prompt selection, evidence
  citation, self-reported flagging, and the no-match short-circuit;
  a real-dataset integration test proving B never touches the document
  retriever.
- Live spot-check: 3–4 B queries against real NVIDIA NIM, all four
  archetypes still route correctly.

**Acceptance:** `tessera query "who at the firm knows about <topic>"`
returns named people with cited evidence; a query for absent expertise
returns the no-match message with zero LLM calls; A/C/D behaviour
unchanged.

### P3-5 — Eval harness + bar extension
- `evals/metrics.py`: person recall@k / MRR; extend `judge_answer()` (or
  a B variant) for person-answers.
- `evals/harness.py`: B cases run the real B path and produce metrics;
  `run_case()` / `run_harness()` / `format_report()` / the bar-check
  handle B aggregates.
- `evals/cases/`: rewrite the 10 B cases with `relevant_people` +
  `ideal_answer`; add 2–3 no-match cases; schema doc updated.
- `evals/QUALITY_BAR.md`: B section, thresholds **provisional** for this
  first sweep.
- Label-completeness audit of `relevant_people`; then gate the B
  thresholds.
- Tests: metric logic (deterministic, no LLM); bar-check with B
  thresholds.

**Acceptance:** `tessera eval --check` reports A/C **and** B metric
blocks; a full sweep is recorded in `checkpoint.md`; every
`relevant_people` id resolves to a real record; B thresholds gated after
the audit.

### P3-6 — Phase 3 exit
- `README.md` — archetype B moves from "not built" to built; phase table,
  architecture diagram, the quality-bar section updated.
- `checkpoint.md` — final clean full sweep, Phase 3 close entry; confirm
  Phase 1 exit criteria + the full (A/C/B) bar.
- Tag **`v0.3.0`**.

**Acceptance:** a clean full sweep passes the extended bar; `v0.3.0`
tagged and pushed.

## 6. Notes and risks

- **B retrieval is a different problem from A/C.** It is structured-data
  ranking with a semantic assist, not document RAG. The risk is
  over-indexing on the embedding similarity and under-weighting the
  structured evidence — the P3-3 acceptance check (evidenced people beat
  self-taggers) exists specifically to catch that.
- **Synthesized-dataset realism.** ~150 fabricated people is a small,
  self-authored world; retrieval can look better than it would against a
  messy real directory. Mitigations: a deliberate weak-signal long tail;
  no-match eval cases; `placeholder.yaml`'s B cases held out; explicit
  honesty in the docs. Same posture as the Phase 2 overfitting note.
- **Confidentiality creep.** It is easy to make `project_history` more
  "realistic" by adding identifying detail. The P3-1 acceptance check
  (no client names) is a hard gate, not a guideline.
- **The `NvidiaClient` request-timeout** note (no timeout, openai SDK
  600s default) still stands from Phase 2 — user decided 2026-09-04 to
  leave it. Not re-opened by Phase 3.
- **Judge noise** on the small eval set: the Phase 2 exit sweep showed
  relevance clearing the 4.5 gate by only 0.10. Adding B cases changes
  the aggregate; watch that the combined groundedness/relevance still
  clears comfortably, and keep the B thresholds provisional until the
  audit for exactly this reason.
- **Carry-forward still open from Phase 2:** the narrow-archetype-A
  relevance margin (adaptive-`k` candidate lever, `checkpoint.md` Notes).
  Independent of Phase 3 — fold it in only if a Phase 3 sweep makes it
  load-bearing.
- **Estimated effort:** ~5–6 work sessions.

## 7. What changes in CLAUDE.md when this plan is adopted

- "Phase 1 objective and boundaries" / the do-not-build list: archetype B
  moves off the "explicitly NOT" list; the list gains "real HR-system
  integration" and "expertise staleness sync" as the Phase 4+ line.
- The technology table gains an `ExpertiseStore` row (local Chroma
  collection → real people-index in Phase 4).
- Design constraint #3 ("archetypes are first-class") gets a sentence
  noting B is now a third distinct retrieval path, not a refusal.
- Working conventions: the pre-PR quality-gate line extends to the B
  eval cases and `retrieval/expertise.py` / `generation/expertise.py`.
- `docs/` list gains this document as the Phase 3 authority.

## 8. Open decisions for review

1. **Dataset size** — ~120–150 (proposed) vs. the full 600.
2. **B threshold values** — person recall@k ≥ 0.80 / MRR ≥ 0.90 /
   judge ≥ 4.5 mirrors the A/C bar; is that the right bar for a
   person-answer, or should person-recall be stricter (a missed expert
   is arguably worse than a missed document)?
3. **`generation/expertise.py` vs. folding B into `answer.py`** — a
   structural call better made once P3-3's result shape is concrete;
   flagged here so it is a conscious decision, not a default.
4. **No-match cases** — 2–3 proposed; enough, or does the B fallback
   deserve its own small labelled set the way off-corpus A refusals
   arguably should?
