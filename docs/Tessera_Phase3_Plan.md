# Tessera — Phase 3 Plan (Claude Code Brief)

**Status: ADOPTED 2026-09-07** (drafted + reviewed same day, PR #38).
Phase 3 is now driven by this document's §5 task sequence; see
`checkpoint.md` for current standing. Phase 2 is complete and tagged
`v0.2.0`; all 5 Phase 1 exit criteria and all 6 gated Phase 2 bar
thresholds hold. Companion documents (`Tessera_Discovery_Findings.md`,
`Tessera_Solution_Design.md`) carry the problem context and full
architecture; `Tessera_Phase2_Plan.md` is the immediate predecessor and
the model for this document's shape.

The four open decisions this draft raised were resolved in review
(2026-09-07); they are folded into the sections below and §8 keeps the
record.

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
  path; a no-match message fires from generation (not the router) when
  nobody clears the relevance floor
- `pipeline.answer_query()` and `cli.py` handle B
- Eval harness: B cases get real metrics (person recall@k / MRR, a
  B-answer judge); the 10 existing B cases rewritten with
  `relevant_people` + `ideal_answer`; a dedicated no-match set
- `evals/QUALITY_BAR.md`: B thresholds added (person recall@k ≥ 0.90 —
  stricter than the A/C bar)
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

**The full ~600 consultants** (decision 1, resolved in review). Matches
the firm Discovery describes, and gives retrieval a genuinely hard
discrimination problem — many plausible people per query, a deep
weak-signal tail. Cost is borne in two places and both are managed:

- **Generation** — a scripted, parameterized generator (fixed
  distributions over practice / office / seniority / skill counts; a
  name pool; project/authorship sampling weighted so evidenced expertise
  clusters realistically). Not 600 hand-written records. The generator
  script is committed alongside the data so the dataset is reproducible.
- **Eval labeling** — `relevant_people` for a B case is labelled by
  running the P3-3 retrieval over the query, reading the top ~15–20
  candidates, and marking who genuinely belongs — not by scanning 600.
  This is the same method P2-2 used to audit A/C `relevant_sources`
  against retrieved-and-cited docs, and P3-5's label audit re-checks it.

The dataset header and `data/expertise/README.md` state that it is
generated, with the generator's parameters, so it is reproducible and
its shape is inspectable.

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
- `generate_expertise_answer(result, llm) -> GeneratedAnswer` in a **new
  `generation/expertise.py` module** (decision 3, resolved in review —
  not folded into `answer.py`; B's evidence-citation and
  self-reported-flagging logic is different enough from document
  citation that a shared module would blur both). It mirrors
  `generate_answer()`'s *shape*: a relevance floor that short-circuits
  to a fixed no-match message with **zero LLM calls** when nothing
  qualifies. Any genuinely shared helper (e.g. the markdown-fence-
  tolerant JSON parse, `filter_relevant()`'s analogue) is lifted to a
  shared location rather than duplicated or forced into `answer.py`.

### 3.4 Router and pipeline

- `router.terminal_response_for()` no longer returns a message for
  `Archetype.EXPERTISE` — only D stays terminal. `NOT_YET_SUPPORTED_
  MESSAGE` is retired; its role (a plain-language "the system can't help
  here") passes to the B no-match message, which now fires from
  generation when nobody clears the relevance floor, not from the router.
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

`evals/QUALITY_BAR.md` gains a B section. Thresholds:

| Metric | Threshold | Gated? |
|---|---|---|
| Person recall@k — B | **≥ 0.90** | yes (after audit) |
| Person MRR — B | ≥ 0.90 | yes (after audit) |
| B groundedness (LLM-judge, 1–5) | ≥ 4.5 | yes (after audit) |
| B relevance (LLM-judge, 1–5) | ≥ 4.5 | yes (after audit) |
| Per-case person-recall — B | > 0.00 (no total misses) | yes (after audit) |
| No-match set — correct-refusal rate | 100% | yes (after audit) |
| Person precision@k — B | reported | no |

Person recall@k is set **stricter than the A/C recall bar (0.90 vs.
0.80)** (decision 2, resolved in review): a missed expert is a worse
failure than a missed document — for a lookup the user still has the
other retrieved docs, but "who knows about X" returning the wrong
shortlist sends the user to the wrong person entirely. `k` for B is the
shortlist length the answer presents (P3-3 fixes it; start at 5).

B thresholds enter **provisional (reported, not gated)** for the first
Phase 3 sweep, then are gated once the §4.3 label-completeness audit
confirms the `relevant_people` sets are sound — the same staged approach
P2-1 → P2-2 took with precision. `--check` gates only what is marked
gated at any given time.

### 4.3 The B eval cases

- The 10 existing B cases (`ql018`–`ql022`, `ql040`–`ql042`, `q003`,
  `q004`) currently have empty `relevant_sources` / `ideal_answer`.
  Rewrite each with `relevant_people` (verified `person_id`s from the
  synthesized dataset) and an `ideal_answer` for the judge.
- Add a `relevant_people` field to the case schema and `evals/README.md`.
- **A dedicated no-match set** (decision 4, resolved in review):
  `evals/cases/expertise_nomatch.yaml`, ~5–6 cases, each a query for
  expertise the firm genuinely doesn't have (a niche topic absent from
  every profile). It is the B-side analogue of an off-corpus A refusal
  set — Task 7's genai-architect review flagged that correct-refusal
  behaviour (CLAUDE.md constraint #2) is currently outside the eval's
  reach, and this closes that gap for B. Scored on one metric: did the
  system return the fixed no-match message with **zero LLM calls**
  (100%, gated after the audit). Held separate from the scored B cases
  so it can't distort the recall / judge aggregates.
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
- **A committed generator script** (`data/expertise/generate.py` or
  under `scripts/`) that produces **~600 consultant records** from fixed
  parameters — practice / office / seniority distributions, a name pool,
  skill-count ranges, and project/authorship sampling weighted so
  evidenced expertise clusters realistically and a weak-signal long tail
  exists. Committed alongside its output (`data/expertise/people.yaml`,
  or sharded by practice if one file is unwieldy) so the dataset is
  reproducible and inspectable.
- Every `authored` path emitted by the generator mechanically verified
  against real `data/corpus/` filenames. **No client names anywhere in
  `project_history`** — the generator only ever emits industry + topic +
  role + year.
- Loader validates the schema at load time (mirrors `loader.py`).

**Acceptance:** dataset regenerates deterministically from the script;
~600 records load and validate; every `authored` path resolves to a real
corpus file; `project_history` contains no client names; a spot-check
confirms realistic variation and a genuine weak-signal tail.

### P3-2 — `ExpertiseStore` port + local implementation
- `store/base.py` (or a new module): `ExpertiseStore` interface +
  `Person` / `PersonMatch` dataclasses.
- Local Chroma implementation, a **separate collection** from documents.
- Profile-summary text generation from the structured record; embed with
  the existing `Embedder`.
- `tessera index-people` CLI command (or `ingest --people`).
- Tests: interface swap proven with a fake, same pattern as
  `test_indexing.py`.

**Acceptance:** all ~600 people indexed; a manual query returns
plausible people; swapping the implementation needs no change outside the
store module.

### P3-3 — Expertise retrieval path
- `retrieval/expertise.py`: `find_experts()` — semantic candidate pool →
  evidence-strength re-rank → top-`k` (k = 5) with per-person evidence.
  Pure per constraint #6.
- `where` pre-filtering on `practice` / `office`.
- A retrieval-only diagnostic script (zero LLM), the P3 analogue of
  `evals/tune_retrieval.py`, for hand-checking rank quality and for
  labelling `relevant_people` in P3-5.
- Tests against fake `Embedder` / `ExpertiseStore`: ranking prefers
  evidenced over claimed; `where` filter passes through; evidence is
  attached to each result.

**Acceptance:** for a hand-checked query ("who knows pharma pricing"),
the people with real pharma-pricing project history and/or authored docs
rank above people who only self-tagged the skill.

### P3-4 — B generation + router/pipeline/CLI wiring
- **New `generation/expertise.py`**: `EXPERTISE_ANSWER_SYSTEM_PROMPT` (in
  `prompts.py`) + `generate_expertise_answer()` with a zero-LLM-call
  no-match fallback. Shared helpers lifted to a common location, not
  duplicated from `answer.py`.
- `router.terminal_response_for()`: B no longer terminal; the fixed
  no-match message replaces `NOT_YET_SUPPORTED_MESSAGE`'s role.
- `pipeline.answer_query()`: B branch, same `AnswerResult` shape.
- `cli.py`: `tessera query` handles B; no-match path verified.
- Tests: fake-LLM unit tests for evidence citation, self-reported
  flagging, and the no-match short-circuit; a real-dataset integration
  test proving B never touches the document retriever.
- Live spot-check: 3–4 B queries against real NVIDIA NIM, all four
  archetypes still route correctly.

**Acceptance:** `tessera query "who at the firm knows about <topic>"`
returns named people with cited evidence and self-reported-only matches
flagged; a query for absent expertise returns the no-match message with
zero LLM calls; A/C/D behaviour unchanged.

### P3-5 — Eval harness + bar extension
- `evals/metrics.py`: person recall@k / MRR; a B-answer judge (person +
  evidence groundedness, fit + self-reported-flagging relevance).
- `evals/harness.py`: B cases run the real B path and produce metrics;
  `run_case()` / `run_harness()` / `format_report()` / the bar-check
  handle B aggregates and the no-match set's correct-refusal rate.
- `evals/cases/`: rewrite the 10 B cases with `relevant_people` (labelled
  via the P3-3 diagnostic — read the top ~15–20, mark who belongs) +
  `ideal_answer`; new `evals/cases/expertise_nomatch.yaml` (~5–6 cases);
  `relevant_people` added to the case schema + `evals/README.md`.
- `evals/QUALITY_BAR.md`: B section per §4.2, thresholds **provisional**
  for the first sweep.
- Label-completeness audit of every `relevant_people` set; **then** gate
  the B thresholds (person recall@k ≥ 0.90, MRR ≥ 0.90, judge ≥ 4.5,
  per-case recall > 0, no-match refusal rate 100%).
- Tests: metric logic (deterministic, no LLM); bar-check with the B
  thresholds; no-match short-circuit counted correctly.

**Acceptance:** `tessera eval --check` reports A/C **and** B metric
blocks plus the no-match refusal rate; a full sweep is recorded in
`checkpoint.md`; every `relevant_people` id resolves to a real record;
B thresholds gated after the audit; A/C thresholds unchanged and still
passing.

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
- **Synthesized-dataset realism.** ~600 generator-produced people is
  still a self-authored world — retrieval can look better than it would
  against a messy real directory, and a generator can accidentally make
  the evidenced/claimed split *too* clean to be a real test. Mitigations:
  a deliberate weak-signal long tail and near-miss people (right
  industry, wrong topic; adjacent skill only); the dedicated no-match
  set; `placeholder.yaml`'s B cases held out; the P3-1 spot-check
  explicitly looks for a realistic tail, not just valid records; honesty
  in the docs. Same posture as the Phase 2 overfitting note.
- **Generator determinism.** The dataset must regenerate byte-stable from
  a fixed seed, or the committed `people.yaml` and the committed script
  drift apart and eval labels rot. P3-1 pins the seed and the acceptance
  check re-runs the generator and diffs.
- **Confidentiality creep.** It is easy to make `project_history` more
  "realistic" by adding identifying detail. The generator only ever
  emits industry + topic + role + year; the P3-1 acceptance check
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
- **Stricter B recall bar (0.90) is a real commitment.** If the first
  gated sweep can't clear it, the options are a genuine retrieval fix
  (P3-3 re-tune) or a documented, user-signed-off threshold change — not
  quietly dropping to 0.80. Recorded here so that's a conscious call
  later, not a surprise.
- **Estimated effort:** ~6–7 work sessions (up from the ~5–6 first
  estimate — the full 600-person dataset and its generator, plus the
  dedicated no-match set, add scope over the initial draft).

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

## 8. Decisions resolved in review (2026-09-07)

1. **Dataset size → the full ~600 consultants.** Produced by a committed,
   seeded generator script, not hand-written. Rationale and cost
   handling in §2.4.
2. **B recall bar → stricter than A/C: person recall@k ≥ 0.90** (vs.
   0.80 for documents). A missed expert sends the user to the wrong
   person entirely, with no other retrieved results to fall back on.
   §4.2.
3. **B generation → a new `generation/expertise.py` module**, not folded
   into `answer.py`. Evidence-citation and self-reported-flagging logic
   is distinct enough that sharing a module would blur both. §3.3.
4. **B fallback → a dedicated labelled set**,
   `evals/cases/expertise_nomatch.yaml` (~5–6 cases), scored on
   correct-refusal rate (100%, gated after audit), held separate from
   the scored B cases. Closes the constraint-#2 gap the Task 7 review
   flagged, for B. §4.3.
