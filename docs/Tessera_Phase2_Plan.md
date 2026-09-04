# Tessera — Phase 2 Plan (Claude Code Brief)

**Status: DRAFT — pending user review (drafted 2026-09-02).**

This document is the working brief for Phase 2 of Project Tessera, in the
same spirit as `Tessera_Phase1_Build_Plan.md`. Phase 1 is complete and
tagged `v0.1.0`; all 5 Phase 1 exit criteria were re-verified live on
2026-09-02 (see `checkpoint.md`). Companion documents
(`Tessera_Discovery_Findings.md`, `Tessera_Solution_Design.md`) carry the
problem context and full architecture.

---

## 0. Context in one paragraph

Phase 1 built a working, locally-run ingestion and retrieval core over the
synthetic pilot corpus, plus a runnable evaluation harness. Phase 2 is the
**credibility layer**: populate the eval set, tune retrieval and generation
against it, and commit to a documented internal quality bar that every
future retrieval/prompt change is measured against. This is the manual
precursor to the Phase 5 CI gate. Per Solution Design §6, the project is
"coherent and demoable if it stops after Phase 2 — a working, evaluated
knowledge assistant on the pilot corpus."

## 1. Objective and boundaries

**Objective:** the evaluation framework populated on the (synthesized,
permanent) query log, retrieval + generation tuned against it, and a
documented, enforced internal quality bar.

**Exit gate (Solution Design §6):** retrieval + answer metrics meet the
agreed internal bar (§2 below), on a ~50-case eval set, reproducibly.

**In scope for Phase 2**
- Formalizing and enforcing the internal quality bar
- Expanding the eval set from 33 to ~50 cases
- Relevant-source labeling completeness audit
- Retrieval constant re-tuning (grid search)
- Retrieval / prompt tuning to clear the bar
- Regression discipline: eval sweep required on every retrieval/prompt PR

**Explicitly NOT in Phase 2 — unchanged from Phase 1's do-not-build list**
- Archetype B (expertise-finding) build — Phase 3, blocked on HR data
  structure (Discovery Findings §9.8). B stays a routed "not yet
  supported" response.
- Archetype D — remains a refusal guardrail only.
- Any AWS deployment, Terraform, CI/CD, monitoring — Phases 4–5.
- PowerPoint/deck ingestion — not in the pilot corpus.
- Access-control enforcement — pilot corpus is low-sensitivity by
  construction.
- A web UI — CLI is sufficient.

The query log Discovery Findings §7/§10 describes will never literally
arrive (Meridian Advisory and its stakeholders are fictional — confirmed
with the user, see `CLAUDE.md`). `evals/cases/query_log.yaml` is a
deliberately, transparently synthesized stand-in. Phase 2 widens it; it
does not wait for a real one.

## 2. The agreed internal bar

Agreed with the user 2026-09-02.

| Metric | Threshold | Gated? |
|---|---|---|
| Routing accuracy | ≥ 95% | yes |
| Mean recall@k (A/C cases with `relevant_sources`) | ≥ 0.80 | yes |
| Mean MRR (A/C) | ≥ 0.90 | yes |
| Mean groundedness (LLM-judge, 1–5) | ≥ 4.5 | yes |
| Mean relevance (LLM-judge, 1–5) | ≥ 4.5 | yes |
| Per-case recall (A/C) | > 0.00 — no total misses | yes |
| Mean precision@k (A/C) | tracked and reported | no |

**Why precision is not gated:** a low precision@k score is confounded by
relevant-source labeling completeness — a chunk that is cited and
genuinely relevant but not listed in the case's `relevant_sources` counts
against precision unfairly. Task P2-2 audits the labels. If they come back
clean and precision is still low, gating it is revisited then.

**Current standing (full sweep, 2026-09-02, 31/33 cases scored; 2
transient NVIDIA 503s excluded):** routing 100%, recall 0.88, MRR 0.97,
groundedness 5.00, relevance 4.89, precision 0.72, no per-case
recall=0.00. Every gated threshold already passes. Phase 2 tuning is
therefore about **margin and robustness against a larger, audited eval
set**, not clawing up to a failing bar — unless the expanded set surfaces
new hard cases that break it, in which case the tuning tasks (P2-3, P2-4)
become load-bearing.

## 3. Known weak spots going in (2026-09-02 sweep)

- **Multi-source archetype-A recall gaps:** `q001` recall 0.40, `ql003`
  and `ql004` recall 0.50 — cases where `relevant_sources` lists 2+ docs
  and only one surfaces in top-k.
- **Archetype-C precision:** `ql017` 0.20, `ql012` / `ql014` 0.40, `q005`
  0.40. Possibly real, possibly a labeling artefact (see §2).
- **Retrieval constants are calibrated against a stale score
  distribution:** `LOOKUP_TOP_K`, `SYNTHESIS_CANDIDATE_K` /
  `SYNTHESIS_MAX_RESULTS` / `SYNTHESIS_MAX_PER_DOCUMENT` (`retriever.py`)
  and `RELEVANCE_THRESHOLD` (`generation/answer.py`) were all tuned before
  the 2026-08-27 title-aware chunk-embedding change shifted every score.
  P2-3 re-tunes them deliberately.

Routing (100%), groundedness (5.00), and relevance (4.89) are already
strong; the grounded-generation prompt itself is not a concern. The open
work is in retrieval.

## 4. Task sequence

Work in order. Each task: branch off `main`, small commits, PR, merge
commit (per `CLAUDE.md` git workflow). Run `pytest` and a fresh eval
sweep before opening each PR. Stop after each task and report against its
acceptance check before continuing.

### P2-1 — Formalize the bar + regression discipline
- Commit this document (`docs/Tessera_Phase2_Plan.md`).
- `evals/QUALITY_BAR.md` — the §2 table plus rationale.
- Harness: `run_harness()` result gains a bar-check; `format_report()`
  prints `PASS` / `FAIL` per threshold; `tessera eval --check` exits
  non-zero on any gated failure.
- `CLAUDE.md` — extend the existing "run pytest + eval harness locally
  before a PR" convention: any PR touching `retriever.py`, `router.py`,
  `chunker.py`, `generation/`, or a prompt must paste a fresh sweep +
  bar-check result in the PR body.
- Tests: bar-check logic (deterministic, no LLM).

**Acceptance:** `tessera eval --check` runs, prints per-threshold
PASS/FAIL, exits 0 against the current corpus and eval set.

### P2-2 — Expand eval set to ~50 + label-completeness audit
- Audit all 21 current A/C cases' `relevant_sources` against the corpus:
  for each, read every doc the system retrieved and cited, decide whether
  it belongs in the label set, record the decision in the case file's
  comments.
- Write ~17 new cases → target **A=20, B=10, C=15, D=5** (~50 total).
  Every `relevant_sources` path mechanically verified against real
  filenames; no id collisions with `q001`–`q008` / `ql001`–`ql025`.
- Hold `placeholder.yaml` (8 cases) as an overfitting check-set — P2-3
  tunes against `query_log.yaml` only.
- Update `test_harness.py` case-count assertions.

**Acceptance:** `load_cases()` parses ~50 cases; a full baseline sweep is
recorded in `checkpoint.md`; every `relevant_sources` path verified to
exist on disk.

### P2-3 — Retrieval constant grid-search re-tune
- New committed `evals/tune_retrieval.py`: a **retrieval-only sweep** —
  archetype taken from the case file, no router call, no judge, **zero
  LLM calls** — so the full grid runs in seconds.
- Grid: `LOOKUP_TOP_K {3,5,7,10}` × `SYNTHESIS_CANDIDATE_K {15,20,30}` ×
  `SYNTHESIS_MAX_RESULTS {8,10,12}` × `SYNTHESIS_MAX_PER_DOCUMENT {2,3}` ×
  `RELEVANCE_THRESHOLD {0.30,0.35,0.40}`.
- Objective: maximize mean(recall × precision) on `query_log.yaml`,
  subject to per-case recall > 0.00 and MRR ≥ 0.90; prefer current values
  when improvement is marginal (< 0.02); confirm the winner also holds on
  `placeholder.yaml`.
- Re-calibrate `RELEVANCE_THRESHOLD` against the on-/off-corpus probe
  queries described in `generation/answer.py`'s docstring — any threshold
  change moves the refusal boundary.
- Update constants in `retriever.py` and `answer.py`; update tests that
  assert on the values.
- One full LLM sweep confirming groundedness / relevance did not regress.

**Acceptance:** chosen config documented with its grid scores; full sweep
+ bar-check pasted in the PR; groundedness / relevance ≥ prior baseline.

### P2-4 — Close remaining bar gaps (conditional)
- Re-run the full sweep after P2-3. **If every gated threshold passes,
  this task is verification only.**
- If gaps remain:
  - **Multi-source A recall < 0.80** — candidate lever: wire metadata
    filtering through the pipeline. `retriever.retrieve()` already accepts
    `where`, but `pipeline.answer_query()` never passes it; the Phase 1
    Task 5 spec ("A uses narrow k *with metadata filtering*") is arguably
    unfinished here. Filter A by `doc_type` / `industry` / `topics`
    inferred from the query.
  - **Routing < 95% or a prompt gap** — targeted prompt edits plus a
    regression spot-check across all four archetypes (the pattern from the
    2026-08-27 tuning pass).

**Acceptance:** all gated thresholds pass on the full ~50-case set.

### P2-5 — Phase 2 exit
- `README.md` — Phase 2 results, the bar, built-vs-designed line updated.
- `checkpoint.md` — final clean full sweep, bar-check output, Phase 2
  close entry.
- Confirm all 5 Phase 1 exit criteria still hold and the Phase 2 bar is
  met.
- Tag **`v0.2.0`** (per `CLAUDE.md`: tags cut at phase boundaries once
  exit criteria are met).
- Optionally bring in `genai-architect` for a structural check at this
  point — not required (the persona agents are gated by convention to
  build-plan Tasks 6/7/8; Phase 2 is outside that scope).

**Acceptance:** a clean full sweep passes the bar; `v0.2.0` tagged and
pushed.

## 5. Notes and risks

- **Overfitting to synthetic data.** ~50 synthesized cases is a small
  tuning target. Mitigations: coarse grid; marginal-improvement tiebreak
  toward current values; `placeholder.yaml` held out as a check-set;
  explicit honesty in the docs that this is synthesized data (same
  convention as the pilot corpus itself).
- **Quota.** A full sweep is ~110 LLM calls at ~50 cases; the grid search
  adds zero (retrieval-only). Trivial against NVIDIA NIM's 10,000/day.
- **Transient NVIDIA 503s** under sweep load are expected (1–3 cases per
  large sweep). Task 7's per-case error handling isolates them; retry the
  specific failed case IDs rather than re-running the whole sweep. See
  `checkpoint.md` Notes.
- **Re-ingest after any embedding change.** `chunk_embedding_text()` or
  anything feeding `embedder.embed_documents()` requires `tessera ingest`
  before it takes effect (`ChromaVectorStore.add()` upserts in place).
- **Estimated effort:** ~4–5 work sessions.

## 6. What changes in CLAUDE.md when this plan is adopted

- Phase framing: "Phase 1 objective and boundaries" section gains a
  pointer to this document for Phase 2.
- Working conventions: the pre-PR quality-gate line is extended per P2-1
  (bar-check output required in PR bodies for retrieval/prompt changes).
- The technology table and design constraints are unchanged.
