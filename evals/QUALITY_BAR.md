# Tessera — internal quality bar

The bar every retrieval- or prompt-affecting change is measured against.
Agreed with the user 2026-09-02; adopted with the Phase 2 plan
(`docs/Tessera_Phase2_Plan.md` §2) on 2026-09-04.

This is the **manual** precursor to the Phase 5 CI gate (Solution Design
§5). Until CI exists, enforcement is: run `tessera eval --check` locally
and paste the result into the PR (see "Regression discipline" below).

## The bar

| Metric | Threshold | Gated? |
|---|---|---|
| Routing accuracy | ≥ 95% | **yes** |
| Mean recall@k — A/C cases with `relevant_sources` | ≥ 0.80 | **yes** |
| Mean MRR — A/C | ≥ 0.90 | **yes** |
| Mean groundedness (LLM-judge, 1–5) | ≥ 4.5 | **yes** |
| Mean relevance (LLM-judge, 1–5) | ≥ 4.5 | **yes** |
| Per-case recall — A/C | > 0.00 (no total misses) | **yes** |
| Mean precision@k — A/C | tracked and reported | no |

### Archetype B (Phase 3)

Added with P3-5 (`docs/Tessera_Phase3_Plan.md` §4.2). Applies whenever
the harness is given an expertise store (`tessera eval` always is —
it requires `tessera index-people`).

| Metric | Threshold | Gated? |
|---|---|---|
| Mean person recall@k — B cases with `relevant_people` | ≥ 0.90 | **yes** |
| Mean person MRR — B | ≥ 0.90 | **yes** |
| B groundedness (LLM-judge, 1–5) | ≥ 4.5 | **yes** |
| B relevance (LLM-judge, 1–5) | ≥ 4.5 | **yes** |
| Per-case person recall — B | > 0.00 (no total misses) | **yes** |
| No-match set — correct-refusal rate | 100% | **yes** |
| Mean person precision@k — B | tracked and reported | no |

**History.** The B rows entered *provisional* (reported, not gated) for the
first Phase 3 sweep, as plan §4.2 stages it. That sweep (P3-5, 2026-09-29,
55/55 cases) scored person recall **0.89** against 0.90, carried by `ql019`
("who **led** … recently", 0.40): retrieval had no notion of role or
recency. The retrieval pass that followed reads "led/ran/managed" and
"recently/latest" from the query (`parse_intent` in
`retrieval/expertise.py`) and the rows were gated in the same change
(`QualityBar.gate_expertise`, default `True`; `False` restores report-only).

Person recall is stricter than A/C's 0.80: a missed expert sends the user
to the wrong person entirely. It is scored against `min(|relevant|, k)`
(see `evals/README.md`) so a case with more genuine experts than
shortlist slots can still reach 1.0. B judge scores are kept apart from
the A/C means, as are all person metrics. "Correct refusal" means the
fixed no-match message with zero generation LLM calls.

`k` is `evals.harness.DEFAULT_K` (5). "A/C cases" are cases whose
`relevant_sources` is non-empty; B/D cases are excluded from every
document-retrieval metric (B has its own person metrics, below).

### Cost (Phase 4)

| Metric | Threshold | Gated? |
|---|---|---|
| Mean cost per answer (routing + answer calls) | no budget agreed yet | **provisional** — reported, not gated |

Reported, with cost per answer by archetype, whenever every model in the
sweep has a price in `config.MODEL_PRICES` (a NIM-only sweep reports
tokens but no cost). The judge's calls are never counted — they are the
cost of measuring, not of answering. Once a budget is agreed with the
user, it becomes `QualityBar.max_mean_cost_per_answer_usd` and
`gate_cost` is flipped (Phase 4 plan §4), the same staging the B rows
went through. The prices are Anthropic's published rates, **not yet
verified against Bedrock's** — check them before quoting a sweep's cost.

### Freshness (Phase 4)

Added with P4-4 (`docs/Tessera_Phase4_Plan.md` §3.3, §4).

| Metric | Threshold | Gated? |
|---|---|---|
| Superseded document cited as current — A/C cases | 0 cases | **yes** |
| Newer-version note on the freshness cases | reported | no |

A case "cites a superseded document as current" when a chunk from a
document with `status: superseded` is among the sources shown to the
model, or — for the cases in `evals/cases/freshness.yaml` — when one of
the case's own `superseded_sources` is retrieved at all. The row covers
every A/C case, not only the freshness set; it is `n/a` (a failure) only
when no case reached document retrieval. The freshness cases are worded
in the retired method's vocabulary so the superseded version is the
closer match; they are also ordinary A/C cases, so recall/MRR and the
judge score them like any other. The note is appended by code, not the
model, and the judge grades the answer without it — it names a document
the model was never shown.

A gated metric with **no value** (e.g. `mean_recall` is `n/a` because the
eval set had no A/C cases) counts as a **failure**, not a skip — a sweep
that can't measure a gated dimension has not cleared the bar.

## Why precision@k is not gated

A low precision@k score is confounded by relevant-source labeling
completeness: a chunk that is retrieved, cited, and genuinely relevant
but simply not listed in the case's `relevant_sources` counts against
precision unfairly. Task P2-2 audits the labels. If they come back clean
and precision is still low, gating it is revisited then.

## Current standing

Phase 3 exit sweep (P3-6) — full clean `tessera eval --check`,
**2026-10-01, 55/55 cases, zero errors** (8 transient NVIDIA 503s, all
recovered by `RetryingLLMClient`):

| Metric | Value | Bar |
|---|---|---|
| Routing accuracy | 100% | ✅ |
| Mean recall@k (A/C) | 0.95 | ✅ |
| Mean MRR (A/C) | 0.97 | ✅ |
| Mean groundedness (A/C) | 4.83 | ✅ |
| Mean relevance (A/C) | 4.60 | ✅ (thin — see below) |
| Per-case recall > 0.00 (A/C) | yes (min 0.50) | ✅ |
| Mean precision@k (A/C) | 0.42 | (not gated) |
| Person recall@k (B) | 0.91 | ✅ (thin — see below) |
| Person MRR (B) | 1.00 | ✅ |
| B groundedness | 5.00 | ✅ |
| B relevance | 4.89 | ✅ |
| Per-case person recall > 0.00 (B) | yes (min 0.60) | ✅ |
| No-match correct-refusal rate (B) | 100% | ✅ |
| Person precision@k (B) | 0.91 | (not gated) |

`=> PASS (gated thresholds)`. Every gated threshold, A/C and B, passes on
a clean sweep — the Phase 3 exit gate (`docs/Tessera_Phase3_Plan.md` §5
P3-6), met. The A/C rows match the Phase 2 exit sweep (P2-5, 2026-09-06,
50/50) except groundedness, 4.77 → 4.83.

**A/C relevance clears by only 0.10**, and it is the noisiest metric
(4.57–4.75 across recent sweeps). The cause is the P2-4 A-path
diversification trade-off: narrow single-target lookups ("Do we have a
framework for X?") return 5 same-family documents where the query
wanted one, and the judge marks a few of them down for breadth
(`ql004`, `ql007`, `ql027`, `ql028` scored 3 again on the exit sweep).
Carried forward — candidate lever: adaptive `k` for archetype A (fewer
documents when the top hit dominates on score, all `LOOKUP_TOP_K` when
the family scores are tight). See `checkpoint.md` "Notes / open flags".

**B person recall clears by only 0.01**, on 9 labelled cases: one more
missed expert on any case fails the bar. Lowest cases on the exit sweep:
`ql019` (0.60, "who **led** … recently"), `ql041`/`ql042` (0.80).
Untried levers: a larger `CANDIDATE_K`, and a `where` filter derived from
a place or practice named in the query.

**Precision fell 0.72 → 0.42** with the P2-4 change and stays
ungated — the label audit (P2-2) came back with only 4 under-labeled
cases fixed, so the low score is the corpus's deliberate near-duplicate
hard-negatives plus genuinely-relevant-but-unlabeled adjacent docs, not
a labeling gap worth gating against.

## Regression discipline

Per `CLAUDE.md` "Working conventions": any PR that touches `retrieval/`
(`retriever.py`, `router.py`, `expertise.py`), `chunker.py`,
`generation/` (including any prompt), the corpus (`data/corpus/`, including
front-matter `status`), the expertise dataset or its
generator (`data/expertise/`), or the eval set must paste a fresh `tessera eval --check` result — the full
report plus its `=> PASS/FAIL` line — into the PR body. A gated failure
blocks the merge.

## Changing the bar

The thresholds live in `evals.harness.QualityBar`. Changing a gated
threshold is a deliberate decision that needs the user's sign-off and a
note here recording what changed and why — it is not a routine code edit.
