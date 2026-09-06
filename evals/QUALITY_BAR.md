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

`k` is `evals.harness.DEFAULT_K` (5). "A/C cases" are cases whose
`relevant_sources` is non-empty; B/D cases never reach retrieval and are
excluded from every retrieval metric.

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

Phase 2 exit sweep (P2-5) — full clean `tessera eval --check`,
**2026-09-06, 50/50 cases, zero errors**:

| Metric | Value | Bar |
|---|---|---|
| Routing accuracy | 100% | ✅ |
| Mean recall@k | 0.95 | ✅ |
| Mean MRR | 0.97 | ✅ |
| Mean groundedness | 4.77 | ✅ |
| Mean relevance | 4.60 | ✅ (thin — see below) |
| Per-case recall > 0.00 | yes (min 0.50) | ✅ |
| Mean precision@k | 0.42 | (not gated) |

`=> PASS (gated thresholds)`. Every gated threshold passes on a
reproducible clean sweep — this is the Phase 2 exit gate (Solution
Design §6), met.

**Relevance clears by only 0.10**, and it is the noisiest metric
(4.60–4.75 across recent sweeps). The cause is the P2-4 A-path
diversification trade-off: narrow single-target lookups ("Do we have a
framework for X?") now return 5 same-family documents where the query
wanted one, and the judge marks a few of them down for breadth
(`ql007`, `ql004`, `ql027`, `ql028`). Carried forward as the first
post-Phase-2 tuning item — candidate lever: adaptive `k` for archetype A
(fewer documents when the top hit dominates on score, all
`LOOKUP_TOP_K` when the family scores are tight). See `checkpoint.md`
"Notes / open flags".

**Precision fell 0.72 → 0.42** with the same P2-4 change and stays
ungated — the label audit (P2-2) came back with only 4 under-labeled
cases fixed, so the low score is the corpus's deliberate near-duplicate
hard-negatives plus genuinely-relevant-but-unlabeled adjacent docs, not
a labeling gap worth gating against.

## Regression discipline

Per `CLAUDE.md` "Working conventions": any PR that touches `retriever.py`,
`router.py`, `chunker.py`, `generation/` (including any prompt), or the
eval set must paste a fresh `tessera eval --check` result — the full
report plus its `=> PASS/FAIL` line — into the PR body. A gated failure
blocks the merge.

## Changing the bar

The thresholds live in `evals.harness.QualityBar`. Changing a gated
threshold is a deliberate decision that needs the user's sign-off and a
note here recording what changed and why — it is not a routine code edit.
