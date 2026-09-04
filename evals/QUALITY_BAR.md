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

Full sweep, 2026-09-02, 31/33 cases scored (2 transient NVIDIA 503s
excluded):

| Metric | Value | Bar |
|---|---|---|
| Routing accuracy | 100% | ✅ |
| Mean recall@k | 0.88 | ✅ |
| Mean MRR | 0.97 | ✅ |
| Mean groundedness | 5.00 | ✅ |
| Mean relevance | 4.89 | ✅ |
| Per-case recall > 0.00 | yes | ✅ |
| Mean precision@k | 0.72 | (not gated) |

Every gated threshold already passes. Phase 2 tuning (P2-3, P2-4) is
about **margin and robustness against a larger, audited eval set**, not
clawing up to a failing bar — unless the expanded ~50-case set surfaces
hard cases that break it.

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
