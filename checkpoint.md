# Tessera — Checkpoint

Last updated: 2026-08-17

## Status

Task 1 complete and merged (PR #2). Repo scaffold and synthetic corpus
exist. No retrieval/generation logic implemented yet — everything under
`src/tessera/` beyond `__init__.py` is a docstring stub naming which
task implements it.

## Done

- [x] Read `docs/Tessera_Phase1_Build_Plan.md`, `docs/Tessera_Discovery_Findings.md`,
      `docs/Tessera_Solution_Design.md` in full.
- [x] Created `CLAUDE.md` at repo root (plan §6).
- [x] Created this checkpoint file.
- [x] Created `/start-day` and `/end-day` custom commands.
- [x] Initialized git tracking, connected GitHub remote, adopted PR-based
      workflow (branch → PR → merge commit, branch-protected `main`, tags at
      phase boundaries, CI/CD deferred to Phase 5).
- [x] Switched Phase 1 LLM decision from Claude API to DeepSeek API (Claude
      via Bedrock remains the Phase 4 target).
- [x] Added README with Phase 1 architecture diagram.
- [x] **Task 1 — Repo scaffold and synthetic corpus** (PR #2, merged).
      Full `src/tessera/`, `evals/`, `tests/` scaffold stubbed per build
      plan §4. Corpus: 52 markdown files (38 methodology + 14
      thought-leadership) with 5-key YAML front matter. Independent Opus
      review + fixes applied: corrected a profit/revenue elasticity error,
      generalized an over-specific worked example (confidentiality), and
      added 11 structurally-complex documents (tables, checkboxes, deep
      heading nesting, code/formula fences, blockquotes, min/max-length
      outliers) so chunking/retrieval have real cases to discriminate
      between rather than a uniform corpus. All validated: front matter
      parses, `doc_type` matches directory, no real/identifiable company
      data.

## Next task to pick up

**Task 2 — Ingestion and chunking** (build plan §5, Task 2). Not started
— waiting on go-ahead.

Task 2 covers:
1. Implement `src/tessera/ingestion/loader.py` — reads the corpus with
   front-matter metadata intact.
2. Implement `src/tessera/ingestion/chunker.py` — section-aware chunking
   that respects markdown heading boundaries (not fixed-size), preserving
   framework coherence. Each chunk carries source metadata for citation.

The corpus now has real edge cases to test the chunker against: a table
(must not split mid-row), a 46-item nested checkbox list, 4-level heading
nesting (`##`→`###`→`####`), fenced code/formula blocks (must not split
inside a fence), a ~227-word doc (minimum-chunk-size path), a ~1,427-word
doc (multi-chunk-per-document path), and blockquotes.

**Acceptance check for Task 2:** chunk count reasonable; no chunk orphaned
from its source metadata; headings not split mid-section.

## Task sequence (build plan §5, for reference)

1. ~~Repo scaffold and synthetic corpus~~ — done (PR #2)
2. Ingestion and chunking ← **next**
3. Embedding and vector store behind interfaces
4. Archetype router
5. Archetype-aware retrieval
6. Grounded generation with citations
7. Evaluation harness
8. CLI and README

Work one task at a time. Stop after each and report against its acceptance
check before continuing to the next.

## Notes / open flags

- The corpus's `## Related Frameworks` sections (all 30 original methodology
  docs) are deliberately kept as near-duplicate, low-signal chunks — a
  conscious choice to serve as hard negatives for retrieval precision@k,
  not an oversight. Don't "clean these up" in a later task without
  revisiting this decision explicitly.
- Nothing yet has required an item from the "explicitly NOT in Phase 1"
  list in `CLAUDE.md` — flag here if that changes.
