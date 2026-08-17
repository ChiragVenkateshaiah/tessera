# Tessera — Checkpoint

Last updated: 2026-08-17

## Status

Task 2 complete and merged (PR #5). Corpus loads and chunks cleanly.
Retrieval/generation logic still not implemented — everything under
`src/tessera/` beyond ingestion/ and `__init__.py` is a docstring stub
naming which task implements it.

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
- [x] **Task 2 — Ingestion and chunking** (PR #5, merged). `loader.py`
      parses front matter + body via `python-frontmatter`, validates the
      5-key schema at load time. `chunker.py` splits on markdown heading
      boundaries (hand-rolled heading tree, not fixed-size windows), tags
      every chunk with its full heading path; oversized sections subdivide
      at paragraph boundaries only, with fenced code blocks and tables
      always kept atomic even past the word budget. Verified against every
      structural edge case Task 1 added: 46/46 checkboxes intact, fence
      markers always balanced, table isolated cleanly from surrounding
      prose, 3-level heading nesting preserved. 336 chunks from 52 docs, 25
      tests passing (synthetic fixtures + real-corpus integration). Also
      added CLAUDE.md constraint #6 (query path stays transport-agnostic —
      pure functions, no infra coupling) — `chunker.py` already holds it;
      it's now an explicit bar for Tasks 4-6 to be checked against as
      they're built.

## Next task to pick up

**Task 3 — Embedding and vector store behind interfaces** (build plan §5,
Task 3). Not started — waiting on go-ahead.

Task 3 covers:
1. Define `Embedder` interface (`src/tessera/embedding/base.py`), then the
   `sentence-transformers` local implementation (`embedding/local.py`).
2. Define `VectorStore` interface (`src/tessera/store/base.py`), then the
   Chroma implementation (`store/chroma.py`) — local, persistent, at
   `data/vectorstore/` (gitignored). This is the decided choice, not
   pgvector/Pinecone/Milvus — see `CLAUDE.md` tech table and
   `docs/Tessera_Solution_Design.md` §4; OpenSearch Serverless is the
   Phase 4 target behind the same interface.
3. Index the 336 chunks from Task 2 into the store.

Note the heavy ML deps (`sentence-transformers`, `chromadb`) aren't
installed in `.venv` yet — Task 2 only installed the lightweight subset
(`python-frontmatter`, `pyyaml`, `pytest`) to avoid a slow `uv sync`. Task 3
will need the full `uv sync` (or targeted `uv pip install`), which is slow
(timed out once at 2 minutes) — plan for that, possibly run in background.

**Acceptance check for Task 3:** corpus indexed; a manual similarity query
returns plausible chunks; swapping implementations requires no changes
outside the `embedding/` and `store/` modules.

## Task sequence (build plan §5, for reference)

1. ~~Repo scaffold and synthetic corpus~~ — done (PR #2)
2. ~~Ingestion and chunking~~ — done (PR #5)
3. Embedding and vector store behind interfaces ← **next**
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
- Forward context for whoever picks up Phase 4: `docs/adr/` now records a
  hybrid Go/Python production architecture decision (Go edge/routing +
  session state, Python RAG core unchanged from Phase 1, serverless →
  Kubernetes evolution, GitHub Actions + Terraform for CI/CD). Documentation
  only — doesn't change Task 2-8 scope or the current task sequence below.
