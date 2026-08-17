# Tessera — Checkpoint

Last updated: 2026-08-17

## Status

Task 4 complete (PR pending merge). Archetype routing works end-to-end
against a live LLM. `retrieval/retriever.py`, `generation/prompts.py`'s
grounded-answer half, and `pipeline.py` still not implemented.

## Done

- [x] Read `docs/Tessera_Phase1_Build_Plan.md`, `docs/Tessera_Discovery_Findings.md`,
      `docs/Tessera_Solution_Design.md` in full.
- [x] Created `CLAUDE.md` at repo root (plan §6).
- [x] Created this checkpoint file.
- [x] Created `/start-day` and `/end-day` custom commands.
- [x] Initialized git tracking, connected GitHub remote, adopted PR-based
      workflow (branch → PR → merge commit, branch-protected `main`, tags at
      phase boundaries, CI/CD deferred to Phase 5).
- [x] Switched Phase 1 LLM decision from Claude API to DeepSeek API, then
      from DeepSeek to **Gemini API** (free via existing Gemini Pro
      subscription — DeepSeek required a funded balance). Claude via
      Bedrock remains the Phase 4 target throughout; `generation/gemini.py`
      is the current LLMClient implementation.
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
- [x] **Task 3 — Embedding and vector store behind interfaces** (PR #7,
      merged). `Embedder`/`VectorStore` interfaces defined first;
      `LocalEmbedder` (sentence-transformers all-MiniLM-L6-v2, 384-dim,
      CPU) and `ChromaVectorStore` (local persistent, explicit cosine
      space — Chroma defaults to L2) as the concrete implementations.
      Along the way, pinned `torch` to the CPU-only wheel index in
      `pyproject.toml`/`uv.lock` — the default PyPI build on Linux pulls
      ~2GB of unneeded NVIDIA CUDA packages transitively; confirmed zero
      `nvidia-*` in the lockfile after the fix. 47 tests total (37 new):
      all 336 chunks indexed, plausible top results on real queries,
      off-corpus queries score well below on-corpus ones, and — direct
      proof of the interface-swap acceptance criterion — the exact same
      indexing/query function runs unchanged against both real
      implementations and a pair of fakes defined only in the test file.
- [x] **Task 4 — Archetype router** (PR pending). `LLMClient` interface
      (`generation/base.py`) + `GeminiClient` (`generation/gemini.py`)
      built first (pulled forward from Task 6, per the option-a decision
      above), then `retrieval/router.py`: LLM-based classification into
      A/B/C/D via a prompt in `generation/prompts.py`, JSON-parsed with
      markdown-fence tolerance, returning an inspectable `RoutingDecision`
      dataclass and logging every decision. `terminal_response_for()`
      gives B the "not yet supported" message and D the confidentiality
      refusal, per the build plan. All 8 Discovery Findings §7 placeholder
      queries confirmed routing correctly against the live Gemini API
      (each individually verified during this session — see rate-limit
      note below for why not all 8 landed in one single clean run).
      10 unit tests (fake LLM client, no network) + 8 live tests (opt-in,
      see below).

      **Important operational finding: Gemini's free tier caps
      `gemini-3.6-flash` at 20 requests/*day*** (not just 5/minute) — hit
      both limits repeatedly while testing this session. Fixed the live
      test to require an explicit `RUN_LIVE_LLM_TESTS=1` opt-in, not just
      "a key is present," so a routine `pytest tests/` run never silently
      spends that budget. **This is a real constraint for Task 7's eval
      harness** — 20 requests/day cannot run a meaningful eval sweep in
      one sitting. Options to revisit when Task 7 starts: a paid Gemini
      tier, spreading eval runs across days, or a different default model
      with a higher free quota. Flagging now so it isn't a surprise later.

## Next task to pick up

**Task 5 — Archetype-aware retrieval** (build plan §5, Task 5). Not
started — waiting on go-ahead.

Task 5 covers `retrieval/retriever.py`: retrieval strategy varies by
archetype — A (lookup) uses narrow k with metadata filtering, C
(synthesis) uses broader k across sources. Returns chunks with source
attribution. B and D never reach this — `router.terminal_response_for()`
(Task 4) already returns their fixed response before retrieval would run.

**Acceptance check for Task 5:** same query under A vs C settings returns
visibly different retrieval breadth.

## Task sequence (build plan §5, for reference)

1. ~~Repo scaffold and synthetic corpus~~ — done (PR #2)
2. ~~Ingestion and chunking~~ — done (PR #5)
3. ~~Embedding and vector store behind interfaces~~ — done (PR #7)
4. ~~Archetype router~~ — done (PR pending merge)
5. Archetype-aware retrieval ← **next**
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
