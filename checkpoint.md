# Tessera — Checkpoint

Last updated: 2026-08-14

## Status

Repo scaffolding step. `CLAUDE.md`, this checkpoint, and daily-ritual
slash commands (`/start-day`, `/end-day`) have been created. Git tracking
and GitHub remote set up. No application code written yet.

## Done

- [x] Read `docs/Tessera_Phase1_Build_Plan.md`, `docs/Tessera_Discovery_Findings.md`,
      `docs/Tessera_Solution_Design.md` in full.
- [x] Created `CLAUDE.md` at repo root (plan §6).
- [x] Created this checkpoint file.
- [x] Created `/start-day` and `/end-day` custom commands.
- [x] Initialized git tracking and connected GitHub remote.

## Next task to pick up

**Task 1 — Repo scaffold and synthetic corpus** (build plan §5, Task 1).
Not started — waiting on go-ahead.

Task 1 covers:
1. Create the full repository structure per build plan §4 (`src/tessera/`,
   `evals/`, `data/`, `tests/`, `pyproject.toml`, `.env.example`, `.gitignore`,
   `README.md`).
2. Generate a synthetic pilot corpus:
   - ~25–30 methodology wiki pages (markdown, headings) covering frameworks
     like market entry analysis, cost transformation, pricing strategy,
     operating model design, due diligence. Internal how-to style, no client
     data.
   - ~10–12 thought-leadership pieces (markdown, longer form) spanning
     financial services, retail, pharma, industrials.
   - Front-matter metadata per file: `title`, `doc_type`
     (`methodology`|`thought_leadership`), `industry`, `topics`, `date`.
   - Deliberately vary length and overlap topics so retrieval has to
     discriminate — toy documents make retrieval quality meaningless.

**Acceptance check for Task 1:** corpus exists, files parse, metadata
consistent.

## Task sequence (build plan §5, for reference)

1. Repo scaffold and synthetic corpus ← **next**
2. Ingestion and chunking
3. Embedding and vector store behind interfaces
4. Archetype router
5. Archetype-aware retrieval
6. Grounded generation with citations
7. Evaluation harness
8. CLI and README

Work one task at a time. Stop after each and report against its acceptance
check before continuing to the next.

## Notes / open flags

- None yet. Flag here anything that seems to require an item from the
  "explicitly NOT in Phase 1" list in `CLAUDE.md`, rather than building it.
