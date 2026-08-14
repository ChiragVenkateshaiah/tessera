# CLAUDE.md — Tessera working conventions

## Project context

Tessera is an internal knowledge assistant for Meridian Advisory, a 600-person
consulting firm whose consultants lose significant time searching for prior
work, frameworks, and internal expertise. Discovery established that this is
not one retrieval problem but four distinct query archetypes, and that client
confidentiality — specifically documents that are "anonymized" but still
identifiable to an industry insider — is the defining risk. The pilot
deliberately scopes to low-sensitivity content (methodology wiki + published
thought leadership) to prove the system works before going near client
material.

Full reasoning behind these constraints lives in `docs/`:
- `docs/Tessera_Discovery_Findings.md` — problem context, the four query
  archetypes, the confidentiality model.
- `docs/Tessera_Solution_Design.md` — full architecture, including the
  AWS/production target this phase is deliberately not building yet.
- `docs/Tessera_Phase1_Build_Plan.md` — the authoritative instruction set for
  this phase. Re-read it before making structural decisions.

## Phase 1 objective and boundaries

**Objective:** a working, locally-run ingestion and retrieval core over the
pilot corpus, plus an empty-but-functional evaluation harness — provable on
placeholder queries, ready to be populated with the real query log when it
arrives.

**In scope:** synthetic pilot corpus, ingestion + section-aware chunking,
embedding + local vector store, retrieval for archetype A (lookup) and C
(synthesis), grounded generation with mandatory citations, eval harness
scaffold (runnable, metrics implemented, cases empty), CLI for smoke-testing.

**Explicitly NOT in Phase 1 — do not build these:**
- Archetype B (expertise-finding) — blocked on unknown HR data structure.
  Return "not yet supported."
- Archetype D (comparative) — out of pilot scope by design (confidentiality).
  Implement only as a refusal guardrail — never actually attempt it.
- Any AWS deployment, Terraform, CI/CD, or monitoring — Phases 4–5.
- PowerPoint/deck ingestion — not in the pilot corpus.
- Access-control enforcement — pilot corpus is low-sensitivity by
  construction; this sidesteps the confidentiality problem, it doesn't solve
  it.
- A web UI — CLI is sufficient.

If a task seems to require something on this list, stop and flag it rather
than building it.

## Design constraints that shape the code

These are reasons, not preferences:

1. **Swappable ports.** Phase 4 moves this to AWS (Bedrock, OpenSearch, S3).
   Every external dependency — embedding model, vector store, LLM client,
   document source — sits behind a thin interface so the swap is a config
   change, not a rewrite. The single most important structural decision in
   Phase 1.
2. **Grounded generation only.** Every claim in an answer must trace to a
   retrieved chunk. The system says "we don't have anything on that" rather
   than fabricating.
3. **Archetypes are first-class.** Retrieval behaviour differs by archetype
   (lookup: narrow, precise; synthesis: broad, multi-source). Do not collapse
   them into one pipeline.
4. **Evals are infrastructure, not an afterthought.** The harness is built now
   even though real test cases arrive later, because it becomes the CI gate
   in Phase 5.
5. **Local-first.** No cloud dependencies in Phase 1 except the LLM API call
   itself.

## Technology decisions

| Concern | Phase 1 choice | Phase 4 target | Notes |
|---|---|---|---|
| Language | Python 3.11+ | same | |
| Env / deps | `uv` (or venv + pip) | same | Lockfile committed |
| Embeddings | `sentence-transformers` local model | Bedrock Titan / Cohere | Behind `Embedder` interface |
| Vector store | Chroma (local, persistent) | OpenSearch Serverless | Behind `VectorStore` interface |
| LLM | DeepSeek API | Claude via Bedrock | Behind `LLMClient` interface; free tier for Phase 1, swapped for Claude/Bedrock in Phase 4 |
| Config | `pydantic-settings` + `.env` | same + Parameter Store | No hardcoded values |
| Testing | `pytest` | same | |
| CLI | `typer` | n/a | |

`.env` is gitignored. An `.env.example` is committed.

## Working conventions

- Small, reviewable commits with clear messages; commit at each task
  boundary.
- Type hints throughout; docstrings on public interfaces.
- No secrets in the repo. No hardcoded paths.
- Tests for chunking, routing, and metrics logic — the deterministic parts.
  Do not over-test LLM outputs; that is what the eval harness is for.
- When a decision is ambiguous, prefer the option that keeps the AWS
  migration cheap.
- Work task by task per `docs/Tessera_Phase1_Build_Plan.md` §5. Stop after
  each task and report against its acceptance check before continuing.
- See `checkpoint.md` at repo root for where the build currently stands and
  what the next task is.

## Git workflow

`main` is branch-protected (no direct pushes, no force-push, no deletion).
Every change ships through a PR:

1. Branch off `main`: `git checkout -b <type>/<short-description>` (e.g.
   `feat/task1-corpus`, `fix/chunker-heading-split`).
2. Commit on the branch as work progresses.
3. Push the branch and open a PR with `gh pr create`. One PR per push to
   `main` — including scaffolding and doc-only changes.
4. Merge via **merge commit** (`gh pr merge --merge`), not squash or rebase —
   keeps per-branch commit history visible on `main`.
5. Delete the branch after merge.

**Tags:** cut a git tag at each Phase boundary, once that phase's exit
criteria are met (build plan §7 for Phase 1) — e.g. `v0.1.0` when Phase 1
exits. Not cut per-task; tasks are checkpoints, phases are releases.

**CI/CD:** intentionally not set up in Phase 1 (build plan §1 do-not-build
list). It earns its place at Phase 5, gated by the eval harness built in
Task 7 — no change ships if retrieval/answer quality regresses (Solution
Design §5). Until then, quality gating is manual: run `pytest` and the eval
harness locally before opening a PR.
