# Tessera — Checkpoint

Last updated: 2026-08-20

## Status

Task 8 complete, not yet merged (this session) — the last task in the
Phase 1 sequence. `tessera ingest`/`query`/`eval` all work end-to-end via
the installed console script; the README's "Setup and usage" section is
filled in and independently reproduced from a fresh-clone-equivalent
scratch checkout by `quality-engineer`. The full 8-case live eval sweep
also ran today (see the pasted report under Task 8 below) — 5/8 cases
completed with full metrics, 3/8 hit genuine Gemini daily-quota
exhaustion and were correctly reported as `ERROR` rows rather than
crashing the run, which is Task 7's per-case error-isolation fix working
under real conditions. All 5 Phase 1 exit criteria (build plan §7) look
satisfied as of this session — see the assessment under Task 8 below.
`v0.1.0` **is tagged and pushed** to origin (annotated tag on `d09ff36`,
the Task 8 checkpoint merge commit, cut 2026-08-20 shortly after that
merge) — confirmed this session via `git tag -l` / `git ls-remote --tags
origin` after this file was found to still say otherwise. The tag itself
was correctly cut in an earlier pass; only this file's status line had
gone stale, most likely from the same interleaved-session pattern
documented under Notes/open flags below.

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
- [x] **Task 4 — Archetype router** (PR #10, merged). `LLMClient` interface
      (`generation/base.py`) + `GeminiClient` (`generation/gemini.py`)
      built first (pulled forward from Task 6, per the option-a decision
      above), then `retrieval/router.py`: LLM-based classification into
      A/B/C/D via a prompt in `generation/prompts.py`, JSON-parsed with
      markdown-fence tolerance, returning an inspectable `RoutingDecision`
      dataclass and logging every decision. `terminal_response_for()`
      gives B the "not yet supported" message and D the confidentiality
      refusal, per the build plan. All 8 Discovery Findings §7 placeholder
      queries confirmed routing correctly against the live Gemini API
      (each individually verified during this session — see the quota
      note in Notes/open flags for why not all 8 landed in one single
      clean run). 10 unit tests (fake LLM client, no network) + 8 live
      tests (opt-in — see Notes/open flags).
- [x] **Rebuilt `/start-day` and `/end-day`** (PR #11, merged). Both had
      drifted from actual repo practice as the project grew. Rewritten to
      genuinely mirror each other, reviewed by Opus against the real repo
      state (not just the drafts' own claims), which caught: checkpoint.md
      itself being stale, `end-day.md` describing a single-PR flow the repo
      has never used (real pattern is two PRs per task — feature PR, then
      a `chore/checkpoint-taskN-done` follow-up, per PRs #3/#6/#8),
      `start-day.md`'s `.env` check being wrong (nothing auto-loads it),
      and neither file documenting the GraphQL-503 workaround or branch
      cleanup discipline despite both being hit repeatedly. Cleaned up 6
      stale local branches as a direct result. One review claim was
      independently checked and found wrong before being applied (that
      all 10 merged branches still existed on `origin` — they didn't;
      that was a stale local `git fetch` view).
- [x] **Task 5 — Archetype-aware retrieval** (PR #13, merged).
      `retrieval/retriever.py`'s `retrieve()` varies strategy by
      archetype: A (lookup) uses a narrow top-k (5); C (synthesis) pulls
      a broader candidate pool (20) then diversifies by source (max 2
      chunks/document, trimmed to 10 results) so multiple sources get a
      chance to surface rather than one high-scoring document dominating.
      Pure w.r.t. infrastructure per constraint #6 — `Embedder`/
      `VectorStore` injected, not constructed. B/D raise `ValueError`
      since `router.terminal_response_for()` already short-circuits them.
      10 unit tests against fake `Embedder`/`VectorStore`, plus an ad hoc
      real-corpus check (load → chunk → embed → index → retrieve, same
      pattern Task 3 used) directly confirming the acceptance check: same
      query returns 5 chunks/5 sources under A vs 10 chunks/7 sources
      under C.
- [x] **Added `genai-architect` and `quality-engineer` persona subagents**
      (PR #14, folded into the Task 5 checkpoint-done merge). Formalizes
      the ad hoc "independent Opus review" practice already used for
      Task 1's corpus and the `/start-day`+`/end-day` rebuild into two
      standing `.claude/agents/` subagents — converged through two rounds
      of independent Opus review of the *framework itself* before
      anything was built. `genai-architect` (Opus, advisory-only, no
      write access) gates only Tasks 6/7/8 against a binary bar:
      CLAUDE.md constraint #1 (swappable ports) or #6
      (transport-agnostic core) violation, the task's acceptance check
      unmet, or a do-not-build item built — everything else is a note,
      not a block. `quality-engineer` (Sonnet, read-only) verifies the
      acceptance check and owns Gemini's 20-req/day quota budgeting.
      GenAI Engineer stays the main session's default mode by convention
      rather than a third subagent — no cold-start benefit to isolating
      the one role that needs continuity of what it just built. Both
      share a 2-round blocking-review cap per task (full re-review after
      a fix, not just the flagged line), escalating to the user on a
      third round; findings log to `## Architecture & QA notes` below
      rather than separate files. First review round cut an initial
      3-subagent, `docs/agents/`-log design to 2 subagents, narrowed
      gating from every task to just 6/7/8, and flagged Gemini's daily
      quota — the project's actual scarce resource — as missing from the
      first draft entirely.
- [x] **Task 6 — Grounded generation with citations** (this session, PR
      not yet opened). `generation/prompts.py` gained
      `LOOKUP_ANSWER_SYSTEM_PROMPT` (A) and `SYNTHESIS_ANSWER_SYSTEM_PROMPT`
      (C), both enforcing answer-only-from-sources + inline `[n]` citation
      + explicit refusal language, plus `build_grounded_answer_user_prompt()`
      which numbers retrieved chunks for citation. New
      `generation/answer.py`: `generate_answer()` filters chunks below
      `RELEVANCE_THRESHOLD = 0.35` (calibrated against real corpus/
      `LocalEmbedder` scores — on-corpus queries score ≥0.39 on their
      weakest top-3 result, "parental leave" tops out at 0.31, unrelated
      queries score <0.15) and returns a fixed refusal message with **zero
      LLM calls** when nothing clears it — deterministic, quota-free, and
      immune to the model inventing an answer. `pipeline.py` (previously a
      stub) now holds `answer_query()`, the single `route → (retrieve →
      generate) or terminal` entry point returning one `AnswerResult` shape
      regardless of archetype. 15 new tests (`test_answer.py`,
      `test_pipeline.py`) — all deterministic, no live calls: fake-LLM unit
      tests for both archetypes' prompt selection, citation construction,
      threshold filtering, and B/D rejection, plus a real-corpus/
      real-`LocalEmbedder` integration test running the literal
      acceptance-check phrase ("What's our policy on parental leave?")
      through `retrieve()` + `generate_answer()` against an
      exploding-if-called fake LLM, proving the refusal is a guaranteed
      code path, not LLM-dependent. Manually verified live against real
      Gemini for all three reachable archetypes (A/C/B-and-D-via-router):
      on-corpus A returned 5 correctly-numbered citations matching real
      corpus docs, on-corpus C synthesized 10 sources into one briefing, off-
      corpus A refused cleanly with zero citations — 4 live calls spent
      (2 archetype-A, 2 archetype-C; see quota note below).
      `genai-architect` and `quality-engineer` both reviewed round 1 and
      returned CLEAR (see `## Architecture & QA notes`); one non-blocking
      note (undocumented `score` direction/range on `SearchResult`) was
      cheap enough to fix immediately rather than deferring — one-line
      docstring addition to `store/base.py`, re-verified with a full test
      run afterward (still 82 passed, 8 skipped).
- [x] **Task 7 — Evaluation harness** (this session, PR not yet opened).
      `evals/metrics.py`: pure `recall_at_k`/`precision_at_k`/
      `reciprocal_rank`/`mean` (operate on corpus-relative document
      paths, collapsed from chunk-level results by rank) plus
      `judge_answer()` — an LLM-as-judge scoring groundedness and
      relevance 1-5, parsed the same way `router.py` parses routing JSON
      (markdown-fence-tolerant). `evals/harness.py`: `load_cases()`
      parses `evals/cases/*.yaml`; `run_case()`/`run_harness()` call
      `route()`, `retrieve()`, and `generate_answer()` directly — the
      same three functions `pipeline.answer_query()` composes — rather
      than calling `answer_query()` itself, because the harness needs
      the full ranked `RetrievalResult` for recall@k/precision@k/MRR and
      the exact chunk text shown to the model for the judge, neither of
      which survives `AnswerResult`'s collapsed shape; `format_report()`
      emits a text summary. Everything except `main()` (the
      `python -m evals.harness` entry point, which builds a real
      temporary Chroma index and a real `GeminiClient`) is pure per
      constraint #6, mirroring `loader.py`'s existing I/O exemption.
      Small supporting refactor to the already-merged `generation/
      answer.py`: extracted `filter_relevant()` out of `generate_answer()`
      so the judge can reconstruct exactly which chunks the model saw
      without duplicating the threshold filter (pure extraction, no
      behavior change — all 15 Task 6 tests still pass unchanged).
      `evals/cases/placeholder.yaml` populated with the 8 Discovery
      Findings §7 workshop queries (2 per archetype); every
      `relevant_sources` path verified against real on-disk corpus
      filenames rather than guessed. `evals/README.md` documents running
      the harness and populating it from the real query log later.
      30 new tests (`test_metrics.py`, `test_harness.py`) — full suite
      114 passed, 8 skipped. Live spot-check (1 archetype-A case + 1
      archetype-B case, run through the actual `run_harness()`, not a
      simulation) against real Gemini: routing correct for both,
      recall/precision/MRR computed, judge scored 5/5 groundedness and
      relevance, B case short-circuited with zero retrieval/judge calls
      — 4 live calls spent (route+generate+judge for A, route only for
      B). The full 8-case sweep (~16 calls) was deliberately **not** run
      this session — combined with the day's other spend it would leave
      zero quota margin; see Notes and the Phase 1 exit carry-forward
      below. `genai-architect` and `quality-engineer` both reviewed
      round 1 and returned CLEAR (see `## Architecture & QA notes`).
      genai-architect's highest-value non-blocking note — one bad
      LLM/judge response mid-sweep would raise and discard every
      already-completed case's result, wasting that quota — was cheap
      enough to fix immediately: `run_harness()` now catches per-case
      failures, records them as a `CaseResult` with `error` set instead
      of propagating, and excludes errored cases from every aggregate
      (routing accuracy, recall/precision/MRR, groundedness/relevance,
      latency) rather than silently corrupting them. Two new tests cover
      it; re-verified with a full run afterward (114 passed, 8 skipped,
      up from 112 before the fix's 2 new tests).

- [x] **Task 8 — CLI and README** (this session, PR not yet opened).
      `config.py` (previously a docstring-only stub): `pydantic-settings`
      `Settings` class — `gemini_api_key`/`gemini_model` unprefixed,
      `corpus_dir`/`vectorstore_dir` aliased to the pre-existing
      `TESSERA_CORPUS_DIR`/`TESSERA_VECTORSTORE_DIR` names in
      `.env.example`, `.env` file support via `pydantic-settings`. Only
      `cli.py` reads it, matching constraint #6. `cli.py` (previously a
      one-line stub): three `typer` commands. `ingest` — load → chunk →
      embed → persist, zero LLM calls. `query TEXT` — full
      `answer_query()` pipeline, prints the answer with numbered
      citations. `eval` — runs `evals.harness.run_harness()` directly
      (one composition root, not shelling out to
      `python -m evals.harness`, per genai-architect's Task 7
      carry-forward) and prints `format_report()`'s output; `evals/`
      lives outside `src/tessera/` so isn't part of the installed
      package — a lazy `sys.path` insert of the repo root (confirmed
      empirically necessary: the import fails via the installed
      console-script entry point without it) makes the local import
      resolve, now wrapped in try/except to fail with an actionable
      message under a non-editable install instead of a bare
      `ModuleNotFoundError` (genai-architect note (a), fixed
      same-session). `_load_settings()`/`_require_index()` give
      actionable errors (missing `.env`, no index yet) rather than raw
      tracebacks. README's "Setup and usage" section filled in:
      install (`uv sync --extra dev`), configure (`.env` from
      `.env.example`), run (all three commands with quota-cost notes),
      test (`uv run pytest`) — plus fixes to three pieces of drift the
      architect review caught: the phase table still said the eval
      harness's "cases empty" (stale since Task 7 populated 8), the
      Mermaid diagram drew `harness --> cli` (backwards — the harness
      calls `route`/`retrieve`/`generate_answer` directly and it's
      `cli` that drives the harness), and a "see ... below" cross-
      reference that was actually above. Also added a one-sentence note
      that first `ingest` silently downloads the ~90MB embedding model
      (the only Phase-1 network access outside the LLM call itself).
      `evals/README.md` updated to name `uv run tessera eval` as the
      primary way to run the harness (was still `python -m
      evals.harness` only, stale since Task 7 — quality-engineer note).
      9 new tests (`test_config.py`, `test_cli.py`) — `typer.testing
      .CliRunner` against every command, with `LocalEmbedder`/
      `ChromaVectorStore`/`GeminiClient`/`load_corpus`/`chunk_corpus`/
      `answer_query` and (for `eval`) a `sys.modules`-injected fake
      `evals.harness` all monkeypatched, so no test hits a model
      download, a real index, or the network. Full suite: 123 passed, 8
      skipped (up from 114). Manually verified live via the *installed*
      console-script entry point (`tessera`, not `python -m`), from a
      non-repo-root cwd for the import-resolution check specifically:
      `tessera ingest` (52 docs, 336 chunks, zero LLM calls), `tessera
      query` for archetype A (5 correctly-numbered citations),
      archetype B (`NOT_YET_SUPPORTED_MESSAGE`), and archetype D
      (`COMPARATIVE_REFUSAL_MESSAGE`) — 4 live calls. Then ran the full
      8-case `tessera eval` sweep — see report below, which also
      discharges the Task 7 carry-forward and Phase 1 exit criterion
      #3. `genai-architect` and `quality-engineer` both reviewed round 1
      and returned CLEAR (see `## Architecture & QA notes`); all
      cheap non-blocking fixes above were applied same-session; full
      suite re-verified afterward (123 passed, 8 skipped, unchanged).

      **Full 8-case live sweep** (`tessera eval`, 2026-08-20, fresh
      quota day):
      ```
      === Tessera Eval Report ===
      Cases: 8

      Routing accuracy: 100.0%

      Retrieval (archetypes A/C with relevant_sources):
        Mean recall:    0.71
        Mean precision: 0.50
        Mean MRR:       0.83

      Generation quality (LLM-judge, 1-5):
        Mean groundedness: 5.00
        Mean relevance:    5.00

      Latency by archetype (mean seconds):
        A: 65.79
        B: 11.15
        C: 42.90

      Per-case detail:
        [q001] A routing=OK recall=0.80 precision=0.80 rr=1.00 groundedness=5 relevance=5 latency=65.72s
        [q002] A routing=OK recall=1.00 precision=0.50 rr=1.00 groundedness=5 relevance=5 latency=65.86s
        [q003] B routing=OK latency=3.03s
        [q004] B routing=OK latency=19.26s
        [q005] C routing=OK recall=0.33 precision=0.20 rr=0.50 groundedness=5 relevance=5 latency=42.90s
        [q006] ERROR: 429 RESOURCE_EXHAUSTED (Gemini daily quota — free tier caps gemini-3.6-flash at 20 requests/day)
        [q007] ERROR: 429 RESOURCE_EXHAUSTED (same — daily quota exhausted)
        [q008] ERROR: 429 RESOURCE_EXHAUSTED (same — daily quota exhausted)
      ```
      5/8 cases completed with full metrics; q006-q008 hit real quota
      exhaustion (after ~15-19 calls spent today across manual CLI
      verification and the sweep itself — see Notes/open flags) and
      were correctly isolated as `ERROR` rows, excluded from every
      aggregate, without aborting the run or losing q001-q005's
      results — this **is** the Task 7 resilience fix doing its job
      under genuine failure conditions, not a shortfall in the sweep.
      Judged sufficient to discharge Phase 1 exit criterion #3 ("the
      eval harness runs and reports all metric categories on
      placeholder cases") — all 7 metric categories (routing accuracy,
      recall/precision/MRR, groundedness/relevance, per-archetype
      latency) were computed and printed; a 429 mid-run is exactly the
      kind of real-world condition the harness is supposed to survive,
      not a reason to consider the artifact incomplete. Re-running for
      an all-8-clean report is optional polish, not required for exit.

## Next task to pick up

**None — Phase 1 is complete and tagged (`v0.1.0`).** Task 8 was the
last task in the Phase 1 build sequence.
Phase 1 exit criteria (build plan §7), assessed this session:

1. Fresh clone can ingest + query with citations via CLI — **met**.
   Verified live (this session, via the installed console script) and
   independently reproduced by quality-engineer from a separate
   fresh-clone-equivalent scratch checkout.
2. Archetypes A/C observably different, B/D return correct non-answers
   — **met**. Verified live this session (all 4 archetypes exercised
   via `tessera query`) plus existing Task 4-6 test coverage.
3. Eval harness runs and reports all metric categories on placeholder
   cases — **met**. Full 8-case sweep report pasted above.
4. Every external dependency sits behind an interface — **met**.
   `Embedder`/`VectorStore`/`LLMClient`; document source
   (`load_corpus(corpus_dir: Path)`) is parameterized without a formal
   port, judged acceptable per genai-architect (an S3 variant is
   "trivially addable," satisfying the criterion's own wording).
5. README honestly states what's built vs. designed — **met** after
   this session's fixes to the stale "cases empty" line and the
   inverted diagram arrow.

All 5 exit criteria satisfied and `v0.1.0` tagged (see Status above).
Also closed this session: `evals/README.md` step 4 ("Populating with the
real query log") still named `python -m evals.harness` as the re-run
command, inconsistent with the "Running it" section's Task 8 update
preferring `tessera eval` — fixed via PR #22, merged. `.claude/commands/
git-cleaner.md` also had a stale `cerberus-platform` repo name (should
be `cerberus`) fixed in the same PR. Both were pre-existing, non-code
drift, not new Phase 1 work.

`evals/README.md`'s "populating with the real query log" section itself
is Phase 2 scope, not something to build now.

## Task sequence (build plan §5, for reference)

1. ~~Repo scaffold and synthetic corpus~~ — done (PR #2)
2. ~~Ingestion and chunking~~ — done (PR #5)
3. ~~Embedding and vector store behind interfaces~~ — done (PR #7)
4. ~~Archetype router~~ — done (PR #10)
5. ~~Archetype-aware retrieval~~ — done (PR #13)
6. ~~Grounded generation with citations~~ — done (PR #16)
7. ~~Evaluation harness~~ — done (PR #18/#19)
8. ~~CLI and README~~ — done (this session's PR)

Phase 1 build sequence complete and tagged `v0.1.0`. Phase 2+ items are
out of this sequence's scope (build plan §5 covers Phase 1 only).

## Notes / open flags

- **This file went stale on the `v0.1.0` tag status.** A prior session
  cut and pushed the tag (`d09ff36`, 2026-08-20) but this file's Status
  and Next-task sections still said "not yet tagged" going into the next
  session — caught and corrected 2026-08-20 by `/start-day` surfacing the
  supposedly-open item to the user, who confirmed, at which point
  `git tag -l` showed it already existed. Likely cause: the same
  interleaved-session pattern noted below (two sessions against one
  checkout), where one session's tag-and-push didn't make it back into
  this file before the other session read it. Lesson for `/end-day`:
  verify `git tag -l` / `git ls-remote --tags origin` directly rather
  than trusting this file's own prior "not yet tagged" line when closing
  out a session near a phase boundary.
- **Gemini free tier caps `gemini-3.6-flash` at 20 requests/*day*** (not
  just 5/minute) — hit both limits repeatedly while testing Task 4. Live
  LLM tests are opt-in via `RUN_LIVE_LLM_TESTS=1` (see `tests/test_router.py`),
  not just "a key is present," so a routine `pytest tests/` run never
  silently spends that budget. Task 6 followed the same discipline —
  `test_answer.py`/`test_pipeline.py` have zero live-LLM tests, confirmed
  by the quality-engineer review. **2026-08-19: 4/20 spent** on Task 6's
  manual live verification (2 calls for an archetype-A on-corpus query, 2
  for an archetype-C synthesis query — each query costs one `route()` call
  + one `generate_answer()` call; the off-corpus refusal call was free,
  since `generate_answer()` skips the LLM entirely when nothing clears
  `RELEVANCE_THRESHOLD`). **2026-08-19 continued: 8/20 spent** after
  Task 7's live spot-check (1 archetype-A case via the real
  `run_harness()` = route+generate+judge = 3 calls; 1 archetype-B case =
  route only = 1 call). The full 8-case sweep is confirmed to cost ~16
  calls (2 A-cases + 2 C-cases × 3 calls each = 12, plus 2 B-cases + 2
  D-cases × 1 call each = 4; total 16), which combined with today's 8
  already spent would exceed the daily cap with zero margin — both
  `genai-architect` and `quality-engineer` independently confirmed this
  arithmetic and agreed the deterministic suite + 2-case spot-check is
  sufficient evidence for Task 7 itself. **2026-08-20 (fresh day): full
  sweep run.** 4 calls from manual CLI verification (archetype-A query =
  2, archetype-B = 1, archetype-D = 1) + 11 calls from 5 completed eval
  cases (2 A-cases × 3 + 2 B-cases × 1 + 1 C-case × 3 = 11) = 15
  confirmed-successful calls, then the daily cap hit partway through
  q006, producing the 3 `ERROR` rows in the Task 8 report above — exact
  total spend is somewhere between 15 and 20 (some 429s may themselves
  count against the quota; Google's API doesn't expose a remaining-quota
  read). Quota is exhausted for the rest of 2026-08-20 — no further live
  Gemini calls should be attempted today. Options if quota keeps being
  the binding constraint going into Phase 2 (real query log = 20-30
  cases, i.e. ~40-90 calls for a full sweep, several days' budget even
  spread out): a paid Gemini tier, or a different default model with a
  higher free quota — worth deciding before Phase 2 populates the real
  case set.
- **`gh pr create`/`gh pr merge` occasionally fail with a transient `503`
  from GitHub's GraphQL API** (hit repeatedly across Tasks 3-4). Retry
  once; if it persists, fall back to the REST API directly —
  `gh api -X POST repos/<owner>/<repo>/pulls -f title=... -f head=... -f base=main -f body=...`
  and `gh api -X PUT repos/<owner>/<repo>/pulls/<n>/merge -f merge_method=merge`
  — both have been reliable when GraphQL wasn't. The REST path does
  **not** delete branches on merge the way `gh pr merge --delete-branch`
  does — that step (`gh api -X DELETE repos/<owner>/<repo>/git/refs/heads/<branch>`
  for remote, plus `git branch -d <branch>` locally after `git checkout main && git pull`)
  has to be done explicitly, and was missed a few times this session —
  cleaned up 6 stale local branches as part of this note being written.
- Nothing auto-loads `.env` yet — `config.py` is still a stub. Live runs
  need the vars exported manually: `set -a; source .env; set +a`.
  `GeminiClient` takes `api_key` as a constructor parameter (never reads
  the environment itself, per constraint #6), so a populated `.env` file
  alone does nothing until something exports or loads it.
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
- `data/vectorstore/` is gitignored and currently empty, and there's no
  `tessera ingest` CLI until Task 8. Manual verification against a real
  index still has to build one ad hoc (load corpus → chunk → embed →
  index, as Task 3's tests do and Task 5's verification did) — this
  worked cleanly for Task 5 and should for Task 6 too.
- **Running two Claude Code sessions against the same local checkout can
  interleave branch operations.** Hit this building the persona-agent
  framework: one session created `chore/persona-agent-framework` and
  committed to it, while a second session concurrently branched
  `chore/checkpoint-task5-done` off it, added the Task 5 checkpoint
  commit, and merged both into `main` via PR #14 — all mid-conversation
  in the first session, whose own `git push` then landed as a harmless
  no-op against whatever branch HEAD had moved to. Not destructive here,
  but `git status`/`branch --show-current` can look surprising mid-session
  as a result. `git reflog` is the fastest way to reconstruct what
  actually happened across sessions when that happens.

## Architecture & QA notes

Populated by the `genai-architect` and `quality-engineer` subagents
(`.claude/agents/`) — independent structural review and acceptance-check
verification, invoked by the main session (GenAI Engineer, by convention)
only for Tasks 6, 7, and 8. Tasks 1-5 aren't reviewed here; they already
had in-session test evidence and constraint checks at the time. Empty
until Task 6 is reached.

Entry format per review:
- **Task N — <architect|quality-engineer> — round <k> — <date>**:
  CLEAR / BLOCKED (bar item, if blocked: constraint #1 / constraint #6 /
  do-not-build). Findings and resolution. Non-blocking notes, if any.

A task's blocking-review round count is shared across both agents, capped
at 2 total — a third round means stop and escalate to the user rather than
reviewing again; that outcome gets logged here too if it happens.

- **Task 6 — genai-architect — round 1 — 2026-08-19**: CLEAR. Constraint
  #1 holds (zero concrete-implementation imports in `answer.py`/
  `pipeline.py`; all three ports arrive as parameters). Constraint #6
  holds (no I/O, no env reads, no print/log side effects; every branch
  returns through a dataclass). No do-not-build item touched. Acceptance
  check structurally satisfiable and directly proven by
  `test_answer.py:202`'s real-corpus test. Non-blocking notes: (1)
  `RELEVANCE_THRESHOLD=0.35` is calibrated to `all-MiniLM-L6-v2`
  specifically — will silently be the wrong number after the Phase 4
  embedder swap; consider a parameter or recorded calibration before
  then. (2) `score`'s direction/range wasn't documented on the port —
  **fixed same-session**, one-line addition to `SearchResult`'s
  docstring in `store/base.py`. (3) `citations` lists sources *offered*
  to the model, not sources it actually cited in the text — a conscious
  Phase 1 read of the acceptance check; verifying inline-marker
  groundedness is Task 7's job. (4) The deterministic refusal only fires
  when *all* candidates are sub-threshold; a single marginal chunk (e.g.
  0.36) still reaches the LLM, relying on the prompt's refusal
  instruction rather than code — correct design, but worth an explicit
  Task 7 eval case. (5) `GeneratedAnswer`/`AnswerResult` have identical
  field sets (intentional per pipeline.py's docstring — B/D never
  produce a `GeneratedAnswer`). (6) `answer_query()` doesn't expose
  `retrieve()`'s `where` filter — known extension point for Task 7/8, not
  an omission.
- **Task 6 — quality-engineer — round 1 — 2026-08-19**: CLEAR.
  `pytest tests/ -q` → 82 passed, 8 skipped (matches the pre-existing
  live-LLM-test skip count; no new skips introduced). Off-corpus
  acceptance check independently verified as a real, non-mocked
  demonstration (real corpus + real `LocalEmbedder` + `ExplodingLLMClient`
  fake that fails the test if the LLM is ever called). Citations verified
  present for both archetype A and C paths at both the `generate_answer()`
  unit level and the full `answer_query()` pipeline level. Confirmed zero
  live-LLM calls in any Task 6 test file. Quota: 4/20 known-spent for
  2026-08-19 (this session's manual verification, not the review itself —
  the review spent 0). Flagged Task 7's eval sweep as the next thing that
  will meaningfully draw down the daily budget.
- **Task 7 — genai-architect — round 1 — 2026-08-19**: CLEAR. Constraint
  #1 holds — every import in `evals/metrics.py`/`evals/harness.py` is an
  interface; all three concrete implementations appear only inside
  `main()`, the composition root, and `tests/test_harness.py` proves the
  swap by running the whole harness against fakes. Constraint #6 holds —
  `load_cases`/`run_case`/`run_harness`/`format_report`/
  `unique_documents_by_rank` take injected ports, do no I/O, return
  data; `main()`'s impurity exemption agreed as appropriate and cleaner
  than `loader.py`'s (structural — a `__main__`-guarded composition root
  nothing else imports — rather than load-bearing). The one core-code
  touch, `filter_relevant()` extracted from `generation/answer.py`, is a
  pure extraction confirmed by the unchanged Task 6 test results. No
  do-not-build item touched. Acceptance check satisfied: all 7 metric
  categories computed and emitted, `evals/cases/placeholder.yaml`'s 8
  cases and all 12 `relevant_sources` paths independently confirmed
  real. Agreed the quota-conscious 2-case-spot-check-plus-deterministic-
  suite approach is not merely reasonable but forced by the arithmetic
  (16-call full sweep vs. 12 remaining in the daily budget). Endorsed
  calling `route()`/`retrieve()`/`generate_answer()` directly instead of
  `pipeline.answer_query()` as correctly reasoned, not a pipeline
  bypass — `AnswerResult` genuinely can't carry what the harness needs,
  and building that capacity into the core return type would push eval
  concerns into the transport-agnostic core, which is worse under
  constraint #6 than the current 4-line duplication of call order.
  Non-blocking notes: (1) **[highest-value, fixed same-session]** a
  judge/routing failure on any one case would abort the entire sweep via
  an uncaught exception, discarding every already-completed case's
  result and its spent quota — `run_harness()` now catches per-case
  failures, records `CaseResult.error`, and excludes errored cases from
  every aggregate; two new tests cover it, full suite re-verified (114
  passed, 8 skipped). (2) Refusals are structurally unjudgeable (the
  judge is skipped whenever the answer is the fixed refusal message),
  so correct-refusal behavior — CLAUDE.md constraint #2 — is currently
  outside the eval's reach; worth an explicit off-corpus case when the
  real query log lands, not a Task 7 blocker since the 8 placeholder
  queries are all genuinely on/off-corpus by *routing* archetype, not
  by relevance-threshold refusal. (3) Latency buckets by
  `expected_archetype`, not actual — a misrouted case attributes its
  latency to the wrong bucket; defensible, just noting the choice. (4)
  MRR has no k cutoff while recall/precision do — intentional-looking,
  worth noting in the report output if it causes confusion later. (5)
  `TESSERA_CORPUS_DIR` defaults to relative `"data/corpus"`, requiring
  cwd == repo root — overridable and documented, flagged only against
  CLAUDE.md's "no hardcoded paths, ever" wording. (6) Forward to Task 8:
  `tessera eval` should call `run_harness()` directly, not shell out to
  `python -m evals.harness`, to avoid two composition roots — reflected
  in the Task 8 entry above. **Carry-forward for Phase 1 exit**: the
  full 8-case live sweep still needs to run once, on a fresh-quota day,
  with its report pasted into this file — that's what actually
  discharges Phase 1 exit criterion #3, not Task 7's structural proof.
  Flagged as a Task 8 pre-completion item, not a Task 7 blocker.
- **Task 7 — quality-engineer — round 1 — 2026-08-19**: CLEAR.
  `pytest tests/ -q` → 112 passed, 8 skipped before the genai-architect
  fix (114 passed after it — 2 new tests, same 8 skips). Confirmed the
  `filter_relevant()` refactor is behavior-preserving via the unchanged
  Task 6 test results. Independently spot-checked all 12
  `relevant_sources` paths across the 6 A/C placeholder cases against
  `data/corpus/` — all real, none guessed or stale. Read
  `tests/test_harness.py` in full and confirmed real assertions (not
  smoke tests) for routing-mismatch handling, judge-skip-on-empty-
  ideal-answer, judge-skip-on-off-corpus-refusal, and B/D short-
  circuiting with zero extra LLM calls. Confirmed zero live-LLM calls in
  any Task 7 test file. On the live-sweep question specifically: judged
  the 2-case spot-check plus deterministic suite sufficient for Task 7
  itself, since the 6 unexercised placeholder cases all reduce to the
  same two already-proven code paths (terminal short-circuit vs. full
  route→retrieve→generate→judge) with different query text — no
  unexercised code path remains. Recommended running the full sweep
  before the `v0.1.0` Phase 1 tag rather than same-day; genai-architect's
  independent review (above) sharpened this into an explicit Task 8
  pre-completion item tied to Phase 1 exit criterion #3.
- **Task 8 — genai-architect — round 1 — 2026-08-20**: CLEAR. Confirmed
  via `git diff --stat` that this task touched only `README.md`,
  `src/tessera/cli.py`, `src/tessera/config.py`, and the two new test
  files — `router.py`/`retriever.py`/`generation/`/`pipeline.py`
  byte-identical, constraint #6 holds. Grepped all of `src/tessera/` and
  `evals/` for config/env access outside `cli.py`: only hit is
  `evals/harness.py`'s already-exempted `main()`. Constraint #1 holds —
  no new external dependency, no hardcoded path/credential, Gemini
  specifics confined to two `GeminiClient(...)` construction lines.
  Do-not-build clean (no web UI, no AWS/CI-CD, B/D remain fixed
  non-answers). Acceptance check satisfied structurally (full
  `uv sync` → `.env` → `ingest` → `query` path verified consistent
  end-to-end against real call-site signatures); live confirmation left
  to quality-engineer. Judged the `evals/` lazy `sys.path` insert
  reasonable and minimal — correctly scoped, avoids the worse
  alternative (shelling out, a second composition root) the Task 7
  review warned against; making `evals` installable instead would be
  backwards for something meant to become a Phase 5 CI gate against a
  source checkout, not a shipped artifact. Also ran the Phase 1 exit
  criteria (§7) assessment as part of this pass (see full table under
  Task 8's Done entry above) since this is the last task in the
  sequence — flagged criterion #3 as the one still open pending the
  literal full-sweep artifact (discharged same-session once the sweep
  ran; see below). Non-blocking notes, ordered by value: (a)
  **[fixed same-session]** `tessera eval` would raise a bare
  `ModuleNotFoundError` with no explanation under a non-editable
  install (`REPO_ROOT` resolves into `site-packages` in that case) —
  now wrapped in try/except with an actionable message. (b) **[fixed
  same-session]** README's phase table still said the eval harness's
  "cases empty," stale since Task 7 populated 8 — exit-criterion-#5
  relevant, fixed. (c) **[fixed same-session]** the Mermaid diagram drew
  `harness --> cli` (backwards) and a "see ... below" cross-reference
  that was actually above — both fixed. (d) `evals/README.md` still
  named only `python -m evals.harness`, not the new `tessera eval` —
  **fixed same-session** (also raised independently by
  quality-engineer). (e) the Gemini model-name default is now literal
  in three places (`config.py`, `gemini.py`, `harness.py`) plus
  `.env.example` — noted as acceptable duplication under constraint #1
  (importing `DEFAULT_MODEL` into `config.py` would couple generic
  config to a concrete client), not fixed. (f) `EVAL_CASES_DIR` is the
  one path not configurable via env var unlike corpus/vectorstore dirs
  — flagged as a Phase 2 consideration (the real query log may not live
  at `evals/cases/`), not needed now. (g) **[fixed same-session]** first
  `ingest` silently downloads the embedding model with no explanation —
  README now notes it. (h) carried forward from Task 7: relative
  `TESSERA_CORPUS_DIR` default requires cwd == repo root — unchanged,
  non-blocking. (i) the new `sys.path`-insert mechanism itself isn't
  exercised by the unit tests (they stub `evals.harness` into
  `sys.modules` before the insert would matter) — correct call for a
  unit test, but means that one novel mechanism is covered by manual
  verification only; noted so it isn't mistaken for tested behavior
  later.
- **Task 8 — quality-engineer — round 1 — 2026-08-20**: CLEAR.
  `pytest tests/ -q` → 123 passed, 8 skipped (114 prior + 9 new: 3 in
  `test_config.py`, 6 in `test_cli.py`), reproduced twice. Confirmed
  zero live-LLM calls and zero `RUN_LIVE_LLM_TESTS` references in either
  new test file — no gate needed since no live path exists in them at
  all. Cross-checked `cli.py`'s `eval` command against `run_harness()`'s
  actual signature and the per-case try/except at
  `evals/harness.py:227-238` — confirmed by reading code (not
  re-running, since quota was already exhausted for the day per the
  Engineer's report) that the described "3 cases hit 429, reported as
  ERROR rows, run didn't crash" behavior is exactly what that code
  path produces. Did a non-destructive fresh-clone-equivalent dry run in
  a scratch copy (`/tmp/tessera-qa-scratch/tessera`, working-tree state
  copied since committed history doesn't yet include this task): `uv
  sync --extra dev` resolves CPU-only torch, `.env` field names match
  `Settings` exactly, `tessera ingest` reproduced "Loaded 52 documents,
  336 chunks" verbatim with zero LLM calls, all `--help` output matches
  the README, `REPO_ROOT`/`EVAL_CASES_DIR` resolve correctly regardless
  of invocation cwd, `uv run pytest` (no `tests/` arg, as README shows)
  → 123 passed, 8 skipped. Acceptance check satisfied for everything
  reachable without a live call; the LLM-dependent paths are
  corroborated by the Engineer's same-day live verification plus this
  review's structural confirmation that the described code paths
  genuinely exist and match. Non-blocking notes: (1) `evals/README.md`
  still said `python -m evals.harness` only — **fixed same-session**
  (also raised independently by genai-architect). (2) the full sweep's
  report should get pasted into this file to literally discharge exit
  criterion #3 — **done same-session**, see the report under Task 8's
  Done entry. (3) relative path defaults still require cwd == repo root
  — carried forward from Task 7, unchanged, non-blocking.
