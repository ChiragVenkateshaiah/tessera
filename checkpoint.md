# Tessera — Checkpoint

Last updated: 2026-08-19

## Status

Task 7 complete, not yet merged (this session). The eval harness runs
end-to-end (routing accuracy, recall@k/precision@k/MRR, LLM-judge
groundedness/relevance, latency per archetype) and survives a bad
LLM response mid-sweep without losing already-completed cases. The full
8-case live sweep is deliberately deferred to a fresh quota day — see
Notes and the Phase 1 exit carry-forward below.

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

## Next task to pick up

**Task 8 — CLI and README** (build plan §5, Task 8). Not started —
waiting on go-ahead.

`tessera ingest` and `tessera query "..."` at minimum; `tessera eval`
runs the harness (`evals.harness.run_harness()` directly — per
genai-architect's Task 7 review, the CLI command should call it
directly rather than shelling out to `python -m evals.harness`, so
there's one composition root, not two). README covers setup, the phase
boundary (what's built vs. designed), and how to run each command.

**Acceptance check for Task 8:** a fresh clone can be set up and queried
following the README alone.

Task 8 is also subject to the `genai-architect`/`quality-engineer`
review process (see `## Architecture & QA notes` below).

**Before Task 8 is marked done**, per genai-architect's carry-forward:
run the full 8-case live sweep (`python -m evals.harness`) on a day with
enough Gemini quota headroom (~16 calls needed) and paste the emitted
report into this file — that's what actually discharges Phase 1 exit
criterion #3 ("the eval harness runs and reports all metric categories
on placeholder cases"), which Task 7 itself only proved structurally
(deterministic tests + a 2-case live spot-check), not with the literal
full-sweep artifact.

## Task sequence (build plan §5, for reference)

1. ~~Repo scaffold and synthetic corpus~~ — done (PR #2)
2. ~~Ingestion and chunking~~ — done (PR #5)
3. ~~Embedding and vector store behind interfaces~~ — done (PR #7)
4. ~~Archetype router~~ — done (PR #10)
5. ~~Archetype-aware retrieval~~ — done (PR #13)
6. ~~Grounded generation with citations~~ — done (PR #16)
7. ~~Evaluation harness~~ — done (this session's PR)
8. CLI and README ← **next**

Work one task at a time. Stop after each and report against its acceptance
check before continuing to the next.

## Notes / open flags

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
  sufficient evidence for Task 7 itself. **The full sweep still needs to
  happen once**, on a day with quota headroom, before Task 8 is marked
  done — see the Task 8 entry under Next task to pick up. Options if
  quota keeps being the binding constraint: a paid Gemini tier,
  spreading the sweep across multiple days, or a different default model
  with a higher free quota.
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
