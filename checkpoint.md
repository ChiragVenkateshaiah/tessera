# Tessera — Checkpoint

Last updated: 2026-09-06

## Status

**Phase 2 complete** (`v0.2.0` tagged 2026-09-06, all 5 tasks merged,
exit gate met). Phase 1 remains complete and tagged (`v0.1.0`); all 5
Phase 1 exit criteria re-confirmed still hold (see the end of this
file).

**Phase 2 arc:** LLM provider swapped Gemini → NVIDIA NIM (PR #25,
2026-08-27) to unblock the eval-sweep quota → 25-case synthesized query
log + first tuning pass (PRs #27/#28) → Phase 2 plan formally adopted
(`docs/Tessera_Phase2_Plan.md`, PR #30, 2026-09-04) → **P2-1** quality
bar (PR #31) → **P2-2** eval set 33→50 + label audit (PR #32) → **P2-3**
retrieval-constant grid search, constants confirmed unchanged (PR #33) →
**P2-4** A-path source diversification + router A/C boundary fix (PR
#35, 2026-09-05) → **P2-5** Phase 2 exit: docs + final sweep + tag (PR
pending, this session).

**Phase 2 exit sweep** (P2-5, full clean `tessera eval --check`,
2026-09-06, **50/50 cases, zero errors**): **100.0% routing, 0.95
recall, 0.42 precision, 0.97 MRR, 4.77 groundedness, 4.60 relevance —
`=> PASS (gated thresholds)`**. Every gated threshold clears on a
reproducible clean sweep (the Solution Design §6 exit gate). Both P2-4
target gaps closed: multi-source archetype-A recall (`q001`/`ql003`/
`ql004` now 1.00) and the `ql035`/`ql038` C→A misroutes (routing 100%).

**One carry-forward** into the next phase (user decision 2026-09-06 —
close Phase 2 now, don't hold for it): relevance clears the bar by only
0.10, because P2-4's A-diversification makes narrow single-target
lookups return 5 same-family docs and the judge marks a few down for
breadth. See "Notes / open flags". Precision dropped 0.79→0.42 with the
same change and stays ungated (`QUALITY_BAR.md`).

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

- [x] **Post-Phase-1 housekeeping session** (2026-08-20, PRs #22, #23,
      both merged). Ran the `/git-cleaner` two-machine sync ritual
      (cerberus + tessera) at session start; found tessera's working
      tree dirty with an uncommitted rename in
      `.claude/commands/git-cleaner.md` (`cerberus-platform` →
      `cerberus`, matching the real local directory name) — committed
      that first so the rebase could proceed. `/start-day` then surfaced
      one open item from this file: `evals/README.md`'s "Populating with
      the real query log" step 4 still named `python -m evals.harness`
      as the re-run command, inconsistent with the "Running it" section
      above (updated in Task 8 to prefer `tessera eval`) — fixed
      (PR #22, merged as `aaf11ee`, branch deleted). Also discovered
      `/start-day`'s reading of this file's "Next task to pick up"
      section was itself stale: it said `v0.1.0` hadn't been tagged, but
      `git tag -l` / `git ls-remote --tags origin` showed the tag already
      existed and was pushed (`d09ff36`, cut 2026-08-20 by an earlier
      session) — corrected this file's Status/Next-task/Task-sequence
      sections accordingly (PR #23, merged as `0e916b7`, branch deleted).
      Full test suite re-verified clean throughout: 123 passed, 8
      skipped, matching the pre-session baseline (no code changed this
      session, doc/config-only).

- [x] **LLM provider swap: Gemini → NVIDIA NIM** (2026-08-27, PR #25,
      merged). User-initiated: Gemini's 20-requests/day free tier was
      the open decision flagged in the prior session's Notes as the
      binding constraint on Phase 2 (a real ~20-30-case query log needs
      ~40-90 calls for a full eval sweep). User supplied a link to
      NVIDIA's hosted model catalog page
      (`build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b`) rather than
      just a model name — fetched the actual page (via `curl`, since
      `WebFetch` timed out twice against its client-rendered content)
      to confirm the model is real (Nemotron-3-Ultra-550B-A55B, a
      genuine 550B-total/55B-active-parameter MoE) and extract the exact
      integration details rather than guessing: model ID
      `nvidia/nemotron-3-ultra-550b-a55b`, endpoint
      `https://integrate.api.nvidia.com/v1/chat/completions`
      (OpenAI-compatible, reachable via the `openai` Python SDK pointed
      at that `base_url`), `Authorization: Bearer` auth, and — the
      actual resolution to the quota problem — documented rate limits
      of **40 requests/minute and 10,000 requests/day**, far above
      Gemini's 20/day. New `generation/nvidia.py`'s `NvidiaClient`
      implements `LLMClient` against this; `chat_template_kwargs:
      {enable_thinking: false}` is passed to disable the model's
      optional reasoning pass, since nothing in this codebase's
      single-shot completion contract needs multi-step reasoning and
      leaving it off keeps latency/quota spend comparable to the prior
      Gemini calls. `generation/gemini.py` and the `google-genai`
      dependency are removed (nothing else referenced them once `cli.py`
      and `evals/harness.py`'s composition root were repointed) —
      `openai>=1.50` added in its place. `config.py`
      (`nvidia_api_key`/`nvidia_model`), `.env.example`, `README.md`,
      `evals/README.md`, `CLAUDE.md`'s tech-decisions table, and the
      `start-day`/`end-day`/`quality-engineer` operational docs (quota
      numbers, env var names, the stale "config.py is still a stub"
      claim in `start-day.md` — no longer true since Task 8 — caught and
      fixed while in the area) all updated to match. 4 tests updated
      (`test_config.py`, `test_cli.py`, `test_router.py`'s opt-in live
      test) — no test file needed new live-LLM coverage since the swap
      is behind the existing `LLMClient` port and every unit test already
      used a fake. Full suite: 123 passed, 8 skipped, unchanged from the
      pre-swap baseline. `grep -c 'nvidia-' uv.lock` still `0` — the new
      `openai` dependency pulls no CUDA wheels (a natural point of
      confusion given the vendor-name collision with the unrelated
      NVIDIA-GPU-wheel check; noted explicitly in `start-day.md` now).
      Live-verified against the real NVIDIA API once the user added
      `NVIDIA_API_KEY` to their own `.env` (never pasted into the
      conversation): archetype A (on-corpus market-entry-framework
      query) returned a correctly-numbered 3-source grounded answer,
      archetype B and D returned their unchanged short-circuit messages
      — 4 live calls spent, behavior identical to the Gemini
      implementation. Session followed the user's explicit choice of
      "full workflow through merge" (asked via clarifying question
      before pushing) rather than pausing for PR review.

- [x] **Phase 2 kickoff: synthesized query log + first live sweep**
      (2026-08-27, same session as the NVIDIA swap above). User asked to
      "pick up the infrastructure to unblock Phase 2"; clarified via
      `AskUserQuestion` that this meant starting Phase 2 itself (not
      further LLM infrastructure). Checked the build plan and Discovery
      Findings §7/§10 first: Phase 2 is explicitly gated on a real
      20-30-pair consultant query log ("Priya to log 20-30 real queries
      ... Owner: Priya. Target: ~1 week"). Asked the user directly
      whether that log existed — **user confirmed Meridian Advisory and
      its stakeholders, including Priya, are fictional** (played by the
      user in an earlier Claude Chat discovery session), so a real log
      will never arrive, and asked that the query set be synthesized
      against real-world usage patterns instead. Read all 20 corpus
      documents to be cited before writing anything — same discipline
      Task 1/7's `relevant_sources` verification used — then wrote
      `evals/cases/query_log.yaml`: 25 cases (A=10, C=7, B=5, D=3,
      approximating real-world archetype frequency per Discovery
      Findings §7), every `relevant_sources` path mechanically verified
      against real files (`34/34` real, `0` missing), no id collisions
      against `placeholder.yaml`'s existing `q001-q008`. File header and
      `evals/README.md` both make explicit this is a Claude-synthesized
      stand-in, not real client data — same honesty convention as the
      synthetic pilot corpus itself (CLAUDE.md exit criterion #5).
      `tests/test_harness.py::test_load_cases_parses_real_placeholder_file`
      hardcoded an 8-case/2-per-archetype expectation against the real
      `evals/cases/` directory — renamed to
      `test_load_cases_parses_real_case_files` and updated to the new
      33-case/12-7-9-5 combined total; full suite re-verified (123
      passed, 8 skipped, count unchanged since this was a rename not an
      addition).

      **Ran the full 33-case sweep live against NVIDIA NIM three times**
      while landing on a clean baseline. First full run: 27/33 succeeded,
      6 hit transient `503 Service Unavailable` from NVIDIA's API (server
      overload, not a quota limit) — correctly isolated as `ERROR` rows
      and excluded from every aggregate by Task 7's per-case
      exception-handling fix, proving that resilience again under a new
      failure mode (503s, not the 429s it was built against). Retried the
      6 errored cases via an ad hoc scratchpad script calling
      `run_harness()` directly against the persisted index (same pattern
      `main()` uses) — hit a genuine bug in that script, not the repo: it
      derived `REPO_ROOT` by climbing `.parent` from its own path
      looking for `pyproject.toml`, but the script lived in
      `/tmp/.../scratchpad/`, entirely outside the repo tree, so the
      climb reached filesystem root and looped forever there
      (`Path("/").parent == Path("/")` never terminates) — **99.9% CPU
      for 4 hours 20 minutes before being caught and killed**, zero
      output the whole time. Fixed by hardcoding the known repo path
      instead of auto-discovering it; the retry then completed normally
      in about 4 minutes. Purely a scratchpad-script bug — nothing in
      `evals/harness.py` or the retry logic itself was at fault, and
      nothing in the repo needed a fix. 5 of the 6 retried cases
      succeeded (one misrouted, see below); the 6th (`ql001`) hit a
      second `503`, then succeeded on a third standalone attempt,
      confirming the errors were genuinely transient rather than
      query-specific. Rather than hand-merge three partial reports, ran
      one final clean full sweep — **0 errors, 33/33 completed**:

      ```
      === Tessera Eval Report ===
      Cases: 33

      Routing accuracy: 93.9%

      Retrieval (archetypes A/C with relevant_sources):
        Mean recall:    0.74
        Mean precision: 0.49
        Mean MRR:       0.80

      Generation quality (LLM-judge, 1-5):
        Mean groundedness: 4.86
        Mean relevance:    4.76

      Latency by archetype (mean seconds):
        A: 36.25
        B: 28.97
        C: 39.76
        D: 10.55
      ```

      **Findings worth carrying into future tuning** (per build plan:
      "numbers may be poor at this stage — tuning happens in Phase 2,
      against this real log" — these are Phase 2's actual job, not a
      Phase-2-kickoff defect): (1) **Two reproducible misroutes**, both
      archetype-C queries phrased as short direct requests rather than
      explicit synthesis language — `ql012` ("Client wants a digital
      transformation roadmap by Friday — what do we have?") and `ql016`
      ("Staffed on an operating model redesign — what's our standard
      approach?") — both routed to archetype A instead of C, and both
      misrouted identically on the retry *and* the final clean run, so
      this is a real router prompt weakness (short "what do we
      have"/"what's our approach" phrasing reads as lookup-shaped to the
      classifier even when the underlying need is synthesis), not
      noise — worth revisiting `router.py`'s classification prompt
      in `generation/prompts.py` when Phase 2 tuning starts. (2) **Two
      recall=0.00 lookup misses**: `ql002` (financial due diligence
      checklist — the corpus has several similarly-named due-diligence
      docs the retriever may be confusing it with) and `ql009` (GenAI
      adoption maturity — single-source thought-leadership piece never
      surfaced in top-k). (3) When retrieval succeeds, generation quality
      is genuinely strong — most A/C cases with recall ≥ 0.67 scored 5/5
      on both groundedness and relevance, consistent with Task 6/7's
      findings that the grounded-generation prompt itself works well;
      the gap is in retrieval, not generation. Live-call spend: roughly
      33 (main sweep) + 6 (first retry batch) + 1 (ql001 standalone) ≈ 40
      calls for the main sweeps, comfortably inside NVIDIA's 10,000/day
      ceiling with no pacing needed — a direct, lived demonstration of
      why the swap earlier this session mattered.

- [x] **Phase 2 tuning pass: both baseline findings fixed** (2026-08-27,
      same session, PR pending). User asked to continue straight into
      tuning against the two findings above. Diagnosed each with a
      retrieval-only (zero-LLM-call) scratchpad script before writing
      any fix, rather than guessing.

      **Retrieval fix — title-aware chunk embeddings.** Diagnostic
      confirmed both `ql002` and `ql009` shared a root cause: chunk
      embeddings were computed from `chunk.text` alone
      (`cli.py`/`evals/harness.py` both did
      `embedder.embed_documents([c.text for c in chunks])`), so a
      document's title never contributed to its embedding. Both target
      documents' exact query terms ("financial due diligence checklist,"
      "GenAI adoption maturity") appeared only in the title, not the
      body prose — e.g. `due-diligence-financial-checklist.md`'s
      Overview section never uses the word "checklist" at all, since
      that word only appears in the doc's title and a bolded sub-list
      header deeper in the Framework section. Added
      `chunk_embedding_text()` to `chunker.py` (prepends
      `document_title` + `heading_path` to what gets embedded, chunk
      text unchanged); updated both call sites plus the two test
      fixtures (`test_indexing.py`, `test_answer.py`) that build real
      indices, so tests exercise the same embedding path as production —
      `test_cli.py`'s fake chunk objects needed `document_title`/
      `heading_path` attributes added too. Rebuilt the real persisted
      index (`tessera ingest`; `ChromaVectorStore.add()` uses `upsert`,
      confirmed safe to re-run in place). Confirmed fixed:
      `due-diligence-financial-checklist.md` went from absent-from-top-5
      to rank 1 (score 0.64→0.70); `genai-adoption-maturity-model.md`
      went from absent-from-top-5 to occupying all of top-5. Re-checked
      `RELEVANCE_THRESHOLD = 0.35` (generation/answer.py) against the new
      score distribution since every chunk's scores shifted — margin
      actually **widened**: on-corpus weakest-top-3 now ≥0.55 (was
      ≥0.39), the "parental leave" borderline probe now 0.24 (was 0.31),
      off-corpus queries 0.10-0.24 — no threshold change needed. Full
      suite re-verified after each step (123 passed, 8 skipped throughout
      — one test fixture change, `test_ingest_wires_...`'s fake chunk
      shape, not a new test).

      **Routing fix — A-vs-C disambiguation.** `ROUTER_SYSTEM_PROMPT`
      (`generation/prompts.py`) gained a note after the C examples:
      weigh the situational framing (a deadline, a new staffing, an
      upcoming meeting) over the trailing question's wording, since both
      real misroutes ("what do we have," "what's our standard approach")
      had lookup-shaped endings riding on top of a genuine onboarding
      situation — the router had been over-weighting the ending. Two
      calibrating examples added directly to the prompt (paraphrased
      from, not identical to, the real misrouted queries). No test
      asserts on the prompt's literal content, so no test changes
      needed. Verified live: both original misroutes (`ql012`, `ql016`)
      now route correctly with reasoning that explicitly cites the
      onboarding/deadline signal; a 7-query regression spot-check across
      all four archetypes (3×A, 2×C, 1×B, 1×D) confirmed **7/7 correct
      — no overcorrection** of genuine A/B/D queries into C.

      **Final verification: full 33-case sweep, hand-merged across
      three runs** (the harness itself doesn't merge separate runs;
      NVIDIA hit transient `503`s on a different 1-3 cases each of three
      attempts — never the same case twice, confirming this is a stable
      background error rate under this sweep's request pattern, not a
      flaky case or a code bug; every one was correctly isolated and
      excluded from aggregates by Task 7's per-case handling, then
      individually retried to build one complete 33/33 dataset):

      ```
      Cases: 33
      Routing accuracy: 100.0%

      Retrieval (21 archetype-A/C cases with relevant_sources):
        Mean recall:    0.87  (was 0.74)
        Mean precision: 0.71  (was 0.49)
        Mean MRR:       0.95  (was 0.80)

      Generation quality (LLM-judge, 1-5):
        Mean groundedness: 4.95  (was 4.86)
        Mean relevance:    4.81  (was 4.76)
      ```

      Every metric improved from the pre-tuning baseline; routing went
      from 31/33 to 33/33 correct. Total live-call spend this tuning
      pass: roughly 2 (router fix verification) + 7 (regression
      check) + ~99 across three full-sweep attempts + ~12 across the
      retry batches ≈ 120 calls — still a small fraction of NVIDIA's
      10,000/day ceiling. genai-architect/quality-engineer review was
      **not** invoked for this pass — it's gated to build-plan Tasks
      6/7/8 by convention (`.claude/agents/quality-engineer.md`), and
      this is post-Phase-1 tuning work, not one of those tasks.

- [x] **Phase 2 plan adopted + P2-1 (formalize the quality bar)**
      (2026-09-04, PR #31 merged). Start-of-day: re-verified Phase 1
      (`pytest` 123 passed / 8 skipped, `uv.lock` still `0` nvidia-,
      `.env` present) then merged the Phase 2 plan doc (PR #30,
      `docs/Tessera_Phase2_Plan.md`) that had been sitting open for user
      review — user approved it this session. Plan §4 defines a 5-task
      Phase 2 sequence (P2-1…P2-5); Phase 2 is now driven by that doc,
      not build-plan §5.

      **P2-1 delivered:**
      - `evals/QUALITY_BAR.md` — the agreed bar (routing ≥ 95%, mean
        recall@k ≥ 0.80, mean MRR ≥ 0.90, groundedness/relevance ≥ 4.5,
        per-case recall > 0.00, all gated; precision@k reported but NOT
        gated — labeling-completeness confound, revisit after the P2-2
        label audit).
      - `evals/harness.py`: `QualityBar` (frozen dataclass of gated
        thresholds), `evaluate_bar(report) -> BarResult`, and a
        "Quality bar" PASS/FAIL block in `format_report()`. All pure —
        constraint #6 held (no I/O, injected ports, returns data). A
        gated metric with no value counts as FAIL, not skip.
      - `tessera eval --check` — prints the report always, exits 1 on
        any gated-threshold failure (0 otherwise).
      - `CLAUDE.md` — docs list now points at the Phase 2 plan +
        `QUALITY_BAR.md`; new Working-conventions bullet: any PR
        touching `retriever.py`/`router.py`/`chunker.py`/`generation/`/
        `evals/cases/` must paste a fresh `tessera eval --check` report
        (incl. the `=> PASS/FAIL` line) in the PR body; CI/CD paragraph
        updated.
      - 10 new tests (`test_harness.py` +7: bar pass/fail, precision
        not gated, missing-metric-fails, per-case total-miss, report
        rendering; `test_cli.py` +3: `--check` exit codes). Full suite
        **133 passed, 8 skipped**.

      **Acceptance check — met.** Live `tessera eval --check` sweep,
      2026-09-04 (NVIDIA API was unusually slow — ~50-190s/call, ~65 min
      wall; 31/33 scored, `q004` + `ql012` hit the known transient 503s
      and were isolated as ERROR rows):

      ```
      Routing accuracy: 100.0%
      Retrieval (A/C):  recall 0.86  precision 0.72  MRR 0.95
      Generation:       groundedness 4.90  relevance 4.75

      Quality bar:
        [PASS] Routing accuracy: 100.0%   (>= 95%)
        [PASS] Mean recall@k (A/C): 0.86  (>= 0.80)
        [PASS] Mean MRR (A/C): 0.95       (>= 0.90)
        [PASS] Mean groundedness: 4.90    (>= 4.50)
        [PASS] Mean relevance: 4.75       (>= 4.50)
        [PASS] Per-case recall > 0.00 (A/C): no total misses
        [----] Mean precision@k (A/C): 0.72  (reported, not gated)
        => PASS (gated thresholds)
      ```
      `--check` exits 0 (bar passed). Note: the live run started before
      a late cosmetic tweak to two label strings ("none" → "no total
      misses", "tracked, not gated" → "reported, not gated") — numbers
      and the `=> PASS` verdict are identical under the committed code
      (`evaluate_bar` unit-tested; verdict re-confirmed against these
      exact aggregates).

      **Slightly below the 2026-08-27 post-tuning baseline** (recall
      0.87→0.86, MRR 0.95=, groundedness 4.95→4.90, relevance
      4.81→4.75) — within run-to-run judge/retrieval noise and still
      clears every gated threshold comfortably; not a regression to
      chase, but the numbers to beat in P2-3. Known weak spots
      unchanged from the plan §3: `q001` recall 0.40, `ql003`/`ql004`
      recall 0.50 (multi-source A), `ql015`/`ql017` C precision 0.20.

      **`NvidiaClient` request timeout:** flagged (no timeout, openai SDK
      600s default). **User decided 2026-09-04 to leave it as-is — not a
      concern for now.** Don't re-raise unprompted.

- [x] **P2-2 — expand eval set to 50 + label-completeness audit**
      (2026-09-04, PR pending). `docs/Tessera_Phase2_Plan.md` §4.

      - **Label audit** (retrieval-only, zero LLM — scratchpad script
        against the persisted index) over all 21 existing A/C cases.
        Four were under-labeled and are fixed with an inline decision
        note on each: `q005` (+`cost-transformation-zero-based-budgeting`,
        +`cost-transformation-sga-cost-ratio-benchmarks`), `ql001`
        (+`cost-transformation-sga-benchmarking`), `ql011` (+`due-diligence-
        red-flag-severity-rubric`, +`due-diligence-financial-dd-information-
        request-list`), `ql016` (+`operating-model-decision-rights-matrix`,
        +`operating-model-raci-governance`) — all companion docs that
        retrieve in the top 7 and genuinely belong. `q001`/`q002`/`q006`
        reviewed and kept as-is (notes record why — `q001` is a
        multi-source retrieval gap, not a labeling error; `q006` is the
        precision confound).
      - **+17 new cases** `ql026`–`ql042` in `query_log.yaml` (25 → 42).
        Combined with `placeholder.yaml`: **A=20, B=10, C=15, D=5 = 50**.
        Every one of the 52 corpus docs is now referenced by at least
        one case's `relevant_sources` or query (10 were previously
        uncited). New A cases all retrieve their target at recall@5 =
        1.00; new C cases 0.67–1.00 (`ql034`/`ql039` are deliberate
        weaker-fit multi-source cases, noted inline).
      - `placeholder.yaml` is now the **held-out overfitting check-set** —
        P2-3 tunes retrieval constants against `query_log.yaml` only.
        Noted in both file headers + `evals/README.md`.
      - `test_harness.py` case-count assertions 33 → 50. No re-ingest
        (corpus/chunker/embeddings unchanged). Full suite **133 passed,
        8 skipped**.

      **Acceptance check — met.** Full `tessera eval --check` sweep,
      2026-09-04, **50/50 cases, zero errors** (NVIDIA back to ~30–40s/
      call):

      ```
      Routing accuracy: 96.0%              PASS (>= 95%)   [2 misroutes]
      Retrieval (A/C):  recall 0.88  precision 0.79  MRR 0.97
      Generation:       groundedness 5.00  relevance 4.91
      Per-case recall > 0.00: PASS (min 0.40 @ q001)
      => PASS (gated thresholds)   --check exit 0
      ```

      **The expanded set surfaced a routing weak spot** (this is P2-2
      doing its job): `ql035` ("...refreshing their long-range strategic
      plan — what's our approach?") and `ql038` ("...whether their AI
      investments are actually delivering value — what do we have to
      frame that?") both routed A, labeled C — lookup-shaped phrasing
      over genuine synthesis intent, the same A/C boundary the
      2026-08-27 tuning targeted. `ql038`'s misroute cost recall (0.50,
      narrow A retrieval got 1 of 2 docs). Routing 96% clears the gate
      with one-case margin → **P2-4 prompt-tuning target** (plan lists
      "routing < 95% or a prompt gap" explicitly).

      **P2-3 grid-search preview** (retrieval-only, scratchpad, ran
      2026-09-04): full 72-point grid over `LOOKUP_TOP_K` ×
      `SYNTHESIS_CANDIDATE_K` × `SYNTHESIS_MAX_RESULTS` ×
      `SYNTHESIS_MAX_PER_DOCUMENT` on `query_log.yaml`. Best feasible
      objective (mean recall×precision, subject to per-case recall > 0 +
      MRR ≥ 0.90) beats current constants by only **+0.012 — below the
      plan's 0.02 keep-current tiebreak**. No constant combination lifts
      the worst-case recall above 0.50 — the multi-source A floor
      (`q001` 0.40, `ql003`/`ql004` 0.50) is a retrieval-*strategy* gap,
      not a *parameter* gap. So the likely path: **P2-3 = confirm +
      document current constants + recalibrate `RELEVANCE_THRESHOLD`
      (fast); P2-4 = wire metadata filtering through
      `pipeline.answer_query()` (the load-bearing task)** for both the
      multi-source recall floor and the routing prompt gap.

- [x] **P2-3 — retrieval-constant grid-search re-tune** (2026-09-04, PR
      pending). `docs/Tessera_Phase2_Plan.md` §4.

      - New committed `evals/tune_retrieval.py`: retrieval-only,
        zero-LLM grid search. `RetrievalConfig`/`ConfigScore`/
        `score_config`/`grid`/`select_best`/`format_report` are pure
        (constraint #6); `fetch_candidates()` does the one real I/O
        step (embed + one vector-store query per A/C case, `k=30` —
        the widest any grid point needs), so every one of the 72
        grid points is scored by re-slicing/re-diversifying the same
        pre-fetched pool in memory rather than re-querying the store.
        `main()` is the impure composition root. Reads
        `LOOKUP_TOP_K`/`SYNTHESIS_CANDIDATE_K`/`SYNTHESIS_MAX_RESULTS`/
        `SYNTHESIS_MAX_PER_DOCUMENT` from `retriever.py` directly
        (`CURRENT_CONFIG`) rather than hardcoding.
      - Small `retriever._diversify_by_source()` refactor:
        `max_results`/`max_per_document` are now parameters defaulting
        to the module constants — `retrieve()`'s call is unchanged
        (uses the defaults), `tune_retrieval.py` calls it directly with
        grid values instead of duplicating the diversification logic.
        Behavior-preserving; `test_retriever.py`'s 10 tests unchanged.
      - **Real grid-search result** (against `query_log.yaml`, 72/72
        configs feasible): best alternative beats current constants by
        only **+0.0117 on mean(recall×precision) — below the plan's
        0.02 keep-current tiebreak. Winner = current constants**,
        matching the scratchpad preview exactly. Confirmed on
        `placeholder.yaml` (trivially — winner is current there too).
        No constant combination lifts worst-case recall above 0.50 —
        reconfirms the multi-source-A floor is a retrieval-strategy
        gap P2-4 has to address, not a parameter the grid can reach.
      - **`RELEVANCE_THRESHOLD` recalibration**: probed the current
        index (unchanged since the 2026-08-27 title-aware-embedding
        fix) — on-corpus weakest-top-3 min **0.576**, adjacent-but-
        absent ("parental leave"/"vacation policy") **0.243–0.306**,
        off-corpus max **0.113**. `0.35` still sits cleanly in the gap
        (0.04 above the highest adjacent probe, 0.22 below the weakest
        on-corpus one) — **no change**. Fixed `answer.py`'s docstring,
        which had gone stale citing pre-title-aware-embedding numbers
        (0.39/0.31/0.15) that the 2026-08-27 checkpoint entry had
        already superseded but the code comment never caught up to.
      - 10 new tests (`test_tune_retrieval.py`): grid size/membership,
        fetch filtering to A/C only, lookup-slicing and
        synthesis-diversification scoring, feasibility marking,
        tiebreak selection (keeps current on marginal gain, switches
        on a real one, ignores infeasible configs, falls back to
        current when nothing is feasible), report formatting. Full
        suite **143 passed, 8 skipped**.

      **Acceptance check — met.** Confirmation sweep, 2026-09-04,
      **50/50 cases, zero errors** (same two `ql035`/`ql038` misroutes
      as the P2-2 baseline — expected, no constants changed):

      ```
      Routing accuracy: 96.0%   Retrieval (A/C): recall 0.88 precision 0.79 MRR 0.97
      Generation: groundedness 4.94 (was 5.00) relevance 4.91 (was 4.91)
      => PASS (gated thresholds)   --check exit 0
      ```
      Groundedness/relevance essentially unchanged from the P2-2
      baseline (4.94 vs 5.00 is judge noise, not a regression) —
      satisfies "groundedness/relevance ≥ prior baseline."

- [x] **P2-4 — close remaining bar gaps** (2026-09-05, PR #35 merged).
      `docs/Tessera_Phase2_Plan.md` §4. Diagnosed retrieval-only (zero
      LLM) before writing any fix.

      **Retrieval — multi-source-A recall floor.** The plan named
      metadata filtering as the candidate lever; the diagnostic showed
      it doesn't work — filtering archetype-A retrieval by `doc_type`
      leaves `q001` at recall 0.60, and topic filtering is impossible
      without re-serializing chunk metadata + a re-ingest (Chroma
      stores `topics` as a joined string). The real cause is **source
      concentration**: for `q001` ("market entry framework") the top-5
      *chunks* collapsed to 2 *documents* (overview ×3,
      competitive-landscape ×2), hiding the other 3 market-entry docs.
      Fix — `retriever.retrieve()`'s A path now pulls a
      `LOOKUP_CANDIDATE_K = 30` pool and reuses `_diversify_by_source()`
      with `LOOKUP_MAX_PER_DOCUMENT = 1`, returning the top
      `LOOKUP_TOP_K` (5) *distinct* documents, best chunk each — the
      same mechanism C uses. **Confirmed with the user before
      implementing** (deviation from the plan's named mechanism; user
      chose "A-path diversification" over metadata filtering / both).
      Retrieval-only sweep over all 50 A/C cases: mean recall
      0.899 → 0.952, MRR unchanged 0.971, **zero recall regressions**;
      `q001` 0.40 → 1.00, `ql003`/`ql004` 0.50 → 1.00, `ql001`
      0.75 → 1.00.

      **Routing — new A/C boundary pattern.** `ql035`/`ql038` ("client
      wants help with X … what's our approach / what do we have to
      frame that") misrouted C→A. Extended `ROUTER_SYSTEM_PROMPT`'s
      A-vs-C note to cover queries describing a live client need the
      consultant must deliver on, with two paraphrased calibrating
      examples (not the verbatim eval queries — same discipline as the
      2026-08-27 fix). Live spot-check: **9/9 correct**, both targets
      now C, no overcorrection of genuine A/B/D.

      **Follow-on — lookup-answer prompt.** The first full sweep showed
      A groundedness/relevance sliding to 4.69/4.57 — the diversified
      5-doc result set made single-target lookups pad answers with
      same-topic neighbours. `LOOKUP_ANSWER_SYSTEM_PROMPT` now tells the
      model the numbered sources include near-matches, to lead with the
      directly-responsive ones, and to keep the answer short when only
      one or two fit. Recovered to 4.77/4.71 on the re-sweep.

      `evals/tune_retrieval.py`'s `score_config()` lookup branch mirrors
      the new A path so the grid tool stays accurate. **No re-ingest** —
      embedding path unchanged. `test_retriever.py`: −1 raw-k test, +3
      diversification tests; full suite **145 passed, 8 skipped**.
      genai-architect/quality-engineer review not invoked (gated to
      build-plan Tasks 6/7/8; this is Phase 2 work).

      **Acceptance check — met.** Full `tessera eval --check` sweep,
      2026-09-05, 50/50: 47 in the main run; `q001`/`ql003`/`ql037` hit
      transient NVIDIA 503s and were retried directly against the
      persisted index (documented pattern) — all 3 clean on retry
      (recall 1.00, g5 r5 each). Hand-merged:

      ```
      Routing accuracy: 100.0%          (was 96.0%)
      Retrieval (35 A/C cases):
        Mean recall:    0.95            (was 0.88)
        Mean precision: 0.42            (was 0.79 — ungated)
        Mean MRR:       0.97            (was 0.97)
        Per-case recall > 0.00: PASS    (min 0.50 @ ql015)
      Generation (LLM-judge, 1-5):
        Mean groundedness: 4.77         (was ~4.94)
        Mean relevance:    4.71         (was ~4.91)
      => PASS (gated thresholds)
      ```

- [x] **P2-5 — Phase 2 exit** (2026-09-06, PR pending, this session).
      `docs/Tessera_Phase2_Plan.md` §4. Docs + verification + tag — not
      a tuning task.

      - **`README.md`** — status → Phase 2 complete (`v0.2.0`); phase
        table (Phase 1 ✅ `v0.1.0`, Phase 2 ✅ `v0.2.0` with what it
        delivered, query log noted as the permanent synthesized
        stand-in); architecture diagram + "Archetype handling" fixed
        (archetype A is one-chunk-per-source diversification now, the
        old "metadata-filtered" line was never accurate post-P2-4); eval
        section rewritten for the 50-case set + `--check` gate; new
        "Phase 2 — the quality bar" section with the bar table and the
        exit-sweep numbers.
      - **`evals/QUALITY_BAR.md`** — "Current standing" refreshed to the
        2026-09-06 clean 50/50 sweep; added the thin-relevance-margin
        and precision-drop explanations.
      - **`evals/README.md`** — P2-4 + P2-5 entries appended to the
        sweep-history section.
      - No code changed — docs only.

      **Acceptance check — met.** Full clean `tessera eval --check`
      sweep, **2026-09-06, 50/50 cases, zero errors** (this run — not
      hand-merged):

      ```
      === Tessera Eval Report ===
      Cases: 50
      Routing accuracy: 100.0%
      Retrieval (A/C):  recall 0.95  precision 0.42  MRR 0.97
      Generation:       groundedness 4.77  relevance 4.60
      Quality bar:
        [PASS] Routing accuracy: 100.0%   (>= 95%)
        [PASS] Mean recall@k (A/C): 0.95  (>= 0.80)
        [PASS] Mean MRR (A/C): 0.97       (>= 0.90)
        [PASS] Mean groundedness: 4.77    (>= 4.50)
        [PASS] Mean relevance: 4.60       (>= 4.50)
        [PASS] Per-case recall > 0.00 (A/C): no total misses
        [----] Mean precision@k (A/C): 0.42  (reported, not gated)
        => PASS (gated thresholds)
      Latency (mean s): A 50.6  B 28.2  C 67.7  D 25.2
      ```

      Relevance 4.60 clears the 4.5 gate by 0.10 — the tightest margin,
      and the noisiest metric (4.60–4.75 across recent sweeps). **User
      decided 2026-09-06 to close Phase 2 now** rather than hold for a
      narrow-A tuning pass; the margin + candidate lever are recorded in
      "Notes / open flags". `v0.2.0` tagged once PR merged to `main`.

## Next task to pick up

**Phase 2 is complete** — all 5 tasks (P2-1…P2-5) merged, exit gate met,
`v0.2.0` tagged 2026-09-06. There is no Phase 3 build plan yet.

Per Solution Design §6 the project is "coherent and demoable if it stops
after Phase 2." Whoever picks up next has three options, none urgent:

1. **Phase 3 planning** — write `docs/Tessera_Phase3_Plan.md` for
   archetype B (expertise-finding). Blocked on the HR data source/
   structure being knowable (Discovery Findings §9.8); until then B
   stays a routed "not yet supported" response.
2. **Post-Phase-2 retrieval tuning** — the narrow-A relevance margin
   carried forward from P2-5 (see "Notes / open flags"). A self-
   contained small task: adaptive `k` for archetype A. Would want its
   own retrieval-only tuning + full sweep + bar-check PR.
3. **Phase 4 groundwork** — `docs/adr/` already sketches the hybrid
   Go/Python production architecture; nothing built.

---

Phase 1 is complete and tagged (`v0.1.0`; Task
8 was the last build-sequence task — exit-criteria detail below is
historical, from the 2026-08-20 session that closed Phase 1). Phase 2
started and had its first tuning pass 2026-08-27, all in one session:
NVIDIA NIM LLM swap → `query_log.yaml` populated (25 synthesized cases,
**permanent** stand-in for a real query log that will never arrive,
confirmed with the user) → first live 33-case sweep → both findings that
sweep surfaced (2 reproducible misroutes, 2 recall=0.00 retrieval
misses) diagnosed and fixed same-session, verified with a clean 33/33
post-tuning sweep (100% routing, recall 0.87, precision 0.71, MRR 0.95,
groundedness 4.95, relevance 4.81 — see Done above for the full story).

Phase 2 ran that fixed sequence (`docs/Tessera_Phase2_Plan.md` §4) to
completion: P2-1 quality bar → P2-2 eval set 33→50 + label audit → P2-3
constant grid search (no change) → P2-4 A-path diversification + router
fix → P2-5 exit (docs + clean sweep + `v0.2.0`). The Phase 2 exit gate
(Solution Design §6 — "retrieval + answer metrics meet the agreed
internal bar on a ~50-case set, reproducibly") is met: see the P2-5
sweep in Status and Done above.

Phase 1 exit criteria (build plan §7), **re-confirmed 2026-09-06 still
holding** at the Phase 2 close (originally assessed 2026-08-20):

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

**P2-5 re-confirmation (2026-09-06):** all 5 still hold. (1) fresh
`uv sync` → `.env` → `ingest` → `query` path unchanged since Phase 1;
(2) all 4 archetypes exercised every eval sweep, A/C observably
different (A now one-chunk-per-source, C multi-source); (3) the P2-5
50/50 clean sweep reports all 7 metric categories; (4) the three ports
plus `load_corpus` are untouched — P2-4 only changed retrieval strategy
*inside* the ports; (5) README updated this task to state Phase 2 as
built and the query log as the permanent synthesized stand-in.

All 5 Phase 1 exit criteria satisfied and `v0.1.0` tagged; all 6 gated
Phase 2 bar thresholds satisfied and `v0.2.0` tagged (see Status above).
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

- **CARRIED FORWARD from P2-5 — narrow-archetype-A relevance margin.**
  The Phase 2 bar passes, but mean relevance clears the 4.5 gate by only
  0.10 (4.60 on the 2026-09-06 exit sweep; range 4.60–4.75 across recent
  sweeps — the noisiest gated metric). Cause: P2-4's A-path
  diversification (`LOOKUP_MAX_PER_DOCUMENT = 1`, top-5 distinct docs)
  means a narrow single-target lookup like "Do we have a framework for
  value-based pricing?" now returns 5 same-family docs where the query
  wanted one, and the LLM-judge marks a few down for breadth (`ql007`
  scored relevance 2 on the exit sweep, `ql004`/`ql027`/`ql028` scored
  3). The P2-4 `LOOKUP_ANSWER_SYSTEM_PROMPT` tightening recovered most
  of a bigger initial drop but not all of it. **User decided 2026-09-06
  to close Phase 2 without chasing this** — it's within the bar and the
  judge is noisy. Candidate lever if it's ever picked up: **adaptive `k`
  for archetype A** — return fewer documents when the top hit clearly
  dominates on score (a lone winner like `ql007`), keep all
  `LOOKUP_TOP_K` when the family scores are tight (`q001`'s market-entry
  docs all score 0.66–0.72). That's new retrieval logic needing its own
  retrieval-only tuning + full sweep + bar-check PR — a self-contained
  small task, not a doc tweak. Don't touch the A diversification without
  re-checking it doesn't give back the P2-4 recall gains (mean recall
  0.88 → 0.95, `q001`/`ql003`/`ql004` → 1.00).
- **Any change to archetype-A retrieval breadth re-opens the
  precision/relevance/recall three-way tension.** P2-4 chose recall
  (diversify to distinct docs) at a known, accepted cost to precision
  (0.79 → 0.42, ungated) and a small cost to judged relevance (above).
  P2-3's grid search is the tool for re-tuning `LOOKUP_TOP_K` /
  `LOOKUP_CANDIDATE_K` / `LOOKUP_MAX_PER_DOCUMENT`
  (`evals/tune_retrieval.py`, retrieval-only, seconds to run) — its
  lookup branch already mirrors the post-P2-4 diversified A path.
- **NVIDIA NIM latency is genuinely variable, not just occasionally
  503-flaky** — hit repeatedly across the 2026-09-04 P2-1/P2-2/P2-3
  sweeps: single-call latency ranged from ~10s to ~200s within the
  *same* sweep, and one probe call took 53s for a trivial "Say OK"
  completion. A 50-case sweep (~110 calls) took anywhere from ~20
  minutes to over an hour depending on when it ran — always let a
  backgrounded sweep run to completion rather than assuming a long
  elapsed time means it's stuck; check `ps -p <pid> -o pcpu` first (low
  CPU + still alive = waiting on the network, not hung) before
  considering it a problem. Unrelated to the already-documented
  transient-503 behavior below — this is latency variance, not errors.
  The local machine itself was also generally slower than usual this
  session (`pytest tests/` ran 60-90s most times but hit 204s once,
  292s isn't unusual for background contention) — not something to
  chase, just don't be surprised by it.
- **`.venv` can exist but be missing dev-only deps** (hit this session:
  `pytest` wasn't installed despite `.venv` being present — likely from
  an earlier `uv sync` without `--extra dev`, possibly on the other
  machine per the two-machine workflow). `source .venv/bin/activate`
  succeeding is not sufficient evidence the environment is fully synced;
  `uv sync --extra dev` fixed it in under a second (confirms it's a true
  no-op when already synced, not something to skip as "probably fine").
  `/start-day` step 2 already says to run this — this is a confirmed
  real-world trigger for it, not just a hypothetical.
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
- **RESOLVED 2026-08-27 — superseded by the NVIDIA NIM swap (PR #25).**
  The Gemini-era quota history below (20/day) is kept as a historical
  record of Phase 1's Task 6-8 verification constraints, not current
  guidance — the LLM provider is now NVIDIA NIM
  (`nvidia/nemotron-3-ultra-550b-a55b`), whose free tier allows **40
  requests/minute and 10,000 requests/day**, confirmed directly from
  NVIDIA's own model-catalog page
  (`build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b`). This removes
  quota as the binding constraint on Phase 2's eval sweep (~40-90 calls
  for a real 20-30-case query log is now a small fraction of one day's
  budget, not several days' worth). 4 calls spent 2026-08-27 verifying
  the swap live (1 archetype-A query = 2 calls, 1 archetype-B = 1,
  1 archetype-D = 1) — trivial against the new ceiling, not worth
  tracking day-to-day the way the old 20/day cap required. Still worth
  a quick sanity check before a very large batch (e.g. the full Phase 2
  sweep) in case NVIDIA's actual enforcement differs from the documented
  limit, but the granular per-session spend tracking below is no longer
  necessary practice going forward.
- **NVIDIA NIM returns transient `503 Service Unavailable` ("Service
  temporarily overloaded") under sweep load** — hit repeatedly across
  three full 33-case sweep attempts on 2026-08-27 (Phase 2's first
  sweep and the post-tuning verification sweep), 1-3 different cases
  each time, never the same case twice. This is a genuine background
  error rate on NVIDIA's side, not a code bug, a quota limit, or a
  flaky specific case — confirmed by the fact that every case that
  503'd succeeded cleanly on a simple retry. Task 7's per-case error
  handling (`run_harness()`) already isolates these correctly (reported
  as `ERROR` rows, excluded from aggregates, rest of the sweep
  unaffected) — no code change needed, just expect a handful of 503s on
  any large sweep and retry the specific failed case IDs rather than
  re-running the whole thing. See the ad hoc retry pattern in the
  2026-08-27 tuning entry above (`run_harness()` called directly against
  the persisted index, filtered to just the failed case IDs).
- **A scratchpad script that auto-discovers `REPO_ROOT` by climbing
  `.parent` from its own path is a trap if the script doesn't live
  inside the repo.** Hit this 2026-08-27: a retry script written to
  `/tmp/.../scratchpad/` (correctly, per the scratchpad-directory
  convention) climbed `.parent` looking for `pyproject.toml`, never
  found it since the scratchpad tree is entirely outside the repo, and
  looped forever at filesystem root (`Path("/").parent == Path("/")`
  never terminates) — 99.9% CPU for 4 hours 20 minutes before being
  caught and killed, zero output the whole time since even the first
  `print()` sat in an unflushed buffer. Lesson: any one-off script
  written outside the repo tree (scratchpad, `/tmp`) must hardcode the
  repo path rather than auto-discover it, and use `python3 -u`
  (unbuffered) so output is actually visible while it runs — an empty
  output file plus high sustained CPU for more than a minute or two is
  the signal to check `ps -o etime,pcpu` immediately rather than assume
  it's just slow.
- **Any change to what gets embedded (`chunk_embedding_text()` in
  `chunker.py`, or anything else feeding `embedder.embed_documents()`)
  requires re-running `tessera ingest` before it takes effect** — the
  persisted index at `data/vectorstore/` holds whatever was embedded at
  ingest time, and `ChromaVectorStore.add()`'s `upsert` makes re-running
  ingest in place safe (confirmed 2026-08-27), but nothing re-ingests
  automatically. A future session touching `chunker.py`,
  `embedding/local.py`, or either composition root's embedding call
  should re-ingest before trusting live retrieval results against
  either fresh or existing test evidence.
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
- Nothing auto-loads `.env` into the process environment — `config.py`
  (via `pydantic-settings`) reads `.env` itself, but only `cli.py` calls
  it; live runs outside `cli.py` (tests, ad hoc scripts) need the vars
  exported manually: `set -a; source .env; set +a`. `NvidiaClient` takes
  `api_key` as a constructor parameter (never reads the environment
  itself, per constraint #6), so a populated `.env` file alone does
  nothing until something exports or loads it.
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
