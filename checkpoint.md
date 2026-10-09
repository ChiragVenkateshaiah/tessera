# Tessera — Checkpoint

Last updated: 2026-10-09 (end of day)

## Status

**Phase 5 in progress — P5-0 to P5-6 done (P5-6 on 2026-10-07…09, #81).**
- **P5-0 to P5-2** (#71–#73, 2026-10-05/06): baseline sweeps + the judge's
  noise floor (A/C relevance ±0.03, B ±0.11); plan adoption + the `lc`
  extra + the spike; the `Pipeline` protocol and the marker split.
- **P5-3** (#75): LangSmith tracing through a redacting client (taint
  set + 6-word fragments, whole-trace hiding, env-var guard). The gated
  redaction test passes. **The live LangSmith step is still pending a key.**
- **P5-4** (#77): native delete-then-add (`VectorStore.delete_document()`),
  plus a LangChain ingestion stack (loader, splitter, embeddings, store,
  `index()` + `SQLRecordManager`, `cleanup="full"`), with parity and
  stale-chunk tests. `tessera ingest --stack lc`.
- **P5-5** (#78): LangChain retrievers (vector, BM25, hybrid, multi-query,
  rerank, parent-doc, `TesseraRetriever`), all principal-bound through
  `ScopedStore` and one shared contract. Leak and scope-binding tests
  pass. `tessera eval --stack lc` = native generation + LangChain
  retrieval. `lc-defaults` keeps every layer at native-equal quality.
- **P5-6** (#81): LangChain router, LCEL chain,
  `ChatNVIDIA`/`ChatGoogleGenerativeAI`, `.with_retry()` retries,
  `PeopleRetriever`. All three acceptance items met; five sweeps
  `=> PASS`, every switch within noise but one explained dip
  (`evals/reports/p5-6-generation.md`). Also `tessera eval --cases`,
  `--workers`, an adaptive shared `Pacer`, `evals/compare_sweeps.py`.
- **Next:** two questions the user deferred to the next session
  (`lc-defaults` for the generation switches; whether to run the planned
  `lc-model-client-2` sweep), then P5-7.
- **Decided 2026-10-08 (user):** LangGraph becomes the default stack
  after P5-10 if its sweeps pass; native stays as the eval/CI reference
  (plan §10.1).
- **Also on 2026-10-07:** the reviewer's-guide format for PRs (#76,
  `.claude/commands/pr-review.md`).

**Phase 4 complete (`v0.4.0`, 2026-10-03).** All seven plan tasks merged
(P4-1 #56, P4-2 code #59, P4-3 #60, P4-4 #63, P4-5 #64, P4-6 #65, P4-7
exit) plus the fb001 lookup fix (#61). Exit sweep: full clean `tessera
eval --check`, **92/92 cases, zero errors, `=> PASS`** — routing 100%,
A/C recall 0.96 / MRR 0.97 / groundedness 4.98 / relevance 4.95, B
unchanged (person recall 0.91, still +0.01), superseded cited 0, leaks
0/13, authorized recall 1.00, injection 100%. All 5 Phase 1 exit
criteria re-confirmed on a fresh clone. **One acceptance left open by
user decision (2026-10-03, tag with it recorded as a limit):** P4-2's
live Bedrock sweep, blocked on the AWS account (see Notes). The chat-UI
decision stays deferred until Bedrock latency is measured. **Next:
Phase 5** (LangChain / LangGraph / LangSmith, a parallel measured stack)
— plan drafted in PR #67, awaiting adoption.

Phase 4 = four production-readiness features aimed at the documented
reasons GenAI projects stall after proof of concept
(`docs/Tessera_Phase4_Plan.md`, replanned and adopted 2026-10-01, PR
#57): P4-2 Bedrock + routing + cost · P4-3 traces + feedback loop · P4-4
freshness + data-quality report · P4-5/P4-6 restricted tier +
permission-aware retrieval with a gated leakage eval · P4-7 exit
(`v0.4.0`). The ephemeral AWS deployment is **Phase 5**; CI/CD +
monitoring are Phase 6. The user wants AWS work done AI-assisted end to
end; `apply`/`destroy` still need their explicit go-ahead per run.

**Phase 3 complete** (`v0.3.0`, PR #53, 2026-10-01). All six plan tasks
plus two unplanned passes merged; the Phase 3 exit sweep — full clean
`tessera eval --check`, **55/55 cases, zero ERROR rows** — passes every
gated A/C and B threshold (`=> PASS`). All 5 Phase 1 exit criteria
re-confirmed on a genuinely fresh clone (machine `y520`). Two thin
margins carried forward (A/C relevance +0.10 — resolved 2026-10-02 by
#61; B person recall +0.01 — still open).

**Phase 3 adopted 2026-09-07** (`docs/Tessera_Phase3_Plan.md`, PR #38 +
#39; CLAUDE.md updated per its §7). Phase 3 = archetype B
(expertise-finding) built end to end over a synthesized ~600-person firm
expertise dataset, per that doc's §5 six-task sequence (P3-1…P3-6).

**P3-1 done** (PR #40, 2026-09-07): `data/expertise/` — a seeded
deterministic generator (`generate.py`, `SEED=20260907`) + its committed
600-consultant output (`people/*.yaml`, one file per practice) + schema
(`README.md`), and the loader `src/tessera/ingestion/expertise_loader.py`
(`Person`/`Skill`/`ProjectEntry`, `load_expertise`). Dataset is inert so
far — nothing in the query path reads it yet; the P3-1 verification sweep
matched the P2-5 baseline (`=> PASS`).

**P3-2 done** (PR #42, 2026-09-29): `ExpertiseStore` port +
`PersonMatch` (`store/base.py`), `ChromaExpertiseStore`
(`store/chroma_expertise.py`, separate `tessera_people` collection),
`profile_summary_text()` (`expertise_loader.py`), `tessera index-people`,
`TESSERA_EXPERTISE_DIR` setting. All 600 people indexed. Nothing in the
query path reads the people index yet.

**P3-3 done** (PR #44, 2026-09-29): `retrieval/expertise.py` —
`find_experts()` (candidate pool of 50 → evidence-strength re-rank →
top 5 with per-person `Evidence`), structured `Evidence` on `PersonMatch`,
and `evals/diagnose_expertise.py`. Bar check `=> PASS` (50/50, zero
errors; matches P2-5 baseline).

**P3-4 done** (PR #46, 2026-09-29): archetype B now answers end to end.
`generation/expertise.py` (two evidence floors; nobody qualifying →
fixed message, zero LLM calls), `EXPERTISE_ANSWER_SYSTEM_PROMPT`, B no
longer terminal in the router (`NOT_YET_SUPPORTED_MESSAGE` retired),
`pipeline.answer_query()` B branch with an optional `expertise_store`,
`tessera query` prints a People section. Bar check `=> PASS` (50/50, zero
errors).

**P3-5 done** (PR #48, 2026-09-29): archetype B is now measured — person
recall / precision / MRR, a B judge, a 6-case no-match set, B rows in the
bar. B thresholds entered **provisional** (user decision 2026-09-29, plan
§4.2 staging) because the first sweep missed person recall by 0.01
(0.89 vs 0.90).

**LLM retry/backoff + eval progress** (PR #50, 2026-09-29):
`RetryingLLMClient` (`generation/resilient.py`) decorates the `LLMClient`
port — 429/5xx retried with 45 s / 15 s exponential backoff (cap 120 s,
`Retry-After` honoured, 6 attempts), optional call spacing; `tessera eval`
paces calls 3 s apart and prints `[n/total]` progress and retry notices to
stderr. Motivated by NVIDIA NIM throttling far below its documented rate.

**B-retrieval pass** (PR #51, 2026-09-30): `parse_intent` reads "led/ran/
managed" and "recently/latest" from the query; `PersonMatch.rank_score`
orders the shortlist while `evidence_score` stays intent-independent (a
first version that discounted `evidence_score` pushed `ql019` under the
generation floor — caught by the gated sweep). Person recall **0.89 →
0.91**; **the B bar is now GATED** (`QualityBar.gate_expertise` default
`True`). Gated `tessera eval --check` → `=> PASS`.

**P3-6 done** (2026-10-01): README/QUALITY_BAR/evals README updated to
Phase 3 built; exit sweep 55/55 clean, `=> PASS`; `v0.3.0` tagged.

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
#36, 2026-09-06).

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
      same session, PR #28 merged). User asked to continue straight into
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
      (2026-09-04, PR #32 merged). `docs/Tessera_Phase2_Plan.md` §4.

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
      #33 merged). `docs/Tessera_Phase2_Plan.md` §4.

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

- [x] **P2-5 — Phase 2 exit** (2026-09-06, PR #36 merged).
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

- [x] **Phase 3 plan adopted** (2026-09-07, PRs #38 draft + #39
      adoption). `docs/Tessera_Phase3_Plan.md` written, reviewed
      same-day, four open decisions resolved in review (§8): dataset =
      **full ~600** from a seeded generator; B recall bar = **person
      recall@k ≥ 0.90** (stricter than A/C's 0.80); B generation = a
      **new `generation/expertise.py` module**; B fallback = a
      **dedicated `evals/cases/expertise_nomatch.yaml`**. CLAUDE.md
      updated per plan §7 (B off the do-not-build list; `ExpertiseStore`
      row; constraint #3 note; quality-bar check extended to
      `retrieval/expertise.py` / `generation/` / `data/expertise/`).

- [x] **P3-1 — expertise dataset + seeded generator + schema**
      (2026-09-07, PR #40 merged). `docs/Tessera_Phase3_Plan.md` §5.

      - `data/expertise/generate.py` — deterministic generator
        (`SEED = 20260907` + sorted corpus iteration → byte-identical
        re-runs, test-enforced). **600 consultants**, 10 practices,
        junior-heavy pyramid (206/182/114/76/22 Analyst→Partner). Every
        practice topic asserted against the real corpus vocabulary at
        generation time; every `authored` path is a real corpus file. A
        deliberate weak-signal tail (~17% flagged thin: 2 skills, low
        levels, mostly self-reported, ≤1 project, no authorship) and a
        `last_updated` staleness tail (62% 2026, then 2025/2024/2023).
      - `data/expertise/people/*.yaml` — committed output, one file per
        practice (~19k lines). Do-not-hand-edit.
      - `data/expertise/README.md` — schema; the
        evidenced-vs-`self_reported` decision (~45% of skill entries
        evidenced = backed by a project or authored doc on that exact
        topic); the **structural no-client-names guarantee**
        (`project_history` is `{industry, topic, role, year}` from closed
        vocabularies — no free-text field a client name could occupy).
      - `src/tessera/ingestion/expertise_loader.py` — `Person` / `Skill`
        / `ProjectEntry` dataclasses + `load_person` / `load_expertise`,
        the archetype-B analogue of `loader.py`. Validates every field
        against closed sets; given the corpus, validates `authored` paths
        + all skill/project topics against the real corpus. Constraint-#6
        I/O exemption same as `loader.py`; no hardcoded paths.
      - 25 new tests (`test_expertise_loader.py`,
        `test_expertise_generate.py`). Full suite **170 passed, 8
        skipped** (was 145/8).

      **Acceptance check — met.** Regenerates deterministically (test +
      manual diff); ~600 records load and validate; every `authored`
      path resolves to a real corpus file; `project_history` has no
      client names (structural — verified zero client-name-shaped
      strings in the committed files); spot-check confirms realistic
      pyramid + evidenced/claimed split + weak-signal tail, and sample
      queries (pharma-pricing, supply-chain, genai, due-diligence,
      decarbonization) each surface ~6–13 strong evidenced experts above
      a tail of ~15–24 weak self-reported claimers.

      **Quality-bar sweep** (dataset not yet wired into any query path —
      P3-2/P3-3 do that — so expected to match the P2-5 baseline). Full
      `tessera eval --check`, 2026-09-07, 49/50 in the main run (`ql031`
      transient NVIDIA 503, NVIDIA slow/flaky this run — A-latency ~2.5×
      normal — retried clean: A routing=OK recall 1.00 g5 r5):

      ```
      Routing 100.0%   Retrieval (A/C): recall 0.95  precision 0.43  MRR 0.97
      Generation: groundedness 4.82  relevance 4.62
      => PASS (gated thresholds)
      ```
      Within judge noise of P2-5 (4.77 / 4.60), as expected. B cases all
      still route B and short-circuit (no retrieval/judge). `genai-
      architect` / `quality-engineer` not invoked — gated by convention
      to build-plan Tasks 6/7/8; this is Phase 3.

- [x] **P3-2 — `ExpertiseStore` port + local Chroma implementation**
      (2026-09-29, PR #42 merged). `docs/Tessera_Phase3_Plan.md` §5.

      - `store/base.py`: `ExpertiseStore` ABC (`add` / `search` /
        `count`) + `PersonMatch(person, score, evidence=())`. Score is
        cosine similarity, higher = better (same contract as
        `SearchResult`). `evidence` is left empty by the store; P3-3
        fills it.
      - `store/chroma_expertise.py`: `ChromaExpertiseStore`, its own
        `tessera_people` collection (cosine) in the same persist dir as
        the document index. The full `Person` is stored as JSON in
        metadata so `search` rebuilds it without re-reading the YAML.
        `where` filters: `practice` / `office` / `title` exact match, and
        a `topic:<name>` boolean flag per evidenced topic
        (`topic_filter_key()`).
      - `expertise_loader.profile_summary_text()`: the text embedded per
        person (title, practice, office, skills with level + basis,
        project topics with industries; no name). Uses the existing
        `Embedder` port.
      - `tessera index-people` (zero LLM calls); `Settings.expertise_dir`
        / `TESSERA_EXPERTISE_DIR`.
      - 8 new tests (`test_expertise_store.py` +7, `test_cli.py` +1).
        Full suite **178 passed, 8 skipped**.

      **Acceptance check — met.** `tessera index-people` → 600 people
      indexed; manual queries ("who knows about pricing in retail",
      "procurement savings expert in London") return plausible people;
      the same index/search function runs unchanged against a fake store
      and Chroma. No `eval --check` sweep run: the PR touched none of the
      bar-gated paths and nothing in the query path reads the people
      index yet.

      **Observed for P3-3:** pure semantic search is loose on industry
      (the "retail" query's top hits were pricing people but not mostly
      retail projects) and doesn't separate evidenced from self-reported
      skills, which is exactly the evidence-strength re-rank's job.

- [x] **P3-3 — expertise retrieval path** (2026-09-29, PR #44 merged).
      `docs/Tessera_Phase3_Plan.md` §5.

      - `retrieval/expertise.py`: `find_experts(query, embedder, store,
        where=None, k=5, reference_year=2026) -> ExpertiseResult`. Pulls
        `CANDIDATE_K = 50` profiles, then re-ranks by evidence: authored
        docs (weight 1.5), projects (1.0, halving every 5 years, ×1.5
        boost when the query names the project's industry), skills (0.5 ×
        level/5, self-reported ×0.25); final = evidence score + semantic
        cosine. "Relevant" topic / industry / authored-doc is decided by
        embedding the distinct names in the pool with the injected
        `Embedder` and mapping cosine through a ramp (topics 0.35→0.65,
        industries 0.50→0.75; calibrated on real similarity: exact topic
        ≈0.9, unrelated ≈0.3–0.4). Pure per constraint #6; recency is
        measured from a passed-in year, not the clock.
      - `store/base.py`: `Evidence(kind, description, strength,
        self_reported)`; `PersonMatch.evidence` is now
        `tuple[Evidence, ...]` with `evidence_score` and `is_evidenced`
        (False when a match rests on self-reported skills alone — what
        P3-4 must flag). `score` stays raw semantic cosine.
      - `evals/diagnose_expertise.py`: zero-LLM ranked-people + evidence
        printer (`--practice`, `--office`, `-k`).
      - `Person.evidenced_topics` docstring corrected (project-based
        only; authored docs are credited in retrieval from document
        names).
      - 13 new tests (`test_expertise_retrieval.py`). Full suite **191
        passed, 8 skipped**.

      **Acceptance check — met.** "Who knows pharma pricing" on the real
      dataset + embedder: all top-5 have real pharma-pricing project
      history or authored docs; self-reported-only people rank below
      everyone with real pharma-pricing evidence. People evidenced on
      *adjacent* topics (e.g. a value-based-pricing project in pharma) can
      legitimately sit above someone with one old exact-topic project.

      **Quality-bar sweep** (`tessera eval --check`, 2026-09-29, 50/50,
      zero errors; document path untouched, so expected ≈ P2-5):
      ```
      Routing 100.0%   Retrieval (A/C): recall 0.95  precision 0.42  MRR 0.97
      Generation: groundedness 4.80  relevance 4.69
      => PASS (gated thresholds)
      ```

- [x] **P3-4 — B generation + router/pipeline/CLI wiring** (2026-09-29,
      PR #46 merged). `docs/Tessera_Phase3_Plan.md` §5.

      - `generation/expertise.py`: `generate_expertise_answer(result,
        llm)` + `filter_qualified()`. **Two floors** on retrieval's
        `evidence_score`, calibrated on 19 probe queries against the real
        dataset/embedder: `EXPERTISE_QUERY_FLOOR = 1.0` (the *best* match
        must reach it or the answer is `NO_EXPERT_MESSAGE` with **zero
        LLM calls**) and `EXPERTISE_PERSON_FLOOR = 0.05` (drops people
        with no topical backing but keeps weak self-reported-only people
        so the answer can show them flagged). Probe results: present
        topics best match 2.09–4.36 (weakest top-5 ≥ 1.37); absent topics
        0.00; adjacent-but-absent 0.66 ("HR compensation design") and
        0.19 ("legal contracts") — spurious partial-topic hits, both
        below the gate.
      - `prompts.py`: `EXPERTISE_ANSWER_SYSTEM_PROMPT` +
        `build_expertise_user_prompt()` (evidence lines, `Basis:
        EVIDENCED` / `SELF-REPORTED ONLY`, per-person profile date).
      - Router: only D is terminal now; `NOT_YET_SUPPORTED_MESSAGE`
        removed.
      - `pipeline.answer_query(..., expertise_store=None)`: B branch
        `find_experts` → `generate_expertise_answer`; without a store a B
        query gets `EXPERTISE_UNAVAILABLE_MESSAGE`. `AnswerResult` and
        `GeneratedAnswer` gain a defaulted `experts` field (citations stay
        empty for B). `retrieve()` still raises for B.
      - `tessera query`: prints "People:" with a `[self-reported only]`
        flag; passes no expertise store when the people index is empty,
        so A/C/D work without `index-people`.
      - Harness: B cases stay **routing-only** (`EXPERTISE_NOT_SCORED_NOTE`)
        until P3-5; otherwise B would fall through to `retrieve()` and
        error.
      - README: B bullet and phase-table row updated (full README pass is
        P3-6).
      - 11 new tests (`test_expertise_generation.py`) plus updated
        pipeline / router / harness / CLI tests. Full suite **202 passed,
        8 skipped**.

      **Acceptance check — met.** Live NVIDIA NIM `tessera query`:
      "pharma pricing" and "supply chain network optimization" → five
      named people each with concrete project / authorship / skill
      evidence, a dated-snapshot caveat, and a People list; "quantum
      computing" and "medieval poetry" → the no-match message, no
      generation call; A, C and D queries unchanged. Unit tests prove the
      zero-LLM no-match, self-reported flagging in the prompt, and that a
      B query never touches the document store.

      **Not exercised live:** no live query produced a self-reported-only
      match, so the model's actual wording for that flag is unobserved
      (prompt content is unit-tested). Worth including such a case in
      P3-5's set. Also seen live: the model orders people by its own
      judgement, so the answer's `[n]` order can differ from the People
      list order (numbers still match the right person).

      **Quality-bar sweep** (`tessera eval --check`, 2026-09-29, 50/50,
      zero errors; all 10 B cases route B):
      ```
      Routing 100.0%   Retrieval (A/C): recall 0.95  precision 0.42  MRR 0.97
      Generation: groundedness 4.77  relevance 4.69
      => PASS (gated thresholds)
      ```

- [x] **P3-5 — eval harness B metrics + bar extension + no-match set**
      (2026-09-29, PR #48 merged). `docs/Tessera_Phase3_Plan.md` §5.

      - `evals/metrics.py`: `shortlist_recall_at_k` (denominator
        `min(|relevant|, k)` — plain recall@5 caps at 0.25 for a query with
        20 genuine experts; identical to recall_at_k when the set fits),
        `EXPERTISE_JUDGE_SYSTEM_PROMPT`; `judge_answer(..., system=)`.
      - `evals/harness.py`: B cases run `find_experts` →
        `generate_expertise_answer` (needs an `expertise_store`; without
        one a B case is routing-only) and are scored on the people the
        answer *presents*. Person metrics and B judge scores live in
        their own `CaseResult` / `EvalReport` fields so A/C means are
        never blended. No-match cases are correct only with the fixed
        message **and zero generation LLM calls** (`_CountingLLM`); a
        misrouted no-match case counts as wrong. B rows in
        `evaluate_bar` / `format_report`, behind `QualityBar.gate_expertise`
        (default `False` = provisional).
      - `tessera eval` requires `tessera index-people`;
        `python -m evals.harness` builds a temporary people index.
      - Cases: 9 scored B cases with `relevant_people` (derived
        mechanically from raw records — project/authored evidence on the
        topic, plus industry or role+recency where the query says so —
        then audited against retrieval's top 20); new
        `evals/cases/expertise_nomatch.yaml` (6). Set is now 55 cases
        (A20 B15 C15 D5). A test asserts every id resolves.
      - Label audit findings: `ql042` set widened (query names reshoring
        *and* network moves); `ql020` ("open banking and payments") is
        **not** a clean absence (best evidence 1.69 — "banking" matches
        core-banking experts) so it stays a scored case against that
        nearest expertise; `ql022` (clinical trials) moved to no-match.
      - 26 new tests. Full suite **230 passed, 8 skipped**.

      **Sweep** (`tessera eval`, 2026-09-29, 55/55, zero errors — run
      through a scratchpad-only pacing wrapper, see Notes):
      ```
      Routing 100.0%   Retrieval (A/C): recall 0.95  precision 0.42  MRR 0.97
      Generation (A/C): groundedness 4.83  relevance 4.71
      Expertise (B): person recall 0.89  precision 0.89  MRR 1.00
                     groundedness 5.00  relevance 4.67
      No-match refusal rate: 100%
      ```
      Every gated A/C row passes (unchanged from Phase 2). The one B row
      short of its threshold is person recall (0.89 vs 0.90): `ql019`
      0.40, `ql041` 0.80, `ql042` 0.80. The run's own verdict was `=>
      FAIL` because B was gated in that build; with B provisional as
      merged, the gated verdict on the same numbers is PASS — **derived,
      not a separate run** (a re-run costs ~1.5 h of LLM calls).

      **Real gap found:** `ql019` ("who *led* our M&A integration
      engagements recently") — the label is right (lead roles since 2023,
      5 people) but retrieval gets 2 of 5: it doesn't weigh role. A
      throwaway experiment weighting project evidence by role seniority
      did **not** fix it (still 2/5; it did lift `ql042`), so it was
      reverted rather than shipped to fit an 8-case set.

- [x] **LLM retry/backoff, call pacing, eval progress** (2026-09-29, PR #50
      merged). Not a numbered plan task; done first because sweeps had
      become unreliable and opaque (see Notes).

      - `generation/resilient.py`: `RetryingLLMClient(inner, max_attempts=6,
        min_interval, on_retry, sleep, clock)` — a decorator over the
        port, provider-agnostic so the Phase 4 Bedrock client reuses it.
        429 base 45 s, 5xx base 15 s, doubling, cap 120 s, `Retry-After`
        honoured; non-transient errors (400/401, parse errors, no status)
        raised immediately. Clock/sleep injected; never prints (progress via
        callback).
      - `NvidiaClient(sdk_max_retries=)` — composition roots pass 0 so the
        SDK's fast retries don't multiply with ours.
      - `run_harness(on_case_complete=)`; `tessera eval` = retry + 3 s
        pacing + progress on stderr; `tessera query` = retry.
      - 18 new tests; suite **246 passed, 8 skipped**.
      - Stock `tessera eval --check` (no wrapper), 55/55, zero errors, 83
        retries absorbed, ~1.5 h, `=> PASS` (B still provisional): groundedness
        4.77, relevance **4.57** (thin — see Notes).

- [x] **B-retrieval pass: lead/recency intent; gate the B bar**
      (2026-09-30, PR #51 merged).

      - `retrieval/expertise.py`: `parse_intent(query)` → `QueryIntent(lead,
        recent)` from a closed lexicon (lexical on purpose: transparent,
        free, testable; a query with no cue words scores exactly as before —
        test-enforced). Lead intent: only `LEAD_ROLES` (engagement
        partner/lead/manager, workstream lead) count in full; other project
        roles, authored docs and skills ×0.25. Recent intent: recency
        half-life 5 y → 1.5 y.
      - `PersonMatch.rank_score` (new) orders the shortlist;
        `evidence_score` stays intent-independent because the generation
        floors (`EXPERTISE_QUERY_FLOOR = 1.0`, `EXPERTISE_PERSON_FLOOR =
        0.05`) were calibrated on that scale.
      - **The bug:** version 1 discounted `evidence_score` itself.
        Retrieval-only checks looked better (`ql019` 2/5 → 3/5) but the first
        gated sweep scored `ql019` **0.00** — every score fell under the
        floor, so the answer was "no obvious expert" for a query with real
        experts. Lesson: retrieval-only checks cannot see interactions with
        generation floors; only an end-to-end sweep did. Guarded now by
        real-data tests (the `ql019` query must clear the floor with ≥3
        people).
      - Held-out validation (retrieval-only, 8 "led … (recently)" queries on
        other topics, checked against raw records): lead-role hits in the
        top 5 **15/40 → 24/40**; no-cue queries return identical top 5.
      - `QualityBar.gate_expertise` default → `True`; `evals/QUALITY_BAR.md`
        B section updated (history + gated). 262 passed, 8 skipped.

      **Sweep** (`tessera eval --check`, 2026-09-30, B rows gated, exit 0):
      ```
      Routing 100.0%   A/C: recall 0.95  precision 0.43  MRR 0.97
      Generation (A/C): groundedness 4.74  relevance 4.61
      B: person recall 0.91  precision 0.91  MRR 1.00
         groundedness 5.00  relevance 4.89   No-match refusal 100%
      => PASS (gated thresholds)
      ```
      Four A/C cases (`ql032`–`ql035`) were ERROR rows (NVIDIA degraded
      again; all 6 attempts exhausted, excluded from A/C means by the
      harness). Re-run on their own afterwards: all ok — recall 1.0 / 1.0 /
      0.75 / 1.0, judge 5/5. All B cases completed in the sweep itself.

      **Limits, stated plainly:** `ql019` is **0.60, not fixed** — of its 5
      gold people one (`c0049`, a pricing-practice profile with a single
      M&A engagement) never enters the 50-person semantic candidate pool
      (pre-existing pool starvation; `CANDIDATE_K` is an untried lever) and
      one (`c0189`, a 2023 lead) decays to rank 14 under "recently". The
      label counts core `ma-integration` only; `integration-management` (an
      adjacent co-tag) leads would widen the set to 7 and the top-5 hit rate
      would read 4/5 — the stricter label was kept deliberately. Person
      recall clears the bar by **0.01** on 9 cases; `ql041`/`ql042` (0.80
      each) unchanged.

- [x] **P3-6 — Phase 3 exit** (2026-10-01, PR #53 merged; `v0.3.0` tagged).
      `docs/Tessera_Phase3_Plan.md` §5. Docs + verification + tag — no
      code changed.

      - **`README.md`** — status → Phase 3 complete (`v0.3.0`); phase
        table (Phase 3 ✅ with what it delivered); "not in Phases 1–3"
        gains real HR integration / live sync; architecture diagram gains
        the expertise path (dataset → `expertise_loader` →
        `ExpertiseStore`/`chroma_expertise` → `retrieval/expertise.py` →
        generation); B bullet rewritten from "in progress"; setup/usage
        cover `TESSERA_EXPERTISE_DIR`, `tessera index-people`, B query
        output + call cost, the 55-case eval set, retry/pacing, and the
        widened `--check` trigger list; "Phase 2 — the quality bar" →
        "The quality bar" with A/C + B rows and the exit-sweep numbers.
        Also fixed the stale `cerberus-platform` repo name → `cerberus`
        (user request).
      - **`evals/QUALITY_BAR.md`** — "Current standing" → the P3-6 exit
        sweep with both thin margins explained; regression-discipline
        list matches CLAUDE.md (now includes `expertise.py` and
        `data/expertise/`).
      - **`evals/README.md`** — Phase 3 + P3-6 sweep-history entries;
        "What's in here" lists `diagnose_expertise.py` and
        `tune_retrieval.py`.

      **Acceptance check — met.** Full clean `tessera eval --check`
      sweep, **2026-10-01, 55/55 cases, zero ERROR rows** (8 transient
      NVIDIA 503s, all recovered by `RetryingLLMClient`; wall clock
      22 min — far faster than the 1–1.5 h of 09-29/09-30). Run on a
      fresh clone with freshly built indexes (336 chunks, 600 people),
      Python 3.14.4:

      ```
      === Tessera Eval Report ===
      Cases: 55
      Routing accuracy: 100.0%
      Retrieval (A/C):  recall 0.95  precision 0.42  MRR 0.97
      Generation (A/C): groundedness 4.83  relevance 4.60
      Expertise (B):    person recall 0.91  precision 0.91  MRR 1.00
                        groundedness 5.00  relevance 4.89
      No-match refusal rate (B): 100%
      Quality bar:
        [PASS] Routing accuracy: 100.0%              (>= 95%)
        [PASS] Mean recall@k (A/C): 0.95             (>= 0.80)
        [PASS] Mean MRR (A/C): 0.97                  (>= 0.90)
        [PASS] Mean groundedness: 4.83               (>= 4.50)
        [PASS] Mean relevance: 4.60                  (>= 4.50)
        [PASS] Per-case recall > 0.00 (A/C): no total misses
        [----] Mean precision@k (A/C): 0.42          (reported, not gated)
        [PASS] Person recall@k (B): 0.91             (>= 0.90)
        [PASS] Person MRR (B): 1.00                  (>= 0.90)
        [PASS] B groundedness: 5.00                  (>= 4.50)
        [PASS] B relevance: 4.89                     (>= 4.50)
        [PASS] Per-case person recall > 0.00 (B): no total misses
        [PASS] No-match correct-refusal rate (B): 100%
        [----] Person precision@k (B): 0.91          (reported, not gated)
        => PASS (gated thresholds)
      Latency (mean s): A 19.3  B 16.1  C 31.4  D 4.2
      ```

      Lowest cases: A/C relevance 3 on `ql004`/`ql007`/`ql027`/`ql028`
      (the same four narrow-A lookups as P2-5); A/C recall min 0.50
      (`ql015`); B person recall `ql019` 0.60, `ql041`/`ql042` 0.80 — all
      unchanged from PR #51.

      **Phase 1 exit criteria re-confirmed (2026-10-01)** — see the end of
      "Next task to pick up". `v0.3.0` tagged on the merge commit.

- [x] **`tessera chat` + tuning probe** (2026-10-01, PR #54 merged,
      post-Phase-3, not a plan task). `tessera chat [--transcript FILE]`: interactive session
      over `answer_query()` — loads indexes + embedder once, labels each
      answer with archetype + latency, survives a failed question, appends
      a Markdown transcript. Each question is independent (no
      conversation memory — that's ADR 0003 session state). Shared
      `render_answer()` / `_open_stores()` now back both `query` and
      `chat`. 5 new CLI tests; live smoke test (1 A, 1 D) clean. CLI-only
      — no retrieval/prompt change, so no bar check required. Tuning
      probe result recorded under "Next task to pick up". Purpose: the
      user's 20-question manual test and a LinkedIn demo recording.

- [x] **P4-1 — HTTP API + `tessera serve`** (2026-10-01, PR #56 merged).
      `src/tessera/api.py`: `create_app(llm, embedder, store,
      expertise_store, *, llm_name)` — receives built dependencies, never
      reads config (composition-root adapter, constraint #6).
      `POST /api/ask` → JSON (`archetype`, `archetype_label`, `answer`,
      `citations`, `experts` with evidence, `latency_s`); blank/oversized
      questions → 422; a pipeline exception → `502 {"error": …}` with the
      trace only in the server log; one question at a time (a lock).
      `GET /api/health` (index counts, people search on/off, LLM name);
      `GET /` placeholder page until P4-2. `tessera serve [--host]
      [--port]` builds the deps once, as `chat` does. Archetype labels
      moved to `tessera/labels.py` (shared by CLI + API). Deps: `fastapi`,
      `uvicorn`; dev: `httpx` (TestClient); lockfile still has zero
      `nvidia-*` packages. 12 new tests (`tests/test_api.py`) → 279
      passed, 8 skipped.

      **Acceptance — met.** Live `tessera serve` + `curl`: A ("red-flag
      severity rubric") → 5 citations led by the rubric, 40.6 s; B
      ("digital capability assessments") → 5 evidenced experts, 59.3 s;
      D ("Acme vs Globex") → refusal, 5.8 s; blank question → 422. NIM
      latency was high again; server log clean.
      Gotcha: `pkill -f 'tessera serve …'` killed its own shell (the
      pattern matched the command line) — use `pkill -f '[t]essera serve'`.

- [x] **Phase 4 planned, adopted, then replanned and re-adopted**
      (2026-10-01, PRs #55 and #57).
      - **#55** adopted a first Phase 4: local chat UI → Claude on
        Bedrock → one container → ephemeral Terraform stack (deploy →
        demo → destroy). User decisions recorded in that plan: FastAPI +
        plain HTML/JS, Claude on Bedrock, Chroma indexes baked into the
        image, Terraform.
      - P4-1 (HTTP API, #56) was built under it. The live answers took
        40–60 s on NIM, and the user decided a UI on that latency wasn't
        worth building yet — and asked instead what would make the
        project stand out.
      - **Research** (web, 2026-10-01): Gartner — >50% of GenAI projects
        abandoned after POC, citing poor data quality, inadequate risk
        controls, escalating costs, unclear business value; MIT NANDA
        2025 — 95% of pilots show no P&L impact, a "learning gap" (no
        feedback, no improvement); vendor consensus that RAG leaks content
        users couldn't open directly unless permissions are enforced
        before ranking. Sources are listed in plan §0.
      - **#57 replanned Phase 4** around those causes, all four features
        chosen by the user: Bedrock + routing + cost per answer (P4-2);
        traces + feedback-to-eval loop (P4-3); freshness + data-quality
        report (P4-4); restricted tier + permission-aware retrieval with
        gated leakage and prompt-injection evals (P4-5/P4-6); exit P4-7.
        AWS → Phase 5 (the #55 design carried over, plan §9); CI/CD +
        monitoring → Phase 6. Access control follows Discovery §4:
        per-engagement ethical walls over the 600 consultants,
        deny-by-default, fail closed, and a human review gate rather than
        any claim to detect anonymized-but-identifiable content.
      - **Chat UI: decision pending** until after P4-2 (plan §7), with the
        user's note: "a persona switcher showing access control would be
        the strongest demo shot".
      - CLAUDE.md updated twice (per #55 §7, then per #57 §10).

- [x] **P4-2 code — Claude on Bedrock, model routing, cost accounting**
      (2026-10-02, PR #59 merged). **Acceptance not yet met** — the live
      Bedrock sweep is blocked on the AWS account (see Next task, Notes).
      - `generation/bedrock.py`: `BedrockClient` over the Anthropic SDK's
        `AnthropicBedrockMantle` (`anthropic[bedrock]` 1.11), credentials
        from a named AWS profile (default `tessera`), `output_config.effort`
        for answers, temperature 0.0 sent only to Haiku 4.5 via
        `extra_body` (SDK 1.x removed the kwarg; Opus 4.7+ rejects it),
        refusal / empty text raise.
      - Port: `LLMClient.complete_with_usage()` (default usage=None, so
        `complete()` and every caller unchanged); NIM, Bedrock and
        `RetryingLLMClient` report tokens. `generation/usage.py`:
        `UsageRecorder` made **per answer inside the pipeline** (nothing
        threaded through the composition roots — a deliberate deviation
        from plan §3.1.3's "injected recorder"), `UsageSummary` (cost None,
        never understated, when a call is unmetered or a model unpriced),
        `ModelPrice`; `config.MODEL_PRICES` = Anthropic's rates, unverified
        for Bedrock.
      - `answer_query(router_llm=)`, `AnswerResult.usage`; harness
        `router_llm`/`judge_llm`/`prices`, per-case tokens + cost (judge
        never counted), cost by archetype, provisional "Mean cost per
        answer" bar row (`QualityBar.max_mean_cost_per_answer_usd`,
        `gate_cost`).
      - CLI `TESSERA_LLM_PROVIDER=nvidia|bedrock` (default nvidia), eval
        judge stays on NIM, a missing AWS profile fails fast; API
        `usage` + `cost_usd`.
      - Regression sweep (NIM, 55/55, zero errors, 21 min): `=> PASS`,
        every gated row matching the Phase 3 exit.
      - IAM correction recorded in plan §3.1.1/§8: the Mantle endpoint
        authorizes `bedrock-mantle:CreateInference`, not
        `bedrock:InvokeModel*`.

- [x] **Chat-UI decision: deferred** (user, 2026-10-02). Plan §7 put it
      after P4-2 once Bedrock latency was measured; that can't be measured
      yet, so P4-3 started first. Ask again once the live Bedrock sweep
      has run.

- [x] **P4-3 — request traces + feedback-to-eval loop** (2026-10-02, PR #60
      merged). **Acceptance met.**
      - `tessera/trace.py`: the pipeline returns a `Trace` as data (route
        reasoning, every retrieved chunk/person with its score and whether
        it cleared the floor and reached the model, floors applied,
        fixed-response flag); `trace_record()` builds the JSON line. Only
        `cli.py`/`api.py` write it (`data/traces/traces.jsonl`,
        gitignored).
      - `tessera/feedback/`: `TraceLog` + `FeedbackStore` ports, local
        JSONL implementations, `candidate_cases()` (latest thumbs-down per
        trace → an unlabelled candidate marked `status: candidate`, which
        `load_cases` refuses).
      - API: `trace_id` on every answer, `include_trace`, `POST
        /api/feedback` (201/404/503/422). CLI: `query`/`chat` print the
        trace id; `tessera feedback add | review | to-cases` (`to-cases`
        refuses to write into `evals/cases/`).
      - Not yet traced: the principal and per-filter removal counts — they
        arrive with P4-4/P4-6.
      - **The loop, live:** "Where's our worked example for calculating
        price elasticity from transaction data?" answered "no worked
        example exists" — wrong; the trace showed only the Elasticity
        Calculation Reference's Overview chunk was shown. Thumbs-down →
        `to-cases` → labelled as **`fb001`** in the new
        `evals/cases/feedback.yaml` (labels drafted by Claude from the
        document; user approved the merge) → scored G2/R1 by the next sweep
        (56/56, `=> PASS`, A/C relevance 4.56 — 0.06 above the gate).

- [x] **Lookup retrieval fix for fb001: parent-document expansion**
      (2026-10-02, PR #61). A still picks 5 distinct documents (P2-4
      family recall unchanged by construction) but shows the **top 2
      whole**, in document order (`LOOKUP_EXPAND_DOCUMENTS = 2`); the
      caller's `where` applies to the expansion (P4-6 will rely on it);
      expanded chunks are re-checked against the document path. Prompt and
      citations number **one source per document**
      (`prompts.group_by_document` / `format_source_group`), also for
      synthesis.
      - Root cause (retrieval-only probe): the worked example's Framework
        chunks rank 6–7 *within their own document* (0.59/0.55, below
        Overview 0.62), so no chunk-selection rule can surface them; three
        of the five slots went to 0.39–0.42 filler.
      - **Rejected first:** a relative score floor + extra chunks from
        strong documents — 101-setting grid: A recall worst case 0.50, and
        never both worked-example chunks. Don't retry.
      - First sweep **failed** (groundedness 4.47): the judge still numbered
        chunks while answers cited documents. Fixed with one shared
        formatter + a test pinning the judge's source block to the answer
        prompt's.
      - Final sweep (56/56, zero errors, 29 min): `=> PASS`, **A/C
        groundedness 4.94, relevance 4.92**; fb001 G5/R5; the P2-5
        narrow-A carry-forward (`ql007`/`ql027`/`ql028` relevance 3) now
        5/5/5. Watch item: `ql011` (C) relevance 5→3 in both post-change
        sweeps. Sweep input tokens +28%.

- [x] **P4-4 — freshness + data-quality report** (2026-10-03, PR #63).
      Front matter `status: current|superseded` + `superseded_by` (target
      must exist and be current, no chains), and `review_status`; five
      superseded methodology docs (`*-2018.md`/`*-2019.md`). Superseded
      chunks are excluded in the store query (`$ne`, so an old index still
      works); a second query finds superseded docs that would have ranked,
      their current version takes the slot if it didn't rank, and a
      **fixed code-appended note** names both (the judge grades the answer
      without it). Trace `removed.superseded`. `tessera data-report`
      (`ingestion/data_quality.py`, lenient `scan_corpus`; near-dup
      threshold 0.85 calibrated: 19 known Related Frameworks pairs, highest
      other pair 0.81). 7 `freshness.yaml` cases worded in the retired
      methods' vocabulary (old version +0.12–0.28 above current); gated row
      "Superseded cited as current" = 0. Sweep 63/63 `=> PASS`.
- [x] **P4-5 — restricted tier data, walls, access eval sets**
      (2026-10-03, PR #64). `sensitivity`/`engagement` labels (unlabelled
      outside `methodology/`+`thought_leadership/` = restricted, fail
      closed; store chunk without a label reads restricted). 12 fictional
      engagement summaries (`data/corpus/engagements/`); `data/access/
      walls.yaml` from a seeded generator (`SEED=20261003`, 7-person
      teams, drift test) + 3 demo personas (`c0048` cleared partner,
      `c0014` walled analyst, `c0065` Kestrel-only); 2 pending-review case
      studies (never embedded). `evals/cases/access.yaml`: 13 leakage, 12
      authorized, 4 injection (`ac-i04` retrieves an internal test doc with
      a planted instruction + canary `CANARY-OVERRIDE-4417` — written with
      the user's explicit go-ahead after auto mode blocked it). Leak check
      deterministic for every case (uncleared restricted chunk, or a
      forbidden marker — an engagement fact the question never mentions).
      **Baseline (no filter): 13/13 leaked, 0/4 injection, 52 cases saw
      restricted content**; access rows provisional. Authorized recall
      scores a refused cleared principal as a miss.
- [x] **P4-6 — permission-aware retrieval** (2026-10-03, PR #65).
      `Principal` (`tessera/principal.py`, demo identity) passed into
      `answer_query()`/`retrieve()`; `permission_filter()` in every store
      query (internal + cleared engagements, deny by default) plus a Python
      re-check of every result; trace `principal` + `removed.restricted`
      (stripped from the asker's `include_trace` — existence oracle).
      `--as` on `query`/`chat`, `as_person` on `/api/ask`. Access rows
      **gated**. First sweep failed injection 75%: `ac-i03` was a correct
      plain-prose decline; **user signed off** on the contract also
      accepting the prompt's decline wording. Final sweep 92/92 `=> PASS`:
      leaks 0/13, authorized recall 1.00, injection 100%.
- [x] **P4-7 — Phase 4 exit** (2026-10-03, PR #66). README (status, phase table,
      "Phase 4 — production readiness" with the Gartner/MIT framing and
      honest limits), QUALITY_BAR current standing, this entry, Phase 1
      exit criteria re-confirmed on a fresh clone, final clean sweep
      (92/92, zero errors, `=> PASS`, numbers in Status). Tag: `v0.4.0` on the PR's merge commit, P4-2's live Bedrock sweep recorded as an open limit (user, 2026-10-03).
- [x] **P4-2 closed on Gemini; ADR 0007** (2026-10-05, PR #69).
      - **Why the move:** on AWS, Bedrock Haiku went from "not available"
        to `INVALID PAYMENT` once a debit card was the default, then the
        billing console warned that Indian-issued cards may fail. The
        user stopped on AWS there. On GCP, Claude Opus 5.5 hit "project's
        quota" (zero partner-model quota on new projects). The user chose
        GCP-native Gemini.
      - **The client:** `generation/gemini.py`, a `GeminiClient` using
        google-genai with `enterprise=True` and ADC. Flash routes and Pro
        answers, both at thinking `low`. Thinking tokens are billed as
        output, and Gemini 3 is sent no temperature. Selected by
        `TESSERA_LLM_PROVIDER=gemini`; the CLI fails fast without
        credentials or a project.
      - **Models, checked against the live model list:**
        `gemini-3.8-flash` (rejects `minimal` thinking; its default
        level took 28 s on a one-word reply vs 3 s at `low`) and
        `gemini-3.1-pro-preview` (the only Gemini 3 Pro). Prices are
        Google's published rates; Flash's is introductory until
        2026-12-31.
      - **Sweep, judge on NIM:** `=> PASS`. Routing 100%; A/C recall 0.96,
        MRR 0.96, groundedness 4.98, relevance 4.93; B person recall
        0.91, relevance 4.89; leaks 0/13; authorized recall 1.00;
        injection 100%. Cost $0.0120 per answer (A .0120, B .0091,
        C .0184, D .0008). Median latency A 11.8 s, B 11.3 s, C 15.5 s,
        D 3.8 s.
      - **Docs:** ADR 0007 written, ADR 0003 superseded, update notes on
        ADRs 0004/0005, the Phase 4 plan and the Solution Design.
        CLAUDE.md, README and QUALITY_BAR updated. Bedrock stays a
        dormant provider.
- [x] **Phase 5 plan + ADR 0006 adopted** (2026-10-05, PR #67), amended
      for ADR 0007:
      - `ChatGoogleGenerativeAI` instead of a Bedrock adapter;
      - sweeps on NIM, plus 3 Gemini sweeps at P5-10 (≈ $6–9; user
        decision);
      - Phase 6 = Cloud Run + chat UI; the video after Phase 7.
      A third Plan-agent review (§14) found 1 blocking and 8 should-fix
      issues, all folded in. `main` was merged into the plan branch (not
      rebased: the old draft commits touched `checkpoint.md`).
- [x] **P5-0 — baseline + judge noise floor** (2026-10-06, PR #71).
      Golden snapshot `evals/snapshots/v0.4.0.json`, `tessera eval
      --json`, and two NIM sweeps from a clean tree at `c76a095`, both
      `=> PASS`. Noise floor on the 88 cases scored in both: routing and
      retrieval identical on every case; groundedness unchanged; A/C
      relevance ±0.03 (one case, `q005` 4→5), B relevance ±0.11 (one
      case, `ql019` 4→3 — one judge point on 9 cases). Each sweep had 2
      ERROR rows on NIM 503s (`ql036`/`ql039`, `q002`/`q006`); the
      exports keep them (user decision), and the four were re-run alone,
      all 5/5, scores in the PR body. Sweeps took 1h47 and 2h10 (130 and
      160 transient 429/503s).
- [x] **P5-1 — adopt; dependencies; spike** (2026-10-06). Results below.
      **Install mode: optional extra `lc`** (user, 2026-10-06; plan §10.2).

      **Dependencies.** `[project.optional-dependencies].lc` in
      `pyproject.toml`. Resolved on Python 3.14.4 — every package
      installs and imports; no 3.14 blocker.
      | Package | Locked | Floor and why |
      |---|---|---|
      | langchain-core | 1.6.6 | ≥ 1.2.5, CVE-2025-68664 (dumps/loads serialization injection) |
      | langchain-classic | 1.0.8 | |
      | langchain-community | 0.4.2 | |
      | langchain-text-splitters | 1.1.3 | |
      | langchain-chroma | 1.1.0 | |
      | langchain-huggingface | 1.2.2 | |
      | langchain-nvidia-ai-endpoints | 1.4.3 | |
      | langchain-google-genai | 4.4.0 | |
      | rank-bm25 | 0.2.2 | |
      | langgraph | 1.2.13 | ≥ 1.0.10, CVE-2026-28277 (msgpack checkpoint deserialization) |
      | langgraph-checkpoint | 4.2.0 | ≥ 3.0.0, CVE-2025-64439 (JsonPlusSerializer RCE) |
      | langgraph-checkpoint-sqlite | 3.1.1 | ≥ 3.0.1, CVE-2025-67644 (filter-key SQL injection) |
      | langsmith | 0.14.4 | |
      | greenlet | 3.5.6 | **not in the plan** — see below |
      Lock checks: 124 → 156 packages, **0 pre-existing entries changed**
      (version, source, wheels and deps compared field by field;
      `google-genai` stays 2.28.0, `torch` stays on the CPU index);
      `grep -cE '^name = "nvidia-' uv.lock` = **0**. `langgraph` pulls
      `langgraph-prebuilt` transitively; it is installed but stays
      forbidden to import (P5-3's import test).

      **Where the plan's assumptions differed (each one is for the named
      task to act on):**
      1. **`SQLRecordManager` needs `greenlet`**, which `langchain-classic`
         doesn't declare (SQLAlchemy's asyncio import fails). Added to the
         extra. (P5-4)
      2. **`index()` replaces chunk ids with content hashes**, breaking
         `<stem>::<n>` (§3.2.2). `key_encoder=lambda d: d.id` keeps the ids
         and still cleans up shrunk documents (5 → 3: 2 deleted), **but an
         edited chunk is then "skipped" and its old text stays** — it needs
         `force_update=True` (re-embeds every chunk each run). P5-4 picks:
         id-keyed + `force_update`, or hashed ids with the chunk id in
         metadata (and `_chunk_index` reading metadata). (P5-4)
      3. **`langchain_chroma`'s `similarity_search_by_vector_with_relevance_scores`
         returns cosine *distance*, despite the name.** Ids and order match
         native exactly on every principal; native score = `1 − value`
         (max diff 0.0). `similarity_search_with_relevance_scores` (text
         query) returns similarity, also exact. The §3.4 re-scoring step
         must convert. (P5-5)
      4. **Precomputed embeddings work** (mechanism (a)): upsert through
         the chroma client into `tessera_lc_chunks`, read through
         `langchain_chroma.Chroma`. Prefix-on-write/strip-on-read (b) not
         needed. `HuggingFaceEmbeddings` = native vectors (max diff 8.6e-8
         documents, 0 queries). All native filters work unchanged
         (`$and`/`$or`/`$in`/`$nin`/`$ne`); a cleared principal sees 5
         restricted chunks on an engagement query, walled and none see 0.
      5. **Splitters vs native chunker** (70 docs / 427 chunks today):
         `MarkdownHeaderTextSplitter` is fence-aware (a `#` inside ```
         isn't a heading); `strip_headers=False` keeps `## Overview  ` in
         the text. `RecursiveCharacterTextSplitter(800)` **splits tables
         and fences** (2 table pieces without a header row; 2 pieces with
         unbalanced ```), which native never does. Expected; the
         comparison records it. (P5-4)
      6. **`ChatNVIDIA` differs from `NvidiaClient` (§3.5 parity, S11):**
         - **thinking is ON by default** (reasoning in `reasoning_content`;
           "Say OK" = 18 output tokens vs 2 with it off). Off via
           `.bind(chat_template_kwargs={"enable_thinking": False})` or
           `model_kwargs={...}` — both put it in the body;
         - **sends `max_tokens=1024` by default** (native sends none) and
           `stream: false`;
         - **errors are a bare `Exception("[503] {...}")`** — no status
           attribute, so `is_retryable` can't read it; the `retry` switch
           must parse the message prefix;
         - **`with_structured_output(include_raw=True)` raises
           `NotImplementedError`** (all methods). Without it:
           default/`json_schema` → `response_format: json_schema`;
           `function_calling`/`strict` → `guided_json` (no tools sent);
         - `usage_metadata` is populated (input/output/total).
         (P5-6)
      7. **`ChatGoogleGenerativeAI` (Agent Platform, ADC):**
         `vertexai=True, project, location` works with no API key;
         `thinking_level="low"` is sent as `ThinkingLevel.LOW`; **no
         temperature is sent** (parity with native); `usage_metadata`
         `output_tokens` **includes thinking** (87 = 1 + 86 reasoning,
         also in `output_token_details.reasoning`) — matches native's
         candidates + thoughts. Differences:
         - **`.content` is a list of blocks** (with a thought signature),
           not a string — use `.text` / a string parser;
         - **automatic function calling is not disabled** (the SDK warns;
           native disables it);
         - **`max_output_tokens=None`** by default (native sets
           `DEFAULT_MAX_OUTPUT_TOKENS`);
         - **`max_retries=6` by default** — would stack on Tessera's
           retries; set 0 under the native `retry` value;
         - errors: 503 → `GoogleAPIError` with `.code == 503`; 429 →
           `GoogleRateLimitError` with no `.code` (the `ClientError` cause
           has it);
         - `with_structured_output(include_raw=True)` works in all three
           modes (`json_schema`/`json_mode` → `response_mime_type`;
           `function_calling` → tools).
         Spend: ~8 Flash calls, < $0.01. (P5-6)
      8. **LangSmith hooks don't cover every field** (real `Client`,
         mocked HTTP, marker in inputs/outputs/metadata/error/event/child
         run): no hooks → leaks in `inputs`, `outputs`, `extra`
         (metadata), `error`, `events`; `hide_inputs`+`hide_outputs` →
         `error`, `events`, `extra`; `+ hide_metadata` → `error`,
         `events`; `anonymizer` → **`events`**. Only **`anonymizer` +
         `process_buffered_run_ops`** (which requires
         `run_ops_buffer_size`) reached **0**. P5-3's redacting client
         needs that combination, and the gated test must plant markers in
         errors and events, not just inputs/outputs. (P5-3)
      9. **Env-driven tracing is real:** with `LANGSMITH_TRACING=true` and
         no `tracing_context`, a bare `@traceable` posts to
         `api.smith.langchain.com/runs/multipart`. `tracing_context(enabled=False)`
         stops it. The env-var guard is needed. (P5-3)
      10. **The `Client` mounts its own adapter on a passed `session`**, so
          an HTTP mock must patch `Session.send` (not mount an adapter)
          — the first spike run reported a false "0 leaks" because of
          this. (P5-3 test design)

      **Confirmed as the plan assumed:** runtime context
      (`context_schema` / `Runtime`) keeps two invocations' recorders
      apart; append reducers; `Send` map-reduce; subgraph as a node, with
      `subgraphs=True` streaming; stream modes `values`/`updates`/
      `custom`/`debug` (and multi-mode tuples; `messages` is empty with
      no LLM); `RetryPolicy(retry_on=…)`; `get_state_history`;
      `update_state` before a node runs (time travel — adds a checkpoint);
      the Functional API (`@entrypoint`/`@task`); `interrupt()` +
      `Command(resume=…)` **across two processes** via `SqliteSaver`, and
      **the interrupted node re-runs from its start on resume** (a side
      effect before `interrupt()` ran twice). Fixed `run_id` and metadata
      via `langsmith_extra` reach the payload. `Client` has
      `create_dataset`/`create_examples`/`create_feedback`/
      `create_annotation_queue`/`add_runs_to_annotation_queue`/
      `push_prompt`/`pull_prompt`; `evaluate` and `evaluate_comparative`
      import. **Not exercised offline** (P5-9 does them, mocked then
      live): `evaluate()` runs, `evaluate_comparative`, annotation
      queues, the prompt hub.

      **Checkpoint serializer** (`langgraph-checkpoint` 4.2.0): `str`,
      `int`, `float`, `list`, `dict`, `set`, `bytes`, `datetime`,
      `date`, `Path` round-trip; **`tuple` comes back as `list`**;
      dataclass, enum and pydantic round-trip **with a warning that
      unregistered types "will be blocked in a future version"**
      (`LANGGRAPH_STRICT_MSGPACK=true` blocks now); a plain class fails
      (`TypeError`). Confirms §3.8.2's "state holds JSON types only";
      P5-8 should set strict mode.

      Spike scripts were scratchpad one-offs (no LLM except items 6–7:
      ~8 NIM calls, ~8 Flash calls).

- [x] **P5-2 — the `Pipeline` protocol, `PipelineRun`, the marker split,
      the harness through it** (2026-10-06, PR #73).
      - `pipeline.py`: `Pipeline` protocol (`run(query, principal) ->
        PipelineRun`) and `NativePipeline`. `PipelineRun` = the
        `AnswerResult` + every retrieval attempt, the chunks shown, the
        generator's output, the expertise shortlist, and routing /
        generation usage **always metered apart** (`generation_calls`).
        Ports and step functions are constructor parameters, defaulting to
        the core functions **looked up at run time** (monkeypatching
        `tessera.pipeline.find_experts` still works; P5-3 can wrap a step
        with `@traceable` in the composition root). `answer_query()` is a
        thin wrapper, signature unchanged (a test pins it equal to
        `NativePipeline.run().answer`).
      - Harness: every case runs through `Pipeline.run()` and is scored by
        `score_run()` from the `PipelineRun`; `_CountingLLM` is gone.
        `run_case`/`run_harness` take `pipeline=` (default native). Metrics
        read the **final** retrieval attempt; `restricted_seen` and the
        context check read **every** attempt.
      - Markers (§3.1.3): `forbidden_markers` is now `engagement → facts`
        and counts only for an engagement the principal isn't cleared for;
        `canary_markers` are answer-only (`ac-i04`). `access.yaml` was
        rewritten from its own leakage cases (12 engagements × 4 facts);
        every case keeps the same markers. A list-shaped
        `forbidden_markers` is now a load error. Leak strings:
        `chunk:<e>`, `marker:<fact>`, `canary:<phrase>` (was
        `marker:CANARY…`).
      - Context-marker check: zero-call, every chunk of every attempt,
        `CaseResult.context_marker_hits` (`<engagement>:<fact>`),
        `EvalReport.context_marker_checked/_cases`, bar row **reported**
        (`QualityBar.gate_context_markers=False`; gating needs sign-off,
        plan §10.5). It is **not** added to `leaked`.
      - Snapshot: runs through `NativePipeline` with routing forced by an
        injected `route_fn`; records `context_marker_hits`;
        `--check … --allow-added FIELD`. New reference
        `evals/snapshots/p5-2.json` (use it, not `v0.4.0.json`, from P5-3
        on).
      - CLI: `--stack native|lc` on `query`/`chat`/`serve`/`eval`,
        `TESSERA_STACK` default (`Settings.stack`); `lc` exits 2 ("isn't
        built yet") until P5-4..P5-7. The `--json` meta records `stack`.
        `query`/`chat`/`serve` still call `answer_query()` for native (the
        API needs no change); `eval` builds a `NativePipeline`.
      - **Proofs:** (1) snapshot identical to `v0.4.0.json` on 92 cases
        with `--allow-added context_marker_hits` (strictly: 92 diffs, all
        `None -> []`); (2) live sweep `evals/baselines/p5-2-native.json`
        at `94bc58c`, `=> PASS` — routing 100%, recall 0.99, MRR 0.97,
        groundedness 4.97, relevance 4.97, person recall 0.93, leaks 0/13,
        authorized recall 0.92, injection 100%, context-marker hits 0/16;
        (3) route-matched diff vs both P5-0 sweeps: 82 cases routed the
        same, **0 deterministic differences**, judge means within the
        noise floor (only `ql019` B relevance 4→3, as in P5-0); re-routed
        `ac-l06` D→A and `ac-a06` A→D (the Cobalt router instability —
        `ac-a06` refused is the 0.92 authorized recall). Person recall
        0.93 is `ql042` (the B miss) erroring out of the mean, not a gain.
      - 6 ERROR rows on NIM 503s (`ac-i04`, `q005`, `ql011`, `ql015`,
        `ql016`, `ql042`); 5 passed re-run alone, each identical to P5-0
        deterministically — incl. **`ac-i04` live: contract held, no
        leak, no canary**. **`ql016` failed four times** (sweep + 3
        re-runs) and is unscored in P5-2. Suite 456 passed (18 new).
- [x] **P5-3 — LangSmith tracing with taint-based redaction (native)**
      (2026-10-07, PR #75; sweep `evals/baselines/p5-3-native.json`).
      - `observability/guard.py` (framework-free): every command refuses
        to start while `LANGSMITH_/LANGCHAIN_` `TRACING`, `TRACING_V2`,
        `OTEL_ENABLED`, `RUNS_ENDPOINTS` or `ALLOW_UNPROCESSED_PAYLOADS` is
        set — **even with Tessera's tracing on** (stricter than the plan
        text; matches CLAUDE.md "refused, not honoured").
      - `observability/taint.py`: the corpus taint set (restricted and
        quarantined chunk texts, sentences, titles, paths; codenames as
        whole words), a per-request `RequestTaint` contextvar, and a
        `Redactor` that replaces exact terms **and any 6-word fragment**
        with `[withheld-N]`. The fragment pass was added because the gate
        found 6 markers leaking in a run name cut mid-sentence.
      - `observability/langsmith_tracing.py`: the redacting `Client`
        (`anonymizer` + `process_buffered_run_ops`; buffer size and
        timeout out of reach, so the tracer flushes only after a trace
        ends). Whole-trace hiding when retrieval touched restricted or
        quarantined content; unknown traces hidden (fail closed); no
        removal count; opaque per-process HMAC principal refs. The Tessera
        `trace_id` (UUID v7) is the root run id. Steps and LLM ports are
        wrapped at the composition root.
      - CLI/API: `TESSERA_LANGSMITH_*` config; an `Answerer` keeps
        `answer_query` when tracing is off; `create_app(answer=…)`; the
        trace id is created before the run.
      - Tests: the gated redaction test (real `Client`, `Session.send`
        mocked, 29 access cases + 4 scenarios, a leaky fake model, a
        plain-client control), the env guard (zero outbound), the
        allow-list import walk, the no-agents check, and the subprocess
        native-only test. Sweep `=> PASS`, 0 ERROR rows, 65 min.
      - **Pending:** the live step (one traced query per archetype in
        LangSmith) needs `TESSERA_LANGSMITH_API_KEY`.
- [x] **Reviewer's-guide format** (2026-10-07, PR #76). The user found an
      ELI5 review didn't help them read a diff and chose a 5-part guide:
      decision summary, reading order, `file:line` checks, before/after,
      risks + accept criteria, then the "Files changed" link. It lives
      in `.claude/commands/pr-review.md` (also `/pr-review <n>`), and
      CLAUDE.md links it. Triggers: `gh pr view <n> --web` in chat,
      `/pr-review <n>`, and after every `gh pr create`.
- [x] **P5-4 — LangChain ingestion and indexing** (2026-10-07, PR #77;
      evidence `evals/reports/p5-4-ingestion.md`, sweep
      `evals/baselines/p5-4-native.json`).
      - Native: `VectorStore.delete_document()` (port change, default
        `NotImplementedError`); `ingestion/indexing.py` `index_corpus()`
        = delete every loaded document's chunks (quarantined included),
        then add the indexable ones. `ingest` and `evals/snapshot.py` use
        it; the snapshot is identical. `chunker.make_chunk()` builds ids
        and labels for both splitters.
      - LangChain (`lc/`, `integrations/`): `TesseraCorpusLoader`;
        `split_document` (all six heading levels — the corpus has
        `####`); `TesseraChroma` (embeds the title/heading prefix, stores
        the body) + `LangChainChromaStore` (chunk id from metadata);
        `index_with_record_manager` with **`cleanup="full"`** — the plan
        said `incremental`, which leaves a newly quarantined document
        indexed (pinned as a strict xfail).
      - `tessera ingest --stack lc` with `TESSERA_LC_*` switches. Evidence:
        prefix on = recall 0.959 vs ~0.84 off; the LangChain splitter ties
        native at 800–1600 chars but splits fences and tables, so
        `lc-defaults` keeps the native splitter; `index()` skips unchanged
        chunks (14.5 s vs 26 s re-run). One HNSW approximate-search
        difference on `ac-l03`'s superseded score (same document).
      - Sweep `=> PASS`, 0 ERROR rows, 62 min. Suite 522 passed.
- [x] **P5-5 — LangChain retrieval** (2026-10-07, PR #78; evidence
      `evals/reports/p5-5-retrieval.md`; sweeps
      `evals/baselines/p5-5-lc-multiquery.json` and `p5-5-native.json`).
      - `retrieval/retriever.py`: `retrieve_from_candidates()` split out
        (shared by both stacks; snapshot identical), `expand=` for whole
        documents.
      - `lc/retrievers.py`: `ScopedStore` (per request, principal-bound;
        every read ANDs the permission filter); one contract (cosine
        re-score from stored vectors, then `retrieve_from_candidates`);
        generators: vector, BM25 (per scope set, content-fingerprint
        cache), hybrid `EnsembleRetriever`, `MultiQueryRetriever` (union
        **ordered by cosine** — its native order put the original
        question last, and authorized recall fell to 5/12), cross-encoder
        rerank, `ParentDocumentRetriever`; `TesseraRetriever` bridge.
      - `tessera eval --stack lc`: native routing/generation + LangChain
        index and `TESSERA_LC_RETRIEVER`. `query`/`chat`/`serve` still
        refuse `lc`.
      - Evidence: parity with native exact; leak test 7 retrievers × 29
        access cases (0 leaks, authorized 12/12); scope binding 6 kinds ×
        3 principals; redaction over LangChain's own retriever runs.
        Retrieval-only: vector = native; BM25 alone held-out 0.367; hybrid
        0.25 within noise; rerank worse and 40× slower; parent-doc 3×
        context. `lc-defaults` retriever = vector (`lc`).
      - Multi-query sweep `=> PASS` (1 ERROR `ql039`, re-run: recall
        0.67, G5/R5): **0 cases' recall/RR changed** vs native, latency
        2×, **rephrasing calls unmetered** (fix: P5-7 runtime-context
        recorders). Native sweep `=> PASS`, 3 ERROR rows (`ac-l08`,
        `ac-a03`, `ql014`) on an overloaded NIM evening; each re-run alone
        scored (no leak; authorized 1.00; recall 1.00 G5/R5).
      - **Review finding (Risk 1), fixed in the PR:** the BM25 cache was
        keyed on the chunk count, so a same-count relabel in one process
        served a stale index (restricted text in BM25's own run output;
        `rescore()` kept it from the answer). Now keyed on a content
        fingerprint; regression test added. No re-sweep for that commit
        (neither sweep uses BM25), stated on the PR.
      - Suite 555 passed. **Deferred to P5-7:** the `Send` fan-out
        multi-query.
- [x] **P5-6 — Generation, routing, retries, expertise** (2026-10-07…09,
      branch `feat/p5-6-lc-generation`, **PR #81**; evidence
      `evals/reports/p5-6-generation.md` + `.json`; sweeps
      `evals/baselines/p5-6-lc-{router,prompt-chain,model-client,retry}.json`
      and `p5-6-native.json`).
      - `lc/generation.py`: `ChatPromptTemplate`s with the `prompts.py`
        text; the grounded chain `RunnableParallel` →
        `RunnablePassthrough.assign(RunnableLambda(format_source_group))`
        → `RunnableBranch` → model → `StrOutputParser`; structured routing
        `with_structured_output(RouteDecision, method="json_schema")` (no
        `include_raw` — `ChatNVIDIA` raises on it; usage comes from a
        `RecorderCallback`); `with_lc_retries` (`.with_retry()` on errors
        translated to `TransientLLMError`) + `eval_rate_limiter()`.
      - `lc/chat_models.py`: `UsageCallback` (tokens **and** latency, under
        the configured model id), `LangChainLLMClient` (LangChain model
        behind the native port; errors carry `status_code`; client-side
        timeouts read as 504), `TesseraChatModel` (native port as a chat
        model). `lc/expertise.py`: `PeopleRetriever` around native
        `find_experts`.
      - Shared code changed: `generation/answer.py`
        (`citations_for`/`finish_answer`), the `UsageRecorder`;
        `generation/resilient.py`: a thread-safe `Pacer`, adaptive (AIMD,
        3–30 s) in sweeps and shared by every client.
      - `tessera eval`: `--cases ID…` (re-runs ERROR rows; refuses
        `--check`), `--workers N` (default 1; refused with `retry=lc`),
        pacing summary in the export's meta. `evals/compare_sweeps.py`:
        case-by-case and judge-mean comparison on cases scored in both,
        with `--floor-from` (max pairwise spread of native sweeps).
      - Sweeps (NIM, one switch `lc` each, retriever native): `router` 46
        min, 0 ERROR; `prompt_chain` 2 h 51, 0 ERROR; `model_client` 3 h
        39, 5 ERROR (2 `ChatNVIDIA` 60 s read timeouts → fixed `ea1dbf9`,
        `timeout=600`); `retry` 56 min, 2 ERROR; native (4 workers) 2 h
        44, 11 ERROR. All `=> PASS`; every ERROR row scored on a re-run
        alone. Against native, every judge mean is within the 7-sweep
        floor (A/C relevance ±0.03, groundedness ±0.06, B relevance ±0.12)
        **except `router` A/C relevance −0.06**: two cases (`ql017`,
        `fr005`) with the same route, documents and generator in both
        sweeps — answer sampling, not the router. The only routing
        changes are the Cobalt flip cases `ac-a06`/`ac-l06`.
      - **Retry finding:** in the same calm sweep the native judge
        recovered from 19 throttles; `.with_retry()` (≈ 31 s over six
        attempts, no `Retry-After`) lost 2 answer calls. Proposed
        `lc-defaults`: router/prompt_chain/model_client `lc`, retry
        `native`, expertise the user's call — **awaiting the user.**
      - Suite 610 passed, 8 skipped, 1 xfailed. **Deferred to P5-7 (per
        plan):** `RunnableBranch` vs conditional edges, `RetryPolicy`,
        time to first token.

## Next task to pick up

**Open questions for the user** (deferred by the user on 2026-10-09 to
the next session — ask these first, with the recommendations):

1. **`lc-defaults` for the generation switches.** The P5-6 PR merged
   with all five at `native` (the pre-P5-6 defaults); nothing changed.
   `evals/reports/p5-6-generation.md` §6 has the evidence.
   **Recommendation:** `TESSERA_LC_ROUTER=lc`,
   `TESSERA_LC_PROMPT_CHAIN=lc`, `TESSERA_LC_MODEL_CLIENT=lc` (each within
   noise; P5-5's rule picks the LangChain layer at native-equal quality);
   `TESSERA_LC_RETRY=native` (honours `Retry-After`, counted, shares the
   pacer, allows `--workers`; in the `retry` sweep `.with_retry()` lost 2
   answer calls while the native judge recovered from all 19 throttles —
   question 2's sweep firms this up); `TESSERA_LC_EXPERTISE=native` (`PeopleRetriever` only wraps
   native `find_experts`, no sweep measured it alone, and P5-7's `all-lc`
   sweep measures it anyway). Once decided: change `config.py` +
   `.env.example`, turn §6 from "Proposed" into the decision, small PR —
   this changes the `lc` stack's defaults, so it needs a `--stack lc`
   `tessera eval --check` in the PR body (P5-7's `lc-defaults` sweep can
   be that sweep if the change goes in with P5-7's first PR).
2. **Run the planned `lc-model-client-2` sweep?** The 2026-10-08 plan had
   six P5-6 sweeps: the `model_client` switch again (with the timeout
   fix and adaptive pacing), then `retry` compared with it. On 2026-10-09
   only `retry` ran (start-of-day read the feature branch's log and
   missed #80's plan on `main`), and `retry` was compared with the native
   sweep instead. The P5-6 acceptance is met without it, and #81 merged.
   **Recommendation: yes, at the start of a calm morning** (free on NIM,
   ~1 h on 10-09's load):
   `TESSERA_LC_RETRIEVER=native TESSERA_LC_MODEL_CLIENT=lc tessera eval
   --check --stack lc --workers 1 --json evals/baselines/p5-6-lc-model-client-2.json`.
   It isolates `.with_retry()` (the only difference from the `retry`
   sweep) and re-checks `model_client` with the timeout fix. Add it to
   `evals/reports/p5-6-generation.md` §3–4 in a small PR.

**Then P5-7.**

**Then P5-7 — LangGraph orchestration, the corrective subgraph, `Send`,
the Functional API** (`docs/Tessera_Phase5_Plan.md` §3.7, §5). Read §3.7
in full first. **Acceptance (plan §5, verbatim):**
- equivalence on fakes and replay is exact, with concurrent usage isolated;
- the scope-binding test is extended to the loop;
- the corrective-variant sweeps are reported;
- `lc-defaults` `=> PASS`, and `all-lc` is reported;
- the redaction test is extended to graph runs and streamed events.

P5-7 also picks up what P5-6 deferred: `RunnableBranch` vs
conditional edges, LangGraph's `RetryPolicy` (against
`RetryingLLMClient`), time to first token under streaming, the `Send`
multi-query, and runtime-context usage recorders (multi-query's
unmetered calls).

**Still open from P5-3:** the live LangSmith step (needs the user's
LangSmith key). Doesn't block.

---

**Phases 1–3 are complete** (`v0.1.0`, `v0.2.0`, `v0.3.0`); Phase 4 is in
progress per `docs/Tessera_Phase4_Plan.md` (above).

**Tuning probe on both thin margins (2026-10-01, retrieval-only, zero
LLM calls) — no principled lever; don't re-try these two:**
- **B: `CANDIDATE_K` is not the problem.** Person recall is identical
  (0.911, every case unchanged) at `CANDIDATE_K` 50/75/100/150/200 —
  every gold person is already in the pool (at k=600 the `ql019` gold
  ranks are 1, 2, 4, 16, 25). The misses are ordering among strongly
  evidenced near-neighbours (`ql041`: a sustainability principal with
  industrials projects at #2; `ql042`: a supply-chain EM outside the
  30-person label at #3). Re-weighting to move them would fit 3 of 9
  cases. The honest way to widen this margin is **more labelled B
  cases**, not tuning.
- **A/C relevance: adaptive `k` by top-hit dominance doesn't separate
  the low-relevance cases.** Top-1 − top-2 score gaps: `ql027` 0.095,
  `ql028` 0.072 (no dominant hit — adaptive `k` wouldn't trigger);
  `ql004` 0.195 (dominant, but it has 2 relevant docs — trimming would
  give back its P2-4 recall gain); `ql007` 0.208 (the only clean win).
  The gap-dominant cases that would also trigger (`ql009`/`ql026`/
  `ql030`/`ql031`/`ql032`) already score relevance 5. Best case ≈ one
  case 3→5 ≈ +0.06 on the 35-case mean — below run-to-run judge noise
  (4.57–4.75).
- Probe scripts were scratchpad one-offs: they monkeypatch
  `retrieval.expertise.CANDIDATE_K` and read `find_experts`/`retrieve`
  scores directly against the persisted index.

**Carried forward from Phase 3 (not blocking — the exit sweep passed):**
- ~~**A/C mean relevance 4.60 vs 4.5**~~ — **resolved 2026-10-02** by
  parent-document expansion (#61): relevance 4.92, `ql007`/`ql027`/
  `ql028` at 5, `ql004` at 4.
- **B person recall 0.91 vs 0.90 on 9 labelled cases** — one more miss
  fails the bar. Untried levers: `CANDIDATE_K` (pool starvation,
  `c0049` in `ql019`), a `where` filter from a place/practice named in
  the query.
- **Self-reported-only matches** have never been seen live (with 600
  people the top 5 are always evidenced); the flagging path is
  unit-tested only.
- **Resumable sweep** (persist per-case results, skip completed on
  re-run) — still the natural next eval improvement if ERROR rows keep
  costing re-runs. Not needed on 10-01's clean run.

**Phase 1 exit criteria — re-confirmed 2026-10-01 at the Phase 3 close**,
on a fresh clone of `main` on a machine that had never run Tessera
(`y520`): (1) `uv sync --extra dev` → `.env` → `tessera ingest` (336
chunks) + `tessera index-people` (600 people) → `tessera query` with
citations — **met**, live; (2) archetypes observably different — an A
query returned a 5-source cited answer, a B query returned 5 named,
evidenced experts with the snapshot date, an off-corpus query returned
the fixed refusal, and the sweep exercised A/B/C/D with 100% routing —
**met** (B is now a real answer, not a non-answer, by design since
P3-4); (3) the harness reports every metric category, A/C and B —
**met**; (4) every external dependency is behind a port —
`Embedder`/`VectorStore`/`LLMClient`/`ExpertiseStore`, plus the
parameterized `load_corpus`/`load_expertise` — **met**; (5) README
states Phase 3 as built and Phases 4–5 as documented-not-built —
**met**.

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

## Task sequence

**Phase 1 (build plan §5)** — complete, tagged `v0.1.0`:
1–8: ~~scaffold+corpus~~ (#2) · ~~ingestion+chunking~~ (#5) ·
~~embedding+store~~ (#7) · ~~router~~ (#10) · ~~archetype retrieval~~
(#13) · ~~grounded generation~~ (#16) · ~~eval harness~~ (#18/#19) ·
~~CLI+README~~ (#20).

**Phase 2 (`docs/Tessera_Phase2_Plan.md` §4)** — complete, tagged
`v0.2.0`:
P2-1 ~~quality bar~~ (#31) · P2-2 ~~eval set 33→50~~ (#32) · P2-3
~~constant grid search~~ (#33) · P2-4 ~~A-diversification + router
fix~~ (#35) · P2-5 ~~exit~~ (#36).

**Phase 3 (`docs/Tessera_Phase3_Plan.md` §5)** — adopted 2026-09-07
(#38/#39):
- ~~P3-1 — expertise dataset + seeded generator + schema~~ — done (#40)
- ~~P3-2 — `ExpertiseStore` port + local Chroma impl~~ — done (#42)
- ~~P3-3 — expertise retrieval path (`retrieval/expertise.py`)~~ — done (#44)
- ~~P3-4 — B generation (`generation/expertise.py`) + router/pipeline/CLI~~ — done (#46)
- ~~P3-5 — eval harness B metrics + bar extension + no-match set~~ — done (#48)
- ~~LLM retry/backoff + eval progress~~ — done (#50, not a plan task)
- ~~B-retrieval pass (lead/recency intent) + gate the B bar~~ — done (#51, not a plan task)
- ~~P3-6 — Phase 3 exit, tag `v0.3.0`~~ — done (#53, 2026-10-01)

**Phase 4 (`docs/Tessera_Phase4_Plan.md` §5)** — adopted 2026-10-01 (#55),
replanned + re-adopted the same day (#57):
- ~~P4-1 — HTTP API + `tessera serve`~~ — done (#56)
- P4-2 — Claude on Bedrock + model routing + cost accounting — **code
  done (#59); live Bedrock sweep (acceptance) pending AWS account access**
- ~~UI decision (plan §7)~~ — deferred until Bedrock latency is measured
- ~~P4-3 — traces + feedback-to-eval loop~~ — done (#60)
- ~~Lookup parent-document expansion (fb001)~~ — done (#61, not a plan task)
- ~~P4-4 — freshness + data-quality report~~ — done (#63)
- ~~P4-5 — restricted tier: data, walls, review gate, eval sets (leakage
  eval shown failing)~~ — done (#64)
- ~~P4-6 — permission-aware retrieval (leaks 0, authorized recall ≥ 0.80,
  injection 100%)~~ — done (#65)
- ~~P4-7 — Phase 4 exit, tag `v0.4.0`~~ — done (#66)

- ~~P4-2 live acceptance sweep~~ — done on Gemini (#69, ADR 0007)

**Phase 5** — LangChain / LangGraph / LangSmith as a parallel, measured
stack (`docs/Tessera_Phase5_Plan.md`, adopted 2026-10-05, PR #67):
- ~~P5-0 — baseline + noise floor~~ — done (#71)
- ~~P5-1 — adopt (CLAUDE.md §9 edits); dependencies; spike~~ — done (#72)
- ~~P5-2 — `Pipeline` protocol, `PipelineRun`, the marker split~~ — done (#73)
- ~~P5-3 — LangSmith tracing with taint redaction (native first)~~ — done (#75; live LangSmith step pending a key)
- ~~P5-4 — LangChain ingestion and indexing~~ — done (#77)
- ~~P5-5 — LangChain retrieval~~ — done (#78)
- ~~P5-6 — generation, routing, retries, expertise~~ — done (#81; `lc-defaults` for it pending the user)
- P5-7 — LangGraph orchestration, corrective subgraph, `Send`, Functional API **← next**
- P5-8 — human-review workflow
- P5-9 — LangSmith datasets, experiments, feedback, prompt hub
- P5-10 — comparison and exit, incl. 3 Gemini sweeps; tag `v0.5.0`

**Phase 6** — ephemeral Google Cloud deployment: Cloud Run + chat UI,
Terraform, project `tessera-510716`. **Phase 7** — CI/CD (GitHub Actions
→ GCP via WIF) + monitoring. **Then** one final deploy → record →
destroy for the LinkedIn video. CLAUDE.md carries this numbering since
P5-1.

## Notes / open flags

- **Start-of-day miss (2026-10-09):** `/start-day` read `git log` on the
  checked-out feature branch, not `main`, and missed #80 (10-08's
  end-of-day checkpoint, merged to `main` after the branch's last
  commit). The day's checkpoint was first written as if 10-08 had no
  end-day, and the P5-6 PR hit a merge conflict; fixed by merging `main`
  and rebuilding this file on #80's version. **Read `git log main` and
  `checkpoint.md` on `main`** (`git show main:checkpoint.md`) when a
  feature branch is checked out.
- **NIM spend 2026-10-09 (free tier):** the `retry` sweep (56 min, 1
  worker; the judge throttled 19×, pacer peaked at 30 s) + 2 single-case
  re-runs, ≈ 300 calls. **The calmest NIM yet** (14:09–15:05 IST).
- **Adaptive pacing, first live run** (the `retry` sweep): 19 throttled
  calls, the interval rose to its 30 s cap and ended there, 56 min for
  92 cases. It only paced the native judge (`retry=lc`'s models use their
  own limiter), so it is not yet measured on a sweep where it paces
  every call; the next native or `retry=native` sweep is that test.
- **`.with_retry()` retries are silent and uncounted**: LangChain gives no
  hook. A counter in `with_lc_retries`'s translating wrapper would fix
  the count without changing behaviour. Its backoff is tenacity's
  exponential jitter, ≈ 1+2+4+8+16 s over six attempts.
- **Per-call latency isn't in the exports.** A case's `latency_seconds`
  is wall time including pacing, backoff and worker waits.
  `Usage.latency_s` has the per-call figure plan §4.2 wants; export its
  per-case sum before P5-7's latency comparison.
- **The native noise floor underestimates answer sampling.** `router`'s
  −0.06 A/C relevance (vs `p5-6-native`) came from two re-sampled answers
  (`ql017`, `fr005`) on unchanged routes and retrieval. Read a 1–2-case
  judge swing as noise when the deterministic fields are identical.
  The report's baseline is `p5-6-native` with a 7-sweep floor (A/C
  groundedness ±0.06); 10-08's numbers below used `p5-5-native` and 6
  sweeps (±0.05). Both are recorded; the conclusions agree.
- **Cobalt flip cases** `ac-a06`/`ac-l06` route A or D at random across
  sweeps. `routing_accuracy` excludes access cases, so it stays 100%;
  `compare_sweeps` shows them as routing Δ.

- **NIM spend 2026-10-08 (free tier):** 5 full sweeps (router 46 min;
  prompt_chain 2 h 51; model_client 3 h 39; native 2 h 44 with 4
  workers), one aborted `lc-retry` run with 4 workers (11 cases), 17
  single-case re-runs and ~10 smoke calls — about 1,700 calls. Gemini: $0.
  NIM was overloaded from midday into the evening (100–300 retried
  429/503s per sweep). Start sweeps in the morning.
- **Sweep concurrency and pacing — what was measured (2026-10-08):**
  - Pipeline code is not the cost: retrieval ~64 ms against 10–30 s per
    NIM call. Sweep time follows NIM load (the same LangChain code: 46 min
    at 11:00, 2 h 51 at 13:00).
  - `--workers 4` at a fixed 3 s interval: native took 2 h 44 with **11**
    ERROR rows vs 2 h 48 and 3 sequentially the evening before. Under
    throttling the limit is NIM's real capacity, far below 40 rpm.
  - With `TESSERA_LC_RETRY=lc`, the LangChain models pace with their own
    `InMemoryRateLimiter` apart from the judge: 4 workers failed 8 of the
    first 11 cases on 429. Now refused (`--workers > 1` with `retry=lc`).
  - Fix built: one adaptive `Pacer` per sweep (AIMD, 3–30 s) shared by
    every client; the default stays 1 worker until a sweep with adaptive
    pacing shows more workers pay. **Not yet measured live.**
  - Per-case `latency_seconds` with workers > 1 includes time queued in
    the pacer: don't compare it with sequential latency.
- **`ChatNVIDIA` defaults that changed results** (add to spike items 6–7):
  a **60 s read timeout** (openai SDK: 600 s) whose `ReadTimeout` carries
  no status, so nothing retried it — fixed (`ea1dbf9`); and
  `with_structured_output(method=…)` is ignored (it picks json_schema for
  a Pydantic schema; a test proves no `tools` are sent).
  `langchain-community` now warns that it "is being sunset" (BM25,
  cross-encoder live there): count it in P5-10's maintenance column.
- **Provenance:** `--json` now reads the commit **at the start** of a
  sweep. Before `6700ece` it read it at the end, so a commit (or a tracked
  edit) during a sweep relabelled the export (`+dirty`). For work during
  a sweep, a `git worktree` keeps the main checkout untouched.
- **Shell gotchas hit today:** `git merge … | tail -1 && …` hid an
  aborted merge (untracked files in the way) and the chain pushed without
  it — use `set -o pipefail` in every chained git step. `pkill -f <path>`
  matched its own shell command and killed it — kill by PID instead.
- **Re-running a sweep's ERROR row at an older commit:** `git worktree
  add --detach ../tessera-at-<sha> <sha>`, then from that directory
  `set -a; source ../tessera/.env; set +a`,
  `TESSERA_VECTORSTORE_DIR=<main>/data/vectorstore PYTHONPATH=src python
  -c "import sys, tessera.cli as c; sys.argv=['tessera','eval',…]; c.app()"`
  (the console script imports the main checkout's `src`).
- **System Atlas** (living architecture page, outside the repo by the
  user's choice): https://claude.ai/artifact/UgSncCxiqhLMLMnJCq4G1E —
  edition 1.1. Edition 2 adds the P5-6 sweep results; later editions add
  P5-7 onward. Ask before copying it into the repo.

- **NIM spend 2026-10-07 (free tier):** four full sweeps (P5-3 native
  65 min; P5-4 native 62 min; P5-5 multi-query 95 min with ~45 extra
  rephrasing calls; P5-5 native **2 h 48 min with 214 retried 429/503s
  and 3 ERROR rows** in the evening), plus 5 single-case re-runs. About
  1,300 calls, well inside 10,000/day. Afternoon sweeps were about 1 h;
  the evening was the worst NIM load yet. Start sweeps early.
- ~~**Single-case re-run scripts**~~ — promoted 2026-10-08: `tessera eval
  --cases ID[,ID…]` (refuses `--check`; same `run_harness` path). The
  route-matched diff is `python -m evals.compare_sweeps`.
- **The test suite now takes ~11–13 min** (610 tests, 2026-10-08): real Chroma indexes
  of the corpus in several modules, and the cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB, downloaded once to the
  Hugging Face cache). Phase 7's CI must cache that model. Run targeted
  files while iterating.
- **Chroma's HNSW search is approximate, and separately built indexes
  differ** (parallel inserts). `test_the_vector_retriever_reproduces_native_retrieve`
  flaked once in the full run (`q001`: an expanded document returned
  `::5` instead of `::1`). It now reads **one** collection through both
  stores. `tests/test_lc_ingestion.py`'s store-parity test still compares
  two separate builds; it carries the 0.01 superseded tolerance but could
  flake the same way. **Fixed on `feat/p5-6-lc-generation` (`c93544e`):**
  the test builds with `search_ef` above the collection size (exact
  search), tolerance tightened to 1e-6; reaches `main` with the P5-6 PR. The 2026-10-07 full run
  before the fix: 1 failed (that test), 555 passed.
- **`--stack lc` today** = `tessera eval` only: native routing and
  generation, plus LangChain ingestion and retrieval layers per
  `TESSERA_LC_*`. It needs `tessera ingest --stack lc` first (collection
  `tessera_lc_chunks` and `lc_record_manager.sqlite` in
  `data/vectorstore/`). `query`/`chat`/`serve` refuse `lc` until P5-7.
- **Multi-query's rephrasing calls are unmetered** (made inside the
  retriever, outside the pipeline's usage recorders). P5-7's
  runtime-context recorders are the fix; until then, never quote its
  cost from the sweep's token totals.
- **Plan departures recorded in the reports, not yet in the plan text:**
  `index()` uses `cleanup="full"` (not `incremental`; P5-4), and
  LangChain retrievers report `removed["restricted"]=0` because a scoped
  store never counts outside its scope (P5-5). Fold both into
  `docs/Tessera_Phase5_Plan.md` at P5-10 (or earlier if a reviewer asks).
- **`BM25Cache.invalidate()`** exists for P5-8's review workflow. With the
  content-fingerprint key it isn't needed for correctness any more, but
  it frees memory after a review.
- **Merging:** this session `gh pr merge` worked when the user asked for
  the merge (#75, #76, #78). The 2026-10-05 note that auto mode blocks it
  may depend on how the request is phrased. Still merge only on the
  user's explicit go-ahead.
- **PR reviews:** after every `gh pr create`, post the reviewer's guide
  (`.claude/commands/pr-review.md`). When the user runs `gh pr view <n>
  --web` and the head hasn't changed, point to the guide already posted
  rather than repeating it.
- **zsh aliases clash with short shell functions** (`g`, `gl`): wrap
  multi-line helper scripts in `bash -c '…'`.
- **Future idea (user, 2026-10-07), not planned or built:** payments
  transaction data as a later "domain pack" (a possible Phase 8). The
  governance machinery carries over (ports, deny-by-default scopes,
  leakage evals, taint redaction, traces, the eval gate). The domain does
  not: a structured-query archetype (SQL/metrics, not embeddings),
  team/merchant/region scopes with row- and column-level rules, PCI and
  card-data tokenization, an audit trail of records read,
  numeric-exactness evals, and RBI data-localization checks. The user was
  offered an ADR (docs only); no decision yet.
- **A user-level kitty tab-title hook** was installed on 2026-10-07
  (`~/.claude/hooks/kitty-tab-title.sh` + `CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1`
  in `~/.claude/settings.json`; `cerberus` → `cerberus-platform`).
  Outside the repo. Not yet confirmed that the title holds for a whole
  session.

- **NIM was heavily overloaded all of 2026-10-06** (free tier, judge and
  answers). Three 92-case sweeps: 1h47 / 2h10 / 2h48, with 130 / 160 /
  216 transient 429/503s and 2 / 2 / 6 ERROR rows; failures came in
  bursts (`ql011`, `ql015`, `ql016` within minutes). `ql016` then 503'd on
  three single-case re-runs in a row. Expect 2–3 h per sweep on a bad day;
  plan sweeps early, and re-run ERROR cases alone afterwards. **Spend
  today:** 3 full sweeps (~280 calls each) + 13 single-case re-runs +
  ~10 spike calls on NIM; ~8 Gemini Flash calls (< $0.01). Well inside
  10,000/day.
- **`ql016` is unscored in P5-2** (four 503s). It passed in both P5-0
  sweeps and its path is unchanged; re-run it once NIM is calm if a clean
  P5-2 record is wanted. Not blocking.
- **ERROR rows stay in the committed exports** (user decision,
  2026-10-06, option 1): never merge re-run results into an export.
  Re-run failed cases alone with the same `run_harness` path at the same
  commit, and put their scores in the PR body. The re-run and
  route-matched-diff scripts were scratchpad one-offs (P5-0's
  `rerun_cases.py`: `cli._build_llms` + `run_harness` filtered by
  `EvalCase.id` — note the field is `id`, not `case_id`; `route_diff.py`:
  route-matched comparison of two `--json` exports on the deterministic
  fields). If they're needed every task, promote them into `evals/` (a
  `tessera eval --cases` filter and an `evals/compare.py`) rather than
  rewriting them.
- **`uv sync` is exact**: `uv sync --extra dev` *uninstalls* the `lc`
  extra. Use `uv sync --extra dev --extra lc` for Phase 5 work; drop
  `--extra lc` only to test a native-only install (`/start-day` says so
  since P5-1).
- **The CPU-only lock check is now `grep -cE '^name = "nvidia-' uv.lock`**
  (a bare `grep 'nvidia-'` counts `langchain-nvidia-ai-endpoints`). It
  exits 1 when the count is 0 — expected, so don't chain `&&` after it.
- **Compare snapshots against `evals/snapshots/p5-2.json` from P5-3 on.**
  `v0.4.0.json` predates `context_marker_hits` and needs
  `--allow-added context_marker_hits`.
- **The user reads chat replies in ASD-STE100 + ~20% ELI5** (their
  `~/.claude/CLAUDE.md`, 2026-10-06; chat only — files, commits, PR bodies
  and anything the user publishes stay in normal English).

- **Cloud is Google Cloud now (ADR 0007, 2026-10-05).**
  - Project `tessera-510716`.
  - `gcloud` 587.0.0 is installed at `~/.local/google-cloud-sdk`
    (symlinked into `~/.local/bin`, no sudo).
  - ADC is set up with the quota project. If a Gemini call fails on
    auth: `gcloud auth application-default login`, **tick "Select all"**
    on the consent page (the first try missed the cloud-platform scope),
    then `gcloud auth application-default set-quota-project tessera-510716`.
  - `.env` holds `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION=global`.
  - The console says "Agent Platform" (Vertex AI was renamed in April
    2026). The API is still `aiplatform.googleapis.com`.
- **Bedrock is dormant, not deleted.**
  - The AWS account has a UPI AutoPay mandate and an Indian debit card
    (the default). The "not available for this account" message was
    the pre-payment error, not proof that Opus is gated.
  - If AWS is ever revisited, the user's own fallback ladder was a
    credit card next.
  - Don't spend time on it unless asked.
- **Gemini quirks:**
  - `gemini-3.8-flash` rejects `thinking_level=minimal` (400).
  - At its default thinking level, Flash took 28 s on a one-word reply.
  - The answer model is a *preview*.
  - Flash's price doubles on 2027-01-01, so restate costs after that.
  - The GCP billing report hasn't been reconciled with `MODEL_PRICES`
    yet. Check it after a sweep or two.
- **Always-on cloud rejected on cost** (user, 2026-10-05: "$170+/month
  is too much"). Phases 6–7 stay ephemeral, and nothing that bills while
  idle goes in without a stated monthly cost and approval.
- **Showcase video:** recorded once, after Phase 7, showing the chat UI
  running on Cloud Run. The draft storyboard has 6 scenes: the problem;
  live on Cloud Run (lookup, expertise, D refusal, persona switch); under
  the hood (LangGraph, LangSmith trace, review interrupt); evidence (eval
  report, native-vs-LangChain comparison); ops (eval-gated PR, CI/CD
  deploy); teardown verified. The persona switcher for access control is
  the key shot. OBS for recording.
- **Merging is the user's step.** On 2026-10-05 auto mode blocked `gh pr
  merge` ("Merge Without Review"); on 2026-10-07 it ran when the user had
  asked for the merge (see the newer note above). Either way, merge only
  on the user's go-ahead, or hand them `! gh pr merge <n> --merge
  --delete-branch`. `gh pr edit` still fails (Projects classic), so
  retitle through `gh api -X PATCH`.
- **`... | tail -1` hides pytest's exit code.** On 2026-10-05 a commit
  chained after `pytest | tail -1 &&` went in with 7 failures (fixed on
  the branch before end of day). Check the summary line, or use
  `set -o pipefail`, before committing.
- **CLI tests stub `evals.harness` in `sys.modules`.** Any new harness
  function the `eval` command uses must be imported lazily inside its
  branch (as `evaluate_bar` and `report_to_dict` are), or seven CLI
  tests fail on ImportError.
- **NIM / Gemini spend 2026-10-05:** one 92-case Gemini sweep ($1.10)
  plus ~8 single Gemini calls (smoke tests, one live query, the `ql037`
  re-run). NIM was used as judge only (~90 calls). The sweep took 51 min
  with 40 transient 429/503s; one case (`ql037`) exhausted its retries
  on a NIM 503 and passed on re-run.

- **Plans get a Plan-agent review before adoption** (user practice,
  restated 2026-10-03: "Review the plan with planner review model"). Spawn
  the `Plan` subagent read-only against the draft *and the real code*;
  fold every finding in and map each one in a review section of the plan.
  Both Phase 5 reviews found real defects that a docs-only read would
  have missed (e.g. a marker check that would fail `ac-i04` forever, a
  `grep 'nvidia-'` that matches `langchain-nvidia-ai-endpoints`,
  LangChain's env-var tracing bypassing a redacting client).
- **The user wants depth over speed for the portfolio** (2026-10-03):
  "we do not have to crunch the time". Don't propose shrinkable scope by
  default; and for Phase 5 the frameworks are to be used broadly, "irrespective
  of whether it is required", to learn them — measured against native.
- **Phase 5 needs a LangSmith account + API key** (free developer tier) for
  the live steps of P5-3 and P5-9 only; everything else is built against a
  real `langsmith.Client` with mocked HTTP. Ask the user when P5-3 comes up.
- **The Phase 5 plan PR (#67) is docs-only by design.** Checkpoint updates
  go to `main` via end-day PRs, not the plan PR, so `main` never waits on
  plan adoption (fixed 2026-10-03 after the plan branch had picked up
  checkpoint edits).
- **NIM spend 2026-10-03:** 6 full sweeps (P4-4: 63 cases; P4-5: 91 and
  92; P4-6: 92 and 92; P4-7 exit: 92) plus ~12 single live calls
  (fresh-clone queries, the `ac-i03` re-run) — roughly 1,400 calls, well
  inside 10,000/day. Sweep wall time 15–34 min; 3–17 transient 429/5xx
  per sweep, all recovered by retry, zero ERROR rows.
- **The router is unstable on single named-project questions**
  (2026-10-03): "What did Project Cobalt recommend…" was routed to the D
  refusal on some sweeps (as `ac-a06` twice, then as `ac-l06`) and not
  others. Harmless for a walled user; over-blocks a cleared one
  (authorized recall 0.92 on those sweeps, still ≥ 0.80). The plan keeps
  the router unchanged; a candidate fix is a router-prompt note that a
  single named engagement is a lookup, not a comparison — needs its own
  sweep.
- **Replacement slot can be a Related Frameworks chunk** (2026-10-03):
  when a superseded doc's best chunk is its Related Frameworks section,
  the current version's best chunk is often its (identical) Related
  Frameworks too, so the answer cites the current doc by that section.
  Harmless (the doc is cited and the note points at it); fetching the
  current doc's best *non*-Related-Frameworks chunk would be cleaner.
- **Auto mode blocks writing a prompt-injection test fixture** as
  "Instruction Poisoning" — and then a follow-up command that only checked
  for the file. The user allowed it explicitly; the fixture is
  `methodology/workshop-facilitation-client-workshop-template.md`. Expect
  the same for any new injection fixture: ask first.
- **zsh: `echo ===X` fails** ("==X not found" — `=cmd` expansion). Use
  `echo "--- X"` in shell snippets.
- **Bedrock account blocked (2026-10-02)** — superseded 2026-10-05 by
  ADR 0007 (see the Bedrock-dormant note above). No `tessera` AWS profile
  or IAM user was ever created; the user's other AWS profiles
  (`novapay`, `cerberus*`) stay untouched.
- **Anthropic SDK 1.x (installed 1.11) removed `temperature`/`top_p`/
  `top_k` from `messages.create()`** — passing one is a `TypeError`
  before any request. `BedrockClient` sends temperature only to Haiku 4.5,
  via `extra_body`. `tests/test_bedrock.py`'s fake SDK enforces the real
  signature (`inspect.signature(Messages.create)`), so a dropped kwarg
  fails a test, not a sweep.
- **`config.MODEL_PRICES` are Anthropic's first-party rates**; AWS's
  Bedrock pricing page didn't list these models on 2026-10-02. Verify
  before quoting any Bedrock sweep's cost.
- **The eval judge must see sources numbered exactly as the answer prompt
  numbers them.** Answers cite one `[n]` per document (#61); when the judge
  numbered chunks instead, every citation after a multi-chunk document
  pointed at the wrong text and the bar failed (groundedness 4.47).
  `prompts.format_source_group` feeds both; a harness test pins them
  together. Any future change to source formatting must keep it shared.
- **Watch `ql011`** (C, "staffed on a DD engagement — what should I read
  first?"): relevance 5 → 3 in both sweeps after #61's per-document
  grouping, though synthesis retrieval is unchanged. C's mean relevance
  held (4.87) because `ql017` rose 3 → 5. If it stays at 3, look at how
  grouping reorders a synthesis prompt.
- **Traces and feedback are runtime data under `data/` (gitignored)** —
  they hold users' questions. `data/feedback/candidates.yaml` is the
  staging file; promoted cases go in `evals/cases/feedback.yaml` with the
  trace id and what went wrong in its header.
- **NIM sweep time varied 21–46 min on 2026-10-02** (morning 21 min;
  evening 46 min with 41 retried 429/503s). Four full sweeps + ~6 live
  queries today, all free tier, zero ERROR rows.

- **`gh pr edit` fails on this repo** (2026-10-01): it errors with
  "Projects (classic) is being deprecated … (repository.pullRequest.
  projectCards)" and exits non-zero, so anything chained after it with
  `&&` doesn't run (it skipped a merge once). Retitle through REST
  instead: `gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -f title=...`.
  `gh pr create`/`ready`/`merge` were unaffected.
- **`pkill -f '<cmd>'` can kill its own shell** when the pattern appears
  in the shell's command line (hit stopping `tessera serve`, 2026-10-01).
  Use a bracketed pattern: `pkill -f '[t]essera serve'`.
- **AWS CLI v2.37.7 installed 2026-10-01** in `~/.local/aws-cli` (binary
  `~/.local/bin/aws`), no sudo; installed ahead of the plan, then kept
  at the user's request (needed from P4-2). Not connected to any
  Tessera profile yet. `~/.aws` holds other projects' profiles
  (`novapay`, `cerberus*`) — never reuse them for Tessera. The user was
  told `~/.aws/credentials` is mode 775 (`chmod 600` advised); not
  changed by us.
- **NIM spend 2026-10-01:** one full 55-case exit sweep (~140 calls) plus
  ~15 live probe/demo calls — well inside the free tier. From P4-2 on,
  Bedrock sweeps cost real money: state the cost first (CLAUDE.md).

- **A fresh clone needs both indexes before `eval`** (hit 2026-10-01 on
  `y520`): `data/vectorstore/` is gitignored, so run `tessera ingest`
  *and* `tessera index-people` (~20 s each, zero LLM calls) — `tessera
  eval` exits if the people index is empty. `uv sync --extra dev` on a
  cold machine took ~1m40s and resolved **Python 3.14.4** (suite and
  sweep both clean on it; earlier sweeps' version wasn't recorded).
- **The session scratchpad directory may not exist yet.** A backgrounded
  `tessera eval > $SCRATCHPAD/x.out` failed instantly on 2026-10-01
  because the redirect target's directory was missing — the background
  task still reported exit 0 (the trailing `echo`/`date` succeeded), so
  check the output file, not just the task status. `mkdir -p` it first.
  No LLM calls were spent.

- **NVIDIA NIM throttles far below its documented 40 rpm and stays flaky
  (2026-09-29, again 2026-09-30) — MITIGATED by PR #50, not gone.** On
  09-29 three unpaced sweeps cascaded into instant 429s (a single probe
  also 429'd; it cleared after ~3 min, then tripped again after ~7
  calls). Since #50, `tessera eval` retries with backoff and paces 3 s
  apart, and prints progress, so a plain `tessera eval --check` now
  completes — but a 55-case sweep still takes **~1–1.5 h**, and on
  09-30 four cases exhausted all 6 attempts (503/504/429) and came back as
  ERROR rows (excluded from the means; re-run individually afterwards).
  When probing for a cooldown, send one tiny call every few minutes —
  hammering makes it worse. A retry-exhausted case is the harness's
  per-case isolation working, not a bug. A resumable sweep (persist
  per-case results, skip completed on re-run) is the natural next
  improvement if ERROR rows keep costing re-runs.
- **Retrieval-only checks can't see interactions with generation floors.**
  `evidence_score` is the scale `EXPERTISE_QUERY_FLOOR` / `PERSON_FLOOR`
  were calibrated on; any retrieval change that shifts it (an intent
  discount, a weight change) can silently turn "experts found" into "no
  obvious expert". Keep `evidence_score` intent-independent and change
  ordering through `rank_score`; re-check the floors (probe present vs
  absent topics) if the scale ever moves, and run an end-to-end sweep.

- **The expertise dataset is generated, and its committed output must
  stay in sync with `data/expertise/generate.py`.** Any change to the
  generator's parameters (seed, distributions, name pools, practice
  counts) requires re-running `python data/expertise/generate.py` and
  committing the regenerated `data/expertise/people/*.yaml`.
  `tests/test_expertise_generate.py::test_generator_output_matches_committed_dataset`
  fails loudly on drift, so this can't slip silently — but a session
  touching the generator should expect to regenerate. Also: **any change
  to the dataset that P3-3+ retrieval reads means re-indexing** (P3-2's
  `tessera index-people`) before live retrieval results can be trusted,
  the same rule the document store already has.
- **NVIDIA NIM was unusually slow the 2026-09-07 P3-1 sweep** —
  per-archetype mean latency A 106s / C 118s (roughly 2.5× the usual
  ~40s), one C case 248s, plus a 503 on `ql031`. Consistent with the
  existing latency-variance note below, just a pronounced instance; a
  50-case sweep took well over an hour. Not a code problem — let
  backgrounded sweeps run.

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
