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

**Meridian Advisory, its stakeholders (including "Priya," the discovery
findings' named owner of the query-log deliverable), and the engagement
itself are fictional** — this is a solo-maintained portfolio project
simulating a real consulting engagement, not an actual client relationship
(confirmed directly by the user, 2026-08-27). The scenario is realistic by
design and worth maintaining in character for the corpus, discovery
findings, and build-plan documents — but it means anything gated on real
external input (the query log Discovery Findings §7/§10 describes, for
one) will never literally arrive. Where the build plan or checkpoint.md
frames something as "waiting on the client," read that as "this input
needs to be synthesized, deliberately and transparently, rather than
waited for" — see `evals/cases/query_log.yaml`'s header comment and
checkpoint.md's 2026-08-27 entry for how that was handled the one time it
came up.

Full reasoning behind these constraints lives in `docs/`:
- `docs/Tessera_Discovery_Findings.md` — problem context, the four query
  archetypes, the confidentiality model.
- `docs/Tessera_Solution_Design.md` — full architecture, including the
  AWS/production target this phase is deliberately not building yet.
- `docs/Tessera_Phase1_Build_Plan.md` — the authoritative instruction set for
  Phase 1 (complete, tagged `v0.1.0`). Re-read it before making structural
  decisions about the retrieval core.
- `docs/Tessera_Phase2_Plan.md` — the authoritative brief for Phase 2
  (complete, tagged `v0.2.0`): eval set populated + tuned, a documented
  internal quality bar. `evals/QUALITY_BAR.md` carries the bar itself.
- `docs/Tessera_Phase3_Plan.md` — the authoritative brief for Phase 3
  (complete, tagged `v0.3.0`): archetype B (expertise-finding) built end
  to end over a synthesized firm expertise dataset.
- `docs/Tessera_Phase4_Plan.md` — the authoritative brief for Phase 4
  (complete, tagged `v0.4.0`): four production-readiness
  features aimed at the documented reasons GenAI projects stall after
  proof of concept — a cloud LLM with cost accounting (Gemini on
  Google Cloud's Agent Platform since ADR 0007; the plan says Claude on
  Bedrock), request
  traces + a feedback-to-eval loop, document freshness + a data-quality
  report, and permission-aware retrieval with a gated leakage eval. Its
  §9 (the ephemeral deployment) is renumbered: now Phase 6, re-scoped by
  ADR 0007.
- `docs/Tessera_Phase5_Plan.md` — the authoritative brief for Phase 5 (the
  current phase, adopted 2026-10-05): LangChain, LangGraph and LangSmith
  built as a **parallel, measured stack** beside the native core, which
  stays the default and the baseline (ADR 0006). §5 is the task sequence.
  Phase 6 = an **ephemeral** Google Cloud deployment with the chat UI
  (deploy → record a demo → destroy, verified); Phase 7 = CI/CD with the
  eval gate + monitoring.
- `docs/adr/` — forward-looking architecture decisions for Phase 4+
  (e.g. the hybrid Go/Python production split). ADR 0006 (accepted) is
  the parallel framework stack Phase 5 builds; the rest is documentation
  only, none of it built in Phases 1–5. Phase 6 deploys the Solution Design's
  minimal "pilot footprint" (one container service), not the ADR
  0002/0003 Go edge + DynamoDB design. **ADR 0007 (2026-10-05) moved the
  cloud from AWS to Google Cloud and the LLM from Claude on Bedrock to
  Gemini on Agent Platform** — AWS couldn't take payment from an
  Indian-issued card, and Claude on GCP had zero partner-model quota.
  Where docs written before it say AWS/Bedrock/Lambda, read GCP/Gemini/
  Cloud Run.

## Phase 1 objective and boundaries

**Objective:** a working, locally-run ingestion and retrieval core over the
pilot corpus, plus an empty-but-functional evaluation harness — provable on
placeholder queries, ready to be populated with the real query log when it
arrives.

**In scope:** synthetic pilot corpus, ingestion + section-aware chunking,
embedding + local vector store, retrieval for archetype A (lookup) and C
(synthesis), grounded generation with mandatory citations, eval harness
scaffold (runnable, metrics implemented, cases empty), CLI for smoke-testing.

**Explicitly NOT in Phases 1–5 — do not build these:**
- Archetype D (comparative) — out of pilot scope by design (confidentiality).
  Implement only as a refusal guardrail — never actually attempt it.
- Real HR-system integration (Workday, an SSO directory, an SFDC
  people-index) — Phase 7+. Phase 3's `ExpertiseStore` is a local
  implementation behind the port, the same as `ChromaVectorStore` is for
  documents.
- Expertise staleness / live-sync mechanism — Phase 7+. Phase 3 ships a
  dated static snapshot whose age the answer surfaces.
- Any cloud deployment — Phase 6. Phases 4–5 call the cloud LLM from the
  local machine and provision nothing. Phase 6's stack (Terraform, one
  Cloud Run service, Artifact Registry, in the dedicated GCP project
  `tessera-510716`; ADR 0007) is ephemeral by design: deployed for a demo
  recording, then destroyed and the teardown verified — never left
  running between sessions. An always-on deployment was rejected on
  cost (user, 2026-10-05). CI/CD and monitoring are Phase 7+.
- Agents and tool calling (user, 2026-10-03) — no `create_agent`,
  `langgraph.prebuilt`, `ToolNode` or `add_messages` anywhere in `src/`
  (Phase 5 adds an import test). LangGraph is used for orchestration,
  conditional routing, a bounded loop without tools, and interrupts for
  the review workflow. Structured-output routing prefers JSON-schema /
  guided-JSON mode; if a model needs forced function calling for it, that
  is a schema constraint, not agentic tool use.
- LLM caching in any gating sweep — `set_llm_cache` is never used there,
  so every bar check measures live calls.
- The Go edge layer, DynamoDB session state, Kubernetes (ADRs
  0002–0004), OpenSearch Serverless, Bedrock Titan embeddings, an
  S3-hosted corpus — documented AWS direction (superseded as the cloud
  target by ADR 0007), not built.
- Conversation memory / multi-turn context — each question is answered
  independently (CLI and UI alike). The LangGraph query graph is
  compiled **without** a checkpointer; only the review workflow
  (`tessera corpus review`) checkpoints, to pause and resume a review.
- PowerPoint/deck ingestion — not in the pilot corpus.
- Real authentication / SSO / identity — Phase 7+. Phase 4's "ask as
  person X" (`--as`, an API field) is a **demo identity**, and every place
  that accepts it says so.
- Automated detection of anonymized-but-identifiable content — Discovery
  §4 says automated detection must not be presented as a solution.
  Phase 4 gates such material behind a human review flag instead.
- A chat UI — **decided 2026-10-05 (user): yes, as a cloud-native app.**
  A chat page served by the same FastAPI app (`api.py`, no Node
  toolchain), deployed with the API on Cloud Run in the ephemeral stack,
  and shown running on GCP in the final video. It includes the persona
  switcher that shows access control (demo identities, labelled as
  such). Built in Phase 6, not before.

**Access-control enforcement** was on this list through Phase 3 (the
pilot corpus was low-sensitivity by construction — "sidesteps the
confidentiality problem, doesn't solve it"). Phase 4 builds
permission-aware retrieval over a synthetic restricted tier: ethical
walls per engagement, deny-by-default, filtered before ranking, proven by
a gated leakage eval — `docs/Tessera_Phase4_Plan.md` §3.5.

If a task seems to require something on this list, stop and flag it rather
than building it.

**Archetype B (expertise-finding)** was on this list through Phases 1–2
("blocked on unknown HR data structure; return 'not yet supported'"). It
is the subject of Phase 3 — `docs/Tessera_Phase3_Plan.md`. The HR data
structure is synthesized (deliberately, transparently), the same way the
pilot corpus and query log were.

## Design constraints that shape the code

These are reasons, not preferences:

1. **Swappable ports.** Phases 4–6 start moving this to the cloud (Gemini on Agent Platform now, per ADR 0007; the original target was AWS).
   Every external dependency — embedding model, vector store, LLM client,
   document source — sits behind a thin interface so the swap is a config
   change, not a rewrite. The single most important structural decision in
   Phase 1. A port change is named as one (e.g. Phase 5's
   `VectorStore.delete_document()`, for delete-then-add indexing).
2. **Grounded generation only.** Every claim in an answer must trace to a
   retrieved chunk. The system says "we don't have anything on that" rather
   than fabricating.
3. **Archetypes are first-class.** Retrieval behaviour differs by archetype
   (lookup: narrow, precise; synthesis: broad, multi-source). Do not collapse
   them into one pipeline. Archetype B (expertise-finding, Phase 3) is a
   third distinct retrieval path — structured-evidence ranking over the
   expertise dataset via `ExpertiseStore`, not document retrieval, and not
   a refusal.
4. **Evals are infrastructure, not an afterthought.** The harness is built now
   even though real test cases arrive later, because it becomes the CI gate
   in Phase 7.
5. **Local-first.** Local stays the development and evaluation
   environment; the only cloud dependency on the query path is the LLM
   call (NVIDIA NIM by default, Gemini on Agent Platform when selected
   by config). Cloud hosting is used only for Phase 6's ephemeral demo
   deployment.
6. **The query path stays transport-agnostic.** `router.py`, `retriever.py`,
   `generation/`, and `pipeline.py` — everything between "a query came in"
   and "here's a grounded answer" — must be pure with respect to
   infrastructure: given a query and whatever typed context it needs, they
   return data (a string, a dataclass, a dict), never touching HTTP, env
   vars, a session store, or any specific deployment target. No print/log
   side effects standing in for a return value, no reaching into global
   config for something that should be a parameter. This is a sibling of
   "swappable ports" (#1), not a separate idea: it's what lets a future
   edge/routing layer (see `docs/adr/0002-hybrid-go-python-production-architecture.md`)
   call into this core via a plain function call, subprocess, or HTTP
   wrapper without the core itself changing. `chunker.py` already holds this
   line (verified in Task 2 review) — hold every later module to the same
   bar as it's built.

   Ingestion code (`loader.py`) is exempt from the no-I/O part of this rule
   — reading the corpus off disk is its job — but stays parameterized like
   everything else: no hardcoded paths or credentials, ever. The same goes
   for the composition roots, `cli.py` and (Phase 4) `api.py`: they may
   read config and speak a transport (terminal, HTTP), and they build the
   dependencies the core receives — the core never reaches for them.
   Phase 4 holds the same line for identity and telemetry: the asking
   person arrives as a typed `Principal` parameter (never read from a
   session or global), and traces and token usage come back as data
   that only the composition roots write out.

   Phase 5 holds it for the frameworks (`docs/Tessera_Phase5_Plan.md`
   §3.2.2):
   - **An allow-list import boundary.** Only `tessera/lc/`,
     `tessera/review/`, `tessera/integrations/`, `tessera/observability/`
     and `evals/langsmith_sync.py` may import `langchain*`, `langgraph*`
     or `langsmith`. A test walks the tree, and a subprocess test proves a
     native-only run of `pipeline`, `cli` and `api` loads none of them.
   - **LangChain objects are built by lazily imported factories in
     `integrations/`**, which the composition roots call only when the
     LangChain stack or LangSmith is selected.
   - **Graph nodes are pure over plain-data state.** Dependencies and
     usage recorders arrive per invocation through LangGraph's runtime
     context, never in state and never bound when the graph is built.
   - **`@traceable` wraps calls at the composition roots**, not inside the
     core.
   - **`review/` is exempt from the no-I/O rule** the way `loader.py` is:
     the review workflow reads and writes the corpus, the audit log and
     its checkpoint database, all through parameters.

## Technology decisions

| Concern | Phase 1 choice | Phase 4 target | Notes |
|---|---|---|---|
| Language | Python 3.11+ | same | |
| Env / deps | `uv` (or venv + pip) | same | Lockfile committed |
| Embeddings | `sentence-transformers` local model | Bedrock Titan / Cohere | Behind `Embedder` interface |
| Vector store | Chroma (local, persistent) | OpenSearch Serverless | Behind `VectorStore` interface |
| Expertise store (Phase 3) | Chroma collection, separate from documents | Real people-index / HR API | Behind `ExpertiseStore` interface; reuses the `Embedder` port |
| LLM | NVIDIA NIM API (`nemotron-3-ultra-550b-a55b`) | Gemini on Agent Platform (ADR 0007) | Behind `LLMClient` interface; free NIM API key, 40 rpm / 10,000 req/day (swapped from the Gemini API's 20/day free tier, which was blocking eval-harness sweeps). Phase 4 adds config-selected cloud providers with token/cost accounting: `gemini` (`gemini-3.8-flash` routes, `gemini-3.1-pro-preview` answers, ADC auth) and `bedrock` (built, dormant — the AWS account can't pay). NIM stays the eval judge |
| HTTP / UI (Phase 4) | FastAPI, `tessera serve` (P4-1); the chat page is built in Phase 6 | same, on Cloud Run (Phase 6) | `api.py` is a composition root like `cli.py`; no Node toolchain |
| Feedback (Phase 4) | `FeedbackStore` port, local JSONL | a managed store | Feedback becomes *candidate* eval cases for human labelling, never auto-added |
| Access data (Phase 4) | `data/access/walls.yaml` (seeded generator, synthetic) | a real entitlement / ethical-wall system | Deny-by-default; filtered in the store query, before ranking |
| LangChain stack (Phase 5) | `src/tessera/lc/`, a second `Pipeline` behind per-layer switches (`TESSERA_LC_<SWITCH>`): loader, splitter, embeddings, `langchain_chroma`, retrievers (+ BM25 hybrid, multi-query, rerank), structured-output router, LCEL chains, `ChatNVIDIA` / `ChatGoogleGenerativeAI`, retries, LangGraph orchestration, corrective loop | the stack the user picks at P5-10 | Optional extra: `uv sync --extra lc` (user, 2026-10-06). `native` stays the default until the comparison decides |
| LangGraph review (Phase 5) | `tessera corpus review`: `interrupt()` + SQLite checkpointer | same | Demo-identity reviewer; checkpoint state holds JSON types only |
| LangSmith (Phase 5) | Tracing for both stacks with taint-based redaction; datasets, experiments, feedback, annotation queues, prompt hub | same | Free developer tier; tracing only through Tessera's redacting client, never from env vars |
| Packaging (Phase 6) | Container image (indexes + embedding model baked in) | same | Runs locally via `docker run` and on Cloud Run |
| IaC (Phase 6) | Terraform (`infra/`), ephemeral stack | same + CI/CD in Phase 7 | Labelled `project=tessera`, `ephemeral=true`, in the dedicated GCP project; `terraform destroy` after every demo |
| Config | `pydantic-settings` + `.env` | same + Secret Manager | No hardcoded values |
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
- When a decision is ambiguous, prefer the option that keeps the cloud
  migration cheap.
- Work task by task per the current phase's plan — `docs/Tessera_Phase1_Build_Plan.md`
  §5 for Phase 1, `docs/Tessera_Phase2_Plan.md` §4 for Phase 2,
  `docs/Tessera_Phase3_Plan.md` §5 for Phase 3, `docs/Tessera_Phase4_Plan.md`
  §5 for Phase 4, `docs/Tessera_Phase5_Plan.md` §5 for Phase 5. Stop after
  each task and report against its acceptance check before continuing.
- **Phase plans get an independent review before adoption.** Run a
  read-only `Plan` subagent against the draft *and the real code*. Fold
  in every finding, and add a review section to the plan mapping each
  finding to its fix. The Phase 5 reviews found real defects a docs-only
  read would have missed.
- **Portfolio depth over speed** (user, 2026-10-03). The thorough,
  evidenced version is the default. Don't propose shrinkable scope or
  time-boxed shortcuts to save time; still flag cost (cloud LLM spend) and
  anything on the do-not-build list.
- See `checkpoint.md` at repo root for where the build currently stands and
  what the next task is.
- **Quality-bar regression check (Phase 2+).** Any PR that touches
  `retrieval/` (`retriever.py`, `router.py`, `expertise.py`), `chunker.py`,
  `generation/` (including any prompt string), `pipeline.py`, the corpus
  (`data/corpus/`), the access data (`data/access/`), the expertise
  dataset or its generator (`data/expertise/`), the eval set
  (`evals/cases/`), or (Phase 5) `lc/`, `review/`, `integrations/` or
  `observability/` must run a fresh `tessera eval --check` and paste the
  report — including its final
  `=> PASS/FAIL` line — into the PR body. A gated-threshold failure blocks
  the merge. The bar lives in `evals/QUALITY_BAR.md` /
  `evals.harness.QualityBar`. A PR that changes code both stacks share
  pastes a sweep for each stack.
- **Sweep ERROR rows (user, 2026-10-06).** A case that exhausts its
  retries (an NVIDIA NIM 503, most often) stays an `ERROR` row in the
  committed `--json` export — re-run results are never merged into an
  export. Re-run the failed cases alone through the same `run_harness`
  path at the same commit, put their scores in the PR body, and compare
  stacks or sweeps only on cases scored in both.
- **LangSmith (Phase 5+).** Tracing is enabled only by Tessera's config,
  through its redacting client passed explicitly
  (`tracing_context(client=…)`); `LANGSMITH_TRACING` / `LANGCHAIN_TRACING_V2`
  in the environment are refused, not honoured. Redaction is never
  bypassed, not even for debugging.

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

**CI/CD:** intentionally not set up in Phases 1–6 (do-not-build list). It
earns its place at Phase 7, gated by the eval harness built in Task 7 —
no change ships if retrieval/answer quality regresses (Solution
Design §5). Until then, quality gating is manual: run `pytest` and
`tessera eval --check` locally before opening a PR, and paste the
bar-check result into the PR body for any retrieval/prompt change (see
"Working conventions" above and `evals/QUALITY_BAR.md`).

**Cloud LLM spend (Phase 4+):** a sweep with answers on Gemini (or
Bedrock) costs real money — state the expected cost before starting one,
and keep the judge on NIM (free) unless the user decides otherwise.

**Cloud (Phase 6, Google Cloud per ADR 0007):** every cloud change goes through Terraform in `infra/`,
reviewed as a `terraform plan` before any `apply`. `apply` and `destroy`
are run only with the user's explicit go-ahead for that specific run —
they create and delete billed resources. After a demo, `terraform destroy`
and verify nothing labelled `project=tessera` remains in project
`tessera-510716`. `/end-day` checks for live resources whenever a deploy
happened that session.
