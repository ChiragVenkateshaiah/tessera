# 0006 — Framework adoption (LangChain, LangGraph): adapters, not core

**Status:** Proposed — Phase 4.5 (`docs/Tessera_Phase4_5_Plan.md`), not
yet built. Becomes Accepted when that plan is adopted.

## Context

Tessera's query path was built without an LLM framework, on purpose.
CLAUDE.md constraint #1 puts every external dependency behind a thin port
(`Embedder`, `VectorStore`, `LLMClient`, `ExpertiseStore`,
`FeedbackStore`). Constraint #6 keeps everything between "a query came
in" and "here's a grounded answer" pure: injected dependencies, data in,
data out. That is what lets the Phase 5 Lambda, and later the ADR 0002 Go
edge, call the core with a plain function call. It has paid off: four
phases of tuning (P2-4 diversification, parent-document expansion, the
freshness and permission filters) sit in plain functions the eval
harness measures directly.

LangChain and LangGraph are now the default vocabulary for LLM
application work, and the project needs to show fluency with them. The
question this ADR answers is *where* they go. Each layer of Tessera has
already chosen its abstraction, and a framework adopted carelessly would
replace those choices with its own.

## Decision

**Adopt both, as adapters at the edges of the core, never inside it.**

1. **LangGraph is an orchestration layer.** `orchestration/langgraph.py`
   composes the same pure pipeline steps the native orchestrator does, as
   a `StateGraph` over plain-data state. Dependencies (clients, stores,
   usage recorders) are passed **per invocation** through LangGraph's
   runtime context. They are never carried in state, and never bound when
   the graph is built: a graph compiled once and serving concurrent
   requests would otherwise mix their usage. Both orchestrators sit behind
   one `Orchestrator` protocol, selected by config. "Must match native" is
   made exact: native's model responses are recorded and replayed for
   LangGraph, and the two must produce identical results. A live sweep
   then holds LangGraph to the same bar.
2. **LangGraph is also the engine for one workflow that genuinely needs
   it:** the human-review pass for quarantined documents. It pauses at an
   `interrupt` and resumes from a SQLite checkpointer, possibly days later
   in another process. Hand-rolling durable pause and resume is the kind
   of plumbing a framework should own.
3. **LangChain enters only at ports.** `LangChainLLMClient` implements
   `LLMClient` over any LangChain chat model, beside `NvidiaClient` and
   `BedrockClient`. `TesseraRetriever` exposes Tessera's retrieval, with
   its permission and freshness filters, *as* a LangChain `BaseRetriever`
   so other LangChain code can call it. Neither changes what happens
   inside retrieval or generation.
4. **An enforced import boundary, as an allow-list:** only
   `generation/langchain.py`, `orchestration/langgraph.py`,
   `integrations/` and `review/` may import `langchain*` or
   `langgraph*`; a test enforces it. A subprocess test checks that
   importing the CLI, the API or the pipeline loads neither framework, so
   the frameworks can stay an optional install.
5. **One port change, made for correctness rather than for the
   framework:** `VectorStore.delete_document()`. Indexing a single
   reviewed document must not leave stale chunks carrying old labels
   behind; Chroma only upserts, and chunk ids are positional.

## Consequences

**Gains:**
- The project shows LangGraph and LangChain where each earns its place,
  with eval evidence rather than claims: two orchestrators, one bar, a
  per-case diff.
- The review workflow gets durable human-in-the-loop pause and resume
  without custom state-machine and persistence code.
- Tessera's permission-aware retrieval becomes reusable from any
  LangChain application, and the leak guarantee is tested through that
  interface.
- The core stays framework-free, so Phase 5 can deploy either
  orchestrator, and dropping a framework later would mean deleting
  adapters, not rewriting the core.

**Costs:**
- **Two ways to run the same pipeline.** Every orchestrator-affecting
  change needs sweeps on both until the user picks a default (plan
  §10.1).
- **Dependency weight and version churn.** LangChain's APIs move quickly;
  pinned versions and thin adapters limit the blast radius, but don't
  remove it. The weight also lands in the Phase 5 container image.
- **A refactor of the most load-bearing code.** Shared pipeline steps mean
  touching `pipeline.py` and the harness. That is why a golden retrieval
  snapshot (forced archetypes, zero LLM calls) is taken at `v0.4.0` first,
  and the refactor must reproduce it byte for byte.
- **Some framework features are given up on purpose:** LCEL chains,
  LangChain's retriever and vector-store integrations, the prebuilt
  agents, and LangSmith-centred tracing. Each would duplicate something
  Tessera already has and has tuned.

## Alternatives Considered

- **Rewrite the pipeline on LangChain** (its Chroma integration, its
  retrievers, LCEL chains). Rejected. It would put four phases of tuned
  retrieval behaviour back in question for no quality gain, tie the core
  to a fast-moving framework, and break the pure-function boundary
  constraint #6 exists for.
- **Stay framework-free.** Rejected for this project. It would be
  defensible engineering, but it leaves an expected skill undemonstrated,
  and the review workflow's pause and resume would need hand-built
  persistence.
- **LangGraph only as a wrapper around `answer_query()`** (a one-node
  graph). Rejected as ceremony. It demonstrates nothing a function call
  doesn't, and its routing wouldn't be visible in the graph.
- **An agent (a ReAct loop with tools) as the LangGraph showcase.**
  Deferred. Nothing in the four archetypes calls for autonomous tool use
  yet, and an agent that exists only to exist would be harder to defend
  than the review workflow, which implements a requirement Discovery §5
  already wrote down.
