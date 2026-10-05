# 0006 — Framework adoption: a parallel LangChain / LangGraph / LangSmith stack beside the native core

**Status:** Accepted — 2026-10-05, with `docs/Tessera_Phase5_Plan.md`.
Not yet built.

## Context

Tessera's query path was built without an LLM framework, on purpose.
CLAUDE.md constraint #1 puts every external dependency behind a thin port.
Constraint #6 keeps everything between "a query came in" and "here's a
grounded answer" pure: dependencies are injected, and data goes in and
out. Four phases of tuning live in those plain functions, and the eval
harness measures them directly: P2-4 diversification, parent-document
expansion, the freshness and permission filters.

LangChain, LangGraph and LangSmith are now the market's default stack for
LLM applications. The project's goal for Phase 5, set by the user, is
**thorough hands-on experience with all three**, used broadly across the
system, not only where a layer strictly needs them. The question is how
to get that depth without throwing away the thing that makes Tessera
worth learning on: a tuned, measured baseline for every layer.

A first draft of this ADR ("adapters, not core") answered a narrower
question, where the frameworks earn their place, and was superseded the
same day when the user reset the goal.

## Decision

**Build a complete second implementation of the pipeline on the
frameworks, beside the native one, and measure the two against each
other layer by layer with the same harness.**

1. **Two implementations of one `Pipeline` protocol.** `native` is
   today's code, unchanged in behaviour and proven so by a golden
   retrieval snapshot taken before any dependency or code change. `lc` is the LangChain
   stack:
   - a custom loader;
   - `MarkdownHeaderTextSplitter` / `RecursiveCharacterTextSplitter`;
   - `HuggingFaceEmbeddings`;
   - `langchain_chroma`;
   - archetype-aware retrievers, plus BM25 hybrid and multi-query
     variants;
   - `ChatPromptTemplate` and LCEL;
   - `with_structured_output` routing;
   - callback-based usage;
   - LangGraph orchestration with a corrective-retrieval loop.
2. **Every LangChain layer is switchable to its native counterpart**, so
   one component can be measured with everything else held fixed. The
   evidence is an ablation, not a single before-and-after.
3. **LangGraph also runs the human-review workflow:** an `interrupt`, plus
   a SQLite checkpointer for durable pause and resume. Agents, tool
   calling and conversation memory stay out (the user's decision; they
   remain on the do-not-build list).
4. **LangSmith observes both stacks.** The native stack is wrapped with
   `@traceable` in the composition root, never in the core. LangSmith also
   hosts datasets and experiments built from Tessera's eval cases.
   **Redaction is by taint, not by label.** Each request's restricted and
   quarantined text, engagement markers and codenames are collected, and
   any run that touched them is hidden in full, because restricted text
   sits inside rendered prompts with no metadata to key on. Tracing is
   enabled only through Tessera's redacting client, never by LangChain's
   environment variables. The access set never goes into a LangSmith
   dataset. A gated test checks the real client's outbound payloads,
   through mocked HTTP. The harness stays the quality gate (what
   Phase 7's CI runs); LangSmith is the exploration and comparison tool.
5. **The rules that held for the native core hold for the framework
   stack:**
   - LangChain objects are built in the composition roots;
   - graph nodes are pure over plain-data state;
   - dependencies and usage recorders are passed per invocation through
     LangGraph's runtime context;
   - the permission filter applies **before ranking** in every retriever.
     BM25 uses per-permission-scope indexes for exactly that reason.
   - **The import boundary is an allow-list:** only `lc/`, `review/`,
     `integrations/` and `observability/` may import the frameworks. A
     subprocess test keeps the native stack importable without them.
6. **One port change, made for correctness:** `VectorStore.delete_document()`.
   Indexing a single reviewed document, in either stack, must not leave
   stale chunks carrying old labels behind.

## Consequences

**Gains:**
- **Depth in all three frameworks**, across ingestion, retrieval,
  generation, orchestration, human-in-the-loop and observability, on a real
  system rather than a tutorial.
- **Every framework choice is backed by numbers.** The per-layer ablation
  shows what each component changed in quality, latency, cost and code.
  That is the basis for a defensible "here's when I'd use it and when I
  wouldn't".
- **The confidentiality model is exercised exactly where frameworks make
  it easy to lose:** in-memory BM25, query rewriting, outbound tracing.
- **The native core stays deployable and framework-free.** Phase 6 can ship
  whichever stack the evidence favours.

**Costs:**
- **Two pipelines to maintain** until the user picks a default. Shared
  changes need sweeps on both stacks.
- **A large phase:** ten tasks and roughly 15–20 full sweeps (free on NIM,
  but slow).
- **Dependency weight, version churn and legacy packages.** LangChain's
  packages and class locations move. The classic retrievers
  (`EnsembleRetriever`, `MultiQueryRetriever`, `ParentDocumentRetriever`)
  now sit in the maintenance-mode `langchain-classic`. They are learned
  because real codebases use them, and pinning plus an optional install
  limits the cost. The comparison records it.
- **An external service.** LangSmith receives trace data, so redaction has
  to be maintained and tested for the life of the integration.
- **Some layers have no framework counterpart:** B's evidence ranking, the
  generation floors, the deterministic superseded note. The LangChain
  stack reuses Tessera's functions there, and the comparison says so
  rather than pretending otherwise.

## Alternatives Considered

- **Adapters only** (the superseded first draft: a LangGraph orchestrator
  over native steps, two LangChain adapters, a review interrupt). Rejected
  for this phase's goal. It shows the frameworks where they are
  unavoidable, not how they work across a system, and it leaves out
  LangSmith.
- **Migrate the core onto LangChain.** Rejected. It loses the
  side-by-side measurement that makes the learning concrete, puts four
  phases of tuned behaviour back in question with nothing to compare
  against, and requires rewriting constraint #6.
- **A separate toy project for framework practice.** Rejected. Tessera's
  existing evals, confidentiality model and tuned baseline are what make
  the comparison meaningful; a toy has none of them.
- **Agents and conversation memory as the LangGraph showcase.** Kept out
  by the user's decision. Orchestration, a corrective loop, interrupts and
  checkpointing cover LangGraph's core concepts without lifting
  do-not-build items.
