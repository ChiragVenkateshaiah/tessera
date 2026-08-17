# 0002 — Hybrid Go/Python split for production architecture

**Status:** Proposed — Phase 4+, not built in Phase 1

## Context

Phase 1 is a single Python codebase (`src/tessera/`) behind swappable
interfaces (`Embedder`, `VectorStore`, `LLMClient` — see `CLAUDE.md`,
"Swappable ports"), run locally via a CLI. That's correct for Phase 1: no
production traffic, no concurrency requirements, nothing to route.

Moving toward a production deployment (Phase 4+) introduces problems Phase
1 doesn't have: many concurrent requests, session/auth state, streaming
responses, and eventually a path off managed serverless toward
self-hosted GPU inference (ADR 0004). This ADR decides the language split
for that production stack, ahead of building it, so the direction is
recorded before Phase 4 work starts.

## Decision

Split the production stack by concern, not by rewriting Phase 1's Python
core:

- **Python owns the AI/RAG core** — everything already scoped under
  `src/tessera/`: retrieval, archetype-aware strategy selection, the
  archetype router itself (LLM-based classification into A/B/C/D, per
  Task 4 — `src/tessera/retrieval/router.py`), grounded generation with
  citations. This does not move. Archetype classification is a semantic
  judgment call made by an LLM; it stays where Phase 1 already put it,
  in Python, behind the `LLMClient` interface.
- **Go owns the edge/transport layer** — request ingress, authentication,
  session-state hydration and persistence (DynamoDB, ADR 0003), request
  validation, and streaming the response back to the client. Go's job is
  everything that has to happen *before* and *around* the RAG pipeline
  runs, at high concurrency, with a fast and predictable latency profile.

**Explicitly not Go's job:** semantic query understanding, archetype
routing, retrieval strategy, or anything that requires an LLM call. Go
never makes an LLM-quality decision — it decides whether a request is
well-formed and who it's from, then hands off to Python for everything
that requires understanding what the request means.

## Consequences

**Positive:**
- Go's goroutine model is a strong fit for the edge tier specifically —
  many concurrent, short-lived, I/O-bound operations (auth checks, session
  reads/writes), which is a different workload shape than the RAG
  pipeline's fewer, longer, LLM-bound calls.
- Static typing at the API/session boundary catches malformed request or
  session-schema bugs at compile time, before they reach the RAG layer —
  a real and useful property, distinct from (and unrelated to) LLM
  hallucination, which is a generation-time phenomenon inside the Python
  layer and isn't something a routing tier's type system touches.
- Go is native to the tooling this architecture eventually needs anyway
  (Docker, Kubernetes, Terraform are all written in Go), which makes the
  Stage 2 migration (ADR 0004) a continuation of the same language rather
  than a second rewrite.
- Demonstrates a polyglot production pattern — a legitimate and
  increasingly common shape (Go/TS edge in front of a Python ML core) —
  which is real portfolio signal for the stated goal of this project.

**Negative — and these are genuine costs, not caveats:**
- Two toolchains to build, lint, test, and deploy (Go's `go vet`/
  `golangci-lint`/`gofmt` alongside Python's `ruff`/`mypy`/`pytest`) —
  roughly doubles the CI configuration surface (see ADR 0005).
- A network hop and a serialization boundary sit between Go and Python
  where today there is a single in-process Python call. That's added
  latency and a new failure mode (the Go→Python call itself) that Phase
  1's local pipeline doesn't have to reason about.
- Debugging a request that crosses the Go→Python boundary means
  correlating logs/traces across two runtimes — meaningfully harder than
  debugging a single-language stack trace.
- For a solo-maintained portfolio repo, this is genuinely more
  operational overhead than a Python-only equivalent would be. The
  justification here is demonstrated engineering range, not that it's the
  lowest-effort path to a working product — a single-language stack would
  ship faster and be easier to maintain alone.

## Alternatives Considered

- **Python-only stack** (FastAPI or a Python Lambda handling both routing
  and RAG). Simplest option, consistent with Phase 1's language choice
  throughout, lowest overhead for a project maintained by one person.
  This remains the right default if the goal were purely "ship a working
  pilot fastest" — it's the same shape as the Solution Design's existing
  minimal "pilot footprint" (§4). Rejected as the *production* target
  specifically because it doesn't demonstrate the polyglot infrastructure
  pattern this ADR set exists to showcase, and because Go is a genuinely
  better fit for the edge tier at real concurrency.
- **Node.js/TypeScript BFF** instead of Go. Also a legitimate cloud-native
  edge-tier choice with a mature async I/O model. Rejected in favor of Go
  because of the direct alignment with the Stage 2 Kubernetes tooling
  ecosystem (ADR 0004) — Go carries forward into that stage without a
  second language switch.
