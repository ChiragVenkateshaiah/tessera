# 0003 — AWS serverless foundation (Stage 1)

**Status:** Proposed — Phase 4+, not built in Phase 1

## Context

ADR 0002 decided the language split (Go edge, Python RAG core). This ADR
makes it concrete as a request flow on AWS serverless infrastructure,
extending `docs/Tessera_Solution_Design.md` §4, which already lists the
intended AWS services (API Gateway, Lambda, Bedrock, OpenSearch
Serverless, S3, IAM, Secrets Manager) but didn't specify a language or a
step-by-step flow for the API/routing layer, nor a session-state design.
This ADR fills both gaps.

Note the Solution Design also defines a smaller **pilot footprint** — "the
minimum of the above... sized for a pilot, not a platform." That footprint
(a single Python Lambda, no Go layer) remains the lower-complexity default
if Phase 4 pilot infrastructure is actually built before this fuller
architecture is. This ADR describes the intended full-production target,
not a replacement for the pilot footprint.

## Decision

Request flow:

1. **Amazon API Gateway** is the entry point.
2. **Go Lambda** handles the request first: authentication, pulling the
   user's session/conversation history from **Amazon DynamoDB**, and
   request validation. This is edge/transport work only — see ADR 0002 for
   why archetype classification does not happen here.
3. The Go Lambda makes a synchronous internal call into the **Python
   RAG Lambda/service** — the deployed form of Phase 1's `pipeline.py`
   (route → retrieve → generate), unchanged in logic from Phase 1, now
   running as a service instead of a local CLI invocation.
4. Python streams the grounded, cited answer back; Go relays that stream
   to the client and asynchronously writes the updated session state to
   DynamoDB, so the write doesn't block the response.

The **session-schema contract between Go and DynamoDB and Python** is a
new interface this design introduces — the same swappable-ports discipline
Phase 1 already applies to `Embedder`/`VectorStore`/`LLMClient`, extended
to the session-store boundary. Both languages must agree on that schema
explicitly; it's a first-class design artifact, not an implicit contract.

## Consequences

**Positive:**
- Clean separation matching ADR 0002: Go never touches retrieval or
  generation logic; Python never touches auth or session persistence.
- DynamoDB's low-latency key-value access fits session hydration well,
  and Go's SDK/typing makes the read/write path to it fast and
  schema-checked at compile time.
- Async session persistence (after streaming starts) keeps the
  user-visible latency close to just the RAG pipeline's own latency.

**Negative:**
- Two Lambda invocations per request instead of one — worst case, two
  cold starts, and always at least one extra network hop, versus a single
  Python Lambda handling both routing and RAG.
- The session schema is a second interface to design, version, and keep
  Go and Python in sync on — a source of integration bugs that doesn't
  exist in a single-language stack.
- This is materially more infrastructure than the Solution Design's
  existing "pilot footprint" — appropriate for a full-production target,
  not for getting a pilot into someone's hands quickly.

## Alternatives Considered

- **Single Python Lambda for both routing and RAG.** Matches the
  Solution Design's existing pilot footprint exactly: lower complexity,
  no cross-language boundary, faster to ship. This is the right choice if
  Phase 4 pilot infrastructure is built before this fuller architecture —
  the two are not mutually exclusive; the pilot footprint can ship first
  and this design can replace it later without changing the Python RAG
  core at all, since that core doesn't change between the two options.
- **AWS Step Functions** orchestrating the Go and Python steps instead of
  a direct Lambda-to-Lambda call. Gains built-in retry/visibility at the
  cost of more moving parts and a less direct latency path; not chosen
  because the two-step flow here is simple enough not to need
  orchestration-level tooling.
