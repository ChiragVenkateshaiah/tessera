# 0004 — Kubernetes evolution path (Stage 2)

**Status:** Proposed — Phase 5+, not built in Phase 1

## Context

ADR 0003 establishes an AWS serverless Stage 1. Serverless has real
ceilings: Lambda execution-duration and concurrency limits, no
self-hosted GPU inference (Bedrock only), and a cost model that stops
being favorable at sustained high volume. This ADR records the intended
migration path if/when those ceilings are actually hit — it is not a
plan to run Kubernetes as standing infrastructure for a portfolio project.

**Concrete triggers** for actually starting this migration (not
hypothetical — these are the conditions that would justify the added
operational cost below):
- Sustained inference volume where self-hosted GPU cost undercuts Bedrock
  at that volume.
- A need for a specific open-weight model not available via Bedrock.
- Serverless execution-duration/concurrency limits actively blocking a
  requirement (e.g. a synthesis query archetype that needs longer than
  Lambda's max duration).

Absent one of these, staying on Stage 1 is the correct choice — this ADR
exists so the path is designed, not so it's assumed to be the next step
by default.

## Decision

When triggered: the Go edge services (ADR 0002/0003) containerize into
Docker images and deploy as Kubernetes-native services, acting as
ingress-adjacent routing/BFF microservices in front of GPU-backed pods.
Those pods run the Python RAG core plus a self-hosted inference server
(e.g. vLLM) for open-weight models, or continue calling Bedrock from
inside the cluster where Claude is still the right model choice — this
is not an all-or-nothing switch away from Bedrock.

## Consequences

**Positive:**
- Continues the Go investment from Stage 1 rather than a second rewrite —
  the same language now runs as containers instead of Lambdas.
- Unlocks self-hosted GPU inference and open-weight models, which
  Bedrock-only Stage 1 cannot offer.
- Kubernetes-native demonstrates a materially different (and highly
  valued) skill set than serverless alone.

**Negative — this is a substantial operational step-up, not a formality:**
- Cluster provisioning, node/GPU autoscaling, and observability
  (metrics/logs/traces across the cluster) all become first-class
  problems that simply don't exist in serverless.
- GPU pods are expensive when idle; without careful scale-to-zero or
  scheduled scaling, this is the single most expensive line item in the
  entire architecture, by a wide margin.
- This is meaningfully more than a portfolio project needs to run
  continuously. Treat this as an **architecture-readiness exercise** —
  designed, and possibly stood up once for demonstration — not
  infrastructure this repo commits to running indefinitely.

## Alternatives Considered

- **ECS/Fargate instead of full Kubernetes.** Materially lower
  operational overhead (no cluster/node management), still containerized,
  still demonstrates production container orchestration and service
  design. This is the pragmatic middle ground a team without dedicated
  platform engineers would likely choose in practice. Not selected as the
  primary path specifically because Kubernetes is the more
  portfolio-visible, broadly-expected skill signal for this project's
  stated purpose — but ECS/Fargate is the honest recommendation if the
  goal were purely operational simplicity, and worth revisiting if the
  Kubernetes overhead above proves not worth it in practice.
- **Stay serverless indefinitely, Bedrock-only.** Simplest, zero
  additional operational surface — but forecloses the open-weight-model
  and cost-at-scale story entirely. Correct default until one of the
  triggers above is actually met.
