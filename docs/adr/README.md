# Architecture Decision Records

This directory records significant architecture decisions for Tessera —
what was decided, why, and what was given up in exchange. An ADR is a
short document, not a design spec: it captures the decision and its
trade-offs at the time it was made, so a later reader (including a future
session of us) understands the reasoning without having to reconstruct it.

**Scope note:** every ADR currently in this directory describes the
**Phase 4+ / full-production target**, not Phase 1. Phase 1 is local,
Python-only, and unaffected by anything recorded here — see
`docs/Tessera_Phase1_Build_Plan.md` for what's actually being built right
now. These are documented ahead of need, the same way
`docs/Tessera_Solution_Design.md` §4–5 documents the AWS/MLOps layers
without building them in Phase 1. Status `Proposed` means "the direction
we intend to pick up," not "in progress."

## Format

Each ADR has five sections: **Status**, **Context**, **Decision**,
**Consequences** (both what improves and what it costs — an ADR that only
lists upside isn't documenting a trade-off, it's marketing), and
**Alternatives Considered**.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-hybrid-go-python-production-architecture.md) | Hybrid Go/Python split for production architecture | Proposed — Phase 4+ |
| [0003](0003-aws-serverless-foundation-stage1.md) | AWS serverless foundation (Stage 1) | Proposed — Phase 4+ |
| [0004](0004-kubernetes-evolution-stage2.md) | Kubernetes evolution path (Stage 2) | Proposed — Phase 5+ |
| [0005](0005-cicd-mlops-github-actions-terraform.md) | CI/CD and MLOps via GitHub Actions + Terraform | Proposed — Phase 5 |
