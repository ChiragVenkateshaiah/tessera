# 0005 — CI/CD and MLOps via GitHub Actions + Terraform

**Status:** Proposed — Phase 5, not built in Phase 1 (explicitly on the
Phase 1 do-not-build list — see `CLAUDE.md`)

## Context

`docs/Tessera_Solution_Design.md` §5 already specifies the shape of the
MLOps layer: Terraform for infrastructure-as-code, and "a CI/CD pipeline
that runs the evaluation set as a gate — no change ships if
retrieval/answer quality regresses." It does not name a specific CI/CD
tool. This ADR names one, and accounts for the two-language build
introduced by ADR 0002.

## Decision

**GitHub Actions** is the CI/CD engine. Pipeline stages:

1. Lint and test both toolchains — `ruff`/`mypy`/`pytest` for Python,
   `golangci-lint`/`gofmt`/`go test` for Go.
2. Build Docker images for the Go edge services.
3. Run `evals/harness.py` against the current corpus/eval cases as a merge
   gate — this is the same eval-gate principle Solution Design §5 already
   commits to; this ADR just wires it to a concrete CI tool. No change
   merges if retrieval or answer-quality metrics regress.
4. `terraform plan`/`apply` for infrastructure changes.
5. Deploy.

## Consequences

**Positive:**
- Ties deploy directly to the eval gate already promised in the Solution
  Design — quality regression and shipping are structurally coupled, not
  a manual step someone can skip under deadline pressure.
- GitHub Actions is free for public repos (this repo is public), so this
  costs nothing to run continuously, unlike the Stage 2 Kubernetes/GPU
  infrastructure in ADR 0004.
- Native to where the code already lives — no separate CI system to
  provision or authenticate against.

**Negative:**
- Pipeline must maintain two language toolchains' worth of lint/test
  configuration — more CI surface than a single-language repo, consistent
  with the cost already accepted in ADR 0002.
- GitHub Actions is less natively integrated with AWS than
  CodePipeline/CodeBuild would be (credentials have to be brokered in via
  OIDC or stored secrets rather than being implicitly in-account).

## Alternatives Considered

- **AWS CodePipeline/CodeBuild.** More natively integrated with the AWS
  target infrastructure (ADR 0003/0004) — no cross-account credential
  brokering needed. Rejected because it's less visible as a portfolio
  artifact: a reviewer can browse a GitHub Actions run in the repo's own
  UI without AWS console access, which matters for this project's stated
  purpose.
- **GitLab CI.** Comparable capability to GitHub Actions; no reason to
  prefer it given the repo already lives on GitHub.
