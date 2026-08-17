# 0001 — Record architecture decisions

**Status:** Accepted

## Context

Tessera's build plan and solution design documents (`docs/Tessera_Phase1_Build_Plan.md`,
`docs/Tessera_Solution_Design.md`) already capture Phase 1 scope and the
intended AWS/MLOps target at a narrative level. As the project grows a
production-oriented direction beyond the pilot — specifically, a decision
to split the eventual stack across Go and Python — those decisions need a
format that survives independent of the narrative documents: one file per
decision, with the trade-offs made explicit, so a later reader can see not
just what was chosen but what was given up.

## Decision

Use lightweight architecture decision records (ADRs) in `docs/adr/`,
numbered sequentially, one decision per file. Each ADR has: Status,
Context, Decision, Consequences (positive and negative), Alternatives
Considered. New ADRs are added, not edited in place, once accepted — if a
decision is later reversed, a new ADR supersedes the old one and the old
one's status changes to `Superseded by 000X`, so the history stays
legible.

## Consequences

**Positive:** decisions and their trade-offs are discoverable without
re-reading the full solution design; a future phase (or a hiring manager
reading the repo) can see the reasoning, not just the outcome.

**Negative:** another document type to keep from going stale — an ADR
that no longer reflects reality is worse than no ADR, since it actively
misleads. Discipline required: update status (`Superseded`, `Deprecated`)
when a later decision changes course, rather than leaving it silently
wrong.

## Alternatives Considered

- **No ADRs, decisions live only in `Tessera_Solution_Design.md`.**
  Simpler, one fewer directory — but that document is already a narrative
  client-facing artifact, not built for per-decision trade-off tracking,
  and editing it in place loses the history of what was reconsidered and
  why.
