---
description: Close out a work session — verify, checkpoint, PR, merge, tag if a phase closed
---

End-of-day ritual for Tessera. This is the inverse of `/start-day`: that
command reads `## Status`, `## Done`, `## Next task to pick up`, and
`## Notes / open flags` from checkpoint.md plus recent git/GitHub state —
this command is what writes all of that down, so the correspondence is
"what start-day reads" ↔ "what end-day produces," not a literal
step-for-step mirror. `main` is branch-protected — nothing pushes there
directly.

## 1. Verify before touching anything

- `git status` and `git diff` — see everything changed this session.
- Run `pytest tests/ -q` — **without** `RUN_LIVE_LLM_TESTS=1` unless you
  have quota headroom today and specifically need to verify a live-LLM
  change (check checkpoint.md's Notes first: `gemini-3.6-flash` is capped
  at 20 requests/*day* on the free tier — a careless full live run can
  burn the whole day's budget in one go). Don't check in on a broken
  suite; fix it or clearly flag the failure in checkpoint.md's Notes
  before committing.

## 2. Update checkpoint.md

- Update the `## Status` paragraph (2-3 lines: what's complete, what's
  still unimplemented) — this goes stale fast and is easy to forget since
  nothing else in the file forces you back to it.
- Update `## Task sequence` — strike through the newly finished task and
  move the `← **next**` marker. Leave the PR number as a placeholder for
  now if code hasn't shipped yet this session (step 4 fills it in).
- Move finished items from `## Next task to pick up` into `## Done`, with
  enough detail that a cold read of checkpoint.md alone (no git log
  needed) explains what shipped, why, and which PR it landed in.
- Update `## Next task to pick up` — copy that task's acceptance check
  verbatim from `docs/Tessera_Phase1_Build_Plan.md` §5, so `/start-day`
  doesn't have to re-derive it from the plan. If the current task is only
  partially done, be specific about what's left, not just "continue
  Task N."
- Update the "Last updated" date.
- Add to `## Notes / open flags` — and only there, even if it surfaced
  mid-task rather than as a clean end-of-session finding — anything the
  next session needs: operational gotchas hit this session (rate limits,
  slow installs, flaky external services and their workarounds),
  ambiguities resolved and why, anything that touched the "explicitly NOT
  in Phase 1" list.
- If this session discovered something that changes a **standing**
  convention (a new design constraint, a fixed workflow bug) rather than
  just today's status, that belongs in `CLAUDE.md` too, not only
  checkpoint.md. Ask the user whether `CLAUDE.md` needs a matching update
  before moving on.

## 3. Confirm and commit

- Show the user the diff to checkpoint.md (and CLAUDE.md, if touched) and
  confirm before committing.
- Check the current branch. Branch protection blocks *pushing* to `main`,
  not committing locally — so a local commit sitting on `main` is
  possible and is exactly what this check catches. If that's happened, or
  no branch exists yet for this session's work, create one now:
  `git checkout -b <type>/<short-description>`.
- Stage relevant files (never `git add -A` blindly — review what's
  staged). Check nothing in `.env`, credentials, or other secrets is
  being committed.
- Commit with a message describing what was actually built this session,
  matching this repo's existing commit style (check `git log` if
  unsure). Do not commit unless the user has confirmed.

## 4. Ship it

This repo's actual pattern (see PRs #3, #6, #8) is **two PRs per task**:
the feature/work PR first, then a small follow-up `chore/checkpoint-taskN-done`
PR once the first has merged and the real PR number is known. If this
session was docs/config-only with no separate "task" to reference, one PR
covering both the work and the checkpoint update is fine — use judgment,
don't force two PRs where there's nothing to sequence.

**Ship the code first:**
- `git push -u origin HEAD`, then `gh pr create --fill` (or explicit
  `--title`/`--body` if `--fill` gives a weak summary). Confirm with the
  user before opening.
  - If `gh pr create` fails with a transient `503` from GitHub's GraphQL
    API (known flakiness — see checkpoint.md Notes), retry once, then
    fall back to the REST API:
    `REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)` then
    `gh api -X POST "repos/$REPO/pulls" -f title=... -f head=<branch> -f base=main -f body=...`.
- Confirm with the user, then merge with a merge commit:
  `gh pr merge --merge --delete-branch`. Do not squash or rebase — repo
  convention, see `CLAUDE.md` — Git workflow.
  - REST fallback if `gh pr merge` also 503s:
    `gh api -X PUT "repos/$REPO/pulls/<n>/merge" -f merge_method=merge`.
    **This does not delete branches** — follow up explicitly:
    `gh api -X DELETE "repos/$REPO/git/refs/heads/<branch>"` for remote,
    and `git checkout main && git pull && git branch -d <branch>`
    locally. Skipping this is how branch cruft accumulates — it has
    happened before.

**Then, with the real PR number in hand, ship the checkpoint update** (its
own branch/PR per the pattern above, or folded into the same PR if there
was no separate task PR this session):
- Fill in the real PR number(s) in checkpoint.md wherever it was left as
  a placeholder in step 2.
- Same push/PR/merge/branch-cleanup sequence as above.

**If this session's work satisfies a Phase's exit criteria** (build plan
§7 for Phase 1), tag the merge commit on `main`:
`git checkout main && git pull && git tag -a vX.Y.0 -m "..." && git push origin vX.Y.0`.
Confirm the version number and message with the user first. Most sessions
will not close a phase — skip this step unless they clearly do.

## 5. Report

Report the PR(s) merged (and tag if cut), test-suite status, and a
one-line summary of session progress against the Phase 1 task sequence —
the same shape of summary `/start-day` will read back at the start of the
next session.
