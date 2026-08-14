---
description: Close out a work session — update checkpoint, PR, merge, tag if a phase closed
---

End-of-day ritual for Tessera. `main` is branch-protected — nothing pushes
there directly. Do these in order.

1. Run `git status` and `git diff` to see everything changed this session.
2. Update `checkpoint.md`:
   - Move any newly finished items from "Next task" into "Done."
   - Update "Next task to pick up" to reflect the actual next task (or the
     remaining part of the current task if it's partially done — be
     specific about what's left, not just "continue Task N").
   - Update the "Last updated" date.
   - Add anything to "Notes / open flags" that the next session needs to
     know — ambiguities hit, decisions made and why, anything that touched
     the "explicitly NOT in Phase 1" list.
3. Show the user the diff to checkpoint.md and confirm before committing.
4. Check the current branch. If work happened directly on `main` (shouldn't,
   but check), or no branch was created yet for this session's work, create
   one now: `git checkout -b <type>/<short-description>`.
5. Stage relevant files (never `git add -A` blindly — review what's staged).
   Check nothing in `.env`, credentials, or other secrets is being committed.
6. Commit with a message describing what was actually built this session,
   following this repo's existing commit message style (check `git log` if
   unsure). Do not commit unless the user has confirmed.
7. Push the branch and open a PR: `gh pr create --fill` (or with an explicit
   title/body if `--fill` doesn't produce a good summary). Confirm with the
   user before opening.
8. Confirm with the user, then merge with a merge commit —
   `gh pr merge --merge --delete-branch`. Do not squash or rebase (repo
   convention, see `CLAUDE.md` — Git workflow).
9. If this session's work satisfies a Phase's exit criteria (build plan §7
   for Phase 1), tag the merge commit on `main`:
   `git checkout main && git pull && git tag -a vX.Y.0 -m "..." && git push --tags`.
   Confirm the version number and message with the user first. Most sessions
   will not close a phase — skip this step unless they clearly do.
10. Report the PR merged (and tag if cut) and a one-line summary of session
    progress against the Phase 1 task sequence.
