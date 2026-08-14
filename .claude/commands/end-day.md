---
description: Close out a work session — update checkpoint, commit, and push
---

End-of-day ritual for Tessera. Do these in order.

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
4. Stage relevant files (never `git add -A` blindly — review what's staged).
   Check nothing in `.env`, credentials, or other secrets is being committed.
5. Commit with a message describing what was actually built this session,
   following this repo's existing commit message style (check `git log` if
   unsure). Do not commit unless the user has confirmed.
6. Push to the remote. Confirm with the user first if this session hasn't
   pushed before, or if there's any ambiguity about which branch.
7. Report the final commit(s) pushed and a one-line summary of session
   progress against the Phase 1 task sequence.
