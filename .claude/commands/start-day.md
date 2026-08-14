---
description: Orient at the start of a work session — read checkpoint, plan, and recent history
---

Start-of-day ritual for Tessera. Do these in order, then stop and summarize
for the user before writing any code.

1. Read `checkpoint.md` at repo root — this is the source of truth for where
   the build stopped and what the next task is.
2. Read `CLAUDE.md` at repo root — reconfirm Phase 1 objective, the
   "do not build" list, and working conventions. If checkpoint.md and
   CLAUDE.md disagree about status, checkpoint.md wins (it's updated more
   often) but flag the mismatch.
3. Run `git log --oneline -15` and `git status` to see recent commits and
   any uncommitted work from a previous session. If there's uncommitted
   work, surface it — don't silently discard or continue past it.
4. Cross-check the "Next task to pick up" in checkpoint.md against
   `docs/Tessera_Phase1_Build_Plan.md` §5 — confirm the acceptance check for
   that task and re-read any earlier task's acceptance check if it's unclear
   whether it was actually satisfied.
5. Report back concisely: what was last completed, what's uncommitted (if
   anything), and what task is next with its acceptance check. Then wait for
   go-ahead — do not start the task automatically.
