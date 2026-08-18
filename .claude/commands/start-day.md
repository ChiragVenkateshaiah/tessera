---
description: Orient at the start of a work session — pick up exactly where the previous session left off
---

Start-of-day ritual for Tessera. Do these in order, then stop and report —
do not start any task automatically. `/end-day` is what writes the state
this command reads; if something here seems missing, check whether
`/end-day` actually wrote it down last time before assuming drift.

## 1. Read status

- `checkpoint.md` at repo root — the source of truth. It has six
  sections: `## Status`, `## Done`, `## Next task to pick up`,
  `## Task sequence`, `## Notes / open flags`, `## Architecture & QA
  notes`. Read `## Notes / open flags` in full — that's where operational
  gotchas live (rate limits, slow installs, flaky externals, deliberate
  design choices that look like bugs at a glance). Read
  `## Architecture & QA notes` too if the next task is 6, 7, or 8, or if
  the previous session ended with an unresolved BLOCKED/escalated finding
  there. Also skim the most recent one or two entries under `## Done` —
  some gotchas discovered mid-task ended up recorded there historically
  before this section got normalized; if `/end-day`
  is doing its job going forward, new ones land in Notes.
- `CLAUDE.md` at repo root — Phase 1 objective, the do-not-build list,
  design constraints, git workflow, tech decisions. CLAUDE.md carries no
  status itself (it explicitly defers to checkpoint.md for that) — but if
  a checkpoint note seems to contradict a CLAUDE.md *convention*, flag it,
  since that may mean CLAUDE.md needs an update, not just a passing note.
- If the next task touches anything AWS/Go/Kubernetes-shaped, skim the
  relevant file under `docs/adr/` first. That's Phase 4+ direction —
  useful context, not something to build now.

## 2. Confirm the environment is actually usable

- `.venv` exists? `source .venv/bin/activate`. If missing, flag it before
  continuing — `uv sync --extra dev` rebuilds it. This should be fast:
  torch is pinned to the CPU-only wheel index in `pyproject.toml`. If a
  sync is slow or pulls a lot, that pin may have regressed — check with
  `grep -c 'nvidia-' uv.lock` (should be `0`).
- `.env` exists with `GEMINI_API_KEY` filled in? (Gitignored, never in
  git — check locally only.) Nothing auto-loads it yet — `config.py` is
  still a stub, and `GeminiClient` takes `api_key` as a constructor
  parameter rather than reading the environment itself (CLAUDE.md
  constraint #6). A populated `.env` alone does nothing; live LLM calls
  need the vars exported first: `set -a; source .env; set +a`.
- Run `pytest tests/ -q` — **without** `RUN_LIVE_LLM_TESTS=1` (see the
  quota note in checkpoint.md's Notes for why). Confirms the repo is in a
  known-good state before touching anything. Expect something in the
  neighborhood of "N passed, 8 skipped" in well under a minute — the 8
  skipped are the opt-in live-LLM tests; that's normal, not a problem. A
  genuine failure means something's broken from a previous session or
  environment drift — but check checkpoint.md's Notes first in case the
  last session already flagged a known-failing test, rather than assuming
  drift.

## 3. Check git/GitHub state

- `git fetch --prune` first — a stale local view of `origin` (deleted
  remote branches still showing as `remotes/origin/...` locally) will
  make step 3 report ghosts.
- `git log --oneline -15`, `git status`, `git branch -a --no-merged main`
  — the `--no-merged` filter matters: only branches with real unmerged
  work should show up here, not merged-and-already-deleted cruft.
  Anything real that's dangling (uncommitted work, an unmerged branch) —
  surface it, don't silently continue past it.
- `gh pr list` — any PR left open/unmerged from a previous session? If
  this fails with a `503`, that's known GitHub GraphQL flakiness (see
  checkpoint.md Notes) — retry once, or fall back to
  `gh api repos/<owner>/<repo>/pulls`.
- `main` is branch-protected — new work starts on a fresh branch, never
  directly on `main`.

## 4. Confirm the next task

- Cross-check checkpoint.md's "Next task to pick up" against
  `docs/Tessera_Phase1_Build_Plan.md` §5 — confirm that task's acceptance
  check (checkpoint.md should already carry it verbatim; if it doesn't,
  that's a gap `/end-day` should have filled last time).
- Check `## Notes / open flags` for anything that changes *how* the next
  task should run — e.g. Gemini's free-tier daily quota means any task
  calling the live LLM repeatedly (the eval harness, especially) needs
  deliberate pacing, not a blind loop.

## 5. Report and wait

Report concisely: what was last completed, environment/test-suite health,
anything left dangling from git/GitHub, and what task is next with its
acceptance check. Then stop — wait for go-ahead, don't start the task
automatically.
