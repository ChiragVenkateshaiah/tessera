---
description: Write the reviewer's guide for a Tessera PR (decision summary, reading order, what to check, before/after, risks)
argument-hint: <PR number>
---

# Reviewer's guide for a PR

Write the reviewer's guide for PR **#$ARGUMENTS** in this repository,
in the format below.

**When this runs.** Write the guide in either of these cases:
- the user runs `/pr-review <n>`;
- the user runs `gh pr view <n> --web` in the chat (as `! gh pr view <n> --web`).
  The browser opening is the user's cue that they are reviewing PR `<n>`.

Also post the guide, without being asked, right after opening any PR with
`gh pr create`.

## Why this format

The user reviews and merges every PR (`main` is branch-protected) and uses
reviews to build engineering judgement. An analogy (ELI5) explains an idea,
but it does not help someone read a diff. The guide points at real code, so
each claim can be found and checked on GitHub's "Files changed" page.

## How to gather the facts

Do this before writing anything. Never guess a line number.

1. `gh pr view <n> --json title,state,headRefName,baseRefName,additions,deletions,files,body`
   — the scope, the branch and the PR body (its test and sweep results).
2. `gh pr diff <n>` — the change itself.
3. Find line numbers in the PR's **head** version, not in whatever branch
   is checked out locally:
   `git fetch origin <headRefName>` then `git show origin/<headRefName>:<path> | grep -n ...`.
   For a merged PR whose branch was deleted, use `gh pr view <n> --json mergeCommit`
   and read the files at that commit.
4. Note data or generated files (eval exports, snapshots, lockfiles,
   generated datasets) so the reader can skip them.

## The format

Plain technical English. Short sentences. Keep the real names: files,
functions, settings, commands. No analogies inside the guide. Be honest
about weak spots; the guide is not a sales pitch.

```markdown
# Reviewer's guide: PR #<n>, <short title>

## 1. Decision summary
- **What changed:** <one or two sentences>
- **Recommendation:** merge | merge after <fix> | do not merge, because <reason>
- **Main risk:** <the one thing most worth the reader's attention>

**Size:** <files> files, +<additions> / −<deletions>. <Which part is data
or generated, and can be skipped; how much code is left to read.>

## 2. Reading order

| # | File | What it is for | Time |
|---|---|---|---|
| 1 | `path` | <purpose> | <n> min |

<Why this order: usually dependencies first, then callers, then tests.>

## 3. What to check in each file

### 3.<k> `path`
- `function()` (line N): <what it does>.
- **Check:** <what the reader should see there, and why it matters>.
- <Optional> **Engineering question to ask yourself:** <a question, with its answer>.

## 4. Behavior, before and after

<Real commands and real output (from the tests, a sweep or a live run).
Label any output that is illustrative rather than captured.>

## 5. Risks and accept/reject criteria

| Risk | Impact | Tested? |
|---|---|---|

**Accept if all are true:**
- [x] / [ ] <criterion, with the evidence>

**Reject if** you find any of these:
- <a concrete finding that would justify rejecting>

**Exercise for you, before you accept:** <one `file:line` range and one
question that tests whether the reader understood the key design point>.
Merge with `! gh pr merge <n> --merge --delete-branch`.

---

**View the diff in the browser:** https://github.com/ChiragVenkateshaiah/tessera/pull/<n>/files
```

## Rules

- Every `file:line` reference must be checked against the PR's head (see
  "How to gather the facts"). If the PR was updated after the guide was
  written, regenerate the guide.
- The decision summary comes first and stays three lines.
- Section 5 always says what was **not** tested.
- The last line is always the link to the PR's "Files changed" page.
- For a PR that is already merged, say so in the decision summary, and
  write the guide as a review of what went in.
- The PR body stays the record in normal English. This guide is a chat
  reply for the reviewer, not a file in the repo.
