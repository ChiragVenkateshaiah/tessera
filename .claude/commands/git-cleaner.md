Run the two-machine sync ritual before starting any work session. Check both
repos for uncommitted changes, pull with rebase from origin, and report what's
new. Repos: ~/projects/cerberus and ~/projects/tessera.

For each repo in that order:

1. Run `git status --short`. If there are modified, staged, or untracked files
   (ignore .python-version, __pycache__, .DS_Store, .venv) warn the user and
   stop — a dirty tree must be committed or stashed before rebasing.
2. Show the current branch name.
3. Run `git fetch origin`.
4. Run `git rebase origin/<current-branch>`. If it succeeds, show new commits
   with `git log --oneline ORIG_HEAD..HEAD` (or say "already up to date").
   If rebase fails, stop and guide the user to resolve conflicts — do not abort
   automatically.
5. Run `git remote prune origin` and report any pruned branches.

After both repos print a plain summary:
- All clean and up to date → "Ready to work."
- Any issues → list exactly what needs attention.

Cerberus only: if `terraform/envs/dev/.terraform/` or
`terraform/bootstrap/.terraform/` is missing, append a reminder to run
`terraform init` in that stack before any plan or apply.
