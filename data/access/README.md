# `data/access/` — synthetic ethical walls

Meridian's ethical walls (Discovery §4) are per-engagement lists of the
people cleared for that engagement; everyone else is walled from it.
`docs/Tessera_Phase4_Plan.md` §3.5 is the design. Like the corpus and the
expertise dataset, **all of this is fictional** and synthesized on purpose.

- **`generate.py`**: the generator. Deterministic (`SEED = 20261003`):
  it reads the restricted engagement summaries in `data/corpus/` and the
  people in `data/expertise/people/` and staffs each engagement with a
  pyramid of seven (partner, principal, engagement manager, two
  consultants, two analysts), preferring people whose project history
  covers the engagement's lead topic. Run from the repo root:
  `python data/access/generate.py`.
- **`walls.yaml`**: its output. **Do not hand-edit**; change the
  generator and re-run. `tests/test_access.py` fails if they drift, and
  checks the walls against the corpus's engagements and the people.

`walls.yaml` also names three **demo personas**, used by the access eval
sets and the demo: a partner cleared for Halcyon, a pricing analyst
cleared for nothing, and a partner cleared for Kestrel only. These are
demo identities: Tessera has no authentication, and anyone asking "as"
a person is taken at their word (plan §6).

Read and validated by `src/tessera/ingestion/access_loader.py`
(`load_walls`). Deny by default: a person is cleared for an engagement
only if its list names them.

**Reviewers (Phase 5, `docs/Tessera_Phase5_Plan.md` §3.8.1).** The
`reviewers:` list names the Legal staff who may approve a pending document
for indexing with `tessera corpus review --as <id>`. Discovery §4 says
Legal / General Counsel sets the barriers, so reviewers are not
consultants: they have their own ids (`legal-01`, `legal-02`), sit on no
ethical wall, and are not in the expertise dataset. Each carries the
engagements they are cleared to see for review:
- `general_counsel` (`legal-01`) is cleared for every engagement and may
  approve, reclassify or reject;
- `legal_associate` (`legal-02`) is cleared for a seeded half and may only
  leave documents pending. The rule is about permission, not about the
  document: a reviewer who can't see every engagement can't rule out a
  resemblance to one they can't see.

Like the personas, these are demo identities.
