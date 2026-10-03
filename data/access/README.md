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
