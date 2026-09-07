# `data/expertise/` — synthesized firm expertise dataset

Archetype B (expertise-finding) answers "who at the firm knows about X"
from structured staffing/expertise data rather than documents
(`docs/Tessera_Solution_Design.md` §1.2B). Discovery left the location
and structure of that data open (Discovery Findings §9.8), and **Meridian
Advisory and its staff are fictional** — so, like the pilot corpus and
the query log before it, this dataset is **deliberately, transparently
synthesized**, not waited for. `docs/Tessera_Phase3_Plan.md` §2 is the
design.

## What's here

- **`generate.py`** — the generator. Deterministic: a fixed seed
  (`SEED = 20260907`) plus sorted iteration over `data/corpus/`, so
  re-running produces byte-identical output. Run it from the repo root:

  ```sh
  python data/expertise/generate.py
  ```

- **`people/*.yaml`** — its output, one file per practice, ~600
  consultants total. Committed so the dataset is reviewable and the
  generator's shape is inspectable. **Do not hand-edit** — change
  `generate.py` and re-run.

The read + validate side is `src/tessera/ingestion/expertise_loader.py`
(`load_expertise(people_dir, corpus_dir)`), the archetype-B analogue of
`loader.py` for documents.

## Record schema

```yaml
- person_id: c0142            # stable slug, c0001–c0600
  name: Maya Fernandez        # synthesized, non-real
  title: Engagement Manager   # Analyst | Consultant | Engagement Manager | Principal | Partner
  office: London              # one of a fixed set
  practice: pricing           # the practice this person sits in
  skills:
    - topic: value-based-pricing   # a real corpus topic
      level: 4                     # 1–5 self-assessment
      basis: evidenced             # evidenced | self_reported
  project_history:
    - industry: pharma             # a fixed set, wider than the corpus's 5
      topic: value-based-pricing   # a real corpus topic
      role: workstream lead        # a fixed set of engagement roles
      year: 2023
  authored:
    - thought_leadership/pharma-value-based-contracting.md   # a real corpus file
  languages: [English, French]
  last_updated: 2026-04-18    # snapshot date; the B answer surfaces its age
```

### `basis`: evidenced vs. self_reported

The core modelling decision (plan §2.2). A skill is **`evidenced`** when
the person has a `project_history` entry *or* an `authored` corpus
document on that exact topic; otherwise it is **`self_reported`** — a
claim with nothing behind it. Retrieval (P3-3) weights evidenced above
self_reported; the B answer (P3-4) flags a match that rests only on
self-reported skills. Roughly 45% of skill entries are evidenced.

### Confidentiality

`project_history` carries **industry + topic + role + year and nothing
else** — there is no free-text field, so no client name can appear
(plan §2.3). The loader enforces this structurally: unknown keys, or an
industry/topic/role outside the known sets, fail the load. No
compensation, ratings, tenure, or personal data — this is a skills
directory, not an HR record. `authored` points only at the already-public
pilot corpus.

## Shape

- Seniority pyramid (~34/30/20/11/5% Analyst→Partner).
- ~600 people across 10 practices; every corpus topic that a practice
  claims is asserted against the real corpus vocabulary at generation
  time.
- A deliberate **weak-signal tail** (~17% of records flagged thin at
  generation: 2 skills, low levels, mostly self-reported, ≤1 project,
  no authorship) plus near-miss people (right practice, adjacent topic
  only) — so a query has a few strong matches and a long tail of weak
  ones, and retrieval has to discriminate.
- A **staleness tail** in `last_updated` (~62% 2026, then 2025 / 2024 /
  2023) so the "how old is this evidence" surface in the B answer has
  something to show.
