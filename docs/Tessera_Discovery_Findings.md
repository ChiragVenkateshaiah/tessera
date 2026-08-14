# Project Tessera — Discovery Findings

**Client:** Meridian Advisory (600-person management consulting firm, 4 countries)
**Engagement:** Internal knowledge assistant — retrieval, citations, retrieval quality
**Author:** Chirag — Forward Deployed Engineer
**Status:** Draft for client review
**Distribution:** Priya Anand (Sponsor), Marcus Chen (Senior Partner, Financial Services), Dev Ramaswamy (Head of Enterprise Tech)

---

## 0. How to read this document

This is a *findings* document, not a solution design. Its job is to capture what we learned in discovery, confirm a shared understanding of the problem, and name what is still open — before any architecture or build commitment. If anything below misrepresents what was said or intended, that is exactly what this review round is for. A solution design and delivery plan follow once these findings are agreed and the query log (Section 7) is in hand.

---

## 1. Problem statement

Meridian's consultants lose a significant share of their time locating existing knowledge — prior project work, reusable frameworks, and internal expertise. The firm's own estimate is that a junior consultant spends roughly eight hours per week searching, contributing to margin erosion on fixed-fee engagements. This estimate is directional and self-reported (see Section 6); it is credible as a motivation but is not a measured baseline.

The desired outcome is an assistant that lets a consultant ask a natural-language question and receive a useful, sourced answer — closer to consulting a well-read senior partner than to running a keyword search. The hard part is not the assistant's helpfulness; it is delivering that helpfulness without breaching client confidentiality (Section 4).

## 2. Stakeholders

| Person | Role | Posture | What they need to say yes |
|---|---|---|---|
| Priya Anand | VP, Knowledge & Capability | Sponsor; wants this to succeed | Confidence we will not repeat the prior failure |
| Marcus Chen | Senior Partner, Financial Services | Skeptic (watched three prior attempts fail); now cautiously supportive | Credible handling of client-confidentiality risk before any excitement about time saved |
| Dev Ramaswamy | Head of Enterprise Tech | Guarded; concerned with data residency | Clarity on where queries and documents go relative to third-party model providers; accountability when the system is wrong |

Note: Marcus moved from blocker to cautious ally during the session, primarily in response to an honest acknowledgment of the limits of automated detection. Preserving that trust is a first-order concern for the engagement, not a soft one.

## 3. Content landscape

| Source | Approx. volume | Notes |
|---|---|---|
| PowerPoint decks (shared drive) | ~15,000 | Project deliverables and training; knowledge is often locked in diagrams/graphics, not text; organization quality not yet confirmed |
| Confluence wiki | ~3,000 pages | Internal methodology; largely low-sensitivity; curation quality not yet confirmed |
| SharePoint Word docs | ~8,000 | Proposals, case studies, client memos; mix of final and draft material |
| HR system | firm-wide | Bio / expertise data; relevant to expertise-finding queries; location and structure not yet confirmed |
| Slack history | 5 years | **Scope undecided** — flagged, not resolved (see Sections 5 and 9) |

## 4. Confidentiality — the central constraint

This is the defining risk of the engagement and the reason prior attempts are viewed with suspicion.

**How ethical walls work today.** Enforcement is manual and trust-based. When an engagement carries a conflict dimension, Legal/General Counsel establishes an information barrier: a named list of who is cleared and who is walled off, circulated to relevant partners. There is no automated enforcement layer; discretion is treated as a professional obligation, and a breach is a career-ending matter.

**No clean document tagging.** There is no reliable metadata system classifying documents by client, industry, or confidentiality level. Classification today is a judgment call.

**The anonymized-but-identifiable problem (critical).** Roughly a third of "anonymized" training material remains identifiable to an industry insider — higher in Financial Services, where the population of comparable firms is small. Sensitivity is not carried in specific keywords; it emerges from combinations of attributes that uniquely identify a client (e.g., size + region + recent corporate event). No keyword filter or naive redaction catches this, because there is nothing lexically secret to catch — identification is an inference, not a match. Today, the "is this safe to reuse?" decision is an unstructured, roughly ninety-second gut check by a busy partner, with no review board or sign-off.

**Design implication.** A permission model that simply inherits existing tags is insufficient, because the existing tags mark identifiable documents as safe. The leak risk lives inside the supposedly-safe set, not only the obviously-restricted set. This must be treated as a first-class design problem, and automated detection alone must not be presented as a solution.

## 5. Agreed pilot scope

To make progress without touching the minefield, the pilot is scoped to demonstrably low-sensitivity content:

**In scope for pilot**
- Confluence methodology wiki (internal frameworks, no client data)
- Published thought leadership (already public)

**Conditionally in scope**
- Generic training material — **only** after a human review pass, because this is where anonymized-but-identifiable documents hide

**Explicitly out of pilot scope**
- Client deliverables and any restricted-engagement material
- Comparative queries that would pull across engagements (see Section 8) — these re-trigger the confidentiality problem
- Slack history — undecided; not in the pilot

**Rationale.** Start where the risk is low, prove the system works, and earn the trust required before going anywhere near client-sensitive material. This also materially reduces Dev's data-residency concern for the pilot, since public and internal-methodology content is a different risk conversation than client deliverables — a conversation still owed, but not blocking for the pilot.

## 6. Success criteria

| Metric | Status | Approach |
|---|---|---|
| Weekly adoption (target ~50%) | Measurable | Usage telemetry built into the app |
| Satisfaction (NPS-style) | Measurable | Added to Meridian's existing internal-tool survey cycle |
| Time-to-information cut in half | **Deferred** | Requires a before/after baseline no one will run at pilot stage; treated as a later ROI exercise, not a launch gate |

The pilot's honest success bar, agreed in the room: consultants actually use the tool *and* report finding things faster. Both are measurable. The "eight hours per week" figure and the "cut in half" target are directional, not baselined, and should not be used as pass/fail gates.

## 7. Query archetypes (retrieval design input)

Real queries surfaced in the session fall into four technically distinct types. This grouping matters: "RAG" here is not one retrieval problem but four, each needing different handling.

**A — Prior-work / document lookup** ("find the thing that exists")
- Have we done work in [industry] on [topic] before?
- What did we propose last time we pitched [type of client]?
- Do we have a template/framework for [type of analysis]?
- Has anyone built a model for [specific thing] I can reuse?

**B — Expertise-finding** ("find the person, not the document")
- Who at the firm knows about X?
- *Distinct retrieval path: answer is a person, sourced from HR/staffing data, not documents.*

**C — Topic synthesis / "get me up to speed"** ("synthesize across many sources")
- I'm staffed on a retail-bank cost transformation Monday — what should I read first?
- Client meeting in an hour; they asked about [thing] — do we have anything? *(synthesis + speed-critical)*
- *Highest hallucination risk; requires multi-document reasoning, not single-doc retrieval.*

**D — Comparative** ("compare across engagements") — **confidentiality-sensitive**
- How did we approach margin improvement for Client X vs. the standard playbook?
- *Useful answers may pull from restricted engagements; likely excluded from pilot scope.*

**Client commitment:** Priya to log 20–30 real queries in consultants' actual wording, each paired with a rough description of what a good answer looks like. Owner: Priya. Target: ~1 week. This question/ideal-answer pairing is the evaluation set that will let us measure whether the system is any good; it is the single highest-value input to the build.

## 8. Lessons from the previous attempt

- Partners stopped using the prior tool within two months.
- Two distinct failure modes: (1) irrelevant results, and (2) naive security promises — the prior vendor claimed the system would "automatically detect and redact sensitive information," which fails for the reasons in Section 4.
- The credibility lesson: over-promising, especially on confidentiality, is what burned trust. The engagement's posture is deliberately the opposite — acknowledge limits honestly, design for the hard cases, and start where it is safe.

## 9. Open questions (to resolve before or during solution design)

1. **Legal named-list format** — is the who-is-cleared-for-which-client record structured data we could integrate with, or a manually maintained document? Determines whether access control is automatable.
2. **Budget range** — not landed in the workshop. Proof-of-concept vs. production-platform framing shapes model and infrastructure choices.
3. **Go/no-go decision-maker and kill criterion** — who owns the ship decision, and what happens if adoption underperforms.
4. **Data residency** — Dev's deferred conversation: where queries and documents go relative to third-party model providers. Not blocking for the pilot; required before client-sensitive content.
5. **Slack scope** — in for a later phase, or out entirely.
6. **Content freshness** — how quickly newly added content must become searchable (real-time vs. nightly).
7. **Deck extraction** — how much knowledge is locked in graphics/diagrams vs. recoverable text/speaker notes.
8. **HR expertise data** — location, structure, and update mechanism (needed for archetype B).
9. **Relevance failure specifics** — deeper detail on *how* the old tool's results were irrelevant, to avoid repeating it.

## 10. Next steps

| Owner | Action | Timing |
|---|---|---|
| Chirag | Circulate this Discovery Findings doc for review/correction | This week |
| Priya | Deliver query log (20–30 pairs, real wording + ideal answer) | ~1 week |
| Meridian | Review and confirm/correct findings, esp. pilot scope and confidentiality framing | On receipt |
| Chirag | On alignment, produce solution design (architecture, evaluation approach, risks) and a realistic delivery plan | After findings agreed + query log received |

**Sequence, explicitly:** findings agreed → solution design → delivery plan → build. No architecture, stack, or delivery-methodology commitments are made in this document; they follow the solution design, once there is a validated problem to design against.
