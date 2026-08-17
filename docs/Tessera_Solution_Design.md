# Project Tessera — Solution Design (v1)

**Client:** Meridian Advisory
**Engagement:** Internal knowledge assistant — retrieval, synthesis, citations
**Author:** Chirag — Forward Deployed Engineer
**Status:** Draft — depends on Discovery Findings (agreed) and query log (pending)
**Companion doc:** `Tessera_Discovery_Findings.md`

---

## 0. How to read this document

This design turns the agreed discovery findings into a buildable approach. It is deliberately layered: the GenAI core and evaluation framework are specified in depth because they are built first; the AWS architecture and MLOps layer are documented as the intended full-production target, not as v1 deliverables. Where the design depends on an input not yet in hand — chiefly Priya's query log — that dependency is marked explicitly rather than assumed away.

**Build philosophy:** make it work, then make it production-grade. The GenAI core is proven locally against a real evaluation set before any cloud or pipeline maturity is added. Infrastructure is supporting cast, not the deliverable.

**Layer status at a glance:**

| Layer | v1 intent | Depth here |
|---|---|---|
| 1. GenAI core | Build locally | Designed in depth |
| 2. Evaluation framework | Build locally (scaffold now, populate on query log) | Designed in depth |
| 3. Access-control / confidentiality model | Design now, scope-limit for pilot | Designed, scoped |
| 4. AWS architecture | Document for full production | Documented, not built in v1 |
| 5. MLOps / DevOps | Document for full production | Documented, not built in v1 |

---

## 1. GenAI core

### 1.1 Guiding principle

Tessera is not one retrieval problem. Discovery surfaced four technically distinct query archetypes (Discovery Findings §7), and the core is designed around that fact. A single naive RAG pipeline would serve some archetypes well and others poorly. The core routes and handles each archetype according to its nature.

### 1.2 Archetype-specific handling

**A — Prior-work / document lookup** ("find the thing that exists")
- Nature: precision retrieval against a known corpus.
- Approach: semantic retrieval over chunked documents with metadata filters (industry, topic, doc type). Return ranked documents with citations; light synthesis to summarize what was found.
- Primary quality risk: recall — missing a relevant deck that exists.

**B — Expertise-finding** ("find the person, not the document")
- Nature: not document retrieval at all — the answer is a person.
- Approach: separate retrieval path over structured HR/staffing/expertise data. Depends on an open question (Discovery Findings §9.8): where that data lives and how it is structured. Treated as a distinct index, not folded into document retrieval.
- Primary quality risk: stale or self-reported expertise data.

**C — Topic synthesis / "get me up to speed"** ("synthesize across many sources")
- Nature: multi-document reasoning, not single-doc lookup. Highest value and highest hallucination risk.
- Approach: broader retrieval (higher k), then a synthesis step that composes an answer grounded strictly in retrieved sources, with inline citations and explicit "here's what we have / here's what we don't" framing. Guardrail: the model must not synthesize beyond retrieved content.
- Primary quality risk: hallucination and over-confident synthesis; also latency for the speed-critical "meeting in an hour" case.

**D — Comparative** ("compare across engagements") — confidentiality-sensitive
- Nature: comparison that may span restricted material.
- Approach: **out of pilot scope** (Discovery Findings §5). Where it appears, the system should recognize the pattern and respond that comparative cross-engagement answers are restricted, rather than attempt them. Designed as a guardrail, not a feature, for v1.

### 1.3 Ingestion and extraction

- **Confluence + published thought leadership** (pilot corpus): predominantly text; standard extraction, chunking, and embedding.
- **PowerPoint** (later phases): knowledge is frequently locked in diagrams and graphics, not text (Discovery Findings §9.7). Text-only extraction will under-serve these; a richer extraction strategy (including visual/structured content) is required before decks enter scope. Out of pilot corpus.
- **Chunking:** semantic/section-aware rather than fixed-size, to preserve the coherence of methodology frameworks.

### 1.4 Retrieval and generation

- **Retrieval:** vector search with metadata filtering; k tuned per archetype (lookup narrow, synthesis broad).
- **Generation:** Claude via Bedrock in production; grounded generation with mandatory citations. Every claim traceable to a retrieved source.
- **Guardrails:** answers grounded in retrieved content only; explicit handling for "we don't have anything on that" rather than fabrication; comparative-query refusal per §1.2D.

### 1.5 Routing

A lightweight classification step routes each incoming query to its archetype path (A/B/C/D). Routing quality is itself measured (see §2), because a misrouted query gets the wrong retrieval strategy.

## 2. Evaluation framework

This is the credibility layer. Most demos skip it; Tessera treats it as first-class, because the entire discovery narrative was about *not* repeating a failure — and "we can prove it works" is the answer to that.

### 2.1 Dependency (explicit)

The evaluation set is built from Priya's query log: 20–30 real queries in consultants' actual wording, each paired with a description of a good answer (Discovery Findings §7). **This design scaffolds the framework now; the eval set is populated when the log arrives.** Until then, the eight illustrative workshop queries serve as a placeholder set, clearly marked as non-representative.

### 2.2 What gets measured

- **Retrieval quality:** for lookup/synthesis archetypes — are the right sources retrieved? Metrics: recall@k, precision@k, mean reciprocal rank against a labeled relevant-set.
- **Answer quality:** groundedness (every claim cited and supported), relevance to the question, and appropriate "we don't have this" behavior. Assessed via a rubric; where automated, an LLM-as-judge against the ideal-answer descriptions, with human spot-checks.
- **Routing accuracy:** is each query sent to the correct archetype path?
- **Safety behavior:** does the system correctly refuse out-of-scope comparative/restricted queries?
- **Latency:** tracked per archetype, with attention to the speed-critical synthesis case.

### 2.3 Method

- Golden set of query/ideal-answer pairs from the query log.
- Relevant-source labeling for retrieval metrics (which documents *should* surface for each query).
- Regression discipline: the eval set runs on every meaningful change to retrieval or prompting, so quality changes are visible rather than discovered in production.

### 2.4 Why this matters for the pilot

The agreed success bar is adoption + satisfaction (Discovery Findings §6). The eval framework is the *internal* instrument that lets us reach that bar with confidence before real users touch it — it is how we avoid shipping the previous attempt's "irrelevant results" problem.

## 3. Access-control / confidentiality model

The distinctive hard problem (Discovery Findings §4). For the pilot, scope removes most of the risk; the design still records how the full problem would be approached, because that is the intellectually important part of the engagement.

### 3.1 Pilot posture

- Pilot corpus is low-sensitivity by construction (methodology + published thought leadership). This sidesteps, rather than solves, the anonymized-but-identifiable problem — a deliberate sequencing choice, not an oversight.
- Training decks enter only after a human review pass.
- Comparative cross-engagement queries are refused (§1.2D).

### 3.2 The full problem (documented, not built in pilot)

- **Why naive approaches fail:** sensitivity is not lexical. "A $47B bank in Toronto post-2023-merger" contains no keyword to flag, yet may uniquely identify a client. Automated detection alone is insufficient and must not be presented as a solution (Discovery Findings §4).
- **Direction of a real solution** (for later design, not commitment): a combination of (a) conservative ingestion — deciding what enters the index at all; (b) provenance/eligibility metadata sourced from the Legal named-list system of record (format is an open question, §9.1); (c) human-in-the-loop review for the ambiguous middle; and (d) query-time access checks tied to the requesting consultant's clearances. No single automated layer is trusted to catch identifiability on its own.
- **Accountability:** Dev's point — a model that is right 95% of the time still leaks 1 in 20. The design must assume automated classification is an assist to human judgment, not a replacement for it.

## 4. AWS architecture (documented for full production)

Not built in v1. Recorded so the portfolio demonstrates end-to-end thinking and to give a clear target once the core proves out.

**Intended footprint:**

- **Model serving:** Amazon Bedrock (Claude) — aligns with the platform choice and keeps model access managed.
- **Vector store / retrieval:** Amazon OpenSearch Serverless (vector) or equivalent managed vector store; Bedrock Knowledge Bases as an option to reduce glue code.
- **Document storage:** S3 for the source corpus, with lifecycle and access policies.
- **Compute / orchestration:** Lambda for ingestion and query-handling functions; Step Functions if multi-step orchestration grows.
- **API layer:** API Gateway fronting the query service.
- **Identity / access:** IAM for service permissions; integration with Meridian's identity provider for consultant-level access (ties to §3).
- **Secrets / config:** Secrets Manager / Parameter Store.
- **Data residency:** region selection and tenancy decisions driven by Dev's deferred residency conversation (Discovery Findings §9.4) — required before any client-sensitive content, not for the low-sensitivity pilot.

**Pilot footprint (when the core moves off local):** the minimum of the above — S3 + a managed vector store + Bedrock + a single query function — sized for a pilot, not a platform.

**Detailed language/runtime decisions for the intended footprint** (a hybrid Go edge/routing layer in front of this Python core, evolving from serverless to Kubernetes) are recorded in `docs/adr/` rather than here — this section stays the service-level architecture; the ADRs carry the concrete decision, trade-offs, and alternatives considered for the API/routing layer this section left unspecified.

## 5. MLOps / DevOps layer (documented for full production)

Not built in v1. Added after layers 1–3 work.

- **Infrastructure as Code:** Terraform for reproducible, reviewable infrastructure.
- **CI/CD:** pipeline that runs the evaluation set (Section 2) as a gate — no change ships if retrieval/answer quality regresses. This is the DevOps practice and the eval framework reinforcing each other.
- **Monitoring / observability:** usage telemetry (also the adoption metric, Discovery Findings §6), query/latency/cost dashboards, and answer-quality sampling in production.
- **Cost tracking:** per-query cost visibility, relevant to the (still open) budget question (Discovery Findings §9.2).
- **Model/version discipline:** track prompt versions, retrieval config, and eval results together so any quality change is attributable.

## 6. Build sequence

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| 0 | Receive query log; populate eval set | Query log in hand |
| 1 | GenAI core (archetypes A + C first) running locally | Answers grounded + cited on placeholder set |
| 2 | Evaluation framework populated and passing on real query log | Retrieval + answer metrics meet an agreed internal bar |
| 3 | Expertise-finding (archetype B) once HR data source is known | HR data structure confirmed (§9.8) |
| 4 | Minimal AWS footprint; move core off local | Core stable locally |
| 5 | MLOps layer: IaC, CI/CD-with-eval-gate, monitoring | Core stable on AWS |

**Explicit dependency:** Phases 1–2 are the star and are built first. Phases 4–5 are documented in full here but are not v1 work. The project is coherent and demoable if it stops after Phase 2 — a working, evaluated knowledge assistant on the pilot corpus.

## 7. Open dependencies (carried from discovery)

- Query log (§2.1) — blocks eval population; shapes archetype tuning.
- Legal named-list format (§3.2) — blocks any real access-control build.
- HR expertise data source (§1.2B) — blocks archetype B.
- Data residency decision (§4) — blocks client-sensitive content.
- Budget range — shapes model/infra choices at production scale.

Nothing in this list blocks Phase 1. The core can begin as soon as the query log lands.
