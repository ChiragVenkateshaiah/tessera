# Tessera — Phase 1 Build Plan (Claude Code Brief)

**Read this first, in full, before writing any code.**

This document is the working brief for Phase 1 of Project Tessera. It is written to be dropped into an empty repository and used as the instruction set for the build. Companion documents (`docs/Tessera_Discovery_Findings.md`, `docs/Tessera_Solution_Design.md`) carry the problem context and full architecture; this document covers only what gets built now.

---

## 0. Context in one paragraph

Tessera is an internal knowledge assistant for Meridian Advisory, a 600-person consulting firm whose consultants lose significant time searching for prior work, frameworks, and internal expertise. Discovery established that this is not one retrieval problem but four distinct query archetypes, and that client confidentiality — specifically documents that are "anonymized" but still identifiable to an industry insider — is the defining risk. The pilot deliberately scopes to low-sensitivity content (methodology wiki + published thought leadership) to prove the system works before going near client material.

## 1. Phase 1 objective and boundaries

**Objective:** a working, locally-run ingestion and retrieval core over the pilot corpus, plus an empty-but-functional evaluation harness — provable on placeholder queries, ready to be populated with the real query log when it arrives.

**In scope for Phase 1**
- Synthetic pilot corpus (the client's real corpus does not exist for this build)
- Ingestion + section-aware chunking
- Embedding + local vector store
- Retrieval for archetype A (lookup) and archetype C (synthesis)
- Grounded generation with mandatory citations
- Evaluation harness scaffold — runnable, metrics implemented, test cases empty
- CLI for smoke-testing queries

**Explicitly NOT in Phase 1 — do not build these**
- Archetype B (expertise-finding) — blocked on unknown HR data structure
- Archetype D (comparative) — out of pilot scope by design; implement only as a refusal guardrail
- Any AWS deployment, Terraform, CI/CD, or monitoring — Phases 4–5
- PowerPoint/deck ingestion — not in pilot corpus
- Access-control enforcement — pilot corpus is low-sensitivity by construction
- Web UI — CLI is sufficient for Phase 1

If a task seems to require something on the NOT list, stop and flag it rather than building it.

## 2. Design constraints that shape the code

These are not preferences; they are the reasons the architecture looks the way it does.

1. **Swappable ports.** Phase 4 moves this to AWS (Bedrock, OpenSearch, S3). Every external dependency — embedding model, vector store, LLM client, document source — sits behind a thin interface so the swap is a config change, not a rewrite. This is the single most important structural decision in Phase 1.
2. **Grounded generation only.** Every claim in an answer must trace to a retrieved chunk. The system says "we don't have anything on that" rather than fabricating. This directly answers the prior vendor's failure.
3. **Archetypes are first-class.** Retrieval behaviour differs by archetype (lookup: narrow, precise; synthesis: broad, multi-source). Do not collapse them into one pipeline.
4. **Evals are infrastructure, not an afterthought.** The harness is built in Phase 1 even though the real test cases arrive later, because it becomes the CI gate in Phase 5.
5. **Local-first.** No cloud dependencies in Phase 1 except the LLM API call itself.

## 3. Technology decisions

| Concern | Phase 1 choice | Phase 4 target | Notes |
|---|---|---|---|
| Language | Python 3.11+ | same | |
| Env / deps | `uv` (or venv + pip) | same | Lockfile committed |
| Embeddings | `sentence-transformers` local model | Bedrock Titan / Cohere | Behind `Embedder` interface |
| Vector store | Chroma (local, persistent) | OpenSearch Serverless | Behind `VectorStore` interface |
| LLM | DeepSeek API | Claude via Bedrock | Behind `LLMClient` interface; free tier for Phase 1, swapped for Claude/Bedrock in Phase 4 |
| Config | `pydantic-settings` + `.env` | same + Parameter Store | No hardcoded values |
| Testing | `pytest` | same | |
| CLI | `typer` | n/a | |

`.env` is gitignored. An `.env.example` is committed.

## 4. Repository structure to create

```
tessera/
├── README.md
├── CLAUDE.md                    # working conventions for this repo
├── pyproject.toml
├── .env.example
├── .gitignore
├── docs/
│   ├── Tessera_Discovery_Findings.md
│   ├── Tessera_Solution_Design.md
│   └── Tessera_Phase1_Build_Plan.md
├── data/
│   ├── corpus/                  # synthetic pilot corpus (committed)
│   └── vectorstore/             # local Chroma persistence (gitignored)
├── src/tessera/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── loader.py            # reads corpus files
│   │   └── chunker.py           # section-aware chunking
│   ├── embedding/
│   │   ├── base.py              # Embedder interface
│   │   └── local.py             # sentence-transformers impl
│   ├── store/
│   │   ├── base.py              # VectorStore interface
│   │   └── chroma.py            # Chroma impl
│   ├── retrieval/
│   │   ├── router.py            # archetype classification
│   │   └── retriever.py         # archetype-aware retrieval
│   ├── generation/
│   │   ├── base.py              # LLMClient interface
│   │   ├── deepseek.py          # DeepSeek impl
│   │   └── prompts.py           # grounded-answer prompts
│   ├── pipeline.py              # query -> route -> retrieve -> generate
│   └── cli.py
├── evals/
│   ├── harness.py               # runner
│   ├── metrics.py               # recall@k, precision@k, MRR, groundedness
│   ├── cases/
│   │   └── placeholder.yaml     # 8 workshop queries, marked non-representative
│   └── README.md                # how to populate with the real query log
└── tests/
```

## 5. Task breakdown

Work in this order. Each task has an acceptance check — do not move on until it passes.

### Task 1 — Repo scaffold and synthetic corpus
Create the structure above. Then generate a synthetic pilot corpus that plausibly represents Meridian's low-sensitivity content:
- ~25–30 methodology wiki pages (markdown, with headings): consulting frameworks such as market entry analysis, cost transformation, pricing strategy, operating model design, due diligence. Written as internal how-to methodology, no client data.
- ~10–12 thought-leadership pieces (markdown, longer form): industry perspectives across financial services, retail, pharma, industrials.
- Front-matter metadata per file: `title`, `doc_type` (methodology|thought_leadership), `industry`, `topics`, `date`.

Corpus realism matters — retrieval quality is meaningless against toy documents. Vary length, overlap topics deliberately so retrieval has to discriminate.

**Acceptance:** corpus exists, files parse, metadata consistent.

### Task 2 — Ingestion and chunking
Loader reads the corpus with metadata intact. Chunker splits section-aware (respect markdown heading boundaries) rather than fixed-size, preserving framework coherence. Each chunk carries source metadata for citation.

**Acceptance:** chunk count reasonable; no chunk orphaned from its source metadata; headings not split mid-section.

### Task 3 — Embedding and vector store behind interfaces
Define `Embedder` and `VectorStore` interfaces first, then the local implementations. Index the corpus. Persist locally.

**Acceptance:** corpus indexed; a manual similarity query returns plausible chunks; swapping implementations requires no changes outside the `embedding/` and `store/` modules.

### Task 4 — Archetype router
Lightweight classifier (LLM-based is acceptable, keep the prompt in `prompts.py`) mapping a query to A (lookup), B (expertise), C (synthesis), or D (comparative). B returns "not yet supported"; D returns the confidentiality refusal.

**Acceptance:** the 8 placeholder queries route correctly; routing decision is logged and inspectable.

### Task 5 — Archetype-aware retrieval
Retriever varies strategy by archetype: A uses narrow k with metadata filtering; C uses broader k across sources. Return chunks with source attribution.

**Acceptance:** same query under A vs C settings returns visibly different retrieval breadth.

### Task 6 — Grounded generation with citations
Prompt design enforcing: answer only from retrieved chunks; cite sources inline; explicitly state when the corpus has nothing relevant. Separate prompt shapes for A (found-documents summary) and C (multi-source synthesis).

**Acceptance:** every answer carries citations; an off-corpus question ("what's our policy on parental leave") produces a clean "we don't have anything on that" rather than invention.

### Task 7 — Evaluation harness
Runner that loads cases from `evals/cases/*.yaml`, executes them through the pipeline, and reports metrics: recall@k, precision@k, MRR (retrieval); groundedness and relevance (answer, LLM-as-judge with the case's ideal-answer description); routing accuracy; latency per archetype.

Case schema:
```yaml
- id: q001
  query: "Do we have a framework for market entry analysis?"
  archetype: A
  relevant_sources: ["methodology/market-entry-analysis.md"]
  ideal_answer: "Points to the market entry methodology page, summarises the key steps, cites the source."
```

`evals/README.md` documents how to populate from the real query log (20–30 pairs, actual consultant wording + ideal answer).

**Acceptance:** harness runs end-to-end on the 8 placeholder cases and emits a metrics report. Numbers may be poor — the harness working is the deliverable, not the scores.

### Task 8 — CLI and README
`tessera ingest` and `tessera query "..."` at minimum; `tessera eval` runs the harness. README covers setup, the phase boundary (what is built vs designed), and how to run each command.

**Acceptance:** a fresh clone can be set up and queried following the README alone.

## 6. CLAUDE.md content to create

Create `CLAUDE.md` at repo root containing: the one-paragraph project context (§0), the design constraints (§2), the "do not build" list (§1), the technology table (§3), and a note that `docs/` holds the discovery and design rationale. This keeps future Claude Code sessions oriented without re-reading everything.

## 7. Phase 1 exit criteria

Phase 1 is complete when:
1. A fresh clone can ingest the corpus and answer a query with citations from the CLI.
2. Archetypes A and C behave observably differently; B and D return correct non-answers.
3. The eval harness runs and reports all metric categories on placeholder cases.
4. Every external dependency sits behind an interface with at least one alternative implementation stubbed or trivially addable.
5. README honestly states what is built versus designed-for-later.

**Not required to exit Phase 1:** good eval scores. Tuning happens in Phase 2, against the real query log. Building the machine comes before tuning it.

## 8. Working conventions

- Small, reviewable commits with clear messages; commit at each task boundary.
- Type hints throughout; docstrings on public interfaces.
- No secrets in the repo. No hardcoded paths.
- Tests for chunking, routing, and metrics logic — the deterministic parts. Do not over-test LLM outputs; that is what the eval harness is for.
- When a decision is ambiguous, prefer the option that keeps the AWS migration cheap.
