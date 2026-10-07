# P5-5 — LangChain retrieval: the evidence

Phase 5 plan §3.4. Every retrieval-only number here comes from a
zero-LLM-call run on 2026-10-07. The multi-query sweep (§4) is live, on
NIM. Reproduce with:

```sh
python -m evals.compare_retrievers --out evals/reports/p5-5-retrieval.json
pytest tests/test_lc_retrieval.py tests/test_langsmith_redaction.py -q
TESSERA_LC_RETRIEVER=multiquery tessera eval --check --stack lc
```

## 1. The design in one paragraph

Every LangChain retriever reaches the index **only** through a
`ScopedStore`. A `ScopedStore` is built per request and bound to one
principal. Every query, row read and vector read through it ANDs that
principal's `permission_filter`. A **candidate generator** produces a
ranked list: vector, BM25, the hybrid `EnsembleRetriever`,
`MultiQueryRetriever`, a cross-encoder reranker, or
`ParentDocumentRetriever`. Then **one contract** runs for all of them:
1. every candidate is re-scored with cosine similarity, from its stored
   vector;
2. the native `retrieve_from_candidates` applies the freshness and
   permission re-check, diversification, the superseded probe and
   replacements, and lookup's expansion.

A variant therefore changes only which chunks are candidates and in what
order. With the vector generator it reproduces native `retrieve()`
exactly.

## 2. Retrieval-only comparison

The LangChain collection was built through native delete-then-add, so
only the retriever changes. Every A/C case ran with its archetype forced
and an internal-only principal. The scoring is the harness's: unique
documents, @5. The hybrid weight was **tuned on `query_log.yaml` (31
cases)** and is reported on **held-out `placeholder.yaml` (4 cases; one
document there moves recall by 0.05)**.

| Retriever | Tuned recall | Tuned precision | Tuned MRR | Tuned misses | Held-out recall | Held-out precision | Held-out MRR | Held-out misses | Context chars (tuned) | Latency ms (tuned) |
|---|---|---|---|---|---|---|---|---|---|---|
| vector (= native) | 0.959 | 0.400 | 0.968 | 0 | 0.900 | 0.600 | 0.875 | 0 | 7,317 | 64 |
| parent_doc | 0.959 | 0.387 | 0.968 | 0 | 0.900 | 0.600 | 0.875 | 0 | 21,923 | 84 |
| bm25 | 0.793 | 0.290 | 0.729 | 2 | 0.367 | 0.300 | 0.292 | 1 | 7,943 | 63 |
| hybrid bm25=0.25 **(chosen weight)** | 0.968 | 0.387 | 0.968 | 0 | 0.850 | 0.550 | 1.000 | 0 | 7,606 | 94 |
| hybrid bm25=0.5 | 0.960 | 0.381 | 0.952 | 0 | 0.800 | 0.500 | 0.875 | 0 | 7,537 | 93 |
| hybrid bm25=0.75 | 0.960 | 0.381 | 0.884 | 0 | 0.850 | 0.550 | 0.875 | 0 | 7,777 | 93 |
| rerank (cross-encoder) | 0.900 | 0.355 | 0.903 | 1 | 0.900 | 0.600 | 0.875 | 0 | 6,874 | 2,578 |

**What it shows:**
- **BM25 alone is the weakest by far.** Its held-out recall is 0.367. The
  questions are paraphrases ("what should I read first?"), not keyword
  matches, and lexical ranking misses them. It still has a use as a
  minority voice in the hybrid.
- **The hybrid at 0.25 is a coin toss, not a win.** On the tuning set it
  gains 0.009 recall, about one more relevant document across 31 cases.
  On the held-out set recall drops by 0.05 while MRR rises to 1.000. That
  is below the project's 0.02 "prefer current" rule
  (`evals/tune_retrieval.py`), so it is not a default.
- **The cross-encoder reranker loses on this corpus and costs 40 times
  the latency.** It loses one tuned case completely, and takes 2.6 s per
  retrieval on CPU against 64 ms. `ms-marco-MiniLM` was trained on web
  search passages. Here, the title and heading prefix that the vectors
  embed carries most of the signal, and the reranker scores the body
  alone.
- **`ParentDocumentRetriever` finds the same documents and sends 3 times
  the context** (whole documents for all 5 lookup results, where native
  expands only the top 2). The answer model would read 22k characters
  instead of 7k, for the same recall.

**Decision (`lc-defaults`):** `TESSERA_LC_RETRIEVER=lc`. That is the
vector retriever through the contract, identical to native.
`TESSERA_LC_HYBRID_BM25_WEIGHT=0.25` is recorded for the profiles that
select the hybrid.

## 3. The safety evidence

| Check | Where | Result |
|---|---|---|
| **Parity**: the vector retriever vs native `retrieve()`, every A/C case and every access case (with principals) | `tests/test_lc_retrieval.py` | chunk ids, order, scores (< 1e-5), superseded documents and counts identical |
| **Leak test, every retriever**: vector, parent_doc, bm25, hybrid, multiquery, rerank, `TesseraRetriever`. All 29 access cases as their principals | `tests/test_lc_retrieval.py` | 0 unpermitted chunks; 0 context-marker hits for uncleared engagements; authorized recall 12/12 for each |
| **Scope binding**: a recording store sees every `where`, for 6 retriever kinds × 3 principals (internal-only, walled, cleared), lookup and synthesis | `tests/test_lc_retrieval.py` | every call carries the principal's `permission_filter`, including BM25's row read, the re-score vectors, multi-query's rephrasings, the rerank pool and the parent lookups |
| **Redaction of LangChain's own runs**: every access case, per retriever kind, through a real `langsmith.Client` | `tests/test_langsmith_redaction.py` | 0 leaks. `ArchetypeRetriever`, `BM25Retriever`, `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever` and `ParentDocumentRetriever` all present in the payload, nested under Tessera's traced retrieve step |

Multi-query's leak test uses a scripted "rephraser" that asks, by name,
for three restricted engagements and their facts ("Project Halcyon …
£340m"). The walls hold: the rephrasings run through the same scoped
store.

## 4. Multi-query, live

`TESSERA_LC_RETRIEVER=multiquery tessera eval --check --stack lc` ran on
NIM. Routing and generation were native; the rephrasings came from the
routing model (Nemotron). The run took 95 minutes, with 120 transient
429/503 errors retried. **`=> PASS`.** One ERROR row (`ql039`, retries
exhausted) stays in the export, per the project rule. Re-run alone at the
same commit, `ql039` scored recall 0.67, RR 1.00, groundedness 5 and
relevance 5. Native's score on it is also recall 0.67.

Compared with the P5-4 native sweep (`evals/baselines/p5-4-native.json`)
on the **90 cases scored in both and routed alike** (`ac-a06` re-routed
A→D, the known Cobalt instability):

| | native (P5-4) | multi-query |
|---|---|---|
| A/C recall (53 labelled cases) | 0.9745 | 0.9745 |
| A/C MRR | 0.9717 | 0.9717 |
| A/C precision | 0.362 | 0.355 |
| groundedness (42 judged) | 4.929 | 4.976 |
| relevance (42 judged) | 4.929 | 4.952 |
| cases whose recall or RR changed | — | **0** |
| mean latency per A/C case | 29.3 s | 57.7 s |

- **Multi-query changes no case's retrieval outcome.** The union widens
  the pool, but the cosine re-order and per-document diversification
  bring back the same documents in the same order. The judge means move
  by +0.05 and +0.02. The shown chunks differ slightly, so this is not
  pure judge noise (P5-0 measured A/C relevance noise at ±0.03), but it is
  small.
- **It doubles latency**: three rephrasings, then four retrievals.
- **Its LLM calls aren't metered.** The rephrasings run inside the
  retriever, outside the pipeline's usage recorders, so the sweep's token
  totals even fell (180k → 176k input). On a paid model this cost would be
  invisible. The plan's fix arrives with P5-7: usage recorders passed per
  invocation through LangGraph's runtime context, not bound at build time.
  Until then, this is a known gap of the multi-query retriever.

**Decision:** not a default. It brings no recall gain, doubles latency,
and has unmetered cost.

## 5. Findings worth knowing

- **`MultiQueryRetriever`'s output order means nothing.** It returns the
  union of each query's hits in query order, with the original question
  **last**. Kept as-is, the rephrasings' hits crowded out the original
  question's: authorized recall was 5/12 with the scripted rephraser. For
  a union with no ranking, the contract orders by cosine to the original
  question. A fused (RRF) or reranked order is kept, because that order
  is the point of those generators.
- **Scoped retrieval can't count what it withholds.** Native reports how
  many restricted chunks the permission filter kept out (the operator's
  trace). Counting them means querying outside the principal's scope,
  which the `ScopedStore` never does. For the LangChain retrievers,
  `removed["restricted"]` is always 0.
- **BM25's default tokenizer is `str.split`.** It is case-sensitive and
  keeps punctuation ("Pricing," ≠ "pricing"). Tessera uses lowercase word
  tokens.
- **BM25 and "filter before ranking".** One BM25 index over the whole
  corpus would rank restricted text for a walled user before any
  post-filter ran. Instead, each principal's BM25 ranks only over the
  union of their scopes (internal plus their engagements). The rows are
  read through their `ScopedStore` on every request, and the BM25 build is
  cached per scope set and per **content fingerprint** of those rows (ids,
  text, labels). `invalidate()` drops every index (the review workflow's
  hook, P5-8).
- **Review finding (Risk 1), fixed in this PR.** The first version keyed
  the cache on the chunk count. A document relabelled from internal to
  restricted and re-indexed in the same process keeps its count, so the
  walled user got the stale index, with the restricted text in
  `BM25Retriever`'s own output and trace run. It did not reach the answer:
  `rescore()` re-reads each candidate's vector in scope and dropped it. The
  fingerprint key now rebuilds the index;
  `test_a_relabel_in_the_same_process_rebuilds_the_bm25_index` pins it.
  The fix also caches the built BM25 index, where the count-keyed version
  cached only the rows and rebuilt BM25 on every request.
- **`ParentDocumentRetriever`'s docstore is unfiltered.** `mget(ids)`
  returns whatever it holds. It is safe here only because the ids come
  from child hits that passed the scope. Every parent is re-checked
  anyway. A docstore keyed by a guessable id, and fetched any other way,
  would be a leak.
- **Package reality.** `EnsembleRetriever`, `MultiQueryRetriever`,
  `ParentDocumentRetriever`, `ContextualCompressionRetriever` and
  `CrossEncoderReranker` come from `langchain-classic`, which is in
  maintenance mode. `BM25Retriever` and `HuggingFaceCrossEncoder` come
  from `langchain-community`. They were learned because real codebases
  use them.
- **Deferred: the `Send` fan-out version of multi-query.** It is a
  LangGraph map-reduce (plan §3.7), so it is built and measured with the
  orchestration in P5-7, beside this one.
