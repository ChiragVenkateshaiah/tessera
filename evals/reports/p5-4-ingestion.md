# P5-4 — LangChain ingestion and indexing: the evidence

Phase 5 plan §3.3. Every number here is from a zero-LLM-call run on
2026-10-07 at the P5-4 branch. Reproduce with:

```sh
python -m evals.compare_ingestion --out evals/reports/p5-4-ingestion.json
pytest tests/test_lc_ingestion.py tests/test_stale_chunks.py -q
```

## 1. Splitter and embedding-prefix variants (retrieval-only)

Each variant builds a fresh index of the real corpus. The index uses
native delete-then-add, the native store and the native embedder, so the
splitter and the embedding prefix are the only things that change. Then
the real `retrieve()` runs over every A/C case with its archetype forced.
The scoring is the eval harness's: unique documents by rank, recall@5 and
reciprocal rank. The chunk size was **chosen on `query_log.yaml` only (31
cases)**. The **held-out `placeholder.yaml`** set has only 4 cases, so one
relevant document there moves recall by 0.05.

| Variant | Chunks | Words med / p90 / max | Unbalanced fences | Headerless table pieces | Tuned recall | Tuned MRR | Tuned misses | Held-out recall | Held-out MRR | Held-out misses |
|---|---|---|---|---|---|---|---|---|---|---|
| native | 427 | 61 / 158 / 270 | 0 | 0 | 0.959 | 0.968 | 0 | 0.900 | 0.875 | 0 |
| lc 400 | 806 | 42 / 59 / 124 | 6 | 13 | 0.943 | 0.968 | 0 | 0.850 | 0.875 | 0 |
| lc 800 | 504 | 63 / 107 / 188 | 2 | 4 | 0.959 | 0.968 | 0 | 0.850 | 0.875 | 0 |
| lc 1200 | 437 | 61 / 148 / 312 | 2 | 1 | 0.959 | 0.968 | 0 | 0.850 | 0.875 | 0 |
| lc 1600 **(chosen lc)** | 407 | 60 / 176 / 312 | 2 | 2 | 0.959 | 0.968 | 0 | 0.900 | 0.875 | 0 |
| lc 800 keep headers | 514 | 65 / 109 / 190 | 2 | 4 | 0.959 | 0.968 | 0 | 0.850 | 1.000 | 0 |
| native (no prefix) | 427 | 61 / 158 / 270 | 0 | 0 | 0.841 | 0.852 | 2 | 0.800 | 0.875 | 0 |
| lc 400 (no prefix) | 806 | 42 / 59 / 124 | 6 | 13 | 0.841 | 0.856 | 2 | 0.750 | 1.000 | 0 |
| lc 800 (no prefix) | 504 | 63 / 107 / 188 | 2 | 4 | 0.833 | 0.852 | 2 | 0.750 | 0.875 | 0 |
| lc 1200 (no prefix) | 437 | 61 / 148 / 312 | 2 | 1 | 0.858 | 0.852 | 2 | 0.750 | 0.875 | 0 |
| lc 1600 (no prefix) | 407 | 60 / 176 / 312 | 2 | 2 | 0.841 | 0.852 | 2 | 0.800 | 0.875 | 0 |
| lc 800 keep headers (no prefix) | 514 | 65 / 109 / 190 | 2 | 4 | 0.837 | 0.825 | 3 | 0.750 | 0.875 | 0 |

**What it shows:**
- **The embedding prefix is the biggest lever on this page.** The prefix
  puts the title and heading path ahead of the body. Without it, tuned
  recall falls from 0.959 to 0.83–0.86 for every splitter, and 2–3 cases
  miss completely. These are the Phase 2 title-only misses (`ql002`,
  `ql009`) coming back. `embed_prefix` stays **on**.
- **The LangChain splitter ties native at 800–1600 characters, but only
  ties.** At those sizes it matches native on the tuning set. At 400 it
  is worse (recall 0.943, 806 chunks). 1600 was chosen on the tuning set
  (equal scores, fewest chunks), and on the held-out set it again ties
  native (0.900).
- **It breaks structure that native keeps whole.** At every size, it
  leaves code fences unbalanced (a chunk opens a fence and the next chunk
  closes it). It also separates table rows from their header row (13
  pieces at 400). The native chunker never splits inside a table or a
  fence, by design. These pieces still retrieve. The cost is in what the
  answer model reads: a table piece without its header row.
- **`strip_headers=False`** (the heading line kept in the text) changes
  nothing on recall. It folds an empty parent heading into the child's
  text (see §3). `strip_headers=True` is the default, because native's
  stored text has no heading line, and the stored text is what the prompt
  renders.

**Decision (the `lc-defaults` profile):**
- `TESSERA_LC_SPLITTER=native`: equal recall, with no structural damage.
- `TESSERA_LC_CHUNK_SIZE=1600`: the size used whenever the splitter is
  `lc`, as in the `all-lc` profile.
- `TESSERA_LC_EMBED_PREFIX=true`.

## 2. Parity (`tests/test_lc_ingestion.py`)

| Layer | Check | Result |
|---|---|---|
| `loader` | `TesseraCorpusLoader`: metadata, body and order equal to native, document by document. The round trip to the native `Document` is exact. The same `CorpusError` on a bad file. Quarantined documents are held out. | identical |
| `loader` → chunker | native chunks from either loader | identical |
| `embeddings` | `HuggingFaceEmbeddings` vs `LocalEmbedder` over all 427 embedding texts and a query | max abs difference < 1e-5 |
| embedding text | the LangChain store's prefix vs `chunk_embedding_text`, on every chunk | identical |
| `store` | the native `retrieve()` over the LangChain collection vs the native one, for every A/C case: chunk ids, order, text, scores (< 1e-6), removal counts | identical |
| `store` via `index()` | the same comparison over a collection that `index()` wrote | identical, except one superseded-match score (below) |

**One approximate-search difference.** The superseded-document report
comes from a second, filtered nearest-neighbour query. Chroma's HNSW
search is approximate, and `index()` inserts rows in a different order
(content-hash ids, in batches), so its search graph differs. For `ac-l03`
it returned the superseded document's second-best chunk (`::2`, 0.3903),
where native found the best one (`::0`, 0.3924). The document is the
same, the main results are identical, and the score difference is 0.002.
The test compares superseded matches by document, with a 0.01 score
margin.

## 3. Indexing: native delete-then-add vs LangChain `index()` (`tests/test_stale_chunks.py`)

| Scenario | upsert only (pre-P5-4) | native delete-then-add | `index()` `cleanup="full"` | `index()` `cleanup="incremental"` |
|---|---|---|---|---|
| a document shrinks 5 → 3 chunks | **stale tail stays** | clean | clean | clean |
| a document is relabelled (engagement halcyon → kestrel) | — | clean | clean | clean |
| a document is newly quarantined | **stays indexed** | removed | removed | **stays indexed** (strict xfail) |
| a file is deleted from the corpus | stays | **stays** (not loaded, so not deleted) | removed | stays |

**Re-running on an unchanged corpus** (the real index, `tessera ingest`
vs `tessera ingest --stack lc`, wall time including the model load):

| | Embedded | Deleted | Unchanged | Wall time |
|---|---|---|---|---|
| native delete-then-add | 427 | 427 | 0 | 26.0 s |
| LangChain `index()` | 0 | 0 | 427 | 14.5 s |

This is what the record manager buys. Native re-embeds everything on
every run. `index()` hashes each chunk and skips the ones it has already
written. At 427 chunks the saving is seconds. On a real firm corpus it is
most of the ingestion cost.

- **`incremental` cleanup only touches sources present in this run's
  batch.** A newly quarantined document is held out of the batch, so its
  chunks stay searchable. For an anonymized-but-identifiable case study,
  that is the wrong failure. The plan said `incremental`; Tessera uses
  **`full`**, which is correct because every run indexes the whole corpus.
  The test pins the `incremental` gap as a strict xfail.
- **`index()` gives rows content-hash ids**, so the `<stem>::<n>` chunk id
  goes in row metadata. The store adapter reads it from there, so the
  retriever's `_chunk_index` is unchanged. The P5-1 spike's alternative,
  id-keyed rows with `force_update=True`, re-embeds every chunk on every
  run and gives up what `index()` is for.
- **The prefix with `index()`.** `index()` calls the store's
  `add_documents`, which embeds `page_content`. `TesseraChroma` overrides
  `add_documents` to embed the prefixed text and store the body only. This
  is the spike's mechanism (a), placed where `index()` reaches it.
- **Native's remaining gap:** a file deleted from the corpus. Native never
  loads it, so it never deletes it, until the index is rebuilt.

## 4. The native snapshot after delete-then-add

`python -m evals.snapshot --check evals/snapshots/p5-2.json` builds its
index through the same `index_corpus` that `tessera ingest` uses. The
result is **identical (92 cases)**.

## 5. Splitter behaviour worth knowing (from the parity tests)

- The corpus uses `####` headings (the M&A day-one runbook's hour
  blocks). The splitter therefore splits on all six levels, as native
  does. A first version that split only to `###` left `#### Hours 0-4` in
  the chunk text.
- `RecursiveCharacterTextSplitter` cuts fenced code. One chunk of the
  elasticity reference starts with the code comment `# beta_price is the
  elasticity estimate`, because the opening fence went to the previous
  chunk.
- With `strip_headers=False`, an empty parent heading is folded into the
  child section: the text starts `## Approach  \n### Step one`. Native
  drops a heading that has no text of its own.
