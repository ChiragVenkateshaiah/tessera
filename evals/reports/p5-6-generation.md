# P5-6 — Generation, routing, retries, expertise: the evidence

Phase 5 plan §3.5–3.6. The tests are zero-call. The four per-switch
sweeps (§3) and the native sweep ran live on NIM, 2026-10-08/09.
Reproduce with:

```sh
pytest tests/test_lc_generation.py tests/test_langsmith_redaction.py tests/test_resilient.py -q
# one switch on LangChain per sweep, every other generation switch native,
# TESSERA_LC_RETRIEVER=native (identical to native over the lc collection, P5-4/P5-5):
TESSERA_LC_ROUTER=lc        tessera eval --check --stack lc --json evals/baselines/p5-6-lc-router.json
TESSERA_LC_PROMPT_CHAIN=lc  tessera eval --check --stack lc --json evals/baselines/p5-6-lc-prompt-chain.json
TESSERA_LC_MODEL_CLIENT=lc  tessera eval --check --stack lc --json evals/baselines/p5-6-lc-model-client.json
TESSERA_LC_MODEL_CLIENT=lc TESSERA_LC_RETRY=lc \
                            tessera eval --check --stack lc --json evals/baselines/p5-6-lc-retry.json
tessera eval --check --stack native --json evals/baselines/p5-6-native.json
python -m evals.compare_sweeps evals/baselines/p5-6-native.json \
    evals/baselines/p5-6-lc-{router,prompt-chain,model-client,retry}.json \
    --floor-from evals/baselines/p5-{0-native-1,0-native-2,2-native,3-native,4-native,5-native,6-native}.json \
    --out evals/reports/p5-6-generation.json
```

## 1. The design in one paragraph

Four switches, each one layer:
- **`router`**: `with_structured_output(RouteDecision, method="json_schema")`
  on a LangChain chat model. On the native client, the same Pydantic
  model parses the prompt-instructed JSON.
- **`prompt_chain`**: an LCEL chain, built from Runnables on purpose:
  `RunnableParallel` → `RunnablePassthrough.assign(sources=RunnableLambda(format_source_group))`
  → `RunnableBranch` (A → lookup prompt, else synthesis) → model →
  `StrOutputParser`. The relevance floor, the fixed "nothing on that"
  message, the citations and the superseded note stay deterministic,
  outside the model (`finish_answer`, shared with native).
- **`model_client`**: `ChatNVIDIA` (or `ChatGoogleGenerativeAI`) in place
  of `NvidiaClient` (or `GeminiClient`), with parity settings: thinking
  off, no `max_tokens`, the same temperature, a 600 s read timeout. For
  Gemini: `max_retries=0`, `thinking_level="low"`, `location="global"`.
- **`retry`**: `.with_retry()` on translated 429/5xx errors (exponential
  jitter, no `Retry-After`) plus an `InMemoryRateLimiter` for pacing, in
  place of `RetryingLLMClient`. It needs `model_client=lc`.

Expertise (B) gets a `PeopleRetriever` (a `BaseRetriever` returning
`Document`s with `PersonMatch` metadata). The ranking inside is native's
`find_experts`: evidence scoring, intent re-weighting and the two
generation floors have no framework counterpart, so they are reused as
plain functions (plan §3.6).

Usage is metered on every path. A `RecorderCallback` (one per
invocation) adds each LangChain call's tokens **and latency** to the
pipeline's `UsageRecorder`, under the configured model id. The native
client inside an LCEL chain (`TesseraChatModel`) meters through the port.

## 2. Zero-call evidence

| Check | Tests (`tests/test_lc_generation.py` unless named) | Result |
|---|---|---|
| **Prompt identity**: the router, lookup, synthesis and expertise templates render the `prompts.py` text | `test_the_router_template_renders_the_native_prompt`, `test_the_grounded_chain_renders_the_native_prompt`, `test_the_expertise_chain_renders_the_native_prompt` | identical, literal braces escaped |
| **Fake chat model**: the LCEL chain and native generation give the same answer, citations and superseded note; no call when nothing is relevant; the native client is metered inside a chain; native-client routing matches native | `test_the_lcel_chain_matches_native_generation`, `test_no_relevant_source_is_the_fixed_message_without_a_call`, `test_the_lcel_chain_meters_native_client_calls_into_the_recorder`, `test_the_router_on_the_native_client_matches_native_routing`, `test_an_unparseable_route_is_a_routing_error_on_both`, `test_the_expertise_chain_matches_native`, `test_people_retriever_returns_native_find_experts` | pass |
| **Model parity**: `ChatNVIDIA` sends what `NvidiaClient` sends; routing is `response_format: json_schema` with no forced tool call; the Gemini factory's parity fields; reasoning tokens counted once | `test_chatnvidia_sends_what_nvidia_client_sends`, `test_structured_routing_on_chatnvidia_is_json_schema_and_metered`, `test_the_gemini_factory_has_native_parity_fields`, `test_reasoning_tokens_are_counted_once` | pass |
| **Fault injection**: scripted 429/503/timeout sequences through both retry layers | `test_native_retries_translate_and_honour_retry_after`, `test_lc_with_retry_recovers_from_transient_errors`, `test_a_client_error_is_raised_at_once_by_both`, `test_lc_with_retry_gives_up_after_six_attempts`, `test_a_gemini_429_hidden_on_the_cause_is_found`, `test_chatnvidia_errors_carry_a_status_the_retry_layer_reads`, `test_a_client_side_read_timeout_is_retried_on_both_paths`, `test_a_real_client_error_is_still_not_retried`; adaptive pacing in `tests/test_resilient.py` | pass |
| **Runnables exercised**: `.batch`, `astream_events` | `test_batch_and_astream_events` | pass |
| **Redaction of LangChain's own LLM, prompt and parser runs**: every access case, through a real `langsmith.Client` | `tests/test_langsmith_redaction.py::test_langchain_generation_runs_are_redacted_too` | 0 leaks |

Full suite at the branch head: 610 passed, 8 skipped, 1 xfailed.

## 3. The live per-switch sweeps

Each sweep put one switch on LangChain and held the others native, on
NIM (Nemotron answers and judge). Baseline: the native sweep at
`ea1dbf9`, which includes P5-6's changes to shared code
(`generation/answer.py`'s `citations_for`/`finish_answer`, the
`UsageRecorder`). Comparisons use only the cases scored in both
(CLAUDE.md's ERROR-row rule).

**The noise floor** is the largest judge-mean difference over every pair
of the seven native sweeps P5-0…P5-6: A/C relevance ±0.03, A/C
groundedness ±0.06, B relevance ±0.12, B groundedness ±0.00.

| Sweep | `--check` | Duration | Retried 429/503 | ERROR rows (all scored on re-run) |
|---|---|---|---|---|
| native, 4 workers (`ea1dbf9`) | PASS | 2 h 44 min | 303 | 11 |
| `router` (`6c61734`) | PASS | 46 min | — | 0 |
| `prompt_chain` (`6c61734`) | PASS | 2 h 51 min | 176 | 0 |
| `model_client` (`6c61734`) | PASS | 3 h 39 min | — | 5 (2 read timeouts, fixed in `ea1dbf9`; 3 × 503 past six attempts) |
| `retry` (`e3b1c67`) | PASS | 56 min | judge: 19; LangChain models: not counted (§5) | 2 (`ql001` 503, `ql042` 429: `.with_retry()` gave up after six attempts) |

Against the native sweep, on the cases scored in both (`compare_sweeps`
output; JSON in `evals/reports/p5-6-generation.json`):

| Sweep | Scored in both | Routing Δ | Docs Δ | People Δ | Recall/RR Δ | A/C ground. | A/C relev. | B ground. | B relev. | Leaks | Injection fails | Tokens in/out | Outside floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `router` | 81 | 2 | 2 | 0 | 2 | 4.97→4.91 | 4.97→4.91 | 5.00→5.00 | 4.78→4.89 | 0 | 0 | 154,501/26,226→159,001/26,309 | A/C relevance |
| `prompt_chain` | 81 | 1 | 1 | 0 | 2 | 4.97→4.94 | 4.97→4.94 | 5.00→5.00 | 4.78→4.78 | 0 | 0 | 154,501/26,226→156,647/26,851 | none |
| `model_client` | 76 | 1 | 1 | 0 | 0 | 4.97→4.97 | 4.97→5.00 | 5.00→5.00 | 4.75→4.75 | 0 | 0 | 146,360/24,638→148,714/24,195 | none |
| `retry` | 79 | 0 | 0 | 0 | 0 | 4.97→4.97 | 4.97→4.94 | 5.00→5.00 | 4.75→4.75 | 0 | 0 | 150,354/25,527→150,354/24,465 | none |

The ERROR rows stay in the exports. Re-run alone through the same path
at the same commit, every one scored:
- `model_client` (`6c61734`): `ac-l10` no leak; `ac-i04` contract held;
  `nm004` no-match correct; `q005` recall 0.60, RR 1.00, G5/R5; `ql018`
  person recall 1.00, G5/R5.
- `retry` (`e3b1c67`): `ql001` recall 1.00, RR 1.00, G5/R5 (native: the
  same); `ql042` person recall 0.80, G5/R5 (native: the same).
- native (`ea1dbf9`): all 11 scored (commit `e3b1c67`'s message has the
  list); `ql026` needed a second re-run.

**Reading it:**
- **Every sweep passes the bar, with 0 leaks and 0 injection failures.**
- **No switch changes a retrieval outcome.** The only routing and
  document changes are `ac-a06` and `ac-l06`, the known Cobalt cases.
  Across the seven native sweeps each of them flips between A and D
  (P5-0: `ac-l06` D, P5-2: `ac-a06` D, P5-6: both D), so a flip is not a
  switch effect. The recall/RR changes are those same two cases being
  answered.
- **`prompt_chain`, `model_client` and `retry` are within noise** on
  every judge mean (`retry`'s A/C relevance −0.03 is one case, `q005`
  5→4, at the floor).
- **`router` shows A/C relevance −0.06 and groundedness −0.06**, outside
  the relevance floor. Two cases cause it: `ql017` (G5/R5 → G3/R4) and
  `fr005` (R5 → R4). The router can't reach those answers: both cases
  routed C in both sweeps, retrieved the same five documents in the same
  order, and were answered by the native generator on the same model.
  What differs is one sampled answer each, and the judge's reading of it.
  On `ql017` the `router` sweep's judge says the three strategic shifts
  were "invented", and the `prompt_chain` sweep's judge marks a near-
  identical answer down for "not enumerating" the same three shifts.
  **This is answer-sampling noise that the floor underestimates**, not a
  router effect. Seven native sweeps don't bound a two-case swing on a
  35-case mean.
- **Tokens:** the small rises (+1–3 % input) come from the Cobalt cases
  answered (an A answer instead of a D refusal), not from the prompts,
  whose identity is tested.
- **Latency can't be compared from these exports** (§5).

**Reading the `retry` sweep.** NIM was calm on 2026-10-09's afternoon
(56 min, against 2–3 h on 10-06…10-08), so it is the gentlest test the
switch has had. Even so, the two retry layers in the same sweep behaved
differently. The native judge (`RetryingLLMClient`, shared adaptive
pacer) was throttled 19 times and recovered every time. The LangChain
answer models lost two cases: `.with_retry()`'s backoff totals about
31 s over six attempts and ignores `Retry-After`, so a burst of 503s or
a 429 cool-down outlasts it. Native's 15 s/45 s steps (cap 120 s) wait
out the same bursts. One calm sweep can't give rates, and the LangChain
retries weren't counted, but the direction matches the mechanism.

## 4. What the `retry` switch compares

| | native (`RetryingLLMClient`) | `lc` (`.with_retry()` + `InMemoryRateLimiter`) |
|---|---|---|
| Transient errors retried | 429, 5xx, client-side timeouts (as 504) | the same, after translation to `TransientLLMError` |
| Backoff | 45 s (429) / 15 s (5xx), exponential, cap 120 s | tenacity's exponential jitter: 1, 2, 4, 8, 16 s (+ up to 1 s), about 31 s over six attempts |
| `Retry-After` | honoured | ignored |
| Attempts | 6 | 6 |
| Pacing in a sweep | one adaptive `Pacer` (3–30 s, AIMD), shared with the judge | a fixed 1 call / 3 s token bucket; the judge keeps its own pacer |
| Retries visible | each one printed and counted | silent, uncounted |
| Parallel cases (`--workers`) | allowed | refused |

**Not yet run:** the 2026-10-08 plan's `lc-model-client-2` sweep
(`model_client=lc` again, with the timeout fix and adaptive pacing). It
differs from the `retry` sweep only in the retry layer, so it is the
clean comparison for this table; until it runs, `retry` is read against
the native sweep.

## 5. Findings worth knowing

- **`ChatNVIDIA` timed out where native waits.** Its default read
  timeout is 60 s, and a `requests` `ReadTimeout` carries no HTTP status,
  so neither retry layer retried it: two `model_client` cases failed at
  once. Native's openai SDK reads for 600 s and retries timeouts twice
  itself. The sweep was partly measuring a different timeout policy.
  Fixed in `ea1dbf9`: `timeout=600`, and a client-side timeout or dropped
  connection now reads as 504 on both retry paths.
- **More workers don't help a throttled NIM.** The native sweep with 4
  workers took 2 h 44 min (303 retries, 11 ERROR rows), against 2 h 48
  min (214 retries, 3 ERROR rows) one at a time on 2026-10-07. A fixed 3
  s interval can't find NIM's real capacity, so the extra workers only
  met more 429/503s. `--workers` now defaults to 1, and every client in
  a sweep shares one adaptive pacer (`50e818a`).
- **The `lc` retry value runs two pacers against one rate limit.** The
  LangChain models pace with their `InMemoryRateLimiter`; the native
  judge keeps the shared pacer. That is what the switch measures, so it
  stays, but it is why `--workers > 1` is refused with it.
- **`.with_retry()` retries are invisible.** LangChain gives no hook for
  them, so the `retry` sweep reports ERROR rows but no retry count. A
  counter in the translating wrapper (`with_lc_retries`) would fix it
  without changing behaviour; it is a follow-up, not part of this sweep.
- **Latency per call isn't in the exports.** `latency_seconds` per case
  is wall time, including pacing, the 429/5xx backoff and (with
  workers) the wait for a slot: native shows 285 s per case at 4
  workers. Plan §4.2 asks for per-call latency excluding pacing and
  backoff. `Usage.latency_s` records exactly that on both stacks, but the
  harness doesn't export it. Exporting the case's summed call latency
  is the fix, before P5-7's latency comparison.
- **`with_structured_output(include_raw=True)` raises on `ChatNVIDIA`**
  (P5-1 spike), and isn't needed: the `RecorderCallback` meters the
  routing call's usage. The routing test proves the request is
  `response_format: json_schema` with no forced tool call.
- **Deferred to P5-7, per the plan:** the `RunnableBranch` vs LangGraph
  conditional-edge comparison, LangGraph's `RetryPolicy`, and time to
  first token under streaming (needs the graph's streaming path; it
  informs whether Phase 6's chat UI streams).

## 6. Decision (`lc-defaults`)

**Proposed, for the user's decision** (the profile P5-7's `lc-defaults`
sweep runs):
- `TESSERA_LC_ROUTER=lc`, `TESSERA_LC_PROMPT_CHAIN=lc`,
  `TESSERA_LC_MODEL_CLIENT=lc`: each is within noise (the `router`
  relevance dip is answer sampling, §3), so the LangChain layer is
  chosen, as P5-5 chose the LangChain vector retriever at native-equal
  quality.
- `TESSERA_LC_RETRY=native`: `RetryingLLMClient` around the LangChain
  chat models. It honours `Retry-After`, waits out NIM's bursts, counts
  and prints its retries, shares the sweep's adaptive pacer with the
  judge, and allows `--workers`. LangGraph's `RetryPolicy` gets its own
  comparison in P5-7.
- `TESSERA_LC_EXPERTISE` stays as configured until the user decides:
  `PeopleRetriever` wraps native's `find_experts` and returns the same
  people (`test_people_retriever_returns_native_find_experts`), and the
  plan budgets no separate expertise sweep. It gets measured in P5-7's
  `lc-defaults` and `all-lc` sweeps either way.
