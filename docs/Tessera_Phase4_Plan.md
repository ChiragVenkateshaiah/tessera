# Tessera — Phase 4 Plan (Claude Code Brief)

**Status: DRAFT 2026-10-01** — for review; not yet adopted. Until it is,
CLAUDE.md's do-not-build list still applies (no web UI, no AWS). Phase 3
is complete and tagged `v0.3.0`; all 5 Phase 1 exit criteria and every
gated A/C and B bar threshold hold (`checkpoint.md`, Phase 3 close).
Companion documents: `Tessera_Solution_Design.md` §4 (AWS target),
`docs/adr/0002`–`0005` (production direction), `Tessera_Phase3_Plan.md`
(the model for this document's shape).

The four design decisions this plan depends on were made by the user
before drafting (2026-10-01); §8 keeps the record.

---

## 0. Context in one paragraph

Phases 1–3 built and evaluated Tessera as a local CLI: archetypes A and C
over the pilot corpus, B over the synthesized expertise dataset, D as a
refusal, all held to a documented quality bar. Nobody but the developer
can try it. Phase 4 makes it *showable*: a local browser chat UI over the
existing pipeline, then a **deliberately ephemeral** AWS deployment of
that same app (stand up → verify → record a demo → tear down), with the
LLM swapped to Claude on Bedrock behind the existing `LLMClient` port. It
is a portfolio project with no users, so an always-on deployment would
only burn money while idle; the deployment is designed to be created and
destroyed in one command each, and the phase is not done until the
teardown is verified.

## 1. Objective and boundaries

**Objective:** a local chat UI over `pipeline.answer_query()`; Claude on
Bedrock as a config-selectable `LLMClient` that passes the quality bar;
the UI + API packaged as one container and deployed to AWS with
Terraform; a recorded demo of the deployed app; every AWS resource
destroyed afterwards, verified.

**In scope:**
- A thin HTTP layer (`src/tessera/api.py`, FastAPI) and `tessera serve`.
- A static chat page (plain HTML/CSS/JS, no Node toolchain).
- `generation/bedrock.py` — `BedrockClient(LLMClient)`.
- A container image runnable both locally (`docker run`) and on Lambda.
- Terraform under `infra/` for a minimal, tagged, destroyable stack.
- A deploy → demo → destroy runbook, executed once.

**Explicitly NOT in Phase 4** (moves to, or stays on, the do-not-build
list):
- An always-on deployment, or anything left running after the demo.
- CI/CD, monitoring dashboards, alerting beyond a budget alarm — Phase 5.
- The Go edge layer, DynamoDB session state, Kubernetes (ADRs 0002–0004)
  — documented direction, not built.
- OpenSearch Serverless, Bedrock Titan embeddings, S3-hosted corpus —
  documented, not built (§8 decision 3 explains why).
- Conversation memory / multi-turn context. Each question is answered
  independently, as `tessera chat` does; the UI shows a scrolling
  history but sends only the current question.
- Real authentication, SSO, access control — the corpus and dataset stay
  low-sensitivity by construction. A shared demo token (§3.5) is a cost
  guard, not security.
- Archetype D beyond its refusal; real HR integration; deck ingestion —
  unchanged from Phases 1–3.

## 2. What does not change

`router.py`, `retriever.py`, `retrieval/expertise.py`, `generation/`
prompts, `pipeline.py`, the corpus, the expertise dataset, and the
embedder (`all-MiniLM-L6-v2`) are untouched. The only new code on the
query path is an `LLMClient` implementation — exactly the swap the ports
were built for. `api.py` is a composition root like `cli.py`: it builds
the dependencies and calls `answer_query()`; the core stays pure
(CLAUDE.md constraint #6).

## 3. Architecture

### 3.1 HTTP layer (`src/tessera/api.py`)

- `POST /api/ask` `{"question": str}` → JSON form of `AnswerResult`:
  archetype, answer, citations (marker, title, heading path, document
  path), experts (name, title, practice, office, `last_updated`,
  evidenced flag, evidence lines), plus `latency_s`.
- `GET /api/health` → index counts + LLM provider name (no secrets).
- `GET /` → the static chat page.
- Dependencies built once at startup (indexes, embedder, LLM client),
  as `tessera chat` does. Errors return a JSON error body, never a stack
  trace.
- Locally: `tessera serve [--host 127.0.0.1] [--port 8000]` (uvicorn).
- `render_answer()` stays the CLI's text form; the API returns
  structured data and the page renders it.

### 3.2 Chat page (`src/tessera/web/`)

Plain `index.html` + `app.js` + `styles.css`, shipped as package data so
the container picks them up. Shows the question history; per answer: an
archetype badge (A lookup / B expertise / C synthesis / D declined),
the answer with numbered citation markers, a Sources list or People
cards (with "self-reported only" flags and the snapshot date), and
latency. A visible "each question is answered on its own" note so the
missing memory isn't mistaken for a bug. Works at phone width. Markdown
in answers is rendered with a small, pinned, sanitising library or a
minimal renderer — no raw HTML injection from model output.

### 3.3 Claude on Bedrock (`src/tessera/generation/bedrock.py`)

- `BedrockClient(LLMClient)` using the Anthropic SDK's
  `AnthropicBedrockMantle(aws_region=...)` client; credentials come from
  the AWS credential chain (local profile; the Lambda execution role in
  AWS), never from `.env` values in code.
- Model by config: `BEDROCK_MODEL`, default `anthropic.claude-opus-5-5`.
  A cheaper Sonnet/Haiku ID is a one-line config change and the user's
  call (§9).
- Thinking can't be disabled on Opus 5.5; depth is set with
  `output_config.effort`. The router call is a short classification →
  `low`; answer generation → `medium` (the model's default), revisited
  only if the bar says so.
- Wrapped in the existing `RetryingLLMClient` (429/5xx backoff) with the
  SDK's own retries set to 0, as `NvidiaClient` already is.
- Selection by config: `TESSERA_LLM_PROVIDER=nvidia|bedrock` (default
  `nvidia`, so nothing changes until it's set). `cli.py` and `api.py`
  share one factory.

### 3.4 Judge stays fixed

The eval harness currently uses one `LLMClient` for answering *and*
judging. Swapping to Bedrock would silently change the judge too, and
every groundedness/relevance number would stop being comparable with
Phases 2–3. P4-3 adds an optional separate judge client
(`run_harness(..., judge_llm=...)`, CLI `--judge-provider`), and the
Bedrock bar check runs **answers on Claude, judge on Nemotron** — only
one variable changes.

### 3.5 Packaging and AWS shape

```
browser ──HTTPS──▶ Lambda Function URL ──▶ Lambda (container image)
                                              ├─ FastAPI app (UI + /api)
                                              ├─ Chroma indexes (baked in)
                                              ├─ MiniLM weights (baked in)
                                              └─▶ Bedrock (Claude) via IAM role
```

- **One Lambda, container image from ECR**, run through the AWS Lambda
  Web Adapter so the same FastAPI app runs unmodified locally and on
  Lambda. ~3 GB memory; image size is dominated by CPU torch.
- **Lambda Function URL, not API Gateway.** API Gateway caps the
  integration at ~30 s; answers take 3–60 s (NIM has spiked to 200 s).
  Function URLs allow the Lambda's full timeout (set 120 s).
- **Indexes and model weights baked into the image at build time**
  (`tessera ingest` + `index-people` run in the Dockerfile; the
  HuggingFace cache pinned to an image path). No runtime download, no
  S3 dependency. Lambda's filesystem is read-only outside `/tmp`, so
  the Chroma directory is copied to `/tmp` on cold start if Chroma needs
  to write its SQLite files — verified in P4-4, not assumed.
- **Cost guards:** a shared demo token checked by the app (sent as a
  header by the page, entered once), reserved concurrency of 2, an AWS
  Budgets alarm, CloudWatch log retention of 1 day, and every resource
  tagged `project=tessera`, `ephemeral=true`.
- **No secrets needed in AWS:** Bedrock auth is the execution role,
  scoped to `bedrock:InvokeModel*` on the chosen model only. The NVIDIA
  key is never deployed.

### 3.6 Terraform (`infra/`)

ECR repository, Lambda function (image), Function URL, IAM role +
least-privilege policy, log group, budget alarm. Remote state is not
needed for a single developer; local state is gitignored. The image is
built and pushed by a small script that Terraform's apply depends on, or
in a documented two-step (push, then apply) — decided in P4-5. Teardown
is `terraform destroy` plus deleting the ECR images, and is **verified**
with a tag query returning nothing.

## 4. Evaluation

- **Local UI/API changes** (P4-1, P4-2) don't touch the query path; they
  need `pytest` (API contract via FastAPI's `TestClient` with fakes),
  not a bar check.
- **Bedrock** (P4-3) touches `generation/` → a full `tessera eval
  --check` with answers on Claude and judge on Nemotron must pass the
  existing bar, A/C and B, before anything is deployed. Prompts were
  tuned on Nemotron; if a threshold fails, prompt changes go through the
  same bar-check discipline (and must keep passing on Nemotron, which
  stays the default provider locally).
- **Deployed smoke test** (P4-6): the 20-question manual set, run
  through the deployed UI, recorded. The full bar sweep is not re-run
  against the deployed URL — the image runs the same code and indexes
  as the P4-4 container, which is smoke-tested locally first.

## 5. Task sequence

### P4-1 — HTTP API + `tessera serve`
`api.py` (§3.1), `fastapi` + `uvicorn` deps, JSON contract tests with
fakes, `tessera serve`. **Acceptance:** `pytest` green; `curl` against a
local `tessera serve` returns a correct A, B and D answer as JSON.

### P4-2 — Local chat UI
`src/tessera/web/` (§3.2) served at `/`. **Acceptance:** the user's
20-question manual test runs through the browser locally; answers,
citations and people match what `tessera chat` prints for the same
questions; page usable at phone width.

### P4-3 — Claude on Bedrock + fixed judge
`generation/bedrock.py`, provider factory + config, separate judge
client in the harness (§3.3, §3.4), `.env.example` updated. Needs AWS
credentials locally and Bedrock model access. **Acceptance:** a full
`tessera eval --check` with `TESSERA_LLM_PROVIDER=bedrock` (judge on
Nemotron) → `=> PASS`; pasted into the PR.

### P4-4 — Container image
`Dockerfile` (CPU torch, Lambda Web Adapter, indexes + model baked in),
`.dockerignore`. **Acceptance:** `docker run` locally serves the UI;
the 20 questions answer correctly through it using Bedrock; image size
and cold-start time recorded.

### P4-5 — Terraform
`infra/` (§3.6), budget alarm, tags, least-privilege IAM. **Acceptance:**
`terraform validate` + `terraform plan` clean and reviewed by the user;
nothing applied yet; the deploy and teardown runbook written.

### P4-6 — Deploy, demo, destroy, close
Apply → smoke test → record the demo → `terraform destroy` → verify
(tag query empty, ECR empty, Function URL dead) → README/checkpoint
updated with what was deployed, for how long, and what it cost.
**Acceptance:** demo recorded against the live URL; teardown verified;
`v0.4.0` tagged.

## 6. Notes and risks

- **Prompts were tuned on Nemotron.** Claude may score differently on
  the same bar — better or worse. P4-3 exists so that's found before
  anything is deployed, not on camera.
- **Cold start.** CPU torch + MiniLM + Chroma in a ~2 GB image may take
  10–20 s to cold start. Mitigation for the recording: warm it with a
  health call first. Provisioned concurrency is not worth paying for in
  a demo.
- **Chroma on a read-only filesystem** (§3.5) — the most likely P4-4
  surprise.
- **A public URL can be found and hammered.** The demo token, reserved
  concurrency of 2 and the budget alarm cap the damage; the stack is up
  for hours, not weeks.
- **Cost estimate (verify before applying).** Lambda, ECR and logs for
  a day are cents. Bedrock: one bar sweep ≈ 140 calls; at first-party
  Opus 5.5 rates ($4 / $20 per MTok) that is single-digit dollars, and
  Bedrock's own pricing applies. The judge stays on the free NIM tier.
- **The LinkedIn framing.** Meridian is fictional; the demo should say
  "simulated engagement, synthetic data" and show the eval gate.

## 7. What changes in CLAUDE.md when this plan is adopted

- Do-not-build list: remove "A web UI — CLI is sufficient"; replace
  "Any AWS deployment, Terraform, CI/CD, or monitoring — Phases 4–5"
  with "an always-on AWS deployment, CI/CD, monitoring — Phase 5+; Phase
  4's AWS stack is ephemeral (deploy → demo → destroy)"; add the §1 NOT
  list's new items (Go edge, DynamoDB sessions, OpenSearch, Titan,
  conversation memory).
- Constraint #5 (local-first): local stays the development and eval
  environment; AWS is used only for the ephemeral demo deployment, and
  Bedrock is the only cloud LLM dependency added.
- Constraint #6: `api.py` named alongside `cli.py` as a composition root
  that may touch transport/config; the query path stays pure.
- Technology table: LLM row gains Bedrock as a selectable provider;
  new rows for HTTP/UI (FastAPI + static page), packaging (container
  image, Lambda Web Adapter) and IaC (Terraform, ephemeral).
- Working conventions: never leave AWS resources running across
  sessions; `/end-day` checks for tagged resources if a deploy happened.
- `docs/` list gains this document as the Phase 4 authority.

## 8. Decisions made before drafting (user, 2026-10-01)

1. **UI stack → FastAPI + plain HTML/JS.** One app serves UI and API
   locally and as a single Lambda; scales to zero; no Node toolchain.
   (Streamlit/Gradio rejected: always-on server on AWS, generic look.)
2. **LLM on AWS → Claude on Bedrock**, behind `LLMClient`, selected by
   config. Proves the port swap; must re-pass the bar. (Keeping NIM in
   AWS rejected: flaky 503s/latency on camera, no Bedrock story.)
3. **Vector store on AWS → Chroma indexes baked into the image**, same
   MiniLM embedder. Zero idle cost, no new adapter, no recalibration of
   the similarity thresholds. (OpenSearch Serverless rejected for this
   phase: bills per OCU-hour from creation, plus an adapter and a
   re-tune; stays documented-not-built.)
4. **IaC → Terraform.** Apply/destroy as single commands, verifiable
   teardown, matches ADR 0005. CI/CD stays out of scope.

## 9. Open items before P4-3 (user)

- AWS account, region (one where the chosen Claude model is available
  on Bedrock), and a budget ceiling for the alarm.
- Bedrock model access enabled for the chosen Claude model in that
  region; AWS CLI credentials configured on this machine.
- Model choice: `anthropic.claude-opus-5-5` (default) or a cheaper
  Sonnet/Haiku ID — a config value, decidable at P4-3.
