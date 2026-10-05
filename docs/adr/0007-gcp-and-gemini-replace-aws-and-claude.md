# 0007 — Google Cloud and Gemini replace AWS and Claude on Bedrock

**Status:** Accepted — 2026-10-05 (user decision). Supersedes ADR 0003
as the cloud target, and the AWS-specific parts of ADRs 0004 and 0005.

## Context

From the Solution Design through Phase 4, Tessera's production target was
AWS: Claude on Bedrock as the LLM (built in P4-2 as `BedrockClient`,
unit-tested against the real SDK signature), and an ephemeral Lambda
deployment as the phase after Phase 4. P4-2's acceptance check, a full
eval sweep with answers on Bedrock, stayed open at `v0.4.0` because the
AWS account could not call any Claude model.

Getting it unblocked, 2026-10-02 to 2026-10-05:

1. Bedrock returned `AccessDeniedException: … is not available for this
   account` for Haiku 4.5, Opus 4.8 and Opus 5.5, even after the
   Anthropic use-case form.
2. The account's default payment method was a UPI AutoPay mandate. With
   an Indian debit card added and made the default, Haiku's error changed
   to **"INVALID PAYMENT"**: the model was reachable and payment was the
   blocker. Opus 5.5 still said "not available … contact AWS Sales".
3. The billing console then showed: *"Your payment card issued in India
   might result in failed payments. Customers with cards issued in India,
   with legal entities or billing address registered in other countries
   might face payment failures."* AWS's suggested workarounds were a card
   issued outside India, or paying each invoice manually. The user called
   this the hard stop for AWS.
4. On Google Cloud, which was already billable, Claude Opus 5.5 in Agent
   Platform Studio returned *"You've reached your project's quota for
   Anthropic Claude Opus 5.5"*. New projects get zero quota for partner
   models, and raising it means a quota request that can take days.
   Gemini Flash and Pro answered straight away.

Separately, on the same day, the user confirmed that the cloud showcase
stays **ephemeral**: deploy, record the demo, destroy, verify. An
always-on footprint (OpenSearch Serverless alone is roughly $170+/month)
was rejected on cost.

## Decision

**Google Cloud is Tessera's cloud. Gemini on Agent Platform (formerly
Vertex AI; renamed at Google Cloud Next, April 2026) is the production
LLM.**

- **LLM:** `GeminiClient` (`generation/gemini.py`) behind the existing
  `LLMClient` port, selected by `TESSERA_LLM_PROVIDER=gemini`. The
  routing/answer split carries over from the Bedrock design:
  - `gemini-3.8-flash` routes (Haiku's role);
  - `gemini-3.1-pro-preview` answers (Opus's role);
  - both at thinking level `low`.
  Auth is Application Default Credentials (`gcloud auth
  application-default login` locally, a service account when deployed);
  no key goes in config.
- **Judge:** stays on NVIDIA NIM (free), as before.
- **Project:** a dedicated GCP project (`tessera-510716`). Budget alerts,
  cost reports and the teardown check are all scoped to it, so nothing
  else shares the bill or the cleanup.
- **Deployment, when its phase comes:** Cloud Run plus Artifact
  Registry, provisioned with Terraform's `google` provider. It stays
  ephemeral and pay-per-use: nothing that bills while idle without a
  stated monthly cost and the user's approval. That phase's own plan
  makes the detailed choices (where the index lives, the embedding
  model).
- **CI/CD, when its phase comes:** GitHub Actions authenticating to GCP
  through Workload Identity Federation (no service-account keys in
  GitHub). The eval gate principle of ADR 0005 is unchanged.
- **Bedrock:** `BedrockClient` and its tests stay as a dormant provider.
  It costs nothing to keep, and it documents a second working port
  implementation.

## Consequences

**Positive:**
- Unblocks P4-2's live acceptance sweep and the deployment phase with an
  account that can actually pay.
- The swappable-port design (CLAUDE.md constraint #1) gets a real test.
  Moving providers took one new client, a config switch and a few tests;
  the query path didn't change.
- The answering model (Gemini) and the judge (Nemotron) come from
  different model families, so the judge isn't scoring its own family's
  output.
- A dedicated project makes "verify nothing is left running" a
  single-project check rather than a tag filter across a shared account.

**Negative:**
- The answer model is a **preview** (`gemini-3.1-pro-preview`, the only
  Gemini 3 Pro offered). Preview models can change or be withdrawn, so
  the id is config, and the eval gate catches a quality change.
- **Flash pricing is introductory**: $0.75 / $3.75 per million tokens
  until 2026-12-31, then $1.50 / $7.50. Cost figures quoted after that
  date must use the new rate.
- Gemini 3 is left at its default temperature (Google advises against
  lowering it), so routing is less repeatable than the 0.0 Tessera asked
  for elsewhere. The sweep's routing accuracy is the check on that.
- The design docs and ADRs 0002–0005 were written for AWS. They're kept
  as the record of what was planned, with this ADR as the pointer, rather
  than rewritten.

## Alternatives Considered

- **Stay on AWS with a credit card, or a card issued outside India.**
  This was the next step on the user's own fallback list, and it may well
  work. The user chose to stop on AWS at the billing banner rather than
  keep trying payment methods.
- **Claude on Agent Platform** (Google Cloud's Model Garden). Keeps
  Claude, but needs a partner-model quota increase first. It stays a
  config-level option for later: a quota request, plus an Anthropic client
  aimed at Agent Platform.
- **NVIDIA NIM as the production LLM.** Free, and it already passes the
  bar, but it doesn't use a cloud provider's model platform, and its
  free-tier throttling makes it a poor serving dependency.
- **The Gemini Developer API with an API key.** Simpler auth, but a
  long-lived key instead of cloud IAM, and outside the GCP project's
  billing, quota and audit trail.
