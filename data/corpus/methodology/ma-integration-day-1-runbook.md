---
title: "M&A Integration: Day 1 Runbook (T-30 to T+7)"
doc_type: methodology
industry: cross-industry
topics: [ma-integration, day-1, integration-management]
date: 2025-08-04
---

## Overview

The Day 1 readiness page covers what has to be true before close. This runbook covers what actually happens, in what order, across the five workstreams that carry Day 1 execution. It assumes an IMO is already stood up and a Day 1 readiness assessment has been completed.

Every step below carries an owner and, where relevant, a dependency. A step whose dependency has not cleared does not start early on the basis that it is probably fine — that is the single most common cause of a Day 1 incident.

## When to Use It

From roughly thirty days before expected close through the first week of combined operation. Adapt the timeline for deals with regulatory conditions that make the close date uncertain: in those cases anchor the sequence to "close minus N" rather than to calendar dates, and rehearse the compressed version.

## The Runbook

### T-30: Lock and Freeze

The purpose of this phase is to stop the target state moving. Anything still being designed at T-30 will not be ready.

#### IT

1. Freeze all non-essential change in both estates. Owner: IT integration lead. Emergency changes require IMO sign-off.
2. Complete the Day 1 systems inventory — what must work at close, what can wait. Owner: IT integration lead.
3. Confirm network interconnect design and order any circuits. Dependency: legal confirmation that pre-close connectivity is permitted under the interim operating covenants.
4. Build and test the read-only directory sync in a staging environment. No production sync before close.

#### HR

1. Finalise the Day 1 organisation structure to at least two levels below the executive team. Owner: HR integration lead.
2. Confirm employment transfer mechanics and any works council or employee representative consultation status. Dependency: legal.
3. Prepare individual communications for anyone whose reporting line changes at close.
4. Confirm payroll continuity — which entity runs the first post-close payroll, and on what cycle.

#### Finance

1. Agree the Day 1 chart of accounts mapping. Full harmonisation is a later programme; Day 1 needs consolidation to work, not elegance.
2. Confirm banking arrangements, signatories, and payment authority limits for the combined entity.
3. Lock the opening balance sheet process and who signs it.

#### Communications

1. Draft and approve the Day 1 message set: employees, customers, suppliers, and any regulator that requires notification.
2. Prepare the leader briefing pack and rehearse it with every people manager who will deliver it.

#### Legal and Regulatory

1. Confirm all conditions precedent status and the realistic close date. Owner: deal counsel.
2. Confirm which regulatory notifications are pre-close, at-close, and post-close.

### T-7: Rehearse

The purpose of this phase is to find what is broken while there is still time to fix it.

#### IT

1. Run a full cutover rehearsal against a non-production copy, including rollback. Owner: IT integration lead.
2. Confirm the incident escalation path and staffing for the close weekend, by name and by hour.
3. Verify that every Day 1 critical system has a named owner reachable during the cutover window.

#### HR

1. Confirm that every employee record required for Day 1 access provisioning is complete and reconciled between the two estates.
2. Brief people managers. A manager who first reads the Day 1 message at the same time as their team cannot answer questions, and unanswered questions on Day 1 drive attrition risk among exactly the people the deal needs to retain.

#### Finance

1. Dry-run the first consolidated reporting pack using the agreed mapping.
2. Confirm supplier payment continuity — identify any supplier whose contract terminates or reprices on change of control.

#### Communications

1. Final approval of all messages, with a holding statement prepared for the scenario where close slips.

#### Legal and Regulatory

1. Confirm signing and closing logistics, including time zones for any multi-jurisdiction signing.

### T-1: Go/No-Go

A single decision forum, chaired by the IMO lead, with one representative per workstream. Each workstream reports green, amber, or red against its Day 1 critical list. The rule is explicit: any red on a Day 1 critical item defers the associated activity, not the close — the close is a legal event and does not wait for integration readiness.

Record the decision and the rationale. A deferred activity leaves the forum with a named owner and a new date.

### Day 1: Execute

#### Hours 0-4

1. Legal close confirmed and communicated to the IMO. Nothing below starts before this confirmation.
2. Employee message released, timed so that no employee learns of the close from an external source first.
3. Customer and supplier notifications issued.
4. Access provisioning begins for the Day 1 critical population only. Dependency: legal close confirmation.

#### Hours 4-12

1. People manager briefings delivered across all time zones.
2. Command centre opens. Every workstream staffs it for the full first day.
3. First reconciliation of access provisioning — who was meant to have access, who actually does.

#### Hours 12-24

1. First incident review. Categorise, assign, and set resolution expectations.
2. Confirm payroll systems are reading correct data ahead of the first cycle.
3. Executive readout: what worked, what broke, what changes for day two.

### T+7: Stabilise and Hand Over

1. Close out or formally accept every Day 1 incident. Owner: IMO lead.
2. Reconcile deferred activities from the T-1 forum against their new dates.
3. Complete the first synergy baseline measurement, so that later tracking has a genuine starting point rather than a retrospective estimate.
4. Transition from Day 1 command centre operation into the standing integration workstream cadence.
5. Run a short retrospective while memory is fresh, and update this runbook.

## Common Pitfalls

- **Starting provisioning before legal close confirmation.** Convenient, occasionally catastrophic, and a compliance exposure in regulated sectors.
- **Treating the T-1 forum as a status update.** It is a decision forum. If nothing can be deferred at it, it is not doing its job.
- **Declaring success at the end of Day 1.** Most Day 1 failures surface in the first payroll cycle and the first month-end close, both of which fall after the command centre has usually stood down.
- **Rehearsing only the happy path.** The rehearsal that matters is the rollback.

## Related Frameworks

M&A integration — Day 1 readiness (parent), IMO setup, synergy quantification, change management — communication planning, operating model design — RACI and governance.
