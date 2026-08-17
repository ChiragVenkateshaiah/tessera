---
title: "Operating Model Design: Shared Services Design"
doc_type: methodology
industry: financial-services
topics: [operating-model, shared-services, cost-transformation]
date: 2023-03-22
---

## Overview

Shared services consolidates common support functions (finance operations, HR administration, procurement, parts of IT) that were previously duplicated across business units or geographies into a single delivery organization serving all of them. Done well, it's both a cost lever and a service-quality lever; done poorly, it becomes a bureaucratic layer that business units route around.

## When to Use It

Use when a client has meaningful duplication of support-function activity across business units or geographies — a common finding in financial services clients with multiple business lines that each built out their own finance, HR, and procurement teams historically, or in any company that has grown through acquisition without integrating back-office functions.

## Framework

1. **Scope what belongs in shared services.** Not every support activity is a good shared-services candidate — transactional, standardizable activity (payroll processing, accounts payable, tier-1 IT helpdesk) is a strong fit; activity requiring deep business-unit context (strategic finance business partnering, HR business-partner roles) usually should stay embedded in the business.
2. **Design the service catalog and SLAs.** Define what the shared service organization delivers, to what standard, and how performance is measured — without a clear service catalog, business units perceive shared services as a black box and lose trust in it quickly.
3. **Choose a funding model.** Shared services costs can be allocated (charged back to business units on a formula basis) or centrally funded. Chargeback creates better cost discipline and demand signal but adds administrative overhead; central funding is simpler but can lead to over-consumption.
4. **Plan the migration in waves.** Moving all functions into shared services simultaneously is high-risk. Sequence by ease of standardization and value at stake — typically starting with the most transactional, easiest-to-standardize processes to build credibility before tackling more complex ones.

**Worked example.** A financial services group with three distinct business lines (retail banking, wealth management, insurance) each maintaining separate finance-operations teams found through the operating model diagnostic that roughly 60% of finance-operations activity across the three units was transactional and near-identical in process (invoice processing, expense management, reconciliations) despite being run by three separate teams with three separate systems. Consolidating this transactional layer into a single shared finance-operations unit, while leaving strategic finance business-partnering embedded within each business line, delivered a material headcount reduction alongside faster month-end close, because the consolidated team could specialize and standardize in a way three fragmented teams could not.

## Common Pitfalls

- **Over-scoping into shared services.** Pulling business-partnering or judgment-heavy roles into a shared services model damages the business relationship those roles exist to support, and usually gets reversed within a year.
- **No service catalog or SLA discipline.** Without a clear, measured service standard, shared services becomes a cost center nobody can defend and business units lobby to exit it.
- **Big-bang migration.** Attempting to move everything into the new model at once, rather than in sequenced waves, creates service disruption at exactly the moment the new model most needs to prove itself.

## Related Frameworks

Operating model design (parent), organization design principles, cost transformation — SG&A benchmarking (shared services is a primary structural lever for closing an SG&A benchmark gap).
