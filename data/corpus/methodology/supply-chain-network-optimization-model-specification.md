---
title: "Supply Chain: Network Optimization Model Specification"
doc_type: methodology
industry: industrials
topics: [supply-chain, network-optimization, modeling, scenario-analysis]
date: 2026-01-20
---

## Overview

The network optimization methodology page covers when a network study is the right tool and how to frame it with a client. This page is the model specification: what the model contains, what data it needs, how scenarios are constructed, and — the section most often skipped — how to read the output without over-trusting it.

A network model is a decision-support tool, not a decision. It will always produce an answer. Whether that answer is worth acting on depends almost entirely on the quality of the constraint set and the honesty of the cost data, both of which are addressed below.

## When to Use It

Use this specification when scoping or reviewing a network optimization model — typically in footprint rationalisation, reshoring and nearshoring evaluations, distribution network redesign, and post-merger network consolidation. It assumes a mixed-integer programme formulation, which covers the large majority of the network work we do.

It is not the right specification for pure inventory optimization (a different problem class, addressed by the omnichannel inventory work) or for transport route optimization at the operational level.

## Framework

### Model Formulation

#### Decision Variables

The variable set determines what questions the model can answer. Adding variables is cheap analytically and expensive practically — every additional variable is a data requirement.

- **Binary facility variables.** For each candidate site, whether it is open. This is what makes the problem mixed-integer and what makes it slow; keep the candidate set disciplined.
- **Continuous flow variables.** Volume of each product family moving from each source to each destination in each period.
- **Binary sourcing variables.** Whether a given customer or region is served from a given facility, where single-sourcing is required.
- **Continuous capacity variables.** Where capacity can be added incrementally rather than only through opening or closing a site.
- **Slack variables.** Explicit unmet-demand variables with a high penalty cost. Include these even when unmet demand is unacceptable — an infeasible model tells you nothing, whereas a model that reports where it had to leave demand unmet tells you exactly which constraint is binding.

#### Objective Function

```
minimise:
      SUM over sites        [ fixed_cost(s) * open(s) ]
    + SUM over site,product [ variable_prod_cost(s,p) * produced(s,p) ]
    + SUM over lanes        [ freight_rate(o,d,p) * flow(o,d,p) ]
    + SUM over lanes        [ duty_rate(o,d,p) * value(p) * flow(o,d,p) ]
    + SUM over sites        [ inventory_holding_cost(s) * avg_inventory(s) ]
    + SUM over sites        [ amortised_transition_cost(s) * status_change(s) ]
    + SUM over demand       [ unmet_penalty * shortfall(d,p) ]
```

Two terms deserve attention because they are the ones most often omitted. Transition cost — the one-off cost of opening, closing, or repurposing a site, amortised across the model horizon — is what stops the model recommending a churn of the network every period. Duty and tariff cost belongs in the objective rather than being applied afterwards, because it changes the optimal flow pattern rather than just the total.

#### Constraint Families

| Family | Form | Typical failure if omitted |
|---|---|---|
| Capacity | Flow out of a site ≤ capacity × open flag | Model loads infeasible volume into the cheapest site |
| Demand satisfaction | Inflow to each demand point + shortfall = demand | Model silently under-serves |
| Service level | Flow only permitted on lanes within a transit-time bound | Recommends a footprint that cannot hit committed lead times |
| Single sourcing | Sum of sourcing binaries per customer = 1 | Splits a customer across three sites, which operations will not run |
| Minimum viable scale | Site volume ≥ threshold × open flag | Opens a site to carry trivial volume |
| Capability | Product families restricted to qualified sites | Assigns regulated or specialised production to a site that cannot make it |
| Balance | Inflow = outflow at every intermediate node | Inventory appears from nowhere |

The capability constraint is the one clients most often forget to mention and most often care most about. Ask explicitly, product family by product family, which sites are qualified and what qualification would cost and take.

### Data Requirements

#### Required Inputs and Quality Thresholds

- **Demand.** By product family, destination region, and period. At least twenty-four months of history. Where the model horizon extends beyond two years, the demand forecast becomes the dominant source of error and should be treated as a scenario variable, not an input.
- **Freight rates.** By lane and mode. Contracted rates where available; indexed market rates otherwise, with the index and date recorded.
- **Facility cost.** Fixed and variable, separated. The single most common data problem is a client cost allocation that buries genuinely fixed cost inside a per-unit rate, which makes closing a site look less attractive than it is. Rebuild the fixed/variable split from source rather than accepting the management accounting view.
- **Capacity.** Practical rather than nameplate, with the basis stated — shifts, uptime assumption, and changeover allowance.
- **Duty and tariff rates.** By origin, destination, and commodity classification, with the date of the rate schedule recorded.
- **Transition cost.** Opening, closing, and requalification, including severance, decommissioning, and any customer requalification requirement.

#### Data Quality Gates

Do not run the model until all four pass:

1. Reconstructed total network cost from model inputs sits within roughly 5% of the client's reported total.
2. Demand history reconciles to reported revenue by region.
3. Every lane in the flow data has a rate, with no default rate applied to more than a small fraction of volume.
4. Capacity figures have been confirmed by site operations, not only by central planning.

### Scenario Design

A single optimal answer is the least useful output a network model produces. Build the scenario set deliberately.

- **Baseline.** Current network, current cost, current demand. This is a validation run, not a scenario — if the model cannot reproduce today's cost, nothing downstream is trustworthy.
- **Unconstrained optimum.** Greenfield, ignoring transition cost. Never a recommendation; it establishes the theoretical cost floor and therefore the size of the prize.
- **Constrained optimum.** Realistic transition cost, service constraints, and any sites the client has ruled out of scope.
- **Stress scenarios.** Demand up and down, freight rate shifts, tariff changes on the highest-exposure lanes, and single-site loss. The single-site loss runs are usually the most valuable output of the whole exercise and are frequently omitted because they do not fit the cost-saving narrative.
- **Policy scenarios.** Where trade policy is material, treat it as a distinct scenario axis rather than folding it into general disruption, since it moves duty cost and permitted sourcing simultaneously.

### Reading the Output

**Look at the gap structure, not the recommendation.** If the constrained optimum sits close to the current network, the finding is that the network is broadly right and effort belongs elsewhere. That is a legitimate and valuable answer, and a team invested in a redesign narrative will be tempted not to report it.

**Check which constraints are binding.** The shadow prices tell you what the network is actually limited by. A model where the service-level constraint binds hardest is telling you the client has a service design question, not a footprint question.

**Test stability, not just optimality.** Re-run the optimum under each stress scenario and record how much cost it gives up. A configuration that is second-best in the base case but robust across all stress scenarios is usually the better recommendation, and a model reported only on base-case cost will never surface it.

**Round the answer.** Model output at facility-level precision implies confidence the input data does not support. Report a configuration and a cost range, not a number to three significant figures.

## Common Pitfalls

- **Accepting the client's fixed/variable cost split.** It is built for management reporting, not for optimization, and it systematically distorts open/close decisions.
- **Letting the candidate site list grow unchecked.** Solve time grows sharply with binary variables, and a list nobody has screened produces a long run that answers a question nobody asked.
- **Omitting transition cost.** Produces recommendations that are theoretically optimal and practically undeliverable.
- **Reporting a single optimal network.** The scenario spread is the deliverable. A single answer invites a debate about the model rather than about the decision.
- **Treating the optimizer's answer as the recommendation.** The model narrows the option set; judgement about execution capability, customer relationships, and organisational appetite selects from within it.

## Related Frameworks

Supply chain — network optimization (parent), strategic planning — scenario planning, cost transformation — procurement, operating model design — overview.
