---
title: "Supply Chain: Network Optimization"
doc_type: methodology
industry: industrials
topics: [supply-chain, network-optimization, logistics]
date: 2024-05-30
---

## Overview

Supply chain network optimization determines the optimal configuration of a client's manufacturing, warehousing, and distribution footprint — how many facilities, where located, serving which demand — to minimize total delivered cost while meeting service-level requirements. It's a quantitative modeling exercise with major strategic consequences, since network decisions are capital-intensive and difficult to reverse once built.

## When to Use It

Use when a client's supply chain footprint was built incrementally (often through organic growth or acquisition, each layering on facilities without a full network re-optimization) and current total landed cost or service performance suggests the footprint may no longer be optimal — or proactively when demand patterns, input costs, or trade policy shift materially enough to warrant a re-evaluation.

## Framework

1. **Build the network model.** Map current facilities, capacity, cost structure (fixed and variable), and demand by location, together with transportation cost and lead-time data between every relevant node pair. Model quality depends heavily on transportation cost data accuracy — this is frequently the weakest link in client data availability.
2. **Define service constraints explicitly.** Total cost minimization without service-level constraints (maximum delivery lead time by customer segment, minimum inventory availability) produces a network that's cheap but potentially unacceptable to customers — service requirements need to be inputs to the optimization, not an afterthought applied to the output.
3. **Run scenario-based optimization, not a single answer.** Model the optimal network under multiple demand and cost scenarios (see scenario planning methodology) rather than a single point forecast — network decisions are long-lived enough that they need to perform reasonably across a range of futures, not just optimally against one forecast.
4. **Account for transition cost and risk in the recommendation.** The theoretically optimal network rarely accounts for the cost, disruption, and execution risk of moving from the current footprint to the new one. A credible recommendation weighs steady-state optimality against a realistic, risk-adjusted transition plan — sometimes a directionally-optimal but incremental move beats a theoretically perfect but highly disruptive one.

## Common Pitfalls

- **Poor transportation cost data quality.** Network optimization models are only as good as the underlying cost data — garbage-in produces a confidently wrong recommendation.
- **Optimizing cost without service constraints.** A network that minimizes cost while ignoring customer service requirements will be technically optimal and commercially unacceptable.
- **Ignoring transition cost and risk.** Recommending a theoretically optimal network without weighing the cost and disruption of actually getting there from the current footprint produces plans that look great on a slide and stall in implementation.

## Related Frameworks

Cost transformation — procurement cost reduction (adjacent lever on inbound logistics and materials cost), scenario planning (informs the demand/cost scenarios used in network modeling).
