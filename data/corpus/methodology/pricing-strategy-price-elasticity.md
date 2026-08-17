---
title: "Pricing Strategy: Price Elasticity Analysis"
doc_type: methodology
industry: retail
topics: [pricing-strategy, price-elasticity, analytics]
date: 2022-11-14
---

## Overview

Price elasticity analysis quantifies how demand responds to a price change, expressed as the percentage change in quantity demanded for a given percentage change in price. It's the analytical backbone that lets a pricing recommendation move from "we believe this price change will improve margin" to a defensible, quantified estimate of the revenue and volume trade-off.

## When to Use It

Use elasticity analysis whenever a client is considering a price change of meaningful size (beyond routine annual inflation-linked adjustment) and needs to understand the volume impact before committing — particularly relevant in retail and consumer categories with enough transaction-level data to model demand response directly.

## Framework

1. **Assemble transaction-level data.** Elasticity estimation needs granular price and volume data across enough price variation (across time, geography, or SKU) to isolate the price effect from other demand drivers (seasonality, promotion, macro conditions).
2. **Control for confounders.** Raw price-volume correlation is contaminated by promotions, seasonality, and competitor actions happening at the same time. A credible elasticity estimate isolates the price effect using regression techniques that control for these factors, not a simple before/after comparison.
3. **Segment elasticity by category and customer.** Elasticity varies enormously by product category (staples vs. discretionary) and customer segment (price-sensitive vs. loyalty-driven). A single blended elasticity figure across an entire portfolio is rarely actionable.
4. **Translate elasticity into a price-volume-margin trade-off.** Combine the elasticity estimate with current margin structure to model the net profit impact of a proposed price change. Elasticity below 1 (inelastic demand) always improves revenue on a price increase, but the profit answer depends on contribution margin, not on the elasticity-vs-1 threshold alone — the breakeven elasticity above which a price increase starts destroying profit rises as contribution margin rises (at a 20% contribution margin, for instance, profit keeps improving up to an elasticity near 5). Don't apply the elasticity-below-1 rule of thumb to a profit decision without checking it against the actual margin structure.

## Common Pitfalls

- **Correlation mistaken for elasticity.** Failing to control for promotions and seasonality produces elasticity estimates that are really measuring something else entirely.
- **Applying portfolio-average elasticity to individual SKU decisions.** Category-level and even SKU-level elasticity can differ sharply from the portfolio average, especially between private label and branded goods.
- **Ignoring competitive response in the volume model.** An elasticity estimate built from historical data implicitly assumes competitors don't react to the price change — for a price move large enough to trigger a competitive response, pair this analysis with competitive pricing response modeling.

## Related Frameworks

Pricing strategy (parent), competitive pricing response, market entry analysis — market sizing (elasticity informs realistic revenue projections).
