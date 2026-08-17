---
title: "Pricing Strategy: Elasticity Calculation Reference"
doc_type: methodology
industry: retail
topics: [pricing-strategy, price-elasticity, analytics]
date: 2025-09-22
---

## Overview

This page holds the actual formulas behind price elasticity analysis. The methodology page covers when to use elasticity and how to interpret it; this one covers how to compute it correctly, and — more importantly — how to convert an elasticity estimate into a defensible profit statement, which is where most elasticity work goes wrong.

## When to Use It

Reach for this page when you are building or reviewing an elasticity model, or when a client challenges the arithmetic behind a price recommendation. Analysts should read the breakeven section below before writing any sentence of the form "demand is elastic, so we should not raise price."

## Framework

**Arc versus point elasticity.** For a discrete observed price change, use arc (midpoint) elasticity; it is symmetric with respect to the direction of the change, which simple percentage-change elasticity is not.

```
              (Q2 - Q1) / ((Q1 + Q2) / 2)
arc_e  =  ------------------------------------
              (P2 - P1) / ((P1 + P2) / 2)
```

Point elasticity is the derivative form, appropriate only when you have a fitted demand curve rather than two observations.

**Estimating from transaction data.** The standard specification is log-log, which yields a coefficient that is directly interpretable as elasticity. Confounder controls are not optional — without them the coefficient measures promotional response and seasonality, not price response.

```python
# log-log demand specification
# beta_price is the elasticity estimate
model = ols(
    formula=(
        "log_units ~ log_price"
        " + C(week_of_year)"        # seasonality
        " + promo_flag"             # own promotional activity
        " + log_competitor_price"   # cross-price effects
        " + C(store_id)"            # store fixed effects
        " + holiday_flag"
    ),
    data=transactions,
).fit(cov_type="cluster", cov_kwds={"groups": transactions.store_id})

elasticity = model.params["log_price"]        # expect negative
ci_low, ci_high = model.conf_int().loc["log_price"]
```

Report the confidence interval alongside the point estimate. An elasticity of -1.4 with an interval spanning -0.6 to -2.2 does not support a precise price recommendation, and presenting only the point estimate hides that.

**Breakeven elasticity — the part most analyses get wrong.** The common shorthand that a price increase is only profitable when demand is inelastic is a statement about *revenue*, not profit. For profit, the threshold depends on contribution margin. A price increase raises profit whenever the magnitude of elasticity is below:

```
breakeven |e|  =  1 / contribution_margin_ratio

where contribution_margin_ratio = (price - variable_cost) / price
```

Worked substitution, for a product priced at 10.00 with variable cost of 8.00:

```
contribution_margin_ratio = (10.00 - 8.00) / 10.00 = 0.20
breakeven |e|             = 1 / 0.20             = 5.0

=> a price increase improves profit for any |e| below 5.0,
   well into the range conventionally described as "elastic".

By contrast, at a 60% contribution margin:
breakeven |e| = 1 / 0.60 = 1.67
=> the same product is far more exposed to volume loss.
```

The practical implication is that low-margin businesses tolerate far more elasticity before a price increase stops paying, and high-margin businesses far less. Any recommendation that applies a single elasticity threshold across a portfolio with varied margin structure is wrong for most of that portfolio.

**Converting to a price-volume-margin statement.** Once you have elasticity and margin, the expected profit change from a proportional price move is computed directly rather than asserted:

```
new_volume  = Q * (1 + e * pct_price_change)
new_margin  = (P * (1 + pct_price_change)) - V
profit_delta = (new_volume * new_margin) - (Q * (P - V))
```

Run this across the confidence interval of the elasticity estimate, not just the point estimate, and present the resulting range.

## Common Pitfalls

- **Quoting the revenue rule as a profit rule.** "Elasticity above one means don't raise price" is true for revenue and frequently false for profit. Use the breakeven formula above, which accounts for margin, and state the contribution margin you used.
- **Reporting a point estimate without an interval.** Elasticity estimates from observational retail data are noisy. A recommendation built on the point estimate alone will not survive a competent client analytics team.
- **Ignoring the cross-price term.** Omitting competitor price from the specification pushes competitive response into the own-price coefficient, usually biasing the estimate toward apparent elasticity.

## Related Frameworks

Pricing strategy — price elasticity analysis (parent), competitive pricing response, value-based pricing, market entry analysis — market sizing.
