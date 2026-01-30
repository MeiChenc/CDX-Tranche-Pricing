---
name: quant-pricing
description: Use when implementing or validating a quant finance pricing model (assumptions, math, Greeks, numerics, calibration).
---
Workflow:
1) Restate the instrument payoff + market conventions (rates/dividends/day count) in 5 bullets.
2) Choose a model family and justify (closed-form vs PDE vs MC).
3) Specify formulas/algorithm steps and numeric stability constraints.
4) Define required outputs: price, Greeks, calibration/sweep.
5) Provide a minimal test oracle (known limiting cases / parity / monotonicity).
