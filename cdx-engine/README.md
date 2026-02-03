# CDX Engine (MVP)

This folder contains a minimal CDX tranche pricing engine aligned to the project goals:

- Index/constituent curve loading, QC, and normalization.
- Basis adjustment to align average constituent EL with index EL.
- LHP tranche pricing with Gaussian copula + Gauss-Hermite quadrature.
- Base correlation bootstrap, surface interpolation, arbitrage checks, and risk.

## Quick Start

```bash
pip install -r requirements.txt
```

## Plotting

Generate basis-adjustment diagnostics (expected-loss comparison + beta curve) using the latest available date:

```bash
python scripts/plot_basis_adjustment.py
```

Outputs:
- `plots/expected_loss_compare_<YYYY-MM-DD>.png`
- `plots/basis_beta_<YYYY-MM-DD>.png`

## Layout

```
cdx-engine/
  data/                 # input CSVs
  configs/              # parameter YAML
  src/                  # core modules
  tests/                # unit tests
  notebooks/            # analysis notebooks
  scripts/              # batch entrypoints
```

## Notes

- The implementation uses analytic integration (Gauss-Hermite), not Monte Carlo.
- The MVP focuses on deterministic outputs to support calibration and DV01 workflows.
