# CDX Tranche Pricing

This repository is a CDX tranche pricing and correlation analytics project built around the `cdx-engine/` package.

At a high level, it does four things:

1. reads index, tranche, constituent, and OIS market data
2. bootstraps hazard curves and applies basis adjustments
3. calibrates Gaussian-equivalent base-correlation smiles and surfaces
4. runs validation and risk analytics such as LOOCV, PCA, node sensitivities, and hedging studies

The current production-style code lives in [cdx-engine](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine). The [legacy](/Users/ginachen/Desktop/CDX-Tranche-Pricing/legacy) folder contains earlier notebooks, plots, and raw research artifacts.

## Repository Map

```text
CDX-Tranche-Pricing/
  cdx-engine/
    src/          core pricing, calibration, basis, interpolation, and risk modules
    scripts/      runnable workflows and analytics entry points
    tests/        pytest suite for pricing, calibration, basis, and arbitrage checks
    data/         local CSV inputs
    outputs/      generated tables and plots
    notebooks/    research and workflow notebooks
    configs/      runtime parameters
  legacy/         older notebooks and exploratory outputs
```

## What The Project Models

The engine prices tranche expected loss and tranche PVs under a one-factor Gaussian copula in the large homogeneous portfolio approximation.

The core modeling flow is:

1. bootstrap an index survival curve from quoted index spreads
2. optionally bootstrap constituent curves and apply a beta-style basis adjustment so constituent-implied loss matches index-implied loss
3. price base tranches `0-K` under candidate correlations
4. solve for base correlation at each detachment so model PV matches tranche quotes
5. interpolate those calibrated nodes across detachment and tenor to get a continuous surface
6. use that surface for validation, scenario analysis, and risk decomposition

This repo does not currently route the top-level workflow through the older G-VG text in the previous root README. The code in `cdx-engine/src/` is centered on Gaussian copula LHP plus base-correlation analytics.

## Input Data

The main input files are expected under [cdx-engine/data](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/data):

- `cdx_timeseries.csv`: index and tranche quotes by date and tenor
- `constituents_timeseries.csv`: constituent CDS spreads and recovery metadata
- `ois_timeseries.csv`: OIS discount curve points by date and tenor

The loader in [io_data.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/io_data.py) normalizes column names, parses tenor strings like `5Y` or `3M`, and builds `MarketSnap` objects containing:

- available tenors for a date
- index quotes
- tranche quotes
- constituent spreads
- a simple stale/bad-date flag

## End-To-End Walkthrough

### 1. Data QC and market snapshots

Start with [run_daily.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/run_daily.py).

It reads [params.yaml](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/configs/params.yaml), loads the market files, and flags dates that look unreliable based on:

- minimum constituent coverage
- large quote jumps on sparse dates

This is the ingestion and sanity-check step.

### 2. Discount and hazard curves

Two curve objects drive most of the engine:

- discount curve from OIS data via `read_ois_discount_curve(...)`
- default curve via [curves.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/curves.py)

Important pieces in `curves.py`:

- `Curve.survival(t)`: piecewise-constant hazard survival probability
- `bootstrap_from_cds_spreads(...)`: bootstraps hazard rates from CDS par spreads
- `build_index_curve(...)`: builds the index hazard curve

Assumptions baked into the bootstrap:

- spreads are decimal inside the engine
- quarterly premium payments by default
- half-period accrual-on-default approximation
- optional OIS discounting

### 3. Basis adjustment

The code does not rely only on the raw index curve. It can build a basis-adjusted curve that uses constituent information to reconcile bottom-up and top-down loss views.

Relevant files:

- [basis.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/basis.py)
- [basis_adjustment_utils.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/basis_adjustment_utils.py)

The workflow is:

1. bootstrap constituent hazard curves from constituent CDS spreads
2. expand them onto the index tenor grid
3. solve for a tenor-by-tenor beta scaling so average constituent expected loss matches the index expected loss
4. rebuild an adjusted average curve

That adjusted curve is then used as the survival input for tranche pricing and base-correlation calibration.

This is an important project choice: correlation is calibrated on top of a survival curve that has already been basis-adjusted.

### 4. Tranche pricing

The tranche pricer is implemented in:

- [copula_lhp.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/copula_lhp.py)
- [pricer_tranche.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/pricer_tranche.py)

The mechanics are:

1. for a given systemic factor draw, compute conditional default probability under a one-factor Gaussian copula
2. convert it into conditional portfolio loss
3. clip that portfolio loss into tranche loss for `K1-K2`
4. integrate with Gauss-Hermite quadrature
5. aggregate expected loss increments into premium and protection legs

Key functions:

- `conditional_default_prob(...)`
- `tranche_expected_loss(...)`
- `price_tranche_lhp(...)`

`price_tranche_lhp(...)` returns a `TranchePV` object with:

- `premium_leg`
- `protection_leg`
- `pv = protection_leg - premium_leg`

### 5. Base-correlation calibration

Calibration is in:

- [calibration_basecorr.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/calibration_basecorr.py)
- [calibration_basecorr_relaxed.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/calibration_basecorr_relaxed.py)

The project uses the standard base-correlation decomposition:

`PV(K1,K2) = PV(0,K2 ; rho(K2)) - PV(0,K1 ; rho(K1))`

For each tenor, detachments such as `3%`, `7%`, `10%`, and `15%` are solved sequentially.

There are two calibration styles in the code:

- exact bracketing/root solve in `calibration_basecorr.py`
- more forgiving scan-plus-minimization logic in `calibration_basecorr_relaxed.py`

The relaxed version is the main workhorse in the scripts because real market quotes do not always bracket perfectly.

### 6. Interpolation into a surface

Once node calibrations exist by tenor and detachment, [interpolation.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/interpolation.py) builds a continuous surface using `PchipInterpolator`.

Interpolation is done in two stages:

1. interpolate across tenor for each fixed detachment
2. interpolate across detachment at the requested tenor

PCHIP is used to preserve shape better than a naive cubic spline.

### 7. Arbitrage checks

[arbitrage.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/arbitrage.py) provides simple monotonicity checks and a basic smile fix:

- detect non-monotone smiles across detachment
- detect non-monotone term structures across tenor
- optionally enforce monotonicity with cumulative maxima

### 8. Validation with LOOCV

The LOOCV logic most relevant to your earlier question is in:

- [run_tenor_loocv.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/run_tenor_loocv.py)

This script does two related validations.

First, tenor LOOCV:

1. hide an entire tenor, for example `7Y`
2. calibrate base-correlation smiles on the remaining tenors
3. interpolate the surface back to the hidden tenor
4. compare predicted hidden-tenor correlations and implied spreads against the actual calibrated tenor

Second, restricted point LOOCV inside a smile:

1. take a calibrated smile for a tenor
2. hide a middle detachment such as `7%` or `10%`
3. interpolate from the remaining smile nodes
4. compare the predicted node with the actual calibrated node

That second plot is the one you shared earlier. The red `X` is the interpolated prediction at the hidden detachment, and the blue point is the actual calibrated node.

### 9. Time-series analytics and risk

Once daily surfaces are calibrated, the project extends into correlation risk analytics.

Main scripts:

- [plot_basecorr_surface.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/plot_basecorr_surface.py)
- [plot_continuous_basecorr_pchip.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/plot_continuous_basecorr_pchip.py)
- [analyze_basecorr_timeseries.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/analyze_basecorr_timeseries.py)
- [run_node_correlation_sensitivity.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/run_node_correlation_sensitivity.py)
- [run_factor_risk_decomposition.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/run_factor_risk_decomposition.py)
- [run_scenario_correlation_hedging.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/run_scenario_correlation_hedging.py)

These scripts produce:

- daily node calibration snapshots
- continuous smile and surface plots
- PCA on historical base-correlation moves
- node-level tranche sensitivities
- mapping of node risks into factor space
- scenario-based hedge diagnostics

There are also optional scripts for:

- stress testing
- factor shock pricing
- Greek hedge stability
- tranche risk dashboards

## Core Files By Responsibility

### Pricing core

- [copula_lhp.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/copula_lhp.py)
- [pricer_tranche.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/pricer_tranche.py)
- [utils_math.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/utils_math.py)

### Curves and market data

- [io_data.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/io_data.py)
- [curves.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/curves.py)
- [dates.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/dates.py)

### Basis adjustment

- [basis.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/basis.py)
- [basis_adjustment_utils.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/basis_adjustment_utils.py)

### Correlation calibration and interpolation

- [calibration_basecorr.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/calibration_basecorr.py)
- [calibration_basecorr_relaxed.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/calibration_basecorr_relaxed.py)
- [interpolation.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/interpolation.py)
- [arbitrage.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/arbitrage.py)
- [loocv.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/loocv.py)

### Risk and hedging

- [risk.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/risk.py)
- [hedging.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/hedging.py)

## How To Run

Setup:

```bash
cd cdx-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended core workflow:

```bash
python3 scripts/run_daily.py
python3 scripts/plot_basis_adjustment.py --date 2024-12-03 --outdir outputs
python3 scripts/plot_basecorr_surface.py --all-days --outdir outputs
python3 scripts/plot_continuous_basecorr_pchip.py --date 2024-12-03 --outdir outputs --no-show
python3 scripts/analyze_basecorr_timeseries.py --max-days 30 --run-pca --outdir outputs
python3 scripts/run_node_correlation_sensitivity.py --date 2024-12-03 --outdir outputs/run_node_correlation_sensitivity
python3 scripts/run_factor_risk_decomposition.py --sens-file outputs/run_node_correlation_sensitivity/data/node_level_sensitivities.csv --pca-dir outputs/analyze_basecorr_timeseries --outdir outputs/run_factor_risk_decomposition
python3 scripts/run_scenario_correlation_hedging.py --input outputs/run_factor_risk_decomposition/data/factor_exposures_by_tranche.csv --factor-scores outputs/analyze_basecorr_timeseries/data/factor_scores.csv --outdir outputs/run_scenario_correlation_hedging
```

LOOCV example:

```bash
python3 scripts/run_tenor_loocv.py --hide-tenor 7 --date 2024-12-03
```

## Testing

The test suite lives under [cdx-engine/tests](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/tests) and covers:

- LHP loss logic
- tranche pricing behavior
- curve bootstrap behavior
- basis adjustment
- calibration
- arbitrage checks

Run:

```bash
cd cdx-engine
python3 -m pytest -q
```

## Output Conventions

Generated artifacts are usually written under [cdx-engine/outputs](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/outputs), grouped by script name. Most folders contain:

- `data/` for CSV or text diagnostics
- `plots/` for charts

Examples already committed in the repo include:

- calibrated base-correlation surfaces
- continuous PCHIP smiles
- LOOCV plots
- PCA heatmaps
- factor exposure tables
- scenario hedging diagnostics

## What To Read First

If you want to understand the repo quickly, read in this order:

1. [cdx-engine/README.md](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/README.md)
2. [curves.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/curves.py)
3. [pricer_tranche.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/pricer_tranche.py)
4. [calibration_basecorr_relaxed.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/src/calibration_basecorr_relaxed.py)
5. [run_tenor_loocv.py](/Users/ginachen/Desktop/CDX-Tranche-Pricing/cdx-engine/scripts/run_tenor_loocv.py)

That sequence gets you from market data to curves to pricing to smile calibration to validation.
