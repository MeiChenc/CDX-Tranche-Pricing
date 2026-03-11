# CDX Engine

CDX tranche pricing and correlation risk analytics engine with:

- Gaussian copula LHP tranche pricing
- hazard curve + basis adjustment diagnostics
- base-correlation calibration and PCHIP interpolation
- historical base-correlation PCA analytics
- node-level and factor-level correlation risk
- scenario-based correlation hedging diagnostics

## Environment

```bash
cd cdx-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Input Files

All must exist under `data/`:

- `cdx_timeseries.csv`
- `constituents_timeseries.csv`
- `ois_timeseries.csv`

## Canonical Run Order (Core Pipeline)

Run scripts in this exact order.

1. Data sanity check

```bash
python3 scripts/run_daily.py
```

2. Basis-adjusted hazard diagnostics

```bash
python3 scripts/plot_basis_adjustment.py --date 2024-12-03 --outdir outputs
```

3. Base-correlation node calibration snapshots

```bash
python3 scripts/plot_basecorr_surface.py --all-days --outdir outputs
```

4. Continuous PCHIP smile/surface diagnostics

```bash
python3 scripts/plot_continuous_basecorr_pchip.py --date 2024-12-03 --outdir outputs --no-show
```

5. Historical surface analytics + PCA artifacts

```bash
python3 scripts/analyze_basecorr_timeseries.py --max-days 30 --run-pca --outdir outputs
```

6. Node-level correlation sensitivity

```bash
python3 scripts/run_node_correlation_sensitivity.py --date 2024-12-03 --outdir outputs/corr_sensitivity
```

7. Factor risk decomposition (project node sensitivity to PC space)

```bash
python3 scripts/run_factor_risk_decomposition.py \
  --sens-file outputs/corr_sensitivity/node_level_sensitivities.csv \
  --pca-dir outputs \
  --outdir outputs/factor_risk
```

8. Scenario-based correlation hedging diagnostics

```bash
python3 scripts/run_scenario_correlation_hedging.py \
  --input outputs/factor_risk/factor_exposures_by_tranche.csv \
  --factor-scores outputs/factor_risk/factor_scores.csv \
  --outdir outputs/scenario_hedging
```

## Output Dependency Checkpoints

If these files exist, the project can proceed end-to-end:

1. From Step 5 (PCA foundation):
- `outputs/basecorr_node_timeseries.csv`
- `outputs/pca_loadings.csv`
- `outputs/pca_explained_variance.csv`

2. From Step 6:
- `outputs/corr_sensitivity/node_level_sensitivities.csv`

3. From Step 7:
- `outputs/factor_risk/factor_exposures_by_tranche.csv`
- `outputs/factor_risk/factor_scores.csv`

4. From Step 8:
- `outputs/scenario_hedging/data/scenario_hedging_summary.csv`
- `outputs/scenario_hedging/plots/vol_reduction_method_comparison.png`

## Framework Map (How Components Connect)

1. Pricing core (GC/LHP)
- `src/copula_lhp.py`
- `src/pricer_tranche.py`

2. Curves and basis adjustment
- `src/curves.py`
- `src/basis.py`
- `src/basis_adjustment_utils.py`
- `src/io_data.py`

3. Base-correlation calibration and interpolation
- `src/calibration_basecorr_relaxed.py`
- `src/interpolation.py`
- `scripts/risk_engine_common.py` (shared risk script utilities)

4. Time-series/PCA analytics
- `scripts/analyze_basecorr_timeseries.py`
- `scripts/run_pca_explained_variance.py` (supplementary)

5. Risk analytics engines
- `scripts/run_node_correlation_sensitivity.py`
- `scripts/run_factor_risk_decomposition.py`
- `scripts/run_scenario_correlation_hedging.py`

## Script Classification

Core scripts (must keep for primary workflow):

- `run_daily.py`
- `plot_basis_adjustment.py`
- `plot_basecorr_surface.py`
- `plot_continuous_basecorr_pchip.py`
- `analyze_basecorr_timeseries.py`
- `run_node_correlation_sensitivity.py`
- `run_factor_risk_decomposition.py`
- `run_scenario_correlation_hedging.py`
- `risk_engine_common.py`

Extended analytics (optional but useful):

- `run_scenario_stress_tests.py`
- `run_greek_hedge_stability.py`
- `build_tranche_risk_dashboard.py`
- `run_factor_shock_pricing.py`

Research/demo scripts (non-blocking):

- `plot_2D_basecorr.py`
- `run_loocv.py`
- `run_tenor_loocv.py`

## Notes

- All spread inputs are interpreted as bps and converted in code.
- Base-correlation risk and hedging outputs are under `outputs/corr_sensitivity/`, `outputs/factor_risk/`, and `outputs/scenario_hedging/`.
- Scenario hedging outputs are split by type:
  - tabular data: `outputs/scenario_hedging/data/`
  - figures: `outputs/scenario_hedging/plots/`

## Project Layout

```text
cdx-engine/
  data/                     raw market inputs
  src/                      pricing/calibration/risk core modules
  scripts/                  runnable analytics entrypoints
  tests/                    unit tests
  outputs/                  generated artifacts
    corr_sensitivity/
    factor_risk/
    scenario_hedging/
      data/
      plots/
```
