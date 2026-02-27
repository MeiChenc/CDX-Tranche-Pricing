# CDX Engine

CDX tranche calibration and diagnostics toolkit with:

- index/constituent curve bootstrapping,
- basis adjustment (Mid-driven beta),
- LHP tranche pricing,
- base-correlation calibration,
- surface/smile visualization,
- multi-day diagnostics (heatmap + PCA on surface moves).

## Environment

```bash
cd cdx-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Inputs

Main input CSVs in `data/`:

- `cdx_timeseries.csv`
- `ois_timeseries.csv`
- `constituents_timeseries.csv`

Core assumptions in current scripts:

- spread columns are in bps and converted to decimal in code,
- tranche running spread + upfront are used in calibration objective,
- known bad day `11/24` is dropped in time-series analytics script.

## Core Scripts

### 1) Basis Adjustment Diagnostics

```bash
python scripts/plot_basis_adjustment.py --date 2024-12-03
```

Outputs:

- `plots/expected_loss_compare_<date>.png`
- `plots/basis_beta_<date>.png`
- `plots/cum_hazard_compare_mid_basisadj_<date>.png`

### 2) Base-Corr Surface (Raw Nodes, Pre-PCHIP)

Single day:

```bash
python scripts/plot_basecorr_surface.py --date 2024-12-03
```

All valid days:

```bash
python scripts/plot_basecorr_surface.py --all-days
```

Latest N days:

```bash
python scripts/plot_basecorr_surface.py --all-days --max-days 8
```

Notes:

- calibration uses **Basis_Adjusted** curve,
- plotting keeps available `BRENT` points and masks non-BRENT nodes,
- per-day residuals are saved to `plots/basecorr_surface_residuals_<date>.csv`,
- per-day surface is saved to `plots/basecorr_surface_basis_adjusted_<date>.png`.

### 3) Continuous PCHIP Smiles + Surface

Single day:

```bash
python scripts/plot_continuous_basecorr_pchip.py --date 2024-12-03
```

Compare two days side-by-side:

```bash
python scripts/plot_continuous_basecorr_pchip.py --date 2024-11-25 --compare-date 2024-12-01
```

Options:

- `--smile-tenors` (default: all available tenors),
- interactive plots on by default (`--no-show` to disable).

### 4) Multi-Day Diagnostics (Heatmap + PCA)

```bash
python scripts/analyze_basecorr_timeseries.py --max-days 30
```

With PCA artifacts:

```bash
python scripts/analyze_basecorr_timeseries.py --max-days 30 --run-pca
```

Key outputs:

- `plots/basecorr_daily_diagnostics.csv`
- `plots/basecorr_node_timeseries.csv`
- `plots/heatmap_mean_abs_delta_rho.png`
- `plots/heatmap_p95_abs_delta_rho.png`
- `plots/basecorr_abs_delta_heatmap_matrix.csv`
- `plots/surface_move_stats.json`
- `plots/pca_explained_variance.png`
- `plots/pc1_loadings_heatmap.png`
- `plots/pc2_loadings_heatmap.png`
- `plots/pc3_loadings_heatmap.png` (if available)
- `plots/pc_scores_timeseries.csv`
- `plots/pc1_vs_spreaddiff_corr.txt`

## Model Notes

- Tranche pricing uses deterministic LHP + Gauss-Hermite quadrature.
- Premium/protection legs use improved discretization:
  - trapezoid premium leg + accrual-on-default approximation,
  - midpoint-style protection discounting on loss increments.
- Base-correlation convention:
  - tranche `(k1,k2)` uses `PV(0,k2;rho(k2)) - PV(0,k1;rho(k1))`.

## Project Layout

```text
cdx-engine/
  data/         input CSVs
  src/          pricing/calibration/basis modules
  scripts/      analysis/plot entrypoints
  tests/        unit tests
  notebooks/    exploratory work
  plots/        generated charts/tables
```
