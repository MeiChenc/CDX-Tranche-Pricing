# CDX Tranche Pricing

CDX Tranche Pricing is a Python project for pricing CDX index tranches, calibrating base-correlation surfaces, and running portfolio-style backtests on derived trading signals without look-ahead bias. The production code lives in `cdx-engine/`; `legacy/` contains earlier notebooks and research artifacts.

## Project Objective

Backtest a portfolio strategy using the provided financial data, while enforcing realistic execution and evaluation rules:

- trades are executed at the `OPEN` price
- commission is `$0.10` per trade
- models must not use future information
- performance is evaluated with Sharpe ratio and Maximum Drawdown

## Installation / Setup

### Prerequisites

- Python 3.10+
- `pip`
- local copies of the CSV input files in `cdx-engine/data/`

### Setup Commands

```bash
cd cdx-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage Example

### Run the data-loading pipeline

```bash
cd cdx-engine
python3 scripts/run_daily.py
```

### Run the pricing model from Python

```python
from src.curves import Curve
from src.pricer_tranche import price_tranche_lhp

curve = Curve(times=[3.0, 5.0, 7.0, 10.0], hazard=[0.02, 0.024, 0.027, 0.03])

pv = price_tranche_lhp(
    tenor=5.0,
    k1=0.03,
    k2=0.07,
    rho=0.35,
    curve=curve,
    recovery=0.40,
    n_quad=64,
    payment_freq=4,
)

print(pv.premium_leg, pv.protection_leg, pv.pv)
```

## Data Description

The core engine uses three market-data files under `cdx-engine/data/`:

- `cdx_timeseries.csv`: CDX index and tranche quotes by `Date` and `Tenor`
- `constituents_timeseries.csv`: constituent CDS spreads and recovery assumptions
- `ois_timeseries.csv`: OIS rates used for discounting

Current CSV headers in the repository:

- `cdx_timeseries.csv`: `Date, Tenor, Index_Bid, Index_Ask, Index_Last, Index_Mid, Equity_0_3_Spread, Equity_0_3_Upfront, Mezz_3_7_Spread, Mezz_3_7_Upfront, Mezz_7_10_Spread, Senior_10_15_Spread, SuperSenior_15_100_Spread, Index_0_100_Spread`
- `constituents_timeseries.csv`: `Date, Company, Recovery, Spread_5Y, Spread_7Y, Spread_10Y`
- `ois_timeseries.csv`: `Date, Tenor, OIS_Rate`

The loader in `cdx-engine/src/io_data.py` parses dates, normalizes tenor fields such as `5Y` or `3M`, and converts each date into a market snapshot used by the pricing and calibration pipeline.

## Data Format For Backtests

If you use this repository to backtest a trading strategy, the expected daily bar input should be a CSV with at least:

- `Date`
- `Open`
- `Close`

Typical structure:

```csv
Date,Open,Close
2024-01-02,100.25,101.10
2024-01-03,101.05,100.80
```

Backtest rules:

- signals may be computed only from information available up to the decision time
- execution occurs at the `Open` price
- end-of-day valuation may use the `Close` price
- each trade is charged a commission of `$0.10`

## Methodology

The project combines pricing, calibration, and risk modeling components:

- Alpha Modeling: derive trade signals from changes in spreads, tranche valuations, or correlation dislocations
- Risk Modeling: measure exposure through tranche PV sensitivity, base-correlation moves, and curve/risk-factor decomposition
- Pricing Model: a one-factor Gaussian copula with large homogeneous portfolio approximation
- Curve Construction: bootstrap index and constituent hazard curves from CDS spreads
- Calibration: solve for base correlations so model tranche PV matches observed tranche quotes
- Interpolation: use PCHIP interpolation to build continuous tenor/detachment surfaces

## Constraints

- No look-ahead bias: training, signal generation, and portfolio decisions must only use information known at that time.
- Trading happens at the `OPEN` price.
- Commission is fixed at `$0.10` per trade.
- Any backtest built on top of the model should separate in-sample calibration from out-of-sample evaluation.

## Evaluation

Primary portfolio metrics:

- Sharpe ratio
- Maximum Drawdown

Useful secondary diagnostics:

- cumulative return
- hit rate
- turnover
- average trade P&L

## Repository Layout

```text
CDX-Tranche-Pricing/
  cdx-engine/
    src/          pricing, calibration, interpolation, and risk modules
    scripts/      runnable workflows
    tests/        pytest suite
    data/         input CSVs
    configs/      runtime parameters
    notebooks/    research notebooks
    outputs/      generated plots and tables
  legacy/         earlier notebooks and exploratory outputs
```
