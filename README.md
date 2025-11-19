# CDX Tranche Pricing – Gaussian Copula vs G-VG Model

This repository implements a **standard CDS / CDO modeling pipeline** for a CDX investment-grade index with multiple maturities (1Y, 2Y, 3Y, 5Y, 7Y, 10Y), and compares:

- **One-factor Gaussian Copula** (flat correlation, LHP approximation)
- **G-VG Model** (Gaussian–Variance-Gamma mixture factor with regime-switching correlation)

using Bloomberg-style CDX index & tranche quotes.

---

## 1. Data

The project uses two input files under `data/`:

- `cdx_constituents_multi_tenor.csv`  
  - 125 index constituents  
  - Used only for the number of names (weights assumed equal)  
  - Recovery is set to 40% at index level (CDX standard convention)

- `cdx_market_data_multi_tenor.json`  
  - For each tenor in `{1Y,2Y,3Y,5Y,7Y,10Y}`:
    - Index **full_index** spread (bps)
    - Tranche quotes:
      - Equity 0–3%: running spread (bps) + upfront (%)
      - Mezz 3–7%, 7–10%, Senior 10–15%, 15–100%: running spreads (bps)

Example (5Y section):

```json
"5Y": {
  "index_bid": 51.5312,
  "index_ask": 51.5312,
  "equity_0_3_running": 821.54,
  "equity_0_3_upfront": 28.3014,
  "mezz_3_7": 95.08,
  "mezz_7_10": 113.1,
  "senior_10_15": 55.94,
  "senior_15_100": 21.72,
  "full_index": 51.97
}
