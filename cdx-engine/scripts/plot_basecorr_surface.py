from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration_basecorr_relaxed import calibrate_basecorr_relaxed
from src.basis_adjustment_utils import build_basis_adjusted_curve
from src.curves import build_index_curve
from src.io_data import read_ois_discount_curve
from src.pricer_tranche import price_tranche_lhp, tranche_expected_loss
from src.utils_math import hermite_nodes_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot base correlation surface from CDX time series data.")
    parser.add_argument("--date", type=str, default=None, help="Date to plot (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite nodes for pricing.")
    parser.add_argument("--grid-size", type=int, default=200, help="Grid size for bracketing search.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING).")
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")



def _build_curve(
    snapshot: pd.DataFrame,
    disc_curve,
) -> tuple[np.ndarray, np.ndarray]:
    tenors = snapshot["tenor"].to_numpy(dtype=float)
    index_spreads = (snapshot["Index_Mid"].to_numpy(dtype=float)) / 10000.0 #Use mid spread for calculation
    curve = build_index_curve(tenors, index_spreads, recovery=0.4, disc_curve=disc_curve)
    return tenors, curve




def _row_quotes(row: pd.Series) -> tuple[Dict[float, float], Dict[float, float]]:
    tranche_spreads = {
        0.03: row["Equity_0_3_Spread"] / 10000.0,
        0.07: row["Mezz_3_7_Spread"] / 10000.0,
        0.10: row["Mezz_7_10_Spread"] / 10000.0,
        0.15: row["Senior_10_15_Spread"] / 10000.0,
        1.0: row["SuperSenior_15_100_Spread"] / 10000.0,
    }
    tranche_upfronts = {
        0.03: row["Equity_0_3_Upfront"] / 100.0,
        0.07: row["Mezz_3_7_Upfront"] / 100.0,
    }
    return tranche_spreads, tranche_upfronts


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    index_df = pd.read_csv("data/cdx_timeseries.csv", parse_dates=["Date"])
    index_df["Tenor"] = index_df["Tenor"].astype(str).str.upper()
    index_df["tenor"] = index_df["Tenor"].str.replace("Y", "", regex=False).astype(float)

    numeric_cols = [
        "Index_Mid",
        "Equity_0_3_Spread",
        "Equity_0_3_Upfront",
        "Mezz_3_7_Spread",
        "Mezz_3_7_Upfront",
        "Mezz_7_10_Spread",
        "Senior_10_15_Spread",
        "SuperSenior_15_100_Spread",
    ]
    _coerce_numeric(index_df, numeric_cols)

    required_cols = [
        "Equity_0_3_Spread",
        "Mezz_3_7_Spread",
        "Mezz_7_10_Spread",
        "Senior_10_15_Spread",
        "SuperSenior_15_100_Spread",
    ]
    if args.date:
        target_date = pd.to_datetime(args.date).date()
        snapshot = index_df[index_df["Date"].dt.date == target_date].copy()
        if snapshot.empty:
            raise SystemExit(f"No rows found for date {target_date}.")
        missing = snapshot[required_cols].isna().any()
        if missing.any():
            missing_cols = missing[missing].index.tolist()
            raise SystemExit(
                f"Missing tranche quotes for date {target_date}: {missing_cols}. "
                "Pick a date with complete quotes."
            )
    else:
        # Choose the latest date with complete tranche quotes
        valid = index_df.dropna(subset=required_cols)
        if valid.empty:
            raise SystemExit("No dates with complete tranche quotes found.")
        target_date = valid["Date"].max().date()
        snapshot = valid[valid["Date"].dt.date == target_date].copy()

    disc_curve = read_ois_discount_curve("data/ois_timeseries.csv", target_date)
    tenors, index_curve = _build_curve(snapshot, disc_curve)
    cons_df = pd.read_csv(ROOT / "data" / "constituents_timeseries.csv", parse_dates=["Date"])
    curve = build_basis_adjusted_curve(target_date, snapshot, cons_df, disc_curve)
    logging.info("Using basis-adjusted curve for tranche pricing.")
    logging.info("DEBUG: Index lambdas by tenor: %s", dict(zip(tenors, curve.hazard)))
    surv = {float(t): curve.survival(float(t)) for t in tenors}
    dprob = {float(t): curve.default_prob(float(t)) for t in tenors}
    logging.info("DEBUG: Index survival by tenor: %s", surv)
    logging.info("DEBUG: Index default prob by tenor: %s", dprob)

    tenors_sorted = np.sort(tenors)
    dets = [0.03, 0.07, 0.10, 0.15]
    surface = np.full((len(dets), len(tenors_sorted)), np.nan, dtype=float)
    logging.info("Target date: %s | tenors: %s", target_date, tenors_sorted.tolist())

    for j, tenor in enumerate(tenors_sorted):
        row = snapshot[snapshot["tenor"] == tenor].iloc[0]
        tranche_spreads, tranche_upfronts = _row_quotes(row)
        logging.info("Using coarse grid basecorr fit (grid_size=%d, n_quad=%d).", args.grid_size, args.n_quad)
        basecorr = calibrate_basecorr_relaxed(
            tenor,
            dets,
            tranche_spreads,
            tranche_upfronts,
            curve,
            recovery=0.4,
            grid_size=args.grid_size,
            n_quad=args.n_quad,
            payment_freq=4,
            disc_curve=disc_curve,
        )

        for i, det in enumerate(dets):
            surface[i, j] = basecorr.get(det, np.nan)
        logging.info("Tenor %.2f basecorr: %s", tenor, {d: basecorr.get(d, np.nan) for d in dets})
        for det in dets:
            logging.info("Calibrated rho for tenor %.2f det %.2f: %.6f", tenor, det, basecorr.get(det, np.nan))

    # Drop tenors with missing calibrations to avoid NaN surfaces.
    valid_cols = np.all(np.isfinite(surface), axis=0)
    tenors_plot = tenors_sorted[valid_cols]
    surface_plot = surface[:, valid_cols]
    logging.info("Valid tenors for surface: %s", tenors_plot.tolist())

    T, D = np.meshgrid(tenors_plot, dets)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, D * 100.0, surface_plot, cmap="viridis", edgecolor="k", linewidth=0.3)

    ax.set_title(f"Base Correlation Surface: {target_date}")
    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("Detachment (%)")
    ax.set_zlabel("Base Correlation")
    fig.colorbar(surf, shrink=0.6, label="Base Correlation")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
