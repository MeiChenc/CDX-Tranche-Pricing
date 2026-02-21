from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration_basecorr_relaxed import calibrate_basecorr_relaxed
from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle
from src.curves import Curve, build_index_curve
from src.io_data import read_ois_discount_curve
from src.pricer_tranche import price_tranche_lhp, tranche_expected_loss
from src.utils_math import hermite_nodes_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot base correlation surface from CDX time series data.")
    parser.add_argument("--date", type=str, default=None, help="Date to plot (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite nodes for pricing.")
    parser.add_argument("--grid-size", type=int, default=200, help="Grid size for bracketing search.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING).")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for diagnostics.")
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


def _curve_from_cum_hazard(tenors: np.ndarray, cum_hazard: np.ndarray) -> Curve:
    tenors = np.asarray(tenors, dtype=float)
    cum_hazard = np.asarray(cum_hazard, dtype=float)
    if tenors.ndim != 1 or cum_hazard.ndim != 1 or tenors.size != cum_hazard.size:
        raise ValueError("tenors and cum_hazard must be 1D arrays with matching shape")
    if tenors.size == 0:
        raise ValueError("empty tenor grid")
    if np.any(np.diff(tenors) <= 0):
        raise ValueError("tenors must be strictly increasing")

    lambdas = np.zeros_like(cum_hazard, dtype=float)
    prev_t = 0.0
    prev_h = 0.0
    for i, (t, h) in enumerate(zip(tenors, cum_hazard)):
        dt = float(t) - prev_t
        if dt <= 0:
            raise ValueError("tenors must be strictly increasing")
        dh = max(float(h) - prev_h, 0.0)
        lambdas[i] = max(dh / dt, 1e-12)
        prev_t = float(t)
        prev_h = float(h)
    return Curve(times=tenors, hazard=lambdas)




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


def _model_legs_from_basecorr(
    tenor: float,
    k1: float,
    k2: float,
    rho_k2: float,
    rho_k1: float | None,
    curve: Curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
) -> Tuple[float, float]:
    pv_k2 = price_tranche_lhp(
        tenor,
        0.0,
        k2,
        rho_k2,
        curve,
        recovery,
        n_quad=n_quad,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )
    if k1 <= 0.0:
        return float(pv_k2.protection_leg), float(pv_k2.premium_leg)
    if rho_k1 is None:
        raise ValueError("rho_k1 is required when k1 > 0")

    pv_k1 = price_tranche_lhp(
        tenor,
        0.0,
        k1,
        rho_k1,
        curve,
        recovery,
        n_quad=n_quad,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )
    model_prot = float(pv_k2.protection_leg - pv_k1.protection_leg)
    model_prem = float(pv_k2.premium_leg - pv_k1.premium_leg)
    return model_prot, model_prem


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    index_df = pd.read_csv("data/cdx_timeseries.csv", parse_dates=["Date"])
    index_df["Tenor"] = index_df["Tenor"].astype(str).str.upper()
    index_df["tenor"] = index_df["Tenor"].str.replace("Y", "", regex=False).astype(float)

    numeric_cols = [
        "Index_Mid",
        "Index_0_100_Spread",
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
    tenors = snapshot["tenor"].to_numpy(dtype=float)
    spreads_0_100 = snapshot["Index_0_100_Spread"].to_numpy(dtype=float) / 10000.0
    spreads_mid = snapshot["Index_Mid"].to_numpy(dtype=float) / 10000.0
    curve_mid = build_index_curve(tenors, spreads_mid, recovery=0.4, disc_curve=disc_curve)
    (
        tenors_basis,
        theoretical_curve,
        _market_curve,
        beta_knot,
        beta_cum,
        _theoretical_spreads,
        _market_spreads,
    ) = build_index_dual_curve_beta_bundle(
        snapshot,
        disc_curve=disc_curve,
        recovery_index=0.4,
        theoretical_col="Index_0_100_Spread",
        market_col="Index_Mid",
    )
    theo_H = -np.log(np.maximum(np.array([theoretical_curve.survival(float(t)) for t in tenors_basis]), 1e-12))
    adjusted_H = beta_cum * theo_H
    curve_adjusted = _curve_from_cum_hazard(tenors_basis, adjusted_H)
    curves = {
        "Basis_Adjusted": curve_adjusted,
    }

    logging.info("Calibrating base correlation with curve: %s", list(curves.keys()))
    logging.info("DEBUG: 0-100 spreads by tenor: %s", dict(zip(tenors, spreads_0_100)))
    logging.info("DEBUG: Mid spreads by tenor: %s", dict(zip(tenors, spreads_mid)))
    logging.info("DEBUG: beta_knot by tenor: %s", dict(zip(tenors_basis, beta_knot)))
    logging.info("DEBUG: beta_cum by tenor: %s", dict(zip(tenors_basis, beta_cum)))
    dprob_adj = {float(t): curve_adjusted.default_prob(float(t)) for t in tenors_basis}
    logging.info("DEBUG: Basis-adjusted default prob by tenor: %s", dprob_adj)

    tenors_sorted = np.sort(tenors)
    dets = [0.03, 0.07, 0.10, 0.15]
    surfaces: Dict[str, np.ndarray] = {
        label: np.full((len(dets), len(tenors_sorted)), np.nan, dtype=float) for label in curves
    }
    residual_rows: List[dict] = []
    logging.info("Target date: %s | tenors: %s", target_date, tenors_sorted.tolist())

    for curve_label, curve in curves.items():
        logging.info("Running basecorr calibration using curve=%s", curve_label)
        surface = surfaces[curve_label]
        for j, tenor in enumerate(tenors_sorted):
            row = snapshot[snapshot["tenor"] == tenor].iloc[0]
            tranche_spreads, tranche_upfronts = _row_quotes(row)
            logging.info("Using coarse grid basecorr fit (grid_size=%d, n_quad=%d).", args.grid_size, args.n_quad)
            basecorr, solve_status = calibrate_basecorr_relaxed(
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
                return_status=True,
            )

            for i, det in enumerate(dets):
                surface[i, j] = basecorr.get(det, np.nan)
            logging.info("%s tenor %.2f basecorr: %s", curve_label, tenor, {d: basecorr.get(d, np.nan) for d in dets})
            for idx_det, det in enumerate(dets):
                rho = float(basecorr.get(det, np.nan))
                status = solve_status.get(det, "UNKNOWN")
                logging.info("%s calibrated rho tenor %.2f det %.2f: %.6f (%s)", curve_label, tenor, det, rho, status)
                k1 = 0.0 if idx_det == 0 else dets[idx_det - 1]
                k2 = det
                rho_k1 = None if idx_det == 0 else float(basecorr.get(k1, np.nan))
                spread = float(tranche_spreads.get(det, 0.0))
                upfront = float(tranche_upfronts.get(det, 0.0))
                model_prot, model_prem = _model_legs_from_basecorr(
                    tenor=tenor,
                    k1=k1,
                    k2=k2,
                    rho_k2=rho,
                    rho_k1=rho_k1,
                    curve=curve,
                    recovery=0.4,
                    n_quad=args.n_quad,
                    payment_freq=4,
                    disc_curve=disc_curve,
                )
                width = float(k2 - k1)
                residual = model_prot - spread * model_prem - upfront * width
                logging.info(
                    "%s legs tenor %.2f [%.2f, %.2f]: rho=%.6f prot=%.10f prem=%.10f spread_mkt=%.6f upfront=%.6f residual=%.10f",
                    curve_label,
                    tenor,
                    k1,
                    k2,
                    rho,
                    model_prot,
                    model_prem,
                    spread,
                    upfront,
                    residual,
                )
                residual_rows.append(
                    {
                        "date": str(target_date),
                        "curve_label": curve_label,
                        "tenor": float(tenor),
                        "k1": k1,
                        "k2": k2,
                        "rho": rho,
                        "status": status,
                        "spread_running": spread,
                        "upfront": upfront,
                        "model_protection_leg": model_prot,
                        "model_premium_leg": model_prem,
                        "residual": residual,
                    }
                )

        # Drop tenors with missing calibrations to avoid NaN surfaces.
        valid_cols = np.all(np.isfinite(surface), axis=0)
        tenors_plot = tenors_sorted[valid_cols]
        surface_plot = surface[:, valid_cols]
        logging.info("%s valid tenors for surface: %s", curve_label, tenors_plot.tolist())

        T, D = np.meshgrid(tenors_plot, dets)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(T, D * 100.0, surface_plot, cmap="viridis", edgecolor="k", linewidth=0.3)

        ax.set_title(f"Base Correlation Surface ({curve_label}): {target_date}")
        ax.set_xlabel("Tenor (Years)")
        ax.set_ylabel("Detachment (%)")
        ax.set_zlabel("Base Correlation")
        fig.colorbar(surf, shrink=0.6, label="Base Correlation")
        plt.tight_layout()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    residual_df = pd.DataFrame(residual_rows)
    residual_path = outdir / f"basecorr_surface_residuals_{target_date}.csv"
    residual_df.to_csv(residual_path, index=False)
    if not residual_df.empty:
        abs_res = np.abs(residual_df["residual"].to_numpy(dtype=float))
        brent_ratio = float(np.mean(residual_df["status"].to_numpy(dtype=str) == "BRENT"))
        logging.info(
            "Residual summary | mean_abs=%.6e max_abs=%.6e brent_ratio=%.2f",
            float(np.mean(abs_res)),
            float(np.max(abs_res)),
            brent_ratio,
        )
    logging.info("Saved calibration residual diagnostics to %s", residual_path)
    plt.show()


if __name__ == "__main__":
    main()
