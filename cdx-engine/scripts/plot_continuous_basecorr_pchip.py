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
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle
from src.calibration_basecorr_relaxed import calibrate_basecorr_relaxed
from src.curves import Curve
from src.interpolation import build_rho_surface
from src.io_data import read_ois_discount_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot continuous PCHIP base-correlation smiles and surface from calibrated node values."
    )
    parser.add_argument("--date", type=str, default=None, help="Primary date (YYYY-MM-DD). Defaults to latest valid.")
    parser.add_argument("--compare-date", type=str, default=None, help="Optional second date for side-by-side surface.")
    parser.add_argument(
        "--smile-tenors",
        type=str,
        default=None,
        help="Comma-separated tenors for smile plot. Defaults to all available tenors.",
    )
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite nodes for tranche pricing.")
    parser.add_argument("--grid-size", type=int, default=200, help="Grid size for rho search.")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib windows.")
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _curve_from_cum_hazard(tenors: np.ndarray, cum_hazard: np.ndarray) -> Curve:
    tenors = np.asarray(tenors, dtype=float)
    cum_hazard = np.asarray(cum_hazard, dtype=float)
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


def _row_quotes(row: pd.Series) -> tuple[Dict[float, float], Dict[float, float], List[float]]:
    tranche_spreads = {
        0.03: row["Equity_0_3_Spread"] / 10000.0,
        0.07: row["Mezz_3_7_Spread"] / 10000.0,
        0.10: row["Mezz_7_10_Spread"] / 10000.0,
        0.15: row["Senior_10_15_Spread"] / 10000.0,
    }
    tranche_upfronts = {
        0.03: row["Equity_0_3_Upfront"] / 100.0,
        0.07: row["Mezz_3_7_Upfront"] / 100.0,
    }
    dets = [0.03, 0.07, 0.10, 0.15]
    return tranche_spreads, tranche_upfronts, dets


def _load_index_df() -> pd.DataFrame:
    index_df = pd.read_csv(ROOT / "data" / "cdx_timeseries.csv", parse_dates=["Date"])
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
    ]
    _coerce_numeric(index_df, numeric_cols)
    return index_df


def _latest_valid_date(index_df: pd.DataFrame) -> date_type:
    required_cols = [
        "Index_Mid",
        "Index_0_100_Spread",
        "Equity_0_3_Spread",
        "Mezz_3_7_Spread",
        "Mezz_7_10_Spread",
        "Senior_10_15_Spread",
    ]
    valid = index_df.dropna(subset=required_cols)
    if valid.empty:
        raise SystemExit("No valid dates with complete index/tranche quotes.")
    return valid["Date"].max().date()


def _calibrate_nodes_for_date(
    index_df: pd.DataFrame,
    target_date: date_type,
    n_quad: int,
    grid_size: int,
) -> Tuple[Dict[float, Dict[float, float]], np.ndarray]:
    snapshot = index_df[index_df["Date"].dt.date == target_date].copy()
    if snapshot.empty:
        raise SystemExit(f"No rows found for date {target_date}.")

    disc_curve = read_ois_discount_curve(ROOT / "data" / "ois_timeseries.csv", target_date)
    (
        tenors_basis,
        theoretical_curve,
        _market_curve,
        _beta_knot,
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

    tenors_sorted = np.sort(snapshot["tenor"].to_numpy(dtype=float))
    basecorr_by_tenor: Dict[float, Dict[float, float]] = {}
    dropped_tenors: List[float] = []
    for tenor in tenors_sorted:
        row = snapshot[snapshot["tenor"] == tenor]
        if row.empty:
            continue
        row = row.iloc[0]
        tranche_spreads, tranche_upfronts, dets = _row_quotes(row)
        basecorr, solve_status = calibrate_basecorr_relaxed(
            tenor,
            dets,
            tranche_spreads,
            tranche_upfronts,
            curve_adjusted,
            recovery=0.4,
            grid_size=grid_size,
            n_quad=n_quad,
            payment_freq=4,
            disc_curve=disc_curve,
            return_status=True,
        )
        brent_dets = [float(k) for k in dets if solve_status.get(k, "") == "BRENT"]
        brent_rhos = [float(basecorr[k]) for k in brent_dets]
        if len(brent_dets) >= 2:
            # Fill missing det nodes from tenor-level PCHIP using available BRENT points.
            interp = PchipInterpolator(np.array(brent_dets, dtype=float), np.array(brent_rhos, dtype=float), extrapolate=True)
            basecorr_by_tenor[float(tenor)] = {float(k): float(interp(float(k))) for k in dets}
            logging.info(
                "Tenor %.2f partial-BRENT nodes kept | brent_dets=%s filled_dets=%s",
                tenor,
                brent_dets,
                [float(k) for k in dets],
            )
        else:
            dropped_tenors.append(float(tenor))
    kept_tenors = np.array(sorted(basecorr_by_tenor.keys()), dtype=float)
    logging.info(
        "BRENT-only tenor filter on %s | kept=%s dropped=%s",
        target_date,
        kept_tenors.tolist(),
        dropped_tenors,
    )
    if kept_tenors.size == 0:
        raise SystemExit(f"No tenors with full BRENT statuses on {target_date}.")
    return basecorr_by_tenor, kept_tenors


def _parse_smile_tenors(text: str | None, available: np.ndarray) -> List[float]:
    if not text:
        return [float(t) for t in available.tolist()]
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    available_set = set(available.tolist())
    chosen = [v for v in vals if v in available_set]
    return chosen if chosen else [float(t) for t in available.tolist()]


def _plot_smiles(
    basecorr_by_tenor: Dict[float, Dict[float, float]],
    smile_tenors: List[float],
    outpath: Path,
    title_suffix: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for tenor in smile_tenors:
        if tenor not in basecorr_by_tenor:
            continue
        pts = basecorr_by_tenor[tenor]
        det_nodes = np.array(sorted(pts.keys()), dtype=float)
        rho_nodes = np.array([pts[k] for k in det_nodes], dtype=float)
        pchip = PchipInterpolator(det_nodes, rho_nodes, extrapolate=True)
        det_dense = np.linspace(det_nodes.min(), det_nodes.max(), 200)
        rho_dense = np.asarray(pchip(det_dense), dtype=float)
        sample_det = np.array([det_nodes.min(), np.median(det_nodes), det_nodes.max()], dtype=float)
        sample_rho = np.asarray(pchip(sample_det), dtype=float)
        logging.info(
            "PCHIP smile tenor %.2fY | nodes(det=%s, rho=%s) | samples(det=%s, rho=%s)",
            tenor,
            np.round(det_nodes, 6).tolist(),
            np.round(rho_nodes, 6).tolist(),
            np.round(sample_det, 6).tolist(),
            np.round(sample_rho, 6).tolist(),
        )
        plt.plot(det_dense * 100.0, rho_dense, label=f"{int(tenor)}Y Smile")
        plt.scatter(det_nodes * 100.0, rho_nodes, s=18)
    plt.title(f"Continuous Base Correlation Smiles (PCHIP){title_suffix}")
    plt.xlabel("Detachment (%)")
    plt.ylabel("Base Correlation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)


def _surface_grid(basecorr_by_tenor: Dict[float, Dict[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_fn = build_rho_surface(basecorr_by_tenor)
    tenors = np.array(sorted(basecorr_by_tenor.keys()), dtype=float)
    dets = np.array(sorted(next(iter(basecorr_by_tenor.values())).keys()), dtype=float)
    t_dense = np.linspace(tenors.min(), tenors.max(), 60)
    d_dense = np.linspace(dets.min(), dets.max(), 80)
    T, D = np.meshgrid(t_dense, d_dense)
    Z = np.zeros_like(T, dtype=float)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = surface_fn(float(T[i, j]), float(D[i, j]))
    return T, D, Z


def _plot_surface_single(basecorr_by_tenor: Dict[float, Dict[float, float]], outpath: Path, title: str) -> None:
    T, D, Z = _surface_grid(basecorr_by_tenor)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, D * 100.0, Z, cmap="viridis", edgecolor="k", linewidth=0.2)
    ax.set_title(title)
    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("Detachment (%)")
    ax.set_zlabel("Base Correlation")
    fig.colorbar(surf, shrink=0.6, label="Base Correlation")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)


def _plot_surface_compare(
    left: Dict[float, Dict[float, float]],
    right: Dict[float, Dict[float, float]],
    left_label: str,
    right_label: str,
    outpath: Path,
) -> None:
    T1, D1, Z1 = _surface_grid(left)
    T2, D2, Z2 = _surface_grid(right)
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    s1 = ax1.plot_surface(T1, D1 * 100.0, Z1, cmap="viridis", edgecolor="k", linewidth=0.2)
    s2 = ax2.plot_surface(T2, D2 * 100.0, Z2, cmap="viridis", edgecolor="k", linewidth=0.2)
    ax1.set_title(f"Continuous Base Correlation Surface: {left_label}")
    ax2.set_title(f"Continuous Base Correlation Surface: {right_label}")
    for ax in (ax1, ax2):
        ax.set_xlabel("Tenor (Years)")
        ax.set_ylabel("Detachment (%)")
        ax.set_zlabel("Base Correlation")
    fig.colorbar(s1, ax=ax1, shrink=0.6)
    fig.colorbar(s2, ax=ax2, shrink=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    index_df = _load_index_df()
    target_date = pd.to_datetime(args.date).date() if args.date else _latest_valid_date(index_df)
    basecorr_by_tenor, tenors_sorted = _calibrate_nodes_for_date(
        index_df=index_df,
        target_date=target_date,
        n_quad=args.n_quad,
        grid_size=args.grid_size,
    )
    smile_tenors = _parse_smile_tenors(args.smile_tenors, tenors_sorted)

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    date_tag = target_date.strftime("%m%d")
    smile_path = outdir / f"continuous_basecorr_smiles_pchip_{target_date}.png"
    _plot_smiles(basecorr_by_tenor, smile_tenors, smile_path, title_suffix=f": {date_tag}")
    logging.info("Saved smile plot to %s", smile_path)

    if args.compare_date:
        compare_date = pd.to_datetime(args.compare_date).date()
        basecorr_cmp, _ = _calibrate_nodes_for_date(
            index_df=index_df,
            target_date=compare_date,
            n_quad=args.n_quad,
            grid_size=args.grid_size,
        )
        compare_path = outdir / f"continuous_basecorr_surface_compare_{target_date}_vs_{compare_date}.png"
        _plot_surface_compare(
            basecorr_by_tenor,
            basecorr_cmp,
            left_label=target_date.strftime("%m%d"),
            right_label=compare_date.strftime("%m%d"),
            outpath=compare_path,
        )
        logging.info("Saved compare surface plot to %s", compare_path)
    else:
        surface_path = outdir / f"continuous_basecorr_surface_{target_date}.png"
        _plot_surface_single(
            basecorr_by_tenor,
            surface_path,
            title=f"Continuous Base Correlation Surface: {date_tag}",
        )
        logging.info("Saved surface plot to %s", surface_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
