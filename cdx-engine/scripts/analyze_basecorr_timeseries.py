from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle
from src.calibration_basecorr_relaxed import calibrate_basecorr_relaxed
from src.curves import Curve
from src.io_data import read_ois_discount_curve
from src.pricer_tranche import price_tranche_lhp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run daily base-correlation calibration diagnostics over a date range."
    )
    parser.add_argument("--date-from", type=str, default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--date-to", type=str, default=None, help="End date YYYY-MM-DD.")
    parser.add_argument("--max-days", type=int, default=30, help="Max number of dates to process.")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite nodes.")
    parser.add_argument("--grid-size", type=int, default=200, help="Grid size for rho search.")
    parser.add_argument("--payment-freq", type=int, default=4, help="Premium leg payment frequency.")
    parser.add_argument("--ewma-span", type=int, default=5, help="EWMA span for smoothed overlays.")
    parser.add_argument("--run-pca", action="store_true", help="Run PCA on daily node surfaces.")
    parser.add_argument("--key-points", type=str, default="5:0.10,10:0.15,7:0.10", help="rho(t,k) tracking points.")
    parser.add_argument("--beta-tenors", type=str, default="1,5,10", help="beta(t) tracking tenors.")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
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
    return float(pv_k2.protection_leg - pv_k1.protection_leg), float(pv_k2.premium_leg - pv_k1.premium_leg)


def _parse_key_points(text: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        t_str, k_str = item.split(":")
        out.append((float(t_str), float(k_str)))
    return out


def _parse_beta_tenors(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _surface_roughness(surface: np.ndarray) -> float:
    if surface.size == 0:
        return float("nan")
    arr = np.asarray(surface, dtype=float)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return float(np.nanmean(np.abs(np.diff(arr, axis=0)))) if arr.size else float("nan")
    d2_det = arr[:-2, :] - 2.0 * arr[1:-1, :] + arr[2:, :]
    d2_ten = arr[:, :-2] - 2.0 * arr[:, 1:-1] + arr[:, 2:]
    return float(np.nanmean(np.abs(d2_det)) + np.nanmean(np.abs(d2_ten)))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

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

    required_cols = [
        "Index_Mid",
        "Index_0_100_Spread",
        "Equity_0_3_Spread",
        "Mezz_3_7_Spread",
        "Mezz_7_10_Spread",
        "Senior_10_15_Spread",
    ]
    valid = index_df.dropna(subset=required_cols)
    if args.date_from:
        valid = valid[valid["Date"].dt.date >= pd.to_datetime(args.date_from).date()]
    if args.date_to:
        valid = valid[valid["Date"].dt.date <= pd.to_datetime(args.date_to).date()]

    # Known bad date: drop all November 24 rows before building daily snapshots.
    bad_mask = (valid["Date"].dt.month == 11) & (valid["Date"].dt.day == 24)
    dropped_bad_rows = int(bad_mask.sum())
    if dropped_bad_rows > 0:
        valid = valid[~bad_mask].copy()
        logging.info("Dropped known bad rows for 11/24: %d", dropped_bad_rows)

    dates = sorted(valid["Date"].dt.date.unique())
    if args.max_days > 0:
        dates = dates[-args.max_days :]
    if not dates:
        raise SystemExit("No valid dates found in the selected range.")

    dets = [0.03, 0.07, 0.10, 0.15]
    tenor_sets = []
    for dt in dates:
        snap = valid[valid["Date"].dt.date == dt].copy()
        tenor_sets.append(set(np.sort(snap["tenor"].to_numpy(dtype=float)).tolist()))
    common_tenors = sorted(set.intersection(*tenor_sets)) if tenor_sets else []
    if not common_tenors:
        raise SystemExit("No common tenor grid across selected dates.")
    logging.info("Fixed grid | tenors=%s dets=%s", common_tenors, dets)

    daily_rows: List[dict] = []
    node_rows: List[dict] = []
    rho_cube: List[np.ndarray] = []
    idx_spread_series: List[float] = []
    used_dates: List[pd.Timestamp] = []

    for dt in dates:
        snapshot = valid[valid["Date"].dt.date == dt].copy()
        if snapshot.empty:
            continue
        try:
            disc_curve = read_ois_discount_curve(ROOT / "data" / "ois_timeseries.csv", dt)
        except Exception as exc:
            logging.warning("Skip %s: OIS curve not available (%s)", dt, exc)
            continue

        (
            tenors_basis,
            theoretical_curve,
            market_curve,
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

        snapshot = snapshot.set_index("tenor")
        if any(t not in snapshot.index for t in common_tenors):
            logging.warning("Skip %s: missing tenor(s) on fixed grid", dt)
            continue

        tenors_sorted = np.array(common_tenors, dtype=float)
        surface = np.full((len(tenors_sorted), len(dets)), np.nan, dtype=float)
        residuals: List[float] = []
        statuses: List[str] = []
        boundary_hits = 0
        mono_violations = 0

        for j, tenor in enumerate(tenors_sorted):
            row = snapshot.loc[tenor]
            tranche_spreads, tranche_upfronts, dets_local = _row_quotes(row)
            basecorr, solve_status = calibrate_basecorr_relaxed(
                tenor,
                dets_local,
                tranche_spreads,
                tranche_upfronts,
                curve_adjusted,
                recovery=0.4,
                grid_size=args.grid_size,
                n_quad=args.n_quad,
                payment_freq=args.payment_freq,
                disc_curve=disc_curve,
                return_status=True,
            )

            rho_tenor = np.array([float(basecorr.get(k, np.nan)) for k in dets], dtype=float)
            if np.any(np.diff(rho_tenor) < -1e-8):
                mono_violations += 1

            for i, det in enumerate(dets):
                rho = float(basecorr.get(det, np.nan))
                rho = float(np.clip(rho, 0.0, 1.0))
                status = solve_status.get(det, "UNKNOWN")
                statuses.append(status)
                if rho <= 1e-3 or rho >= 0.998:
                    boundary_hits += 1
                surface[j, i] = rho
                k1 = 0.0 if i == 0 else dets[i - 1]
                rho_k1 = None if i == 0 else float(basecorr.get(k1, np.nan))
                spread = float(tranche_spreads.get(det, 0.0))
                upfront = float(tranche_upfronts.get(det, 0.0))
                model_prot, model_prem = _model_legs_from_basecorr(
                    tenor=tenor,
                    k1=k1,
                    k2=det,
                    rho_k2=rho,
                    rho_k1=rho_k1,
                    curve=curve_adjusted,
                    recovery=0.4,
                    n_quad=args.n_quad,
                    payment_freq=args.payment_freq,
                    disc_curve=disc_curve,
                )
                residual = model_prot - spread * model_prem - upfront * (det - k1)
                residuals.append(abs(residual))
                node_rows.append(
                    {
                        "date": str(dt),
                        "tenor": float(tenor),
                        "detachment": float(det),
                        "rho": rho,
                        "status": status,
                        "residual": residual,
                    }
                )

        spread_dispersion_bps = float(
            np.nanmean(
                snapshot["Index_0_100_Spread"].to_numpy(dtype=float) - snapshot["Index_Mid"].to_numpy(dtype=float)
            )
        )
        mid_t_ref = 5.0 if 5.0 in snapshot.index else float(tenors_sorted[np.argmin(np.abs(tenors_sorted - 5.0))])
        mid_ref = float(snapshot.loc[mid_t_ref]["Index_Mid"])

        daily_rows.append(
            {
                "date": str(dt),
                "fit_mean_abs_error": float(np.nanmean(residuals)) if residuals else float("nan"),
                "fit_max_abs_error": float(np.nanmax(residuals)) if residuals else float("nan"),
                "brent_ratio": float(np.mean(np.array(statuses) == "BRENT")) if statuses else float("nan"),
                "no_bracket_count": int(np.sum(np.array(statuses) == "NO_BRACKET_MINABS")) if statuses else 0,
                "boundary_hits": int(boundary_hits),
                "mono_violations": int(mono_violations),
                "surface_roughness": _surface_roughness(surface),
                "index_mid_ref_bps": mid_ref,
                "index_mid_ref_tenor": mid_t_ref,
                "index_basis_dispersion_bps": spread_dispersion_bps,
            }
        )
        rho_cube.append(surface)
        idx_spread_series.append(mid_ref)
        used_dates.append(pd.to_datetime(str(dt)))

    if not daily_rows:
        raise SystemExit("No daily results were produced.")

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    daily_df = pd.DataFrame(daily_rows).sort_values("date")
    nodes_df = pd.DataFrame(node_rows).sort_values(["date", "tenor", "detachment"])
    rho_tensor = np.stack(rho_cube, axis=0)  # [day, tenor, detachment]
    if not np.all(np.isfinite(rho_tensor)):
        raise SystemExit("rho tensor contains NaN/Inf, cannot proceed with fixed-grid analytics.")
    if np.any(rho_tensor < -1e-12) or np.any(rho_tensor > 1.0 + 1e-12):
        raise SystemExit("rho out of [0,1] bounds after clipping/assembly.")

    daily_path = outdir / "basecorr_daily_diagnostics.csv"
    nodes_path = outdir / "basecorr_node_timeseries.csv"
    daily_df.to_csv(daily_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)

    # Daily moves: Delta(d) = d - (d-1), with dates sorted ascending.
    rho_diff = np.diff(rho_tensor, axis=0)  # [day-1, tenor, detachment]
    abs_diff = np.abs(rho_diff)
    if not np.all(np.isfinite(rho_diff)):
        raise SystemExit("Delta rho contains NaN/Inf.")

    mean_abs = np.mean(abs_diff, axis=0)  # [tenor, detachment]
    p95_abs = np.percentile(abs_diff, 95, axis=0)

    # Heatmap outputs
    def _plot_heatmap(mat: np.ndarray, title: str, outpath: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Detachment")
        ax.set_ylabel("Tenor")
        ax.set_xticks(np.arange(len(dets)))
        ax.set_xticklabels([f"{int(k*100)}%" for k in dets])
        ax.set_yticks(np.arange(len(common_tenors)))
        ax.set_yticklabels([f"{int(t)}Y" if float(t).is_integer() else f"{t:.1f}Y" for t in common_tenors])
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outpath, dpi=180)

    _plot_heatmap(mean_abs, "Mean |Δrho| across days", outdir / "heatmap_mean_abs_delta_rho.png")
    _plot_heatmap(p95_abs, "P95 |Δrho| across days", outdir / "heatmap_p95_abs_delta_rho.png")
    pd.DataFrame(
        mean_abs,
        index=[f"{t}Y" for t in common_tenors],
        columns=[f"{int(k*100)}%" for k in dets],
    ).to_csv(outdir / "basecorr_abs_delta_heatmap_matrix.csv")

    # surface_move_stats.json
    move_stats = {
        "global": {
            "max_abs_delta_rho": float(np.max(abs_diff)),
            "median_abs_delta_rho": float(np.median(abs_diff)),
            "p95_abs_delta_rho": float(np.percentile(abs_diff, 95)),
        },
        "by_tenor_detachment": {},
    }
    for i_t, tenor in enumerate(common_tenors):
        for i_k, det in enumerate(dets):
            vals = abs_diff[:, i_t, i_k]
            move_stats["by_tenor_detachment"][f"{tenor}Y@{int(det*100)}%"] = {
                "max_abs_delta_rho": float(np.max(vals)),
                "median_abs_delta_rho": float(np.median(vals)),
                "p95_abs_delta_rho": float(np.percentile(vals, 95)),
            }
    with open(outdir / "surface_move_stats.json", "w", encoding="utf-8") as f:
        json.dump(move_stats, f, indent=2)

    # PCA on daily surface moves
    M = rho_diff.reshape(rho_diff.shape[0], -1)
    M_centered = M - np.mean(M, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    n_comp = min(3, Vt.shape[0])
    scores = U[:, :n_comp] * S[:n_comp]
    var = (S**2) / max(1, M_centered.shape[0] - 1)
    var_ratio = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)

    pca_scores = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in used_dates[1:]],
            **{f"pc{i+1}": scores[:, i] for i in range(n_comp)},
        }
    )
    pca_scores.to_csv(outdir / "pc_scores_timeseries.csv", index=False)

    # Explained variance plot
    fig_var, ax_var = plt.subplots(figsize=(7, 4))
    x = np.arange(1, n_comp + 1)
    ax_var.bar(x, var_ratio[:n_comp])
    ax_var.set_xticks(x)
    ax_var.set_xlabel("Principal Component")
    ax_var.set_ylabel("Explained Variance Ratio")
    ax_var.set_title("PCA Explained Variance (Surface Moves)")
    ax_var.grid(True, axis="y", alpha=0.3)
    fig_var.tight_layout()
    fig_var.savefig(outdir / "pca_explained_variance.png", dpi=180)

    # Loading heatmaps (PC1..PC3)
    for i in range(n_comp):
        loading = Vt[i, :].reshape(len(common_tenors), len(dets))
        fig_l, ax_l = plt.subplots(figsize=(8, 6))
        im = ax_l.imshow(loading, aspect="auto", origin="lower", cmap="coolwarm")
        ax_l.set_title(f"PC{i+1} Loadings Heatmap")
        ax_l.set_xlabel("Detachment")
        ax_l.set_ylabel("Tenor")
        ax_l.set_xticks(np.arange(len(dets)))
        ax_l.set_xticklabels([f"{int(k*100)}%" for k in dets])
        ax_l.set_yticks(np.arange(len(common_tenors)))
        ax_l.set_yticklabels([f"{int(t)}Y" if float(t).is_integer() else f"{t:.1f}Y" for t in common_tenors])
        fig_l.colorbar(im, ax=ax_l)
        fig_l.tight_layout()
        fig_l.savefig(outdir / f"pc{i+1}_loadings_heatmap.png", dpi=180)

    # PC1 vs index spread change correlation (with p-value + simple regression summary)
    idx_spread = np.asarray(idx_spread_series, dtype=float)
    d_spread = np.diff(idx_spread)  # aligned with used_dates[1:]
    pc1 = scores[:, 0] if n_comp >= 1 else np.array([])
    corr_val = float("nan")
    p_val = float("nan")
    slope = float("nan")
    intercept = float("nan")
    r2 = float("nan")
    if pc1.size > 1 and d_spread.size == pc1.size:
        corr_val, p_val = pearsonr(pc1, d_spread)
        slope, intercept = np.polyfit(d_spread, pc1, 1)
        pred = slope * d_spread + intercept
        ss_res = float(np.sum((pc1 - pred) ** 2))
        ss_tot = float(np.sum((pc1 - np.mean(pc1)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    with open(outdir / "pc1_vs_spreaddiff_corr.txt", "w", encoding="utf-8") as f:
        f.write(f"n_obs={pc1.size}\n")
        f.write(f"pearson_corr={corr_val}\n")
        f.write(f"pearson_p_value={p_val}\n")
        f.write(f"ols_slope={slope}\n")
        f.write(f"ols_intercept={intercept}\n")
        f.write(f"ols_r2={r2}\n")

    logging.info("Saved diagnostics table: %s", daily_path)
    logging.info("Saved node timeseries: %s", nodes_path)
    logging.info("Saved heatmaps and move stats under %s", outdir)
    logging.info(
        "PCA summary | explained_var=%s | pc1_vs_dSpread corr=%.4f p=%.4g",
        np.round(var_ratio[:n_comp], 6).tolist(),
        corr_val,
        p_val,
    )
    plt.show()


if __name__ == "__main__":
    main()
