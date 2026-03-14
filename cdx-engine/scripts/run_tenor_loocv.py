from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.pricer_tranche import price_tranche_lhp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tenor LOOCV: hide one tenor, fit rho surface, predict hidden-tenor prices.")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD. Defaults to latest valid date.")
    parser.add_argument("--hide-tenor", type=float, default=7.0, help="Hidden tenor in years. Default: 7.0")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite nodes for pricing.")
    parser.add_argument("--grid-size", type=int, default=200, help="Grid size for rho solver.")
    parser.add_argument("--payment-freq", type=int, default=4, help="Premium payment frequency per year.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "run_tenor_loocv"),
        help="Directory to save LOOCV tables/plots.",
    )
    parser.add_argument(
        "--smile-tenors",
        type=str,
        default="5,7,10",
        help="Comma-separated tenors to show in smile outputs (rho vs detachment).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation; still prints tables.",
    )
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _latest_valid_date(df: pd.DataFrame) -> date_type:
    req = [
        "Index_Mid",
        "Index_0_100_Spread",
        "Equity_0_3_Spread",
        "Mezz_3_7_Spread",
        "Mezz_7_10_Spread",
        "Senior_10_15_Spread",
    ]
    valid = df.dropna(subset=req)
    if valid.empty:
        raise SystemExit("No valid dates with complete quotes.")
    return valid["Date"].max().date()


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


def _build_basis_adjusted_curve(snapshot: pd.DataFrame, disc_curve) -> Curve:
    tenors_basis, theoretical_curve, _, _, beta_cum, _, _ = build_index_dual_curve_beta_bundle(
        snapshot,
        disc_curve=disc_curve,
        recovery_index=0.4,
        theoretical_col="Index_0_100_Spread",
        market_col="Index_Mid",
    )
    theo_h = -np.log(np.maximum(np.array([theoretical_curve.survival(float(t)) for t in tenors_basis]), 1e-12))
    adjusted_h = beta_cum * theo_h
    return _curve_from_cum_hazard(tenors_basis, adjusted_h)


def _row_quotes(row: pd.Series) -> tuple[Dict[float, float], Dict[float, float], List[float]]:
    spreads = {
        0.03: float(row["Equity_0_3_Spread"]) / 10000.0,
        0.07: float(row["Mezz_3_7_Spread"]) / 10000.0,
        0.10: float(row["Mezz_7_10_Spread"]) / 10000.0,
        0.15: float(row["Senior_10_15_Spread"]) / 10000.0,
    }
    upfronts = {0.03: 0.0, 0.07: 0.0}
    dets = [0.03, 0.07, 0.10, 0.15]
    return spreads, upfronts, dets


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


def _implied_spread_bps(model_prot: float, model_prem: float) -> float:
    if model_prem <= 1e-12:
        return float("nan")
    return 10000.0 * float(model_prot / model_prem)


def _print_table(hide_tenor: float, rows: List[Tuple[str, float, float, float]]) -> None:
    print(f"--- Running Tenor LOOCV (Hiding {hide_tenor:.1f}Y) ---")
    print(f"{'Tranche':<8} | {'Mkt Par Spread(bp)':>17} | {'Pred Par Spread(bp)':>18} | {'Diff(bp)':>10}")
    for label, mkt, pred, diff in rows:
        print(f"{label:<8} | {mkt:>14.2f} | {pred:>15.2f} | {diff:>10.2f}")


def _print_hidden_tenor_basecorr_diagnostics(
    hide_tenor: float, dets: List[float], true_rho: Dict[float, float], pred_rho: Dict[float, float]
) -> None:
    print(f"\n=== TRUE {hide_tenor:.1f}Y BASE CORR ===")
    print({f"{k:.0%}": float(true_rho[k]) for k in dets})
    print(f"=== PRED {hide_tenor:.1f}Y BASE CORR ===")
    print({f"{k:.0%}": float(pred_rho[k]) for k in dets})
    print("det  |  true   |  pred   |  diff(pred-true)")
    for k in dets:
        t = float(true_rho[k])
        p = float(pred_rho[k])
        print(f"{100.0 * k:>4.0f}% | {t:>7.4f} | {p:>7.4f} | {p - t:>15.4f}")


def _print_crossing_check(curves: Dict[float, Dict[float, float]], dets: List[float], title: str) -> None:
    tenors = sorted(curves.keys())
    if len(tenors) < 2:
        return
    print(f"\n=== {title} ===")
    for i in range(len(tenors) - 1):
        t1 = tenors[i]
        t2 = tenors[i + 1]
        diffs = np.array([float(curves[t2][k] - curves[t1][k]) for k in dets], dtype=float)
        has_cross = bool(np.any(diffs[:-1] * diffs[1:] < 0.0))
        status = "CROSSING DETECTED" if has_cross else "no crossing"
        print(f"{t1:.1f}Y vs {t2:.1f}Y: {status} | node diffs={np.array2string(diffs, precision=4)}")


def _parse_smile_tenors(text: str | None, hide_tenor: float) -> List[float]:
    if not text:
        return [5.0, hide_tenor, 10.0]
    out: List[float] = []
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        out = [5.0, hide_tenor, 10.0]
    return out


def _build_smile_grid(curve_nodes: Dict[float, float], det_grid: np.ndarray) -> np.ndarray:
    ks = np.array(sorted(curve_nodes.keys()), dtype=float)
    ys = np.array([curve_nodes[k] for k in ks], dtype=float)
    interp = PchipInterpolator(ks, ys, extrapolate=True)
    return np.asarray(interp(det_grid), dtype=float)


def _print_smile_table(smile_curves: Dict[float, Dict[float, float]], dets: List[float], title: str) -> None:
    tenors = sorted(smile_curves.keys())
    print(f"\n=== {title} ===")
    header = "Det(%) " + " ".join([f"| {t:>8.1f}Y" for t in tenors])
    print(header)
    for k in dets:
        row = f"{100.0*k:>6.1f} " + " ".join([f"| {float(smile_curves[t][k]):>8.4f}" for t in tenors])
        print(row)


def _save_loocv_smile_plot(
    out_path: Path,
    dets: List[float],
    known_curves: Dict[float, Dict[float, float]],
    hidden_true: Dict[float, float],
    hidden_pred: Dict[float, float],
    hide_tenor: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logging.warning("matplotlib unavailable, skip smile plot: %s", exc)
        return

    det_grid = np.linspace(min(dets), max(dets), 200)
    fig, ax = plt.subplots(figsize=(11, 6))

    for tenor in sorted(known_curves.keys()):
        if abs(tenor - hide_tenor) <= 1e-12:
            continue
        y = _build_smile_grid(known_curves[tenor], det_grid)
        ax.plot(100.0 * det_grid, y, lw=2.0, label=f"{tenor:.0f}Y Known")
        ax.scatter(100.0 * np.array(dets), [known_curves[tenor][k] for k in dets], s=25)

    y_true = _build_smile_grid(hidden_true, det_grid)
    y_pred = _build_smile_grid(hidden_pred, det_grid)
    ax.plot(100.0 * det_grid, y_true, lw=2.5, color="#2f7ed8", label=f"{hide_tenor:.0f}Y Actual")
    ax.plot(100.0 * det_grid, y_pred, lw=2.0, ls="--", color="#d9534f", label=f"{hide_tenor:.0f}Y Interpolated")
    ax.scatter(100.0 * np.array(dets), [hidden_true[k] for k in dets], s=55, color="#0a58ff", zorder=5)
    ax.scatter(100.0 * np.array(dets), [hidden_pred[k] for k in dets], s=60, marker="x", color="#d62828", zorder=6)

    ax.set_title("LOOCV: Calibrated vs. Interpolated Base Correlation Smiles")
    ax.set_xlabel("Detachment (%)")
    ax.set_ylabel("Base Correlation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info("Saved LOOCV smile plot to %s", out_path)


def _print_restricted_point_loocv_table(rows: List[Tuple[float, float, float, float]]) -> None:
    print("\n--- Running Restricted Point-Based LOOCV (Middle Tranches: [0.07, 0.10]) ---")
    print(f"{'Tenor':>7} | {'Detachment':>10} | {'Actual_Rho':>10} | {'Smile_Pred':>10}")
    for tenor, det, actual, pred in rows:
        print(f"{tenor:>7.1f} | {det:>10.2f} | {actual:>10.4f} | {pred:>10.4f}")


def _calibrate_surface_from_snapshot(
    snapshot_rows: pd.DataFrame,
    dets: List[float],
    curve: Curve,
    n_quad: int,
    grid_size: int,
    payment_freq: int,
    disc_curve,
) -> Dict[float, Dict[float, float]]:
    surface: Dict[float, Dict[float, float]] = {}
    for tenor in sorted(snapshot_rows["tenor"].to_numpy(dtype=float)):
        row = snapshot_rows[np.abs(snapshot_rows["tenor"] - tenor) <= 1e-12].iloc[0]
        spreads, upfronts, _ = _row_quotes(row)
        basecorr = calibrate_basecorr_relaxed(
            tenor=tenor,
            dets=dets,
            tranche_spreads=spreads,
            tranche_upfronts=upfronts,
            curve=curve,
            recovery=0.4,
            n_quad=n_quad,
            grid_size=grid_size,
            payment_freq=payment_freq,
            disc_curve=disc_curve,
            quote_convention="spread_only",
        )
        surface[float(tenor)] = {float(k): float(basecorr[k]) for k in dets}
    return surface


def _save_restricted_point_loocv_grid_plot(
    out_path: Path,
    tenor_curves: Dict[float, Dict[float, float]],
    hide_dets: List[float],
) -> List[Tuple[float, float, float, float]]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logging.warning("matplotlib unavailable, skip restricted point LOOCV grid plot: %s", exc)
        return []

    tenors = sorted(tenor_curves.keys())
    if not tenors or not hide_dets:
        return []

    nrows = len(hide_dets)
    ncols = len(tenors)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 3.8 * nrows), squeeze=False)
    det_grid = np.linspace(0.01, 0.20, 250)
    rows: List[Tuple[float, float, float, float]] = []

    for r, hide_det in enumerate(hide_dets):
        for c, tenor in enumerate(tenors):
            ax = axes[r, c]
            curve = tenor_curves[tenor]
            det_nodes = np.array(sorted(curve.keys()), dtype=float)
            rho_nodes = np.array([curve[k] for k in det_nodes], dtype=float)
            if hide_det not in curve:
                continue

            calibrated = PchipInterpolator(det_nodes, rho_nodes, extrapolate=True)
            keep_mask = np.abs(det_nodes - hide_det) > 1e-12
            interp_det_nodes = det_nodes[keep_mask]
            interp_rho_nodes = rho_nodes[keep_mask]
            interpolated = PchipInterpolator(interp_det_nodes, interp_rho_nodes, extrapolate=True)

            y_cal = np.asarray(calibrated(det_grid), dtype=float)
            y_int = np.asarray(interpolated(det_grid), dtype=float)
            actual_rho = float(curve[hide_det])
            pred_rho = float(interpolated(hide_det))
            err = pred_rho - actual_rho
            rows.append((float(tenor), float(hide_det), actual_rho, pred_rho))

            ax.plot(100.0 * det_grid, y_cal, color="#4156ff", lw=2.0, label="Calibrated" if (r == 0 and c == 0) else None)
            ax.plot(
                100.0 * det_grid,
                y_int,
                color="#ff3b30",
                lw=1.8,
                ls="--",
                label="Interpolated" if (r == 0 and c == 0) else None,
            )
            ax.scatter(
                100.0 * det_nodes,
                rho_nodes,
                s=18,
                color="black",
                label="Known" if (r == 0 and c == 0) else None,
                zorder=4,
            )
            ax.scatter(
                [100.0 * hide_det],
                [actual_rho],
                s=55,
                color="#0b3dff",
                label="Actual" if (r == 0 and c == 0) else None,
                zorder=6,
            )
            ax.scatter(
                [100.0 * hide_det],
                [pred_rho],
                s=70,
                marker="x",
                color="#ff2d20",
                linewidths=2.0,
                label="Predicted" if (r == 0 and c == 0) else None,
                zorder=7,
            )

            ax.text(
                1.0,
                float(np.nanmin(y_cal) + 0.01),
                f"Err: {err:+.4f}",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.35", alpha=0.75, pad=1.5),
            )
            if r == 0:
                ax.set_title(f"{tenor:.0f}Y Tenor")
            if c == 0:
                ax.set_ylabel(f"Corr (Hide {int(round(hide_det * 100))}%)")
            ax.set_xlim(0.0, 20.5)
            ax.grid(True, alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.985, 0.03))
    fig.suptitle("LOOCV: Calibrated vs. Interpolated Correlation Smiles", fontsize=18)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info("Saved restricted point LOOCV grid plot to %s", out_path)
    return rows


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    out_dir = Path(args.output_dir)

    df = pd.read_csv(ROOT / "data" / "cdx_timeseries.csv", parse_dates=["Date"])
    df["Tenor"] = df["Tenor"].astype(str).str.upper()
    df["tenor"] = df["Tenor"].str.replace("Y", "", regex=False).astype(float)
    _coerce_numeric(
        df,
        [
            "Index_Mid",
            "Index_0_100_Spread",
            "Equity_0_3_Spread",
            "Mezz_3_7_Spread",
            "Mezz_7_10_Spread",
            "Senior_10_15_Spread",
        ],
    )

    target_date = pd.to_datetime(args.date).date() if args.date else _latest_valid_date(df)
    snapshot = df[df["Date"].dt.date == target_date].copy()
    if snapshot.empty:
        raise SystemExit(f"No rows found for {target_date}.")

    hide_tenor = float(args.hide_tenor)
    if hide_tenor not in snapshot["tenor"].to_numpy(dtype=float):
        raise SystemExit(f"Hidden tenor {hide_tenor:.1f}Y not found for {target_date}.")

    disc_curve = read_ois_discount_curve(ROOT / "data" / "ois_timeseries.csv", target_date)
    curve = _build_basis_adjusted_curve(snapshot, disc_curve=disc_curve)

    train = snapshot[np.abs(snapshot["tenor"] - hide_tenor) > 1e-12].copy()
    test_row = snapshot[np.abs(snapshot["tenor"] - hide_tenor) <= 1e-12].iloc[0]
    dets = [0.03, 0.07, 0.10, 0.15]
    labels = ["0-3%", "3-7%", "7-10%", "10-15%"]

    basecorr_by_tenor = _calibrate_surface_from_snapshot(
        snapshot_rows=train,
        dets=dets,
        curve=curve,
        n_quad=args.n_quad,
        grid_size=args.grid_size,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
    )

    rho_hat = build_rho_surface(basecorr_by_tenor)
    rho_pred = {k: float(np.clip(rho_hat(hide_tenor, k), 1e-4, 0.999)) for k in dets}
    logging.info("Predicted rho at hidden tenor %.1fY: %s", hide_tenor, rho_pred)

    spreads_is, upfronts_is, _ = _row_quotes(test_row)
    basecorr_is = calibrate_basecorr_relaxed(
        tenor=hide_tenor,
        dets=dets,
        tranche_spreads=spreads_is,
        tranche_upfronts=upfronts_is,
        curve=curve,
        recovery=0.4,
        n_quad=args.n_quad,
        grid_size=args.grid_size,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
        quote_convention="spread_only",
    )
    _print_hidden_tenor_basecorr_diagnostics(
        hide_tenor=hide_tenor,
        dets=dets,
        true_rho={float(k): float(basecorr_is[k]) for k in dets},
        pred_rho={float(k): float(rho_pred[k]) for k in dets},
    )
    smile_tenors = _parse_smile_tenors(args.smile_tenors, hide_tenor=hide_tenor)
    pred_curves = {
        t: ({float(k): float(rho_pred[k]) for k in dets} if abs(t - hide_tenor) <= 1e-12 else basecorr_by_tenor[t])
        for t in smile_tenors
        if (abs(t - hide_tenor) <= 1e-12) or (t in basecorr_by_tenor)
    }
    true_curves = {
        t: ({float(k): float(basecorr_is[k]) for k in dets} if abs(t - hide_tenor) <= 1e-12 else basecorr_by_tenor[t])
        for t in smile_tenors
        if (abs(t - hide_tenor) <= 1e-12) or (t in basecorr_by_tenor)
    }
    _print_crossing_check(pred_curves, dets=dets, title="Smile Crossing Check (5Y/hidden/10Y, predicted hidden)")
    _print_crossing_check(true_curves, dets=dets, title="Smile Crossing Check (5Y/hidden/10Y, true hidden)")
    _print_smile_table(pred_curves, dets=dets, title="Interpolated Base Corr Smile Table (rho vs detachment)")
    _print_smile_table(true_curves, dets=dets, title="Calibrated/Actual Base Corr Smile Table (rho vs detachment)")

    smile_plot_path = out_dir / f"loocv_basecorr_smiles_hide_{hide_tenor:.1f}Y.png"
    if not args.no_plot:
        _save_loocv_smile_plot(
            out_path=smile_plot_path,
            dets=dets,
            known_curves=basecorr_by_tenor,
            hidden_true={float(k): float(basecorr_is[k]) for k in dets},
            hidden_pred={float(k): float(rho_pred[k]) for k in dets},
            hide_tenor=hide_tenor,
        )
        print(f"\nSaved smile plot: {smile_plot_path}")
        point_grid_plot_path = out_dir / f"loocv_restricted_point_grid_hide_{hide_tenor:.1f}Y.png"
        grid_rows = _save_restricted_point_loocv_grid_plot(
            out_path=point_grid_plot_path,
            tenor_curves=true_curves,
            hide_dets=[0.07, 0.10],
        )
        if grid_rows:
            _print_restricted_point_loocv_table(grid_rows)
            print(f"Saved restricted-point grid plot: {point_grid_plot_path}")

    spreads_test, _, _ = _row_quotes(test_row)
    table_rows: List[Tuple[str, float, float, float]] = []
    for i, k2 in enumerate(dets):
        k1 = 0.0 if i == 0 else dets[i - 1]
        rho_k2 = rho_pred[k2]
        rho_k1 = None if i == 0 else rho_pred[k1]
        model_prot, model_prem = _model_legs_from_basecorr(
            tenor=hide_tenor,
            k1=k1,
            k2=k2,
            rho_k2=rho_k2,
            rho_k1=rho_k1,
            curve=curve,
            recovery=0.4,
            n_quad=args.n_quad,
            payment_freq=args.payment_freq,
            disc_curve=disc_curve,
        )

        fair_spread_bps = _implied_spread_bps(model_prot, model_prem)
        mkt_spread_bps = 10000.0 * spreads_test[k2]
        spread_err_bps = float(fair_spread_bps - mkt_spread_bps)
        table_rows.append((labels[i], float(mkt_spread_bps), float(fair_spread_bps), spread_err_bps))

    _print_table(hide_tenor, table_rows)


if __name__ == "__main__":
    main()
