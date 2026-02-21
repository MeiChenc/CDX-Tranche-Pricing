from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle
from src.calibration_basecorr_relaxed import calibrate_basecorr_relaxed
from src.curves import Curve
from src.io_data import read_ois_discount_curve
from src.pricer_tranche import price_tranche_lhp, generate_base_el


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot base expected loss curves and flag monotonicity vs detachment "
            "for each tenor."
        )
    )
    parser.add_argument("--date", type=str, default=None, help="Date to plot (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument(
        "--tenors",
        type=str,
        default=None,
        help="Comma-separated tenor years to plot (e.g., 3,5,7,10). Defaults to all available.",
    )
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite nodes for pricing.")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size for bracketing search.")
    parser.add_argument(
        "--payment-freq",
        type=int,
        default=4,
        help="Premium leg payment frequency per year (default 4).",
    )
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for figures.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING).")
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _map_recovery_value(value: float) -> float:
    if pd.isna(value):
        raise ValueError("Missing recovery value for constituent")
    v = float(value)
    if v <= 1.0:
        return v
    if int(v) == 100:
        return 0.4
    if int(v) == 500:
        return 0.25
    raise ValueError(f"Unsupported recovery code: {value}")


def _parse_tenor(text: str) -> float:
    value = text.strip().upper()
    if value.endswith("Y"):
        return float(value[:-1])
    if value.endswith("M"):
        return float(value[:-1]) / 12.0
    if value.endswith("W"):
        return float(value[:-1]) / 52.0
    return float(value)


def _round_tenor(value: float, ndigits: int = 6) -> float:
    return float(round(float(value), ndigits))


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


def _parse_tenors(text: str | None, available: np.ndarray) -> np.ndarray:
    if not text:
        return available
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return np.array(values, dtype=float)


def _monotonic_non_decreasing(values: np.ndarray, tol: float = 1e-10) -> bool:
    if values.size <= 1:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs >= -tol))


def _extrapolate_linear(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x.size < 2:
        return np.full_like(x_new, y[0], dtype=float)
    y_new = np.interp(x_new, x, y)
    left_mask = x_new < x[0]
    right_mask = x_new > x[-1]
    if left_mask.any():
        slope_left = (y[1] - y[0]) / (x[1] - x[0])
        y_new[left_mask] = y[0] + slope_left * (x_new[left_mask] - x[0])
    if right_mask.any():
        slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
        y_new[right_mask] = y[-1] + slope_right * (x_new[right_mask] - x[-1])
    return y_new


def _pchip_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    n = x.size
    if n == 1:
        return np.full_like(x_new, y[0], dtype=float)
    if n == 2:
        return _extrapolate_linear(x, y, x_new)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)

    d0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif (np.sign(delta[0]) != np.sign(delta[1])) and (abs(d0) > abs(3 * delta[0])):
        d0 = 3 * delta[0]
    d[0] = d0

    dn = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif (np.sign(delta[-1]) != np.sign(delta[-2])) and (abs(dn) > abs(3 * delta[-1])):
        dn = 3 * delta[-1]
    d[-1] = dn

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    y_new = np.empty_like(x_new, dtype=float)
    for j, xq in enumerate(x_new):
        if xq <= x[0]:
            y_new[j] = y[0] + d[0] * (xq - x[0])
            continue
        if xq >= x[-1]:
            y_new[j] = y[-1] + d[-1] * (xq - x[-1])
            continue
        i = np.searchsorted(x, xq) - 1
        hi = x[i + 1] - x[i]
        t = (xq - x[i]) / hi
        h00 = (2 * t**3 - 3 * t**2 + 1)
        h10 = (t**3 - 2 * t**2 + t)
        h01 = (-2 * t**3 + 3 * t**2)
        h11 = (t**3 - t**2)
        y_new[j] = h00 * y[i] + h10 * hi * d[i] + h01 * y[i + 1] + h11 * hi * d[i + 1]
    return y_new


def _cubic_spline_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x.size == 1:
        return np.full_like(x_new, y[0], dtype=float)
    if x.size == 2:
        return _extrapolate_linear(x, y, x_new)
    spline = CubicSpline(x, y, bc_type="natural", extrapolate=True)
    return np.asarray(spline(x_new), dtype=float)


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
        "Equity_0_3_Spread",
        "Mezz_3_7_Spread",
        "Mezz_7_10_Spread",
        "Senior_10_15_Spread",
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
        valid = index_df.dropna(subset=required_cols)
        if valid.empty:
            raise SystemExit("No dates with complete tranche quotes found.")
        target_date = valid["Date"].max().date()
        snapshot = valid[valid["Date"].dt.date == target_date].copy()

    disc_curve = read_ois_discount_curve(str(ROOT / "data" / "ois_timeseries.csv"), target_date)
    tenors_beta, theoretical_curve, _, beta_knot, beta_cum, _, _ = build_index_dual_curve_beta_bundle(
        snapshot,
        disc_curve=disc_curve,
        recovery_index=0.4,
        theoretical_col="Index_0_100_Spread",
        market_col="Index_Mid",
    )
    theo_H = -np.log(np.maximum(np.array([theoretical_curve.survival(float(t)) for t in tenors_beta]), 1e-12))
    adjusted_H = beta_cum * theo_H
    adjusted_curve = _curve_from_cum_hazard(tenors_beta, adjusted_H)
    logging.info("Using beta_cum-adjusted curve for 'Adjusted' run.")
    logging.info("DEBUG: beta_knot by tenor: %s", dict(zip(tenors_beta, beta_knot)))
    logging.info("DEBUG: beta_cum by tenor: %s", dict(zip(tenors_beta, beta_cum)))
    label = "Adjusted"
    curve = adjusted_curve

    tenors_available = snapshot["tenor"].to_numpy(dtype=float)
    tenors_plot = _parse_tenors(args.tenors, tenors_available)

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fig_base, ax_base = plt.subplots(figsize=(9, 5))
    monotonic_flags = {
    }
    residual_rows: List[dict] = []

    for tenor in tenors_plot:
        row = snapshot.loc[snapshot["tenor"] == tenor]
        if row.empty:
            logging.warning("Tenor %.2f not found for date %s; skipping.", tenor, target_date)
            continue
        row = row.iloc[0]

        tranche_spreads, tranche_upfronts, dets = _row_quotes(row)
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
            payment_freq=args.payment_freq,
            disc_curve=disc_curve,
        )
        logging.info(
            "Tenor %.2f basecorr: %s | Spread:%s | Upfront:%s",
            tenor,
            {d: basecorr.get(d, np.nan) for d in dets},
            tranche_spreads,
            tranche_upfronts,
        )
        for idx_det, det in enumerate(dets):
            rho = float(basecorr.get(det, np.nan))
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
                payment_freq=args.payment_freq,
                disc_curve=disc_curve,
            )
            residual = model_prot - spread * model_prem - upfront * (k2 - k1)
            logging.info(
                "%s legs tenor %.2f [%.2f, %.2f]: rho=%.6f prot=%.10f prem=%.10f spread_mkt=%.6f upfront=%.6f residual=%.10f",
                label,
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
                    "curve_label": label,
                    "tenor": float(tenor),
                    "k1": k1,
                    "k2": k2,
                    "rho": rho,
                    "spread_running": spread,
                    "upfront": upfront,
                    "model_protection_leg": model_prot,
                    "model_premium_leg": model_prem,
                    "residual": residual,
                }
            )

        dets_plot = [0.0] + dets
        base_els = [0.0]
        for det in dets:
            rho = basecorr[det]
            base_el = generate_base_el(
                tenor,
                0.0,
                det,
                rho,
                curve,
                recovery=0.4,
                n_quad=args.n_quad,
                payment_freq=args.payment_freq,
            )
            base_els.append(base_el)

        base_els_arr = np.array(base_els, dtype=float)
        is_monotonic = _monotonic_non_decreasing(base_els_arr)
        monotonic_flags[float(tenor)] = is_monotonic

        series_label = f"{tenor:.1f}Y ({'mono' if is_monotonic else 'non-mono'})"
        ax_base.plot(dets_plot, base_els, marker="o", label=series_label)

    ax_base.set_title(f"Base Expected Loss Curves ({label} curve)")
    ax_base.set_xlabel("Detachment")
    ax_base.set_ylabel("Base Expected Loss")
    ax_base.grid(True, alpha=0.3)
    ax_base.legend()

    fname = outdir / f"base_expected_loss_curves_{label.lower()}_{target_date}.png"
    fig_base.tight_layout()
    fig_base.savefig(fname, dpi=150)
    logging.info("Saved plot to %s", fname)

    residual_df = pd.DataFrame(residual_rows)
    residual_path = outdir / f"basecorr_residuals_{label.lower()}_{target_date}.csv"
    residual_df.to_csv(residual_path, index=False)
    if not residual_df.empty:
        abs_res = np.abs(residual_df["residual"].to_numpy(dtype=float))
        logging.info(
            "%s residual summary | mean_abs=%.6e max_abs=%.6e",
            label,
            float(np.mean(abs_res)),
            float(np.max(abs_res)),
        )
    logging.info("Saved residual diagnostics to %s", residual_path)

    if monotonic_flags:
        summary = ", ".join(
            f"{tenor:.1f}Y={('OK' if flag else 'FAIL')}" for tenor, flag in sorted(monotonic_flags.items())
        )
        logging.info("%s monotonicity check: %s", label, summary)


if __name__ == "__main__":
    main()
