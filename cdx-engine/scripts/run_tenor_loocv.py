from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _latest_valid_date(df: pd.DataFrame) -> date_type:
    req = [
        "Index_Mid",
        "Index_0_100_Spread",
        "Equity_0_3_Spread",
        "Equity_0_3_Upfront",
        "Mezz_3_7_Spread",
        "Mezz_3_7_Upfront",
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
    upfronts = {
        0.03: float(row["Equity_0_3_Upfront"]) / 100.0,
        0.07: float(row["Mezz_3_7_Upfront"]) / 100.0,
    }
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


def _implied_upfront_pct(model_prot: float, model_prem: float, running_spread: float, width: float) -> float:
    if width <= 0.0:
        return float("nan")
    return 100.0 * float((model_prot - running_spread * model_prem) / width)


def _print_table(hide_tenor: float, rows: List[Tuple[str, float, float, float]]) -> None:
    print(f"--- Running Tenor LOOCV (Hiding {hide_tenor:.1f}Y) ---")
    print(f"{'Tranche':<8} | {'Mkt Price':>10} | {'Pred Price':>10} | {'Upfront(bp)':>11} | {'Diff':>8}")
    for label, mkt, pred, upfront_bp in rows:
        diff = pred - mkt
        print(f"{label:<8} | {mkt:>10.2f} | {pred:>10.2f} | {upfront_bp:>11.2f} | {diff:>8.2f}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    df = pd.read_csv(ROOT / "data" / "cdx_timeseries.csv", parse_dates=["Date"])
    df["Tenor"] = df["Tenor"].astype(str).str.upper()
    df["tenor"] = df["Tenor"].str.replace("Y", "", regex=False).astype(float)
    _coerce_numeric(
        df,
        [
            "Index_Mid",
            "Index_0_100_Spread",
            "Equity_0_3_Spread",
            "Equity_0_3_Upfront",
            "Mezz_3_7_Spread",
            "Mezz_3_7_Upfront",
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

    basecorr_by_tenor: Dict[float, Dict[float, float]] = {}
    for tenor in sorted(train["tenor"].to_numpy(dtype=float)):
        row = train[np.abs(train["tenor"] - tenor) <= 1e-12].iloc[0]
        spreads, upfronts, _ = _row_quotes(row)
        basecorr = calibrate_basecorr_relaxed(
            tenor=tenor,
            dets=dets,
            tranche_spreads=spreads,
            tranche_upfronts=upfronts,
            curve=curve,
            recovery=0.4,
            n_quad=args.n_quad,
            grid_size=args.grid_size,
            payment_freq=args.payment_freq,
            disc_curve=disc_curve,
        )
        basecorr_by_tenor[float(tenor)] = {float(k): float(basecorr[k]) for k in dets}

    rho_hat = build_rho_surface(basecorr_by_tenor)
    rho_pred = {k: float(np.clip(rho_hat(hide_tenor, k), 1e-4, 0.999)) for k in dets}
    logging.info("Predicted rho at hidden tenor %.1fY: %s", hide_tenor, rho_pred)

    spreads_test, upfronts_test, _ = _row_quotes(test_row)
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

        # Keep a single displayed "price" unit:
        # 0-3% as upfront points; others as running spread (bps), matching desk table style.
        if i == 0:
            mkt_price = 100.0 * upfronts_test[k2]
            pred_price = _implied_upfront_pct(model_prot, model_prem, spreads_test[k2], k2 - k1)
        else:
            mkt_price = 10000.0 * spreads_test[k2]
            pred_price = _implied_spread_bps(model_prot, model_prem)
        upfront_bp = 100.0 * float(upfronts_test.get(k2, 0.0))
        table_rows.append((labels[i], float(mkt_price), float(pred_price), upfront_bp))

    _print_table(hide_tenor, table_rows)


if __name__ == "__main__":
    main()
