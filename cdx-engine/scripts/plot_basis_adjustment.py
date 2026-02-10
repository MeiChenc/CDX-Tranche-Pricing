from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basis_adjustment_utils import build_basis_adjustment_bundle
from src.curves import Curve
from src.io_data import read_ois_discount_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot basis adjustment diagnostics (expected loss comparison + beta curve)."
    )
    parser.add_argument("--date", type=str, default=None, help="Date to plot (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for figures.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING).")
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


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


def _load_index_snapshot(target_date: date_type) -> pd.DataFrame:
    index_df = pd.read_csv(ROOT / "data" / "cdx_timeseries.csv", parse_dates=["Date"])
    index_df["Tenor"] = index_df["Tenor"].astype(str).str.upper()
    index_df["tenor"] = index_df["Tenor"].str.replace("Y", "", regex=False).astype(float)
    _coerce_numeric(index_df, ["Index_Mid"])
    snapshot = index_df[index_df["Date"].dt.date == target_date].copy()
    if snapshot.empty:
        raise SystemExit(f"No index rows found for date {target_date}.")
    return snapshot


def _latest_valid_date() -> date_type:
    index_df = pd.read_csv(ROOT / "data" / "cdx_timeseries.csv", parse_dates=["Date"])
    index_df["Tenor"] = index_df["Tenor"].astype(str).str.upper()
    index_df["tenor"] = index_df["Tenor"].str.replace("Y", "", regex=False).astype(float)
    _coerce_numeric(index_df, ["Index_Mid"])
    valid = index_df.dropna(subset=["Index_Mid"])
    if valid.empty:
        raise SystemExit("No dates with complete index quotes found.")
    return valid["Date"].max().date()


def _build_basis_inputs(
    target_date: date_type,
    disc_curve,
    recovery: float,
) -> tuple[np.ndarray, np.ndarray, Curve, Curve, List[Curve], np.ndarray, np.ndarray]:
    snapshot = _load_index_snapshot(target_date)
    all_tenors = snapshot["tenor"].to_numpy(dtype=float)
    index_spreads = snapshot["Index_Mid"].to_numpy(dtype=float) / 10000.0
    logging.info("DEBUG: First 5 index spreads: %s", index_spreads[:5])
    logging.info("DEBUG: First 5 tenors: %s", all_tenors[:5])
    index_curve_full = build_index_curve(all_tenors, index_spreads, recovery=recovery, disc_curve=disc_curve)
    logging.info("DEBUG: Index lambdas by tenor: %s", dict(zip(all_tenors, index_curve_full.hazard)))
    surv = {float(t): index_curve_full.survival(float(t)) for t in all_tenors}
    dprob = {float(t): index_curve_full.default_prob(float(t)) for t in all_tenors}
    logging.info("DEBUG: Index survival by tenor: %s", surv)
    logging.info("DEBUG: Index default prob by tenor: %s", dprob)

    cons_df = pd.read_csv(ROOT / "data" / "constituents_timeseries.csv", parse_dates=["Date"])
    cons_slice = cons_df[cons_df["Date"].dt.date == target_date].copy()
    if cons_slice.empty:
        raise SystemExit(f"No constituent data for date {target_date}.")

    spread_cols: Dict[float, str] = {}
    for col in cons_slice.columns:
        if col.lower().startswith("spread_"):
            tenor = _parse_tenor(col.split("_", 1)[1])
            spread_cols[_round_tenor(tenor)] = col

    tenor_list = [float(t) for t in all_tenors if _round_tenor(float(t)) in spread_cols]
    if not tenor_list:
        raise SystemExit("No overlapping tenors between index and constituent spreads.")

    cols = [spread_cols[_round_tenor(t)] for t in tenor_list]
    cons_spreads = cons_slice[cols].dropna(how="any")
    if cons_spreads.empty:
        raise SystemExit("Constituent spreads empty after filtering.")

    spreads_matrix = cons_spreads.to_numpy(dtype=float) / 10000.0
    cons_meta = cons_slice.loc[cons_spreads.index, ["Company", "Recovery"]]
    recoveries = cons_meta["Recovery"].apply(_map_recovery_value).to_numpy(dtype=float)
    constituent_curves = [
        bootstrap_from_cds_spreads(
            tenor_list,
            issuser_spreads,
            float(rec),
            payment_freq=4,
            disc_curve=disc_curve,
        )
        for issuser_spreads, rec in zip(spreads_matrix, recoveries)
    ]
    for name, rec_raw, rec_mapped, curve in zip(
        cons_meta["Company"].astype(str).to_numpy(),
        cons_meta["Recovery"].to_numpy(),
        recoveries,
        constituent_curves,
    ):
        logging.info(
            "Issuer %s | Recovery raw=%s mapped=%.4f | Lambdas=%s",
            name,
            rec_raw,
            float(rec_mapped),
            curve.hazard,
        )
    tenors = np.array(tenor_list, dtype=float)
    index_spreads = snapshot[snapshot["tenor"].isin(tenor_list)]["Index_Mid"].to_numpy(dtype=float) / 10000.0
    index_curve = build_index_curve(tenors, index_spreads, recovery=recovery, disc_curve=disc_curve)
    return all_tenors, tenors, index_curve_full, index_curve, constituent_curves, spreads_matrix, recoveries


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

    # Endpoints
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

    # Interior points
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # Evaluate
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




def _expected_losses(curves: List[Curve], times: np.ndarray, recoveries: np.ndarray) -> np.ndarray:
    losses = np.zeros_like(times, dtype=float)
    for curve, rec in zip(curves, recoveries):
        losses += np.array([(1.0 - float(rec)) * curve.default_prob(float(t)) for t in times], dtype=float)
    return losses / float(len(curves))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    target_date = pd.to_datetime(args.date).date() if args.date else _latest_valid_date()
    disc_curve = read_ois_discount_curve(ROOT / "data" / "ois_timeseries.csv", target_date)

    index_snapshot = _load_index_snapshot(target_date)
    cons_df = pd.read_csv(ROOT / "data" / "constituents_timeseries.csv", parse_dates=["Date"])
    index_curve_full, adjusted_avg_curve, all_tenors, betas, expanded_curves, recoveries = (
        build_basis_adjustment_bundle(target_date, index_snapshot, cons_df, disc_curve, recovery_index=0.4)
    )

    if adjusted_avg_curve is None or expanded_curves == []:
        logging.warning("Basis adjustment skipped; using index curve only.")
        adjusted_avg_curve = index_curve_full
        expanded_curves = []
        recoveries = np.array([], dtype=float)

    index_el = np.array([(1.0 - 0.4) * index_curve_full.default_prob(float(t)) for t in all_tenors], dtype=float)
    avg_const_el = (
        _expected_losses(expanded_curves, all_tenors, recoveries=recoveries)
        if expanded_curves
        else np.zeros_like(all_tenors, dtype=float)
    )
    avg_adj_el = np.array(
        [(1.0 - 0.4) * adjusted_avg_curve.default_prob(float(t)) for t in all_tenors], dtype=float
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(all_tenors, index_el, marker="o", label="Index Expected Loss")
    ax1.plot(all_tenors, avg_const_el, marker="x", linestyle="--", label="Avg Constituent Expected Loss")
    ax1.plot(all_tenors, avg_adj_el, marker="s", linestyle=":", label="Avg Basis-Adjusted Expected Loss")
    ax1.set_title("Expected Loss Curves: Index vs. Constituent Average")
    ax1.set_xlabel("Tenor (Years)")
    ax1.set_ylabel("Expected Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(outdir / f"expected_loss_compare_{target_date}.png", dpi=200)

    if betas.size:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.plot(all_tenors, betas, marker="o", color="tab:blue")
        ax2.set_title("Basis Adjustment Function beta(t)")
        ax2.set_xlabel("Tenor (Years)")
        ax2.set_ylabel("Beta")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(outdir / f"basis_beta_{target_date}.png", dpi=200)

    logging.info("Saved plots to %s for date %s", outdir, target_date)


if __name__ == "__main__":
    main()
