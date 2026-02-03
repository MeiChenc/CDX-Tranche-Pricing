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

from src.basis import basis_adjust_curves_beta, build_average_curve
from src.curves import Curve, build_constituent_curves, build_index_curve
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


def _load_index_snapshot(target_date: date_type) -> pd.DataFrame:
    index_df = pd.read_csv("data/cdx_timeseries.csv", parse_dates=["Date"])
    index_df["Tenor"] = index_df["Tenor"].astype(str).str.upper()
    index_df["tenor"] = index_df["Tenor"].str.replace("Y", "", regex=False).astype(float)
    _coerce_numeric(index_df, ["Index_Mid"])
    snapshot = index_df[index_df["Date"].dt.date == target_date].copy()
    if snapshot.empty:
        raise SystemExit(f"No index rows found for date {target_date}.")
    return snapshot


def _latest_valid_date() -> date_type:
    index_df = pd.read_csv("data/cdx_timeseries.csv", parse_dates=["Date"])
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
) -> tuple[np.ndarray, Curve, List[Curve], np.ndarray]:
    snapshot = _load_index_snapshot(target_date)
    tenors = snapshot["tenor"].to_numpy(dtype=float)
    index_spreads = snapshot["Index_Mid"].to_numpy(dtype=float) / 10000.0
    index_curve = build_index_curve(tenors, index_spreads, recovery=recovery, disc_curve=disc_curve)

    cons_df = pd.read_csv("data/constituents_timeseries.csv", parse_dates=["Date"])
    cons_slice = cons_df[cons_df["Date"].dt.date == target_date].copy()
    if cons_slice.empty:
        raise SystemExit(f"No constituent data for date {target_date}.")

    spread_cols: Dict[float, str] = {}
    for col in cons_slice.columns:
        if col.lower().startswith("spread_"):
            tenor = _parse_tenor(col.split("_", 1)[1])
            spread_cols[tenor] = col

    tenor_list = [float(t) for t in tenors if float(t) in spread_cols]
    if not tenor_list:
        raise SystemExit("No overlapping tenors between index and constituent spreads.")

    cols = [spread_cols[t] for t in tenor_list]
    cons_spreads = cons_slice[cols].dropna(how="any")
    if cons_spreads.empty:
        raise SystemExit("Constituent spreads empty after filtering.")

    spreads_matrix = cons_spreads.to_numpy(dtype=float) / 10000.0
    constituent_curves = build_constituent_curves(
        tenor_list,
        spreads_matrix,
        recovery=recovery,
        payment_freq=4,
        disc_curve=disc_curve,
    )
    tenors = np.array(tenor_list, dtype=float)
    index_spreads = snapshot[snapshot["tenor"].isin(tenor_list)]["Index_Mid"].to_numpy(dtype=float) / 10000.0
    index_curve = build_index_curve(tenors, index_spreads, recovery=recovery, disc_curve=disc_curve)
    return tenors, index_curve, constituent_curves, spreads_matrix


def _expected_losses(curves: List[Curve], times: np.ndarray, recovery: float) -> np.ndarray:
    losses = np.zeros_like(times, dtype=float)
    for curve in curves:
        losses += np.array([(1.0 - recovery) * curve.default_prob(float(t)) for t in times], dtype=float)
    return losses / float(len(curves))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    target_date = pd.to_datetime(args.date).date() if args.date else _latest_valid_date()
    disc_curve = read_ois_discount_curve("data/ois_timeseries.csv", target_date)

    tenors, index_curve, constituent_curves, _ = _build_basis_inputs(
        target_date, disc_curve, recovery=0.4
    )
    adjusted_curves, betas = basis_adjust_curves_beta(constituent_curves, index_curve, recovery=0.4)
    adjusted_avg_curve = build_average_curve(adjusted_curves)

    index_el = np.array([(1.0 - 0.4) * index_curve.default_prob(float(t)) for t in tenors], dtype=float)
    avg_const_el = _expected_losses(constituent_curves, tenors, recovery=0.4)
    avg_adj_el = np.array(
        [(1.0 - 0.4) * adjusted_avg_curve.default_prob(float(t)) for t in tenors], dtype=float
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(tenors, index_el, marker="o", label="Index Expected Loss")
    ax1.plot(tenors, avg_const_el, marker="x", linestyle="--", label="Avg Constituent Expected Loss")
    ax1.plot(tenors, avg_adj_el, marker="s", linestyle=":", label="Avg Basis-Adjusted Expected Loss")
    ax1.set_title("Expected Loss Curves: Index vs. Constituent Average")
    ax1.set_xlabel("Tenor (Years)")
    ax1.set_ylabel("Expected Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(outdir / f"expected_loss_compare_{target_date}.png", dpi=200)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(tenors, betas, marker="o", color="tab:blue")
    ax2.set_title("Basis Adjustment Function beta(t)")
    ax2.set_xlabel("Tenor (Years)")
    ax2.set_ylabel("Beta")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / f"basis_beta_{target_date}.png", dpi=200)

    logging.info("Saved plots to %s for date %s", outdir, target_date)


if __name__ == "__main__":
    main()
