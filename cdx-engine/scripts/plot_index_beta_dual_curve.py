from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_type
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle
from src.io_data import read_ois_discount_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build theoretical vs market index curves and plot beta term structures."
    )
    parser.add_argument("--date", type=str, default=None, help="Date to run (YYYY-MM-DD). Defaults to latest valid.")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory.")
    parser.add_argument("--recovery", type=float, default=0.4, help="Index recovery assumption.")
    parser.add_argument("--theoretical-col", type=str, default="Index_0_100_Spread", help="Theoretical spread column (bps).")
    parser.add_argument("--market-col", type=str, default="Index_Mid", help="Market spread column (bps).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _coerce_index_frame(index_df: pd.DataFrame) -> pd.DataFrame:
    out = index_df.copy()
    out["Tenor"] = out["Tenor"].astype(str).str.upper()
    out["tenor"] = out["Tenor"].str.replace("Y", "", regex=False).astype(float)
    return out


def _latest_valid_date(index_df: pd.DataFrame, theoretical_col: str, market_col: str) -> date_type:
    valid = index_df.dropna(subset=[theoretical_col, market_col])
    if valid.empty:
        raise SystemExit(
            f"No dates with both {theoretical_col} and {market_col} available."
        )
    return valid["Date"].max().date()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    index_df = pd.read_csv(ROOT / "data" / "cdx_timeseries.csv", parse_dates=["Date"])
    index_df = _coerce_index_frame(index_df)

    if args.date:
        target_date = pd.to_datetime(args.date).date()
    else:
        target_date = _latest_valid_date(index_df, args.theoretical_col, args.market_col)

    snapshot = index_df[index_df["Date"].dt.date == target_date].copy()
    if snapshot.empty:
        raise SystemExit(f"No index rows found for date {target_date}.")

    disc_curve = read_ois_discount_curve(ROOT / "data" / "ois_timeseries.csv", target_date)
    (
        tenors,
        theoretical_curve,
        market_curve,
        beta_knot,
        beta_cum,
        theoretical_spreads,
        market_spreads,
    ) = build_index_dual_curve_beta_bundle(
        snapshot,
        disc_curve=disc_curve,
        recovery_index=args.recovery,
        theoretical_col=args.theoretical_col,
        market_col=args.market_col,
    )

    theo_lambda = np.asarray(theoretical_curve.hazard, dtype=float)
    mkt_lambda = np.asarray(market_curve.hazard, dtype=float)
    theo_surv = np.array([theoretical_curve.survival(float(t)) for t in tenors], dtype=float)
    mkt_surv = np.array([market_curve.survival(float(t)) for t in tenors], dtype=float)
    theo_el = (1.0 - args.recovery) * np.array(
        [theoretical_curve.default_prob(float(t)) for t in tenors], dtype=float
    )
    mkt_el = (1.0 - args.recovery) * np.array(
        [market_curve.default_prob(float(t)) for t in tenors], dtype=float
    )

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.DataFrame(
        {
            "tenor": tenors,
            "theoretical_spread_bps": theoretical_spreads * 10000.0,
            "market_spread_bps": market_spreads * 10000.0,
            "theoretical_lambda": theo_lambda,
            "market_lambda": mkt_lambda,
            "beta_knot": beta_knot,
            "beta_cum": beta_cum,
            "theoretical_survival": theo_surv,
            "market_survival": mkt_surv,
            "theoretical_el": theo_el,
            "market_el": mkt_el,
            "lambda_residual": mkt_lambda - beta_knot * np.maximum(theo_lambda, 0.0),
        }
    )
    csv_path = outdir / f"index_beta_dual_curve_{target_date}.csv"
    diagnostics.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(tenors, theoretical_spreads * 10000.0, marker="o", label=f"{args.theoretical_col}")
    ax.plot(tenors, market_spreads * 10000.0, marker="s", label=f"{args.market_col}")
    ax.set_title("Index Spreads (bps)")
    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("Spread (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(tenors, theo_lambda, marker="o", label="Theoretical lambda")
    ax.plot(tenors, mkt_lambda, marker="s", label="Market lambda")
    ax.set_title("Bootstrapped Hazard by Tenor")
    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("Hazard")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(tenors, beta_knot, marker="o", label="beta_knot")
    ax.plot(tenors, beta_cum, marker="s", label="beta_cum")
    ax.set_title("Beta Term Structure")
    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(tenors, theo_el, marker="o", label="Theoretical EL")
    ax.plot(tenors, mkt_el, marker="s", label="Market EL")
    ax.set_title("Expected Loss Comparison")
    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("EL")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plot_path = outdir / f"index_beta_dual_curve_{target_date}.png"
    fig.savefig(plot_path, dpi=200)

    logging.info("Saved diagnostics CSV: %s", csv_path)
    logging.info("Saved plot: %s", plot_path)
    logging.info(
        "Beta summary | knot[min=%.4f, max=%.4f] cum[min=%.4f, max=%.4f]",
        float(np.min(beta_knot)),
        float(np.max(beta_knot)),
        float(np.min(beta_cum)),
        float(np.max(beta_cum)),
    )


if __name__ == "__main__":
    main()
