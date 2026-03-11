from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.risk_engine_common import configure_logging, configure_plot_style, ensure_directory


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated tranche risk dashboard from generated analytics outputs.")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: latest available from node/scenario files).")
    parser.add_argument("--outdir", type=str, default="outputs/build_tranche_risk_dashboard", help="Risk analytics directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _extract_date_from_path(path: Path) -> pd.Timestamp | None:
    match = DATE_RE.search(path.name)
    if match is None:
        return None
    return pd.to_datetime(match.group(1), errors="coerce")


def _latest_file(paths: list[Path]) -> Path:
    dated: list[tuple[pd.Timestamp, Path]] = []
    for p in paths:
        dt = _extract_date_from_path(p)
        if dt is not None and not pd.isna(dt):
            dated.append((dt, p))
    if not dated:
        raise ValueError("No dated files found for required input set.")
    dated.sort(key=lambda x: x[0])
    return dated[-1][1]


def _resolve_file(outdir: Path, pattern: str, explicit_date: str | None) -> Path:
    if explicit_date is not None:
        candidate = outdir / pattern.format(date=explicit_date)
        if candidate.exists():
            return candidate
    matches = sorted(outdir.glob(pattern.replace("{date}", "*")))
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return _latest_file(matches)


def _plot_dashboard(consolidated: pd.DataFrame, scenario_pnl: pd.DataFrame, out_png: Path, report_date: str) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    plot_df = consolidated.sort_values("worst_scenario_pnl")
    ax1.bar(plot_df["tranche"], plot_df["max_abs_node_sensitivity"], color="#1f77b4")
    ax1.set_title("Max Abs Node Sensitivity")
    ax1.set_ylabel("|dPV/dRho|")
    ax1.tick_params(axis="x", rotation=25)

    ax2.bar(plot_df["tranche"], plot_df["worst_scenario_pnl"], color="#c0392b")
    ax2.set_title("Worst Scenario PnL")
    ax2.set_ylabel("PnL")
    ax2.tick_params(axis="x", rotation=25)

    scatter = ax3.scatter(
        consolidated["dv01_cv_mean"],
        consolidated["hedge_ratio_cv_mean"],
        s=80 + 250 * consolidated["max_abs_node_sensitivity"].fillna(0.0),
        c=consolidated["worst_scenario_pnl"],
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in consolidated.iterrows():
        ax3.annotate(row["tranche"], (row["dv01_cv_mean"], row["hedge_ratio_cv_mean"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax3.set_title("Stability Map")
    ax3.set_xlabel("DV01 CV (mean)")
    ax3.set_ylabel("Hedge Ratio CV (mean)")
    fig.colorbar(scatter, ax=ax3, label="Worst Scenario PnL", shrink=0.8)

    scenario_table = (
        scenario_pnl.groupby("scenario", as_index=False)["pnl"].sum().sort_values("pnl").head(6)
    )
    ax4.axis("off")
    table_vals = scenario_table.round(6).values.tolist()
    table = ax4.table(
        cellText=table_vals,
        colLabels=["Scenario", "Total PnL"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax4.set_title("Top 6 Worst Portfolio Scenarios")

    fig.suptitle(f"CDX Tranche Risk Dashboard ({report_date})", fontsize=15)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    outdir = ensure_directory(ROOT / args.outdir)
    data_out = ensure_directory(outdir / "data")
    plot_out = ensure_directory(outdir / "plots")

    node_summary_file = _resolve_file(
        outdir,
        "node_correlation_sensitivity_summary_{date}.csv",
        args.date,
    )
    scenario_detail_file = _resolve_file(
        outdir,
        "scenario_stress_detail_{date}.csv",
        args.date,
    )

    greek_matches = sorted(outdir.glob("greek_hedge_stability_summary_to_*.csv"))
    if not greek_matches:
        raise FileNotFoundError(
            "Missing Greek stability summary CSV. Run scripts/run_greek_hedge_stability.py first."
        )
    greek_file = _latest_file(greek_matches)

    node_df = pd.read_csv(node_summary_file)
    scenario_df = pd.read_csv(scenario_detail_file)
    greek_df = pd.read_csv(greek_file)

    node_tranche = (
        node_df.groupby("tranche", as_index=False)
        .agg(max_abs_node_sensitivity=("abs_d_pv_d_rho", "max"))
    )
    scenario_tranche = (
        scenario_df.groupby(["tranche", "scenario"], as_index=False)
        .agg(pnl=("pnl", "sum"))
        .sort_values("pnl")
        .groupby("tranche", as_index=False)
        .first()
        .rename(columns={"pnl": "worst_scenario_pnl", "scenario": "worst_scenario_name"})
    )
    greek_tranche = (
        greek_df.groupby("tranche", as_index=False)
        .agg(
            dv01_cv_mean=("dv01_cv_mean", "mean"),
            hedge_ratio_cv_mean=("hedge_ratio_cv_mean", "mean"),
            rho_delta_cv_mean=("rho_delta_cv_mean", "mean"),
        )
    )

    consolidated = node_tranche.merge(scenario_tranche, on="tranche", how="outer").merge(
        greek_tranche, on="tranche", how="outer"
    )
    consolidated = consolidated.sort_values("tranche")

    report_date = args.date
    if report_date is None:
        inferred = _extract_date_from_path(node_summary_file)
        report_date = inferred.date().isoformat() if inferred is not None else "latest"

    out_csv = data_out / f"tranche_risk_dashboard_{report_date}.csv"
    out_png = plot_out / f"tranche_risk_dashboard_{report_date}.png"

    consolidated.to_csv(out_csv, index=False)
    _plot_dashboard(consolidated, scenario_df, out_png, report_date)

    logging.info("Dashboard CSV saved to %s", out_csv)
    logging.info("Dashboard PNG saved to %s", out_png)


if __name__ == "__main__":
    main()
