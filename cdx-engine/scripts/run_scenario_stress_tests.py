from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curves import Curve
from scripts.risk_engine_common import (
    TRANCHES,
    build_adjusted_index_curve,
    calibrate_surface_on_snapshot,
    choose_target_date,
    clip_rho,
    configure_logging,
    configure_plot_style,
    ensure_directory,
    get_snapshot_for_date,
    load_discount_curve_for_date,
    load_index_timeseries,
    tranche_pv_from_surface,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDX tranche scenario and stress testing engine.")
    parser.add_argument("--date", type=str, default=None, help="Market date YYYY-MM-DD (default: latest available).")
    parser.add_argument("--recovery", type=float, default=0.40, help="Recovery rate assumption.")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite quadrature nodes.")
    parser.add_argument("--grid-size", type=int, default=120, help="Calibration grid size.")
    parser.add_argument("--payment-freq", type=int, default=4, help="Premium payment frequency.")
    parser.add_argument("--outdir", type=str, default="outputs/run_scenario_stress_tests", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _shock_curve(curve: Curve, fn: Callable[[float], float]) -> Curve:
    multipliers = np.array([fn(float(t)) for t in curve.times], dtype=float)
    multipliers = np.maximum(multipliers, 1e-8)
    return Curve(times=np.asarray(curve.times, dtype=float), hazard=np.asarray(curve.hazard, dtype=float) * multipliers)


def _shock_surface(surface: Dict[float, Dict[float, float]], fn: Callable[[float, float], float]) -> Dict[float, Dict[float, float]]:
    shocked: Dict[float, Dict[float, float]] = {}
    for tenor, det_map in surface.items():
        shocked[tenor] = {}
        for det, rho in det_map.items():
            shocked[tenor][det] = clip_rho(fn(float(tenor), float(det)) + float(rho))
    return shocked


def _build_scenarios() -> list[dict]:
    return [
        {
            "name": "spread_parallel_widen_25pct",
            "curve_fn": lambda t: 1.25,
            "surface_fn": lambda t, d: 0.0,
        },
        {
            "name": "spread_parallel_tighten_20pct",
            "curve_fn": lambda t: 0.80,
            "surface_fn": lambda t, d: 0.0,
        },
        {
            "name": "spread_curve_steepener",
            "curve_fn": lambda t: 1.15 if t <= 3.0 else (1.00 if t <= 7.0 else 0.90),
            "surface_fn": lambda t, d: 0.0,
        },
        {
            "name": "corr_parallel_up_5pt",
            "curve_fn": lambda t: 1.0,
            "surface_fn": lambda t, d: 0.05,
        },
        {
            "name": "corr_parallel_down_5pt",
            "curve_fn": lambda t: 1.0,
            "surface_fn": lambda t, d: -0.05,
        },
        {
            "name": "corr_smile_steepen",
            "curve_fn": lambda t: 1.0,
            "surface_fn": lambda t, d: 0.03 * (d - 0.08) / 0.08,
        },
        {
            "name": "combined_tail_stress",
            "curve_fn": lambda t: 1.35 if t <= 5.0 else 1.25,
            "surface_fn": lambda t, d: 0.07,
        },
    ]


def _price_grid(
    surface: Dict[float, Dict[float, float]],
    curve: Curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
) -> dict[tuple[float, str], float]:
    out: dict[tuple[float, str], float] = {}
    for tenor, det_map in sorted(surface.items()):
        for tranche_name, k1, k2 in TRANCHES:
            pv = tranche_pv_from_surface(
                tenor=float(tenor),
                k1=float(k1),
                k2=float(k2),
                surface_by_det=det_map,
                curve=curve,
                recovery=recovery,
                n_quad=n_quad,
                payment_freq=payment_freq,
                disc_curve=disc_curve,
            ).pv
            out[(float(tenor), tranche_name)] = float(pv)
    return out


def _plot_scenario_results(summary_df: pd.DataFrame, worst_df: pd.DataFrame, out_png: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    plot_df = summary_df.copy().sort_values("pnl_total")
    x = np.arange(len(plot_df))
    axes[0].bar(x, plot_df["pnl_total"], color=np.where(plot_df["pnl_total"] < 0, "#c0392b", "#2e86c1"))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["scenario"], rotation=40, ha="right")
    axes[0].set_ylabel("Total PnL")
    axes[0].set_title("Portfolio-Level Scenario PnL")

    pivot = worst_df.pivot(index="tranche", columns="scenario", values="pnl")
    mat = pivot.to_numpy(dtype=float)
    if mat.size == 0:
        axes[1].axis("off")
    else:
        vmax = np.nanpercentile(np.abs(mat), 95)
        vmax = max(vmax, 1e-10)
        im = axes[1].imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_yticks(np.arange(len(pivot.index)))
        axes[1].set_yticklabels(pivot.index)
        axes[1].set_xticks(np.arange(len(pivot.columns)))
        axes[1].set_xticklabels(pivot.columns, rotation=35, ha="right")
        axes[1].set_title("Tranche PnL Under Worst Scenarios")
        fig.colorbar(im, ax=axes[1], shrink=0.85, label="PnL")

    fig.suptitle("CDX Tranche Stress Testing Dashboard", fontsize=14)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    data_dir = ROOT / "data"
    outdir = ensure_directory(ROOT / args.outdir)
    data_out = ensure_directory(outdir / "data")
    plot_out = ensure_directory(outdir / "plots")

    index_df = load_index_timeseries(data_dir / "cdx_timeseries.csv")
    target_date = choose_target_date(index_df, args.date)
    snapshot = get_snapshot_for_date(index_df, target_date)
    disc_curve = load_discount_curve_for_date(data_dir, target_date)

    base_curve = build_adjusted_index_curve(snapshot, disc_curve=disc_curve, recovery=args.recovery)
    calib = calibrate_surface_on_snapshot(
        snapshot=snapshot,
        curve=base_curve,
        recovery=args.recovery,
        n_quad=args.n_quad,
        grid_size=args.grid_size,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
    )

    base_prices = _price_grid(
        surface=calib.surface,
        curve=base_curve,
        recovery=args.recovery,
        n_quad=args.n_quad,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
    )

    scenario_rows: list[dict] = []
    scenarios = _build_scenarios()

    for scn in scenarios:
        scn_name = scn["name"]
        shocked_curve = _shock_curve(base_curve, scn["curve_fn"])
        shocked_surface = _shock_surface(calib.surface, scn["surface_fn"])
        shocked_prices = _price_grid(
            surface=shocked_surface,
            curve=shocked_curve,
            recovery=args.recovery,
            n_quad=args.n_quad,
            payment_freq=args.payment_freq,
            disc_curve=disc_curve,
        )
        for (tenor, tranche), pv_base in base_prices.items():
            pv_shock = shocked_prices[(tenor, tranche)]
            pnl = pv_shock - pv_base
            scenario_rows.append(
                {
                    "date": target_date.date().isoformat(),
                    "scenario": scn_name,
                    "tenor": tenor,
                    "tranche": tranche,
                    "pv_base": pv_base,
                    "pv_scenario": pv_shock,
                    "pnl": pnl,
                    "pnl_pct_of_abs_base": pnl / max(abs(pv_base), 1e-8),
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)
    if scenario_df.empty:
        raise ValueError("No scenario outputs produced.")

    summary_df = (
        scenario_df.groupby("scenario", as_index=False)
        .agg(
            pnl_total=("pnl", "sum"),
            pnl_mean=("pnl", "mean"),
            pnl_worst=("pnl", "min"),
            pnl_best=("pnl", "max"),
        )
        .sort_values("pnl_total")
    )
    worst_df = (
        scenario_df.sort_values("pnl")
        .groupby(["tranche", "scenario"], as_index=False)
        .agg(pnl=("pnl", "sum"))
        .sort_values("pnl")
        .groupby("tranche", as_index=False)
        .head(3)
    )

    detail_csv = data_out / f"scenario_stress_detail_{target_date.date()}.csv"
    summary_csv = data_out / f"scenario_stress_summary_{target_date.date()}.csv"
    worst_csv = data_out / f"scenario_stress_worst_{target_date.date()}.csv"
    plot_png = plot_out / f"scenario_stress_{target_date.date()}.png"

    scenario_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    worst_df.to_csv(worst_csv, index=False)
    _plot_scenario_results(summary_df, worst_df, plot_png)

    logging.info("Scenario detail saved to %s", detail_csv)
    logging.info("Scenario summary saved to %s", summary_csv)
    logging.info("Worst-case tranche table saved to %s", worst_csv)
    logging.info("Scenario chart saved to %s", plot_png)


if __name__ == "__main__":
    main()
