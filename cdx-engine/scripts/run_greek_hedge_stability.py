from __future__ import annotations

"""Greek and hedge-ratio stability analytics across dates."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk import compute_dv01
from scripts.risk_engine_common import (
    TRANCHES,
    build_adjusted_index_curve,
    calibrate_surface_on_snapshot,
    clip_rho,
    configure_logging,
    configure_plot_style,
    ensure_directory,
    get_snapshot_for_date,
    load_discount_curve_for_date,
    load_index_timeseries,
    list_available_dates,
    tranche_pv_from_surface,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greek stability and hedge-ratio stability analytics.")
    parser.add_argument("--date-from", type=str, default=None, help="Start date YYYY-MM-DD (inclusive).")
    parser.add_argument("--date-to", type=str, default=None, help="End date YYYY-MM-DD (inclusive).")
    parser.add_argument("--max-days", type=int, default=15, help="Maximum dates to analyze.")
    parser.add_argument("--recovery", type=float, default=0.40, help="Recovery rate assumption.")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite quadrature nodes.")
    parser.add_argument("--grid-size", type=int, default=120, help="Calibration grid size.")
    parser.add_argument("--payment-freq", type=int, default=4, help="Premium payment frequency.")
    parser.add_argument("--outdir", type=str, default="outputs/run_greek_hedge_stability", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _selected_dates(df: pd.DataFrame, date_from: str | None, date_to: str | None, max_days: int) -> list[pd.Timestamp]:
    dates = list_available_dates(df)
    if date_from:
        floor = pd.to_datetime(date_from).normalize()
        dates = [d for d in dates if d >= floor]
    if date_to:
        ceil = pd.to_datetime(date_to).normalize()
        dates = [d for d in dates if d <= ceil]
    if max_days > 0 and len(dates) > max_days:
        dates = dates[-max_days:]
    return dates


def _rho_delta_stability(
    tenor: float,
    k1: float,
    k2: float,
    det_map: dict[float, float],
    curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
) -> tuple[float, float, float]:
    bump_grid = [0.0025, 0.0050, 0.0100]
    estimates: list[float] = []
    node = k2
    for bump in bump_grid:
        up = dict(det_map)
        dn = dict(det_map)
        up[node] = clip_rho(up[node] + bump)
        dn[node] = clip_rho(dn[node] - bump)
        denom = up[node] - dn[node]
        if np.isclose(denom, 0.0):
            continue
        pv_up = tranche_pv_from_surface(
            tenor=tenor,
            k1=k1,
            k2=k2,
            surface_by_det=up,
            curve=curve,
            recovery=recovery,
            n_quad=n_quad,
            payment_freq=payment_freq,
            disc_curve=disc_curve,
        ).pv
        pv_dn = tranche_pv_from_surface(
            tenor=tenor,
            k1=k1,
            k2=k2,
            surface_by_det=dn,
            curve=curve,
            recovery=recovery,
            n_quad=n_quad,
            payment_freq=payment_freq,
            disc_curve=disc_curve,
        ).pv
        estimates.append((pv_up - pv_dn) / denom)
    if not estimates:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(estimates, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    cv = std / max(abs(mean), 1e-12)
    return mean, std, cv


def _plot_stability(summary: pd.DataFrame, out_png: Path) -> None:
    configure_plot_style()
    metrics = [
        ("dv01_cv_mean", "DV01 CV (mean)", "#1f77b4"),
        ("hedge_ratio_cv_mean", "Hedge Ratio CV (mean)", "#2ca02c"),
        ("rho_delta_cv_mean", "Corr Delta CV (mean)", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    s = summary.sort_values("tranche")
    x = np.arange(len(s))
    for ax, (col, title, color) in zip(axes, metrics):
        ax.bar(x, s[col].to_numpy(dtype=float), color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(s["tranche"].tolist(), rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Greek and Hedge-Ratio Stability Diagnostics")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    outdir = ensure_directory(ROOT / args.outdir)
    data_out = ensure_directory(outdir / "data")
    plot_out = ensure_directory(outdir / "plots")
    data_dir = ROOT / "data"
    index_df = load_index_timeseries(data_dir / "cdx_timeseries.csv")
    dates = _selected_dates(index_df, args.date_from, args.date_to, args.max_days)
    if not dates:
        raise ValueError("No dates selected for stability analysis.")

    detail_rows: list[dict] = []
    for dt in dates:
        snapshot = get_snapshot_for_date(index_df, dt)
        disc_curve = load_discount_curve_for_date(data_dir, dt)
        curve = build_adjusted_index_curve(snapshot, disc_curve=disc_curve, recovery=args.recovery)
        calib = calibrate_surface_on_snapshot(
            snapshot=snapshot,
            curve=curve,
            recovery=args.recovery,
            n_quad=args.n_quad,
            grid_size=args.grid_size,
            payment_freq=args.payment_freq,
            disc_curve=disc_curve,
        )
        for tenor, det_map in calib.surface.items():
            for tranche_name, k1, k2 in TRANCHES:
                if k2 not in det_map:
                    continue
                rho = clip_rho(det_map[k2])
                dv = compute_dv01(
                    tenor=float(tenor),
                    k1=float(k1),
                    k2=float(k2),
                    rho=rho,
                    curve=curve,
                    recovery=args.recovery,
                    payment_freq=args.payment_freq,
                )
                rho_delta_mean, rho_delta_std, rho_delta_cv = _rho_delta_stability(
                    tenor=float(tenor),
                    k1=float(k1),
                    k2=float(k2),
                    det_map=det_map,
                    curve=curve,
                    recovery=args.recovery,
                    n_quad=args.n_quad,
                    payment_freq=args.payment_freq,
                    disc_curve=disc_curve,
                )
                detail_rows.append(
                    {
                        "date": dt.date().isoformat(),
                        "tenor": float(tenor),
                        "tranche": tranche_name,
                        "rho_node": rho,
                        "tranche_dv01": float(dv.tranche_dv01),
                        "index_dv01": float(dv.index_dv01),
                        "hedge_ratio": float(dv.hedge_ratio),
                        "rho_delta_mean": rho_delta_mean,
                        "rho_delta_std": rho_delta_std,
                        "rho_delta_cv": rho_delta_cv,
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        raise ValueError("No greek stability rows produced.")

    def _cv(x: pd.Series) -> float:
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        m = float(np.mean(arr))
        s = float(np.std(arr))
        return s / max(abs(m), 1e-12)

    summary_df = (
        detail_df.groupby("tranche", as_index=False)
        .agg(
            dv01_mean=("tranche_dv01", "mean"),
            dv01_std=("tranche_dv01", "std"),
            dv01_cv_mean=("tranche_dv01", _cv),
            hedge_ratio_mean=("hedge_ratio", "mean"),
            hedge_ratio_std=("hedge_ratio", "std"),
            hedge_ratio_cv_mean=("hedge_ratio", _cv),
            rho_delta_mean=("rho_delta_mean", "mean"),
            rho_delta_std=("rho_delta_mean", "std"),
            rho_delta_cv_mean=("rho_delta_mean", _cv),
        )
        .sort_values("tranche")
    )

    end_date = max(dates).date().isoformat()
    detail_csv = data_out / f"greek_hedge_stability_detail_to_{end_date}.csv"
    summary_csv = data_out / f"greek_hedge_stability_summary_to_{end_date}.csv"
    out_png = plot_out / f"greek_hedge_stability_{end_date}.png"
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _plot_stability(summary_df, out_png)

    logging.info("Saved detail table to %s", detail_csv)
    logging.info("Saved summary table to %s", summary_csv)
    logging.info("Saved stability plot to %s", out_png)


if __name__ == "__main__":
    main()
