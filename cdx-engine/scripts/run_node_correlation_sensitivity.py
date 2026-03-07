from __future__ import annotations

"""Node-level correlation sensitivity engine for CDX tranche pricing.

This script treats each calibrated base-correlation node rho(T, K) as a risk
factor. It bumps one node up/down by epsilon, rebuilds the interpolated
base-correlation surface, reprices target tranches, and computes central-
difference sensitivities.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interpolation import build_rho_surface
from src.pricer_tranche import TranchePV, price_tranche_lhp
from scripts.risk_engine_common import (
    DETACHMENTS,
    build_adjusted_index_curve,
    calibrate_surface_on_snapshot,
    choose_target_date,
    configure_logging,
    configure_plot_style,
    ensure_directory,
    get_snapshot_for_date,
    load_discount_curve_for_date,
    load_index_timeseries,
)

RHO_MIN = 1e-4
RHO_MAX = 0.999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bucketed tranche sensitivities to individual base-correlation nodes rho(T,K)."
    )
    parser.add_argument("--date", type=str, default=None, help="Market date YYYY-MM-DD (default: latest available).")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Node bump size for central difference.")
    parser.add_argument(
        "--target-tenors",
        type=str,
        default="1,2,3,5,7,10",
        help="Comma-separated target tenors in years.",
    )
    parser.add_argument(
        "--target-tranches",
        type=str,
        default="0-3,3-7,7-10,10-15",
        help="Comma-separated tranche bounds in percent points, e.g. 0-3,3-7.",
    )
    parser.add_argument("--recovery", type=float, default=0.40, help="Recovery rate assumption.")
    parser.add_argument("--n-quad", type=int, default=64, help="Gauss-Hermite quadrature nodes.")
    parser.add_argument("--grid-size", type=int, default=120, help="Calibration grid size.")
    parser.add_argument("--payment-freq", type=int, default=4, help="Premium payment frequency.")
    parser.add_argument("--outdir", type=str, default="outputs/corr_sensitivity", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _parse_target_tenors(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("target tenors cannot be empty")
    out = sorted(set(values))
    if any(t <= 0 for t in out):
        raise ValueError("target tenors must be positive")
    return out


def _parse_target_tranches(text: str) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"invalid tranche format: {token}")
        low_s, high_s = token.split("-", 1)
        low = float(low_s.strip()) / 100.0
        high = float(high_s.strip()) / 100.0
        if low < 0 or high <= low:
            raise ValueError(f"invalid tranche bounds: {token}")
        out.append((token, low, high))
    if not out:
        raise ValueError("target tranches cannot be empty")
    return out


def _surface_to_dataframe(surface: dict[float, dict[float, float]]) -> pd.DataFrame:
    rows: list[dict] = []
    for t, smile in sorted(surface.items()):
        for k, rho in sorted(smile.items()):
            rows.append({"node_tenor": float(t), "node_detachment": float(k), "rho": float(rho)})
    return pd.DataFrame(rows)


def _surface_from_dataframe(df: pd.DataFrame) -> dict[float, dict[float, float]]:
    out: dict[float, dict[float, float]] = {}
    for t, group in df.groupby("node_tenor"):
        out[float(t)] = {float(k): float(r) for k, r in zip(group["node_detachment"], group["rho"])}
    return out


def _price_target(
    target_tenor: float,
    tranche_k1: float,
    tranche_k2: float,
    rho_surface_fn,
    curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
) -> float:
    rho_hi = float(np.clip(rho_surface_fn(float(target_tenor), float(tranche_k2)), RHO_MIN, RHO_MAX))
    pv_hi = price_tranche_lhp(
        tenor=float(target_tenor),
        k1=0.0,
        k2=float(tranche_k2),
        rho=rho_hi,
        curve=curve,
        recovery=recovery,
        n_quad=n_quad,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )

    if tranche_k1 <= 0:
        return float(TranchePV(premium_leg=pv_hi.premium_leg, protection_leg=pv_hi.protection_leg).pv)

    rho_lo = float(np.clip(rho_surface_fn(float(target_tenor), float(tranche_k1)), RHO_MIN, RHO_MAX))
    pv_lo = price_tranche_lhp(
        tenor=float(target_tenor),
        k1=0.0,
        k2=float(tranche_k1),
        rho=rho_lo,
        curve=curve,
        recovery=recovery,
        n_quad=n_quad,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )

    return float(
        TranchePV(
            premium_leg=float(pv_hi.premium_leg - pv_lo.premium_leg),
            protection_leg=float(pv_hi.protection_leg - pv_lo.protection_leg),
        ).pv
    )


def _run_node_sensitivity(
    base_surface: dict[float, dict[float, float]],
    target_tenors: list[float],
    target_tranches: list[tuple[str, float, float]],
    curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
    epsilon: float,
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    rows: list[dict] = []

    base_surface_fn = build_rho_surface(base_surface)
    node_df = _surface_to_dataframe(base_surface)

    # Precompute base prices so failed shock runs still keep a stable reference.
    base_prices: dict[tuple[float, str], float] = {}
    for target_tenor in target_tenors:
        for tranche_label, k1, k2 in target_tranches:
            try:
                base_prices[(target_tenor, tranche_label)] = _price_target(
                    target_tenor,
                    k1,
                    k2,
                    base_surface_fn,
                    curve,
                    recovery,
                    n_quad,
                    payment_freq,
                    disc_curve,
                )
            except Exception as exc:
                msg = f"base-price failure tenor={target_tenor} tranche={tranche_label}: {exc}"
                logging.error(msg)
                errors.append(msg)

    for node in node_df.itertuples(index=False):
        node_tenor = float(node.node_tenor)
        node_det = float(node.node_detachment)
        rho0 = float(node.rho)

        if rho0 + epsilon > RHO_MAX or rho0 - epsilon < RHO_MIN:
            msg = (
                f"skip node bump at tenor={node_tenor}, det={node_det}: "
                f"rho={rho0:.6f} with epsilon={epsilon:.6f} breaches bounds"
            )
            logging.warning(msg)
            errors.append(msg)
            continue

        node_up = node_df.copy()
        node_dn = node_df.copy()
        mask = (node_df["node_tenor"] == node_tenor) & (node_df["node_detachment"] == node_det)
        node_up.loc[mask, "rho"] = rho0 + epsilon
        node_dn.loc[mask, "rho"] = rho0 - epsilon

        try:
            surface_up = _surface_from_dataframe(node_up)
            surface_dn = _surface_from_dataframe(node_dn)
            surface_up_fn = build_rho_surface(surface_up)
            surface_dn_fn = build_rho_surface(surface_dn)
        except Exception as exc:
            msg = f"surface rebuild failure for node ({node_tenor},{node_det}): {exc}"
            logging.error(msg)
            errors.append(msg)
            continue

        for target_tenor in target_tenors:
            for tranche_label, k1, k2 in target_tranches:
                base_key = (target_tenor, tranche_label)
                if base_key not in base_prices:
                    continue
                base_price = base_prices[base_key]

                try:
                    price_up = _price_target(
                        target_tenor,
                        k1,
                        k2,
                        surface_up_fn,
                        curve,
                        recovery,
                        n_quad,
                        payment_freq,
                        disc_curve,
                    )
                    price_down = _price_target(
                        target_tenor,
                        k1,
                        k2,
                        surface_dn_fn,
                        curve,
                        recovery,
                        n_quad,
                        payment_freq,
                        disc_curve,
                    )
                    corr_sens = (price_up - price_down) / (2.0 * epsilon)
                except Exception as exc:
                    msg = (
                        f"repricing failure target=({target_tenor},{tranche_label}) "
                        f"node=({node_tenor},{node_det}): {exc}"
                    )
                    logging.error(msg)
                    errors.append(msg)
                    continue

                rows.append(
                    {
                        "target_tenor": float(target_tenor),
                        "target_tranche": tranche_label,
                        "shocked_node_tenor": node_tenor,
                        "shocked_node_detachment": node_det,
                        "base_price": base_price,
                        "price_up": price_up,
                        "price_down": price_down,
                        "corr_sensitivity": corr_sens,
                    }
                )

    if not rows:
        raise ValueError("No node-level sensitivities were produced.")

    return pd.DataFrame(rows), errors


def _locality_ratio_for_bucket(
    bucket_df: pd.DataFrame,
    target_tenor: float,
    tranche_center: float,
    all_target_tenors: list[float],
) -> float:
    if bucket_df.empty:
        return float("nan")

    tenor_span = max(max(all_target_tenors) - min(all_target_tenors), 1.0)
    det_span = max(max(DETACHMENTS) - min(DETACHMENTS), 1e-6)

    work = bucket_df.copy()
    work["abs_sens"] = work["corr_sensitivity"].abs()
    work["distance"] = (
        (work["shocked_node_tenor"] - target_tenor).abs() / tenor_span
        + (work["shocked_node_detachment"] - tranche_center).abs() / det_span
    )

    total = float(work["abs_sens"].sum())
    if math.isclose(total, 0.0):
        return 0.0

    nearest = work.nsmallest(4, "distance")
    return float(nearest["abs_sens"].sum() / total)


def _build_summary_tables(
    sensitivity_df: pd.DataFrame,
    target_tenors: list[float],
    target_tranches: list[tuple[str, float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = sensitivity_df.copy()
    work["abs_corr_sensitivity"] = work["corr_sensitivity"].abs()

    locality_rows: list[dict] = []
    for tranche_label, k1, k2 in target_tranches:
        center = 0.5 * (k1 + k2)
        for tenor in target_tenors:
            bucket = work[(work["target_tranche"] == tranche_label) & (work["target_tenor"] == tenor)]
            locality_rows.append(
                {
                    "target_tranche": tranche_label,
                    "target_tenor": tenor,
                    "locality_ratio": _locality_ratio_for_bucket(bucket, tenor, center, target_tenors),
                }
            )

    locality_df = pd.DataFrame(locality_rows)

    tenor_concentration = (
        work.groupby(["target_tranche", "shocked_node_tenor"], as_index=False)["abs_corr_sensitivity"].sum()
    )
    det_concentration = (
        work.groupby(["target_tranche", "shocked_node_detachment"], as_index=False)["abs_corr_sensitivity"].sum()
    )

    tenor_hhi = (
        tenor_concentration.assign(
            share=lambda x: x["abs_corr_sensitivity"]
            / x.groupby("target_tranche")["abs_corr_sensitivity"].transform("sum").replace(0.0, np.nan)
        )
        .assign(hhi=lambda x: x["share"] ** 2)
        .groupby("target_tranche", as_index=False)["hhi"]
        .sum()
        .rename(columns={"hhi": "tenor_concentration"})
    )

    det_hhi = (
        det_concentration.assign(
            share=lambda x: x["abs_corr_sensitivity"]
            / x.groupby("target_tranche")["abs_corr_sensitivity"].transform("sum").replace(0.0, np.nan)
        )
        .assign(hhi=lambda x: x["share"] ** 2)
        .groupby("target_tranche", as_index=False)["hhi"]
        .sum()
        .rename(columns={"hhi": "detachment_concentration"})
    )

    tranche_summary = (
        work.groupby("target_tranche", as_index=False)
        .agg(total_abs_corr_sensitivity=("abs_corr_sensitivity", "sum"))
        .merge(
            locality_df.groupby("target_tranche", as_index=False)["locality_ratio"].mean(),
            on="target_tranche",
            how="left",
        )
        .merge(tenor_hhi, on="target_tranche", how="left")
        .merge(det_hhi, on="target_tranche", how="left")
        .sort_values("total_abs_corr_sensitivity", ascending=False)
    )

    top_nodes = (
        work.groupby(["target_tranche", "shocked_node_tenor", "shocked_node_detachment"], as_index=False)
        .agg(total_abs_corr_sensitivity=("abs_corr_sensitivity", "sum"))
        .sort_values(["target_tranche", "total_abs_corr_sensitivity"], ascending=[True, False])
    )
    top_nodes["rank"] = top_nodes.groupby("target_tranche")["total_abs_corr_sensitivity"].rank(
        method="first", ascending=False
    )
    top_nodes = top_nodes[top_nodes["rank"] <= 3].copy()

    mezz = tranche_summary[tranche_summary["target_tranche"].isin(["3-7", "7-10"])]
    mezz_total = float(mezz["total_abs_corr_sensitivity"].sum()) if not mezz.empty else float("nan")
    tranche_summary["mezz_total_corr_sensitivity"] = mezz_total

    return tranche_summary, top_nodes


def _plot_heatmaps_by_tranche(df: pd.DataFrame, plot_dir: Path) -> None:
    configure_plot_style()
    work = df.copy()

    for tranche, sub in work.groupby("target_tranche"):
        grid = (
            sub.assign(abs_corr_sensitivity=lambda x: x["corr_sensitivity"].abs())
            .groupby(["shocked_node_tenor", "shocked_node_detachment"], as_index=False)["abs_corr_sensitivity"]
            .sum()
            .pivot(index="shocked_node_tenor", columns="shocked_node_detachment", values="abs_corr_sensitivity")
            .sort_index()
        )
        if grid.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        arr = grid.to_numpy(dtype=float)
        vmax = max(float(np.nanpercentile(arr, 95)), 1e-10)
        im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_title(f"Abs Corr Sensitivity Heatmap | Tranche {tranche}")
        ax.set_xlabel("Shocked node detachment")
        ax.set_ylabel("Shocked node tenor")
        ax.set_xticks(np.arange(len(grid.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in grid.columns])
        ax.set_yticks(np.arange(len(grid.index)))
        ax.set_yticklabels([f"{x:.1f}Y" for x in grid.index])
        fig.colorbar(im, ax=ax, label="Sum |dP/drho_node|", shrink=0.85)
        fig.savefig(plot_dir / f"heatmap_tranche_{tranche.replace('-', '_')}.png", dpi=300)
        plt.close(fig)


def _plot_top_nodes(top_nodes: pd.DataFrame, plot_dir: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    bars: list[dict] = []
    for row in top_nodes.sort_values(["target_tranche", "rank"]).itertuples(index=False):
        bars.append(
            {
                "label": f"{row.target_tranche}|T={row.shocked_node_tenor:.1f},K={row.shocked_node_detachment:.2f}",
                "value": row.total_abs_corr_sensitivity,
                "tranche": row.target_tranche,
            }
        )
    plot_df = pd.DataFrame(bars)
    if plot_df.empty:
        return

    ax.bar(plot_df["label"], plot_df["value"], color="#2e86c1")
    ax.set_title("Top-3 Shocked Nodes by Tranche")
    ax.set_ylabel("Total |dP/drho_node|")
    ax.tick_params(axis="x", rotation=60)
    fig.savefig(plot_dir / "top3_nodes_by_tranche.png", dpi=300)
    plt.close(fig)


def _plot_concentrations(df: pd.DataFrame, plot_dir: Path) -> None:
    configure_plot_style()
    work = df.copy()
    work["abs_corr_sensitivity"] = work["corr_sensitivity"].abs()

    tenor_bucket = (
        work.groupby("shocked_node_tenor", as_index=False)["abs_corr_sensitivity"].sum().sort_values("shocked_node_tenor")
    )
    det_bucket = (
        work.groupby("shocked_node_detachment", as_index=False)["abs_corr_sensitivity"].sum().sort_values("shocked_node_detachment")
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar([f"{x:.1f}Y" for x in tenor_bucket["shocked_node_tenor"]], tenor_bucket["abs_corr_sensitivity"], color="#1f618d")
    ax.set_title("Risk Concentration by Tenor Bucket")
    ax.set_ylabel("Total |dP/drho_node|")
    fig.savefig(plot_dir / "risk_concentration_tenor_bucket.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar([f"{x:.2f}" for x in det_bucket["shocked_node_detachment"]], det_bucket["abs_corr_sensitivity"], color="#7d3c98")
    ax.set_title("Risk Concentration by Detachment Bucket")
    ax.set_ylabel("Total |dP/drho_node|")
    fig.savefig(plot_dir / "risk_concentration_detachment_bucket.png", dpi=300)
    plt.close(fig)


def _terminal_summary(tranche_summary: pd.DataFrame, top_nodes: pd.DataFrame) -> None:
    ordered = tranche_summary.sort_values("total_abs_corr_sensitivity", ascending=False)
    highest = ordered.iloc[0]

    overall_node = (
        top_nodes.groupby(["shocked_node_tenor", "shocked_node_detachment"], as_index=False)["total_abs_corr_sensitivity"]
        .sum()
        .sort_values("total_abs_corr_sensitivity", ascending=False)
        .iloc[0]
    )

    locality_sorted = tranche_summary.sort_values("locality_ratio", ascending=False)
    most_local = locality_sorted.iloc[0]
    least_local = locality_sorted.iloc[-1]

    print("\nNode-Level Correlation Sensitivity Summary")
    print(f"- Highest total corr sensitivity tranche: {highest['target_tranche']} ({highest['total_abs_corr_sensitivity']:.6g})")
    print(
        "- Most important node overall: "
        f"T={overall_node['shocked_node_tenor']:.2f}Y, K={overall_node['shocked_node_detachment']:.2f} "
        f"({overall_node['total_abs_corr_sensitivity']:.6g})"
    )
    print(f"- Most localized tranche risk: {most_local['target_tranche']} (locality_ratio={most_local['locality_ratio']:.4f})")
    print(f"- Least localized tranche risk: {least_local['target_tranche']} (locality_ratio={least_local['locality_ratio']:.4f})")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")

    target_tenors = _parse_target_tenors(args.target_tenors)
    target_tranches = _parse_target_tranches(args.target_tranches)

    data_dir = ROOT / "data"
    outdir = ensure_directory(ROOT / args.outdir)
    plot_dir = ensure_directory(outdir / "plots")

    index_df = load_index_timeseries(data_dir / "cdx_timeseries.csv")
    target_date = choose_target_date(index_df, args.date)
    snapshot = get_snapshot_for_date(index_df, target_date)
    disc_curve = load_discount_curve_for_date(data_dir, target_date)

    adjusted_curve = build_adjusted_index_curve(snapshot, disc_curve=disc_curve, recovery=args.recovery)
    calib = calibrate_surface_on_snapshot(
        snapshot=snapshot,
        curve=adjusted_curve,
        recovery=args.recovery,
        n_quad=args.n_quad,
        grid_size=args.grid_size,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
    )

    sensitivity_df, errors = _run_node_sensitivity(
        base_surface=calib.surface,
        target_tenors=target_tenors,
        target_tranches=target_tranches,
        curve=adjusted_curve,
        recovery=args.recovery,
        n_quad=args.n_quad,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
        epsilon=args.epsilon,
    )

    tranche_summary, top_nodes = _build_summary_tables(sensitivity_df, target_tenors, target_tranches)

    detail_path = outdir / "node_level_sensitivities.csv"
    summary_path = outdir / "tranche_summary.csv"
    top_nodes_path = outdir / "top_nodes_by_tranche.csv"
    error_path = outdir / "node_sensitivity_errors.log"

    sensitivity_df.to_csv(detail_path, index=False)
    tranche_summary.to_csv(summary_path, index=False)
    top_nodes.to_csv(top_nodes_path, index=False)

    if errors:
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        logging.warning("Completed with %d recoverable errors. See %s", len(errors), error_path)

    _plot_heatmaps_by_tranche(sensitivity_df, plot_dir)
    _plot_top_nodes(top_nodes, plot_dir)
    _plot_concentrations(sensitivity_df, plot_dir)

    logging.info("Saved node-level sensitivity table: %s", detail_path)
    logging.info("Saved tranche summary table: %s", summary_path)
    logging.info("Saved top-nodes table: %s", top_nodes_path)
    logging.info("Saved plots under: %s", plot_dir)

    _terminal_summary(tranche_summary, top_nodes)


if __name__ == "__main__":
    main()
