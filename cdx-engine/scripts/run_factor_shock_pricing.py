from __future__ import annotations

"""Factor shock pricing for PCA-based correlation surface factors.

For each target tranche, this script applies +1 sigma shocks to PC1/PC2/PC3
via node loadings and reprices tranche PV on the shocked base-correlation
surface.
"""

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

from src.interpolation import build_rho_surface
from src.pricer_tranche import price_tranche_lhp
from scripts.risk_engine_common import (
    build_adjusted_index_curve,
    calibrate_surface_on_snapshot,
    choose_target_date,
    configure_logging,
    ensure_directory,
    get_snapshot_for_date,
    load_discount_curve_for_date,
    load_index_timeseries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Price tranche sensitivity to +1 sigma PCA factor shocks.")
    parser.add_argument("--node-ts", type=str, default="outputs/analyze_basecorr_timeseries/data/basecorr_node_timeseries.csv")
    parser.add_argument("--pca-loadings", type=str, default="outputs/analyze_basecorr_timeseries/data/pca_loadings.csv")
    parser.add_argument("--date", type=str, default=None, help="Base pricing date YYYY-MM-DD (default: latest data date).")
    parser.add_argument("--target-tenors", type=str, default="1,2,3,5,7,10")
    parser.add_argument("--target-tranches", type=str, default="0-3,3-7,7-10,10-15")
    parser.add_argument("--recovery", type=float, default=0.40)
    parser.add_argument("--n-quad", type=int, default=64)
    parser.add_argument("--payment-freq", type=int, default=4)
    parser.add_argument("--out-csv", type=str, default="outputs/run_factor_shock_pricing/data/factor_shock_pricing.csv")
    parser.add_argument("--out-plot", type=str, default="outputs/run_factor_shock_pricing/plots/factor_shock_pricing.png")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _node_key(tenor: float, det: float, ndigits: int = 8) -> tuple[float, float]:
    return (round(float(tenor), ndigits), round(float(det), ndigits))


def _parse_target_tenors(text: str) -> list[float]:
    vals = sorted(set(float(x.strip()) for x in text.split(",") if x.strip()))
    if not vals:
        raise ValueError("target tenors cannot be empty")
    return vals


def _parse_target_tranches(text: str) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for item in text.split(","):
        tok = item.strip()
        if not tok:
            continue
        lo_s, hi_s = tok.split("-", 1)
        lo = float(lo_s) / 100.0
        hi = float(hi_s) / 100.0
        if hi <= lo:
            raise ValueError(f"invalid tranche bounds: {tok}")
        out.append((tok, lo, hi))
    if not out:
        raise ValueError("target tranches cannot be empty")
    return out


def load_node_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing node timeseries file: {path}")
    df = pd.read_csv(path)
    required = {"date", "tenor", "detachment", "rho"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(required)}")

    df = df[["date", "tenor", "detachment", "rho"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tenor"] = pd.to_numeric(df["tenor"], errors="coerce")
    df["detachment"] = pd.to_numeric(df["detachment"], errors="coerce")
    df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
    df = df.dropna(subset=["date", "tenor", "detachment", "rho"]).copy()

    wide = df.pivot_table(index="date", columns=["tenor", "detachment"], values="rho", aggfunc="mean")
    wide = wide.sort_index().sort_index(axis=1)
    if wide.empty:
        raise ValueError("No valid node matrix from basecorr_node_timeseries.csv")
    return df, wide


def load_pca_loadings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing PCA loadings file: {path}")
    df = pd.read_csv(path)
    required = {"tenor", "detachment", "pc1", "pc2", "pc3"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(required)}")
    out = df[["tenor", "detachment", "pc1", "pc2", "pc3"]].copy()
    out["node_key"] = [_node_key(t, k) for t, k in zip(out["tenor"], out["detachment"])]
    return out


def compute_factor_sigmas(node_wide: pd.DataFrame, loadings_df: pd.DataFrame) -> dict[str, float]:
    node_keys = [_node_key(t, k) for t, k in node_wide.columns]
    x = node_wide.to_numpy(dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    x_std = (x - mu) / sd

    lmap = loadings_df.set_index("node_key")[["pc1", "pc2", "pc3"]]
    # Reindex by tuple node keys to avoid tuple .loc ambiguity (row/col dispatch).
    vec_df = lmap.reindex(node_keys).fillna(0.0)
    vec = vec_df.to_numpy(dtype=float)

    scores = x_std @ vec
    sig = np.nanstd(scores, axis=0, ddof=1)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)
    return {"pc1": float(sig[0]), "pc2": float(sig[1]), "pc3": float(sig[2])}


def _surface_df(surface: dict[float, dict[float, float]]) -> pd.DataFrame:
    rows: list[dict] = []
    for tenor, smile in sorted(surface.items()):
        for det, rho in sorted(smile.items()):
            rows.append({"tenor": float(tenor), "detachment": float(det), "rho": float(rho), "node_key": _node_key(tenor, det)})
    return pd.DataFrame(rows)


def _surface_from_df(df: pd.DataFrame) -> dict[float, dict[float, float]]:
    out: dict[float, dict[float, float]] = {}
    for tenor, grp in df.groupby("tenor"):
        out[float(tenor)] = {float(k): float(v) for k, v in zip(grp["detachment"], grp["rho"])}
    return out


def _price_tranche_from_surface(
    tenor: float,
    k1: float,
    k2: float,
    surface_fn,
    curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
) -> float:
    rho_hi = float(np.clip(surface_fn(float(tenor), float(k2)), 1e-4, 0.999))
    pv_hi = price_tranche_lhp(float(tenor), 0.0, float(k2), rho_hi, curve, recovery, n_quad=n_quad, payment_freq=payment_freq, disc_curve=disc_curve)
    if k1 <= 0.0:
        return float(pv_hi.pv)
    rho_lo = float(np.clip(surface_fn(float(tenor), float(k1)), 1e-4, 0.999))
    pv_lo = price_tranche_lhp(float(tenor), 0.0, float(k1), rho_lo, curve, recovery, n_quad=n_quad, payment_freq=payment_freq, disc_curve=disc_curve)
    return float(pv_hi.pv - pv_lo.pv)


def _apply_factor_shock(base_df: pd.DataFrame, loadings_df: pd.DataFrame, pc: str, sigma: float) -> pd.DataFrame:
    l = loadings_df.set_index("node_key")[pc]
    out = base_df.copy()
    out[pc] = out["node_key"].map(l).fillna(0.0)
    out["rho"] = np.clip(out["rho"] + out[pc] * float(sigma), 1e-4, 0.999)
    return out[["tenor", "detachment", "rho", "node_key"]]


def plot_result(df: pd.DataFrame, out_plot: Path) -> None:
    plot_df = df.copy()
    plot_df["label"] = plot_df["tranche"]
    x = np.arange(len(plot_df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w, plot_df["delta_price_pc1"], width=w, label="PC1", color="#1f77b4")
    ax.bar(x, plot_df["delta_price_pc2"], width=w, label="PC2", color="#2ca02c")
    ax.bar(x + w, plot_df["delta_price_pc3"], width=w, label="PC3", color="#d62728")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax.set_ylabel("Price Change")
    ax.set_title("Tranche Price Sensitivity to +1 Sigma PCA Factor Shocks")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    _, node_wide = load_node_matrix(ROOT / args.node_ts)
    loadings = load_pca_loadings(ROOT / args.pca_loadings)
    sigmas = compute_factor_sigmas(node_wide, loadings)

    data_dir = ROOT / "data"
    index_df = load_index_timeseries(data_dir / "cdx_timeseries.csv")
    target_date = choose_target_date(index_df, args.date)
    snapshot = get_snapshot_for_date(index_df, target_date)
    disc_curve = load_discount_curve_for_date(data_dir, target_date)
    curve = build_adjusted_index_curve(snapshot, disc_curve=disc_curve, recovery=args.recovery)
    calib = calibrate_surface_on_snapshot(
        snapshot=snapshot,
        curve=curve,
        recovery=args.recovery,
        n_quad=args.n_quad,
        grid_size=120,
        payment_freq=args.payment_freq,
        disc_curve=disc_curve,
    )

    base_df = _surface_df(calib.surface)
    shock_surfaces: dict[str, dict[float, dict[float, float]]] = {}
    for pc in ("pc1", "pc2", "pc3"):
        shocked_df = _apply_factor_shock(base_df, loadings, pc=pc, sigma=sigmas[pc])
        shock_surfaces[pc] = _surface_from_df(shocked_df)

    base_fn = build_rho_surface(calib.surface)
    shock_fn = {pc: build_rho_surface(surf) for pc, surf in shock_surfaces.items()}

    target_tenors = _parse_target_tenors(args.target_tenors)
    target_tranches = _parse_target_tranches(args.target_tranches)

    rows: list[dict] = []
    for tenor in target_tenors:
        for name, k1, k2 in target_tranches:
            tranche = f"{tenor:.0f}Y|{name}"
            p_base = _price_tranche_from_surface(tenor, k1, k2, base_fn, curve, args.recovery, args.n_quad, args.payment_freq, disc_curve)
            d1 = _price_tranche_from_surface(tenor, k1, k2, shock_fn["pc1"], curve, args.recovery, args.n_quad, args.payment_freq, disc_curve) - p_base
            d2 = _price_tranche_from_surface(tenor, k1, k2, shock_fn["pc2"], curve, args.recovery, args.n_quad, args.payment_freq, disc_curve) - p_base
            d3 = _price_tranche_from_surface(tenor, k1, k2, shock_fn["pc3"], curve, args.recovery, args.n_quad, args.payment_freq, disc_curve) - p_base
            rows.append({"date": str(target_date.date()), "tranche": tranche, "delta_price_pc1": float(d1), "delta_price_pc2": float(d2), "delta_price_pc3": float(d3)})

    result = pd.DataFrame(rows).sort_values("tranche")
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    plot_result(result, ROOT / args.out_plot)
    logging.info("Saved factor shock pricing to %s", out_csv)


if __name__ == "__main__":
    main()
