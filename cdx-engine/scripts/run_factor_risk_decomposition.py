from __future__ import annotations

"""Project node-level correlation sensitivities into PCA factor space.

This script computes tranche-level factor exposures by projecting bucketed
correlation risk dP/drho(T,K) onto PCA loadings L[(T,K), i]:

    dP/dPC_i = sum_{T,K} dP/drho(T,K) * L[(T,K), i]

It is intended for correlation-factor risk reporting and scenario design.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor risk decomposition for CDX tranche correlation risk.")
    parser.add_argument("--sens-file", type=str, default="outputs/run_node_correlation_sensitivity/data/node_level_sensitivities.csv")
    parser.add_argument("--pca-dir", type=str, default="outputs/analyze_basecorr_timeseries/data")
    parser.add_argument("--outdir", type=str, default="outputs/run_factor_risk_decomposition")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _node_key(tenor: float, det: float, ndigits: int = 8) -> tuple[float, float]:
    return (round(float(tenor), ndigits), round(float(det), ndigits))


def _parse_pct_detachment(col: str) -> float:
    text = str(col).strip().replace("%", "")
    return float(text) / 100.0


def _parse_tenor_label(text: str) -> float:
    token = str(text).strip().upper()
    if token.endswith("Y"):
        return float(token[:-1])
    return float(token)


def _load_heatmap_loading_csv(path: Path, pc_col: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    tenor_col = None
    for candidate in ("Tenor", "tenor"):
        if candidate in raw.columns:
            tenor_col = candidate
            break
    if tenor_col is None:
        raise ValueError(f"{path} missing Tenor column")

    det_cols = [c for c in raw.columns if c != tenor_col]
    long_rows: list[dict] = []
    for _, row in raw.iterrows():
        tenor = _parse_tenor_label(row[tenor_col])
        for col in det_cols:
            val = row[col]
            if pd.isna(val):
                continue
            long_rows.append(
                {
                    "tenor": float(tenor),
                    "detachment": _parse_pct_detachment(col),
                    pc_col: float(val),
                }
            )
    return pd.DataFrame(long_rows)


def load_pca_loadings(pca_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load PCA loadings for PC1-3 and explained variance ratios.

    Preferred inputs:
    - <pca_dir>/pca_loadings.csv with columns tenor, detachment, pc1, pc2, pc3

    Fallback inputs:
    - <pca_dir>/basecorr_node_timeseries.csv: recompute PCA loadings on daily rho moves.

    Explained variance ratios are loaded from <pca_dir>/pca_explained_variance.csv when
    available, otherwise derived from recomputed PCA.
    """
    pca_dir = Path(pca_dir)
    loadings_csv = pca_dir / "pca_loadings.csv"
    pc1_csv = pca_dir / "pc1_loadings_heatmap.csv"
    pc2_csv = pca_dir / "pc2_loadings_heatmap.csv"
    pc3_csv = pca_dir / "pc3_loadings_heatmap.csv"
    node_ts_csv = pca_dir / "basecorr_node_timeseries.csv"
    ev_csv = pca_dir / "pca_explained_variance.csv"

    ev_ratio: dict[str, float] = {}
    if ev_csv.exists():
        ev_df = pd.read_csv(ev_csv)
        ev_df.columns = [str(c).strip().lower() for c in ev_df.columns]
        if {"pc", "explained_variance_ratio"}.issubset(ev_df.columns):
            for _, row in ev_df.iterrows():
                pc = int(row["pc"])
                if 1 <= pc <= 3:
                    ev_ratio[f"pc{pc}"] = float(row["explained_variance_ratio"])

    if loadings_csv.exists():
        load_df = pd.read_csv(loadings_csv)
        required = {"tenor", "detachment", "pc1", "pc2", "pc3"}
        if not required.issubset(load_df.columns):
            raise ValueError(f"{loadings_csv} missing required columns {sorted(required)}")
        out = load_df[["tenor", "detachment", "pc1", "pc2", "pc3"]].copy()
        out["node_key"] = [_node_key(t, k) for t, k in zip(out["tenor"], out["detachment"])]
        if len(ev_ratio) < 3:
            logging.warning("Explained variance ratios incomplete; missing values will be set to 0.")
        return out, {f"pc{i}": float(ev_ratio.get(f"pc{i}", 0.0)) for i in (1, 2, 3)}

    if pc1_csv.exists() and pc2_csv.exists() and pc3_csv.exists():
        l1 = _load_heatmap_loading_csv(pc1_csv, "pc1")
        l2 = _load_heatmap_loading_csv(pc2_csv, "pc2")
        l3 = _load_heatmap_loading_csv(pc3_csv, "pc3")
        out = (
            l1.merge(l2, on=["tenor", "detachment"], how="inner")
            .merge(l3, on=["tenor", "detachment"], how="inner")
            .drop_duplicates(subset=["tenor", "detachment"])
        )
        if out.empty:
            raise ValueError("PC loading heatmap CSV files could not be aligned on common nodes.")
        out["node_key"] = [_node_key(t, k) for t, k in zip(out["tenor"], out["detachment"])]
        return out, {f"pc{i}": float(ev_ratio.get(f"pc{i}", 0.0)) for i in (1, 2, 3)}

    # Refuse partial heatmap loading inputs; PC1/PC2/PC3 must come from the same PCA object.
    if pc1_csv.exists() and (not pc2_csv.exists() or not pc3_csv.exists()):
        missing = []
        if not pc2_csv.exists():
            missing.append(pc2_csv.name)
        if not pc3_csv.exists():
            missing.append(pc3_csv.name)
        raise FileNotFoundError(
            f"Missing PCA loading files: {missing}. "
            "Refusing to run PC1-only decomposition because it would invalidate factor risk results."
        )

    if not node_ts_csv.exists():
        raise FileNotFoundError(
            "PCA loadings not found. Provide outputs/pca_loadings.csv or generate outputs/basecorr_node_timeseries.csv "
            "(e.g., run scripts/analyze_basecorr_timeseries.py --run-pca)."
        )

    node_df = pd.read_csv(node_ts_csv)
    req_cols = {"date", "tenor", "detachment", "rho"}
    if not req_cols.issubset(node_df.columns):
        raise ValueError(f"{node_ts_csv} missing required columns {sorted(req_cols)}")

    node_df["date"] = pd.to_datetime(node_df["date"])
    node_df = node_df.sort_values(["date", "tenor", "detachment"]).copy()

    # Create day x node matrix and apply PCA on daily moves (matching existing analytics logic).
    pivot = node_df.pivot_table(index="date", columns=["tenor", "detachment"], values="rho", aggfunc="mean")
    if pivot.shape[0] < 3:
        raise ValueError("Need at least 3 dates in basecorr_node_timeseries.csv for PCA-based factor decomposition.")

    pivot = pivot.sort_index().sort_index(axis=1)
    M = np.diff(pivot.to_numpy(dtype=float), axis=0)
    M = M - np.mean(M, axis=0, keepdims=True)
    if not np.all(np.isfinite(M)):
        raise ValueError("Non-finite entries found in PCA input matrix.")

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    n_comp = min(3, Vt.shape[0])
    if n_comp < 3:
        logging.warning("Only %d components available from history; missing PCs will be set to 0.", n_comp)

    loadings = np.zeros((Vt.shape[1], 3), dtype=float)
    loadings[:, :n_comp] = Vt[:n_comp, :].T

    var = (S**2) / max(1, M.shape[0] - 1)
    ratio = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)
    for i in range(3):
        ev_ratio[f"pc{i+1}"] = float(ratio[i]) if i < ratio.size else 0.0

    nodes = list(pivot.columns)
    load_df = pd.DataFrame(
        {
            "tenor": [float(t) for t, _ in nodes],
            "detachment": [float(k) for _, k in nodes],
            "pc1": loadings[:, 0],
            "pc2": loadings[:, 1],
            "pc3": loadings[:, 2],
        }
    )
    load_df["node_key"] = [_node_key(t, k) for t, k in zip(load_df["tenor"], load_df["detachment"])]
    return load_df, ev_ratio


def load_node_sensitivities(path: Path) -> pd.DataFrame:
    """Load node-level correlation sensitivities dP/drho(T,K)."""
    df = pd.read_csv(path)
    required = {
        "target_tenor",
        "target_tranche",
        "shocked_node_tenor",
        "shocked_node_detachment",
        "corr_sensitivity",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(required)}")

    out = df.copy()
    out["tranche_tenor"] = out["target_tenor"].astype(float)
    out["tranche_name"] = out["target_tranche"].astype(str)
    out["node_tenor"] = out["shocked_node_tenor"].astype(float)
    out["node_detachment"] = out["shocked_node_detachment"].astype(float)
    out["corr_sensitivity"] = out["corr_sensitivity"].astype(float)
    out["node_key"] = [_node_key(t, k) for t, k in zip(out["node_tenor"], out["node_detachment"])]
    return out


def _augment_loadings_for_missing_nodes(loadings_df: pd.DataFrame, missing_keys: list[tuple[float, float]]) -> pd.DataFrame:
    """Fill missing PCA nodes by tenor interpolation/extrapolation per detachment bucket."""
    if not missing_keys:
        return loadings_df

    work = loadings_df.copy()
    add_rows: list[dict] = []
    grouped = {
        float(det): grp.sort_values("tenor").copy()
        for det, grp in work.groupby("detachment")
    }

    for tenor, det in missing_keys:
        det = float(det)
        tenor = float(tenor)
        grp = grouped.get(det)
        if grp is None or grp.empty:
            raise ValueError(f"Cannot interpolate missing node {(tenor, det)}: detachment bucket absent in PCA loadings.")

        x = grp["tenor"].to_numpy(dtype=float)
        row = {"tenor": tenor, "detachment": det}
        for pc in ("pc1", "pc2", "pc3"):
            y = grp[pc].to_numpy(dtype=float)
            row[pc] = float(np.interp(tenor, x, y, left=y[0], right=y[-1]))
        row["node_key"] = _node_key(tenor, det)
        add_rows.append(row)

    if add_rows:
        logging.warning(
            "Interpolated/extrapolated %d missing PCA node loadings to match sensitivity node grid.",
            len(add_rows),
        )
        work = pd.concat([work, pd.DataFrame(add_rows)], ignore_index=True)
        work = work.drop_duplicates(subset=["node_key"], keep="last")
    return work


def align_nodes(loadings_df: pd.DataFrame, sens_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align PCA loading nodes with sensitivity nodes using (tenor, detachment) keys."""
    load_keys = set(loadings_df["node_key"])
    sens_keys = set(sens_df["node_key"])

    missing_in_pca = sorted(sens_keys - load_keys)
    missing_in_sens = sorted(load_keys - sens_keys)

    # Common practical case: sensitivity grid extends tenor while detachment buckets match.
    if missing_in_pca and not missing_in_sens:
        loadings_df = _augment_loadings_for_missing_nodes(loadings_df, missing_in_pca)
        load_keys = set(loadings_df["node_key"])
        missing_in_pca = sorted(sens_keys - load_keys)

    if missing_in_pca or missing_in_sens:
        message = (
            f"Node mismatch between sensitivity and PCA loadings. "
            f"missing_in_pca={len(missing_in_pca)}, missing_in_sens={len(missing_in_sens)}"
        )
        details = []
        if missing_in_pca:
            details.append(f"first_missing_in_pca={missing_in_pca[:5]}")
        if missing_in_sens:
            details.append(f"first_missing_in_sens={missing_in_sens[:5]}")
        raise ValueError(message + (" | " + " | ".join(details) if details else ""))

    # Keep deterministic order by node key.
    load_aligned = loadings_df.sort_values(["tenor", "detachment"]).copy()
    sens_aligned = sens_df.copy()
    return load_aligned, sens_aligned


def validate_pca_consistency(loadings_df: pd.DataFrame, ev_ratio: dict[str, float]) -> None:
    """Validate consistency between explained variance and loading vectors."""
    for pc in ("pc1", "pc2", "pc3"):
        ev = float(ev_ratio.get(pc, 0.0))
        std = float(loadings_df[pc].std())
        maxabs = float(loadings_df[pc].abs().max())

        if ev > 1e-3 and maxabs < 1e-10:
            raise ValueError(
                f"{pc.upper()} explained variance ratio is {ev:.6f}, "
                f"but its loadings are numerically zero. "
                "This indicates inconsistent PCA inputs or a loading file generation bug."
            )
        if ev > 1e-3 and std < 1e-12:
            raise ValueError(
                f"{pc.upper()} explained variance ratio is {ev:.6f}, "
                f"but its loading standard deviation is near zero ({std:.3e}). "
                "This indicates inconsistent PCA inputs or a loading file generation bug."
            )


def compute_factor_exposures(loadings_df: pd.DataFrame, sens_df: pd.DataFrame) -> pd.DataFrame:
    """Compute tranche exposures dP/dPC_i via node-level projection into PCA space."""
    merged = sens_df.merge(
        loadings_df[["node_key", "pc1", "pc2", "pc3"]],
        on="node_key",
        how="left",
        validate="many_to_one",
    )
    if merged[["pc1", "pc2", "pc3"]].isna().any().any():
        raise ValueError("NaN loadings encountered after node alignment; cannot compute factor exposures.")

    for pc in ("pc1", "pc2", "pc3"):
        merged[f"weighted_{pc}"] = merged["corr_sensitivity"] * merged[pc]

    exposure_df = (
        merged.groupby(["tranche_tenor", "tranche_name"], as_index=False)
        .agg(
            pc1_exposure=("weighted_pc1", "sum"),
            pc2_exposure=("weighted_pc2", "sum"),
            pc3_exposure=("weighted_pc3", "sum"),
        )
        .sort_values(["tranche_tenor", "tranche_name"])
    )
    return exposure_df


def compute_normalized_exposures(exposure_df: pd.DataFrame, explained_var: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute absolute-normalized risk contributions and variance-weighted contributions."""
    out = exposure_df.copy()
    abs_sum = (
        out[["pc1_exposure", "pc2_exposure", "pc3_exposure"]]
        .abs()
        .sum(axis=1)
        .replace(0.0, np.nan)
    )

    out["pc1_normalized"] = out["pc1_exposure"].abs() / abs_sum
    out["pc2_normalized"] = out["pc2_exposure"].abs() / abs_sum
    out["pc3_normalized"] = out["pc3_exposure"].abs() / abs_sum
    out[["pc1_normalized", "pc2_normalized", "pc3_normalized"]] = out[
        ["pc1_normalized", "pc2_normalized", "pc3_normalized"]
    ].fillna(0.0)

    out["total_abs_factor_exposure"] = out[["pc1_exposure", "pc2_exposure", "pc3_exposure"]].abs().sum(axis=1)

    abs_mat = out[["pc1_exposure", "pc2_exposure", "pc3_exposure"]].abs().to_numpy(dtype=float)
    dominant_idx = np.argmax(abs_mat, axis=1)
    out["dominant_factor"] = [f"PC{i+1}" for i in dominant_idx]

    contrib_rows: list[dict] = []
    for row in out.itertuples(index=False):
        mapping = {
            "PC1": (row.pc1_exposure, row.pc1_normalized, float(explained_var.get("pc1", 0.0))),
            "PC2": (row.pc2_exposure, row.pc2_normalized, float(explained_var.get("pc2", 0.0))),
            "PC3": (row.pc3_exposure, row.pc3_normalized, float(explained_var.get("pc3", 0.0))),
        }
        for factor, (raw_exp, norm_exp, evr) in mapping.items():
            contrib_rows.append(
                {
                    "tranche_tenor": row.tranche_tenor,
                    "tranche_name": row.tranche_name,
                    "factor": factor,
                    "raw_exposure": raw_exp,
                    "normalized_exposure": norm_exp,
                    "explained_variance_ratio": evr,
                    "variance_weighted_contribution": abs(raw_exp) * evr,
                }
            )

    contrib_df = pd.DataFrame(contrib_rows)
    return out, contrib_df


def _tranche_order_key(tranche_name: str) -> tuple[float, float, str]:
    token = str(tranche_name)
    if "-" not in token:
        return (999.0, 999.0, token)
    lo, hi = token.split("-", 1)
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except Exception:
        return (999.0, 999.0, token)
    return (lo_f, hi_f, token)


def generate_factor_risk_plots(exposure_df: pd.DataFrame, contrib_df: pd.DataFrame, plot_dir: Path) -> None:
    """Generate publication-quality visualizations for factor risk reporting."""
    _configure_plot_style()

    plot_df = exposure_df.copy()
    plot_df["tranche_label"] = plot_df.apply(
        lambda r: f"{float(r['tranche_tenor']):.1f}Y|{r['tranche_name']}", axis=1
    )
    plot_df = plot_df.sort_values(["tranche_tenor", "tranche_name"])

    # 1) Grouped bar chart of PC exposures by tranche.
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(plot_df))
    w = 0.26
    ax.bar(x - w, plot_df["pc1_exposure"], width=w, label="PC1", color="#1f77b4")
    ax.bar(x, plot_df["pc2_exposure"], width=w, label="PC2", color="#ff7f0e")
    ax.bar(x + w, plot_df["pc3_exposure"], width=w, label="PC3", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["tranche_label"], rotation=45, ha="right")
    ax.set_title("PC1/PC2/PC3 Exposure by Tranche")
    ax.set_ylabel("Factor Exposure (dP/dPC)")
    ax.legend()
    fig.savefig(plot_dir / "grouped_pc_exposure_by_tranche.png", dpi=300)
    plt.close(fig)

    # 2) Stacked bar chart of absolute-normalized factor contributions.
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    base = np.zeros(len(plot_df))
    for col, color, label in [
        ("pc1_normalized", "#1f77b4", "PC1"),
        ("pc2_normalized", "#ff7f0e", "PC2"),
        ("pc3_normalized", "#2ca02c", "PC3"),
    ]:
        vals = plot_df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=base, color=color, label=label)
        base += vals
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["tranche_label"], rotation=45, ha="right")
    ax.set_title("Normalized Risk Contribution by Factor (Absolute)")
    ax.set_ylabel("Absolute-Normalized Contribution")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.savefig(plot_dir / "stacked_normalized_factor_contributions.png", dpi=300)
    plt.close(fig)

    # 3) Heatmap tranche x factor exposure.
    heat = plot_df.set_index("tranche_label")[["pc1_exposure", "pc2_exposure", "pc3_exposure"]]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(heat))), constrained_layout=True)
    arr = heat.to_numpy(dtype=float)
    vmax = max(float(np.nanpercentile(np.abs(arr), 95)), 1e-10)
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(list(heat.index))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["PC1", "PC2", "PC3"])
    ax.set_title("Tranche x Factor Exposure Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Exposure")
    fig.savefig(plot_dir / "heatmap_tranche_factor_exposure.png", dpi=300)
    plt.close(fig)

    # 4) Scatter PC1 vs PC2 exposure.
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(plot_df["pc1_exposure"], plot_df["pc2_exposure"], color="#34495e", s=70)
    for row in plot_df.itertuples(index=False):
        ax.annotate(row.tranche_label, (row.pc1_exposure, row.pc2_exposure), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("PC1 Exposure")
    ax.set_ylabel("PC2 Exposure")
    ax.set_title("PC1 vs PC2 Tranche Exposure")
    fig.savefig(plot_dir / "scatter_pc1_vs_pc2_exposure.png", dpi=300)
    plt.close(fig)

    # 5) Optional: exposure along capital structure (averaged across tenor buckets).
    cap = plot_df.copy()
    cap["order_key"] = cap["tranche_name"].map(_tranche_order_key)
    cap = cap.sort_values("order_key")
    cap_avg = cap.groupby("tranche_name", as_index=False)[["pc1_exposure", "pc2_exposure", "pc3_exposure"]].mean()
    cap_avg["order_key"] = cap_avg["tranche_name"].map(_tranche_order_key)
    cap_avg = cap_avg.sort_values("order_key")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(cap_avg["tranche_name"], cap_avg["pc1_exposure"], marker="o", label="PC1")
    ax.plot(cap_avg["tranche_name"], cap_avg["pc2_exposure"], marker="o", label="PC2")
    ax.plot(cap_avg["tranche_name"], cap_avg["pc3_exposure"], marker="o", label="PC3")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Factor Exposure Across Capital Structure")
    ax.set_ylabel("Average Exposure by Tranche")
    ax.legend()
    fig.savefig(plot_dir / "factor_exposure_capital_structure.png", dpi=300)
    plt.close(fig)


def _build_factor_summary(exposure_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in exposure_df.itertuples(index=False):
        abs_vals = np.array([abs(row.pc1_exposure), abs(row.pc2_exposure), abs(row.pc3_exposure)], dtype=float)
        total = float(np.sum(abs_vals))
        if total <= 0:
            concentration = 0.0
            diversification = 0.0
        else:
            shares = abs_vals / total
            concentration = float(np.max(shares))
            diversification = float(1.0 - concentration)

        rows.append(
            {
                "tranche_tenor": row.tranche_tenor,
                "tranche_name": row.tranche_name,
                "pc1_abs": abs(row.pc1_exposure),
                "pc2_abs": abs(row.pc2_exposure),
                "pc3_abs": abs(row.pc3_exposure),
                "factor_concentration": concentration,
                "factor_diversification": diversification,
                "dominant_factor": row.dominant_factor,
            }
        )
    return pd.DataFrame(rows).sort_values(["tranche_tenor", "tranche_name"])


def _terminal_summary(exposure_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    p1 = exposure_df.iloc[np.argmax(np.abs(exposure_df["pc1_exposure"].to_numpy(dtype=float)))]
    p2 = exposure_df.iloc[np.argmax(np.abs(exposure_df["pc2_exposure"].to_numpy(dtype=float)))]
    p3 = exposure_df.iloc[np.argmax(np.abs(exposure_df["pc3_exposure"].to_numpy(dtype=float)))]

    most_conc = summary_df.iloc[np.argmax(summary_df["factor_concentration"].to_numpy(dtype=float))]
    most_div = summary_df.iloc[np.argmax(summary_df["factor_diversification"].to_numpy(dtype=float))]

    def label(row: pd.Series) -> str:
        return f"{float(row['tranche_tenor']):.1f}Y|{row['tranche_name']}"

    print("\nFactor Risk Decomposition Summary")
    print(f"- Tranche most exposed to PC1: {label(p1)} ({p1['pc1_exposure']:.6g})")
    print(f"- Tranche most exposed to PC2: {label(p2)} ({p2['pc2_exposure']:.6g})")
    print(f"- Tranche most exposed to PC3: {label(p3)} ({p3['pc3_exposure']:.6g})")
    print(
        f"- Tranche with most concentrated factor risk: {label(most_conc)} "
        f"(concentration={most_conc['factor_concentration']:.4f})"
    )
    print(
        f"- Tranche with most diversified factor risk: {label(most_div)} "
        f"(diversification={most_div['factor_diversification']:.4f})"
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)

    sens_file = ROOT / args.sens_file
    pca_dir = ROOT / args.pca_dir
    outdir = ROOT / args.outdir
    data_dir = outdir / "data"
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    loadings_df, ev_ratio = load_pca_loadings(pca_dir)
    sens_df = load_node_sensitivities(sens_file)
    loadings_df, sens_df = align_nodes(loadings_df, sens_df)
    validate_pca_consistency(loadings_df, ev_ratio)

    exposures = compute_factor_exposures(loadings_df, sens_df)
    exposures, contrib = compute_normalized_exposures(exposures, ev_ratio)
    summary = _build_factor_summary(exposures)

    exposures_path = data_dir / "factor_exposures_by_tranche.csv"
    contrib_path = data_dir / "factor_contributions_by_tranche.csv"
    summary_path = data_dir / "factor_summary.csv"

    exposures.to_csv(exposures_path, index=False)
    contrib.to_csv(contrib_path, index=False)
    summary.to_csv(summary_path, index=False)

    generate_factor_risk_plots(exposures, contrib, plot_dir)

    logging.info("Saved factor exposures: %s", exposures_path)
    logging.info("Saved factor contributions: %s", contrib_path)
    logging.info("Saved factor summary: %s", summary_path)
    logging.info("Saved factor risk plots under %s", plot_dir)

    _terminal_summary(exposures, summary)


if __name__ == "__main__":
    main()
