from __future__ import annotations

"""Compute explained variance ratio of base-correlation surface PCA.

Input:
- outputs/basecorr_node_timeseries.csv with columns: date, tenor, detachment, rho

Output:
- outputs/pca_explained_variance.csv
- outputs/pca_scree_plot.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute explained variance ratio from correlation surface PCA.")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/analyze_basecorr_timeseries/data/basecorr_node_timeseries.csv",
        help="Input node time series CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/run_pca_explained_variance/data/pca_explained_variance.csv",
        help="Output explained variance CSV.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="outputs/run_pca_explained_variance/plots/pca_scree_plot.png",
        help="Output scree plot PNG.",
    )
    parser.add_argument("--n-components", type=int, default=10, help="Max PCA components.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def _load_and_pivot(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"date", "tenor", "detachment", "rho"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_path}: {sorted(required)}")

    df = df[["date", "tenor", "detachment", "rho"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tenor"] = pd.to_numeric(df["tenor"], errors="coerce")
    df["detachment"] = pd.to_numeric(df["detachment"], errors="coerce")
    df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
    df = df.dropna(subset=["date", "tenor", "detachment", "rho"]).copy()

    if df.empty:
        raise ValueError("No valid rows in node time series after cleaning.")

    wide = df.pivot_table(
        index="date",
        columns=["tenor", "detachment"],
        values="rho",
        aggfunc="mean",
    )
    wide = wide.sort_index().sort_index(axis=1)

    if wide.shape[0] < 3:
        raise ValueError("Need at least 3 dates to run PCA.")
    if wide.shape[1] < 1:
        raise ValueError("No surface nodes available to run PCA.")

    return wide


def _standardize_nodes(wide: pd.DataFrame) -> np.ndarray:
    x = wide.to_numpy(dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)

    # Avoid divide-by-zero for flat nodes; they carry no variance after centering.
    std_safe = np.where(std <= 1e-12, 1.0, std)
    x_std = (x - mean) / std_safe

    if not np.all(np.isfinite(x_std)):
        raise ValueError("Standardized matrix contains NaN/Inf values.")

    return x_std


def _run_pca(x_std: np.ndarray, n_components: int) -> np.ndarray:
    n_samples, n_features = x_std.shape
    n_comp = min(max(1, int(n_components)), n_samples, n_features)

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_comp)
        pca.fit(x_std)
        return np.asarray(pca.explained_variance_ratio_, dtype=float)
    except ImportError:
        logging.warning("scikit-learn not installed; falling back to numpy SVD for PCA.")
        # x_std is already standardized (centered + scaled per node).
        _, s, _ = np.linalg.svd(x_std, full_matrices=False)
        var = (s**2) / max(1, n_samples - 1)
        if var.size == 0 or float(np.sum(var)) <= 0:
            return np.zeros(n_comp, dtype=float)
        evr = var / np.sum(var)
        return np.asarray(evr[:n_comp], dtype=float)


def _save_outputs(evr: np.ndarray, out_csv: Path, out_plot: Path) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    pcs = np.arange(1, evr.size + 1, dtype=int)
    cum = np.cumsum(evr)

    evr_df = pd.DataFrame(
        {
            "PC": pcs,
            "explained_variance_ratio": evr,
            "cumulative_variance": cum,
        }
    )
    evr_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(pcs, evr, marker="o", linewidth=1.5)
    ax.set_xticks(pcs)
    ax.set_xlabel("PC")
    ax.set_ylabel("explained_variance_ratio")
    ax.set_title("PCA Scree Plot (Correlation Surface Nodes)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)

    return evr_df


def _print_summary(evr: np.ndarray) -> None:
    def get(i: int) -> float:
        return float(evr[i]) if i < evr.size else 0.0

    pc1 = get(0)
    pc2 = get(1)
    pc3 = get(2)
    cum3 = float(np.sum(evr[:3]))

    print("PCA Explained Variance Summary")
    print(f"PC1 variance: {pc1:.6f}")
    print(f"PC2 variance: {pc2:.6f}")
    print(f"PC3 variance: {pc3:.6f}")
    print(f"Cumulative variance (first 3 PCs): {cum3:.6f}")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    input_path = ROOT / args.input
    out_csv = ROOT / args.out_csv
    out_plot = ROOT / args.out_plot

    wide = _load_and_pivot(input_path)
    x_std = _standardize_nodes(wide)
    evr = _run_pca(x_std, args.n_components)
    _save_outputs(evr, out_csv, out_plot)

    logging.info("Saved explained variance CSV: %s", out_csv)
    logging.info("Saved scree plot: %s", out_plot)
    _print_summary(evr)


if __name__ == "__main__":
    main()
