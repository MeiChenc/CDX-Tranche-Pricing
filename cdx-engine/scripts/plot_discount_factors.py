from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from src.curves import Curve, build_index_curve
from src.io_data import _coerce_tenor_column, _normalize_columns, _standardize_index_columns, read_ois_discount_curve


def _load_recovery_rate(config_path: Path) -> float:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return float(cfg["engine"]["recovery_rate"])


def _latest_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["Date"].max().normalize()


def _ois_curve(ois_csv: Path, target_date: pd.Timestamp) -> Curve:
    ois_df = pd.read_csv(ois_csv, parse_dates=["Date"])
    ois_df = _normalize_columns(ois_df)
    ois_df = _coerce_tenor_column(ois_df)
    ois_slice = ois_df[ois_df["Date"].dt.normalize() == target_date].copy()
    if ois_slice.empty:
        raise ValueError(f"No OIS rows for {target_date.date()}")
    tenors = ois_slice["tenor"].to_numpy(dtype=float)
    rates = ois_slice["OIS_Rate"].to_numpy(dtype=float) / 100.0
    order = np.argsort(tenors)
    return Curve(times=tenors[order], hazard=rates[order])


def _index_curve(cdx_csv: Path, target_date: pd.Timestamp, recovery: float, disc_curve) -> Curve:
    index_df = pd.read_csv(cdx_csv, parse_dates=["Date"])
    index_df = _normalize_columns(index_df)
    index_df = _coerce_tenor_column(index_df)
    index_df = _standardize_index_columns(index_df)
    index_slice = index_df[index_df["Date"].dt.normalize() == target_date].copy()
    if index_slice.empty:
        raise ValueError(f"No CDX rows for {target_date.date()}")
    tenors = index_slice["tenor"].to_numpy(dtype=float)
    spreads_bp = index_slice["index_spread"].to_numpy(dtype=float)
    order = np.argsort(tenors)
    tenors = tenors[order]
    spreads = spreads_bp[order] / 10000.0
    return build_index_curve(tenors, spreads, recovery, payment_freq=4, disc_curve=disc_curve)


def _constituent_curve(cons_csv: Path, target_date: pd.Timestamp, recovery: float, disc_curve) -> Curve:
    cons_df = pd.read_csv(cons_csv, parse_dates=["Date"])
    cons_df = _normalize_columns(cons_df)
    cons_slice = cons_df[cons_df["Date"].dt.normalize() == target_date].copy()
    if cons_slice.empty:
        raise ValueError(f"No constituents rows for {target_date.date()}")

    tenor_cols = {5.0: "Spread_5Y", 7.0: "Spread_7Y", 10.0: "Spread_10Y"}
    tenors = []
    spreads = []
    for tenor, col in tenor_cols.items():
        if col not in cons_slice.columns:
            raise ValueError(f"Missing {col} in constituents data")
        mean_bp = cons_slice[col].astype(float).mean()
        tenors.append(tenor)
        spreads.append(mean_bp / 10000.0)

    return build_index_curve(tenors, spreads, recovery, payment_freq=4, disc_curve=disc_curve)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    config_path = root / "configs" / "params.yaml"

    recovery = _load_recovery_rate(config_path)

    ois_csv = data_dir / "ois_timeseries.csv"
    cdx_csv = data_dir / "cdx_timeseries.csv"
    cons_csv = data_dir / "constituents_timeseries.csv"

    ois_df = pd.read_csv(ois_csv, parse_dates=["Date"])
    target_date = _latest_date(ois_df)

    # OIS curve for discounting and for the plotted DF approximation.
    ois_curve = _ois_curve(ois_csv, target_date)
    disc_curve = read_ois_discount_curve(str(ois_csv), target_date)

    index_curve = _index_curve(cdx_csv, target_date, recovery, disc_curve)
    cons_curve = _constituent_curve(cons_csv, target_date, recovery, disc_curve)

    t_max = max(float(np.max(ois_curve.times)), float(np.max(index_curve.times)), float(np.max(cons_curve.times)))
    ts = np.linspace(0.0, max(10.0, t_max), 200)

    ois_df = np.array([ois_curve.survival(t) for t in ts])
    index_df = np.array([index_curve.survival(t) for t in ts])
    cons_df = np.array([cons_curve.survival(t) for t in ts])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, ois_df, label="OIS discount (as hazard)")
    ax.plot(ts, index_df, label="CDX index-implied DF")
    ax.plot(ts, cons_df, label="Constituent avg-implied DF")
    ax.set_title(f"Discount Factor via survival(t) using data on {target_date.date()}")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("discount_factor(t)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_dir = root / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "discount_factors.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
