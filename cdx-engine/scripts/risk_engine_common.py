from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle
from src.calibration_basecorr_relaxed import calibrate_basecorr_relaxed
from src.curves import Curve
from src.io_data import read_ois_discount_curve
from src.pricer_tranche import TranchePV, price_tranche_lhp

DETACHMENTS: tuple[float, ...] = (0.03, 0.07, 0.10, 0.15)
TRANCHES: tuple[tuple[str, float, float], ...] = (
    ("Equity_0_3", 0.00, 0.03),
    ("Mezz_3_7", 0.03, 0.07),
    ("Mezz_7_10", 0.07, 0.10),
    ("Senior_10_15", 0.10, 0.15),
)

REQUIRED_QUOTE_COLS: tuple[str, ...] = (
    "Index_Mid",
    "Index_0_100_Spread",
    "Equity_0_3_Spread",
    "Mezz_3_7_Spread",
    "Mezz_7_10_Spread",
    "Senior_10_15_Spread",
)


@dataclass
class SurfaceCalibrationResult:
    date: pd.Timestamp
    surface: Dict[float, Dict[float, float]]
    node_table: pd.DataFrame


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s: %(message)s")


def configure_plot_style() -> None:
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


def parse_tenor_to_years(value: str) -> float:
    text = str(value).strip().upper()
    if text.endswith("Y"):
        return float(text[:-1])
    if text.endswith("M"):
        return float(text[:-1]) / 12.0
    if text.endswith("W"):
        return float(text[:-1]) / 52.0
    if text.endswith("D"):
        return float(text[:-1]) / 365.0
    return float(text)


def ensure_directory(path: Path | str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def clip_rho(rho: float, lo: float = 1e-4, hi: float = 0.999) -> float:
    return float(np.clip(float(rho), float(lo), float(hi)))


def load_index_timeseries(cdx_csv: Path | str) -> pd.DataFrame:
    df = pd.read_csv(cdx_csv, parse_dates=["Date"])
    df["Tenor"] = df["Tenor"].astype(str).str.upper()
    df["tenor"] = df["Tenor"].map(parse_tenor_to_years)
    for col in REQUIRED_QUOTE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=list(REQUIRED_QUOTE_COLS)).copy()


def list_available_dates(df: pd.DataFrame) -> List[pd.Timestamp]:
    dates = sorted(pd.to_datetime(df["Date"]).dt.normalize().unique().tolist())
    return [pd.Timestamp(d) for d in dates]


def choose_target_date(df: pd.DataFrame, date_str: str | None) -> pd.Timestamp:
    available = list_available_dates(df)
    if not available:
        raise ValueError("No valid dates available after filtering.")
    if date_str is None:
        return available[-1]
    target = pd.to_datetime(date_str).normalize()
    if target not in set(available):
        nearest = max([d for d in available if d <= target], default=None)
        if nearest is None:
            raise ValueError(f"Requested date {target.date()} not available.")
        logging.warning("Requested date %s unavailable; using latest prior date %s.", target.date(), nearest.date())
        return nearest
    return target


def get_snapshot_for_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    snap = df[df["Date"].dt.normalize() == pd.Timestamp(date).normalize()].copy()
    if snap.empty:
        raise ValueError(f"No market snapshot for date {date.date()}.")
    return snap.sort_values("tenor").drop_duplicates(subset=["tenor"], keep="last")


def load_discount_curve_for_date(data_dir: Path, target_date: pd.Timestamp | pd.Timestamp.date) -> Curve:
    return read_ois_discount_curve(str(Path(data_dir) / "ois_timeseries.csv"), pd.to_datetime(target_date).date())


def _curve_from_cum_hazard(tenors: Iterable[float], cum_hazard: Iterable[float]) -> Curve:
    tenor_arr = np.asarray(list(tenors), dtype=float)
    cum_arr = np.asarray(list(cum_hazard), dtype=float)
    if tenor_arr.size == 0:
        raise ValueError("Tenor grid is empty.")
    if tenor_arr.size != cum_arr.size:
        raise ValueError("tenors and cumulative hazard arrays must have equal length.")
    if np.any(np.diff(tenor_arr) <= 0):
        raise ValueError("Tenors must be strictly increasing.")

    hazard = np.zeros_like(cum_arr, dtype=float)
    prev_t = 0.0
    prev_h = 0.0
    for i, (t, h) in enumerate(zip(tenor_arr, cum_arr)):
        dt = float(t) - prev_t
        if dt <= 0:
            raise ValueError("Invalid tenor spacing while reconstructing hazard curve.")
        dh = max(float(h) - prev_h, 0.0)
        hazard[i] = max(dh / dt, 1e-12)
        prev_t = float(t)
        prev_h = float(h)
    return Curve(times=tenor_arr, hazard=hazard)


def build_adjusted_index_curve(snapshot: pd.DataFrame, disc_curve, recovery: float = 0.4) -> Curve:
    (
        tenors_basis,
        theoretical_curve,
        _market_curve,
        _beta_knot,
        beta_cum,
        _theoretical_spreads,
        _market_spreads,
    ) = build_index_dual_curve_beta_bundle(
        snapshot,
        disc_curve=disc_curve,
        recovery_index=recovery,
        theoretical_col="Index_0_100_Spread",
        market_col="Index_Mid",
    )
    theo_h = -np.log(np.maximum(np.array([theoretical_curve.survival(float(t)) for t in tenors_basis]), 1e-12))
    adjusted_h = np.maximum(beta_cum, 1e-8) * theo_h
    return _curve_from_cum_hazard(tenors_basis, adjusted_h)


def quote_row_to_tranche_inputs(row: pd.Series) -> tuple[dict[float, float], dict[float, float], list[float]]:
    def _opt_pct(value: object) -> float:
        if pd.isna(value):
            return 0.0
        return float(value) / 100.0

    tranche_spreads = {
        0.03: float(row["Equity_0_3_Spread"]) / 10000.0,
        0.07: float(row["Mezz_3_7_Spread"]) / 10000.0,
        0.10: float(row["Mezz_7_10_Spread"]) / 10000.0,
        0.15: float(row["Senior_10_15_Spread"]) / 10000.0,
    }
    tranche_upfronts = {
        0.03: _opt_pct(row.get("Equity_0_3_Upfront", 0.0)),
        0.07: _opt_pct(row.get("Mezz_3_7_Upfront", 0.0)),
    }
    return tranche_spreads, tranche_upfronts, list(DETACHMENTS)


def calibrate_surface_on_snapshot(
    snapshot: pd.DataFrame,
    curve: Curve,
    recovery: float,
    n_quad: int,
    grid_size: int,
    payment_freq: int,
    disc_curve,
) -> SurfaceCalibrationResult:
    by_tenor: Dict[float, Dict[float, float]] = {}
    rows: list[dict] = []
    date = pd.Timestamp(snapshot["Date"].iloc[0]).normalize()

    for tenor, row in snapshot.set_index("tenor").iterrows():
        spreads, upfronts, dets = quote_row_to_tranche_inputs(row)
        basecorr, statuses = calibrate_basecorr_relaxed(
            tenor=float(tenor),
            dets=dets,
            tranche_spreads=spreads,
            tranche_upfronts=upfronts,
            curve=curve,
            recovery=recovery,
            n_quad=n_quad,
            grid_size=grid_size,
            payment_freq=payment_freq,
            disc_curve=disc_curve,
            return_status=True,
        )
        by_tenor[float(tenor)] = {float(k): clip_rho(v) for k, v in basecorr.items()}
        for det in dets:
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "tenor": float(tenor),
                    "detachment": float(det),
                    "rho": float(by_tenor[float(tenor)][float(det)]),
                    "status": statuses.get(det, ""),
                }
            )

    node_table = pd.DataFrame(rows).sort_values(["tenor", "detachment"]).reset_index(drop=True)
    return SurfaceCalibrationResult(date=date, surface=by_tenor, node_table=node_table)


def tranche_pv_from_surface(
    tenor: float,
    k1: float,
    k2: float,
    surface_by_det: Dict[float, float],
    curve: Curve,
    recovery: float,
    n_quad: int,
    payment_freq: int,
    disc_curve,
) -> TranchePV:
    rho_hi = clip_rho(surface_by_det[float(k2)])
    pv_hi = price_tranche_lhp(
        tenor=float(tenor),
        k1=0.0,
        k2=float(k2),
        rho=rho_hi,
        curve=curve,
        recovery=recovery,
        n_quad=n_quad,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )
    if float(k1) <= 0.0:
        return TranchePV(premium_leg=float(pv_hi.premium_leg), protection_leg=float(pv_hi.protection_leg))

    rho_lo = clip_rho(surface_by_det[float(k1)])
    pv_lo = price_tranche_lhp(
        tenor=float(tenor),
        k1=0.0,
        k2=float(k1),
        rho=rho_lo,
        curve=curve,
        recovery=recovery,
        n_quad=n_quad,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )
    return TranchePV(
        premium_leg=float(pv_hi.premium_leg - pv_lo.premium_leg),
        protection_leg=float(pv_hi.protection_leg - pv_lo.protection_leg),
    )
