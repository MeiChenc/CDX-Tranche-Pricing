from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .basis import basis_adjust_curves_beta, build_average_curve
from .curves import Curve, bootstrap_from_cds_spreads, build_index_curve


def _parse_tenor(text: str) -> float:
    value = text.strip().upper()
    if value.endswith("Y"):
        return float(value[:-1])
    if value.endswith("M"):
        return float(value[:-1]) / 12.0
    if value.endswith("W"):
        return float(value[:-1]) / 52.0
    return float(value)


def _round_tenor(value: float, ndigits: int = 6) -> float:
    return float(round(float(value), ndigits))


def _map_recovery_value(value: float) -> float:
    if pd.isna(value):
        raise ValueError("Missing recovery value for constituent")
    v = float(value)
    if v <= 1.0:
        return v
    if int(v) == 100:
        return 0.4
    if int(v) == 500:
        return 0.25
    raise ValueError(f"Unsupported recovery code: {value}")


def _get_index_spreads_decimal(index_snapshot: pd.DataFrame) -> np.ndarray:
    """
    Prefer market 0-100 spread if present; fall back to legacy index columns.
    Input CSV is in bps and output is decimal spread.
    """
    candidate_cols = ["Index_0_100_Spread", "Index_Mid", "Index_Last"]
    for col in candidate_cols:
        if col not in index_snapshot.columns:
            continue
        vals = pd.to_numeric(index_snapshot[col], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isfinite(vals)):
            return vals / 10000.0
    raise ValueError(
        "No usable index spread column found. Expected one of: "
        "Index_0_100_Spread, Index_Mid, Index_Last."
    )


def build_index_dual_curve_beta_bundle(
    index_snapshot: pd.DataFrame,
    disc_curve,
    recovery_index: float = 0.4,
    theoretical_col: str = "Index_0_100_Spread",
    market_col: str = "Index_Mid",
    eps: float = 1e-12,
):
    """
    Build theoretical and market index curves from two spread columns and compute
    beta term structures on a shared tenor grid.

    Returns:
    - tenors: shared tenor grid (years)
    - theoretical_curve: index curve bootstrapped from `theoretical_col`
    - market_curve: index curve bootstrapped from `market_col`
    - beta_knot: lambda_mkt / lambda_theo at tenor knots
    - beta_cum: H_mkt(t) / H_theo(t), with H(t) = -ln(Q(t))
    - theoretical_spreads_decimal: theoretical spreads on grid (decimal)
    - market_spreads_decimal: market spreads on grid (decimal)
    """
    if "tenor" not in index_snapshot.columns:
        raise ValueError("index_snapshot must contain tenor column")
    if theoretical_col not in index_snapshot.columns:
        raise ValueError(f"Missing theoretical spread column: {theoretical_col}")
    if market_col not in index_snapshot.columns:
        raise ValueError(f"Missing market spread column: {market_col}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    work = index_snapshot.copy()
    work["tenor"] = pd.to_numeric(work["tenor"], errors="coerce")
    work[theoretical_col] = pd.to_numeric(work[theoretical_col], errors="coerce")
    work[market_col] = pd.to_numeric(work[market_col], errors="coerce")
    work = work.dropna(subset=["tenor", theoretical_col, market_col]).copy()
    if work.empty:
        raise ValueError(
            f"No overlapping valid rows for columns tenor/{theoretical_col}/{market_col}."
        )
    if (work["tenor"] <= 0).any():
        raise ValueError("Tenors must be strictly positive")

    work = work.sort_values("tenor")
    work = work.drop_duplicates(subset=["tenor"], keep="last")

    tenors = work["tenor"].to_numpy(dtype=float)
    if tenors.size < 1:
        raise ValueError("Need at least one tenor to build curves")

    theoretical_spreads_decimal = work[theoretical_col].to_numpy(dtype=float) / 10000.0
    market_spreads_decimal = work[market_col].to_numpy(dtype=float) / 10000.0

    theoretical_curve = build_index_curve(
        tenors,
        theoretical_spreads_decimal,
        recovery=recovery_index,
        disc_curve=disc_curve,
    )
    market_curve = build_index_curve(
        tenors,
        market_spreads_decimal,
        recovery=recovery_index,
        disc_curve=disc_curve,
    )

    theo_lambda = np.maximum(np.asarray(theoretical_curve.hazard, dtype=float), 0.0)
    mkt_lambda = np.maximum(np.asarray(market_curve.hazard, dtype=float), 0.0)
    beta_knot = mkt_lambda / np.maximum(theo_lambda, eps)

    theo_surv = np.array([theoretical_curve.survival(float(t)) for t in tenors], dtype=float)
    mkt_surv = np.array([market_curve.survival(float(t)) for t in tenors], dtype=float)
    theo_H = -np.log(np.maximum(theo_surv, eps))
    mkt_H = -np.log(np.maximum(mkt_surv, eps))
    beta_cum = mkt_H / np.maximum(theo_H, eps)

    return (
        tenors,
        theoretical_curve,
        market_curve,
        beta_knot,
        beta_cum,
        theoretical_spreads_decimal,
        market_spreads_decimal,
    )


def _hazard_on_grid(curve: Curve, target_times: np.ndarray) -> np.ndarray:
    times = np.asarray(curve.times, dtype=float)
    hazard = np.asarray(curve.hazard, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    idx = np.searchsorted(times, target_times, side="left")
    idx = np.clip(idx, 0, hazard.size - 1)
    return hazard[idx]


def apply_betas(
    constituent_curves: List[Curve],
    target_times: np.ndarray,
    betas: np.ndarray,
) -> List[Curve]:
    target_times = np.asarray(target_times, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if target_times.shape != betas.shape:
        raise ValueError("target_times and betas must have the same shape")

    n_names = len(constituent_curves)
    base_lambdas = np.vstack([_hazard_on_grid(curve, target_times) for curve in constituent_curves])
    target_survival = np.exp(-base_lambdas * betas[None, :] * target_times[None, :])

    adjusted_hazards = np.zeros_like(base_lambdas, dtype=float)
    prev_t = 0.0
    prev_q = np.ones(n_names, dtype=float)
    for idx, t in enumerate(target_times):
        t = float(t)
        dt = t - prev_t
        if dt <= 0:
            raise ValueError("target_times must be strictly increasing")
        q = np.minimum(target_survival[:, idx], prev_q)
        adjusted_hazards[:, idx] = np.maximum(-np.log(q / prev_q) / dt, 1e-12)
        prev_t = t
        prev_q = q

    return [Curve(times=target_times, hazard=adjusted_hazards[i]) for i in range(n_names)]


def expand_curves_to_grid(
    constituent_curves: List[Curve],
    target_times: np.ndarray,
) -> List[Curve]:
    target_times = np.asarray(target_times, dtype=float)
    expanded = []
    for curve in constituent_curves:
        hazard = _hazard_on_grid(curve, target_times)
        expanded.append(Curve(times=target_times, hazard=hazard))
    return expanded


def build_basis_adjusted_curve(
    target_date,
    index_snapshot: pd.DataFrame,
    constituents_df: pd.DataFrame,
    disc_curve,
    recovery_index: float = 0.4,
) -> Curve:
    all_tenors = index_snapshot["tenor"].to_numpy(dtype=float)
    index_spreads = _get_index_spreads_decimal(index_snapshot)
    index_curve_full = build_index_curve(all_tenors, index_spreads, recovery=recovery_index, disc_curve=disc_curve)

    cons_slice = constituents_df[constituents_df["Date"].dt.date == target_date].copy()
    if cons_slice.empty:
        return index_curve_full

    spread_cols: dict[float, str] = {}
    for col in cons_slice.columns:
        if col.lower().startswith("spread_"):
            tenor = _parse_tenor(col.split("_", 1)[1])
            spread_cols[_round_tenor(tenor)] = col

    tenor_list = [float(t) for t in all_tenors if _round_tenor(float(t)) in spread_cols]
    if not tenor_list:
        return index_curve_full

    cols = [spread_cols[_round_tenor(t)] for t in tenor_list]
    cons_spreads = cons_slice[cols].dropna(how="any")
    if cons_spreads.empty:
        return index_curve_full

    spreads_matrix = cons_spreads.to_numpy(dtype=float) / 10000.0
    cons_meta = cons_slice.loc[cons_spreads.index, ["Recovery"]]
    recoveries = cons_meta["Recovery"].apply(_map_recovery_value).to_numpy(dtype=float)
    constituent_curves = [
        bootstrap_from_cds_spreads(
            tenor_list,
            spreads,
            float(rec),
            payment_freq=4,
            disc_curve=disc_curve,
        )
        for spreads, rec in zip(spreads_matrix, recoveries)
    ]

    expanded_curves = expand_curves_to_grid(constituent_curves, all_tenors)

    adjusted_curves, _, _ = basis_adjust_curves_beta(
        expanded_curves, index_curve_full, recovery=recovery_index, recoveries=recoveries
    )
    return build_average_curve(adjusted_curves)


def build_basis_adjustment_bundle(
    target_date,
    index_snapshot: pd.DataFrame,
    constituents_df: pd.DataFrame,
    disc_curve,
    recovery_index: float = 0.4,
):
    """
    Build expanded constituent curves on the index grid, solve beta(t) on the index grid,
    and return the adjusted average curve along with inputs for diagnostics.
    """
    all_tenors = index_snapshot["tenor"].to_numpy(dtype=float)
    index_spreads = _get_index_spreads_decimal(index_snapshot)
    index_curve_full = build_index_curve(all_tenors, index_spreads, recovery=recovery_index, disc_curve=disc_curve)

    cons_slice = constituents_df[constituents_df["Date"].dt.date == target_date].copy()
    if cons_slice.empty:
        return index_curve_full, None, all_tenors, np.array([]), [], np.array([])

    spread_cols: dict[float, str] = {}
    for col in cons_slice.columns:
        if col.lower().startswith("spread_"):
            tenor = _parse_tenor(col.split("_", 1)[1])
            spread_cols[_round_tenor(tenor)] = col

    tenor_list = [float(t) for t in all_tenors if _round_tenor(float(t)) in spread_cols]
    if not tenor_list:
        return index_curve_full, None, all_tenors, np.array([]), [], np.array([])

    cols = [spread_cols[_round_tenor(t)] for t in tenor_list]
    cons_spreads = cons_slice[cols].dropna(how="any")
    if cons_spreads.empty:
        return index_curve_full, None, all_tenors, np.array([]), [], np.array([])

    spreads_matrix = cons_spreads.to_numpy(dtype=float) / 10000.0
    cons_meta = cons_slice.loc[cons_spreads.index, ["Recovery"]]
    recoveries = cons_meta["Recovery"].apply(_map_recovery_value).to_numpy(dtype=float)
    constituent_curves = [
        bootstrap_from_cds_spreads(
            tenor_list,
            spreads,
            float(rec),
            payment_freq=4,
            disc_curve=disc_curve,
        )
        for spreads, rec in zip(spreads_matrix, recoveries)
    ]

    expanded_curves = expand_curves_to_grid(constituent_curves, all_tenors)
    adjusted_curves, betas, adjusted_avg = basis_adjust_curves_beta(
        expanded_curves, index_curve_full, recovery=recovery_index, recoveries=recoveries
    )
    return index_curve_full, adjusted_avg, all_tenors, betas, expanded_curves, recoveries
