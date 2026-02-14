from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from .curves import Curve


def _validate_curve_grid(constituent_curves: List[Curve], index_curve: Curve) -> np.ndarray:
    if not constituent_curves:
        raise ValueError("constituent_curves must be non-empty")
    times = np.asarray(index_curve.times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("index_curve.times must be a non-empty 1D array")
    for curve in constituent_curves:
        if not np.allclose(curve.times, times):
            raise ValueError("All curves must share the same tenor grid")
        if curve.hazard.shape != times.shape:
            raise ValueError("Curve hazard must align with times grid")
    return times


def _average_survival(curves: List[Curve], times: np.ndarray) -> np.ndarray:
    surv = np.zeros_like(times, dtype=float)
    for curve in curves:
        surv += np.array([curve.survival(float(t)) for t in times], dtype=float)
    surv /= float(len(curves))
    return np.clip(surv, 1e-12, 1.0)


def build_average_curve(constituent_curves: List[Curve]) -> Curve:
    """
    Build a single Curve whose survival matches the average survival of constituents.
    """
    times = _validate_curve_grid(constituent_curves, constituent_curves[0])
    avg_survival = _average_survival(constituent_curves, times)
    hazards = np.zeros_like(times, dtype=float)
    prev_t = 0.0
    prev_q = 1.0
    for i, t in enumerate(times):
        dt = float(t - prev_t)
        if dt <= 0:
            raise ValueError("times must be strictly increasing")
        q = float(avg_survival[i])
        q = min(max(q, 1e-12), prev_q)
        hazards[i] = max(-np.log(q / prev_q) / dt, 1e-12)
        prev_t = float(t)
        prev_q = q
    print(f"average hazard: {hazards}")
    return Curve(times=times, hazard=hazards)


def basis_adjust_curves_beta(
    constituent_curves: List[Curve],
    index_curve: Curve,
    recovery: float,
    index_recovery: Optional[float] = None,
    recoveries: Optional[np.ndarray] = None,
    beta_bounds: Tuple[float, float] = (0.0, 50.0),
) -> Tuple[List[Curve], np.ndarray, Curve]:
    """
    Apply a beta scaling to constituent hazards so that average expected loss
    matches the index expected loss at each tenor:

        (1/M) * sum_m (1 - R_m) * (1 - exp(-beta * lambda_m * t)) = EL_index(t)

    Assumptions:
    - lambda_m is the piecewise-constant hazard at the tenor knot.
    - P'_m(t) uses a single beta per tenor; beta is applied to that tenor's hazard.
    - EL_index(t) is computed from the index curve: (1 - R_index) * (1 - Q_index(t)).

    Returns:
    - adjusted_curves: list of adjusted constituent curves
    - betas: beta(t) array
    - adjusted_index_curve: average survival curve of adjusted constituents
    """
    if recovery < 0 or recovery >= 1:
        raise ValueError("recovery must be in [0, 1)")
    if index_recovery is None:
        index_recovery = recovery
    if index_recovery < 0 or index_recovery >= 1:
        raise ValueError("index_recovery must be in [0, 1)")

    times = _validate_curve_grid(constituent_curves, index_curve)
    n_names = len(constituent_curves)
    lambdas = np.vstack([curve.hazard for curve in constituent_curves])
    if recoveries is None:
        recoveries_arr = np.full(n_names, recovery, dtype=float)
    else:
        recoveries_arr = np.asarray(recoveries, dtype=float)
        if recoveries_arr.shape != (n_names,):
            raise ValueError("recoveries must have shape (n_names,)")
    betas = np.zeros_like(times, dtype=float)
    target_survival = np.ones((n_names, times.size), dtype=float)

    lo, hi = beta_bounds
    if lo < 0 or hi <= 0 or hi <= lo:
        raise ValueError("beta_bounds must be positive and increasing")

    class _Objective:
        def __init__(self) -> None:
            self.lam: np.ndarray | None = None
            self.t: float = 0.0
            self.target_el: float = 0.0
            self.recoveries: np.ndarray | None = None

        def update(self, lam: np.ndarray, t: float, target_el: float, recoveries: np.ndarray) -> None:
            self.lam = lam
            self.t = t
            self.target_el = target_el
            self.recoveries = recoveries

        def __call__(self, beta: float) -> float:
            lam = self.lam
            if lam is None or self.recoveries is None:
                return 0.0
            surv = np.exp(-beta * lam * self.t)
            avg_el = np.mean((1.0 - self.recoveries) * (1.0 - surv))
            return avg_el - self.target_el

    objective = _Objective()

    for idx, t in enumerate(times):
        t = float(t)
        if t <= 0:
            betas[idx] = 0.0
            continue

        target_el = (1.0 - index_recovery) * index_curve.default_prob(t) 
        if target_el <= 0:
            betas[idx] = 0.0
            continue

        lam = np.maximum(lambdas[:, idx], 0.0)
        max_el = np.mean((1.0 - recoveries_arr) * (lam > 0).astype(float))
        if target_el > max_el + 1e-10:
            raise ValueError(
                f"Target EL {target_el:.6g} exceeds max achievable {max_el:.6g} at t={t:.4g}."
            )

        objective.update(lam, t, target_el, recoveries_arr)

        f_lo = objective(lo)
        f_hi = objective(hi)
        hi_curr = hi
        while f_hi < 0 and hi_curr < 1e4:
            hi_curr *= 2.0
            f_hi = objective(hi_curr)
        if f_lo > 0:
            beta = lo
        elif f_hi < 0:
            beta = hi_curr
        else:
            beta = float(brentq(objective, lo, hi_curr, maxiter=200, xtol=1e-12))

        betas[idx] = beta
        target_survival[:, idx] = np.exp(-beta * lam * t)

    adjusted_hazards = np.zeros_like(lambdas, dtype=float)
    prev_t = 0.0
    prev_q = np.ones(n_names, dtype=float)
    for idx, t in enumerate(times):
        t = float(t)
        dt = t - prev_t
        if dt <= 0:
            raise ValueError("times must be strictly increasing")
        q = np.minimum(target_survival[:, idx], prev_q)
        adjusted_hazards[:, idx] = np.maximum(-np.log(q / prev_q) / dt, 1e-12)
        prev_t = t
        prev_q = q

    adjusted_curves = [Curve(times=times, hazard=adjusted_hazards[i]) for i in range(n_names)]
    adjusted_index_curve = build_average_curve(adjusted_curves)
    return adjusted_curves, betas, adjusted_index_curve
