from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from scipy.optimize import brentq

from .curves import Curve
from .pricer_tranche import price_tranche_lhp


def _solve_bracketed_root(
    objective: Callable[[float], float],
    rho_bounds: tuple[float, float],
    grid_size: int,
    det: float,
    target_label: str,
) -> float:
    a, b = rho_bounds
    fa, fb = objective(a), objective(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError(f"Objective returned NaN at rho bounds {rho_bounds} for det={det}.")

    if fa * fb > 0:
        rhos = np.linspace(a, b, grid_size)
        vals = np.array([objective(r) for r in rhos])
        finite = np.isfinite(vals)
        if finite.any():
            vals = vals[finite]
            rhos = rhos[finite]
            sign_changes = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
            if sign_changes.size:
                i = int(sign_changes[0])
                a, b = float(rhos[i]), float(rhos[i + 1])
            else:
                min_val = float(vals.min())
                max_val = float(vals.max())
                raise ValueError(
                    f"{target_label} not bracketed within rho bounds {rho_bounds} for det={det}. "
                    f"Objective range ≈ [{min_val:.6g}, {max_val:.6g}]."
                )
        else:
            raise ValueError(
                f"Objective not finite across rho bounds {rho_bounds} for det={det}."
            )

    return brentq(objective, a, b)


def calibrate_basecorr_curve(
    tenor: float,
    detachments: List[float],
    market_pvs: Dict[float, float],
    curve: Curve,
    recovery: float,
    rho_bounds: tuple[float, float] = (1e-4, 0.999),
    grid_size: int = 201,
    n_quad: int = 64,
) -> Dict[float, float]:
    dets = sorted(detachments)
    calibrated: Dict[float, float] = {}

    prev_k = 0.0
    for det in dets:
        k1, k2 = prev_k, det
        target_pv = market_pvs[det]

        def objective(rho: float) -> float:
            pv = price_tranche_lhp(tenor, k1, k2, rho, curve, recovery, n_quad=n_quad).pv
            return pv - target_pv

        rho = _solve_bracketed_root(
            objective,
            rho_bounds=rho_bounds,
            grid_size=grid_size,
            det=det,
            target_label="Target PV",
        )
        calibrated[det] = rho
        prev_k = det

    return calibrated


def calibrate_basecorr_from_quotes(
    tenor: float,
    detachments: List[float],
    tranche_spreads: Dict[float, float],
    tranche_upfronts: Dict[float, float],
    curve: Curve,
    recovery: float,
    rho_bounds: tuple[float, float] = (1e-4, 0.999),
    grid_size: int = 201,
    n_quad: int = 64,
    payment_freq: int = 4,
) -> Dict[float, float]:
    dets = sorted(detachments)
    calibrated: Dict[float, float] = {}

    prev_k = 0.0
    for det in dets:
        k1, k2 = prev_k, det
        spread = tranche_spreads.get(det, 0.0)
        upfront = tranche_upfronts.get(det, 0.0)
        tranche_width = k2 - k1

        def objective(rho: float) -> float:
            pv = price_tranche_lhp(
                tenor, k1, k2, rho, curve, recovery, n_quad=n_quad, payment_freq=payment_freq
            )
            return pv.protection_leg - spread * pv.premium_leg + upfront * tranche_width

        rho = _solve_bracketed_root(
            objective,
            rho_bounds=rho_bounds,
            grid_size=grid_size,
            det=det,
            target_label="Quote PV",
        )
        calibrated[det] = rho
        prev_k = det

    return calibrated
