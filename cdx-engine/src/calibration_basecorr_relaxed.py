from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .pricer_tranche import price_tranche_lhp


def calibrate_basecorr_relaxed(
    tenor: float,
    dets: list[float],
    tranche_spreads: dict[float, float],
    tranche_upfronts: dict[float, float],
    curve,
    recovery: float,
    n_quad: int,
    grid_size: int,
    payment_freq: int,
    disc_curve=None,
    return_status: bool = False,
):
    """
    Coarse grid base correlation calibration by minimizing absolute PV error.
    Uses base correlation convention:
        PV(K1,K2) = PV(0,K2; rho(K2)) - PV(0,K1; rho(K1))
    """
    calibrated: Dict[float, float] = {}
    statuses: Dict[float, str] = {}
    dets = sorted(dets)
    prev_k = 0.0
    prev_base_pv = None

    def solve_rho_for_tranche(
        objective_fn,
        n_scan: int = 80,
        lo: float = 1e-4,
        hi: float = 0.999,
    ) -> Tuple[float, str]:
        grid = np.linspace(lo, hi, n_scan)
        vals = np.array([objective_fn(float(r)) for r in grid], dtype=float)

        for i in range(len(grid) - 1):
            a, b = float(grid[i]), float(grid[i + 1])
            fa, fb = float(vals[i]), float(vals[i + 1])
            if not np.isfinite(fa) or not np.isfinite(fb):
                continue
            if fa == 0.0:
                return a, "ROOT_ON_GRID"
            if fa * fb < 0.0:
                root = brentq(objective_fn, a, b, xtol=1e-10, rtol=1e-10, maxiter=200)
                return float(root), "BRENT"

        def objective_abs(r: float) -> float:
            return float(np.abs(objective_fn(float(r))))

        res = minimize_scalar(
            objective_abs,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 500},
        )
        return float(res.x), "NO_BRACKET_MINABS"
    for det in dets:
        k1, k2 = prev_k, det
        spread = tranche_spreads.get(det, 0.0)
        upfront = tranche_upfronts.get(det, 0.0)
        tranche_width = k2 - k1

        if k1 == 0.0:
            base_pv_k1 = None
        else:
            if prev_base_pv is None:
                raise RuntimeError("Missing base PV for previous detachment.")
            base_pv_k1 = prev_base_pv

        def objective(rho: float) -> float:
            base_pv_k2 = price_tranche_lhp(
                tenor,
                0.0,
                k2,
                rho,
                curve,
                recovery,
                n_quad=n_quad,
                payment_freq=payment_freq,
                disc_curve=disc_curve,
            )
            if base_pv_k1 is None:
                model_prot = base_pv_k2.protection_leg
                model_prem = base_pv_k2.premium_leg
            else:
                model_prot = base_pv_k2.protection_leg - base_pv_k1.protection_leg
                model_prem = base_pv_k2.premium_leg - base_pv_k1.premium_leg
            return model_prot - spread * model_prem #- upfront * tranche_width

        rho_best, status = solve_rho_for_tranche(
            objective,
            n_scan=max(20, int(grid_size)),
            lo=1e-4,
            hi=0.999,
        )
        calibrated[k2] = rho_best
        statuses[k2] = status
        prev_k = k2
        prev_base_pv = price_tranche_lhp(
            tenor,
            0.0,
            k2,
            rho_best,
            curve,
            recovery,
            n_quad=n_quad,
            payment_freq=payment_freq,
            disc_curve=disc_curve,
        )
    if return_status:
        return calibrated, statuses
    return calibrated
