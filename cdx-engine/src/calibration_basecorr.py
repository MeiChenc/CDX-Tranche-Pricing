from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import brentq

from .curves import Curve
from .pricer_tranche import price_tranche_lhp


def calibrate_basecorr_curve(
    tenor: float,
    detachments: List[float],
    market_pvs: Dict[float, float],
    curve: Curve,
    recovery: float,
    rho_bounds: tuple[float, float] = (1e-4, 0.999),
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

        rho = brentq(objective, rho_bounds[0], rho_bounds[1])
        calibrated[det] = rho
        prev_k = det

    return calibrated
