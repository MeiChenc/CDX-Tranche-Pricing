from __future__ import annotations

from typing import Dict, List

import numpy as np

from .calibration_basecorr import calibrate_basecorr_curve
from .interpolation import build_rho_surface
from .pricer_tranche import price_tranche_lhp
from .curves import Curve


def run_loocv(
    tenor: float,
    detachments: List[float],
    market_pvs: Dict[float, float],
    curve: Curve,
    recovery: float,
    n_quad: int = 64,
) -> Dict[float, float]:
    errors: Dict[float, float] = {}
    for det in detachments:
        remaining = [d for d in detachments if d != det]
        remaining_pvs = {k: market_pvs[k] for k in remaining}
        basecorr = calibrate_basecorr_curve(tenor, remaining, remaining_pvs, curve, recovery, n_quad=n_quad)
        surface = build_rho_surface({tenor: basecorr})

        k1 = 0.0
        for d in sorted(detachments):
            if d == det:
                break
            k1 = d
        rho = surface(tenor, det)
        model_pv = price_tranche_lhp(tenor, k1, det, rho, curve, recovery, n_quad=n_quad).pv
        errors[det] = model_pv - market_pvs[det]

    return errors
