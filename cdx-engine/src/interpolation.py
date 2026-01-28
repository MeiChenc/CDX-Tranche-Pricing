from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.interpolate import PchipInterpolator


def build_rho_surface(basecorr_by_tenor: Dict[float, Dict[float, float]]) -> Callable[[float, float], float]:
    tenors = sorted(basecorr_by_tenor.keys())
    dets = sorted(next(iter(basecorr_by_tenor.values())).keys())

    grid = np.array([[basecorr_by_tenor[t][k] for k in dets] for t in tenors])

    tenor_interp = [PchipInterpolator(tenors, grid[:, j]) for j in range(len(dets))]

    def surface(t: float, k: float) -> float:
        rho_by_det = np.array([interp(t) for interp in tenor_interp])
        det_interp = PchipInterpolator(dets, rho_by_det)
        return float(det_interp(k))

    return surface
