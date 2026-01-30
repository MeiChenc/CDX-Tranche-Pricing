from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import brentq

from .curves import Curve


def adjust_curves(constituent_curves: List[Curve], index_curve: Curve) -> List[Curve]:
    times = index_curve.times
    adjusted: List[Curve] = []

    for curve in constituent_curves:
        adjusted_hazard = curve.hazard.copy()
        for idx, t in enumerate(times):
            index_q = index_curve.default_prob(t)
            avg_q = np.mean([c.default_prob(t) for c in constituent_curves])

            if np.isclose(avg_q, index_q, atol=1e-10):
                continue

            def objective(beta: float) -> float:
                hz = max(curve.hazard[idx] + beta, 1e-12)
                return 1.0 - np.exp(-hz * t) - index_q

            beta = brentq(objective, -0.5, 5.0)
            adjusted_hazard[idx] = max(curve.hazard[idx] + beta, 1e-12)

        adjusted.append(Curve(times=curve.times, hazard=adjusted_hazard))

    return adjusted
