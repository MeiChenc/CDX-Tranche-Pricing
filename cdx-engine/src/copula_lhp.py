from __future__ import annotations

import numpy as np

from .utils_math import norm_cdf, norm_ppf


def conditional_default_prob(qt: float, rho: float, m: float) -> float:
    qt = np.clip(qt, 1e-12, 1 - 1e-12)
    threshold = norm_ppf(qt)
    denom = np.sqrt(1.0 - rho)
    return float(norm_cdf((threshold - rho * m) / denom))


def conditional_loss(qt: float, rho: float, m: float, recovery: float) -> float:
    return (1.0 - recovery) * conditional_default_prob(qt, rho, m)
