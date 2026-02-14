from __future__ import annotations

import numpy as np

from .utils_math import norm_cdf, norm_ppf


def conditional_default_prob(q_survival: float, rho: float, m: float) -> float:
    """
    Conditional default probability under one-factor Gaussian copula using survival Q(t).

    p(t|M) = Phi( (Phi^{-1}(Q(t)) - sqrt(rho) * M) / sqrt(1 - rho) )
    """
    q_survival = np.clip(q_survival, 1e-12, 1 - 1e-12)
    if rho < 0 or rho >= 1:
        raise ValueError("rho must be in [0, 1)")
    q_default = 1.0 - q_survival
    q_default = np.clip(q_default, 1e-12, 1 - 1e-12)
    threshold = norm_ppf(q_default)
    denom = np.sqrt(1.0 - rho)
    return float(norm_cdf((threshold - np.sqrt(rho) * m) / denom))


def conditional_loss(q_survival: float, rho: float, m: float, recovery: float) -> float:
    return (1.0 - recovery) * conditional_default_prob(q_survival, rho, m)
