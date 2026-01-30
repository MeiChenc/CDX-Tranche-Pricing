from __future__ import annotations

from .curves import Curve
from .risk import compute_dv01


def compute_hedge_ratio(
    tenor: float,
    k1: float,
    k2: float,
    rho: float,
    curve: Curve,
    recovery: float,
    bump_bp: float = 1.0,
) -> float:
    dv01 = compute_dv01(tenor, k1, k2, rho, curve, recovery, bump_bp=bump_bp)
    return dv01.hedge_ratio
