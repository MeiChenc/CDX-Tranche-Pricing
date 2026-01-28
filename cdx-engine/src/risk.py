from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import Curve, build_index_curve
from .pricer_tranche import price_tranche_lhp


@dataclass
class DV01Result:
    tranche_dv01: float
    index_dv01: float

    @property
    def hedge_ratio(self) -> float:
        if np.isclose(self.index_dv01, 0.0):
            return float("inf")
        return self.tranche_dv01 / self.index_dv01


def compute_dv01(
    tenor: float,
    k1: float,
    k2: float,
    rho: float,
    curve: Curve,
    recovery: float,
    bump_bp: float = 1.0,
    payment_freq: int = 4,
) -> DV01Result:
    base_pv = price_tranche_lhp(tenor, k1, k2, rho, curve, recovery, payment_freq=payment_freq).pv

    bump = bump_bp / 1e4
    bumped_spreads = curve.hazard * (1.0 + bump)
    bumped_curve = Curve(times=curve.times, hazard=bumped_spreads)
    bumped_pv = price_tranche_lhp(
        tenor, k1, k2, rho, bumped_curve, recovery, payment_freq=payment_freq
    ).pv

    tranche_dv01 = (bumped_pv - base_pv) / bump

    index_curve = build_index_curve(curve.times, curve.hazard * (1.0 - recovery), recovery)
    index_pv_base = price_tranche_lhp(
        tenor, k1, k2, rho, index_curve, recovery, payment_freq=payment_freq
    ).pv
    index_bumped = Curve(times=index_curve.times, hazard=index_curve.hazard * (1.0 + bump))
    index_pv_bumped = price_tranche_lhp(
        tenor, k1, k2, rho, index_bumped, recovery, payment_freq=payment_freq
    ).pv
    index_dv01 = (index_pv_bumped - index_pv_base) / bump

    return DV01Result(tranche_dv01=tranche_dv01, index_dv01=index_dv01)
