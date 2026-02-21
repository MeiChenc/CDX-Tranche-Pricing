import numpy as np

from src.curves import Curve
from src.pricer_tranche import price_tranche_lhp


def test_price_tranche_zero_hazard_has_zero_protection_leg() -> None:
    curve = Curve(times=np.array([1.0, 3.0, 5.0]), hazard=np.array([0.0, 0.0, 0.0]))
    pv = price_tranche_lhp(
        tenor=1.0,
        k1=0.0,
        k2=0.03,
        rho=0.2,
        curve=curve,
        recovery=0.4,
        n_quad=32,
        payment_freq=4,
        disc_curve=None,
    )
    assert np.isclose(pv.protection_leg, 0.0, atol=1e-12)
    assert np.isclose(pv.premium_leg, 0.03, atol=1e-10)


def test_price_tranche_positive_hazard_has_positive_legs() -> None:
    curve = Curve(times=np.array([1.0, 3.0, 5.0]), hazard=np.array([0.02, 0.02, 0.02]))
    pv = price_tranche_lhp(
        tenor=5.0,
        k1=0.03,
        k2=0.07,
        rho=0.3,
        curve=curve,
        recovery=0.4,
        n_quad=32,
        payment_freq=4,
        disc_curve=None,
    )
    assert pv.protection_leg > 0.0
    assert pv.premium_leg > 0.0
