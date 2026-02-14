from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import logging
import numpy as np

from .copula_lhp import conditional_loss
from .curves import Curve
from .utils_math import hermite_nodes_weights

@dataclass
class TranchePV:
    premium_leg: float
    protection_leg: float
    upfront: float = 0

    @property
    def pv(self) -> float:
        return self.protection_leg - self.premium_leg

# absolute expected loss (percentage loss for a tranche)
def tranche_expected_loss(
    q_survival: float,
    k1: float,
    k2: float,
    rho: float,
    recovery: float,
    n_quad: int,
) -> float:
    nodes, weights = hermite_nodes_weights(n_quad)
    losses = []
    for m in nodes:
        loss = conditional_loss(q_survival, rho, m, recovery)  #Protection leg
        tranche_loss = np.minimum(np.maximum(loss - k1, 0.0), k2 - k1)
        losses.append(tranche_loss)
    return float(np.sum(weights * np.asarray(losses)))


def tranche_expected_loss_fraction(
    q_survival: float,
    k1: float,
    k2: float,
    rho: float,
    recovery: float,
    n_quad: int,
) -> float:
    width = float(k2 - k1)
    if width <= 0:
        raise ValueError("k2 must be greater than k1")
    return tranche_expected_loss(
        q_survival=q_survival,
        k1=k1,
        k2=k2,
        rho=rho,
        recovery=recovery,
        n_quad=n_quad,
    ) / width


def _discount_factor(disc_curve: Optional[Curve | Callable[[float], float]], t: float) -> float:
    if disc_curve is None:
        return 1.0
    if isinstance(disc_curve, Curve):
        times = np.asarray(disc_curve.times, dtype=float)
        rates = np.asarray(disc_curve.hazard, dtype=float) # it's ois interest rate 
        r = np.interp(t, times, rates, left=rates[0], right=rates[-1])
        return float(np.exp(-r * t))
    return float(disc_curve(t))

def generate_base_el(
    tenor: float,
    k1: float,
    k2: float,
    rho: float,
    curve: Curve,
    recovery: float,
    n_quad: int = 64,
    payment_freq: int = 4,
) -> float:   
    times = np.linspace(0.0, tenor, int(tenor * payment_freq) + 1)[1:]
    losses = [tranche_expected_loss(curve.survival(t), k1, k2, rho, recovery, n_quad) for t in times]
    logging.info("Tranche expected losses for tenor %.2f [k1=%.2f,k2=%.2f]: %s", tenor, k1, k2, losses[-1])
    return losses[-1]

def price_tranche_lhp(
    tenor: float,
    k1: float,
    k2: float,
    rho: float,
    curve: Curve,
    recovery: float,
    n_quad: int = 64,
    payment_freq: int = 4,
    disc_curve: Optional[Curve] = None,
    upfront: float = 0
) -> TranchePV:
    times = np.linspace(0.0, tenor, int(tenor * payment_freq) + 1)[1:]
    losses = [tranche_expected_loss(curve.survival(t), k1, k2, rho, recovery, n_quad) for t in times]
    # logging.info("Tranche expected losses for tenor %.2f [k1=%.2f,k2=%.2f]: %s", tenor, k1, k2, losses)
    premium_leg = 0.0
    protection_leg = 0.0
    prev_loss = 0.0
    dt = 1.0 / payment_freq

    for t, loss in zip(times, losses):
        df = _discount_factor(disc_curve, t)
        premium_leg += df * dt * (k2 - k1 - loss)
        # current_remaining = (k2 - k1) - loss
        # prev_remaining = (k2 - k1) - prev_loss
        # premium_leg += df * dt * (current_remaining + prev_remaining) / 2.0
        protection_leg += df * (loss - prev_loss)
        prev_loss = loss

    return TranchePV(premium_leg=premium_leg, protection_leg=protection_leg, upfront=upfront)


def price_tranche_from_curve( 
    tenors: Iterable[float],
    k1: float,
    k2: float,
    rho: float,
    curve: Curve,
    recovery: float,
    n_quad: int = 64,
    payment_freq: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    tenors = np.asarray(list(tenors), dtype=float)
    expected_losses = np.array(
        [tranche_expected_loss(curve.survival(t), k1, k2, rho, recovery, n_quad) for t in tenors]
    )
    return tenors, expected_losses
