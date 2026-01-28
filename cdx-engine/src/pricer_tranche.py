from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .copula_lhp import conditional_loss
from .curves import Curve
from .utils_math import hermite_nodes_weights


@dataclass
class TranchePV:
    premium_leg: float
    protection_leg: float

    @property
    def pv(self) -> float:
        return self.protection_leg - self.premium_leg


def tranche_expected_loss(
    qt: float,
    k1: float,
    k2: float,
    rho: float,
    recovery: float,
    n_quad: int,
) -> float:
    nodes, weights = hermite_nodes_weights(n_quad)
    losses = []
    for m in nodes:
        loss = conditional_loss(qt, rho, m, recovery)
        tranche_loss = np.minimum(np.maximum(loss - k1, 0.0), k2 - k1)
        losses.append(tranche_loss)
    return float(np.sum(weights * np.asarray(losses)))


def _discount_factor(disc_curve: Optional[Curve], t: float) -> float:
    if disc_curve is None:
        return 1.0
    return disc_curve.survival(t)


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
) -> TranchePV:
    times = np.linspace(0.0, tenor, int(tenor * payment_freq) + 1)[1:]
    losses = [tranche_expected_loss(curve.default_prob(t), k1, k2, rho, recovery, n_quad) for t in times]

    premium_leg = 0.0
    protection_leg = 0.0
    prev_loss = 0.0
    dt = 1.0 / payment_freq

    for t, loss in zip(times, losses):
        df = _discount_factor(disc_curve, t)
        premium_leg += df * dt * (k2 - k1 - loss)
        protection_leg += df * (loss - prev_loss)
        prev_loss = loss

    return TranchePV(premium_leg=premium_leg, protection_leg=protection_leg)


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
        [tranche_expected_loss(curve.default_prob(t), k1, k2, rho, recovery, n_quad) for t in tenors]
    )
    return tenors, expected_losses
