from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Curve:
    times: np.ndarray
    hazard: np.ndarray

    def survival(self, t: float) -> float:
        times = np.asarray(self.times)
        hazard = np.asarray(self.hazard)
        if t <= times[0]:
            return float(np.exp(-hazard[0] * t))
        integral = 0.0
        prev = 0.0
        for time, hz in zip(times, hazard):
            dt = min(t, time) - prev
            if dt > 0:
                integral += hz * dt
                prev = time
            if t <= time:
                break
        if t > times[-1]:
            integral += hazard[-1] * (t - times[-1])
        return float(np.exp(-integral))

    def default_prob(self, t: float) -> float:
        return 1.0 - self.survival(t)


def bootstrap_from_spreads(tenors: Iterable[float], spreads: Iterable[float], recovery: float) -> Curve:
    tenors = np.asarray(list(tenors), dtype=float)
    spreads = np.asarray(list(spreads), dtype=float)
    if len(tenors) != len(spreads):
        raise ValueError("tenors and spreads must be the same length")
    hazard = np.maximum(spreads / max(1e-12, (1.0 - recovery)), 1e-12)
    return Curve(times=tenors, hazard=hazard)


def build_index_curve(tenors: Iterable[float], index_spreads: Iterable[float], recovery: float) -> Curve:
    return bootstrap_from_spreads(tenors, index_spreads, recovery)


def build_constituent_curves(tenors: Iterable[float], spreads_matrix: np.ndarray, recovery: float) -> list[Curve]:
    curves: list[Curve] = []
    for spreads in spreads_matrix:
        curves.append(bootstrap_from_spreads(tenors, spreads, recovery))
    return curves
