from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm


def norm_cdf(x: float) -> float:
    return float(norm.cdf(x))


def norm_ppf(x: float) -> float:
    return float(norm.ppf(x))


def hermite_nodes_weights(n_quad: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(n_quad)
    scaled_nodes = np.sqrt(2.0) * nodes
    scaled_weights = weights / np.sqrt(np.pi)
    return scaled_nodes, scaled_weights
