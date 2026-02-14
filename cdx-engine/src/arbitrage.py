from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def check_monotonicity(values: np.ndarray) -> bool:
    return np.all(np.diff(values) >= -1e-12)


def check_arbitrage(surface: Dict[float, Dict[float, float]]) -> Dict[str, Tuple[float, float]]:
    report: Dict[str, Tuple[float, float]] = {}
    tenors = sorted(surface.keys())
    dets = sorted(next(iter(surface.values())).keys())

    for t in tenors:
        vals = np.array([surface[t][k] for k in dets])
        if not check_monotonicity(vals):
            report[f"tenor_{t}"] = (float(vals.min()), float(vals.max()))

    for k in dets:
        vals = np.array([surface[t][k] for t in tenors])
        if not check_monotonicity(vals):
            report[f"det_{k}"] = (float(vals.min()), float(vals.max()))

    return report


def fix_surface(surface: Dict[float, Dict[float, float]]) -> Dict[float, Dict[float, float]]:
    fixed: Dict[float, Dict[float, float]] = {}
    for tenor, smile in surface.items():
        dets = sorted(smile.keys())
        vals = np.array([smile[k] for k in dets])
        vals_fixed = np.maximum.accumulate(vals)
        fixed[tenor] = {k: float(v) for k, v in zip(dets, vals_fixed)}
    return fixed
