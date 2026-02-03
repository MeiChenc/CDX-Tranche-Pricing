from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq


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
    """
    Simple hazard approximation: λ ≈ S / (1 - R).

    Notes:
    - Spreads are decimal (e.g. 100bp = 0.01).
    - This ignores accrual-on-default, payment frequency, and discounting.
    """
    tenors = np.asarray(list(tenors), dtype=float)
    spreads = np.asarray(list(spreads), dtype=float)
    if len(tenors) != len(spreads):
        raise ValueError("tenors and spreads must be the same length")
    hazard = np.maximum(spreads / max(1e-12, (1.0 - recovery)), 1e-12)
    return Curve(times=tenors, hazard=hazard)


def _discount_factor(disc_curve: Optional[Callable[[float], float] | Curve], t: float) -> float:
    if disc_curve is None:
        return 1.0
    if isinstance(disc_curve, Curve):
        return disc_curve.survival(t)
    return float(disc_curve(t))


def _build_cds_schedule(tenor: float, payment_freq: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if payment_freq <= 0:
        raise ValueError("payment_freq must be positive")
    if tenor <= 0:
        raise ValueError("tenor must be positive")
    step = 1.0 / payment_freq
    pay_times = np.arange(step, tenor + 1e-12, step, dtype=float)
    if pay_times.size == 0 or abs(pay_times[-1] - tenor) > 1e-10:
        pay_times = np.append(pay_times, tenor)
    start_times = np.concatenate(([0.0], pay_times[:-1]))
    accrual_fracs = pay_times - start_times
    return start_times, pay_times, accrual_fracs


def _survival_piecewise_constant(t: float, knots: Sequence[float], lambdas: Sequence[float]) -> float:
    if t <= 0:
        return 1.0
    if len(knots) != len(lambdas) + 1:
        raise ValueError("knots must have length len(lambdas) + 1")
    integ = 0.0
    for i in range(1, len(knots)):
        a, b = knots[i - 1], knots[i]
        if t <= a:
            break
        dt = min(t, b) - a
        if dt > 0:
            integ += lambdas[i - 1] * dt
        if t <= b:
            break
    if t > knots[-1]:
        integ += lambdas[-1] * (t - knots[-1])
    return float(np.exp(-integ))


def _premium_default_legs(
    start_times: np.ndarray,
    pay_times: np.ndarray,
    accrual_fracs: np.ndarray,
    disc_curve: Optional[Callable[[float], float] | Curve],
    knots: Sequence[float],
    lambdas: Sequence[float],
    include_accrual_on_default: bool = True,
) -> Tuple[float, float]:
    a = 0.0
    b = 0.0
    for t0, t1, delta in zip(start_times, pay_times, accrual_fracs):
        q0 = _survival_piecewise_constant(float(t0), knots, lambdas)
        q1 = _survival_piecewise_constant(float(t1), knots, lambdas)
        df = _discount_factor(disc_curve, float(t1))
        a += float(delta) * df * q1
        b += df * (q0 - q1)
        if include_accrual_on_default:
            a += 0.5 * float(delta) * df * (q0 - q1)
    return a, b


def bootstrap_from_cds_spreads(
    tenors: Iterable[float],
    spreads: Iterable[float],
    recovery: float,
    payment_freq: int = 4,
    disc_curve: Optional[Callable[[float], float] | Curve] = None,
) -> Curve:
    """
    Piecewise-constant hazard bootstrapping from CDS par spreads.

    Assumptions:
    - Spreads are decimal (e.g. 100bp = 0.01), par CDS with running coupon.
    - Payment frequency is fixed (CDX typically quarterly).
    - Accrual-on-default uses the standard 1/2 accrual approximation.
    - Discounting uses `disc_curve` if provided; otherwise DF(t)=1.
    """
    tenors_arr = np.asarray(list(tenors), dtype=float)
    spreads_arr = np.asarray(list(spreads), dtype=float)
    if len(tenors_arr) != len(spreads_arr):
        raise ValueError("tenors and spreads must be the same length")
    if np.any(tenors_arr <= 0):
        raise ValueError("tenors must be positive")
    if np.any(spreads_arr < 0):
        raise ValueError("spreads must be non-negative")
    if not np.all(np.diff(tenors_arr) > 0):
        raise ValueError("tenors must be strictly increasing")
    if recovery < 0 or recovery >= 1:
        raise ValueError("recovery must be in [0, 1)")

    knots = [0.0]
    lambdas: list[float] = []

    for tenor, spread in zip(tenors_arr, spreads_arr):
        start_times, pay_times, accrual_fracs = _build_cds_schedule(float(tenor), payment_freq)

        def f(lam: float) -> float:
            lambdas_try = lambdas + [float(lam)]
            knots_try = knots + [float(tenor)]
            a, b = _premium_default_legs(
                start_times,
                pay_times,
                accrual_fracs,
                disc_curve,
                knots_try,
                lambdas_try,
                include_accrual_on_default=True,
            )
            return float(spread) * a - (1.0 - recovery) * b

        lo, hi = 1e-8, 5.0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            hi2 = hi
            for _ in range(20):
                hi2 *= 2.0
                fhi2 = f(hi2)
                if flo * fhi2 <= 0:
                    hi, fhi = hi2, fhi2
                    break
            else:
                raise ValueError("Root not bracketed. Check inputs or expand bracket.")

        lam = float(brentq(f, lo, hi, maxiter=200, xtol=1e-12))

        knots.append(float(tenor))
        lambdas.append(float(lam))

    return Curve(times=tenors_arr, hazard=np.asarray(lambdas, dtype=float))


def build_index_curve(
    tenors: Iterable[float],
    index_spreads: Iterable[float],
    recovery: float,
    payment_freq: int = 4,
    disc_curve: Optional[Callable[[float], float] | Curve] = None,
) -> Curve:
    """
    Build index hazard curve from par spreads using piecewise-constant bootstrapping.

    Spreads are decimal (e.g. 100bp = 0.01). Discounting is optional; if omitted
    the function assumes DF(t)=1 for stability in notebooks/tests.
    """
    return bootstrap_from_cds_spreads(
        tenors,
        index_spreads,
        recovery,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )


def build_constituent_curves(
    tenors: Iterable[float],
    spreads_matrix: np.ndarray,
    recovery: float,
    payment_freq: int = 4,
    disc_curve: Optional[Callable[[float], float] | Curve] = None,
) -> list[Curve]:
    curves: list[Curve] = []
    for spreads in spreads_matrix:
        curves.append(
            bootstrap_from_cds_spreads(
                tenors,
                spreads,
                recovery,
                payment_freq=payment_freq,
                disc_curve=disc_curve,
            )
        )
    return curves
