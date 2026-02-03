import numpy as np

from src.basis import basis_adjust_curves_beta, build_average_curve
from src.curves import Curve


def test_basis_adjustment_homogeneous_hazards_matches_index() -> None:
    times = np.array([1.0, 2.0, 3.0])
    base_hazard = np.array([0.02, 0.02, 0.02])
    curves = [Curve(times=times, hazard=base_hazard.copy()) for _ in range(3)]
    index_curve = Curve(times=times, hazard=np.array([0.03, 0.03, 0.03]))

    adjusted, betas = basis_adjust_curves_beta(curves, index_curve, recovery=0.4)
    assert np.allclose(betas, 1.5, rtol=1e-6, atol=1e-8)

    avg_curve = build_average_curve(adjusted)
    for t in times:
        assert np.isclose(avg_curve.default_prob(float(t)), index_curve.default_prob(float(t)), rtol=1e-6, atol=1e-8)


def test_basis_adjustment_matches_index_expected_loss() -> None:
    times = np.array([1.0, 2.0, 3.0])
    curves = [
        Curve(times=times, hazard=np.array([0.01, 0.01, 0.01])),
        Curve(times=times, hazard=np.array([0.03, 0.03, 0.03])),
    ]
    index_curve = Curve(times=times, hazard=np.array([0.02, 0.02, 0.02]))
    recovery = 0.4

    adjusted, _ = basis_adjust_curves_beta(curves, index_curve, recovery=recovery)

    for t in times:
        target_el = (1.0 - recovery) * index_curve.default_prob(float(t))
        avg_el = np.mean([(1.0 - recovery) * c.default_prob(float(t)) for c in adjusted])
        assert np.isclose(avg_el, target_el, rtol=1e-4, atol=5e-5)
