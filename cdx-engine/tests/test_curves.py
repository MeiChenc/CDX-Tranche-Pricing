import numpy as np

from src.curves import bootstrap_from_cds_spreads, bootstrap_from_spreads


def test_bootstrap_from_spreads_monotonic():
    tenors = [1.0, 3.0, 5.0]
    spreads = [0.01, 0.015, 0.02]
    curve = bootstrap_from_spreads(tenors, spreads, recovery=0.4)
    surv = [curve.survival(t) for t in tenors]
    assert surv[0] >= surv[1] >= surv[2]


def test_bootstrap_from_cds_spreads_monotonic():
    tenors = [1.0, 3.0, 5.0]
    spreads = [0.01, 0.015, 0.02]
    curve = bootstrap_from_cds_spreads(tenors, spreads, recovery=0.4)
    surv = [curve.survival(t) for t in tenors]
    assert surv[0] >= surv[1] >= surv[2]


def test_bootstrap_from_cds_spreads_zero_spread_flat_survival():
    tenors = [1.0, 2.0, 3.0]
    spreads = [0.0, 0.0, 0.0]
    curve = bootstrap_from_cds_spreads(tenors, spreads, recovery=0.4)
    surv = np.array([curve.survival(t) for t in tenors])
    assert np.allclose(surv, 1.0, atol=1e-6)
