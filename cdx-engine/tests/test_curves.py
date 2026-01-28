from src.curves import bootstrap_from_spreads


def test_bootstrap_from_spreads_monotonic():
    tenors = [1.0, 3.0, 5.0]
    spreads = [0.01, 0.015, 0.02]
    curve = bootstrap_from_spreads(tenors, spreads, recovery=0.4)
    surv = [curve.survival(t) for t in tenors]
    assert surv[0] >= surv[1] >= surv[2]
