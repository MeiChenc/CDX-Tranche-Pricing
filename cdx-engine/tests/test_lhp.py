from src.copula_lhp import conditional_default_prob


def test_conditional_default_prob_bounds():
    prob = conditional_default_prob(0.02, rho=0.3, m=0.0)
    assert 0.0 < prob < 1.0
