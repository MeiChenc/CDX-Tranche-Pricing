from src.arbitrage import check_arbitrage, fix_surface


def test_fix_surface_enforces_monotonicity():
    surface = {5.0: {0.03: 0.2, 0.07: 0.15, 0.1: 0.25}}
    report = check_arbitrage(surface)
    assert "tenor_5.0" in report
    fixed = fix_surface(surface)
    report_fixed = check_arbitrage(fixed)
    assert report_fixed == {}
