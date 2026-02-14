from src.calibration_basecorr import calibrate_basecorr_curve
from src.curves import Curve


def test_calibration_recovers_rho():
    curve = Curve(times=[5.0], hazard=[0.02])
    tenor = 5.0
    detachments = [0.03]
    market_pvs = {0.03: 0.0}
    basecorr = calibrate_basecorr_curve(tenor, detachments, market_pvs, curve, recovery=0.4)
    assert 0.0 < basecorr[0.03] < 1.0
