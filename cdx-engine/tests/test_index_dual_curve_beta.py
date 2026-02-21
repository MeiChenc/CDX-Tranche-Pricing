import numpy as np
import pandas as pd
import pytest

from src.basis_adjustment_utils import build_index_dual_curve_beta_bundle


def test_dual_curve_beta_identity_when_spreads_match() -> None:
    snapshot = pd.DataFrame(
        {
            "tenor": [1.0, 3.0, 5.0],
            "Index_0_100_Spread": [100.0, 120.0, 140.0],
            "Index_Mid": [100.0, 120.0, 140.0],
        }
    )

    _, _, _, beta_knot, beta_cum, _, _ = build_index_dual_curve_beta_bundle(snapshot, disc_curve=None)

    assert np.allclose(beta_knot, 1.0, atol=1e-6)
    assert np.allclose(beta_cum, 1.0, atol=1e-6)


def test_dual_curve_beta_above_one_when_market_spreads_higher() -> None:
    snapshot = pd.DataFrame(
        {
            "tenor": [1.0, 3.0, 5.0],
            "Index_0_100_Spread": [100.0, 120.0, 140.0],
            "Index_Mid": [130.0, 156.0, 182.0],  # 1.3x theoretical
        }
    )

    _, _, _, beta_knot, beta_cum, _, _ = build_index_dual_curve_beta_bundle(snapshot, disc_curve=None)

    assert np.all(beta_knot > 1.0)
    assert np.all(beta_cum > 1.0)


def test_dual_curve_beta_requires_columns() -> None:
    snapshot = pd.DataFrame(
        {
            "tenor": [1.0, 3.0, 5.0],
            "Index_0_100_Spread": [100.0, 120.0, 140.0],
        }
    )

    with pytest.raises(ValueError, match="Missing market spread column"):
        build_index_dual_curve_beta_bundle(snapshot, disc_curve=None)
