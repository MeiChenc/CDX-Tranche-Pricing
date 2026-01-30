from __future__ import annotations

from src.curves import Curve
from src.loocv import run_loocv


def main() -> None:
    curve = Curve(times=[5.0], hazard=[0.02])
    detachments = [0.03, 0.07, 0.1]
    market_pvs = {0.03: 0.001, 0.07: 0.002, 0.1: 0.003}
    errors = run_loocv(5.0, detachments, market_pvs, curve, recovery=0.4)
    print(errors)


if __name__ == "__main__":
    main()
