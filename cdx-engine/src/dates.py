from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DayCount:
    convention: str = "ACT/360"

    def year_fraction(self, days: int) -> float:
        if self.convention.upper() == "ACT/360":
            return days / 360.0
        if self.convention.upper() == "ACT/365":
            return days / 365.0
        raise ValueError(f"Unsupported day count: {self.convention}")
