from __future__ import annotations

import yaml

from src.io_data import read_market_data


def main() -> None:
    with open("configs/params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    market, bad_dates = read_market_data(
        "data/cdx_timeseries.csv",
        "data/constituents_timeseries.csv",
        stale_threshold=params["data_qc"]["stale_threshold"],
        min_constituents=params["data_qc"]["min_constituents"],
    )

    print(f"Loaded {len(market)} market snapshots")
    print(f"Bad dates: {bad_dates}")


if __name__ == "__main__":
    main()
