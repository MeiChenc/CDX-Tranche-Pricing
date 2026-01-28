from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class MarketSnap:
    date: pd.Timestamp
    tenors: List[float]
    index_quotes: Dict[float, dict]
    tranche_quotes: Dict[float, dict]
    constituent_spreads: Dict[float, pd.Series]
    stale_or_bad: bool = False


def _extract_tranche_quotes(row: pd.Series) -> dict:
    tranche_cols = [col for col in row.index if col.startswith("tranche_")]
    return {col.replace("tranche_", ""): row[col] for col in tranche_cols if pd.notna(row[col])}


def read_market_data(
    index_csv: str,
    constituents_csv: str,
    stale_threshold: float = 3.0,
    min_constituents: int = 100,
) -> Tuple[List[MarketSnap], List[pd.Timestamp]]:
    index_df = pd.read_csv(index_csv, parse_dates=["date"])
    cons_df = pd.read_csv(constituents_csv, parse_dates=["date"])

    index_df = index_df.sort_values(["date", "tenor"])
    cons_df = cons_df.sort_values(["date", "tenor"])

    market_snaps: List[MarketSnap] = []
    bad_dates: List[pd.Timestamp] = []

    for date, date_slice in index_df.groupby("date"):
        tenors = sorted(date_slice["tenor"].unique())
        index_quotes: Dict[float, dict] = {}
        tranche_quotes: Dict[float, dict] = {}
        constituent_spreads: Dict[float, pd.Series] = {}
        stale_or_bad = False

        cons_slice = cons_df[cons_df["date"] == date]
        for tenor in tenors:
            tenor_index = date_slice[date_slice["tenor"] == tenor].iloc[0]
            index_quotes[tenor] = {
                "index_spread": tenor_index.get("index_spread"),
                "index_upfront": tenor_index.get("index_upfront"),
            }
            tranche_quotes[tenor] = _extract_tranche_quotes(tenor_index)
            tenor_cons = cons_slice[cons_slice["tenor"] == tenor]
            constituent_spreads[tenor] = tenor_cons.drop(columns=["date", "tenor"]).stack()

        missing_constituents = any(len(spreads.dropna()) < min_constituents for spreads in constituent_spreads.values())
        quote_jumps = date_slice[["index_spread"]].pct_change().abs().max().max()
        if missing_constituents and quote_jumps > stale_threshold:
            stale_or_bad = True
            bad_dates.append(date)

        market_snaps.append(
            MarketSnap(
                date=date,
                tenors=tenors,
                index_quotes=index_quotes,
                tranche_quotes=tranche_quotes,
                constituent_spreads=constituent_spreads,
                stale_or_bad=stale_or_bad,
            )
        )

    return market_snaps, bad_dates
