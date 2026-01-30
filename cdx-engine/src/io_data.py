from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd
import numpy as np


@dataclass
class MarketSnap:
    date: pd.Timestamp
    tenors: List[float]
    index_quotes: Dict[float, dict]
    tranche_quotes: Dict[float, dict]
    constituent_spreads: Dict[float, pd.Series]
    stale_or_bad: bool = False


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        if col.lower() == "date" and col != "Date":
            renamed[col] = "Date"
        if col.lower() == "tenor" and col != "tenor":
            renamed[col] = "tenor"
    return df.rename(columns=renamed)


def _parse_tenor_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().upper()
    if text.endswith("Y"):
        return float(text[:-1])
    if text.endswith("W"):
        return float(text[:-1]) / 52.0
    if text.endswith("D"):
        return float(text[:-1]) / 365.0
    if text.endswith("M"):
        return float(text[:-1]) / 12.0
    return float(text)


def _coerce_tenor_column(df: pd.DataFrame) -> pd.DataFrame:
    if "tenor" in df.columns:
        df["tenor"] = df["tenor"].map(_parse_tenor_value)
    return df


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _standardize_index_columns(index_df: pd.DataFrame) -> pd.DataFrame:
    if "index_spread" not in index_df.columns:
        col = _find_column(index_df, "index_spread", "Index_Mid", "Index_Last", "Index_0_100_Spread")
        if col:
            index_df["index_spread"] = index_df[col]
        else:
            bid = _find_column(index_df, "Index_Bid")
            ask = _find_column(index_df, "Index_Ask")
            if bid and ask:
                index_df["index_spread"] = (index_df[bid] + index_df[ask]) / 2.0
            else:
                index_df["index_spread"] = pd.NA
    if "index_upfront" not in index_df.columns:
        col = _find_column(index_df, "index_upfront", "Index_Upfront")
        index_df["index_upfront"] = index_df[col] if col else pd.NA
    return index_df


def _extract_tranche_quotes(row: pd.Series) -> dict:
    tranche_cols = []
    for col in row.index:
        low = col.lower()
        if low.startswith("tranche_") or low.startswith(
            ("equity_", "mezz_", "senior_", "supersenior_")
        ):
            tranche_cols.append(col)
    return {col.lower().replace("tranche_", ""): row[col] for col in tranche_cols if pd.notna(row[col])}


def _extract_constituent_tenors(cons_df: pd.DataFrame) -> List[float]:
    if "tenor" in cons_df.columns:
        return sorted(pd.Series(cons_df["tenor"]).dropna().unique().tolist())
    tenors = []
    for col in cons_df.columns:
        if col.lower().startswith("spread_"):
            tenor = _parse_tenor_value(col.split("_", 1)[1])
            tenors.append(tenor)
    return sorted(set(tenors))


def _get_spread_column_map(cons_df: pd.DataFrame) -> Dict[float, str]:
    spread_cols: Dict[float, str] = {}
    for col in cons_df.columns:
        if col.lower().startswith("spread_"):
            tenor = _parse_tenor_value(col.split("_", 1)[1])
            spread_cols[tenor] = col
    return spread_cols


def read_market_data(
    index_csv: str,
    constituents_csv: str,
    stale_threshold: float = 3.0,
    min_constituents: int = 100,
) -> Tuple[List[MarketSnap], List[pd.Timestamp]]:
    index_df = pd.read_csv(index_csv, parse_dates=["Date"])
    cons_df = pd.read_csv(constituents_csv, parse_dates=["Date"])

    index_df = _normalize_columns(index_df)
    cons_df = _normalize_columns(cons_df)
    index_df = _coerce_tenor_column(index_df)
    cons_df = _coerce_tenor_column(cons_df)
    index_df = _standardize_index_columns(index_df)

    if "tenor" in index_df.columns:
        index_df = index_df.sort_values(["Date", "tenor"])
    if "tenor" in cons_df.columns:
        cons_df = cons_df.sort_values(["Date", "tenor"])

    market_snaps: List[MarketSnap] = []
    bad_dates: List[pd.Timestamp] = []
    cons_tenors = _extract_constituent_tenors(cons_df)
    cons_spread_cols = _get_spread_column_map(cons_df) if "Company" in cons_df.columns else {}

    for date, date_slice in index_df.groupby("Date"):
        date_tenors = sorted(date_slice["tenor"].unique())
        if cons_tenors:
            tenors = [t for t in date_tenors if t in cons_tenors]
            if not tenors:
                tenors = date_tenors
        else:
            tenors = date_tenors
        index_quotes: Dict[float, dict] = {}
        tranche_quotes: Dict[float, dict] = {}
        constituent_spreads: Dict[float, pd.Series] = {}
        stale_or_bad = False

        cons_slice = cons_df[cons_df["Date"] == date]
        for tenor in tenors:
            tenor_index = date_slice[date_slice["tenor"] == tenor].iloc[0]
            index_quotes[tenor] = {
                "index_spread": tenor_index.get("index_spread"),
                "index_upfront": tenor_index.get("index_upfront"),
            }
            tranche_quotes[tenor] = _extract_tranche_quotes(tenor_index)
            if "Company" in cons_slice.columns and cons_spread_cols:
                col = cons_spread_cols.get(tenor)
                if col and col in cons_slice.columns:
                    constituent_spreads[tenor] = cons_slice.set_index("Company")[col]
                else:
                    constituent_spreads[tenor] = pd.Series(dtype=float)
            else:
                tenor_cons = cons_slice[cons_slice["tenor"] == tenor]
                constituent_spreads[tenor] = tenor_cons.drop(columns=["Date", "tenor"]).stack()

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


def read_ois_discount_curve(
    ois_csv: str,
    target_date: pd.Timestamp | None = None,
) -> Callable[[float], float]:
    """
    Build a discount factor function from OIS market data.

    Assumptions:
    - OIS_Rate is in percent (e.g. 3.90 means 3.90%).
    - Tenor strings are like 1W/3M/5Y and converted to year fractions.
    - Zero rates are linearly interpolated; DF uses continuous compounding.
    """
    ois_df = pd.read_csv(ois_csv, parse_dates=["Date"])
    ois_df = _normalize_columns(ois_df)
    ois_df = _coerce_tenor_column(ois_df)
    if "OIS_Rate" not in ois_df.columns:
        col = _find_column(ois_df, "OIS_Rate", "OIS", "Rate")
        if col is None:
            raise ValueError("OIS_Rate column not found in OIS data")
        ois_df["OIS_Rate"] = ois_df[col]

    if target_date is None:
        target_date = ois_df["Date"].max()
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.normalize()
        ois_slice = ois_df[ois_df["Date"].dt.normalize() == target_date].copy()
    else:
        ois_slice = ois_df[ois_df["Date"].dt.date == target_date].copy()
    if ois_slice.empty:
        raise ValueError(f"No OIS rows found for date {target_date}.")

    tenors = ois_slice["tenor"].to_numpy(dtype=float)
    rates = ois_slice["OIS_Rate"].to_numpy(dtype=float) / 100.0
    order = np.argsort(tenors)
    tenors = tenors[order]
    rates = rates[order]

    def df(t: float) -> float:
        if t <= 0:
            return 1.0
        r = np.interp(t, tenors, rates, left=rates[0], right=rates[-1])
        return float(np.exp(-r * t))

    return df
