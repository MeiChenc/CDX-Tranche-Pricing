from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def _ois_curve_from_csv(ois_csv: str, target_date: pd.Timestamp) -> "Curve":
    from src.curves import Curve

    ois_df = pd.read_csv(ois_csv, parse_dates=["Date"])
    ois_df = _normalize_columns(ois_df)
    ois_df = _coerce_tenor_column(ois_df)
    ois_slice = ois_df[ois_df["Date"].dt.normalize() == target_date].copy()
    if ois_slice.empty:
        raise ValueError(f"No OIS rows found for date {target_date}.")

    tenors = ois_slice["tenor"].to_numpy(dtype=float)
    rates = ois_slice["OIS_Rate"].to_numpy(dtype=float) / 100.0
    order = np.argsort(tenors)
    return Curve(times=tenors[order], hazard=rates[order])


def plot_discount_factors(
    ois_csv: str,
    cdx_csv: str,
    constituents_csv: str,
    recovery_rate: float,
    payment_freq: int = 4,
    target_date: pd.Timestamp | None = None,
    output_path: str | None = None,
    output_table_path: str | None = None,
    output_table_txt_path: str | None = None,
    show_plot: bool = True,
    print_table: bool = True,
    print_ascii: bool = True,
) -> pd.DataFrame:
    """
    Plot discount factors using OIS, index, and constituent data.

    Notes:
    - OIS curve uses OIS_Rate as continuous zero rate.
    - Index/constituent curves are hazard curves bootstrapped from CDS spreads.
    - Uses `_discount_factor` semantics: DF(t)=survival(t) for Curve.
    """
    from src.curves import Curve, build_index_curve
    from src.curves import _discount_factor as _df_curve

    ois_df = pd.read_csv(ois_csv, parse_dates=["Date"])
    ois_df = _normalize_columns(ois_df)
    if target_date is None:
        target_date = ois_df["Date"].max().normalize()
    elif isinstance(target_date, pd.Timestamp):
        target_date = target_date.normalize()

    ois_curve = _ois_curve_from_csv(ois_csv, target_date)
    disc_curve = read_ois_discount_curve(ois_csv, target_date)

    index_df = pd.read_csv(cdx_csv, parse_dates=["Date"])
    index_df = _normalize_columns(index_df)
    index_df = _coerce_tenor_column(index_df)
    index_df = _standardize_index_columns(index_df)
    index_slice = index_df[index_df["Date"].dt.normalize() == target_date].copy()
    if index_slice.empty:
        raise ValueError(f"No CDX rows found for date {target_date}.")
    tenors = index_slice["tenor"].to_numpy(dtype=float)
    spreads_bp = index_slice["index_spread"].to_numpy(dtype=float)
    order = np.argsort(tenors)
    tenors = tenors[order]
    spreads = spreads_bp[order] / 10000.0
    index_curve = build_index_curve(
        tenors,
        spreads,
        recovery_rate,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )

    cons_df = pd.read_csv(constituents_csv, parse_dates=["Date"])
    cons_df = _normalize_columns(cons_df)
    cons_slice = cons_df[cons_df["Date"].dt.normalize() == target_date].copy()
    if cons_slice.empty:
        raise ValueError(f"No constituents rows found for date {target_date}.")
    tenor_cols = {5.0: "Spread_5Y", 7.0: "Spread_7Y", 10.0: "Spread_10Y"}
    cons_tenors = []
    cons_spreads = []
    for tenor, col in tenor_cols.items():
        if col not in cons_slice.columns:
            raise ValueError(f"Missing {col} in constituents data.")
        mean_bp = cons_slice[col].astype(float).mean()
        cons_tenors.append(tenor)
        cons_spreads.append(mean_bp / 10000.0)
    cons_curve = build_index_curve(
        cons_tenors,
        cons_spreads,
        recovery_rate,
        payment_freq=payment_freq,
        disc_curve=disc_curve,
    )

    t_max = max(
        float(np.max(ois_curve.times)),
        float(np.max(index_curve.times)),
        float(np.max(cons_curve.times)),
    )
    ts = np.linspace(0.0, max(10.0, t_max), 200)

    ois_vals = np.array([ois_curve.survival(t) for t in ts])
    index_vals = np.array([index_curve.survival(t) for t in ts])
    cons_vals = np.array([cons_curve.survival(t) for t in ts])

    if output_path is None:
        plots_dir = Path(__file__).resolve().parents[1] / "plots"
        output_path = str(plots_dir / "discount_factors.png")
    if output_table_path is None:
        plots_dir = Path(__file__).resolve().parents[1] / "plots"
        output_table_path = str(plots_dir / "discount_factors.csv")
    if output_table_txt_path is None:
        plots_dir = Path(__file__).resolve().parents[1] / "plots"
        output_table_txt_path = str(plots_dir / "discount_factors_table.txt")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, ois_vals, label="OIS discount (hazard as rate)")
    ax.plot(ts, index_vals, label="CDX index-implied DF")
    ax.plot(ts, cons_vals, label="Constituent avg-implied DF")
    ax.set_title(f"Discount Factor via survival(t) using data on {target_date.date()}")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("discount_factor(t)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)

    if print_ascii:
        width = 60
        levels = " .:-=+*#%@"
        idx = np.linspace(0, len(ts) - 1, 30, dtype=int)
        print(f"ASCII plot (t in years) for {target_date.date()}:")
        for label, series in [
            ("OIS", ois_vals),
            ("Index", index_vals),
            ("Const", cons_vals),
        ]:
            values = series[idx]
            scaled = (values - values.min()) / max(1e-12, values.max() - values.min())
            chars = "".join(levels[int(x * (len(levels) - 1))] for x in scaled)
            print(f"{label:>6}: {chars}")

    table_ts = np.array([0.5, 1, 2, 3, 5, 7, 10], dtype=float)
    table = pd.DataFrame(
        {
            "t_years": table_ts,
            "ois_df": [_df_curve(ois_curve, float(t)) for t in table_ts],
            "index_df": [_df_curve(index_curve, float(t)) for t in table_ts],
            "constituent_df": [_df_curve(cons_curve, float(t)) for t in table_ts],
        }
    )
    Path(output_table_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_table_path, index=False)
    Path(output_table_txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_table_txt_path, "w", encoding="utf-8") as fh:
        fh.write(table.to_string(index=False))
        fh.write("\n")
    if print_table:
        print(table.to_string(index=False))

    return table
