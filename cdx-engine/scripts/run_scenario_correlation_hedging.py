from __future__ import annotations

"""Scenario-Based Correlation Hedging Diagnostics.

This script evaluates correlation hedging under deterministic PCA factor scenarios,
not historical out-of-sample backtest performance.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario-Based Correlation Hedging Diagnostics")
    p.add_argument("--input", type=str, default="outputs/run_factor_risk_decomposition/data/factor_exposures_by_tranche.csv")
    p.add_argument("--factor-scores", type=str, default="outputs/run_factor_risk_decomposition/data/factor_scores.csv")
    p.add_argument("--hedges", type=str, default="5Y|3-7,5Y|7-10,10Y|10-15")
    p.add_argument("--dynamic-hedge-basket", type=str, default="true")

    p.add_argument("--ridge-lambda", type=float, default=1e-4)
    p.add_argument("--max-abs-weight", type=float, default=10.0)

    p.add_argument("--include-half-sigma", type=str, default="false")
    p.add_argument("--include-two-sigma", type=str, default="false")

    p.add_argument("--outdir", type=str, default="outputs/run_scenario_correlation_hedging")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s: %(message)s")


def _canonical_tranche_label(raw: str) -> str:
    text = str(raw).strip().upper().replace(" ", "")
    if "|" in text:
        return text
    if text.count("Y") >= 1 and "-" in text:
        i = text.find("Y")
        return f"{text[: i + 1]}|{text[i + 1 :]}"
    return text


def _parse_tranche_label(lbl: str) -> tuple[float, float, float]:
    tok = _canonical_tranche_label(lbl)
    if "|" not in tok:
        return (np.nan, np.nan, np.nan)
    t, tr = tok.split("|", 1)
    if not t.endswith("Y") or "-" not in tr:
        return (np.nan, np.nan, np.nan)
    lo, hi = tr.split("-", 1)
    return (float(t[:-1]), float(lo), float(hi))


def _tenor_bucket(lbl: str) -> str:
    ten, _, _ = _parse_tranche_label(lbl)
    if ten in (1.0, 2.0):
        return "short"
    if ten in (3.0, 5.0):
        return "intermediate"
    if ten in (7.0, 10.0):
        return "long"
    return "other"


def _tranche_bucket(lbl: str) -> str:
    _, lo, hi = _parse_tranche_label(lbl)
    pair = (lo, hi)
    if pair == (0.0, 3.0):
        return "equity"
    if pair == (3.0, 7.0):
        return "lower_mezz"
    if pair == (7.0, 10.0):
        return "upper_mezz"
    if pair == (10.0, 15.0):
        return "senior"
    return "other"


def _format_exposure_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if {"tranche", "pc1_exposure", "pc2_exposure", "pc3_exposure"}.issubset(cols):
        out = df[["tranche", "pc1_exposure", "pc2_exposure", "pc3_exposure"]].copy()
        out["tranche"] = out["tranche"].map(_canonical_tranche_label)
        return out

    need = {"tranche_tenor", "tranche_name", "pc1_exposure", "pc2_exposure", "pc3_exposure"}
    if need.issubset(cols):
        out = df[["tranche_tenor", "tranche_name", "pc1_exposure", "pc2_exposure", "pc3_exposure"]].copy()
        out["tranche"] = out.apply(lambda r: f"{float(r['tranche_tenor']):.0f}Y|{r['tranche_name']}", axis=1)
        out = out[["tranche", "pc1_exposure", "pc2_exposure", "pc3_exposure"]]
        out["tranche"] = out["tranche"].map(_canonical_tranche_label)
        return out

    raise ValueError("Exposure CSV schema is invalid.")


def load_factor_exposures(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    out = _format_exposure_table(pd.read_csv(path))
    for c in ("pc1_exposure", "pc2_exposure", "pc3_exposure"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().copy()
    if out.empty:
        raise ValueError("No valid exposures after cleaning.")
    return out.sort_values("tranche").drop_duplicates(subset=["tranche"], keep="last")


def parse_hedges(text: str) -> list[str]:
    hs = [_canonical_tranche_label(x) for x in text.split(",") if x.strip()]
    if not hs:
        raise ValueError("No hedge instruments provided.")
    return hs


def estimate_sigma_from_scores(path: Path) -> tuple[dict[str, float], str]:
    if not path.exists():
        return {"pc1": 1.0, "pc2": 1.0, "pc3": 1.0}, "unit_shocks"

    df = pd.read_csv(path)
    req = {"pc1_score", "pc2_score", "pc3_score"}
    if not req.issubset(df.columns):
        return {"pc1": 1.0, "pc2": 1.0, "pc3": 1.0}, "unit_shocks"

    sig = {}
    for c, k in [("pc1_score", "pc1"), ("pc2_score", "pc2"), ("pc3_score", "pc3")]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        v = float(s.std(ddof=1)) if len(s) > 1 else 1.0
        sig[k] = v if np.isfinite(v) and v > 0 else 1.0
    return sig, "historical_factor_scores"


def build_scenarios(sigmas: dict[str, float], include_half: bool, include_two: bool) -> pd.DataFrame:
    s1, s2, s3 = sigmas["pc1"], sigmas["pc2"], sigmas["pc3"]
    rows = [
        ("PC1_up_1sigma", +s1, 0.0, 0.0),
        ("PC1_dn_1sigma", -s1, 0.0, 0.0),
        ("PC2_up_1sigma", 0.0, +s2, 0.0),
        ("PC2_dn_1sigma", 0.0, -s2, 0.0),
        ("PC3_up_1sigma", 0.0, 0.0, +s3),
        ("PC3_dn_1sigma", 0.0, 0.0, -s3),
        ("PC1_up_PC2_up", +s1, +s2, 0.0),
        ("PC1_up_PC3_up", +s1, 0.0, +s3),
        ("PC2_up_PC3_up", 0.0, +s2, +s3),
        ("PC1_dn_PC2_dn", -s1, -s2, 0.0),
    ]
    if include_half:
        rows += [
            ("PC1_up_0p5sigma", +0.5 * s1, 0.0, 0.0),
            ("PC2_up_0p5sigma", 0.0, +0.5 * s2, 0.0),
            ("PC3_up_0p5sigma", 0.0, 0.0, +0.5 * s3),
        ]
    if include_two:
        rows += [
            ("PC1_up_2sigma", +2.0 * s1, 0.0, 0.0),
            ("PC1_dn_2sigma", -2.0 * s1, 0.0, 0.0),
            ("PC2_up_2sigma", 0.0, +2.0 * s2, 0.0),
            ("PC2_dn_2sigma", 0.0, -2.0 * s2, 0.0),
            ("PC3_up_2sigma", 0.0, 0.0, +2.0 * s3),
            ("PC3_dn_2sigma", 0.0, 0.0, -2.0 * s3),
        ]

    sc = pd.DataFrame(rows, columns=["scenario", "dpc1", "dpc2", "dpc3"]).drop_duplicates(subset=["scenario"]).reset_index(drop=True)
    return sc


def compute_unhedged_pnl(exposure_df: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    e = exposure_df.set_index("tranche")[["pc1_exposure", "pc2_exposure", "pc3_exposure"]]
    f = scenarios[["dpc1", "dpc2", "dpc3"]].to_numpy(dtype=float)
    pnl = e.to_numpy(dtype=float) @ f.T
    out = (
        pd.DataFrame(pnl, index=e.index, columns=scenarios["scenario"])
        .stack()
        .rename("pnl_unhedged")
        .reset_index()
        .rename(columns={"level_0": "tranche", "level_1": "scenario"})
    )
    return out


def solve_factor_match(target_b: np.ndarray, h: np.ndarray, ridge_lambda: float, max_abs_weight: float) -> np.ndarray:
    # solve H^T w ≈ -b  => minimize ||H^T w + b||^2 + lam||w||^2
    if h.shape[0] == 0:
        return np.zeros(0)
    a = h @ h.T + max(ridge_lambda, 0.0) * np.eye(h.shape[0])
    rhs = -(h @ target_b)
    w = np.linalg.lstsq(a, rhs, rcond=None)[0]
    if max_abs_weight > 0:
        w = np.clip(w, -abs(max_abs_weight), abs(max_abs_weight))
    return w


def solve_scenario_opt(target_pnl: np.ndarray, hedge_pnl: np.ndarray, ridge_lambda: float, max_abs_weight: float) -> np.ndarray:
    # min ||target + hedge @ w||^2 + lam||w||^2
    if hedge_pnl.shape[1] == 0:
        return np.zeros(0)
    a = hedge_pnl.T @ hedge_pnl + max(ridge_lambda, 0.0) * np.eye(hedge_pnl.shape[1])
    rhs = -(hedge_pnl.T @ target_pnl)
    w = np.linalg.lstsq(a, rhs, rcond=None)[0]
    if max_abs_weight > 0:
        w = np.clip(w, -abs(max_abs_weight), abs(max_abs_weight))
    return w


def select_dynamic_basket(target: str, all_tranches: list[str], liquid_pref: list[str], k: int = 3) -> list[str]:
    t_ten, t_lo, t_hi = _parse_tranche_label(target)
    t_ctr = 0.5 * (t_lo + t_hi) if np.isfinite(t_lo) else np.nan
    cands = []
    for tr in all_tranches:
        if tr == target:
            continue
        ten, lo, hi = _parse_tranche_label(tr)
        ctr = 0.5 * (lo + hi) if np.isfinite(lo) else np.nan
        if not (np.isfinite(t_ten) and np.isfinite(t_ctr) and np.isfinite(ten) and np.isfinite(ctr)):
            dist = 1e6
        else:
            dist = abs(ten - t_ten) + 10.0 * abs(ctr - t_ctr)
        liq_bonus = -0.35 if tr in liquid_pref else 0.0
        cands.append((dist + liq_bonus, tr))
    cands.sort(key=lambda x: x[0])
    return [x[1] for x in cands[:k]]


def evaluate_hedges(
    exposure_df: pd.DataFrame,
    scenarios: pd.DataFrame,
    hedge_list: list[str],
    dynamic_basket: bool,
    ridge_lambda: float,
    max_abs_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tranches = exposure_df["tranche"].tolist()
    e_map = exposure_df.set_index("tranche")[["pc1_exposure", "pc2_exposure", "pc3_exposure"]]

    f = scenarios[["dpc1", "dpc2", "dpc3"]].to_numpy(dtype=float)

    w_rows_factor = []
    w_rows_scen = []
    hedged_rows_factor = []
    hedged_rows_scen = []
    summary_rows = []

    for tr in tranches:
        b = e_map.loc[tr].to_numpy(dtype=float)
        if dynamic_basket:
            basket = select_dynamic_basket(tr, tranches, hedge_list, 3)
        else:
            basket = [h for h in hedge_list if h != tr and h in tranches]
        basket = basket[:3]

        h = e_map.loc[basket].to_numpy(dtype=float) if basket else np.zeros((0, 3), dtype=float)

        target_pnl = f @ b
        hedge_pnl_mat = np.column_stack([f @ e_map.loc[hg].to_numpy(dtype=float) for hg in basket]) if basket else np.zeros((len(scenarios), 0))

        w_factor = solve_factor_match(b, h, ridge_lambda, max_abs_weight)
        w_scen = solve_scenario_opt(target_pnl, hedge_pnl_mat, ridge_lambda, max_abs_weight)

        pnl_factor = target_pnl + (hedge_pnl_mat @ w_factor if hedge_pnl_mat.shape[1] else 0.0)
        pnl_scen = target_pnl + (hedge_pnl_mat @ w_scen if hedge_pnl_mat.shape[1] else 0.0)

        t1 = basket[0] if len(basket) > 0 else ""
        t2 = basket[1] if len(basket) > 1 else ""
        t3 = basket[2] if len(basket) > 2 else ""

        w_rows_factor.append({"tranche": tr, "hedge_T1": t1, "hedge_T2": t2, "hedge_T3": t3, "hedge_w_T1": w_factor[0] if len(w_factor) > 0 else np.nan, "hedge_w_T2": w_factor[1] if len(w_factor) > 1 else np.nan, "hedge_w_T3": w_factor[2] if len(w_factor) > 2 else np.nan})
        w_rows_scen.append({"tranche": tr, "hedge_T1": t1, "hedge_T2": t2, "hedge_T3": t3, "hedge_w_T1": w_scen[0] if len(w_scen) > 0 else np.nan, "hedge_w_T2": w_scen[1] if len(w_scen) > 1 else np.nan, "hedge_w_T3": w_scen[2] if len(w_scen) > 2 else np.nan})

        for i, s in enumerate(scenarios["scenario"].tolist()):
            hedged_rows_factor.append({"tranche": tr, "scenario": s, "pnl_unhedged": target_pnl[i], "pnl_hedged": pnl_factor[i], "residual_pnl": pnl_factor[i]})
            hedged_rows_scen.append({"tranche": tr, "scenario": s, "pnl_unhedged": target_pnl[i], "pnl_hedged": pnl_scen[i], "residual_pnl": pnl_scen[i]})

        for method, pnl_h, w in [("factor_match", pnl_factor, w_factor), ("scenario_opt", pnl_scen, w_scen)]:
            vol_u = float(np.std(target_pnl, ddof=1)) if len(target_pnl) > 1 else np.nan
            vol_h = float(np.std(pnl_h, ddof=1)) if len(pnl_h) > 1 else np.nan
            vol_red = 1.0 - (vol_h / vol_u) if np.isfinite(vol_u) and vol_u > 0 else np.nan
            rmse = float(np.sqrt(np.mean(np.square(pnl_h))))
            mae = float(np.mean(np.abs(pnl_h)))
            worst_idx = int(np.argmax(np.abs(pnl_h)))
            summary_rows.append(
                {
                    "tranche": tr,
                    "hedge_method": method,
                    "scenario_vol_unhedged": vol_u,
                    "scenario_vol_hedged": vol_h,
                    "scenario_vol_reduction": vol_red,
                    "residual_rmse": rmse,
                    "residual_mean_abs_pnl": mae,
                    "worst_case_abs_residual": float(np.abs(pnl_h[worst_idx])),
                    "worst_case_scenario": scenarios.iloc[worst_idx]["scenario"],
                    "hedge_w_T1": w[0] if len(w) > 0 else np.nan,
                    "hedge_w_T2": w[1] if len(w) > 1 else np.nan,
                    "hedge_w_T3": w[2] if len(w) > 2 else np.nan,
                }
            )

    return (
        pd.DataFrame(w_rows_factor).sort_values("tranche"),
        pd.DataFrame(w_rows_scen).sort_values("tranche"),
        pd.DataFrame(hedged_rows_factor).sort_values(["tranche", "scenario"]),
        pd.DataFrame(hedged_rows_scen).sort_values(["tranche", "scenario"]),
        pd.DataFrame(summary_rows).sort_values(["hedge_method", "tranche"]),
    )


def build_reporting_outputs(
    summary_df: pd.DataFrame,
    hedged_factor: pd.DataFrame,
    hedged_scenario_opt: pd.DataFrame,
    data_dir: Path,
    near_zero_threshold: float = 1e-8,
) -> pd.DataFrame:
    """Create compact/ranked/aggregated reporting CSV layers."""
    out = summary_df.copy()
    out["near_zero_unhedged_flag"] = pd.to_numeric(out["scenario_vol_unhedged"], errors="coerce") < near_zero_threshold

    compact_cols = [
        "tranche",
        "hedge_method",
        "scenario_vol_reduction",
        "residual_rmse",
        "residual_mean_abs_pnl",
        "worst_case_abs_residual",
        "worst_case_scenario",
    ]
    data_dir.mkdir(parents=True, exist_ok=True)
    out[compact_cols].to_csv(data_dir / "scenario_hedging_summary_compact.csv", index=False)

    near_zero = out[out["near_zero_unhedged_flag"]].sort_values(["hedge_method", "tranche"])
    near_zero.to_csv(data_dir / "near_zero_unhedged_tranches.csv", index=False)

    valid = out[~out["near_zero_unhedged_flag"]].copy()

    def _top_bottom(df: pd.DataFrame, col: str, higher_is_better: bool) -> pd.DataFrame:
        blocks = []
        for method, g in df.groupby("hedge_method", sort=True):
            g = g.copy()
            g[col] = pd.to_numeric(g[col], errors="coerce")
            g = g.dropna(subset=[col])
            if g.empty:
                continue
            order = g.sort_values(col, ascending=not higher_is_better)
            top = order.head(min(10, len(order))).copy()
            top["rank_group"] = "top"
            bot = order.tail(min(10, len(order))).copy()
            bot["rank_group"] = "bottom"
            blocks.append(pd.concat([top, bot], axis=0))
        if not blocks:
            return pd.DataFrame(columns=["tranche", "hedge_method", col, "rank_group"])
        return pd.concat(blocks, axis=0, ignore_index=True)

    _top_bottom(valid, "scenario_vol_reduction", higher_is_better=True).to_csv(data_dir / "top_bottom_by_vol_reduction.csv", index=False)
    _top_bottom(valid, "residual_rmse", higher_is_better=False).to_csv(data_dir / "top_bottom_by_residual_rmse.csv", index=False)

    for c in ("scenario_vol_reduction", "residual_rmse", "worst_case_abs_residual"):
        valid[c] = pd.to_numeric(valid[c], errors="coerce")
    valid["tenor_bucket"] = valid["tranche"].map(_tenor_bucket)
    valid["tranche_bucket"] = valid["tranche"].map(_tranche_bucket)

    tenor_summary = (
        valid.groupby(["hedge_method", "tenor_bucket"], dropna=False)
        .agg(
            average_scenario_vol_reduction=("scenario_vol_reduction", "mean"),
            median_scenario_vol_reduction=("scenario_vol_reduction", "median"),
            average_residual_rmse=("residual_rmse", "mean"),
            worst_case_residual_mean=("worst_case_abs_residual", "mean"),
        )
        .reset_index()
        .sort_values(["hedge_method", "tenor_bucket"])
    )
    tenor_summary.to_csv(data_dir / "summary_by_tenor_bucket.csv", index=False)

    tranche_summary = (
        valid.groupby(["hedge_method", "tranche_bucket"], dropna=False)
        .agg(
            average_scenario_vol_reduction=("scenario_vol_reduction", "mean"),
            median_scenario_vol_reduction=("scenario_vol_reduction", "median"),
            average_residual_rmse=("residual_rmse", "mean"),
            worst_case_residual_mean=("worst_case_abs_residual", "mean"),
        )
        .reset_index()
        .sort_values(["hedge_method", "tranche_bucket"])
    )
    tranche_summary.to_csv(data_dir / "summary_by_tranche_bucket.csv", index=False)

    out.to_csv(data_dir / "scenario_hedging_summary.csv", index=False)
    return out


def make_plots(
    summary_df: pd.DataFrame,
    hedged_factor: pd.DataFrame,
    hedged_scenario_opt: pd.DataFrame,
    outdir: Path,
) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    valid_tranches = set(
        summary_df[
            (summary_df["hedge_method"] == "scenario_opt")
            & (~summary_df["near_zero_unhedged_flag"])
        ]["tranche"].tolist()
    )

    # Plot A: method comparison scatter on vol reduction
    comp = summary_df.pivot(index="tranche", columns="hedge_method", values="scenario_vol_reduction")
    if {"factor_match", "scenario_opt"}.issubset(comp.columns):
        comp = comp.loc[comp.index.intersection(list(valid_tranches))].dropna(subset=["factor_match", "scenario_opt"])
        if not comp.empty:
            fig, ax = plt.subplots(figsize=(7.2, 6.2))
            ax.scatter(comp["factor_match"], comp["scenario_opt"], s=46, alpha=0.9, color="#21618c")
            lo = float(np.nanmin(comp[["factor_match", "scenario_opt"]].to_numpy()))
            hi = float(np.nanmax(comp[["factor_match", "scenario_opt"]].to_numpy()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0)
            delta = (comp["scenario_opt"] - comp["factor_match"]).sort_values(ascending=False).head(5)
            for tr, _ in delta.items():
                ax.annotate(tr, (comp.loc[tr, "factor_match"], comp.loc[tr, "scenario_opt"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Factor-Match Scenario Vol Reduction")
            ax.set_ylabel("Scenario-Opt Scenario Vol Reduction")
            ax.set_title("Scenario-Optimal vs Factor-Matching Hedge")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "vol_reduction_method_comparison.png", dpi=300)
            plt.close(fig)

            # Plot B: improvement bars
            d = (comp["scenario_opt"] - comp["factor_match"]).sort_values()
            fig, ax = plt.subplots(figsize=(12, 5.8))
            colors = ["#c0392b" if v < 0 else "#1f77b4" for v in d.to_numpy()]
            x = np.arange(len(d))
            ax.bar(x, d.to_numpy(), color=colors, alpha=0.9)
            ax.axhline(0.0, color="black", linewidth=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(d.index.tolist(), rotation=45, ha="right")
            ax.set_ylabel("Delta Vol Reduction (Scenario-Opt - Factor-Match)")
            ax.set_title("Scenario-Optimal vs Factor-Matching Hedge: Vol Reduction Improvement")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "vol_reduction_improvement.png", dpi=300)
            plt.close(fig)

    # Plot C: scenario-opt RMSE ranked, highlight best/worst 5
    so = summary_df[(summary_df["hedge_method"] == "scenario_opt") & (~summary_df["near_zero_unhedged_flag"])].copy()
    so["residual_rmse"] = pd.to_numeric(so["residual_rmse"], errors="coerce")
    so = so.dropna(subset=["residual_rmse"]).sort_values("residual_rmse")
    if not so.empty:
        n = len(so)
        best_idx = set(so.head(min(5, n)).index.tolist())
        worst_idx = set(so.tail(min(5, n)).index.tolist())
        colors = []
        for idx in so.index:
            if idx in best_idx:
                colors.append("#1f77b4")
            elif idx in worst_idx:
                colors.append("#c0392b")
            else:
                colors.append("#95a5a6")
        fig, ax = plt.subplots(figsize=(12, 5.8))
        x = np.arange(n)
        ax.bar(x, so["residual_rmse"].to_numpy(), color=colors, alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(so["tranche"].tolist(), rotation=45, ha="right")
        ax.set_ylabel("Residual RMSE")
        ax.set_title("Residual RMSE by Tranche")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "scenario_opt_residual_rmse_ranked.png", dpi=300)
        plt.close(fig)

    # Plot D/E: residual heatmaps by method (exclude near-zero tranches)
    for method, src, fname in [
        ("scenario_opt", hedged_scenario_opt, "scenario_opt_residual_heatmap.png"),
        ("factor_match", hedged_factor, "factor_match_residual_heatmap.png"),
    ]:
        use = src[src["tranche"].isin(valid_tranches)].copy()
        heat = use.pivot(index="tranche", columns="scenario", values="residual_pnl")
        if heat.empty:
            continue
        arr = heat.to_numpy(dtype=float)
        vmax = max(float(np.nanpercentile(np.abs(arr), 95)), 1e-10)
        fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(heat))))
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns), rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(list(heat.index))
        ax.set_title(f"Residual PnL Heatmap under PCA Shock Scenarios ({method})")
        fig.colorbar(im, ax=ax, shrink=0.85, label="Residual Hedged PnL")
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=300)
        plt.close(fig)


def print_summary(summary_df: pd.DataFrame, n_scenarios: int, sigma_source: str) -> None:
    valid = summary_df[~summary_df["near_zero_unhedged_flag"]].copy()
    piv_vol = valid.pivot(index="tranche", columns="hedge_method", values="scenario_vol_reduction")
    piv_rmse = valid.pivot(index="tranche", columns="hedge_method", values="residual_rmse")

    print("Scenario-Based Correlation Hedging Diagnostics")
    print(f"- number of scenarios: {n_scenarios}")
    print(f"- sigma source: {sigma_source}")

    for method in ("factor_match", "scenario_opt"):
        sub = valid[valid["hedge_method"] == method].copy()
        vol = pd.to_numeric(sub["scenario_vol_reduction"], errors="coerce").dropna()
        print(
            f"- {method} mean / median vol reduction: "
            f"{(float(vol.mean()) if not vol.empty else np.nan):.6f} / {(float(vol.median()) if not vol.empty else np.nan):.6f}"
        )

    improved = np.nan
    best_imp = None
    if {"factor_match", "scenario_opt"}.issubset(piv_vol.columns):
        delta = (piv_vol["scenario_opt"] - piv_vol["factor_match"]).dropna()
        improved = int((delta > 0).sum())
        if not delta.empty:
            best_imp = delta.sort_values(ascending=False).iloc[0]
            best_tr = delta.sort_values(ascending=False).index[0]
            print(f"- number of tranches improved under scenario_opt vs factor_match: {improved}")
            print(f"- best tranche improvement: {best_tr} ({best_imp:.6f})")

    so = valid[valid["hedge_method"] == "scenario_opt"].copy()
    so["residual_rmse"] = pd.to_numeric(so["residual_rmse"], errors="coerce")
    so = so.dropna(subset=["residual_rmse"])
    if not so.empty:
        worst_rmse = so.sort_values("residual_rmse", ascending=False).iloc[0]
        print(f"- worst residual RMSE under scenario_opt: {worst_rmse['tranche']} ({worst_rmse['residual_rmse']:.6f})")
        wc_mode = so["worst_case_scenario"].value_counts(dropna=True)
        if not wc_mode.empty:
            print(f"- most frequent worst-case scenario: {wc_mode.index[0]} ({int(wc_mode.iloc[0])} tranches)")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = outdir / "data"
    plot_dir = outdir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_df = load_factor_exposures(ROOT / args.input)
    hedge_list = parse_hedges(args.hedges)

    sigmas, sigma_source = estimate_sigma_from_scores(ROOT / args.factor_scores)
    scenarios = build_scenarios(
        sigmas=sigmas,
        include_half=_bool(args.include_half_sigma),
        include_two=_bool(args.include_two_sigma),
    )

    unhedged = compute_unhedged_pnl(exp_df, scenarios)

    w_factor, w_scen, hedged_factor, hedged_scen, summary = evaluate_hedges(
        exposure_df=exp_df,
        scenarios=scenarios,
        hedge_list=hedge_list,
        dynamic_basket=_bool(args.dynamic_hedge_basket),
        ridge_lambda=args.ridge_lambda,
        max_abs_weight=args.max_abs_weight,
    )

    # Save core outputs (raw appendix tables)
    scenarios.to_csv(data_dir / "factor_scenarios.csv", index=False)
    unhedged.to_csv(data_dir / "scenario_unhedged_pnl.csv", index=False)
    w_factor.to_csv(data_dir / "hedge_weights_factor_match.csv", index=False)
    w_scen.to_csv(data_dir / "hedge_weights_scenario_opt.csv", index=False)
    hedged_factor.to_csv(data_dir / "scenario_hedged_pnl_factor_match.csv", index=False)
    hedged_scen.to_csv(data_dir / "scenario_hedged_pnl_scenario_opt.csv", index=False)
    summary.to_csv(data_dir / "scenario_hedging_summary.csv", index=False)

    summary_enriched = build_reporting_outputs(
        summary_df=summary,
        hedged_factor=hedged_factor,
        hedged_scenario_opt=hedged_scen,
        data_dir=data_dir,
    )
    make_plots(summary_enriched, hedged_factor, hedged_scen, outdir)

    logging.info("Scenario-Based Correlation Hedging Diagnostics outputs saved under %s", outdir)
    print_summary(summary_enriched, n_scenarios=len(scenarios), sigma_source=sigma_source)


if __name__ == "__main__":
    main()
