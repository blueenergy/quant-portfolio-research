"""Industry exposure diagnostics for score-ranked portfolios."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester


def build_industry_exposure(
    dataset: pd.DataFrame,
    score_column: str,
    top_n_values: Sequence[int],
    horizon: int,
    rebalance_interval_days: Optional[int],
    trading_calendar: Optional[Sequence[str]] = None,
    top_industries_per_date: int = 8,
    construction_mode: str = "top_n",
    max_industry_weight: float = 0.3,
) -> Dict[str, Any]:
    if dataset.empty:
        return {"summary": [], "periods": []}
    if score_column not in dataset.columns:
        return {"summary": [], "periods": []}

    config = PortfolioBacktestConfig(
        score_column=score_column,
        horizon=horizon,
        top_n_values=tuple(top_n_values),
        rebalance_interval_days=rebalance_interval_days,
        construction_mode=construction_mode,
        max_industry_weight=max_industry_weight,
    )
    portfolio_backtester = PortfolioBacktester(dataset, config, trading_calendar=trading_calendar)
    selected_dates = portfolio_backtester._select_rebalance_dates(
        sorted(dataset["score_date"].dropna().astype(str).unique())
    )

    period_rows: List[Dict[str, Any]] = []
    all_exposures: List[Dict[str, Any]] = []
    for score_date in selected_dates:
        group = dataset[dataset["score_date"].astype(str) == score_date]
        if "industry" not in group.columns:
            continue
        valid = group[["symbol", "industry", score_column]].dropna(subset=["symbol", score_column]).copy()
        valid["industry"] = valid["industry"].fillna("").replace("", "Unknown")
        if valid.empty:
            continue

        valid = valid.sort_values(score_column, ascending=False)
        universe_weights = _industry_weights(valid["industry"])

        for top_n in top_n_values:
            selected = portfolio_backtester._select_portfolio(valid, int(top_n))
            if selected.empty:
                continue

            selected_weights = _industry_weights(selected["industry"])
            industries = sorted(set(universe_weights).union(selected_weights))
            exposures = []
            for industry in industries:
                selected_weight = selected_weights.get(industry, 0.0)
                universe_weight = universe_weights.get(industry, 0.0)
                active_weight = selected_weight - universe_weight
                exposure = {
                    "score_date": score_date,
                    "top_n": int(top_n),
                    "industry": industry,
                    "selected_weight": selected_weight,
                    "universe_weight": universe_weight,
                    "active_weight": active_weight,
                    "selected_count": int((selected["industry"] == industry).sum()),
                    "universe_count": int((valid["industry"] == industry).sum()),
                }
                exposures.append(exposure)
                all_exposures.append(exposure)

            exposures = sorted(exposures, key=lambda row: abs(row["active_weight"]), reverse=True)
            period_rows.append(
                {
                    "score_date": score_date,
                    "top_n": int(top_n),
                    "universe_count": int(len(valid)),
                    "selected_count": int(len(selected)),
                    "construction_mode": construction_mode,
                    "industry_count": int(len(industries)),
                    "active_share": float(sum(abs(row["active_weight"]) for row in exposures) / 2),
                    "max_abs_active_weight": float(max((abs(row["active_weight"]) for row in exposures), default=0.0)),
                    "top_active_industries": exposures[: max(1, int(top_industries_per_date))],
                }
            )

    return {
        "construction_mode": construction_mode,
        "max_industry_weight": max_industry_weight,
        "summary": summarize_industry_exposure(all_exposures),
        "periods": period_rows,
    }


def summarize_industry_exposure(exposures: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not exposures:
        return []
    df = pd.DataFrame(exposures)
    rows = []
    for (top_n, industry), group in df.groupby(["top_n", "industry"]):
        rows.append(
            {
                "top_n": int(top_n),
                "industry": industry,
                "avg_selected_weight": float(group["selected_weight"].mean()),
                "avg_universe_weight": float(group["universe_weight"].mean()),
                "avg_active_weight": float(group["active_weight"].mean()),
                "avg_abs_active_weight": float(group["active_weight"].abs().mean()),
                "max_abs_active_weight": float(group["active_weight"].abs().max()),
                "observations": int(len(group)),
            }
        )
    return sorted(rows, key=lambda row: (row["top_n"], -row["avg_abs_active_weight"], row["industry"]))


def _industry_weights(industry_series: pd.Series) -> Dict[str, float]:
    counts = industry_series.value_counts(dropna=False)
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {str(industry): float(count / total) for industry, count in counts.items()}
