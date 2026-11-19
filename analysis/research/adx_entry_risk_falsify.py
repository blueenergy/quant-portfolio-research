"""Falsification arms for ADX entry-risk study (hipos / distrib hard filters)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.core.entry_stock_factors import (
    build_entry_factor_panel,
    exclusion_masks,
    factors_aligned_to_dataset,
    mask_dataset_scores,
)

__all__ = [
    "build_entry_factor_panel",
    "exclusion_masks",
    "factors_aligned_to_dataset",
    "mask_dataset_scores",
    "summarize_periods",
    "run_falsification_backtest",
]


def summarize_periods(periods: pd.DataFrame) -> Dict[str, Any]:
    frame = periods.copy()
    frame["year"] = frame["score_date"].astype(str).str[:4]
    yearly = (
        frame.groupby("year", sort=True)["portfolio_return_net"]
        .apply(lambda s: float((1.0 + s.astype(float)).prod() - 1.0))
        .to_dict()
    )
    rets = pd.to_numeric(frame["portfolio_return_net"], errors="coerce").dropna()
    worst = float(rets.min()) if len(rets) else None
    return {
        "cumulative": float((1.0 + rets).prod() - 1.0) if len(rets) else None,
        "yearly_returns": yearly,
        "worst_period_return": worst,
        "n_periods": int(len(frame)),
    }


def run_falsification_backtest(
    dataset: pd.DataFrame,
    score_column: str,
    masks: pd.DataFrame,
    exclude_column: str,
    *,
    db: Any,
    price_cache: Dict[str, List[Dict[str, Any]]],
    index_prices: List[Dict[str, Any]],
    top_n: int,
    horizon: int,
    transaction_cost: float,
    benchmark_symbol: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester

    masked = mask_dataset_scores(dataset, score_column, masks, exclude_column)
    config = PortfolioBacktestConfig(
        score_column=score_column,
        horizon=horizon,
        top_n_values=(top_n,),
        rebalance_interval_days=horizon,
        transaction_cost=transaction_cost,
        index_benchmark_symbol=benchmark_symbol,
        include_trades=True,
        construction_mode="industry_capped",
        max_industry_weight=0.3,
    )
    result = PortfolioBacktester(
        masked,
        config,
        index_price_history=index_prices,
        trading_calendar=[row.get("trade_date") for row in index_prices],
        price_cache=price_cache,
    ).run()
    periods = result.periods[result.periods["top_n"] == top_n].reset_index(drop=True)
    return periods, summarize_periods(periods)
