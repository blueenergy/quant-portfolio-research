"""Index buy-and-hold metrics for sweep and reporting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from analysis.core.html_format import as_float
from analysis.core.portfolio_backtester import PortfolioBacktester


def calculate_index_buy_hold_return(
    index_price_history: Sequence[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> Optional[float]:
    cleaned = PortfolioBacktester._clean_price_history(index_price_history)
    window = [row for row in cleaned if start_date <= row["trade_date"] <= end_date]
    if len(window) < 2:
        return None
    start_close = as_float(window[0].get("close"))
    end_close = as_float(window[-1].get("close"))
    if start_close is None or start_close <= 0 or end_close is None:
        return None
    return end_close / start_close - 1
