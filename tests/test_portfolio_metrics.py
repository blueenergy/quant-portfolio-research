"""Portfolio summary risk metrics beyond Sharpe."""

import pandas as pd

from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester


def test_summary_includes_sortino_calmar_and_information_ratio():
    periods = pd.DataFrame(
        {
            "score_date": ["20240102", "20240202", "20240304", "20240403"],
            "portfolio_return_net": [0.10, -0.05, 0.02, 0.03],
            "benchmark_return": [0.05, -0.02, 0.01, 0.00],
            "excess_return": [0.05, -0.03, 0.01, 0.03],
            "index_benchmark_return": [0.04, -0.03, 0.01, 0.00],
            "index_excess_return": [0.06, -0.02, 0.01, 0.03],
            "turnover": [0.1, 0.1, 0.1, 0.1],
            "selected_count": [2, 2, 2, 2],
            "portfolio_exposure": [1.0, 1.0, 1.0, 1.0],
        }
    )
    backtester = PortfolioBacktester(
        pd.DataFrame(),
        PortfolioBacktestConfig(
            score_column="score",
            horizon=20,
            top_n_values=(2,),
            rebalance_interval_days=20,
        ),
    )

    summary = backtester._summarize_periods(periods, 2)

    assert summary["sharpe"] is not None
    assert summary["sortino"] is not None
    assert summary["calmar"] is not None
    assert summary["information_ratio"] is not None
    assert summary["index_information_ratio"] is not None
    assert summary["calmar"] == summary["annualized_return"] / abs(summary["max_drawdown"])
