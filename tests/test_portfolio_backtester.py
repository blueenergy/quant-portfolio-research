import pandas as pd
import pytest

from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
from analysis.core.trade_fee_model import TradeFeeModel


def _sample_dataset():
    return pd.DataFrame(
        [
            {"score_date": "20250101", "symbol": "A", "composite_cycle_oriented_score": 90, "forward_return_20d": 0.10},
            {"score_date": "20250101", "symbol": "B", "composite_cycle_oriented_score": 80, "forward_return_20d": 0.05},
            {"score_date": "20250101", "symbol": "C", "composite_cycle_oriented_score": 70, "forward_return_20d": -0.03},
            {"score_date": "20250102", "symbol": "A", "composite_cycle_oriented_score": 70, "forward_return_20d": -0.01},
            {"score_date": "20250102", "symbol": "B", "composite_cycle_oriented_score": 95, "forward_return_20d": 0.02},
            {"score_date": "20250102", "symbol": "C", "composite_cycle_oriented_score": 85, "forward_return_20d": 0.08},
        ]
    )


def _sample_dataset_with_prices():
    df = _sample_dataset().copy()
    df["name"] = df["symbol"].map({"A": "Alpha", "B": "Beta", "C": "Gamma"})
    df["entry_date"] = df["score_date"]
    df["entry_price"] = 100.0
    df["exit_date_20d"] = df["score_date"].map({"20250101": "20250121", "20250102": "20250122"})
    df["exit_price_20d"] = df["entry_price"] * (1 + df["forward_return_20d"])
    return df


def _industry_dataset():
    return pd.DataFrame(
        [
            {"score_date": "20250101", "symbol": "A", "industry": "Tech", "composite_cycle_oriented_score": 100, "forward_return_20d": 0.10},
            {"score_date": "20250101", "symbol": "B", "industry": "Tech", "composite_cycle_oriented_score": 95, "forward_return_20d": 0.09},
            {"score_date": "20250101", "symbol": "C", "industry": "Tech", "composite_cycle_oriented_score": 90, "forward_return_20d": 0.08},
            {"score_date": "20250101", "symbol": "D", "industry": "Bank", "composite_cycle_oriented_score": 80, "forward_return_20d": 0.07},
            {"score_date": "20250101", "symbol": "E", "industry": "Bank", "composite_cycle_oriented_score": 70, "forward_return_20d": 0.06},
            {"score_date": "20250101", "symbol": "F", "industry": "Drug", "composite_cycle_oriented_score": 60, "forward_return_20d": 0.05},
        ]
    )


def test_portfolio_backtester_selects_top_n_and_summarizes_returns():
    config = PortfolioBacktestConfig(top_n_values=(2,), rebalance_interval_days=None, rebalance_step=1, transaction_cost=0)
    result = PortfolioBacktester(_sample_dataset(), config).run()

    periods = result.periods
    summary = result.summary.iloc[0]

    assert len(periods) == 2
    assert round(periods.iloc[0]["portfolio_return"], 4) == 0.075
    assert round(periods.iloc[0]["benchmark_return"], 4) == 0.04
    assert round(periods.iloc[1]["portfolio_return"], 4) == 0.05
    assert round(summary["cumulative_return"], 4) == round((1.075 * 1.05) - 1, 4)
    assert summary["periods"] == 2
    assert summary["top_n"] == 2
    assert round(summary["yearly_returns"]["2025"], 4) == round((1.075 * 1.05) - 1, 4)
    assert summary["annualized_return"] is None  # span < 1 month


def test_portfolio_backtester_annualized_and_yearly_returns():
    from analysis.core.portfolio_backtester import (
        PortfolioBacktester,
        _cagr,
        _span_years_from_score_dates,
        _yearly_compound_returns,
        _yearly_index_excess,
    )

    periods = pd.DataFrame(
        [
            {
                "score_date": "20220115",
                "portfolio_return_net": 0.10,
                "benchmark_return": 0.08,
                "excess_return": 0.02,
                "index_benchmark_return": 0.05,
                "turnover": 1.0,
                "selected_count": 2,
            },
            {
                "score_date": "20230115",
                "portfolio_return_net": 0.20,
                "benchmark_return": 0.15,
                "excess_return": 0.05,
                "index_benchmark_return": 0.10,
                "turnover": 0.5,
                "selected_count": 2,
            },
            {
                "score_date": "20240115",
                "portfolio_return_net": -0.05,
                "benchmark_return": -0.03,
                "excess_return": -0.02,
                "index_benchmark_return": -0.02,
                "turnover": 0.5,
                "selected_count": 2,
            },
        ]
    )

    span_years = _span_years_from_score_dates(periods["score_date"].tolist())
    assert span_years is not None
    assert span_years > 1.9

    yearly = _yearly_compound_returns(periods, "portfolio_return_net")
    assert round(yearly["2022"], 4) == 0.10
    assert round(yearly["2023"], 4) == 0.20
    assert round(yearly["2024"], 4) == -0.05

    yearly_excess = _yearly_index_excess(periods)
    assert round(yearly_excess["2022"], 4) == round(1.10 - 1.05, 4)
    assert round(yearly_excess["2023"], 4) == round(1.20 - 1.10, 4)
    assert round(yearly_excess["2024"], 4) == round(0.95 - 0.98, 4)

    cumulative = float((1 + periods["portfolio_return_net"]).prod() - 1)
    annualized = _cagr(cumulative, span_years)
    assert annualized is not None
    assert annualized == (1 + cumulative) ** (1 / span_years) - 1

    config = PortfolioBacktestConfig(top_n_values=(2,), rebalance_interval_days=None, rebalance_step=1, transaction_cost=0)
    backtester = PortfolioBacktester(_sample_dataset(), config)
    summary = backtester._summarize_periods(periods, top_n=2)

    assert summary["yearly_returns"] == yearly
    assert summary["yearly_index_excess"] == yearly_excess
    assert round(summary["annualized_return"], 6) == round(annualized, 6)
    assert round(summary["span_years"], 4) == round(span_years, 4)

def test_portfolio_backtester_calculates_turnover_and_costs():
    config = PortfolioBacktestConfig(top_n_values=(2,), rebalance_interval_days=None, rebalance_step=1, transaction_cost=0.01)
    result = PortfolioBacktester(_sample_dataset(), config).run()

    periods = result.periods

    assert periods.iloc[0]["turnover"] == 1.0
    assert periods.iloc[1]["turnover"] == 0.5
    assert round(periods.iloc[0]["portfolio_return_net"], 4) == 0.065
    assert round(periods.iloc[1]["portfolio_return_net"], 4) == 0.045


def test_portfolio_backtester_supports_multiple_top_n_values():
    config = PortfolioBacktestConfig(top_n_values=(1, 2), rebalance_interval_days=None, rebalance_step=1, transaction_cost=0)
    result = PortfolioBacktester(_sample_dataset(), config).run()

    assert set(result.summary["top_n"]) == {1, 2}
    assert set(result.periods["top_n"]) == {1, 2}


def test_portfolio_backtester_returns_empty_for_missing_score_column():
    config = PortfolioBacktestConfig(score_column="missing_score", top_n_values=(2,))
    result = PortfolioBacktester(_sample_dataset(), config).run()

    assert result.periods.empty
    assert result.summary.empty


def test_portfolio_backtester_adds_index_benchmark_returns():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
    )
    index_price_history = [
        {"trade_date": "20250101", "close": 1000},
        {"trade_date": "20250102", "close": 1010},
        {"trade_date": "20250103", "close": 1020},
        {"trade_date": "20250104", "close": 1040},
        {"trade_date": "20250105", "close": 1050},
        {"trade_date": "20250106", "close": 1060},
        {"trade_date": "20250107", "close": 1080},
        {"trade_date": "20250108", "close": 1090},
        {"trade_date": "20250109", "close": 1100},
        {"trade_date": "20250110", "close": 1110},
        {"trade_date": "20250111", "close": 1120},
        {"trade_date": "20250112", "close": 1130},
        {"trade_date": "20250113", "close": 1140},
        {"trade_date": "20250114", "close": 1150},
        {"trade_date": "20250115", "close": 1160},
        {"trade_date": "20250116", "close": 1170},
        {"trade_date": "20250117", "close": 1180},
        {"trade_date": "20250118", "close": 1190},
        {"trade_date": "20250119", "close": 1200},
        {"trade_date": "20250120", "close": 1210},
        {"trade_date": "20250121", "close": 1220},
        {"trade_date": "20250122", "close": 1230},
    ]

    result = PortfolioBacktester(_sample_dataset(), config, index_price_history=index_price_history).run()

    periods = result.periods
    summary = result.summary.iloc[0]
    assert "index_benchmark_return" in periods.columns
    assert round(periods.iloc[0]["index_benchmark_return"], 4) == 0.22
    assert round(periods.iloc[0]["index_excess_return"], 4) == round(0.075 - 0.22, 4)
    assert summary["index_benchmark_symbol"] == "000852.SH"
    assert summary["index_benchmark_periods"] == 2
    assert "index_benchmark_cumulative_return" in summary


def test_portfolio_backtester_selects_by_trading_day_interval():
    dataset = pd.DataFrame(
        [
            {"score_date": "20250101", "symbol": "A", "composite_cycle_oriented_score": 90, "forward_return_20d": 0.01},
            {"score_date": "20250103", "symbol": "A", "composite_cycle_oriented_score": 91, "forward_return_20d": 0.02},
            {"score_date": "20250106", "symbol": "A", "composite_cycle_oriented_score": 92, "forward_return_20d": 0.03},
            {"score_date": "20250108", "symbol": "A", "composite_cycle_oriented_score": 93, "forward_return_20d": 0.04},
        ]
    )
    calendar = ["20250101", "20250102", "20250103", "20250106", "20250107", "20250108"]
    config = PortfolioBacktestConfig(top_n_values=(1,), rebalance_interval_days=3, transaction_cost=0)

    result = PortfolioBacktester(dataset, config, trading_calendar=calendar).run()

    assert list(result.periods["score_date"]) == ["20250101", "20250106"]
    assert result.summary.iloc[0]["rebalance_interval_days"] == 3


def test_portfolio_backtester_cash_on_dates_records_zero_return_and_exit_turnover():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        cash_on_dates=("20250102",),
    )
    result = PortfolioBacktester(_sample_dataset(), config).run()
    periods = result.periods.sort_values("score_date")

    assert list(periods["score_date"]) == ["20250101", "20250102"]
    assert round(float(periods.iloc[0]["portfolio_return"]), 4) == 0.075
    assert float(periods.iloc[1]["portfolio_return"]) == 0.0
    assert float(periods.iloc[1]["portfolio_exposure"]) == 0.0
    assert int(periods.iloc[1]["selected_count"]) == 0
    assert float(periods.iloc[1]["turnover"]) == 1.0
    assert round(float(result.summary.iloc[0]["cumulative_return"]), 4) == 0.075
    assert round(float(result.summary.iloc[0]["average_exposure"]), 4) == 0.5


def test_non_bull_half_topn_stays_invested_in_fewer_names():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        cash_on_dates=("20250102",),
        non_bull_top_n_scale=0.5,
    )
    result = PortfolioBacktester(_sample_dataset(), config).run()
    periods = result.periods.sort_values("score_date")

    assert list(periods["score_date"]) == ["20250101", "20250102"]
    assert int(periods.iloc[0]["selected_count"]) == 2
    assert int(periods.iloc[1]["selected_count"]) == 1
    assert round(float(periods.iloc[0]["portfolio_return"]), 4) == 0.075
    assert round(float(periods.iloc[1]["portfolio_return"]), 4) == 0.02
    assert round(float(periods.iloc[1]["portfolio_exposure"]), 4) == 1.0
    assert result.summary.iloc[0]["non_bull_action"] == "half_topn"


def test_non_bull_half_topn_slot_keeps_parent_weights():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        cash_on_dates=("20250102",),
        non_bull_top_n_scale=0.5,
        non_bull_keep_parent_slots=True,
    )
    result = PortfolioBacktester(_sample_dataset(), config).run()
    periods = result.periods.sort_values("score_date")

    assert int(periods.iloc[1]["selected_count"]) == 1
    assert round(float(periods.iloc[1]["portfolio_exposure"]), 4) == 0.5
    assert round(float(periods.iloc[1]["portfolio_return"]), 4) == 0.01
    assert result.summary.iloc[0]["non_bull_action"] == "half_topn_slot"


def test_non_bull_top_n_scale_must_be_unit_interval():
    import pytest

    with pytest.raises(ValueError, match="non_bull_top_n_scale"):
        PortfolioBacktestConfig(non_bull_top_n_scale=-0.1)
    with pytest.raises(ValueError, match="non_bull_top_n_scale"):
        PortfolioBacktestConfig(non_bull_top_n_scale=1.2)
    PortfolioBacktestConfig(non_bull_top_n_scale=0.0)
    PortfolioBacktestConfig(non_bull_top_n_scale=0.5)
    PortfolioBacktestConfig(non_bull_top_n_scale=1.0)


def test_intra_period_cash_uses_truncated_return_and_clears_next_turnover():
    dataset = pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "A",
                "composite_cycle_oriented_score": 90,
                "forward_return_20d": 0.10,
                "forward_return_20d_live_cash": 0.02,
            },
            {
                "score_date": "20250101",
                "symbol": "B",
                "composite_cycle_oriented_score": 80,
                "forward_return_20d": 0.08,
                "forward_return_20d_live_cash": 0.01,
            },
            {
                "score_date": "20250103",
                "symbol": "A",
                "composite_cycle_oriented_score": 91,
                "forward_return_20d": 0.03,
                "forward_return_20d_live_cash": 0.03,
            },
            {
                "score_date": "20250106",
                "symbol": "A",
                "composite_cycle_oriented_score": 92,
                "forward_return_20d": 0.04,
                "forward_return_20d_live_cash": 0.04,
            },
            {
                "score_date": "20250106",
                "symbol": "B",
                "composite_cycle_oriented_score": 70,
                "forward_return_20d": 0.05,
                "forward_return_20d_live_cash": 0.05,
            },
        ]
    )
    calendar = ["20250101", "20250102", "20250103", "20250106"]
    cash_dates = ("20250103", "20250106")
    shared = dict(
        top_n_values=(1,),
        rebalance_interval_days=3,
        transaction_cost=0,
        cash_on_dates=cash_dates,
    )
    rebalance_only = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared),
        trading_calendar=calendar,
    ).run()
    live_like = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, intra_period_cash=True),
        trading_calendar=calendar,
    ).run()
    live_like_plan = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, intra_period_cash=True),
        trading_calendar=calendar,
        selection_cache={},
    ).run()

    rebalance_periods = rebalance_only.periods.sort_values("score_date")
    live_periods = live_like.periods.sort_values("score_date")
    assert list(rebalance_periods["score_date"]) == ["20250101", "20250106"]
    assert round(float(rebalance_periods.iloc[0]["portfolio_return"]), 4) == 0.10
    assert float(rebalance_periods.iloc[1]["turnover"]) == 1.0
    assert round(float(live_periods.iloc[0]["portfolio_return"]), 4) == 0.02
    assert bool(live_periods.iloc[0]["intra_period_cash_exit"]) is True
    assert live_periods.iloc[0]["cash_exit_date"] == "20250103"
    assert float(live_periods.iloc[1]["portfolio_return"]) == 0.0
    assert float(live_periods.iloc[1]["turnover"]) == 0.0
    assert int(live_like.summary.iloc[0]["intra_period_cash_exits"]) == 1
    assert round(float(live_like_plan.periods.sort_values("score_date").iloc[0]["portfolio_return"]), 4) == 0.02
    assert float(live_like_plan.periods.sort_values("score_date").iloc[1]["turnover"]) == 0.0


def test_portfolio_backtester_can_emit_trade_details():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0.01,
        initial_capital=105_000,
        include_trades=True,
        lot_size=100,
    )

    result = PortfolioBacktester(_sample_dataset_with_prices(), config).run()

    trades = result.trades
    assert len(trades) == 4
    first = trades.iloc[0]
    assert first["symbol"] == "A"
    assert first["name"] == "Alpha"
    assert first["buy_date"] == "20250101"
    assert first["sell_date"] == "20250121"
    assert round(first["quantity"], 4) == 500.0
    assert round(first["target_position_amount"], 4) == 52500.0
    assert round(first["buy_price"], 4) == 100.0
    assert round(first["sell_price"], 4) == 110.0
    assert round(first["gross_pnl"], 4) == 5000.0
    assert round(first["uninvested_cash"], 4) == 2500.0
    assert round(first["actual_weight"], 4) == round(50000 / 105000, 4)
    assert round(first["estimated_transaction_cost"], 4) == 1050.0
    assert round(first["net_pnl"], 4) == 3950.0


def test_portfolio_backtester_uses_explicit_fee_model():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0.01,
        buy_commission_rate=0.0001,
        sell_commission_rate=0.0001,
        min_commission=5,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0,
        initial_capital=105_000,
        include_trades=True,
        lot_size=100,
    )

    result = PortfolioBacktester(_sample_dataset_with_prices(), config).run()

    first = result.trades.iloc[0]
    assert round(first["buy_fee"], 4) == 5.0
    assert round(first["sell_fee"], 4) == 33.0
    assert round(first["commission"], 4) == 10.5
    assert round(first["stamp_tax"], 4) == 27.5
    assert round(first["estimated_transaction_cost"], 4) == 38.0
    assert round(first["net_pnl"], 4) == 4962.0


def test_portfolio_backtester_skips_trade_when_less_than_one_lot():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0.01,
        initial_capital=10_000,
        include_trades=True,
        lot_size=100,
    )

    result = PortfolioBacktester(_sample_dataset_with_prices(), config).run()

    assert result.trades.empty


def test_portfolio_backtester_supports_industry_capped_selection():
    config = PortfolioBacktestConfig(
        top_n_values=(4,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        construction_mode="industry_capped",
        max_industry_weight=0.5,
    )

    result = PortfolioBacktester(_industry_dataset(), config).run()

    period = result.periods.iloc[0]
    assert period["selected_count"] == 4
    assert period["construction_mode"] == "industry_capped"
    assert round(period["portfolio_return"], 4) == round((0.10 + 0.09 + 0.07 + 0.06) / 4, 4)


def test_portfolio_backtester_supports_industry_neutral_selection():
    config = PortfolioBacktestConfig(
        top_n_values=(4,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        construction_mode="industry_neutral",
    )

    result = PortfolioBacktester(_industry_dataset(), config).run()

    period = result.periods.iloc[0]
    assert period["selected_count"] == 4
    assert period["construction_mode"] == "industry_neutral"
    assert round(period["portfolio_return"], 4) == round((0.10 + 0.09 + 0.07 + 0.05) / 4, 4)


def _open_position_dataset():
    return pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "A",
                "name": "Alpha",
                "composite_cycle_oriented_score": 90,
                "forward_return_20d": 0.10,
                "entry_price": 100.0,
                "entry_date": "20250102",
                "exit_price_20d": 110.0,
                "exit_date_20d": "20250121",
            },
            {
                "score_date": "20250201",
                "symbol": "B",
                "name": "Beta",
                "composite_cycle_oriented_score": 95,
                "entry_price": 100.0,
                "entry_date": "20250202",
                "unrealized_return_20d": 0.05,
                "mark_price_20d": 105.0,
                "mark_date_20d": "20250210",
                "is_open_20d": True,
            },
            {
                "score_date": "20250201",
                "symbol": "C",
                "name": "Gamma",
                "composite_cycle_oriented_score": 80,
                "entry_price": 100.0,
                "entry_date": "20250202",
                "unrealized_return_20d": -0.02,
                "mark_price_20d": 98.0,
                "mark_date_20d": "20250210",
                "is_open_20d": True,
            },
        ]
    )


def test_portfolio_backtester_emits_open_trade_rows_without_affecting_summary():
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        horizon=20,
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0.01,
        buy_commission_rate=0.0001,
        sell_commission_rate=0.0001,
        min_commission=5,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0,
        initial_capital=200_000,
        include_trades=True,
        lot_size=100,
    )

    result = PortfolioBacktester(_open_position_dataset(), config).run()

    assert len(result.periods) == 1
    assert round(result.periods.iloc[0]["portfolio_return"], 4) == 0.10
    assert round(result.summary.iloc[0]["cumulative_return"], 4) == round(result.periods.iloc[0]["portfolio_return_net"], 4)
    open_trades = result.trades[result.trades["is_unrealized"] == True]
    closed_trades = result.trades[result.trades["is_unrealized"] != True]
    assert len(closed_trades) == 1
    assert len(open_trades) == 2

    open_row = open_trades[open_trades["symbol"] == "B"].iloc[0]
    assert open_row["status"] == "open"
    assert pd.isna(open_row["sell_date"]) or open_row["sell_date"] is None
    assert pd.isna(open_row["sell_price"]) or open_row["sell_price"] is None
    assert round(open_row["mark_price"], 4) == 105.0
    buy_amount = float(open_row["buy_amount"])
    mark_amount = float(open_row["quantity"]) * float(open_row["mark_price"])
    fees = TradeFeeModel.from_params(
        transaction_cost=0.01,
        buy_commission_rate=0.0001,
        sell_commission_rate=0.0001,
        min_commission=5,
        stamp_tax_rate=0.0005,
    ).estimate_round_trip(buy_amount, mark_amount)
    assert round(open_row["estimated_transaction_cost"], 4) == round(fees["total_fee"], 4)
    assert round(open_row["net_pnl"], 4) == round(mark_amount - buy_amount - fees["total_fee"], 4)
    assert closed_trades.iloc[0]["status"] == "closed"
    assert bool(closed_trades.iloc[0]["is_unrealized"]) is False


def test_open_trades_do_not_reselect_halted_names_outside_the_book():
    """A low-score name with marks but no 20d exit must not become a 100% ghost lot.

    Combo detail used to call _build_open_trade_rows(selected=None) on the same
    date as a completed TopN book, which re-ranked the halt/delist pool and
    showed e.g. 600225.SH as an open 100% holding that never entered P&L.
    """
    dataset = pd.DataFrame(
        [
            {
                "score_date": "20241219",
                "symbol": "A",
                "name": "Alpha",
                "composite_cycle_oriented_score": 91,
                "forward_return_20d": 0.10,
                "entry_price": 100.0,
                "entry_date": "20241220",
                "exit_price_20d": 110.0,
                "exit_date_20d": "20250120",
            },
            {
                "score_date": "20241219",
                "symbol": "B",
                "name": "Beta",
                "composite_cycle_oriented_score": 88,
                "forward_return_20d": 0.05,
                "entry_price": 100.0,
                "entry_date": "20241220",
                "exit_price_20d": 105.0,
                "exit_date_20d": "20250120",
            },
            {
                "score_date": "20241219",
                "symbol": "HALT",
                "name": "Ghost",
                "composite_cycle_oriented_score": 31,
                "entry_price": 3.86,
                "entry_date": "20241220",
                "unrealized_return_20d": -0.80,
                "mark_price_20d": 0.78,
                "mark_date_20d": "20250227",
                "is_open_20d": True,
            },
        ]
    )
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        horizon=20,
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        initial_capital=1_000_000,
        include_trades=True,
        lot_size=100,
    )
    result = PortfolioBacktester(dataset, config).run()

    assert round(float(result.periods.iloc[0]["portfolio_return"]), 4) == 0.075
    symbols = set(result.trades["symbol"].astype(str))
    assert symbols == {"A", "B"}
    assert "HALT" not in symbols
    assert bool((result.trades["is_unrealized"] != True).all())
    assert bool((result.trades["status"] == "closed").all())
    weights = result.trades["actual_weight"].astype(float)
    assert float(weights.max()) < 0.6


def test_portfolio_backtester_emits_open_trades_when_no_forward_returns():
    dataset = pd.DataFrame(
        [
            {
                "score_date": "20250201",
                "symbol": "B",
                "name": "Beta",
                "composite_cycle_oriented_score": 95,
                "entry_price": 100.0,
                "entry_date": "20250202",
                "unrealized_return_20d": 0.05,
                "mark_price_20d": 105.0,
                "mark_date_20d": "20250210",
                "is_open_20d": True,
            },
            {
                "score_date": "20250201",
                "symbol": "C",
                "name": "Gamma",
                "composite_cycle_oriented_score": 80,
                "entry_price": 100.0,
                "entry_date": "20250202",
                "unrealized_return_20d": -0.02,
                "mark_price_20d": 98.0,
                "mark_date_20d": "20250210",
                "is_open_20d": True,
            },
        ]
    )
    config = PortfolioBacktestConfig(
        top_n_values=(2,),
        horizon=20,
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        initial_capital=200_000,
        include_trades=True,
        lot_size=100,
    )

    result = PortfolioBacktester(dataset, config).run()

    assert result.periods.empty
    assert result.summary.empty
    assert len(result.trades) == 2
    assert set(result.trades["status"]) == {"open"}
    assert bool(result.trades["is_unrealized"].all()) is True


def test_bull_exposure_must_be_in_unit_interval():
    import pytest

    with pytest.raises(ValueError, match="bull_exposure"):
        PortfolioBacktestConfig(bull_exposure=0)
    with pytest.raises(ValueError, match="bull_exposure"):
        PortfolioBacktestConfig(bull_exposure=1.2)
    PortfolioBacktestConfig(bull_exposure=1.0)
    PortfolioBacktestConfig(bull_exposure=0.5)


def test_bull_exposure_scales_invested_weights_after_vol():
    dataset = pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "A",
                "composite_cycle_oriented_score": 90,
                "forward_return_20d": 0.10,
                "vol_20d_annual": 1.0,
            },
            {
                "score_date": "20250101",
                "symbol": "B",
                "composite_cycle_oriented_score": 80,
                "forward_return_20d": 0.10,
                "vol_20d_annual": 1.0,
            },
        ]
    )
    shared = dict(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        volatility_mode="cash_drag",
        target_vol_annual=0.5,
    )
    full = PortfolioBacktester(dataset, PortfolioBacktestConfig(**shared, bull_exposure=1.0)).run()
    half = PortfolioBacktester(dataset, PortfolioBacktestConfig(**shared, bull_exposure=0.5)).run()
    # cash_drag@50% with vol=100% → 50% book; bull 0.5 stacks to 25% book.
    assert round(float(full.periods.iloc[0]["portfolio_return"]), 4) == 0.05
    assert round(float(half.periods.iloc[0]["portfolio_return"]), 4) == 0.025
    assert round(float(full.periods.iloc[0]["portfolio_exposure"]), 4) == 0.5
    assert round(float(half.periods.iloc[0]["portfolio_exposure"]), 4) == 0.25


def _refill_prices():
    def bar(date, open_, high, close):
        return {
            "trade_date": date,
            "open": open_,
            "high": high,
            "low": min(open_, close),
            "close": close,
        }

    a = [
        bar("20250101", 100, 100, 100),
        bar("20250102", 100, 120, 119),
        bar("20250103", 110, 110, 100),
        bar("20250104", 90, 91, 90),
        bar("20250105", 90, 91, 90),
        bar("20250106", 90, 91, 90),
        bar("20250107", 90, 91, 90),
    ]
    b = [
        bar("20250101", 50, 50, 50),
        bar("20250102", 50, 50, 50),
        bar("20250103", 50, 50, 50),
        bar("20250104", 50, 50, 50),
        bar("20250105", 60, 60, 60),
        bar("20250106", 70, 70, 70),
        bar("20250107", 80, 80, 80),
    ]
    c = [
        bar("20250101", 40, 40, 40),
        bar("20250102", 40, 40, 40),
        bar("20250103", 40, 40, 40),
        bar("20250104", 40, 40, 40),
        bar("20250105", 40, 40, 40),
        bar("20250106", 40, 40, 40),
        bar("20250107", 40, 40, 40),
    ]
    return {"A": a, "B": b, "C": c}


def _refill_dataset():
    return pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "A",
                "composite_cycle_oriented_score": 90,
                "forward_return_5d": 90 / 100 - 1,
                "entry_date": "20250102",
                "entry_price": 100.0,
                "exit_date_5d": "20250104",
                "exit_reason_5d": "trailing_stop",
            },
            {
                "score_date": "20250101",
                "symbol": "B",
                "composite_cycle_oriented_score": 80,
                "forward_return_5d": 0.0,
                "entry_date": "20250102",
                "exit_date_5d": "20250107",
                "exit_reason_5d": "horizon",
            },
            {
                "score_date": "20250101",
                "symbol": "C",
                "composite_cycle_oriented_score": 70,
                "forward_return_5d": 0.0,
                "entry_date": "20250102",
                "exit_date_5d": "20250107",
                "exit_reason_5d": "horizon",
            },
        ]
    )


def test_intra_period_refill_requires_price_cache():
    import pytest

    with pytest.raises(ValueError, match="price_cache"):
        PortfolioBacktester(
            _refill_dataset(),
            PortfolioBacktestConfig(
                horizon=5,
                top_n_values=(1,),
                rebalance_interval_days=None,
                rebalance_step=1,
                transaction_cost=0,
                intra_period_refill=True,
                trailing_stop_pct=0.15,
            ),
        )


def test_intra_period_refill_compounds_next_ranked_name_until_period_end():
    dataset = _refill_dataset()
    config = PortfolioBacktestConfig(
        horizon=5,
        top_n_values=(1,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        trailing_stop_pct=0.15,
    )
    off = PortfolioBacktester(dataset, config, price_cache=_refill_prices()).run()
    on = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**{**config.__dict__, "intra_period_refill": True}),
        price_cache=_refill_prices(),
    ).run()
    on_plan = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**{**config.__dict__, "intra_period_refill": True}),
        price_cache=_refill_prices(),
        selection_cache={},
    ).run()

    assert round(float(off.periods.iloc[0]["portfolio_return"]), 4) == round(90 / 100 - 1, 4)
    expected = (90 / 100) * (80 / 50) - 1
    assert round(float(on.periods.iloc[0]["portfolio_return"]), 4) == round(expected, 4)
    assert int(on.periods.iloc[0]["intra_period_refills"]) == 1
    assert round(float(on_plan.periods.iloc[0]["portfolio_return"]), 4) == round(expected, 4)


def test_intra_period_refill_skips_regime_cash_exits():
    dataset = _refill_dataset()
    dataset.loc[dataset["symbol"] == "A", "exit_reason_5d"] = "regime_cash"
    result = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(
            horizon=5,
            top_n_values=(1,),
            rebalance_interval_days=None,
            rebalance_step=1,
            transaction_cost=0,
            intra_period_refill=True,
            trailing_stop_pct=0.15,
        ),
        price_cache=_refill_prices(),
    ).run()
    assert round(float(result.periods.iloc[0]["portfolio_return"]), 4) == round(90 / 100 - 1, 4)
    assert int(result.periods.iloc[0]["intra_period_refills"]) == 0


def _crowd_bar(date, open_, high, close):
    return {
        "trade_date": date,
        "open": open_,
        "high": high,
        "low": min(open_, close),
        "close": close,
    }


def _crowd_prices():
    a = [
        _crowd_bar("20250101", 100, 100, 100),
        _crowd_bar("20250102", 100, 120, 119),
        _crowd_bar("20250103", 110, 110, 100),
        _crowd_bar("20250104", 90, 91, 90),
        _crowd_bar("20250105", 90, 91, 90),
        _crowd_bar("20250106", 90, 91, 90),
        _crowd_bar("20250107", 90, 91, 90),
    ]
    b = [
        _crowd_bar("20250101", 50, 50, 50),
        _crowd_bar("20250102", 50, 50, 50),
        _crowd_bar("20250103", 50, 60, 59),
        _crowd_bar("20250104", 59, 60, 59),
        _crowd_bar("20250105", 59, 60, 50),
        _crowd_bar("20250106", 40, 41, 40),
        _crowd_bar("20250107", 40, 41, 40),
    ]
    c = [
        _crowd_bar("20250101", 80, 80, 80),
        _crowd_bar("20250102", 80, 81, 80),
        _crowd_bar("20250103", 79, 81, 79),
        _crowd_bar("20250104", 78, 80, 78),
        _crowd_bar("20250105", 70, 72, 70),
        _crowd_bar("20250106", 60, 62, 60),
        _crowd_bar("20250107", 50, 51, 50),
    ]
    return {"A": a, "B": b, "C": c}


def _crowd_dataset():
    return pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "A",
                "composite_cycle_oriented_score": 90,
                "entry_date": "20250102",
                "entry_price": 100,
                "forward_return_5d": 90 / 100 - 1,
                "exit_reason_5d": "trailing_stop",
                "stop_trigger_date_5d": "20250103",
                "exit_date_5d": "20250104",
                "exit_price_5d": 90,
            },
            {
                "score_date": "20250101",
                "symbol": "B",
                "composite_cycle_oriented_score": 80,
                "entry_date": "20250102",
                "entry_price": 50,
                "forward_return_5d": 40 / 50 - 1,
                "exit_reason_5d": "trailing_stop",
                "stop_trigger_date_5d": "20250105",
                "exit_date_5d": "20250106",
                "exit_price_5d": 40,
            },
            {
                "score_date": "20250101",
                "symbol": "C",
                "composite_cycle_oriented_score": 70,
                "entry_date": "20250102",
                "entry_price": 80,
                "forward_return_5d": 50 / 80 - 1,
                "exit_reason_5d": "horizon",
                "exit_date_5d": "20250107",
                "exit_price_5d": 50,
            },
        ]
    )


def test_crowd_exit_requires_price_cache():
    import pytest

    with pytest.raises(ValueError, match="price_cache"):
        PortfolioBacktester(
            _crowd_dataset(),
            PortfolioBacktestConfig(
                horizon=5,
                top_n_values=(3,),
                rebalance_interval_days=None,
                rebalance_step=1,
                transaction_cost=0,
                crowd_exit_k=2,
                trailing_stop_pct=0.15,
            ),
        )
    PortfolioBacktester(
        _crowd_dataset(),
        PortfolioBacktestConfig(
            horizon=5,
            top_n_values=(3,),
            rebalance_interval_days=None,
            rebalance_step=1,
            transaction_cost=0,
            crowd_exit_k=0,
            trailing_stop_pct=0.15,
        ),
    )
    with pytest.raises(ValueError, match="crowd_exit_k"):
        PortfolioBacktestConfig(crowd_exit_k=-1)


def test_crowd_exit_flattens_remaining_after_k_trailing_stops():
    dataset = _crowd_dataset()
    shared = dict(
        horizon=5,
        top_n_values=(3,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        trailing_stop_pct=0.15,
    )
    off = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared),
        price_cache=_crowd_prices(),
    ).run()
    k1 = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, crowd_exit_k=1),
        price_cache=_crowd_prices(),
    ).run()
    k2 = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, crowd_exit_k=2),
        price_cache=_crowd_prices(),
    ).run()
    k3 = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, crowd_exit_k=3),
        price_cache=_crowd_prices(),
    ).run()
    k2_plan = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, crowd_exit_k=2),
        price_cache=_crowd_prices(),
        selection_cache={},
    ).run()

    off_ret = (90 / 100 + 40 / 50 + 50 / 80) / 3 - 1
    k1_ret = (90 / 100 + 59 / 50 + 78 / 80) / 3 - 1
    k2_ret = (90 / 100 + 40 / 50 + 60 / 80) / 3 - 1
    assert round(float(off.periods.iloc[0]["portfolio_return"]), 4) == round(off_ret, 4)
    assert round(float(k3.periods.iloc[0]["portfolio_return"]), 4) == round(off_ret, 4)
    assert int(k3.periods.iloc[0]["crowd_flattened"]) == 0
    assert round(float(k1.periods.iloc[0]["portfolio_return"]), 4) == round(k1_ret, 4)
    assert bool(k1.periods.iloc[0]["crowd_exit"]) is True
    assert int(k1.periods.iloc[0]["crowd_flattened"]) == 2
    assert round(float(k2.periods.iloc[0]["portfolio_return"]), 4) == round(k2_ret, 4)
    assert int(k2.periods.iloc[0]["crowd_flattened"]) == 1
    assert round(float(k2_plan.periods.iloc[0]["portfolio_return"]), 4) == round(k2_ret, 4)
    assert int(k2.summary.iloc[0]["crowd_exits"]) == 1


def test_crowd_exit_skips_refill():
    dataset = _crowd_dataset()
    result = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(
            horizon=5,
            top_n_values=(3,),
            rebalance_interval_days=None,
            rebalance_step=1,
            transaction_cost=0,
            trailing_stop_pct=0.15,
            intra_period_refill=True,
            crowd_exit_k=2,
        ),
        price_cache=_crowd_prices(),
    ).run()
    assert int(result.periods.iloc[0]["intra_period_refills"]) == 0
    assert int(result.periods.iloc[0]["crowd_flattened"]) == 1


def _crowd_dataset_all_stopped():
    rows = _crowd_dataset().to_dict(orient="records")
    rows[2] = {
        **rows[2],
        "forward_return_5d": 50 / 80 - 1,
        "exit_reason_5d": "trailing_stop",
        "stop_trigger_date_5d": "20250106",
        "exit_date_5d": "20250107",
    }
    return pd.DataFrame(rows)


def test_crowd_exit_from_cash_next_period_even_if_all_already_stopped():
    shared = dict(
        horizon=5,
        top_n_values=(3,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        trailing_stop_pct=0.15,
    )
    first = _crowd_dataset_all_stopped().to_dict(orient="records")
    second = [
        {
            "score_date": "20250108",
            "symbol": symbol,
            "composite_cycle_oriented_score": score,
            "entry_date": "20250109",
            "entry_price": price,
            "forward_return_5d": 0.01,
            "exit_reason_5d": "horizon",
            "exit_date_5d": "20250114",
        }
        for symbol, score, price in (("A", 91, 90), ("B", 81, 40), ("C", 71, 50))
    ]
    dataset = pd.DataFrame(first + second)
    extra_dates = [
        "20250108",
        "20250109",
        "20250110",
        "20250111",
        "20250112",
        "20250113",
        "20250114",
    ]
    prices = {}
    for symbol, bars in _crowd_prices().items():
        last = bars[-1]
        prices[symbol] = bars + [
            _crowd_bar(date, last["open"], last["high"], last["close"]) for date in extra_dates
        ]

    off = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared),
        price_cache=prices,
    ).run()
    on = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, crowd_exit_k=3),
        price_cache=prices,
    ).run()
    on_plan = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared, crowd_exit_k=3),
        price_cache=prices,
        selection_cache={},
    ).run()

    assert bool(on.periods.iloc[0]["crowd_exit"]) is True
    assert int(on.periods.iloc[0]["crowd_flattened"]) == 0
    assert round(float(on.periods.iloc[0]["portfolio_return"]), 4) == round(
        float(off.periods.iloc[0]["portfolio_return"]), 4
    )
    assert round(float(off.periods.iloc[1]["turnover"]), 4) == 0.0
    assert round(float(on.periods.iloc[1]["turnover"]), 4) == 1.0
    assert round(float(on_plan.periods.iloc[1]["turnover"]), 4) == 1.0


def test_crowd_exit_trades_match_flattened_exit():
    result = PortfolioBacktester(
        _crowd_dataset(),
        PortfolioBacktestConfig(
            horizon=5,
            top_n_values=(3,),
            rebalance_interval_days=None,
            rebalance_step=1,
            transaction_cost=0,
            trailing_stop_pct=0.15,
            crowd_exit_k=1,
            include_trades=True,
        ),
        price_cache=_crowd_prices(),
    ).run()
    trades = {str(row["symbol"]): row for _, row in result.trades.iterrows()}
    assert set(trades) == {"A", "B", "C"}
    assert trades["A"]["sell_date"] == "20250104"
    assert round(float(trades["A"]["sell_price"]), 4) == 90
    assert round(float(trades["A"]["holding_return"]), 4) == round(90 / 100 - 1, 4)
    assert trades["B"]["sell_date"] == "20250104"
    assert round(float(trades["B"]["sell_price"]), 4) == 59
    assert round(float(trades["B"]["holding_return"]), 4) == round(59 / 50 - 1, 4)
    assert trades["C"]["sell_date"] == "20250104"
    assert round(float(trades["C"]["sell_price"]), 4) == 78
    assert round(float(trades["C"]["holding_return"]), 4) == round(78 / 80 - 1, 4)


def test_crowd_exit_plan_path_dynamic_turnover_from_cash():
    score = "composite_cycle_oriented_score"
    names = ("A", "B", "C", "D", "E")
    history = [
        {"score_date": "20241231", "symbol": symbol, score: value}
        for symbol, value in zip(names, (100, 90, 80, 70, 60))
    ]
    first = [
        {
            "score_date": "20250101",
            "symbol": "A",
            score: 95,
            "entry_date": "20250102",
            "entry_price": 100,
            "forward_return_5d": 90 / 100 - 1,
            "exit_reason_5d": "trailing_stop",
            "stop_trigger_date_5d": "20250103",
            "exit_date_5d": "20250104",
            "exit_price_5d": 90,
        },
        {
            "score_date": "20250101",
            "symbol": "B",
            score: 85,
            "entry_date": "20250102",
            "entry_price": 50,
            "forward_return_5d": 40 / 50 - 1,
            "exit_reason_5d": "horizon",
            "exit_date_5d": "20250107",
            "exit_price_5d": 40,
        },
        *[
            {
                "score_date": "20250101",
                "symbol": symbol,
                score: value,
                "entry_date": "20250102",
                "entry_price": 80,
                "forward_return_5d": 0.0,
                "exit_reason_5d": "horizon",
                "exit_date_5d": "20250107",
                "exit_price_5d": 80,
            }
            for symbol, value in (("C", 80), ("D", 75), ("E", 70))
        ],
    ]
    second = [
        {
            "score_date": "20250108",
            "symbol": symbol,
            score: value,
            "entry_date": "20250109",
            "entry_price": 10,
            "forward_return_5d": 0.01,
            "exit_reason_5d": "horizon",
            "exit_date_5d": "20250114",
            "exit_price_5d": 10.1,
        }
        for symbol, value in (("A", 91), ("B", 81), ("C", 20), ("D", 10), ("E", 5))
    ]
    dataset = pd.DataFrame(history + first + second)
    extra_dates = [
        "20250108",
        "20250109",
        "20250110",
        "20250111",
        "20250112",
        "20250113",
        "20250114",
    ]
    prices = {}
    for symbol, bars in _crowd_prices().items():
        last = bars[-1]
        prices[symbol] = bars + [
            _crowd_bar(date, last["open"], last["high"], last["close"]) for date in extra_dates
        ]
    shared = dict(
        horizon=5,
        top_n_values=(5,),
        selection_mode="dynamic_score_threshold",
        threshold_lookback_days=1,
        max_positions=10,
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        trailing_stop_pct=0.15,
        crowd_exit_k=1,
    )
    inline = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared),
        price_cache=prices,
    ).run()
    plan = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**shared),
        price_cache=prices,
        selection_cache={},
    ).run()
    assert list(inline.periods["score_date"]) == ["20250101", "20250108"]
    assert int(inline.periods.iloc[0]["selected_count"]) == 5
    assert int(inline.periods.iloc[1]["selected_count"]) == 2
    assert bool(inline.periods.iloc[0]["crowd_exit"]) is True
    expected_turnover = 2 / 5
    assert round(float(inline.periods.iloc[1]["turnover"]), 4) == round(expected_turnover, 4)
    assert round(float(plan.periods.iloc[1]["turnover"]), 4) == round(expected_turnover, 4)



def test_portfolio_backtester_uses_raw_price_for_lot_quantity():
    """A high hfq factor must not turn a buyable name into zero lots."""
    df = _sample_dataset_with_prices()
    df["entry_raw_price"] = 25.0
    df["exit_raw_price_20d"] = 25.0 * (1 + df["forward_return_20d"])
    config = PortfolioBacktestConfig(
        top_n_values=(1,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        initial_capital=100_000.0,
        include_trades=True,
    )
    result = PortfolioBacktester(df, config).run()

    trade = result.trades.iloc[0]
    # 100k / raw 25 = 4000 shares; adjusted 100 would have given only 1000.
    assert trade["quantity"] == 4000
    assert trade["buy_price"] == 25.0
    assert trade["buy_price_adjusted"] == 100.0
    assert trade["sell_price"] == pytest.approx(27.5)
    assert trade["buy_amount"] == pytest.approx(100_000.0)
    assert trade["holding_return"] == pytest.approx(0.10)
