import pandas as pd
import pytest

from analysis.sweep.combo_detail import run_single_combo
from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester


SCORE = "composite_cycle_oriented_score"


def _rows(score_date, scores, *, returns=0.10, symbols=None, **extra):
    symbols = symbols or [f"S{i:02d}" for i in range(len(scores))]
    if isinstance(returns, (int, float)) or returns is None:
        returns = [returns] * len(scores)
    return [
        {
            "score_date": score_date,
            "symbol": symbol,
            SCORE: score,
            "forward_return_1d": ret,
            **extra,
        }
        for symbol, score, ret in zip(symbols, scores, returns)
    ]


def _config(**overrides):
    values = {
        "horizon": 1,
        "top_n_values": (2,),
        "selection_mode": "dynamic_score_threshold",
        "threshold_lookback_days": 2,
        "max_positions": 20,
        "rebalance_interval_days": None,
        "rebalance_step": 1,
        "transaction_cost": 0,
    }
    values.update(overrides)
    return PortfolioBacktestConfig(**values)


def test_dynamic_threshold_uses_all_prior_score_dates_and_includes_equal_boundary():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50, 40])
        + _rows("20250102", [80, 60, 30])
        + _rows("20250103", [70, 55, 54])
    )
    config = _config(rebalance_interval_days=2, rebalance_step=None)

    result = PortfolioBacktester(
        dataset,
        config,
        trading_calendar=["20250101", "20250102", "20250103"],
    ).run()

    assert list(result.periods["score_date"]) == ["20250103"]
    period = result.periods.iloc[0]
    assert period["score_threshold"] == 55
    assert period["threshold_sample_size"] == 2
    assert period["threshold_rank"] == 2
    assert period["selected_count"] == 2


def test_dynamic_threshold_history_does_not_depend_on_forward_returns():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50, 40], returns=None)
        + _rows("20250102", [80, 60, 30], returns=None)
        + _rows("20250103", [70, 55, 54], returns=[0.1, 0.2, 0.3])
    )

    period = PortfolioBacktester(dataset, _config()).run().periods.iloc[0]

    assert period["score_threshold"] == 55
    assert period["selected_count"] == 2
    assert period["portfolio_return"] == pytest.approx(0.15)


def test_dynamic_threshold_does_not_include_todays_cutoff():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 40, 30])
        + _rows("20250102", [80, 60, 20])
        + _rows("20250103", [120, 100, 49])
    )

    period = PortfolioBacktester(dataset, _config()).run().periods.iloc[0]

    assert period["score_threshold"] == 50
    assert period["selected_count"] == 2


def test_dynamic_threshold_applies_the_same_path_gate_to_history_and_today():
    dataset = pd.DataFrame(
        _rows("20250101", [100, 80, 60], ret_20=-0.1)
        + _rows("20250101", [70, 50], symbols=["P1", "P2"], ret_20=0.1)
        + _rows("20250102", [80, 50, 49], ret_20=0.1)
    )

    period = PortfolioBacktester(
        dataset,
        _config(
            threshold_lookback_days=1,
            path_gate="ret20_pos",
        ),
    ).run().periods.iloc[0]

    assert period["score_threshold"] == 50
    assert period["selected_count"] == 2


def test_dynamic_threshold_requires_a_full_warmup_window():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50])
        + _rows("20250102", [80, 60])
        + _rows("20250103", [70, 55])
    )

    result = PortfolioBacktester(
        dataset,
        _config(threshold_lookback_days=3),
    ).run()

    assert result.periods.empty
    assert result.summary.empty


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_n_values": (2, 3)}, "exactly one"),
        ({"threshold_lookback_days": 0}, "lookback"),
        ({"max_positions": 1}, "max_positions"),
        ({"selection_mode": "unknown"}, "selection_mode"),
    ],
)
def test_dynamic_threshold_rejects_invalid_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        PortfolioBacktester(pd.DataFrame(), _config(**overrides))


def test_dynamic_threshold_caps_selection_at_twenty_positions():
    history = list(range(100, 70, -1))
    current = list(range(120, 90, -1))
    dataset = pd.DataFrame(
        _rows("20250101", history)
        + _rows("20250102", current)
    )

    period = PortfolioBacktester(
        dataset,
        _config(
            top_n_values=(10,),
            threshold_lookback_days=1,
            max_positions=20,
        ),
    ).run().periods.iloc[0]

    assert period["selected_count"] == 20
    assert period["portfolio_exposure"] == pytest.approx(1.0)
    assert period["cash_weight"] == pytest.approx(0.0)


def test_dynamic_threshold_keeps_cash_when_fewer_than_base_slots_qualify():
    history = list(range(100, 70, -1))
    current = [110, 109, 108, 107, 106, 105, 89, 88]
    dataset = pd.DataFrame(
        _rows("20250101", history)
        + _rows("20250102", current)
    )

    period = PortfolioBacktester(
        dataset,
        _config(
            top_n_values=(10,),
            threshold_lookback_days=1,
            max_positions=20,
        ),
    ).run().periods.iloc[0]

    assert period["selected_count"] == 6
    assert period["portfolio_exposure"] == pytest.approx(0.6)
    assert period["cash_weight"] == pytest.approx(0.4)
    assert period["portfolio_return"] == pytest.approx(0.06)


def test_dynamic_threshold_is_fully_invested_above_base_slots_without_leverage():
    history = list(range(100, 70, -1))
    current = list(range(120, 105, -1))
    dataset = pd.DataFrame(
        _rows("20250101", history)
        + _rows("20250102", current)
    )

    period = PortfolioBacktester(
        dataset,
        _config(
            top_n_values=(10,),
            threshold_lookback_days=1,
            max_positions=20,
        ),
    ).run().periods.iloc[0]

    assert period["selected_count"] == 15
    assert period["portfolio_exposure"] == pytest.approx(1.0)
    assert period["portfolio_return"] == pytest.approx(0.1)


def test_dynamic_threshold_volatility_modes_preserve_base_stock_exposure():
    dataset = pd.DataFrame(
        _rows("20250101", [100, 90, 80, 70])
        + _rows("20250102", [110, 95, 69], vol_20d_annual=0.4)
    )
    common = {
        "top_n_values": (4,),
        "threshold_lookback_days": 1,
        "max_positions": 10,
        "target_vol_annual": 0.2,
    }

    cash_drag = PortfolioBacktester(
        dataset,
        _config(volatility_mode="cash_drag", **common),
    ).run().periods.iloc[0]
    renormalize = PortfolioBacktester(
        dataset,
        _config(volatility_mode="renormalize", **common),
    ).run().periods.iloc[0]

    assert cash_drag["selected_count"] == 2
    assert cash_drag["portfolio_exposure"] == pytest.approx(0.25)
    assert renormalize["portfolio_exposure"] == pytest.approx(0.5)


def test_dynamic_threshold_empty_selection_records_cash_period_and_exit_turnover():
    dataset = pd.DataFrame(
        _rows("20250101", [80, 50], symbols=["A", "B"])
        + _rows("20250102", [70, 60], symbols=["A", "B"])
        + _rows("20250103", [59, 58], symbols=["A", "B"])
    )

    periods = PortfolioBacktester(
        dataset,
        _config(
            threshold_lookback_days=1,
            transaction_cost=0.01,
        ),
    ).run().periods

    assert len(periods) == 2
    empty = periods.iloc[1]
    assert empty["selected_count"] == 0
    assert empty["portfolio_return"] == 0
    assert empty["portfolio_exposure"] == 0
    assert empty["cash_weight"] == 1
    assert empty["turnover"] == pytest.approx(1.0)
    assert empty["portfolio_return_net"] == pytest.approx(-0.01)


def test_dynamic_threshold_empty_industry_selection_records_cash_period():
    dataset = pd.DataFrame(
        _rows("20250101", [80, 70], industry="Tech")
        + _rows("20250102", [60, 50], industry="Tech")
    )

    period = PortfolioBacktester(
        dataset,
        _config(
            threshold_lookback_days=1,
            construction_mode="industry_capped",
        ),
    ).run().periods.iloc[0]

    assert period["selected_count"] == 0
    assert period["portfolio_exposure"] == 0
    assert period["cash_weight"] == 1


def test_dynamic_threshold_turnover_uses_symbol_and_cash_weights():
    dataset = pd.DataFrame(
        _rows("20250101", [80, 50], symbols=["A", "B"])
        + _rows("20250102", [70, 60], symbols=["A", "B"])
        + _rows("20250103", [70, 60, 40], symbols=["B", "C", "A"])
    )

    periods = PortfolioBacktester(
        dataset,
        _config(threshold_lookback_days=1),
    ).run().periods

    assert list(periods["turnover"]) == pytest.approx([1.0, 0.5])


def test_dynamic_threshold_cached_plan_matches_inline_execution():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50, 40])
        + _rows("20250102", [80, 60, 30])
        + _rows("20250103", [70, 55, 54])
    )
    config = _config()

    inline = PortfolioBacktester(dataset, config).run()
    cached = PortfolioBacktester(dataset, config, selection_cache={}).run()

    pd.testing.assert_frame_equal(
        inline.periods.reset_index(drop=True),
        cached.periods.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        inline.summary.reset_index(drop=True),
        cached.summary.reset_index(drop=True),
    )


def test_fixed_top_n_remains_the_default_and_matches_explicit_mode():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50, 40])
        + _rows("20250102", [80, 60, 30])
    )
    base = {
        "horizon": 1,
        "top_n_values": (2,),
        "rebalance_interval_days": None,
        "rebalance_step": 1,
        "transaction_cost": 0,
    }

    default_result = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(**base),
    ).run()
    explicit_result = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(selection_mode="fixed_top_n", **base),
    ).run()

    assert PortfolioBacktestConfig().selection_mode == "fixed_top_n"
    assert PortfolioBacktestConfig().threshold_lookback_days == 10
    assert PortfolioBacktestConfig().max_positions == 20
    pd.testing.assert_frame_equal(default_result.periods, explicit_result.periods)


def test_combo_detail_uses_dynamic_threshold_selection_parameters():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50, 40])
        + _rows("20250102", [70, 50, 49])
    )

    detail = run_single_combo(
        dataset,
        SCORE,
        "top_n",
        2,
        1,
        rebalance_interval_days=1,
        transaction_cost=0,
        selection_mode="dynamic_score_threshold",
        threshold_lookback_days=1,
        max_positions=3,
    )

    assert detail["meta"]["selection_mode"] == "dynamic_score_threshold"
    assert detail["meta"]["threshold_lookback_days"] == 1
    assert detail["meta"]["max_positions"] == 3
    assert detail["summary"]["selection_mode"] == "dynamic_score_threshold"
    assert detail["periods"][0]["selected_count"] == 2


def test_dynamic_trade_detail_matches_summary_selection_for_closed_and_open_rows():
    dataset = pd.DataFrame(
        _rows("20250101", [90, 50, 40])
        + [
            {
                "score_date": "20250102",
                "symbol": "A",
                SCORE: 70,
                "forward_return_1d": 0.1,
                "entry_price": 100,
                "entry_date": "20250102",
                "exit_price_1d": 110,
                "exit_date_1d": "20250103",
            },
            {
                "score_date": "20250102",
                "symbol": "B",
                SCORE: 50,
                "forward_return_1d": None,
                "entry_price": 100,
                "entry_date": "20250102",
                "mark_price_1d": 105,
                "mark_date_1d": "20250103",
            },
            {
                "score_date": "20250102",
                "symbol": "C",
                SCORE: 49,
                "forward_return_1d": 0.2,
                "entry_price": 100,
                "entry_date": "20250102",
                "exit_price_1d": 120,
                "exit_date_1d": "20250103",
            },
        ]
    )

    result = PortfolioBacktester(
        dataset,
        _config(
            threshold_lookback_days=1,
            include_trades=True,
            initial_capital=200_000,
        ),
    ).run()

    assert result.periods.iloc[0]["selected_count"] == 2
    assert result.periods.iloc[0]["portfolio_return"] == pytest.approx(0.05)
    assert set(result.trades["symbol"]) == {"A", "B"}
    assert set(result.trades["status"]) == {"closed", "open"}
    # B was bought and is merely still open, so it keeps its exposure.
    assert result.periods.iloc[0]["portfolio_exposure"] == pytest.approx(1.0)
    assert result.periods.iloc[0]["unfilled_weight"] == pytest.approx(0.0)


def _unbuyable_dataset():
    """20250102 holds a top-scored name that could not be bought (no entry price)."""
    return pd.DataFrame(
        _rows("20250101", [100, 90, 80, 70], entry_price=10.0, exit_price_1d=11.0)
        + [
            {
                "score_date": "20250102",
                "symbol": "S00",
                SCORE: 110,
                "forward_return_1d": None,
            },
            {
                "score_date": "20250102",
                "symbol": "S01",
                SCORE: 100,
                "forward_return_1d": 0.10,
                "entry_price": 10.0,
                "entry_date": "20250102",
                "exit_price_1d": 11.0,
                "exit_date_1d": "20250103",
            },
        ]
    )


def test_dynamic_threshold_books_unbuyable_names_as_cash_not_flat_exposure():
    period = PortfolioBacktester(
        _unbuyable_dataset(),
        _config(threshold_lookback_days=1),
    ).run().periods.iloc[0]

    assert period["selected_count"] == 2
    assert period["unfilled_count"] == 1
    assert period["unfilled_weight"] == pytest.approx(0.5)
    assert period["portfolio_exposure"] == pytest.approx(0.5)
    assert period["cash_weight"] == pytest.approx(0.5)
    assert period["portfolio_return"] == pytest.approx(0.05)


def test_dynamic_threshold_turnover_excludes_orders_that_never_filled():
    period = PortfolioBacktester(
        _unbuyable_dataset(),
        _config(threshold_lookback_days=1),
    ).run().periods.iloc[0]

    # Only the filled half moved out of cash, so a full 1.0 would overstate fees.
    assert period["turnover"] == pytest.approx(0.5)


def test_dynamic_threshold_trade_amounts_reconcile_with_period_exposure():
    result = PortfolioBacktester(
        _unbuyable_dataset(),
        _config(threshold_lookback_days=1, include_trades=True, initial_capital=200_000),
    ).run()

    period = result.periods.iloc[0]
    traded = result.trades[result.trades["score_date"] == "20250102"]
    assert set(traded["symbol"]) == {"S01"}
    assert traded["target_position_amount"].sum() == pytest.approx(
        period["portfolio_exposure"] * 200_000
    )


def _compounding_dataset():
    """Three rebalances, two names each, every one returning +10% at 10.0 -> 11.0."""
    return pd.DataFrame(
        _rows("20250101", [100, 90, 80, 70], entry_price=10.0, exit_price_1d=11.0)
        + _rows("20250102", [100, 90, 80, 70], entry_price=10.0, exit_price_1d=11.0)
        + _rows("20250103", [100, 90, 80, 70], entry_price=10.0, exit_price_1d=11.0)
    )


def _compounding_config(**overrides):
    values = {
        "selection_mode": "fixed_top_n",
        "top_n_values": (2,),
        "cash_on_dates": (),
        "include_trades": True,
        "initial_capital": 200_000,
    }
    values.update(overrides)
    return _config(**values)


def test_trade_amounts_compound_with_running_equity():
    """Books are sized on the equity at each rebalance, not on a frozen initial capital."""
    result = PortfolioBacktester(_compounding_dataset(), _compounding_config()).run()

    totals = result.trades.groupby("score_date")["target_position_amount"].sum()
    # 200k -> x1.1 -> 220k -> x1.1 -> 242k, fully invested each time.
    assert totals.tolist() == pytest.approx([200_000, 220_000, 242_000])
    assert result.summary.iloc[0]["cumulative_return"] == pytest.approx(0.331)


def test_trade_amounts_reconcile_with_exposure_throughout_the_equity_path():
    """Every period's book equals exposure x the equity that period actually started with."""
    result = PortfolioBacktester(
        _compounding_dataset(),
        _compounding_config(cash_on_dates=("20250102",), non_bull_top_n_scale=0.0),
    ).run()

    assert list(result.periods["score_date"]) == ["20250101", "20250102", "20250103"]
    # The flat-cash overlay really did empty the middle period, so the loop below
    # exercises the cash branch's equity advance rather than three invested periods.
    assert result.periods.iloc[1]["portfolio_exposure"] == pytest.approx(0.0)

    equity = 200_000.0
    for _, period in result.periods.iterrows():
        traded = result.trades[result.trades["score_date"] == period["score_date"]]
        assert traded["target_position_amount"].sum() == pytest.approx(
            period["portfolio_exposure"] * equity
        )
        equity *= 1.0 + period["portfolio_return_net"]
