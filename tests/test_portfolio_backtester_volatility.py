import pandas as pd
import pytest

from stock_data_access.indicators.volatility import compute_volatility_features

from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
from analysis.sweep.portfolio_sweep_report import add_volatility_columns, build_volatility_variants


def _vol_dataset():
    return pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "LOW",
                "composite_growth_cycle_score": 100,
                "forward_return_1d": 0.10,
                "vol_20d_annual": 0.20,
            },
            {
                "score_date": "20250101",
                "symbol": "HIGH",
                "composite_growth_cycle_score": 90,
                "forward_return_1d": 0.02,
                "vol_20d_annual": 0.60,
            },
            {
                "score_date": "20250102",
                "symbol": "LOW",
                "composite_growth_cycle_score": 100,
                "forward_return_1d": 0.04,
                "vol_20d_annual": 0.20,
            },
            {
                "score_date": "20250102",
                "symbol": "HIGH",
                "composite_growth_cycle_score": 90,
                "forward_return_1d": 0.01,
                "vol_20d_annual": 0.60,
            },
        ]
    )


def test_build_volatility_variants_defaults_to_off_only():
    variants = build_volatility_variants()

    assert len(variants) == 1
    assert variants[0].mode == "off"
    assert variants[0].target_vol is None


def test_build_volatility_variants_expands_modes_and_target_vols():
    variants = build_volatility_variants(["off", "cash_drag"], [0.30, 0.40])

    assert [variant.name for variant in variants] == [
        "vol_off",
        "vol_cash_drag_30",
        "vol_cash_drag_40",
    ]


def test_portfolio_backtester_off_mode_matches_equal_weight_baseline():
    dataset = _vol_dataset()
    base = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(
            score_column="composite_growth_cycle_score",
            horizon=1,
            top_n_values=(2,),
            rebalance_interval_days=1,
            transaction_cost=0,
            volatility_mode="off",
        ),
    ).run()
    weighted = PortfolioBacktester(
        dataset,
        PortfolioBacktestConfig(
            score_column="composite_growth_cycle_score",
            horizon=1,
            top_n_values=(2,),
            rebalance_interval_days=1,
            transaction_cost=0,
            volatility_mode="cash_drag",
            target_vol_annual=999.0,
        ),
    ).run()

    assert round(base.periods.iloc[0]["portfolio_return"], 6) == 0.06
    assert round(weighted.periods.iloc[0]["portfolio_return"], 6) == 0.06
    assert base.summary.iloc[0]["average_exposure"] == pytest.approx(1.0)
    assert weighted.summary.iloc[0]["average_exposure"] == pytest.approx(1.0)


def test_portfolio_backtester_cash_drag_reduces_exposure_and_return():
    result = PortfolioBacktester(
        _vol_dataset(),
        PortfolioBacktestConfig(
            score_column="composite_growth_cycle_score",
            horizon=1,
            top_n_values=(2,),
            rebalance_interval_days=1,
            transaction_cost=0,
            volatility_mode="cash_drag",
            target_vol_annual=0.30,
        ),
    ).run()

    period = result.periods.iloc[0]
    summary = result.summary.iloc[0]

    # scales: LOW=1.0, HIGH=0.5 -> exposure = (0.5 + 0.25) = 0.75
    assert period["portfolio_exposure"] == pytest.approx(0.75)
    assert summary["average_exposure"] == pytest.approx(0.75)
    # weighted return = 0.5*0.10 + 0.25*0.02 = 0.055
    assert period["portfolio_return"] == pytest.approx(0.055)


def test_portfolio_backtester_renormalize_stays_fully_invested_and_tilts_to_low_vol():
    result = PortfolioBacktester(
        _vol_dataset(),
        PortfolioBacktestConfig(
            score_column="composite_growth_cycle_score",
            horizon=1,
            top_n_values=(2,),
            rebalance_interval_days=1,
            transaction_cost=0,
            volatility_mode="renormalize",
            target_vol_annual=0.30,
        ),
    ).run()

    period = result.periods.iloc[0]
    summary = result.summary.iloc[0]

    assert period["portfolio_exposure"] == pytest.approx(1.0)
    assert summary["average_exposure"] == pytest.approx(1.0)
    # weights: LOW=1.0, HIGH=0.5 -> renorm = 2/3 and 1/3
    assert period["portfolio_return"] == pytest.approx((2 / 3) * 0.10 + (1 / 3) * 0.02)


def test_add_volatility_columns_uses_only_prices_on_or_before_score_date():
    dataset = pd.DataFrame(
        [
            {"symbol": "AAA", "score_date": "20250110"},
        ]
    )
    base_prices = []
    close = 10.0
    for day in range(1, 11):
        trade_date = f"202501{day:02d}"
        base_prices.append(
            {"trade_date": trade_date, "open": close, "high": close + 1, "low": close - 1, "close": close}
        )
        close += 0.5
    future_spike = {"trade_date": "20250111", "open": close, "high": close + 50, "low": close - 1, "close": close + 40}
    price_cache = {"AAA": base_prices + [future_spike]}
    mutated = {"AAA": base_prices + [{**future_spike, "close": 99, "high": 100}]}

    baseline = add_volatility_columns(dataset, price_cache, vol_windows=(3,), atr_window=3, percentile_lookback=5)
    after_future_spike = add_volatility_columns(dataset, mutated, vol_windows=(3,), atr_window=3, percentile_lookback=5)

    assert baseline.loc[0, "vol_3d_annual"] == after_future_spike.loc[0, "vol_3d_annual"]
    assert baseline.loc[0, "atr_pct_3d"] == after_future_spike.loc[0, "atr_pct_3d"]


def test_add_volatility_columns_matches_shared_indicator_math():
    trade_dates = pd.date_range("2025-01-01", periods=25, freq="D").strftime("%Y%m%d").tolist()
    prices = []
    for idx, trade_date in enumerate(trade_dates):
        close = 10.0 + idx * 0.4 + (0.6 if idx % 5 == 0 else 0.0)
        prices.append(
            {
                "trade_date": trade_date,
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.4,
                "close": close,
            }
        )
    dataset = pd.DataFrame(
        [
            {"symbol": "AAA", "score_date": "20250112"},
            {"symbol": "AAA", "score_date": "20250125"},
        ]
    )

    enriched = add_volatility_columns(
        dataset,
        {"AAA": prices},
        vol_windows=(3, 5),
        atr_window=3,
        percentile_lookback=5,
    )

    for row_idx, score_date in enumerate(dataset["score_date"]):
        expected = compute_volatility_features(
            [row for row in prices if row["trade_date"] <= score_date],
            vol_windows=(3, 5),
            atr_window=3,
            percentile_lookback=5,
        )
        assert enriched.loc[row_idx, "vol_3d_annual"] == pytest.approx(expected["vol_3d_annual"])
        assert enriched.loc[row_idx, "vol_5d_annual"] == pytest.approx(expected["vol_5d_annual"])
        assert enriched.loc[row_idx, "atr_pct_3d"] == pytest.approx(expected["atr_pct_3d"])


def test_add_volatility_columns_preserves_non_default_index():
    dataset = pd.DataFrame(
        [{"symbol": "AAA", "score_date": "20250110"}],
        index=[99],
    )
    prices = []
    close = 10.0
    for day in range(1, 11):
        prices.append(
            {
                "trade_date": f"202501{day:02d}",
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
            }
        )
        close += 0.5

    enriched = add_volatility_columns(dataset, {"AAA": prices}, vol_windows=(3,), atr_window=3)

    assert list(enriched.index) == [99]
    assert enriched.loc[99, "vol_3d_annual"] is not None
