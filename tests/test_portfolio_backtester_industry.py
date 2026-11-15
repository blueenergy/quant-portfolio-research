"""Portfolio backtester industry L1 filter."""

import pandas as pd
import pytest

from shared.industry_universe import (
    SW_L1_CODE_COL,
    SW_L1_REGIME_COL,
    attach_sw_l1,
    apply_industry_filter,
)
from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester


def test_portfolio_backtester_industry_l1_filter():
    dataset = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003"],
            "score_date": ["20240101", "20240101", "20240101"],
            "composite_growth_cycle_score": [90.0, 80.0, 70.0],
            SW_L1_CODE_COL: ["801080.SI", "801080.SI", "801780.SI"],
            "industry": ["电子", "电子", "银行"],
        }
    )
    config = PortfolioBacktestConfig(
        score_column="composite_growth_cycle_score",
        horizon=20,
        top_n_values=(2,),
        industry_l1_codes=("801080.SI",),
        construction_mode="top_n",
    )
    backtester = PortfolioBacktester(dataset, config)
    group = dataset.groupby("score_date").get_group("20240101")
    pool = backtester._score_valid_pool(group)
    assert set(pool["symbol"].tolist()) == {"000001", "000002"}


def test_portfolio_backtester_industry_l1_selection_plan_uses_filter():
    """Sweep fast path (_build_selection_plan) must apply L1 filter, not only _score_valid_pool."""
    dataset = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score_date": ["20240101", "20240101", "20240101", "20240101"],
            "composite_growth_cycle_score": [90.0, 80.0, 70.0, 60.0],
            SW_L1_CODE_COL: ["801080.SI", "801080.SI", "801170.SI", "801170.SI"],
            "forward_return_20d": [0.1, 0.05, 0.2, 0.15],
        }
    )
    selection_cache: dict = {}
    config_080 = PortfolioBacktestConfig(
        score_column="composite_growth_cycle_score",
        horizon=20,
        top_n_values=(1,),
        industry_l1_codes=("801080.SI",),
        construction_mode="top_n",
        path_gate="off",
        rebalance_interval_days=None,
    )
    config_170 = PortfolioBacktestConfig(
        score_column="composite_growth_cycle_score",
        horizon=20,
        top_n_values=(1,),
        industry_l1_codes=("801170.SI",),
        construction_mode="top_n",
        path_gate="off",
        rebalance_interval_days=None,
    )
    bt_080 = PortfolioBacktester(dataset, config_080, selection_cache=selection_cache)
    bt_170 = PortfolioBacktester(dataset, config_170, selection_cache=selection_cache)
    plan_080 = bt_080._get_selection_plan("forward_return_20d", [1])
    plan_170 = bt_170._get_selection_plan("forward_return_20d", [1])
    sym_080 = plan_080[0]["per_top_n"][1]["selected"]["symbol"].iloc[0]
    sym_170 = plan_170[0]["per_top_n"][1]["selected"]["symbol"].iloc[0]
    assert sym_080 == "A"
    assert sym_170 == "C"
    assert sym_080 != sym_170


def _industry_regime_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score_date": ["20240101", "20240101", "20240101", "20240101"],
            "composite_growth_cycle_score": [90.0, 80.0, 70.0, 60.0],
            SW_L1_CODE_COL: ["801080.SI", "801080.SI", "801170.SI", "801170.SI"],
            SW_L1_REGIME_COL: ["bull", "bear", "bull", "bear"],
            "forward_return_20d": [0.1, 0.05, 0.2, 0.15],
        }
    )


def _industry_regime_config(**overrides) -> PortfolioBacktestConfig:
    params = dict(
        score_column="composite_growth_cycle_score",
        horizon=20,
        top_n_values=(4,),
        construction_mode="top_n",
        path_gate="off",
        rebalance_interval_days=None,
    )
    params.update(overrides)
    return PortfolioBacktestConfig(**params)


def test_portfolio_backtester_industry_regime_scale_survives_plan_path():
    """The regime column must reach the cached selection plan.

    Sweeps always pass a ``selection_cache``, so ``_get_selection_plan`` is the
    production path. A column missing from its whitelist would silently disable
    the scale (``row.get`` returns None -> factor 1.0) instead of failing loudly.
    """
    selection_cache: dict = {}
    bt_off = PortfolioBacktester(
        _industry_regime_dataset(), _industry_regime_config(), selection_cache=selection_cache
    )
    bt_on = PortfolioBacktester(
        _industry_regime_dataset(),
        _industry_regime_config(industry_regime_scale=0.0),
        selection_cache=selection_cache,
    )

    plan = bt_on._get_selection_plan("forward_return_20d", [4])
    selected = plan[0]["per_top_n"][4]["selected"]
    assert SW_L1_REGIME_COL in selected.columns, "plan path dropped the regime column"
    assert set(selected["symbol"]) == {"A", "B", "C", "D"}

    weights_off = bt_off._compute_weights(selected)
    weights_on = bt_on._compute_weights(selected)

    # Non-bull names are zeroed; bull names keep the untouched equal weight.
    assert weights_on["B"] == 0.0
    assert weights_on["D"] == 0.0
    assert weights_on["A"] == weights_off["A"]
    assert weights_on["C"] == weights_off["C"]
    # Freed weight stays in cash rather than being redistributed.
    assert sum(weights_on.values()) < sum(weights_off.values())


def test_portfolio_backtester_industry_regime_scale_off_is_byte_identical():
    """Default scale=1.0 must leave weights exactly as before the feature."""
    dataset = _industry_regime_dataset()
    bt = PortfolioBacktester(dataset, _industry_regime_config(), selection_cache={})
    selected = bt._get_selection_plan("forward_return_20d", [4])[0]["per_top_n"][4]["selected"]
    weights = bt._compute_weights(selected)
    assert weights == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def test_portfolio_backtester_industry_regime_scale_half_halves_non_bull():
    dataset = _industry_regime_dataset()
    bt = PortfolioBacktester(
        dataset, _industry_regime_config(industry_regime_scale=0.5), selection_cache={}
    )
    selected = bt._get_selection_plan("forward_return_20d", [4])[0]["per_top_n"][4]["selected"]
    weights = bt._compute_weights(selected)
    assert weights["A"] == 0.25
    assert weights["C"] == 0.25
    assert weights["B"] == 0.125
    assert weights["D"] == 0.125


def test_portfolio_backtester_industry_regime_scale_requires_the_column():
    """A missing column must fail loudly, not weight everything at 1.0."""
    dataset = _industry_regime_dataset().drop(columns=[SW_L1_REGIME_COL])
    with pytest.raises(ValueError, match="sw_l1_regime"):
        PortfolioBacktester(
            dataset, _industry_regime_config(industry_regime_scale=0.0), selection_cache={}
        )
    # The default (off) must stay permissive for datasets that never carry it.
    PortfolioBacktester(dataset, _industry_regime_config(), selection_cache={})


def test_portfolio_backtester_industry_regime_scale_rejects_out_of_range():
    with pytest.raises(ValueError, match="industry_regime_scale"):
        _industry_regime_config(industry_regime_scale=1.5)


def test_portfolio_backtester_industry_regime_missing_label_is_untouched():
    """Unmapped rows (empty label) must not be treated as non-bull."""
    dataset = _industry_regime_dataset()
    dataset.loc[dataset["symbol"] == "B", SW_L1_REGIME_COL] = ""
    bt = PortfolioBacktester(
        dataset, _industry_regime_config(industry_regime_scale=0.0), selection_cache={}
    )
    selected = bt._get_selection_plan("forward_return_20d", [4])[0]["per_top_n"][4]["selected"]
    weights = bt._compute_weights(selected)
    assert weights["B"] == 0.25, "a missing label must fall back to no intervention"
    assert weights["D"] == 0.0


def _cap_dataset():
    return pd.DataFrame(
        {
            "symbol": ["E1", "E2", "E3", "E4", "B1", "B2", "B3", "B4"],
            "score_date": ["20240101"] * 8,
            "composite_growth_cycle_score": [90, 85, 80, 75, 70, 65, 60, 55],
            SW_L1_CODE_COL: ["801080.SI"] * 4 + ["801780.SI"] * 4,
            # Legacy sub-industry: each electronics name has a different label.
            "industry": ["元器件", "半导体", "IT设备", "陶瓷", "银行", "银行", "银行", "银行"],
            "forward_return_20d": [0.1] * 8,
        }
    )


def _cap_config(**overrides):
    params = dict(
        score_column="composite_growth_cycle_score",
        horizon=20,
        top_n_values=(4,),
        construction_mode="industry_capped",
        max_industry_weight=0.5,
        path_gate="off",
        rebalance_interval_days=None,
    )
    params.update(overrides)
    return PortfolioBacktestConfig(**params)


def test_industry_cap_uses_sw_l1_not_legacy_industry():
    dataset = _cap_dataset()
    backtester = PortfolioBacktester(dataset, _cap_config())
    group = dataset.groupby("score_date").get_group("20240101")
    selected = backtester._select_portfolio(backtester._score_valid_pool(group), 4)

    counts = selected[SW_L1_CODE_COL].value_counts().to_dict()
    assert counts == {"801080.SI": 2, "801780.SI": 2}
    # The legacy sub-industry taxonomy would have allowed all four electronics
    # names because each has a distinct ``industry`` label.
    assert selected["industry"].nunique() > 2


def test_industry_neutral_uses_sw_l1_not_legacy_industry():
    dataset = _cap_dataset()
    backtester = PortfolioBacktester(
        dataset,
        _cap_config(construction_mode="industry_neutral"),
    )
    group = dataset.groupby("score_date").get_group("20240101")
    selected = backtester._select_portfolio(backtester._score_valid_pool(group), 4)

    counts = selected[SW_L1_CODE_COL].value_counts().to_dict()
    assert counts == {"801080.SI": 2, "801780.SI": 2}
