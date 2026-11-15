import json

import pandas as pd
import pytest

from analysis.sweep.params import (
    normalize_bull_exposures,
    normalize_crowd_exit_ks,
    normalize_industry_regime_scales,
    normalize_non_bull_actions,
    normalize_path_gates,
    normalize_refill_modes,
    normalize_trailing_stop_pcts,
    resolve_industry_regime_scales,
    resolve_regime_modes,
)
from analysis.sweep.portfolio_sweep_report import (
    build_sweep_variants,
    build_volatility_variants,
    render_html,
    resolve_tradable_horizons,
    run_parameter_sweep,
    summarize_stability,
    validate_rebalance_intervals,
)
from analysis.sweep.score_columns import (
    ScoreSweep,
    add_growth_cycle_weight_columns,
    add_weighted_score_columns,
    build_score_sweeps,
    parse_growth_cycle_weights,
)
from shared.industry_universe import SW_L1_CODE_COL, SW_L1_REGIME_COL


def _sample_dataset():
    return pd.DataFrame(
        [
            {"score_date": "20250101", "symbol": "A", "industry": "Tech", "composite_growth_cycle_score": 100, "growth_score": 100, "cycle_score": 20, "forward_return_1d": 0.10},
            {"score_date": "20250101", "symbol": "B", "industry": "Tech", "composite_growth_cycle_score": 95, "growth_score": 90, "cycle_score": 40, "forward_return_1d": 0.08},
            {"score_date": "20250101", "symbol": "C", "industry": "Bank", "composite_growth_cycle_score": 90, "growth_score": 50, "cycle_score": 100, "forward_return_1d": 0.04},
            {"score_date": "20250101", "symbol": "D", "industry": "Drug", "composite_growth_cycle_score": 85, "growth_score": 40, "cycle_score": 80, "forward_return_1d": 0.02},
            {"score_date": "20250102", "symbol": "A", "industry": "Tech", "composite_growth_cycle_score": 70, "growth_score": 70, "cycle_score": 20, "forward_return_1d": 0.01},
            {"score_date": "20250102", "symbol": "B", "industry": "Tech", "composite_growth_cycle_score": 60, "growth_score": 60, "cycle_score": 30, "forward_return_1d": 0.02},
            {"score_date": "20250102", "symbol": "C", "industry": "Bank", "composite_growth_cycle_score": 95, "growth_score": 40, "cycle_score": 95, "forward_return_1d": 0.03},
            {"score_date": "20250102", "symbol": "D", "industry": "Drug", "composite_growth_cycle_score": 90, "growth_score": 30, "cycle_score": 90, "forward_return_1d": 0.04},
        ]
    )


def _index_prices():
    return [
        {"trade_date": "20250101", "close": 1000},
        {"trade_date": "20250102", "close": 1010},
        {"trade_date": "20250103", "close": 1020},
    ]


def _sample_dataset_with_industry_regime():
    """Same rows, plus each name's own SW L1 industry and its regime label.

    Top-2 per date are A (100) and B (95); B is non-bull, so scaling B down is
    what moves the book's exposure.
    """
    dataset = _sample_dataset()
    dataset[SW_L1_CODE_COL] = ["801080.SI", "801080.SI", "801780.SI", "801170.SI"] * 2
    dataset[SW_L1_REGIME_COL] = ["bull", "bear", "bull", "bear"] * 2
    return dataset


def test_build_sweep_variants_includes_topn_active_and_neutral():
    variants = build_sweep_variants([0.2, 0.3])

    assert [variant.name for variant in variants] == [
        "growth_cycle_topn",
        "growth_cycle_active_cap20",
        "growth_cycle_active_cap30",
        "growth_cycle_neutral",
    ]


def test_validate_rebalance_intervals_defaults_to_horizon():
    assert validate_rebalance_intervals(20, None) == [20]


def test_validate_rebalance_intervals_rejects_mismatched_interval():
    with pytest.raises(ValueError, match="must equal horizon=20"):
        validate_rebalance_intervals(20, [10, 20])


def test_resolve_tradable_horizons_from_rebalance_list():
    assert resolve_tradable_horizons(20, [10, 40, 20, 10]) == [10, 20, 40]
    assert resolve_tradable_horizons(20, None) == [20]
    assert resolve_tradable_horizons(None, [30]) == [30]


def test_parse_growth_cycle_weights_accepts_ratios_and_growth_only():
    sweeps = parse_growth_cycle_weights(["20:80", "0.6:0.4", "80"])

    assert [sweep.name for sweep in sweeps] == [
        "growth_cycle_g20_c80",
        "growth_cycle_g60_c40",
        "growth_cycle_g80_c20",
    ]
    assert sweeps[0].growth_weight == 0.2
    assert sweeps[0].cycle_weight == 0.8


def test_add_growth_cycle_weight_columns_builds_weighted_score():
    sweeps = parse_growth_cycle_weights(["25:75"])
    dataset = add_growth_cycle_weight_columns(_sample_dataset(), sweeps)

    assert dataset.loc[0, "growth_cycle_g25_c75_score"] == 40
    assert dataset.loc[2, "growth_cycle_g25_c75_score"] == 87.5


def test_run_parameter_sweep_merges_returns_and_exposure_metrics():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([0.5]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
    )

    rows = payload["rows"]
    variants = {row["variant"] for row in rows}

    assert variants == {"growth_cycle_topn", "growth_cycle_active_cap50", "growth_cycle_neutral"}
    assert all(row["avg_active_share"] is not None for row in rows)
    assert all(row["risk_adjusted_score"] is not None for row in rows)
    assert all(row.get("volatility_column") == "vol_20d_annual" for row in rows)
    assert payload["stability"]


def test_run_parameter_sweep_carries_dynamic_threshold_parameters():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        selection_mode="dynamic_score_threshold",
        threshold_lookback_days=1,
        max_positions=3,
    )

    assert payload["selection_mode"] == "dynamic_score_threshold"
    assert payload["threshold_lookback_days"] == 1
    assert payload["max_positions"] == 3
    assert payload["rows"]
    assert all(
        row["selection_mode"] == "dynamic_score_threshold"
        and row["threshold_lookback_days"] == 1
        and row["max_positions"] == 3
        for row in payload["rows"]
    )


def test_run_parameter_sweep_carries_generic_score_weights():
    dataset = _sample_dataset().copy()
    dataset["fundamental_score"] = [80, 70, 60, 50, 50, 60, 70, 80]
    dataset["value_score"] = [40, 50, 60, 70, 70, 60, 50, 40]
    sweeps = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=["fundamental:0.6,value:0.4"],
    )
    dataset = add_weighted_score_columns(dataset, sweeps)
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=sweeps,
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
    )

    assert payload["rows"]
    assert all(
        row["score_weights"] == {"fundamental": 0.6, "value": 0.4}
        for row in payload["rows"]
    )


def test_normalize_trailing_stop_pcts():
    assert normalize_trailing_stop_pcts() == [None]
    assert normalize_trailing_stop_pcts(trailing_stop_pct=0.15) == [0.15]
    assert normalize_trailing_stop_pcts(trailing_stop_pct=0) == [None]
    assert normalize_trailing_stop_pcts(trailing_stop_pcts=[0, 0.15]) == [None, 0.15]
    assert normalize_trailing_stop_pcts(trailing_stop_pcts=[0, 10, 15, 20]) == [None, 0.1, 0.15, 0.2]


def test_normalize_bull_exposures_and_refill_modes():
    assert normalize_bull_exposures() == [1.0]
    assert normalize_bull_exposures([1.0, 0.5, 0.5]) == [1.0, 0.5]
    assert normalize_refill_modes() == [False]
    assert normalize_refill_modes(["off", "on", "true"]) == [False, True]
    assert normalize_crowd_exit_ks() == [0]
    assert normalize_crowd_exit_ks([0, 3, 3, 5]) == [0, 3, 5]
    assert normalize_non_bull_actions() == ["cash"]
    assert normalize_non_bull_actions(["cash", "half", "half_topn_slot", "half"]) == [
        "cash",
        "half_topn",
        "half_topn_slot",
    ]


def test_run_parameter_sweep_bull_exposure_and_refill_axes():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        bull_exposures=[1.0, 0.5],
        refill_modes=["off"],
        price_cache={"A": [{"trade_date": "20250101", "open": 10, "high": 10, "close": 10}]},
    )
    exposures = {row.get("bull_exposure") for row in payload["rows"]}
    assert exposures == {1.0, 0.5}
    assert payload["bull_exposures"] == [1.0, 0.5]
    assert all(row.get("intra_period_refill") is False for row in payload["rows"])


def test_run_parameter_sweep_crowd_exit_axis():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        crowd_exit_ks=[0, 2],
        price_cache={"A": [{"trade_date": "20250101", "open": 10, "high": 10, "close": 10}]},
    )
    ks = {row.get("crowd_exit_k") for row in payload["rows"]}
    assert ks == {0, 2}
    assert payload["crowd_exit_ks"] == [0, 2]


def test_run_parameter_sweep_non_bull_action_axis():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        regime_modes=["bull_g60_else_cash"],
        cash_on_dates=["20250102"],
        non_bull_actions=["cash", "half_topn", "half_topn_slot"],
    )
    actions = {row.get("non_bull_action") for row in payload["rows"]}
    assert actions == {"cash", "half_topn", "half_topn_slot"}
    assert payload["non_bull_actions"] == ["cash", "half_topn", "half_topn_slot"]
    cash_row = next(
        row
        for row in payload["rows"]
        if row.get("non_bull_action") == "cash" and row.get("construction_mode") == "top_n"
    )
    half_row = next(
        row
        for row in payload["rows"]
        if row.get("non_bull_action") == "half_topn" and row.get("construction_mode") == "top_n"
    )
    assert float(cash_row["average_selected_count"]) < float(half_row["average_selected_count"])


def test_normalize_path_gates_defaults_to_off():
    assert normalize_path_gates() == ["off"]
    assert normalize_path_gates(["off", "ret20_pos"]) == ["off", "ret20_pos"]


def test_resolve_regime_rules_dedupes_and_defaults():
    from analysis.sweep.params import normalize_regime_rules, count_regime_mode_arms

    assert normalize_regime_rules(["k_slow_anchor", "k_slow_anchor"]) == ["k_slow_anchor"]
    assert normalize_regime_rules(None, regime_rule="k_slow_anchor_with_volume") == [
        "k_slow_anchor_with_volume"
    ]
    assert normalize_regime_rules(None) == ["k_slow_anchor"]
    assert count_regime_mode_arms(["off", "bull_g60_else_cash"], ["k_slow_anchor", "k_slow_anchor_with_vol_and_atr"]) == 3


def test_resolve_regime_modes_treats_cash_flag_as_first_class():
    assert resolve_regime_modes() == ["off"]
    assert resolve_regime_modes(regime_cash=True) == ["bull_g60_else_cash"]
    assert resolve_regime_modes(regime_modes=["bull_g60_else_cash"]) == ["bull_g60_else_cash"]
    assert resolve_regime_modes(regime_modes=["off", "cash"], regime_cash=True) == [
        "off",
        "bull_g60_else_cash",
    ]


def test_run_parameter_sweep_path_gate_axis():
    dataset = _sample_dataset().copy()
    dataset["ret_20"] = [0.05, -0.02, 0.01, -0.03, 0.04, -0.01, 0.02, 0.03]
    dataset["above_ma60"] = 1.0
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        path_gates=["off", "ret20_pos"],
    )
    gates = {row.get("path_gate") for row in payload["rows"]}
    assert gates == {"off", "ret20_pos"}
    assert payload["path_gates"] == ["off", "ret20_pos"]


def test_run_parameter_sweep_includes_trailing_stop_pct():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        trailing_stop_pct=0.15,
    )
    assert all(row.get("trailing_stop_pct") == 0.15 for row in payload["rows"])


def test_run_parameter_sweep_passes_volatility_column_to_rows():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        vol_variants=build_volatility_variants(["cash_drag"], [0.30]),
        volatility_column="vol_60d_annual",
    )

    assert payload["volatility_column"] == "vol_60d_annual"
    assert all(row["volatility_column"] == "vol_60d_annual" for row in payload["rows"])


def test_run_parameter_sweep_handles_growth_cycle_weight_dimension():
    sweeps = parse_growth_cycle_weights(["20:80", "80:20"])
    dataset = add_growth_cycle_weight_columns(_sample_dataset(), sweeps)
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=sweeps,
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
    )

    score_variants = {row["score_variant"] for row in payload["rows"]}

    assert score_variants == {"growth_cycle_g20_c80", "growth_cycle_g80_c20"}
    assert all(row["growth_weight"] in (0.2, 0.8) for row in payload["rows"])
    assert {row["score_variant"] for row in payload["stability"]} == score_variants


def test_summarize_stability_groups_by_variant_and_topn():
    rows = [
        {"variant": "a", "top_n": 10, "cumulative_return": 0.2, "index_excess_cumulative_return": 0.1, "max_drawdown": -0.05, "avg_active_share": 0.3},
        {"variant": "a", "top_n": 10, "cumulative_return": 0.1, "index_excess_cumulative_return": 0.05, "max_drawdown": -0.1, "avg_active_share": 0.2},
    ]

    stability = summarize_stability(rows)

    assert len(stability) == 1
    assert stability[0]["observations"] == 2
    assert round(stability[0]["avg_index_excess_return"], 4) == 0.075
    assert stability[0]["worst_max_drawdown"] == -0.1


def test_summarize_stability_keeps_rows_whose_axis_value_is_none():
    """分组键为 None（如不设止盈）的行必须留在表里，且键要还原成 None 而不是 NaN。"""
    rows = [
        {"variant": "a", "top_n": 10, "trailing_stop_pct": None, "cumulative_return": 0.2, "index_excess_cumulative_return": 0.1, "max_drawdown": -0.05, "avg_active_share": 0.3},
        {"variant": "a", "top_n": 10, "trailing_stop_pct": 0.08, "cumulative_return": 0.05, "index_excess_cumulative_return": 0.02, "max_drawdown": -0.08, "avg_active_share": 0.3},
    ]
    axes = [{"key": "trailing_stop_pct", "label": "浮动止盈", "values": [None, 0.08]}]

    stability = summarize_stability(rows, sweep_axes=axes)

    by_stop = {row["trailing_stop_pct"]: row for row in stability}
    assert set(by_stop) == {None, 0.08}
    assert by_stop[None]["avg_index_excess_return"] == 0.1
    # NaN 会被 Starlette 的 JSONResponse(allow_nan=False) 打成 500，必须序列化得过去
    json.dumps(stability, allow_nan=False)


def test_render_html_contains_ranking_and_stability_sections():
    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([0.5]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
    )

    html = render_html(payload, "Sweep Test", "20250101", "20250103", index_buy_hold_return=0.02)

    assert "综合排行" in html
    assert "稳定性汇总" in html
    assert "growth_cycle_neutral" in html
    assert "Avg Active Share" in html
    assert payload.get("sweep_view") is not None


def test_render_html_includes_trailing_stop_axis_when_swept():
    rows = [
        {
            "score_variant": "growth_cycle_g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": None,
            "cumulative_return": 0.2,
            "index_excess_cumulative_return": 0.1,
            "sharpe": 1.1,
            "max_drawdown": -0.1,
            "average_turnover": 0.2,
            "risk_adjusted_score": 0.8,
        },
        {
            "score_variant": "growth_cycle_g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": 0.15,
            "cumulative_return": 0.15,
            "index_excess_cumulative_return": 0.05,
            "sharpe": 1.0,
            "max_drawdown": -0.12,
            "average_turnover": 0.18,
            "risk_adjusted_score": 0.6,
        },
    ]
    from analysis.sweep.sweep_result_view import build_sweep_result_view

    payload = {
        "horizon": 20,
        "rebalance_intervals": [20],
        "rows": rows,
        "stability": [],
    }
    payload["sweep_view"] = build_sweep_result_view(payload, {"trailing_stop_pcts": [0.0, 0.15]})
    html = render_html(payload, "Trailing Stop Sweep", "20250101", "20250103")

    assert "实验设计" in html
    assert "分面最优对比" in html
    assert "浮动止盈" in html
    assert "15%" in html


def test_rows_table_omits_identity_columns_already_in_sweep_axes():
    from analysis.sweep.portfolio_sweep_report import _rows_table
    from analysis.sweep.sweep_result_view import build_sweep_result_view

    rows = [
        {
            "score_variant": "value_score",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "max_industry_weight": 0.3,
            "rebalance_interval_days": 40,
            "trailing_stop_pct": None,
            "cumulative_return": 0.2,
            "annualized_return": 0.08,
            "index_excess_cumulative_return": 0.1,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "average_turnover": 0.2,
            "risk_adjusted_score": 0.5,
        },
        {
            "score_variant": "value_score",
            "variant": "growth_cycle_topn",
            "top_n": 20,
            "max_industry_weight": 0.25,
            "rebalance_interval_days": 40,
            "trailing_stop_pct": 0.15,
            "cumulative_return": 0.15,
            "annualized_return": 0.06,
            "index_excess_cumulative_return": 0.05,
            "sharpe": 0.9,
            "max_drawdown": -0.12,
            "average_turnover": 0.18,
            "risk_adjusted_score": 0.4,
        },
    ]
    params = {
        "top_n_values": [10, 20],
        "active_caps": [0.3, 0.25],
        "trailing_stop_pcts": [0.0, 0.15],
    }
    sweep_view = build_sweep_result_view({"rows": rows}, params)
    html = _rows_table(rows, sweep_axes=sweep_view["sweep_axes"])

    assert html.count("<th>Top N</th>") == 1
    assert "<th>TopN</th>" not in html
    assert html.count("<th>行业上限</th>") == 1


def test_render_html_does_not_warn_for_independent_multi_horizon_sweeps():
    payload = {
        "horizon": 40,
        "rebalance_intervals": [10, 20, 30, 40],
        "rows": [
            {"horizon": interval, "rebalance_interval_days": interval}
            for interval in (10, 20, 30, 40)
        ],
        "stability": [],
    }

    html = render_html(payload, "Multi Horizon Sweep", "20250101", "20250103")

    assert "持有窗口重叠" not in html


def test_render_html_warns_for_actual_holding_window_overlap():
    payload = {
        "horizon": 20,
        "rebalance_intervals": [10, 20],
        "rows": [
            {"horizon": 20, "rebalance_interval_days": 10},
            {"horizon": 20, "rebalance_interval_days": 20},
        ],
        "stability": [],
    }

    html = render_html(payload, "Overlap Sweep", "20250101", "20250103")

    assert "持有窗口重叠" in html
    assert "horizon=20d/rebalance=10d" in html


def test_run_parameter_sweep_includes_regime_score_variant():
    from shared.regime_label import (
        DEFAULT_UNCERTAIN_GROWTH_WEIGHT,
        REGIME_SCORE_COLUMN,
        REGIME_SCORE_VARIANT,
        build_regime_weight_schedule,
    )
    from analysis.sweep.portfolio_sweep_report import add_dynamic_blend_column

    sweeps = parse_growth_cycle_weights(["20:80", "60:40"])
    dataset = add_growth_cycle_weight_columns(_sample_dataset(), sweeps)
    rebalance_dates = sorted(dataset["score_date"].astype(str).unique())
    weight_by_date, _ = build_regime_weight_schedule(_index_prices(), rebalance_dates)
    dataset = add_dynamic_blend_column(
        dataset,
        "growth_score",
        "cycle_score",
        weight_by_date,
        REGIME_SCORE_COLUMN,
        DEFAULT_UNCERTAIN_GROWTH_WEIGHT,
    )
    sweeps = list(sweeps) + [ScoreSweep(name=REGIME_SCORE_VARIANT, score_column=REGIME_SCORE_COLUMN)]
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=sweeps,
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
    )

    score_variants = {row["score_variant"] for row in payload["rows"]}
    assert score_variants == {"growth_cycle_g20_c80", "growth_cycle_g60_c40", REGIME_SCORE_VARIANT}


def test_run_parameter_sweep_regime_cash_axis():
    dataset = add_growth_cycle_weight_columns(_sample_dataset(), parse_growth_cycle_weights(["60:40"]))
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=parse_growth_cycle_weights(["60:40"]),
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        regime_modes=["off", "bull_g60_else_cash"],
        cash_on_dates=["20250102"],
    )
    modes = {row["regime_mode"] for row in payload["rows"]}
    assert modes == {"off", "bull_g60_else_cash"}
    cash_row = next(row for row in payload["rows"] if row["regime_mode"] == "bull_g60_else_cash")
    off_row = next(row for row in payload["rows"] if row["regime_mode"] == "off")
    assert cash_row["average_exposure"] < off_row["average_exposure"]


def test_run_parameter_sweep_tags_multiple_regime_rules_on_cash_rows():
    dataset = add_growth_cycle_weight_columns(_sample_dataset(), parse_growth_cycle_weights(["60:40"]))
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=parse_growth_cycle_weights(["60:40"]),
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        regime_modes=["bull_g60_else_cash"],
        regime_rules=["k_slow_anchor", "k_slow_anchor_with_volume"],
        regime_cash_on_dates_by_rule={
            "k_slow_anchor": ["20250103"],
            "k_slow_anchor_with_volume": ["20250104"],
        },
        cash_on_dates=["20250103"],
    )
    cash_rules = {
        row.get("regime_rule")
        for row in payload["rows"]
        if row.get("regime_mode") == "bull_g60_else_cash"
    }
    assert cash_rules == {"k_slow_anchor", "k_slow_anchor_with_volume"}
    axis_keys = [axis["key"] for axis in (payload.get("sweep_view") or {}).get("sweep_axes") or []]
    assert "regime_rule" in axis_keys


def test_normalize_industry_regime_scales_defaults_to_off_and_rejects_out_of_range():
    assert normalize_industry_regime_scales() == [1.0]
    assert normalize_industry_regime_scales([1.0, 0.5, 0.0, 0.5]) == [1.0, 0.5, 0.0]
    assert resolve_industry_regime_scales(industry_regime_scale=0.5) == [0.5]
    assert resolve_industry_regime_scales(industry_regime_scales=[0.0, 1.0]) == [0.0, 1.0]
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="industry_regime_scale"):
            normalize_industry_regime_scales([bad])


def test_run_parameter_sweep_expands_industry_regime_scale_axis():
    dataset = add_growth_cycle_weight_columns(
        _sample_dataset_with_industry_regime(), parse_growth_cycle_weights(["60:40"])
    )
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=parse_growth_cycle_weights(["60:40"]),
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        industry_regime_scales=[1.0, 0.0],
    )
    assert {row["industry_regime_scale"] for row in payload["rows"]} == {1.0, 0.0}
    axis_keys = [axis["key"] for axis in (payload.get("sweep_view") or {}).get("sweep_axes") or []]
    assert "industry_regime_scale" in axis_keys

    off_row = next(row for row in payload["rows"] if row["industry_regime_scale"] == 1.0)
    scaled_row = next(row for row in payload["rows"] if row["industry_regime_scale"] == 0.0)
    assert scaled_row["average_exposure"] < off_row["average_exposure"]


def test_run_parameter_sweep_regime_cash_allows_empty_cash_dates():
    dataset = add_growth_cycle_weight_columns(_sample_dataset(), parse_growth_cycle_weights(["60:40"]))
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        score_sweeps=parse_growth_cycle_weights(["60:40"]),
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        regime_modes=["off", "bull_g60_else_cash"],
        cash_on_dates=[],
    )
    cash_row = next(row for row in payload["rows"] if row["regime_mode"] == "bull_g60_else_cash")
    off_row = next(row for row in payload["rows"] if row["regime_mode"] == "off")
    assert cash_row["cumulative_return"] == off_row["cumulative_return"]
    assert cash_row["average_exposure"] == off_row["average_exposure"]
