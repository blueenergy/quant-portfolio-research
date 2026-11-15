from analysis.sweep.sweep_result_view import (
    build_sweep_result_view,
    combo_key_from_dimensions,
    format_axis_value,
    infer_sweep_axes,
    normalize_axis_value,
)


def _sample_rows():
    return [
        {
            "score_variant": "growth_cycle_g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": None,
            "risk_adjusted_score": 0.8,
        },
        {
            "score_variant": "growth_cycle_g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": 0.1,
            "risk_adjusted_score": 0.6,
        },
        {
            "score_variant": "growth_cycle_g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": 0.15,
            "risk_adjusted_score": 0.7,
        },
        {
            "score_variant": "growth_cycle_g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": 0.2,
            "risk_adjusted_score": 0.5,
        },
    ]


def test_infer_sweep_axes_from_params_trailing_stop_pcts():
    params = {"trailing_stop_pcts": [0.0, 0.1, 0.15, 0.2], "top_n_values": [10]}
    axes = infer_sweep_axes(params, _sample_rows())
    keys = [axis["key"] for axis in axes]
    assert keys[0] == "trailing_stop_pct"
    assert len(axes[0]["values"]) == 4


def test_format_regime_mode_axis_values():
    assert format_axis_value("regime_mode", "off") == "关闭"
    assert format_axis_value("regime_mode", "bull_g60_else_cash") == "牛市60:40/其余空仓"
    assert format_axis_value("non_bull_action", "cash") == "非牛空仓"
    assert format_axis_value("non_bull_action", "half_topn") == "非牛Top减半满仓"
    assert format_axis_value("non_bull_action", "half_topn_slot") == "非牛Top减半留现金"
    assert format_axis_value("trailing_stop_pct", None) == "关闭"
    assert format_axis_value("trailing_stop_pct", 0.15) == "15%"
    assert format_axis_value("trailing_stop_pct", 15) == "15%"


def test_format_industry_regime_scale_axis_values():
    assert format_axis_value("industry_regime_scale", 1.0) == "关闭"
    assert format_axis_value("industry_regime_scale", 0.0) == "非牛行业清仓"
    assert format_axis_value("industry_regime_scale", 0.5) == "非牛行业×0.5"
    assert normalize_axis_value("industry_regime_scale", "0.5") == 0.5


def test_infer_sweep_axes_from_params_industry_regime_scales():
    params = {"industry_regime_scales": [1.0, 0.5, 0.0]}
    axes = infer_sweep_axes(params, _sample_rows())
    axis = next(item for item in axes if item["key"] == "industry_regime_scale")
    assert axis["values"] == [1.0, 0.5, 0.0]
    # A single value is not an axis.
    assert not [
        item
        for item in infer_sweep_axes({"industry_regime_scales": [1.0]}, _sample_rows())
        if item["key"] == "industry_regime_scale"
    ]


def test_format_industry_cap_hides_placeholder_for_top_n():
    top_n_row = {"construction_mode": "top_n", "variant": "growth_cycle_topn", "max_industry_weight": 0.3}
    capped_row = {
        "construction_mode": "industry_capped",
        "variant": "growth_cycle_active_cap20",
        "max_industry_weight": 0.2,
    }
    assert format_axis_value("max_industry_weight", 0.3) == "30%"
    assert format_axis_value("max_industry_weight", 0.3, top_n_row) == "无上限"
    assert format_axis_value("max_industry_weight", 0.2, capped_row) == "20%"
    assert format_axis_value("max_industry_weight", 0.3, {"construction_mode": "industry_neutral"}) == "无上限"


def test_combo_key_keeps_placeholder_industry_cap_slug():
    rows = [
        {
            "score_variant": "growth_cycle_g60_c40",
            "variant": "growth_cycle_topn",
            "construction_mode": "top_n",
            "top_n": 10,
            "rebalance_interval_days": 30,
            "trailing_stop_pct": 0.08,
            "max_industry_weight": 0.3,
        }
    ]
    axes = [
        {"key": "trailing_stop_pct", "label": "浮动止盈", "values": [0.08]},
        {"key": "max_industry_weight", "label": "行业上限", "values": [0.3]},
    ]
    key = combo_key_from_dimensions(rows[0], axes)
    assert "max_industry_weight=30" in key
    assert "wu-shang-xian" not in key


def test_normalize_axis_value_trailing_stop_accepts_percent_points():
    assert normalize_axis_value("trailing_stop_pct", 10) == 0.1


def test_build_sweep_result_view_facet_best_per_axis():
    rows = _sample_rows()
    params = {"trailing_stop_pcts": [0.0, 0.1, 0.15, 0.2]}
    view = build_sweep_result_view({"rows": rows}, params)
    assert view["row_count"] == 4
    facet = view["facet_best"]["trailing_stop_pct"]
    assert len(facet) == 4
    assert facet[0]["row"]["risk_adjusted_score"] == 0.8


def test_combo_key_from_dimensions_includes_trailing_stop():
    rows = _sample_rows()
    params = {"trailing_stop_pcts": [0.0, 0.1, 0.15, 0.2]}
    axes = infer_sweep_axes(params, rows)
    key_off = combo_key_from_dimensions(rows[0], axes)
    key_10 = combo_key_from_dimensions(rows[1], axes)
    assert key_off != key_10
    assert "trailing_stop_pct=" in key_10


def test_normalize_axis_value_trailing_stop_zero_is_off():
    assert normalize_axis_value("trailing_stop_pct", 0) is None


def test_multi_score_axis_uses_result_variants_for_facet_matching():
    rows = [
        {
            "score_variant": "fundamental_score",
            "risk_adjusted_score": 0.8,
        },
        {
            "score_variant": "value_score",
            "risk_adjusted_score": 0.6,
        },
    ]
    params = {
        "score_specs": [
            {"mode": "column", "column": "fundamental_score"},
            {"mode": "column", "column": "value_score"},
        ],
    }

    view = build_sweep_result_view({"rows": rows}, params)

    score_axis = next(axis for axis in view["sweep_axes"] if axis["key"] == "score_variant")
    assert score_axis["values"] == ["fundamental_score", "value_score"]
    assert [item["value"] for item in view["facet_best"]["score_variant"]] == [
        "fundamental_score",
        "value_score",
    ]
