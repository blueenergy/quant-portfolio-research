"""Tests for generic single-column and weighted score specifications."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.core.score_spec import (
    canonical_score_weights,
    parse_score_spec,
    weighted_score_name,
)
from analysis.sweep.score_columns import (
    add_weighted_score_columns,
    build_score_sweeps,
)
from analysis.sweep.sweep_engine import combo_param_hash


def test_parse_column_and_weighted_score_specs():
    column = parse_score_spec("fundamental_score")
    weighted = parse_score_spec("growth:30,cycle:70")

    assert column.mode == "column"
    assert column.column == "fundamental_score"
    assert weighted.mode == "weighted"
    assert weighted.weights == {"cycle": 0.7, "growth": 0.3}


def test_weighted_score_spec_validation():
    with pytest.raises(ValueError, match="Unsupported score dimension"):
        parse_score_spec("unknown:1")
    with pytest.raises(ValueError, match="non-negative"):
        parse_score_spec("growth:-1,cycle:2")
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_score_sweeps(
            "fundamental_score",
            ["30:70"],
            [{"mode": "column", "column": "value_score"}],
        )


def test_weighted_score_columns_normalize_available_dimensions():
    sweeps = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=[{"mode": "weighted", "weights": {"fundamental": 1, "value": 1}}],
    )
    dataset = pd.DataFrame(
        [
            {"fundamental_score": 80, "value_score": 40},
            {"fundamental_score": 60, "value_score": None},
        ]
    )

    out = add_weighted_score_columns(dataset, sweeps)
    column = sweeps[0].score_column
    assert out[column].tolist() == [60.0, 60.0]
    assert sweeps[0].weights == {"fundamental": 0.5, "value": 0.5}


def test_rank_weighted_spec_produces_a_distinct_rank_blend_sweep():
    sweeps = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=[{"mode": "rank_weighted", "weights": {"growth": 30, "cycle": 70}}],
    )
    assert len(sweeps) == 1
    assert sweeps[0].rank_based is True
    assert sweeps[0].name == "rankblend_c70_g30"
    assert sweeps[0].score_column == "rankblend_c70_g30_score"
    assert sweeps[0].weights == {"cycle": 0.7, "growth": 0.3}


def test_rank_weighted_blend_ranks_within_each_score_date():
    sweeps = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=[{"mode": "rank_weighted", "weights": {"growth": 1, "cycle": 1}}],
    )
    # growth spreads 10x wider than cycle, so a raw blend would be driven by growth
    # alone; ranking must give both dimensions equal say.
    dataset = pd.DataFrame(
        [
            {"score_date": "20240102", "growth_score": 10, "cycle_score": 50},
            {"score_date": "20240102", "growth_score": 90, "cycle_score": 49},
            {"score_date": "20240103", "growth_score": 90, "cycle_score": 49},
            {"score_date": "20240103", "growth_score": 10, "cycle_score": 50},
        ]
    )

    out = add_weighted_score_columns(dataset, sweeps)
    column = sweeps[0].score_column
    # Within each date each row wins one dimension and loses the other, so both tie.
    assert out[column].tolist() == [0.75, 0.75, 0.75, 0.75]


def test_rank_weighted_blend_differs_from_raw_weighted_blend():
    specs = [
        {"mode": "weighted", "weights": {"growth": 1, "cycle": 1}},
        {"mode": "rank_weighted", "weights": {"growth": 1, "cycle": 1}},
    ]
    sweeps = build_score_sweeps("composite_growth_cycle_score", score_specs=specs)
    dataset = pd.DataFrame(
        [
            {"score_date": "20240102", "growth_score": 10, "cycle_score": 70},
            {"score_date": "20240102", "growth_score": 90, "cycle_score": 60},
        ]
    )

    out = add_weighted_score_columns(dataset, sweeps)
    raw, ranked = sweeps[0].score_column, sweeps[1].score_column
    # Raw blend prefers the wide-dispersion growth name; ranks tie the two rows.
    assert out[raw].tolist() == [40.0, 75.0]
    assert out[ranked].tolist() == [0.75, 0.75]


def test_rank_weighted_blend_keeps_rows_missing_one_dimension():
    sweeps = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=[{"mode": "rank_weighted", "weights": {"growth": 1, "cycle": 1}}],
    )
    dataset = pd.DataFrame(
        [
            {"score_date": "20240102", "growth_score": 10, "cycle_score": 70},
            {"score_date": "20240102", "growth_score": 90, "cycle_score": None},
            {"score_date": "20240102", "growth_score": None, "cycle_score": None},
        ]
    )

    out = add_weighted_score_columns(dataset, sweeps)
    values = out[sweeps[0].score_column].tolist()
    assert values[0] == pytest.approx(0.75)
    assert values[1] == pytest.approx(1.0)
    assert pd.isna(values[2])


def test_weighted_score_columns_reject_missing_dimension_column():
    sweeps = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=["fundamental:1,value:1"],
    )
    with pytest.raises(ValueError, match="value_score"):
        add_weighted_score_columns(pd.DataFrame([{"fundamental_score": 80}]), sweeps)


def test_single_weight_spec_remains_weighted_blend():
    sweep = build_score_sweeps(
        "composite_growth_cycle_score",
        score_specs=["value:1"],
    )[0]
    assert sweep.name == "blend_v100"
    assert sweep.score_column == "blend_v100_score"
    assert sweep.weights == {"value": 1.0}
    assert sweep.growth_weight is None


def test_weighted_name_is_stable_and_precision_safe():
    weights = canonical_score_weights({"cycle": 7, "growth": 3})
    assert weighted_score_name(weights) == "blend_c70_g30"
    assert weighted_score_name({"growth": 0.333333, "cycle": 0.666667}).startswith(
        "blend_c66p6667_g33p3333_"
    )


def test_legacy_growth_cycle_hash_is_unchanged():
    row = {
        "score_variant": "growth_cycle_g30_c70",
        "score_column": "growth_cycle_g30_c70_score",
        "growth_weight": 0.3,
        "cycle_weight": 0.7,
        "variant": "growth_cycle_topn",
        "construction_mode": "top_n",
        "max_industry_weight": None,
        "path_gate": "off",
        "volatility_variant": "vol_off",
        "volatility_mode": "off",
        "target_vol_annual": None,
        "volatility_column": "vol_20d_annual",
        "rebalance_interval_days": 20,
        "transaction_cost": 0.001,
        "buy_commission_rate": None,
        "sell_commission_rate": None,
        "min_commission": None,
        "stamp_tax_rate": None,
        "transfer_fee_rate": None,
        "index_benchmark_symbol": "000852.SH",
        "horizon": 20,
    }
    assert combo_param_hash(row) == "a8cf9a3372912a70559512e95ecd646d"


def test_generic_weights_change_combo_hash_even_when_display_fields_match():
    base = {
        "score_variant": "blend",
        "score_column": "blend_score",
        "variant": "growth_cycle_topn",
    }
    first = combo_param_hash({**base, "score_weights": {"growth": 0.3, "cycle": 0.7}})
    second = combo_param_hash({**base, "score_weights": {"growth": 0.4, "cycle": 0.6}})
    assert first != second
