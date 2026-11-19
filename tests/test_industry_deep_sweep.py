"""Tests for industry deep sweep summarizer."""

from analysis.research.run_industry_deep_sweep import summarize_industry_deep_rows


def test_summarize_industry_deep_rows_picks_best_by_gate_and_weight():
    rows = [
        {
            "industry_l1": "801730.SI",
            "score_variant": "growth_cycle_g30_c70",
            "growth_weight": 0.3,
            "cycle_weight": 0.7,
            "path_gate": "off",
            "top_n": 8,
            "risk_adjusted_score": 1.0,
            "index_excess_cumulative_return": 0.10,
        },
        {
            "industry_l1": "801730.SI",
            "score_variant": "growth_cycle_g50_c50",
            "growth_weight": 0.5,
            "cycle_weight": 0.5,
            "path_gate": "off",
            "top_n": 8,
            "risk_adjusted_score": 1.5,
            "index_excess_cumulative_return": 0.12,
        },
        {
            "industry_l1": "801730.SI",
            "score_variant": "growth_cycle_g30_c70",
            "growth_weight": 0.3,
            "cycle_weight": 0.7,
            "path_gate": "ret20_pos",
            "top_n": 10,
            "risk_adjusted_score": 0.8,
            "index_excess_cumulative_return": 0.05,
        },
    ]
    summary = summarize_industry_deep_rows(rows, industry_l1="801730.SI")
    assert summary.row_count == 3
    assert summary.best_overall["risk_adjusted_score"] == 1.5
    assert summary.best_by_path_gate["off"]["index_excess_cumulative_return"] == 0.12
    assert summary.best_by_path_gate["ret20_pos"]["top_n"] == 10
    assert summary.best_by_growth_weight["g50_c50"]["risk_adjusted_score"] == 1.5
