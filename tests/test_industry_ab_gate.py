"""Tests for industry A/B gate."""

from analysis.research.industry_ab_gate import evaluate_industry_ab


def test_evaluate_industry_ab_passes_when_candidate_meets_thresholds():
    rows = [
        {
            "score_variant": "g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 8,
            "rebalance_interval_days": 20,
            "index_excess_cumulative_return": 0.10,
            "max_drawdown": -0.20,
            "average_turnover": 0.50,
        },
        {
            "score_variant": "g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 8,
            "rebalance_interval_days": 20,
            "industry_l1": "801080.SI",
            "index_excess_cumulative_return": 0.09,
            "max_drawdown": -0.18,
            "average_turnover": 0.55,
            "average_selected_count": 7.5,
        },
    ]
    decision = evaluate_industry_ab(rows, industry_l1="801080.SI")
    assert decision.passed


def test_evaluate_industry_ab_fails_on_large_excess_drop():
    rows = [
        {
            "score_variant": "g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 8,
            "rebalance_interval_days": 20,
            "index_excess_cumulative_return": 0.20,
            "max_drawdown": -0.20,
            "average_turnover": 0.50,
        },
        {
            "score_variant": "g30_c70",
            "variant": "growth_cycle_topn",
            "top_n": 8,
            "rebalance_interval_days": 20,
            "industry_l1": "801080.SI",
            "index_excess_cumulative_return": 0.05,
            "max_drawdown": -0.18,
            "average_turnover": 0.55,
            "average_selected_count": 7.5,
        },
    ]
    decision = evaluate_industry_ab(rows, industry_l1="801080.SI")
    assert not decision.passed
