from tools.process_portfolio_research_jobs import resolve_score_column


def test_model_id_wins_over_stale_rule_score_column():
    assert resolve_score_column(
        {
            "model_id": "research-rolling500-hs300",
            "score_column": "composite_growth_cycle_score",
        }
    ) == "model_score"


def test_rule_score_unchanged_without_model_id():
    assert resolve_score_column({"score_column": "value_score"}) == "value_score"
    assert resolve_score_column({}) == "composite_growth_cycle_score"
    assert resolve_score_column({"model_id": "   "}) == "composite_growth_cycle_score"
