from shared.regime_label import REGIME_SCORE_VARIANT
from analysis.research.regime_weight_ab_gate import (
    STATIC_BASELINE_2080,
    STATIC_BASELINE_6040,
    evaluate_regime_weight_ab,
)


def _row(score_variant, sharpe, drawdown, excess, turnover=0.5, top_n=20):
    return {
        "score_variant": score_variant,
        "variant": "growth_cycle_topn",
        "top_n": top_n,
        "rebalance_interval_days": 20,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "index_excess_cumulative_return": excess,
        "average_turnover": turnover,
    }


def test_evaluate_regime_weight_ab_passes_when_all_windows_meet_gates():
    full_rows = [
        _row(STATIC_BASELINE_2080, 1.0, -0.20, 0.10, turnover=0.50),
        _row(STATIC_BASELINE_6040, 1.2, -0.31, 0.20, turnover=0.55),
        _row(REGIME_SCORE_VARIANT, 1.05, -0.18, 0.095, turnover=0.55),
    ]
    bear_rows = [
        _row(STATIC_BASELINE_6040, 0.8, -0.31, -0.05, turnover=0.55),
        _row(REGIME_SCORE_VARIANT, 0.9, -0.20, 0.00, turnover=0.60),
    ]
    bull_rows = [
        _row(STATIC_BASELINE_2080, 1.0, -0.15, 0.08, turnover=0.50),
        _row(REGIME_SCORE_VARIANT, 0.98, -0.16, 0.075, turnover=0.52),
    ]
    payload = {
        "regime_schedule": [
            {"score_date": "20250101", "label": "bull", "w_growth": 0.6},
            {"score_date": "20250201", "label": "bear", "w_growth": 0.0},
        ]
    }

    decision = evaluate_regime_weight_ab(
        full_rows,
        bear_rows=bear_rows,
        bull_rows=bull_rows,
        full_payload=payload,
    )

    assert decision.passed is True
    assert decision.full_window["passed"] is True
    assert decision.bear_window["passed"] is True
    assert decision.bull_window["passed"] is True


def test_evaluate_regime_weight_ab_fails_when_bear_drawdown_not_improved():
    full_rows = [
        _row(STATIC_BASELINE_2080, 1.0, -0.20, 0.10),
        _row(REGIME_SCORE_VARIANT, 1.05, -0.18, 0.095),
    ]
    bear_rows = [
        _row(STATIC_BASELINE_6040, 0.8, -0.31, -0.05),
        _row(REGIME_SCORE_VARIANT, 0.7, -0.32, -0.06),
    ]

    decision = evaluate_regime_weight_ab(full_rows, bear_rows=bear_rows)

    assert decision.passed is False
    assert any("Algorithm bear segment failed" in reason for reason in decision.reasons)


def test_evaluate_regime_weight_ab_fails_when_regime_is_near_static_2080_without_benefit():
    full_rows = [
        _row(STATIC_BASELINE_2080, 1.0, -0.20, 0.10),
        _row(REGIME_SCORE_VARIANT, 0.90, -0.22, 0.05),
    ]
    payload = {
        "regime_schedule": [
            {"score_date": "20250101", "w_growth": 0.21},
            {"score_date": "20250201", "w_growth": 0.19},
        ]
    }

    decision = evaluate_regime_weight_ab(full_rows, full_payload=payload)

    assert decision.passed is False
    assert any("near static 20:80" in reason for reason in decision.reasons)
