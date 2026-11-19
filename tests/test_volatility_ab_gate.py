from analysis.research.volatility_ab_gate import evaluate_volatility_ab, preset_patch_for_gate


def _row(mode, sharpe, drawdown, excess, target_vol=None, variant="growth_cycle_topn", top_n=10):
    return {
        "score_variant": "composite_growth_cycle_score",
        "variant": variant,
        "top_n": top_n,
        "rebalance_interval_days": 20,
        "volatility_mode": mode,
        "volatility_variant": f"vol_{mode}",
        "target_vol_annual": target_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "index_excess_cumulative_return": excess,
        "average_exposure": 0.8 if mode != "off" else 1.0,
    }


def test_evaluate_volatility_ab_passes_when_candidate_improves_risk_metrics():
    rows = [
        _row("off", 1.0, -0.20, 0.10),
        _row("cash_drag", 1.2, -0.15, 0.09, target_vol=0.30),
        _row("renormalize", 1.1, -0.14, 0.08, target_vol=0.30),
    ]

    decision = evaluate_volatility_ab(rows)

    assert decision.passed is True
    assert decision.recommended_mode == "cash_drag"
    assert decision.recommended_target_vol == 0.30
    assert preset_patch_for_gate(decision) == {
        "volatility_position_scaling_enabled": True,
        "target_vol_annual": 0.30,
    }


def test_evaluate_volatility_ab_fails_when_drawdown_worsens():
    rows = [
        _row("off", 1.0, -0.15, 0.10),
        _row("cash_drag", 1.3, -0.20, 0.11, target_vol=0.30),
    ]

    decision = evaluate_volatility_ab(rows)

    assert decision.passed is False
    assert preset_patch_for_gate(decision) == {}
