"""Tests for the per-name industry-regime A/B gate."""

import math

import pytest

from analysis.research.industry_regime_ab_gate import (
    evaluate_industry_regime_ab,
    exposure_matched_log_excess,
    summarize_comparisons,
)


def _row(scale: float, **overrides) -> dict:
    row = {
        "score_variant": "growth_cycle_g60_c40",
        "variant": "growth_cycle_topn",
        "construction_mode": "top_n",
        "top_n": 10,
        "rebalance_interval_days": 20,
        "trailing_stop_pct": 0.15,
        "path_gate": "off",
        "volatility_variant": "vol_off",
        "regime_mode": "off",
        "regime_rule": None,
        "non_bull_action": "cash",
        "bull_exposure": 1.0,
        "crowd_exit_k": 0,
        "intra_period_refill": False,
        "industry_regime_scale": scale,
        "cumulative_return": 0.10,
        "average_exposure": 1.0,
        "index_benchmark_cumulative_return": 0.08,
        "index_excess_cumulative_return": 0.02,
        "max_drawdown": -0.20,
        "average_turnover": 0.50,
        "yearly_returns": {"2021": -0.10},
    }
    row.update(overrides)
    return row


def test_exposure_matched_log_excess_discounts_the_benchmark_by_exposure():
    row = _row(1.0, cumulative_return=0.10, average_exposure=0.5,
               index_benchmark_cumulative_return=0.08)
    assert exposure_matched_log_excess(row) == math.log1p(0.10) - 0.5 * math.log1p(0.08)


def test_exposure_matched_log_excess_returns_none_on_missing_fields():
    assert exposure_matched_log_excess({"cumulative_return": 0.1}) is None
    assert exposure_matched_log_excess(_row(1.0, average_exposure=None)) is None


def test_exact_period_matched_excess_agrees_with_closed_form_at_full_exposure():
    """At w_bar=1 the two forms are the same quantity."""
    periods = [0.10, -0.05, 0.02]
    compounded = math.prod(1 + r for r in periods) - 1
    row = _row(
        1.0,
        cumulative_return=0.20,
        average_exposure=1.0,
        index_benchmark_cumulative_return=compounded,
    )
    assert exposure_matched_log_excess(row, periods) == pytest.approx(
        exposure_matched_log_excess(row)
    )


def test_closed_form_is_optimistic_below_full_exposure():
    """The regression that motivated the exact path: a ~4pp free lunch at w=0.5."""
    periods = [0.10, -0.05, 0.02]
    compounded = math.prod(1 + r for r in periods) - 1
    row = _row(
        1.0,
        cumulative_return=0.20,
        average_exposure=0.5,
        index_benchmark_cumulative_return=compounded,
    )
    assert exposure_matched_log_excess(row) > exposure_matched_log_excess(row, periods)


def test_gate_threads_period_returns_and_reports_the_closed_form_gap():
    periods = [0.10, -0.05, 0.02]
    rows = [_row(1.0, cumulative_return=0.05), _row(0.0, cumulative_return=0.06)]
    decision = evaluate_industry_regime_ab(rows, period_index_returns=periods)
    entry = decision.comparisons[0]
    assert entry["closed_form_bias"] == pytest.approx(
        entry["candidate_matched_excess_closed_form"] - entry["candidate_matched_excess"]
    )


def test_gate_passes_when_the_overlay_improves_matched_excess():
    rows = [
        _row(1.0),
        _row(0.0, cumulative_return=0.12, max_drawdown=-0.18, yearly_returns={"2021": -0.08}),
    ]
    decision = evaluate_industry_regime_ab(rows, crash_year="2021")
    assert decision.passed
    assert decision.comparisons[0]["matched_excess_delta"] > 0
    assert decision.comparisons[0]["binding"] is True


def test_gate_fails_when_the_overlay_hurts_matched_excess():
    rows = [_row(1.0), _row(0.0, cumulative_return=0.05, max_drawdown=-0.25)]
    decision = evaluate_industry_regime_ab(rows)
    assert not decision.passed
    assert decision.comparisons[0]["matched_excess_delta"] < 0
    assert any("matched excess delta" in reason for reason in decision.reasons)


def test_zero_delta_pair_passes_but_is_flagged_as_not_binding():
    """An overlay that never binds is no evidence of harm, but proves nothing."""
    rows = [_row(1.0), _row(0.5)]
    decision = evaluate_industry_regime_ab(rows)
    assert decision.passed
    assert decision.comparisons[0]["binding"] is False
    summary = summarize_comparisons(decision.comparisons)
    assert summary[0]["binding_pairs"] == 0
    assert summary[0]["mean_matched_excess_delta_when_binding"] is None


def test_gate_fails_when_the_crash_year_worsens():
    rows = [
        _row(1.0, yearly_returns={"2021": -0.10}),
        _row(0.0, cumulative_return=0.12, yearly_returns={"2021": -0.20}),
    ]
    decision = evaluate_industry_regime_ab(rows, crash_year="2021")
    assert not decision.passed
    assert any("2021 return worsened" in reason for reason in decision.reasons)


def test_gate_fails_when_no_pairs_overlap():
    decision = evaluate_industry_regime_ab([_row(0.0)])
    assert not decision.passed
    assert "No overlapping" in decision.reasons[0]


def test_summary_averages_only_binding_pairs_separately():
    rows = [
        _row(1.0),
        _row(0.0, cumulative_return=0.12),          # binds, delta > 0
        _row(0.0, variant="other"),                 # different config -> no baseline
    ]
    decision = evaluate_industry_regime_ab(rows)
    summary = summarize_comparisons(decision.comparisons)
    assert summary[0]["pairs"] == 1
    assert summary[0]["binding_pairs"] == 1
    assert summary[0]["mean_matched_excess_delta_when_binding"] > 0
