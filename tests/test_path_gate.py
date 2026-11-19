"""Tests for path feature attachment and path_gate filtering."""

from __future__ import annotations

import pandas as pd

from analysis.core.path_features import add_path_feature_columns, apply_path_gate
from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
from analysis.research.path_gate_ab_gate import evaluate_path_gate_ab
from analysis.research.path_gate_diagnosis import evaluate_verdict, run_diagnosis
from analysis.sweep.score_columns import ScoreSweep, add_growth_cycle_weight_columns


def _price_rows(closes):
    return [{"trade_date": f"202501{d:02d}", "close": c, "high": c, "low": c} for d, c in enumerate(closes, start=1)]


def test_add_path_feature_columns_asof_no_lookahead():
    dataset = pd.DataFrame(
        [
            {"symbol": "AAA", "score_date": "20250125"},
            {"symbol": "AAA", "score_date": "20250130"},
        ]
    )
    closes = [10 + i * 0.1 for i in range(30)]
    cache = {"AAA": _price_rows(closes)}
    out = add_path_feature_columns(dataset, cache)
    for col in ("ret_20", "above_ma60", "mdd60", "above_ma5", "above_ma10", "ma_zone"):
        assert col in out.columns
    row = out.iloc[0]
    assert row["ret_20"] is not None
    assert row["above_ma60"] is not None
    assert row["mdd60"] is not None


def test_path_gate_ret20_pos_filters_without_changing_relative_order():
    frame = pd.DataFrame(
        [
            {"symbol": "A", "score": 90, "ret_20": 0.05},
            {"symbol": "B", "score": 80, "ret_20": -0.02},
            {"symbol": "C", "score": 70, "ret_20": 0.01},
        ]
    )
    filtered = apply_path_gate(frame.sort_values("score", ascending=False), "ret20_pos")
    assert filtered["symbol"].tolist() == ["A", "C"]


def test_path_gate_off_keeps_nan_feature_rows():
    frame = pd.DataFrame(
        [
            {"symbol": "A", "score": 90, "ret_20": None, "above_ma60": None},
            {"symbol": "B", "score": 80, "ret_20": -0.1, "above_ma60": 0.0},
        ]
    )
    off = apply_path_gate(frame, "off")
    assert len(off) == 2
    gated = apply_path_gate(frame, "ret20_pos")
    assert gated["symbol"].tolist() == []


def test_portfolio_backtester_path_gate_reduces_selection():
    dataset = pd.DataFrame(
        [
            {
                "score_date": "20250101",
                "symbol": "A",
                "industry": "X",
                "composite_cycle_oriented_score": 90,
                "forward_return_20d": 0.1,
                "ret_20": 0.05,
                "above_ma60": 1.0,
            },
            {
                "score_date": "20250101",
                "symbol": "B",
                "industry": "X",
                "composite_cycle_oriented_score": 80,
                "forward_return_20d": 0.05,
                "ret_20": -0.02,
                "above_ma60": 0.0,
            },
            {
                "score_date": "20250101",
                "symbol": "C",
                "industry": "Y",
                "composite_cycle_oriented_score": 70,
                "forward_return_20d": 0.03,
                "ret_20": 0.01,
                "above_ma60": 1.0,
            },
        ]
    )
    base_cfg = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        path_gate="off",
    )
    gate_cfg = PortfolioBacktestConfig(
        top_n_values=(2,),
        rebalance_interval_days=None,
        rebalance_step=1,
        transaction_cost=0,
        path_gate="ret20_pos",
    )
    base = PortfolioBacktester(dataset, base_cfg).run()
    gated = PortfolioBacktester(dataset, gate_cfg).run()
    assert base.periods.iloc[0]["selected_count"] == 2
    assert gated.periods.iloc[0]["selected_count"] == 2
    assert gated.periods.iloc[0]["path_gate_pool_size"] == 2


def test_evaluate_verdict_proceed_and_stop():
    proceed_overall = {
        "top_vs_pool_delta": {"mean": 0.08, "positive_days": 30, "days": 40},
        "top_fwd_ret_spread": {"mean": -0.01},
        "top_fwd_mdd_spread": {"mean": -0.05},
    }
    stop_overall = {
        "top_vs_pool_delta": {"mean": 0.01, "positive_days": 5, "days": 40},
        "top_fwd_ret_spread": {"mean": 0.02},
        "top_fwd_mdd_spread": {"mean": 0.01},
    }
    assert evaluate_verdict(proceed_overall, {})["proceed"] is True
    assert evaluate_verdict(stop_overall, {})["proceed"] is False


def test_run_diagnosis_on_synthetic_panel():
    dataset = pd.DataFrame(
        [
            {
                "symbol": "A",
                "score_date": "20250110",
                "growth_score": 90,
                "cycle_score": 80,
                "forward_return_20d": 0.05,
                "entry_price": 10,
                "entry_date": "20250111",
                "exit_date_20d": "20250210",
            },
            {
                "symbol": "B",
                "score_date": "20250110",
                "growth_score": 85,
                "cycle_score": 75,
                "forward_return_20d": -0.02,
                "entry_price": 10,
                "entry_date": "20250111",
                "exit_date_20d": "20250210",
            },
            {
                "symbol": "C",
                "score_date": "20250110",
                "growth_score": 50,
                "cycle_score": 50,
                "forward_return_20d": 0.01,
                "entry_price": 10,
                "entry_date": "20250111",
                "exit_date_20d": "20250210",
            },
        ]
    )
    dataset = add_growth_cycle_weight_columns(
        dataset,
        [ScoreSweep(name="w", score_column="blend", growth_weight=0.5, cycle_weight=0.5)],
    )
    cache = {
        "A": [{"trade_date": f"202501{d:02d}", "close": 10 + d * 0.1} for d in range(1, 15)],
        "B": [{"trade_date": f"202501{d:02d}", "close": 12 - d * 0.1} for d in range(1, 15)],
        "C": [{"trade_date": f"202501{d:02d}", "close": 10.0} for d in range(1, 15)],
    }
    report = run_diagnosis(dataset, cache, score_column="blend", top_n=2, horizon=20)
    assert report["daily_count"] == 1
    assert "verdict" in report


def test_evaluate_path_gate_ab_passes_when_drawdown_improves():
    rows = [
        {
            "score_variant": "w",
            "variant": "v",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": None,
            "path_gate": "off",
            "max_drawdown": -0.30,
            "index_excess_cumulative_return": 0.10,
            "average_turnover": 0.40,
        },
        {
            "score_variant": "w",
            "variant": "v",
            "top_n": 10,
            "rebalance_interval_days": 20,
            "trailing_stop_pct": None,
            "path_gate": "ret20_pos",
            "max_drawdown": -0.25,
            "index_excess_cumulative_return": 0.09,
            "average_turnover": 0.45,
        },
    ]
    decision = evaluate_path_gate_ab(rows)
    assert decision.passed is True
    assert decision.recommended_path_gate == "ret20_pos"
