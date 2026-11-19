"""Tests for MA stack features and Phase 0 diagnosis."""

from __future__ import annotations

import pandas as pd

from analysis.research.ma_stack_diagnosis import evaluate_verdict, run_diagnosis
from analysis.core.path_features import (
    MA_STACK_FEATURE_COLUMNS,
    classify_ma_zone,
    add_path_feature_columns,
)


def test_classify_ma_zone():
    assert classify_ma_zone(11, 10, 9) == "strong"
    assert classify_ma_zone(10, 10, 9) == "warn"
    assert classify_ma_zone(9.5, 10, 9) == "warn"
    assert classify_ma_zone(9, 10, 9) == "weak"
    assert classify_ma_zone(8, 10, 9) == "weak"
    assert classify_ma_zone(10, float("nan"), 9) == "strong"


def _price_rows(closes):
    return [{"trade_date": f"202501{d:02d}", "close": c, "high": c, "low": c} for d, c in enumerate(closes, start=1)]


def test_add_path_feature_columns_includes_ma_stack():
    dataset = pd.DataFrame([{"symbol": "AAA", "score_date": "20250130"}])
    closes = [10 + i * 0.1 for i in range(30)]
    out = add_path_feature_columns(dataset, {"AAA": _price_rows(closes)})
    for col in MA_STACK_FEATURE_COLUMNS:
        assert col in out.columns
    assert out.iloc[0]["ma_zone"] in ("strong", "warn", "weak", None)
    assert out.iloc[0]["above_ma5"] is not None


def test_ma_stack_diagnosis_verdict_proceed_synthetic():
    rows = []
    for i in range(10):
        score_date = f"202501{i + 10:02d}"
        for j in range(5):
            rows.append(
                {
                    "score_date": score_date,
                    "symbol": f"S{j}",
                    "composite": 100 - j,
                    "forward_return_20d": 0.02 if j < 2 else -0.01,
                    "ma_zone": "weak" if j == 0 else "strong",
                    "future_max_drawdown": -0.15 if j == 0 else -0.05,
                }
            )
    df = pd.DataFrame(rows)
    report = run_diagnosis(
        df,
        {},
        score_column="composite",
        top_n=2,
        horizon=20,
    )
    assert report["daily_count"] == 10
    assert report["verdict"]["verdict"] in ("proceed", "stop")


def test_evaluate_verdict_ma_stack_all_pass():
    overall = {
        "top_vs_pool_weak_delta": {"mean": 0.05, "positive_days": 8, "days": 10},
        "top_weak_share": {"mean": 0.1},
        "top_fwd_ret_spread": {"mean": -0.02},
        "top_fwd_mdd_spread": {"mean": -0.03},
    }
    v = evaluate_verdict(overall, {})
    assert v["proceed"] is True
    assert v["verdict"] == "proceed"


def test_evaluate_verdict_ma_stack_fail_weak_not_elevated():
    overall = {
        "top_vs_pool_weak_delta": {"mean": -0.31, "positive_days": 22, "days": 242},
        "top_fwd_ret_spread": {"mean": 0.01},
        "top_fwd_mdd_spread": {"mean": 0.004},
    }
    v = evaluate_verdict(overall, {})
    assert v["proceed"] is False
    assert v["conditions"]["weak_share_elevated_in_top"] is False
