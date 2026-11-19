"""Tests for industry slice diagnosis helpers."""

import pandas as pd

from analysis.research.industry_slice_diagnosis import _evaluate_slice, evaluate_verdict


def test_evaluate_slice_skips_low_names():
    dataset = pd.DataFrame(
        {
            "symbol": ["a", "b"],
            "score_date": ["20240101", "20240101"],
            "score": [1.0, 0.5],
            "forward_return_20d": [0.01, -0.01],
        }
    )
    result = _evaluate_slice(
        dataset,
        score_column="score",
        horizon=20,
        top_k_values=[5],
        min_names=15,
        broad_benchmark_prices=[],
        slice_label="test",
    )
    assert result["skipped"]
    assert result["reason"] == "below_min_names"


def test_evaluate_verdict_proceed_when_many_material_l1():
    anchor = {
        "ic": {"ic_mean": 0.02},
        "top_k": {"8": {"mean_excess_vs_broad": 0.01}},
    }
    industries = []
    for idx in range(6):
        industries.append(
            {
                "skipped": False,
                "ic": {"ic_mean": 0.08},
                "top_k": {"8": {"mean_excess_vs_industry_index": 0.02, "mean_excess_vs_broad": 0.0}},
                "l1_code": f"80{idx:04d}.SI",
            }
        )
    verdict = evaluate_verdict(anchor, industries, primary_horizon=20, min_distinct_l1=5)
    assert verdict["verdict"] == "proceed"


def test_evaluate_verdict_stop_when_few_material_l1():
    anchor = {
        "ic": {"ic_mean": 0.02},
        "top_k": {"8": {"mean_excess_vs_broad": 0.01}},
    }
    industries = [
        {
            "skipped": False,
            "ic": {"ic_mean": 0.021},
            "top_k": {"8": {"mean_excess_vs_industry_index": 0.0105, "mean_excess_vs_broad": 0.01}},
            "l1_code": "801080.SI",
        }
    ]
    verdict = evaluate_verdict(anchor, industries, primary_horizon=20, min_distinct_l1=5)
    assert verdict["verdict"] == "stop"
