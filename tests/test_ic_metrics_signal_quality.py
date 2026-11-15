"""Signal-quality block: IC / ICIR / non-overlap t / MDE80."""

import numpy as np
import pandas as pd

import pytest

from analysis.factor.ic_metrics import nonoverlap_summary, signal_quality


def _dataset():
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2023-01-02", periods=60, freq="D").strftime("%Y%m%d")
    for date in dates:
        noise = rng.normal(0.0, 3.0, size=30)
        for i in range(30):
            rows.append(
                {
                    "score_date": date,
                    "symbol": f"S{i:02d}",
                    "score": float(i),
                    "forward_return_20d": float(i) + float(noise[i]),
                }
            )
    return pd.DataFrame(rows)


def test_signal_quality_reports_ic_and_nonoverlap_metrics():
    report = signal_quality(_dataset(), ["score"], [20], min_names=20)

    block = report["score"]["h20"]
    assert block["observations"] == 60
    assert block["ic_mean"] > 0.8
    assert block["positive_ratio"] > 0.8
    assert block["nonoverlap_observations"] == 3
    assert block["t_nonoverlap"] is not None
    assert block["mde80"] is not None
    assert [row["year"] for row in block["by_year"]] == ["2023"]


def test_signal_quality_skips_missing_columns_and_insufficient_names():
    dataset = _dataset().drop(columns=["forward_return_20d"])
    assert signal_quality(dataset, ["score"], [20]) == {}

    small = _dataset().groupby("score_date").head(5)
    assert signal_quality(small, ["score"], [20], min_names=20) == {}


def test_nonoverlap_summary_samples_every_horizon_th_date():
    per_date = pd.DataFrame(
        {
            "score_date": [f"202301{d:02d}" for d in range(1, 21)],
            "rank_ic": np.arange(1.0, 21.0),
            "count": 30,
        }
    )
    block = nonoverlap_summary(per_date, 5)

    # Every 5th date -> IC 1, 6, 11, 16; mean 8.5, sample std sqrt(125/3).
    expected_std = float(np.sqrt(125.0 / 3.0))
    expected_se = expected_std / np.sqrt(4.0)
    assert block["step"] == 5
    assert block["observations"] == 4
    assert block["ic_mean"] == pytest.approx(8.5)
    assert block["t_stat"] == pytest.approx(8.5 / expected_se)
    assert block["mde80"] == pytest.approx(2.8 * expected_se)


def test_nonoverlap_summary_is_the_daily_summary_at_horizon_one():
    per_date = pd.DataFrame(
        {
            "score_date": ["20230101", "20230102", "20230103"],
            "rank_ic": [0.1, -0.2, 0.3],
            "count": 30,
        }
    )
    assert nonoverlap_summary(per_date, 1)["observations"] == 3


def test_nonoverlap_summary_handles_an_empty_series():
    block = nonoverlap_summary(pd.DataFrame(columns=["score_date", "rank_ic"]), 20)
    assert block == {
        "step": 20,
        "observations": 0,
        "ic_mean": None,
        "ic_std": None,
        "ic_ir": None,
        "t_stat": None,
        "mde80": None,
        "positive_ratio": None,
    }
