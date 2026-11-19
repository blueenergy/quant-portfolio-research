"""Tests for ADX falsification masking."""

from __future__ import annotations

import pandas as pd

from analysis.research.adx_entry_risk_falsify import exclusion_masks, mask_dataset_scores


def test_mask_dataset_scores_nan_excluded() -> None:
    dataset = pd.DataFrame(
        {
            "symbol": ["A", "B", "A"],
            "score_date": ["20240101", "20240101", "20240201"],
            "growth_score": [90.0, 80.0, 70.0],
        }
    )
    masks = exclusion_masks(
        pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "score_date": ["20240101", "20240101"],
                "hipos120": [0.99, 0.5],
                "distrib": [0.1, 0.01],
            }
        )
    )
    out = mask_dataset_scores(dataset, "growth_score", masks, "exclude_hipos97")
    assert pd.isna(out.loc[out["symbol"] == "A"].iloc[0]["growth_score"])
    assert out.loc[out["symbol"] == "B"].iloc[0]["growth_score"] == 80.0
