"""Tests for entry_stock_filter reconcile helpers."""

from __future__ import annotations

import pandas as pd

from analysis.sweep.entry_stock_filter import (
    compare_top_n_across_filters,
    mask_score_column_stats,
    top_n_symbols_by_score_date,
)


def test_mask_score_column_stats() -> None:
    df = pd.DataFrame({"composite": [1.0, None, 3.0]})
    stats = mask_score_column_stats(df, "composite")
    assert stats["nan_count"] == 1
    assert stats["nan_frac"] == 1 / 3


def test_top_n_and_compare_detects_diff() -> None:
    base = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score_date": ["20240101"] * 3,
            "score": [3.0, 2.0, 1.0],
        }
    )
    filtered = base.copy()
    filtered.loc[filtered["symbol"] == "A", "score"] = None
    picks = top_n_symbols_by_score_date(base, "score", ["20240101"], 2)
    assert picks["20240101"] == ["A", "B"]
    cmp = compare_top_n_across_filters(
        {"off": base, "filt": filtered},
        "score",
        ["20240101"],
        2,
    )
    assert cmp["dates_with_diff"] == 1
