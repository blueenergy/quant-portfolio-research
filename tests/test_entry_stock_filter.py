"""Tests for entry_stock_filter sweep helper."""

from __future__ import annotations

import pandas as pd

from analysis.sweep.entry_stock_filter import (
    ENTRY_STOCK_FILTER_OFF,
    apply_entry_stock_filter,
    normalize_entry_stock_filters,
)


def test_normalize_entry_stock_filters_defaults_and_dedup() -> None:
    assert normalize_entry_stock_filters(None) == [ENTRY_STOCK_FILTER_OFF]
    assert normalize_entry_stock_filters([]) == [ENTRY_STOCK_FILTER_OFF]
    assert normalize_entry_stock_filters(["distrib05", "off", "distrib05"]) == [
        "distrib05",
        "off",
    ]


def test_apply_entry_stock_filter_off_is_noop() -> None:
    dataset = pd.DataFrame(
        {
            "symbol": ["A"],
            "score_date": ["20240101"],
            "composite_growth_cycle_score": [1.0],
        }
    )
    out = apply_entry_stock_filter(
        None,
        dataset,
        "composite_growth_cycle_score",
        ENTRY_STOCK_FILTER_OFF,
        "20240101",
        "20241231",
    )
    assert out["composite_growth_cycle_score"].iloc[0] == 1.0


def test_apply_entry_stock_filter_distrib05_masks() -> None:
    dataset = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "score_date": ["20240101", "20240101"],
            "composite_growth_cycle_score": [90.0, 80.0],
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "trade_date": ["20240101", "20240101"],
            "hipos120": [0.5, 0.5],
            "distrib": [0.06, 0.01],
        }
    )
    out = apply_entry_stock_filter(
        None,
        dataset,
        "composite_growth_cycle_score",
        "distrib05",
        "20240101",
        "20241231",
        factor_panel=panel,
    )
    assert pd.isna(out.loc[out["symbol"] == "A"].iloc[0]["composite_growth_cycle_score"])
    assert out.loc[out["symbol"] == "B"].iloc[0]["composite_growth_cycle_score"] == 80.0
