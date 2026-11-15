"""Direct vol/cycle percentile entry filters."""

import pandas as pd
import pytest

from analysis.sweep.entry_stock_filter import (
    apply_entry_stock_filter,
    normalize_entry_stock_filters,
    requires_entry_factor_panel,
)

SCORE = "model_score"


def _dataset(*, cycle_ascending=True):
    rows = []
    for date in ["20250101", "20250102"]:
        for i in range(10):
            rows.append(
                {
                    "symbol": f"S{i}",
                    "score_date": date,
                    SCORE: 100 - i,
                    "vol_20d_annual": 0.1 * (i + 1),
                    "cycle_score": float(i * 10 if cycle_ascending else (9 - i) * 10),
                }
            )
    return pd.DataFrame(rows)


def test_vol70_masks_the_top_thirty_percent_per_date():
    out = apply_entry_stock_filter(None, _dataset(), SCORE, "vol70", "20250101", "20250102")
    masked = set(out.loc[out[SCORE].isna(), "symbol"])
    assert masked == {"S7", "S8", "S9"}
    assert int(out[SCORE].notna().sum()) == 14


def test_cycle70_masks_high_cycle_names():
    out = apply_entry_stock_filter(None, _dataset(), SCORE, "cycle70", "20250101", "20250102")
    masked = set(out.loc[out[SCORE].isna(), "symbol"])
    assert masked == {"S7", "S8", "S9"}


def test_combined_cap_is_the_union_of_both_exposures():
    out = apply_entry_stock_filter(
        None,
        _dataset(cycle_ascending=False),
        SCORE,
        "vol70_cycle70",
        "20250101",
        "20250102",
    )
    masked = set(out.loc[out[SCORE].isna(), "symbol"])
    # vol cap removes S7-S9, cycle cap removes S0-S2.
    assert masked == {"S0", "S1", "S2", "S7", "S8", "S9"}


def test_missing_exposure_column_fails_loudly():
    dataset = _dataset().drop(columns=["vol_20d_annual"])
    with pytest.raises(ValueError, match="vol_20d_annual"):
        apply_entry_stock_filter(None, dataset, SCORE, "vol70", "20250101", "20250102")


def test_new_modes_normalize_and_do_not_require_the_legacy_factor_panel():
    assert normalize_entry_stock_filters(["vol70", "cycle80"]) == ["vol70", "cycle80"]
    assert requires_entry_factor_panel("vol70") is False
    assert requires_entry_factor_panel("hipos95") is True
