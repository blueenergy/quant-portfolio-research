"""Tests for ADX entry-risk exposure schedule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.research.index_adx_entry_risk import (
    EXPOSURE_FULL,
    EXPOSURE_OFF,
    apply_hysteresis,
    exposure_schedule_t_plus_one,
    raw_exposure_from_rules,
)


def test_hysteresis_delays_drop() -> None:
    target = pd.Series([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    out = apply_hysteresis(target, confirm_days=2, exit_confirm_days=2)
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 0.0


def test_exposure_schedule_shifts_one_day() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "di_bear_strengthening": [False, True, True],
            "adx_peak_fade": [False, False, False],
            "overheat_fade": [False, False, False],
        }
    )
    raw = raw_exposure_from_rules(frame)
    assert raw.iloc[1] == EXPOSURE_OFF
    frame2 = frame.copy()
    for col in ("plus_di", "minus_di", "adx", "adx_rising", "adx_peak_10", "di_spread_narrow_3d", "hipos120", "ema60_dist_high"):
        if col not in frame2:
            frame2[col] = False if col.endswith("_narrow_3d") or col.endswith("_rising") or col.endswith("_high") else 1.0
    frame2["di_bear_strengthening"] = [False, True, True]
    frame2["adx_peak_fade"] = False
    frame2["overheat_fade"] = False
    frame2["minus_di"] = [10, 30, 30]
    frame2["plus_di"] = [20, 10, 10]
    frame2["adx"] = [20, 25, 26]
    frame2["adx_rising"] = [False, True, True]
    sched = exposure_schedule_t_plus_one(frame2)
    assert sched.iloc[0] == EXPOSURE_FULL
