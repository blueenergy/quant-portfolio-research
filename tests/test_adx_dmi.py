"""Unit tests for Wilder ADX/DMI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.market_indicators import compute_adx_dmi


def _trending_ohlc(n: int = 80) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = pd.RangeIndex(n)
    close = pd.Series(100.0 + np.arange(n) * 0.5, index=idx, dtype=float)
    high = close + 0.3
    low = close - 0.2
    return high, low, close


def test_adx_rises_in_uptrend() -> None:
    high, low, close = _trending_ohlc()
    frame = compute_adx_dmi(high, low, close, period=14)
    valid = frame["adx"].dropna()
    assert len(valid) > 20
    assert float(valid.iloc[-1]) > float(valid.iloc[len(valid) // 2])


def test_flat_market_lower_adx_than_trend() -> None:
    n = 80
    idx = pd.RangeIndex(n)
    close = pd.Series(100.0, index=idx)
    high = close + 0.1
    low = close - 0.1
    flat = compute_adx_dmi(high, low, close, period=14)["adx"].dropna()
    _, _, trend_close = _trending_ohlc(n)
    trend = compute_adx_dmi(
        trend_close + 0.3, trend_close - 0.2, trend_close, period=14
    )["adx"].dropna()
    assert float(trend.iloc[-1]) > float(flat.iloc[-1])


def test_future_data_does_not_change_past_adx() -> None:
    high, low, close = _trending_ohlc(60)
    full = compute_adx_dmi(high, low, close, period=14)
    truncated = compute_adx_dmi(high.iloc[:40], low.iloc[:40], close.iloc[:40], period=14)
    past = full.iloc[:40]
    diff = (past["adx"] - truncated["adx"]).abs().max()
    assert diff < 1e-9 or np.isnan(diff)


def test_di_direction_in_uptrend() -> None:
    high, low, close = _trending_ohlc()
    frame = compute_adx_dmi(high, low, close, period=14)
    tail = frame.dropna().tail(5)
    assert (tail["plus_di"] > tail["minus_di"]).all()
