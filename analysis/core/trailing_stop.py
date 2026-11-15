"""Trailing-stop parameter normalization shared by core backtests and sweeps."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


def normalize_trailing_stop_ratio(value: Any) -> Optional[float]:
    """Return a trailing-stop ratio in (0, 1]; accept decimals or legacy percent points."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if number > 1:
        number = number / 100.0
    if number <= 0 or number > 1:
        return None
    return number


def normalize_trailing_stop_pcts(
    trailing_stop_pct: Optional[float] = None,
    trailing_stop_pcts: Optional[Sequence[float]] = None,
) -> List[Optional[float]]:
    """Return effective trailing-stop values; None/0 means disabled."""
    if trailing_stop_pcts is not None:
        values: List[Optional[float]] = []
        for raw in trailing_stop_pcts:
            values.append(normalize_trailing_stop_ratio(raw))
        return values or [None]
    if trailing_stop_pct is None:
        return [None]
    return [normalize_trailing_stop_ratio(trailing_stop_pct)]
