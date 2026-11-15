"""``stock_scores`` read contract shared by portfolio and research consumers.

These helpers encode how downstream roles locate and interpret score rows.
Keep write-path / scorer logic out of this module.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd
from stock_data_access.score_query import merge_score_scope, row_covers_score_scope, score_scope_filter

UniverseIndex = Union[str, Sequence[str], None]


def normalize_score_date(value: Any) -> str:
    """Normalize caller dates to YYYYMMDD (API may pass hyphenated values)."""
    return str(value).replace("-", "").replace("/", "").strip()[:8]


def resolve_score_date(stock_scores, base_date: str, universe_index: str) -> str:
    """Return the latest ``score_date`` at or before ``base_date`` for ``universe_index``.

    Does not fall back to untagged or other-index rows — a missing universe
    must surface as empty rather than silently using another board's date.
    Raises ``ValueError`` when nothing is found in scope.
    """
    normalized = normalize_score_date(base_date)
    row = stock_scores.find_one(
        merge_score_scope({"score_date": {"$lte": normalized}}, universe_index),
        sort=[("score_date", -1)],
        projection={"_id": 0, "score_date": 1},
    )
    if not row:
        raise ValueError(f"No stock_scores found for {universe_index} at or before {base_date}")
    return str(row["score_date"])


def latest_stock_score_date(
    stock_scores,
    end_date: str,
    universe_index: UniverseIndex = None,
) -> Optional[str]:
    """Return the latest ``score_date`` at or before ``end_date``, or ``None``.

    Unlike ``resolve_score_date``, this does not fall back to untagged rows and
    does not raise when empty — matching research watermark / coverage checks.
    """
    if stock_scores is None:
        return None
    normalized = normalize_score_date(end_date)
    query: Dict[str, Any] = merge_score_scope(
        {"score_date": {"$lte": normalized}},
        universe_index,
    )
    row = stock_scores.find_one(
        query,
        projection={"_id": 0, "score_date": 1},
        sort=[("score_date", -1)],
    )
    return str(row["score_date"]) if row and row.get("score_date") else None


def extract_composite_value(value: Any, key: str) -> Optional[float]:
    """Pull a numeric strategy score from a ``composite_score`` dict payload."""
    if isinstance(value, dict):
        try:
            return float(value.get(key))
        except (TypeError, ValueError):
            return None
    return None


def compute_score_series(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Derive the ranking series used by plan generation from score rows.

    Supported ``config`` keys:
    - ``score_type``: ``growth_cycle_weighted`` (default), ``weighted_score``, or column-based
    - ``growth_weight`` / ``cycle_weight`` for the growth-cycle blend
    - ``score_weights``: ``{"dimension": weight, ...}`` for ``weighted_score`` —
      raw blend over atomic ``{dimension}_score`` columns with available-weight
      renormalization, mirroring research ``add_weighted_score_columns``
      (non rank-based mode)
    - ``score_column``: column name, or ``composite_score.<strategy>``
    """
    score_type = str(config.get("score_type") or "growth_cycle_weighted")
    if score_type == "growth_cycle_weighted":
        growth = pd.to_numeric(df.get("growth_score"), errors="coerce")
        cycle = pd.to_numeric(df.get("cycle_score"), errors="coerce")
        # ``dict.get(key, default)`` does not apply the default when the key is
        # present with ``None``. Research presets for score-column models carry
        # explicit ``growth_weight: null`` / ``cycle_weight: null``; treat those
        # as "use the template default" instead of crashing on float(None).
        growth_weight = config.get("growth_weight")
        growth_weight = 0.3 if growth_weight is None else float(growth_weight)
        cycle_weight = config.get("cycle_weight")
        cycle_weight = 0.7 if cycle_weight is None else float(cycle_weight)
        available_weight = growth.notna().astype(float) * growth_weight + cycle.notna().astype(float) * cycle_weight
        weighted_sum = growth.fillna(0) * growth_weight + cycle.fillna(0) * cycle_weight
        return weighted_sum.where(available_weight > 0) / available_weight.where(available_weight > 0)
    if score_type == "weighted_score":
        weights = config.get("score_weights") or {}
        available_weight = pd.Series(0.0, index=df.index)
        weighted_sum = pd.Series(0.0, index=df.index)
        for dimension, weight in weights.items():
            values = pd.to_numeric(df.get(f"{dimension}_score"), errors="coerce")
            available_weight = available_weight + values.notna().astype(float) * float(weight)
            weighted_sum = weighted_sum + values.fillna(0) * float(weight)
        return weighted_sum.where(available_weight > 0) / available_weight.where(available_weight > 0)
    score_column = str(config.get("score_column") or "score_value")
    if score_column.startswith("composite_score."):
        key = score_column.split(".", 1)[1]
        return df["composite_score"].apply(lambda value: extract_composite_value(value, key))
    return pd.to_numeric(df.get(score_column), errors="coerce")
