"""Unit tests for shared stock_scores read contract (Phase 0)."""

from __future__ import annotations

import pandas as pd
import pytest

from shared.scores import (
    compute_score_series,
    extract_composite_value,
    latest_stock_score_date,
    normalize_score_date,
    resolve_score_date,
)


class _Cursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def __iter__(self):
        return iter(self.docs)


class _Collection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])

    def find_one(self, query=None, projection=None, sort=None):
        docs = list(self.find(query, projection))
        if sort:
            for key, direction in reversed(sort):
                docs = sorted(docs, key=lambda doc: doc.get(key) or "", reverse=direction < 0)
        return docs[0] if docs else None

    def find(self, query=None, projection=None):
        rows = [self._project(doc, projection) for doc in self.docs if self._matches(doc, query or {})]
        return _Cursor(rows)

    def _project(self, doc, projection):
        if not projection:
            return dict(doc)
        out = {}
        include_id = projection.get("_id", 1)
        for key, flag in projection.items():
            if key == "_id":
                continue
            if flag:
                out[key] = doc.get(key)
        if include_id:
            out["_id"] = doc.get("_id")
        return out

    def _matches(self, doc, query):
        for key, expected in query.items():
            actual = doc.get(key)
            if isinstance(expected, dict):
                if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                    return False
                if "$in" in expected:
                    allowed = list(expected["$in"])
                    if isinstance(actual, list):
                        if not any(item in allowed for item in actual):
                            return False
                    elif actual not in allowed:
                        return False
                continue
            if isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
        return True


def test_normalize_score_date_strips_separators():
    assert normalize_score_date("2025-01-07") == "20250107"
    assert normalize_score_date("2025/01/07") == "20250107"
    assert normalize_score_date("20250107") == "20250107"


def test_resolve_score_date_prefers_universe_and_does_not_fallback():
    coll = _Collection(
        [
            {"score_date": "20250105", "index_codes": ["hs300"]},
            {"score_date": "20250106", "index_codes": ["csi1000"]},
            {"score_date": "20250104", "index_codes": ["csi1000"]},
        ]
    )
    assert resolve_score_date(coll, "2025-01-07", "csi1000") == "20250106"
    with pytest.raises(ValueError, match="csi500"):
        resolve_score_date(coll, "20250107", "csi500")


def test_resolve_score_date_raises_when_empty():
    with pytest.raises(ValueError, match="No stock_scores"):
        resolve_score_date(_Collection(), "20250101", "csi1000")


def test_latest_stock_score_date_no_fallback_and_optional():
    coll = _Collection(
        [
            {"score_date": "20250106", "index_codes": ["hs300"]},
            {"score_date": "20250105", "index_codes": ["csi1000"]},
        ]
    )
    assert latest_stock_score_date(coll, "20250107", ["csi1000"]) == "20250105"
    assert latest_stock_score_date(coll, "20250107", ["csi500"]) is None
    assert latest_stock_score_date(None, "20250107", ["csi1000"]) is None


def test_extract_composite_value():
    assert extract_composite_value({"growth_oriented": "12.5"}, "growth_oriented") == 12.5
    assert extract_composite_value({"growth_oriented": None}, "growth_oriented") is None
    assert extract_composite_value(80.0, "growth_oriented") is None


def test_compute_score_series_growth_cycle_weighted():
    df = pd.DataFrame(
        {
            "growth_score": [100.0, None],
            "cycle_score": [50.0, 80.0],
        }
    )
    series = compute_score_series(df, {"score_type": "growth_cycle_weighted", "growth_weight": 0.3, "cycle_weight": 0.7})
    assert series.iloc[0] == pytest.approx(0.3 * 100 + 0.7 * 50)
    assert series.iloc[1] == pytest.approx(80.0)


def test_compute_score_series_growth_cycle_none_weights_use_defaults():
    df = pd.DataFrame(
        {
            "growth_score": [100.0, None],
            "cycle_score": [50.0, 80.0],
        }
    )
    series = compute_score_series(
        df,
        {"score_type": "growth_cycle_weighted", "growth_weight": None, "cycle_weight": None},
    )
    assert series.iloc[0] == pytest.approx(0.3 * 100 + 0.7 * 50)
    assert series.iloc[1] == pytest.approx(80.0)


def test_compute_score_series_weighted_score():
    df = pd.DataFrame(
        {
            "value_score": [40.0, None, 40.0],
            "fundamental_score": [80.0, 60.0, None],
        }
    )
    series = compute_score_series(
        df,
        {"score_type": "weighted_score", "score_weights": {"value": 0.5, "fundamental": 0.5}},
    )
    assert series.iloc[0] == pytest.approx(0.5 * 40 + 0.5 * 80)
    # missing dimension renormalizes onto the available weight
    assert series.iloc[1] == pytest.approx(60.0)
    assert series.iloc[2] == pytest.approx(40.0)


def test_compute_score_series_weighted_score_renormalizes_away_all_missing():
    df = pd.DataFrame({"value_score": [None], "fundamental_score": [None]})
    series = compute_score_series(
        df,
        {"score_type": "weighted_score", "score_weights": {"value": 0.5, "fundamental": 0.5}},
    )
    assert pd.isna(series.iloc[0])


def test_compute_score_series_weighted_score_empty_weights_is_all_nan():
    df = pd.DataFrame({"value_score": [40.0], "fundamental_score": [80.0]})
    series = compute_score_series(df, {"score_type": "weighted_score", "score_weights": {}})
    assert pd.isna(series.iloc[0])


def test_compute_score_series_composite_column():
    df = pd.DataFrame({"composite_score": [{"cycle_oriented": 66.0}, {"cycle_oriented": 70.0}]})
    series = compute_score_series(df, {"score_type": "column", "score_column": "composite_score.cycle_oriented"})
    assert list(series) == [66.0, 70.0]


