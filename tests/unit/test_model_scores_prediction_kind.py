"""prediction_kind separates a model's research and shadow streams."""

from __future__ import annotations

import pytest

from analysis.core import model_scores


def _fake_client(captured, groups=None):
    class Collection:
        def find(self, query, projection=None):
            captured.update(query)
            return []

        def aggregate(self, pipeline):
            return iter(list(groups or []))

    class Database:
        def __getitem__(self, name):
            return Collection()

    class Client:
        def __init__(self, uri):
            pass

        def __getitem__(self, name):
            return Database()

        def close(self):
            pass

    return Client


def test_research_kind_excludes_shadow_rows(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured))
    frame = model_scores.load_model_scores_range(
        "m1", prediction_kind="research", enabled=True, mongo_uri="mongodb://x"
    )
    assert frame.empty
    assert captured["model_id"] == "m1"
    assert captured["$or"] == [
        {"research": True},
        {"research": {"$ne": True}, "shadow": {"$ne": True}},
    ]
    assert "shadow" not in captured or captured["shadow"] is not True


def test_shadow_kind_selects_only_shadow_rows(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured))
    model_scores.load_model_scores_range(
        "m1", prediction_kind="shadow", enabled=True, mongo_uri="mongodb://x"
    )
    assert captured["shadow"] is True
    assert "$or" not in captured


def test_legacy_none_keeps_all_rows(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured))
    model_scores.load_model_scores_range("m1", enabled=True, mongo_uri="mongodb://x")
    assert captured == {"model_id": "m1"}


def test_invalid_kind_raises(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured))
    with pytest.raises(ValueError):
        model_scores.load_model_scores_range(
            "m1", prediction_kind="live", enabled=True, mongo_uri="mongodb://x"
        )


def test_overlapping_sources_raise(monkeypatch):
    captured = {}
    groups = [
        {"_id": "source_a", "first": "20260101", "last": "20260601"},
        {"_id": "source_b", "first": "20260501", "last": "20261201"},
    ]
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured, groups))
    with pytest.raises(ValueError, match="ambiguous prediction sources"):
        model_scores.load_model_scores_range(
            "m1",
            prediction_kind="research",
            enabled=True,
            mongo_uri="mongodb://x",
        )


def test_disjoint_sources_are_allowed(monkeypatch):
    captured = {}
    groups = [
        {"_id": "source_a", "first": "20170424", "last": "20241119"},
        {"_id": "source_b", "first": "20241217", "last": "20260909"},
    ]
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured, groups))
    frame = model_scores.load_model_scores_range(
        "m1",
        prediction_kind="research",
        enabled=True,
        mongo_uri="mongodb://x",
    )
    assert frame.empty


def test_prediction_source_selects_one_source(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_scores, "MongoClient", _fake_client(captured))
    model_scores.load_model_scores_range(
        "m1",
        prediction_kind="research",
        prediction_source="source_a",
        enabled=True,
        mongo_uri="mongodb://x",
    )
    assert captured["source"] == "source_a"
