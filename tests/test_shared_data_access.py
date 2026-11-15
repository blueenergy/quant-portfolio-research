"""Tests for the portfolio-research read facade."""

from __future__ import annotations

import mongomock
import pytest

from shared.data_access import (
    ResearchDataAccess,
    open_research_data_access,
    open_research_data_access_for_cli,
)


def _access():
    client = mongomock.MongoClient()
    return ResearchDataAccess(client["quant_data"], app_db=client["finance"])


def test_open_research_data_access_returns_facade():
    access = _access()
    assert isinstance(access, ResearchDataAccess)
    assert access.stock_scores is not None
    assert access.money_flow is not None


def test_research_data_access_blocks_write_paths():
    access = _access()
    with pytest.raises(AttributeError, match="save_stock_score"):
        access.save_stock_score("000001.SZ", {}, score_date="20240102")


def test_get_volume_price_data_reads_market_collection():
    access = _access()
    access.data_db["volume_price"].insert_one(
        {"symbol": "000001.SZ", "trade_date": "20240102", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1}
    )

    rows = access.get_volume_price_data(
        "000001.SZ", start_date="20240101", end_date="20240103", adjust="none"
    )

    assert [row["trade_date"] for row in rows] == ["20240102"]
    assert rows[0]["close"] == 10.0


def test_get_money_flow_data_filters_by_date_and_sorts():
    access = _access()
    access.money_flow.insert_many(
        [
            {"symbol": "000001.SZ", "trade_date": "20240103", "net_amount": 3},
            {"symbol": "000001.SZ", "trade_date": "20240102", "net_amount": 2},
            {"symbol": "000001.SZ", "trade_date": "20231229", "net_amount": 1},
        ]
    )

    rows = access.get_money_flow_data("000001.SZ", start_date="20240101", end_date="20240131")

    assert [row["trade_date"] for row in rows] == ["20240102", "20240103"]


def test_index_constituent_snapshot_is_cached_and_asof_scoped(monkeypatch):
    import shared.data_access as data_access

    access = _access()
    access.index_constituents.insert_many(
        [
            {
                "index_code": "csi1000",
                "symbol": "000001.SZ",
                "weight_trade_date": "20231229",
                "update_date": "20240102",
            },
            {
                "index_code": "csi1000",
                "symbol": "000002.SZ",
                "weight_trade_date": "20231229",
                "update_date": "20240102",
            },
        ]
    )

    asof = access.get_index_constituent_symbols_asof("csi1000", "20240105")
    latest = access.get_latest_index_constituent_symbols("csi1000")

    assert sorted(asof) == ["000001.SZ", "000002.SZ"]
    assert sorted(latest) == ["000001.SZ", "000002.SZ"]

    # A later as-of date resolving to the same snapshot must reuse the cached
    # member list instead of scanning the snapshot again.
    calls = []
    original = data_access.snapshot_symbols_for_keys

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(data_access, "snapshot_symbols_for_keys", counting)
    again = access.get_index_constituent_symbols_asof("csi1000", "20240106")

    assert sorted(again) == ["000001.SZ", "000002.SZ"]
    assert calls == []


def test_open_research_data_access_for_cli_disables_tushare_by_default(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "secret")
    access = open_research_data_access_for_cli(enable_tushare=False)
    assert isinstance(access, ResearchDataAccess)
    assert "TUSHARE_TOKEN" not in __import__("os").environ


def test_open_research_data_access_for_cli_keeps_tushare_when_enabled(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "secret")
    access = open_research_data_access_for_cli(enable_tushare=True)
    assert isinstance(access, ResearchDataAccess)
    assert __import__("os").environ.get("TUSHARE_TOKEN") == "secret"


def test_open_research_data_access_accepts_explicit_dbs():
    client = mongomock.MongoClient()
    access = open_research_data_access(client["quant_data"], app_db=client["finance"])
    assert isinstance(access, ResearchDataAccess)
    assert access._app_db.name == "finance"
