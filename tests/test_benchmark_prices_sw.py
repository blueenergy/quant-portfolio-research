"""Tests for Shenwan benchmark price loading."""

import os

import mongomock
import pytest

os.environ.pop("TUSHARE_TOKEN", None)

from analysis.core.benchmark_prices import load_index_benchmark_prices


@pytest.fixture
def dm():
    from shared.data_access import ResearchDataAccess

    client = mongomock.MongoClient()
    return ResearchDataAccess(client["quant_data"], app_db=client["finance"])


def test_load_index_benchmark_prices_sw_daily(dm):
    dm.sw_daily.insert_many(
        [
            {"ts_code": "801080.SI", "trade_date": "20240101", "close": 1000.0},
            {"ts_code": "801080.SI", "trade_date": "20240102", "close": 1010.0},
        ]
    )
    rows = load_index_benchmark_prices(dm, "801080.SI", "20240101")
    assert len(rows) == 2
    assert rows[0]["trade_date"] == "20240101"
