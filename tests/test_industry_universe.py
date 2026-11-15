"""Tests for Shenwan L1 industry universe helpers."""

import os

import mongomock
import pandas as pd
import pytest

os.environ.pop("TUSHARE_TOKEN", None)

from shared.industry_universe import (
    SW_L1_CODE_COL,
    SW_L1_NAME_COL,
    apply_industry_filter,
    attach_sw_l1,
    resolve_industry_l1_codes,
    sw_l1_coverage_summary,
)


@pytest.fixture
def dm():
    from shared.data_access import ResearchDataAccess

    client = mongomock.MongoClient()
    return ResearchDataAccess(client["quant_data"], app_db=client["finance"])


def _member(con_code, l1_code="801080.SI", l1_name="电子"):
    return {
        "con_code": con_code,
        "src": "SW2021",
        "is_new": "Y",
        "l1_code": l1_code,
        "l1_name": l1_name,
    }


def test_attach_sw_l1_maps_symbols(dm):
    dm.sw_index_member.insert_one(_member("000001.SZ", "801780.SI", "银行"))
    dm.sw_index_member.insert_one(_member("000002.SZ", "801080.SI", "电子"))

    dataset = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "999999"],
            "score_date": ["20240101", "20240101", "20240101"],
        }
    )
    out = attach_sw_l1(dataset, dm)

    assert out.loc[out["symbol"] == "000001", SW_L1_CODE_COL].iloc[0] == "801780.SI"
    assert out.loc[out["symbol"] == "000002", SW_L1_NAME_COL].iloc[0] == "电子"
    assert out.loc[out["symbol"] == "999999", SW_L1_CODE_COL].iloc[0] == ""


def test_apply_industry_filter_preserves_order(dm):
    frame = pd.DataFrame(
        {
            "symbol": ["a", "b", "c"],
            "score": [3, 2, 1],
            SW_L1_CODE_COL: ["801080.SI", "801080.SI", "801780.SI"],
        }
    )
    filtered = apply_industry_filter(frame, ["801080.SI"])
    assert filtered["symbol"].tolist() == ["a", "b"]


def test_resolve_industry_l1_codes_by_name():
    catalog = [{"code": "801080.SI", "name": "电子"}, {"code": "801780.SI", "name": "银行"}]
    assert resolve_industry_l1_codes(["电子"], catalog) == ["801080.SI"]
    assert resolve_industry_l1_codes(["801780.SI"], catalog) == ["801780.SI"]


def test_sw_l1_coverage_summary():
    dataset = pd.DataFrame(
        {
            SW_L1_CODE_COL: ["801080.SI", "", "801080.SI"],
            SW_L1_NAME_COL: ["电子", "", "电子"],
        }
    )
    summary = sw_l1_coverage_summary(dataset)
    assert summary["rows"] == 3
    assert summary["mapped_rows"] == 2
    assert summary["unmapped_rows"] == 1
    assert summary["distinct_l1"] == 1
