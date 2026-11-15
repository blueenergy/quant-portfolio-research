"""Tests for the per-name SW L1 industry regime column."""

import datetime
import os

import mongomock
import pandas as pd
import pytest

os.environ.pop("TUSHARE_TOKEN", None)

from analysis.core.industry_regime import attach_sw_l1_regime, sw_l1_regime_coverage
from shared.industry_universe import SW_L1_CODE_COL, SW_L1_REGIME_COL

_ELECTRONICS = "801080.SI"
_BANK = "801780.SI"


@pytest.fixture
def dm():
    from shared.data_access import ResearchDataAccess

    client = mongomock.MongoClient()
    return ResearchDataAccess(client["quant_data"], app_db=client["finance"])


def _trade_dates(count: int) -> list:
    start = datetime.date(2024, 1, 1)
    return [(start + datetime.timedelta(days=index)).strftime("%Y%m%d") for index in range(count)]


def _seed_series(dm, code: str, closes: list) -> None:
    dm.sw_daily.insert_many(
        [
            {"ts_code": code, "trade_date": trade_date, "close": float(close)}
            for trade_date, close in zip(_trade_dates(len(closes)), closes)
        ]
    )


def _rising(count: int, start: float = 1000.0, step: float = 5.0) -> list:
    return [start + step * index for index in range(count)]


_RISING_DAYS = 40
_LAST_DATE = _trade_dates(_RISING_DAYS)[-1]


def test_attach_sw_l1_regime_labels_each_row_from_its_own_industry(dm):
    _seed_series(dm, _ELECTRONICS, _rising(40))
    _seed_series(dm, _BANK, list(reversed(_rising(40))))

    dataset = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "score_date": [_LAST_DATE, _LAST_DATE],
            SW_L1_CODE_COL: [_ELECTRONICS, _BANK],
        }
    )
    out = attach_sw_l1_regime(dataset, dm)

    assert list(out.columns) == list(dataset.columns) + [SW_L1_REGIME_COL]
    assert out.loc[0, SW_L1_REGIME_COL] == "bull"
    assert out.loc[1, SW_L1_REGIME_COL] == "bear"


def test_attach_sw_l1_regime_has_no_lookahead(dm):
    """A label must not change because later prices were added to the index."""
    closes = _rising(30)
    _seed_series(dm, _ELECTRONICS, closes)
    dataset = pd.DataFrame(
        {"symbol": ["A"], "score_date": [_trade_dates(30)[-1]], SW_L1_CODE_COL: [_ELECTRONICS]}
    )
    truncated = attach_sw_l1_regime(dataset, dm).loc[0, SW_L1_REGIME_COL]

    dm.sw_daily.delete_many({})
    _seed_series(dm, _ELECTRONICS, closes + _rising(20, start=1200.0, step=-40.0))
    extended = attach_sw_l1_regime(dataset, dm).loc[0, SW_L1_REGIME_COL]

    assert truncated == extended


def test_attach_sw_l1_regime_preserves_row_order_and_index(dm):
    _seed_series(dm, _ELECTRONICS, _rising(40))
    dataset = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score_date": [_LAST_DATE] * 3,
            SW_L1_CODE_COL: [_ELECTRONICS, "", _ELECTRONICS],
        },
        index=[7, 3, 11],
    )
    out = attach_sw_l1_regime(dataset, dm)

    assert list(out.index) == [7, 3, 11]
    assert list(out["symbol"]) == ["A", "B", "C"]
    # An unmapped row keeps an empty label so the backtester never treats it as
    # non-bull just because the mapping was missing.
    assert out.loc[3, SW_L1_REGIME_COL] == ""
    assert out.loc[7, SW_L1_REGIME_COL] == "bull"


def test_attach_sw_l1_regime_unknown_index_leaves_labels_empty(dm):
    dataset = pd.DataFrame(
        {"symbol": ["A"], "score_date": [_LAST_DATE], SW_L1_CODE_COL: ["801999.SI"]}
    )
    out = attach_sw_l1_regime(dataset, dm)
    assert out.loc[0, SW_L1_REGIME_COL] == ""
    assert sw_l1_regime_coverage(out)["labelled_share"] == 0.0


def test_attach_sw_l1_regime_skips_dataset_without_industry_column(dm):
    dataset = pd.DataFrame({"symbol": ["A"], "score_date": [_LAST_DATE]})
    out = attach_sw_l1_regime(dataset, dm)
    assert SW_L1_REGIME_COL not in out.columns


def test_sw_l1_regime_coverage_counts_labelled_and_bull_rows(dm):
    _seed_series(dm, _ELECTRONICS, _rising(40))
    dataset = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score_date": [_LAST_DATE] * 4,
            SW_L1_CODE_COL: [_ELECTRONICS, _ELECTRONICS, "", ""],
        }
    )
    coverage = sw_l1_regime_coverage(attach_sw_l1_regime(dataset, dm))

    assert coverage["rows"] == 4
    assert coverage["labelled_rows"] == 2
    assert coverage["labelled_share"] == 0.5
    assert coverage["bull_share"] == 1.0
