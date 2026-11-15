"""Point-in-time SW L1 membership in ``attach_sw_l1``."""

import pandas as pd
import pytest

from shared.industry_universe import SW_L1_CODE_COL, SW_L1_NAME_COL, attach_sw_l1

# Real stint history for 000004.SZ, used as the boundary fixture:
#   医药生物  2008-12-31 .. 2021-06-10
#   计算机    2021-06-11 .. (current, out_date empty)
_STINTS = [
    {
        "src": "SW2021",
        "con_code": "000004.SZ",
        "in_date": "20081231",
        "out_date": "20210610",
        "l1_code": "801150.SI",
        "l1_name": "医药生物",
        "is_new": "N",
    },
    {
        "src": "SW2021",
        "con_code": "000004.SZ",
        "in_date": "20210611",
        "out_date": None,
        "l1_code": "801750.SI",
        "l1_name": "计算机",
        "is_new": "Y",
    },
    {
        "src": "SW2021",
        "con_code": "000001.SZ",
        "in_date": "20150105",
        "out_date": None,
        "l1_code": "801780.SI",
        "l1_name": "银行",
        "is_new": "Y",
    },
]


class _FakeColl:
    def __init__(self, docs):
        self._docs = docs

    def find(self, query, projection=None):
        codes = set((query.get("con_code") or {}).get("$in") or [])
        src = query.get("src")
        is_new = query.get("is_new")
        out = []
        for doc in self._docs:
            if src and doc.get("src") != src:
                continue
            if is_new and doc.get("is_new") != is_new:
                continue
            if codes and doc.get("con_code") not in codes:
                continue
            out.append({k: v for k, v in doc.items() if k != "_id"})
        return out


class _FakeDb:
    def __init__(self, docs):
        self.sw_index_member = _FakeColl(docs)


def _dataset(rows):
    return pd.DataFrame(rows, columns=["symbol", "score_date"])


def _attach(rows, **kwargs):
    return attach_sw_l1(_dataset(rows), _FakeDb(_STINTS), **kwargs)


def test_pit_uses_the_industry_in_force_on_each_score_date():
    out = _attach(
        [["000004.SZ", "20200101"], ["000004.SZ", "20220101"]],
        as_of_column="score_date",
    )
    assert list(out[SW_L1_NAME_COL]) == ["医药生物", "计算机"]
    assert list(out[SW_L1_CODE_COL]) == ["801150.SI", "801750.SI"]


def test_pit_treats_in_date_and_out_date_as_inclusive():
    out = _attach(
        [["000004.SZ", "20210610"], ["000004.SZ", "20210611"]],
        as_of_column="score_date",
    )
    # out_date is the last day *in* the old industry; the new one starts next day.
    assert list(out[SW_L1_NAME_COL]) == ["医药生物", "计算机"]


def test_pit_does_not_borrow_a_stint_that_starts_later():
    """A date before the first stint must stay unmapped, not snap forward."""
    out = _attach([["000001.SZ", "20140101"]], as_of_column="score_date")
    assert out.loc[0, SW_L1_CODE_COL] == ""
    assert out.loc[0, SW_L1_NAME_COL] == ""


def test_pit_leaves_unmapped_symbols_empty():
    out = _attach([["000999.SZ", "20200101"]], as_of_column="score_date")
    assert out.loc[0, SW_L1_CODE_COL] == ""


def test_default_keeps_the_current_snapshot_behaviour():
    """No as_of_column -> one industry per symbol, identical on every row."""
    out = _attach([["000004.SZ", "20200101"], ["000004.SZ", "20220101"]])
    assert list(out[SW_L1_NAME_COL]) == ["计算机", "计算机"]


def test_pit_preserves_row_order_and_index():
    dataset = pd.DataFrame(
        {"symbol": ["000004.SZ", "000001.SZ", "000004.SZ"], "score_date": ["20220101", "20200101", "20200101"]},
        index=[5, 2, 9],
    )
    out = attach_sw_l1(dataset, _FakeDb(_STINTS), as_of_column="score_date")
    assert list(out.index) == [5, 2, 9]
    assert list(out[SW_L1_NAME_COL]) == ["计算机", "银行", "医药生物"]


def test_pit_rejects_a_missing_as_of_column():
    with pytest.raises(ValueError, match="as_of_column"):
        attach_sw_l1(_dataset([["000004.SZ", "20200101"]]), _FakeDb(_STINTS), as_of_column="nope")
