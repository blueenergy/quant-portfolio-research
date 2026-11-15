"""Combo detail must apply the same SW L1 filter as the sweep summary."""

import pandas as pd

from analysis.sweep.combo_detail import run_single_combo
from shared.industry_universe import SW_L1_CODE_COL

SCORE = "composite_cycle_oriented_score"


def _rows(score_date, scores, l1_codes, returns=0.10):
    return [
        {
            "score_date": score_date,
            "symbol": symbol,
            "name": symbol,
            "industry": l1_code,
            SCORE: score,
            "forward_return_1d": returns,
            "entry_price": 100.0,
            "entry_date": score_date,
            "exit_price_1d": 110.0,
            "exit_date_1d": "20250103",
            SW_L1_CODE_COL: l1_code,
        }
        for symbol, score, l1_code in zip(
            [f"S{i:02d}" for i in range(len(scores))],
            scores,
            l1_codes,
        )
    ]


def test_run_single_combo_applies_industry_l1_filter():
    dataset = pd.DataFrame(
        _rows(
            "20250101",
            [90, 70, 80],
            ["801080.SI", "801080.SI", "801170.SI"],
        )
        + _rows(
            "20250102",
            [80, 60, 70],
            ["801080.SI", "801080.SI", "801170.SI"],
        )
    )

    detail = run_single_combo(
        dataset,
        SCORE,
        "top_n",
        2,
        1,
        rebalance_interval_days=1,
        transaction_cost=0,
        industry_l1_codes=("801080.SI",),
    )

    assert detail["meta"]["industry_l1_codes"] == ["801080.SI"]
    assert {
        trade["symbol"] for trade in detail["trades"]
    } == {"S00", "S01"}
    assert all(
        period["path_gate_pool_size"] <= 2
        for period in detail["periods"]
        if period["selected_count"]
    )
