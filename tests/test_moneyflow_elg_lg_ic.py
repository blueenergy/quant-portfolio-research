"""Unit tests for raw elg+lg money-flow IC helpers. No Mongo."""

from __future__ import annotations

import pandas as pd

from analysis.core.score_backtester import BacktestConfig
from analysis.research.moneyflow_elg_lg_ic import (
    add_size_neutral,
    attach_money_flow_factors,
    coverage_stats,
    evaluate,
    money_flow_docs_to_frame,
    money_flow_row_metrics,
    to_ts_symbol,
)
from shared.data_access import ResearchDataAccess


def test_research_data_access_exposes_money_flow_and_daily_basic(monkeypatch):
    """The research facade is built on a mapping, not attribute access.

    This facade was rewritten during the split from stock-scoring-system: it no
    longer wraps DatabaseManager, so collections come from ``db[name]``. The
    two lib-backed accessors are stubbed because a fake dict is enough to show
    which collections the facade exposes.
    """
    import shared.data_access as mod

    class _Db(dict):
        def __missing__(self, key):
            self[key] = object()
            return self[key]

    monkeypatch.setattr(mod, "MarketDataAccess", lambda db: object())
    monkeypatch.setattr(mod, "SwDataAccess", lambda db: object())

    db = _Db()
    db["money_flow"] = object()
    db["financial_daily_basic"] = object()

    access = ResearchDataAccess(db)
    assert access.money_flow is db["money_flow"]
    assert access.daily_basic is db["financial_daily_basic"]


def test_money_flow_row_metrics_elg_plus_lg_over_buy_turnover():
    row = {
        "buy_elg_amount": 80,
        "sell_elg_amount": 20,
        "buy_lg_amount": 30,
        "sell_lg_amount": 10,
        "buy_md_amount": 40,
        "buy_sm_amount": 50,
    }
    metrics = money_flow_row_metrics(row)
    assert metrics["elg_net"] == 60
    assert metrics["main_net"] == 80
    assert metrics["turnover_wan"] == 200
    assert metrics["main_net_to_amt"] == 0.4
    assert metrics["elg_net_to_amt"] == 0.3


def test_join_strips_exchange_suffix_and_keeps_score_date_alignment():
    panel = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "score_date": ["20240103", "20240103"],
            "forward_return_1d": [0.02, -0.01],
        }
    )
    mf = money_flow_docs_to_frame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": 20240103,
                "buy_elg_amount": 100,
                "sell_elg_amount": 0,
                "buy_lg_amount": 0,
                "sell_lg_amount": 0,
                "buy_md_amount": 50,
                "buy_sm_amount": 50,
            },
            {
                "symbol": "000002.SZ",
                "trade_date": "20240102",
                "buy_elg_amount": 10,
                "sell_elg_amount": 0,
                "buy_lg_amount": 0,
                "sell_lg_amount": 0,
                "buy_md_amount": 10,
                "buy_sm_amount": 10,
            },
        ]
    )
    out = attach_money_flow_factors(panel, mf)
    assert out.loc[out["symbol"] == "000001", "main_net"].iloc[0] == 100
    assert pd.isna(out.loc[out["symbol"] == "000002", "main_net"].iloc[0])
    assert "forward_return_1d" in out.columns
    assert list(out["score_date"]) == ["20240103", "20240103"]


def test_mainflow20_is_right_aligned_no_lookahead():
    docs = []
    for i, date in enumerate(["20240102", "20240103", "20240104", "20240105"]):
        docs.append(
            {
                "symbol": "000001.SZ",
                "trade_date": date,
                "buy_elg_amount": 10 * (i + 1),
                "sell_elg_amount": 0,
                "buy_lg_amount": 0,
                "sell_lg_amount": 0,
                "buy_md_amount": 10,
                "buy_sm_amount": 0,
            }
        )
    frame = money_flow_docs_to_frame(docs, flow_window=2, min_periods=2)
    day2 = frame.loc[frame["trade_date"] == "20240103"].iloc[0]
    expected = (10 + 20) / ((10 + 10) + (20 + 10))
    assert abs(day2["mainflow20"] - expected) < 1e-12
    day3 = frame.loc[frame["trade_date"] == "20240104"].iloc[0]
    not_including_day4 = (20 + 30) / ((20 + 10) + (30 + 10))
    assert abs(day3["mainflow20"] - not_including_day4) < 1e-12


def test_coverage_floor_and_missing_rows_excluded_from_ic():
    panel = pd.DataFrame(
        {
            "symbol": [f"{i:06d}" for i in range(6)],
            "score_date": ["20240103"] * 6,
            "industry": ["bank"] * 6,
            "forward_return_1d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "main_net_to_amt": [0.1, 0.2, 0.3, None, None, None],
            "main_net": [1, 2, 3, None, None, None],
            "elg_net_to_amt": [0.1, 0.2, 0.3, None, None, None],
            "mainflow20": [None] * 6,
        }
    )
    stats = coverage_stats(panel, floor=0.8)
    assert stats["columns"]["main_net_to_amt"]["coverage"] == 0.5
    assert stats["primary_ok"] is False
    report = evaluate(panel, BacktestConfig(horizons=(1,), min_names_per_date=3), coverage=stats)
    assert report["verdict"]["label"] == "insufficient_coverage"
    ic = report["factors"]["main_net_to_amt"]["1"]["raw"]
    assert ic["observations"] == 1
    assert ic["avg_names"] == 3


def test_size_neutral_residuals_out_log_mv():
    n = 8
    panel = pd.DataFrame(
        {
            "symbol": [f"{i:06d}" for i in range(n)],
            "score_date": ["20240103"] * n,
            "main_net_to_amt": [float(i) for i in range(n)],
            "total_mv": [100.0 * (i + 1) for i in range(n)],
        }
    )
    col = add_size_neutral(panel, "main_net_to_amt")
    resid = panel[col]
    log_mv = pd.Series([__import__("math").log(v) for v in panel["total_mv"]])
    corr = resid.corr(log_mv)
    assert abs(corr) < 1e-9


def test_to_ts_symbol_covers_sh_sz_bj():
    assert to_ts_symbol("000001") == "000001.SZ"
    assert to_ts_symbol("600000") == "600000.SH"
    assert to_ts_symbol("688001") == "688001.SH"
    assert to_ts_symbol("430047") == "430047.BJ"
    assert to_ts_symbol("000001.SZ") == "000001.SZ"


def test_evaluate_does_not_read_money_flow_score():
    n = 8
    panel = pd.DataFrame(
        {
            "symbol": [f"{i:06d}" for i in range(n)],
            "score_date": ["20240103"] * n,
            "industry": ["bank"] * n,
            "money_flow_score": [90] * n,
            "main_net": [float(i) for i in range(n)],
            "main_net_to_amt": [0.01 * i for i in range(n)],
            "elg_net_to_amt": [0.005 * i for i in range(n)],
            "mainflow20": [0.02 * i for i in range(n)],
            "forward_return_1d": [0.001 * i for i in range(n)],
            "total_mv": [1000.0] * n,
        }
    )
    report = evaluate(panel, BacktestConfig(horizons=(1,), min_names_per_date=5))
    assert "money_flow_score" not in report["evaluated_columns"]
    assert report["coverage"]["primary_ok"] is True
    assert report["factors"]["main_net_to_amt"]["1"]["raw"]["observations"] == 1
