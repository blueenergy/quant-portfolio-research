
import pandas as pd
import pytest

from analysis.core.score_backtester import (
    BacktestConfig,
    ScoreBacktester,
    export_dataset_to_path,
    simulate_open_holding,
)


class _Cursor:
    def __init__(self, docs):
        self.docs = list(docs)
        self.pulled = 0

    def sort(self, key, direction):
        self.docs = sorted(self.docs, key=lambda d: d.get(key, ""), reverse=direction < 0)
        return self

    def batch_size(self, _size):
        return self

    def __iter__(self):
        # A generator, so merely getting an iterator pulls nothing: the count
        # tells a test whether the caller drained the cursor or streamed it.
        for doc in self.docs:
            self.pulled += 1
            yield doc


class _ScoreCollection:
    def __init__(self, docs):
        self.docs = list(docs)
        self.last_query = None
        self.last_cursor = None

    def find(self, query=None, projection=None):
        query = query or {}
        self.last_query = dict(query)
        score_date_query = query.get("score_date") or {}
        index_codes_query = query.get("index_codes") or {}
        docs = self.docs
        if "$gte" in score_date_query:
            docs = [d for d in docs if d.get("score_date", "") >= score_date_query["$gte"]]
        if "$lte" in score_date_query:
            docs = [d for d in docs if d.get("score_date", "") <= score_date_query["$lte"]]
        if "$in" in index_codes_query:
            wanted = set(index_codes_query["$in"])
            docs = [d for d in docs if wanted.intersection(set(d.get("index_codes") or []))]
        self.last_cursor = _Cursor(docs)
        return self.last_cursor


class _FakeDbManager:
    def __init__(self):
        self.score_docs = []
        self.price_data = {}
        self.price_call_count = {}
        self.price_adjustments = []
        self.index_members = {}
        self.stock_scores = _ScoreCollection(self.score_docs)

    def get_volume_price_data(
        self,
        symbol,
        days=None,
        start_date=None,
        end_date=None,
        adjust="hfq",
        projection=None,
    ):
        self.price_call_count[symbol] = self.price_call_count.get(symbol, 0) + 1
        self.price_adjustments.append(adjust)
        rows = self.price_data.get(symbol, [])
        if start_date:
            rows = [r for r in rows if r["trade_date"] >= start_date]
        if end_date:
            rows = [r for r in rows if r["trade_date"] <= end_date]
        return rows

    def get_latest_index_constituent_symbols(self, index_code):
        return self.index_members.get(index_code, [])

    def get_index_constituent_symbols_asof(self, index_code, as_of_date):
        keyed = self.index_members.get(index_code)
        if isinstance(keyed, dict):
            dates = sorted(keyed.keys())
            chosen = None
            for d in dates:
                if d <= as_of_date:
                    chosen = d
                else:
                    break
            return list(keyed.get(chosen, [])) if chosen else []
        return list(keyed or [])


def _make_db():
    db = _FakeDbManager()
    db.index_members["csi1000"] = ["000001", "000002", "000003"]
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    for idx, symbol in enumerate(symbols, start=1):
        score = idx * 10
        db.score_docs.append(
            {
                "symbol": symbol,
                "score_date": "20250101",
                "index_codes": ["csi1000"] if idx <= 3 else ["hs300"],
                "composite_score": {
                    "balanced": score,
                    "trading_oriented": score + 1,
                    "cycle_oriented": score + 2,
                },
                "scores": {
                    "technical": score,
                    "fundamental": 60 - score,
                    "value": 50,
                    "growth": score,
                    "money_flow": score,
                    "cycle": score,
                },
            }
        )
        db.price_data[symbol] = [
            {"trade_date": "20250101", "close": 100.0},
            {"trade_date": "20250102", "close": 100.0 * (1 + score / 1000)},
            {"trade_date": "20250103", "close": 100.0 * (1 + score / 500)},
            {"trade_date": "20250104", "close": 100.0 * (1 + score / 1000) * (1 + score / 500)},
        ]
    db.stock_scores = _ScoreCollection(db.score_docs)
    return db


def test_build_dataset_calculates_forward_returns():
    db = _make_db()
    config = BacktestConfig(horizons=(1, 2), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(db, config).run()

    assert len(result.dataset) == 5
    row = result.dataset[result.dataset["symbol"] == "000005"].iloc[0]
    assert row["composite_score"] == 50
    assert row["composite_balanced_score"] == 50
    assert row["composite_trading_oriented_score"] == 51
    assert row["composite_cycle_oriented_score"] == 52
    assert row["entry_date"] == "20250102"
    assert row["entry_price"] == 105.0
    assert row["entry_price_type"] == "next_open"
    assert row["exit_date_2d"] == "20250104"
    assert round(row["exit_price_2d"], 4) == 115.5
    assert round(row["forward_return_1d"], 4) == round(110.0 / 105.0 - 1, 4)
    assert db.price_adjustments and set(db.price_adjustments) == {"hfq"}
    assert round(row["forward_return_2d"], 4) == 0.10


def test_build_dataset_writes_unrealized_return_when_horizon_incomplete():
    db = _FakeDbManager()
    db.index_members["csi1000"] = ["000001"]
    db.score_docs = [
        {
            "symbol": "000001",
            "score_date": "20250101",
            "index_codes": ["csi1000"],
            "composite_score": 50,
            "scores": {"cycle": 50},
        }
    ]
    db.price_data["000001"] = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250103", "open": 108, "high": 110, "close": 109},
    ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    result = ScoreBacktester(db, BacktestConfig(horizons=(20,))).run()
    row = result.dataset.iloc[0]

    forward_return = row.get("forward_return_20d")
    assert forward_return is None or forward_return != forward_return
    assert round(row["mark_price_20d"], 4) == 108.0
    assert round(row["unrealized_return_20d"], 4) == round(108 / 100 - 1, 4)
    assert bool(row["is_open_20d"]) is True


def test_rank_ic_and_quantile_spread_are_positive_for_predictive_scores():
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(_make_db(), config).run()

    summary = result.summary[result.summary["score_column"] == "composite_score"].iloc[0]
    assert summary["rank_ic_mean"] > 0.99
    assert summary["top_bottom_spread"] > 0


def test_topk_summary_top_beats_universe_when_scores_rank_returns():
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3, top_k_sizes=(2, 4))
    result = ScoreBacktester(_make_db(), config).run()

    tops = result.topk_summary
    assert not tops.empty
    row = tops[(tops["score_column"] == "cycle_score") & (tops["horizon"] == 1) & (tops["top_k"] == 2)].iloc[0]
    assert row["mean_top_vs_universe"] > 0
    assert row["mean_top_minus_bottom"] > 0
    assert "mean_effective_k" in row


def test_topk_summary_disabled_when_top_k_sizes_empty():
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3, top_k_sizes=())
    result = ScoreBacktester(_make_db(), config).run()
    assert result.topk_summary.empty


def test_export_dataset_to_path_writes_csv(tmp_path):
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(_make_db(), config).run()
    out = tmp_path / "panel.csv"
    written = export_dataset_to_path(result.dataset, out)
    assert written == out
    assert written.read_text(encoding="utf-8").splitlines()[0].startswith("symbol")


def test_export_dataset_to_path_round_trips_parquet(tmp_path):
    """The image shipped without pyarrow for months, so pin the fast path down."""
    import pandas as pd
    import pytest

    pytest.importorskip("pyarrow")
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(_make_db(), config).run()

    written = export_dataset_to_path(result.dataset, tmp_path / "panel.parquet")

    assert written.suffix == ".parquet", "fell back to gzip CSV despite pyarrow being installed"
    expected = result.dataset.sort_values(["score_date", "symbol"]).reset_index(drop=True)
    back = pd.read_parquet(written)
    assert list(back.columns) == list(expected.columns)
    assert back.dtypes.astype(str).tolist() == expected.dtypes.astype(str).tolist()
    assert back.equals(expected)


def test_export_dataset_to_path_says_so_when_parquet_is_unavailable(tmp_path, monkeypatch, caplog):
    import logging

    import pandas as pd

    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(_make_db(), config).run()

    def _no_pyarrow(*_args, **_kwargs):
        raise ImportError("pyarrow not installed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _no_pyarrow)
    with caplog.at_level(logging.WARNING):
        written = export_dataset_to_path(result.dataset, tmp_path / "panel.parquet")

    assert written.name == "panel.csv.gz"
    assert "pyarrow unavailable" in caplog.text


def test_inverse_factor_has_negative_rank_ic():
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(_make_db(), config).run()

    summary = result.summary[result.summary["score_column"] == "fundamental_score"].iloc[0]
    assert summary["rank_ic_mean"] < -0.99


def test_trailing_stop_disabled_matches_horizon_exit():
    db = _FakeDbManager()
    db.index_members["csi1000"] = ["000001"]
    db.score_docs = [
        {
            "symbol": "000001",
            "score_date": "20250101",
            "index_codes": ["csi1000"],
            "composite_score": 50,
            "scores": {"cycle": 50},
        }
    ]
    db.price_data["000001"] = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250103", "open": 110, "high": 120, "close": 115},
        {"trade_date": "20250104", "open": 95, "high": 98, "close": 96},
        {"trade_date": "20250105", "open": 90, "high": 91, "close": 90},
    ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    off = ScoreBacktester(db, BacktestConfig(horizons=(2,), trailing_stop_pct=None)).run()
    zero = ScoreBacktester(db, BacktestConfig(horizons=(2,), trailing_stop_pct=0)).run()
    row_off = off.dataset.iloc[0]
    row_zero = zero.dataset.iloc[0]
    assert row_off["exit_reason_2d"] == "horizon"
    assert row_zero["exit_reason_2d"] == "horizon"
    assert row_off["exit_date_2d"] == row_zero["exit_date_2d"] == "20250104"


def test_normalize_trailing_stop_pct_accepts_percent_points():
    assert ScoreBacktester._normalize_trailing_stop_pct(8) == 0.08
    assert ScoreBacktester._normalize_trailing_stop_pct(15) == 0.15
    assert ScoreBacktester._normalize_trailing_stop_pct(0.15) == 0.15
    assert ScoreBacktester._normalize_trailing_stop_pct(0) is None
    assert ScoreBacktester._normalize_trailing_stop_pct(None) is None


def test_trailing_stop_exits_next_open_after_close_drawdown_from_peak():
    db = _FakeDbManager()
    db.index_members["csi1000"] = ["000001"]
    db.score_docs = [
        {
            "symbol": "000001",
            "score_date": "20250101",
            "index_codes": ["csi1000"],
            "composite_score": 50,
            "scores": {"cycle": 50},
        }
    ]
    db.price_data["000001"] = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250103", "open": 110, "high": 120, "close": 115},
        {"trade_date": "20250104", "open": 115, "high": 125, "close": 124},
        {"trade_date": "20250105", "open": 100, "high": 126, "close": 100},
        {"trade_date": "20250106", "open": 95, "high": 98, "close": 96},
        {"trade_date": "20250107", "open": 90, "high": 91, "close": 90},
    ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    result = ScoreBacktester(db, BacktestConfig(horizons=(5,), trailing_stop_pct=0.15)).run()
    row = result.dataset.iloc[0]
    assert row["exit_reason_5d"] == "trailing_stop"
    assert row["stop_trigger_date_5d"] == "20250105"
    assert row["exit_date_5d"] == "20250106"
    assert row["exit_price_5d"] == 95
    assert round(row["forward_return_5d"], 4) == round(95 / 100 - 1, 4)

    percent_points = ScoreBacktester(db, BacktestConfig(horizons=(5,), trailing_stop_pct=15)).run()
    percent_row = percent_points.dataset.iloc[0]
    assert percent_row["exit_reason_5d"] == "trailing_stop"
    assert percent_row["exit_date_5d"] == row["exit_date_5d"]
    assert round(percent_row["forward_return_5d"], 4) == round(row["forward_return_5d"], 4)


def test_regime_cash_exit_is_next_open_and_does_not_change_horizon_return():
    prices = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250103", "open": 102, "high": 103, "close": 102},
        {"trade_date": "20250104", "open": 101, "high": 104, "close": 103},
        {"trade_date": "20250105", "open": 110, "high": 111, "close": 110},
        {"trade_date": "20250106", "open": 120, "high": 121, "close": 120},
        {"trade_date": "20250107", "open": 130, "high": 131, "close": 130},
    ]
    backtester = ScoreBacktester(_FakeDbManager(), BacktestConfig(horizons=(5,)))
    result = backtester._calculate_forward_returns(
        prices,
        (5,),
        "20250101",
        cash_exit_dates={"20250103"},
    )

    assert result["exit_reason_5d"] == "horizon"
    assert result["exit_date_5d"] == "20250107"
    assert result["exit_reason_5d_live_cash"] == "regime_cash"
    assert result["exit_date_5d_live_cash"] == "20250104"
    assert result["exit_price_5d_live_cash"] == 101
    assert round(result["forward_return_5d_live_cash"], 4) == round(101 / 100 - 1, 4)


def test_regime_cash_loses_to_an_earlier_trailing_stop():
    prices = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250103", "open": 110, "high": 120, "close": 115},
        {"trade_date": "20250104", "open": 115, "high": 125, "close": 124},
        {"trade_date": "20250105", "open": 100, "high": 126, "close": 100},
        {"trade_date": "20250106", "open": 95, "high": 98, "close": 96},
        {"trade_date": "20250107", "open": 90, "high": 91, "close": 90},
    ]
    backtester = ScoreBacktester(_FakeDbManager(), BacktestConfig(horizons=(5,), trailing_stop_pct=0.15))
    result = backtester._calculate_forward_returns(
        prices,
        (5,),
        "20250101",
        trailing_stop_pct=0.15,
        cash_exit_dates={"20250107"},
    )

    assert result["exit_reason_5d"] == "trailing_stop"
    assert result["exit_reason_5d_live_cash"] == "trailing_stop"
    assert result["exit_date_5d_live_cash"] == result["exit_date_5d"] == "20250106"


def test_simulate_open_holding_trailing_stop_from_known_entry():
    prices = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 120, "close": 119},
        {"trade_date": "20250103", "open": 110, "high": 110, "close": 100},
        {"trade_date": "20250104", "open": 90, "high": 91, "close": 90},
        {"trade_date": "20250105", "open": 88, "high": 89, "close": 88},
    ]
    result = simulate_open_holding(prices, entry_idx=1, horizon=3, trailing_stop_pct=0.15)
    assert result is not None
    assert result["exit_reason"] == "trailing_stop"
    assert result["entry_date"] == "20250102"
    assert result["exit_date"] == "20250104"
    assert round(result["holding_return"], 4) == round(90 / 100 - 1, 4)

    percent_points = simulate_open_holding(prices, entry_idx=1, horizon=3, trailing_stop_pct=15)
    assert percent_points is not None
    assert percent_points["exit_reason"] == "trailing_stop"
    assert percent_points["exit_date"] == result["exit_date"]
    assert round(percent_points["holding_return"], 4) == round(result["holding_return"], 4)


def test_simulate_open_holding_skips_limit_up_entry():
    prices = [
        {"trade_date": "20250101", "open": 100, "high": 100, "low": 100, "close": 100},
        {"trade_date": "20250102", "open": 110, "high": 110, "low": 110, "close": 110},
        {"trade_date": "20250103", "open": 111, "high": 112, "low": 110, "close": 111},
        {"trade_date": "20250104", "open": 112, "high": 113, "low": 111, "close": 112},
    ]
    assert simulate_open_holding(prices, entry_idx=1, horizon=2) is None


def test_simulate_open_holding_star_ten_percent_is_not_limit_up():
    # STAR has a 20% limit; a one-price +10% day is tradable and must not be
    # skipped by the old fixed 9.5% threshold.
    prices = [
        {"trade_date": "20250101", "symbol": "688001.SH", "open": 100, "high": 100, "low": 100, "close": 100},
        {"trade_date": "20250102", "symbol": "688001.SH", "open": 110, "high": 110, "low": 110, "close": 110},
        {"trade_date": "20250103", "symbol": "688001.SH", "open": 111, "high": 112, "low": 110, "close": 111},
    ]
    result = simulate_open_holding(prices, entry_idx=1, horizon=1)
    assert result is not None
    assert result["entry_price"] == 110


def test_simulate_open_holding_star_twenty_percent_is_limit_up():
    prices = [
        {"trade_date": "20250101", "symbol": "688001.SH", "open": 100, "high": 100, "low": 100, "close": 100},
        {"trade_date": "20250102", "symbol": "688001.SH", "open": 120, "high": 120, "low": 120, "close": 120},
        {"trade_date": "20250103", "symbol": "688001.SH", "open": 121, "high": 122, "low": 120, "close": 121},
    ]
    assert simulate_open_holding(prices, entry_idx=1, horizon=1) is None


def test_simulate_open_holding_chinext_ten_percent_is_not_limit_up():
    prices = [
        {"trade_date": "20250101", "symbol": "300001.SZ", "open": 100, "high": 100, "low": 100, "close": 100},
        {"trade_date": "20250102", "symbol": "300001.SZ", "open": 110, "high": 110, "low": 110, "close": 110},
        {"trade_date": "20250103", "symbol": "300001.SZ", "open": 111, "high": 112, "low": 110, "close": 111},
    ]
    result = simulate_open_holding(prices, entry_idx=1, horizon=1)
    assert result is not None
    assert result["entry_price"] == 110


def test_composite_strategy_scores_are_reported():
    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(_make_db(), config).run()

    score_columns = set(result.summary["score_column"])

    assert "composite_trading_oriented_score" in score_columns
    assert "composite_cycle_oriented_score" in score_columns
    assert "composite_growth_cycle_score" in score_columns


def test_growth_cycle_experimental_score_is_weighted_average():
    db = _make_db()
    for doc in db.score_docs:
        if doc["symbol"] == "000005":
            doc["scores"]["growth"] = 80
            doc["scores"]["cycle"] = 50

    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(db, config).run()

    row = result.dataset[result.dataset["symbol"] == "000005"].iloc[0]

    assert row["growth_score"] == 80
    assert row["cycle_score"] == 50
    assert row["composite_growth_cycle_score"] == 68


def test_preload_price_cache_parallel_matches_sequential(monkeypatch):
    config = BacktestConfig(horizons=(1, 2), quantiles=5, min_names_per_date=3)
    panel = ScoreBacktester(_make_db(), config)._panel_from_records(_make_db().stock_scores.find({}))

    monkeypatch.setenv("PORTFOLIO_RESEARCH_PRELOAD_WORKERS", "1")
    seq = ScoreBacktester(_make_db(), config)
    seq._preload_price_cache(panel)

    monkeypatch.setenv("PORTFOLIO_RESEARCH_PRELOAD_WORKERS", "4")
    par = ScoreBacktester(_make_db(), config)
    par._preload_price_cache(panel)

    assert set(par._price_cache) == set(seq._price_cache)
    for symbol in seq._price_cache:
        assert par._price_cache[symbol] == seq._price_cache[symbol]


def test_build_dataset_parallel_preload_matches_sequential(monkeypatch):
    config = BacktestConfig(horizons=(1, 2), quantiles=5, min_names_per_date=3)

    monkeypatch.setenv("PORTFOLIO_RESEARCH_PRELOAD_WORKERS", "1")
    seq = ScoreBacktester(_make_db(), config).run()

    monkeypatch.setenv("PORTFOLIO_RESEARCH_PRELOAD_WORKERS", "4")
    par = ScoreBacktester(_make_db(), config).run()

    assert par.dataset.equals(seq.dataset)


def test_build_dataset_reuses_one_price_query_per_symbol():
    db = _make_db()
    db.score_docs.append(
        {
            "symbol": "000001",
            "score_date": "20250102",
            "index_codes": ["csi1000"],
            "composite_score": {
                "balanced": 20,
                "trading_oriented": 21,
                "cycle_oriented": 22,
            },
            "scores": {
                "technical": 20,
                "fundamental": 40,
                "value": 50,
                "growth": 20,
                "money_flow": 20,
                "cycle": 20,
            },
        }
    )
    db.price_data["000001"].append({"trade_date": "20250105", "close": 110.0})
    db.stock_scores = _ScoreCollection(db.score_docs)

    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(db, config).run()

    assert len(result.dataset[result.dataset["symbol"] == "000001"]) == 2
    assert db.price_call_count["000001"] == 1


def test_build_dataset_skips_one_price_limit_up_entry():
    db = _make_db()
    db.price_data["000005"] = [
        {"trade_date": "20250101", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        {"trade_date": "20250102", "open": 110.0, "high": 110.0, "low": 110.0, "close": 110.0},
        {"trade_date": "20250103", "open": 120.0, "high": 121.0, "low": 119.0, "close": 120.0},
    ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(db, config).run()

    assert "000005" not in set(result.dataset["symbol"])


def test_build_dataset_delays_one_price_limit_down_exit():
    db = _make_db()
    db.price_data["000005"] = [
        {"trade_date": "20250101", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        {"trade_date": "20250102", "open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0},
        {"trade_date": "20250103", "open": 90.0, "high": 90.0, "low": 90.0, "close": 90.0},
        {"trade_date": "20250104", "open": 92.0, "high": 94.0, "low": 91.0, "close": 93.0},
    ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    config = BacktestConfig(horizons=(1,), quantiles=5, min_names_per_date=3)
    result = ScoreBacktester(db, config).run()
    row = result.dataset[result.dataset["symbol"] == "000005"].iloc[0]

    assert row["exit_date_1d"] == "20250104"
    assert row["exit_delayed_days_1d"] == 1
    assert row["exit_price_1d"] == 92.0
    assert round(row["forward_return_1d"], 4) == -0.08


def test_index_code_filter_limits_score_records():
    config = BacktestConfig(horizons=(1,), quantiles=3, min_names_per_date=3, index_codes=("csi1000",))
    result = ScoreBacktester(_make_db(), config).run()

    assert set(result.dataset["symbol"]) == {"000001", "000002", "000003"}


def test_universe_index_filter_does_not_require_score_index_codes():
    db = _make_db()
    for doc in db.score_docs:
        doc.pop("index_codes", None)

    config = BacktestConfig(horizons=(1,), quantiles=3, min_names_per_date=3, universe_index_codes=("csi1000",))
    result = ScoreBacktester(db, config).run()

    assert set(result.dataset["symbol"]) == {"000001", "000002", "000003"}


def test_universe_index_codes_pushdown_filters_stock_scores_query():
    db = _make_db()
    config = BacktestConfig(horizons=(1,), quantiles=3, min_names_per_date=3, universe_index_codes=("csi1000",))
    ScoreBacktester(db, config).run()
    assert db.stock_scores.last_query["index_codes"] == {"$in": ["csi1000"]}


def test_universe_index_codes_pushdown_falls_back_when_no_rows(monkeypatch, capsys):
    db = _make_db()
    for doc in db.score_docs:
        doc["index_codes"] = ["hs300"]

    config = BacktestConfig(horizons=(1,), quantiles=3, min_names_per_date=3, universe_index_codes=("csi1000",))
    result = ScoreBacktester(db, config).run()

    assert "index_codes" not in (db.stock_scores.last_query or {})
    assert "falling back to date-range query" in capsys.readouterr().out
    assert set(result.dataset["symbol"]) == {"000001", "000002", "000003"}


def test_universe_index_filter_uses_asof_membership_per_score_date():
    db = _FakeDbManager()
    db.index_members["csi1000"] = {
        "20250101": ["000001", "000002"],
        "20250103": ["000003", "000004"],
    }
    for symbol, score_date in (
        ("000001", "20250101"),
        ("000002", "20250101"),
        ("000003", "20250103"),
        ("000004", "20250103"),
        ("000005", "20250103"),
    ):
        db.score_docs.append(
            {
                "symbol": symbol,
                "score_date": score_date,
                "composite_score": {"balanced": 10},
                "scores": {"technical": 10, "fundamental": 10, "value": 10, "growth": 10, "money_flow": 10, "cycle": 10},
            }
        )
        db.price_data[symbol] = [
            {"trade_date": score_date, "close": 100.0},
            {"trade_date": "20250104", "close": 101.0},
            {"trade_date": "20250105", "close": 102.0},
        ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    config = BacktestConfig(horizons=(1,), quantiles=3, min_names_per_date=2, universe_index_codes=("csi1000",))
    result = ScoreBacktester(db, config).run()

    assert set(result.dataset["symbol"]) == {"000001", "000002", "000003", "000004"}


def test_universe_filter_keeps_the_panel_order():
    """A filter must not reorder: the panel is emitted in input order, not group order."""
    db = _FakeDbManager()
    db.index_members["csi1000"] = {
        "20250103": ["000003", "000004"],
        "20250101": ["000001"],
    }
    panel = pd.DataFrame(
        {
            "symbol": ["000003", "000001", "000004"],
            "score_date": ["20250103", "20250101", "20250103"],
            "name": ["", "", ""],
            "industry": ["", "", ""],
            "growth_score": [1.0, 2.0, 3.0],
        }
    )

    config = BacktestConfig(universe_index_codes=("csi1000",))
    filtered = ScoreBacktester(db, config)._filter_panel_by_universe(panel)

    # Grouping by date and concatenating the groups would reorder these to
    # 20250103's rows first (000003, 000004, then 000001).
    assert list(filtered["symbol"]) == ["000003", "000001", "000004"]
    assert list(filtered["growth_score"]) == [1.0, 2.0, 3.0]


def test_universe_filter_keeps_the_panel_when_no_group_can_be_formed():
    """No usable score_date means no group, and a non-empty frame does not promise one.

    `groupby` drops null keys, so an all-null score_date yields zero groups. The
    filter then keeps the panel rather than returning an empty one -- the same call
    it makes when a universe snapshot is missing, since in both cases there is no
    universe to test the rows against.
    """
    db = _FakeDbManager()
    db.index_members["csi1000"] = {"20250103": ["000003"]}
    panel = pd.DataFrame(
        {
            "symbol": ["000003", "000001"],
            "score_date": [None, None],
            "name": ["", ""],
            "industry": ["", ""],
            "growth_score": [1.0, 2.0],
        }
    )

    config = BacktestConfig(universe_index_codes=("csi1000",))
    filtered = ScoreBacktester(db, config)._filter_panel_by_universe(panel)

    assert list(filtered["symbol"]) == ["000003", "000001"]


def test_build_dataset_fills_ragged_keys_column_by_column():
    """Rows do not all carry the same keys, so every column is filled per row.

    The accumulator appends once per emitted row for the union of the keys seen
    so far. A row that lacks a key has to land as a missing value rather than
    desynchronise the columns, which is what `pd.DataFrame(list_of_dicts)` did.
    """
    db = _FakeDbManager()
    db.index_members["csi1000"] = ["000001", "000002"]
    for symbol in ("000001", "000002"):
        db.score_docs.append(
            {
                "symbol": symbol,
                "score_date": "20250101",
                "index_codes": ["csi1000"],
                "composite_score": 50,
                "scores": {"growth": 10, "cycle": 20},
            }
        )
    # 000001 has the full horizon; 000002 runs out of history, so its row gets
    # the mark/unrealized keys instead of forward_return_20d.
    db.price_data["000001"] = [
        {"trade_date": f"202501{day:02d}", "open": 100, "high": 101, "close": 100 + day}
        for day in range(1, 26)
    ]
    db.price_data["000002"] = [
        {"trade_date": "20250101", "open": 100, "high": 100, "close": 100},
        {"trade_date": "20250102", "open": 100, "high": 100, "close": 101},
    ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    dataset = ScoreBacktester(db, BacktestConfig(horizons=(20,))).build_dataset()
    by_symbol = dataset.set_index("symbol")

    # Every column either row produced survives, so the frame is the key union.
    assert {"forward_return_20d", "unrealized_return_20d"} <= set(dataset.columns)
    assert pd.notna(by_symbol.loc["000001", "forward_return_20d"])
    assert pd.isna(by_symbol.loc["000001", "unrealized_return_20d"])
    assert pd.isna(by_symbol.loc["000002", "forward_return_20d"])
    assert pd.notna(by_symbol.loc["000002", "unrealized_return_20d"])
    # A column the first row put in a double buffer stays float64 even when a
    # later row has no value for it.
    assert dataset["forward_return_20d"].dtype == "float64"


def test_panel_builder_reports_a_column_boxed_by_a_none_first_value(caplog):
    """A key first seen with None never leaves the list path, so say so.

    `_is_numeric` reads the value on the first row carrying a key and never
    revisits the decision. Writers are expected to omit keys they have no value
    for, which is what makes that first value real; a writer that emits None
    instead silently boxes every float that follows, 24 bytes apiece on top of the
    pointer. Checking the finished dtypes once turns that into a log line.
    """
    import logging

    db = _FakeDbManager()
    db.index_members["csi1000"] = ["000001", "000002"]
    for symbol in ("000001", "000002"):
        db.score_docs.append(
            {
                "symbol": symbol,
                "score_date": "20250101",
                "index_codes": ["csi1000"],
                "composite_score": 50,
                "scores": {"growth": 10},
            }
        )
        db.price_data[symbol] = [
            {"trade_date": f"202501{day:02d}", "open": 100, "high": 101, "close": 100 + day}
            for day in range(1, 6)
        ]
    db.stock_scores = _ScoreCollection(db.score_docs)

    backtester = ScoreBacktester(db, BacktestConfig(horizons=(1,)))
    real = backtester._calculate_forward_returns
    injected = {"done": False}

    def _none_then_floats(*args, **kwargs):
        # The first row emits an explicit None for the key and every later row a
        # float. Omitting the key instead is what a writer is supposed to do.
        row = real(*args, **kwargs)
        if row and not injected["done"]:
            injected["done"] = True
            row["forward_return_1d"] = None
        return row

    backtester._calculate_forward_returns = _none_then_floats
    with caplog.at_level(logging.WARNING):
        dataset = backtester.build_dataset()

    assert injected["done"], "the stub never saw a row to poison, so nothing was tested"
    # The column still lands as float64 -- via the list path, which is the point.
    assert dataset["forward_return_1d"].dtype == "float64"
    assert "forward_return_1d" in caplog.text
    assert "boxed-list path" in caplog.text


def test_load_score_records_streams_the_cursor():
    """Taking one record must pull one document, not the whole result set.

    `_load_score_records` used to return ``list(cursor)``, which held ~8 GB
    resident on a csi1000 panel over ~11 years for the whole flattening.
    """
    db = _make_db()
    records = ScoreBacktester(db, BacktestConfig())._load_score_records()

    assert not isinstance(records, list)
    assert next(records)["symbol"] == "000001"
    assert db.stock_scores.last_cursor.pulled == 1


def test_score_panel_contract_names_the_two_ways_it_is_violated():
    backtester = ScoreBacktester(_make_db(), BacktestConfig(horizons=(1,)))

    # The pre-refactor form: a list of documents, not a frame.
    with pytest.raises(TypeError, match="must be a DataFrame"):
        backtester.build_dataset_from_panel([{"symbol": "000001", "score_date": "20250101"}])

    # A frame that is missing part of the contract. name/industry are required
    # rather than defaulted, so a producer cannot leave them to the consumer.
    with pytest.raises(ValueError, match="missing required columns: name, industry"):
        backtester.build_dataset_from_panel(
            pd.DataFrame({"symbol": ["000001"], "score_date": ["20250101"]})
        )


def test_prepare_score_panel_reports_each_step():
    config = BacktestConfig(horizons=(1,), quantiles=3, min_names_per_date=3, index_codes=("csi1000",))
    steps = []

    panel = ScoreBacktester(_make_db(), config).prepare_score_panel(
        on_step=lambda name, elapsed: steps.append((name, elapsed))
    )

    assert [name for name, _ in steps] == [
        "load_score_records",
        "filter_records_by_universe",
        "preload_price_cache",
    ]
    assert all(elapsed >= 0 for _, elapsed in steps)
    # The callback must not disturb what the panel actually returns.
    assert panel.equals(ScoreBacktester(_make_db(), config).prepare_score_panel())


def _linear_scan_slice(price_history, score_date, max_horizon):
    """The pre-bisect implementation, kept here as the equivalence oracle."""
    future = [row for row in price_history if row.get("trade_date") and row["trade_date"] >= score_date]
    return future[: max_horizon + 1]


def test_slice_future_prices_matches_linear_scan_on_sorted_history():
    backtester = ScoreBacktester(_FakeDbManager(), BacktestConfig())
    history = [{"trade_date": f"2025010{i}", "close": float(i)} for i in range(1, 10)]

    for score_date in ("20241231", "20250101", "20250105", "20250109", "20250110"):
        for max_horizon in (0, 1, 3, 20):
            assert backtester._slice_future_prices(
                history, score_date, max_horizon, symbol="000001"
            ) == _linear_scan_slice(history, score_date, max_horizon)


def test_slice_future_prices_falls_back_when_history_unusable():
    backtester = ScoreBacktester(_FakeDbManager(), BacktestConfig())
    unsorted_history = [
        {"trade_date": "20250105", "close": 5.0},
        {"trade_date": "20250101", "close": 1.0},
        {"trade_date": "20250107", "close": 7.0},
    ]
    undated_history = [
        {"trade_date": "20250101", "close": 1.0},
        {"trade_date": None, "close": 2.0},
        {"trade_date": "20250105", "close": 5.0},
    ]

    for history in (unsorted_history, undated_history):
        assert backtester._slice_future_prices(
            history, "20250102", 5, symbol="000001"
        ) == _linear_scan_slice(history, "20250102", 5)


def test_slice_future_prices_rebuilds_index_when_cache_replaced():
    backtester = ScoreBacktester(_FakeDbManager(), BacktestConfig())
    first = [{"trade_date": "20250101", "close": 1.0}, {"trade_date": "20250102", "close": 2.0}]
    second = [{"trade_date": "20250201", "close": 9.0}, {"trade_date": "20250202", "close": 8.0}]

    assert backtester._slice_future_prices(first, "20250101", 5, symbol="000001") == first
    # A fresh history object under the same symbol must not answer from the old index.
    assert backtester._slice_future_prices(second, "20250201", 5, symbol="000001") == second


def test_to_float_rejects_every_flavour_of_missing():
    import numpy as np
    import pandas as pd
    from decimal import Decimal

    backtester = ScoreBacktester(_FakeDbManager(), BacktestConfig())

    for missing in (
        None,
        "",
        float("nan"),
        Decimal("NaN"),
        np.nan,
        np.float64("nan"),
        pd.NA,
        pd.NaT,
        "not-a-number",
        object(),
    ):
        assert backtester._to_float(missing) is None, missing

    assert backtester._to_float(3) == 3.0
    assert backtester._to_float("4.5") == 4.5
    assert backtester._to_float(Decimal("2.5")) == 2.5
    assert backtester._to_float(np.float64(1.5)) == 1.5
    assert backtester._to_float(float("inf")) == float("inf")


def test_calculate_forward_returns_emits_raw_prices_for_lot_sizing():
    """hfq prices carry adj_factor; lot sizing needs the raw execution price."""
    bt = ScoreBacktester(None, BacktestConfig(horizons=(20,)))
    rows = []
    for idx in range(25):
        close = 10.0 + idx * 0.1
        rows.append(
            {
                "trade_date": f"202501{idx + 1:02d}",
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "adj_factor": 6.9761,
            }
        )
    returns = bt._calculate_forward_returns(rows, [20], score_date="20250101")

    entry = rows[1]
    exit_row = rows[21]
    assert returns["entry_price"] == pytest.approx(entry["open"])
    assert returns["entry_raw_price"] == pytest.approx(entry["open"] / 6.9761)
    assert returns["exit_price_20d"] == pytest.approx(exit_row["open"])
    assert returns["exit_raw_price_20d"] == pytest.approx(exit_row["open"] / 6.9761)
    assert returns["forward_return_20d"] == pytest.approx(
        exit_row["open"] / entry["open"] - 1
    )


def test_price_history_preserves_adj_factor_for_raw_lot_pricing():
    db = _FakeDbManager()
    db.price_data["000001"] = [
        {
            "trade_date": "20250101",
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "adj_factor": 6.9761,
        }
    ]
    bt = ScoreBacktester(db, BacktestConfig(horizons=(1,)))

    rows = bt._load_symbol_price_history("000001", "20250101")

    assert rows[0]["adj_factor"] == pytest.approx(6.9761)


def test_build_dataset_emits_raw_entry_price_from_adjusted_history():
    db = _make_db()
    for rows in db.price_data.values():
        for row in rows:
            row["adj_factor"] = 7.0
    bt = ScoreBacktester(db, BacktestConfig(horizons=(1,), min_names_per_date=3))

    dataset = bt.build_dataset()

    row = dataset[dataset["symbol"] == "000001"].iloc[0]
    assert row["entry_price"] == pytest.approx(100.0 * (1 + 10 / 1000))
    assert row["entry_raw_price"] == pytest.approx(row["entry_price"] / 7.0)
