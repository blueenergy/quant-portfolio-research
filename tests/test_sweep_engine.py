from pathlib import Path

import pandas as pd
import pytest

from analysis.core.portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktester
from analysis.sweep.portfolio_sweep_report import build_sweep_variants, run_parameter_sweep
from analysis.core.score_backtester import ScoreBacktester, BacktestConfig
from analysis.sweep.sweep_engine import (
    LocalCheckpointStore,
    SweepCheckpoint,
    _ru_maxrss_to_mb,
    cgroup_memory_metrics,
    combo_param_hash,
    legacy_score_records_path,
    load_panel_cache,
    log_sweep_perf,
    merge_volatility_columns,
    panel_cache_key,
    panel_cache_paths,
    save_panel_cache,
    save_score_panel_cache,
)


def _sample_dataset():
    return pd.DataFrame(
        [
            {"score_date": "20250101", "symbol": "A", "industry": "Tech", "composite_growth_cycle_score": 100, "growth_score": 100, "cycle_score": 20, "forward_return_1d": 0.10, "vol_20d_annual": 0.3},
            {"score_date": "20250101", "symbol": "B", "industry": "Tech", "composite_growth_cycle_score": 95, "growth_score": 90, "cycle_score": 40, "forward_return_1d": 0.08, "vol_20d_annual": 0.25},
            {"score_date": "20250101", "symbol": "C", "industry": "Bank", "composite_growth_cycle_score": 90, "growth_score": 50, "cycle_score": 100, "forward_return_1d": 0.04, "vol_20d_annual": 0.2},
            {"score_date": "20250102", "symbol": "A", "industry": "Tech", "composite_growth_cycle_score": 70, "growth_score": 70, "cycle_score": 20, "forward_return_1d": 0.01, "vol_20d_annual": 0.28},
            {"score_date": "20250102", "symbol": "B", "industry": "Tech", "composite_growth_cycle_score": 60, "growth_score": 60, "cycle_score": 30, "forward_return_1d": 0.02, "vol_20d_annual": 0.22},
            {"score_date": "20250102", "symbol": "C", "industry": "Bank", "composite_growth_cycle_score": 95, "growth_score": 40, "cycle_score": 95, "forward_return_1d": 0.03, "vol_20d_annual": 0.18},
        ]
    )


def _index_prices():
    return [
        {"trade_date": "20250101", "close": 1000},
        {"trade_date": "20250102", "close": 1010},
        {"trade_date": "20250103", "close": 1020},
    ]


def test_combo_param_hash_is_stable():
    row = {
        "score_variant": "growth_cycle_g60_c40",
        "score_column": "growth_cycle_g60_c40_score",
        "variant": "growth_cycle_topn",
        "construction_mode": "top_n",
        "path_gate": "off",
        "volatility_variant": "vol_off",
        "volatility_mode": "off",
        "rebalance_interval_days": 20,
        "transaction_cost": 0.001,
        "horizon": 20,
    }
    first = combo_param_hash(row, watermark={"stock_scores_max_date": "20250106"}, trailing_stop_pct=None)
    second = combo_param_hash(row, watermark={"stock_scores_max_date": "20250106"}, trailing_stop_pct=None)
    assert first == second
    assert combo_param_hash(row, watermark={"stock_scores_max_date": "20250106"}, trailing_stop_pct=0.15) != first
    watermark = {"stock_scores_max_date": "20250106"}
    assert combo_param_hash({**row, "bull_exposure": 1.0}, watermark=watermark) == first
    assert combo_param_hash({**row, "intra_period_refill": False}, watermark=watermark) == first
    assert combo_param_hash({**row, "bull_exposure": 0.5}, watermark=watermark) != first
    assert combo_param_hash({**row, "intra_period_refill": True}, watermark=watermark) != first
    assert combo_param_hash({**row, "crowd_exit_k": 0}, watermark=watermark) == first
    assert combo_param_hash({**row, "crowd_exit_k": 3}, watermark=watermark) != first
    assert combo_param_hash({**row, "non_bull_action": "cash"}, watermark=watermark) == first
    assert combo_param_hash({**row, "non_bull_action": "half_topn"}, watermark=watermark) != first


def test_combo_param_hash_includes_industry_l1_filter():
    base = {
        "score_column": "composite_growth_cycle_score",
        "variant": "growth_cycle_topn",
        "rebalance_interval_days": 20,
    }

    assert combo_param_hash(base) != combo_param_hash(
        {**base, "industry_l1_codes": ["801080.SI"]}
    )
    assert combo_param_hash(
        {**base, "industry_l1_codes": ["801080.SI"]}
    ) != combo_param_hash({**base, "industry_l1_codes": ["801170.SI"]})


def test_combo_param_hash_includes_dynamic_threshold_parameters():
    base = {
        "score_column": "composite_growth_cycle_score",
        "variant": "growth_cycle_topn",
        "selection_mode": "dynamic_score_threshold",
        "threshold_lookback_days": 10,
        "max_positions": 20,
    }

    assert combo_param_hash(base) != combo_param_hash(
        {**base, "threshold_lookback_days": 20}
    )
    assert combo_param_hash(base) != combo_param_hash(
        {**base, "max_positions": 30}
    )


def test_panel_cache_roundtrip(tmp_path):
    dataset = _sample_dataset()
    price_cache = {"A": [{"trade_date": "20250101", "close": 10.0}]}
    score_panel = pd.DataFrame({"symbol": ["A"], "score_date": ["20250101"], "growth_score": [80.0]})
    key = panel_cache_key(
        universe_index=["csi1000"],
        start_date="20250101",
        end_date="20250102",
        horizon=1,
        stock_scores_max_date="20250102",
    )
    save_panel_cache(key, dataset, price_cache, score_panel=score_panel, cache_root=tmp_path)
    loaded = load_panel_cache(key, cache_root=tmp_path)
    assert loaded is not None
    loaded_dataset, loaded_price_cache, loaded_panel = loaded
    assert len(loaded_dataset) == len(dataset)
    assert loaded_price_cache["A"][0]["close"] == 10.0
    assert loaded_panel is not None
    assert loaded_panel["symbol"].tolist() == ["A"]
    assert loaded_panel["growth_score"].tolist() == [80.0]


def test_save_panel_cache_drops_the_legacy_score_records_pickle(tmp_path):
    """The pre-refactor pickle has no reader any more, so it must not linger."""
    dataset = _sample_dataset()
    price_cache = {"A": [{"trade_date": "20250101", "close": 10.0}]}
    key = panel_cache_key(
        universe_index=["csi1000"],
        start_date="20250101",
        end_date="20250102",
        horizon=1,
        stock_scores_max_date="20250102",
    )
    legacy = legacy_score_records_path(key, cache_root=tmp_path)
    legacy.write_bytes(b"pickle of the list-of-dicts panel")

    save_panel_cache(key, dataset, price_cache, cache_root=tmp_path)

    assert not legacy.exists()
    # The live files are untouched by the cleanup.
    assert (tmp_path / f"panel_{key}.parquet").exists()
    assert (tmp_path / f"panel_{key}_price_cache.pkl").exists()


def test_save_score_panel_cache_completes_an_entry_that_predates_the_panel(tmp_path):
    """The cache-hit path must be able to write the panel it just rebuilt.

    `save_panel_cache` runs only on a miss, so a key cached before the columnar
    panel existed rebuilt it from Mongo on every run and threw the result away.
    Completing the entry has to leave the files the hit path already loaded
    alone -- rewriting those from a cache read is what this helper exists to
    avoid.
    """
    dataset = _sample_dataset()
    price_cache = {"A": [{"trade_date": "20250101", "close": 10.0}]}
    key = panel_cache_key(
        universe_index=["csi1000"],
        start_date="20250101",
        end_date="20250102",
        horizon=1,
        stock_scores_max_date="20250102",
    )
    dataset_path = Path(save_panel_cache(key, dataset, price_cache, cache_root=tmp_path))
    price_cache_path = panel_cache_paths(key, tmp_path)[1]
    assert load_panel_cache(key, cache_root=tmp_path)[2] is None

    dataset_before = dataset_path.read_bytes()
    price_cache_before = price_cache_path.read_bytes()

    score_panel = pd.DataFrame({"symbol": ["A"], "score_date": ["20250101"], "growth_score": [80.0]})
    save_score_panel_cache(key, score_panel, cache_root=tmp_path)

    loaded_dataset, loaded_price_cache, loaded_panel = load_panel_cache(key, cache_root=tmp_path)
    assert loaded_panel is not None
    assert loaded_panel["symbol"].tolist() == ["A"]
    assert loaded_panel["growth_score"].tolist() == [80.0]
    assert len(loaded_dataset) == len(dataset)
    assert loaded_price_cache["A"][0]["close"] == 10.0
    assert dataset_path.read_bytes() == dataset_before
    assert price_cache_path.read_bytes() == price_cache_before


def test_panel_cache_roundtrip_without_parquet_engine(tmp_path, monkeypatch):
    dataset = _sample_dataset()
    price_cache = {"A": [{"trade_date": "20250101", "close": 10.0}]}
    score_panel = pd.DataFrame({"symbol": ["A"], "score_date": ["20250101"], "growth_score": [80.0]})
    key = panel_cache_key(
        universe_index=["csi1000"],
        start_date="20250101",
        end_date="20250102",
        horizon=1,
        stock_scores_max_date="20250102",
    )

    def raise_missing_engine(*_args, **_kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", raise_missing_engine)
    saved_path = save_panel_cache(key, dataset, price_cache, score_panel=score_panel, cache_root=tmp_path)

    assert saved_path.name == f"panel_{key}.csv.gz"
    assert saved_path.exists()
    assert not (tmp_path / f"panel_{key}.parquet").exists()

    loaded = load_panel_cache(key, cache_root=tmp_path)
    assert loaded is not None
    loaded_dataset, loaded_price_cache, loaded_panel = loaded
    assert loaded_dataset["score_date"].tolist() == dataset.sort_values(["score_date", "symbol"])["score_date"].tolist()
    assert loaded_dataset["symbol"].tolist() == dataset.sort_values(["score_date", "symbol"])["symbol"].tolist()
    assert loaded_price_cache == price_cache
    # The score panel falls back to gzip CSV too, and comes back the same.
    assert loaded_panel is not None
    assert loaded_panel["symbol"].tolist() == ["A"]
    assert loaded_panel["growth_score"].tolist() == [80.0]


def test_sweep_checkpoint_resume(tmp_path):
    checkpoint = SweepCheckpoint(tmp_path / "ckpt")
    rows = [{"top_n": 10, "risk_adjusted_score": 1.2}]
    checkpoint.save_rows("abc", rows)
    assert checkpoint.is_done("abc")
    assert checkpoint.get_rows("abc") == rows
    assert checkpoint.completed_count() == 1


def test_sweep_checkpoint_reloads_a_local_store_from_disk(tmp_path):
    SweepCheckpoint(tmp_path / "ckpt").save_rows("abc", [{"top_n": 10}])

    resumed = SweepCheckpoint(tmp_path / "ckpt")

    assert resumed.is_done("abc")
    assert resumed.get_rows("abc") == [{"top_n": 10}]


class _MemoryStore:
    """Stands in for a remote store; keeps the payload off the filesystem."""

    def __init__(self, payload=None):
        self.payload = dict(payload or {})
        self.saves = 0

    def load(self):
        return dict(self.payload)

    def save(self, payload):
        self.payload = dict(payload)
        self.saves += 1

    def describe(self):
        return "memory://checkpoint"


def test_sweep_checkpoint_accepts_a_custom_store_and_writes_through(tmp_path):
    store = _MemoryStore()

    checkpoint = SweepCheckpoint(store=store)
    checkpoint.save_rows("abc", [{"top_n": 10}])

    assert store.saves == 1
    assert store.payload == {"abc": [{"top_n": 10}]}
    assert checkpoint.describe() == "memory://checkpoint"
    # No directory is created when the store is remote.
    assert checkpoint.directory is None
    assert not list(tmp_path.iterdir())


def test_sweep_checkpoint_resumes_from_a_custom_store():
    store = _MemoryStore({"done": [{"top_n": 20}]})

    checkpoint = SweepCheckpoint(store=store)

    assert checkpoint.is_done("done")
    assert checkpoint.get_rows("done") == [{"top_n": 20}]
    assert checkpoint.completed_count() == 1


def test_sweep_checkpoint_requires_a_directory_or_a_store():
    with pytest.raises(ValueError):
        SweepCheckpoint()


def test_local_checkpoint_store_tolerates_a_corrupt_index(tmp_path):
    store = LocalCheckpointStore(tmp_path / "ckpt")
    store.index_path.write_text("{not json", encoding="utf-8")

    assert store.load() == {}


def test_merge_volatility_columns():
    base = _sample_dataset()
    target = base.drop(columns=["vol_20d_annual"]).copy()
    target["forward_return_1d"] = target["forward_return_1d"] + 0.01
    merged = merge_volatility_columns(base, target)
    assert merged["vol_20d_annual"].notna().all()


def test_portfolio_backtester_copy_dataset_false_shares_reference():
    dataset = _sample_dataset().rename(columns={"forward_return_1d": "forward_return_20d"})
    dataset["composite_cycle_oriented_score"] = dataset["composite_growth_cycle_score"]
    config = PortfolioBacktestConfig(top_n_values=(2,), horizon=20, rebalance_interval_days=None, rebalance_step=1)
    backtester = PortfolioBacktester(dataset, config, copy_dataset=False)
    assert backtester.dataset is dataset


def test_run_parameter_sweep_with_checkpoint(tmp_path):
    checkpoint = SweepCheckpoint(tmp_path / "ckpt")
    dataset = _sample_dataset()
    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([0.5]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        checkpoint=checkpoint,
        watermark={"stock_scores_max_date": "20250102"},
    )
    assert payload["rows"]
    assert payload.get("sweep_checkpoint_completed", 0) >= 1

    # Second run should hit checkpoint and still return equivalent rows.
    payload_again = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=1,
        rebalance_intervals=[1],
        variants=build_sweep_variants([0.5]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        checkpoint=checkpoint,
        watermark={"stock_scores_max_date": "20250102"},
    )
    assert len(payload_again["rows"]) == len(payload["rows"])


def test_multi_horizon_sweep_submits_one_combined_outer_job_batch(monkeypatch):
    captured_batches = []

    monkeypatch.setattr(
        "analysis.sweep.portfolio_sweep_report.build_industry_exposure",
        lambda *args, **kwargs: {"periods": []},
    )

    def capture_outer_jobs(jobs, **kwargs):
        captured_batches.append(list(jobs))
        return []

    monkeypatch.setattr("analysis.sweep.portfolio_sweep_report.run_outer_jobs", capture_outer_jobs)

    payload = run_parameter_sweep(
        dataset=_sample_dataset(),
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=2,
        rebalance_intervals=[1, 2],
        variants=build_sweep_variants([0.5]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        independent_horizons=[1, 2],
        parallel_workers=2,
    )

    assert len(captured_batches) == 1
    assert {job.horizon for job in captured_batches[0]} == {1, 2}
    assert {(job.horizon, job.interval) for job in captured_batches[0]} == {(1, 1), (2, 2)}
    assert payload["rebalance_intervals"] == [1, 2]


def test_multi_horizon_sweep_runs_each_horizon_in_process_pool():
    dataset = _sample_dataset()
    dataset["forward_return_2d"] = dataset["forward_return_1d"] * 2

    payload = run_parameter_sweep(
        dataset=dataset,
        score_column="composite_growth_cycle_score",
        top_n_values=[2],
        horizon=2,
        rebalance_intervals=[1, 2],
        variants=build_sweep_variants([]),
        transaction_cost=0,
        index_benchmark_symbol="000852.SH",
        index_price_history=_index_prices(),
        independent_horizons=[1, 2],
        parallel_workers=2,
    )

    assert payload["rows"]
    assert {row["horizon"] for row in payload["rows"]} == {1, 2}
    assert all(row["horizon"] == row["rebalance_interval_days"] for row in payload["rows"])


def test_build_dataset_from_panel_reuses_price_cache():
    class _Db:
        stock_scores = None
        index_constituents = None

        def get_volume_price_data(self, symbol, start_date=None, end_date=None, projection=None):
            return [
                {"trade_date": "20250101", "open": 10, "high": 11, "low": 9, "close": 10},
                {"trade_date": "20250102", "open": 10, "high": 11, "low": 9, "close": 10.5},
                {"trade_date": "20250103", "open": 10.5, "high": 11.5, "low": 10, "close": 11},
            ]

    db = _Db()
    config = BacktestConfig(horizons=(1,), start_date="20250101", end_date="20250102", score_columns=("growth_score",))
    backtester = ScoreBacktester(db, config)
    panel = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "score_date": ["20250101"],
            "name": ["AAA 股份"],
            "industry": ["Tech"],
            "growth_score": [80.0],
        }
    )
    backtester._price_cache = {
        "AAA": [
            {"trade_date": "20250101", "open": 10, "high": 11, "low": 9, "close": 10},
            {"trade_date": "20250102", "open": 10, "high": 11, "low": 9, "close": 10.5},
            {"trade_date": "20250103", "open": 10.5, "high": 11.5, "low": 10, "close": 11},
        ]
    }
    dataset_off = backtester.build_dataset_from_panel(panel, trailing_stop_pct=None)
    dataset_stop = backtester.build_dataset_from_panel(panel, trailing_stop_pct=0.15)
    assert not dataset_off.empty
    assert not dataset_stop.empty
    assert dataset_off.iloc[0]["forward_return_1d"] == pytest.approx(0.05)
    assert dataset_stop.iloc[0]["forward_return_1d"] == pytest.approx(0.05)


def _rich_dataset():
    records = []
    industries = ["Tech", "Bank", "Drug", "Energy", "Auto"]
    for di, day in enumerate(range(1, 7)):
        for i in range(16):
            records.append(
                {
                    "score_date": f"202501{day:02d}",
                    "symbol": f"S{i:02d}",
                    "industry": industries[i % len(industries)],
                    "composite_growth_cycle_score": 100 - i + (di % 3),
                    "growth_score": 90 - i,
                    "cycle_score": 40 + i,
                    "forward_return_1d": 0.01 + ((i * 7 + di * 3) % 11) * 0.003 - 0.015,
                    "vol_20d_annual": 0.15 + ((i + di) % 5) * 0.06,
                    "ret_20": ((i * 5 + di) % 9) * 0.01 - 0.03,
                    "above_ma60": float((i + di) % 2),
                    "mdd60": -0.05 - (i % 4) * 0.01,
                }
            )
    return pd.DataFrame(records)


@pytest.mark.parametrize("volatility_mode,target_vol", [("off", None), ("cash_drag", 0.3), ("renormalize", 0.3)])
@pytest.mark.parametrize("path_gate", ["off", "ret20_pos"])
@pytest.mark.parametrize("construction_mode", ["top_n", "industry_capped"])
def test_selection_plan_matches_inline(volatility_mode, target_vol, path_gate, construction_mode):
    """The shared selection-plan path must be numerically identical to the inline path."""
    dataset = _rich_dataset()
    index_prices = [{"trade_date": f"202501{d:02d}", "close": 1000 + d * 3} for d in range(1, 12)]
    config = PortfolioBacktestConfig(
        score_column="composite_growth_cycle_score",
        horizon=1,
        top_n_values=(3, 5),
        rebalance_interval_days=1,
        transaction_cost=0.001,
        index_benchmark_symbol="000852.SH",
        construction_mode=construction_mode,
        max_industry_weight=0.4,
        volatility_mode=volatility_mode,
        target_vol_annual=target_vol,
        path_gate=path_gate,
    )

    inline = PortfolioBacktester(dataset, config, index_price_history=index_prices).run()
    cached = PortfolioBacktester(
        dataset,
        config,
        index_price_history=index_prices,
        sorted_groups={},
        selection_cache={},
    ).run()

    pd.testing.assert_frame_equal(
        inline.summary.reset_index(drop=True),
        cached.summary.reset_index(drop=True),
    )


def test_selection_cache_shared_across_volatility_variants():
    """A single shared cache reused across volatility variants keeps results correct."""
    dataset = _rich_dataset()
    index_prices = [{"trade_date": f"202501{d:02d}", "close": 1000 + d * 3} for d in range(1, 12)]
    sorted_groups: dict = {}
    selection_cache: dict = {}

    def _summary(volatility_mode, target_vol):
        config = PortfolioBacktestConfig(
            score_column="composite_growth_cycle_score",
            horizon=1,
            top_n_values=(3, 5),
            rebalance_interval_days=1,
            construction_mode="top_n",
            volatility_mode=volatility_mode,
            target_vol_annual=target_vol,
            path_gate="off",
        )
        shared = PortfolioBacktester(
            dataset, config, index_price_history=index_prices,
            sorted_groups=sorted_groups, selection_cache=selection_cache,
        ).run().summary
        isolated = PortfolioBacktester(dataset, config, index_price_history=index_prices).run().summary
        pd.testing.assert_frame_equal(shared.reset_index(drop=True), isolated.reset_index(drop=True))

    _summary("off", None)
    _summary("cash_drag", 0.3)
    _summary("renormalize", 0.3)
    # off + cash_drag + renormalize share one selection plan key (top_n/path_gate identical).
    assert len(selection_cache) == 1
    assert len(sorted_groups) == 1


def test_log_sweep_perf_emits_line(capsys):
    metrics = log_sweep_perf("unit_test", 0.5, combo_count=4)
    captured = capsys.readouterr().out
    assert "sweep_perf" in captured
    assert "process_peak_rss_mb=" in captured
    assert metrics.combo_count == 4


def test_ru_maxrss_conversion_uses_platform_units():
    assert _ru_maxrss_to_mb(5_075_316, "linux") == pytest.approx(4_956.363)
    assert _ru_maxrss_to_mb(5 * 1024 * 1024 * 1024, "darwin") == pytest.approx(5_120)


def test_cgroup_memory_metrics_reads_v2_files(tmp_path):
    (tmp_path / "memory.current").write_text(str(5 * 1024**3), encoding="utf-8")
    (tmp_path / "memory.peak").write_text(str(6 * 1024**3), encoding="utf-8")
    (tmp_path / "memory.max").write_text(str(8 * 1024**3), encoding="utf-8")
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 535\noom 0\noom_kill 2\n",
        encoding="utf-8",
    )

    metrics = cgroup_memory_metrics(tmp_path)

    assert metrics["current_mb"] == pytest.approx(5 * 1024)
    assert metrics["peak_mb"] == pytest.approx(6 * 1024)
    assert metrics["limit_mb"] == pytest.approx(8 * 1024)
    assert metrics["utilization_pct"] == pytest.approx(62.5)
    assert metrics["max_events"] == 535
    assert metrics["oom_kill_events"] == 2


def test_cgroup_memory_metrics_tolerates_unavailable_or_unlimited_files(tmp_path):
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    metrics = cgroup_memory_metrics(tmp_path)

    assert metrics["current_mb"] is None
    assert metrics["peak_mb"] is None
    assert metrics["limit_mb"] is None
    assert metrics["utilization_pct"] is None
    assert metrics["max_events"] is None
    assert metrics["oom_kill_events"] is None
