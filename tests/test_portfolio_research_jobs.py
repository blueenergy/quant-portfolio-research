import argparse
import signal
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tests.fakes import _Db
from tools import process_portfolio_research_jobs as worker
from tools import run_portfolio_research_loop as research_loop
from research_db import get_research_collections as get_trading_collections
from utils import job_queue


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        worker._set_active_job(None, None)


def test_process_portfolio_research_job_persists_result(monkeypatch):
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {
            "job_id": "job-1",
            "user_id": "user-1",
            "name": "research",
            "status": "pending",
            "params": {"universe_index": "csi1000"},
            "created_at": datetime.utcnow(),
        }
    )

    def fake_run_research_job(job, heartbeat=None):
        return {
            "result_id": "result-1",
            "artifact": {"html_path": "/tmp/report.html"},
            "data_watermark": {"stock_scores_max_date": "20250106"},
            "result_doc": {
                "result_id": "result-1",
                "job_id": job["job_id"],
                "user_id": job["user_id"],
                "candidate_strategy_config": {"preset_id": "research_strategy", "strategy_template_id": "growth_cycle_index_enhanced", "status": "draft"},
                "best_row": {"risk_adjusted_score": 1.2},
                "created_at": datetime.utcnow(),
            },
        }

    monkeypatch.setattr(worker, "run_research_job", fake_run_research_job)

    out = worker.process_one(cols, "worker-1")

    assert out == {"job_id": "job-1", "status": "completed", "result_id": "result-1"}
    assert cols.research_jobs.find_one({"job_id": "job-1"})["status"] == "completed"
    assert cols.research_results.find_one({"result_id": "result-1"})["candidate_strategy_config"]["preset_id"] == "research_strategy"
    status = cols.worker_status.find_one({"worker_type": "portfolio_research", "worker_id": "worker-1"})
    assert status["status"] == "completed"


def test_heartbeat_job_updates_progress_fields():
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {
            "job_id": "job-heartbeat",
            "status": "running",
            "created_at": datetime.utcnow(),
        }
    )

    worker._heartbeat_job(cols, "job-heartbeat", "building_dataset", "loading prices")

    doc = cols.research_jobs.find_one({"job_id": "job-heartbeat"})
    assert doc["progress_stage"] == "building_dataset"
    assert doc["progress_message"] == "loading prices"
    assert doc["heartbeat_at"] is not None
    assert doc["updated_at"] is not None


def test_requeue_stale_running_jobs_prefers_heartbeat_at():
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    stale = datetime.utcnow() - timedelta(minutes=90)
    fresh = datetime.utcnow() - timedelta(minutes=5)
    cols.research_jobs.insert_many(
        [
            {"job_id": "stale-heartbeat", "status": "running", "heartbeat_at": stale, "updated_at": fresh},
            {"job_id": "fresh-heartbeat", "status": "running", "heartbeat_at": fresh, "updated_at": stale},
        ]
    )

    count = worker._requeue_stale_running_jobs(cols, 60)

    assert count == 1
    assert cols.research_jobs.find_one({"job_id": "stale-heartbeat"})["status"] == "pending"
    assert cols.research_jobs.find_one({"job_id": "fresh-heartbeat"})["status"] == "running"


def test_process_one_updates_heartbeat_during_run(monkeypatch):
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {
            "job_id": "job-2",
            "user_id": "user-1",
            "name": "research",
            "status": "pending",
            "params": {"universe_index": "csi1000"},
            "created_at": datetime.utcnow(),
        }
    )
    seen_stages = []

    def fake_run_research_job(job, heartbeat=None):
        if heartbeat:
            heartbeat("building_dataset")
            seen_stages.append("building_dataset")
        return {
            "result_id": "result-2",
            "artifact": {"html_path": "/tmp/report.html"},
            "data_watermark": {"stock_scores_max_date": "20250106"},
            "result_doc": {
                "result_id": "result-2",
                "job_id": job["job_id"],
                "user_id": job["user_id"],
                "candidate_strategy_config": {"preset_id": "research_strategy", "strategy_template_id": "growth_cycle_index_enhanced", "status": "draft"},
                "best_row": {"risk_adjusted_score": 1.2},
                "created_at": datetime.utcnow(),
            },
        }

    monkeypatch.setattr(worker, "run_research_job", fake_run_research_job)

    out = worker.process_one(cols, "worker-2")

    assert out["status"] == "completed"
    doc = cols.research_jobs.find_one({"job_id": "job-2"})
    assert seen_stages == ["building_dataset"]
    assert doc["progress_stage"] == "building_dataset"
    assert doc["heartbeat_at"] is not None


def test_run_worker_resolves_trading_collections(monkeypatch, tmp_path):
    trading_db = _Db()
    cols = get_trading_collections(trading_db)

    monkeypatch.setattr("research_db.get_research_collections", lambda: cols)
    monkeypatch.setattr("research_db.ensure_research_indexes", lambda collections: None)

    args = argparse.Namespace(
        worker_id="worker-run",
        limit=1,
        poll_interval=0,
        stale_running_minutes=5,
        heartbeat_seconds=0,
        lock_file=str(tmp_path / "worker.lock"),
        json=False,
        verbose_empty=False,
    )

    assert worker._run_worker(args) == 0


def test_heartbeat_ticker_refreshes_running_job():
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {"job_id": "job-tick", "status": "running", "created_at": datetime.utcnow()}
    )

    ticker = worker._HeartbeatTicker(cols, "job-tick", interval_seconds=0)
    ticker._beat_once()

    doc = cols.research_jobs.find_one({"job_id": "job-tick"})
    assert doc["heartbeat_at"] is not None
    assert doc["updated_at"] is not None


def test_heartbeat_ticker_skips_non_running_job():
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {"job_id": "job-done", "status": "completed", "created_at": datetime.utcnow()}
    )

    worker._HeartbeatTicker(cols, "job-done", interval_seconds=0)._beat_once()

    doc = cols.research_jobs.find_one({"job_id": "job-done"})
    assert "heartbeat_at" not in doc


def test_claim_job_sets_heartbeat_at():
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {"job_id": "job-claim", "status": "pending", "created_at": datetime.utcnow()}
    )

    claimed = worker._claim_job(cols, "worker-claim")

    assert claimed["status"] == "running"
    assert claimed["heartbeat_at"] is not None
    assert claimed["heartbeat_at"] == claimed["started_at"]


def test_shutdown_signal_requeues_active_running_job():
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {"job_id": "job-active", "status": "running", "worker_id": "w1", "created_at": datetime.utcnow()}
    )
    worker._set_active_job(cols, "job-active")
    try:
        with pytest.raises(SystemExit) as exc:
            worker._handle_shutdown_signal(signal.SIGTERM, None)
        assert exc.value.code == 128 + int(signal.SIGTERM)

        doc = cols.research_jobs.find_one({"job_id": "job-active"})
        assert doc["status"] == "pending"
        assert doc["worker_id"] is None
        assert "signal" in doc["error_message"]
    finally:
        worker._set_active_job(None, None)


def test_shutdown_signal_without_active_job_is_noop():
    worker._set_active_job(None, None)
    with pytest.raises(SystemExit):
        worker._handle_shutdown_signal(signal.SIGINT, None)


def test_process_one_clears_active_job(monkeypatch):
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    cols.research_jobs.insert_one(
        {
            "job_id": "job-clear",
            "user_id": "user-1",
            "status": "pending",
            "params": {"universe_index": "csi1000"},
            "created_at": datetime.utcnow(),
        }
    )

    def fake_run_research_job(job, heartbeat=None):
        return {
            "result_id": "result-clear",
            "artifact": {"html_path": "/tmp/report.html"},
            "data_watermark": {"stock_scores_max_date": "20250106"},
            "result_doc": {"result_id": "result-clear", "job_id": job["job_id"], "user_id": job["user_id"], "created_at": datetime.utcnow()},
        }

    monkeypatch.setattr(worker, "run_research_job", fake_run_research_job)

    worker.process_one(cols, "worker-clear")

    # A leftover active job would make the next shutdown signal requeue a job
    # that already completed.
    assert job_queue.active_job_id() is None


def test_main_smoke_runs_end_to_end(monkeypatch, tmp_path):
    cols = get_trading_collections(_Db())

    monkeypatch.setattr("research_db.get_research_collections", lambda: cols)
    monkeypatch.setattr("research_db.ensure_research_indexes", lambda collections: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_portfolio_research_jobs.py",
            "--limit",
            "1",
            "--poll-interval",
            "0",
            "--lock-file",
            str(tmp_path / "worker.lock"),
        ],
    )

    assert worker.main() == 0


def test_research_loop_iteration_processes_up_to_limit(monkeypatch):
    trading_db = _Db()
    cols = get_trading_collections(trading_db)
    args = argparse.Namespace(
        worker_id="loop-worker",
        limit_per_iteration=2,
        heartbeat_seconds=0,
        stale_running_minutes=5,
        stale_check_interval_seconds=60,
    )
    calls = []

    class FakeWorker:
        @staticmethod
        def _requeue_stale_running_jobs(collections, stale_minutes):  # noqa: ARG004
            return 0

        @staticmethod
        def process_one(collections, worker_id, heartbeat_seconds):  # noqa: ARG004
            calls.append(worker_id)
            if len(calls) <= 3:
                return {"job_id": f"job-{len(calls)}", "status": "completed", "result_id": f"result-{len(calls)}"}
            return None

    monkeypatch.setattr(research_loop, "_log_processed", lambda result: None)

    processed_count, last_stale_check = research_loop._process_iteration(cols, FakeWorker, args, None)

    assert processed_count == 2
    assert calls == ["loop-worker", "loop-worker"]
    assert last_stale_check is not None


def test_research_loop_requeues_stale_jobs_when_due():
    cols = get_trading_collections(_Db())
    args = argparse.Namespace(worker_id="loop-worker", stale_running_minutes=5, stale_check_interval_seconds=60)
    calls = []

    class FakeWorker:
        @staticmethod
        def _requeue_stale_running_jobs(collections, stale_minutes):  # noqa: ARG004
            calls.append(stale_minutes)
            return 1

        @staticmethod
        def _record_worker_status(collections, worker_id, status, fields=None, increments=None):  # noqa: ARG004
            calls.append(status)

    last_stale_check, requeued_count = research_loop._maybe_requeue_stale_jobs(cols, FakeWorker, args, None)

    assert requeued_count == 1
    assert last_stale_check is not None
    assert calls == [5, "requeued_stale_jobs"]


def test_trailing_stop_values_from_params_prefers_list():
    values = worker._trailing_stop_values_from_params({"trailing_stop_pcts": [0, 0.15, 0.2]})
    assert values == [None, 0.15, 0.2]


def test_trailing_stop_values_from_params_falls_back_to_single_value():
    values = worker._trailing_stop_values_from_params({"trailing_stop_pct": 0.1})
    assert values == [0.1]


def test_trailing_stop_row_key_maps_disabled_values():
    assert worker._trailing_stop_row_key(None) == "off"
    assert worker._trailing_stop_row_key(0) == "off"
    assert worker._trailing_stop_row_key(0.15) == "0.15"


def test_candidate_strategy_config_benchmarks_against_its_own_universe():
    job = {"job_id": "job-1", "params": {"universe_index": "hs300"}}
    config = worker._candidate_strategy_config(job, {"top_n": 10, "rebalance_interval_days": 20})
    assert config["universe_index"] == "hs300"
    assert config["index_benchmark_symbol"] == "000300.SH"


def test_candidate_strategy_config_keeps_dynamic_threshold_fields():
    job = {
        "job_id": "job-1",
        "params": {
            "selection_mode": "dynamic_score_threshold",
            "threshold_lookback_days": 15,
            "max_positions": 25,
        },
    }

    config = worker._candidate_strategy_config(
        job,
        {"top_n": 10, "rebalance_interval_days": 20},
    )

    assert config["selection_mode"] == "dynamic_score_threshold"
    assert config["threshold_lookback_days"] == 15
    assert config["max_positions"] == 25


def test_candidate_strategy_config_keeps_explicit_benchmark_override():
    job = {"job_id": "job-1", "params": {"universe_index": "hs300", "index_benchmark_symbol": "000905.SH"}}
    config = worker._candidate_strategy_config(job, {"top_n": 10})
    assert config["index_benchmark_symbol"] == "000905.SH"


def test_candidate_strategy_config_keeps_cash_regime_mode():
    job = {"job_id": "job-1", "params": {"universe_index": "csi1000"}}

    cash_config = worker._candidate_strategy_config(
        job,
        {"top_n": 10, "regime_mode": "bull_g60_else_cash"},
    )
    always_invest_config = worker._candidate_strategy_config(
        job,
        {"top_n": 10, "regime_mode": "off"},
    )

    assert cash_config["regime_cash"] is True
    assert "regime_cash" not in always_invest_config


def test_candidate_strategy_config_keeps_volume_price_regime_rule():
    job = {
        "job_id": "job-1",
        "params": {
            "universe_index": "csi1000",
            "regime_rule": "k_slow_anchor_with_vol_and_atr",
        },
    }

    cash_config = worker._candidate_strategy_config(
        job,
        {"top_n": 10, "regime_mode": "bull_g60_else_cash"},
    )
    always_invest_config = worker._candidate_strategy_config(
        job,
        {"top_n": 10, "regime_mode": "off"},
    )

    assert cash_config["regime_rule"] == "k_slow_anchor_with_vol_and_atr"
    assert "regime_rule" not in always_invest_config


def test_run_research_job_passes_volume_price_regime_rule(monkeypatch):
    from tools.process_portfolio_research_jobs import run_research_job

    captured = {}
    panel = pd.DataFrame({"score_date": ["20240102"], "symbol": ["000001.SZ"]})

    def fake_schedule(*args, **kwargs):
        captured["kwargs"] = kwargs
        return [], [{"score_date": "20240102", "label": "bull", "action": "invest"}]

    monkeypatch.setattr("shared.data_access.open_research_data_access", lambda: type("Db", (), {})())
    monkeypatch.setattr(
        "tools.process_portfolio_research_jobs._latest_stock_score_date",
        lambda *args, **kwargs: "20240102",
    )
    monkeypatch.setattr(
        "analysis.core.benchmark_prices.load_index_benchmark_prices",
        lambda *args, **kwargs: [{"trade_date": "20240102", "close": 1.0, "vol": 100.0, "high": 1.1, "low": 0.9}],
    )
    monkeypatch.setattr(
        "analysis.sweep.sweep_engine.load_panel_cache",
        lambda *args, **kwargs: (panel.copy(), {}, None),
    )
    monkeypatch.setattr("shared.regime_label.build_regime_cash_schedule", fake_schedule)
    monkeypatch.setattr(
        "tools.process_portfolio_research_jobs._data_watermark",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "shared.industry_universe.attach_sw_l1",
        lambda dataset, *args, **kwargs: dataset,
    )

    class _StubBacktester:
        def __init__(self, *args, **kwargs):
            self.price_cache = {}

        def prepare_score_panel(self, *args, **kwargs):
            return panel.copy()

        def build_dataset_from_panel(self, *args, **kwargs):
            return panel.copy()

        def add_live_cash_return_columns(self, dataset, *args, **kwargs):
            return dataset

    monkeypatch.setattr("analysis.sweep.portfolio_sweep_report.ScoreBacktester", _StubBacktester)
    monkeypatch.setattr(
        "analysis.sweep.score_columns.add_weighted_score_columns",
        lambda dataset, *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        "analysis.sweep.portfolio_sweep_report.run_parameter_sweep",
        lambda **kwargs: {"rows": []},
    )

    job = {
        "job_id": "test-regime-rule",
        "params": {
            "start_date": "20240101",
            "end_date": "20240102",
            "universe_index": "csi1000",
            "score_column": "fundamental_score",
            "growth_cycle_weights": ["30:70"],
            "top_n_values": [5],
            "horizon": 20,
            "regime_cash": True,
            "regime_rule": "k_slow_anchor_with_vol_and_atr",
            "use_panel_cache": True,
            "resume_checkpoint": False,
        },
    }

    try:
        run_research_job(job)
    except Exception as exc:
        captured["error"] = repr(exc)

    assert captured.get("kwargs", {}).get("regime_rule") == "k_slow_anchor_with_vol_and_atr", captured


@pytest.mark.parametrize(
    ("universe", "expected"),
    [("csi1000", "000852.SH"), ("csi500", "000905.SH"), ("hs300", "000300.SH"), ("star50", "000688.SH")],
)
def test_candidate_strategy_config_resolves_benchmark_per_universe(universe, expected):
    job = {"job_id": "job-1", "params": {"universe_index": universe}}
    config = worker._candidate_strategy_config(job, {"top_n": 10})
    assert config["index_benchmark_symbol"] == expected


def test_sweep_checkpoint_stays_on_disk_for_a_local_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND", "local")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_CHECKPOINT_DIR", str(tmp_path / "ckpt"))

    checkpoint = worker._sweep_checkpoint("job-1")

    assert checkpoint.directory == tmp_path / "ckpt" / "job-1"
    assert checkpoint.describe().endswith("completed_jobs.json")


def test_sweep_checkpoint_goes_to_s3_when_artifacts_do(monkeypatch):
    # The K8s workers are replaced mid-sweep by image rollouts, so their
    # checkpoint cannot live on the pod's own emptyDir.
    monkeypatch.setenv("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND", "s3")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_BUCKET", "aipoc")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_SECRET_KEY", "sk")

    checkpoint = worker._sweep_checkpoint("job-1")

    assert checkpoint.directory is None
    assert checkpoint.describe() == (
        "s3://aipoc/portfolio-research/job-1/checkpoint/completed_jobs.json"
    )
