"""Shared job queue mechanics.

Tested against mongomock rather than a hand-rolled double, because the parts
that matter here are exactly the ones a simplified fake tends to get wrong:
``find_one_and_update`` honouring the sort, and ``$or`` / ``$exists`` deciding
which running jobs count as stale.
"""

import logging
import signal
import threading
import time
from datetime import datetime, timedelta

import mongomock
import numpy as np
import pytest
from pymongo.errors import AutoReconnect

from utils.job_queue import (
    HeartbeatTicker,
    JobQueue,
    active_job_id,
    clean_for_mongo,
    handle_shutdown_signal,
    run_worker_forever,
    set_active_job,
    worker_lock,
)


def _raise(error: Exception):
    def _boom(*_args, **_kwargs):
        raise error

    return _boom


@pytest.fixture
def db():
    return mongomock.MongoClient().jobs_test


@pytest.fixture
def queue(db):
    return JobQueue(db["factor_backtest_jobs"], worker_type="factor_backtest", worker_status=db["worker_status"])


@pytest.fixture(autouse=True)
def _clear_active_job():
    yield
    set_active_job(None, None)


def _pending(job_id: str, created_at: datetime = None, **extra):
    return {
        "job_id": job_id,
        "status": "pending",
        "created_at": created_at or datetime.utcnow(),
        **extra,
    }


# ===================== claim =====================
def test_claim_takes_the_oldest_pending_job_first(queue):
    now = datetime.utcnow()
    # Inserted newest-first, so insertion order cannot pass for created_at order.
    queue.jobs.insert_many(
        [
            _pending("newest", now),
            _pending("oldest", now - timedelta(hours=2)),
            _pending("middle", now - timedelta(hours=1)),
        ]
    )

    assert queue.claim("w1")["job_id"] == "oldest"
    assert queue.claim("w1")["job_id"] == "middle"
    assert queue.claim("w1")["job_id"] == "newest"


def test_claim_hands_a_job_to_exactly_one_worker(queue):
    queue.jobs.insert_one(_pending("only"))

    first = queue.claim("w1")
    second = queue.claim("w2")

    assert first["job_id"] == "only"
    assert second is None
    assert queue.jobs.find_one({"job_id": "only"})["worker_id"] == "w1"


def test_claim_returns_none_on_an_empty_queue(queue):
    assert queue.claim("w1") is None


def test_claim_respects_worker_pool(queue):
    queue.jobs.insert_many(
        [
            _pending("default"),
            {**_pending("burst"), "worker_pool": "entry-filter-mvp"},
        ]
    )
    assert queue.claim("w1")["job_id"] == "default"
    assert queue.claim("w1", worker_pool="entry-filter-mvp")["job_id"] == "burst"
    assert queue.claim("w1") is None
    assert queue.claim("w1", worker_pool="entry-filter-mvp") is None


def test_claim_marks_the_job_running_with_a_fresh_heartbeat(queue):
    queue.jobs.insert_one(_pending("job-1", error_message="failed last time"))

    claimed = queue.claim("w1")

    assert claimed["status"] == "running"
    assert claimed["worker_id"] == "w1"
    assert claimed["heartbeat_at"] is not None
    assert claimed["started_at"] is not None
    # A retry must not keep showing the previous attempt's error.
    assert claimed["error_message"] is None


def test_claim_counts_attempts_across_retries(queue):
    queue.jobs.insert_one(_pending("job-1"))

    assert queue.claim("w1")["attempts"] == 1
    queue.release("job-1", "requeued")
    assert queue.claim("w1")["attempts"] == 2


def test_claim_ignores_jobs_that_are_not_pending(queue):
    queue.jobs.insert_many(
        [
            {"job_id": "done", "status": "completed", "created_at": datetime.utcnow()},
            {"job_id": "broken", "status": "failed", "created_at": datetime.utcnow()},
            {"job_id": "busy", "status": "running", "created_at": datetime.utcnow()},
        ]
    )

    assert queue.claim("w1") is None


# ===================== stale requeue =====================
def test_requeue_stale_uses_heartbeat_at_over_updated_at(queue):
    stale = datetime.utcnow() - timedelta(minutes=90)
    fresh = datetime.utcnow() - timedelta(minutes=5)
    queue.jobs.insert_many(
        [
            {"job_id": "dead", "status": "running", "heartbeat_at": stale, "updated_at": fresh},
            # Still beating, just hasn't changed stage in a while: must be left alone.
            {"job_id": "slow", "status": "running", "heartbeat_at": fresh, "updated_at": stale},
        ]
    )

    assert queue.requeue_stale(60) == 1
    assert queue.jobs.find_one({"job_id": "dead"})["status"] == "pending"
    assert queue.jobs.find_one({"job_id": "slow"})["status"] == "running"


def test_requeue_stale_falls_back_to_updated_at_when_never_beat(queue):
    old = datetime.utcnow() - timedelta(minutes=90)
    queue.jobs.insert_one({"job_id": "no-beat", "status": "running", "updated_at": old})

    assert queue.requeue_stale(60) == 1
    assert queue.jobs.find_one({"job_id": "no-beat"})["status"] == "pending"


def test_requeue_stale_clears_the_worker_and_says_why(queue):
    old = datetime.utcnow() - timedelta(minutes=90)
    queue.jobs.insert_one({"job_id": "dead", "status": "running", "worker_id": "w1", "heartbeat_at": old})

    queue.requeue_stale(60)

    doc = queue.jobs.find_one({"job_id": "dead"})
    assert doc["worker_id"] is None
    assert "stale for 60 minutes" in doc["error_message"]


def test_requeue_stale_is_disabled_by_a_non_positive_threshold(queue):
    old = datetime.utcnow() - timedelta(days=7)
    queue.jobs.insert_one({"job_id": "dead", "status": "running", "heartbeat_at": old})

    assert queue.requeue_stale(0) == 0
    assert queue.jobs.find_one({"job_id": "dead"})["status"] == "running"


def test_requeue_stale_leaves_finished_jobs_alone(queue):
    old = datetime.utcnow() - timedelta(days=7)
    queue.jobs.insert_many(
        [
            {"job_id": "done", "status": "completed", "heartbeat_at": old},
            {"job_id": "broken", "status": "failed", "heartbeat_at": old},
        ]
    )

    assert queue.requeue_stale(60) == 0


# ===================== release, heartbeat, fail, complete =====================
def test_release_returns_a_running_job_to_the_queue(queue):
    queue.jobs.insert_one(_pending("job-1"))
    queue.claim("w1")

    queue.release("job-1", "worker redeployed")

    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["status"] == "pending"
    assert doc["worker_id"] is None
    assert doc["error_message"] == "worker redeployed"


def test_release_does_not_resurrect_a_finished_job(queue):
    queue.jobs.insert_one({"job_id": "done", "status": "completed"})

    queue.release("done", "worker redeployed")

    assert queue.jobs.find_one({"job_id": "done"})["status"] == "completed"


def test_heartbeat_records_the_stage_and_message(queue):
    queue.jobs.insert_one(_pending("job-1"))

    queue.heartbeat("job-1", "running_diagnostics", "quantile diagnostics on 20 factors")

    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["progress_stage"] == "running_diagnostics"
    assert doc["progress_message"] == "quantile diagnostics on 20 factors"
    assert doc["heartbeat_at"] is not None


def test_heartbeat_without_a_message_keeps_the_previous_one(queue):
    queue.jobs.insert_one(_pending("job-1"))
    queue.heartbeat("job-1", "loading_prices", "reading prices")

    queue.heartbeat("job-1", "computing_factors")

    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["progress_stage"] == "computing_factors"
    assert doc["progress_message"] == "reading prices"


def test_fail_records_the_error_text(queue):
    queue.jobs.insert_one(_pending("job-1"))

    queue.fail("job-1", ValueError("158 factors x 17y needs about 12.2 GB"))

    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["status"] == "failed"
    assert "12.2 GB" in doc["error_message"]
    assert doc["completed_at"] is not None


def test_complete_stores_whatever_the_worker_points_at(queue):
    queue.jobs.insert_one(_pending("job-1"))

    queue.complete("job-1", result_id="res-1", artifact={"json_path": "/tmp/report.json"})

    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["status"] == "completed"
    assert doc["result_id"] == "res-1"
    assert doc["artifact"] == {"json_path": "/tmp/report.json"}
    assert doc["completed_at"] is not None


# ===================== worker status =====================
def test_worker_status_upserts_then_accumulates_counters(queue):
    queue.record_worker_status("w1", "running", {"current_job_id": "job-1"})
    queue.record_worker_status("w1", "completed", {"current_job_id": None}, {"processed_count": 1})
    queue.record_worker_status("w1", "completed", {"current_job_id": None}, {"processed_count": 1})

    doc = queue.worker_status.find_one({"worker_type": "factor_backtest", "worker_id": "w1"})
    assert doc["status"] == "completed"
    assert doc["processed_count"] == 2
    assert doc["created_at"] is not None


def test_worker_status_is_scoped_per_worker_type(db):
    factor = JobQueue(db["a"], worker_type="factor_backtest", worker_status=db["worker_status"])
    research = JobQueue(db["b"], worker_type="portfolio_research", worker_status=db["worker_status"])

    factor.record_worker_status("same-host", "running")
    research.record_worker_status("same-host", "idle")

    assert db["worker_status"].count_documents({"worker_id": "same-host"}) == 2
    assert factor.worker_status.find_one({"worker_type": "factor_backtest"})["status"] == "running"


def test_worker_status_is_a_noop_without_the_collection(db):
    queue = JobQueue(db["a"], worker_type="factor_backtest")
    queue.record_worker_status("w1", "running")  # must not raise


def test_worker_status_survives_a_mongo_blip(queue, monkeypatch, caplog):
    """Production lost both replicas at once to this write, so it may not raise.

    The observed error was ``_OperationCancelled``, a subclass of the public
    ``AutoReconnect`` used here.
    """
    monkeypatch.setattr(queue.worker_status, "update_one", _raise(AutoReconnect("pool paused")))

    with caplog.at_level(logging.WARNING):
        queue.record_worker_status("w1", "running")

    assert "worker status write failed" in caplog.text


def test_worker_status_still_raises_a_non_mongo_error(queue, monkeypatch):
    # An unencodable field is a bug in the caller, not a blip, so it stays loud.
    monkeypatch.setattr(queue.worker_status, "update_one", _raise(TypeError("not encodable")))

    with pytest.raises(TypeError):
        queue.record_worker_status("w1", "running")


# ===================== isolation between queues =====================
def test_two_queues_do_not_see_each_other_s_jobs(db):
    factor = JobQueue(db["factor_backtest_jobs"], worker_type="factor_backtest")
    research = JobQueue(db["portfolio_research_jobs"], worker_type="portfolio_research")
    factor.jobs.insert_one(_pending("factor-1"))
    research.jobs.insert_one(_pending("research-1"))

    assert factor.claim("w1")["job_id"] == "factor-1"
    assert research.claim("w1")["job_id"] == "research-1"
    # A long factor job cannot delay the research queue, which is the whole
    # reason the two run in separate workers.
    assert factor.claim("w1") is None
    assert research.claim("w1") is None


# ===================== heartbeat ticker =====================
def test_ticker_refreshes_a_running_job_in_the_background(queue):
    queue.jobs.insert_one(_pending("job-1"))
    queue.claim("w1")
    before = queue.jobs.find_one({"job_id": "job-1"})["heartbeat_at"]

    ticker = HeartbeatTicker(queue, "job-1", 0.01).start()
    try:
        deadline = time.time() + 2
        while time.time() < deadline:
            if queue.jobs.find_one({"job_id": "job-1"})["heartbeat_at"] > before:
                break
            time.sleep(0.01)
    finally:
        ticker.stop()

    assert queue.jobs.find_one({"job_id": "job-1"})["heartbeat_at"] > before


def test_ticker_does_not_beat_for_a_job_that_stopped_running(queue):
    queue.jobs.insert_one({"job_id": "done", "status": "completed"})

    HeartbeatTicker(queue, "done", 0.01)._beat_once()

    assert "heartbeat_at" not in queue.jobs.find_one({"job_id": "done"})


def test_ticker_with_a_zero_interval_starts_no_thread(queue):
    before = threading.active_count()
    ticker = HeartbeatTicker(queue, "job-1", 0).start()
    try:
        assert threading.active_count() == before
    finally:
        ticker.stop()


# ===================== worker lock =====================
def test_worker_lock_blocks_a_second_holder(tmp_path):
    path = str(tmp_path / "worker.lock")
    with worker_lock(path, "factor backtest"):
        with pytest.raises(RuntimeError, match="another factor backtest worker holds lock"):
            with worker_lock(path, "factor backtest"):
                pass


def test_worker_lock_frees_the_path_on_exit(tmp_path):
    path = str(tmp_path / "worker.lock")
    with worker_lock(path, "factor backtest"):
        pass
    with worker_lock(path, "factor backtest"):
        pass  # must not raise


def test_worker_lock_creates_missing_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "worker.lock")
    with worker_lock(path, "factor backtest"):
        pass


# ===================== shutdown handling =====================
def test_shutdown_hands_the_active_job_back(queue):
    queue.jobs.insert_one(_pending("job-1"))
    queue.claim("w1")
    set_active_job(queue, "job-1")

    with pytest.raises(SystemExit) as exc:
        handle_shutdown_signal(signal.SIGTERM, None)

    assert exc.value.code == 128 + int(signal.SIGTERM)
    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["status"] == "pending"
    assert "signal" in doc["error_message"]


def test_shutdown_without_an_active_job_just_exits(queue):
    set_active_job(None, None)

    with pytest.raises(SystemExit):
        handle_shutdown_signal(signal.SIGINT, None)


def test_active_job_id_tracks_the_in_flight_job(queue):
    assert active_job_id() is None
    set_active_job(queue, "job-1")
    assert active_job_id() == "job-1"
    set_active_job(None, None)
    assert active_job_id() is None


# ===================== clean_for_mongo =====================
def test_clean_for_mongo_unwraps_numpy_scalars():
    cleaned = clean_for_mongo({"ic": np.float64(0.031), "days": np.int64(240)})

    assert cleaned == {"ic": pytest.approx(0.031), "days": 240}
    assert isinstance(cleaned["ic"], float)
    assert isinstance(cleaned["days"], int)


def test_clean_for_mongo_replaces_nan_and_inf_with_none():
    assert clean_for_mongo({"t": float("nan"), "ir": float("inf"), "x": float("-inf")}) == {
        "t": None,
        "ir": None,
        "x": None,
    }


def test_clean_for_mongo_walks_nested_containers_and_tuples():
    report = {"factors": [{"name": "MA20", "quantiles": (np.float64(1.0), float("nan"))}]}

    assert clean_for_mongo(report) == {"factors": [{"name": "MA20", "quantiles": [1.0, None]}]}


def test_clean_for_mongo_stringifies_non_string_keys():
    assert clean_for_mongo({5: "five", np.int64(10): "ten"}) == {"5": "five", "10": "ten"}


# ===================== own-orphan recovery =====================
def test_requeue_own_orphans_reclaims_what_this_worker_left_running(queue):
    queue.jobs.insert_one(_pending("job-1"))
    queue.claim("pod-a")

    assert queue.requeue_own_orphans("pod-a") == 1
    doc = queue.jobs.find_one({"job_id": "job-1"})
    assert doc["status"] == "pending"
    assert doc["worker_id"] is None
    assert "restarted" in doc["error_message"]


def test_requeue_own_orphans_leaves_another_worker_s_job_alone(queue):
    queue.jobs.insert_many([_pending("mine"), _pending("theirs")])
    queue.claim("pod-a")
    queue.claim("pod-b")

    assert queue.requeue_own_orphans("pod-a") == 1
    # pod-b may still be alive and running it; only its own restart may reclaim it.
    assert queue.jobs.find_one({"worker_id": "pod-b"})["status"] == "running"


def test_requeue_own_orphans_ignores_finished_jobs(queue):
    queue.jobs.insert_one(_pending("job-1"))
    queue.claim("pod-a")
    queue.complete("job-1", result_id="res-1")

    assert queue.requeue_own_orphans("pod-a") == 0
    assert queue.jobs.find_one({"job_id": "job-1"})["status"] == "completed"


def test_requeue_own_orphans_recovers_faster_than_the_stale_window(queue):
    # The job was claimed just now, so the stale threshold cannot see it yet.
    queue.jobs.insert_one(_pending("job-1"))
    queue.claim("pod-a")

    assert queue.requeue_stale(120) == 0
    assert queue.requeue_own_orphans("pod-a") == 1


# ===================== run_worker_forever =====================
@pytest.fixture
def slept(monkeypatch):
    """Collect the sleeps instead of taking them, so backoff is assertable."""
    recorded = []
    monkeypatch.setattr("utils.job_queue.time.sleep", recorded.append)
    return recorded


def _steps(*outcomes):
    """A step function that replays one outcome per call, raising the exceptions."""
    remaining = list(outcomes)

    def _step():
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

    return _step


def test_run_worker_forever_gives_up_after_consecutive_failures(slept):
    rc = run_worker_forever(
        _raise(AutoReconnect("mongo is down")),
        interval_seconds=5,
        max_consecutive_failures=3,
        logger=logging.getLogger("test"),
    )

    assert rc == 1
    # Two backoffs before the third failure ends it, each double the last.
    assert slept == [5.0, 10.0]


def test_run_worker_forever_caps_the_backoff(slept):
    rc = run_worker_forever(
        _raise(AutoReconnect("mongo is down")),
        interval_seconds=30,
        max_consecutive_failures=5,
        logger=logging.getLogger("test"),
        max_backoff_seconds=60.0,
    )

    assert rc == 1
    assert slept == [30.0, 60.0, 60.0, 60.0]


def test_run_worker_forever_resets_the_count_after_a_good_iteration(slept):
    # Alternating failures would trip a threshold of 2 if the count never reset.
    step = _steps(AutoReconnect("blip"), None, AutoReconnect("blip"), None, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_worker_forever(
            step,
            interval_seconds=5,
            max_consecutive_failures=2,
            logger=logging.getLogger("test"),
        )


def test_run_worker_forever_sleeps_the_poll_interval_between_good_iterations(slept):
    with pytest.raises(KeyboardInterrupt):
        run_worker_forever(
            _steps(None, None, KeyboardInterrupt()),
            interval_seconds=7,
            max_consecutive_failures=3,
            logger=logging.getLogger("test"),
        )

    assert slept == [7.0, 7.0]
