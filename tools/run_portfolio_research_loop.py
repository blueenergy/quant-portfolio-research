#!/usr/bin/env python3
"""Continuously process portfolio research jobs with low pickup latency."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional, Tuple

log = logging.getLogger("portfolio_research_loop")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously process portfolio research jobs.")
    parser.add_argument("--worker-id", default="quant-scorer-portfolio-research", help="Worker id written to claimed jobs")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("PORTFOLIO_RESEARCH_LOOP_INTERVAL_SECONDS", "5")),
        help="Seconds to sleep between polling iterations",
    )
    parser.add_argument(
        "--limit-per-iteration",
        type=int,
        default=int(os.getenv("PORTFOLIO_RESEARCH_LOOP_LIMIT", "2")),
        help="Maximum jobs to process per polling iteration",
    )
    parser.add_argument(
        "--stale-running-minutes",
        type=int,
        default=int(os.getenv("PORTFOLIO_RESEARCH_STALE_MINUTES", "5")),
        help="Requeue running jobs whose heartbeat is older than this many minutes",
    )
    parser.add_argument(
        "--stale-check-interval-seconds",
        type=float,
        default=float(os.getenv("PORTFOLIO_RESEARCH_STALE_CHECK_INTERVAL_SECONDS", "60")),
        help="Seconds between stale-running job recovery checks",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("PORTFOLIO_RESEARCH_HEARTBEAT_SECONDS", "30")),
        help="Background heartbeat interval while a job runs",
    )
    parser.add_argument(
        "--lock-file",
        default=os.getenv("PORTFOLIO_RESEARCH_LOCK_FILE", "/tmp/portfolio_research_worker.lock"),
        help="Exclusive lock file to prevent overlapping research workers",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=int(os.getenv("PORTFOLIO_RESEARCH_MAX_CONSECUTIVE_FAILURES", "10")),
        help="Exit non-zero once this many iterations fail in a row",
    )
    return parser.parse_args()


def _log_processed(result) -> None:
    status = result.get("status")
    job_id = result.get("job_id")
    result_id = result.get("result_id")
    reason = result.get("error_message")
    detail = f" result_id={result_id}" if result_id else ""
    if reason:
        detail += f" error={reason}"
    message = f"job processed: status={status} job_id={job_id}{detail}"
    # a failed job is the worker doing its job, but it still needs to stand out
    log.warning(message) if reason else log.info(message)


def _maybe_requeue_stale_jobs(collections, worker, args: argparse.Namespace, last_stale_check: Optional[float]) -> Tuple[Optional[float], int]:
    now = time.monotonic()
    if last_stale_check is not None and now - last_stale_check < max(1.0, float(args.stale_check_interval_seconds)):
        return last_stale_check, 0

    requeued_count = worker._requeue_stale_running_jobs(collections, args.stale_running_minutes)
    if requeued_count:
        message = f"requeued {requeued_count} stale running research jobs"
        worker._record_worker_status(collections, args.worker_id, "requeued_stale_jobs", {"current_job_id": None, "last_message": message})
        log.warning(message)
    return now, requeued_count


def _process_iteration(collections, worker, args: argparse.Namespace, last_stale_check: Optional[float]) -> Tuple[int, Optional[float]]:
    last_stale_check, _ = _maybe_requeue_stale_jobs(collections, worker, args, last_stale_check)

    processed_count = 0
    while processed_count < args.limit_per_iteration:
        result = worker.process_one(collections, args.worker_id, args.heartbeat_seconds)
        if not result:
            break
        processed_count += 1
        _log_processed(result)
    return processed_count, last_stale_check


def main() -> int:
    from observability.logging_setup import configure_worker_logging
    from research_db import ensure_research_indexes, get_research_collections
    from tools import process_portfolio_research_jobs as worker
    from utils.job_queue import WorkerLockHeld, run_worker_forever
    from utils.portfolio_research_s3 import should_use_s3_backend

    configure_worker_logging()
    args = _parse_args()
    try:
        artifact_backend = "s3" if should_use_s3_backend() else "local"
        log.info("artifact backend: %s", artifact_backend)
        with worker._worker_lock(args.lock_file):
            worker._install_shutdown_handlers()
            collections = get_research_collections()
            ensure_research_indexes(collections)
            log.info(
                "loop started: interval_seconds=%s limit_per_iteration=%s stale_check_interval_seconds=%s",
                args.interval_seconds,
                args.limit_per_iteration,
                args.stale_check_interval_seconds,
            )

            last_stale_check = None

            def _step() -> None:
                nonlocal last_stale_check
                _, last_stale_check = _process_iteration(collections, worker, args, last_stale_check)

            return run_worker_forever(
                _step,
                interval_seconds=float(args.interval_seconds),
                max_consecutive_failures=args.max_consecutive_failures,
                logger=log,
            )
    except WorkerLockHeld as exc:
        log.info("%s; exiting without work", exc)
        return 0
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
        return 0
    except Exception:
        # previously this returned 0, so a crashed worker looked like a clean exit
        log.exception("loop aborted")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
