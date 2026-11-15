"""Mongo-backed job queue mechanics, shared by the worker loops.

Every long-running worker in this repo needs the same things: claim one pending
job atomically, keep a heartbeat alive while it runs, hand the job back if the
process dies or gets redeployed, and record what the worker is doing so an
operator can see queue depth. Those mechanics do not depend on what the job
computes, so they live here and each worker binds them to its own collection.

A worker differs from another only in ``worker_type`` and the collection it
reads, which is what ``JobQueue`` carries. Job documents are expected to have
``job_id``, ``status`` in {pending, running, completed, failed}, and
``created_at``; everything else belongs to the worker.
"""

from __future__ import annotations

import fcntl
import logging
import math
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import PyMongoError

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.utcnow()


def clean_for_mongo(value: Any) -> Any:
    """Make a computed result storable: no numpy scalars, no NaN or inf.

    Mongo rejects NaN and inf in some drivers and stores numpy scalars as
    opaque binary, so a report built by pandas has to be walked before it is
    written next to the job.
    """
    if isinstance(value, dict):
        return {str(key): clean_for_mongo(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_mongo(item) for item in value]
    if hasattr(value, "item"):
        try:
            return clean_for_mongo(value.item())
        except Exception:
            # not a scalar numpy value after all; fall through to the checks below
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@dataclass(frozen=True)
class JobQueue:
    """One worker type's view of its job collection.

    ``worker_status`` is optional so a caller holding a collection bundle that
    predates it still works; status recording then becomes a no-op instead of
    an attribute error.
    """

    jobs: Any
    worker_type: str
    worker_status: Any = None

    def _pending_query(self, worker_pool: Optional[str] = None) -> Dict[str, Any]:
        query: Dict[str, Any] = {"status": "pending"}
        pool = str(worker_pool or "").strip()
        if pool:
            query["worker_pool"] = pool
            return query
        query["$or"] = [
            {"worker_pool": {"$exists": False}},
            {"worker_pool": None},
            {"worker_pool": ""},
        ]
        return query

    def claim(self, worker_id: str, *, worker_pool: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Take the oldest pending job, or return None when the queue is empty.

        ``find_one_and_update`` is atomic, so several workers may poll the same
        queue without ever handing the same job out twice.

        When ``worker_pool`` is set, only jobs tagged with that pool are claimed.
        When omitted/empty, only jobs without a dedicated pool are claimed.
        """
        now = utcnow()
        return self.jobs.find_one_and_update(
            self._pending_query(worker_pool),
            {
                "$set": {
                    "status": "running",
                    "worker_id": worker_id,
                    "started_at": now,
                    "updated_at": now,
                    "heartbeat_at": now,
                    "error_message": None,
                },
                "$inc": {"attempts": 1},
            },
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    def requeue_stale(self, stale_minutes: int) -> int:
        """Hand back jobs whose worker died without failing them.

        The threshold has to exceed the heartbeat interval by a wide margin: a
        job that is merely slow must never be requeued while it still runs, or
        two workers end up computing it at once.
        """
        if stale_minutes <= 0:
            return 0
        now = utcnow()
        cutoff = now - timedelta(minutes=stale_minutes)
        stale_filter = {
            "status": "running",
            "$or": [
                {"heartbeat_at": {"$lt": cutoff}},
                {"heartbeat_at": {"$exists": False}, "updated_at": {"$lt": cutoff}},
            ],
        }
        result = self.jobs.update_many(
            stale_filter,
            {
                "$set": {
                    "status": "pending",
                    "worker_id": None,
                    "error_message": f"requeued after running stale for {stale_minutes} minutes",
                    "updated_at": now,
                }
            },
        )
        return int(getattr(result, "modified_count", 0) or 0)

    def requeue_own_orphans(self, worker_id: str) -> int:
        """Hand back jobs this worker still owns. Call at startup only.

        A container that was OOM killed leaves its job marked running with a
        heartbeat that only goes stale much later, so the job would sit idle
        for the whole stale window even though the process restarted seconds
        ago. A freshly started worker is running nothing, so anything still
        claimed under its own id is orphaned and can go back right away.

        This is safe only because a worker id maps to one process: the file
        lock enforces that within a host, and on Kubernetes the id is the pod
        name. Two processes sharing an id would requeue each other's work.
        """
        now = utcnow()
        result = self.jobs.update_many(
            {"status": "running", "worker_id": worker_id},
            {
                "$set": {
                    "status": "pending",
                    "worker_id": None,
                    "error_message": f"requeued after worker {worker_id} restarted",
                    "updated_at": now,
                }
            },
        )
        return int(getattr(result, "modified_count", 0) or 0)

    def release(self, job_id: str, message: str) -> None:
        """Put a running job back in the queue, e.g. on shutdown."""
        now = utcnow()
        self.jobs.update_one(
            {"job_id": job_id, "status": "running"},
            {
                "$set": {
                    "status": "pending",
                    "worker_id": None,
                    "error_message": message,
                    "updated_at": now,
                }
            },
        )

    def heartbeat(self, job_id: str, stage: str, message: Optional[str] = None) -> None:
        """Record which stage the job reached and that it is still alive."""
        now = utcnow()
        update: Dict[str, Any] = {
            "updated_at": now,
            "heartbeat_at": now,
            "progress_stage": stage,
        }
        if message is not None:
            update["progress_message"] = message
        self.jobs.update_one({"job_id": job_id}, {"$set": update})

    def fail(self, job_id: str, error: BaseException) -> None:
        now = utcnow()
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error_message": str(error), "completed_at": now, "updated_at": now}},
        )

    def complete(self, job_id: str, **fields: Any) -> None:
        """Mark the job done, plus whatever the worker wants to point at."""
        now = utcnow()
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "completed", "completed_at": now, "updated_at": now, **fields}},
        )

    def record_worker_status(
        self,
        worker_id: str,
        status: str,
        fields: Optional[Dict[str, Any]] = None,
        increments: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.worker_status is None:
            return
        now = utcnow()
        update: Dict[str, Any] = {
            "$set": {
                "worker_id": worker_id,
                "worker_type": self.worker_type,
                "status": status,
                "last_seen_at": now,
                "updated_at": now,
                **(fields or {}),
            },
            "$setOnInsert": {"created_at": now},
        }
        if increments:
            update["$inc"] = increments
        try:
            self.worker_status.update_one(
                {"worker_type": self.worker_type, "worker_id": worker_id},
                update,
                upsert=True,
            )
        except PyMongoError:
            # This collection only feeds operator dashboards, so a blip on the way
            # to Mongo must not reach the caller: it used to end the worker loop.
            log.warning(
                "worker status write failed for %s/%s", self.worker_type, worker_id, exc_info=True
            )


class HeartbeatTicker:
    """Refresh heartbeat_at on a background thread so long stages stay live.

    Stage-boundary heartbeats leave a gap during a single long stage, and some
    stages run for tens of minutes without returning. This ticker keeps
    heartbeat_at fresh regardless of stage length, which is what lets the
    stale-requeue threshold stay small enough to be useful.
    """

    def __init__(self, queue: JobQueue, job_id: str, interval_seconds: float) -> None:
        self._queue = queue
        self._job_id = job_id
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _beat_once(self) -> None:
        now = utcnow()
        self._queue.jobs.update_one(
            {"job_id": self._job_id, "status": "running"},
            {"$set": {"heartbeat_at": now, "updated_at": now}},
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._beat_once()
            except Exception as exc:  # pragma: no cover - best effort background beat
                print(f"heartbeat ticker error for job {self._job_id}: {exc}", flush=True)

    def start(self) -> "HeartbeatTicker":
        if self._interval > 0 and self._thread is None:
            name = f"{self._queue.worker_type}-heartbeat"
            self._thread = threading.Thread(target=self._run, name=name, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class WorkerLockHeld(RuntimeError):
    """Another worker of the same type already holds the lock.

    A distinct type so entry points can exit quietly on a duplicate start
    without also swallowing genuine RuntimeErrors from the work itself.
    Subclasses RuntimeError to keep existing handlers working.
    """


@contextmanager
def worker_lock(lock_path: str, label: str = "worker"):
    """Refuse to start a second worker of this type on the same host.

    Separate containers have separate ``/tmp``, so this guards against a stray
    manual run overlapping the scheduled loop, not against replicas.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise WorkerLockHeld(f"another {label} worker holds lock {path}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


MAX_FAILURE_BACKOFF_SECONDS = 60.0


def run_worker_forever(
    step: Callable[[], None],
    *,
    interval_seconds: float,
    max_consecutive_failures: int,
    logger: logging.Logger,
    max_backoff_seconds: float = MAX_FAILURE_BACKOFF_SECONDS,
) -> int:
    """Poll until the process is told to stop, surviving a transient failure.

    One failed iteration is usually a blip on the way to Mongo, and exiting over
    it hands the queue back for no reason -- worse, every replica sees the same
    blip, so they all leave at once. A sustained outage is a different thing and
    still has to be visible, so consecutive failures stretch the sleep and
    eventually return non-zero for the supervisor to act on.
    """
    consecutive_failures = 0
    while True:
        try:
            step()
        except Exception:
            consecutive_failures += 1
            logger.exception("iteration failed (%d consecutive)", consecutive_failures)
            if consecutive_failures >= max_consecutive_failures:
                logger.error("giving up after %d consecutive failures", consecutive_failures)
                return 1
            backoff = min(
                max(1.0, interval_seconds) * 2 ** (consecutive_failures - 1),
                max_backoff_seconds,
            )
            time.sleep(backoff)
            continue
        consecutive_failures = 0
        time.sleep(max(1.0, interval_seconds))


_ACTIVE_JOB: Dict[str, Any] = {"queue": None, "job_id": None}


def set_active_job(queue: Optional[JobQueue], job_id: Optional[str]) -> None:
    """Remember the in-flight job so a signal handler can hand it back."""
    _ACTIVE_JOB["queue"] = queue if job_id else None
    _ACTIVE_JOB["job_id"] = job_id


def active_job_id() -> Optional[str]:
    """The job this process is running, if any. Cleared when the job finishes."""
    return _ACTIVE_JOB.get("job_id")


def handle_shutdown_signal(signum, _frame) -> None:
    """Release the in-flight job so a redeploy does not orphan it as running."""
    queue = _ACTIVE_JOB.get("queue")
    job_id = _ACTIVE_JOB.get("job_id")
    if queue is not None and job_id:
        try:
            queue.release(job_id, f"requeued after worker received signal {signum} (likely redeploy)")
            print(f"released running job {job_id} back to pending on signal {signum}", flush=True)
        except Exception as exc:  # pragma: no cover - best effort on shutdown
            print(f"failed to release job {job_id} on signal {signum}: {exc}", flush=True)
    raise SystemExit(128 + signum)


def install_shutdown_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handle_shutdown_signal)
        except (ValueError, OSError):  # pragma: no cover - non-main thread / unsupported
            pass
