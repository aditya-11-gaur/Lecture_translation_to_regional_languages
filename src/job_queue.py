"""
job_queue.py
────────────
Global single-GPU job queue for multi-user Streamlit deployments.

Design
──────
• One pipeline job runs at a time (GPU is a shared, non-divisible resource).
• Incoming jobs are placed in a FIFO queue and assigned a UUID.
• A single background daemon thread dequeues and runs jobs sequentially.
• Each session polls its job's status via `get_job_status(job_id)`.
• Streamlit sessions submit via `submit_job()` and poll via `get_job_status()`.

Thread safety
─────────────
`_jobs` dict and `_queue` are accessed only through the module-level lock
`_lock`, except for `status` and `progress_pct` fields on individual Job
objects which are updated atomically by the worker thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("nptel_pipeline.job_queue")

# ── Job states ────────────────────────────────────────────────────────────────
PENDING  = "pending"    # in queue, not started
RUNNING  = "running"    # currently executing on GPU
DONE     = "done"       # finished successfully
FAILED   = "failed"     # raised an exception
CANCELED = "canceled"   # removed before starting (future use)


@dataclass
class Job:
    job_id:       str
    fn:           Callable       # callable that runs the pipeline
    kwargs:       dict           # kwargs forwarded to fn
    submitted_at: datetime       = field(default_factory=datetime.now)
    started_at:   datetime | None = None
    finished_at:  datetime | None = None
    status:       str            = PENDING
    result:       Any            = None
    error:        str | None     = None
    progress_pct: int            = 0
    progress_msg: str            = "Queued…"


# ── Global state ──────────────────────────────────────────────────────────────
_lock:  threading.Lock      = threading.Lock()
_queue: queue.Queue[str]    = queue.Queue()   # holds job_ids in FIFO order
_jobs:  dict[str, Job]      = {}              # job_id → Job
_worker_started: bool       = False


# ── Worker thread ─────────────────────────────────────────────────────────────

def _worker() -> None:
    """Daemon thread: dequeues and executes jobs one at a time."""
    logger.info("[JobQueue] Worker thread started")
    while True:
        job_id = _queue.get()          # blocks until a job is available
        with _lock:
            job = _jobs.get(job_id)
        if job is None:
            _queue.task_done()
            continue

        job.status     = RUNNING
        job.started_at = datetime.now()
        job.progress_pct = 1
        job.progress_msg = "Starting pipeline…"
        logger.info("[JobQueue] Starting job %s", job_id)

        try:
            # Inject a progress callback that writes into the Job object
            def _progress_cb(step: int, total: int, message: str) -> None:
                job.progress_pct = max(1, min(99, int(step / max(total, 1) * 100)))
                job.progress_msg = message

            job.result  = job.fn(progress=_progress_cb, **job.kwargs)
            job.status  = DONE
            job.progress_pct = 100
            job.progress_msg = "Complete ✅"
            logger.info("[JobQueue] Job %s finished successfully", job_id)

        except Exception as exc:
            job.status        = FAILED
            job.error         = f"{type(exc).__name__}: {exc}"
            job.progress_pct  = 0
            job.progress_msg  = f"Failed: {exc}"
            logger.error("[JobQueue] Job %s failed: %s\n%s",
                         job_id, exc, traceback.format_exc())

        finally:
            job.finished_at = datetime.now()
            _queue.task_done()


def _ensure_worker() -> None:
    """Start the background worker thread if it hasn't been started yet."""
    global _worker_started
    if not _worker_started:
        t = threading.Thread(target=_worker, daemon=True, name="pipeline-worker")
        t.start()
        _worker_started = True


# ── Public API ────────────────────────────────────────────────────────────────

def submit_job(fn: Callable, **kwargs) -> str:
    """
    Submit a job to the queue.

    Parameters
    ----------
    fn      : callable with signature fn(progress=..., **kwargs) → result
    **kwargs: forwarded to fn (everything except `progress`)

    Returns
    -------
    job_id : str  (UUID4 hex)
    """
    _ensure_worker()
    job_id = uuid.uuid4().hex
    job    = Job(job_id=job_id, fn=fn, kwargs=kwargs)
    with _lock:
        _jobs[job_id] = job
        _queue.put(job_id)
    logger.info("[JobQueue] Submitted job %s (queue depth: %d)",
                job_id, _queue.qsize())
    return job_id


def get_job_status(job_id: str) -> dict:
    """
    Return a snapshot of the job's current state.

    Returns a dict with keys:
        status        : "pending" | "running" | "done" | "failed"
        queue_position: int  (0 = next to run, -1 = not in queue / running/done)
        progress_pct  : int  0–100
        progress_msg  : str
        result        : Any  (only when status=="done")
        error         : str  (only when status=="failed")
        elapsed_s     : float  (seconds since submission)
    """
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"status": "unknown", "queue_position": -1,
                "progress_pct": 0, "progress_msg": "Job not found"}

    # Calculate queue position for pending jobs
    position = -1
    if job.status == PENDING:
        pending_ids = list(_queue.queue)   # snapshot (not thread-safe but best-effort)
        try:
            position = pending_ids.index(job_id)
        except ValueError:
            position = 0

    elapsed = (datetime.now() - job.submitted_at).total_seconds()

    return {
        "status":         job.status,
        "queue_position": position,
        "progress_pct":   job.progress_pct,
        "progress_msg":   job.progress_msg,
        "result":         job.result,
        "error":          job.error,
        "elapsed_s":      elapsed,
    }


def queue_depth() -> int:
    """Number of jobs currently waiting (not counting the running one)."""
    return _queue.qsize()


def active_job_count() -> int:
    """Total jobs in the system (pending + running)."""
    with _lock:
        return sum(1 for j in _jobs.values() if j.status in (PENDING, RUNNING))


def cleanup_old_jobs(max_age_hours: float = 24.0) -> int:
    """Remove finished/failed jobs older than max_age_hours. Returns count removed."""
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    removed = 0
    with _lock:
        to_delete = [
            jid for jid, j in _jobs.items()
            if j.status in (DONE, FAILED) and j.finished_at and j.finished_at < cutoff
        ]
        for jid in to_delete:
            del _jobs[jid]
            removed += 1
    return removed
