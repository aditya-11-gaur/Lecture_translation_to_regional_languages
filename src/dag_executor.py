"""
dag_executor.py
───────────────
DAG-based pipeline executor for the NPTEL dubbing pipeline.

Concepts
────────
Node
    A single unit of computation with:
    - a unique ID within a request
    - a callable (the actual work)
    - declared resource type: "gpu", "api", "cpu"
    - upstream dependency node IDs

PipelineDAG
    One DAG instance per request. Nodes share a result store so
    downstream nodes can read upstream outputs.

DAGExecutor (singleton)
    Schedules nodes across k concurrent request slots.
    Resource semaphores ensure:
      - GPU: only 1 node at a time across ALL requests
      - API: max 8 concurrent calls across all requests
      - CPU: unlimited (thread-pool bounded by os.cpu_count())
    k controls how many requests have "active" (non-waiting) nodes.
    Initially k=1; can be raised at runtime via set_concurrency(k).

Usage (internal, called by job_queue.py)
────────────────────────────────────────
    dag = build_pipeline_dag(request_id, kwargs)
    executor = get_executor()
    result = executor.run(dag)   # blocks until all nodes complete
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("nptel_pipeline.dag")

# ── Resource types ────────────────────────────────────────────────────────────
GPU = "gpu"   # Demucs, Whisper, OpenVoice, Wav2Lip — serial globally
API = "api"   # GCP TTS, Sarvam, Gemini — rate-limited concurrency
CPU = "cpu"   # FFmpeg, subtitles, alignment — free concurrency

# Node states
WAITING  = "waiting"   # dependencies not yet satisfied
READY    = "ready"     # all deps done, waiting for a worker
RUNNING  = "running"
DONE     = "done"
FAILED   = "failed"


# ── Node definition ───────────────────────────────────────────────────────────

@dataclass
class Node:
    """One unit of work in the pipeline DAG."""
    node_id:    str
    fn:         Callable            # fn(ctx: dict) → Any
    deps:       list[str]           # node_ids that must complete first
    resource:   str = CPU           # GPU | API | CPU
    label:      str = ""            # human-readable name for UI

    # Runtime state (set by executor)
    status:     str  = field(default=WAITING, init=False)
    result:     Any  = field(default=None,    init=False)
    error:      str  = field(default=None,    init=False)


# ── DAG container ─────────────────────────────────────────────────────────────

class PipelineDAG:
    """
    One DAG instance per user request.

    ctx (context dict) is shared across all nodes in this DAG.
    Upstream nodes write their output into ctx; downstream nodes read it.
    """
    def __init__(self, request_id: str):
        self.request_id  = request_id
        self.nodes:  dict[str, Node] = {}
        self.ctx:    dict[str, Any]  = {}   # shared result store
        self._lock   = threading.Lock()

    def add(self, node: Node) -> "PipelineDAG":
        self.nodes[node.node_id] = node
        return self

    def _ready_nodes(self) -> list[Node]:
        """Return nodes whose dependencies are all DONE."""
        ready = []
        for n in self.nodes.values():
            if n.status != WAITING:
                continue
            if all(self.nodes[d].status == DONE for d in n.deps):
                ready.append(n)
        return ready

    def is_complete(self) -> bool:
        return all(n.status in (DONE, FAILED) for n in self.nodes.values())

    def has_failure(self) -> bool:
        return any(n.status == FAILED for n in self.nodes.values())

    def progress(self) -> tuple[int, int]:
        """(done_count, total_count)"""
        done  = sum(1 for n in self.nodes.values() if n.status == DONE)
        total = len(self.nodes)
        return done, total


# ── Global executor (singleton) ───────────────────────────────────────────────

class DAGExecutor:
    """
    Singleton that schedules nodes from multiple DAGs concurrently.

    Resource semaphores
    ───────────────────
    gpu_sem   : Semaphore(1)   — only 1 GPU node runs at any moment
    api_sem   : Semaphore(8)   — max 8 simultaneous API calls
    request_sem: Semaphore(k)  — max k requests have active nodes

    The thread pool has enough threads for all concurrent work.
    """

    def __init__(self, k: int = 1):
        self._k           = k
        self._request_sem = threading.Semaphore(k)
        self._gpu_sem     = threading.Semaphore(1)
        self._api_sem     = threading.Semaphore(8)
        self._pool        = ThreadPoolExecutor(
            max_workers=max(4, k * 8),
            thread_name_prefix="dag-worker",
        )
        self._lock        = threading.Lock()
        logger.info("[DAGExecutor] initialised k=%d", k)

    # ── Public ────────────────────────────────────────────────────────────────

    def set_concurrency(self, k: int) -> None:
        """Adjust max concurrent requests. Takes effect for new requests."""
        self._k           = k
        self._request_sem = threading.Semaphore(k)
        # Resize the thread pool
        old_pool, self._pool = self._pool, ThreadPoolExecutor(
            max_workers=max(4, k * 8),
            thread_name_prefix="dag-worker",
        )
        old_pool.shutdown(wait=False)
        logger.info("[DAGExecutor] concurrency set to k=%d", k)

    def run(
        self,
        dag: PipelineDAG,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """
        Execute the DAG, blocking until all nodes are done (or one fails).

        progress_cb(done, total, message) is called whenever a node finishes.

        Returns dag.ctx (the shared result store).
        Raises RuntimeError if any node fails.
        """
        request_id = dag.request_id
        logger.info("[DAGExecutor] Request %s: acquiring slot (k=%d)", request_id, self._k)
        self._request_sem.acquire()
        logger.info("[DAGExecutor] Request %s: slot acquired", request_id)

        try:
            self._execute(dag, progress_cb)
        finally:
            self._request_sem.release()
            logger.info("[DAGExecutor] Request %s: slot released", request_id)

        if dag.has_failure():
            failed = [n for n in dag.nodes.values() if n.status == FAILED]
            msgs   = ", ".join(f"{n.node_id}: {n.error}" for n in failed)
            raise RuntimeError(f"Pipeline failed at node(s): {msgs}")

        return dag.ctx

    # ── Internal ──────────────────────────────────────────────────────────────

    def _execute(
        self,
        dag: PipelineDAG,
        progress_cb: Callable | None,
    ) -> None:
        """Main scheduling loop for one DAG request."""
        futures: dict[str, Future] = {}

        while not dag.is_complete():
            if dag.has_failure():
                # Cancel pending futures
                for f in futures.values():
                    f.cancel()
                break

            with dag._lock:
                ready = dag._ready_nodes()
                for node in ready:
                    if node.node_id not in futures:
                        node.status = READY
                        f = self._pool.submit(self._run_node, dag, node, progress_cb)
                        futures[node.node_id] = f

            time.sleep(0.1)   # polling interval

        # Wait for all submitted futures to settle
        for f in futures.values():
            try:
                f.result()
            except Exception:
                pass

    def _run_node(
        self,
        dag: PipelineDAG,
        node: Node,
        progress_cb: Callable | None,
    ) -> None:
        """Execute a single node, acquiring the appropriate resource semaphore."""
        sem = (
            self._gpu_sem if node.resource == GPU else
            self._api_sem if node.resource == API else
            None
        )

        if sem:
            logger.debug("[DAGExecutor] Node %s waiting for %s lock", node.node_id, node.resource)
            sem.acquire()

        try:
            node.status = RUNNING
            logger.info("[DAGExecutor] → %s [%s]", node.node_id, node.resource.upper())
            node.result = node.fn(dag.ctx)
            if node.result is not None:
                dag.ctx[node.node_id] = node.result
            node.status = DONE
            done, total = dag.progress()
            logger.info("[DAGExecutor] ✓ %s (%d/%d)", node.node_id, done, total)
            if progress_cb:
                label = node.label or node.node_id
                progress_cb(done, total, f"✓ {label}")
        except Exception as exc:
            node.status = FAILED
            node.error  = f"{type(exc).__name__}: {exc}"
            logger.error("[DAGExecutor] ✗ %s failed: %s", node.node_id, exc, exc_info=True)
        finally:
            if sem:
                sem.release()


# ── Singleton accessor ────────────────────────────────────────────────────────

_executor: DAGExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> DAGExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = DAGExecutor(k=1)
    return _executor


def set_concurrency(k: int) -> None:
    """Raise or lower the number of concurrently active pipeline requests."""
    get_executor().set_concurrency(k)
