"""
dag_executor.py
───────────────
DAG-based executor for the NPTEL dubbing pipeline.

Concepts
────────
Node
    One stage of a job: a callable ``fn(ctx)``, the resource it needs
    (GPU / API / CPU) and the nodes it depends on.
      deps   — must finish DONE; if one fails or is skipped, this node is
               skipped too (so a failed language skips the rest of its chain)
      after  — must merely have finished, in any state (e.g. the final mux
               waits for every language but runs with whichever succeeded)
      critical — a failure aborts the whole job; otherwise only the
               dependents are skipped

PipelineDAG
    The nodes of one job plus a shared ``ctx`` dict through which nodes pass
    their outputs downstream.

DAGExecutor (singleton)
    Runs the nodes of every active job.  A node starts as soon as its inputs
    are ready AND a slot of its resource is free; slots are shared by all
    jobs, so one job's API-bound TTS overlaps another job's GPU work while
    the GPU itself only ever runs DAG_GPU_SLOTS stages at once.

Usage
─────
    dag = build_pipeline_dag(...)          # src/pipeline_dag.py
    ctx = get_executor().run(dag, progress_cb)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("nptel_pipeline.dag")

# ── Resource types ────────────────────────────────────────────────────────────
GPU = "gpu"   # Demucs, Whisper, OpenVoice, XTTS, DeepFilterNet, Wav2Lip
API = "api"   # Sarvam / Google Cloud calls
CPU = "cpu"   # ffmpeg, alignment, mixing, subtitles

# ── Node states ───────────────────────────────────────────────────────────────
WAITING = "waiting"   # not started
RUNNING = "running"
DONE    = "done"
FAILED  = "failed"
SKIPPED = "skipped"   # an upstream node failed, or the job was aborted

_FINISHED = (DONE, FAILED, SKIPPED)


@dataclass
class Node:
    """One unit of work in the pipeline DAG."""
    node_id:  str
    fn:       Callable[[dict], Any]
    deps:     list[str] = field(default_factory=list)
    resource: str = CPU
    label:    str = ""
    after:    list[str] = field(default_factory=list)
    critical: bool = True

    # Runtime state (set by the executor)
    status:      str = field(default=WAITING, init=False)
    error:       str | None = field(default=None, init=False)
    started_at:  float | None = field(default=None, init=False)
    finished_at: float | None = field(default=None, init=False)

    @property
    def seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class PipelineDAG:
    """The nodes of one job and the context they share."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.nodes: dict[str, Node] = {}
        self.ctx: dict[str, Any] = {}

    def add(self, node: Node) -> Node:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.node_id}")
        if node.resource not in (GPU, API, CPU):
            raise ValueError(f"Unknown resource {node.resource!r} for {node.node_id}")
        self.nodes[node.node_id] = node
        return node

    def validate(self) -> None:
        """Every dependency exists and there are no cycles."""
        for n in self.nodes.values():
            for d in (*n.deps, *n.after):
                if d not in self.nodes:
                    raise ValueError(f"{n.node_id} depends on unknown node {d}")
        state: dict[str, int] = {}          # 1 = visiting, 2 = done

        def visit(nid: str, path: list[str]) -> None:
            if state.get(nid) == 2:
                return
            if state.get(nid) == 1:
                raise ValueError("Cycle in DAG: " + " → ".join(path + [nid]))
            state[nid] = 1
            n = self.nodes[nid]
            for d in (*n.deps, *n.after):
                visit(d, path + [nid])
            state[nid] = 2

        for nid in self.nodes:
            visit(nid, [])

    def is_finished(self) -> bool:
        return all(n.status in _FINISHED for n in self.nodes.values())

    def progress(self) -> tuple[int, int]:
        """(finished_count, total_count)"""
        done = sum(1 for n in self.nodes.values() if n.status in _FINISHED)
        return done, len(self.nodes)

    def failed_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.status == FAILED]


class DAGExecutor:
    """Schedules the nodes of all active DAGs onto shared resource slots."""

    def __init__(self, gpu_slots: int = 1, cpu_slots: int = 4, api_slots: int = 4):
        self._slots = {GPU: gpu_slots, CPU: cpu_slots, API: api_slots}
        self._in_use = {GPU: 0, CPU: 0, API: 0}
        # One condition guards all node state and slot counts; it is notified
        # whenever a node finishes so every waiting job re-checks.
        self._cond = threading.Condition()
        # Never more threads than slots: a node is only submitted once its
        # slot is held, so no worker thread ever blocks waiting for one.
        self._pool = ThreadPoolExecutor(
            max_workers=sum(self._slots.values()), thread_name_prefix="dag-worker",
        )
        logger.info("[DAGExecutor] slots: gpu=%d cpu=%d api=%d",
                    gpu_slots, cpu_slots, api_slots)

    def usage(self) -> dict[str, tuple[int, int]]:
        """{resource: (in_use, slots)} snapshot."""
        with self._cond:
            return {r: (self._in_use[r], self._slots[r]) for r in self._slots}

    def run(
        self,
        dag: PipelineDAG,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """
        Execute *dag*, blocking until every node has finished.

        ``progress_cb(finished, total, message)`` is called when nodes start
        and finish.  Returns ``dag.ctx``.  Raises RuntimeError if a critical
        node failed (after letting already-running nodes finish).
        """
        dag.validate()
        logger.info("[DAG %s] %d stages", dag.request_id, len(dag.nodes))
        t0 = time.monotonic()

        with self._cond:
            while True:
                self._skip_blocked(dag)
                aborted = any(n.critical for n in dag.failed_nodes())
                running = [n for n in dag.nodes.values() if n.status == RUNNING]

                if aborted:
                    if not running:
                        for n in dag.nodes.values():
                            if n.status == WAITING:
                                n.status, n.error = SKIPPED, "job aborted"
                        break
                else:
                    started = self._dispatch(dag, progress_cb)
                    if dag.is_finished():
                        break
                    if not running and not started and not self._ready(dag):
                        raise RuntimeError(f"DAG {dag.request_id} stalled")
                self._cond.wait(timeout=1.0)

        logger.info("[DAG %s] finished in %.1fs", dag.request_id, time.monotonic() - t0)
        critical = [n for n in dag.failed_nodes() if n.critical]
        if critical:
            n = critical[0]
            raise RuntimeError(f"{n.label or n.node_id} failed: {n.error}")
        return dag.ctx

    # ── Internal (called with self._cond held) ────────────────────────────────

    def _skip_blocked(self, dag: PipelineDAG) -> None:
        """Skip waiting nodes whose hard dependency failed or was skipped."""
        changed = True
        while changed:
            changed = False
            for n in dag.nodes.values():
                if n.status != WAITING:
                    continue
                bad = next((d for d in n.deps
                            if dag.nodes[d].status in (FAILED, SKIPPED)), None)
                if bad:
                    n.status = SKIPPED
                    n.error = f"skipped because {bad} {dag.nodes[bad].status}"
                    logger.warning("[DAG %s] – %s (%s)", dag.request_id, n.node_id, n.error)
                    changed = True

    def _ready(self, dag: PipelineDAG) -> list[Node]:
        return [
            n for n in dag.nodes.values()
            if n.status == WAITING
            and all(dag.nodes[d].status == DONE for d in n.deps)
            and all(dag.nodes[d].status in _FINISHED for d in n.after)
        ]

    def _dispatch(self, dag: PipelineDAG, progress_cb) -> bool:
        started = False
        for n in self._ready(dag):
            if self._in_use[n.resource] >= self._slots[n.resource]:
                continue
            self._in_use[n.resource] += 1
            n.status = RUNNING
            n.started_at = time.monotonic()
            started = True
            logger.info("[DAG %s] → %s [%s]", dag.request_id, n.node_id, n.resource.upper())
            self._pool.submit(self._run_node, dag, n, progress_cb)
        if started and progress_cb:
            self._report(dag, progress_cb)
        return started

    @staticmethod
    def _report(dag: PipelineDAG, progress_cb) -> None:
        done, total = dag.progress()
        running = [n.label or n.node_id for n in dag.nodes.values() if n.status == RUNNING]
        msg = ("Running: " + ", ".join(running)) if running else "Finishing…"
        try:
            progress_cb(done, total, msg)
        except Exception as exc:
            logger.debug("progress callback failed: %s", exc)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run_node(self, dag: PipelineDAG, node: Node, progress_cb) -> None:
        error = None
        try:
            result = node.fn(dag.ctx)
            if result is not None:
                dag.ctx[node.node_id] = result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("[DAG %s] ✗ %s failed: %s", dag.request_id, node.node_id,
                         exc, exc_info=True)
        finally:
            if node.resource == GPU:
                _release_gpu_memory()
            with self._cond:
                node.finished_at = time.monotonic()
                node.status = FAILED if error else DONE
                node.error = error
                self._in_use[node.resource] -= 1
                if not error:
                    logger.info("[DAG %s] ✓ %s (%.1fs)", dag.request_id,
                                node.node_id, node.seconds)
                if progress_cb:
                    self._report(dag, progress_cb)
                self._cond.notify_all()


def _release_gpu_memory() -> None:
    """Return cached VRAM to the driver so the next GPU stage has it all."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── Singleton accessor ────────────────────────────────────────────────────────

_executor: DAGExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> DAGExecutor:
    """The process-wide executor, sized from config.settings."""
    global _executor
    with _executor_lock:
        if _executor is None:
            from config.settings import DAG_API_SLOTS, DAG_CPU_SLOTS, DAG_GPU_SLOTS
            _executor = DAGExecutor(
                gpu_slots=DAG_GPU_SLOTS, cpu_slots=DAG_CPU_SLOTS, api_slots=DAG_API_SLOTS,
            )
        return _executor
