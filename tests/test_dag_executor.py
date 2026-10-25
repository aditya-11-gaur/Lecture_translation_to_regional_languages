"""
Tests for src/dag_executor.py – dependency order, resource slots shared
across jobs, and failure handling.  Nodes sleep briefly and record when they
ran, so overlap (or its absence) is measured, not assumed.
"""

import threading
import time

import pytest

from src.dag_executor import (
    API, CPU, DONE, FAILED, GPU, SKIPPED, DAGExecutor, Node, PipelineDAG,
)


class Spans:
    """Records the (start, end) time of each node."""

    def __init__(self):
        self.spans: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def task(self, name: str, seconds: float = 0.05, fail: bool = False):
        def fn(ctx):
            t0 = time.monotonic()
            time.sleep(seconds)
            with self._lock:
                self.spans[name] = (t0, time.monotonic())
            if fail:
                raise ValueError(f"{name} broke")
            return f"{name} result"
        return fn

    def overlap(self, a: str, b: str) -> bool:
        (s1, e1), (s2, e2) = self.spans[a], self.spans[b]
        return s1 < e2 and s2 < e1

    def max_concurrent(self, names) -> int:
        events = sorted([(self.spans[n][0], 1) for n in names] +
                        [(self.spans[n][1], -1) for n in names])
        cur = best = 0
        for _, delta in events:
            cur += delta
            best = max(best, cur)
        return best


def node(dag, name, spans, resource=CPU, deps=(), after=(), seconds=0.05,
         fail=False, critical=True):
    dag.add(Node(name, spans.task(name, seconds, fail), list(deps), resource,
                 name, list(after), critical))


def run_concurrently(executor, dags):
    errors = []

    def target(d):
        try:
            executor.run(d)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(d,)) for d in dags]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return errors


class TestOrdering:

    def test_dependencies_run_first_and_results_reach_ctx(self):
        spans, dag = Spans(), PipelineDAG("job")
        node(dag, "a", spans)
        node(dag, "b", spans, deps=["a"])
        node(dag, "c", spans, deps=["b"])
        ctx = DAGExecutor(cpu_slots=4).run(dag)
        assert spans.spans["a"][1] <= spans.spans["b"][0]
        assert spans.spans["b"][1] <= spans.spans["c"][0]
        assert ctx["c"] == "c result"

    def test_independent_stages_run_in_parallel(self):
        spans, dag = Spans(), PipelineDAG("job")
        for name in ("hi", "te", "od"):
            node(dag, name, spans, resource=API, seconds=0.2)
        t0 = time.monotonic()
        DAGExecutor(api_slots=4).run(dag)
        assert time.monotonic() - t0 < 0.45
        assert spans.max_concurrent(["hi", "te", "od"]) == 3


class TestResourceSlots:

    def test_cpu_slots_cap_concurrency(self):
        spans, dag = Spans(), PipelineDAG("job")
        names = [f"n{i}" for i in range(5)]
        for name in names:
            node(dag, name, spans, resource=CPU, seconds=0.1)
        DAGExecutor(cpu_slots=2).run(dag)
        assert spans.max_concurrent(names) == 2

    def test_gpu_is_exclusive_across_jobs_but_other_resources_overlap(self):
        """Job B's API stage runs while job A holds the GPU; GPU stages never overlap."""
        spans = Spans()
        job_a, job_b = PipelineDAG("A"), PipelineDAG("B")
        node(job_a, "a_gpu1", spans, resource=GPU, seconds=0.2)
        node(job_a, "a_gpu2", spans, resource=GPU, seconds=0.2, deps=["a_gpu1"])
        node(job_b, "b_gpu", spans, resource=GPU, seconds=0.2)
        node(job_b, "b_api", spans, resource=API, seconds=0.2)

        t0 = time.monotonic()
        errors = run_concurrently(DAGExecutor(gpu_slots=1, api_slots=2), [job_a, job_b])
        assert not errors
        gpu = ["a_gpu1", "a_gpu2", "b_gpu"]
        assert spans.max_concurrent(gpu) == 1
        assert any(spans.overlap("b_api", g) for g in gpu)
        # 3 GPU stages back to back; the API stage adds no wall time
        assert time.monotonic() - t0 < 0.8


class TestFailures:

    def test_language_failure_skips_its_chain_only(self):
        spans, dag = Spans(), PipelineDAG("job")
        node(dag, "stt", spans)
        node(dag, "tts_hi", spans, deps=["stt"], fail=True, critical=False)
        node(dag, "align_hi", spans, deps=["tts_hi"], critical=False)
        node(dag, "tts_te", spans, deps=["stt"], critical=False)
        node(dag, "align_te", spans, deps=["tts_te"], critical=False)
        node(dag, "mux", spans, after=["align_hi", "align_te"])
        DAGExecutor(cpu_slots=4).run(dag)

        status = {n: dag.nodes[n].status for n in dag.nodes}
        assert status == {"stt": DONE, "tts_hi": FAILED, "align_hi": SKIPPED,
                          "tts_te": DONE, "align_te": DONE, "mux": DONE}
        assert "tts_hi" in dag.nodes["align_hi"].error

    def test_critical_failure_raises_after_running_stages_finish(self):
        spans, dag = Spans(), PipelineDAG("job")
        node(dag, "slow", spans, seconds=0.3)
        node(dag, "broken", spans, fail=True)
        node(dag, "after_slow", spans, deps=["slow"])
        with pytest.raises(RuntimeError, match="broken failed: ValueError: broken broke"):
            DAGExecutor(cpu_slots=4).run(dag)
        assert dag.nodes["slow"].status == DONE        # not abandoned mid-run
        assert dag.nodes["after_slow"].status == SKIPPED
        assert "after_slow" not in spans.spans

    def test_validate_rejects_cycles_and_unknown_deps(self):
        spans = Spans()
        cyclic = PipelineDAG("job")
        node(cyclic, "a", spans, deps=["b"])
        node(cyclic, "b", spans, deps=["a"])
        with pytest.raises(ValueError, match="Cycle"):
            cyclic.validate()

        dangling = PipelineDAG("job")
        node(dangling, "a", spans, deps=["missing"])
        with pytest.raises(ValueError, match="unknown node missing"):
            dangling.validate()


class TestProgress:

    def test_progress_reaches_total(self):
        spans, dag = Spans(), PipelineDAG("job")
        node(dag, "a", spans)
        node(dag, "b", spans, deps=["a"])
        calls = []
        DAGExecutor().run(dag, lambda done, total, msg: calls.append((done, total, msg)))
        assert calls[-1][:2] == (2, 2)
        assert any(msg == "Running: a" for _, _, msg in calls)
