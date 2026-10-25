"""
Tests for src/job_queue.py – concurrent jobs and job-name reservation.
"""

import threading
import time

import pytest

from src import job_queue
from src.job_queue import DONE, RUNNING, JobNameInUse, get_job_status, submit_job


def wait_for(job_id, status, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if get_job_status(job_id)["status"] == status:
            return True
        time.sleep(0.02)
    return False


def blocking_job(release: threading.Event):
    def fn(progress, **kwargs):
        progress(1, 2, "working")
        release.wait(5)
        return {"ok": True, **kwargs}
    return fn


class TestJobQueue:

    def test_same_name_is_rejected_while_active_then_free_again(self):
        release = threading.Event()
        first = submit_job(blocking_job(release), name="lecture_x")
        with pytest.raises(JobNameInUse):
            submit_job(blocking_job(release), name="lecture_x")
        release.set()
        assert wait_for(first, DONE)
        again = submit_job(blocking_job(release), name="lecture_x")
        assert wait_for(again, DONE)

    def test_two_jobs_run_at_the_same_time(self):
        from config.settings import MAX_CONCURRENT_JOBS
        assert MAX_CONCURRENT_JOBS >= 2
        release = threading.Event()
        a = submit_job(blocking_job(release), name="job_a", video_path="a.mp4")
        b = submit_job(blocking_job(release), name="job_b", video_path="b.mp4")
        try:
            assert wait_for(a, RUNNING) and wait_for(b, RUNNING)
            assert job_queue.is_name_active("job_a")
        finally:
            release.set()
        assert wait_for(a, DONE) and wait_for(b, DONE)
        assert get_job_status(b)["result"] == {"ok": True, "video_path": "b.mp4"}
        assert not job_queue.is_name_active("job_a")
