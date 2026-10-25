"""
Tests for src/utils.py – retry decorators, logging setup, preflight checks.
"""

import logging
import os
from unittest.mock import patch, MagicMock

import pytest

from src.utils import (
    setup_logging,
    retry_on_failure,
    PreflightError,
    check_ffmpeg,
    check_ffprobe,
)


class TestSetupLogging:
    """Test the logging configuration helper."""

    def test_returns_logger(self):
        logger = setup_logging(level="DEBUG")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "nptel_pipeline"

    def test_sets_level(self):
        logger = setup_logging(level="WARNING")
        assert logger.level == logging.WARNING


class TestRetryOnFailure:
    """Test the synchronous retry decorator."""

    def test_succeeds_first_try(self):
        call_count = 0

        @retry_on_failure(max_retries=3, backoff_base=0)
        def always_ok():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert always_ok() == "ok"
        assert call_count == 1

    def test_retries_on_exception(self):
        call_count = 0

        @retry_on_failure(max_retries=3, backoff_base=0)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "finally"

        assert fail_twice() == "finally"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        @retry_on_failure(max_retries=2, backoff_base=0)
        def always_fail():
            raise RuntimeError("always")

        with pytest.raises(RuntimeError, match="always"):
            always_fail()

    def test_custom_exceptions(self):
        call_count = 0

        @retry_on_failure(max_retries=3, backoff_base=0, retryable_exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            raises_type_error()
        assert call_count == 1  # no retries for TypeError


class TestPreflightChecks:
    """Test preflight system checks (mocked)."""

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_check_ffmpeg_found(self, mock_which):
        assert check_ffmpeg() is True

    @patch("shutil.which", return_value=None)
    def test_check_ffmpeg_missing(self, mock_which):
        assert check_ffmpeg() is False

    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    def test_check_ffprobe_found(self, mock_which):
        assert check_ffprobe() is True

    @patch("shutil.which", return_value=None)
    def test_check_ffprobe_missing(self, mock_which):
        assert check_ffprobe() is False
