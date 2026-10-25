"""
utils.py
────────
Shared utilities: retry logic, logging setup, preflight checks.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import subprocess
import time
from typing import Callable, Type

logger = logging.getLogger("nptel_pipeline")


# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(level: str | None = None) -> logging.Logger:
    """
    Configure structured logging for the pipeline.

    Args:
        level: Log level string (DEBUG / INFO / WARNING / ERROR).
               Defaults to LOG_LEVEL env-var or "INFO".

    Returns:
        The root pipeline logger.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        numeric_level = level
    else:
        numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s.%(module)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger("nptel_pipeline")
    root.setLevel(numeric_level)
    if not root.handlers:
        root.addHandler(handler)

    return root


# ── Retry decorator ──────────────────────────────────────────────────────────

def retry_on_failure(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable | None = None,
):
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries:          Maximum number of retry attempts.
        backoff_base:         Base for exponential backoff (seconds).
        retryable_exceptions: Tuple of exception types to retry on.
        on_retry:             Optional callback(attempt, exception) called before retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff_base ** attempt
                        logger.warning(
                            "Attempt %d/%d for %s failed: %s. Retrying in %.1fs…",
                            attempt + 1, max_retries, func.__name__, exc, wait,
                        )
                        if on_retry:
                            on_retry(attempt + 1, exc)
                        time.sleep(wait)
                    else:
                        logger.error(
                            "All %d attempts for %s exhausted. Last error: %s",
                            max_retries + 1, func.__name__, exc,
                        )
            raise last_exc
        return wrapper
    return decorator


# ── Async retry helper ───────────────────────────────────────────────────────

def retry_on_failure_async(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Async version of retry_on_failure."""
    import asyncio

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff_base ** attempt
                        logger.warning(
                            "Async attempt %d/%d for %s failed: %s. Retrying in %.1fs…",
                            attempt + 1, max_retries, func.__name__, exc, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            "All %d async attempts for %s exhausted.",
                            max_retries + 1, func.__name__,
                        )
            raise last_exc
        return wrapper
    return decorator


# ── File identity ────────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    """SHA-256 of a file's contents — identifies a video/audio across runs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Preflight checks ─────────────────────────────────────────────────────────

class PreflightError(RuntimeError):
    """Raised when a required external dependency is missing."""
    pass


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def check_ffprobe() -> bool:
    """Check if ffprobe is available on PATH."""
    return shutil.which("ffprobe") is not None


def run_preflight_checks() -> list[str]:
    """
    Run all preflight checks and return a list of warnings.
    Raises PreflightError if critical dependencies are missing.
    """
    issues: list[str] = []

    if not check_ffmpeg():
        raise PreflightError(
            "ffmpeg not found on PATH. Install it: https://ffmpeg.org/download.html"
        )

    if not check_ffprobe():
        raise PreflightError(
            "ffprobe not found on PATH. It usually comes with ffmpeg."
        )

    return issues
