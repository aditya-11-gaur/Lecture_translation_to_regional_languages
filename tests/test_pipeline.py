"""
Tests for src/pipeline.py – orchestration, caching, and progress callbacks.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.transcriber import Segment


class TestPipelineCacheHelpers:
    """Test internal caching helpers."""

    @patch("src.pipeline.ENABLE_CACHE", True)
    def test_save_and_load_cache(self, tmp_output_dir):
        from src.pipeline import _save_cache, _load_cache

        _save_cache(tmp_output_dir, "test_key", {"hello": "world"}, "fp1")
        result = _load_cache(tmp_output_dir, "test_key", "fp1")
        assert result == {"hello": "world"}

    @patch("src.pipeline.ENABLE_CACHE", False)
    def test_cache_disabled(self, tmp_output_dir):
        from src.pipeline import _save_cache, _load_cache

        _save_cache(tmp_output_dir, "no_cache", {"data": True}, "fp1")
        result = _load_cache(tmp_output_dir, "no_cache", "fp1")
        assert result is None

    @patch("src.pipeline.ENABLE_CACHE", True)
    def test_load_missing_cache(self, tmp_output_dir):
        from src.pipeline import _load_cache

        result = _load_cache(tmp_output_dir, "nonexistent_key", "fp1")
        assert result is None

    @patch("src.pipeline.ENABLE_CACHE", True)
    def test_cache_from_other_inputs_is_ignored(self, tmp_output_dir):
        """A different video/engine under the same job name must not reuse results."""
        from src.pipeline import _save_cache, _load_cache

        _save_cache(tmp_output_dir, "english_segments", [{"text": "old lecture"}], "fp_old")
        assert _load_cache(tmp_output_dir, "english_segments", "fp_new") is None

    @patch("src.pipeline.ENABLE_CACHE", True)
    def test_untagged_legacy_cache_is_ignored(self, tmp_output_dir):
        """Caches written before fingerprinting can't be trusted."""
        from src.pipeline import _cache_dir, _load_cache

        with open(os.path.join(_cache_dir(tmp_output_dir), "english_segments.json"), "w") as f:
            json.dump([{"start": 0, "end": 1, "text": "old"}], f)
        assert _load_cache(tmp_output_dir, "english_segments", "fp1") is None

    def test_fingerprint_changes_with_any_input(self):
        from src.pipeline import _fingerprint

        base = _fingerprint("translated", [{"text": "hi"}], "sarvam", "code-mixed", True)
        assert base == _fingerprint("translated", [{"text": "hi"}], "sarvam", "code-mixed", True)
        assert base != _fingerprint("translated", [{"text": "hi"}], "gcp", "code-mixed", True)
        assert base != _fingerprint("translated", [{"text": "hi"}], "sarvam", "formal", True)
        assert base != _fingerprint("translated", [{"text": "bye"}], "sarvam", "code-mixed", True)


class TestSegmentsFromDicts:
    """Test cache round-tripping for segments."""

    def test_segments_round_trip(self, sample_segments):
        from src.pipeline import _segments_from_dicts

        dicts = [s.to_dict() for s in sample_segments]
        restored = _segments_from_dicts(dicts)
        assert len(restored) == len(sample_segments)
        for orig, rest in zip(sample_segments, restored):
            assert rest.start == orig.start
            assert rest.end == orig.end
            assert rest.text == orig.text


class TestProgressCallback:
    """Test that the pipeline calls the progress callback."""

    def test_noop_progress(self):
        from src.pipeline import _noop_progress

        # Should not raise
        _noop_progress(1, 7, "test message")

    def test_custom_progress_called(self):
        """Verify the pipeline invokes a custom progress callback."""
        calls = []

        def tracker(step, total, msg):
            calls.append((step, total, msg))

        # We can't easily run the full pipeline without mocking everything,
        # but we can verify the signature is correct
        from src.pipeline import run_pipeline
        import inspect

        sig = inspect.signature(run_pipeline)
        assert "progress" in sig.parameters


class TestPipelineSignature:
    """Test run_pipeline accepts all expected parameters."""

    def test_has_target_langs_param(self):
        from src.pipeline import run_pipeline
        import inspect

        sig = inspect.signature(run_pipeline)
        assert "target_langs" in sig.parameters

    def test_has_progress_param(self):
        from src.pipeline import run_pipeline
        import inspect

        sig = inspect.signature(run_pipeline)
        assert "progress" in sig.parameters

    def test_has_tts_engine_param(self):
        from src.pipeline import run_pipeline
        import inspect

        sig = inspect.signature(run_pipeline)
        assert "tts_engine" in sig.parameters


class TestPipelinePreflightIntegration:
    """Preflight checks run before any stage."""

    @patch("src.pipeline.run_preflight_checks", side_effect=RuntimeError("preflight failed"))
    def test_preflight_runs_first(self, mock_preflight, tmp_output_dir):
        from src.pipeline import run_pipeline

        with pytest.raises(RuntimeError, match="preflight failed"):
            run_pipeline("fake_video.mp4", output_dir=tmp_output_dir, do_tts=False)
        mock_preflight.assert_called_once()
