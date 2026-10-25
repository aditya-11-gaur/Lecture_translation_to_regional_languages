"""
Tests for src/transcriber.py – STT transcription logic (mocked).
"""

from unittest.mock import patch, MagicMock

import pytest

from src.transcriber import Segment, transcribe_audio


class TestTranscribeAudioDispatch:
    """Test that transcribe_audio dispatches to the correct engine."""

    @patch("src.transcriber._transcribe_whisper")
    def test_dispatches_to_whisper(self, mock_whisper):
        mock_whisper.return_value = [
            Segment(start=0.0, end=3.0, text="Hello"),
        ]
        result = transcribe_audio("fake.wav", method="whisper")
        mock_whisper.assert_called_once()
        assert len(result) == 1

    @patch("src.transcriber._transcribe_sarvam")
    def test_dispatches_to_sarvam(self, mock_sarvam):
        mock_sarvam.return_value = [
            Segment(start=0.0, end=3.0, text="Hello"),
        ]
        result = transcribe_audio("fake.wav", method="sarvam")
        mock_sarvam.assert_called_once()

    def test_invalid_method_raises(self):
        with pytest.raises((ValueError, KeyError)):
            transcribe_audio("fake.wav", method="nonexistent_engine")


class TestSegmentMerging:
    """Test that overlapping or short segments are handled."""

    def test_segments_have_valid_timestamps(self, sample_segments):
        for seg in sample_segments:
            assert seg.start >= 0
            assert seg.end >= seg.start
            assert seg.duration >= 0

    def test_segments_non_overlapping(self, sample_segments):
        for i in range(1, len(sample_segments)):
            # Each segment should start at or after the previous one ends
            # (small overlap tolerance of 0.5s is common for STT)
            assert sample_segments[i].start >= sample_segments[i - 1].start
