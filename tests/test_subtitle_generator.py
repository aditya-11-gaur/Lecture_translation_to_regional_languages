"""
Tests for src/subtitle_generator.py – SRT/VTT generation and line wrapping.
"""

import os
import tempfile

import pytest

from src.transcriber import Segment
from src.subtitle_generator import (
    wrap_subtitle_text,
    generate_vtt,
)


class TestWrapSubtitleText:
    """Test subtitle line-wrapping logic."""

    def test_short_text_no_wrap(self):
        text = "Hello world"
        result = wrap_subtitle_text(text, max_chars=42)
        assert "\n" not in result
        assert result == "Hello world"

    def test_long_text_wrapped(self):
        text = "This is a somewhat longer sentence that should definitely be wrapped"
        result = wrap_subtitle_text(text, max_chars=42)
        lines = result.split("\n")
        assert len(lines) == 2
        for line in lines:
            assert len(line) <= 50  # some tolerance for word boundary splits

    def test_single_word(self):
        text = "Supercalifragilisticexpialidocious"
        result = wrap_subtitle_text(text, max_chars=20)
        # Even if word is longer than limit, should not crash
        assert isinstance(result, str)

    def test_empty_text(self):
        result = wrap_subtitle_text("", max_chars=42)
        assert result == ""

    def test_exact_limit(self):
        text = "A" * 42
        result = wrap_subtitle_text(text, max_chars=42)
        assert "\n" not in result


class TestGenerateSrt:
    """Test SRT subtitle generation."""

    def test_srt_basic(self, sample_segments, tmp_output_dir):
        from src.subtitle_generator import generate_srt

        result = generate_srt(sample_segments, "en", "test_video", tmp_output_dir)
        assert os.path.isfile(result)

        content = open(result, "r", encoding="utf-8-sig").read()
        assert "1\n" in content
        assert "-->" in content
        assert "Welcome" in content

    def test_srt_has_bom(self, sample_segments, tmp_output_dir):
        """SRT files should have UTF-8 BOM for code-mixed subtitle compatibility."""
        from src.subtitle_generator import generate_srt

        result = generate_srt(sample_segments, "en", "bom_test", tmp_output_dir)
        with open(result, "rb") as f:
            raw = f.read(3)
        assert raw == b"\xef\xbb\xbf", "SRT should start with UTF-8 BOM"

    def test_srt_skips_empty_segments(self, tmp_output_dir):
        segments = [
            Segment(start=0.0, end=3.0, text="Hello"),
            Segment(start=3.0, end=3.0, text="Zero duration"),
            Segment(start=4.0, end=7.0, text="World"),
        ]
        from src.subtitle_generator import generate_srt

        result = generate_srt(segments, "en", "skip_test", tmp_output_dir)
        content = open(result, "r", encoding="utf-8-sig").read()
        assert "1\n" in content
        assert "Hello" in content


class TestGenerateVtt:
    """Test WebVTT subtitle generation."""

    def test_vtt_header(self, sample_segments, tmp_output_dir):
        result = generate_vtt(sample_segments, "en", "test_video", tmp_output_dir)
        assert os.path.isfile(result)

        content = open(result, "r", encoding="utf-8").read()
        assert content.startswith("WEBVTT")
        assert "-->" in content

    def test_vtt_timestamp_format(self, sample_segments, tmp_output_dir):
        result = generate_vtt(sample_segments, "en", "fmt_test", tmp_output_dir)
        content = open(result, "r", encoding="utf-8").read()
        # VTT uses HH:MM:SS.mmm format (dot separator)
        assert "00:00:00.000" in content


class TestGenerateAllSubtitles:
    """Test the batch subtitle generator."""

    def test_generates_english_and_translated(
        self, sample_segments, translated_segments, tmp_output_dir
    ):
        from src.subtitle_generator import generate_all_subtitles

        paths = generate_all_subtitles(
            sample_segments,
            translated_segments,
            "test_video",
            tmp_output_dir,
        )
        assert "en" in paths
        assert os.path.isfile(paths["en"])
        for lang in translated_segments:
            assert lang in paths
            assert os.path.isfile(paths[lang])

    def test_generates_vtt_alongside_srt(
        self, sample_segments, translated_segments, tmp_output_dir
    ):
        """generate_all_subtitles should also produce VTT files."""
        from src.subtitle_generator import generate_all_subtitles

        generate_all_subtitles(
            sample_segments,
            translated_segments,
            "vtt_test",
            tmp_output_dir,
        )
        # VTT files should exist alongside SRT
        for lang in ["en"] + list(translated_segments.keys()):
            vtt_path = os.path.join(tmp_output_dir, f"vtt_test_{lang}.vtt")
            assert os.path.isfile(vtt_path), f"VTT file missing for {lang}"


class TestCodeMixedSubtitles:
    """Test subtitle handling for code-mixed text (Hindi + English)."""

    def test_code_mixed_srt(self, tmp_output_dir):
        """Code-mixed segments with mixed scripts should work correctly."""
        from src.subtitle_generator import generate_srt

        segments = [
            Segment(start=0.0, end=4.0,
                    text="यह algorithm बहुत efficient है।"),
            Segment(start=4.5, end=8.0,
                    text="Binary search tree में O(log n) time complexity होती है।"),
        ]
        result = generate_srt(segments, "hi", "codemix_test", tmp_output_dir)
        content = open(result, "r", encoding="utf-8-sig").read()
        assert "algorithm" in content
        assert "efficient" in content
        assert "Binary search tree" in content
        # O(log n) may be wrapped across lines; check parts exist
        assert "O(log" in content
        assert "n)" in content

    def test_code_mixed_wrap(self):
        """Line wrapping should handle mixed-script text."""
        text = "यह algorithm बहुत efficient है और binary search tree में काम करता है"
        result = wrap_subtitle_text(text, max_chars=35)
        # Should split into 2 lines
        lines = result.split("\n")
        assert len(lines) == 2


class TestEmptyCuesSkipped:
    """A translated segment can be empty (it was only a filler like "ah")."""

    def test_srt_and_vtt_skip_empty_text(self, tmp_output_dir):
        from src.subtitle_generator import generate_srt, generate_vtt

        segs = [Segment(0.0, 2.0, "Hello."), Segment(2.5, 3.0, ""), Segment(3.5, 5.0, "Bye.")]
        srt = open(generate_srt(segs, "hi", "empty", tmp_output_dir), encoding="utf-8-sig").read()
        vtt = open(generate_vtt(segs, "hi", "empty", tmp_output_dir), encoding="utf-8").read()
        assert srt.count("-->") == 2 and vtt.count("-->") == 2
        assert "\n2\n" in srt and "\n3\n" not in srt   # numbering stays consecutive
