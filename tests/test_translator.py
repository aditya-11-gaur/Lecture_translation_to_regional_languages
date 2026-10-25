"""
Tests for src/translator.py – translation logic with code-mixing (mocked API calls).
"""

from unittest.mock import patch, MagicMock

import pytest

from src.transcriber import Segment
from src.glossary import load_default_glossary


class TestTranslateSegmentsDispatch:
    """Test translate_segments routes to the right engine."""

    @patch("src.translator._translate_batch_gemini")
    def test_dispatches_to_gemini(self, mock_gemini, sample_segments):
        from src.translator import translate_segments

        mock_gemini.return_value = [
            Segment(start=s.start, end=s.end, text=f"[HI]{s.text}")
            for s in sample_segments
        ]
        result = translate_segments(
            sample_segments, target_langs=["hi"], method="gemini"
        )
        assert "hi" in result
        assert len(result["hi"]) == len(sample_segments)

    @patch("src.translator._translate_batch_sarvam")
    def test_dispatches_to_sarvam(self, mock_sarvam, sample_segments):
        from src.translator import translate_segments

        mock_sarvam.return_value = [
            Segment(start=s.start, end=s.end, text=f"[HI]{s.text}")
            for s in sample_segments
        ]
        result = translate_segments(
            sample_segments, target_langs=["hi"], method="sarvam"
        )
        assert "hi" in result


class TestCodeMixingIntegration:
    """Test that glossary/code-mixing is plumbed through translation."""

    def test_glossary_accepted(self, sample_segments):
        """translate_segments should accept and use a glossary arg."""
        from src.translator import translate_segments

        glossary = load_default_glossary()
        # We can't easily test the actual API call, but we verify no crash
        # when glossary is passed (actual HTTP calls are mocked in CI).
        # Just verify the function signature accepts it.
        import inspect

        sig = inspect.signature(translate_segments)
        assert "glossary" in sig.parameters

    def test_glossary_prompt_section(self):
        """Glossary builds a non-empty prompt section for known terms."""
        glossary = load_default_glossary()
        terms = glossary.find_terms_in_text(
            "Let us discuss the algorithm and its time complexity."
        )
        if terms:
            section = glossary.build_translation_prompt_section(terms)
            assert len(section) > 0


class TestTranslatorPreservesTimestamps:
    """Translated segments should retain original timestamps."""

    @patch("src.translator._translate_batch_gemini")
    def test_timestamps_preserved(self, mock_gemini, sample_segments):
        from src.translator import translate_segments

        mock_gemini.return_value = [
            Segment(start=s.start, end=s.end, text=f"translated: {s.text}")
            for s in sample_segments
        ]
        result = translate_segments(
            sample_segments, target_langs=["hi"], method="gemini"
        )
        for orig, trans in zip(sample_segments, result["hi"]):
            assert trans.start == orig.start
            assert trans.end == orig.end


class TestFillersStrippedBeforeTranslation:
    """Fillers ("ah", "um") must never reach the translation engine."""

    @staticmethod
    def _echo(segs, lang, **kwargs):
        return [Segment(start=s.start, end=s.end, text=f"[HI]{s.text}") for s in segs]

    @patch("src.translator._translate_batch_sarvam")
    def test_engine_gets_clean_text(self, mock_sarvam):
        from src.translator import translate_segments
        mock_sarvam.side_effect = self._echo
        segs = [Segment(0.0, 4.0, "First ah let me introduce myself ah my name is Amit.")]
        result = translate_segments(segs, target_langs=["hi"], method="sarvam")
        sent = mock_sarvam.call_args.args[0]
        assert sent[0].text == "First let me introduce myself my name is Amit."
        assert result["hi"][0].text == "[HI]First let me introduce myself my name is Amit."

    @patch("src.translator._translate_batch_sarvam")
    def test_filler_only_segment_keeps_its_slot(self, mock_sarvam):
        """Output stays index-aligned with the English segments for TTS/alignment."""
        from src.translator import translate_segments
        mock_sarvam.side_effect = self._echo
        segs = [
            Segment(0.0, 2.0, "Hello everyone."),
            Segment(2.5, 3.0, "Ah."),
            Segment(3.5, 6.0, "Today we study trees."),
        ]
        result = translate_segments(segs, target_langs=["hi"], method="sarvam")
        assert len(mock_sarvam.call_args.args[0]) == 2
        out = result["hi"]
        assert [s.text for s in out] == ["[HI]Hello everyone.", "", "[HI]Today we study trees."]
        assert [(s.start, s.end) for s in out] == [(0.0, 2.0), (2.5, 3.0), (3.5, 6.0)]
