"""
Tests for src/filler_detector.py – speech filler detection and handling.
"""

import pytest

from src.filler_detector import (
    FILLER_WORDS,
    DISCOURSE_MARKERS,
    FillerInfo,
    detect_fillers_in_text,
    get_text_without_fillers,
    get_text_with_target_fillers,
)


class TestFillerConstants:
    """Test that filler constants are properly defined."""

    def test_filler_words_not_empty(self):
        assert len(FILLER_WORDS) > 0

    def test_contains_common_fillers(self):
        for word in ("um", "uh", "ah", "hmm"):
            assert word in FILLER_WORDS

    def test_discourse_markers_not_empty(self):
        assert len(DISCOURSE_MARKERS) > 0

    def test_contains_common_markers(self):
        for word in ("so", "well", "like", "you know"):
            assert word in DISCOURSE_MARKERS


class TestFillerInfoDataclass:
    """Test FillerInfo creation and serialisation."""

    def test_create_filler_info(self):
        fi = FillerInfo(word="um", start=1.0, end=1.3, index_in_text=0)
        assert fi.word == "um"
        assert fi.start == 1.0
        assert fi.end == 1.3
        assert fi.index_in_text == 0

    def test_to_dict(self):
        fi = FillerInfo(word="uh", start=2.0, end=2.2, index_in_text=5)
        d = fi.to_dict()
        assert d["word"] == "uh"
        assert d["start"] == 2.0


class TestDetectFillersInText:
    """Test text-based filler detection."""

    def test_detect_single_filler(self):
        fillers = detect_fillers_in_text("Um, hello there.")
        found_words = [f["word"].strip(".,!?").lower() for f in fillers]
        assert "um" in found_words

    def test_detect_multiple_fillers(self):
        fillers = detect_fillers_in_text("Um, uh, hello, ah, yes.")
        found_words = {f["word"].strip(".,!?").lower() for f in fillers}
        assert "um" in found_words
        assert "uh" in found_words

    def test_no_fillers(self):
        fillers = detect_fillers_in_text("This is a clean sentence.")
        filler_words_only = [
            f for f in fillers if f["word"].lower() in FILLER_WORDS
        ]
        assert len(filler_words_only) == 0

    def test_empty_text(self):
        fillers = detect_fillers_in_text("")
        assert fillers == []


class TestGetTextWithoutFillers:
    """Test removing fillers from text."""

    def test_remove_single_filler(self):
        result = get_text_without_fillers("Um, hello there.")
        assert "um" not in result.lower()

    def test_remove_multiple(self):
        result = get_text_without_fillers("Um, uh, hello.")
        low = result.lower()
        # Fillers should be removed
        assert "um" not in low.split()

    def test_preserve_non_fillers(self):
        result = get_text_without_fillers("Hello world")
        assert "hello" in result.lower()
        assert "world" in result.lower()

    def test_empty_text(self):
        assert get_text_without_fillers("") == ""


class TestGetTextWithTargetFillers:
    """Test target-language filler insertion."""

    def test_hindi_fillers(self):
        text = "Um, hello there."
        result = get_text_with_target_fillers(text, "hi")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_telugu_fillers(self):
        text = "Uh, test sentence."
        result = get_text_with_target_fillers(text, "te")
        assert isinstance(result, str)

    def test_odia_fillers(self):
        text = "Ah, well, test."
        result = get_text_with_target_fillers(text, "od")
        assert isinstance(result, str)

    def test_unknown_lang_returns_text(self):
        text = "Um, hello."
        result = get_text_with_target_fillers(text, "xx")
        assert isinstance(result, str)


class TestFillerRemovalKeepsSentenceEnds:

    def test_sentence_end_moves_to_previous_word(self):
        assert get_text_without_fillers("This is the result ah. Next we") == "This is the result. Next we"

    def test_mid_sentence_filler_and_its_comma_removed(self):
        assert get_text_without_fillers("So, ah, the algorithm works") == "So, the algorithm works"

    def test_filler_only_text_becomes_empty(self):
        assert get_text_without_fillers("Ah.") == ""
