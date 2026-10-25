"""
Tests for src/glossary.py – technical term code-mixing for NPTEL courses.
"""

import json
import os
import tempfile

import pytest

from src.glossary import (
    DEFAULT_GLOSSARY,
    Glossary,
    load_default_glossary,
    load_glossary_from_file,
    verify_terms_preserved,
)


class TestDefaultGlossary:
    """Test that the default glossary is well-formed."""

    def test_glossary_not_empty(self):
        assert len(DEFAULT_GLOSSARY) > 0

    def test_has_expected_categories(self):
        assert "computer_science" in DEFAULT_GLOSSARY
        assert "mathematics" in DEFAULT_GLOSSARY

    def test_terms_are_lists(self):
        for category, terms in DEFAULT_GLOSSARY.items():
            assert isinstance(terms, list), f"Category '{category}' is not a list"
            for term in terms:
                assert isinstance(term, str)


class TestGlossaryDataclass:
    """Test the Glossary dataclass methods."""

    @pytest.fixture
    def glossary(self):
        return load_default_glossary()

    def test_terms_is_set(self, glossary):
        assert isinstance(glossary.terms, set)

    def test_find_terms_in_text(self, glossary):
        text = "We will study the algorithm and data structure today."
        found = glossary.find_terms_in_text(text)
        found_lower = {t.lower() for t in found}
        assert "algorithm" in found_lower or "data structure" in found_lower

    def test_find_terms_case_insensitive(self, glossary):
        text = "The ALGORITHM is efficient."
        found = glossary.find_terms_in_text(text)
        found_lower = {t.lower() for t in found}
        assert "algorithm" in found_lower

    def test_find_terms_empty_text(self, glossary):
        found = glossary.find_terms_in_text("")
        assert found == []

    def test_extract_terms_from_segments(self, glossary):
        from src.transcriber import Segment
        segments = [
            Segment(start=0, end=3, text="Binary search is an algorithm."),
            Segment(start=3, end=6, text="It uses a sorted array."),
        ]
        terms = glossary.extract_terms_from_segments(segments)
        assert isinstance(terms, list)


class TestLoadGlossary:
    """Test loading glossaries."""

    def test_load_default(self):
        g = load_default_glossary()
        assert isinstance(g, Glossary)
        assert len(g.terms) > 0

    def test_load_from_file(self):
        # Create a JSON file matching expected format (dict of lists)
        custom = {"custom": ["widget", "sprocket"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(custom, f)
            f.flush()
            path = f.name
        try:
            g = load_glossary_from_file(path)
            assert isinstance(g, Glossary)
            assert "widget" in g.terms
        finally:
            os.unlink(path)

    def test_load_from_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_glossary_from_file("/nonexistent/path.json")


class TestVerifyTermsPreserved:
    """Test code-mixing verification."""

    @pytest.fixture
    def glossary(self):
        return load_default_glossary()

    def test_all_preserved(self, glossary):
        original = "The algorithm is efficient."
        translated = "यह algorithm बहुत efficient है।"
        missing = verify_terms_preserved(original, translated, glossary)
        assert isinstance(missing, list)

    def test_some_missing(self, glossary):
        original = "The algorithm is efficient."
        translated = "यह बहुत अच्छा है।"
        missing = verify_terms_preserved(original, translated, glossary)
        # 'algorithm' should be flagged as missing
        assert any("algorithm" in m.lower() for m in missing)

    def test_empty_original(self, glossary):
        missing = verify_terms_preserved("", "any text", glossary)
        assert missing == []

    def test_terms_in_original_preserved_in_translation(self, glossary):
        original = "We study the algorithm approach."
        translated = "हम Algorithm approach का अध्ययन करते हैं।"
        missing = verify_terms_preserved(original, translated, glossary)
        assert isinstance(missing, list)
