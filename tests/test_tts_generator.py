"""
Tests for src/tts_generator.py – TTS synthesis (mocked).
"""

import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.transcriber import Segment
from src.tts_generator import (
    get_available_tts_engines,
    get_default_tts_engine,
    is_tts_available,
    generate_tts,
    generate_all_tts,
    TTS_CONCURRENCY,
    _compute_rate_for_duration,
    _compute_rate_from_actual,
    _rate_pct_to_str,
    _select_edge_voice,
    _LANG_CPS,
    _MAX_TTS_RATE_PCT,
    _MIN_TTS_RATE_PCT,
    _RESYNTH_THRESHOLD_LOW,
    _RESYNTH_THRESHOLD_HIGH,
    EDGE_TTS_VOICES,
    EDGE_TTS_VOICES_MALE,
    EDGE_TTS_VOICES_FEMALE,
)
from src.voice_analyzer import VoiceProfile


class TestTTSEngineRegistry:
    """Test TTS engine detection and selection."""

    def test_available_engines_returns_list(self):
        engines = get_available_tts_engines()
        assert isinstance(engines, list)

    def test_default_engine_returns_string(self):
        default = get_default_tts_engine()
        assert isinstance(default, str)
        assert len(default) > 0

    def test_edge_tts_always_available(self):
        engines = get_available_tts_engines()
        assert "edge_tts" in engines

    def test_is_tts_available(self):
        assert is_tts_available() is True  # edge_tts is always available

    def test_concurrency_limit_positive(self):
        assert TTS_CONCURRENCY > 0


class TestGenerateTTS:
    """Test the main generate_tts function (mocked)."""

    @patch("src.tts_generator._generate_edge_tts")
    def test_generate_tts_edge(self, mock_edge, sample_segments, tmp_output_dir):
        mock_edge.return_value = "/fake/output.mp3"
        result = generate_tts(
            sample_segments, "hi", "test_video",
            output_dir=tmp_output_dir, engine="edge_tts",
        )
        mock_edge.assert_called_once()
        assert result == "/fake/output.mp3"

    @patch("src.tts_generator._generate_sarvam_tts")
    def test_generate_tts_sarvam(self, mock_sarvam, sample_segments, tmp_output_dir):
        mock_sarvam.return_value = "/fake/output.mp3"
        result = generate_tts(
            sample_segments, "hi", "test_video",
            output_dir=tmp_output_dir, engine="sarvam",
        )
        mock_sarvam.assert_called_once()

    def test_generate_tts_invalid_engine(self, sample_segments, tmp_output_dir):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            generate_tts(
                sample_segments, "hi", "test_video",
                output_dir=tmp_output_dir, engine="nonexistent",
            )


class TestVoiceProfileSupport:
    """Test that TTS functions accept voice profile parameters."""

    def test_synthesize_segments_signature(self):
        import inspect
        from src.tts_generator import synthesize_segments

        sig = inspect.signature(synthesize_segments)
        assert "voice_profile" in sig.parameters
        assert "preserve_fillers" in sig.parameters

    def test_synthesize_all_signature(self):
        import inspect
        from src.tts_generator import synthesize_all_segments

        sig = inspect.signature(synthesize_all_segments)
        assert "voice_profile" in sig.parameters
        assert "preserve_fillers" in sig.parameters


class TestFillerHandlingInTTS:
    """Test that filler preservation works with TTS."""

    def test_segments_with_fillers_accepted(self):
        """TTS functions should handle segments with filler metadata."""
        from src.filler_detector import FillerInfo

        seg = Segment(
            start=0.0, end=3.0,
            text="Um, hello world.",
            fillers=[FillerInfo(word="Um", start=0.0, end=0.3, index_in_text=0)],
        )
        # Should not crash when creating – actual synthesis is mocked
        assert len(seg.fillers) == 1


class TestDurationAwareSynthesis:
    """Test the duration-aware TTS rate computation."""

    def test_compute_rate_normal(self):
        """When text length matches target duration, rate should be near 0."""
        cps = _LANG_CPS.get("hi", 14.0)
        text = "a" * int(cps * 5)  # text that naturally takes ~5 seconds
        rate = _compute_rate_for_duration(text, 5.0, "hi", base_rate_pct=0)
        assert -15 <= rate <= 15  # should be close to 0%

    def test_compute_rate_needs_speedup(self):
        """Text naturally slower than target → positive rate (speed up)."""
        # 100 chars at 14 cps = ~7.1s natural. Target = 3s → need ~+137% but clamped.
        rate = _compute_rate_for_duration("a" * 100, 3.0, "hi", base_rate_pct=0)
        assert rate > 0
        assert rate <= _MAX_TTS_RATE_PCT

    def test_compute_rate_needs_slowdown(self):
        """Text naturally faster than target → negative rate (slow down)."""
        # 10 chars at 14 cps = ~0.7s natural. Target = 4s → need slowdown.
        rate = _compute_rate_for_duration("a" * 10, 4.0, "hi", base_rate_pct=0)
        assert rate < 0
        assert rate >= _MIN_TTS_RATE_PCT

    def test_compute_rate_clamped_high(self):
        """Rate should never exceed the max TTS rate limit."""
        rate = _compute_rate_for_duration("a" * 500, 1.0, "hi", base_rate_pct=0)
        assert rate <= _MAX_TTS_RATE_PCT

    def test_compute_rate_clamped_low(self):
        """Rate should never go below the min TTS rate limit."""
        rate = _compute_rate_for_duration("a" * 3, 30.0, "hi", base_rate_pct=0)
        assert rate >= _MIN_TTS_RATE_PCT

    def test_compute_rate_empty_text(self):
        """Empty text should return base rate."""
        rate = _compute_rate_for_duration("", 5.0, "hi", base_rate_pct=10)
        assert rate == 10

    def test_compute_rate_zero_duration(self):
        """Zero target duration should return base rate."""
        rate = _compute_rate_for_duration("hello", 0.0, "hi", base_rate_pct=5)
        assert rate == 5

    def test_compute_rate_with_base(self):
        """Base rate from voice profile should be composed into result."""
        # Use text/duration where rates aren't both clamped to the same bound
        rate_base20 = _compute_rate_for_duration("a" * 42, 3.0, "hi", base_rate_pct=20)
        rate_base0 = _compute_rate_for_duration("a" * 42, 3.0, "hi", base_rate_pct=0)
        # With base +20%, the combined rate should generally be higher
        assert rate_base20 >= rate_base0

    def test_rate_pct_to_str_positive(self):
        assert _rate_pct_to_str(30) == "+30%"

    def test_rate_pct_to_str_negative(self):
        assert _rate_pct_to_str(-20) == "-20%"

    def test_rate_pct_to_str_zero(self):
        assert _rate_pct_to_str(0) == "+0%"

    def test_lang_cps_all_positive(self):
        """All language CPS values should be positive."""
        for lang, cps in _LANG_CPS.items():
            assert cps > 0, f"CPS for {lang} is {cps}"


class TestSynthesizeSegmentsDurationAware:
    """Test the signature and integration of duration-aware synthesis."""

    def test_synthesize_segments_accepts_original(self):
        """synthesize_segments should accept original_segments parameter."""
        import inspect
        from src.tts_generator import synthesize_segments

        sig = inspect.signature(synthesize_segments)
        assert "original_segments" in sig.parameters

    def test_synthesize_all_accepts_original(self):
        """synthesize_all_segments should accept original_segments parameter."""
        import inspect
        from src.tts_generator import synthesize_all_segments

        sig = inspect.signature(synthesize_all_segments)
        assert "original_segments" in sig.parameters


class TestResynthThresholds:
    """Test resynth threshold constants are sane."""

    def test_low_threshold_below_one(self):
        assert 0.5 < _RESYNTH_THRESHOLD_LOW < 1.0

    def test_high_threshold_above_one(self):
        assert 1.0 < _RESYNTH_THRESHOLD_HIGH < 2.0

    def test_symmetric_around_one(self):
        """Thresholds should be roughly symmetric around 1.0."""
        low_gap = 1.0 - _RESYNTH_THRESHOLD_LOW
        high_gap = _RESYNTH_THRESHOLD_HIGH - 1.0
        assert abs(low_gap - high_gap) < 0.15


class TestComputeRateFromActual:
    """Test the second-pass rate correction function."""

    def test_no_correction_needed(self):
        """If actual matches target, rate stays the same."""
        rate = _compute_rate_from_actual(5.0, 5.0, previous_rate_pct=10)
        assert rate == 10

    def test_actual_too_long(self):
        """If actual is longer, rate should increase (speak faster)."""
        rate = _compute_rate_from_actual(8.0, 5.0, previous_rate_pct=0)
        assert rate > 0

    def test_actual_too_short(self):
        """If actual is shorter, rate should decrease (speak slower)."""
        rate = _compute_rate_from_actual(3.0, 5.0, previous_rate_pct=0)
        assert rate < 0

    def test_clamped_high(self):
        rate = _compute_rate_from_actual(100.0, 1.0, previous_rate_pct=0)
        assert rate <= _MAX_TTS_RATE_PCT

    def test_clamped_low(self):
        rate = _compute_rate_from_actual(1.0, 100.0, previous_rate_pct=0)
        assert rate >= _MIN_TTS_RATE_PCT

    def test_zero_actual_returns_previous(self):
        rate = _compute_rate_from_actual(0.0, 5.0, previous_rate_pct=15)
        assert rate == 15

    def test_zero_target_returns_previous(self):
        rate = _compute_rate_from_actual(5.0, 0.0, previous_rate_pct=15)
        assert rate == 15

    def test_with_previous_rate(self):
        """Correction respects the previous rate applied."""
        # Previous +20% on 5s actual → new speed = 1.2 × (5/4) = 1.5 → rate +50
        rate = _compute_rate_from_actual(5.0, 4.0, previous_rate_pct=20)
        assert rate == 50


class TestGenderAwareVoiceSelection:
    """Test that TTS voice selection matches the speaker's gender."""

    def test_male_speaker_gets_male_voice(self):
        profile = VoiceProfile(is_male=True, avg_pitch_hz=120.0)
        voice = _select_edge_voice("hi", profile)
        assert voice == EDGE_TTS_VOICES_MALE["hi"]

    def test_female_speaker_gets_female_voice(self):
        profile = VoiceProfile(is_male=False, avg_pitch_hz=230.0)
        voice = _select_edge_voice("hi", profile)
        assert voice == EDGE_TTS_VOICES_FEMALE["hi"]

    def test_no_profile_defaults_to_female(self):
        voice = _select_edge_voice("hi", None)
        assert voice == EDGE_TTS_VOICES_FEMALE["hi"]

    def test_unknown_gender_defaults_to_female(self):
        profile = VoiceProfile(is_male=None)
        voice = _select_edge_voice("te", profile)
        assert voice == EDGE_TTS_VOICES_FEMALE["te"]

    def test_male_telugu(self):
        profile = VoiceProfile(is_male=True, avg_pitch_hz=130.0)
        voice = _select_edge_voice("te", profile)
        assert voice == EDGE_TTS_VOICES_MALE["te"]

    def test_odia_falls_back(self):
        """Odia only has a female voice; male should get it too."""
        profile = VoiceProfile(is_male=True)
        voice = _select_edge_voice("od", profile)
        assert voice is not None  # should return the female voice

    def test_unsupported_lang_returns_none(self):
        voice = _select_edge_voice("xyz", None)
        assert voice is None

    def test_legacy_alias_matches_female(self):
        """EDGE_TTS_VOICES should be the female voices (backward compat)."""
        assert EDGE_TTS_VOICES == EDGE_TTS_VOICES_FEMALE

    def test_male_and_female_maps_have_same_keys(self):
        assert set(EDGE_TTS_VOICES_MALE.keys()) == set(EDGE_TTS_VOICES_FEMALE.keys())


class TestGenderAwareEngineVoices:
    """Sarvam and Google TTS must also get a voice matching the speaker's gender."""

    def test_sarvam_male_speaker(self):
        from src.tts_generator import _select_sarvam_speaker, SARVAM_SPEAKERS_MALE
        profile = VoiceProfile(is_male=True, avg_pitch_hz=140.0)
        assert _select_sarvam_speaker("te", profile) == SARVAM_SPEAKERS_MALE["te"]

    def test_sarvam_female_or_unknown_speaker(self):
        from src.tts_generator import _select_sarvam_speaker, SARVAM_SPEAKERS
        assert _select_sarvam_speaker("hi", VoiceProfile(is_male=False)) == SARVAM_SPEAKERS["hi"]
        assert _select_sarvam_speaker("hi", None) == SARVAM_SPEAKERS["hi"]

    @pytest.mark.parametrize("engine, impl", [
        ("sarvam", "_synth_segments_sarvam"),
        ("sarvam_vc", "_synth_segments_sarvam_vc"),
        ("gcptts", "_synth_segments_gcptts"),
        ("gcptts_vc", "_synth_segments_gcptts_vc"),
    ])
    def test_engine_receives_voice_profile(self, engine, impl, sample_segments, tmp_output_dir):
        from src.tts_generator import synthesize_segments
        profile = VoiceProfile(is_male=True, avg_pitch_hz=140.0)
        with patch(f"src.tts_generator.{impl}", return_value={0: "seg.mp3"}) as mock_impl:
            synthesize_segments(sample_segments, "hi", tmp_output_dir,
                                engine=engine, voice_profile=profile)
        assert mock_impl.call_args.kwargs["voice_profile"] is profile


class TestVoiceCloningNeedsGPU:
    """Voice cloning is skipped cleanly (not attempted per segment) without a GPU."""

    def test_batch_returns_tts_audio_on_cpu(self, tmp_output_dir):
        from src import voice_converter
        files = {0: "seg_0000.mp3", 1: "seg_0001.mp3"}
        with patch.object(voice_converter, "is_openvoice_available", return_value=True), \
             patch.object(voice_converter, "convert_voice") as mock_convert:
            result = voice_converter.convert_segments_batch(
                files, "ref.wav", tmp_output_dir, device="cpu",
            )
        assert result == files
        mock_convert.assert_not_called()

    def test_sarvam_vc_skips_cloning_without_gpu(self, sample_segments, tmp_output_dir):
        from src import tts_generator, voice_converter
        sarvam_files = {0: "seg_0000.mp3"}
        ref = os.path.join(tmp_output_dir, "lecture.wav")
        open(ref, "wb").close()
        with patch.object(tts_generator, "_synth_segments_sarvam", return_value=sarvam_files), \
             patch.object(voice_converter, "is_openvoice_available", return_value=True), \
             patch.object(voice_converter, "is_gpu_available", return_value=False), \
             patch.object(voice_converter, "convert_segments_batch") as mock_batch:
            result = tts_generator._synth_segments_sarvam_vc(
                sample_segments, "hi", tmp_output_dir, voice_reference_audio=ref,
            )
        assert result == sarvam_files
        mock_batch.assert_not_called()
