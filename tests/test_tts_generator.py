"""
Tests for src/tts_generator.py – TTS synthesis (mocked).
"""

from unittest.mock import patch

import pytest

from src.transcriber import Segment
from src.tts_generator import (
    get_available_tts_engines,
    get_default_tts_engine,
    split_tts_engine,
    synthesize_segments,
    TTS_ENGINE_REGISTRY,
)
from src.voice_analyzer import VoiceProfile


class TestTTSEngineRegistry:
    """Test TTS engine detection and selection."""

    def test_available_engines_returns_list(self):
        assert isinstance(get_available_tts_engines(), list)

    def test_default_engine_returns_string(self):
        default = get_default_tts_engine()
        assert isinstance(default, str) and default

    def test_only_supported_engines(self):
        """Edge TTS was removed; Sarvam, Google Neural2 and XTTS remain."""
        assert set(TTS_ENGINE_REGISTRY) == {"sarvam", "sarvam_vc", "gcptts", "gcptts_vc", "xtts"}


class TestSplitTTSEngine:
    """The _vc engines are synthesis + a separate voice cloning step."""

    @pytest.mark.parametrize("engine, expected", [
        ("sarvam_vc", ("sarvam", True)),
        ("gcptts_vc", ("gcptts", True)),
        ("sarvam", ("sarvam", False)),
        ("xtts", ("xtts", False)),
    ])
    def test_split(self, engine, expected):
        assert split_tts_engine(engine) == expected


class TestSynthesizeSegmentsSignature:

    def test_parameters(self):
        import inspect
        params = inspect.signature(synthesize_segments).parameters
        for name in ("voice_profile", "preserve_fillers", "original_segments", "voice_reference_audio"):
            assert name in params

    def test_unknown_engine_raises(self, sample_segments, tmp_output_dir):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            synthesize_segments(sample_segments, "hi", tmp_output_dir, engine="edge_tts")


class TestOnePace:
    """Every sentence is synthesised at the same natural pace."""

    def test_sarvam_uses_one_pace_and_lossless_audio(self, sample_segments, tmp_output_dir):
        import base64
        from unittest.mock import MagicMock
        from src import tts_generator
        client = MagicMock()
        client.text_to_speech.convert.return_value = MagicMock(audios=[base64.b64encode(b"RIFF").decode()])
        with patch.object(tts_generator, "SARVAM_API_KEY", "key"), \
             patch("sarvamai.SarvamAI", return_value=client):
            files = tts_generator._synth_segments_sarvam(sample_segments, "hi", tmp_output_dir)
        calls = client.text_to_speech.convert.call_args_list
        assert len(calls) == len(sample_segments)          # one call per sentence, no second pass
        assert {c.kwargs["pace"] for c in calls} == {tts_generator.SARVAM_PACE}
        assert all(c.kwargs["output_audio_codec"] == "wav" for c in calls)
        assert all(p.endswith(".wav") for p in files.values())


class TestGenderAwareEngineVoices:
    """Sarvam and Google TTS must get a voice matching the speaker's gender."""

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
        ("sarvam_vc", "_synth_segments_sarvam"),
        ("gcptts", "_synth_segments_gcptts"),
        ("gcptts_vc", "_synth_segments_gcptts"),
    ])
    def test_engine_receives_voice_profile(self, engine, impl, sample_segments, tmp_output_dir):
        profile = VoiceProfile(is_male=True, avg_pitch_hz=140.0)
        with patch(f"src.tts_generator.{impl}", return_value={0: "seg.mp3"}) as mock_impl:
            synthesize_segments(sample_segments, "hi", tmp_output_dir,
                                engine=engine, voice_profile=profile)
        assert mock_impl.call_args.kwargs["voice_profile"] is profile


class TestFallbacks:

    def test_gcp_without_key_returns_nothing(self, sample_segments, tmp_output_dir):
        from src import tts_generator
        with patch.object(tts_generator, "GCP_API_KEY", ""):
            assert tts_generator._synth_segments_gcptts(sample_segments, "hi", tmp_output_dir) == {}

    def test_xtts_unsupported_language_uses_sarvam(self, sample_segments, tmp_output_dir):
        from src import tts_generator
        with patch.object(tts_generator, "_synth_segments_sarvam", return_value={0: "s.mp3"}) as m:
            out = tts_generator._synth_segments_xtts(sample_segments, "te", tmp_output_dir)
        assert out == {0: "s.mp3"}
        m.assert_called_once()


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
