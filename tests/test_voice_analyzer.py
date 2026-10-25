"""
Tests for src/voice_analyzer.py – speaker voice profile extraction.
"""

import math
import os
import shutil
import subprocess
import tempfile

import pytest

from src.voice_analyzer import VoiceProfile, apply_voice_profile_to_audio


class TestVoiceProfile:
    """Test VoiceProfile fields."""

    def test_to_dict(self):
        vp = VoiceProfile(
            avg_pitch_hz=150.0,
            pitch_range=(80.0, 250.0),
            avg_volume_dbfs=-18.0,
            is_male=True,
            duration_seconds=60.0,
        )
        d = vp.to_dict()
        assert d["avg_pitch_hz"] == 150.0
        assert d["is_male"] is True
        assert "speaking_rate_wps" not in d


class TestApplyVoiceProfile:
    """Test the post-TTS pitch-shifting function."""

    def test_function_exists(self):
        assert callable(apply_voice_profile_to_audio)

    def test_unknown_speaker_pitch_means_no_shift(self, tmp_output_dir):
        from src.voice_analyzer import pitch_shift_to_match
        assert pitch_shift_to_match(VoiceProfile(avg_pitch_hz=0.0), "unused.wav") == 0.0


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tone(path: str, freq: int, dur: float = 1.0, lead_silence: float = 0.0) -> str:
    """Sine 'voice' at *freq* Hz, optionally preceded by silence."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={lead_silence}",
         "-f", "lavfi", "-i", f"sine=f={freq}:d={dur}:sample_rate=16000",
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", "-ac", "1", path],
        capture_output=True, check=True,
    )
    return path


class TestMeasuredPitchShift:
    """The post-TTS pitch shift is measured from the dubbed audio itself."""

    def test_autocorrelation_finds_sine_pitch(self):
        import numpy as np
        from src.voice_analyzer import _autocorrelation_pitch
        frame = 10000 * np.sin(2 * np.pi * 200 * np.arange(480) / 16000)
        assert _autocorrelation_pitch(frame, 16000) == pytest.approx(200, rel=0.03)

    @needs_ffmpeg
    def test_shift_brings_tts_to_speaker_pitch(self, tmp_output_dir):
        from src.voice_analyzer import pitch_shift_to_match
        tts = _tone(os.path.join(tmp_output_dir, "tts.wav"), 110)
        speaker = VoiceProfile(avg_pitch_hz=125.0, is_male=True)
        assert pitch_shift_to_match(speaker, tts) == pytest.approx(12 * math.log2(125 / 110), abs=0.3)

    @needs_ffmpeg
    def test_male_speaker_lowers_a_higher_tts_voice(self, tmp_output_dir):
        """The old logic raised the pitch for every male lecturer, whatever the TTS voice."""
        from src.voice_analyzer import pitch_shift_to_match
        tts = _tone(os.path.join(tmp_output_dir, "tts.wav"), 210)
        assert pitch_shift_to_match(VoiceProfile(avg_pitch_hz=140.0, is_male=True), tts) < 0

    @needs_ffmpeg
    def test_shift_is_clamped(self, tmp_output_dir):
        from src.voice_analyzer import pitch_shift_to_match, MAX_PITCH_SHIFT_SEMITONES
        tts = _tone(os.path.join(tmp_output_dir, "tts.wav"), 400)
        assert pitch_shift_to_match(VoiceProfile(avg_pitch_hz=100.0), tts) == -MAX_PITCH_SHIFT_SEMITONES

    @needs_ffmpeg
    def test_leading_silence_is_skipped(self, tmp_output_dir):
        """A dubbed track that opens with a long silence still gets measured."""
        from src.voice_analyzer import _analyze_pitch
        tts = _tone(os.path.join(tmp_output_dir, "tts.wav"), 150, lead_silence=70.0)
        assert _analyze_pitch(tts, max_seconds=60)["avg_hz"] == 0.0
        assert _analyze_pitch(tts, max_seconds=60, skip_silence=True)["avg_hz"] == pytest.approx(150, rel=0.03)


    @needs_ffmpeg
    def test_pitch_shift_moves_pitch_but_keeps_duration(self, tmp_output_dir):
        from src.audio_aligner import _get_duration
        from src.voice_analyzer import _analyze_pitch
        tts = _tone(os.path.join(tmp_output_dir, "tts.wav"), 200, dur=3.0)
        out = os.path.join(tmp_output_dir, "shifted.flac")
        apply_voice_profile_to_audio(tts, VoiceProfile(avg_pitch_hz=224.0), out, semitones=2.0)
        assert _analyze_pitch(out)["avg_hz"] == pytest.approx(200 * 2 ** (2 / 12), rel=0.03)
        assert _get_duration(out) == pytest.approx(3.0, abs=0.05)
