"""
Timing tests that run real ffmpeg on synthetic audio (tones + silence).

These guard the bugs where (a) a segment bleeding into the following gap
shifted every later segment, (b) a VAD-trimmed segment pulled every later
segment earlier, and (c) "-v error" hid silencedetect/volumedetect output so
the VAD trim, silence splitter and music gate never ran.
"""

import os
import shutil
import subprocess

import pytest

from src.transcriber import Segment, split_segments_at_silences, _snap_to_word_boundary
from src import audio_aligner
from src.audio_aligner import align_dubbed_audio, _voiced_end_in_window
from src.audio_separator import mix_audio_tracks

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed",
)

# Onsets are measured with silencedetect on an mp3; crossfades and encoder
# delay shift them by a few tens of ms.
ONSET_TOLERANCE_S = 0.1


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True)


def _tone(path: str, dur: float, freq: int = 440) -> str:
    _ffmpeg("-f", "lavfi", "-i", f"sine=f={freq}:d={dur}", "-ar", "48000", "-ac", "1", path)
    return path


def _silence(path: str, dur: float) -> str:
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", str(dur), path)
    return path


def _concat(out: str, *paths: str) -> str:
    inputs = [arg for p in paths for arg in ("-i", p)]
    labels = "".join(f"[{i}:a]" for i in range(len(paths)))
    _ffmpeg(*inputs, "-filter_complex", f"{labels}concat=n={len(paths)}:v=0:a=1", out)
    return out


def _silencedetect(path: str, key: str) -> list[float]:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "silencedetect=noise=-40dB:d=0.3", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return [float(line.split(f"{key}:")[1].split("|")[0])
            for line in r.stderr.splitlines() if key in line]


def _onsets(path: str) -> list[float]:
    """Times where sound resumes after a silence."""
    return _silencedetect(path, "silence_end")


def _offsets(path: str) -> list[float]:
    """Times where sound stops."""
    return _silencedetect(path, "silence_start")


@needs_ffmpeg
class TestAlignmentDoesNotDrift:

    def test_bleed_does_not_delay_next_segment(self, tmp_path):
        # 4 s of TTS for a 2 s window exceeds MAX_TEMPO, so segment 0 bleeds
        # into the gap.  Segment 1 must still start at exactly 5.0 s.
        a0 = _tone(str(tmp_path / "a0.wav"), 4.0)
        a1 = _tone(str(tmp_path / "a1.wav"), 2.0, 880)
        out = str(tmp_path / "out.mp3")
        align_dubbed_audio(
            [Segment(0.0, 2.0, "x"), Segment(5.0, 7.0, "y")], {0: a0, 1: a1}, 10.0, out,
        )
        onset = min(t for t in _onsets(out) if t > 1.0)
        assert onset == pytest.approx(5.0, abs=ONSET_TOLERANCE_S)

    def test_vad_trim_does_not_pull_next_segment_earlier(self, tmp_path):
        # Speaker talks 0–3 s then pauses until the 10 s window ends; 2 s of
        # TTS triggers the VAD trim.  Segment 1 must still start at 12.0 s.
        src = _concat(
            str(tmp_path / "src.wav"),
            _tone(str(tmp_path / "speech.wav"), 3.0),
            _silence(str(tmp_path / "pause.wav"), 13.0),
        )
        b0 = _tone(str(tmp_path / "b0.wav"), 2.0)
        b1 = _tone(str(tmp_path / "b1.wav"), 2.0, 880)
        out = str(tmp_path / "out.mp3")
        align_dubbed_audio(
            [Segment(0.0, 10.0, "x"), Segment(12.0, 14.0, "y")], {0: b0, 1: b1}, 16.0, out,
            source_audio_path=src,
        )
        # The trim really happened: segment 0 fills the voiced 3.3 s, not 2 s + padding
        assert _offsets(out)[0] == pytest.approx(3.3, abs=ONSET_TOLERANCE_S)
        onset = min(t for t in _onsets(out) if t > 1.0)
        assert onset == pytest.approx(12.0, abs=ONSET_TOLERANCE_S)

    def test_output_has_exact_total_duration(self, tmp_path):
        a0 = _tone(str(tmp_path / "a0.wav"), 4.0)
        out = str(tmp_path / "out.mp3")
        align_dubbed_audio([Segment(0.0, 2.0, "x")], {0: a0}, 10.0, out)
        assert audio_aligner._get_duration(out) == pytest.approx(10.0, abs=0.06)


@needs_ffmpeg
class TestSilenceDetectionIsActive:

    @pytest.fixture
    def speech_pause_speech(self, tmp_path):
        """3 s tone, 4 s silence, 3 s tone, 6 s silence (16 s total)."""
        tone = _tone(str(tmp_path / "t.wav"), 3.0)
        return _concat(
            str(tmp_path / "src.wav"),
            tone, _silence(str(tmp_path / "s4.wav"), 4.0),
            tone, _silence(str(tmp_path / "s6.wav"), 6.0),
        )

    def test_voiced_end_finds_trailing_silence(self, speech_pause_speech, tmp_path):
        voiced_end = _voiced_end_in_window(speech_pause_speech, 0.0, 16.0, str(tmp_path))
        assert voiced_end == pytest.approx(10.3, abs=0.15)

    def test_splitter_splits_at_internal_pause(self, speech_pause_speech):
        text = "alpha beta gamma delta, epsilon zeta eta theta"
        out = split_segments_at_silences([Segment(0.0, 16.0, text)], speech_pause_speech)
        assert len(out) == 2
        # Split lands inside the 3–7 s pause, and no word is cut or lost
        assert 3.0 < out[0].end < 7.0
        assert out[0].end == out[1].start
        assert " ".join(s.text for s in out) == text

    def test_music_gate_skips_silent_accompaniment(self, tmp_path):
        speech = _tone(str(tmp_path / "speech.wav"), 2.0)
        accomp = _silence(str(tmp_path / "accomp.wav"), 2.0)
        result = mix_audio_tracks(speech, accomp, str(tmp_path / "mixed.mp3"))
        assert result == speech

    def test_music_gate_mixes_real_accompaniment(self, tmp_path):
        speech = _tone(str(tmp_path / "speech.wav"), 2.0)
        accomp = _tone(str(tmp_path / "accomp.wav"), 2.0, 220)
        out = str(tmp_path / "mixed.mp3")
        assert mix_audio_tracks(speech, accomp, out) == out
        assert os.path.isfile(out)


class TestSnapToWordBoundary:

    def test_prefers_nearby_punctuation(self):
        text = "first clause here, second clause here"
        assert text[:_snap_to_word_boundary(text, 20)] == "first clause here,"

    def test_falls_back_to_nearest_space(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        cut = _snap_to_word_boundary(text, 30)
        assert text[cut] == " "

    def test_never_cuts_inside_a_word(self):
        text = "supercalifragilistic expialidocious"
        assert _snap_to_word_boundary(text, 5) == text.index(" ")

    def test_unsplittable_text(self):
        assert _snap_to_word_boundary("word", 2) == 4
