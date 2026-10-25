"""
Tests for src/audio_aligner.py – timing plan, tempo limits and helpers.
(The audio-level behaviour is tested with real ffmpeg in test_alignment_timing.py.)
"""

import os

import pytest

from src.audio_aligner import (
    MIN_TEMPO,
    MAX_TEMPO,
    HARD_MAX_TEMPO,
    MAX_DRIFT_S,
    MIN_PAUSE_S,
    CROSSFADE_MS,
    _build_atempo_chain,
    _codec_args,
    _plan_timing,
    _safe_remove,
    _snap_to_onset,
)


class TestATempoChain:
    """Test the atempo filter chain builder."""

    def test_normal_tempo(self):
        assert "atempo=" in _build_atempo_chain(1.5)

    def test_extreme_slow(self):
        chain = _build_atempo_chain(0.3)
        assert chain.count("atempo=") >= 2

    def test_unity_tempo(self):
        assert "1.0" in _build_atempo_chain(1.0)

    def test_zero_tempo(self):
        assert _build_atempo_chain(0) == "atempo=1.0"


class TestTempoLimits:
    """One natural speaking speed: the aligner may only nudge it."""

    def test_narrow_everyday_range(self):
        assert 0.9 <= MIN_TEMPO < 1.0 < MAX_TEMPO <= 1.2

    def test_hard_limit_above_normal_limit(self):
        assert MAX_TEMPO < HARD_MAX_TEMPO <= 1.4

    def test_crossfade_reasonable(self):
        assert 30 <= CROSSFADE_MS <= 100


class TestTimingPlan:
    """_plan_timing decides when each sentence plays and how fast."""

    def test_short_sentence_is_only_slightly_slowed(self):
        (start, tempo), = _plan_timing([0.0], [5.0], [1.0], 10.0)
        assert start == 0.0 and tempo == MIN_TEMPO

    def test_long_sentence_uses_the_pause_after_at_natural_speed(self):
        plan = _plan_timing([0.0, 5.0], [2.0, 7.0], [4.0, 2.0], 10.0)
        assert plan[0] == (0.0, 1.0)
        assert plan[1][0] == 5.0            # next sentence still on time

    def test_no_room_speeds_up_a_little_and_drifts_the_next(self):
        plan = _plan_timing([0.0, 3.0], [3.0, 6.0], [4.0, 2.0], 10.0)
        assert plan[0] == (0.0, MAX_TEMPO)
        late = plan[1][0] - 3.0
        assert 0 < late <= MAX_DRIFT_S

    def test_drift_stays_bounded_and_recovers_in_a_pause(self):
        starts = [0.0, 2.0, 4.0, 6.0, 20.0]
        ends = [2.0, 4.0, 6.0, 8.0, 22.0]
        plan = _plan_timing(starts, ends, [2.5, 2.5, 2.5, 2.5, 2.0], 30.0)
        for (start, tempo), pref in zip(plan, starts):
            assert start - pref <= MAX_DRIFT_S
            assert tempo <= MAX_TEMPO
        assert plan[-1][0] == 20.0          # back on time after the long pause

    def test_extreme_overrun_never_overlaps_or_exceeds_hard_limit(self):
        """Words are never cut: if even HARD_MAX_TEMPO can't fit, the dub drifts."""
        starts = [0.0, 2.0, 4.0, 6.0]
        durations = [3.0] * 4
        plan = _plan_timing(starts, [2.0, 4.0, 6.0, 8.0], durations, 30.0)
        ends = [s + d / t for (s, t), d in zip(plan, durations)]
        assert all(t <= HARD_MAX_TEMPO for _, t in plan)
        assert all(nxt[0] >= end + MIN_PAUSE_S - 1e-9 for nxt, end in zip(plan[1:], ends))

    def test_last_sentence_fits_before_the_end(self):
        (start, tempo), = _plan_timing([8.0], [9.0], [2.4], 10.0)
        assert start + 2.4 / tempo <= 10.0 + 1e-6

    def test_sentences_without_audio_are_skipped(self):
        plan = _plan_timing([0.0, 3.0, 6.0], [2.0, 5.0, 8.0], [1.9, None, 1.9], 10.0)
        assert plan[1] is None and plan[2][0] == 6.0


class TestOnsetSnap:

    def test_moves_start_to_end_of_silence(self):
        assert _snap_to_onset(2.0, [(1.2, 2.3)]) == 2.3

    def test_keeps_start_when_speech_already_there(self):
        assert _snap_to_onset(2.0, [(0.5, 1.5)]) == 2.0

    def test_never_moves_further_than_the_limit(self):
        assert _snap_to_onset(2.0, [(1.0, 4.0)]) == 2.0


class TestCodecArgs:

    @pytest.mark.parametrize("name, codec", [
        ("x.flac", "flac"), ("x.wav", "pcm_s16le"), ("x.mp3", "libmp3lame"),
    ])
    def test_codec_follows_extension(self, name, codec):
        assert codec in _codec_args(name)


class TestSafeRemove:
    """Test file cleanup utility."""

    def test_removes_existing_file(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "test.txt")
        with open(path, "w") as f:
            f.write("hello")
        _safe_remove(path)
        assert not os.path.isfile(path)

    def test_nonexistent_file_no_error(self):
        _safe_remove("/nonexistent/file.txt")  # should not raise


class TestAlignDubbedAudioSignature:

    def test_function_exists(self):
        import inspect
        from src.audio_aligner import align_dubbed_audio
        params = list(inspect.signature(align_dubbed_audio).parameters)
        assert params[:4] == ["original_segments", "segment_audio_files",
                              "total_duration", "output_path"]
        assert "source_audio_path" in params
