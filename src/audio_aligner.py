"""
audio_aligner.py
────────────────
Temporal alignment for dubbed audio.

Takes per-segment TTS audio files and stretches / compresses each one so that
it fills the time window of the corresponding original English segment.
The result is a full-length audio track (same duration as the source) where
each dubbed sentence plays at the same moment the English sentence was spoken.

Approach   (informed by research)
────────
  This module uses the **Rubber Band** library (via ffmpeg's ``rubberband``
  audio filter) for high-quality, pitch-preserving time-stretching:

  • **Phase-vocoder based** – much higher quality than ffmpeg's ``atempo``
    filter, especially for speech.  Preserves pitch, formants, and
    transient crispness.
  • **Formant preservation** – ``formant=preserved`` keeps vowel quality
    natural even at extreme stretch ratios.
  • **Naturalness bounds** – Research ("Dubbing in Practice", Brannon et al.
    2022) shows that isochronic constraints should not sacrifice vocal
    naturalness.  We therefore limit the tempo ratio to
    ``[MIN_TEMPO, MAX_TEMPO]`` (default 0.65×–1.6×).  When the TTS audio
    exceeds these bounds the segment is stretched to the limit and then:
      – if too short → silence is padded at the end;
      – if too long  → the audio is hard-truncated to the target window.
  • **Lossless intermediate format** – all intermediate files use 48 kHz
    PCM WAV to avoid cumulative re-encoding artefacts.
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
import tempfile

from src.transcriber import Segment
from src.utils import is_rubberband_available

logger = logging.getLogger("nptel_pipeline")


# ── Configuration ────────────────────────────────────────────────────────────

# Tempo limits to preserve natural-sounding speech.
# With duration-aware TTS (two-pass), the raw mismatch is typically small.
# Rubber Band fine-tunes.  Bounds are wider than pure-duration-aware would
# need, as a safety net for CPS estimation errors.
# tempo < 1 → slow down (TTS too fast);  tempo > 1 → speed up (TTS too slow).
MIN_TEMPO = 0.55   # never slow down more than ~45 % (~1.82× stretch)
MAX_TEMPO = 1.60   # never speed up more than ~60 %

# Ratio below which rubber-band stretching is skipped entirely.
# When TTS audio is less than NATURAL_SPEED_CUTOFF × target duration,
# stretching would produce incomprehensible slow speech. Instead:
# play TTS at natural speed and pad the remainder with silence.
# Lowered from 0.45 → 0.25 so we try to rubber-band before giving up.
NATURAL_SPEED_CUTOFF = 0.25

# Intermediate sample rate (Hz)
INTERMEDIATE_SR = 48_000

# Cross-fade duration at segment boundaries (ms)
CROSSFADE_MS = 50    # 50 ms – smooths transitions between speech segments

# Fade-out applied when audio is hard-truncated (ms)
FADE_OUT_MS = 300    # 300 ms – longer fade prevents the "voice snap" artefact

# When a segment is too long, allow it to bleed into the inter-segment gap
# by up to this many seconds (instead of hard-truncation).
OVERLAP_TOLERANCE_S = 1.5


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_duration(audio_path: str) -> float:
    """Return duration in seconds of an audio file (via ffprobe)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _rubberband_stretch(
    input_path: str,
    tempo: float,
    output_path: str,
) -> str:
    """
    Time-stretch *input_path* by *tempo* using the Rubber Band filter.

    ``tempo`` > 1 ⇒ audio gets shorter (faster playback).
    ``tempo`` < 1 ⇒ audio gets longer  (slower playback).

    Falls back to ffmpeg's ``atempo`` filter when Rubber Band is not available.

    Returns *output_path*.
    """
    if is_rubberband_available():
        # Build filter string with Rubber Band options tuned for speech:
        #   - formant=preserved  → keep vowel / speech-formant quality
        #   - transients=smooth  → fewer artefacts for spoken-word content
        #   - detector=soft      → better for non-percussive (speech) audio
        #   - window=long        → higher frequency resolution
        #   - pitchq=quality     → best pitch estimation quality
        rb_filter = (
            f"rubberband=tempo={tempo:.6f}"
            ":formant=preserved"
            ":transients=smooth"
            ":detector=soft"
            ":window=long"
            ":pitchq=quality"
        )
        af_filter = rb_filter
    else:
        # Fallback: ffmpeg's built-in atempo filter
        # atempo only supports [0.5, 100.0]; chain multiple for extreme values
        logger.debug("Rubber Band not available, falling back to atempo")
        af_filter = _build_atempo_chain(tempo)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", af_filter,
        "-ar", str(INTERMEDIATE_SR),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-vn",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def _build_atempo_chain(tempo: float) -> str:
    """
    Build an ffmpeg ``atempo`` filter chain for the given tempo ratio.

    The ``atempo`` filter only accepts values in [0.5, 100.0], so extreme
    ratios need to be chained (e.g. 0.3 → atempo=0.5,atempo=0.6).
    """
    if tempo <= 0:
        return "atempo=1.0"
    parts: list[str] = []
    remaining = tempo
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 100.0:
        parts.append("atempo=100.0")
        remaining /= 100.0
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def _time_stretch_segment(
    input_path: str, target_duration: float, output_path: str,
) -> str:
    """
    Time-stretch *input_path* so that the result fits within
    *target_duration* seconds.

    The tempo ratio is clamped to [MIN_TEMPO, MAX_TEMPO] to protect
    naturalness.  When clamping is active:
      • ratio was below MIN_TEMPO (audio far too short) → stretch to limit,
        then pad with silence to reach target_duration.
      • ratio was above MAX_TEMPO (audio far too long) → compress to limit,
        then hard-truncate to target_duration.

    Returns *output_path*.
    """
    actual_dur = _get_duration(input_path)

    if actual_dur <= 0 or target_duration <= 0:
        # Nothing useful — just copy
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-c:a", "pcm_s16le",
             "-ar", str(INTERMEDIATE_SR), "-ac", "1", output_path],
            capture_output=True, check=True,
        )
        return output_path

    raw_tempo = actual_dur / target_duration  # >1 ⇒ need to speed up

    # If close enough to 1×, just re-encode without stretching
    if 0.97 <= raw_tempo <= 1.03:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-c:a", "pcm_s16le",
             "-ar", str(INTERMEDIATE_SR), "-ac", "1", output_path],
            capture_output=True, check=True,
        )
        return output_path

    # ── Natural-speed fallback for extreme under-duration ───────────
    # When TTS audio is far shorter than the target window (raw_tempo <
    # NATURAL_SPEED_CUTOFF), rubber-band would need to stretch more than
    # 1/NATURAL_SPEED_CUTOFF ≈ 2.2×, which produces incomprehensible speech.
    # Better to play at natural speed and fill the gap with silence.
    if raw_tempo < NATURAL_SPEED_CUTOFF:
        tmp_dir = os.path.dirname(output_path) or tempfile.gettempdir()
        nat_path = os.path.join(tmp_dir, "_nat_" + os.path.basename(output_path))
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-c:a", "pcm_s16le",
             "-ar", str(INTERMEDIATE_SR), "-ac", "1", nat_path],
            capture_output=True, check=True,
        )
        pad_dur = target_duration - actual_dur
        sil_path = os.path.join(tmp_dir, "_nspad_" + os.path.basename(output_path))
        _generate_silence(pad_dur, sil_path)
        _concat_two(nat_path, sil_path, output_path)
        _safe_remove(nat_path, sil_path)
        logger.debug(
            "Natural-speed fallback: %.2fs TTS → %.2fs window (silence pad %.2fs)",
            actual_dur, target_duration, pad_dur,
        )
        return output_path

    clamped_tempo = max(MIN_TEMPO, min(MAX_TEMPO, raw_tempo))
    needs_pad   = raw_tempo < MIN_TEMPO   # stretched audio shorter than window
    needs_trunc = raw_tempo > MAX_TEMPO   # compressed audio longer than window

    tmp_dir = os.path.dirname(output_path) or tempfile.gettempdir()
    stretched = os.path.join(tmp_dir, "_rb_tmp_" + os.path.basename(output_path))

    _rubberband_stretch(input_path, clamped_tempo, stretched)

    stretched_dur = _get_duration(stretched)

    if needs_pad and stretched_dur < target_duration:
        # Pad silence at the end to fill remaining time
        pad_dur = target_duration - stretched_dur
        silence = os.path.join(tmp_dir, "_pad_" + os.path.basename(output_path))
        _generate_silence(pad_dur, silence)
        # Concatenate via filter (avoids concat demuxer format issues)
        _concat_two(stretched, silence, output_path)
        _safe_remove(stretched, silence)

    elif needs_trunc and stretched_dur > target_duration:
        # Hard-truncate to target_duration with a gentle fade-out
        fade_sec = FADE_OUT_MS / 1000.0
        fade_start = max(0, target_duration - fade_sec)
        af = f"afade=t=out:st={fade_start:.4f}:d={fade_sec:.4f}"
        cmd = [
            "ffmpeg", "-y",
            "-i", stretched,
            "-af", af,
            "-t", f"{target_duration:.4f}",
            "-c:a", "pcm_s16le",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        _safe_remove(stretched)

    else:
        # Stretched file is close to target — use as-is
        os.replace(stretched, output_path)

    return output_path


def _concat_two(a: str, b: str, out: str) -> None:
    """Concatenate two WAV files into *out* via the concat filter."""
    cmd = [
        "ffmpeg", "-y",
        "-i", a,
        "-i", b,
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[outa]",
        "-map", "[outa]",
        "-c:a", "pcm_s16le",
        "-ar", str(INTERMEDIATE_SR),
        "-ac", "1",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _generate_silence(
    duration: float, output_path: str, sample_rate: int | None = None,
) -> str:
    """Generate a silent WAV file of the given duration (seconds)."""
    sr = sample_rate or INTERMEDIATE_SR
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sr}:cl=mono",
        "-t", f"{duration:.4f}",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def _enforce_duration(input_wav: str, target_dur: float, output_mp3: str) -> None:
    """
    Ensure *input_wav* is exactly *target_dur* seconds and write to
    *output_mp3*.  Pads with silence if short, trims if long.
    """
    actual = _get_duration(input_wav)
    diff = actual - target_dur

    if abs(diff) < 0.05:
        # Close enough – just encode
        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-c:a", "libmp3lame", "-q:a", "2",
            output_mp3,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    elif diff > 0:
        # Audio is TOO LONG → trim
        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-t", f"{target_dur:.4f}",
            "-c:a", "libmp3lame", "-q:a", "2",
            output_mp3,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        logger.debug("Trimmed %.2fs to match target %.2fs", diff, target_dur)
    else:
        # Audio is TOO SHORT → pad silence at end
        pad_dur = target_dur - actual
        # Use the apad filter to extend to exact sample count, then trim
        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-af", f"apad=whole_dur={target_dur}",
            "-t", f"{target_dur:.4f}",
            "-c:a", "libmp3lame", "-q:a", "2",
            output_mp3,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        logger.debug("Padded %.2fs to match target %.2fs", pad_dur, target_dur)


def _safe_remove(*paths: str) -> None:
    """Silently delete files."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _voiced_end_in_window(
    source_audio_path: str,
    seg_start: float,
    seg_end: float,
    tmp_dir: str,
    silence_db: float = -38.0,
    min_silence_dur: float = 0.6,
) -> float:
    """
    Find when the speaker last spoke within the segment window [seg_start, seg_end].

    Uses ffmpeg silencedetect on the source audio (original lecture) to locate
    the final non-silent frame.  Returns the time offset from seg_start at which
    the last voiced speech ends (plus a small 300 ms tail buffer).

    When analysis fails, falls back to the full window duration so that normal
    stretching behaviour is preserved.

    This is the key correction for large windows that include long trailing
    silences (e.g. the professor paused for 30 s after a single sentence):
    the dubbed TTS audio should fill the SPEAKING portion, not the full window.
    """
    window_dur = seg_end - seg_start
    if window_dur <= 0:
        return window_dur

    seg_wav = os.path.join(tmp_dir, f"_vad_{seg_start:.2f}.wav")
    try:
        # Extract the source window at 16 kHz mono (fast, enough for VAD)
        subprocess.run(
            ["ffmpeg", "-y", "-i", source_audio_path,
             "-ss", f"{seg_start:.3f}", "-t", f"{window_dur:.3f}",
             "-ar", "16000", "-ac", "1", seg_wav],
            capture_output=True, check=True,
        )

        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", seg_wav,
             "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_dur}",
             "-f", "null", "/dev/null"],
            capture_output=True, text=True,
        )

        # Parse silence_start / silence_end pairs
        silence_intervals: list[tuple[float, float]] = []
        s_start: float | None = None
        for line in result.stderr.split("\n"):
            if "silence_start" in line:
                try:
                    s_start = float(line.split("silence_start:")[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
            elif "silence_end" in line and s_start is not None:
                try:
                    s_end = float(line.split("silence_end:")[1].strip().split("|")[0].strip())
                    silence_intervals.append((s_start, s_end))
                    s_start = None
                except (IndexError, ValueError):
                    pass
        # Unclosed silence (extends to window end)
        if s_start is not None:
            silence_intervals.append((s_start, window_dur))

        if not silence_intervals:
            # No silence detected — voice fills the whole window
            return window_dur

        # Find trailing silence: the last silence block that reaches the window end
        trailing_start = window_dur
        for si_start, si_end in reversed(silence_intervals):
            if si_end >= window_dur - 0.1:
                trailing_start = si_start
            else:
                break  # non-trailing silence found; stop

        # Effective voiced end = start of trailing silence + 300 ms tail buffer
        voiced_end = min(trailing_start + 0.30, window_dur)
        logger.debug(
            "VAD window %.1f–%.1f: voiced ends at +%.1fs / %.1fs",
            seg_start, seg_end, voiced_end, window_dur,
        )
        return max(voiced_end, 1.0)  # always give at least 1 s target

    except Exception as exc:
        logger.debug("VAD voiced-end detection failed for %.1f–%.1f: %s", seg_start, seg_end, exc)
        return window_dur
    finally:
        try:
            os.remove(seg_wav)
        except OSError:
            pass


# ── Main public function ─────────────────────────────────────────────────────

def align_dubbed_audio(
    original_segments: list[Segment],
    segment_audio_files: dict[int, str],
    total_duration: float,
    output_path: str,
    source_audio_path: str | None = None,
) -> str:
    """
    Build a full-length dubbed audio track with proper temporal alignment.

    Strategy (duration-aware pipeline):
      1. TTS already targets roughly the right duration via SSML rate control.
      2. This function fine-tunes via Rubber Band.
      3. If source_audio_path is provided, VAD detects the actual voiced end
         within each segment window.  For segments where the professor paused
         significantly (TTS << window), the target is clamped to the voiced
         portion — preventing long silent gaps mid-lecture.
      4. Silence pads short segments; bleed tolerates slightly long ones.

    Args:
        original_segments:   The English Segment list (carries start/end times).
        segment_audio_files: Mapping of segment index → per-segment TTS audio path.
        total_duration:      Total duration (seconds) of the original audio/video.
        output_path:         Where to write the aligned output audio (mp3).
        source_audio_path:   Optional path to the original (vocals) audio. When
                             provided, enables VAD-based voiced-end correction
                             for large windows with short TTS.

    Returns:
        Path to the generated full-length audio file.
    """
    with tempfile.TemporaryDirectory(prefix="align_") as tmp_dir:
        pieces: list[str] = []   # ordered list of WAV files to concat

        prev_end = 0.0

        # Pre-compute the gap *after* each segment (to the next segment's start).
        # This lets us decide how much overlap we can tolerate.
        gap_after: list[float] = []
        for idx in range(len(original_segments)):
            if idx + 1 < len(original_segments):
                gap_after.append(
                    original_segments[idx + 1].start - original_segments[idx].end
                )
            else:
                gap_after.append(total_duration - original_segments[idx].end)

        for idx, seg in enumerate(original_segments):
            gap = seg.start - prev_end

            if gap < -0.05:
                logger.warning(
                    "Segment %d starts %.3fs before previous end (overlap); "
                    "clamping to prev_end=%.3f",
                    idx, -gap, prev_end,
                )
                gap = 0.0

            # Insert silence to fill the gap before this segment
            if gap > 0.05:
                sil_path = os.path.join(tmp_dir, f"sil_{idx:04d}.wav")
                _generate_silence(gap, sil_path)
                pieces.append(sil_path)

            if idx in segment_audio_files and os.path.isfile(segment_audio_files[idx]):
                target_dur = seg.end - max(seg.start, prev_end if gap < 0 else seg.start)
                if target_dur <= 0:
                    target_dur = seg.end - seg.start

                tts_dur = _get_duration(segment_audio_files[idx])
                raw_ratio = tts_dur / target_dur if target_dur > 0 else 1.0

                # ── VAD-based voiced-end correction ───────────────────────
                # When the TTS is much shorter than the window (low ratio),
                # the professor probably paused inside the window. Use the
                # source audio to find when they actually stopped speaking,
                # so TTS fills the voiced portion rather than the full window.
                USE_VAD_THRESHOLD = 0.65  # apply when ratio < this
                if (
                    source_audio_path
                    and raw_ratio < USE_VAD_THRESHOLD
                    and target_dur > 4.0
                ):
                    voiced_end = _voiced_end_in_window(
                        source_audio_path, seg.start, seg.end, tmp_dir,
                    )
                    # Only shrink the target (never expand beyond original window)
                    if voiced_end < target_dur * 0.85:
                        logger.debug(
                            "Seg %d: VAD-trimmed target %.1fs → %.1fs (saves %.1fs silence)",
                            idx, target_dur, voiced_end, target_dur - voiced_end,
                        )
                        target_dur = voiced_end
                        raw_ratio = tts_dur / target_dur if target_dur > 0 else 1.0

                # ── Smart overlap: extend target if there's room after ──
                # Allows the TTS audio to breathe into the gap instead of
                # being stretched aggressively or hard-truncated.
                available_bleed = min(
                    max(gap_after[idx], 0.0),
                    OVERLAP_TOLERANCE_S,
                )

                if raw_ratio > MAX_TEMPO and available_bleed > 0.05:
                    # TTS is longer than our stretch limit allows — use some
                    # of the following gap instead of extreme compression
                    extra = min(tts_dur / MAX_TEMPO - target_dur, available_bleed)
                    effective_dur = target_dur + extra
                    logger.debug(
                        "Seg %d: bleed %.2fs into gap (ratio %.2f→%.2f)",
                        idx, extra, raw_ratio, tts_dur / effective_dur,
                    )
                    target_dur = effective_dur
                    # Reduce the gap that will be inserted before the NEXT segment
                    gap_after[idx] -= extra

                stretched_path = os.path.join(tmp_dir, f"seg_{idx:04d}_stretched.wav")
                _time_stretch_segment(
                    segment_audio_files[idx], target_dur, stretched_path,
                )
                pieces.append(stretched_path)
            else:
                # No TTS for this segment → fill with silence
                seg_dur = seg.end - seg.start
                if seg_dur > 0:
                    sil_path = os.path.join(tmp_dir, f"fill_{idx:04d}.wav")
                    _generate_silence(seg_dur, sil_path)
                    pieces.append(sil_path)

            prev_end = seg.end

        # Trailing silence after last segment
        trailing = total_duration - prev_end
        if trailing > 0.1:
            trail_path = os.path.join(tmp_dir, "trailing.wav")
            _generate_silence(trailing, trail_path)
            pieces.append(trail_path)

        if not pieces:
            raise ValueError("No audio pieces to assemble")

        # ── Optional cross-fade between consecutive segment pieces ───────
        if CROSSFADE_MS > 0 and len(pieces) > 1:
            pieces = _apply_crossfades(pieces, tmp_dir)

        # ── Final concatenation via concat demuxer ───────────────────────
        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in pieces:
                f.write(f"file '{p}'\n")

        # Concat WAV intermediates → temporary WAV, then enforce exact duration
        raw_concat = os.path.join(tmp_dir, "raw_concat.wav")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:a", "pcm_s16le",
            "-ar", str(INTERMEDIATE_SR),
            "-ac", "1",
            raw_concat,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # ── Enforce EXACT duration ───────────────────────────────────────
        _enforce_duration(raw_concat, total_duration, output_path)

    # TemporaryDirectory auto-cleans everything

    size_kb = os.path.getsize(output_path) / 1024
    logger.info("Aligned audio → %s (%.0f KB)", output_path, size_kb)
    return output_path


def _apply_crossfades(
    pieces: list[str], tmp_dir: str,
) -> list[str]:
    """
    Apply short cross-fades between adjacent audio pieces for smoother
    transitions.  Returns a new list of piece paths (some may be replaced
    with cross-faded versions).

    We apply a tiny fade-out at the end of each segment and fade-in at the
    start of the next.  This avoids discontinuity clicks.
    """
    fade_sec = CROSSFADE_MS / 1000.0
    result: list[str] = []
    for i, piece in enumerate(pieces):
        faded = os.path.join(tmp_dir, f"cf_{i:04d}.wav")
        af_parts: list[str] = []
        if i > 0:
            af_parts.append(f"afade=t=in:st=0:d={fade_sec:.4f}")
        try:
            dur = _get_duration(piece)
        except Exception:
            dur = 0
        if i < len(pieces) - 1 and dur > fade_sec:
            af_parts.append(f"afade=t=out:st={dur - fade_sec:.4f}:d={fade_sec:.4f}")

        if af_parts:
            cmd = [
                "ffmpeg", "-y",
                "-i", piece,
                "-af", ",".join(af_parts),
                "-c:a", "pcm_s16le",
                "-ar", str(INTERMEDIATE_SR),
                "-ac", "1",
                faded,
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                result.append(faded)
                continue
            except subprocess.CalledProcessError:
                logger.debug("Cross-fade failed for piece %d, using original", i)

        result.append(piece)
    return result
