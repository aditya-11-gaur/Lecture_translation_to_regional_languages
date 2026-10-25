"""
audio_aligner.py
────────────────
Temporal alignment for dubbed audio.

Takes per-sentence TTS audio files and lays them out on a full-length track
so each dubbed sentence plays when the English sentence was spoken.

Approach
────────
  • **One natural speaking speed.**  TTS is synthesised at a single pace for
    the whole lecture.  The aligner only nudges a sentence — at most
    MIN_TEMPO (8 % slower) to fill a little of its own time, at most
    MAX_TEMPO (15 % faster) to fit before the next sentence — so the voice
    never audibly speeds up and slows down from sentence to sentence.
  • **Pauses absorb the rest.**  A sentence that is still too long runs into
    the pause after it and starts the next sentence up to MAX_DRIFT_S late;
    the drift is recovered in later pauses.  Only beyond that does it speed
    up further (HARD_MAX_TEMPO).  Speech is never truncated mid-word.
  • **WSOLA time-stretching** (ffmpeg ``atempo``).  Measured on real lecture
    dubs, Rubber Band's phase vocoder dropped estimated speech quality
    (SQUIM PESQ) from 3.5 to 1.2–2.0 even at tempo ≈ 1.0; atempo kept 3.35.
  • **Real speech onsets.**  Whisper places a sentence's first word ~0.3 s
    early; when the source audio is silent at that point, the sentence
    starts where the speech actually resumes (up to ONSET_SNAP_S later).
  • **Even loudness.**  Every sentence is normalised to TARGET_LUFS.
  • **Lossless intermediates** (48 kHz PCM); the output codec follows the
    output file's extension (.flac / .wav / .mp3).
"""

from __future__ import annotations
import json
import logging
import os
import re
import subprocess
import tempfile
import wave

from src.transcriber import Segment

logger = logging.getLogger("nptel_pipeline")


# ── Configuration ────────────────────────────────────────────────────────────

# Speaking-rate limits (tempo > 1 ⇒ faster).  Kept narrow on purpose: before
# this, per-sentence speed control made the heard rate swing 8–17 chars/s.
MIN_TEMPO = 0.92        # at most 8 % slower
MAX_TEMPO = 1.15        # at most 15 % faster before borrowing the next pause
HARD_MAX_TEMPO = 1.35   # only when the dub would otherwise run MAX_DRIFT_S late
MAX_DRIFT_S = 1.5       # how late a sentence may start vs. the original

# A sentence always leaves at least this much silence before the next one.
MIN_PAUSE_S = 0.15

# Move a sentence start at most this far forward to the real speech onset.
ONSET_SNAP_S = 0.6

# Loudness every sentence is normalised to (and the gain limits for it).
TARGET_LUFS = -20.0
MAX_GAIN_DB = 12.0
PEAK_CEILING_DBFS = -1.0

# Intermediate sample rate (Hz)
INTERMEDIATE_SR = 48_000

# Fade at piece boundaries (ms) – avoids clicks between sentences / silence
CROSSFADE_MS = 50


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


def _wav_duration(wav_path: str) -> float:
    """Exact duration of an intermediate PCM WAV (falls back to ffprobe)."""
    try:
        with wave.open(wav_path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, OSError):
        return _get_duration(wav_path)


def _build_atempo_chain(tempo: float) -> str:
    """
    Build an ffmpeg ``atempo`` filter chain for the given tempo ratio.

    The ``atempo`` filter only accepts values in [0.5, 100.0]; chain multiple
    for extreme values (e.g. 0.3 → atempo=0.5,atempo=0.6).
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


def _codec_args(path: str) -> list[str]:
    """ffmpeg codec arguments for an output file, chosen by its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".flac":
        return ["-c:a", "flac"]
    if ext == ".wav":
        return ["-c:a", "pcm_s16le"]
    return ["-c:a", "libmp3lame", "-b:a", "192k"]


def _loudness(audio_path: str) -> tuple[float | None, float | None]:
    """(integrated loudness LUFS, true peak dBFS) via ffmpeg ebur128."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    summary = r.stderr[r.stderr.rfind("Summary:"):]
    lufs = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    return (float(lufs.group(1)) if lufs else None,
            float(peak.group(1)) if peak else None)


def _normalising_gain(audio_path: str) -> float:
    """Gain (dB) that brings a sentence to TARGET_LUFS without clipping."""
    lufs, peak = _loudness(audio_path)
    if lufs is None or lufs < -60:          # (near-)silent or too short to measure
        return 0.0
    gain = TARGET_LUFS - lufs
    if peak is not None:
        gain = min(gain, PEAK_CEILING_DBFS - peak)
    return max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain))


def _render_sentence(input_path: str, tempo: float, gain_db: float, output_path: str) -> str:
    """Re-time (WSOLA) and level one sentence into an intermediate WAV."""
    filters = []
    if abs(tempo - 1.0) > 0.005:
        filters.append(_build_atempo_chain(tempo))
    if abs(gain_db) > 0.1:
        filters.append(f"volume={gain_db:.2f}dB")
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ar", str(INTERMEDIATE_SR), "-ac", "1", "-c:a", "pcm_s16le", output_path]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


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


def _enforce_duration(input_wav: str, target_dur: float, output_path: str) -> None:
    """
    Write *input_wav* to *output_path* at exactly *target_dur* seconds
    (padding with silence or trimming), encoded per the output extension.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", f"apad=whole_dur={target_dur:.4f}",
        "-t", f"{target_dur:.4f}",
        *_codec_args(output_path),
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _safe_remove(*paths: str) -> None:
    """Silently delete files."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _silences(source_audio_path: str, silence_db: float = -38.0,
              min_silence_dur: float = 0.2) -> list[tuple[float, float]]:
    """(start, end) of every silence in the source audio (ffmpeg silencedetect)."""
    # silencedetect reports at info level — don't pass "-v error" here
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", source_audio_path,
         "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_dur}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    out, start = [], None
    for line in r.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].split()[0])
        elif "silence_end:" in line and start is not None:
            out.append((start, float(line.split("silence_end:")[1].split("|")[0])))
            start = None
    return out


def _snap_to_onset(start: float, silences: list[tuple[float, float]]) -> float:
    """
    If *start* falls inside a silence that ends within ONSET_SNAP_S, return
    that silence's end (where the speech really starts); else *start*.
    """
    for s, e in silences:
        if s - 0.05 <= start < e and e - start <= ONSET_SNAP_S:
            return e
    return start


def _plan_timing(
    starts: list[float],
    ends: list[float],
    durations: list[float | None],
    total_duration: float,
) -> list[tuple[float, float] | None]:
    """
    Decide when each sentence plays and its tempo.

    Args:
        starts, ends:   preferred (original) times of each sentence.
        durations:      natural length of each sentence's TTS, None if it has none.
        total_duration: length of the track.

    Returns:
        (start, tempo) per sentence, None where there is no audio.
    """
    plan: list[tuple[float, float] | None] = []
    cursor = 0.0
    n = len(starts)
    for i in range(n):
        d = durations[i]
        if not d:
            plan.append(None)
            continue
        next_start = next((starts[j] for j in range(i + 1, n) if durations[j]), None)
        start = max(starts[i], cursor + MIN_PAUSE_S if cursor > 0 else 0.0)

        # Fill a little of the sentence's own time — never noticeably slower
        own = ends[i] - starts[i]
        tempo = max(MIN_TEMPO, d / own) if 0 < d < own else 1.0

        # Finish before the next sentence; past the end of the video no drift
        deadline = (next_start - MIN_PAUSE_S) if next_start is not None else total_duration
        drift_ok = MAX_DRIFT_S if next_start is not None else 0.0
        if start + d / tempo > deadline:
            budget = deadline - start
            tempo = max(1.0, min(MAX_TEMPO, d / budget)) if budget > 0 else MAX_TEMPO
            room = deadline + drift_ok - start
            if start + d / tempo > deadline + drift_ok:
                tempo = min(HARD_MAX_TEMPO, d / room) if room > 0 else HARD_MAX_TEMPO
        plan.append((start, tempo))
        cursor = start + d / tempo
    return plan


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

    Args:
        original_segments:   The English Segment list (carries start/end times).
        segment_audio_files: Mapping of segment index → per-segment TTS audio path.
        total_duration:      Total duration (seconds) of the original audio/video.
        output_path:         Where to write the aligned track (.flac/.wav/.mp3).
        source_audio_path:   Optional original (vocals) audio; used to start each
                             sentence at the real speech onset.

    Returns:
        Path to the generated full-length audio file.
    """
    silences = _silences(source_audio_path) if source_audio_path else []
    starts = [_snap_to_onset(s.start, silences) for s in original_segments]
    ends = [max(s.end, st) for s, st in zip(original_segments, starts)]

    with tempfile.TemporaryDirectory(prefix="align_") as tmp_dir:
        sources: dict[int, str] = {}
        durations: list[float | None] = []
        for idx in range(len(original_segments)):
            path = segment_audio_files.get(idx)
            if path and os.path.isfile(path):
                sources[idx] = path
                durations.append(_get_duration(path))
            else:
                durations.append(None)

        plan = _plan_timing(starts, ends, durations, total_duration)

        pieces: list[str] = []   # ordered list of WAV files to concat
        # Where the assembled track actually ends so far.  Every gap is
        # measured from here, so a piece that comes out a few ms long or short
        # can never shift the sentences that follow it.
        cursor = 0.0
        late = []
        for idx, step in enumerate(plan):
            if step is None:
                continue
            start, tempo = step
            gap = start - cursor
            if gap > 0.001:
                sil_path = os.path.join(tmp_dir, f"sil_{idx:04d}.wav")
                _generate_silence(gap, sil_path)
                pieces.append(sil_path)
                cursor += _wav_duration(sil_path)

            piece = os.path.join(tmp_dir, f"seg_{idx:04d}.wav")
            _render_sentence(sources[idx], tempo, _normalising_gain(sources[idx]), piece)
            pieces.append(piece)
            cursor += _wav_duration(piece)
            if start - starts[idx] > 0.3:
                late.append(start - starts[idx])
            if abs(tempo - 1.0) > 0.16:
                logger.info("  Align seg %d: tempo %.2f (TTS %.2fs, window %.2fs)",
                            idx, tempo, durations[idx], ends[idx] - starts[idx])

        tempos = [t for _, t in filter(None, plan)]
        if tempos:
            logger.info(
                "Alignment: %d sentences, tempo %.2f–%.2f, %d started late (max %.1fs)",
                len(tempos), min(tempos), max(tempos), len(late), max(late, default=0.0),
            )

        # Trailing silence after last segment
        trailing = total_duration - cursor
        if trailing > 0.1:
            trail_path = os.path.join(tmp_dir, "trailing.wav")
            _generate_silence(trailing, trail_path)
            pieces.append(trail_path)

        if not pieces:
            raise ValueError("No audio pieces to assemble")

        # ── Short fades between consecutive pieces (no clicks) ───────────
        if CROSSFADE_MS > 0 and len(pieces) > 1:
            pieces = _apply_crossfades(pieces, tmp_dir)

        # ── Final concatenation via concat demuxer ───────────────────────
        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in pieces:
                f.write(f"file '{p}'\n")

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

    size_kb = os.path.getsize(output_path) / 1024
    logger.info("Aligned audio → %s (%.0f KB)", output_path, size_kb)
    return output_path


def _apply_crossfades(
    pieces: list[str], tmp_dir: str,
) -> list[str]:
    """
    Apply a short fade-in at the start and fade-out at the end of each piece
    so joins between sentences and silence never click.  Returns the new
    list of piece paths (same lengths as before).
    """
    fade_sec = CROSSFADE_MS / 1000.0
    result: list[str] = []
    for i, piece in enumerate(pieces):
        faded = os.path.join(tmp_dir, f"cf_{i:04d}.wav")
        af_parts: list[str] = []
        if i > 0:
            af_parts.append(f"afade=t=in:st=0:d={fade_sec:.4f}")
        try:
            dur = _wav_duration(piece)
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
