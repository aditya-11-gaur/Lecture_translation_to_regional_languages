"""
voice_analyzer.py
─────────────────
Extract speaker voice characteristics from the original audio
for voice-preserving TTS synthesis.

Extracts:
  • Average pitch / fundamental frequency (F0)
  • Volume / loudness (dBFS)
  • Pitch range (to detect male vs female speakers)

The gender picks a matching TTS voice; the pitch drives the post-TTS
pitch shift (pitch_shift_to_match) towards the original speaker.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("nptel_pipeline")


@dataclass
class VoiceProfile:
    """Characteristics of the original speaker's voice."""
    avg_pitch_hz: float = 0.0                  # average fundamental frequency
    pitch_range: tuple[float, float] = (0.0, 0.0)  # (min, max) Hz
    avg_volume_dbfs: float = 0.0               # average volume in dBFS
    is_male: bool | None = None                # inferred gender (None = unknown)
    duration_seconds: float = 0.0              # total audio duration

    def to_dict(self) -> dict:
        return {
            "avg_pitch_hz": round(self.avg_pitch_hz, 1),
            "pitch_range": (round(self.pitch_range[0], 1), round(self.pitch_range[1], 1)),
            "avg_volume_dbfs": round(self.avg_volume_dbfs, 1),
            "is_male": self.is_male,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def analyze_voice(audio_path: str) -> VoiceProfile:
    """
    Analyze the original audio to build a voice profile.

    Args:
        audio_path: Path to the source audio file (isolated vocals if available).

    Returns:
        VoiceProfile with extracted characteristics.
    """
    profile = VoiceProfile()

    # Duration
    profile.duration_seconds = _get_duration(audio_path)

    # Volume analysis via ffmpeg
    try:
        volume_info = _analyze_volume(audio_path)
        profile.avg_volume_dbfs = volume_info.get("mean_volume", 0.0)
    except Exception as exc:
        logger.warning("Volume analysis failed: %s", exc)

    # Pitch analysis (sample 60 s of non-silent audio for speed)
    try:
        pitch_info = _analyze_pitch(audio_path, max_seconds=60, skip_silence=True)
        profile.avg_pitch_hz = pitch_info.get("avg_hz", 0.0)
        profile.pitch_range = pitch_info.get("range", (0.0, 0.0))

        if profile.avg_pitch_hz > 0:
            profile.is_male = profile.avg_pitch_hz < 165
    except Exception as exc:
        logger.warning("Pitch analysis failed: %s", exc)

    logger.info(
        "Voice profile: pitch=%.0f Hz (%s), volume=%.1f dBFS",
        profile.avg_pitch_hz,
        "male" if profile.is_male else ("female" if profile.is_male is False else "unknown"),
        profile.avg_volume_dbfs,
    )

    return profile


def _get_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _analyze_volume(audio_path: str) -> dict:
    """Analyze volume using ffmpeg's volumedetect filter."""
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", "volumedetect",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    info: dict = {}
    for line in stderr.split("\n"):
        if "mean_volume" in line:
            try:
                val = float(line.split("mean_volume:")[1].strip().split()[0])
                info["mean_volume"] = val
            except (ValueError, IndexError):
                pass
        elif "max_volume" in line:
            try:
                val = float(line.split("max_volume:")[1].strip().split()[0])
                info["max_volume"] = val
            except (ValueError, IndexError):
                pass

    return info


def _analyze_pitch(
    audio_path: str, max_seconds: float = 60, skip_silence: bool = False,
) -> dict:
    """
    Estimate fundamental frequency (F0) by autocorrelation on voiced frames.

    Only analyses up to max_seconds of audio for speed.  With
    ``skip_silence`` the pauses are removed first, so a dubbed track that
    opens with a long silence still yields max_seconds of speech.
    """
    # Convert to raw PCM for analysis
    silence_args = (
        ["-af", "silenceremove=start_periods=1:start_threshold=-40dB"
                ":stop_periods=-1:stop_duration=0.3:stop_threshold=-40dB"]
        if skip_silence else []
    )
    duration_args = ["-t", str(max_seconds)] if max_seconds > 0 else []
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        *silence_args,
        *duration_args,
        "-ar", "16000",
        "-ac", "1",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    raw_data = result.stdout

    if not raw_data:
        return {"avg_hz": 0.0, "range": (0.0, 0.0)}

    # Parse 16-bit signed PCM samples
    sample_rate = 16000
    num_samples = len(raw_data) // 2
    if num_samples == 0:
        return {"avg_hz": 0.0, "range": (0.0, 0.0)}

    samples = np.frombuffer(raw_data[:num_samples * 2], dtype="<i2").astype(np.float64)

    # Analyze pitch using autocorrelation on voiced frames
    frame_size = int(0.03 * sample_rate)   # 30ms frames
    hop_size = int(0.015 * sample_rate)    # 15ms hop (faster)

    pitches: list[float] = []

    for start in range(0, len(samples) - frame_size, hop_size):
        frame = samples[start:start + frame_size]

        # Check if frame is voiced (energy above threshold)
        energy = float(np.dot(frame, frame)) / len(frame)
        if energy < 500000:  # silence/noise threshold
            continue

        # Autocorrelation-based pitch detection
        pitch = _autocorrelation_pitch(frame, sample_rate)
        if 60 < pitch < 500:  # valid speech range
            pitches.append(pitch)

    if not pitches:
        return {"avg_hz": 0.0, "range": (0.0, 0.0)}

    # Filter outliers (keep middle 80%)
    pitches.sort()
    n = len(pitches)
    trimmed = pitches[n // 10: 9 * n // 10] if n > 20 else pitches

    avg_hz = sum(trimmed) / len(trimmed) if trimmed else 0.0
    min_hz = trimmed[0] if trimmed else 0.0
    max_hz = trimmed[-1] if trimmed else 0.0

    return {"avg_hz": avg_hz, "range": (min_hz, max_hz)}


def _autocorrelation_pitch(frame, sample_rate: int) -> float:
    """
    Estimate pitch of a single frame using autocorrelation.
    Returns estimated frequency in Hz.
    """
    x = np.asarray(frame, dtype=np.float64)
    n = len(x)
    # Search for pitch between 60 Hz and 500 Hz
    min_lag = sample_rate // 500   # 500 Hz
    max_lag = sample_rate // 60    # 60 Hz

    if max_lag >= n:
        max_lag = n - 1

    # Compute energy for normalization
    energy = float(np.dot(x, x))
    if energy == 0 or max_lag <= min_lag:
        return 0.0

    # corr[lag] = sum(x[i] * x[i + lag]) / energy, for every lag at once
    corr = np.correlate(x, x, mode="full")[n - 1:] / energy
    best_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    best_corr = corr[best_lag]

    if best_corr < 0.25:
        return 0.0

    return sample_rate / best_lag


# Largest post-TTS pitch correction.  The shift moves formants with the pitch
# (see apply_voice_profile_to_audio), which stays natural for small shifts —
# and the TTS voice already matches the lecturer's gender.
MAX_PITCH_SHIFT_SEMITONES = 3.0


def pitch_shift_to_match(profile: VoiceProfile, tts_audio_path: str) -> float:
    """
    Semitones to shift *tts_audio_path* so its average pitch matches the
    original speaker's.

    Measures the TTS audio itself rather than assuming a typical TTS pitch:
    the voice actually used may be male or female, cloned, or already
    pitch-adjusted at synthesis time (edge-tts).
    """
    if profile.avg_pitch_hz <= 0:
        return 0.0
    tts_hz = _analyze_pitch(tts_audio_path, max_seconds=60, skip_silence=True)["avg_hz"]
    if tts_hz <= 0:
        return 0.0
    semitones = 12 * math.log2(profile.avg_pitch_hz / tts_hz)
    return max(-MAX_PITCH_SHIFT_SEMITONES, min(MAX_PITCH_SHIFT_SEMITONES, semitones))


def apply_voice_profile_to_audio(
    audio_path: str,
    profile: VoiceProfile,
    output_path: str,
    semitones: float | None = None,
) -> str:
    """
    Post-process TTS audio to better match the original speaker.

    Applies a pitch shift (semitones) to match the original speaker's pitch.
    ``semitones`` defaults to ``pitch_shift_to_match(profile, audio_path)``.
    The output codec follows *output_path*'s extension (.flac / .wav / .mp3).
    """
    if semitones is None:
        semitones = pitch_shift_to_match(profile, audio_path)

    # Resample-based shift (pitch and formants move together) + WSOLA tempo
    # restore.  Measured on lecture dubs, Rubber Band's formant-preserving
    # shift dropped estimated speech quality (SQUIM PESQ) from 3.5 to 1.3;
    # this keeps it at 3.5.
    codec = (["-c:a", "flac"] if output_path.lower().endswith(".flac")
             else ["-c:a", "pcm_s16le"] if output_path.lower().endswith(".wav")
             else ["-c:a", "libmp3lame", "-b:a", "192k"])
    if abs(semitones) <= 0.5:
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, *codec, output_path],
                       capture_output=True, check=True)
        return output_path

    sr = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    ratio = 2 ** (semitones / 12)
    af = f"asetrate={sr * ratio:.3f},aresample={sr},atempo={1 / ratio:.6f}"
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-af", af, *codec, output_path],
                   capture_output=True, check=True)
    return output_path
