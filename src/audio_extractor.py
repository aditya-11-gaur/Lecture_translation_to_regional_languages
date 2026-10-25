"""
audio_extractor.py
──────────────────
Extract audio track from a video file using ffmpeg (via subprocess).
Outputs a mono WAV at 16 kHz – ideal for speech-to-text APIs.
"""

import os
import subprocess
from config.settings import AUDIO_FORMAT, AUDIO_SAMPLE_RATE, OUTPUT_DIR


def extract_audio(
    video_path: str,
    output_dir: str | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = 1,
    suffix: str = "",
) -> str:
    """
    Extract audio from the given video file as 16-bit PCM WAV.

    The defaults (16 kHz mono) suit speech recognition.  Music separation,
    voice-cloning references and the final mix need the full-band audio:
    extract that with e.g. ``sample_rate=44100, channels=2, suffix="_full"``.

    Args:
        video_path:  Path to the input video file.
        output_dir:  Directory to write the audio file. Defaults to OUTPUT_DIR.
        sample_rate: Output sample rate (Hz).
        channels:    Output channel count.
        suffix:      Appended to the file name (before the extension).

    Returns:
        Path to the extracted audio file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}{suffix}.{AUDIO_FORMAT}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                          # drop video
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", str(channels),
        audio_path,
    ]

    print(f"[audio_extractor] Extracting audio → {audio_path}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[audio_extractor] Done. Size: {os.path.getsize(audio_path) / 1024:.1f} KB")

    return audio_path
