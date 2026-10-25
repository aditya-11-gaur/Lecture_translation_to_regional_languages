"""
audio_separator.py
──────────────────
Vocal / background music separation using Demucs (by Meta Research).

This enables the "Preserve Background Music" pipeline mode:

  ┌──────────────────────────────────────────────────────────────────┐
  │  Original audio  ─►  Demucs  ─►  vocals.wav   ─►  STT / TTS    │
  │                              └►  no_vocals.wav ─►  kept as-is   │
  │                                                                  │
  │  Final mix = dubbed vocals + original no_vocals (music/effects) │
  └──────────────────────────────────────────────────────────────────┘

Demucs installation
───────────────────
  pip install demucs

First run downloads the htdemucs model (~80 MB) automatically.
No GPU is required — CPU mode is used as fallback automatically.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger("nptel_pipeline")


# ─── Availability ─────────────────────────────────────────────────────────────

def is_demucs_available() -> bool:
    """Return True if Demucs is importable."""
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Separation ───────────────────────────────────────────────────────────────

def separate_audio(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs",
) -> tuple[str, str]:
    """
    Separate ``audio_path`` into vocals and accompaniment using Demucs.

    Uses ``--two-stems=vocals`` mode which produces exactly two stems:
      • ``vocals.wav``        — the singer/speaker
      • ``no_vocals.wav``     — everything else (music, instruments, effects)

    Args:
        audio_path:  Path to the source audio (WAV, MP3, …).
        output_dir:  Where Demucs writes its output tree.
        model:       Demucs model name (default: htdemucs, ~80 MB download).

    Returns:
        (vocals_path, accompaniment_path) — both as absolute WAV paths.

    Raises:
        RuntimeError: if Demucs is not installed or the subprocess fails.
    """
    if not is_demucs_available():
        raise RuntimeError(
            "Demucs is not installed.\n"
            "Install it with:  pip install demucs"
        )

    logger.info("Separating audio with Demucs (model=%s)…", model)

    sep_dir = os.path.join(output_dir, "demucs_sep")
    os.makedirs(sep_dir, exist_ok=True)

    # Run Demucs as a subprocess so we don't need to manage its torch state
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", model,
        "-o", sep_dir,
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Demucs writes:  sep_dir/{model}/{audio_basename_no_ext}/vocals.wav
    #                 sep_dir/{model}/{audio_basename_no_ext}/no_vocals.wav
    stem_name = os.path.splitext(os.path.basename(audio_path))[0]
    model_dir  = os.path.join(sep_dir, model, stem_name)

    vocals_path = os.path.join(model_dir, "vocals.wav")
    accomp_path = os.path.join(model_dir, "no_vocals.wav")

    for p in (vocals_path, accomp_path):
        if not os.path.exists(p):
            raise RuntimeError(
                f"Demucs finished but expected output not found: {p}\n"
                f"Directory contents: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'missing'}"
            )

    logger.info(
        "Separation complete:\n  vocals:        %s\n  accompaniment: %s",
        vocals_path, accomp_path,
    )
    return vocals_path, accomp_path


# ─── Mixing ───────────────────────────────────────────────────────────────────

def mix_audio_tracks(
    speech_path: str,
    accompaniment_path: str,
    output_path: str,
    speech_gain_db: float = 0.0,
    music_gain_db: float = -6.0,
    silence_gate_db: float = -35.0,
) -> str:
    """
    Mix dubbed speech back with the original accompaniment using ffmpeg.

    ``speech_gain_db`` adjusts the TTS track level (0 = unchanged).
    ``music_gain_db``  lowers the music slightly so the dubbed voice is
    clearly audible over the background.
    ``silence_gate_db`` — if the accompaniment's max volume (dB) is below
    this threshold, the track is considered near-silent (pure-lecture video
    with no background music) and mixing is skipped to avoid wasting time
    mixing silence into the output.

    Returns ``output_path`` (or ``speech_path`` when mixing is skipped).
    """
    # ── Energy gate: check if accompaniment has real content ────────────
    try:
        # volumedetect reports at info level — don't pass "-v error" here
        _vol_check = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", accompaniment_path,
             "-af", "volumedetect", "-f", "null", "/dev/null"],
            capture_output=True, text=True,
        )
        _max_vol = None
        for _line in _vol_check.stderr.split("\n"):
            if "max_volume" in _line:
                # e.g. "max_volume: -14.5 dB"
                _max_vol = float(_line.split(":")[1].strip().split()[0])
                break
        if _max_vol is not None and _max_vol < silence_gate_db:
            logger.info(
                "Accompaniment max volume %.1f dB is below silence gate %.1f dB "
                "(lecture has no background music) — skipping music mix",
                _max_vol, silence_gate_db,
            )
            return speech_path
        logger.info("Accompaniment energy: max=%.1f dB — proceeding with mix", _max_vol or 0)
    except Exception as _check_exc:
        logger.debug("Accompaniment energy check failed (%s) — proceeding with mix anyway", _check_exc)

    # Build volume filter expressions
    s_vol = f"volume={speech_gain_db}dB" if speech_gain_db != 0.0 else "anull"
    m_vol = f"volume={music_gain_db}dB"  if music_gain_db  != 0.0 else "anull"

    filter_complex = (
        f"[0:a]{s_vol}[speech];"
        f"[1:a]{m_vol}[music];"
        "[speech][music]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", speech_path,
        "-i", accompaniment_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mix failed: {result.stderr.decode()}"
        )

    logger.info("Mixed audio written to %s", output_path)
    return output_path


def mix_all_tracks(
    aligned_audio: dict[str, str],
    accompaniment_path: str,
    output_dir: str,
    video_name: str,
    music_gain_db: float = -6.0,
) -> dict[str, str]:
    """
    Mix every aligned dubbed audio track with the original accompaniment.

    Returns a dict mapping lang_code → mixed audio path (or original
    dubbed path when the accompaniment is near-silent / mixing is skipped).
    """
    mixed: dict[str, str] = {}
    for lang_code, dubbed_path in aligned_audio.items():
        out_path = os.path.join(output_dir, f"{video_name}_{lang_code}_mixed.mp3")
        try:
            result_path = mix_audio_tracks(
                speech_path=dubbed_path,
                accompaniment_path=accompaniment_path,
                output_path=out_path,
                music_gain_db=music_gain_db,
            )
            # mix_audio_tracks returns speech_path when mixing was skipped
            mixed[lang_code] = result_path
        except Exception as exc:
            logger.warning(
                "Audio mix failed for %s (falling back to speech-only): %s",
                lang_code, exc,
            )
            mixed[lang_code] = dubbed_path
    return mixed
