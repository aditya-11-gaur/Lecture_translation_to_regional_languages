"""
prosody_transfer.py
───────────────────
Pitch (F0) contour transfer from an English source track to a dubbed target
track, using Praat via Parselmouth.

Design notes
────────────
After time-stretching, each dubbed segment has (approximately)
the same duration as its English counterpart.  We therefore operate on the
*full* aligned audio files, which are already duration-matched.

The key fix vs. the naïve approach: the source and target may still differ
slightly in total duration (rounding, padding).  We correct this by scaling
each point's time coordinate in the extracted PitchTier so that the tier
spans [0, target_duration] before replacing the manipulation object's pitch.

F0 floor / ceiling are derived from the optional VoiceProfile so that we
only track the speaker's actual pitch range and don't capture creaks or
falsetto artefacts.

Limitation
──────────
This transfers the *lecturer's* English intonation contour onto the Hindi/
Telugu/Odia dubbed audio.  Because Hindi has its own prosodic rules, some
intonation patterns will sound unnatural.  For NPTEL lectures (academic,
monotone style) the effect is mild and mainly preserves sentence-final
rising/falling cues and emphasis.  Prosody transfer is therefore optional
and disabled by default.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger("nptel_pipeline")


def is_parselmouth_available() -> bool:
    try:
        import parselmouth  # noqa: F401
        return True
    except ImportError:
        return False


def apply_prosody_transfer(
    source_audio_path: str,
    target_audio_path: str,
    output_audio_path: str,
    f0_floor: float | None = None,
    f0_ceiling: float | None = None,
    voice_profile=None,
) -> str:
    """
    Copy the pitch (F0) contour from *source_audio_path* onto
    *target_audio_path* and write the result to *output_audio_path*.

    Args:
        source_audio_path:  Original English audio (any format; converted
                            internally to WAV for Praat).
        target_audio_path:  Dubbed audio after alignment (any
                            format).
        output_audio_path:  Destination path for the prosody-transferred
                            audio.  Format is inferred from the extension.
        f0_floor:           Minimum F0 in Hz for pitch tracking.  Defaults
                            to 60 Hz (male voice) or 100 Hz (female voice)
                            derived from *voice_profile*, or 75 Hz if no
                            profile is available.
        f0_ceiling:         Maximum F0 in Hz.  Defaults to 250/400 Hz from
                            the voice profile, or 600 Hz as fallback.
        voice_profile:      Optional VoiceProfile from voice_analyzer.py.
                            Used to narrow the F0 search range and reduce
                            creakiness / falsetto artefacts.

    Returns:
        *output_audio_path* on success, *target_audio_path* on failure
        (so the pipeline can continue with the unmodified dubbed audio).
    """
    if not is_parselmouth_available():
        raise RuntimeError(
            "praat-parselmouth is not installed. "
            "Run: pip install praat-parselmouth"
        )

    import parselmouth
    from parselmouth.praat import call  # noqa: PLC0415

    # ── Derive F0 range from voice profile ───────────────────────────────
    if voice_profile is not None:
        is_male = getattr(voice_profile, "is_male", None)
        if is_male is True:
            f0_floor   = f0_floor   or 60.0
            f0_ceiling = f0_ceiling or 250.0
        elif is_male is False:
            f0_floor   = f0_floor   or 100.0
            f0_ceiling = f0_ceiling or 400.0
        else:
            f0_floor   = f0_floor   or 75.0
            f0_ceiling = f0_ceiling or 500.0
    else:
        f0_floor   = f0_floor   or 75.0
        f0_ceiling = f0_ceiling or 600.0

    tmp_files: list[str] = []

    def _to_wav(path: str, label: str) -> str:
        """Convert *path* to a mono 44.1 kHz WAV in a temp file."""
        if path.lower().endswith(".wav"):
            return path
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_{label}.wav", delete=False
        )
        tmp.close()
        tmp_files.append(tmp.name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "44100", tmp.name],
            capture_output=True, check=True,
        )
        return tmp.name

    try:
        src_wav = _to_wav(source_audio_path, "src")
        tgt_wav = _to_wav(target_audio_path, "tgt")

        logger.info(
            "Prosody transfer: extracting F0 from %s (floor=%.0f Hz, ceil=%.0f Hz)",
            os.path.basename(source_audio_path), f0_floor, f0_ceiling,
        )

        s_source = parselmouth.Sound(src_wav)
        s_target = parselmouth.Sound(tgt_wav)

        src_dur = s_source.duration
        tgt_dur = s_target.duration

        # ── Extract PitchTier from source ─────────────────────────────
        pitch_source = s_source.to_pitch(
            time_step=0.01,
            pitch_floor=f0_floor,
            pitch_ceiling=f0_ceiling,
        )
        pitch_tier = call(pitch_source, "Down to PitchTier")

        # ── Time-scale pitch tier to match target duration ────────────
        # Without this step, when src_dur ≠ tgt_dur Praat either errors
        # out or produces garbage (out-of-range pitch frames).
        if abs(src_dur - tgt_dur) > 0.05:  # >50 ms mismatch → rescale
            logger.debug(
                "Prosody transfer: scaling pitch tier %.3fs → %.3fs",
                src_dur, tgt_dur,
            )
            scale = tgt_dur / src_dur
            n_pts = call(pitch_tier, "Get number of points")
            scaled_tier = call("Create PitchTier", "scaled", 0.0, tgt_dur)
            for idx in range(1, n_pts + 1):
                t = call(pitch_tier, "Get time from index", idx)
                v = call(pitch_tier, "Get value at index", idx)
                t_sc = t * scale
                if 0.0 < t_sc <= tgt_dur:
                    call(scaled_tier, "Add point", t_sc, v)
            pitch_tier = scaled_tier

        # ── Replace target's pitch tier via Manipulation object ───────
        # Parameters: time_step=0.01 s, f0_floor, f0_ceiling
        manipulation = call(
            s_target, "To Manipulation", 0.01, f0_floor, f0_ceiling
        )
        call([pitch_tier, manipulation], "Replace pitch tier")

        # ── Resynthesize (overlap-add) ────────────────────────────────
        resynthesized = call(manipulation, "Get resynthesis (overlap-add)")

        # Write WAV then encode to final format via ffmpeg
        tmp_wav = tempfile.NamedTemporaryFile(
            suffix="_prosody_tmp.wav", delete=False
        )
        tmp_wav.close()
        tmp_files.append(tmp_wav.name)

        resynthesized.save(tmp_wav.name, "WAV")

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_wav.name,
                "-ac", "2", "-ar", "44100", "-b:a", "192k",
                output_audio_path,
            ],
            capture_output=True, check=True,
        )

        logger.info("Prosody transfer complete → %s",
                    os.path.basename(output_audio_path))
        return output_audio_path

    except Exception as exc:
        logger.error("Prosody transfer failed: %s", exc)
        # Non-fatal: caller continues with the unmodified dubbed audio
        return target_audio_path

    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass
