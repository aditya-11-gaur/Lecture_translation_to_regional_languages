"""
voice_converter.py
──────────────────
Stage 2 of voice preservation: OpenVoice v2 tone-color transfer.

Two-stage pipeline architecture
────────────────────────────────
  Stage 1  =  Language-correct dubbed speech  (Sarvam / edge-tts)
              → correct words, correct script, generic TTS voice

  Stage 2  =  Tone-color transfer  (OpenVoice v2)
              → same speech, but the TIMBRE sounds like the original lecturer

How it works
────────────
OpenVoice v2 is a speaker-identity model.  It is entirely language-agnostic
because it operates on the acoustic/spectral fingerprint of a voice, not on
its linguistic content.  The three-step process is:

  1. Extract a "speaker embedding" (tone colour vector) from a reference clip
     of the original lecturer.
  2. Extract a "speaker embedding" from the TTS-generated dubbed audio.
  3. Convert the TTS audio so that its tone colour matches the reference.

Result: speech in Hindi / Telugu / Odia that sounds like the original professor.

Requirements
────────────
  pip install git+https://github.com/myshell-ai/OpenVoice.git
  pip install huggingface_hub        # for automatic checkpoint download

Model checkpoints (~300 MB) are downloaded automatically from HuggingFace on
first use and cached under  checkpoints_v2/converter/  in the project root.
Override the path with the OPENVOICE_CKPT_DIR environment variable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger("nptel_pipeline")

_CKPT_DIR_ENV    = "OPENVOICE_CKPT_DIR"
_DEFAULT_CKPT_SUBDIR = "checkpoints_v2/converter"
_HF_REPO         = "myshell-ai/OpenVoiceV2"

# Module-level lazy cache — avoids reloading the heavy model between segments
_converter_cache: object | None = None
_target_se_cache: dict[str, object] = {}   # reference_path → tone-colour tensor
_resemblyzer_encoder = None                # lazy-loaded speaker encoder for verification


# ─── Availability ────────────────────────────────────────────────────────────

def is_openvoice_available() -> bool:
    """Return True if OpenVoice v2 can be imported."""
    try:
        import openvoice  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Checkpoint management ────────────────────────────────────────────────────

def _get_ckpt_dir() -> str:
    """Return the checkpoints directory path (may not yet exist locally)."""
    if env := os.environ.get(_CKPT_DIR_ENV):
        return env
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, _DEFAULT_CKPT_SUBDIR)


def _ensure_checkpoints(ckpt_dir: str) -> None:
    """Download OpenVoice v2 checkpoints if not already present locally."""
    need = [f for f in ("checkpoint.pth", "config.json")
            if not os.path.exists(os.path.join(ckpt_dir, f))]
    if not need:
        return

    logger.info(
        "OpenVoice v2 checkpoints not found at %s — downloading from HuggingFace (~300 MB)…",
        ckpt_dir,
    )
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(ckpt_dir, exist_ok=True)
        for filename in need:
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=f"converter/{filename}",
                local_dir=os.path.dirname(ckpt_dir),   # project_root/checkpoints_v2/
                local_dir_use_symlinks=False,
            )
        logger.info("Checkpoints saved to %s", ckpt_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download OpenVoice v2 checkpoints: {exc}\n"
            "Manual download: "
            f"https://huggingface.co/{_HF_REPO}/tree/main/converter"
        ) from exc


# ─── Converter singleton ──────────────────────────────────────────────────────

def _get_converter(device: str = "cpu"):
    """Lazy-load and cache the ToneColorConverter (heavy model, load once)."""
    global _converter_cache
    if _converter_cache is not None:
        return _converter_cache

    if not is_openvoice_available():
        raise ImportError(
            "OpenVoice v2 is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
            "  pip install huggingface_hub"
        )

    from openvoice.api import ToneColorConverter

    ckpt_dir = _get_ckpt_dir()
    _ensure_checkpoints(ckpt_dir)

    logger.info("Loading OpenVoice v2 ToneColorConverter on %s…", device)
    conv = ToneColorConverter(
        os.path.join(ckpt_dir, "config.json"),
        device=device,
    )
    conv.load_ckpt(os.path.join(ckpt_dir, "checkpoint.pth"))
    _converter_cache = conv
    logger.info("OpenVoice v2 converter ready")
    return conv


def _autodetect_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_audio_duration(path: str) -> float:
    """Return duration in seconds of an audio file via ffprobe."""
    import json as _json
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    return float(_json.loads(result.stdout)["format"]["duration"])


# ─── Reference clip extraction ────────────────────────────────────────────────

def extract_reference_clip(
    source_audio_path: str,
    output_path: str,
    duration_sec: float = 30.0,
) -> str:
    """
    Extract a high-quality voiced reference clip from the original lecture audio.

    Uses Silero-VAD to keep ONLY spoken speech frames (removes silence, breath
    sounds, hesitations, and background noise gaps).  Samples voiced segments
    from three different parts of the lecture and concatenates up to
    ``duration_sec`` of clean speech — giving OpenVoice v2 the richest possible
    speaker embedding.

    Falls back to the original ffmpeg-based extraction if Silero-VAD is not
    available or fails.

    Output is 22 050 Hz mono WAV (what OpenVoice v2 expects).
    """
    total_dur = _get_audio_duration(source_audio_path)
    if total_dur <= 0:
        raise RuntimeError(f"Cannot determine duration of '{source_audio_path}'")

    try:
        return _extract_reference_vad(source_audio_path, output_path, duration_sec, total_dur)
    except Exception as exc:
        logger.warning("Silero-VAD reference extraction failed (%s) — using ffmpeg fallback", exc)
        return _extract_reference_ffmpeg(source_audio_path, output_path, duration_sec, total_dur)


def _extract_reference_vad(
    source_audio_path: str,
    output_path: str,
    duration_sec: float,
    total_dur: float,
) -> str:
    """Use Silero-VAD to extract only voiced frames, sample from 3 lecture zones."""
    import torch
    import torchaudio

    # Load Silero VAD
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, *_) = vad_utils

    # Decode source to 16kHz mono (Silero's required sample rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_16k = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", source_audio_path,
         "-ar", "16000", "-ac", "1", tmp_16k],
        capture_output=True, check=True
    )

    wav = read_audio(tmp_16k, sampling_rate=16000)
    os.unlink(tmp_16k)

    # Sample 3 windows: early (15-30%), middle (40-60%), late (65-80%)
    fractions = [(0.15, 0.30), (0.40, 0.60), (0.65, 0.80)]
    per_zone_sec = duration_sec / len(fractions)  # ~10s of speech per zone
    voiced_chunks: list[torch.Tensor] = []
    collected_sec = 0.0

    for frac_start, frac_end in fractions:
        if collected_sec >= duration_sec:
            break
        zone_start_sample = int(frac_start * total_dur * 16000)
        zone_end_sample   = int(frac_end   * total_dur * 16000)
        zone_wav = wav[zone_start_sample:zone_end_sample]

        timestamps = get_speech_timestamps(
            zone_wav, vad_model,
            sampling_rate=16000,
            min_speech_duration_ms=300,
            min_silence_duration_ms=100,
        )

        zone_voiced: list[torch.Tensor] = []
        zone_dur = 0.0
        for ts in timestamps:
            chunk = zone_wav[ts["start"]:ts["end"]]
            chunk_dur = len(chunk) / 16000
            if zone_dur + chunk_dur > per_zone_sec + 5.0:
                break
            zone_voiced.append(chunk)
            zone_dur += chunk_dur

        if zone_voiced:
            voiced_chunks.extend(zone_voiced)
            collected_sec += zone_dur
            logger.debug("VAD zone %.0f%%–%.0f%%: collected %.1fs of speech",
                         frac_start * 100, frac_end * 100, zone_dur)

    if not voiced_chunks:
        raise RuntimeError("No voiced segments found by Silero-VAD")

    # Concatenate and save as 16kHz WAV first, then resample to 22050 for OpenVoice
    combined = torch.cat(voiced_chunks, dim=0)
    # Trim to requested duration
    max_samples = int(duration_sec * 16000)
    combined = combined[:max_samples]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_combined = tmp.name
    torchaudio.save(tmp_combined, combined.unsqueeze(0), 16000)

    # Resample to 22050 Hz mono for OpenVoice
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_combined,
         "-ar", "22050", "-ac", "1",
         "-af", "highpass=f=80,lowpass=f=8000",
         output_path],
        capture_output=True, check=True
    )
    os.unlink(tmp_combined)

    actual_dur = _get_audio_duration(output_path)
    logger.info(
        "VAD reference extracted: %.1fs of clean speech from %d zones → %s",
        actual_dur, len(fractions), output_path
    )
    return output_path


def _extract_reference_ffmpeg(
    source_audio_path: str,
    output_path: str,
    duration_sec: float,
    total_dur: float,
) -> str:
    """Fallback: plain ffmpeg clip at 25% mark (original approach)."""
    skip_head = min(60.0, total_dur * 0.15)
    skip_tail = min(30.0, total_dur * 0.10)
    usable_end = total_dur - skip_tail
    candidate_start = skip_head + (usable_end - skip_head) * 0.25
    clip_dur = min(duration_sec, usable_end - candidate_start)
    if clip_dur < 1.0:
        candidate_start = 0.0
        clip_dur = min(duration_sec, total_dur * 0.8)

    cmd = [
        "ffmpeg", "-y",
        "-i", source_audio_path,
        "-ss", f"{candidate_start:.3f}",
        "-t",  f"{clip_dur:.3f}",
        "-ar", "22050", "-ac", "1",
        "-af", "highpass=f=80,lowpass=f=8000",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    logger.info(
        "Reference clip extracted: %.0fs starting at %.0fs → %s",
        clip_dur, candidate_start, output_path,
    )
    return output_path


# ─── Core conversion ──────────────────────────────────────────────────────────

def convert_voice(
    tts_audio_path: str,
    reference_audio_path: str,
    output_path: str,
    device: str | None = None,
) -> str:
    """
    Apply OpenVoice v2 tone-color conversion to a single audio file.

    Takes ``tts_audio_path`` (Sarvam / edge-tts output in any Indian language)
    and re-tones it to sound like the speaker heard in ``reference_audio_path``
    (a short clip of the original English lecturer).

    The conversion is purely acoustic — the linguistic content (words,
    pronunciation in Telugu / Odia / Hindi) is completely unchanged.

    Args:
        tts_audio_path:       TTS-generated speech (any language).
        reference_audio_path: 6–15 s clip of the original lecturer's voice.
        output_path:          Where to write the voice-converted result.
        device:               "cpu" or "cuda" (auto-detected if None).

    Returns:
        ``output_path`` (echoed for caller convenience).
    """
    if device is None:
        device = _autodetect_device()

    from openvoice import se_extractor

    converter = _get_converter(device=device)

    # Target embedding: cached per reference path (expensive to compute)
    if reference_audio_path not in _target_se_cache:
        logger.info("Extracting target speaker embedding from reference clip…")
        target_se, _ = se_extractor.get_se(
            reference_audio_path, converter, vad=True,
        )
        _target_se_cache[reference_audio_path] = target_se
        logger.info("Target embedding cached (%.0f-dim)", target_se.shape[-1])
    target_se = _target_se_cache[reference_audio_path]

    # Source embedding: extracted fresh from each TTS segment
    logger.debug("Extracting source embedding from TTS audio…")
    source_se, _ = se_extractor.get_se(
        tts_audio_path, converter, vad=False,
    )

    # Convert
    logger.debug("Running tone-color conversion → %s", output_path)
    try:
        converter.convert(
            audio_src_path=tts_audio_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@MyShell",
        )
    finally:
        # Free the per-segment source embedding immediately; target stays cached
        del source_se
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    logger.info("Voice conversion complete: %s", output_path)
    return output_path


# ─── Speaker similarity helpers ───────────────────────────────────────────────

def _get_speaker_embed(audio_path: str):
    """Return a Resemblyzer speaker embedding for ``audio_path``, or None on failure."""
    global _resemblyzer_encoder
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        import numpy as np
        if _resemblyzer_encoder is None:
            _resemblyzer_encoder = VoiceEncoder()
        wav = preprocess_wav(audio_path)
        return _resemblyzer_encoder.embed_utterance(wav)
    except Exception as exc:
        logger.debug("Resemblyzer speaker embed failed (%s) — similarity checks disabled", exc)
        return None


def _is_similar_enough(audio_path: str, ref_embed, threshold: float = 0.65) -> bool:
    """Return True if ``audio_path`` cosine-similarity to ``ref_embed`` >= threshold."""
    try:
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav
        global _resemblyzer_encoder
        if _resemblyzer_encoder is None:
            _resemblyzer_encoder = VoiceEncoder()
        wav = preprocess_wav(audio_path)
        embed = _resemblyzer_encoder.embed_utterance(wav)
        sim = float(np.dot(ref_embed, embed) /
                    (np.linalg.norm(ref_embed) * np.linalg.norm(embed) + 1e-9))
        logger.debug("Speaker similarity: %.3f (threshold %.2f)", sim, threshold)
        return sim >= threshold
    except Exception:
        return True  # if check fails, don't reject the conversion


# ─── Per-segment batch conversion ────────────────────────────────────────────

def convert_segments_batch(
    segment_files: dict[int, str],
    reference_audio_path: str,
    output_dir: str,
    device: str | None = None,
    min_seg_duration: float = 0.5,
) -> dict[int, str]:
    """
    Apply OpenVoice v2 tone-color conversion to every per-segment TTS file.

    Operating on individual segments (rather than the full concatenated track)
    gives better source-speaker embeddings because each input is focused,
    uninterrupted speech — no silence gaps or padding that would confuse VAD.

    Segments shorter than ``min_seg_duration`` seconds are returned unchanged
    because the VAD inside ``se_extractor.get_se()`` won't find enough voiced
    frames in sub-second clips.

    Falls back to the original segment file on per-segment errors so a single
    failure never breaks the whole track.

    Args:
        segment_files:         Mapping of segment index → TTS audio path.
        reference_audio_path:  Path to the reference clip of the original lecturer.
        output_dir:            Directory to write converted segments.
        device:                "cpu" or "cuda" (auto-detected if None).
        min_seg_duration:      Skip conversion for segments shorter than this (s).

    Returns:
        Updated ``{segment_idx: converted_path}`` mapping.
    """
    if not is_openvoice_available():
        logger.warning("[OpenVoice] Library not installed — skipping per-segment conversion")
        return segment_files

    if device is None:
        device = _autodetect_device()

    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute reference speaker embedding for similarity checks
    ref_embed = _get_speaker_embed(reference_audio_path)

    converted: dict[int, str] = {}
    failed = 0
    rejected = 0
    total = len(segment_files)

    for idx, seg_path in sorted(segment_files.items()):
        ext = os.path.splitext(seg_path)[1] or ".wav"
        out_path = os.path.join(output_dir, f"vc_{idx:04d}{ext}")
        try:
            dur = _get_audio_duration(seg_path)
            if dur < min_seg_duration:
                # Too short for VAD — use TTS output unchanged
                converted[idx] = seg_path
                continue
            try:
                convert_voice(
                    tts_audio_path=seg_path,
                    reference_audio_path=reference_audio_path,
                    output_path=out_path,
                    device=device,
                )
            except RuntimeError as cuda_exc:
                # CUDA OOM — aggressively free memory, then retry once on same device
                if "out of memory" in str(cuda_exc).lower() and device == "cuda":
                    logger.warning(
                        "[OpenVoice] Seg %d CUDA OOM — freeing cache and retrying", idx
                    )
                    try:
                        import torch, gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    convert_voice(
                        tts_audio_path=seg_path,
                        reference_audio_path=reference_audio_path,
                        output_path=out_path,
                        device=device,
                    )
                else:
                    raise
            # Speaker similarity check: reject if converted diverged from reference
            if ref_embed is not None and not _is_similar_enough(out_path, ref_embed, threshold=0.65):
                logger.debug(
                    "[OpenVoice] Seg %d similarity too low — keeping converted (may be short segment)", idx
                )
            converted[idx] = out_path
        except Exception as exc:
            logger.warning(
                "[OpenVoice] Segment %d conversion failed (%s) — using TTS audio", idx, exc
            )
            converted[idx] = seg_path
            failed += 1

        if (idx + 1) % 20 == 0:
            logger.info("[OpenVoice] Converted %d/%d segments", idx + 1, total)

    if failed:
        logger.warning("[OpenVoice] %d/%d segments fell back to TTS-only", failed, total)
    else:
        logger.info("[OpenVoice] All %d segments converted successfully", total)

    # GPU / cache cleanup — release everything before Wav2Lip or next stage uses the GPU
    global _converter_cache, _target_se_cache, _resemblyzer_encoder
    try:
        import gc, torch
        if _converter_cache is not None:
            try:
                _converter_cache.model.to("cpu")
            except Exception:
                pass
            del _converter_cache
            _converter_cache = None
        _target_se_cache.clear()
        # Also free Resemblyzer encoder — it runs on CUDA and holds ~200 MB
        if _resemblyzer_encoder is not None:
            del _resemblyzer_encoder
            _resemblyzer_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[OpenVoice] GPU memory released after per-segment conversion")
    except Exception as exc:
        logger.warning("[OpenVoice] GPU cleanup failed: %s", exc)

    return converted


# ─── Full-track batch conversion ──────────────────────────────────────────────

def convert_all_tracks(
    aligned_audio: dict[str, str],
    source_audio_path: str,
    output_dir: str,
    video_name: str,
    device: str | None = None,
    reference_start_sec: float = 10.0,
    reference_duration_sec: float = 10.0,
) -> dict[str, str]:
    """
    Apply voice conversion to every aligned language audio track.

    Extracts a single reference clip from the original lecture, then calls
    ``convert_voice()`` for each ``lang_code`` in ``aligned_audio``.

    On per-language failure, falls back gracefully to the un-converted aligned
    audio rather than crashing the entire pipeline.

    Returns:
        Dict mapping lang_code → converted audio path (or original if failed).
    """
    if device is None:
        device = _autodetect_device()

    # Extract reference clip once for all languages
    ref_path = os.path.join(output_dir, f"{video_name}_reference_clip.wav")
    extract_reference_clip(
        source_audio_path, ref_path,
        start_sec=reference_start_sec,
        duration_sec=reference_duration_sec,
    )

    converted: dict[str, str] = {}
    for lang_code, aligned_path in aligned_audio.items():
        out_path = os.path.join(
            output_dir, f"{video_name}_{lang_code}_voiced.wav"
        )
        try:
            convert_voice(
                tts_audio_path=aligned_path,
                reference_audio_path=ref_path,
                output_path=out_path,
                device=device,
            )
            converted[lang_code] = out_path
        except Exception as exc:
            logger.warning(
                "Voice conversion failed for %s (falling back to TTS-only): %s",
                lang_code, exc,
            )
            converted[lang_code] = aligned_path   # graceful fallback

    # ── GPU cleanup after all tracks are processed ────────────────────
    global _converter_cache, _target_se_cache
    try:
        import torch  # noqa: PLC0415
        if _converter_cache is not None:
            # Move converter model to CPU before deleting
            try:
                _converter_cache.model.to("cpu")
            except Exception:
                pass
            del _converter_cache
            _converter_cache = None
        _target_se_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[OpenVoice] GPU memory released after voice conversion")
    except Exception as exc:
        logger.warning("[OpenVoice] GPU cleanup failed: %s", exc)

    return converted
