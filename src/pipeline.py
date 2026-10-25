"""
pipeline.py
───────────
Main orchestrator – chains every stage of the pipeline together.

Flow:
  Video → Preflight → Audio extraction → Voice analysis
       → STT (English) → Translation (hi/te/od)
       → Subtitle generation (en + 3 langs)
       → Per-segment TTS → Temporal alignment → MKV muxing

Supports:
  • Intermediate caching — resume from last successful step.
  • Progress callbacks    — for Streamlit / CLI progress bars.
  • Glossary / code-mixing and filler preservation integration.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import subprocess
from typing import Callable, Protocol

from config.settings import (
    OUTPUT_DIR,
    TARGET_LANGUAGES,
    ENABLE_CACHE,
    CACHE_DIR_NAME,
    ENABLE_CODE_MIXING,
    ENABLE_VOICE_PRESERVATION,
    VOICE_PITCH_SHIFT,
    PRESERVE_FILLERS,
    get_default_engine,
    DEFAULT_TARGET_LANGS,
    DEFAULT_TTS_ENGINE,
    SEPARATE_MUSIC,
    SARVAM_TRANSLATION_MODE,
)
from src.audio_extractor import extract_audio
from src.transcriber import transcribe_audio, split_segments_at_silences, Segment
from src.translator import translate_segments
from src.subtitle_generator import generate_all_subtitles
from src.tts_generator import synthesize_all_segments
from config.settings import ENABLE_VOICE_CLONING as _DEFAULT_VOICE_CLONING
from src.audio_aligner import align_dubbed_audio
from src.video_muxer import mux_to_mkv
from src.utils import file_sha256, run_preflight_checks, setup_logging
from src.glossary import generate_glossary_from_transcript, load_default_glossary, Glossary
from src.voice_analyzer import (
    analyze_voice, apply_voice_profile_to_audio, pitch_shift_to_match, VoiceProfile,
)
from src.filler_detector import build_filler_map_for_alignment
from src.voice_converter import (
    NO_GPU_MESSAGE, convert_all_tracks, convert_segments_batch,
    extract_reference_clip, is_gpu_available, is_openvoice_available,
)
from src.audio_separator import (
    separate_audio, mix_all_tracks, is_demucs_available,
)
from src.lip_sync import LipSyncProcessor, is_wav2lip_available

logger = logging.getLogger("nptel_pipeline")


# ── Progress callback protocol ───────────────────────────────────────────────

class ProgressCallback(Protocol):
    """Callable that receives (step_number, total_steps, message)."""

    def __call__(self, step: int, total: int, message: str) -> None: ...


def _noop_progress(step: int, total: int, message: str) -> None:
    """Default progress callback that just logs."""
    logger.info("Step %d/%d: %s", step, total, message)


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_dir(output_dir: str) -> str:
    """Return the cache sub-directory for intermediate artefacts."""
    d = os.path.join(output_dir, CACHE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _fingerprint(*parts: object) -> str:
    """Stable hash of everything a cached artefact was computed from."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _save_cache(output_dir: str, key: str, data: object, fingerprint: str) -> None:
    """Persist a JSON-serialisable object to the run cache, tagged with its inputs."""
    if not ENABLE_CACHE:
        return
    path = os.path.join(_cache_dir(output_dir), f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "data": data}, f, ensure_ascii=False, indent=2)
    logger.debug("Cache saved: %s", path)


def _load_cache(output_dir: str, key: str, fingerprint: str) -> object | None:
    """
    Load a cached artefact if caching is enabled and it was computed from the
    same inputs (*fingerprint*).  A cache left by a different video, engine or
    setting under the same job name is ignored.
    """
    if not ENABLE_CACHE:
        return None
    path = os.path.join(_cache_dir(output_dir), f"{key}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        logger.info("Cache '%s' is from a different video or settings — recomputing", key)
        return None
    logger.info("Resuming from cache: %s", key)
    return payload["data"]


def _segments_from_dicts(dicts: list[dict]) -> list[Segment]:
    """Reconstruct Segment objects from cached dictionaries."""
    return [Segment.from_dict(d) for d in dicts]


# ── Media duration helper ────────────────────────────────────────────────────

def _get_media_duration(path: str) -> float:
    """Return duration in seconds of a media file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_dir: str | None = None,
    stt_method: str | None = None,
    translate_method: str | None = None,
    do_tts: bool = True,
    tts_engine: str | None = None,
    target_langs: list[str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    enable_voice_cloning: bool = _DEFAULT_VOICE_CLONING,
    separate_music: bool = SEPARATE_MUSIC,
    voice_profile: object | None = None,  # Legacy
    enable_prosody: bool = False,
    enable_glossary: bool = True,
    enable_enhancer: bool = False,
    enable_lip_sync: bool = False,
) -> dict:
    """
    Execute the full lecture-translation pipeline.

    Args:
        video_path:        Path to the input video.
        output_dir:        Where to store all outputs.
        stt_method:        Engine id for transcription (auto-detected if None).
        translate_method:  Engine id for translation  (auto-detected if None).
        do_tts:            Whether to generate translated audio.
        tts_engine:        TTS engine id ("edge_tts" or "sarvam").
        target_langs:      List of language codes to translate into.
        progress:          Optional progress callback (step, total, message).

    Returns:
        Dict with paths to all generated artefacts.
    """
    if stt_method is None:
        stt_method = get_default_engine(capability="stt")
    if translate_method is None:
        translate_method = get_default_engine(capability="translate")
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if target_langs is None:
        target_langs = list(DEFAULT_TARGET_LANGS)
    if tts_engine is None:
        tts_engine = DEFAULT_TTS_ENGINE
    if progress is None:
        progress = _noop_progress

    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    artefacts: dict = {"video": video_path}
    total_steps = 7 + (1 if separate_music else 0) + (1 if enable_lip_sync and do_tts else 0)

    # ── Preflight checks ─────────────────────────────────────
    progress(0, total_steps, "Running preflight checks…")
    run_preflight_checks()
    logger.info("Preflight checks passed")

    # ── Step 1: Extract audio ────────────────────────────────
    progress(1, total_steps, "Extracting audio from video")
    audio_path = extract_audio(video_path, output_dir)
    artefacts["audio"] = audio_path

    # ── Step 1b: Music separation (optional) ─────────────────
    accompaniment_path: str | None = None
    vocals_path: str | None = None
    if separate_music:
        if not is_demucs_available():
            logger.warning(
                "Music separation requested but Demucs is not installed. "
                "Falling back to original audio. Install with: pip install demucs"
            )
        else:
            progress(1, total_steps, "Separating vocals from background music (Demucs)…")
            try:
                vocals_path, accompaniment_path = separate_audio(audio_path, output_dir)
                logger.info(
                    "Music separation complete: vocals=%s, accompaniment=%s",
                    vocals_path, accompaniment_path,
                )
                artefacts["vocals"] = vocals_path
                artefacts["accompaniment"] = accompaniment_path
            except Exception as exc:
                logger.error("Music separation failed: %s", exc)
                # Fall back gracefully
                vocals_path = None
                accompaniment_path = None

    # STT audio: use isolated vocals if separation succeeded
    stt_audio_path = vocals_path if vocals_path else audio_path

    # ── Voice analysis (optional) ────────────────────────────
    voice_profile: VoiceProfile | None = None
    if ENABLE_VOICE_PRESERVATION and do_tts:
        progress(1, total_steps, "Analysing speaker voice characteristics")
        try:
            # Isolated vocals when available — background music skews the pitch
            voice_profile = analyze_voice(stt_audio_path)
            logger.info("Voice profile: rate=%.2f wps, pitch=%.0fHz",
                        voice_profile.speaking_rate_wps, voice_profile.avg_pitch_hz)
        except Exception as exc:
            logger.warning("Voice analysis failed: %s", exc)

    # ── Step 2: Transcribe (English STT) ─────────────────────
    progress(2, total_steps, "Transcribing English audio")
    # Caches are tagged with what produced them, so a different video under
    # the same job name — or a different engine/setting — is never reused.
    en_fp = _fingerprint("english", file_sha256(video_path), stt_method, separate_music)
    cached_en = _load_cache(output_dir, "english_segments", en_fp)
    if cached_en is not None:
        english_segments = _segments_from_dicts(cached_en)
        logger.info("Loaded %d cached English segments", len(english_segments))
    else:
        try:
            english_segments = transcribe_audio(stt_audio_path, method=stt_method)
        except ValueError as _ve:
            _msg = str(_ve).lower()
            _no_speech = "no speech" in _msg or "empty transcript" in _msg
            # ── Auto-fallback: if no speech detected AND we have not already
            # separated music, try Demucs automatically before giving up ──
            if _no_speech and not separate_music and is_demucs_available():
                logger.warning(
                    "No speech detected in original audio — "
                    "auto-retrying with Demucs music/vocal separation…"
                )
                progress(2, total_steps,
                         "No speech found — auto-separating vocals from music…")
                try:
                    vocals_path, accompaniment_path = separate_audio(
                        audio_path, output_dir
                    )
                    artefacts["vocals"] = vocals_path
                    artefacts["accompaniment"] = accompaniment_path
                    stt_audio_path = vocals_path
                    english_segments = transcribe_audio(
                        stt_audio_path, method=stt_method
                    )
                    logger.info(
                        "Auto-separation succeeded: %d segments transcribed",
                        len(english_segments),
                    )
                except Exception as _sep_err:
                    raise ValueError(
                        "No speech detected in the audio, and automatic music "
                        "separation also failed.\n"
                        f"STT error: {_ve}\n"
                        f"Demucs error: {_sep_err}"
                    ) from _sep_err
            else:
                raise
        # ── Post-transcription: split large segments at internal silences ──
        # Only needed for Gemini/Sarvam which produce sentence-level timestamps
        # with potentially huge windows (e.g. 45s containing 15 words + pauses).
        # Whisper already handles this via word-level timestamps + VAD filter —
        # running the splitter on Whisper output would corrupt proportional text.
        if stt_method != "whisper":
            english_segments = split_segments_at_silences(
                english_segments, stt_audio_path
            )
        _save_cache(output_dir, "english_segments",
                    [s.to_dict() for s in english_segments], en_fp)
    artefacts["english_segments"] = [s.to_dict() for s in english_segments]

    # ── Step 3: Translate ──────────────────────────────────
    progress(3, total_steps, "Translating segments…")
    tr_fp = _fingerprint(
        "translated", artefacts["english_segments"],
        translate_method, SARVAM_TRANSLATION_MODE, enable_glossary,
    )
    cached_trans = _load_cache(output_dir, "translated_segments", tr_fp)
    if cached_trans is not None:
        # data is {lang: [dict, ...]}
        translated = {
            l: _segments_from_dicts(segs) for l, segs in cached_trans.items()
            if l in target_langs
        }
        logger.info("Loaded cached translations for %d languages (filtered by target_langs)", len(translated))
        
        # Check if we need to translate more
        missing = [l for l in target_langs if l not in translated]
        if missing:
            logger.info("Translating missing languages: %s", missing)
            # Re-run translation for missing
            from config.settings import ENABLE_CODE_MIXING
            import config.settings
            original_glossary_setting = config.settings.ENABLE_CODE_MIXING
            config.settings.ENABLE_CODE_MIXING = enable_glossary
            try:
                new_trans = translate_segments(english_segments, missing, method=translate_method)
                translated.update(new_trans)
                # Update cache
                cached_trans.update({l: [s.__dict__ for s in segs] for l, segs in new_trans.items()})
                _save_cache(output_dir, "translated_segments", cached_trans, tr_fp)
                config.settings.ENABLE_CODE_MIXING = original_glossary_setting
            except Exception as exc:
                config.settings.ENABLE_CODE_MIXING = original_glossary_setting
                raise exc
    else:
        # Override glossary based on UI toggle
        from config.settings import ENABLE_CODE_MIXING
        import config.settings
        original_glossary_setting = config.settings.ENABLE_CODE_MIXING
        config.settings.ENABLE_CODE_MIXING = enable_glossary
        
        try:
            translated = translate_segments(
                english_segments, target_langs, method=translate_method,
            )
            # Restore setting
            config.settings.ENABLE_CODE_MIXING = original_glossary_setting
        except Exception as exc:
            config.settings.ENABLE_CODE_MIXING = original_glossary_setting
            raise exc

        _save_cache(output_dir, "translated_segments", {
            l: [s.__dict__ for s in segs] for l, segs in translated.items()
        }, tr_fp)
    artefacts["translated"] = translated

    # ── Step 4: Generate subtitles (.srt) ────────────────────
    progress(4, total_steps, "Generating subtitle files")
    video_duration = _get_media_duration(video_path)
    logger.info("Master duration (from video): %.3fs", video_duration)

    subtitle_paths = generate_all_subtitles(
        english_segments, translated, video_name, output_dir,
        total_duration=video_duration,
    )
    artefacts["subtitles"] = subtitle_paths

    # ── Steps 5–7: TTS → Align → Mux ────────────────────────
    if do_tts:
        # Step 5: Per-segment TTS synthesis (duration-aware)
        progress(5, total_steps, "Synthesising per-segment TTS audio")
        per_seg_files = synthesize_all_segments(
            translated, output_dir, engine=tts_engine,
            voice_profile=voice_profile,
            preserve_fillers=PRESERVE_FILLERS,
            original_segments=english_segments,
            voice_reference_audio=audio_path,
        )

        # ── Stage 2: Per-segment voice cloning (OpenVoice v2) ────────
        # Convert each TTS segment to match the original lecturer's voice
        # BEFORE alignment so that rubber-band stretches the already-converted
        # audio.  Operating on short isolated segments gives the speaker
        # embedding extractor clean voiced input → better clone quality.
        #
        # Skip when the chosen TTS engine already handles voice cloning
        # internally (e.g. "sarvam_vc"), to avoid double-conversion.
        vc_handled_by_engine = tts_engine in ("sarvam_vc", "gcptts_vc")
        if (enable_voice_cloning or vc_handled_by_engine) and not is_gpu_available():
            # The _vc engines also skip cloning on their own; record it so the
            # UI can tell the user why the dub uses the plain TTS voice.
            logger.warning(NO_GPU_MESSAGE)
            artefacts["voice_cloned"] = False
            artefacts["voice_cloning_note"] = NO_GPU_MESSAGE
        elif enable_voice_cloning and not vc_handled_by_engine:
            if not is_openvoice_available():
                logger.warning(
                    "Voice cloning requested but OpenVoice v2 is not installed. "
                    "Install it with: pip install git+https://github.com/myshell-ai/OpenVoice.git"
                )
            elif per_seg_files:
                progress(5, total_steps, "Stage 2: Per-segment voice cloning (OpenVoice v2)…")
                ref_path = os.path.join(output_dir, f"{video_name}_reference_clip.wav")
                try:
                    extract_reference_clip(audio_path, ref_path)

                    vc_per_seg: dict[str, dict[int, str]] = {}
                    for lang_code, seg_files in per_seg_files.items():
                        vc_dir = os.path.join(output_dir, f"vc_segments_{lang_code}")
                        os.makedirs(vc_dir, exist_ok=True)
                        vc_per_seg[lang_code] = convert_segments_batch(
                            seg_files, ref_path, vc_dir, device="cuda",
                        )
                    per_seg_files = vc_per_seg
                    artefacts["voice_cloned"] = True
                    logger.info("Stage 2 complete: per-segment voice cloning done")
                except Exception as exc:
                    logger.error("Stage 2 (per-segment voice cloning) failed: %s — continuing without", exc)
                    artefacts["voice_cloned"] = False
        elif vc_handled_by_engine:
            artefacts["voice_cloned"] = True
            logger.info("Voice cloning handled by '%s' engine — skipping separate Stage 2", tts_engine)

        # Step 6: Temporal alignment
        progress(6, total_steps, "Aligning dubbed audio to original timing")
        total_dur = video_duration
        aligned_audio: dict[str, str] = {}

        for lang_code, seg_files in per_seg_files.items():
            lang_name = TARGET_LANGUAGES[lang_code]["name"]
            out_path = os.path.join(
                output_dir, f"{video_name}_{lang_code}_aligned.mp3"
            )
            logger.info("Aligning %s…", lang_name)
            try:
                align_dubbed_audio(
                    english_segments, seg_files, total_dur, out_path,
                    source_audio_path=stt_audio_path,
                )
                
                # --- Advanced Enhancement: Prosody Transfer ---
                if enable_prosody:
                    from src.prosody_transfer import apply_prosody_transfer, is_parselmouth_available
                    if is_parselmouth_available():
                        progress(6, total_steps, f"Fine-tuning prosody for {lang_name}…")
                        prosody_path = out_path.replace(".mp3", "_prosody.mp3")
                        try:
                            out_path = apply_prosody_transfer(
                                audio_path, out_path, prosody_path,
                                voice_profile=voice_profile,
                            )
                        except Exception as e:
                            logger.error("Prosody transfer failed: %s", e)
                    else:
                        logger.warning("Prosody requested but Parselmouth not installed.")

                # --- Advanced Enhancement: Neural Enhancer ---
                if enable_enhancer:
                    from src.audio_enhancer import apply_neural_enhancement
                    progress(6, total_steps, f"Enhancing audio quality for {lang_name}…")
                    try:
                        enhanced_path = apply_neural_enhancement(out_path)
                        out_path = enhanced_path
                    except Exception as e:
                        logger.error("Neural enhancement failed: %s", e)

                aligned_audio[lang_code] = out_path
                
                # Post-alignment pitch shift to match original speaker
                if voice_profile and VOICE_PITCH_SHIFT and tts_engine != "xtts":
                    pitched_path = os.path.join(
                        output_dir,
                        f"{video_name}_{lang_code}_aligned_pitched.mp3",
                    )
                    try:
                        # Measured from the dubbed track itself, so it is right
                        # whichever voice (male/female/cloned) was used
                        shift = pitch_shift_to_match(voice_profile, out_path)
                        apply_voice_profile_to_audio(
                            out_path, voice_profile, pitched_path, semitones=shift,
                        )
                        out_path = pitched_path
                        logger.info("%s: applied pitch shift (%.1f semitones)",
                                    lang_name, shift)
                    except Exception as pexc:
                        logger.warning("Pitch shift failed for %s: %s", lang_name, pexc)
                aligned_audio[lang_code] = out_path
            except Exception as exc:
                logger.error("Alignment failed for %s: %s", lang_name, exc)

        artefacts["aligned_audio"] = aligned_audio



        # ── Music mix: dub vocals + original accompaniment ────
        if separate_music and accompaniment_path:
            progress(total_steps - (2 if enable_voice_cloning else 1),
                     total_steps,
                     "Mixing dubbed audio with original background music…")
            try:
                mixed_audio = mix_all_tracks(
                    aligned_audio=aligned_audio,
                    accompaniment_path=accompaniment_path,
                    output_dir=output_dir,
                    video_name=video_name,
                )
                aligned_audio = mixed_audio
                artefacts["aligned_audio"] = aligned_audio
                logger.info("Music mix complete")
            except Exception as exc:
                logger.error("Music mix failed: %s", exc)

        # ── Step 7: Lip synchronisation (before mux so MKV gets synced video) ──
        lip_synced_videos: dict[str, str] = {}
        lip_sync_errors: dict[str, str] = {}
        if enable_lip_sync:
            lip_sync_step = 7 + (1 if separate_music else 0)
            progress(lip_sync_step, total_steps, "Applying lip synchronisation (Wav2Lip)…")
            lip_sync_dir = os.path.join(output_dir, "lipsync")
            os.makedirs(lip_sync_dir, exist_ok=True)
            lip_debug_dir = os.path.join(output_dir, "lipsync_debug")
            processor = LipSyncProcessor(debug_dir=lip_debug_dir)
            lip_sync_results = processor.batch_sync(
                video_path=video_path,
                audio_paths=aligned_audio,
                output_dir=lip_sync_dir,
            )
            for lang_code, ls_result in lip_sync_results.items():
                if ls_result.success:
                    lip_synced_videos[lang_code] = ls_result.output_path
                    mode = "passthrough" if ls_result.fallback_used else "Wav2Lip"
                    q = f", quality={ls_result.quality_score:.2f}" if ls_result.quality_score is not None else ""
                    if ls_result.fallback_used and ls_result.error_message:
                        lip_sync_errors[lang_code] = ls_result.error_message
                        logger.error("Lip sync [%s] fell back to %s. Reason: %s",
                                     lang_code, mode, ls_result.error_message)
                    else:
                        logger.info("Lip sync [%s] %s: %s%s", lang_code, mode,
                                    ls_result.output_path, q)
                else:
                    lip_sync_errors[lang_code] = ls_result.error_message or "Unknown error"
                    logger.error("Lip sync failed for '%s': %s", lang_code, ls_result.error_message)

        artefacts["lip_synced_videos"] = lip_synced_videos
        artefacts["lip_sync_errors"] = lip_sync_errors
        artefacts["lip_sync_used_wav2lip"] = any(
            not r.fallback_used for r in lip_sync_results.values() if r.success
        ) if enable_lip_sync else False

        # ── Step 8 (was 7): Mux into MKV ─────────────────────────────────
        mux_step = total_steps
        progress(mux_step, total_steps, "Muxing final MKV video")

        # Standard multi-language MKV (original video, all audio tracks)
        mkv_path = os.path.join(output_dir, f"{video_name}_dubbed.mkv")
        try:
            mux_to_mkv(
                original_video=video_path,
                audio_tracks=aligned_audio,
                subtitle_tracks=subtitle_paths,
                output_path=mkv_path,
            )
            artefacts["dubbed_video"] = mkv_path
        except Exception as exc:
            logger.error("MKV muxing failed: %s", exc)

        # Per-language lip-synced MKVs (lip-synced video + that language's audio)
        lip_synced_mkvs: dict[str, str] = {}
        for lang_code, lipsync_mp4 in lip_synced_videos.items():
            if not os.path.isfile(lipsync_mp4):
                continue
            ls_mkv = os.path.join(output_dir, f"{video_name}_lipsync_{lang_code}.mkv")
            try:
                mux_to_mkv(
                    original_video=video_path,          # original English audio source
                    video_source=lipsync_mp4,           # lip-synced video frames
                    audio_tracks={lang_code: aligned_audio[lang_code]},
                    subtitle_tracks={lang_code: subtitle_paths[lang_code]}
                        if lang_code in subtitle_paths else {},
                    output_path=ls_mkv,
                )
                lip_synced_mkvs[lang_code] = ls_mkv
                logger.info("Lip-synced MKV: %s", ls_mkv)
            except Exception as exc:
                logger.error("Lip-synced MKV muxing failed for %s: %s", lang_code, exc)

        artefacts["lip_synced_mkvs"] = lip_synced_mkvs
    else:
        logger.info("Steps 5-7 (TTS / Alignment / Muxing) skipped")

    logger.info("Pipeline complete!")
    return artefacts
