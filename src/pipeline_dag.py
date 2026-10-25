"""
pipeline_dag.py
───────────────
Breaks one dubbing job into a DAG of stages for the DAG executor.

Every stage declares the resource it needs and runs as soon as its inputs
are ready and a slot of that resource is free (see dag_executor.py):

  GPU  demucs, stt (Whisper), clone_*, enhance_*, lipsync_*, tts_* for XTTS
  API  stt (Sarvam), translate_*, tts_* (Sarvam / Google Neural2)
  CPU  extract, probe, voice_analyze, voice_ref, subtitles_*, align_*,
       pitch_*, mix_*, lipsync_mkv_*, mux

Per-language chains fan out, so Hindi, Telugu and Odia are translated,
synthesised, cloned and aligned side by side:

  probe ────────────────────────────────────────────────┐
  extract ─┬─ demucs ─┬─ voice_analyze ──────────┐      │
           │          └─ stt ─┬─ subtitles_en ───┼──────┼──────────────┐
           └─ voice_ref ──────┼──────────────┐   │      │              │
                              └─ translate_hi ┬─ subtitles_hi ────────┤
                                              └─ tts_hi ─ clone_hi ─ align_hi
                                                 ─ [enhance_hi] ─ pitch_hi ─ mix_hi ─ mux
                                                 ─ [lipsync_hi ─ lipsync_mkv_hi]
                                 (same chain for te / od)

A failure inside one language's chain skips the rest of that chain only;
the job fails when a shared stage fails or no language produced audio.

Context keys (ctx) written by the stages
────────────────────────────────────────
  video_duration, audio_path (16 kHz mono, for STT), audio_full_path
  (44.1 kHz stereo, for Demucs / voice reference), vocals_path, accompaniment_path,
  voice_profile, stt_audio_path, english_segments, voice_ref,
  translated:<lc>, subtitle:<lc> (incl. "en"), tts:<lc>, voice:<lc>,
  speech:<lc> (voice track), audio:<lc> (final track), lipsync:<lc>,
  lipsync_mkv:<lc>, dubbed_video, voice_cloning_note, warnings
"""

from __future__ import annotations

import logging
import os
import subprocess

from config.settings import (
    ENABLE_VOICE_PRESERVATION,
    PRESERVE_FILLERS,
    SARVAM_TRANSLATION_MODE,
    TARGET_LANGUAGES,
    VOICE_PITCH_SHIFT,
)
from src.audio_aligner import align_dubbed_audio
from src.audio_extractor import extract_audio
from src.audio_separator import is_demucs_available, mix_audio_tracks, separate_audio
from src.dag_executor import API, CPU, FAILED, GPU, Node, PipelineDAG
from src.filler_detector import get_text_without_fillers
from src.lip_sync import LipSyncProcessor
from src.pipeline import (
    _fingerprint, _get_media_duration, _load_cache, _save_cache, _segments_from_dicts,
)
from src.subtitle_generator import generate_srt, generate_vtt
from src.transcriber import Segment, split_segments_at_silences, transcribe_audio
from src.translator import translate_segments
from src.tts_generator import split_tts_engine, synthesize_segments
from src.utils import file_sha256
from src.video_muxer import mux_to_mkv
from src.voice_analyzer import analyze_voice, apply_voice_profile_to_audio, pitch_shift_to_match
from src.voice_converter import (
    NO_GPU_MESSAGE, convert_segments_batch_in_subprocess, extract_reference_clip,
    is_gpu_available, is_openvoice_available,
)

logger = logging.getLogger("nptel_pipeline")


def build_pipeline_dag(
    request_id: str,
    video_path: str,
    output_dir: str,
    *,
    stt_method: str,
    translate_method: str,
    target_langs: list[str],
    do_tts: bool = True,
    tts_engine: str = "sarvam_vc",
    separate_music: bool = True,
    enable_voice_cloning: bool = True,
    enable_prosody: bool = False,
    enable_enhancer: bool = False,
    enable_lip_sync: bool = False,
) -> PipelineDAG:
    """Build the stage graph for one dubbing job (nothing runs yet)."""
    dag = PipelineDAG(request_id)
    dag.ctx["warnings"] = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    synth_engine, engine_clones = split_tts_engine(tts_engine)

    def add(node_id, fn, deps=(), resource=CPU, label="", after=(), critical=True):
        dag.add(Node(node_id, fn, list(deps), resource, label or node_id,
                     list(after), critical))

    def warn(ctx, message: str) -> None:
        logger.warning(message)
        ctx["warnings"].append(message)

    # ── Shared stages ────────────────────────────────────────────────────────

    def probe(ctx):
        ctx["video_duration"] = _get_media_duration(video_path)
        logger.info("Master duration (from video): %.3fs", ctx["video_duration"])

    add("probe", probe, label="Read video duration")

    def extract(ctx):
        ctx["audio_path"] = extract_audio(video_path, output_dir)
        # Full-band copy: separating music and cloning the voice from the
        # 16 kHz STT copy capped both at 8 kHz
        ctx["audio_full_path"] = extract_audio(
            video_path, output_dir, sample_rate=44100, channels=2, suffix="_full",
        )

    add("extract", extract, label="Extract audio")

    audio_ready = "extract"
    if separate_music and is_demucs_available():
        def demucs(ctx):
            try:
                ctx["vocals_path"], ctx["accompaniment_path"] = separate_audio(
                    ctx["audio_full_path"], output_dir,
                )
            except Exception as exc:
                logger.error("Music separation failed: %s — using the original audio", exc)

        add("demucs", demucs, ["extract"], GPU, "Separate music (Demucs)")
        audio_ready = "demucs"
    elif separate_music:
        logger.warning("Music separation requested but Demucs is not installed. "
                       "Falling back to original audio. Install with: pip install demucs")

    analyze_node = None
    if ENABLE_VOICE_PRESERVATION and do_tts:
        def voice_analyze(ctx):
            # Isolated vocals when available — background music skews the pitch
            try:
                ctx["voice_profile"] = analyze_voice(ctx.get("vocals_path") or ctx["audio_path"])
            except Exception as exc:
                logger.warning("Voice analysis failed: %s", exc)

        add("voice_analyze", voice_analyze, [audio_ready], CPU, "Analyse speaker voice")
        analyze_node = "voice_analyze"

    # Transcript cache is tagged with the video contents and STT settings
    en_fp = _fingerprint("english", file_sha256(video_path), stt_method, separate_music)
    cached_en = _load_cache(output_dir, "english_segments", en_fp)

    def transcribe(ctx, stt_audio: str) -> tuple[list[Segment], str]:
        try:
            segments = transcribe_audio(stt_audio, method=stt_method)
        except ValueError as exc:
            msg = str(exc).lower()
            no_speech = "no speech" in msg or "empty transcript" in msg
            # Music can drown the speech: retry on Demucs vocals before giving up
            if not (no_speech and not separate_music and is_demucs_available()):
                raise
            logger.warning("No speech detected in original audio — "
                           "auto-retrying with Demucs music/vocal separation…")
            try:
                ctx["vocals_path"], ctx["accompaniment_path"] = separate_audio(
                    ctx["audio_full_path"], output_dir,
                )
                stt_audio = ctx["vocals_path"]
                segments = transcribe_audio(stt_audio, method=stt_method)
                logger.info("Auto-separation succeeded: %d segments transcribed", len(segments))
            except Exception as sep_err:
                raise ValueError(
                    "No speech detected in the audio, and automatic music "
                    "separation also failed.\n"
                    f"STT error: {exc}\n"
                    f"Demucs error: {sep_err}"
                ) from sep_err
        # Sarvam's sentence-level windows can span long pauses; Whisper's
        # word timestamps already avoid that
        if stt_method != "whisper":
            segments = split_segments_at_silences(segments, stt_audio)
        return segments, stt_audio

    def stt(ctx):
        stt_audio = ctx.get("vocals_path") or ctx["audio_path"]
        if cached_en is not None:
            segments = _segments_from_dicts(cached_en)
            logger.info("Loaded %d cached English segments", len(segments))
        else:
            segments, stt_audio = transcribe(ctx, stt_audio)
            _save_cache(output_dir, "english_segments",
                        [s.to_dict() for s in segments], en_fp)
        ctx["stt_audio_path"] = stt_audio
        ctx["english_segments"] = segments

    if cached_en is not None:
        stt_resource = CPU
    elif stt_method == "whisper" or (not separate_music and is_demucs_available()):
        stt_resource = GPU   # Whisper, or the no-speech Demucs retry
    else:
        stt_resource = API
    add("stt", stt, [audio_ready], stt_resource, "Speech-to-text")

    def write_subtitles(ctx, segments: list[Segment], lc: str) -> None:
        dur = ctx["video_duration"]
        ctx[f"subtitle:{lc}"] = generate_srt(segments, lc, video_name, output_dir, dur)
        generate_vtt(segments, lc, video_name, output_dir, dur)

    def subtitles_en(ctx):
        # Hesitation fillers ("ah", "um") are left out of the captions
        write_subtitles(ctx, [
            Segment(start=s.start, end=s.end, text=get_text_without_fillers(s.text))
            for s in ctx["english_segments"]
        ], "en")

    add("subtitles_en", subtitles_en, ["stt", "probe"], CPU, "English subtitles")

    # ── Voice cloning (decided up front: it needs a CUDA GPU) ────────────────
    clone = False
    if do_tts and (engine_clones or enable_voice_cloning):
        if not is_gpu_available():
            logger.warning(NO_GPU_MESSAGE)
            dag.ctx["voice_cloning_note"] = NO_GPU_MESSAGE
        elif not is_openvoice_available():
            note = "OpenVoice v2 is not installed — voice cloning is not possible"
            logger.warning("%s. Install it with: "
                           "pip install git+https://github.com/myshell-ai/OpenVoice.git", note)
            dag.ctx["voice_cloning_note"] = note
        else:
            clone = True

    if clone:
        ref_path = os.path.join(output_dir, f"{video_name}_reference_clip.wav")

        def voice_ref(ctx):
            # The lecturer's voice without music, at full bandwidth
            extract_reference_clip(ctx.get("vocals_path") or ctx["audio_full_path"], ref_path)
            ctx["voice_ref"] = ref_path

        add("voice_ref", voice_ref, [audio_ready], CPU, "Voice reference clip", critical=False)

    # ── Per-language chains ──────────────────────────────────────────────────

    def add_language(lc: str) -> None:
        lang_name = TARGET_LANGUAGES[lc]["name"]

        def translate(ctx):
            english = ctx["english_segments"]
            fp = _fingerprint("translated", [s.to_dict() for s in english],
                              translate_method, SARVAM_TRANSLATION_MODE, lc)
            cached = _load_cache(output_dir, f"translated_{lc}", fp)
            if cached is not None:
                segments = _segments_from_dicts(cached)
            else:
                segments = translate_segments(english, [lc], method=translate_method)[lc]
                _save_cache(output_dir, f"translated_{lc}",
                            [s.to_dict() for s in segments], fp)
            ctx[f"translated:{lc}"] = segments

        add(f"translate_{lc}", translate, ["stt"], API, f"Translate → {lang_name}",
            critical=False)

        def subtitles(ctx):
            write_subtitles(ctx, ctx[f"translated:{lc}"], lc)

        add(f"subtitles_{lc}", subtitles, [f"translate_{lc}", "probe"], CPU,
            f"{lang_name} subtitles", critical=False)

        if not do_tts:
            return

        def tts(ctx):
            segments = ctx[f"translated:{lc}"]
            files = synthesize_segments(
                segments, lc, output_dir, engine=synth_engine,
                voice_profile=ctx.get("voice_profile"),
                preserve_fillers=PRESERVE_FILLERS,
                original_segments=ctx["english_segments"],
                voice_reference_audio=ctx.get("vocals_path") or ctx["audio_full_path"],
            )
            # Segments that were only a filler ("ah") are silent on purpose
            spoken = [i for i, s in enumerate(segments) if s.text.strip()]
            missing = [i for i in spoken if i not in files]
            if spoken and len(missing) == len(spoken):
                raise RuntimeError(f"{tts_engine} produced no audio for {lang_name}")
            if missing:
                warn(ctx, f"{lang_name}: {len(missing)} of {len(spoken)} sentences have "
                          f"no dubbed audio (TTS failed for them)")
            ctx[f"tts:{lc}"] = files

        add(f"tts_{lc}", tts,
            [f"translate_{lc}"] + ([analyze_node] if analyze_node else []),
            GPU if synth_engine == "xtts" else API, f"TTS → {lang_name}", critical=False)

        if clone:
            def clone_voice(ctx):
                files = ctx[f"tts:{lc}"]
                vc_dir = os.path.join(output_dir, f"vc_segments_{lc}")
                os.makedirs(vc_dir, exist_ok=True)
                converted = convert_segments_batch_in_subprocess(files, ctx["voice_ref"], vc_dir)
                if files and all(converted.get(i) == p for i, p in files.items()):
                    warn(ctx, f"{lang_name}: voice cloning failed — using the plain TTS voice")
                ctx[f"voice:{lc}"] = converted

            add(f"clone_{lc}", clone_voice, [f"tts_{lc}", "voice_ref"], GPU,
                f"Clone voice → {lang_name}", critical=False)

        def align(ctx):
            # Cloned audio when cloning ran, else the TTS audio
            files = ctx.get(f"voice:{lc}") or ctx[f"tts:{lc}"]
            out = os.path.join(output_dir, f"{video_name}_{lc}_aligned.flac")
            align_dubbed_audio(ctx["english_segments"], files, ctx["video_duration"], out,
                               source_audio_path=ctx["stt_audio_path"])
            if enable_prosody:
                out = prosody(ctx, out)
            ctx[f"speech:{lc}"] = out

        add(f"align_{lc}", align, [f"tts_{lc}", "probe"], CPU, f"Align → {lang_name}",
            after=[f"clone_{lc}"] if clone else [], critical=False)
        speech_ready = [f"align_{lc}"]

        if enable_enhancer:
            def enhance(ctx):
                from src.audio_enhancer import apply_neural_enhancement
                ctx[f"speech:{lc}"] = apply_neural_enhancement(ctx[f"speech:{lc}"])

            add(f"enhance_{lc}", enhance, [f"align_{lc}"], GPU,
                f"Noise removal → {lang_name}", critical=False)

        if VOICE_PITCH_SHIFT and analyze_node and synth_engine != "xtts":
            def pitch(ctx):
                speech, profile = ctx[f"speech:{lc}"], ctx.get("voice_profile")
                if not profile:
                    return
                out = os.path.join(output_dir, f"{video_name}_{lc}_aligned_pitched.flac")
                try:
                    # Measured from the dubbed track itself, so it is right
                    # whichever voice (male/female/cloned) was used
                    shift = pitch_shift_to_match(profile, speech)
                    apply_voice_profile_to_audio(speech, profile, out, semitones=shift)
                    ctx[f"speech:{lc}"] = out
                    logger.info("%s: applied pitch shift (%.1f semitones)", lang_name, shift)
                except Exception as exc:
                    logger.warning("Pitch shift failed for %s: %s", lang_name, exc)

            # Noise removal is optional polish: wait for it, don't depend on it
            add(f"pitch_{lc}", pitch, [f"align_{lc}"], CPU, f"Match pitch → {lang_name}",
                after=[f"enhance_{lc}"] if enable_enhancer else [], critical=False)
            speech_ready = [f"pitch_{lc}"]

        def mix(ctx):
            # The one lossy encode of the dubbed track (everything before is lossless)
            speech, accomp = ctx[f"speech:{lc}"], ctx.get("accompaniment_path")
            out = os.path.join(output_dir, f"{video_name}_{lc}_mixed.mp3")
            final = speech
            if separate_music and accomp:
                try:
                    final = mix_audio_tracks(speech_path=speech, accompaniment_path=accomp,
                                             output_path=out)
                except Exception as exc:
                    logger.warning("Audio mix failed for %s (falling back to speech-only): %s",
                                   lang_name, exc)
            if final == speech:   # no music to mix in
                final = encode_final_track(speech, out)
            ctx[f"audio:{lc}"] = final

        add(f"mix_{lc}", mix, speech_ready, CPU, f"Mix music → {lang_name}",
            after=[f"enhance_{lc}"] if enable_enhancer else [], critical=False)

        if enable_lip_sync:
            def lipsync(ctx):
                ls_dir = os.path.join(output_dir, "lipsync")
                os.makedirs(ls_dir, exist_ok=True)
                processor = LipSyncProcessor(debug_dir=os.path.join(output_dir, "lipsync_debug"))
                result = processor.sync(video_path, ctx[f"audio:{lc}"],
                                        os.path.join(ls_dir, f"lipsync_{lc}.mp4"), lc)
                ctx[f"lipsync:{lc}"] = result
                if not result.success:
                    logger.error("Lip sync failed for '%s': %s", lc, result.error_message)
                elif result.fallback_used and result.error_message:
                    logger.error("Lip sync [%s] fell back to passthrough. Reason: %s",
                                 lc, result.error_message)

            add(f"lipsync_{lc}", lipsync, [f"mix_{lc}"], GPU, f"Lip sync → {lang_name}",
                critical=False)

            def lipsync_mkv(ctx):
                result = ctx[f"lipsync:{lc}"]
                if not (result.success and os.path.isfile(result.output_path)):
                    return
                out = os.path.join(output_dir, f"{video_name}_lipsync_{lc}.mkv")
                subs = {lc: ctx[f"subtitle:{lc}"]} if f"subtitle:{lc}" in ctx else {}
                mux_to_mkv(original_video=video_path, video_source=result.output_path,
                           audio_tracks={lc: ctx[f"audio:{lc}"]},
                           subtitle_tracks=subs, output_path=out)
                ctx[f"lipsync_mkv:{lc}"] = out
                logger.info("Lip-synced MKV: %s", out)

            add(f"lipsync_mkv_{lc}", lipsync_mkv, [f"lipsync_{lc}"], CPU,
                f"Lip-synced MKV → {lang_name}", after=[f"subtitles_{lc}"], critical=False)

    def prosody(ctx, path: str) -> str:
        from src.prosody_transfer import apply_prosody_transfer, is_parselmouth_available
        if not is_parselmouth_available():
            logger.warning("Prosody requested but Parselmouth not installed.")
            return path
        try:
            return apply_prosody_transfer(ctx["audio_path"], path,
                                          os.path.splitext(path)[0] + "_prosody.flac",
                                          voice_profile=ctx.get("voice_profile"))
        except Exception as exc:
            logger.error("Prosody transfer failed: %s", exc)
            return path

    for lc in target_langs:
        add_language(lc)

    # ── Final MKV: every language that made it, plus all subtitles ──────────
    if do_tts:
        def mux(ctx):
            audio = {lc: ctx[f"audio:{lc}"] for lc in target_langs if f"audio:{lc}" in ctx}
            if not audio:
                raise RuntimeError("No dubbed audio was produced for any language — "
                                   + failure_summary(dag))
            subs = {lc: ctx[f"subtitle:{lc}"] for lc in ["en", *target_langs]
                    if f"subtitle:{lc}" in ctx}
            out = os.path.join(output_dir, f"{video_name}_dubbed.mkv")
            mux_to_mkv(original_video=video_path, audio_tracks=audio,
                       subtitle_tracks=subs, output_path=out)
            ctx["dubbed_video"] = out

        add("mux", mux, ["subtitles_en"], CPU, "Mux MKV",
            after=[f"mix_{lc}" for lc in target_langs]
                  + [f"subtitles_{lc}" for lc in target_langs])

    logger.info("[DAG %s] built %d stages for %s", request_id, len(dag.nodes), target_langs)
    return dag


def encode_final_track(speech_path: str, output_path: str) -> str:
    """Encode a dubbed voice track the same way mix_audio_tracks writes a mix."""
    subprocess.run(["ffmpeg", "-y", "-i", speech_path, "-ar", "44100", "-ac", "2",
                    "-c:a", "libmp3lame", "-b:a", "192k", output_path],
                   capture_output=True, check=True)
    return output_path


def failure_summary(dag: PipelineDAG) -> str:
    """'Stage: error; …' for every failed stage of *dag*."""
    parts = [f"{n.label}: {n.error}" for n in dag.nodes.values() if n.status == FAILED]
    return "; ".join(parts) or "no stage reported an error"
