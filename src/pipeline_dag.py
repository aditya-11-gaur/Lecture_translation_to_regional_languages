"""
pipeline_dag.py
───────────────
Constructs a PipelineDAG for a single dubbing request.

Each stage of the existing pipeline becomes one or more Node objects.
Multi-language stages (translate, tts, voice_clone, align, mix, lipsync)
fan out into per-language nodes that run in PARALLEL subject only to
resource semaphores (GPU serial, API rate-limited).

Node context keys (written by upstream, read by downstream)
───────────────────────────────────────────────────────────
  audio_extract     → ctx["audio_path"]        (str)
  demucs            → ctx["vocals_path"]        (str)
                    → ctx["accomp_path"]        (str)
  voice_analyze     → ctx["voice_profile"]      (VoiceProfile)
  stt               → ctx["english_segments"]   (list[Segment])
  translate_{lc}    → ctx["segments_{lc}"]      (list[Segment])
  subtitle_{lc}     → ctx["srt_{lc}"]           (str path)
  tts_{lc}          → ctx["tts_files_{lc}"]     (dict[int,str])
  vc_{lc}           → ctx["vc_files_{lc}"]      (dict[int,str])
  align_{lc}        → ctx["aligned_{lc}"]       (str path)
  mix_{lc}          → ctx["mixed_{lc}"]         (str path)
  lipsync_{lc}      → ctx["lipsync_{lc}"]       (str path)
  mux               → ctx["dubbed_video"]        (str path)
                    → ctx["result"]              (dict — final pipeline result)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.dag_executor import PipelineDAG, Node, GPU, API, CPU

logger = logging.getLogger("nptel_pipeline.dag")


def build_pipeline_dag(
    request_id: str,
    video_path: str,
    output_dir: str,
    target_langs: list[str],
    stt_method: str = "whisper",
    translate_method: str = "sarvam",
    tts_engine: str = "sarvam_vc",
    separate_music: bool = True,
    enable_voice_cloning: bool = True,
    enable_lip_sync: bool = False,
    enable_glossary: bool = True,
    enable_prosody: bool = False,
    enable_enhancer: bool = False,
    **extra_kwargs,
) -> PipelineDAG:
    """
    Build and return a fully-wired PipelineDAG for one dubbing request.

    All node callables close over the parameters above. Each callable
    receives and writes to the shared `ctx` dict.
    """
    dag = PipelineDAG(request_id=request_id)
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Audio extraction ────────────────────────────────────────────
    def _audio_extract(ctx: dict) -> Any:
        from src.audio_extractor import extract_audio
        audio_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
        extract_audio(video_path, audio_path)
        ctx["audio_path"] = audio_path
        ctx["video_path"] = video_path

    dag.add(Node("audio_extract", _audio_extract, deps=[], resource=CPU,
                 label="Audio Extraction"))

    # ── Stage 1b: Music separation (Demucs) ──────────────────────────────────
    if separate_music:
        def _demucs(ctx: dict) -> Any:
            from src.audio_separator import separate_audio
            vocals, accomp = separate_audio(ctx["audio_path"], output_dir)
            ctx["vocals_path"] = vocals
            ctx["accomp_path"] = accomp

        dag.add(Node("demucs", _demucs, deps=["audio_extract"], resource=GPU,
                     label="Music Separation (Demucs)"))
        _after_demucs = "demucs"
        _vocals_key   = "vocals_path"
    else:
        # Skip Demucs — STT uses raw audio
        def _passthrough(ctx: dict) -> Any:
            ctx["vocals_path"] = ctx["audio_path"]
            ctx["accomp_path"] = None

        dag.add(Node("demucs", _passthrough, deps=["audio_extract"], resource=CPU,
                     label="Audio Passthrough"))
        _after_demucs = "demucs"
        _vocals_key   = "vocals_path"

    # ── Stage: Voice analysis ────────────────────────────────────────────────
    def _voice_analyze(ctx: dict) -> Any:
        from src.voice_analyzer import analyze_voice
        profile = analyze_voice(ctx[_vocals_key])
        ctx["voice_profile"] = profile

    dag.add(Node("voice_analyze", _voice_analyze, deps=[_after_demucs], resource=CPU,
                 label="Voice Analysis"))

    # ── Stage 2: STT (Whisper) ────────────────────────────────────────────────
    def _stt(ctx: dict) -> Any:
        from src.transcriber import transcribe
        segments = transcribe(
            ctx[_vocals_key],
            method=stt_method,
            cache_dir=os.path.join(output_dir, ".cache"),
        )
        ctx["english_segments"] = segments

    dag.add(Node("stt", _stt, deps=[_after_demucs], resource=GPU,
                 label="Speech-to-Text (Whisper)"))

    # ── Stage: Dynamic glossary ───────────────────────────────────────────────
    if enable_glossary:
        def _glossary(ctx: dict) -> Any:
            from src.glossary import build_glossary
            glossary = build_glossary(ctx["english_segments"])
            ctx["glossary"] = glossary

        dag.add(Node("glossary", _glossary, deps=["stt"], resource=API,
                     label="Dynamic Glossary Extraction"))
        _after_glossary = "glossary"
    else:
        def _no_glossary(ctx: dict) -> Any:
            ctx["glossary"] = {}

        dag.add(Node("glossary", _no_glossary, deps=["stt"], resource=CPU,
                     label="Glossary (disabled)"))
        _after_glossary = "glossary"

    # ── Per-language fan-out ──────────────────────────────────────────────────
    # Each language gets its own translate → subtitle → tts → vc → align → mix → lipsync chain.
    # All these chains run IN PARALLEL (subject to GPU semaphore for vc/lipsync).

    lipsync_nodes = []   # collected to wire into mux

    for lc in target_langs:

        # ── Translation ──────────────────────────────────────────────────────
        def _translate(ctx: dict, _lc=lc) -> Any:
            from src.translator import translate_segments
            translated = translate_segments(
                ctx["english_segments"],
                target_lang=_lc,
                method=translate_method,
                glossary=ctx.get("glossary", {}),
                cache_dir=os.path.join(output_dir, ".cache"),
            )
            ctx[f"segments_{_lc}"] = translated

        dag.add(Node(f"translate_{lc}", _translate,
                     deps=[_after_glossary, "voice_analyze"], resource=API,
                     label=f"Translation → {lc}"))

        # ── Subtitle generation ───────────────────────────────────────────────
        def _subtitle(ctx: dict, _lc=lc) -> Any:
            from src.subtitle_generator import generate_subtitles
            srt, vtt = generate_subtitles(
                ctx[f"segments_{_lc}"],
                lang_code=_lc,
                output_dir=output_dir,
                video_name=os.path.splitext(os.path.basename(video_path))[0],
            )
            ctx[f"srt_{_lc}"] = srt
            ctx[f"vtt_{_lc}"] = vtt

        dag.add(Node(f"subtitle_{lc}", _subtitle,
                     deps=[f"translate_{lc}"], resource=CPU,
                     label=f"Subtitles → {lc}"))

        # ── TTS synthesis ─────────────────────────────────────────────────────
        def _tts(ctx: dict, _lc=lc) -> Any:
            from src.tts_generator import synthesize_all_segments
            tts_dir = os.path.join(output_dir, f"tts_segments_{_lc}")
            os.makedirs(tts_dir, exist_ok=True)
            files = synthesize_all_segments(
                segments=ctx[f"segments_{_lc}"],
                lang_code=_lc,
                out_dir=tts_dir,
                engine=tts_engine,
                original_segments=ctx["english_segments"],
                voice_profile=ctx.get("voice_profile"),
                voice_reference_audio=ctx.get(_vocals_key),
            )
            ctx[f"tts_files_{_lc}"] = files

        dag.add(Node(f"tts_{lc}", _tts,
                     deps=[f"translate_{lc}"], resource=API,
                     label=f"TTS → {lc}"))

        # ── Voice cloning (optional, GPU) ─────────────────────────────────────
        vc_dep = f"tts_{lc}"
        if enable_voice_cloning and "_vc" not in tts_engine:
            def _vc(ctx: dict, _lc=lc) -> Any:
                from src.voice_converter import convert_segments_batch, extract_reference_clip, is_openvoice_available
                if not is_openvoice_available():
                    ctx[f"vc_files_{_lc}"] = ctx[f"tts_files_{_lc}"]
                    return
                tts_dir  = os.path.join(output_dir, f"tts_segments_{_lc}")
                vc_dir   = os.path.join(tts_dir, "vc_converted")
                ref_path = os.path.join(tts_dir, "_vc_reference.wav")
                os.makedirs(vc_dir, exist_ok=True)
                extract_reference_clip(ctx[_vocals_key], ref_path)
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                converted = convert_segments_batch(ctx[f"tts_files_{_lc}"], ref_path, vc_dir, device=device)
                ctx[f"vc_files_{_lc}"] = converted

            dag.add(Node(f"vc_{lc}", _vc,
                         deps=[f"tts_{lc}"], resource=GPU,
                         label=f"Voice Cloning → {lc}"))
            vc_dep = f"vc_{lc}"
        else:
            def _vc_passthrough(ctx: dict, _lc=lc) -> Any:
                ctx[f"vc_files_{_lc}"] = ctx[f"tts_files_{_lc}"]

            dag.add(Node(f"vc_{lc}", _vc_passthrough,
                         deps=[f"tts_{lc}"], resource=CPU,
                         label=f"Voice Clone Passthrough → {lc}"))
            vc_dep = f"vc_{lc}"

        # ── Temporal alignment ────────────────────────────────────────────────
        def _align(ctx: dict, _lc=lc) -> Any:
            from src.audio_aligner import align_segments
            aligned_path = align_segments(
                segment_files=ctx[f"vc_files_{_lc}"],
                original_segments=ctx["english_segments"],
                output_dir=output_dir,
                lang_code=_lc,
                video_name=os.path.splitext(os.path.basename(video_path))[0],
            )
            ctx[f"aligned_{_lc}"] = aligned_path

        dag.add(Node(f"align_{lc}", _align,
                     deps=[vc_dep, "stt"], resource=CPU,
                     label=f"Temporal Alignment → {lc}"))

        # ── Music remix ───────────────────────────────────────────────────────
        def _mix(ctx: dict, _lc=lc) -> Any:
            from src.audio_separator import mix_audio
            mixed = mix_audio(
                dubbed_audio=ctx[f"aligned_{_lc}"],
                accompaniment=ctx.get("accomp_path"),
                output_dir=output_dir,
                lang_code=_lc,
                video_name=os.path.splitext(os.path.basename(video_path))[0],
            )
            ctx[f"mixed_{_lc}"] = mixed

        dag.add(Node(f"mix_{lc}", _mix,
                     deps=[f"align_{lc}", "demucs"], resource=CPU,
                     label=f"Music Remix → {lc}"))

        # ── Lip sync (optional, GPU) ──────────────────────────────────────────
        if enable_lip_sync:
            def _lipsync(ctx: dict, _lc=lc) -> Any:
                from src.lip_sync import run_lipsync
                ls_path = run_lipsync(
                    video_path=ctx["video_path"],
                    audio_path=ctx[f"mixed_{_lc}"],
                    output_dir=output_dir,
                    lang_code=_lc,
                )
                ctx[f"lipsync_{_lc}"] = ls_path

            dag.add(Node(f"lipsync_{lc}", _lipsync,
                         deps=[f"mix_{lc}"], resource=GPU,
                         label=f"Lip Sync → {lc}"))
            lipsync_nodes.append(f"lipsync_{lc}")
        else:
            lipsync_nodes.append(f"mix_{lc}")

    # ── Final mux ─────────────────────────────────────────────────────────────
    def _mux(ctx: dict) -> Any:
        from src.video_muxer import mux_video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        dubbed_path = mux_video(
            video_path=ctx["video_path"],
            audio_tracks={lc: ctx.get(f"mixed_{lc}") for lc in target_langs},
            subtitle_tracks={lc: ctx.get(f"srt_{lc}") for lc in target_langs},
            output_dir=output_dir,
            video_name=video_name,
        )
        ctx["dubbed_video"] = dubbed_path
        # Build the same result dict the old pipeline returned
        ctx["result"] = {
            "dubbed_video":   dubbed_path,
            "subtitles":      {lc: ctx.get(f"srt_{lc}") for lc in target_langs},
            "aligned_audio":  {lc: ctx.get(f"aligned_{lc}") for lc in target_langs},
            "lip_synced_videos": {lc: ctx.get(f"lipsync_{lc}") for lc in target_langs
                                  if ctx.get(f"lipsync_{lc}")},
            "lip_sync_used_wav2lip": enable_lip_sync,
            "lip_sync_errors": {},
            "lip_synced_mkvs": {},
        }

    dag.add(Node("mux", _mux,
                 deps=lipsync_nodes + [f"subtitle_{lc}" for lc in target_langs],
                 resource=CPU,
                 label="MKV Mux"))

    logger.info(
        "[build_pipeline_dag] request=%s nodes=%d langs=%s",
        request_id, len(dag.nodes), target_langs,
    )
    return dag
