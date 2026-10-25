"""
pipeline.py
───────────
Entry point for one dubbing job: run_pipeline() builds the job's stage DAG
(src/pipeline_dag.py) and runs it on the shared DAG executor, which starts
each stage once its inputs are ready and a GPU / CPU / API slot is free.

Flow (per target language, languages in parallel):
  Video → audio extraction → Demucs → STT (English)
       → translation → subtitles
       → TTS → voice cloning → alignment → pitch match → music mix → MKV

Also holds the transcript/translation cache helpers the stages use.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import subprocess
import uuid
from typing import Callable, Protocol

from config.settings import (
    OUTPUT_DIR,
    ENABLE_CACHE,
    CACHE_DIR_NAME,
    get_default_engine,
    DEFAULT_TARGET_LANGS,
    DEFAULT_TTS_ENGINE,
    SEPARATE_MUSIC,
)
from config.settings import ENABLE_VOICE_CLONING as _DEFAULT_VOICE_CLONING
from src.transcriber import Segment
from src.utils import run_preflight_checks

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
    enable_prosody: bool = False,
    enable_enhancer: bool = False,
    enable_lip_sync: bool = False,
) -> dict:
    """
    Run the full lecture-dubbing pipeline for one video.

    Args:
        video_path:        Path to the input video.
        output_dir:        Where to store all outputs.
        stt_method:        "whisper" or "sarvam" (auto-detected if None).
        translate_method:  "sarvam" or "gcp" (auto-detected if None).
        do_tts:            Whether to generate dubbed audio (else subtitles only).
        tts_engine:        TTS engine id from TTS_ENGINE_REGISTRY.
        target_langs:      Language codes to translate into.
        progress:          Optional callback (finished_stages, total_stages, message).

    Returns:
        Dict with paths to all generated artefacts, plus "warnings" and
        "language_errors" for languages that failed while others succeeded.

    Raises:
        RuntimeError if a shared stage fails or no language produced output.
    """
    from src.dag_executor import DONE, FAILED, SKIPPED, get_executor
    from src.pipeline_dag import build_pipeline_dag, failure_summary

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

    progress(0, 1, "Running preflight checks…")
    run_preflight_checks()
    logger.info("Preflight checks passed")

    dag = build_pipeline_dag(
        f"{video_name}-{uuid.uuid4().hex[:6]}", video_path, output_dir,
        stt_method=stt_method, translate_method=translate_method,
        target_langs=target_langs, do_tts=do_tts, tts_engine=tts_engine,
        separate_music=separate_music, enable_voice_cloning=enable_voice_cloning,
        enable_prosody=enable_prosody, enable_enhancer=enable_enhancer,
        enable_lip_sync=enable_lip_sync,
    )
    ctx = get_executor().run(dag, progress)
    nodes = dag.nodes

    translated = {lc: ctx[f"translated:{lc}"] for lc in target_langs
                  if f"translated:{lc}" in ctx}
    if not translated:
        raise RuntimeError("Translation failed for every language — " + failure_summary(dag))

    # A language that failed part-way: its first failed stage explains why
    language_errors: dict[str, str] = {}
    for lc in target_langs:
        failed = [n for nid, n in nodes.items() if nid.endswith(f"_{lc}") and n.status == FAILED]
        if failed:
            language_errors[lc] = f"{failed[0].label} failed: {failed[0].error}"

    warnings = list(ctx["warnings"])
    for lc in target_langs:
        n = nodes.get(f"clone_{lc}")
        if n and n.status in (FAILED, SKIPPED) and lc not in language_errors:
            warnings.append(f"Voice cloning for {lc} did not run ({n.error}) — plain TTS voice used")

    lip_synced_videos: dict[str, str] = {}
    lip_sync_errors: dict[str, str] = {}
    lip_results = {lc: ctx[f"lipsync:{lc}"] for lc in target_langs if f"lipsync:{lc}" in ctx}
    for lc, res in lip_results.items():
        if res.success:
            lip_synced_videos[lc] = res.output_path
            if res.fallback_used and res.error_message:
                lip_sync_errors[lc] = res.error_message
        else:
            lip_sync_errors[lc] = res.error_message or "Unknown error"
    for lc in target_langs:
        n = nodes.get(f"lipsync_{lc}")
        if n and n.status == FAILED:
            lip_sync_errors[lc] = n.error

    artefacts: dict = {
        "video": video_path,
        "audio": ctx.get("audio_path"),
        "vocals": ctx.get("vocals_path"),
        "accompaniment": ctx.get("accompaniment_path"),
        "english_segments": [s.to_dict() for s in ctx["english_segments"]],
        "translated": translated,
        "subtitles": {lc: ctx[f"subtitle:{lc}"] for lc in ["en", *target_langs]
                      if f"subtitle:{lc}" in ctx},
        "voice_cloned": any(n.status == DONE for nid, n in nodes.items()
                            if nid.startswith("clone_")),
        "voice_cloning_note": ctx.get("voice_cloning_note"),
        "warnings": warnings,
        "language_errors": language_errors,
        # Per-stage timing (monotonic clock, comparable across jobs)
        "stages": [
            {"stage": nid, "resource": n.resource, "status": n.status,
             "start": n.started_at, "end": n.finished_at}
            for nid, n in nodes.items()
        ],
    }
    if do_tts:
        artefacts.update({
            "aligned_audio": {lc: ctx[f"audio:{lc}"] for lc in target_langs
                              if f"audio:{lc}" in ctx},
            "dubbed_video": ctx.get("dubbed_video"),
            "lip_synced_videos": lip_synced_videos,
            "lip_sync_errors": lip_sync_errors,
            "lip_sync_used_wav2lip": any(r.success and not r.fallback_used
                                         for r in lip_results.values()),
            "lip_synced_mkvs": {lc: ctx[f"lipsync_mkv:{lc}"] for lc in target_langs
                                if f"lipsync_mkv:{lc}" in ctx},
        })
    else:
        logger.info("TTS / alignment / muxing skipped (subtitles only)")

    for msg in warnings:
        logger.warning("Job warning: %s", msg)
    for lc, msg in language_errors.items():
        logger.error("Language %s failed: %s", lc, msg)
    logger.info("Pipeline complete!")
    return artefacts
