"""
tts_generator.py
────────────────
Text-to-Speech: convert translated text segments into audio files
in Hindi, Telugu, and Odia.

Supported TTS engines:
  • edge-tts   — Free, no API key, neural-quality Microsoft Edge voices
  • sarvam     — Requires Sarvam API key
  • sarvam_vc  — Sarvam + OpenVoice v2 voice cloning
  • gcptts     — Google Cloud TTS (Neural2/Studio voices, requires GCP_API_KEY)
  • gcptts_vc  — GCP TTS + OpenVoice v2 voice cloning
  • xtts       — Coqui XTTS v2 (local GPU)
"""

from __future__ import annotations
import asyncio
import logging
import os
import subprocess
import tempfile
import edge_tts
from config.settings import (
    SARVAM_API_KEY,
    GCP_API_KEY,
    SARVAM_BASE_URL,
    TARGET_LANGUAGES,
    OUTPUT_DIR,
)
from src.transcriber import Segment
from src.voice_analyzer import VoiceProfile
from src.filler_detector import get_text_without_fillers, get_text_with_target_fillers

logger = logging.getLogger("nptel_pipeline")

# ── Concurrency limit for TTS API calls ──────────────────────────────
TTS_CONCURRENCY = 8

# ── Estimated characters-per-second for each language (edge-tts voices) ──────
# Used for first-pass duration estimation.  The two-pass system measures
# real durations after the first synthesis, so these only need to be
# in the right ballpark.
_LANG_CPS: dict[str, float] = {
    "hi": 10.0,   # Hindi Devanagari chars/sec (measured from edge-tts)
    "te":  9.0,   # Telugu script chars/sec
    "od":  9.0,   # Odia script chars/sec
}
_DEFAULT_CPS = 10.0

# How far the TTS rate can be pushed via SSML prosody
_MAX_TTS_RATE_PCT = 60    # e.g. "+60%"  (1.6× speed)
_MIN_TTS_RATE_PCT = -35   # e.g. "-35%"  (0.65× speed)

# Two-pass re-synthesis threshold: if actual_dur / target_dur is outside
# this range, re-synthesise with a corrected rate.
_RESYNTH_THRESHOLD_LOW  = 0.80   # 20% too short
_RESYNTH_THRESHOLD_HIGH = 1.20   # 20% too long


# ── Voice mapping: lang_code → edge-tts voice name ──────────────────────────
# Run `edge-tts --list-voices` to see all available voices.
# Male and female variants for gender-aware voice selection.
EDGE_TTS_VOICES_FEMALE: dict[str, str] = {
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "od": "or-IN-SubhasiniNeural",
}
EDGE_TTS_VOICES_MALE: dict[str, str] = {
    "hi": "hi-IN-MadhurNeural",
    "te": "te-IN-MohanNeural",
    "od": "or-IN-SubhasiniNeural",   # no male Odia voice available; fall back to female
}

# Legacy alias used by tests / external callers
EDGE_TTS_VOICES = EDGE_TTS_VOICES_FEMALE


def _select_edge_voice(lang_code: str, voice_profile: VoiceProfile | None = None) -> str | None:
    """Pick the best edge-tts voice for *lang_code* based on the voice profile."""
    if voice_profile and voice_profile.is_male:
        voice = EDGE_TTS_VOICES_MALE.get(lang_code)
        if voice:
            return voice
    if voice_profile and voice_profile.is_male is False:
        voice = EDGE_TTS_VOICES_FEMALE.get(lang_code)
        if voice:
            return voice
    # Unknown gender or no profile → prefer female (default)
    return EDGE_TTS_VOICES_FEMALE.get(lang_code) or EDGE_TTS_VOICES_MALE.get(lang_code)

# ── TTS engine registry ─────────────────────────────────────────────────────
# Import availability check lazily to avoid circular imports at module load
def _openvoice_available() -> bool:
    try:
        from src.voice_converter import is_openvoice_available
        return is_openvoice_available()
    except Exception:
        return False

TTS_ENGINE_REGISTRY: dict[str, dict] = {
    "edge_tts":   {"name": "Edge TTS (Free)",                       "available": True},
    "sarvam":     {"name": "Sarvam AI",                              "available": bool(SARVAM_API_KEY)},
    "sarvam_vc":  {"name": "Sarvam AI + Voice Clone ✨",             "available": bool(SARVAM_API_KEY)},
    "gcptts":     {"name": "Google Cloud TTS (Neural2)",             "available": bool(GCP_API_KEY)},
    "gcptts_vc":  {"name": "Google Cloud TTS + Voice Clone ✨",       "available": bool(GCP_API_KEY)},
    "xtts":       {"name": "Coqui XTTS v2 (Local GPU)",             "available": True},
}


def get_available_tts_engines() -> list[str]:
    """Return TTS engine IDs that are currently usable."""
    return [eid for eid, info in TTS_ENGINE_REGISTRY.items() if info["available"]]


def get_default_tts_engine() -> str:
    """Return the best available TTS engine."""
    available = get_available_tts_engines()
    for preferred in ("gcptts_vc", "sarvam_vc", "gcptts", "sarvam", "xtts", "edge_tts"):
        if preferred in available:
            return preferred
    return "edge_tts"


def is_tts_available() -> bool:
    """Check if any TTS engine is usable."""
    return len(get_available_tts_engines()) > 0


# ── Duration helpers ─────────────────────────────────────────────────────────

def _get_audio_duration(path: str) -> float:
    """Return duration in seconds of an audio file (via ffprobe)."""
    import json as _json
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = _json.loads(result.stdout)
    return float(info["format"]["duration"])


def _compute_rate_for_duration(
    text: str,
    target_dur: float,
    lang_code: str,
    base_rate_pct: int = 0,
) -> int:
    """
    Estimate the edge-tts ``rate`` percentage needed so that *text*
    is spoken in approximately *target_dur* seconds.

    edge-tts rate semantics:  rate=+X% → speak (1+X/100)× faster
      → actual_duration ≈ natural_duration / (1 + X/100)

    So to hit a target:  (1 + rate/100) = natural_duration / target_dur
                         rate = (natural_dur / target_dur - 1) × 100

    Returns an integer percentage, e.g. +30 or -20.
    """
    if target_dur <= 0 or not text.strip():
        return base_rate_pct

    cps = _LANG_CPS.get(lang_code, _DEFAULT_CPS)
    estimated_natural_dur = max(len(text.strip()) / cps, 0.5)

    speed_factor = estimated_natural_dur / target_dur
    rate_pct = int((speed_factor - 1.0) * 100)
    rate_pct = max(_MIN_TTS_RATE_PCT, min(_MAX_TTS_RATE_PCT, rate_pct))
    return rate_pct


def _compute_rate_from_actual(
    actual_dur: float,
    target_dur: float,
    previous_rate_pct: int = 0,
) -> int:
    """
    Compute a corrected rate for the **second pass** using the
    *measured* duration from the first synthesis.

    The actual duration already accounts for the previous rate,
    so: new_rate = previous_speed × (actual_dur / target_dur) - 1
    """
    if target_dur <= 0 or actual_dur <= 0:
        return previous_rate_pct

    prev_speed = 1.0 + (previous_rate_pct / 100.0)
    correction = actual_dur / target_dur
    new_speed = prev_speed * correction
    rate_pct = int((new_speed - 1.0) * 100)
    rate_pct = max(_MIN_TTS_RATE_PCT, min(_MAX_TTS_RATE_PCT, rate_pct))
    return rate_pct


def _rate_pct_to_str(pct: int) -> str:
    """Convert integer rate percentage to edge-tts format string."""
    if pct >= 0:
        return f"+{pct}%"
    return f"{pct}%"


# ── Main entry points ───────────────────────────────────────────────────────

def generate_tts(
    segments: list[Segment],
    lang_code: str,
    video_name: str,
    output_dir: str | None = None,
    engine: str | None = None,
) -> str | None:
    """
    Synthesize speech for the translated segments in the given language.

    Args:
        segments:   Translated Segment list.
        lang_code:  Target language code (hi / te / od).
        video_name: Base name of the source video.
        output_dir: Directory to write audio.
        engine:     TTS engine id ("edge_tts" or "sarvam").

    Returns:
        Path to the generated audio file, or None on failure.
    """
    if engine is None:
        engine = get_default_tts_engine()
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    lang_name = TARGET_LANGUAGES[lang_code]["name"]
    out_path = os.path.join(output_dir, f"{video_name}_{lang_code}_audio.mp3")

    logger.info("Generating %s audio via %s → %s", lang_name, engine, out_path)

    if engine == "edge_tts":
        return _generate_edge_tts(segments, lang_code, out_path)
    elif engine == "sarvam":
        return _generate_sarvam_tts(segments, lang_code, out_path)
    elif engine == "xtts":
        return _generate_xtts(segments, lang_code, out_path)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


def generate_all_tts(
    translated_segments: dict[str, list[Segment]],
    video_name: str,
    output_dir: str | None = None,
    engine: str | None = None,
) -> dict[str, str]:
    """
    Generate TTS audio for all target languages.

    Returns:
        Dict mapping lang code → audio file path.
    """
    if engine is None:
        engine = get_default_tts_engine()

    if not is_tts_available():
        logger.warning("No TTS engine available (generate_all_tts). Skipping.")
        return {}

    paths: dict[str, str] = {}
    for lang_code, segments in translated_segments.items():
        result = generate_tts(segments, lang_code, video_name, output_dir, engine)
        if result:
            paths[lang_code] = result
    return paths


# ── Per-segment synthesis (for temporal alignment pipeline) ──────────────────

def synthesize_segments(
    segments: list[Segment],
    lang_code: str,
    output_dir: str,
    engine: str | None = None,
    voice_profile: VoiceProfile | None = None,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
    voice_reference_audio: str | None = None,
) -> dict[int, str]:
    """
    Synthesise each segment individually and return a mapping of
    segment index → audio file path (NOT concatenated).

    This is used by the audio_aligner to time-stretch each piece
    to match the original segment duration.

    Args:
        segments:          Translated Segment list.
        lang_code:         Target language code.
        output_dir:        Directory for output files.
        engine:            TTS engine id.
        voice_profile:     If given, adjust TTS rate/pitch to match speaker.
        preserve_fillers:  If True, remove fillers from text before TTS.
        original_segments: English segments with original timing.  When
                           provided, enables *duration-aware* synthesis:
                           the TTS rate is adjusted per-segment so the
                           output roughly matches the target time window,
                           dramatically reducing the need for post-hoc
                           time-stretching.
    """
    if engine is None:
        engine = get_default_tts_engine()

    seg_dir = os.path.join(output_dir, f"tts_segments_{lang_code}")
    os.makedirs(seg_dir, exist_ok=True)

    lang_name = TARGET_LANGUAGES[lang_code]["name"]
    segment_files: dict[int, str] = {}

    if engine == "edge_tts":
        segment_files = _synth_segments_edge(
            segments, lang_code, seg_dir,
            voice_profile=voice_profile,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )
    elif engine == "sarvam":
        segment_files = _synth_segments_sarvam(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )
    elif engine == "sarvam_vc":
        segment_files = _synth_segments_sarvam_vc(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
            voice_reference_audio=voice_reference_audio,
        )
    elif engine == "gcptts":
        segment_files = _synth_segments_gcptts(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )
        if not segment_files:
            logger.warning("[GCP TTS] All segments failed — falling back to sarvam engine")
            segment_files = _synth_segments_sarvam(
                segments, lang_code, seg_dir,
                preserve_fillers=preserve_fillers,
                original_segments=original_segments,
            )
    elif engine == "gcptts_vc":
        segment_files = _synth_segments_gcptts_vc(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
            voice_reference_audio=voice_reference_audio,
        )
        if not segment_files:
            logger.warning("[GCP TTS VC] All segments failed — falling back to sarvam_vc engine")
            segment_files = _synth_segments_sarvam_vc(
                segments, lang_code, seg_dir,
                preserve_fillers=preserve_fillers,
                original_segments=original_segments,
                voice_reference_audio=voice_reference_audio,
            )
    elif engine == "xtts":
        segment_files = _synth_segments_xtts(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            voice_reference_audio=voice_reference_audio,
            original_segments=original_segments,
        )
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")

    logger.info("%s: %d/%d segment files ready",
                lang_name, len(segment_files), len(segments))
    return segment_files


def synthesize_all_segments(
    translated_segments: dict[str, list[Segment]],
    output_dir: str,
    engine: str | None = None,
    voice_profile: VoiceProfile | None = None,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
    voice_reference_audio: str | None = None,
) -> dict[str, dict[int, str]]:
    """
    Per-segment synthesis for every target language.

    Args:
        original_segments: English segments with original timing.
                           Enables duration-aware rate adjustment.

    Returns:
        { lang_code: { segment_idx: audio_file_path, ... }, ... }
    """
    if engine is None:
        engine = get_default_tts_engine()
    if not is_tts_available():
        logger.warning("No TTS engine available. Skipping.")
        return {}

    result: dict[str, dict[int, str]] = {}
    for lang_code, segments in translated_segments.items():
        result[lang_code] = synthesize_segments(
            segments, lang_code, output_dir, engine,
            voice_profile=voice_profile,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
            voice_reference_audio=voice_reference_audio,
        )
    return result


# ── Per-segment edge-tts ─────────────────────────────────────────────────────

def _synth_segments_edge(
    segments: list[Segment], lang_code: str, out_dir: str,
    voice_profile: VoiceProfile | None = None,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
) -> dict[int, str]:
    """
    Synthesise each segment with edge-tts using **two-pass** duration-aware
    synthesis when *original_segments* is provided.

    Pass 1 — Estimate & Synthesise:
      Compute an initial TTS rate from character-count heuristics and
      synthesise all segments concurrently.

    Pass 2 — Measure & Correct:
      Measure the actual duration of each produced file.  For any segment
      where ``actual_dur / target_dur`` falls outside the threshold
      (±20 %), re-synthesise with a corrected rate derived from the
      *measured* duration.

    This two-pass approach eliminates the dependency on accurate CPS
    estimates and produces audio that is much closer to the target
    duration *before* the aligner (Rubber Band) ever touches it.

    Returns index → file map.
    """
    voice = _select_edge_voice(lang_code, voice_profile)
    if not voice:
        logger.warning("No edge-tts voice for '%s'", lang_code)
        return {}

    lang_name = TARGET_LANGUAGES[lang_code]["name"]
    logger.info("%s: using voice %s", lang_name, voice)
    files: dict[int, str] = {}
    sem = asyncio.Semaphore(TTS_CONCURRENCY)

    # Pitch from voice profile (applied to both passes)
    pitch_str = "+0Hz"
    if voice_profile:
        pitch_str = voice_profile.edge_tts_pitch_str

    # Build per-segment target durations from original segments
    target_durations: dict[int, float] = {}
    if original_segments and len(original_segments) == len(segments):
        for idx, oseg in enumerate(original_segments):
            dur = oseg.end - oseg.start
            if dur > 0:
                target_durations[idx] = dur
        if target_durations:
            logger.info(
                "%s: Duration-aware two-pass synthesis for %d/%d segments",
                lang_name, len(target_durations), len(segments),
            )

    # Per-segment text cache (so pass 2 reuses the same text)
    seg_texts: dict[int, str] = {}
    # Track rate used in pass 1 (for correction in pass 2)
    pass1_rates: dict[int, int] = {}

    # ── helpers ───────────────────────────────────────────────
    def _pitch_kwargs() -> dict[str, str]:
        if pitch_str != "+0Hz":
            return {"pitch": pitch_str}
        return {}

    async def _synth(idx: int, text: str, rate_pct: int | None, path: str) -> bool:
        """Synthesise one segment. Returns True on success."""
        kwargs: dict[str, str] = {}
        if rate_pct is not None and rate_pct != 0:
            kwargs["rate"] = _rate_pct_to_str(rate_pct)
        kwargs.update(_pitch_kwargs())

        async with sem:
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(text, voice, **kwargs)
                    await comm.save(path)
                    return True
                except Exception as exc:
                    if attempt == 2:
                        logger.warning("edge-tts failed for seg %d: %s", idx, exc)
                    else:
                        await asyncio.sleep(1)
        return False

    # ── Pass 1: initial synthesis with estimated rates ────────
    async def _pass1():
        async def _do_one(idx: int, seg: Segment):
            text = seg.text.strip()
            if not text:
                return
            if preserve_fillers and seg.fillers:
                text = get_text_with_target_fillers(text, lang_code)
            seg_texts[idx] = text

            seg_path = os.path.join(out_dir, f"seg_{idx:04d}.mp3")

            rate_pct: int | None = None
            if idx in target_durations:
                rate_pct = _compute_rate_for_duration(
                    text, target_durations[idx], lang_code,
                )
                pass1_rates[idx] = rate_pct

            ok = await _synth(idx, text, rate_pct, seg_path)
            if ok:
                files[idx] = seg_path

        tasks = [_do_one(idx, seg) for idx, seg in enumerate(segments)]
        await asyncio.gather(*tasks)
        logger.info("%s: Pass 1 — %d/%d segments synthesized",
                    lang_name, len(files), len(segments))

    asyncio.run(_pass1())

    # ── Pass 2: measure durations and re-synthesise outliers ──
    if target_durations and files:
        resynth_needed: list[tuple[int, float, float, int]] = []  # (idx, actual, target, old_rate)

        for idx, path in files.items():
            if idx not in target_durations:
                continue
            try:
                actual = _get_audio_duration(path)
            except Exception:
                continue
            target = target_durations[idx]
            ratio = actual / target

            if ratio < _RESYNTH_THRESHOLD_LOW or ratio > _RESYNTH_THRESHOLD_HIGH:
                old_rate = pass1_rates.get(idx, 0)
                new_rate = _compute_rate_from_actual(actual, target, old_rate)
                if new_rate != old_rate:
                    resynth_needed.append((idx, actual, target, new_rate))

        if resynth_needed:
            logger.info(
                "%s: Pass 2 — re-synthesising %d segments (duration mismatch >20%%)",
                lang_name, len(resynth_needed),
            )

            async def _pass2():
                async def _redo(idx: int, actual: float, target: float, new_rate: int):
                    text = seg_texts.get(idx, "")
                    if not text:
                        return
                    seg_path = os.path.join(out_dir, f"seg_{idx:04d}.mp3")
                    logger.debug(
                        "  seg %d: actual=%.2fs target=%.2fs → rate=%+d%%",
                        idx, actual, target, new_rate,
                    )
                    ok = await _synth(idx, text, new_rate, seg_path)
                    if ok:
                        files[idx] = seg_path

                tasks = [_redo(*args) for args in resynth_needed]
                await asyncio.gather(*tasks)

            # Recreate semaphore for the new event loop that asyncio.run() will
            # create — the one from pass 1 was closed and must not be reused.
            sem = asyncio.Semaphore(TTS_CONCURRENCY)
            asyncio.run(_pass2())
            logger.info("%s: Pass 2 complete", lang_name)
        else:
            logger.info("%s: Pass 2 — all segments within threshold, no re-synthesis needed", lang_name)

    return files


# ── Per-segment sarvam TTS ───────────────────────────────────────────────────

# ── Natural characters-per-second estimates for Sarvam bulbul:v3 at pace=1.0 ─
# Used to pre-scale `pace` so each segment roughly fills its target window.
_SARVAM_NATURAL_CPS: dict[str, float] = {
    "hi": 5.5,   # Hindi Devanagari
    "te": 5.0,   # Telugu script
    "od": 5.0,   # Odia script
}
_SARVAM_MIN_PACE = 0.5
_SARVAM_MAX_PACE = 2.0


def _synth_segments_sarvam(
    segments: list[Segment], lang_code: str, out_dir: str,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
) -> dict[int, str]:
    """
    Synthesise each segment with Sarvam Bulbul v3, returning index → file map.

    When *original_segments* is provided, the ``pace`` parameter is estimated
    per-segment so the output duration roughly matches the target window,
    minimising the rubber-band stretch the aligner needs to apply.
    """
    import base64
    from sarvamai import SarvamAI

    if not SARVAM_API_KEY:
        logger.warning("Sarvam API key not set")
        return {}

    lang_info = TARGET_LANGUAGES[lang_code]
    lang_name = lang_info["name"]
    sarvam_code = lang_info["sarvam_code"]
    speaker = SARVAM_SPEAKERS.get(lang_code, "priya")
    natural_cps = _SARVAM_NATURAL_CPS.get(lang_code, 5.5)

    # Build per-segment target durations when available
    target_durations: dict[int, float] = {}
    if original_segments and len(original_segments) == len(segments):
        for i, oseg in enumerate(original_segments):
            dur = oseg.end - oseg.start
            if dur > 0:
                target_durations[i] = dur
        if target_durations:
            logger.info(
                "[Sarvam] Duration-aware pace control for %d/%d segments",
                len(target_durations), len(segments),
            )

    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    files: dict[int, str] = {}

    for idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        if preserve_fillers and seg.fillers:
            text = get_text_with_target_fillers(text, lang_code)
        if len(text) > 2500:
            text = text[:2500]

        # Compute pace: natural_duration / target_duration
        pace: float | None = None
        if idx in target_durations:
            natural_dur = max(len(text) / natural_cps, 0.5)
            raw_pace = natural_dur / target_durations[idx]
            pace = max(_SARVAM_MIN_PACE, min(_SARVAM_MAX_PACE, raw_pace))

        try:
            call_kwargs: dict = dict(
                text=text,
                target_language_code=sarvam_code,
                speaker=speaker,
                model="bulbul:v3",
                output_audio_codec="mp3",
            )
            if pace is not None:
                call_kwargs["pace"] = round(pace, 2)

            resp = client.text_to_speech.convert(**call_kwargs)
            if resp.audios:
                audio_bytes = base64.b64decode(resp.audios[0])
                seg_path = os.path.join(out_dir, f"seg_{idx:04d}.mp3")
                with open(seg_path, "wb") as f:
                    f.write(audio_bytes)
                files[idx] = seg_path
                logger.debug(
                    "[Sarvam] Seg %d: pace=%.2f target=%.2fs",
                    idx, pace if pace is not None else 1.0,
                    target_durations.get(idx, 0),
                )
        except Exception as exc:
            logger.warning("Sarvam TTS error on segment %d: %s", idx + 1, exc)

        if (idx + 1) % 10 == 0:
            logger.info("  %s: %d/%d segments", lang_name, idx + 1, len(segments))

    return files


# ── Sarvam + Voice Clone engine ──────────────────────────────────────────────

def _synth_segments_sarvam_vc(
    segments: list[Segment],
    lang_code: str,
    out_dir: str,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
    voice_reference_audio: str | None = None,
) -> dict[int, str]:
    """
    Combined engine: Sarvam bulbul:v3 TTS → OpenVoice v2 voice cloning.

    Step 1 — Sarvam generates high-quality, naturally-fluent Indic speech
              with duration-aware pace control.
    Step 2 — OpenVoice v2 transfers the *timbre* of the original lecturer
              onto every segment so the dubbed voice sounds like the professor.

    Falls back to plain Sarvam output if:
      • OpenVoice v2 is not installed, OR
      • no voice reference audio is available, OR
      • the reference clip cannot be extracted.
    """
    # Step 1: Sarvam TTS (all segments)
    sarvam_files = _synth_segments_sarvam(
        segments, lang_code, out_dir,
        preserve_fillers=preserve_fillers,
        original_segments=original_segments,
    )

    if not sarvam_files:
        return sarvam_files

    # Step 2: Voice conversion
    if not voice_reference_audio or not os.path.exists(voice_reference_audio):
        logger.warning("[sarvam_vc] No voice reference — returning plain Sarvam output")
        return sarvam_files

    try:
        from src.voice_converter import (
            convert_segments_batch,
            extract_reference_clip,
            is_openvoice_available,
        )
    except ImportError:
        logger.warning("[sarvam_vc] voice_converter not importable — returning plain Sarvam output")
        return sarvam_files

    if not is_openvoice_available():
        logger.warning(
            "[sarvam_vc] OpenVoice v2 not installed — returning plain Sarvam output.\n"
            "Install with: pip install git+https://github.com/myshell-ai/OpenVoice.git"
        )
        return sarvam_files

    # Extract a clean reference clip from the original lecture audio
    ref_path = os.path.join(out_dir, "_vc_reference.wav")
    try:
        extract_reference_clip(voice_reference_audio, ref_path)
    except Exception as exc:
        logger.error("[sarvam_vc] Reference clip extraction failed: %s — skipping voice clone", exc)
        return sarvam_files

    try:
        import torch as _torch
        vc_device = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        vc_device = "cpu"

    vc_dir = os.path.join(out_dir, "vc_converted")
    os.makedirs(vc_dir, exist_ok=True)

    logger.info("[sarvam_vc] Applying voice cloning to %d segments (device=%s)…", len(sarvam_files), vc_device)
    converted = convert_segments_batch(
        sarvam_files, ref_path, vc_dir, device=vc_device,
    )
    logger.info("[sarvam_vc] Voice cloning complete: %d/%d segments converted", len(converted), len(sarvam_files))
    return converted


# ── Edge TTS implementation ──────────────────────────────────────────────────

def _generate_edge_tts(
    segments: list[Segment], lang_code: str, out_path: str
) -> str | None:
    """
    Generate speech using edge-tts (Microsoft Edge neural voices).
    Synthesises each segment individually then concatenates with ffmpeg.
    """
    voice = EDGE_TTS_VOICES.get(lang_code)
    if not voice:
        logger.warning("No edge-tts voice configured for '%s'", lang_code)
        return None

    lang_name = TARGET_LANGUAGES[lang_code]["name"]

    # Create temp dir for segment audio files
    tmp_dir = tempfile.mkdtemp(prefix="tts_edge_")
    segment_files: list[str] = []

    async def _synthesize_all():
        for idx, seg in enumerate(segments):
            if not seg.text.strip():
                continue
            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp3")
            communicate = edge_tts.Communicate(seg.text, voice)
            await communicate.save(seg_path)
            segment_files.append(seg_path)
            if (idx + 1) % 10 == 0:
                logger.info("  %s: %d/%d segments", lang_name, idx + 1, len(segments))

    # Run the async synthesis
    asyncio.run(_synthesize_all())

    if not segment_files:
        logger.warning("No segments produced for %s", lang_name)
        return None

    logger.info("%s: Synthesized %d segments, concatenating…",
                lang_name, len(segment_files))

    # Concatenate with ffmpeg
    list_file = os.path.join(tmp_dir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Clean up temp files
    for sf in segment_files:
        try:
            os.remove(sf)
        except OSError:
            pass
    try:
        os.remove(list_file)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    size_kb = os.path.getsize(out_path) / 1024
    logger.info("%s: Done → %s (%.0f KB)", lang_name, out_path, size_kb)
    return out_path


# ── Sarvam TTS implementation ─────────────────────────────────────────────────

# Default Sarvam speaker per language (must be compatible with bulbul:v3)
SARVAM_SPEAKERS: dict[str, str] = {
    "hi": "priya",
    "te": "kavitha",
    "od": "priya",
}


def _generate_sarvam_tts(
    segments: list[Segment], lang_code: str, out_path: str
) -> str | None:
    """
    Generate speech using Sarvam Bulbul v3 TTS.

    Each segment is synthesised individually.  The SDK returns base64-encoded
    WAV audio in `response.audios`.  We decode, save each segment, then
    concatenate with ffmpeg – same strategy as edge-tts.
    """
    import base64
    from sarvamai import SarvamAI

    if not SARVAM_API_KEY:
        logger.warning("Sarvam API key not set, skipping")
        return None

    lang_info = TARGET_LANGUAGES[lang_code]
    lang_name = lang_info["name"]
    sarvam_code = lang_info["sarvam_code"]  # e.g. "hi-IN"
    speaker = SARVAM_SPEAKERS.get(lang_code, "anushka")

    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    tmp_dir = tempfile.mkdtemp(prefix="tts_sarvam_")
    segment_files: list[str] = []

    for idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        # Sarvam has a 2500 char limit; truncate if needed
        if len(text) > 2500:
            text = text[:2500]

        try:
            resp = client.text_to_speech.convert(
                text=text,
                target_language_code=sarvam_code,
                speaker=speaker,
                model="bulbul:v3",
                output_audio_codec="mp3",
            )

            if resp.audios:
                audio_bytes = base64.b64decode(resp.audios[0])
                seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp3")
                with open(seg_path, "wb") as f:
                    f.write(audio_bytes)
                segment_files.append(seg_path)

        except Exception as exc:
            logger.warning("Sarvam TTS error on segment %d: %s", idx + 1, exc)

        if (idx + 1) % 10 == 0:
            logger.info("  %s: %d/%d segments", lang_name, idx + 1, len(segments))

    if not segment_files:
        logger.warning("No segments produced for %s", lang_name)
        return None

    logger.info("%s: Synthesized %d segments, concatenating…",
                lang_name, len(segment_files))

    # Concatenate with ffmpeg
    list_file = os.path.join(tmp_dir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Clean up temp files
    for sf in segment_files:
        try:
            os.remove(sf)
        except OSError:
            pass
    try:
        os.remove(list_file)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    size_kb = os.path.getsize(out_path) / 1024
    logger.info("%s: Done → %s (%.0f KB)", lang_name, out_path, size_kb)
    return out_path


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────

def _get_elevenlabs_client():
    from elevenlabs.client import ElevenLabs
    from config.settings import ELEVENLABS_API_KEY
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not set")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)

def _generate_elevenlabs_tts(
    segments: list[Segment], lang_code: str, out_path: str
) -> str | None:
    """Concatenated synthesis for legacy modes."""
    client = _get_elevenlabs_client()
    text = " ".join(s.text for s in segments)
    
    # Defaulting to a pre-trained voice if custom voice cloning wasn't run
    voice_id = "pNInz6obpgDQGcFmaJgB" # Adam (Works fairly well multilingually)
    
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_44100_192",
            text=text,
            model_id="eleven_multilingual_v2"
        )
        with open(out_path, "wb") as w:
            for chunk in audio:
                if chunk:
                    w.write(chunk)
        return out_path
    except Exception as e:
        logger.error(f"ElevenLabs TTS failed: {e}")
        return None

def _synth_segments_elevenlabs(
    segments: list[Segment],
    lang_code: str,
    out_dir: str,
    preserve_fillers: bool = True,
) -> dict[int, str]:
    """Synthesise segments via ElevenLabs TTS."""
    segment_files: dict[int, str] = {}
    client = _get_elevenlabs_client()
    
    # In a full flow, you would dynamically create a Voice using the 'vocals.wav'
    # For now, using a neutral stable voice for Multilingual V2
    voice_id = "pNInz6obpgDQGcFmaJgB"

    # Synchronous sequential generation to avoid strict rate limits on basic tier
    for idx, seg in enumerate(segments):
        out_path = os.path.join(out_dir, f"segment_{idx:03d}.mp3")
        text = seg.text if preserve_fillers else get_text_without_fillers(seg)
        if not text.strip():
            continue
            
        try:
            logger.debug(f"ElevenLabs generating {idx}...")
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                output_format="mp3_44100_192",
                text=text,
                model_id="eleven_multilingual_v2"
            )
            with open(out_path, "wb") as w:
                for chunk in audio:
                    if chunk:
                        w.write(chunk)
            segment_files[idx] = out_path
        except Exception as e:
            logger.error(f"ElevenLabs failed for segment {idx}: {e}")
            
    return segment_files


# ── Coqui XTTS v2 ─────────────────────────────────────────────────────────────

# XTTS v2 natively accepts a list of reference audio paths in
# get_conditioning_latents(), computing speaker embeddings per clip and merging
# them.  We exploit this to give the model a much richer view of the lecturer's
# voice than a single fixed clip.
#
# Reference clip selection strategy
# ──────────────────────────────────
# 1. Skip the first / last N seconds of the lecture (titles, applause, music).
# 2. Divide the viable timeline into NUM_REF_CLIPS equal intervals.
# 3. From each interval, score every segment by:
#      score = duration × speech_energy_ratio
#    where speech_energy_ratio = RMS of segment / median RMS of all segments
#    (penalises quiet or noisy stretches).
# 4. Pick the best-scoring segment per interval.
# 5. Extract a clip of exactly CLIP_DURATION seconds, centred on the segment,
#    resampled to 22050 Hz mono (XTTS native sample rate).
# 6. Apply mild noise gate / silence trim so the clip starts and ends on speech.

_XTTS_REF_CLIPS   = 5       # number of reference clips to build
_XTTS_CLIP_SECS   = 10.0    # target duration of each clip (seconds)
_XTTS_MIN_SEG_DUR = 4.0     # ignore segments shorter than this
_XTTS_SKIP_HEAD   = 45.0    # skip first N seconds (intro / title card)
_XTTS_SKIP_TAIL   = 30.0    # skip last N seconds


def _rms_energy(audio_path: str, start: float, duration: float) -> float:
    """
    Return the mean RMS energy of a time window inside *audio_path*.
    Uses ffmpeg to decode a short chunk to raw PCM, then computes RMS.
    Returns 0.0 on any error.
    """
    import struct
    import subprocess as _sp

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", audio_path,
            "-f", "s16le", "-ac", "1", "-ar", "16000",
            "pipe:1",
        ]
        out = _sp.run(cmd, capture_output=True, check=True).stdout
        if len(out) < 2:
            return 0.0
        n = len(out) // 2
        samples = struct.unpack(f"<{n}h", out[: n * 2])
        rms = (sum(s * s for s in samples) / n) ** 0.5
        return rms
    except Exception:
        return 0.0


def _extract_reference_clip(
    audio_path: str,
    start: float,
    duration: float,
    out_path: str,
) -> str:
    """Extract a mono 22050 Hz WAV clip from *audio_path*."""
    import subprocess as _sp

    _sp.run(
        [
            "ffmpeg", "-y",
            "-ss", str(max(0.0, start)),
            "-t", str(duration),
            "-i", audio_path,
            "-ac", "1", "-ar", "22050",
            out_path,
        ],
        capture_output=True, check=True,
    )
    return out_path


def _clip_is_valid(path: str, min_duration: float = 1.0) -> bool:
    """Return True if the WAV file exists, is non-empty, and has audio content."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1024:   # < 1 KB → certainly empty/corrupt
        return False
    try:
        import subprocess as _sp
        out = _sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        dur = float(out.stdout.strip())
        return dur >= min_duration
    except Exception:
        return False


def build_xtts_reference_clips(
    segments: list[Segment],
    audio_path: str,
    cache_dir: str,
    *,
    num_clips: int = _XTTS_REF_CLIPS,
    clip_duration: float = _XTTS_CLIP_SECS,
    min_seg_dur: float = _XTTS_MIN_SEG_DUR,
    skip_head: float = _XTTS_SKIP_HEAD,
    skip_tail: float = _XTTS_SKIP_TAIL,
) -> list[str]:
    """
    Build a diverse set of XTTS speaker-reference clips from the lecture audio.

    Returns a list of paths to short WAV clips, spread across the lecture and
    filtered for speech energy, suitable for passing directly to
    ``tts.tts_to_file(speaker_wav=clips)``.

    Falls back to a single clip near the start if no suitable segments are found.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # ── Check cached clips from a previous run ────────────────────────
    # Only reuse if every cached file is valid (non-empty, has real audio).
    cached = sorted(
        f for f in os.listdir(cache_dir)
        if f.startswith("xtts_ref_") and f.endswith(".wav")
    )
    if len(cached) >= 2:
        paths = [os.path.join(cache_dir, f) for f in cached]
        valid_paths = [p for p in paths if _clip_is_valid(p)]
        if len(valid_paths) >= 2:
            logger.info("[XTTS] Using %d valid cached reference clips", len(valid_paths))
            return valid_paths
        else:
            logger.warning(
                "[XTTS] Cached reference clips are corrupt/empty — re-extracting"
            )
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ── Adapt skip margins to actual video length ─────────────────────
    # For short videos (demo clips etc.) the fixed 45s/30s margins would
    # exclude the entire lecture.  Scale them to at most 15% of total_dur.
    if segments:
        total_dur = segments[-1].end
    else:
        total_dur = 0.0

    if total_dur > 0:
        skip_head = min(skip_head, total_dur * 0.15)
        skip_tail = min(skip_tail, total_dur * 0.10)

    viable = [
        s for s in segments
        if (s.end - s.start) >= min_seg_dur
        and s.start >= skip_head
        and s.end <= max(total_dur - skip_tail, skip_head + min_seg_dur)
    ]

    if not viable:
        # No usable segments — fall back to first real speech in the file
        logger.warning(
            "[XTTS] No segments meet reference-clip criteria "
            "(total_dur=%.1fs, skip_head=%.1fs, skip_tail=%.1fs); "
            "using first available segment",
            total_dur, skip_head, skip_tail,
        )
        # Use the longest segment regardless of position
        fallback_seg = max(segments, key=lambda s: s.end - s.start) if segments else None
        fallback_start = fallback_seg.start if fallback_seg else 0.0
        fallback_dur   = min(clip_duration, fallback_seg.end - fallback_seg.start if fallback_seg else clip_duration)
        out = os.path.join(cache_dir, "xtts_ref_000.wav")
        _extract_reference_clip(audio_path, fallback_start, fallback_dur, out)
        if not _clip_is_valid(out):
            raise RuntimeError(
                f"[XTTS] Could not extract any valid reference clip from {audio_path}. "
                "Check that the audio file has speech content."
            )
        return [out]

    # ── Compute RMS for every candidate ──────────────────────────────
    energies = []
    for seg in viable:
        rms = _rms_energy(audio_path, seg.start, min(seg.end - seg.start, 8.0))
        energies.append(rms)

    median_rms = sorted(energies)[len(energies) // 2] or 1.0

    # Scored: longer segments with higher-than-median energy score better
    scored = [
        (seg, (seg.end - seg.start) * (rms / median_rms))
        for seg, rms in zip(viable, energies)
    ]

    # ── Divide timeline into intervals and pick best per interval ─────
    t_start = viable[0].start
    t_end   = viable[-1].end
    interval = (t_end - t_start) / num_clips if num_clips > 0 else (t_end - t_start)

    clips: list[str] = []
    seen_starts: set[float] = set()

    for i in range(num_clips):
        lo = t_start + i * interval
        hi = lo + interval

        # Segments whose midpoint falls within this interval
        candidates = [
            (seg, score)
            for seg, score in scored
            if lo <= (seg.start + seg.end) / 2 < hi
            and seg.start not in seen_starts
        ]

        if not candidates:
            continue

        best_seg, best_score = max(candidates, key=lambda x: x[1])
        seen_starts.add(best_seg.start)

        # Centre the clip on the segment; pad to clip_duration
        seg_dur    = best_seg.end - best_seg.start
        clip_start = best_seg.start + max(0.0, (seg_dur - clip_duration) / 2)
        clip_actual = min(clip_duration, seg_dur)

        out_path = os.path.join(cache_dir, f"xtts_ref_{i:03d}.wav")
        try:
            _extract_reference_clip(audio_path, clip_start, clip_actual, out_path)
            if _clip_is_valid(out_path):
                clips.append(out_path)
                logger.debug(
                    "[XTTS] Ref clip %d: t=%.1f–%.1f  score=%.0f  → %s",
                    i, clip_start, clip_start + clip_actual,
                    best_score, os.path.basename(out_path),
                )
            else:
                logger.warning("[XTTS] Ref clip %d extracted but is empty/silent, skipping", i)
        except Exception as exc:
            logger.warning("[XTTS] Could not extract ref clip %d: %s", i, exc)

    if not clips:
        # Absolute fallback — use the highest-scoring segment
        best_seg = max(scored, key=lambda x: x[1])[0]
        out = os.path.join(cache_dir, "xtts_ref_000.wav")
        _extract_reference_clip(audio_path, best_seg.start,
                                min(clip_duration, best_seg.end - best_seg.start), out)
        if _clip_is_valid(out):
            clips = [out]
        else:
            raise RuntimeError(
                f"[XTTS] All reference clip extractions failed for {audio_path}."
            )

    logger.info(
        "[XTTS] Built %d reference clip(s) for speaker conditioning "
        "(total lecture %.0fs, margins head=%.0fs tail=%.0fs)",
        len(clips), total_dur, skip_head, skip_tail,
    )
    return clips


# ── Google Cloud TTS implementation ──────────────────────────────────────────

# Best Google Cloud TTS voices per language.
# Neural2 voices: same WaveNet architecture as YouTube uses for auto-dubbing.
# Falls back to Standard when Neural2/Studio unavailable for a language.
# Odia (or-IN) is not yet supported by Cloud TTS → fall through to Sarvam.
_GCP_TTS_VOICES: dict[str, dict] = {
    #  lang_code  : { bcp47 , male voice              , female voice             }
    "hi": {"bcp47": "hi-IN",  "male": "hi-IN-Neural2-B",  "female": "hi-IN-Neural2-C"},
    "te": {"bcp47": "te-IN",  "male": "te-IN-Standard-B", "female": "te-IN-Standard-A"},
    # "od" intentionally omitted — Cloud TTS does not support Odia
}

# Characters-per-second used for SSML rate estimation
_GCP_NATURAL_CPS: dict[str, float] = {
    "hi": 12.0,
    "te": 10.0,
}

# SSML prosody rate bounds (as percentage of natural speed, e.g. "80%" or "140%")
_GCP_MIN_RATE_PCT = 65
_GCP_MAX_RATE_PCT = 175

_GCP_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


def _synth_segments_gcptts(
    segments: list[Segment],
    lang_code: str,
    out_dir: str,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
    voice_profile: "VoiceProfile | None" = None,
) -> dict[int, str]:
    """
    Synthesise segments via Google Cloud TTS (Neural2/Studio voices).

    Uses the REST API directly — no SDK dependency, just requests + GCP_API_KEY.

    Highlights:
    • Neural2 voices for Hindi (same model family YouTube auto-dubbing uses)
    • SSML <prosody rate="X%"> for per-segment duration-aware speed control
    • Falls back to Sarvam TTS for Odia (not yet in Cloud TTS)
    • Falls back to Edge TTS when GCP_API_KEY is missing

    Returns {segment_idx: audio_path} mapping.
    """
    if not GCP_API_KEY:
        logger.warning("[GCP TTS] GCP_API_KEY not set — falling back to Edge TTS")
        return _synth_segments_edge(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
            voice_profile=voice_profile,
        )

    voice_info = _GCP_TTS_VOICES.get(lang_code)
    if voice_info is None:
        # Language not supported by Cloud TTS (e.g. Odia) → use Sarvam
        logger.info(
            "[GCP TTS] '%s' not supported — delegating to Sarvam TTS", lang_code
        )
        return _synth_segments_sarvam(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )

    # Choose male/female voice based on voice profile
    if voice_profile and voice_profile.is_male:
        voice_name = voice_info["male"]
    else:
        voice_name = voice_info["female"]

    bcp47 = voice_info["bcp47"]
    natural_cps = _GCP_NATURAL_CPS.get(lang_code, 11.0)
    lang_name = TARGET_LANGUAGES[lang_code]["name"]

    # Build per-segment target durations
    target_durations: dict[int, float] = {}
    if original_segments and len(original_segments) == len(segments):
        for i, oseg in enumerate(original_segments):
            d = oseg.end - oseg.start
            if d > 0:
                target_durations[i] = d

    logger.info("[GCP TTS] %s — voice: %s, %d segments", lang_name, voice_name, len(segments))

    url = f"{_GCP_TTS_URL}?key={GCP_API_KEY}"
    files: dict[int, str] = {}
    failed = 0

    for idx, seg in enumerate(segments):
        text = seg.text if preserve_fillers else get_text_without_fillers(seg)
        text = text.strip()
        if not text:
            continue

        # Compute <prosody rate> to hit the target duration
        if idx in target_durations:
            natural_dur = max(len(text) / natural_cps, 0.5)
            raw_rate = (natural_dur / target_durations[idx]) * 100.0
            rate_pct = int(max(_GCP_MIN_RATE_PCT, min(_GCP_MAX_RATE_PCT, raw_rate)))
        else:
            rate_pct = 100

        # Escape XML special chars for SSML
        ssml_text = (text
                     .replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;")
                     .replace('"', "&quot;"))
        ssml = f'<speak><prosody rate="{rate_pct}%">{ssml_text}</prosody></speak>'

        payload = {
            "input": {"ssml": ssml},
            "voice": {"languageCode": bcp47, "name": voice_name},
            "audioConfig": {"audioEncoding": "MP3", "sampleRateHertz": 24000},
        }

        try:
            import requests as _req
            import time as _time

            last_exc = None
            data = None
            for _attempt in range(3):
                try:
                    resp = _req.post(url, json=payload, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as _e:
                    last_exc = _e
                    if _attempt < 2:
                        _time.sleep(2 ** _attempt)
            if data is None:
                raise last_exc
            import base64 as _b64
            audio_bytes = _b64.b64decode(data["audioContent"])
            seg_path = os.path.join(out_dir, f"gcpseg_{idx:04d}.mp3")
            with open(seg_path, "wb") as fh:
                fh.write(audio_bytes)
            files[idx] = seg_path
            logger.debug("[GCP TTS] Seg %d: rate=%d%% target=%.1fs text='%s…'",
                         idx, rate_pct, target_durations.get(idx, 0), text[:40])
        except Exception as exc:
            logger.warning("[GCP TTS] Seg %d failed: %s", idx, exc)
            failed += 1

        if (idx + 1) % 20 == 0:
            logger.info("[GCP TTS] %s: %d/%d segments", lang_name, idx + 1, len(segments))

    logger.info("[GCP TTS] %s: %d/%d OK, %d failed", lang_name, len(files), len(segments), failed)
    return files


def _synth_segments_gcptts_vc(
    segments: list[Segment],
    lang_code: str,
    out_dir: str,
    preserve_fillers: bool = True,
    original_segments: list[Segment] | None = None,
    voice_reference_audio: str | None = None,
    voice_profile: "VoiceProfile | None" = None,
) -> dict[int, str]:
    """
    Combined engine: Google Cloud TTS Neural2 → OpenVoice v2 voice cloning.

    Step 1 — Cloud TTS Neural2: clear, high-quality Indic speech with
              duration-aware SSML rate control.
    Step 2 — OpenVoice v2: transfers the original lecturer's timbre onto
              every segment.

    Falls back to plain Cloud TTS if OpenVoice is unavailable or the
    reference clip cannot be extracted.
    """
    # Step 1: GCP TTS
    gcp_files = _synth_segments_gcptts(
        segments, lang_code, out_dir,
        preserve_fillers=preserve_fillers,
        original_segments=original_segments,
        voice_profile=voice_profile,
    )

    if not gcp_files:
        return gcp_files

    # Step 2: Voice conversion
    if not voice_reference_audio or not os.path.exists(voice_reference_audio):
        logger.warning("[gcptts_vc] No voice reference — returning plain GCP TTS output")
        return gcp_files

    try:
        from src.voice_converter import (
            convert_segments_batch,
            extract_reference_clip,
            is_openvoice_available,
        )
    except ImportError:
        logger.warning("[gcptts_vc] voice_converter not importable — returning plain GCP TTS output")
        return gcp_files

    if not is_openvoice_available():
        logger.warning("[gcptts_vc] OpenVoice v2 not installed — returning plain GCP TTS output")
        return gcp_files

    ref_path = os.path.join(out_dir, "_gcptts_vc_reference.wav")
    try:
        extract_reference_clip(voice_reference_audio, ref_path)
    except Exception as exc:
        logger.error("[gcptts_vc] Reference clip extraction failed: %s — skipping voice clone", exc)
        return gcp_files

    try:
        import torch as _torch
        vc_device = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        vc_device = "cpu"

    vc_dir = os.path.join(out_dir, "vc_converted_gcp")
    os.makedirs(vc_dir, exist_ok=True)

    logger.info("[gcptts_vc] Applying voice cloning to %d segments (device=%s)…",
                len(gcp_files), vc_device)
    converted = convert_segments_batch(gcp_files, ref_path, vc_dir, device=vc_device)
    logger.info("[gcptts_vc] Done: %d/%d segments converted", len(converted), len(gcp_files))
    return converted


XTTS_SUPPORTED_LANGS = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
    "zh-cn", "hu", "ko", "ja", "hi"
]

def _synth_segments_xtts(
    segments: list[Segment],
    lang_code: str,
    out_dir: str,
    preserve_fillers: bool = True,
    voice_reference_audio: str | None = None,
    original_segments: list[Segment] | None = None,
) -> dict[int, str]:
    """
    Synthesise segments via XTTS v2 with multi-clip speaker conditioning.
    Uses low-level model API so conditioning latents are computed ONCE
    (faster + richer: gpt_cond_len=30, max_ref_length=60).
    """
    segment_files: dict[int, str] = {}

    if lang_code not in XTTS_SUPPORTED_LANGS:
        logger.warning(
            "[XTTS] Language '%s' is not supported by XTTS v2. Falling back to Edge TTS.",
            lang_code,
        )
        return _synth_segments_edge(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )

    if not voice_reference_audio or not os.path.exists(voice_reference_audio):
        logger.warning("[XTTS] No voice reference provided. Falling back to Edge TTS.")
        return _synth_segments_edge(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )

    try:
        from TTS.api import TTS
        import torch
        import torchaudio
    except ImportError:
        logger.error("Coqui TTS library is not installed. Run: pip install TTS")
        return {}

    # ── Build diverse reference clips ────────────────────────────────
    timing_segs = original_segments if original_segments else segments
    ref_clips_dir = os.path.join(out_dir, "xtts_ref_clips")
    ref_clips = build_xtts_reference_clips(timing_segs, voice_reference_audio, ref_clips_dir)
    logger.info(
        "[XTTS] Speaker conditioning: %d clip(s) → %s",
        len(ref_clips), [os.path.basename(p) for p in ref_clips],
    )

    # ── Load model ────────────────────────────────────────────────────
    logger.info("[XTTS] Loading multilingual XTTS v2 model…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    xtts_model = tts.synthesizer.tts_model

    # ── Compute speaker conditioning ONCE from all reference clips ────
    logger.info("[XTTS] Computing speaker embedding (gpt_cond_len=30, max_ref_length=60)…")
    try:
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=ref_clips,
            gpt_cond_len=30,
            gpt_cond_chunk_len=6,
            max_ref_length=60,
        )
    except Exception as exc:
        logger.error("[XTTS] Failed to compute speaker embedding: %s", exc)
        del tts, xtts_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _synth_segments_edge(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
            original_segments=original_segments,
        )

    # ── Synthesise each segment ───────────────────────────────────────
    CPS = 14.0  # approximate Hindi characters per second for speed estimation
    failed = 0
    try:
        for idx, seg in enumerate(segments):
            out_path = os.path.join(out_dir, f"segment_{idx:03d}.wav")
            text = seg.text if preserve_fillers else get_text_without_fillers(seg)
            text = text.replace(".", " ").strip()
            if not text:
                continue

            target_dur = seg.end - seg.start
            if target_dur > 0:
                estimated_natural_dur = max(len(text) / CPS, 1.0)
                target_speed = max(0.65, min(1.8, estimated_natural_dur / target_dur))
            else:
                target_speed = 1.0

            try:
                logger.debug(
                    "[XTTS] Segment %d/%d: '%s…' (speed=%.2f)",
                    idx, len(segments) - 1, text[:50], target_speed,
                )
                out_dict = xtts_model.inference(
                    text=text,
                    language=lang_code,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.60,
                    repetition_penalty=10.0,
                    top_k=50,
                    top_p=0.85,
                    speed=target_speed,
                    enable_text_splitting=False,
                )
                wav_tensor = torch.tensor(out_dict["wav"]).unsqueeze(0)
                torchaudio.save(out_path, wav_tensor, 24000)
                if _clip_is_valid(out_path, min_duration=0.05):
                    segment_files[idx] = out_path
                else:
                    logger.warning("[XTTS] Segment %d produced empty/silent output", idx)
                    failed += 1
            except Exception as e:
                logger.error("[XTTS] Failed segment %d ('%s…'): %s", idx, text[:40], e)
                failed += 1

    finally:
        # ── GPU cleanup ───────────────────────────────────────────────
        try:
            tts.to("cpu")
        except Exception:
            pass
        del tts, xtts_model, gpt_cond_latent, speaker_embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[XTTS] GPU memory released after synthesis")

    total = len(segments)
    logger.info("[XTTS] Synthesis complete: %d/%d OK, %d failed", len(segment_files), total, failed)
    if len(segment_files) == 0:
        logger.error("[XTTS] ALL segments failed — dubbed audio will be SILENT!")
    elif failed > total * 0.3:
        logger.warning("[XTTS] >30%% of segments failed (%d/%d)", failed, total)
    return segment_files

def _generate_xtts(
    segments: list[Segment], lang_code: str, out_path: str
) -> str | None:
    return None
