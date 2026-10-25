"""
tts_generator.py
────────────────
Text-to-Speech: convert translated text segments into audio files
in Hindi, Telugu, and Odia.

Supported TTS engines:
  • sarvam     — Sarvam bulbul:v3 (requires SARVAM_API_KEY)
  • gcptts     — Google Cloud TTS Neural2 (requires GCP_API_KEY)
  • xtts       — Coqui XTTS v2 (local GPU)

"sarvam_vc" / "gcptts_vc" are the same engines followed by OpenVoice v2
voice cloning; the pipeline runs that cloning as its own GPU step (see
split_tts_engine), so this module only synthesises speech.
"""

from __future__ import annotations
import logging
import os
from config.settings import (
    SARVAM_API_KEY,
    GCP_API_KEY,
    TARGET_LANGUAGES,
)
from src.transcriber import Segment
from src.voice_analyzer import VoiceProfile
from src.filler_detector import get_text_without_fillers, get_text_with_target_fillers
from src.utils import file_sha256

logger = logging.getLogger("nptel_pipeline")


# ── TTS engine registry ─────────────────────────────────────────────────────
TTS_ENGINE_REGISTRY: dict[str, dict] = {
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
    for preferred in ("sarvam_vc", "gcptts_vc", "sarvam", "gcptts", "xtts"):
        if preferred in available:
            return preferred
    return "sarvam"


def split_tts_engine(engine: str) -> tuple[str, bool]:
    """
    Split a registry engine id into (synthesis engine, voice cloning wanted).

    "sarvam_vc" → ("sarvam", True);  "gcptts" → ("gcptts", False).
    """
    if engine.endswith("_vc"):
        return engine[: -len("_vc")], True
    return engine, False


def is_tts_available() -> bool:
    """Check if any TTS engine is usable."""
    return len(get_available_tts_engines()) > 0


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
        engine:            TTS engine id ("sarvam", "gcptts", "xtts"; a "_vc"
                           suffix is ignored here — cloning is a separate step).
        voice_profile:     Picks a voice matching the speaker's gender.
        preserve_fillers:  If True, remove fillers from text before TTS.
        original_segments: English segments with original timing (XTTS picks
                           its speaker-reference clips from them).  Speech is
                           always synthesised at one natural pace; fitting it
                           to the timing is the aligner's job.
    """
    if engine is None:
        engine = get_default_tts_engine()
    engine, _ = split_tts_engine(engine)

    seg_dir = os.path.join(output_dir, f"tts_segments_{lang_code}")
    os.makedirs(seg_dir, exist_ok=True)

    lang_name = TARGET_LANGUAGES[lang_code]["name"]
    segment_files: dict[int, str] = {}

    if engine == "sarvam":
        segment_files = _synth_segments_sarvam(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            voice_profile=voice_profile,
        )
    elif engine == "gcptts":
        segment_files = _synth_segments_gcptts(
            segments, lang_code, seg_dir,
            preserve_fillers=preserve_fillers,
            voice_profile=voice_profile,
        )
        if not segment_files:
            logger.warning("[GCP TTS] All segments failed — falling back to sarvam engine")
            segment_files = _synth_segments_sarvam(
                segments, lang_code, seg_dir,
                preserve_fillers=preserve_fillers,
                voice_profile=voice_profile,
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


# ── Per-segment sarvam TTS ───────────────────────────────────────────────────

# One pace for every sentence of a lecture.  Per-sentence pace targeting made
# the heard speaking rate swing ~2x between sentences; timing is now handled
# by the aligner (pauses + small tempo nudges) instead.
SARVAM_PACE = 1.0
SARVAM_SAMPLE_RATE = 48_000


def _synth_segments_sarvam(
    segments: list[Segment], lang_code: str, out_dir: str,
    preserve_fillers: bool = True,
    voice_profile: VoiceProfile | None = None,
) -> dict[int, str]:
    """
    Synthesise each segment with Sarvam Bulbul v3, returning index → file map.

    Every sentence uses the same natural pace and comes back as lossless
    48 kHz WAV.  A male speaker voice is used when *voice_profile* says the
    lecturer is male.
    """
    import base64
    from sarvamai import SarvamAI

    if not SARVAM_API_KEY:
        logger.warning("Sarvam API key not set")
        return {}

    lang_info = TARGET_LANGUAGES[lang_code]
    lang_name = lang_info["name"]
    speaker = _select_sarvam_speaker(lang_code, voice_profile)
    logger.info("[Sarvam] %s voice: %s, pace %.2f", lang_name, speaker, SARVAM_PACE)

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

        try:
            resp = client.text_to_speech.convert(
                text=text,
                target_language_code=lang_info["sarvam_code"],
                speaker=speaker,
                model="bulbul:v3",
                pace=SARVAM_PACE,
                speech_sample_rate=SARVAM_SAMPLE_RATE,
                output_audio_codec="wav",
            )
            if resp.audios:
                seg_path = os.path.join(out_dir, f"seg_{idx:04d}.wav")
                with open(seg_path, "wb") as f:
                    f.write(base64.b64decode(resp.audios[0]))
                files[idx] = seg_path
        except Exception as exc:
            logger.warning("Sarvam TTS error on segment %d: %s", idx + 1, exc)

        if (idx + 1) % 10 == 0:
            logger.info("  %s: %d/%d segments", lang_name, idx + 1, len(segments))

    return files


# ── Sarvam TTS implementation ─────────────────────────────────────────────────

# Default Sarvam speaker per language (must be compatible with bulbul:v3)
SARVAM_SPEAKERS: dict[str, str] = {
    "hi": "priya",
    "te": "kavitha",
    "od": "priya",
}

# Male bulbul:v3 speakers, used when the lecturer is male.  "rahul" over the
# v3 default "aditya": measured on lecture dubs, aditya is very deep (~105 Hz)
# with 10–15 % creaky voice, which sounds rough; rahul (~130–145 Hz in hi/te/od)
# had almost none and the cleanest harmonics-to-noise ratio of the male voices.
SARVAM_SPEAKERS_MALE: dict[str, str] = {
    "hi": "rahul",
    "te": "rahul",
    "od": "rahul",
}


def _select_sarvam_speaker(lang_code: str, voice_profile: VoiceProfile | None = None) -> str:
    """Pick a Sarvam speaker whose gender matches the original lecturer."""
    if voice_profile and voice_profile.is_male:
        return SARVAM_SPEAKERS_MALE.get(lang_code, "rahul")
    return SARVAM_SPEAKERS.get(lang_code, "priya")


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

    # ── Discard clips cut from a different lecture ────────────────────
    # The job folder can be reused for a new video under the same name.
    source_id = file_sha256(audio_path)
    marker = os.path.join(cache_dir, "source.sha256")
    try:
        with open(marker, encoding="utf-8") as f:
            same_source = f.read().strip() == source_id
    except OSError:
        same_source = False
    if not same_source:
        for f in os.listdir(cache_dir):
            if f.startswith("xtts_ref_") and f.endswith(".wav"):
                os.remove(os.path.join(cache_dir, f))
        with open(marker, "w", encoding="utf-8") as f:
            f.write(source_id)

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
    "hi": {"bcp47": "hi-IN",  "male": "hi-IN-Neural2-B",  "female": "hi-IN-Neural2-A"},
    "te": {"bcp47": "te-IN",  "male": "te-IN-Standard-B", "female": "te-IN-Standard-A"},
    # "od" intentionally omitted — Cloud TTS does not support Odia
}

_GCP_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


def _synth_segments_gcptts(
    segments: list[Segment],
    lang_code: str,
    out_dir: str,
    preserve_fillers: bool = True,
    voice_profile: "VoiceProfile | None" = None,
) -> dict[int, str]:
    """
    Synthesise segments via Google Cloud TTS (Neural2/Standard voices).

    Uses the REST API directly — no SDK dependency, just requests + GCP_API_KEY.

    Highlights:
    • Neural2 voice for Hindi, Standard voice for Telugu, matched to the
      lecturer's gender
    • One natural speaking rate for every sentence, uncompressed 24 kHz WAV
    • Falls back to Sarvam TTS for Odia (not yet in Cloud TTS)

    Returns {segment_idx: audio_path} mapping.
    """
    if not GCP_API_KEY:
        logger.error("[GCP TTS] GCP_API_KEY is not set in .env")
        return {}

    voice_info = _GCP_TTS_VOICES.get(lang_code)
    if voice_info is None:
        # Language not supported by Cloud TTS (e.g. Odia) → use Sarvam
        logger.info(
            "[GCP TTS] '%s' not supported — delegating to Sarvam TTS", lang_code
        )
        return _synth_segments_sarvam(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
            voice_profile=voice_profile,
        )

    # Choose male/female voice based on voice profile
    if voice_profile and voice_profile.is_male:
        voice_name = voice_info["male"]
    else:
        voice_name = voice_info["female"]

    bcp47 = voice_info["bcp47"]
    lang_name = TARGET_LANGUAGES[lang_code]["name"]
    logger.info("[GCP TTS] %s — voice: %s, %d segments", lang_name, voice_name, len(segments))

    import base64 as _b64
    import time as _time
    import requests as _req

    url = f"{_GCP_TTS_URL}?key={GCP_API_KEY}"

    def _gcp_synth(text: str) -> bytes:
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": bcp47, "name": voice_name},
            # Neural2's native rate, uncompressed (LINEAR16 comes with a WAV header)
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
        }
        last_exc = None
        for attempt in range(3):
            try:
                r = _req.post(url, json=payload, timeout=20)
                r.raise_for_status()
                return _b64.b64decode(r.json()["audioContent"])
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    _time.sleep(2 ** attempt)
        raise last_exc

    files: dict[int, str] = {}
    failed = 0
    for idx, seg in enumerate(segments):
        text = seg.text if preserve_fillers else get_text_without_fillers(seg.text)
        text = text.strip()
        if not text:
            continue
        try:
            seg_path = os.path.join(out_dir, f"gcpseg_{idx:04d}.wav")
            with open(seg_path, "wb") as fh:
                fh.write(_gcp_synth(text))
            files[idx] = seg_path
        except Exception as exc:
            logger.warning("[GCP TTS] Seg %d failed: %s", idx, exc)
            failed += 1

        if (idx + 1) % 20 == 0:
            logger.info("[GCP TTS] %s: %d/%d segments", lang_name, idx + 1, len(segments))

    logger.info("[GCP TTS] %s: %d/%d OK, %d failed", lang_name, len(files), len(segments), failed)
    return files


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
            "[XTTS] Language '%s' is not supported by XTTS v2. Falling back to Sarvam TTS.",
            lang_code,
        )
        return _synth_segments_sarvam(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
        )

    if not voice_reference_audio or not os.path.exists(voice_reference_audio):
        logger.warning("[XTTS] No voice reference provided. Falling back to Sarvam TTS.")
        return _synth_segments_sarvam(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
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
        return _synth_segments_sarvam(
            segments, lang_code, out_dir,
            preserve_fillers=preserve_fillers,
        )

    # ── Synthesise each segment ───────────────────────────────────────
    failed = 0
    try:
        for idx, seg in enumerate(segments):
            out_path = os.path.join(out_dir, f"segment_{idx:03d}.wav")
            text = seg.text if preserve_fillers else get_text_without_fillers(seg.text)
            text = text.replace(".", " ").strip()
            if not text:
                continue

            try:
                logger.debug(
                    "[XTTS] Segment %d/%d: '%s…'",
                    idx, len(segments) - 1, text[:50],
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
                    speed=1.0,   # one natural speed; the aligner fits the timing
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
