"""
filler_detector.py
──────────────────
Detect and preserve speech fillers (um, uh, ah, etc.) in transcriptions.

When speakers use hesitation markers, these fillers contribute to the
natural rhythm and pacing of speech.  This module:

  • Detects filler words in transcript segments
  • Stores filler metadata (position, original timing) in Segment objects
  • Provides utilities to extract original filler audio from the source
  • Supports splicing original filler audio into dubbed output
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("nptel_pipeline")

# ── Filler word definitions ──────────────────────────────────────────────────

# English hesitation markers / fillers
FILLER_WORDS: frozenset[str] = frozenset({
    "um", "umm", "uh", "uhh", "ah", "ahh", "er", "err",
    "hmm", "hm", "mm", "mmm", "em", "erm", "ehm",
    "oh",
})

# Extended fillers (discourse markers that serve as pauses)
DISCOURSE_MARKERS: frozenset[str] = frozenset({
    "like", "you know", "i mean", "basically",
    "actually", "so", "right", "okay", "ok", "well",
    "let me see", "lets see",
})


@dataclass
class FillerInfo:
    """Metadata about a detected filler word."""
    word: str           # The filler word
    start: float        # Start time in seconds (absolute)
    end: float          # End time in seconds (absolute)
    index_in_text: int  # Character index in the segment text where filler appears

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "index_in_text": self.index_in_text,
        }


def detect_fillers_in_words(
    words: list[str],
    start_times: list[float],
    end_times: list[float],
    include_discourse_markers: bool = False,
) -> list[FillerInfo]:
    """
    Detect filler words from word-level timestamps.

    Args:
        words:       List of transcribed words.
        start_times: Per-word start times (seconds).
        end_times:   Per-word end times (seconds).
        include_discourse_markers: Also detect "like", "you know", etc.

    Returns:
        List of FillerInfo for detected fillers.
    """
    filler_set = set(FILLER_WORDS)
    if include_discourse_markers:
        filler_set |= set(DISCOURSE_MARKERS)

    fillers: list[FillerInfo] = []
    char_pos = 0

    for word, st, et in zip(words, start_times, end_times):
        clean = word.strip().lower().rstrip(".,!?;:")
        if clean in filler_set:
            fillers.append(FillerInfo(
                word=word.strip(),
                start=st,
                end=et,
                index_in_text=char_pos,
            ))
        char_pos += len(word) + 1  # +1 for space

    return fillers


def detect_fillers_in_text(text: str) -> list[dict]:
    """
    Simple text-based filler detection (no timestamps).
    Returns list of {"word": ..., "index_in_text": ...}.
    """
    words = text.split()
    results = []
    char_pos = 0

    for word in words:
        clean = word.strip().lower().rstrip(".,!?;:")
        if clean in FILLER_WORDS:
            results.append({
                "word": word.strip(),
                "index_in_text": char_pos,
            })
        char_pos += len(word) + 1

    return results


def extract_filler_audio(
    audio_path: str,
    filler: FillerInfo,
    output_path: str,
) -> str:
    """
    Extract the original audio for a filler word.

    This preserves the speaker's actual hesitation sound instead of
    synthesizing a fake one via TTS.
    """
    duration = filler.end - filler.start
    if duration <= 0:
        duration = 0.3  # minimum

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", f"{filler.start:.4f}",
        "-t", f"{duration:.4f}",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def build_filler_map_for_alignment(
    segments,
    original_audio_path: str,
    output_dir: str,
) -> dict[int, list[tuple[FillerInfo, str]]]:
    """
    For each segment that has fillers, extract the original filler audio.

    Returns:
        { segment_index: [(FillerInfo, audio_path), ...], ... }
    """
    os.makedirs(output_dir, exist_ok=True)
    filler_map: dict[int, list[tuple[FillerInfo, str]]] = {}

    for idx, seg in enumerate(segments):
        if not hasattr(seg, 'fillers') or not seg.fillers:
            continue

        segment_fillers = []
        for i, filler_data in enumerate(seg.fillers):
            if isinstance(filler_data, FillerInfo):
                filler = filler_data
            elif isinstance(filler_data, dict):
                filler = FillerInfo(**filler_data)
            else:
                continue

            filler_audio = os.path.join(
                output_dir, f"filler_{idx:04d}_{i:02d}.wav"
            )
            try:
                extract_filler_audio(original_audio_path, filler, filler_audio)
                segment_fillers.append((filler, filler_audio))
            except Exception as exc:
                logger.warning("Failed to extract filler audio seg %d: %s", idx, exc)

        if segment_fillers:
            filler_map[idx] = segment_fillers

    logger.info("Built filler map: %d segments with fillers", len(filler_map))
    return filler_map


def get_text_without_fillers(text: str) -> str:
    """
    Remove filler words from text for cleaner translation / TTS synthesis.

    A sentence end carried by a removed filler ("… the result ah.") is moved
    onto the previous word so sentence boundaries survive.
    """
    words = text.split()
    cleaned = []
    for word in words:
        clean = word.strip().lower().rstrip(".,!?;:")
        if clean not in FILLER_WORDS:
            cleaned.append(word)
            continue
        sentence_end = word[len(word.rstrip(".!?")):]
        if sentence_end and cleaned and cleaned[-1][-1] not in ".!?":
            cleaned[-1] = cleaned[-1].rstrip(",;:") + sentence_end
    result = " ".join(cleaned)
    # Clean up double spaces
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


def get_text_with_target_fillers(text: str, target_lang: str) -> str:
    """
    Replace English fillers with target-language equivalents for more
    natural TTS output.
    """
    FILLER_TRANSLATIONS: dict[str, dict[str, str]] = {
        "hi": {"um": "अम्म", "umm": "अम्म", "uh": "अह", "uhh": "अह",
                "hmm": "हम्म", "hm": "हम्म", "ah": "आह", "ahh": "आह",
                "er": "अर", "err": "अर", "oh": "ओह"},
        "te": {"um": "అమ్మ", "umm": "అమ్మ", "uh": "అహ", "uhh": "అహ",
                "hmm": "హ్మ్మ్", "hm": "హ్మ్మ్", "ah": "ఆహ", "ahh": "ఆహ",
                "er": "ఎర్", "err": "ఎర్", "oh": "ఓహ్"},
        "od": {"um": "ଅମ୍ମ", "umm": "ଅମ୍ମ", "uh": "ଅହ", "uhh": "ଅହ",
                "hmm": "ହ୍ମ୍ମ", "hm": "ହ୍ମ୍ମ", "ah": "ଆହ", "ahh": "ଆହ",
                "er": "ଏର୍", "err": "ଏର୍", "oh": "ଓହ"},
    }

    lang_fillers = FILLER_TRANSLATIONS.get(target_lang, {})
    if not lang_fillers:
        return text

    words = text.split()
    result = []
    for word in words:
        clean = word.strip().lower().rstrip(".,!?;:")
        if clean in lang_fillers:
            result.append(lang_fillers[clean])
        else:
            result.append(word)
    return " ".join(result)
