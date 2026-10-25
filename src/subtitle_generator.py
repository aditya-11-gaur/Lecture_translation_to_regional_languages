"""
subtitle_generator.py
─────────────────────
Generate .srt subtitle files from timestamped Segment lists.
Produces one .srt per language with proper temporal alignment.
"""

from __future__ import annotations
import logging
import os
from config.settings import OUTPUT_DIR, ALL_LANGUAGES, MAX_CHARS_PER_LINE
from src.transcriber import Segment

logger = logging.getLogger("nptel_pipeline")


def _format_srt_timestamp(seconds: float) -> str:
    """Convert seconds → SRT timestamp  HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds → WebVTT timestamp  HH:MM:SS.mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def wrap_subtitle_text(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> str:
    """
    Wrap subtitle text to fit within max_chars per line.

    Handles code-mixed text (e.g., "यह algorithm बहुत efficient है")
    by splitting at word boundaries.  Produces at most 2 lines for
    readability.  If the text is short enough, returns it unchanged.

    For code-mixed subtitles, we prefer to break at script boundaries
    when possible (i.e., keep the English technical term on the same
    line as its surrounding target-language text).
    """
    if len(text) <= max_chars:
        return text

    words = text.split()
    if not words:
        return text

    # Try to split into 2 roughly equal lines
    mid = len(text) // 2
    best_split = len(text)
    split_idx = len(words)

    running = 0
    for i, word in enumerate(words):
        running += len(word) + (1 if i > 0 else 0)
        if abs(running - mid) < abs(best_split - mid):
            best_split = running
            split_idx = i + 1

    line1 = " ".join(words[:split_idx])
    line2 = " ".join(words[split_idx:])
    if line2:
        return f"{line1}\n{line2}"

    return text


def generate_srt(
    segments: list[Segment],
    lang_code: str,
    video_name: str,
    output_dir: str | None = None,
    total_duration: float | None = None,
) -> str:
    """
    Write an .srt subtitle file for the given segments.

    For code-mixed subtitles (e.g., Hindi text with English technical terms),
    the file is written with a UTF-8 BOM for maximum compatibility with
    media players and subtitle editors that need to display mixed scripts.

    Args:
        segments:       List of Segment objects (already in the target language).
        lang_code:      Language code (en / hi / te / od).
        video_name:     Base name of the source video (used for filename).
        output_dir:     Directory to write to (default: OUTPUT_DIR).
        total_duration: If given, clamp all timestamps to this value.

    Returns:
        Path to the generated .srt file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    lang_name = ALL_LANGUAGES.get(lang_code, {}).get("name", lang_code)
    srt_path = os.path.join(output_dir, f"{video_name}_{lang_code}.srt")

    lines: list[str] = []
    counter = 0
    for seg in segments:
        start = seg.start
        end = seg.end
        if total_duration is not None:
            start = min(start, total_duration)
            end = min(end, total_duration)
        # Skip empty cues (e.g. a segment that was only a filler like "ah")
        if end <= start or not seg.text.strip():
            continue
        counter += 1
        start_ts = _format_srt_timestamp(start)
        end_ts = _format_srt_timestamp(end)
        wrapped = wrap_subtitle_text(seg.text)
        lines.append(str(counter))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(wrapped)
        lines.append("")  # blank line

    # Write with BOM for code-mixed subtitle compatibility
    with open(srt_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    logger.info("Written %s subtitles (%d cues) → %s", lang_name, counter, srt_path)
    return srt_path


def generate_vtt(
    segments: list[Segment],
    lang_code: str,
    video_name: str,
    output_dir: str | None = None,
    total_duration: float | None = None,
) -> str:
    """
    Write a WebVTT subtitle file for the given segments.

    Returns:
        Path to the generated .vtt file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    lang_name = ALL_LANGUAGES.get(lang_code, {}).get("name", lang_code)
    vtt_path = os.path.join(output_dir, f"{video_name}_{lang_code}.vtt")

    lines: list[str] = ["WEBVTT", ""]
    counter = 0
    for seg in segments:
        start = seg.start
        end = seg.end
        if total_duration is not None:
            start = min(start, total_duration)
            end = min(end, total_duration)
        # Skip empty cues (e.g. a segment that was only a filler like "ah")
        if end <= start or not seg.text.strip():
            continue
        counter += 1
        start_ts = _format_vtt_timestamp(start)
        end_ts = _format_vtt_timestamp(end)
        wrapped = wrap_subtitle_text(seg.text)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(wrapped)
        lines.append("")  # blank line

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Written %s VTT subtitles (%d cues) → %s", lang_name, counter, vtt_path)
    return vtt_path


def generate_all_subtitles(
    english_segments: list[Segment],
    translated_segments: dict[str, list[Segment]],
    video_name: str,
    output_dir: str | None = None,
    total_duration: float | None = None,
) -> dict[str, str]:
    """
    Generate .srt **and** .vtt files for English + all translated languages.

    VTT files are generated alongside SRT for web-based preview (Streamlit).

    Args:
        english_segments:    English transcript segments.
        translated_segments: { lang_code: [Segment, ...], ... }
        video_name:          Base name of the source video.
        output_dir:          Where to write subtitle files.
        total_duration:      Master duration (s) – clamp all timestamps.

    Returns:
        Dict mapping lang code → srt file path.
    """
    paths: dict[str, str] = {}

    # English subtitles
    paths["en"] = generate_srt(
        english_segments, "en", video_name, output_dir, total_duration,
    )
    generate_vtt(english_segments, "en", video_name, output_dir, total_duration)

    # Translated subtitles
    for lang_code, segments in translated_segments.items():
        paths[lang_code] = generate_srt(
            segments, lang_code, video_name, output_dir, total_duration,
        )
        generate_vtt(segments, lang_code, video_name, output_dir, total_duration)

    return paths
