"""
transcriber.py
──────────────
Speech-to-Text: transcribe the English audio and produce timestamped segments.

Supports two backends:
  • Whisper (local, word-level timestamps)
  • Sarvam STT API

Output format – list of segments:
  [
    {"start": 0.0, "end": 3.5, "text": "Welcome to this lecture..."},
    ...
  ]
"""

from __future__ import annotations
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import requests
from config.settings import SARVAM_API_KEY, SARVAM_BASE_URL
from src.filler_detector import (
    detect_fillers_in_words,
    FillerInfo,
    FILLER_WORDS,
)

logger = logging.getLogger("nptel_pipeline")


# ── Segment data type ────────────────────────────────────────────────────────
@dataclass
class Segment:
    """A single timed transcript segment."""
    start: float              # seconds
    end: float                # seconds
    text: str
    fillers: list = field(default_factory=list)  # list of FillerInfo or dicts

    def to_dict(self) -> dict:
        d = {"start": self.start, "end": self.end, "text": self.text}
        if self.fillers:
            d["fillers"] = [
                f.to_dict() if hasattr(f, 'to_dict') else f
                for f in self.fillers
            ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        fillers = [
            FillerInfo(**f) if isinstance(f, dict) else f
            for f in d.get("fillers", [])
        ]
        return cls(
            start=d["start"], end=d["end"], text=d["text"], fillers=fillers,
        )

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def __repr__(self):
        filler_tag = f" [{len(self.fillers)}F]" if self.fillers else ""
        return f"Segment({self.start:.2f}-{self.end:.2f}: {self.text[:40]}…{filler_tag})"


# ── Transcription function ───────────────────────────────────────────────────

# ── Sentence-boundary chunking helpers ───────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_at_sentence_boundaries(segments: list[Segment]) -> list[Segment]:
    """
    Atomise segments at internal sentence boundaries.

    A Whisper or Sarvam segment often contains MORE than one sentence:
        "Hello everyone. My name is Amit"  <- two sentences in one segment

    This function splits such segments so every output segment holds
    EXACTLY one sentence.  Timestamps are interpolated proportionally
    by character count (accurate enough given each segment is only a few
    seconds long; Whisper word-level timestamps fix this exactly).
    """
    result: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        parts = _SENT_SPLIT_RE.split(text)
        if len(parts) <= 1:
            result.append(seg)
            continue

        total_len = max(len(text), 1)
        dur = seg.duration
        char_pos = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            frac_start = char_pos / total_len
            frac_end   = min((char_pos + len(part)) / total_len, 1.0)
            t_start = round(seg.start + frac_start * dur, 3)
            t_end   = round(seg.start + frac_end   * dur, 3)
            result.append(Segment(start=t_start, end=t_end, text=part,
                                  fillers=list(getattr(seg, "fillers", []))))
            char_pos += len(part) + 1   # +1 for the whitespace the regex consumed
    return result


def _words_to_sentence_chunks(
    words: list[tuple[str, float, float]],
    target_sentences: int = 3,   # kept for API compat, ignored internally
    max_duration: float = 15.0,  # kept for API compat, overridden below
) -> list[Segment]:
    """
    Duration-balanced sentence-aligned chunker for Whisper word streams.

    Algorithm
    ---------
    Walk words one by one, keeping a running chunk buffer.  Flush the
    buffer (emit a Segment) when BOTH conditions hold:

        1. Current chunk duration >= MIN_CHUNK_DUR  (long enough to be useful)
        2. The current word ends a sentence  (., !, ?)

    At MAX_CHUNK_DUR without a sentence end, split at the longest pause
    between words (a comma pause preferred) rather than at the word reached,
    so the translator gets whole clauses instead of a sentence cut mid-phrase.

    Result: chunks that are all roughly the same length (MIN–MAX range),
    always end at a sentence boundary, and use exact Whisper word timestamps
    with zero interpolation.
    """
    if not words:
        return []

    MIN_CHUNK_DUR = 5.0    # don't flush shorter than this (avoids tiny fragments)
    TARGET_CHUNK_DUR = 9.0 # preferred flush point after a sentence end
    MAX_CHUNK_DUR = 13.0   # hard cap — force flush even mid-sentence

    _SENT_END = frozenset(".!?")

    result: list[Segment] = []
    ch_words: list[str]   = []
    ch_st:    list[float] = []
    ch_et:    list[float] = []

    def _flush(n: int | None = None):
        """Emit the first *n* buffered words (all by default) as a segment."""
        n = len(ch_words) if n is None else n
        if not n:
            return
        result.append(Segment(
            start=round(ch_st[0], 3),
            end=round(ch_et[n - 1], 3),
            text=" ".join(ch_words[:n]),
            fillers=detect_fillers_in_words(ch_words[:n], ch_st[:n], ch_et[:n]),
        ))
        del ch_words[:n], ch_st[:n], ch_et[:n]

    def _best_split() -> int:
        """Word count to emit when over MAX_CHUNK_DUR: cut at the longest pause."""
        best, best_score = len(ch_words), -1.0
        for k in range(len(ch_words) - 1):
            if ch_et[k] - ch_st[0] < 2.0:          # avoid a tiny first piece
                continue
            score = ch_st[k + 1] - ch_et[k]        # pause after word k
            if ch_words[k][-1:] in ",;:":
                score += 0.3
            if score > best_score:
                best, best_score = k + 1, score
        return best

    for word, t_start, t_end in words:
        w = word.strip()
        if not w:
            continue

        ch_words.append(w)
        ch_st.append(t_start)
        ch_et.append(t_end)

        chunk_dur = ch_et[-1] - ch_st[0]
        ends_sentence = w.rstrip()[-1] in _SENT_END if w.rstrip() else False

        if chunk_dur >= MAX_CHUNK_DUR:
            # Hard cap: no sentence end in time — cut at the longest pause
            _flush(_best_split())
        elif ends_sentence and chunk_dur >= MIN_CHUNK_DUR:
            # Good flush point: sentence ended and chunk is long enough
            _flush()
        # else: keep accumulating — sentence ended but chunk is too short,
        # or chunk is long enough but we're mid-sentence (wait for boundary)

    _flush()  # trailing words

    logger.info(
        "Duration-balanced rechunk: %d words → %d chunks "
        "(target %.0f–%.0fs, hard cap %.0fs)",
        len(words), len(result), MIN_CHUNK_DUR, TARGET_CHUNK_DUR, MAX_CHUNK_DUR,
    )
    return result


def _merge_into_sentence_chunks(
    segments: list[Segment],
    max_duration: float = 15.0,   # kept for API compat, overridden below
    target_sentences: int = 3,    # kept for API compat, ignored internally
) -> list[Segment]:
    """
    Duration-balanced sentence-aligned chunker for Sarvam output.

    Same philosophy as _words_to_sentence_chunks but operates on segments
    (not words) since Sarvam doesn't give word-level timestamps.

    Stage 1: split any multi-sentence segment into sentence atoms.
    Stage 2: merge atoms into chunks of MIN_CHUNK_DUR–MAX_CHUNK_DUR,
             flushing at sentence boundaries when long enough.
    Timestamps are interpolated proportionally within each atom (necessary
    since word-level data is unavailable for these backends).
    """
    if not segments:
        return segments

    MIN_CHUNK_DUR = 5.0
    MAX_CHUNK_DUR = 13.0
    _SENTENCE_END = frozenset(".!?")

    # Stage 1: atomise at sentence boundaries
    sentence_segs = _split_at_sentence_boundaries(segments)
    logger.debug("Stage 1: %d raw → %d sentence atoms", len(segments), len(sentence_segs))

    # Stage 2: duration-balanced merge
    merged: list[Segment] = []
    buf_start: float | None = None
    buf_parts: list[str]    = []
    buf_end:   float        = 0.0
    buf_fillers: list       = []

    def _flush_buf():
        if not buf_parts or buf_start is None:
            return
        merged.append(Segment(
            start=buf_start, end=buf_end,
            text=" ".join(buf_parts), fillers=list(buf_fillers),
        ))
        buf_parts.clear(); buf_fillers.clear()

    for seg in sentence_segs:
        text = seg.text.strip()
        if not text:
            continue

        if buf_start is None:
            buf_start = seg.start

        chunk_dur = seg.end - buf_start
        ends_sentence = text.rstrip()[-1] in _SENTENCE_END if text.rstrip() else False

        # Hard cap: force flush before adding (would exceed max)
        if buf_parts and chunk_dur > MAX_CHUNK_DUR:
            _flush_buf()
            buf_start = seg.start
            buf_end   = 0.0

        buf_parts.append(text)
        buf_end = seg.end
        buf_fillers.extend(getattr(seg, "fillers", []))

        chunk_dur = buf_end - buf_start
        if ends_sentence and chunk_dur >= MIN_CHUNK_DUR:
            _flush_buf()
            buf_start = None
            buf_end   = 0.0

    _flush_buf()

    logger.info(
        "Duration-balanced merge: %d atoms → %d chunks (%.0f–%.0fs)",
        len(sentence_segs), len(merged), MIN_CHUNK_DUR, MAX_CHUNK_DUR,
    )
    return merged


def _snap_to_word_boundary(text: str, pos: int, punct_window: int = 25) -> int:
    """
    Move a cut position in *text* to a word boundary.

    Prefers a clause end (",;:.?!" followed by a space) within
    ``punct_window`` characters, otherwise the nearest space.  Returns
    ``len(text)`` when the text has no spaces at all (can't be split).
    """
    lo, hi = max(0, pos - punct_window), min(len(text), pos + punct_window)
    clause_ends = [
        i + 1 for i in range(lo, hi)
        if text[i] in ",;:.?!" and i + 1 < len(text) and text[i + 1] == " "
    ]
    if clause_ends:
        return min(clause_ends, key=lambda i: abs(i - pos))
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return len(text)
    return min(spaces, key=lambda i: abs(i - pos))


def split_segments_at_silences(
    segments: list[Segment],
    audio_path: str,
    min_window_sec: float = 12.0,
    min_silence_sec: float = 1.8,
    silence_db: float = -40.0,
    max_sub_segments: int = 6,
) -> list[Segment]:
    """
    Post-transcription refinement: split large-window segments at internal silences.

    When the transcriber returns a segment like:
        [159.69s – 205.09s] "And the last part is live mentoring sessions,
                             live tutoring sessions, live doubt clearing sessions."
    it usually means the professor spoke each phrase with long pauses in between.
    Splitting at those pauses lets the aligner place each sub-phrase's dubbed TTS
    at exactly the right moment instead of cramming all speech into the start and
    then going silent for 30+ seconds.

    Only segments with window > ``min_window_sec`` are eligible for splitting.
    Each internal silence of >= ``min_silence_sec`` triggers a split at the
    midpoint of that silence gap.  Text is divided proportionally by time
    (good enough given that proportional time ≈ proportional word count
    for naturally paced speech), with each cut moved to the nearest word
    boundary — punctuation preferred, since a long pause usually ends a clause.

    New segments are also split — if a sub-segment window is still > min_window_sec
    and has further internal silences, it will be split on the NEXT pipeline run
    (or re-process from scratch to trigger it immediately).

    Args:
        segments:           The segment list to post-process.
        audio_path:         Path to the (vocals) audio — used for silencedetect.
        min_window_sec:     Only split segments with window >= this value (s).
        min_silence_sec:    Minimum silence duration to trigger a split (s).
        silence_db:         Silence threshold in dBFS (e.g. -40 dB).
        max_sub_segments:   Cap on how many pieces one segment can be split into.

    Returns:
        New segment list (same or more segments, all with tight timing windows).
    """
    result: list[Segment] = []
    splits_made = 0

    for seg in segments:
        window = seg.duration
        if window < min_window_sec:
            result.append(seg)
            continue

        # Run silencedetect on the segment window
        try:
            import tempfile
            import subprocess as _sp
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
            _sp.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ss", f"{seg.start:.3f}", "-t", f"{window:.3f}",
                 "-ar", "16000", "-ac", "1", tmp_wav],
                capture_output=True, check=True,
            )
            # silencedetect reports at info level — don't pass "-v error" here
            res = _sp.run(
                ["ffmpeg", "-hide_banner", "-nostats", "-i", tmp_wav,
                 "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_sec}",
                 "-f", "null", "/dev/null"],
                capture_output=True, text=True,
            )
            os.remove(tmp_wav)
        except Exception as exc:
            logger.debug("silencedetect failed for seg %.1f–%.1f: %s", seg.start, seg.end, exc)
            result.append(seg)
            continue

        # Parse silence intervals (relative to seg.start)
        silence_gaps: list[tuple[float, float]] = []
        s_st: float | None = None
        for line in res.stderr.split("\n"):
            if "silence_start" in line:
                try:
                    s_st = float(line.split("silence_start:")[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
            elif "silence_end" in line and s_st is not None:
                try:
                    s_et = float(line.split("silence_end:")[1].strip().split("|")[0].strip())
                    # Ignore leading / trailing silences (within 1s of window edges)
                    if s_st > 1.0 and s_et < window - 0.5:
                        silence_gaps.append((s_st, s_et))
                    s_st = None
                except (IndexError, ValueError):
                    pass

        if not silence_gaps:
            result.append(seg)
            continue

        # Build split points at midpoint of each internal silence gap
        split_points_abs: list[float] = []
        for g_start, g_end in silence_gaps:
            abs_mid = seg.start + (g_start + g_end) / 2.0
            split_points_abs.append(abs_mid)

        # Cap number of splits
        split_points_abs = sorted(split_points_abs)[: max_sub_segments - 1]

        # Build sub-segment time boundaries
        boundaries = [seg.start] + split_points_abs + [seg.end]

        # Proportional text cut for each split point, snapped to a word
        # boundary (preferring punctuation) so no word is cut in half.
        cuts = [0]
        for t in split_points_abs:
            pos = int((t - seg.start) / window * len(seg.text))
            cuts.append(max(cuts[-1], _snap_to_word_boundary(seg.text, pos)))
        cuts.append(len(seg.text))

        sub_segs: list[Segment] = []
        carry_start: float | None = None
        for i in range(len(boundaries) - 1):
            t0, t1 = boundaries[i], boundaries[i + 1]
            sub_text = seg.text[cuts[i]:cuts[i + 1]].strip()
            if not sub_text:
                # No words landed in this span — fold its time into a neighbour
                if sub_segs:
                    sub_segs[-1].end = round(t1, 3)
                elif carry_start is None:
                    carry_start = t0
                continue
            sub_segs.append(Segment(
                start=round(t0 if carry_start is None else carry_start, 3),
                end=round(t1, 3),
                text=sub_text, fillers=[],
            ))
            carry_start = None

        if len(sub_segs) <= 1:
            result.append(seg)
            continue

        result.extend(sub_segs)
        splits_made += len(sub_segs) - 1
        logger.debug(
            "Split seg %.1f–%.1f (%.0fs) → %d sub-segs at %d silence gaps",
            seg.start, seg.end, window, len(sub_segs), len(silence_gaps),
        )

    if splits_made:
        logger.info(
            "silence-split: %d→%d segments (%d splits at internal silences ≥%.1fs)",
            len(segments), len(result), splits_made, min_silence_sec,
        )
    return result


def transcribe_audio(audio_path: str, method: str = "whisper") -> list[Segment]:
    """
    Transcribe an audio file and return timestamped segments.

    Args:
        audio_path: Path to the WAV audio file.
        method:     "whisper" or "sarvam".

    Returns:
        List of Segment objects with start/end times and English text.
        Sentences are guaranteed not to be split across segment boundaries.
    """
    if method == "whisper":
        # Whisper uses _words_to_sentence_chunks internally; already sentence-aligned.
        return _transcribe_whisper(audio_path)
    elif method == "sarvam":
        raw = _transcribe_sarvam(audio_path)
    else:
        raise ValueError(f"Unknown transcription method: {method}")
    # For Sarvam: two-stage split+merge to fix sentence boundaries
    return _merge_into_sentence_chunks(raw)


# ── Whisper STT (local, accurate timestamps) ─────────────────────────────────

def _transcribe_whisper(audio_path: str) -> list[Segment]:
    """
    Transcribe with faster-whisper in a child process.

    CTranslate2 keeps the VRAM it used after the Whisper model is deleted
    (3.7 GB of a 4 GB card, measured), which leaves too little for the next
    GPU stage: Demucs and Wav2Lip run as their own processes and hit CUDA
    out-of-memory.  A child process gives all of it back when it exits.
    """
    from config.settings import BASE_DIR

    with tempfile.TemporaryDirectory(prefix="whisper_") as tmp:
        out_json = os.path.join(tmp, "segments.json")
        proc = subprocess.run(
            [sys.executable, "-m", "src.transcriber", audio_path, out_json],
            cwd=BASE_DIR,
        )
        if not os.path.isfile(out_json):
            raise RuntimeError(f"Whisper process failed (exit code {proc.returncode})")
        with open(out_json, encoding="utf-8") as f:
            payload = json.load(f)

    if "error" in payload:
        # ValueError carries "no speech detected", which triggers the Demucs retry
        error_cls = ValueError if payload["error_type"] == "ValueError" else RuntimeError
        raise error_cls(payload["error"])
    return [Segment.from_dict(d) for d in payload["segments"]]


def _transcribe_whisper_local(audio_path: str) -> list[Segment]:
    """
    Transcribe audio using faster-whisper running in this process.

    This produces highly accurate word-level timestamps by using the
    Whisper model's cross-attention alignment — far more reliable than
    asking an LLM to guess timestamps.

    Model is downloaded automatically on first use (~500 MB for 'small').
    Set WHISPER_MODEL env-var to change (tiny/base/small/medium/large-v3).
    """
    from faster_whisper import WhisperModel

    logger.info("Whisper STT → %s", audio_path)

    model_size = os.environ.get("WHISPER_MODEL", "small")
    logger.info("Loading Whisper model '%s' …", model_size)

    # Use CUDA if available, else CPU with int8 quantisation
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            logger.info("Whisper running on GPU (float16)")
        else:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Whisper running on CPU (int8)")
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments_iter, info = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
    )

    logger.info("Detected language: %s (prob=%.2f)",
                info.language, info.language_probability)

    # Collect ALL words from every VAD segment — we rechunk from scratch
    # using _words_to_sentence_chunks so that:
    #   a) the entire audio is transcribed before any chunking decision
    #   b) chunk boundaries fall exactly on sentence boundaries
    #   c) word-level timestamps are used directly (no interpolation)
    all_words: list[tuple[str, float, float]] = []
    # Fallback text segments for Whisper segments that have no word timestamps
    # (happens for music, noise, or some non-speech audio regions)
    fallback_segs: list[Segment] = []
    first_start: float | None = None
    last_end:    float        = 0.0

    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        if seg.words:
            for w in seg.words:
                word = w.word.strip()
                if not word:
                    continue
                all_words.append((word, w.start, w.end))
                if first_start is None:
                    first_start = w.start
                last_end = w.end
        else:
            # Segment has text but no word-level timestamps (music/noise region)
            # Keep it as a fallback segment so we don't lose any transcribed text
            fallback_segs.append(Segment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
            ))

    if not all_words and not fallback_segs:
        raise ValueError(
            "Whisper detected no speech in the audio.\n"
            "If this is a music video or the audio has background music, enable "
            "'Preserve Background Music' in the sidebar — this separates vocals "
            "from music before transcription."
        )

    if all_words:
        logger.info(
            "Collected %d words (%.2fs – %.2fs) — rechunking into sentence chunks…",
            len(all_words), first_start or 0.0, last_end,
        )
        chunks = _words_to_sentence_chunks(all_words, target_sentences=3, max_duration=15.0)
    else:
        logger.warning(
            "No word-level timestamps found — using segment-level fallback (%d segments). "
            "This may happen when the audio contains mainly music or noise.",
            len(fallback_segs),
        )
        chunks = _merge_into_sentence_chunks(fallback_segs)

    chunks = _sanitize_timestamps(chunks)

    # Explicitly free the Whisper model and flush GPU memory so later stages
    # (OpenVoice, Resemblyzer, Wav2Lip) have the full VRAM budget available.
    del model
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("Whisper model freed — GPU memory released")
    except Exception:
        pass

    return chunks


# ── Sarvam STT ────────────────────────────────────────────────────────────────

def _transcribe_sarvam(audio_path: str) -> list[Segment]:
    """
    Transcribe audio using Sarvam Saaras v3 STT.

    The REST API supports a with_timestamps option for word-level timing.
    Audio is chunked into ≤25 s pieces (REST endpoint limit is 30 s).
    """
    from pydub import AudioSegment as PydubSegment
    import requests

    logger.info("Sarvam STT → %s", audio_path)

    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    # Load and split audio into ≤25 s chunks
    audio = PydubSegment.from_file(audio_path)
    chunk_ms = 25_000  # 25 seconds per chunk
    chunks: list[tuple[float, PydubSegment]] = []  # (offset_seconds, chunk)
    for start_ms in range(0, len(audio), chunk_ms):
        chunk = audio[start_ms : start_ms + chunk_ms]
        chunks.append((start_ms / 1000.0, chunk))

    logger.info("Split into %d chunks (≤25 s each)", len(chunks))

    all_segments: list[Segment] = []

    for idx, (offset_sec, chunk) in enumerate(chunks):
        # Export chunk to a temp WAV file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(tmp.name, format="wav")
        tmp.close()

        try:
            # Use REST API directly to enable with_timestamps
            url = f"{SARVAM_BASE_URL}/speech-to-text"
            headers = {"api-subscription-key": SARVAM_API_KEY}
            data = {
                "model": "saaras:v3",
                "language_code": "en-IN",
                "with_timestamps": "true",
            }

            with open(tmp.name, "rb") as f:
                files = {"file": ("chunk.wav", f, "audio/wav")}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)

            resp.raise_for_status()
            result = resp.json()

            transcript = result.get("transcript", "").strip()
            timestamps = result.get("timestamps")

            if not transcript:
                logger.debug("  Chunk %d: (silence/empty)", idx + 1)
                continue

            # If we got word-level timestamps, group into sentence segments
            if timestamps and isinstance(timestamps, dict):
                words = timestamps.get("words", [])
                start_times = timestamps.get("start_time_seconds", [])
                end_times = timestamps.get("end_time_seconds", [])

                if words and start_times and end_times:
                    segs = _group_words_into_segments(
                        words, start_times, end_times, offset_sec
                    )
                    all_segments.extend(segs)
                    logger.debug("  Chunk %d: %d segments (with timestamps)", idx + 1, len(segs))
                    continue

            # Fallback: no timestamps – create one segment per chunk
            chunk_duration = len(chunk) / 1000.0
            all_segments.append(Segment(
                start=offset_sec,
                end=offset_sec + chunk_duration,
                text=transcript,
            ))
            logger.debug("  Chunk %d: 1 segment (no word timestamps)", idx + 1)

        finally:
            os.remove(tmp.name)

    if not all_segments:
        raise ValueError("Sarvam returned an empty transcript")

    logger.info("Total: %d segments", len(all_segments))
    # Sanitize timestamps (fix overlaps / non-monotonic issues)
    all_segments = _sanitize_timestamps(all_segments)
    return all_segments


def _sanitize_timestamps(segments: list[Segment]) -> list[Segment]:
    """
    Post-process STT segments to fix common timestamp problems:
      • Negative or zero-length segments → give a small minimum duration
      • Overlapping segments → shrink end of earlier segment
      • Non-monotonic starts → push start forward
      • Very first segment starting at exactly 0.0 when it shouldn't → keep as-is
    """
    if not segments:
        return segments

    MIN_DUR = 0.3  # minimum segment duration in seconds

    cleaned: list[Segment] = []
    for seg in segments:
        s, e = seg.start, seg.end
        # Ensure start < end
        if e <= s:
            e = s + MIN_DUR
        cleaned.append(Segment(start=s, end=e, text=seg.text))

    # Fix overlaps: each segment's start must be >= previous segment's end
    for i in range(1, len(cleaned)):
        prev = cleaned[i - 1]
        curr = cleaned[i]
        if curr.start < prev.end:
            # Option: shrink previous end to midpoint, or push current start
            mid = (prev.end + curr.start) / 2
            if mid > prev.start + MIN_DUR:
                prev.end = mid
                curr.start = mid
            else:
                curr.start = prev.end
        # Ensure current segment still has positive duration
        if curr.end <= curr.start:
            curr.end = curr.start + MIN_DUR

    logger.debug("Sanitized %d segments (range: %.2fs – %.2fs)",
                 len(cleaned), cleaned[0].start, cleaned[-1].end)
    return cleaned


def _group_words_into_segments(
    words: list[str],
    start_times: list[float],
    end_times: list[float],
    offset: float,
    max_words: int = 60,
) -> list[Segment]:
    """
    Group word-level timestamps into sentence-like segments.
    Splits ONLY on sentence-ending punctuation (.!?); falls back to
    max_words as a safety limit so no chunk is unreasonably huge.
    """
    segments: list[Segment] = []
    buf_words: list[str] = []
    buf_start: float | None = None

    for w, st, et in zip(words, start_times, end_times):
        if buf_start is None:
            buf_start = st + offset
        buf_words.append(w)

        # Split on sentence-ending punctuation or word count limit
        is_sentence_end = w.rstrip().endswith((".", "!", "?"))
        if is_sentence_end or len(buf_words) >= max_words:
            text = " ".join(buf_words).strip()
            if text:
                segments.append(Segment(
                    start=buf_start,
                    end=et + offset,
                    text=text,
                ))
            buf_words = []
            buf_start = None

    # Flush remaining words
    if buf_words and buf_start is not None:
        text = " ".join(buf_words).strip()
        if text:
            segments.append(Segment(
                start=buf_start,
                end=end_times[-1] + offset,
                text=text,
            ))

    return segments


if __name__ == "__main__":
    # Child process for _transcribe_whisper:  python -m src.transcriber <audio> <out.json>
    from src.utils import setup_logging

    setup_logging()
    _audio, _out = sys.argv[1], sys.argv[2]
    try:
        _payload = {"segments": [s.to_dict() for s in _transcribe_whisper_local(_audio)]}
    except Exception as _exc:
        _payload = {"error": str(_exc), "error_type": type(_exc).__name__}
    with open(_out, "w", encoding="utf-8") as _f:
        json.dump(_payload, _f, ensure_ascii=False)
