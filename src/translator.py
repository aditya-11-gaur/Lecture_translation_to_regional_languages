"""
translator.py
─────────────
Translate English transcript segments into target languages (Hindi, Telugu, Odia).

Supports two backends:
  • Sarvam Translate API (code-mixed mode by default)
  • Google Cloud Translation v2

Input:  list of Segment objects (English)
Output: dict  { "hi": [Segment, ...], "te": [...], "od": [...] }
        Each translated segment preserves the original start/end timestamps.
"""

from __future__ import annotations
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from config.settings import (
    SARVAM_API_KEY,
    SARVAM_BASE_URL,
    GCP_API_KEY,
    TARGET_LANGUAGES,
    ENABLE_CODE_MIXING,
    SARVAM_TRANSLATION_MODE,
)
from src.transcriber import Segment
from src.filler_detector import get_text_without_fillers
from src.glossary import Glossary, load_default_glossary, verify_terms_preserved
from src.utils import retry_on_failure

logger = logging.getLogger("nptel_pipeline")

# ── Duration ratio estimation ─────────────────────────────────────────────────
# Characters-per-second at a natural speaking pace for each script.
# English: ~14 chars/sec (Latin, short words)
# Hindi Devanagari: ~5.5 chars/sec  (more chars per phoneme)
# Telugu / Odia: ~5.0 chars/sec
_SPOKEN_CPS: dict[str, float] = {
    "en": 14.0,
    "hi": 5.5,
    "te": 5.0,
    "od": 5.0,
}

def _estimate_duration_ratio(
    source_text: str,
    translated_text: str,
    source_lang: str = "en",
    target_lang: str = "hi",
) -> float:
    """
    Estimate how much longer (ratio > 1.0) or shorter (ratio < 1.0) the
    translated speech will be compared to the source speech.

    Based on character length and empirical chars-per-second speaking rates.
    A ratio of 1.2 means the translation will take ~20% longer to speak.
    """
    src_cps = _SPOKEN_CPS.get(source_lang, 14.0)
    tgt_cps = _SPOKEN_CPS.get(target_lang, 5.5)
    src_dur = max(len(source_text.strip()), 1) / src_cps
    tgt_dur = max(len(translated_text.strip()), 1) / tgt_cps
    return tgt_dur / src_dur



def translate_segments(
    segments: list[Segment],
    target_langs: list[str] | None = None,
    method: str = "sarvam",
    glossary: Glossary | None = None,
) -> dict[str, list[Segment]]:
    """
    Translate all segments to each target language.

    Args:
        segments:     English transcript segments.
        target_langs: Language codes to translate into (default: all targets).
        method:       "sarvam" or "gcp".
        glossary:     Technical term glossary for code-mixing (auto-loaded if None).

    Returns:
        Dict mapping language code → list of translated Segments.
    """
    if target_langs is None:
        target_langs = list(TARGET_LANGUAGES.keys())

    # Hesitation fillers ("ah", "um") must not reach the translator: Sarvam's
    # code-mixed mode leaves the phrase around them untranslated, and any
    # filler that survives ends up in the subtitles and the dubbed speech.
    # Segments that were nothing but fillers come back with empty text, at
    # the same index, so the output still lines up with *segments*.
    all_segments = segments
    cleaned = [get_text_without_fillers(s.text) for s in all_segments]
    speech_idx = [i for i, text in enumerate(cleaned) if text]
    segments = [
        Segment(start=all_segments[i].start, end=all_segments[i].end, text=cleaned[i])
        for i in speech_idx
    ]
    n_changed = sum(1 for s, c in zip(all_segments, cleaned) if c != s.text.strip())
    if n_changed:
        logger.info("Removed filler words from %d/%d segments before translation",
                    n_changed, len(all_segments))

    # Static term list for the GCP path and Sarvam's non-code-mixed modes (code-mixed mode
    # keeps technical terms in English on its own and ignores it)
    if glossary is None and ENABLE_CODE_MIXING:
        glossary = load_default_glossary()

    # Extract terms actually used in the transcript for focused prompts
    terms_in_use: list[str] = []
    if glossary:
        terms_in_use = glossary.extract_terms_from_segments(segments)
        logger.info("Code-mixing: %d technical terms found in transcript", len(terms_in_use))

    results: dict[str, list[Segment]] = {}

    def _translate_one_lang(lang: str) -> tuple[str, list[Segment]]:
        lang_name = TARGET_LANGUAGES[lang]["name"]
        logger.info("Translating to %s (%s)…", lang_name, lang)
        if method == "sarvam":
            translated = _translate_batch_sarvam(
                segments, lang, glossary=glossary, terms_in_use=terms_in_use,
            )
        elif method == "gcp":
            translated = _translate_batch_gcp(
                segments, lang, glossary=glossary, terms_in_use=terms_in_use,
            )
        else:
            raise ValueError(f"Unknown translation method: {method}")
        full = [Segment(start=s.start, end=s.end, text="") for s in all_segments]
        for i, seg in zip(speech_idx, translated):
            full[i] = seg
        return lang, full

    # Translate languages concurrently
    if len(target_langs) > 1 and method == "sarvam":
        # Sarvam is one-at-a-time API, parallelise across languages
        with ThreadPoolExecutor(max_workers=min(3, len(target_langs))) as pool:
            futures = {pool.submit(_translate_one_lang, l): l for l in target_langs}
            for future in as_completed(futures):
                lang, translated = future.result()
                results[lang] = translated
    else:
        for lang in target_langs:
            _, translated = _translate_one_lang(lang)
            results[lang] = translated

    return results


def _fail_if_nothing_translated(
    engine: str, lang_name: str, segments: list[Segment],
    fallback_count: int, last_error: Exception | None,
) -> None:
    """
    Both engines keep the English text for a segment whose call failed.  When
    that happens to every segment the engine is unusable (bad key, no
    credits, network down) and the "translation" is just English, so fail
    instead of dubbing English text with a Hindi/Telugu/Odia voice.
    """
    if segments and fallback_count >= len(segments):
        raise RuntimeError(
            f"{engine} translation to {lang_name} failed for every segment: {last_error}"
        )


# ── Term protection helpers (placeholder substitution for non-prompt APIs) ────

def _protect_terms(
    text: str,
    glossary: Glossary | None,
    terms_in_use: list[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Replace glossary terms in *text* with stable placeholder tokens so that a
    translation API that cannot accept a custom prompt won't translate them.

    Tokens look like ``XLTERM0X``, ``XLTERM1X``, … — uppercase, no spaces,
    unlikely to be transliterated by any Indian-language MT system.

    Returns:
        (protected_text, placeholder_map)  where placeholder_map maps each
        token back to the original (cased) term as it appeared in text.
    """
    if not glossary or not glossary._pattern:
        return text, {}

    # Collect only terms that actually appear in this segment
    if terms_in_use:
        terms_found = [
            t for t in terms_in_use
            if re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE)
        ]
    else:
        terms_found = glossary.find_terms_in_text(text)

    if not terms_found:
        return text, {}

    # Longest-first to avoid partial overlaps (e.g. "neural" before "neural network")
    terms_found = sorted(terms_found, key=len, reverse=True)

    protected = text
    placeholder_map: dict[str, str] = {}

    for i, term in enumerate(terms_found):
        token = f"XLTERM{i}X"
        # Capture the exact casing as it appears in text
        m = re.search(r"\b" + re.escape(term) + r"\b", protected, re.IGNORECASE)
        if m:
            placeholder_map[token] = m.group(0)
            protected = re.sub(
                r"\b" + re.escape(term) + r"\b",
                token,
                protected,
                flags=re.IGNORECASE,
            )

    return protected, placeholder_map


def _restore_terms(text: str, placeholder_map: dict[str, str]) -> str:
    """Substitute placeholder tokens back with their original English terms."""
    if not placeholder_map:
        return text
    for token, original in placeholder_map.items():
        text = text.replace(token, original)
    return text


# ── Sarvam batch translation ──────────────────────────────────────────────────

def _translate_batch_sarvam(
    segments: list[Segment], target_lang: str,
    glossary: Glossary | None = None,
    terms_in_use: list[str] | None = None,
) -> list[Segment]:
    """
    Translate segments one-by-one using Sarvam Translate API.

    Uses the sarvamai SDK.  Language codes expected by Sarvam are BCP-47
    style (e.g. "hi-IN"), stored in TARGET_LANGUAGES[lang]["sarvam_code"].
    Post-processes to verify technical terms are preserved (code-mixing).
    """
    from sarvamai import SarvamAI

    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    lang_info = TARGET_LANGUAGES[target_lang]
    lang_name = lang_info["name"]
    sarvam_code = lang_info["sarvam_code"]

    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    translated: list[Segment] = []
    fallback_count = 0
    last_error: Exception | None = None

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            translated.append(Segment(start=seg.start, end=seg.end, text=""))
            continue

        # In code-mixed mode, Sarvam natively keeps English technical terms —
        # placeholder protection is not needed and would degrade quality.
        # For other modes, apply placeholder protection as a safety net.
        use_code_mixed = SARVAM_TRANSLATION_MODE == "code-mixed"
        if use_code_mixed:
            protected_text, placeholder_map = text, {}
        else:
            protected_text, placeholder_map = _protect_terms(text, glossary, terms_in_use)

        @retry_on_failure(
            max_retries=2, backoff_base=2.0,
            retryable_exceptions=(Exception,),
        )
        def _call_sarvam(t):
            return client.text.translate(
                input=t,
                source_language_code="en-IN",
                target_language_code=sarvam_code,
                model="mayura:v1",
                mode=SARVAM_TRANSLATION_MODE,
            )

        try:
            resp = _call_sarvam(protected_text)
            new_text = resp.translated_text or protected_text
        except Exception as exc:
            logger.warning("Sarvam error on segment %d: %s", i + 1, exc)
            new_text = text  # keep original on failure
            placeholder_map = {}
            fallback_count += 1
            last_error = exc

        # Restore placeholders for non-code-mixed modes
        if not use_code_mixed:
            new_text = _restore_terms(new_text, placeholder_map)

        # Log any terms that still got transliterated
        if not use_code_mixed and glossary and terms_in_use:
            missing = verify_terms_preserved(text, new_text, glossary)
            if missing:
                logger.debug(
                    "Sarvam: terms still lost after placeholder restore in seg %d: %s",
                    i, missing,
                )

        seg_duration = seg.end - seg.start
        if seg_duration > 0:
            ratio = _estimate_duration_ratio(text, new_text, "en", target_lang)
            if ratio > 1.20 or ratio < 0.80:
                logger.debug(
                    "[Sarvam] Seg %d duration ratio=%.2f (%.1fs src → est %.1fs tts) "
                    "— may need FFmpeg stretch",
                    i, ratio, seg_duration, seg_duration * ratio,
                )
        translated.append(Segment(start=seg.start, end=seg.end, text=new_text))

        if (i + 1) % 50 == 0:
            logger.info("  %s: %d/%d …", lang_name, i + 1, len(segments))

    if fallback_count:
        logger.warning("%s: %d/%d segments fell back to English",
                       lang_name, fallback_count, len(segments))
    _fail_if_nothing_translated("Sarvam", lang_name, segments, fallback_count, last_error)
    logger.info("%s: %d segments translated ✓", lang_name, len(translated))
    return translated


# ── Google Cloud Translation v2 ───────────────────────────────────────────────

# GCP language codes for Cloud Translate v2
_GCP_LANG_CODES: dict[str, str] = {
    "hi": "hi",
    "te": "te",
    "od": "or",   # Odia BCP-47 is "or" in Cloud Translate
}

# Batch size: Cloud Translate v2 allows up to 128 strings per request
_GCP_BATCH_SIZE = 100


def _translate_batch_gcp(
    segments: list[Segment],
    target_lang: str,
    glossary: Glossary | None = None,
    terms_in_use: list[str] | None = None,
) -> list[Segment]:
    """
    Translate segments using Google Cloud Translation API v2.

    Uses the REST endpoint directly (no SDK needed — just requests + API key).
    Batches up to 100 segments per call for efficiency.

    Technical terms are protected with placeholder tokens (same mechanism as
    Sarvam) because Cloud Translate v2 does not support inline glossaries
    (that requires v3 with a service account).

    Falls back to the original English text on per-segment errors.
    """
    if not GCP_API_KEY:
        raise ValueError("GCP_API_KEY is not set in .env")

    lang_name = TARGET_LANGUAGES[target_lang]["name"]
    gcp_code = _GCP_LANG_CODES.get(target_lang, target_lang)

    url = f"https://translation.googleapis.com/language/translate/v2?key={GCP_API_KEY}"
    all_translated: list[Segment] = []
    fallback_count = 0
    last_error: Exception | None = None

    for batch_start in range(0, len(segments), _GCP_BATCH_SIZE):
        batch = segments[batch_start: batch_start + _GCP_BATCH_SIZE]

        # Protect technical terms before sending to Google Translate
        protected_texts: list[str] = []
        placeholder_maps: list[dict[str, str]] = []
        for seg in batch:
            ptext, pmap = _protect_terms(seg.text.strip(), glossary, terms_in_use)
            protected_texts.append(ptext or " ")   # empty string errors on API
            placeholder_maps.append(pmap)

        payload = {
            "q": protected_texts,
            "source": "en",
            "target": gcp_code,
            "format": "text",
        }

        @retry_on_failure(max_retries=3, backoff_base=2.0, retryable_exceptions=(Exception,))
        def _call_gcp(p):
            resp = requests.post(url, json=p, timeout=30)
            resp.raise_for_status()
            return resp.json()

        try:
            data = _call_gcp(payload)
            translations = data["data"]["translations"]
        except Exception as exc:
            logger.error("[GCP Translate] Batch %d failed: %s — falling back to English",
                         batch_start // _GCP_BATCH_SIZE + 1, exc)
            last_error = exc
            for seg in batch:
                all_translated.append(Segment(start=seg.start, end=seg.end, text=seg.text))
            fallback_count += len(batch)
            continue

        for i, (seg, item, pmap) in enumerate(zip(batch, translations, placeholder_maps)):
            raw_text = item.get("translatedText", seg.text)
            # Restore English technical terms the placeholder protected
            restored = _restore_terms(raw_text, pmap)

            if glossary and terms_in_use:
                missing = verify_terms_preserved(seg.text, restored, glossary)
                if missing:
                    logger.debug("[GCP Translate] Terms lost in seg %d: %s",
                                 batch_start + i, missing)

            all_translated.append(Segment(start=seg.start, end=seg.end, text=restored))

        logger.info("[GCP Translate] %s: batch %d/%d done",
                    lang_name, batch_start // _GCP_BATCH_SIZE + 1,
                    (len(segments) + _GCP_BATCH_SIZE - 1) // _GCP_BATCH_SIZE)

    if fallback_count:
        logger.warning("[GCP Translate] %s: %d/%d segments fell back to English",
                       lang_name, fallback_count, len(segments))
    _fail_if_nothing_translated("Google", lang_name, segments, fallback_count, last_error)
    logger.info("[GCP Translate] %s: %d segments translated ✓", lang_name, len(all_translated))
    return all_translated

