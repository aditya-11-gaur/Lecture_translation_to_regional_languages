"""
translator.py
─────────────
Translate English transcript segments into target languages (Hindi, Telugu, Odia).

Supports two backends:
  • Sarvam Translate API
  • Gemini (batch translation per language)

Input:  list of Segment objects (English)
Output: dict  { "hi": [Segment, ...], "te": [...], "od": [...] }
        Each translated segment preserves the original start/end timestamps.
"""

from __future__ import annotations
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
import requests
from config.settings import (
    SARVAM_API_KEY,
    SARVAM_BASE_URL,
    GEMINI_API_KEY,
    GCP_API_KEY,
    TARGET_LANGUAGES,
    ENABLE_CODE_MIXING,
)
from src.transcriber import Segment
from src.glossary import Glossary, generate_glossary_from_transcript, load_default_glossary, verify_terms_preserved
from src.utils import retry_on_failure

logger = logging.getLogger("nptel_pipeline")


# ── Maximum segments per Gemini batch call ────────────────────────────────────
_BATCH_SIZE = 40  # keep well within token limits


def translate_segments(
    segments: list[Segment],
    target_langs: list[str] | None = None,
    method: str = "gemini",
    glossary: Glossary | None = None,
) -> dict[str, list[Segment]]:
    """
    Translate all segments to each target language.

    Args:
        segments:     English transcript segments.
        target_langs: Language codes to translate into (default: all targets).
        method:       "sarvam" or "gemini".
        glossary:     Technical term glossary for code-mixing (auto-loaded if None).

    Returns:
        Dict mapping language code → list of translated Segments.
    """
    if target_langs is None:
        target_langs = list(TARGET_LANGUAGES.keys())

    # Auto-generate dynamic glossary from transcript if not provided
    if glossary is None and ENABLE_CODE_MIXING:
        glossary = generate_glossary_from_transcript(
            segments, fallback_to_static=True,
        )

    # Extract terms actually used in the transcript for focused prompts
    terms_in_use: list[str] = []
    if glossary:
        terms_in_use = glossary.extract_terms_from_segments(segments)
        logger.info("Code-mixing: %d technical terms found in transcript", len(terms_in_use))

    results: dict[str, list[Segment]] = {}

    def _translate_one_lang(lang: str) -> tuple[str, list[Segment]]:
        lang_name = TARGET_LANGUAGES[lang]["name"]
        logger.info("Translating to %s (%s)…", lang_name, lang)
        if method == "gemini":
            translated = _translate_batch_gemini(
                segments, lang, glossary=glossary, terms_in_use=terms_in_use,
            )
        elif method == "sarvam":
            translated = _translate_batch_sarvam(
                segments, lang, glossary=glossary, terms_in_use=terms_in_use,
            )
        elif method == "gcp":
            translated = _translate_batch_gcp(
                segments, lang, glossary=glossary, terms_in_use=terms_in_use,
            )
        else:
            raise ValueError(f"Unknown translation method: {method}")
        return lang, translated

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


# ── Gemini batch translation ─────────────────────────────────────────────────

def _translate_batch_gemini(
    segments: list[Segment], target_lang: str,
    glossary: Glossary | None = None,
    terms_in_use: list[str] | None = None,
) -> list[Segment]:
    """
    Translate a list of segments to `target_lang` using Gemini.
    Sends segments in batches to stay within token limits.
    Integrates glossary for code-mixing of technical terms.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=GEMINI_API_KEY)

    lang_name = TARGET_LANGUAGES[target_lang]["name"]
    all_translated: list[Segment] = []

    # Build code-mixing prompt section
    code_mix_section = ""
    if glossary and terms_in_use:
        code_mix_section = glossary.build_translation_prompt_section(terms_in_use)

    # Process in batches
    for batch_start in range(0, len(segments), _BATCH_SIZE):
        batch = segments[batch_start : batch_start + _BATCH_SIZE]
        batch_texts = [{"id": i, "text": seg.text} for i, seg in enumerate(batch)]

        prompt = f"""Translate the following English text segments into {lang_name}.

Return ONLY a valid JSON array (no markdown, no code fences) where each element is:
{{"id": <same id as input>, "text": "<translated text in {lang_name}>"}}

Rules:
- Translate EVERY segment. Do not skip any.
- Keep technical terms, proper nouns, and acronyms EXACTLY in English as-is. Do NOT transliterate them into the target script and do NOT add transliterations in parentheses.
- The translation should be natural and fluent in {lang_name}.
- Do NOT repeat words or phrases. Each sentence should be concise without redundancy.
- Do NOT add filler translations or paraphrase the same idea multiple times.
- Keep the same number of segments as input.
- The translated text should be roughly similar in length to the source English text.
- Return ONLY the JSON array, nothing else.{code_mix_section}

Input segments:
{json.dumps(batch_texts, ensure_ascii=False)}"""

        logger.info("  Batch %d (%d segments) → %s",
                    batch_start // _BATCH_SIZE + 1, len(batch), lang_name)

        @retry_on_failure(
            max_retries=3, backoff_base=2.0,
            retryable_exceptions=(Exception,),
        )
        def _call_gemini(p):
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=p,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )

        response = _call_gemini(prompt)
        raw = response.text.strip()

        # Strip markdown fences if present
        json_text = raw
        if json_text.startswith("```"):
            json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
            json_text = re.sub(r"\s*```$", "", json_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Gemini translation JSON: %s", e)
            logger.debug("Raw:\n%s", raw[:500])
            raise ValueError(f"Gemini returned invalid JSON for {lang_name}: {e}")

        # Build a lookup by id for robustness
        translated_map = {item["id"]: item["text"] for item in data}

        for i, seg in enumerate(batch):
            new_text = translated_map.get(i, seg.text)  # fallback to English

            # Verify code-mixing: check technical terms are preserved
            if glossary and terms_in_use:
                missing = verify_terms_preserved(seg.text, new_text, glossary)
                if missing:
                    logger.debug("Terms lost in translation for seg %d: %s", i, missing)

            all_translated.append(
                Segment(start=seg.start, end=seg.end, text=new_text)
            )

    logger.info("%s: %d segments translated ✓", lang_name, len(all_translated))
    return all_translated


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

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            translated.append(Segment(start=seg.start, end=seg.end, text=""))
            continue

        # Protect technical terms before handing text to Sarvam.
        # Sarvam is a direct translation API (no custom prompt), so we
        # substitute glossary terms with opaque placeholder tokens that the
        # MT engine will leave untouched, then restore them afterwards.
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
            )

        try:
            resp = _call_sarvam(protected_text)
            new_text = resp.translated_text or protected_text
        except Exception as exc:
            logger.warning("Sarvam error on segment %d: %s", i + 1, exc)
            new_text = text  # keep original on failure; no placeholders to restore
            placeholder_map = {}
            fallback_count += 1

        # Restore placeholders → original English terms
        new_text = _restore_terms(new_text, placeholder_map)

        # Log any terms that still got transliterated despite protection
        if glossary and terms_in_use:
            missing = verify_terms_preserved(text, new_text, glossary)
            if missing:
                logger.debug(
                    "Sarvam: terms still lost after placeholder restore in seg %d: %s",
                    i, missing,
                )

        translated.append(Segment(start=seg.start, end=seg.end, text=new_text))

        if (i + 1) % 50 == 0:
            logger.info("  %s: %d/%d …", lang_name, i + 1, len(segments))

    if fallback_count:
        logger.warning("%s: %d/%d segments fell back to English",
                       lang_name, fallback_count, len(segments))
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
    logger.info("[GCP Translate] %s: %d segments translated ✓", lang_name, len(all_translated))
    return all_translated

