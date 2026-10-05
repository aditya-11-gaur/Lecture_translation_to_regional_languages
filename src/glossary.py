"""
glossary.py
───────────
Technical term glossary for code-mixing in translations.

NPTEL lectures use many English technical terms that should NOT be
translated but instead kept as-is (or transliterated) in the target
language output.  This module provides:

  • Dynamic glossary generation via LLM (Gemini) – analyses the actual
    transcript and extracts terms specific to the lecture
  • A static fallback glossary of common CS / Math / Engineering terms
  • A way to load custom glossaries from JSON files
  • Prompt-building helpers for translation engines
  • Post-processing to verify terms were preserved
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger("nptel_pipeline")

# ── Default technical terms ──────────────────────────────────────────────────

DEFAULT_GLOSSARY: dict[str, list[str]] = {
    "computer_science": [
        "algorithm", "API", "array", "backend", "bandwidth", "binary",
        "bit", "blockchain", "boolean", "browser", "buffer", "bug",
        "byte", "cache", "callback", "class", "client", "cloud",
        "cluster", "code", "codec", "compiler", "computational complexity",
        "CPU", "CUDA", "daemon", "data structure", "database",
        "deadlock", "debug", "deep learning", "DevOps", "DNS",
        "Docker", "driver", "encryption", "endpoint", "Ethernet",
        "exception", "fiber", "firewall", "firmware", "framework",
        "frontend", "function", "garbage collector", "gateway", "Git",
        "GPU", "graph", "GUI", "hash", "hash table", "heap",
        "HTTP", "HTTPS", "hypervisor", "IDE", "index", "inheritance",
        "instance", "interface", "internet", "interpreter", "IP",
        "iterator", "Java", "JavaScript", "JSON", "kernel",
        "Kubernetes", "latency", "library", "linked list", "Linux",
        "load balancer", "localhost", "log", "loop", "machine learning",
        "malware", "memory", "merge sort", "metadata", "method",
        "microservice", "middleware", "module", "mutex",
        "namespace", "neural network", "node", "NoSQL",
        "object", "OOP", "open source", "operating system",
        "overflow", "packet", "pagination", "parameter", "parser",
        "patch", "pipeline", "pixel", "pointer", "polymorphism",
        "port", "process", "processor", "programming", "protocol",
        "proxy", "Python", "query", "queue", "RAM", "recursion",
        "Redis", "refactor", "regex", "register", "repository",
        "REST", "ROM", "router", "runtime", "sandbox",
        "scheduler", "schema", "script", "SDK", "semaphore",
        "server", "servlet", "shell", "socket", "software",
        "SQL", "SSH", "SSL", "stack", "startup", "string",
        "subnet", "subroutine", "switch", "syntax", "TCP",
        "template", "tensor", "terminal", "thread", "throughput",
        "TLS", "token", "topology", "transaction", "tree",
        "tuple", "type", "UDP", "UI", "Unicode", "unit test",
        "URL", "USB", "variable", "virtual machine", "VLAN",
        "VM", "VPN", "webhook", "XML", "YAML",
    ],
    "mathematics": [
        "algebra", "asymptote", "Bayes", "Bayesian",
        "bijection", "binomial", "calculus", "Cartesian", "Cauchy",
        "combinatorics", "complex number", "convex", "corollary",
        "cosine", "degree", "derivative", "determinant",
        "differential equation", "Dirichlet", "distribution",
        "divergence", "eigenvalue", "eigenvector", "Euler",
        "exponential", "factorial", "Fibonacci", "Fourier",
        "Gaussian", "gradient",
        "Hamiltonian", "Hessian", "homomorphism", "hypothesis",
        "identity", "inequality", "infinity", "integral",
        "interpolation", "inverse", "isomorphism", "Jacobian",
        "Lagrangian", "Laplacian", "lemma", "limit",
        "linear algebra", "logarithm", "manifold", "Markov",
        "matrix", "maximum", "mean", "median", "minimum",
        "modular", "modulus", "Monte Carlo", "norm",
        "normal distribution", "operator", "optimal", "orthogonal",
        "partial derivative", "permutation",
        "Poisson", "polynomial", "prime", "probability",
        "proof", "proposition", "quadratic", "quaternion",
        "random variable", "real number", "regression",
        "Riemann", "scalar", "sequence", "series", "set",
        "sigma", "simplex", "sine", "singular", "spectrum",
        "standard deviation", "statistic", "stochastic",
        "subspace", "summation", "symmetric", "tangent",
        "Taylor series", "theorem", "topology",
        "transformation", "transpose", "trigonometry",
        "variance", "vector", "Venn diagram",
    ],
    "engineering": [
        "actuator", "amplifier", "analog", "antenna",
        "capacitor", "circuit", "controller", "current",
        "DAC", "diode", "digital", "DSP", "duty cycle",
        "electrode", "EMF", "feedback", "filter", "frequency",
        "gain", "ground", "IC", "impedance", "inductor",
        "inverter", "MOSFET", "multiplexer", "noise",
        "ohm", "op-amp", "oscillator", "PCB", "PID",
        "power supply", "PWM", "rectifier", "relay",
        "resistance", "resonance", "RF", "semiconductor",
        "sensor", "signal", "simulation", "SPICE",
        "thermistor", "transducer", "transformer",
        "transistor", "VLSI", "voltage", "waveform",
    ],
    "general_academic": [
        "abstract", "analysis", "appendix", "benchmark",
        "case study", "citation", "conclusion", "correlation",
        "data", "dataset", "empirical", "experiment",
        "hypothesis", "methodology",
        "metric", "model", "observation", "optimization",
        "peer review", "prototype", "reference",
        "research", "result", "sample",
        "state of the art", "survey", "thesis", "validation",
    ],
}


@dataclass
class Glossary:
    """
    A configurable glossary of technical terms to preserve during translation.
    """
    terms: set[str] = field(default_factory=set)
    _pattern: re.Pattern | None = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self.terms:
            self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile a regex pattern for matching terms (case-insensitive)."""
        if not self.terms:
            self._pattern = None
            return
        # Sort by length (longest first) to match longest terms first
        sorted_terms = sorted(self.terms, key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_terms]
        self._pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE,
        )

    def add_terms(self, terms: list[str] | set[str]) -> None:
        """Add terms to the glossary."""
        self.terms.update(t.strip() for t in terms if t.strip())
        self._compile_pattern()

    def remove_terms(self, terms: list[str] | set[str]) -> None:
        """Remove terms from the glossary."""
        self.terms -= set(terms)
        self._compile_pattern()

    def find_terms_in_text(self, text: str) -> list[str]:
        """Find all glossary terms present in the given text."""
        if not self._pattern:
            return []
        return list(set(self._pattern.findall(text)))

    def build_translation_prompt_section(
        self, terms_in_use: list[str] | None = None,
    ) -> str:
        """
        Build a prompt section instructing the translator to preserve terms.

        Args:
            terms_in_use: If given, only include these terms. Otherwise use all.
        """
        terms = terms_in_use or sorted(self.terms)
        if not terms:
            return ""

        # Deduplicate and limit to reasonable size for prompt
        unique_terms = sorted(set(terms))
        if len(unique_terms) > 100:
            unique_terms = unique_terms[:100]

        return (
            "\n\nIMPORTANT — Code-mixing / Technical Terms:\n"
            "The following technical terms must be kept EXACTLY in English as-is. "
            "Do NOT translate them, do NOT transliterate them, and do NOT add a "
            "translation or transliteration in parentheses after them. "
            "Just use the English term directly inside the target-language sentence. "
            "For example, 'The algorithm is efficient' in Hindi → "
            "'यह algorithm बहुत efficient है।' — NOT 'algorithm (एल्गोरिदम)'.\n"
            "Terms to preserve: " + ", ".join(unique_terms)
        )

    def extract_terms_from_segments(self, segments) -> list[str]:
        """Extract unique technical terms found across all segments."""
        found: set[str] = set()
        for seg in segments:
            text = seg.text if hasattr(seg, "text") else str(seg)
            found.update(self.find_terms_in_text(text))
        return sorted(found)


# ── Dynamic glossary generation via LLM ──────────────────────────────────────

_GLOSSARY_PROMPT = """\
You are an expert at identifying technical terms in educational lecture transcripts.

Given the following transcript excerpt from an NPTEL lecture, extract ALL English \
technical terms, proper nouns, acronyms, named algorithms, named theorems, \
library/framework names, and domain-specific jargon that should be kept in \
English (not translated) when dubbing the lecture into an Indian regional language \
like Hindi, Telugu, or Odia.

Rules:
- Include terms that a native speaker of the target language would naturally \
  say in English even while speaking their own language (code-mixing).
- Include: algorithm names (e.g. "merge sort"), data structure names, \
  mathematical concepts (e.g. "Fourier transform"), programming language \
  names, library names, acronyms (e.g. "CPU", "API"), metric names, \
  named theorems, tool names, and any other domain-specific terms.
- Do NOT include common English words that have natural translations \
  (e.g. "computer" → कंप्यूटर is fine to translate; but "Python" the \
  language should stay as "Python").
- Do NOT include stop words, articles, prepositions, or generic verbs.
- Return ONLY a JSON array of strings — no markdown, no code fences, \
  no explanation.

Example output: ["merge sort", "API", "Fourier transform", "Python", "TCP/IP"]

Transcript:
{transcript_text}
"""

# Maximum characters of transcript to send in a single glossary-generation call.
_MAX_TRANSCRIPT_CHARS = 12_000


def generate_glossary_from_transcript(
    segments,
    *,
    gemini_api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    fallback_to_static: bool = True,
) -> Glossary:
    """
    Dynamically generate a glossary by sending the transcript to an LLM.

    The model analyses the actual lecture content and returns the specific
    technical terms that should be preserved in English during translation.
    This is far more accurate than a fixed static list because it adapts
    to the subject matter of each individual lecture.

    Args:
        segments:           List of transcript Segment objects (or any objects
                            with a ``.text`` attribute).
        gemini_api_key:     Gemini API key.  Falls back to env / settings if None.
        model:              Gemini model to use.
        fallback_to_static: If the API call fails, fall back to the static
                            default glossary instead of raising.

    Returns:
        A Glossary populated with terms extracted from the transcript.
    """
    # ── Resolve API key ──────────────────────────────────────────────────
    if gemini_api_key is None:
        try:
            from config.settings import GEMINI_API_KEY
            gemini_api_key = GEMINI_API_KEY
        except ImportError:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    if not gemini_api_key:
        logger.warning("No Gemini API key available for dynamic glossary; "
                       "falling back to static glossary.")
        return load_default_glossary() if fallback_to_static else Glossary()

    # ── Build a condensed transcript ─────────────────────────────────────
    texts: list[str] = []
    char_count = 0
    for seg in segments:
        t = seg.text if hasattr(seg, "text") else str(seg)
        t = t.strip()
        if not t:
            continue
        if char_count + len(t) > _MAX_TRANSCRIPT_CHARS:
            # Truncate to stay within prompt limits
            remaining = _MAX_TRANSCRIPT_CHARS - char_count
            if remaining > 50:
                texts.append(t[:remaining] + "…")
            break
        texts.append(t)
        char_count += len(t)

    transcript_text = "\n".join(texts)
    if not transcript_text.strip():
        logger.warning("Empty transcript; returning empty glossary.")
        return Glossary()

    # ── Call the LLM ─────────────────────────────────────────────────────
    try:
        from google import genai
        from google.genai import types
        from src.utils import retry_on_failure

        client = genai.Client(api_key=gemini_api_key)

        prompt = _GLOSSARY_PROMPT.format(transcript_text=transcript_text)

        @retry_on_failure(
            max_retries=2, backoff_base=2.0,
            retryable_exceptions=(Exception,),
        )
        def _call():
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

        response = _call()
        raw = response.text.strip()

        # Strip markdown fences if present
        json_text = raw
        if json_text.startswith("```"):
            json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
            json_text = re.sub(r"\s*```$", "", json_text)

        data = json.loads(json_text)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        terms = {str(t).strip() for t in data if isinstance(t, str) and t.strip()}
        logger.info("Dynamic glossary: LLM extracted %d terms from transcript", len(terms))

        glossary = Glossary(terms=terms)
        return glossary

    except Exception as exc:
        logger.warning("Dynamic glossary generation failed: %s", exc)
        if fallback_to_static:
            logger.info("Falling back to static default glossary.")
            return load_default_glossary()
        raise


def load_default_glossary(categories: list[str] | None = None) -> Glossary:
    """
    Load the built-in technical glossary.

    Args:
        categories: Which categories to include. None = all.
    """
    terms: set[str] = set()
    cats = categories or list(DEFAULT_GLOSSARY.keys())
    for cat in cats:
        if cat in DEFAULT_GLOSSARY:
            terms.update(DEFAULT_GLOSSARY[cat])
    glossary = Glossary(terms=terms)
    logger.info("Loaded default glossary: %d terms from %s", len(terms), cats)
    return glossary


def load_glossary_from_file(path: str) -> Glossary:
    """
    Load a custom glossary from a JSON file.

    Expected format:
        {"terms": ["word1", "word2", ...]}
    or:
        {"category1": ["word1", ...], "category2": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms: set[str] = set()
    if isinstance(data, dict):
        if "terms" in data and isinstance(data["terms"], list):
            terms.update(data["terms"])
        else:
            for key, values in data.items():
                if isinstance(values, list):
                    terms.update(values)
    elif isinstance(data, list):
        terms.update(data)

    glossary = Glossary(terms=terms)
    logger.info("Loaded custom glossary from %s: %d terms", path, len(terms))
    return glossary


def verify_terms_preserved(
    original_text: str,
    translated_text: str,
    glossary: Glossary,
) -> list[str]:
    """
    Check which glossary terms from the original are missing in the translation.

    Returns:
        List of terms that were in the original but NOT in the translated text.
    """
    original_terms = set(glossary.find_terms_in_text(original_text))
    # Check both original form and common transliteration patterns
    missing = []
    for term in original_terms:
        if term.lower() not in translated_text.lower():
            missing.append(term)
    return missing
