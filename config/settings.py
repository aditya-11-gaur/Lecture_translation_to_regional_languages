import os
from dotenv import load_dotenv

load_dotenv()

# Reduce CUDA memory fragmentation — must be set before torch allocates any CUDA memory.
# Picked up from .env; default to expandable_segments if not already set.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── API Keys ────────────────────────────────────────────────
# Add any new API key as  NAME_API_KEY  in your .env file.
# Register it in the dict below so the rest of the app picks it up.

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GCP_API_KEY    = os.getenv("GCP_API_KEY", "")

# Master registry: engine_id → { display name, env‑var name, key value, capabilities }
# To add a new engine later, just:
#   1. add its key to .env  (e.g. OPENAI_API_KEY=...)
#   2. add a row here
#
# capabilities: "stt" = speech-to-text, "translate" = translation
# key = "local" means no API key needed (runs locally)
ENGINE_REGISTRY: dict[str, dict] = {
    "whisper": {"name": "Whisper (Local)",               "env_var": None,              "key": "local",       "capabilities": ["stt"]},
    "gemini":  {"name": "Google Gemini",                 "env_var": "GEMINI_API_KEY",  "key": GEMINI_API_KEY, "capabilities": ["stt", "translate"]},
    "sarvam":  {"name": "Sarvam AI",                     "env_var": "SARVAM_API_KEY",  "key": SARVAM_API_KEY, "capabilities": ["stt", "translate"]},
    "gcp":     {"name": "Google Cloud (Translation+TTS)", "env_var": "GCP_API_KEY",    "key": GCP_API_KEY,   "capabilities": ["translate"]},
}


def _is_engine_available(info: dict) -> bool:
    """Check whether an engine is usable (local or has API key)."""
    return info["key"] == "local" or bool(info["key"])


def get_available_engines(capability: str | None = None) -> list[str]:
    """Return engine IDs that are usable, optionally filtered by capability."""
    out = []
    for eid, info in ENGINE_REGISTRY.items():
        if not _is_engine_available(info):
            continue
        if capability and capability not in info.get("capabilities", []):
            continue
        out.append(eid)
    return out


def get_default_engine(capability: str | None = None) -> str:
    """Return the best default engine for a given capability."""
    available = get_available_engines(capability)
    if not available:
        return "gemini"          # will fail at call-time with a clear error
    # For STT prefer whisper (word-level timestamps, no gaps), for translate prefer gcp/sarvam
    if capability == "stt":
        preference = ("whisper", "gemini", "sarvam")
    elif capability == "translate":
        preference = ("gcp", "sarvam", "gemini")
    else:
        preference = ("gemini", "whisper", "sarvam")
    for preferred in preference:
        if preferred in available:
            return preferred
    return available[0]

# ─── Sarvam API Base URL ─────────────────────────────────────
SARVAM_BASE_URL = "https://api.sarvam.ai"

# ─── Language Configuration ──────────────────────────────────
# Source language
SOURCE_LANG = "en"

# Target languages with their codes for Sarvam & display names
TARGET_LANGUAGES = {
    "hi": {"name": "Hindi",   "sarvam_code": "hi-IN", "bcp47": "hi-IN"},
    "te": {"name": "Telugu",  "sarvam_code": "te-IN", "bcp47": "te-IN"},
    "od": {"name": "Odia",    "sarvam_code": "od-IN", "bcp47": "or-IN"},
}

ALL_LANGUAGES = {
    "en": {"name": "English", "sarvam_code": "en-IN", "bcp47": "en-IN"},
    **TARGET_LANGUAGES,
}

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ─── Audio Settings ──────────────────────────────────────────
AUDIO_FORMAT = "wav"          # wav works best for STT
AUDIO_SAMPLE_RATE = 16000     # 16 kHz mono – standard for speech APIs

# ─── Subtitle Settings ───────────────────────────────────────
SUBTITLE_FORMAT = "srt"       # srt or vtt
MAX_CHARS_PER_LINE = 42       # for subtitle line wrapping

# ─── Voice Preservation ──────────────────────────────────────
ENABLE_VOICE_PRESERVATION = True   # Analyze & match speaker characteristics
VOICE_PITCH_SHIFT = True           # Apply pitch shifting post-TTS

# ─── Filler Preservation ─────────────────────────────────────
PRESERVE_FILLERS = True            # Keep um/uh/ah hesitation markers
INCLUDE_DISCOURSE_MARKERS = False  # Also keep "like", "you know", etc.

# ─── Code-Mixing / Glossary ──────────────────────────────────
ENABLE_CODE_MIXING = True          # Keep technical terms in English
CUSTOM_GLOSSARY_PATH = None        # Path to custom glossary JSON (optional)
GLOSSARY_CATEGORIES = None         # None = all, or list of category names

# ─── Pipeline Defaults ──────────────────────────────────────
# These are used by pipeline.py and main.py when no explicit value is provided.
DEFAULT_TARGET_LANGS  = ["hi"]     # Translate to Hindi only by default
DEFAULT_TTS_ENGINE    = "sarvam"   # Sarvam AI TTS – best for Indian languages without GCP key
SEPARATE_MUSIC        = True        # Always separate background music with Demucs
ENABLE_VOICE_CLONING  = True        # Apply OpenVoice v2 tone-color transfer by default
ENABLE_LIP_SYNC       = True       # Wav2Lip lip synchronisation (requires Wav2Lip + checkpoint)

# ─── Caching ─────────────────────────────────────────────────
ENABLE_CACHE = True                # Cache intermediate results to allow resume
CACHE_DIR_NAME = ".cache"          # Subdirectory name inside output dir
