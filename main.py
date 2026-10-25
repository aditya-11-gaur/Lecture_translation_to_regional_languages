"""
NPTEL Lecture Translation Pipeline
───────────────────────────────────
Usage:
    python main.py <video_path> [--stt ENGINE] [--translate ENGINE] [--no-tts]

    Available engines are detected automatically from your .env file.

Example:
    python main.py input/lecture.mp4
    python main.py input/lecture.mp4 --stt whisper --translate gcp --no-tts
"""

import argparse
import json
import logging
import os
import sys

from config.settings import get_available_engines, get_default_engine, ENGINE_REGISTRY, DEFAULT_TARGET_LANGS, DEFAULT_TTS_ENGINE, TARGET_LANGUAGES
from src.utils import setup_logging
from src.pipeline import run_pipeline
from src.tts_generator import TTS_ENGINE_REGISTRY, get_available_tts_engines


def main():
    available = get_available_engines()

    parser = argparse.ArgumentParser(
        description="Translate NPTEL lecture videos into Hindi, Telugu & Odia"
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--stt",
        choices=[e for e, info in ENGINE_REGISTRY.items() if "stt" in info["capabilities"]],
        default=get_default_engine("stt"),
        help=f"Speech-to-Text backend (default: {get_default_engine('stt')}). Available: {', '.join(available) or 'none – add an API key!'}",
    )
    parser.add_argument(
        "--translate",
        choices=[e for e, info in ENGINE_REGISTRY.items() if "translate" in info["capabilities"]],
        default=get_default_engine("translate"),
        help=f"Translation backend (default: {get_default_engine('translate')}). Available: {', '.join(available) or 'none'}",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=list(TARGET_LANGUAGES.keys()),
        default=list(DEFAULT_TARGET_LANGS),
        metavar="LANG",
        help=f"Target language codes (default: {' '.join(DEFAULT_TARGET_LANGS)}). Choices: {', '.join(TARGET_LANGUAGES.keys())}",
    )
    parser.add_argument(
        "--tts-engine",
        choices=get_available_tts_engines(),
        default=DEFAULT_TTS_ENGINE,
        help=f"TTS engine (default: {DEFAULT_TTS_ENGINE}). Choices: {', '.join(TTS_ENGINE_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--no-music",
        action="store_true",
        help="Disable background music separation (on by default)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip TTS audio generation",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    # Set up structured logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    logger = logging.getLogger("nptel_pipeline")

    if not os.path.isfile(args.video):
        logger.error("Video file not found: %s", args.video)
        sys.exit(1)

    # Warn if chosen engine has no API key
    for label, engine in [("STT", args.stt), ("Translate", args.translate)]:
        if engine not in available:
            logger.warning("%s engine '%s' has no API key configured. It will fail at runtime.",
                           label, engine)

    result = run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        stt_method=args.stt,
        translate_method=args.translate,
        do_tts=not args.no_tts,
        tts_engine=args.tts_engine,
        target_langs=args.langs,
        separate_music=not args.no_music,
    )

    # Dump summary
    logger.info("Generated artefacts:\n%s",
                json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
