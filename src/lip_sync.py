"""
lip_sync.py
───────────
Lip synchronisation for dubbed video, using the Wav2Lip model
("A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild",
https://github.com/Rudrabha/Wav2Lip).

The heavy lifting runs in src/lipsync_worker.py as its own process (GPU
memory is fully released afterwards).  Wav2Lip's inference.py is not used:
the worker streams the video frame by frame, so every output frame stays at
its source timestamp; finds faces once per video and reuses them for every
language; regenerates only the mouth and jaw, blended in with a feathered
mask so the rest of the face keeps its original sharpness; and encodes once
at high quality.  Frames without a face pass through unchanged.

Setup (one-time):  git submodule update --init && python setup_wav2lip.py

When Wav2Lip or its checkpoints are missing, the module falls back to a
passthrough that only swaps the audio track, so the pipeline always
produces output.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("nptel_pipeline")

# Default path – can be overridden by the WAV2LIP_PATH environment variable.
# Priority:
#   1. WAV2LIP_PATH env var (explicit override)
#   2. Wav2Lip/ inside this repo  (after git submodule update --init)
#   3. ../Wav2Lip  relative to the repo root (original manual-clone location)
#
#   __file__ → .../multilingual_video_translation/src/lip_sync.py
#   2× dirname → .../multilingual_video_translation/  (repo root)
#   3× dirname → .../working_project/
_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IN_REPO      = os.path.join(_REPO_ROOT, "Wav2Lip")         # submodule location
_SIBLING      = os.path.join(os.path.dirname(_REPO_ROOT), "Wav2Lip")  # ../Wav2Lip

if os.path.isdir(_IN_REPO):
    _DEFAULT_WAV2LIP_PATH = _IN_REPO
else:
    _DEFAULT_WAV2LIP_PATH = _SIBLING

WAV2LIP_REPO = os.environ.get("WAV2LIP_PATH", _DEFAULT_WAV2LIP_PATH)

# Google Drive file IDs for the pretrained checkpoints
_WAV2LIP_GAN_GDRIVE_ID  = "1_OvqStxNxLc7bXzlaVG5sz695p-FVfYY"   # wav2lip_gan.pth (~416 MB)
_S3FD_URL               = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"


def _ensure_wav2lip_checkpoints(wav2lip_path: str = WAV2LIP_REPO) -> None:
    """
    Auto-download required Wav2Lip checkpoints if not already present.

    Downloads (one-time, ~430 MB total):
      • checkpoints/wav2lip_gan.pth  — from Google Drive via gdown
      • face_detection/detection/sfd/s3fd.pth — from adrianbulat.com
    """
    # ── wav2lip_gan.pth ──────────────────────────────────────────────────────
    ckpt_dir  = os.path.join(wav2lip_path, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "wav2lip_gan.pth")
    if not os.path.isfile(ckpt_path):
        logger.info(
            "wav2lip_gan.pth not found — downloading from Google Drive (~436 MB)…"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            import gdown
            gdown.download(
                id=_WAV2LIP_GAN_GDRIVE_ID,
                output=ckpt_path,
                quiet=False,
            )
            logger.info("Downloaded wav2lip_gan.pth → %s", ckpt_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download wav2lip_gan.pth: {exc}\n"
                "Manual download from the official Wav2Lip folder:\n"
                "  https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k\n"
                f"  → place wav2lip_gan.pth at {ckpt_path}"
            ) from exc

    # ── s3fd.pth (face detection) ────────────────────────────────────────────
    sfd_dir  = os.path.join(wav2lip_path, "face_detection", "detection", "sfd")
    sfd_path = os.path.join(sfd_dir, "s3fd.pth")
    if not os.path.isfile(sfd_path):
        logger.info("s3fd.pth not found — downloading face detection model (~85 MB)…")
        os.makedirs(sfd_dir, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(_S3FD_URL, sfd_path)
            logger.info("Downloaded s3fd.pth → %s", sfd_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download s3fd.pth: {exc}\n"
                f"Manual download: {_S3FD_URL}\n"
                f"  → place at {sfd_path}"
            ) from exc


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class LipSyncResult:
    """Result of a single lip-sync run."""
    output_path: str
    success: bool
    language: str
    processing_time_sec: float
    fallback_used: bool = False             # True when Wav2Lip was unavailable or failed
    error_message: Optional[str] = None
    face_frames: int = 0                    # frames whose mouth was regenerated
    total_frames: int = 0


# ── Public helper ────────────────────────────────────────────────────────────

def is_wav2lip_available(
    wav2lip_path: str = WAV2LIP_REPO,
    checkpoint: str = "checkpoints/wav2lip_gan.pth",
) -> bool:
    """
    Return True only when Wav2Lip is properly installed with all checkpoints.
    The pipeline can run without it (passthrough mode), so this is checked
    only to surface status in the UI.
    """
    inference_py = os.path.join(wav2lip_path, "inference.py")
    ckpt_path    = os.path.join(wav2lip_path, checkpoint)
    sfd_path     = os.path.join(wav2lip_path, "face_detection", "detection", "sfd", "s3fd.pth")
    return (
        os.path.isfile(inference_py)
        and os.path.isfile(ckpt_path)
        and os.path.isfile(sfd_path)
    )


def is_wav2lip_repo_present(wav2lip_path: str = WAV2LIP_REPO) -> bool:
    """Return True if the Wav2Lip repo is cloned (inference.py exists), regardless of checkpoints."""
    return os.path.isfile(os.path.join(wav2lip_path, "inference.py"))


# ── Main processor class ──────────────────────────────────────────────────────

class LipSyncProcessor:
    """
    Make the speaker's mouth match a dubbed audio track.

    Typical usage::

        processor = LipSyncProcessor(face_cache_dir="output/job/lipsync")
        result = processor.sync("lecture.mp4", "lecture_hi_mixed.mp3",
                                "output/job/lipsync/lipsync_hi.mp4", language="hi")

    Face boxes are cached in *face_cache_dir*, so syncing a second language
    of the same video skips face detection.
    """

    def __init__(
        self,
        wav2lip_path: str = WAV2LIP_REPO,
        checkpoint: str = "checkpoints/wav2lip_gan.pth",
        face_cache_dir: Optional[str] = None,
    ):
        self.wav2lip_path = wav2lip_path
        self.checkpoint = os.path.join(wav2lip_path, checkpoint)
        self.face_cache_dir = face_cache_dir

        # Auto-download checkpoints if Wav2Lip repo is present but weights are missing
        if os.path.isfile(os.path.join(wav2lip_path, "inference.py")):
            try:
                _ensure_wav2lip_checkpoints(wav2lip_path)
            except Exception as exc:
                logger.warning("Wav2Lip checkpoint download failed: %s", exc)

        self._available = is_wav2lip_available(wav2lip_path, checkpoint)
        if not self._available:
            logger.warning(
                "Wav2Lip not ready at '%s' — lip sync will use PASSTHROUGH mode "
                "(audio-swap only, no facial animation).",
                wav2lip_path,
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        language: str = "unknown",
    ) -> LipSyncResult:
        """
        Apply lip synchronisation: make the speaker's mouth match *audio_path*.

        Args:
            video_path:   Original video containing the speaker's face.
            audio_path:   Dubbed / translated audio track (same length as the video).
            output_path:  Destination MP4 path for the synced output.
            language:     Target language code (for logging).

        Returns:
            :class:`LipSyncResult` — always produces a file (falls back to passthrough).
        """
        t_start = time.time()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not self._available:
            logger.warning("Wav2Lip unavailable — using audio-swap passthrough for '%s'", language)
            result = self._passthrough(video_path, audio_path, output_path, language)
            result.fallback_used = True
            result.processing_time_sec = time.time() - t_start
            return result

        try:
            logger.info("Lip sync (Wav2Lip) for language: %s", language)
            stats = self._run_worker(video_path, audio_path, output_path)
            logger.info(
                "Lip sync [%s]: %d/%d frames with a face, %.0fs on %s",
                language, stats["face_frames"], stats["frames"], stats["seconds"], stats["device"],
            )
            return LipSyncResult(
                output_path=output_path,
                success=True,
                language=language,
                processing_time_sec=time.time() - t_start,
                face_frames=stats["face_frames"],
                total_frames=stats["frames"],
            )
        except Exception as exc:
            logger.error("Wav2Lip failed for '%s': %s — falling back to audio-swap", language, exc)
            result = self._passthrough(video_path, audio_path, output_path, language)
            result.fallback_used = True
            result.error_message = str(exc)
            result.processing_time_sec = time.time() - t_start
            return result

    def batch_sync(
        self,
        video_path: str,
        audio_paths: Dict[str, str],
        output_dir: str,
    ) -> Dict[str, LipSyncResult]:
        """Lip-sync every language: ``{lang_code: audio_path}`` → ``{lang_code: result}``."""
        os.makedirs(output_dir, exist_ok=True)
        return {
            lang: self.sync(video_path, audio, os.path.join(output_dir, f"lipsync_{lang}.mp4"), lang)
            for lang, audio in audio_paths.items()
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_worker(self, video_path: str, audio_path: str, output_path: str) -> dict:
        """Run src/lipsync_worker.py in a child process; return its stats."""
        with tempfile.TemporaryDirectory(prefix="lipsync_") as tmp:
            request = {
                "video": os.path.abspath(video_path),
                "audio": os.path.abspath(audio_path),
                "output": os.path.abspath(output_path),
                "wav2lip_dir": os.path.abspath(self.wav2lip_path),
                "checkpoint": os.path.abspath(self.checkpoint),
                "box_cache": (os.path.join(os.path.abspath(self.face_cache_dir), "face_boxes.npz")
                              if self.face_cache_dir else None),
                "result": os.path.join(tmp, "result.json"),
            }
            req_path = os.path.join(tmp, "request.json")
            with open(req_path, "w", encoding="utf-8") as f:
                json.dump(request, f)
            proc = subprocess.run(
                [sys.executable, "-m", "src.lipsync_worker", req_path],
                cwd=_REPO_ROOT, capture_output=True, text=True, errors="replace",
            )
            if not os.path.isfile(request["result"]):
                raise RuntimeError(f"lip-sync process exited with code {proc.returncode}:\n"
                                   f"{proc.stderr[-2000:]}")
            with open(request["result"], encoding="utf-8") as f:
                stats = json.load(f)
        if "error" in stats:
            raise RuntimeError(stats["error"])
        return stats

    def _passthrough(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        language: str,
    ) -> LipSyncResult:
        """
        Fallback: replace the audio track only using ffmpeg (no face animation).
        This always produces a valid output file so the pipeline is never blocked.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i",  video_path,
            "-i",  audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            return LipSyncResult(
                output_path=output_path,
                success=True,
                language=language,
                processing_time_sec=0.0,
            )
        except subprocess.CalledProcessError as exc:
            return LipSyncResult(
                output_path=output_path,
                success=False,
                language=language,
                processing_time_sec=0.0,
                error_message=str(exc),
            )
