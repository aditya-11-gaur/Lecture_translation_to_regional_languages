"""
setup_wav2lip.py
================
One-time setup for the Wav2Lip submodule.
Run this after cloning the repo (or after `git submodule update --init`):

    python setup_wav2lip.py

What it does
------------
1. Applies the librosa-compatibility patch to Wav2Lip/audio.py
   (librosa ≥0.10 requires keyword args for filters.mel)
2. Downloads the two required model weights (~500 MB total):
     • Wav2Lip/checkpoints/wav2lip_gan.pth   (416 MB)
     • Wav2Lip/face_detection/detection/sfd/s3fd.pth  (86 MB)
"""

from __future__ import annotations
import os, sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WAV2LIP   = os.path.join(REPO_ROOT, "Wav2Lip")

# ── 1. Apply audio.py patch ────────────────────────────────────────────────

AUDIO_PY = os.path.join(WAV2LIP, "audio.py")
if not os.path.isfile(AUDIO_PY):
    sys.exit(
        "ERROR: Wav2Lip/audio.py not found.\n"
        "Run: git submodule update --init --recursive"
    )

with open(AUDIO_PY, "r", encoding="utf-8") as f:
    src = f.read()

OLD = "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
NEW = "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"

if OLD in src:
    patched = src.replace(OLD, NEW)
    with open(AUDIO_PY, "w", encoding="utf-8") as f:
        f.write(patched)
    print("✓ Patched Wav2Lip/audio.py (librosa keyword args)")
elif NEW in src:
    print("✓ Wav2Lip/audio.py already patched")
else:
    print("⚠ Could not find expected line in audio.py — check manually")

# ── 2. Download model weights ───────────────────────────────────────────────
# Same sources the app uses (src/lip_sync.py): wav2lip_gan.pth from the official
# Google Drive folder via gdown, s3fd.pth from adrianbulat.com.

MODELS = [
    (os.path.join(WAV2LIP, "checkpoints", "wav2lip_gan.pth"), 416),
    (os.path.join(WAV2LIP, "face_detection", "detection", "sfd", "s3fd.pth"), 86),
]

# A partial download or an HTML error page would otherwise count as present
for path, expected_mb in MODELS:
    if os.path.isfile(path):
        actual_mb = os.path.getsize(path) / (1024 * 1024)
        if actual_mb < expected_mb * 0.9:
            print(f"⚠ {os.path.basename(path)} looks truncated ({actual_mb:.0f} MB) — re-downloading")
            os.remove(path)

print("\nDownloading model weights (this only runs once):")
sys.path.insert(0, REPO_ROOT)
from src.lip_sync import _ensure_wav2lip_checkpoints  # noqa: E402

try:
    _ensure_wav2lip_checkpoints(WAV2LIP)
except RuntimeError as exc:
    sys.exit(f"ERROR: {exc}")
for path, _ in MODELS:
    print(f"✓ {os.path.relpath(path, REPO_ROOT)} ({os.path.getsize(path) / (1024 * 1024):.0f} MB)")

print("\nSetup complete. Run the app with: streamlit run app.py")
