"""
lipsync_worker.py
─────────────────
Wav2Lip lip sync for one video + one dubbed audio track.  Runs as its own
process (started by src/lip_sync.py) so every byte of GPU memory is returned
when it exits:

    python -m src.lipsync_worker <request.json>

Compared with Wav2Lip's own inference.py it:
  • streams frames instead of loading the video into RAM (no 30 s chunks),
  • keeps output frame n at the source time of frame n — cutting the video
    into face / no-face clips and re-joining them had put the mouth 4 frames
    (0.13 s) behind,
  • finds faces once per video (every DETECT_EVERY-th frame, interpolated and
    smoothed) and caches the boxes for the other languages,
  • blends only the generated mouth/jaw area with a feathered mask instead of
    pasting the whole 96×96 face box,
  • encodes once with x264 at CRF 18 (Wav2Lip went via a DIVX .avi).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from fractions import Fraction

import cv2
import numpy as np

DETECT_EVERY = 3          # run the face detector on every 3rd frame
DETECT_MAX_WIDTH = 640    # detect on frames scaled to at most this width
SMOOTH_FRAMES = 5         # moving-average window for face boxes
PADS = (0, 10, 0, 0)      # top, bottom, left, right padding (px) — includes the chin
BATCH = 32                # faces per Wav2Lip forward pass
IMG_SIZE = 96             # Wav2Lip's face resolution
MEL_STEP = 16             # mel frames per video frame window
CRF = 18                  # output quality (lower = better)


# ── Video I/O ────────────────────────────────────────────────────────────────

def probe(video_path: str) -> tuple[float, int, int]:
    """(fps, width, height) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height", "-of", "json", video_path],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return float(Fraction(s["r_frame_rate"])), int(s["width"]), int(s["height"])


def read_frames(video_path: str, fps: float, width: int, height: int):
    """Yield BGR frames at a constant *fps*, optionally scaled to width×height."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video_path, "-r", f"{fps:.6f}",
         "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


# ── Face boxes ───────────────────────────────────────────────────────────────

def detect_boxes(video_path: str, fps: float, width: int, height: int, device: str) -> np.ndarray:
    """
    Face box (x1, y1, x2, y2) for every frame, NaN where there is no face.

    The detector runs on every DETECT_EVERY-th frame (downscaled); frames in
    between get boxes interpolated from the two detections around them, and
    only when both found a face.  Boxes are then smoothed within each run of
    face frames so the generated mouth doesn't jitter.
    """
    import face_detection

    scale = min(1.0, DETECT_MAX_WIDTH / width)
    dw, dh = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device,
    )

    sampled: dict[int, tuple | None] = {}
    batch, batch_idx = [], []

    def run_batch():
        for idx, det in zip(batch_idx, detector.get_detections_for_batch(np.array(batch))):
            sampled[idx] = None if det is None else tuple(v / scale for v in det)
        batch.clear(); batch_idx.clear()

    n = 0
    for frame in read_frames(video_path, fps, dw, dh):
        if n % DETECT_EVERY == 0:
            batch.append(frame); batch_idx.append(n)
            if len(batch) == 16:
                run_batch()
        n += 1
    if batch:
        run_batch()
    del detector

    return boxes_from_samples(sampled, n, width, height)


def boxes_from_samples(
    sampled: dict[int, tuple | None], n: int, width: int, height: int,
) -> np.ndarray:
    """
    Per-frame boxes from detections on sampled frames: interpolate between two
    neighbouring detections (only when both found a face), smooth within each
    run of face frames, then pad (chin) and clamp to the frame.
    """
    boxes = np.full((n, 4), np.nan)
    keys = sorted(sampled)
    for a, b in zip(keys, keys[1:] + [None]):
        box_a = sampled[a]
        if box_a is None:
            continue
        boxes[a] = box_a
        box_b = sampled.get(b) if b is not None else None
        if box_b is not None:
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                boxes[f] = np.array(box_a) * (1 - t) + np.array(box_b) * t

    # Smooth within each continuous run of face frames
    has = ~np.isnan(boxes[:, 0])
    smoothed = boxes.copy()
    half = SMOOTH_FRAMES // 2
    for f in np.flatnonzero(has):
        lo, hi = f, f
        while lo > 0 and f - lo < half and has[lo - 1]:
            lo -= 1
        while hi < n - 1 and hi - f < half and has[hi + 1]:
            hi += 1
        smoothed[f] = boxes[lo:hi + 1].mean(axis=0)

    # Padding (chin), clamped to the frame
    top, bottom, left, right = PADS
    smoothed[:, 0] = np.clip(smoothed[:, 0] - left, 0, width)
    smoothed[:, 1] = np.clip(smoothed[:, 1] - top, 0, height)
    smoothed[:, 2] = np.clip(smoothed[:, 2] + right, 0, width)
    smoothed[:, 3] = np.clip(smoothed[:, 3] + bottom, 0, height)
    return smoothed


# ── Mouth generation ─────────────────────────────────────────────────────────

def mouth_mask() -> np.ndarray:
    """
    Feathered blend mask in Wav2Lip's 96×96 face space: 1 over the mouth and
    jaw (Wav2Lip only regenerates the lower half), fading out at the top and
    sides so no box edge is visible.
    """
    m = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    m[int(IMG_SIZE * 0.52):int(IMG_SIZE * 0.97), int(IMG_SIZE * 0.12):int(IMG_SIZE * 0.88)] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=IMG_SIZE * 0.06)
    return m


def load_model(wav2lip_dir: str, checkpoint: str, device: str):
    import torch
    from models import Wav2Lip

    state = torch.load(checkpoint, map_location=device)["state_dict"]
    model = Wav2Lip()
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval()


def mel_chunks(audio_path: str, fps: float, n_frames: int) -> np.ndarray:
    """One (80, MEL_STEP) mel window per video frame, as Wav2Lip's inference.py."""
    import audio
    mel = audio.melspectrogram(audio.load_wav(audio_path, 16000))
    mel = np.nan_to_num(mel)
    chunks = []
    for i in range(n_frames):
        start = int(i * 80.0 / fps)
        if start + MEL_STEP > mel.shape[1]:
            start = max(0, mel.shape[1] - MEL_STEP)
        chunks.append(mel[:, start:start + MEL_STEP])
    return np.stack(chunks)


def run(request: dict) -> dict:
    import torch

    t0 = time.time()
    video, audio_path, output = request["video"], request["audio"], request["output"]
    wav2lip_dir = request["wav2lip_dir"]
    sys.path.insert(0, wav2lip_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fps, width, height = probe(video)

    # Face boxes: cached per video, shared by every language of the job
    cache = request.get("box_cache")
    key = f"{os.path.getsize(video)}:{os.path.getmtime(video)}:{fps}:{width}x{height}"
    boxes = None
    if cache and os.path.isfile(cache):
        data = np.load(cache, allow_pickle=False)
        if str(data["key"]) == key:
            boxes = data["boxes"]
    if boxes is None:
        boxes = detect_boxes(video, fps, width, height, device)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, key=key, boxes=boxes)
    n_frames = len(boxes)
    has_face = ~np.isnan(boxes[:, 0])

    mels = mel_chunks(audio_path, fps, n_frames)
    model = load_model(wav2lip_dir, request["checkpoint"], device)
    mask96 = mouth_mask()

    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
         "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-t", f"{n_frames / fps:.4f}", output],
        stdin=subprocess.PIPE,
    )

    def flush(pending: list[tuple[int, np.ndarray]]) -> None:
        faces = [(i, f) for i, f in pending if has_face[i]]
        if faces:
            crops, masked = [], []
            for i, f in faces:
                x1, y1, x2, y2 = boxes[i].astype(int)
                crop = cv2.resize(f[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
                crops.append(crop)
            crops = np.asarray(crops)
            masked = crops.copy()
            masked[:, IMG_SIZE // 2:] = 0
            img = np.concatenate((masked, crops), axis=3) / 255.0
            mel = mels[[i for i, _ in faces]][:, None]
            with torch.no_grad():
                pred = model(
                    torch.from_numpy(mel).float().to(device),
                    torch.from_numpy(img.transpose(0, 3, 1, 2)).float().to(device),
                )
            pred = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).clip(0, 255)
            for (i, f), p in zip(faces, pred):
                x1, y1, x2, y2 = boxes[i].astype(int)
                w, h = x2 - x1, y2 - y1
                mouth = cv2.resize(p, (w, h)).astype(np.float32)
                mk = cv2.resize(mask96, (w, h))[..., None]
                region = f[y1:y2, x1:x2].astype(np.float32)
                f[y1:y2, x1:x2] = (region * (1 - mk) + mouth * mk).astype(np.uint8)
        for _, f in pending:
            writer.stdin.write(f.tobytes())
        pending.clear()

    pending: list[tuple[int, np.ndarray]] = []
    n_written = 0
    for i, frame in enumerate(read_frames(video, fps, width, height)):
        if i >= n_frames:
            break
        pending.append((i, frame.copy() if has_face[i] else frame))
        n_written += 1
        if len(pending) >= BATCH:
            flush(pending)
    flush(pending)
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the lip-synced video")

    return {"frames": n_written, "face_frames": int(has_face.sum()), "fps": fps,
            "seconds": round(time.time() - t0, 1), "device": device}


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as _f:
        _req = json.load(_f)
    try:
        _result = run(_req)
    except Exception as _exc:
        _result = {"error": f"{type(_exc).__name__}: {_exc}"}
    with open(_req["result"], "w", encoding="utf-8") as _f:
        json.dump(_result, _f)
