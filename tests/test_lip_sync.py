"""
Tests for the lip-sync worker (src/lipsync_worker.py) and LipSyncProcessor.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from src.lip_sync import LipSyncProcessor, is_wav2lip_available
from src.lipsync_worker import IMG_SIZE, PADS, boxes_from_samples, mouth_mask

needs_wav2lip = pytest.mark.skipif(
    not is_wav2lip_available() or shutil.which("ffmpeg") is None,
    reason="Wav2Lip checkpoints or ffmpeg not installed",
)


class TestFaceBoxes:

    def test_interpolates_only_between_two_detections(self):
        sampled = {0: (10, 10, 50, 50), 3: (13, 10, 53, 50), 6: None}
        boxes = boxes_from_samples(sampled, 8, 640, 360)
        has = ~np.isnan(boxes[:, 0])
        assert has.tolist() == [True, True, True, True, False, False, False, False]

    def test_moves_smoothly_and_pads_the_chin(self):
        sampled = {0: (10, 10, 50, 50), 3: (13, 10, 53, 50)}
        boxes = boxes_from_samples(sampled, 4, 640, 360)
        assert np.all(np.diff(boxes[:, 0]) >= 0)              # x1 moves monotonically
        assert boxes[0, 3] == pytest.approx(50 + PADS[1])     # bottom padded for the chin

    def test_clamped_to_frame(self):
        boxes = boxes_from_samples({0: (-5, 0, 100, 355)}, 1, 100, 360)
        x1, y1, x2, y2 = boxes[0]
        assert x1 >= 0 and x2 <= 100 and y2 <= 360


class TestMouthMask:

    def test_only_lower_face_is_replaced_with_soft_edges(self):
        m = mouth_mask()
        assert m.shape == (IMG_SIZE, IMG_SIZE)
        assert m[: IMG_SIZE // 3].max() < 0.01            # eyes / forehead untouched
        assert m[int(IMG_SIZE * 0.75), IMG_SIZE // 2] > 0.95   # mouth fully replaced
        assert 0.0 < m[int(IMG_SIZE * 0.75), int(IMG_SIZE * 0.1)] < 0.9   # feathered sides


@needs_wav2lip
class TestWorkerEndToEnd:

    def test_frame_exact_output_with_dubbed_audio(self, tmp_output_dir):
        video = os.path.join(tmp_output_dir, "in.mp4")
        audio = os.path.join(tmp_output_dir, "dub.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2",
                        "-pix_fmt", "yuv420p", video], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=2", audio], check=True)

        out = os.path.join(tmp_output_dir, "out.mp4")
        result = LipSyncProcessor(face_cache_dir=tmp_output_dir).sync(video, audio, out, "hi")

        assert result.success and not result.fallback_used, result.error_message
        assert result.total_frames == 50 and result.face_frames == 0   # no face in testsrc
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
             "stream=codec_type,nb_read_frames", "-of", "json", out],
            capture_output=True, text=True, check=True).stdout)
        streams = {s["codec_type"]: s for s in probe["streams"]}
        assert int(streams["video"]["nb_read_frames"]) == 50
        assert "audio" in streams
        assert os.path.isfile(os.path.join(tmp_output_dir, "face_boxes.npz"))   # reused per language
